# SPDX-License-Identifier: Apache-2.0
"""Device side of the pareto-comm protocol: @gluon.jit helpers inlined into
the kernels (spins, the counter advance, the publish fence, failure reporting,
opaque scalars) and the host-side sentinel fill."""

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from .protocol import FAILED, LIMIT, SEQ_MISMATCH

LIM = gl.constexpr(LIMIT)
FAILED_C = gl.constexpr(FAILED)
SEQ_C = gl.constexpr(SEQ_MISMATCH)
# two bf16 quiet NaNs: what a failed call writes instead of its result
_NAN2 = gl.constexpr(0x7FC07FC0)


# ------------------------------------------------------------------- spins
@gluon.jit
def spin(ps, NOUT: gl.constexpr, ASM: gl.constexpr, CONS: gl.constexpr):
    """One asmgen.spin block over the pointers ps (asmgen.slots: the outputs)."""
    return gl.inline_asm_elementwise(
        ASM, CONS, ps, dtype=(gl.int32,) * NOUT, is_pure=False, pack=1
    )


@gluon.jit
def spin_flag(p, v, ASM: gl.constexpr, CONS: gl.constexpr):
    """Per-wave spin on one dword per lane (asmgen flag_eq / flag_ge) ->
    (value, counter)."""
    return gl.inline_asm_elementwise(
        ASM, CONS, [p, v], dtype=(gl.int32, gl.int32), is_pure=False, pack=1
    )


@gluon.jit
def trap_if_failed(it):
    """s_trap 2 in a wave whose spin failed (counter >= FAILED: timeout, abort or
    tag mismatch), after its stores (the diagnostics record) drained."""
    return gl.inline_asm_elementwise(
        "v_cmp_gt_u32 vcc, 0x7ffffffd, $1\ns_cbranch_vccnz 1f\ns_waitcnt vmcnt(0)\ns_trap 2\n1:\ns_mov_b32 $0, 0",
        "=s,v,~{vcc},~{memory}",
        [it],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def diag_report(
    DIAG_BASE: gl.constexpr,
    it,
    lane,
    INST: gl.constexpr,
    KIND: gl.constexpr,
    ME: gl.constexpr,
    role,
    pos,
    cnt,
    TRAP: gl.constexpr,
):
    """A failed spin (it >= FAILED) writes a 16-int32 record into the
    diagnostics ring (magic last), then traps when TRAP."""
    bad = it >= FAILED_C
    z = lane * 0
    slot = (INST * 131 + role * 31 + pos) % 256
    p = cptr(DIAG_BASE, gl.int32) + slot * 16 + z
    gl.store(p + 1, it, mask=bad)
    gl.store(p + 2, z + ME, mask=bad)
    gl.store(p + 3, z + INST, mask=bad)
    gl.store(p + 4, z + KIND, mask=bad)
    gl.store(p + 5, z + role, mask=bad)
    gl.store(p + 6, z + pos, mask=bad)
    gl.store(p + 7, z + cnt, mask=bad)
    # DIAG_MAGIC last: a record is complete once its magic is there
    gl.store(p, z + 0x50415231, mask=bad)
    if TRAP:
        trap_if_failed(it)


@gluon.jit
def seq_tag(m, site):
    """Call tag of (m, site) in 0 .. 0x7FFF: equal on every rank of a matched
    call, different (but for 2^-15) otherwise."""
    h = (m * -1640531535) ^ (site * 668265261)
    h = h ^ (h >> 15)
    return h & 32767


@gluon.jit
def nan_if_failed(v, it):
    """A failed call's result: two bf16 NaNs per int32 word."""
    return gl.where(it >= FAILED_C, gl.full(v.shape, _NAN2, v.dtype, v.type.layout), v)


@gluon.jit
def store_nan_if_failed(p, it, mask):
    """A failed call overwrites its result words at p (int32, it's shape) with NaN
    pairs; called after the normal stores (same lanes, same addresses)."""
    gl.store(
        p,
        gl.full(it.shape, _NAN2, gl.int32, it.type.layout),
        mask=mask & (it >= FAILED_C),
    )


# ---------------------------------------------------------------- counters
@gluon.jit
def cta_barrier(x):
    return gl.inline_asm_elementwise(
        "s_barrier\ns_mov_b32 $0, $1",
        "=s,s,~{memory}",
        [x],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def advance(ctr, cnt, NSLOT: gl.constexpr):
    """Store the next rotation counter after every wave has loaded it
    (s_barrier); it wraps at the largest multiple of NSLOT below 2^31."""
    CWRAP: gl.constexpr = (2147483647 // NSLOT) * NSLOT
    c = cta_barrier(cnt)
    gl.store(ctr, gl.where(c + 1 >= CWRAP, 0, c + 1))


# ----------------------------------------------------------------- publish
@gluon.jit
def publish_fence(x):
    """CTA publish fence: every wave drains its memory ops, then s_barrier;
    returns an opaque zero that is added into the flag offset so the flag store
    cannot be scheduled above the fence."""
    return gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_barrier\ns_mov_b32 $0, 0",
        "=s,s,~{memory}",
        [x],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def vmdrain(x):
    """Drain this wave's outstanding memory ops (no CTA barrier); returns an
    opaque zero."""
    return gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)\ns_mov_b32 $0, 0",
        "=s,s,~{memory}",
        [x],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


# -------------------------------------------------------------------- misc
@gluon.jit
def now(x):
    """The scalar real-time clock (100 MHz)."""
    return gl.inline_asm_elementwise(
        "s_memrealtime $0\ns_waitcnt lgkmcnt(0)",
        "=s,s,~{memory}",
        [x],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def canon_bf16x2(v):
    """Two packed bf16: a -0.0 half (0x8000, the sentinel) becomes +0.0."""
    lo = gl.where((v & 65535) == 32768, v & -65536, v)
    return gl.where((lo & -65536) == -2147483648, lo & 65535, lo)


@gluon.jit
def zero(x):
    """An opaque scalar zero (adding it to an offset keeps the compiler from
    hoisting / merging the access)."""
    return gl.inline_asm_elementwise(
        "s_mov_b32 $0, 0", "=s,s,~{memory}", [x], dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit
def cptr(A: gl.constexpr, DT: gl.constexpr):
    """A per-process constant address compiled in as an immediate."""
    return gl.cast(A, gl.int64).to(gl.pointer_type(DT))


# -------------------------------------------------------------------- host
@gluon.jit
def fill_kernel(p_ptr, n, val, BLOCK: gl.constexpr):
    """n int32 words at p_ptr := val (grid cdiv(n, BLOCK))."""
    L: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=L)
    gl.store(p_ptr + offs, gl.full([BLOCK], val, gl.int32, L), mask=offs < n)


class I32Ptr:
    """Minimal object Triton accepts as a pointer argument (data_ptr + dtype)."""

    def __init__(self, addr):
        self._a = int(addr)
        self.dtype = torch.int32

    def data_ptr(self):
        return self._a


def cdiv(a, b):
    return (a + b - 1) // b


def fill(addr, nwords, val, block=1024):
    """Launch fill_kernel over nwords int32 words at device address addr."""
    fill_kernel[(cdiv(int(nwords), block),)](
        I32Ptr(addr), int(nwords), int(val), BLOCK=block, num_warps=4
    )
