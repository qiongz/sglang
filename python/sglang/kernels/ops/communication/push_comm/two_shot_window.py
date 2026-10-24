# SPDX-License-Identifier: Apache-2.0
"""two_shot_window: two-stage push all-reduce over a fixed window of rows on the
c12 wire, with the data as its own arrival signal (no flags, no remote fence).

No transmitted halfword is ever 0x8000 (the sentinel): such units escape. RS
slots alternate two parities per window slot; a reducer re-arms the other
parity before its poll, and a sender writes generation gen only after this
rank's finishers released gen - 1."""

from __future__ import annotations

import itertools

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import buffer_store

from . import asmgen
from .codec import _enc1, _ex, _pk, _popc
from .launch import FastLauncher, PushInst
from .primitives import FAILED_C, SEQ_C, I32Ptr
from .primitives import canon_bf16x2 as _canon2
from .primitives import cdiv, diag_report, fill_kernel
from .primitives import publish_fence as _fence
from .primitives import seq_tag
from .primitives import spin as _spin
from .protocol import FAILED, LIMIT, SpinBound, no_trap


@gluon.jit
def _cptr32(A: gl.constexpr):
    return gl.cast(A, gl.int64).to(gl.pointer_type(gl.int32))


@gluon.jit
def _sel(pd, j, N: gl.constexpr):
    """pd[j] for a runtime (wave-uniform) j: a select chain over the N
    immediates."""
    d = pd[0]
    for jj in gl.static_range(1, N):
        d = gl.where(j == jj, pd[jj], d)
    return d


R = 4  # rows per position
SEGW = 520  # dwords per segment: 3 planes x 128 + escape plane 128 + header 8
SENT = -2147450880  # 0x80008000: two bf16 -0.0
SENT64 = -9223231297218904064  # 0x8000800080008000
_LIM = gl.constexpr(LIMIT)
_R = gl.constexpr(R)
_SEGW = gl.constexpr(SEGW)
_SENT = gl.constexpr(SENT)
_SENT64 = gl.constexpr(SENT64)
# four bf16 NaN: what a failed finisher wave stores at the masked tail
_NAN64 = gl.constexpr(0x7FC07FC07FC07FC0)
# kernel kind in the diagnostics records (push_comm_monitor.KIND_NAMES)
KIND = 4
_KIND = gl.constexpr(KIND)

# _nan_seg's asm: a wave with a failed lane ($1) overwrites the 32 B at $2 of
# every valid-row lane ($3 < $4) with NaN, after the normal stores.
_NAN_ASM = gl.constexpr(
    "\n".join(
        [
            f"v_cmp_le_u32 vcc, {FAILED:#x}, $1",
            "s_cbranch_vccz 5f",
            "v_cmp_gt_i32 vcc, $4, $3",
            "s_and_saveexec_b64 s[40:41], vcc",
            "s_cbranch_execz 4f",
            "v_mov_b32 $0, 0x7fc07fc0",
        ]
        + [f"global_store_dword $2, $0, off offset:{4 * i}" for i in range(8)]
        + ["4:", "s_mov_b64 exec, s[40:41]", "5:"]
    )
)
_NAN_CONS = gl.constexpr("=&v,v,v,v,s,~{vcc},~{memory},~{s40},~{s41}")


@gluon.jit
def _has8k(w):
    return ((w & 65535) == 32768) | (((w >> 16) & 65535) == 32768)


@gluon.jit
def _lo32(q):
    return q.to(gl.int32)


@gluon.jit
def _hi32(q):
    return (q >> 32).to(gl.int32)


@gluon.jit
def _q64(a, b):
    return (a.to(gl.int64) & 4294967295) | (b.to(gl.int64) << 32)


@gluon.jit
def _umax(p0, p1, p2, p3):
    """Largest biased exponent of the 8 bf16 of a unit (4 packed pairs)."""
    e0 = gl.maximum(_ex(p0 & 65535), _ex((p0 >> 16) & 65535))
    e1 = gl.maximum(_ex(p1 & 65535), _ex((p1 >> 16) & 65535))
    e2 = gl.maximum(_ex(p2 & 65535), _ex((p2 >> 16) & 65535))
    e3 = gl.maximum(_ex(p3 & 65535), _ex((p3 >> 16) & 65535))
    return gl.maximum(gl.maximum(e0, e1), gl.maximum(e2, e3))


@gluon.jit
def _codes(p0, p1, p2, p3, B7):
    """One unit -> (w0, w1, w2, escape)."""
    c0, k0 = _enc1(p0 & 65535, B7)
    c1, k1 = _enc1((p0 >> 16) & 65535, B7)
    c2, k2 = _enc1(p1 & 65535, B7)
    c3, k3 = _enc1((p1 >> 16) & 65535, B7)
    c4, k4 = _enc1(p2 & 65535, B7)
    c5, k5 = _enc1((p2 >> 16) & 65535, B7)
    c6, k6 = _enc1(p3 & 65535, B7)
    c7, k7 = _enc1((p3 >> 16) & 65535, B7)
    w0 = c0 | (c1 << 12) | (c2 << 24)
    w1 = (c2 >> 8) | (c3 << 4) | (c4 << 16) | (c5 << 28)
    w2 = (c5 >> 4) | (c6 << 8) | (c7 << 20)
    esc = (k0 & k1 & k2 & k3 & k4 & k5 & k6 & k7) == 0
    # data as flag: no 0x8000 halfword on the wire
    esc = esc | _has8k(w0) | _has8k(w1) | _has8k(w2)
    return w0, w1, w2, esc


@gluon.jit
def _encode2(pe, po, g):
    """Even / odd units of the lane (pair tuples of 4 canonical bf16-pair tiles
    [16, 16]) -> code dwords, escape flags, compacted escape slots and the row
    header (base, mask) as [16, 1]."""
    emax = gl.max(
        gl.maximum(
            _umax(pe[0], pe[1], pe[2], pe[3]), _umax(po[0], po[1], po[2], po[3])
        ),
        axis=1,
    )
    base = gl.expand_dims(gl.maximum(emax - 15, 0), 1)
    B7 = base << 7
    a0, a1, a2, ea = _codes(pe[0], pe[1], pe[2], pe[3], B7)
    b0, b1, b2, eb = _codes(po[0], po[1], po[2], po[3], B7)
    u_e = 2 * g
    u_o = 2 * g + 1
    msk = gl.expand_dims(
        gl.sum(gl.where(ea, 1 << u_e, 0) | gl.where(eb, 1 << u_o, 0), axis=1), 1
    )
    # data as flag: neither half of the mask dword may be 0x8000 -> escape
    # unit 0 / 16 as well
    fl = (msk & 65535) == 32768
    fh = ((msk >> 16) & 65535) == 32768
    ea = ea | (fl & (u_e == 0)) | (fh & (u_e == 16))
    msk = msk | gl.where(fl, 1, 0) | gl.where(fh, 65536, 0)
    w0e = gl.where(ea, pe[0], a0)
    w1e = gl.where(ea, pe[1], a1)
    w2e = gl.where(ea, pe[2], a2)
    w0o = gl.where(eb, po[0], b0)
    w1o = gl.where(eb, po[1], b1)
    w2o = gl.where(eb, po[2], b2)
    se = _popc(msk & ((1 << u_e) - 1))
    so = _popc(msk & ((1 << u_o) - 1))
    return (w0e, w1e, w2e, pe[3], ea, se), (w0o, w1o, w2o, po[3], eb, so), base, msk


_DEC8_ASM = gl.constexpr(
    """
v_lshlrev_b32 $4, 4, $6
v_bfi_b32 $0, $10, $6, $4
v_alignbit_b32 $4, $7, $6, 24
v_lshlrev_b32 $5, 12, $7
v_bfi_b32 $1, $10, $4, $5
v_alignbit_b32 $4, $8, $7, 28
v_lshrrev_b32 $5, 16, $7
v_lshl_or_b32 $2, $4, 16, $5
v_lshrrev_b32 $4, 8, $8
v_lshrrev_b32 $5, 4, $8
v_bfi_b32 $3, $10, $4, $5
"""
    + "".join(
        f"""v_and_b32 $4, 0x08000800, ${k}
v_lshlrev_b32 $4, 4, $4
v_and_b32 $5, 0x07ff07ff, ${k}
v_pk_min_u16 ${k}, $5, $11
v_pk_mul_lo_u16 ${k}, ${k}, $9
v_pk_add_u16 ${k}, $5, ${k}
v_or_b32 ${k}, ${k}, $4
"""
        for k in range(4)
    )
)
_DEC8_CONS = gl.constexpr("=&v,=&v,=&v,=&v,=&v,=&v,v,v,v,v,v,v")


@gluon.jit
def _decode1p(w0, w1, w2, w3, esc, B7):
    """The c12 decode of one unit, two codes at a time in packed 16-bit math."""
    bb = B7 | (B7 << 16)
    cf = gl.full(w0.shape, 65535, gl.int32, w0.type.layout)
    c1 = gl.full(w0.shape, 65537, gl.int32, w0.type.layout)
    d = gl.inline_asm_elementwise(
        _DEC8_ASM,
        _DEC8_CONS,
        [w0, w1, w2, bb + w0 * 0, cf, c1],
        dtype=(gl.int32,) * 6,
        is_pure=True,
        pack=1,
    )
    return (
        gl.where(esc, w0, d[0]),
        gl.where(esc, w1, d[1]),
        gl.where(esc, w2, d[2]),
        gl.where(esc, w3, d[3]),
    )


@gluon.jit
def _nan_seg(fail, o64, row, m):
    """A finisher wave whose wait failed (or that read a foreign call tag)
    overwrites its slice of the output (lanes of valid rows, 32 B each at o64)
    with bf16 NaN: per wave, in asm, after the normal stores."""
    return gl.inline_asm_elementwise(
        _NAN_ASM,
        _NAN_CONS,
        [fail, o64.to(gl.int64, bitcast=True), row, m],
        dtype=gl.int32,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _put(bp, soff, ce, co, base, msk, r, g, ok):
    """Store one encoded segment slice (lane: row r, units 2g, 2g + 1) at
    dword offset soff from the scalar base bp."""
    b64 = bp.to(gl.pointer_type(gl.int64))
    o = soff // 2 + r * 16 + g
    buffer_store(_q64(ce[0], co[0]), b64, o, mask=ok)
    buffer_store(_q64(ce[1], co[1]), b64, o + 64, mask=ok)
    buffer_store(_q64(ce[2], co[2]), b64, o + 128, mask=ok)
    buffer_store(
        _q64(base + g * 0, msk + g * 0),
        b64,
        soff // 2 + 256 + r + g * 0,
        mask=ok & (g == 0),
    )
    buffer_store(ce[3], bp, soff + 384 + r * 32 + ce[5], mask=ok & ce[4])
    buffer_store(co[3], bp, soff + 384 + r * 32 + co[5], mask=ok & co[4])


@gluon.jit
def _rearm(bp, soff, r, g, ok, HDR: gl.constexpr):
    """Re-arm one consumed segment slice (planes, the lane's escape dwords and,
    with HDR, the row header)."""
    b64 = bp.to(gl.pointer_type(gl.int64))
    o = soff // 2 + r * 16 + g
    s64 = gl.full(o.shape, _SENT64, gl.int64, o.type.layout)
    buffer_store(s64, b64, o, mask=ok)
    buffer_store(s64, b64, o + 64, mask=ok)
    buffer_store(s64, b64, o + 128, mask=ok)
    buffer_store(s64, b64, o + 192, mask=ok)
    if HDR:
        buffer_store(s64, b64, soff // 2 + 256 + r + g * 0, mask=ok & (g == 0))


@gluon.jit
def _esc_ptrs(segp, r, g, msk):
    """The escaped units' compacted plane-3 slots of one segment slice; a unit
    that did not escape points at the row header's base dword instead (already
    arrived, never a 0x8000 halfword), so one poll serves every lane."""
    u_e = 2 * g
    u_o = 2 * g + 1
    hb = segp + 512 + r * 2 + g * 0
    pe = gl.where(
        ((msk >> u_e) & 1) != 0, segp + 384 + r * 32 + _popc(msk & ((1 << u_e) - 1)), hb
    )
    po = gl.where(
        ((msk >> u_o) & 1) != 0, segp + 384 + r * 32 + _popc(msk & ((1 << u_o) - 1)), hb
    )
    return pe, po


@gluon.jit
def _encode1(p, u):
    """One unit per lane ([rows, 32] tile, lane <-> unit u of its row): code
    dwords, escape flag / compacted slot and the row header, with the
    data-as-flag fix-ups of _encode2."""
    emax = gl.max(_umax(p[0], p[1], p[2], p[3]), axis=1)
    base = gl.expand_dims(gl.maximum(emax - 15, 0), 1)
    a0, a1, a2, ea = _codes(p[0], p[1], p[2], p[3], base << 7)
    msk = gl.expand_dims(gl.sum(gl.where(ea, 1 << u, 0), axis=1), 1)
    fl = (msk & 65535) == 32768
    fh = ((msk >> 16) & 65535) == 32768
    ea = ea | (fl & (u == 0)) | (fh & (u == 16))
    msk = msk | gl.where(fl, 1, 0) | gl.where(fh, 65536, 0)
    w = (
        gl.where(ea, p[0], a0),
        gl.where(ea, p[1], a1),
        gl.where(ea, p[2], a2),
        p[3],
        ea,
        _popc(msk & ((1 << u) - 1)),
    )
    return w, base, msk


@gluon.jit
def _reduce1u(
    lp,
    pd,
    b,
    kc,
    pos,
    m,
    err,
    ME: gl.constexpr,
    W: gl.constexpr,
    NK: gl.constexpr,
    AG_OFF: gl.constexpr,
    R1_ASM: gl.constexpr,
    R1_CONS: gl.constexpr,
    R1_NOUT: gl.constexpr,
    E1_ASM: gl.constexpr,
    E1_CONS: gl.constexpr,
    E1_NOUT: gl.constexpr,
    role,
    gen,
    REL_OFF: gl.constexpr,
    NF: gl.constexpr,
    GE_ASM: gl.constexpr,
    GE_CONS: gl.constexpr,
    cnt=0,
    tg=0,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TAG: gl.constexpr = 0,
):
    """REDUCE, one unit per lane (the reducer is on every position's critical
    path). DIAG: a failed wave pushes nothing. TAG: a foreign tag is forwarded
    poisoned (tg ^ 0x4000)."""
    LR: gl.constexpr = gl.BlockedLayout([1, 1], [2, 32], [4, 1], [1, 0])
    rw = gl.expand_dims(gl.arange(0, 8, layout=gl.SliceLayout(1, LR)), 1)
    u = gl.expand_dims(gl.arange(0, 32, layout=gl.SliceLayout(0, LR)), 0)
    k = kc * 2 + rw // 4
    r = rw % 4
    row = pos * _R + r
    rok = row < m
    rq = gl.where(rok, r, 0)
    errt = err + rw * 0 + u * 0
    # generation gen reads RS parity gen % 2 and re-arms parity gen - 1 before
    # its poll (the poll's vmcnt(0) drains it); safe once this rank's
    # finishers released gen - 1, i.e. every reducer finished reading it
    bq = b * 2 + (gen & 1)
    bo = b * 2 + ((gen + 1) & 1)
    fl = lp + REL_OFF + b * NF + (u + rw * 32) % NF
    rl = gl.inline_asm_elementwise(
        GE_ASM,
        GE_CONS,
        [fl, gl.zeros([8, 32], gl.int32, LR) + gen],
        dtype=(gl.int32, gl.int32),
        is_pure=False,
        pack=1,
    )
    gl.store(errt, rl[1], mask=(rl[1] >= _LIM) & (u == 0) & (rw == 0))
    dw = rl[0] * 0
    # all 4 rows: the previous generation may have had more valid rows than
    # this one (its m is not ours)
    s32 = gl.full([8, 32], _SENT, gl.int32, LR)
    for src in gl.static_range(W):
        sp = lp + (bo * W + src) * NK * _SEGW
        o = k * _SEGW + r * 32 + u + dw
        buffer_store(s32, sp, o)
        buffer_store(s32, sp, o + 128)
        buffer_store(s32, sp, o + 256)
        buffer_store(s32, sp, o + 384)
    hoff = (((bo * W + u % W) * NK + k) * _SEGW + 512 + r * 2) // 2 + dw
    buffer_store(
        gl.full([8, 32], _SENT64, gl.int64, LR),
        lp.to(gl.pointer_type(gl.int64)),
        hoff,
        mask=u < W,
    )
    prs = ()
    for src in gl.static_range(W):
        prs += (lp + ((bq * W + src) * NK + k) * _SEGW + rq * 32 + u,)
    # every lane polls each source's row header itself
    for src in gl.static_range(W):
        prs += (lp + ((bq * W + src) * NK + k) * _SEGW + 512 + rq * 2 + u * 0,)
    rp = _spin(prs, R1_NOUT, R1_ASM, R1_CONS)
    itv = rp[5 * W]
    gl.store(errt, itv, mask=(itv >= _LIM) & (u == 0))
    bss = ()
    mks = ()
    eps = ()
    any_e = gl.zeros([8, 32], gl.int32, LR)
    tbad = rw * 0 + u * 0
    for src in gl.static_range(W):
        sb = rp[3 * W + 2 * src]
        sm = rp[3 * W + 2 * src + 1]
        if TAG:  # base word: base | tag << 16 (every row valid: rq)
            tbad = tbad | ((sb >> 16) != tg).to(gl.int32)
            sb = sb & 255
        bss += (sb,)
        mks += (sm,)
        segp = lp + ((bq * W + src) * NK + k) * _SEGW
        e = ((sm >> u) & 1) != 0
        eps += (
            gl.where(
                e,
                segp + 384 + rq * 32 + _popc(sm & ((1 << u) - 1)),
                segp + 512 + rq * 2 + u * 0,
            ),
        )
        any_e = any_e | e.to(gl.int32)
    ev = _spin(eps + (any_e,), E1_NOUT, E1_ASM, E1_CONS)
    gl.store(errt, ev[W], mask=(ev[W] >= _LIM) & (u == 0))
    okp = rok
    if DIAG:
        fail = gl.maximum(gl.maximum(rl[1], itv), ev[W])  # wave-uniform
        if TAG:
            st = gl.where(tbad != 0, SEQ_C, 0)
            gl.store(errt, st, mask=(st == SEQ_C) & (u == 0))
            diag_report(
                DIAG_BASE,
                gl.maximum(fail, st),
                u + rw * 0,
                INST,
                _KIND,
                ME,
                role,
                pos,
                cnt,
                TRAP,
            )
        else:
            diag_report(
                DIAG_BASE, fail, u + rw * 0, INST, _KIND, ME, role, pos, cnt, TRAP
            )
        okp = rok & (fail < FAILED_C)  # a failed wave pushes nothing
    a = ()
    for i in gl.static_range(8):
        a += (gl.zeros([8, 32], gl.float32, LR),)
    for src in gl.static_range(W):
        sm = mks[src]
        d = _decode1p(
            rp[3 * src],
            rp[3 * src + 1],
            rp[3 * src + 2],
            ev[src],
            ((sm >> u) & 1) != 0,
            bss[src] << 7,
        )
        na = ()
        for t in gl.static_range(4):  # fixed rank order 0..W-1 (fp32)
            na += (
                a[2 * t] + (d[t] << 16).to(gl.float32, bitcast=True),
                a[2 * t + 1] + (d[t] & -65536).to(gl.float32, bitcast=True),
            )
        a = na
    pk = ()
    for t in gl.static_range(4):
        pk += (_canon2(_pk(a[2 * t], a[2 * t + 1])),)
    w, base, msk = _encode1(pk, u)
    dz = 0
    o = k * _SEGW + r * 32 + u
    ho = (k * _SEGW + 512 + r * 2) // 2 + u * 0
    if TAG:  # a row that read a foreign tag forwards a poisoned one
        hq = _q64(base | (gl.where(tbad != 0, tg ^ 16384, tg) << 16), msk + u * 0)
    else:
        hq = _q64(base + u * 0, msk + u * 0)
    for jj in gl.static_range(W):
        dst = pd[(ME + jj) % W] + AG_OFF + dz + (b * W + ME) * NK * _SEGW
        buffer_store(w[0], dst, o, mask=okp)
        buffer_store(w[1], dst, o + 128, mask=okp)
        buffer_store(w[2], dst, o + 256, mask=okp)
        buffer_store(hq, dst.to(gl.pointer_type(gl.int64)), ho, mask=okp & (u == 0))
        buffer_store(w[3], dst, k * _SEGW + 384 + r * 32 + w[5], mask=okp & w[4])


@gluon.jit(do_not_specialize=["m"])
def two_shot_window_kernel(
    x_ptr,
    out_ptr,
    ctr_ptr,
    err_ptr,
    m,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    P: gl.constexpr,
    AG_OFF: gl.constexpr,
    REL_OFF: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    FP_ASM: gl.constexpr,
    FP_CONS: gl.constexpr,
    FP_NOUT: gl.constexpr,
    GE_ASM: gl.constexpr,
    GE_CONS: gl.constexpr,
    FE_ASM: gl.constexpr,
    FE_CONS: gl.constexpr,
    FE_NOUT: gl.constexpr,
    R1_ASM: gl.constexpr,
    R1_CONS: gl.constexpr,
    R1_NOUT: gl.constexpr,
    E1_ASM: gl.constexpr,
    E1_CONS: gl.constexpr,
    E1_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
):
    """Grid (NROLE, positions). Failure handling (module docstring): the *_ASM
    spins built with abort= poll the host's abort word; DIAG / DIAG_BASE /
    INST / TRAP = protocol.Diag.kw()."""
    pos = gl.program_id(1)
    role = gl.program_id(0)
    # pos >= 0 lets the body use unsigned pos % P / pos // P
    if (pos >= 0) & (pos * _R < m):
        _window_body(
            x_ptr,
            out_ptr,
            ctr_ptr,
            err_ptr,
            m,
            role,
            pos,
            ME,
            W,
            H,
            P,
            AG_OFF,
            REL_OFF,
            P0,
            P1,
            P2,
            P3,
            P4,
            P5,
            P6,
            P7,
            FP_ASM,
            FP_CONS,
            FP_NOUT,
            GE_ASM,
            GE_CONS,
            FE_ASM,
            FE_CONS,
            FE_NOUT,
            R1_ASM,
            R1_CONS,
            R1_NOUT,
            E1_ASM,
            E1_CONS,
            E1_NOUT,
            0,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            0,
        )


@gluon.jit(do_not_specialize=["m", "site"])
def two_shot_window_tag_kernel(
    x_ptr,
    out_ptr,
    ctr_ptr,
    err_ptr,
    m,
    site,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    P: gl.constexpr,
    AG_OFF: gl.constexpr,
    REL_OFF: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    FP_ASM: gl.constexpr,
    FP_CONS: gl.constexpr,
    FP_NOUT: gl.constexpr,
    GE_ASM: gl.constexpr,
    GE_CONS: gl.constexpr,
    FE_ASM: gl.constexpr,
    FE_CONS: gl.constexpr,
    FE_NOUT: gl.constexpr,
    R1_ASM: gl.constexpr,
    R1_CONS: gl.constexpr,
    R1_NOUT: gl.constexpr,
    E1_ASM: gl.constexpr,
    E1_CONS: gl.constexpr,
    E1_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 1,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
):
    """two_shot_window_kernel with the call-sequence tags (TAG, needs DIAG) and
    the call's runtime site. A separate kernel: the extra kernel argument
    alone would change two_shot_window_kernel's kernel-argument loads."""
    gl.static_assert(DIAG == 1, "two_shot_window: call tags need DIAG")
    pos = gl.program_id(1)
    role = gl.program_id(0)
    if (pos >= 0) & (pos * _R < m):
        _window_body(
            x_ptr,
            out_ptr,
            ctr_ptr,
            err_ptr,
            m,
            role,
            pos,
            ME,
            W,
            H,
            P,
            AG_OFF,
            REL_OFF,
            P0,
            P1,
            P2,
            P3,
            P4,
            P5,
            P6,
            P7,
            FP_ASM,
            FP_CONS,
            FP_NOUT,
            GE_ASM,
            GE_CONS,
            FE_ASM,
            FE_CONS,
            FE_NOUT,
            R1_ASM,
            R1_CONS,
            R1_NOUT,
            E1_ASM,
            E1_CONS,
            E1_NOUT,
            site,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            1,
        )


@gluon.jit
def _tail_ld(x_ptr, e, n, i: gl.constexpr):
    """Masked tail: the 4 bf16 at element e + 4 i, +0 past n (never read)."""
    ei = e + 4 * i
    q = gl.load(
        x_ptr.to(gl.pointer_type(gl.int64)) + ei // 4, mask=ei + 4 <= n, other=0
    )
    x16 = x_ptr.to(gl.pointer_type(gl.int16))
    for t in gl.static_range(4):
        v = gl.load(x16 + ei + t, mask=(ei + t < n) & (ei + 4 > n), other=0)
        q = q | ((v.to(gl.int64) & 65535) << (16 * t))
    return q


@gluon.jit
def _tail_st(out_ptr, e, n, q, i: gl.constexpr):
    """Masked tail: store the int64 q of the finisher's tile at element
    e + 4 i -- whole below n, element-wise (int16) for the int64 that n splits,
    nothing past n."""
    ei = e + 4 * i
    gl.store(out_ptr.to(gl.pointer_type(gl.int64)) + ei // 4, q, mask=ei + 4 <= n)
    o16 = out_ptr.to(gl.pointer_type(gl.int16))
    for t in gl.static_range(4):
        gl.store(
            o16 + ei + t,
            ((q >> (16 * t)) & 65535).to(gl.int16),
            mask=(ei + t < n) & (ei + 4 > n),
        )


@gluon.jit
def _tail_q(q, fail, DIAG: gl.constexpr):
    """The finisher's last-position int64 as stored: four bf16 NaN when the
    wave failed. With DIAG the element-exact tail stores carry the NaN
    (_nan_seg's 32-byte slices could pass element n - 1)."""
    if DIAG:
        return gl.where(
            fail >= FAILED_C, gl.full(q.shape, _NAN64, gl.int64, q.type.layout), q
        )
    else:
        return q


@gluon.jit(do_not_specialize=["m", "n"])
def two_shot_window_tail_kernel(
    x_ptr,
    out_ptr,
    ctr_ptr,
    err_ptr,
    m,
    n,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    P: gl.constexpr,
    AG_OFF: gl.constexpr,
    REL_OFF: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    FP_ASM: gl.constexpr,
    FP_CONS: gl.constexpr,
    FP_NOUT: gl.constexpr,
    GE_ASM: gl.constexpr,
    GE_CONS: gl.constexpr,
    FE_ASM: gl.constexpr,
    FE_CONS: gl.constexpr,
    FE_NOUT: gl.constexpr,
    R1_ASM: gl.constexpr,
    R1_CONS: gl.constexpr,
    R1_NOUT: gl.constexpr,
    E1_ASM: gl.constexpr,
    E1_CONS: gl.constexpr,
    E1_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
):
    """two_shot_window_kernel over n elements as ceil(n / H) rows, the last
    partial (read as +0, never written). A separate kernel so that n does not
    change the whole-row kernel."""
    pos = gl.program_id(1)
    role = gl.program_id(0)
    if (pos >= 0) & (pos * _R < m):
        _window_body(
            x_ptr,
            out_ptr,
            ctr_ptr,
            err_ptr,
            m,
            role,
            pos,
            ME,
            W,
            H,
            P,
            AG_OFF,
            REL_OFF,
            P0,
            P1,
            P2,
            P3,
            P4,
            P5,
            P6,
            P7,
            FP_ASM,
            FP_CONS,
            FP_NOUT,
            GE_ASM,
            GE_CONS,
            FE_ASM,
            FE_CONS,
            FE_NOUT,
            R1_ASM,
            R1_CONS,
            R1_NOUT,
            E1_ASM,
            E1_CONS,
            E1_NOUT,
            0,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            0,
            n,
            1,
        )


@gluon.jit(do_not_specialize=["m", "site", "n"])
def two_shot_window_tail_tag_kernel(
    x_ptr,
    out_ptr,
    ctr_ptr,
    err_ptr,
    m,
    site,
    n,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    P: gl.constexpr,
    AG_OFF: gl.constexpr,
    REL_OFF: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    FP_ASM: gl.constexpr,
    FP_CONS: gl.constexpr,
    FP_NOUT: gl.constexpr,
    GE_ASM: gl.constexpr,
    GE_CONS: gl.constexpr,
    FE_ASM: gl.constexpr,
    FE_CONS: gl.constexpr,
    FE_NOUT: gl.constexpr,
    R1_ASM: gl.constexpr,
    R1_CONS: gl.constexpr,
    R1_NOUT: gl.constexpr,
    E1_ASM: gl.constexpr,
    E1_CONS: gl.constexpr,
    E1_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 1,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
):
    """two_shot_window_tail_kernel with call tags (TAG, needs DIAG) and the
    runtime site."""
    gl.static_assert(DIAG == 1, "two_shot_window: call tags need DIAG")
    pos = gl.program_id(1)
    role = gl.program_id(0)
    if (pos >= 0) & (pos * _R < m):
        _window_body(
            x_ptr,
            out_ptr,
            ctr_ptr,
            err_ptr,
            m,
            role,
            pos,
            ME,
            W,
            H,
            P,
            AG_OFF,
            REL_OFF,
            P0,
            P1,
            P2,
            P3,
            P4,
            P5,
            P6,
            P7,
            FP_ASM,
            FP_CONS,
            FP_NOUT,
            GE_ASM,
            GE_CONS,
            FE_ASM,
            FE_CONS,
            FE_NOUT,
            R1_ASM,
            R1_CONS,
            R1_NOUT,
            E1_ASM,
            E1_CONS,
            E1_NOUT,
            site,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            1,
            n,
            1,
        )


@gluon.jit
def _window_body(
    x_ptr,
    out_ptr,
    ctr_ptr,
    err_ptr,
    m,
    role,
    pos,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    P: gl.constexpr,
    AG_OFF: gl.constexpr,
    REL_OFF: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    FP_ASM: gl.constexpr,
    FP_CONS: gl.constexpr,
    FP_NOUT: gl.constexpr,
    GE_ASM: gl.constexpr,
    GE_CONS: gl.constexpr,
    FE_ASM: gl.constexpr,
    FE_CONS: gl.constexpr,
    FE_NOUT: gl.constexpr,
    R1_ASM: gl.constexpr,
    R1_CONS: gl.constexpr,
    R1_NOUT: gl.constexpr,
    E1_ASM: gl.constexpr,
    E1_CONS: gl.constexpr,
    E1_NOUT: gl.constexpr,
    site,
    DIAG: gl.constexpr,
    DIAG_BASE: gl.constexpr,
    INST: gl.constexpr,
    TRAP: gl.constexpr,
    TAG: gl.constexpr,
    n=0,
    TAIL: gl.constexpr = 0,
):
    OWN: gl.constexpr = H // W
    NK: gl.constexpr = OWN // 256
    NS: gl.constexpr = W * NK // 4
    NR: gl.constexpr = NK // 2
    FS: gl.constexpr = 1  # segments per finisher wave
    NF: gl.constexpr = W * NK // (4 * FS)
    NROLE: gl.constexpr = NS + NR + NF
    b = pos % P
    # the CTA's first memory access, a scalar load
    cnt = gl.load(ctr_ptr + b * NROLE + role)
    gen = cnt + pos // P
    npos = (m + _R - 1) // _R
    last = pos + P >= npos
    tg = 0
    if TAG:
        tg = seq_tag(m, site)

    L2: gl.constexpr = gl.BlockedLayout([1, 1], [4, 16], [4, 1], [1, 0])
    rw = gl.expand_dims(
        gl.arange(0, 16, layout=gl.SliceLayout(1, L2)), 1
    )  # row' = wave * 4 + row
    g = gl.expand_dims(gl.arange(0, 16, layout=gl.SliceLayout(0, L2)), 0)  # unit pair
    wv = rw // 4  # wave = segment of the CTA
    r = rw % 4
    row = pos * _R + r
    rok = row < m
    rq = gl.where(rok, r, 0)  # an invalid row polls row 0's (valid) slots instead
    pd = (
        _cptr32(P0),
        _cptr32(P1),
        _cptr32(P2),
        _cptr32(P3),
        _cptr32(P4),
        _cptr32(P5),
        _cptr32(P6),
        _cptr32(P7),
    )
    lp = pd[ME]
    err = err_ptr + g * 0 + rw * 0

    if role < NS:
        j = role // (NK // 4)
        k = (role % (NK // 4)) * 4 + wv
        x64 = (
            x_ptr.to(gl.pointer_type(gl.int64))
            + gl.where(rok, row, pos * _R) * (H // 4)
            + (j * OWN + k * 256) // 4
            + 4 * g
        )
        if TAIL:
            # the position holding element n - 1 (CTA-uniform)
            if pos == npos - 1:
                te = row * H + j * OWN + k * 256 + 16 * g
                q0 = _tail_ld(x_ptr, te, n, 0)
                q1 = _tail_ld(x_ptr, te, n, 1)
                q2 = _tail_ld(x_ptr, te, n, 2)
                q3 = _tail_ld(x_ptr, te, n, 3)
            else:
                q0 = gl.load(x64)
                q1 = gl.load(x64 + 1)
                q2 = gl.load(x64 + 2)
                q3 = gl.load(x64 + 3)
        else:
            q0 = gl.load(x64)
            q1 = gl.load(x64 + 1)
            q2 = gl.load(x64 + 2)
            q3 = gl.load(x64 + 3)
        pe = (
            _canon2(_lo32(q0)),
            _canon2(_hi32(q0)),
            _canon2(_lo32(q1)),
            _canon2(_hi32(q1)),
        )
        po = (
            _canon2(_lo32(q2)),
            _canon2(_hi32(q2)),
            _canon2(_lo32(q3)),
            _canon2(_hi32(q3)),
        )
        ce, co, base, msk = _encode2(pe, po, g)
        # write-after-read (local release): this rank's finishers released
        # generation gen - 1 of slot b
        fl = lp + REL_OFF + b * NF + (g + rw * 16) % NF
        rl = gl.inline_asm_elementwise(
            GE_ASM,
            GE_CONS,
            [fl, gl.zeros([16, 16], gl.int32, L2) + gen],
            dtype=(gl.int32, gl.int32),
            is_pure=False,
            pack=1,
        )
        gl.store(err, rl[1], mask=(rl[1] >= _LIM) & (g == 0) & (rw == 0))
        okp = rok
        if DIAG:  # a failed wave pushes nothing (the peers' waits fail)
            diag_report(
                DIAG_BASE, rl[1], g + rw * 0, INST, _KIND, ME, role, pos, cnt, TRAP
            )
            okp = rok & (rl[1] < FAILED_C)
        if TAG:
            base = base | (tg << 16)
        dst = _sel(pd, j, W) + ((b * 2 + (gen & 1)) * W + ME) * NK * _SEGW
        _put(dst, k * _SEGW + rw * 0 + g * 0, ce, co, base, msk, r, g, okp)
    elif role < NS + NR:
        _reduce1u(
            lp,
            pd,
            b,
            role - NS,
            pos,
            m,
            err_ptr,
            ME,
            W,
            NK,
            AG_OFF,
            R1_ASM,
            R1_CONS,
            R1_NOUT,
            E1_ASM,
            E1_CONS,
            E1_NOUT,
            role,
            gen,
            REL_OFF,
            NF,
            GE_ASM,
            GE_CONS,
            cnt,
            tg,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            TAG,
        )
    else:
        f = role - NS - NR
        # window wrap: the AG data carries no generation, so wait until this
        # finisher index released gen - 1 of b (immediate unless wrapped)
        frl = gl.inline_asm_elementwise(
            GE_ASM,
            GE_CONS,
            [
                lp + REL_OFF + b * NF + f + g * 0 + rw * 0,
                gl.zeros([16, 16], gl.int32, L2) + gen,
            ],
            dtype=(gl.int32, gl.int32),
            is_pure=False,
            pack=1,
        )
        gl.store(err, frl[1], mask=(frl[1] >= _LIM) & (g == 0) & (rw == 0))
        fdw = frl[0] * 0
        # FS segments per wave (fewer, longer finishers: they spend most of
        # their life waiting in a CU slot)
        fps = ()
        fhs = ()
        for t in gl.static_range(FS):
            f_idx = (f * 4 + wv) * FS + t
            f_segp = lp + AG_OFF + ((b * W + f_idx // NK) * NK + f_idx % NK) * _SEGW
            fps += (f_segp + rq * 32 + 2 * g + fdw,)
            fhs += (f_segp + 512 + rq * 2 + g * 0 + fdw,)
        f_fp = _spin(fps + fhs, FP_NOUT, FP_ASM, FP_CONS)
        f_itv = f_fp[8 * FS]
        gl.store(err, f_itv, mask=(f_itv >= _LIM) & (g == 0))
        feps = ()
        fany = gl.zeros([16, 16], gl.int32, L2)
        for t in gl.static_range(FS):
            f_idx = (f * 4 + wv) * FS + t
            f_segp = lp + AG_OFF + ((b * W + f_idx // NK) * NK + f_idx % NK) * _SEGW
            f_sm = f_fp[6 * FS + 2 * t + 1]
            f_pe, f_po = _esc_ptrs(f_segp, rq, g, f_sm)
            feps += (f_pe, f_po)
            fany = fany | (((f_sm >> (2 * g)) | (f_sm >> (2 * g + 1))) & 1)
        f_ev = _spin(feps + (fany,), FE_NOUT, FE_ASM, FE_CONS)
        gl.store(err, f_ev[2 * FS], mask=(f_ev[2 * FS] >= _LIM) & (g == 0))
        f_fail = 0
        if DIAG:
            # wave-uniform
            f_fail = gl.maximum(gl.maximum(frl[1], f_itv), f_ev[2 * FS])
            if TAG:  # the row headers' tags (polled above by every lane)
                f_tb = rw * 0 + g * 0
                for t in gl.static_range(FS):
                    f_tb = f_tb | ((f_fp[6 * FS + 2 * t] >> 16) != tg).to(gl.int32)
                f_st = gl.where((f_tb != 0) & rok, SEQ_C, 0)
                gl.store(err, f_st, mask=(f_st == SEQ_C) & (g == 0))
                f_fail = gl.maximum(f_fail, f_st)
            diag_report(
                DIAG_BASE, f_fail, g + rw * 0, INST, _KIND, ME, role, pos, cnt, TRAP
            )
        for t in gl.static_range(FS):
            f_idx = (f * 4 + wv) * FS + t
            f_j = f_idx // NK
            f_k = f_idx % NK
            f_sb = f_fp[6 * FS + 2 * t]
            if TAG:
                f_sb = f_sb & 255
            f_sm = f_fp[6 * FS + 2 * t + 1]
            f_B7 = f_sb << 7
            f_de = _decode1p(
                f_fp[6 * t],
                f_fp[6 * t + 2],
                f_fp[6 * t + 4],
                f_ev[2 * t],
                ((f_sm >> (2 * g)) & 1) != 0,
                f_B7,
            )
            f_do = _decode1p(
                f_fp[6 * t + 1],
                f_fp[6 * t + 3],
                f_fp[6 * t + 5],
                f_ev[2 * t + 1],
                ((f_sm >> (2 * g + 1)) & 1) != 0,
                f_B7,
            )
            f_o64 = (
                out_ptr.to(gl.pointer_type(gl.int64))
                + row * (H // 4)
                + (f_j * OWN + f_k * 256) // 4
                + 4 * g
            )
            if TAIL:
                if pos == npos - 1:  # element-exact stores (and NaN) below n
                    f_te = row * H + f_j * OWN + f_k * 256 + 16 * g
                    _tail_st(
                        out_ptr,
                        f_te,
                        n,
                        _tail_q(_q64(f_de[0], f_de[1]), f_fail, DIAG),
                        0,
                    )
                    _tail_st(
                        out_ptr,
                        f_te,
                        n,
                        _tail_q(_q64(f_de[2], f_de[3]), f_fail, DIAG),
                        1,
                    )
                    _tail_st(
                        out_ptr,
                        f_te,
                        n,
                        _tail_q(_q64(f_do[0], f_do[1]), f_fail, DIAG),
                        2,
                    )
                    _tail_st(
                        out_ptr,
                        f_te,
                        n,
                        _tail_q(_q64(f_do[2], f_do[3]), f_fail, DIAG),
                        3,
                    )
                else:
                    gl.store(f_o64, _q64(f_de[0], f_de[1]), mask=rok)
                    gl.store(f_o64 + 1, _q64(f_de[2], f_de[3]), mask=rok)
                    gl.store(f_o64 + 2, _q64(f_do[0], f_do[1]), mask=rok)
                    gl.store(f_o64 + 3, _q64(f_do[2], f_do[3]), mask=rok)
                    if DIAG:
                        _nan_seg(f_fail, f_o64, row, m)
            else:
                gl.store(f_o64, _q64(f_de[0], f_de[1]), mask=rok)
                gl.store(f_o64 + 1, _q64(f_de[2], f_de[3]), mask=rok)
                gl.store(f_o64 + 2, _q64(f_do[0], f_do[1]), mask=rok)
                gl.store(f_o64 + 3, _q64(f_do[2], f_do[3]), mask=rok)
                if DIAG:  # a failed wave: NaN over its slice (cold, per wave)
                    _nan_seg(f_fail, f_o64, row, m)
            _rearm(
                lp + AG_OFF, ((b * W + f_j) * NK + f_k) * _SEGW + g * 0, r, g, rok, True
            )
        f_dz = _fence(role)
        gl.store(
            lp + REL_OFF + b * NF + f + f_dz + g * 0 + rw * 0,
            gl.zeros([16, 16], gl.int32, L2) + gen + 1,
            mask=(g == 0) & (rw == 0),
        )
    if last:
        gl.store(
            ctr_ptr + b * NROLE + role + g * 0 + rw * 0,
            gl.zeros([16, 16], gl.int32, L2) + gen + 1,
            mask=(g == 0) & (rw == 0),
        )


# ------------------------------------------------------------------ host side
WINDOW = 2048  # rows


def _positions(max_tokens):
    return min(WINDOW, cdiv(int(max_tokens), R) * R) // R


def _roles(world, hidden):
    nk = hidden // world // 256
    return world * nk // 4, nk // 2, world * nk // 4


def ipc_bytes(world, hidden, max_tokens):
    """IPC bytes per rank: two RS parities, the AG region, release flags."""
    P = _positions(max_tokens)
    seg_region = P * world * (hidden // world // 256) * SEGW
    return 4 * (3 * seg_region + P * _roles(world, hidden)[2])


def ctr_words(world, hidden, max_tokens):
    """Counter words of one instance (one per (window position, role))."""
    return _positions(max_tokens) * sum(_roles(world, hidden))


_FL = FastLauncher(two_shot_window_kernel)
_FL_TAG = FastLauncher(two_shot_window_tag_kernel)
_FL_TAIL = FastLauncher(two_shot_window_tail_kernel)
_FL_TAIL_TAG = FastLauncher(two_shot_window_tail_tag_kernel)
_KEYS = itertools.count(1)


class TwoShotWindowInst(PushInst):
    """One two_shot_window instance on a region of the workspace (bf16 [m, H],
    H % (TP * 1024) == 0, any m <= max_tokens)."""

    def __init__(
        self,
        rank,
        world,
        hidden,
        max_tokens,
        ipc,
        ctr,
        err,
        spin=None,
        diag=None,
        tag=False,
        site_box=None,
        fast=True,
    ):
        assert world in (2, 4, 8) and hidden % (world * 1024) == 0, (
            f"two_shot_window: hidden {hidden} must be a multiple of TP x 1024 ({world * 1024})"
        )
        self.rank, self.world, self.hidden = rank, world, int(hidden)
        self.max_tokens = int(max_tokens)
        self._key = next(_KEYS)
        self.fast = bool(fast)
        self.P = _positions(self.max_tokens)
        nk = self.hidden // world // 256
        self.ns, self.nr, self.nf = _roles(world, self.hidden)
        self.nrole = self.ns + self.nr + self.nf
        seg_region = self.P * world * nk * SEGW
        self.ag_off = 2 * seg_region  # RS: two parities per window slot
        self.rel_off = 3 * seg_region
        self.bytes_per_rank = 4 * (self.rel_off + self.P * self.nf)
        assert self.bytes_per_rank == ipc_bytes(world, self.hidden, self.max_tokens)
        self.buf = ipc.take(self.bytes_per_rank)
        loc = self.buf.addrs[rank]
        n = 3 * seg_region  # RS + AG slots at the sentinel; release flags 0
        fill_kernel[(cdiv(n, 1024),)](I32Ptr(loc), n, SENT, BLOCK=1024, num_warps=4)
        assert ctr.numel() >= self.P * self.nrole and err.numel() >= 1
        self.ctr, self.err = ctr[: self.P * self.nrole], err[:1]
        self.spin = spin if spin is not None else SpinBound()
        self.diag = diag
        self.tag = bool(tag)
        assert not self.tag or diag is not None, (
            "call tags need the diagnostics (diag=)"
        )
        self.site_box = site_box if site_box is not None else [0]
        ab = None
        if diag is not None:
            self.spin = no_trap(self.spin)
            ab = diag.abort
        b = self.spin
        offs = (0, 4, 512, 516, 1024, 1028)
        fp = asmgen.spin(
            "lp",
            nsrc=1,
            offs=offs,
            nsrc2=1,
            offs2=(0, 4),
            bound=b,
            abort=ab,
        )
        ge = asmgen.spin("flag_ge", bound=b, abort=ab)
        fe = asmgen.spin("lp", nsrc=2, offs=(0,), act=1, bound=b, abort=ab)
        r1 = asmgen.spin(
            "lp",
            nsrc=world,
            offs=(0, 512, 1024),
            nsrc2=world,
            offs2=(0, 4),
            bound=b,
            abort=ab,
        )
        e1 = asmgen.spin("lp", nsrc=world, offs=(0,), act=1, bound=b, abort=ab)
        addrs = [int(a) for a in self.buf.addrs] + [0] * (8 - world)
        self._kw = dict(
            ME=rank,
            W=world,
            H=self.hidden,
            P=self.P,
            AG_OFF=self.ag_off,
            REL_OFF=self.rel_off,
            **{f"P{j}": addrs[j] for j in range(8)},
            FP_ASM=fp[0],
            FP_CONS=fp[1],
            FP_NOUT=fp[2],
            GE_ASM=ge[0],
            GE_CONS=ge[1],
            FE_ASM=fe[0],
            FE_CONS=fe[1],
            FE_NOUT=fe[2],
            R1_ASM=r1[0],
            R1_CONS=r1[1],
            R1_NOUT=r1[2],
            E1_ASM=e1[0],
            E1_CONS=e1[1],
            E1_NOUT=e1[2],
            num_warps=4,
        )
        if diag is not None:
            self._kw.update(diag.kw())

    def _launch(self, x, out, warmup=False):
        m = x.shape[0]
        grid = (self.nrole, cdiv(m, R))
        fl, args = _FL, (x, out, self.ctr, self.err, m)
        if self.tag:
            fl, args = _FL_TAG, args + (int(self.site_box[0]),)
        if warmup:  # compile (no launch) and prime the fast-launch cache
            return fl.warm(grid, args, self._kw, key=(self._key,), fast=self.fast)
        fl(grid, args, self._kw, key=(self._key,), fast=self.fast)

    def precompile(self, x, ms=None, tail=False):
        """Compile the kernel without launching (ms: unused, the kernel does not
        specialize m); tail: also the masked-tail kernel."""
        self._launch(x, torch.empty_like(x), warmup=True)
        if tail:
            xf = x.reshape(-1)
            self._launch_tail(xf, torch.empty_like(xf), xf.numel(), warmup=True)

    def _launch_tail(self, x, out, n, warmup=False):
        m = cdiv(n, self.hidden)
        grid = (self.nrole, cdiv(m, R))
        fl, args = _FL_TAIL, (x, out, self.ctr, self.err, m, n)
        if self.tag:  # call tags: two_shot_window_tail_tag_kernel, with the site
            fl, args = (
                _FL_TAIL_TAG,
                (
                    x,
                    out,
                    self.ctr,
                    self.err,
                    m,
                    int(self.site_box[0]),
                    n,
                ),
            )
        if warmup:
            return fl.warm(
                grid, args, self._kw, key=(self._key, "tail"), fast=self.fast
            )
        fl(grid, args, self._kw, key=(self._key, "tail"), fast=self.fast)
