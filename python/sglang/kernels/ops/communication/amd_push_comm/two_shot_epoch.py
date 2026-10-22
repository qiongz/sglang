# SPDX-License-Identifier: Apache-2.0
"""two_shot_epoch: two-stage push all-reduce over a fixed window of rows with an
epoch flag per segment (stored after a publish fence), for large messages.

Wires: c12 (12-bit lossless codec) and int8 (its own format; error no worse
than QuickReduce INT8). Schedules: grid (one CTA per (role, position)) or
persistent channels. A sender writes generation gen of a window slot only after
this rank's finishers released gen - 1, and every wait is on a smaller CTA id.
two_shot_epoch_norm_kernel fuses the residual add + RMSNorm into the finishers
(lossless wire): res_out = bf16(f32(bf16(AR)) + f32(res)), out = bf16(res_out *
rstd * w), the row's sum of squares reduced in a fixed order."""

from __future__ import annotations

import itertools
import logging

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl
from triton.experimental.gluon.language.amd.cdna3 import buffer_load
from triton.runtime.jit import constexpr_function

from . import asmgen
from .codec import (
    _dec_i8,
    _decode,
    _enc_i8,
    _encode,
    _hi,
    _lo,
    _pair,
    _pk,
    _popc,
    _quad,
    _rbf,
)
from .launch import FastLauncher, PushInst
from .primitives import FAILED_C, LIM, SEQ_C, diag_report
from .primitives import publish_fence as _fence
from .primitives import seq_tag, spin_flag, vmdrain
from .primitives import zero as _zero
from .protocol import FAILED, SpinBound, no_trap

logger = logging.getLogger(__name__)

R = 8  # rows per position (= 2 rows per warp x 4 warps)
NW = 4
SEGW = 4 * 256 + 2 * R  # dwords per segment: planes 0..3 + header
SEGS = gl.constexpr(SEGW)
WIRES = {"c12": 1, "int8": 2}
# int8 call tags: one word per row (tag << 16) in the unused header dwords
TAGO = gl.constexpr(1024)

# the default bound's flag spins (asm, cons, nout, n_in)
_EQ = asmgen.spin("flag_eq")
_GE = asmgen.spin("flag_ge")
_SPIN_CONS = gl.constexpr(_EQ[1])


def spin_asm(bound, abort=None):
    """two_shot_epoch_kernel's SPIN_EQ / SPIN_GE constexprs for a
    protocol.SpinBound (abort: the host's abort word)."""
    return dict(
        SPIN_EQ=asmgen.spin("flag_eq", bound=bound, abort=abort)[0],
        SPIN_GE=asmgen.spin("flag_ge", bound=bound, abort=abort)[0],
    )


@gluon.jit
def _spin_eq(p, ep, ASM: gl.constexpr):
    """Per-wave spin until every lane's flag equals ep."""
    return spin_flag(p, ep, ASM, _SPIN_CONS)


@gluon.jit
def _spin_ge(p, thr, ASM: gl.constexpr):
    """Per-wave spin until every lane's counter is >= thr, compared as the signed
    difference counter - thr (survives the int32 wrap; returns the difference)."""
    return spin_flag(p, thr, ASM, _SPIN_CONS)


@constexpr_function
def _nan_asm(rf, w4):
    """A wave with a failed lane ($6) stores NaN over output ($7) rows $8 ..
    $8 + rf - 1 (< $9); $0 .. $5 are scratch."""
    rf, w4 = (int(a.value if isinstance(a, gl.constexpr) else a) for a in (rf, w4))
    return "\n".join(
        [
            f"v_cmp_le_u32 vcc, {FAILED:#x}, $6",
            "s_cbranch_vccz 5f",
            "v_mov_b32 $0, 0x7fc07fc0",
            "v_mbcnt_lo_u32_b32 $2, -1, 0",
            "v_mbcnt_hi_u32_b32 $2, -1, $2",
            "v_lshlrev_b32 $2, 2, $2",
            "s_mov_b32 $3, $8",
            "2:",
            "s_cmp_ge_i32 $3, $9",
            "s_cbranch_scc1 5f",
            "s_sub_u32 $4, $3, $8",
            f"s_cmp_ge_u32 $4, {rf}",
            "s_cbranch_scc1 5f",
            f"s_mul_i32 $4, $3, {w4}",
            "v_add_u32 $1, $4, $2",
            f"s_mov_b32 $5, {w4}",
            "3:",
            "global_store_dword $1, $0, $7",
            "v_add_u32 $1, 256, $1",
            "s_sub_u32 $5, $5, 256",
            "s_cmp_eq_u32 $5, 0",
            "s_cbranch_scc0 3b",
            "s_add_u32 $3, $3, 1",
            "s_branch 2b",
            "5:",
        ]
    )


@gluon.jit
def _nan_fail(
    itr,
    out_ptr,
    r0,
    m,
    RF: gl.constexpr,
    HID: gl.constexpr,
):
    """A finisher wave whose wait failed overwrites the whole rows r0 .. r0 + RF -
    1 (< m) of the output with bf16 NaN, after its own normal stores (same wave:
    program order). One compare + branch when nothing failed."""
    ASM: gl.constexpr = _nan_asm(RF, HID * 2)
    gl.inline_asm_elementwise(
        ASM,
        "=&v,=&v,=&v,=&s,=&s,=&s,v,s,s,s,~{vcc},~{memory}",
        [
            itr,
            out_ptr.to(gl.int64, bitcast=True),
            r0,
            m,
        ],
        dtype=(gl.int32,) * 6,
        is_pure=False,
        pack=1,
    )


@constexpr_function
def _nan_asm_norm(rf, w4):
    """_nan_asm over the output ($7) and the residual output ($8) of the rows
    $9 .. $9 + rf - 1 (< $10)."""
    rf, w4 = (int(a.value if isinstance(a, gl.constexpr) else a) for a in (rf, w4))
    return "\n".join(
        [
            f"v_cmp_le_u32 vcc, {FAILED:#x}, $6",
            "s_cbranch_vccz 5f",
            "v_mov_b32 $0, 0x7fc07fc0",
            "v_mbcnt_lo_u32_b32 $2, -1, 0",
            "v_mbcnt_hi_u32_b32 $2, -1, $2",
            "v_lshlrev_b32 $2, 2, $2",
            "s_mov_b32 $3, $9",
            "2:",
            "s_cmp_ge_i32 $3, $10",
            "s_cbranch_scc1 5f",
            "s_sub_u32 $4, $3, $9",
            f"s_cmp_ge_u32 $4, {rf}",
            "s_cbranch_scc1 5f",
            f"s_mul_i32 $4, $3, {w4}",
            "v_add_u32 $1, $4, $2",
            f"s_mov_b32 $5, {w4}",
            "3:",
            "global_store_dword $1, $0, $7",
            "global_store_dword $1, $0, $8",
            "v_add_u32 $1, 256, $1",
            "s_sub_u32 $5, $5, 256",
            "s_cmp_eq_u32 $5, 0",
            "s_cbranch_scc0 3b",
            "s_add_u32 $3, $3, 1",
            "s_branch 2b",
            "5:",
        ]
    )


@gluon.jit
def _nan_fail_norm(
    itr,
    out_ptr,
    resout_ptr,
    r0,
    m,
    RF: gl.constexpr,
    HID: gl.constexpr,
):
    """_nan_fail over the output and the residual output."""
    ASM: gl.constexpr = _nan_asm_norm(RF, HID * 2)
    gl.inline_asm_elementwise(
        ASM,
        "=&v,=&v,=&v,=&s,=&s,=&s,v,s,s,s,s,~{vcc},~{memory}",
        [
            itr,
            out_ptr.to(gl.int64, bitcast=True),
            resout_ptr.to(gl.int64, bitcast=True),
            r0,
            m,
        ],
        dtype=(gl.int32,) * 6,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _norm_store(
    vals,
    rts,
    wts,
    cols,
    oks,
    rout,
    rro,
    eps,
    ri2,
    u2,
    q3,
    S2: gl.constexpr,
    NSEG: gl.constexpr,
    KK: gl.constexpr,
    HID: gl.constexpr,
):
    """One row's res_out and out; the squares summed over the [8, 32] tile in
    a fixed order."""
    sq = gl.zeros([8, 32], gl.float32, S2)
    rrs = ()
    for kk in gl.static_range(KK):
        rt = rts[kk]
        okk = (kk * 8 + ri2 < NSEG) & (u2 >= 0)
        for qq in gl.static_range(4):
            rp = _pair(rt, q3, qq)
            lo = vals[kk * 8 + qq * 2] + _lo(rp)
            hi = vals[kk * 8 + qq * 2 + 1] + _hi(rp)
            sq = sq + gl.where(okk, lo * lo + hi * hi, gl.zeros_like(lo))
            rrs += (lo, hi)
    inv = gl.rsqrt(gl.sum(gl.sum(sq, axis=1), axis=0) / HID + eps)
    for kk in gl.static_range(KK):
        wt = wts[kk]
        o4 = ()
        r4 = ()
        for qq in gl.static_range(4):
            wp = _pair(wt, q3, qq)
            lo = rrs[kk * 8 + qq * 2]
            hi = rrs[kk * 8 + qq * 2 + 1]
            o4 += (_pk(lo * inv * _lo(wp), hi * inv * _hi(wp)),)
            r4 += (_pk(lo, hi),)
        gl.store(rro + cols[kk], _quad(r4[0], r4[1], r4[2], r4[3], q3), mask=oks[kk])
        gl.store(rout + cols[kk], _quad(o4[0], o4[1], o4[2], o4[3], q3), mask=oks[kk])


@gluon.jit
def _tag_put(segp, ri2, u2, nval, tw, TO: gl.constexpr):
    """Call tags off the c12 wire: the rows' tag words (tw: tag << 16, scalar
    or one per row) at segment dword TO + 2 r, from lane u2 == 0 of each valid
    row."""
    r1 = gl.sum(ri2, axis=1)
    tv = gl.expand_dims(r1 * 0 + tw, 1)
    gl.store(segp + TO + ri2 * 2 + u2 * 0, tv + u2 * 0, mask=(ri2 < nval) & (u2 == 0))


@gluon.jit
def _tag_get(lz, seg, rr, u2, TO: gl.constexpr):
    """The tag field of the row's tag word (one load per row; see _tag_put)."""
    return buffer_load(lz, seg + TO + rr * 2 + u2 * 0, cache=".cv") >> 16


@gluon.jit
def _put(
    segp,
    enc,
    ri2,
    u2,
    rv2,
    nval,
    DRAIN: gl.constexpr = 1,
    TAG: gl.constexpr = 0,
    tw=0,
    WIRE: gl.constexpr = 1,
):
    """Store one encoded segment (8 rows x 256 columns, nval valid). TAG: tw =
    tag << 16, in the base words (c12) or the rows' tag words (int8)."""
    o = ri2 * 32 + u2
    if WIRE == 2:
        _put_i8(segp, enc, o, rv2, nval)
        if TAG:  # int8's tag word (dwords 1024.. are unused by WIRE 2)
            _tag_put(segp, ri2, u2, nval, tw, TAGO)
    else:
        _put_c12(segp, enc, o, ri2, u2, rv2, nval, TAG, tw)
    if DRAIN:
        # bound each wave's in-flight remote stores to one segment: keeps the
        # many concurrent writer waves from over-filling the fabric queues
        vmdrain(nval)


@gluon.jit
def _put_i8(segp, enc, o, rv2, nval):
    w0, w1, hdr = enc
    gl.store(segp + o, w0, mask=rv2)
    gl.store(segp + 256 + o, w1, mask=rv2)
    # [8 rows, 4 dwords] of bf16 scale pairs = 128 B, from one wave
    LH8: gl.constexpr = gl.BlockedLayout([1, 1], [16, 4], [4, 1], [1, 0])
    hh = gl.convert_layout(hdr, LH8)
    rh = gl.expand_dims(gl.arange(0, 8, layout=gl.SliceLayout(1, LH8)), 1)
    dh = gl.expand_dims(gl.arange(0, 4, layout=gl.SliceLayout(0, LH8)), 0)
    gl.store(segp + 512 + rh * 4 + dh, hh, mask=rh < nval)


@gluon.jit
def _put_c12(segp, enc, o, ri2, u2, rv2, nval, TAG: gl.constexpr, tw):
    w0, w1, w2, w3, esc, base, msk = enc
    gl.store(segp + o, w0, mask=rv2)
    gl.store(segp + 256 + o, w1, mask=rv2)
    gl.store(segp + 512 + o, w2, mask=rv2)
    # the k-th escaped unit of a row stores its raw pair 3 at slot k
    slot3 = _popc(gl.expand_dims(msk, 1) & ((1 << u2) - 1))
    gl.store(segp + 768 + ri2 * 32 + slot3, w3, mask=rv2 & (esc != 0))
    # the 8 rows' {base, mask}: one 64 B line, one request
    if TAG:
        base = base | tw
    hv = gl.where(u2 == 0, gl.expand_dims(base, 1), gl.expand_dims(msk, 1))
    LH: gl.constexpr = gl.BlockedLayout([1, 1], [8, 8], [1, 4], [0, 1])
    hh = gl.convert_layout(hv, LH)
    rh = gl.expand_dims(gl.arange(0, 8, layout=gl.SliceLayout(1, LH)), 1)
    uh = gl.expand_dims(gl.arange(0, 32, layout=gl.SliceLayout(0, LH)), 0)
    gl.store(segp + 1024 + rh * 2 + uh, hh, mask=(rh < nval) & (uh < 2))


@gluon.jit
def _get(lz, seg, rr, u2, TAG: gl.constexpr = 0, WIRE: gl.constexpr = 1):
    """Load + decode one segment slice -> 8 fp32 tiles (TAG: + the row's tag
    field)."""
    o = seg + rr * 32 + u2
    if WIRE == 2:
        w0 = buffer_load(lz, o, cache=".cv")
        w1 = buffer_load(lz, o + 256, cache=".cv")
        hd = buffer_load(lz, seg + 512 + rr * 4 + u2 // 8, cache=".cv")
        if TAG:
            return _dec_i8(w0, w1, hd, u2) + (_tag_get(lz, seg, rr, u2, TAGO),)
        else:
            return _dec_i8(w0, w1, hd, u2)
    else:
        w0 = buffer_load(lz, o, cache=".cv")
        w1 = buffer_load(lz, o + 256, cache=".cv")
        w2 = buffer_load(lz, o + 512, cache=".cv")
        ho = seg + 1024 + rr * 2 + u2 * 0
        bs = buffer_load(lz, ho, cache=".cv")
        mk = buffer_load(lz, ho + 1, cache=".cv")
        esc = (mk >> u2) & 1
        w3 = buffer_load(
            lz,
            seg + 768 + rr * 32 + _popc(mk & ((1 << u2) - 1)),
            mask=esc != 0,
            other=0,
            cache=".cv",
        )
        if TAG:
            p0, p1, p2, p3 = _decode(w0, w1, w2, w3, esc, bs & 255)
            return (
                _lo(p0),
                _hi(p0),
                _lo(p1),
                _hi(p1),
                _lo(p2),
                _hi(p2),
                _lo(p3),
                _hi(p3),
                bs >> 16,
            )
        else:
            p0, p1, p2, p3 = _decode(w0, w1, w2, w3, esc, bs)
            return (
                _lo(p0),
                _hi(p0),
                _lo(p1),
                _hi(p1),
                _lo(p2),
                _hi(p2),
                _lo(p3),
                _hi(p3),
            )


# ----------------------------------------------------------------- the kernel
@gluon.jit
def _ctr_load(p, PERSIST: gl.constexpr):
    """The rotation counter as a scalar load (spelled out when persistent: the
    previous position's stores make LLVM pick a vector load)."""
    if PERSIST:
        return gl.inline_asm_elementwise(
            "s_load_dword $0, $1, 0x0\ns_waitcnt lgkmcnt(0)",
            "=s,s,~{memory}",
            [p.to(gl.int64)],
            dtype=gl.int32,
            is_pure=False,
            pack=1,
        )
    else:
        return gl.load(p)


@gluon.jit
def _peer(ptrs, pv, j, NRANK: gl.constexpr, PERSIST: gl.constexpr):
    """Peer j's buffer base (j a runtime value). Grid schedule: a scalar load of
    the pointer table per use; persistent: a select from the table loaded once
    per CTA (a dependent global load per put would serialise a channel)."""
    if PERSIST:
        r = pv[0]
        for i in gl.static_range(1, NRANK):
            r = gl.where(j == i, pv[i], r)
        return r.to(gl.pointer_type(gl.int32))
    else:
        return gl.load(ptrs + j).to(gl.pointer_type(gl.int32))


@gluon.jit
def _peerc(ptrs, pv, J, PERSIST: gl.constexpr):
    """_peer for a peer index that is a compile-time value in the code (ME,
    (ME + jj) % NRANK, ...): the if-chain folds away when J is a constexpr."""
    if PERSIST:
        r = pv[0]
        for i in gl.static_range(1, len(pv)):
            if J == i:
                r = pv[i]
        return r.to(gl.pointer_type(gl.int32))
    else:
        return gl.load(ptrs + J).to(gl.pointer_type(gl.int32))


@gluon.jit
def _epoch_pos(
    pos,
    role,
    x_ptr,
    out_ptr,
    ptrs,
    pv,
    ctr_ptr,
    err_ptr,
    m,
    site,
    ME: gl.constexpr,
    NRANK: gl.constexpr,
    P: gl.constexpr,
    AG_OFF: gl.constexpr,
    RSF_OFF: gl.constexpr,
    AGF_OFF: gl.constexpr,
    REL_OFF: gl.constexpr,
    NSEND: gl.constexpr,
    RED: gl.constexpr,
    F: gl.constexpr,
    WIRE: gl.constexpr,
    HIDDEN: gl.constexpr,
    PERSIST: gl.constexpr,
    DRAIN: gl.constexpr,
    SPIN_EQ: gl.constexpr,
    SPIN_GE: gl.constexpr,
    DIAG: gl.constexpr,
    DIAG_BASE: gl.constexpr,
    INST: gl.constexpr,
    TRAP: gl.constexpr,
    TAG: gl.constexpr,
    NORM: gl.constexpr = 0,
    res_ptr=0,
    w_ptr=0,
    resout_ptr=0,
    eps=0.0,
):
    """One position (8 rows) of one role. NORM: the finishers add the residual
    and apply the RMSNorm (_norm_store). TAG: a reducer forwards a foreign tag
    poisoned; a finisher that reads one fails its wave (SEQ_MISMATCH)."""
    HID: gl.constexpr = HIDDEN
    OWN: gl.constexpr = HID // NRANK
    NK: gl.constexpr = OWN // 256  # 256-column blocks per owner
    NSEG: gl.constexpr = NRANK * NK  # = HID / 256 (owner, block) segments
    KK: gl.constexpr = (NSEG + 7) // 8  # finisher tiles of 8 segments
    NAGF: gl.constexpr = NRANK * RED
    b = pos % P  # buffer position (window of P positions)
    NROLE: gl.constexpr = NSEND + RED + F
    # the CTA's first memory access, a scalar load
    cnt = _ctr_load(ctr_ptr + b * NROLE + role, PERSIST)
    gen = cnt + pos // P  # generation of buffer position b used by this CTA
    ep = gen + 1
    # the last use of b in this launch advances the counter
    last = pos + P >= (m + 7) // 8
    if TAG:
        tg = seq_tag(m, site)

    L3: gl.constexpr = gl.BlockedLayout([1, 1, 4], [2, 32, 1], [4, 1, 1], [2, 1, 0])
    S2: gl.constexpr = gl.SliceLayout(2, L3)
    ri = gl.arange(0, 8, layout=gl.SliceLayout(1, S2))
    ri2 = gl.expand_dims(ri, 1)
    u2 = gl.expand_dims(gl.arange(0, 32, layout=gl.SliceLayout(0, S2)), 0)
    q = gl.arange(0, 4, layout=gl.SliceLayout(0, gl.SliceLayout(1, L3)))
    q3 = gl.expand_dims(gl.expand_dims(q, 0), 1)
    ri3 = gl.expand_dims(ri2, 2)
    u3 = gl.expand_dims(u2, 2)
    LP: gl.constexpr = gl.BlockedLayout([1], [64], [4], [0])
    sp = gl.arange(0, 256, layout=LP)
    lp = _peerc(ptrs, pv, ME, PERSIST)
    row0 = pos * 8
    x32 = x_ptr.to(gl.pointer_type(gl.int32))

    if role < NSEND:
        rows3 = row0 + ri3
        rv3 = rows3 < m
        rv2 = (row0 + ri2) < m
        xs = ()
        for jj in gl.static_range(NRANK // NSEND):
            for k in gl.static_range(NK):
                j = role + jj * NSEND
                col = j * (OWN // 2) + k * 128 + u3 * 4 + q3
                xs += (gl.load(x32 + rows3 * (HID // 2) + col, mask=rv3, other=0),)
        enc = ()
        for i in gl.static_range((NRANK // NSEND) * NK):
            t = xs[i]
            if WIRE == 2:
                pp = (
                    _pair(t, q3, 0),
                    _pair(t, q3, 1),
                    _pair(t, q3, 2),
                    _pair(t, q3, 3),
                )
                enc += (
                    _enc_i8(
                        _lo(pp[0]),
                        _hi(pp[0]),
                        _lo(pp[1]),
                        _hi(pp[1]),
                        _lo(pp[2]),
                        _hi(pp[2]),
                        _lo(pp[3]),
                        _hi(pp[3]),
                        S2,
                    ),
                )
            else:
                enc += (
                    _encode(
                        _pair(t, q3, 0),
                        _pair(t, q3, 1),
                        _pair(t, q3, 2),
                        _pair(t, q3, 3),
                        u2,
                    ),
                )
        # write-after-read (local release): this rank's finishers released
        # generation gen - 1 of b
        rl = REL_OFF + b * F + (sp % F)
        _r, itr = _spin_ge(lp + rl, sp * 0 + gen, SPIN_GE)
        gl.store(err_ptr + sp * 0, itr, mask=(itr >= LIM) & (sp == 0))
        ok = sp == 0
        if DIAG:
            # a failed wave pushes nothing and a failed wave 0 publishes no flag,
            # so the peers' waits fail too and every rank reports NaN
            diag_report(DIAG_BASE, itr, sp, INST, 3, ME, role, pos, cnt, TRAP)
            # per wave: no cross-wave reduction on the hot path
            good = itr < FAILED_C
            # the same lane -> element map
            rv2 = rv2 & gl.convert_layout(gl.reshape(good, [8, 32]), S2)
            ok = ok & good
        for jj in gl.static_range(NRANK // NSEND):
            j = role + jj * NSEND
            dp = _peer(ptrs, pv, j, NRANK, PERSIST) + _zero(role)
            for k in gl.static_range(NK):
                segp = dp + ((b * NSEG) + ME * NK + k) * SEGS
                _put(
                    segp,
                    enc[jj * NK + k],
                    ri2,
                    u2,
                    rv2,
                    m - row0,
                    DRAIN,
                    TAG,
                    tg << 16 if TAG else 0,
                    WIRE,
                )
        d = _fence(role)
        for jj in gl.static_range(NRANK // NSEND):
            j = role + jj * NSEND
            dp = _peer(ptrs, pv, j, NRANK, PERSIST)
            fo = RSF_OFF + b * NRANK + ME + d
            gl.store(dp + fo + sp * 0, sp * 0 + ep, mask=ok)

    elif role < NSEND + RED:
        red = role - NSEND
        fl = RSF_OFF + b * NRANK + (sp % NRANK)
        _v, itr = _spin_eq(lp + fl, sp * 0 + ep, SPIN_EQ)
        gl.store(err_ptr + sp * 0, itr, mask=(itr >= LIM) & (sp == 0))
        rv2 = (row0 + ri2) < m
        ok = sp == 0
        if DIAG:  # a failed wait: nothing pushed, no flag (see the sender)
            diag_report(DIAG_BASE, itr, sp, INST, 3, ME, role, pos, cnt, TRAP)
            good = itr < FAILED_C
            rv2 = rv2 & gl.convert_layout(gl.reshape(good, [8, 32]), S2)
            ok = ok & good
        lz = lp + _zero(role)
        tbad = (ri2 + u2) < 0  # TAG: rows that read a foreign call tag
        for kk in gl.static_range(NK // RED):
            k = red + kk * RED
            g = _get(lz, ((b * NSEG) + k) * SEGS, ri2, u2, TAG, WIRE)
            a0, a1, a2, a3, a4, a5, a6, a7 = (
                g[0],
                g[1],
                g[2],
                g[3],
                g[4],
                g[5],
                g[6],
                g[7],
            )
            if TAG:
                # rows >= m: no header written (stale tag)
                tbad = tbad | ((g[8] != tg) & rv2)
            for src in gl.static_range(1, NRANK):  # fixed rank order 0..NRANK-1
                g = _get(lz, ((b * NSEG) + src * NK + k) * SEGS, ri2, u2, TAG, WIRE)
                if TAG:
                    tbad = tbad | ((g[8] != tg) & rv2)
                a0 = a0 + g[0]
                a1 = a1 + g[1]
                a2 = a2 + g[2]
                a3 = a3 + g[3]
                a4 = a4 + g[4]
                a5 = a5 + g[5]
                a6 = a6 + g[6]
                a7 = a7 + g[7]
            if WIRE == 2:
                # requantize the fp32 sum directly: rounding it to bf16 first
                # would add a third rounding to the two quantizations
                ec = _enc_i8(a0, a1, a2, a3, a4, a5, a6, a7, S2)
            else:
                ec = _encode(
                    _pk(a0, a1), _pk(a2, a3), _pk(a4, a5), _pk(a6, a7), u2
                )  # rounded once
            for jj in gl.static_range(NRANK):
                dj = (ME + jj) % NRANK  # stagger destinations across owners
                dp = _peerc(ptrs, pv, dj, PERSIST)
                segp = dp + AG_OFF + ((b * NSEG) + ME * NK + k) * SEGS
                if TAG:  # a row that read a foreign tag forwards a poisoned one
                    tw = (
                        gl.where(gl.max(tbad.to(gl.int32), 1) > 0, tg ^ 16384, tg) << 16
                    )
                    _put(segp, ec, ri2, u2, rv2, m - row0, DRAIN, TAG, tw, WIRE)
                else:
                    _put(segp, ec, ri2, u2, rv2, m - row0, DRAIN, 0, 0, WIRE)
        if TAG:
            st = gl.where(tbad, SEQ_C, 0)
            gl.store(err_ptr + ri2 * 0 + u2 * 0, st, mask=st == SEQ_C)
            diag_report(DIAG_BASE, st, ri2 * 32 + u2, INST, 3, ME, role, pos, cnt, TRAP)
        d = _fence(role)
        for jj in gl.static_range(NRANK):
            dp = _peerc(ptrs, pv, jj, PERSIST)
            fo = AGF_OFF + b * NAGF + ME * RED + red + d
            gl.store(dp + fo + sp * 0, sp * 0 + ep, mask=ok)

    else:
        fi = role - NSEND - RED
        fl = AGF_OFF + b * NAGF + (sp % NAGF)
        _v, itr = _spin_eq(lp + fl, sp * 0 + ep, SPIN_EQ)
        gl.store(err_ptr + sp * 0, itr, mask=(itr >= LIM) & (sp == 0))
        if DIAG:
            diag_report(DIAG_BASE, itr, sp, INST, 3, ME, role, pos, cnt, TRAP)
        lz = lp + AG_OFF + _zero(role)
        ax = ri2  # axis 0 of the finisher tile = segment kk * 8 + ax
        ax3 = gl.expand_dims(ax, 2)
        RF: gl.constexpr = 8 // F
        tbad = (ri2 + u2) < 0  # TAG: segment rows that carried a foreign call tag
        for rr_ in range(RF):
            rloc = fi * RF + rr_
            row = row0 + rloc
            if row < m:
                rout = out_ptr.to(gl.pointer_type(gl.int32)) + row * (HID // 2)
                vals = ()
                cols = ()
                oks = ()
                for kk in gl.static_range(KK):
                    # segment = owner * NK + block (a partial tile reads a
                    # duplicate)
                    sg2 = gl.minimum(kk * 8 + ax, NSEG - 1)
                    sg3 = gl.minimum(kk * 8 + ax3, NSEG - 1)
                    seg2 = ((b * NSEG) + sg2) * SEGS
                    if TAG:
                        v0, v1, v2, v3, v4, v5, v6, v7, tb = _get(
                            lz, seg2, rloc, u2, TAG, WIRE
                        )
                        tbad = tbad | (tb != tg)
                    else:
                        v0, v1, v2, v3, v4, v5, v6, v7 = _get(
                            lz, seg2, rloc, u2, 0, WIRE
                        )
                    cols += ((sg3 // NK) * (OWN // 2) + (sg3 % NK) * 128 + u3 * 4 + q3,)
                    oks += ((kk * 8 + ax3 < NSEG) & (q3 >= 0) & (u3 >= 0),)
                    vals += (
                        _rbf(v0),
                        _rbf(v1),
                        _rbf(v2),
                        _rbf(v3),
                        _rbf(v4),
                        _rbf(v5),
                        _rbf(v6),
                        _rbf(v7),
                    )
                if NORM:
                    rres = res_ptr.to(gl.pointer_type(gl.int32)) + row * (HID // 2)
                    w32 = w_ptr.to(gl.pointer_type(gl.int32))
                    rro = resout_ptr.to(gl.pointer_type(gl.int32)) + row * (HID // 2)
                    rts = ()
                    wts = ()
                    for kk in gl.static_range(KK):
                        rts += (gl.load(rres + cols[kk]),)
                        wts += (gl.load(w32 + cols[kk]),)
                    _norm_store(
                        vals,
                        rts,
                        wts,
                        cols,
                        oks,
                        rout,
                        rro,
                        eps,
                        ri2,
                        u2,
                        q3,
                        S2,
                        NSEG,
                        KK,
                        HID,
                    )
                else:
                    for kk in gl.static_range(KK):
                        gl.store(
                            rout + cols[kk],
                            _quad(
                                _pk(vals[kk * 8], vals[kk * 8 + 1]),
                                _pk(vals[kk * 8 + 2], vals[kk * 8 + 3]),
                                _pk(vals[kk * 8 + 4], vals[kk * 8 + 5]),
                                _pk(vals[kk * 8 + 6], vals[kk * 8 + 7]),
                                q3,
                            ),
                            mask=oks[kk],
                        )
        if TAG:  # a foreign tag fails the wave like a failed wait
            st = gl.maximum(
                gl.convert_layout(gl.reshape(itr, [8, 32]), S2),
                gl.where(tbad, SEQ_C, 0),
            )
            gl.store(err_ptr + ri2 * 0 + u2 * 0, st, mask=st == SEQ_C)
            diag_report(
                DIAG_BASE,
                gl.where(tbad, SEQ_C, 0),
                ri2 * 32 + u2,
                INST,
                3,
                ME,
                role,
                pos,
                cnt,
                TRAP,
            )
            if NORM:
                _nan_fail_norm(st, out_ptr, resout_ptr, row0 + fi * RF, m, RF, HID)
            else:
                _nan_fail(st, out_ptr, row0 + fi * RF, m, RF, HID)
        elif DIAG:  # a failed wave: NaN over this finisher's rows
            if NORM:
                _nan_fail_norm(itr, out_ptr, resout_ptr, row0 + fi * RF, m, RF, HID)
            else:
                _nan_fail(itr, out_ptr, row0 + fi * RF, m, RF, HID)
        # release (local): this rank has consumed generation gen of position b
        # (the fence drains the loads)
        d = _fence(role)
        gl.store(lp + REL_OFF + b * F + fi + d + sp * 0, sp * 0 + gen + 1, mask=sp == 0)

    if last:
        gl.store(ctr_ptr + b * NROLE + role + sp * 0, sp * 0 + gen + 1, mask=sp == 0)


@gluon.jit(do_not_specialize=["m", "site"])
def two_shot_epoch_kernel(
    x_ptr,
    out_ptr,
    ptrs,
    ctr_ptr,
    err_ptr,
    m,
    site,
    ME: gl.constexpr,
    NRANK: gl.constexpr,
    P: gl.constexpr,
    AG_OFF: gl.constexpr,
    RSF_OFF: gl.constexpr,
    AGF_OFF: gl.constexpr,
    REL_OFF: gl.constexpr,
    NSEND: gl.constexpr,
    RED: gl.constexpr,
    F: gl.constexpr,
    WIRE: gl.constexpr = 1,
    HIDDEN: gl.constexpr = 6144,
    PERSIST: gl.constexpr = 0,
    DRAIN: gl.constexpr = 1,
    XS: gl.constexpr = 0,
    SPIN_EQ: gl.constexpr = _EQ[0],
    SPIN_GE: gl.constexpr = _GE[0],
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TAG: gl.constexpr = 0,
):
    """Grid (NROLE, NPC). PERSIST = 0: one position per CTA. PERSIST = 1: CTA
    (role, c) streams positions c, c + NPC, ...; all CTAs must be co-resident.
    XS = 2 pins the senders' channels to their XCDs."""
    role = gl.program_id(0)
    if PERSIST:
        NR: gl.constexpr = NSEND + RED + F
        c0 = gl.program_id(1)
        if XS == 2:
            # workgroup L runs on XCD L % 8; its first NPC * NSEND / 8 CTAs are
            # sender role x % NSEND, the rest reducers / finishers
            L = gl.program_id(0) + NR * gl.program_id(1)
            x = L % 8
            i = L // 8
            SP = (gl.num_programs(1) * NSEND) // 8
            j = (i - SP) * 8 + x
            role = gl.where(i < SP, x % NSEND, NSEND + j % (NR - NSEND))
            c0 = gl.where(i < SP, (x // NSEND) * SP + i, j // (NR - NSEND))
        pv = ()  # peer bases, loaded once per CTA
        for i in gl.static_range(NRANK):
            pv += (gl.load(ptrs + i).to(gl.int64),)
        npos = (m + 7) // 8
        kch = gl.num_programs(1)
        for pos in range(c0, npos, kch):
            _epoch_pos(
                pos,
                role,
                x_ptr,
                out_ptr,
                ptrs,
                pv,
                ctr_ptr,
                err_ptr,
                m,
                site,
                ME,
                NRANK,
                P,
                AG_OFF,
                RSF_OFF,
                AGF_OFF,
                REL_OFF,
                NSEND,
                RED,
                F,
                WIRE,
                HIDDEN,
                PERSIST,
                DRAIN,
                SPIN_EQ,
                SPIN_GE,
                DIAG,
                DIAG_BASE,
                INST,
                TRAP,
                TAG,
            )
    else:
        _epoch_pos(
            gl.program_id(1),
            role,
            x_ptr,
            out_ptr,
            ptrs,
            ptrs,
            ctr_ptr,
            err_ptr,
            m,
            site,
            ME,
            NRANK,
            P,
            AG_OFF,
            RSF_OFF,
            AGF_OFF,
            REL_OFF,
            NSEND,
            RED,
            F,
            WIRE,
            HIDDEN,
            PERSIST,
            DRAIN,
            SPIN_EQ,
            SPIN_GE,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            TAG,
        )


@gluon.jit(do_not_specialize=["m", "site"])
def two_shot_epoch_norm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    resout_ptr,
    ptrs,
    ctr_ptr,
    err_ptr,
    m,
    eps,
    site,
    ME: gl.constexpr,
    NRANK: gl.constexpr,
    P: gl.constexpr,
    AG_OFF: gl.constexpr,
    RSF_OFF: gl.constexpr,
    AGF_OFF: gl.constexpr,
    REL_OFF: gl.constexpr,
    NSEND: gl.constexpr,
    RED: gl.constexpr,
    F: gl.constexpr,
    WIRE: gl.constexpr = 1,
    HIDDEN: gl.constexpr = 6144,
    PERSIST: gl.constexpr = 0,
    DRAIN: gl.constexpr = 1,
    XS: gl.constexpr = 0,
    SPIN_EQ: gl.constexpr = _EQ[0],
    SPIN_GE: gl.constexpr = _GE[0],
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TAG: gl.constexpr = 0,
):
    """two_shot_epoch_kernel with the residual add + RMSNorm; a separate kernel
    so that its arguments do not change the plain kernel."""
    role = gl.program_id(0)
    if PERSIST:
        NR: gl.constexpr = NSEND + RED + F
        c0 = gl.program_id(1)
        if XS == 2:
            # workgroup L runs on XCD L % 8; its first NPC * NSEND / 8 CTAs are
            # sender role x % NSEND, the rest reducers / finishers
            L = gl.program_id(0) + NR * gl.program_id(1)
            x = L % 8
            i = L // 8
            SP = (gl.num_programs(1) * NSEND) // 8
            j = (i - SP) * 8 + x
            role = gl.where(i < SP, x % NSEND, NSEND + j % (NR - NSEND))
            c0 = gl.where(i < SP, (x // NSEND) * SP + i, j // (NR - NSEND))
        pv = ()  # peer bases, loaded once per CTA
        for i in gl.static_range(NRANK):
            pv += (gl.load(ptrs + i).to(gl.int64),)
        npos = (m + 7) // 8
        kch = gl.num_programs(1)
        for pos in range(c0, npos, kch):
            _epoch_pos(
                pos,
                role,
                x_ptr,
                out_ptr,
                ptrs,
                pv,
                ctr_ptr,
                err_ptr,
                m,
                site,
                ME,
                NRANK,
                P,
                AG_OFF,
                RSF_OFF,
                AGF_OFF,
                REL_OFF,
                NSEND,
                RED,
                F,
                WIRE,
                HIDDEN,
                PERSIST,
                DRAIN,
                SPIN_EQ,
                SPIN_GE,
                DIAG,
                DIAG_BASE,
                INST,
                TRAP,
                TAG,
                1,
                res_ptr,
                w_ptr,
                resout_ptr,
                eps,
            )
    else:
        _epoch_pos(
            gl.program_id(1),
            role,
            x_ptr,
            out_ptr,
            ptrs,
            ptrs,
            ctr_ptr,
            err_ptr,
            m,
            site,
            ME,
            NRANK,
            P,
            AG_OFF,
            RSF_OFF,
            AGF_OFF,
            REL_OFF,
            NSEND,
            RED,
            F,
            WIRE,
            HIDDEN,
            PERSIST,
            DRAIN,
            SPIN_EQ,
            SPIN_GE,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            TAG,
            1,
            res_ptr,
            w_ptr,
            resout_ptr,
            eps,
        )


# ------------------------------------------------------------------ host side
F_ROLES = 8  # finisher CTAs per position


def region_dwords(P, world, hidden):
    """(AG_OFF, RSF_OFF, AGF_OFF, REL_OFF, total) in dwords for P positions."""
    red = hidden // 256 // world
    seg = P * (hidden // 256) * SEGW
    rsf = 2 * seg
    agf = rsf + P * world
    rel = agf + P * world * red
    return seg, rsf, agf, rel, rel + P * F_ROLES


def _window(max_tokens, window):
    return min(int(window), ((int(max_tokens) + R - 1) // R) * R)


def _check_geom(geom):
    """The table geometry this module runs: f = 8, ldspad = 0."""
    g = dict(geom)
    assert g.pop("f", F_ROLES) == F_ROLES and g.pop("ldspad", 0) == 0, geom
    return g


def ipc_bytes(world, hidden, max_tokens, window=2048, **_geom):
    """IPC bytes per rank of one instance (any wire, with or without tags)."""
    P = _window(max_tokens, window) // R
    return 4 * region_dwords(P, world, hidden)[-1]


def ctr_words(world, hidden, max_tokens, window=2048, **_geom):
    """Counter words of one instance (one per (window position, role))."""
    nrole = world + hidden // 256 // world + F_ROLES
    return (_window(max_tokens, window) // R) * nrole


# per-CU resources (VGPRs per SIMD lane incl. AGPRs, wave slots per SIMD, SGPRs
# per SIMD, LDS bytes per CU) for the persistent co-residency limit
_CU_RES = {"gfx950": (512, 8, 800, 160 * 1024)}
# a persistent launch uses at most min(2 per CU, half the co-resident capacity)
SAFE_CTAS_PER_CU = 2


def _meta(asm, key):
    import re

    m = re.search(rf"\.{key}:\s+(\d+)", asm)
    return int(m.group(1)) if m else 0


def ctas_per_cu(asm, shared, arch, num_warps=NW):
    """Co-resident CTAs per CU of a compiled kernel (VGPRs, SGPRs, wave slots,
    LDS)."""
    vfile, slots, sfile, lds = _CU_RES[arch.split(":")[0]]
    v = _meta(asm, "vgpr_count")
    a = _meta(asm, "agpr_count")
    sg = _meta(asm, "sgpr_count")
    valloc = -(-(-(-v // 4) * 4 + a) // 8) * 8
    w_simd = min(slots, vfile // max(valloc, 8), sfile // max(-(-sg // 16) * 16, 16))
    by_waves = w_simd // max(1, num_warps // 4)
    by_lds = lds // shared if shared else 1 << 30
    if v == 0:  # no register metadata: assume the worst
        by_waves = 1
    return max(1, min(by_waves, by_lds))


_FL = FastLauncher(two_shot_epoch_kernel)
_FL_NORM = FastLauncher(two_shot_epoch_norm_kernel)
_KEYS = itertools.count(1)


class TwoShotEpochInst(PushInst):
    """One two_shot_epoch instance on a region of the workspace.

    npc: None = the grid schedule; an int K = K persistent channels per role
    (xs = 2: the senders' channels pinned to their XCDs), clamped to the
    compiled variant's co-residency limit; wire: "c12" or "int8".
    """

    can_tail = False  # no masked-tail kernel: partial rows are zero-padded

    def __init__(
        self,
        rank,
        world,
        hidden,
        max_tokens,
        ipc,
        ctr,
        err,
        window=2048,
        npc=None,
        xs=0,
        drain=None,
        spin=None,
        diag=None,
        tag=False,
        site_box=None,
        fast=True,
        wire="c12",
        **geom,
    ):
        assert world in (2, 4, 8) and hidden % (world * 256) == 0, (
            f"two_shot_epoch: every owner must hold whole 256-column blocks "
            f"(hidden {hidden}, TP {world})"
        )
        assert not _check_geom(geom), f"two_shot_epoch: unknown geometry {geom}"
        if wire not in WIRES:
            raise ValueError(
                f"two_shot_epoch: wire {wire!r} (supported: {sorted(WIRES)})"
            )
        assert npc is None or int(npc) >= 1, "npc must be >= 1 (None: grid)"
        assert xs in (0, 2), "xs: 0 or 2 (the senders' channels pinned)"
        self.rank, self.world, self.hidden = rank, world, int(hidden)
        self._key = next(_KEYS)
        self.fast = bool(fast)
        self.npc = None if npc is None else int(npc)
        self.xs = int(xs)
        self.drain = None if drain is None else int(drain)
        self._eff_cache = {}
        self._limit = {}  # XS -> max CTAs of a persistent launch
        self._warned = set()
        self.max_tokens = int(max_tokens)
        self.window = _window(self.max_tokens, window)
        self.P = self.window // R
        self.red = self.hidden // 256 // world  # one reducer per owned block
        self.nrole = world + self.red + F_ROLES
        self.ag_off, self.rsf_off, self.agf_off, self.rel_off, tot = region_dwords(
            self.P, world, self.hidden
        )
        self.bytes_per_rank = 4 * tot
        assert self.bytes_per_rank == ipc_bytes(world, hidden, max_tokens, window)
        self.buf = ipc.take(self.bytes_per_rank)
        self.spin = spin if spin is not None else SpinBound()
        self.diag = diag
        if diag is not None:
            self.spin = no_trap(self.spin)
        self._spin_kw = spin_asm(self.spin, None if diag is None else diag.abort)
        if diag is not None:
            self._spin_kw.update(diag.kw())
        self.tag = bool(tag)
        assert not self.tag or diag is not None, (
            "call tags need the diagnostics (diag=)"
        )
        self.wire = WIRES[wire]
        # the residual add + RMSNorm: lossless wire only
        self.can_norm = wire == "c12"
        self.site_box = site_box if site_box is not None else [0]
        assert ctr.numel() >= self.P * self.nrole and err.numel() >= 1
        self.ctr, self.err = ctr[: self.P * self.nrole], err[:1]

    def _kw(self, persist=0, spec=None):
        kw = dict(
            ME=self.rank,
            NRANK=self.world,
            P=self.P,
            AG_OFF=self.ag_off,
            RSF_OFF=self.rsf_off,
            AGF_OFF=self.agf_off,
            REL_OFF=self.rel_off,
            NSEND=self.world,
            RED=self.red,
            F=F_ROLES,
            WIRE=self.wire,
            HIDDEN=self.hidden,
            PERSIST=persist,
            DRAIN=(1 - persist) if self.drain is None else self.drain,
            num_warps=NW,
            **self._spin_kw,
        )
        if self.tag:
            kw["TAG"] = 1
        if persist:
            kw["XS"] = spec[1]
        return kw

    def _args(self, x, out, m):
        return (x, out, self.buf.ptrs, self.ctr, self.err, m, self.site_box[0])

    def _args_norm(self, x, res, w, out, res_out, m, eps):
        b = self.buf
        return (
            x,
            res,
            w,
            out,
            res_out,
            b.ptrs,
            self.ctr,
            self.err,
            m,
            eps,
            self.site_box[0],
        )

    def _chan_limit(self, x, xs, norm=False):
        """Max CTAs of a persistent launch, from the compiled kernel: min(2 x
        CUs, half the co-resident capacity), so every channel stays resident
        with other work on half of the device."""
        lim = self._limit.get((xs, norm))
        if lim is not None:
            return lim
        if norm:
            w = torch.ones(self.hidden, dtype=x.dtype, device=x.device)
            ck = _FL_NORM.warm(
                (self.nrole, 1),
                self._args_norm(x, x, w, x, x, x.shape[0], 1e-6),
                self._kw(1, (1, xs)),
                key=(self._key, "norm", xs),
                fast=self.fast,
            )
        else:
            ck = _FL.warm(
                (self.nrole, 1),
                self._args(x, x, x.shape[0]),
                self._kw(1, (1, xs)),
                key=(self._key, xs),
                fast=self.fast,
            )
        props = torch.cuda.get_device_properties(x.device)
        ncu = props.multi_processor_count
        per_cu = ctas_per_cu(ck.asm["amdgcn"], ck.metadata.shared, props.gcnArchName)
        lim = min(SAFE_CTAS_PER_CU * ncu, ncu * per_cu // 2)
        self._limit[(xs, norm)] = lim
        return lim

    def _xs2(self, spec):
        """XS = 2 needs K * NSEND % 8 == 0 and NROLE * K % 8 == 0: the largest
        such K <= the requested one, else the XS = 0 mapping."""
        k = spec[0]
        for kk in range(k, 0, -1):
            if (kk * self.world) % 8 == 0 and (kk * self.nrole) % 8 == 0:
                return (kk, 2)
        return (k, 0)

    def _eff(self, m, x, norm=False):
        """(spec or None, grid) of a call with m rows, the persistent spec
        clamped to the co-residency limit (below one channel per role: the
        grid schedule; logged once)."""
        hit = self._eff_cache.get((m, norm))
        if hit is not None:
            return hit
        npos = (m + R - 1) // R
        spec = None if self.npc is None else (self.npc, self.xs)
        if spec is not None:
            lim = self._chan_limit(x, spec[1], norm)
            if self.nrole * spec[0] > lim:
                k = lim // self.nrole
                if (k, spec) not in self._warned:
                    self._warned.add((k, spec))
                    logger.warning(
                        "two_shot_epoch TP%d H%d: %d channels per role -> %s",
                        self.world,
                        self.hidden,
                        spec[0],
                        k if k >= 1 else "grid",
                    )
                spec = (k, spec[1]) if k >= 1 else None
        if spec is not None and spec[1] == 2:
            spec = self._xs2(spec)
        if spec is None:
            out = (None, (self.nrole, npos))
        else:
            # XS = 2 maps roles from the grid: always the full (NROLE, K) grid
            out = (spec, (self.nrole, spec[0] if spec[1] == 2 else min(spec[0], npos)))
        self._eff_cache[(m, norm)] = out
        return out

    def _launch(self, x, out, warmup=False, spec="auto"):
        m = x.shape[0]
        args = self._args(x, out, m)
        if warmup:  # compile the given spec (None = grid) without launching
            if spec == "auto":
                spec = self._eff(m, x)[0]
            return _FL.warm(
                (self.nrole, 1),
                args,
                self._kw(int(spec is not None), spec),
                key=(self._key, None if spec is None else spec[1]),
                fast=self.fast,
            )
        spec, grid = self._eff(m, x)
        _FL(
            grid,
            args,
            lambda: self._kw(int(spec is not None), spec),
            key=(self._key, None if spec is None else spec[1]),
            fast=self.fast,
        )

    def _launch_norm(self, x, res, w, eps, out, res_out, warmup=False, spec="auto"):
        m = x.shape[0]
        args = self._args_norm(x, res, w, out, res_out, m, eps)
        if warmup:  # compile the given spec (None = grid) without launching
            if spec == "auto":
                spec = self._eff(m, x, True)[0]
            return _FL_NORM.warm(
                (self.nrole, 1),
                args,
                self._kw(int(spec is not None), spec),
                key=(self._key, "norm", None if spec is None else spec[1]),
                fast=self.fast,
            )
        spec, grid = self._eff(m, x, True)
        _FL_NORM(
            grid,
            args,
            lambda: self._kw(int(spec is not None), spec),
            key=(self._key, "norm", None if spec is None else spec[1]),
            fast=self.fast,
        )

    def specs(self):
        """The schedule spec of this instance (None = grid)."""
        if self.npc is None:
            return {None}
        sp = (self.npc, self.xs)
        return {self._xs2(sp) if self.xs == 2 else sp}

    def precompile(self, x, ms=None, norm=False):
        """Compile the schedule's variant, and the grid / clamped variant a
        co-residency clamp may use, without launching (norm: also those of the
        residual add + RMSNorm kernel)."""
        out = torch.empty_like(x)
        w = torch.ones(self.hidden, dtype=x.dtype, device=x.device)
        for nm in (False, True) if norm and self.can_norm else (False,):

            def warm(sp, nm=nm):
                if nm:
                    self._launch_norm(x, x, w, 1e-6, out, out, warmup=True, spec=sp)
                else:
                    self._launch(x, out, warmup=True, spec=sp)

            for sp in self.specs():
                warm(sp)
                if sp is not None:
                    lim = self._chan_limit(x, sp[1], nm)
                    if self.nrole * sp[0] > lim:
                        u = (lim // self.nrole, sp[1])
                        if u[0] >= 1:
                            warm(u)

    def ar_add_rmsnorm(self, x, res, w, eps):
        """(out, res_out) of the all-reduce of x [m, H] + the residual add +
        RMSNorm with weight w [H] (bf16, contiguous)."""
        assert self.can_norm
        o2 = torch.empty((2,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        self._launch_norm(x, res, w, float(eps), o2[0], o2[1])
        return o2[0], o2[1]
