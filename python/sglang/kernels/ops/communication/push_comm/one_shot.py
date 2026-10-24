# SPDX-License-Identifier: Apache-2.0
"""one_shot: single-stage push all-reduce for small messages.

NSEND sender CTAs per row push every chunk of the row to every peer's slot
[row][src]; one finisher CTA per row polls all peers' chunks in one spin and
sums in rank order. 3 slots per row: in call n the finisher re-arms the slot of
call n - 1, which no peer writes again before its call n + 2.
one_shot_norm_kernel fuses the residual add + RMSNorm: the finisher holds the
whole row, so it stores res_out = bf16(f32(bf16(AR)) + f32(res)) and out =
bf16(res_out * rstd * w), the squares summed over its 4 waves."""

from __future__ import annotations

import itertools

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from . import asmgen
from .launch import FastLauncher, PushInst
from .primitives import FAILED_C, LIM, SEQ_C, I32Ptr
from .primitives import advance as _adv
from .primitives import cdiv, cptr, diag_report, fill_kernel, seq_tag
from .primitives import spin as _spin
from .primitives import store_nan_if_failed
from .protocol import SENT32, SpinBound, no_trap
from .two_shot import _bf_hi, _bf_lo, _pack2, m_class_rep


@gluon.jit
def one_shot_kernel(
    x_ptr,
    out_ptr,
    m,
    L0: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    CTR: gl.constexpr,
    ERR: gl.constexpr,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    MAXR: gl.constexpr,
    NSEND: gl.constexpr,
    FS_ASM: gl.constexpr,
    FS_CONS: gl.constexpr,
    FS_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
):
    _one_shot(
        x_ptr,
        out_ptr,
        m,
        0,
        0,
        x_ptr,
        x_ptr,
        out_ptr,
        0.0,
        L0,
        P0,
        P1,
        P2,
        P3,
        P4,
        P5,
        P6,
        P7,
        CTR,
        ERR,
        ME,
        W,
        H,
        MAXR,
        NSEND,
        FS_ASM,
        FS_CONS,
        FS_NOUT,
        DIAG,
        DIAG_BASE,
        INST,
        TRAP,
        0,
        0,
        0,
    )


@gluon.jit(do_not_specialize=["site"])
def one_shot_tag_kernel(
    x_ptr,
    out_ptr,
    m,
    site,
    L0: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    CTR: gl.constexpr,
    ERR: gl.constexpr,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    MAXR: gl.constexpr,
    NSEND: gl.constexpr,
    FS_ASM: gl.constexpr,
    FS_CONS: gl.constexpr,
    FS_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TAG: gl.constexpr = 1,
):
    """one_shot_kernel with the call-sequence tags (TAG) and the call's runtime
    site. A separate kernel: the extra kernel argument alone would change
    one_shot_kernel's kernel-argument loads."""
    _one_shot(
        x_ptr,
        out_ptr,
        m,
        site,
        0,
        x_ptr,
        x_ptr,
        out_ptr,
        0.0,
        L0,
        P0,
        P1,
        P2,
        P3,
        P4,
        P5,
        P6,
        P7,
        CTR,
        ERR,
        ME,
        W,
        H,
        MAXR,
        NSEND,
        FS_ASM,
        FS_CONS,
        FS_NOUT,
        DIAG,
        DIAG_BASE,
        INST,
        TRAP,
        TAG,
        0,
        0,
    )


@gluon.jit(do_not_specialize=["site"])
def one_shot_norm_kernel(
    x_ptr,
    res_ptr,
    w_ptr,
    out_ptr,
    resout_ptr,
    m,
    eps,
    site,
    L0: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    CTR: gl.constexpr,
    ERR: gl.constexpr,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    MAXR: gl.constexpr,
    NSEND: gl.constexpr,
    FS_ASM: gl.constexpr,
    FS_CONS: gl.constexpr,
    FS_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TAG: gl.constexpr = 0,
):
    """one_shot_kernel with the residual add + RMSNorm (TAG 1: + the call
    tags); a separate kernel so that its arguments do not change the plain
    kernel."""
    _one_shot(
        x_ptr,
        out_ptr,
        m,
        site,
        0,
        res_ptr,
        w_ptr,
        resout_ptr,
        eps,
        L0,
        P0,
        P1,
        P2,
        P3,
        P4,
        P5,
        P6,
        P7,
        CTR,
        ERR,
        ME,
        W,
        H,
        MAXR,
        NSEND,
        FS_ASM,
        FS_CONS,
        FS_NOUT,
        DIAG,
        DIAG_BASE,
        INST,
        TRAP,
        TAG,
        0,
        1,
    )


@gluon.jit(do_not_specialize=["n"])
def one_shot_tail_kernel(
    x_ptr,
    out_ptr,
    m,
    n,
    L0: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    CTR: gl.constexpr,
    ERR: gl.constexpr,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    MAXR: gl.constexpr,
    NSEND: gl.constexpr,
    FS_ASM: gl.constexpr,
    FS_CONS: gl.constexpr,
    FS_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TALN: gl.constexpr = 0,
):
    """one_shot_kernel over n elements as ceil(n / H) rows, the last partial
    (read as +0, never written). A separate kernel so that n does not change
    the whole-row kernel. TALN = 1 (n % 4 == 0) keeps 8-byte accesses."""
    _one_shot(
        x_ptr,
        out_ptr,
        m,
        0,
        n,
        x_ptr,
        x_ptr,
        out_ptr,
        0.0,
        L0,
        P0,
        P1,
        P2,
        P3,
        P4,
        P5,
        P6,
        P7,
        CTR,
        ERR,
        ME,
        W,
        H,
        MAXR,
        NSEND,
        FS_ASM,
        FS_CONS,
        FS_NOUT,
        DIAG,
        DIAG_BASE,
        INST,
        TRAP,
        0,
        1 + TALN,
        0,
    )


@gluon.jit(do_not_specialize=["site", "n"])
def one_shot_tail_tag_kernel(
    x_ptr,
    out_ptr,
    m,
    site,
    n,
    L0: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    CTR: gl.constexpr,
    ERR: gl.constexpr,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    MAXR: gl.constexpr,
    NSEND: gl.constexpr,
    FS_ASM: gl.constexpr,
    FS_CONS: gl.constexpr,
    FS_NOUT: gl.constexpr,
    DIAG: gl.constexpr = 1,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TAG: gl.constexpr = 1,
    TALN: gl.constexpr = 0,
):
    """one_shot_tail_kernel with the call-sequence tags (TAG) and the call's
    runtime site. A separate kernel: the extra kernel argument alone would
    change one_shot_tail_kernel's kernel-argument loads."""
    _one_shot(
        x_ptr,
        out_ptr,
        m,
        site,
        n,
        x_ptr,
        x_ptr,
        out_ptr,
        0.0,
        L0,
        P0,
        P1,
        P2,
        P3,
        P4,
        P5,
        P6,
        P7,
        CTR,
        ERR,
        ME,
        W,
        H,
        MAXR,
        NSEND,
        FS_ASM,
        FS_CONS,
        FS_NOUT,
        DIAG,
        DIAG_BASE,
        INST,
        TRAP,
        TAG,
        1 + TALN,
        0,
    )


@gluon.jit
def _one_shot(
    x_ptr,
    out_ptr,
    m,
    site,
    n,
    res_ptr,
    w_ptr,
    resout_ptr,
    eps,
    L0: gl.constexpr,
    P0: gl.constexpr,
    P1: gl.constexpr,
    P2: gl.constexpr,
    P3: gl.constexpr,
    P4: gl.constexpr,
    P5: gl.constexpr,
    P6: gl.constexpr,
    P7: gl.constexpr,
    CTR: gl.constexpr,
    ERR: gl.constexpr,
    ME: gl.constexpr,
    W: gl.constexpr,
    H: gl.constexpr,
    MAXR: gl.constexpr,
    NSEND: gl.constexpr,
    FS_ASM: gl.constexpr,
    FS_CONS: gl.constexpr,
    FS_NOUT: gl.constexpr,
    DIAG: gl.constexpr,
    DIAG_BASE: gl.constexpr,
    INST: gl.constexpr,
    TRAP: gl.constexpr,
    TAG: gl.constexpr,
    TAIL: gl.constexpr,
    NORM: gl.constexpr,
):
    gl.static_assert(TAIL == 0 or NORM == 0, "masked tail: plain all-reduce only")

    NROLE: gl.constexpr = NSEND + 1
    CPAD: gl.constexpr = 32  # counter words per (row, role) cell
    NSLOT: gl.constexpr = 3
    NC: gl.constexpr = H // 256  # 256-column chunks per row
    NPW: gl.constexpr = (NC + 3) // 4  # chunks per finisher wave
    # chunk iterations per sender wave
    NT: gl.constexpr = (NC + 4 * NSEND - 1) // (4 * NSEND)
    # call tags: int32 [slot, row, W] after the slots
    TAG_OFF: gl.constexpr = (3 * MAXR * W * H) // 2
    p = gl.program_id(0)
    xcd = p % 8
    q = p // 8
    role = q % NROLE
    li_ = q // NROLE
    row = xcd + 8 * li_
    if row < m:
        ctr = cptr(CTR, gl.int32) + (row * NROLE + role) * CPAD
        err = cptr(ERR, gl.int32)
        # the rotation counter is the CTA's first memory access (a scalar
        # load); advance() stores it only after every wave has loaded it
        cnt = gl.load(ctr)
        slot = cnt % NSLOT
        rowslot = slot * MAXR + row
        local = cptr(L0, gl.int16)
        pdst = (
            cptr(P0, gl.int16),
            cptr(P1, gl.int16),
            cptr(P2, gl.int16),
            cptr(P3, gl.int16),
            cptr(P4, gl.int16),
            cptr(P5, gl.int16),
            cptr(P6, gl.int16),
            cptr(P7, gl.int16),
        )
        if role < NSEND:
            LS: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
            ks = gl.expand_dims(gl.arange(0, 4, layout=gl.SliceLayout(1, LS)), 1)
            cs = gl.expand_dims(gl.arange(0, 256, layout=gl.SliceLayout(0, LS)), 0)
            xvs = ()
            for it in gl.static_range(NT):
                # chunk index; an idle wave reads a valid duplicate
                c = gl.minimum(role * 4 + ks + 4 * NSEND * it, NC - 1)
                if TAIL == 2:
                    # n % 4 == 0: one unpredicated 8-byte load; lanes past n
                    # read the last valid group and select +0
                    te = row * H + c * 256 + cs
                    tok = gl.max_constancy(te < n, [1, 4])
                    tec = gl.max_contiguous(
                        gl.multiple_of(gl.where(tok, te, te % 4 + n - 4), [1, 4]),
                        [1, 4],
                    )
                    xv = gl.load(x_ptr + tec).to(gl.int16, bitcast=True)
                    xv = gl.where(tok, xv, gl.zeros_like(xv))
                elif TAIL:  # the elements past n read as +0
                    xv = gl.load(
                        x_ptr.to(gl.pointer_type(gl.int16)) + row * H + c * 256 + cs,
                        mask=row * H + c * 256 + cs < n,
                        other=0,
                    )
                else:
                    xv = gl.load(
                        x_ptr.to(gl.pointer_type(gl.int16)) + row * H + c * 256 + cs
                    )
                xvs += (gl.where(xv == -32768, gl.zeros_like(xv), xv),)
            _adv(ctr, cnt, NSLOT)
            for it in gl.static_range(NT):
                c = role * 4 + ks + 4 * NSEND * it
                live = (c < NC) & (cs >= 0)
                for jj in gl.static_range(W):
                    if jj != ME:
                        pd = (
                            pdst[jj]
                            + ((rowslot * W + ME) * H)
                            + gl.minimum(c, NC - 1) * 256
                            + cs
                        )
                        pd = gl.max_contiguous(gl.multiple_of(pd, [1, 8]), [1, 4])
                        gl.store(pd, xvs[it], mask=live)
            if TAG:
                if role == 0:  # this rank's tag -> every rank's [slot, row, ME]
                    tg = seq_tag(m, site)
                    for jj in gl.static_range(W):
                        t32 = (
                            pdst[jj].to(gl.pointer_type(gl.int32))
                            + TAG_OFF
                            + rowslot * W
                            + ME
                        )
                        gl.store(
                            t32 + ks * 0 + cs * 0,
                            tg + ks * 0 + cs * 0,
                            mask=(ks == 0) & (cs == 0),
                        )
        else:
            LP: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
            wi = gl.expand_dims(gl.arange(0, 4, layout=gl.SliceLayout(1, LP)), 1)
            li = gl.expand_dims(gl.arange(0, 64, layout=gl.SliceLayout(0, LP)), 0)
            l32 = local.to(gl.pointer_type(gl.int32))
            x32 = x_ptr.to(gl.pointer_type(gl.int32)) + row * (H // 2)
            r32 = res_ptr.to(gl.pointer_type(gl.int32)) + row * (H // 2)
            w32 = w_ptr.to(gl.pointer_type(gl.int32))
            rmslot = ((slot + 2) % 3) * MAXR + row
            c2s = ()
            oks = ()
            own = ()
            rls = ()
            wls = ()
            pas = ()
            for i in gl.static_range(NPW):
                ci = wi + 4 * i + li * 0
                ok = ci < NC
                # int32 index of the lane's 4 columns
                c2 = gl.minimum(ci, NC - 1) * 128 + li * 2
                c2s += (c2,)
                oks += (ok,)
                if TAIL == 2:
                    # n % 4 == 0: the lane's 4 columns are all below n or all
                    # past it (past n: read the last valid group, select +0)
                    ok4 = 2 * c2 + 4 <= n - row * H
                    c2c = gl.where(ok4, c2, (n - row * H) // 2 - 2)
                    own += (
                        gl.where(ok4, gl.load(x32 + c2c), 0),
                        gl.where(ok4, gl.load(x32 + c2c + 1), 0),
                    )
                elif TAIL:
                    # whole bf16 pairs below n as int32; the pair that n
                    # splits (n odd) contributes its low int16 only
                    x16 = x_ptr.to(gl.pointer_type(gl.int16)) + row * H
                    nr = n - row * H
                    for hh in gl.static_range(2):
                        e = 2 * (c2 + hh)
                        v = gl.load(x32 + c2 + hh, mask=e + 1 < nr, other=0)
                        v0 = (
                            gl.load(x16 + e, mask=e + 1 == nr, other=0).to(gl.int32)
                            & 65535
                        )
                        own += (v | v0,)
                else:
                    own += (gl.load(x32 + c2), gl.load(x32 + c2 + 1))
                if NORM:
                    rls += (gl.load(r32 + c2), gl.load(r32 + c2 + 1))
                    wls += (gl.load(w32 + c2), gl.load(w32 + c2 + 1))
                for jj in gl.static_range(W):
                    if jj != ME:
                        # re-arm call n - 1's slot of this source / chunk
                        ra = l32 + ((rmslot * W + jj) * H) // 2 + c2
                        gl.store(ra, gl.full([4, 64], -2147450880, gl.int32, LP))
                        gl.store(ra + 1, gl.full([4, 64], -2147450880, gl.int32, LP))
                        pas += (l32 + ((rowslot * W + jj) * H) // 2 + c2,)
            if TAG:
                # re-arm call n - 1's tags (as the payload above), then poll
                # the W tags of this call: lane l <-> rank l % W
                gl.store(
                    l32 + TAG_OFF + rmslot * W + li + wi * 0,
                    gl.full([4, 64], -2147483648, gl.int32, LP),
                    mask=(wi == 0) & (li < W),
                )
                pas += (l32 + TAG_OFF + rowslot * W + li % W + wi * 0,)
            _adv(ctr, cnt, NSLOT)
            fp = _spin(pas, FS_NOUT, FS_ASM, FS_CONS)
            itf = fp[2 * NPW * (W - 1) + TAG]
            gl.store(err + (wi + li) * 0, itf, mask=itf >= LIM)
            if TAG:
                tg = seq_tag(m, site)
                tbad = (fp[2 * NPW * (W - 1)] != tg).to(gl.int32)
                itf = gl.where(
                    (gl.expand_dims(gl.max(tbad, 1), 1) > 0) & (itf < FAILED_C),
                    gl.full([4, 64], SEQ_C, gl.int32, LP),
                    itf,
                )
                gl.store(err + (wi + li) * 0, itf, mask=itf == SEQ_C)
            if DIAG:
                diag_report(DIAG_BASE, itf, wi + li, INST, 1, ME, role, row, cnt, TRAP)
            o32 = out_ptr.to(gl.pointer_type(gl.int32)) + row * (H // 2)
            ars = ()
            ss = gl.zeros([4, 64], gl.float32, LP)
            for i in gl.static_range(NPW):
                for h in gl.static_range(2):
                    lo = gl.zeros([4, 64], gl.float32, LP)
                    hi = gl.zeros([4, 64], gl.float32, LP)
                    for jj in gl.static_range(W):
                        if jj == ME:
                            v = own[2 * i + h]
                        else:
                            v = fp[2 * (i * (W - 1) + jj - (1 if jj > ME else 0)) + h]
                        if jj == 0:
                            lo = _bf_lo(v)
                            hi = _bf_hi(v)
                        else:
                            lo = lo + _bf_lo(v)
                            hi = hi + _bf_hi(v)
                    pk = _pack2(lo, hi)
                    # -0.0 -> +0.0 per bf16 half: -0.0 is the sentinel
                    pk = gl.where((pk & 65535) == 32768, pk & -65536, pk)
                    pk = gl.where((pk & -65536) == -2147483648, pk & 65535, pk)
                    ars += (pk,)
            if NORM:
                # res_out, the row's sum of squares over the 4 waves, out
                xrs = ()
                for i in gl.static_range(NPW):
                    for h in gl.static_range(2):
                        a = ars[2 * i + h]
                        r = rls[2 * i + h]
                        q0 = _bf_lo(a) + _bf_lo(r)
                        q1 = _bf_hi(a) + _bf_hi(r)
                        xrs += (q0, q1)
                        ss = ss + gl.where(oks[i], q0 * q0 + q1 * q1, gl.zeros_like(q0))
                inv = gl.rsqrt(gl.sum(gl.sum(ss, 1), 0) / H + eps)
                ro32 = resout_ptr.to(gl.pointer_type(gl.int32)) + row * (H // 2)
                for i in gl.static_range(NPW):
                    for h in gl.static_range(2):
                        q0 = xrs[2 * (2 * i + h)]
                        q1 = xrs[2 * (2 * i + h) + 1]
                        wv = wls[2 * i + h]
                        gl.store(ro32 + c2s[i] + h, _pack2(q0, q1), mask=oks[i])
                        gl.store(
                            o32 + c2s[i] + h,
                            _pack2(q0 * inv * _bf_lo(wv), q1 * inv * _bf_hi(wv)),
                            mask=oks[i],
                        )
            elif TAIL == 2:  # n % 4 == 0: whole 4-column groups, 8-byte stores
                nr = n - row * H
                for i in gl.static_range(NPW):
                    ok4 = oks[i] & (2 * c2s[i] + 4 <= nr)
                    gl.store(o32 + c2s[i], ars[2 * i], mask=ok4)
                    gl.store(o32 + c2s[i] + 1, ars[2 * i + 1], mask=ok4)
            elif TAIL:
                # whole pairs below n as int32, the split pair's low int16,
                # nothing past n
                o16 = out_ptr.to(gl.pointer_type(gl.int16)) + row * H
                nr = n - row * H
                for i in gl.static_range(NPW):
                    for hh in gl.static_range(2):
                        e = 2 * (c2s[i] + hh)
                        gl.store(
                            o32 + c2s[i] + hh,
                            ars[2 * i + hh],
                            mask=oks[i] & (e + 1 < nr),
                        )
                        gl.store(
                            o16 + e,
                            ars[2 * i + hh].to(gl.int16),
                            mask=oks[i] & (e + 1 == nr),
                        )
            else:
                for i in gl.static_range(NPW):
                    gl.store(o32 + c2s[i], ars[2 * i], mask=oks[i])
                    gl.store(o32 + c2s[i] + 1, ars[2 * i + 1], mask=oks[i])
            # a failed call returns NaN over exactly the n elements
            if DIAG and TAIL == 2:  # whole 4-column groups
                tnr = n - row * H
                for i in gl.static_range(NPW):
                    ok4 = oks[i] & (2 * c2s[i] + 4 <= tnr)
                    store_nan_if_failed(o32 + c2s[i], itf, ok4)
                    store_nan_if_failed(o32 + c2s[i] + 1, itf, ok4)
            elif DIAG and TAIL:  # the split pair as int16
                t16 = out_ptr.to(gl.pointer_type(gl.int16)) + row * H
                tnr = n - row * H
                for i in gl.static_range(NPW):
                    for h in gl.static_range(2):
                        e = 2 * (c2s[i] + h)
                        store_nan_if_failed(
                            o32 + c2s[i] + h, itf, oks[i] & (e + 1 < tnr)
                        )
                        gl.store(
                            t16 + e,
                            gl.full(itf.shape, 32704, gl.int16, itf.type.layout),
                            mask=oks[i] & (e + 1 == tnr) & (itf >= FAILED_C),
                        )
            elif DIAG:  # a failed call returns NaN, never a finite wrong sum
                for i in gl.static_range(NPW):
                    for h in gl.static_range(2):
                        store_nan_if_failed(o32 + c2s[i] + h, itf, oks[i])
                        if NORM:
                            store_nan_if_failed(
                                resout_ptr.to(gl.pointer_type(gl.int32))
                                + row * (H // 2)
                                + c2s[i]
                                + h,
                                itf,
                                oks[i],
                            )


# ---------------------------------------------------------------- host side
CPAD = 32


def ctr_words(world, hidden, max_tokens, NSEND=2):
    """Counter words of one instance: one CPAD-word cell per (row, role)."""
    return max_tokens * (int(NSEND) + 1) * CPAD


def ipc_bytes(world, hidden, max_tokens, tag=False, **_geom):
    """IPC bytes per rank of one instance (+ the call tags [3, rows, W])."""
    return 3 * max_tokens * world * hidden * 2 + (
        3 * max_tokens * world * 4 if tag else 0
    )


_FL = FastLauncher(one_shot_kernel)
_FL_TAG = FastLauncher(one_shot_tag_kernel)
_FL_NORM = FastLauncher(one_shot_norm_kernel)
_FL_TAIL = FastLauncher(one_shot_tail_kernel)
_FL_TAIL_TAG = FastLauncher(one_shot_tail_tag_kernel)
_KEYS = itertools.count(1)


class OneShotInst(PushInst):
    """One one_shot instance on a region of the workspace (same contract as
    two_shot.TwoShotInst). NSEND: sender CTAs per row; CPAD: counter words per
    (row, role) cell."""

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
        NSEND=2,
    ):
        assert world in (2, 4, 8)
        self.rank, self.world, self.hidden = rank, world, int(hidden)
        self.max_tokens = int(max_tokens)
        self._key = next(_KEYS)
        self.fast = bool(fast)
        assert self.hidden % 256 == 0
        npw = (self.hidden // 256 + 3) // 4
        assert npw * (world - 1) <= 32, (
            f"one_shot finisher polls {npw * (world - 1)} groups > 32"
        )
        self.cfg = dict(NSEND=int(NSEND))
        region = 3 * self.max_tokens * world * self.hidden  # int16
        self.tag = bool(tag)
        assert not self.tag or diag is not None, (
            "call tags need the diagnostics (diag=)"
        )
        self.site_box = site_box if site_box is not None else [0]
        ntag = 3 * self.max_tokens * world if self.tag else 0  # int32
        self.bytes_per_rank = region * 2 + ntag * 4
        assert self.bytes_per_rank == ipc_bytes(
            world, self.hidden, self.max_tokens, tag=self.tag
        )
        self.buf = ipc.take(self.bytes_per_rank)
        loc = self.buf.addrs[rank]
        fill_kernel[(cdiv(region // 2, 1024),)](
            I32Ptr(loc), region // 2, SENT32, BLOCK=1024, num_warps=4
        )
        if ntag:
            fill_kernel[(cdiv(ntag, 1024),)](
                I32Ptr(loc + region * 2), ntag, -2147483648, BLOCK=1024, num_warps=4
            )
        self.nrole = self.cfg["NSEND"] + 1
        self.ncell = self.max_tokens * self.nrole * CPAD
        assert ctr.numel() >= self.ncell and err.numel() >= 1
        self.ctr, self.err = ctr, err[:1]
        self.addrs = [int(a) for a in self.buf.addrs] + [0] * (8 - world)
        self.spin = spin if spin is not None else SpinBound()
        self.diag = diag
        if diag is not None:
            self.spin = no_trap(self.spin)
        asm, cons, nout, _ = asmgen.spin(
            "ag",
            n=npw * (world - 1),
            nq=int(self.tag),
            bound=self.spin,
            abort=None if diag is None else diag.abort,
        )
        self._spin = dict(FS_ASM=asm, FS_CONS=cons, FS_NOUT=nout)

    def _kw(self):
        a = self.addrs
        kw = dict(
            L0=a[self.rank],
            P0=a[0],
            P1=a[1],
            P2=a[2],
            P3=a[3],
            P4=a[4],
            P5=a[5],
            P6=a[6],
            P7=a[7],
            CTR=self.ctr.data_ptr(),
            ERR=self.err.data_ptr(),
            ME=self.rank,
            W=self.world,
            H=self.hidden,
            MAXR=self.max_tokens,
            NSEND=self.cfg["NSEND"],
            num_warps=4,
            **self._spin,
        )
        if self.diag is not None:
            kw.update(self.diag.kw())
        if self.tag:
            kw["TAG"] = 1
        return kw

    def _launch(self, x, out, warmup=False, m=None):
        m = x.shape[0] if m is None else m
        grid = (8 * cdiv(m, 8) * self.nrole,)
        if self.tag:
            fl, args = _FL_TAG, (x, out, m, self.site_box[0])
        else:
            fl, args = _FL, (x, out, m)
        if warmup:  # compile (no launch) and prime the fast-launch cache
            return fl.warm(grid, args, self._kw, key=(self._key, "ar"), fast=self.fast)
        fl(grid, args, self._kw, key=(self._key, "ar"), fast=self.fast)

    can_norm = True  # the residual add + RMSNorm kernel, every form

    def _launch_norm(self, x, res, w, eps, out, res_out, warmup=False, m=None):
        m = x.shape[0] if m is None else m
        grid = (8 * cdiv(m, 8) * self.nrole,)
        args = (x, res, w, out, res_out, m, eps, self.site_box[0])
        if warmup:
            return _FL_NORM.warm(
                grid, args, self._kw, key=(self._key, "norm"), fast=self.fast
            )
        _FL_NORM(grid, args, self._kw, key=(self._key, "norm"), fast=self.fast)

    def precompile(self, x, ms=(1,), tail=False, norm=False):
        """Compile every row-count specialization class of ms without
        launching; tail: also the masked-tail kernel; norm: also the residual
        add + RMSNorm kernel."""
        out = torch.empty_like(x)
        for mm in sorted({m_class_rep(v) for v in ms}):
            self._launch(x, out, warmup=True, m=mm)
            if norm:
                w = torch.ones(self.hidden, dtype=x.dtype, device=x.device)
                self._launch_norm(x, x, w, 1e-6, out, out, warmup=True, m=mm)
            if tail:  # both tail forms: TALN (n % 4 == 0) and the general one
                for nn in (x.numel(), x.numel() - 1):
                    self._launch_tail(x.view(-1), out.view(-1), nn, warmup=True, m=mm)

    def _launch_tail(self, x, out, n, warmup=False, m=None):
        m = -(-n // self.hidden) if m is None else m
        grid = (8 * cdiv(m, 8) * self.nrole,)
        aln = int(n % 4 == 0)  # TALN: the 8-byte tail form

        def kw():
            return dict(self._kw(), **({"TALN": 1} if aln else {}))

        if self.tag:  # call tags: one_shot_tail_tag_kernel, with the site
            fl, args = _FL_TAIL_TAG, (x, out, m, self.site_box[0], n)
        else:
            fl, args = _FL_TAIL, (x, out, m, n)
        if warmup:
            return fl.warm(grid, args, kw, key=(self._key, "tail", aln), fast=self.fast)
        fl(grid, args, kw, key=(self._key, "tail", aln), fast=self.fast)

    def plain_launcher(self, n, tail, tspec):
        """A closure x -> all-reduce(x) for messages of n elements (tail: the
        masked-tail entry) on the variant the warm-up compiled, or None."""
        if not self.fast:
            return None
        m = -(-n // self.hidden)
        grid = (8 * cdiv(m, 8) * self.nrole,)
        d = torch.empty(0, dtype=torch.bfloat16)
        site = self.site_box
        if tail:
            aln = int(n % 4 == 0)
            key = (self._key, "tail", aln)
            if self.tag:
                lf = _FL_TAIL_TAG.bound(grid, (d, d, m, site[0], n), key, tspec)
                if lf is None:
                    return None

                def run(x):
                    out = torch.empty(n, dtype=x.dtype, device=x.device)
                    lf(x, out, m, site[0], n)
                    return out

                return run
            lf = _FL_TAIL.bound(grid, (d, d, m, n), key, tspec)
            if lf is None:
                return None

            def run(x):
                out = torch.empty(n, dtype=x.dtype, device=x.device)
                lf(x, out, m, n)
                return out

            return run
        key = (self._key, "ar")
        if self.tag:
            lf = _FL_TAG.bound(grid, (d, d, m, site[0]), key, tspec)
            if lf is None:
                return None

            def run(x):
                out = torch.empty_like(x)
                lf(x, out, m, site[0])
                return out

            return run
        lf = _FL.bound(grid, (d, d, m), key, tspec)
        if lf is None:
            return None

        def run(x):
            out = torch.empty_like(x)
            lf(x, out, m)
            return out

        return run

    def ar_add_rmsnorm(self, x, res, w, eps):
        """(out, res_out) of the all-reduce of x [m, H] + the residual add +
        RMSNorm with weight w [H] (bf16, contiguous)."""
        o2 = torch.empty((2,) + tuple(x.shape), dtype=x.dtype, device=x.device)
        self._launch_norm(x, res, w, float(eps), o2[0], o2[1])
        return o2[0], o2[1]
