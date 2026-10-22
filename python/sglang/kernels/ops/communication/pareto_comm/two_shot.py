# SPDX-License-Identifier: Apache-2.0
"""two_shot: two-stage push all-reduce (reduce-scatter to the owner, then
all-gather), one launch per call, data as arrival signal.

Per row: W sender CTAs (one per owner), RED reducer CTAs (256 owned columns
each), F finisher CTAs, consecutive on the XCD of row % 8 so that every spin
waits only for smaller CTA ids. 3 slots per row; the senders re-arm the slot of
call n - 1, which no peer writes again before its call n + 2."""

from __future__ import annotations

import itertools

import torch
from triton.experimental import gluon
from triton.experimental.gluon import language as gl

from . import asmgen
from .launch import FastLauncher, PushInst
from .primitives import (
    FAILED_C,
    LIM,
    SEQ_C,
    I32Ptr,
)
from .primitives import advance as _adv
from .primitives import (
    cdiv,
    cptr,
    diag_report,
    fill_kernel,
    nan_if_failed,
)
from .primitives import now as _now
from .primitives import (
    seq_tag,
)
from .primitives import spin as _spin
from .protocol import SENT32, SpinBound, no_trap


@gluon.jit
def _bf_lo(v):
    return (v << 16).to(gl.float32, bitcast=True)


@gluon.jit
def _bf_hi(v):
    return (v & -65536).to(gl.float32, bitcast=True)


@gluon.jit
def _pack2(a, b):
    ai = a.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 65535
    bi = b.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32)
    return ai | (bi << 16)


@gluon.jit
def _now_slp(x):
    return gl.inline_asm_elementwise(
        "s_sleep 1\ns_memrealtime $0\ns_waitcnt lgkmcnt(0)",
        "=s,s,~{memory}",
        [x],
        dtype=gl.int64,
        is_pure=False,
        pack=1,
    )


@gluon.jit
def _tload(x_ptr, row, xc, n, HID: gl.constexpr, TAIL: gl.constexpr):
    """Masked-tail x load (elements past n read as +0). TAIL == 2 (n % 4 == 0):
    one unpredicated 8-byte load per lane; a lane past n reads the last valid
    group and selects +0."""
    if TAIL == 2:
        e = row * HID + xc
        ok = gl.max_constancy(e < n, [1, 4])
        ec = gl.max_contiguous(
            gl.multiple_of(gl.where(ok, e, e % 4 + n - 4), [1, 4]), [1, 4]
        )
        v = gl.load(x_ptr + ec).to(gl.int16, bitcast=True)
        return gl.where(ok, v, gl.zeros_like(v))
    else:
        return gl.load(
            x_ptr.to(gl.pointer_type(gl.int16)) + row * HID + xc,
            mask=row * HID + xc < n,
            other=0,
        )


@gluon.jit
def _drain(r, LO: gl.constexpr, N: gl.constexpr, CONS: gl.constexpr):
    """PIPE: keep both poll batches' registers (outputs LO..LO+N-1) alive up to
    here, then wait for the batch still in flight."""
    ks = ()
    for i in gl.static_range(N):
        ks += (r[LO + i],)
    return gl.inline_asm_elementwise(
        "s_waitcnt vmcnt(0)", CONS, ks, dtype=gl.int32, is_pure=False, pack=1
    )


@gluon.jit(do_not_specialize=["site"])
def two_shot_kernel(
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
    BLK: gl.constexpr,
    AG_OFF: gl.constexpr,
    SSQ_OFF: gl.constexpr,
    MAXR: gl.constexpr,
    F: gl.constexpr,
    PIPE: gl.constexpr,
    RS_ASM: gl.constexpr,
    RS_CONS: gl.constexpr,
    RS_NOUT: gl.constexpr,
    RS_KLO: gl.constexpr,
    RS_KN: gl.constexpr,
    RS_KCONS: gl.constexpr,
    AG_ASM: gl.constexpr,
    AG_CONS: gl.constexpr,
    AG_NOUT: gl.constexpr,
    AG_KLO: gl.constexpr,
    AG_KN: gl.constexpr,
    AG_KCONS: gl.constexpr,
    SDLY: gl.constexpr = 0,
    SDG: gl.constexpr = 8,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TAG: gl.constexpr = 0,
):
    OW: gl.constexpr = H // W  # owned columns per rank
    RED: gl.constexpr = (OW + 255) // 256  # reducer CTAs per row
    NROLE: gl.constexpr = W + RED + F
    p = gl.program_id(0)
    # row r on XCD r % 8, its CTAs consecutive (SEND < REDUCE < FINISH) in
    # that XCD's id class: every spin waits only for smaller ids
    xcd = p % 8
    q = p // 8
    role = q % NROLE
    li = q // NROLE
    # (xcd + 8) % 8 == xcd, but this form compiles to shorter scalar code
    row = (xcd + 8) % 8 + 8 * li
    if row < m:
        _body(
            x_ptr,
            out_ptr,
            m,
            site,
            row,
            role,
            0,
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
            BLK,
            AG_OFF,
            SSQ_OFF,
            MAXR,
            F,
            NROLE,
            PIPE,
            RS_ASM,
            RS_CONS,
            RS_NOUT,
            RS_KLO,
            RS_KN,
            RS_KCONS,
            AG_ASM,
            AG_CONS,
            AG_NOUT,
            AG_KLO,
            AG_KN,
            AG_KCONS,
            SDLY,
            SDG,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            TAG,
            0,
        )


@gluon.jit(do_not_specialize=["site", "n"])
def two_shot_tail_kernel(
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
    BLK: gl.constexpr,
    AG_OFF: gl.constexpr,
    SSQ_OFF: gl.constexpr,
    MAXR: gl.constexpr,
    F: gl.constexpr,
    PIPE: gl.constexpr,
    RS_ASM: gl.constexpr,
    RS_CONS: gl.constexpr,
    RS_NOUT: gl.constexpr,
    RS_KLO: gl.constexpr,
    RS_KN: gl.constexpr,
    RS_KCONS: gl.constexpr,
    AG_ASM: gl.constexpr,
    AG_CONS: gl.constexpr,
    AG_NOUT: gl.constexpr,
    AG_KLO: gl.constexpr,
    AG_KN: gl.constexpr,
    AG_KCONS: gl.constexpr,
    SDLY: gl.constexpr = 0,
    SDG: gl.constexpr = 8,
    DIAG: gl.constexpr = 0,
    DIAG_BASE: gl.constexpr = 0,
    INST: gl.constexpr = 0,
    TRAP: gl.constexpr = 1,
    TAG: gl.constexpr = 0,
    TALN: gl.constexpr = 0,
):
    """two_shot_kernel over n elements as ceil(n / H) rows, the last partial
    (read as +0, never written). A separate kernel so that n does not change
    the whole-row kernel. TALN = 1 (n % 4 == 0) keeps 8-byte accesses."""
    OW: gl.constexpr = H // W
    RED: gl.constexpr = (OW + 255) // 256
    NROLE: gl.constexpr = W + RED + F
    p = gl.program_id(0)
    xcd = p % 8
    q = p // 8
    role = q % NROLE
    li = q // NROLE
    row = (xcd % 8) + 8 * li
    if row < m:
        _body(
            x_ptr,
            out_ptr,
            m,
            site,
            row,
            role,
            n,
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
            BLK,
            AG_OFF,
            SSQ_OFF,
            MAXR,
            F,
            NROLE,
            PIPE,
            RS_ASM,
            RS_CONS,
            RS_NOUT,
            RS_KLO,
            RS_KN,
            RS_KCONS,
            AG_ASM,
            AG_CONS,
            AG_NOUT,
            AG_KLO,
            AG_KN,
            AG_KCONS,
            SDLY,
            SDG,
            DIAG,
            DIAG_BASE,
            INST,
            TRAP,
            TAG,
            1 + TALN,
        )


@gluon.jit
def _body(
    x_ptr,
    out_ptr,
    m,
    site,
    row,
    role,
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
    BLK: gl.constexpr,
    AG_OFF: gl.constexpr,
    SSQ_OFF: gl.constexpr,
    MAXR: gl.constexpr,
    F: gl.constexpr,
    NROLE: gl.constexpr,
    PIPE: gl.constexpr,
    RS_ASM: gl.constexpr,
    RS_CONS: gl.constexpr,
    RS_NOUT: gl.constexpr,
    RS_KLO: gl.constexpr,
    RS_KN: gl.constexpr,
    RS_KCONS: gl.constexpr,
    AG_ASM: gl.constexpr,
    AG_CONS: gl.constexpr,
    AG_NOUT: gl.constexpr,
    AG_KLO: gl.constexpr,
    AG_KN: gl.constexpr,
    AG_KCONS: gl.constexpr,
    SDLY: gl.constexpr,
    SDG: gl.constexpr,
    DIAG: gl.constexpr,
    DIAG_BASE: gl.constexpr,
    INST: gl.constexpr,
    TRAP: gl.constexpr,
    TAG: gl.constexpr,
    TAIL: gl.constexpr,
):
    HID: gl.constexpr = H
    NSEND: gl.constexpr = W  # one sender CTA per owner
    NSLOT: gl.constexpr = 3  # slots per row, re-armed by the senders
    CPAD: gl.constexpr = 32  # counter words per (row, role) cell
    NK: gl.constexpr = H // BLK // W  # owned blocks per rank
    OW: gl.constexpr = NK * BLK  # owned columns per rank
    RED: gl.constexpr = (OW + 255) // 256  # 256-column pieces of the owned columns
    WHOLE: gl.constexpr = BLK == 256  # piece == block (no partial piece)
    NPART: gl.constexpr = RED * W  # reserved record entries per row
    # record: NPART entries reserved for a fused epilogue, then the W call tags
    # (TROOM: in a second 64-entry half)
    TROOM: gl.constexpr = TAG and (NPART + W > 64)
    SROW: gl.constexpr = (128 if TROOM else 64) if TAG else (32 if NPART <= 32 else 64)
    NC: gl.constexpr = H // 256  # 256-column chunks per row (finishers)
    CPF: gl.constexpr = NC // F  # chunks per finisher
    NPW: gl.constexpr = (CPF + 3) // 4  # chunks per finisher wave
    FULLW: gl.constexpr = (CPF % 4 == 0) and WHOLE and (NPART == NC)
    NT: gl.constexpr = (RED + 3) // 4  # sender piece iterations per owner
    LW: gl.constexpr = 1 if W == 2 else (2 if W == 4 else 3)
    SLIVE: gl.constexpr = NPART + W if TAG else NPART
    if TAG:
        tg = seq_tag(m, site)
    ctr = cptr(CTR, gl.int32) + (row * NROLE + role) * CPAD
    err = cptr(ERR, gl.int32)
    # the rotation counter: the CTA's first memory access, a scalar load
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
        # SEND: every x load before the first remote store (in-order vmcnt: a
        # load behind a remote store waits for its acknowledgement)
        LS: gl.constexpr = gl.BlockedLayout([1, 4], [1, 64], [4, 1], [1, 0])
        ks = gl.expand_dims(gl.arange(0, 4, layout=gl.SliceLayout(1, LS)), 1)
        cs = gl.expand_dims(gl.arange(0, 256, layout=gl.SliceLayout(0, LS)), 0)
        xvs = ()
        mss = ()
        for it in gl.static_range(NT):
            for u in gl.static_range(W // NSEND):
                if WHOLE:
                    ksc = gl.minimum(ks + 4 * it, RED - 1)
                    if TAIL:  # the elements past n read as +0
                        xc = (ksc * W + role + u * NSEND) * 256 + cs
                        xv = _tload(x_ptr, row, xc, n, HID, TAIL)
                    else:
                        xv = gl.load(
                            x_ptr.to(gl.pointer_type(gl.int16))
                            + row * HID
                            + (ksc * W + role + u * NSEND) * 256
                            + cs
                        )
                elif TAIL:
                    o4 = gl.minimum((ks + 4 * it) * 256 + (cs // 4) * 4, OW - 4)
                    xc = ((o4 // BLK) * W + role + u * NSEND) * BLK + o4 % BLK + cs % 4
                    xv = _tload(x_ptr, row, xc, n, HID, TAIL)
                else:
                    # a mask constant over each lane's 4 columns keeps the 8-byte
                    # loads / stores; a clamp would split them into 2-byte ones
                    o4 = gl.minimum((ks + 4 * it) * 256 + (cs // 4) * 4, OW - 4)
                    xv = gl.load(
                        x_ptr.to(gl.pointer_type(gl.int16))
                        + row * HID
                        + ((o4 // BLK) * W + role + u * NSEND) * BLK
                        + o4 % BLK
                        + cs % 4
                    )
                xvs += (gl.where(xv == -32768, gl.zeros_like(xv), xv),)
            if WHOLE or OW % 256 == 0:
                mss += ((ks + 4 * it + cs * 0) < RED,)
            else:
                mss += ((ks + 4 * it) * 256 + (cs // 4) * 4 < OW,)
        _adv(ctr, cnt, NSLOT)
        if SDLY > 0:
            # staggered reduce-scatter: row group row // SDG starts SDLY ticks
            # (10 ns) after the previous one
            t0 = _now(cnt)
            tq = t0 + (row // SDG) * SDLY
            tc = t0
            while tc < tq:
                tc = _now_slp(cnt)
        for it in gl.static_range(NT):
            for u in gl.static_range(W // NSEND):
                # the destination among W immediates (role is wave-uniform)
                j = role + u * NSEND
                dst = pdst[0]
                for jj in gl.static_range(1, W):
                    dst = gl.where(j == jj, pdst[jj], dst)
                if WHOLE:
                    ksc = gl.minimum(ks + 4 * it, RED - 1)
                    pd = dst + ((rowslot * W + ME) * NK + ksc) * 256 + cs
                else:
                    pd = dst + (rowslot * W + ME) * OW + (ks + 4 * it) * 256 + cs
                if not FULLW:
                    # peer bases carry no axis info: without the hint the store
                    # splits into 4 scalar i16 stores
                    pd = gl.max_contiguous(gl.multiple_of(pd, [1, 8]), [1, 4])
                gl.store(pd, xvs[it * (W // NSEND) + u], mask=mss[it])
        if TAG:
            if role == 0:  # this rank's tag into every rank's record (slot, row)
                for jj in gl.static_range(W):
                    t32 = (
                        pdst[jj].to(gl.pointer_type(gl.int32))
                        + SSQ_OFF
                        + rowslot * SROW
                        + NPART
                        + ME
                    )
                    gl.store(
                        t32 + ks * 0 + cs * 0,
                        tg + ks * 0 + cs * 0,
                        mask=(ks == 0) & (cs == 0),
                    )
        # re-arm the slot of call n - 1: a peer writes it again only in its
        # call n + 2, which needs this rank's call n + 1, after this kernel
        rmslot = ((slot + 2) % 3) * MAXR + row
        for it in gl.static_range(NT):
            for u in gl.static_range(W // NSEND):
                j = role + u * NSEND
                if WHOLE:
                    ksc = gl.minimum(ks + 4 * it, RED - 1)
                    gl.store(
                        local + ((rmslot * W + j) * NK + ksc) * 256 + cs,
                        gl.full([4, 256], -32768, gl.int16, LS),
                        mask=mss[it],
                    )
                    gl.store(
                        local + AG_OFF + ((rmslot * W + j) * NK + ksc) * 256 + cs,
                        gl.full([4, 256], -32768, gl.int16, LS),
                        mask=mss[it],
                    )
                else:
                    oo = (ks + 4 * it) * 256 + cs
                    pq = gl.max_contiguous(
                        gl.multiple_of(local + (rmslot * W + j) * OW + oo, [1, 8]),
                        [1, 4],
                    )
                    gl.store(pq, gl.full([4, 256], -32768, gl.int16, LS), mask=mss[it])
                    pq = gl.max_contiguous(
                        gl.multiple_of(
                            local + AG_OFF + (rmslot * W + j) * OW + oo, [1, 8]
                        ),
                        [1, 4],
                    )
                    gl.store(pq, gl.full([4, 256], -32768, gl.int16, LS), mask=mss[it])
        if role == 0:
            q32 = local.to(gl.pointer_type(gl.int32)) + SSQ_OFF + rmslot * SROW
            gl.store(
                q32 + ks * 64 + cs,
                gl.full([4, 256], -2147483648, gl.int32, LS),
                mask=(ks * 64 + cs) < SLIVE,
            )
    elif role < NSEND + RED:
        # ---- REDUCE piece k (owned positions k * 256 .. k * 256 + 255), [4, 64]
        # view, one bf16 per lane
        k = role - NSEND
        LR: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
        wr = gl.expand_dims(gl.arange(0, 4, layout=gl.SliceLayout(1, LR)), 1)
        lr = gl.expand_dims(gl.arange(0, 64, layout=gl.SliceLayout(0, LR)), 0)
        cr = wr * 64 + lr
        if not WHOLE:
            # a partial piece polls / reads a valid duplicate
            ocr = gl.minimum(k * 256 + cr, OW - 1)
            live = k * 256 + cr < OW
        if WHOLE:
            pr = local + ((rowslot * W) * NK + k) * 256 + cr
        else:
            pr = local + (rowslot * W) * OW + ocr
        prs = ()
        for src in gl.static_range(W):
            prs += (pr + src * OW,)
        _adv(ctr, cnt, NSLOT)
        rp = _spin(prs, RS_NOUT, RS_ASM, RS_CONS)
        itv = rp[W]
        gl.store(err + cr * 0, itv, mask=itv >= LIM)
        if DIAG:
            diag_report(DIAG_BASE, itv, cr, INST, 2, ME, role, row, cnt, TRAP)
        # fp32 sum in fixed rank order, rounded once; -0.0 (the sentinel) -> +0.0
        s = rp[0].to(gl.int16).to(gl.bfloat16, bitcast=True).to(gl.float32)
        for src in gl.static_range(1, W):
            s = s + rp[src].to(gl.int16).to(gl.bfloat16, bitcast=True).to(gl.float32)
        si = s.to(gl.bfloat16).to(gl.int16, bitcast=True)
        si = gl.where(si == -32768, gl.zeros_like(si), si)
        if DIAG:
            # a failed reduce pushes NaN, never a partial sum, so every rank's
            # finisher outputs NaN for these columns
            si = gl.where(itv >= FAILED_C, gl.full(si.shape, 32704, gl.int16, LR), si)
        for jj in gl.static_range(W):
            if WHOLE:
                gl.store(
                    pdst[jj] + AG_OFF + ((rowslot * W + ME) * NK + k) * 256 + cr, si
                )
            else:
                gl.store(
                    pdst[jj] + AG_OFF + (rowslot * W + ME) * OW + ocr, si, mask=live
                )
        if PIPE:
            _drain(rp, RS_KLO, RS_KN, RS_KCONS)
    else:
        # ---- FINISH: chunks f * CPF .. f * CPF + CPF - 1 (256 columns each);
        # wave w owns NPW chunks (w * NPW + i), lane <-> 4 columns of each; one
        # per-wave spin polls all NPW chunks (and the call tags) together
        f = role - NSEND - RED
        LP: gl.constexpr = gl.BlockedLayout([1, 1], [1, 64], [4, 1], [1, 0])
        wi = gl.expand_dims(gl.arange(0, 4, layout=gl.SliceLayout(1, LP)), 1)
        li = gl.expand_dims(gl.arange(0, 64, layout=gl.SliceLayout(0, LP)), 0)
        ag32 = (local + AG_OFF).to(gl.pointer_type(gl.int32))
        ssq32 = local.to(gl.pointer_type(gl.int32)) + SSQ_OFF
        pas = ()
        c2s = ()
        oks = ()
        for i in gl.static_range(NPW):
            if FULLW:
                b = f * CPF + wi * NPW + i + li * 0
                kb = b >> LW
                jb = b & (W - 1)
                c2 = b * 128 + li * 2
                ok = li >= 0
            else:
                ci = wi * NPW + i + li * 0
                ok = ci < CPF
                # an idle wave slot polls a valid duplicate
                gch = f * CPF + gl.minimum(ci, CPF - 1)
                gc = gch * 256 + li * 4  # global column of this lane's 4 columns
                bb = gc // BLK
                jb = bb % W
                ob = (bb // W) * BLK + gc % BLK  # owned position in owner jb's region
                c2 = gc // 2
                pa = ((rowslot * W + jb) * OW + ob) // 2
            if FULLW:
                pas += (ag32 + ((rowslot * W + jb) * NK + kb) * 128 + li * 2,)
            else:
                pas += (ag32 + pa,)
            c2s += (c2,)
            oks += (ok,)
        if SROW == 32:
            pq0 = ssq32 + rowslot * SROW + (li & 31) + wi * 0
        elif TAG:
            # lane l polls the tag of rank l % W
            pq0 = ssq32 + rowslot * SROW + NPART + li % W + wi * 0
        else:
            pq0 = ssq32 + rowslot * SROW + li + wi * 0
        NQ: gl.constexpr = 1 if TAG else 0
        _adv(ctr, cnt, NSLOT)
        if NQ == 1:
            fp = _spin(pas + (pq0,), AG_NOUT, AG_ASM, AG_CONS)
            qv0 = fp[2 * NPW]
        else:
            fp = _spin(pas, AG_NOUT, AG_ASM, AG_CONS)
            qv0 = fp[0]
        ds = ()
        for kk in gl.static_range(2 * NPW):
            ds += (fp[kk],)
        itf = fp[2 * NPW + NQ]
        gl.store(err + (wi + li) * 0, itf, mask=itf >= LIM)
        if TAG:
            # a tag that differs from this rank's: the ranks run different
            # calls, the wave fails (SEQ_MISMATCH)
            if TROOM:  # lane l polled the tag of rank l % W
                tbad = (qv0 != tg).to(gl.int32)
            else:
                tbad = ((li >= NPART) & (li < NPART + W) & (qv0 != tg)).to(gl.int32)
            itf = gl.where(
                (gl.expand_dims(gl.max(tbad, 1), 1) > 0) & (itf < FAILED_C),
                gl.full([4, 64], SEQ_C, gl.int32, LP),
                itf,
            )
            gl.store(err + (wi + li) * 0, itf, mask=itf == SEQ_C)
        if DIAG:
            diag_report(DIAG_BASE, itf, wi + li, INST, 2, ME, role, row, cnt, TRAP)
            nds = ()
            for kk in gl.static_range(2 * NPW):
                nds += (nan_if_failed(ds[kk], itf),)
            ds = nds
            qv0 = nan_if_failed(qv0, itf)
        o32 = out_ptr.to(gl.pointer_type(gl.int32)) + row * (HID // 2)
        if TAIL == 2:
            # n % 4 == 0: a lane's 4 columns are all below n or all past it,
            # so its two words go out as one 8-byte store
            nr = n - row * HID
            for i in gl.static_range(NPW):
                ok4 = oks[i] & (2 * c2s[i] + 4 <= nr)
                gl.store(o32 + c2s[i], ds[2 * i], mask=ok4)
                gl.store(o32 + c2s[i] + 1, ds[2 * i + 1], mask=ok4)
        elif TAIL:
            # whole pairs below n as int32, the pair that n splits (n odd) as
            # its low int16, nothing past n
            nr = n - row * HID
            o16 = out_ptr.to(gl.pointer_type(gl.int16)) + row * HID
            for i in gl.static_range(NPW):
                for hh in gl.static_range(2):
                    e = 2 * (c2s[i] + hh)
                    gl.store(
                        o32 + c2s[i] + hh, ds[2 * i + hh], mask=oks[i] & (e + 1 < nr)
                    )
                    gl.store(
                        o16 + e,
                        ds[2 * i + hh].to(gl.int16),
                        mask=oks[i] & (e + 1 == nr),
                    )
        else:
            for i in gl.static_range(NPW):
                if FULLW:
                    gl.store(o32 + c2s[i], ds[2 * i])
                    gl.store(o32 + c2s[i] + 1, ds[2 * i + 1])
                else:
                    gl.store(o32 + c2s[i], ds[2 * i], mask=oks[i])
                    gl.store(o32 + c2s[i] + 1, ds[2 * i + 1], mask=oks[i])
        if PIPE:
            _drain(fp, AG_KLO, AG_KN, AG_KCONS)


@gluon.jit
def fill_ssq_kernel(p_ptr, n, srow, live, BLOCK: gl.constexpr):
    """Entries 0..live-1 of every srow-entry record start at the fp32 -0.0
    sentinel, the rest are a permanent +0.0."""
    L: gl.constexpr = gl.BlockedLayout([4], [64], [4], [0])
    offs = gl.program_id(0) * BLOCK + gl.arange(0, BLOCK, layout=L)
    v = gl.where(
        (offs % srow) < live,
        gl.full([BLOCK], -2147483648, gl.int32, L),
        gl.zeros([BLOCK], gl.int32, L),
    )
    gl.store(p_ptr + offs, v, mask=offs < n)


# ---------------------------------------------------------------- host side
def block_size(hidden, world):
    """BLK = first of 256 / 128 / 64 with hidden % (world * BLK) == 0; None if
    none fits."""
    for blk in (256, 128, 64):
        if hidden % (world * blk) == 0:
            return blk
    return None


def shape(hidden, world, tag=False):
    """Derived per-(H, W) quantities, host side (mirrors the kernel's constexpr
    derivation)."""
    blk = block_size(hidden, world)
    assert blk is not None and hidden % 256 == 0, (
        f"hidden {hidden} / TP {world}: no block size"
    )
    nk = hidden // blk // world
    ow = nk * blk
    red = cdiv(ow, 256)
    npart = red * world
    assert npart <= 64, f"{npart} partial sums per row > 64"
    # call tags at entries NPART .. NPART + W - 1 of a 64-entry record; with 64
    # reserved entries (TROOM) of a 128-entry one
    troom = tag and npart + world > 64
    return dict(
        BLK=blk,
        NB=hidden // blk,
        NK=nk,
        OW=ow,
        RED=red,
        NPART=npart,
        SROW=(128 if troom else 64) if tag else (32 if npart <= 32 else 64),
        SLIVE=npart + (world if tag else 0),
        NC=hidden // 256,
        TROOM=int(troom),
    )


# F: finisher CTAs per row; PIPE: two staggered poll batches; SDLY / SDG:
# staggered reduce-scatter, row group row // SDG starts SDLY x 10 ns later
DEFAULT = dict(F=4, PIPE=0, SDLY=0, SDG=8)
# fixed: one sender CTA per owner, 3 slots per row re-armed by the senders
_FIXED = dict(RA=1, NSLOT=3)
NSLOT, CPAD, PSLP = 3, 32, 4


def spin_kw(world, npw, cfg, bound=None, abort=None, tag=False):
    """The RS / AG spin constexprs ({RS,AG}_{ASM, CONS, NOUT, KLO, KN,
    KCONS}) of one geometry."""
    out = {}
    pipe = cfg["PIPE"]
    rs = ("rs_pipe", dict(nsrc=world, pslp=PSLP)) if pipe else ("rs", dict(nsrc=world))
    nq = 1 if tag else 0
    ag = (
        ("ag_pipe", dict(n=npw, nq=nq, pslp=PSLP))
        if pipe
        else ("ag", dict(n=npw, nq=nq))
    )
    for name, (kind, params) in (("RS", rs), ("AG", ag)):
        asm, cons, nout, _ = asmgen.spin(kind, bound=bound, abort=abort, **params)
        keep = asmgen.slots(kind, **params).get("keep", [0])
        out.update(
            {
                f"{name}_ASM": asm,
                f"{name}_CONS": cons,
                f"{name}_NOUT": nout,
                f"{name}_KLO": keep[0],
                f"{name}_KN": len(keep),
                f"{name}_KCONS": "=v," + "v," * (len(keep) - 1) + "v",
            }
        )
    return out


def check_cfg(world, hidden, cfg, tag=False):
    """Legality of a geometry for (TP, hidden) -> (cfg, shape, npw, nrole)."""
    c = dict(cfg)
    assert c.pop("NSEND", world) == world, "two_shot: NSEND must be the TP size"
    for k, v in _FIXED.items():
        assert c.pop(k, v) == v, f"two_shot: {k} must be {v}"
    assert set(c) <= set(DEFAULT), (
        f"two_shot: unknown geometry {sorted(set(c) - set(DEFAULT))}"
    )
    c = dict(DEFAULT, **c)
    s = shape(hidden, world, tag)
    assert s["NC"] % c["F"] == 0, f"F {c['F']} must divide {s['NC']} chunks"
    assert c["SDLY"] >= 0 and c["SDG"] >= 1
    npw = cdiv(s["NC"] // c["F"], 4)
    assert not c["PIPE"] or npw <= 2, "PIPE keeps two poll batches alive: NPW <= 2"
    return c, s, npw, world + s["RED"] + c["F"]


def ctr_words(world, hidden, max_tokens, **cfg):
    """Counter words of one instance: one CPAD-word cell per (row, role)."""
    return max_tokens * check_cfg(world, hidden, cfg)[3] * CPAD


def ipc_bytes(world, hidden, max_tokens, tag=False, **cfg):
    """IPC bytes per rank of one instance: RS and AG slots, per-row records."""
    s = check_cfg(world, hidden, cfg, tag)[1]
    return 2 * NSLOT * max_tokens * hidden * 2 + NSLOT * max_tokens * s["SROW"] * 4


def m_class_rep(m):
    """Representative of the kernels' row-count specialization class of m
    (Triton specializes an int argument on == 1 and % 16 == 0): 1, 16 or 17."""
    return 1 if m == 1 else (16 if m % 16 == 0 else 17)


def m_classes(lo, hi):
    """The specialization classes (as representatives) m in [lo, hi] can hit."""
    out = set()
    if lo <= 1 <= hi:
        out.add(1)
    if hi // 16 >= max(1, -(-lo // 16)):
        out.add(16)
    if any(v != 1 and v % 16 for v in range(max(lo, 1), min(hi, lo + 32) + 1)):
        out.add(17)
    return sorted(out)


_FL = FastLauncher(two_shot_kernel)
_FL_TAIL = FastLauncher(two_shot_tail_kernel)
_KEYS = itertools.count(1)


class TwoShotInst(PushInst):
    """One two_shot geometry on a region of the workspace. The owner
    synchronizes and runs a barrier after construction: the sentinel fill must
    land on every rank before any peer pushes."""

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
        **cfg,
    ):
        assert world in (2, 4, 8)
        self.rank, self.world, self.hidden = rank, world, int(hidden)
        self.max_tokens = int(max_tokens)
        self._key = next(_KEYS)
        self.fast = bool(fast)
        # tag: call-sequence check (needs diag: a mismatch is reported and
        # returns NaN); site_box[0] is the next launch's call site
        self.tag = bool(tag)
        assert not self.tag or diag is not None, (
            "call tags need the diagnostics (diag=)"
        )
        self.site_box = site_box if site_box is not None else [0]
        self.cfg, self.s, self.npw, self.nrole = check_cfg(
            world, self.hidden, cfg, self.tag
        )
        c, s = self.cfg, self.s
        self.spin = spin if spin is not None else SpinBound()
        self.diag = diag
        if diag is not None:
            self.spin = no_trap(self.spin)
        region = NSLOT * self.max_tokens * self.hidden  # int16 per RS / AG region
        self.ag_off = region
        self.ssq_off = region  # in int32 elements (= 2 int16 regions)
        ssq_words = NSLOT * self.max_tokens * s["SROW"]
        self.bytes_per_rank = 2 * region * 2 + ssq_words * 4
        assert self.bytes_per_rank == ipc_bytes(
            world, self.hidden, self.max_tokens, tag=self.tag, **cfg
        )
        self.buf = ipc.take(self.bytes_per_rank)
        loc = self.buf.addrs[rank]
        nwords = (2 * region * 2) // 4
        fill_kernel[(cdiv(nwords, 1024),)](
            I32Ptr(loc), nwords, SENT32, BLOCK=1024, num_warps=4
        )
        fill_ssq_kernel[(cdiv(ssq_words, 1024),)](
            I32Ptr(loc + region * 4),
            ssq_words,
            s["SROW"],
            s["SLIVE"],
            BLOCK=1024,
            num_warps=4,
        )
        self.ncell = self.max_tokens * self.nrole * CPAD
        assert ctr.numel() >= self.ncell and err.numel() >= 1
        self.ctr, self.err = ctr, err[:1]
        self.addrs = [int(a) for a in self.buf.addrs] + [0] * (8 - world)
        abort = None if diag is None else diag.abort
        self._asm = spin_kw(world, self.npw, c, self.spin, abort, self.tag)

    def _kw(self):
        a, c, s = self.addrs, self.cfg, self.s
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
            BLK=s["BLK"],
            AG_OFF=self.ag_off,
            SSQ_OFF=self.ssq_off,
            MAXR=self.max_tokens,
            F=c["F"],
            PIPE=c["PIPE"],
            **self._asm,
        )
        if c["SDLY"]:
            kw.update(SDLY=c["SDLY"], SDG=c["SDG"])
        if self.diag is not None:
            kw.update(self.diag.kw())
        if self.tag:
            kw["TAG"] = 1
        kw["num_warps"] = 4
        return kw

    def _launch(self, x, out, warmup=False, m=None):
        if m is None:
            m = x.shape[0]
        grid = (8 * cdiv(m, 8) * self.nrole,)
        args = (x, out, m, self.site_box[0])
        if warmup:  # compile (no launch) and prime the fast-launch cache
            return _FL.warm(grid, args, self._kw, key=(self._key, "ar"), fast=self.fast)
        _FL(grid, args, self._kw, key=(self._key, "ar"), fast=self.fast)

    def precompile(self, x, ms=(1,), tail=False):
        """Compile every row-count class of ms (and the masked tails) without
        launching."""
        out = torch.empty_like(x)
        for mm in sorted({m_class_rep(v) for v in ms}):
            self._launch(x, out, warmup=True, m=mm)
            if tail:  # both tail forms: TALN (n % 4 == 0) and the general one
                for nn in (x.numel(), x.numel() - 1):
                    self._launch_tail(x.view(-1), out.view(-1), nn, warmup=True, m=mm)

    def _launch_tail(self, x, out, n, warmup=False, m=None):
        m = -(-n // self.hidden) if m is None else m
        grid = (8 * cdiv(m, 8) * self.nrole,)
        args = (x, out, m, self.site_box[0], n)
        aln = int(n % 4 == 0)  # TALN: the 8-byte tail form

        def kw():
            return dict(self._kw(), **({"TALN": 1} if aln else {}))

        if warmup:
            return _FL_TAIL.warm(
                grid, args, kw, key=(self._key, "tail", aln), fast=self.fast
            )
        _FL_TAIL(grid, args, kw, key=(self._key, "tail", aln), fast=self.fast)

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
            lf = _FL_TAIL.bound(
                grid, (d, d, m, site[0], n), (self._key, "tail", aln), tspec
            )
            if lf is None:
                return None

            def run(x):
                out = torch.empty(n, dtype=x.dtype, device=x.device)
                lf(x, out, m, site[0], n)
                return out

            return run
        lf = _FL.bound(grid, (d, d, m, site[0]), (self._key, "ar"), tspec)
        if lf is None:
            return None

        def run(x):
            out = torch.empty_like(x)
            lf(x, out, m, site[0])
            return out

        return run
