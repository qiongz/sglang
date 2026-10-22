# SPDX-License-Identifier: Apache-2.0
"""Element codecs of the two-shot kernels (@gluon.jit helpers, inlined): bf16
pair helpers and the c12 and int8 value codecs. The segment layouts that
carry them live with the kernels (two_shot_epoch, two_shot_window)."""

from triton.experimental import gluon
from triton.experimental.gluon import language as gl


@gluon.jit
def _popc(x):
    return gl.inline_asm_elementwise(
        "v_bcnt_u32_b32 $0, $1, 0", "=v,v", [x], dtype=gl.int32, is_pure=True, pack=1
    )


# ------------------------------------------------------------ element helpers
@gluon.jit
def _pair(t, q3, Q: gl.constexpr):
    # pair Q of a [8, 32, 4] dword tile (in-register select: the axis is per
    # thread)
    return gl.sum(gl.where(q3 == Q, t, 0), axis=2)


@gluon.jit
def _quad(p0, p1, p2, p3, q3):
    a0 = gl.expand_dims(p0, 2)
    a1 = gl.expand_dims(p1, 2)
    a2 = gl.expand_dims(p2, 2)
    a3 = gl.expand_dims(p3, 2)
    return gl.where(q3 == 0, a0, gl.where(q3 == 1, a1, gl.where(q3 == 2, a2, a3)))


@gluon.jit
def _lo(p):
    return (p << 16).to(gl.float32, bitcast=True)


@gluon.jit
def _hi(p):
    return (p & -65536).to(gl.float32, bitcast=True)


@gluon.jit
def _bf(v):
    return v.to(gl.bfloat16).to(gl.int16, bitcast=True).to(gl.int32) & 65535


@gluon.jit
def _pk(a, b):
    return _bf(a) | (_bf(b) << 16)


@gluon.jit
def _rbf(v):
    # round an fp32 tile to bf16 and back (identity on values that are bf16)
    return v.to(gl.bfloat16).to(gl.float32)


# ------------------------------------------------------------------ c12 codec
@gluon.jit
def _ex(b):
    return (b >> 7) & 255


@gluon.jit
def _enc1(b, B7):
    mag = b & 32767
    ok = (mag == 0) | (mag > B7)
    code = gl.where(mag == 0, 0, mag - B7) & 2047
    return code | ((b >> 4) & 2048), ok


@gluon.jit
def _dec1(c, B7):
    cm = c & 2047
    mag = gl.where(cm == 0, 0, cm + B7)
    return mag | ((c & 2048) << 4)


@gluon.jit
def _encode(p0, p1, p2, p3, u2):
    """4 raw pair tiles [8, 32] -> (w0, w1, w2, raw pair 3, esc, base [8],
    escape mask [8])."""
    b0 = p0 & 65535
    b1 = (p0 >> 16) & 65535
    b2 = p1 & 65535
    b3 = (p1 >> 16) & 65535
    b4 = p2 & 65535
    b5 = (p2 >> 16) & 65535
    b6 = p3 & 65535
    b7 = (p3 >> 16) & 65535
    e = gl.maximum(
        gl.maximum(gl.maximum(_ex(b0), _ex(b1)), gl.maximum(_ex(b2), _ex(b3))),
        gl.maximum(gl.maximum(_ex(b4), _ex(b5)), gl.maximum(_ex(b6), _ex(b7))),
    )
    base = gl.maximum(gl.max(e, axis=1) - 15, 0)
    B7 = gl.expand_dims(base << 7, 1)
    c0, k0 = _enc1(b0, B7)
    c1, k1 = _enc1(b1, B7)
    c2, k2 = _enc1(b2, B7)
    c3, k3 = _enc1(b3, B7)
    c4, k4 = _enc1(b4, B7)
    c5, k5 = _enc1(b5, B7)
    c6, k6 = _enc1(b6, B7)
    c7, k7 = _enc1(b7, B7)
    esc = gl.where(k0 & k1 & k2 & k3 & k4 & k5 & k6 & k7, 0, 1)
    msk = gl.sum(esc << u2, axis=1)
    w0 = c0 | (c1 << 12) | (c2 << 24)
    w1 = (c2 >> 8) | (c3 << 4) | (c4 << 16) | (c5 << 28)
    w2 = (c5 >> 4) | (c6 << 8) | (c7 << 20)
    w0 = gl.where(esc != 0, p0, w0)
    w1 = gl.where(esc != 0, p1, w1)
    w2 = gl.where(esc != 0, p2, w2)
    return w0, w1, w2, p3, esc, base, msk


@gluon.jit
def _decode(w0, w1, w2, w3, esc, base):
    """(3 code dwords, escape pair 3, esc flag, base [8] or broadcast [8, 32])
    -> 4 raw pair tiles."""
    B7 = base << 7
    d0 = _dec1(w0, B7)
    d1 = _dec1(w0 >> 12, B7)
    d2 = _dec1(((w0 >> 24) & 255) | (w1 << 8), B7)
    d3 = _dec1(w1 >> 4, B7)
    d4 = _dec1(w1 >> 16, B7)
    d5 = _dec1(((w1 >> 28) & 15) | (w2 << 4), B7)
    d6 = _dec1(w2 >> 8, B7)
    d7 = _dec1(w2 >> 20, B7)
    e = esc != 0
    p0 = gl.where(e, w0, d0 | (d1 << 16))
    p1 = gl.where(e, w1, d2 | (d3 << 16))
    p2 = gl.where(e, w2, d4 | (d5 << 16))
    p3 = gl.where(e, w3, d6 | (d7 << 16))
    return p0, p1, p2, p3


# ------------------------------------------------------------------ int8 codec
# int8: groups of 32 elements share one bf16 scale = (signed extreme) / -128,
# so the extreme maps to -128 exactly.
@gluon.jit
def _q8(v, inv, LO: gl.constexpr):
    # round-half-even of v * (1 / scale) (magic-number RNE), clamp to [LO, 127]
    y = v * inv
    r = (y + 12582912.0) - 12582912.0
    return gl.minimum(gl.maximum(r, LO), 127.0).to(gl.int32) & 255


@gluon.jit
def _sx(w, SH: gl.constexpr):
    # sign-extended byte (SH = 24, 16, 8, 0 selects byte 0..3) as fp32
    return ((w << SH) >> 24).to(gl.float32)


@gluon.jit
def _enc_i8(v0, v1, v2, v3, v4, v5, v6, v7, S2: gl.constexpr):
    """8 fp32 element tiles [8, 32] -> (w0, w1, hdr [8, 4])."""
    am = gl.maximum(
        gl.maximum(
            gl.maximum(gl.abs(v0), gl.abs(v1)), gl.maximum(gl.abs(v2), gl.abs(v3))
        ),
        gl.maximum(
            gl.maximum(gl.abs(v4), gl.abs(v5)), gl.maximum(gl.abs(v6), gl.abs(v7))
        ),
    )
    g4 = gl.reshape(am, [8, 4, 2, 4])  # [row, dword, half, unit]
    gm = gl.max(g4, axis=3)
    mx = gl.maximum(
        gl.maximum(gl.maximum(v0, v1), gl.maximum(v2, v3)),
        gl.maximum(gl.maximum(v4, v5), gl.maximum(v6, v7)),
    )
    # largest signed value of the group
    gp = gl.max(gl.reshape(mx, [8, 4, 2, 4]), axis=3)
    ext = gl.where(gp == gm, gm, -gm)  # the signed extreme
    gb = (ext * -0.0078125).to(gl.int32, bitcast=True)  # ext / -128, nearest bf16
    sb = (gb + 32767 + ((gb >> 16) & 1)) & -65536
    # half index (group parity inside a header dword) with the same layout as gm
    hf = gl.max(
        gl.reshape(
            (gl.expand_dims(gl.arange(0, 32, layout=gl.SliceLayout(0, S2)), 0) // 4) % 2
            + gl.expand_dims(gl.arange(0, 8, layout=gl.SliceLayout(1, S2)), 1) * 0,
            [8, 4, 2, 4],
        ),
        axis=3,
    )
    s_lo = gl.sum(gl.where(hf == 0, sb, 0), axis=2)
    s_hi = gl.sum(gl.where(hf == 1, sb, 0), axis=2)
    hdr = ((s_lo >> 16) & 65535) | (s_hi & -65536)
    su = gl.reshape(gl.expand_dims(sb, 3) + gl.zeros_like(g4).to(gl.int32), [8, 32])
    sc = gl.convert_layout(su, S2).to(gl.float32, bitcast=True)
    inv = gl.where(sc != 0.0, 1.0 / sc, 0.0)
    LO: gl.constexpr = -128.0
    w0 = (
        _q8(v0, inv, LO)
        | (_q8(v1, inv, LO) << 8)
        | (_q8(v2, inv, LO) << 16)
        | (_q8(v3, inv, LO) << 24)
    )
    w1 = (
        _q8(v4, inv, LO)
        | (_q8(v5, inv, LO) << 8)
        | (_q8(v6, inv, LO) << 16)
        | (_q8(v7, inv, LO) << 24)
    )
    return w0, w1, hdr


@gluon.jit
def _dec_i8(w0, w1, hd, u2):
    sc = gl.where(((u2 // 4) % 2) == 1, hd & -65536, hd << 16).to(
        gl.float32, bitcast=True
    )
    return (
        _sx(w0, 24) * sc,
        _sx(w0, 16) * sc,
        _sx(w0, 8) * sc,
        _sx(w0, 0) * sc,
        _sx(w1, 24) * sc,
        _sx(w1, 16) * sc,
        _sx(w1, 8) * sc,
        _sx(w1, 0) * sc,
    )
