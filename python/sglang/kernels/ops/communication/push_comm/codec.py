# SPDX-License-Identifier: Apache-2.0
"""Element codecs of the two-shot kernels (@gluon.jit helpers, inlined): bf16
pair helpers and the c12 and xe8 value codecs. The segment layouts that
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


# ------------------------------------------------------------------- xe8 codec
# xe8 (lossy, 9 bits / value): per 32 consecutive values the first largest |v|
# travels as its bf16 (high byte in the header, low byte in its code slot);
# the other 31 are int8 on a bf16 scale of their own signed extremes. A row
# block holding a NaN, an infinite group maximum, or a group whose other 31
# values are nonzero but all below 2^-100 is sent raw (bf16, header 2).
# Header per group: bf16 scale [31:16] | escape high byte [15:8] |
# escape position [6:2] | raw [1].
@gluon.jit
def _pz(v):
    # -0.0 -> +0.0 (an integer compare: nothing for the compiler to fold)
    b = v.to(gl.int32, bitcast=True)
    return gl.where(b == -2147483648, 0, b).to(gl.float32, bitcast=True)


@gluon.jit
def _bc4(t, S2: gl.constexpr):
    # [8, 8] per (row, group of 4 units) -> [8, 32] per unit
    g3 = gl.reshape(gl.zeros([8, 32], t.dtype, S2), [8, 8, 4])
    return gl.convert_layout(gl.reshape(gl.expand_dims(t, 2) + g3, [8, 32]), S2)


@gluon.jit
def _xs(gx, gn):
    # signed scale before rounding: the signed extreme / -128, widened to the
    # opposite extreme / 127 when larger (no value clamps)
    pos = gl.abs(gx) > gl.abs(gn)
    ext = gl.where(pos, gx, gn)
    t = gl.abs(ext) * (1.0 / 128.0)
    oq = gl.where(pos, -gn, gx) / 127.0
    t = gl.where(oq > t, oq, t)
    return gl.where(ext >= 0.0, t * -1.0, t)


@gluon.jit
def _qx(v, inv):
    # rint(v * inv), the product rounded once, clamped to [-128, 127]
    r = gl.fma(v, inv, gl.full_like(v, 12582912.0)) - 12582912.0
    return gl.minimum(gl.maximum(r, -128.0), 127.0)


@gluon.jit
def _enc_xe8(v, u2, S2: gl.constexpr):
    """8 fp32 tiles [8, 32] (value k of unit u) -> (w0, w1, header [8, 8],
    raw-row flag [8, 32])."""
    TINY: gl.constexpr = 7.888609052210118e-31  # 2^-100
    vs = ()
    for k in gl.static_range(8):
        vs += (_pz(v[k]),)
    am = gl.abs(vs[0])
    nan = vs[0] != vs[0]
    for k in gl.static_range(1, 8):
        am = gl.maximum(am, gl.abs(vs[k]))
        nan = nan | (vs[k] != vs[k])
    gmax = gl.max(gl.reshape(am, [8, 8, 4]), axis=2)
    gm = _bc4(gmax, S2)
    kk = gl.where(gl.abs(vs[7]) == am, 7, 0)
    for j in gl.static_range(7):
        kk = gl.where(gl.abs(vs[6 - j]) == am, 6 - j, kk)
    lp = (u2 % 4) * 8
    gpos = gl.min(gl.reshape(gl.where(am == gm, lp + kk, 32), [8, 8, 4]), axis=2)
    gposb = _bc4(gpos, S2)
    rmx = gl.where(lp == gposb, 0.0, vs[0])
    rmn = rmx
    ev = gl.where(lp == gposb, vs[0], 0.0)
    for k in gl.static_range(1, 8):
        x_ = lp + k == gposb
        r_ = gl.where(x_, 0.0, vs[k])
        rmx = gl.maximum(rmx, r_)
        rmn = gl.minimum(rmn, r_)
        ev = gl.where(x_, vs[k], ev)
    escb = _bf(ev)
    gx = gl.max(gl.reshape(rmx, [8, 8, 4]), axis=2)
    gn = gl.min(gl.reshape(rmn, [8, 8, 4]), axis=2)
    rest = gl.maximum(gl.maximum(gx, 0.0), gl.maximum(gn * -0.0078125, 0.0) * 128.0)
    bad = _bc4(
        gl.where((gmax == float("inf")) | ((rest > 0.0) & (rest < TINY)), 1, 0), S2
    ) | gl.where(nan, 1, 0)
    er = gl.expand_dims(gl.max(bad, axis=1), 1) + gl.zeros_like(bad)
    tB = _xs(gx, gn).to(gl.int32, bitcast=True)
    sBb = (tB + 32767 + ((tB >> 16) & 1)) & -65536  # bf16 RNE
    sc = _bc4(sBb, S2).to(gl.float32, bitcast=True)
    inv = gl.where(sc != 0.0, 1.0 / sc, 0.0)
    w0 = gl.zeros_like(kk)
    w1 = gl.zeros_like(kk)
    for k in gl.static_range(8):
        x_ = lp + k == gposb
        c = gl.where(x_, escb, _qx(gl.where(x_, 0.0, vs[k]), inv).to(gl.int32)) & 255
        if k < 4:
            w0 = w0 | (c << (8 * k))
        else:
            w1 = w1 | (c << (8 * (k - 4)))
    eG = gl.max(
        gl.reshape(escb, [8, 8, 4]), axis=2
    )  # only the escape's lane is nonzero
    e8 = gl.max(gl.reshape(er, [8, 8, 4]), axis=2)
    hdr = gl.where(e8 != 0, 2, (sBb & -65536) | ((eG >> 8) << 8) | (gpos << 2))
    return w0, w1, hdr, er


@gluon.jit
def _dec_xe8(w0, w1, w2, w3, hd, u2):
    """One unit's codes (w0, w1), raw planes 2..3 and group header -> 8 fp32
    values (a raw row: its bf16 values from planes 0..3)."""
    raw = (hd & 2) != 0
    sc = (hd & -65536).to(gl.float32, bitcast=True)
    rel = ((hd >> 2) & 31) - (u2 % 4) * 8  # the escape's index in this unit
    ce = (gl.where(rel < 4, w0, w1) >> ((rel & 3) * 8)) & 255
    ef = ((((hd >> 8) & 255) << 8 | ce) << 16).to(gl.float32, bitcast=True)
    rv = (_lo(w0), _hi(w0), _lo(w1), _hi(w1), _lo(w2), _hi(w2), _lo(w3), _hi(w3))
    out = ()
    for k in gl.static_range(8):
        if k < 4:
            q = (w0 << (24 - 8 * k)) >> 24  # sign-extended byte
        else:
            q = (w1 << (24 - 8 * (k - 4))) >> 24
        d = gl.where(rel == k, ef, q.to(gl.float32) * sc + 0.0)
        out += (gl.where(raw, rv[k], d),)
    return out
