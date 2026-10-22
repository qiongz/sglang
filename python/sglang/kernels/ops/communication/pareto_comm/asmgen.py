# SPDX-License-Identifier: Apache-2.0
"""Generator and checker of the spin-loop inline asm (pure Python).

spin(kind, bound=, abort=, **params) -> (asm, cons, nout, n_in), checked by
check(): operand classes, early-clobber outputs, clobbers, uncached (sc0 sc1)
polls and self-contained local labels. Every spin carries the wall-clock bound
of protocol.timed_tail_lines."""

from __future__ import annotations

import functools
import re

from .protocol import DEFAULT_SPIN, LIMIT, TCLOB, SpinBound, timed_tail_lines

__all__ = ["spin", "slots", "check", "check_labels", "AsmDriftError", "KINDS"]

KINDS = ("rs", "ag", "rs_pipe", "ag_pipe", "flag_eq", "flag_ge", "lp")
_IND = " " * 8  # indentation of the "indented" layout (rs, flag spins)
_TAIL_RE = re.compile(
    r"s_cmp_lt_u32 (\$\d+), " + str(LIMIT) + r"\n(\s*)s_cbranch_scc1 1b"
)


class AsmDriftError(ValueError):
    """The asm text, the constraint string, nout and n_in disagree, or a protocol
    invariant (clobbers, early-clobber outputs, uncached polls, self-contained
    labels) is broken."""


def _drift(cond, msg):
    if not cond:
        raise AsmDriftError(msg)


# ------------------------------------------------------------------ builders
# Each builder returns (lines, constraints, layout, roles).
def _rs_u16(n, lim):
    ic, iflag, ip = n, n + 1, n + 2
    L = [f"s_mov_b32 ${ic}, 0", "1:"]
    L += [f"global_load_ushort ${i}, ${ip + i}, off sc0 sc1" for i in range(n)]
    L += ["s_waitcnt vmcnt(0)", f"v_mov_b32 ${iflag}, 0"]
    for i in range(n):
        L += [
            f"v_cmp_eq_u32 vcc, 0x8000, ${i}",
            f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
        ]
    L += [
        f"v_cmp_ne_u32 vcc, 0, ${iflag}",
        "s_cbranch_vccz 2f",
        f"s_add_u32 ${ic}, ${ic}, 1",
    ]
    L += [f"s_cmp_lt_u32 ${ic}, {lim}", "s_cbranch_scc1 1b", "2:"]
    cons = ["=&v"] * n + ["=&s", "=&v"] + ["v"] * n + ["~{vcc}", "~{scc}", "~{memory}"]
    roles = dict(
        data=list(range(n)), ctr=ic, scratch=[iflag], ptr=list(range(ip, ip + n))
    )
    return L, cons, "indented", roles


def _ag(n, nq, lim):
    nd = 2 * n
    iq = nd
    ic = nd + nq
    ia, it, iflag = ic + 1, ic + 2, ic + 3
    ip = ic + 4
    L = [f"s_mov_b32 ${ic}, 0", "1:"]
    for i in range(n):
        L.append(f"global_load_dword ${2 * i}, ${ip + i}, off sc0 sc1")
        L.append(f"global_load_dword ${2 * i + 1}, ${ip + i}, off offset:4 sc0 sc1")
    for j in range(nq):
        L.append(f"global_load_dword ${iq + j}, ${ip + n + j}, off sc0 sc1")
    L.append("s_waitcnt vmcnt(0)")
    L.append(f"v_pk_min_i16 ${ia}, $0, $1")
    for k in range(2, nd):
        L.append(f"v_pk_min_i16 ${ia}, ${ia}, ${k}")
    L += [
        f"v_mov_b32 ${iflag}, 0",
        f"v_and_b32 ${it}, 0xffff, ${ia}",
        f"v_cmp_eq_u32 vcc, 0x8000, ${it}",
        f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
        f"v_lshrrev_b32 ${it}, 16, ${ia}",
        f"v_cmp_eq_u32 vcc, 0x8000, ${it}",
        f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
    ]
    for j in range(nq):
        L += [
            f"v_cmp_eq_u32 vcc, 0x80000000, ${iq + j}",
            f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
        ]
    L += [
        f"v_cmp_ne_u32 vcc, 0, ${iflag}",
        "s_cbranch_vccz 2f",
        f"s_add_u32 ${ic}, ${ic}, 1",
    ]
    L += [f"s_cmp_lt_u32 ${ic}, {lim}", "s_cbranch_scc1 1b", "2:"]
    cons = ["=&v"] * (nd + nq) + ["=&s", "=&v", "=&v", "=&v"] + ["v"] * (n + nq)
    cons += ["~{vcc}", "~{scc}", "~{memory}"]
    roles = dict(
        data=list(range(nd)),
        q=list(range(iq, iq + nq)),
        ctr=ic,
        scratch=[ia, it, iflag],
        ptr=list(range(ip, ip + n + nq)),
    )
    return L, cons, "flat", roles


def _pspin(loads, check_fn, slp, lim):
    """Two poll batches A / B staggered by s_sleep slp; on exit the other batch is
    still in flight and its registers (role 'keep') must stay live until a
    drain (two_shot._drain)."""
    nl = len(loads)
    io = 0
    out = list(range(io, io + nl))
    io += nl
    ic = io
    io += 1
    ia, it, iflag = io, io + 1, io + 2
    io += 3
    A = list(range(io, io + nl))
    io += nl
    B = list(range(io, io + nl))
    io += nl
    ip = io
    nptr = max(pi for _, pi, _ in loads) + 1

    def issue(R):
        return [
            f"{op} ${R[i]}, ${ip + pi}, off{(' offset:%d' % off) if off else ''} sc0 sc1"
            for i, (op, pi, off) in enumerate(loads)
        ]

    def chk(R):
        return check_fn(R, ia, it, iflag) + [f"v_cmp_ne_u32 vcc, 0, ${iflag}"]

    X = (
        [f"s_mov_b32 ${ic}, 0"]
        + issue(A)
        + ([f"s_sleep {slp}"] if slp else [])
        + issue(B)
    )
    X += ["1:", f"s_waitcnt vmcnt({nl})"] + chk(A) + ["s_cbranch_vccz 3f"] + issue(A)
    X += [f"s_waitcnt vmcnt({nl})"] + chk(B) + ["s_cbranch_vccz 4f"] + issue(B)
    X += [
        f"s_add_u32 ${ic}, ${ic}, 1",
        f"s_cmp_lt_u32 ${ic}, {lim}",
        "s_cbranch_scc1 1b",
    ]
    X += ["4:"] + [f"v_mov_b32 ${out[i]}, ${B[i]}" for i in range(nl)] + ["s_branch 5f"]
    X += ["3:"] + [f"v_mov_b32 ${out[i]}, ${A[i]}" for i in range(nl)] + ["5:"]
    cons = (
        ["=&v"] * nl + ["=&s", "=&v", "=&v", "=&v"] + ["=&v"] * (2 * nl) + ["v"] * nptr
    )
    cons += ["~{vcc}", "~{scc}", "~{memory}"]
    roles = dict(
        data=out,
        ctr=ic,
        scratch=[ia, it, iflag],
        keep=A + B,
        ptr=list(range(ip, ip + nptr)),
    )
    return X, cons, "flat", roles


def _chk_u16(R, ia, it, iflag):
    X = [f"v_mov_b32 ${iflag}, 0"]
    for r in R:
        X += [
            f"v_cmp_eq_u32 vcc, 0x8000, ${r}",
            f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
        ]
    return X


def _mk_chk_pk(nd, nq):
    def chk(R, ia, it, iflag):
        X = [f"v_pk_min_i16 ${ia}, ${R[0]}, ${R[1]}"]
        for k in range(2, nd):
            X.append(f"v_pk_min_i16 ${ia}, ${ia}, ${R[k]}")
        X += [
            f"v_mov_b32 ${iflag}, 0",
            f"v_and_b32 ${it}, 0xffff, ${ia}",
            f"v_cmp_eq_u32 vcc, 0x8000, ${it}",
            f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
            f"v_lshrrev_b32 ${it}, 16, ${ia}",
            f"v_cmp_eq_u32 vcc, 0x8000, ${it}",
            f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
        ]
        for j in range(nq):
            X += [
                f"v_cmp_eq_u32 vcc, 0x80000000, ${R[nd + j]}",
                f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
            ]
        return X

    return chk


def _rs_pipe(n, pslp, lim):
    return _pspin([("global_load_ushort", i, 0) for i in range(n)], _chk_u16, pslp, lim)


def _ag_pipe(n, nq, pslp, lim):
    ld = []
    for i in range(n):
        ld += [("global_load_dword", i, 0), ("global_load_dword", i, 4)]
    for j in range(nq):
        ld.append(("global_load_dword", n + j, 0))
    L, cons, layout, roles = _pspin(ld, _mk_chk_pk(2 * n, nq), pslp, lim)
    out = roles["data"]
    roles = dict(roles, data=out[: 2 * n], q=out[2 * n :])
    return L, cons, layout, roles


def _lp(nsrc, offs, lim, act=0, nsrc2=0, offs2=()):
    """Poll nsrc x len(offs) (+ nsrc2 x len(offs2)) dwords until no halfword
    is 0x8000. act: a last input; 0 on the whole wave skips the poll."""
    nd = nsrc * len(offs) + nsrc2 * len(offs2)
    ic = nd
    ia, it, iflag = ic + 1, ic + 2, ic + 3
    ip = ic + 4
    nptr = nsrc + nsrc2
    L = [f"s_mov_b32 ${ic}, 0"]
    if act:
        L += [f"v_cmp_ne_u32 vcc, 0, ${ip + nptr}", "s_cbranch_vccz 2f"]
    L += ["1:"]
    k = 0
    for i in range(nsrc):
        for o in offs:
            L.append(
                f"global_load_dword ${k}, ${ip + i}, off{(' offset:%d' % o) if o else ''} sc0 sc1"
            )
            k += 1
    for i in range(nsrc2):
        for o in offs2:
            L.append(
                f"global_load_dword ${k}, ${ip + nsrc + i}, off{(' offset:%d' % o) if o else ''} sc0 sc1"
            )
            k += 1
    L.append("s_waitcnt vmcnt(0)")
    if nd == 1:
        L.append(f"v_mov_b32 ${ia}, $0")
    else:
        L.append(f"v_pk_min_i16 ${ia}, $0, $1")
        for j in range(2, nd):
            L.append(f"v_pk_min_i16 ${ia}, ${ia}, ${j}")
    L += [
        f"v_mov_b32 ${iflag}, 0",
        f"v_and_b32 ${it}, 0xffff, ${ia}",
        f"v_cmp_eq_u32 vcc, 0x8000, ${it}",
        f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
        f"v_lshrrev_b32 ${it}, 16, ${ia}",
        f"v_cmp_eq_u32 vcc, 0x8000, ${it}",
        f"v_cndmask_b32_e64 ${iflag}, ${iflag}, 1, vcc",
    ]
    L += [
        f"v_cmp_ne_u32 vcc, 0, ${iflag}",
        "s_cbranch_vccz 2f",
        f"s_add_u32 ${ic}, ${ic}, 1",
    ]
    L += [f"s_cmp_lt_u32 ${ic}, {lim}", "s_cbranch_scc1 1b", "2:"]
    cons = (
        ["=&v"] * nd
        + ["=&s", "=&v", "=&v", "=&v"]
        + ["v"] * (nptr + (1 if act else 0))
        + ["~{vcc}", "~{scc}", "~{memory}"]
    )
    roles = dict(
        data=list(range(nd)),
        ctr=ic,
        scratch=[ia, it, iflag],
        ptr=list(range(ip, ip + nptr)),
        **({"arg": [ip + nptr]} if act else {}),
    )
    return L, cons, "flat", roles


def _flag(op, ticks, trap, bk=0, abort=None):
    """two_shot_epoch's per-lane flag spins, timed inline (s43 is cleared after
    the counter; the tail follows the counter increment)."""
    test = {
        "eq": ["v_cmp_ne_u32 vcc, $3, $0"],
        "ge": ["v_sub_u32 $0, $0, $3", "v_cmp_gt_i32 vcc, 0, $0"],
    }[op]
    L = [
        "s_mov_b32 $1, 0",
        "s_mov_b32 s43, 0",
        "1:",
        "global_load_dword $0, $2, off sc0 sc1",
        "s_waitcnt vmcnt(0)",
    ]
    L += test + ["s_cbranch_vccz 2f", "s_add_u32 $1, $1, 1"]
    L += timed_tail_lines("$1", ticks, trap, bk, abort) + ["2:"]
    cons = ["=&v", "=&s", "v", "v", "~{vcc}", "~{scc}", "~{memory}"] + TCLOB[1:].split(
        ","
    )
    roles = dict(data=[0], ctr=1, scratch=[], ptr=[2], arg=[3])
    return L, cons, "indented", roles


def _render(lines, layout):
    if layout == "indented":  # leading newline, 8-space indent, indented trailing line
        return "\n" + "\n".join(_IND + ln for ln in lines) + "\n" + _IND
    return "\n".join(lines)


# ---------------------------------------------------------------- parameters
_SPEC = {  # kind -> (required, defaults)
    "rs": (("nsrc",), {}),
    "ag": (("n",), dict(nq=0)),
    "rs_pipe": (("nsrc",), dict(pslp=4)),
    "ag_pipe": (("n",), dict(nq=0, pslp=4)),
    "flag_eq": ((), {}),
    "flag_ge": ((), {}),
    "lp": (("nsrc", "offs"), dict(act=0, nsrc2=0, offs2=())),
}


def _norm(kind, params):
    if kind not in _SPEC:
        raise ValueError(f"unknown spin kind {kind!r}; kinds: {KINDS}")
    req, dflt = _SPEC[kind]
    bad = set(params) - set(req) - set(dflt)
    if bad:
        raise TypeError(f"spin({kind!r}): unknown parameter(s) {sorted(bad)}")
    miss = [r for r in req if r not in params]
    if miss:
        raise TypeError(f"spin({kind!r}): missing parameter(s) {miss}")
    p = dict(dflt, **params)
    for k in ("nsrc", "n", "nq", "pslp"):
        if k in p:
            _drift(
                isinstance(p[k], int) and not isinstance(p[k], bool),
                f"{k} must be an int, got {p[k]!r}",
            )
    if "nsrc" in p and kind == "lp":
        _drift(1 <= p["nsrc"] <= 16, f"lp: nsrc must be 1..16, got {p['nsrc']}")
        p["offs"] = tuple(int(o) for o in p["offs"])
        p["offs2"] = tuple(int(o) for o in p["offs2"])
        _drift(
            0 <= p["nsrc2"] <= 16
            and all(0 <= o < 4096 and o % 4 == 0 for o in p["offs2"]),
            f"lp: bad second group nsrc2={p['nsrc2']} offs2={p['offs2']}",
        )
        _drift(
            all(0 <= o < 4096 and o % 4 == 0 for o in p["offs"])
            and len(p["offs"]) >= 1,
            f"lp: offs must be dword-aligned byte offsets < 4096, got {p['offs']}",
        )
        _drift(p["act"] in (0, 1), f"lp: act must be 0 or 1, got {p['act']}")
    elif "nsrc" in p:
        _drift(p["nsrc"] in (2, 4, 8), f"nsrc must be 2, 4 or 8, got {p['nsrc']}")
    if "n" in p:
        _drift(1 <= p["n"] <= 32, f"n must be 1..32, got {p['n']}")
    if "nq" in p:
        _drift(p["nq"] in (0, 1), f"nq must be 0 or 1, got {p['nq']}")
    if "pslp" in p:  # s_sleep simm16[6:0]
        _drift(0 <= p["pslp"] <= 127, f"pslp must be 0..127, got {p['pslp']}")
    return tuple(sorted(p.items()))


def _build(kind, p, ticks, trap, bk=0, abort=None):
    p = dict(p)
    if kind == "rs":
        return _rs_u16(p["nsrc"], LIMIT)
    if kind == "ag":
        return _ag(p["n"], p["nq"], LIMIT)
    if kind == "rs_pipe":
        return _rs_pipe(p["nsrc"], p["pslp"], LIMIT)
    if kind == "ag_pipe":
        return _ag_pipe(p["n"], p["nq"], p["pslp"], LIMIT)
    if kind == "lp":
        return _lp(p["nsrc"], p["offs"], LIMIT, p["act"], p["nsrc2"], p["offs2"])
    return _flag(kind[5:], ticks, trap, bk, abort)


def _counts(cons):
    """(nout, n_in) of a constraint list: '=' entries are outputs, '~{..}'
    clobbers, the rest inputs."""
    nout = sum(1 for c in cons if c.startswith("="))
    n_in = sum(1 for c in cons if not c.startswith(("=", "~")))
    return nout, n_in


# ---------------------------------------------------------------- transforms
def _timed(asm, ticks, trap, bk=0, abort=None):
    """Turn the poll-count bound into the wall-clock bound
    (protocol.timed_tail_lines); the abort exit label is the first label after
    the bound check (2 for rs / ag, 4 for the pipelined forms)."""

    def tail(mt):
        ic, ind = mt.group(1), mt.group(2)
        nxt = re.search(r"^\s*(\d+):\s*$", asm[mt.end() :], re.M)
        lines = timed_tail_lines(
            ic, ticks, trap, bk, abort, nxt.group(1) if nxt else "2"
        )
        return ("\n" + ind).join(lines)

    out, n = _TAIL_RE.subn(tail, asm)
    _drift(
        n == 1 and "8:" not in asm and "9:" not in asm,
        "unexpected spin shape for the timed tail",
    )
    return "s_mov_b32 s43, 0\n" + out


_DEF_NUM = re.compile(r"^(\s*)(\d+):(\s*)$")
_BR = re.compile(r"^(\s*)(s_branch|s_cbranch_[a-z0-9]+)\s+(\S+)(\s*)$")
_REF_NUM = re.compile(r"^(\d+)([bf])$")


def check_labels(asm):
    """Every branch target is a local label defined once inside the block, in
    the direction of its b / f suffix, so the block cannot branch into a
    neighbouring inlined asm block."""
    lines = asm.split("\n")
    defs = {}
    for i, ln in enumerate(lines):
        mn = _DEF_NUM.match(ln)
        if mn:
            lab = mn.group(2)
            _drift(lab not in defs, f"label {lab} defined twice")
            defs[lab] = i
        elif ln.strip().endswith(":"):
            raise AsmDriftError(f"unrecognised label line {i}: {ln.strip()}")
    for i, ln in enumerate(lines):
        if not ln.strip().startswith(("s_branch", "s_cbranch")):
            continue
        m = _BR.match(ln)
        _drift(m is not None, f"unparsed branch at line {i}: {ln.strip()}")
        tgt = m.group(3)
        r = _REF_NUM.match(tgt)
        _drift(r is not None, f"branch target {tgt!r} is not a numeric local label")
        lab, d = r.group(1), r.group(2)
        _drift(lab in defs, f"branch to {tgt} leaves the block (no {lab}: inside it)")
        _drift(
            defs[lab] < i if d == "b" else defs[lab] > i,
            f"{tgt} at line {i} resolves outside the block",
        )


# ---------------------------------------------------------------- the checker
_SLOT = re.compile(r"\$(\d+)")
_PHYS_S = re.compile(r"(?<![\w$])s(\d+)(?!\w)|(?<![\w$])s\[(\d+):(\d+)\]")
_PHYS_V = re.compile(r"(?<![\w$])v(\d+)(?!\w)|(?<![\w$])v\[(\d+):(\d+)\]")
_SCC_WRITERS = (
    "s_cmp",
    "s_add",
    "s_addc",
    "s_sub",
    "s_subb",
    "s_and",
    "s_or",
    "s_xor",
    "s_lsh",
    "s_cselect",
)


def check(asm, cons, nout, n_in):
    """Raise AsmDriftError unless the operand counts, register classes,
    early-clobber outputs, clobber list, sc0 sc1 polls (a cached poll never
    terminates) and local labels of asm / cons agree."""
    _drift(isinstance(asm, str) and isinstance(cons, str), "asm and cons must be str")
    ents = cons.split(",")
    _drift(all(ents), f"empty constraint entry in {cons!r}")
    kinds = [
        "o" if e.startswith("=") else ("c" if e.startswith("~") else "i") for e in ents
    ]
    _drift(
        kinds == sorted(kinds, key="oic".index),
        f"constraint order must be outputs, inputs, clobbers: {cons!r}",
    )
    outs = [e for e, k in zip(ents, kinds) if k == "o"]
    ins = [e for e, k in zip(ents, kinds) if k == "i"]
    clob = {e for e, k in zip(ents, kinds) if k == "c"}
    _drift(len(outs) == nout, f"cons has {len(outs)} '=' outputs, nout = {nout}")
    _drift(len(ins) == n_in, f"cons has {len(ins)} inputs, n_in = {n_in}")
    for e in outs:
        _drift(e in ("=&v", "=&s"), f"output {e!r} is not an early-clobber v/s slot")
    for e in ins:
        _drift(e in ("v", "s"), f"input {e!r} is not a v/s slot")
    cls = [e[-1] for e in outs] + list(ins)
    total = nout + n_in
    refs = {int(k) for k in _SLOT.findall(asm)}
    _drift(
        not refs or max(refs) < total,
        f"asm references ${max(refs) if refs else 0} but only {total} slots exist",
    )
    miss = [k for k in range(nout, total) if k not in refs]
    _drift(
        not miss,
        f"input slot(s) {miss} never referenced: outputs and inputs drifted apart",
    )
    body = [ln.strip() for ln in asm.split("\n") if ln.strip()]
    for ln in body:
        op = ln.split()[0]
        want = (
            "s"
            if op.startswith("s_")
            else ("v" if op.startswith(("v_", "global_")) else None)
        )
        for k in _SLOT.findall(ln):
            if want is not None:
                _drift(
                    cls[int(k)] == want,
                    f"slot ${k} is class {cls[int(k)]!r} but used by {op}: {ln}",
                )
        if op.startswith("global_load"):
            _drift(ln.endswith("sc0 sc1"), f"poll without sc0 sc1: {ln}")
    if "vcc" in asm:
        _drift("~{vcc}" in clob, "body names vcc but ~{vcc} is not clobbered")
    if any(ln.split()[0].startswith(_SCC_WRITERS) for ln in body):
        _drift("~{scc}" in clob, "body writes scc but ~{scc} is not clobbered")
    if any(ln.startswith("global_") for ln in body):
        _drift(
            "~{memory}" in clob, "body accesses memory but ~{memory} is not clobbered"
        )
    for rx, pre in ((_PHYS_S, "s"), (_PHYS_V, "v")):
        for m in rx.finditer(asm):
            if m.group(1):
                regs = [int(m.group(1))]
            else:
                regs = range(int(m.group(2)), int(m.group(3)) + 1)
            for r in regs:
                _drift(
                    f"~{{{pre}{r}}}" in clob,
                    f"body names physical {pre}{r} but ~{{{pre}{r}}} is not clobbered",
                )
    check_labels(asm)


# ---------------------------------------------------------------- public API
@functools.lru_cache(maxsize=None)
def _spin_cached(kind, p, ticks, trap, bk=0, abort=None):
    L, cons_l, layout, roles = _build(kind, p, ticks, trap, bk, abort)
    asm, cons = _render(L, layout), ",".join(cons_l)
    nout, n_in = _counts(cons_l)
    outs = (
        roles["data"]
        + roles.get("q", [])
        + [roles["ctr"]]
        + roles["scratch"]
        + roles.get("keep", [])
    )
    _drift(
        sorted(outs) == list(range(nout)),
        f"{kind}: output roles do not tile 0..{nout - 1}",
    )
    _drift(
        sorted(roles["ptr"] + roles.get("arg", [])) == list(range(nout, nout + n_in)),
        f"{kind}: input roles do not tile {nout}..{nout + n_in - 1}",
    )
    _drift(
        cons_l[roles["ctr"]] == "=&s",
        f"{kind}: the counter slot ${roles['ctr']} is not =&s",
    )
    check_labels(asm)
    if kind not in ("flag_eq", "flag_ge"):  # the flag spins are built timed
        asm = _timed(asm, ticks, trap, bk, abort)
        _drift("=&s" in cons and "~{memory}" in cons, "timed-tail precondition")
        cons = cons + TCLOB
    check(asm, cons, nout, n_in)
    return asm, cons, nout, n_in


def spin(kind, bound=None, abort=None, **params):
    """-> (asm, cons, nout, n_in), checked; kind is one of rs, ag, rs_pipe,
    ag_pipe, flag_eq, flag_ge, lp. bound: protocol.SpinBound (None: 300 s,
    trap). abort: device address of the host's abort word, also polled."""
    if bound is None:
        bound = DEFAULT_SPIN
    _drift(isinstance(bound, SpinBound), f"bound must be a SpinBound, got {bound!r}")
    ticks, trap, bk = bound.key()
    return _spin_cached(
        kind,
        _norm(kind, params),
        ticks,
        trap,
        bk,
        None if abort is None else int(abort),
    )


def slots(kind, **params):
    """Slot indices by role (data, q, ctr, scratch, keep, ptr[, arg]) from the
    same expressions that number the $k operands (e.g. the PIPE drain must keep
    slots['keep'] alive)."""
    roles = _build(kind, _norm(kind, params), 1, True)[3]
    return {k: (list(v) if isinstance(v, list) else v) for k, v in roles.items()}
