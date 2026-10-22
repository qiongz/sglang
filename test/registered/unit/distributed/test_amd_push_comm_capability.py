"""CPU tests of amd-push-comm (no GPU, no HIP call): protocol helpers,
the spin-asm generator, the selection tables and rules, the guard and refusal
paths, the workspace layout and the failure monitor."""

import json
import os
import signal
import struct
import subprocess
import sys
import time
import uuid
from types import SimpleNamespace

import pytest
import torch

from sglang.kernels.ops.communication.amd_push_comm import asmgen
from sglang.kernels.ops.communication.amd_push_comm import protocol as P
from sglang.kernels.ops.communication.amd_push_comm import table
from sglang.srt.distributed.device_communicators import amd_push_comm as R
from sglang.srt.distributed.device_communicators import amd_push_comm_monitor as M
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=20, suite="base-a-test-cpu")

ARCH, TC = "gfx950", "triton-3.7.0"
MIB = 1 << 20


def _gluon():
    try:
        from triton.experimental import gluon  # noqa: F401

        return True
    except Exception:
        return False


# ------------------------------------------------------------------ protocol
def test_spin_bound():
    b = P.SpinBound()
    assert (b.timeout_s, b.trap, b.backoff) == (300.0, True, 0)
    assert b.key() == (30_000_000_000, True, 0)
    with pytest.raises(ValueError):
        P.SpinBound(timeout_s=0)
    with pytest.raises(ValueError):
        P.SpinBound(backoff=-1)


def test_diag_constexprs():
    d = P.Diag(abort=0x1000, base=0x2000, inst=7)
    assert d.kw() == dict(DIAG=1, DIAG_BASE=0x2000, INST=7, TRAP=0)
    assert P.Diag(abort=0x1000, base=0x2000, trap=True).kw()["TRAP"] == 1
    assert P.FAILED == P.SEQ_MISMATCH < P.ABORTED < P.LIMIT


def test_timed_tail():
    t = P.timed_tail_lines("$4", 30_000_000_000, True)
    assert t[-1] == "s_trap 2" and f"s_mov_b32 $4, {P.LIMIT}" in t
    assert not any("s_load_dword" in ln for ln in t)
    a = P.timed_tail_lines("$4", 100, False, abort=0x7F00ABC00000, exit_label="4")
    assert (
        "s_trap 2" not in a and "s_branch 4f" in a and f"s_mov_b32 $4, {P.ABORTED}" in a
    )
    assert "s_load_dword s40, s[40:41], 0x0 glc" in a
    assert P.timed_tail_lines("$4", 100, False, bk=64)[:2] == [
        "s_cmp_lt_u32 $4, 64",
        "s_cbranch_scc1 6f",
    ]


# -------------------------------------------------------------------- asmgen
_KINDS = (
    [("rs", dict(nsrc=n)) for n in (2, 4, 8)]
    + [("rs_pipe", dict(nsrc=n)) for n in (2, 4, 8)]
    + [("ag", dict(n=n, nq=q)) for n in (1, 2, 7, 21) for q in (0, 1)]
    + [("ag_pipe", dict(n=n, nq=q)) for n in (1, 2) for q in (0, 1)]
    + [("flag_eq", {}), ("flag_ge", {})]
)


@pytest.mark.parametrize("kind, params", _KINDS)
@pytest.mark.parametrize("abort", [None, 0x7F00ABC00000])
def test_spin_products_are_checked(kind, params, abort):
    asm, cons, nout, n_in = asmgen.spin(
        kind, bound=P.SpinBound(trap=abort is None), abort=abort, **params
    )
    asmgen.check(asm, cons, nout, n_in)
    roles = asmgen.slots(kind, **params)
    outs = (
        roles["data"]
        + roles.get("q", [])
        + [roles["ctr"]]
        + roles["scratch"]
        + roles.get("keep", [])
    )
    assert sorted(outs) == list(range(nout))
    assert cons.endswith(P.TCLOB)
    assert ("s_trap 2" in asm) == (abort is None)


@pytest.mark.parametrize(
    "kind, params", [("rs", dict(nsrc=8)), ("ag", dict(n=2, nq=1)), ("flag_ge", {})]
)
def test_abort_changes_only_the_tail(kind, params):
    a0 = asmgen.spin(kind, bound=P.SpinBound(trap=False), **params)[0].split("\n")
    a1 = asmgen.spin(
        kind, bound=P.SpinBound(trap=False), abort=0x7F00ABC00000, **params
    )[0].split("\n")
    i = next(k for k, ln in enumerate(a0) if "s_and_b32 s40" in ln)
    assert a0[: i + 3] == a1[: i + 3]  # the polling loop is unchanged


def test_spin_bound_changes_the_deadline():
    a = asmgen.spin("rs", nsrc=4)[0]
    b = asmgen.spin("rs", nsrc=4, bound=P.SpinBound(timeout_s=480))[0]
    assert a != b and f"{48_000_000_000 & 0xFFFFFFFF}" in b


def test_check_catches_drift():
    asm, cons, nout, n_in = asmgen.spin("ag", n=2, nq=1)
    ents = cons.split(",")
    with pytest.raises(asmgen.AsmDriftError):  # an input dropped from the constraints
        asmgen.check(asm, ",".join(ents[:nout] + ents[nout + 1 :]), nout, n_in - 1)
    with pytest.raises(asmgen.AsmDriftError):  # a cached poll never terminates
        asmgen.check(asm.replace("off sc0 sc1", "off", 1), cons, nout, n_in)
    with pytest.raises(asmgen.AsmDriftError):  # the timed tail's SGPRs not clobbered
        asmgen.check(asm, cons.replace(",~{s43}", ""), nout, n_in)
    with pytest.raises(asmgen.AsmDriftError):  # a branch leaving the block
        asmgen.check(
            asm.replace("s_cbranch_vccz 2f", "s_cbranch_vccz 7f"), cons, nout, n_in
        )
    with pytest.raises(ValueError):
        asmgen.spin("rs", nsrc=3)
    with pytest.raises(TypeError):
        asmgen.spin("ag", n=2, slp=1)


# -------------------------------------------------------------------- tables
@pytest.mark.skipif(not _gluon(), reason="needs Triton Gluon")
@pytest.mark.parametrize("tp", [2, 4, 8])
def test_table_geometries_are_legal(tp):
    """Every impl of the shipped segment table is a legal geometry (two_shot
    with room for the call tags), under every profile."""
    from sglang.kernels.ops.communication.amd_push_comm import one_shot, two_shot

    t = table.load_segments(ARCH, tp, TC)
    seen = 0
    for key, prof in t["profiles"].items():
        p = R._PlainAR(tp, 512 * MIB, t, table.StockProfile(**prof))
        for sp in p.specs:
            h, g = sp.hidden, sp.geom
            if sp.kind == "two_shot":
                two_shot.check_cfg(tp, h, g)
                two_shot.check_cfg(tp, h, g, tag=True)
            elif sp.kind == "one_shot":
                assert h % 256 == 0 and ((h // 256 + 3) // 4) * (tp - 1) <= 32
                assert one_shot.ctr_words(tp, h, 1, **g) > 0
            elif sp.kind == "two_shot_window":
                assert h % (tp * 1024) == 0
            else:
                nk = h // 256 // tp
                assert h % (tp * 256) == 0 and nk % g.get("red", nk) == 0
                assert 8 % g.get("f", 8) == 0
                assert g.get("wire", "c12") in ("c12", "int8")
            seen += 1
    assert seen


def _seg_table(world=4, plans=None, impls=None):
    """A small segment table: the same plan under every profile unless given."""
    impls = impls or {
        "one": dict(kind="one_shot", tier="exact", width=1024, geom=dict(NSEND=2)),
        "two": dict(
            kind="two_shot", tier="exact", width=4096, geom=dict(F=4, NSEND=world)
        ),
        "epoch": dict(kind="two_shot_epoch", tier="exact", width=4096, geom=dict(f=8)),
    }

    def segs(*rows):
        out, lo = [], 0
        for hi, head in rows:
            out.append(dict(lo=lo, hi=hi, head=head, bound="x", why="test"))
            lo = hi
        return out

    default = dict(
        fast=segs((1024, "one"), (64 << 10, "two"), (MIB, "stock"), (8 * MIB, "epoch")),
        aligned=segs(
            (1024, "one"), (64 << 10, "two"), (MIB, "stock"), (8 * MIB, "epoch")
        ),
        unaligned=segs((1024, "one"), (MIB, "two"), (8 * MIB, "epoch")),
        padded=segs((8 * MIB, "epoch")),
    )
    profiles = {}
    for key in table.PROFILE_KEYS:
        lv = None if key == "qr_off" else key.split("_")[1].upper()
        profiles[key] = dict(
            qr_level=lv,
            qr_min=0 if lv is None else (2 * MIB if "_bf16" in key else MIB),
            qr_max=0 if lv is None else 2 << 30,
            ca_min=0,
            ca_max=64 * MIB,
        )
        if key.endswith("_qf"):
            profiles[key]["qr_first"] = True
    t = dict(
        kind="ar_segments",
        arch="gfx950",
        tp=world,
        toolchain="triton-3.7.0",
        source="test",
        impls=impls,
        profiles=profiles,
        plans={
            f"{k}/{c}": (plans or {}).get(f"{k}/{c}", default[c])
            for k in profiles
            for c in table.CLASSES
        },
    )
    return table.validate(t)


def test_stock_profile():
    p = table.StockProfile()
    assert (p.arm(1024), p.arm(1026), p.arm(128 * MIB)) == ("ca", "rccl", "rccl")
    q = table.StockProfile(qr_level="INT8", qr_min=8 * MIB, qr_max=2 << 30)
    assert (q.arm(4 * MIB), q.arm(128 * MIB)) == ("ca", "qr8")
    assert table.StockProfile(ca_min=1 * MIB).arm(1 * MIB) == "rccl"
    # QuickReduce ahead of the custom all-reduce in its range
    q = table.StockProfile(
        qr_level="INT8", qr_min=8 * MIB, qr_max=2 << 30, qr_first=True
    )
    assert (q.arm(4 * MIB), q.arm(8 * MIB), q.arm(128 * MIB)) == ("ca", "qr8", "qr8")


def test_segment_table_validation():
    t = _seg_table()
    for mutate in (
        lambda t: t.update(kind="ar_bytes"),
        lambda t: t["plans"].pop("qr_off/padded"),
        lambda t: t["plans"]["qr_off/fast"][1].update(lo=5),  # a gap
        lambda t: t["plans"]["qr_off/fast"][0].update(head="nope"),
        lambda t: t["impls"]["one"].update(tier="int5"),
        lambda t: t["profiles"].update(qr_int3={}),
    ):
        bad = json.loads(json.dumps(t))
        mutate(bad)
        with pytest.raises(ValueError):
            table.validate(bad)


def test_profile_key():
    t = _seg_table()
    assert table.profile_key(t, None) == "qr_off"
    q = table.StockProfile(qr_level="INT8", qr_min=MIB, qr_max=2 << 30)
    assert table.profile_key(t, q) == "qr_int8"
    q = table.StockProfile(qr_level="INT8", qr_min=2 * MIB, qr_max=2 << 30)
    assert table.profile_key(t, q) == "qr_int8_bf16"
    q = table.StockProfile(qr_level="INT4", qr_min=3 * MIB, qr_max=2 << 30)
    assert table.profile_key(t, q).startswith("qr_int4")  # same level
    q = table.StockProfile(qr_level="INT8", qr_min=MIB, qr_max=2 << 30, qr_first=True)
    assert table.profile_key(t, q) == "qr_int8_qf"
    # a table without the QuickReduce-first profiles: QuickReduce off, every
    # size whose stock path differs from it is refused
    for k in [k for k in t["profiles"] if k.endswith("_qf")]:
        del t["profiles"][k]
    assert table.profile_key(t, q) == "qr_off"
    segs = table.plain_segments(t, q)
    assert all(
        s["algo"] == "stock" for s in segs if s["hi"] > MIB and s["lo"] < 64 * MIB
    )


def test_off_profile_sizes_are_refused():
    """A deployment whose stock limits differ from the table's profile refuses
    the sizes whose stock path differs; everything else is as in the table."""
    t = _seg_table()

    def heads(profile, cls="fast"):
        return [
            (s["lo"], s["hi"], s["name"]) for s in table.plain_segments(t, profile, cls)
        ]

    assert heads(None) == [
        (0, 1024, "one"),
        (1024, 65536, "two"),
        (65536, MIB, "stock"),
        (MIB, 8 * MIB, "epoch"),
    ]
    # the custom all-reduce only up to 4 MiB: above it the stock path is RCCL,
    # which the table did not compare against there
    off = table.plain_segments(t, table.StockProfile(ca_max=4 * MIB))
    assert [s["name"] for s in off if s["lo"] >= 4 * MIB] == ["stock"]
    assert [s["name"] for s in off if s["hi"] <= 4 * MIB] == [
        "one",
        "two",
        "stock",
        "epoch",
    ]
    # custom all-reduce from 1 KiB on: the smallest sizes run RCCL instead
    off = table.plain_segments(t, table.StockProfile(ca_min=1024))
    assert off[0]["name"] == "stock" and off[0]["hi"] == 1024
    # unaligned sizes: RCCL under every profile, nothing to refuse
    assert heads(table.StockProfile(ca_max=4 * MIB), "unaligned") == heads(
        None, "unaligned"
    )


def test_lossy_wire_only_under_its_profile():
    impls = dict(
        _seg_table()["impls"],
        i8=dict(kind="two_shot_epoch", tier="int8", width=4096, geom=dict(wire="int8")),
    )
    plans = {}
    for c in table.CLASSES:
        plans[f"qr_int8/{c}"] = [
            dict(lo=0, hi=MIB, head="two", bound="x", why="t"),
            dict(lo=MIB, hi=8 * MIB, head="i8", bound="x", why="t"),
        ]
    t = _seg_table(plans=plans, impls=impls)
    q = table.StockProfile(qr_level="INT8", qr_min=MIB, qr_max=2 << 30)
    segs = table.plain_segments(t, q)
    assert segs[-1]["name"] == "i8" and segs[-1]["geom"] == dict(wire="int8")
    for p in (None, table.StockProfile(qr_level="FP", qr_min=MIB, qr_max=2 << 30)):
        assert "i8" not in {s["name"] for s in table.plain_segments(t, p)}


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_segment_tables_ship(tp):
    t = table.load_segments(ARCH, tp, TC)
    assert t is not None and t["kind"] == "ar_segments"
    # 9 profiles with the custom all-reduce first, 8 with QuickReduce first
    assert set(t["profiles"]) == set(table.PROFILE_KEYS)
    for key, prof in t["profiles"].items():
        assert bool(prof.get("qr_first")) == key.endswith("_qf"), key
    assert all(table.usable_impl(n, im) for n, im in t["impls"].items())
    for key in t["profiles"]:
        for cls in table.CLASSES:
            segs = t["plans"][f"{key}/{cls}"]
            for s in segs:
                assert s["bound"] and s["why"]
                if s["head"] != "stock" and s.get("worst") is not None:
                    assert s["worst"] <= 1 + s["tol"] + 1e-6, (tp, key, cls, s)
                    # a lossy push wire only where the profile's floor allows it
                    if t["impls"][s["head"]]["tier"] == "int8":
                        assert key.startswith(("qr_int8", "qr_int6", "qr_int4"))
    algos = {s["algo"] for s in table.plain_segments(t)}
    assert (
        {"one_shot", "two_shot"}
        <= algos
        <= {"one_shot", "two_shot", "two_shot_epoch", "two_shot_window", "stock"}
    )
    assert table.load_segments(ARCH, tp, "triton-3.6.0") is None
    padded = table.plain_segments(t, cls="padded")
    assert padded and {s["algo"] for s in padded} <= {
        "two_shot_epoch",
        "two_shot_window",
        "stock",
    }
    assert not any("points" in k or k in ("aa", "points_u", "points_p") for k in t), (
        "no measurements in the shipped table"
    )


def test_masked_tail_stays_in_its_segment():
    """A one_shot / two_shot size served by the masked tail is picked by its own
    bytes, whatever segment its padded size would fall in."""
    t = table.load_segments(ARCH, 4, TC)
    p = R._PlainAR(4, 64 * MIB, t)
    for sp in p.specs:
        sp.inst = SimpleNamespace(
            max_tokens=sp.rows,
            kind=sp.kind,
            can_tail=sp.kind in ("one_shot", "two_shot", "two_shot_window"),
        )
    p.bind()
    n_checked = 0
    for sg in p.segs_u:
        if sg["algo"] in ("one_shot", "two_shot", "two_shot_window"):
            # near the top of the segment: unaligned, a partial last row
            n = sg["hi"] // 2 - 1
            r = p._pick(torch.empty(n, dtype=torch.bfloat16))
            assert r is not None and r[0].kind == sg["algo"], (sg, r)
            n_checked += 1
    assert n_checked


@pytest.mark.parametrize("tp", [2, 4, 8])
def test_default_max_bytes_serves_the_whole_table(tp):
    """The default max_bytes serves every push segment of the shipped table;
    the IPC memory stops growing at the windowed instances."""
    import inspect

    default = inspect.signature(R.AmdPushCommunicator).parameters["max_bytes"].default
    assert default == 512 * MIB
    t = table.load_segments(ARCH, tp, TC)
    p = R._PlainAR(tp, default, t)
    for sp in p.specs:  # stand-ins for the instances: only their row capacity matters
        sp.inst = SimpleNamespace(max_tokens=sp.rows, hidden=sp.hidden, kind=sp.kind)
    p.bind()
    big = 0
    for segs, adj in ((p.segs, 0), (p.segs_q, -8), (p.segs_u, -1)):
        for s in segs:
            if s["algo"] == "stock" or s["hi"] <= 64 * MIB:
                continue
            mid = (max(s["lo"], 64 * MIB) + s["hi"]) // 4  # elements, mid-segment
            n = mid // 4096 * 4096 + adj
            assert 2 * n > max(s["lo"], 64 * MIB)
            r = p._pick(torch.empty(n, dtype=torch.bfloat16))
            assert r is not None and r[0].kind == s["algo"], (tp, s["name"], n)
            big += 1
    assert big and p.max_bytes == 512 * MIB

    def ipc(p):
        return sum(
            R._sizes(sp.kind, tp, sp.hidden, sp.rows, sp.geom, False)[0]
            for sp in p.specs
        )

    assert ipc(p) == ipc(R._PlainAR(tp, 128 * MIB, t))


@pytest.mark.parametrize(
    "kind, tier, geom, ok",
    [
        ("two_shot", "exact", dict(F=4, NSEND=8), True),
        ("two_shot", "exact", dict(F=2, PIPE=1, SDLY=10, SDG=8), True),
        ("two_shot_epoch", "exact", dict(window=1024, f=8, ldspad=0), True),
        ("two_shot_epoch", "exact", dict(wire="c12", npc=24, xs=2), True),
        ("two_shot_epoch", "int8", dict(wire="int8"), True),
        ("two_shot_window", "exact", {}, True),
        # knobs, wires and kinds this module does not have
        ("two_shot", "exact", dict(F=4, LDSPAD=10240), False),
        ("two_shot", "exact", dict(F=4, RW=4), False),
        ("two_shot_epoch", "exact", dict(fpipe=1), False),
        ("two_shot_epoch", "exact", dict(wire="bf16"), False),
        ("two_shot_epoch", "int4", dict(wire="int4"), False),
        ("two_shot_epoch", "int8", dict(wire="int4"), False),
        ("two_shot_window", "exact", dict(fs=2), False),
        ("two_shot_window", "int8", {}, False),
        ("lc13", "exact", {}, False),
        # the int8 wire: tier and wire must agree
        ("two_shot_epoch", "int8", {}, False),
        ("two_shot_epoch", "exact", dict(wire="int8"), False),
    ],
)
def test_usable_impls(kind, tier, geom, ok):
    im = dict(kind=kind, tier=tier, width=4096, geom=geom)
    assert table.usable_impl("a", im) == ok


def _plain(world=4):
    p = R._PlainAR(world, 8 * MIB, _seg_table(world))
    for sp in p.specs:  # stand-ins for the instances: only their row capacity matters
        sp.inst = SimpleNamespace(max_tokens=sp.rows, hidden=sp.hidden, kind=sp.kind)
    p.bind()
    return p


def test_plain_selection_and_padding():
    p = _plain()
    assert [(s["lo"], s["hi"], s["name"]) for s in p.segs] == [
        (0, 1024, "one"),
        (1024, 65536, "two"),
        (65536, MIB, "stock"),
        (MIB, 8 * MIB, "epoch"),
    ]
    assert [s["name"] for s in p.segs_u] == ["one", "two", "epoch"]

    def pick(n):
        r = p._pick(torch.empty(n, dtype=torch.bfloat16))
        return None if r is None else (r[0].kind, r[1], r[2])

    # one_shot / two_shot: a partial last row runs the masked tail (the padded
    # count is the rows the kernel covers, nothing is copied)
    assert pick(512) == ("one_shot", 1024, 1024)
    assert pick(1) == ("one_shot", 1024, 1024)
    assert pick(4096) == ("two_shot", 4096, 0)  # divides: no padding
    assert pick(30000) == ("two_shot", 4096, 32768)
    assert pick(40000) is None  # aligned 80000 B: the stock path is the choice
    assert pick(40001) == ("two_shot", 4096, 40960)  # unaligned: its own plan
    assert pick(3_000_000) == ("two_shot_epoch", 4096, 3_002_368)
    # two_shot_epoch: no other widths, a size its width does not divide is
    # zero-padded
    assert pick(MIB + 1024) == ("two_shot_epoch", 4096, MIB + 4096)
    assert pick(5 * MIB) is None  # above the table
    # a size just below the top whose padding passes it: the instance holds
    # whole rows of the top
    top = (8 * MIB) // 2
    assert pick(top - 1) == ("two_shot_epoch", 4096, top)
    assert p._pick(torch.empty(64, dtype=torch.float32)) is None
    assert p._pick(torch.empty(8, 8, dtype=torch.bfloat16).t()) is None


# ---------------------------------------------------------- guard / refusals
def test_platform_guard(monkeypatch):
    R.amd_push_comm_platform.cache_clear()
    monkeypatch.setattr(R, "is_hip", lambda: False)
    assert R.amd_push_comm_platform() == (None, None, "not ROCm")
    R.amd_push_comm_platform.cache_clear()
    monkeypatch.setattr(R, "is_hip", lambda: True)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 0)
    for arch, ok in (
        ("gfx942:sramecc+:xnack-", False),
        ("gfx1250", False),
        ("gfx950:sramecc+:xnack-", True),
    ):
        monkeypatch.setattr(
            torch.cuda,
            "get_device_properties",
            lambda d, a=arch: SimpleNamespace(gcnArchName=a),
        )
        R.amd_push_comm_platform.cache_clear()
        a, tc, why = R.amd_push_comm_platform()
        if not ok:
            assert a is None and "arch" in why
        elif _gluon():
            import triton

            want = triton.__version__ in R.SUPPORTED_TRITON
            assert (a == "gfx950") == want and (
                tc == f"triton-{triton.__version__}"
            ) == want
    R.amd_push_comm_platform.cache_clear()


def test_disabled_communicator_refuses(monkeypatch):
    monkeypatch.setattr(R.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(R.dist, "get_world_size", lambda group=None: 4)
    monkeypatch.setattr(R, "amd_push_comm_platform", lambda: (None, None, "test"))
    comm = R.AmdPushCommunicator(object(), torch.device("cpu"))
    x = torch.zeros(4, 6144, dtype=torch.bfloat16)
    assert comm.disabled
    assert comm.all_reduce(x) is None
    assert not comm.should_push_ar(x)
    comm.close()
    with pytest.raises(ValueError):
        R.AmdPushCommunicator(object(), torch.device("cpu"), seq_check=True)


def test_unsupported_world_size(monkeypatch):
    monkeypatch.setattr(R.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(R.dist, "get_world_size", lambda group=None: 3)
    monkeypatch.setattr(R, "amd_push_comm_platform", lambda: ("gfx950", TC, ""))
    comm = R.AmdPushCommunicator(object(), torch.device("cpu"))
    assert (
        comm.disabled
        and comm.all_reduce(torch.zeros(1, 6144, dtype=torch.bfloat16)) is None
    )


class _Stream:
    def __init__(self, name, waits):
        self.name, self.waits = name, waits

    def wait_stream(self, other):
        self.waits.append((self.name, other.name))

    def __eq__(self, o):
        return isinstance(o, _Stream) and o.name == self.name

    def __hash__(self):
        return hash(self.name)


def _guarded(monkeypatch, guard):
    """A communicator with only the call guards, on fake CUDA streams."""
    import threading

    st = SimpleNamespace(capturing=False, cur=None, waits=[], runs=[])
    st.dflt = _Stream("default", st.waits)
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: st.capturing)
    monkeypatch.setattr(torch.cuda, "current_stream", lambda device=None: st.cur)
    monkeypatch.setattr(torch.cuda, "default_stream", lambda device=None: st.dflt)
    monkeypatch.setattr(
        torch._C,
        "_cuda_getCurrentRawStream",
        lambda index: hash(st.cur.name),
        raising=False,
    )
    comm = R.AmdPushCommunicator.__new__(R.AmdPushCommunicator)
    comm.disabled = comm.closed = comm.compile_guard = False
    comm.device, comm.mon, comm.site_box = torch.device("cpu"), None, [0]
    comm.stream_guard, comm._last_stream, comm._last_raw = guard, None, None
    comm._dev_index = 0
    comm._lock = threading.Lock()
    comm.plain = SimpleNamespace(
        supports=lambda x: True, all_reduce=lambda x: st.runs.append(1) or x
    )
    return comm, st


def test_stream_guard_serial_orders_any_stream(monkeypatch):
    comm, st = _guarded(monkeypatch, "serial")
    x = torch.zeros(4)
    a, b = _Stream("A", st.waits), _Stream("B", st.waits)
    for s in (a, a, b, a, st.dflt):
        st.cur = s
        assert comm.all_reduce(x) is x
    assert st.waits == [("B", "A"), ("A", "B"), ("default", "A")]
    # graph capture neither waits nor records its stream
    st.capturing, st.cur = True, b
    assert comm.all_reduce(x) is x
    assert len(st.waits) == 3 and comm._last_stream == st.dflt


def test_stream_guard_serial_refuses_a_concurrent_thread(monkeypatch):
    comm, st = _guarded(monkeypatch, "serial")
    x = torch.zeros(4)
    st.cur = _Stream("A", st.waits)
    with comm._lock:  # another thread is inside a call
        assert not comm.should_push_ar(x)
        assert comm.all_reduce(x) is None and st.runs == []
    assert comm.all_reduce(x) is x


def test_stream_guard_default_and_bad_value(monkeypatch):
    comm, st = _guarded(monkeypatch, "default")
    x = torch.zeros(4)
    st.cur = _Stream("A", st.waits)
    assert comm.all_reduce(x) is None
    st.capturing = True
    assert comm.all_reduce(x) is x
    st.capturing, st.cur = False, st.dflt
    assert comm.all_reduce(x) is x and st.waits == []
    monkeypatch.setattr(R.dist, "get_rank", lambda group=None: 0)
    monkeypatch.setattr(R.dist, "get_world_size", lambda group=None: 4)
    with pytest.raises(ValueError):
        R.AmdPushCommunicator(object(), torch.device("cpu"), stream_guard=True)


def test_workspace_layout():
    ws = R.AmdPushCommWorkspace(None, torch.device("cpu"), 0, 4)
    ws.reserve("a", 100, counters=10)
    ws.reserve("b", 3 << 20, counters=70)
    ws.reserve("c", 0)
    lay, total = ws.layout()
    assert lay == {"a": (0, 100), "b": (2 << 20, 3 << 20), "c": (6 << 20, 0)}
    assert total == 6 << 20
    with pytest.raises(AssertionError):
        ws.reserve("a", 1)


# ------------------------------------------------------------------- monitor
def _record(page, slot, it, rank, inst, kind, role, row, cnt):
    struct.pack_into(
        "<8i",
        page.mm,
        M.OFF_DIAG + M.DIAG_REC * slot,
        0,
        it,
        rank,
        inst,
        kind,
        role,
        row,
        cnt,
    )
    struct.pack_into("<i", page.mm, M.OFF_DIAG + M.DIAG_REC * slot, P.DIAG_MAGIC)


class _Pages:
    def __init__(self, world, pids):
        tag = f"test.{uuid.uuid4().hex[:8]}"
        self.own = [
            M.SharedPage(M.page_path(tag, r), create=True) for r in range(world)
        ]
        for p, pid in zip(self.own, pids):
            struct.pack_into("<i", p.mm, M.OFF_PID, pid)

    def close(self):
        for p in self.own:
            p.close(unlink=True)


def test_shared_page_layout():
    pages = _Pages(2, [os.getpid()] * 2)
    try:
        p = pages.own[0]
        assert (p.abort, p.pid) == (0, os.getpid())
        p.set_abort(M.ABORT_PEER_DEAD, 1, 7)
        p.set_abort(M.ABORT_USER, 0, 9)  # first writer wins
        assert p.abort_info()["reason"] == M.ABORT_PEER_DEAD
        p.set_inst(3, 2, 6144, 0x1234)
        assert p.inst(3) == dict(kind="two_shot", hidden=6144, geom=0x1234)
        _record(p, 200, P.ABORTED, 0, 3, 2, 5, 17, 42)
        r = p.records()
        assert len(r) == 1 and (r[0]["reason"], r[0]["row"], r[0]["cnt"]) == (
            "abort",
            17,
            42,
        )
        assert r[0]["slot"] == 200
        # the ring the kernels write (primitives.diag_report: 16 int32 per slot)
        assert M.OFF_DIAG % 64 == 0 and M.OFF_DIAG + M.DIAG_REC * P.DIAG_SLOTS <= M.PAGE
        assert M.OFF_INST + 16 * M.INST_MAX <= M.OFF_DIAG
    finally:
        pages.close()


@pytest.fixture
def peer():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    pages = _Pages(3, [os.getpid(), child.pid, os.getpid()])
    yield child, pages
    if child.poll() is None:
        os.kill(child.pid, signal.SIGKILL)
        child.wait()
    pages.close()


def _monitor(pages, fired):
    views = [M.SharedPage(p.path) for p in pages.own]
    return M.Monitor(
        0, views, [p.pid for p in views], on_fatal=fired.append, period=0.01
    )


def _wait(fired, t=2.0):
    t0 = time.monotonic()
    while not fired and time.monotonic() - t0 < t:
        time.sleep(0.002)


def test_monitor_peer_killed(peer):
    child, pages = peer
    fired = []
    m = _monitor(pages, fired)
    time.sleep(0.05)
    assert fired == []
    os.kill(child.pid, signal.SIGKILL)
    child.wait()
    _wait(fired)
    m.stop()
    assert len(fired) == 1 and fired[0]["culprit"] == 1
    assert (
        pages.own[0].abort == M.ABORT_PEER_DEAD
        and pages.own[2].abort == M.ABORT_PEER_ABORT
    )
    with pytest.raises(RuntimeError):
        m.check()


def test_monitor_stopped_peer_is_not_fatal(peer):
    child, pages = peer
    fired = []
    m = _monitor(pages, fired)
    os.kill(child.pid, signal.SIGSTOP)
    time.sleep(0.2)
    os.kill(child.pid, signal.SIGCONT)
    m.stop()
    assert fired == []
    m.check()


def test_monitor_own_record_and_peer_abort(peer):
    _, pages = peer
    fired = []
    m = _monitor(pages, fired)
    _record(pages.own[0], 5, P.LIMIT, 0, 1, 3, 9, 4, 11)
    _wait(fired)
    m.stop()
    assert fired[0]["pages"][0]["records"][0]["reason"] == "timeout"
    assert all(p.abort for p in pages.own)


# ------------------------------------------------------------ fast launch
class _Fn:
    arg_names = ["x", "out", "m", "site"]
    do_not_specialize = ["site"]
    params = []


def _launcher():
    from sglang.kernels.ops.communication.amd_push_comm import launch

    return launch, launch.FastLauncher(_Fn())


def test_launch_key_same_pointer_other_spec():
    _, fl = _launcher()
    base = torch.zeros(4096, dtype=torch.bfloat16)
    a = base[:1024]  # same data pointer, other numel
    b = base.view(torch.int16)[:2048]  # same data pointer, other dtype
    c = base[8:]  # 16-byte aligned, another pointer
    d = base[1:]  # unaligned
    ka, kb, kc, kd = (fl._key((t, t, 1, 3), "k") for t in (a, b, c, d))
    assert a.data_ptr() == b.data_ptr()
    assert ka != kb and ka == kc and ka != kd
    assert ka == fl._key((base, base, 1, 3), "k")  # numel is not specialized


def test_launch_trusted_key():
    launch, fl = _launcher()
    x = torch.zeros(6144, dtype=torch.bfloat16)
    out = torch.empty_like(x)
    for m in (1, 7, 16, 64, 2**31):
        args = (x, out, m, 5)
        assert fl._key(args, "k") == fl._key_trusted(args, "k", launch.TRUSTED_BF16)
    # only operands that are aligned and < 2 GiB may be vouched for
    d = torch.zeros(4096, dtype=torch.bfloat16)[1:]
    args = (d, d, 1, 3)
    assert fl._key(args, "k") != fl._key_trusted(args, "k", launch.TRUSTED_BF16)


def test_launch_bound_equals_cache_hit(monkeypatch):
    launch, fl = _launcher()

    class Ck:
        module, function, packed_metadata = object(), "fn", "meta"

        def __init__(self):
            self.runs = []

        def run(self, *a):
            self.runs.append(a)

    ck = Ck()
    x = torch.zeros(6144, dtype=torch.bfloat16)
    args = (x, x, 4, 9)
    fl.cache[fl._key(args, "k")] = (ck, ("T1", "T2"))
    monkeypatch.setattr(launch, "_no_hooks", lambda: True)
    monkeypatch.setattr(
        torch._C, "_cuda_getCurrentRawStream", lambda d: 1234, raising=False
    )
    monkeypatch.setattr(torch._C, "_cuda_getDevice", lambda: 0, raising=False)
    assert fl.bound((8,), (x, x, 1, 9), "k", launch.TRUSTED_BF16) is None  # m spec
    assert fl.bound((8,), args, "other", launch.TRUSTED_BF16) is None
    lf = fl.bound((8,), args, "k", launch.TRUSTED_BF16)
    y = torch.zeros(6144, dtype=torch.bfloat16)
    lf(y, y, 4, 11)
    fl(8, (y, y, 4, 11), {}, "k")
    assert len(ck.runs) == 2 and ck.runs[0] == ck.runs[1]
    assert ck.runs[0][:9] == (8, 1, 1, 1234, "fn", "meta", None, None, None)
    assert ck.runs[0][-3:] == (11, "T1", "T2")


def test_plain_binds_once_per_size():
    """The plain all-reduce binds a one_shot / two_shot launch once per element
    count, retries while the variant is not compiled, and never binds other
    kernels."""
    p = _plain()
    calls = []

    class Inst:
        fast, max_tokens = True, 1 << 20

        def __init__(self, ready):
            self.ready = ready

        def plain_launcher(self, n, tail, tspec):
            calls.append((n, tail))
            if not self.ready[0]:
                return None
            return lambda t: t + 1

    ready = [False]
    for sg in p.segs + p.segs_u + p.segs_q:
        if sg["algo"] in ("one_shot", "two_shot"):
            sg["inst"] = Inst(ready)
        elif sg["algo"] != "stock":
            sg["inst"] = SimpleNamespace(max_tokens=1 << 20, all_reduce=lambda t: t)
    p._picks.clear()
    x = torch.zeros(4096, dtype=torch.bfloat16)  # two_shot, whole rows
    assert p._bind(x) is R._MISS  # not compiled yet: bound later
    ready[0] = True
    run = p._bind(x)
    assert run is not None and run is not R._MISS
    assert torch.equal(run(x), x + 1)
    assert calls == [(4096, False), (4096, False)]
    assert p._bind(torch.zeros(3_000_000, dtype=torch.bfloat16)) is None  # c12
