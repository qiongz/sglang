# SPDX-License-Identifier: Apache-2.0
"""Pareto segment tables of push-comm
(configs/push_comm/<arch>_ar_tp<TP>_<toolchain>.segments.json): per
deployment profile (custom all-reduce / QuickReduce limits) and alignment
class, the (lo, hi] message-byte segments and the kernel serving each ("stock":
refused); a plan may name an identical earlier plan. A table is used only on
its arch and Triton version."""

from __future__ import annotations

import json
import os
from typing import Optional

import msgspec

CONFIG_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "configs",
    "push_comm",
)
TIERS = ("exact", "bf16", "int8", "int6", "int4")
QR_ARMS = {"FP": "qrfp", "INT8": "qr8", "INT6": "qr6", "INT4": "qr4"}
CLASSES = ("fast", "aligned", "unaligned", "padded")
# "..._qf": QuickReduce ahead of the custom all-reduce in its size range
PROFILE_KEYS = (
    ("qr_off",)
    + tuple(f"qr_{lv.lower()}{sfx}" for sfx in ("", "_bf16") for lv in QR_ARMS)
    + tuple(f"qr_{lv.lower()}{sfx}_qf" for sfx in ("", "_bf16") for lv in QR_ARMS)
)


class StockProfile(msgspec.Struct, frozen=True):
    """The framework's own all-reduce path by size (what a refused call runs):
    custom all-reduce on (ca_min, ca_max], QuickReduce at qr_level (None: off)
    on [qr_min, qr_max] (both 16-byte multiples), qr_first when QuickReduce
    wins the overlap."""

    ca_max: int = 64 << 20
    qr_level: Optional[str] = None
    qr_min: int = 0
    qr_max: int = 0
    ca_min: int = 0
    qr_first: bool = False

    def arm(self, nbytes):
        """The stock path of a bf16 message of nbytes: "ca", "qrfp" / "qr8" /
        "qr6" / "qr4" or "rccl"."""
        qr = (
            self.qr_level is not None
            and nbytes % 16 == 0
            and self.qr_min <= nbytes <= self.qr_max
        )
        if qr and self.qr_first:
            return QR_ARMS[self.qr_level]
        if nbytes % 16 == 0 and self.ca_min < nbytes <= self.ca_max:
            return "ca"
        if qr:
            return QR_ARMS[self.qr_level]
        return "rccl"

    def breaks(self):
        """Sizes after which arm() may change (as (lo, hi] segment edges)."""
        out = {self.ca_min, self.ca_max}
        if self.qr_level is not None:
            out |= {max(self.qr_min - 1, 0), self.qr_max}
        return out


def segments_table_name(arch, tp, toolchain):
    return f"{arch}_ar_tp{int(tp)}_{toolchain}.segments.json"


def validate(t):
    """Raise ValueError unless t is a well-formed segment table; returns t."""

    def bad(msg):
        raise ValueError(f"segment table: {msg}")

    for k in ("kind", "arch", "tp", "toolchain", "impls", "profiles", "plans"):
        if k not in t:
            bad(f"missing {k!r}")
    if t["kind"] != "ar_segments":
        bad(f"kind {t['kind']!r}")
    for n, im in t["impls"].items():
        if im.get("tier") not in TIERS:
            bad(f"impl {n}: tier {im.get('tier')!r}")
    for key in t["profiles"]:
        if key not in PROFILE_KEYS:
            bad(f"unknown profile {key!r}")
    for key, segs in t["plans"].items():
        prof, _, cls = key.partition("/")
        if prof not in t["profiles"] or cls not in CLASSES:
            bad(f"plan {key!r}: unknown profile or class")
        prev = 0
        for sg in segs:
            if int(sg["lo"]) != prev or int(sg["hi"]) <= prev:
                bad(f"plan {key!r}: segments must tile (0, top] in order")
            if sg["head"] != "stock" and sg["head"] not in t["impls"]:
                bad(f"plan {key!r}: unknown head {sg['head']!r}")
            prev = int(sg["hi"])
    for prof in t["profiles"]:
        for cls in CLASSES:
            if f"{prof}/{cls}" not in t["plans"]:
                bad(f"missing plan {prof}/{cls}")
    return t


def load_segments(arch, tp, toolchain, path=None):
    """The segment table of (arch, TP, toolchain), or None if none ships."""
    p = path or os.path.join(CONFIG_DIR, segments_table_name(arch, tp, toolchain))
    if not os.path.exists(p):
        return None
    with open(p) as f:
        t = json.load(f)
    if (t.get("arch"), t.get("tp"), t.get("toolchain")) != (arch, int(tp), toolchain):
        raise ValueError(
            f"{p} is for {t.get('arch')} / TP{t.get('tp')} / {t.get('toolchain')}, not {arch} / TP{tp} / {toolchain}"
        )
    plans = t.get("plans", {})
    for key, segs in plans.items():
        if isinstance(segs, str):
            plans[key] = plans[segs]
    return validate(t)


# the geometry keys a table impl may carry
_IMPL_KEYS = {
    "one_shot": {"NSEND"},
    "two_shot": {"F", "NSEND", "RA", "NSLOT", "PIPE", "SDLY", "SDG"},
    "two_shot_window": set(),
    "two_shot_epoch": {"window", "ldspad", "f", "npc", "xs", "wire"},
}
WIRE_TIER = {"c12": "exact", "xe8": "int8"}


def usable_impl(name, impl):
    """Can this module run the table impl (known kind, keys and wire)?"""
    if impl["kind"] not in _IMPL_KEYS:
        return False
    geom = impl.get("geom") or {}
    if set(geom) - _IMPL_KEYS[impl["kind"]]:
        return False
    if impl["kind"] == "two_shot_epoch":
        return WIRE_TIER.get(geom.get("wire", "c12")) == impl["tier"]
    return impl["tier"] == "exact"


def profile_key(t, profile):
    """The table profile a deployment uses: same QuickReduce level, order and
    minimum size, else the same level and order, else QuickReduce off."""
    profile = profile or StockProfile()
    best = None
    for key, p in t["profiles"].items():
        if p["qr_level"] != profile.qr_level:
            continue
        if bool(p.get("qr_first", False)) != (
            profile.qr_first and profile.qr_level is not None
        ):
            continue
        if p["qr_min"] == profile.qr_min:
            return key
        best = best or key
    return best or "qr_off"


def plain_segments(t, profile=None, cls="fast", usable=usable_impl):
    """Segments {lo, hi, algo ("stock": refused), H, geom, name} of a table
    for a stock profile and alignment class; sizes whose stock path differs
    from the table's profile are refused."""
    profile = profile or StockProfile()
    key = profile_key(t, profile)
    ref = StockProfile(**t["profiles"][key])
    edges = sorted(ref.breaks() | profile.breaks())
    out = []

    def put(lo, hi, head):
        if head == "stock":
            sg = dict(lo=lo, hi=hi, algo="stock", H=0, geom={}, name="stock")
        else:
            im = t["impls"][head]
            sg = dict(
                lo=lo,
                hi=hi,
                algo=im["kind"],
                H=int(im["width"]),
                geom=dict(im.get("geom") or {}),
                name=head,
            )
        if out and out[-1]["name"] == sg["name"] and out[-1]["hi"] == lo:
            out[-1]["hi"] = hi
        else:
            out.append(sg)

    for sg in t["plans"][f"{key}/{cls}"]:
        lo, hi, head = int(sg["lo"]), int(sg["hi"]), sg["head"]
        if head != "stock" and not usable(head, t["impls"][head]):
            head = "stock"
        cuts = [lo] + [e for e in edges if lo < e < hi] + [hi]
        for a, b in zip(cuts, cuts[1:]):
            h = head
            if h != "stock" and cls != "unaligned":
                # every aligned size of (a, b] has one stock path per profile
                x = b - b % 16
                if x > a and profile.arm(x) != ref.arm(x):
                    h = "stock"
            put(a, b, h)
    return out
