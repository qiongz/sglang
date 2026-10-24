# SPDX-License-Identifier: Apache-2.0
"""push-comm: push all-reduce for AMD Instinct gfx950.

Per message-size segment a measured Pareto segment table selects its push
kernels or the deployment's path: PushCommunicator(group, device).all_reduce(x)
returns the result, or None where the table keeps the deployment's path. Guard:
ROCm, gfx950, Triton 3.7.0 with Gluon, world size 2 / 4 / 8 on one node.
Construction is collective; refusal depends only on rank-uniform facts.

With hidden_sizes, fused_allreduce_rmsnorm(x, residual, weight, eps) returns
(out, residual_out) of the all-reduce of x [m, H] + the residual add + RMSNorm
(the table's fused plans, by message bytes; H only decides which kernels can
run), or None."""

from __future__ import annotations

import ctypes
import logging
import threading
from functools import cache
from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch.distributed import ProcessGroup

from sglang.srt.utils import is_hip

logger = logging.getLogger(__name__)

SUPPORTED_ARCHS = ("gfx950",)
SUPPORTED_TRITON = ("3.7.0",)
SUPPORTED_WORLD_SIZES = (2, 4, 8)
MAX_I32 = (1 << 31) - 1


@cache
def push_comm_platform() -> Tuple[Optional[str], Optional[str], str]:
    """(arch, toolchain, reason): arch / toolchain are None when push-comm
    cannot run here; reason says why."""
    if not is_hip():
        return None, None, "not ROCm"
    try:
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        arch = props.gcnArchName.split(":")[0]
    except Exception as e:
        return None, None, f"no device ({e})"
    if arch not in SUPPORTED_ARCHS:
        return None, None, f"arch {arch} (supported: {SUPPORTED_ARCHS})"
    try:
        import triton
        from triton.experimental import gluon  # noqa: F401
    except Exception as e:
        return None, None, f"no Triton Gluon ({e})"
    if triton.__version__ not in SUPPORTED_TRITON:
        return (
            None,
            None,
            f"Triton {triton.__version__} (supported: {SUPPORTED_TRITON})",
        )
    return arch, f"triton-{triton.__version__}", ""


def push_comm_available() -> bool:
    return push_comm_platform()[0] is not None


# ---------------------------------------------------------------- HIP IPC
_HIP_UNCACHED = 0x3  # hipDeviceMallocUncached
_HANDLE = 64


class _Handle(ctypes.Structure):
    _fields_ = [("reserved", ctypes.c_ubyte * _HANDLE)]


class _Hip:
    def __init__(self):
        h = ctypes.CDLL("libamdhip64.so")
        h.hipExtMallocWithFlags.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_size_t,
            ctypes.c_uint,
        ]
        h.hipIpcGetMemHandle.argtypes = [ctypes.POINTER(_Handle), ctypes.c_void_p]
        h.hipIpcOpenMemHandle.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            _Handle,
            ctypes.c_uint,
        ]
        h.hipMemset.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t]
        h.hipDeviceSynchronize.argtypes = []
        h.hipIpcCloseMemHandle.argtypes = [ctypes.c_void_p]
        h.hipFree.argtypes = [ctypes.c_void_p]
        for f in (
            "hipExtMallocWithFlags",
            "hipIpcGetMemHandle",
            "hipIpcOpenMemHandle",
            "hipMemset",
            "hipDeviceSynchronize",
            "hipIpcCloseMemHandle",
            "hipFree",
        ):
            getattr(h, f).restype = ctypes.c_int
        self.h = h

    def check(self, st, what):
        if st != 0:
            raise RuntimeError(f"{what} failed: hip status {st}")

    def malloc_uncached(self, nbytes):
        p = ctypes.c_void_p()
        self.check(
            self.h.hipExtMallocWithFlags(ctypes.byref(p), nbytes, _HIP_UNCACHED),
            "hipExtMallocWithFlags",
        )
        self.check(self.h.hipMemset(p, 0, nbytes), "hipMemset")
        self.check(self.h.hipDeviceSynchronize(), "hipDeviceSynchronize")
        return p.value

    def export(self, ptr):
        hd = _Handle()
        self.check(
            self.h.hipIpcGetMemHandle(ctypes.byref(hd), ctypes.c_void_p(ptr)),
            "hipIpcGetMemHandle",
        )
        return bytes(hd.reserved)

    def open(self, hb):
        hd = _Handle()
        ctypes.memmove(ctypes.addressof(hd), hb, _HANDLE)
        p = ctypes.c_void_p()
        self.check(
            self.h.hipIpcOpenMemHandle(ctypes.byref(p), hd, 1), "hipIpcOpenMemHandle"
        )
        return p.value


class IpcBuffer:
    """One uncached buffer per rank (remote stores land in HBM; local polls
    never hit a stale line), opened by every peer. addrs[r]: rank r's buffer
    in this process; fixed for the process lifetime (graph safe)."""

    def __init__(self, nbytes, rank, world, group, device):
        self.hip = _Hip()
        self.nbytes, self.rank, self.world, self.device = (
            int(nbytes),
            rank,
            world,
            device,
        )
        self.local = self.hip.malloc_uncached(self.nbytes)
        handles = [None] * world
        dist.all_gather_object(handles, self.hip.export(self.local), group=group)
        self.addrs = [
            self.local if r == rank else self.hip.open(handles[r]) for r in range(world)
        ]
        self.ptrs = torch.tensor(self.addrs, dtype=torch.uint64, device=device)
        self.closed = False
        dist.barrier(group=group)

    def close(self, group=None, sync=True):
        """Unmap the peers' buffers and free ours; collective when group is
        given (barrier, close handles, barrier, free). Idempotent."""
        if self.closed:
            return
        if sync:
            torch.cuda.synchronize(self.device)
        if group is not None:
            dist.barrier(group=group)
        for r, a in enumerate(self.addrs):
            if r != self.rank:
                self.hip.check(
                    self.hip.h.hipIpcCloseMemHandle(ctypes.c_void_p(a)),
                    "hipIpcCloseMemHandle",
                )
        if group is not None:
            dist.barrier(group=group)
        self.hip.check(self.hip.h.hipFree(ctypes.c_void_p(self.local)), "hipFree")
        self.closed = True


class IpcSlice:
    """[off, off + nbytes) of an IpcBuffer on every rank."""

    def __init__(self, parent, off, nbytes):
        self.parent, self.off, self.nbytes = parent, int(off), int(nbytes)
        self.local = parent.local + self.off
        self.addrs = [a + self.off for a in parent.addrs]
        self.ptrs = torch.tensor(self.addrs, dtype=torch.uint64, device=parent.device)


def _up(n, a):
    return -(-int(n) // a) * a


class _Region:
    """take(nbytes) returns the reserved slice, once and for the reserved size."""

    def __init__(self, res, name):
        self.res, self.name = res, name

    def take(self, nbytes):
        r = self.res
        assert int(nbytes) == r["nbytes"], (
            f"workspace {self.name}: take({nbytes}), reserved {r['nbytes']}"
        )
        assert not r["taken"], f"workspace {self.name}: taken twice"
        r["taken"] = True
        return r["buf"]


class PushCommWorkspace:
    """The IPC memory, counter page and error page of one (group, device),
    shared by every kernel instance: reserve() at construction, then one
    collective seal(), then region() / counters() / err() per instance.
    """

    ALIGN = 2 << 20
    CTR_ALIGN = 64

    def __init__(self, group, device, rank, world):
        self.group, self.device, self.rank, self.world = group, device, rank, world
        self._res = {}
        self.sealed = self.closed = False
        self.buf = None
        self.ctr_page = self.err_page = None
        self.nbytes = 0

    def reserve(self, name, nbytes, counters=0):
        assert not self.sealed, f"workspace sealed: cannot reserve {name!r}"
        assert name not in self._res, f"workspace: {name!r} reserved twice"
        self._res[name] = dict(
            nbytes=int(nbytes),
            counters=int(counters),
            err_idx=len(self._res),
            buf=None,
            taken=False,
        )
        return name

    def layout(self):
        """{name: (offset, nbytes)} and the total IPC bytes (before seal)."""
        out, cur = {}, 0
        for name, r in self._res.items():
            off = _up(cur, self.ALIGN)
            out[name] = (off, r["nbytes"])
            cur = off + r["nbytes"]
        return out, _up(cur, self.ALIGN) if self._res else 0

    def seal(self):
        """Allocate (collective: every rank seals the same reservations in the
        same order). Idempotent."""
        if self.sealed:
            return self
        self.sealed = True
        lay, total = self.layout()
        if total:
            self.buf = IpcBuffer(total, self.rank, self.world, self.group, self.device)
            self.nbytes = total
            for n, r in self._res.items():
                r["buf"] = IpcSlice(self.buf, lay[n][0], r["nbytes"])
        cur = 0
        for r in self._res.values():
            r["ctr_off"] = cur
            cur += _up(r["counters"], self.CTR_ALIGN)
        self.ctr_page = torch.zeros(max(cur, 1), dtype=torch.int32, device=self.device)
        self.err_page = torch.zeros(
            max(len(self._res), 1), dtype=torch.int32, device=self.device
        )
        return self

    def region(self, name):
        """What the instance that reserved `name` receives as `ipc`."""
        assert self.sealed, "workspace not sealed"
        return _Region(self._res[name], name)

    def counters(self, name):
        r = self._res[name]
        return self.ctr_page[r["ctr_off"] : r["ctr_off"] + r["counters"]]

    def err(self, name):
        r = self._res[name]
        return self.err_page[r["err_idx"] : r["err_idx"] + 1]

    def err_max(self):
        """Max error word of every instance (a device tensor, no sync)."""
        return self.err_page.max().reshape(1)

    def close(self):
        """Free the IPC memory (collective; no kernel in flight). Idempotent."""
        if self.closed:
            return
        self.closed = True
        if self.buf is not None:
            self.buf.close(self.group)


def _aligned(t):
    """Copy an operand the kernels were not compiled for (16-byte aligned,
    storage < 2 GiB): a JIT inside a collective stalls every peer."""
    if t.data_ptr() % 16 == 0 and t.untyped_storage().size() <= MAX_I32:
        return t
    return t.clone()


def _kernels():
    """The kernel modules, imported on first use (they need Triton)."""
    from sglang.kernels.ops.communication import push_comm as K
    from sglang.kernels.ops.communication.push_comm import (  # noqa: F401
        one_shot,
        table,
        two_shot,
        two_shot_epoch,
        two_shot_window,
    )

    return K


def _instance(kind, rank, world, hidden, rows, res, geom):
    K = _kernels()
    cls = {
        "one_shot": K.one_shot.OneShotInst,
        "two_shot": K.two_shot.TwoShotInst,
        "two_shot_epoch": K.two_shot_epoch.TwoShotEpochInst,
        "two_shot_window": K.two_shot_window.TwoShotWindowInst,
    }[kind]
    return cls(rank, world, hidden, rows, **res, **geom)


def _sizes(kind, world, hidden, rows, geom, tag):
    """(IPC bytes, counter words) of an instance."""
    K = _kernels()
    if kind == "one_shot":
        return (
            K.one_shot.ipc_bytes(world, hidden, rows, tag=tag),
            K.one_shot.ctr_words(world, hidden, rows, **geom),
        )
    if kind == "two_shot":
        return (
            K.two_shot.ipc_bytes(world, hidden, rows, tag=tag, **geom),
            K.two_shot.ctr_words(world, hidden, rows, **geom),
        )
    if kind == "two_shot_window":
        g = {k: v for k, v in geom.items() if k in ("window", "fs")}
        return (
            K.two_shot_window.ipc_bytes(world, hidden, rows, **g),
            K.two_shot_window.ctr_words(world, hidden, rows, **g),
        )
    g = {k: v for k, v in geom.items() if k in ("window", "nsend", "red", "f", "wire")}
    return (
        K.two_shot_epoch.ipc_bytes(world, hidden, rows, tag=tag, **g),
        K.two_shot_epoch.ctr_words(world, hidden, rows, **g),
    )


class _Spec:
    """One kernel instance to build: kind, row width, rows, constructor
    geometry."""

    def __init__(self, kind, hidden, rows, geom):
        self.kind, self.hidden, self.rows, self.geom = (
            kind,
            int(hidden),
            int(rows),
            geom,
        )
        self.name = None
        self.inst = None
        self.ms = [1]
        self.tail = False
        self.norm = False  # a fused-call instance (residual add + RMSNorm)


# kernels with a masked-tail entry
_TAIL_ALGOS = ("one_shot", "two_shot", "two_shot_window")
_MISS = object()


class _PlainAR:
    """All-reduce of a bf16 tensor of any shape by message bytes (segment
    table), viewed as rows of its segment's width H. A partial last row runs a
    masked-tail kernel (two_shot_epoch: a zero-padded copy)."""

    def __init__(self, world, max_bytes, seg_table, profile=None):
        table = _kernels().table
        usable = table.usable_impl
        lists = []
        for cls in ("fast", "unaligned", "aligned"):
            out = []
            for sg in table.plain_segments(seg_table, profile, cls, usable):
                if sg["lo"] >= max_bytes:
                    break
                out.append(dict(sg, hi=min(sg["hi"], int(max_bytes))))
            lists.append(out)
        # bytes % 16 == 0 on the custom all-reduce's fast path (multiples of
        # 16 x TP) / the others / bytes % 16 == 0 off that path
        self.segs, self.segs_u, self.segs_q = lists
        self.ca_q = 16 * int(seg_table.get("tp", 1))
        self._picks = {}  # element count -> _pick result (fixed plans)
        self._lfs = {}  # element count -> bound one_shot / two_shot launch
        # the padded plan: refused where its choice is the stock path
        self.segs_p = [
            dict(sg, hi=min(sg["hi"], int(max_bytes)))
            for sg in table.plain_segments(seg_table, profile, "padded", usable)
            if sg["lo"] < max_bytes
        ]
        self.max_bytes = max(
            [sg["hi"] for sg in self.segs + self.segs_u + self.segs_q] + [0]
        )
        # one instance per distinct kernel, sized for the largest segment it
        # serves
        top = {}
        for sg in self.segs + self.segs_u + self.segs_q:
            if sg["algo"] != "stock":
                k = self._key(sg)
                top[k] = max(top.get(k, 0), sg["hi"])
        self.specs, built = [], {}
        for sg in self.segs + self.segs_u + self.segs_q:
            sg["inst"] = None
            if sg["algo"] == "stock":
                continue
            k = self._key(sg)
            if k not in built:
                built[k] = self._spec(sg, sg["H"], top[k], world)
                self.specs.append(built[k])
            sg["spec"] = built[k]

    @staticmethod
    def _key(sg):
        return (sg["name"], sg["algo"], sg["H"], tuple(sorted(sg["geom"].items())))

    def _spec(self, sg, h, top, world):
        two_shot = _kernels().two_shot
        geom = dict(sg["geom"])
        # whole rows of the top size, its padded form included
        rows = max(1, -(-(top // 2) // h))
        if sg["algo"] == "two_shot_epoch":
            rows = max(8, rows)
        if sg["algo"] == "two_shot_window":
            rows = max(4, rows)
        sp = _Spec(sg["algo"], h, rows, geom)
        sp.ms = two_shot.m_classes(1, rows)
        sp.m0 = 1
        sp.tail = sg["algo"] in _TAIL_ALGOS
        return sp

    def bind(self, device=None):
        self._picks.clear()
        self._lfs.clear()
        for sg in self.segs + self.segs_u + self.segs_q:
            if sg["algo"] != "stock":
                sg["inst"] = sg["spec"].inst
        # padding source of the copy path (shorter than the widest row)
        wmax = max([sp.hidden for sp in self.specs] + [0])
        self._zeros = (
            torch.zeros(wmax, dtype=torch.bfloat16, device=device)
            if device is not None
            else None
        )

    def _plan_of(self, nbytes):
        """The plan of a message of nbytes: % 16 and a multiple of 16 x TP / %
        16 only / the others."""
        return (
            self.segs
            if nbytes % self.ca_q == 0
            else self.segs_q
            if nbytes % 16 == 0
            else self.segs_u
        )

    def _segment(self, nbytes, segs):
        for sg in segs:
            if nbytes <= sg["hi"]:
                return sg
        return None

    def _pick(self, x):
        """(instance, row width, padded element count (0: none)) or None. It
        depends on the element count alone (bf16, contiguous): cached, so
        supports() and all_reduce() pick once."""
        if x.dtype != torch.bfloat16 or not x.is_contiguous():
            return None
        n = x.numel()
        hit = self._picks.get(n, _MISS)
        if hit is not _MISS:
            return hit
        r = self._pick_n(n) if n else None
        if len(self._picks) >= 4096:
            self._picks.clear()
        self._picks[n] = r
        return r

    def _pick_n(self, n):
        segs = self._plan_of(2 * n)
        sg = self._segment(2 * n, segs)
        if sg is None and segs is self.segs_q:
            # above the off-fast-path plan's measured sizes: the fast-path plan
            # (its stock path is the faster one, so its push choices hold)
            segs = self.segs
            sg = self._segment(2 * n, segs)
        if sg is None or sg["inst"] is None:
            return None
        p, h = sg["inst"], sg["H"]
        if n % h == 0 and n // h <= p.max_tokens:
            return p, h, 0
        # a padded copy may land in the next segment: pick again there
        for _ in range(2):
            p, h = sg["inst"], sg["H"]
            npad = -(-n // h) * h
            tail = sg["algo"] in _TAIL_ALGOS
            if not tail and self.segs_p and (2 * n) % 16 == 0:
                sgp = self._segment(2 * n, self.segs_p)
                if sgp is not None and sgp["algo"] == "stock":
                    # with the copy, this size is the stock path's
                    return None
            if tail:
                # masked tail: only the n elements move, the segment of n decides
                return (p, h, npad) if npad // h <= p.max_tokens else None
            sg2 = self._segment(2 * npad, segs)
            if sg2 is sg or (sg2 is None and segs and sg is segs[-1]):
                # (a size just below the top may pad past it: the instance
                # holds whole rows of the top)
                return (p, h, npad) if npad // h <= p.max_tokens else None
            if sg2 is None or sg2["inst"] is None:
                return None
            sg = sg2
        return None

    def supports(self, x):
        return self._pick(x) is not None

    def all_reduce(self, x):
        n = x.numel()
        run = self._lfs.get(n, _MISS)
        if run is _MISS:
            run = self._bind(x)
            if run is not _MISS:  # (_MISS: not compiled yet, bind next call)
                if len(self._lfs) >= 4096:
                    self._lfs.clear()
                self._lfs[n] = run
        if run is not None and run is not _MISS:
            return run(_aligned(x))
        p, h, npad = self._pick(x)
        if not npad:
            return p.all_reduce(_aligned(x).view(-1, h)).view(x.shape)
        n = x.numel()
        if p.can_tail:  # masked tail, no copy
            return p.all_reduce_tail(_aligned(x).reshape(-1)).view(x.shape)
        xp = torch.cat((x.reshape(-1), self._zeros[: npad - n]))
        return p.all_reduce(xp.view(-1, h)).view(-1)[:n].view(x.shape)

    def _bind(self, x):
        """x -> all-reduce(x) for bf16 contiguous x of x's element count,
        launched directly; None for the general path; _MISS if the variant is
        not compiled yet."""
        from sglang.kernels.ops.communication.push_comm.launch import (
            TRUSTED_BF16,
        )

        pk = self._pick(x)
        if pk is None:
            return None
        p, h, npad = pk
        n = x.numel()
        mk = getattr(p, "plain_launcher", None)  # one_shot / two_shot only
        if mk is None or 2 * n > MAX_I32:
            return None
        lf = mk(n, bool(npad), TRUSTED_BF16)
        if lf is None:
            return _MISS if p.fast else None
        if npad:
            return lambda t: lf(t.reshape(-1)).view(t.shape)
        return lambda t: lf(t.view(-1, h)).view(t.shape)


class _FusedAR:
    """All-reduce + residual add + RMSNorm of [m, H] (H in hidden_sizes), by
    message bytes m x H x 2 (the segment table's fused plans). Per H the
    segments become m ranges; one instance per distinct (kernel, H, geometry),
    sized for the largest m it serves. A stock segment, an m above the plans or
    a kernel that cannot run the norm with the build's options is refused."""

    def __init__(self, world, seg_table, profile, hidden_sizes, max_tokens, tag):
        K = _kernels()
        table, two_shot = K.table, K.two_shot
        self.sel = {}  # H -> [(lo m, hi m, spec)]
        self.specs = []
        pool = {}
        for h in dict.fromkeys(int(v) for v in hidden_sizes):

            def legal(g, h=h):
                try:
                    two_shot.check_cfg(world, h, g, tag)
                    return True
                except AssertionError:
                    return False

            rb, ranges = 2 * h, []
            for sg in table.fused_segments(seg_table, profile, h):
                lo, hi = sg["lo"] // rb + 1, min(sg["hi"] // rb, int(max_tokens))
                if lo > hi or sg["algo"] == "stock":
                    continue
                geom = table.fused_geom(seg_table["impls"][sg["name"]], h, legal)
                if geom is None:
                    continue
                key = (sg["algo"], h, repr(sorted(geom.items())))
                sp = pool.get(key)
                if sp is None:
                    sp = pool[key] = _Spec(sg["algo"], h, hi, geom)
                    sp.m0 = lo
                    sp.norm = True
                    self.specs.append(sp)
                sp.rows, sp.m0 = max(sp.rows, hi), min(sp.m0, lo)
                ranges.append((lo, hi, sp))
            if ranges:
                self.sel[h] = ranges
        for sp in self.specs:
            if sp.kind == "two_shot_epoch":
                sp.rows = max(8, sp.rows)
            sp.ms = two_shot.m_classes(sp.m0, sp.rows)

    def pick(self, x, residual, weight):
        """The instance serving the fused call, or None (no side effect)."""
        if x.dim() != 2 or x.dtype != torch.bfloat16 or not x.is_contiguous():
            return None
        m, h = x.shape
        ranges = self.sel.get(h)
        if ranges is None or m == 0:
            return None
        if (
            residual.dtype != torch.bfloat16
            or tuple(residual.shape) != (m, h)
            or weight.dtype != torch.bfloat16
            or weight.numel() != h
            or residual.device != x.device
            or weight.device != x.device
        ):
            return None
        for lo, hi, sp in ranges:
            if lo <= m <= hi:
                return sp.inst if sp.inst.can_norm else None
        return None

    def serves(self, m, h):
        """Would a fused [m, h] call be taken (its operands aside)?"""
        return any(
            lo <= m <= hi and sp.inst.can_norm for lo, hi, sp in self.sel.get(h, ())
        )


# ------------------------------------------------------------ communicator
_KIND_CODE = {"one_shot": 1, "two_shot": 2, "two_shot_epoch": 3, "two_shot_window": 4}
STREAM_GUARDS = ("serial", "default", "off")


class PushCommunicator:
    """One per (process group, device); see the module docstring.

    group: a CPU (gloo) group. max_bytes: the largest message served.
    stock_profile: the deployment's own all-reduce path (table.StockProfile).
    stream_guard: "serial" (any stream; calls are ordered and single-threaded),
    "default" (refuse non-default streams outside capture) or "off".
    segments_table: override the shipped table (tests).
    hidden_sizes: the row widths H of the fused all-reduce + residual add +
    RMSNorm (fused_allreduce_rmsnorm; () = none); fused_max_tokens: the
    largest m it serves (the plans may stop earlier).
    """

    def __init__(
        self,
        group: ProcessGroup,
        device,
        *,
        max_bytes: int = 1 << 30,
        stock_profile=None,
        timeout_s: float = 300.0,
        trap: bool = True,
        backoff: int = 0,
        fast_launch: bool = True,
        stream_guard: str = "serial",
        compile_guard: bool = True,
        warm: bool = True,
        monitor: bool = False,
        on_fatal=None,
        monitor_period: float = 0.05,
        seq_check: bool = False,
        segments_table=None,
        hidden_sizes=(),
        fused_max_tokens: int = 32768,
    ):
        self.group = group
        self.rank = dist.get_rank(group=group)
        self.world = dist.get_world_size(group=group)
        if isinstance(device, int):
            device = torch.device("cuda", device)
        elif isinstance(device, str):
            device = torch.device(device)
        if device.type == "cuda" and device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        self.device = device
        self._dev_index = device.index if device.type == "cuda" else -1
        if stream_guard not in STREAM_GUARDS:
            raise ValueError(
                f"stream_guard: one of {STREAM_GUARDS}, not {stream_guard!r}"
            )
        self.stream_guard, self.compile_guard = stream_guard, bool(compile_guard)
        # "serial": the stream of the previous call outside capture, and the
        # lock held by the host thread inside a call
        self._last_stream = None
        self._last_raw = None  # ... and its raw handle (the hot-path comparison)
        self._lock = threading.Lock()
        self.fast_launch = bool(fast_launch)
        self.disabled = True
        self.closed = False
        self.plain = None
        self.fused = None
        self.mon = None
        self._pages = None
        self.ws = None
        self.site_box = [0]
        if seq_check and not monitor:
            raise ValueError("seq_check needs the monitor (monitor=True)")
        arch, toolchain, why = push_comm_platform()
        if arch is None:
            logger.info("push-comm disabled: %s", why)
            return
        if self.world not in SUPPORTED_WORLD_SIZES:
            logger.info("push-comm disabled: world size %d", self.world)
            return
        from sglang.srt.distributed.parallel_state import in_the_same_node_as

        if not all(in_the_same_node_as(group, source_rank=0)):
            logger.info("push-comm disabled: the group spans several nodes")
            return
        from sglang.kernels.ops.communication.push_comm.protocol import SpinBound

        table = _kernels().table
        self.spin = SpinBound(timeout_s=timeout_s, trap=trap, backoff=backoff)
        btab = (
            segments_table
            if segments_table is not None
            else table.load_segments(arch, self.world, toolchain)
        )
        with torch.cuda.device(device):
            self._build(
                btab,
                max_bytes,
                stock_profile,
                warm,
                monitor,
                on_fatal,
                monitor_period,
                seq_check,
                hidden_sizes,
                fused_max_tokens,
            )
        self.disabled = self.plain is None and self.fused is None
        logger.info(
            "push-comm rank %d/%d: %s; IPC %.1f MB",
            self.rank,
            self.world,
            ", ".join(self._summary()),
            (self.ws.nbytes if self.ws else 0) / 1e6,
        )

    # ------------------------------------------------------------- build
    def _build(
        self,
        btab,
        max_bytes,
        profile,
        warm,
        monitor,
        on_fatal,
        monitor_period,
        seq_check,
        hidden_sizes=(),
        fused_max_tokens=32768,
    ):
        if btab is None:
            logger.info("push-comm: no segment table for TP%d", self.world)
            return
        if max_bytes > 0:
            p = _PlainAR(self.world, int(max_bytes), btab, profile)
            if p.specs:
                self.plain = p
        if hidden_sizes:
            f = _FusedAR(
                self.world, btab, profile, hidden_sizes, fused_max_tokens, seq_check
            )
            if f.specs:
                self.fused = f
        if self.plain is None and self.fused is None:
            return
        specs = (self.plain.specs if self.plain else []) + (
            self.fused.specs if self.fused else []
        )
        diag = self._start_pages() if monitor else None

        self.ws = PushCommWorkspace(self.group, self.device, self.rank, self.world)
        for i, sp in enumerate(specs):
            nbytes, nctr = _sizes(
                sp.kind,
                self.world,
                sp.hidden,
                sp.rows,
                sp.geom,
                seq_check,
            )
            sp.name = self.ws.reserve(
                f"{i}:{sp.kind}/h{sp.hidden}", nbytes, counters=nctr
            )
        self.ws.seal()
        for sp in specs:
            res = dict(
                ipc=self.ws.region(sp.name),
                ctr=self.ws.counters(sp.name),
                err=self.ws.err(sp.name),
                spin=self.spin,
                fast=self.fast_launch,
            )
            if diag is not None:
                res["diag"] = diag(_KIND_CODE[sp.kind], sp.hidden, sp.geom)
                if seq_check:
                    res.update(tag=True, site_box=self.site_box)
            sp.inst = _instance(
                sp.kind, self.rank, self.world, sp.hidden, sp.rows, res, sp.geom
            )
        # the sentinel fills must land on every rank before any peer pushes
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        if self.plain is not None:
            self.plain.bind(self.device)
        if warm:
            self._warmup(specs)
        if monitor:
            from sglang.srt.distributed.device_communicators.push_comm_monitor import (
                Monitor,
            )

            self.mon = Monitor(
                self.rank,
                self._pages,
                [p.pid for p in self._pages],
                on_fatal=on_fatal,
                period=monitor_period,
            )

    def _warmup(self, specs):
        """Compile every instance, schedule and row-count class without
        launching, barrier, then one collective first touch of each instance
        (its smallest served m); later compiles are logged."""
        from sglang.kernels.ops.communication.push_comm import launch

        for sp in specs:
            x = torch.zeros(1, sp.hidden, dtype=torch.bfloat16, device=self.device)
            if sp.norm:  # the fused kernel (and the plain one: the first touch)
                sp.inst.precompile(x, ms=sp.ms, norm=sp.inst.can_norm)
            elif sp.tail and sp.inst.can_tail:
                sp.inst.precompile(x, ms=sp.ms, tail=True)
            else:
                sp.inst.precompile(x, ms=sp.ms)
        torch.cuda.synchronize(self.device)
        dist.barrier(group=self.group)
        for sp in specs:
            x = torch.zeros(sp.m0, sp.hidden, dtype=torch.bfloat16, device=self.device)
            sp.inst.all_reduce(x)
        torch.cuda.synchronize(self.device)
        assert int(self.ws.err_max().item()) == 0, (
            "push-comm: spin failure during the warm-up"
        )
        launch.seal()

    def _start_pages(self):
        """Create this rank's shared page (registered with the GPU), open every
        peer's; returns the instances' Diag factory."""
        import os
        import uuid
        import zlib

        from sglang.srt.distributed.device_communicators import push_comm_monitor as M

        tag = [f"{os.getpid()}.{uuid.uuid4().hex[:8]}" if self.rank == 0 else None]
        dist.broadcast_object_list(
            tag, src=dist.get_global_rank(self.group, 0), group=self.group
        )
        own = M.SharedPage(M.page_path(tag[0], self.rank), create=True, register=True)
        dist.barrier(group=self.group)
        self._pages = [
            own if r == self.rank else M.SharedPage(M.page_path(tag[0], r))
            for r in range(self.world)
        ]
        nxt = [0]

        def diag(kind, hidden, geom):
            i = nxt[0]
            nxt[0] += 1
            h = zlib.crc32(repr(sorted((k, repr(v)) for k, v in geom.items())).encode())
            return own.diag(i, kind, hidden, trap=False, geom_hash=h)

        return diag

    def _summary(self):
        out = []
        if self.plain is not None:
            out.append(f"any shape <= {self.plain.max_bytes} B")
        if self.fused is not None:
            out.append(
                "fused add + RMSNorm "
                + ", ".join(
                    f"H{h} m <= {max(hi for _, hi, _ in r)}"
                    for h, r in self.fused.sel.items()
                )
            )
        return out or ["nothing served"]

    # ------------------------------------------------------------- calls
    def _call_ok(self, x):
        if (
            self.disabled
            or self.closed
            or not isinstance(x, torch.Tensor)
            or x.device != self.device
        ):
            return False
        if self.compile_guard and torch.compiler.is_compiling():
            return False
        if self.stream_guard == "default":
            if not torch.cuda.is_current_stream_capturing() and (
                torch.cuda.current_stream(self.device)
                != torch.cuda.default_stream(self.device)
            ):
                return False
        elif self.stream_guard == "serial" and self._lock.locked():
            return False  # another thread is inside a call
        return True

    def _order_stream(self):
        """Outside capture, a stream other than the previous call's waits for
        it, so the calls run in their issue order on any streams. The raw
        stream handle is compared (a Stream object only when it changed)."""
        raw = torch._C._cuda_getCurrentRawStream(self._dev_index)
        if raw == self._last_raw:
            return
        cur = torch.cuda.current_stream(self.device)
        if self._last_stream is not None:
            cur.wait_stream(self._last_stream)
        self._last_stream, self._last_raw = cur, raw

    def should_push_ar(self, x) -> bool:
        """Would all_reduce(x) be taken over? (no side effect)"""
        return self._call_ok(x) and self.plain is not None and self.plain.supports(x)

    def should_fuse(self, x) -> bool:
        """Would fused_allreduce_rmsnorm take x [m, H] (given bf16 residual
        [m, H] and weight [H] operands)? No side effect."""
        return (
            self._call_ok(x)
            and self.fused is not None
            and x.dim() == 2
            and x.dtype == torch.bfloat16
            and x.is_contiguous()
            and self.fused.serves(x.shape[0], x.shape[1])
        )

    def fused_allreduce_rmsnorm(
        self, x, residual, weight, eps: float, site: int = 0
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """(out, residual_out) of the all-reduce of x [m, H] + residual, then
        RMSNorm with weight [H]: residual_out = bf16(f32(bf16(AR)) +
        f32(residual)), out = bf16(residual_out * rstd * weight). None when not
        taken over (the caller keeps its own path); x and residual are not
        modified."""
        if not self._call_ok(x) or self.fused is None:
            return None
        p = self.fused.pick(x, residual, weight)
        if p is None:
            return None
        args = (
            _aligned(x),
            _aligned(residual.contiguous()),
            _aligned(weight.contiguous()),
        )
        self.site_box[0] = int(site)
        cap = torch.cuda.is_current_stream_capturing()
        if self.mon is not None and not cap:
            self.mon.check()
        if self.stream_guard != "serial":
            return p.ar_add_rmsnorm(*args, eps)
        # atomic, as all_reduce: refused without side effect
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if not cap:
                self._order_stream()
            return p.ar_add_rmsnorm(*args, eps)
        finally:
            self._lock.release()

    def all_reduce(self, x, site: int = 0):
        """All-reduce x; returns the result, or None when not taken over. site:
        the call site's identity for seq_check."""
        if not self._call_ok(x) or self.plain is None or not self.plain.supports(x):
            return None
        self.site_box[0] = int(site)
        if self.mon is None and self.stream_guard != "serial":
            return self.plain.all_reduce(x)
        cap = torch.cuda.is_current_stream_capturing()
        if self.mon is not None and not cap:
            self.mon.check()
        if self.stream_guard != "serial":
            return self.plain.all_reduce(x)
        # atomic: a concurrent thread may have entered after the check above;
        # the call is refused without side effect
        if not self._lock.acquire(blocking=False):
            return None
        try:
            if not cap:
                self._order_stream()
            return self.plain.all_reduce(x)
        finally:
            self._lock.release()

    # ------------------------------------------------------------- state
    @property
    def err(self):
        """Max error word of every instance (device tensor, no sync)."""
        if self.ws is None or not self.ws.sealed:
            return torch.zeros(1, dtype=torch.int32, device=self.device)
        return self.ws.err_max()

    def check(self):
        """Synchronous error check: 0 = no spin failed."""
        torch.cuda.synchronize(self.device)
        return int(self.err.max().item())

    @property
    def bytes_per_rank(self):
        return self.ws.nbytes if self.ws else 0

    def close(self, abort: bool = False):
        """Free the IPC memory. Collective, idempotent; no call may be in flight.
        abort=True (monitor): set every rank's abort word, wait up to 2 s for
        this queue, no collective step; free only if the queue drained."""
        if self.closed:
            return
        self.closed = True
        if abort:
            assert self.mon is not None, "close(abort=True) needs the monitor"
            import time

            from sglang.srt.distributed.device_communicators.push_comm_monitor import (
                ABORT_USER,
            )

            self.mon.abort_all(ABORT_USER)
            self.mon.stop()
            t0 = time.monotonic()
            idle = False
            while time.monotonic() - t0 < 2.0:
                if torch.cuda.current_stream(self.device).query():
                    idle = True
                    break
                time.sleep(0.005)
            if idle and self.ws is not None and self.ws.buf is not None:
                self.ws.buf.close(None, sync=False)
            for p in self._pages:
                p.close()
            return
        if self.ws is not None:
            self.ws.close()
        if self.mon is not None:
            self.mon.stop()
        if self._pages is not None:
            for p in self._pages:
                p.close(unlink=True)
