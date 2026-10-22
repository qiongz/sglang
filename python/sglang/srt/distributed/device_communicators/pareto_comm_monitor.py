# SPDX-License-Identifier: Apache-2.0
"""Optional failure handling of pareto-comm (monitor=True).

A host-mapped /dev/shm page per rank holds the abort word the kernels poll and
the diagnostics ring they write; a monitor thread makes every rank fatal on the
first failed wait or dead peer by setting every page's abort word."""

from __future__ import annotations

import ctypes
import logging
import mmap
import os
import struct
import threading
import time

from sglang.kernels.ops.communication.pareto_comm.protocol import (
    ABORTED,
    DIAG_MAGIC,
    DIAG_SLOTS,
    LIMIT,
    SEQ_MISMATCH,
    Diag,
)

logger = logging.getLogger(__name__)

PAGE = 64 << 10
OFF_ABORT, OFF_ABORT_RANK, OFF_ABORT_DETAIL = 0x0, 0x4, 0x8
OFF_HEART, OFF_PID = 0x40, 0x80
OFF_INST, INST_MAX = 0x100, (0x1000 - 0x100) // 16
OFF_DIAG, DIAG_REC = 0x1000, 64

ABORT_FATAL, ABORT_PEER_DEAD, ABORT_PEER_ABORT, ABORT_USER = 1, 2, 3, 4
ABORT_NAMES = {
    0: "running",
    ABORT_FATAL: "local failure",
    ABORT_PEER_DEAD: "peer died",
    ABORT_PEER_ABORT: "peer aborted",
    ABORT_USER: "close(abort=True)",
}
KIND_NAMES = {1: "one_shot", 2: "two_shot", 3: "two_shot_epoch", 4: "two_shot_window"}
REASON = {LIMIT: "timeout", ABORTED: "abort", SEQ_MISMATCH: "call-sequence mismatch"}
_HOST_REGISTER_MAPPED = 0x2

# The own page goes to a fixed address when free: the kernels compile its
# address in as an immediate, so a fixed one keeps Triton's disk cache hits.
FIXED_BASE = 0x6C0000000000
_MAP_FIXED_NOREPLACE = 0x100000
_fixed_next = [0]


def _hip():
    h = ctypes.CDLL("libamdhip64.so")
    h.hipHostRegister.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_uint]
    h.hipHostGetDevicePointer.argtypes = [
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
        ctypes.c_uint,
    ]
    h.hipHostUnregister.argtypes = [ctypes.c_void_p]
    for f in ("hipHostRegister", "hipHostGetDevicePointer", "hipHostUnregister"):
        getattr(h, f).restype = ctypes.c_int
    return h


def _libc():
    c = ctypes.CDLL(None, use_errno=True)
    c.mmap.restype = ctypes.c_void_p
    c.mmap.argtypes = [
        ctypes.c_void_p,
        ctypes.c_size_t,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_int,
        ctypes.c_long,
    ]
    c.munmap.argtypes = [ctypes.c_void_p, ctypes.c_size_t]
    return c


class _FixedMap:
    """A MAP_SHARED mapping at a requested address."""

    def __init__(self, fd, hint):
        self._c = _libc()
        fail = ctypes.c_void_p(-1).value
        a = self._c.mmap(
            ctypes.c_void_p(hint),
            PAGE,
            mmap.PROT_READ | mmap.PROT_WRITE,
            mmap.MAP_SHARED | _MAP_FIXED_NOREPLACE,
            fd,
            0,
        )
        if a in (None, fail) or a != hint:
            if a not in (None, fail):
                self._c.munmap(ctypes.c_void_p(a), PAGE)
            raise OSError("fixed address not available")
        self.addr = a
        self.buf = (ctypes.c_char * PAGE).from_address(a)

    def close(self):
        if self.buf is not None:
            self.buf = None
            self._c.munmap(ctypes.c_void_p(self.addr), PAGE)


class SharedPage:
    """A /dev/shm page. create=True: a new zeroed page with this process's pid
    (register=True maps it into the GPU for this rank's kernels);
    create=False: a peer's existing page (host access only)."""

    def __init__(self, path, create=False, register=False):
        self.path = path
        if create:
            fd = os.open(path, os.O_RDWR | os.O_CREAT | os.O_EXCL, 0o600)
            os.ftruncate(fd, PAGE)
        else:
            fd = os.open(path, os.O_RDWR)
        self._fixed = None
        try:
            if register:
                try:
                    self._fixed = _FixedMap(fd, FIXED_BASE + _fixed_next[0] * PAGE)
                    _fixed_next[0] += 1
                except OSError:
                    self._fixed = None
            if self._fixed is not None:
                self.mm = self._fixed.buf
            else:
                self.mm = mmap.mmap(
                    fd, PAGE, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE
                )
        finally:
            os.close(fd)
        self.owner = bool(create)
        self.dev = 0
        self._hip = None
        if create:
            self.mm[:] = bytes(PAGE)
            struct.pack_into("<i", self.mm, OFF_PID, os.getpid())
        if register:
            self._register()

    def _register(self):
        self._hip = _hip()
        self._buf = ctypes.c_char.from_buffer(
            self.mm
        )  # pins the mapping while registered
        self.host = ctypes.addressof(self._buf)
        st = self._hip.hipHostRegister(
            ctypes.c_void_p(self.host), PAGE, _HOST_REGISTER_MAPPED
        )
        if st != 0:
            raise RuntimeError(f"hipHostRegister({self.path}) failed: hip status {st}")
        d = ctypes.c_void_p()
        st = self._hip.hipHostGetDevicePointer(
            ctypes.byref(d), ctypes.c_void_p(self.host), 0
        )
        if st != 0:
            raise RuntimeError(
                f"hipHostGetDevicePointer({self.path}) failed: hip status {st}"
            )
        self.dev = int(d.value)

    # --------------------------------------------------------------- fields
    def _i32(self, off):
        return struct.unpack_from("<i", self.mm, off)[0]

    @property
    def abort(self):
        return self._i32(OFF_ABORT)

    @property
    def pid(self):
        return self._i32(OFF_PID)

    def set_abort(self, reason, by_rank, detail=0):
        """First writer wins (a later reason does not overwrite the first)."""
        if self.abort == 0:
            struct.pack_into("<ii", self.mm, OFF_ABORT_RANK, int(by_rank), int(detail))
            struct.pack_into("<i", self.mm, OFF_ABORT, int(reason))

    def abort_info(self):
        r, by, det = struct.unpack_from("<iii", self.mm, OFF_ABORT)
        return dict(
            reason=r, reason_name=ABORT_NAMES.get(r, str(r)), by_rank=by, detail=det
        )

    def heartbeat(self, ns):
        struct.pack_into("<Q", self.mm, OFF_HEART, int(ns))

    def set_inst(self, i, kind, hidden, geom_hash=0):
        assert 0 <= i < INST_MAX, (
            f"instance id {i} out of the page's table ({INST_MAX})"
        )
        struct.pack_into(
            "<iiii",
            self.mm,
            OFF_INST + 16 * i,
            int(kind),
            int(hidden),
            int(geom_hash) & 0x7FFFFFFF,
            0,
        )

    def inst(self, i):
        k, h, g, _ = struct.unpack_from("<iiii", self.mm, OFF_INST + 16 * i)
        return dict(kind=KIND_NAMES.get(k, k), hidden=h, geom=g)

    def diag(self, i, kind, hidden, trap=False, geom_hash=0):
        """The Diag of instance i of this (registered) page; records the
        instance in the table."""
        assert self.dev, "the page is not registered with the GPU"
        self.set_inst(i, kind, hidden, geom_hash)
        return Diag(
            abort=self.dev + OFF_ABORT,
            base=self.dev + OFF_DIAG,
            inst=int(i),
            trap=bool(trap),
        )

    def records(self):
        """The complete diagnostics records (magic present), decoded."""
        out = []
        for s in range(DIAG_SLOTS):
            f = struct.unpack_from("<16i", self.mm, OFF_DIAG + DIAG_REC * s)
            if f[0] != DIAG_MAGIC:
                continue
            out.append(
                dict(
                    slot=s,
                    reason=REASON.get(f[1], f"counter {f[1]}"),
                    rank=f[2],
                    inst=f[3],
                    kind=KIND_NAMES.get(f[4], f[4]),
                    role=f[5],
                    row=f[6],
                    cnt=f[7],
                    inst_def=self.inst(f[3]) if 0 <= f[3] < INST_MAX else None,
                )
            )
        return out

    def ndiag(self):
        return sum(
            struct.unpack_from("<i", self.mm, OFF_DIAG + DIAG_REC * s)[0] == DIAG_MAGIC
            for s in range(DIAG_SLOTS)
        )

    def close(self, unlink=False):
        if self.dev and self._hip is not None:
            self._hip.hipHostUnregister(ctypes.c_void_p(self.host))
            self.dev = 0
            del self._buf
        if self.mm is not None:
            if self._fixed is not None:
                self._fixed.close()
            else:
                self.mm.close()
            self.mm = None
        if unlink and self.owner:
            try:
                os.unlink(self.path)
            except FileNotFoundError:
                pass


def page_path(tag, rank):
    return f"/dev/shm/sglang_pareto_comm.{tag}.r{rank}"


# ------------------------------------------------------------------ monitor
def pid_alive(pid):
    """The process exists and is not a zombie."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        with open(f"/proc/{pid}/stat") as f:
            st = f.read()
        return st[st.rindex(")") + 2] not in "ZX"
    except (OSError, ValueError, IndexError):
        return True


def default_on_fatal(info):
    logger.critical("pareto-comm fatal: %s", info)
    os.abort()


class Monitor:
    """Rank `rank`'s monitor over pages[r] (pages[rank] is this rank's own,
    registered page) and the peers' pids."""

    def __init__(self, rank, pages, pids, on_fatal=None, period=0.05, start=True):
        self.rank, self.pages, self.pids = rank, pages, list(pids)
        self.page = pages[rank]
        self.on_fatal = on_fatal or default_on_fatal
        self.period = float(period)
        self.info = None  # set once: the fatal event
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._th = None
        if start:
            self.start()

    def start(self):
        self._th = threading.Thread(
            target=self._run, name=f"pareto-comm-monitor-r{self.rank}", daemon=True
        )
        self._th.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                if self.poll():
                    return
            except Exception:  # a monitor bug must not die silently
                logger.exception("pareto-comm monitor rank %d", self.rank)
            self._stop.wait(self.period)

    def poll(self):
        """One round; True once fatal (the thread then stops)."""
        if self.info is not None:
            return True
        self.page.heartbeat(time.monotonic_ns())
        ev = None
        if self.page.ndiag():
            ev = (ABORT_FATAL, self.rank, "kernel failure on this rank")
        elif self.page.abort:
            a = self.page.abort_info()
            ev = (
                a["reason"],
                a["by_rank"],
                f"abort word set ({a['reason_name']}) by rank {a['by_rank']}",
            )
        else:
            for r, (p, pid) in enumerate(zip(self.pages, self.pids)):
                if r == self.rank:
                    continue
                if not pid_alive(pid):
                    ev = (ABORT_PEER_DEAD, r, f"rank {r} (pid {pid}) died")
                    break
                if p.abort:
                    ev = (
                        ABORT_PEER_ABORT,
                        r,
                        f"rank {r} aborted ({p.abort_info()['reason_name']})",
                    )
                    break
        if ev is None:
            return False
        self.fatal(*ev)
        return True

    def abort_all(self, reason, detail=0):
        """Set the abort word of every page (own: reason; peers: peer abort)."""
        self.page.set_abort(reason, self.rank, detail)
        for r, p in enumerate(self.pages):
            if r != self.rank:
                p.set_abort(ABORT_PEER_ABORT, self.rank, detail)

    def fatal(self, reason, culprit, what):
        with self._lock:
            if self.info is not None:
                return
            self.abort_all(reason, culprit)
            pages = {
                r: dict(abort=p.abort_info(), records=p.records())
                for r, p in enumerate(self.pages)
                if p.mm is not None
            }
            self.info = dict(
                rank=self.rank,
                event=what,
                reason=ABORT_NAMES.get(reason, reason),
                culprit=culprit,
                pages=pages,
            )
        logger.error("pareto-comm rank %d: %s", self.rank, what)
        self.on_fatal(self.info)

    def check(self):
        """Eager-call check (one host-memory read): raise if this rank is fatal
        or its abort word is set."""
        if self.info is not None or self.page.abort:
            what = (
                self.info["event"]
                if self.info is not None
                else self.page.abort_info()["reason_name"]
            )
            raise RuntimeError(f"pareto-comm rank {self.rank} aborted: {what}")

    def stop(self):
        self._stop.set()
        if self._th is not None and self._th is not threading.current_thread():
            self._th.join(timeout=2 * self.period + 1.0)
