"""Multi-GPU correctness of pareto-comm (gfx950, Triton 3.7): lossless kernels
bitwise against the fp32 rank-order sum (modulo the sign of zero) and identical
on every rank, the xe8 wire against its reference and QuickReduce INT8, CUDA
graphs, refusals, counter wrap, failure handling, streams, IPC memory; the
fused all-reduce + residual add + RMSNorm (residual output bitwise, output
within 1 ulp of the fp32 RMSNorm, the same with the monitor and call tags)."""

import json
import multiprocessing
import os
import socket
import time
import traceback
import unittest

import torch
import torch.distributed as dist

from sglang.test.ci.ci_register import register_amd_ci
from sglang.test.test_utils import CustomTestCase

register_amd_ci(est_time=600, suite="stage-c-test-large-8-gpu-amd-mi35x")

MIB = 1 << 20


def _available():
    if not torch.cuda.is_available() or torch.version.hip is None:
        return False
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        pareto_comm_available,
    )

    return pareto_comm_available()


def _open_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("", 0))
        return s.getsockname()[1]


# --------------------------------------------------------------- reference
def _inputs(case, world, shape, device):
    """Every rank's input of one case (each rank builds all of them, so it can
    compute the reference): normal values, -0.0, subnormals and large values."""
    xs = []
    for r in range(world):
        g = torch.Generator(device=device).manual_seed(1000 * case + r)
        x = torch.randn(shape, device=device, generator=g) * 0.5
        flat = x.view(-1)
        flat[: min(8, flat.numel())] = -0.0
        if flat.numel() > 32:
            flat[16:20] = 1e-39 * (r + 1)  # bf16 subnormals
            flat[24] = 3.0e4 * (1 if r % 2 else -1)
        xs.append(x.to(torch.bfloat16))
    return xs


def _ref_all_reduce(xs):
    acc = xs[0].float()
    for t in xs[1:]:
        acc = acc + t.float()
    out = acc.to(torch.bfloat16)
    return torch.where(out.view(torch.int16) == -32768, torch.zeros_like(out), out)


def _bf16_bits(v):
    """fp32 -> bf16 bits (int64), round to nearest even."""
    b = v.float().view(torch.int32).to(torch.int64) & 0xFFFFFFFF
    return ((b + 0x7FFF + ((b >> 16) & 1)) >> 16) & 0xFFFF


def _bits_f32(b):
    b = (b.to(torch.int64) << 16) & 0xFFFFFFFF
    return (
        torch.where(b >= (1 << 31), b - (1 << 32), b)
        .to(torch.int32)
        .view(torch.float32)
    )


def _xe8(v):
    """fp32 values [..., 256k] -> the xe8 wire's decoded values (fp32): per 32
    values the first largest |v| as its bf16, the other 31 as
    clamp(rint(v / scale), -128, 127) * scale, scale = bf16(the signed extreme
    / -128, widened to the opposite extreme / 127); a 256-value row block
    with a NaN, an infinite group maximum or a group of nonzero values all
    below 2^-100 is sent as bf16."""
    v = torch.where(v == 0, torch.zeros_like(v), v).float()
    g = v.reshape(-1, 32)
    nan = g.isnan().any(-1)
    a = torch.where(g.isnan(), torch.zeros_like(g), g).abs()
    am = a.amax(-1)
    pos = torch.argmax((a == am[:, None]).to(torch.int8), -1)
    isx = torch.arange(32, device=v.device)[None, :] == pos[:, None]
    rest = torch.where(isx | g.isnan(), torch.zeros_like(g), g)
    mx, mn = rest.amax(-1), rest.amin(-1)
    rm = torch.maximum(mx.clamp_min(0), (mn * -0.0078125).clamp_min(0) * 128.0)
    bad = nan | am.isinf() | ((rm > 0) & (rm < 2.0**-100))
    raw = bad.reshape(-1, 8).any(-1).repeat_interleave(8)
    p = mx.abs() > mn.abs()
    ext = torch.where(p, mx, mn)
    t = ext.abs() * (1.0 / 128.0)
    oq = torch.where(p, -mn, mx) / 127.0
    t = torch.where(oq > t, oq, t)
    t = torch.where(ext >= 0, -t, t)
    sc = _bits_f32(_bf16_bits(t))
    inv = torch.where(sc != 0, 1.0 / sc, torch.zeros_like(sc))
    q = torch.clamp(torch.round(rest.double() * inv.double()[:, None]), -128, 127)
    d = q.float() * sc[:, None] + 0.0
    e = torch.gather(g, 1, pos[:, None])
    d = torch.where(isx, _bits_f32(_bf16_bits(e)).expand_as(d), d)
    d = torch.where(raw[:, None], _bits_f32(_bf16_bits(g)), d)
    return d.reshape(v.shape)


def _ref_xe8(xs):
    """The xe8 wire's all-reduce of flat bf16 messages (zero-padded to whole
    256-value row blocks, as the kernel's padded rows): each rank's values
    decoded, summed in fp32 in rank order, the sum encoded again, rounded
    once to bf16."""
    n = xs[0].numel()
    pad = -n % 256
    acc = None
    for x in xs:
        d = _xe8(torch.nn.functional.pad(x.reshape(-1).float(), (0, pad)))
        acc = d if acc is None else acc + d
    out = _bits_f32(_bf16_bits(_xe8(acc))).to(torch.bfloat16)
    return out[:n].reshape(xs[0].shape)


def _err_stats(y, xs):
    """Error of y against the exact (float64) sum: rel RMS, q99 / q99.9 /
    q99.99 and max of |error|, mean |error|, |mean error| (bias) and the
    bias noise level (std / sqrt(n))."""
    exact = xs[0].double()
    for x in xs[1:]:
        exact = exact + x.double()
    e = (y.double() - exact).reshape(-1)
    ae = e.abs()
    sub = ae[:: max(1, ae.numel() // (1 << 24))].float()
    q = torch.quantile(sub, torch.tensor([0.99, 0.999, 0.9999], device=sub.device))
    return dict(
        relrms=float((e.pow(2).sum() / exact.pow(2).sum()).sqrt()),
        q99=float(q[0]),
        q999=float(q[1]),
        q9999=float(q[2]),
        maxabs=float(ae.max()),
        meanabs=float(ae.mean()),
        bias=float(e.mean().abs()),
        bias_noise=float(e.std() / e.numel() ** 0.5),
    )


def _canon(t):
    return torch.where(t.view(torch.int16) == -32768, torch.zeros_like(t), t)


def _same_bits(a, b):
    """Bitwise equal, modulo the sign of zero."""
    return torch.equal(_canon(a).view(torch.int16), _canon(b).view(torch.int16))


def _identical(a, b):
    return torch.equal(a.view(torch.int16), b.view(torch.int16))


def _kernel(plain, x):
    """Which kernel serves x, and how a partial last row runs."""
    pk = plain._pick(x)
    if pk is None:
        return None
    p, h, npad = pk
    how = "masked tail" if p.can_tail else "zero-padded copy"
    return f"{type(p).__name__} H={h}" + (f" ({how})" if npad else "")


def _same_everywhere(t, group):
    """Every rank holds the same bits of t."""
    ts = [torch.empty_like(t) for _ in range(dist.get_world_size(group))]
    dist.all_gather(ts, t.contiguous(), group=group)
    return all(_identical(ts[0], u) for u in ts)


def _check(comm, case, n, rank, world, device, group, ref=_ref_all_reduce):
    xs = _inputs(case, world, (n,), device)
    y = comm.all_reduce(xs[rank])
    assert y is not None, f"n={n}: not served"
    assert y.shape == xs[rank].shape
    assert _same_bits(y, ref(xs)), f"n={n} ({_kernel(comm.plain, xs[rank])}): differs"
    assert _same_everywhere(y, group), f"n={n}: differs across ranks"
    return y


def _fast_path_size(plain, sg, cap=16 * MIB):
    """The largest element count of segment sg (of the fast-path plan) that is a
    multiple of 16 x TP bytes, at most cap bytes; None if there is none."""
    q = plain.ca_q // 2
    n = min(sg["hi"], cap, plain.max_bytes) // 2 // q * q
    return n if 2 * n > sg["lo"] else None


def _served(comm, plain=None):
    """One served size (elements) per kernel of the fast-path plan (of plain,
    default the communicator's)."""
    plain = plain or comm.plain
    out = {}
    for sg in plain.segs:
        n = _fast_path_size(plain, sg)
        if sg["algo"] != "stock" and sg["name"] not in out and n:
            x = torch.empty(n, dtype=torch.bfloat16)
            if comm.plain.supports(x):
                out[sg["name"]] = n
    return sorted(out.values())


def _stock_sizes(comm):
    out = []
    for sg in comm.plain.segs:
        n = _fast_path_size(comm.plain, sg, 1 << 40)
        if sg["algo"] == "stock" and n:
            out.append(n)
    return out


# ------------------------------------------------------------------ workers
def _init(rank, world, port):
    device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(device)
    dist.init_process_group(
        backend="gloo",
        init_method=f"tcp://127.0.0.1:{port}",
        rank=rank,
        world_size=world,
    )
    return device, dist.group.WORLD


def _shipped_sizes(segs, cap):
    """Element counts probing each segment of a plan: its upper edge, just
    above its lower edge and an odd count inside, up to cap bytes."""
    out = []
    for sg in segs:
        lo, hi = sg["lo"], min(sg["hi"], cap)
        if lo >= cap:
            break
        for b in (hi, lo + 2, (lo + hi) // 2 | 2):
            if lo < b <= hi:
                out.append((b // 2, sg))
    return out


def _shipped(comm, t, rank, world, device, group):
    """The shipped table: every segment's edges and an odd size inside,
    aligned and unaligned; push segments bitwise exact, stock ones refused."""
    from sglang.kernels.ops.communication.pareto_comm import table

    cap = min(comm.plain.max_bytes, 64 * MIB)
    tails = 0
    cases = _shipped_sizes(table.plain_segments(t), cap)
    cases += _shipped_sizes(table.plain_segments(t, cls="unaligned"), cap)
    # aligned sizes off the custom all-reduce's fast path (not multiples of
    # 16 x TP) have their own plan
    cases += [
        (n - 8, sg)
        for n, sg in _shipped_sizes(table.plain_segments(t, cls="aligned"), cap)
        if n > 8 and n % 8 == 0
    ]
    for case, (n, sg) in enumerate(cases):
        aligned = (2 * n) % 16 == 0
        plan = comm.plain._plan_of(2 * n)
        want = next((s for s in plan if 2 * n <= s["hi"]), None)
        if want is None and plan is comm.plain.segs_q:
            # above the off-fast-path plan: the fast-path plan
            plan = comm.plain.segs
            want = next((s for s in plan if 2 * n <= s["hi"]), None)
        if want is None:
            continue
        xs = _inputs(800 + case, world, (n,), device)
        y = comm.all_reduce(xs[rank])
        if want["algo"] == "stock":
            assert y is None, f"shipped n={n}: stock segment taken over"
            continue
        if y is None and want["algo"] == "two_shot_epoch" and aligned:
            # aligned sizes that two_shot_epoch serves zero-padded have their
            # own (padded) plan
            psegs = table.plain_segments(t, cls="padded")
            if (
                next((s for s in psegs if 2 * n <= s["hi"]), {"algo": None})["algo"]
                == "stock"
            ):
                continue
        if y is None and want["algo"] == "two_shot_epoch":
            # two_shot_epoch reduces a zero-padded copy, picked by its size
            h = comm.plain._segment(2 * n, plan)["H"]
            padded = 2 * -(-n // h) * h
            assert next(s for s in plan if padded <= s["hi"])["algo"] == "stock", (
                f"shipped n={n}: not served"
            )
            continue
        assert y is not None, f"shipped n={n} ({want['name']}): not served"
        assert _same_bits(y, _ref_all_reduce(xs)), (
            f"shipped n={n} ({want['name']}): differs"
        )
        assert _same_everywhere(y, group), f"shipped n={n}: differs across ranks"
        tails += "masked tail" in (_kernel(comm.plain, xs[rank]) or "")
    assert tails > 0, "no case ran a masked tail"
    # above 64 MiB (the default max_bytes serves the whole table): one size in
    # the middle of each push segment of the fast-path / unaligned plans
    big = 0
    for segs, adj in ((comm.plain.segs, 0), (comm.plain.segs_u, -1)):
        for sg in segs:
            if sg["algo"] == "stock" or sg["hi"] <= 64 * MIB:
                continue
            mid = (max(sg["lo"], 64 * MIB) + min(sg["hi"], 256 * MIB)) // 4
            n = mid // 4096 * 4096 + adj
            _check(comm, 880 + big, n, rank, world, device, group)
            big += 1
    # the top of the table: 768 MiB and 1 GiB of whole rows, and 768 MiB +
    # 4 KiB (a partial last row: the zero-padded copy)
    for nbytes in (768 * MIB, 1024 * MIB, 768 * MIB + 4096):
        sg = comm.plain._segment(nbytes, comm.plain.segs)
        if nbytes > comm.plain.max_bytes or sg is None or sg["algo"] == "stock":
            continue
        _check(comm, 890 + big, nbytes // 2, rank, world, device, group)
        big += 1
    assert comm.check() == 0


def _graph(comm, rank, world, device, group, ref=_ref_all_reduce, sizes=None):
    sizes = sizes or _served(comm)
    stat = {n: torch.zeros(n, dtype=torch.bfloat16, device=device) for n in sizes}
    outs = {}
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        for n in sizes:
            outs[n] = comm.all_reduce(stat[n])
            assert outs[n] is not None, f"graph n={n}: not served"
    for rep in range(3):
        refs = {}
        for i, n in enumerate(sizes):
            xs = _inputs(500 + 10 * rep + i, world, (n,), device)
            stat[n].copy_(xs[rank])
            refs[n] = ref(xs)
        g.replay()
        torch.cuda.synchronize()
        for n in sizes:
            assert _same_bits(outs[n], refs[n]), f"graph n={n}: differs"
            assert _same_everywhere(outs[n], group)
    assert comm.check() == 0


def _wrap(comm, rank, world, device, group, ref=_ref_all_reduce, sizes=None):
    """Move every rotation counter just below the int32 wrap (by a multiple of
    6: every slot stays), two_shot_epoch generations and their release flags
    to 2^31 - 64, then run calls across the wrap."""
    from sglang.kernels.ops.communication.pareto_comm import (
        primitives,
        two_shot,
        two_shot_epoch,
        two_shot_window,
    )

    delta = 2147483586
    for sp in comm.plain.specs:
        p = sp.inst
        if isinstance(p, two_shot_epoch.TwoShotEpochInst):
            gen = (1 << 31) - 64
            p.ctr.fill_(gen)
            primitives.fill(
                p.buf.addrs[p.rank] + 4 * p.rel_off, p.P * two_shot_epoch.F_ROLES, gen
            )
        elif isinstance(p, two_shot_window.TwoShotWindowInst):
            # two_shot_window's RS slots alternate parities by generation: move every
            # slot's generation by an even step and its release flags with it
            ctr = p.ctr.view(p.P, p.nrole)
            assert bool((ctr == ctr[:, :1]).all()), (
                "two_shot_window counters of a slot differ"
            )
            step = ((1 << 31) - 64 - int(ctr.max().item())) & ~1
            ctr += step
            for b, g in enumerate(ctr[:, 0].tolist()):
                primitives.fill(
                    p.buf.addrs[p.rank] + 4 * (p.rel_off + b * p.nf), p.nf, g
                )
        else:
            p.ctr[: p.ncell].view(-1, two_shot.CPAD)[:, 0] += delta
    torch.cuda.synchronize()
    dist.barrier(group=group)
    for n in sizes or _served(comm):
        zero = torch.zeros(n, dtype=torch.bfloat16, device=device)
        for it in range(70):
            if it % 23:
                comm.all_reduce(zero)
                continue
            xs = _inputs(9000 + it, world, (n,), device)
            assert _same_bits(comm.all_reduce(xs[rank]), ref(xs)), (
                f"wrap: n={n} call {it} differs"
            )
    torch.cuda.synchronize()
    assert comm.check() == 0


def _refusals(comm, rank, world, device, group):
    before = comm.ws.ctr_page.clone()
    n = _served(comm)[0]
    x = torch.randn(n, device=device).to(torch.bfloat16)
    cases = [
        ("fp32", x.float()),
        ("non-contiguous", torch.zeros(64, 8, dtype=torch.bfloat16, device=device).t()),
        ("empty", x[:0]),
        (
            "above max_bytes",
            torch.zeros(
                comm.plain.max_bytes // 2 + 8, dtype=torch.bfloat16, device=device
            ),
        ),
    ]
    cases += [
        (f"stock n={m}", torch.zeros(m, dtype=torch.bfloat16, device=device))
        for m in _stock_sizes(comm)
    ]
    for name, t in cases:
        assert comm.all_reduce(t) is None, f"{name}: taken over"
        assert not comm.should_push_ar(t), f"{name}: should_push_ar"
    # stream_guard="default": a non-default stream outside capture is refused
    guard, comm.stream_guard = comm.stream_guard, "default"
    try:
        s = torch.cuda.Stream(device)
        with torch.cuda.stream(s):
            assert comm.all_reduce(x) is None, (
                "non-default stream outside capture: taken over"
            )
    finally:
        comm.stream_guard = guard
    torch.cuda.synchronize()
    assert torch.equal(before, comm.ws.ctr_page), "a refused call touched a counter"


def _launcher(rank, world, device, group, reference):
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
    )

    comm = ParetoCommunicator(group, device, fast_launch=False)
    try:
        for case, (n, ref) in enumerate(reference):
            xs = _inputs(case, world, (n,), device)
            y = comm.all_reduce(xs[rank])
            assert _identical(y, ref), f"fast_launch=False n={n} differs"
    finally:
        comm.close()


def _main(rank, world, device, group):
    from sglang.kernels.ops.communication.pareto_comm import table
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
        pareto_comm_platform,
    )

    arch, toolchain, _ = pareto_comm_platform()
    t = table.load_segments(arch, world, toolchain)
    assert t is not None, "no shipped bytes table"
    comm = ParetoCommunicator(group, device)
    assert not comm.disabled
    _shipped(comm, t, rank, world, device, group)
    _graph(comm, rank, world, device, group)
    _refusals(comm, rank, world, device, group)
    reference = []
    for case, n in enumerate(_served(comm)):
        reference.append((n, comm.all_reduce(_inputs(case, world, (n,), device)[rank])))
    _wrap(comm, rank, world, device, group)
    comm.close()
    comm.close()  # idempotent
    assert (
        comm.all_reduce(torch.zeros(1024, dtype=torch.bfloat16, device=device)) is None
    )
    _launcher(rank, world, device, group, reference)


def _streams(rank, world, device, group):
    """stream_guard="serial": side streams, alternating streams, graph capture
    on a side stream, two host threads; bitwise results, no compile after the
    warm-up."""
    import threading

    import triton

    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
    )

    comm = ParetoCommunicator(group, device)
    assert comm.stream_guard == "serial"
    sizes = _served(comm)[:4]
    s1, s2, s3 = (torch.cuda.Stream(device) for _ in range(3))
    case = 300
    try:
        for n in sizes:  # side stream
            case += 1
            xs = _inputs(case, world, (n,), device)
            with torch.cuda.stream(s1):
                s1.wait_stream(torch.cuda.default_stream(device))
                y = comm.all_reduce(xs[rank])
            torch.cuda.synchronize()
            assert _identical(y, _ref_all_reduce(xs)), f"side stream n={n}"
        outs = []
        for i in range(24):  # switching streams, no host sync in between
            case += 1
            n = sizes[i % len(sizes)]
            st = (s1, s2, torch.cuda.default_stream(device))[i % 3]
            xs = _inputs(case, world, (n,), device)
            with torch.cuda.stream(st):
                st.wait_stream(torch.cuda.default_stream(device))
                outs.append((n, comm.all_reduce(xs[rank]), xs))
        torch.cuda.synchronize()
        for n, y, xs in outs:
            assert _identical(y, _ref_all_reduce(xs)), f"stream switch n={n}"
        n = sizes[0]  # a graph captured on a side stream
        xg = _inputs(case + 1, world, (n,), device)[rank].clone()
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s3):
            s3.wait_stream(torch.cuda.default_stream(device))
            comm.all_reduce(xg)
            torch.cuda.synchronize()
            with torch.cuda.graph(g, stream=s3):
                yg = comm.all_reduce(xg)
        for _ in range(3):
            case += 1
            xs = _inputs(case, world, (n,), device)
            xg.copy_(xs[rank])
            torch.cuda.synchronize()
            g.replay()
            torch.cuda.synchronize()
            assert _identical(yg, _ref_all_reduce(xs)), "graph on a side stream"
        res = {}

        def call(key, st, xs):
            torch.cuda.set_device(device)
            with torch.cuda.stream(st):
                st.wait_stream(torch.cuda.default_stream(device))
                res[key] = comm.all_reduce(xs[rank])

        # another thread's calls (rank-uniform: one thread after the other)
        xs = _inputs(case + 1, world, (n,), device)
        t = threading.Thread(target=call, args=("thread", s1, xs))
        t.start()
        t.join()
        torch.cuda.synchronize()
        assert _identical(res["thread"], _ref_all_reduce(xs)), "another thread"
        # a call while another thread is inside one: refused, nothing launched
        before = comm.ws.ctr_page.clone()
        with comm._lock:
            t = threading.Thread(target=call, args=("overlap", s2, xs))
            t.start()
            t.join()
        torch.cuda.synchronize()
        assert res["overlap"] is None, "a concurrent thread's call was taken over"
        assert torch.equal(before, comm.ws.ctr_page), "a refused call launched"
        # no compile after the warm-up: whole rows, partial last rows, padded
        # copies, every row-count class of the served kernels
        misses = []
        hook = triton.knobs.runtime.jit_cache_hook
        triton.knobs.runtime.jit_cache_hook = lambda *a, **k: misses.append(1) and False
        try:
            for i, n in enumerate(_served(comm)):
                for v in (n, n - 1, n + 8 * world, n // 3 + 1):
                    x = _inputs(500 + i, world, (v,), device)[rank]
                    comm.all_reduce(x)
            torch.cuda.synchronize()
        finally:
            triton.knobs.runtime.jit_cache_hook = hook
        assert not misses, f"{len(misses)} kernel compiles after the warm-up"
        assert comm.check() == 0
    finally:
        comm.close()


def _pressure_sizes(comm, cap=16 * MIB):
    """Sizes of >= 16 rows of each served kernel's width (grids of many CTAs):
    16 and 64 rows and the largest fast-path size of each push segment."""
    out = set()
    for sg in comm.plain.segs:
        if sg["algo"] == "stock":
            continue
        h = sg["H"]
        for n in (16 * h, 64 * h, _fast_path_size(comm.plain, sg, cap)):
            if n and n // h >= 16 and sg["lo"] < 2 * n <= min(sg["hi"], cap):
                if comm.plain.supports(torch.empty(n, dtype=torch.bfloat16)):
                    out.add(n)
    return sorted(out)


def _pressure(rank, world, device, group):
    """Back-to-back calls next to a busy GEMM stream, with and without a
    delayed rank, eager and in a graph: bitwise results, no failed spin."""
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
    )

    comm = ParetoCommunicator(group, device)
    sizes = _pressure_sizes(comm)
    assert sizes, "no served size of >= 16 rows"
    cases = []
    for i, n in enumerate(sizes):
        xs = _inputs(700 + i, world, (n,), device)
        cases.append((n, xs[rank], _ref_all_reduce(xs)))
    g = torch.Generator(device=device).manual_seed(rank)
    a = torch.randn(8192, 8192, device=device, generator=g).to(torch.bfloat16)
    side = torch.cuda.Stream(device)
    main = torch.cuda.current_stream(device)

    def gemms(k):
        for _ in range(k):
            torch.mm(a, a)

    def run_eager(delay):
        side.wait_stream(main)
        with torch.cuda.stream(side):
            gemms(16)
        if delay:
            torch.cuda._sleep(50_000_000)  # rank 0 launches late
        return [(comm.all_reduce(x), ref, n) for _ in range(3) for n, x, ref in cases]

    try:
        for rep_ in range(2):
            for delay in (False, rank == 0):
                outs = run_eager(delay)
                torch.cuda.synchronize()
                for y, ref, n in outs:
                    assert y is not None, f"pressure n={n}: not served"
                    assert _same_bits(y, ref), (
                        f"pressure n={n} (delay {delay}): differs"
                    )
        # one graph: fork the GEMM stream, the all-reduces on the capture stream
        stat = [x.clone() for _, x, _ in cases]
        gr = torch.cuda.CUDAGraph()
        cap = torch.cuda.Stream(device)
        cap.wait_stream(main)
        with torch.cuda.stream(cap):
            with torch.cuda.graph(gr, stream=cap):
                side.wait_stream(cap)
                with torch.cuda.stream(side):
                    gemms(8)
                ys = [comm.all_reduce(x) for x in stat]
                cap.wait_stream(side)
        torch.cuda.synchronize()
        for _ in range(3):
            gr.replay()
            torch.cuda.synchronize()
            for y, (n, _, ref) in zip(ys, cases):
                assert _same_bits(y, ref), f"pressure graph n={n}: differs"
        assert comm.check() == 0, "a spin failed under pressure"
    finally:
        comm.close()


def _ipc(rank, world, device, group):
    """IPC memory per rank under every profile, with call tags: one instance
    <= 256 MiB, the workspace <= 1 GiB, little beyond it on the device."""
    from sglang.kernels.ops.communication.pareto_comm import table
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
        pareto_comm_platform,
    )

    arch, toolchain, _ = pareto_comm_platform()
    t = table.load_segments(arch, world, toolchain)
    stats = {}
    for i, (key, prof) in enumerate(t["profiles"].items()):
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        free0 = torch.cuda.mem_get_info(device)[0]
        comm = ParetoCommunicator(
            group,
            device,
            stock_profile=table.StockProfile(**prof),
            monitor=True,
            seq_check=True,
            warm=False,
        )
        try:
            torch.cuda.synchronize()
            used = free0 - torch.cuda.mem_get_info(device)[0]
            lay, total = comm.ws.layout()
            inst = max(nb for _, nb in lay.values())
            stats[key] = dict(
                ipc_mib=comm.bytes_per_rank / MIB,
                max_instance_mib=inst / MIB,
                instances=len(lay),
                device_mib=used / MIB,
            )
            assert total == comm.bytes_per_rank
            assert inst <= 256 * MIB, (key, inst)
            assert comm.bytes_per_rank <= 1024 * MIB, (key, comm.bytes_per_rank)
            if i:
                assert comm.bytes_per_rank <= used <= comm.bytes_per_rank + 64 * MIB, (
                    key,
                    used,
                    comm.bytes_per_rank,
                )
        finally:
            comm.close()
    if rank == 0:
        print("IPC_STATS " + json.dumps(dict(world=world, stats=stats)), flush=True)


def _failure_handling(rank, world, device, group):
    from sglang.kernels.ops.communication.pareto_comm import table
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
        _PlainAR,
        pareto_comm_platform,
    )

    arch, toolchain, _ = pareto_comm_platform()
    # the sizes: one per kernel of the selection
    plan = _PlainAR(world, 512 * MIB, table.load_segments(arch, world, toolchain))
    forms = dict(
        off={}, monitor=dict(monitor=True), seq=dict(monitor=True, seq_check=True)
    )
    fatal = []
    outs = {}
    for name, kw in forms.items():
        comm = ParetoCommunicator(group, device, on_fatal=fatal.append, **kw)
        try:
            got = {}
            for case, n in enumerate(_served(comm, plan)):
                for d in (0, 3):  # a size of whole rows and a partial last row
                    xs = _inputs(700 + 2 * case + d, world, (n - d,), device)
                    y = comm.all_reduce(xs[rank], site=3)
                    if y is None:  # a partial size's own plan may be the stock path
                        assert d or name == "off", f"{name} n={n}: not served"
                        continue
                    if (
                        d
                        and _kernel(comm.plain, xs[rank])
                        and not _kernel(comm.plain, xs[rank]).startswith(
                            "TwoShotEpochInst"
                        )
                    ):
                        # failure handling keeps the masked tail (no copy)
                        assert "masked tail" in _kernel(comm.plain, xs[rank]), name
                    assert _same_bits(y, _ref_all_reduce(xs)), f"{name} n={n - d}"
                    got[n - d] = y
            assert comm.check() == 0
            outs[name] = got
            if name == "seq":
                _seq_mismatch(comm, _served(comm, plan)[0], fatal, rank, device)
        finally:
            comm.close()
        assert not fatal or name == "seq", f"{name}: unexpected fatal event {fatal}"
    assert outs["monitor"].keys() == outs["seq"].keys()
    for n, b in outs["monitor"].items():
        assert _identical(b, outs["seq"][n]), f"forms differ at n={n}"
        if n in outs["off"]:
            assert _same_bits(outs["off"][n], b), f"forms differ at n={n}"
    # a mismatch on every other kernel and on a partial last row; a fresh
    # communicator each (the first mismatch is fatal)
    for n in _served(comm, plan)[1:] + [_served(comm, plan)[0] - 3]:
        fatal.clear()
        comm = ParetoCommunicator(
            group, device, on_fatal=fatal.append, monitor=True, seq_check=True
        )
        try:
            _seq_mismatch(comm, n, fatal, rank, device)
        finally:
            comm.close()


def _seq_mismatch(comm, n, fatal, rank, device):
    """Rank 0 runs a call of another site: every rank fails it (NaN, the
    mismatch record, a fatal event), the communicator refuses further calls and
    closes without its peers."""
    from sglang.kernels.ops.communication.pareto_comm.protocol import SEQ_MISMATCH

    x = torch.ones(n, dtype=torch.bfloat16, device=device)
    kernel = _kernel(comm.plain, x)
    y = comm.all_reduce(x, site=11 if rank == 0 else 12)
    torch.cuda.synchronize()
    deadline = time.monotonic() + 30
    while not fatal and time.monotonic() < deadline:
        time.sleep(0.05)
    assert fatal, f"call-sequence mismatch ({kernel}): no fatal event"
    assert torch.isnan(y.float()).all(), (
        f"call-sequence mismatch ({kernel}): a finite result"
    )
    assert int(comm.err.max().item()) >= SEQ_MISMATCH
    try:
        comm.all_reduce(x)
        raise AssertionError("an aborted communicator took a call")
    except RuntimeError:
        pass
    comm.close(abort=True)


def _test_bytes_table(world, int8=False):
    """A small synthetic segment table under the QuickReduce profiles
    (two_shot_epoch on the c12 or xe8 wire above 1 MiB)."""
    from sglang.kernels.ops.communication.pareto_comm import table

    w4 = 4096
    impls = {
        "one": dict(kind="one_shot", tier="exact", width=1024, geom=dict(NSEND=2)),
        "two": dict(
            kind="two_shot",
            tier="exact",
            width=w4,
            geom=dict(F=4, NSEND=world, RA=1, NSLOT=3, PIPE=1),
        ),
        "epoch": dict(kind="two_shot_epoch", tier="exact", width=w4, geom=dict(f=8)),
        "i8": dict(kind="two_shot_epoch", tier="int8", width=w4, geom=dict(wire="xe8")),
    }
    if not int8:
        # persistent channels, the senders pinned to their XCDs
        impls["epoch"] = dict(
            kind="two_shot_epoch",
            tier="exact",
            width=w4,
            geom=dict(f=8, npc=16, xs=2),
        )

    def segs(*rows):
        out, lo = [], 0
        for hi, head in rows:
            out.append(dict(lo=lo, hi=hi, head=head, bound="test", why="test"))
            lo = hi
        return out

    profiles, plans = {}, {}
    for key in table.PROFILE_KEYS:
        lv = None if key == "qr_off" else key.split("_")[1].upper()
        profiles[key] = dict(
            qr_level=lv,
            qr_min=0 if lv is None else (2 * MIB if key.endswith("_bf16") else MIB),
            qr_max=0 if lv is None else 2 << 30,
            ca_min=0,
            ca_max=64 * MIB if lv is None else MIB,
        )
        big = "epoch"
        if int8 and lv == "INT8":
            big = "i8"
        plans[f"{key}/fast"] = plans[f"{key}/aligned"] = segs(
            (1024, "one"), (64 << 10, "two"), (MIB, "stock"), (8 * MIB, big)
        )
        plans[f"{key}/unaligned"] = segs(
            (1024, "one"), (MIB, "two"), (8 * MIB, "epoch")
        )
        plans[f"{key}/padded"] = segs((8 * MIB, big))
    return table.validate(
        dict(
            kind="ar_segments",
            arch="gfx950",
            tp=world,
            toolchain="triton-3.7.0",
            source="test table (synthetic choices)",
            impls=impls,
            profiles=profiles,
            plans=plans,
        )
    )


def _plain(rank, world, device, group):
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
    )

    comm = ParetoCommunicator(
        group, device, max_bytes=8 * MIB, segments_table=_test_bytes_table(world)
    )
    try:
        assert any(
            sp.inst.specs() != {None}
            for sp in comm.plain.specs
            if sp.kind == "two_shot_epoch"
        )
        sizes = [1, 7, 100, 512, 513, 4097, 30000, 40001, 3_000_000, 3_000_001]
        for case, n in enumerate(sizes):
            _check(comm, 300 + case, n, rank, world, device, group)
        # the stock range (aligned 64 KiB .. 1 MiB) is refused; a 2-D view of the
        # same bytes too (selection by bytes, whatever the shape)
        for shape in ((40000,), (10, 4000), (5000, 8)):
            assert (
                comm.all_reduce(torch.zeros(shape, dtype=torch.bfloat16, device=device))
                is None
            ), shape
        assert comm.check() == 0
    finally:
        comm.close()


def _test_fused_table(world):
    """The plain test table + fused plans for every capability class:
    one_shot up to 64 KiB where it can run, two_shot to 4 MiB, the stock path
    to 8 MiB, then two_shot_epoch (persistent, pipelined finisher) where it can
    run, else two_shot, to 32 MiB."""
    t = _test_bytes_table(world)

    def segs(*rows):
        out, lo = [], 0
        for hi, head in rows:
            out.append(dict(lo=lo, hi=hi, head=head, bound="test", why="test"))
            lo = hi
        return out

    t["fused"] = {
        "ADD_RMS": dict(
            profile="qr_off",
            floor="bf16",
            plans={
                "two_shot": segs(
                    (4 * MIB, "two"), (8 * MIB, "stock"), (32 * MIB, "two")
                ),
                "one_shot+two_shot": segs(
                    (64 << 10, "one"),
                    (4 * MIB, "two"),
                    (8 * MIB, "stock"),
                    (32 * MIB, "two"),
                ),
                "two_shot+two_shot_epoch": segs(
                    (4 * MIB, "two"), (8 * MIB, "stock"), (32 * MIB, "epoch")
                ),
                "one_shot+two_shot+two_shot_epoch": segs(
                    (64 << 10, "one"),
                    (4 * MIB, "two"),
                    (8 * MIB, "stock"),
                    (32 * MIB, "epoch"),
                ),
            },
        )
    }
    return t


def _fused_inputs(case, world, m, h, device):
    xs = _inputs(case, world, (m, h), device)
    g = torch.Generator(device=device).manual_seed(77 + case)
    res = (torch.randn(m, h, device=device, generator=g)).to(torch.bfloat16)
    res.view(-1)[:4] = -0.0
    w = (1 + 0.1 * torch.randn(h, device=device, generator=g)).to(torch.bfloat16)
    return xs, res, w


def _ref_fused(xs, res, w, eps):
    s = _ref_all_reduce(xs).float() + res.float()
    out = s * torch.rsqrt(s.pow(2).mean(-1, keepdim=True) + eps) * w.float()
    return out.to(torch.bfloat16), s.to(torch.bfloat16)


def _ulp(got, want):
    g, w = got.float(), want.float()
    e = torch.floor(torch.log2(w.abs().clamp_min(1e-30)))
    d = (g - w).abs() / torch.pow(2.0, e - 7)
    big = w.abs() > 1e-3 * w.abs().amax(-1, keepdim=True)
    return float((d * big).max()) if g.numel() else 0.0


def _check_fused(comm, case, m, h, rank, world, device, group, site=0):
    eps = 1e-6
    xs, res, w = _fused_inputs(case, world, m, h, device)
    r = comm.fused_allreduce_rmsnorm(xs[rank], res, w, eps, site=site)
    assert r is not None, f"fused [{m}, {h}]: not served"
    out, res_out = r
    want_out, want_res = _ref_fused(xs, res, w, eps)
    assert _same_bits(res_out, want_res), f"fused [{m}, {h}]: residual differs"
    assert _ulp(out, want_out) <= 1.0, (
        f"fused [{m}, {h}]: out {_ulp(out, want_out)} ulp"
    )
    assert _same_everywhere(out, group) and _same_everywhere(res_out, group)


def _fused_ms(comm, h):
    out = []
    for lo, hi, _ in comm.fused.sel.get(h, ()):
        out += sorted({lo, hi, (lo + hi) // 2 | 1})
    return out


def _fused(rank, world, device, group, monitor=False):
    import random

    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
    )

    widths = (6144, 4096, 7168) if world > 2 else (6144, 4096)
    comm = ParetoCommunicator(
        group,
        device,
        max_bytes=8 * MIB,
        segments_table=_test_fused_table(world),
        hidden_sizes=widths,
        monitor=monitor,
        seq_check=monitor,
        on_fatal=(lambda info: None) if monitor else None,
    )
    try:
        assert comm.fused is not None and sorted(comm.fused.sel) == sorted(widths)
        case = 0
        for h in widths:
            for m in _fused_ms(comm, h):
                case += 1
                _check_fused(comm, 2000 + case, m, h, rank, world, device, group, case)
        # one CUDA graph of several fused calls, replayed with new inputs
        h = widths[0]
        ms = _fused_ms(comm, h)[:: max(1, len(_fused_ms(comm, h)) // 4)][:4]
        bufs = [_fused_inputs(3000 + i, world, m, h, device) for i, m in enumerate(ms)]
        xs_in = [b[0][rank].clone() for b in bufs]
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        g = torch.cuda.CUDAGraph()
        with torch.cuda.stream(s), torch.cuda.graph(g, stream=s):
            outs = [
                comm.fused_allreduce_rmsnorm(x, b[1], b[2], 1e-6)
                for x, b in zip(xs_in, bufs)
            ]
        for rep in range(3):
            refs = []
            for i, (m, b) in enumerate(zip(ms, bufs)):
                xs, res, w = _fused_inputs(3100 + 10 * rep + i, world, m, h, device)
                xs_in[i].copy_(xs[rank])
                b[1].copy_(res)
                b[2].copy_(w)
                refs.append(_ref_fused(xs, res, w, 1e-6))
            g.replay()
            torch.cuda.synchronize()
            for (o, r), (wo, wr) in zip(outs, refs):
                assert _same_bits(r, wr) and _ulp(o, wo) <= 1.0, "graph replay"
        del g
        # a rotating late rank
        for i, m in enumerate(_fused_ms(comm, h)[:6]):
            if rank == i % world:
                torch.cuda._sleep(200_000)
            _check_fused(comm, 3500 + i, m, h, rank, world, device, group)
        # plain and fused calls interleaved at random (one seed on every rank)
        rnd = random.Random(7)
        for i in range(60):
            hh = rnd.choice(widths)
            m = rnd.choice(_fused_ms(comm, hh))
            if rnd.random() < 0.5:
                _check(
                    comm,
                    4000 + i,
                    rnd.choice([100, 4097, 30000]),
                    rank,
                    world,
                    device,
                    group,
                )
            else:
                _check_fused(comm, 4000 + i, m, hh, rank, world, device, group)
        # refusals: no launch, no counter touched
        before = comm.ws.ctr_page.clone()
        x = torch.zeros(4, h, dtype=torch.bfloat16, device=device)
        w = torch.ones(h, dtype=torch.bfloat16, device=device)
        top = max(hi for _, hi, _ in comm.fused.sel[h])
        big = torch.zeros(top + 1, h, dtype=torch.bfloat16, device=device)
        assert comm.fused_allreduce_rmsnorm(big, big, w, 1e-6) is None
        assert comm.fused_allreduce_rmsnorm(x, x.float(), w, 1e-6) is None
        x2 = torch.zeros(4, 2560, dtype=torch.bfloat16, device=device)
        assert comm.fused_allreduce_rmsnorm(x2, x2, w[:2560], 1e-6) is None
        stock_m = (6 * MIB) // (2 * h)
        xs_ = torch.zeros(stock_m, h, dtype=torch.bfloat16, device=device)
        assert comm.fused_allreduce_rmsnorm(xs_, xs_, w, 1e-6) is None
        torch.cuda.synchronize()
        assert torch.equal(before, comm.ws.ctr_page)
        assert comm.check() == 0
    finally:
        comm.close()


def _fused_monitor(rank, world, device, group):
    _fused(rank, world, device, group, monitor=True)


def _qr_int8(group, device, level="INT8"):
    """sglang's QuickReduce at a quantized level (None where it is not
    available)."""
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = level
    try:
        from sglang.srt.distributed.device_communicators.quick_all_reduce import (
            QuickAllReduce,
        )

        qr = QuickAllReduce(group, device)
    except Exception:
        return None
    return None if qr.disabled else qr


def _int8(rank, world, device, group):
    from sglang.kernels.ops.communication.pareto_comm import table, two_shot_epoch
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
        pareto_comm_platform,
    )

    # custom all-reduce up to 1 MiB, QuickReduce INT8 above
    prof = table.StockProfile(ca_max=MIB, qr_level="INT8", qr_min=MIB, qr_max=2 << 30)
    tt = _test_bytes_table(world, int8=True)
    # aligned element counts, 2 .. 8 MiB (whole 4096-rows and zero-padded)
    sizes = [MIB, MIB + 40 * world, 3 * MIB // 2, 4 * MIB - 8 * world]
    # default profile: the same sizes take the lossless wire
    comm = ParetoCommunicator(group, device, max_bytes=8 * MIB, segments_table=tt)
    try:
        for case, n in enumerate(sizes):
            _check(comm, 50 + case, n, rank, world, device, group)
        assert all(
            sp.inst.wire == 1 for sp in comm.plain.specs if sp.kind == "two_shot_epoch"
        ), "an int8 instance without QuickReduce INT8"
    finally:
        comm.close()
    comm = ParetoCommunicator(
        group, device, max_bytes=8 * MIB, segments_table=tt, stock_profile=prof
    )
    qr = _qr_int8(group, device)
    stats = []
    try:
        assert any(
            isinstance(sp.inst, two_shot_epoch.TwoShotEpochInst) and sp.inst.wire == 10
            for sp in comm.plain.specs
        )
        for case, n in enumerate(sizes):
            y = _check(comm, 60 + case, n, rank, world, device, group, ref=_ref_xe8)
            xs = _inputs(60 + case, world, (n,), device)
            st = dict(n=n, push=_err_stats(y, xs))
            if qr is not None and (2 * n) % 16 == 0:
                st["qr_int8"] = q8 = _err_stats(qr.quick_all_reduce(xs[rank]), xs)
                # every metric no worse than QuickReduce INT8; the bias within
                # noise or no worse
                for k in ("relrms", "q99", "q999", "q9999", "maxabs", "meanabs"):
                    assert st["push"][k] <= q8[k], (n, k, st)
                assert st["push"]["bias"] <= max(
                    q8["bias"], 4 * st["push"]["bias_noise"]
                ), (n, st)
            stats.append(st)
        _graph(comm, rank, world, device, group, _ref_xe8, sizes[:2])
        _wrap(comm, rank, world, device, group, _ref_xe8, sizes[:1])
    finally:
        comm.close()
        if qr is not None:
            qr.close()
    _lossy_seq_check(group, device, tt, prof, 10, sizes, 60, _ref_xe8)
    # the shipped table: int8 only where QuickReduce INT8 runs
    arch, toolchain, _ = pareto_comm_platform()
    t = table.load_segments(arch, world, toolchain)
    qmin = {2: 8 * MIB, 4: 64 * MIB, 8: 2048 * MIB}[world]  # sglang bf16 minimum
    sp = table.StockProfile(qr_level="INT8", qr_min=qmin, qr_max=2 << 30)
    segs = [s for s in table.plain_segments(t, sp) if s["geom"].get("wire") == "xe8"]
    assert all(s["lo"] >= qmin for s in segs)
    assert not any(s["geom"].get("wire") == "xe8" for s in table.plain_segments(t)), (
        "xe8 in the default plan"
    )
    if segs and segs[0]["lo"] < 256 * MIB:
        n = (segs[0]["lo"] + 64 * MIB) // 2
        comm = ParetoCommunicator(
            group, device, max_bytes=segs[0]["lo"] + 64 * MIB, stock_profile=sp
        )
        try:
            y = _check(comm, 90, n, rank, world, device, group, ref=_ref_xe8)
            stats.append(
                dict(
                    n=n,
                    shipped=True,
                    push=_err_stats(y, _inputs(90, world, (n,), device)),
                )
            )
        finally:
            comm.close()
    if rank == 0:
        print("INT8_STATS " + json.dumps(dict(world=world, stats=stats)), flush=True)


def _lossy_seq_check(group, device, tt, prof, wire, sizes, case0, ref):
    """A lossy wire with failure handling: the call tags ride its rows,
    the same bits as without; a call-sequence mismatch on it is fatal."""
    from sglang.kernels.ops.communication.pareto_comm import two_shot_epoch
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
    )

    rank, world = dist.get_rank(group), dist.get_world_size(group)
    for mismatch in (False, True):
        fatal = []
        comm = ParetoCommunicator(
            group,
            device,
            max_bytes=8 * MIB,
            segments_table=tt,
            stock_profile=prof,
            monitor=True,
            seq_check=True,
            on_fatal=fatal.append,
        )
        try:
            if mismatch:
                x = torch.ones(sizes[0], dtype=torch.bfloat16, device=device)
                assert _kernel(comm.plain, x).startswith("TwoShotEpochInst"), _kernel(
                    comm.plain, x
                )
                _seq_mismatch(comm, sizes[0], fatal, rank, device)
                continue
            assert any(
                isinstance(sp.inst, two_shot_epoch.TwoShotEpochInst)
                and sp.inst.wire == wire
                and sp.inst.tag
                for sp in comm.plain.specs
            )
            for case, n in enumerate(sizes):
                _check(comm, case0 + case, n, rank, world, device, group, ref=ref)
            assert comm.check() == 0 and not fatal
        finally:
            comm.close()


def _worker(rank, world, port, mode, q):
    try:
        device, group = _init(rank, world, port)
        {
            "main": _main,
            "failure": _failure_handling,
            "plain": _plain,
            "int8": _int8,
            "streams": _streams,
            "pressure": _pressure,
            "ipc": _ipc,
            "fused": _fused,
            "fused_monitor": _fused_monitor,
        }[mode](rank, world, device, group)
        q.put((rank, None))
    except Exception:
        q.put((rank, traceback.format_exc()))
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _run(world, mode, timeout=1500):
    ctx = multiprocessing.get_context("spawn")
    q = ctx.Queue()
    port = _open_port()
    ps = [
        ctx.Process(target=_worker, args=(r, world, port, mode, q))
        for r in range(world)
    ]
    for p in ps:
        p.start()
    deadline = time.monotonic() + timeout
    errs = {}
    for _ in ps:
        try:
            r, e = q.get(timeout=max(1, deadline - time.monotonic()))
            errs[r] = e
        except Exception:
            break
    for p in ps:
        p.join(max(0, deadline - time.monotonic()))
        if p.is_alive():
            p.terminate()
            p.join()
    bad = {r: e for r, e in errs.items() if e}
    assert len(errs) == world, (
        f"{mode} TP{world}: only {sorted(errs)} reported (timeout or crash)"
    )
    assert not bad, f"{mode} TP{world} failed:\n" + "\n".join(
        f"rank {r}:\n{e}" for r, e in bad.items()
    )


class TestParetoCommAllReduce(CustomTestCase):
    pass


def _case(world, mode):
    def test(self):
        if not _available():
            self.skipTest("pareto-comm needs ROCm gfx950 with Triton 3.7")
        if torch.cuda.device_count() < world:
            self.skipTest(f"needs {world} GPUs")
        _run(world, mode)

    return test


for _mode, _worlds in (
    ("main", (2, 4, 8)),
    ("failure", (4, 8)),
    ("streams", (2, 8)),
    ("pressure", (2, 8)),
    ("ipc", (2, 4, 8)),
    ("plain", (4,)),
    ("int8", (2, 4, 8)),
    ("fused", (2, 4, 8)),
    ("fused_monitor", (4,)),
):
    for _w in _worlds:
        setattr(TestParetoCommAllReduce, f"test_{_mode}_tp{_w}", _case(_w, _mode))


if __name__ == "__main__":
    unittest.main()
