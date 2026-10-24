"""Benchmark push-comm against the all-reduce backends SGLang selects on
ROCm, each called at the layer the dispatch calls it (custom all-reduce,
QuickReduce per level, RCCL, tensor_model_parallel_all_reduce), plus an A/A
arm. Modes: eager (one call after a cold-L2 flush and a host barrier),
eager_stream (back-to-back), graph.

  torchrun --nproc_per_node=8 benchmark/kernels/all_reduce/benchmark_push_comm.py \\
    --hidden 6144 --tokens 1,64,512,4096 --mode all --cold-l2 --csv-out tp8.csv"""

import argparse
import csv
import gc
import math
import os
import statistics
from typing import Callable, Dict, List, Optional

import torch
import torch.distributed as dist

ARMS = ("push", "push_aa", "push_qr8", "stock", "ca", "qr_fp", "qr_int8", "rccl")
_L2_FLUSH_RATIO = 5


def parse_list(raw: str) -> List[int]:
    return [int(v) for v in raw.split(",") if v.strip()]


def parse_args():
    p = argparse.ArgumentParser(
        description="push-comm vs SGLang's all-reduce backends (AMD gfx950)."
    )
    p.add_argument(
        "--hidden", type=str, default="6144", help="comma-separated row widths H"
    )
    p.add_argument(
        "--tokens", type=str, default="1,2,4,8,16,32,64,128,256,512,1024,2048,4096"
    )
    p.add_argument(
        "--bytes", type=str, default="", help="comma-separated flat message sizes"
    )
    p.add_argument("--arms", type=str, default=",".join(ARMS))
    p.add_argument("--max-bytes", type=int, default=1 << 30, help="push max_bytes")
    p.add_argument(
        "--mode",
        type=str,
        default="all",
        help="comma-separated of eager (one call per timing, after an L2 flush "
        "with --cold-l2: includes the host launch cost), eager_stream "
        "(back-to-back eager calls, steady-state throughput), graph; or all",
    )
    p.add_argument("--cold-l2", action="store_true")
    p.add_argument(
        "--no-eager-barrier",
        action="store_true",
        help="eager mode: no host barrier before each call (the time then "
        "includes the skew between the ranks' host launches)",
    )
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--max-copies", type=int, default=256)
    p.add_argument("--csv-out", type=str, default=None)
    p.add_argument(
        "--plan",
        action="store_true",
        help="CPU only: print the shipped tables' coverage and exit",
    )
    return p.parse_args()


def print_plan():
    """The byte segments of the shipped tables (sglang's default all-reduce
    path, aligned sizes on the custom all-reduce's fast path)."""
    from sglang.kernels.ops.communication.push_comm import table

    for tp in (2, 4, 8):
        t = table.load_segments("gfx950", tp, "triton-3.7.0")
        if t is None:
            print(f"TP{tp}: no table")
            continue
        segs = table.plain_segments(t)
        print(
            f"TP{tp}: " + " ".join(f"({s['lo']}, {s['hi']}]:{s['name']}" for s in segs)
        )


def kernel_name(comm, x) -> Optional[str]:
    """The kernel and row width push-comm serves x with (None: refused)."""
    pk = None if comm is None or comm.plain is None else comm.plain._pick(x)
    if pk is None:
        return None
    name = type(pk[0]).__name__
    lossy = name == "TwoShotEpochInst" and pk[0].wire != 1
    return f"{name} H={pk[1]}" + (" xe8" if lossy else "")


def reference(xs: List[torch.Tensor]) -> torch.Tensor:
    acc = xs[0].float()
    for t in xs[1:]:
        acc = acc + t.float()
    return acc.to(torch.bfloat16)


def canon(t: torch.Tensor) -> torch.Tensor:
    return torch.where(t.view(torch.int16) == -32768, torch.zeros_like(t), t)


def inputs(shape, world: int, case: int, device) -> List[torch.Tensor]:
    """Every rank's input (each rank builds all of them for the reference)."""
    out = []
    for r in range(world):
        g = torch.Generator(device=device).manual_seed(1000 * case + r)
        out.append(torch.randn(shape, device=device, generator=g).to(torch.bfloat16))
    return out


def quick_all_reduce(group, device, level: str, max_mb: int):
    """A QuickAllReduce at one level, or None where it does not initialize."""
    from sglang.srt.distributed.device_communicators.quick_all_reduce import (
        QuickAllReduce,
    )

    saved = {
        k: os.environ.get(k)
        for k in (
            "ROCM_QUICK_REDUCE_QUANTIZATION",
            "ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB",
        )
    }
    os.environ["ROCM_QUICK_REDUCE_QUANTIZATION"] = level
    os.environ["ROCM_QUICK_REDUCE_MAX_SIZE_BYTES_MB"] = str(max_mb)
    try:
        qr = QuickAllReduce(group=group, device=device)
    finally:
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    return None if qr.disabled else qr


def time_eager(
    fn: Callable, flush, warmup, iters, repeats, barrier_group=None
) -> List[float]:
    """Per-repeat median of per-call times (us); flush (or None) before each
    call and, with barrier_group, a host barrier after the flush, so every
    rank issues the call at the same time."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    reps = []
    for _ in range(repeats):
        dist.barrier()
        ev = [
            (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
            for _ in range(iters)
        ]
        for s, e in ev:
            if flush is not None:
                flush.zero_()
            if barrier_group is not None:
                torch.cuda.synchronize()
                dist.barrier(group=barrier_group)
            s.record()
            fn()
            e.record()
        torch.cuda.synchronize()
        reps.append(statistics.median(s.elapsed_time(e) * 1000.0 for s, e in ev))
    return reps


def time_stream(fn: Callable, warmup, iters, repeats) -> List[float]:
    """Per-repeat mean time (us) of iters back-to-back calls, no flush."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    reps = []
    for _ in range(repeats):
        dist.barrier()
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        for _ in range(iters):
            fn()
        e.record()
        e.synchronize()
        reps.append(s.elapsed_time(e) * 1000.0 / iters)
    return reps


def time_graph(graph, ncalls, warmup, iters, repeats) -> List[float]:
    """Per-repeat time per call (us) of a graph holding ncalls calls."""
    for _ in range(warmup):
        graph.replay()
    torch.cuda.synchronize()
    reps = []
    for _ in range(repeats):
        dist.barrier()
        s, e = (
            torch.cuda.Event(enable_timing=True),
            torch.cuda.Event(enable_timing=True),
        )
        s.record()
        for _ in range(max(1, iters // ncalls)):
            graph.replay()
        e.record()
        e.synchronize()
        reps.append(s.elapsed_time(e) * 1000.0 / (max(1, iters // ncalls) * ncalls))
    return reps


def capture(fns: List[Callable]):
    """One CUDA graph with every call of fns (SGLang's capture stream), or None
    when a call refuses."""
    from sglang.srt.distributed.parallel_state import graph_capture

    outs = []
    with graph_capture() as ctx:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g, stream=ctx.stream):
            for fn in fns:
                outs.append(fn())
    if any(o is None for o in outs):
        return None
    return g


def main():
    args = parse_args()
    if args.plan:
        print_plan()
        return
    hiddens, tokens = parse_list(args.hidden), parse_list(args.tokens)
    arms = [a for a in args.arms.split(",") if a]
    assert set(arms) <= set(ARMS), f"unknown arms {set(arms) - set(ARMS)}"

    from sglang.kernels.ops.communication.push_comm import table
    from sglang.kernels.ops.communication.push_comm.two_shot_epoch import (
        TwoShotEpochInst,
    )
    from sglang.srt.distributed.communication_op import tensor_model_parallel_all_reduce
    from sglang.srt.distributed.device_communicators.push_comm import (
        PushCommunicator,
        push_comm_platform,
    )
    from sglang.srt.distributed.parallel_state import (
        destroy_distributed_environment,
        destroy_model_parallel,
        get_tp_group,
        init_distributed_environment,
        initialize_model_parallel,
        set_custom_all_reduce,
    )
    from sglang.test.test_utils import publish_build_topology

    # torchrun's rendezvous variables
    rank = int(os.environ.get("RANK", "0"))
    world = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    device = torch.device(f"cuda:{local_rank}")
    torch.cuda.set_device(device)
    set_custom_all_reduce(True)
    init_distributed_environment(
        world_size=world,
        rank=rank,
        local_rank=local_rank,
        distributed_init_method="env://",
        backend="nccl",
    )
    publish_build_topology(world_rank=rank, tp_size=world)
    initialize_model_parallel()
    tp = get_tp_group()

    arch, toolchain, _ = push_comm_platform()
    seg_table = table.load_segments(arch, world, toolchain) if arch else None
    comms = {}
    if {"push", "push_aa"} & set(arms):
        comms["push"] = PushCommunicator(tp.cpu_group, device, max_bytes=args.max_bytes)
    if "push_qr8" in arms and seg_table is not None:
        comms["push_qr8"] = PushCommunicator(
            tp.cpu_group,
            device,
            max_bytes=args.max_bytes,
            stock_profile=table.StockProfile(**seg_table["profiles"]["qr_int8"]),
        )
    shapes = [((m, h), f"{m}x{h}") for h in hiddens for m in tokens]
    shapes += [((b // 2,), f"{b}B") for b in parse_list(args.bytes)]
    max_mb = max(2 * math.prod(s) for s, _ in shapes) // (1 << 20) + 1
    qrs = {}
    for arm, level in (("qr_fp", "FP"), ("qr_int8", "INT8")):
        if arm in arms:
            qrs[arm] = quick_all_reduce(tp.cpu_group, device, level, max_mb)
    l2 = torch.cuda.get_device_properties(device).L2_cache_size
    flush_bytes = _L2_FLUSH_RATIO * l2
    flush = (
        torch.empty(flush_bytes, dtype=torch.uint8, device=device)
        if args.cold_l2
        else None
    )

    def arm_fn(arm: str, x: torch.Tensor) -> Optional[Callable]:
        """The call of one arm on x, or None where the backend refuses x."""
        if arm in ("push", "push_aa", "push_qr8"):
            c = comms.get("push_qr8" if arm == "push_qr8" else "push")
            if c is None or not c.should_push_ar(x):
                return None
            return lambda: c.all_reduce(x)
        if arm == "stock":
            return lambda: tensor_model_parallel_all_reduce(x)
        if arm == "ca":
            ca = tp.ca_comm
            if ca is None or ca.disabled or not ca.should_custom_ar(x):
                return None
            return lambda: ca.custom_all_reduce(x)
        if arm in ("qr_fp", "qr_int8"):
            qr = qrs.get(arm)
            nb = x.numel() * x.element_size()
            if qr is None or nb % 16 or nb > qr.qr_max_size:
                return None
            return lambda: qr.quick_all_reduce(x)
        if arm == "rccl":

            def rccl():
                dist.all_reduce(x, group=tp.device_group)
                return x

            return rccl
        raise ValueError(arm)

    modes = (
        ("eager", "eager_stream", "graph")
        if args.mode == "all"
        else tuple(args.mode.split(","))
    )
    assert set(modes) <= {"eager", "eager_stream", "graph"}, modes
    rows: List[Dict] = []
    if rank == 0:
        print(
            f"world={world} arms={arms} cold_l2={args.cold_l2} (flush {flush_bytes} B)"
        )
        print(
            "| mode | shape | bytes | stock method | "
            + " | ".join(arms)
            + " | push exact |"
        )
        print("|---" * (len(arms) + 5) + "|")
    for mode in modes:
        for case, (shape, name) in enumerate(shapes):
            xs = inputs(shape, world, case, device)
            nbytes = 2 * xs[rank].numel()
            ref = reference(xs)
            exact = {}
            for arm in ("push", "push_qr8"):
                if arm in arms:
                    fn = arm_fn(arm, xs[rank])
                    if fn is not None:
                        y = fn()
                        torch.cuda.synchronize()
                        inst = comms[arm].plain._pick(xs[rank])[0]
                        lossy = isinstance(inst, TwoShotEpochInst) and inst.wire != 1
                        if not lossy:
                            exact[arm] = torch.equal(
                                canon(y).view(torch.int16), canon(ref).view(torch.int16)
                            )
                        else:
                            err = (y.float() - ref.float()).norm() / ref.float().norm()
                            exact[arm] = bool(err < 0.05)
            ncopies = 1
            if mode == "graph" and args.cold_l2 and nbytes < flush_bytes:
                ncopies = min(args.max_copies, -(-flush_bytes // nbytes) + 1)
            copies = [xs[rank].clone() for _ in range(max(2, ncopies))]
            resolved = (
                tp._resolve_outplace_all_reduce_method(input_=copies[0]) or "inplace"
            )
            res: Dict[str, Optional[List[float]]] = {}
            for arm in arms:
                fns = [arm_fn(arm, c) for c in copies]
                if fns[0] is None:
                    res[arm] = None
                    continue
                if mode == "graph":
                    g = capture(fns)
                    res[arm] = (
                        None
                        if g is None
                        else time_graph(
                            g, len(fns), args.warmup, args.iters, args.repeats
                        )
                    )
                    del g
                elif mode == "eager_stream":
                    res[arm] = time_stream(
                        fns[0], args.warmup, args.iters, args.repeats
                    )
                else:
                    res[arm] = time_eager(
                        fns[0],
                        flush,
                        args.warmup,
                        args.iters,
                        args.repeats,
                        None if args.no_eager_barrier else tp.cpu_group,
                    )
            # mean over ranks of each repeat; -1 marks a refused size
            vec = []
            for arm in arms:
                vec += res[arm] if res[arm] is not None else [-1.0] * args.repeats
            t = torch.tensor(vec, dtype=torch.float64, device=device)
            dist.all_reduce(t, group=tp.device_group)
            t /= world
            ok = torch.tensor([int(all(exact.values())) if exact else 1], device=device)
            dist.all_reduce(ok, op=dist.ReduceOp.MIN, group=tp.device_group)
            if rank == 0:
                vals = t.tolist()
                row = dict(
                    mode=mode,
                    world=world,
                    shape=name,
                    bytes=nbytes,
                    stock_method=resolved,
                    push_kernel=kernel_name(comms.get("push"), xs[rank]),
                    push_qr8_kernel=kernel_name(comms.get("push_qr8"), xs[rank]),
                    push_exact=bool(ok.item()),
                    cold_l2=bool(args.cold_l2),
                    copies=len(copies),
                )
                for i, arm in enumerate(arms):
                    r = vals[i * args.repeats : (i + 1) * args.repeats]
                    if r[0] < 0:
                        row[f"{arm}_us"] = row[f"{arm}_min"] = row[f"{arm}_max"] = None
                    else:
                        row[f"{arm}_us"] = statistics.median(r)
                        row[f"{arm}_min"], row[f"{arm}_max"] = min(r), max(r)
                rows.append(row)
                cells = [
                    "stock"
                    if row[f"{a}_us"] is None and a.startswith("push")
                    else ("-" if row[f"{a}_us"] is None else f"{row[f'{a}_us']:.1f}")
                    for a in arms
                ]
                print(
                    f"| {mode} | {name} | {nbytes} | {resolved} | "
                    + " | ".join(cells)
                    + f" | {row['push_exact']} |",
                    flush=True,
                )
            del copies
    if rank == 0 and args.csv_out and rows:
        os.makedirs(os.path.dirname(args.csv_out) or ".", exist_ok=True)
        with open(args.csv_out, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    # release the captured graphs (they hold the device group's communicator)
    # before tearing the groups down
    gc.collect()
    torch.cuda.synchronize()
    for c in comms.values():
        c.close()
    for qr in qrs.values():
        if qr is not None:
            qr.close()
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
