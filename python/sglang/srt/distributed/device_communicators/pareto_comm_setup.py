# SPDX-License-Identifier: Apache-2.0
"""Runtime wiring of pareto-comm (--enable-pareto-comm).

pareto-comm sits in front of the group's own all-reduce path. Per
message-size segment its table takes the fastest of its own kernels and that
path; it serves only the segments where its own kernel wins, every other call
is refused and runs on the custom all-reduce, QuickReduce or RCCL exactly as
without it. Those communicators are therefore always built as usual, and the
communicator is told what they serve (the stock profile) so that it never
replaces a faster path or a path of a different precision.

Built only for the TP group, on one node, before CUDA graph capture, by every
rank of the group at the same time (inside GroupCoordinator.__init__, i.e.
before the KV cache memory baseline is measured, so its IPC memory is
accounted for like the custom all-reduce's).
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch

from sglang.srt.utils import get_bool_env_var, is_gfx95_supported, is_hip

logger = logging.getLogger(__name__)

# The probe covers 16 B .. 4 GiB bf16 messages in 16-byte units.
_PROBE_MAX_LOG2_UNITS = 28


def pareto_comm_blockers() -> List[str]:
    """Server configurations pareto-comm is not wired for (empty: none).

    Read once at startup; a non-empty list keeps the communicator unbuilt (on
    anything but ROCm gfx950 nothing of pareto-comm is imported)."""
    from sglang.srt.runtime_context import (
        get_disagg,
        get_exec,
        get_parallel,
    )

    if not (is_hip() and is_gfx95_supported()):
        return ["not ROCm gfx950"]
    parallel, exec_ = get_parallel(), get_exec()
    out = []
    if parallel.nnodes != 1:
        out.append("multi-node")
    if parallel.attn_dp_size > 1:
        out.append("DP attention")
    if exec_.comm.disable_custom_all_reduce:
        # the segment tables assume the custom all-reduce as the stock path
        out.append("--disable-custom-all-reduce")
    if exec_.deterministic.enable_deterministic_inference:
        out.append("deterministic inference")
    if exec_.graph.enable_torch_compile:
        out.append("torch.compile")
    if (
        exec_.overlap.enable_two_batch_overlap
        or exec_.overlap.enable_single_batch_overlap
    ):
        out.append("two-batch / single-batch overlap")
    if get_disagg().enable_pdmux:
        out.append("PD multiplexing")
    if exec_.moe.elastic_ep_backend is not None:
        out.append("elastic EP")
    return out


def _probe_window(accepts) -> Optional[tuple]:
    """(lo, hi): the bf16 message bytes lo < n <= hi that `accepts` takes (one
    contiguous window, 16-byte granularity), (0, 0) when it takes none, None
    when the window is not contiguous. Asked on meta tensors (no memory, no
    launch)."""

    def picks(units: int) -> bool:
        return bool(accepts(_meta(16 * units)))

    p2 = [1 << k for k in range(_PROBE_MAX_LOG2_UNITS + 1)]
    hit = [picks(u) for u in p2]
    if not any(hit):
        return 0, 0
    i0 = hit.index(True)
    i1 = len(hit) - 1 - hit[::-1].index(True)
    # first accepted size in (lo, hi]
    lo, hi = (p2[i0 - 1] if i0 else 0), p2[i0]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (lo, mid) if picks(mid) else (mid, hi)
    first = hi
    # last accepted size in [lo, hi)
    lo = p2[i1]
    hi = p2[i1 + 1] if i1 + 1 < len(p2) else p2[i1] + 1
    while hi - lo > 1:
        mid = (lo + hi) // 2
        lo, hi = (mid, hi) if picks(mid) else (lo, mid)
    last = lo
    for u in {first, (first + last) // 2, last}:
        if not picks(u):
            return None
    return 16 * (first - 1), 16 * last


def _meta(nbytes: int) -> torch.Tensor:
    return torch.empty(nbytes // 2, dtype=torch.bfloat16, device="meta")


def _quick_reduce_level(qr_comm) -> Optional[str]:
    """QuickReduce quantization level name of an enabled qr_comm, or None when
    it cannot be read (unknown implementation)."""
    inner = getattr(qr_comm, "_communicator", qr_comm)
    level = getattr(inner, "qr_quant_level", None)
    return getattr(level, "name", None)


def stock_profile_of(group):
    """The group's own all-reduce path by message size (StockProfile), probed
    before pareto-comm is inserted: the custom all-reduce's and QuickReduce's
    own size windows (their eligibility checks), QuickReduce's level, and which
    of the two the dispatch chain prefers where both serve a size. The result
    is checked against the chain itself at every probed size. None when it
    cannot be expressed (then pareto-comm stays off)."""
    from sglang.kernels.ops.communication.pareto_comm.table import (
        QR_ARMS,
        StockProfile,
    )

    resolve = group._resolve_outplace_all_reduce_method
    ca_comm, qr_comm = group.ca_comm, group.qr_comm
    ca = (0, 0)
    if ca_comm is not None and not ca_comm.disabled:
        ca = _probe_window(ca_comm.should_custom_ar)
    qr = (0, 0)
    if qr_comm is not None and not qr_comm.disabled:
        qr = _probe_window(qr_comm.should_quick_allreduce)
    if ca is None or qr is None:
        return None
    level, qr_first = None, False
    if qr[1] > 0:
        level = _quick_reduce_level(qr_comm)
        if level not in QR_ARMS:
            return None
        both = [
            n
            for n in (qr[0] + 16, qr[1], ca[0] + 16, ca[1])
            if ca[0] < n <= ca[1] and qr[0] < n <= qr[1]
        ]
        qr_first = bool(both) and resolve(_meta(both[0])) == "qr"
    profile = StockProfile(
        ca_min=ca[0],
        ca_max=ca[1],
        qr_level=level,
        qr_min=qr[0] + 16 if qr[1] > 0 else 0,
        qr_max=qr[1],
        qr_first=qr_first,
    )
    # the profile must reproduce the chain (an unexpected backend in between,
    # e.g. pcie_ipc or pymscclpp taking some sizes, turns pareto-comm off)
    sizes = {16 << k for k in range(_PROBE_MAX_LOG2_UNITS + 1)}
    for e in (ca[0], ca[1], qr[0], qr[1]):
        sizes |= {e - 16, e, e + 16, e + 32}
    for n in sorted(x for x in sizes if x > 0):
        got = resolve(_meta(n))
        want = {"ca": "ca", "rccl": None}.get(profile.arm(n), "qr")
        if got != want:
            return None
    return profile


def fused_hidden_sizes():
    """The row widths of the fused all-reduce + residual add + RMSNorm the
    communicator serves: the model's hidden size on the aiter route (the only
    route that calls the fused entry point), else none. The model config is
    built before the parallel groups, so the fused instances share the one
    IPC allocation made before the KV cache memory is measured."""
    if not get_bool_env_var("SGLANG_USE_AITER"):
        return ()
    from sglang.srt.runtime_context import process_model_config

    return (int(process_model_config().hidden_size),)


def create_pareto_comm(group, hidden_sizes=None):
    """Build the group's ParetoCommunicator (collective over the group), or
    return None. Every rank of the group reaches the same decision: the inputs
    are the group's shared configuration. hidden_sizes: the fused calls' row
    widths (None: fused_hidden_sizes())."""
    from sglang.srt.distributed.device_communicators.pareto_comm import (
        ParetoCommunicator,
        pareto_comm_platform,
    )

    arch, _, why = pareto_comm_platform()
    if arch is None:
        logger.info("pareto-comm off for group %s: %s", group.unique_name, why)
        return None
    profile = stock_profile_of(group)
    if profile is None:
        logger.warning(
            "pareto-comm off for group %s: the custom all-reduce / QuickReduce "
            "configuration is not understood",
            group.unique_name,
        )
        return None
    comm = ParetoCommunicator(
        group.cpu_group,
        group.device,
        stock_profile=profile,
        hidden_sizes=fused_hidden_sizes() if hidden_sizes is None else hidden_sizes,
    )
    if comm.disabled:
        comm.close()
        return None
    logger.info(
        "pareto-comm on for group %s rank %d/%d: stock profile %s; "
        "IPC %.1f MiB per rank",
        group.unique_name,
        group.rank_in_group,
        group.world_size,
        profile,
        comm.bytes_per_rank / (1 << 20),
    )
    return comm
