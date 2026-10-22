"""Runtime wiring of pareto-comm (CPU): dispatch order, the stock profile
probed from the group's own dispatch chain, refused configurations and
teardown order. The communicator itself is covered by
test_pareto_comm_capability.py and the multi-GPU kernel tests."""

from __future__ import annotations

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")
setup = pytest.importorskip(
    "sglang.srt.distributed.device_communicators.pareto_comm_setup"
)
table = pytest.importorskip("sglang.kernels.ops.communication.pareto_comm.table")

MiB = 1 << 20


def _nbytes(x):
    return x.numel() * x.element_size()


def _ca(lo, hi):
    ca = Mock(disabled=False)
    ca.should_custom_ar.side_effect = lambda x: (
        _nbytes(x) % 16 == 0 and lo < _nbytes(x) <= hi
    )
    return ca


def _qr(lo, hi, level="INT8"):
    return SimpleNamespace(
        disabled=False,
        qr_quant_level=SimpleNamespace(name=level),
        should_quick_allreduce=lambda x: (
            _nbytes(x) % 16 == 0 and lo <= _nbytes(x) <= hi
        ),
    )


def _coordinator(ca=None, qr=None, pareto=None):
    c = parallel_state.GroupCoordinator.__new__(parallel_state.GroupCoordinator)
    c.world_size = 8
    c.unique_name = "tp:0"
    c.hpu_communicator = None
    c.xpu_communicator = None
    c.npu_communicator = None
    c.pymscclpp_comm = None
    c.torch_symm_mem_comm = None
    c.pcie_ipc_comm = None
    c._fi_workspace_hint = None
    c.ca_comm = ca
    c.qr_comm = qr
    c.pynccl_comm = Mock()
    c.pynccl_comm.change_state.return_value = nullcontext()
    c.is_symmetric_memory_enabled = Mock(return_value=True)
    c.debug_check_symmetric_mempool = Mock()
    if pareto is not None:
        c.pareto_comm = pareto
    return c


def _pareto(accepts):
    pareto = Mock()
    pareto.should_push_ar.return_value = accepts
    return pareto


def _dispatch(c, x):
    with (
        patch.object(parallel_state.torch.compiler, "is_compiling", return_value=False),
        patch.object(
            parallel_state, "outplace_all_reduce", return_value="out"
        ) as outplace,
    ):
        assert c.all_reduce(x) == "out"
    return outplace.call_args.kwargs["outplace_all_reduce_method"]


def _bf16(nbytes):
    return torch.empty(nbytes // 2, dtype=torch.bfloat16, device="meta")


@pytest.mark.parametrize("accepts", [False, True])
def test_pareto_comm_goes_before_the_custom_all_reduce(accepts):
    c = _coordinator(ca=_ca(0, 64 * MiB), pareto=_pareto(accepts))
    method = _dispatch(c, _bf16(1 * MiB))
    assert method == ("pareto_comm" if accepts else "ca")
    # a refused call never reaches the symmetric-memory pynccl short cut
    # before the custom all-reduce, and an accepted one never does either
    c.pynccl_comm.all_reduce.assert_not_called()


def test_no_pareto_comm_keeps_the_chain():
    c = _coordinator(ca=_ca(0, 64 * MiB), qr=_qr(4 * MiB, 2048 * MiB))
    assert c._resolve_outplace_all_reduce_method(_bf16(1 * MiB)) == "ca"
    assert c._resolve_outplace_all_reduce_method(_bf16(128 * MiB)) == "qr"


def test_piecewise_cuda_graph_keeps_pynccl():
    c = _coordinator(ca=None, pareto=_pareto(True))
    with patch.object(
        parallel_state, "is_in_tc_piecewise_cuda_graph", return_value=True
    ):
        assert c._resolve_outplace_all_reduce_method(_bf16(1 * MiB)) == "pynccl"
    c.pareto_comm.should_push_ar.assert_not_called()


def test_out_place_runs_the_communicator():
    pareto = _pareto(True)
    pareto.all_reduce.return_value = "reduced"
    c = _coordinator(ca=_ca(0, 64 * MiB), pareto=pareto)
    assert c._all_reduce_out_place("x", "pareto_comm") == "reduced"
    pareto.all_reduce.assert_called_once_with("x")


def _chain(first, second):
    def resolve(x):
        for name, accepts in (first, second):
            if accepts(x):
                return name
        return None

    return resolve


def test_stock_profile_custom_all_reduce_first():
    """The order before QuickReduce-first dispatch: the custom all-reduce, then QuickReduce."""
    ca, qr = _ca(0, 64 * MiB), _qr(4 * MiB, 2048 * MiB)
    resolve = _chain(("ca", ca.should_custom_ar), ("qr", qr.should_quick_allreduce))
    group = SimpleNamespace(
        _resolve_outplace_all_reduce_method=resolve, ca_comm=ca, qr_comm=qr
    )
    p = setup.stock_profile_of(group)
    assert p == table.StockProfile(
        ca_min=0,
        ca_max=64 * MiB,
        qr_level="INT8",
        qr_min=4 * MiB,
        qr_max=2048 * MiB,
        qr_first=False,
    )


def test_stock_profile_matches_the_dispatch_chain():
    """Whatever the order of the chain, the profile gives every size the path
    the group's own dispatch gives it."""
    c = _coordinator(ca=_ca(0, 64 * MiB), qr=_qr(4 * MiB, 2048 * MiB))
    p = setup.stock_profile_of(c)
    sizes = [16 << k for k in range(28)]
    sizes += [s + d for s in (4 * MiB, 64 * MiB, 2048 * MiB) for d in (-16, 16)]
    for n in sizes:
        expected = {"ca": "ca", "qr8": "qr", "rccl": None}[p.arm(n)]
        assert c._resolve_outplace_all_reduce_method(_bf16(n)) == expected, n


def test_stock_profile_quick_reduce_first():
    """The QuickReduce-first order (QuickReduce before the custom all-reduce)."""
    ca, qr = _ca(0, 64 * MiB), _qr(4 * MiB, 2048 * MiB)
    resolve = _chain(("qr", qr.should_quick_allreduce), ("ca", ca.should_custom_ar))
    group = SimpleNamespace(
        _resolve_outplace_all_reduce_method=resolve, ca_comm=ca, qr_comm=qr
    )
    p = setup.stock_profile_of(group)
    assert (p.ca_min, p.ca_max, p.qr_min, p.qr_max, p.qr_first) == (
        0,
        64 * MiB,
        4 * MiB,
        2048 * MiB,
        True,
    )
    for n in (16, 4 * MiB - 16, 4 * MiB, 64 * MiB, 2048 * MiB + 16):
        expected = {"ca": "ca", "qr8": "qr", "rccl": None}[p.arm(n)]
        assert resolve(_bf16(n)) == expected, n


def test_stock_profile_reads_a_wrapped_quick_reduce():
    inner = _qr(4 * MiB, 2048 * MiB, level="INT4")
    wrapper = SimpleNamespace(
        disabled=False,
        _communicator=inner,
        should_quick_allreduce=inner.should_quick_allreduce,
    )
    c = _coordinator(ca=_ca(0, 64 * MiB), qr=wrapper)
    assert setup.stock_profile_of(c).qr_level == "INT4"


def test_a_backend_in_between_turns_pareto_comm_off():
    """A size the chain gives to neither (here: pymscclpp-like) cannot be
    expressed by the profile."""
    ca, qr = _ca(0, 64 * MiB), _qr(4 * MiB, 2048 * MiB)

    def resolve(x):
        if _nbytes(x) == 1 * MiB:
            return "pymscclpp"
        return _chain(("ca", ca.should_custom_ar), ("qr", qr.should_quick_allreduce))(x)

    group = SimpleNamespace(
        _resolve_outplace_all_reduce_method=resolve, ca_comm=ca, qr_comm=qr
    )
    assert setup.stock_profile_of(group) is None


def test_unknown_quick_reduce_level_turns_pareto_comm_off():
    qr = _qr(4 * MiB, 2048 * MiB)
    qr.qr_quant_level = None
    c = _coordinator(ca=_ca(0, 64 * MiB), qr=qr)
    assert setup.stock_profile_of(c) is None


def test_no_custom_all_reduce():
    c = _coordinator(ca=None, qr=None)
    assert setup.stock_profile_of(c) == table.StockProfile(ca_max=0)


def _context(**over):
    cfg = dict(
        nnodes=1,
        attn_dp_size=1,
        disable_custom_all_reduce=False,
        enable_deterministic_inference=False,
        enable_torch_compile=False,
        enable_two_batch_overlap=False,
        enable_single_batch_overlap=False,
        enable_pdmux=False,
        elastic_ep_backend=None,
    )
    cfg.update(over)
    ns = SimpleNamespace
    parallel = ns(nnodes=cfg["nnodes"], attn_dp_size=cfg["attn_dp_size"])
    exec_ = ns(
        comm=ns(disable_custom_all_reduce=cfg["disable_custom_all_reduce"]),
        deterministic=ns(
            enable_deterministic_inference=cfg["enable_deterministic_inference"]
        ),
        graph=ns(enable_torch_compile=cfg["enable_torch_compile"]),
        overlap=ns(
            enable_two_batch_overlap=cfg["enable_two_batch_overlap"],
            enable_single_batch_overlap=cfg["enable_single_batch_overlap"],
        ),
        moe=ns(elastic_ep_backend=cfg["elastic_ep_backend"]),
    )
    disagg = ns(enable_pdmux=cfg["enable_pdmux"])
    rc = "sglang.srt.runtime_context"
    return (
        patch(f"{rc}.get_parallel", return_value=parallel),
        patch(f"{rc}.get_exec", return_value=exec_),
        patch(f"{rc}.get_disagg", return_value=disagg),
    )


@pytest.mark.parametrize(
    "over, reason",
    [
        ({}, None),
        ({"nnodes": 2}, "multi-node"),
        ({"attn_dp_size": 2}, "DP attention"),
        ({"disable_custom_all_reduce": True}, "--disable-custom-all-reduce"),
        ({"enable_deterministic_inference": True}, "deterministic inference"),
        ({"enable_torch_compile": True}, "torch.compile"),
        ({"enable_two_batch_overlap": True}, "two-batch / single-batch overlap"),
        ({"enable_pdmux": True}, "PD multiplexing"),
        ({"elastic_ep_backend": "mooncake"}, "elastic EP"),
    ],
)
def test_blockers(over, reason):
    a, b, c = _context(**over)
    with (
        a,
        b,
        c,
        patch.object(setup, "is_hip", return_value=True),
        patch.object(setup, "is_gfx95_supported", return_value=True),
    ):
        got = setup.pareto_comm_blockers()
    assert got == ([] if reason is None else [reason])


@pytest.mark.parametrize("hip, gfx95", [(False, False), (True, False)])
def test_off_and_not_imported_outside_rocm_gfx950(hip, gfx95):
    """CUDA (or another ROCm arch) with the flag on: refused before anything of
    pareto-comm is imported; the dispatch chain is the stock one."""
    import sys

    mod = "sglang.srt.distributed.device_communicators.pareto_comm"
    sys.modules.pop(mod, None)
    before = set(sys.modules)
    a, b, c = _context()
    with (
        a,
        b,
        c,
        patch.object(setup, "is_hip", return_value=hip),
        patch.object(setup, "is_gfx95_supported", return_value=gfx95),
    ):
        assert setup.pareto_comm_blockers() == ["not ROCm gfx950"]
    assert not [m for m in set(sys.modules) - before if "pareto_comm" in m]
    c = _coordinator(ca=_ca(0, 64 * MiB))
    assert c.pareto_comm is None
    assert c._resolve_outplace_all_reduce_method(_bf16(1 * MiB)) == "ca"


def test_released_before_the_process_groups():
    order = []
    c = parallel_state.GroupCoordinator.__new__(parallel_state.GroupCoordinator)
    c.device_group = MagicMock()
    c.cpu_group = MagicMock()
    c.pcie_ipc_comm = None
    c.pynccl_comm = None
    c.pymscclpp_comm = None
    c.ca_comm = None
    c.mq_broadcaster = None
    c.pareto_comm = Mock()
    c.pareto_comm.close.side_effect = lambda: order.append("pareto")
    with patch.object(
        torch.distributed,
        "destroy_process_group",
        side_effect=lambda g: order.append("group"),
    ):
        c.destroy()
    assert order == ["pareto", "group", "group"]
    assert c.pareto_comm is None
