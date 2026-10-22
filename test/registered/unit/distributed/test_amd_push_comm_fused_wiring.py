"""Runtime wiring of amd-push-comm's fused all-reduce + residual add + RMSNorm
(CPU): the GroupCoordinator entry point, the layer-boundary gate that is
independent of aiter's fusion switches, the norm's fallback when the fused
call is declined, and the row widths the communicator is built for."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import torch

from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=5, suite="base-a-test-cpu")

parallel_state = pytest.importorskip("sglang.srt.distributed.parallel_state")
setup = pytest.importorskip(
    "sglang.srt.distributed.device_communicators.amd_push_comm_setup"
)
add_norm = pytest.importorskip("sglang.srt.layers.layer_boundary.residual.add_norm")
layernorm = pytest.importorskip("sglang.srt.layers.layernorm")


def _coordinator(push=None, ca=None):
    c = parallel_state.GroupCoordinator.__new__(parallel_state.GroupCoordinator)
    c.amd_push_comm = push
    c.ca_comm = ca
    return c


def _args(m=4, h=6144):
    x = torch.zeros(m, h, dtype=torch.bfloat16)
    return x, torch.zeros_like(x), torch.ones(h, dtype=torch.bfloat16), 1e-6


# --------------------------------------------- GroupCoordinator entry point
def test_fused_entry_tries_push_comm_first():
    push = Mock()
    push.fused_allreduce_rmsnorm.return_value = ("out", "residual_out")
    ca = Mock(disabled=False)
    c = _coordinator(push, ca)
    assert c.fused_allreduce_rmsnorm(*_args()) == ("out", "residual_out")
    push.fused_allreduce_rmsnorm.assert_called_once()
    ca.custom_fused_ar_rms.assert_not_called()


def test_declined_fused_call_keeps_the_original_path():
    push = Mock()
    push.fused_allreduce_rmsnorm.return_value = None
    c = _coordinator(push, ca=None)  # no custom all-reduce: the original None
    assert c.fused_allreduce_rmsnorm(*_args()) is None
    push.fused_allreduce_rmsnorm.assert_called_once()


def test_piecewise_cuda_graph_skips_push_comm():
    push = Mock()
    c = _coordinator(push, ca=None)
    with patch.object(
        parallel_state, "is_in_tc_piecewise_cuda_graph", return_value=True
    ):
        assert c.fused_allreduce_rmsnorm(*_args()) is None
    push.fused_allreduce_rmsnorm.assert_not_called()


def test_no_push_comm_keeps_the_entry_point():
    c = _coordinator(None, ca=None)
    assert c.fused_allreduce_rmsnorm(*_args()) is None


# ----------------------------------------------------- layer-boundary gate
def _gate(
    serves=True,
    aiter=True,
    dp_attention=False,
    tp=8,
    attn_tp=8,
    moe_ep=1,
    moe_tp=8,
    comm=True,
):
    push = Mock()
    push.should_fuse.return_value = serves
    group = SimpleNamespace(amd_push_comm=push if comm else None)
    parallel = SimpleNamespace(
        tp_size=tp, attn_tp_size=attn_tp, moe_ep_size=moe_ep, moe_tp_size=moe_tp
    )
    with (
        patch.object(add_norm, "_use_aiter", aiter),
        patch.object(add_norm, "is_dp_attention_enabled", return_value=dp_attention),
        patch.object(add_norm, "get_parallel", return_value=parallel),
        patch.object(parallel_state, "get_tp_group", return_value=group),
    ):
        return add_norm.amd_push_comm_fusion_applies(torch.zeros(4, 6144))


@pytest.mark.parametrize(
    "over, want",
    [
        ({}, True),
        ({"moe_ep": 8, "moe_tp": 1}, True),  # EP over the TP group's ranks
        ({"serves": False}, False),
        ({"comm": False}, False),
        ({"aiter": False}, False),
        ({"dp_attention": True}, False),
        ({"attn_tp": 4}, False),
        ({"moe_ep": 4, "moe_tp": 1}, False),
    ],
)
def test_fusion_gate_is_independent_of_aiter_switches(over, want):
    assert _gate(**over) is want


# ------------------------------------------------------- the norm's fallback
class _Norm:
    variance_epsilon = 1e-6

    def __init__(self):
        self.calls = []

    def forward(self, x, residual, post):
        self.calls.append((x.clone(), residual.clone(), post))
        return x + residual, x + residual


def _forward_with_fusion(fused_result, aiter_fusion, push, post=None):
    import sglang.srt.distributed as D

    x = torch.ones(4, 8)
    res = torch.full((4, 8), 2.0)
    norm = _Norm()
    exec_ = SimpleNamespace(
        comm=SimpleNamespace(enable_aiter_allreduce_fusion=aiter_fusion)
    )
    parallel = SimpleNamespace(attn_tp_size=8, moe_ep_size=1, moe_tp_size=8)
    with (
        patch.object(layernorm, "_use_aiter", True),
        patch.object(layernorm, "get_exec", return_value=exec_),
        patch.object(layernorm, "get_parallel", return_value=parallel),
        patch.object(
            D, "get_tp_group", return_value=SimpleNamespace(amd_push_comm=push)
        ),
        patch.object(
            D,
            "tensor_model_parallel_fused_allreduce_rmsnorm",
            return_value=fused_result,
        ),
        patch.object(
            D, "tensor_model_parallel_all_reduce", side_effect=lambda t: t * 8
        ),
    ):
        out = layernorm._forward_with_allreduce_fusion(
            norm, x, res, post, torch.ones(8), True
        )
    return out, norm.calls


def test_declined_push_fusion_completes_the_sum():
    """aiter's fusion off (GLM-5.2 prefill, or decode after aiter's fusion is
    turned off for it), amd-push-comm declines: the result is AR + add + norm,
    not the norm of the partial sum."""
    out, calls = _forward_with_fusion(None, aiter_fusion=False, push=Mock())
    assert len(calls) == 1
    x, res, post = calls[0]
    assert torch.equal(x, torch.full((4, 8), 8.0)) and torch.equal(
        res, torch.full((4, 8), 2.0)
    )
    assert post is None
    # the residual addition is applied once, before the norm
    out, calls = _forward_with_fusion(None, False, Mock(), post=torch.ones(4, 8))
    assert torch.equal(calls[0][1], torch.full((4, 8), 3.0)) and calls[0][2] is None


def test_taken_push_fusion_returns_its_result():
    out, calls = _forward_with_fusion(("o", "r"), aiter_fusion=False, push=Mock())
    assert out == ("o", "r") and calls == []


def test_without_push_comm_the_fallback_is_unchanged():
    # aiter's fusion off and no amd-push-comm: as before (the norm alone; the
    # gates never route a partial sum here in that configuration)
    out, calls = _forward_with_fusion(None, aiter_fusion=False, push=None)
    assert torch.equal(calls[0][0], torch.ones(4, 8))


# ------------------------------------------------- widths and construction
def test_fused_hidden_sizes():
    with patch.object(setup, "get_bool_env_var", return_value=False):
        assert setup.fused_hidden_sizes() == ()
    cfg = SimpleNamespace(hidden_size=6144)
    with (
        patch.object(setup, "get_bool_env_var", return_value=True),
        patch("sglang.srt.runtime_context.process_model_config", return_value=cfg),
    ):
        assert setup.fused_hidden_sizes() == (6144,)


def test_communicator_built_with_the_fused_widths():
    import sglang.srt.distributed.device_communicators.amd_push_comm as R

    made = {}

    class Comm:
        disabled = False
        bytes_per_rank = 0

        def __init__(self, *a, **kw):
            made.update(kw)

    group = SimpleNamespace(
        unique_name="tp:0", cpu_group=None, device="cpu", rank_in_group=0, world_size=8
    )
    with (
        patch.object(R, "amd_push_comm_platform", return_value=("gfx950", "t", "")),
        patch.object(R, "AmdPushCommunicator", Comm),
        patch.object(setup, "stock_profile_of", return_value="profile"),
    ):
        assert setup.create_amd_push_comm(group, hidden_sizes=(6144,)) is not None
    assert made["hidden_sizes"] == (6144,) and made["stock_profile"] == "profile"
