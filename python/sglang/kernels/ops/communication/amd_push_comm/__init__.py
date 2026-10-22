# SPDX-License-Identifier: Apache-2.0
"""amd-push-comm: push all-reduce kernels for AMD Instinct gfx950 (Gluon).

Every rank pushes into its peers' memory and polls only its own; the result is
the fp32 rank-order sum rounded once to bf16, bitwise identical on every rank.
Kernels: one_shot, two_shot, two_shot_epoch (c12 / int8 wires),
two_shot_window. The communicator lives in
sglang.srt.distributed.device_communicators.amd_push_comm."""
