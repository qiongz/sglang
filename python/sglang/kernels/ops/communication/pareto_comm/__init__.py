# SPDX-License-Identifier: Apache-2.0
"""pareto-comm kernels: push all-reduce for AMD Instinct gfx950 (Gluon).

Every rank pushes into its peers' memory and polls only its own; the result is
the fp32 rank-order sum rounded once to bf16, bitwise identical on every rank.
Kernels: one_shot, two_shot, two_shot_epoch (c12 / xe8 wires),
two_shot_window. The communicator lives in
sglang.srt.distributed.device_communicators.pareto_comm."""
