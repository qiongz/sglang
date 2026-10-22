# SPDX-License-Identifier: Apache-2.0
"""Host side of the amd-push-comm protocol: failure codes, the wall-clock spin
bound and the failure-reporting constexprs (pure Python, no triton or torch).
Receive slots start at the bf16 -0.0 sentinel (0x8000) and senders never send
-0.0, so the data itself signals arrival; two_shot_epoch uses epoch flags."""

from __future__ import annotations

import msgspec

# counter / error values of a failed spin: timed out, aborted by the host, or
# a call-tag mismatch; values >= FAILED mark a failed call
LIMIT = (1 << 31) - 1
ABORTED = LIMIT - 1
SEQ_MISMATCH = LIMIT - 2
FAILED = SEQ_MISMATCH
DIAG_MAGIC = 0x50415231  # first word of a complete diagnostics record
DIAG_SLOTS = 256  # records of 16 int32 in the diagnostics ring
TCLOB = ",~{s40},~{s41},~{s42},~{s43}"  # SGPRs of the timed tail
SENT32 = -2147450880  # two bf16 sentinels in one int32 word (0x80008000)


class SpinBound(msgspec.Struct, frozen=True):
    """Wall-clock bound of every spin. trap: a timed-out wave executes
    s_trap 2; backoff: s_sleep 1 per failed poll after this many polls."""

    timeout_s: float = 300.0
    trap: bool = True
    backoff: int = 0

    def __post_init__(self):
        if not 0 < self.ticks < (1 << 63):
            raise ValueError(f"spin timeout {self.timeout_s} s out of range")
        if not 0 <= int(self.backoff) < (1 << 31):
            raise ValueError(f"spin backoff {self.backoff} out of range")

    @property
    def ticks(self):
        return int(float(self.timeout_s) * 1e8)  # s_memrealtime: 100 MHz

    def key(self):
        return self.ticks, bool(self.trap), int(self.backoff)


DEFAULT_SPIN = SpinBound()


def no_trap(bound):
    """With failure reporting a failed spin reports and returns NaN instead."""
    return msgspec.structs.replace(bound, trap=False)


class Diag(msgspec.Struct, frozen=True):
    """Failure reporting of one kernel instance: the host's abort word and
    diagnostics ring (device addresses), the instance id in its records, and
    s_trap after reporting (no host monitor)."""

    abort: int
    base: int
    inst: int = 0
    trap: bool = False

    def kw(self):
        return dict(
            DIAG=1,
            DIAG_BASE=int(self.base),
            INST=int(self.inst),
            TRAP=int(bool(self.trap)),
        )


def timed_tail_lines(ic, ticks, trap, bk=0, abort=None, exit_label="2"):
    """Every 4096 failed polls: check the abort word (-> ABORTED), then the
    deadline armed by the first s_memrealtime (-> LIMIT, s_trap 2 with trap).
    Clobbers s40..s43."""
    lo, hi = ticks & 0xFFFFFFFF, ticks >> 32
    pre = (
        [f"s_cmp_lt_u32 {ic}, {bk}", "s_cbranch_scc1 6f", "s_sleep 1", "6:"]
        if bk
        else []
    )
    gate = [f"s_and_b32 s40, {ic}, 0xfff", "s_cmp_lg_u32 s40, 0", "s_cbranch_scc1 1b"]
    ab = []
    if abort is not None:
        a = int(abort)
        assert 0 < a < (1 << 64) and a % 4 == 0, f"abort word address {a:#x}"
        ab = [
            f"s_mov_b32 s40, {a & 0xFFFFFFFF:#x}",
            f"s_mov_b32 s41, {a >> 32:#x}",
            "s_load_dword s40, s[40:41], 0x0 glc",
            "s_waitcnt lgkmcnt(0)",
            "s_cmp_eq_u32 s40, 0",
            "s_cbranch_scc1 9f",
            f"s_mov_b32 {ic}, {ABORTED}",
            f"s_branch {exit_label}f",
            "9:",
        ]
    clock = [
        "s_memrealtime s[40:41]",
        "s_waitcnt lgkmcnt(0)",
        "s_cmp_eq_u32 s43, 0",
        "s_cbranch_scc0 8f",
        f"s_add_u32 s42, s40, {lo}",
        f"s_addc_u32 s43, s41, {hi}",
        "s_branch 1b",
        "8:",
        "s_sub_u32 s40, s40, s42",
        "s_subb_u32 s41, s41, s43",
        "s_cmp_lt_i32 s41, 0",
        "s_cbranch_scc1 1b",
        f"s_mov_b32 {ic}, {LIMIT}",
    ]
    return pre + gate + ab + clock + (["s_trap 2"] if trap else [])
