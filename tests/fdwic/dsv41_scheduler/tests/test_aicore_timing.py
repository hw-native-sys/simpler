# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Actual kernel endpoints are mandatory and rank maxima precede averaging."""

import json
import runpy
from pathlib import Path

import pytest

pytestmark = pytest.mark.manual
TIMING = runpy.run_path(str(Path(__file__).resolve().parents[1] / "production/timing.py"))


def log(rank):
    lines = []
    for invocation, duration in enumerate(([1000, 1000, 9000], [1000, 9000, 1000])[rank], 1):
        identity = f"[STRACE] pid={100 + rank} inv={invocation} clk=dev ts=0"
        wall = invocation * 20000
        core = 10**12 + invocation * 20000
        lines += [
            f"{identity} name=chip.run.runner_run.device_wall dur=10000 dev_id={rank} "
            f"dev_start_cycle={wall} dev_end_cycle={wall + 10000} dev_cnt_hz=1000000000",
            f"{identity} name=chip.run.runner_run.aicore_wall dur={duration} clock_source=aicore "
            f"aicore_start_cycle={core} aicore_end_cycle={core + duration} aicore_cnt_hz=1000000000",
        ]
    return "\n".join(lines)


def test_native_uses_kernel_endpoints_and_per_round_rank_maximum():
    performance = {"warmup": 1, "rounds": 2, "per_rank_device_us": {"100": [10, 10], "101": [10, 10]}}
    result = TIMING["native_performance"](log(0) + "\n" + log(1), performance, [0, 1])
    assert result["aicore_wall_us"] == [9, 9]
    assert result["aicore_us_mean"] == 9
    assert result["per_rank_aicore_us"] == {"100": [1, 9], "101": [9, 1]}


@pytest.mark.parametrize("fault", ["absent", "partial", "duplicate"])
def test_old_loop_or_missing_endpoints_never_fall_back_to_runwall(fault):
    text = log(0)
    if fault == "absent":
        text = "\n".join(line for line in text.splitlines() if "aicore_wall" not in line)
    elif fault == "partial":
        text = text.replace("aicore_end_cycle=", "old_end_cycle=")
    else:
        text += "\n" + text.splitlines()[-1]
    with pytest.raises(ValueError):
        TIMING["paired_spans"](text)


def test_replay_rejects_profiled_samples_and_missing_rank(tmp_path):
    for rank in range(2):
        directory = tmp_path / f"rank{rank}"
        (directory / "host").mkdir(parents=True)
        (directory / "host/host.test.log").write_text(log(rank))
        (directory / "console.log").write_text(
            "\n".join(f"run={index} scheduler_mode=1 profiling=0 warmup={int(index == 0)}" for index in range(3))
        )
        report = dict(
            passed=True,
            validated_against_fused_prefix=True,
            rank=rank,
            ranks=2,
            device=rank,
            runtime="host_build_graph",
            warmup=1,
            rounds=2,
            snapshot_inputs=True,
        )
        (directory / "result.json").write_text(json.dumps(report))
    assert TIMING["summarize_replay"](tmp_path)["samples_ms"] == [0.009, 0.009]
    console = tmp_path / "rank0/console.log"
    console.write_text(console.read_text().replace("profiling=0", "profiling=3"))
    with pytest.raises(ValueError, match="must be L0"):
        TIMING["summarize_replay"](tmp_path)
    (tmp_path / "rank1/result.json").unlink()
    with pytest.raises(ValueError, match="Incomplete rank"):
        TIMING["summarize_replay"](tmp_path)


def test_native_runner_timing_needs_no_replay_prefix(tmp_path):
    for rank in range(2):
        directory = tmp_path / f"rank{rank}"
        (directory / "host").mkdir(parents=True)
        (directory / "host/host.test.log").write_text(log(rank))
        (directory / "console.log").write_text(
            "\n".join(f"run={i} scheduler_mode=5 profiling=0 warmup={int(i == 0)}" for i in range(3))
        )
        (directory / "result.json").write_text(
            json.dumps(
                dict(
                    passed=True,
                    execution="native_moe_host_control",
                    numerical_validation=False,
                    validated_against_fused_prefix=False,
                    rank=rank,
                    ranks=2,
                    device=rank,
                    runtime="percounter",
                    warmup=1,
                    rounds=2,
                    snapshot_inputs=False,
                )
            )
        )
    assert TIMING["summarize_replay"](tmp_path)["samples_ms"] == [0.009, 0.009]
