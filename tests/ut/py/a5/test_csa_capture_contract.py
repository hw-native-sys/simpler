# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------


"""Fail-closed CSA trace checks, including mutations that preserve total row counts."""

import json

import pytest

from examples.a5.host_build_graph.deepseek_v4_pro_csa import run as runner
from simpler_setup.tools import swimlane_converter


def capture(tmp_path, monkeypatch):
    deps = {
        "runtime": "host_build_graph",
        "tasks": [
            {"task_id": "0", "kernel_ids": [1, 2, -1], "block_num": 2},
            {"task_id": "1", "kernel_ids": [-1, -1, -1], "block_num": 1},
            {"task_id": "2", "kernel_ids": [-1, 3, -1], "block_num": 1},
        ],
        "edges": [{"pred": "0", "succ": "1"}, {"pred": "1", "succ": "2"}],
    }
    rows = [
        {"task_id": 0, "core_type": kind, "run_epoch": 1, "start_time_us": 1, "end_time_us": 3}
        for kind in ("aic", "aiv", "aic", "aiv")
    ] + [{"task_id": 2, "core_type": "aiv", "run_epoch": 1, "start_time_us": 4, "end_time_us": 5}]
    raw = {"chip_swimlane_level": 1, "metadata": {"clock_freq_hz": 1000000000}, "aicore_tasks": [[0]] * 5}
    raw_path, deps_path = tmp_path / "raw.json", tmp_path / "deps.json"
    raw_path.write_text(json.dumps(raw))
    deps_path.write_text(json.dumps(deps))
    monkeypatch.setattr(swimlane_converter, "read_perf_data", lambda *args: {"tasks": rows})
    monkeypatch.setattr(runner.workload.fixture, "manifest", lambda: {"expected_physical_tasks": 5})
    return raw_path, deps_path, rows


def test_valid_mixed_capture_with_dummy_join(tmp_path, monkeypatch):
    raw, deps, _ = capture(tmp_path, monkeypatch)
    assert runner.validate_task_capture(raw, deps, 1)[2]["physical_tasks"] == 5


@pytest.mark.parametrize("mutation", ["wrong_lane", "old_epoch", "all_zero_epoch", "dummy_early_consumer"])
def test_same_count_corruption_cannot_pass_capture_validation(tmp_path, monkeypatch, mutation):
    raw, deps, rows = capture(tmp_path, monkeypatch)
    if mutation == "wrong_lane":
        rows[1]["core_type"] = "aic"
    elif mutation == "old_epoch":
        rows[0]["run_epoch"] = 2
    elif mutation == "all_zero_epoch":
        for row in rows:
            row["run_epoch"] = 0
    else:
        rows[-1]["start_time_us"] = 2
    with pytest.raises(ValueError):
        runner.validate_task_capture(raw, deps, 1)
