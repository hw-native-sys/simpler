# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Percounter nested execution intervals retain exact accounting and runtime identity."""

import json

import pytest
from _capture_builder import Capture

from simpler_setup.tools.swimlane_converter import generate_chrome_trace_json, read_perf_data


@pytest.mark.parametrize(("level", "memory_event"), [(3, None), (4, "bypass_load"), (4, "ssbuf_store"), (4, "dsb_all")])
def test_nested_kernel_in_step_exports_without_double_counting(tmp_path, level, memory_event):
    capture = Capture(runtime="percounter", level=level, scheduler_producer="aicore")
    capture.task(task_id=7, core_id=0, start=40, end=60, run_epoch=0)
    phases = [
        ("scalar_window", 10, 100, 0),
        ("scalar_percounter_step", 20, 80, 1),
        ("scalar_percounter_run_task", 30, 70, 2),
    ]
    if memory_event:
        phases.append((f"scalar_{memory_event}", 32, 35, 3))
    for kind, begin, end, depth in phases:
        metrics = {"scalar_depth": depth, "scalar_calls": 1}
        if depth == 0:
            metrics.update(scalar_schema=2, function_min_cycles=1 if level == 3 else 0)
        capture.aicore_sched_phase(phase=kind, start=begin, end=end, metrics=metrics)
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(capture.build()))
    data = read_perf_data(raw)
    profile = data["scalar_profiles"][0]
    assert profile["complete"]
    assert profile["kernel_cycles"] == 20
    assert profile["exclusive_cycles"]["scalar_percounter_run_task"] == (17 if memory_event else 20)
    assert profile["exclusive_cycles"]["scalar_percounter_step"] == 20
    assert sum(profile["exclusive_cycles"].values()) == 90
    trace = tmp_path / "trace.json"
    generate_chrome_trace_json(
        data["tasks"],
        str(trace),
        runtime_name="percounter",
        scheduler_phases=data["scheduler_records"],
        scheduler_streams=data["scheduler_streams"],
        scalar_profiles=data["scalar_profiles"],
        scheduler_task_producer="aicore",
        chip_swimlane_level=level,
    )
    output = json.loads(trace.read_text())
    assert output["metadata"]["runtime"] == "percounter"
    assert output["metadata"]["scalar_function_capture"]["memory_operations"] == (
        "per_call" if level == 4 else "disabled"
    )
    names = {e["name"] for e in output["traceEvents"] if e["ph"] == "X" and e["pid"] == 2}
    assert {"percounter_step", "percounter_run_task"} <= names
    if memory_event:
        assert memory_event in names
    assert not any(name.startswith("scheduler_") for name in names)


@pytest.mark.parametrize("level", [2, 3, 4])
def test_scheduler_view_keeps_block_identity_and_deferred_publisher(tmp_path, level):
    capture = Capture(runtime="percounter", level=level, scheduler_producer="aicore")
    capture.task(task_id=7, core_id=0, start=40, end=61, run_epoch=0)
    raw = capture.build()
    raw["percounter_task_events"] = {
        "dropped": 0,
        "unmatched_deferred_updates": 0,
        "records": [
            {
                "task_id": 7,
                "block_index": block,
                "block_count": 2,
                "kind": kind,
                "cycles": cycles,
                "run_epoch": 0,
                "worker_id": worker,
                "cluster_index": cluster,
                "cluster_lane": lane,
                **extra,
            }
            for block, kind, cycles, worker, cluster, lane, extra in [
                (0, "claim", 10, 0, 0, 0, {"claim_start_cycles": 6}),
                (0, "ready", 30, 0, 0, 0, {}),
                (0, "ready", 35, 2, 0, 1, {}),
                (0, "kernel_end", 55, 0, 0, 0, {"kernel_start_cycles": 40}),
                (0, "kernel_end", 60, 2, 0, 1, {"kernel_start_cycles": 41}),
                (0, "defer_registered", 61, 0, 0, 0, {"deferred_group": 4}),
                (0, "completion_issue", 62, 0, 0, 0, {"contribution": -1}),
                (0, "completion_issue", 65, 0, 0, 0, {"contribution": 2}),
                (0, "deferred_completion_issue", 90, 1, 1, 0, {"contribution": 1, "deferred_group": 4}),
                (1, "claim", 15, 0, 0, 0, {"claim_start_cycles": 12}),
                (1, "ready", 32, 0, 0, 0, {}),
                (1, "completion_issue", 70, 3, 0, 2, {"contribution": 1}),
            ]
        ],
    }
    path = tmp_path / "raw.json"
    path.write_text(json.dumps(raw))
    data = read_perf_data(path)
    trace = generate_chrome_trace_json(
        data["tasks"],
        None,
        runtime_name="percounter",
        scheduler_task_producer="aicore",
        chip_swimlane_level=level,
        percounter_task_events=data["percounter_task_events"],
    )
    view = [e for e in trace["traceEvents"] if e["pid"] == 3]
    cycle_us = 1e6 / raw["metadata"]["clock_freq_hz"]
    assert any(e["name"] == "process_name" and e["args"]["name"] == "Scheduler View" for e in view)
    block_0 = [e for e in view if e["ph"] == "i" and e["args"]["block_index"] == 0]
    assert [e["args"]["kind"] for e in block_0] == [
        "claim_start",
        "claim",
        "ready",
        "ready",
        "kernel_end",
        "kernel_end",
        "defer_registered",
        "completion_issue",
        "completion_issue",
        "deferred_completion_issue",
    ]
    assert {e["tid"] for e in block_0} == {block_0[0]["tid"]}
    assert block_0[0]["ts"] == pytest.approx(0)
    assert block_0[1]["ts"] == pytest.approx(4 * cycle_us)
    acquisition = next(e for e in view if e["ph"] == "X" and e["name"].startswith("Claim start → claim"))
    assert acquisition["tid"] == block_0[0]["tid"]
    assert acquisition["dur"] == pytest.approx(4 * cycle_us)
    assert block_0[3]["name"].startswith("Ready (V0)")
    assert block_0[5]["name"].startswith("Kernel end (V0)")
    assert block_0[7]["name"].startswith("Completion issued -1")
    assert block_0[-1]["args"]["worker_id"] == 1
    assert block_0[-1]["args"]["cluster_index"] == 1
    assert block_0[-1]["ts"] == pytest.approx((90 - 6) * cycle_us)
    block_1 = [e for e in view if e["ph"] == "i" and e["args"]["block_index"] == 1]
    assert [e["args"]["kind"] for e in block_1] == ["claim_start", "claim", "ready", "completion_issue"]
    assert block_1[0]["tid"] != block_0[0]["tid"]
