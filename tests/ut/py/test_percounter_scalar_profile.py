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
def test_calendar_scheduler_tracks_preserve_role_and_defer_actor(tmp_path, level):
    capture = Capture(
        runtime="percounter", level=level, scheduler_producer="aicore", core_types=["aic", "aic", "aiv", "aiv"]
    )
    capture.task(task_id=7, core_id=0, start=40, end=55, run_epoch=0)
    capture.task(task_id=7, core_id=2, start=44, end=62, run_epoch=0)
    raw = capture.build()
    raw["percounter_task_events"] = {
        "dropped": 0,
        "unmatched_deferred_updates": 0,
        "folded_task_ids": [],
        "inline_completed_task_ids": [],
        "range_records": [
            {
                "kind": kind,
                "cycles": cycles,
                "run_epoch": 0,
                "worker_id": 0,
                "cluster_index": 0,
                "cluster_lane": 0,
                "range_index": 2,
                "graph_shell_id": 5,
                **extra,
            }
            for kind, cycles, extra in [
                ("range_enter_begin", 6, {}),
                ("graph_ready_observed", 13, {"watermark": 16}),
                ("range_entered", 15, {}),
            ]
        ],
        "records": [
            {
                "task_id": 7,
                "block_index": 0,
                "block_count": 1,
                "kind": kind,
                "cycles": cycles,
                "run_epoch": 0,
                "worker_id": worker,
                "cluster_index": cluster,
                "cluster_lane": lane,
                **extra,
            }
            for kind, cycles, worker, cluster, lane, extra in [
                ("row_begin", 16, 0, 0, 0, {}),
                ("row_begin", 20, 2, 0, 1, {}),
                ("inputs_ready", 26, 0, 0, 0, {}),
                ("ready", 30, 0, 0, 0, {}),
                ("ready", 35, 2, 0, 1, {}),
                ("kernel_end", 55, 0, 0, 0, dict(dispatch_cycles=32, kernel_start_cycles=40, finish_cycles=58)),
                ("kernel_end", 62, 2, 0, 1, dict(dispatch_cycles=38, kernel_start_cycles=44, finish_cycles=64)),
                ("defer_registered", 61, 0, 0, 0, dict(deferred_group=4)),
                ("local_retired", 64, 0, 0, 0, {}),
                ("completion_issue", 65, 2, 0, 1, dict(contribution=1)),
                ("local_retired", 66, 2, 0, 1, {}),
                ("defer_poll_begin", 70, 1, 1, 0, dict(deferred_group=4, source_worker_id=0)),
                ("defer_signal_ready", 85, 1, 1, 0, dict(deferred_group=4, source_worker_id=0)),
                ("deferred_completion_issue", 90, 1, 1, 0, dict(deferred_group=4, source_worker_id=0, contribution=1)),
            ]
        ],
    }
    raw["aicpu_lifecycle_records"] = [
        {
            "aicpu_thread_id": 1,
            "graph_shell_id": 5,
            "materialize_start_cycles": 10,
            "materialize_end_cycles": 20,
        },
        {
            "aicpu_thread_id": 1,
            "graph_shell_id": 5,
            "graph_watermark_published_cycles": 12,
            "watermark": 16,
        },
    ]
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
        aicpu_lifecycle_records=data["aicpu_lifecycle_records"],
    )
    events = trace["traceEvents"]
    scheduler = [e for e in events if e["pid"] == 3]
    workers = [e for e in events if e["pid"] == 4 and e["ph"] == "X"]
    assert len(workers) == 2
    assert all(e["args"]["block_idx"] == 0 and e["args"]["block_num"] == 1 for e in workers)
    assert workers[0]["dur"] == pytest.approx(15e6 / raw["metadata"]["clock_freq_hz"])
    assert sum(e["ph"] == "X" and "Output fence" in e["name"] for e in scheduler) == 2
    assert all("Kernel execution" not in e["name"] for e in scheduler)
    ready = [e for e in scheduler if e["ph"] == "i" and e["args"]["kind"] == "ready"]
    assert len({e["tid"] for e in ready}) == 2
    register = next(e for e in scheduler if e["ph"] == "i" and e["args"]["kind"] == "defer_registered")
    done = next(e for e in scheduler if e["ph"] == "i" and e["args"]["kind"] == "deferred_completion_issue")
    assert register["tid"] != done["tid"]
    assert done["args"]["source_worker_id"] == register["args"]["worker_id"]
    assert any(e.get("pid") == 6 and e["ph"] == "X" and e["args"]["graph_shell_id"] == 5 for e in events)
    assert any(e.get("pid") == 6 and e["name"] == "Graph watermark published" for e in events)
    published = next(e for e in events if e.get("pid") == 6 and e["name"] == "Graph watermark published")
    observed = next(
        e for e in scheduler if e.get("cat") == "percounter_range" and e["args"]["kind"] == "graph_ready_observed"
    )
    assert observed["ts"] > published["ts"]
    assert observed["args"]["watermark"] == 16
