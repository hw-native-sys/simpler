# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for simpler_setup.tools.critical_path."""

import json

from simpler_setup.tools import critical_path


def _write_rank_artifacts(rank_dir, name_map_filename):
    rank_dir.mkdir(parents=True)
    (rank_dir / "l2_swimlane_records.json").write_text(
        json.dumps(
            {
                "metadata": {"clock_freq_hz": 1_000_000},
                "aicore_tasks": [[0, 1, 0, 0, 10, 0]],
            }
        )
    )
    (rank_dir / "deps.json").write_text(
        json.dumps(
            {
                "tasks": [{"task_id": 1, "kernel_ids": [7, -1, -1]}],
                "edges": [],
            }
        )
    )
    (rank_dir / name_map_filename).write_text(json.dumps({"callable_id_to_name": {"7": "kernel_add"}}))


def _write_visualization_artifacts(rank_dir):
    rank_dir.mkdir(parents=True)
    (rank_dir / "l2_swimlane_records.json").write_text(
        json.dumps(
            {
                "metadata": {"clock_freq_hz": 1_000_000},
                "aicore_tasks": [
                    [0, 1, 0, 0, 10, 0],
                    [1, 2, 0, 0, 19, 0],
                    [1, 3, 0, 20, 30, 0],
                ],
            }
        )
    )
    (rank_dir / "deps.json").write_text(
        json.dumps(
            {
                "tasks": [
                    {"task_id": 1, "kernel_ids": [1, -1, -1]},
                    {"task_id": 2, "kernel_ids": [2, -1, -1]},
                    {"task_id": 3, "kernel_ids": [3, -1, -1]},
                ],
                "edges": [{"pred": 1, "succ": 3}],
            }
        )
    )
    (rank_dir / "name_map_case.json").write_text(
        json.dumps(
            {
                "callable_id_to_name": {
                    "1": "kernel_a",
                    "2": "kernel_b",
                    "3": "kernel_c",
                }
            }
        )
    )
    events = [
        {"args": {"name": "Worker View"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 4},
        {
            "args": {"name": "AIC_0"},
            "cat": "__metadata",
            "name": "thread_name",
            "ph": "M",
            "pid": 4,
            "tid": 10000,
        },
        {
            "args": {"taskId": 1},
            "cat": "event",
            "id": 10,
            "name": "kernel_a(t1)",
            "ph": "X",
            "pid": 4,
            "tid": 10000,
            "ts": 0,
            "dur": 10,
        },
        {
            "args": {"taskId": 2},
            "cat": "event",
            "id": 11,
            "name": "kernel_b(t2)",
            "ph": "X",
            "pid": 4,
            "tid": 10010,
            "ts": 0,
            "dur": 19,
        },
        {
            "args": {"taskId": 3},
            "cat": "event",
            "id": 12,
            "name": "kernel_c(t3)",
            "ph": "X",
            "pid": 4,
            "tid": 10010,
            "ts": 20,
            "dur": 10,
        },
        {
            "args": {"phase": "dummy_task", "task_id": 99},
            "cat": "event",
            "id": 13,
            "name": "dummy(t99)",
            "ph": "X",
            "pid": 4,
            "tid": 19000,
            "ts": 5,
            "dur": 0.02,
        },
        {"cat": "flow", "id": 100, "name": "dependency", "ph": "s", "pid": 4, "tid": 10000, "ts": 0},
        {"cat": "flow", "id": 100, "name": "dependency", "ph": "f", "pid": 4, "tid": 10010, "ts": 20},
        {"cat": "flow", "id": 101, "name": "dispatch", "ph": "s", "pid": 3, "tid": 10000, "ts": 0},
        {"cat": "flow", "id": 101, "name": "dispatch", "ph": "f", "pid": 4, "tid": 10000, "ts": 0},
        {"cat": "flow", "id": 102, "name": "complete", "ph": "s", "pid": 4, "tid": 10010, "ts": 30},
        {"cat": "flow", "id": 102, "name": "complete", "ph": "f", "pid": 2, "tid": 0, "ts": 31},
        {"args": {"taskId": 1}, "cat": "event", "name": "kernel_a(t1)", "ph": "X", "pid": 3, "tid": 1},
    ]
    (rank_dir / "merged_swimlane_20260720_120000.json").write_text(
        json.dumps({"traceEvents": events, "displayTimeUnit": "us", "source": "unit-test"})
    )


def test_main_accepts_name_map_variants_and_writes_report_beside_each_records(tmp_path):
    simpler_output = tmp_path / "outputs" / "case"
    pypto_output = tmp_path / "build_output" / "case" / "dfx_outputs"
    _write_rank_artifacts(simpler_output, "name_map_TestCase.json")
    _write_rank_artifacts(pypto_output, "name_map.json")

    rc = critical_path.main([str(tmp_path)])

    assert rc == 0
    for output_dir in (simpler_output, pypto_output):
        report = output_dir / "critical_path_report.md"
        assert report.exists()
        assert "kernel_add" in report.read_text()
    assert not (tmp_path / "critical_path_report.md").exists()


def test_main_writes_static_and_observed_full_traces(tmp_path):
    output_dir = tmp_path / "outputs" / "case"
    _write_visualization_artifacts(output_dir)

    rc = critical_path.main([str(output_dir)])

    assert rc == 0
    static_trace = json.loads((output_dir / "CPM_static.json").read_text())
    observed_trace = json.loads((output_dir / "CPM_observed.json").read_text())
    assert static_trace["displayTimeUnit"] == observed_trace["displayTimeUnit"] == "us"
    assert static_trace["source"] == observed_trace["source"] == "unit-test"

    static_events = static_trace["traceEvents"]
    observed_events = observed_trace["traceEvents"]
    source_events = json.loads((output_dir / "merged_swimlane_20260720_120000.json").read_text())["traceEvents"]
    assert len(static_events) == len(observed_events) == len(source_events)
    assert {event["pid"] for event in static_events + observed_events} == {2, 3, 4}

    def worker_task_names(events):
        return {
            event["args"]["taskId"]: event["name"]
            for event in events
            if event.get("ph") == "X" and event.get("pid") == 4 and "taskId" in event.get("args", {})
        }

    assert worker_task_names(static_events) == {
        1: "kernel_a(t1)",
        2: "·(t2)",
        3: "kernel_c(t3)",
    }
    assert worker_task_names(observed_events) == {
        1: "·(t1)",
        2: "kernel_b(t2)",
        3: "kernel_c(t3)",
    }
    for events in (static_events, observed_events):
        other_view_tasks = [event for event in events if event.get("ph") == "X" and event.get("pid") != 4]
        assert other_view_tasks == [
            {"args": {"taskId": 1}, "cat": "event", "name": "kernel_a(t1)", "ph": "X", "pid": 3, "tid": 1}
        ]
    assert "dummy(t99)" in {event.get("name") for event in static_events}
    assert "dummy(t99)" in {event.get("name") for event in observed_events}
    assert {event["id"] for event in static_events if event.get("cat") == "flow"} == {100, 101, 102}
    assert {event["id"] for event in observed_events if event.get("cat") == "flow"} == {100, 101, 102}


def test_main_reports_stale_merged_trace_and_continues_without_partial_cpm_outputs(tmp_path, capsys):
    stale_output = tmp_path / "outputs" / "a_stale"
    good_output = tmp_path / "outputs" / "b_good"
    _write_visualization_artifacts(stale_output)
    _write_visualization_artifacts(good_output)

    merged_path = stale_output / "merged_swimlane_20260720_120000.json"
    merged_trace = json.loads(merged_path.read_text())
    merged_trace["traceEvents"] = [
        event
        for event in merged_trace["traceEvents"]
        if not (event.get("pid") == 4 and event.get("ph") == "X" and event.get("args", {}).get("taskId") == 2)
    ]
    merged_path.write_text(json.dumps(merged_trace))

    rc = critical_path.main([str(tmp_path)])

    assert rc == 2
    assert (stale_output / "critical_path_report.md").exists()
    assert not (stale_output / "CPM_static.json").exists()
    assert not (stale_output / "CPM_observed.json").exists()
    assert (good_output / "critical_path_report.md").exists()
    assert (good_output / "CPM_static.json").exists()
    assert (good_output / "CPM_observed.json").exists()
    error = capsys.readouterr().err
    assert "error: invalid merged_swimlane*.json" in error
    assert "missing 1 critical task(s): 2" in error
