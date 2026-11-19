# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Contract tests for simpler_setup.tools.critical_path."""

import copy
import json

import pytest

from simpler_setup.tools import critical_path
from simpler_setup.tools._runtime_dispatch import HBG_RUNTIME, TMR_RUNTIME


def _write_rank_artifacts(rank_dir, name_map_filename):
    rank_dir.mkdir(parents=True)
    (rank_dir / "chip_swimlane_records.json").write_text(
        json.dumps(
            {
                "chip_swimlane_level": 4,
                "metadata": {"runtime": TMR_RUNTIME, "clock_freq_hz": 1_000_000},
                "aicore_tasks": [[0, 1, 0, 0, 10, 0, 42]],
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
    (rank_dir / "chip_swimlane_records.json").write_text(
        json.dumps(
            {
                "chip_swimlane_level": 4,
                "metadata": {"runtime": TMR_RUNTIME, "clock_freq_hz": 1_000_000},
                "aicore_tasks": [
                    [0, 1, 0, 0, 10, 0, 42],
                    [1, 2, 0, 0, 19, 0, 42],
                    [1, 3, 0, 20, 30, 0, 42],
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
            "name": "kernel_a(r0t1)",
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
            "name": "kernel_b(r0t2)",
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
            "name": "kernel_c(r0t3)",
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
            "name": "dummy(r0t99)",
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
        {"args": {"taskId": 1}, "cat": "event", "name": "kernel_a(r0t1)", "ph": "X", "pid": 3, "tid": 1},
    ]
    for event in events:
        if "taskId" in event.get("args", {}):
            event["args"]["run_epoch"] = 42
    (rank_dir / "merged_swimlane_20260720_120000.json").write_text(
        json.dumps(
            {
                "traceEvents": events,
                "displayTimeUnit": "us",
                "source": "unit-test",
                # Every trace names the TaskId layout its labels follow; critical_path
                # re-formats ids from this file alone and refuses one that does not.
                "metadata": {"runtime": TMR_RUNTIME},
            }
        )
    )


@pytest.mark.parametrize("row_width", [5, 6, 7])
def test_build_graph_accepts_task_record_widths(tmp_path, row_width):
    output_dir = tmp_path / "outputs" / "case"
    _write_visualization_artifacts(output_dir)
    records_path = output_dir / "chip_swimlane_records.json"
    records = json.loads(records_path.read_text())
    records["aicore_tasks"].append([2, 1, 7, 2, 12, 3, 42])
    records["aicore_tasks"] = [row[:row_width] for row in records["aicore_tasks"]]
    records_path.write_text(json.dumps(records))

    graph = critical_path.build_graph(output_dir, tmp_path, tol=2)
    result = critical_path.analyze_rank(graph, tol=2)

    assert graph.start == {"1": 0, "2": 0, "3": 20}
    assert graph.end == {"1": 12, "2": 19, "3": 30}
    assert graph.dur == {"1": 12, "2": 19, "3": 10}
    assert graph.nblock == {"1": 2, "2": 1, "3": 1}
    assert graph.name == {"1": "kernel_a", "2": "kernel_b", "3": "kernel_c"}
    assert graph.hb_pred == {"3": ["1"]}
    assert graph.core_prev == {"3": ("2", 19)}
    assert result.cpm_path == ["1", "3"]
    assert result.cpm_len == 22
    assert [segment.task for segment in result.segments] == ["2", "3"]
    assert result.compute_total == 29
    assert result.stall_total == 1
    assert result.compute_total + result.stall_total == result.makespan == 30


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
        1: "kernel_a(r0t1)",
        2: "·(r0t2)",
        3: "kernel_c(r0t3)",
    }
    assert worker_task_names(observed_events) == {
        1: "·(r0t1)",
        2: "kernel_b(r0t2)",
        3: "kernel_c(r0t3)",
    }
    for events in (static_events, observed_events):
        other_view_tasks = [event for event in events if event.get("ph") == "X" and event.get("pid") != 4]
        assert other_view_tasks == [
            {
                "args": {"taskId": 1, "run_epoch": 42},
                "cat": "event",
                "name": "kernel_a(r0t1)",
                "ph": "X",
                "pid": 3,
                "tid": 1,
            }
        ]
    assert "dummy(r0t99)" in {event.get("name") for event in static_events}
    assert "dummy(r0t99)" in {event.get("name") for event in observed_events}
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
    assert "missing 1 critical task(s): run 42: task 2" in error


def _write_two_run_artifacts(output_dir, runtime):
    _write_visualization_artifacts(output_dir)
    records_path = output_dir / "chip_swimlane_records.json"
    records = json.loads(records_path.read_text())
    records["metadata"]["runtime"] = runtime
    # The same task ids recur, but the second run is gated by task 1 rather
    # than task 2. Their critical paths must be chosen independently.
    records["aicore_tasks"].extend([[0, 1, 0, 100, 119, 0, 43], [1, 2, 0, 100, 110, 0, 43], [1, 3, 0, 120, 130, 0, 43]])
    records_path.write_text(json.dumps(records))
    trace_path = output_dir / "merged_swimlane_20260720_120000.json"
    trace = json.loads(trace_path.read_text())
    trace["metadata"]["runtime"] = runtime
    second_run = []
    for event in trace["traceEvents"]:
        if event.get("ph") == "M":
            continue
        cloned = copy.deepcopy(event)
        if "id" in cloned:
            cloned["id"] += 1000
        if "ts" in cloned:
            cloned["ts"] += 100
        if "taskId" in cloned.get("args", {}):
            cloned["args"]["run_epoch"] = 43
            if cloned.get("pid") == 4:
                cloned["dur"] = {1: 19, 2: 10, 3: 10}[cloned["args"]["taskId"]]
        second_run.append(cloned)
    trace["traceEvents"].extend(second_run)
    trace_path.write_text(json.dumps(trace))
    return trace_path


@pytest.mark.parametrize("runtime", [TMR_RUNTIME, HBG_RUNTIME])
def test_repeated_task_ids_keep_per_run_paths_and_trace_highlights(tmp_path, runtime):
    trace_path = _write_two_run_artifacts(tmp_path / "case", runtime)
    output_dir = trace_path.parent
    graphs = critical_path.build_graphs(output_dir, tmp_path, tol=2)
    assert [g.run_epoch for g in graphs] == [42, 43]
    assert [g.dur["1"] for g in graphs] == [10, 19]
    assert all(g.nblock == {"1": 1, "2": 1, "3": 1} for g in graphs)
    assert all(g.hb_pred == {"3": ["1"]} for g in graphs)
    assert [g.core_prev for g in graphs] == [{"3": ("2", 19)}, {"3": ("2", 110)}]
    results = [critical_path.analyze_rank(g, tol=2) for g in graphs]
    assert [r.cpm_len for r in results] == [20, 29]
    assert [[s.task for s in r.segments] for r in results] == [["2", "3"], ["1", "3"]]
    assert all((r.compute_total, r.stall_total, r.makespan) == (29, 1, 30) for r in results)
    with pytest.raises(ValueError, match="multiple runs"):
        critical_path.build_graph(output_dir, tmp_path, tol=2)

    assert critical_path.main([str(output_dir)]) == 0
    report = (output_dir / "critical_path_report.md").read_text()
    assert "case / run 42" in report and "case / run 43" in report
    assert report.count("compute+stall = 30 ticks vs makespan 30 ticks (exact)") == 2
    source = json.loads(trace_path.read_text())["traceEvents"]
    expected = {
        critical_path.STATIC_TRACE_NAME: {(42, 1), (42, 3), (43, 1), (43, 3)},
        critical_path.OBSERVED_TRACE_NAME: {(42, 2), (42, 3), (43, 1), (43, 3)},
    }
    for filename, selected in expected.items():
        events = json.loads((output_dir / filename).read_text())["traceEvents"]
        assert len(events) == len(source)
        highlighted = set()
        for event, original in zip(events, source):
            if event.get("ph") == "X" and event.get("pid") == 4 and "taskId" in event.get("args", {}):
                if not event["name"].startswith("·("):
                    highlighted.add((event["args"]["run_epoch"], event["args"]["taskId"]))
                assert {k: v for k, v in event.items() if k != "name"} == {
                    k: v for k, v in original.items() if k != "name"
                }
            else:
                assert event == original
        assert highlighted == selected


def test_two_ten_tick_runs_do_not_turn_inter_run_idle_into_compute(tmp_path):
    _write_rank_artifacts(tmp_path / "case", "name_map.json")
    path = tmp_path / "case" / "chip_swimlane_records.json"
    data = json.loads(path.read_text())
    data["aicore_tasks"] = [[0, 1, 0, 0, 10, 0, 0], [0, 1, 0, 100, 110, 0, 8]]
    path.write_text(json.dumps(data))
    graphs = critical_path.build_graphs(path.parent, tmp_path, tol=2)
    assert [g.run_epoch for g in graphs] == [0, 8]
    for graph in graphs:
        result = critical_path.analyze_rank(graph, tol=2)
        assert (result.compute_total, result.stall_total, result.makespan) == (10, 0, 10)
        assert graph.nblock == {"1": 1}


@pytest.mark.parametrize("row_width", [5, 6])
def test_legacy_records_and_unstamped_traces_remain_readable(tmp_path, row_width):
    output_dir = tmp_path / "case"
    _write_visualization_artifacts(output_dir)
    records_path = output_dir / "chip_swimlane_records.json"
    data = json.loads(records_path.read_text())
    data["aicore_tasks"] = [row[:row_width] for row in data["aicore_tasks"]]
    records_path.write_text(json.dumps(data))
    trace_path = output_dir / "merged_swimlane_20260720_120000.json"
    trace = json.loads(trace_path.read_text())
    for event in trace["traceEvents"]:
        event.get("args", {}).pop("run_epoch", None)
    trace_path.write_text(json.dumps(trace))
    assert critical_path.main([str(output_dir)]) == 0
    assert (output_dir / critical_path.OBSERVED_TRACE_NAME).exists()


@pytest.mark.parametrize("bad_epoch", [None, -1, True, "42"])
def test_invalid_epoch_is_rejected_before_writing_a_report(tmp_path, capsys, bad_epoch):
    output_dir = tmp_path / "case"
    _write_rank_artifacts(output_dir, "name_map.json")
    path = output_dir / "chip_swimlane_records.json"
    data = json.loads(path.read_text())
    data["aicore_tasks"][0][6] = bad_epoch
    path.write_text(json.dumps(data))
    assert critical_path.main([str(output_dir)]) == 2
    assert "invalid run_epoch" in capsys.readouterr().err
    assert not (output_dir / "critical_path_report.md").exists()


def test_mixed_legacy_and_identified_rows_are_not_assigned_to_a_run(tmp_path, capsys):
    output_dir = tmp_path / "case"
    _write_visualization_artifacts(output_dir)
    path = output_dir / "chip_swimlane_records.json"
    data = json.loads(path.read_text())
    data["aicore_tasks"][1] = data["aicore_tasks"][1][:6]
    path.write_text(json.dumps(data))
    assert critical_path.main([str(output_dir)]) == 2
    assert "mixing identified and legacy" in capsys.readouterr().err
    assert not (output_dir / "critical_path_report.md").exists()


def test_same_task_id_in_another_run_cannot_hide_a_missing_trace_bar(tmp_path, capsys):
    trace_path = _write_two_run_artifacts(tmp_path / "case", TMR_RUNTIME)
    trace = json.loads(trace_path.read_text())
    trace["traceEvents"] = [
        event
        for event in trace["traceEvents"]
        if not (event.get("pid") == 4 and event.get("args", {}) == {"taskId": 1, "run_epoch": 43})
    ]
    trace_path.write_text(json.dumps(trace))
    assert critical_path.main([str(trace_path.parent)]) == 2
    assert "run 43: task 1" in capsys.readouterr().err
    assert not (trace_path.parent / critical_path.STATIC_TRACE_NAME).exists()
    assert not (trace_path.parent / critical_path.OBSERVED_TRACE_NAME).exists()
