# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scalar detail must conserve time and keep worker/run attribution."""

import copy
import gzip
import json
import re
from pathlib import Path

import pytest
from _capture_builder import Capture

from simpler_setup.tools.hbg.phases import SCALAR_PHASES
from simpler_setup.tools.hbg.scalar_profile import SCALAR_FUNCTIONS, analyze_streams, partition, trace_events
from simpler_setup.tools.swimlane_converter import _atomic_json, generate_chrome_trace_json, read_perf_data


def record(kind, begin, end, depth, **extra):
    return {
        "kind": kind,
        "start_cycles": begin,
        "end_cycles": end,
        "scalar_depth": depth,
        "scalar_calls": 1,
        "task_id": None,
        "run_epoch": 1,
        **extra,
    }


def window():
    return record("scalar_window", 10, 100, 0, scalar_schema=2, function_min_cycles=0)


def test_scalar_wire_vocabulary_matches_cpp_and_names_every_function():
    header = Path(__file__).resolve().parents[3] / "src/common/host_build_graph/sched_phase_kind.h"
    source = header.read_text()
    kinds = dict(re.findall(r"\b(Scalar\w+)\s*=\s*(\d+)", source))
    wire_names = dict(re.findall(r'case SchedPhaseKind::(Scalar\w+):\s*return "(scalar_\w+)";', source))
    assert set(kinds) == set(wire_names)
    assert len(set(kinds.values())) == len(kinds)
    assert set(wire_names.values()) == SCALAR_PHASES
    assert set(SCALAR_FUNCTIONS) == SCALAR_PHASES - {"scalar_window"}


def test_long_function_names_and_primitives_survive_filtered_nesting():
    rows = [
        window() | {"start_cycles": 100, "end_cycles": 10_100, "function_min_cycles": 1000},
        record("scalar_execution_token_refill", 200, 8200, 1, task_id=7),
        record("scalar_execution_token_receive", 1000, 5000, 2, task_id=7),
        record("scalar_prepared_acquire", 2000, 4000, 3, task_id=7),
        record("scalar_dcci_out", 2500, 2600, 4),
        record("scalar_admission", 6000, 8000, 2, task_id=7),
    ]
    profile = partition(rows, [])
    profile.update(worker_id=0, scheduler_id=0, clock_freq_hz=1_000_000_000)
    for row in profile["rows"]:
        row["start_time_us"] = row["start_cycles"] / 1000
        row["end_time_us"] = row["end_cycles"] / 1000
    events = [
        event
        for event in trace_events([profile], pid=2, tid_base=30000, display=lambda t: f"t{t}")
        if event["ph"] == "X"
    ]
    assert [(event["name"], event["dur"]) for event in events] == [
        ("scheduler_refill_execution_tokens(t7)", 8.0),
        ("scheduler_execution_token_receive(t7)", 4.0),
        ("scheduler_acquire_execution_token(t7)", 2.0),
        ("DCCI: SINGLE CACHELINE_OUT issue", 0.1),
        ("scheduler_ordered_can_enter_slot(t7)", 2.0),
    ]
    assert [event["args"]["scalar_depth"] for event in events] == [1, 2, 3, 4, 2]
    assert sum(profile["exclusive_cycles"].values()) == 10_000


def test_nested_intervals_conserve_time_without_counting_parents_twice():
    records = [window(), record("scalar_fill_slots", 15, 35, 1), record("scalar_atomic_claim", 20, 25, 2)]
    kernels = [{"start_cycles": 40, "end_cycles": 80, "run_epoch": 1, "task_id": 7}]
    result = partition(records, kernels)
    assert result["window_cycles"] == 90
    assert result["kernel_cycles"] == 40
    assert result["nonkernel_cycles"] == 50
    assert result["exclusive_cycles"]["scalar_fill_slots"] == 15
    assert result["exclusive_cycles"]["scalar_atomic_claim"] == 5
    assert result["exclusive_cycles"]["scalar_unattributed"] == 30
    assert sum(result["exclusive_cycles"].values()) == 90


@pytest.mark.parametrize(
    "children",
    [
        [record("scalar_fill_slots", 15, 40, 1), record("scalar_seal", 30, 45, 1)],
        [record("scalar_fill_slots", 15, 40, 1), record("scalar_atomic_claim", 35, 45, 2)],
        [record("scalar_atomic_claim", 15, 20, 3)],
    ],
)
def test_overlaps_escaped_children_and_missing_parents_are_rejected(children):
    with pytest.raises(ValueError):
        partition([window(), *children], [])


def test_equal_endpoints_and_zero_duration_do_not_invent_time():
    result = partition(
        [
            window(),
            record("scalar_fill_slots", 20, 30, 1),
            record("scalar_atomic_claim", 20, 30, 2),
            record("scalar_seal", 30, 30, 1),
        ],
        [],
    )
    assert result["exclusive_cycles"].get("scalar_fill_slots", 0) == 0
    assert result["exclusive_cycles"]["scalar_atomic_claim"] == 10
    assert result["exclusive_cycles"]["scalar_unattributed"] == 80


def test_kernel_join_is_scoped_to_worker_and_run():
    stream = {
        "producer": "aicore",
        "worker_id": 2,
        "scheduler_id": 3,
        "capture": {"committed": 1, "dropped": 0, "truncated": False},
    }
    rows = [[2, 7, 0, 30, 40, 0, 1], [1, 7, 0, 20, 60, 0, 1], [2, 7, 0, 50, 80, 0, 2]]
    profiles = analyze_streams([[window()]], [stream], rows)
    assert profiles[0]["kernel_cycles"] == 10
    stream["capture"]["dropped"] = 1
    stream["capture"]["truncated"] = True
    partial = analyze_streams([[window()]], [stream], rows)[0]
    assert not partial["complete"]
    assert not partial["exclusive"]
    assert "exclusive_cycles" not in partial


@pytest.mark.parametrize("schema", [None, 1, 3])
def test_unsupported_scalar_schema_is_rejected(schema):
    with pytest.raises(ValueError, match="supported Scalar window"):
        partition([window() | {"scalar_schema": schema}], [])


@pytest.mark.parametrize("level", [3, 4])
def test_common_converter_draws_one_typed_scalar_hierarchy(tmp_path, level):
    capture = Capture(runtime="host_build_graph", level=level, clock_freq_hz=1_000_000).task(
        task_id=7, reg_task_id=7, start=40, end=70, dispatch=35, finish=75, receive_to_start=5
    )
    for kind, begin, end, depth in [
        ("scalar_window", 10, 100, 0),
        ("scalar_fill_slots", 15, 30, 1),
        ("scalar_atomic_claim", 20, 25, 2),
        ("scalar_seal", 75, 90, 1),
    ]:
        metrics = {"scalar_depth": depth, "scalar_calls": 1}
        if depth == 0:
            metrics.update(scalar_schema=2, function_min_cycles=0)
        capture.aicore_sched_phase(
            phase=kind, start=begin, end=end, scheduler_id=0, worker_id=0, physical_core_id=9, metrics=metrics
        )
    capture.aicore_sched_phase(phase="idle", start=30, end=40, scheduler_id=0, worker_id=0, physical_core_id=9)
    raw = tmp_path / "raw.json"
    document = capture.build()
    for stream in document["scheduler_records"]["streams"]:
        stream.update(cluster_index=7, cluster_lane=1)
    raw.write_text(json.dumps(document))
    data = read_perf_data(raw)
    assert data["scalar_profiles"][0]["nonkernel_cycles"] == 60
    output = tmp_path / "trace.json"
    generate_chrome_trace_json(
        data["tasks"],
        str(output),
        runtime_name="host_build_graph",
        scheduler_phases=data["scheduler_records"],
        scheduler_streams=data["scheduler_streams"],
        scalar_profiles=data["scalar_profiles"],
        scheduler_task_producer=data["scheduler_task_producer"],
        chip_swimlane_level=data["chip_swimlane_level"],
    )
    events = json.loads(output.read_text())["traceEvents"]
    names = {e["args"]["name"] for e in events if e["ph"] == "M" and e["name"] == "thread_name" and e["pid"] == 2}
    assert names == {"Cluster 07 / V0 / AIV_0 (physical 9)"}
    assert not any(e.get("pid") == 3 for e in events)
    assert any(
        e.get("pid") == 4 and e.get("name") == "thread_name" and e["args"]["name"] == "Cluster 07 / V0 / AIV_0"
        for e in events
    )
    detail = [e for e in events if e["ph"] == "X" and e["pid"] == 2]
    assert {e["tid"] for e in detail} == {30000}
    assert all("cat" not in e and "args" not in e for e in detail)
    assert json.loads(output.read_text())["metadata"]["scalar_detail_display"]["all_events_retained"]
    by_name = {e["name"]: e for e in detail}
    assert "idle" not in by_name
    parent, child = by_name["scheduler_fill_own_slots"], by_name["scheduler_gm_compare_exchange (return ready)"]
    assert parent["ts"] <= child["ts"] < child["ts"] + child["dur"] <= parent["ts"] + parent["dur"]
    assert "window" not in by_name
    assert data["scalar_profiles"][0]["window_cycles"] == 90


def test_predicated_skip_is_not_counted_as_kernel_execution():
    stream = {
        "producer": "aicore",
        "worker_id": 2,
        "scheduler_id": 3,
        "capture": {"committed": 1, "dropped": 0, "truncated": False},
    }
    identities = {"records": [{"core_id": 2, "reg_task_id": 4, "run_epoch": 1, "executed": False}]}
    result = analyze_streams([[window()]], [stream], [[2, 7, 4, 30, 40, 0, 1]], identities)[0]
    assert result["kernel_cycles"] == 0
    assert result["exclusive_cycles"]["scalar_task_skip"] == 10
    assert result["nonkernel_cycles"] == 90


@pytest.mark.parametrize(
    ("phase", "function"),
    [
        ("scalar_execution_token_refill", "scheduler_refill_execution_tokens"),
        ("scalar_prepared_acquire", "scheduler_acquire_execution_token"),
        ("scalar_cooperative_discover", "scheduler_discover_cooperative_task"),
        ("scalar_cluster_mix_service", "scheduler_cluster_mix_service"),
        ("scalar_group_acquire", "scheduler_acquire_group_lane"),
        ("scalar_cluster_mix_needs_service", "scheduler_cluster_mix_needs_service"),
        ("scalar_execution_continuation", "scheduler_try_execution_continuation"),
        ("scalar_execution_token_fanin", "scheduler_execution_token_observe_fanin"),
        ("scalar_cluster_mix_ready_replacement", "scheduler_cluster_mix_ready_replacement"),
        ("scalar_ordered_poll", "scheduler_ordered_poll_may_progress"),
        ("scalar_admission_claim", "scheduler_stage_admission_candidate"),
        ("scalar_admission_publish", "scheduler_ordered_publish_pending"),
        ("scalar_admission_commit", "scheduler_try_commit_admission"),
        ("scalar_admission_refresh", "scheduler_refresh_local_obligation"),
        ("scalar_ordinary_park", "scheduler_ordered_park_ordinary"),
        ("scalar_block_claim", "scheduler_claim_ordered_block"),
        ("scalar_block_cursor_read", "scheduler_read_claim_cursor"),
        ("scalar_admission_close", "scheduler_close_ordered_obligation"),
        ("scalar_ordinary_permit", "scheduler_can_enter_ordinary"),
        ("scalar_dispatch_metadata", "scheduler_load_dispatch_metadata"),
        ("scalar_dispatch_fill", "scheduler_fill_dispatch_slot"),
        ("scalar_dispatch_publish", "scheduler_publish_dispatch_slot"),
        ("scalar_inline_complete", "scheduler_complete_inline_task"),
        ("scalar_cursor_validate", "scheduler_read_claim_cursor: validate context"),
        ("scalar_claim_seal_probe", "scheduler_read_claim_cursor: claim_seal hint"),
        ("scalar_claim_seal_confirm", "scheduler_read_claim_cursor: claim_seal atomic confirm"),
        ("scalar_next_block_probe", "scheduler_read_claim_cursor: next_block hint"),
        ("scalar_next_block_confirm", "scheduler_read_claim_cursor: next_block atomic confirm"),
        ("scalar_poll_local_state", "scheduler_ordered_poll_may_progress: local state"),
        ("scalar_poll_ready_queue", "scheduler_ordered_poll_may_progress: ready queue"),
        ("scalar_poll_ready_pool", "scheduler_ready_pool_may_have_work"),
        ("scalar_poll_admission_tail", "scheduler_ordered_poll_may_progress: admission commit_tail"),
        ("scalar_poll_mix_state", "scheduler_ordered_poll_may_progress: MIX state"),
        ("scalar_poll_mix_close_seal", "scheduler_ordered_poll_may_progress: MIX close seal"),
        ("scalar_poll_mix_assignment", "scheduler_ordered_poll_may_progress: MIX assigned_seq"),
        ("scalar_poll_mix_arrived", "scheduler_ordered_poll_may_progress: MIX arrived_seq"),
        ("scalar_poll_mix_released", "scheduler_ordered_poll_may_progress: MIX released_seq"),
        ("scalar_trace_finish", "SchedulerScalarTrace::finish"),
        ("scalar_trace_finish_clean", "SchedulerScalarTrace::finish: DCCI issue"),
        ("scalar_trace_finish_barrier", "SchedulerScalarTrace::finish: DSB"),
    ],
)
def test_fill_detail_roundtrips_nested_function_names_and_task_identity(tmp_path, phase, function):
    capture = Capture(runtime="host_build_graph", level=3, clock_freq_hz=1_000_000).task(
        task_id=7, reg_task_id=7, start=40, end=70, dispatch=35, finish=75, receive_to_start=5
    )
    for kind, begin, end, depth in [
        ("scalar_window", 10, 100, 0),
        ("scalar_fill_slots", 15, 35, 1),
        ("scalar_ordered_fill", 16, 34, 2),
        (phase, 20, 25, 3),
    ]:
        metrics = {"scalar_depth": depth, "scalar_calls": 1, "iterations_detailed": True}
        if depth == 0:
            metrics.update(scalar_schema=2, function_min_cycles=0)
        capture.aicore_sched_phase(
            phase=kind,
            start=begin,
            end=end,
            task_id=7 if depth == 3 else None,
            scheduler_id=0,
            worker_id=0,
            metrics=metrics,
        )
    raw = tmp_path / "raw.json"
    raw.write_text(json.dumps(capture.build()))
    data = read_perf_data(raw)
    profile = data["scalar_profiles"][0]
    assert profile["complete"]
    assert profile["exclusive_cycles"][phase] == 5
    assert sum(profile["exclusive_cycles"].values()) == 90
    output = tmp_path / "trace.json"
    generate_chrome_trace_json(
        data["tasks"],
        str(output),
        runtime_name="host_build_graph",
        scheduler_phases=data["scheduler_records"],
        scheduler_streams=data["scheduler_streams"],
        scalar_profiles=data["scalar_profiles"],
    )
    details = [e for e in json.loads(output.read_text())["traceEvents"] if e["ph"] == "X" and e["pid"] == 2]
    parent = next(e for e in details if e["name"] == "scheduler_fill_ordered_slots")
    child = next(e for e in details if e["name"].startswith(function + "("))
    assert child["name"] == f"{function}(t7)"
    assert next(row for row in profile["rows"] if row["kind"] == phase)["task_id"] == 7
    assert parent["ts"] <= child["ts"] < child["ts"] + child["dur"] <= parent["ts"] + parent["dur"]


def test_compact_trace_keeps_every_interval_and_raw_source_unchanged(tmp_path):
    event = {
        "ph": "X",
        "cat": "aicore_scalar",
        "cname": "generic_work",
        "name": "DCCI: SINGLE CACHELINE_OUT issue(7)",
        "pid": 2,
        "tid": 30000,
        "ts": 1.0000000001,
        "dur": 0.0120000001,
        "args": {
            **record("scalar_dcci_out", 10, 22, 2, task_id=7),
            "scalar_calls": 8,
            "capture_complete": True,
            "rank": 0,
        },
    }
    source = {"traceEvents": [copy.deepcopy(event) for _ in range(100)], "metadata": {"runtime": "host_build_graph"}}
    original = copy.deepcopy(source)
    plain, zipped = tmp_path / "trace.json", tmp_path / "trace.json.gz"
    _atomic_json(plain, source)
    _atomic_json(zipped, source)
    shown = json.loads(plain.read_text())
    with gzip.open(zipped, "rt") as stream:
        assert json.load(stream) == shown
    assert source == original
    assert len(shown["traceEvents"]) == 100
    for row in shown["traceEvents"]:
        assert row["name"].endswith(" [calls=8]")
        assert (row["ts"], row["dur"], row["pid"], row["tid"]) == (event["ts"], event["dur"], 2, 30000)
        assert row["args"] == {"rank": 0}
    lane = shown["metadata"]["scalar_detail_display"]["lanes"][0]
    assert lane == {"pid": 2, "tid": 30000, "run_epochs": [1], "capture_complete_values": [True]}
    assert plain.stat().st_size < len(json.dumps(source, indent=2)) / 2
    assert zipped.stat().st_size < plain.stat().st_size / 4


def test_compact_trace_retains_per_event_identity_when_runs_share_a_lane(tmp_path):
    events = [
        {
            "ph": "X",
            "cat": "aicore_scalar",
            "name": "atomicCAS: return ready",
            "pid": 2,
            "tid": 30000,
            "ts": epoch * 10,
            "dur": 1,
            "args": {"run_epoch": epoch, "capture_complete": True, "scalar_calls": 1},
        }
        for epoch in (1, 2)
    ]
    task = {
        "ph": "X",
        "name": "model kernel",
        "pid": 3,
        "tid": 0,
        "ts": 0,
        "dur": 1,
        "args": {"task_id": 7, "run_epoch": 1},
    }
    path = tmp_path / "trace.json"
    _atomic_json(path, {"traceEvents": [*events, task]})
    shown = json.loads(path.read_text())
    assert [row["args"]["run_epoch"] for row in shown["traceEvents"][:2]] == [1, 2]
    assert shown["traceEvents"][2] == task


def test_filtered_function_scopes_keep_memory_children_and_exact_time():
    root = window() | {"scalar_schema": 2, "function_min_cycles": 10, "short_function_intervals_omitted": 2}
    rows = [
        root,
        record("scalar_ordered_fill", 20, 80, 1),
        record("scalar_atomic_cas", 30, 35, 3),
        record("scalar_dcci_out", 35, 39, 3),
        record("scalar_dsb_all", 40, 55, 3),
        record("scalar_bypass_load", 82, 84, 2),
    ]
    original = copy.deepcopy(rows)
    result = partition(rows, [])
    assert rows == original
    assert len(result["rows"]) == len(rows)
    assert result["exclusive_cycles"]["scalar_atomic_cas"] == 5
    assert result["exclusive_cycles"]["scalar_dcci_out"] == 4
    assert result["exclusive_cycles"]["scalar_dsb_all"] == 15
    assert sum(result["exclusive_cycles"].values()) == 90
    leaf = next(r for r in result["rows"] if r["kind"] == "scalar_atomic_cas")
    assert leaf["scalar_depth"] == 2 and leaf["raw_scalar_depth"] == 3
    assert result["short_function_intervals_omitted"] == 2


@pytest.mark.parametrize(
    "children",
    [
        [record("scalar_seal", 20, 50, 1), record("scalar_dsb_all", 30, 40, 1)],
        [record("scalar_seal", 20, 50, 1), record("scalar_dsb_all", 40, 60, 3)],
        [record("scalar_dcci_out", 95, 105, 3)],
    ],
)
def test_filtered_capture_still_rejects_sibling_overlap_and_escaped_bounds(children):
    root = window() | {"scalar_schema": 2, "function_min_cycles": 10}
    with pytest.raises(ValueError):
        partition([root, *children], [])


def test_filtered_schema_requires_an_explicit_valid_threshold():
    for extra in ({}, {"function_min_cycles": -1}, {"function_min_cycles": "10"}):
        with pytest.raises(ValueError, match="threshold"):
            root = window()
            del root["function_min_cycles"]
            partition([root | extra], [])


@pytest.mark.parametrize("frequency", [1_000_000_000, 2_500_000_000])
def test_scalar_duration_uses_integer_counters_without_rounding(frequency):
    row = record("scalar_bypass_load", 12345123, 12345136, 1)
    row.update(start_time_us=12345.123, end_time_us=12345.136)
    profile = {"worker_id": 0, "scheduler_id": 0, "complete": True, "clock_freq_hz": frequency, "rows": [row]}
    events = trace_events([profile], pid=2, tid_base=30000, display=str)
    interval = next(event for event in events if event["ph"] == "X")
    assert interval["dur"] == 13 * 1_000_000 / frequency
    assert interval["ts"] == row["start_time_us"]
    if frequency == 1_000_000_000:
        assert json.dumps(interval["dur"]) == "0.013"
    else:
        assert interval["dur"] == 0.0052
