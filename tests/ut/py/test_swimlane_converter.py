#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import json
from pathlib import Path

import pytest

from simpler_setup.tools import swimlane_converter as sc


def _task_row(task_id, core_id, core_type="aiv", *, func_id=0, dispatch=10.0, start=11.0, end=20.0, receive=10.5):
    return {
        "task_id": task_id,
        "func_id": func_id,
        "core_id": core_id,
        "core_type": core_type,
        "start_time_us": start,
        "end_time_us": end,
        "duration_us": end - start,
        "dispatch_time_us": dispatch,
        "finish_time_us": end + 1.0,
        "receive_time_us": receive,
        "local_setup_us": start - receive,
    }


def _count_dependency_flow_starts(trace_path, *, pid, tid=None):
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    return sum(
        1
        for e in events
        if e.get("cat") == "flow"
        and e.get("name") in ("dependency", "hb_violation")
        and e.get("ph") == "s"
        and e.get("pid") == pid
        and (tid is None or e.get("tid") == tid)
    )


def _first_worker_dependency_flow(trace_path):
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    flow_id = next(
        e["id"]
        for e in events
        if e.get("cat") == "flow"
        and e.get("name") in ("dependency", "hb_violation")
        and e.get("ph") == "s"
        and e.get("pid") == 4
    )
    return [e for e in events if e.get("cat") == "flow" and e.get("id") == flow_id and e.get("pid") == 4]


def _first_scheduler_dependency_flow(trace_path):
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    flow_id = next(
        e["id"]
        for e in events
        if e.get("cat") == "flow"
        and e.get("name") in ("dependency", "hb_violation")
        and e.get("ph") == "s"
        and e.get("pid") == 3
    )
    return [e for e in events if e.get("cat") == "flow" and e.get("id") == flow_id and e.get("pid") == 3]


def _worker_flow_finish_tids(trace_path):
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    return {
        e["tid"]
        for e in events
        if e.get("cat") == "flow"
        and e.get("name") in ("dependency", "hb_violation")
        and e.get("ph") == "f"
        and e.get("pid") == 4
    }


def _has_spmd_block_level_track(trace_path):
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    return any(
        e.get("ph") == "M" and e.get("name") == "thread_name" and e.get("args", {}).get("name") == "SPMD (block-level)"
        for e in events
    )


def _core_tid(core_id):
    return 10000 + core_id * 10


def _generate_trace(tasks, deps_edges, deps_block_map, tmp_path):
    out = tmp_path / "trace.json"
    sc.generate_chrome_trace_json(
        tasks,
        str(out),
        deps_edges=deps_edges,
        deps_block_map=deps_block_map,
    )
    return out


def _write_l3_rank(root, rank, *, host_shift_ns, task_id, clock_domain="same-boot", dispatch="d0"):
    rank_dir = root / f"rank{rank}" / dispatch
    rank_dir.mkdir(parents=True)
    device_base = 100 + rank * 100_000
    records = {
        "chip_swimlane_level": 4,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": 1,
            "core_types": ["aiv"],
            "core_to_thread": [0],
            "orchestrator_source": "host",
            "orchestrator_clock_domain": "host_monotonic_ns",
            "host_clock_domain_id": clock_domain,
            "host_orchestration_origin_ns": host_shift_ns + 1_500,
            "host_capture": {
                "status": "complete",
                "expected_records": 1,
                "recorded_records": 1,
                "dropped_records": 0,
                "error": None,
            },
            "clock_anchors": {
                "device_timestamp_unit": "syscnt_cycles",
                "samples": [
                    {
                        "position": "pre_host_orchestration",
                        "sample_idx": 0,
                        "host_before_ns": host_shift_ns + 990,
                        "device_cycles": device_base,
                        "host_after_ns": host_shift_ns + 1_010,
                        "error": None,
                    },
                    {
                        "position": "post_device_execution",
                        "sample_idx": 0,
                        "host_before_ns": host_shift_ns + 8_980,
                        "device_cycles": device_base + 8_000,
                        "host_after_ns": host_shift_ns + 9_020,
                        "error": None,
                    },
                ],
            },
        },
        "aicore_tasks": [[0, task_id, 1, device_base + 1_000, device_base + 1_100, 0]],
        "aicpu_tasks": [[0, 1, device_base + 900, device_base + 1_200]],
        "aicpu_scheduler_phases": [
            [{"kind": "dispatch", "start_cycles": device_base + 800, "end_cycles": device_base + 850}]
        ],
        "host_orchestrator_phases": [
            [
                {
                    "submit_idx": 0,
                    "task_id": task_id,
                    "start_host_ns": host_shift_ns + 1_500,
                    "end_host_ns": host_shift_ns + 1_800,
                }
            ]
        ],
    }
    (rank_dir / "chip_swimlane_records.json").write_text(json.dumps(records))
    (rank_dir / "name_map.json").write_text(json.dumps({"callable_id_to_name": {"0": f"kernel_r{rank}"}}))
    return rank_dir


def _write_dispatch_identity(capture_dir, *, run_id, task_slot, group_index, group_size):
    rank = int(capture_dir.parent.name.removeprefix("rank"))
    capture_index = int(capture_dir.name.removeprefix("d"))
    (capture_dir / "dispatch_identity.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "run_id": run_id,
                "task_slot": task_slot,
                "group_index": group_index,
                "group_size": group_size,
                "chip_rank": rank,
                "local_capture_index": capture_index,
                "endpoint_dispatch_id": capture_index + 1,
                "pipeline_slot": 0,
                "pipeline_generation": 1,
                "callable_digest": "ab" * 32,
            }
        )
    )


def test_l3_directory_merge_uses_common_host_origin_and_rank_namespaces(tmp_path):
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    output = tmp_path / "l3.json"
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0", "-o", str(output)])

    output_path, rank_metadata = sc._generate_l3_trace(args, root)

    assert output_path == output
    assert [item["rank"] for item in rank_metadata] == [0, 1]
    trace = json.loads(output.read_text())
    assert trace["metadata"]["global_origin_ns"] == 1_500
    assert trace["metadata"]["host_clock_domain_id"] == "same-boot"
    assert trace["metadata"]["cross_rank_uncertainty_ns"] == 40
    assert trace["metadata"]["pre_anchor_group_duration_spread_ns"] == 0
    assert trace["metadata"]["pre_anchor_group_duration_max_ns"] == 20
    assert trace["metadata"]["dispatch_pairing"] == "local_capture_index"

    process_names = {
        event["args"]["name"]
        for event in trace["traceEvents"]
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    assert "rank0 / Worker View" in process_names
    assert "rank1 / Worker View" in process_names
    worker_events = {
        event["args"]["taskId"]: event
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event.get("cat") == "event" and event.get("pid") % 100 == 4
    }
    assert worker_events[7]["pid"] == 4
    assert worker_events[7]["ts"] == 0.5
    assert worker_events[8]["pid"] == 104
    assert worker_events[8]["ts"] == 10.5


def test_l3_directory_merge_keeps_scheduler_streams_and_lifecycle_records(tmp_path):
    root = tmp_path / "dfx_outputs"
    for rank in (0, 1):
        capture_dir = _write_l3_rank(root, rank, host_shift_ns=rank * 10_000, task_id=rank + 7)
        records_path = capture_dir / "chip_swimlane_records.json"
        records = json.loads(records_path.read_text())
        device_base = records["aicore_tasks"][0][3] - 1_000
        records["scheduler_records"] = {
            "schema_version": 1,
            "streams": [
                {
                    "platform": "a5",
                    "runtime": "host_build_graph",
                    "producer": "aicore",
                    "scheduler_id": rank + 2,
                    "worker_id": rank + 4,
                    "core_type": "aiv",
                    "physical_core_id": rank,
                    "capture": {"committed": 1, "dropped": 0, "truncated": False},
                    "records": [
                        {
                            "start_cycles": device_base + 700,
                            "end_cycles": device_base + 750,
                            "loop_iter": 0,
                            "kind": "ready_claim",
                            "tasks_processed": 1,
                            "task_id": rank + 7,
                        }
                    ],
                    "metrics": [],
                }
            ],
        }
        records["aicpu_lifecycle_records"] = [
            {
                "worker_id": rank + 4,
                "aicpu_thread_id": rank,
                "core_type": "aiv",
                "physical_core_id": rank,
                "register_release_cycles": device_base + 600,
            }
        ]
        records_path.write_text(json.dumps(records))

    output = tmp_path / "l3.json"
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0", "-o", str(output)])

    sc._generate_l3_trace(args, root)

    events = json.loads(output.read_text())["traceEvents"]
    process_names = {
        event["args"]["name"] for event in events if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    assert "rank0 / AICore Scheduler" in process_names
    assert "rank1 / AICore Scheduler" in process_names
    register_releases = [event for event in events if event.get("name") == "register_release"]
    assert {event["args"]["rank"] for event in register_releases} == {0, 1}


def test_l3_parent_dispatch_identity_pairs_different_local_capture_indexes(tmp_path):
    root = tmp_path / "dfx_outputs"
    rank0 = _write_l3_rank(root, 0, host_shift_ns=0, task_id=7, dispatch="d0")
    rank1 = _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8, dispatch="d1")
    _write_dispatch_identity(rank0, run_id=17, task_slot=5, group_index=0, group_size=2)
    _write_dispatch_identity(rank1, run_id=17, task_slot=5, group_index=1, group_size=2)
    output = tmp_path / "semantic.json"
    args = sc._build_parser().parse_args([str(root), "--dispatch-id", "17:5", "-o", str(output)])

    _, rank_metadata = sc._generate_l3_trace(args, root)

    trace = json.loads(output.read_text())
    assert [Path(item["input"]).parent.name for item in rank_metadata] == ["d0", "d1"]
    assert trace["metadata"]["dispatch_pairing"] == "parent_dispatch_identity"
    assert trace["metadata"]["dispatch_identity"] == {
        "run_id": 17,
        "task_slot": 5,
        "group_size": 2,
        "callable_digest": "ab" * 32,
    }
    assert [item["dispatch_identity"]["group_index"] for item in rank_metadata] == [0, 1]


def test_l3_local_capture_selector_rejects_different_parent_dispatches(tmp_path):
    root = tmp_path / "dfx_outputs"
    rank0 = _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    rank1 = _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    _write_dispatch_identity(rank0, run_id=17, task_slot=5, group_index=0, group_size=2)
    _write_dispatch_identity(rank1, run_id=17, task_slot=6, group_index=1, group_size=2)
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    with pytest.raises(ValueError, match="different parent dispatches"):
        sc._generate_l3_trace(args, root)


def test_l3_auto_discovery_rejects_incomplete_parent_group(tmp_path):
    root = tmp_path / "dfx_outputs"
    rank0 = _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    _write_dispatch_identity(rank0, run_id=17, task_slot=5, group_index=0, group_size=2)

    with pytest.raises(ValueError, match="incomplete parent dispatch 17:5"):
        sc.discover_l3_conversion_targets(root)


def test_l3_auto_discovery_pairs_two_groups_despite_reordered_d_paths(tmp_path):
    root = tmp_path / "dfx_outputs"
    captures = {
        (0, "d0"): (17, 5, 0),
        (0, "d1"): (17, 6, 0),
        (1, "d0"): (17, 6, 1),
        (1, "d1"): (17, 5, 1),
    }
    for (rank, dispatch), (run_id, task_slot, group_index) in captures.items():
        capture_dir = _write_l3_rank(
            root,
            rank,
            host_shift_ns=rank * 10_000,
            task_id=rank * 10 + int(dispatch.removeprefix("d")),
            dispatch=dispatch,
        )
        _write_dispatch_identity(
            capture_dir,
            run_id=run_id,
            task_slot=task_slot,
            group_index=group_index,
            group_size=2,
        )

    targets = sc.discover_l3_conversion_targets(root)

    assert [(target["dispatch"], target["dispatch_id"]) for target in targets] == [
        (None, "17:5"),
        (None, "17:6"),
    ]
    assert [[path.name for path in target["capture_dirs"]] for target in targets] == [["d0", "d1"], ["d1", "d0"]]


def test_l3_auto_discovery_keeps_paired_groups_when_the_remainder_is_asymmetric(tmp_path, capsys):
    # A group is paired by (run_id, task_slot), so it is unaffected by what the
    # leftover dN sets look like. Refusing the whole root would discard exactly
    # the pairing the parent identity exists to make.
    root = tmp_path / "dfx_outputs"
    for rank, group_index in ((0, 0), (1, 1)):
        capture_dir = _write_l3_rank(root, rank, host_shift_ns=rank * 10_000, task_id=rank, dispatch="d0")
        _write_dispatch_identity(capture_dir, run_id=17, task_slot=5, group_index=group_index, group_size=2)
    # An extra individually submitted capture on rank0 only: no sibling to pair
    # it with, and no identity that would let it pair by anything but its name.
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=99, dispatch="d1")

    targets = sc.discover_l3_conversion_targets(root)

    assert [(target["dispatch"], target["dispatch_id"]) for target in targets] == [(None, "17:5")]
    assert "refusing to pair asymmetric local capture indexes" in capsys.readouterr().err


def test_rank_namespace_does_not_turn_rank_into_a_counter_series():
    trace = {
        "traceEvents": [
            {"ph": "C", "pid": 2, "tid": 1, "args": {"AIC": 3, "AIV": 4}},
            {"ph": "X", "pid": 4, "tid": 2, "args": {"taskId": 9}},
        ]
    }

    sc._namespace_rank_trace(trace, 2)

    counter, task = trace["traceEvents"]
    assert counter["pid"] == 202
    assert counter["args"] == {"AIC": 3, "AIV": 4}
    assert task["pid"] == 204
    assert task["args"]["rank"] == 2


def test_rank_namespace_rejects_a_view_pid_wider_than_the_stride():
    trace = {"traceEvents": [{"ph": "X", "pid": sc._RANK_PID_STRIDE, "tid": 1, "args": {}}]}

    with pytest.raises(ValueError, match="does not fit the per-Rank stride"):
        sc._namespace_rank_trace(trace, 1)


def test_l3_directory_merge_rejects_different_or_missing_host_clock_domains(tmp_path):
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7, clock_domain="boot-a")
    rank1_dir = _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8, clock_domain="boot-b")
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    with pytest.raises(ValueError, match="different Host clock domains"):
        sc._generate_l3_trace(args, root)

    rank1_path = rank1_dir / "chip_swimlane_records.json"
    rank1 = json.loads(rank1_path.read_text())
    rank1["metadata"].pop("host_clock_domain_id")
    rank1_path.write_text(json.dumps(rank1))
    with pytest.raises(ValueError, match="missing metadata.host_clock_domain_id"):
        sc._generate_l3_trace(args, root)


@pytest.mark.parametrize(
    ("level", "description"),
    [(1, "AICore timing only"), (3, "AICore + Scheduler task timing + scheduler phases")],
)
def test_task_statistics_without_scheduler_timestamps_hides_scheduler_metrics(capsys, level, description):
    tasks = [
        {
            "task_id": 1,
            "func_id": 0,
            "core_id": 0,
            "core_type": "aic",
            "start_time_us": 1.0,
            "end_time_us": 6.0,
            "duration_us": 5.0,
            "dispatch_time_us": 0.0,
            "finish_time_us": 0.0,
            "receive_time_us": 0.5,
            "local_setup_us": 0.5,
        }
    ]

    sc.print_task_statistics(tasks, {"0": "kernel"}, chip_swimlane_level=level)

    output = capsys.readouterr().out
    row = next(line for line in output.splitlines() if line.startswith("0        kernel"))
    total = next(line for line in output.splitlines() if line.startswith("TOTAL"))
    assert f"Source chip_swimlane_level: {level} ({description}; recorded in chip_swimlane_records.json)" in output
    assert row.split() == ["0", "kernel", "1", "5.00", "-", "-", "-", "-", "-", "0.50"]
    assert total.split() == ["TOTAL", "1", "5.00", "-"]
    assert "AICore Observed Span: 5.50 us (from earliest AICore receive to latest AICore end)" in output
    assert "Total Test Time" not in output


def test_load_func_names_auto_discovery_and_explicit_precedence(tmp_path):
    input_path = tmp_path / "chip_swimlane_records.json"
    name_map_path = tmp_path / "name_map_case.json"
    name_map_path.write_text(
        json.dumps(
            {
                "callable_id_to_name": {"0": "kernel"},
                "orchestrator_name": "orchestrator",
            }
        )
    )
    args = sc._build_parser().parse_args([str(input_path)])

    func_names, orchestrator_name = sc._load_func_names(args, input_path)

    assert func_names == {"0": "kernel"}
    assert orchestrator_name == "orchestrator"

    explicit_path = tmp_path / "explicit.json"
    explicit_path.write_text(json.dumps({"callable_id_to_name": {"0": "explicit"}}))
    explicit_args = sc._build_parser().parse_args([str(input_path), "--func-names", str(explicit_path)])
    func_names, _ = sc._load_func_names(explicit_args, input_path)

    assert func_names == {"0": "explicit"}


def test_host_orchestrator_phases_without_anchors_are_marked_unaligned(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 4,
                "metadata": {
                    "clock_freq_hz": 1_000_000,
                    "num_cores": 1,
                    "core_types": ["aiv"],
                    "core_to_thread": [0],
                    "orchestrator_source": "host",
                    "orchestrator_clock_domain": "host_monotonic_ns",
                    "host_orchestration_origin_ns": 1_000,
                    "timeline_relation": "host_orchestration_precedes_device",
                    "host_capture": {
                        "status": "complete",
                        "expected_records": 1,
                        "recorded_records": 1,
                        "dropped_records": 0,
                        "error": None,
                    },
                },
                "aicore_tasks": [[0, 7, 1, 100, 110, 0]],
                "aicpu_tasks": [[0, 1, 90, 120]],
                "aicpu_scheduler_phases": [
                    [{"kind": "dispatch", "start_cycles": 80, "end_cycles": 85, "tasks_processed": 1}]
                ],
                "host_orchestrator_phases": [
                    [{"submit_idx": 0, "task_id": 7, "start_host_ns": 1_000, "end_host_ns": 3_000}]
                ],
            }
        )
    )

    data = sc.read_perf_data(raw)

    assert data["scheduler_task_producer"] == "aicpu"
    assert data["orchestrator_source"] == "host"
    assert data["aicpu_orchestrator_phases"][0][0]["start_time_us"] == 0.0
    assert data["aicpu_orchestrator_phases"][0][0]["end_time_us"] == 2.0
    assert data["aicpu_scheduler_phases"][0][0]["start_time_us"] == 2.0
    assert data["tasks"][0]["dispatch_time_us"] == 12.0
    assert data["timeline_metadata"] == {
        "layout": "causal_composite",
        "trace_status": "complete",
        "relation": "host_orchestration_precedes_device",
        "clock_alignment": {
            "status": "unaligned",
            "method": "nominal_frequency_offset_interp_v1",
            "anchor_uncertainty_ns": None,
            "host_timestamp_quantization_ns": 0,
            "max_uncertainty_ns": None,
            "reason": "missing_clock_anchors",
        },
        "host_capture": {
            "status": "complete",
            "expected_records": 1,
            "recorded_records": 1,
            "dropped_records": 0,
            "error": None,
        },
        "host_records_complete": True,
        "cross_domain_gap_unknown": True,
        "cross_domain_latency_available": False,
        "logical_seam_us": 2.0,
        "source_timeline_origin_ns": 1_000,
        "timeline_origin_ns": 1_000,
    }

    trace_path = tmp_path / "merged_swimlane.json"
    sc.generate_chrome_trace_json(
        data["tasks"],
        str(trace_path),
        scheduler_phases=data["aicpu_scheduler_phases"],
        orchestrator_phases=data.get("aicpu_orchestrator_phases"),
        orchestrator_source=data["orchestrator_source"],
        timeline_metadata=data["timeline_metadata"],
        core_to_thread=data["core_to_thread"],
    )
    trace = json.loads(trace_path.read_text())
    assert trace["metadata"]["clock_alignment"]["status"] == "unaligned"
    assert any(
        event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "Host Orchestrator"
        for event in trace["traceEvents"]
    )
    assert not any(
        event.get("cat") == "flow" and event.get("name") == "submit→dispatch" for event in trace["traceEvents"]
    )


def test_aicpu_orchestrator_uses_host_timeline_when_clock_anchors_exist(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 4,
                "metadata": {
                    "clock_freq_hz": 1_000_000_000,
                    "num_cores": 1,
                    "core_types": ["aiv"],
                    "core_to_thread": [0],
                    "host_clock_domain_id": "same-boot",
                    "host_timeline_origin_ns": 1_000,
                    "clock_anchors": {
                        "device_timestamp_unit": "syscnt_cycles",
                        "samples": [
                            {
                                "position": "pre_host_orchestration",
                                "sample_idx": 0,
                                "host_before_ns": 990,
                                "device_cycles": 100,
                                "host_after_ns": 1_010,
                                "error": None,
                            },
                            {
                                "position": "post_device_execution",
                                "sample_idx": 0,
                                "host_before_ns": 5_080,
                                "device_cycles": 4_100,
                                "host_after_ns": 5_120,
                                "error": None,
                            },
                        ],
                    },
                },
                "aicore_tasks": [[0, 7, 1, 2_100, 2_200, 0]],
                "aicpu_tasks": [[0, 1, 2_000, 2_300]],
                "aicpu_scheduler_phases": [
                    [{"kind": "dispatch", "start_cycles": 1_900, "end_cycles": 1_950, "tasks_processed": 1}]
                ],
                "aicpu_orchestrator_phases": [
                    [{"submit_idx": 0, "task_id": 7, "start_cycles": 1_800, "end_cycles": 1_850}]
                ],
            }
        )
    )

    data = sc.read_perf_data(raw)

    assert data["orchestrator_source"] == "aicpu"
    assert data["tasks"][0]["start_time_us"] == 2.05
    assert data["timeline_metadata"]["layout"] == "clock_aligned"
    assert data["timeline_metadata"]["clock_alignment"]["status"] == "calibrated"
    assert data["timeline_metadata"]["host_clock_domain_id"] == "same-boot"
    assert data["timeline_metadata"]["source_timeline_origin_ns"] == 1_000


def test_aicore_scheduler_records_keep_common_shape_and_stream_metadata(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 3,
                "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 1, "core_types": ["aiv"]},
                "aicore_tasks": [[0, 7, 7, 120, 180, 10]],
                "scheduler_tasks": {
                    "schema_version": 1,
                    "producer": "aicore",
                    "records": [[0, 7, 115, 185]],
                },
                "aicpu_lifecycle_records": [
                    {
                        "worker_id": 6,
                        "aicpu_thread_id": 1,
                        "core_type": "aiv",
                        "physical_core_id": 9,
                        "handshake_observed_cycles": 90,
                        "handshake_partition_complete_cycles": 91,
                        "config_start_cycles": 92,
                        "topology_complete_cycles": 93,
                        "context_publish_complete_cycles": 94,
                        "bootstrap_wait_start_cycles": 95,
                        "bootstrap_complete_cycles": 96,
                        "register_release_cycles": 97,
                        "exit_signal_cycles": 181,
                        "exit_ack_cycles": 182,
                    }
                ],
                "scheduler_records": {
                    "schema_version": 1,
                    "streams": [
                        {
                            "platform": "a5",
                            "runtime": "host_build_graph",
                            "producer": "aicore",
                            "scheduler_id": 2,
                            "worker_id": 6,
                            "core_type": "aiv",
                            "physical_core_id": 9,
                            "capture": {"committed": 2, "dropped": 0, "truncated": False},
                            "records": [
                                {
                                    "start_cycles": 100,
                                    "end_cycles": 110,
                                    "loop_iter": 3,
                                    "kind": "ready_claim",
                                    "tasks_processed": 1,
                                    "task_id": 7,
                                },
                                {
                                    "start_cycles": 111,
                                    "end_cycles": 119,
                                    "loop_iter": 3,
                                    "kind": "idle",
                                    "tasks_processed": 0,
                                    "task_id": None,
                                },
                            ],
                            "metrics": [{"record_index": 0, "claim_retries": 2}],
                        }
                    ],
                },
            }
        )
    )

    data = sc.read_perf_data(raw)

    assert [task["task_id"] for task in data["tasks"]] == [7]
    assert data["scheduler_task_producer"] == "aicore"
    assert data["tasks"][0]["dispatch_time_us"] == pytest.approx(0.025)
    assert data["tasks"][0]["finish_time_us"] == pytest.approx(0.095)
    assert data["scheduler_streams"][0]["producer"] == "aicore"
    assert data["scheduler_records"][0][0]["claim_retries"] == 2
    assert data["scheduler_records"][0][1]["task_id"] is None
    assert data["aicpu_lifecycle_records"][0]["register_release_time_us"] == pytest.approx(0.007)

    trace_path = tmp_path / "merged_swimlane.json"
    sc.generate_chrome_trace_json(
        data["tasks"],
        str(trace_path),
        scheduler_phases=data["scheduler_records"],
        scheduler_streams=data["scheduler_streams"],
        aicpu_lifecycle_records=data["aicpu_lifecycle_records"],
    )
    events = json.loads(trace_path.read_text())["traceEvents"]
    assert any(
        event.get("name") == "process_name" and event.get("args", {}).get("name") == "AICore Scheduler"
        for event in events
    )
    assert any(event.get("cat") == "scheduler" and event.get("name") == "idle(0)" for event in events)
    assert any(
        event.get("name") == "process_name" and event.get("args", {}).get("name") == "AICPU Lifecycle"
        for event in events
    )
    assert any(event.get("cat") == "aicpu_lifecycle" and event.get("name") == "bootstrap_wait" for event in events)


def test_level_two_rejects_missing_scheduler_task_timing(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 2,
                "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 1, "core_types": ["aiv"]},
                "aicore_tasks": [[0, 7, 7, 120, 180, 10]],
            }
        )
    )

    with pytest.raises(ValueError, match="level 2 requires Scheduler task timing for every AICore task"):
        sc.read_perf_data(raw)


@pytest.mark.parametrize("producer", ["aicpu", "aicore"])
def test_level_two_accepts_task_timing_from_either_scheduler_producer(tmp_path, producer):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 2,
                "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 1, "core_types": ["aiv"]},
                "aicore_tasks": [[0, 7, 7, 120, 180, 10]],
                "scheduler_tasks": {
                    "schema_version": 1,
                    "producer": producer,
                    "records": [[0, 7, 115, 185]],
                },
            }
        )
    )

    data = sc.read_perf_data(raw)

    assert data["scheduler_task_producer"] == producer
    assert data["tasks"][0]["dispatch_time_us"] == pytest.approx(0.005)
    assert data["tasks"][0]["finish_time_us"] == pytest.approx(0.075)


@pytest.mark.parametrize("dispatch_cycles,finish_cycles", [(125, 185), (115, 175)])
def test_level_two_accepts_cross_producer_clock_skew(tmp_path, dispatch_cycles, finish_cycles):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 2,
                "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 1, "core_types": ["aiv"]},
                "aicore_tasks": [[0, 7, 7, 120, 180, 10]],
                "scheduler_tasks": {
                    "schema_version": 1,
                    "producer": "aicpu",
                    "records": [[0, 7, dispatch_cycles, finish_cycles]],
                },
            }
        )
    )

    data = sc.read_perf_data(raw)

    assert data["tasks"][0]["task_id"] == 7


@pytest.mark.parametrize(
    "aicore_timing,scheduler_timing,error",
    [
        ((180, 120, 10), (115, 185), "expected 0 < start_cycles <= end_cycles"),
        ((120, 180, 120), (115, 185), "expected 0 <= receive_to_start_cycles < start_cycles"),
        ((120, 180, 10), (185, 115), "expected 0 < dispatch_cycles <= finish_cycles"),
    ],
)
def test_level_two_rejects_invalid_same_producer_timing(tmp_path, aicore_timing, scheduler_timing, error):
    start_cycles, end_cycles, receive_to_start_cycles = aicore_timing
    dispatch_cycles, finish_cycles = scheduler_timing
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 2,
                "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 1, "core_types": ["aiv"]},
                "aicore_tasks": [[0, 7, 7, start_cycles, end_cycles, receive_to_start_cycles]],
                "scheduler_tasks": {
                    "schema_version": 1,
                    "producer": "aicpu",
                    "records": [[0, 7, dispatch_cycles, finish_cycles]],
                },
            }
        )
    )

    with pytest.raises(ValueError, match=error):
        sc.read_perf_data(raw)


def test_level_one_accepts_aicore_only_timing(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 1,
                "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 1, "core_types": ["aiv"]},
                "aicore_tasks": [[0, 7, 7, 120, 180, 10]],
            }
        )
    )

    data = sc.read_perf_data(raw)

    assert [task["task_id"] for task in data["tasks"]] == [7]
    assert "dispatch_time_us" not in data["tasks"][0]
    assert "finish_time_us" not in data["tasks"][0]
    assert "scheduler_task_producer" not in data


def test_level_one_skips_overhead_counters_without_scheduler_timing(tmp_path):
    trace_path = tmp_path / "merged_swimlane.json"
    sc.generate_chrome_trace_json(
        [
            {
                "task_id": 7,
                "func_id": -1,
                "core_id": 0,
                "core_type": "aiv",
                "start_time_us": 1.0,
                "end_time_us": 2.0,
                "duration_us": 1.0,
                "receive_time_us": 0.5,
                "local_setup_us": 0.5,
            }
        ],
        str(trace_path),
        deps_edges={7: [8]},
        emit_overhead=True,
    )

    events = json.loads(trace_path.read_text())["traceEvents"]
    assert not any(event.get("cat") == "overhead" for event in events)


@pytest.mark.parametrize(
    ("scheduler_tasks", "error"),
    [
        ({"schema_version": 2, "producer": "aicore", "records": []}, "schema_version"),
        ({"schema_version": 1, "producer": "host", "records": []}, "producer"),
        ({"schema_version": 1, "producer": "aicore", "records": [[0, 1, 2]]}, "four-column"),
    ],
)
def test_scheduler_tasks_reject_schema_drift(tmp_path, scheduler_tasks, error):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 2,
                "metadata": {"clock_freq_hz": 1_000_000_000},
                "aicore_tasks": [],
                "scheduler_tasks": scheduler_tasks,
            }
        )
    )

    with pytest.raises(ValueError, match=error):
        sc.read_perf_data(raw)


def test_scheduler_tasks_reject_ambiguous_legacy_stream(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 2,
                "metadata": {"clock_freq_hz": 1_000_000_000},
                "aicore_tasks": [],
                "scheduler_tasks": {"schema_version": 1, "producer": "aicore", "records": []},
                "aicpu_tasks": [],
            }
        )
    )

    with pytest.raises(ValueError, match="both scheduler_tasks and legacy aicpu_tasks"):
        sc.read_perf_data(raw)


def test_scheduler_records_reject_schema_drift(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 3,
                "metadata": {"clock_freq_hz": 1_000_000_000},
                "scheduler_records": {
                    "schema_version": 1,
                    "streams": [{"records": [{"kind": "idle"}], "metrics": []}],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="must contain exactly"):
        sc.read_perf_data(raw)


def test_scheduler_metrics_cannot_overwrite_fixed_record_fields(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 3,
                "metadata": {"clock_freq_hz": 1_000_000_000},
                "scheduler_records": {
                    "schema_version": 1,
                    "streams": [
                        {
                            "records": [
                                {
                                    "start_cycles": 10,
                                    "end_cycles": 20,
                                    "loop_iter": 0,
                                    "kind": "idle",
                                    "tasks_processed": 0,
                                    "task_id": None,
                                }
                            ],
                            "metrics": [{"record_index": 0, "start_cycles": 30}],
                        }
                    ],
                },
            }
        )
    )

    with pytest.raises(ValueError, match="metric overwrites fixed record fields"):
        sc.read_perf_data(raw)


def test_lifecycle_interval_can_start_at_relative_time_origin(tmp_path):
    trace_path = tmp_path / "merged_swimlane.json"
    sc.generate_chrome_trace_json(
        [],
        str(trace_path),
        aicpu_lifecycle_records=[
            {
                "worker_id": 0,
                "aicpu_thread_id": 1,
                "handshake_observed_time_us": 0.0,
                "handshake_partition_complete_time_us": 2.0,
            }
        ],
    )

    events = json.loads(trace_path.read_text())["traceEvents"]
    interval = next(event for event in events if event.get("name") == "handshake_partition")
    assert interval["ts"] == 0.0
    assert interval["dur"] == 2.0


def test_lifecycle_register_release_can_be_at_relative_time_origin(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 1,
                "metadata": {"clock_freq_hz": 1_000_000_000, "num_cores": 1, "core_types": ["aiv"]},
                "aicore_tasks": [[0, 7, 7, 120, 180, 10]],
                "aicpu_lifecycle_records": [{"worker_id": 0, "register_release_cycles": 90}],
            }
        )
    )

    data = sc.read_perf_data(raw)
    assert data["aicpu_lifecycle_records"][0]["register_release_time_us"] == 0.0

    trace_path = tmp_path / "merged_swimlane.json"
    sc.generate_chrome_trace_json(
        data["tasks"],
        str(trace_path),
        aicpu_lifecycle_records=data["aicpu_lifecycle_records"],
    )

    events = json.loads(trace_path.read_text())["traceEvents"]
    register_release = next(event for event in events if event.get("name") == "register_release")
    assert register_release["ts"] == 0.0


def test_lifecycle_omits_missing_register_release(tmp_path):
    trace_path = tmp_path / "merged_swimlane.json"
    sc.generate_chrome_trace_json(
        [],
        str(trace_path),
        aicpu_lifecycle_records=[{"worker_id": 0}],
    )

    events = json.loads(trace_path.read_text())["traceEvents"]
    assert not any(event.get("name") == "register_release" for event in events)


def test_host_capture_is_complete_when_the_pool_holds_more_than_the_submit_projection(tmp_path):
    """A pool record count above the projected one is normal, not incomplete.

    The producer records every timed host operation — bind segments and the
    sub-operations of a submit — while this file carries only the ones that
    submit a task. Completeness therefore compares `expected_records` (the pass's
    task count) against the projection, and `pool_records` is context.
    """
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 4,
                "metadata": {
                    "clock_freq_hz": 1_000_000,
                    "num_cores": 1,
                    "core_types": ["aiv"],
                    "core_to_thread": [0],
                    "orchestrator_source": "host",
                    "orchestrator_clock_domain": "host_monotonic_ns",
                    "host_orchestration_origin_ns": 1_000,
                    "timeline_relation": "host_orchestration_precedes_device",
                    "host_capture": {
                        "status": "complete",
                        "expected_records": 2,
                        "recorded_records": 2,
                        "pool_records": 339,
                        "dropped_records": 0,
                        "error": None,
                    },
                },
                "aicore_tasks": [[0, 7, 1, 100, 110, 0]],
                "aicpu_tasks": [[0, 1, 90, 120]],
                "host_orchestrator_phases": [
                    [
                        {"submit_idx": 0, "task_id": 7, "start_host_ns": 1_000, "end_host_ns": 3_000},
                        {"submit_idx": 1, "task_id": 8, "start_host_ns": 3_000, "end_host_ns": 4_000},
                    ]
                ],
            }
        )
    )

    data = sc.read_perf_data(raw)

    assert data["timeline_metadata"]["host_records_complete"] is True
    assert data["timeline_metadata"]["host_capture"]["status"] == "complete"
    assert data["timeline_metadata"]["host_capture"]["pool_records"] == 339
    assert "converter_validation_errors" not in data["timeline_metadata"]["host_capture"]


def test_host_and_device_timestamps_use_calibrated_clock_alignment(tmp_path):
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(
        json.dumps(
            {
                "chip_swimlane_level": 4,
                "metadata": {
                    "clock_freq_hz": 1_000_000_000,
                    "num_cores": 1,
                    "core_types": ["aiv"],
                    "core_to_thread": [0],
                    "orchestrator_source": "host",
                    "orchestrator_clock_domain": "host_monotonic_ns",
                    "host_orchestration_origin_ns": 1_500,
                    "timeline_relation": "host_orchestration_precedes_device",
                    "host_capture": {
                        "status": "complete",
                        "expected_records": 1,
                        "recorded_records": 1,
                        "dropped_records": 0,
                        "error": None,
                    },
                    "clock_anchors": {
                        "device_timestamp_unit": "syscnt_cycles",
                        "samples": [
                            {
                                "position": "pre_host_orchestration",
                                "sample_idx": 0,
                                "host_before_ns": 990,
                                "device_cycles": 100,
                                "host_after_ns": 1_010,
                                "error": None,
                            },
                            {
                                "position": "post_device_execution",
                                "sample_idx": 0,
                                "host_before_ns": 5_080,
                                "device_cycles": 4_100,
                                "host_after_ns": 5_120,
                                "error": None,
                            },
                        ],
                    },
                },
                "aicore_tasks": [[0, 7, 1, 2_100, 2_200, 0]],
                "aicpu_tasks": [[0, 1, 2_000, 2_300]],
                "aicpu_scheduler_phases": [
                    [{"kind": "dispatch", "start_cycles": 1_900, "end_cycles": 1_950, "tasks_processed": 1}]
                ],
                "host_orchestrator_phases": [
                    [{"submit_idx": 0, "task_id": 7, "start_host_ns": 1_500, "end_host_ns": 1_800}]
                ],
            }
        )
    )

    data = sc.read_perf_data(raw)

    assert data["aicpu_orchestrator_phases"][0][0]["start_time_us"] == 0.0
    assert data["aicpu_orchestrator_phases"][0][0]["end_time_us"] == 0.3
    assert data["tasks"][0]["dispatch_time_us"] == 1.447
    assert data["tasks"][0]["start_time_us"] == 1.55
    assert data["timeline_metadata"] == {
        "layout": "clock_aligned",
        "trace_status": "complete",
        "relation": "host_orchestration_precedes_device",
        "clock_alignment": {
            "status": "calibrated",
            "method": "nominal_frequency_offset_interp_v1",
            "anchor_uncertainty_ns": 20,
            "host_timestamp_quantization_ns": 0,
            "max_uncertainty_ns": 20,
            "selected_sample_idx": {
                "pre_host_orchestration": 0,
                "post_device_execution": 0,
            },
            "anchor_group_duration_ns": {
                "pre_host_orchestration": 20,
                "post_device_execution": 40,
            },
        },
        "host_capture": {
            "status": "complete",
            "expected_records": 1,
            "recorded_records": 1,
            "dropped_records": 0,
            "error": None,
        },
        "host_records_complete": True,
        "cross_domain_latency_available": True,
        "source_timeline_origin_ns": 1_500,
        "timeline_origin_ns": 1_500,
    }


def test_dropped_host_capture_is_visible_and_disables_cross_domain_flows(tmp_path):
    for case_name, host_phases, capture_status, dropped_records, capture_error in (
        (
            "partial",
            [[{"submit_idx": 0, "task_id": 7, "start_host_ns": 1_500, "end_host_ns": 1_800}]],
            "dropped",
            1,
            "record_allocation_failed",
        ),
        ("all_dropped", [], "dropped", 1, "record_allocation_failed"),
        ("silent_missing", [], "complete", 0, None),
    ):
        raw = tmp_path / f"{case_name}.json"
        raw.write_text(
            json.dumps(
                {
                    "chip_swimlane_level": 4,
                    "metadata": {
                        "clock_freq_hz": 1_000_000_000,
                        "num_cores": 1,
                        "core_types": ["aiv"],
                        "core_to_thread": [0],
                        "orchestrator_source": "host",
                        "host_orchestration_origin_ns": 1_500 if host_phases else 0,
                        "timeline_relation": "host_orchestration_precedes_device",
                        "host_timestamp_quantization_ns": 0,
                        "host_capture": {
                            "status": capture_status,
                            "expected_records": sum(len(records) for records in host_phases) + 1,
                            "recorded_records": sum(len(records) for records in host_phases),
                            "dropped_records": dropped_records,
                            "error": capture_error,
                        },
                        "clock_anchors": {
                            "device_timestamp_unit": "syscnt_cycles",
                            "samples": [
                                {
                                    "position": "pre_host_orchestration",
                                    "sample_idx": 0,
                                    "host_before_ns": 990,
                                    "device_cycles": 100,
                                    "host_after_ns": 1_010,
                                    "error": None,
                                },
                                {
                                    "position": "post_device_execution",
                                    "sample_idx": 0,
                                    "host_before_ns": 5_080,
                                    "device_cycles": 4_100,
                                    "host_after_ns": 5_120,
                                    "error": None,
                                },
                            ],
                        },
                    },
                    "aicore_tasks": [[0, 7, 1, 2_100, 2_200, 0]],
                    "aicpu_tasks": [[0, 1, 2_000, 2_300]],
                    "aicpu_scheduler_phases": [[{"kind": "dispatch", "start_cycles": 1_900, "end_cycles": 1_950}]],
                    "host_orchestrator_phases": host_phases,
                }
            )
        )

        data = sc.read_perf_data(raw)

        assert data["timeline_metadata"]["layout"] == "clock_aligned"
        assert data["timeline_metadata"]["trace_status"] == "partial"
        assert data["timeline_metadata"]["clock_alignment"]["status"] == "calibrated"
        assert data["timeline_metadata"]["host_capture"]["status"] == capture_status
        assert data["timeline_metadata"]["host_records_complete"] is False
        assert data["timeline_metadata"]["cross_domain_latency_available"] is False
        assert data["timeline_metadata"].get("host_records_missing", False) is (not host_phases)
        if case_name == "silent_missing":
            assert data["timeline_metadata"]["host_capture"]["converter_validation_errors"] == [
                "expected_record_count_mismatch"
            ]

        trace_path = tmp_path / f"{case_name}_trace.json"
        sc.generate_chrome_trace_json(
            data["tasks"],
            str(trace_path),
            scheduler_phases=data["aicpu_scheduler_phases"],
            orchestrator_phases=data.get("aicpu_orchestrator_phases"),
            orchestrator_source=data["orchestrator_source"],
            timeline_metadata=data["timeline_metadata"],
            core_to_thread=data["core_to_thread"],
        )
        trace = json.loads(trace_path.read_text())
        assert not any(
            event.get("cat") == "flow" and event.get("name") == "submit→dispatch" for event in trace["traceEvents"]
        )


def test_graph_prepare_phases_create_graph_execution_envelopes(tmp_path):
    out = tmp_path / "trace.json"
    outer_a = 3
    outer_b = 7
    task_a0 = (1 << 32) | (outer_a << 10)
    task_a1 = (1 << 32) | ((outer_a << 10) | 1)
    task_b0 = (1 << 32) | (outer_b << 10)
    scheduler_phases = [
        [
            {
                "phase": "graph_prepare",
                "task_id": outer_a,
                "start_time_us": 1.0,
                "end_time_us": 1.4,
                "tasks_processed": 1,
            },
            {
                "phase": "graph_prepare",
                "task_id": outer_a,
                "start_time_us": 1.5,
                "end_time_us": 1.8,
                "tasks_processed": 1,
            },
            {
                "phase": "graph_prepare",
                "task_id": outer_b,
                "start_time_us": 5.0,
                "end_time_us": 5.2,
                "tasks_processed": 1,
            },
        ]
    ]
    tasks = [
        _task_row(task_a0, 0, dispatch=2.0, start=2.2, end=3.0, receive=2.1),
        _task_row(task_a1, 1, dispatch=3.2, start=3.4, end=4.0, receive=3.3),
        _task_row(task_b0, 0, dispatch=5.3, start=5.5, end=6.0, receive=5.4),
    ]

    sc.generate_chrome_trace_json(tasks, str(out), scheduler_phases=scheduler_phases, core_to_thread=[0, 0])

    with open(out) as f:
        events = json.load(f)["traceEvents"]
    assert any(
        event.get("ph") == "M" and event.get("pid") == 5 and event.get("args", {}).get("name") == "Graph Execution"
        for event in events
    )
    graph_events = [event for event in events if event.get("cat") == "graph_execution"]
    assert [event["args"]["outer_task_id"] for event in graph_events] == [outer_a, outer_b]
    assert graph_events[0]["args"]["visible_in_graph_task_count"] == 2
    assert graph_events[0]["args"]["prepare_slice_count"] == 2
    assert graph_events[0]["ts"] == 1.0
    assert graph_events[0]["dur"] == 4.0
    assert (
        sum(event.get("cat") == "scheduler" and event.get("name", "").startswith("graph_prepare(") for event in events)
        == 3
    )


def test_spmd_pred_routes_dependency_to_earliest_slice(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [
        _task_row(pred_id, core_id, dispatch=10.0 + core_id, start=11.0 + core_id, end=20.0 + core_id)
        for core_id in range(4)
    ]
    tasks.append(_task_row(succ_id, 10))
    deps_edges = {pred_id: [succ_id]}
    deps_block_map = {pred_id: 4, succ_id: 1}

    out = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 1
    assert _count_dependency_flow_starts(out, pid=3) == 1
    assert not _has_spmd_block_level_track(out)
    flow = _first_worker_dependency_flow(out)
    assert flow[0]["output_task_count"] == 4
    assert flow[0]["input_task_count"] == 1
    assert flow[0]["tid"] == _core_tid(0)
    assert flow[0]["ts"] == tasks[0]["receive_time_us"]
    sched_flow = _first_scheduler_dependency_flow(out)
    assert sched_flow[0]["output_task_count"] == 4
    assert sched_flow[0]["input_task_count"] == 1
    assert sched_flow[0]["ts"] == tasks[0]["dispatch_time_us"]
    assert sched_flow[1]["ts"] == tasks[4]["dispatch_time_us"]


def test_spmd_succ_routes_dependency_to_earliest_slice(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [
        _task_row(pred_id, 0, dispatch=0.0, start=-0.5, end=-0.1, receive=-0.6),
        _task_row(succ_id, 26, func_id=1, dispatch=0.2, start=1.44, end=3.02, receive=0.0),
        _task_row(succ_id, 33, func_id=1, dispatch=0.1, start=1.14, end=2.92, receive=0.06),
    ]
    deps_edges = {pred_id: [succ_id]}
    deps_block_map = {pred_id: 1, succ_id: 2}

    out = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 1
    assert _count_dependency_flow_starts(out, pid=3) == 1
    worker_flow = _first_worker_dependency_flow(out)
    scheduler_flow = _first_scheduler_dependency_flow(out)
    assert worker_flow[0]["output_task_count"] == 1
    assert worker_flow[0]["input_task_count"] == 2
    assert worker_flow[1]["tid"] == _core_tid(26)
    assert worker_flow[1]["ts"] == 0.0
    assert scheduler_flow[1]["tid"] == _aicpu_tid(33)
    assert scheduler_flow[1]["ts"] == 0.1


def test_hb_violation_flows_render_between_bar_starts(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [
        _task_row(pred_id, 0, dispatch=10.0, start=11.0, end=20.0, receive=10.5),
        _task_row(succ_id, 1, dispatch=15.0, start=22.0, end=30.0, receive=19.0),
    ]

    out = _generate_trace(tasks, {pred_id: [succ_id]}, {pred_id: 1, succ_id: 1}, tmp_path)

    worker_flow = _first_worker_dependency_flow(out)
    assert [event["name"] for event in worker_flow] == ["hb_violation", "hb_violation"]
    assert [event["ts"] for event in worker_flow] == [10.5, 19.0]

    scheduler_flow = _first_scheduler_dependency_flow(out)
    assert [event["name"] for event in scheduler_flow] == ["hb_violation", "hb_violation"]
    assert [event["ts"] for event in scheduler_flow] == [10.0, 15.0]


def test_spmd_to_spmd_one_edge_on_earliest_slice(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [_task_row(pred_id, core_id, dispatch=10.0 + core_id) for core_id in range(4)]
    tasks.extend(_task_row(succ_id, core_id, dispatch=30.0 + core_id) for core_id in range(4))
    deps_edges = {pred_id: [succ_id]}
    deps_block_map = {pred_id: 4, succ_id: 4}

    out = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 1
    assert not _has_spmd_block_level_track(out)
    flow = _first_worker_dependency_flow(out)
    assert flow[0]["output_task_count"] == 4
    assert flow[0]["input_task_count"] == 4


def test_spmd_mix_to_mix_uses_anchor_cartesian_product(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [
        _task_row(pred_id, 0, "aic", func_id=1, dispatch=10.0, start=11.0, end=20.0),
        _task_row(pred_id, 1, "aiv", func_id=2, dispatch=10.1, start=11.1, end=20.1),
        _task_row(pred_id, 3, "aiv", func_id=2, dispatch=10.3, start=11.3, end=20.3),
        _task_row(succ_id, 4, "aic", func_id=1, dispatch=30.0, start=31.0, end=40.0, receive=30.5),
        _task_row(succ_id, 5, "aiv", func_id=2, dispatch=30.1, start=31.1, end=40.1, receive=30.6),
        _task_row(succ_id, 7, "aiv", func_id=2, dispatch=30.3, start=31.3, end=40.3, receive=30.8),
    ]
    deps_edges = {pred_id: [succ_id]}
    deps_block_map = {pred_id: 3, succ_id: 3}

    out = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 4
    finish_tids = _worker_flow_finish_tids(out)
    assert finish_tids == {_core_tid(4), _core_tid(5)}


def test_spmd_aiv_only_pred_connects_to_mix_spmd_succ_both_anchors(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [
        _task_row(pred_id, 24, "aiv", dispatch=10.0, start=11.0, end=20.0),
        _task_row(pred_id, 30, "aiv", dispatch=10.3, start=11.3, end=20.3),
        _task_row(succ_id, 0, "aic", func_id=1, dispatch=30.0, start=31.0, end=40.0, receive=30.5),
        _task_row(succ_id, 24, "aiv", func_id=2, dispatch=30.1, start=31.1, end=40.1, receive=30.6),
        _task_row(succ_id, 27, "aiv", func_id=2, dispatch=30.3, start=31.3, end=40.3, receive=30.8),
    ]
    deps_edges = {pred_id: [succ_id]}
    deps_block_map = {pred_id: 16, succ_id: 24}

    out = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 2
    assert _worker_flow_finish_tids(out) == {_core_tid(0), _core_tid(24)}


def test_mix_keeps_worker_view_dependency_flows(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [
        _task_row(pred_id, 0, "aic", dispatch=10.0, start=11.0, end=20.0, receive=10.5),
        _task_row(pred_id, 1, "aiv", dispatch=10.1, start=11.1, end=20.1, receive=10.6),
        _task_row(pred_id, 2, "aiv", dispatch=10.2, start=11.2, end=20.2, receive=10.7),
        _task_row(succ_id, 3, "aic", dispatch=30.0, start=31.0, end=40.0, receive=30.5),
        _task_row(succ_id, 4, "aiv", dispatch=30.1, start=31.1, end=40.1, receive=30.6),
        _task_row(succ_id, 5, "aiv", dispatch=30.2, start=31.2, end=40.2, receive=30.7),
    ]
    deps_edges = {pred_id: [succ_id]}
    deps_block_map = {pred_id: 1, succ_id: 1}

    out = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 9
    assert not _has_spmd_block_level_track(out)
    with open(out) as f:
        mix_flows = [
            e
            for e in json.load(f)["traceEvents"]
            if e.get("cat") == "flow"
            and e.get("name") in ("dependency", "hb_violation")
            and e.get("ph") == "s"
            and e.get("pid") == 4
        ]
    assert all(e["output_task_count"] == 1 and e["input_task_count"] == 1 for e in mix_flows)


def test_spmd_fallback_without_block_map(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [_task_row(pred_id, core_id, dispatch=10.0 + core_id) for core_id in range(3)]
    tasks.append(_task_row(succ_id, 10))
    deps_edges = {pred_id: [succ_id]}

    out = _generate_trace(tasks, deps_edges, None, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 1
    flow = _first_worker_dependency_flow(out)
    assert flow[0]["tid"] == _core_tid(0)


def test_worker_flow_anchor_rows_picks_earliest_visible_slice_per_func_id():
    task_map = {
        1: [
            _task_row(1, 5, "aiv", func_id=2, start=12.0, receive=10.0),
            _task_row(1, 0, "aiv", func_id=2, start=11.0, receive=10.0),
            _task_row(1, 2, "aic", func_id=1, start=13.0, receive=12.0),
            _task_row(1, 7, "aic", func_id=1, start=12.0, receive=11.0),
        ]
    }
    rows = sc._worker_flow_anchor_rows(1, task_map, {1})
    assert len(rows) == 2
    by_func = {r["func_id"]: r["core_id"] for r in rows}
    assert by_func == {1: 7, 2: 0}


def test_identify_spmd_task_ids_respects_authoritative_block_num_one():
    task_map = {
        1: [_task_row(1, 0), _task_row(1, 1), _task_row(1, 2)],
        2: [_task_row(2, 0), _task_row(2, 1)],
    }
    deps_block_map = {1: 1, 2: 4}
    spmd_ids = sc._identify_spmd_task_ids(task_map, deps_block_map)
    assert spmd_ids == {2}


def test_spmd_task_display_name_suffix():
    assert sc._task_display_name(16, {"16": "fa_fused_aic"}, "r2t18", spmd=True) == "fa_fused_aic_spmd(r2t18)"
    assert sc._task_display_name(16, {"16": "fa_fused_aic"}, "r2t18", spmd=False) == "fa_fused_aic(r2t18)"
    assert sc._task_display_name(-1, {}, "r2t18", spmd=True) == "task_spmd(r2t18)"
    assert sc._task_display_name(0, {"0": "spmd_write_aiv"}, "t0", spmd=True) == "spmd_write_aiv(t0)"
    assert sc._task_display_name(0, {"0": "SPMDKernel"}, "t0", spmd=True) == "SPMDKernel(t0)"


def test_spmd_cross_type_single_anchor_pair(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [_task_row(pred_id, core_id, "aic", dispatch=10.0 + core_id) for core_id in range(1, 9, 3)]
    tasks.extend(_task_row(succ_id, core_id, "aiv", dispatch=30.0 + core_id) for core_id in range(24, 40, 2))
    deps_edges = {pred_id: [succ_id]}
    deps_block_map = {pred_id: 8, succ_id: 16}

    out = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 1


def _complete_flows(trace_path):
    with open(trace_path) as f:
        events = json.load(f)["traceEvents"]
    return [e for e in events if e.get("cat") == "flow" and e.get("name") == "complete"]


def _aicpu_tid(core_id):
    # Non-overlapping single-task-per-core cases keep the base Scheduler View lane.
    return 10000 + core_id * 10


def test_complete_flow_uses_independent_view_anchors(tmp_path):
    task_id = 100
    tasks = [
        _task_row(task_id, 26, dispatch=0.2, start=1.44, end=3.02, receive=0.0),
        _task_row(task_id, 33, dispatch=0.1, start=1.14, end=2.92, receive=0.06),
    ]
    deps_edges = {}
    deps_block_map = {task_id: 2}
    scheduler_phases = [[{"phase": "complete", "start_time_us": 3.5, "end_time_us": 4.5}]]
    core_to_thread = [0] * 34

    out = tmp_path / "trace.json"
    sc.generate_chrome_trace_json(
        tasks,
        str(out),
        deps_edges=deps_edges,
        deps_block_map=deps_block_map,
        scheduler_phases=scheduler_phases,
        core_to_thread=core_to_thread,
    )

    flows = _complete_flows(out)
    starts_p4 = [e for e in flows if e.get("ph") == "s" and e.get("pid") == 4]
    starts_p3 = [e for e in flows if e.get("ph") == "s" and e.get("pid") == 3]
    assert len(starts_p4) == 1
    assert len(starts_p3) == 1

    p4 = starts_p4[0]
    p3 = starts_p3[0]
    assert p4["tid"] == _core_tid(26)
    assert p4["ts"] == tasks[0]["end_time_us"] - 0.01
    assert p3["tid"] == _aicpu_tid(33)
    assert p3["ts"] == tasks[1]["finish_time_us"] - 0.01

    finishes = [e for e in flows if e.get("ph") == "f"]
    assert len(finishes) == 2
    assert len({(e["pid"], e["tid"], e["ts"]) for e in finishes}) == 1


def test_complete_phase_preserves_runtime_fin_count(tmp_path):
    task_id = 101
    tasks = [
        _task_row(task_id, 0, start=1.0, end=2.0),
        _task_row(task_id, 1, start=1.5, end=2.5),
    ]
    scheduler_phases = [
        [
            {
                "phase": "complete",
                "start_time_us": 2.5,
                "end_time_us": 3.5,
                # A5/a2a3 runtime count: two AICore FINs, one of which may
                # be a non-final SPMD sub-block retire.
                "tasks_processed": 2,
            }
        ]
    ]

    out = tmp_path / "trace.json"
    sc.generate_chrome_trace_json(
        tasks,
        str(out),
        scheduler_phases=scheduler_phases,
        core_to_thread=[0, 0],
        deps_edges={},
        deps_block_map={task_id: 2},
    )

    with out.open() as f:
        events = json.load(f)["traceEvents"]
    complete = next(e for e in events if e.get("cat") == "scheduler" and e.get("name") == "complete(2)")
    assert complete["args"]["finishes_processed"] == 2
    assert complete["args"]["finish_rows_attributed"] == 2


def test_hbg_resolution_thread_uses_one_lane_and_exports_queue_depths(tmp_path):
    out = tmp_path / "trace.json"
    scheduler_phases = [
        [],
        [
            {
                "phase": "resolve_standalone",
                "start_time_us": 1.0,
                "end_time_us": 2.0,
                "tasks_processed": 1,
                "shared_at_start": [1, 2, 3],
                "shared_at_end": [4, 5, 6],
            },
            {
                "phase": "async_poll",
                "start_time_us": 2.0,
                "end_time_us": 3.0,
                "shared_at_start": [4, 5, 6],
                "shared_at_end": [7, 8, 9],
            },
            {
                "phase": "dummy",
                "start_time_us": 3.0,
                "end_time_us": 4.0,
                "shared_at_start": [7, 8, 9],
                "shared_at_end": [10, 11, 12],
            },
        ],
    ]

    sc.generate_chrome_trace_json([], str(out), scheduler_phases=scheduler_phases, core_to_thread=[0])

    events = json.loads(out.read_text())["traceEvents"]
    p_phases = [event for event in events if event.get("cat") == "scheduler" and event.get("tid") // 10 == 3001]
    assert [(event["name"], event["tid"]) for event in p_phases] == [
        ("resolve(1)", 30010),
        ("async_poll(0)", 30010),
        ("dummy(0)", 30010),
    ]
    queue_samples = [event for event in events if event.get("name") == "shared_ready_queue"]
    assert [(event["ts"], event["args"]) for event in queue_samples] == [
        (2.0, {"AIC": 4, "AIV": 5, "MIX": 6}),
        (3.0, {"AIC": 7, "AIV": 8, "MIX": 9}),
        (4.0, {"AIC": 10, "AIV": 11, "MIX": 12}),
    ]


def test_tmr_nested_resolve_stays_on_scheduler_sublane(tmp_path):
    out = tmp_path / "trace.json"
    scheduler_phases = [
        [
            {"phase": "complete", "start_time_us": 1.0, "end_time_us": 4.0},
            {"phase": "resolve", "start_time_us": 2.0, "end_time_us": 3.0},
        ]
    ]

    sc.generate_chrome_trace_json([], str(out), scheduler_phases=scheduler_phases, core_to_thread=[0])

    events = json.loads(out.read_text())["traceEvents"]
    complete = next(event for event in events if event.get("name") == "complete(0)")
    resolve = next(event for event in events if event.get("name") == "resolve(0)")
    assert complete["tid"] == 30000
    assert resolve["tid"] == 30001


def test_complete_flow_worker_view_only_without_scheduler_phases(tmp_path):
    # Without scheduler_phases the complete-flow block is skipped entirely:
    # neither view gets a complete arrow (regression guard on the gate).
    task_id = 100
    tasks = [_task_row(task_id, 0)]

    out = _generate_trace(tasks, {}, {task_id: 1}, tmp_path)
    assert _complete_flows(out) == []


def test_aicpu_worker_lanes_and_full_dummy_ids_follow_runtime_threads(tmp_path):
    out = tmp_path / "trace.json"
    dummy_r1t1 = (1 << 32) | 1
    dummy_r2t1 = (2 << 32) | 1
    alloc_r3t1 = (3 << 32) | 1
    scheduler_phases = [
        [],
        [{"phase": "dummy_task", "task_id": dummy_r1t1, "start_time_us": 1.0, "end_time_us": 1.0}],
        [{"phase": "dummy_task", "task_id": dummy_r2t1, "start_time_us": 2.0, "end_time_us": 2.0}],
        [],
    ]
    orchestrator_phases = [[{"phase": "orch_submit", "task_id": alloc_r3t1, "start_time_us": 3.0, "end_time_us": 4.0}]]

    sc.generate_chrome_trace_json(
        [],
        str(out),
        scheduler_phases=scheduler_phases,
        orchestrator_phases=orchestrator_phases,
        core_to_thread=[0, 1, 2],
        deps_edges={dummy_r1t1: [dummy_r2t1]},
        deps_kernel_map={dummy_r1t1: [-1, -1, -1], dummy_r2t1: [-1, -1, -1]},
    )

    with open(out) as f:
        events = json.load(f)["traceEvents"]
    aicpu_lanes = {
        event["tid"]: event["args"]["name"]
        for event in events
        if event.get("ph") == "M"
        and event.get("pid") == 4
        and event.get("args", {}).get("name", "").startswith("AICPU_")
    }
    assert aicpu_lanes == {
        19000: "AICPU_0",
        19001: "AICPU_1",
        19002: "AICPU_2",
        19003: "AICPU_3",
    }
    assert next(event for event in events if event.get("name") == "dummy(r1t1)")["tid"] == 19001
    assert next(event for event in events if event.get("name") == "dummy(r2t1)")["tid"] == 19002
    assert next(event for event in events if event.get("name") == "alloc(r3t1)")["tid"] == 19003

    flow = _first_worker_dependency_flow(out)
    assert [(event["ph"], event["tid"]) for event in flow] == [("s", 19001), ("f", 19002)]


def test_deps_dummy_without_runtime_record_is_not_rendered_as_alloc(tmp_path, capsys):
    out = tmp_path / "trace.json"
    dummy_task_id = (1 << 32) | 1

    sc.generate_chrome_trace_json(
        [],
        str(out),
        scheduler_phases=[[]],
        orchestrator_phases=[
            [{"phase": "orch_submit", "task_id": dummy_task_id, "start_time_us": 2.0, "end_time_us": 3.0}]
        ],
        deps_kernel_map={dummy_task_id: [-1, -1, -1]},
    )

    with open(out) as f:
        events = json.load(f)["traceEvents"]
    assert not any(event.get("name") == "alloc(r1t1)" for event in events)
    assert "dummy(r1t1) has no dummy_task scheduler record" in capsys.readouterr().err


def test_predicated_skip_uses_aicpu_worker_lane_and_dependency_anchor(tmp_path):
    out = tmp_path / "trace.json"
    skipped_task_id = (1 << 32) | 2
    consumer_task_id = (1 << 32) | 3
    scheduler_phases = [
        [{"phase": "predicated_skip", "task_id": skipped_task_id, "start_time_us": 2.0, "end_time_us": 2.0}]
    ]
    orchestrator_phases = [
        [{"phase": "orch_submit", "task_id": skipped_task_id, "start_time_us": 1.0, "end_time_us": 1.5}]
    ]

    sc.generate_chrome_trace_json(
        [_task_row(consumer_task_id, 0, dispatch=3.0, start=4.0, end=5.0, receive=3.5)],
        str(out),
        func_id_to_name={"21": "exp_gate_mm"},
        scheduler_phases=scheduler_phases,
        orchestrator_phases=orchestrator_phases,
        core_to_thread=[0],
        deps_edges={skipped_task_id: [consumer_task_id]},
        deps_kernel_map={skipped_task_id: [21, -1, -1]},
        deps_block_map={skipped_task_id: 2, consumer_task_id: 1},
    )

    with open(out) as f:
        events = json.load(f)["traceEvents"]
    marker = next(event for event in events if event.get("name") == "exp_gate_mm_spmd(r1t2)")
    assert marker["pid"] == 4
    assert marker["tid"] == 19000
    assert marker["dur"] == 0.02
    assert marker["args"] == {
        "loop_iter": 0,
        "task_id": skipped_task_id,
        "event-hint": "exp_gate_mm_spmd(r1t2)",
        "predicated_pass": False,
    }
    assert "cname" not in marker
    assert not any(event.get("name") == "alloc(r1t2)" for event in events)

    flow = _first_worker_dependency_flow(out)
    assert [(event["ph"], event["tid"]) for event in flow] == [("s", 19000), ("f", _core_tid(0))]


def test_predicated_skip_without_deps_is_not_rendered_as_alloc(tmp_path):
    out = tmp_path / "trace.json"
    skipped_task_id = (1 << 32) | 2

    sc.generate_chrome_trace_json(
        [],
        str(out),
        scheduler_phases=[
            [{"phase": "predicated_skip", "task_id": skipped_task_id, "start_time_us": 2.0, "end_time_us": 2.0}]
        ],
        orchestrator_phases=[
            [{"phase": "orch_submit", "task_id": skipped_task_id, "start_time_us": 1.0, "end_time_us": 1.5}]
        ],
        core_to_thread=[0],
    )

    with open(out) as f:
        events = json.load(f)["traceEvents"]
    marker = next(event for event in events if event.get("name") == "task(r1t2)")
    assert marker["pid"] == 4
    assert marker["tid"] == 19000
    assert marker["args"] == {
        "loop_iter": 0,
        "task_id": skipped_task_id,
        "event-hint": "task(r1t2)",
        "predicated_pass": False,
    }
    assert not any(event.get("name") == "alloc(r1t2)" for event in events)
