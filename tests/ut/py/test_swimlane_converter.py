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

from simpler_setup.tools import containment
from simpler_setup.tools import swimlane_converter as sc
from simpler_setup.tools.strace_timing import parse_spans, to_host_swimlane


def _containment_placement(document, *, runner_start_ns=1_000, runner_dur_ns=5_000, wall_ns=2_000, sched=(700, 100)):
    """Place a capture inside a synthetic Host window, the way the tools do.

    The `sched` span is what joins the two artifacts: the capture records the
    same window in absolute cycles, so the pair fixes the offset between the two
    device timelines without any clock anchor.
    """
    prefix = "[mono_ns=1000][T0x1][TIMING] emit_host_span: "
    head = "[STRACE] v=1 pid=42 tid=42 inv=1 hid=abc"
    lines = [
        f"{prefix}{head} depth=1 name=chip.run.runner_run ts={runner_start_ns} dur={runner_dur_ns} ",
        f"{prefix}{head} depth=2 name=chip.run.runner_run.device_wall ts=0 dur={wall_ns} clk=dev",
        f"{prefix}{head} depth=3 name=chip.run.runner_run.device_wall.sched ts={sched[0]} dur={sched[1]} clk=dev",
    ]
    (window,) = containment.host_windows(parse_spans(lines))
    return containment.place(window, containment.capture_windows(document))


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


def _write_l3_host_log(root, rank, *, host_shift_ns, sched_window_ns):
    """The Host log that brackets one Rank's device work.

    Containment reads the outer window out of this, so a Rank capture without
    one cannot be placed. The `sched` window differs per Rank, which is what
    lets the two artifacts be paired without either naming the other.
    """
    pid = 1000 + rank
    prefix = f"[mono_ns={host_shift_ns + 1_000}][T0x1][TIMING] emit_host_span: "
    lines = [
        f"{prefix}[STRACE] v=1 pid={pid} tid={pid} inv=1 hid=abc depth=1 "
        f"name=chip.run.runner_run ts={host_shift_ns + 1_000} dur=8000 ",
        f"{prefix}[STRACE] v=1 pid={pid} tid={pid} inv=1 hid=abc depth=2 "
        f"name=chip.run.runner_run.device_wall ts=0 dur=2000 clk=dev",
        f"{prefix}[STRACE] v=1 pid={pid} tid={pid} inv=1 hid=abc depth=3 "
        f"name=chip.run.runner_run.device_wall.sched ts=700 dur={sched_window_ns} clk=dev",
    ]
    (root / f"host.{pid}.log").write_text("\n".join(lines) + "\n")


def _write_l3_rank(root, rank, *, host_shift_ns, task_id, clock_domain="same-boot", dispatch="d0"):
    rank_dir = root / f"rank{rank}" / dispatch
    rank_dir.mkdir(parents=True)
    device_base = 100 + rank * 100_000
    # One scheduler phase per Rank, each a different length, so the Host log's
    # `sched` window identifies which Rank it brackets.
    sched_record_ns = 50 + rank * 100
    _write_l3_host_log(root, rank, host_shift_ns=host_shift_ns, sched_window_ns=sched_record_ns + 50)
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
        },
        "aicore_tasks": [[0, task_id, 1, device_base + 1_000, device_base + 1_100, 0]],
        "aicpu_tasks": [[0, 1, device_base + 900, device_base + 1_200]],
        "aicpu_scheduler_phases": [
            [
                {
                    "kind": "dispatch",
                    "start_cycles": device_base + 800,
                    "end_cycles": device_base + 800 + sched_record_ns,
                }
            ]
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
    # The axis starts at the earliest window any Rank can have begun in, and
    # each Rank is drawn at the earliest position its own window allows.
    assert trace["metadata"]["global_origin_ns"] == 1_000
    assert trace["metadata"]["host_clock_domain_id"] == "same-boot"
    assert trace["metadata"]["layout"] == "containment_spliced_multi_rank"
    # 8 us of window around a 2 us run wall, twice: either end of a cross-Rank
    # read carries its own Rank's 6 us.
    assert trace["metadata"]["cross_rank_uncertainty_ns"] == 12_000
    assert trace["metadata"]["dispatch_pairing"] == "local_capture_index"
    assert [item["placement"]["slack_ns"] for item in rank_metadata] == [6_000, 6_000]
    assert [item["placement"]["outer_pid"] for item in rank_metadata] == [1_000, 1_001]
    assert [item["host_pairing"]["source"] for item in rank_metadata] == ["device_window_fit"] * 2

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
    assert worker_events[7]["pid"] == 104
    assert worker_events[7]["ts"] == 0.925
    assert worker_events[8]["pid"] == 204
    assert worker_events[8]["ts"] == 10.925
    # Every drawn slice carries the bound its Rank was placed under, so a
    # displacement between two Ranks can be read against it without leaving
    # the slice. The full record stays in `metadata.ranks[]`.
    assert [worker_events[task]["args"]["slack_ns"] for task in (7, 8)] == [6_000, 6_000]
    assert {
        event["args"]["slack_ns"]
        for event in trace["traceEvents"]
        if event.get("ph") == "X" and event["pid"] >= sc._RANK_PID_STRIDE
    } == {6_000}


def test_l3_directory_merge_groups_the_two_clock_domains_into_blocks(tmp_path):
    """Everything read off the Host log first, then the Chip captures.

    Interleaving them one Rank at a time puts a `CLOCK_MONOTONIC` window
    between two device timelines, which is the comparison the merge exists to
    make and the one a reader should not have to scroll past.
    """
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    output = tmp_path / "l3.json"
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0", "-o", str(output)])

    sc._generate_l3_trace(args, root)

    events = json.loads(output.read_text())["traceEvents"]
    names = {event["pid"]: event["args"]["name"] for event in events if event.get("name") == "process_name"}
    # Perfetto orders process groups by pid and ignores `process_sort_index`,
    # so the pid order is the layout. Assert on it, not on the sort index.
    lanes = [names[pid] for pid in sorted(names)]
    assert lanes[:6] == [
        "rank0 / Host",
        "rank0 / Device phases (placed)",
        "rank0 / Placement Bound",
        "rank1 / Host",
        "rank1 / Device phases (placed)",
        "rank1 / Placement Bound",
    ], lanes
    assert all(name.startswith(("rank0 / ", "rank1 / ")) for name in lanes[6:])
    assert not any(name.endswith(("Host", "Device phases (placed)", "Placement Bound")) for name in lanes[6:]), lanes
    # The sort index mirrors the pid, for any viewer that does read it.
    sort = {event["pid"]: event["args"]["sort_index"] for event in events if event.get("name") == "process_sort_index"}
    assert sorted(sort, key=lambda pid: sort[pid]) == sorted(sort)


def test_l3_directory_merge_draws_the_host_log_beside_the_capture(tmp_path):
    """One trace carries both clock domains, joined by the placement.

    The Host call tree is on its own clock and carries no placement error; the
    `clk=dev` phases the same log holds are placed into the window and cover
    the head of the run the capture records nothing for.
    """
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    output = tmp_path / "l3.json"
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0", "-o", str(output)])

    sc._generate_l3_trace(args, root)

    events = json.loads(output.read_text())["traceEvents"]
    names = {event["pid"]: event["args"]["name"] for event in events if event.get("name") == "process_name"}
    slices = [event for event in events if event.get("ph") == "X"]
    by_lane = {
        names[pid]: [event for event in slices if event["pid"] == pid]
        for pid in names
        if names[pid].startswith("rank0")
    }
    # The window is the same interval whether it is read off the Host lane or
    # off the bound lane — the second is drawn from the first.
    runner = next(e for e in by_lane["rank0 / Host"] if e["name"] == "chip.run.runner_run")
    outer = next(e for e in by_lane["rank0 / Placement Bound"] if e["name"].startswith("chip.run.runner_run ("))
    assert (runner["ts"], runner["dur"]) == (outer["ts"], outer["dur"])
    # The placed run wall starts at the window it was placed in, per the
    # lower-bound placement policy.
    wall = next(e for e in by_lane["rank0 / Device phases (placed)"] if e["name"].endswith("device_wall"))
    assert wall["ts"] == runner["ts"]
    assert wall["args"]["slack_ns"] == 6_000
    # Nothing is drawn before the axis origin.
    assert min(event["ts"] for event in slices) == 0.0


def test_l3_directory_merge_leaves_the_standalone_host_swimlane_alone(tmp_path):
    """The merge reads the same log; it does not change what the other tool writes."""
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    logs = sorted(root.glob("host.*.log"))
    spans = [span for log in logs for span in parse_spans(log.read_text().splitlines())]

    before = to_host_swimlane(spans)
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0", "-o", str(tmp_path / "l3.json")])
    sc._generate_l3_trace(args, root)

    assert to_host_swimlane(spans) == before


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
                "aicpu_thread_id": rank,
                "register_release_start_cycles": device_base + 600,
                "register_release_end_cycles": device_base + 650,
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

    sc._namespace_rank_trace(trace, 2, 6_000)

    counter, task = trace["traceEvents"]
    # Rank 2's views land in the third Chip stride: the first is reserved for
    # the Host-domain block, which has to sort below every Chip view.
    assert counter["pid"] == 302
    assert counter["args"] == {"AIC": 3, "AIV": 4}
    assert task["pid"] == 304
    assert task["args"]["rank"] == 2


def test_rank_namespace_rejects_a_view_pid_wider_than_the_stride():
    trace = {"traceEvents": [{"ph": "X", "pid": sc._RANK_PID_STRIDE, "tid": 1, "args": {}}]}

    with pytest.raises(ValueError, match="does not fit the per-Rank stride"):
        sc._namespace_rank_trace(trace, 1, 6_000)


def test_l3_directory_merge_rejects_ranks_from_different_host_clocks(tmp_path, capsys):
    """Two Hosts' windows are not on one axis, and containment cannot make them so.

    What would put them there is the window of the level that dispatched to
    both, which is a level up and not in these artifacts — so this refuses and
    says which input is missing rather than splicing two unrelated clocks.
    """
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7, clock_domain="boot-a")
    rank1_dir = _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8, clock_domain="boot-b")
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    with pytest.raises(ValueError, match="different Host clocks"):
        sc._generate_l3_trace(args, root)

    # A capture from before the field existed is placed anyway: the merge it is
    # asked for is same-host by construction, and the warning says so.
    rank1_path = rank1_dir / "chip_swimlane_records.json"
    rank1 = json.loads(rank1_path.read_text())
    rank1["metadata"].pop("host_clock_domain_id")
    rank1_path.write_text(json.dumps(rank1))
    sc._generate_l3_trace(args, root)
    assert "predates metadata.host_clock_domain_id" in capsys.readouterr().err


def test_l3_directory_merge_needs_a_host_log_to_place_ranks(tmp_path):
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    for log in root.glob("host.*.log"):
        log.unlink()
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    with pytest.raises(ValueError, match="no host.*log under"):
        sc._generate_l3_trace(args, root)


def test_l3_directory_merge_refuses_to_guess_between_look_alike_ranks(tmp_path):
    """Ranks running the same shape leave nothing to pair them by.

    Neither artifact names the other — the Host log's identity attributes are
    the dispatch's, identical across the group — so when the device windows
    match too, guessing would place a Rank's work in another Rank's window.
    """
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    _write_l3_host_log(root, 1, host_shift_ns=10_000, sched_window_ns=100)
    rank1_path = root / "rank1" / "d0" / "chip_swimlane_records.json"
    rank1 = json.loads(rank1_path.read_text())
    device_base = 100 + 100_000
    rank1["aicpu_scheduler_phases"] = [
        [{"kind": "dispatch", "start_cycles": device_base + 800, "end_cycles": device_base + 850}]
    ]
    rank1_path.write_text(json.dumps(rank1))
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    with pytest.raises(ValueError, match="pair ambiguously"):
        sc._generate_l3_trace(args, root)

    pinned = sc._build_parser().parse_args(
        [str(root), "--dispatch", "d0", "--rank-pid", "0=1000", "--rank-pid", "1=1001"]
    )
    _, rank_metadata = sc._generate_l3_trace(pinned, root)
    assert [item["host_pairing"]["pid"] for item in rank_metadata] == [1_000, 1_001]
    assert [item["host_pairing"]["source"] for item in rank_metadata] == ["pinned", "pinned"]


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


def test_host_orchestrator_phases_without_a_containing_window_stay_composite(tmp_path):
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
    assert trace["metadata"]["layout"] == "causal_composite"
    assert any(
        event.get("ph") == "M"
        and event.get("name") == "process_name"
        and event.get("args", {}).get("name") == "Host Orchestrator"
        for event in trace["traceEvents"]
    )
    assert not any(
        event.get("cat") == "flow" and event.get("name") == "submit→dispatch" for event in trace["traceEvents"]
    )


def test_single_capture_uses_host_timeline_when_a_containing_window_is_given(tmp_path):
    """One capture placed inside the window the Host log says contained it.

    The `sched` window joins the two artifacts onto one device-phase timeline,
    and the window's spare width becomes the placement's published slack.
    """
    document = {
        "chip_swimlane_level": 4,
        "metadata": {
            "clock_freq_hz": 1_000_000_000,
            "num_cores": 1,
            "core_types": ["aiv"],
            "core_to_thread": [0],
            "host_clock_domain_id": "same-boot",
            "host_timeline_origin_ns": 1_000,
        },
        "aicore_tasks": [[0, 7, 1, 2_100, 2_200, 0]],
        "aicpu_tasks": [[0, 1, 2_000, 2_300]],
        "aicpu_scheduler_phases": [
            [{"kind": "dispatch", "start_cycles": 1_900, "end_cycles": 1_950, "tasks_processed": 1}]
        ],
        "aicpu_orchestrator_phases": [[{"submit_idx": 0, "task_id": 7, "start_cycles": 1_800, "end_cycles": 1_850}]],
    }
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(json.dumps(document))
    placement = _containment_placement(document)

    data = sc.read_perf_data(raw, placement=placement)

    assert data["orchestrator_source"] == "aicpu"
    # Device cycle 2100 sits 925 ns into the run wall, which is drawn from the
    # window's start at Host ns 1000; the origin is this capture's own 1000.
    assert data["tasks"][0]["start_time_us"] == 0.925
    assert data["timeline_metadata"]["layout"] == "containment_spliced"
    assert data["timeline_metadata"]["host_clock_domain_id"] == "same-boot"
    assert data["timeline_metadata"]["source_timeline_origin_ns"] == 1_000
    assert data["timeline_metadata"]["placement"]["slack_ns"] == 3_000
    assert data["timeline_metadata"]["placement"]["join"]["sources"] == ["sched", "device_wall"]
    # The `sched` window is 50 ns wider than the records inside it, which is
    # exactly how well the two artifacts can be joined.
    assert data["timeline_metadata"]["placement"]["join"]["residual_ns"] == 50


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
                        "aicpu_thread_id": 1,
                        "handshake_start_cycles": 90,
                        "handshake_complete_cycles": 91,
                        "config_start_cycles": 92,
                        "topology_complete_cycles": 93,
                        "context_publish_start_cycles": 93,
                        "context_publish_complete_cycles": 94,
                        "bootstrap_wait_start_cycles": 95,
                        "bootstrap_complete_cycles": 96,
                        "register_release_start_cycles": 97,
                        "register_release_end_cycles": 98,
                        "exit_signal_start_cycles": 181,
                        "exit_signal_end_cycles": 182,
                        "exit_wait_start_cycles": 183,
                        "exit_wait_end_cycles": 184,
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
    assert data["aicpu_lifecycle_records"][0]["register_release_start_time_us"] == pytest.approx(0.007)

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
    assert any(event.get("cat") == "scheduler" and event.get("name") == "idle" for event in events)
    assert any(
        event.get("name") == "process_name" and event.get("args", {}).get("name") == "AICPU Lifecycle"
        for event in events
    )
    assert any(
        event.get("name") == "thread_name" and event.get("args", {}).get("name") == "AICPU Thread 1" for event in events
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
                "aicpu_thread_id": 1,
                "handshake_start_time_us": 0.0,
                "handshake_complete_time_us": 2.0,
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
                "aicpu_lifecycle_records": [
                    {
                        "aicpu_thread_id": 0,
                        "register_release_start_cycles": 90,
                        "register_release_end_cycles": 91,
                    }
                ],
            }
        )
    )

    data = sc.read_perf_data(raw)
    assert data["aicpu_lifecycle_records"][0]["register_release_start_time_us"] == 0.0

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
        aicpu_lifecycle_records=[{"aicpu_thread_id": 0}],
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


def test_host_and_device_timestamps_share_one_axis_through_containment(tmp_path):
    """A host-orchestrating runtime's two clocks, joined without calibration.

    Host orchestration records are Host ns already; the device records reach the
    same axis through the window that contained them. The seam is bounded rather
    than closed, so the layout publishes `slack_ns` instead of claiming a fit.
    """
    document = {
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
        },
        "aicore_tasks": [[0, 7, 1, 2_100, 2_200, 0]],
        "aicpu_tasks": [[0, 1, 2_000, 2_300]],
        "aicpu_scheduler_phases": [
            [{"kind": "dispatch", "start_cycles": 1_900, "end_cycles": 1_950, "tasks_processed": 1}]
        ],
        "host_orchestrator_phases": [[{"submit_idx": 0, "task_id": 7, "start_host_ns": 1_500, "end_host_ns": 1_800}]],
    }
    raw = tmp_path / "chip_swimlane_records.json"
    raw.write_text(json.dumps(document))
    placement = _containment_placement(document)

    data = sc.read_perf_data(raw, placement=placement)

    # Host orchestration is on its own clock and keeps its exact offsets; the
    # device records land just after it, where the window allows them earliest.
    assert data["aicpu_orchestrator_phases"][0][0]["start_time_us"] == 0.0
    assert data["aicpu_orchestrator_phases"][0][0]["end_time_us"] == 0.3
    assert data["tasks"][0]["dispatch_time_us"] == 0.325
    assert data["tasks"][0]["start_time_us"] == 0.425
    assert data["timeline_metadata"]["layout"] == "containment_spliced"
    assert data["timeline_metadata"]["cross_domain_latency_available"] is True
    assert data["timeline_metadata"]["placement"]["slack_ns"] == 3_000
    assert data["timeline_metadata"]["source_timeline_origin_ns"] == 1_500
    assert data["timeline_metadata"]["timeline_origin_ns"] == 1_500


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
                    },
                    "aicore_tasks": [[0, 7, 1, 2_100, 2_200, 0]],
                    "aicpu_tasks": [[0, 1, 2_000, 2_300]],
                    "aicpu_scheduler_phases": [[{"kind": "dispatch", "start_cycles": 1_900, "end_cycles": 1_950}]],
                    "host_orchestrator_phases": host_phases,
                }
            )
        )

        data = sc.read_perf_data(raw, placement=_containment_placement(json.loads(raw.read_text())))

        assert data["timeline_metadata"]["layout"] == "containment_spliced"
        assert data["timeline_metadata"]["trace_status"] == "partial"
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


def test_aicore_scheduler_uses_one_lane_and_display_names(tmp_path):
    out = tmp_path / "trace.json"
    scheduler_phases = [
        [
            {
                "phase": "complete",
                "start_time_us": 1.0,
                "end_time_us": 4.0,
                "tasks_processed": 1,
                "task_id": 23,
            },
            {
                "phase": "resolve",
                "start_time_us": 2.0,
                "end_time_us": 3.0,
                "tasks_processed": 1,
                "task_id": 23,
            },
            {
                "phase": "state_probe",
                "start_time_us": 4.0,
                "end_time_us": 5.0,
                "tasks_processed": 1,
                "task_id": 5,
            },
            {
                "phase": "dispatch",
                "start_time_us": 5.0,
                "end_time_us": 6.0,
                "tasks_processed": 1,
                "task_id": 5,
            },
            {
                "phase": "worksteal",
                "start_time_us": 6.0,
                "end_time_us": 7.0,
                "tasks_processed": 1,
                "task_id": 7,
            },
            {
                "phase": "refill",
                "start_time_us": 7.0,
                "end_time_us": 8.0,
                "tasks_processed": 1,
                "task_id": 9,
            },
        ]
    ]
    scheduler_streams = [
        {
            "producer": "aicore",
            "scheduler_id": 4,
            "worker_id": 36,
            "core_type": "aiv",
            "physical_core_id": 26,
        }
    ]

    sc.generate_chrome_trace_json(
        [], str(out), scheduler_phases=scheduler_phases, scheduler_streams=scheduler_streams, core_to_thread=[0]
    )

    events = json.loads(out.read_text())["traceEvents"]
    scheduler_metadata = [
        event
        for event in events
        if event.get("ph") == "M"
        and event.get("pid") == 2
        and event.get("name") == "thread_name"
        and event.get("tid") != 3999
    ]
    assert [(event["tid"], event["args"]["name"]) for event in scheduler_metadata] == [(30000, "Scheduler_26")]
    phases = [event for event in events if event.get("cat") == "scheduler"]
    assert {event["tid"] for event in phases} == {30000}
    assert [event["name"] for event in phases] == [
        "Completion(t23)",
        "Resolve(t23)",
        "StateProbe(t5)",
        "Dispatch(t5)",
        "Worksteal(t7)",
        "Refill(t9)",
    ]
    assert [event["args"]["task_id"] for event in phases] == [23, 23, 5, 5, 7, 9]


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


def _append_l3_invocations(root, rank, *, host_shift_ns, count, captured_index, sched_window_ns):
    """Rewrite one Rank's Host log as a process that ran `count` times.

    A Host log covers the whole run, not the one invocation a capture holds, so
    this is the ordinary shape rather than an edge case. Every invocation gets
    the same window and the same `sched` phase — a repeated call of one
    callable — and is told apart only by the dispatch its root `chip.run` span
    names, which `next_dispatch_id_` increments once per worker per dispatch.

    ``captured_index`` keeps its invocation on the window `_write_l3_rank` gave
    the capture, so the decoys sit a millisecond either side of it.
    """
    pid = 1000 + rank
    lines = []
    for index in range(count):
        inv = index + 1
        start_ns = host_shift_ns + 1_000 + (index - captured_index) * 1_000_000
        prefix = f"[mono_ns={start_ns}][T0x1][TIMING] emit_host_span: "
        head = f"[STRACE] v=1 pid={pid} tid={pid} inv={inv} hid=abc"
        lines += [
            f"{prefix}{head} depth=0 name=chip.run ts={start_ns - 500} dur=9000 "
            f"run_id=17 dispatch_id={inv} slot_id=0 generation=1 run_epoch={inv}",
            f"{prefix}{head} depth=1 name=chip.run.runner_run ts={start_ns} dur=8000 ",
            f"{prefix}{head} depth=2 name=chip.run.runner_run.device_wall ts=0 dur=2000 clk=dev",
            f"{prefix}{head} depth=3 name=chip.run.runner_run.device_wall.sched ts=700 dur={sched_window_ns} clk=dev",
        ]
    (root / f"host.{pid}.log").write_text("\n".join(lines) + "\n")


def _strip_dispatch_attributes(root):
    """Make the logs look like a capture that predates the identity attributes."""
    for log in root.glob("host.*.log"):
        log.write_text(log.read_text().replace(" run_id=17 dispatch_id=", " no_id=17 no_dispatch="))


def test_l3_directory_merge_picks_the_captured_invocation_out_of_a_repeated_run(tmp_path):
    """A Host log holds every invocation; only one of them is the capture's.

    The device windows are identical across a repeated call, so the fit alone
    can neither pick one nor afford to enumerate them. The dispatch the capture
    names narrows it to that round, and the fit then only has the two Ranks of
    that round to tell apart.
    """
    root = tmp_path / "dfx_outputs"
    for rank, task_id in ((0, 7), (1, 8)):
        host_shift_ns = 5_000_000 + rank * 10_000
        capture_dir = _write_l3_rank(root, rank, host_shift_ns=host_shift_ns, task_id=task_id)
        _append_l3_invocations(
            root,
            rank,
            host_shift_ns=host_shift_ns,
            count=4,
            captured_index=2,
            sched_window_ns=100 + rank * 100,
        )
        _write_dispatch_identity(capture_dir, run_id=17, task_slot=5, group_index=rank, group_size=2)
        # `_write_dispatch_identity` numbers this from the dN index; the capture
        # is d0 but the run's third dispatch, which is the case that matters.
        path = capture_dir / "dispatch_identity.json"
        identity = json.loads(path.read_text())
        identity["endpoint_dispatch_id"] = 3
        path.write_text(json.dumps(identity))

    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])
    _, rank_metadata = sc._generate_l3_trace(args, root)

    assert [item["host_pairing"]["pid"] for item in rank_metadata] == [1_000, 1_001]
    assert [item["host_pairing"]["inv"] for item in rank_metadata] == [3, 3]


def test_l3_directory_merge_refuses_a_pairing_search_it_cannot_finish(tmp_path):
    """Without the dispatch, the same repeated run is not scorable at all.

    Enumerating whole assignments over every invocation in the log is
    factorial in the log's length, so it has to refuse rather than grind.
    """
    root = tmp_path / "dfx_outputs"
    for rank in range(4):
        host_shift_ns = 60_000_000 + rank * 10_000
        _write_l3_rank(root, rank, host_shift_ns=host_shift_ns, task_id=7 + rank)
        _append_l3_invocations(
            root,
            rank,
            host_shift_ns=host_shift_ns,
            count=20,
            captured_index=0,
            sched_window_ns=100 + rank * 100,
        )
    _strip_dispatch_attributes(root)
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    with pytest.raises(ValueError, match="too many to score"):
        sc._generate_l3_trace(args, root)


def test_l3_rank_pid_pin_names_the_invocation_when_the_process_ran_more_than_once(tmp_path):
    """`RANK=PID` cannot name one of four runs of that process."""
    root = tmp_path / "dfx_outputs"
    for rank, task_id in ((0, 7), (1, 8)):
        host_shift_ns = 5_000_000 + rank * 10_000
        _write_l3_rank(root, rank, host_shift_ns=host_shift_ns, task_id=task_id)
        _append_l3_invocations(
            root,
            rank,
            host_shift_ns=host_shift_ns,
            count=4,
            captured_index=2,
            sched_window_ns=100 + rank * 100,
        )
    _strip_dispatch_attributes(root)

    bare = sc._build_parser().parse_args(
        [str(root), "--dispatch", "d0", "--rank-pid", "0=1000", "--rank-pid", "1=1001"]
    )
    with pytest.raises(ValueError, match=r"4 placeable invocations \(inv 1, 2, 3, 4\)"):
        sc._generate_l3_trace(bare, root)

    pinned = sc._build_parser().parse_args(
        [str(root), "--dispatch", "d0", "--rank-pid", "0=1000:3", "--rank-pid", "1=1001:3"]
    )
    _, rank_metadata = sc._generate_l3_trace(pinned, root)

    assert [(item["host_pairing"]["pid"], item["host_pairing"]["inv"]) for item in rank_metadata] == [
        (1_000, 3),
        (1_001, 3),
    ]


def _write_l3_scheduler_log(root, *, pid=900, spans):
    """The L3 process's own log, which the run writes into the same case root.

    `_submit_l3_locked` binds this process's log to `CallConfig.output_prefix`
    just as each ChipWorker child does, so the scheduler's `node.*` spans sit
    beside the Rank captures rather than anywhere the merge has to be told
    about.
    """
    lines = []
    for name, ts, dur, tid, inv in spans:
        prefix = f"[mono_ns={ts}][T0x1][TIMING] emit_host_span: "
        lines.append(
            f"{prefix}[STRACE] v=1 pid={pid} tid={tid} inv={inv} hid=def depth=1 name={name} ts={ts} dur={dur} "
            f"run_id=17 task_slot=5 worker_id=0 dispatch_id={inv}"
        )
    (root / f"host.{pid}.log").write_text("\n".join(lines) + "\n")


def test_l3_directory_merge_draws_the_dispatching_process_beside_the_ranks(tmp_path):
    """The L3 scheduler's own lanes belong in the L3 swimlane.

    Its spans are Host CLOCK_MONOTONIC and same-host cross-process comparable,
    so they go straight onto the axis — no placement, and no `slack_ns`, which
    is a device-clock term only.
    """
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    _write_l3_scheduler_log(
        root,
        spans=[
            ("node.submit", 200, 300, 900, 1),
            ("node.dispatch", 600, 9_000, 901, 1),
            ("node.complete", 9_800, 400, 901, 1),
        ],
    )
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    output_path, _ = sc._generate_l3_trace(args, root)
    trace = json.loads(output_path.read_text())

    assert trace["metadata"]["dispatcher_pids"] == [900]
    # The axis now starts at the scheduler's first span, which opens before any
    # Rank's window: an origin taken from the windows alone would put it at a
    # negative timestamp.
    assert trace["metadata"]["global_origin_ns"] == 200

    dispatcher = [event for event in trace["traceEvents"] if event.get("ph") == "X" and event["pid"] < 11]
    assert {event["name"] for event in dispatcher} == {"node.submit", "node.dispatch", "node.complete"}
    assert all("slack_ns" not in event["args"] for event in dispatcher)
    assert [event["ts"] for event in dispatcher if event["name"] == "node.dispatch"] == [0.4]
    # Perfetto orders process groups by pid, so the scheduler has to sort below
    # every Rank's Host lane, which in turn sorts below every Chip view.
    process_pids = sorted(
        event["pid"] for event in trace["traceEvents"] if event.get("ph") == "M" and event.get("name") == "process_name"
    )
    assert process_pids[0] == 1
    assert min(pid for pid in process_pids if pid >= sc._HOST_BLOCK_PID_BASE) == sc._HOST_BLOCK_PID_BASE


def test_l3_directory_merge_leaves_out_the_dispatches_it_is_not_merging(tmp_path):
    """The scheduler's log covers the run; the merge covers one dispatch.

    Grouping is per `(pid, inv)`, which is how a multi-round L3 run separates
    its dispatches, so the round that did not produce these captures is left
    out whole.
    """
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    _write_l3_scheduler_log(
        root,
        spans=[
            ("node.dispatch", 600, 9_000, 901, 1),
            ("node.dispatch", 5_000_000, 9_000, 901, 2),
        ],
    )
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    output_path, _ = sc._generate_l3_trace(args, root)
    trace = json.loads(output_path.read_text())

    dispatcher = [event for event in trace["traceEvents"] if event.get("ph") == "X" and event["pid"] < 11]
    assert [event["args"]["os_pid"] for event in dispatcher] == [900]
    assert [event["ts"] for event in dispatcher] == [0.0]


def test_l3_directory_merge_draws_the_scheduler_loops_that_carry_no_invocation(tmp_path):
    """`node.scheduler_loop` belongs to a thread, not to a run.

    It is emitted with `inv=0`, so every loop of the whole run shares one
    pseudo-invocation and the invocation cannot be the selection unit: taking
    them as a group draws all of them or none. They are selected span by span
    instead, against the story the dispatch spans already staked out — which
    reaches past the Ranks' own windows at both ends, since a loop both
    precedes the first dispatch and follows the last completion.
    """
    root = tmp_path / "dfx_outputs"
    _write_l3_rank(root, 0, host_shift_ns=0, task_id=7)
    _write_l3_rank(root, 1, host_shift_ns=10_000, task_id=8)
    _write_l3_scheduler_log(
        root,
        spans=[
            # `graph_build` closes before the Ranks open and is kept only
            # because its invocation is: the invocation is the unit here.
            ("node.graph_build", 200, 400, 900, 1),
            ("node.dispatch", 700, 9_000, 901, 1),
            # Bracketing loops: one before the Ranks open, one after they close.
            ("node.scheduler_loop", 300, 100, 901, 0),
            ("node.scheduler_loop", 9_600, 200, 901, 0),
            # A loop from a later dispatch, outside the story entirely.
            ("node.scheduler_loop", 5_000_000, 200, 901, 0),
        ],
    )
    args = sc._build_parser().parse_args([str(root), "--dispatch", "d0"])

    output_path, _ = sc._generate_l3_trace(args, root)
    trace = json.loads(output_path.read_text())

    loops = [
        event for event in trace["traceEvents"] if event.get("ph") == "X" and event["name"] == "node.scheduler_loop"
    ]
    assert sorted(event["ts"] for event in loops) == [0.1, 9.4]
    assert "node.graph_build" in {
        event["name"] for event in trace["traceEvents"] if event.get("ph") == "X" and event["pid"] == 1
    }
