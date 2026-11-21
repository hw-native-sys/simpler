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
    claim_location = {"slot": 0, "mix": True, "participant_mask": 3} if level == 2 else {}
    raw["percounter_task_events"] = {
        "dropped": 0,
        "unmatched_deferred_updates": 0,
        "folded_task_ids": [],
        "inline_completed_task_ids": [],
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
                (0, "claim", 10, 0, 0, 0, {"claim_start_cycles": 6, **claim_location}),
                (0, "ready", 30, 0, 0, 0, {}),
                (0, "ready", 35, 2, 0, 1, {}),
                (0, "kernel_end", 55, 0, 0, 0, {"kernel_start_cycles": 40}),
                (0, "kernel_end", 60, 2, 0, 1, {"kernel_start_cycles": 41}),
                (0, "defer_registered", 61, 0, 0, 0, {"deferred_group": 4}),
                (0, "completion_issue", 65, 2, 0, 1, {"contribution": 1}),
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
        percounter_block_view=True,
    )
    if level == 2:
        default = generate_chrome_trace_json(
            data["tasks"],
            None,
            runtime_name="percounter",
            scheduler_task_producer="aicore",
            chip_swimlane_level=level,
            percounter_task_events=data["percounter_task_events"],
        )["traceEvents"]
        assert any(e.get("pid") == 7 and e["ph"] == "b" for e in default)
        assert not any(e.get("pid") == 3 for e in default)
        assert not any(e.get("pid") == 7 for e in trace["traceEvents"])
    view = [e for e in trace["traceEvents"] if e["pid"] == 3]
    assert any(e.get("name", "").startswith("Kernel execution · ") for e in view) == (level >= 3)
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
    assert block_0[7]["name"].startswith("Completion issued +1")
    assert block_0[-1]["args"]["worker_id"] == 1
    assert block_0[-1]["args"]["cluster_index"] == 1
    assert block_0[-1]["ts"] == pytest.approx((90 - 6) * cycle_us)
    block_1 = [e for e in view if e["ph"] == "i" and e["args"]["block_index"] == 1]
    assert [e["args"]["kind"] for e in block_1] == ["claim_start", "claim", "ready", "completion_issue"]
    assert block_1[0]["tid"] != block_0[0]["tid"]


def _l2_scheduler_milestones(tmp_path, rows, deps_edges, kernels, *, folded_task_ids=(), inline_completed_task_ids=()):
    capture = Capture(runtime="percounter", level=2, scheduler_producer="aicore", clock_freq_hz=50_000_000)
    capture.task(task_id=8, core_id=0, start=80, end=90, run_epoch=0)
    raw = capture.build()
    raw["percounter_task_events"] = {
        "dropped": 0,
        "unmatched_deferred_updates": 0,
        "folded_task_ids": list(folded_task_ids),
        "inline_completed_task_ids": list(inline_completed_task_ids),
        "records": [
            {
                "task_id": task,
                "kind": kind,
                "cycles": cycles,
                "run_epoch": 0,
                "worker_id": lane,
                "cluster_index": 0,
                "cluster_lane": lane,
                "block_index": 0,
                "block_count": 1,
                **extra,
            }
            for task, kind, cycles, lane, extra in [
                *rows,
                (8, "claim", 30, 0, {"claim_start_cycles": 26}),
                (8, "ready", 76, 0, {}),
            ]
        ],
    }
    path = tmp_path / "l2.json"
    path.write_text(json.dumps(raw))
    data = read_perf_data(path)
    trace = generate_chrome_trace_json(
        data["tasks"],
        None,
        runtime_name="percounter",
        scheduler_task_producer="aicore",
        chip_swimlane_level=2,
        percounter_task_events=data["percounter_task_events"],
        deps_edges=deps_edges,
        deps_kernel_map=kernels,
        deps_block_map=dict.fromkeys(kernels, 1),
    )
    return {
        event["args"]["kind"]: event["ts"]
        for event in trace["traceEvents"]
        if event.get("cat") == "percounter_task_timing" and event["ph"] == "i" and event["args"]["task_id"] == 8
    }


@pytest.mark.parametrize("deferred", [False, True])
def test_l2_ideal_ready_uses_the_last_mix_lane_credit(tmp_path, deferred):
    rows = [(7, "completion_issue", 40, 0, {"contribution": 1})]
    if deferred:
        rows += [
            (7, "defer_registered", 50, 1, {"deferred_group": 6}),
            (7, "deferred_completion_issue", 65, 1, {"contribution": 1, "deferred_group": 6}),
        ]
    else:
        rows.append((7, "completion_issue", 65, 1, {"contribution": 1}))
    milestones = _l2_scheduler_milestones(tmp_path, rows, {7: [8]}, {7: [0, 1, -1]})
    # These markers are 35 cycles apart regardless of the window's time origin.
    assert milestones["ideal_ready"] - milestones["claim"] == pytest.approx(35 * 1e6 / 50_000_000)


@pytest.mark.parametrize("all_folded", [True, False])
def test_l2_ideal_ready_expands_only_marked_folded_joins(tmp_path, all_folded):
    rows = [
        (1, "completion_issue", 40, 0, {"contribution": 1}),
        (2, "completion_issue", 65, 0, {"contribution": 1}),
    ]
    milestones = _l2_scheduler_milestones(
        tmp_path,
        rows,
        {1: [3], 2: [3], 3: [4], 4: [8]},
        {1: [0, -1, -1], 2: [0, -1, -1], 3: [-1, -1, -1], 4: [-1, -1, -1]},
        folded_task_ids=(3, 4) if all_folded else (4,),
    )
    if all_folded:
        assert milestones["ideal_ready"] - milestones["claim"] == pytest.approx(35 * 1e6 / 50_000_000)
    else:
        assert "ideal_ready" not in milestones


@pytest.mark.parametrize("folded", [False, True])
def test_l2_ideal_ready_omits_host_completed_predecessors(tmp_path, folded):
    rows = [(1, "completion_issue", 65, 0, {"contribution": 1})]
    edges = {1: [3], 2: [3], 3: [8]} if folded else {1: [8], 2: [8]}
    milestones = _l2_scheduler_milestones(
        tmp_path,
        rows,
        edges,
        {1: [0, -1, -1]},
        folded_task_ids=(3,) if folded else (),
        inline_completed_task_ids=(2,),
    )
    assert milestones["ideal_ready"] - milestones["claim"] == pytest.approx(35 * 1e6 / 50_000_000)


def test_l2_physical_slots_keep_mix_members_and_deferred_settlement_separate():
    from simpler_setup.tools.percounter.schedule_timing import schedule_events  # noqa: PLC0415

    def event(kind, ts, lane, **fields):
        return dict(
            kind=kind,
            time_us=ts,
            run_epoch=1,
            task_id=5,
            block_index=0,
            cluster_index=2,
            cluster_lane=lane,
            worker_id=10 + lane,
            **fields,
        )

    rows = [
        event("claim", 2, 1, slot=0, mix=True, participant_mask=7, claim_start_us=1),
        event("mix_publish", 3, 1),
        event("inputs_pending", 4, 0),
        event("inputs_ready", 8, 0),
        event("entry_commit", 10, 0),
        event("entry_commit", 12, 1),
        event("ready", 13, 0),
        event("ready", 14, 1),
        event("kernel_end", 21, 0, dispatch_us=14, kernel_start_us=15, finish_us=22),
        event("kernel_end", 23, 1, dispatch_us=15, kernel_start_us=17, finish_us=24),
        event("kernel_end", 25, 2, dispatch_us=16, kernel_start_us=18, finish_us=26),
        event("defer_registered", 27, 2, deferred_group=9),
        event("completion_issue", 23, 0, contribution=1),
        event("completion_issue", 28, 1, contribution=1),
        event("deferred_completion_issue", 40, 2, deferred_group=9, contribution=1),
    ]
    ev = list(schedule_events({"records": rows}, 7, lambda t: f"t{t}"))
    names = {e["id2"]["local"]: e["name"] for e in ev if e["ph"] == "n"}
    bars = [
        (names[e["id2"]["local"]], e["name"].partition(" · ")[0], e["ts"], ev[i + 1]["ts"])
        for i, e in enumerate(ev)
        if e["ph"] == "b"
    ]
    assert ("C02 / MIX0 (AIV0 proxy)", "Prepare MIX descriptor → SSBUF publish", 2, 3) in bars
    assert ("C02 / MIX0/AIC", "Predecessor pending → inputs observed", 4, 8) in bars
    assert all(name != "Kernel execution" for _, name, _, _ in bars)
    assert ("C02 / MIX0/AIV0", "Inputs proven → entry commitment", 8, 12) in bars
    assert ("C02 / MIX0/AIC", "MIX rendezvous → selection", 10, 13) in bars
    assert ("C02 / MIX0/AIC", "Output fence → completion issue", 22, 23) in bars
    assert ("C02 / MIX0/AIV0", "Output fence → completion issue", 24, 28) in bars
    assert ("C02 / deferred 0", "Defer registration → settlement issue", 27, 40) in bars
    partial = [r for r in rows if not (r["kind"] == "kernel_end" and r["cluster_lane"] == 0)]
    partial_issues = [
        e for e in schedule_events({"records": partial}, 7, str) if e.get("name", "").startswith("Output fence →")
    ]
    assert [(e["ts"], e["args"]["worker_id"]) for e in partial_issues] == [(24, 11)]
