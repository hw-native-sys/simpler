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

from simpler_setup.tools import swimlane_converter as sc


def _task_row(task_id, core_id, core_type="aiv", *, dispatch=10.0, start=11.0, end=20.0, receive=10.5):
    return {
        "task_id": task_id,
        "func_id": 0,
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


def test_spmd_pred_routes_dependency_to_min_core_subtask(tmp_path):
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
    sched_flow = _first_scheduler_dependency_flow(out)
    assert sched_flow[0]["output_task_count"] == 4
    assert sched_flow[0]["input_task_count"] == 1
    assert sched_flow[0]["ts"] == tasks[0]["finish_time_us"] - 0.01
    assert sched_flow[1]["ts"] == tasks[4]["dispatch_time_us"]

    deps_dup = {pred_id: [succ_id, succ_id, succ_id]}
    out_dup = _generate_trace(tasks, deps_dup, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out_dup, pid=4) == 1
    assert _count_dependency_flow_starts(out_dup, pid=3) == 1

    # SPMD(4) -> SPMD(4): one logical edge, symmetric fan metadata.
    to_spmd_tasks = [_task_row(pred_id, core_id, dispatch=10.0 + core_id) for core_id in range(4)]
    to_spmd_tasks.extend(_task_row(succ_id, core_id, dispatch=30.0 + core_id) for core_id in range(4))
    out_to_spmd = _generate_trace(to_spmd_tasks, deps_edges, {pred_id: 4, succ_id: 4}, tmp_path)
    assert _count_dependency_flow_starts(out_to_spmd, pid=4) == 1
    assert _count_dependency_flow_starts(out_to_spmd, pid=3) == 1
    assert not _has_spmd_block_level_track(out_to_spmd)
    to_spmd_flow = _first_worker_dependency_flow(out_to_spmd)
    assert to_spmd_flow[0]["output_task_count"] == 4
    assert to_spmd_flow[0]["input_task_count"] == 4
    to_spmd_sched = _first_scheduler_dependency_flow(out_to_spmd)
    assert to_spmd_sched[0]["output_task_count"] == 4
    assert to_spmd_sched[0]["input_task_count"] == 4
    assert to_spmd_sched[0]["ts"] == to_spmd_tasks[0]["finish_time_us"] - 0.01
    assert to_spmd_sched[1]["ts"] == to_spmd_tasks[4]["dispatch_time_us"]

    # Scheduler mirror falls back to the first sibling with a captured finish.
    pred_rows = [_task_row(pred_id, core_id, dispatch=10.0 + core_id) for core_id in range(4)]
    pred_rows[0]["finish_time_us"] = 0.0
    tasks_no_min_finish = list(pred_rows)
    tasks_no_min_finish.append(_task_row(succ_id, 10))
    out_fallback = _generate_trace(tasks_no_min_finish, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out_fallback, pid=4) == 1
    assert _count_dependency_flow_starts(out_fallback, pid=3) == 1
    sched_fallback = _first_scheduler_dependency_flow(out_fallback)
    assert sched_fallback[0]["tid"] == _core_tid(1)
    assert sched_fallback[0]["ts"] == tasks_no_min_finish[1]["finish_time_us"] - 0.01


def test_spmd_succ_routes_dependency_to_min_core_subtask(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [_task_row(pred_id, 0)]
    tasks.extend(
        _task_row(
            succ_id, core_id, dispatch=20.0 + core_id, start=21.0 + core_id, end=30.0 + core_id, receive=20.5 + core_id
        )
        for core_id in range(4)
    )
    deps_edges = {pred_id: [succ_id]}
    deps_block_map = {pred_id: 1, succ_id: 4}

    out = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out, pid=4) == 1
    assert _count_dependency_flow_starts(out, pid=3) == 1
    flow = _first_worker_dependency_flow(out)
    assert flow[0]["output_task_count"] == 1
    assert flow[0]["input_task_count"] == 4
    assert flow[1]["tid"] == _core_tid(0)
    assert flow[1]["ts"] == tasks[1]["receive_time_us"]
    sched_flow = _first_scheduler_dependency_flow(out)
    assert sched_flow[0]["output_task_count"] == 1
    assert sched_flow[0]["input_task_count"] == 4
    assert sched_flow[1]["ts"] == tasks[1]["dispatch_time_us"]

    # Scheduler mirror lands on dispatch even when succ finish was not captured.
    for row in tasks[1:]:
        row["finish_time_us"] = 0.0
    out_no_finish = _generate_trace(tasks, deps_edges, deps_block_map, tmp_path)
    assert _count_dependency_flow_starts(out_no_finish, pid=4) == 1
    assert _count_dependency_flow_starts(out_no_finish, pid=3) == 1

    # SPMD succ: no visible setup -> inbound anchors on exec start; bind_id matches anchor slice.
    succ_no_setup = _task_row(succ_id, 0, dispatch=20.0, start=21.0, end=30.0, receive=20.99)
    succ_no_setup["local_setup_us"] = 0.01
    tasks_no_setup = [_task_row(pred_id, 0), succ_no_setup]
    out_no_setup = _generate_trace(tasks_no_setup, deps_edges, deps_block_map, tmp_path)
    with open(out_no_setup) as f:
        events = json.load(f)["traceEvents"]
    flow_f = next(
        e
        for e in events
        if e.get("cat") == "flow" and e.get("ph") == "f" and e.get("pid") == 4 and e.get("name") == "dependency"
    )
    assert flow_f["ts"] == 21.0
    dst_slice = next(e for e in events if e.get("ph") == "X" and e.get("id") == flow_f["bind_id"])
    assert dst_slice["ts"] == 21.0

    # SPMD tasks do not participate in complete-phase flow arrows.
    scalar_task_id = 300
    spmd_tasks = [_task_row(succ_id, core_id) for core_id in range(4)]
    spmd_tasks.append(_task_row(scalar_task_id, 10))
    scheduler_phases = [
        [
            {
                "phase": "complete",
                "start_time_us": 5.0,
                "end_time_us": 25.0,
                "tasks_processed": 1,
                "loop_iter": 0,
            }
        ]
    ]
    core_to_thread = [0] * 11
    out_complete = tmp_path / "trace_complete.json"
    sc.generate_chrome_trace_json(
        spmd_tasks,
        str(out_complete),
        deps_edges={succ_id: [scalar_task_id]},
        deps_block_map={succ_id: 4, scalar_task_id: 1},
        scheduler_phases=scheduler_phases,
        core_to_thread=core_to_thread,
    )
    with open(out_complete) as f:
        complete_events = json.load(f)["traceEvents"]
    complete_starts = [
        e for e in complete_events if e.get("cat") == "flow" and e.get("name") == "complete" and e.get("ph") == "s"
    ]
    spmd_tids = {_core_tid(c) for c in range(4)}
    assert not any(e.get("tid") in spmd_tids for e in complete_starts)
    assert any(e.get("tid") == _core_tid(10) for e in complete_starts)


def test_spmd_mix_to_mix_uses_anchor_cartesian_product(tmp_path):
    pred_id = 100
    succ_id = 200
    tasks = [
        _task_row(pred_id, 0, "aic", dispatch=10.0, start=11.0, end=20.0),
        _task_row(pred_id, 1, "aiv", dispatch=10.1, start=11.1, end=20.1),
        _task_row(pred_id, 3, "aiv", dispatch=10.3, start=11.3, end=20.3),
        _task_row(succ_id, 4, "aic", dispatch=30.0, start=31.0, end=40.0, receive=30.5),
        _task_row(succ_id, 5, "aiv", dispatch=30.1, start=31.1, end=40.1, receive=30.6),
        _task_row(succ_id, 7, "aiv", dispatch=30.3, start=31.3, end=40.3, receive=30.8),
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
        _task_row(succ_id, 0, "aic", dispatch=30.0, start=31.0, end=40.0, receive=30.5),
        _task_row(succ_id, 24, "aiv", dispatch=30.1, start=31.1, end=40.1, receive=30.6),
        _task_row(succ_id, 27, "aiv", dispatch=30.3, start=31.3, end=40.3, receive=30.8),
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
