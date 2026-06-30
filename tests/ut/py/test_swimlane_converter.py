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
    flow = _first_worker_dependency_flow(out)
    assert flow[0]["output_task_count"] == 1
    assert flow[0]["input_task_count"] == 4
    assert flow[1]["tid"] == _core_tid(0)
    assert flow[1]["ts"] == tasks[1]["receive_time_us"]


def test_spmd_to_spmd_one_edge_on_min_core_subtask(tmp_path):
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


def test_dependency_flow_anchor_rows_picks_min_core_per_type():
    task_map = {
        1: [
            _task_row(1, 5, "aiv"),
            _task_row(1, 0, "aiv"),
            _task_row(1, 2, "aic"),
            _task_row(1, 7, "aic"),
        ]
    }
    rows = sc._dependency_flow_anchor_rows(1, task_map, {1})
    assert len(rows) == 2
    by_type = {r["core_type"]: r["core_id"] for r in rows}
    assert by_type == {"aic": 2, "aiv": 0}


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
