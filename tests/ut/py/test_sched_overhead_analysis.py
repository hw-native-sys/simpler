# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for sched_overhead_analysis Head/Tail OH + compute-core utilization."""

from simpler_setup.tools.sched_overhead_analysis import (
    build_task_graph,
    compute_critical_path,
    compute_head_tail,
    compute_utilization,
    decompose_core_capacity,
    per_id_timing,
    print_distribution,
)


def _task(core_id, dispatch, start, end, finish, core_type="aic"):
    return {
        "core_id": core_id,
        "core_type": core_type,
        "dispatch_time_us": float(dispatch),
        "start_time_us": float(start),
        "end_time_us": float(end),
        "finish_time_us": float(finish),
        "duration_us": float(end - start),
    }


def test_head_first_task_uses_start_minus_dispatch():
    heads, tails = compute_head_tail([_task(0, 10, 12, 20, 22)])
    assert heads == [2.0]  # start - dispatch = 12 - 10
    assert tails == [2.0]  # finish - end = 22 - 20


def test_head_min_picks_core_free_when_busy():
    # t1 dispatched at 5 (< t0 end 10): min(start-dispatch=12-5=7, start-last_end=12-10=2) = 2.
    heads, _ = compute_head_tail([_task(0, 0, 1, 10, 11), _task(0, 5, 12, 20, 21)])
    assert heads[0] == 1.0  # first task: 1 - 0
    assert heads[1] == 2.0  # min(7, 2)


def test_head_min_picks_dispatch_when_core_idle():
    # t1 dispatched at 12 (>= t0 end 10): min(start-dispatch=13-12=1, start-last_end=13-10=3) = 1.
    heads, _ = compute_head_tail([_task(0, 0, 1, 10, 11), _task(0, 12, 13, 20, 21)])
    assert heads[1] == 1.0


def test_head_and_tail_clamp_to_zero():
    # Overlap/skew: t1 start 9 < t0 end 10 -> start-last_end = -1 -> min(-) -> clamp 0;
    # finish 15 < end 16 -> tail -1 -> clamp 0.
    heads, tails = compute_head_tail([_task(0, 0, 1, 10, 11), _task(0, 5, 9, 16, 15)])
    assert heads[1] == 0.0
    assert tails[1] == 0.0


def test_cores_independent_for_head():
    heads, _ = compute_head_tail([_task(0, 0, 5, 10, 11), _task(1, 0, 3, 8, 9)])
    # Both are first-on-core -> start - dispatch.
    assert sorted(heads) == [3.0, 5.0]


def test_compute_utilization_by_core_type():
    # window = max(end) - min(start) = 10 - 0 = 10.
    tasks = [
        _task(0, 0, 0, 10, 11, "aic"),
        _task(1, 0, 0, 5, 6, "aic"),
        _task(2, 0, 0, 2, 3, "aiv"),
    ]
    u = compute_utilization(tasks)
    assert u["aic"]["n_cores"] == 2
    assert abs(u["aic"]["busy_us"] - 15.0) < 1e-9
    assert abs(u["aic"]["util"] - 15.0 / (2 * 10)) < 1e-9  # 0.75
    assert u["aiv"]["n_cores"] == 1
    assert abs(u["aiv"]["util"] - 2.0 / (1 * 10)) < 1e-9  # 0.2
    assert u["all"]["n_cores"] == 3
    assert abs(u["all"]["util"] - 17.0 / (3 * 10)) < 1e-9


def test_compute_utilization_empty():
    assert compute_utilization([]) == {}


def _gtask(tid, core_id, dispatch, start, end, finish, core_type="aic"):
    t = _task(core_id, dispatch, start, end, finish, core_type)
    t["task_id"] = tid
    return t


def test_capacity_sums_to_k_times_window():
    # A (root) -> B on one AIC core. Scheduler leaves the core idle [5,7] while
    # B (ready at A.end=5) waits = Sched-starve. Buckets must sum to K * window.
    tasks = [_gtask(1, 0, 0, 0, 5, 6), _gtask(2, 0, 7, 7, 12, 13)]
    deps = {"edges": [{"pred": 1, "succ": 2}]}
    w0, w1 = 0.0, 12.0
    ready, gating, *_ = build_task_graph(tasks, deps, w0)
    d = decompose_core_capacity(tasks, ready, gating, "aic", 1, w0, w1)
    total = d["busy"] + d["sched_starve"] + d["dep_same"] + d["dep_cross"]
    assert abs(total - 1 * (w1 - w0)) < 1e-6
    assert abs(d["busy"] - 10.0) < 1e-6
    assert abs(d["sched_starve"] - 2.0) < 1e-6  # [5,7] idle with B ready


def test_capacity_dep_cross_when_blocked_on_other_engine():
    # An AIV task V0 produces; an AIC task C1 depends on it. While V0 runs, the
    # idle AIC core is blocked on a cross-type producer -> dep_cross.
    v0 = _gtask(1, 50, 0, 0, 10, 11, core_type="aiv")
    c1 = _gtask(2, 1, 10, 10, 15, 16, core_type="aic")  # ready at V0.end=10
    tasks = [v0, c1]
    deps = {"edges": [{"pred": 1, "succ": 2}]}
    w0, w1 = 0.0, 15.0
    ready, gating, *_ = build_task_graph(tasks, deps, w0)
    d = decompose_core_capacity(tasks, ready, gating, "aic", 1, w0, w1)
    # AIC core idle [0,10] waiting on AIV producer -> dep_cross; busy [10,15]=5.
    assert abs(d["busy"] - 5.0) < 1e-6
    assert abs(d["dep_cross"] - 10.0) < 1e-6
    assert d["dep_same"] == 0.0


def test_critical_path_splits_exec_vs_scheduler():
    tasks = [_gtask(1, 0, 0, 0, 5, 6), _gtask(2, 0, 7, 7, 12, 13)]  # B deps A; hop gap 7-5=2
    deps = {"edges": [{"pred": 1, "succ": 2}]}
    ready, gating, end_by_id, *_ = build_task_graph(tasks, deps, 0.0)
    dispatch, start, _e, finish = per_id_timing(tasks)
    cp = compute_critical_path({2: {1}}, end_by_id, finish, start, dispatch, 0.0)
    assert cp is not None
    assert cp["hops"] == 1
    assert abs(cp["exec"] - 10.0) < 1e-6  # A 5 + B 5
    assert abs(cp["sched"] - 2.0) < 1e-6  # B.start - A.end


def test_print_distribution(capsys):
    print_distribution("Head OH", [1.0, 2.0, 3.0, 4.0])
    out = capsys.readouterr().out
    assert "Head OH distribution (N=4)" in out
    assert "Mean:" in out and "Total:" in out

    print_distribution("Head OH", [])
    assert "(no tasks)" in capsys.readouterr().out
