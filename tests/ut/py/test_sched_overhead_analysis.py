# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for sched_overhead_analysis: overhead model, aicore switch, Head/Tail OH."""

from simpler_setup.tools.sched_overhead_analysis import (
    aicore_switch_stats,
    build_task_graph,
    compute_critical_path,
    compute_head_tail,
    compute_overhead,
    parse_scheduler_from_json_phases,
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


def _gtask(tid, core_id, dispatch, start, end, finish, core_type="aic"):
    t = _task(core_id, dispatch, start, end, finish, core_type)
    t["task_id"] = tid
    return t


def test_overhead_idle_core_with_ready_undispatched():
    # P[0,5] (root). C depends on P -> ready=P.end=5, but dispatched at 8. During
    # [5,8] the AIC core is idle and C is ready+undispatched -> overhead 3us.
    tasks = [_gtask(1, 0, 0, 0, 5, 6), _gtask(2, 0, 8, 8, 15, 16)]
    deps = {"edges": [{"pred": 1, "succ": 2}]}
    oh = compute_overhead(tasks, deps, 0.0, 15.0)
    assert abs(oh["overhead_by_type"]["aic"] - 3.0) < 1e-6
    assert abs(oh["has_overhead"] - 3.0) < 1e-6


def test_overhead_offperf_pred_falls_back_to_dispatch():
    # The only predecessor (99) is absent from the perf set. ready must NOT be w0
    # (that over-counts early overhead) — it falls back to the task's dispatch, so
    # [ready, dispatch] is empty and the task contributes no overhead.
    tasks = [_gtask(2, 0, 8, 8, 15, 16)]
    deps = {"edges": [{"pred": 99, "succ": 2}]}
    oh = compute_overhead(tasks, deps, 0.0, 15.0)
    assert oh["overhead_by_type"]["aic"] == 0.0


def test_overhead_mix_task_credits_both_engines():
    # MIX task (tid=2, records on BOTH aic+aiv) depends on aic producer P[0,5];
    # ready=5, dispatched 8. During [5,8] an idle aic core AND an idle aiv core
    # both see it as ready work -> both engines overhead simultaneously.
    p = _gtask(1, 0, 0, 0, 5, 6, core_type="aic")
    m_aic = _gtask(2, 0, 8, 8, 15, 16, core_type="aic")
    m_aiv = _gtask(2, 10, 10, 8, 15, 16, core_type="aiv")  # same tid=2
    deps = {"edges": [{"pred": 1, "succ": 2}]}
    oh = compute_overhead([p, m_aic, m_aiv], deps, 0.0, 15.0)
    assert abs(oh["overhead_by_type"]["aic"] - 3.0) < 1e-6
    assert abs(oh["overhead_by_type"]["aiv"] - 3.0) < 1e-6
    assert abs(oh["all_overhead"] - 3.0) < 1e-6  # both blocked together


def test_aicore_switch_per_core_and_bound():
    # AIC core0: A[0,10] then B (dispatch=8 < A.end=10) -> switch gap [10,12]=2us.
    # AIV core1: one task, no switch. Bound: lower=min over cores=0, upper=2+0=2.
    tasks = [
        _gtask(1, 0, 0, 0, 10, 11),
        _gtask(2, 0, 8, 12, 20, 21),
        _gtask(3, 1, 0, 0, 10, 11, core_type="aiv"),
    ]
    oh = compute_overhead(tasks, {"edges": []}, 0.0, 20.0)
    per_core, events, split = aicore_switch_stats(tasks, oh["ready_steps"], 0.0, 20.0)
    assert abs(per_core["aic"][0] - 2.0) < 1e-6
    assert per_core["aiv"][1] == 0.0
    assert len(events["aic"]) == 1
    lower = min(min(v.values()) for v in per_core.values() if v)
    upper = sum(min(v.values()) for v in per_core.values() if v)
    assert lower == 0.0 and abs(upper - 2.0) < 1e-6
    # no other ready work during the gap -> the switch is 'independent'
    assert abs(split["aic"][1] - 2.0) < 1e-6 and split["aic"][0] == 0.0


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


def test_graph_prepare_is_work_not_reconstructed_idle():
    data = {
        "aicpu_scheduler_phases": [
            [
                {"phase": "complete", "start_time_us": 0.0, "end_time_us": 1.0, "loop_iter": 1},
                {
                    "phase": "graph_prepare",
                    "start_time_us": 1.0,
                    "end_time_us": 3.0,
                    "tasks_processed": 4,
                    "loop_iter": 1,
                },
                {"phase": "dispatch", "start_time_us": 3.0, "end_time_us": 4.0, "loop_iter": 2},
            ]
        ]
    }

    thread = parse_scheduler_from_json_phases(data)[0]
    assert thread["graph_prepare_us"] == 2.0
    assert thread["idle_us"] == 0.0
    assert thread["total_us"] == 4.0
