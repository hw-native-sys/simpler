# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Tests for sched_overhead_analysis: overhead model, aicore switch, Head/Tail OH."""

import os

from simpler_setup.tools.sched_overhead_analysis import (
    _scheduler_phases_for_report,
    _summarize_scheduler_loops,
    aicore_switch_stats,
    auto_select_chip_swimlane_records_json,
    build_task_graph,
    compute_critical_path,
    compute_head_tail,
    compute_overhead,
    parse_scheduler_from_json_phases,
    per_id_timing,
    print_distribution,
    run_analysis,
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


def test_aicore_scheduler_runs_common_analysis_and_own_phase_breakdown(tmp_path, capsys):
    perf_path = tmp_path / "chip_swimlane_records.json"
    perf_path.write_text("{}")
    deps_path = tmp_path / "deps.json"
    deps_path.write_text('{"edges": []}')
    task = _task(0, 1, 2, 4, 5)
    task["task_id"] = 1
    data = {
        "tasks": [task],
        "scheduler_task_producer": "aicore",
        "scheduler_records": [[{"phase": "ready_claim", "start_time_us": 1.0, "end_time_us": 1.5}]],
        "scheduler_streams": [{"producer": "aicore", "capture": {"committed": 1, "dropped": 0, "truncated": False}}],
    }

    assert run_analysis(perf_path, print_sources=False, perf_data=data, deps_json_path=deps_path) == 0
    output = capsys.readouterr().out
    assert "Part 1: Overhead verdict" in output
    assert "Part 3: Head OH" in output
    assert "Part 4: Tail OH" in output
    assert "Part 5: AICore scheduler phase breakdown" in output
    assert "ready_claim" in output
    assert "Part 6: Critical-path latency attribution" in output
    assert "AICPU scheduler loop breakdown" not in output


def test_level_two_aicore_runs_common_analysis_without_phase_records(tmp_path, capsys):
    perf_path = tmp_path / "chip_swimlane_records.json"
    perf_path.write_text("{}")
    deps_path = tmp_path / "deps.json"
    deps_path.write_text('{"edges": []}')
    task = _task(0, 1, 2, 4, 5)
    task["task_id"] = 1
    data = {
        "tasks": [task],
        "scheduler_task_producer": "aicore",
    }

    assert run_analysis(perf_path, print_sources=False, perf_data=data, deps_json_path=deps_path) == 0
    output = capsys.readouterr().out
    assert "Part 1: Overhead verdict" in output
    assert "Part 5: AICore scheduler phase breakdown" in output
    assert "phase records unavailable at chip-swimlane Level 2" in output
    assert "Part 6: Critical-path latency attribution" in output


def test_level_two_aicpu_runs_common_analysis_without_phase_records(tmp_path, capsys):
    perf_path = tmp_path / "chip_swimlane_records.json"
    perf_path.write_text("{}")
    deps_path = tmp_path / "deps.json"
    deps_path.write_text('{"edges": []}')
    task = _task(0, 1, 2, 4, 5)
    task["task_id"] = 1
    data = {
        "tasks": [task],
        "scheduler_task_producer": "aicpu",
    }

    assert run_analysis(perf_path, print_sources=False, perf_data=data, deps_json_path=deps_path) == 0
    output = capsys.readouterr().out
    assert "Part 1: Overhead verdict" in output
    assert "Part 5: AICPU scheduler loop breakdown" in output
    assert "phase records unavailable at chip-swimlane Level 2" in output
    assert "Part 6: Critical-path latency attribution" in output


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
    assert cp["sched"] + cp["exec"] <= cp["span"]


def test_critical_path_span_includes_root_dispatch_head():
    tasks = [_gtask(1, 0, 1, 3, 8, 9)]
    _ready, _gating, end_by_id, *_ = build_task_graph(tasks, {"edges": []}, 3.0)
    dispatch, start, _end, finish = per_id_timing(tasks)

    cp = compute_critical_path({}, end_by_id, finish, start, dispatch, 3.0)

    assert cp == {"hops": 0, "span": 8.0, "sched": 2.0, "exec": 5.0}


def test_print_distribution(capsys):
    print_distribution("Head OH", [1.0, 2.0, 3.0, 4.0])
    out = capsys.readouterr().out
    assert "Head OH distribution (N=4)" in out
    assert "Mean:" in out and "Total:" in out

    print_distribution("Head OH", [])
    assert "(no tasks)" in capsys.readouterr().out


def test_parse_scheduler_distinguishes_logical_tasks_from_finishes():
    task_id = (1 << 32) | 7
    data = {
        "core_to_thread": [0, 0],
        "tasks": [
            {"task_id": task_id, "core_id": 0, "finish_time_us": 1.25},
            {"task_id": task_id, "core_id": 1, "finish_time_us": 1.75},
        ],
        "aicpu_scheduler_phases": [
            [
                {
                    "phase": "complete",
                    "start_time_us": 1.0,
                    "end_time_us": 2.0,
                    "loop_iter": 1,
                    "tasks_processed": 2,
                }
            ]
        ],
    }

    threads = parse_scheduler_from_json_phases(data)

    assert threads[0]["completed"] == 1
    assert threads[0]["logical_tasks"] == 1
    assert threads[0]["finishes"] == 2
    assert threads[0]["tasks_per_loop"] == 1
    assert threads[0]["finishes_per_loop"] == 2


def test_parse_scheduler_attributes_spmd_task_to_final_finish_thread():
    task_id = (1 << 32) | 7
    data = {
        "core_to_thread": [0, 1],
        "tasks": [
            {"task_id": task_id, "core_id": 0, "finish_time_us": 1.25},
            {"task_id": task_id, "core_id": 1, "finish_time_us": 1.75},
        ],
        "aicpu_scheduler_phases": [
            [
                {
                    "phase": "complete",
                    "start_time_us": 1.0,
                    "end_time_us": 2.0,
                    "loop_iter": 1,
                    "tasks_processed": 1,
                }
            ],
            [
                {
                    "phase": "complete",
                    "start_time_us": 1.0,
                    "end_time_us": 2.0,
                    "loop_iter": 1,
                    "tasks_processed": 1,
                }
            ],
        ],
    }

    threads = parse_scheduler_from_json_phases(data)

    assert threads[0]["completed"] == 0
    assert threads[0]["logical_tasks"] == 0
    assert threads[0]["finishes"] == 1
    assert threads[0]["tasks_per_loop"] == 0
    assert threads[0]["finishes_per_loop"] == 1
    assert threads[1]["completed"] == 1
    assert threads[1]["logical_tasks"] == 1
    assert threads[1]["finishes"] == 1
    assert threads[1]["tasks_per_loop"] == 1
    assert threads[1]["finishes_per_loop"] == 1


def test_parse_scheduler_counts_hbg_p_thread_standalone_phases():
    data = {
        "aicpu_scheduler_phases": [
            [
                {"phase": "resolve", "start_time_us": 1.0, "end_time_us": 2.0, "loop_iter": 7},
                {"phase": "async_poll", "start_time_us": 3.0, "end_time_us": 5.0, "loop_iter": 9},
                {"phase": "dummy", "start_time_us": 6.0, "end_time_us": 7.0, "loop_iter": 9},
            ]
        ]
    }

    threads = parse_scheduler_from_json_phases(data)

    assert threads[0]["resolve_us"] == 1.0
    assert threads[0]["async_poll_us"] == 2.0
    assert threads[0]["dummy_us"] == 1.0
    assert threads[0]["idle_us"] == 2.0
    assert threads[0]["total_us"] == 6.0
    assert threads[0]["loops"] == 9
    assert threads[0]["role"] == "resolution"
    assert threads[0]["phases_seen"] == {"resolve", "async_poll", "dummy", "idle"}


def test_parse_scheduler_classifies_release_as_scheduler_work():
    data = {
        "aicpu_scheduler_phases": [
            [
                {
                    "phase": "release",
                    "start_time_us": 1.0,
                    "end_time_us": 2.0,
                    "loop_iter": 7,
                    "tasks_processed": 5,
                },
                {"phase": "resolve", "start_time_us": 2.0, "end_time_us": 3.0, "loop_iter": 8},
            ]
        ]
    }

    threads = parse_scheduler_from_json_phases(data)

    assert threads[0]["role"] == "scheduler"
    assert threads[0]["release_us"] == 1.0
    assert threads[0]["phases_seen"] == {"release", "resolve"}


def test_parse_scheduler_uses_explicit_hbg_resolve_discriminator_at_parent_boundary():
    data = {
        "aicpu_scheduler_phases": [
            [
                {"phase": "dummy", "start_time_us": 1.0, "end_time_us": 2.0, "loop_iter": 1},
                {
                    "phase": "resolve_standalone",
                    "start_time_us": 2.0,
                    "end_time_us": 2.0,
                    "loop_iter": 2,
                },
            ]
        ]
    }

    threads = parse_scheduler_from_json_phases(data)

    assert threads[0]["role"] == "resolution"
    assert threads[0]["phases_seen"] == {"dummy", "resolve"}


def test_parse_scheduler_treats_legacy_resolve_touching_parent_boundary_as_standalone():
    data = {
        "aicpu_scheduler_phases": [
            [
                {"phase": "complete", "start_time_us": 1.0, "end_time_us": 2.0, "loop_iter": 1},
                {"phase": "resolve", "start_time_us": 2.0, "end_time_us": 2.0, "loop_iter": 2},
            ]
        ]
    }

    threads = parse_scheduler_from_json_phases(data)

    assert threads[0]["phases_seen"] == {"complete", "resolve"}


def test_parse_scheduler_does_not_double_count_tmr_nested_resolve():
    data = {
        "aicpu_scheduler_phases": [
            [
                {
                    "phase": "complete",
                    "start_time_us": 1.0,
                    "end_time_us": 5.0,
                    "loop_iter": 3,
                    "tasks_processed": 1,
                },
                {"phase": "resolve", "start_time_us": 2.0, "end_time_us": 4.0, "loop_iter": 3},
                {"phase": "dummy", "start_time_us": 6.0, "end_time_us": 9.0, "loop_iter": 4},
                {"phase": "resolve", "start_time_us": 7.0, "end_time_us": 8.0, "loop_iter": 4},
            ]
        ]
    }

    threads = parse_scheduler_from_json_phases(data)

    assert threads[0]["complete_us"] == 4.0
    assert threads[0]["dummy_us"] == 3.0
    assert threads[0]["resolve_us"] == 0.0
    assert threads[0]["idle_us"] == 1.0
    assert threads[0]["total_us"] == 8.0
    assert threads[0]["role"] == "scheduler"
    assert threads[0]["phases_seen"] == {"complete", "dummy", "idle"}


def test_parse_scheduler_does_not_double_count_tmr_resolve_starting_with_its_parent():
    # TMR stamps a Dummy bar's start and the first dummy's Resolve two
    # get_sys_cnt_aicpu() reads apart, so on a2a3's 20 ns sys-cnt tick they
    # routinely coincide. The Resolve is still nested and still excluded.
    data = {
        "aicpu_scheduler_phases": [
            [
                {
                    "phase": "dummy",
                    "start_time_us": 1.0,
                    "end_time_us": 5.0,
                    "loop_iter": 1,
                    "tasks_processed": 2,
                },
                {"phase": "resolve", "start_time_us": 1.0, "end_time_us": 3.0, "loop_iter": 1},
                {
                    "phase": "dispatch",
                    "start_time_us": 5.0,
                    "end_time_us": 6.0,
                    "loop_iter": 1,
                    "tasks_processed": 1,
                },
            ]
        ]
    }

    threads = parse_scheduler_from_json_phases(data)

    assert threads[0]["dummy_us"] == 4.0
    assert threads[0]["resolve_us"] == 0.0
    assert threads[0]["total_us"] == 5.0
    assert threads[0]["role"] == "scheduler"


def test_scheduler_loop_summary_keeps_scheduler_and_resolution_rates_separate():
    threads = {
        0: {"role": "scheduler", "total_us": 100.0, "loops": 10, "completed": 2},
        1: {"role": "scheduler", "total_us": 300.0, "loops": 30, "completed": 6},
        2: {"role": "resolution", "total_us": 20.0, "loops": 200, "completed": 0},
    }

    summary = _summarize_scheduler_loops(threads)

    assert summary["scheduler"] == {
        "total_us": 400.0,
        "loops": 40,
        "completed": 8,
        "avg_loop_us": 10.0,
    }
    assert summary["resolution"] == {
        "total_us": 20.0,
        "loops": 200,
        "completed": 0,
        "avg_loop_us": 0.1,
    }


def test_scheduler_phase_report_suppresses_absent_runtime_phases():
    threads = {
        0: {"phases_seen": {"complete", "dispatch", "idle"}},
        1: {"phases_seen": {"resolve", "async_poll"}},
    }

    assert _scheduler_phases_for_report(threads) == ["complete", "async_poll", "dispatch", "resolve", "idle"]


def test_auto_select_reaches_both_the_l2_and_the_l3_capture_depths(tmp_path, monkeypatch):
    # An L2 case writes its records at outputs/<case>/, an L3 case writes one
    # per chip below outputs/<case>/rankN/dN/ — two levels deeper. A depth-fixed
    # glob resolves only one of them and calls the other run "no records".
    outputs = tmp_path / "outputs"
    l2_records = outputs / "case_l2" / "chip_swimlane_records.json"
    l3_records = outputs / "case_l3" / "rank1" / "d0" / "chip_swimlane_records.json"
    for path in (l2_records, l3_records):
        path.parent.mkdir(parents=True)
        path.write_text("{}")
    monkeypatch.chdir(tmp_path)

    os.utime(l2_records, (1_000, 1_000))
    os.utime(l3_records, (2_000, 2_000))
    assert auto_select_chip_swimlane_records_json() == l3_records

    os.utime(l2_records, (3_000, 3_000))
    assert auto_select_chip_swimlane_records_json() == l2_records
