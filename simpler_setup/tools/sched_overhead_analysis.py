#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scheduler overhead analysis for PTO2 — is the scheduler the bottleneck, or starved?

Inputs (BOTH required, captured in SEPARATE runs — do not co-run the flags, as
dep_gen perturbs the swimlane timing):
  1. Per-task perf profiling data (l2_swimlane_records_*.json) with
     ``aicpu_scheduler_phases``, from a ``--enable-l2-swimlane`` (level >= 3) run.
  2. deps.json (the task DAG) from a separate ``--enable-dep-gen`` run. It drives
     ready(C) = max(producer.end), which separates scheduler bubbles from
     dependency stalls. Required — the report errors without it.

Report (see docs/dfx/sched-overhead-model.md for the model):
  Part 1 Overhead verdict (per-engine + system all/has overhead, % of makespan) |
  Part 2 aicore switch (per-core pickup totals + makespan bound) |
  Part 3/4 Head/Tail OH distributions | Part 5 scheduler loop budget |
  Part 6 critical-path attribution.

Usage:
    python -m simpler_setup.tools.sched_overhead_analysis \\
        --l2-swimlane-records-json <swimlane.json> --deps-json <deps.json>
"""

import argparse
import bisect
import json
import sys
from collections import defaultdict
from pathlib import Path


def _to_uint64(v):
    """Coerce a JSON-encoded uint64 (int, or string — deps.json quotes uint64s
    so JavaScript-based consumers don't lose precision past 2^53 - 1) to a
    Python int. Returns None when unparseable."""
    try:
        n = int(v)
    except (TypeError, ValueError):
        return None
    if n < 0:
        n &= (1 << 64) - 1
    return n


def compute_dag_stats_from_deps(deps_data, perf_data, threads):
    """Annotate ``threads`` in-place with fanout / fanin per-thread aggregates
    derived from deps.json edges + per-task scheduling-thread attribution.

    Why this lives in Python and not the runtime: the DAG edge set is already
    captured structurally by dep_gen (deps.json), and the per-task → scheduler-
    thread map is in ``l2_swimlane_records.json::core_to_thread``. Re-instrumenting
    the AICPU to track fanout edge counts is duplicate work; running this in
    Python over the existing artifacts is cheaper, more accurate (deps.json
    captures #599 race-window edges that fanout[] dropped), and lets the
    analysis work on default builds that don't have SIMPLER_SCHED_PROFILING=1.

    Edge dedup: deps.json may carry multiple records for the same (pred, succ)
    pair (different ``source``: explicit / creator / tensormap). The runtime
    fanin builder dedups to a single edge per pair, so this function projects
    onto unique (pred, succ) before counting — yields a count that matches the
    runtime's view, not the structural superset.

    Per-thread attribution: each completed task has a ``core_id``; the JSON's
    top-level ``core_to_thread`` maps that to a scheduler thread index. fanout
    work for task T is billed to the thread that retired T (where on_mixed_
    task_complete walks T's fanout list). fanin work for the consumers it makes
    ready is tightly coupled in time and billed to the same thread.

    Mutates ``threads`` dict so existing report code that reads
    ``t["fanout_edges"]`` / ``["fanout_max_degree"]`` / ``["fanin_edges"]`` /
    ``["fanin_max_degree"]`` works unchanged.
    """
    if not isinstance(deps_data, dict) or not isinstance(perf_data, dict):
        return
    raw_edges = deps_data.get("edges", [])

    # Unique (pred, succ) — collapses explicit + creator + tensormap variants.
    edges_by_pred = defaultdict(set)
    edges_by_succ = defaultdict(set)
    for e in raw_edges:
        if not isinstance(e, dict):
            continue
        pred = _to_uint64(e.get("pred"))
        succ = _to_uint64(e.get("succ"))
        if pred is None or succ is None:
            continue
        edges_by_pred[pred].add(succ)
        edges_by_succ[succ].add(pred)

    # core_id → scheduler-thread mapping (from runtime header passthrough).
    core_to_thread = perf_data.get("core_to_thread") or []

    def task_thread(task):
        """Map a perf task record to its scheduler-thread index, or None when
        unattributable."""
        cid = task.get("core_id")
        if not isinstance(cid, int):
            return None
        if 0 <= cid < len(core_to_thread):
            tidx = core_to_thread[cid]
            return tidx if tidx >= 0 else None
        return None

    # Per-thread accumulators. Tasks that don't map to a known thread fall
    # through into an "unattributed" bucket so the run total still reconciles
    # against the raw edge counts in deps.json.
    per_thread_fanout = defaultdict(lambda: {"edges": 0, "max": 0, "tasks": 0})
    per_thread_fanin = defaultdict(lambda: {"edges": 0, "max": 0, "tasks": 0})

    # Dedup by task_id: mixed (AIC+AIV) tasks emit one perf row per subtask /
    # core (see l2_swimlane_collector.cpp:567 — collected_perf_records_ is keyed by
    # core_idx). Without dedup a mixed task's fanout would be charged once per
    # subtask, inflating per-thread edge counts by the subtask count.
    seen_task_ids = set()
    for task in perf_data.get("tasks", []):
        tid_raw = _to_uint64(task.get("task_id"))
        if tid_raw is None or tid_raw in seen_task_ids:
            continue
        seen_task_ids.add(tid_raw)
        thr = task_thread(task)
        if thr is None:
            continue
        out_count = len(edges_by_pred.get(tid_raw, ()))
        in_count = len(edges_by_succ.get(tid_raw, ()))
        fo = per_thread_fanout[thr]
        fi = per_thread_fanin[thr]
        fo["edges"] += out_count
        fo["max"] = max(fo["max"], out_count)
        fo["tasks"] += 1
        fi["edges"] += in_count
        fi["max"] = max(fi["max"], in_count)
        fi["tasks"] += 1

    for thr_idx, fo in per_thread_fanout.items():
        t = threads.get(thr_idx)
        if t is None:
            continue
        t["fanout_edges"] = fo["edges"]
        t["fanout_max_degree"] = fo["max"]
    for thr_idx, fi in per_thread_fanin.items():
        t = threads.get(thr_idx)
        if t is None:
            continue
        t["fanin_edges"] = fi["edges"]
        t["fanin_max_degree"] = fi["max"]


def auto_select_l2_swimlane_records_json():
    """Find the latest outputs/<case>/l2_swimlane_records.json (sorted by mtime)."""
    outputs_dir = Path.cwd() / "outputs"
    files = sorted(outputs_dir.glob("*/l2_swimlane_records.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No outputs/*/l2_swimlane_records.json found under {outputs_dir}")
    return files[0]


def parse_scheduler_from_json_phases(data):
    """Extract scheduler Phase breakdown from l2_swimlane_records JSON.

    Computes per-thread loop counts, task counts, and phase totals
    from aicpu_scheduler_phases records (present at l2_swimlane_level >= 3).

    Returns:
        dict: Thread data keyed by thread index, with per-phase us / pct,
              pop_hit / pop_miss, loops, completed, tasks_per_loop. Returns
              empty dict if phase data is not available.
    """
    phases_by_thread = data.get("aicpu_scheduler_phases", [])
    if not phases_by_thread:
        return {}

    threads = {}
    for tid, records in enumerate(phases_by_thread):
        if not records:
            continue

        # Keep every mutually-exclusive outer work phase. Inner phases such
        # as resolve/drain_prepare overlap their parent and separate-lane
        # dummy_task markers are not scheduler occupancy, so exclude those.
        # Legacy explicit idle/scan records are also excluded because idle is
        # reconstructed from gaps between outer work records.
        outer_phases = (
            "complete",
            "async_poll",
            "dispatch",
            "release",
            "dummy",
            "early_dispatch",
            "drain",
            "graph_prepare",
        )
        work_recs = sorted(
            (r for r in records if r.get("phase") in outer_phases),
            key=lambda r: r.get("start_time_us", 0),
        )
        if not work_recs:
            continue

        phase_us = {phase: 0.0 for phase in outer_phases}
        phase_us["idle"] = 0.0
        total_tasks = 0
        max_loop_iter = 0
        pop_hit = 0
        pop_miss = 0
        prev_end = None

        for rec in work_recs:
            phase = rec["phase"]
            start = rec.get("start_time_us", 0)
            end = rec.get("end_time_us", 0)
            # Idle = wall-clock gap between this record and the previous
            # work record on this thread. Pre-first-work and post-last-work
            # spans are not captured here (see device's cumulative
            # sched_idle_cycle in the cold-path summary log for those).
            if prev_end is not None and start > prev_end:
                phase_us["idle"] += start - prev_end
            dur = end - start
            if dur > 0:
                phase_us[phase] += dur
            if phase == "complete":
                total_tasks += rec.get("tasks_processed", 0)
            # Per-emit queue-health deltas; only present on dispatch records.
            # Summing across records gives the run-cumulative pop_hit /
            # pop_miss (the runtime's final-drain emit closes the tail).
            if phase == "dispatch":
                pop_hit += rec.get("pop_hit", 0)
                pop_miss += rec.get("pop_miss", 0)
            max_loop_iter = max(max_loop_iter, rec.get("loop_iter", 0))
            if prev_end is None or end > prev_end:
                prev_end = end

        total_us = sum(phase_us.values())
        loops = max_loop_iter
        tasks_per_loop = total_tasks / loops if loops > 0 else 0.0
        pop_total = pop_hit + pop_miss
        pop_hit_rate = pop_hit / pop_total * 100 if pop_total > 0 else 0.0

        t = {
            "completed": total_tasks,
            "total_us": total_us,
            "loops": loops,
            "tasks_per_loop": tasks_per_loop,
            "pop_hit": pop_hit,
            "pop_miss": pop_miss,
            "pop_hit_rate": pop_hit_rate,
            "format": "json_phase",
        }
        for p, us in phase_us.items():
            t[f"{p}_us"] = us
            t[f"{p}_pct"] = us / total_us * 100 if total_us > 0 else 0

        threads[tid] = t

    return threads


def validate_perf_tasks_for_overhead_analysis(tasks):
    """Validate required per-task fields for overhead deep-dive analysis.

    Returns:
        tuple[bool, str]: (is_valid, error_message)
    """
    required_fields = [
        "core_id",
        "duration_us",
        "start_time_us",
        "end_time_us",
        "dispatch_time_us",
        "finish_time_us",
    ]

    missing = []
    for idx, task in enumerate(tasks):
        missing_fields = [field for field in required_fields if field not in task]
        if missing_fields:
            task_label = task.get("task_id", idx)
            missing.append(f"task={task_label} missing={','.join(missing_fields)}")
            if len(missing) >= 5:
                break

    if missing:
        detail = "; ".join(missing)
        # These fields are produced by runtime-side JSON export in:
        # src/platform/shared/host/performance_collector.cpp (dispatch_time_us, finish_time_us)
        msg = "\n".join(
            [
                "Perf JSON is incompatible with scheduler overhead deep-dive analysis.",
                f"Missing required fields (showing up to 5 tasks): {detail}",
                "",
                "Why this happens:",
                "  - The input is not a runtime-generated l2_swimlane_records_*.json, OR",
                "  - The runtime binary does not include / emit dispatch+finish timestamps.",
                "",
                "How to fix:",
                "  1) Re-run workload with profiling enabled (e.g. run_example.py --enable-l2-swimlane).",
                "  2) Pass the newly generated outputs/<case>/l2_swimlane_records.json via --l2-swimlane-records-json.",
                "  3) Verify each task includes dispatch_time_us and finish_time_us.",
                "",
                "Note:",
                "  - swimlane_converter conversion can still succeed; only deep-dive analysis requires these fields.",
            ]
        )
        return False, msg

    return True, ""


def compute_head_tail(tasks):
    """Per-task Head OH and Tail OH (microseconds), clamped at 0.

    ``head = min(start - dispatch, start - last_task_end)`` when a previous task
    ran on the same core, else ``start - dispatch``. The ``min`` auto-selects
    the right case with no running/pending classification: when the core was
    idle at dispatch (``dispatch >= last_task_end``) it picks ``start - dispatch``
    (dispatch latency); when the core was still busy it picks
    ``start - last_task_end`` (wait from core-free), excluding the queue wait.

    ``tail = finish - end``. Both clamp at 0 — dual-issue / concurrent execution
    can make ``start < last_task_end``, and AICore↔AICPU clock skew can make a
    raw delta negative.

    ``last_task_end`` is the previous same-core task's ``end_time_us`` in
    start-time order. Returns ``(heads, tails)`` — flat per-task µs lists.
    """
    by_core = defaultdict(list)
    for t in tasks:
        by_core[t["core_id"]].append(t)
    heads, tails = [], []
    for core_tasks in by_core.values():
        core_tasks.sort(key=lambda t: t["start_time_us"])
        last_end = None
        for t in core_tasks:
            head = t["start_time_us"] - t["dispatch_time_us"]
            if last_end is not None:
                head = min(head, t["start_time_us"] - last_end)
            heads.append(max(0.0, head))
            tails.append(max(0.0, t["finish_time_us"] - t["end_time_us"]))
            last_end = t["end_time_us"]
    return heads, tails


def print_distribution(label, vals):
    """Print percentile distribution + mean/total for a list of µs values."""
    vals = sorted(vals)
    n = len(vals)
    if n == 0:
        print(f"  {label}: (no tasks)")
        return
    print(f"  {label} distribution (N={n}):")
    for p in (10, 25, 50, 75, 90, 95, 99):
        idx = min(int(n * p / 100), n - 1)
        print(f"    P{p:<4} {vals[idx]:>8.2f} us")
    print(f"    Max:  {vals[-1]:>8.2f} us")
    print(f"    Mean: {sum(vals) / n:>8.2f} us   Total: {sum(vals):>10.1f} us")


def build_task_graph(tasks, deps_data, window_start):
    """Per-task input-ready time + producer info from the DAG.

    ``ready(C) = max over C's producers of producer end_time``; a task with no
    producer in the perf set (root) is ready at ``window_start``. Readiness keys
    off producer *end* (the kernel wrote its output to memory — the data exists),
    NOT producer *finish* (when the AICPU later *observed* completion). The gap
    between a producer's end and finish is the AICPU's completion-detection
    latency — a scheduler cost, not a data-availability wait — so counting it as
    dependency would hide scheduler overhead. With instant detect+dispatch a
    consumer could start at producer end, so any idle after that is the
    scheduler's. Mixed tasks (one record per subtask/core) fold to a single end.

    Returns dicts keyed by uint64 task_id:
      ready_by_id    : input-ready time (us)
      gating_type_by_id : core_type of the producer that *gates* ready (max end)
      end_by_id, type_by_id
    """
    end_by_id, type_by_id = {}, {}
    for t in tasks:
        tid = _to_uint64(t.get("task_id"))
        if tid is None:
            continue
        end_by_id[tid] = max(end_by_id.get(tid, t["end_time_us"]), t["end_time_us"])
        type_by_id.setdefault(tid, t.get("core_type"))

    preds_by_id = defaultdict(set)
    for e in (deps_data or {}).get("edges", []):
        if not isinstance(e, dict):
            continue
        p, s = _to_uint64(e.get("pred")), _to_uint64(e.get("succ"))
        if p is not None and s is not None and p != s:
            preds_by_id[s].add(p)

    ready_by_id, gating_type_by_id = {}, {}
    for tid in end_by_id:
        ends = [(end_by_id[p], type_by_id.get(p)) for p in preds_by_id.get(tid, ()) if p in end_by_id]
        if ends:
            end_max, gate_type = max(ends, key=lambda x: x[0])
            ready_by_id[tid] = end_max
            gating_type_by_id[tid] = gate_type
        else:
            ready_by_id[tid] = window_start
            gating_type_by_id[tid] = None
    return ready_by_id, gating_type_by_id, end_by_id, type_by_id, preds_by_id


def compute_overhead(tasks, deps_data, w0, w1):  # noqa: PLR0912
    """Dependency- and MIX-aware overhead decomposition over the makespan.

    At each instant, for each core type T, T is *overhead* when an idle T-core
    exists (``k_T - running_T > 0``) AND a ready, undispatched T-task exists
    (``[ready, dispatch]``). Readiness keys off producer ``end_time``; a task
    whose producers are all absent from the perf set falls back to its own
    dispatch (no unverifiable early readiness). A MIX task (records on BOTH
    engines) counts as ready work for BOTH. System aggregates:

      ``all_overhead``  every present engine is overhead (whole chip blocked —
                        e.g. a MIX waiting to launch)
      ``has_overhead``  every engine that HAS ready work is overhead (engines
                        with no ready work are ignored)

    Returns us per bucket: ``window``, ``overhead_by_type[T]``, ``all_overhead``,
    ``has_overhead``; plus ``k_by_type`` and ``ready_steps[T]`` (a (times, cum)
    step function of the undispatched-ready count, for the switch split).
    """
    types = sorted({t.get("core_type") for t in tasks if t.get("core_type")})
    k = {ty: len({t["core_id"] for t in tasks if t.get("core_type") == ty}) for ty in types}

    types_of = defaultdict(set)
    disp, end = {}, {}
    for t in tasks:
        tid = _to_uint64(t.get("task_id"))
        if tid is None:
            continue
        types_of[tid].add(t.get("core_type"))
        disp[tid] = min(disp.get(tid, t["dispatch_time_us"]), t["dispatch_time_us"])
        end[tid] = max(end.get(tid, t["end_time_us"]), t["end_time_us"])

    preds = defaultdict(set)
    for e in (deps_data or {}).get("edges", []):
        if not isinstance(e, dict):
            continue
        p, s = _to_uint64(e.get("pred")), _to_uint64(e.get("succ"))
        if p is not None and s is not None and p != s:
            preds[s].add(p)
    ready = {}
    for tid, dp in disp.items():
        in_perf = [p for p in preds.get(tid, ()) if p in end]
        ready[tid] = max(end[p] for p in in_perf) if in_perf else dp

    run = {ty: defaultdict(int) for ty in types}
    rw = {ty: defaultdict(int) for ty in types}
    times = {w0, w1}
    for t in tasks:
        ty = t.get("core_type")
        if ty not in run:
            continue
        s = max(w0, min(t["start_time_us"], w1))
        e = max(w0, min(t["end_time_us"], w1))
        if e > s:
            run[ty][s] += 1
            run[ty][e] -= 1
            times.update((s, e))
    for tid, dp in disp.items():
        r = max(w0, min(ready[tid], w1))
        dd = max(w0, min(dp, w1))
        if dd > r:
            for ty in types_of[tid]:  # MIX -> credit both engines
                if ty in rw:
                    rw[ty][r] += 1
                    rw[ty][dd] -= 1
            times.update((r, dd))

    out = {
        "window": w1 - w0,
        "k_by_type": k,
        "overhead_by_type": {ty: 0.0 for ty in types},
        "all_overhead": 0.0,
        "has_overhead": 0.0,
    }
    ready_steps = {ty: ([], []) for ty in types}
    order = sorted(times)
    rc = {ty: 0 for ty in types}
    wc = {ty: 0 for ty in types}
    for i in range(len(order) - 1):
        a, b = order[i], order[i + 1]
        for ty in types:
            rc[ty] += run[ty][a]
            wc[ty] += rw[ty][a]
            ready_steps[ty][0].append(a)
            ready_steps[ty][1].append(wc[ty])
        length = b - a
        if length <= 0:
            continue
        ov = {ty: (k[ty] - rc[ty] > 0 and wc[ty] > 0) for ty in types}
        for ty in types:
            if ov[ty]:
                out["overhead_by_type"][ty] += length
        if types and all(ov[ty] for ty in types):
            out["all_overhead"] += length
        work = [ty for ty in types if wc[ty] > 0]
        if work and all(k[ty] - rc[ty] > 0 for ty in work):
            out["has_overhead"] += length
    out["ready_steps"] = ready_steps
    return out


def aicore_switch_stats(tasks, ready_steps, w0, w1):
    """Per-core 'aicore switch' totals + makespan bound + overhead split.

    aicore switch = on a core, the gap ``[prev_end, start]`` of a task whose
    ``dispatch < prev_end`` (pre-dispatched / pending pickup, ~0.8 us each). Per
    core these sum to ~8-11 us; report the PER-CORE totals, never the all-cores
    sum (it reads as a scary aggregate but switches overlap across cores). Each
    switch is split: ``overhead`` when the engine has another undispatched-ready
    task at that instant (the idle pickup core coincides with ready work) vs
    ``independent`` (nothing else ready). Returns ``per_core[T] = {core: us}``,
    ``events[T] = [gap, ...]``, ``split[T] = [overhead_us, independent_us]``.
    """

    def _ready_at(ty, t):
        ts, cum = ready_steps.get(ty, ([], []))
        if not ts or t < ts[0]:
            return 0
        return cum[bisect.bisect_right(ts, t) - 1]

    by_core = defaultdict(list)
    ctype = {}
    for t in tasks:
        by_core[t.get("core_id")].append((t["start_time_us"], t["end_time_us"], t["dispatch_time_us"]))
        ctype.setdefault(t.get("core_id"), t.get("core_type"))

    per_core = defaultdict(dict)
    events = defaultdict(list)
    split = defaultdict(lambda: [0.0, 0.0])  # type -> [overhead_us, independent_us]
    for cid, recs in by_core.items():
        ty = ctype[cid]
        recs.sort()
        tot = 0.0
        for i in range(len(recs) - 1):
            a_end = recs[i][1]
            b_start, b_disp = recs[i + 1][0], recs[i + 1][2]
            gap = min(b_start, w1) - max(a_end, w0)
            if gap <= 0 or not (b_disp < a_end):
                continue
            tot += gap
            events[ty].append(gap)
            ga, gb = max(a_end, w0), min(b_start, w1)
            ts = ready_steps.get(ty, ([], []))[0]
            pts = sorted({ga, gb, *(x for x in ts if ga < x < gb)})
            for j in range(len(pts) - 1):
                aa, bb = pts[j], pts[j + 1]
                if _ready_at(ty, aa) > 0:
                    split[ty][0] += bb - aa
                else:
                    split[ty][1] += bb - aa
        per_core[ty][cid] = tot
    return per_core, events, split


def per_id_timing(tasks):
    """Fold per-core perf records to one timing tuple per task_id.

    Mixed/SPMD tasks emit one record per core; the task's dispatch/start is the
    earliest, its end/finish the latest. Returns dicts keyed by uint64 task_id.
    """
    dispatch, start, end, finish = {}, {}, {}, {}
    for t in tasks:
        tid = _to_uint64(t.get("task_id"))
        if tid is None:
            continue
        dispatch[tid] = min(dispatch.get(tid, t["dispatch_time_us"]), t["dispatch_time_us"])
        start[tid] = min(start.get(tid, t["start_time_us"]), t["start_time_us"])
        end[tid] = max(end.get(tid, t["end_time_us"]), t["end_time_us"])
        finish[tid] = max(finish.get(tid, t["finish_time_us"]), t["finish_time_us"])
    return dispatch, start, end, finish


def compute_critical_path(preds_by_id, end_by_id, finish_by_id, start_by_id, dispatch_by_id, w0):
    """Walk the makespan-determining path and split it into compute vs scheduler.

    Final task = latest finish; backtrack each hop to the gating producer (max
    end). Per hop producer->consumer the scheduler injects
    ``consumer.start - producer.end`` (detect + resolve + head); the root adds
    ``root.start - root.dispatch``. Returns a dict with the path length, the
    scheduler-injected total, and the per-segment sums (detect/resolve/head).
    """
    if not finish_by_id:
        return None
    cur = max(finish_by_id, key=lambda k: finish_by_id[k])
    makespan_end = finish_by_id[cur]
    sched_total = exec_total = 0.0
    hops = 0
    path_start = start_by_id.get(cur, w0)
    while True:
        preds = [(end_by_id[p], p) for p in preds_by_id.get(cur, ()) if p in end_by_id]
        exec_total += max(0.0, end_by_id.get(cur, 0.0) - start_by_id.get(cur, 0.0))
        if not preds:
            # root: dispatch->start head
            sched_total += max(0.0, start_by_id.get(cur, w0) - dispatch_by_id.get(cur, w0))
            path_start = min(path_start, start_by_id.get(cur, w0))
            break
        pend, p = max(preds)
        sched_total += max(0.0, start_by_id.get(cur, 0.0) - pend)  # producer.end -> consumer.start
        hops += 1
        cur = p
        path_start = min(path_start, start_by_id.get(cur, w0))
    span = makespan_end - path_start
    return {"hops": hops, "span": span, "sched": sched_total, "exec": exec_total}


def run_analysis(  # noqa: PLR0912, PLR0915
    l2_swimlane_records_path,
    print_sources=True,
    deps_json_path=None,
    perf_data=None,
):
    """Run scheduler overhead analysis report.

    Args:
        l2_swimlane_records_path: Path to l2_swimlane_records_*.json.
        print_sources: Whether to print selected input files.
        perf_data: Optional pre-parsed perf JSON dict. When provided, skip
            re-reading from disk — main() already parses the file to probe
            for phase data, so passing the result through saves a second
            load on large artifacts.
        deps_json_path: Optional deps.json (dep_gen replay output) co-located
            with the perf JSON. When present, per-thread fanout / fanin
            aggregates are derived from it.

    Returns:
        int: 0 on success, non-zero on failure.
    """
    l2_swimlane_records_path = Path(l2_swimlane_records_path)

    if not l2_swimlane_records_path.exists():
        print(f"Error: Perf JSON not found: {l2_swimlane_records_path}", file=sys.stderr)
        return 1

    # Auto-discover deps.json sibling when caller didn't specify one.
    if deps_json_path is None:
        sibling = l2_swimlane_records_path.parent / "deps.json"
        if sibling.exists():
            deps_json_path = sibling

    if print_sources:
        print(f"Perf data:  {l2_swimlane_records_path}")
        if deps_json_path is not None:
            print(f"Deps JSON:  {deps_json_path}")

    # === Part 1: Per-task time breakdown from perf data ===
    if perf_data is not None:
        data = perf_data
    else:
        # Lazy import to avoid an import cycle: swimlane_converter imports
        # run_analysis from this module at top level. read_perf_data does the
        # AICore↔AICPU join — direct json.load would see only the raw
        # aicore_tasks / aicpu_tasks arrays.
        from .swimlane_converter import read_perf_data  # noqa: PLC0415

        data = read_perf_data(l2_swimlane_records_path)
    tasks = data["tasks"]
    n_total = len(tasks)

    if n_total == 0:
        print("Error: No tasks found in perf data", file=sys.stderr)
        return 1

    valid, err = validate_perf_tasks_for_overhead_analysis(tasks)
    if not valid:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    # Deps (DAG) are REQUIRED: ready(C) = max(producer.end) drives the
    # scheduler-starvation split — without it we can't separate scheduler
    # bubbles from dependency stalls, which is the whole point of this tool.
    # Capture deps.json SEPARATELY with --enable-dep-gen (do NOT co-run with
    # --enable-l2-swimlane: dep_gen perturbs timing).
    if deps_json_path is None:
        print(
            "Error: scheduler-overhead analysis needs the task DAG (deps.json). Capture it in a "
            "SEPARATE run with --enable-dep-gen (not co-run with --enable-l2-swimlane), then pass --deps-json.",
            file=sys.stderr,
        )
        return 1
    try:
        with open(deps_json_path) as df:
            deps_data = json.load(df)
    except (OSError, ValueError) as e:
        print(f"Error: failed to read deps.json {deps_json_path}: {e}", file=sys.stderr)
        return 1

    w0 = min(t["start_time_us"] for t in tasks)
    w1 = max(t["end_time_us"] for t in tasks)
    window = w1 - w0
    _ready_by_id, _gating_type_by_id, end_by_id, _type_by_id, preds_by_id = build_task_graph(tasks, deps_data, w0)
    dispatch_by_id, start_by_id, _end2, finish_by_id = per_id_timing(tasks)
    heads, tails = compute_head_tail(tasks)

    present_types = sorted({t.get("core_type") for t in tasks if t.get("core_type")})
    type_labels = {"aic": "C (AIC)", "aiv": "V (AIV)", "mix": "MIX"}
    oh = compute_overhead(tasks, deps_data, w0, w1)
    per_core, sw_events, sw_split = aicore_switch_stats(tasks, oh["ready_steps"], w0, w1)

    def _pct(x, denom):
        return x / denom * 100 if denom > 0 else 0.0

    all_oh = _pct(oh["all_overhead"], window)
    has_oh = _pct(oh["has_overhead"], window)
    if has_oh >= 25:
        verdict = "SCHEDULER-BOUND — a free core has ready, undispatched work for much of the makespan."
    elif has_oh >= 8:
        verdict = (
            f"PARTIALLY SCHEDULER-LIMITED — overhead {has_oh:.0f}% of the makespan; "
            "the rest is compute + dependency-limited parallelism."
        )
    else:
        verdict = "COMPUTE/DEPENDENCY-BOUND — little reducible scheduler overhead."

    # === Part 1: Overhead verdict (makespan, dependency- & MIX-aware) ===
    print()
    print("=" * 90)
    print("Part 1: Overhead verdict  (idle core + ready, undispatched work — share of makespan)")
    print("=" * 90)
    print(f"  Total tasks: {n_total}   window: {window:.1f} us  (first start -> last end)")
    pe = "  ".join(
        f"{type_labels.get(ty, ty.upper())} {_pct(oh['overhead_by_type'][ty], window):.1f}%" for ty in present_types
    )
    print(f"  Per-engine overhead: {pe}")
    print(
        f"  System: all_overhead {all_oh:.1f}% (every engine blocked)  |  "
        f"has_overhead {has_oh:.1f}% (every working engine starved)"
    )
    print("  overhead(T) = idle T-core AND a ready, undispatched T-task (a MIX task counts for both).")
    print("  An engine with no ready work is NOT overhead — its idle is dependency-mandated progress.")
    print(f"  -> {verdict}")
    print()

    # === Part 2: aicore switch (pre-dispatched pickup) — per-core + makespan bound ===
    print("=" * 90)
    print("Part 2: aicore switch  (gap [prev_end, start] when dispatch < prev_end; PER CORE)")
    print("=" * 90)
    print("  Each switch is the core picking up an already-dispatched task (~sub-us). Reported")
    print("  per core (NOT summed over cores — switches overlap). A switch is 'overhead' when the")
    print("  engine has other ready work that instant, else 'independent'. makespan bound:")
    print("  lower = min over all cores, upper = sum of per-engine minima.")
    print()
    s_fmt = "  {:<10} {:>6} {:>10} {:>10} {:>10} {:>9} {:>9}"
    print(s_fmt.format("Engine", "Cores", "min/core", "mean/core", "max/core", "n events", "us/event"))
    print("  " + "-" * 74)
    type_min = {}
    for ty in present_types:
        pcs = list(per_core.get(ty, {}).values())
        evs = sw_events.get(ty, [])
        if not pcs:
            continue
        type_min[ty] = min(pcs)
        per_ev = (sum(evs) / len(evs)) if evs else 0.0
        print(
            s_fmt.format(
                type_labels.get(ty, ty.upper()),
                len(pcs),
                f"{min(pcs):.1f}us",
                f"{sum(pcs) / len(pcs):.1f}us",
                f"{max(pcs):.1f}us",
                str(len(evs)),
                f"{per_ev:.2f}us",
            )
        )
    print()
    for ty in present_types:
        ov_us, ind_us = sw_split.get(ty, [0.0, 0.0])
        tot = ov_us + ind_us
        if tot > 0:
            print(
                f"    {type_labels.get(ty, ty.upper())} switch split: "
                f"{_pct(ov_us, tot):.0f}% during overhead, {_pct(ind_us, tot):.0f}% independent"
            )
    lower = min((min(v.values()) for v in per_core.values() if v), default=0.0)
    upper = sum(type_min.values())
    print(
        f"  makespan switch bound: [{lower:.2f}, {upper:.2f}] us  "
        f"({_pct(lower, window):.2f}% .. {_pct(upper, window):.2f}% of makespan)"
    )
    print()

    # === Part 3: Head OH statistics & distribution ===
    print("=" * 90)
    print("Part 3: Head OH statistics & distribution")
    print("=" * 90)
    print("  head = min(start - dispatch, start - last_task_end) per task (clamped at 0)")
    print()
    print_distribution("Head OH", heads)
    print()

    # === Part 4: Tail OH distribution ===
    print("=" * 90)
    print("Part 4: Tail OH distribution")
    print("=" * 90)
    print("  tail = finish - end per task")
    print()
    n = len(tails)
    if n == 0:
        print("Error: Empty tail-overhead set", file=sys.stderr)
        return 1
    print_distribution("Tail OH", tails)
    print()

    # === Part 5: AICPU scheduler loop breakdown (+ tail-vs-loop cause analysis) ===
    threads = parse_scheduler_from_json_phases(data)
    if not threads:
        print(
            "Error: perf JSON has no aicpu_scheduler_phases — rerun the case "
            "with --enable-l2-swimlane so phase data is captured.",
            file=sys.stderr,
        )
        return 1

    # Per-thread fanout / fanin from the (already-loaded, required) deps.json.
    dag_stats_available = True
    compute_dag_stats_from_deps(deps_data, data, threads)

    n_threads = len(threads)

    print("=" * 90)
    print("Part 5: AICPU scheduler loop breakdown")
    print(f"  {n_threads} scheduler threads")
    print("=" * 90)
    print()

    fmt2 = "  {:<10} {:>7} {:>10} {:>12} {:>11}"
    print(fmt2.format("Thread", "Loops", "Completed", "ns/loop", "Total (us)"))
    print("  " + "-" * 54)
    for tid in sorted(threads.keys()):
        t = threads[tid]
        ns_per_loop = t["total_us"] * 1000 / t["loops"] if t["loops"] else 0
        print(fmt2.format("T" + str(tid), t["loops"], t["completed"], f"{ns_per_loop:.0f}", f"{t['total_us']:.1f}"))
    total_us = sum(t["total_us"] for t in threads.values())
    total_completed = sum(t["completed"] for t in threads.values())
    total_loops = sum(t["loops"] for t in threads.values())
    avg_ns_per_loop = total_us * 1000 / total_loops if total_loops > 0 else 0
    print(fmt2.format("SUM", total_loops, total_completed, f"{avg_ns_per_loop:.0f}", f"{total_us:.1f}"))
    print()

    # Phase breakdown. Idle is reconstructed from gaps between work
    # records on the same thread (no explicit idle record is emitted by
    # the device anymore).
    phases = [
        "complete",
        "async_poll",
        "dispatch",
        "release",
        "dummy",
        "early_dispatch",
        "drain",
        "graph_prepare",
        "idle",
    ]
    phase_labels = {
        "complete": "Complete (poll handshake, resolve deps)",
        "async_poll": "AsyncPoll (async-wait completion: SDMA/RoCE/URMA/CCU)",
        "dispatch": "Dispatch (pop queue, build payload, flush)",
        "release": "Release (drain deferred task releases)",
        "dummy": "Dummy (complete dependency-only tasks)",
        "early_dispatch": "Early dispatch (speculative staging)",
        "drain": "Drain (sync-start stop-the-world handling)",
        "graph_prepare": "Graph prepare (bounded node materialization)",
        "idle": "Idle (spinning, no progress — reconstructed from gaps)",
    }

    # Total (us) is summed across all scheduler threads, so it can exceed the
    # wall-clock window (e.g. idle ~= n_threads x per-thread idle); "% of total"
    # is each phase's share of that summed scheduler CPU.
    fmt3 = "  {:<55} {:>11} {:>10} {:>14}"
    header = fmt3.format("Phase (summed over threads)", "Total (us)", "% of total", "Avg/task (us)")
    print(header)
    print("  " + "-" * (len(header) - 2))
    phase_totals = {}
    for p in phases:
        key = p + "_us"
        tot = sum(t.get(key, 0) for t in threads.values())
        phase_totals[p] = tot
        pct = tot / total_us * 100 if total_us > 0 else 0
        avg = tot / total_completed if total_completed > 0 else 0
        print(fmt3.format(phase_labels[p], f"{tot:.1f}", f"{pct:.1f}%", f"{avg:.2f}"))
    print()

    # DAG stats (fanout / fanin) — populated only when deps.json was
    # provided. When absent, suppress the rows so missing-artifact zeros
    # aren't mistaken for measurements.
    if dag_stats_available:
        fanout_edges = sum(t.get("fanout_edges", 0) for t in threads.values())
        fanout_max = max((t.get("fanout_max_degree", 0) for t in threads.values()), default=0)
        fanout_avg = fanout_edges / total_completed if total_completed > 0 else 0
        print(
            f"  Fanout (notify consumers): total edges={fanout_edges}, "
            f"max_degree={fanout_max}, avg_degree={fanout_avg:.1f}"
        )
        fanin_edges = sum(t.get("fanin_edges", 0) for t in threads.values())
        fanin_max = max((t.get("fanin_max_degree", 0) for t in threads.values()), default=0)
        fanin_avg = fanin_edges / total_completed if total_completed > 0 else 0
        print(
            f"  Fanin  (release producers): total edges={fanin_edges}, "
            f"max_degree={fanin_max}, avg_degree={fanin_avg:.1f}"
        )
    else:
        fanout_edges = fanout_max = fanin_edges = fanin_max = 0
        fanout_avg = fanin_avg = 0.0
        print("  Fanout / Fanin: (deps.json not provided — pass --deps-json or rerun with --enable-dep-gen)")
    print()

    # Pop stats (per-emit dispatch deltas summed across all dispatch records).
    if any("pop_hit" in t for t in threads.values()):
        pop_hit = sum(t.get("pop_hit", 0) for t in threads.values())
        pop_miss = sum(t.get("pop_miss", 0) for t in threads.values())
        pop_total = pop_hit + pop_miss
        pop_hit_rate = pop_hit / pop_total * 100 if pop_total > 0 else 0
        print(f"  Pop: hit={pop_hit}, miss={pop_miss}, hit_rate={pop_hit_rate:.1f}%")
    else:
        pop_hit = pop_miss = 0
        pop_hit_rate = 0.0
        print("  Pop: (no per-emit pop deltas in input — needs --enable-l2-swimlane at level >= 3)")

    print()
    # Tail-vs-loop cause analysis (closes Part 5).
    # Scheduler loop time, reported in ns — a loop iteration is sub-us, so us
    # rounds to a misleading 0.0; ns keeps it readable.
    avg_loop_us = total_us / total_loops if total_loops > 0 else 0
    avg_loop_ns = avg_loop_us * 1000
    avg_tail_oh = sum(tails) / n
    loop_ratio = avg_tail_oh / avg_loop_us if avg_loop_us > 0 else 0
    print(f"  Avg scheduler loop iteration: {avg_loop_ns:.0f} ns (approx avg polling interval per loop)")
    print()
    print(f"  Avg Tail OH = {avg_tail_oh:.1f} us ~= {loop_ratio:.1f} x avg loop iteration ({avg_loop_ns:.0f} ns)")
    print(f"  -> On average, a completed task waits ~{loop_ratio:.1f} loop iterations before being detected")
    print()

    # Data-driven insight: find the dominant phase (excluding idle which is not useful work)
    work_phases = {p: phase_totals.get(p, 0) for p in phases if p != "idle"}
    dominant_phase = max(work_phases, key=lambda p: work_phases[p])
    dominant_pct = work_phases[dominant_phase] / total_us * 100 if total_us > 0 else 0
    key_phase_label = phase_labels[dominant_phase].split(" (")[0]
    print(f"  Key insight: {key_phase_label} phase consumes ~{dominant_pct:.0f}% of scheduler CPU.")
    if dominant_phase == "dispatch":
        pop_note = "low hit rate suggests ready queue often empty" if pop_hit_rate < 50 else "good hit rate"
        print(f"  Pop hit_rate={pop_hit_rate:.1f}%: {pop_note}.")
        print("  Cache flush (dc cvac + dsb sy) is the dominant non-pop cost.")
    elif dominant_phase == "complete":
        if dag_stats_available:
            print(f"  Fanout: avg_degree={fanout_avg:.1f}, max_degree={fanout_max}.")
            print(f"  Fanin:  avg_degree={fanin_avg:.1f}, max_degree={fanin_max}.")
            if fanin_edges > fanout_edges:
                print("  Fanin traversal (release_producer + check_consumed) dominates the complete phase.")
            else:
                print("  Fanout traversal and atomic ops dominate the complete phase.")
        else:
            print("  DAG stats unavailable (no deps.json); cannot attribute complete-phase cost further.")
    print("=" * 90)

    # === Part 6: Critical-path latency attribution ===
    cp = compute_critical_path(preds_by_id, end_by_id, finish_by_id, start_by_id, dispatch_by_id, w0)
    print()
    print("=" * 90)
    print("Part 6: Critical-path latency attribution")
    print("=" * 90)
    if cp and cp["span"] > 0:
        sched_pct = cp["sched"] / cp["span"] * 100
        exec_pct = cp["exec"] / cp["span"] * 100
        print(f"  Makespan-determining path: {cp['hops']} hops, span {cp['span']:.1f} us")
        print(f"    Compute (exec) on path : {cp['exec']:.1f} us ({exec_pct:.1f}%)")
        print(f"    Scheduler injected     : {cp['sched']:.1f} us ({sched_pct:.1f}%)")
        print(f"    Other (dep wait on path): {max(0.0, cp['span'] - cp['exec'] - cp['sched']):.1f} us")
        print(f"  -> scheduler adds ~{sched_pct:.1f}% to the critical path's end-to-end latency.")
    else:
        print("  (could not resolve a critical path from the DAG)")
    print("=" * 90)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Scheduler overhead analysis for PTO2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # auto-select latest files
  %(prog)s --l2-swimlane-records-json outputs/<case>_<ts>/l2_swimlane_records.json
  %(prog)s --l2-swimlane-records-json outputs/<case>_<ts>/l2_swimlane_records.json \
      --deps-json outputs/<case>_<ts>/deps.json
        """,
    )
    parser.add_argument(
        "--l2-swimlane-records-json",
        help="Path to l2_swimlane_records_*.json file. If not specified, uses the latest in outputs/",
    )
    parser.add_argument(
        "--deps-json",
        help=(
            "Path to deps.json (dep_gen replay output). When provided, "
            "fanout / fanin per-thread aggregates are computed from it. "
            "Defaults to deps.json next to the perf JSON."
        ),
    )
    args = parser.parse_args()

    # Resolve perf path
    try:
        l2_swimlane_records_path = (
            Path(args.l2_swimlane_records_json)
            if args.l2_swimlane_records_json
            else auto_select_l2_swimlane_records_json()
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not l2_swimlane_records_path.exists():
        print(f"Error: Perf JSON not found: {l2_swimlane_records_path}", file=sys.stderr)
        return 1

    # Single load — go through swimlane_converter.read_perf_data so the raw
    # per-stream JSON gets joined into the dict run_analysis() expects. Pass
    # the parsed dict on so the file is only read once.
    # Lazy import: swimlane_converter imports run_analysis from this module
    # at top level, so the import must happen at call time.
    try:
        from .swimlane_converter import read_perf_data  # noqa: PLC0415

        perf_data = read_perf_data(l2_swimlane_records_path)
    except (OSError, ValueError) as e:
        print(f"Error: failed to read perf JSON {l2_swimlane_records_path}: {e}", file=sys.stderr)
        return 1

    deps_json_path = Path(args.deps_json) if args.deps_json else None

    return run_analysis(
        l2_swimlane_records_path,
        print_sources=True,
        deps_json_path=deps_json_path,
        perf_data=perf_data,
    )


if __name__ == "__main__":
    sys.exit(main())
