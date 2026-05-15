#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Scheduler overhead analysis for PTO2.

Inputs:
  1. Per-task perf profiling data (l2_perf_records_*.json), v2 schema with
     ``aicpu_scheduler_phases`` populated by ``--enable-l2-swimlane``.
  2. deps.json (optional, dep_gen replay output) colocated with the perf JSON,
     used to derive per-thread fanout / fanin DAG stats.

Usage:
    python -m simpler_setup.tools.sched_overhead_analysis                   # auto-select latest files
    python -m simpler_setup.tools.sched_overhead_analysis --l2-perf-records-json <path>
    python -m simpler_setup.tools.sched_overhead_analysis --l2-perf-records-json <path> --deps-json <path>
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path


def _to_uint64(v):
    """Coerce JSON-encoded uint64 (int or string after the deps.json v2 schema
    bump in #769) to a Python int. Returns None when unparseable."""
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
    thread map is in ``l2_perf_records.json::core_to_thread``. Re-instrumenting
    the AICPU to track fanout edge counts is duplicate work; running this in
    Python over the existing artifacts is cheaper, more accurate (deps.json
    captures #599 race-window edges that fanout[] dropped), and lets the
    analysis work on default builds that don't have PTO2_SCHED_PROFILING=1.

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
    # core (see l2_perf_collector.cpp:567 — collected_perf_records_ is keyed by
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


def auto_select_l2_perf_records_json():
    """Find the latest outputs/<case>/l2_perf_records.json (sorted by mtime)."""
    outputs_dir = Path.cwd() / "outputs"
    files = sorted(outputs_dir.glob("*/l2_perf_records.json"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not files:
        raise FileNotFoundError(f"No outputs/*/l2_perf_records.json found under {outputs_dir}")
    return files[0]


def parse_scheduler_from_json_phases(data):
    """Extract scheduler Phase breakdown from l2_perf_records JSON (version >= 2).

    Computes per-thread loop counts, task counts, and phase totals
    from aicpu_scheduler_phases records.

    Returns:
        dict: Thread data keyed by thread index, with per-phase us / pct,
              pop_hit / pop_miss, loops, completed, tasks_per_loop. Returns
              empty dict if phase data is not available.
    """
    if data.get("version", 1) < 2:
        return {}
    phases_by_thread = data.get("aicpu_scheduler_phases", [])
    if not phases_by_thread:
        return {}

    # Map JSON phase names to internal names
    phase_map = {
        "complete": "complete",
        "dispatch": "dispatch",
        "scan": "scan",
        "idle": "idle",
    }

    threads = {}
    for tid, records in enumerate(phases_by_thread):
        if not records:
            continue

        phase_us = {p: 0.0 for p in ["complete", "scan", "dispatch", "idle"]}
        total_tasks = 0
        max_loop_iter = 0
        pop_hit = 0
        pop_miss = 0

        for rec in records:
            phase = phase_map.get(rec.get("phase", ""))
            if phase is None:
                continue
            dur = rec.get("end_time_us", 0) - rec.get("start_time_us", 0)
            if dur > 0:
                phase_us[phase] += dur
            if rec.get("tasks_processed", 0) > 0 and phase == "complete":
                total_tasks += rec["tasks_processed"]
            loop_iter = rec.get("loop_iter", 0)
            max_loop_iter = max(max_loop_iter, loop_iter)
            # Per-emit queue-health deltas; only present on dispatch records.
            # Summing across records gives the run-cumulative pop_hit /
            # pop_miss (the runtime's final-drain emit closes the tail).
            if phase == "dispatch":
                pop_hit += rec.get("pop_hit", 0)
                pop_miss += rec.get("pop_miss", 0)

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
        # src/platform/src/host/performance_collector.cpp (dispatch_time_us, finish_time_us)
        msg = "\n".join(
            [
                "Perf JSON is incompatible with scheduler overhead deep-dive analysis.",
                f"Missing required fields (showing up to 5 tasks): {detail}",
                "",
                "Why this happens:",
                "  - The input is not a runtime-generated l2_perf_records_*.json, OR",
                "  - The runtime binary does not include / emit dispatch+finish timestamps.",
                "",
                "How to fix:",
                "  1) Re-run workload with profiling enabled (e.g. run_example.py --enable-l2-swimlane).",
                "  2) Pass the newly generated outputs/<case>/l2_perf_records.json via --l2-perf-records-json.",
                "  3) Verify each task includes dispatch_time_us and finish_time_us.",
                "",
                "Note:",
                "  - swimlane_converter conversion can still succeed; only deep-dive analysis requires these fields.",
            ]
        )
        return False, msg

    return True, ""


def run_analysis(  # noqa: PLR0912, PLR0915
    l2_perf_records_path,
    print_sources=True,
    deps_json_path=None,
    perf_data=None,
):
    """Run scheduler overhead analysis report.

    Args:
        l2_perf_records_path: Path to l2_perf_records_*.json.
        print_sources: Whether to print selected input files.
        perf_data: Optional pre-parsed perf JSON dict. When provided, skip
            re-reading from disk — main() already parses the file to probe
            for v2 phase data, so passing the result through saves a second
            load on large artifacts.
        deps_json_path: Optional deps.json (dep_gen replay output) co-located
            with the perf JSON. When present, per-thread fanout / fanin
            aggregates are derived from it.

    Returns:
        int: 0 on success, non-zero on failure.
    """
    l2_perf_records_path = Path(l2_perf_records_path)

    if not l2_perf_records_path.exists():
        print(f"Error: Perf JSON not found: {l2_perf_records_path}", file=sys.stderr)
        return 1

    # Auto-discover deps.json sibling when caller didn't specify one.
    if deps_json_path is None:
        sibling = l2_perf_records_path.parent / "deps.json"
        if sibling.exists():
            deps_json_path = sibling

    if print_sources:
        print(f"Perf data:  {l2_perf_records_path}")
        if deps_json_path is not None:
            print(f"Deps JSON:  {deps_json_path}")

    # === Part 1: Per-task time breakdown from perf data ===
    if perf_data is not None:
        data = perf_data
    else:
        with open(l2_perf_records_path) as f:
            data = json.load(f)
    tasks = data["tasks"]
    n_total = len(tasks)

    if n_total == 0:
        print("Error: No tasks found in perf data", file=sys.stderr)
        return 1

    valid, err = validate_perf_tasks_for_overhead_analysis(tasks)
    if not valid:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    all_exec = sum(t["duration_us"] for t in tasks)
    all_head = sum(t["start_time_us"] - t["dispatch_time_us"] for t in tasks)
    all_tail = sum(t["finish_time_us"] - t["end_time_us"] for t in tasks)
    min_disp = min(t["dispatch_time_us"] for t in tasks)
    max_fin = max(t["finish_time_us"] for t in tasks)
    wall = max_fin - min_disp

    all_latency = all_exec + all_head + all_tail

    print()
    print("=" * 90)
    print("Part 1: Per-task time breakdown (from perf profiling data)")
    print("=" * 90)
    print(f"Total tasks: {n_total}")
    print(f"Wall-clock:  {wall:.1f} us")
    print()
    fmt = "  {:<35} {:>12} {:>14} {:>13}"
    print(fmt.format("Component", "Total (us)", "Avg/task (us)", "% of Latency"))
    print("  " + "-" * 78)
    print(
        fmt.format(
            "Kernel Exec (end-start)",
            f"{all_exec:.1f}",
            f"{all_exec / n_total:.2f}",
            f"{all_exec / all_latency * 100:.1f}%",
        )
    )
    print(
        fmt.format(
            "Head OH (start-dispatch)",
            f"{all_head:.1f}",
            f"{all_head / n_total:.2f}",
            f"{all_head / all_latency * 100:.1f}%",
        )
    )
    print(
        fmt.format(
            "Tail OH (finish-end)",
            f"{all_tail:.1f}",
            f"{all_tail / n_total:.2f}",
            f"{all_tail / all_latency * 100:.1f}%",
        )
    )
    print()

    # === Part 2: AICPU scheduler loop breakdown ===
    threads = parse_scheduler_from_json_phases(data)
    if not threads:
        print(
            "Error: perf JSON has no aicpu_scheduler_phases — rerun the case "
            "with --enable-l2-swimlane so phase data is captured.",
            file=sys.stderr,
        )
        return 1

    # Per-thread fanout / fanin come from deps.json when colocated. Without it
    # the report still prints Part 2 phase timings but suppresses the DAG-stats
    # rows so missing-artifact zeros are not mistaken for measured values.
    dag_stats_available = False
    if deps_json_path is not None:
        try:
            with open(deps_json_path) as df:
                deps_data = json.load(df)
            compute_dag_stats_from_deps(deps_data, data, threads)
            dag_stats_available = True
        except (OSError, ValueError) as ex:
            print(f"Warning: failed to load deps.json for DAG stats: {ex}", file=sys.stderr)

    n_threads = len(threads)

    print("=" * 90)
    print("Part 2: AICPU scheduler loop breakdown")
    print(f"  {n_threads} scheduler threads")
    print("=" * 90)
    print()

    fmt2 = "  {:<10} {:>7} {:>10} {:>12} {:>11}"
    print(fmt2.format("Thread", "Loops", "Completed", "Tasks/loop", "Total (us)"))
    print("  " + "-" * 54)
    for tid in sorted(threads.keys()):
        t = threads[tid]
        print(
            fmt2.format(
                "T" + str(tid), t["loops"], t["completed"], f"{t['tasks_per_loop']:.1f}", f"{t['total_us']:.1f}"
            )
        )
    total_us = sum(t["total_us"] for t in threads.values())
    total_completed = sum(t["completed"] for t in threads.values())
    total_loops = sum(t["loops"] for t in threads.values())
    avg_tpl = total_completed / total_loops if total_loops > 0 else 0
    print(fmt2.format("SUM", total_loops, total_completed, f"{avg_tpl:.1f}", f"{total_us:.1f}"))
    print()

    # Phase breakdown
    phases = ["complete", "scan", "dispatch", "idle"]
    phase_labels = {
        "complete": "Complete (poll handshake, resolve deps)",
        "scan": "Scan (update perf header)",
        "dispatch": "Dispatch (pop queue, build payload, flush)",
        "idle": "Idle (spinning, no progress)",
    }

    fmt3 = "  {:<50} {:>11} {:>10} {:>14}"
    print(fmt3.format("Phase", "Total (us)", "% of total", "Avg/task (us)"))
    print("  " + "-" * 89)
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
        print("  Pop: (no per-emit pop deltas in input — needs --enable-l2-swimlane on a v2 JSON capture)")

    print()
    print("=" * 90)
    print("Part 3: Tail OH distribution & cause analysis")
    print("=" * 90)
    print()

    tails = [t["finish_time_us"] - t["end_time_us"] for t in tasks]
    tails.sort()
    n = len(tails)
    if n == 0:
        print("Error: Empty tail-overhead set", file=sys.stderr)
        return 1

    print(f"  Tail OH distribution (N={n}):")
    for pct_val in [10, 25, 50, 75, 90, 95, 99]:
        idx = min(int(n * pct_val / 100), n - 1)
        print(f"    P{pct_val:<4}  {tails[idx]:>7.1f} us")
    print(f"    Max:   {tails[-1]:>7.1f} us")
    print(f"    Mean:  {sum(tails) / n:>7.1f} us")
    print()

    # Scheduler loop time
    avg_loop_us = total_us / total_loops if total_loops > 0 else 0
    avg_tail_oh = sum(tails) / n
    loop_ratio = avg_tail_oh / avg_loop_us if avg_loop_us > 0 else 0
    print(f"  Avg scheduler loop iteration: {avg_loop_us:.1f} us (approx avg polling interval per loop)")
    print()
    print(f"  Avg Tail OH = {avg_tail_oh:.1f} us ~= {loop_ratio:.1f} x avg loop iteration ({avg_loop_us:.1f} us)")
    print(f"  -> On average, a completed task waits ~{loop_ratio:.1f} loop iterations before being detected")
    print()

    # Data-driven insight: find the dominant phase (excluding idle which is not useful work)
    work_phases = {p: phase_totals.get(p, 0) for p in ["scan", "complete", "dispatch"]}
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
    elif dominant_phase == "scan":
        print("  Scan phase overhead indicates frequent perf header updates.")
    print("=" * 90)

    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Scheduler overhead analysis for PTO2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                          # auto-select latest files
  %(prog)s --l2-perf-records-json outputs/<case>_<ts>/l2_perf_records.json
  %(prog)s --l2-perf-records-json outputs/<case>_<ts>/l2_perf_records.json --deps-json outputs/<case>_<ts>/deps.json
        """,
    )
    parser.add_argument(
        "--l2-perf-records-json",
        help="Path to l2_perf_records_*.json file. If not specified, uses the latest in outputs/",
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
        l2_perf_records_path = (
            Path(args.l2_perf_records_json) if args.l2_perf_records_json else auto_select_l2_perf_records_json()
        )
    except FileNotFoundError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    if not l2_perf_records_path.exists():
        print(f"Error: Perf JSON not found: {l2_perf_records_path}", file=sys.stderr)
        return 1

    # Single load — pass the parsed dict to run_analysis() so it doesn't
    # reread the file (large artifacts hit JSON parsing twice otherwise).
    try:
        with open(l2_perf_records_path) as _f:
            perf_data = json.load(_f)
    except (OSError, ValueError) as e:
        print(f"Error: failed to read perf JSON {l2_perf_records_path}: {e}", file=sys.stderr)
        return 1

    deps_json_path = Path(args.deps_json) if args.deps_json else None

    return run_analysis(
        l2_perf_records_path,
        print_sources=True,
        deps_json_path=deps_json_path,
        perf_data=perf_data,
    )


if __name__ == "__main__":
    sys.exit(main())
