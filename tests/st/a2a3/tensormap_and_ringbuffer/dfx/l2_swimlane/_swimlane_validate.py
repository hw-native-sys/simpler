# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""Shared l2_swimlane post-case validation.

The vector_example and paged_attention swimlane tests run the same capture →
tool smoke → differential gate sequence; the only difference between them is
the workload itself. The helpers below are workload-agnostic so each test
file owns only its CALLABLE + cases.

The differential gate is the load-bearing assertion: it parses the script's
printed Pop / Fanout / Fanin totals and cross-checks them against an oracle
computed straight from the raw artifacts. The paged_attention test exercises
the per-task dedup branch in ``compute_dag_stats_from_deps`` because mixed
AIC+AIV tasks produce multiple perf rows per ``task_id``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

from simpler_setup.scene_test import _outputs_dir, _sanitize_for_filename

_REQUIRED_TASK_FIELDS = ("task_id", "func_id", "core_id", "core_type", "start_time_us", "end_time_us")


def validate_perf_artifact(case_label: str, *, expected_task_count: int | None = None) -> None:
    """Locate the latest output dir for ``case_label`` and run the full
    capture-→-tools-→-differential sequence.

    Args:
        case_label: full SceneTest case label (``f"{cls_name}_{case_name}"``)
            used to glob the per-case ``outputs/<label>_<ts>/`` directory.
        expected_task_count: when provided, assert ``len(tasks) == N``.
            Workloads whose task count varies with sim/onboard timing should
            leave this ``None`` and rely on the differential gate.
    """
    safe_label = _sanitize_for_filename(case_label)
    matches = sorted(_outputs_dir().glob(f"{safe_label}_*"), key=lambda p: p.stat().st_mtime)
    if not matches:
        return
    perf = matches[-1] / "l2_perf_records.json"
    assert perf.exists(), f"l2_perf_records.json missing under {matches[-1]} — swimlane capture failed?"

    with perf.open() as f:
        data = json.load(f)
    tasks = data.get("tasks")
    assert isinstance(tasks, list), "tasks field missing or not a list"
    assert len(tasks) > 0, f"perf records empty under {perf}"
    if expected_task_count is not None:
        assert len(tasks) == expected_task_count, (
            f"got {len(tasks)} perf records, expected {expected_task_count} under {perf}"
        )
    # Spot-check a single record's required fields — guards against drift in
    # the swimlane schema that swimlane_converter.py / deps_to_graph.py rely on.
    first = tasks[0]
    for key in _REQUIRED_TASK_FIELDS:
        assert key in first, f"perf record missing required field '{key}': {first}"

    # ---- Tool smoke: swimlane_converter ----
    # Exit-code-only check; we don't validate the Perfetto JSON content. A
    # schema change that breaks the converter fires here in the same CI
    # step that produced the artifact.
    subprocess.run(
        [
            sys.executable,
            "-m",
            "simpler_setup.tools.swimlane_converter",
            str(perf),
            "-o",
            str(matches[-1] / "_smoke_swimlane.json"),
        ],
        check=True,
        timeout=60,
    )

    # ---- Tool smoke: sched_overhead_analysis ----
    # pop_hit / pop_miss come from the dispatch-phase extras the runtime writes
    # (l2_perf_collector.cpp). The differential block below cross-validates
    # the script's printed numbers against an independent oracle computed
    # straight from the raw artifacts — any regression in either the runtime
    # capture path or the parser arithmetic fails here in the same CI step
    # that produced the data.
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "simpler_setup.tools.sched_overhead_analysis",
            "--l2-perf-records-json",
            str(perf),
        ],
        check=True,
        timeout=120,
        capture_output=True,
        text=True,
    )
    for header in ("Part 1:", "Part 2:", "Part 3:"):
        assert header in result.stdout, f"sched_overhead missing section header '{header}'\nstdout:\n{result.stdout}"
    # Bad pattern: AICPU didn't capture real cycle counters → tool "succeeds"
    # but every metric is 0. Match the line that's printed unconditionally
    # in Part 2 and assert its value is non-zero.
    m = re.search(r"Avg scheduler loop iteration:\s+([\d.]+)\s+us", result.stdout)
    assert m, f"sched_overhead stdout missing 'Avg scheduler loop iteration'\nstdout:\n{result.stdout}"
    assert float(m.group(1)) > 0.0, (
        f"sched_overhead reports zero loop iteration (avg_loop_us={m.group(1)}). "
        f"AICPU likely didn't capture dispatch_time/finish_time cycle counters — "
        f"the L2 perf collector path may have regressed.\nstdout:\n{result.stdout}"
    )
    verify_sched_overhead_differential(result.stdout, data, matches[-1])


def verify_sched_overhead_differential(stdout: str, perf: dict, artifact_dir: Path) -> None:
    """Cross-check the script's printed Pop / Fanout / Fanin totals against
    an oracle computed independently from the raw artifacts. The script and
    the oracle should agree exactly — if they don't, either the runtime
    capture regressed or the parser arithmetic drifted, and the bug is
    caught in the same CI step that produced the data.

    Args:
        stdout: captured ``sched_overhead_analysis`` stdout.
        perf: parsed ``l2_perf_records.json`` dict — passed in by the caller
            so we don't re-read multi-MB profiling artifacts here.
        artifact_dir: per-case output directory. ``deps.json`` is looked up
            beside the perf JSON; absent → fanout / fanin half is skipped.

    The per-task dedup branch is exercised on mixed AIC+AIV workloads where
    the perf JSON emits one row per subtask/core for a single ``task_id``.
    """
    # Oracle: pop_hit / pop_miss are the sum across all dispatch records.
    # Compares against the "Pop: hit=N, miss=M" line the script prints.
    phases = perf.get("aicpu_scheduler_phases", [])
    oracle_pop_hit = sum(r.get("pop_hit", 0) for thr_recs in phases for r in thr_recs if r.get("phase") == "dispatch")
    oracle_pop_miss = sum(r.get("pop_miss", 0) for thr_recs in phases for r in thr_recs if r.get("phase") == "dispatch")
    pop_match = re.search(r"Pop:\s*hit=(\d+),\s*miss=(\d+)", stdout)
    assert pop_match, f"sched_overhead stdout missing 'Pop: hit=N, miss=M' line\nstdout:\n{stdout}"
    printed_pop_hit, printed_pop_miss = int(pop_match.group(1)), int(pop_match.group(2))
    assert printed_pop_hit == oracle_pop_hit, (
        f"Pop hit mismatch: printed={printed_pop_hit}, oracle={oracle_pop_hit} "
        f"(summed from dispatch-record extras)\nstdout:\n{stdout}"
    )
    assert printed_pop_miss == oracle_pop_miss, (
        f"Pop miss mismatch: printed={printed_pop_miss}, oracle={oracle_pop_miss}\nstdout:\n{stdout}"
    )

    # Fanout / fanin differential — only meaningful when deps.json is
    # colocated (i.e. --enable-dep-gen was also on). When absent, skip.
    deps_path = artifact_dir / "deps.json"
    if not deps_path.exists():
        return
    with deps_path.open() as f:
        deps = json.load(f)
    unique_edges = set()
    for e in deps.get("edges", []):
        try:
            pred, succ = int(e["pred"]), int(e["succ"])
        except (TypeError, ValueError, KeyError):
            continue
        if pred < 0:
            pred &= (1 << 64) - 1
        if succ < 0:
            succ &= (1 << 64) - 1
        unique_edges.add((pred, succ))

    # Per-thread oracle: a task's fanout is billed to the thread that
    # retired it (core_to_thread[task.core_id]). Sum across threads ==
    # total edges (modulo unattributed tasks, e.g. alloc-only with no
    # core_id). The script prints the sum-across-threads total.
    core_to_thread = perf.get("core_to_thread") or []
    edges_by_pred: dict[int, set[int]] = {}
    edges_by_succ: dict[int, set[int]] = {}
    for pred, succ in unique_edges:
        edges_by_pred.setdefault(pred, set()).add(succ)
        edges_by_succ.setdefault(succ, set()).add(pred)
    # Dedup by task_id: mixed tasks emit one perf row per subtask/core.
    oracle_fanout = 0
    oracle_fanin = 0
    seen_tids: set[int] = set()
    for task in perf.get("tasks", []):
        cid = task.get("core_id")
        if not isinstance(cid, int) or not (0 <= cid < len(core_to_thread)):
            continue
        if core_to_thread[cid] < 0:
            continue
        try:
            tid = int(task["task_id"])
        except (TypeError, ValueError, KeyError):
            continue
        if tid < 0:
            tid &= (1 << 64) - 1
        if tid in seen_tids:
            continue
        seen_tids.add(tid)
        oracle_fanout += len(edges_by_pred.get(tid, ()))
        oracle_fanin += len(edges_by_succ.get(tid, ()))

    fanout_match = re.search(r"Fanout \(.*?\):\s*total edges=(\d+),\s*max_degree=(\d+)", stdout)
    fanin_match = re.search(r"Fanin\s+\(.*?\):\s*total edges=(\d+),\s*max_degree=(\d+)", stdout)
    assert fanout_match, f"sched_overhead stdout missing 'Fanout' line\nstdout:\n{stdout}"
    assert fanin_match, f"sched_overhead stdout missing 'Fanin' line\nstdout:\n{stdout}"
    printed_fanout = int(fanout_match.group(1))
    printed_fanin = int(fanin_match.group(1))
    assert printed_fanout == oracle_fanout, (
        f"Fanout edges mismatch: printed={printed_fanout}, oracle={oracle_fanout} "
        f"(derived from {len(unique_edges)} unique deps.json edges + core_to_thread)\n"
        f"stdout:\n{stdout}"
    )
    assert printed_fanin == oracle_fanin, (
        f"Fanin edges mismatch: printed={printed_fanin}, oracle={oracle_fanin}\nstdout:\n{stdout}"
    )
