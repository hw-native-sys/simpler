#!/usr/bin/env python3
# Copyright (c) PyPTO Contributors.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Swimlane JSON to Perfetto JSON Converter

Converts performance data JSON (.json) to Chrome Trace Event Format JSON
for visualization in Perfetto (https://ui.perfetto.dev/).

Usage:
    python -m simpler_setup.tools.swimlane_converter  # latest l2_swimlane_records_*.json under ./outputs/
    python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json
    python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json -o out.json
    python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json -k kernel_config.py
    python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json -v

SPMD (block_num>1): dependency flows use min core_id anchors per core_type
on physical lanes; see docs/dfx/l2-swimlane-profiling.md §3.5.
"""

import argparse
import bisect
import importlib.util
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any


def _func_id_to_letter(func_id):
    """Map a non-negative integer func_id to a numeric+letter label.

    0 → '0_a', 1 → '1_b', …, 25 → '25_z', 26 → '26_aa', 27 → '27_ab', …
    """
    try:
        n = int(func_id)
    except (TypeError, ValueError):
        return str(func_id)
    letters = []
    m = n + 1  # shift so that 0 maps to 'a' (1-based bijective base-26)
    while m > 0:
        m, rem = divmod(m - 1, 26)
        letters.append(chr(ord("a") + rem))
    return str(n) + "_" + "".join(reversed(letters))


def _task_display_name(func_id, func_id_to_name, tdisp, *, spmd=False):
    """Build the swimlane event label for a task.

    Naming, in priority order:
      - ``func_id < 0`` (unresolved): ``task(rXtY)``. This is the no-deps.json
        case — without a dep_gen capture the host never carries func_id, so
        every lane is an anonymous ``task(...)`` distinguished only by id.
      - a name mapping exists for the func_id: ``<name>(rXtY)``.
      - otherwise: ``func_<letter>(rXtY)`` (resolved id, but no name map entry).

    SPMD logical tasks append ``_spmd`` before the ``(rXtY)`` suffix unless the
    name already contains ``spmd`` (case-insensitive), e.g.
    ``fa_fused_aic_spmd(r2t18)``.
    """
    try:
        resolved = int(func_id) >= 0
    except (TypeError, ValueError):
        resolved = False
    if not resolved:
        label = f"task({tdisp})"
    elif func_id_to_name and str(func_id) in func_id_to_name:
        label = f"{func_id_to_name[str(func_id)]}({tdisp})"
    else:
        label = f"func_{_func_id_to_letter(func_id)}({tdisp})"
    if spmd and "(" in label:
        base, rest = label.split("(", 1)
        if "spmd" not in base.lower():
            return f"{base}_spmd({rest}"
    return label


def normalize_pto2_task_id_int(v):
    """Unsigned 64-bit PTO2 task id (matches host JSON / device ``task_id.raw``).

    Normalizes signed values to unsigned for ``(ring_id << 32) | local_id``.
    Returns None if ``v`` is not convertible to int.
    """
    try:
        t = int(v)
    except (TypeError, ValueError):
        return None
    if t < 0:
        t &= (1 << 64) - 1
    return t


def format_task_display(task_id):
    """Format PTO2 task_id for human-readable labels.

    Layout: 64-bit raw = (ring_id << 32) | local_id (same as runtime PTO2TaskId).

    Returns:
        ``r{ring}t{local}`` when ring != 0 (e.g. r2t100), else ``t{local}`` for single-ring (ring 0).

    For invalid or non-numeric values, returns str(task_id).
    """
    tid = normalize_pto2_task_id_int(task_id)
    if tid is None:
        return str(task_id)
    ring = (tid >> 32) & 0xFF
    local = tid & 0xFFFFFFFF
    if ring == 0:
        return f"t{local}"
    return f"r{ring}t{local}"


def read_perf_data(filepath):  # noqa: PLR0912, PLR0915
    """Read performance data from a swimlane JSON file.

    Host dumps raw cycle-domain per-stream records plus metadata; this
    function does the AICore↔AICPU join. Schema:

        {
          "l2_swimlane_level": <1..4>,
          "metadata": {
            "clock_freq_hz": <int>,
            "num_cores": <int>,
            "core_types": ["aic"|"aiv", ...],   # indexed by core_id
            "core_to_thread": [<int>, ...]      # optional (level >= 3)
          },
          "aicore_tasks": [[core_id, task_token_raw, reg_task_id, start_cycles,
                            end_cycles, receive_to_start_cycles], ...],
          "aicpu_tasks":  [[core_id, reg_task_id, dispatch_cycles, finish_cycles], ...],
          "aicpu_scheduler_phases":     [ [ {kind, start_cycles, end_cycles, ...}, ... ], ... ],
          "aicpu_orchestrator_phases":  [ [ {submit_idx, task_id, start_cycles, end_cycles}, ... ], ... ]
        }

    aicore_tasks columns (v3 schema): the trailing receive_to_start_cycles
    is a uint32 delta = AICore-side `start_time - receive_time`, where
    receive_time is captured immediately after AICore's
    `read_reg(DATA_MAIN_BASE)` returns the new task_id (before the per-task
    dcci + ack pair). Lets DFX split per-task head_OH into the
    AICPU→AICore NoC propagation (dispatch_ts → receive_time, hardware-
    bound) and the AICore-local dcci + ack cost (receive_time → start_time,
    software-tunable). Archived v2 JSON without this column still parses;
    the field is exposed as 0 for those.

    Returns a dict shaped for `generate_chrome_trace_json`,
    `print_task_statistics`, and `sched_overhead_analysis`: `tasks`,
    `aicpu_scheduler_phases`, `aicpu_orchestrator_phases`,
    `core_to_thread`.

    The join logic that used to live in `export_swimlane_json` (host C++):

      - per-core `reg_task_id → (task_token_raw, start_cycles, end_cycles)` map
        from `aicore_tasks` (the AICore is the canonical identity producer)
      - `base_time_cycles` = min non-zero timestamp across all streams (task,
        phase, orch)
      - cycles → µs via `clock_freq_hz` from metadata (a2a3=50 MHz, a5=1 GHz —
        the freq MUST come from the host, never be hardcoded here)
      - join `aicpu_tasks` by `(core_id, reg_task_id)`; unmatched rows are
        dropped and counted
      - AICORE_TIMING (level=1): aicpu_tasks is empty by construction, so
        synthesize one task per aicore record (dispatch/finish = 0)
      - sort joined `tasks` by `task_id` (= task_token_raw)
      - convert phase records from `*_cycles` → `*_time_us`

    Raises:
        ValueError: If the JSON is malformed.
    """
    with open(filepath) as f:
        data = json.load(f)

    level = int(data.get("l2_swimlane_level"))
    if level not in [1, 2, 3, 4]:
        raise ValueError(f"Unsupported l2_swimlane_level: {level} (expected 1, 2, 3, or 4)")

    metadata = data.get("metadata") or {}
    clock_freq_hz = int(metadata.get("clock_freq_hz") or 0)
    if clock_freq_hz <= 0:
        raise ValueError(f"metadata missing/zero clock_freq_hz: {clock_freq_hz}")
    core_types = list(metadata.get("core_types") or [])
    core_to_thread = list(metadata.get("core_to_thread") or [])

    aicore_rows = data.get("aicore_tasks") or []
    aicpu_rows = data.get("aicpu_tasks") or []
    sched_phases_raw = data.get("aicpu_scheduler_phases") or []
    orch_phases_raw = data.get("aicpu_orchestrator_phases") or []

    # AICore lookup keyed by (core_id, reg_task_id). Two dispatches of the
    # same PTO2 task_token_raw to the same core (SPMD over-subscription, MIX
    # cluster spread) each get their own reg_task_id, so this key is unique
    # per dispatch even when task_token_raw collides.
    #
    # `*rest` makes v2 rows (5 cols, no receive_to_start_cycles) and v3 rows
    # (6 cols) both parse — archived JSON from before the receive_time split
    # still loads with r2s_cycles defaulting to 0.
    aicore_lookup: dict[tuple[int, int], tuple[int, int, int, int]] = {}
    for row in aicore_rows:
        core_id, task_token_raw, reg_task_id, start_cycles, end_cycles, *rest = row
        r2s_cycles = int(rest[0]) if rest else 0
        aicore_lookup[(int(core_id), int(reg_task_id))] = (
            int(task_token_raw),
            int(start_cycles),
            int(end_cycles),
            r2s_cycles,
        )

    # base_time = min non-zero timestamp across every stream that will be
    # emitted. Used as the cycle-domain zero for all µs conversions.
    base_time_cycles = None

    def _track(v):
        nonlocal base_time_cycles
        if v > 0 and (base_time_cycles is None or v < base_time_cycles):
            base_time_cycles = v

    for row in aicore_rows:
        # Column count varies (v2: 5, v3: 6); only the timing columns matter
        # for base_time tracking. For v3, the per-task receive_time =
        # start_cycles - receive_to_start_cycles is earlier than start_cycles
        # itself; track it so Worker View task bars that start at receive_time
        # don't land at a negative offset relative to the kernel start.
        start_c = int(row[3])
        end_c = int(row[4])
        r2s_c = int(row[5]) if len(row) > 5 else 0
        _track(start_c - r2s_c)
        _track(end_c)
    for _, _, d, f in aicpu_rows:
        _track(int(d))
        _track(int(f))
    for thread_records in sched_phases_raw:
        for pr in thread_records:
            _track(int(pr.get("start_cycles", 0)))
            _track(int(pr.get("end_cycles", 0)))
    for thread_records in orch_phases_raw:
        for pr in thread_records:
            _track(int(pr.get("start_cycles", 0)))
            _track(int(pr.get("end_cycles", 0)))

    if base_time_cycles is None:
        base_time_cycles = 0

    cycles_to_us_factor = 1_000_000.0 / float(clock_freq_hz)

    def _to_us(cycles):
        if cycles <= 0:
            return 0.0
        return (cycles - base_time_cycles) * cycles_to_us_factor

    def _core_type(core_id):
        if 0 <= core_id < len(core_types):
            return core_types[core_id]
        return "aiv"

    tasks = []
    unmatched_per_core: dict[int, int] = defaultdict(int)

    if aicpu_rows:
        for row in aicpu_rows:
            core_id, reg_task_id, dispatch_cycles, finish_cycles = row
            core_id = int(core_id)
            reg_task_id = int(reg_task_id)
            ac = aicore_lookup.get((core_id, reg_task_id))
            if ac is None:
                unmatched_per_core[core_id] += 1
                continue
            task_token_raw, start_cycles, end_cycles, r2s_cycles = ac
            start_us = _to_us(start_cycles)
            end_us = _to_us(end_cycles)
            dispatch_us = _to_us(int(dispatch_cycles))
            # receive_to_start delta is in cycles; convert via the same
            # cycles_to_us_factor that drives the absolute timestamps. No
            # base_time subtraction — this is a delta.
            local_setup_us = r2s_cycles * cycles_to_us_factor
            receive_us = start_us - local_setup_us
            tasks.append(
                {
                    "task_id": task_token_raw,
                    "func_id": -1,
                    "core_id": core_id,
                    "core_type": _core_type(core_id),
                    "ring_id": (task_token_raw >> 32) & 0xFFFFFFFF,
                    "start_time_us": start_us,
                    "end_time_us": end_us,
                    "duration_us": end_us - start_us,
                    "dispatch_time_us": dispatch_us,
                    "finish_time_us": _to_us(int(finish_cycles)),
                    "receive_time_us": receive_us,
                    "local_setup_us": local_setup_us,
                    "propagation_us": receive_us - dispatch_us,
                }
            )
    elif level == 1:
        # AICORE_TIMING fallback: AICPU records are absent (complete_task
        # bypassed). The AICore stream alone is the source of truth.
        for row in aicore_rows:
            core_id, task_token_raw, _reg_task_id, start_cycles, end_cycles, *rest = row
            r2s_cycles = int(rest[0]) if rest else 0
            core_id = int(core_id)
            task_token_raw = int(task_token_raw)
            start_us = _to_us(int(start_cycles))
            end_us = _to_us(int(end_cycles))
            local_setup_us = r2s_cycles * cycles_to_us_factor
            tasks.append(
                {
                    "task_id": task_token_raw,
                    "func_id": -1,
                    "core_id": core_id,
                    "core_type": _core_type(core_id),
                    "ring_id": (task_token_raw >> 32) & 0xFFFFFFFF,
                    "start_time_us": start_us,
                    "end_time_us": end_us,
                    "duration_us": end_us - start_us,
                    "dispatch_time_us": 0.0,
                    "finish_time_us": 0.0,
                    "receive_time_us": start_us - local_setup_us,
                    "local_setup_us": local_setup_us,
                    # propagation_us requires AICPU dispatch_ts; absent at level 1.
                }
            )

    tasks.sort(key=lambda t: int(t["task_id"]))

    total_unmatched = sum(unmatched_per_core.values())
    if total_unmatched > 0:
        worst = sorted(unmatched_per_core.items(), key=lambda kv: -kv[1])[:3]
        worst_str = ", ".join(f"core {c}: {n}" for c, n in worst)
        print(
            f"Warning: {total_unmatched} aicpu_task(s) had no matching AICore record (top offenders: {worst_str}); "
            "the missing AICore buffer(s) were dropped on rotation. Bump PLATFORM_AICORE_BUFFERS_PER_CORE if you "
            "see this regularly.",
            file=sys.stderr,
        )

    def _phase_us(pr):
        # Host already omits pop_hit / pop_miss for Complete records (terse
        # emit), so we don't need to re-strip zero deltas here.
        out = dict(pr)
        out["start_time_us"] = _to_us(int(pr.get("start_cycles", 0)))
        out["end_time_us"] = _to_us(int(pr.get("end_cycles", 0)))
        out.pop("start_cycles", None)
        out.pop("end_cycles", None)
        return out

    aicpu_scheduler_phases = []
    for thread_records in sched_phases_raw:
        converted = []
        for pr in thread_records:
            kind = pr.get("kind", "unknown")
            out = _phase_us(pr)
            # Downstream code branches on "phase" as the sched-record
            # discriminator; surface "kind" under that name.
            out["phase"] = kind
            out.pop("kind", None)
            converted.append(out)
        aicpu_scheduler_phases.append(converted)

    aicpu_orchestrator_phases = []
    for thread_records in orch_phases_raw:
        converted = []
        for pr in thread_records:
            out = _phase_us(pr)
            out["phase"] = "orch_submit"
            converted.append(out)
        aicpu_orchestrator_phases.append(converted)

    out = {
        "l2_swimlane_level": level,
        "tasks": tasks,
    }
    if aicpu_scheduler_phases:
        out["aicpu_scheduler_phases"] = aicpu_scheduler_phases
    if aicpu_orchestrator_phases:
        out["aicpu_orchestrator_phases"] = aicpu_orchestrator_phases
    if core_to_thread:
        out["core_to_thread"] = core_to_thread
    return out


def load_deps_json(deps_path):
    """Load a dep_gen replay output (``deps.json``).

    deps.json is the sole source of truth for the task graph in this tool:
    the device hot path no longer records per-task fanout (see PR #863). The
    typical workflow is a dep_gen run once per topology (``--enable-dep-gen``)
    to produce ``deps.json``, then any number of ``--enable-l2-swimlane`` runs
    that join their per-task timing against that captured graph.

    Returns:
        dict[int, list[int]] mapping ``pred_raw → [succ_raw, ...]``, or
        ``None`` if the file is missing, unreadable, or not v2-shaped. Tasks
        with no successors are absent from the dict (``defaultdict``-like
        lookup-miss semantics).
    """
    deps_path = Path(deps_path)
    if not deps_path.exists():
        return None
    try:
        with deps_path.open() as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        print(f"Warning: failed to read {deps_path}: {e}", file=sys.stderr)
        return None
    edges = data.get("edges")
    if not isinstance(edges, list):
        print(f"Warning: {deps_path} has no 'edges' array", file=sys.stderr)
        return None
    # The converter only needs flow-event endpoints (not the per-edge tensor
    # annotations). Project annotated edges down to a (pred, succ) set and
    # dedup so multiple annotated edges sharing the same pair (distinct arg
    # / source / overlap) collapse to a single flow event.
    by_pred: dict[int, list[int]] = defaultdict(list)
    seen: set[tuple[int, int]] = set()
    for edge in edges:
        if not isinstance(edge, dict):
            continue
        pred = normalize_pto2_task_id_int(edge.get("pred"))
        succ = normalize_pto2_task_id_int(edge.get("succ"))
        if pred is None or succ is None:
            continue
        key = (pred, succ)
        if key in seen:
            continue
        seen.add(key)
        by_pred[pred].append(succ)
    return dict(by_pred)


def load_deps_kernel_map(deps_path):
    """Build a ``task_id → kernel_ids[3]`` map from deps.json's ``tasks[]``.

    a2a3 dep_gen captures per-task ``kernel_ids = [aic, aiv0, aiv1]`` so the
    swimlane post-processor can resolve ``func_id`` at AICORE_TIMING (level=1)
    where the AICore record alone is on disk and carries ``func_id == -1``.
    The trace generator uses the per-record ``core_type`` to pick the right
    subslot: ``aic → kernel_ids[0]``, ``aiv → kernel_ids[1]`` (falling back
    to ``[2]`` if AIV0 is inactive). Same pattern fanout edges already use
    (deps.json is the offline-joined identity source).

    Returns:
        dict[int, list[int]] mapping ``task_id_raw → [aic, aiv0, aiv1]``,
        or ``None`` if the file is missing / unreadable / lacks the field.
        Entries without ``kernel_ids`` (pre-schema deps.json from older
        runs) are silently skipped — the caller treats a missing map as
        "no override available" and emits the ``func_-1_(...)`` fallback.
    """
    deps_path = Path(deps_path)
    if not deps_path.exists():
        return None
    try:
        with deps_path.open() as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return None
    kmap: dict[int, list[int]] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = normalize_pto2_task_id_int(task.get("task_id"))
        kids = task.get("kernel_ids")
        if tid is None or not isinstance(kids, list) or len(kids) != 3:
            continue
        kmap[tid] = [int(k) for k in kids]
    return kmap if kmap else None


def load_deps_block_map(deps_path):
    """Build a ``task_id → block_num`` map from deps.json's ``tasks[]``.

    Returns:
        dict[int, int] mapping ``task_id_raw → block_num``, or ``None`` if
        the file is missing / unreadable / lacks the field. Entries without
        ``block_num`` default to 1 (non-SPMD).
    """
    deps_path = Path(deps_path)
    if not deps_path.exists():
        return None
    try:
        with deps_path.open() as f:
            data = json.load(f)
    except (OSError, ValueError):
        return None
    tasks = data.get("tasks")
    if not isinstance(tasks, list):
        return None
    bmap: dict[int, int] = {}
    for task in tasks:
        if not isinstance(task, dict):
            continue
        tid = normalize_pto2_task_id_int(task.get("task_id"))
        if tid is None:
            continue
        try:
            block_num = int(task.get("block_num", 1))
        except (TypeError, ValueError):
            block_num = 1
        block_num = max(block_num, 1)
        bmap[tid] = block_num
    return bmap if bmap else None


def _identify_spmd_task_ids(task_map, deps_block_map=None):
    """Return task_ids whose dependency flow endpoints collapse to one subtask row."""
    spmd_ids: set[int] = set()
    if deps_block_map:
        for tid, block_num in deps_block_map.items():
            if block_num > 1:
                spmd_ids.add(tid)
    for tid, recs in task_map.items():
        if tid in spmd_ids or len(recs) <= 1:
            continue
        if deps_block_map and tid in deps_block_map:
            continue  # authoritative — don't second-guess block_num==1
        core_types = {r.get("core_type") for r in recs}
        if len(core_types) == 1:
            spmd_ids.add(tid)
    return spmd_ids


def _dependency_flow_anchor_rows(task_id, task_map, spmd_task_ids):
    """Dependency anchor rows.

    Non-SPMD keeps every subtask row. SPMD collapses rows by
    ``(func_id, task_id)`` and keeps the earliest-start row in each group, so
    MIX tasks with shared task_id but distinct AIC/AIV func_id values remain
    visible as separate dependency endpoints.
    """
    recs = task_map.get(task_id, [])
    if not recs:
        return []
    if task_id not in spmd_task_ids:
        return recs
    by_func: dict[int, dict] = {}
    for row in recs:
        func_id = row.get("func_id", -1)
        prev = by_func.get(func_id)
        if prev is None or (row.get("start_time_us", float("inf")), row.get("core_id", 0)) < (
            prev.get("start_time_us", float("inf")),
            prev.get("core_id", 0),
        ):
            by_func[func_id] = row
    return list(by_func.values())


def _dependency_flow_row_pairs(pred_id, succ_id, task_map, spmd_task_ids):
    """(pred_row, succ_row) pairs for one logical dependency edge."""
    pred_rows = _dependency_flow_anchor_rows(pred_id, task_map, spmd_task_ids)
    succ_rows = _dependency_flow_anchor_rows(succ_id, task_map, spmd_task_ids)
    if not pred_rows or not succ_rows:
        return []
    return [(pred_row, succ_row) for pred_row in pred_rows for succ_row in succ_rows]


def _dependency_task_fan_count(task_id, spmd_task_ids, task_map, deps_block_map=None):
    """Logical subtask count for dependency metadata (SPMD block_num, else 1)."""
    if task_id in spmd_task_ids:
        if deps_block_map and task_id in deps_block_map:
            return deps_block_map[task_id]
        return len(task_map[task_id])
    return 1


def _append_dependency_flow_pair(  # noqa: PLR0913
    events,
    flow_id,
    flow_name,
    src_pid,
    src_tid,
    src_ts,
    src_event_id,
    dst_pid,
    dst_tid,
    dst_ts,
    dst_event_id,
    *,
    input_task_count=1,
    output_task_count=1,
):
    flow_s = {
        "cat": "flow",
        "id": flow_id,
        "name": flow_name,
        "ph": "s",
        "pid": src_pid,
        "tid": src_tid,
        "ts": src_ts,
        "input_task_count": input_task_count,
        "output_task_count": output_task_count,
    }
    if src_event_id is not None:
        flow_s["bind_id"] = src_event_id
    events.append(flow_s)

    flow_f = {
        "cat": "flow",
        "id": flow_id,
        "name": flow_name,
        "ph": "f",
        "pid": dst_pid,
        "tid": dst_tid,
        "ts": dst_ts,
        "bp": "e",
        "input_task_count": input_task_count,
        "output_task_count": output_task_count,
    }
    if dst_event_id is not None:
        flow_f["bind_id"] = dst_event_id
    events.append(flow_f)


def resolve_func_id_from_kernel_map(task_id, core_type, kernel_map):
    """Look up the active ``func_id`` for an AICORE_TIMING record via dep_gen.

    Picks the kernel_ids[3] subslot by record ``core_type``. Returns the
    resolved func_id (>= 0) on a hit, or -1 if no usable subslot was found
    (caller keeps the original -1 and emits the ``func_-1_(...)`` fallback
    name). The choice for ``aiv`` prefers AIV0 ([1]) and falls back to AIV1
    ([2]) — works for pure-AIV and MIX-with-single-AIV records; for MIX
    records that span both AIVs the host swimlane record only tells us the
    lane is "aiv", so the resolver may name an AIV1 lane after AIV0's
    kernel. Acceptable trade-off until the host emits a lane-disambiguated
    core_type ("aiv0" / "aiv1").
    """
    if kernel_map is None or task_id is None:
        return -1
    kids = kernel_map.get(int(task_id))
    if not kids:
        return -1
    if core_type == "aic":
        return kids[0] if kids[0] >= 0 else -1
    # "aiv": prefer AIV0, fall back to AIV1.
    for idx in (1, 2):
        if kids[idx] >= 0:
            return kids[idx]
    return -1


def load_kernel_config(config_path):
    """Load kernel configuration from kernel_config.py file.

    Args:
        config_path: Path to kernel_config.py file

    Returns:
        dict: Mapping from func_id (as string) to function name
              Example: {"0": "QK", "1": "SF", "2": "PV", "3": "UP"}
              Entries without 'func_id' or 'name' are skipped with a warning

    Raises:
        ValueError: If file cannot be loaded or KERNELS definition is missing
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ValueError(f"Kernel config file not found: {config_path}")

    spec = importlib.util.spec_from_file_location("kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract func_id to name mapping from KERNELS list
    if not hasattr(module, "KERNELS"):
        raise ValueError("kernel_config.py missing KERNELS definition")

    func_id_to_name = {}
    for kernel in module.KERNELS:
        # Skip entries without func_id
        if "func_id" not in kernel:
            print(f"Warning: Kernel entry missing 'func_id', skipping: {kernel}", file=sys.stderr)
            continue

        func_id = kernel["func_id"]

        # If name is missing, we'll fall back to default naming (Func_{func_id})
        if "name" not in kernel:
            print(
                f"Warning: Kernel entry for func_id={func_id} missing 'name', will use default naming",
                file=sys.stderr,
            )
            continue

        # Store as string to match JSON format
        func_id_to_name[str(func_id)] = kernel["name"]

    return func_id_to_name


def load_func_names_json(json_path):
    """Load name mapping from a SceneTest JSON file.

    Each level's mapping carries ``callable_id_to_name`` for its
    next-level-down callables and a ``level`` tag.  The tool uses
    ``callable_id_to_name`` directly — no cross-level merging.

    Returns:
        tuple: (callable_id_to_name dict, orchestrator_name str or None)
    """
    path = Path(json_path)
    if not path.exists():
        raise ValueError(f"Func names JSON not found: {path}")
    with open(path) as f:
        data = json.load(f)
    return data.get("callable_id_to_name", {}), data.get("orchestrator_name")


def print_task_statistics(tasks, func_id_to_name=None):
    """Print task statistics grouped by func_id.

    Exec = kernel execution time (end_time_us - start_time_us) on AICore.
    Latency = AICPU view: finish_time_us - dispatch_time_us (includes head OH + Exec + tail OH).
    High Latency with low Exec means scheduler/polling overhead (tail OH = finish_ts recorded
    when the scheduler loop next sees the completed handshake; reordering the loop to process
    completed tasks first reduces this).

    Args:
        tasks: List of task dicts
        func_id_to_name: Optional dict mapping func_id to function name
    """
    # Group tasks by func_id with extended metrics
    func_stats: defaultdict[Any, dict[str, Any]] = defaultdict(
        lambda: {
            "durations": [],
            "head_overheads": [],
            "tail_overheads": [],
            "propagations": [],  # dispatch_ts → AICore receive_time (NoC + FFTS)
            "local_setups": [],  # receive_time → start_time (dcci + ack on AICore)
            "latencies": [],
            "total_exec_time": 0.0,
            "total_latency": 0.0,
        }
    )

    # Track global min dispatch and max finish times
    min_dispatch_time = float("inf")
    max_finish_time = float("-inf")

    for task in tasks:
        func_id = task["func_id"]
        duration = task["duration_us"]
        func_stats[func_id]["durations"].append(duration)

        # Calculate new metrics if dispatch_time_us and finish_time_us are available
        if "dispatch_time_us" in task and "finish_time_us" in task:
            dispatch_time = task["dispatch_time_us"]
            finish_time = task["finish_time_us"]
            start_time = task["start_time_us"]
            end_time = task["end_time_us"]

            # Head overhead: start_time_us - dispatch_time_us
            head_overhead = start_time - dispatch_time
            func_stats[func_id]["head_overheads"].append(head_overhead)

            # Tail overhead: finish_time_us - end_time_us
            tail_overhead = finish_time - end_time
            func_stats[func_id]["tail_overheads"].append(tail_overhead)

            # Head OH split (v3 schema only — falls back to absent when the
            # AICore record came from a pre-receive_time build).
            if "propagation_us" in task:
                func_stats[func_id]["propagations"].append(task["propagation_us"])
            if "local_setup_us" in task:
                func_stats[func_id]["local_setups"].append(task["local_setup_us"])

            # Latency: finish_time_us - dispatch_time_us
            latency = finish_time - dispatch_time
            func_stats[func_id]["latencies"].append(latency)

            # Accumulate execution time and latency for ratio calculation
            func_stats[func_id]["total_exec_time"] += duration
            func_stats[func_id]["total_latency"] += latency

            # Track global times
            min_dispatch_time = min(min_dispatch_time, dispatch_time)
            max_finish_time = max(max_finish_time, finish_time)

    # Print statistics
    print("\n" + "=" * 140)
    print("Task Statistics by Function")
    print("  Exec = kernel time on AICore; Latency = dispatch->finish (incl. head OH + Exec + tail OH)")
    print("  Head OH split (v3): Prop = NoC propagation (dispatch_ts→AICore receive); Local = dcci+ack (receive→start)")
    print("=" * 140)
    print(
        f"{'Func_ID':<8} {'Func_Name':<12} {'Count':>5}   {'Avg Exec(us)':>12}  "
        f"{'Avg Latency(us)':>15}  {'Exec%':>6}   {'Avg Head OH(us)':>15}  {'Avg Tail OH(us)':>15}  "
        f"{'Avg Prop(us)':>12}  {'Avg Local(us)':>13}"
    )
    print("-" * 140)

    # Sort by func_id for consistent output
    total_count = 0
    total_duration = 0.0

    for func_id in sorted(func_stats.keys()):
        stats = func_stats[func_id]
        durations = stats["durations"]
        count = len(durations)
        sum_duration = sum(durations)
        avg_duration = sum_duration / count

        # Accumulate totals
        total_count += count
        total_duration += sum_duration

        # Get function name
        if func_id_to_name and str(func_id) in func_id_to_name:
            func_name = func_id_to_name[str(func_id)]
        else:
            func_name = f"func_{_func_id_to_letter(func_id)}"

        # Calculate averages
        avg_head_overhead = (
            sum(stats["head_overheads"]) / len(stats["head_overheads"]) if stats["head_overheads"] else 0
        )
        avg_tail_overhead = (
            sum(stats["tail_overheads"]) / len(stats["tail_overheads"]) if stats["tail_overheads"] else 0
        )
        avg_latency = stats["total_latency"] / count if count > 0 else 0
        # `None` (not NaN) signals "no v3 receive_time data on this func" so
        # the print line below renders a dash. NaN would force ruff's
        # PLR0124 self-compare idiom.
        avg_propagation = sum(stats["propagations"]) / len(stats["propagations"]) if stats["propagations"] else None
        avg_local_setup = sum(stats["local_setups"]) / len(stats["local_setups"]) if stats["local_setups"] else None

        # Calculate execution ratio: total_exec_time / total_latency
        exec_ratio = (stats["total_exec_time"] / stats["total_latency"] * 100) if stats["total_latency"] > 0 else 0

        prop_str = f"{avg_propagation:>12.2f}" if avg_propagation is not None else f"{'-':>12}"
        local_str = f"{avg_local_setup:>13.2f}" if avg_local_setup is not None else f"{'-':>13}"
        print(
            f"{func_id:<8} {func_name:<12} {count:>5}   {avg_duration:>12.2f}  {avg_latency:>15.2f}  "
            f"{exec_ratio:>5.1f}%   {avg_head_overhead:>15.2f}  {avg_tail_overhead:>15.2f}  "
            f"{prop_str}  {local_str}"
        )

    # Print total row
    print("-" * 140)

    # Calculate total latency (sum of all latencies)
    total_latency_sum = sum(stats["total_latency"] for stats in func_stats.values())
    print(f"{'TOTAL':<21} {total_count:>5}   {total_duration:>12.2f}  {total_latency_sum:>15.2f}")

    # Print total test execution time
    if min_dispatch_time != float("inf") and max_finish_time != float("-inf"):
        total_test_time = max_finish_time - min_dispatch_time
        print(f"\nTotal Test Time: {total_test_time:.2f} us (from earliest dispatch to latest finish)")

    # Task execution vs Scheduler overhead summary
    if total_count > 0 and total_latency_sum > 0:
        avg_exec_us = total_duration / total_count
        avg_latency_us = total_latency_sum / total_count
        exec_latency_ratio_pct = total_duration / total_latency_sum * 100
        print("\n--- Task execution vs Scheduler overhead ---")
        print(
            f"  Per-task (all):  Avg Exec = {avg_exec_us:.2f} us,  "
            f"Avg Latency (dispatch->finish) = {avg_latency_us:.2f} us,  "
            f"Exec/Latency = {exec_latency_ratio_pct:.2f}%"
        )
        print("  (Latency = dispatch→finish; Exec = AICore kernel time per task)")

    print("=" * 110)


def build_overhead_counter_events(tasks, deps_edges, pid=2):  # noqa: PLR0912
    """Per-engine + system overhead counter tracks for the Perfetto trace.

    Emits 8 counter (``"ph":"C"``) series under the AICPU Scheduler process
    (``pid=2``) — it is scheduler-overhead analysis, so it lives in the sched
    group — overlaid alongside the scheduler's own tracks and the AICore task
    bars to see, at every instant, why time is or is not wasted (see
    docs/dfx/sched-overhead-model.md):

      {aic,aiv}_idle     core of that type NOT executing (k - running)
      {aic,aiv}_ready    that type's tasks whose producers have ended but that
                         are not yet dispatched. A MIX task (records on BOTH
                         engines) counts for BOTH.
      {aic,aiv}_overhead 1 when idle>0 AND ready>0 (free core + ready work the
                         scheduler hasn't placed)
      all_overhead       1 when EVERY present engine is overhead (whole chip
                         blocked — e.g. a MIX waiting to launch)
      has_overhead       1 when every engine that HAS ready work is overhead
                         (engines with no work are ignored)

    Readiness keys off producer end_time; a task whose predecessors are all
    absent from the perf set falls back to its own dispatch (no unverifiable
    early readiness). Needs ``deps_edges`` (pred -> [succ]); returns [] without it.
    """
    if not deps_edges or not tasks:
        return []

    def _u64(x):
        try:
            x = int(x)
        except (TypeError, ValueError):
            return None
        return x & ((1 << 64) - 1) if x < 0 else x

    cores_by_type = defaultdict(set)
    for t in tasks:
        cores_by_type[t.get("core_type")].add(t.get("core_id"))
    types = [ty for ty in ("aic", "aiv") if cores_by_type.get(ty)]
    if not types:
        return []
    k = {ty: len(cores_by_type[ty]) for ty in types}

    types_of = defaultdict(set)
    disp, end = {}, {}
    for t in tasks:
        tid = _u64(t.get("task_id"))
        if tid is None:
            continue
        types_of[tid].add(t.get("core_type"))
        disp[tid] = min(disp.get(tid, t["dispatch_time_us"]), t["dispatch_time_us"])
        end[tid] = max(end.get(tid, t["end_time_us"]), t["end_time_us"])

    preds = defaultdict(set)
    for pred, succs in deps_edges.items():
        p = _u64(pred)
        for s in succs:
            ss = _u64(s)
            if p is not None and ss is not None and p != ss:
                preds[ss].add(p)
    ready = {}
    for tid, dp in disp.items():
        in_perf = [p for p in preds.get(tid, ()) if p in end]
        ready[tid] = max(end[p] for p in in_perf) if in_perf else dp

    w0 = min(t["start_time_us"] for t in tasks)
    w1 = max(t["end_time_us"] for t in tasks)
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

    tids = {
        "oh_aic_idle": 9101,
        "oh_aic_ready": 9102,
        "oh_aic_overhead": 9103,
        "oh_aiv_idle": 9111,
        "oh_aiv_ready": 9112,
        "oh_aiv_overhead": 9113,
        "oh_all_overhead": 9121,
        "oh_has_overhead": 9122,
    }
    # No process metadata: pid=2 is the AICPU Scheduler process (named/sorted
    # elsewhere). Emitting it here would override that — these counters just
    # join the scheduler group as extra tracks (names prefixed "oh_").
    events = []
    run_c = {ty: 0 for ty in types}
    rw_c = {ty: 0 for ty in types}
    prev = {}
    order = sorted(times)
    for i in range(len(order) - 1):
        a = order[i]
        for ty in types:
            run_c[ty] += run[ty][a]
            rw_c[ty] += rw[ty][a]
        ov = {}
        vals = {}
        for ty in types:
            idle = k[ty] - run_c[ty]
            vals[f"oh_{ty}_idle"] = idle
            vals[f"oh_{ty}_ready"] = rw_c[ty]
            ov[ty] = 1 if (idle > 0 and rw_c[ty] > 0) else 0
            vals[f"oh_{ty}_overhead"] = ov[ty]
        vals["oh_all_overhead"] = 1 if all(ov[ty] for ty in types) else 0
        work = [ty for ty in types if rw_c[ty] > 0]
        vals["oh_has_overhead"] = 1 if (work and all(k[ty] - run_c[ty] > 0 for ty in work)) else 0
        for name, v in vals.items():
            if name in tids and prev.get(name) != v:
                events.append(
                    {
                        "ph": "C",
                        "cat": "overhead",
                        "name": name,
                        "pid": pid,
                        "tid": tids[name],
                        "ts": round(a, 3),
                        "args": {name: v},
                    }
                )
                prev[name] = v
    return events


def generate_chrome_trace_json(  # noqa: PLR0912, PLR0913, PLR0915
    tasks,
    output_path,
    func_id_to_name=None,
    verbose=False,
    scheduler_phases=None,
    orchestrator_phases=None,
    core_to_thread=None,
    orchestrator_name=None,
    deps_edges=None,
    deps_kernel_map=None,
    deps_block_map=None,
    emit_overhead=False,
):
    """Generate Chrome Trace Event Format JSON from task data.

    Args:
        tasks: List of task dicts with fields:
            - task_id, func_id, core_id, core_type
            - start_time_us, end_time_us, duration_us
            - dispatch_time_us (optional, AICPU dispatch timestamp)
            - finish_time_us (optional, AICPU finish timestamp)
        output_path: Path to output JSON file
        func_id_to_name: Optional dict mapping func_id to function name
        verbose: Print progress information
        scheduler_phases: Optional list of per-thread phase record lists (l2_swimlane_level >= 3)
        orchestrator_phases: Optional list of per-task orchestrator phase records (l2_swimlane_level >= 4)
        core_to_thread: Optional list mapping core_id (index) to scheduler thread index (-1 = unassigned)

    Generates processes in the trace:
        - pid=1 "AICPU Orchestrator": orchestrator phase bars (l2_swimlane_level >= 4)
        - pid=2 "AICPU Scheduler": scheduler phase bars (l2_swimlane_level >= 3)
        - pid=3 "Scheduler View": dispatch_time_us to finish_time_us (AICPU perspective)
        - pid=4 "Worker View": per-subtask kernel execution on physical cores
    """
    if verbose:
        print("Generating Chrome Trace JSON...")
        print(f"  Tasks: {len(tasks)}")
        if func_id_to_name:
            print(f"  Function names: {len(func_id_to_name)} entries")

    # Step 1: Build core_to_tid mapping (using only core_id, not core_type)
    unique_cores = set()
    for task in tasks:
        unique_cores.add(task["core_id"])

    core_to_tid = {}
    for core_id in sorted(unique_cores):
        core_to_tid[core_id] = 10000 + core_id * 10

    if verbose:
        print(f"  Unique cores: {len(unique_cores)}")

    # Recover func_id for AICORE_TIMING (level=1) records, which the host
    # emits as func_id=-1. Resolve once here against dep_gen's per-task
    # kernel_ids[3] (picking the subslot by core_type) and write it back onto
    # the task, so every downstream consumer — Worker View, Scheduler View, and
    # event-hints — sees the same real func_id. See
    # resolve_func_id_from_kernel_map() for the AIV0-vs-AIV1 tie-break and the
    # host-side contract.
    if deps_kernel_map is not None:
        for task in tasks:
            if int(task["func_id"]) < 0:
                resolved = resolve_func_id_from_kernel_map(task["task_id"], task.get("core_type"), deps_kernel_map)
                if resolved >= 0:
                    task["func_id"] = resolved

    # Step 2: Generate JSON events
    events = []

    # Metadata event: Process names and sort order.
    # pid is renumbered in pipeline order (top → bottom in Perfetto):
    #   pid=1  AICPU Orchestrator  (submits tasks — earliest)
    #   pid=2  AICPU Scheduler     (pops ready, dispatches, completes)
    #   pid=3  Scheduler View      (AICPU-eye view of each worker's dispatch→finish)
    #   pid=4  Worker View         (physical AIC/AIV execution rows)
    # sort_index intentionally equals pid so JSON ordering is self-evident.
    task_map: dict[int, list] = defaultdict(list)
    for t in tasks:
        task_map[t["task_id"]].append(t)
    spmd_task_ids = _identify_spmd_task_ids(task_map, deps_block_map)

    events.append({"args": {"name": "Worker View"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 4})
    events.append({"args": {"sort_index": 4}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 4})

    # Check if any task has AICPU timestamps
    has_aicpu_data = any(task.get("dispatch_time_us", 0) >= 0 and task.get("finish_time_us", 0) > 0 for task in tasks)

    if has_aicpu_data:
        events.append(
            {"args": {"name": "Scheduler View"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 3}
        )
        events.append(
            {"args": {"sort_index": 3}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 3}
        )

    # Metadata events: Thread names (one per core)
    for core_id, tid in core_to_tid.items():
        # Find first task with this core_id to get core_type
        core_type = None
        for task in tasks:
            if task["core_id"] == core_id:
                core_type = task["core_type"]
                break

        # core_type is now a string ("aic" or "aiv")
        core_type_str = (core_type or "unknown").upper()
        thread_name = f"{core_type_str}_{core_id}"
        events.append(
            {"args": {"name": thread_name}, "cat": "__metadata", "name": "thread_name", "ph": "M", "pid": 4, "tid": tid}
        )

    # Duration events (Complete events "X")
    # Build task_id -> event_id mapping for flow events
    task_to_event_id: dict[tuple[int, int], int] = {}
    task_to_aicpu_event_id: dict[tuple[int, int], int] = {}
    task_to_aicpu_tid: dict[tuple[int, int], int] = {}
    event_id = 0

    # Invert deps (pred -> [succ]) into a fanin map (succ -> [pred]) so each task
    # bar can show both its consumers (fanout) and producers (fanin) with counts.
    fanin_map: dict[int, list] = defaultdict(list)
    if deps_edges:
        for pred, succs in deps_edges.items():
            for succ in succs:
                fanin_map[succ].append(pred)

    for task in tasks:
        tid = core_to_tid[task["core_id"]]
        local_setup_us = task.get("local_setup_us", 0.0) or 0.0
        receive_time_us = task.get("receive_time_us")
        ts = receive_time_us if receive_time_us is not None else task["start_time_us"]
        dur = task["end_time_us"] - ts

        # func_id is already resolved (level=1 records recovered from
        # dep_gen's kernel_ids up front; see the pre-pass above). Without a
        # deps.json the id stays -1 and the lane is named task(rXtY).
        func_id = task["func_id"]
        tdisp = format_task_display(task["task_id"])
        task_name = _task_display_name(func_id, func_id_to_name, tdisp, spmd=task["task_id"] in spmd_task_ids)

        # fanout (consumers) / fanin (producers) hints from deps.json — the device
        # hot path no longer carries them. Each leads with the degree (count) so
        # broadcast / reduction nodes are obvious without expanding the list.
        fanout_ids = deps_edges.get(task["task_id"], []) if deps_edges else []
        fanin_ids = fanin_map.get(task["task_id"], [])
        fanout_str = f"{len(fanout_ids)}: [" + ", ".join(format_task_display(x) for x in fanout_ids) + "]"
        fanin_str = f"{len(fanin_ids)}: [" + ", ".join(format_task_display(x) for x in fanin_ids) + "]"

        events.append(
            {
                "args": {
                    "event-hint": f"Task:{tdisp}, FuncId:{func_id}, CoreId:{task['core_id']}",
                    "fanout-hint": fanout_str,
                    "fanin-hint": fanin_str,
                    "duration-us": dur,
                    "kernel-duration-us": task["duration_us"],
                    "local_setup_us": local_setup_us,
                    "taskId": task["task_id"],
                },
                "cat": "event",
                "id": event_id,
                "name": task_name,
                "ph": "X",
                "pid": 4,
                "tid": tid,
                "ts": ts,
                "dur": dur,
            }
        )

        # Record mapping for flow events
        task_to_event_id[(task["task_id"], task["core_id"])] = event_id
        event_id += 1

    # Scheduler View duration events (dispatch_time to finish_time)
    # Assign overlapping tasks on the same core to different tids so Perfetto
    # renders each bar on its own row (Perfetto requires strict nesting on a tid).
    if has_aicpu_data:
        # Build per-core sorted task lists and assign sub-lanes.
        # Each core gets a base tid from core_to_tid; overlapping tasks get base+1.
        _core_aicpu_tasks: dict[int, list] = defaultdict(list)
        for task in tasks:
            d = task.get("dispatch_time_us", 0)
            f = task.get("finish_time_us", 0)
            if d < 0 or f <= 0:
                continue
            _core_aicpu_tasks[task["core_id"]].append(task)
        for ct_list in _core_aicpu_tasks.values():
            ct_list.sort(key=lambda t: t["dispatch_time_us"])

        aicpu_tid_set: set[int] = set()
        for core_id, ct_list in _core_aicpu_tasks.items():
            base_tid = core_to_tid[core_id]
            # Greedy lane assignment: track finish time per sub-lane
            lane_finish = [0.0]  # lane 0 = base_tid
            for task in ct_list:
                d = task["dispatch_time_us"]
                assigned = -1
                for lane_idx, lf in enumerate(lane_finish):
                    if lf <= d:
                        assigned = lane_idx
                        break
                if assigned < 0:
                    assigned = len(lane_finish)
                    lane_finish.append(0.0)
                lane_finish[assigned] = task["finish_time_us"]
                tid = base_tid if assigned == 0 else base_tid + assigned
                task_to_aicpu_tid[(task["task_id"], task["core_id"])] = tid
                aicpu_tid_set.add(tid)

        # Thread name metadata for Scheduler View (one entry per unique tid used)
        for core_id, base_tid in core_to_tid.items():
            ct_list = _core_aicpu_tasks.get(core_id)
            core_type_str = ct_list[0]["core_type"].upper() if ct_list else "unknown"
            base_name = f"{core_type_str}_{core_id}"
            # Base lane always gets metadata (even if no tasks, for consistency)
            if base_tid in aicpu_tid_set or not aicpu_tid_set:
                events.append(
                    {
                        "args": {"name": base_name},
                        "cat": "__metadata",
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 3,
                        "tid": base_tid,
                    }
                )
            # Overflow lane (at most one: dual-slot dispatch means max 2 concurrent tasks per core)
            overflow_tid = base_tid + 1
            if overflow_tid in aicpu_tid_set:
                events.append(
                    {
                        "args": {"name": base_name},
                        "cat": "__metadata",
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 3,
                        "tid": overflow_tid,
                    }
                )

        for task in tasks:
            dispatch_us = task.get("dispatch_time_us", 0)
            finish_us = task.get("finish_time_us", 0)
            # 0us is a valid timestamp (base-time aligned); only reject negative/invalid values.
            if dispatch_us < 0 or finish_us <= 0:
                continue

            tid = task_to_aicpu_tid.get((task["task_id"], task["core_id"]), core_to_tid[task["core_id"]])
            aicpu_dur = finish_us - dispatch_us

            # Get function name if available (task(rXtY) when no deps.json
            # resolved the func_id; see _task_display_name).
            func_id = task["func_id"]
            tdisp = format_task_display(task["task_id"])
            task_name = _task_display_name(func_id, func_id_to_name, tdisp, spmd=task["task_id"] in spmd_task_ids)

            events.append(
                {
                    "args": {
                        "event-hint": f"Task:{tdisp}, FuncId:{func_id}, CoreId:{task['core_id']}",
                        "dispatch-time-us": dispatch_us,
                        "finish-time-us": finish_us,
                        "aicpu-duration-us": aicpu_dur,
                        "taskId": task["task_id"],
                    },
                    "cat": "event",
                    "id": event_id,
                    "name": task_name,
                    "ph": "X",
                    "pid": 3,
                    "tid": tid,
                    "ts": dispatch_us,
                    "dur": aicpu_dur,
                }
            )
            task_to_aicpu_event_id[(task["task_id"], task["core_id"])] = event_id
            event_id += 1

    # Flow events (Flow events "s" and "f" for dependencies). Edges come from
    # deps.json (dep_gen replay); without one we emit no flow events at all,
    # since the device hot path no longer carries fanout (PR #863).
    # SPMD logical tasks anchor dependency arrows on the min-core_id subtask
    # row per core_type (AIC and AIV separately when both are present).
    flow_id = 0
    hb_violation_count = 0
    deps_flow_count = 0
    edges_by_pred = deps_edges or {}
    flow_epsilon = 0.01

    for pred_id, succ_ids in edges_by_pred.items():
        if pred_id not in task_map:
            continue

        for succ_id in succ_ids:
            if succ_id not in task_map:
                if verbose:
                    print(
                        f"Warning: Task {format_task_display(pred_id)} (raw {pred_id}) "
                        f"references non-existent successor {format_task_display(succ_id)} (raw {succ_id})"
                    )
                continue

            row_pairs = _dependency_flow_row_pairs(pred_id, succ_id, task_map, spmd_task_ids)
            if not row_pairs:
                continue

            output_task_count = _dependency_task_fan_count(pred_id, spmd_task_ids, task_map, deps_block_map)
            input_task_count = _dependency_task_fan_count(succ_id, spmd_task_ids, task_map, deps_block_map)
            for pred_row, succ_row in row_pairs:
                src_ts_end = pred_row["end_time_us"] - flow_epsilon
                dst_ts_start = succ_row.get("receive_time_us") or succ_row["start_time_us"]
                hb_violated = (src_ts_end + flow_epsilon) > dst_ts_start
                flow_name = "hb_violation" if hb_violated else "dependency"
                if hb_violated:
                    hb_violation_count += 1
                _append_dependency_flow_pair(
                    events,
                    flow_id,
                    flow_name,
                    4,
                    core_to_tid[pred_row["core_id"]],
                    src_ts_end,
                    task_to_event_id.get((pred_id, pred_row["core_id"])),
                    4,
                    core_to_tid[succ_row["core_id"]],
                    dst_ts_start,
                    task_to_event_id.get((succ_id, succ_row["core_id"])),
                    input_task_count=input_task_count,
                    output_task_count=output_task_count,
                )
                flow_id += 1
                deps_flow_count += 1

    if verbose:
        if deps_edges is not None:
            print(f"  Dependency flow events: {deps_flow_count} edges (source: deps.json)")
            if spmd_task_ids:
                print(
                    f"  SPMD tasks: {len(spmd_task_ids)} logical task(s); "
                    "dependency arrows anchor on min core_id subtask per core_type"
                )
        else:
            print("  Flow events: 0 (no deps.json — re-run dep_gen and pass --deps-json to add arrows)")
        if hb_violation_count > 0:
            print(f"  Happens-before violations: {hb_violation_count} edge(s) flagged as 'hb_violation'")

    # AICPU Scheduler phase events (l2_swimlane_level >= 3)
    if scheduler_phases:

        def sched_lane_tid(thread_idx, lane=0):
            return 30000 + thread_idx * 10 + lane

        # Process metadata
        events.append(
            {"args": {"name": "AICPU Scheduler"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 2}
        )
        events.append(
            {"args": {"sort_index": 2}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 2}
        )

        # Phase color mapping. The Perfetto sched lane only renders the
        # two work phases (complete, dispatch). Idle is the wall-clock gap
        # between consecutive work bars — Perfetto's empty-track regions
        # already convey that visually, so we don't paint a synthetic bar
        # for it. (Idle is still tallied numerically by
        # sched_overhead_analysis.py Part 2 via gap reconstruction.)
        phase_colors = {
            # Outer phases — mutually time-exclusive within an iter
            "complete": "good",  # green
            "dispatch": "terrible",  # red
            "release": "olive",  # deferred-release drain (on_task_release work)
            "dummy": "grey",  # dummy_drain pass (Resolve nests inside)
            "early_dispatch": "rail_animation",  # speculative early-dispatch staging
            # Inner phase — nests inside Complete or Dummy via time containment
            "resolve": "vsync_highlight_color",  # on_task_complete: walk consumer list
            # Separate-lane (Worker View AICPU_N) — fallback color if it ever lands on Sched
            "dummy_task": "grey",
        }

        # Per-complete subtask-finish counts surface "how many AICore FINs
        # did the AICPU poll in this phase" — useful context that
        # tasks_processed (logical task count) doesn't convey. Computed
        # with "phase contains finish_us" semantics so it matches how the
        # AICPU actually attributes finishes to its polling windows.
        complete_phases_by_thread_pre = []
        complete_starts_by_thread_pre = []
        for thread_records in scheduler_phases:
            sorted_completes = sorted(
                (r for r in thread_records if r.get("phase") == "complete"),
                key=lambda r: r["start_time_us"],
            )
            complete_phases_by_thread_pre.append(sorted_completes)
            complete_starts_by_thread_pre.append([c["start_time_us"] for c in sorted_completes])

        def _find_containing_complete(thread_idx: int, finish_us: float):
            # Bisect into the per-thread sorted start_time_us list. Complete
            # phases on a thread don't overlap, so the only complete that can
            # CONTAIN finish_us is the last one whose start is <= finish_us
            # (= entry at idx-1 after bisect_right). Fall back to the next
            # starting complete (entry at idx) if it doesn't contain.
            phases = complete_phases_by_thread_pre[thread_idx]
            starts = complete_starts_by_thread_pre[thread_idx]
            if not phases:
                return None
            idx = bisect.bisect_right(starts, finish_us)
            if idx > 0:
                prev_c = phases[idx - 1]
                if prev_c["start_time_us"] <= finish_us <= prev_c["end_time_us"]:
                    return prev_c
            if idx < len(phases):
                return phases[idx]
            return None

        finishes_per_complete: dict[int, int] = defaultdict(int)
        if core_to_thread:
            for t in tasks:
                f_us = t.get("finish_time_us")
                if f_us is None or f_us < 0:
                    continue
                t_cid = t["core_id"]
                if t_cid >= len(core_to_thread):
                    continue
                t_thr = core_to_thread[t_cid]
                if t_thr < 0 or t_thr >= len(complete_phases_by_thread_pre):
                    continue
                t_comp = _find_containing_complete(t_thr, f_us)
                if t_comp is None:
                    continue
                finishes_per_complete[id(t_comp)] += 1

        # Worker View (pid=4) AICPU lanes: AICPU_N treated as a worker tier
        # alongside AIC_N (matrix) and AIV_N (vector). DummyTask phases (per
        # dummy DAG-node identity markers from the scheduler thread that
        # drained them) and Alloc phases (per `alloc_tensors()` call from the
        # orchestrator) both render on these lanes — they are the activities
        # that an AICPU performs when it acts as a worker. The lane index is
        # the AICPU id; with the current 1:1 sched-thread-to-AICPU mapping,
        # `thread_idx` IS the AICPU id, and the single orch is on AICPU 0.
        # Tids 19000+aicpu_id keep them visually grouped after the physical
        # AIC/AIV lanes (which sit at 10000+core_id*10).
        AICPU_TID_BASE = 19000  # noqa: N806
        num_aicpu_lanes = max(len(scheduler_phases), len(orchestrator_phases or []))
        for aicpu_id in range(num_aicpu_lanes):
            events.append(
                {
                    "args": {"name": f"AICPU_{aicpu_id}"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 4,
                    "tid": AICPU_TID_BASE + aicpu_id,
                }
            )

        # Dummy task X event width — start/end on the device coincide
        # (identification is a single MSR read); render as a 0.02 µs sliver
        # so Perfetto picks it up instead of collapsing it to a hairline.
        DUMMY_BAR_MIN_DUR_US = 0.02  # noqa: N806

        for thread_idx, thread_records in enumerate(scheduler_phases):
            tid = sched_lane_tid(thread_idx, 0)
            resolve_tid = sched_lane_tid(thread_idx, 1)

            # Thread name metadata
            events.append(
                {
                    "args": {"name": f"Sched_{thread_idx}"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 2,
                    "tid": tid,
                }
            )
            if any(record.get("phase") == "resolve" for record in thread_records):
                events.append(
                    {
                        "args": {"name": f"Sched_{thread_idx}"},
                        "cat": "__metadata",
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 2,
                        "tid": resolve_tid,
                    }
                )

            # Render work phases (complete / dispatch) plus the real operations
            # that otherwise hide inside an idle stretch (poll = completion-scan
            # that retired nothing; release = on_task_release drain). Genuine
            # spin emits no record and shows as a blank gap.
            #
            # `dummy_task` is special: it does NOT live on the sched track —
            # it represents a DAG fence node briefly inhabiting the AICPU as a
            # virtual worker, so we route it to Worker View (pid=4) AICPU_N
            # (where N = the AICPU id of the sched thread that drained it).
            for record in thread_records:
                phase = record.get("phase", "unknown")
                if phase == "dummy_task":
                    start_us = record["start_time_us"]
                    end_us = record["end_time_us"]
                    dur = max(end_us - start_us, DUMMY_BAR_MIN_DUR_US)
                    task_id_low32 = record.get("tasks_processed", 0)
                    events.append(
                        {
                            "args": {
                                "phase": "dummy_task",
                                "loop_iter": record.get("loop_iter", 0),
                                "dummy_task_id_low32": task_id_low32,
                                "event-hint": f"dummy(t{task_id_low32})",
                            },
                            "cat": "event",
                            "cname": "grey",
                            "name": f"dummy(t{task_id_low32})",
                            "ph": "X",
                            "pid": 4,
                            "tid": AICPU_TID_BASE + thread_idx,
                            "ts": start_us,
                            "dur": dur,
                        }
                    )
                    continue
                if phase not in ("complete", "dispatch", "release", "resolve", "early_dispatch", "dummy"):
                    continue
                start_us = record["start_time_us"]
                end_us = record["end_time_us"]
                dur = end_us - start_us
                tasks_processed = record.get("tasks_processed", 0)

                # Queue-depth snapshot fields. Layout per
                # L2SwimlaneAicpuSchedPhaseRecord docstring: [AIC, AIV, MIX].
                shared_at_start = record.get("shared_at_start")
                shared_at_end = record.get("shared_at_end")
                depths_valid = (
                    isinstance(shared_at_start, list)
                    and isinstance(shared_at_end, list)
                    and len(shared_at_start) == 3
                    and len(shared_at_end) == 3
                )

                # Phase block. When queue depths are present, fold them into
                # args so hover on a complete/dispatch bar surfaces the
                # before/after queue state alongside the phase metadata.
                phase_args = {
                    "phase": phase,
                    "loop_iter": record.get("loop_iter", 0),
                    "tasks_processed": tasks_processed,
                }
                if depths_valid:
                    # Perfetto's args SQL parses key names; `[...]` looks like
                    # an array-index op and crashes the details-panel query.
                    # Encode the AIC/AIV/MIX layout inline so the key stays
                    # parser-safe while still self-documenting.
                    phase_args.update(
                        {
                            "shared_at_start (aic,aiv,mix)": list(shared_at_start),
                            "shared_at_end (aic,aiv,mix)": list(shared_at_end),
                        }
                    )
                if phase == "complete":
                    # finishes_processed kept in args (hover) for forensics —
                    # SPMD cases where one logical task has N subtask FINs
                    # observed in the phase. Label stays minimal.
                    finishes_count = finishes_per_complete.get(id(record), 0)
                    phase_args["finishes_processed"] = finishes_count
                display_name = f"{phase}({tasks_processed})"
                event_tid = resolve_tid if phase == "resolve" else tid
                events.append(
                    {
                        "args": phase_args,
                        "cat": "scheduler",
                        "cname": phase_colors.get(phase, "generic_work"),
                        "name": display_name,
                        "ph": "X",
                        "pid": 2,
                        "tid": event_tid,
                        "ts": start_us,
                        "dur": dur,
                    }
                )

                # Queue-depth counter tracks (Perfetto "ph": "C"). Emit ONE
                # sample per phase at its end_us — phase N's end is phase N+1's
                # start, so emitting both is redundant. Two samples at the
                # SAME ts (e.g. final-drain emit where start_time==end_time)
                # also breaks Perfetto's rate calc (divide-by-zero → NULL).
                # Only complete/dispatch carry real queue depths; release/
                # resolve/early_dispatch zero-fill them, so skip their counter
                # samples to avoid spurious 0 dips.
                if phase not in ("complete", "dispatch"):
                    continue
                if not depths_valid:
                    continue
                # Shared queue: dedicated tid 3999 so all 3 schedulers'
                # snapshots compose onto one timeline (it's the same global
                # queue regardless of who sampled it). Samples from different
                # threads at slightly different ts are fine — Perfetto plots
                # them in time order to render the step function.
                events.append(
                    {
                        "args": {"AIC": shared_at_end[0], "AIV": shared_at_end[1], "MIX": shared_at_end[2]},
                        "cat": "queue",
                        "name": "shared_ready_queue",
                        "ph": "C",
                        "pid": 2,
                        "tid": 3999,
                        "ts": end_us,
                    }
                )

        # Name the shared-queue pseudo-thread + give it a sort index that
        # places it after the 3 scheduler threads but still inside the
        # AICPU Scheduler process row, so the user reads top-to-bottom:
        # Sched_0 / Sched_1 / Sched_2 / Shared queue (global).
        events.append(
            {
                "args": {"name": "shared_ready_queue (global)"},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": 2,
                "tid": 3999,
            }
        )
        events.append(
            {
                "args": {"sort_index": 100},
                "cat": "__metadata",
                "name": "thread_sort_index",
                "ph": "M",
                "pid": 2,
                "tid": 3999,
            }
        )

    # AICPU Orchestrator lane (l2_swimlane_level >= 4)
    #
    # Per-event AicpuPhaseRecord[] is the single source of truth for
    # orchestrator timing. There is no separate aggregate summary — the
    # device-side LOG_INFO_V9 "orch_start=… orch_end=… orch_cost=…" log
    # line covers the run-window envelope for debugging without swimlane.
    if orchestrator_phases:
        # Process metadata
        orch_process_label = f"AICPU {orchestrator_name}" if orchestrator_name else "AICPU Orchestrator"
        events.append(
            {"args": {"name": orch_process_label}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 1}
        )
        events.append(
            {"args": {"sort_index": 1}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 1}
        )

        # Thread name metadata for each orchestrator thread
        for orch_idx in range(len(orchestrator_phases)):
            tid = 4000 + orch_idx
            name = f"Orch_{orch_idx}"
            events.append(
                {"args": {"name": name}, "cat": "__metadata", "name": "thread_name", "ph": "M", "pid": 1, "tid": tid}
            )

        # Per-task orchestrator phase bars. As of PR-X the device folds
        # all 6 sub-step phases into one ORCH_SUBMIT record covering the
        # submit's entire [start, end] window. Legacy per-sub-step phase
        # strings remain in the color map so old captures still render.
        orch_phase_colors = {
            "orch_submit": "rail_animation",  # purple — primary
            # Legacy per-sub-step phases (old captures only):
            "orch_sync": "thread_state_iowait",
            "orch_alloc": "terrible",
            "orch_params": "good",
            "orch_lookup": "thread_state_running",
            "orch_insert": "olive",
            "orch_fanin": "rail_animation",
        }

        # Build the regular-task and dummy-task id sets so the orch-phase loop
        # below can identify which orch_submit envelopes belong to
        # `alloc_tensors()` calls (no AICore record, no DummyTask phase — purely
        # inline-completed by the orchestrator on host CPU). Those get a
        # parallel ALLOC bar on Worker View pid=4 so the DAG node is visible.
        regular_task_ids = {int(t.get("task_id", -1)) for t in tasks}
        dummy_low32 = set()
        if scheduler_phases:
            for thread_records in scheduler_phases:
                for rec in thread_records:
                    if rec.get("phase") == "dummy_task":
                        dummy_low32.add(rec.get("tasks_processed", 0))

        # Alloc bars land on the same AICPU_N lane that hosts the orchestrator
        # (one AICPU lane per AICPU; the orch is on AICPU 0 in the single-orch
        # case). The AICPU lane metadata is emitted by the scheduler_phases
        # block above (via AICPU_TID_BASE). If there are orch records but no
        # sched records (level=4 with no SCHED_PHASES data), emit them here so
        # the alloc bars below have valid lane metadata.
        AICPU_TID_BASE = 19000  # noqa: N806
        if not scheduler_phases:
            for orch_idx in range(len(orchestrator_phases)):
                events.append(
                    {
                        "args": {"name": f"AICPU_{orch_idx}"},
                        "cat": "__metadata",
                        "name": "thread_name",
                        "ph": "M",
                        "pid": 4,
                        "tid": AICPU_TID_BASE + orch_idx,
                    }
                )

        for orch_idx, thread_records in enumerate(orchestrator_phases):
            tid = 4000 + orch_idx
            for record in thread_records:
                phase = record.get("phase", "unknown")
                start_us = record["start_time_us"]
                end_us = record["end_time_us"]
                dur = end_us - start_us
                submit_idx = record.get("submit_idx", 0)
                task_id = record.get("task_id", -1)

                # Strip "orch_" prefix for display name
                display_name = phase.replace("orch_", "") if phase.startswith("orch_") else phase

                # Full PTO2TaskId in JSON (device uses task_id.raw, same as TensorMap) → rXtY / tY
                if task_id >= 0:
                    label = f"{display_name}({format_task_display(task_id)})"
                else:
                    label = f"{display_name}({submit_idx})"

                event = {
                    "args": {"phase": phase, "submit_idx": submit_idx, "task_id": task_id},
                    "cat": "orchestrator",
                    "cname": orch_phase_colors.get(phase, "generic_work"),
                    "name": label,
                    "ph": "X",
                    "pid": 1,
                    "tid": tid,
                    "ts": start_us,
                    "dur": dur,
                }
                events.append(event)

                # Classification: orch_submit envelopes whose task_id is NOT in
                # the AICore record set AND NOT in DummyTask phases are
                # alloc_tensors() calls. Render a parallel "alloc(...)" bar on
                # the Worker View AICPU_{orch_idx} lane so the DAG node is
                # visible (matches the dummy treatment — both are AICPU acting
                # as worker).
                if phase == "orch_submit" and task_id >= 0:
                    is_regular = task_id in regular_task_ids
                    task_low32 = task_id & 0xFFFFFFFF
                    is_dummy = task_low32 in dummy_low32
                    if not is_regular and not is_dummy:
                        events.append(
                            {
                                "args": {
                                    "phase": "alloc",
                                    "task_id": task_id,
                                    "event-hint": f"alloc({format_task_display(task_id)})",
                                },
                                "cat": "event",
                                "cname": "olive",
                                "name": f"alloc({format_task_display(task_id)})",
                                "ph": "X",
                                "pid": 4,
                                "tid": AICPU_TID_BASE + orch_idx,
                                "ts": start_us,
                                "dur": max(dur, 0.02),
                            }
                        )

    # Scheduler View dependency mirror (AICPU timestamps).
    if has_aicpu_data:
        for pred_id, succ_ids in edges_by_pred.items():
            if pred_id not in task_map:
                continue

            for succ_id in succ_ids:
                if succ_id not in task_map:
                    continue

                row_pairs = _dependency_flow_row_pairs(pred_id, succ_id, task_map, spmd_task_ids)
                if not row_pairs:
                    continue

                output_task_count = _dependency_task_fan_count(pred_id, spmd_task_ids, task_map, deps_block_map)
                input_task_count = _dependency_task_fan_count(succ_id, spmd_task_ids, task_map, deps_block_map)
                for pred_row, succ_row in row_pairs:
                    src_finish_us = pred_row.get("finish_time_us", 0)
                    dst_dispatch_us = succ_row.get("dispatch_time_us", 0)
                    dst_finish_us = succ_row.get("finish_time_us", 0)
                    # Skip when AICPU timestamps are missing or zero (matches Scheduler
                    # View bar emission, which rejects finish_us <= 0).
                    if src_finish_us <= 0 or dst_dispatch_us < 0 or dst_finish_us <= 0:
                        continue
                    src_ts = src_finish_us - flow_epsilon
                    aicpu_hb_violated = (src_ts + flow_epsilon) > dst_dispatch_us
                    aicpu_flow_name = "hb_violation" if aicpu_hb_violated else "dependency"
                    _append_dependency_flow_pair(
                        events,
                        flow_id,
                        aicpu_flow_name,
                        3,
                        task_to_aicpu_tid.get(
                            (pred_row["task_id"], pred_row["core_id"]), core_to_tid[pred_row["core_id"]]
                        ),
                        src_ts,
                        task_to_aicpu_event_id.get((pred_row["task_id"], pred_row["core_id"])),
                        3,
                        task_to_aicpu_tid.get(
                            (succ_row["task_id"], succ_row["core_id"]), core_to_tid[succ_row["core_id"]]
                        ),
                        dst_dispatch_us,
                        task_to_aicpu_event_id.get((succ_row["task_id"], succ_row["core_id"])),
                        input_task_count=input_task_count,
                        output_task_count=output_task_count,
                    )
                    flow_id += 1

    # Complete-phase flow arrows. The complete phase wraps the AICPU's
    # completion-polling loop: it observes AICore subtask FINs, increments
    # the slot's per-task subtask counter, and on the LAST subtask of a
    # logical task it walks the fanout list and releases each consumer's
    # fanin refcount. The arrows are built in two stages.
    #
    # Inbound: per-task, NOT per-subtask. A task is logically "completed"
    # only when its LAST subtask is observed (the one that triggers
    # phase_complete_count++ in firmware). For SPMD with N subtasks across
    # N cores, the earlier N-1 subtasks just bump the slot's
    # completed_subtasks counter inside whatever complete phase happened to
    # poll them; only the LAST subtask's finish actually completes the
    # task. So per task: take max(finish_time_us) across its subtasks and
    # find the complete phase that CONTAINS that time. The visual arrow starts
    # on the same earliest-start Worker View subtask slice used by SPMD
    # dependency arrows, then lands at the last subtask's AICPU finish
    # timestamp inside the complete phase. This keeps the related SPMD flow
    # arrows anchored on one visible record without changing completion
    # attribution semantics.
    #
    # Outbound: per-consumer, gated on full fanin. Each consumer in
    # deps.json has multiple producer fanin edges; refcount += 1 fires
    # whenever ANY producer's complete walks its fanout, but the consumer
    # only becomes ready when ALL producers have completed. The complete
    # that triggers the LAST refcount bump is the one that "released" the
    # consumer — that's the causal edge. Compute by walking complete
    # phases in temporal order and tracking each consumer's satisfied
    # fanin count.
    if scheduler_phases and core_to_thread:
        complete_phases_by_thread = []
        complete_starts_by_thread = []
        for thread_records in scheduler_phases:
            sorted_completes = sorted(
                (r for r in thread_records if r.get("phase") == "complete"),
                key=lambda r: r["start_time_us"],
            )
            complete_phases_by_thread.append(sorted_completes)
            complete_starts_by_thread.append([c["start_time_us"] for c in sorted_completes])

        # Group subtask records by task_id; SPMD tasks have multiple rows.
        tasks_by_id: dict[int, list[dict]] = defaultdict(list)
        for t in tasks:
            tasks_by_id[t["task_id"]].append(t)

        # For each task: completion = LAST subtask's finish observation.
        # The owning thread is determined by core_to_thread of that last
        # subtask's core — typical case is the same thread observed
        # earlier subtasks too, but we don't assume. The flow starts on the
        # earliest-start Worker View subtask slice and ends at the LAST
        # subtask's AICPU finish timestamp, preserving completion attribution
        # while keeping related SPMD arrows on the same record.
        task_to_complete: dict[int, dict] = {}
        task_last_subtask: dict[int, tuple[float, float, int]] = {}  # tid -> (last_end_us, last_finish_us, core_id)
        task_anchor_subtasks: dict[int, list[dict]] = {}
        for tid, recs in tasks_by_id.items():
            valid_finishes = [
                (r.get("finish_time_us"), r.get("end_time_us"), r["core_id"])
                for r in recs
                if r.get("finish_time_us") is not None and r["finish_time_us"] >= 0 and r.get("end_time_us") is not None
            ]
            anchor_rows = _dependency_flow_anchor_rows(tid, task_map, spmd_task_ids)
            if not valid_finishes or not anchor_rows:
                continue
            last_finish_us, last_end_us, last_cid = max(valid_finishes, key=lambda x: x[0])
            if last_cid >= len(core_to_thread):
                continue
            owning_thread = core_to_thread[last_cid]
            if owning_thread < 0 or owning_thread >= len(complete_phases_by_thread):
                continue
            task_last_subtask[tid] = (last_end_us, last_finish_us, last_cid)
            task_anchor_subtasks[tid] = anchor_rows
            # Find the complete phase that CONTAINS this last_finish_us.
            # Fall back to the next-starting complete if none contains
            # (rare: AICore reported the finish but the scheduler hadn't
            # entered its next complete phase by run end). Bisect for O(log N).
            chosen = None
            phases = complete_phases_by_thread[owning_thread]
            starts = complete_starts_by_thread[owning_thread]
            if phases:
                idx = bisect.bisect_right(starts, last_finish_us)
                if idx > 0:
                    prev_c = phases[idx - 1]
                    if prev_c["start_time_us"] <= last_finish_us <= prev_c["end_time_us"]:
                        chosen = prev_c
                if chosen is None and idx < len(phases):
                    chosen = phases[idx]
            if chosen is not None:
                task_to_complete[tid] = chosen

        # ---- Inbound: one arrow per anchor, mirrored on Worker View and Scheduler View ----
        # Source ts = <bar end> - epsilon so it lands INSIDE the task X event
        # selected by _dependency_flow_anchor_rows(). Without this anchoring
        # Perfetto can't bind the flow to a slice and the arrow is invisible
        # when you click the task. Same convention as the `dependency` arrows,
        # and — like the Scheduler View dependency mirror — the complete flow
        # is drawn in both task views so clicking the task in either lands the
        # arrow. The pid=2 endpoint (thread + ts) is identical for both mirrors,
        # so completion attribution is unchanged; only the source view differs.
        FLOW_EPSILON_US = 0.01
        for tid, comp in task_to_complete.items():
            _last_end_us, last_finish_us, last_cid = task_last_subtask[tid]
            owning_thread = core_to_thread[last_cid]
            dst_tid = sched_lane_tid(owning_thread, 0)
            dst_ts = comp["start_time_us"]
            if comp["start_time_us"] <= last_finish_us <= comp["end_time_us"]:
                dst_ts = last_finish_us
            for anchor in task_anchor_subtasks[tid]:
                # Worker View (pid=4): anchor on the kernel slice (end_time_us).
                src_tid = core_to_tid[anchor["core_id"]]
                src_event_id = task_to_event_id.get((tid, anchor["core_id"]))
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "complete",
                        "ph": "s",
                        "pid": 4,
                        "tid": src_tid,
                        "ts": anchor["end_time_us"] - FLOW_EPSILON_US,
                        **({"bind_id": src_event_id} if src_event_id is not None else {}),
                    }
                )
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "complete",
                        "ph": "f",
                        "pid": 2,
                        "tid": dst_tid,
                        "ts": dst_ts,
                        "bp": "e",
                    }
                )
                flow_id += 1

                # Scheduler View (pid=3): mirror on the AICPU dispatch→finish
                # bar (source ts = finish_time_us). Skip when this anchor has
                # no AICPU finish — its pid=3 bar doesn't exist to bind to.
                anchor_finish_us = anchor.get("finish_time_us")
                if anchor_finish_us is None or anchor_finish_us <= 0:
                    continue
                sched_src_tid = task_to_aicpu_tid.get((tid, anchor["core_id"]), core_to_tid[anchor["core_id"]])
                sched_src_event_id = task_to_aicpu_event_id.get((tid, anchor["core_id"]))
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "complete",
                        "ph": "s",
                        "pid": 3,
                        "tid": sched_src_tid,
                        "ts": anchor_finish_us - FLOW_EPSILON_US,
                        **({"bind_id": sched_src_event_id} if sched_src_event_id is not None else {}),
                    }
                )
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "complete",
                        "ph": "f",
                        "pid": 2,
                        "tid": dst_tid,
                        "ts": dst_ts,
                        "bp": "e",
                    }
                )
                flow_id += 1

        # ---- Outbound: per-consumer, gated on full fanin ----
        if deps_edges is not None:
            # Invert deps_edges to consumer → predecessors for fanin counting.
            preds_for_consumer: dict[int, list[int]] = defaultdict(list)
            for pred, succs in deps_edges.items():
                for succ in succs:
                    preds_for_consumer[succ].append(pred)
            fanin_total = {c: len(preds) for c, preds in preds_for_consumer.items()}
            fanin_satisfied: dict[int, int] = defaultdict(int)

            # Reverse map: complete_phase id → list of task_ids it completed.
            complete_to_tasks: dict[int, list[int]] = defaultdict(list)
            for tid, comp in task_to_complete.items():
                complete_to_tasks[id(comp)].append(tid)

            # Walk completes in temporal order (by end_time). Within each,
            # walk the tasks it completed; for each completed task, bump
            # its consumers' satisfied fanin. The complete that pushes a
            # consumer's satisfied count to its total is the one that
            # released that consumer.
            all_completes = []
            for thr_idx, phases in enumerate(complete_phases_by_thread):
                for p in phases:
                    all_completes.append((p["end_time_us"], thr_idx, p))
            # Explicit key restricts the comparison to (end_time_us, thr_idx).
            # Without it, ties in both fields fall through to comparing the
            # third element (a dict), which raises TypeError in Python 3.
            all_completes.sort(key=lambda x: (x[0], x[1]))

            # Earliest dispatch per task_id (for arrow target).
            earliest_dispatch_us: dict[int, tuple[float, int]] = {}
            for tid, recs in tasks_by_id.items():
                valid = [
                    (r.get("dispatch_time_us"), r["core_id"])
                    for r in recs
                    if r.get("dispatch_time_us") is not None and r["dispatch_time_us"] >= 0
                ]
                if not valid:
                    continue
                d_us, d_cid = min(valid, key=lambda x: x[0])
                if d_cid >= len(core_to_thread):
                    continue
                d_thr = core_to_thread[d_cid]
                if d_thr < 0:
                    continue
                earliest_dispatch_us[tid] = (d_us, d_thr)

            for end_us, comp_thr, comp in all_completes:
                completed_tids = complete_to_tasks.get(id(comp), ())
                if not completed_tids:
                    continue
                triggered: list[int] = []
                for completed_tid in completed_tids:
                    for consumer in deps_edges.get(completed_tid, ()):
                        fanin_satisfied[consumer] += 1
                        if fanin_satisfied[consumer] == fanin_total.get(consumer, 0):
                            triggered.append(consumer)
                if not triggered:
                    continue
                src_tid = sched_lane_tid(comp_thr, 0)
                for consumer in triggered:
                    if consumer not in earliest_dispatch_us:
                        continue
                    d_us, d_thr = earliest_dispatch_us[consumer]
                    # Skip degenerate "dispatched before complete ended" —
                    # the consumer was popped/dispatched off a still-in-flight
                    # release path while the complete was still running;
                    # the arrow would point backwards.
                    if d_us < end_us:
                        continue
                    events.append(
                        {
                            "cat": "flow",
                            "id": flow_id,
                            "name": "complete→ready",
                            "ph": "s",
                            "pid": 2,
                            "tid": src_tid,
                            # Anchor inside the complete phase X event so
                            # clicking the complete block surfaces this arrow.
                            "ts": end_us - FLOW_EPSILON_US,
                        }
                    )
                    events.append(
                        {
                            "cat": "flow",
                            "id": flow_id,
                            "name": "complete→ready",
                            "ph": "f",
                            "pid": 2,
                            "tid": sched_lane_tid(d_thr, 0),
                            "ts": d_us,
                            "bp": "e",
                        }
                    )
                    flow_id += 1

    # Scheduler DISPATCH → task execution arrows
    if scheduler_phases and has_aicpu_data:
        # Build core_id → scheduler thread mapping.
        # Prefer explicit core_to_thread from perf JSON (written by AICPU after orchestration).
        # Fall back to voting heuristic for older data without the mapping.
        core_to_sched_thread = {}

        if core_to_thread:
            for core_id, thread_idx in enumerate(core_to_thread):
                if thread_idx >= 0:
                    core_to_sched_thread[core_id] = thread_idx
            if verbose:
                print(f"  Core-to-thread mapping: {len(core_to_sched_thread)} cores (from perf JSON)")
        else:
            # Fallback: infer via voting (for perf JSON without core_to_thread field)
            dispatch_phases_by_thread = {}
            for thread_idx, thread_records in enumerate(scheduler_phases):
                dispatch_records = [r for r in thread_records if r.get("phase") == "dispatch"]
                if dispatch_records:
                    dispatch_phases_by_thread[thread_idx] = dispatch_records

            core_thread_votes = defaultdict(lambda: defaultdict(int))
            for task in tasks:
                dispatch_us = task.get("dispatch_time_us", 0)
                if dispatch_us < 0:
                    continue
                core_id = task["core_id"]
                for thread_idx, dispatch_records in dispatch_phases_by_thread.items():
                    for dr in dispatch_records:
                        if dr["start_time_us"] <= dispatch_us <= dr["end_time_us"]:
                            core_thread_votes[core_id][thread_idx] += 1
                            break

            for core_id, votes in core_thread_votes.items():
                core_to_sched_thread[core_id] = max(votes.items(), key=lambda kv: kv[1])[0]
            if verbose:
                print(f"  Core-to-thread mapping: {len(core_to_sched_thread)} cores (inferred via voting)")

        for task in tasks:
            dispatch_us = task.get("dispatch_time_us", 0)
            if dispatch_us < 0:
                continue

            matched_thread = core_to_sched_thread.get(task["core_id"])

            if matched_thread is not None:
                sched_tid = sched_lane_tid(matched_thread, 0)
                core_tid = core_to_tid[task["core_id"]]
                aicpu_tid = task_to_aicpu_tid.get((task["task_id"], task["core_id"]), core_tid)

                # Flow: scheduler DISPATCH → Worker View task start
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dispatch",
                        "ph": "s",
                        "pid": 2,
                        "tid": sched_tid,
                        "ts": dispatch_us,
                    }
                )
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dispatch",
                        "ph": "f",
                        "pid": 4,
                        "tid": core_tid,
                        "ts": task["start_time_us"],
                        "bp": "e",
                    }
                )
                flow_id += 1

                # Flow: scheduler DISPATCH → Scheduler View task start
                aicpu_eid = task_to_aicpu_event_id.get((task["task_id"], task["core_id"]))
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dispatch",
                        "ph": "s",
                        "pid": 2,
                        "tid": sched_tid,
                        "ts": dispatch_us,
                    }
                )
                flow_f = {
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dispatch",
                    "ph": "f",
                    "pid": 3,
                    "tid": aicpu_tid,
                    "ts": dispatch_us,
                    "bp": "e",
                }
                if aicpu_eid is not None:
                    flow_f["bind_id"] = aicpu_eid
                events.append(flow_f)
                flow_id += 1

    # Orchestrator → scheduler dispatch:
    # Anchor each task's dispatch arrow on the end of its orch_submit record
    # (covers the entire submit_task() span). Legacy captures with the older
    # per-sub-step phases (orch_fanin / orch_params) are accepted as fallbacks.
    if orchestrator_phases and scheduler_phases:
        orch_anchor_by_task = {}
        for orch_idx, thread_records in enumerate(orchestrator_phases):
            for record in thread_records:
                phase = record.get("phase")
                task_id = record.get("task_id", -1)
                if task_id < 0:
                    continue
                tid_k = normalize_pto2_task_id_int(task_id)
                if tid_k is None:
                    continue
                # First-seen orch_submit wins; legacy orch_fanin / orch_params
                # only fill in when no orch_submit exists for that task. The
                # explicit "not already submit→dispatch" guard preserves first-
                # seen semantics even if a (defensive) duplicate orch_submit
                # ever appears for the same task.
                existing = orch_anchor_by_task.get(tid_k)
                if phase == "orch_submit" and (existing is None or existing[2] != "submit→dispatch"):
                    orch_anchor_by_task[tid_k] = (record, orch_idx, "submit→dispatch")
                elif existing is None and phase == "orch_fanin":
                    orch_anchor_by_task[tid_k] = (record, orch_idx, "fanin→dispatch")
                elif existing is None and phase == "orch_params":
                    orch_anchor_by_task[tid_k] = (record, orch_idx, "params→dispatch")

        if has_aicpu_data and orch_anchor_by_task:
            for task in tasks:
                tid = normalize_pto2_task_id_int(task.get("task_id"))
                if tid is None:
                    continue

                dispatch_us = task.get("dispatch_time_us", 0)
                if dispatch_us < 0:
                    continue

                matched_thread = core_to_sched_thread.get(task["core_id"])
                if matched_thread is None:
                    continue

                sched_tid = sched_lane_tid(matched_thread, 0)

                anchor = orch_anchor_by_task.get(tid)
                if anchor is None:
                    continue

                anchor_rec, orch_idx, flow_name = anchor
                anchor_us = anchor_rec["end_time_us"]

                orch_tid = 4000 + orch_idx

                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": flow_name,
                        "ph": "s",
                        "pid": 1,
                        "tid": orch_tid,
                        "ts": anchor_us,
                    }
                )
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": flow_name,
                        "ph": "f",
                        "pid": 2,
                        "tid": sched_tid,
                        "ts": dispatch_us,
                        "bp": "e",
                    }
                )
                flow_id += 1

    if verbose:
        print(f"  Total events: {len(events)}")
        print(f"  Flow events: {flow_id}")

    # Step 3: Write JSON file (with traceEvents wrapper to match C++ output)
    if emit_overhead:
        oh = build_overhead_counter_events(tasks, deps_edges)
        events.extend(oh)
        if verbose:
            print(f"  Overhead Analysis: {sum(1 for e in oh if e.get('ph') == 'C')} counter points (8 tracks)")

    with open(output_path, "w") as f:
        json.dump({"traceEvents": events}, f, indent=2)

    if verbose:
        print(f"JSON written to: {output_path}")


def _build_parser():
    parser = argparse.ArgumentParser(
        description="Convert swimlane performance JSON to Chrome Trace Event JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                            # Use latest .json in outputs/, output to outputs/
  %(prog)s outputs/<case>_<ts>/l2_swimlane_records.json   # Output: outputs/merged_swimlane_20260210_143526.json
  %(prog)s outputs/<case>_<ts>/l2_swimlane_records.json -o custom_output.json
  %(prog)s outputs/<case>_<ts>/l2_swimlane_records.json \
      -k examples/host_build_graph/paged_attention/kernels/kernel_config.py
  %(prog)s outputs/<case>_<ts>/l2_swimlane_records.json -v
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input JSON file (.json). If not specified, uses the latest l2_swimlane_records_*.json in outputs/",
    )
    parser.add_argument("-o", "--output", help="Output JSON file (default: <input_dir>/merged_swimlane.json)")
    parser.add_argument(
        "-k",
        "--kernel-config",
        help="Path to kernel_config.py file for func_id to function name mapping",
    )
    parser.add_argument(
        "--func-names",
        help="Path to func_id_names_*.json (SceneTest format) for func_id to function name mapping",
    )
    parser.add_argument(
        "--deps-json",
        help=(
            "Path to a dep_gen replay deps.json (defaults to sibling of the perf JSON). "
            "Without one the trace has no dependency arrows — re-run with --enable-dep-gen first."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    parser.add_argument(
        "--overhead",
        action="store_true",
        help="Add an 'Overhead Analysis' track (8 counter lines: per-engine "
        "idle/ready/overhead + system all_overhead/has_overhead). Needs deps.json.",
    )
    return parser


def _resolve_input_path(args):
    """Resolve input path, auto-selecting newest outputs/<case>/l2_swimlane_records.json if unspecified."""
    if args.input is not None:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            return None
        return input_path

    outputs_dir = Path.cwd() / "outputs"
    json_files = list(outputs_dir.glob("*/l2_swimlane_records.json"))
    if not json_files:
        print(f"Error: No outputs/*/l2_swimlane_records.json found under {outputs_dir}", file=sys.stderr)
        print("Run a test with --enable-l2-swimlane first, or specify an explicit input.", file=sys.stderr)
        return None

    input_path = max(json_files, key=lambda p: p.stat().st_mtime)
    if args.verbose:
        print(f"Auto-selected latest file: {input_path}")
    return input_path


def _resolve_output_path(args, input_path):
    """Determine output path from args or derive from input directory name."""
    if args.output:
        return Path(args.output)

    # Default: write merged_swimlane.json next to the input. The parent
    # directory name (e.g. outputs/<case>_<ts>/) already disambiguates runs.
    return input_path.parent / "merged_swimlane.json"


def _print_verbose_data_info(data, verbose):
    """Print verbose summary of loaded performance data, including phase counts
    when present (l2_swimlane_level >= SCHED_PHASES)."""
    if not verbose:
        return
    print("\n=== Performance Data ===")
    print(f"  L2 perf level: {data['l2_swimlane_level']}")
    print(f"  Task Count: {len(data['tasks'])}")
    if data["tasks"]:
        start_times = [t["start_time_us"] for t in data["tasks"]]
        end_times = [t["end_time_us"] for t in data["tasks"]]
        min_time = min(start_times)
        max_time = max(end_times)
        print(f"  Time Range: {min_time:.3f} us - {max_time:.3f} us (span: {max_time - min_time:.3f} us)")
    print()
    scheduler_phases = data.get("aicpu_scheduler_phases")
    orchestrator_phases = data.get("aicpu_orchestrator_phases")
    core_to_thread = data.get("core_to_thread")
    if scheduler_phases:
        print(f"  Scheduler threads: {len(scheduler_phases)}")
        print(f"  Total phase records: {sum(len(t) for t in scheduler_phases)}")
    if orchestrator_phases:
        print(f"  Orchestrator threads: {len(orchestrator_phases)}")
        print(f"  Total orchestrator phase records: {sum(len(t) for t in orchestrator_phases)}")
        # submit_count is derivable as the number of orch_submit records (one per submit).
        # Legacy captures fall back to orch_fanin (was last phase of submit pre-fold).
        submit_count = sum(1 for thread in orchestrator_phases for r in thread if r.get("phase") == "orch_submit")
        if submit_count == 0:
            submit_count = sum(1 for thread in orchestrator_phases for r in thread if r.get("phase") == "orch_fanin")
        if submit_count:
            print(f"  Orchestrator: {submit_count} tasks submitted")
    if core_to_thread:
        print(f"  Core-to-thread mapping: {len(core_to_thread)} cores")


def _load_func_names(args):
    """Load func_id→name mapping from --func-names JSON or -k kernel_config.py.

    Returns:
        tuple: (func_id_to_name dict, orchestrator_name str or None)
    """
    if args.func_names:
        if args.verbose:
            print(f"Loading func names from: {args.func_names}")
        func_names, orchestrator_name = load_func_names_json(args.func_names)
        if args.verbose:
            print(f"  Loaded {len(func_names)} function name mappings:")
            for func_id, name in sorted(func_names.items(), key=lambda x: int(x[0])):
                print(f"    func_id={func_id}: {name}")
            if orchestrator_name:
                print(f"  Orchestrator: {orchestrator_name}")
            print()
        return func_names, orchestrator_name

    if args.kernel_config:
        if args.verbose:
            print(f"Loading kernel config from: {args.kernel_config}")
        func_names = load_kernel_config(args.kernel_config)
        if args.verbose:
            print(f"  Loaded {len(func_names)} function name mappings from kernel_config.py:")
            for func_id, name in sorted(func_names.items(), key=lambda x: int(x[0])):
                print(f"    func_id={func_id}: {name}")
            print()
        return func_names, None

    return {}, None


def main():
    args = _build_parser().parse_args()

    input_path = _resolve_input_path(args)
    if input_path is None:
        return 1

    try:
        if args.verbose:
            print(f"Reading performance data from: {input_path}")
        data = read_perf_data(input_path)
        _print_verbose_data_info(data, args.verbose)

        func_names, orchestrator_name = _load_func_names(args)

        output_path = _resolve_output_path(args, input_path)

        deps_path = Path(args.deps_json) if args.deps_json else Path(input_path).parent / "deps.json"
        deps_edges = load_deps_json(deps_path)
        # Load the per-task kernel_ids map separately so the trace generator
        # can resolve func_id=-1 records (AICORE_TIMING / level=1) back to
        # the real kernel name. Optional — pre-schema deps.json without
        # kernel_ids and AICPU_TIMING+ runs both leave this at None.
        deps_kernel_map = load_deps_kernel_map(deps_path)
        deps_block_map = load_deps_block_map(deps_path)
        if deps_edges is not None:
            if args.verbose:
                print(f"  Using deps.json edges ({sum(len(v) for v in deps_edges.values())} total) from {deps_path}")
                if deps_kernel_map is not None:
                    print(f"  Using deps.json kernel_ids for {len(deps_kernel_map)} tasks (level=1 name recovery)")
        else:
            print(
                f"Warning: no usable deps.json at {deps_path}; Perfetto trace will have no dependency arrows. "
                f"Run a dep_gen capture (--enable-dep-gen) and pass --deps-json <path> to add them.",
                file=sys.stderr,
            )

        generate_chrome_trace_json(
            data["tasks"],
            str(output_path),
            func_names,
            args.verbose,
            orchestrator_name=orchestrator_name,
            scheduler_phases=data.get("aicpu_scheduler_phases"),
            orchestrator_phases=data.get("aicpu_orchestrator_phases"),
            core_to_thread=data.get("core_to_thread"),
            deps_edges=deps_edges,
            deps_kernel_map=deps_kernel_map,
            deps_block_map=deps_block_map,
            emit_overhead=args.overhead,
        )
        if args.overhead and deps_edges is None:
            print(
                "Warning: --overhead needs deps.json for task readiness; no deps found, "
                "Overhead Analysis track skipped.",
                file=sys.stderr,
            )

        print("\n✓ Conversion complete")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"\nTo visualize: Open https://ui.perfetto.dev/ and drag in {output_path}")

        print_task_statistics(data["tasks"], func_names)

        # Scheduler-overhead deep-dive is a SEPARATE manual tool now: it needs
        # the task DAG (deps.json) captured in its own --enable-dep-gen run
        # (co-running dep_gen with swimlane perturbs the timing), so it can't be
        # produced accurately inline here. Run it explicitly:
        #   python -m simpler_setup.tools.sched_overhead_analysis \
        #       --l2-swimlane-records-json <this> --deps-json <deps from dep_gen run>
        print(
            "\nScheduler-overhead deep-dive: run sched_overhead_analysis manually with a "
            "separately-captured deps.json (--enable-dep-gen)."
        )

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
