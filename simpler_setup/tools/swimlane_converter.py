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
    python -m simpler_setup.tools.swimlane_converter  # latest l2_perf_records_*.json under ./outputs/
    python -m simpler_setup.tools.swimlane_converter l2_perf_records_20260210_143526.json
    python -m simpler_setup.tools.swimlane_converter l2_perf_records_20260210_143526.json -o out.json
    python -m simpler_setup.tools.swimlane_converter l2_perf_records_20260210_143526.json -k kernel_config.py
    python -m simpler_setup.tools.swimlane_converter l2_perf_records_20260210_143526.json -v
"""

import argparse
import importlib.util
import json
import sys
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Any

from .device_log_resolver import infer_device_id_from_log_path, resolve_device_log_path
from .sched_overhead_analysis import parse_scheduler_threads
from .sched_overhead_analysis import run_analysis as run_sched_overhead_analysis


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


def read_perf_data(filepath):
    """Read performance data from JSON file.

    Args:
        filepath: Path to input JSON file

    Returns:
        dict: Parsed performance data with keys:
            - version
            - tasks (list)

    Raises:
        ValueError: If JSON format is invalid
    """
    with open(filepath) as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ["version", "tasks"]
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate version
    if data["version"] not in [1, 2]:
        raise ValueError(f"Unsupported version: {data['version']} (expected 1 or 2)")

    return data


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


def parse_sched_cpu_from_device_log(log_path, task_count):
    """Parse device log for PTO2 scheduler stats and return scheduler CPU time per task (us).

    Looks for lines like: "Thread N: completed=X tasks in Yus (Z loops, W tasks/loop)"
    Sums the 'total_us' values (one per scheduler thread, typically 3) and divides by task_count.

    Returns:
        float: scheduler_us_per_task, or None if parsing failed / file missing
    """
    path = Path(log_path)
    if not path.exists() or task_count <= 0:
        return None

    try:
        threads = parse_scheduler_threads(path)
    except Exception:
        return None

    if not threads:
        return None

    total_sched_cpu_us = sum(t.get("total_us", 0.0) for t in threads.values())
    if total_sched_cpu_us <= 0:
        return None

    total_completed = sum(t.get("completed", 0) for t in threads.values())
    if task_count > 0 and total_completed > 0 and abs(total_completed - task_count) / task_count > 0.5:
        print(
            f"Warning: device log has {total_completed} completed tasks "
            f"but perf JSON has {task_count}; skipping Sched CPU metric "
            f"(device log may be from a different run)",
            file=sys.stderr,
        )
        return None

    return {
        "us_per_task": total_sched_cpu_us / task_count,
        "num_sched_threads": len(threads),
    }


def print_task_statistics(tasks, func_id_to_name=None, sched_info=None):
    """Print task statistics grouped by func_id.

    Exec = kernel execution time (end_time_us - start_time_us) on AICore.
    Latency = AICPU view: finish_time_us - dispatch_time_us (includes head OH + Exec + tail OH).
    High Latency with low Exec means scheduler/polling overhead (tail OH = finish_ts recorded
    when the scheduler loop next sees the completed handshake; reordering the loop to process
    completed tasks first reduces this).

    Args:
        tasks: List of task dicts
        func_id_to_name: Optional dict mapping func_id to function name
        sched_info: Optional dict with 'us_per_task' (float) and 'num_sched_threads' (int),
            parsed from device log by parse_sched_cpu_from_device_log()
    """
    # Group tasks by func_id with extended metrics
    func_stats: defaultdict[Any, dict[str, Any]] = defaultdict(
        lambda: {
            "durations": [],
            "head_overheads": [],
            "tail_overheads": [],
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
    print("\n" + "=" * 110)
    print("Task Statistics by Function")
    print("  Exec = kernel time on AICore; Latency = dispatch->finish (incl. head OH + Exec + tail OH)")
    print("=" * 110)
    print(
        f"{'Func_ID':<8} {'Func_Name':<12} {'Count':>5}   {'Avg Exec(us)':>12}  "
        f"{'Avg Latency(us)':>15}  {'Exec%':>6}   {'Avg Head OH(us)':>15}  {'Avg Tail OH(us)':>15}"
    )
    print("-" * 110)

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

        # Calculate execution ratio: total_exec_time / total_latency
        exec_ratio = (stats["total_exec_time"] / stats["total_latency"] * 100) if stats["total_latency"] > 0 else 0

        print(
            f"{func_id:<8} {func_name:<12} {count:>5}   {avg_duration:>12.2f}  {avg_latency:>15.2f}  "
            f"{exec_ratio:>5.1f}%   {avg_head_overhead:>15.2f}  {avg_tail_overhead:>15.2f}"
        )

    # Print total row
    print("-" * 110)

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
        if sched_info is not None:
            sched_cpu = sched_info["us_per_task"]
            num_cores = len(set(t["core_id"] for t in tasks))
            exec_sched_ratio = (avg_exec_us / sched_cpu * 100) if sched_cpu > 0 else 0
            per_core_exec = avg_exec_us / num_cores if num_cores > 0 else 0
            per_core_ratio = (per_core_exec / sched_cpu * 100) if sched_cpu > 0 else 0
            num_threads = sched_info["num_sched_threads"]
            print(
                f"  Sched CPU (from device log): {sched_cpu:.2f} us/task  "
                f"(Exec/Sched = {exec_sched_ratio:.1f}%, PerCore/Sched = {per_core_ratio:.1f}%)"
            )
            print(
                f"  (Latency = dispatch→finish; Sched CPU = scheduler thread CPU per task; "
                f"PerCore = avg_exec/{num_cores}_cores vs sched_cpu, {num_threads} sched threads)"
            )
        else:
            print("  (Latency = dispatch→finish; Sched CPU = scheduler thread CPU per task)")

    print("=" * 110)


def generate_chrome_trace_json(  # noqa: PLR0912, PLR0915
    tasks,
    output_path,
    func_id_to_name=None,
    verbose=False,
    scheduler_phases=None,
    orchestrator_data=None,
    orchestrator_phases=None,
    core_to_thread=None,
    orchestrator_name=None,
):
    """Generate Chrome Trace Event Format JSON from task data.

    Args:
        tasks: List of task dicts with fields:
            - task_id, func_id, core_id, core_type
            - start_time_us, end_time_us, duration_us
            - fanout, fanout_count
            - dispatch_time_us (optional, AICPU dispatch timestamp)
            - finish_time_us (optional, AICPU finish timestamp)
        output_path: Path to output JSON file
        func_id_to_name: Optional dict mapping func_id to function name
        verbose: Print progress information
        scheduler_phases: Optional list of per-thread phase record lists (version 2)
        orchestrator_data: Optional dict with orchestrator summary (version 2)
        orchestrator_phases: Optional list of per-task orchestrator phase records (version 2)
        core_to_thread: Optional list mapping core_id (index) to scheduler thread index (-1 = unassigned)

    Generates processes in the trace:
        - pid=1 "AICore View": start_time_us to end_time_us (kernel execution)
        - pid=2 "AICPU View": dispatch_time_us to finish_time_us (AICPU perspective)
        - pid=3 "AICPU Scheduler": scheduler phase bars (version 2)
        - pid=4 "AICPU Orchestrator": orchestrator phase bars or summary (version 2)
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

    # Step 2: Generate JSON events
    events = []

    # Metadata event: Process names and sort order
    # Display order: Orchestrator (pid=4) → Scheduler (pid=3) → AICPU View (pid=2) → AICore View (pid=1)
    events.append({"args": {"name": "AICore View"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 1})
    events.append({"args": {"sort_index": 4}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 1})

    # Check if any task has AICPU timestamps
    has_aicpu_data = any(task.get("dispatch_time_us", 0) >= 0 and task.get("finish_time_us", 0) > 0 for task in tasks)

    if has_aicpu_data:
        events.append(
            {"args": {"name": "AICPU View"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 2}
        )
        events.append(
            {"args": {"sort_index": 3}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 2}
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
            {"args": {"name": thread_name}, "cat": "__metadata", "name": "thread_name", "ph": "M", "pid": 1, "tid": tid}
        )

    # Duration events (Complete events "X")
    # Build task_id -> event_id mapping for flow events
    task_to_event_id: dict[tuple[int, int], int] = {}
    task_to_aicpu_event_id: dict[tuple[int, int], int] = {}
    task_to_aicpu_tid: dict[tuple[int, int], int] = {}
    event_id = 0

    for task in tasks:
        tid = core_to_tid[task["core_id"]]
        ts = task["start_time_us"]
        dur = task["duration_us"]

        # Build fanout hint string (packed ids → rXtY / tY for readability)
        fanout_str = "[" + ", ".join(format_task_display(x) for x in task["fanout"]) + "]"

        # Get function name if available
        func_id = task["func_id"]
        tdisp = format_task_display(task["task_id"])
        if func_id_to_name and str(func_id) in func_id_to_name:
            func_name = func_id_to_name[str(func_id)]
            task_name = f"{func_name}({tdisp})"
        else:
            task_name = f"func_{_func_id_to_letter(func_id)}({tdisp})"

        events.append(
            {
                "args": {
                    "event-hint": f"Task:{tdisp}, FuncId:{func_id}, CoreId:{task['core_id']}",
                    "fanout-hint": fanout_str,
                    "duration-us": dur,
                    "taskId": task["task_id"],
                },
                "cat": "event",
                "id": event_id,
                "name": task_name,
                "ph": "X",
                "pid": 1,
                "tid": tid,
                "ts": ts,
                "dur": dur,
            }
        )

        # Record mapping for flow events
        task_to_event_id[(task["task_id"], task["core_id"])] = event_id
        event_id += 1

    # AICPU View duration events (dispatch_time to finish_time)
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

        # Thread name metadata for AICPU View (one entry per unique tid used)
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
                        "pid": 2,
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
                        "pid": 2,
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

            # Get function name if available
            func_id = task["func_id"]
            tdisp = format_task_display(task["task_id"])
            if func_id_to_name and str(func_id) in func_id_to_name:
                func_name = func_id_to_name[str(func_id)]
                task_name = f"{func_name}({tdisp})"
            else:
                task_name = f"func_{_func_id_to_letter(func_id)}({tdisp})"

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
                    "pid": 2,
                    "tid": tid,
                    "ts": dispatch_us,
                    "dur": aicpu_dur,
                }
            )
            task_to_aicpu_event_id[(task["task_id"], task["core_id"])] = event_id
            event_id += 1

    # Flow events (Flow events "s" and "f" for dependencies)
    task_map: dict[int, list] = defaultdict(list)
    for t in tasks:
        task_map[t["task_id"]].append(t)
    flow_id = 0

    for task in tasks:
        src_tid = core_to_tid[task["core_id"]]
        src_ts_end = task["end_time_us"]
        # Get event ID for source task
        src_event_id = task_to_event_id[(task["task_id"], task["core_id"])]
        # Flow start timestamp (at end of source task, slightly before)
        # Use a small offset (0.01 us) for visual clarity
        flow_start_us = src_ts_end - 0.01

        for succ_task_id in task["fanout"]:
            if succ_task_id not in task_map:
                if verbose:
                    print(
                        f"Warning: Task {format_task_display(task['task_id'])} (raw {task['task_id']}) "
                        f"references non-existent successor {format_task_display(succ_task_id)} (raw {succ_task_id})"
                    )
                continue

            for succ_task in task_map[succ_task_id]:
                dst_tid = core_to_tid[succ_task["core_id"]]
                dst_ts_start = succ_task["start_time_us"]
                dst_event_id = task_to_event_id[(succ_task["task_id"], succ_task["core_id"])]

                # Flow start event (at end of source task)
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dependency",
                        "ph": "s",
                        "pid": 1,
                        "tid": src_tid,
                        "ts": flow_start_us,
                        "bind_id": src_event_id,
                    }
                )
                # Flow finish event (at start of destination task)
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dependency",
                        "ph": "f",
                        "pid": 1,
                        "tid": dst_tid,
                        "ts": dst_ts_start,
                        "bp": "e",
                        "bind_id": dst_event_id,
                    }
                )
                flow_id += 1

    # AICPU Scheduler phase events (version 2)
    if scheduler_phases:
        # Process metadata
        events.append(
            {"args": {"name": "AICPU Scheduler"}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 3}
        )
        events.append(
            {"args": {"sort_index": 2}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 3}
        )

        # Phase color mapping
        phase_colors = {
            "complete": "good",  # green
            "dispatch": "terrible",  # red
            "scan": "thread_state_running",  # blue
            "idle": "yellow",  # yellow
        }

        for thread_idx, thread_records in enumerate(scheduler_phases):
            tid = 3000 + thread_idx

            # Thread name metadata
            events.append(
                {
                    "args": {"name": f"Sched_{thread_idx}"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 3,
                    "tid": tid,
                }
            )

            for record in thread_records:
                phase = record.get("phase", "unknown")
                start_us = record["start_time_us"]
                end_us = record["end_time_us"]
                dur = end_us - start_us
                tasks_processed = record.get("tasks_processed", 0)

                event = {
                    "args": {
                        "phase": phase,
                        "loop_iter": record.get("loop_iter", 0),
                        "tasks_processed": tasks_processed,
                    },
                    "cat": "scheduler",
                    "cname": phase_colors.get(phase, "generic_work"),
                    "name": f"{phase}({tasks_processed})",
                    "ph": "X",
                    "pid": 3,
                    "tid": tid,
                    "ts": start_us,
                    "dur": dur,
                }
                events.append(event)

    # AICPU Orchestrator event (version 2)
    if orchestrator_phases or orchestrator_data:
        # Process metadata
        orch_process_label = f"AICPU {orchestrator_name}" if orchestrator_name else "AICPU Orchestrator"
        events.append(
            {"args": {"name": orch_process_label}, "cat": "__metadata", "name": "process_name", "ph": "M", "pid": 4}
        )
        events.append(
            {"args": {"sort_index": 1}, "cat": "__metadata", "name": "process_sort_index", "ph": "M", "pid": 4}
        )

        # Normalize orchestrator_phases: support both per-thread nested format
        # (list of lists) and legacy flat format (list of dicts)
        orch_threads = orchestrator_phases if orchestrator_phases else []

        # Thread name metadata for each orchestrator thread
        for orch_idx in range(len(orch_threads)):
            tid = 4000 + orch_idx
            name = f"Orch_{orch_idx}"
            events.append(
                {"args": {"name": name}, "cat": "__metadata", "name": "thread_name", "ph": "M", "pid": 4, "tid": tid}
            )
        if not orch_threads and orchestrator_data:
            events.append(
                {
                    "args": {"name": orchestrator_name or "Orchestrator"},
                    "cat": "__metadata",
                    "name": "thread_name",
                    "ph": "M",
                    "pid": 4,
                    "tid": 4000,
                }
            )

    if orchestrator_phases:
        # Per-task orchestrator phase bars
        orch_phase_colors = {
            "orch_sync": "thread_state_iowait",  # orange
            "orch_alloc": "terrible",  # red
            "orch_params": "good",  # green
            "orch_lookup": "thread_state_running",  # blue
            "orch_heap": "yellow",
            "orch_insert": "olive",
            "orch_fanin": "rail_animation",
            "orch_finalize": "cq_build_passed",
            "orch_scope_end": "generic_work",
        }

        for orch_idx, thread_records in enumerate(orch_threads):
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
                    "pid": 4,
                    "tid": tid,
                    "ts": start_us,
                    "dur": dur,
                }
                events.append(event)

    elif orchestrator_data:
        # Fallback: cumulative summary as single bar
        orch_start = orchestrator_data["start_time_us"]
        orch_end = orchestrator_data["end_time_us"]
        orch_dur = orch_end - orch_start
        phase_us = orchestrator_data.get("phase_us", {})

        # Build args with phase breakdown (cumulative totals, shown in detail panel)
        orch_args = {
            "submit_count": orchestrator_data.get("submit_count", 0),
        }
        total_phase_us = sum(phase_us.values())
        if total_phase_us > 0:
            for phase_name, dur in phase_us.items():
                if dur > 0:
                    pct = dur / total_phase_us * 100
                    orch_args[f"{phase_name}_us"] = round(dur, 3)
                    orch_args[f"{phase_name}_%"] = round(pct, 1)

        # Total orchestrator bar
        events.append(
            {
                "args": orch_args,
                "cat": "orchestrator",
                "name": f"Orchestrator({orchestrator_data.get('submit_count', 0)} tasks)",
                "ph": "X",
                "pid": 4,
                "tid": 4000,
                "ts": orch_start,
                "dur": orch_dur,
            }
        )

    # AICPU View fanout arrows (duplicate AICore View flow events using AICPU timestamps)
    if has_aicpu_data:
        for task in tasks:
            src_finish_us = task.get("finish_time_us", 0)
            # 0us is valid for the first task; keep it for dependency visualization.
            if src_finish_us < 0:
                continue

            src_tid = task_to_aicpu_tid.get((task["task_id"], task["core_id"]), core_to_tid[task["core_id"]])
            src_aicpu_eid = task_to_aicpu_event_id.get((task["task_id"], task["core_id"]))

            for succ_task_id in task["fanout"]:
                if succ_task_id not in task_map:
                    continue

                for succ_task in task_map[succ_task_id]:
                    dst_dispatch_us = succ_task.get("dispatch_time_us", 0)
                    if dst_dispatch_us < 0:
                        continue

                    dst_tid = task_to_aicpu_tid.get(
                        (succ_task["task_id"], succ_task["core_id"]), core_to_tid[succ_task["core_id"]]
                    )
                    dst_aicpu_eid = task_to_aicpu_event_id.get((succ_task["task_id"], succ_task["core_id"]))

                    flow_s = {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dependency",
                        "ph": "s",
                        "pid": 2,
                        "tid": src_tid,
                        "ts": src_finish_us - 0.01,
                    }
                    if src_aicpu_eid is not None:
                        flow_s["bind_id"] = src_aicpu_eid
                    events.append(flow_s)

                    flow_f = {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dependency",
                        "ph": "f",
                        "pid": 2,
                        "tid": dst_tid,
                        "ts": dst_dispatch_us,
                        "bp": "e",
                    }
                    if dst_aicpu_eid is not None:
                        flow_f["bind_id"] = dst_aicpu_eid
                    events.append(flow_f)

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
                sched_tid = 3000 + matched_thread
                core_tid = core_to_tid[task["core_id"]]
                aicpu_tid = task_to_aicpu_tid.get((task["task_id"], task["core_id"]), core_tid)

                # Flow: scheduler DISPATCH → AICore View task start
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dispatch",
                        "ph": "s",
                        "pid": 3,
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
                        "pid": 1,
                        "tid": core_tid,
                        "ts": task["start_time_us"],
                        "bp": "e",
                    }
                )
                flow_id += 1

                # Flow: scheduler DISPATCH → AICPU View task start
                aicpu_eid = task_to_aicpu_event_id.get((task["task_id"], task["core_id"]))
                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": "dispatch",
                        "ph": "s",
                        "pid": 3,
                        "tid": sched_tid,
                        "ts": dispatch_us,
                    }
                )
                flow_f = {
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dispatch",
                    "ph": "f",
                    "pid": 2,
                    "tid": aicpu_tid,
                    "ts": dispatch_us,
                    "bp": "e",
                }
                if aicpu_eid is not None:
                    flow_f["bind_id"] = aicpu_eid
                events.append(flow_f)
                flow_id += 1

    # Orchestrator → scheduler dispatch:
    # - Prefer orch_fanin end → dispatch (explicit deps / fanin path).
    # - If no orch_fanin for this task (e.g. aicpu_build_graph without fanin records), use orch_params end → dispatch.
    if orchestrator_phases and scheduler_phases:
        orch_fanin_by_task = {}
        orch_params_by_task = {}
        for orch_idx, thread_records in enumerate(orch_threads):
            for record in thread_records:
                phase = record.get("phase")
                task_id = record.get("task_id", -1)
                if task_id < 0:
                    continue
                tid_k = normalize_pto2_task_id_int(task_id)
                if tid_k is None:
                    continue
                if phase == "orch_fanin":
                    orch_fanin_by_task[tid_k] = (record, orch_idx)
                elif phase == "orch_params" and tid_k not in orch_params_by_task:
                    orch_params_by_task[tid_k] = (record, orch_idx)

        if has_aicpu_data and (orch_fanin_by_task or orch_params_by_task):
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

                sched_tid = 3000 + matched_thread

                row_pair = orch_fanin_by_task.get(tid)
                flow_name = "fanin→dispatch"
                if row_pair is None:
                    row_pair = orch_params_by_task.get(tid)
                    flow_name = "params→dispatch"

                if row_pair is None:
                    continue

                anchor_rec, orch_idx = row_pair
                anchor_us = anchor_rec["end_time_us"]

                orch_tid = 4000 + orch_idx

                events.append(
                    {
                        "cat": "flow",
                        "id": flow_id,
                        "name": flow_name,
                        "ph": "s",
                        "pid": 4,
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
                        "pid": 3,
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
  %(prog)s                                       # Use latest .json in outputs/, output to outputs/
  %(prog)s l2_perf_records_20260210_143526.json   # Output: outputs/merged_swimlane_20260210_143526.json
  %(prog)s l2_perf_records_20260210_143526.json -o custom_output.json
  %(prog)s l2_perf_records_20260210_143526.json -k examples/host_build_graph/paged_attention/kernels/kernel_config.py
  %(prog)s l2_perf_records_20260210_143526.json -d 0
  %(prog)s l2_perf_records_20260210_143526.json -v
        """,
    )
    parser.add_argument(
        "input",
        nargs="?",
        help="Input JSON file (.json). If not specified, uses the latest l2_perf_records_*.json in outputs/",
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
    parser.add_argument("--device-log", help="Device log file/path/glob override used for scheduler analysis")
    parser.add_argument("-d", "--device-id", help="Device id for auto-selection from device-<id>")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    return parser


def _resolve_input_path(args):
    """Resolve input path, auto-selecting newest outputs/<case>/l2_perf_records.json if unspecified."""
    if args.input is not None:
        input_path = Path(args.input)
        if not input_path.exists():
            print(f"Error: Input file not found: {input_path}", file=sys.stderr)
            return None
        return input_path

    outputs_dir = Path.cwd() / "outputs"
    json_files = list(outputs_dir.glob("*/l2_perf_records.json"))
    if not json_files:
        print(f"Error: No outputs/*/l2_perf_records.json found under {outputs_dir}", file=sys.stderr)
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
    """Print verbose summary of loaded performance data including v2 phase counts."""
    if not verbose:
        return
    print("\n=== Performance Data ===")
    print(f"  Version: {data['version']}")
    print(f"  Task Count: {len(data['tasks'])}")
    if data["tasks"]:
        start_times = [t["start_time_us"] for t in data["tasks"]]
        end_times = [t["end_time_us"] for t in data["tasks"]]
        min_time = min(start_times)
        max_time = max(end_times)
        print(f"  Time Range: {min_time:.3f} us - {max_time:.3f} us (span: {max_time - min_time:.3f} us)")
    print()
    if data["version"] != 2:
        return
    scheduler_phases = data.get("aicpu_scheduler_phases")
    orchestrator_data = data.get("aicpu_orchestrator")
    orchestrator_phases = data.get("aicpu_orchestrator_phases")
    core_to_thread = data.get("core_to_thread")
    if scheduler_phases:
        print(f"  Scheduler threads: {len(scheduler_phases)}")
        print(f"  Total phase records: {sum(len(t) for t in scheduler_phases)}")
    if orchestrator_data:
        print(f"  Orchestrator: {orchestrator_data.get('submit_count', 0)} tasks")
    if orchestrator_phases:
        print(f"  Orchestrator threads: {len(orchestrator_phases)}")
        print(f"  Total orchestrator phase records: {sum(len(t) for t in orchestrator_phases)}")
    if core_to_thread:
        print(f"  Core-to-thread mapping: {len(core_to_thread)} cores")


def _report_device_log(resolved_device_log, log_strategy):
    """Print device log resolution result."""
    if resolved_device_log is not None:
        print(f"\nDevice log: {resolved_device_log}")
        print(f"Selection: {log_strategy}")
        inferred_device_id = infer_device_id_from_log_path(resolved_device_log)
        if inferred_device_id is not None:
            print(f"Inferred Device ID: {inferred_device_id}")
    else:
        print("\nDevice log: (not resolved)")
        print(f"Selection: {log_strategy}")


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

        resolved_device_log, log_strategy = resolve_device_log_path(
            device_id=args.device_id,
            device_log=args.device_log,
            l2_perf_records_path=input_path,
        )

        generate_chrome_trace_json(
            data["tasks"],
            str(output_path),
            func_names,
            args.verbose,
            orchestrator_name=orchestrator_name,
            scheduler_phases=data.get("aicpu_scheduler_phases"),
            orchestrator_data=data.get("aicpu_orchestrator"),
            orchestrator_phases=data.get("aicpu_orchestrator_phases"),
            core_to_thread=data.get("core_to_thread"),
        )

        print("\n✓ Conversion complete")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"\nTo visualize: Open https://ui.perfetto.dev/ and drag in {output_path}")

        _report_device_log(resolved_device_log, log_strategy)

        sched_info = None
        if resolved_device_log is not None:
            sched_info = parse_sched_cpu_from_device_log(resolved_device_log, len(data["tasks"]))
            if args.verbose and sched_info is not None:
                print(f"  Parsed sched CPU from device log: {sched_info['us_per_task']:.2f} us/task")

        print_task_statistics(data["tasks"], func_names, sched_info=sched_info)

        if resolved_device_log is not None:
            print("\n=== Scheduler Overhead Deep Dive ===")
            deep_dive_rc = run_sched_overhead_analysis(
                input_path,
                resolved_device_log,
                print_sources=True,
                selection_strategy=log_strategy,
            )
            if deep_dive_rc != 0:
                print(
                    "Warning: Scheduler overhead deep-dive failed; conversion output is still available. "
                    "Check the detailed error above for root cause and fix route "
                    "(typically missing dispatch_time_us/finish_time_us in perf JSON).",
                    file=sys.stderr,
                )
        else:
            print("\n[Info] Scheduler overhead deep-dive skipped (no device log resolved).")

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
