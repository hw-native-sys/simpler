#!/usr/bin/env python3
"""
Swimlane JSON to Perfetto JSON Converter

Converts performance data JSON (.json) to Chrome Trace Event Format JSON
for visualization in Perfetto (https://ui.perfetto.dev/).

Usage:
    python3 swimlane_converter.py  # Uses latest perf_swimlane_*.json in outputs/
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json -o custom_output.json
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json -k kernel_config.py
    python3 swimlane_converter.py perf_swimlane_20260210_143526.json -v
"""

import json
import argparse
import sys
from pathlib import Path
from datetime import datetime
import importlib.util

try:
    from device_log_resolver import infer_device_id_from_log_path, resolve_device_log_path
except ImportError:
    from tools.device_log_resolver import infer_device_id_from_log_path, resolve_device_log_path

try:
    from sched_overhead_analysis import parse_scheduler_threads, run_analysis as run_sched_overhead_analysis
except ImportError:
    from tools.sched_overhead_analysis import parse_scheduler_threads, run_analysis as run_sched_overhead_analysis


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
    with open(filepath, 'r') as f:
        data = json.load(f)

    # Validate required fields
    required_fields = ['version', 'tasks']
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")

    # Validate version
    if data['version'] not in [1, 2]:
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

    # Load the Python module dynamically
    spec = importlib.util.spec_from_file_location("kernel_config", config_path)
    if spec is None or spec.loader is None:
        raise ValueError(f"Cannot load module from: {config_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    # Extract func_id to name mapping from KERNELS list
    if not hasattr(module, 'KERNELS'):
        raise ValueError(f"kernel_config.py missing KERNELS definition")

    func_id_to_name = {}
    for kernel in module.KERNELS:
        # Skip entries without func_id
        if 'func_id' not in kernel:
            print(f"Warning: Kernel entry missing 'func_id', skipping: {kernel}", file=sys.stderr)
            continue

        func_id = kernel['func_id']

        # If name is missing, we'll fall back to default naming (Func_{func_id})
        if 'name' not in kernel:
            print(f"Warning: Kernel entry for func_id={func_id} missing 'name', will use default naming", file=sys.stderr)
            continue

        # Store as string to match JSON format
        func_id_to_name[str(func_id)] = kernel['name']

    return func_id_to_name


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

    total_sched_cpu_us = sum(t.get('total_us', 0.0) for t in threads.values())
    if total_sched_cpu_us <= 0:
        return None

    return total_sched_cpu_us / task_count


def print_task_statistics(tasks, func_id_to_name=None, sched_cpu_us_per_task=None):
    """Print task statistics grouped by func_id.

    Exec = kernel execution time (end_time_us - start_time_us) on AICore.
    Latency = AICPU view: finish_time_us - dispatch_time_us (includes head OH + Exec + tail OH).
    High Latency with low Exec means scheduler/polling overhead (tail OH = finish_ts recorded
    when the scheduler loop next sees the completed handshake; reordering the loop to process
    completed tasks first reduces this).

    Args:
        tasks: List of task dicts
        func_id_to_name: Optional dict mapping func_id to function name
        sched_cpu_us_per_task: Optional AICPU scheduler CPU time per task (us),
            parsed from device log
    """
    from collections import defaultdict

    # Group tasks by func_id with extended metrics
    func_stats = defaultdict(lambda: {
        'durations': [],
        'head_overheads': [],
        'tail_overheads': [],
        'latencies': [],
        'total_exec_time': 0.0,
        'total_latency': 0.0
    })

    # Track global min dispatch and max finish times
    min_dispatch_time = float('inf')
    max_finish_time = float('-inf')

    for task in tasks:
        func_id = task['func_id']
        duration = task['duration_us']
        func_stats[func_id]['durations'].append(duration)

        # Calculate new metrics if dispatch_time_us and finish_time_us are available
        if 'dispatch_time_us' in task and 'finish_time_us' in task:
            dispatch_time = task['dispatch_time_us']
            finish_time = task['finish_time_us']
            start_time = task['start_time_us']
            end_time = task['end_time_us']

            # Head overhead: start_time_us - dispatch_time_us
            head_overhead = start_time - dispatch_time
            func_stats[func_id]['head_overheads'].append(head_overhead)

            # Tail overhead: finish_time_us - end_time_us
            tail_overhead = finish_time - end_time
            func_stats[func_id]['tail_overheads'].append(tail_overhead)

            # Latency: finish_time_us - dispatch_time_us
            latency = finish_time - dispatch_time
            func_stats[func_id]['latencies'].append(latency)

            # Accumulate execution time and latency for ratio calculation
            func_stats[func_id]['total_exec_time'] += duration
            func_stats[func_id]['total_latency'] += latency

            # Track global times
            min_dispatch_time = min(min_dispatch_time, dispatch_time)
            max_finish_time = max(max_finish_time, finish_time)

    # Print statistics
    print("\n" + "=" * 160)
    print("Task Statistics by Function")
    print("  Exec = kernel time on AICore; Latency = dispatch->finish (incl. head OH + Exec + tail OH)")
    print("=" * 160)
    print(f"{'Func_ID':<8} {'Func_Name':<12} {'Count':^6} {'Total_Exec/Latency(us)':^25} {'Avg_Exec/Latency(us)':^23} "
          f"{'Min_Exec/Latency(us)':^23} {'Max_Exec/Latency(us)':^23} {'Avg_Head/Tail_OH(us)':^23} {'Exec_%':^8}")
    print("-" * 160)

    # Sort by func_id for consistent output
    total_count = 0
    total_duration = 0.0

    for func_id in sorted(func_stats.keys()):
        stats = func_stats[func_id]
        durations = stats['durations']
        count = len(durations)
        sum_duration = sum(durations)
        avg_duration = sum_duration / count
        min_duration = min(durations)
        max_duration = max(durations)

        # Accumulate totals
        total_count += count
        total_duration += sum_duration

        # Get function name
        if func_id_to_name and str(func_id) in func_id_to_name:
            func_name = func_id_to_name[str(func_id)]
        else:
            func_name = f"Func_{func_id}"

        # Calculate averages for new metrics
        avg_head_overhead = sum(stats['head_overheads']) / len(stats['head_overheads']) if stats['head_overheads'] else 0
        avg_tail_overhead = sum(stats['tail_overheads']) / len(stats['tail_overheads']) if stats['tail_overheads'] else 0
        avg_latency = stats['total_latency'] / count if count > 0 else 0
        min_latency = min(stats['latencies']) if stats['latencies'] else 0
        max_latency = max(stats['latencies']) if stats['latencies'] else 0
        total_latency = stats['total_latency']

        # Calculate execution ratio: total_exec_time / total_latency
        exec_ratio = (stats['total_exec_time'] / stats['total_latency'] * 100) if stats['total_latency'] > 0 else 0

        # Format combined exec/latency values
        total_combined = f"{sum_duration:.2f}/{total_latency:.2f}"
        avg_combined = f"{avg_duration:.2f}/{avg_latency:.2f}"
        min_combined = f"{min_duration:.2f}/{min_latency:.2f}"
        max_combined = f"{max_duration:.2f}/{max_latency:.2f}"
        overhead_combined = f"{avg_head_overhead:.2f}/{avg_tail_overhead:.2f}"

        print(f"{func_id:<8} {func_name:<12} {count:^6} {total_combined:^25} {avg_combined:^23} "
              f"{min_combined:^23} {max_combined:^23} {overhead_combined:^23} {exec_ratio:^7.2f}%")

    # Print total row
    print("-" * 160)

    # Calculate total latency (sum of all latencies)
    total_latency_sum = sum(stats['total_latency'] for stats in func_stats.values())
    total_combined = f"{total_duration:.2f}/{total_latency_sum:.2f}"
    print(f"{'TOTAL':<21} {total_count:^6} {total_combined:^25}")

    # Print total test execution time
    if min_dispatch_time != float('inf') and max_finish_time != float('-inf'):
        total_test_time = max_finish_time - min_dispatch_time
        print(f"\nTotal Test Time: {total_test_time:.2f} us (from earliest dispatch to latest finish)")

    # Task execution vs Scheduler overhead summary
    if total_count > 0 and total_latency_sum > 0:
        avg_exec_us = total_duration / total_count
        avg_latency_us = total_latency_sum / total_count
        exec_latency_ratio_pct = total_duration / total_latency_sum * 100
        print("\n--- Task execution vs Scheduler overhead ---")
        print(f"  Per-task (all):  Avg Exec = {avg_exec_us:.2f} us,  Avg Latency (dispatch->finish) = {avg_latency_us:.2f} us,  Exec/Latency = {exec_latency_ratio_pct:.2f}%")
        if sched_cpu_us_per_task is not None:
            exec_sched_cpu_ratio = (avg_exec_us / sched_cpu_us_per_task * 100) if sched_cpu_us_per_task > 0 else 0
            print(f"  Sched CPU (from device log): {sched_cpu_us_per_task:.2f} us/task  (Exec/Sched_CPU = {exec_sched_cpu_ratio:.2f}%)")
        print("  (Latency = dispatch to finish; Sched CPU = AICPU scheduler thread CPU time per task, from device log.)")

    print("=" * 160)




def generate_chrome_trace_json(tasks, output_path, func_id_to_name=None, verbose=False,
                                scheduler_phases=None, orchestrator_data=None,
                                orchestrator_phases=None):
    """Generate Chrome Trace Event Format JSON from task data.

    Args:
        tasks: List of task dicts with fields:
            - task_id, func_id, core_id, core_type
            - start_time_us, end_time_us, duration_us
            - kernel_ready_time_us, fanout, fanout_count
            - dispatch_time_us (optional, AICPU dispatch timestamp)
            - finish_time_us (optional, AICPU finish timestamp)
        output_path: Path to output JSON file
        func_id_to_name: Optional dict mapping func_id to function name
        verbose: Print progress information
        scheduler_phases: Optional list of per-thread phase record lists (version 2)
        orchestrator_data: Optional dict with orchestrator summary (version 2)
        orchestrator_phases: Optional list of per-task orchestrator phase records (version 2)

    Generates processes in the trace:
        - pid=1 "AICore View": start_time_us to end_time_us (kernel execution)
        - pid=2 "AICPU View": dispatch_time_us to finish_time_us (AICPU perspective)
        - pid=3 "AICPU Scheduler": scheduler phase bars (version 2)
        - pid=4 "AICPU Orchestrator": orchestrator phase bars or summary (version 2)
    """
    if verbose:
        print(f"Generating Chrome Trace JSON...")
        print(f"  Tasks: {len(tasks)}")
        if func_id_to_name:
            print(f"  Function names: {len(func_id_to_name)} entries")

    # Step 1: Build core_to_tid mapping (using only core_id, not core_type)
    unique_cores = set()
    for task in tasks:
        unique_cores.add(task['core_id'])

    core_to_tid = {}
    tid_counter = 1000
    for core_id in sorted(unique_cores):
        core_to_tid[core_id] = tid_counter
        tid_counter += 1

    if verbose:
        print(f"  Unique cores: {len(unique_cores)}")

    # Step 2: Generate JSON events
    events = []

    # Metadata event: Process names and sort order
    # Display order: Orchestrator (pid=4) → Scheduler (pid=3) → AICPU View (pid=2) → AICore View (pid=1)
    events.append({
        "args": {"name": "AICore View"},
        "cat": "__metadata",
        "name": "process_name",
        "ph": "M",
        "pid": 1
    })
    events.append({
        "args": {"sort_index": 4},
        "cat": "__metadata",
        "name": "process_sort_index",
        "ph": "M",
        "pid": 1
    })

    # Check if any task has AICPU timestamps
    has_aicpu_data = any(
        task.get('dispatch_time_us', 0) > 0 and task.get('finish_time_us', 0) > 0
        for task in tasks
    )

    if has_aicpu_data:
        events.append({
            "args": {"name": "AICPU View"},
            "cat": "__metadata",
            "name": "process_name",
            "ph": "M",
            "pid": 2
        })
        events.append({
            "args": {"sort_index": 3},
            "cat": "__metadata",
            "name": "process_sort_index",
            "ph": "M",
            "pid": 2
        })

    # Metadata events: Thread names (one per core)
    for core_id, tid in core_to_tid.items():
        # Find first task with this core_id to get core_type
        core_type = None
        for task in tasks:
            if task['core_id'] == core_id:
                core_type = task['core_type']
                break

        # core_type is now a string ("aic" or "aiv")
        core_type_str = core_type.upper()
        thread_name = f"{core_type_str}_{core_id}"
        events.append({
            "args": {"name": thread_name},
            "cat": "__metadata",
            "name": "thread_name",
            "ph": "M",
            "pid": 1,
            "tid": tid
        })

        # Also add thread name for AICPU View if data exists
        if has_aicpu_data:
            events.append({
                "args": {"name": thread_name},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": 2,
                "tid": tid
            })

    # Duration events (Complete events "X")
    # Build task_id -> event_id mapping for flow events
    task_to_event_id = {}
    event_id = 0

    for task in tasks:
        tid = core_to_tid[task['core_id']]
        ts = task['start_time_us']
        dur = task['duration_us']

        # Build fanout hint string
        fanout_str = "[" + ", ".join(str(x) for x in task['fanout']) + "]"

        # Get function name if available
        func_id = task['func_id']
        if func_id_to_name and str(func_id) in func_id_to_name:
            func_name = func_id_to_name[str(func_id)]
            # New format: FuncName(task_id)
            task_name = f"{func_name}({task['task_id']})"
        else:
            # Fallback format: Func_{func_id}(task_id)
            task_name = f"Func_{func_id}({task['task_id']})"

        events.append({
            "args": {
                "event-hint": f"Task:{task['task_id']}, FuncId:{func_id}, CoreId:{task['core_id']}",
                "fanout-hint": fanout_str,
                "duration-us": dur,
                "kernel-ready-time-us": task['kernel_ready_time_us'],
                "taskId": task['task_id']
            },
            "cat": "event",
            "id": event_id,
            "name": task_name,
            "ph": "X",
            "pid": 1,
            "tid": tid,
            "ts": ts,
            "dur": dur
        })

        # Record mapping for flow events
        task_to_event_id[task['task_id']] = event_id
        event_id += 1

    # AICPU View duration events (dispatch_time to finish_time)
    if has_aicpu_data:
        for task in tasks:
            dispatch_us = task.get('dispatch_time_us', 0)
            finish_us = task.get('finish_time_us', 0)
            if dispatch_us <= 0 or finish_us <= 0:
                continue

            tid = core_to_tid[task['core_id']]
            aicpu_dur = finish_us - dispatch_us

            # Get function name if available
            func_id = task['func_id']
            if func_id_to_name and str(func_id) in func_id_to_name:
                func_name = func_id_to_name[str(func_id)]
                task_name = f"{func_name}({task['task_id']})"
            else:
                task_name = f"Func_{func_id}({task['task_id']})"

            events.append({
                "args": {
                    "event-hint": f"Task:{task['task_id']}, FuncId:{func_id}, CoreId:{task['core_id']}",
                    "dispatch-time-us": dispatch_us,
                    "finish-time-us": finish_us,
                    "aicpu-duration-us": aicpu_dur,
                    "taskId": task['task_id']
                },
                "cat": "event",
                "id": event_id,
                "name": task_name,
                "ph": "X",
                "pid": 2,
                "tid": tid,
                "ts": dispatch_us,
                "dur": aicpu_dur
            })
            event_id += 1

    # Flow events (Flow events "s" and "f" for dependencies)
    task_map = {t['task_id']: t for t in tasks}
    flow_id = 0

    for task in tasks:
        src_tid = core_to_tid[task['core_id']]
        src_ts_end = task['end_time_us']

        for succ_task_id in task['fanout']:
            if succ_task_id not in task_map:
                if verbose:
                    print(f"Warning: Task {task['task_id']} references non-existent successor {succ_task_id}")
                continue

            succ_task = task_map[succ_task_id]
            dst_tid = core_to_tid[succ_task['core_id']]
            dst_ts_start = succ_task['start_time_us']

            # Get event IDs for source and destination tasks
            src_event_id = task_to_event_id[task['task_id']]
            dst_event_id = task_to_event_id[succ_task['task_id']]

            # Flow start timestamp (at end of source task, slightly before)
            # Use a small offset (0.01 us) for visual clarity
            flow_start_us = src_ts_end - 0.01

            # Flow start event (at end of source task)
            events.append({
                "cat": "flow",
                "id": flow_id,
                "name": "dependency",
                "ph": "s",
                "pid": 1,
                "tid": src_tid,
                "ts": flow_start_us,
                "bind_id": src_event_id
            })

            # Flow finish event (at start of destination task)
            events.append({
                "cat": "flow",
                "id": flow_id,
                "name": "dependency",
                "ph": "f",
                "pid": 1,
                "tid": dst_tid,
                "ts": dst_ts_start,
                "bp": "e",
                "bind_id": dst_event_id
            })

            flow_id += 1

    # AICPU Scheduler phase events (version 2)
    if scheduler_phases:
        # Process metadata
        events.append({
            "args": {"name": "AICPU Scheduler"},
            "cat": "__metadata",
            "name": "process_name",
            "ph": "M",
            "pid": 3
        })
        events.append({
            "args": {"sort_index": 2},
            "cat": "__metadata",
            "name": "process_sort_index",
            "ph": "M",
            "pid": 3
        })

        # Phase color mapping
        phase_colors = {
            "complete": "good",       # green
            "dispatch": "terrible",   # red
            "scan": "thread_state_running",  # blue
            "early_ready": "yellow",  # yellow
        }

        for thread_idx, thread_records in enumerate(scheduler_phases):
            tid = 3000 + thread_idx

            # Thread name metadata
            events.append({
                "args": {"name": f"Sched_{thread_idx}"},
                "cat": "__metadata",
                "name": "thread_name",
                "ph": "M",
                "pid": 3,
                "tid": tid
            })

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
                        "tasks_processed": tasks_processed
                    },
                    "cat": "scheduler",
                    "cname": phase_colors.get(phase, "generic_work"),
                    "name": f"{phase}({tasks_processed})",
                    "ph": "X",
                    "pid": 3,
                    "tid": tid,
                    "ts": start_us,
                    "dur": dur
                }
                events.append(event)

    # AICPU Orchestrator event (version 2)
    if orchestrator_phases or orchestrator_data:
        # Process metadata
        events.append({
            "args": {"name": "AICPU Orchestrator"},
            "cat": "__metadata",
            "name": "process_name",
            "ph": "M",
            "pid": 4
        })
        events.append({
            "args": {"sort_index": 1},
            "cat": "__metadata",
            "name": "process_sort_index",
            "ph": "M",
            "pid": 4
        })

        # Thread name metadata
        events.append({
            "args": {"name": "Orchestrator"},
            "cat": "__metadata",
            "name": "thread_name",
            "ph": "M",
            "pid": 4,
            "tid": 4000
        })

    if orchestrator_phases:
        # Per-task orchestrator phase bars
        orch_phase_colors = {
            "orch_sync": "thread_state_iowait",      # orange
            "orch_alloc": "terrible",                  # red
            "orch_params": "good",                     # green
            "orch_lookup": "thread_state_running",     # blue
            "orch_heap": "yellow",
            "orch_insert": "olive",
            "orch_fanin": "rail_animation",
            "orch_finalize": "cq_build_passed",
            "orch_scope_end": "generic_work",
        }

        for record in orchestrator_phases:
            phase = record.get("phase", "unknown")
            start_us = record["start_time_us"]
            end_us = record["end_time_us"]
            dur = end_us - start_us
            submit_idx = record.get("submit_idx", 0)

            # Strip "orch_" prefix for display name
            display_name = phase.replace("orch_", "") if phase.startswith("orch_") else phase

            event = {
                "args": {
                    "phase": phase,
                    "submit_idx": submit_idx
                },
                "cat": "orchestrator",
                "cname": orch_phase_colors.get(phase, "generic_work"),
                "name": f"{display_name}({submit_idx})",
                "ph": "X",
                "pid": 4,
                "tid": 4000,
                "ts": start_us,
                "dur": dur
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
        events.append({
            "args": orch_args,
            "cat": "orchestrator",
            "name": f"Orchestrator({orchestrator_data.get('submit_count', 0)} tasks)",
            "ph": "X",
            "pid": 4,
            "tid": 4000,
            "ts": orch_start,
            "dur": orch_dur
        })

    # AICPU View fanout arrows (duplicate AICore View flow events using AICPU timestamps)
    if has_aicpu_data:
        for task in tasks:
            src_finish_us = task.get('finish_time_us', 0)
            if src_finish_us <= 0:
                continue

            src_tid = core_to_tid[task['core_id']]

            for succ_task_id in task['fanout']:
                if succ_task_id not in task_map:
                    continue

                succ_task = task_map[succ_task_id]
                dst_dispatch_us = succ_task.get('dispatch_time_us', 0)
                if dst_dispatch_us <= 0:
                    continue

                dst_tid = core_to_tid[succ_task['core_id']]

                events.append({
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dependency",
                    "ph": "s",
                    "pid": 2,
                    "tid": src_tid,
                    "ts": src_finish_us - 0.01
                })
                events.append({
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dependency",
                    "ph": "f",
                    "pid": 2,
                    "tid": dst_tid,
                    "ts": dst_dispatch_us,
                    "bp": "e"
                })
                flow_id += 1

    # Scheduler DISPATCH → task execution arrows
    if scheduler_phases and has_aicpu_data:
        # Build index of DISPATCH phase records per thread for fast lookup
        dispatch_phases_by_thread = {}
        for thread_idx, thread_records in enumerate(scheduler_phases):
            dispatch_records = [r for r in thread_records if r.get("phase") == "dispatch"]
            if dispatch_records:
                dispatch_phases_by_thread[thread_idx] = dispatch_records

        for task in tasks:
            dispatch_us = task.get('dispatch_time_us', 0)
            if dispatch_us <= 0:
                continue

            # Find the scheduler DISPATCH phase that contains this dispatch timestamp
            matched_thread = None
            for thread_idx, dispatch_records in dispatch_phases_by_thread.items():
                for dr in dispatch_records:
                    if dr["start_time_us"] <= dispatch_us <= dr["end_time_us"]:
                        matched_thread = thread_idx
                        break
                if matched_thread is not None:
                    break

            if matched_thread is not None:
                sched_tid = 3000 + matched_thread
                core_tid = core_to_tid[task['core_id']]

                # Flow: scheduler DISPATCH → AICore View task start
                events.append({
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dispatch",
                    "ph": "s",
                    "pid": 3,
                    "tid": sched_tid,
                    "ts": dispatch_us
                })
                events.append({
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dispatch",
                    "ph": "f",
                    "pid": 1,
                    "tid": core_tid,
                    "ts": task['start_time_us'],
                    "bp": "e"
                })
                flow_id += 1

                # Flow: scheduler DISPATCH → AICPU View task start
                events.append({
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dispatch",
                    "ph": "s",
                    "pid": 3,
                    "tid": sched_tid,
                    "ts": dispatch_us
                })
                events.append({
                    "cat": "flow",
                    "id": flow_id,
                    "name": "dispatch",
                    "ph": "f",
                    "pid": 2,
                    "tid": core_tid,
                    "ts": dispatch_us,
                    "bp": "e"
                })
                flow_id += 1

    if verbose:
        print(f"  Total events: {len(events)}")
        print(f"  Flow events: {flow_id}")

    # Step 3: Write JSON file (with traceEvents wrapper to match C++ output)
    with open(output_path, 'w') as f:
        json.dump({"traceEvents": events}, f, indent=2)

    if verbose:
        print(f"JSON written to: {output_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Convert swimlane performance JSON to Chrome Trace Event JSON',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                       # Use latest .json in outputs/, output to outputs/
  %(prog)s perf_swimlane_20260210_143526.json   # Output: outputs/merged_swimlane_20260210_143526.json
  %(prog)s perf_swimlane_20260210_143526.json -o custom_output.json
  %(prog)s perf_swimlane_20260210_143526.json -k examples/host_build_graph/paged_attention/kernels/kernel_config.py
  %(prog)s perf_swimlane_20260210_143526.json -d 0
  %(prog)s perf_swimlane_20260210_143526.json -v
        """
    )

    parser.add_argument('input', nargs='?', help='Input JSON file (.json). If not specified, uses the latest perf_swimlane_*.json file in outputs/ directory')
    parser.add_argument('-o', '--output', help='Output JSON file (default: outputs/merged_swimlane_<timestamp>.json)')
    parser.add_argument('-k', '--kernel-config', help='Path to kernel_config.py file for func_id to function name mapping')
    parser.add_argument('--device-log', help='Device log file/path/glob override used for scheduler analysis')
    parser.add_argument('-d', '--device-id', help='Device id for auto-selection from device-<id>')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

    args = parser.parse_args()

    # If no input file specified, find the latest .json file in outputs/ directory
    if args.input is None:
        outputs_dir = Path(__file__).parent.parent / "outputs"
        json_files = list(outputs_dir.glob("perf_swimlane_*.json"))

        if not json_files:
            print(f"Error: No perf_swimlane_*.json files found in {outputs_dir}", file=sys.stderr)
            print("Please specify an input file or ensure .json files exist in outputs/", file=sys.stderr)
            return 1

        # Get the most recently modified file
        input_path = max(json_files, key=lambda p: p.stat().st_mtime)

        if args.verbose:
            print(f"Auto-selected latest file: {input_path.name}")
    else:
        input_path = Path(args.input)

    # Check input file exists
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}", file=sys.stderr)
        return 1

    try:
        # Read performance data JSON
        if args.verbose:
            print(f"Reading performance data from: {input_path}")

        data = read_perf_data(input_path)

        if args.verbose:
            print("\n=== Performance Data ===")
            print(f"  Version: {data['version']}")
            print(f"  Task Count: {len(data['tasks'])}")

            # Calculate time range
            if data['tasks']:
                start_times = [t['start_time_us'] for t in data['tasks']]
                end_times = [t['end_time_us'] for t in data['tasks']]
                min_time = min(start_times)
                max_time = max(end_times)
                print(f"  Time Range: {min_time:.3f} us - {max_time:.3f} us (span: {max_time - min_time:.3f} us)")
            print()

        # Load function name mapping from kernel_config.py if provided
        func_names = {}
        if args.kernel_config:
            if args.verbose:
                print(f"Loading kernel config from: {args.kernel_config}")
            func_names = load_kernel_config(args.kernel_config)
            if args.verbose:
                print(f"  Loaded {len(func_names)} function name mappings from kernel_config.py:")
                for func_id, name in sorted(func_names.items(), key=lambda x: int(x[0])):
                    print(f"    func_id={func_id}: {name}")
                print()

        # Determine output path
        if args.output:
            output_path = Path(args.output)
        else:
            # Extract timestamp from input filename
            # Expected format: perf_swimlane_YYYYMMDD_HHMMSS.json
            input_stem = input_path.stem  # filename without extension

            # Try to extract timestamp from filename
            if input_stem.startswith("perf_swimlane_"):
                timestamp_part = input_stem[len("perf_swimlane_"):]  # e.g., "20260210_143526"
            else:
                # Fallback: use current timestamp
                timestamp_part = datetime.now().strftime("%Y%m%d_%H%M%S")

            # Generate output filename and save to outputs/ directory
            outputs_dir = Path(__file__).parent.parent / "outputs"
            outputs_dir.mkdir(exist_ok=True)  # Ensure outputs directory exists
            output_path = outputs_dir / f"merged_swimlane_{timestamp_part}.json"

        # Resolve device log path for scheduler analysis
        resolved_device_log, log_strategy = resolve_device_log_path(
            device_id=args.device_id,
            device_log=args.device_log,
            perf_path=input_path,
        )

        # Extract version 2 data if available
        scheduler_phases = data.get('aicpu_scheduler_phases')
        orchestrator_data = data.get('aicpu_orchestrator')
        orchestrator_phases = data.get('aicpu_orchestrator_phases')

        if args.verbose and data['version'] == 2:
            if scheduler_phases:
                total_phase_records = sum(len(t) for t in scheduler_phases)
                print(f"  Scheduler threads: {len(scheduler_phases)}")
                print(f"  Total phase records: {total_phase_records}")
            if orchestrator_data:
                print(f"  Orchestrator: {orchestrator_data.get('submit_count', 0)} tasks")
            if orchestrator_phases:
                print(f"  Orchestrator phases: {len(orchestrator_phases)} per-task records")

        # Generate Perfetto JSON
        generate_chrome_trace_json(data['tasks'], str(output_path), func_names, args.verbose,
                                   scheduler_phases=scheduler_phases,
                                   orchestrator_data=orchestrator_data,
                                   orchestrator_phases=orchestrator_phases)

        print(f"\n✓ Conversion complete")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"\nTo visualize: Open https://ui.perfetto.dev/ and drag in {output_path}")

        if resolved_device_log is not None:
            print(f"\nDevice log: {resolved_device_log}")
            print(f"Selection: {log_strategy}")
            inferred_device_id = infer_device_id_from_log_path(resolved_device_log)
            if inferred_device_id is not None:
                print(f"Inferred Device ID: {inferred_device_id}")
        else:
            print(f"\nDevice log: (not resolved)")
            print(f"Selection: {log_strategy}")

        # Optional: parse sched CPU from device log
        sched_cpu_us = None
        if resolved_device_log is not None:
            sched_cpu_us = parse_sched_cpu_from_device_log(resolved_device_log, len(data['tasks']))
            if args.verbose and sched_cpu_us is not None:
                print(f"  Parsed sched CPU from device log: {sched_cpu_us:.2f} us/task")

        # Print task statistics (incl. task execution vs scheduler overhead)
        print_task_statistics(data['tasks'], func_names, sched_cpu_us_per_task=sched_cpu_us)

        # Integrated deep-dive overhead analysis
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
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
