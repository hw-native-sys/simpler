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
    if data['version'] != 1:
        raise ValueError(f"Unsupported version: {data['version']} (expected 1)")

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


def print_task_statistics(tasks, func_id_to_name=None):
    """Print task statistics grouped by func_id.

    Args:
        tasks: List of task dicts
        func_id_to_name: Optional dict mapping func_id to function name
    """
    from collections import defaultdict

    # Group tasks by func_id
    func_stats = defaultdict(list)
    for task in tasks:
        func_id = task['func_id']
        duration = task['duration_us']
        func_stats[func_id].append(duration)

    # Print statistics
    print("\n" + "=" * 104)
    print("Task Statistics by Function")
    print("=" * 104)
    print(f"{'Func ID':<8} {'Func Name':<20} {'Count':>8} {'Total (us)':>14} {'Avg (us)':>12} {'Min (us)':>12} {'Max (us)':>12}")
    print("-" * 104)

    # Sort by func_id for consistent output
    total_count = 0
    total_duration = 0.0

    for func_id in sorted(func_stats.keys()):
        durations = func_stats[func_id]
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

        print(f"{func_id:<8} {func_name:<20} {count:>8} {sum_duration:>14.2f} {avg_duration:>12.2f} {min_duration:>12.2f} {max_duration:>12.2f}")

    # Print total row
    print("-" * 104)
    print(f"{'TOTAL':<29} {total_count:>8} {total_duration:>14.2f}")
    print("=" * 104)



def generate_chrome_trace_json(tasks, output_path, func_id_to_name=None, verbose=False):
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

    Generates two processes in the trace:
        - pid=1 "AICore View": start_time_us to end_time_us (kernel execution)
        - pid=2 "AICPU View": dispatch_time_us to finish_time_us (AICPU perspective)
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

    # Metadata event: Process names
    events.append({
        "args": {"name": "AICore View"},
        "cat": "__metadata",
        "name": "process_name",
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
  %(prog)s perf_swimlane_20260210_143526.json -v
        """
    )

    parser.add_argument('input', nargs='?', help='Input JSON file (.json). If not specified, uses the latest perf_swimlane_*.json file in outputs/ directory')
    parser.add_argument('-o', '--output', help='Output JSON file (default: outputs/merged_swimlane_<timestamp>.json)')
    parser.add_argument('-k', '--kernel-config', help='Path to kernel_config.py file for func_id to function name mapping')
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

        # Generate Perfetto JSON
        generate_chrome_trace_json(data['tasks'], str(output_path), func_names, args.verbose)

        print(f"\n✓ Conversion complete")
        print(f"  Input:  {input_path}")
        print(f"  Output: {output_path}")
        print(f"\nTo visualize: Open https://ui.perfetto.dev/ and drag in {output_path}")

        # Print task statistics
        print_task_statistics(data['tasks'], func_names)

        return 0

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())
