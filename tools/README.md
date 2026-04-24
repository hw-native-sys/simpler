# Swimlane Performance Analysis Tools

This directory contains performance analysis tools for the PTO Runtime.

## Tool List

- **[swimlane_converter.py](#swimlane_converterpy)** - Convert to Chrome Trace Event visualization format
- **[sched_overhead_analysis.py](#sched_overhead_analysispy)** - Scheduler overhead analysis (Tail OH breakdown)
- **[perf_to_mermaid.py](#perf_to_mermaidpy)** - Convert to Mermaid dependency graph
- **[benchmark_rounds.sh](#benchmark_roundssh)** - Batch-run examples and report per-round elapsed time
- **[device_log_resolver.py](#device_log_resolverpy)** - Device log path resolution library

---

## swimlane_converter.py

Convert performance profiling JSON files into Chrome Trace Event format for visualization in Perfetto.

### Overview

`swimlane_converter.py` converts PTO Runtime profiling data (`l2_perf_records_*.json`) into a format that can be visualized in the Perfetto trace viewer (<https://ui.perfetto.dev/>). It also produces a task execution statistics summary grouped by function, and emits a scheduler overhead deep-dive report when a device log is resolved.

### Basic Usage

```bash
# Auto-detect the latest profiling file in the outputs/ directory
python3 tools/swimlane_converter.py

# Specify an input file
python3 tools/swimlane_converter.py outputs/l2_perf_records_20260210_143526.json

# Specify an output file
python3 tools/swimlane_converter.py outputs/l2_perf_records_20260210_143526.json -o custom_output.json

# Load function name mapping from kernel_config.py
python3 tools/swimlane_converter.py outputs/l2_perf_records_20260210_143526.json \
    -k examples/host_build_graph/paged_attention/kernels/kernel_config.py

# Select the device log automatically using a specific device id (device-<id>)
python3 tools/swimlane_converter.py outputs/l2_perf_records_20260210_143526.json -d 0

# Verbose mode (for debugging)
python3 tools/swimlane_converter.py outputs/l2_perf_records_20260210_143526.json -v
```

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `input` | | Input JSON file (l2_perf_records_*.json). If omitted, the latest file in outputs/ is used |
| `--output` | `-o` | Output JSON file (default: outputs/merged_swimlane_`<timestamp>`.json) |
| `--kernel-config` | `-k` | Path to kernel_config.py, used for function name mapping |
| `--device-log` | | Device log file/directory/glob that overrides inputs (highest priority) |
| `--device-id` | `-d` | Device id used to auto-select the log from the `device-<id>` directory |
| `--verbose` | `-v` | Enable verbose output |

### Device Log Selection Priority

`swimlane_converter.py` and `sched_overhead_analysis.py` use the same resolution rules (provided by `device_log_resolver.py`):

1. `--device-log` (file/directory/glob) explicit override
2. `-d/--device-id` maps to the `device-<id>` directory
3. Auto-scan `device-*`, selecting the `.log` closest to the perf timestamp

Log root resolution order:

- `$ASCEND_WORK_PATH/log/debug/`
- `~/ascend/log/debug/` (fallback)

### Outputs

The tool produces three kinds of output:

#### 1. Perfetto JSON File

A Chrome Trace Event format JSON file that can be visualized in Perfetto:

- File location: `outputs/merged_swimlane_<timestamp>.json`
- Open <https://ui.perfetto.dev/> and drag-and-drop the file to visualize

#### 2. Task Statistics

A statistics summary grouped by function (printed to the console), including Exec/Latency comparison and scheduling overhead analysis:

- **Exec**: kernel execution time on AICore (end_time - start_time)
- **Latency**: end-to-end latency from the AICPU perspective (finish_time - dispatch_time, including head OH + Exec + tail OH)
- **Head/Tail OH**: scheduling head/tail overhead
- **Exec_%**: Exec / Latency percentage (kernel utilization)

When the device log is resolved, Sched CPU (actual CPU time of the AICPU scheduler thread per task) and the Exec/Sched_CPU ratio are also printed.

#### 3. Scheduler Overhead Deep-Dive (Automatic)

When the device log is successfully resolved, `swimlane_converter.py` invokes the `sched_overhead_analysis` logic directly and emits in the same run:

- Part 1: Per-task time breakdown
- Part 2: AICPU scheduler loop breakdown
- Part 3: Tail OH distribution & cause analysis

### Integration with run_example.py

When running a test with profiling enabled, the converter is invoked automatically:

```bash
# Run the test with profiling enabled - merged_swimlane.json is generated automatically after the test passes
python examples/scripts/run_example.py \
    -k examples/host_build_graph/vector_example/kernels \
    -g examples/host_build_graph/vector_example/golden.py \
    --enable-l2-swimlane
```

After the test passes, the tool will:

1. Auto-detect the latest `l2_perf_records_*.json` in outputs/
2. Load function names from the kernel_config.py specified via `-k`
3. Propagate the effective runtime device id (`-d`) to `swimlane_converter.py`
4. Resolve the device log automatically and print the selection strategy
5. Produce `merged_swimlane_*.json` for visualization
6. Print the task statistics and scheduler overhead deep-dive report to the console

---

## sched_overhead_analysis.py

Analyze AICPU scheduler overhead and quantitatively decompose the sources of Tail OH (the latency between task completion and scheduler acknowledgement).

### Overview

`sched_overhead_analysis.py` analyzes two data sources:

1. **Perf profiling data** (`l2_perf_records_*.json`): extracts per-task Exec / Head OH / Tail OH time breakdowns
2. **Device log**: parses the AICPU scheduler thread's loop breakdown (scan / complete / dispatch / idle), lock contention, and fanout statistics

Three device log formats are supported:

1. **New two-level tree** (`PTO2_SCHED_PROFILING=1`): `=== Scheduler Phase Breakdown: total=Xus, Y tasks ===` followed by per-phase lines
2. **Legacy detailed** (`PTO2_SCHED_PROFILING=1`): `completed=X tasks in Yus (Z loops, W tasks/loop)` followed by `--- Phase Breakdown ---` with phase lines carrying fanout/fanin/pop statistics
3. **Summary** (`PTO2_SCHED_PROFILING=0`): `Scheduler summary: total_time=Xus, loops=Y, tasks_scheduled=Z`

### Basic Usage

```bash
# Auto-pick the latest perf data and device log
python3 tools/sched_overhead_analysis.py

# Use a specific device id to auto-pick the device-<id> log
python3 tools/sched_overhead_analysis.py --l2-perf-records-json outputs/l2_perf_records_20260210_143526.json -d 0

# Specify files explicitly
python3 tools/sched_overhead_analysis.py \
    --l2-perf-records-json outputs/l2_perf_records_20260210_143526.json \
    --device-log ~/ascend/log/debug/device-0/device-*.log
```

### Command-Line Options

| Option | Description |
| ------ | ----------- |
| `--l2-perf-records-json` | Path to the l2_perf_records_*.json file. If omitted, the latest file in outputs/ is auto-selected |
| `--device-log` | Device log file/directory/glob that overrides inputs (highest priority) |
| `-d, --device-id` | Device id used to auto-pick the log from `device-<id>` |

### Outputs

Output is emitted in three parts:

- **Part 1: Per-task time breakdown** - Exec / Head OH / Tail OH percentages of Latency
- **Part 2: AICPU scheduler loop breakdown** - per-scheduler-thread loop statistics, per-phase (scan / complete / dispatch / idle) time ratios, lock contention, and fanout/fanin/pop statistics
- **Part 3: Tail OH distribution & cause analysis** - Tail OH quantile distribution (P10-P99), correlation between scheduler loop iteration time and Tail OH, and data-driven insights into the dominant phase

---

## perf_to_mermaid.py

Convert profiling data into Mermaid flowchart format to visualize task dependencies.

### Overview

`perf_to_mermaid.py` converts PTO Runtime profiling data (`l2_perf_records_*.json`) into Mermaid flowchart format. The generated Markdown file can be:

- Rendered directly in GitHub/GitLab
- Viewed at <https://mermaid.live/>
- Viewed in Mermaid-capable editors (e.g., VS Code with the Mermaid plugin)

### Basic Usage

```bash
# Auto-detect the latest profiling file in the outputs/ directory
python3 tools/perf_to_mermaid.py

# Specify an input file
python3 tools/perf_to_mermaid.py outputs/l2_perf_records_20260210_143526.json

# Specify an output file
python3 tools/perf_to_mermaid.py outputs/l2_perf_records_20260210_143526.json -o diagram.md

# Load function name mapping from kernel_config.py
python3 tools/perf_to_mermaid.py outputs/l2_perf_records_20260210_143526.json \
    -k examples/host_build_graph/paged_attention/kernels/kernel_config.py

# Use compact style (only task id and function name)
python3 tools/perf_to_mermaid.py outputs/l2_perf_records_20260210_143526.json --style compact

# Specify flowchart direction (left to right)
python3 tools/perf_to_mermaid.py outputs/l2_perf_records_20260210_143526.json --direction LR

# Verbose mode
python3 tools/perf_to_mermaid.py outputs/l2_perf_records_20260210_143526.json -v
```

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `input` | | Input JSON file (l2_perf_records_*.json). If omitted, the latest file in outputs/ is used |
| `--output` | `-o` | Output Markdown file (default: outputs/mermaid_diagram_`<timestamp>`.md) |
| `--kernel-config` | `-k` | Path to kernel_config.py, used for function name mapping |
| `--style` | | Node style: `detailed` (default, includes function name and task id) or `compact` (task id only) |
| `--direction` | | Flowchart direction: `TD` (top-down, default) or `LR` (left-to-right) |
| `--verbose` | `-v` | Enable verbose output |

### Outputs

Generates a Markdown file containing a Mermaid flowchart:

#### Detailed Style (Default)

```mermaid
flowchart TD

    Task0["QK(0)"]
    Task1["SF(1)"]
    Task2["PV(2)"]
    Task3["UP(3)"]
    Task4["QK(4)"]
    Task5["SF(5)"]
    Task6["PV(6)"]
    Task7["UP(7)"]
    Task8["QK(8)"]
    Task9["SF(9)"]
    Task10["PV(10)"]
    Task11["UP(11)"]
    Task12["QK(12)"]
    Task13["SF(13)"]
    Task14["PV(14)"]
    Task15["UP(15)"]

    Task0 --> Task1
    Task1 --> Task2
    Task2 --> Task3
    Task3 --> Task7
    Task4 --> Task5
    Task5 --> Task6
    Task6 --> Task7
    Task8 --> Task9
    Task9 --> Task10
    Task10 --> Task11
    Task11 --> Task15
    Task12 --> Task13
    Task13 --> Task14
    Task14 --> Task15

    %% Styling by core type
    classDef aicStyle fill:#66A3FF,stroke:#333,stroke-width:2px,color:#000
    classDef aivStyle fill:#FFB366,stroke:#333,stroke-width:2px,color:#000

    class Task0,Task2,Task4,Task6,Task8,Task10,Task12,Task14 aicStyle
    class Task1,Task3,Task5,Task7,Task9,Task11,Task13,Task15 aivStyle
```

---

## benchmark_rounds.sh

Batch-run a predefined set of examples, parse the timing lines from the device log, and report per-round elapsed time.

### Overview

`benchmark_rounds.sh` iterates through the test cases configured in the `EXAMPLES` array (located under `tests/st/tensormap_and_ringbuffer/`), invokes `run_example.py` for each example in turn, then extracts the `orch_start` / `orch_end` / `sched_end` timestamps from the generated device log to compute per-round elapsed time.

Currently preconfigured examples:

- `alternating_matmul_add`
- `benchmark_bgemm`
- `paged_attention_unroll`
- `batch_paged_attention`
- `paged_attention`

### Basic Usage

```bash
# Use defaults (device 0, 10 rounds)
./tools/benchmark_rounds.sh

# Specify device and rounds
./tools/benchmark_rounds.sh -d 4 -n 20

# Extra arguments are passed through to run_example.py
./tools/benchmark_rounds.sh -d 0 -n 5 --case 1
```

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `--device` | `-d` | Device ID (default: 0) |
| `--rounds` | `-n` | Number of rounds per example (default: 10) |
| `--help` | `-h` | Show help |

All unrecognized arguments are passed through to `run_example.py`.

### Outputs

For each example it prints:

- Elapsed time (microseconds) for each round
- Average time and total round count

A final summary is printed: passed / failed counts.

### Device Log Resolution

The script locates the device log as follows:

- Prefer `$ASCEND_WORK_PATH/log/debug/device-<id>/`
- Fall back to `~/ascend/log/debug/device-<id>/`
- Snapshot existing log files before running, then wait for new log files to appear after running (up to 15 seconds)

---

## device_log_resolver.py

Device log path resolution library, shared by `swimlane_converter.py` and `sched_overhead_analysis.py`.

### Overview

`device_log_resolver.py` provides deterministic device log path resolution logic, supporting three selection priorities:

1. **Explicit path** (`--device-log`): supports file, directory, and glob patterns
2. **Device ID** (`--device-id`): selects the latest `.log` from `<log_root>/device-<id>/`
3. **Auto-scan**: iterates all `device-*` directories and selects the `.log` closest to the perf timestamp

### Main Functions

| Function | Description |
| -------- | ----------- |
| `get_log_root()` | Returns the log root path (`$ASCEND_WORK_PATH/log/debug/` or `~/ascend/log/debug/`) |
| `infer_device_id_from_log_path(log_path)` | Infers the device id from a path (e.g., `device-0`) |
| `resolve_device_log_path(device_id, device_log, l2_perf_records_path)` | Resolves the device log path by priority, returning `(Path, strategy_string)` |

### Usage

This module is not used as a standalone command-line tool; it is imported by other tools:

```python
from device_log_resolver import resolve_device_log_path

log_path, strategy = resolve_device_log_path(
    device_id="0",
    device_log=None,
    l2_perf_records_path=Path("outputs/l2_perf_records_20260210_143526.json"),
)
```

---

## Shared Configuration

### Input File Format

The analysis tools share the same input format - the `l2_perf_records_*.json` files generated by the PTO Runtime:

```json
{
  "version": 1,
  "tasks": [
    {
      "task_id": 0,
      "func_id": 0,
      "core_id": 0,
      "core_type": "aic",
      "start_time_us": 100.0,
      "end_time_us": 250.5,
      "duration_us": 150.5,
      "fanout": [1, 2],
      "fanout_count": 2
    }
  ]
}
```

### Kernel Config Format

To display meaningful function names in the output, provide a `kernel_config.py` file:

```python
KERNELS = [
    {
        "func_id": 0,
        "name": "QK",
        # ... other fields
    },
    {
        "func_id": 1,
        "name": "SF",
        # ... other fields
    },
]
```

The tools extract the `func_id` to `name` mapping from the `KERNELS` list.

---

## Tool Selection Guide

### Use swimlane_converter.py when you need

- A detailed timeline execution view
- To analyze task scheduling across different cores
- To see precise execution times and intervals
- Task execution statistics
- Professional performance analysis and optimization

### Use perf_to_mermaid.py when you need

- A quick look at task dependencies
- To embed a dependency graph in documentation
- To share dependency structure in a code review
- Only topological structure, without timeline detail
- Direct viewing in GitHub/GitLab

### Use benchmark_rounds.sh when you need

- To batch-run multiple examples and compare elapsed time
- Per-round elapsed time statistics
- End-to-end performance regression testing on hardware

### Recommended Workflow

```bash
# 1. Run the test to produce profiling data
python examples/scripts/run_example.py -k ./kernels -g ./golden.py --enable-l2-swimlane

# 2. Generate Perfetto visualization (automatic)
# -> outputs/merged_swimlane_*.json

# 3. Generate Mermaid dependency graph
python3 tools/perf_to_mermaid.py -k ./kernels/kernel_config.py

# 4. Batch benchmark (on hardware)
./tools/benchmark_rounds.sh -d 0 -n 20

# 5. Analyze results
# - Detailed performance analysis: Perfetto (https://ui.perfetto.dev/)
# - Dependency overview: Mermaid diagram (GitHub/editor)
# - Statistics summary: console output
```

---

## Troubleshooting

### Error: cannot find l2_perf_records_*.json file

- Make sure the test was run with the `--enable-l2-swimlane` flag
- Check that the outputs/ directory exists and contains profiling data

### Warning: Kernel entry missing 'func_id' or 'name'

- Check the kernel_config.py file format
- Make sure every KERNELS entry has a 'func_id' and 'name' field

### Error: Unsupported version

- The tools only support version 1 of the profiling data format
- Regenerate the profiling data with the latest runtime

### Error: Perf JSON missing required fields for scheduler overhead analysis

- This error means the input `l2_perf_records_*.json` lacks fields required by the deep-dive analysis (typically `dispatch_time_us` / `finish_time_us`)
- The basic conversion in `swimlane_converter.py` can still succeed, but the deep-dive will be skipped or fail
- Remediation:
  1. Re-run with `--enable-l2-swimlane` to produce a new `outputs/l2_perf_records_*.json`
  2. Re-run `swimlane_converter.py` or `sched_overhead_analysis.py`
  3. Verify that each task in the JSON contains `dispatch_time_us` and `finish_time_us`

### benchmark_rounds.sh has no timing data

- Make sure profiling is enabled at runtime (`PTO2_PROFILING` environment variable)
- Check that the device log directory is accessible
- Confirm the log contains `orch_start` / `orch_end` / `sched_end` timestamp lines (requires `PTO2_PROFILING=1`)

### Mermaid diagram does not render on GitHub

- Make sure the file has the `.md` extension
- Check that the Mermaid syntax is correct
- GitHub sometimes needs a refresh before rendering Mermaid diagrams

---

## Output File Reference

| File | Tool | Purpose | Format |
| ---- | ---- | ------- | ------ |
| `l2_perf_records_*.json` | Runtime | Raw profiling data | JSON |
| `merged_swimlane_*.json` | swimlane_converter.py | Perfetto visualization | Chrome Trace Event JSON |
| `mermaid_diagram_*.md` | perf_to_mermaid.py | Dependency graph | Markdown + Mermaid |

---

## Related Resources

- [Perfetto Trace Viewer](https://ui.perfetto.dev/)
- [Mermaid Live Editor](https://mermaid.live/)
- [Mermaid documentation](https://mermaid.js.org/)
