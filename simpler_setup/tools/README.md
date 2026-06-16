# Profiling & Debug Tools (shipped in the wheel)

End-user CLIs for analyzing PTO Runtime profiling data and tensor dumps.
All are invokable as Python modules once the `simpler` wheel is installed —
no repo checkout required.

> Dev-only scripts (`benchmark_rounds.sh`, `verify_packaging.sh`) live in the
> repo-level [`tools/`](../../tools/) directory and are **not** shipped.

## Tool list

- **[swimlane_converter](#swimlane_converter)** — perf JSON → Chrome Trace Event (Perfetto)
- **[sched_overhead_analysis](#sched_overhead_analysis)** — scheduler overhead / Tail OH breakdown
- **[device_log_timing](#device_log_timing)** — Total / Orch / Sched from a CANN device log (no swimlane JSON)
- **[dump_viewer](#dump_viewer)** — tensor dump inspection and export
- **[deps_viewer](#deps_viewer)** — `deps.json` (dep_gen) → text or pan/zoom HTML dependency graph

Auto-detection paths (`outputs/*/l2_swimlane_records.json`, `outputs/*/tensor_dump/`)
are resolved relative to the **current working directory** — run these from the
directory that holds your `outputs/`. Each test case writes into its own
`outputs/<case>_<ts>/` directory; the tools auto-pick the latest by mtime.

---

## swimlane_converter

Convert performance profiling JSON files into Chrome Trace Event format for visualization in Perfetto.

### Overview

Converts PTO Runtime profiling data (`l2_swimlane_records_*.json`) into the format used by the Perfetto trace viewer (<https://ui.perfetto.dev/>). It also produces a task execution statistics summary grouped by function and a scheduler overhead deep-dive report (the same one `sched_overhead_analysis` emits).

### Basic Usage

```bash
# Auto-detect the latest profiling file under ./outputs/
python -m simpler_setup.tools.swimlane_converter

# Specify an input file
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json

# Specify an output file
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json -o custom_output.json

# Load function name mapping from kernel_config.py
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json \
    -k examples/host_build_graph/paged_attention/kernels/kernel_config.py

# Verbose mode (for debugging)
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json -v

# Reuse a deps.json captured in an earlier dep_gen run (different output dir)
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json \
    --deps-json outputs/<case>_<earlier_ts>/deps.json
```

> Dependency arrows in the Perfetto trace come from `deps.json` (dep_gen
> replay). The device hot path no longer records fanout, so the typical
> workflow is **two runs**: a one-time `--enable-dep-gen` capture per
> topology to produce `deps.json`, then any number of
> `--enable-l2-swimlane` runs that consume it. If no `deps.json` is found
> alongside the perf JSON (and `--deps-json` isn't passed), the trace
> still renders but has no arrows; the converter prints a warning.

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `input` | | Input JSON file (l2_swimlane_records_*.json). If omitted, the latest file in outputs/ is used |
| `--output` | `-o` | Output JSON file (default: outputs/merged_swimlane_`<timestamp>`.json) |
| `--kernel-config` | `-k` | Path to kernel_config.py, used for function name mapping |
| `--func-names` | | Path to func_id_names_*.json (SceneTest format) for function name mapping |
| `--deps-json` | | Path to a dep_gen `deps.json` (defaults to sibling of input). Without one, no dependency arrows are drawn. |
| `--verbose` | `-v` | Enable verbose output |

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

#### 3. Scheduler Overhead Deep-Dive (Automatic)

`swimlane_converter` invokes `sched_overhead_analysis` directly on the same
perf JSON and emits, in the same run:

- Part 1: Per-task time breakdown
- Part 2: AICPU scheduler loop breakdown (from `aicpu_scheduler_phases`)
- Part 3: Tail OH distribution & cause analysis

When `deps.json` is colocated (produced by `--enable-dep-gen`), Part 2 also
prints per-thread fanout / fanin aggregates.

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

1. Auto-detect the latest `l2_swimlane_records_*.json` in outputs/
2. Load function names from the kernel_config.py specified via `-k`
3. Produce `merged_swimlane_*.json` for visualization
4. Print the task statistics and scheduler overhead deep-dive report to the console

---

## sched_overhead_analysis

Analyze AICPU scheduler overhead and quantitatively decompose the sources of Tail OH (the latency between task completion and scheduler acknowledgement).

### Overview

`sched_overhead_analysis` reads two artifacts produced by the runtime:

1. **Perf profiling data** (`l2_swimlane_records_*.json`, l2_swimlane_level >= 3): per-task Exec / Head OH / Tail OH time breakdowns plus `aicpu_scheduler_phases` — per-thread, per-loop-iteration phase records carrying scan / complete / dispatch / idle timings and per-emit pop_hit / pop_miss deltas.
2. **`deps.json`** (optional, dep_gen replay output): structural task DAG. When colocated with the perf JSON, Part 2 prints per-thread fanout / fanin aggregates derived from it.

### Basic Usage

```bash
# Auto-pick the latest perf data under ./outputs/ (deps.json sibling is auto-detected)
python -m simpler_setup.tools.sched_overhead_analysis

# Specify the perf JSON explicitly
python -m simpler_setup.tools.sched_overhead_analysis \
    --l2-swimlane-records-json outputs/<case>_<ts>/l2_swimlane_records.json

# Override the deps.json location
python -m simpler_setup.tools.sched_overhead_analysis \
    --l2-swimlane-records-json outputs/<case>_<ts>/l2_swimlane_records.json \
    --deps-json outputs/<case>_<ts>/deps.json
```

> For Total / Orch / Sched timing from a **plain** run (no swimlane JSON), use
> [`device_log_timing`](#device_log_timing) instead — `sched_overhead_analysis`
> is the swimlane-JSON deep dive.

### Command-Line Options

| Option | Description |
| ------ | ----------- |
| `--l2-swimlane-records-json` | Path to the l2_swimlane_records_*.json file. If omitted, the latest file in outputs/ is auto-selected |
| `--deps-json` | Path to deps.json (dep_gen replay output) for fanout / fanin aggregates. Defaults to the deps.json sibling of the perf JSON. |

### Outputs

Output is emitted in three parts:

- **Part 1: Per-task time breakdown** — Exec / Head OH / Tail OH percentages of Latency
- **Part 2: AICPU scheduler loop breakdown** — per-scheduler-thread loop statistics, per-phase (scan / complete / dispatch / idle) time ratios, pop_hit / pop_miss totals, and (when deps.json is available) per-thread fanout / fanin aggregates
- **Part 3: Tail OH distribution & cause analysis** — Tail OH quantile distribution (P10–P99), correlation between scheduler loop iteration time and Tail OH, and data-driven insights into the dominant phase

The perf JSON must be captured at l2_swimlane_level >= 3 so that `aicpu_scheduler_phases` is non-empty (rerun the case with `--enable-l2-swimlane` if the tool reports the field is missing).

---

## device_log_timing

Print per-round **Total / Orch / Sched** timing parsed from a CANN device log's
`PTO2_PROFILING` orch/sched markers. Unlike `sched_overhead_analysis` (which
reads the swimlane JSON), this needs **no swimlane capture** — use it for a
plain benchmark run, a `--rounds N` sweep, or an external workload that never
produces an `l2_swimlane_records.json`.

### Basic Usage

```bash
# Explicit file or glob (quote the glob so the shell doesn't expand it; a glob
# parses all matched rotated files)
python -m simpler_setup.tools.device_log_timing \
    --device-log '~/ascend/log/debug/device-0/device-*.log'

# Auto-pick the newest log under device-<id>/
python -m simpler_setup.tools.device_log_timing -d 0
```

To get the same table emitted automatically by the test harness, pass
`--enable-device-log-timing` to `scene_test` / pytest (onboard L2 only; works
with `--rounds N`). See [docs/dfx/l2-timing.md](../../docs/dfx/l2-timing.md) for
the full guide, including the `RunTiming` host_wall / device_wall numbers and
how they relate to Orch / Sched / Total.

### Command-Line Options

| Option | Description |
| ------ | ----------- |
| `--device-log` | Path / dir / glob of a CANN device log. A glob parses every matched (rotated) file. |
| `-d`, `--device-id` | Device id: auto-pick the newest log under `device-<id>/`. |

### CANN device-log environment variables

| Env var | Effect |
| ------- | ------ |
| `ASCEND_PROCESS_LOG_PATH` | Relocates the log root to `$ASCEND_PROCESS_LOG_PATH/debug` (highest precedence, above `ASCEND_WORK_PATH/log/debug` and the `<euid-home>/ascend/log/debug` default). Resolved automatically. |
| `ASCEND_SLOG_PRINT_TO_STDOUT=1` | Routes CANN logs to stdout — **no device log file is written**; the CLI errors out and the harness flag skips. Unset it (or set `0`) to capture device timing. |
| `ASCEND_HOST_LOG_FILE_NUM` | Rotated files retained per process (default 10). Each file caps at 20 MB; a long run can rotate mid-run, so the harness reads **all** files written after the run started, and a `--device-log` glob parses all matches. |

The default root (no relocation env var) is `<euid-home>/ascend/log/debug`,
using the effective uid's passwd home (e.g. `/root` under sudo / `task-submit`),
which is where the driver actually writes device logs — not `$HOME`, which sudo
often leaves pointing at the invoking user.

---

## deps_viewer

Render the dep_gen `deps.json` task graph as either grep-friendly text
(default) or a self-contained pan/zoom HTML page. Pairs naturally with
[`swimlane_converter`](#swimlane_converter): swimlane is the timing view,
this is the structural view.

### Overview

`deps_viewer` reads `deps.json` produced by the dep_gen replay (see
[docs/dfx/dep_gen.md](../../docs/dfx/dep_gen.md)) and supports two modes:

- **Default text mode** — emits `deps_viewer.txt` with:
  - `SUMMARY` (input path plus task / edge / tensor counts)
    - `tasks`: number of rendered task ids
    - `unique_task_edges`: number of unique `(pred, succ)` pairs
    - `annotated_edges`: total number of annotated edge rows
    - `perf_sidecar`: `yes` when `l2_swimlane_records.json` was successfully loaded
    - `func_name_map`: `yes` when at least one task name resolved to a named
      `func_name` from `--func-names` or an auto-discovered `name_map_*.json`.
      `func_name_map` stays `no` unless a real human-readable name was resolved.
  - `TASK INDEX` (one line per task for grep)
    - `kind=` distinguishes `submit` / `dummy` / `alloc` / `unknown`
    - `func_id=` is taken only from `tasks[].kernel_ids` and shows the aligned
      three-slot `[aic,aiv0,aiv1]` array for `submit`
    - `kind=alloc` / `kind=dummy` render as `func_id=none`
  - `TASK DETAILS` (per-task `FANIN` / `FANOUT` blocks showing peer task references only)
  Best for "what does task X depend on?" and large-graph debugging.
- **`--format html`** — renders the task graph as Graphviz SVG wrapped in a
  self-contained HTML file viewable in any modern browser.
  - Add `--show-tensor-info` to restore per-task tensor rows and edge routing
    to specific arg ports in the HTML view.

### Basic Usage

```bash
# Auto-pick the newest deps.json under ./outputs/ -> deps_viewer.txt
python -m simpler_setup.tools.deps_viewer

# Specific path -> deps_viewer.txt next to deps.json
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json

# Explicit text output path
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json -o graph.txt

# HTML output
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json \
    --format html -o graph.html

# HTML output with per-task tensor details and arg-port routing
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json \
    --format html --show-tensor-info -o graph.html

# Force-directed HTML layout for large graphs (>~1000 nodes)
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json \
    --format html --engine sfdp

# Override task labels with a func_id -> name mapping
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json \
    --func-names outputs/<case>_<ts>/name_map_TestPA_basic.json
```

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `input` | | Path to `deps.json` (default: newest under `./outputs/`) |
| `--output` | `-o` | Output path (default: `deps_viewer.txt` for text, `deps_viewer.html` for HTML) |
| `--format` | | Output format: `text` (default) or `html` |
| `--engine` | | HTML-only Graphviz layout engine: `dot` (default), `sfdp`, `neato`, `fdp`, `circo`, `twopi` |
| `--direction` | | HTML-only flow direction for hierarchical layouts: `LR` (default) / `TB` / `BT` / `RL` |
| `--show-tensor-info` | | HTML-only: render per-task tensor rows and route edges to specific arg ports |
| `--func-names` | | JSON file with `callable_id_to_name` (or flat `{func_id: name}`) for task-label enrichment |

### Dependencies

Text output has no extra dependencies. HTML output requires Graphviz on PATH:

```bash
brew install graphviz    # macOS
apt install graphviz     # Debian/Ubuntu
```

The HTML viewer is self-contained — no JavaScript or fonts are downloaded
at view time.

### Browser controls

- **drag** → pan
- **wheel** → zoom about cursor
- **f** → fit to view
- **r** → reset to 1:1

---

## dump_viewer

Inspect and export tensors captured by the runtime tensor-dump feature.
See [docs/tensor-dump.md](../../docs/dfx/tensor-dump.md) for the full capture workflow;
this section only documents CLI invocation.

### Basic Usage

```bash
# List all tensors (auto-picks latest outputs/tensor_dump_* dir)
python -m simpler_setup.tools.dump_viewer

# Filter by stage/role/func_id
python -m simpler_setup.tools.dump_viewer --func 3 --stage before --role input

# Export the current selection to txt
python -m simpler_setup.tools.dump_viewer --func 3 --stage before --role input --export

# Export a specific tensor by index (always exports)
python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/tensor_dump/ --index 42
```

---

## Shared Configuration

### Input File Format

The analysis tools share the same input format - the `l2_swimlane_records_*.json` files generated by the PTO Runtime:

```json
{
  "l2_swimlane_level": 4,
  "tasks": [
    {
      "task_id": 0,
      "func_id": 0,
      "core_id": 7,
      "core_type": "aiv",
      "ring_id": 0,
      "start_time_us": 47.46,
      "end_time_us": 55.9,
      "duration_us": 8.44,
      "dispatch_time_us": 45.94,
      "finish_time_us": 60.52
    },
    {
      "task_id": 4294967296,
      "func_id": 1,
      "core_id": 7,
      "core_type": "aiv",
      "ring_id": 1,
      "start_time_us": 68.68,
      "end_time_us": 70.42,
      "duration_us": 1.74,
      "dispatch_time_us": 68.24,
      "finish_time_us": 71.2
    }
  ]
}
```

Dependency edges come from `deps.json` (dep_gen replay) at post-process time —
not from the perf JSON. See [`swimlane_converter --deps-json`](#swimlane_converter).

Top-level layout depends on `l2_swimlane_level`:

- All levels: `l2_swimlane_level`, `tasks[]` (per-task fields above).
- `>= 3`: also `aicpu_scheduler_phases[]` (per-thread phase records:
  scan / complete / dispatch / idle) and `core_to_thread[]` (core_id →
  scheduler thread index).
- `>= 4`: also `aicpu_orchestrator_phases[]` (per-task orchestrator
  phase records).

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

### Use swimlane_converter when you need

- A detailed timeline execution view
- To analyze task scheduling across different cores
- To see precise execution times and intervals
- Task execution statistics
- Professional performance analysis and optimization

### Use deps_viewer when you need

- A structural view of task dependencies (who feeds whom)
- Fast grep-friendly inspection via the default text output
- A single-file HTML you can open offline, drag-pan / wheel-zoom in any
  browser when you want a visual layout
- Optional per-task tensor rows and arg-port routing in HTML via
  `--show-tensor-info`
- A graph that survives without an associated timing run (deps.json is
  produced by structural replay, not by hardware profiling)

### Recommended Workflow

```bash
# 1. Run the test to produce both timing + structural data
pytest tests/st/... --enable-l2-swimlane --enable-dep-gen

# 2. Perfetto timeline (automatic via SceneTest)
# -> outputs/<case>_<ts>/merged_swimlane.json
#    open at https://ui.perfetto.dev/

# 3. Structural dependency graph (manual, default text output)
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json
# -> outputs/<case>_<ts>/deps_viewer.txt

# 4. Same graph as HTML
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json \
    --format html -o outputs/<case>_<ts>/deps_viewer.html

```

For batch-run hardware regression, see the dev-only script
[`tools/benchmark_rounds.sh`](../../tools/benchmark_rounds.sh).

---

## Troubleshooting

### Error: cannot find l2_swimlane_records_*.json file

- Make sure the test was run with the `--enable-l2-swimlane` flag
- Check that the outputs/ directory exists and contains profiling data

### Warning: Kernel entry missing 'func_id' or 'name'

- Check the kernel_config.py file format
- Make sure every KERNELS entry has a 'func_id' and 'name' field

### Error: Unsupported l2_swimlane_level

- The tools accept l2_swimlane_level 1–4 (the integer captured at runtime
  via `--enable-l2-swimlane <N>`)
- Regenerate the profiling data with a supported level

### Error: Perf JSON missing required fields for scheduler overhead analysis

- This error means the input `l2_swimlane_records_*.json` lacks fields required by the deep-dive analysis (typically `dispatch_time_us` / `finish_time_us`)
- The basic conversion in `swimlane_converter` can still succeed, but the deep-dive will be skipped or fail
- Remediation:
  1. Re-run with `--enable-l2-swimlane` to produce a new `outputs/*/l2_swimlane_records.json`
  2. Re-run `swimlane_converter` or `sched_overhead_analysis`
  3. Verify that each task in the JSON contains `dispatch_time_us` and `finish_time_us`

### `deps_viewer` complains that Graphviz `dot` is not on PATH

- This only affects `--format html`
- Install graphviz: `brew install graphviz` (macOS) or `apt install graphviz` (Debian/Ubuntu)
- Verify with `which dot`; should print a path
- Use a different layout engine with `--engine sfdp` for very large graphs

---

## Output File Reference

| File | Tool | Purpose | Format |
| ---- | ---- | ------- | ------ |
| `l2_swimlane_records_*.json` | Runtime | Raw timing profiling data | JSON |
| `merged_swimlane_*.json` | swimlane_converter | Perfetto visualization | Chrome Trace Event JSON |
| `deps.json` | Runtime (dep_gen replay) | Structural task dependency graph + per-edge tensor info | JSON |
| `deps_viewer.txt` | deps_viewer | Grep-friendly dependency graph view | Plain text |
| `deps_viewer.html` | deps_viewer | Pan/zoom dependency graph viewer | HTML (self-contained) |

---

## Related Resources

- [Perfetto Trace Viewer](https://ui.perfetto.dev/)
- [Graphviz documentation](https://graphviz.org/documentation/)
