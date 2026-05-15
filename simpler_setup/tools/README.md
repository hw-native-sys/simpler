# Profiling & Debug Tools (shipped in the wheel)

End-user CLIs for analyzing PTO Runtime profiling data and tensor dumps.
All are invokable as Python modules once the `simpler` wheel is installed —
no repo checkout required.

> Dev-only scripts (`benchmark_rounds.sh`, `verify_packaging.sh`) live in the
> repo-level [`tools/`](../../tools/) directory and are **not** shipped.

## Tool list

- **[swimlane_converter](#swimlane_converter)** — perf JSON → Chrome Trace Event (Perfetto)
- **[sched_overhead_analysis](#sched_overhead_analysis)** — scheduler overhead / Tail OH breakdown
- **[deps_to_graph](#deps_to_graph)** — `deps.json` (dep_gen) → pan/zoom HTML dependency graph
- **[dump_viewer](#dump_viewer)** — inspect / export tensor dumps (see [docs/tensor-dump.md](../../docs/dfx/tensor-dump.md) for full workflow)

Auto-detection paths (`outputs/*/l2_perf_records.json`, `outputs/*/tensor_dump/`)
are resolved relative to the **current working directory** — run these from the
directory that holds your `outputs/`. Each test case writes into its own
`outputs/<case>_<ts>/` directory; the tools auto-pick the latest by mtime.

---

## swimlane_converter

Convert performance profiling JSON files into Chrome Trace Event format for visualization in Perfetto.

### Overview

Converts PTO Runtime profiling data (`l2_perf_records_*.json`) into the format used by the Perfetto trace viewer (<https://ui.perfetto.dev/>). It also produces a task execution statistics summary grouped by function and a scheduler overhead deep-dive report (the same one `sched_overhead_analysis` emits).

### Basic Usage

```bash
# Auto-detect the latest profiling file under ./outputs/
python -m simpler_setup.tools.swimlane_converter

# Specify an input file
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_perf_records.json

# Specify an output file
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_perf_records.json -o custom_output.json

# Load function name mapping from kernel_config.py
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_perf_records.json \
    -k examples/host_build_graph/paged_attention/kernels/kernel_config.py

# Verbose mode (for debugging)
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_perf_records.json -v
```

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `input` | | Input JSON file (l2_perf_records_*.json). If omitted, the latest file in outputs/ is used |
| `--output` | `-o` | Output JSON file (default: outputs/merged_swimlane_`<timestamp>`.json) |
| `--kernel-config` | `-k` | Path to kernel_config.py, used for function name mapping |
| `--func-names` | | Path to func_id_names_*.json (SceneTest format) for function name mapping |
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

1. Auto-detect the latest `l2_perf_records_*.json` in outputs/
2. Load function names from the kernel_config.py specified via `-k`
3. Produce `merged_swimlane_*.json` for visualization
4. Print the task statistics and scheduler overhead deep-dive report to the console

---

## sched_overhead_analysis

Analyze AICPU scheduler overhead and quantitatively decompose the sources of Tail OH (the latency between task completion and scheduler acknowledgement).

### Overview

`sched_overhead_analysis` reads two artifacts produced by the runtime:

1. **Perf profiling data** (`l2_perf_records_*.json`, v2): per-task Exec / Head OH / Tail OH time breakdowns plus `aicpu_scheduler_phases` — per-thread, per-loop-iteration phase records carrying scan / complete / dispatch / idle timings and per-emit pop_hit / pop_miss deltas.
2. **`deps.json`** (optional, dep_gen replay output): structural task DAG. When colocated with the perf JSON, Part 2 prints per-thread fanout / fanin aggregates derived from it.

### Basic Usage

```bash
# Auto-pick the latest perf data under ./outputs/ (deps.json sibling is auto-detected)
python -m simpler_setup.tools.sched_overhead_analysis

# Specify the perf JSON explicitly
python -m simpler_setup.tools.sched_overhead_analysis \
    --l2-perf-records-json outputs/<case>_<ts>/l2_perf_records.json

# Override the deps.json location
python -m simpler_setup.tools.sched_overhead_analysis \
    --l2-perf-records-json outputs/<case>_<ts>/l2_perf_records.json \
    --deps-json outputs/<case>_<ts>/deps.json
```

### Command-Line Options

| Option | Description |
| ------ | ----------- |
| `--l2-perf-records-json` | Path to the l2_perf_records_*.json file. If omitted, the latest file in outputs/ is auto-selected |
| `--deps-json` | Path to deps.json (dep_gen replay output) for fanout / fanin aggregates. Defaults to the deps.json sibling of the perf JSON. |

### Outputs

Output is emitted in three parts:

- **Part 1: Per-task time breakdown** — Exec / Head OH / Tail OH percentages of Latency
- **Part 2: AICPU scheduler loop breakdown** — per-scheduler-thread loop statistics, per-phase (scan / complete / dispatch / idle) time ratios, pop_hit / pop_miss totals, and (when deps.json is available) per-thread fanout / fanin aggregates
- **Part 3: Tail OH distribution & cause analysis** — Tail OH quantile distribution (P10–P99), correlation between scheduler loop iteration time and Tail OH, and data-driven insights into the dominant phase

The perf JSON must be a v2 capture with non-empty `aicpu_scheduler_phases` (rerun the case with `--enable-l2-swimlane` if the tool reports the field is missing).

---

## deps_to_graph

Render the dep_gen `deps.json` task graph as a self-contained pan/zoom HTML
page (Graphviz SVG + inline vanilla-JS drag-pan + wheel-zoom). Pairs naturally
with [`swimlane_converter`](#swimlane_converter): swimlane is the timing view,
this is the structural view.

### Overview

`deps_to_graph` reads `deps.json` produced by the dep_gen replay (see
[docs/dfx/dep_gen.md](../../docs/dfx/dep_gen.md)) and emits an HTML file
viewable in any modern browser, no internet needed. Two modes:

- **Default** — every task is a shape-coded node (AIC blue box / AIV orange
  ellipse / mix green diamond / alloc dashed grey), edges are bare arrows.
  Best for "is task X reachable from task Y?" topology questions on dense
  graphs.
- **`--show-tensor-info`** — every task is an HTML-table node with input
  rows on top, identity header in the middle, output rows on the bottom;
  each slot row shows `arg<i> <TYPE> <Tname>:<dtype>` plus `raw:` / `shape:` /
  `offset:`. Edges route from `pred:out_<idx>` to `succ:in_<arg>` by
  matching `tensor_id`, so "which output of X feeds which input of Y" is
  visually obvious. This is the answer to issue #666's "what slice does
  this edge carry?" question.

### Basic Usage

```bash
# Auto-pick the newest deps.json under ./outputs/
python -m simpler_setup.tools.deps_to_graph

# Specific path
python -m simpler_setup.tools.deps_to_graph outputs/<case>_<ts>/deps.json

# Specify an output HTML path
python -m simpler_setup.tools.deps_to_graph outputs/<case>_<ts>/deps.json -o graph.html

# Show per-edge tensor slice info (compartments + matched ports)
python -m simpler_setup.tools.deps_to_graph outputs/<case>_<ts>/deps.json --show-tensor-info

# Force-directed layout for large graphs (>~1000 nodes)
python -m simpler_setup.tools.deps_to_graph outputs/<case>_<ts>/deps.json --engine sfdp

# Override node labels with a func_id -> name mapping
python -m simpler_setup.tools.deps_to_graph outputs/<case>_<ts>/deps.json \
    --func-names outputs/<case>_<ts>/name_map_TestPA_basic.json
```

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `input` | | Path to `deps.json` (default: newest under `./outputs/`) |
| `--output` | `-o` | Output HTML path (default: same dir as input, `deps_graph.html`) |
| `--engine` | | Graphviz layout engine: `dot` (default, hierarchical), `sfdp` (force-directed, recommended >1000 nodes), `neato`, `fdp`, `circo`, `twopi` |
| `--direction` | | Flow direction for hierarchical layouts: `LR` (default) / `TB` / `BT` / `RL`. Ignored by sfdp/neato. |
| `--func-names` | | JSON file with `callable_id_to_name` (or flat `{func_id: name}`) for node-label enrichment |
| `--show-tensor-info` | | Render each task as an HTML-table node with input/output slot compartments; route edges between matching ports. Default: off (bare topology). |

### Dependencies

Requires the Graphviz `dot` binary on PATH:

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

### Use swimlane_converter when you need

- A detailed timeline execution view
- To analyze task scheduling across different cores
- To see precise execution times and intervals
- Task execution statistics
- Professional performance analysis and optimization

### Use deps_to_graph when you need

- A structural view of task dependencies (who feeds whom)
- Per-edge tensor slice info — which `(tensor_id, offset, shape)` an edge
  carries — via `--show-tensor-info`
- A single-file HTML you can open offline, drag-pan / wheel-zoom in any
  browser
- A graph that survives without an associated timing run (deps.json is
  produced by structural replay, not by hardware profiling)

### Recommended Workflow

```bash
# 1. Run the test to produce both timing + structural data
pytest tests/st/... --enable-l2-swimlane --enable-dep-gen

# 2. Perfetto timeline (automatic via SceneTest)
# -> outputs/<case>_<ts>/merged_swimlane.json
#    open at https://ui.perfetto.dev/

# 3. Structural dependency graph (manual)
python -m simpler_setup.tools.deps_to_graph outputs/<case>_<ts>/deps.json
# -> outputs/<case>_<ts>/deps_graph.html (drag / wheel / f / r)

# 4. Same graph with per-edge tensor info
python -m simpler_setup.tools.deps_to_graph outputs/<case>_<ts>/deps.json \
    --show-tensor-info -o outputs/<case>_<ts>/deps_graph_with_tensors.html
```

For batch-run hardware regression, see the dev-only script
[`tools/benchmark_rounds.sh`](../../tools/benchmark_rounds.sh).

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
- The basic conversion in `swimlane_converter` can still succeed, but the deep-dive will be skipped or fail
- Remediation:
  1. Re-run with `--enable-l2-swimlane` to produce a new `outputs/*/l2_perf_records.json`
  2. Re-run `swimlane_converter` or `sched_overhead_analysis`
  3. Verify that each task in the JSON contains `dispatch_time_us` and `finish_time_us`

### `deps_to_graph` complains that Graphviz `dot` is not on PATH

- Install graphviz: `brew install graphviz` (macOS) or `apt install graphviz` (Debian/Ubuntu)
- Verify with `which dot`; should print a path
- Use a different layout engine with `--engine sfdp` for very large graphs

---

## Output File Reference

| File | Tool | Purpose | Format |
| ---- | ---- | ------- | ------ |
| `l2_perf_records_*.json` | Runtime | Raw timing profiling data | JSON |
| `merged_swimlane_*.json` | swimlane_converter | Perfetto visualization | Chrome Trace Event JSON |
| `deps.json` | Runtime (dep_gen replay) | Structural task dependency graph + per-edge tensor info | JSON (v2) |
| `deps_graph.html` | deps_to_graph | Pan/zoom dependency graph viewer | HTML (self-contained) |

---

## Related Resources

- [Perfetto Trace Viewer](https://ui.perfetto.dev/)
- [Graphviz documentation](https://graphviz.org/documentation/)
