# Profiling & Debug Tools (shipped in the wheel)

End-user CLIs for analyzing PTO Runtime profiling data and args dumps.
All are invokable as Python modules once the `simpler` wheel is installed —
no repo checkout required.

> Dev-only scripts (`benchmark_rounds.sh`, `verify_packaging.sh`) live in the
> repo-level [`tools/`](../../tools/) directory and are **not** shipped.

## Tool list

- **[swimlane_converter](#swimlane_converter)** — perf JSON → Chrome Trace Event (Perfetto)
- **[sched_overhead_analysis](#sched_overhead_analysis)** — scheduler overhead / Tail OH breakdown
- **[strace_timing](#strace_timing)** — per-stage `simpler_run` breakdown (host + AICPU phases) from `[STRACE]` log markers → TPOT table, per-round table (`--rounds-table`), nested tree (`--tree`), or Perfetto JSON
- **[dump_viewer](#dump_viewer)** — inspect / export args dumps (see [docs/args-dump.md](../../docs/dfx/args-dump.md) for full workflow)
- **[deps_viewer](#deps_viewer)** — `deps.json` (dep_gen) → text or pan/zoom HTML dependency graph

Auto-detection paths (`outputs/*/l2_swimlane_records.json`, `outputs/*/args_dump/`)
are resolved relative to the **current working directory** — run these from the
directory that holds your `outputs/`. Each test case writes into its own
`outputs/<case>_<ts>/` directory; the tools auto-pick the latest by mtime.

---

## swimlane_converter

Convert performance profiling JSON files into Chrome Trace Event format for visualization in Perfetto.

### Overview

Converts PTO Runtime profiling data (`l2_swimlane_records_*.json`) into the format used by the Perfetto trace viewer (<https://ui.perfetto.dev/>) and prints a per-function task-execution summary. With `--overhead` (needs `deps.json`) it also adds an **Overhead Analysis** counter group under the AICPU Scheduler track — 8 lines (`oh_{aic,aiv}_{idle,ready,overhead}` + `oh_all_overhead` / `oh_has_overhead`) you can overlay on the task bars. See [docs/dfx/sched-overhead-model.md](../../docs/dfx/sched-overhead-model.md) for the model.

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

### SPMD dependency visualization

For SPMD logical tasks (`block_num > 1` in `deps.json`), dependency
arrows anchor on representative subtask rows on physical core lanes
(not a dedicated block-level track). SPMD tasks use the minimum-`core_id`
subtask row per `core_type` as the dependency anchor; MIX-type SPMD
tasks pick the minimum separately for AIC and AIV. See
[docs/dfx/l2-swimlane-profiling.md §3.5](../../docs/dfx/l2-swimlane-profiling.md#35-dependency-arrows-from-dep_gen).

Each logical `(pred, succ)` edge emits flows for the Cartesian product
of pred/succ anchor rows (`|pred_anchors| × |succ_anchors|`), not a
per-subtask crossbar.

SPMD lane labels append `_spmd` before `(rXtY)` unless the function
name already contains `spmd` (case-insensitive), e.g.
`v_proj_spmd(r2t10)` vs `SPMD_WRITE_AIV(t0)`.

With `-v`, the converter prints
`dependency arrows anchor on min core_id subtask per core_type` when
SPMD tasks are present.

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `input` | | Input JSON file (l2_swimlane_records_*.json). If omitted, the latest file in outputs/ is used |
| `--output` | `-o` | Output JSON file (default: outputs/merged_swimlane_`<timestamp>`.json) |
| `--kernel-config` | `-k` | Path to kernel_config.py, used for function name mapping |
| `--func-names` | | Path to func_id_names_*.json (SceneTest format) for function name mapping |
| `--deps-json` | | Path to a dep_gen `deps.json` (defaults to sibling of input). Without one, no dependency arrows are drawn. |
| `--overhead` | | Add the 8-line Overhead Analysis counter group (needs `deps.json`). See [sched-overhead-model](../../docs/dfx/sched-overhead-model.md). |
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

#### 3. Scheduler Overhead Deep-Dive

`swimlane_converter` no longer runs the deep-dive inline — it needs the task DAG
(`deps.json`) from a *separate* `--enable-dep-gen` run, which can't be produced
accurately alongside the swimlane capture. Run
[`sched_overhead_analysis`](#sched_overhead_analysis) manually with both
artifacts to get the scheduler-starvation / critical-path report.

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

Answer **"is the AICPU scheduler the bottleneck, or is it starved?"** by
measuring, dependency- and MIX-aware, how much of the makespan a free core has
ready, undispatched work — vs. legitimately busy or dependency-limited. Full
model: [docs/dfx/sched-overhead-model.md](../../docs/dfx/sched-overhead-model.md).

### Overview

`sched_overhead_analysis` needs **two artifacts, captured in SEPARATE runs**
(co-running the flags perturbs timing — `dep_gen` adds per-submit overhead):

1. **Perf profiling data** (`l2_swimlane_records_*.json`, level >= 3) from a
   `--enable-l2-swimlane` run — per-task dispatch/start/end/finish +
   `aicpu_scheduler_phases`.
2. **`deps.json`** (the task DAG) from a separate `--enable-dep-gen` run. It
   drives `ready(C) = max(producer.end)`, which is what separates scheduler
   bubbles from dependency stalls. **Required** — the tool errors without it.

### Basic Usage

```bash
# Capture once (two separate runs of the same case):
pytest <case> --platform a2a3 --device N --enable-dep-gen        # -> deps.json
pytest <case> --platform a2a3 --device N --enable-l2-swimlane    # -> l2_swimlane_records.json (clean timing)

# Analyze:
python -m simpler_setup.tools.sched_overhead_analysis \
    --l2-swimlane-records-json outputs/<swimlane case>/l2_swimlane_records.json \
    --deps-json outputs/<dep_gen case>/deps.json
```

> `deps.json` is topology-invariant — capture it once per graph and reuse it for
> any number of swimlane runs. For Host / Device / Effective / Orch / Sched timing
> from a plain run, use [`strace_timing --rounds-table`](#strace_timing) instead.

### Command-Line Options

| Option | Description |
| ------ | ----------- |
| `--l2-swimlane-records-json` | Path to the l2_swimlane_records_*.json file (level >= 3). If omitted, the latest under outputs/ is auto-selected. |
| `--deps-json` | Path to deps.json from a `--enable-dep-gen` run. **Required.** Falls back to a `deps.json` sibling of the perf JSON if present. |

### Outputs

Emitted in six parts:

- **Part 1: Overhead verdict** — per-engine overhead (idle T-core *and* a ready, undispatched T-task, MIX-aware) + system `all_overhead` / `has_overhead`, all as % of makespan. An engine with no ready work is not overhead (dependency-mandated idle, not waste).
- **Part 2: aicore switch** — the pre-dispatched pickup gap (`dispatch < prev_end`), reported **per core** (min/mean/max, ~0.8 µs each), the overhead-vs-independent split, and the makespan switch bound `[min over cores, sum of per-engine minima]`.
- **Part 3 / 4: Head / Tail OH distributions** — P10–P99 + mean + total (per-task pickup and detect-latency magnitude).
- **Part 5: AICPU scheduler loop breakdown** — per-thread loops, ns/loop, complete/dispatch/idle phase ratios, pop_hit / pop_miss, fanout / fanin, + the tail-vs-loop cause analysis.
- **Part 6: Critical-path latency attribution** — along the makespan path, scheduler-injected µs vs compute µs ("scheduler adds X% to the critical path").

The perf JSON must be captured at l2_swimlane_level >= 3 so that `aicpu_scheduler_phases` is non-empty (rerun the case with `--enable-l2-swimlane` if the tool reports the field is missing).

---

## strace_timing

Per-stage breakdown of every `simpler_run()` from `[STRACE]` host-trace
markers in a log (host stderr or CANN device log). The runtime emits one
`[STRACE]` line per span on scope exit (RAII, gated on `SIMPLER_HOST_STRACE`,
`LOG_INFO_V9`), including the AICPU device-phase subdivision (`clk=dev`). See
[docs/dfx/host-trace.md](../../docs/dfx/host-trace.md) for the marker grammar.

```bash
# Per-callable TPOT table (decode = most-invoked hid bucket; prefill = once-seen)
python -m simpler_setup.tools.strace_timing path/to/log

# Per-round Host/Device/Orch/Sched table (the benchmark/--rounds N view)
python -m simpler_setup.tools.strace_timing path/to/log --rounds-table

# Indented nested span tree per callable (simpler_run → bind / runner_run →
# device_wall → preamble/config_validate/arena_wire/sm_reset/orch/sched/post_orch)
python -m simpler_setup.tools.strace_timing path/to/log --tree

# Also emit a Chrome-trace / Perfetto JSON (one named lane per invocation, with
# separate host and device(clk=dev) tracks; nested by span containment)
python -m simpler_setup.tools.strace_timing path/to/log --trace-out strace.json
```

Groups spans by `(pid, inv)`, rebuilds each invocation's tree from `depth`,
buckets by callable hash `hid`, and reports each callable's mean `simpler_run`
plus per-stage means. It reads the host-emitted `[STRACE]` lines and shows the
host stages (`bind`/`runner_run`/`validate`) alongside the AICPU phases.

`--tree` renders one nested span tree per callable; each node's duration is the
**median across every invocation** of that callable (not one invocation's
value). This matters for a callable whose invocations differ in cost — e.g.
qwen3 decode, where the pypto-serving profile warmup dispatches a tiny-KV step
(seq_len≈257, ~28 ms) before the real 3.5k-context steps (~40 ms); a
single-invocation tree would report the warmup value.

`--rounds-table` renders one row per invocation of the busiest `hid` —
**Host** always, plus **Device / Effective / Orch / Sched** when present, in the
format `tools/benchmark_rounds.sh` parses. `Effective` is the orch∪sched merged
window (`max(orch_end,sched_end) − min(orch_start,sched_start)`, the old
device-log "Total"), recomputed from the orch/sched markers' `ts`+`dur` — no
device log needed. The scene test only *emits* the markers to stderr; tee a run
to a file (`python test_*.py … --rounds N > run.log 2>&1`) and pass `run.log`
here. Because grouping is per `(pid, inv)`, this captures **L3 multi-round**
(every chip-child invocation), not just round 0.

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

# Transitive reduction: select non-redundant edges, print what was removed
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json \
    --edge-mode reduced

# Redundant-only: select the transitively-implied edges reduced would drop
python -m simpler_setup.tools.deps_viewer outputs/<case>_<ts>/deps.json \
    --edge-mode omitted
```

`--edge-mode` selects which structural `(pred, succ)` edges are visible:

- `full` (default) — every dependency edge.
- `reduced` — the minimal (transitively-reduced) edge set: every edge already
  implied by a longer path is dropped, e.g. `A->C` when `A->B->C` exists.
- `omitted` — only the redundant edges `reduced` would drop (its complement),
  for auditing exactly which dependencies are transitively covered.

`reduced` and `omitted` print the redundant edges to stdout as a
`<task> -> <task>` list, where each task uses the same label as the rendered
graph — the bare `local` counter when every task is in ring 0, or the explicit
`(ring, local)` tuple once any task lives in ring >= 1. Text output emits only
the selected edge set. HTML output keeps every edge in the Graphviz layout and
colors unselected edges like the page background, so `reduced` / `omitted`
preserve the full-graph node placement and routing while showing only the
selected edge set. When `-o` is omitted the graph is written to a mode-specific
stem (`deps_viewer_reduced.*` / `deps_viewer_omitted.*`) rather than
`deps_viewer.*` so it never clobbers a full-graph render in the same directory.
Reduction is purely structural (it ignores the per-edge tensor/arg identity)
and is skipped with a warning if the graph contains a cycle.

### Command-Line Options

| Option | Short | Description |
| ------ | ----- | ----------- |
| `input` | | Path to `deps.json` (default: newest under `./outputs/`) |
| `--output` | `-o` | Output path; default stem is `deps_viewer`, or `deps_viewer_{mode}` for `reduced` / `omitted` |
| `--format` | | Output format: `text` (default) or `html` |
| `--edge-mode` | | Select visible edges: `full`, `reduced`, or `omitted`; HTML preserves full layout. |
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

Inspect and export args captured by the runtime args-dump feature.
See [docs/args-dump.md](../../docs/dfx/args-dump.md) for the full capture workflow;
this section only documents CLI invocation.

### Basic Usage

```bash
# List all args (auto-picks latest outputs/*/args_dump dir)
python -m simpler_setup.tools.dump_viewer

# Filter by task/stage/role
python -m simpler_setup.tools.dump_viewer --task 0x0000000200000a00 --stage before --role input

# Export the current selection to txt
python -m simpler_setup.tools.dump_viewer --task 0x0000000200000a00 --stage before --role input --export

# Export a specific arg by index (always exports)
python -m simpler_setup.tools.dump_viewer outputs/<case>_<ts>/args_dump/ --index 42
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
