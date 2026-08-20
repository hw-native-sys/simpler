# How-to: profile a kernel

Diagnostics are **off by default** and are enabled per run. Turn on the one that
answers your question — each writes its artifacts under the run's output
directory, and each has an analysis CLI that reads them.

Start by asking which question you have:

| Question | Enable | Then read |
| -------- | ------ | --------- |
| Where did wall-clock go, host vs device? | nothing — `[STRACE]` markers are always emitted | [L2 Timing](../../dfx/l2-timing.md), [host trace](../../dfx/host-trace.md) |
| Which task ran when, on which core? | `--enable-chip-swimlane` | [Chip Swimlane Profiling](../../dfx/chip-swimlane-profiling.md) |
| Is the scheduler the bottleneck, or starved? | `--enable-chip-swimlane --enable-dep-gen --enable-swimlane-overhead` | [Scheduler-Overhead Model](../../dfx/sched-overhead-model.md) |
| Why is *one* task slow inside the core? | `--dump-args`, then Core Swimlane | [Core Swimlane Profiling](../../dfx/core-swimlane-profiling.md) |
| What do the AICore hardware counters say? | `--enable-pmu` | [PMU Profiling](../../dfx/pmu-profiling.md) |
| What does the dependency graph actually look like? | `--enable-dep-gen` | [dep_gen](../../dfx/dep-gen.md) |
| Am I near a ring / dep-pool capacity limit? | `--enable-scope-stats` | [Scope Stats](../../dfx/scope-stats.md) |
| What arguments did a task really receive? | `--dump-args` | [Args Dump](../../dfx/args-dump.md) |

## Enabling from pytest

```bash
# per-task timing across cores (bare flag = level 4, full detail)
pytest examples/my_example --platform a2a3 --device 4 --enable-chip-swimlane

# scheduler overhead: needs the swimlane AND a dependency graph
pytest examples/my_example --platform a2a3 --device 4 \
    --enable-chip-swimlane --enable-dep-gen --enable-swimlane-overhead

# hardware counters, and captured per-task arguments
pytest examples/my_example --platform a2a3 --device 4 --enable-pmu
pytest examples/my_example --platform a2a3 --device 4 --dump-args
```

`--enable-swimlane-overhead` requires `--enable-chip-swimlane` plus a `deps.json`;
if it is absent, re-run adding `--enable-dep-gen`.

**For single-round runs, `--enable-chip-swimlane` is L2-only.** Asking for it
on an L3 test is rejected up front — scope to a single chip with `--level 2`,
or drop the flag.

Use `--rounds N` for repeated non-diagnostic timing. All diagnostic flags above
are disabled when `N > 1`; warm up separately, then collect diagnostics with a
single round.

## Enabling from your own code

The same switches are `CallConfig` fields, so a direct-`Worker` program sets
them itself. Any enabled diagnostic requires `output_prefix` to be set —
`CallConfig::validate()` rejects the config otherwise.

```python
cfg = CallConfig()
cfg.enable_chip_swimlane = 4      # 0 = off; 1..4 select detail
cfg.enable_pmu = 1              # 0 = off; >0 selects the event type
cfg.output_prefix = "outputs/my_run"    # required once any diagnostic is on
worker.run(handle, args, cfg)
```

In a `@scene_test`, per-case knobs live under `"config"`, including
`aicpu_thread_num` and `runtime_env` sizing. HBG interprets
`ring_task_window[0]` and `ring_heap[0]` as its whole-graph task capacity and
heap size. `tensormap_and_ringbuffer` uses all slots and `ring_dep_pool`.

## Reading the results

The analysis CLIs ship in the wheel under `simpler_setup.tools`; run them with
`python -m`. Full flag documentation is in
[`simpler_setup/tools/README.md`](../../../simpler_setup/tools/README.md).

| Tool | Consumes | Gives you |
| ---- | -------- | --------- |
| `strace_timing` | the run log | host/device wall-clock breakdown, per-round table |
| `swimlane_converter` | `chip_swimlane_records_*.json` | a Perfetto trace plus a per-function task summary; `--overhead` overlays the scheduler-overhead counters |
| `sched_overhead_analysis` | swimlane + deps | whether the scheduler is the bottleneck or starved |
| `core_swimlane` | an args dump | intra-core pipeline trace for one task |
| `deps_viewer`, `critical_path` | `deps.json` | the dependency graph, and its critical path |
| `dump_viewer` | an args dump | the captured per-task arguments |
| `scope_stats_plot` | `scope_stats.jsonl` | per-scope resource peaks |

```bash
python -m simpler_setup.tools.strace_timing <run.log> --rounds-table
python -m simpler_setup.tools.swimlane_converter <chip_swimlane_records_*.json>
```

## Before optimizing anything

Check [`docs/investigations/`](../../investigations/README.md) first. It records
proposals that were measured and dropped — several plausible-sounding
optimizations on the profiling and dispatch paths are already shut down there
with numbers, and re-deriving them costs a hardware round-trip.
