---
name: dfx-analyze
description: Analyze an onboard run's performance/scheduling/dependency/dump data using simpler's BUILT-IN DFX tools (simpler_setup.tools.*) instead of hand-rolling instrumentation. Use AFTER an onboard run when you need per-run device timing (Total/Orch/Sched), AICPU scheduler-overhead / Tail-OH breakdown, the task dependency graph, scope ring-fill peaks, or to inspect args dumps. These are simpler's own tools (shipped in the wheel), distinct from any cross-repo workload. Reach for this before writing custom timing/logging into the runtime.
---

# Analyze DFX data (simpler's own tools)

simpler already ships end-user analysis CLIs under `simpler_setup.tools` —
**use them; do not re-invent timing/instrumentation in the runtime.** Canonical
reference (tool flags, examples, output paths): `simpler_setup/tools/README.md`.
Per-DFX docs: `docs/dfx/` (`l2-timing.md`, `sched-overhead-model.md`,
`l2-swimlane-profiling.md`, `scope-stats.md`, `dep_gen.md`, `args-dump.md`).

## Pick the tool by question

| You want… | Tool | Needs |
| --------- | ---- | ----- |
| Per-run **Host / Device / Effective / Orch / Sched** timing | `strace_timing --rounds-table` | nothing extra — `[STRACE]` markers are on stderr (`SIMPLER_PROFILING`, compile-time default on, **NOT** gated by swimlane) |
| AICPU **scheduler overhead / Tail-OH / critical-path** breakdown | `sched_overhead_analysis` | a `--enable-l2-swimlane` (level≥3) run + `--enable-dep-gen` run |
| Swimlane → **Perfetto** Chrome trace | `swimlane_converter` | `--enable-l2-swimlane` run (`--overhead` track needs deps.json too) |
| Task **dependency graph** (text / HTML) | `deps_viewer` | `--enable-dep-gen` run → `deps.json` |
| **Per-scope ring-fill peaks** (task_window / heap / tensormap) | `scope_stats_plot` | `--enable-scope-stats` run → `scope_stats.jsonl` |
| Inspect / export **args dumps** | `dump_viewer` | `--enable-dump-tensor` run → `args_dump/` |

## First reflex: Host/Device/Orch/Sched needs nothing extra

To answer "where did the time go / is this AICPU-orchestration bound", you do
**not** need swimlane or custom logging — just tee the run's stderr, then:

```bash
python test_*.py -p <platform> -d <device> --rounds N --skip-golden > run.log 2>&1
python -m simpler_setup.tools.strace_timing run.log --rounds-table
# prints per-round Host / Device / Effective / Orch / Sched (us);
# Orch≈Sched≈Effective ⇒ AICPU-bound. (Effective = orch∪sched window.)
```

For the per-thread `loops`/`tasks_scheduled` deep-dive (not in the markers),
rebuild with `PTO2_SCHED_PROFILING=1` and read the device log directly.

## Where the inputs are written

DFX artifacts land in the run's output dir with fixed filenames:

- simpler scene tests (`tests/st`): `outputs/<case>_<ts>/` (the tools auto-pick
  the latest by mtime when run from the dir holding `outputs/`).
- JIT examples / pypto-lib: `build_output/_jit_*/dfx_outputs/`.

## Don't

- ❌ Hand-roll per-stage / submit-drain / per-scope timing in the runtime to get
  numbers these tools already produce. If a tool is missing a metric, extend the
  tool, not the hot path (and never log on AICPU hot paths — see
  `codestyle.md` rule 7).
