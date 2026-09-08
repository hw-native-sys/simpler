# DFX and Profiling

Every profiling / diagnostics reference lives in this directory. Start with the
framework doc if you are adding a collector, or jump straight to the per-feature
doc for the data you want.

Analysis CLIs that consume these outputs are documented in
[simpler_setup/tools/README.md](../../simpler_setup/tools/README.md).

## Framework and conventions

| Document | What it covers |
| -------- | -------------- |
| [Profiling Framework](profiling-framework.md) | Collector architecture, host/device split, how perf levels gate collection |
| [Profiling / DFX Configuration Naming Rules](profiling-config-naming.md) | Naming conventions every new DFX knob must follow |
| [Profiling Name Map](profiling-name-map.md) | `func_id` → human-readable name mapping consumed by the trace tools |
| [DFX Backpressure](backpressure-design.md) | Per-lane block-on-contention across collectors |
| [DFX Buffer Capacity Audit (a2a3)](dfx-buffer-capacity-audit.md) | Per-buffer capacity accounting and overflow behavior |

## Timing and scheduling

| Document | What it covers |
| -------- | -------------- |
| [L2 Timing](l2-timing.md) | `host_wall` / `device_wall` / Effective / Orch / Sched breakdown from `[STRACE]` |
| [Host runtime trace markers](host-trace.md) | The `[STRACE]` marker set emitted by the host runtime, and `simpler.trace` for emitting your own |
| [The host_build_graph bind phases](hbg-bind-phases.md) | What the bind segments are, the switches that expose them, and how to compare two branches on the qwen and dsv4 decode cases |
| [Device-side phase timing](device-phases.md) | Fixed AICPU phases and the variable-phase plan |
| [Scheduler-Overhead Model](sched-overhead-model.md) | Is the scheduler the bottleneck, or starved — the model behind `--overhead` |

## Per-task traces

| Document | What it covers |
| -------- | -------------- |
| [Chip Swimlane Profiling](chip-swimlane-profiling.md) | Per-task timing and scheduler phases across cores |
| [Core Swimlane Profiling](core-swimlane-profiling.md) | Intra-core AICore pipeline trace for a single task |
| [PMU Profiling](pmu-profiling.md) | Per-task AICore hardware counters |

## Data and dependencies

| Document | What it covers |
| -------- | -------------- |
| [Args Dump](args-dump.md) | Per-task argument capture, and replaying a task from it |
| [dep_gen](dep-gen.md) | Complete per-submit dependency graph, tensor-annotated |
| [Scope Stats](scope-stats.md) | Per-scope resource-usage peaks (ring fill, dep pool) |

## Related, outside this directory

| Document | What it covers |
| -------- | -------------- |
| [Log System](../logging.md) | Log levels and sinks — the host/device logging path, distinct from profiling collectors |
| [troubleshooting/](../troubleshooting/README.md) | Device error codes and timeout defaults, for when a run fails rather than runs slowly |
