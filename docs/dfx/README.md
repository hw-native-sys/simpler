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

## A5 L0 hardware capture

The basic L0 capture supports the onboard A5 `tensormap_and_ringbuffer` runtime.
Pass `--enable-l0-swimlane` to a scene test, or set
`CallConfig.enable_l0_swimlane = True`. After execution, the collector copies
the hardware GM ring into `l0_dfx_capture/l0_raw/run_<epoch>/core_*.bin` and
writes `manifest.json` under the run's output prefix.

Decode the ordered raw records and generate a Chrome Trace Event JSON file:

```bash
python -m simpler_setup.tools.l0_swimlane_converter <run-directory> \
    --decoder-library build/lib/a5/onboard/tensormap_and_ringbuffer/libhost_runtime.so
```

Open the generated `l0_swimlane_trace.json` in Perfetto or a compatible trace
viewer. `core_*.events.json` archives can also be converted without the decoder
library. Cached events use the system counter frequency and per-core AIC or AIV
frequency from the capture manifest. Rerun with `--decoder-library` after changing
a clock or replacing raw files. When raw files are present, the converter also
checks their SHA-256 against the cached events.

Capture copies data only after execution and is intended for small kernels.
Long runs can overwrite the ring and lose samples; inspect the manifest and
trace completeness metadata. A failed core does not discard the other cores:
valid streams are retained and missing cores are reported with an incomplete
capture marker. AICores keep producing samples while polling
their idle loops, even when they have no task. Hardware sleep produces only
a small amount of data. Incremental copying and HBG capture are unsupported.

## Related, outside this directory

| Document | What it covers |
| -------- | -------------- |
| [Log System](../logging.md) | Log levels and sinks — the host/device logging path, distinct from profiling collectors |
| [troubleshooting/](../troubleshooting/README.md) | Device error codes and timeout defaults, for when a run fails rather than runs slowly |
