# PMU Profiling (a2a3)

This document describes how to use a2a3 PMU profiling, what output it
produces, and the current usage limitations.

## Overview

PMU profiling collects per-task AICore hardware counter data and exports a
LuoPan-compatible CSV file on the host side.

Use `a2a3` hardware runs for meaningful PMU data. The simulation path can
exercise the PMU export flow, but does not provide real hardware counters.

## Design

### Design Goals

The PMU path is designed around four goals:

- attribute counters to individual runtime tasks instead of only whole-run
  aggregates
- keep the hot execution path lightweight so profiling does not heavily disturb
  scheduling and kernel execution
- separate counter collection from host-side export so PMU handling does not
  block device progress
- produce a stable CSV output that can be consumed directly by LuoPan

### Layered Responsibilities

The design splits PMU work across host, AICPU, and AICore:

- **Host** owns user entry, event-type selection, PMU session setup, and final
  CSV export
- **AICPU** owns PMU control and task-level sampling, because it already
  observes task dispatch/completion boundaries
- **AICore** only defines the counting window around the kernel body, without
  taking on export or formatting work

This split keeps each side focused on the part it can identify most naturally:
host is good at orchestration and file export, AICPU is good at associating
counter snapshots with runtime tasks, and AICore is best used only to bracket
actual execution.

### Data Flow

At a high level, one PMU run follows this flow:

1. The user enables PMU and selects an event type.
2. The host prepares a PMU session for the run.
3. AICPU programs the selected PMU event group before task execution starts.
4. AICore opens the PMU counting window only around each kernel execution.
5. When a task completes, AICPU reads the counters and associates them with
   that task.
6. The host asynchronously drains the collected records and writes the final
   CSV.

The key idea is that PMU data is sampled at task completion time, not as a
post-run aggregate. That makes the output suitable for per-task analysis in
LuoPan.

### Why This Architecture

This architecture is chosen for a few practical reasons:

- PMU counters are per-core resources, so task attribution has to happen at the
  point where task boundaries are visible
- AICPU already tracks when a task is dispatched and when it finishes, so it is
  the natural place to bind PMU snapshots to task metadata
- AICore should stay minimal in the PMU path; otherwise profiling overhead would
  distort the execution being measured
- host-side asynchronous export avoids turning profiling into a synchronous
  device-to-host bottleneck

In short, the design tries to maximize per-task observability while minimizing
intrusion into the normal runtime path.

## Usage

### SceneTest CLI

Enable PMU with the default event group:

```bash
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-pmu
```

or with pytest:

```bash
pytest tests/st/<case> --platform a2a3 -d 0 --enable-pmu
```

The bare flag is equivalent to:

```bash
--enable-pmu 2
```

which selects `PIPE_UTILIZATION`.

Pass an explicit event type to collect a different counter group:

```bash
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-pmu 4
```

`--rounds > 1` disables PMU collection in the test harness.

### Event Types

| Value | Event Type | Example Counters |
| ----- | ---------- | ---------------- |
| `1` | `ARITHMETIC_UTILIZATION` | cube/vector execution counters |
| `2` | `PIPE_UTILIZATION` | vector, cube, scalar, MTE busy cycles |
| `4` | `MEMORY` | UB/L1/L2/main memory requests |
| `5` | `MEMORY_L0` | L0A/L0B/L0C requests |
| `6` | `RESOURCE_CONFLICT` | bank and vector resource stalls |
| `7` | `MEMORY_UB` | UB and memory bandwidth counters |
| `8` | `L2_CACHE` | L2 cache hit/miss allocation counters |

Invalid nonzero values fall back to `PIPE_UTILIZATION`.

The `SIMPLER_PMU_EVENT_TYPE` environment variable can override the CLI event
type:

```bash
SIMPLER_PMU_EVENT_TYPE=4 \
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-pmu
```

## Output

The PMU artifact is a CSV file under `outputs/`:

```text
outputs/pmu_<YYYYMMDD>_<HHMMSS>_<mmm>.csv
```

This CSV is the file to load into LuoPan.

Common columns:

| Column | Meaning |
| ------ | ------- |
| `thread_id` | AICPU scheduler thread id |
| `core_id` | Logical AICore id in the runtime. |
| `task_id` | Runtime task id, printed as hex |
| `func_id` | Kernel function id. |
| `core_type` | `0` = AIC, `1` = AIV. |
| `pmu_total_cycles` | 64-bit PMU total cycle counter snapshot. |
| event-specific counters | Counter columns selected by the event type. |
| `event_type` | Numeric event type used for the run. |

For the default `PIPE_UTILIZATION` event type (`2`), the counter columns are:

```text
vec_busy_cycles,cube_busy_cycles,scalar_busy_cycles,mte1_busy_cycles,
mte2_busy_cycles,mte3_busy_cycles,icache_miss,icache_req
```

## Limitations

Current PMU collection assumes each logical AICore has at most one in-flight
task. The runtime's default dual-issue dispatch can preload a pending task while
another task is still running on the same core. In that mode the per-core PMU
registers can be polluted by overlapping task windows, so per-task PMU rows are
not reliable.

For PMU runs, rebuild the runtime with:

```text
PTO2_DISABLE_DUAL_ISSUE=1
```

Current limitations:

- `--enable-pmu` enables collection but does not rebuild the runtime with
  dual-issue disabled.
- `PTO2_DISABLE_DUAL_ISSUE` is compile-time state in the runtime headers.
- The limitation applies to runtimes that define and use this macro
  (`host_build_graph` and `tensormap_and_ringbuffer`).
- `a2a3sim` can validate the PMU export path, but the counter values are not
  suitable for performance analysis.

Keep PMU comparisons consistent by using the same dual-issue setting across all
runs being compared.
