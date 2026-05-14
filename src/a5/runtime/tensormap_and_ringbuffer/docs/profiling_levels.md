# PTO Runtime2 Profiling Levels

This document describes the profiling macro hierarchy and logging control in the PTO Runtime2 system.

## Overview

PTO Runtime2 uses a hierarchical profiling system with compile-time macros to control profiling code compilation and log output. The `enable_l2_swimlane` runtime flag controls data collection (performance buffers, shared memory writes) but does NOT control log output.

### CI coverage

The default CI build leaves `PTO2_PROFILING=1` (base) and the three sub-flags (`PTO2_ORCH_PROFILING`, `PTO2_SCHED_PROFILING`, `PTO2_TENSORMAP_PROFILING`) at `0`. The `profiling-flags-smoke` job in `.github/workflows/ci.yml` exercises the non-default combinations and runs the smallest full-pipeline example (`examples/<arch>/tensormap_and_ringbuffer/vector_example/`) against the rebuilt binaries:

| Combo | `CXX` defines |
| ----- | ------------- |
| `pto2-off` | `-DPTO2_PROFILING=0` |
| `orch` | `-DPTO2_ORCH_PROFILING=1` |
| `orch-tensormap` | `-DPTO2_ORCH_PROFILING=1 -DPTO2_TENSORMAP_PROFILING=1` |
| `sched` | `-DPTO2_SCHED_PROFILING=1` |
| `orch-sched` | `-DPTO2_ORCH_PROFILING=1 -DPTO2_SCHED_PROFILING=1` |
| `all-on` | `-DPTO2_ORCH_PROFILING=1 -DPTO2_SCHED_PROFILING=1 -DPTO2_TENSORMAP_PROFILING=1` |

Each combo runs sequentially on both `a2a3sim` and `a5sim` inside a single CI job (12 iterations total in one bash loop, not a matrix — apt/python/pip setup is paid once instead of 12 times). Failures across iterations are accumulated and reported together at the end. Compile failures, format-string mismatches, runtime crashes, or output-shape regressions in any individually gated code path show up there with the failing leg attributing the specific switch.

## Profiling Macro Hierarchy

```text
PTO2_PROFILING (base level, default=1)
├── PTO2_ORCH_PROFILING (orchestrator, default=0, requires PTO2_PROFILING=1)
|   └──PTO2_TENSORMAP_PROFILING (tensormap, default=0, requires PTO2_ORCH_PROFILING=1)
├── PTO2_SCHED_PROFILING (scheduler, default=0, requires PTO2_PROFILING=1)
└── --enable-l2-swimlane (Dump profiling merged swimlane json file for visualization, requires PTO2_PROFILING=1)

```

### Compile-Time Validation

Each sub-level macro requires `PTO2_PROFILING=1`:

```cpp
#if PTO2_ORCH_PROFILING && !PTO2_PROFILING
#error "PTO2_ORCH_PROFILING requires PTO2_PROFILING=1"
#endif

#if PTO2_SCHED_PROFILING && !PTO2_PROFILING
#error "PTO2_SCHED_PROFILING requires PTO2_PROFILING=1"
#endif

#if PTO2_TENSORMAP_PROFILING && !PTO2_ORCH_PROFILING
#error "PTO2_TENSORMAP_PROFILING requires PTO2_ORCH_PROFILING=1"
#endif
```

## Profiling Levels

### Level 0: No Profiling (PTO2_PROFILING=0)

**What's compiled:**

- Debug/diagnostic logs (always present)
- Progress tracking (`PTO2 progress: completed=...`)
- Stall detection and dump (triggered only after `MAX_IDLE_ITERATIONS` idle loops)
- Deadlock/livelock detection (`diagnose_stuck_state`, called on stall)

**What's NOT compiled:**

- All `CYCLE_COUNT_*` timing counters (`sched_*_cycle`, orchestrator cost counters)
- Scheduler/Orchestrator profiling summary logs guarded by `#if PTO2_PROFILING`
- Performance data collection paths (`enable_l2_swimlane` runtime flag becomes ineffective because profiling code is not compiled)

**Log output (normal run, no stall):**

- No `sched_start/sched_end/sched_cost` timestamps
- No `orch_start/orch_end/orch_cost` timestamps
- No `Scheduler summary: total_time=...`
- No `PTO2 total submitted tasks` log
- `PTO2 progress: completed=... total=...` may appear (thread 0 only, at task completion milestones)

---

### Level 1: Basic Profiling (PTO2_PROFILING=1)

**What's compiled:**

- Base timing counters for scheduler loop (`sched_complete/dispatch/idle/scan`)
- Per-thread orchestration timing (`orch_start`, `orch_end`, `orch_cost`)
- Stage-level orchestration end timestamp (`orch_stage_end`, printed by last orch thread only, marks the moment all orch threads have finished and core transition is about to be requested; only when `orch_to_sched_` is true)
- PTO2 total submitted tasks count (printed by last orch thread, after orch timing line)
- Scheduler summary output (`total_time`, `loops`, `tasks_scheduled`)
- Scheduler lifetime timestamps and cost (`sched_start`, `sched_end`, `sched_cost` — captured inside `resolve_and_dispatch_pto2()`, printed before Scheduler summary)

**What's NOT compiled:**

- Detailed phase breakdowns
- TensorMap statistics

**Log output (additional lines vs Level 0, per normal run):**

- `Thread %d: orch_start=%llu orch_end=%llu orch_cost=%.3fus` — each orch thread, after orchestration fully complete
- `PTO2 total submitted tasks = %d, already executed %d tasks` — last orch thread only (×1), after orch timing line
- `Thread %d: orch_stage_end=%llu` — last orch thread only (×1), only when `orch_to_sched_=true`
- `Thread %d: sched_start=%llu sched_end=%llu sched_cost=%.3fus` — each sched thread, printed before Scheduler summary
- `Thread %d: Scheduler summary: total_time=%.3fus, loops=%llu, tasks_scheduled=%d` — each sched thread
- `Thread %d: sched_start=%llu sched_end(timeout)=%llu sched_cost=%.3fus` — timeout path only (replaces normal `sched_end`)

**LOG_INFO_V9 count (normal run):**

- `orch_to_sched_=false` (default): `N_sched*2 + N_orch*1 + 1` (orch_timing + PTO2_total + sched_timing + Scheduler_summary)
- `orch_to_sched_=true` (`PTO2_ORCH_TO_SCHED=1`): adds 1 (`orch_stage_end`)

> See the table at the end for concrete counts based on the `paged_attention` example.

**Example log output — `orch_to_sched_=false`** (from `paged_attention`, device 10):

```text
Thread 2: orch_start=48214752948321 orch_end=48214752959379 orch_cost=230.000us
Thread 3: orch_start=48214752948316 orch_end=48214752961505 orch_cost=275.000us
PTO2 total submitted tasks = 13, already executed 13 tasks
Thread 1: sched_start=48214752948235 sched_end=48214752962379 sched_cost=295.000us
Thread 1: Scheduler summary: total_time=159.560us, loops=3782, tasks_scheduled=6
Thread 0: sched_start=48214752948200 sched_end=48214752963571 sched_cost=320.000us
Thread 0: Scheduler summary: total_time=183.180us, loops=4611, tasks_scheduled=7
```

**Example log output — `orch_to_sched_=true`** (`PTO2_ORCH_TO_SCHED=1`, from `paged_attention`, device 11):

```text
Thread 3: orch_stage_end=48236915058307
Thread 3: orch_start=48236915044001 orch_end=48236915058781 orch_cost=308.000us
Thread 2: orch_start=48236915044003 orch_end=48236915058782 orch_cost=308.000us
PTO2 total submitted tasks = 13, already executed 13 tasks
Thread 0: sched_start=48236915043911 sched_end=48236915059191 sched_cost=318.000us
Thread 0: Scheduler summary: total_time=187.920us, loops=4561, tasks_scheduled=4
Thread 1: sched_start=48236915043947 sched_end=48236915061881 sched_cost=372.000us
Thread 1: Scheduler summary: total_time=168.620us, loops=3880, tasks_scheduled=9
```

> With `orch_to_sched_=true`, orch threads transition to schedulers after orchestration. They print `orch_end` but do NOT print `Scheduler summary` or `sched_end` (they have no cores assigned at shutdown time).

**Note:**

- All logs above are controlled by compile-time macro `PTO2_PROFILING`, not by `enable_l2_swimlane`.
- `enable_l2_swimlane` only controls shared-memory data collection / swimlane export.
- Enable `orch_to_sched_` via environment variable: `PTO2_ORCH_TO_SCHED=1`.

---

### Level 2: Scheduler Detailed Profiling (PTO2_SCHED_PROFILING=1)

**Requires:** `PTO2_PROFILING=1`

**What's compiled:**

- All Level 1 features
- Detailed scheduler phase counters
- Phase-specific statistics (complete, scan, dispatch, idle)
- Hit rate tracking (complete poll, ready queue pop)

**Log output:** 18 LOG_INFO_V9 logs (11 debug + 2 basic + 7 scheduler detailed - 2 replaced)

- Replaces scheduler summary with detailed breakdown

**Scheduler output:**

```text
Thread X: === Scheduler Phase Breakdown: total=XXXus, XXX tasks ===
Thread X:   complete       : XXXus (XX.X%)
Thread X:     poll         : XXXus (XX.X%)  hit=XXX, miss=XXX, hit_rate=XX.X%
Thread X:     otc_lock     : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:     otc_fanout   : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:     otc_fanin    : XXXus (XX.X%)  atomics=XXX
Thread X:     otc_self     : XXXus (XX.X%)  atomics=XXX
Thread X:     perf         : XXXus (XX.X%)
Thread X:   dispatch       : XXXus (XX.X%)
Thread X:     poll         : XXXus (XX.X%)
Thread X:     pop          : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:     setup        : XXXus (XX.X%)
Thread X:   scan           : XXXus (XX.X%)
Thread X:   idle           : XXXus (XX.X%)
Thread X:   avg/complete   : XXXus
Thread X: Scheduler summary: total_time=XXXus, loops=XXX, tasks_scheduled=XXX
```

Per-thread fanout / fanin edge counts and ready-queue pop hit / miss
stats live in the v2 JSON `aicpu_scheduler_phases[]` and `deps.json`;
consume them via `simpler_setup/tools/sched_overhead_analysis.py`.

---

### Level 3: Orchestrator Detailed Profiling (PTO2_ORCH_PROFILING=1)

**Requires:** `PTO2_PROFILING=1`

**What's compiled:**

- All Level 1 features
- Detailed orchestrator phase counters
- Per-phase cycle tracking
- Atomic operation counters
- Wait time tracking

**Log output:** 30 LOG_INFO_V9 logs (11 debug + 2 basic + 1 scheduler summary + 17 orchestrator detailed - 1 replaced)

- Replaces basic orchestration completion with detailed breakdown

**Orchestrator output:**

```text
Thread X: === Orchestrator Profiling: XXX tasks, total=XXXus ===
Thread X:   sync_tensormap : XXXus (XX.X%)
Thread X:   task_ring_alloc: XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   param_copy     : XXXus (XX.X%)  atomics=XXX
Thread X:   lookup+dep     : XXXus (XX.X%)
Thread X:   heap_alloc     : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   tensormap_ins  : XXXus (XX.X%)
Thread X:   fanin+ready    : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   finalize+SM    : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   scope_end      : XXXus  atomics=XXX
Thread X:   avg/task       : XXXus
```

**Note:** Orchestrator logs always print when `PTO2_ORCH_PROFILING=1`, regardless of `enable_l2_swimlane` flag.

---

### Level 4: TensorMap Profiling (PTO2_TENSORMAP_PROFILING=1)

**Requires:** `PTO2_PROFILING=1` AND `PTO2_ORCH_PROFILING=1`

**What's compiled:**

- All Level 3 features
- TensorMap lookup statistics
- Hash chain walk tracking
- Overlap check counters

**Log output:** 34 LOG_INFO_V9 logs (30 from Level 3 + 4 tensormap)

**TensorMap output:**

```text
Thread X: === TensorMap Lookup Stats ===
Thread X:   lookups        : XXX, inserts: XXX
Thread X:   chain walked   : total=XXX, avg=X.X, max=X
Thread X:   overlap checks : XXX, hits=XXX (XX.X%)
```

---

## Runtime Flag: enable_l2_swimlane

L2 swimlane enablement is published through the handshake
`enable_profiling_flag` bitmask (bit1 = `PROFILING_FLAG_L2_SWIMLANE`).
AICPU code reads it via `is_l2_swimlane_enabled()` (set at launch time
by the platform from `kernel_args.l2_perf_data_base` + the bitmask). It
controls **data collection**, NOT log output.

### When enable_l2_swimlane=true

- Performance buffers are allocated and written
- Per-task timing data is collected
- Phase profiling data is recorded
- Orchestrator summary is written to shared memory

### When enable_l2_swimlane=false

- No performance data collection
- No shared memory writes
- Logs still print (controlled by macros only)

### Usage

```cpp
// AICPU path — read enablement from the platform accessor, not the Runtime struct.
if (is_l2_swimlane_enabled()) {
    // ... perf-collection code ...
}
```

---

## Common Profiling Configurations

### Development (minimal overhead)

```bash
# No profiling overhead
PTO2_PROFILING=0
```

### Basic Performance Monitoring

```bash
# Minimal overhead, summary logs only
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=0
PTO2_SCHED_PROFILING=0
```

### Scheduler Performance Analysis

```bash
# Detailed scheduler breakdown
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=0
PTO2_SCHED_PROFILING=1
```

### Orchestrator Performance Analysis

```bash
# Detailed orchestrator breakdown
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=1
PTO2_SCHED_PROFILING=0
```

### Full Profiling (maximum overhead)

```bash
# All profiling features enabled
PTO2_PROFILING=1
PTO2_ORCH_PROFILING=1
PTO2_SCHED_PROFILING=1
PTO2_TENSORMAP_PROFILING=1
```

---

## Setting Profiling Macros

### At compile time

```bash
# In CMakeLists.txt or build command
add_definitions(-DPTO2_PROFILING=1)
add_definitions(-DPTO2_ORCH_PROFILING=1)
```

### In source code (before including headers)

```cpp
#define PTO2_PROFILING 1
#define PTO2_ORCH_PROFILING 1
#include "pto_runtime2_types.h"
```

---

## Log Output Summary

> Example: `paged_attention` on Ascend hardware, 2 sched threads + 2 orch threads, normal run (no stall/timeout).

| Level | Macro Settings | LOG_INFO_V9 Count (`orch_to_sched_=false`) | LOG_INFO_V9 Count (`orch_to_sched_=true`) | Description |
| ----- | -------------- | ------------------------------------------ | ----------------------------------------- | ----------- |
| 0 | `PTO2_PROFILING=0` | 0 | 0 | No timing output |
| 1 | `PTO2_PROFILING=1` | 7 | 8 | Timing timestamps + scheduler summary |
| 2 | `+PTO2_SCHED_PROFILING=1` | — | — | Scheduler detailed phase breakdown |
| 3 | `+PTO2_ORCH_PROFILING=1` | — | — | Orchestrator detailed phase breakdown |
| 4 | `+PTO2_TENSORMAP_PROFILING=1` | — | — | TensorMap lookup stats |

---

## Implementation Notes

### Key Principles

1. **Macros control compilation and logging**
   - `#if PTO2_PROFILING` controls whether profiling code is compiled
   - Logs print when macro is enabled, regardless of runtime flag

2. **Runtime flag controls data collection**
   - `enable_l2_swimlane` controls performance buffer allocation
   - Controls shared memory writes for host-side export
   - Does NOT control log output

3. **Consistent behavior across components**
   - Scheduler logs: macro-controlled only
   - Orchestrator logs: macro-controlled only
   - Data collection: runtime flag controlled

### Code Locations

- Macro definitions: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_runtime2_types.h`
- Scheduler profiling: `src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp` and `scheduler_cold_path.cpp`
- Orchestrator profiling: `src/a5/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- TensorMap profiling: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`

---

## Performance Impact

### Compilation overhead

- Level 0: No overhead
- Level 1: Minimal (counter increments, basic arithmetic)
- Level 2-4: Low to moderate (additional counters, cycle measurements)

### Runtime overhead

- Logging: Negligible (device logs are asynchronous)
- Data collection (`enable_l2_swimlane=true`): Low to moderate
  - Performance buffer writes
  - Shared memory updates
  - Per-task timing measurements

### Recommendation

- Use Level 0 for production
- Use Level 1-2 for performance monitoring
- Use Level 3-4 for detailed performance analysis only
