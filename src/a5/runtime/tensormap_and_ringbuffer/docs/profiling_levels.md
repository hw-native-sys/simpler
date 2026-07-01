# PTO Runtime2 Profiling Levels

This document describes the profiling macro hierarchy and logging control in the PTO Runtime2 system.

## Overview

PTO Runtime2 uses a hierarchical profiling system with compile-time macros to control profiling code compilation and log output. The `enable_l2_swimlane` runtime flag (integer perf_level 0–4) controls data collection granularity (performance buffers, shared memory writes) but does NOT control log output.

## Profiling Macro Hierarchy

Defaults and dependency validation are centralized in
`src/common/task_interface/profiling_config.h`. Runtime headers include that
file before using the macros, so both a2a3 and a5 share the same default
values and compile-time checks.

```text
PTO2_PROFILING (base level, default=1)
├── PTO2_ORCH_PROFILING (orchestrator, default=0, requires PTO2_PROFILING=1)
|   └──PTO2_TENSORMAP_PROFILING (tensormap, default=0, requires PTO2_ORCH_PROFILING=1)
├── PTO2_SCHED_PROFILING (scheduler, default=0, requires PTO2_PROFILING=1)
└── --enable-l2-swimlane [PERF_LEVEL] (L2 swimlane data collection, 0-4, bare=4, requires PTO2_PROFILING=1)

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
- Stall detection and dump (triggered after the `SCHEDULER_TIMEOUT_MS` wall-clock no-progress budget)
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

- Base timing counters for the scheduler loop (`sched_complete/dispatch/idle/scan`)
- Host-side phase windows: each sched/orch thread publishes its
  start/end window via `aicpu_phase_set_window`, which the host reduces
  into the `Orch` / `Sched` `[STRACE]` markers

**What's NOT compiled:**

- Per-thread scheduler/orchestrator device-log lines (moved to Level 2 / Level 3)
- Detailed phase breakdowns
- TensorMap statistics

**Log output (additional lines vs Level 0, per normal run):**

- None on the device side. The per-thread `orch_start/orch_end/orch_cost`,
  `sched_start/sched_end/sched_cost`, and `Scheduler summary` lines are NOT
  emitted at this level — `orch_*` is gated by `PTO2_ORCH_PROFILING` (Level 3),
  `sched_*` and `Scheduler summary` by `PTO2_SCHED_PROFILING` (Level 2).
  Level 1 only feeds the host-side `Orch` / `Sched` `[STRACE]` timeline.

**LOG_INFO_V9 count (normal run):**

- `0` (device-side profiling logs). The timeline is delivered host-side via the
  phase windows, not through per-thread device logs.

**Note:**

- The host-side `[STRACE]` phase windows are controlled by compile-time macro
  `PTO2_PROFILING`, not by `enable_l2_swimlane`.
- `enable_l2_swimlane` only controls shared-memory data collection / swimlane export.

---

### Level 2: Scheduler Detailed Profiling (PTO2_SCHED_PROFILING=1)

**Requires:** `PTO2_PROFILING=1`

**What's compiled:**

- All Level 1 features
- Detailed scheduler phase counters
- Phase-specific statistics (complete, scan, dispatch, idle)
- Hit rate tracking (complete poll, ready queue pop)

**Log output (per scheduler thread, normal run):** the `sched_start/sched_end/
sched_cost` line, the full phase breakdown, and the `Scheduler summary` line
(all gated by `PTO2_SCHED_PROFILING`). The `Scheduler summary` line first
appears at this level — it is not emitted at Level 1.

**Scheduler output:**

```text
Thread X: sched_start=XXX sched_end=XXX sched_cost=XXXus
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
stats live in `aicpu_scheduler_phases[]` (in `l2_swimlane_records.json`
captured at l2_swimlane_level >= 3) and `deps.json`; consume them via
`simpler_setup/tools/sched_overhead_analysis.py`.

---

### Level 3: Orchestrator Detailed Profiling (PTO2_ORCH_PROFILING=1)

**Requires:** `PTO2_PROFILING=1`

**What's compiled:**

- All Level 1 features
- Detailed orchestrator phase counters
- Per-phase cycle tracking
- Atomic operation counters
- Wait time tracking

**Log output (per orchestrator thread, normal run):** the orchestrator phase
breakdown, followed by the `orch_start/orch_end/orch_cost` line and the
`PTO2 total submitted tasks` line — all gated by `PTO2_ORCH_PROFILING`. This
level adds orchestrator-side logs only; the scheduler side is unchanged from
Level 1 (add `PTO2_SCHED_PROFILING` / Level 2 for scheduler detail).

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
Thread X: orch_start=XXX orch_end=XXX orch_cost=XXXus
PTO2 total submitted tasks = XXX, already executed XXX tasks
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

**Log output (per orchestrator thread, normal run):** all Level 3 orchestrator
output plus the 4-line TensorMap lookup stats block below (gated by
`PTO2_TENSORMAP_PROFILING`, nested inside `PTO2_ORCH_PROFILING`).

**TensorMap output:**

```text
Thread X: === TensorMap Lookup Stats ===
Thread X:   lookups        : XXX, inserts: XXX
Thread X:   chain walked   : total=XXX, avg=X.X, max=X
Thread X:   overlap checks : XXX, hits=XXX (XX.X%)
```

---

## Runtime Flag: enable_l2_swimlane (perf_level)

`--enable-l2-swimlane` accepts an integer perf_level (0–4). Transport
mirrors the PMU pattern — two independent channels (one binary, one int):

- **Binary on/off** — `KernelArgs::enable_profiling_flag` bit1
  (`PROFILING_FLAG_L2_SWIMLANE`). Set by the host whenever level > 0; read
  by AICore (which only needs on/off to decide whether to write timing) and
  by AICPU kernel entry via `set_l2_swimlane_enabled(bool)`.
- **Granular level (0–4)** — `L2SwimlaneDataHeader::l2_swimlane_level`
  (shared memory). Host writes it in `L2SwimlaneCollector::initialize`; AICPU
  promotes it from the header in `l2_swimlane_aicpu_init` and exposes it via
  `get_l2_swimlane_level()` (typed `L2SwimlaneLevel`) for
  `>= AICPU_TIMING / SCHED_PHASES / ORCH_PHASES` gates.

On sim, the binary on/off travels via the dlsym'd `set_l2_swimlane_enabled`
entry point; the granular level still goes through the shared-memory
header just like on onboard.

| Level | Collects |
| ----- | -------- |
| 0 | Nothing (disabled) |
| 1 | AICore timing only (start/end/task_id/func_id/core_type) |
| 2 | + dispatch_time, finish_time |
| 3 | + Scheduler phases (`SCHED_*`) |
| 4 | + Orchestrator phases (full) |

Bare `--enable-l2-swimlane` = level 4 (backward compatible).

### Level gating in AICPU code

Use the strongly-typed `L2SwimlaneLevel` enum so each gate names the
content it depends on instead of relying on magic numbers:

```cpp
// Any level > 0: AICPU task record buffer init / flush.
// Cheap binary check, available immediately after kernel entry.
if (is_l2_swimlane_enabled()) { ... }

// AICPU dispatch/finish timestamps.
// Granular checks below require l2_swimlane_aicpu_init to have already run
// (so the level has been promoted from the shared-memory header).
if (get_l2_swimlane_level() >= L2SwimlaneLevel::AICPU_TIMING) { ... }

// Scheduler main-loop phase records (SCHED_*)
if (get_l2_swimlane_level() >= L2SwimlaneLevel::SCHED_PHASES) { ... }

// Orchestrator phase records
if (get_l2_swimlane_level() >= L2SwimlaneLevel::ORCH_PHASES) { ... }
```

`L2SwimlaneLevel` is defined in `common/l2_swimlane_profiling.h` with
underlying type `uint32_t` (matches the `L2SwimlaneDataHeader::l2_swimlane_level`
shared-memory field and mirrors `PmuEventType : uint32_t`):

| Enumerator | Underlying value |
| ---------- | ---------------- |
| `DISABLED` | 0 |
| `AICORE_TIMING` | 1 |
| `AICPU_TIMING` | 2 |
| `SCHED_PHASES` | 3 |
| `ORCH_PHASES` | 4 |

### When enable_l2_swimlane=0

- No performance data collection
- No shared memory writes
- Logs still print (controlled by macros only)

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

Pass compile definitions through the build command or CI `CXXFLAGS`.
This overrides the defaults in `profiling_config.h` without changing source.

```bash
# Example: disable all profiling code
CXXFLAGS="-DPTO2_PROFILING=0" pip install --no-build-isolation -e .

# Example: enable orchestrator and tensormap profiling
CXXFLAGS="-DPTO2_ORCH_PROFILING=1 -DPTO2_TENSORMAP_PROFILING=1" \
    pip install --no-build-isolation -e .
```

### In source code (before including headers)

Source-level overrides are only for local experiments. They must appear before
any header includes `profiling_config.h`; do not add duplicated fallback
definitions to runtime headers.

```cpp
#define PTO2_PROFILING 1
#define PTO2_ORCH_PROFILING 1
#include "pto_runtime2_types.h"
```

---

## Log Output Summary

> Example: `paged_attention` on Ascend hardware, 2 sched threads + 2 orch threads, normal run (no stall/timeout).

| Level | Macro Settings | LOG_INFO_V9 Count | Description |
| ----- | -------------- | ----------------- | ----------- |
| 0 | `PTO2_PROFILING=0` | 0 | No timing output |
| 1 | `PTO2_PROFILING=1` | 0 | Host-side `Orch`/`Sched` `[STRACE]` windows only; no device logs |
| 2 | `+PTO2_SCHED_PROFILING=1` | per sched thread | `sched_start` + phase breakdown + `Scheduler summary` |
| 3 | `+PTO2_ORCH_PROFILING=1` | per orch thread | Orchestrator phase breakdown + `orch_start` + `PTO2 total` |
| 4 | `+PTO2_TENSORMAP_PROFILING=1` | per orch thread | + TensorMap lookup stats (4 lines) |

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

- Macro defaults and validation: `src/common/task_interface/profiling_config.h`
- Scheduler profiling: `src/a5/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp` and `scheduler_cold_path.cpp`
- Orchestrator profiling: `src/a5/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- TensorMap profiling: `src/a5/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`

---

## Performance Impact

### Compilation overhead

- Level 0: No overhead
- Level 1: Minimal (counter increments, basic arithmetic)
- Level 2-4: Low to moderate (additional counters, cycle measurements)

### Runtime overhead

- Logging: Negligible (device logs are asynchronous)
- Data collection (`enable_l2_swimlane>0`): Low to moderate
  - Performance buffer writes
  - Shared memory updates
  - Per-task timing measurements

### Recommendation

- Use Level 0 for production
- Use Level 1-2 for performance monitoring
- Use Level 3-4 for detailed performance analysis only
