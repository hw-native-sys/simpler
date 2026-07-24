# PTO Runtime2 Profiling Levels

This document describes the profiling macro hierarchy and logging control in the PTO Runtime2 system.

## Overview

PTO Runtime2 uses a hierarchical profiling system with compile-time macros to control profiling code compilation and log output. The `enable_l2_swimlane` runtime flag (integer perf_level 0–4) controls data collection granularity (performance buffers, shared memory writes) but does NOT control log output.

> **host_build_graph (host-orch) note.** The profiling **macros** below
> (`SIMPLER_DFX`, `SIMPLER_ORCH_PROFILING`, …) are shared with
> `tensormap_and_ringbuffer`. But the orchestrator-timing **device-log lines**
> (`orch_start` / `orch_end` / `orch_cost`) and the
> device-log line-count formulas that include `N_orch` describe the
> **device-orch** case that `tensormap_and_ringbuffer` runs. In
> host_build_graph the orchestrator runs on the **host** and the device boots
> scheduler-only — `aicpu_executor.cpp` carries no on-device orchestrator path
> at all — so those orch-timing lines do **not** appear in the device log; only
> the scheduler-timing lines do. Orchestrator profiling for host_build_graph is
> a host-side measurement.

## Profiling Macro Hierarchy

Defaults and dependency validation are centralized in
`src/common/task_interface/profiling_config.h`. Runtime headers include that
file before using the macros, so both a2a3 and a5 share the same default
values and compile-time checks.

```text
SIMPLER_DFX (base level, default=1)
├── SIMPLER_ORCH_PROFILING (orchestrator, default=0, requires SIMPLER_DFX=1)
|   └──SIMPLER_TENSORMAP_PROFILING (tensormap, default=0, requires SIMPLER_ORCH_PROFILING=1)
├── SIMPLER_SCHED_PROFILING (scheduler, default=0, requires SIMPLER_DFX=1)
└── --enable-l2-swimlane [PERF_LEVEL] (L2 swimlane data collection, 0-4, bare=4, requires SIMPLER_DFX=1)

```

### Compile-Time Validation

Each sub-level macro requires `SIMPLER_DFX=1`:

```cpp
#if SIMPLER_ORCH_PROFILING && !SIMPLER_DFX
#error "SIMPLER_ORCH_PROFILING requires SIMPLER_DFX=1"
#endif

#if SIMPLER_SCHED_PROFILING && !SIMPLER_DFX
#error "SIMPLER_SCHED_PROFILING requires SIMPLER_DFX=1"
#endif

#if SIMPLER_TENSORMAP_PROFILING && !SIMPLER_ORCH_PROFILING
#error "SIMPLER_TENSORMAP_PROFILING requires SIMPLER_ORCH_PROFILING=1"
#endif
```

## Profiling Levels

### Level 0: No Profiling (SIMPLER_DFX=0)

**What's compiled:**

- Debug/diagnostic logs (always present)
- Progress tracking (`PTO2 progress: completed=...`)
- Stall detection and dump (triggered after the `SCHEDULER_TIMEOUT_MS` wall-clock no-progress budget)
- Deadlock/livelock detection (`diagnose_stuck_state`, called on stall)

**What's NOT compiled:**

- All `CYCLE_COUNT_*` timing counters (`sched_*_cycle`, orchestrator cost counters)
- Scheduler/Orchestrator profiling summary logs guarded by `#if SIMPLER_DFX`
- Performance data collection paths (`enable_l2_swimlane` runtime flag becomes ineffective because profiling code is not compiled)

**Log output (normal run, no stall):**

- No `sched_start/sched_end/sched_cost` timestamps
- No `orch_start/orch_end/orch_cost` timestamps
- No `Scheduler summary: total_time=...`
- No `PTO2 total submitted tasks` log
- `PTO2 progress: completed=... total=...` may appear (thread 0 only, at task completion milestones)

---

### Level 1: Basic Profiling (SIMPLER_DFX=1)

**What's compiled:**

- Base timing counters for scheduler loop (`sched_complete/dispatch/idle/scan`)
- Per-thread orchestration timing (`orch_start`, `orch_end`, `orch_cost`)
- PTO2 total submitted tasks count (printed by last orch thread, after orch timing line)
- Scheduler summary output (`total_time`, `loops`, `tasks_scheduled`)
- Scheduler lifetime timestamps and cost (`sched_start`, `sched_end`, `sched_cost` — captured inside `resolve_and_dispatch_pto2()`, printed before Scheduler summary)

**What's NOT compiled:**

- Detailed phase breakdowns
- TensorMap statistics

**Log output (additional lines vs Level 0, per normal run):**

- `Thread %d: orch_start=%llu orch_end=%llu orch_cost=%.3fus` — each orch thread, after orchestration fully complete
- `PTO2 total submitted tasks = %d, already executed %d tasks` — last orch thread only (×1), after orch timing line
- `Thread %d: sched_start=%llu sched_end=%llu sched_cost=%.3fus` — each sched thread, printed before Scheduler summary
- `Thread %d: Scheduler summary: total_time=%.3fus, loops=%llu, tasks_scheduled=%d` — each sched thread
- `Thread %d: sched_start=%llu sched_end(timeout)=%llu sched_cost=%.3fus` — timeout path only (replaces normal `sched_end`)

**LOG_INFO_V9 count (normal run):**

- `N_sched*2 + N_orch*1 + 1` (orch_timing + PTO2_total + sched_timing + Scheduler_summary)

> See the table at the end for concrete counts based on the `paged_attention` example.

**Example log output** (from `paged_attention`, device 10):

```text
Thread 2: orch_start=48214752948321 orch_end=48214752959379 orch_cost=230.000us
Thread 3: orch_start=48214752948316 orch_end=48214752961505 orch_cost=275.000us
PTO2 total submitted tasks = 13, already executed 13 tasks
Thread 1: sched_start=48214752948235 sched_end=48214752962379 sched_cost=295.000us
Thread 1: Scheduler summary: total_time=159.560us, loops=3782, tasks_scheduled=6
Thread 0: sched_start=48214752948200 sched_end=48214752963571 sched_cost=320.000us
Thread 0: Scheduler summary: total_time=183.180us, loops=4611, tasks_scheduled=7
```

**Note:**

- All logs above are controlled by compile-time macro `SIMPLER_DFX`, not by `enable_l2_swimlane`.
- `enable_l2_swimlane` only controls shared-memory data collection / swimlane export.

---

### Level 2: Scheduler Detailed Profiling (SIMPLER_SCHED_PROFILING=1)

**Requires:** `SIMPLER_DFX=1`

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
stats live in `aicpu_scheduler_phases[]` (in `l2_swimlane_records.json`
captured at l2_swimlane_level >= 3) and `deps.json`; consume them via
`simpler_setup/tools/sched_overhead_analysis.py`.

---

### Level 3: Orchestrator Detailed Profiling (SIMPLER_ORCH_PROFILING=1)

**Requires:** `SIMPLER_DFX=1`

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

**Note:** Orchestrator logs always print when `SIMPLER_ORCH_PROFILING=1`, regardless of `enable_l2_swimlane` flag.

---

### Level 4: TensorMap Profiling (SIMPLER_TENSORMAP_PROFILING=1)

**Requires:** `SIMPLER_DFX=1` AND `SIMPLER_ORCH_PROFILING=1`

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

## Runtime Flag: enable_l2_swimlane (perf_level)

`--enable-l2-swimlane` accepts an integer perf_level (0–4). Transport
mirrors the PMU pattern — two independent channels (one binary, one int):

- **Binary on/off** — `KernelArgs::enable_profiling_flag` bit1
  (`SIMPLER_DFX_FLAG_L2_SWIMLANE`). Set by the host whenever level > 0; read
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
| 1 | AICore timing only (start/end/task_token_raw) — AICPU `complete_task` is bypassed |
| 2 | + AICPU dispatch_time, finish_time |
| 3 | + Scheduler phases (`SCHED_*`) |
| 4 | + Orchestrator phases (full) |

At level 1 the AICore record carries the full PTO2 `task_token_raw`
(`(ring_id << 32) | local_id`), read straight from
`LocalContext.async_ctx.task_token.raw` inside the AICore helper —
already in cache from the dispatch payload, so no extra GM load.
Identity fields the AICPU side used to write at level 1 (`func_id`,
`core_type`) are derived host-side:

- `func_id` ← `deps.json`'s per-task `kernel_ids[]`, joined by
  `task_id` at post-process by `swimlane_converter.py`. Same model
  `fanout` already uses.
- `core_type` ← per-core static table published by the host into the
  collector (`L2SwimlaneCollector::set_core_types`).

AICore buffer rotation no longer piggy-backs on `complete_task`. AICPU
counts dispatches per core in the dispatch path (scheduler_dispatch in
tensormap_and_ringbuffer; aicpu_executor in host_build_graph) and rotates
the AICore buffer when the count is about to cross a
`PLATFORM_AICORE_BUFFER_SIZE` boundary — strictly before
`write_reg(DATA_MAIN_BASE)` for the first task of the new batch. The
hook is `l2_swimlane_aicpu_on_aicore_dispatch`. No AICore-side signal is
needed: AICPU has full dispatch visibility on its own. Race safety comes
from the completion-before-dispatch invariant (AICore per core is
single-threaded and AICPU does not dispatch task K+1 until K FIN'd), which
guarantees AICore has FIN'd — and `dcci`'d out — every record in the old
buffer by rotation time. This decoupling is what lets level 1 skip
`complete_task` without losing rotations.

Fanout edges are no longer carried on the device hot path — `swimlane_converter.py`
joins them from the sibling `deps.json` (produced by dep_gen) at post-process time.

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
SIMPLER_DFX=0
```

### Basic Performance Monitoring

```bash
# Minimal overhead, summary logs only
SIMPLER_DFX=1
SIMPLER_ORCH_PROFILING=0
SIMPLER_SCHED_PROFILING=0
```

### Scheduler Performance Analysis

```bash
# Detailed scheduler breakdown
SIMPLER_DFX=1
SIMPLER_ORCH_PROFILING=0
SIMPLER_SCHED_PROFILING=1
```

### Orchestrator Performance Analysis

```bash
# Detailed orchestrator breakdown
SIMPLER_DFX=1
SIMPLER_ORCH_PROFILING=1
SIMPLER_SCHED_PROFILING=0
```

### Full Profiling (maximum overhead)

```bash
# All profiling features enabled
SIMPLER_DFX=1
SIMPLER_ORCH_PROFILING=1
SIMPLER_SCHED_PROFILING=1
SIMPLER_TENSORMAP_PROFILING=1
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
#define SIMPLER_DFX 1
#define SIMPLER_ORCH_PROFILING 1
#include "pto_runtime2_types.h"
```

---

## Log Output Summary

> Example: `paged_attention` on Ascend hardware, normal run (no stall/timeout).

| Level | Macro Settings | LOG_INFO_V9 Count | Description |
| ----- | -------------- | ----------------- | ----------- |
| 0 | `SIMPLER_DFX=0` | 0 | No timing output |
| 1 | `SIMPLER_DFX=1` | 7 | Timing timestamps + scheduler summary |
| 2 | `+SIMPLER_SCHED_PROFILING=1` | — | Scheduler detailed phase breakdown |
| 3 | `+SIMPLER_ORCH_PROFILING=1` | — | Orchestrator detailed phase breakdown |
| 4 | `+SIMPLER_TENSORMAP_PROFILING=1` | — | TensorMap lookup stats |

---

## Implementation Notes

### Key Principles

1. **Macros control compilation and logging**
   - `#if SIMPLER_DFX` controls whether profiling code is compiled
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
- Scheduler profiling: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/scheduler/scheduler_dispatch.cpp` and `scheduler_cold_path.cpp`
- Orchestrator profiling: `src/a2a3/runtime/tensormap_and_ringbuffer/aicpu/aicpu_executor.cpp`
- TensorMap profiling: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/pto_tensormap.h`

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
