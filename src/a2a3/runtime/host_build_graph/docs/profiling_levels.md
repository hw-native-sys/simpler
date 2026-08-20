# Profiling Levels

This document describes the profiling macro hierarchy and logging control in the simpler runtime.

## Overview

The runtime uses a hierarchical profiling system with compile-time macros to control profiling code compilation and log output. The `enable_chip_swimlane` runtime flag (integer perf_level 0–4) controls data collection granularity (performance buffers, shared memory writes) but does NOT control log output.

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
└── --enable-chip-swimlane [PERF_LEVEL] (chip swimlane data collection, 0-4, bare=4, requires SIMPLER_DFX=1)

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
- Performance data collection paths (`enable_chip_swimlane` runtime flag becomes ineffective because profiling code is not compiled)

**Log output (normal run, no stall):**

- No `sched_start/sched_end/sched_cost` timestamps
- No `orch_start/orch_end/orch_cost` timestamps
- No `Scheduler summary: total_time=...`
- No `PTO2 total submitted tasks` log
- `PTO2 progress: completed=... total=...` may appear (thread 0 only, at task completion milestones)

---

### Level 1: Basic Profiling (SIMPLER_DFX=1)

host_build_graph boots **scheduler-only** — the orchestrator runs on the host,
so the device log carries no `orch_start`/`orch_end`/`orch_cost` lines and no
`PTO2 total submitted tasks` line (see the note at the top of this file). Every
AICPU thread schedules its own core slice, so `N_sched == aicpu_thread_num`.

**What's compiled:**

- Base timing counters for scheduler loop (`sched_complete/dispatch/idle/scan`)
- Scheduler summary output (`total_time`, `loops`, `tasks_scheduled`)
- Scheduler lifetime timestamps and cost (`sched_start`, `sched_end`, `sched_cost` — captured inside `resolve_and_dispatch_pto2()`, printed before Scheduler summary)

**What's NOT compiled:**

- Detailed phase breakdowns
- TensorMap statistics

**Log output (additional lines vs Level 0, per normal run):**

- `Thread %d: sched_start=%llu sched_end=%llu sched_cost=%.3fus` — each scheduler thread, printed before Scheduler summary
- `Thread %d: Scheduler summary: total_time=%.3fus, loops=%llu, tasks_scheduled=%d` — each scheduler thread
- `Thread %d: sched_start=%llu sched_end(timeout)=%llu sched_cost=%.3fus` — timeout path only (replaces normal `sched_end`)

**LOG_INFO count (normal run):**

- `N_sched*2` (sched_timing + Scheduler_summary per scheduler thread)

> See the table at the end for concrete counts based on the `paged_attention` example.

**Example log output** (`paged_attention`, `aicpu_thread_num=4`, scheduler-only):

```text
Thread 0: sched_start=48214752948200 sched_end=48214752963571 sched_cost=320.000us
Thread 0: Scheduler summary: total_time=183.180us, loops=4611, tasks_scheduled=4
Thread 1: sched_start=48214752948235 sched_end=48214752962379 sched_cost=295.000us
Thread 1: Scheduler summary: total_time=159.560us, loops=3782, tasks_scheduled=3
Thread 2: sched_start=48214752948260 sched_end=48214752961840 sched_cost=280.000us
Thread 2: Scheduler summary: total_time=151.220us, loops=3510, tasks_scheduled=3
Thread 3: sched_start=48214752948290 sched_end=48214752961505 sched_cost=275.000us
Thread 3: Scheduler summary: total_time=147.940us, loops=3402, tasks_scheduled=3
```

**Note:**

- All logs above are controlled by compile-time macro `SIMPLER_DFX`, not by `enable_chip_swimlane`.
- `enable_chip_swimlane` only controls shared-memory data collection / swimlane export.

---

### Level 2: Scheduler Detailed Profiling (SIMPLER_SCHED_PROFILING=1)

**Requires:** `SIMPLER_DFX=1`

**What's compiled:**

- All Level 1 features
- Detailed scheduler phase counters
- Phase-specific statistics (complete, scan, dispatch, idle)
- Hit rate tracking (complete poll, ready queue pop)

**Log output:** 18 LOG_INFO logs (11 debug + 2 basic + 7 scheduler detailed - 2 replaced)

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
stats live in `aicpu_scheduler_phases[]` (in `chip_swimlane_records.json`
captured at chip_swimlane_level >= 3) and `deps.json`; consume them via
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

**Log output:** 30 LOG_INFO logs (11 debug + 2 basic + 1 scheduler summary + 17 orchestrator detailed - 1 replaced)

- Replaces basic orchestration completion with detailed breakdown

**Orchestrator output:**

```text
Thread X: === Orchestrator Profiling: XXX tasks, total=XXXus ===
Thread X:   task_ring_alloc: XXXus (XX.X%)
Thread X:   param_copy     : XXXus (XX.X%)  atomics=XXX
Thread X:   lookup+dep     : XXXus (XX.X%)
Thread X:   heap_alloc     : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   tensormap_ins  : XXXus (XX.X%)
Thread X:   fanin+ready    : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   finalize+SM    : XXXus (XX.X%)  work=XXXus wait=XXXus  atomics=XXX
Thread X:   scope_end      : XXXus  atomics=XXX
Thread X:   avg/task       : XXXus
```

**Note:** Orchestrator logs always print when `SIMPLER_ORCH_PROFILING=1`, regardless of `enable_chip_swimlane` flag.

---

### Level 4: TensorMap Profiling (SIMPLER_TENSORMAP_PROFILING=1)

**Requires:** `SIMPLER_DFX=1` AND `SIMPLER_ORCH_PROFILING=1`

**What's compiled:**

- All Level 3 features
- TensorMap lookup statistics
- Hash chain walk tracking
- Overlap check counters

**Log output:** 34 LOG_INFO logs (30 from Level 3 + 4 tensormap)

**TensorMap output:**

```text
Thread X: === TensorMap Lookup Stats ===
Thread X:   lookups        : XXX, inserts: XXX
Thread X:   chain walked   : total=XXX, avg=X.X, max=X
Thread X:   overlap checks : XXX, hits=XXX (XX.X%)
```

---

## Prepare-Path Timing: One Pool, Three Views

`host_build_graph` times its prepare path — the `chip.run.bind` stage's
segments, and the host orchestrator's submit-level operations inside it. One
recorder feeds three views, and two independent switches decide which of them
appear.

### What is recorded

Sixteen kinds, all on the host monotonic clock the `[STRACE]` host spans use,
so records and spans read against each other with no alignment step.

| Group | Kinds |
| ----- | ----- |
| Bind segments (partition the stage) | `args`, `arena_build`, `static_arena`, `gm_heap`, `shared_mem`, `runtime_init`, `host_orch`, `graph_upload`, `arena_h2d`, `host_view_close` |
| Orchestrator operations (inside `host_orch`) | `submit_task`, `alloc_tensors`, `record_node`, `graph_submit`, `build_definition` |

Three of the orchestrator kinds end with a task submitted — `submit_task`,
`alloc_tensors`, `graph_submit` — so their count is the pass's `total_tasks`.
The other two are sub-operations of one of those, which is why a per-kind total
is a **cost share, not an interval**: a per-event mean is `total_ns / count`, and
the spread is not recoverable from it.

### The two switches

| Switch | Turns on |
| ------ | -------- |
| `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE` (env, off unless it starts with `1`, `t` or `T`) | the `LOG_TIMING` breakdown; needs no rebuild and works at any `--rounds` |
| `--enable-chip-swimlane 4` (CallConfig `chip_swimlane_level`) | the host lane in `chip_swimlane_records.json`, with its records clock-aligned against the device timeline |

Collection is armed by **either** — a level-4 run collects records with the
variable unset, and the variable collects them with the level at 0. What is
recorded does not depend on which one armed the pass: the swimlane's share is a
projection at export, not a branch at record time, so the two configurations
measure the same overhead and their numbers are comparable.

```bash
# breakdown only, any --rounds, no rebuild
SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 python -m pytest <case> --platform <platform> --device 0

# host lane on the chip swimlane, aligned against the device timeline
python -m pytest <case> --platform <platform> --device 0 --enable-chip-swimlane 4
```

### The three views

- **`LOG_TIMING` lines**, at the default log threshold, gated by the env
  variable. One `bind phase=<p> start_ns=<n> dur_ns=<n> <attrs>` line per
  segment, plus one `host-orch phase=<p> total_ns=<n> count=<k> detail_sum=<n>
  dropped=<n>` line per orchestrator kind. They come from per-kind counters, not
  from the record pool. The counters use lock-free atomic additions across the
  main and recording-worker lanes, with every phase isolated on its own cache
  line so concurrent `graph_submit` and `record_node` updates do not
  false-share. The per-event pool is armed when the level-4 artifact is being
  written (a producer that wants records *and* an output prefix) or whenever the
  chip swimlane is at `ORCH_PHASES`; a steady-state run satisfies neither, so it
  pays no pool append and no artifact lock at all. A total stays exact even if
  the pool was never armed or overflowed — that is what makes this the channel
  for steady state, where `--rounds > 1` switches every artifact collector off.

  Every line is written at the end of the pass, keeping the log write off the path
  being measured. The line therefore carries its own `start_ns`; the log prefix
  timestamps the write, not the segment.

- **`host_phase_records.jsonl`** in the per-case output directory, when the run
  has one. One JSON Lines object per pass carrying `pid` / `inv` — the identity
  the `[STRACE]` tree groups by — and one record per operation. This is the
  channel to read for a distribution or a per-event timeline; the summed lines
  cannot express either. Every record carries its producer Linux tid.
  `strace_timing.py --swimlane --host-phase-records <path>` draws each record
  inside the matching `chip.run.bind`; `record_node` and `build_definition`
  appear on the `graph record worker` lane, while outer `graph_submit` events
  appear on the `graph submit main` lane.

- **The host lanes of `chip_swimlane_records.json`**, at level 4 only, joined to
  the device timeline through the clock anchors (see `host/clock_correlation.h`).
  Two projections of the pool land there:

  | Key | Kinds | Rendered as |
  | --- | ----- | ----------- |
  | `host_orchestrator_phases` | the task-submitting kinds | `Host Orchestrator` process |
  | `host_device_uploads` | `arena_h2d`, with the complete runtime-image byte count | `Host Prepare` / `H2D` lane |

  The upload lane is the one place the whole question — orchestration plus H2D
  inside a millisecond — is visible against the device execution it precedes; the
  rest of the bind stage is host-only setup with no device counterpart.

  The `host_capture` block reports `expected_records` (the pass's task count)
  against `recorded_records` (the submit projection), plus `pool_records` for the
  whole population — a pool count above the projection is normal, not incomplete.

The stage's *duration* is already published as the `chip.run.bind` `[STRACE]`
marker, so the marker and this breakdown are a total and its parts rather than two
spellings of one number. The parts are not markers themselves: the marker grammar
is the platform's public per-run-stage contract (see `pto_runtime_c_api.h` and
[docs/dfx/host-trace.md](../../../../../docs/dfx/host-trace.md)), whose consumers
key off a fixed stage set, and a runtime's internal breakdown of one stage does
not belong in it.

### Cost and capacity

A record costs two clock reads and a store. Its per-kind counter update is a
lock-free atomic add; only the per-event pool append is serialized, and only
when the pool is armed. It sits on the path being measured, which is why neither
switch is on by default.

The pool holds `PLATFORM_HOST_PHASE_BUFFERS × PLATFORM_HOST_PHASE_RECORDS_PER_BUFFER`
records (5 × 1024 = 5120, 200 KiB) in fixed-size buffers rotated in index order,
the same shape as the device sched/orch pools with host DDR as its backing.
Beyond that, records are dropped from the tail and counted: the per-event views
truncate, the per-kind totals stay exact, and `dropped=` on every `host-orch` line
plus `dropped_records` in `host_capture` say so.

---

## Runtime Flag: enable_chip_swimlane (perf_level)

`--enable-chip-swimlane` accepts an integer perf_level (0–4). Transport
mirrors the PMU pattern — two independent channels (one binary, one int):

- **Binary on/off** — `KernelArgs::enable_profiling_flag` bit1
  (`SIMPLER_DFX_FLAG_CHIP_SWIMLANE`). Set by the host whenever level > 0; read
  by AICore (which only needs on/off to decide whether to write timing) and
  by AICPU kernel entry via `set_chip_swimlane_enabled(bool)`.
- **Granular level (0–4)** — `ChipSwimlaneDataHeader::chip_swimlane_level`
  (shared memory). Host writes it in `ChipSwimlaneCollector::initialize`; AICPU
  promotes it from the header in `chip_swimlane_aicpu_init` and exposes it via
  `get_chip_swimlane_level()` (typed `ChipSwimlaneLevel`) for
  `>= AICPU_TIMING / SCHED_PHASES / ORCH_PHASES` gates.

On sim, the binary on/off travels via the dlsym'd `set_chip_swimlane_enabled`
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
  collector (`ChipSwimlaneCollector::set_core_types`).

AICore buffer rotation no longer piggy-backs on `complete_task`. AICPU
counts dispatches per core in the dispatch path (scheduler_dispatch in
tensormap_and_ringbuffer; aicpu_executor in host_build_graph) and rotates
the AICore buffer when the count is about to cross a
`PLATFORM_AICORE_BUFFER_SIZE` boundary — strictly before
`write_reg(DATA_MAIN_BASE)` for the first task of the new batch. The
hook is `chip_swimlane_aicpu_on_aicore_dispatch`. No AICore-side signal is
needed: AICPU has full dispatch visibility on its own. Race safety comes
from the completion-before-dispatch invariant (AICore per core is
single-threaded and AICPU does not dispatch task K+1 until K FIN'd), which
guarantees AICore has FIN'd — and `dcci`'d out — every record in the old
buffer by rotation time. This decoupling is what lets level 1 skip
`complete_task` without losing rotations.

Fanout edges are no longer carried on the device hot path — `swimlane_converter.py`
joins them from the sibling `deps.json` (produced by dep_gen) at post-process time.

Bare `--enable-chip-swimlane` = level 4 (backward compatible).

### Level gating in AICPU code

Use the strongly-typed `ChipSwimlaneLevel` enum so each gate names the
content it depends on instead of relying on magic numbers:

```cpp
// Any level > 0: AICPU task record buffer init / flush.
// Cheap binary check, available immediately after kernel entry.
if (is_chip_swimlane_enabled()) { ... }

// AICPU dispatch/finish timestamps.
// Granular checks below require chip_swimlane_aicpu_init to have already run
// (so the level has been promoted from the shared-memory header).
if (get_chip_swimlane_level() >= ChipSwimlaneLevel::AICPU_TIMING) { ... }

// Scheduler main-loop phase records (SCHED_*)
if (get_chip_swimlane_level() >= ChipSwimlaneLevel::SCHED_PHASES) { ... }

// Orchestrator phase records
if (get_chip_swimlane_level() >= ChipSwimlaneLevel::ORCH_PHASES) { ... }
```

`ChipSwimlaneLevel` is defined in `common/chip_swimlane_profiling.h` with
underlying type `uint32_t` (matches the `ChipSwimlaneDataHeader::chip_swimlane_level`
shared-memory field and mirrors `PmuEventType : uint32_t`):

| Enumerator | Underlying value |
| ---------- | ---------------- |
| `DISABLED` | 0 |
| `AICORE_TIMING` | 1 |
| `AICPU_TIMING` | 2 |
| `SCHED_PHASES` | 3 |
| `ORCH_PHASES` | 4 |

### When enable_chip_swimlane=0

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
# Example: disable all device profiling code
CXXFLAGS="-DSIMPLER_DFX=0" pip install --no-build-isolation -e .

# Example: enable orchestrator and tensormap profiling
CXXFLAGS="-DSIMPLER_ORCH_PROFILING=1 -DSIMPLER_TENSORMAP_PROFILING=1" \
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

> Example: `paged_attention` on Ascend hardware, `aicpu_thread_num=4`, normal
> run (no stall/timeout). host_build_graph boots scheduler-only, so the Level-1
> count is `N_sched*2` with no orchestrator lines (`N_sched == aicpu_thread_num`).

| Level | Macro Settings | LOG_INFO Count | Description |
| ----- | -------------- | -------------- | ----------- |
| 0 | `SIMPLER_DFX=0` | 0 | No timing output |
| 1 | `SIMPLER_DFX=1` | 8 | Scheduler timing + summary (4 threads × 2) |
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
   - `enable_chip_swimlane` controls performance buffer allocation
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
- Data collection (`enable_chip_swimlane>0`): Low to moderate
  - Performance buffer writes
  - Shared memory updates
  - Per-task timing measurements

### Recommendation

- Use Level 0 for production
- Use Level 1-2 for performance monitoring
- Use Level 3-4 for detailed performance analysis only
