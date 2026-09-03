# Profiling Levels

This document describes the profiling macro hierarchy and logging control in the simpler runtime.

## Overview

The runtime uses a hierarchical profiling system with compile-time macros to control profiling code compilation and log output. The `enable_chip_swimlane` runtime flag (integer perf_level 0–4) controls data collection granularity (performance buffers, shared memory writes) but does NOT control log output.

> **A5 HBG scheduler selection.** Diagnostic flags never select the scheduler.
> Ordinary DAGs remain on the A5 HBG AICore Scheduler, while Graph replay
> remains on its explicit AICPU compatibility path. Both producers export the
> same `scheduler_records` schema; stream metadata identifies `producer` so
> tools never apply AICPU scheduling assumptions to AICore intervals.
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
- Progress tracking (`progress: completed=...`)
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
- No `total submitted tasks` log
- `progress: completed=... total=...` may appear (thread 0 only, at task completion milestones)

---

### Level 1: Basic Profiling (SIMPLER_DFX=1)

host_build_graph boots **scheduler-only** — the orchestrator runs on the host,
so the device log carries no `orch_start`/`orch_end`/`orch_cost` lines and no
`total submitted tasks` line (see the note at the top of this file). Every
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
stats live in `scheduler_records.streams[]` (in `chip_swimlane_records.json`
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
recorder feeds three views, and three independent switches decide which of them
appear.

### What is recorded

Twenty-one kinds, all on the host monotonic clock the `[STRACE]` host spans use,
so records and spans read against each other with no alignment step.

| Group | Kinds |
| ----- | ----- |
| Bind segments (one interval each, inside the stage) | `args`, `arena_build`, `static_arena`, `gm_heap`, `shared_mem`, `runtime_init`, `host_orch`, `graph_upload`, `arena_h2d`, `host_view_close` |
| Orchestrator operations (inside `host_orch`) | `submit_task`, `alloc_tensors`, `record_in_graph_task`, `graph_submit`, `build_definition`, `graph_begin`, `recording_wait`, `graph_commit`, `submit_admit`, `record_handoff`, `generated_args` |

Three of the orchestrator kinds end with a task submitted — `submit_task`,
`alloc_tensors`, `graph_submit` — so their count is the bind's `total_tasks`
(`host_phase_kind_submits_task` is the one place that answers this). The other
eight are sub-operations of one of those, or the segments between them, which is
why a per-kind total is a **cost share, not an interval**: a per-event mean is
`total_ns / count`, and the spread is not recoverable from it.

The last three come from the generated orchestration `.so` rather than the runtime,
through the ops table's `record_orch_phase`, so they carry submit group 0 rather
than the submission they belong to.

### A phase is an interval; a quantity is an attribute

The two shapes of information on this path are not interchangeable, and choosing
the wrong one produces a number that reads as data and is not:

- **A record is an interval** — one operation, start to end. Its `detail` says
  *which* operation (a task id, a Graph key, the submission index) or *how much*
  it covered (`build_definition`'s in-graph task count, `recording_wait`'s in-flight
  count). That is the whole contract.
- **A quantity about a segment is an attribute** — `bytes=`, `heap_used=`,
  `spilled=`, `minflt=`, `nvcsw=`. It goes in the segment's attribute string,
  which is what the segment's span carries. That string is capped at the span
  attribute field's width (`SIMPLER_HOST_SPAN_ATTRIBUTES_CAPACITY`), and the
  kernel counters are formatted first, so an overlong one loses a caller quantity
  the artifact's `detail` can still supply rather than a counter nothing else
  carries.

So: **a new interval to name earns a new kind; a new quantity about an interval
that already exists is an attribute on it.** Adding a kind to carry a statistic
puts a measurement into the timeline where a reader expects a duration, and the
breakdown will then sum it.

Which is exactly why `detail` is summed only where it counts something
(`host_phase_kind_detail_is_quantity`). Nine of the eleven orchestrator kinds
carry an identity, and a sum over identities — task ids added together — was
printed as `detail_sum` for as long as the column was unconditional.

### The three switches

| Switch | Turns on |
| ------ | -------- |
| `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE` (env, off unless it starts with `1`, `t` or `T`) | the breakdown in the log — segment spans plus the `host-orch` cost-share lines; needs no records, no rebuild, and works at any `--rounds` |
| `SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE` (env, same spelling) | per-event collection into the pool, which reaches `host_phase_records.jsonl` when the run has an output directory |
| `--enable-chip-swimlane 4` (CallConfig `chip_swimlane_level`) | per-event collection *and* the host lane in `chip_swimlane_records.json`, with its records clock-aligned against the device timeline |

The first switch is orthogonal to the other two: it reads the per-kind counters,
which are exact whether or not records are kept, so a run can take the breakdown
with no records and records with no breakdown.

Collection is armed by **either** of the other two — a level-4 run collects
records with the variable unset, and the variable collects them with the level at
0. What is recorded does not depend on which one armed the bind: the swimlane's
share is a projection at export, not a branch at record time, so the two
configurations measure the same overhead and their numbers are comparable.

Collecting and writing are separate questions. A collected bind reaches
`host_phase_records.jsonl` whenever the run has an output directory, so a level-4
run produces the artifact too; conversely the variable arms nothing when there is
no directory to write into, since a pool no reader drains would be pure cost.

```bash
# breakdown only, any --rounds, no rebuild — no pool, no artifact
SIMPLER_HBG_BIND_BREAKDOWN_ENABLE=1 python -m pytest <case> --platform <platform> --device 0

# per-event records; --enable-scope-stats is the cheapest way to get an output directory
SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE=1 \
  python -m pytest <case> --platform <platform> --device 0 --enable-scope-stats

# host lane on the chip swimlane, aligned against the device timeline
python -m pytest <case> --platform <platform> --device 0 --enable-chip-swimlane 4
```

### The three views

- **The log**, at the default threshold, gated by
  `SIMPLER_HBG_BIND_BREAKDOWN_ENABLE`. Two shapes, because a segment and a cost
  share are different things: one `[STRACE]` span named
  `chip.run.bind.<segment>` per segment, carrying `ts` / `dur` / `<attrs>` at
  depth 2 inside the `chip.run.bind` span; and one `host-orch phase=<p>
  total_ns=<n> count=<k> detail_sum=<n> dropped=<n>` `LOG_TIMING` line per
  orchestrator kind, which stays a line because those kinds nest inside each
  other and a total over them is not an interval.
  Both come from per-kind counters, not from the record pool. The counters use
  lock-free atomic additions across the main and recording-worker lanes, with
  every phase isolated on its own cache line so concurrent `graph_submit` and
  `record_in_graph_task` updates do not false-share. The per-event pool is armed when the
  artifact is wanted (`SIMPLER_HBG_HOST_PHASE_RECORDS_ENABLE` *and* an output
  prefix) or whenever the chip swimlane is at `ORCH_PHASES`; a steady-state run
  satisfies neither, so it pays no pool append and no artifact lock at all. A
  total stays exact even if the pool was never armed or overflowed — that is what
  makes this the channel for steady state, where `--rounds > 1` switches every
  artifact collector off.

  Both are written at the end of the bind, keeping the write off the path being
  measured. The span therefore carries its own `ts`, taken when the segment
  opened, rather than leaving its start to be inferred from when the record was
  written — `STRACE_HOST_SPAN_AT_A` exists for that, and `chip.run` itself is
  emitted the same way.

- **`host_phase_records.jsonl`** in the per-case output directory, when a
  collecting bind has one. One JSON Lines object per bind carrying `pid` / `inv`
  — the identity the `[STRACE]` tree groups by — and one record per operation.
  This is the channel to read for a distribution or a per-event timeline; the
  summed lines cannot express either. Every record carries its producer Linux
  tid. `strace_timing.py --swimlane --host-phase-records <path>` draws each record
  inside the matching `chip.run.bind`; `record_in_graph_task` and `build_definition`
  appear on the `graph record worker` lane, while outer `graph_submit` events
  appear on the `graph submit main` lane.

- **The host lanes of `chip_swimlane_records.json`**, at level 4 only, joined to
  the device timeline through the clock anchors (see `host/clock_correlation.h`).
  Two projections of the pool land there:

  | Key | Kinds | Rendered as |
  | --- | ----- | ----------- |
  | `host_orchestrator_phases` | the task-submitting kinds | `Host Orchestrator` process |
  | `host_device_uploads` | `graph_upload`, `arena_h2d`, with byte counts | `Host Prepare` / `H2D` lane |

  The upload lane is the one place the whole question — orchestration plus H2D
  inside a millisecond — is visible against the device execution it precedes; the
  rest of the bind stage is host-only setup with no device counterpart.

  The `host_capture` block reports `expected_records` (the bind's task count)
  against `recorded_records` (the submit projection), plus `pool_records` for the
  whole population — a pool count above the projection is normal, not incomplete.

The stage's *duration* is the `chip.run.bind` `[STRACE]` span, and its segments
are `[STRACE]` spans one level below it, so the two are a stage and its parts in
one format. A runtime subdividing a stage it owns is ordinary: the tensormap
runtime's `chip.run.bind.args` and `chip.run.bind.prebuilt` do it, and the device
sub-phases are that runtime's own AICPU breakdown, read back from a cycle buffer
and re-emitted as spans (see `runtime_c_api.h` and
[docs/dfx/host-trace.md](../../../../../docs/dfx/host-trace.md)). What is *not*
a span is a summed cost share over kinds that nest inside each other — those have
no honest position on a timeline, and they are the `host-orch phase=` lines.

### Cost and capacity

A record costs two clock reads and a store. Its per-kind counter update is a
lock-free atomic add; only the per-event pool append is serialized, and only
when the pool is armed. It sits on the path being measured, which is why no
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
  by AICore for the initial profiler activation and by AICPU kernel entry via
  `set_chip_swimlane_enabled(bool)`.
- **Granular level (0–4)** — `ChipSwimlaneDataHeader::chip_swimlane_level`
  (shared memory). Host writes it in `ChipSwimlaneCollector::initialize`; AICPU
  promotes it from the header in `chip_swimlane_aicpu_init` and exposes it via
  `get_chip_swimlane_level()` (typed `ChipSwimlaneLevel`) for
  `>= SCHEDULE_TIMING / SCHED_PHASES / ORCH_PHASES` gates. The A5 HBG host also
  copies the level into `SchedulerRunControl`, where the AICore Scheduler reads
  it once and gates task timing, per-task scheduling timing, and phase timing
  independently.

On sim, the binary on/off travels via the dlsym'd `set_chip_swimlane_enabled`
entry point; the granular level still goes through the shared-memory
header just like on onboard.

| Level | Collects |
| ----- | -------- |
| 0 | Nothing (disabled) |
| 1 | AICore timing only (start/end/task_token_raw) — AICPU `complete_task` is bypassed |
| 2 | + Scheduler per-task dispatch_time, finish_time |
| 3 | + Scheduler phases (`SCHED_*`) |
| 4 | + Orchestrator phases (full) |

The A5 HBG AICore Scheduler records the same per-task timing contract as the
AICPU Scheduler: `dispatch_time` is the end of dispatch publication and
`finish_time` is when the Scheduler starts processing the completion. The raw
`scheduler_tasks.producer` field identifies which Scheduler produced these
timestamps. At level 2 and above, A5 HBG also exports AICPU lifecycle timestamps
for handshake, topology/configuration, context publication, bootstrap wait,
register release, and exit; these are supplemental control-plane records.

At level 3 and above, A5 HBG AICore Scheduler task intervals reuse the per-task
trace. Consecutive taskless scheduler-loop iterations are coalesced into one
`idle` interval and stored in one fixed-capacity, no-wrap buffer per Scheduler;
overflow increments `capture.dropped` and sets `capture.truncated`. Coalescing
keeps capture size dependent on idle-to-active transitions instead of Host CPU
speed in simulation. The buffer is not allocated below level 3. Interval
endpoints are captured at operation entry and exit; no interval is synthesized
from aggregate durations.

At level 1 the AICore record carries the full `task_token_raw`
(a `TaskId::raw`; see `src/common/host_build_graph/task_id.h`), read straight from
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

// On the AICPU compatibility path, gate its Scheduler task-timing producer.
// The A5 HBG AICore Scheduler applies the same enum contract host-side.
// Granular checks below require chip_swimlane_aicpu_init to have already run
// (so the level has been promoted from the shared-memory header).
if (get_chip_swimlane_level() >= ChipSwimlaneLevel::SCHEDULE_TIMING) { ... }

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
| `TASK_TIMING` | 1 |
| `SCHEDULE_TIMING` | 2 |
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
#include "runtime_types.h"
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
- TensorMap profiling: `src/a2a3/runtime/tensormap_and_ringbuffer/runtime/tensormap.h`

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
