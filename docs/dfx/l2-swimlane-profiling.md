# L2 Swimlane Profiling — Per-task Timing & Scheduler Phases

## 1. Background & Motivation

Why a kernel takes the time it takes is rarely visible from
end-to-end runtime numbers. Two cases dominate the profiler diet:

- **Task-level timing.** When kernel A is slow, is the kernel slow,
  or is something else holding it up? Where does each AICore task
  start and end on the wall clock, and what fanout / fan-in chain
  does it sit in?
- **Scheduler overhead.** Inside AICPU's scheduling loop, time is
  split across "process completed tasks", "dispatch ready tasks",
  "incremental scan for roots", and "idle wait". When the AICPU
  thread is hot, knowing which of those four phases dominates
  pinpoints the fix.

L2 swimlane profiling captures both: per-task `(start, end,
dispatch, finish)` records on the AICore side, plus per-iteration
phase records on the AICPU scheduler side and a one-shot
orchestrator summary. The host writes a Chrome Trace Event JSON
that loads directly in Perfetto, and the same file feeds a
scheduler-overhead deep-dive report when a device log is
available.

## 2. Overview

- **Per-task AICore timing** — `start_time_us`, `end_time_us`,
  `duration_us`, plus AICPU-stamped `dispatch_time_us` / `finish_time_us`.
- **Per-task fanout chain** — successor `task_id`s recorded in
  the L2 record so dependency arrows show up in the Perfetto
  view.
- **AICPU scheduler phases** — per-iteration breakdown into
  `complete` / `dispatch`. Idle iterations no longer emit a record
  on a2a3; the host tooling reconstructs idle spans from the gap
  between consecutive work records on the same thread. Legacy
  captures (and a5) may still carry `scan` / `idle` records — both
  are silently dropped by the parser (idle is double-painted
  by the gap reconstruction; `scan` was never emitted in a2a3).
- **Orchestrator submit envelope** — one record per `submit_task()`
  / `alloc_tensors()` call covering the whole submit's
  `[start, end]` window (`orch_submit` phase). Per-sub-step
  cumulative cycles (sync / alloc / params / lookup / insert /
  fanin) still live in the cold-path device log via the
  `g_orch_*_cycle` counters — that's where you go for "which
  sub-step dominates overall"; the per-submit record covers
  "which submit was slow".
- **Standard outputs** — raw `l2_swimlane_records.json`, plus a
  Perfetto-loadable `merged_swimlane_*.json` produced by
  `swimlane_converter`.

Enable in one line:

```bash
python tests/st/<case>/test_<name>.py -p <platform> -d 0 --enable-l2-swimlane
```

## 3. How to Use

### 3.1 Enable L2 swimlane

`--enable-l2-swimlane` accepts an optional integer **perf_level**
(0–4). A bare flag defaults to level 4 (full collection,
backward-compatible with the old boolean behavior).

| Level | Collects | Notes |
| ----- | -------- | ----- |
| 0 | Nothing (disabled) | Default when flag is absent |
| 1 | AICore timing only (start_time_us/end_time_us/task_id/func_id/core_type) | No AICPU timestamps, no fanout |
| 2 | + dispatch_time_us, finish_time_us, fanout | Full per-task record |
| 3 | + scheduler phases (`aicpu_scheduler_phases[]`) | Skips orchestrator phases |
| 4 | + orchestrator phases (`aicpu_orchestrator_phases[]`) | Full collection |

```bash
# Standalone runner
python tests/st/<case>/test_<name>.py -p <platform> -d 0 --enable-l2-swimlane [PERF_LEVEL]

# pytest — same flag shape
pytest tests/st/<case> --platform <platform> -d 0 --enable-l2-swimlane [PERF_LEVEL]

# Bare flag (no integer) — shorthand for level 4 (full collection)
python tests/st/<case>/test_<name>.py -p <platform> -d 0 --enable-l2-swimlane
```

- `<platform>` — one of `a2a3` / `a2a3sim` / `a5` / `a5sim`; the
  integer perf_level interface is identical across them.
- `[PERF_LEVEL]` — optional integer 0–4 (see table above). Omit the
  argument entirely (bare `--enable-l2-swimlane`) for the level-4
  shorthand; omit the flag entirely for level 0 (disabled).

The flag sets `CallConfig::enable_l2_swimlane` to the chosen
level. The host then allocates the per-core / per-thread shared
region and publishes its base address through
`kernel_args.l2_swimlane_data_base`. AICore writes timing into
per-task WIP slots; AICPU commits the records on FIN. Per-task
dispatch/finish timestamps and fanout are recorded only at
level >= 2, scheduler phase records only at level >= 3, and
orchestrator phase records only at level >= 4.

The JSON output `"l2_swimlane_level"` field is the captured perf_level:
`1` = AICore timing only, `2` = +dispatch/fanout,
`3` = +scheduler phases, `4` = +orchestrator phases.

`--rounds > 1` collects only on the **first** round so warm-up
runs are not double-counted.

### 3.2 Output

The raw artifact lands under the per-task output prefix
(`CallConfig::output_prefix`, set by
`scene_test.py::_build_output_prefix` to
`outputs/<ClassName>_<case>_<YYYYMMDD_HHMMSS>/` for SceneTest
runs):

```text
<output_prefix>/
├── l2_swimlane_records.json     # raw runtime output
├── name_map_<case>.json     # optional func_id → name mapping
└── merged_swimlane.json     # Perfetto trace (added by converter)
```

Filenames are fixed (no per-file timestamp) — the directory is the
per-task uniqueness boundary.

`l2_swimlane_records.json` carries the raw records — this is the file
you pass to `swimlane_converter`. Important fields per task:

| Field | Meaning |
| ----- | ------- |
| `task_id` | Runtime task id (`(ring_id << 32) \| local_id`); also exposed split as`ring_id` |
| `func_id` | Kernel function id |
| `core_id` / `core_type` | Physical core index and `"aic"` / `"aiv"` string |
| `start_time_us` / `end_time_us` / `duration_us` | AICore execution window in microseconds |
| `dispatch_time_us` | AICPU timestamp when this task was dispatched (filled at level >= 2) |
| `finish_time_us` | AICPU timestamp when AICPU observed FIN (filled at level >= 2) |
| `fanout[]` / `fanout_count` | Successor task ids (level >= 2), used by Perfetto dependency arrows |

Phase records (per scheduler thread, level >= 3 for
`aicpu_scheduler_phases[]` and level >= 4 for
`aicpu_orchestrator_phases[]`):

| Field | Meaning |
| ----- | ------- |
| `start_time_us` / `end_time_us` | Phase start / end timestamps in microseconds |
| `phase` | Lowercase phase name. Scheduler: `complete` / `dispatch` (`scan` / `idle` may appear in legacy captures and a5; both are dropped by the parser). Orchestrator: `orch_submit` — one record per `submit_task()` / `alloc_tensors()` call spanning its full `[start, end]` window. Legacy per-sub-step strings (`orch_sync` / `orch_alloc` / `orch_params` / `orch_lookup` / `orch_insert` / `orch_fanin`) may appear in old captures. |
| `loop_iter` (scheduler) / `submit_idx` (orchestrator) | Iteration / submit-call counter for the producing thread |
| `tasks_processed` (scheduler) / `task_id` (orchestrator) | Phase-specific union field |
| `pop_hit` / `pop_miss` (dispatch only) | Ready-queue pop deltas since the previous dispatch emit |

`core_to_thread[]` (level >= 3) maps `core_id` (array index) to the
scheduler thread index that retired that core's tasks (`-1` =
unassigned).

### 3.3 Convert and view in Perfetto

`swimlane_converter` turns the raw records into a Perfetto trace
and produces a per-function task-execution summary:

```bash
# Auto-detects the latest outputs/*/l2_swimlane_records.json
python -m simpler_setup.tools.swimlane_converter

# Pin to a specific case + add func_id → name mapping
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case>_<ts>/l2_swimlane_records.json \
    --func-names outputs/<case>_<ts>/name_map_<case>.json

# Custom output path
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case>_<ts>/l2_swimlane_records.json -o my_trace.json
```

The output is `outputs/<case>_<ts>/merged_swimlane.json` (or your
`-o` override). Open <https://ui.perfetto.dev/> and drag the file
in. The trace contains:

- **AICore View** — one swim-lane per AICore (AIC / AIV / mix
  channel). Each task shows `func_name(t<task_id>)`; dependency
  arrows follow `fanout[]`.
- **AICPU View** — scheduler thread lanes with per-iteration
  phase blocks coloured by `phase`.
- **AICPU Scheduler** — orchestrator phase summary at the top.

When the run also emitted a device log (`device-*` file under
`outputs/`), `swimlane_converter` resolves the nearest log by
mtime and runs `sched_overhead_analysis` automatically. The
report is printed to stdout; it correlates AICPU phase records
with the device log to attribute each scheduler iteration to a
specific overhead source.

### 3.4 Adding human-readable names

Without name mapping, tasks show as `func_<id>(t<task_id>)`. To
get readable lane labels, add a `name` field to your CALLABLE
spec:

```python
@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestPagedAttention(SceneTestCase):
    CALLABLE = {
        "orchestration": {
            "source": "kernels/orchestration/orch.cpp",
            "function_name": "build_paged_attention_graph",
            "name": "PagedAttn",                          # optional
            "signature": [D.IN, D.IN, D.IN, D.OUT],
        },
        "incores": [
            {"func_id": 0, "name": "QK", "source": "...", "core_type": "aic"},
            {"func_id": 1, "name": "SF", "source": "...", "core_type": "aiv"},
            {"func_id": 2, "name": "PV", "source": "...", "core_type": "aic"},
        ],
    }
```

SceneTest extracts this into `<output_prefix>/name_map_<case>.json`
and passes it to `swimlane_converter` automatically. See
[profiling-name-map.md](../profiling-name-map.md) for the full
schema and L3 example.

## 4. Capabilities

What the swimlane shows:

- **Per-task wall-clock placement.** Where each task ran on which
  AICore, with `start_time_us` / `end_time_us` / `duration_us` in
  microseconds (converted from device cycles).
- **Dispatch and finish overhead.** `dispatch_time_us` and
  `finish_time_us` come from AICPU, so the gap between
  `dispatch_time_us` and `start_time_us` is the AICPU→AICore
  hand-off latency, and the gap between `end_time_us` and
  `finish_time_us` is the FIN-observation latency.
- **Dependency chains.** `fanout[]` lets Perfetto draw arrows
  between predecessor and successor tasks.
- **Scheduler-loop time decomposition.** Per-iteration AICPU
  phase records show how long the scheduler spent in each of
  its two work phases (complete / dispatch); idle is recovered
  from the gap between records.
- **Orchestrator overhead breakdown.** Per-submit envelope
  records (`orch_submit`) pin "which submit is slow"; cumulative
  cycle counts in the cold-path device log (`g_orch_*_cycle`)
  cover the per-sub-step breakdown for "which sub-step dominates".

## 5. Design Highlights

### 5.1 Common interfaces

`kernel_args.l2_swimlane_data_base` is the single device-side handle
host publishes for the run. The shared region carries a fixed
`L2SwimlaneDataHeader` plus per-core / per-thread state (same struct
shape on both architectures):

```text
L2SwimlaneDataHeader                                (host init, device R/W)
├── queues  [MAX_AICPU_THREADS][READYQUEUE_SIZE]
├── queue_heads / queue_tails (per-thread)
└── num_cores

L2SwimlaneAicpuTaskPool[num_cores]                    (per-core AICPU pool state)
├── free_queue {buffer_ptrs[SLOT_COUNT], head, tail}
├── current_buf_ptr           (AICPU active L2SwimlaneAicpuTaskBuffer*)
├── aicore_ring_ptr           (legacy; kept for ABI continuity)
├── total_record_count
├── dropped_record_count
└── mismatch_record_count     (legacy; no longer written)

L2SwimlaneAicoreTaskPool[num_cores]              (per-core AICore pool state)
├── head {current_buf_ptr, current_buf_seq,     (single 64B cache line;
│         total_record_count,                    AICPU writes, AICore dcci-
│         dropped_record_count}                  polls per task; AICPU bumps
│                                                current_buf_seq on rotation
│                                                so AICore detects the change)
└── free_queue {buffer_ptrs[SLOT_COUNT], head, tail}

[L2SwimlaneAicoreTaskBuffer × PLATFORM_AICORE_BUFFERS_PER_CORE per core]
└── L2SwimlaneAicoreTaskRecord records[PLATFORM_AICORE_BUFFER_SIZE]  (1024 records, 32B each)

a2a3 layout (post-#942):
[L2SwimlaneAicpuSchedPhasePool[num_sched_phase_threads]]  (per-AICPU-thread)
└── L2SwimlaneAicpuSchedPhaseBuffer × PROF_BUFFERS_PER_THREAD; alias of TaskPool

[L2SwimlaneAicpuOrchPhasePool[num_orch_phase_threads]]    (per-AICPU-thread)
└── L2SwimlaneAicpuOrchPhaseBuffer × PROF_BUFFERS_PER_THREAD; alias of TaskPool

a5 layout (pre-split; pending port to the a2a3 shape above):
[L2SwimlaneAicpuPhasePool[num_phase_threads]]   (single unified pool, gated
                                                 by L2SwimlaneAicpuPhaseHeader
                                                 + L2_SWIMLANE_AICPU_PHASE_MAGIC)

(a2a3 phase metadata — num_sched_phase_threads, num_orch_phase_threads,
 num_phase_cores, core_to_thread[] — lives inside L2SwimlaneDataHeader,
 not a separate cache line; the legacy header + magic gate were removed
 in #941. Host gates on num_{sched,orch}_phase_threads > 0.)
```

Task records are identical across architectures:

- `L2SwimlaneAicpuTaskRecord` — per-task AICPU-owned fields (task_id, dispatch_time,
  finish_time, func_id, core_type, reg_task_id), 64-byte aligned.
  `reg_task_id` is the join key against the matching AICore record.
- `L2SwimlaneAicoreTaskRecord` — slim AICore-only record (start, end, task_id),
  32 bytes; AICore writes one per task into its currently-active
  per-core buffer.

Phase records diverge — a2a3 split them into two type-tagged streams
in #942, a5 still uses the legacy unified shape:

- a2a3:
  - `L2SwimlaneAicpuSchedPhaseRecord` (40 B) — per-iteration scheduler
    phase; kind ∈ {Complete, Dispatch} + loop_iter + tasks_processed +
    pop_hit / pop_miss deltas.
  - `L2SwimlaneAicpuOrchPhaseRecord` (32 B) — per-submit orchestrator
    envelope; task_id + submit_idx + start/end.
- a5:
  - `L2SwimlaneAicpuPhaseRecord` (40 B) — single record type carrying
    both sched and orch via a `phase_id` discriminator; pending port to
    the split shape.

`swimlane_converter` shape-detects per source and produces the same
output JSON for both. On a2a3 the orch stream replaces the per-sub-step
records folded into ORCH_SUBMIT; there is no separate shared-memory
aggregate. The run-window envelope is emitted to device log via
`LOG_INFO_V9 "orch_start=… orch_end=… orch_cost=…"`.

**Producer/consumer protocol on AICore (AICore-as-producer with rotation).**
AICore writes a slim `L2SwimlaneAicoreTaskRecord` into its currently-active per-core
`L2SwimlaneAicoreTaskBuffer` at `records[slot_within_buf++]`. The active buffer is
published via a per-core `L2SwimlaneActiveHead` cache line (`current_buf_ptr` +
`current_buf_seq` + counters); AICore `dcci`'s it per task — cheap relative
to the baseline `dcci(payload, ENTIRE_DATA_CACHE)` it already pays per
task. AICPU drives rotation: immediately before each `write_reg(DATA_MAIN_BASE)`
for task `K`, if `K % PLATFORM_AICORE_BUFFER_SIZE == 0`, AICPU enqueues
the current buffer to the per-thread ready queue (kind `AicoreTask`),
pops the next from `L2SwimlaneAicoreTaskPool::free_queue`, and bumps
`L2SwimlaneActiveHead::current_buf_seq`. AICore detects the bumped seq on
its next task's `dcci`, refreshes its local cache, and resets its slot
counter to 0.

**Race safety.** The runtime's completion-before-dispatch invariant
guarantees all tasks `< K` have FIN'd before AICPU dispatches `K`, so
by the time AICPU enqueues the old buffer, AICore has finished writing
(and `dcci+dsb`'d) records for all those tasks. No spin-wait, no
cross-direction read on the hot path.

**Sizing.** `PLATFORM_AICORE_BUFFER_SIZE = 1024` (power of two, modulo
lowers to AND) and `PLATFORM_AICORE_BUFFERS_PER_CORE = 4` (1 active +
3 recycled). Host-side `BufferPoolManager` refills the recycled pool
from the ready queue while the session runs, so session length is
bounded only by how fast the host drains — not by the per-core buffer
sum.

**Measured impact.** Hardware bench on a2a3 paged_attention_unroll
Case1 with swimlane=4: rotation design delivers sched −4 µs / orch −19 µs
vs the upstream/main baseline, comparable to the no-rotation predecessor
(which had this PR's earlier commit; the rotation adds about 3 µs
sched overhead per session as price for unbounded session length).

### 5.2 a2a3 — shared-memory streaming

`halHostRegister` maps device memory into host virtual address
space so the host can read device buffers directly.
`L2SwimlaneCollector` runs two background threads on top of a
[`BufferPoolManager<L2SwimlaneModule>`](../src/common/platform/include/host/buffer_pool_manager.h):
a mgmt thread that polls SPSC ready queues and recycles full
buffers **while kernels are still executing**, plus a poll
thread that drains the L2 hand-off queue into
`on_buffer_collected`.

`L2SwimlaneModule` declares four buffer kinds going through one ready
queue per AICPU thread:

- **kind 0** `AicpuTask`        — per-core `L2SwimlaneAicpuTaskBuffer` (AICPU writes).
- **kind 1** `AicpuSchedPhase`  — per-thread `L2SwimlaneAicpuSchedPhaseBuffer` (AICPU writes).
- **kind 2** `AicpuOrchPhase`   — per-thread `L2SwimlaneAicpuOrchPhaseBuffer` (AICPU writes).
- **kind 3** `AicoreTask`       — per-core `L2SwimlaneAicoreTaskBuffer` (AICore writes,
  AICPU enqueues on rotation).

Each `ReadyQueueEntry::kind` carries the discriminator. This is the
only multi-kind module in the current framework — PMU and TensorDump
are single-kind.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ L2SwimlaneCollector          │               │ AICPU + AICore           │
│                          │               │                          │
│ initialize(prefix)       │  alloc +      │ AICore on task end:      │
│   rtMalloc + halRegister │──register────>│   write slim record into │
│   pre-fill free queues   │              │   AicoreTaskBuffer (kind │
│   for kinds 0/1/2/3      │               │   3); AICPU rotates it   │
│                          │               │                          │
│ start(tf)                │               │ AICPU on FIN:            │
│   ┌────────────────────┐ │ SPSC ready    │   commit AicpuTask       │
│   │ mgmt thread        │ │ queues        │   record (kind 0); fill  │
│   │ (BufferPool driver)│ │<──4 kinds────<│   func_id / dispatch /   │
│   │   poll ready queue │<┼──multiplexed──│   finish; rotate buffer  │
│   │   recycle buffers  │─┼──free queue──>│   when full              │
│   └────────────────────┘ │               │ AICPU scheduler thread:  │
│   ┌────────────────────┐ │               │   per work iter: write   │
│   │ poll thread        │ │               │   SchedPhaseRecord       │
│   │   reads via host   │ │ shared mem    │   (kind 1). Per submit:  │
│   │   mapping; copies  │<┼──mapping─────<│   write OrchPhaseRecord  │
│   │   to host vectors  │ │               │   (kind 2).              │
│   └────────────────────┘ │               │                          │
│ stop()                   │               │                          │
│   join mgmt → join poll  │               │                          │
│ read_phase_header_metadata()             │                          │
│ reconcile_counters()     │               │                          │
│ export_swimlane_json()   │               │                          │
│   → l2_swimlane_records.json │               │                          │
└──────────────────────────┘               └──────────────────────────┘
```

**Lifecycle** (`device_runner.cpp`):

```text
init_l2_swimlane()
  l2_swimlane_collector_.initialize(num_aicore, ..., output_prefix_)
  kernel_args_.args.l2_swimlane_data_base = l2_swimlane_collector_.get_l2_swimlane_shm_device_ptr()
start(tf)                          ← spawn mgmt + poll threads
launch AICPU / AICore
rtStreamSynchronize
stop()                             ← join mgmt → join poll
read_phase_header_metadata()       ← single-shot read of the
                                     core→thread mapping
reconcile_counters()               ← three-bucket accounting for both
                                     PERF and PHASE pools (total /
                                     collected / dropped); any non-zero
                                     current_buf_ptr is a flush bug
export_swimlane_json()             ← writes <output_prefix>/l2_swimlane_records.json
finalize(unregister, free)
```

[`L2SwimlaneCollector`](../src/a2a3/platform/include/host/l2_swimlane_collector.h)
on a2a3 inherits from
[`profiling_common::ProfilerBase<L2SwimlaneCollector, L2SwimlaneModule>`](../src/common/platform/include/host/profiler_base.h):
the base class owns the mgmt thread, the poll thread, and the
`BufferPoolManager<L2SwimlaneModule>` they share. `L2SwimlaneCollector`
supplies the L2-specific pieces — the `L2SwimlaneModule` trait
(notably `kBufferKinds = 4` and `kind_of()`), `initialize` that
allocates and pre-fills all four kinds of free queues, an
`on_buffer_collected` callback that branches on `info.type` across
`AICPU_TASK` / `AICPU_SCHED_PHASE` / `AICPU_ORCH_PHASE` / `AICORE_TASK`
to copy into the right per-core or per-thread vector, plus
`read_phase_header_metadata` /
`reconcile_counters` / `export_swimlane_json` / `finalize`. The
mgmt/poll threading and `Module` trait pattern are shared with
PMU and TensorDump — see
[profiling-framework.md](../profiling-framework.md) for the
framework reference.

### 5.3 a5 — same framework, host-shadow transport

a5's `L2SwimlaneCollector` derives from
`ProfilerBase<L2SwimlaneCollector, L2SwimlaneModule>` and shares the
mgmt + poll thread structure with a2a3. The single behavioral
deviation from §5.2 is the **transport channel**: a5 has no
`halHostRegister`, so each device buffer is paired with a
host-shadow `malloc()` and the mgmt loop synchronizes the two via
`profiling_copy.h` (`rtMemcpy` onboard, plain `memcpy` in sim).

The AICore-side write target is a per-core, **stable**
`L2SwimlaneAicoreRing` (`dual_issue_slots[PLATFORM_L2_AICORE_RING_SIZE]`)
allocated once by the host and addressed via
`L2SwimlaneAicpuTaskPool::aicore_ring_ptr` (AICPU side) and
`KernelArgs::aicore_l2_swimlane_ring_addrs[block_idx]` forwarded into
`set_aicore_l2_swimlane_ring()` by `KERNEL_ENTRY` (AICore side). The ring
address never changes during a run, so AICore's write address is
decoupled from the AICPU's rotating `L2SwimlaneAicpuTaskBuffer`. Buffer rotation is
internal to `l2_swimlane_aicpu_complete_task` when `records[count]` hits
`PLATFORM_PROF_BUFFER_SIZE`. The runtime `Handshake` carries no
profiling fields.

The framework's `MemoryOps` therefore carries five callbacks on
a5 (`alloc` / `reg` / `free_` / `copy_to_device` /
`copy_from_device`); the mgmt loop mirrors the entire shm region
(`L2SwimlaneDataHeader` + per-core `L2SwimlaneAicpuTaskPool` +
`L2SwimlaneAicpuPhaseHeader` + per-thread `L2SwimlaneAicpuPhasePool`)
device → host at the top of every tick, then pushes back only the
fields host actually modified (advanced `queue_heads[q]`, refilled
`free_queue.tail` and `buffer_ptrs[slot]`) via
`BufferPoolManager::write_range_to_device`. The bulk
`mirror_shm_to_device` is deliberately **not** called from the mgmt
loop: it would race with AICPU writes to device-only fields
(`current_buf_ptr`, `total/dropped/mismatch` counters, `queue_tails`,
`free_queue.head`, `L2SwimlaneAicpuPhaseHeader::magic`,
`L2SwimlaneAicpuPhaseHeader::core_to_thread[]`) and roll them back to
whatever the host shadow held at the start of the tick. Per-buffer
payloads (`L2SwimlaneAicpuTaskBuffer` / `L2SwimlaneAicpuPhaseBuffer`)
are pulled on demand inside `ProfilerAlgorithms::process_entry` after
a popped ready-entry resolves to its host shadow. `BufferPoolManager`'s
`release_owned_buffers` frees the device pointer via the
collector's `release_fn` and the paired shadow via `std::free()`.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ L2SwimlaneCollector          │               │ AICPU + AICore           │
│   : ProfilerBase<...>    │               │                          │
│                          │               │                          │
│ initialize()             │  alloc + reg  │ AICore on task end:      │
│   rtMalloc shm           │──+ shadow────>│   write timing into      │
│   per-core L2SwimlaneAicpuTaskBuffer  │   memset 0    │   per-core ring slot     │
│   per-core AicoreRing    │   + push 0s   │   dual_issue_slots[      │
│   per-thread L2SwimlaneAicpuPhaseBuffer │               │     task_id & 1]         │
│   register_mapping(s)    │               │                          │
│   set_memory_context     │               │ AICPU on FIN:            │
│                          │               │   read ring slot →       │
│                          │               │   commit into records[]  │
│ start(thread_factory)    │               │                          │
│   mgmt_thread starts     │               │ AICPU per-thread flush   │
│   poll_thread starts     │               │   on exit: enqueue       │
│                          │               │   current_buf_ptr →      │
│ mgmt every 10us tick:    │               │   ready_queue            │
│   copy_from_device(shm)  │<──memcpy─────<│                          │
│   for each ready entry:  │               │                          │
│     copy buf from device │<──memcpy─────<│                          │
│     resolve host ptr     │               │                          │
│     push to L2 ready_q   │               │                          │
│   advance queue_heads,   │               │                          │
│     refill free_queues   │               │                          │
│   write_range_to_device  │──memcpy──────>│                          │
│     for each modified    │               │                          │
│     field                │               │                          │
│                          │               │                          │
│ poll thread:             │               │                          │
│   wait_pop_ready          │               │                          │
│   on_buffer_collected →  │               │                          │
│     copy_perf/phase      │               │                          │
│   notify_copy_done       │               │                          │
│                          │               │                          │
│ rtStreamSynchronize      │               │                          │
│ stop()                   │               │                          │
│   join mgmt + poll       │               │                          │
│ read_phase_header_meta   │               │                          │
│ reconcile_counters       │               │                          │
│   sanity-check leftovers │               │                          │
│   + 3-bucket cross-check │               │                          │
│ export_swimlane_json()   │               │                          │
│ finalize(free)           │               │                          │
└──────────────────────────┘               └──────────────────────────┘
```

**Lifecycle** (`device_runner.cpp`):

```text
init_l2_swimlane()
  l2_swimlane_collector_.initialize(num_aicore, ..., output_prefix_)
  kernel_args_.args.l2_swimlane_data_base = l2_swimlane_collector_.get_l2_swimlane_setup_device_ptr()
  kernel_args_.args.aicore_l2_swimlane_ring_addrs =
      l2_swimlane_collector_.get_aicore_ring_addrs_device_ptr()
l2_swimlane_collector_.start(thread_factory)   ← mgmt + poll threads
launch AICPU / AICore
rtStreamSynchronize
l2_swimlane_collector_.stop()                  ← join mgmt + poll, drain final batch
l2_swimlane_collector_.read_phase_header_metadata()
l2_swimlane_collector_.reconcile_counters()    ← sanity-check + 3-bucket cross-check
l2_swimlane_collector_.export_swimlane_json()
l2_swimlane_collector_.finalize()
```

[`L2SwimlaneCollector`](../src/a5/platform/include/host/l2_swimlane_collector.h)
on a5 inherits the same CRTP base
([`profiling_common::ProfilerBase`](../src/common/platform/include/host/profiler_base.h))
as a2a3 and parameterizes
[`BufferPoolManager`](../src/common/platform/include/host/buffer_pool_manager.h)
with `L2SwimlaneModule` (`kBufferKinds = 2`). The only a5-specific
glue is the 5-callback `MemoryOps` and the per-tick shm mirror.

a5's per-thread AICPU flush hooks (`l2_swimlane_aicpu_flush` /
`l2_swimlane_aicpu_flush_phase_buffers`) are the only data path on the
records side — host never reads from `current_buf_ptr` to recover
records. `reconcile_counters` is purely passive: it logs an error if
any `current_buf_ptr` is non-zero with a non-empty buffer (a
device-flush bug), then runs the three-bucket cross-check
`collected + dropped + mismatch == device_total` per pool (PERF +
PHASE), same shape as a2a3.

### 5.4 a2a3 vs a5 at a glance

| Aspect | a2a3 | a5 |
| ------ | ---- | -- |
| Task record | `L2SwimlaneAicpuTaskRecord` (64 B) + `L2SwimlaneAicoreTaskRecord` (32 B) | identical |
| Phase record | split: `L2SwimlaneAicpuSchedPhaseRecord` (40 B) + `L2SwimlaneAicpuOrchPhaseRecord` (32 B) | unified `L2SwimlaneAicpuPhaseRecord` (40 B, `phase_id`-tagged); pending port |
| AICore WIP-slot protocol | identical | |
| AICPU commit on FIN | identical | |
| Buffer model | rotating pool (free + ready queues) per kind | identical |
| Ready queue | per-AICPU-thread, multiplexes 4 kinds via `ReadyQueueEntry::kind` | per-AICPU-thread, 2 kinds via `is_phase` |
| Host threads | mgmt + poll, streams during execution | identical |
| Host-class shape | `ProfilerBase<L2SwimlaneCollector, L2SwimlaneModule>` (`kBufferKinds = 4`) | same base, `kBufferKinds = 2` |
| Host transport | `halHostRegister` shared memory | host-shadow `malloc` + per-tick `rtMemcpy`/`memcpy` |
| `MemoryOps` callbacks | 3 (`alloc`, `reg`, `free_`) | 5 (+ `copy_to_device`, `copy_from_device`) |
| `reconcile_counters` | passive cross-check (collected + dropped + mismatch == device_total) | identical |
| Lifecycle | `initialize` → `start` → `stop` → `read_phase_header_metadata` → `reconcile_counters` → `export_swimlane_json` → `finalize` | identical |

## 6. Overhead

L2 swimlane is opt-in and zero-overhead when disabled — without
`--enable-l2-swimlane` neither host nor device allocates the L2
perf shared region and the timing-write code paths are skipped.

When enabled, the dominant per-task overhead is:

- `get_sys_cnt()` reads at task start / end on AICore.
- Two cache-line writes into the WIP slot.
- The AICPU commit on FIN, which copies the WIP record into the
  ring buffer plus a few metadata fields.

Phase-record overhead (only at `--enable-l2-swimlane >= 3`):

- a2a3 — one 40 B `L2SwimlaneAicpuSchedPhaseRecord` per work-emitting
  scheduler iteration (Complete + Dispatch, idle iters do not emit),
  plus one 32 B `L2SwimlaneAicpuOrchPhaseRecord` per `submit_task()`.
- a5 — one 40 B `L2SwimlaneAicpuPhaseRecord` per emitted phase
  (legacy unified shape).

Both architectures drain buffers concurrently with execution via the
mgmt + poll thread pair; a5 additionally pays per-tick
`rtMemcpy`/`memcpy` round-trips to keep the host shadow in sync,
which overlap with device execution.

`--rounds > 1` collects only on the first round so the steady-state
benchmark is not perturbed.

## 7. Limitations

### 7.1 a2a3

- Records can be lost on device when both the per-core / per-thread
  free queue and the host's recycled pool are empty for too long.
  AICPU increments `dropped_record_count` and continues; the host's
  `reconcile_counters()` reports `collected + dropped == total` per
  pool. If `dropped > 0`, raise `PLATFORM_PROF_BUFFERS_PER_CORE` /
  `PLATFORM_PROF_BUFFERS_PER_THREAD` so the recycle pool has more
  headroom.
- A non-zero `current_buf_ptr` after `stop()` is logged as ERROR
  and never recovered — host treats device flush as the sole data
  path. Such a leftover indicates an AICPU flush bug, not a tail
  loss to tune around.
- `a2a3sim` exercises the export pipeline; the simulated device
  clock is not realistic for absolute-timing analysis. Use real
  hardware for steady-state numbers.

### 7.2 a5

- Each per-core `L2SwimlaneAicpuTaskBuffer` and per-thread `L2SwimlaneAicpuPhaseBuffer` is
  fixed-size. Tasks past `PLATFORM_PROF_BUFFER_SIZE` per core (and
  phases past `PLATFORM_PHASE_RECORDS_PER_THREAD` per thread) are
  silently dropped via AICPU early return; the host surfaces the
  count in the finalize log line. Raise the constants in
  [platform_config.h](../src/a5/platform/include/common/platform_config.h)
  for workloads that exceed them.
- `a5sim` exercises the export pipeline; the simulated device
  clock is not realistic for absolute-timing analysis.

### 7.3 Common

- Only the **first** round records when `--rounds > 1` is in use.
- The current implementation captures incore-level scope only —
  L3 composition and orchestrator-internal sub-tasks are visible
  through the orchestrator phase summary, not as nested swimlane
  scopes.

## 8. FAQ / Debug Guide

**No `l2_swimlane_records.json` produced.** Check that
`--enable-l2-swimlane` was passed. Verify `<output_prefix>` exists
in the run log; if `--rounds > 1`, only the first round records.

**`merged_swimlane.json` is missing.** `swimlane_converter` runs
automatically after a SceneTest with `--enable-l2-swimlane`; if it
did not, run it manually:

```bash
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_swimlane_records.json
```

**Tasks show as `func_<id>` instead of human names.** The
CALLABLE spec lacks `"name"` fields, or
`name_map_<case>.json` was not produced. See [profiling-name-map.md](../profiling-name-map.md).

**Some tasks missing from the swimlane.** Likely dropped on device
because the buffer pool ran out. On a2a3 check
`reconcile_counters()` output for non-zero `dropped`; raise
`PLATFORM_PROF_BUFFERS_PER_CORE` /
`PLATFORM_PROF_BUFFERS_PER_THREAD`. On a5 raise
`PLATFORM_PROF_BUFFER_SIZE`.

**`current_buf_ptr` non-empty at finalize on a2a3.** The host logs
this as ERROR and does not recover. AICPU did not flush its
active L2 swimlane buffer at run end. Check the AICPU flush path runs
for every thread that produced records.

**Phase records empty.** Either the runtime did not emit phase
data (only `tensormap_and_ringbuffer` does, and only when phase init
ran — on a2a3 gated on
`L2SwimlaneDataHeader::num_sched_phase_threads > 0` (sched) or
`num_orch_phase_threads > 0` (orch); on a5 gated on
`L2SwimlaneAicpuPhaseHeader::magic`), or the host did not pre-zero
those fields. Verify the runtime calls `l2_swimlane_aicpu_init_phase()`
in its scheduler init path; check the host's
`L2SwimlaneCollector::initialize` zero-inits the relevant metadata
fields.

**`dispatch_time_us` < `finish_time_us` mismatch.** Verify the runtime
overwrites `task_id` with the full encoding on FIN
(`tensormap_and_ringbuffer` does
`(ring_id << 32) | local_id`); a half-filled record means AICore
wrote the WIP slot but AICPU never committed.

**Scheduler-overhead deep-dive missing from converter output.**
The converter runs `sched_overhead_analysis` only when a device
log is resolvable. Pass `-d <device-id>` or place a `device-*`
log under `outputs/` close in time to the `l2_swimlane_records.json`
mtime; see `simpler_setup/tools/README.md` for the resolver
rules.

## 9. Related docs

- [profiling-framework.md](../profiling-framework.md) — shared
  host-side collector framework (a2a3 only).
- [profiling-name-map.md](../profiling-name-map.md) — `func_id` →
  human name mapping for swimlane labels.
- [chip-level-arch.md](../chip-level-arch.md) — host / AICPU /
  AICore program boundaries this feature spans.
- [task-flow.md](../task-flow.md) — where AICPU dispatch and
  completion sit in the per-task state machine.
- `simpler_setup/tools/README.md` — `swimlane_converter` /
  `sched_overhead_analysis` CLI reference.
