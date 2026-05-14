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

- **Per-task AICore timing** — `start_time`, `end_time`,
  `duration`, plus AICPU-stamped `dispatch_time` / `finish_time`.
- **Per-task fanout chain** — successor `task_id`s recorded in
  the L2 record so dependency arrows show up in the Perfetto
  view.
- **AICPU scheduler phases** — per-iteration breakdown into
  `SCHED_COMPLETE` / `SCHED_DISPATCH` / `SCHED_SCAN` /
  `SCHED_IDLE_WAIT`.
- **Orchestrator phase summary** — cumulative cycle counts for
  the orchestrator's nine sub-steps (sync / alloc / params /
  lookup / heap / insert / fanin / finalize / scope_end).
- **Standard outputs** — raw `l2_perf_records.json`, plus a
  Perfetto-loadable `merged_swimlane_*.json` produced by
  `swimlane_converter`.

Enable in one line:

```bash
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-l2-swimlane
```

## 3. How to Use

### 3.1 Enable L2 swimlane

```bash
# Standalone runner
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-l2-swimlane
python tests/st/<case>/test_<name>.py -p a5sim --enable-l2-swimlane

# pytest
pytest tests/st/<case> --platform a2a3 -d 0 --enable-l2-swimlane
pytest examples/a5/host_build_graph/vector_example --platform a5sim --enable-l2-swimlane
```

The flag flips `CallConfig::enable_l2_swimlane`. The host then
allocates the per-core / per-thread shared region and publishes
its base address through `kernel_args.l2_perf_data_base`. AICore
writes timing into per-task WIP slots; AICPU commits the records
on FIN and emits its own per-iteration phase records.

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
├── l2_perf_records.json     # raw runtime output
├── name_map_<case>.json     # optional func_id → name mapping
└── merged_swimlane.json     # Perfetto trace (added by converter)
```

Filenames are fixed (no per-file timestamp) — the directory is the
per-task uniqueness boundary.

`l2_perf_records.json` carries the raw records — this is the file
you pass to `swimlane_converter`. Important fields per task:

| Field | Meaning |
| ----- | ------- |
| `task_id` | Runtime task id, hex (low 32 bits = AICore register token; full 64 bits filled by AICPU) |
| `func_id` | Kernel function id |
| `core_type` | `0` = AIC, `1` = AIV |
| `start_time` / `end_time` / `duration` | AICore device-clock cycles (`get_sys_cnt`) |
| `dispatch_time` | AICPU timestamp when this task was dispatched |
| `finish_time` | AICPU timestamp when AICPU observed FIN |
| `fanout[]` / `fanout_count` | Successor task ids, used by Perfetto dependency arrows |

Phase records (per scheduler thread):

| Field | Meaning |
| ----- | ------- |
| `start_time` / `end_time` | Phase start / end timestamps |
| `loop_iter` | Scheduler loop iteration number |
| `phase_id` | One of `SCHED_COMPLETE` / `SCHED_DISPATCH` / `SCHED_SCAN` / `SCHED_IDLE_WAIT`, or `ORCH_*` for orchestrator phases |
| `tasks_processed` (scheduler) / `task_id` (orchestrator) | Phase-specific union field |

### 3.3 Convert and view in Perfetto

`swimlane_converter` turns the raw records into a Perfetto trace
and produces a per-function task-execution summary:

```bash
# Auto-detects the latest outputs/*/l2_perf_records.json
python -m simpler_setup.tools.swimlane_converter

# Pin to a specific case + add func_id → name mapping
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case>_<ts>/l2_perf_records.json \
    --func-names outputs/<case>_<ts>/name_map_<case>.json

# Custom output path
python -m simpler_setup.tools.swimlane_converter \
    outputs/<case>_<ts>/l2_perf_records.json -o my_trace.json
```

The output is `outputs/<case>_<ts>/merged_swimlane.json` (or your
`-o` override). Open <https://ui.perfetto.dev/> and drag the file
in. The trace contains:

- **AICore View** — one swim-lane per AICore (AIC / AIV / mix
  channel). Each task shows `func_name(t<task_id>)`; dependency
  arrows follow `fanout[]`.
- **AICPU View** — scheduler thread lanes with per-iteration
  phase blocks coloured by `phase_id`.
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
  AICore, with start / end / duration in device cycles.
- **Dispatch and finish overhead.** `dispatch_time` and
  `finish_time` come from AICPU, so the gap between
  `dispatch_time` and `start_time` is the AICPU→AICore
  hand-off latency, and the gap between `end_time` and
  `finish_time` is the FIN-observation latency.
- **Dependency chains.** `fanout[]` lets Perfetto draw arrows
  between predecessor and successor tasks.
- **Scheduler-loop time decomposition.** Per-iteration AICPU
  phase records show how long the scheduler spent in each of
  its four phases.
- **Orchestrator overhead breakdown.** Cumulative cycle counts
  for the nine orchestrator sub-steps (sync, alloc, params,
  lookup, heap, insert, fanin, finalize, scope_end), useful for
  spotting where graph-build cost lives.

## 5. Design Highlights

### 5.1 Common interfaces

`kernel_args.l2_perf_data_base` is the single device-side handle
host publishes for the run. The shared region carries a fixed
`L2PerfDataHeader` plus per-core / per-thread state (same struct
shape on both architectures):

```text
L2PerfDataHeader                                (host init, device R/W)
├── queues  [MAX_AICPU_THREADS][READYQUEUE_SIZE]
├── queue_heads / queue_tails (per-thread)
└── num_cores

L2PerfBufferState[num_cores]                    (per-core perf state)
├── free_queue {buffer_ptrs[SLOT_COUNT], head, tail}
├── current_buf_ptr           (AICPU active L2PerfBuffer*)
├── aicore_ring_ptr           (stable L2PerfAicoreRing*, host writes once)
├── total_record_count
├── dropped_record_count
└── mismatch_record_count     (ring slot / task_id invariant violations)

[L2PerfAicoreRing[num_cores]]                   (stable AICore staging)
└── L2PerfRecord dual_issue_slots[PLATFORM_L2_AICORE_RING_SIZE]

[AicpuPhaseHeader + PhaseBufferState[num_threads]]  (optional)
├── magic / num_sched_threads
├── orch_summary  (cumulative orchestrator cycles)
└── per-thread phase buffers (PhaseBufferState aliases L2PerfBufferState;
                              `aicore_ring_ptr` / `mismatch_record_count`
                              unused for PHASE)
```

The records themselves are identical across architectures:

- `L2PerfRecord` — per-task timing + fanout, 64-byte aligned.
- `AicpuPhaseRecord` — per-iteration scheduler / orchestrator
  phase, 32 bytes.
- `AicpuOrchSummary` — one-shot cumulative orchestrator cycles.

This is the key reason a single `swimlane_converter` consumes
both architectures' output unchanged.

**Producer/consumer protocol on AICore.** AICore writes per-task
timing into a stable per-core `L2PerfAicoreRing` at
`dual_issue_slots[reg_task_id % PLATFORM_L2_AICORE_RING_SIZE]`.
The ring address is published once via a per-core table on
`KernelArgs` (a2a3: `aicore_ring_addr`; a5:
`aicore_l2_perf_ring_addrs`), forwarded by `KERNEL_ENTRY` into
platform-owned AICore state, and never reassigned — so AICore is
fully decoupled from any AICPU-side records-buffer rotation. AICPU,
on observing FIN, validates the slot's register token, copies the slot
record into the current `L2PerfBuffer::records[count]`, fills
`func_id` / `core_type` / `dispatch_time` / `finish_time` / `fanout`,
advances `count`, and rotates the records buffer in place when it
fills up. The ring is sized to the runtime's in-flight issue depth
(2 for dual-issue today; raise to the next power of two when issue
depth grows). The "completion-before-dispatch" runtime invariant
guarantees AICore never overwrites a slot before AICPU has read it;
violations are surfaced via the dedicated `mismatch_record_count`
counter.

### 5.2 a2a3 — shared-memory streaming

`halHostRegister` maps device memory into host virtual address
space so the host can read device buffers directly.
`L2PerfCollector` runs two background threads on top of a
[`BufferPoolManager<L2PerfModule>`](../src/a2a3/platform/include/host/profiling_common/buffer_pool_manager.h):
a mgmt thread that polls SPSC ready queues and recycles full
buffers **while kernels are still executing**, plus a poll
thread that drains the L2 hand-off queue into
`on_buffer_collected`.

`L2PerfModule` declares two buffer kinds going through one ready
queue per AICPU thread:

- **kind 0**: per-core `L2PerfBuffer` (task records).
- **kind 1**: per-thread `PhaseBuffer` (scheduler / orchestrator
  phase records).

The `is_phase` flag on each `ReadyQueueEntry` picks between them.
This is the only multi-kind module in the current framework — PMU
and TensorDump are single-kind.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ L2PerfCollector          │               │ AICPU + AICore           │
│                          │               │                          │
│ initialize(prefix)       │  alloc +      │ AICore on task end:      │
│   rtMalloc + halRegister │──register────>│   write timing into      │
│   pre-fill free queues   │              │   ring[reg_task_id % N]  │
│   for kind 0 + kind 1    │               │                          │
│                          │               │ AICPU on FIN:            │
│ start(tf)                │               │   commit ring slot →     │
│   ┌────────────────────┐ │ SPSC ready    │     records[count],      │
│   │ mgmt thread        │ │ queues        │   fill func_id /         │
│   │ (BufferPool driver)│ │<──L2Perf──────│   dispatch / finish /    │
│   │   poll ready queue │<┼──+ Phase─────<│   fanout; rotate buffer  │
│   │   recycle buffers  │─┼──free queue──>│   when full              │
│   └────────────────────┘ │               │ AICPU scheduler thread:  │
│   ┌────────────────────┐ │               │   per-loop-iter:         │
│   │ poll thread        │ │               │     write AicpuPhase-    │
│   │   reads via host   │ │ shared mem    │     Record into          │
│   │   mapping; copies  │<┼──mapping─────<│     PhaseBuffer          │
│   │   to host vectors  │ │               │                          │
│   └────────────────────┘ │               │                          │
│ stop()                   │               │                          │
│   join mgmt → join poll  │               │                          │
│ read_phase_header_metadata()             │                          │
│ reconcile_counters()     │               │                          │
│ export_swimlane_json()   │               │                          │
│   → l2_perf_records.json │               │                          │
└──────────────────────────┘               └──────────────────────────┘
```

**Lifecycle** (`device_runner.cpp`):

```text
init_l2_perf()
  l2_perf_collector_.initialize(num_aicore, ..., output_prefix_)
  kernel_args_.args.l2_perf_data_base = l2_perf_collector_.get_l2_perf_shm_device_ptr()
start(tf)                          ← spawn mgmt + poll threads
launch AICPU / AICore
rtStreamSynchronize
stop()                             ← join mgmt → join poll
read_phase_header_metadata()       ← single-shot read of orch_summary +
                                     core→thread mapping
reconcile_counters()               ← three-bucket accounting for both
                                     PERF and PHASE pools (total /
                                     collected / dropped); any non-zero
                                     current_buf_ptr is a flush bug
export_swimlane_json()             ← writes <output_prefix>/l2_perf_records.json
finalize(unregister, free)
```

[`L2PerfCollector`](../src/a2a3/platform/include/host/l2_perf_collector.h)
on a2a3 inherits from
[`profiling_common::ProfilerBase<L2PerfCollector, L2PerfModule>`](../src/a2a3/platform/include/host/profiling_common/profiler_base.h):
the base class owns the mgmt thread, the poll thread, and the
`BufferPoolManager<L2PerfModule>` they share. `L2PerfCollector`
supplies the L2-specific pieces — the `L2PerfModule` trait
(notably `kBufferKinds = 2` and `kind_of()`), `initialize` that
allocates and pre-fills both kinds of free queues, an
`on_buffer_collected` callback that branches on
`info.type == PERF_RECORD` vs `PHASE` to copy into the per-core
or per-thread vector, plus `read_phase_header_metadata` /
`reconcile_counters` / `export_swimlane_json` / `finalize`. The
mgmt/poll threading and `Module` trait pattern are shared with
PMU and TensorDump — see
[profiling-framework.md](../profiling-framework.md) for the
framework reference.

### 5.3 a5 — same framework, host-shadow transport

a5's `L2PerfCollector` derives from
`ProfilerBase<L2PerfCollector, L2PerfModule>` and shares the
mgmt + poll thread structure with a2a3. The single behavioral
deviation from §5.2 is the **transport channel**: a5 has no
`halHostRegister`, so each device buffer is paired with a
host-shadow `malloc()` and the mgmt loop synchronizes the two via
`profiling_copy.h` (`rtMemcpy` onboard, plain `memcpy` in sim).

The AICore-side write target is a per-core, **stable**
`L2PerfAicoreRing` (`dual_issue_slots[PLATFORM_L2_AICORE_RING_SIZE]`)
allocated once by the host and addressed via
`L2PerfBufferState::aicore_ring_ptr` (AICPU side) and
`KernelArgs::aicore_l2_perf_ring_addrs[block_idx]` forwarded into
`set_aicore_l2_perf_ring()` by `KERNEL_ENTRY` (AICore side). The ring
address never changes during a run, so AICore's write address is
decoupled from the AICPU's rotating `L2PerfBuffer`. Buffer rotation is
internal to `l2_perf_aicpu_complete_record` when `records[count]` hits
`PLATFORM_PROF_BUFFER_SIZE`. The runtime `Handshake` carries no
profiling fields.

The framework's `MemoryOps` therefore carries five callbacks on
a5 (`alloc` / `reg` / `free_` / `copy_to_device` /
`copy_from_device`); the mgmt loop mirrors the entire shm region
(`L2PerfDataHeader` + per-core `L2PerfBufferState` + per-thread
`PhaseBufferState`) device → host at the top of every tick, then
pushes back only the fields host actually modified (advanced
`queue_heads[q]`, refilled `free_queue.tail` and
`buffer_ptrs[slot]`) via `BufferPoolManager::write_range_to_device`.
The bulk `mirror_shm_to_device` is deliberately **not** called from
the mgmt loop: it would race with AICPU writes to device-only
fields (`current_buf_ptr`, `total/dropped/mismatch` counters,
`queue_tails`, `free_queue.head`, `AicpuPhaseHeader::magic`,
`orch_summary`, `core_to_thread[]`) and roll them back to whatever
the host shadow held at the start of the tick. Per-buffer
payloads (`L2PerfBuffer` / `PhaseBuffer`) are pulled on demand
inside `ProfilerAlgorithms::process_entry` after a popped
ready-entry resolves to its host shadow. `BufferPoolManager`'s
`release_owned_buffers` frees the device pointer via the
collector's `release_fn` and the paired shadow via `std::free()`.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ L2PerfCollector          │               │ AICPU + AICore           │
│   : ProfilerBase<...>    │               │                          │
│                          │               │                          │
│ initialize()             │  alloc + reg  │ AICore on task end:      │
│   rtMalloc shm           │──+ shadow────>│   write timing into      │
│   per-core L2PerfBuffer  │   memset 0    │   per-core ring slot     │
│   per-core AicoreRing    │   + push 0s   │   dual_issue_slots[      │
│   per-thread PhaseBuffer │               │     task_id & 1]         │
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
init_l2_perf()
  l2_perf_collector_.initialize(num_aicore, ..., output_prefix_)
  kernel_args_.args.l2_perf_data_base = l2_perf_collector_.get_l2_perf_setup_device_ptr()
  kernel_args_.args.aicore_l2_perf_ring_addrs =
      l2_perf_collector_.get_aicore_ring_addrs_device_ptr()
l2_perf_collector_.start(thread_factory)   ← mgmt + poll threads
launch AICPU / AICore
rtStreamSynchronize
l2_perf_collector_.stop()                  ← join mgmt + poll, drain final batch
l2_perf_collector_.read_phase_header_metadata()
l2_perf_collector_.reconcile_counters()    ← sanity-check + 3-bucket cross-check
l2_perf_collector_.export_swimlane_json()
l2_perf_collector_.finalize()
```

[`L2PerfCollector`](../src/a5/platform/include/host/l2_perf_collector.h)
on a5 inherits the same CRTP base
([`profiling_common::ProfilerBase`](../src/a5/platform/include/host/profiling_common/profiler_base.h))
as a2a3 and parameterizes
[`BufferPoolManager`](../src/a5/platform/include/host/profiling_common/buffer_pool_manager.h)
with `L2PerfModule` (`kBufferKinds = 2`). The only a5-specific
glue is the 5-callback `MemoryOps` and the per-tick shm mirror.

a5's per-thread AICPU flush hooks (`l2_perf_aicpu_flush_buffers` /
`l2_perf_aicpu_flush_phase_buffers`) are the only data path on the
records side — host never reads from `current_buf_ptr` to recover
records. `reconcile_counters` is purely passive: it logs an error if
any `current_buf_ptr` is non-zero with a non-empty buffer (a
device-flush bug), then runs the three-bucket cross-check
`collected + dropped + mismatch == device_total` per pool (PERF +
PHASE), same shape as a2a3.

### 5.4 a2a3 vs a5 at a glance

| Aspect | a2a3 | a5 |
| ------ | ---- | -- |
| Record shape | identical (`L2PerfRecord` / `AicpuPhaseRecord` / `AicpuOrchSummary`) | |
| AICore WIP-slot protocol | identical | |
| AICPU commit on FIN | identical | |
| Buffer model | rotating pool (free + ready queues) per kind | identical |
| Ready queue | per-AICPU-thread, multiplexes PERF + PHASE via `is_phase` | identical |
| Host threads | mgmt + poll, streams during execution | identical |
| Host-class shape | `ProfilerBase<L2PerfCollector, L2PerfModule>` (`kBufferKinds = 2`) | identical |
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

Per scheduler-loop iteration, AICPU also writes a 32-byte
`AicpuPhaseRecord` per phase (4 phases × 32 B = 128 B per
iteration). Both architectures drain buffers concurrently with
execution via the mgmt + poll thread pair; a5 additionally pays
per-tick `rtMemcpy`/`memcpy` round-trips to keep the host shadow in
sync, which overlap with device execution.

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

- Each per-core `L2PerfBuffer` and per-thread `PhaseBuffer` is
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

**No `l2_perf_records.json` produced.** Check that
`--enable-l2-swimlane` was passed. Verify `<output_prefix>` exists
in the run log; if `--rounds > 1`, only the first round records.

**`merged_swimlane.json` is missing.** `swimlane_converter` runs
automatically after a SceneTest with `--enable-l2-swimlane`; if it
did not, run it manually:

```bash
python -m simpler_setup.tools.swimlane_converter outputs/<case>_<ts>/l2_perf_records.json
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
active L2 perf buffer at run end. Check the AICPU flush path runs
for every thread that produced records.

**Phase records empty.** Either the runtime did not emit phase
data (only `tensormap_and_ringbuffer` does, and only when
`AicpuPhaseHeader::magic == AICPU_PHASE_MAGIC`), or the host's
`AicpuPhaseHeader` was not initialized. Verify the runtime sets
the magic in its scheduler init path.

**`dispatch_time` < `finish_time` mismatch.** Verify the runtime
overwrites `task_id` with the full encoding on FIN
(`tensormap_and_ringbuffer` does
`(ring_id << 32) | local_id`); a half-filled record means AICore
wrote the WIP slot but AICPU never committed.

**Scheduler-overhead deep-dive missing from converter output.**
The converter runs `sched_overhead_analysis` only when a device
log is resolvable. Pass `-d <device-id>` or place a `device-*`
log under `outputs/` close in time to the `l2_perf_records.json`
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
