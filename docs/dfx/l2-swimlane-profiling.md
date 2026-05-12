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
header plus per-core / per-thread state:

```text
L2PerfDataHeader / L2PerfSetupHeader            (host init, device R/W)
├── num_cores
└── (a2a3) per-thread ready queues
    (a5)  per-core / per-thread buffer pointers

L2PerfBufferState[num_cores]                    (per-core perf state)
├── free_queue (a2a3 only)
├── current_buf_ptr (a2a3) / fixed buffer (a5)
├── total_record_count
└── dropped_record_count

[AicpuPhaseHeader + PhaseBufferState[num_threads]]  (optional)
├── magic / num_sched_threads
├── orch_summary  (cumulative orchestrator cycles)
└── per-thread phase buffers
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
The ring address is published once via
`Handshake::l2_perf_aicore_ring_addr` and never reassigned, so AICore
is fully decoupled from any AICPU-side records-buffer rotation. AICPU,
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

### 5.3 a5 — bulk rtMemcpy after stream sync

`L2PerfCollector` allocates one `L2PerfBuffer` per core and one
`PhaseBuffer` per AICPU thread on device, plus a single
`L2PerfSetupHeader` that stores all buffer pointers and the
`AicpuPhaseHeader`. AICore writes timing into per-task WIP slots
and AICPU completes records + writes phase data directly into
device memory; when a buffer fills up, records are silently
dropped (AICPU-side early return) and accounted in
`PmuBufferState`-style counters. Collection is a single bulk
pass after `rtStreamSynchronize`: copy `L2PerfSetupHeader` back,
then for each per-core / per-thread buffer copy the 64-byte
header, read `count`, and copy `count*sizeof(record)` bytes.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ L2PerfCollector          │               │ AICPU + AICore           │
│                          │               │                          │
│ initialize()             │  alloc +      │ AICore on task end:      │
│   rtMalloc setup region  │──copy────────>│   write timing into      │
│   one L2PerfBuffer per   │               │   wip[task_id & 1]       │
│     core                 │               │                          │
│   one PhaseBuffer per    │               │ AICPU on FIN:            │
│     scheduler thread     │               │   commit wip slot →      │
│                          │               │     records[count],      │
│ ── kernel execution ──   │               │   fill func_id /         │
│                          │               │   dispatch / finish /    │
│                          │               │   fanout                 │
│                          │               │                          │
│                          │               │ AICPU scheduler thread:  │
│                          │               │   per-loop-iter:         │
│                          │               │     write AicpuPhase-    │
│                          │               │     Record               │
│                          │               │                          │
│ rtStreamSynchronize      │               │                          │
│ collect_all()            │  batch        │                          │
│   memcpy setup region    │<──memcpy─────<│                          │
│   per core/thread:       │               │                          │
│     copy buffer hdr →    │               │                          │
│     read count           │               │                          │
│     copy count*record    │               │                          │
│ export_swimlane_json()   │               │                          │
│   → l2_perf_records.json │               │                          │
│ finalize(free)           │               │                          │
└──────────────────────────┘               └──────────────────────────┘
```

**Lifecycle** (`device_runner.cpp`):

```text
initialize()
  l2_perf_collector_.initialize(num_aicore, ...)
  kernel_args_.args.l2_perf_data_base = l2_perf_collector_.get_l2_perf_setup_device_ptr()
launch AICPU / AICore
rtStreamSynchronize
collect_all()                      ← rtMemcpy setup region back, then per-core
                                     and per-thread buffers (header → records)
export_swimlane_json(output_prefix)
finalize()
```

[`L2PerfCollector`](../src/a5/platform/include/host/l2_perf_collector.h)
on a5 is self-contained — `initialize` / `collect_all` /
`export_swimlane_json` / `finalize`. There are no helper threads
to coordinate, so the a5 collector does not derive from
`ProfilerBase`.

### 5.4 a2a3 vs a5 at a glance

| Aspect | a2a3 | a5 |
| ------ | ---- | -- |
| Record shape | identical (`L2PerfRecord` / `AicpuPhaseRecord` / `AicpuOrchSummary`) | |
| AICore WIP-slot protocol | identical | |
| AICPU commit on FIN | identical | |
| Host transport | `halHostRegister` shared memory | `rtMemcpy` after `rtStreamSynchronize` |
| Buffer model | rotating pool (free + ready queues) per kind | one `L2PerfBuffer` per core + one `PhaseBuffer` per thread |
| Ready queue | per-AICPU-thread, multiplexes PERF + PHASE via `is_phase` | none (all buffers pre-allocated) |
| Host threads | mgmt + poll, streams during execution | drain after sync |
| Host-class shape | `ProfilerBase` subclass (mgmt + poll thread + `BufferPoolManager<L2PerfModule>`, `kBufferKinds = 2`) | self-contained `initialize`/`collect_all`/`finalize` |
| Lifecycle | `initialize` → `start` → `stop` → `read_phase_header_metadata` → `reconcile_counters` → `export_swimlane_json` → `finalize` | `initialize` → `collect_all` → `export_swimlane_json` → `finalize` |

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
iteration). Streaming (a2a3) drains both kinds of buffers
concurrently with execution; bulk drain (a5) defers the host-side
cost to after `rtStreamSynchronize`.

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
