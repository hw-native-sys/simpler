# PMU Profiling — Per-task AICore Hardware Counters

## 1. Background & Motivation

AICore performance issues — pipeline stalls, memory-bandwidth shortfalls,
cache misses — are invisible at the runtime layer. The hardware exposes
a Performance Monitoring Unit (PMU) per core with a small bank of
counters that can be programmed to track these microarchitectural
events, but reading them only at run start/end conflates every task
into one number.

PMU profiling samples those counters once per runtime task, so each
row in the output corresponds to a single kernel invocation. That
makes it possible to attribute a hot counter (e.g. high `mte2_busy`
or low `cube_busy`) to a specific `func_id` instead of "the run".

## 2. Overview

- **One row per task.** Counters are sampled at task completion, not
  as a post-run aggregate.
- **Selectable event group.** A single `--enable-pmu N` flag picks
  which counter group is active for the run (`PIPE_UTILIZATION`,
  `MEMORY`, `L2_CACHE`, …).
- **CSV output, fixed schema.** A `pmu.csv` lands under the per-task
  output prefix; the column order is the same on both architectures
  for tooling parity.
- **Cross-architecture.** Same Python entry point, same CSV format on
  `a2a3` and `a5`. Wired through both `host_build_graph` and
  `tensormap_and_ringbuffer` runtimes.

Enable in one line:

```bash
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-pmu
```

## 3. How to Use

### 3.1 Enable PMU

Bare flag selects the default `PIPE_UTILIZATION` event group:

```bash
# Standalone runner
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-pmu
python tests/st/<case>/test_<name>.py -p a5   -d 0 --enable-pmu

# pytest
pytest tests/st/<case> --platform a2a3 -d 0 --enable-pmu
pytest tests/st/<case> --platform a5   -d 0 --enable-pmu
```

`--enable-pmu` alone is equivalent to `--enable-pmu 2`. Pass an explicit
value (see §3.3) to switch event groups:

```bash
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-pmu 4
```

The `SIMPLER_PMU_EVENT_TYPE` environment variable overrides the CLI
event type when set:

```bash
SIMPLER_PMU_EVENT_TYPE=4 \
python tests/st/<case>/test_<name>.py -p a2a3 -d 0 --enable-pmu
```

`--rounds > 1` disables PMU collection in the test harness so warm-up
rounds are not double-counted.

### 3.2 Output

The PMU artifact is a CSV file under the per-task output prefix
(`CallConfig::output_prefix`, set by `scene_test.py::_build_output_prefix`
to `outputs/<ClassName>_<case>_<YYYYMMDD_HHMMSS>/` for SceneTest runs):

```text
<output_prefix>/pmu.csv
```

The filename is fixed (no per-file timestamp) — the directory is the
per-task uniqueness boundary.

Common columns (in order, identical across architectures):

| Column | Meaning |
| ------ | ------- |
| `thread_id` | AICPU scheduler thread that drives this core |
| `core_id` | Logical AICore id in the runtime |
| `task_id` | Runtime task id, printed as hex |
| `func_id` | Kernel function id |
| `core_type` | `0` = AIC, `1` = AIV |
| `pmu_total_cycles` | 64-bit `PMU_CNT_TOTAL` snapshot |
| event-specific counters | Counter columns selected by the event type |
| `event_type` | Numeric event type used for the run |

The number of counter columns varies by event type — each event group
populates a different subset of the hardware counter slots, and the
CSV lists only the slots that have a defined name. Use the
`event_type` column to discover which counters are present in a given
file.

Example row (`PIPE_UTILIZATION` on a2a3, abbreviated):

```text
thread_id,core_id,task_id,func_id,core_type,pmu_total_cycles,
  vec_busy_cycles,cube_busy_cycles,scalar_busy_cycles,
  mte1_busy_cycles,mte2_busy_cycles,mte3_busy_cycles,
  icache_miss,icache_req,event_type
2,5,0x0000000200000a00,0,1,18432,
  1024,0,512,0,256,128,
  3,448,2
```

Read: `func_id=0` ran on AIV core 5 (driven by AICPU thread 2),
took ~18 K total cycles, vector pipe was busy for 1024 cycles, cube
was idle, MTE2 (load) ran 256 cycles, etc.

### 3.3 Event types

| Value | Event Type | Example Counters |
| ----- | ---------- | ---------------- |
| `1` | `ARITHMETIC_UTILIZATION` | cube/vector execution counters |
| `2` | `PIPE_UTILIZATION` | vector, cube, scalar, MTE busy cycles |
| `4` | `MEMORY` | UB / L1 / L2 / main memory requests |
| `5` | `MEMORY_L0` | L0A / L0B / L0C requests |
| `6` | `RESOURCE_CONFLICT` | bank and vector resource stalls |
| `7` | `MEMORY_UB` | UB and memory bandwidth counters |
| `8` | `L2_CACHE` | L2 cache hit / miss / allocation counters |

Invalid nonzero values fall back to `PIPE_UTILIZATION`.

Each architecture programs its own per-counter event-code space
(`pmu_resolve_event_config_a2a3` / `pmu_resolve_event_config_a5`), so
the specific counter column names differ per event group per
architecture. The two `PIPE_UTILIZATION` rosters as a concrete example:

a2a3 (DAV_2201, 8 slots):

```text
vec_busy_cycles, cube_busy_cycles, scalar_busy_cycles,
mte1_busy_cycles, mte2_busy_cycles, mte3_busy_cycles,
icache_miss, icache_req
```

a5 (DAV_3510, 10 slots):

```text
pmu_idc_aic_vec_busy_o, cube_instr_busy, scalar_instr_busy,
mte1_instr_busy, mte2_instr_busy, mte3_instr_busy,
icache_req, icache_miss, pmu_fix_instr_busy
```

## 4. Capabilities

What you can read out of `pmu.csv`:

- **Per-task pipeline utilization** (`PIPE_UTILIZATION`) — how busy
  vector / cube / scalar / MTE pipes were during each kernel.
- **Per-task arithmetic mix** (`ARITHMETIC_UTILIZATION`) — fp16 / int8
  / fp32 instruction counts on cube and vector pipes.
- **Memory traffic** (`MEMORY`, `MEMORY_L0`, `MEMORY_UB`, `L2_CACHE`)
  — read/write request counts and cache hit/miss tallies at each
  level of the memory hierarchy.
- **Resource contention** (`RESOURCE_CONFLICT`) — bank-conflict and
  vector-resource-stall cycle counts.
- **Per-task total cycles** (`pmu_total_cycles`, present in every
  event group).

For a single run, only one event group is active. Iterate the run
under different `--enable-pmu N` values to cover other counter groups.

## 5. Design Highlights

Three layers cooperate; the split is the same across architectures:

- **Host** owns user entry, event-type selection, allocation, and CSV
  export. Publishes a single device pointer through
  `kernel_args.pmu_data_base` that points at the architecture's PMU
  shared region.
- **AICPU** programs the PMU event group at init, starts/stops the
  counters via `PMU_CTRL_0/1`, observes per-task FIN, and commits one
  `PmuRecord` per task. Stamps `PmuBufferState::owning_thread_id` with
  the AICPU scheduler thread that drives the core (per-buffer, stable
  for a run).
- **AICore** brackets the counting window around the kernel body via
  `CTRL` SPR bit 0.

The per-task counter readout and the host↔device buffer transport are
architecture-specific. Sections 5.2 and 5.3 describe each architecture
end-to-end; §5.4 is a side-by-side comparison.

### 5.1 Common interfaces

`kernel_args.pmu_data_base` is the single device-side handle host
publishes for the run. Its target struct shape differs per
architecture (a2a3 uses `PmuDataHeader`; a5 uses `PmuSetupHeader`),
but in both cases it carries:

- `num_cores` — number of AICore instances in use
- `event_type` — `PmuEventType` value the host wrote at init time

AICPU reads it on init to find per-core state. On every task FIN it
commits a `PmuRecord` and increments `PmuBufferState::total_record_count`.
On the drop path it increments `PmuBufferState::dropped_record_count`
instead. Host uses these two counters at finalize for the cross-check:

```text
collected_on_host + dropped == total
```

### 5.2 a2a3 — shared-memory streaming (DAV_2201, 8 counters)

AICPU reads the 8 PMU counters via MMIO (`read_reg(reg_base, PMU_CNTi)`)
directly into a `PmuRecord` on every task FIN. Buffers rotate through
an SPSC free queue per core; full buffers flow through a per-thread
ready queue to a host mgmt thread that recycles them, while a host
poll thread streams records to CSV during execution.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ PmuCollector             │               │ AICPU thread             │
│                          │               │                          │
│ init()                   │  alloc +      │ pmu_aicpu_init()         │
│   rtMalloc + halRegister │──register────>│   read PmuDataHeader     │
│   pre-fill free queues   │              │   pop initial buffer     │
│                          │               │   per-core               │
│                          │               │                          │
│ start(tf)                │               │ per-task FIN:            │
│   ┌────────────────────┐ │               │   read 8 PMU_CNTs+TOTAL  │
│   │ mgmt thread        │ │               │     into records[count]  │
│   │ (BufferPool driver)│ │ SPSC ready    │   if buffer full:        │
│   │   poll ready queue │<┼──queues──────<│     push ready entry,    │
│   │   recycle buffers  │─┼──free queue──>│     pop next buffer      │
│   └────────────────────┘ │               │                          │
│   ┌────────────────────┐ │ shared mem    │ pmu_aicpu_flush():       │
│   │ poll thread        │ │ mapping       │   push remaining full    │
│   │   read records via │<┼──────────────<│   buffers to ready_q     │
│   │   host mapping     │ │               │                          │
│   │   append to CSV    │ │               │                          │
│   └────────────────────┘ │               └──────────────────────────┘
│                          │
│ stop()                   │
│   join mgmt → join poll  │
│ reconcile_counters()     │
│ finalize()               │
└──────────────────────────┘
```

Device memory layout (`pmu_data_base` →):

```text
PmuDataHeader                                   (host init, AICPU/host R/W)
├── queues  [MAX_AICPU_THREADS][READYQUEUE_SIZE]
├── queue_heads / queue_tails (per-thread)
├── num_cores
└── event_type

PmuBufferState[num_cores]                       (per-core state)
├── free_queue {buffer_ptrs[SLOT_COUNT], head, tail}
├── current_buf_ptr          (AICPU active buffer)
├── current_buf_seq
├── dropped_record_count
└── total_record_count

PmuBuffer pool (rotated)                        (BUFFERS_PER_CORE per core)
└── PmuRecord records[RECORDS_PER_BUFFER] + count
```

**Lifecycle** (`device_runner.cpp`):

```text
init_pmu()
  pmu_collector_.init(num_aicore, num_threads, csv_path, event_type, ...)
  kernel_args_.args.pmu_data_base = pmu_collector_.get_pmu_shm_device_ptr()
start(tf)                       ← spawn mgmt thread (drains AICPU L1 ready
                                  queue, recycles full buffers via
                                  BufferPoolManager) + poll thread (drains
                                  L2 hand-off, appends to CSV)
launch AICPU / AICore
rtStreamSynchronize             ← wait for kernel completion
stop()                          ← join mgmt → join poll
reconcile_counters()            ← assert collected + dropped == total;
                                  any non-empty current_buf_ptr is a
                                  flush bug, logged as ERROR
finalize(unregister, free)
```

[`PmuCollector`](../src/a2a3/platform/include/host/pmu_collector.h)
inherits from
[`profiling_common::ProfilerBase<PmuCollector, PmuModule>`](../src/a2a3/platform/include/host/profiling_common/profiler_base.h):
the base class owns the mgmt thread, the poll thread, and the
`BufferPoolManager<PmuModule>` they share. `PmuCollector` only supplies
the PMU-specific pieces — the `PmuModule` trait that describes the
shared-memory layout, an `init()` that allocates and pre-fills the free
queues, an `on_buffer_collected()` callback that appends records to the
CSV, and `reconcile_counters()` / `finalize()`. The mgmt/poll threading,
buffer pooling, and `Module` trait pattern are shared with TensorDump
and L2Perf — see [profiling-framework.md](../profiling-framework.md) for
the framework reference.

### 5.3 a5 — streaming with host shadow buffers (DAV_3510, 10 counters)

AICore reads the 10 PMU counters via the `ld_dev` MMIO load intrinsic
into a per-core dual-issue staging slot indexed by `reg_task_id & 1`.
AICPU, on observing FIN, validates the slot's recorded `task_id`
against the register token, copies the record into
`PmuBuffer::records[count]`, fills `func_id` / `core_type`, and
advances `count`. When a buffer is full, AICPU switches to a new
buffer via the SPSC free queue / ready queue protocol (identical to
a2a3). At shutdown, AICPU flushes any partially-filled buffers via
`pmu_aicpu_flush_buffers()`. The host runs a dedicated collector
thread that polls ready queues via `rtMemcpy`, writes records to CSV,
and recycles buffers back into SPSC free queues.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ PmuCollector             │               │ AICore                   │
│                          │               │                          │
│ init()                   │  alloc +      │ per-task end:            │
│   rtMalloc data region   │──copy────────>│   ld_dev 10 PMU_CNTs +   │
│   pre-fill free queues   │               │     PMU_CNT_TOTAL        │
│   build PmuDataHeader    │               │   write into             │
│                          │               │   dual_issue_slots[      │
│ start(tf)                │               │     reg_task_id & 1]     │
│   ┌────────────────────┐ │               │                          │
│   │ collector thread   │ │               │ AICPU thread             │
│   │ poll_and_collect() │ │               │ on FIN:                  │
│   │  poll ready queues │ │  rtMemcpy     │   match slot's task_id   │
│   │  via copy hooks    │<┼──(shadow)────<│     vs reg_task_id       │
│   │  write CSV         │ │               │   copy into              │
│   │  recycle → free_q  │─┼──copy hook──>│     records[count]       │
│   └────────────────────┘ │               │   fill func_id/core_type │
│                          │               │   ++count                │
│                          │               │   if buffer full:        │
│                          │  SPSC ready   │     push ready entry,    │
│                          │<──queues─────<│     pop next from free_q │
│                          │               │                          │
│ rtStreamSynchronize      │               │ pmu_aicpu_flush():       │
│ drain_remaining_buffers()│               │   push remaining full    │
│   final sync + drain     │               │   buffers to ready_q     │
│ reconcile_counters()     │               │                          │
│ finalize(free)           │               │                          │
└──────────────────────────┘               └──────────────────────────┘
```

Device memory layout:

```text
[PmuDataHeader]                         (kernel_args.pmu_data_base)
├── queues  [MAX_AICPU_THREADS][READYQUEUE_SIZE]
├── queue_heads / queue_tails (per-thread)
├── num_cores
└── event_type

[PmuBufferState[num_cores]]             (per-core state)
├── free_queue {buffer_ptrs[SLOT_COUNT], head, tail}
├── current_buf_ptr          (AICPU active buffer)
├── current_buf_seq
├── dropped_record_count
├── total_record_count
└── owning_thread_id

PmuBuffer pool (rotated)                (BUFFERS_PER_CORE per core)
├── count                 (header, 64 B)
├── dual_issue_slots[2]   (AICore staging, indexed task_id & 1)
└── records[RECORDS_PER_BUFFER]
```

`halHostRegister` is not supported on DAV_3510, so the a5 collector
maintains separate host shadow buffers and synchronizes via `rtMemcpy`
(onboard) or `memcpy` (sim). The platform copy hooks
`pmu_platform_copy_to/from_device` abstract this difference.

Each `Handshake` carries `pmu_buffer_addr` and `pmu_reg_base` so each
AICore worker locates its `PmuBuffer` and the PMU MMIO register block.

**Lifecycle** (`device_runner.cpp`):

```text
init()
  pmu_collector_.init(num_aicore, num_threads, csv_path, event_type, ...)
  kernel_args_.args.pmu_data_base = pmu_collector_.get_pmu_shm_device_ptr()
                                      → AICPU programs PMU event group,
                                        publishes (pmu_buffer_addr,
                                        pmu_reg_base) into Handshakes
start(tf)                       ← spawn collector thread
                                  (poll_and_collect: drain AICPU ready
                                  queues via copy hooks, write CSV,
                                  recycle buffers into free queues)
launch AICPU / AICore
rtStreamSynchronize
stop()                          ← signal_execution_complete, join thread
drain_remaining_buffers()       ← final sync: drain remaining ready
                                  queue entries + scan current_buf_ptr
                                  for partially-filled buffers
reconcile_counters()            ← assert collected + dropped == total
finalize(free)
```

**Slot match key vs logical `task_id`.** `pmu_aicpu_complete_record`
takes both a 32-bit `reg_task_id` (the value AICore read from
`DATA_MAIN_BASE` and stored in `slot->task_id`) and a 64-bit logical
`task_id` written into the record itself. Runtimes whose logical id
encodes more than 32 bits (e.g. `tensormap_and_ringbuffer`'s
`(ring_id<<32)|local_id`) carry both — slot match must use the
register token, otherwise the slot will never validate.

The two dual-issue slots exist because dispatch can have up to two
tasks in flight on a single AICore. Parity on `reg_task_id & 1` keeps
adjacent dispatches from colliding (the runtime's `dispatch_seq++`
guarantees neighboring register tokens differ by 1 → different slots).

[`PmuCollector`](../src/a5/platform/include/host/pmu_collector.h) on
a5 uses the same streaming lifecycle as a2a3 — `init` / `start` /
`stop` / `drain_remaining_buffers` / `reconcile_counters` / `finalize`.
The collector thread runs `poll_and_collect()` concurrently with kernel
execution. The only architectural difference from a2a3 is the memory
transport: host shadow buffers + `rtMemcpy` copy hooks instead of
`halHostRegister` shared memory.

### 5.4 a2a3 vs a5 at a glance

| Aspect | a2a3 | a5 |
| ------ | ---- | -- |
| HW counter slots | 8 (DAV_2201) | 10 (DAV_3510) |
| Counter readout | AICPU MMIO `read_reg` | AICore MMIO `ld_dev` |
| Per-core staging | direct write into `records[count]` | dual-issue slots, AICPU commits on FIN |
| Host transport | `halHostRegister` shared memory | host shadow buffers + `rtMemcpy` copy hooks |
| Buffer model | rotating pool (free + ready queues) | rotating pool (free + ready queues, same SPSC protocol) |
| Host threads | mgmt + poll, streams during execution | collector thread, streams during execution |
| Host-class shape | `ProfilerBase` subclass (mgmt + poll thread + `BufferPoolManager<PmuModule>`) | streaming `init`/`start`/`stop`/`drain`/`finalize` |
| Lifecycle | `init` → `start` → `stop` → `reconcile_counters` → `finalize` | `init` → `start` → `stop` → `drain_remaining_buffers` → `reconcile_counters` → `finalize` |

## 6. Overhead

PMU profiling is opt-in and zero-overhead when disabled — without
`--enable-pmu` neither host nor device allocates PMU storage and the
counter-read code paths are skipped.

When enabled, the dominant per-task overhead is the MMIO counter read
(8 reads on a2a3, 10 on a5) plus a single record copy. On both
architectures, streaming keeps host-side work off the critical path —
the collector thread drains buffers concurrently with kernel execution.
On a5 the copy hooks add `rtMemcpy` round-trips that a2a3's shared
memory avoids, but these overlap with device execution.

For meaningful per-task numbers on a2a3 the runtime collapses to
single-issue dispatch automatically whenever `--enable-pmu` is set (see
§7.1) — this serialization itself costs throughput, so PMU-on
measurements are not comparable to PMU-off baselines.

## 7. Limitations

### 7.1 a2a3

PMU collection assumes each logical AICore has at most one in-flight
task. The default dual-issue dispatch preloads a pending task while
another task is still running on the same core, so per-core PMU
registers can carry overlapping task windows. To keep counters scoped
to a single task, `--enable-pmu` automatically collapses dispatch to
single-issue at runtime — both `host_build_graph` and
`tensormap_and_ringbuffer` runtimes branch on `is_pmu_enabled()` in
their dispatch path. No separate flag or rebuild is required.

Notes on this constraint:

- PMU-on runs serialize dispatch per core, so throughput is lower than
  PMU-off baselines. The two are not directly comparable.
- `a2a3sim` exercises the export pipeline; counter values come from
  the simulation backend, not real hardware, so they are not suitable
  for performance analysis.

### 7.2 a5

- `a5sim` exercises the export pipeline; the simulated counter
  register block does not model AICore execution, so counter values
  are 0. The CSV still carries one row per task with a zero counter
  tuple — useful for validating the end-to-end data flow.
- The per-core on-device `PmuBuffer` capacity is controlled by
  `PLATFORM_PMU_RECORDS_PER_BUFFER` (default 512). When full, AICPU
  switches to a new buffer via the free queue. If no free buffer is
  available, records are dropped. Increase `PLATFORM_PMU_BUFFERS_PER_CORE`
  (default 4) in
  [platform_config.h](../src/a5/platform/include/common/platform_config.h)
  if your workload produces bursts that exhaust the buffer pool.
- A non-zero `diff` in the host's `record count mismatch` warning
  means AICPU attempted to commit `diff` records whose dual-issue
  slot still carried an older `task_id`. With AICore's slot-write
  order (`counters → pmu_total_cycles → store barrier → task_id →
  dcci → dsb → write FIN to COND`), `diff` should always be zero on
  DAV_3510. A persistent non-zero `diff` is a sharp diagnostic —
  find the regression rather than tuning
  `PLATFORM_PMU_RECORDS_PER_BUFFER`. Common causes:

  1. The `reg_task_id` producer/consumer drifted out of sync (AICPU
     uses a different task-id encoding than AICore wrote into the
     slot).
  2. AICPU calls `pmu_aicpu_complete_record` for a task AICore never
     executed (e.g. an AICPU-only task path; AICore never wrote that
     slot, so `task_id` stays stale).
  3. AICore's `dcci` / `dsb` ordering around the slot write was
     rearranged, or a barrier was weakened from full `dsb` to a
     store-only flavor.
  4. The hardware target's `dcci(..., CACHELINE_OUT)` semantics
     differ (e.g. non-DAV_3510 ports) and no longer guarantee HBM
     writeback before the following `dsb`.

## 8. FAQ / Debug Guide

**No `pmu.csv` produced.** Check that `--enable-pmu` was passed (or
`SIMPLER_PMU_EVENT_TYPE` was set with the flag). Verify
`<output_prefix>` exists in the run log; if `--rounds > 1`, PMU
collection is suppressed by the harness.

**All counter columns are zero.** Either the platform is `a2a3sim` /
`a5sim` (counter registers are not modelled), or the active event
group does not populate the columns shown — check the `event_type`
column and the per-architecture event table in §3.3.

**Counter values look polluted on a2a3.** Dual-issue dispatch is
overlapping tasks on the same core. `--enable-pmu` should already
collapse dispatch to single-issue at runtime (§7.1); if pollution
persists, verify that `is_pmu_enabled()` returns true on every AICPU
thread and that the dispatch loop branch in `scheduler_dispatch.cpp`
and `aicpu_executor.cpp` hasn't been bypassed.

**`record count mismatch (... diff=M)` on a5.** Slot-mismatch loss —
this should be 0 on DAV_3510. Treat as a regression: see §7.2 for the
four common causes and check the `reg_task_id` / barrier / `dcci`
chain rather than tuning buffer sizes.

**`current_buf_ptr` non-empty at finalize on a2a3.** The host logs
this as ERROR and does not recover. It indicates AICPU did not flush
its active PMU buffer at run end. Check `pmu_aicpu_flush_buffers` is
called for every AICPU thread, and that the per-thread core list
covers every core that produced records.

**Dropped records on a2a3.** `PmuBufferState::dropped_record_count`
nonzero means the AICPU could not get a free buffer in time
(`free_queue` empty). Increase `PLATFORM_PMU_BUFFERS_PER_CORE` so the
mgmt thread has more headroom to recycle buffers.

**Dropped records on a5.** `PmuBufferState::dropped_record_count`
nonzero means the AICPU could not get a free buffer in time
(`free_queue` empty). Increase `PLATFORM_PMU_BUFFERS_PER_CORE` so the
collector thread has more headroom to recycle buffers.

## 9. Related docs

- [profiling-framework.md](../profiling-framework.md) — shared host-side
  collector framework.
- [chip-level-arch.md](../chip-level-arch.md) — host / AICPU / AICore
  program boundaries the PMU path spans.
- [task-flow.md](../task-flow.md) — where AICPU dispatch and completion
  sit in the per-task state machine.
