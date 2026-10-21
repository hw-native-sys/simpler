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

Each invocation publishes its PMU event group in the Runtime descriptor using
the existing stream-ordered descriptor upload. AICPU programs the selected
events and resets that run's record counters before its producers start, for
both retained and single-run collection. Host admission updates export state
without writing the shared device header or counters. Buffer ownership and
queue cursors persist across runs; per-run terminal banks and driver leases
retain their existing completion requirements.

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

`--rounds > 1` disables PMU collection in the test harness so benchmark
rounds stay uninstrumented.

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
| `run_epoch` | Which run produced the row — the process-unique run identity |

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

### 3.4 Background output across runs (`collect_across_runs`)

By default a run's `pmu.csv` is finished at that run's own boundary: the host
drains the collection queues and merges the file before the next run starts, so
`run()` returning means the file exists.

With `collect_across_runs=True` on a local level-3 `Worker` **and** PMU enabled,
PMU keeps only the device-side reads it must take while the run still owns the
device, and a background writer finishes the file while the next run executes.
The device still runs one task graph at a time; what overlaps is the host's
receipt and the merge.

What changes for you:

| Question | Default | `collect_across_runs=True` |
| -------- | ------- | -------------------------- |
| after `run()` returns | `pmu.csv` exists | it may not exist yet |
| how to be sure it exists | nothing to do | call `Worker.flush_diagnostics()` |
| runs awaiting publication | none | at most two; a third **fails before it is submitted** |
| a run that produced no records | no file | no file, and that is a success only when the zero is proved |
| a failed write | the partial file is removed | the temp files are kept as evidence, and the next run under that directory is refused until they are removed |

`flush_diagnostics()` is the barrier, and its **return** is the success signal —
not the presence of `pmu.csv`:

```python
worker = Worker(..., collect_across_runs=True)
worker.run(...)          # pmu.csv may still be in the background
worker.flush_diagnostics()  # raises if any closed run is incomplete
# every run closed before the call has now published its file — except a run
# proved to have produced no records, which has no file and is still a success
```

A PMU CSV has no column in which to say "this file is partial", so anything
short of a proved-complete run raises rather than passing silently. A run is
complete only when its transport cut is known and had no failed queue, its
processing settled, its counters were readable, and `collected + dropped`
matched the device's own total. Records the device dropped are still published —
the rows that exist are real — but the call still raises.

**A run that produced nothing writes no file, and that is success**, exactly as
it is by default. The zero has to be proved, though: all of the device's record
counters at zero *and* the transport cut known and settled. A run whose counters
could not be read, or whose cut never settled, is not an empty run — it raises,
and `pmu.csv` is absent for a different reason. So absence of the file never
distinguishes the two; the call's return does.

A failure is **sticky for the life of the device runner**: there is no
acknowledgement API, so once one run has failed, every later
`flush_diagnostics()` raises until the worker is closed. It survives a collector
rebuild (a core-count change) and PMU being turned off and on again.
`Worker.close()` runs the same flush once more, before teardown, and that is the
last call that can report it; a failure that first happens inside the native
teardown leaves only the log lines and the preserved temp files.

Each retained run also freezes its own event type and column set at admission,
so a run that selects a different event type cannot change the columns of a
predecessor whose file is still being written.

**Pool sizing under retention.** PMU's buffer pool is allowed to grow to twice
the bytes it was seeded with (device plus host shadow together) while runs are
retained, where it is otherwise unbounded. Reaching that ceiling stops further
growth — nothing already in use is reclaimed — and the device then drops records
as it does when a pool runs dry, which makes that run incomplete and raises from
the flush. There is no host-memory or disk quota: disk is bounded only by the
filesystem, and a full disk is reported as a write failure.

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
  `PmuRecord` per task. The owning AICPU scheduler thread index is
  carried out of band on the ready-queue entry (its per-thread queue
  index in `PmuDataHeader::queues[thread][...]`) and threaded through
  to the CSV row as `thread_id`.
- **AICore** brackets the counting window around the kernel body via
  `CTRL` SPR bit 0, with a `pipe_barrier(PIPE_ALL)` either side of each
  toggle. That shape is the toolchain's: `metrics_prof_start` /
  `metrics_prof_stop` in `__clang_cce_aicore_functions.h` (CANN 9.0.0
  BiSheng) set and clear the same bit 0 fenced the same way, and its `#if`
  covers `__DAV_C220_VEC__`, `__DAV_C220_CUBE__` and `__NPU_ARCH__ == 3510`
  — both architectures here. `set_ctrl` is a scalar-pipe write while the
  measured work runs on the other pipes, so the fences are what order the
  enable against the kernel body. Conforming to that contract is not
  evidence about what the counters hold afterwards.

  Where the close sits relative to the completion signal
  differs by architecture, and neither order is an acknowledgement that a
  reader finished:
  - **a2a3** — both executors close the window and publish FIN on the core,
    and the counters are read later by the AICPU, released by that FIN. A
    notification is not a completed read, so the read may land after the
    close either way.
  - **a5** — both executors close the window and then read the counters
    **on the executing core** (`pmu_aicore_read_counters`), staging them
    before the completion is published. No AICPU read of a live counter is
    involved.

  Whether the counter registers hold their value after the close is not
  stated by any contract this repo can cite — and both flows above **do**
  depend on it, since each reads after its own close. It is an unresolved
  hardware assumption carried by this subsystem, not an absence of
  dependency.

The per-task counter readout and the host↔device buffer transport are
architecture-specific. Sections 5.2 and 5.3 describe each architecture
end-to-end; §5.4 is a side-by-side comparison.

### 5.1 Common interfaces

`kernel_args.pmu_data_base` is the single device-side handle host
publishes for the run. Its target struct is `PmuDataHeader` on both
architectures; in both cases it carries:

- `num_cores` — number of AICore instances in use
- `event_type` — initial pool metadata; each invocation selects its event group through its Runtime descriptor

AICPU reads it on init to find per-core state. On every task FIN it
commits a `PmuRecord` and increments `PmuBufferState::total_record_count`.
On the drop path it increments `PmuBufferState::dropped_record_count`
instead. a5 also tracks `PmuBufferState::mismatch_record_count` for
records lost to ring-slot `task_id` mismatch (a hard invariant
violation, distinct from capacity drops). Host uses these counters at
finalize for the cross-check:

```text
collected_on_host + dropped + mismatch == total   (a5, 3 buckets)
collected_on_host + dropped == total              (a2a3, 2 buckets)
```

#### Control read-back, per core and per run

Each core's producer makes **two** observations of its own control window and
writes them into **two separate fields** of the entry it claims —
`DiagnosticRunTerminal::producer_note` at its run init and
`::producer_note_close` at that core's own close:

| Field | Code | Meaning |
| ----- | ---- | ------- |
| `producer_note` | `PMU_NOTE_NO_WINDOW` | the platform gave this core no register address — sim, or a misconfigured table |
| `producer_note` | `PMU_NOTE_ENABLED` | the init read-back carried the enable this run wrote |
| `producer_note` | `PMU_NOTE_DISABLED` | it did not; the payload is the observed word |
| `producer_note_close` | `PMU_NOTE_HELD` | the close read carried the enable **after every task of the run had executed** |
| `producer_note_close` | `PMU_NOTE_LOST` | it was not; the payload is the close word |
| `producer_note_close` | zero | no close observation was published into this entry |

The init read is taken immediately after the run programmed `PMU_CTRL_0` — and
`PMU_CTRL_1` on a5 — before a single task ran. The close read is taken in
`pmu_aicpu_finalize`, **before** the restore writes the pre-run word back, so it
is an observation at close, not a record of the state throughout execution or
the state the teardown leaves. One read per core per run, on a teardown path,
with nothing written back from it and no decision taken from it. The close code is decided by the close
word **alone**: folding the init code into it would make one count answer two
questions, and "never enabled" and "stopped being enabled" would both read
`lost`.

**Why two fields and not one overwritten.** An init read-back proves the window
accepted this run's enable; it proves nothing about the window during the tasks,
which is the interval the counters are supposed to cover. The *pair* is the
discriminator, so both have to survive:

| at init | at close | narrows toward |
| ------- | -------- | -------------- |
| `enabled` | `held` | the enable was present at **both ends** — the per-task gate or the measurement, rather than the window's arming |
| `enabled` | `lost` | the enable was present at the start and absent at the end — what cleared it, rather than whether it was written |
| `not-enabled` | either | the init write, rather than the window |
| `enabled` | absent | the close observation did not reach this entry — see below for the three states that covers |

Three limits, all load-bearing:

- **Two endpoint reads do not describe the interval between them.** `held` says
  the enable was there before the first task and still there after the last one.
  It does **not** say it was there continuously: a disable followed by a
  re-enable inside the run satisfies exactly the same pair, and nothing here
  distinguishes that from an uninterrupted window.
- **An absent close note does not prove the close never ran.**
  `note_run_close` writes only into an entry already carrying that run's epoch,
  so a close whose entry has been re-claimed by a successor writes nothing; a
  note can also be lost in publication. "No close code" therefore covers at
  least a close that did not execute, a close whose epoch check refused, and a
  close whose note did not survive — and the field alone cannot separate them.
- **An `enabled` init read-back does not prove the write reached the intended
  register.** It proves the address the run programmed reads back the value it
  wrote. A window that is mapped and writable but is not this core's PMU block
  satisfies that too. Distinct calculated virtual addresses do not prove
  distinct physical registers; the HAL PMU window's geometry remains unverified.

A single field holding only the later observation would additionally collapse
the first two table rows into each other, which is the distinction this
observation exists to draw. None of the rows is proof that a counter counted,
and none is a cause: they say where to look next, **per core** — the aggregate
counts on the collector's line are not a verdict, and a repair must not be
chosen from them.

The note's payload also carries **the physical core id that logical core
reported**, because that id is what selected the window: a PMU row names the
logical core, while the address it was measured through came from
`pmu_reg_addrs[physical_core_id]`. Together with the base and strides
`get_aicore_reg_info` states once per table build, those two give the exact
address the run used.

It is a read with no write and no decision attached. The host counts the notes
by code when it settles the run's bank — per producer, never summed — and states
them in **one `LOG_TIMING` line per run**, which is the default log threshold
(`kDefaultThreshold` and `InitArgs::log_level` are both 25), so the line is in
the child's host log without any flag. That matters: at `LOG_INFO` it would sit
below the threshold and a run's evidence would exist only under a verbosity
nobody enables.

**What it cannot say.** The read-back comes from the address this run
programmed, so it cannot tell a correct window from one that is mapped but is
not this core's PMU block: such a write would land somewhere, and the read
would return whatever that somewhere holds. What a write to a wrong register
page actually does on this device is **not** established here — treat a wrong
mapping as a hypothesis, not as a known silent drop. Separating the two would
need a register in the block whose value does not come from our own writes, and
this architecture's `RegId` set has none, so no offset outside it is accessed.
The note also says nothing about the AICore `CTRL` bit-0 gate, which is an
AICore SPR with no AICPU read path, and nothing about whether any cycle was
counted — that holds for the close read as much as for the init one. A run
reporting `PMU_NOTE_HELD` on every core and zero cycles on every row has had two
candidates narrowed (the window was armed, and it stayed armed), not a cause
established.

**Coverage of the printed line.** The per-code counts cover every producer that
claimed the run; the physical id and register values printed beside them are
the **first** reporting producer's, not every core's. The notes themselves are
per producer in the bank, so a reader who needs a specific core's note needs
the host to print more than the first — which it does not do today.

### 5.2 a2a3 — shared-memory streaming (DAV_2201, 8 counters)

AICPU reads the 8 PMU counters via MMIO (`read_reg(reg_base, PMU_CNTi)`)
directly into a `PmuRecord` on every task FIN. Buffers rotate through
an SPSC free queue per core; full buffers flow through a per-thread
ready queue to host drain/refill shards. Drain refills free queues from
shard-local recycled lanes; collector shards stream records to CSV during
execution, and the replenish thread routes done buffers to lanes below their
recycled watermarks before allocating any remaining top-up. It never writes
device free queues. PMU has no init-seeded recycled surplus by default, so the
runtime watermark is only a minimal reserve batch rather than a
core-count-scaled extra allocation target.

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
│   │ drain/refill shard │ │               │     into records[count]  │
│   │ + replenish thread │ │ SPSC ready    │   if buffer full:        │
│   │   poll ready queue │<┼──queues──────<│     push ready entry,    │
│   │   refill freeQ     │─┼──free queue──>│     pop next buffer      │
│   └────────────────────┘ │               │                          │
│   ┌────────────────────┐ │ shared mem    │ pmu_aicpu_flush():       │
│   │ collector shard    │ │ mapping       │   push remaining full    │
│   │   read records via │<┼──────────────<│   buffers to ready_q     │
│   │   host mapping     │ │               │                          │
│   │   append to shard  │ │               │                          │
│   │   temp CSV         │ │               │                          │
│   └────────────────────┘ │               └──────────────────────────┘
│                          │
│ stop()                   │
│   join mgmt → collectors │
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
start(tf)                       ← spawn split mgmt threads (drain AICPU ready
                                  queues and refill free queues from
                                  recycled lanes; replenish drains done
                                  buffers into recycled lanes)
                                  + collector shards (drain host hand-off,
                                  append to shard-local temp CSV files)
launch AICPU / AICore
rtStreamSynchronize             ← wait for kernel completion
stop()                          ← join mgmt/replenish → join collectors
reconcile_counters()            ← merge shard files into pmu.csv, then assert
                                  collected + dropped == total;
                                  any non-empty current_buf_ptr is a
                                  flush bug, logged as ERROR
finalize(unregister, free)
```

[`PmuCollector`](../../src/a2a3/platform/include/host/pmu_collector.h)
inherits from
[`profiling_common::ProfilerBase<PmuCollector, PmuModule>`](../../src/common/platform/include/host/profiler_base.h):
the base class owns split mgmt threads, collector shards, and the
`BufferPoolManager<PmuModule>` they share. `PmuCollector` only supplies
the PMU-specific pieces — the `PmuModule` trait that describes the
shared-memory layout, an `init()` that allocates and pre-fills the free
queues, an `on_buffer_collected()` callback that appends records to a
shard-local temporary CSV, and `reconcile_counters()` / `finalize()` that
merge those files into `pmu.csv`. A failed merge preserves the temporary
files and does not count their rows as successfully exported. Successful
reconciliation removes the temporary files. An abrupt process exit can leave
them in the CSV directory; the next `start()` for the same output path removes
those stale run-scoped files before collector threads launch. The
mgmt/collector threading, buffer pooling, and `Module` trait pattern are shared
with ArgsDump and ChipSwimlane — see
[profiling-framework.md](profiling-framework.md) for the framework reference.

### 5.3 a5 — same framework, host-shadow transport (DAV_3510, 10 counters)

AICore reads the 10 PMU counters via the `ld_dev` MMIO load intrinsic
into a per-core dual-issue staging slot indexed by `reg_task_id & 1`.
AICPU, on observing FIN, validates the slot's recorded `task_id`
against the register token, copies the record into
`PmuBuffer::records[count]`, fills `func_id` / `core_type`, and
advances `count`. When a buffer fills up, AICPU switches to a new
buffer via the SPSC free queue / ready queue protocol (identical to
a2a3). At shutdown, AICPU flushes any partially-filled buffers via
`pmu_aicpu_flush_buffers()`.

a5's `PmuCollector` derives from
`ProfilerBase<PmuCollector, PmuModule>` and uses the same framework
abstractions as a2a3, including the same split mgmt + collector shard
shape (`kMaxCollectorThreads` = `PLATFORM_MAX_AICPU_THREADS`, i.e. 5 on
a5 vs 4 on a2a3, capping the shard arrays; the live drain/collector
count is `min(aicpu_thread_num, kMaxCollectorThreads)`). The
behavioral deviation from §5.2 is the **transport channel**: a5 has no
`halHostRegister`, so
each device buffer is paired with a host-shadow `malloc()` and the
mgmt loop synchronizes the two via `profiling_copy.h` (`rtMemcpy`
onboard, `memcpy` in sim). `MemoryOps` therefore carries five
callbacks (`alloc` / `reg` / `free_` / `copy_to_device` /
`copy_from_device`); the mgmt loop mirrors the entire shm region
(`PmuDataHeader` + per-core `PmuBufferState`) device → host at the
top of every tick, then pushes back only the fields host modified
(advanced `queue_heads[q]`, refilled `free_queue.tail` and
`buffer_ptrs[slot]`) via `BufferPoolManager::write_range_to_device`.
The bulk `mirror_shm_to_device` is **not** called from the mgmt
loop: it would race with AICPU writes to device-only fields
(`current_buf_ptr`, `total/dropped` counters, `queue_tails`,
`free_queue.head`) and roll them back. Each popped `PmuBuffer` is
still pulled on demand inside
`ProfilerAlgorithms::process_entry`.

```text
        HOST                                         DEVICE
┌──────────────────────────┐               ┌──────────────────────────┐
│ PmuCollector             │               │ AICore                   │
│   : ProfilerBase<...>    │               │                          │
│                          │               │                          │
│ init()                   │  alloc + reg  │ per-task end:            │
│   rtMalloc data region   │──+ shadow────>│   ld_dev 10 PMU_CNTs +   │
│   per-core PmuBuffers    │   memset 0    │     PMU_CNT_TOTAL        │
│   register_mapping(s)    │   + push 0s   │   write into             │
│   build PmuDataHeader    │               │   dual_issue_slots[      │
│                          │               │     reg_task_id & 1]     │
│ start(thread_factory)    │               │                          │
│   mgmt_thread starts     │               │ AICPU thread             │
│   poll_thread starts     │               │ on FIN:                  │
│                          │               │   match slot's task_id   │
│ mgmt every 10us tick:    │               │     vs reg_task_id       │
│   copy_from_device(shm)  │<──memcpy─────<│   copy into              │
│   for each ready entry:  │               │     records[count]       │
│     copy buf from device │<──memcpy─────<│   fill func_id/core_type │
│     resolve host ptr     │               │   ++count                │
│     push to host ready_q │               │   if buffer full:        │
│   advance queue_heads,   │               │     push ready entry,    │
│     refill free_queues   │               │     pop next from free_q │
│   write_range_to_device  │──memcpy──────>│                          │
│     for each modified    │               │ pmu_aicpu_flush():       │
│     field                │               │   push remaining full    │
│                          │               │   buffers to ready_q     │
│ poll thread:             │               │                          │
│   wait_pop_ready         │               │                          │
│   on_buffer_collected →  │               │                          │
│     write shard temp CSV │               │                          │
│   notify_copy_done       │               │                          │
│                          │               │                          │
│ rtStreamSynchronize      │               │                          │
│ stop()                   │               │                          │
│   join mgmt + poll       │               │                          │
│ reconcile_counters()     │               │                          │
│   sanity-check leftovers │               │                          │
│   + cross-check          │               │                          │
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
├── current_buf_ptr            (AICPU active buffer)
├── aicore_ring_ptr            (stable PmuAicoreRing*, host writes once)
├── current_buf_seq
├── total_record_count
├── dropped_record_count
└── mismatch_record_count      (ring slot / task_id invariant violations)

PmuAicoreRing[num_cores]                (stable AICore staging, never rotated)
└── PmuRecord dual_issue_slots[PLATFORM_PMU_AICORE_RING_SIZE]

PmuBuffer pool (rotated)                (BUFFERS_PER_CORE per core)
└── PmuRecord records[RECORDS_PER_BUFFER] + count
```

`halHostRegister` is not supported on DAV_3510, so the a5 collector
maintains a paired host-shadow `malloc()` per device buffer and
synchronizes via `rtMemcpy` (onboard) / `memcpy` (sim). The framework
copy hooks `profiling_copy_to_device` / `profiling_copy_from_device`
(in [`profiling_copy.h`](../../src/common/platform/include/host/profiling_copy.h))
abstract this difference.

Each AICore worker resolves its PMU MMIO base at kernel entry from
`KernelArgs::regs[get_physical_core_id()]` (the per-physical-core
register-base table the host already fills for AICPU) and reads its
`PmuAicoreRing` from `KernelArgs::aicore_pmu_ring_addrs[block_idx]`
(filled by the host in `PmuCollector::init`). Both addresses are
forwarded by `KERNEL_ENTRY` into platform-owned per-core slots
([`aicore_profiling_state.h`](../../src/a5/platform/include/aicore/aicore_profiling_state.h));
the runtime `Handshake` carries no profiling fields. Because the
resolved reg base is valid from Phase 1 onward (no AICPU-side init
dependency), `aicore_execute` caches it once after Phase 3 alongside
the rings rather than re-reading per record.

#### Two stagings, one commit

The two-slot per-core `PmuAicoreRing` works because the AICPU drains it on
every FIN: with a core running one task at a time, the `task_id % 2`
parity keeps task N and N+1 apart. A5 `host_build_graph` with the AICore
Scheduler has no such consumer — the on-core schedulers dispatch and
complete, and the AICPU never sees a FIN — so that route stages instead
into a **per-subtask cell array** sized with the trace cells and indexed
the same way (`SchedulerTaskPmuCell`, one cache line each). The executing
core writes its cell between `pmu_aicore_end()` and its completion
publication, stamping it with the task id plus one; the AICPU reads every
cell once in the run's completion gate and commits it through the same
`pmu_aicpu_record_staged_task` path the ring drain ends in. Both stagings
therefore share one accounting: `total` counts every commit attempted,
`dropped` a missing buffer, `mismatch` a ring slot the core never filled.
The cell array is reserved only when the bind's run enables PMU, so a run
with the owner off pays nothing for it.

Two things gate that read, and neither is the gate's arrival:

- **Proved retirement.** A participant whose partition reported an
  unreleased core still arrives at the completion gate, so the settlement
  counts the cores that acknowledged exit instead. Short of the full count
  it reads no cell and restores no register — a core that may still be
  executing is neither a safe source nor a safe target — and the run's PMU
  accounting is left unknown.
- **Expected coverage.** What the run owed is every active subtask of every
  executable task, taken from the graph rather than from the cells, so an
  absent cell is a lost measurement rather than a subtask that did not run.
  A run that did not complete its graph, or a completed one short of a
  cell, leaves its producers unsettled and reports unknown instead of
  publishing the subset that did arrive.

**Lifecycle** (`device_runner.cpp`):

```text
init_pmu()
  pmu_collector_.init(num_aicore, num_threads, csv_path, event_type, ...)
  kernel_args_.args.pmu_data_base             = pmu_collector_.get_pmu_shm_device_ptr()
  kernel_args_.args.aicore_pmu_ring_addrs = pmu_collector_.get_aicore_ring_addrs_device_ptr()
                                      → AICPU pmu_aicpu_init() resolves
                                        per-core PMU MMIO bases from
                                        regs[physical_core_ids[i]] and
                                        caches state->aicore_ring_ptr.
                                        AICore separately resolves its
                                        own base at kernel entry from
                                        regs[get_physical_core_id()].
pmu_collector_.start(thread_factory)   ← mgmt + poll threads
launch AICPU / AICore
rtStreamSynchronize
pmu_collector_.stop()                  ← join mgmt + poll, drain final batch
pmu_collector_.reconcile_counters()    ← merge shard files, then verify
                                          collected + dropped + mismatch == device_total
pmu_collector_.finalize(free)
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

[`PmuCollector`](../../src/a5/platform/include/host/pmu_collector.h) on
a5 inherits the same CRTP base
([`profiling_common::ProfilerBase`](../../src/common/platform/include/host/profiler_base.h))
as a2a3 and parameterizes
[`BufferPoolManager`](../../src/common/platform/include/host/buffer_pool_manager.h)
with `PmuModule`. The only a5-specific glue is the 5-callback
`MemoryOps` and the per-tick shm mirror.

a5's per-thread AICPU flush (`pmu_aicpu_flush_buffers`) is the only
data path on the records side — host never reads from
`current_buf_ptr` to recover records. `reconcile_counters` is purely
passive: it logs an error if any `current_buf_ptr` is non-zero with
a non-empty buffer (a device-flush bug), then runs the three-bucket
cross-check `collected + dropped + mismatch == device_total` against
device-side counters.

### 5.4 a2a3 vs a5 at a glance

| Aspect | a2a3 | a5 |
| ------ | ---- | -- |
| HW counter slots | 8 (DAV_2201) | 10 (DAV_3510) |
| Counter readout | AICPU MMIO `read_reg` | AICore MMIO `ld_dev` |
| Per-core staging | direct write into `records[count]` | dual-issue slots, AICPU commits on FIN |
| Buffer model | rotating pool (free + ready queues, SPSC protocol) | identical |
| Host threads | split mgmt + collector shards, writes shard-local temp files during execution and merges at reconcile | same split mgmt + collector shards (5 = `PLATFORM_MAX_AICPU_THREADS` vs a2a3's 4) |
| Host-class shape | `ProfilerBase<PmuCollector, PmuModule>` | identical |
| Host transport | `halHostRegister` shared memory | host-shadow `malloc` + per-tick `rtMemcpy`/`memcpy` |
| `MemoryOps` callbacks | 3 (`alloc`, `reg`, `free_`) | 5 (+ `copy_to_device`, `copy_from_device`) |
| `reconcile_counters` | passive cross-check (collected + dropped == device_total) | passive cross-check with mismatch bucket (collected + dropped + mismatch == device_total); leftover non-empty `current_buf_ptr` logged as a device flush bug |
| Lifecycle | `init` → `start` → `stop` → `reconcile_counters` → `finalize` | identical |

## 6. Overhead

PMU profiling is opt-in and zero-overhead when disabled — without
`--enable-pmu` neither host nor device allocates PMU storage and the
counter-read code paths are skipped.

When enabled, the dominant per-task overhead is the MMIO counter read
(8 reads on a2a3, 10 on a5) plus a single record copy. On both
architectures, shard-local streaming keeps host-side work off the critical
path without serializing collectors on one output stream. Reconciliation
merges the bounded temporary files into the final CSV after collection stops.
Both a2a3 and a5 use split mgmt plus collector shards (a5 with 7 shards,
a2a3 with 4). a5's copy hooks add `rtMemcpy` round-trips that a2a3's
shared memory avoids, but these overlap with device execution.

For meaningful per-task numbers on a2a3 the runtime collapses to
single-issue dispatch automatically whenever `--enable-pmu` is set (see
§7.1) — this serialization itself costs throughput, so PMU-on
measurements are not comparable to PMU-off baselines.

## 7. Limitations

### 7.0 The counting window has another writer, and the lease that holds it off

simpler programs the AICore PMU control register by writing MMIO directly from
the AICPU. TS writes the same register, and nothing in the driver or the
runtime arbitrates between the two: the window mapping is per-process with no
refcount and no owner check. An AICore-utilization query on the same chip
(`npu-smi info -t usages`) resets `PMU_CTRL_0` to a resting value, and so does a
background periodic source that has not been identified. Every counter reads
zero from that point in the run.

How often that happens depends on how long the window stays open, which is why
it is invisible in a short payload and routine in a long one: the same 513-task
payload loses every counter in roughly one run in seven at a half-second
window, and in most runs at four seconds.

While driver prof channel 53 (`CHANNEL_FFTS_PROFILE_BUFFER_TASK`) is started on
a device, the reset does not happen. `AicorePmuLease`
([aicore_pmu_lease.h](../../src/common/platform/onboard/host/aicore_pmu_lease.h))
starts it for the duration of a PMU collection, reference-counted so that it
spans every retained run rather than being dropped by the first run to close.

Three properties of that mechanism decide how it has to be used:

- **Starting the channel is what protects; the configuration's content does
  not.** `cfgMode` 0 and 3 protect alike, and so does clearing the config's
  sub-task bit. Measured 9/9 runs losing every counter with no channel held,
  0/15 with one held, over 30780 rows.
- **`cfgMode = 0` is nevertheless required**, because the lease is taken in the
  process that submits the tasks. At `cfgMode = 3` the device emits 128 bytes
  per task into a host ring that no reader drains, and that ring overwrites its
  oldest bytes without accounting for the loss. At `cfgMode = 0` a
  4000-kernel in-process workload produced nothing at all.
- **The channel is exclusive across processes**, enforced below user space: a
  second start returns `-10` while one is held. So a PMU-enabled run denies
  channel 53 to msprof on that device for as long as it collects, and a
  profiler already holding it denies the lease to us. The second case is
  logged and collection proceeds unprotected; it never fails a run.

**Why holding the channel protects is not established.** It may be an
implementation detail of the device firmware, which is not open and which a
CANN upgrade may change with no signal here. That is why
`TestPmuCountingWindow`
([test_pmu_counting_window.py](../../tests/st/a2a3/host_build_graph/dfx/pmu/test_pmu_counting_window.py))
asserts per row rather than in aggregate, and asserts a floor on the spinning
task as well: a window disabled partway through leaves a row that is non-zero
but short, which was observed at 1.1% of the expected count with every row
non-zero. An aggregate check, or a check for zeros alone, reads that run as a
pass.

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
  [platform_config.h](../../src/a5/platform/include/common/platform_config.h)
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
collection is suppressed by the harness. With
`collect_across_runs=True` the file may simply not be finished yet —
call `Worker.flush_diagnostics()` first (§3.4).

**`flush_diagnostics()` raises but `pmu.csv` is there.** Expected, and
the file is not the answer. A CSV has no column that can say "partial",
so a run whose records or transport cut could not be proved complete
publishes the rows it has *and* fails the flush. The error string names
the run, its verdict and its counts. The failure is sticky for the
device runner's life, so every later call raises too (§3.4).

**A run is refused before it starts, with "both retained PMU runs are
still unpublished".** Two unpublished runs is the capacity. Call
`flush_diagnostics()` between runs, or accept that a slow disk bounds
how fast runs may be submitted.

**A run is refused with "still holds a failed run's temporary files".**
A previous run's `pmu.csv.e<epoch>.*.tmp` files are its failure
evidence and are never deleted for you. Read them, remove them, and the
destination is usable again.

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
host drain/replenish path has more buffer headroom.

**Dropped records on a5.** `PmuBufferState::dropped_record_count`
nonzero means the AICPU could not get a free buffer in time
(`free_queue` empty). Increase `PLATFORM_PMU_BUFFERS_PER_CORE` so the
host drain/replenish path has more buffer headroom.

## 9. Related docs

- [profiling-framework.md](profiling-framework.md) — shared host-side
  collector framework.
- [chip-level-arch.md](../chip-level-arch.md) — host / AICPU / AICore
  program boundaries the PMU path spans.
- [task-flow.md](../task-flow.md) — where AICPU dispatch and completion
  sit in the per-task state machine.
