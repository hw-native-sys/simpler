# Profiling Framework

Shared host-side infrastructure that the PMU, L2Perf, and TensorDump
collectors are built on. Each architecture maintains its own copy of the
framework headers under `src/<arch>/platform/include/host/profiling_common/`
([a2a3](../src/a2a3/platform/include/host/profiling_common/),
[a5](../src/a5/platform/include/host/profiling_common/)) — the two copies
are kept byte-for-byte structurally aligned so reviewers can diff them, but
each arch is free to evolve independently. This page describes the shape;
§8 covers the a5-specific deviations driven by transport differences.

The per-collector pages
([pmu-profiling.md](dfx/pmu-profiling.md),
[l2-swimlane-profiling.md](dfx/l2-swimlane-profiling.md),
[tensor-dump.md](dfx/tensor-dump.md))
describe the data each subsystem collects and how it enables it on-device.

## 1. Why a shared framework

Each profiling subsystem on a2a3 needs the same plumbing on the host:

- A management thread that polls the AICPU's per-thread SPSC ready queues
  and recycles full buffers back to the device while kernels are still
  running.
- A collector thread that drains the host-side hand-off queue and copies
  records out of each ready buffer.
- A pool of pre-registered device buffers (allocated up-front, refilled on
  demand) keyed by "kind" — PMU has 1 kind, TensorDump has 1, L2Perf has 2
  (perf records + phase markers).
- A dev↔host pointer map so the management thread can resolve a device
  pointer popped off a ready queue to the host-mapped pointer the collector
  thread will read.
- A teardown sequence that flushes both queues without losing late entries.

Before unification this was three near-identical implementations. The
framework collapses it to one control-flow implementation parameterized on
a small per-subsystem trait.

## 2. Layered view

```text
                ┌──────────────────────────────────────────┐
                │  PmuCollector / L2PerfCollector /        │  Derived (CRTP)
                │  TensorDumpCollector                     │  ─ on_buffer_collected
                └─────────────┬────────────────────────────┘  ─ kIdleTimeoutSec / kSubsystemName
                              │ public ProfilerBase<Derived, Module>
                ┌─────────────▼────────────────────────────┐
                │  ProfilerBase<Derived, Module>           │  Thread orchestration
                │  ─ owns mgmt thread + collector thread   │  ─ start/stop lifecycle
                │  ─ runs ProfilerAlgorithms<Module>       │  ─ consume → notify_copy_done
                └─────────────┬────────────────────────────┘
                              │ has-a
                ┌─────────────▼────────────────────────────┐
                │  BufferPoolManager<Module>               │  Data structures (no threads)
                │  ─ ready_queue / done_queue              │  ─ recycled pools (per kind)
                │  ─ alloc_and_register / resolve_host_ptr │  ─ MemoryOps (type-erased)
                └──────────────────────────────────────────┘
                              ▲
                              │ Module trait wires layout into algorithms
              ┌───────────────┴────────────────┐
              │  PmuModule / L2PerfModule /    │  Pure static trait (no state)
              │  DumpModule                    │  ─ DataHeader / ReadyEntry / FreeQueue
              └────────────────────────────────┘  ─ kBufferKinds / kReadyQueueSize
                                                  ─ resolve_entry / for_each_instance
```

`ProfilerBase` is the owner: it holds `BufferPoolManager manager_` as a
member ([profiler_base.h:414](../src/a2a3/platform/include/host/profiling_common/profiler_base.h#L414)),
spawns and joins both threads, and dispatches collected buffers to
`Derived::on_buffer_collected` via CRTP. `BufferPoolManager` owns no
threads — it is just the shared data structure both threads access.
`Module` is a stateless trait that tells the generic algorithms how the
subsystem's shared-memory layout is shaped.

## 3. The three roles

### 3.1 `BufferPoolManager<Module>` — data layer

Defined in [`buffer_pool_manager.h`](../src/a2a3/platform/include/host/profiling_common/buffer_pool_manager.h).
Owns:

- `ready_queue_` — mgmt → collector hand-off, guarded by mutex+cv.
- `done_queue_` — collector → mgmt recycle channel, guarded by mutex.
- `recycled_[kind]` — per-kind pool of free device buffers (mgmt-only).
- `dev_to_host_` — single source of truth for `resolve_host_ptr`.
- `MemoryOps` — type-erased `alloc / reg / free_` callbacks, plus the
  `shared_mem_host` and `device_id` stashed once at start.

Owns no threads. Every entry point is documented as one of:

- mgmt-only (recycled pool ops, `drain_done_into_recycled`),
- collector-only (`notify_copy_done`),
- shared with internal locking (`push_to_ready` / `wait_pop_ready` /
  `try_pop_ready`),
- start/stop-only (`set_memory_context`, `release_owned_buffers`,
  `clear_mappings`).

### 3.2 `ProfilerBase<Derived, Module>` — control layer

Defined in [`profiler_base.h`](../src/a2a3/platform/include/host/profiling_common/profiler_base.h).
Provides:

- The two threads and their lifecycle (`start` / `stop`).
- `mgmt_loop` — drains `done_queue` → recycled, polls every AICPU
  per-thread ready queue (bounded by `PLATFORM_MAX_AICPU_THREADS`),
  invokes `ProfilerAlgorithms<Module>::process_entry` per popped entry,
  and tops up free queues with `proactive_replenish`.
- `poll_and_collect_loop` — `wait_pop_ready` with a 100 ms cv tick,
  dispatches to `Derived::on_buffer_collected`, then calls
  `manager_.notify_copy_done(...)` itself; idle-timeout hang detector.
- `set_memory_context` / `clear_memory_context` so `Derived::init` can
  stash the alloc/reg/free callbacks before threads start; if init aborts
  before stashing, `start(tf)` becomes a no-op.

`ProfilerAlgorithms<Module>` (in the same header, [profiler_base.h:170](../src/a2a3/platform/include/host/profiling_common/profiler_base.h#L170))
is where the unified algorithms live:

- `try_pop_aicpu_entry` — barrier-correct head/tail advance over the
  per-thread ready queue, with a range-check guard against device-side
  corruption.
- `process_entry` — three-level fallback (recycled → drain done → alloc)
  to refill the originating free_queue with **exactly one** buffer per
  popped entry, then resolve host_ptr and push to ready. The 1-in/1-out
  ratio bounds per-tick latency.
- `proactive_replenish` — drain done, then top every (kind, instance)
  free queue up to `kSlotCount`, batch-allocating `batch_size(kind)`
  buffers when the recycled pool of a kind drains mid-fill so recovery
  from a double-empty condition takes one tick instead of N.

### 3.3 `Module` — trait layer

A stateless `struct` per subsystem (`PmuModule`, `L2PerfModule`,
`DumpModule`) that tells the generic algorithms what the shared-memory
layout looks like. The contract lives in the docblock at the top of
[`profiler_base.h`](../src/a2a3/platform/include/host/profiling_common/profiler_base.h);
the required members are:

| Member | Purpose |
| ------ | ------- |
| `using DataHeader / ReadyEntry / ReadyBufferInfo / FreeQueue` | Layout types |
| `kBufferKinds` (PMU=1, Dump=1, L2Perf=2) | Number of per-kind recycled pools |
| `kReadyQueueSize`, `kSlotCount` | AICPU ready queue / free queue depth |
| `kSubsystemName` | Tag used in framework log lines |
| `header_from_shm(void*) → DataHeader*` | Cast shared-memory base to header |
| `batch_size(int kind) → int` | Per-kind batch-alloc count |
| `resolve_entry(shm, header, q, entry) → optional<EntrySite>` | Map a popped ready entry to (kind, free_queue, buffer_size, partial info); return `nullopt` to drop |
| `for_each_instance(shm, header, cb)` | Enumerate every (kind, instance, FreeQueue*, buffer_size) for `proactive_replenish` |
| `kind_of(info) → int` | **Multi-kind only.** Tells the framework which recycled bin a finished buffer belongs to. Single-kind modules omit this; the framework passes 0 |

The Module structs are defined alongside their collectors in
[pmu_collector.h](../src/a2a3/platform/include/host/pmu_collector.h),
[l2_perf_collector.h](../src/a2a3/platform/include/host/l2_perf_collector.h),
and [tensor_dump_collector.h](../src/a2a3/platform/include/host/tensor_dump_collector.h)
— each is a few dozen lines of static methods over the subsystem's own
`DataHeader` / ringbuffer types.

### 3.4 `Derived` — domain layer

Each collector inherits as `class XxxCollector : public ProfilerBase<XxxCollector, XxxModule>`
and only has to provide:

- `void on_buffer_collected(const ReadyBufferInfo& info)` — copy the
  records out of `info.host_buffer_ptr` and update collector-specific
  state (CSV row, in-memory aggregator, file writer thread, …). The
  framework calls `manager_.notify_copy_done(...)` afterwards; **Derived
  must not call it directly.**
- `static constexpr int kIdleTimeoutSec` — bound on no-progress idle in
  the collector loop. Use the subsystem's `PLATFORM_*_TIMEOUT_SECONDS`
  constant.
- `static constexpr const char* kSubsystemName` — appears in the idle
  timeout log line (e.g. `"PMU"`, `"L2Perf"`, `"TensorDump"`).
- `init(...)` and `finalize(...)` — domain-specific setup/teardown.
  `init` must call `set_memory_context()` on the success path so
  `start(tf)` is not a no-op. `finalize` must release framework-owned
  buffers (`release_owned_buffers`) and drop the mapping table
  (`clear_mappings`).

## 4. End-to-end data flow

```text
  AICPU                       mgmt thread                       collector thread
  ─────                       ───────────                       ────────────────
  write record into         drain_done_into_recycled
  current free buffer       ──────────────────────────►
                            try_pop_aicpu_entry(q)
                            process_entry:
                              pop_recycled / alloc_and_register
                                (refill originating free_queue, 1-in/1-out)
                              resolve_host_ptr
                              push_to_ready ──────────────────► wait_pop_ready
                                                                Derived::on_buffer_collected
                                                                  (copy records out)
                                                                notify_copy_done
                            ◄────────────────────────────────── (done_queue)
                            (next tick) drain into recycled

                                          ▲
                                          │
                            proactive_replenish: top every
                            free_queue up to kSlotCount;
                            batch-alloc when a kind drains.
```

Both queues plus the per-kind recycled pools and the dev↔host map all
live in the single `BufferPoolManager` instance owned by `ProfilerBase`.
The mgmt thread is the only writer to the ready queue; the collector
thread is the only writer to the done queue. Recycled pools are
mgmt-only.

## 5. Lifecycle

```text
  Derived::init(...)
    rtMalloc + register pre-allocated buffers
    register_mapping for each (dev, host) pair
    set_memory_context(alloc_cb, register_cb, free_cb, ud, shm_host, device_id)

  ProfilerBase::start(thread_factory)
    assemble MemoryOps from stashed callbacks (sim mode installs an
      identity reg wrapper so register == nullptr is supported uniformly)
    manager_.set_memory_context(ops, shm_host, device_id)
    spawn mgmt thread       ← started first; mgmt is the only writer to L2
    spawn collector thread

    ... AICPU / AICore execute ...

  ProfilerBase::stop()
    mgmt_running_ = false
    join mgmt thread        ← mgmt's final-drain pass flushes the last
                              entries into ready_queue before exiting
    execution_complete_ = true
    join collector thread   ← drains ready_queue once more, then exits

  Derived::finalize(unregister, free)
    manager_.release_owned_buffers([&](void* p) { unregister + free })
    free buffers still held in collector-owned free_queues / current_buf_ptr
    manager_.clear_mappings()
    clear_memory_context()
```

The order in `stop()` is load-bearing: mgmt is joined **before**
`execution_complete_` is signalled so its final-drain output has a
consumer; the collector then drains and exits. On return both queues are
empty and `on_buffer_collected` has been called for every entry that was
in either queue.

`Derived::finalize` is responsible for the buffers the collector still
owns at stop time (`free_queue` slots and `current_buf_ptr`); the
framework only releases what it had in recycled / done / ready. This
split matters because the AICPU may still be referencing free-queue
buffers via shared memory until execution ends, so they cannot be freed
mid-run by the framework.

## 6. Concurrency invariants

| State | Reader(s) | Writer(s) | Synchronization |
| ----- | --------- | --------- | --------------- |
| `ready_queue_` | collector | mgmt | `ready_mutex_` + `ready_cv_` |
| `done_queue_` | mgmt | collector | `done_mutex_` |
| `recycled_[kind]` | mgmt | mgmt | none (single-threaded access) |
| `dev_to_host_` | mgmt (`alloc_and_register`, `resolve_host_ptr`) | mgmt | none during run; collector touches it only in `release_owned_buffers` / `clear_mappings`, after `stop()` has joined mgmt |
| `MemoryOps` / `shared_mem_host_` / `device_id_` | both threads | start-only | `set_memory_context` is called once before threads spawn; read-only afterwards |
| AICPU per-thread ready queues (`header->queues[q]`) | mgmt (head advance) | AICPU (tail advance) | `rmb` / `wmb` paired with AICPU writers |
| Per-instance `FreeQueue` | AICPU (head advance) | mgmt (tail advance) | `rmb` / `wmb` paired with AICPU readers |

Two things follow:

- `dev_to_host_` is unlocked because mgmt owns it during the run and the
  collector only touches it when mgmt is joined. Adding a collector path
  that mutates the map mid-run would require revisiting this.
- The mgmt thread must never zero AICPU-owned fields (`count`, `head`,
  `tail` on the AICPU side). The AICPU is the sole writer to those and
  resets them itself on flush/drop/pop.

## 7. Adding a new collector

1. Define the subsystem's shared-memory types (`DataHeader`,
   `ReadyQueueEntry`, `FreeQueue`, `ReadyBufferInfo`) somewhere both host
   and AICPU can include.
2. Write a `XxxModule` struct satisfying the contract in §3.3. Multi-kind
   modules also implement `kind_of`.
3. Write a `XxxCollector : public profiling_common::ProfilerBase<XxxCollector, XxxModule>`:
   - `init(...)`: `rtMalloc` + register pre-allocated buffers, populate
     the shared header, call `register_mapping` per buffer, then call
     `set_memory_context(...)`.
   - `on_buffer_collected(info)`: copy records out of
     `info.host_buffer_ptr`. **Do not** call `notify_copy_done`.
   - `kIdleTimeoutSec`, `kSubsystemName`.
   - `finalize(unregister, free)`: `release_owned_buffers` + free
     collector-owned buffers + `clear_mappings` + `clear_memory_context`.
4. Wire it into `device_runner` so `start(tf)` is called before the
   kernel launch and `stop()` before `finalize`.

Existing collectors are the canonical examples:

- [`PmuCollector`](../src/a2a3/platform/include/host/pmu_collector.h)
  — single kind, per-core instances. See [pmu-profiling.md](dfx/pmu-profiling.md).
- [`TensorDumpCollector`](../src/a2a3/platform/include/host/tensor_dump_collector.h)
  — single kind, per-AICPU-thread instances. See [tensor-dump.md](dfx/tensor-dump.md).
- [`L2PerfCollector`](../src/a2a3/platform/include/host/l2_perf_collector.h)
  — two kinds (perf records + phase markers), per-core / per-thread
  instances; the canonical multi-kind example. See
  [l2-swimlane-profiling.md](dfx/l2-swimlane-profiling.md).

## 8. a5 specifics — host-shadow transport

a5's framework headers (under
[`src/a5/platform/include/host/profiling_common/`](../src/a5/platform/include/host/profiling_common/))
mirror a2a3's class shapes — same `ProfilerBase<Derived, Module>` /
`BufferPoolManager<Module>` / `ProfilerAlgorithms<Module>` decomposition,
same Module concept contract, same start/stop lifecycle. The only
behavioral deviation is the **transport channel**: a5 has no
`halHostRegister`, so device↔host synchronization goes through
`profiling_copy.h` (`rtMemcpy` onboard, `memcpy` in sim) against a
host-shadow `malloc()` paired with each device buffer. Three mechanical
changes capture that:

1. **`MemoryOps` carries 5 callbacks instead of 3.** In addition to
   `alloc` / `reg` / `free_`, a5 adds `copy_to_device` and
   `copy_from_device`. `reg` allocates the paired host shadow (instead of
   mapping a HAL view onto the device pointer); a default
   `default_host_shadow_register` (`malloc + memset 0 + copy_to_device`)
   is installed when the collector passes `nullptr` for `register_cb`.
2. **The mgmt loop mirrors the shm region into the host shadow per tick,
   then writes back only the fields it modified.** At the top of every
   tick the loop calls `manager_.mirror_shm_from_device()` to pull the
   `DataHeader` (with its `queue_tails`) and all `BufferState`s into the
   host shadow. As host advances `queue_heads[q]` or refills a
   `free_queue.tail` / `buffer_ptrs[slot]`, those individual fields are
   pushed back to device with `manager_.write_range_to_device(&field,
   sizeof(field))`. The bulk `mirror_shm_to_device` is deliberately
   **not** called from the mgmt loop — it would race with AICPU writes
   to device-only fields (`current_buf_ptr`, `total/dropped/mismatch`
   counters, `queue_tails`, `free_queue.head`,
   `AicpuPhaseHeader::magic`, `orch_summary`, `core_to_thread[]`),
   rolling them back to whatever the host shadow had at the start of
   the tick. Per-buffer payloads (`L2PerfBuffer` / `PmuBuffer` /
   `DumpMetaBuffer`) are still pulled on demand inside
   `ProfilerAlgorithms::process_entry` after resolving the host pointer
   for a popped ready entry. The bulk `mirror_shm_to_device` is kept
   for init/teardown where AICPU is not yet running or has exited.
3. **`release_owned_buffers` frees the paired host shadow.** When the
   framework recycles a device buffer at finalize time, it also frees the
   `malloc()`'d shadow tracked in `dev_to_host_`. `clear_mappings()` does
   the same for any remaining mappings (per-state buffers and the shm
   region itself).

Each collector's `reconcile_counters` is purely passive — same shape as
a2a3. It pulls the post-`stop()` `BufferState`s, logs an error if any
`current_buf_ptr` still references a non-empty buffer (a device flush
bug, since AICPU is the only writer that can clear it), and runs the
three-bucket cross-check
`collected_on_host + dropped + mismatch == device_total` against
device-side counters per pool. Host never reads from `current_buf_ptr`
to recover records — recovering would mask AICPU flush bugs.

### 8.1 Profiling state lives on `KernelArgs`, not `Handshake`

a5 carries the same "profiling out of the runtime synchronization protocol"
design as a2a3 (PR #714) plus the AICore-side rings of PR #709. The runtime
`Handshake` struct contains **no profiling fields** — enablement bits and
per-core ring/reg addresses travel through `KernelArgs`:

| `KernelArgs` field | Producer | Consumer |
| ------------------ | -------- | -------- |
| `enable_profiling_flag` (bitmask) | host (DeviceRunner) | AICPU `kernel.cpp` → `set_l2_swimlane_enabled` / `set_pmu_enabled` / `set_dump_tensor_enabled`; AICore `KERNEL_ENTRY` → `set_aicore_profiling_flag` |
| `aicore_l2_perf_ring_addrs` (table) | host (`L2PerfCollector::initialize`) | AICore `KERNEL_ENTRY` indexes `table[block_idx]` → `set_aicore_l2_perf_ring` |
| `aicore_pmu_ring_addrs` (table) | host (`PmuCollector::init`) | AICore `KERNEL_ENTRY` → `set_aicore_pmu_ring` |
| `pmu_reg_addrs` (table) | host alloc; AICPU `pmu_aicpu_init` fills entries | AICore `KERNEL_ENTRY` → `set_pmu_reg_addrs`; AICore `aicore_execute` reads `get_aicore_pmu_reg_base()` per-record |

AICore's per-core profiling slots live in
[`aicore/aicore_profiling_state.h`](../src/a5/platform/include/aicore/aicore_profiling_state.h)
— `[[block_local]]` static on onboard, `pthread_key_t` TLS in sim. The
runtime `aicore_execute(runtime, block_idx, core_type)` signature is
unchanged; adding a new profiling field touches `KernelArgs` and this
state surface, never the runtime protocol.

### 8.2 Stable AICore staging ring (decouples AICore write from AICPU buffer rotation)

L2Perf and PMU on a5 both use the "AICore writes, AICPU commits" model.
The AICore-side write target is a per-core
[`L2PerfAicoreRing`](../src/a5/platform/include/common/l2_perf_profiling.h) /
[`PmuAicoreRing`](../src/a5/platform/include/common/pmu_profiling.h) of
`PLATFORM_{L2,PMU}_AICORE_RING_SIZE` (= 2, dual-issue) slots, allocated
once by the host and addressed by
`BufferState::aicore_ring_ptr` (AICPU-visible) and the per-core
`aicore_*_ring_addrs[block_idx]` (AICore-visible). The address is
never reassigned, so AICore's write target is stable across AICPU's
rotating `L2PerfBuffer` / `PmuBuffer` flips — flipping is now
fully internal to `*_complete_record` and never crosses into Handshake.

Everything else — Module concept contract, alloc policy
(1-in/1-out + proactive replenish), `kIdleTimeoutSec` / `kSubsystemName`
contract, mgmt-then-poll start/stop ordering, buffer-pool sizing
constants — matches a2a3 exactly. New collectors should be reviewed
against both arches when added.
