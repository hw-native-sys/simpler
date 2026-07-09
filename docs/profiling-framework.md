# Profiling Framework

Shared profiling infrastructure that the PMU, L2Swimlane, DepGen,
ArgsDump, and ScopeStats collectors are built on. The host-side framework
headers live in
[`src/common/platform/include/host/`](../src/common/platform/include/host/)
and are consumed verbatim by both a2a3 and a5 collectors (PR #944
unified the previously-divergent per-arch copies into one set). The AICPU
device-side producer algorithms live in
[`src/common/platform/include/aicpu/profiler_device_engine.h`](../src/common/platform/include/aicpu/profiler_device_engine.h)
and are shared by the device writers that publish buffers into those host
collectors. This page describes both halves; §8 covers the a5-specific
transport deviations that the collectors themselves still carry.

The per-collector pages
([pmu-profiling.md](dfx/pmu-profiling.md),
[l2-swimlane-profiling.md](dfx/l2-swimlane-profiling.md),
[dep_gen.md](dfx/dep_gen.md),
[args-dump.md](dfx/args-dump.md),
[scope-stats.md](dfx/scope-stats.md))
describe the data each subsystem collects and how it enables it on-device.

For everyday per-run timing (no collector, always on under `SIMPLER_HOST_STRACE`):
[l2-timing.md](dfx/l2-timing.md) covers host_wall / device_wall (`[STRACE]` markers) +
device-log Orch/Sched/Total, and [host-trace.md](dfx/host-trace.md) +
[device-phases.md](dfx/device-phases.md) cover the `[STRACE]` per-stage
breakdown of `simpler_run` (host stages + AICPU phase subdivision).

## 1. Why a shared framework

Each profiling subsystem needs the same plumbing on the host:

- A management path that polls the AICPU's per-thread SPSC ready queues,
  refills device free queues from shard-local recycled lanes, and recycles
  collector-returned buffers while kernels are still running. A module may
  opt into split drain/refill threads plus a replenish thread. Runtime
  allocation, when enabled, is confined to the replenish thread and publishes
  only to host-side recycled lanes.
- Collector thread shards that drain host-side hand-off queues and copy
  records out of each ready buffer.
- A pool of pre-registered device buffers (allocated up-front, refilled on
  demand from host-side watermarks) keyed by "kind". PMU, DepGen,
  ArgsDump, and ScopeStats have one kind; L2Swimlane has four.
- A dev↔host pointer map so the management thread can resolve a device
  pointer popped off a ready queue to the host-mapped pointer the collector
  thread will read.
- A teardown sequence that flushes the device queues and host shards without
  losing late entries.

The AICPU producer side has a matching repeated shape: wait for ready-queue
space, publish a full buffer, wait for a free replacement, install it as the
current buffer, and account dropped records if bounded backpressure expires.

Before unification this was near-identical control flow repeated across
collectors. The framework collapses the host side to one implementation
parameterized on a small per-subsystem trait, and the device side to
`DeviceProfilerEngine<Module>` plus small local AICPU module traits.

## 2. Layered view

```text
                ┌──────────────────────────────────────────┐
                │  Pmu / L2Swimlane / DepGen / Dump / Scope │  Derived (CRTP)
                │  collectors                               │  ─ on_buffer_collected
                └─────────────┬────────────────────────────┘  ─ kIdleTimeoutSec / kSubsystemName
                              │ public ProfilerBase<Derived, Module>
                ┌─────────────▼────────────────────────────┐
                │  ProfilerBase<Derived, Module>           │  Thread orchestration
                │  ─ owns drain/replenish + collector      │  ─ start/stop lifecycle
                │  ─ runs ProfilerAlgorithms<Module>       │  ─ consume → notify_copy_done
                └─────────────┬────────────────────────────┘
                              │ has-a
                ┌─────────────▼────────────────────────────┐
                │  BufferPoolManager<Module>               │  Data structures (no threads)
                │  ─ ready/done/recycled SPSC shards       │  ─ retired holding pool
                │  ─ block allocation / resolve_host_ptr   │  ─ MemoryOps (type-erased)
                └──────────────────────────────────────────┘
                              ▲
                              │ Module trait wires layout into algorithms
              ┌───────────────┴────────────────┐
              │  Pmu / L2Swimlane / DepGen /      │  Pure static trait (no state)
              │  Dump / Scope modules             │  ─ DataHeader / ReadyEntry / FreeQueue
              └────────────────────────────────┘  ─ kBufferKinds / kReadyQueueSize
                                                  ─ kHostPoolQueueSize / kHostRecycledQueueSize
                                                  ─ resolve_entry / for_each_instance

  AICPU device side:

                ┌──────────────────────────────────────────┐
                │  AICPU writer files                      │  record fill / flush / local state
                │  pmu / l2 / dep_gen / dump / scope       │  subsystem hooks
                └─────────────┬────────────────────────────┘
                              │ uses
                ┌─────────────▼────────────────────────────┐
                │  DeviceProfilerEngine<DeviceModule>      │  ready/free/switch algorithms
                └─────────────┬────────────────────────────┘
                              │ DeviceModule trait wires layout into algorithms
                ┌─────────────▼────────────────────────────┐
                │  XxxDeviceModule                         │  DataHeader / State / FreeQueue
                │                                          │  write_ready_entry / drop hooks
                └──────────────────────────────────────────┘
```

`ProfilerBase` is the owner: it holds `BufferPoolManager manager_` as a
member, spawns and joins the mgmt / collector threads, and dispatches
collected buffers to
`Derived::on_buffer_collected` via CRTP. `BufferPoolManager` owns no
threads — it is just the shared data structure both threads access.
`Module` is a stateless trait that tells the generic algorithms how the
subsystem's shared-memory layout is shaped.

## 3. The three roles

### 3.1 `BufferPoolManager<Module>` — data layer

Defined in [`buffer_pool_manager.h`](../src/common/platform/include/host/buffer_pool_manager.h).
Owns:

- `ready_shards_` — mgmt drain → collector hand-off shards, each backed by a
  fixed-capacity SPSC ring plus a cv wakeup for blocking waits.
- `done_shards_` — collector → replenish recycle shards, each backed by a
  fixed-capacity SPSC ring. The producer is the collector shard; the single
  consumer is the replenish thread.
- `recycled_[shard][kind]` — shard-local SPSC lanes of free device buffers;
  runtime drain reads the owning shard while replenish writes returned buffers
  and optional watermark allocations.
- `retired_[shard][kind]` — mutex-protected exceptional holding pools for
  buffers that were removed from a queue but could not be published to the
  next queue. They are drained only during teardown.
- `dev_to_host_` / block ranges — source of truth for `resolve_host_ptr`,
  including carved sub-buffers from one registered allocation block.
- `MemoryOps` — type-erased `alloc / reg / free_` callbacks, plus the
  `shared_mem_host` and `device_id` stashed once at start.

Owns no threads. Every entry point is documented as one of:

- lane-owned SPSC operations (`push_to_ready`, `push_recycled`,
  `pop_recycled`, `drain_done_into_recycled`),
- collector producer operations (`notify_copy_done`, one shard per collector),
- shared operations with narrow internal locking for blocking waits / mappings
  (`wait_pop_ready`, `resolve_host_ptr`),
- lifecycle / exceptional operations (`retire_unqueued_buffer`,
  `set_memory_context`, `release_owned_buffers`, `clear_mappings`).

### 3.2 `ProfilerBase<Derived, Module>` — control layer

Defined in [`profiler_base.h`](../src/common/platform/include/host/profiler_base.h).
Provides:

- The mgmt thread(s), collector thread(s), and their lifecycle (`start` /
  `stop`).
- Split mgmt threads — `mgmt_drain_loop` drains ready queues and refills the
  originating free queue from the current drain shard's local recycled pool
  (`ProfilerAlgorithms<Module>::process_entry` per popped entry), while
  `mgmt_replenish_loop` drains done buffers into shard-local recycled pools
  and keeps optional recycled watermarks topped up by batched allocation. The
  replenish thread never writes device free queues, so the drain hot path
  stays allocation-free and remains the only runtime writer to free queues.
  A one-shot `proactive_replenish` seeds every free queue before the threads
  start. Split drain threads do not bulk-mirror the whole
  shared-memory region; they refresh only their queue indices / entries
  before advancing `queue_heads`. On an empty scan, split drain does a short
  busy-poll window before falling back to the 10 us sleep, so micro-bursts
  are less likely to miss AICPU's bounded wait window.
- Optional collector sharding (`Module::kCollectorThreadCount`) — each
  collector drains one host ready shard and returns finished buffers through
  the matching done shard.
- `poll_and_collect_loop` — per-shard `wait_pop_ready` with a 100 ms cv
  tick, dispatches to `Derived::on_buffer_collected`, then calls
  `manager_.notify_copy_done(...)` itself; idle-timeout hang detector.
- `set_memory_context` / `clear_memory_context` so `Derived::init` can
  stash the alloc/reg/free callbacks before threads start; if init aborts
  before stashing, `start(tf)` becomes a no-op.

`ProfilerAlgorithms<Module>` (in the same
[profiler_base.h](../src/common/platform/include/host/profiler_base.h))
is where the unified algorithms live:

- `try_pop_aicpu_entry` — barrier-correct head/tail advance over the
  per-thread ready queue, with a range-check guard against device-side
  corruption.
- `process_entry` — resolve/copy the popped buffer, refill the originating
  free queue only from the current drain shard's local recycled lane, then
  push to the host ready shard. Runtime drain does not allocate and does not
  consume done shards directly. If the host ready shard is full, the
  undelivered buffer is retired rather than written to the done shard, keeping
  the done shard's producer side collector-only.
- `proactive_replenish` — before worker threads start, top every
  (kind, instance) free queue up to `kSlotCount` and optionally warm
  recycled lanes. If recycled is dry while filling free queues it
  batch-allocates one registered block and carves it into `batch_size(kind)`
  buffers.
- `replenish_recycled_pools` — startup/runtime helper used by
  `proactive_replenish` and `mgmt_replenish_loop` to keep per-kind,
  per-shard recycled lanes above `recycled_warm_target`. Each pass allocates
  `max(kSlotCount, target - current)` buffers when a lane is below target:
  small gaps still amortize HAL registration across a slot-sized block, while
  large gaps return to the watermark in one allocation. It allocates into
  recycled lanes only; it does not publish to device free queues.

### 3.3 `Module` — trait layer

A stateless `struct` per subsystem (`PmuModule`, `L2SwimlaneModule`,
`DepGenModule`, `DumpModule`, `ScopeStatsModule`) that tells the generic
algorithms what the shared-memory layout looks like. The contract lives in the
docblock at the top of
[`profiler_base.h`](../src/common/platform/include/host/profiler_base.h);
the required members are:

| Member | Purpose |
| ------ | ------- |
| `using DataHeader / ReadyEntry / ReadyBufferInfo / FreeQueue` | Layout types |
| `kBufferKinds` | Number of buffer kinds inside each recycled shard |
| `kReadyQueueSize`, `kSlotCount` | AICPU ready queue / free queue depth |
| `kHostPoolQueueSize` | Optional host done ring depth |
| `kHostRecycledQueueSize` | Optional per-kind, per-shard recycled ring depth; defaults to `kHostPoolQueueSize` |
| `kSubsystemName` | Tag used in framework log lines |
| `kMgmtDrainThreadCount` | Optional; number of mgmt drain shards (defaults to 1) |
| `kCollectorThreadCount` | Optional number of collector / host ready-queue shards |
| `refresh_replenish_metadata(mgr, header)` | Optional hook to refresh cached queue metadata before a replenish pass |
| `recycled_warm_target(kind[, shard]) → int` | Optional startup/runtime low-water mark for shard-local recycled lanes |
| `header_from_shm(void*) → DataHeader*` | Cast shared-memory base to header |
| `batch_size(int kind) → int` | Per-kind batch-alloc count |
| `resolve_entry(shm, header, q, entry) → optional<EntrySite>` | Map a popped ready entry to (kind, free_queue, buffer_size, partial info); return `nullopt` to drop |
| `for_each_instance(shm, header, cb)` | Enumerate every (kind, instance, FreeQueue*, buffer_size) for `proactive_replenish` |
| `kind_of(info) → int` | **Multi-kind only.** Tells the framework which recycled bin a finished buffer belongs to. Single-kind modules omit this; the framework passes 0 |

The Module structs are defined alongside their collectors in
[pmu_collector.h](../src/a2a3/platform/include/host/pmu_collector.h),
[l2_swimlane_collector.h](../src/common/platform/include/host/l2_swimlane_collector.h),
[dep_gen_collector.h](../src/common/platform/include/host/dep_gen_collector.h),
[args_dump_collector.h](../src/common/platform/include/host/args_dump_collector.h),
and
[scope_stats_collector.h](../src/common/platform/include/host/scope_stats_collector.h)
— each is a few dozen lines of static methods over the subsystem's own
`DataHeader` / ringbuffer types.

### 3.4 `Derived` — domain layer

Each collector inherits as `class XxxCollector : public ProfilerBase<XxxCollector, XxxModule>`
and only has to provide:

- `void on_buffer_collected(const ReadyBufferInfo& info)` or the optional
  shard-aware overload `void on_buffer_collected(const ReadyBufferInfo& info,
  int collector_shard)` — copy records out of `info.host_buffer_ptr` and
  update collector-specific state (CSV row, in-memory aggregator, file writer
  thread, …). PMU and ArgsDump use the shard id for lock-free local
  accumulation. DepGen and ScopeStats have one orchestrator producer, so they
  retain their single-accumulator path. The framework calls
  `manager_.notify_copy_done(...)` afterwards; **Derived
  must not call it directly.**
- `static constexpr int kIdleTimeoutSec` — bound on no-progress idle in
  the collector loop. Use the subsystem's `PLATFORM_*_TIMEOUT_SECONDS`
  constant.
- `static constexpr const char* kSubsystemName` — appears in the idle
  timeout log line (e.g. `"PMU"`, `"L2Swimlane"`, `"ArgsDump"`).
- `init(...)` and `finalize(...)` — domain-specific setup/teardown.
  `init` must call `set_memory_context()` on the success path so
  `start(tf)` is not a no-op. `finalize` must release framework-owned
  buffers (`release_owned_buffers`) and drop the mapping table
  (`clear_mappings`).

### 3.5 `DeviceProfilerEngine<Module>` — AICPU producer algorithm layer

Defined in
[`profiler_device_engine.h`](../src/common/platform/include/aicpu/profiler_device_engine.h).
This is the device-side counterpart to `ProfilerAlgorithms<Module>`: it
owns the AICPU producer control flow, while each subsystem keeps its record
schema, flush/finalize behavior, and small layout hooks locally.

The engine currently provides:

- `wait_for_ready_queue_space` — bounded wait for a per-thread ready queue
  slot.
- `wait_for_free_queue_entry` — bounded wait for a replacement buffer in a
  free queue, with acquire ordering before reading `buffer_ptrs[]`.
- `enqueue_ready` — write the ready entry, `wmb()`, then advance
  `queue_tails[q]`.
- `pop_free` — pop a free buffer, clear its count, install
  `current_buf_ptr/current_buf_seq`, update any local cache hook, then
  publish with `wmb()`.
- `switch_buffer` — enqueue the full current buffer, clear current state,
  advance seq, then try to install a replacement; on enqueue failure it
  accounts dropped records, clears count, and keeps the workload moving.

Each AICPU writer defines a local `XxxDeviceModule` trait with:

| Member | Purpose |
| ------ | ------- |
| `Context` | Per-call context such as header pointer, ready-queue thread, core/pool id, or local cache pointer |
| `using DataHeader / State / FreeQueue / Buffer` | Device shared-memory layout types |
| `kReadyQueueSize`, `kSlotCount`, `kBackpressureWaitCycles` | Queue geometry and bounded wait budget |
| `header(ctx)`, `ready_thread(ctx)`, `free_queue(state)` | Locate the shared header and queues |
| `current_ptr/set_current_ptr`, `current_seq/set_current_seq` | Access the active device buffer state |
| `count/set_count` | Access the active buffer's record count |
| `write_ready_entry(ctx, tail, ptr, seq)` | Fill subsystem-specific ready-entry fields (`core_index`, `kind`, `thread_index`, `instance_index`, …) |
| `account_dropped`, `on_enqueue_failed`, `on_no_replacement` | Preserve subsystem-specific drop accounting and logs |
| `on_pop_success`, `on_current_cleared`, `on_null_free_slot`, `on_switch_complete` | Preserve local caches and optional switch logs |

Current users:

- ScopeStats, DepGen, ArgsDump, and PMU use the engine for
  ready/free/switch.
- L2Swimlane uses the engine for ready enqueue / free wait primitives and
  AICPU task-buffer pop/switch.
- L2Swimlane scheduler/orchestrator phase pools and AICore rotation remain
  local special cases. Their seq recovery, retry, and AICore-visible head
  publishing rules differ from standard `switch_buffer`.

## 4. End-to-end data flow

```text
  AICPU                       mgmt thread(s)                    collector shard(s)
  ─────                       ──────────────                    ──────────────────
  write record into         try_pop_aicpu_entry(q)
  current free buffer       ──────────────────────────►
                            process_entry:
                              resolve_host_ptr
                              pop recycled[q]
                                (top up originating free_queue)
                              push_to_ready(shard q) ─────────► wait_pop_ready(q)
                                                                Derived::on_buffer_collected
                                                                  (copy records out)
                                                                notify_copy_done(q)
                            ◄────────────────────────────────── done shard q

                                          ▲
                                          │
                            replenish thread:
                            done shard q → recycled[q]
```

The queue shards plus the shard-local recycled pools and the dev↔host map all
live in the single `BufferPoolManager` instance owned by `ProfilerBase`.
Each ready shard has one collector consumer; each done shard is written by
its matching collector and drained by the replenish thread into the same
recycled shard. Split drain refills the originating free queue on the hot
path from `recycled[q]`; replenish does not write free queues at runtime.
Backpressure fallbacks that cannot publish a buffer to ready/free/recycled
place it in `retired_[q][kind]`, which is outside the hot SPSC path and is
released at teardown.

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
    spawn drain mgmt thread(s)
    spawn replenish mgmt thread
                              ← started first; mgmt writes host ready shards
    spawn collector thread(s)

    ... AICPU / AICore execute ...

  ProfilerBase::stop()
    mgmt_running_ = false
    join mgmt thread(s)     ← drain final-pass flushes the last entries into
                              ready shards before exiting
    execution_complete_ = true
    join collector thread(s)← each shard drains once more, then exits

  Derived::finalize(unregister, free)
    manager_.release_owned_buffers([&](void* p) { unregister + free })
    free buffers still held in collector-owned free_queues / current_buf_ptr
    manager_.clear_mappings()
    clear_memory_context()
```

The order in `stop()` is load-bearing: mgmt is joined **before**
`execution_complete_` is signalled so its final-drain output has a
consumer; collectors then drain and exit. On return all host shards are
empty and `on_buffer_collected` has been called for every entry that was in
any shard.

`Derived::finalize` is responsible for the buffers the collector still
owns at stop time (`free_queue` slots and `current_buf_ptr`); the
framework only releases what it had in recycled / retired / done / ready. This
split matters because the AICPU may still be referencing free-queue
buffers via shared memory until execution ends, so they cannot be freed
mid-run by the framework.

## 6. Concurrency invariants

| State | Reader(s) | Writer(s) | Synchronization |
| ----- | --------- | --------- | --------------- |
| `ready_shards_[q]` | collector q | mgmt drain q | fixed-capacity SPSC ring; cv only wakes blocking waits |
| `done_shards_[q]` | replenish | collector q | fixed-capacity SPSC ring |
| `recycled_[shard][kind]` | drain shard at runtime | replenish | fixed-capacity SPSC ring; startup code may pop any shard before threads start |
| `retired_[shard][kind]` | teardown | exceptional fallback paths | mutex-protected holding pool; not used by the hot recycle path |
| `dev_to_host_` / block ranges | mgmt (`resolve_host_ptr`) | init/startup allocation | `mapping_mutex_`; collector touches it only in `release_owned_buffers` / `clear_mappings`, after `stop()` has joined mgmt |
| `MemoryOps` / `shared_mem_host_` / `device_id_` | both threads | start-only | `set_memory_context` is called once before threads spawn; read-only afterwards |
| AICPU per-thread ready queues (`header->queues[q]`) | mgmt (head advance) | AICPU (tail advance) | `read_range_from_device` in split drain, then `write_range_to_device` for `queue_heads[q]` |
| Per-instance `FreeQueue` | AICPU (head advance) | owning drain shard (tail advance) | SPSC ownership; host refreshes `head` before writing `buffer_ptrs[]` / `tail` |

Two things follow:

- `dev_to_host_` has a narrow mapping lock; ready/done/recycled hand-off is
  SPSC by shard, so the hot drain/refill path stays on the owning lane.
- Device-side queue backpressure is bounded for the profiling writers that
  use this protocol. If the host does not make ready-queue space or
  free-queue entries visible within the short wait budget, AICPU records a
  drop and keeps the workload moving instead of spinning indefinitely.
- The AICPU writer publishes a full buffer to the ready queue before
  acquiring its replacement buffer. If no replacement is visible yet, the
  current pointer is cleared and later records first try to recover from
  the free queue before counting a per-record drop. This matters under a
  one-buffer stress shape: the host cannot return a replacement until it
  first observes the full ready buffer.
- The mgmt thread must never zero AICPU-owned fields (`count`, `head`,
  `tail` on the AICPU side). The AICPU is the sole writer to those and
  resets them itself on flush/drop/pop.

## 7. Adding a new collector

1. Define the subsystem's shared-memory types (`DataHeader`,
   `ReadyQueueEntry`, `FreeQueue`, `ReadyBufferInfo`) somewhere both host
   and AICPU can include.
2. Write a `XxxModule` struct satisfying the contract in §3.3. Multi-kind
   modules also implement `kind_of`.
3. If the subsystem has an AICPU device writer, define a local
   `XxxDeviceModule` satisfying §3.5 and use
   `profiling_device::DeviceProfilerEngine<XxxDeviceModule>` for
   ready/free/switch control flow. Keep record fill, flush/finalize, and
   unusual cross-core protocols local unless their semantics match the
   standard engine contract.
4. Write a `XxxCollector : public profiling_common::ProfilerBase<XxxCollector, XxxModule>`:
   - `init(...)`: `rtMalloc` + register pre-allocated buffers, populate
     the shared header, call `register_mapping` per buffer, then call
     `set_memory_context(...)`.
   - `on_buffer_collected(info)`: copy records out of
     `info.host_buffer_ptr`. **Do not** call `notify_copy_done`.
   - `kIdleTimeoutSec`, `kSubsystemName`.
   - `finalize(unregister, free)`: `release_owned_buffers` + free
     collector-owned buffers + `clear_mappings` + `clear_memory_context`.
5. Wire it into `device_runner` so `start(tf)` is called before the
   kernel launch and `stop()` before `finalize`.

Existing collectors are the canonical examples:

- [`PmuCollector`](../src/a2a3/platform/include/host/pmu_collector.h)
  — single kind, per-core instances. See [pmu-profiling.md](dfx/pmu-profiling.md).
- [`DepGenCollector`](../src/common/platform/include/host/dep_gen_collector.h)
  — single kind, one instance. See [dep_gen.md](dfx/dep_gen.md).
- [`ArgsDumpCollector`](../src/common/platform/include/host/args_dump_collector.h)
  — single kind, per-AICPU-thread instances. See [args-dump.md](dfx/args-dump.md).
- [`ScopeStatsCollector`](../src/common/platform/include/host/scope_stats_collector.h)
  — single kind, one instance. See [scope-stats.md](dfx/scope-stats.md).
- [`L2SwimlaneCollector`](../src/common/platform/include/host/l2_swimlane_collector.h)
  — four kinds (AICPU task, scheduler phase, orchestrator phase, AICore
  task), per-core / per-thread instances; the canonical multi-kind example. See
  [l2-swimlane-profiling.md](dfx/l2-swimlane-profiling.md).

## 8. a5 specifics — host-shadow transport

a5's framework headers (under
[`src/common/platform/include/host/`](../src/common/platform/include/host/))
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
   `L2SwimlaneAicpuPhaseHeader::magic`, `core_to_thread[]`), rolling them
   back to whatever the host shadow had at the start of the tick. Per-buffer
   payloads (`L2SwimlaneAicpuTaskBuffer` / `PmuBuffer` /
   `DumpMetaBuffer`) are still pulled on demand inside
   `ProfilerAlgorithms::process_entry` after resolving the host pointer
   for a popped ready entry. The bulk `mirror_shm_to_device` is kept
   for init/teardown where AICPU is not yet running or has exited.
3. **Teardown has one device-release path and one shadow-release path.**
   `release_owned_buffers` canonicalizes any carved sub-buffer back to its
   registered allocation block before calling the collector's `release_fn`.
   Paired `malloc()` shadows tracked in `dev_to_host_` are released by
   `clear_mappings()` after collector-owned queues/current buffers have been
   handled, or by `release_all_owned()` on init rollback.

Most collectors keep `reconcile_counters` passive — same shape as a2a3. It
pulls the post-`stop()` `BufferState`s, logs an error if any
`current_buf_ptr` still references a non-empty buffer (a device flush bug,
since AICPU is the only writer that can clear it), and runs the collector's
counter cross-check/accounting against device-side counters. ArgsDump and
ScopeStats are the recovery-oriented exceptions: on abnormal exit they may
recover a non-empty `current_buf_ptr` host-side before export, so already
recorded dump metadata or scope samples still reach the output files.

### 8.1 Profiling state lives on `KernelArgs`, not `Handshake`

a5 carries the same "profiling out of the runtime synchronization protocol"
design as a2a3 (PR #714) plus the AICore-side rings of PR #709. The runtime
`Handshake` struct contains **no profiling fields** — enablement bits and
per-core ring/reg addresses travel through `KernelArgs`:

| `KernelArgs` field | Producer | Consumer |
| ------------------ | -------- | -------- |
| `enable_profiling_flag` (bitmask) | host (DeviceRunner) | AICPU `kernel.cpp` → `set_l2_swimlane_enabled` / `set_pmu_enabled` / `set_dump_args_enabled`; AICore `KERNEL_ENTRY` → `set_aicore_profiling_flag` |
| `aicore_l2_swimlane_ring_addrs` (table) | host (`L2SwimlaneCollector::initialize`) | AICore `KERNEL_ENTRY` indexes `table[block_idx]` → `set_aicore_l2_swimlane_ring` |
| `aicore_pmu_ring_addrs` (table) | host (`PmuCollector::init`) | AICore `KERNEL_ENTRY` → `set_aicore_pmu_ring` |
| `regs` (per-physical-core register-base table) | host (already required for AICPU MMIO) | AICore `KERNEL_ENTRY` resolves `regs[get_physical_core_id()]` → `set_aicore_pmu_reg_base`; AICore `aicore_execute` caches the value at Phase-3 |

AICore's per-core profiling slots live in
[`aicore/aicore_profiling_state.h`](../src/a5/platform/include/aicore/aicore_profiling_state.h)
— `[[block_local]]` static on onboard, `pthread_key_t` TLS in sim. The
runtime `aicore_execute(runtime, block_idx, core_type)` signature is
unchanged; adding a new profiling field touches `KernelArgs` and this
state surface, never the runtime protocol.

### 8.2 Stable AICore staging ring (decouples AICore write from AICPU buffer rotation)

L2Swimlane and PMU on a5 both use the "AICore writes, AICPU commits" model.
The AICore-side write target is a per-core
[`L2SwimlaneAicoreRing`](../src/a5/platform/include/common/l2_swimlane_profiling.h) /
[`PmuAicoreRing`](../src/a5/platform/include/common/pmu_profiling.h) of
`PLATFORM_{L2,PMU}_AICORE_RING_SIZE` (= 2, dual-issue) slots, allocated
once by the host and addressed by
`BufferState::aicore_ring_ptr` (AICPU-visible) and the per-core
`aicore_*_ring_addrs[block_idx]` (AICore-visible). The address is
never reassigned, so AICore's write target is stable across AICPU's
rotating `L2SwimlaneAicpuTaskBuffer` / `PmuBuffer` flips — flipping is now
fully internal to `*_complete_record` and never crosses into Handshake.

Everything else — Module concept contract, alloc policy
(drain-shard top-up + proactive replenish), `kIdleTimeoutSec` / `kSubsystemName`
contract, mgmt-then-poll start/stop ordering, buffer-pool sizing
constants — matches a2a3 exactly. New collectors should be reviewed
against both arches when added.
