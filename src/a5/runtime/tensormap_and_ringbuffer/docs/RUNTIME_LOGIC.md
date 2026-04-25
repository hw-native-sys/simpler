# PTO2 Runtime System Design

## Overview

PTO2 (Parallel Task Orchestration v2) is a runtime system for executing task graphs on Ascend AI processors. It coordinates four layers of execution:

- **Host** (x86/ARM CPU): compiles kernels, allocates device memory, initializes the Runtime, and launches AICPU/AICore threads.
- **AICPU** (device ARM cores): runs the orchestrator (task graph builder) and scheduler threads.
- **AICore** (AI compute cores): executes kernel functions dispatched by the scheduler.
- **Shared Memory** (Global Memory): ring buffers, task descriptors, heap, and TensorMap shared between orchestrator and schedulers.

```text
┌───────────────────────────────────────────────────────────────────────┐
│                            Host (CPU)                                 │
│  test_*.py (SceneTestCase) → compile kernels → init Runtime           │
│  → upload binaries → launch AICPU/AICore → collect results            │
└───────────────────────────┬───────────────────────────────────────────┘
                            │ device memory / GM
┌───────────────────────────▼───────────────────────────────────────────┐
│                     AICPU (4 threads)                                  │
│  Thread 3: Orchestrator (builds task graph)                           │
│  Threads 0-2: Schedulers (dispatch tasks to AICore)                   │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                   Shared Memory (GM)                             │  │
│  │  SharedMemoryHeader │ TaskDescriptors[] │ DepListPool           │  │
│  │  GM Heap (output buffers)                                       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Scheduler ──Handshake/Registers──► AICore workers (AIC + AIV)        │
└───────────────────────────────────────────────────────────────────────┘
```

---

## 1. Runtime Variants

Three runtime backends exist under `src/runtime/`, each representing a different orchestration and scheduling strategy.

### 1.1 host_build_graph

The host builds the complete task graph before launching device execution. The orchestration SO is loaded and executed on the host CPU.

- **Task storage**: fixed `Task[]` array (up to 131072 tasks)
- **Scheduling**: AICPU receives the pre-built graph and dispatches tasks by traversing dependencies
- **Use case**: development and debugging; no device-side orchestration overhead

### 1.2 aicpu_build_graph

The orchestration runs on an AICPU thread, building the task graph on device. Supports concurrent build + schedule (`build_mode=1`).

- **Task storage**: same `Task[]` array as host_build_graph
- **AicpuBuildApi**: `add_task`, `add_successor_conditional`, `publish_task`, `device_malloc`
- **Use case**: reduced host→device data transfer; graph can depend on device-side data

### 1.3 tensormap_and_ringbuffer (PTO2)

The primary production runtime. Uses ring buffers for task slots and output memory, with a TensorMap for automatic dependency tracking.

- **Task storage**: `PTO2TaskDescriptor[]` in shared memory ring buffer
- **Memory**: GM Heap ring for output buffer allocation
- **Dependencies**: automatically derived from tensor read/write patterns via TensorMap
- **Thread model**: 3 scheduler threads + 1 orchestrator thread on AICPU
- **Multi-ring**: HeapRing, TaskRing, and DepPool are split into `PTO2_MAX_RING_DEPTH` (4) independent instances for nested scope isolation. See [MULTI_RING.md](MULTI_RING.md) for details.
- **Use case**: production workloads; supports streaming, flow control, and large batch sizes

---

## 2. Platform Abstraction

Two platform implementations exist under `src/platform/`, sharing a common interface.

### 2.1 a2a3 (Real Ascend Hardware)

| Component | Description |
| --------- | ----------- |
| `device_runner.cpp` | Uses CANN APIs: `rtMalloc`, `rtMemcpy`, `rtLaunchKernel` |
| `memory_allocator.cpp` | Wraps `rtMalloc`/`rtFree` with allocation tracking |
| `aicore/kernel.cpp` | `KERNEL_ENTRY(aicore_kernel)` → `aicore_execute` |
| `aicpu/kernel.cpp` | `DynTileFwkBackendKernelServer` entry → `aicpu_execute` |
| `spin_hint.h` | ARM `wfe`/`yield` instructions for efficient spinning |

### 2.2 a2a3sim (Thread Simulation)

| Component | Description |
| --------- | ----------- |
| `device_runner.cpp` | Uses `std::thread` to simulate AICPU/AICore |
| `memory_allocator.cpp` | Wraps `malloc`/`free` |
| `aicore/kernel.cpp` | `aicore_execute_wrapper` sets `g_sim_reg_base` per core |
| `upload_kernel_binary` | `dlopen` kernel SO, `dlsym` entry point |

### 2.3 Platform Constants (`platform_config.h`)

| Constant | Value | Description |
| -------- | ----- | ----------- |
| `PLATFORM_MAX_BLOCKDIM` | 24 | Maximum blocks (each = 1 AIC + 2 AIV) |
| `PLATFORM_MAX_AICPU_THREADS` | 4 | AICPU thread count (3 schedulers + 1 orchestrator) |
| `PLATFORM_MAX_AIC_PER_THREAD` | 24 | Max AIC cores per scheduler thread |
| `PLATFORM_MAX_AIV_PER_THREAD` | 48 | Max AIV cores per scheduler thread |
| `PLATFORM_PROF_SYS_CNT_FREQ` | 50 MHz | System counter frequency for profiling |

---

## 3. Shared Memory Layout

The orchestrator and schedulers communicate through a contiguous shared memory region in Global Memory (GM). Each ring level has its own TaskDescriptor and DepListPool sections. See [MULTI_RING.md §4.3–4.4](MULTI_RING.md) for the per-ring shared memory header and handle layout.

```text
┌─────────────────────────────┐  offset 0
│  PTO2SharedMemoryHeader     │  (per-ring flow control + layout, global flags)
├─────────────────────────────┤  aligned
│  Per-ring regions ×4:       │
│    PTO2TaskDescriptor[N]    │  N = task_window_size per ring
│    PTO2TaskPayload[N]       │
│    PTO2TaskSlotState[N]     │
└─────────────────────────────┘
```

### 3.1 SharedMemoryHeader Fields

| Field | Writer | Reader | Purpose |
| ----- | ------ | ------ | ------- |
| `current_task_index` | Orchestrator | Scheduler | Next task ID to allocate (task ring head) |
| `last_task_alive` | Scheduler | Orchestrator | Oldest still-active task (task ring tail) |
| `heap_top` | Orchestrator | Scheduler | Heap ring allocation pointer |
| `heap_tail` | Scheduler | Orchestrator | Heap ring reclamation pointer |
| `orchestrator_done` | Orchestrator | Scheduler | Signals orchestration completion |
| `task_window_size` | Init | Both | Number of task slots (per-ring, in `PTO2SharedMemoryRingHeader`) |
| `heap_size` | Init | Both | Heap total size (per-ring, in `PTO2SharedMemoryRingHeader`) |
| `task_descriptors_offset` | Init | Both | Offset to TaskDescriptor array in SM (per-ring) |
| `total_size` | Init | Both | Total shared memory size |
| `graph_output_ptr` | Orchestrator | Host | Address of final output (packed buffer) |
| `graph_output_size` | Orchestrator | Host | Size of final output in bytes |

### 3.2 Size Calculation

```text
total = ALIGN(Header)
      + Σ_ring [ ALIGN(window_size * sizeof(TaskDescriptor))
               + ALIGN(window_size * sizeof(TaskPayload))
               + ALIGN(window_size * sizeof(TaskSlotState)) ]
```

Alignment is 64 bytes (`PTO2_ALIGN_SIZE`).

---

## 4. Ring Buffer Mechanisms

> **Multi-ring extension**: All three ring buffers (TaskRing, HeapRing, DepPool) are replicated per scope depth. Each ring level has independent watermarks and reclamation. See [MULTI_RING.md](MULTI_RING.md) for details.

### 4.1 Task Ring

The task ring manages task slot allocation with back-pressure flow control.

**Structure** (`PTO2TaskRing`):

- `descriptors`: pointer to `TaskDescriptor[]` in shared memory
- `window_size`: number of slots (power of 2)
- `current_index`: next task ID to allocate (monotonically increasing)
- `last_alive_ptr`: pointer to `header->last_task_alive`

**Slot mapping**: `slot = task_id & (window_size - 1)`

**Allocation** (`pto2_task_ring_alloc`):

```text
active_count = current_index - *last_alive_ptr
if active_count < window_size - 1:
    allocate slot, advance current_index
else:
    spin-wait (back-pressure from scheduler)
```

**Reclamation**: Scheduler threads advance `last_task_alive` via lock-free CAS when the oldest task reaches state CONSUMED (4). This frees slots for reuse.

**Flow control**: When the ring is full, the orchestrator blocks until the scheduler advances `last_task_alive`. With `PTO2_RING_TASK_WINDOW=16` and 208 tasks, slots are recycled ~13 times each.

### 4.2 Heap Ring

The heap ring manages output buffer allocation from a circular GM heap.

**Structure** (`PTO2HeapRing`):

- `base`: GM heap base address
- `size`: total heap size (default 1 GB)
- `top`: allocation pointer (local to orchestrator)
- `tail_ptr`: pointer to `header->heap_tail` (updated by scheduler)

**Allocation**: Buffers are allocated contiguously from `top`. When reaching the end, allocation wraps to the beginning if `tail` has advanced far enough. Buffers never straddle the wrap-around boundary.

**Reclamation**: When `last_task_alive` advances past a task, its `packed_buffer_end` is used to advance `heap_tail`, freeing the memory region.

### 4.3 Dependency List Pool

A simple bump allocator for `PTO2DepListEntry` nodes used in fanin/fanout linked lists.

- **Entry 0**: NULL sentinel (`task_id=-1, next_offset=0`)
- **Allocation**: `pool->top++`, wraps around when full
- **Reclamation**: implicit — old entries become unreachable as `last_task_alive` advances

### 4.4 Flow Control and Back-Pressure

The ring buffer mechanism provides **flow control** between the orchestrator (producer) and the scheduler (consumer). When a ring is exhausted, the orchestrator **blocks** — it cannot submit new tasks or allocate more output memory until the scheduler reclaims slots/space by advancing the watermarks.

**Task Ring back-pressure**: When `active_count = current_index - last_task_alive >= window_size - 1`, `pto2_task_ring_alloc` spin-waits until the scheduler completes tasks and advances `last_task_alive`.

**Heap Ring back-pressure**: When the heap has insufficient contiguous space, `pto2_heap_ring_alloc` spin-waits until the scheduler advances `heap_tail` past completed tasks' output buffers.

**TensorMap pool back-pressure**: When the entry pool is exhausted, `new_entry()` spin-waits on `pto2_orchestrator_sync_tensormap(force=true)` until cleanup frees entries (see Section 5.4).

This back-pressure is essential for correctness with small ring sizes — for example, with `PTO2_RING_TASK_WINDOW=16` and 208 tasks, the orchestrator blocks ~192 times, each time waiting for the scheduler to drain completed tasks before continuing.

### 4.5 Deadlock Detection

A ring that is **too small** can cause a **deadlock**. The root cause is the scope mechanism: each task's `fanout_count` includes a reference from its owning scope. The scope reference is only released when `scope_end()` runs — but `scope_end()` is called by the orchestrator, which is blocked waiting for ring space. This creates a circular dependency:

```text
Orchestrator blocked on task_ring_alloc (ring full)
    → needs scheduler to advance last_task_alive
    → needs tasks to reach CONSUMED state (fanout_count == 0)
    → needs scope_end() to release scope reference
    → needs orchestrator to continue
    → DEADLOCK
```

The runtime detects this automatically by counting spin iterations in the allocation functions:

**Periodic BLOCKED warnings** (every 10,000 spins):

```text
[TaskRing] BLOCKED (Flow Control): current=208, last_alive=192, active=16/16 (100.0%), spins=10000
[HeapRing] BLOCKED: requesting 4096 bytes, available=0, top=65536, tail=0, spins=10000
```

**Deadlock detection** (after 100,000 spins with no progress):

```text
FATAL: Flow Control Deadlock Detected!
Task Ring is FULL and no progress after 100000 spins.
  - Active tasks:  16
  - Window size:   16
Root Cause:
  Tasks cannot transition to CONSUMED state because fanout_count
  includes 1 for the owning scope, and scope_end() requires the
  orchestrator to continue — creating a circular dependency.
Solution:
  Recommended: 32 (at least 2x current active tasks)
```

The FATAL message is logged to the device log and the process exits. The solution is to increase the ring size so that it can hold at least all tasks within the largest parallel scope. For example, if a scope submits 13 tasks, `task_window >= 14` is required (13 + 1 to distinguish full from empty).

**Sizing guideline**: `task_window_size` must be larger than the maximum number of tasks in any single `PTO2_SCOPE`. A safe choice is `2 × max_tasks_per_scope` or simply the default 65536 for production.

---

## 5. TensorMap and Automatic Dependency Tracking

### 5.1 Purpose

TensorMap maintains a mapping from tensor memory regions to their producer task IDs. When a new task reads a tensor (INPUT/INOUT), TensorMap automatically discovers the producer and establishes a dependency edge.

### 5.2 Hash Table Design

- **Key**: tensor base address (`buffer.addr`)
- **Value**: producer task ID, with overlap detection for sub-regions
- **Overlap**: `COVERED` (new region fully contains old) or `OTHER` (partial overlap)
- Sub-tensors of the same base tensor hash to the same bucket, enabling overlap detection

### 5.3 Entry Pool Management

Unlike the Task Ring and Heap Ring, TensorMap entries are **not** managed by a ring buffer. Instead, a **fixed-size pool + free list** is used:

1. **Free list first**: `free_entry_list[]` stores pointers to released entries. Allocation pops from here (O(1)).
2. **Bump allocation**: if free list is empty, `entry_pool[next_entry_idx++]` allocates from the end of the pool.
3. **Blocking reclaim**: if the pool is fully exhausted, `pto2_orchestrator_sync_tensormap(force=true)` reads the latest `last_task_alive` and calls `cleanup_retired` to batch-free all entries belonging to retired tasks, returning them to the free list.

This design avoids the complexity of ring-based wrapping while still being bounded by `PTO2_TENSORMAP_POOL_SIZE` (default 65536 entries).

### 5.4 Stale Entry Cleanup: Three-Layer Defense

TensorMap must ensure entries for retired tasks (`producer_task_id < last_task_alive`) are removed, so that:

- The pool does not grow unboundedly (capacity is finite)
- Lookup performance does not degrade as stale entries accumulate in bucket chains

Three complementary mechanisms achieve this:

**Layer 1 — Chain Truncation during Lookup** (lazy, per-bucket):

Since `insert` always prepends to the bucket head, entries in each bucket chain are in **descending task_id order**. When `pto2_tensormap_lookup` encounters the first stale entry (`producer_task_id < last_task_alive`), all subsequent entries in the chain are guaranteed stale too. The entire tail is truncated in one operation using `prev_in_bucket` pointers for O(1) unlinking.

This guarantees lookup only traverses valid entries — O(valid_entries_in_bucket), not O(total_entries).

**Layer 2 — Periodic Batch Cleanup** (`cleanup_retired`, per-task):

Every time the orchestrator submits a task (Step 0 of `pto2_submit_task`), it calls `pto2_orchestrator_sync_tensormap`. When `last_task_alive` has advanced by more than `PTO2_TENSORMAP_CLEANUP_INTERVAL` (default 64) tasks since the last cleanup, `pto2_tensormap_cleanup_retired` runs:

This uses the **per-task entry chain** (`task_entry_head[task_slot]`) — each task's entries are doubly-linked together at insert time via `next_in_task`/`prev_in_task`, allowing O(entries_per_task) cleanup without scanning the entire pool or all buckets. Freed entries are returned to `free_entry_list` for immediate reuse.

**Layer 3 — Back-Pressure on Pool Exhaustion** (blocking):

If both the free list and bump region are depleted, `new_entry()` blocks until `pto2_orchestrator_sync_tensormap(force=true)` frees entries by advancing `last_task_alive` through `cleanup_retired`.

This forms a back-pressure mechanism analogous to the Task Ring's flow control.

**Summary**:

| Layer | Trigger | Method | Guarantees |
| ----- | ------- | ------ | ---------- |
| Chain Truncation | Every lookup | Truncate stale tail of bucket chain | Lookup only visits valid entries |
| Periodic Cleanup | Every 64 retired tasks | Walk per-task chains, free entries | Pool capacity reclaimed in bounded time |
| Pool Back-Pressure | Pool exhausted | Block until scheduler advances watermark | Hard capacity bound, no OOM |

In steady state, the number of valid TensorMap entries ≈ `active_tasks × avg_outputs_per_task`. With the default `task_window=65536` and `pool_size=65536`, this is well within bounds. With small windows (e.g., `task_window=16`), active entries are even fewer (~16 × a few), and cleanup runs frequently.

### 5.5 Dependency Discovery Flow

When `pto2_submit_task` processes parameters:

1. **INPUT/INOUT**: `pto2_tensormap_lookup` searches for overlapping producers (with chain truncation)
2. For each producer found: `pto2_add_consumer_to_producer` adds the dependency
3. **OUTPUT/INOUT**: `pto2_tensormap_insert` registers the current task as the new producer at bucket head
4. Stale entries are pruned lazily during lookup (Layer 1) and periodically by cleanup (Layer 2)

---

## 6. Task Descriptor and States

### 6.1 PTO2TaskDescriptor (Hot Path)

| Field | Description |
| ----- | ----------- |
| `task_id` | Canonical mixed-task ID (64-bit: `ring_id << 32 \| local_id`). See [MULTI_RING.md §3](MULTI_RING.md). |
| `kernel_id[3]` | Per-slot kernel IDs: `[AIC, AIV0, AIV1]`; `INVALID_KERNEL_ID` = inactive |
| `active_mask` | Bitmask of active subtask slots: `bit0=AIC`, `bit1=AIV0`, `bit2=AIV1` |
| `completed_subtasks` | Atomic counter; each subtask increments on completion. Trigger condition: `completed_subtasks == total_required_subtasks` |
| `fanin_count` | Number of producer dependencies (set by scheduler during wiring) |
| `fanout_lock` | Per-task spinlock for concurrent fanout modification (used by scheduler wiring + completion) |
| `fanout_head` | Head of fanout consumer list (pointer, protected by `fanout_lock`) |
| `fanout_count` | 1 (scope ref) + number of consumers |
| `packed_buffer_base` | Start of packed buffer in GM Heap |
| `packed_buffer_end` | End of packed buffer (for heap reclamation) |

### 6.1b PTO2TaskPayload (Cold Path)

| Field | Description |
| ----- | ----------- |
| `tensors[16]` | Tensor descriptors for parameters |
| `scalar_value[16]` | Scalar parameter values |
| `is_tensor[16]` | Whether each parameter is tensor or scalar |
| `param_count` | Number of valid parameters |
| `fanin_slot_states[]` | Producer slot state pointers (used by `on_task_release`) |
| `fanin_actual_count` | Actual fanin count |

### 6.2 Task State Machine

```text
  [0] PENDING ──fanin satisfied──► [1] READY ──dispatch──► [2] RUNNING
      ▲                                                         │
      │                                                         ▼
  slot recycled ◄── [4] CONSUMED ◄──fanout done── [3] COMPLETED
```

In the scheduler's `task_state[]` array (`std::atomic<PTO2TaskState>`):

- **0 (PENDING)**: waiting for dependencies (`fanin_refcount < fanin_count`)
- **1 (READY)**: all dependencies satisfied, waiting in ready queue
- **2 (RUNNING)**: currently executing on a worker
- **3 (COMPLETED)**: hardware execution complete, output may still be in use
- **4 (CONSUMED)**: output fully consumed, buffers can be released

---

## 7. Orchestrator

### 7.1 PTO2OrchestratorState

The orchestrator runs on AICPU Thread 3 and builds the task graph by calling the user-provided orchestration function.

Key members:

- `rings[PTO2_MAX_RING_DEPTH]`: per-ring `PTO2RingSet` (HeapRing + TaskRing + FaninPool). See [MULTI_RING.md §4.2](MULTI_RING.md).
- `tensor_map`, `tensor_pool`: dependency tracking
- `scope_tasks[]`, `scope_begins[]`, `scope_stack_top`: scope nesting stack (flat buffer partitioned by level)
- `scheduler`: pointer to scheduler state (for wiring queue and ready queue access)
- `gm_heap_base`, `gm_heap_size`: GM heap for output buffers

### 7.2 Task Submission Flow (`pto2_submit_mixed_task`)

| Step | Operation |
| ---- | --------- |
| 0 | `pto2_orchestrator_sync_tensormap` — prune stale TensorMap entries |
| 1 | `pto2_task_ring_alloc` — allocate task slot (may block on flow control) |
| 2 | Initialize task descriptor + slot state, copy parameters |
| 3 | **Lookup**: for each INPUT/INOUT param, search TensorMap for producers; collect producer pointers in `PTO2FaninBuilder` |
| 4 | **Insert**: register OUTPUT/INOUT args in TensorMap |
| 5 | **Record fanin metadata**: store producer pointers in `payload->fanin_inline_slot_states[]` (+ spill pool if >16); increment each producer's `fanout_count` (no lock needed — single writer). This step runs **before** `payload.init()`. |
| 6 | **Push to wiring queue**: push to global `PTO2SpscQueue`; scheduler thread 0 asynchronously wires fanout edges (lock + dep_pool + early_finished check + ready push) |

> **Note**: Fanout wiring (Steps 4–7 in earlier versions) has been moved from the
> orchestrator submit hot path to the scheduler's global `wiring_queue` (SPSC). This reduces the
> orchestrator's shared L2 cache / memory bus pressure, as the orchestrator no longer
> acquires `fanout_lock` or allocates from `dep_pool` during submission.

### 7.3 Deferred Fanout Wiring (Scheduler Wiring Queue)

The orchestrator pushes each submitted task to the global `scheduler->wiring_queue` (a wait-free SPSC queue). Scheduler thread 0 drains this queue in batches, deferring if the queue holds fewer than a full batch of items to reduce contention (unless a final flush is needed at end of execution). For each task:

1. Sets `fanin_count = N + 1` (+1 redundance to prevent premature readiness)
2. For each producer in `payload->fanin_slot_states[]`:
   - **Acquires** the producer's `fanout_lock`
   - Checks `task_state >= COMPLETED` (early-finished optimization)
   - If not completed: prepends consumer to producer's `fanout_head` via `dep_pool.prepend`
   - **Releases** `fanout_lock`
3. Atomically releases the +1 redundance + early_finished count via `fanin_refcount.fetch_add`
4. If all deps satisfied: pushes task to ready queue

The scheduler's completion handler mirrors this:

1. **Acquire** `fanout_lock`, mark `task_state = COMPLETED`, read `fanout_head`, **release** lock
2. Traverse fanout list, incrementing each consumer's `fanin_refcount`
3. Mark `task_state = CONSUMED` when `fanout_refcount` reaches `fanout_count`

This protocol guarantees every consumer is accounted for exactly once.

### 7.4 Scope Mechanism (`PTO2_SCOPE`)

Scopes control the lifetime of intermediate buffers. Each scope:

- Tracks tasks submitted within it via a flat `scope_tasks[]` buffer partitioned by `scope_begins[]`
- On `scope_end`: increments `fanout_refcount` for scope tasks; when it reaches `fanout_count`, the task's packed buffer can be reclaimed

```cpp
PTO2_SCOPE(rt) {
    // Tasks submitted here belong to this scope
    pto2_rt_submit_aic_task(FUNC_QK, args);
    pto2_rt_submit_aiv_task(FUNC_SF, args);
}
// scope_end: scope reference released from all tasks above
```

---

## 8. Scheduler

### 8.1 Thread Model

With `aicpu_thread_num=4`, the AICPU runs 4 threads:

| Thread | Role | Cores |
| ------ | ---- | ----- |
| 0 | Scheduler | 6 AIC + ~13 AIV |
| 1 | Scheduler | 6 AIC + ~13 AIV |
| 2 | Scheduler | 6 AIC + ~13 AIV |
| 3 | Orchestrator | none |

Core assignment: AICs and AIVs are divided equally among the 3 scheduler threads.

### 8.2 Scheduler Main Loop

Each scheduler thread runs a tight loop with two main phases:

**Phase 1 — Completion Handling**:

- Poll register `COND` on each managed core
- When `TASK_FIN_STATE` detected: record completion timestamps, call `on_subtask_complete(task_id, subslot)` to increment the completion counter; when `completed_subtasks == total_required_subtasks`, trigger `on_mixed_task_complete(task_id)` which marks `task_state[slot] = COMPLETED`, acquires fanout lock, traverses fanout list (incrementing consumers' `fanin_refcount`), marks `task_state[slot] = CONSUMED`, and advances `last_task_alive` watermark

**Phase 2 — Dispatch**:

- For each idle core: pop a task from the matching shape-based ready queue (lock-free MPMC Vyukov queue, one per resource shape)
- Build `PTO2DispatchPayload` from `TaskDescriptor` with `task_id`, `subslot`, `kernel_id`, and `core_type`
- Write task pointer to `Handshake.task`, signal AICore via register `DATA_MAIN_BASE`

After these phases, the scheduler updates profiling headers and checks for termination (all tasks completed and orchestrator done).

### 8.3 Ready Queue Design

Ready queues use a lock-free bounded MPMC (Vyukov) design:

- One `PTO2ReadyQueue` per resource shape (5 shapes: `AIC_ONLY`, `AIV_X1`, `AIV_X2`, `AIC_AIV_X1`, `AIC_AIV_X2`)
- **Push**: any thread (orchestrator via `init_task`, or scheduler on completion) pushes newly-ready tasks to the queue matching `pto2_active_mask_to_shape(task->active_mask)`
- **Pop**: scheduler threads pop from the queue matching the idle core's resource shape
- Per-slot sequence counters prevent ABA problems
- `enqueue_pos` and `dequeue_pos` are on separate cache lines to avoid false sharing

### 8.4 Watermark Advancement (last_task_alive)

After a task reaches state CONSUMED (4), the scheduler tries to advance `last_task_alive`:

```text
advance_ring_pointers(ring_id):  // protected by per-ring advance_lock
    while la < current_task_index:
        if task_state[la & mask] < CONSUMED: break
        reset slot for reuse
        la++
    sync_to_sm()  // release-store last_task_alive
```

This is protected by a per-ring try-lock (`advance_lock`) in `RingSchedState`, ensuring only one scheduler thread advances a given ring's watermark at a time.

### 8.5 SchedulerContext

All scheduler-side state and methods live in `SchedulerContext` (`runtime/scheduler/scheduler_context.h`). It is held as a `sched_ctx_` member of `AicpuExecutor`; `AicpuExecutor` is a thin wrapper that owns the lifecycle atomics and the orchestration SO handle, and delegates everything else to `SchedulerContext`.

Public surface (called from `AicpuExecutor::init/run/deinit`):

| Method | Phase | Purpose |
| ------ | ----- | ------- |
| `init(runtime, thread_num, sched_thread_num, orch_to_sched, regs_base)` | once per run | Handshake + assign cores, reset counters, latch `regs_base`, bind `func_id_to_addr_` |
| `bind_runtime(rt)` | device-orch only | Wire `sched_` to `rt->scheduler` once the orchestrator thread creates `rt` |
| `resolve_and_dispatch(runtime, thread_idx)` | per scheduler thread | Main dispatch loop |
| `shutdown(thread_idx)` | per thread on exit | `platform_deinit_aicore_regs` for this thread's cores |
| `on_orchestration_done(runtime, rt, thread_idx, total_tasks)` | orchestrator thread | Publish core assignments, latch task count, fold inline-completed tasks, flip `orchestrator_done_`, drive orch→sched core transition (or `emergency_shutdown` on fatal) |
| `deinit()` | once per run | Reset every scheduler-owned field to its post-construction default |
| Read-only accessors | various | `aic_count()` / `aiv_count()` / `is_completed()` / `completed_tasks_count()` / `wait_pto2_init_complete()` |

Private internals are split across three .cpp files by responsibility:

- `scheduler_completion.cpp` — completion polling, drain protocol
- `scheduler_dispatch.cpp` — task dispatch loop and helpers
- `scheduler_cold_path.cpp` — exit checks, stall diagnostics, profiling, lifecycle (`init/deinit`), core management (`handshake_all_cores` / `assign_cores_to_threads` / `reassign_cores_for_all_threads` / `emergency_shutdown`), and `on_orchestration_done`

`AicpuExecutor` calls neither `handshake_*`, `assign_*`, `reassign_*`, nor `emergency_shutdown` directly — they are private, invoked only by `init` and `on_orchestration_done`.

---

## 9. AICore Worker Interaction

### 9.1 Handshake Protocol

Each AICore worker has a `Handshake` struct in shared memory:

| Field | Direction | Purpose |
| ----- | --------- | ------- |
| `task` | AICPU→AICore | Pointer to `PTO2DispatchPayload` |
| `control` | AICPU→AICore | 0=normal, 1=shutdown |
| `perf_records_addr` | AICPU→AICore | Performance buffer address |

### 9.2 Register-Based Dispatch

Instead of polling a shared-memory status flag, the production protocol uses hardware registers.

> **Multi-ring note**: `task_id` is 64-bit but registers are 32-bit. A per-core monotonic dispatch counter (`s_dispatch_seq`) replaces `task_id` in register writes to prevent collisions. See [MULTI_RING.md §6](MULTI_RING.md).

| Register | Direction | Usage |
| -------- | --------- | ----- |
| `DATA_MAIN_BASE` | AICPU→AICore | Write `task_id` to dispatch (idle=0x7FFFFFFD); `EXIT_SIGNAL` to shut down |
| `COND` | AICore→AICPU | `[bit31=state, bits30:0=task_id]`: ACK (state=0) or FIN (state=1) |

**AICore execution loop**:

1. Poll `DATA_MAIN_BASE` for value != AICPU_IDLE_TASK_ID
2. Read payload from `Handshake.task`
3. Write ACK to `COND`
4. Execute kernel function via `func_id_to_addr` lookup
5. Write FIN to `COND`

### 9.3 PTO2DispatchPayload

Built by the scheduler from `PTO2TaskDescriptor`:

| Field | Description |
| ----- | ----------- |
| `task_id` | Mixed-task identifier (for completion aggregation) |
| `subslot` | Which subtask slot this dispatch represents (`AIC`, `AIV0`, or `AIV1`) |
| `kernel_id` | Function ID for this subtask slot |
| `core_type` | AIC or AIV |
| `function_bin_addr` | GM address of compiled kernel binary |
| `num_args` | Number of arguments |
| `args[]` | Tensor addresses and scalar values |

---

## 10. Kernel and Orchestration Loading

### 10.1 Kernel Binary Loading

1. **Host** compiles each kernel source (`.cpp`) into a binary (`.o` or `.so`)
2. `host_api.upload_kernel_binary(func_id, binary, size)` uploads to GM
3. The returned GM address is stored in `Runtime.func_id_to_addr_[func_id]`
4. When dispatching, the scheduler copies this address into `PTO2DispatchPayload.function_bin_addr`

### 10.2 Orchestration SO Loading

1. **Host** compiles the orchestration source into a shared library (`.so`)
2. The SO binary is embedded into `Runtime.device_orch_so_storage_[]` and copied to device
3. **AICPU Thread 3** writes the SO to a temp file, calls `dlopen`
4. `dlsym("aicpu_orchestration_config")` returns configuration (expected arg count)
5. `dlsym("aicpu_orchestration_entry")` returns the orchestration function pointer
6. Thread 3 creates a `PTO2Runtime`, calls the orchestration function within a `PTO2_SCOPE`
7. After orchestration completes: `dlclose`, delete temp file

### 10.3 Thread Startup Synchronization

| Flag | Set by | Waited by | Purpose |
| ---- | ------ | --------- | ------- |
| `runtime_init_ready_` | Thread 3 | Threads 0-2 | Runtime and SM handle initialized |
| `pto2_init_done_` | First init thread | Others | One-time memset of arrays started (exchange guard) |
| `pto2_init_complete_` | Init thread | Thread 3 + others | One-time init of per-task arrays done |

Startup sequence:

1. Thread 3: create SM handle + runtime → set `runtime_init_ready_`
2. Scheduler threads: wait for `runtime_init_ready_` → one thread wins `pto2_init_done_` exchange → memset per-task arrays → set `pto2_init_complete_`; other threads wait for `pto2_init_complete_`
3. Thread 3: wait for `pto2_init_complete_` → configure orchestrator-scheduler pointers
4. Scheduler threads: enter main loop
5. Thread 3: call orchestration function → set `orchestrator_done_`

---

## 11. PTO2 Orchestration API

The orchestration API is defined in `pto_orchestration_api.h`. Orchestration code depends only on this header.

### 11.1 Core API

| Function/Macro | Purpose |
| -------------- | ------- |
| `pto2_rt_submit_task(mixed_kernels, args)` | Submit a mixed task with `MixedKernels` struct |
| `pto2_rt_submit_aic_task(kernel_id, args)` | Convenience: submit AIC-only task |
| `pto2_rt_submit_aiv_task(kernel_id, args)` | Convenience: submit AIV-only task |
| `PTO2_SCOPE() { ... }` | RAII scope for buffer lifetime |
| `pto2_rt_orchestration_done()` | Signal orchestration complete |

### 11.2 Parameter Construction

| Function | Description |
| -------- | ----------- |
| `make_tensor_external(ptr, shapes, ndim, dtype)` | Wrap an existing device pointer as a tensor |
| `TensorCreateInfo(shapes, ndim, dtype)` | Describe a runtime-created output buffer |
| `Arg::add_input(tensor)` | INPUT parameter — read by the task |
| `Arg::add_output(create_info)` | OUTPUT parameter — runtime allocates and returns a Tensor |
| `Arg::add_inout(tensor)` | INOUT parameter — existing tensor read then written |
| `Arg::add_scalar(value)` | 64-bit scalar parameter |

### 11.3 Resource Shapes

Tasks are queued by resource shape, which is derived from the `active_mask` in the `MixedKernels` struct:

| Shape | Active Mask | Description |
| ----- | ----------- | ----------- |
| `AIC_ONLY` | AIC only | AIC cores (matrix multiplication) |
| `AIV_X1` | AIV0 or AIV1 only | Single AIV core (vector operations) |
| `AIV_X2` | AIV0 + AIV1 | Two AIV cores |
| `AIC_AIV_X1` | AIC + one AIV | AIC + single AIV core |
| `AIC_AIV_X2` | AIC + AIV0 + AIV1 | Full cluster (AIC + two AIV cores) |

### 11.4 Orchestration Export Interface

Each orchestration `.so` must export:

```cpp
extern "C" PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count);
extern "C" void aicpu_orchestration_entry(uint64_t* args, int arg_count);
```

---

## 12. Example: Batch Paged Attention

### 12.1 Kernel Configuration (`kernel_config.py`)

```python
KERNELS = [
    {"func_id": 0, "name": "QK",      "source": "aic/aic_qk_matmul.cpp",       "core_type": "aic"},
    {"func_id": 1, "name": "SF",      "source": "aiv/aiv_softmax_prepare.cpp", "core_type": "aiv"},
    {"func_id": 2, "name": "PV",      "source": "aic/aic_pv_matmul.cpp",       "core_type": "aic"},
    {"func_id": 3, "name": "UP",      "source": "aiv/aiv_online_update.cpp",   "core_type": "aiv"},
    {"func_id": 5, "name": "AIV_HUB", "source": "aiv/aiv_hub.cpp",            "core_type": "aiv"},
]

ORCHESTRATION = {
    "source": "orchestration/paged_attention_orch.cpp",
    "function_name": "aicpu_orchestration_entry",
}

RUNTIME_CONFIG = {
    "runtime": "tensormap_and_ringbuffer",
    "aicpu_thread_num": 4,
    "block_dim": 24,
}
```

### 12.2 Orchestration Structure

```cpp
void aicpu_orchestration_entry(uint64_t* args, int arg_count) {
    // Unpack args: query, key_cache, value_cache, block_table, context_lens, out, config
    for (q_idx = 0; q_idx < q_loop; q_idx++) {
        for (batch_start = 0; batch_start < batch; batch_start += IN_CORE_BATCH) {
            PTO2_SCOPE() {
                // Describe accumulator tensors (oi, li, mi) with TensorCreateInfo
                // Submit AIV_HUB to initialize accumulators
                for (bn = 0; bn < max_bn; bn++) {
                    // Allocate intermediate tensors (sij, pij, mij, lij, oi_new)
                    // Submit QK (CUBE) → SF (VECTOR) → PV (CUBE) → UP (VECTOR)
                }
            }
        }
    }
}
```
