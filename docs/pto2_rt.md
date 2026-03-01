# PTO2 Runtime System Design

## Overview

PTO2 (Parallel Task Orchestration v2) is a runtime system for executing task graphs on Ascend AI processors. It coordinates four layers of execution:

- **Host** (x86/ARM CPU): compiles kernels, allocates device memory, initializes the Runtime, and launches AICPU/AICore threads.
- **AICPU** (device ARM cores): runs the orchestrator (task graph builder) and scheduler threads.
- **AICore** (AI compute cores): executes kernel functions dispatched by the scheduler.
- **Shared Memory** (Global Memory): ring buffers, task descriptors, heap, and TensorMap shared between orchestrator and schedulers.

```
┌───────────────────────────────────────────────────────────────────────┐
│                            Host (CPU)                                 │
│  golden.py → code_runner.py → compile kernels → init Runtime          │
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
- **Use case**: production workloads; supports streaming, flow control, and large batch sizes

---

## 2. Platform Abstraction

Two platform implementations exist under `src/platform/`, sharing a common interface.

### 2.1 a2a3 (Real Ascend Hardware)

| Component | Description |
|-----------|-------------|
| `device_runner.cpp` | Uses CANN APIs: `rtMalloc`, `rtMemcpy`, `rtLaunchKernel` |
| `memory_allocator.cpp` | Wraps `rtMalloc`/`rtFree` with allocation tracking |
| `aicore/kernel.cpp` | `KERNEL_ENTRY(aicore_kernel)` → `aicore_execute` |
| `aicpu/kernel.cpp` | `DynTileFwkBackendKernelServer` entry → `aicpu_execute` |
| `spin_hint.h` | ARM `wfe`/`yield` instructions for efficient spinning |

### 2.2 a2a3sim (Thread Simulation)

| Component | Description |
|-----------|-------------|
| `device_runner.cpp` | Uses `std::thread` to simulate AICPU/AICore |
| `memory_allocator.cpp` | Wraps `malloc`/`free` |
| `aicore/kernel.cpp` | `aicore_execute_wrapper` sets `g_sim_reg_base` per core |
| `upload_kernel_binary` | `dlopen` kernel SO, `dlsym` entry point |

### 2.3 Platform Constants (`platform_config.h`)

| Constant | Value | Description |
|----------|-------|-------------|
| `PLATFORM_MAX_BLOCKDIM` | 24 | Maximum blocks (each = 1 AIC + 2 AIV) |
| `PLATFORM_MAX_AICPU_THREADS` | 4 | AICPU thread count (3 schedulers + 1 orchestrator) |
| `PLATFORM_MAX_AIC_PER_THREAD` | 24 | Max AIC cores per scheduler thread |
| `PLATFORM_MAX_AIV_PER_THREAD` | 48 | Max AIV cores per scheduler thread |
| `PLATFORM_PROF_SYS_CNT_FREQ` | 50 MHz | System counter frequency for profiling |

---

## 3. Shared Memory Layout

The orchestrator and schedulers communicate through a contiguous shared memory region in Global Memory (GM):

```
┌─────────────────────────────┐  offset 0
│  PTO2SharedMemoryHeader     │  (flow control, config, sync flags)
├─────────────────────────────┤  aligned
│  PTO2TaskDescriptor[N]      │  N = task_window_size (default 65536)
├─────────────────────────────┤  aligned
│  PTO2DepListEntry[M+1]      │  M = dep_list_pool_size (entry 0 = NULL sentinel)
└─────────────────────────────┘
```

### 3.1 SharedMemoryHeader Fields

| Field | Writer | Reader | Purpose |
|-------|--------|--------|---------|
| `current_task_index` | Orchestrator | Scheduler | Next task ID to allocate (task ring head) |
| `last_task_alive` | Scheduler | Orchestrator | Oldest still-active task (task ring tail) |
| `heap_top` | Orchestrator | Scheduler | Heap ring allocation pointer |
| `heap_tail` | Scheduler | Orchestrator | Heap ring reclamation pointer |
| `orchestrator_done` | Orchestrator | Scheduler | Signals orchestration completion |
| `task_window_size` | Init | Both | Number of task slots |
| `heap_size` | Init | Both | Heap total size |

### 3.2 Size Calculation

```
total = ALIGN(Header) + ALIGN(window_size * sizeof(TaskDescriptor))
      + ALIGN((dep_pool_size + 1) * sizeof(DepListEntry))
```

Alignment is 64 bytes (`PTO2_ALIGN_SIZE`).

---

## 4. Ring Buffer Mechanisms

### 4.1 Task Ring

The task ring manages task slot allocation with back-pressure flow control.

**Structure** (`PTO2TaskRing`):
- `descriptors`: pointer to `TaskDescriptor[]` in shared memory
- `window_size`: number of slots (power of 2)
- `current_index`: next task ID to allocate (monotonically increasing)
- `last_alive_ptr`: pointer to `header->last_task_alive`

**Slot mapping**: `slot = task_id & (window_size - 1)`

**Allocation** (`pto2_task_ring_alloc`):
```
active_count = current_index - *last_alive_ptr
if active_count < window_size - 1:
    allocate slot, advance current_index
else:
    spin-wait (back-pressure from scheduler)
```

**Reclamation**: Scheduler threads advance `last_task_alive` via lock-free CAS when the oldest task reaches state CONSUMED (3). This frees slots for reuse.

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

```
Orchestrator blocked on task_ring_alloc (ring full)
    → needs scheduler to advance last_task_alive
    → needs tasks to reach CONSUMED state (fanout_count == 0)
    → needs scope_end() to release scope reference
    → needs orchestrator to continue
    → DEADLOCK
```

The runtime detects this automatically by counting spin iterations in the allocation functions:

**Periodic BLOCKED warnings** (every 10,000 spins):
```
[TaskRing] BLOCKED (Flow Control): current=208, last_alive=192, active=16/16 (100.0%), spins=10000
[HeapRing] BLOCKED: requesting 4096 bytes, available=0, top=65536, tail=0, spins=10000
```

**Deadlock detection** (after 100,000 spins with no progress):
```
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

1. **Free list first**: `free_entry_list[]` stores indices of released entries. Allocation pops from here (O(1)).
2. **Bump allocation**: if free list is empty, `next_entry_idx++` allocates from the end of the pool.
3. **Blocking reclaim**: if the pool is fully exhausted, `pto2_orchestrator_sync_tensormap(force=true)` reads the latest `last_task_alive` and calls `cleanup_retired` to batch-free all entries belonging to retired tasks, returning them to the free list.

This design avoids the complexity of ring-based wrapping while still being bounded by `PTO2_TENSORMAP_POOL_SIZE` (default 65536 entries).

### 5.4 Stale Entry Cleanup: Three-Layer Defense

TensorMap must ensure entries for retired tasks (`producer_task_id < last_task_alive`) are removed, so that:
- The pool does not grow unboundedly (capacity is finite)
- Lookup performance does not degrade as stale entries accumulate in bucket chains

Three complementary mechanisms achieve this:

**Layer 1 — Chain Truncation during Lookup** (lazy, per-bucket):

Since `insert` always prepends to the bucket head, entries in each bucket chain are in **descending task_id order**. When `pto2_tensormap_lookup` encounters the first stale entry (`producer_task_id < last_task_alive`), all subsequent entries in the chain are guaranteed stale too. The entire tail is truncated in one operation:

```cpp
// pto2_tensormap_lookup: chain truncation
if (!pto2_tensormap_entry_valid(tm, entry)) {
    *prev_ptr = -1;  // cut chain here
    while (offset >= 0) {
        stale->in_bucket = false;  // mark for reuse
        offset = stale->next_in_bucket;
    }
    return;
}
```

This guarantees lookup only traverses valid entries — O(valid_entries_in_bucket), not O(total_entries).

**Layer 2 — Periodic Batch Cleanup** (`cleanup_retired`, per-task):

Every time the orchestrator submits a task (Step 0 of `pto2_submit_task`), it calls `pto2_orchestrator_sync_tensormap`. When `last_task_alive` has advanced by more than `PTO2_TENSORMAP_CLEANUP_INTERVAL` (default 64) tasks since the last cleanup, `pto2_tensormap_cleanup_retired` runs:

```cpp
// pto2_tensormap_cleanup_retired: batch free by per-task chain
for (task_id = old_last_task_alive; task_id < new_last_task_alive; task_id++) {
    task_slot = task_id & (TASK_WINDOW_SIZE - 1);
    offset = task_entry_head[task_slot];
    while (offset >= 0) {
        free_entry(offset);   // remove from bucket + return to free list
        offset = next;
    }
    task_entry_head[task_slot] = -1;
}
```

This uses the **per-task entry chain** (`task_entry_head[task_slot]`) — each task's entries are linked together at insert time, allowing O(entries_per_task) cleanup without scanning the entire pool or all buckets. Freed entries are returned to `free_entry_list` for immediate reuse.

**Layer 3 — Back-Pressure on Pool Exhaustion** (blocking):

If both the free list and bump region are depleted, `new_entry()` spins on `pto2_orchestrator_sync_tensormap(force=true)`, waiting for the scheduler to advance `last_task_alive` so that `cleanup_retired` can free entries:

```cpp
// PTO2TensorMap::new_entry: back-pressure
while (free_num == 0) {
    pto2_orchestrator_sync_tensormap(this, /*force=*/true);
}
```

This forms a back-pressure mechanism analogous to the Task Ring's flow control.

**Summary**:

| Layer | Trigger | Method | Guarantees |
|-------|---------|--------|------------|
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

### 6.1 PTO2TaskDescriptor

| Field | Description |
|-------|-------------|
| `task_id` | Monotonically increasing ID |
| `kernel_id` | Function ID (maps to compiled kernel binary) |
| `worker_type` | CUBE (AIC) or VECTOR (AIV) |
| `fanin_head` | Head of fanin dependency list (DepListPool offset) |
| `fanin_count` | Number of producer dependencies |
| `fanout_lock` | Spinlock for concurrent fanout modification |
| `fanout_head` | Head of fanout consumer list |
| `fanout_count` | 1 (scope ref) + number of consumers |
| `packed_buffer_base/end` | GM heap region for output buffers |
| `output_index[]` | Maps outputs to param indices |
| `params[]` | Tensor and scalar parameters |

### 6.2 Task State Machine

```
  [0] INITIAL ──scan/orch_ready──► [1] READY ──dispatch──► RUNNING
      ▲                                                        │
      │                                                        ▼
  slot recycled ◄── [3] CONSUMED ◄──fanout done── [2] COMPLETED
```

In the scheduler's `s_pto2_task_completed[]` array:
- **0**: not yet ready (initial or recycled)
- **1**: ready for dispatch (all fanin satisfied)
- **2**: hardware execution complete
- **3**: fanout traversed, fully consumed

---

## 7. Orchestrator

### 7.1 PTO2OrchestratorState

The orchestrator runs on AICPU Thread 3 and builds the task graph by calling the user-provided orchestration function.

Key members:
- `task_ring`, `heap_ring`, `dep_pool`: ring buffer state
- `tensor_map`, `tensor_pool`: dependency tracking
- `scope_tasks[]`, `scope_stack_top`: scope nesting stack
- `aicpu_fanin_refcount`, `aicpu_task_completed`, `aicpu_completed_by_task`: pointers to scheduler-side arrays for parallel mode

### 7.2 Task Submission Flow (`pto2_submit_task`)

| Step | Operation |
|------|-----------|
| 0 | `pto2_orchestrator_sync_tensormap` — prune stale TensorMap entries |
| 1 | `pto2_task_ring_alloc` — allocate task slot (may block on flow control) |
| 1b | Reset `completed[slot]=0`, `completed_by_task[slot]=-1` for recycled slots |
| 2 | Initialize task descriptor, copy parameters |
| 3 | **Lookup**: for each INPUT/INOUT param, search TensorMap for producers |
| 4 | **Dependency**: `pto2_add_consumer_to_producer` for each producer found |
| 5 | **Heap alloc**: `pto2_alloc_packed_buffer` for OUTPUT params (addr=0) |
| 6 | **Insert**: register OUTPUT/INOUT params in TensorMap |
| 7 | **Fanin**: finalize `fanin_count`; if already satisfied, push to orch_ready_queue |
| 8 | **Publish**: `STORE_RELEASE(current_task_index)` makes task visible to scanners |

### 7.3 Lock Protocol for Concurrent Dependency Setup

The orchestrator and scheduler run concurrently. When adding a consumer to a producer's fanout list:

1. **Orchestrator acquires** the producer's `fanout_lock`
2. **Check early-return**: if `completed[prod_slot] >= 2` AND `completed_by_task[prod_slot] == producer_id`, the producer already finished — directly increment the consumer's refcount
3. **Normal path**: prepend consumer to the producer's fanout list
4. **Unlock**

The scheduler's completion handler mirrors this:
1. Set `completed_by_task[slot] = task_id` (RELEASE)
2. Set `completed[slot] = 2` (RELEASE)
3. **Acquire** `fanout_lock`, read `fanout_head`, **release** lock
4. Traverse fanout, incrementing each consumer's `fanin_refcount`

This lock protocol guarantees every consumer is accounted for exactly once.

### 7.4 Scope Mechanism (`PTO2_SCOPE`)

Scopes control the lifetime of intermediate buffers. Each scope:
- Tracks tasks submitted within it via `scope_tasks[]`
- On `scope_end`: decrements `fanout_count` for scope tasks; when it reaches 0, the task's packed buffer can be reclaimed

```cpp
PTO2_SCOPE(rt) {
    // Tasks submitted here belong to this scope
    pto2_rt_submit_task(rt, FUNC_QK, PTO2_WORKER_CUBE, params, n);
    pto2_rt_submit_task(rt, FUNC_SF, PTO2_WORKER_VECTOR, params, n);
}
// scope_end: scope reference released from all tasks above
```

---

## 8. Scheduler

### 8.1 Thread Model

With `aicpu_thread_num=4`, the AICPU runs 4 threads:

| Thread | Role | Cores |
|--------|------|-------|
| 0 | Scheduler | 6 AIC + ~13 AIV |
| 1 | Scheduler | 6 AIC + ~13 AIV |
| 2 | Scheduler | 6 AIC + ~13 AIV |
| 3 | Orchestrator | none |

Core assignment: AICs and AIVs are divided equally among the 3 scheduler threads.

### 8.2 Scheduler Main Loop

Each scheduler thread runs a tight loop with four phases:

**Phase 1 — Completion Handling**:
- Poll register `COND` on each managed core
- When `TASK_FIN_STATE` detected: record completion timestamps, set `completed[slot]=2`, acquire fanout lock, traverse fanout list, set `completed[slot]=3`, advance `last_task_alive` watermark

**Phase 2 — Dispatch**:
- For each idle core: pop a task from the ready queue (own shard first, then steal from other shards)
- Build `PTO2DispatchPayload` from `TaskDescriptor`
- Write task pointer to `Handshake.task`, signal AICore via register `DATA_MAIN_BASE`

**Phase 3 — Incremental Scan**:
- Atomically claim task indices from `next_scan_index`
- For root tasks (`fanin_count == 0`): CAS `completed[slot]` 0→1, push to ready queue

**Phase 4 — Orch Ready Queue Drain**:
- Consume entries pushed by the orchestrator's early-ready path (Step 7 in submit)
- CAS `completed[slot]` 0→1, push to ready queue

### 8.3 Ready Queue Sharding and Work Stealing

Ready queues are sharded to reduce lock contention:

- `active_shards` (default 3, configurable via `PTO2_READY_QUEUE_SHARDS`)
- Separate queues for AIC and AIV tasks, each with `active_shards` shards
- **Push**: thread `t` pushes to shard `t % active_shards`
- **Pop**: try own shard first, then scan other shards (work stealing)

### 8.4 Watermark Advancement (last_task_alive)

After a task reaches state 3 (CONSUMED), the scheduler tries to advance `last_task_alive`:

```
while la < current_task_index:
    if completed[la & mask] < 3: break
    reset fanin_refcount[la & mask] = 0
    CAS(last_task_alive, la, la+1)
    advance heap_tail from task's packed_buffer_end
    la++
```

This is lock-free (CAS-based) and multiple scheduler threads can attempt it concurrently.

---

## 9. AICore Worker Interaction

### 9.1 Handshake Protocol

Each AICore worker has a `Handshake` struct in shared memory:

| Field | Direction | Purpose |
|-------|-----------|---------|
| `task` | AICPU→AICore | Pointer to `PTO2DispatchPayload` |
| `control` | AICPU→AICore | 0=normal, 1=shutdown |
| `perf_records_addr` | AICPU→AICore | Performance buffer address |

### 9.2 Register-Based Dispatch

Instead of polling `Handshake.task_status`, the production protocol uses hardware registers:

| Register | Direction | Usage |
|----------|-----------|-------|
| `DATA_MAIN_BASE` | AICPU→AICore | Write `task_id + 1` to dispatch; `EXIT_SIGNAL` to shut down |
| `COND` | AICore→AICPU | `[bit31=state, bits30:0=task_id]`: ACK (state=0) or FIN (state=1) |

**AICore execution loop**:
1. Poll `DATA_MAIN_BASE` for non-zero value
2. Read payload from `Handshake.task`
3. Write ACK to `COND`
4. Execute kernel function via `func_id_to_addr` lookup
5. Write FIN to `COND`

### 9.3 PTO2DispatchPayload

Built by the scheduler from `PTO2TaskDescriptor`:

| Field | Description |
|-------|-------------|
| `task_id` | Task identifier |
| `kernel_id` | Function ID |
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
|------|--------|-----------|---------|
| `sm_header_ready_` | Thread 3 | Threads 0-2 | SM header initialized |
| `pto2_init_complete_` | First init thread | Others | One-time memset of arrays done |
| `orch_pointers_ready_` | Thread 3 | Threads 0-2 | Parallel mode pointers configured |

Startup sequence:
1. Thread 3: create SM handle → set `sm_header_ready_`
2. Scheduler threads: wait for `sm_header_ready_` → one-time init → set `pto2_init_complete_`
3. Thread 3: wait for `pto2_init_complete_` → configure pointers → set `orch_pointers_ready_`
4. Scheduler threads: wait for `orch_pointers_ready_` → enter main loop
5. Thread 3: call orchestration function → set `orchestrator_done`

---

## 11. PTO2 Orchestration API

The orchestration API is defined in `pto_orchestration_api.h`. Orchestration code depends only on this header.

### 11.1 Core API

| Function/Macro | Purpose |
|----------------|---------|
| `pto2_rt_submit_task(rt, kernel_id, worker_type, params, n)` | Submit a task with parameters |
| `PTO2_SCOPE(rt) { ... }` | RAII scope for buffer lifetime |
| `pto2_rt_orchestration_done(rt)` | Signal orchestration complete |
| `pto2_rt_init_tensor_pool(rt)` | Initialize tensor pool for `make_tensor()` |

### 11.2 Parameter Construction

| Function | Description |
|----------|-------------|
| `make_tensor_external(ptr, shapes, ndim, dtype)` | Wrap an existing device pointer as a tensor |
| `make_tensor(shapes, ndim, dtype)` | Create an intermediate tensor (addr=0, allocated by runtime from heap) |
| `make_input_param(tensor)` | INPUT parameter — read by the task |
| `make_output_param(tensor)` | OUTPUT parameter — written by the task (auto-allocated if addr=0) |
| `make_inout_param(tensor)` | INOUT parameter — read then written |
| `make_scalar_param(value)` | 64-bit scalar parameter |

### 11.3 Worker Types

| Type | Target |
|------|--------|
| `PTO2_WORKER_CUBE` | AIC cores (matrix multiplication) |
| `PTO2_WORKER_VECTOR` | AIV cores (vector operations) |

### 11.4 Orchestration Export Interface

Each orchestration `.so` must export:

```cpp
extern "C" PTO2OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count);
extern "C" int aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count);
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
void aicpu_orchestration_entry(PTO2Runtime* rt, uint64_t* args, int arg_count) {
    // Unpack args: query, key_cache, value_cache, block_table, context_lens, out, config
    for (q_idx = 0; q_idx < q_loop; q_idx++) {
        for (batch_start = 0; batch_start < batch; batch_start += IN_CORE_BATCH) {
            PTO2_SCOPE(rt) {
                // Allocate accumulator tensors (oi, li, mi) via make_tensor()
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

The task graph per chunk (16 batches):
```
AIV_HUB ──► QK ──► SF ──► PV ──► UP
                                    │
             QK ──► SF ──► PV ──► UP  (next block, depends on UP above via INOUT oi/li/mi)
```

With `batch=256`, `IN_CORE_BATCH=16`: 16 chunks × 13 tasks = 208 tasks, parallelizable across cores.

### 12.3 Golden Test Cases (`golden.py`)

```python
ALL_CASES = {
    "Case1":       {"batch": 1,   "num_heads": 16, "head_dim": 16,  "context_len": 16},
    "CaseBatch256": {"batch": 256, "num_heads": 1,  "head_dim": 256, "context_len": 16},
    ...
}

def generate_inputs(params) -> list:
    # Returns [(name, tensor_or_scalar), ...] for host→device transfer
    return [("query", query), ("key_cache", key_cache), ..., ("out", out), ("config", config)]

def compute_golden(tensors, params):
    # PyTorch reference implementation of online softmax paged attention
    tensors["out"][:] = paged_attention(...)
```

---

## 13. End-to-End Execution Flow

### 13.1 Build and Run (`run_example.py` / `code_runner.py`)

```
1. Parse kernel_config.py (KERNELS, ORCHESTRATION, RUNTIME_CONFIG)
2. Compile in parallel:
   - Runtime shared library
   - Orchestration SO
   - Each kernel binary (AIC/AIV)
3. Load host binary: bind_host_binary() → Runtime class
4. For each test case:
   a. golden.py:generate_inputs() → func_args, arg_types, arg_sizes
   b. runtime.initialize(orch_so, func_name, func_args, arg_types, arg_sizes, kernels)
      → allocates device memory, uploads binaries, prepares SM and heap
   c. launch_runtime(runtime, aicpu_threads=4, block_dim=24)
      → spawns AICPU + AICore threads
   d. runtime.finalize() → copy results back to host
   e. Compare output vs golden.py:compute_golden()
```

### 13.2 Device-Side Execution Timeline

```
Time ──────────────────────────────────────────────────────────────────────►

Thread 3:  [create SM] [wait init] [set pointers] [orchestrate: submit 208 tasks] [done]
                │            ▲           │
                ▼            │           ▼
Threads 0-2: [wait SM] [init arrays] [wait ptrs] [scan/dispatch/complete loop] [shutdown]
                                                       │
                                                       ▼
AICore:                                          [execute kernels...]
```

---

## 14. Configurable Parameters

### 14.1 Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `PTO2_RING_TASK_WINDOW` | 65536 | Task ring window size (power of 2, >= 4) |
| `PTO2_RING_HEAP` | 1 GB | GM heap size (>= 1024) |
| `PTO2_RING_DEP_POOL` | 65536 | Dependency list pool size (>= 16) |
| `PTO2_READY_QUEUE_SHARDS` | 3 | Ready queue shard count per core type |
| `PA_CASE` | Case1 | Test case name for batch_paged_attention |
| `PA_SEQ_LEN` | - | Comma-separated per-batch sequence lengths |

### 14.2 Compile-Time Constants (`pto_runtime2_types.h`)

| Constant | Value | Description |
|----------|-------|-------------|
| `PTO2_TASK_WINDOW_SIZE` | 65536 | Default task window |
| `PTO2_HEAP_SIZE` | 1 GB | Default heap size |
| `PTO2_DEP_LIST_POOL_SIZE` | 65536 | Default dep list pool |
| `PTO2_TENSORMAP_POOL_SIZE` | 65536 | TensorMap entry pool |
| `PTO2_TENSORMAP_NUM_BUCKETS` | 65536 | TensorMap hash buckets |
| `PTO2_ALIGN_SIZE` | 64 | Memory alignment |
| `PTO2_PACKED_OUTPUT_ALIGN` | 1024 | Output buffer alignment |

---

## 15. Relation to pypto Frontend

The `docs/pypto-frontend-coding-style.md` describes the Python-to-C++ code generation pipeline:

### 15.1 Function Types in pypto

| Type | Description |
|------|-------------|
| **Opaque** | Default function type; may contain `pl.incore()` calls |
| **Orchestration** | Host/AICPU orchestration function; calls InCore functions |
| **InCore** | AICore kernel subgraph (load/compute/store) |

### 15.2 Code Generation Pipeline

```
pypto IR ──► Orchestration Codegen ──► orchestration.cpp (uses PTO2 API)
pypto IR ──► InCore Codegen ──► kernel.cpp (AIC/AIV kernels)
```

The generated orchestration code uses the same PTO2 API described in Section 11:
- `make_tensor_external()` for external inputs/outputs
- `make_tensor()` for intermediate buffers
- `pto2_rt_submit_task()` for kernel submission
- `PTO2_SCOPE()` for buffer lifetime management

Dependencies are inferred automatically by the TensorMap from tensor read/write patterns — the orchestration code does not need to specify explicit dependency edges.

### 15.3 Backend Targets

| Backend | Output | Description |
|---------|--------|-------------|
| PTO | `.pto` → `ptoas` → C++ | PTO ISA assembly |
| CCE | C++ with `set_flag`/`wait_flag` | Direct C++ with synchronization |

---

## 16. Planned Feature: In-Cluster Function Group Scheduling

Sections 10–11 (in the pypto-frontend-coding-style) describe the **language-level** semantics of cluster allocation and block_incore functions. This section describes the **runtime-level** changes required to support these features in the PTO2 runtime, orchestration codegen, and scheduler.

### 16.1 Function Group as a Scheduling Unit

All incore functions submitted between `allocate_cluster()` and `free_cluster()` (or scope-based automatic release) form an **in-cluster function group**. The runtime must treat this group as a co-scheduled unit: every task in the group executes on the **same physical cluster** identified by `clusterID`.

The key invariant:

```
allocate_cluster() → clusterID
    submit_task(kernel_A, clusterID, ...)   ─┐
    submit_task(kernel_B, clusterID, ...)    │  function group
    submit_task(kernel_C, clusterID, ...)   ─┘
free_cluster(clusterID)   // or automatic release when clusterID tensor leaves scope
```

All tasks within the group carry the same `clusterID` constraint. The scheduler dispatches them **only** to the cores belonging to that cluster, while still respecting data dependencies for ordering.

### 16.2 Required Changes to PTO2TaskDescriptor

The current `PTO2TaskDescriptor` must be extended to record function group membership:

| New Field | Type | Description |
|-----------|------|-------------|
| `cluster_id` | `int32_t` | ID of the allocated cluster (-1 = unconstrained) |
| `group_id` | `int32_t` | Function group identifier (all tasks in the same allocate/free scope share the same group_id) |

When `cluster_id >= 0`, the scheduler **must not** dispatch the task to any core outside the designated cluster. When `cluster_id == -1`, the task follows the current unconstrained scheduling policy.

### 16.3 Required Changes to Orchestration API

New API functions for orchestration code (generated or hand-written):

```cpp
// Allocate a cluster. Blocks if no cluster is available.
// Returns a clusterID (integer) identifying the allocated cluster.
int32_t pto2_rt_allocate_cluster(PTO2Runtime* rt);

// Release a cluster back to the free pool.
// All tasks in the group must have completed before release.
void pto2_rt_free_cluster(PTO2Runtime* rt, int32_t cluster_id);

// Submit a task constrained to a specific cluster.
void pto2_rt_submit_task_clustered(PTO2Runtime* rt, int kernel_id,
                                    int worker_type, PTOParam* params,
                                    int n, int32_t cluster_id);
```

**Scope-based usage pattern** (generated by codegen):

```cpp
PTO2_SCOPE(rt) {
    int32_t cid = pto2_rt_allocate_cluster(rt);  // may block

    // All tasks in this group are pinned to cluster cid
    pto2_rt_submit_task_clustered(rt, FUNC_A, PTO2_WORKER_VECTOR, ..., cid);
    pto2_rt_submit_task_clustered(rt, FUNC_B, PTO2_WORKER_CUBE,   ..., cid);
    pto2_rt_submit_task_clustered(rt, FUNC_C, PTO2_WORKER_VECTOR, ..., cid);

    pto2_rt_free_cluster(rt, cid);
    // or: automatic release when scope ends and clusterID tensor is reclaimed
}
```

### 16.4 Scheduler Changes: Cluster-Aware Dispatch

The scheduler must be extended to support cluster-constrained tasks:

1. **Cluster ↔ Core mapping**: A static mapping from `cluster_id` to the set of physical cores (e.g., cluster 0 = {AIC0, AIV0, AIV1}). This mapping is platform-specific and configured at initialization.

2. **Ready queue partitioning**: When popping a task for a core, the scheduler checks `task.cluster_id`:
   - If `-1`: dispatch to any idle core of the correct type (current behavior).
   - If `>= 0`: dispatch **only** to a core belonging to that cluster.

3. **Cluster free pool**: A ring or bitset tracking which clusters are currently free. `allocate_cluster` pops from this pool (blocking if empty); `free_cluster` pushes back.

4. **Dependency ordering within a group**: Tasks within a function group are still ordered by TensorMap dependencies (PIPE_IN/PIPE_OUT produce read/write edges). The scheduler respects these edges as usual — cluster pinning only constrains *where*, not *when*.

### 16.5 Cluster Allocation Back-Pressure

`pto2_rt_allocate_cluster` uses the same spin-wait pattern as the ring buffer allocators:

```
spin_count = 0
while (no free cluster):
    spin_count++
    if spin_count % BLOCK_NOTIFY_INTERVAL == 0:
        LOG_WARN("[Cluster] BLOCKED: no free cluster, spins=%d", spin_count)
    if spin_count >= CLUSTER_SPIN_LIMIT:
        LOG_ERROR("FATAL: Cluster allocation deadlock — all clusters occupied")
        exit(1)
```

This provides the same deadlock detection as the task ring and heap ring (Section 4.5).

---

## 17. Planned Feature: block_incore (SPMD → MPMD) Task Submission

### 17.1 Current Approach: SPMD Expanded to MPMD

A **block_incore** function is written as a single SPMD kernel parameterized by `(block_dim, block_id)`. At the runtime level, the orchestration layer **expands** this single logical SPMD call into `block_dim` **individual MPMD tasks**, each with a distinct `block_id`:

```
block_incore call (block_dim=N):
    ──► submit_task(kernel, block_id=0, ...)
    ──► submit_task(kernel, block_id=1, ...)
    ──► ...
    ──► submit_task(kernel, block_id=N-1, ...)
```

Each expanded task is an independent `PTO2TaskDescriptor` submitted through the standard `pto2_rt_submit_task` path. The scheduler treats them as N separate tasks that can be dispatched to any available cores.

### 17.2 Orchestration Codegen for block_incore

The generated orchestration code for a `block_incore` call produces a loop:

```cpp
PTO2_SCOPE(rt) {
    for (int bid = 0; bid < block_dim; bid++) {
        PTOParam params[] = {
            make_input_param(input),
            make_output_param(output),
            make_scalar_param(block_dim),
            make_scalar_param(bid),
        };
        pto2_rt_submit_task(rt, KERNEL_FUNC_ID, PTO2_WORKER_VECTOR, params, 4);
    }
}
```

The kernel binary is the same for all `block_dim` tasks — only the `block_id` scalar parameter differs. The runtime's TensorMap tracks per-task tensor dependencies as usual.

### 17.3 Combined with In-Cluster Function Group

When block_incore is used **within** a cluster function group, each of the `block_dim` expanded tasks carries the `cluster_id` constraint.

If the block_incore function group requires `block_dim` clusters (one per block, as described in Section 11.2), the orchestration allocates `block_dim` clusters and assigns each block to its own cluster:

```cpp
int32_t cluster_ids[block_dim];
for (int bid = 0; bid < block_dim; bid++) {
    cluster_ids[bid] = pto2_rt_allocate_cluster(rt);
}

PTO2_SCOPE(rt) {
    for (int bid = 0; bid < block_dim; bid++) {
        // Each block's function group runs on its own cluster
        pto2_rt_submit_task_clustered(rt, FUNC_A, PTO2_WORKER_VECTOR, ..., cluster_ids[bid]);
        pto2_rt_submit_task_clustered(rt, FUNC_B, PTO2_WORKER_CUBE,   ..., cluster_ids[bid]);
        pto2_rt_submit_task_clustered(rt, FUNC_C, PTO2_WORKER_VECTOR, ..., cluster_ids[bid]);
    }
}

for (int bid = 0; bid < block_dim; bid++) {
    pto2_rt_free_cluster(rt, cluster_ids[bid]);
}
```

### 17.4 Performance Considerations and Future Optimization

The SPMD-to-MPMD expansion is the simplest correct approach, but has overhead:

| Concern | Current (MPMD expansion) | Potential Optimization |
|---------|--------------------------|------------------------|
| Task descriptors | `block_dim` descriptors per call | Batch descriptor: single descriptor with `block_dim` field |
| Orchestrator submission | O(block_dim) `submit_task` calls | Single `submit_block_task` call |
| Scheduler scan | O(block_dim) tasks to scan and dispatch | Group-aware dispatch: scan one, expand to block_dim dispatches |
| TensorMap entries | O(block_dim × params) entries | Shared-tensor optimization: one entry per logical tensor |
| Ring pressure | block_dim slots consumed simultaneously | Block-aware flow control: reserve block_dim slots atomically |

**Measurement-first strategy**: The current MPMD expansion is used as the baseline. Performance profiling (Perfetto swimlane, scheduler overhead breakdown) will identify whether the submission overhead, ring pressure, or TensorMap pressure is the bottleneck. Optimization is applied only where measured data shows a need.

---

## 18. Planned Feature: block_incore as InCore Function (Cube + Vector)

### 18.1 InCore Function Structure

An **incore function** (see Section 15.1) is a subgraph of load/compute/store operations that executes on AICore. A single incore function may involve **both** AIC (cube/matrix) and AIV (vector) cores working together — for example, a fused matmul+activation where the matmul runs on AIC and the activation runs on AIV.

A **block_incore** function can also be an incore function. This means each block instance is itself an incore subgraph that may require **both a cube kernel and a vector kernel** co-operating on the same data.

### 18.2 Task Submission for InCore block_incore

When a block_incore function is an incore function consisting of a cube kernel and a vector kernel, the orchestration expands each block into **two tasks** (or more, depending on the pipeline depth):

```
block_incore call (block_dim=N, incore = cube + vector):
    for bid in 0..N-1:
        submit_task(cube_kernel,   WORKER_CUBE,   ..., block_id=bid)
        submit_task(vector_kernel, WORKER_VECTOR, ..., block_id=bid)
```

The TensorMap automatically tracks the dependency between the cube and vector tasks through their shared intermediate tensors (the cube kernel writes the intermediate, the vector kernel reads it).

### 18.3 Cluster Binding for InCore block_incore

When combined with cluster allocation, both the cube and vector tasks of each block are pinned to the **same cluster**, ensuring they execute on co-located cores with local interconnect:

```cpp
int32_t cid = pto2_rt_allocate_cluster(rt);
PTO2_SCOPE(rt) {
    // Cube kernel on AIC within cluster cid
    pto2_rt_submit_task_clustered(rt, CUBE_KERNEL, PTO2_WORKER_CUBE, ..., cid);
    // Vector kernel on AIV within cluster cid (depends on cube output via TensorMap)
    pto2_rt_submit_task_clustered(rt, VEC_KERNEL,  PTO2_WORKER_VECTOR, ..., cid);
}
pto2_rt_free_cluster(rt, cid);
```

The intermediate data between cube and vector can use PIPE_IN/PIPE_OUT (local interconnect, no GM allocation) or regular tensors (GM-backed), depending on whether the cluster's local interconnect is used.

### 18.4 Execution Model Summary

```
                      block_incore (block_dim=4, incore=cube+vector)
                      │
        ┌─────────────┼─────────────┬─────────────┐
        ▼             ▼             ▼             ▼
    Block 0       Block 1       Block 2       Block 3
    Cluster 0     Cluster 1     Cluster 2     Cluster 3
   ┌────────┐    ┌────────┐    ┌────────┐    ┌────────┐
   │AIC: cube│    │AIC: cube│    │AIC: cube│    │AIC: cube│
   │   │     │    │   │     │    │   │     │    │   │     │
   │   ▼     │    │   ▼     │    │   ▼     │    │   ▼     │
   │AIV: vec │    │AIV: vec │    │AIV: vec │    │AIV: vec │
   └────────┘    └────────┘    └────────┘    └────────┘
   (local pipe)  (local pipe)  (local pipe)  (local pipe)
```

Each block runs its cube and vector kernels on the same cluster. Data flows through the local interconnect (TPUSH/TPOP) within each cluster. Cross-block data flows through global memory.

### 18.5 Required Data Structure Changes (Summary)

| Component | Change |
|-----------|--------|
| `PTO2TaskDescriptor` | Add `cluster_id`, `group_id`, `block_id`, `block_dim` fields |
| `PTO2SharedMemoryHeader` | Add cluster free pool (bitset or ring) |
| Orchestration API | Add `allocate_cluster`, `free_cluster`, `submit_task_clustered` |
| Scheduler | Cluster-aware dispatch, cluster→core mapping table |
| Ready Queue | Optional: per-cluster queues for pinned tasks |
| TensorMap | No change — PIPE_IN/PIPE_OUT handled as minimal-shape tensors |
| Codegen | Generate cluster allocation + block_dim expansion loop + clustered submit |
