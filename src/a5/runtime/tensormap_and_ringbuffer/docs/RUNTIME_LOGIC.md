# `tensormap_and_ringbuffer` Runtime Design

## Overview

`tensormap_and_ringbuffer` is a runtime for executing task graphs on Ascend AI processors. It coordinates four layers of execution:

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

Two runtime backends exist under `src/runtime/`, each representing a different orchestration and scheduling strategy.

### 1.1 host_build_graph

The host builds the complete task graph before launching device execution. The orchestration SO is loaded and executed on the host CPU.

- **Task storage**: fixed `Task[]` array (up to 131072 tasks)
- **Scheduling**: AICPU receives the pre-built graph and dispatches tasks by traversing dependencies
- **Use case**: development and debugging; no device-side orchestration overhead

### 1.2 tensormap_and_ringbuffer

The primary production runtime. Uses ring buffers for task slots and output memory, with a TensorMap for automatic dependency tracking.

- **Task storage**: `TaskDescriptor[]` in shared memory ring buffer
- **Memory**: GM Heap ring for output buffer allocation
- **Dependencies**: automatically derived from tensor read/write patterns via TensorMap
- **Thread model**: `aicpu_thread_num - 1` scheduler threads + 1 orchestrator
  thread on AICPU; the default five-thread configuration uses 4 schedulers
- **Multi-ring**: HeapRing, TaskRing, and DepPool are split into `CHIP_MAX_RING_DEPTH` (4) independent instances for nested scope isolation. See [MULTI_RING.md](MULTI_RING.md) for details.
- **Use case**: production workloads; supports streaming, flow control, and large batch sizes

---

## 2. Platform Abstraction

Two platform implementations exist under `src/platform/`, sharing a common interface.

### 2.1 a5 (Real Ascend Hardware)

| Component | Description |
| --------- | ----------- |
| `device_runner.cpp` | Uses CANN APIs: `rtMalloc`, `rtMemcpy`, `rtLaunchKernel` |
| `memory_allocator.cpp` | Wraps `rtMalloc`/`rtFree` with allocation tracking |
| `aicore/kernel.cpp` | `KERNEL_ENTRY(aicore_kernel)` → `aicore_execute` |
| `aicpu/kernel.cpp` | `DynTileFwkBackendKernelServer` entry → `aicpu_execute` |
| `spin_hint.h` | ARM `wfe`/`yield` instructions for efficient spinning |

### 2.2 a5sim (Thread Simulation)

| Component | Description |
| --------- | ----------- |
| `device_runner.cpp` | Uses `std::thread` to simulate AICPU/AICore |
| `memory_allocator.cpp` | Wraps `malloc`/`free` |
| `aicore/kernel.cpp` | `aicore_execute_wrapper` sets `g_sim_reg_base` per core |
| `upload_chip_callable_buffer` | Copy ChipCallable bytes to a host scratch, `dlopen` each child SO, `dlsym` "kernel_entry", patch the scratch's `resolved_addr_` with the function pointer |

### 2.3 Platform Constants (`platform_config.h`)

| Constant | Value | Description |
| -------- | ----- | ----------- |
| `PLATFORM_MAX_BLOCKDIM` | 36 | Maximum blocks (each = 1 AIC + 2 AIV) |
| `PLATFORM_MAX_AICPU_THREADS` | 5 | Maximum active AICPU threads (up to 4 schedulers + 1 orchestrator) |
| `PLATFORM_MAX_AIC_PER_THREAD` | 36 | Max AIC cores per scheduler thread |
| `PLATFORM_MAX_AIV_PER_THREAD` | 72 | Max AIV cores per scheduler thread |
| `PLATFORM_PROF_SYS_CNT_FREQ` | 1000 MHz | System counter frequency for profiling |

### 2.4 Host Temporary Buffer

TRB bind normally allocates one device buffer per ordinary non-child tensor
during host-side argument staging, copies input bytes as needed, records
copy-back metadata, and frees those temporary buffers during runtime
validation. TRB replaces those per-run malloc/free pairs with a retained buffer
that its runs reuse. This is always on for TRB — an internal allocation
optimization, not user-facing configuration.

The buffer is owned per pipeline slot, not per runner: TRB's task args are
`HOST_PER_RUN`, so two runs holding different slot leases stage through
different buffers even though they share arena bank 0 for device scratch.

The platform side is deliberately thin: `DeviceRunnerBase` only remembers a
`{addr, size}` slot per pipeline slot, exposed through two HostApi callbacks —
`get_retained_temp_buffer` and `set_retained_temp_buffer`. It is not an
allocator; all grow/pack/slice logic lives in `runtime_maker.cpp`
(`RetainedTempBump`).

On each trb bind, `RetainedTempBump`:

- packs the run's non-child, non-empty tensors to a required size, aligning
  each slice up to 1024 bytes (TRB kernels require 1024-aligned device
  pointers, which `device_malloc` already guarantees, so the retained base and
  every 1024-aligned slice stay aligned without a base fix-up);
- reads the slot via `get_retained_temp_buffer`; if `required` exceeds the
  retained size it `device_free`s the old buffer, `device_malloc`s a new one,
  and writes it back via `set_retained_temp_buffer` (no data preserved across
  this grow — the buffer is per-run scratch). Smaller later runs keep the
  larger buffer; the slot only grows;
- bump-slices each tensor from the retained base at the next 1024-aligned
  offset. Slices always fit because the buffer was sized from the same
  tensors; a miss is a caller bug (reported, bind fails). The runtime never
  falls back to `device_malloc` mid-run.

Slices are recorded as `BufferNoop` leases: per-tensor release is a no-op, and
the retained buffer is neither freed at end of run nor per run — each slot's
buffer lives on the runner and is freed once in `finalize`. The uniform host-runtime contract
requires the retained-buffer callbacks on every backend; bind has no
per-tensor allocation fallback.

Public device-memory APIs keep their original semantics. `device_malloc_ctx`,
`device_free_ctx`, `Worker.malloc()`, and `Worker.free()` still allocate and
free caller-owned device memory directly; the retained buffer only affects
TRB's internal temporary tensor staging.

---

## 3. Shared Memory Layout

The orchestrator and schedulers communicate through a contiguous shared memory region in Global Memory (GM). Each ring level has its own TaskDescriptor and DepListPool sections. See [MULTI_RING.md §4.3–4.4](MULTI_RING.md) for the per-ring shared memory header and handle layout.

```text
┌─────────────────────────────┐  offset 0
│  SharedMemoryHeader     │  (per-ring flow control + layout, global flags)
├─────────────────────────────┤  aligned
│  Per-ring regions ×4:       │
│    TaskDescriptor[N]    │  N = task_window_size per ring
│    TaskPayload[N]       │
│    ChipTaskSlotState[N]     │
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
| `task_window_size` | Init | Both | Number of task slots (per-ring, in `SharedMemoryRingHeader`) |
| `heap_size` | Init | Both | Heap total size (per-ring, in `SharedMemoryRingHeader`) |
| `task_descriptors_offset` | Init | Both | Offset to TaskDescriptor array in SM (per-ring) |
| `total_size` | Init | Both | Total shared memory size |

### 3.2 Size Calculation

```text
total = ALIGN(Header)
      + Σ_ring [ ALIGN(window_size * sizeof(TaskDescriptor))
               + ALIGN(window_size * sizeof(TaskPayload))
               + ALIGN(window_size * sizeof(ChipTaskSlotState)) ]
```

Alignment is 64 bytes (`CHIP_ALIGN_SIZE`).

---

## 4. Ring Buffer Mechanisms

> **Multi-ring extension**: All three ring buffers (TaskRing, HeapRing, DepPool) are replicated per scope depth. Each ring level has independent watermarks and reclamation. See [MULTI_RING.md](MULTI_RING.md) for details.

### 4.1 Task Ring

The task ring manages task slot allocation with back-pressure flow control.

**Structure** (the task-window half of `TaskAllocator`):

- `descriptors`: pointer to `TaskDescriptor[]` in shared memory
- `window_size`: number of slots (power of 2)
- `current_index`: next task ID to allocate (monotonically increasing)
- `last_alive_ptr`: pointer to `header->last_task_alive`

**Slot mapping**: `slot = task_id & (window_size - 1)`

**Allocation** (`TaskAllocator::alloc`):

```text
active_count = current_index - *last_alive_ptr
if active_count < window_size - 1:
    allocate slot, advance current_index
else:
    spin-wait (back-pressure from scheduler)
```

**Reclamation**: Scheduler threads advance `last_task_alive` via lock-free CAS when the oldest task reaches state CONSUMED (4). This frees slots for reuse.

**Flow control**: When the ring is full, the orchestrator blocks until the scheduler advances `last_task_alive`. With a 16-slot `runtime_env.ring_task_window` and 208 tasks, slots are recycled ~13 times each.

### 4.2 Heap Ring

The heap ring manages output buffer allocation from a circular GM heap.

**Structure** (the heap half of the same `TaskAllocator`):

- `base`: GM heap base address
- `size`: total heap size (default 1 GB)
- `top`: allocation pointer (local to orchestrator)
- `tail_ptr`: pointer to `header->heap_tail` (updated by scheduler)

**Allocation**: Buffers are allocated contiguously from `top`. When reaching the end, allocation wraps to the beginning if `tail` has advanced far enough. Buffers never straddle the wrap-around boundary.

**Reclamation**: When `last_task_alive` advances past a task, its `packed_buffer_end` is used to advance `heap_tail`, freeing the memory region.

When an empty ring is parked at a non-zero offset and neither free arc can hold a request that fits the full
capacity, allocation restarts at offset zero: `heap_tail` resets to zero and `heap_top` advances past the new
allocation. Reclaim markers from tasks in the preceding coordinate space are ignored until the first post-rebase
allocation retires.

### 4.3 Dependency List Pool

A simple bump allocator for `DepListEntry` nodes used in fanin/fanout linked lists.

- **Entry 0**: NULL sentinel (`task_id=-1, next_offset=0`)
- **Allocation**: `pool->top++`, wraps around when full
- **Reclamation**: implicit — old entries become unreachable as `last_task_alive` advances

### 4.4 Flow Control and Back-Pressure

The ring buffer mechanism provides **flow control** between the orchestrator (producer) and the scheduler (consumer). When a ring is exhausted, the orchestrator **blocks** — it cannot submit new tasks or allocate more output memory until the scheduler reclaims slots/space by advancing the watermarks.

**Task Ring back-pressure**: When `active_count = current_index - last_task_alive >= window_size - 1`, `TaskAllocator::alloc` spin-waits until the scheduler completes tasks and advances `last_task_alive`.

**Heap Ring back-pressure**: When the heap has insufficient contiguous space, `TaskAllocator::alloc` spin-waits until the scheduler advances `heap_tail` past completed tasks' output buffers.

**TensorMap pool back-pressure**: Before STEP 4 registers a task's outputs, the orchestrator's `ensure_tensormap_capacity` reserves pool space for the inserts. When the shared entry pool is exhausted, it reclaims retired entries across all rings and spin-waits until reclaim actually frees entries, with a 500 ms wall-clock deadlock backstop (see Section 5.4).

On A5, a reclaim consumer that observes no reclaim progress for 10 ms sets its
ring bit in `publication_request_mask`. The scheduler force-publishes its exact
local `last_task_alive` under `advance_lock` and acknowledges the request before
the orchestrator uses the watermark for structural deadlock detection.
TensorMap pressure requests this publication from every ring because all rings
share one entry pool. K=16 batching is enabled only after every reclaim
consumer is wired to the current request/ack masks; incomplete wiring keeps
per-advance publication.

This back-pressure is essential for correctness with small ring sizes — for example, with a 16-slot `runtime_env.ring_task_window` and 208 tasks, the orchestrator blocks ~192 times, each time waiting for the scheduler to drain completed tasks before continuing.

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

**Sizing guideline**: `task_window_size` must be larger than the maximum number of tasks in any single `SIMPLER_SCOPE`. A safe choice is `2 × max_tasks_per_scope` or simply the default 65536 for production.

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
3. **Blocking reclaim**: if the pool is short of the inserts a task needs, the orchestrator's `ensure_tensormap_capacity` reads the latest `last_task_alive` for every ring and calls `reclaim_retired_all` (`cleanup_retired` per ring) to batch-free entries belonging to retired tasks, returning them to the free list, before the inserts proceed.

This design avoids the complexity of ring-based wrapping while still being bounded by `CHIP_TENSORMAP_POOL_SIZE` (default 65536 entries).

### 5.4 Stale Entry Cleanup: Three-Layer Defense

TensorMap must ensure entries for retired tasks (`producer_task_id < last_task_alive`) are removed, so that:

- The pool does not grow unboundedly (capacity is finite)
- Lookup performance does not degrade as stale entries accumulate in bucket chains

Three complementary mechanisms achieve this:

**Layer 1 — Chain Truncation during Lookup** (lazy, per-bucket):

Since `insert` always prepends to the bucket head, entries in each bucket chain are in **descending task_id order**. When `ChipTensorMap::lookup` encounters the first stale entry (`producer_task_id < last_task_alive`), all subsequent entries in the chain are guaranteed stale too. The entire tail is truncated in one operation using `prev_in_bucket` pointers for O(1) unlinking.

This guarantees lookup only traverses valid entries — O(valid_entries_in_bucket), not O(total_entries).

**Layer 2 — Periodic Batch Cleanup** (`cleanup_retired`, per-task):

Every time the orchestrator submits a task (Step 0 of `OrchestratorState::submit_task`), it calls `ChipTensorMap::sync_tensormap`. When `last_task_alive` has advanced by more than `CHIP_TENSORMAP_CLEANUP_INTERVAL` (default 64) tasks since the last cleanup, `ChipTensorMap::cleanup_retired` runs:

This uses the **per-task entry chain** (`task_entry_head[task_slot]`) — each task's entries are doubly-linked together at insert time via `next_in_task`/`prev_in_task`. A slot's chain can hold more than one task's entries: a task at `local_id + N * window` reuses the slot and prepends to the chain already there, and cleanup can lag that reuse. Cleanup therefore walks the chain and frees only the entries whose `producer_task_id` matches the retiring task, unlinking each and leaving the rest linked — O(entries_in_slot), with no scan of the entire pool or all buckets. Freed entries are returned to `free_entry_list` for immediate reuse.

**Layer 3 — Back-Pressure on Pool Exhaustion** (blocking):

Before STEP 4 inserts a task's outputs, `ensure_tensormap_capacity` checks the free list + bump region against the task's needed entry count. If short, it reclaims retired entries across all rings and blocks until reclaim frees enough entries. Progress is measured by entries actually freed, not by watermark movement — a ring can retire zero-output tasks, advancing `last_task_alive` without freeing any entry. A pool that frees nothing for a 500 ms wall-clock timeout is a genuine deadlock: it latches `SIMPLER_ERROR_TENSORMAP_OVERFLOW` and unwinds, matching the task allocator and fanin spill pool.

This forms a back-pressure mechanism analogous to the Task Ring's flow control.

**Summary**:

| Layer | Trigger | Method | Guarantees |
| ----- | ------- | ------ | ---------- |
| Chain Truncation | Every lookup | Truncate stale tail of bucket chain | Lookup only visits valid entries |
| Periodic Cleanup | Every 64 retired tasks | Walk per-task chains, free entries | Pool capacity reclaimed in bounded time |
| Pool Back-Pressure | Pool exhausted | Block until scheduler advances watermark | Hard capacity bound, no OOM |

In steady state, the number of valid TensorMap entries ≈ `active_tasks × avg_outputs_per_task`. With the default `task_window=65536` and `pool_size=65536`, this is well within bounds. With small windows (e.g., `task_window=16`), active entries are even fewer (~16 × a few), and cleanup runs frequently.

### 5.5 Dependency Discovery Flow

When `OrchestratorState::submit_task` processes parameters:

1. **INPUT/INOUT**: `ChipTensorMap::lookup` searches for overlapping producers (with chain truncation)
2. For each producer found: `append_fanin_or_fail` adds the dependency
3. **OUTPUT/INOUT**: `ChipTensorMap::insert` registers the current task as the new producer at bucket head
4. Stale entries are pruned lazily during lookup (Layer 1) and periodically by cleanup (Layer 2)

---

## 6. Task Descriptor and States

### 6.1 TaskDescriptor (Hot Path)

| Field | Description |
| ----- | ----------- |
| `task_id` | Canonical mixed-task ID (64-bit: `ring_id << 32 \| local_id`). See [MULTI_RING.md §3](MULTI_RING.md). |
| `kernel_id[3]` | Per-slot kernel IDs: `[AIC, AIV0, AIV1]`; `INVALID_KERNEL_ID` = inactive |
| `active_mask` | Bitmask of active subtask slots: `bit0=AIC`, `bit1=AIV0`, `bit2=AIV1` |
| `completed_subtasks` | Atomic counter; each subtask increments on completion. Trigger condition: `completed_subtasks == total_required_subtasks` |
| `fanin_count` | Number of producer dependencies (set by Orch-side wiring) |
| `fanout_lock` | Per-task spinlock for concurrent fanout modification (used by Orch-side wiring + scheduler completion) |
| `fanout_head` | Head of fanout consumer list (pointer, protected by `fanout_lock`) |
| `fanout_count` | 1 (scope ref) + number of consumers |
| `packed_buffer_base` | Start of packed buffer in GM Heap |
| `packed_buffer_end` | End of packed buffer (for heap reclamation) |

### 6.1b TaskPayload (Cold Path)

| Field | Description |
| ----- | ----------- |
| `tensors[16]` | Tensor descriptors for parameters |
| `scalar_value[16]` | Scalar parameter values |
| `is_tensor[16]` | Whether each parameter is tensor or scalar |
| `param_count` | Number of valid parameters |
| `fanin_slot_states[]` | Producer slot state pointers (used by `on_task_release`) |
| `fanin_actual_count` | Total fanin edge count (all flag combinations, including RETAIN-only and reduction-dropped entries) |
| `fanin_wait_count` | DEP_WAIT edge count after reduction — the readiness denominator |
| `early_dispatch_blocked` | set when reduction dropped an edge to a producer that does not allow early resolve; `early_dispatch_target()` = `fanin_wait_count` + this, so that producer keeps this task ineligible for early dispatch exactly as it did before reduction |

### 6.2 Task State Machine

```text
  [0] PENDING ──worker(s) done──► [1] COMPLETED ──fanout done──► [2] CONSUMED
      ▲                                                                │
      │                                                                ▼
      └──────────────────── slot recycled ◄───────────────────────────┘
```

In the scheduler's `task_state[]` array (`std::atomic<ChipTaskState>`):

- **0 (PENDING)**: slot is allocated and remains PENDING through "waiting on
  producers", "queued in ready queue", and "dispatched to a worker"; ready vs
  running is derived from `fanin_refcount` and per-core `running_slot_state`
- **1 (COMPLETED)**: hardware execution complete, output may still be in use
- **2 (CONSUMED)**: output fully consumed, buffers can be released

---

## 7. Orchestrator

### 7.1 OrchestratorState

The orchestrator runs on AICPU Thread 3 and builds the task graph by calling the user-provided orchestration function.

Key members:

- `rings[CHIP_MAX_RING_DEPTH]`: per-ring `ChipRingSet` (`TaskAllocator` + `FaninPool`). See [MULTI_RING.md §4.2](MULTI_RING.md).
- `tensor_map`, `tensor_pool`: dependency tracking
- `scope_tasks[]`, `scope_begins[]`, `scope_stack_top`: scope nesting stack (flat buffer partitioned by level)
- `scheduler`: pointer to scheduler state (for Orch-side wiring helpers and ready queue access)
- `gm_heap_base`, `gm_heap_size`: GM heap for output buffers
- `wait_reach[CHIP_MAX_RING_DEPTH]`, `submit_seq`: per-slot frozen WAIT-ancestor bitmap plus its global submission sequence (`WaitReachEntry`, 16 B per slot), backing the step-5.5 bounded reduction. Orchestrator-private runtime-arena storage, never read by scheduler threads. Costs `16 B x window_size` per ring — **1 MiB** at the default 4 rings x 16384 slots, and linear in `runtime_env.ring_task_window` if a ring is enlarged.

### 7.2 Task Submission Flow (`OrchestratorState::submit_task`)

| Step | Operation |
| ---- | --------- |
| 0 | `ChipTensorMap::sync_tensormap` — prune stale TensorMap entries |
| 1 | `TaskAllocator::alloc` — allocate task slot (may block on flow control) |
| 2 | Initialize task descriptor + slot state, copy parameters |
| 3 | **Lookup**: for each INPUT/INOUT param, search TensorMap for producers; collect producer pointers in `FaninBuilder` |
| 4 | **Insert**: register OUTPUT/INOUT args in TensorMap |
| 5 | **Record fanin metadata**: store producer edges (slot pointer + `DepFlags` packed in the low bits) in `payload->fanin_inline_edges[]` (+ spill pool if >64); claim each live producer by incrementing `fanout_count` under that producer's `fanout_lock`. Creator edges are `DEP_WAIT\|DEP_RETAIN`, tensormap-modifier edges `DEP_WAIT`. This step runs **before** `payload.init()`. |
| 5.5 | **Bounded transitive reduction** (`reduce_wait_edges`, issue #1376): publish this task's frozen 64-bit WAIT-ancestor bitmap over the last `WAIT_REACH_WINDOW` global submissions, then clear `DEP_WAIT` on any direct edge already covered by a WAIT path through another producer's transitive ancestors (`WAIT\|RETAIN` demotes to RETAIN-only; `WAIT`-only drops to `DEP_NONE`, the entry stays for pin accounting). Candidates beyond the window keep their WAIT. Runs before the payload flush, so `fanin_wait_count` reflects the reduced readiness set. |
| 6 | **Orch-side wiring / ready publish**: the orchestrator wires live fanout edges into the per-ring dep_pool; zero-fanin and already-completed fanin tasks publish directly to ready queues. Only `DEP_WAIT` edges gate readiness — they count toward `fanin_count` and are linked onto the producer's `fanout_head` for completion notification. A `DEP_WAIT`-only edge releases its submit→wire retention pin **at wiring** (and on the already-completed fast path), so its producer can be CONSUMED without waiting for this consumer; a `DEP_RETAIN` edge keeps the pin until this consumer's `on_task_release`. A `RETAIN`-only edge — produced by step 5.5 when it demotes a redundant `WAIT\|RETAIN` edge — neither gates readiness nor links a fanout node; it only holds the lifetime pin until this consumer's `on_task_release`. |

> **Note**: Fanout wiring is now completed before publish in the orchestrator submit path.
> Scheduler threads consume ready queues directly.

### 7.3 Orch-Side Fanout Wiring

The orchestrator completes fanout wiring before publishing a task to the ready queues. For each task with live producers:

1. Sets `fanin_count = N + 1` (+1 redundance to prevent premature readiness)
2. For each producer in `payload->fanin_slot_states[]`:
   - **Acquires** the producer's `fanout_lock`
   - Checks `task_state >= COMPLETED` (early-finished optimization)
   - If not completed: prepends consumer to producer's `fanout_head` via `dep_pool.prepend`
   - **Releases** `fanout_lock`
3. Atomically releases the +1 redundance + completed-fanin count via `fanin_refcount.fetch_add`
4. If all deps satisfied: pushes task to the routed ready queue

Zero-fanin tasks and tasks whose claimed producers are already completed skip dep_pool entry allocation and publish directly to the routed ready queue.

The scheduler's completion handler mirrors this:

1. **Acquire** `fanout_lock`, mark `task_state = COMPLETED`, read `fanout_head`, **release** lock
2. Traverse fanout list, incrementing each consumer's `fanin_refcount`
3. Mark `task_state = CONSUMED` when `fanout_refcount` reaches `fanout_count`

This protocol guarantees every consumer is accounted for exactly once.

### 7.4 Scope Mechanism (`SIMPLER_SCOPE`)

Scopes control the lifetime of intermediate buffers. Each scope:

- Tracks tasks submitted within it via a flat `scope_tasks[]` buffer partitioned by `scope_begins[]`
- On `scope_end`: increments `fanout_refcount` for scope tasks; when it reaches `fanout_count`, the task's packed buffer can be reclaimed

```cpp
SIMPLER_SCOPE(rt) {
    // Tasks submitted here belong to this scope
    rt_submit_aic_task(FUNC_QK, args);
    rt_submit_aiv_task(FUNC_SF, args);
}
// scope_end: scope reference released from all tasks above
```

**Output tensor lifetime — single-scope only.** `submit_task` returns a
`TaskOutputTensors`, and `get_ref(i)` hands back a `const Tensor&`. Both are
backed by pointers into the submitting task's `TaskPayload::tensors[]`,
which lives in a ring-buffer slot. After `scope_end` the slot becomes
eligible for reuse; once `advance_ring_pointers` reaches it,
`reset_for_reuse()` runs and the next `submit_task` overwrites the same
Tensor storage in place.

Therefore the `TaskOutputTensors` instance, the references it returns, and
any pointer derived from them MUST NOT outlive the `SIMPLER_SCOPE` in which
submit was called. The typical safe pattern is:

```cpp
SIMPLER_SCOPE() {
    TaskOutputTensors outs = rt_submit_aic_task(FUNC_QK, args);
    const Tensor &y = outs.get_ref(0);
    // Use y here and in subsequent submits within the same scope.
}   // outs and y both go out of scope; no dangling references can escape.
```

Anti-patterns that compile but silently break:

```cpp
const Tensor *kept = nullptr;
SIMPLER_SCOPE() {
    TaskOutputTensors outs = rt_submit_aic_task(FUNC_QK, args);
    kept = &outs.get_ref(0);          // escapes the scope
}
// `kept` still points at a payload slot. After enough submits in later
// scopes, the slot is reused and `*kept` aliases an unrelated task's
// tensor — a wrong-tensor read with no runtime diagnostic.

TaskOutputTensors outs;               // declared in outer scope
SIMPLER_SCOPE() {
    outs = rt_submit_aic_task(FUNC_QK, args);
}
const Tensor &t = outs.get_ref(0);    // same hazard: outs survives scope
```

This invariant is intentionally not runtime-checked. A reused slot carries
a different but valid `owner_task_id`, so an assertion based on
`owner_task_id` cannot distinguish "still the original task" from
"silently aliased to a newer task". Treat the rule as a static contract,
verified by review.

---

## 8. Scheduler

### 8.1 Thread Model

With the default `aicpu_thread_num=5`, the AICPU runs 5 threads. The core counts below
describe the 36-cluster target in §2.3; the runtime partitions the detected
cores across the scheduler threads, so smaller SKUs use proportionally smaller
slices.

| Thread | Role | Cores |
| ------ | ---- | ----- |
| 0 | Scheduler | 9 AIC + 18 AIV |
| 1 | Scheduler | 9 AIC + 18 AIV |
| 2 | Scheduler | 9 AIC + 18 AIV |
| 3 | Scheduler | 9 AIC + 18 AIV |
| 4 | Orchestrator | none |

Core assignment: AICs and AIVs are divided equally among the 4 scheduler threads.

### 8.2 Scheduler Main Loop

Each scheduler thread runs a tight loop with two main phases:

**Phase 1 — Completion Handling**:

- Poll register `COND` on each managed core
- When `TASK_FIN_STATE` detected: record completion timestamps, call `on_subtask_complete(task_id, subslot)` to increment the completion counter; when `completed_subtasks == total_required_subtasks`, trigger `on_task_complete(task_id)` which marks `task_state[slot] = COMPLETED`, acquires fanout lock, traverses fanout list (incrementing consumers' `fanin_refcount`), marks `task_state[slot] = CONSUMED`, and advances `last_task_alive` watermark

**Phase 2 — Dispatch** (full model in §8.6):

- Service each source (normal ready ▸ speculative early) in occupancy order — `sync_start`
  Tier-0 ▸ MIX ▸ AIC/AIV, idle ▸ pending — popping from the matching shape-based ready queue
  (lock-free MPMC Vyukov queue, one per resource shape)
- Build `DispatchPayload` from `TaskDescriptor` with `task_id`, `subslot`, `kernel_id`, and `core_type`
- Write task pointer to `Handshake.task`, signal AICore via register `DATA_MAIN_BASE` (DMB offset `0xD0` on a5)
- After normal ready queues are empty, Phase **4b** may stage speculative early-dispatch candidates onto spare slots (`early_dispatch_queues[]` / `early_sync_start_queue`)

After these phases, the scheduler updates profiling headers and checks for termination (all tasks completed and orchestrator done).

### 8.3 Ready Queue Design

Ready queues use a lock-free bounded MPMC (Vyukov) design:

- One `ChipReadyQueue` per resource shape — 3 shapes (`NUM_RESOURCE_SHAPES`): `MIX`
  (AIC+AIV cluster), `AIC`, `AIV`. Alongside `ready_queues[]` there is a per-shape
  `ready_sync_queues[]` (sync_start Tier-0) and the speculative `early_dispatch_queues[]` /
  `early_sync_start_queue` — see §8.6 for the full source × tier model.
- **Push**: any thread (orchestrator via `init_task`, or scheduler on completion) pushes newly-ready tasks to the queue matching `task->active_mask.to_shape()` (sync_start cohorts to the sync lane)
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
    if force_publish or la - last_published_to_sm >= 16 or la == current_task_index:
        sync_to_sm()  // release-store last_task_alive
```

This is protected by a per-ring try-lock (`advance_lock`) in `RingSchedState`, ensuring only one scheduler thread advances a given ring's watermark at a time. A scheduler thread that changes a ring head to `CONSUMED` but loses this try-lock sets the ring bit in `advance_pending_mask`; no-progress loops drain that mask. An orchestrator reclaim consumer with no reclaim progress for 10 ms instead sets `publication_request_mask`. Scheduler thread 0 drains only that request mask in both productive and no-progress iterations, force-publishes the exact watermark, and sets the matching bit in `publication_ack_mask`. The two masks occupy separate cache lines, so scheduler lock contention neither triggers acknowledgments nor invalidates the productive-loop request poll. Runtime wiring, relocation, and reuse enable K=16 batching only after the allocator, fanin pool, and dependency pool all point to the current masks; otherwise `sync_to_sm()` publishes each advance.

For ring-heap stall triage, a set request bit means no retry has yet acquired
`advance_lock`. A matching acknowledgment means the published
`last_task_alive` was synchronized with scheduler-local reclamation; only that
acknowledged value is eligible for the exact open-scope head check.

### 8.5 SchedulerContext

All scheduler-side state and methods live in `SchedulerContext` (`runtime/scheduler/scheduler_context.h`). It is held as a `sched_ctx_` member of `AicpuExecutor`; `AicpuExecutor` is a thin wrapper that owns the lifecycle atomics and the orchestration SO handle, and delegates everything else to `SchedulerContext`.

Public surface (called from `AicpuExecutor::init/run/deinit`):

| Method | Phase | Purpose |
| ------ | ----- | ------- |
| `pre_handshake_init(runtime, aicpu_thread_num, sched_thread_num, regs_base)` | leader, once per run | Zero state, reset counters, latch `regs_base`, bind `func_id_to_addr_` before any thread handshakes |
| `handshake_partition(runtime, tidx, nthreads)` | every thread | Handshake this thread's disjoint core slice in parallel |
| `post_handshake_init(runtime)` | leader, after barrier | Build worker-id lists in core order, assign cores to threads |
| `bind_runtime(rt)` | device-orch only | Wire `sched_` to `rt->scheduler` once the orchestrator thread creates `rt` |
| `resolve_and_dispatch(runtime, thread_idx)` | per scheduler thread | Main dispatch loop |
| `shutdown(thread_idx)` | per thread on exit | `platform_deinit_aicore_regs` for this thread's cores |
| `on_orchestration_done(runtime, rt, thread_idx, total_tasks)` | orchestrator thread | Publish core assignments, latch task count, fold inline-completed tasks, flip `orchestrator_done_` (or `emergency_shutdown` on fatal) |
| `deinit()` | once per run | Reset every scheduler-owned field to its post-construction default |
| Read-only accessors | various | `aic_count()` / `aiv_count()` / `is_completed()` / `completed_tasks_count()` |

Private internals are split across three .cpp files by responsibility:

- `scheduler_completion.cpp` — completion polling, drain protocol
- `scheduler_dispatch.cpp` — task dispatch loop and helpers
- `scheduler_cold_path.cpp` — exit checks, stall diagnostics, profiling, lifecycle (`pre_handshake_init` / `handshake_partition` / `post_handshake_init` / `deinit`), core management (`assign_cores_to_threads` / `emergency_shutdown`), and `on_orchestration_done`

`AicpuExecutor` calls neither `handshake_*`, `assign_*`, `reassign_*`, nor `emergency_shutdown` directly — they are private, invoked only by `init` and `on_orchestration_done`.

### 8.6 Dispatch model — two sources, sync tiers, occupancy order

`resolve_and_dispatch` places ready and speculative work onto AICore cores under one
occupancy model. Two orthogonal axes decide *what* runs and *where*:

- **Source** — `NORMAL` (all producers done; the task sits in a ready queue and launches on
  pickup) vs `EARLY` (a *speculative* pre-stage of a not-yet-released task; its dispatch
  payload carries a non-zero `src_payload` gate and launches later by a doorbell). Normal
  strictly precedes early.
- **Cohort** — `SYNC_START` (an SPMD cohort that must launch atomically) vs `REGULAR` (each
  block launches independently). "is it ready" (source) and "does it need a rendezvous"
  (cohort) are orthogonal.

Within each source the occupancy order is **`sync_start` ▸ MIX ▸ AIC/AIV** (shape), and per
shape **idle ▸ pending** (an idle core takes its running slot; a busy core takes its gated
pending slot, promoted on completion). This order lives in one shared skeleton,
`run_staging_order`; the normal and early sources differ only in the per-shape stage callback
(pickup vs gated).

#### Queues

| Source | Regular lanes | sync_start lane |
| ------ | ------------- | --------------- |
| NORMAL (ready) | `ready_queues[MIX\|AIC\|AIV]` | `ready_sync_queues[MIX\|AIC\|AIV]` (per-shape) |
| EARLY (speculative) | `early_dispatch_queues[MIX\|AIC\|AIV]` | `early_sync_start_queue` (single) |

A task routes to the early sync lane iff `task_attrs.requires_sync_start()`. Early
dispatch runs only once both normal ready lanes are empty **and** the local
`CoreTracker` has a spare slot (`has_any_free_slot`, a2a3 #1288).

**Direct-only eligibility (a2a3 #1285/#1292):** a consumer is an early candidate only when
every *direct* producer is flagged `allow_early_resolve` (slot-state hint from Arg, or
unconditional true for hidden alloc creators). There is no auto-chain inheritance.

#### sync_start local stage, drain fallback, and rendezvous

A sync_start cohort of `block_num` blocks must occupy all its cores before any of them run.
An early Tier-0 pop first claims the cohort's task-local staging owner, then chooses one of
two all-or-nothing cases:

1. **Case A — single-owner local stage** — when the popping scheduler's own
   `CoreTracker` has at least `block_num` compatible idle + pending slots and no global
   drain is already published, it force-gates and stages the complete cohort locally.
   AIC/AIV capacity is the number of idle plus pending-capable cores; MIX capacity is the
   number of active-mask-compatible split-placement clusters. This path never writes
   `sync_start_pending`, `drain_attempt`, or the drain ack tokens.
2. **Case B — global drain fallback** — otherwise `enter_drain_mode` CAS-claims
   `sync_start_pending`. Scheduler thread 0 coordinates the generation-tagged ack tree,
   checks global capacity, and retries after completions when capacity is short. Once
   released, every scheduler thread CAS-claims block ranges and stages only its own cores.
3. **Shared staging** — `stage_sync_start_cores` publishes every claimed block with a
   non-zero `src_payload` gate: idle cores → running slots, busy cores → pending slots
   (`pending_gated` while waiting for the doorbell). It returns both logical blocks staged
   and running-slot cores, because one MIX block may use multiple cores.
4. **Rendezvous launch** — after the complete `staged_core_mask` is published,
   `running_slot_count` is seeded with the staged running-slot cores. The rendezvous reads
   that seed before the full mask; when it equals `popcount(staged_core_mask)` **and** the
   producer has released, `maybe_rendezvous_ring` rings every gated core's doorbell
   together — the cohort starts as one.

If a global drain is published concurrently after Case A's zero check, its ack barrier
waits until the local scheduler returns to the loop; the coordinator then observes the
locally occupied tracker slots before making its global capacity decision.

At scheduler-phase profiling level, local speculative staging is emitted as
`EarlyDispatch` rather than being folded into the normal `Dispatch` bar. This lets the
single-owner regression prove that the cohort was staged by Case A.

Doorbell ownership is exclusive (`claim_all_staged_doorbell_bits` /
`claim_late_staged_doorbell_bits`); launch is latched via `early_dispatch_launch_state`.

#### Early-candidate gate: producer must publish every block

Producer-side `propagate_dispatch_fanin` no-ops until the producer is **fully published**:
`published_block_count == logical_block_num` (a2a3 #1326). Publication is recorded after
prepare/publish makes payload + low-32 dispatch tokens visible. Early queue entries carry a
`task_id_snapshot` generation tag (#1336) so recycled slots cannot revive stale candidates.

Wiring and propagation serialize under `fanout_lock` with `DISPATCH_PROPAGATED` so
late-wired consumers still receive exactly one early-candidate contribution per eligible
edge (#1405).

#### a5 platform notes

- `RUNTIME_MAX_WORKER = 108` — doorbell table / `staged_core_mask` words must cover 108 cores.
- AICore index fields are `s_block_idx` / `s_block_num` (not a2a3 `block_idx` / `block_num`).
- DMB MMIO offset is **`0xD0`** (a2a3 uses `0xA0`). Ready dispatch writes the low 32 bits;
  early release rings high 32 via 64-bit STR `(token<<32)|token`. AICore gated path spins on
  `read_dmb_high32() == task_id` before ACK.
- Ready path keeps `src_payload == 0` and ACK-then-execute behavior unchanged.

#### MIX per-core placement

A MIX task spans a cluster (1 AIC + 2 AIV). `classify_mix_cluster` admits a cluster whenever
every used core has a free slot; `prepare_block_for_dispatch` then places **per core**
(`to_pending && !is_core_idle`): idle cores → running, busy cores → pending. Cross-core start
skew within a block is tolerated by AICore incore synchronization.

---

## 9. AICore Worker Interaction

### 9.1 Handshake Protocol

Each AICore worker has a `Handshake` struct in shared memory:

| Field | Direction | Purpose |
| ----- | --------- | ------- |
| `task` | AICPU→AICore | Pointer to `DispatchPayload` |
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

1. Poll `DATA_MAIN_BASE` (low 32) for value != AICPU_IDLE_TASK_ID
2. Read payload from `Handshake.task`
3. If `src_payload != 0` (early/gated): materialize args from `src_payload`, spin until
   `read_dmb_high32() == task_id`, **then** ACK
4. Else (ready path): ACK immediately
5. Execute kernel function via `func_id_to_addr` lookup
6. Write FIN to `COND`

### 9.3 DispatchPayload

Built by the scheduler from `TaskDescriptor`:

| Field | Description |
| ----- | ----------- |
| `task_id` | Mixed-task identifier (for completion aggregation) |
| `subslot` | Which subtask slot this dispatch represents (`AIC`, `AIV0`, or `AIV1`) |
| `kernel_id` | Function ID for this subtask slot |
| `core_type` | AIC or AIV |
| `function_bin_addr` | GM address of compiled kernel binary |
| `src_payload` | Non-zero ⇒ gated early path (AICore materializes args + waits high-32 doorbell) |
| `num_args` | Number of arguments |
| `args[]` | Tensor addresses and scalar values (filled by AICPU on ready path; by AICore when gated) |

---

## 10. Kernel and Orchestration Loading

### 10.1 Kernel Binary Loading

1. **Host** compiles each kernel source (`.cpp`) into a binary (`.o` or `.so`)
   and packs all children into a single `ChipCallable` buffer alongside the
   orchestration SO.
2. `host_api.upload_chip_callable_buffer(callable)` H2Ds the whole buffer
   once and returns the device address of the ChipCallable header.
3. For each child, host computes
   `chip_dev + offsetof(ChipCallable, storage_) + callable->child_offset(i)`
   and stores it in `Runtime.dev.func_id_to_addr_[child_func_id(i)]`.
4. When dispatching, the scheduler reads `func_id_to_addr_[fid]`, casts to
   `const CoreCallable*`, reads `resolved_addr_`, and copies that into
   `DispatchPayload.function_bin_addr`.

### 10.2 Orchestration SO Loading

1. **Host** compiles the orchestration source into a shared library (`.so`)
2. The SO binary is embedded into `Runtime.device_orch_so_storage_[]` and copied to device
3. The orchestrator thread writes the SO to a temp file, calls `dlopen`
4. `dlsym("aicpu_orchestration_config")` returns configuration (expected arg count)
5. `dlsym("aicpu_orchestration_entry")` returns the orchestration function pointer
6. The orchestrator thread creates a `RuntimeContext`, calls the orchestration function within a `SIMPLER_SCOPE`
7. After orchestration completes: `dlclose`, delete temp file

### 10.3 Thread Startup Synchronization

| Flag | Set by | Waited by | Purpose |
| ---- | ------ | --------- | ------- |
| `runtime_init_ready_` | Orchestrator thread | Scheduler threads | Runtime and SM handle initialized |
| `orchestrator_done_` | Orchestrator thread | Scheduler threads when `SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE=1` | Full task graph built |

Profiling-subsystem init (`dump_args` / `pmu` / `dep_gen` / `chip_swimlane`) runs
once in `SchedulerContext::init()` on the single-threaded cold path, before any
scheduler/orchestrator thread starts — so it needs no cross-thread init
handshake.

Startup sequence:

1. Orchestrator thread: create SM handle + runtime → set `runtime_init_ready_`
2. Scheduler threads: wait for `runtime_init_ready_` → enter main loop
3. Orchestrator thread: configure orchestrator-scheduler pointers → call orchestration function → set `orchestrator_done_`

With `SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE=1`, scheduler threads still wait for
`runtime_init_ready_` first, then additionally wait for `orchestrator_done_`
before entering `resolve_and_dispatch()`. The default is off, preserving the
current overlapped orch/sched pipeline. Serial mode is intended for measurement
and debugging. It does not dispatch AICore work until `orchestrator_done_` is
set. Large graphs may still require larger task-ring, heap, or dependency-pool
capacity because no task execution/reclaim happens during graph build.

---

## 11. Orchestration API

The orchestration API is defined in `orchestration_api.h`. Orchestration code depends only on this header.

### 11.1 Core API

| Function/Macro | Purpose |
| -------------- | ------- |
| `rt_submit_task(mixed_kernels, args)` | Submit a mixed task with `MixedKernels` struct |
| `rt_submit_aic_task(kernel_id, args)` | Convenience: submit AIC-only task |
| `rt_submit_aiv_task(kernel_id, args)` | Convenience: submit AIV-only task |
| `SIMPLER_SCOPE() { ... }` | RAII scope for buffer lifetime |
| `rt_orchestration_done()` | Signal orchestration complete |

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
extern "C" OrchestrationConfig aicpu_orchestration_config(uint64_t* args, int arg_count);
extern "C" void aicpu_orchestration_entry(uint64_t* args, int arg_count);
```

---

## 12. Example: Batch Paged Attention

### 12.1 Scene-test Configuration

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

@scene_test(level=2, runtime="tensormap_and_ringbuffer")
class TestBatchPagedAttention(SceneTestCase):
    CALLABLE = {"orchestration": ORCHESTRATION, "incores": KERNELS}
    CASES = [
        {"name": "default", "platforms": ["a5sim", "a5"], "params": {}},
    ]
```

The omitted config selects the A5 default of five AICPU threads. Every run uses
the detected device width, up to 36 clusters; there is no per-run `block_dim`
knob.

### 12.2 Orchestration Structure

```cpp
void aicpu_orchestration_entry(uint64_t* args, int arg_count) {
    // Unpack args: query, key_cache, value_cache, block_table, context_lens, out, config
    for (q_idx = 0; q_idx < q_loop; q_idx++) {
        for (batch_start = 0; batch_start < batch; batch_start += IN_CORE_BATCH) {
            SIMPLER_SCOPE() {
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
