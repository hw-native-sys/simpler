# PTO2 Runtime System Design (host_build_graph)

## Overview

host_build_graph is the **host-orchestration** variant of the PTO2 runtime: it
shares tensormap_and_ringbuffer's scheduler, ring buffers, and shared-memory
layout, and differs only in **when** the orchestrator runs. The host runs the
orchestrator to completion — building the whole task graph, populating shared
memory and the prebuilt arena — then H2Ds that image to the device, where the
AICPU boots scheduler-only (no on-device orchestrator thread). It coordinates
four layers of execution:

- **Host** (x86/ARM CPU): compiles kernels, allocates device memory, **runs the orchestrator to build the task graph**, H2Ds the populated SM + arena, and launches AICPU/AICore threads.
- **AICPU** (device ARM cores): runs scheduler threads only — it attaches the host-populated shared memory (already device-addressed by the host) and dispatches the already-built graph.
- **AICore** (AI compute cores): executes kernel functions dispatched by the scheduler.
- **Shared Memory** (Global Memory): ring buffers, task descriptors, heap, and TensorMap — built on host, attached read-mostly by the schedulers.

```text
┌───────────────────────────────────────────────────────────────────────┐
│                            Host (CPU)                                 │
│  test_*.py (SceneTestCase) → compile kernels → init Runtime           │
│  → run orchestrator (build graph) → H2D populated SM + arena          │
│  → upload binaries → launch AICPU/AICore → collect results            │
└───────────────────────────┬───────────────────────────────────────────┘
                            │ device memory / GM (populated graph image)
┌───────────────────────────▼───────────────────────────────────────────┐
│                     AICPU (N threads)                                  │
│  All threads: Schedulers (attach + dispatch to AICore)                │
│  (no on-device orchestrator thread — host already built the graph)    │
│                                                                       │
│  ┌─────────────────────────────────────────────────────────────────┐  │
│  │                   Shared Memory (GM)                             │  │
│  │  SharedMemoryHeader │ TaskDescriptors[] │ Payloads[] │ SlotStates[] │
│  │  GM Heap (output buffers)                                       │  │
│  └─────────────────────────────────────────────────────────────────┘  │
│                                                                       │
│  Scheduler ──Handshake/Registers──► AICore workers (AIC + AIV)        │
└───────────────────────────────────────────────────────────────────────┘
```

> **Where the orchestrator runs (host_build_graph vs tensormap).** The data
> structures and scheduler mechanics described in the sections below — the
> orchestrator state, ring buffers, TensorMap dependency tracking, and the
> dispatch handshake — are **shared** with `tensormap_and_ringbuffer`. Two
> things differ, both following from **when and where the orchestrator runs**:
>
> - **host_build_graph (this runtime):** the **host** dlopens the orchestration
>   `.so` and runs it to completion, populating shared memory + the prebuilt
>   arena. Dependencies are recorded as **position-independent integer fanin**:
>   each task's payload stores its producers' local-ids (`fanin_local_ids`) and
>   each producer records its highest consumer id (`last_consumer_local_id`).
>   There is no fanout adjacency, dep-pool, or per-task lock. Because fanin is
>   integer ids, the host→device relocation (`relocate_host_orch_image`)
>   collapses to fixing up only the per-slot `task` / `payload` pointers before
>   the H2D copy. The device boots **scheduler-only**: no on-device orchestrator
>   thread and no fanout wiring; on boot it scans the submitted tasks and
>   classifies each (fanin-free → ready queue; otherwise register on its first
>   unmet producer's wake list — see §8), then dispatches.
> - **tensormap_and_ringbuffer:** the orchestrator runs **on-device** on AICPU
>   thread N-1, concurrently with the scheduler threads, recording the same
>   integer fanin during submit.
>
> Where a section below says "the orchestrator runs on AICPU Thread 3" or
> "Thread 3 dlopens the SO", read that as the **tensormap (device-orch)
> mechanics this runtime inherits** — under host_build_graph that same work
> happens on the **host** before launch.

---

## 1. Runtime Variants

Two runtime backends exist under `src/runtime/`, each representing a different orchestration and scheduling strategy.

### 1.1 host_build_graph

The host-orchestration variant of `tensormap_and_ringbuffer`: it shares the same
ring-buffer task storage, GM heap, and TensorMap dependency tracking, and runs
the orchestration SO **on the host CPU** to build the complete task graph before
launching device execution. The device then boots scheduler-only.

- **Task storage**: `PTO2TaskDescriptor[]` in shared memory ring buffer (same as 1.2)
- **Dependencies**: automatically derived from tensor read/write patterns via TensorMap (same as 1.2)
- **Scheduling**: AICPU attaches the host-populated, already-device-addressed SM and dispatches the pre-built graph
- **Use case**: host-side graph construction; device runs no orchestrator thread

### 1.2 tensormap_and_ringbuffer (PTO2)

The primary production runtime. Uses ring buffers for task slots and output memory, with a TensorMap for automatic dependency tracking.

- **Task storage**: `PTO2TaskDescriptor[]` in shared memory ring buffer
- **Memory**: GM Heap ring for output buffer allocation
- **Dependencies**: automatically derived from tensor read/write patterns via TensorMap
- **Thread model**: 3 scheduler threads + 1 orchestrator thread on AICPU
- **Single ring**: host_build_graph builds the whole graph on the host with no
  execution-time reclaim, so HeapRing and TaskRing are single
  whole-graph-resident instances (`PTO2_MAX_RING_DEPTH == 1`); all scope depths
  map to ring 0.
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
| `upload_chip_callable_buffer` | Copy ChipCallable bytes to a host scratch, `dlopen` each child SO, `dlsym` "kernel_entry", patch the scratch's `resolved_addr_` with the function pointer |

### 2.3 Platform Constants (`platform_config.h`)

| Constant | Value | Description |
| -------- | ----- | ----------- |
| `PLATFORM_MAX_BLOCKDIM` | 24 | Maximum blocks (each = 1 AIC + 2 AIV) |
| `PLATFORM_MAX_AICPU_THREADS` | 4 | AICPU thread count (all schedulers; the highest-index thread also runs the host-orch boot) |
| `PLATFORM_MAX_AIC_PER_THREAD` | 24 | Max AIC cores per scheduler thread |
| `PLATFORM_MAX_AIV_PER_THREAD` | 48 | Max AIV cores per scheduler thread |
| `PLATFORM_PROF_SYS_CNT_FREQ` | 50 MHz | System counter frequency for profiling |

---

## 3. Shared Memory Layout

The orchestrator and schedulers communicate through a contiguous shared memory region in Global Memory (GM). The single ring's TaskDescriptor, TaskPayload, and TaskSlotState sections (plus the per-slot `completion_flags`) are laid out by `pto2_sm_layout::ring_segment_offsets`.

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

> **Single ring**: TaskRing and HeapRing are single whole-graph instances
> (`PTO2_MAX_RING_DEPTH == 1`). The host builds the entire graph before the
> device boots and there is no execution-time reclaim, so a per-scope-depth
> ring split would buy nothing — every scope depth maps to ring 0.

### 4.1 Task Ring

The task ring manages task slot allocation with back-pressure flow control.

**Structure** (`PTO2TaskRing`):

- `descriptors`: pointer to `TaskDescriptor[]` in shared memory
- `window_size`: number of slots (power of 2)
- `current_index`: next task ID to allocate (monotonically increasing)
- `last_alive_ptr`: pointer to `header->last_task_alive`

**Slot mapping**: `slot = task_id & (window_size - 1)`

**Allocation** (`PTO2TaskAllocator::alloc`):

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

### 4.3 Dependency Representation (polling completion)

There is no dependency-list pool or fanout adjacency. A task's dependencies are
stored inline on its payload as a flat array of position-independent producer
local-ids:

- `fanin_local_ids[fanin_count]` — the local-ids of this task's direct producers
  (`fanin_count <= PTO2_MAX_FANIN`; there is no spill, so an overflow is fatal).
- A per-slot `completion_flags[id]` byte in the ring header is the device-side
  readiness truth: a task is ready iff every id in its `fanin_local_ids` has its
  `completion_flags` byte set (see §8.2).

Because the fanin is integer ids rather than pointers, the host→device image
needs no fanout/dep-pool relocation — only the per-slot `task` / `payload`
pointers are relocated (see §7.3).

### 4.4 Flow Control and Back-Pressure

The ring buffer mechanism provides **flow control** between the orchestrator (producer) and the scheduler (consumer). When a ring is exhausted, the orchestrator **blocks** — it cannot submit new tasks or allocate more output memory until the scheduler reclaims slots/space by advancing the watermarks.

**Task Ring back-pressure**: When `active_count = current_index - last_task_alive >= window_size - 1`, `PTO2TaskAllocator::alloc` spin-waits until the scheduler completes tasks and advances `last_task_alive`.

**Heap Ring back-pressure**: When the heap has insufficient contiguous space, `PTO2TaskAllocator::alloc` spin-waits until the scheduler advances `heap_tail` past completed tasks' output buffers.

**TensorMap pool back-pressure**: Before STEP 4 registers a task's outputs, the orchestrator's `ensure_tensormap_capacity` reserves pool space for the inserts. When the shared entry pool is exhausted, it reclaims retired entries across all rings and spin-waits until reclaim actually frees entries, with a 500 ms wall-clock deadlock backstop (see Section 5.4).

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
3. **Blocking reclaim**: if the pool is short of the inserts a task needs, the orchestrator's `ensure_tensormap_capacity` reads the latest `last_task_alive` for every ring and calls `reclaim_retired_all` (`cleanup_retired` per ring) to batch-free entries belonging to retired tasks, returning them to the free list, before the inserts proceed.

This design avoids the complexity of ring-based wrapping while still being bounded by `PTO2_TENSORMAP_POOL_SIZE` (default 65536 entries).

### 5.4 Stale Entry Cleanup: Three-Layer Defense

TensorMap must ensure entries for retired tasks (`producer_task_id < last_task_alive`) are removed, so that:

- The pool does not grow unboundedly (capacity is finite)
- Lookup performance does not degrade as stale entries accumulate in bucket chains

Three complementary mechanisms achieve this:

**Layer 1 — Chain Truncation during Lookup** (lazy, per-bucket):

Since `insert` always prepends to the bucket head, entries in each bucket chain are in **descending task_id order**. When `PTO2TensorMap::lookup` encounters the first stale entry (`producer_task_id < last_task_alive`), all subsequent entries in the chain are guaranteed stale too. The entire tail is truncated in one operation using `prev_in_bucket` pointers for O(1) unlinking.

This guarantees lookup only traverses valid entries — O(valid_entries_in_bucket), not O(total_entries).

**Layer 2 — Periodic Batch Cleanup** (`cleanup_retired`, per-task):

Every time the orchestrator submits a task (Step 0 of `PTO2OrchestratorState::submit_task`), it calls `PTO2TensorMap::sync_tensormap`. When `last_task_alive` has advanced by more than `PTO2_TENSORMAP_CLEANUP_INTERVAL` (default 64) tasks since the last cleanup, `PTO2TensorMap::cleanup_retired` runs:

This uses the **per-task entry chain** (`task_entry_head[task_slot]`) — each task's entries are doubly-linked together at insert time via `next_in_task`/`prev_in_task`, allowing O(entries_per_task) cleanup without scanning the entire pool or all buckets. Freed entries are returned to `free_entry_list` for immediate reuse.

**Layer 3 — Back-Pressure on Pool Exhaustion** (blocking):

Before STEP 4 inserts a task's outputs, `ensure_tensormap_capacity` checks the free list + bump region against the task's needed entry count. If short, it reclaims retired entries across all rings and blocks until reclaim frees enough entries. Progress is measured by entries actually freed, not by watermark movement — a ring can retire zero-output tasks, advancing `last_task_alive` without freeing any entry. A pool that frees nothing for a 500 ms wall-clock timeout is a genuine deadlock: it latches `PTO2_ERROR_TENSORMAP_OVERFLOW` and unwinds, matching the task allocator and fanin spill pool.

This forms a back-pressure mechanism analogous to the Task Ring's flow control.

**Summary**:

| Layer | Trigger | Method | Guarantees |
| ----- | ------- | ------ | ---------- |
| Chain Truncation | Every lookup | Truncate stale tail of bucket chain | Lookup only visits valid entries |
| Periodic Cleanup | Every 64 retired tasks | Walk per-task chains, free entries | Pool capacity reclaimed in bounded time |
| Pool Back-Pressure | Pool exhausted | Block until scheduler advances watermark | Hard capacity bound, no OOM |

In steady state, the number of valid TensorMap entries ≈ `active_tasks × avg_outputs_per_task`. With the default `task_window=65536` and `pool_size=65536`, this is well within bounds. With small windows (e.g., `task_window=16`), active entries are even fewer (~16 × a few), and cleanup runs frequently.

### 5.5 Dependency Discovery Flow

When `PTO2OrchestratorState::submit_task` processes parameters:

1. **INPUT/INOUT**: `PTO2TensorMap::lookup` searches for overlapping producers (with chain truncation)
2. For each producer found: `append_fanin_or_fail` adds the dependency
3. **OUTPUT/INOUT**: `PTO2TensorMap::insert` registers the current task as the new producer at bucket head
4. Stale entries are pruned lazily during lookup (Layer 1) and periodically by cleanup (Layer 2)

---

## 6. Task Descriptor and States

### 6.1 PTO2TaskDescriptor (Hot Path)

| Field | Description |
| ----- | ----------- |
| `task_id` | Canonical mixed-task ID (64-bit: `ring_id << 32 \| local_id`; `ring_id` is always 0 in this single-ring runtime). |
| `kernel_id[3]` | Per-slot kernel IDs: `[AIC, AIV0, AIV1]`; `INVALID_KERNEL_ID` = inactive |
| `active_mask` | Bitmask of active subtask slots: `bit0=AIC`, `bit1=AIV0`, `bit2=AIV1` |
| `completed_subtasks` | Atomic counter; each subtask increments on completion. Trigger condition: `completed_subtasks == total_required_subtasks` |
| `packed_buffer_base` | Start of packed buffer in GM Heap |
| `packed_buffer_end` | End of packed buffer (for heap reclamation) |

Fanin/fanout are not on the descriptor: producer ids live on the payload
(`fanin_local_ids`, §6.1b) and consumers are reached at completion via the
per-slot wake list (§8.2), not a fanout adjacency list.

### 6.1b PTO2TaskPayload (Cold Path)

| Field | Description |
| ----- | ----------- |
| `tensors[16]` | Tensor descriptors for parameters |
| `scalar_value[16]` | Scalar parameter values |
| `is_tensor[16]` | Whether each parameter is tensor or scalar |
| `param_count` | Number of valid parameters |
| `fanin_local_ids[fanin_count]` | Producer local-ids (position-independent; readiness = all their `completion_flags` set) |
| `fanin_count` | Number of producer dependencies (`<= PTO2_MAX_FANIN`, no spill) |
| `predicate` | Dispatch predicate (`DispatchPredicate`, cache line 9 / byte 576); evaluated by the scheduler at the ready point (see §8.3) |

`last_consumer_local_id` (the highest consumer id of this task) lives on the
slot state, not the payload; the host consumer-wait gates on it (§8.4).

### 6.2 Task State Machine

`task_state` is the **host-visible mirror** of completion; the device-side
readiness truth is the per-slot `completion_flags` byte (§8.2). The host polls
`task_state` in `wait_for_tensor_ready`, the allocator deadlock detector, and
the cold-path stall dump.

```text
  [0] PENDING ──worker(s) done, on_mixed_task_complete──► [1] COMPLETED
```

- **0 (PENDING)**: slot allocated; remains PENDING while waiting on producers,
  queued, or dispatched.
- **1 (COMPLETED)**: all subtasks done. `on_mixed_task_complete` sets this
  (host mirror) and, in the same step, sets `completion_flags[id]` (device
  readiness) and advances `completed_watermark`.

There is no runtime CONSUMED flip and no slot recycle: host_build_graph is
whole-graph-resident (§8.4).

---

## 7. Orchestrator

### 7.1 PTO2OrchestratorState

The orchestrator builds the task graph by calling the user-provided
orchestration function. In host_build_graph it runs **on the host** (see the
divergence note in the Overview); the `PTO2OrchestratorState` below is the same
structure tensormap drives on AICPU thread N-1.

Key members:

- `ring`: the single `PTO2RingSet` (HeapRing + TaskRing).
- `tensor_map`, `tensor_pool`: dependency tracking
- `scope_tasks[]`, `scope_begins[]`, `scope_stack_top`: scope nesting stack (flat buffer partitioned by level)
- `scheduler`: pointer to scheduler state (for seeding zero-fanin tasks into the ready queue)
- `gm_heap_base`, `gm_heap_size`: GM heap for output buffers

### 7.2 Task Submission Flow (`PTO2OrchestratorState::submit_task`)

| Step | Operation |
| ---- | --------- |
| 0 | `PTO2TensorMap::sync_tensormap` — prune stale TensorMap entries |
| 1 | `PTO2TaskAllocator::alloc` — allocate task slot (may block on flow control) |
| 2 | Initialize task descriptor + slot state, copy parameters |
| 3 | **Lookup**: for each INPUT/INOUT param, search TensorMap for producers; collect producer ids in `PTO2FaninBuilder` |
| 4 | **Insert**: register OUTPUT/INOUT args in TensorMap |
| 5 | **Record fanin**: `append_fanin_or_fail` dedups producers and writes their local-ids into `payload->fanin_local_ids[]`; each producer's `last_consumer_local_id` is bumped to this task's id. `payload.fanin_count` is set from the builder. |
| 6 | **Seed readiness**: a task with zero fanin (`fanin_count == 0`) is pushed straight to the ready queue via `push_ready_routed`. A task with fanin is left for the device boot classify (§8) — the host does not pre-register wake lists. |

> **Note**: There is no fanout adjacency, dep-pool, or per-producer lock. A
> producer inline-completed on the host (e.g. a hidden-alloc task) pre-sets its
> own `completion_flags[id] = 1` in the H2D image so device consumers see it as
> already satisfied. The only cross-task pointers the image carries are the
> per-slot `task` / `payload` pointers, which `relocate_host_orch_image` shifts
> to device addresses before H2D.

### 7.3 Dependency Recording and Relocation

`append_fanin_or_fail` records, per consumer, the deduped list of producer
local-ids into `payload->fanin_local_ids[]` and bumps `fanin_count`; it also
raises each producer's `last_consumer_local_id` to the consumer's id. An
overflow past `PTO2_MAX_FANIN` is fatal (`PTO2_ERROR_...`), since there is no
spill.

`relocate_host_orch_image` runs on the host before H2D. Because fanin is
position-independent integer ids, the only pointers needing fixup are the
per-slot `task` and `payload` pointers (SM-region delta); the fanout adjacency,
dep-pool, and ready-queue pointer relocation of the wiring model are gone.

Readiness and completion are handled entirely device-side by the scheduler:
the boot classify seeds the ready queue and registers wake lists (§8.2), and
`on_mixed_task_complete` publishes each producer's `completion_flags` and drains
its wake list. See §8.2 for the completion protocol.

### 7.4 Scope Mechanism (`PTO2_SCOPE`)

Scopes control the lifetime of intermediate buffers. Each scope:

- Tracks tasks submitted within it via a flat `scope_tasks[]` buffer partitioned by `scope_begins[]`
- Scopes bound intermediate-buffer lifetime **structurally** (the orchestration function that built the graph). host_build_graph is whole-graph-resident, so `scope_end` performs no runtime buffer reclaim — there is no `fanout_refcount`.

```cpp
PTO2_SCOPE(rt) {
    // Tasks submitted here belong to this scope
    rt_submit_aic_task(FUNC_QK, args);
    rt_submit_aiv_task(FUNC_SF, args);
}
// scope_end: scope reference released from all tasks above
```

**Output tensor lifetime — single-scope only.** `submit_task` returns a
`TaskOutputTensors`, and `get_ref(i)` hands back a `const Tensor&`. Both are
backed by pointers into the submitting task's `PTO2TaskPayload::tensors[]`,
which lives in a ring-buffer slot. host_build_graph does not recycle slots
within a run (whole-graph-resident, no execution-time reclaim), so the storage
is not overwritten mid-run; the constraint is instead structural — the
`TaskOutputTensors` and the refs it returns are scoped to the orchestration
function that built the graph and must not be retained past it.

Therefore the `TaskOutputTensors` instance, the references it returns, and
any pointer derived from them MUST NOT outlive the `PTO2_SCOPE` in which
submit was called. The typical safe pattern is:

```cpp
PTO2_SCOPE() {
    TaskOutputTensors outs = rt_submit_aic_task(FUNC_QK, args);
    const Tensor &y = outs.get_ref(0);
    // Use y here and in subsequent submits within the same scope.
}   // outs and y both go out of scope; no dangling references can escape.
```

Anti-patterns that compile but silently break:

```cpp
const Tensor *kept = nullptr;
PTO2_SCOPE() {
    TaskOutputTensors outs = rt_submit_aic_task(FUNC_QK, args);
    kept = &outs.get_ref(0);          // escapes the scope
}
// `kept` still points at a payload slot. After enough submits in later
// scopes, the slot is reused and `*kept` aliases an unrelated task's
// tensor — a wrong-tensor read with no runtime diagnostic.

TaskOutputTensors outs;               // declared in outer scope
PTO2_SCOPE() {
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

With `aicpu_thread_num=4` and `block_dim=24`, the AICPU runs 4 scheduler
threads (thread 3 also performs the one-time host-orch boot before it starts
dispatching):

| Thread | Role | Cores |
| ------ | ---- | ----- |
| 0 | Scheduler | 6 AIC + 12 AIV |
| 1 | Scheduler | 6 AIC + 12 AIV |
| 2 | Scheduler | 6 AIC + 12 AIV |
| 3 | Scheduler (+ host-orch boot) | 6 AIC + 12 AIV |

Core assignment: clusters (1 AIC + 2 AIV) are divided equally among all 4
scheduler threads.

### 8.2 Scheduler Main Loop

Each scheduler thread runs a tight loop with two main phases:

**Phase 1 — Completion Handling (polling)**:

- Poll register `COND` on each managed core.
- When `TASK_FIN_STATE` detected: call `on_subtask_complete`; when
  `completed_subtasks == total_required_subtasks`, call `on_mixed_task_complete`,
  which:
  1. mirrors `task_state = COMPLETED` (host-visible) and sets the device
     readiness truth `completion_flags[my_id] = 1` (release);
  2. drains this task's intrusive **wake list** — `wake_list_head.exchange(SENTINEL)`
     — reclassifying each waiter: a waiter whose remaining fanin is now all met
     is pushed via `push_ready_routed`; otherwise it re-registers on its next
     unmet producer. After the exchange the head is `SENTINEL`, so a consumer
     registering concurrently re-checks the flags instead of being lost;
  3. CAS-advances `completed_watermark` over the contiguous completed prefix
     (§8.4).

**Readiness / wake registration.** A task is ready iff every id in its
`fanin_local_ids` has its `completion_flags` byte set (`fanin_satisfied` /
`classify_fanin_state`, acquire loads). A not-yet-ready task registers itself on
its **first unmet** producer's wake list (`register_wake`); that producer's
completion re-drives the classification. The decision is terminal — tasks are
never re-polled — because `completion_flags` are monotonic. This wake machinery
is seeded by the device **boot classify** (`on_orchestration_done`), which scans
the submitted tasks once and either pushes the fanin-free ones to the ready
queue or registers each remaining task on its first unmet producer.

**Phase 2 — Dispatch**:

- For each idle core: pop a task from the matching shape-based ready queue (lock-free MPMC Vyukov queue, one per resource shape)
- Build `PTO2DispatchPayload` from `TaskDescriptor` with `task_id`, `subslot`, `kernel_id`, and `core_type`
- Write task pointer to `Handshake.task`, signal AICore via register `DATA_MAIN_BASE`

After these phases, the scheduler updates profiling headers and checks for termination (all tasks completed and orchestrator done).

### 8.3 Ready Queue Design

Ready queues use a lock-free bounded MPMC (Vyukov) design:

- One `PTO2ReadyQueue` per resource shape (5 shapes: `AIC_ONLY`, `AIV_X1`, `AIV_X2`, `AIC_AIV_X1`, `AIC_AIV_X2`)
- **Push**: any thread (orchestrator via `init_task`, or scheduler on completion) pushes newly-ready tasks to the queue matching `task->active_mask.to_shape()`
- **Pop**: scheduler threads pop from the queue matching the idle core's resource shape
- Per-slot sequence counters prevent ABA problems
- `enqueue_pos` and `dequeue_pos` are on separate cache lines to avoid false sharing

### 8.4 Completion Watermark (host consumer-wait gate)

`completed_watermark` is the highest id such that every task in
`[0, completed_watermark]` has its `completion_flags` byte set. The tail of
`on_mixed_task_complete` CAS-advances it over the **full contiguous completed
prefix** (bounded by `current_task_index`, not by the completing task's own id)
— capping at `my_id` would make the final value completion-order-dependent and
strand it below the true prefix.

It is **load-bearing**: the host `wait_for_tensor_ready(..., wait_for_consumers)`
gates on `completed_watermark >= producer.last_consumer_local_id` to observe
"every consumer of this producer has retired" — replacing the wiring model's
`fanout_refcount == fanout_count` check.

Slot reclaim is inert: host_build_graph is whole-graph-resident, so
`last_task_alive` is never advanced at runtime and there is no
`advance_ring_pointers` step. `reset_for_reuse()` runs **once at init**
(`pto_shared_memory.cpp`) to zero each slot before the host orchestrator
populates it — it is not a runtime recycle hook.

### 8.5 SchedulerContext

All scheduler-side state and methods live in `SchedulerContext` (`runtime/scheduler/scheduler_context.h`). It is held as a `sched_ctx_` member of `AicpuExecutor`; `AicpuExecutor` is a thin wrapper that owns the lifecycle atomics and delegates everything else to `SchedulerContext`.

Public surface (called from `AicpuExecutor::init/run/deinit`):

| Method | Phase | Purpose |
| ------ | ----- | ------- |
| `init(runtime, aicpu_thread_num, regs_base)` | once per run | Handshake + assign cores, reset counters, latch `regs_base`, bind `func_id_to_addr_` |
| `bind_runtime(rt)` | boot thread | Wire `sched_` to `rt->scheduler` once the boot thread attaches the host-built `rt` |
| `resolve_and_dispatch(runtime, thread_idx)` | per scheduler thread | Main dispatch loop |
| `shutdown(thread_idx)` | per thread on exit | `platform_deinit_aicore_regs` for this thread's cores; PMU finalize when enabled |
| `on_orchestration_done(runtime, rt, thread_idx, total_tasks)` | boot thread | Publish core assignments, latch task count, fold inline-completed tasks, flip `orchestrator_done_` (or `emergency_shutdown` on fatal) |
| `deinit()` | once per run | Reset every scheduler-owned field to its post-construction default |
| Read-only accessors | various | `aic_count()` / `aiv_count()` / `is_completed()` / `completed_tasks_count()` |

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

> **Note**: `task_id` is 64-bit but registers are 32-bit. A per-core monotonic dispatch counter (`s_dispatch_seq`) replaces `task_id` in register writes to prevent collisions.

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
   and packs all children into a single `ChipCallable` buffer alongside the
   orchestration SO.
2. `host_api.upload_chip_callable_buffer(callable)` H2Ds the whole buffer
   once and returns the device address of the ChipCallable header.
3. For each child, host computes
   `chip_dev + offsetof(ChipCallable, storage_) + callable->child_offset(i)`
   and stores it in `Runtime.func_id_to_addr_[child_func_id(i)]`.
4. When dispatching, the scheduler reads `func_id_to_addr_[fid]`, casts to
   `const CoreCallable*`, reads `resolved_addr_`, and copies that into
   `PTO2DispatchPayload.function_bin_addr`.

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

Profiling-subsystem init (`dump_args` / `pmu` / `dep_gen` / `l2_swimlane`) runs
once in `SchedulerContext::init()` on the single-threaded cold path, before any
scheduler/orchestrator thread starts — so it needs no cross-thread init
handshake.

Startup sequence:

1. Thread 3: create SM handle + runtime → set `runtime_init_ready_`
2. Scheduler threads: wait for `runtime_init_ready_` → enter main loop
3. Thread 3: configure orchestrator-scheduler pointers → call orchestration function → set `orchestrator_done_`

---

## 11. PTO2 Orchestration API

The orchestration API is defined in `pto_orchestration_api.h`. Orchestration code depends only on this header.

### 11.1 Core API

| Function/Macro | Purpose |
| -------------- | ------- |
| `rt_submit_task(mixed_kernels, args)` | Submit a mixed task with `MixedKernels` struct |
| `rt_submit_aic_task(kernel_id, args)` | Convenience: submit AIC-only task |
| `rt_submit_aiv_task(kernel_id, args)` | Convenience: submit AIV-only task |
| `PTO2_SCOPE() { ... }` | RAII scope for buffer lifetime |
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
