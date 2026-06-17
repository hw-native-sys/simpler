# Multi-Ring Buffer Architecture

> Extension to the PTO2 runtime. For the base architecture, see [RUNTIME_LOGIC.md](RUNTIME_LOGIC.md).

## 1. Problem

The single-ring design uses one `last_task_alive` watermark shared by HeapRing, TaskRing, and DepPool. When tasks from an inner scope (e.g., per-block iteration) complete, their resources cannot be reclaimed until **all** prior tasks — including those from the outer scope — also complete. This wastes ring capacity and can trigger deadlocks when ring sizes are small.

## 2. Solution

Split HeapRing, TaskRing, and DepPool into arrays of `PTO2_MAX_RING_DEPTH` (4) independent instances. Each scope depth maps to its own ring, with an independent `last_task_alive` watermark.

```text
Scope depth 0  ──►  rings[0] = { HeapRing, TaskRing, DepPool }
Scope depth 1  ──►  rings[1] = { HeapRing, TaskRing, DepPool }
Scope depth 2  ──►  rings[2] = { HeapRing, TaskRing, DepPool }
Scope depth ≥3 ──►  rings[3] = { HeapRing, TaskRing, DepPool }  (clamped)
```

Inner-scope tasks can now be reclaimed independently without waiting for outer-scope tasks to complete.

## 3. Task ID Encoding

Task IDs are widened from 32-bit to 64-bit to carry the ring identity:

```text
task_id.raw = (ring_id << 32) | local_id
```

`PTO2TaskId` exposes direct accessors in `pto_runtime2_types.h`:

| API | Purpose |
| --- | ------- |
| `PTO2TaskId::make(ring_id, local_id)` | Compose a 64-bit task ID (`PTO2TaskId`) |
| `task_id.ring()` | Extract `ring_id` (bits 63-32) |
| `task_id.local()` | Extract `local_id` (bits 31-0) |
| `task_id.raw` | Access the packed 64-bit encoding |

Type changes:

| Field | Before | After |
| ----- | ------ | ----- |
| `PTO2TaskDescriptor.task_id` | `int32_t` | `PTO2TaskId` |
| `PTO2TensorMapEntry.producer_task_id` | `int32_t` | `PTO2TaskId` |
| `PTO2TaskSlotState.ring_id` | N/A | `uint8_t` (new, denormalized for fast access) |

## 4. Data Structures

### 4.1 PTO2RingSet (new)

Bundles the three per-ring resources into a single aggregate (`pto_ring_buffer.h`):

```cpp
struct PTO2RingSet {
    PTO2HeapRing   heap_ring;
    PTO2TaskRing   task_ring;
    PTO2FaninPool fanin_pool;
};
```

### 4.2 PTO2OrchestratorState (modified)

```cpp
// Before: single ring
PTO2HeapRing heap_ring;
PTO2TaskRing task_ring;
PTO2DepListPool dep_pool;

// After: per-ring array (dep_pool moved to scheduler, see §4.5)
PTO2RingSet rings[PTO2_MAX_RING_DEPTH];
```

Ring selection: `current_ring_id() = min(scope_stack_top, PTO2_MAX_RING_DEPTH - 1)`.

### 4.3 PTO2SharedMemoryHeader (modified)

Per-ring flow control and per-ring layout info are grouped together:

```cpp
struct PTO2RingFlowControl {
    std::atomic<int32_t> current_task_index;  // task ring head
    std::atomic<int32_t> last_task_alive;     // task ring tail
    std::atomic<uint64_t> heap_top;           // heap alloc pointer
    std::atomic<uint64_t> heap_tail;          // heap reclaim pointer
};

struct alignas(64) PTO2SharedMemoryRingHeader {
    PTO2RingFlowControl fc;

    // Layout metadata (set once at init)
    uint64_t task_window_size;
    int32_t task_window_mask;       // task_window_size - 1
    uint64_t heap_size;
    uint64_t task_descriptors_offset;

    // Per-ring data pointers (host-side, set by setup_pointers)
    PTO2TaskDescriptor *task_descriptors;
    PTO2TaskPayload *task_payloads;
    PTO2TaskSlotState *slot_states;

    // Accessors (slot = local_id & task_window_mask)
    PTO2TaskDescriptor &get_task_by_slot(int32_t slot);
    PTO2TaskDescriptor &get_task_by_task_id(int32_t local_id);
    PTO2TaskPayload &get_payload_by_slot(int32_t slot);
    PTO2TaskPayload &get_payload_by_task_id(int32_t local_id);
    PTO2TaskSlotState &get_slot_state_by_slot(int32_t slot);
    PTO2TaskSlotState &get_slot_state_by_task_id(int32_t local_id);
};

// In header:
PTO2SharedMemoryRingHeader rings[PTO2_MAX_RING_DEPTH];
```

Per-ring try-locks in the scheduler state prevent concurrent scheduler threads from interleaving watermark writes within the same ring. `FaninPool`/`DepListPool` `reclaim`/`ensure_space` take `PTO2SharedMemoryRingHeader&` directly (no `ring_id` or `fc` parameters).

### 4.4 PTO2SharedMemoryHandle (lifecycle-only)

Slimmed to lifecycle management only. Per-ring data pointers now live in `PTO2SharedMemoryRingHeader` (§4.3). Runtime components (orchestrator, scheduler) store `PTO2SharedMemoryHeader*` directly, eliminating one indirection on every per-ring access.

```cpp
struct PTO2SharedMemoryHandle {
    void *sm_base;
    uint64_t sm_size;
    PTO2SharedMemoryHeader *header;
    bool is_owner;
};
```

### 4.5 PTO2SchedulerState (modified)

```cpp
struct RingSchedState {
    // Cache Line 0: ring pointer (read-only) + hot path (read-write)
    PTO2SharedMemoryRingHeader *ring;  // direct pointer, no indirection
    int32_t last_task_alive;
    std::atomic<int32_t> advance_lock;  // multi-thread CAS

    // Cache Line 1+: Thread 0 only (wiring dep_pool, cache-isolated)
    alignas(64) PTO2DepListPool dep_pool;
};

RingSchedState ring_sched_states[PTO2_MAX_RING_DEPTH];
PTO2SpscQueue wiring_queue;  // global SPSC queue: orchestrator pushes, scheduler thread 0 drains
```

`slot_states`, `task_window_size`, and `task_window_mask` are no longer duplicated — callers access them via `ring->get_slot_state_by_*()` and other ring header accessors. The ring pointer shares cache line 0 with `last_task_alive` and `advance_lock`.

### 4.6 PTO2TensorMap (modified)

```cpp
PTO2TensorMapEntry** task_entry_heads[PTO2_MAX_RING_DEPTH];
int64_t last_task_alives[PTO2_MAX_RING_DEPTH];
```

Entry validity checks and `cleanup_retired` operate per-ring:

```cpp
bool entry_valid(const PTO2TensorMapEntry& e) {
    int32_t ring = e.producer_task_id.ring();
    int32_t local = e.producer_task_id.local();
    return local >= last_task_alives[ring];
}
```

### 4.7 Unchanged Structures

| Structure | Reason |
| --------- | ------ |
| `PTO2DepListEntry` | Stores `PTO2TaskSlotState*` pointer — naturally crosses ring boundaries |
| `PTO2TaskPayload` | `fanin_slot_states[]` are pointers — no ring coupling |
| `PTO2ReadyQueue` | Global ready queues shared across all rings (tasks ready to dispatch regardless of origin ring) |
| `PTO2DispatchPayload` | Built per-dispatch, no ring state needed |

## 5. Reclamation

### 5.1 Per-Ring Watermark Advancement

Each ring's `last_task_alive` advances independently:

```text
advance_ring_pointers(ring_id):  // protected by per-ring advance_lock
    la = ring->fc.last_task_alive
    while ring->get_slot_state_by_task_id(la).task_state >= CONSUMED:
        reset slot for reuse
        la++
    sync_to_sm()  // release-store last_task_alive
```

Per-ring try-locks in the scheduler state prevent concurrent scheduler threads from interleaving heap_tail writes within the same ring.

### 5.2 Cross-Ring Dependencies

Dependency edges use `PTO2TaskSlotState*` pointers, which naturally span rings:

- Ring 1 task depends on ring 0 producer → ring 0's `fanout_head` linked list contains a ring 1 `PTO2TaskSlotState*`
- When ring 0 task completes, it walks its fanout list and decrements ring 1 consumers' `fanin_refcount`
- No special cross-ring logic needed — pointer-based design is ring-agnostic

### 5.3 DepPool Reclamation

DepPool is exclusively managed by scheduler thread 0 (allocation during wiring, reclamation during watermark advancement):

```text
// Called by scheduler thread 0 during wiring_queue drain:
dep_pool_reclaim(ring_id):
    la = ring->fc.last_task_alive
    newest_consumed = la - 1
    mark = ring->get_slot_state_by_task_id(newest_consumed).dep_pool_mark
    if mark > 0:
        ring_sched_states[ring_id].dep_pool.advance_tail(mark)
```

Note: dep entries from ring N's pool may appear in ring M's fanout lists. Reclamation is safe because the entries are accessed during fanout traversal (completion time), which always happens before the consumer task — and therefore the dep entry — becomes eligible for reclamation.

## 6. AICPU Register Protocol Fix

The AICore dispatch protocol uses 32-bit registers. With multi-ring, `task_id` truncation to 32-bit loses the `ring_id`, causing collisions:

```text
Ring 0, local_id=0  →  DATA_MAIN_BASE = 0 + 1 = 1
Ring 1, local_id=0  →  DATA_MAIN_BASE = 0 + 1 = 1  (collision!)
```

AICore uses `last_reg_val` to detect new dispatches — identical values cause skipped tasks and false completions from stale COND registers.

**Fix**: Per-core monotonic dispatch counter `s_dispatch_seq[core_id]` replaces `task_id` in register writes, guaranteeing unique `DATA_MAIN_BASE` values per core regardless of ring origin.

## 7. Configuration

### 7.1 Compile-Time Defaults (per ring)

| Constant | Default | Total (×4 rings) |
| -------- | ------- | ---------------- |
| `PTO2_TASK_WINDOW_SIZE` | 16384 | 65536 |
| `PTO2_HEAP_SIZE` | 256 MB | 1 GB |
| `PTO2_DEP_LIST_POOL_SIZE` | 16384 | 65536 |

### 7.2 Runtime Environment Overrides

Uniform (applies to all rings):

```bash
PTO2_RING_TASK_WINDOW=1024
PTO2_RING_HEAP=1048576
PTO2_RING_DEP_POOL=1024
```

In `kernel_config.py`:

```python
RUNTIME_ENV = {
    "PTO2_RING_TASK_WINDOW": "128",
    "PTO2_RING_HEAP": "262144",
    "PTO2_RING_DEP_POOL": "256",
}
```

### 7.3 Sizing Guidelines

- `task_window` must be ≥ max tasks in any single scope + headroom for concurrent scopes
- `heap` must accommodate peak output buffer allocation across all in-flight tasks on that ring
- `dep_pool` must be ≥ total dependency entries for all in-flight tasks on that ring
- On hardware, back-pressure latency is higher than in simulation — size conservatively
- Adding inner `PTO2_SCOPE` reduces peak per-ring usage, enabling smaller sizes
