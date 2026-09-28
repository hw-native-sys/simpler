# Multi-Ring Buffer Architecture

> Extension to `tensormap_and_ringbuffer`. For the base architecture, see [RUNTIME_LOGIC.md](RUNTIME_LOGIC.md).

## 1. Problem

The single-ring design uses one `last_task_alive` watermark shared by HeapRing, TaskRing, and DepPool. When tasks from an inner scope (e.g., per-block iteration) complete, their resources cannot be reclaimed until **all** prior tasks — including those from the outer scope — also complete. This wastes ring capacity and can trigger deadlocks when ring sizes are small.

## 2. Solution

Split HeapRing, TaskRing, and DepPool into arrays of `CHIP_MAX_RING_DEPTH` (4) independent instances. Each scope depth maps to its own ring, with an independent `last_task_alive` watermark.

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
TaskId::to_uint64(task_id) == (ring_id << 32) | local_id
```

`TaskId` is this runtime's own type, declared with its encoding in
`src/common/tensormap_and_ringbuffer/task_id.h`. The encoding is private, so a
factory is the only way to mint a handle and an accessor the only way to read a
field out of one:

| API | Purpose |
| --- | ------- |
| `TaskId::make(ring_id, local_id)` | Compose a 64-bit task ID (`TaskId`) |
| `task_id.ring()` | Extract `ring_id` (bits 39-32) |
| `task_id.local_id()` | Extract `local_id` (bits 31-0) |
| `TaskId::to_uint64(task_id)` | The packed encoding as one integer, for the uses that need a number rather than the handle: log formatting, JSON serialization, hash distribution |

Type changes:

| Field | Before | After |
| ----- | ------ | ----- |
| `TaskDescriptor.task_id` | `int32_t` | `TaskId` |
| `ChipTensorMapEntry.producer_task_id` | `int32_t` | `TaskId` |
| `ChipTaskSlotState.ring_id` | N/A | `uint8_t` (new, denormalized for fast access) |

## 4. Data Structures

### 4.1 ChipRingSet (new)

Bundles the per-ring resources into a single aggregate (`ring_buffer.h`):

```cpp
struct ChipRingSet {
    TaskAllocator task_allocator;
    FaninPool fanin_pool;
};
```

`TaskAllocator` owns both the task window and the heap. The two are always
allocated together, so it checks each before committing to either and needs no
rollback on partial failure — which is why there is no separate heap-ring or
task-ring type to hold.

### 4.2 OrchestratorState (modified)

```cpp
// Before: one set of ring resources, held directly
TaskAllocator task_allocator;
DepListPool dep_pool;

// After: per-ring array (dep_pool moved to scheduler, see §4.5)
ChipRingSet rings[CHIP_MAX_RING_DEPTH];
```

Ring selection: `current_ring_id() = min(scope_stack_top, CHIP_MAX_RING_DEPTH - 1)`.

### 4.3 SharedMemoryHeader (modified)

Per-ring flow control and per-ring layout info are grouped together:

```cpp
struct ChipRingFlowControl {
    std::atomic<int32_t> current_task_index;  // task ring head
    std::atomic<int32_t> last_task_alive;     // task ring tail
    std::atomic<uint64_t> heap_top;           // heap alloc pointer
    std::atomic<uint64_t> heap_tail;          // heap reclaim pointer
};

struct alignas(64) SharedMemoryRingHeader {
    ChipRingFlowControl fc;

    // Layout metadata (set once at init)
    uint64_t task_window_size;
    int32_t task_window_mask;       // task_window_size - 1
    uint64_t heap_size;
    uint64_t task_descriptors_offset;

    // Per-ring data pointers (host-side, set by setup_pointers)
    TaskDescriptor *task_descriptors;
    TaskPayload *task_payloads;
    ChipTaskSlotState *slot_states;

    // Accessors (slot = local_id & task_window_mask)
    TaskDescriptor &get_task_by_slot(int32_t slot);
    TaskDescriptor &get_task_by_task_id(int32_t local_id);
    TaskPayload &get_payload_by_slot(int32_t slot);
    TaskPayload &get_payload_by_task_id(int32_t local_id);
    ChipTaskSlotState &get_slot_state_by_slot(int32_t slot);
    ChipTaskSlotState &get_slot_state_by_task_id(int32_t local_id);
};

// In header:
SharedMemoryRingHeader rings[CHIP_MAX_RING_DEPTH];
```

Per-ring try-locks in the scheduler state prevent concurrent scheduler threads from interleaving watermark writes within the same ring. `FaninPool`/`DepListPool` `reclaim`/`ensure_space` take `SharedMemoryRingHeader&` directly (no `ring_id` or `fc` parameters).

### 4.4 SharedMemoryHandle (lifecycle-only)

Slimmed to lifecycle management only. Per-ring data pointers now live in `SharedMemoryRingHeader` (§4.3). Runtime components (orchestrator, scheduler) store `SharedMemoryHeader*` directly, eliminating one indirection on every per-ring access.

```cpp
struct SharedMemoryHandle {
    void *sm_base;
    uint64_t sm_size;
    SharedMemoryHeader *header;
    bool is_owner;
};
```

### 4.5 SchedulerState (modified)

```cpp
struct RingSchedState {
    // Cache Line 0: ring pointer (read-only) + hot path (read-write)
    SharedMemoryRingHeader *ring;  // direct pointer, no indirection
    int32_t last_task_alive;
    std::atomic<int32_t> advance_lock;  // multi-thread CAS

    // Cache Line 1+: Orch-side wiring dep_pool, cache-isolated
    alignas(64) DepListPool dep_pool;
};

RingSchedState ring_sched_states[CHIP_MAX_RING_DEPTH];
```

`slot_states`, `task_window_size`, and `task_window_mask` are no longer duplicated — callers access them via `ring->get_slot_state_by_*()` and other ring header accessors. The ring pointer shares cache line 0 with `last_task_alive` and `advance_lock`.

### 4.6 ChipTensorMap (modified)

```cpp
ChipTensorMapEntry** task_entry_heads[CHIP_MAX_RING_DEPTH];
int64_t last_task_alives[CHIP_MAX_RING_DEPTH];
```

Entry validity checks and `cleanup_retired` operate per-ring:

```cpp
bool entry_valid(const ChipTensorMapEntry& e) {
    int32_t ring = e.producer_task_id.ring();
    int32_t local = e.producer_task_id.local_id();
    return local >= last_task_alives[ring];
}
```

### 4.7 Unchanged Structures

| Structure | Reason |
| --------- | ------ |
| `DepListEntry` | Stores `ChipTaskSlotState*` pointer — naturally crosses ring boundaries |
| `TaskPayload` | `fanin_slot_states[]` are pointers — no ring coupling |
| `ChipReadyQueue` | Global ready queues shared across all rings (tasks ready to dispatch regardless of origin ring) |
| `DispatchPayload` | Built per-dispatch, no ring state needed |

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

Per-ring try-locks in the scheduler state prevent concurrent scheduler threads from interleaving heap_tail writes within the same ring. A scheduler thread that changes a ring head to `CONSUMED` but fails to acquire that ring's `advance_lock` records a coalesced request in `advance_pending_mask`. Scheduler no-progress iterations retry pending rings under the same lock; a busy lock leaves the bit set, and a successful retry clears the bit before rescanning.

For ring-heap stall triage, a `CONSUMED` head whose ring bit remains set means the deferred request has not yet been cleared by a retry that acquired `advance_lock`. If the bit clears and `last_task_alive` is still pinned, the stall is not caused by this deferred advance path.

### 5.2 Cross-Ring Dependencies

Dependency edges use `ChipTaskSlotState*` pointers, which naturally span rings:

- Ring 1 task depends on ring 0 producer → ring 0's `fanout_head` linked list contains a ring 1 `ChipTaskSlotState*`
- When ring 0 task completes, it walks its fanout list and decrements ring 1 consumers' `fanin_refcount`
- No special cross-ring logic needed — pointer-based design is ring-agnostic

### 5.3 DepPool Reclamation

DepPool entries are allocated by the orchestrator during Orch-side wiring and reclaimed during watermark advancement:

```text
// Called during ring watermark advancement:
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
| `CHIP_TASK_WINDOW_SIZE` | 16384 | 65536 |
| `CHIP_HEAP_SIZE` | 256 MB | 1 GB |
| `CHIP_DEP_LIST_POOL_SIZE` | 16384 | 65536 |

### 7.2 Runtime Overrides

Each ring resource (`ring_task_window` / `ring_heap` / `ring_dep_pool`) is a
single `CallConfig.runtime_env` field that accepts **either** a scalar (broadcast
to every ring) **or** a list of four per-ring values. Precedence is resolved
independently for each resource and ring:

```text
per-ring CallConfig entry (a scalar is broadcast to every entry)
  > compile-time default
```

`ring_id` is the scope-depth ring selected by the runtime:

```text
scope depth 0 -> ring 0
scope depth 1 -> ring 1
scope depth 2 -> ring 2
scope depth >=3 -> ring 3
```

Per-task via `CallConfig.runtime_env` — different L2 tasks in one launch can
each carry their own sizes. Invalid values raise at submit time (`validate()`).
Assign a scalar to size every ring the same:

```python
cfg = CallConfig()
cfg.runtime_env.ring_task_window = 128   # power of 2, >= 4
cfg.runtime_env.ring_heap = 262144       # bytes/ring, >= 1024
cfg.runtime_env.ring_dep_pool = 256      # 4 .. INT32_MAX
orchestrator.submit_next_level(handle, args, cfg, worker=0)
```

Assign a four-entry list to tune the scope-depth rings independently. The list
must contain exactly four entries; use `0` for an entry that should fall through
to the next precedence tier. All `CallConfig` values are integer byte/count
values, and each field always reads back as a four-entry list.

```python
cfg = CallConfig()
cfg.runtime_env.ring_task_window = [8192, 16384, 131072, 524288]
cfg.runtime_env.ring_heap = [
    128 * 1024 * 1024,
    256 * 1024 * 1024,
    384 * 1024 * 1024,
    512 * 1024 * 1024,
]
cfg.runtime_env.ring_dep_pool = [4096, 8192, 16384, 32768]
orchestrator.submit_next_level(handle, args, cfg, worker=0)
```

Scene tests set the same keys under a nested `runtime_env` block in the
per-case `config` dict — each value is a scalar or a four-entry list:

```python
"config": {
    "runtime_env": {
        "ring_task_window": [8192, 16384, 131072, 524288],
        "ring_heap": [134217728, 268435456, 402653184, 536870912],
        "ring_dep_pool": 256,  # scalar broadcasts to every ring
    }
}
```

There is no process-wide fallback. `PTO2_RING_TASK_WINDOW` / `PTO2_RING_HEAP` /
`PTO2_RING_DEP_POOL` used to size every ring in the process; they are not read any
more, and the runtime logs a warning if one is still exported, because otherwise
the requested sizing would vanish into the compile-time default. Size the rings
per task instead — that is what the `CallConfig` block above does, and it is
strictly more expressive: two tasks in one process can hold different ring sizes,
which the env never allowed.

Use `--enable-scope-stats` to confirm the effective values for a real run. The
first line of `scope_stats/scope_stats.jsonl` includes `task_window_max`,
`heap_max`, and `dep_pool_max`, indexed by `ring`.

### 7.3 Sizing Guidelines

- `task_window` must be ≥ max tasks in any single scope + headroom for concurrent scopes
- `heap` must accommodate peak output buffer allocation across all in-flight tasks on that ring
- `dep_pool` must be ≥ total dependency entries for all in-flight tasks on that ring
- On hardware, back-pressure latency is higher than in simulation — size conservatively
- Adding inner `SIMPLER_SCOPE` reduces peak per-ring usage, enabling smaller sizes
