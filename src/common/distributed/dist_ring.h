/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * DistRing — unified slot + heap allocator for L3+ distributed workers.
 *
 * A single structure owns three correlated per-task resources:
 *
 *   1. A monotonic task id (`next_task_id_`), allocated by the Orchestrator.
 *      Unlike L2's `PTO2TaskAllocator` the id is NOT masked into a fixed-size
 *      window — slot state lives in parent-process heap (never crossed into
 *      child workers), so a ring index buys us nothing at L3 (see the plan's
 *      L2 Consistency Audit, allowed exception #6).
 *   2. `DIST_MAX_RING_DEPTH` independent shared-memory heap slabs (Strict-1,
 *      matches L2's `PTO2_MAX_RING_DEPTH = 4`). Each slab has its own
 *      `mmap(MAP_SHARED)` region, bump cursor, FIFO reclamation pointer,
 *      mutex and cv. Slot → ring mapping is driven by scope depth:
 *         ring_idx = min(scope_depth, DIST_MAX_RING_DEPTH - 1)
 *      so tasks inside nested scopes reclaim independently of the outer
 *      scope's long-lived allocations. A mapping taken before any fork is
 *      inherited by every child process at the same virtual address.
 *   3. The per-task scheduling state (`DistTaskSlotState`). Stored in a
 *      `std::deque<std::unique_ptr<...>>` so push_back never invalidates
 *      pointers, and destruction happens only at `reset_to_empty()` /
 *      process teardown. Slot state records its `ring_idx` and
 *      `ring_slot_idx` so `release(slot)` knows which ring to advance.
 *
 * Back-pressure: only the heap can be full, and per-ring. `alloc(bytes,
 * scope_depth)` picks a ring, spin-waits on that ring's cv; if no progress
 * is made for `timeout_ms` it throws `std::runtime_error`. Other rings
 * remain usable while one ring is full.
 *
 * Lifecycle:
 *
 *   Worker.run() → orch submits N tasks → each submit calls
 *   `ring.alloc(bytes, scope_depth)` (task id allocated, slot state
 *   constructed, ring slab carved) → scheduler dispatches → workers run →
 *   on_consumed calls `ring.release(id)` which advances the slot's ring's
 *   `last_alive` FIFO-wise → drain waits until `active_count() == 0` →
 *   `ring.reset_to_empty()` resets every ring's cursors and drops all slot
 *   states so the next run starts fresh.
 */

#pragma once

#include <array>
#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "dist_types.h"

// User-facing output alignment (Strict-3; matches L2 PTO2_PACKED_OUTPUT_ALIGN).
static constexpr uint64_t DIST_HEAP_ALIGN = 1024;

// Default PER-RING heap size. Total VA reservation is this value times
// DIST_MAX_RING_DEPTH (default 4 GiB across 4 rings of 1 GiB each).
static constexpr uint64_t DIST_DEFAULT_HEAP_RING_SIZE = 1ULL << 30;

// Default back-pressure timeout (ms). Surfaces as std::runtime_error when the
// target ring makes no progress for this long — acts as a deadlock detector.
static constexpr uint32_t DIST_ALLOC_TIMEOUT_MS = 10000;

// Align an unsigned value up to the next multiple of `align` (must be power of 2).
inline uint64_t dist_align_up(uint64_t v, uint64_t align) { return (v + align - 1) & ~(align - 1); }

// Scope-depth → ring-index mapping. `scope_depth` is L2-style 0-based (the
// outermost open scope is 0); deeper scopes share the innermost ring.
inline int32_t dist_ring_idx_for_scope(int32_t scope_depth) {
    if (scope_depth < 0) scope_depth = 0;
    return scope_depth < DIST_MAX_RING_DEPTH ? scope_depth : DIST_MAX_RING_DEPTH - 1;
}

struct DistAllocResult {
    DistTaskSlot slot{DIST_INVALID_SLOT};
    void *heap_ptr{nullptr};
    uint64_t heap_end_offset{0};  // byte offset within the selected ring's heap
    int32_t ring_idx{0};
};

class DistRing {
public:
    DistRing() = default;
    ~DistRing();

    DistRing(const DistRing &) = delete;
    DistRing &operator=(const DistRing &) = delete;

    // Initialise `DIST_MAX_RING_DEPTH` heap rings, each of `heap_bytes`
    // (MAP_SHARED | MAP_ANONYMOUS). Total VA reservation is
    //   DIST_MAX_RING_DEPTH * heap_bytes.
    // `heap_bytes == 0` disables all heaps — `alloc(0, …)` still hands out
    // slots but any `alloc(bytes>0, …)` throws. `timeout_ms == 0` selects the
    // default. Must be called before any fork if the heaps are to be
    // inherited by children.
    void init(uint64_t heap_bytes = DIST_DEFAULT_HEAP_RING_SIZE, uint32_t timeout_ms = DIST_ALLOC_TIMEOUT_MS);

    // Allocate a slot and, if `bytes > 0`, a heap slab from the ring chosen
    // by `scope_depth` (L2-style 0-based: 0 is the outermost scope). The
    // slot's `ring_idx` / `ring_slot_idx` are stamped into the slot state
    // before this call returns.
    //
    // Blocks on the selected ring's cv; throws `std::runtime_error` on
    // timeout. Returns the sentinel `{DIST_INVALID_SLOT, nullptr, 0, 0}` on
    // `shutdown()`. `bytes` is rounded up to `DIST_HEAP_ALIGN`. Passing `0`
    // skips the heap bump entirely (slot-only allocation).
    DistAllocResult alloc(uint64_t bytes = 0, int32_t scope_depth = 0);

    // Release a slot. Reads the slot's `ring_idx` / `ring_slot_idx` to find
    // its ring, marks the slot consumed, and advances that ring's
    // `last_alive_` (and `heap_tail`) as far as FIFO order allows. Other
    // rings are untouched. Safe from any thread.
    void release(DistTaskSlot slot);

    // Pointer to the slot's state. Stable for the slot's lifetime (i.e.
    // until `reset_to_empty()` drops it). Returns nullptr for invalid ids.
    DistTaskSlotState *slot_state(DistTaskSlot slot);

    // Rewind every ring's cursors + released/slot_heap_end vectors and drop
    // all slot states. Requires that no slots are currently live
    // (`active_count() == 0`) — typically called by `DistOrchestrator::drain`
    // right after the active count hits zero.
    void reset_to_empty();

    int32_t active_count() const;
    int32_t next_task_id() const;

    // Per-ring introspection (tests + tooling).
    void *heap_base(int32_t ring_idx) const;
    uint64_t heap_size(int32_t ring_idx) const;
    uint64_t heap_top(int32_t ring_idx) const;
    uint64_t heap_tail(int32_t ring_idx) const;

    void shutdown();

private:
    // One scope-layer heap ring. `DIST_MAX_RING_DEPTH` instances live side by
    // side; the slot deque and `next_task_id_` remain global (parent-heap
    // bookkeeping — see docs/orchestrator.md §5).
    struct HeapRing {
        // mmap region
        void *base{nullptr};
        uint64_t size{0};
        uint64_t top{0};   // next free byte (bump head, can wrap)
        uint64_t tail{0};  // oldest live byte (derived from last_alive_)
        bool mapped{false};

        // Per-ring FIFO ordering — vectors are indexed by a slot's
        // ring_slot_idx (the order it was allocated into this ring).
        std::vector<uint8_t> released;        // 0 = live, 1 = consumed
        std::vector<uint64_t> slot_heap_end;  // byte-offset high-water within this ring
        int32_t last_alive{0};                // FIFO frontier over released/slot_heap_end

        mutable std::mutex mu;
        std::condition_variable cv;

        HeapRing() = default;
        HeapRing(const HeapRing &) = delete;
        HeapRing &operator=(const HeapRing &) = delete;
    };

    uint32_t timeout_ms_{DIST_ALLOC_TIMEOUT_MS};

    // Monotonic across all rings within a run. Reset to 0 by `reset_to_empty`.
    int32_t next_task_id_{0};

    // Parent-heap slot-state pool. push_back never invalidates pointers to
    // existing elements, so slot_state(id) remains stable for every live id.
    std::deque<std::unique_ptr<DistTaskSlotState>> slot_states_;

    // Guards `next_task_id_` and `slot_states_`. Taken briefly during
    // `alloc()` (between picking the task id and pushing the state) and by
    // `slot_state()` / `reset_to_empty()` / `next_task_id()` readers. Not
    // held during the per-ring back-pressure wait — each HeapRing has its
    // own mu / cv for that.
    mutable std::mutex slots_mu_;

    // The 4 scope-layer heap rings (Strict-1).
    std::array<HeapRing, DIST_MAX_RING_DEPTH> rings_;

    // Process-wide shutdown flag (atomic so every ring's waiter sees it
    // without holding that ring's mu_).
    std::atomic<bool> shutdown_{false};

    // Helpers — `try_bump_ring_heap_locked` runs under `ring.mu`;
    // `advance_last_alive_locked` runs under `ring.mu`.
    bool try_bump_ring_heap_locked(HeapRing &ring, uint64_t aligned_bytes, void *&out_ptr, uint64_t &out_end);
    void advance_last_alive_locked(HeapRing &ring);

    // Ring-index validation for the public introspection accessors.
    const HeapRing &ring_at(int32_t ring_idx) const;
};
