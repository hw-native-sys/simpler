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
 * A single structure owns three correlated, per-task resources:
 *
 *   1. A monotonic task id (`next_task_id_`), allocated by the Orchestrator.
 *      Unlike L2's `PTO2TaskAllocator` the id is NOT masked into a fixed-size
 *      window — slot state lives in parent-process heap (never crossed into
 *      child workers), so a ring index buys us nothing at L3 (see the plan's
 *      L2 Consistency Audit, allowed exception #6).
 *   2. A shared-memory heap slab (bump-allocated under one mutex, FIFO
 *      reclaimed via `last_alive_`). This part still mirrors L2 (Strict-2):
 *      the heap must be `mmap(MAP_SHARED)` and forked into child workers,
 *      which forces a pre-sized region.
 *   3. The per-task scheduling state (`DistTaskSlotState`). Stored in a
 *      `std::deque<std::unique_ptr<...>>` so push_back never invalidates
 *      pointers, and destruction happens only at `reset_to_empty()` /
 *      process teardown, giving callers stable references to a slot until
 *      it is consumed.
 *
 * Back-pressure: only the heap can be full. `alloc(bytes)` spin-waits on a
 * cv; if no progress is made for `timeout_ms` it throws `std::runtime_error`
 * so Python callers see an exception rather than a silent deadlock.
 *
 * Lifecycle:
 *
 *   Worker.run() → orch submits N tasks → each submit calls
 *   `ring.alloc()` (task id allocated, slot state constructed) → scheduler
 *   dispatches → workers run → on_consumed calls `ring.release(id)` which
 *   marks the slot consumed and advances `last_alive_` FIFO-wise → drain
 *   waits until `active_count() == 0` → `ring.reset_to_empty()` resets
 *   counters and drops all slot states so the next run starts fresh.
 *
 *   Memory footprint per Worker.run() is bounded by the peak alive task
 *   count; no state accumulates across runs.
 */

#pragma once

#include <condition_variable>
#include <cstdint>
#include <deque>
#include <memory>
#include <mutex>
#include <vector>

#include "dist_types.h"

// User-facing output alignment (Strict-3; matches L2 PTO2_PACKED_OUTPUT_ALIGN).
static constexpr uint64_t DIST_HEAP_ALIGN = 1024;

// Default heap ring size for L3+ Worker: 1 GiB, overridable per-Worker.
static constexpr uint64_t DIST_DEFAULT_HEAP_RING_SIZE = 1ULL << 30;

// Default back-pressure timeout (ms). Surfaces as std::runtime_error when the
// allocator makes no progress for this long — acts as a deadlock detector.
static constexpr uint32_t DIST_ALLOC_TIMEOUT_MS = 10000;

// Align an unsigned value up to the next multiple of `align` (must be power of 2).
inline uint64_t dist_align_up(uint64_t v, uint64_t align) { return (v + align - 1) & ~(align - 1); }

struct DistAllocResult {
    DistTaskSlot slot{DIST_INVALID_SLOT};
    void *heap_ptr{nullptr};
    uint64_t heap_end_offset{0};  // absolute byte offset in the heap region
};

class DistRing {
public:
    DistRing() = default;
    ~DistRing();

    DistRing(const DistRing &) = delete;
    DistRing &operator=(const DistRing &) = delete;

    // Initialise. `heap_bytes == 0` disables the heap — `alloc(0)` still
    // hands out slots, but any `alloc(bytes>0)` throws. `timeout_ms == 0`
    // selects the default. Must be called before any fork if the heap is
    // to be inherited by children.
    void init(uint64_t heap_bytes = DIST_DEFAULT_HEAP_RING_SIZE, uint32_t timeout_ms = DIST_ALLOC_TIMEOUT_MS);

    // Allocate a slot (and, if `bytes > 0`, a heap slab). Blocks on the
    // heap cv; throws `std::runtime_error` on timeout. Returns the sentinel
    // `{DIST_INVALID_SLOT, nullptr, 0}` on `shutdown()`.
    //
    // `bytes` is rounded up to `DIST_HEAP_ALIGN`. Passing `0` skips the heap
    // bump entirely (slot-only allocation).
    DistAllocResult alloc(uint64_t bytes = 0);

    // Release a slot. Marks the slot consumed; advances `last_alive_` (and
    // `heap_tail_`) as far as the FIFO ordering allows. Safe to call from
    // any thread; safe under concurrent `alloc()`.
    void release(DistTaskSlot slot);

    // Pointer to the slot's state. Stable for the slot's lifetime (i.e.
    // until `reset_to_empty()` drops it). Returns nullptr for invalid ids.
    //
    // Thread safety: the pointer refers to heap-allocated storage held by a
    // `std::deque<std::unique_ptr<...>>`. `std::deque::push_back` never
    // invalidates pointers to existing elements, so once a slot id has been
    // handed out by `alloc()`, `slot_state(id)` stays valid across concurrent
    // allocs. The caller is responsible for not using the pointer across a
    // matching `reset_to_empty()` call.
    DistTaskSlotState *slot_state(DistTaskSlot slot);

    // Clear all per-task state and return to `next_task_id_ = last_alive_ = 0`.
    // Requires that no slots are currently live (`active_count() == 0`) —
    // typically called by `DistOrchestrator::drain()` right after the active
    // count hits zero.
    void reset_to_empty();

    int32_t active_count() const;
    int32_t next_task_id() const;
    void *heap_base() const { return heap_base_; }
    uint64_t heap_size() const { return heap_size_; }

    void shutdown();

private:
    uint32_t timeout_ms_{DIST_ALLOC_TIMEOUT_MS};

    // Monotonic within a run. Reset to 0 by `reset_to_empty()`.
    int32_t next_task_id_{0};

    // FIFO consumption frontier. `[last_alive_, next_task_id_)` are live.
    int32_t last_alive_{0};

    // Per-slot bookkeeping, indexed directly by task id (0..next_task_id_).
    // All three grow together under `mu_`; no erase while the run is live.
    std::vector<uint8_t> released_;        // 0 = live, 1 = consumed
    std::vector<uint64_t> slot_heap_end_;  // byte-offset high-water of each slot's allocation
    std::deque<std::unique_ptr<DistTaskSlotState>> slot_states_;

    // Heap region.
    void *heap_base_{nullptr};
    uint64_t heap_size_{0};
    uint64_t heap_top_{0};   // next free byte (bump head, can wrap)
    uint64_t heap_tail_{0};  // oldest live byte (derived from last_alive_)

    mutable std::mutex mu_;
    std::condition_variable cv_;
    bool shutdown_{false};
    bool heap_mapped_{false};

    // Helpers — all called under mu_.
    bool try_bump_heap_locked(uint64_t aligned_bytes, void *&out_ptr, uint64_t &out_end);
    void advance_last_alive_locked();
};
