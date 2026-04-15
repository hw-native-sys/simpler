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
 * DistRing — unified task-slot + heap-buffer allocator (Strict-2).
 *
 * Single allocator guarding both resources under one mutex, mirroring L2's
 * `PTO2TaskAllocator`. The orchestrator calls `alloc(bytes)` to claim a slot
 * plus a contiguous heap slab in one atomic step; there is no partial-failure
 * rollback. `release(slot)` advances a FIFO `last_alive` pointer, which
 * implicitly reclaims heap space belonging to the trailing range of slots.
 *
 * Heap memory is a single `mmap(MAP_SHARED | MAP_ANONYMOUS)` region taken
 * at construction time, before any fork, so forked child workers see the
 * same bytes at the same virtual address.
 *
 * Output-buffer alignment is 1024 B (matches L2's `PTO2_PACKED_OUTPUT_ALIGN`).
 *
 * Back-pressure: `alloc()` spin-waits with cv when either the slot window or
 * the heap is exhausted. If no progress is made for `timeout_ms` the call
 * throws `std::runtime_error` so Python users can catch it and grow the heap
 * instead of deadlocking.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
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
    void init(
        int32_t window_size = DIST_TASK_WINDOW_SIZE, uint64_t heap_bytes = DIST_DEFAULT_HEAP_RING_SIZE,
        uint32_t timeout_ms = DIST_ALLOC_TIMEOUT_MS
    );

    // Allocate a slot (and, if `bytes > 0`, a heap slab). Blocks with
    // back-pressure; throws `std::runtime_error` on timeout. Returns the
    // sentinel `{DIST_INVALID_SLOT, nullptr, 0}` on `shutdown()`.
    //
    // `bytes` is rounded up to `DIST_HEAP_ALIGN`. Passing `0` skips the heap
    // bump entirely (slot-only allocation).
    DistAllocResult alloc(uint64_t bytes = 0);

    // Release a slot. Marks the slot consumed; advances `last_alive_` (and
    // `heap_tail_`) as far as the FIFO ordering allows.
    void release(DistTaskSlot slot);

    int32_t window_size() const { return window_size_; }
    int32_t active_count() const;
    void *heap_base() const { return heap_base_; }
    uint64_t heap_size() const { return heap_size_; }

    void shutdown();

private:
    int32_t window_size_{DIST_TASK_WINDOW_SIZE};
    int32_t window_mask_{DIST_TASK_WINDOW_SIZE - 1};
    uint32_t timeout_ms_{DIST_ALLOC_TIMEOUT_MS};

    // Orch-owned counter (single-writer from alloc(), so no atomic needed —
    // it's still read under mu_).
    int32_t next_task_id_{0};

    // FIFO consumption frontier. `[last_alive_, next_task_id_)` are live.
    int32_t last_alive_{0};

    // Per-slot bookkeeping (sized to window_size_ after init).
    std::vector<uint8_t> released_;        // 0 = live, 1 = consumed
    std::vector<uint64_t> slot_heap_end_;  // byte-offset high-water of each slot's allocation

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
    bool has_window_space_locked() const;
    bool try_bump_heap_locked(uint64_t aligned_bytes, void *&out_ptr, uint64_t &out_end);
    void advance_last_alive_locked();
};
