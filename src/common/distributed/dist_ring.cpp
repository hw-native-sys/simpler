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

#include "dist_ring.h"

#include <sys/mman.h>

#include <chrono>
#include <cstring>
#include <stdexcept>

DistRing::~DistRing() {
    if (heap_mapped_ && heap_base_) {
        munmap(heap_base_, heap_size_);
        heap_base_ = nullptr;
        heap_mapped_ = false;
    }
}

void DistRing::init(int32_t window_size, uint64_t heap_bytes, uint32_t timeout_ms) {
    if (window_size <= 0 || (window_size & (window_size - 1)) != 0) {
        throw std::invalid_argument("DistRing window_size must be a positive power of 2");
    }
    if (heap_mapped_) {
        throw std::logic_error("DistRing::init called twice");
    }

    window_size_ = window_size;
    window_mask_ = window_size - 1;
    timeout_ms_ = timeout_ms == 0 ? DIST_ALLOC_TIMEOUT_MS : timeout_ms;

    next_task_id_ = 0;
    last_alive_ = 0;
    heap_top_ = 0;
    heap_tail_ = 0;
    shutdown_ = false;

    released_.assign(static_cast<size_t>(window_size_), 0);
    slot_heap_end_.assign(static_cast<size_t>(window_size_), 0);

    if (heap_bytes > 0) {
        heap_size_ = heap_bytes;
        heap_base_ = mmap(nullptr, heap_size_, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
        if (heap_base_ == MAP_FAILED) {
            heap_base_ = nullptr;
            heap_size_ = 0;
            throw std::runtime_error("DistRing: heap mmap failed");
        }
        heap_mapped_ = true;
    } else {
        heap_base_ = nullptr;
        heap_size_ = 0;
    }
}

// ---------------------------------------------------------------------------
// alloc — atomic {slot, heap_ptr} under a single mutex
// ---------------------------------------------------------------------------

DistAllocResult DistRing::alloc(uint64_t bytes) {
    if (bytes > 0 && heap_size_ == 0) {
        throw std::runtime_error("DistRing: heap disabled (heap_bytes=0) but alloc(bytes>0) requested");
    }
    uint64_t aligned = bytes > 0 ? dist_align_up(bytes, DIST_HEAP_ALIGN) : 0;
    if (aligned > heap_size_) {
        throw std::runtime_error("DistRing: requested allocation exceeds heap size");
    }

    std::unique_lock<std::mutex> lk(mu_);

    auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);

    void *heap_ptr = nullptr;
    uint64_t heap_end = heap_top_;
    while (true) {
        if (shutdown_) return DistAllocResult{DIST_INVALID_SLOT, nullptr, 0};

        if (has_window_space_locked()) {
            if (aligned == 0) {
                heap_ptr = nullptr;
                heap_end = heap_top_;
                break;
            }
            if (try_bump_heap_locked(aligned, heap_ptr, heap_end)) {
                break;
            }
        }

        // Wait for a release to advance last_alive_ (and heap_tail_) or for
        // shutdown. Treat the timeout as a deadlock signal so Python callers
        // can enlarge the heap instead of stalling forever.
        if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
            if (shutdown_) return DistAllocResult{DIST_INVALID_SLOT, nullptr, 0};
            int32_t active = next_task_id_ - last_alive_;
            throw std::runtime_error(
                aligned > heap_size_ / 2 ?
                    "DistRing: heap exhausted (timed out waiting). Increase heap_ring_size on Worker." :
                    (active >= window_size_ ? "DistRing: task window full (timed out waiting). "
                                              "Either the DAG is too wide or the scheduler has stalled." :
                                              "DistRing: timed out waiting for reclamation.")
            );
        }
    }

    int32_t task_id = next_task_id_++;
    DistTaskSlot slot = task_id & window_mask_;
    released_[static_cast<size_t>(slot)] = 0;
    slot_heap_end_[static_cast<size_t>(slot)] = heap_end;
    return DistAllocResult{slot, heap_ptr, heap_end};
}

// ---------------------------------------------------------------------------
// release — mark consumed and FIFO-advance last_alive_
// ---------------------------------------------------------------------------

void DistRing::release(DistTaskSlot slot) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (slot < 0 || slot >= window_size_) return;
        released_[static_cast<size_t>(slot)] = 1;
        advance_last_alive_locked();
    }
    cv_.notify_all();
}

// ---------------------------------------------------------------------------
// Queries & shutdown
// ---------------------------------------------------------------------------

int32_t DistRing::active_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return next_task_id_ - last_alive_;
}

void DistRing::shutdown() {
    {
        std::lock_guard<std::mutex> lk(mu_);
        shutdown_ = true;
    }
    cv_.notify_all();
}

// ---------------------------------------------------------------------------
// Internal helpers (all called under mu_)
// ---------------------------------------------------------------------------

bool DistRing::has_window_space_locked() const { return (next_task_id_ - last_alive_) < window_size_; }

bool DistRing::try_bump_heap_locked(uint64_t aligned, void *&out_ptr, uint64_t &out_end) {
    uint64_t top = heap_top_;
    uint64_t tail = heap_tail_;

    // Case 1: heap fully live forward (top >= tail). Space either after top
    // to the end of the region, or (after wrap) from 0 to tail-1.
    if (top >= tail) {
        uint64_t at_end = heap_size_ - top;
        if (at_end >= aligned) {
            out_ptr = static_cast<char *>(heap_base_) + top;
            heap_top_ = top + aligned;
            out_end = heap_top_;
            return true;
        }
        // Wrap only when there is real space at the start. Must be strictly >,
        // not ==: leaving a single byte gap prevents top==tail being ambiguous
        // between "full" and "empty".
        if (tail > aligned) {
            out_ptr = heap_base_;
            heap_top_ = aligned;
            out_end = heap_top_;
            return true;
        }
        return false;
    }

    // Case 2: wrapped (top < tail). Allocate in the gap only.
    if (tail - top > aligned) {
        out_ptr = static_cast<char *>(heap_base_) + top;
        heap_top_ = top + aligned;
        out_end = heap_top_;
        return true;
    }
    return false;
}

void DistRing::advance_last_alive_locked() {
    // Advance last_alive_ while the next-oldest task is already released.
    // Reset the released bit as we cross it so the slot can be reused without
    // leaking a stale "consumed" flag into the next generation.
    while (last_alive_ < next_task_id_) {
        DistTaskSlot la_slot = last_alive_ & window_mask_;
        if (released_[static_cast<size_t>(la_slot)] == 0) break;

        uint64_t end = slot_heap_end_[static_cast<size_t>(la_slot)];
        released_[static_cast<size_t>(la_slot)] = 0;
        last_alive_++;
        heap_tail_ = end;
    }
}
