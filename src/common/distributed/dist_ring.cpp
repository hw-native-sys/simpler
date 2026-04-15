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
#include <stdexcept>

DistRing::~DistRing() {
    if (heap_mapped_ && heap_base_) {
        munmap(heap_base_, heap_size_);
        heap_base_ = nullptr;
        heap_mapped_ = false;
    }
}

void DistRing::init(uint64_t heap_bytes, uint32_t timeout_ms) {
    if (heap_mapped_) {
        throw std::logic_error("DistRing::init called twice");
    }

    timeout_ms_ = timeout_ms == 0 ? DIST_ALLOC_TIMEOUT_MS : timeout_ms;

    next_task_id_ = 0;
    last_alive_ = 0;
    heap_top_ = 0;
    heap_tail_ = 0;
    shutdown_ = false;

    released_.clear();
    slot_heap_end_.clear();
    slot_states_.clear();

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
// alloc — slot + heap under a single mutex
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

        if (aligned == 0) {
            heap_ptr = nullptr;
            heap_end = heap_top_;
            break;
        }
        if (try_bump_heap_locked(aligned, heap_ptr, heap_end)) {
            break;
        }

        // Heap full. Wait for a release to advance last_alive_ / heap_tail_,
        // or for shutdown. Timing out surfaces as a Python exception so the
        // user can enlarge `heap_ring_size` instead of deadlocking.
        if (cv_.wait_until(lk, deadline) == std::cv_status::timeout) {
            if (shutdown_) return DistAllocResult{DIST_INVALID_SLOT, nullptr, 0};
            throw std::runtime_error(
                "DistRing: heap exhausted (timed out waiting). Increase heap_ring_size on Worker."
            );
        }
    }

    int32_t task_id = next_task_id_++;
    released_.push_back(0);
    slot_heap_end_.push_back(heap_end);
    slot_states_.emplace_back(std::make_unique<DistTaskSlotState>());
    return DistAllocResult{task_id, heap_ptr, heap_end};
}

// ---------------------------------------------------------------------------
// release — mark consumed and FIFO-advance last_alive_
// ---------------------------------------------------------------------------

void DistRing::release(DistTaskSlot slot) {
    {
        std::lock_guard<std::mutex> lk(mu_);
        if (slot < 0 || slot >= next_task_id_) return;
        if (released_[static_cast<size_t>(slot)] != 0) return;  // idempotent
        released_[static_cast<size_t>(slot)] = 1;
        advance_last_alive_locked();
    }
    cv_.notify_all();
}

// ---------------------------------------------------------------------------
// slot_state accessor — pointer-stable until reset_to_empty()
// ---------------------------------------------------------------------------

DistTaskSlotState *DistRing::slot_state(DistTaskSlot slot) {
    std::lock_guard<std::mutex> lk(mu_);
    if (slot < 0 || slot >= static_cast<int32_t>(slot_states_.size())) return nullptr;
    return slot_states_[static_cast<size_t>(slot)].get();
}

// ---------------------------------------------------------------------------
// reset_to_empty — drop all per-task state, return counters to zero
// ---------------------------------------------------------------------------

void DistRing::reset_to_empty() {
    std::lock_guard<std::mutex> lk(mu_);
    if (last_alive_ != next_task_id_) {
        throw std::logic_error(
            "DistRing::reset_to_empty: tasks still live "
            "(last_alive_ != next_task_id_). Did drain() complete?"
        );
    }
    next_task_id_ = 0;
    last_alive_ = 0;
    heap_top_ = 0;
    heap_tail_ = 0;
    released_.clear();
    slot_heap_end_.clear();
    slot_states_.clear();
}

// ---------------------------------------------------------------------------
// Queries & shutdown
// ---------------------------------------------------------------------------

int32_t DistRing::active_count() const {
    std::lock_guard<std::mutex> lk(mu_);
    return next_task_id_ - last_alive_;
}

int32_t DistRing::next_task_id() const {
    std::lock_guard<std::mutex> lk(mu_);
    return next_task_id_;
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
    // Slot state and heap_end entries stay in their vectors — memory is
    // only reclaimed by reset_to_empty() at drain time — so we don't have
    // to worry about invalidating pointers that other threads may still
    // hold to in-flight slots.
    while (last_alive_ < next_task_id_ && released_[static_cast<size_t>(last_alive_)] == 1) {
        heap_tail_ = slot_heap_end_[static_cast<size_t>(last_alive_)];
        last_alive_++;
    }
}
