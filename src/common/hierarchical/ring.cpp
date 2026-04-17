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

#include "ring.h"

#include <sys/mman.h>

#include <chrono>
#include <stdexcept>

Ring::~Ring() {
    for (HeapRing &r : rings_) {
        if (r.mapped && r.base) {
            munmap(r.base, r.size);
            r.base = nullptr;
            r.mapped = false;
        }
    }
}

void Ring::init(uint64_t heap_bytes, uint32_t timeout_ms) {
    for (const HeapRing &r : rings_) {
        if (r.mapped) {
            throw std::logic_error("Ring::init called twice");
        }
    }

    timeout_ms_ = timeout_ms == 0 ? ALLOC_TIMEOUT_MS : timeout_ms;

    {
        std::lock_guard<std::mutex> lk(slots_mu_);
        next_task_id_ = 0;
        slot_states_.clear();
    }
    shutdown_.store(false, std::memory_order_relaxed);

    for (HeapRing &r : rings_) {
        r.top = 0;
        r.tail = 0;
        r.last_alive = 0;
        r.released.clear();
        r.slot_heap_end.clear();

        if (heap_bytes > 0) {
            void *base = mmap(nullptr, heap_bytes, PROT_READ | PROT_WRITE, MAP_SHARED | MAP_ANONYMOUS, -1, 0);
            if (base == MAP_FAILED) {
                // Unwind any rings already mapped so destruction stays clean.
                for (HeapRing &rr : rings_) {
                    if (rr.mapped && rr.base) {
                        munmap(rr.base, rr.size);
                        rr.base = nullptr;
                        rr.size = 0;
                        rr.mapped = false;
                    }
                }
                throw std::runtime_error("Ring: per-ring heap mmap failed");
            }
            r.base = base;
            r.size = heap_bytes;
            r.mapped = true;
        } else {
            r.base = nullptr;
            r.size = 0;
        }
    }
}

// ---------------------------------------------------------------------------
// alloc — picks a ring by scope depth, reserves slot + heap slab
// ---------------------------------------------------------------------------

AllocResult Ring::alloc(uint64_t bytes, int32_t scope_depth) {
    int32_t ring_idx = ring_idx_for_scope(scope_depth);
    HeapRing &ring = rings_[static_cast<size_t>(ring_idx)];

    if (bytes > 0 && ring.size == 0) {
        throw std::runtime_error("Ring: heap disabled (heap_bytes=0) but alloc(bytes>0) requested");
    }
    uint64_t aligned = bytes > 0 ? align_up(bytes, HEAP_ALIGN) : 0;
    if (aligned > ring.size) {
        throw std::runtime_error("Ring: requested allocation exceeds per-ring heap size");
    }

    // --- Phase 1: wait for heap space + reserve in-ring bookkeeping ---
    // Holds only the target ring's mu. Back-pressure on ring A cannot block
    // alloc calls targeting a different ring, and it cannot block readers of
    // `slots_mu_` (`slot_state`, `next_task_id`, or `release` on a slot that
    // already exists in another ring).
    void *heap_ptr = nullptr;
    uint64_t heap_end = 0;
    int32_t ring_slot_idx = 0;
    {
        std::unique_lock<std::mutex> rlk(ring.mu);
        auto deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(timeout_ms_);

        while (true) {
            if (shutdown_.load(std::memory_order_acquire)) {
                return AllocResult{INVALID_SLOT, nullptr, 0, ring_idx};
            }
            if (aligned == 0) {
                heap_ptr = nullptr;
                heap_end = ring.top;
                break;
            }
            if (try_bump_ring_heap_locked(ring, aligned, heap_ptr, heap_end)) {
                break;
            }
            // Heap full on THIS ring. Wait for a release on this ring (other
            // rings stay usable) or a shutdown.
            if (ring.cv.wait_until(rlk, deadline) == std::cv_status::timeout) {
                if (shutdown_.load(std::memory_order_acquire)) {
                    return AllocResult{INVALID_SLOT, nullptr, 0, ring_idx};
                }
                throw std::runtime_error(
                    "Ring: per-ring heap exhausted (timed out waiting). "
                    "Increase heap_ring_size on Worker."
                );
            }
        }

        // Capture the in-ring FIFO position BEFORE releasing ring.mu so
        // concurrent allocs on the same ring get strictly ordered positions.
        ring_slot_idx = static_cast<int32_t>(ring.released.size());
        ring.released.push_back(0);
        ring.slot_heap_end.push_back(heap_end);
    }

    // --- Phase 2: assign a global task id and park the slot state ---
    // Holds only `slots_mu_`. Deliberately separate from Phase 1 so we never
    // nest `ring.mu` inside `slots_mu_` or vice versa (see `reset_to_empty`,
    // which nests them the other way).
    int32_t task_id;
    {
        std::lock_guard<std::mutex> slk(slots_mu_);
        task_id = next_task_id_++;
        slot_states_.emplace_back(std::make_unique<TaskSlotState>());
        auto *s = slot_states_.back().get();
        s->ring_idx = ring_idx;
        s->ring_slot_idx = ring_slot_idx;
    }
    return AllocResult{task_id, heap_ptr, heap_end, ring_idx};
}

// ---------------------------------------------------------------------------
// release — mark consumed in the slot's own ring and FIFO-advance that ring
// ---------------------------------------------------------------------------

void Ring::release(TaskSlot slot) {
    int32_t ring_idx = 0;
    int32_t ring_slot_idx = 0;
    {
        std::lock_guard<std::mutex> slk(slots_mu_);
        if (slot < 0 || slot >= static_cast<int32_t>(slot_states_.size())) return;
        TaskSlotState *s = slot_states_[static_cast<size_t>(slot)].get();
        if (!s) return;
        ring_idx = s->ring_idx;
        ring_slot_idx = s->ring_slot_idx;
    }

    HeapRing &ring = rings_[static_cast<size_t>(ring_idx)];
    {
        std::lock_guard<std::mutex> rlk(ring.mu);
        if (ring_slot_idx < 0 || ring_slot_idx >= static_cast<int32_t>(ring.released.size())) return;
        if (ring.released[static_cast<size_t>(ring_slot_idx)] != 0) return;  // idempotent
        ring.released[static_cast<size_t>(ring_slot_idx)] = 1;
        advance_last_alive_locked(ring);
    }
    ring.cv.notify_all();
}

// ---------------------------------------------------------------------------
// slot_state accessor — pointer-stable until reset_to_empty()
// ---------------------------------------------------------------------------

TaskSlotState *Ring::slot_state(TaskSlot slot) {
    std::lock_guard<std::mutex> slk(slots_mu_);
    if (slot < 0 || slot >= static_cast<int32_t>(slot_states_.size())) return nullptr;
    return slot_states_[static_cast<size_t>(slot)].get();
}

// ---------------------------------------------------------------------------
// reset_to_empty — rewind every ring and drop all slot state
// ---------------------------------------------------------------------------

void Ring::reset_to_empty() {
    std::lock_guard<std::mutex> slk(slots_mu_);

    // Validate: every ring must be fully drained. Checking each ring under
    // its own mu_ is the cheapest way to stay race-free with in-flight
    // releases — we take the locks one at a time (no nesting between rings).
    for (HeapRing &r : rings_) {
        std::lock_guard<std::mutex> rlk(r.mu);
        if (r.last_alive != static_cast<int32_t>(r.released.size())) {
            throw std::logic_error(
                "Ring::reset_to_empty: tasks still live on at least one ring. "
                "Did drain() complete?"
            );
        }
    }

    next_task_id_ = 0;
    slot_states_.clear();

    // Re-take each ring's mu and rewind. Safe to do one at a time — no
    // in-flight caller should be touching the rings at this point
    // (active_count() == 0 was the drain precondition).
    for (HeapRing &r : rings_) {
        std::lock_guard<std::mutex> rlk(r.mu);
        r.top = 0;
        r.tail = 0;
        r.last_alive = 0;
        r.released.clear();
        r.slot_heap_end.clear();
    }
}

// ---------------------------------------------------------------------------
// Queries & shutdown
// ---------------------------------------------------------------------------

int32_t Ring::active_count() const {
    // total_allocated - total_released. Reads the two halves under separate
    // critical sections to avoid nesting locks (reset_to_empty holds
    // slots_mu_ while iterating each ring's mu, so a reader that nested
    // them the other way could deadlock). A concurrent alloc between the
    // reads can make the number off by a couple — acceptable for a
    // monitoring accessor.
    int32_t total_tasks;
    {
        std::lock_guard<std::mutex> slk(slots_mu_);
        total_tasks = next_task_id_;
    }
    int32_t total_released = 0;
    for (const HeapRing &r : rings_) {
        std::lock_guard<std::mutex> rlk(r.mu);
        total_released += r.last_alive;
    }
    return total_tasks - total_released;
}

int32_t Ring::next_task_id() const {
    std::lock_guard<std::mutex> slk(slots_mu_);
    return next_task_id_;
}

const Ring::HeapRing &Ring::ring_at(int32_t ring_idx) const {
    if (ring_idx < 0 || ring_idx >= MAX_RING_DEPTH) {
        throw std::out_of_range("Ring: ring_idx out of range");
    }
    return rings_[static_cast<size_t>(ring_idx)];
}

void *Ring::heap_base(int32_t ring_idx) const { return ring_at(ring_idx).base; }

uint64_t Ring::heap_size(int32_t ring_idx) const { return ring_at(ring_idx).size; }

uint64_t Ring::heap_top(int32_t ring_idx) const {
    const HeapRing &r = ring_at(ring_idx);
    std::lock_guard<std::mutex> rlk(r.mu);
    return r.top;
}

uint64_t Ring::heap_tail(int32_t ring_idx) const {
    const HeapRing &r = ring_at(ring_idx);
    std::lock_guard<std::mutex> rlk(r.mu);
    return r.tail;
}

void Ring::shutdown() {
    shutdown_.store(true, std::memory_order_release);
    for (HeapRing &r : rings_) {
        r.cv.notify_all();
    }
}

// ---------------------------------------------------------------------------
// Internal helpers — all called under the respective ring's mu
// ---------------------------------------------------------------------------

bool Ring::try_bump_ring_heap_locked(HeapRing &r, uint64_t aligned, void *&out_ptr, uint64_t &out_end) {
    uint64_t top = r.top;
    uint64_t tail = r.tail;

    // Case 1: heap fully live forward (top >= tail). Space either after top
    // to the end of the region, or (after wrap) from 0 to tail-1.
    if (top >= tail) {
        uint64_t at_end = r.size - top;
        if (at_end >= aligned) {
            out_ptr = static_cast<char *>(r.base) + top;
            r.top = top + aligned;
            out_end = r.top;
            return true;
        }
        // Wrap only when there is real space at the start. Must be strictly >,
        // not ==: leaving a single byte gap prevents top==tail being ambiguous
        // between "full" and "empty".
        if (tail > aligned) {
            out_ptr = r.base;
            r.top = aligned;
            out_end = r.top;
            return true;
        }
        return false;
    }

    // Case 2: wrapped (top < tail). Allocate in the gap only.
    if (tail - top > aligned) {
        out_ptr = static_cast<char *>(r.base) + top;
        r.top = top + aligned;
        out_end = r.top;
        return true;
    }
    return false;
}

void Ring::advance_last_alive_locked(HeapRing &r) {
    // Walk forward as long as the next-oldest in-ring slot is released.
    // Slot-state entries and heap_end entries stay in their vectors — memory
    // is reclaimed only by reset_to_empty() at drain time — so we never
    // invalidate pointers that other threads may still hold.
    while (r.last_alive < static_cast<int32_t>(r.released.size()) &&
           r.released[static_cast<size_t>(r.last_alive)] == 1) {
        r.tail = r.slot_heap_end[static_cast<size_t>(r.last_alive)];
        r.last_alive++;
    }
}
