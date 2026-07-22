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

#ifndef PTO_RING_BUFFER_H
#define PTO_RING_BUFFER_H

#include <algorithm>
#include <inttypes.h>
#include <type_traits>

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "aicpu/device_time.h"       // get_sys_cnt_aicpu (deadlock wall-clock backstop)
#include "common/platform_config.h"  // PLATFORM_PROF_SYS_CNT_FREQ (deadlock wall-clock)

// Heap/task deadlock backstop, expressed as an ABSOLUTE wall-clock budget of
// no-reclaim-progress (last_task_alive unchanged), NOT a spin count. The reclaim
// tail is a circular FIFO, so the oldest live task pins the whole live-set until
// its last consumer completes; when the heap sits near-full a single slow-to-
// complete task stalls reclaim for tens of ms while the scheduler catches up. A
// spin-count limit maps to wildly different wall-times under contention and
// false-trips that recoverable transient; a wall-clock deadline is stable across
// chips/load. 500 ms comfortably covers the transient (tens of ms) yet trips a
// genuine deadlock well before the 45 s op-execute timeout. Mirrors upstream/main.
#define PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES (PLATFORM_PROF_SYS_CNT_FREQ / 2)  // 500 ms

// Check the deadlock wall clock only once per (mask+1) stalled spins: get_sys_cnt_aicpu()
// is an MMIO read, so polling it every spin would dominate the back-pressure loop.
// Must be 2^k - 1 (used as an AND mask).
static constexpr int PTO2_ALLOC_DEADLOCK_POLL_MASK = 1023;  // ~every 1024 spins

// Dep pool spin limit - if exceeded, dep pool capacity too small for workload
#define PTO2_DEP_POOL_SPIN_LIMIT 100000

class PTO2TaskAllocator {
public:
    void init(
        PTO2TaskDescriptor *descriptors, int32_t window_size, std::atomic<int32_t> *current_index_ptr,
        std::atomic<int32_t> *last_alive_ptr, void *heap_base, uint64_t heap_size, std::atomic<int32_t> *error_code_ptr,
        int32_t initial_local_task_id = 0
    ) {
        descriptors_ = descriptors;
        window_size_ = window_size;
        window_mask_ = window_size - 1;
        current_index_ptr_ = current_index_ptr;
        last_alive_ptr_ = last_alive_ptr;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        error_code_ptr_ = error_code_ptr;
        local_task_id_ = initial_local_task_id;
        heap_top_ = 0;
        heap_tail_ = 0;
        last_alive_seen_ = 0;
    }

    // Surgical reset for arena reuse: just the per-run counters. The
    // arena-internal pointers (descriptors_, current_index_ptr_, etc.) are
    // still valid, since wire_arena_pointers was called before this on the
    // AICPU side.
    void reset_for_reuse() {
        local_task_id_ = 0;
        heap_top_ = 0;
        heap_tail_ = 0;
        last_alive_seen_ = 0;
    }

    PTO2TaskAllocResult alloc(int32_t output_size) {
        uint64_t aligned_size =
            output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;

        int32_t prev_last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        int32_t last_alive = prev_last_alive;
        update_heap_tail(last_alive);
        bool blocked_on_heap = false;
        uint64_t block_ts = 0;      // wall-clock anchor at the first no-progress spin
        bool block_timing = false;  // false until reclaim stalls (last_alive frozen)
        int poll_count = 0;         // rate-limits the MMIO-costly sys-counter read

        while (true) {
            // Check both resources; commit only if both available
            if (local_task_id_ - last_alive + 1 < window_size_) {
                void *heap_ptr = try_bump_heap(aligned_size);
                if (heap_ptr) {
                    int32_t task_id = commit_task();
                    return {task_id, task_id & window_mask_, heap_ptr, static_cast<char *>(heap_ptr) + aligned_size};
                }
                blocked_on_heap = true;
            } else {
                blocked_on_heap = false;
            }

            // Spin: wait for scheduler to advance last_task_alive
            last_alive = last_alive_ptr_->load(std::memory_order_acquire);
            update_heap_tail(last_alive);
            if (last_alive > prev_last_alive) {
                // Reclaim advanced -> productive back-pressure, not a deadlock.
                prev_last_alive = last_alive;
                block_timing = false;
            } else {
                // No reclaim progress. Anchor the wall clock on the first such
                // spin, then declare deadlock only after an ABSOLUTE time budget
                // elapses (contention-independent). Read the sys counter once
                // per 1024 spins to amortize its MMIO cost.
                if (!block_timing) {
                    block_ts = get_sys_cnt_aicpu();
                    block_timing = true;
                    poll_count = 0;
                } else if (((++poll_count) & PTO2_ALLOC_DEADLOCK_POLL_MASK) == 0) {
                    if (get_sys_cnt_aicpu() - block_ts > PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES) {
                        report_deadlock(blocked_on_heap);
                        return {-1, -1, nullptr, nullptr};
                    }
                }
            }
            SPIN_WAIT_HINT();
        }
    }

    int32_t active_count() const {
        int32_t last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        return local_task_id_ - last_alive;
    }

    // Task ring start/end: tail = oldest live task (last_task_alive), head =
    // next task id to allocate. head - tail == active_count().
    int32_t task_tail() const { return last_alive_ptr_->load(std::memory_order_acquire); }
    int32_t task_head() const { return local_task_id_; }

    int32_t window_size() const { return window_size_; }

    uint64_t heap_available() const {
        uint64_t tail = heap_tail_;
        if (heap_top_ >= tail) {
            uint64_t at_end = heap_size_ - heap_top_;
            uint64_t at_begin = tail;
            return at_end > at_begin ? at_end : at_begin;
        }
        return tail - heap_top_;
    }

    uint64_t heap_top() const { return heap_top_; }
    // Heap ring start: reclaim pointer (oldest byte still live). heap_top() is
    // the end (next allocation). heap_top - heap_tail == heap_used_bytes().
    uint64_t heap_tail() const { return heap_tail_; }
    uint64_t heap_capacity() const { return heap_size_; }
    uint64_t heap_used_bytes() const {
        if (heap_size_ == 0) return 0;
        return (heap_top_ + heap_size_ - heap_tail_) % heap_size_;
    }

private:
    // --- Task Ring ---
    PTO2TaskDescriptor *descriptors_ = nullptr;
    int32_t window_size_ = 0;
    int32_t window_mask_ = 0;
    std::atomic<int32_t> *current_index_ptr_ = nullptr;
    std::atomic<int32_t> *last_alive_ptr_ = nullptr;

    // --- Heap ---
    void *heap_base_ = nullptr;
    uint64_t heap_size_ = 0;

    // --- Local state (single-writer, no atomics needed) ---
    int32_t local_task_id_ = 0;    // Next task ID to allocate
    uint64_t heap_top_ = 0;        // Current heap allocation pointer
    uint64_t heap_tail_ = 0;       // Heap reclamation pointer (derived from consumed tasks)
    int32_t last_alive_seen_ = 0;  // last_task_alive at last heap_tail derivation

    // --- Shared ---
    std::atomic<int32_t> *error_code_ptr_ = nullptr;

    int32_t commit_task() {
        int32_t task_id = local_task_id_++;
        current_index_ptr_->store(local_task_id_, std::memory_order_release);
        return task_id;
    }

    void update_heap_tail(int32_t last_alive) {
        if (last_alive <= last_alive_seen_) return;
        last_alive_seen_ = last_alive;

        PTO2TaskDescriptor &desc = descriptors_[(last_alive - 1) & window_mask_];
        heap_tail_ =
            static_cast<uint64_t>(static_cast<char *>(desc.packed_buffer_end) - static_cast<char *>(heap_base_));
    }

    void *try_bump_heap(uint64_t alloc_size) {
        uint64_t top = heap_top_;
        if (alloc_size == 0) return static_cast<char *>(heap_base_) + top;
        uint64_t tail = heap_tail_;
        void *result;

        if (top >= tail) {
            uint64_t space_at_end = heap_size_ - top;
            if (space_at_end >= alloc_size) {
                result = static_cast<char *>(heap_base_) + top;
                heap_top_ = top + alloc_size;
            } else if (tail > alloc_size) {
                result = heap_base_;
                heap_top_ = alloc_size;
            } else {
                return nullptr;
            }
        } else if (tail - top > alloc_size) {
            result = static_cast<char *>(heap_base_) + top;
            heap_top_ = top + alloc_size;
        } else {
            return nullptr;
        }

        return result;
    }

    void report_deadlock(bool heap_blocked) {
        if (error_code_ptr_) {
            int32_t code = heap_blocked ? PTO2_ERROR_HEAP_RING_DEADLOCK : PTO2_ERROR_FLOW_CONTROL_DEADLOCK;
            error_code_ptr_->store(code, std::memory_order_release);
        }
    }
};

struct PTO2RingSet {
    PTO2TaskAllocator task_allocator;
};

#endif  // PTO_RING_BUFFER_H
