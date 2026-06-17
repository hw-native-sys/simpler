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

// Block notification interval (in spin counts)
#define PTO2_BLOCK_NOTIFY_INTERVAL 10000
// Alloc spin limit - after this, report deadlock and exit
#define PTO2_ALLOC_SPIN_LIMIT 100000

// Dep pool spin limit - if exceeded, dep pool capacity too small for workload
#define PTO2_DEP_POOL_SPIN_LIMIT 100000

inline void latch_pool_error(std::atomic<int32_t> *error_code_ptr, int32_t error_code)
{
    if (error_code_ptr == nullptr) return;
    int32_t expected = PTO2_ERROR_NONE;
    error_code_ptr->compare_exchange_strong(expected, error_code, std::memory_order_acq_rel);
}

class PTO2TaskAllocator
{
public:
    void init(PTO2TaskDescriptor *descriptors, int32_t window_size, std::atomic<int32_t> *current_index_ptr, std::atomic<int32_t> *last_alive_ptr, void *heap_base, uint64_t heap_size, std::atomic<int32_t> *error_code_ptr, int32_t initial_local_task_id = 0)
    {
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

    PTO2TaskAllocResult alloc(int32_t output_size)
    {
        uint64_t aligned_size = output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;

        int spin_count = 0;
        int32_t prev_last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        int32_t last_alive = prev_last_alive;
        update_heap_tail(last_alive);
        bool blocked_on_heap = false;

        while (true)
        {
            // Check both resources; commit only if both available
            if (local_task_id_ - last_alive + 1 < window_size_)
            {
                void *heap_ptr = try_bump_heap(aligned_size);
                if (heap_ptr)
                {
                    int32_t task_id = commit_task();
                    return {task_id, task_id & window_mask_, heap_ptr, static_cast<char *>(heap_ptr) + aligned_size};
                }
                blocked_on_heap = true;
            }
            else
            {
                blocked_on_heap = false;
            }

            // Spin: wait for scheduler to advance last_task_alive
            spin_count++;
            last_alive = last_alive_ptr_->load(std::memory_order_acquire);
            update_heap_tail(last_alive);
            if (last_alive > prev_last_alive)
            {
                spin_count = 0;
                prev_last_alive = last_alive;
            }
            else
            {
                if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0)
                {}
                if (spin_count >= PTO2_ALLOC_SPIN_LIMIT)
                {
                    report_deadlock(blocked_on_heap);
                    return {-1, -1, nullptr, nullptr};
                }
            }
            SPIN_WAIT_HINT();
        }
    }

    int32_t active_count() const
    {
        int32_t last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        return local_task_id_ - last_alive;
    }

    // Task ring start/end: tail = oldest live task (last_task_alive), head =
    // next task id to allocate. head - tail == active_count().
    int32_t task_tail() const
    {
        return last_alive_ptr_->load(std::memory_order_acquire);
    }
    int32_t task_head() const
    {
        return local_task_id_;
    }

    int32_t window_size() const
    {
        return window_size_;
    }

    uint64_t heap_available() const
    {
        uint64_t tail = heap_tail_;
        if (heap_top_ >= tail)
        {
            uint64_t at_end = heap_size_ - heap_top_;
            uint64_t at_begin = tail;
            return at_end > at_begin ? at_end : at_begin;
        }
        return tail - heap_top_;
    }

    uint64_t heap_top() const
    {
        return heap_top_;
    }
    // Heap ring start: reclaim pointer (oldest byte still live). heap_top() is
    // the end (next allocation). heap_top - heap_tail == heap_used_bytes().
    uint64_t heap_tail() const
    {
        return heap_tail_;
    }
    uint64_t heap_capacity() const
    {
        return heap_size_;
    }
    uint64_t heap_used_bytes() const
    {
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

    int32_t commit_task()
    {
        int32_t task_id = local_task_id_++;
        current_index_ptr_->store(local_task_id_, std::memory_order_release);
        return task_id;
    }

    void update_heap_tail(int32_t last_alive)
    {
        if (last_alive <= last_alive_seen_) return;
        last_alive_seen_ = last_alive;

        PTO2TaskDescriptor &desc = descriptors_[(last_alive - 1) & window_mask_];
        heap_tail_ = static_cast<uint64_t>(static_cast<char *>(desc.packed_buffer_end) - static_cast<char *>(heap_base_));
    }

    void *try_bump_heap(uint64_t alloc_size)
    {
        uint64_t top = heap_top_;
        if (alloc_size == 0) return static_cast<char *>(heap_base_) + top;
        uint64_t tail = heap_tail_;
        void *result;

        if (top >= tail)
        {
            uint64_t space_at_end = heap_size_ - top;
            if (space_at_end >= alloc_size)
            {
                result = static_cast<char *>(heap_base_) + top;
                heap_top_ = top + alloc_size;
            }
            else if (tail > alloc_size)
            {
                result = heap_base_;
                heap_top_ = alloc_size;
            }
            else
            {
                return nullptr;
            }
        }
        else if (tail - top > alloc_size)
        {
            result = static_cast<char *>(heap_base_) + top;
            heap_top_ = top + alloc_size;
        }
        else
        {
            return nullptr;
        }

        return result;
    }

    void report_deadlock(bool heap_blocked)
    {
        if (error_code_ptr_)
        {
            int32_t code = heap_blocked ? PTO2_ERROR_HEAP_RING_DEADLOCK : PTO2_ERROR_FLOW_CONTROL_DEADLOCK;
            error_code_ptr_->store(code, std::memory_order_release);
        }
    }
};

struct PTO2RingSet
{
    PTO2TaskAllocator task_allocator;
};

#endif  // PTO_RING_BUFFER_H
