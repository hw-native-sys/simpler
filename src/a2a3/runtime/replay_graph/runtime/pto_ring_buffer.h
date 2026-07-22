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

#include <inttypes.h>

#include "common/unified_log.h"
#include "pto_runtime2_types.h"

// Historical filename: replay_graph uses two graph-level bump arenas.
class PTO2TaskAllocator {
public:
    void init(
        int32_t window_size, std::atomic<int32_t> *task_count_ptr, int32_t *task_slot_map, void *heap_base,
        uint64_t heap_size, std::atomic<int32_t> *error_code_ptr, int32_t initial_task_id = 0
    ) {
        window_size_ = window_size;
        task_window_mask_ = window_size - 1;
        slot_arena_size_ = window_size / PTO2_REPLAY_GRAPH_BUFFER_COUNT;
        task_count_ptr_ = task_count_ptr;
        task_slot_map_ = task_slot_map;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        heap_arena_size_ = heap_size / PTO2_REPLAY_GRAPH_BUFFER_COUNT;
        error_code_ptr_ = error_code_ptr;
        next_task_id_ = initial_task_id;
        active_buffer_ = 0;
        for (int32_t i = 0; i < PTO2_REPLAY_GRAPH_BUFFER_COUNT; i++) {
            buffers_[i].slot_base = i * slot_arena_size_;
            buffers_[i].slot_top = 0;
            buffers_[i].heap_base = static_cast<char *>(heap_base_) + static_cast<uint64_t>(i) * heap_arena_size_;
            buffers_[i].heap_top = 0;
        }
    }

    void begin_buffer(int32_t buffer_id) {
        if (buffer_id < 0 || buffer_id >= PTO2_REPLAY_GRAPH_BUFFER_COUNT) {
            report_invalid_buffer(buffer_id);
            return;
        }
        active_buffer_ = buffer_id;
        buffers_[buffer_id].slot_top = 0;
        buffers_[buffer_id].heap_top = 0;
    }

    PTO2TaskAllocResult alloc(int32_t output_size) {
        uint64_t aligned = output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;
        BufferArena &arena = buffers_[active_buffer_];
        if (arena.slot_top >= slot_arena_size_) {
            report_task_overflow();
            return {-1, -1, nullptr, nullptr};
        }
        if (arena.heap_top + aligned > heap_arena_size_) {
            report_heap_overflow(output_size);
            return {-1, -1, nullptr, nullptr};
        }

        void *base = static_cast<char *>(arena.heap_base) + arena.heap_top;
        arena.heap_top += aligned;
        int32_t task_id = next_task_id_++;
        int32_t slot = arena.slot_base + arena.slot_top++;
        task_slot_map_[task_id & task_window_mask_] = slot;
        task_count_ptr_->store(next_task_id_, std::memory_order_release);
        return {task_id, slot, base, static_cast<char *>(base) + aligned};
    }

    int32_t active_count() const { return next_task_id_; }
    int32_t active_buffer() const { return active_buffer_; }
    int32_t task_tail() const { return buffers_[active_buffer_].slot_base; }
    int32_t task_head() const { return buffers_[active_buffer_].slot_base + buffers_[active_buffer_].slot_top; }
    int32_t task_available() const { return slot_arena_size_ - buffers_[active_buffer_].slot_top; }
    int32_t window_size() const { return window_size_; }
    uint64_t heap_available() const { return heap_arena_size_ - buffers_[active_buffer_].heap_top; }
    uint64_t heap_top() const { return buffers_[active_buffer_].heap_top; }
    uint64_t heap_tail() const { return 0; }
    uint64_t heap_capacity() const { return heap_arena_size_; }
    uint64_t heap_used_bytes() const { return buffers_[active_buffer_].heap_top; }

private:
    struct BufferArena {
        int32_t slot_base{0};
        int32_t slot_top{0};
        void *heap_base{nullptr};
        uint64_t heap_top{0};
    };

    int32_t window_size_{0};
    int32_t task_window_mask_{0};
    int32_t slot_arena_size_{0};
    std::atomic<int32_t> *task_count_ptr_{nullptr};
    int32_t *task_slot_map_{nullptr};
    void *heap_base_{nullptr};
    uint64_t heap_size_{0};
    uint64_t heap_arena_size_{0};
    std::atomic<int32_t> *error_code_ptr_{nullptr};
    int32_t next_task_id_{0};
    int32_t active_buffer_{0};
    BufferArena buffers_[PTO2_REPLAY_GRAPH_BUFFER_COUNT];

    void report_task_overflow() {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Replay Graph Task Window Overflow!");
        LOG_ERROR("========================================");
        LOG_ERROR(
            "Graph exceeds task arena: buffer=%d, current=%d, capacity=%d.", active_buffer_,
            buffers_[active_buffer_].slot_top, slot_arena_size_
        );
        LOG_ERROR("Increase PTO2_RING_TASK_WINDOW so one half holds the largest graph.");
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            error_code_ptr_->store(PTO2_ERROR_FLOW_CONTROL_DEADLOCK, std::memory_order_release);
        }
    }

    void report_heap_overflow(int32_t requested) {
        LOG_ERROR("========================================");
        LOG_ERROR("FATAL: Replay Graph Heap Overflow!");
        LOG_ERROR("========================================");
        LOG_ERROR(
            "Graph exceeds heap arena: buffer=%d, top=%" PRIu64 ", size=%" PRIu64 ", requested=%d.", active_buffer_,
            buffers_[active_buffer_].heap_top, heap_arena_size_, requested
        );
        LOG_ERROR("Increase PTO2_RING_HEAP so one half holds the largest graph outputs.");
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            error_code_ptr_->store(PTO2_ERROR_HEAP_RING_DEADLOCK, std::memory_order_release);
        }
    }

    void report_invalid_buffer(int32_t buffer_id) {
        LOG_ERROR("FATAL: invalid replay graph buffer id=%d", buffer_id);
        if (error_code_ptr_) {
            error_code_ptr_->store(PTO2_ERROR_INVALID_ARGS, std::memory_order_release);
        }
    }
};

struct PTO2DepListPool {
    PTO2DepListEntry *base{nullptr};
    int32_t capacity{0};
    int32_t top{1};
    int32_t high_water{0};
    std::atomic<int32_t> *error_code_ptr{nullptr};

    void init(PTO2DepListEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr) {
        base = in_base;
        capacity = in_capacity;
        top = 1;
        high_water = 0;
        error_code_ptr = in_error_code_ptr;
        base[0].slot_state = nullptr;
        base[0].next = nullptr;
    }

    PTO2DepListEntry *alloc() {
        if (top >= capacity) {
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Replay Graph Dependency Pool Overflow!");
            LOG_ERROR("========================================");
            LOG_ERROR("Whole graph needs at least %d fanout entries; capacity=%d.", top, capacity - 1);
            LOG_ERROR("Increase PTO2_RING_DEP_POOL to hold every graph edge.");
            LOG_ERROR("========================================");
            if (error_code_ptr) {
                error_code_ptr->store(PTO2_ERROR_DEP_POOL_OVERFLOW, std::memory_order_release);
            }
            return nullptr;
        }
        PTO2DepListEntry *entry = &base[top++];
        if (used() > high_water) high_water = used();
        return entry;
    }

    PTO2DepListEntry *prepend(PTO2DepListEntry *head, PTO2TaskSlotState *slot_state) {
        PTO2DepListEntry *entry = alloc();
        if (!entry) return nullptr;
        entry->slot_state = slot_state;
        entry->next = head;
        return entry;
    }

    int32_t used() const { return top - 1; }
    int32_t available() const { return capacity - top; }
};

#endif  // PTO_RING_BUFFER_H
