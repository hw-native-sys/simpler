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

#pragma once

#include <cstdint>

#include "aicpu/device_time.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"

namespace profiling_device {

// Common AICPU-side producer algorithm for profiling collectors.
//
// Module supplies the concrete shared-memory layout and subsystem-specific
// ready-entry/drop hooks; this engine owns the queue handoff and buffer-switch
// control flow.
template <typename Module>
struct DeviceProfilerEngine {
    using Context = typename Module::Context;
    using DataHeader = typename Module::DataHeader;
    using State = typename Module::State;
    using FreeQueue = typename Module::FreeQueue;
    using Buffer = typename Module::Buffer;

    static bool wait_for_ready_queue_space(DataHeader *header, int thread_idx, uint32_t *tail_out, uint32_t *head_out) {
        if (header == nullptr || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS) {
            return false;
        }

        const uint64_t start = get_sys_cnt_aicpu();
        do {
            uint32_t current_tail = header->queue_tails[thread_idx];
            uint32_t current_head = header->queue_heads[thread_idx];
            uint32_t next_tail = (current_tail + 1) % Module::kReadyQueueSize;
            if (next_tail != current_head) {
                *tail_out = current_tail;
                *head_out = current_head;
                return true;
            }
            if (Module::kBackpressureWaitCycles == 0) {
                break;
            }
            if (get_sys_cnt_aicpu() - start >= Module::kBackpressureWaitCycles) {
                break;
            }
        } while (true);

        return false;
    }

    static bool wait_for_free_queue_entry(FreeQueue *free_queue, uint32_t *head_out, uint32_t *tail_out) {
        if (free_queue == nullptr) {
            return false;
        }

        const uint64_t start = get_sys_cnt_aicpu();
        do {
            uint32_t head = free_queue->head;
            uint32_t tail = free_queue->tail;
            if (head != tail) {
                *head_out = head;
                *tail_out = tail;
                rmb();  // acquire: order the tail read above before the caller's buffer_ptrs read
                return true;
            }
            if (Module::kBackpressureWaitCycles == 0) {
                break;
            }
            if (get_sys_cnt_aicpu() - start >= Module::kBackpressureWaitCycles) {
                break;
            }
        } while (true);

        return false;
    }

    static int enqueue_ready(Context ctx, uint64_t buffer_ptr, uint32_t buffer_seq) {
        DataHeader *header = Module::header(ctx);
        int q = Module::ready_thread(ctx);
        uint32_t current_tail = 0;
        uint32_t current_head = 0;
        if (!wait_for_ready_queue_space(header, q, &current_tail, &current_head)) {
            return -1;
        }

        uint32_t next_tail = (current_tail + 1) % Module::kReadyQueueSize;
        Module::write_ready_entry(ctx, current_tail, buffer_ptr, buffer_seq);
        wmb();  // publish: entry fields visible before the tail advance
        header->queue_tails[q] = next_tail;
        return 0;
    }

    static Buffer *pop_free(Context ctx, State *state, uint32_t next_seq) {
        if (state == nullptr) {
            return nullptr;
        }

        FreeQueue *free_queue = Module::free_queue(state);
        uint32_t head = 0;
        uint32_t tail = 0;
        if (!wait_for_free_queue_entry(free_queue, &head, &tail)) {
            return nullptr;
        }

        uint64_t buf_ptr = free_queue->buffer_ptrs[head % Module::kSlotCount];
        rmb();  // acquire-strengthening: order buffer_ptrs read before taking ownership via head advance
        free_queue->head = head + 1;
        if (buf_ptr == 0) {
            Module::on_null_free_slot(ctx, state);
            return nullptr;
        }

        auto *buf = reinterpret_cast<Buffer *>(buf_ptr);
        Module::set_count(buf, 0);
        wmb();
        Module::set_current_ptr(state, buf_ptr);
        Module::set_current_seq(state, next_seq);
        Module::on_pop_success(ctx, state, buf);
        wmb();
        return buf;
    }

    static void switch_buffer(Context ctx, State *state) {
        if (state == nullptr) {
            return;
        }

        auto *full_buf = reinterpret_cast<Buffer *>(Module::current_ptr(state));
        if (full_buf == nullptr) {
            return;
        }

        uint32_t seq = Module::current_seq(state);
        int rc = enqueue_ready(ctx, Module::current_ptr(state), seq);
        if (rc != 0) {
            Module::account_dropped(ctx, state, Module::count(full_buf));
            Module::on_enqueue_failed(ctx, state, full_buf);
            Module::set_count(full_buf, 0);
            wmb();
            return;
        }

        uint32_t next_seq = seq + 1;
        Module::set_current_ptr(state, 0);
        Module::set_current_seq(state, next_seq);
        Module::on_current_cleared(ctx, state);
        wmb();

        Buffer *new_buf = pop_free(ctx, state, next_seq);
        if (new_buf == nullptr) {
            Module::on_no_replacement(ctx, state);
        }
        Module::on_switch_complete(ctx, state, new_buf);
    }
};

}  // namespace profiling_device
