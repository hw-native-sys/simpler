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
#include "common/dfx_backpressure_device.h"
#include "common/memory_barrier.h"
#include "common/platform_config.h"

namespace profiling_device {

// Common AICPU-side producer algorithm for profiling collectors.
//
// Module supplies the concrete shared-memory layout and subsystem-specific
// ready-entry/drop hooks; this engine owns the queue handoff and buffer-switch
// control flow.
//
// The push/pop gates here implement the device half of the block-on-contention
// backpressure protocol: on a full ready queue or empty free queue the writer
// parks at its buffer-switch gate (via dfx_backpressure_device.h) until the host
// clears the freeze, and only breaks to the single failure exit when the
// 30-second host-crash backstop (Module::kBackpressureWaitCycles) trips. Full
// design — dual-signal freeze, conjunction release, (0,0) escape-window and
// deadlock-freedom arguments — in docs/dfx/global-backpressure-design.md.
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

        // Push gate with incremental release support:
        // 1. Check RQ slot availability (unified check, no duplication)
        // 2. If slot exists, allow use even during freeze (incremental release)
        // 3. Gate control happens in enqueue_ready() after push operation
        // 4. No slot: mark contention and spin (triggers backpressure mechanism)
        // Timeout-protected to prevent infinite spin on host crash or hardware failure
        bool contended_signalled = false;
        const uint64_t start = get_sys_cnt_aicpu();
        do {
            // Unified RQ slot check (check once, use result for both freeze and non-freeze)
            uint32_t current_tail = header->queue_tails[thread_idx];
            uint32_t current_head = header->queue_heads[thread_idx];
            uint32_t next_tail = (current_tail + 1) % Module::kReadyQueueSize;

            if (next_tail != current_head) {
                // Slot available - return for both freeze and non-freeze cases
                // enqueue_ready will handle the push and then wait for gate if needed
                *tail_out = current_tail;
                *head_out = current_head;
                return true;
            }

            // No slot available - mark contention and spin
            // This triggers backpressure: freeze may open, RQ may drain
            dfx_backpressure::mark_rq_contended(header, &contended_signalled);

            // Timeout protection
            if (get_sys_cnt_aicpu() - start >= Module::kBackpressureWaitCycles) {
                break;  // timeout — fall through to the single failure exit below
            }
            SPIN_WAIT_HINT();
        } while (true);

        return false;
    }

    static bool
    wait_for_free_queue_entry(DataHeader *header, FreeQueue *free_queue, uint32_t *head_out, uint32_t *tail_out) {
        if (free_queue == nullptr) {
            return false;
        }

        // Pop gate. The loop, not any single call, is what holds a starved lane:
        // 1. FQ slot available -> take it and return.
        // 2. FQ empty -> raise fq_contended (leader signal, once per wait).
        // 3. pop_freeze_barrier parks only while the host holds fq_freeze open.
        //    Raising fq_contended does not itself park this lane: until the host
        //    observes the signal and opens the freeze, the barrier sees
        //    fq_freeze_active==0 and returns at once, so this loop re-checks and
        //    bridges the host round-trip. Once frozen, the barrier spins here
        //    until release; its timeout arms only then and is the sole give-up
        //    (host dead/hung mid-freeze).
        // 4. Freeze released -> loop re-checks and picks up the refilled slot.
        bool contended_signalled = false;

        do {
            // Step 1: Check FQ slot availability
            uint32_t head = free_queue->head;
            uint32_t tail = free_queue->tail;
            if (head != tail) {
                *head_out = head;
                *tail_out = tail;
                rmb();  // acquire: order the tail read above before the caller's buffer_ptrs read
                return true;
            }

            // Step 2: No slot available - mark contention
            dfx_backpressure::mark_fq_contended(header, &contended_signalled);

            // Step 3: Park while the host holds the freeze open; returns at once
            // when it is not open. Timeout fires only while frozen.
            if (!dfx_backpressure::pop_freeze_barrier(header, Module::kBackpressureWaitCycles)) {
                break;  // gate timeout — fall through to the single failure exit below
            }

            // Step 4: Gate not held (never opened yet, or released) — re-check the
            // free queue; an open→drain→refill cycle leaves a slot to pick up here.
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

        // Push-gate global-sync park: the held buffer is now flushed (nothing in
        // hand), so blocking here is a clean, hostage-free stop. Every lane that
        // reaches its push gate during rq_freeze converges here → one aligned
        // common-mode gap; production resumes only after the host clears the
        // freeze (conjunction release).
        if (!dfx_backpressure::push_freeze_barrier(header, Module::kBackpressureWaitCycles)) {
            // Gate timeout: the entry is already published to the ready queue,
            // so the caller (switch_buffer) accounts the dropped records.
            return -1;
        }
        return 0;
    }

    static Buffer *pop_free(Context ctx, State *state, uint32_t next_seq) {
        if (state == nullptr) {
            return nullptr;
        }

        FreeQueue *free_queue = Module::free_queue(state);
        uint32_t head = 0;
        uint32_t tail = 0;
        if (!wait_for_free_queue_entry(Module::header(ctx), free_queue, &head, &tail)) {
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
