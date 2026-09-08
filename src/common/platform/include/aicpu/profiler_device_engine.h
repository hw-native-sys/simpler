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

// SPIN_WAIT_HINT() is the repo-wide AICPU spin-wait relax used by every
// resource-wait spin (ring/dep-pool full, lock_fanout ticket locks, scheduler
// dispatch). It is platform-tiered: onboard silicon expands to a no-op (AICPU
// owns its A55 core), sim adds sched_yield() so the oversubscribed host cores
// don't starve the AICore threads running real kernels.
//
// Reuse whatever the platform already provides; only supply the fallback — for
// pure host, struct-only builds where spin_hint.h is off the path and the spin
// templates are never instantiated — when nobody else has defined it.
#ifndef SPIN_WAIT_HINT
#if __has_include("spin_hint.h")
#include "spin_hint.h"
#else
#define SPIN_WAIT_HINT() ((void)0)
#endif
#endif

namespace profiling_device {

// Common AICPU-side producer algorithm for profiling collectors.
//
// Module supplies the concrete shared-memory layout and subsystem-specific
// ready-entry/drop hooks; this engine owns the queue handoff and buffer-switch
// control flow.
//
// The push/pop gates here implement the device half of the per-lane
// block-on-contention backpressure protocol: on a full ready queue or empty free
// queue the writer spins at its own buffer-switch gate until the host drains that
// queue or refills it, and no peer lane is involved. Each gate carries one
// `Module::kBackpressureWaitCycles` budget across the whole wait — a 30-second
// backstop against a spin the host will never end, not a normal-path wait — and
// reports failure only while
// ownership is still on the device, so the caller can account the affected records
// dropped. Full design — per-lane recovery, the single-writer free-queue
// invariant, and the deadlock-freedom argument — in
// docs/dfx/backpressure-design.md.
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

        // Push gate. Recovery is entirely local: the one drain shard that serves
        // this thread's ready queue advances its head, so spinning here until a
        // slot appears needs no peer lane to stop and no host handshake. The
        // budget only bounds a drain that never makes progress.
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

            if (get_sys_cnt_aicpu() - start >= Module::kBackpressureWaitCycles) {
                break;  // timeout — fall through to the single failure exit below
            }
            SPIN_WAIT_HINT();
        } while (true);

        return false;
    }

    static bool wait_for_free_queue_entry(FreeQueue *free_queue, uint32_t *head_out, uint32_t *tail_out) {
        if (free_queue == nullptr) {
            return false;
        }

        // Pop gate. This lane's own drain shard refills this free_queue from its
        // shard-local recycled pool, which the replenish thread keeps stocked
        // independently, so a dry lane recovers without any peer lane parking.
        //
        // The budget spans the whole wait rather than one iteration: it must
        // measure "the host stopped making progress", so re-arming it per attempt
        // would let a permanently short pool spin here forever.
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

            if (get_sys_cnt_aicpu() - start >= Module::kBackpressureWaitCycles) {
                break;  // timeout — fall through to the single failure exit below
            }
            SPIN_WAIT_HINT();
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

    static Buffer *claim_free(Context ctx, State *state, FreeQueue *free_queue, uint32_t head, uint32_t next_seq) {
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

    // Non-blocking startup pop. It shares the ownership/head/seq bookkeeping
    // with pop_free(), but does not enter the runtime backpressure gate. This
    // keeps initialization failure observable instead of waiting for a host
    // management loop that may not have started yet.
    static Buffer *try_pop_free(Context ctx, State *state, uint32_t next_seq) {
        if (state == nullptr) {
            return nullptr;
        }

        FreeQueue *free_queue = Module::free_queue(state);
        if (free_queue == nullptr) {
            return nullptr;
        }
        uint32_t head = free_queue->head;
        uint32_t tail = free_queue->tail;
        if (head == tail) {
            return nullptr;
        }
        rmb();  // acquire: order the tail read above before buffer_ptrs
        return claim_free(ctx, state, free_queue, head, next_seq);
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
        return claim_free(ctx, state, free_queue, head, next_seq);
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
