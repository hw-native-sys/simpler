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
 * @file l2_perf_collector_aicore.h
 * @brief AICore performance data collection interface
 *
 * Provides lightweight performance recording interface for AICore kernels.
 * Uses dcci for efficient cache management instead of memory barriers.
 */

#ifndef PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_
#define PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_

#include "common/l2_perf_profiling.h"
#include "aicore/aicore.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_kernel.h based on platform:
// - src/a2a3/platform/onboard/aicore/inner_kernel.h (real hardware)
// - src/a2a3/platform/sim/aicore/inner_kernel.h (simulation)
// Both provide unified get_sys_cnt_aicore() interface
#include "inner_kernel.h"

// ============= Public Interface =============

/**
 * AICore-local rotation state. Tracks which buffer this core is currently
 * writing into and which slot is next. Reset by `l2_perf_aicore_record_task`
 * when it observes a generation bump on the shared `AicoreRotation` channel
 * (AICPU rotates by writing `current_buf_ptr` + bumping `generation`, so the
 * AICore-local state self-recovers without any AICore-side spin-wait).
 */
struct AicoreLocalState {
    __gm__ L2PerfAicoreBuffer *cached_buf = nullptr;
    // Must start != AICPU's initial generation (1) so the first record_task
    // call observes a generation mismatch and loads the buffer pointer.
    uint32_t cached_generation = 0;
    uint32_t slot_within_buf = 0;
};

/**
 * Record task execution performance data.
 *
 * AICore writes a slim L2PerfAicoreRecord into its currently-published
 * per-core L2PerfAicoreBuffer at `records[slot_within_buf++]`. The
 * publication channel is an AicoreRotation cache line addressed via
 * `KernelArgs::aicore_ring_addr[block_idx]` (now points to AicoreRotation,
 * not directly to a buffer). AICPU updates `rotation->current_buf_ptr` and
 * bumps `rotation->generation` at dispatch boundaries; AICore detects the
 * change by `dcci`-ing the rotation line per task and comparing generation
 * to its locally cached copy.
 *
 * AICPU and AICore never read each other's data on the hot path. The host
 * post-processor joins the AICore stream (multi-buffer per core, in order)
 * with the AICPU stream by `reg_task_id` at flush time. See
 * `docs/dfx/l2-swimlane-profiling.md`.
 *
 * Race avoidance: AICPU rotates strictly before `write_reg(DATA_MAIN_BASE)`
 * for the first task of a new BUFFER_SIZE batch. The runtime's
 * completion-before-dispatch invariant guarantees all prior tasks have FIN'd,
 * so AICore has already finished writing their records before AICPU enqueues
 * the old buffer to the ready queue.
 *
 * @param rotation Per-core AicoreRotation channel (cached at kernel entry
 *                 from KernelArgs::aicore_ring_addr[block_idx])
 * @param local    Per-core AICore-local state (caller-owned static)
 * @param task_id  Register dispatch id (DATA_MAIN_BASE), low 32 bits
 * @param start_time Start timestamp (get_sys_cnt)
 * @param end_time   End timestamp
 */
__aicore__ __attribute__((always_inline)) static inline void l2_perf_aicore_record_task(
    __gm__ AicoreRotation *rotation, AicoreLocalState *local, uint32_t task_id, uint64_t start_time, uint64_t end_time
) {
    // Re-fetch rotation channel each task; cheap relative to the
    // baseline `dcci(payload, ENTIRE_DATA_CACHE)` we already pay per task.
    dcci(rotation, SINGLE_CACHE_LINE);
    if (rotation->generation != local->cached_generation) {
        local->cached_generation = rotation->generation;
        local->cached_buf = reinterpret_cast<__gm__ L2PerfAicoreBuffer *>(rotation->current_buf_ptr);
        local->slot_within_buf = 0;
    }
    if (local->cached_buf == nullptr) {
        // Rotation channel published a null pointer (AICPU couldn't pop a
        // fresh buffer from free_queue). Drop silently — AICPU side already
        // bumped dropped_record_count.
        return;
    }

    uint32_t slot = local->slot_within_buf;
    if (slot >= PLATFORM_AICORE_BUFFER_SIZE) {
        // Defensive: AICPU should rotate before this can happen. If it
        // didn't, refuse to write past the end rather than corrupt adjacent
        // memory.
        return;
    }

    __gm__ L2PerfAicoreRecord *record = &local->cached_buf->records[slot];
    record->start_time = start_time;
    record->end_time = end_time;
    record->task_id = task_id;
    local->slot_within_buf = slot + 1;

    // Flush record to GM so host can read it after the buffer is enqueued.
    dcci(record, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}

#endif  // PLATFORM_AICORE_L2_PERF_COLLECTOR_AICORE_H_
