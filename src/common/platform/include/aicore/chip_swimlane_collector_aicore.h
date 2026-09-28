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
 * @file chip_swimlane_collector_aicore.h
 * @brief AICore performance data collection interface
 *
 * Provides lightweight performance recording interface for AICore kernels.
 * Uses dcci for efficient cache management instead of memory barriers.
 */

#pragma once

#include "common/chip_swimlane_profiling.h"
#include "aicore/aicore.h"

// Include platform-specific timestamp implementation
// Build system selects the correct inner_kernel.h based on platform:
// - src/<platform>/platform/onboard/aicore/inner_kernel.h (real hardware)
// - src/<platform>/platform/sim/aicore/inner_kernel.h (simulation)
// Both provide unified get_sys_cnt_aicore() interface
#include "inner_kernel.h"

// ============= Public Interface =============

/**
 * AICore-local rotation state. Tracks which buffer this core is currently
 * writing into and which slot is next. Reset by `chip_swimlane_aicore_reserve_task_record`
 * when it observes a `current_buf_seq` bump on the shared `ChipSwimlaneActiveHead`
 * cache line.
 */
struct ChipSwimlaneAicoreLocalState {
    __gm__ ChipSwimlaneAicoreTaskBuffer *cached_buf = nullptr;
    // Must start != AICPU's initial head.current_buf_seq (0) so the first
    // reservation observes a mismatch and loads the buffer pointer.
    uint32_t cached_buf_seq = UINT32_MAX;
    uint32_t slot_within_buf = 0;
};

/**
 * Reserve the record slot for one task.
 *
 * AICore reserves from the currently-published per-core buffer before ACK.
 * The returned record pointer remains the task's write target after FIN even
 * if AICPU has published the next buffer by then.
 */
__aicore__ __attribute__((always_inline)) static inline __gm__ ChipSwimlaneAicoreTaskRecord *
chip_swimlane_aicore_reserve_task_record(__gm__ ChipSwimlaneActiveHead *head, ChipSwimlaneAicoreLocalState *local) {
    dcci(head, SINGLE_CACHE_LINE);
    if (head->current_buf_seq != local->cached_buf_seq) {
        local->cached_buf_seq = head->current_buf_seq;
        local->cached_buf = reinterpret_cast<__gm__ ChipSwimlaneAicoreTaskBuffer *>(head->current_buf_ptr);
        local->slot_within_buf = 0;
    }
    if (local->cached_buf == nullptr) {
        return nullptr;
    }

    uint32_t slot = local->slot_within_buf;
    if (slot >= PLATFORM_AICORE_BUFFER_SIZE) {
        // Refuse to write past the end if AICPU failed to rotate.
        return nullptr;
    }

    local->slot_within_buf = slot + 1;
    return &local->cached_buf->records[slot];
}

/**
 * Commit one task's timestamps and identity to its pre-ACK reservation.
 *
 * `end_time` is captured immediately after execute_task and before FIN. The
 * commit itself runs after FIN and never reads the possibly-rotated head.
 */
__aicore__ __attribute__((always_inline)) static inline void chip_swimlane_aicore_commit_task_record(
    __gm__ ChipSwimlaneAicoreTaskRecord *record, const __gm__ TaskId &task_token, uint32_t reg_task_id,
    uint64_t receive_time, uint64_t start_time, uint64_t end_time
) {
    if (record == nullptr) {
        return;
    }

    record->start_time = start_time;
    record->end_time = end_time;
    TaskId::assign(record->task_token, task_token);
    record->reg_task_id = reg_task_id;
    // 32-bit delta; receive_time always precedes start_time on the same core.
    record->receive_to_start_cycles = static_cast<uint32_t>(start_time - receive_time);

    dcci(record, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}

/**
 * Compatibility wrapper for platforms that have not moved reservation before ACK.
 */
__aicore__ __attribute__((always_inline)) static inline void chip_swimlane_aicore_record_task(
    __gm__ ChipSwimlaneActiveHead *head, ChipSwimlaneAicoreLocalState *local, const __gm__ TaskId &task_token,
    uint32_t reg_task_id, uint64_t receive_time, uint64_t start_time, uint64_t end_time
) {
    __gm__ ChipSwimlaneAicoreTaskRecord *record = chip_swimlane_aicore_reserve_task_record(head, local);
    chip_swimlane_aicore_commit_task_record(record, task_token, reg_task_id, receive_time, start_time, end_time);
}
