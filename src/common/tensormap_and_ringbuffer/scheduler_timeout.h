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

#include "runtime_status/scheduler_error.h"
#include "shared_memory.h"

// Counts and task identity observed by a TMR scheduler at the no-progress deadline.
struct StallClassification {
    int32_t detail;          // SIMPLER_STALL_DETAIL_*
    int32_t cnt_running;     // tasks observed RUNNING (on a core)
    int32_t cnt_ready;       // fanin-satisfied but not dispatched
    int32_t cnt_waiting;     // still waiting on fanin
    int32_t cnt_async_wait;  // all AICore subtasks finished, deferred completion pending
    int32_t completed;       // completed_tasks_ snapshot
    int32_t total;           // total_tasks_ snapshot
    int32_t orch_done;       // orchestrator_done flag (0/1)
    TaskId stuck_task_id;    // S1/S6: first task in the dominant class, invalid() if none
    int32_t stuck_core;      // S1: core hosting it (-1 if none)
};

// The first reporter owns the error and its thread identity. Later reporters
// still contribute their bits, including when a different code won the latch.
inline bool latch_scheduler_error(SharedMemoryHeader *header, int32_t thread_idx, int32_t code) {
    if (header == nullptr || code == SIMPLER_ERROR_NONE) return false;
    const bool first = try_latch_runtime_error(header->sched_error_code, code);
    if (first) header->sched_error_thread.store(thread_idx, std::memory_order_release);
    if (thread_idx >= 0 && thread_idx < 32) {
        header->sched_error_bitmap.fetch_or(1U << static_cast<uint32_t>(thread_idx), std::memory_order_acq_rel);
    }
    return first;
}

// Only the reporter whose timeout was latched publishes its snapshot. The
// caller elects the owner of emergency shutdown independently of this result.
inline void report_scheduler_timeout(SharedMemoryHeader *header, int32_t thread_idx, const StallClassification &cls) {
    if (!latch_scheduler_error(header, thread_idx, SIMPLER_ERROR_SCHEDULER_TIMEOUT)) return;
    header->sched_stall_completed.store(cls.completed, std::memory_order_relaxed);
    header->sched_stall_total.store(cls.total, std::memory_order_relaxed);
    header->sched_stall_cnt_running.store(cls.cnt_running, std::memory_order_relaxed);
    header->sched_stall_cnt_ready.store(cls.cnt_ready, std::memory_order_relaxed);
    header->sched_stall_cnt_waiting.store(cls.cnt_waiting, std::memory_order_relaxed);
    header->sched_stall_cnt_async_wait.store(cls.cnt_async_wait, std::memory_order_relaxed);
    header->sched_stall_orch_done.store(cls.orch_done, std::memory_order_relaxed);
    header->sched_stall_task_id.store(cls.stuck_task_id, std::memory_order_relaxed);
    header->sched_stall_core.store(cls.stuck_core, std::memory_order_relaxed);
    // detail published last (release) so a host reading a non-NONE detail
    // sees the locators above already settled.
    header->sched_stall_detail.store(cls.detail, std::memory_order_release);
}
