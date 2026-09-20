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

#include <atomic>
#include <cstdint>

#include "host_build_graph/runtime_status.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_types.h"

// This bridge is intentionally AICPU/host-only. AICore code publishes its
// detailed first error through SchedulerRunControl; AICPU maps that internal
// diagnostic to the existing host runtime status ABI without making this
// std::atomic helper visible to either AICore compiler.
inline int32_t aicore_scheduler_runtime_error_code(uint64_t scheduler_error) {
    if (scheduler_error == 0) return SIMPLER_ERROR_NONE;
    if (scheduler_error == static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)) {
        return SIMPLER_ERROR_SCHEDULER_TIMEOUT;
    }
    return SIMPLER_ERROR_INVALID_ARGS;
}

inline bool latch_aicore_scheduler_runtime_error(std::atomic<int32_t> *sched_error_code, uint64_t scheduler_error) {
    if (sched_error_code == nullptr || scheduler_error == 0) return false;
    int32_t expected = SIMPLER_ERROR_NONE;
    return sched_error_code->compare_exchange_strong(
        expected, aicore_scheduler_runtime_error_code(scheduler_error), std::memory_order_acq_rel,
        std::memory_order_acquire
    );
}

inline bool record_aicore_scheduler_runtime_error(
    SchedulerRunControl *run_control, SchedulerGraphResult scheduler_error, SchedulerErrorSite error_site
) {
    if (run_control == nullptr || scheduler_error == SchedulerGraphResult::OK) return false;
    uint64_t expected = 0;
    if (!__atomic_compare_exchange_n(
            &run_control->error_claimed, &expected, UINT64_C(1), false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
        ))
        return false;
    __atomic_store_n(&run_control->error_site, static_cast<uint64_t>(error_site), __ATOMIC_RELAXED);
    __atomic_store_n(&run_control->scheduler_error, static_cast<uint64_t>(scheduler_error), __ATOMIC_RELEASE);
    return true;
}

/**
 * The run's host runtime status with the AICore's recorded error folded into
 * the shared header first.
 *
 * Folding is first-wins (`latch_aicore_scheduler_runtime_error` compare-
 * exchanges from `SIMPLER_ERROR_NONE`), so an error already latched keeps its
 * own code and a control holding none writes nothing. A null `sched_error_code`
 * has nothing to report; a null `run_control` reports the header as it stands,
 * because a caller that could not look has not established that the run was
 * clean.
 *
 * This folds whatever the control holds **at the moment of the call** and makes
 * no demand on when that is: the supervisor calls it mid-run and the
 * failed-init exit calls it before any shutdown. Whether a fold is *complete*
 * is a property of the call site rather than of this function — only the
 * finalizer's call runs after every participant has finished its shutdown
 * partition, which is what lets that one cover every core instead of the
 * caller's own.
 */
inline int32_t
settle_aicore_runtime_status(const SchedulerRunControl *run_control, std::atomic<int32_t> *sched_error_code) {
    if (sched_error_code == nullptr) return SIMPLER_ERROR_NONE;
    if (run_control != nullptr) {
        (void)latch_aicore_scheduler_runtime_error(sched_error_code, run_control->scheduler_error);
    }
    return runtime_status_from_error_code(sched_error_code->load(std::memory_order_acquire));
}
