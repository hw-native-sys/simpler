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

#include "scheduler_async_control.h"
#include "scheduler_ready.h"
#include "scheduler_protocol_execution.h"

// The join RMW selects one finisher. The proof is never published or queued.
static inline __aicore__ bool scheduler_async_complete_task(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerAsyncCompletion &completion
) {
    if (local == nullptr || local->async == nullptr || run == nullptr) return false;
    auto &async = *local->async;
    const int64_t task_ref = completion.task_ref;
    SchedulerScalarScope scalar_step(local, SchedPhaseKind::ScalarAsyncFinish, task_ref);
    auto failed = [&]() __aicore__ {
        scheduler_async_detail::fail(async, task_ref);
        scheduler_record_error(
            run, task_ref, scheduler_async_error_result(&async), &graph, local, SchedulerErrorSite::ASYNC_FINISH_FAILED
        );
        return false;
    };
    if (async.error_code != 0 || async.layout == nullptr || base != async.base ||
        completion.result != SchedulerAsyncCompletionResult::FINISH_OWNER ||
        completion.run_epoch != async.layout->run_epoch || !scheduler_task_ref_valid(graph, task_ref) ||
        !scheduler_async_pending::ready_to_finish(completion.joined))
        return failed();
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return failed();
    uint32_t index = pto2::resident::ASYNC_INVALID_INDEX;
    if (!scheduler_async_detail::task_index(async, task_ref, index) || index == pto2::resident::ASYNC_INVALID_INDEX)
        return failed();
    auto *control = scheduler_task_control_at(base, local, task_ref);
    const auto &metadata = *scheduler_task_metadata_at(base, local, task_ref);
    uint32_t total = 0;
    if (!scheduler_subtask_count_checked(metadata.logical_block_num, scheduler_task_lanes(metadata).count(), &total))
        return failed();
    // Immutable geometry and the join owner define the expected gate. Only the
    // CAS confirms all physical returns and the single async marker match it.
    const uint64_t gate = scheduler_completion_encode(total, 0, 1);
    if (!scheduler_completion_valid(gate) || scheduler_async_pending::registered(completion.joined) > total)
        return failed();
    scheduler_protocol_task_done(base, local, task_ref, true, true);
    scheduler_cache_barrier();
    if (scheduler_gm_compare_exchange(control->completion_state, gate, scheduler_completion_completed_state(total)) !=
        gate)
        return failed();
    scheduler_protocol_task_done(base, local, task_ref, false, true);
    scheduler_cache_barrier();
    scheduler_publish_task_done(base, local, task_ref);
    scheduler_remember_task_proof(local, task_ref, SchedulerTaskProof::DONE);
    return true;
}

static inline __aicore__ bool scheduler_service_async_slice(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    bool &made_progress
) {
    made_progress = false;
    if (local->async == nullptr) return true;
    auto &async = *local->async;
    if (!scheduler_async_service_due(async)) return true;
    SchedulerScalarScope scalar_async(local, SchedPhaseKind::ScalarAsyncService);
    SchedulerScalarScope scalar_step(local, SchedPhaseKind::ScalarAsyncCheck);
    const bool retry = async.retry_counter_address != 0 && async.error_code == 0;
    // The resident loop also checks for errors periodically. Check again before
    // consuming any condition or publishing credit after a successful retry.
    if (retry && !scheduler_async_counter_reached(async.retry_counter_address, async.retry_counter_threshold))
        return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    if (async.error_code != 0) {
        scheduler_record_error(
            run, async.failed_task_ref, scheduler_async_error_result(&async), &graph, local,
            SchedulerErrorSite::ASYNC_CHECK_FAILED
        );
        return false;
    }
    SchedulerAsyncCompletion completion{};
    bool ok = scheduler_async_check_step(
        async, UINT32_C(1) << local->config.self_core_type, 8, SchedulerAsyncCompletionObserver{}, made_progress,
        completion, retry
    );
    scalar_step.close();
    if (ok && completion.result == SchedulerAsyncCompletionResult::FINISH_OWNER)
        ok = scheduler_async_complete_task(graph, base, local, run, completion);
    if (!ok)
        scheduler_record_error(
            run, async.failed_task_ref, scheduler_async_error_result(&async), &graph, local,
            SchedulerErrorSite::ASYNC_CHECK_FAILED
        );
    return ok;
}
