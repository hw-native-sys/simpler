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

static inline __aicore__ bool scheduler_async_finish_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    SchedulerAsyncLocalState &async, bool &made_progress, uint64_t profiling_level = 0
) {
    using namespace scheduler_async_detail;
    made_progress = false;
    if (async.error_code != 0 || async.layout == nullptr || local == nullptr || run == nullptr || base != async.base)
        return fail(async, -1);
    uint32_t index = ASYNC_INVALID_INDEX;
    uint64_t owned = 0;
    if (!claim(
            async, async.layout->finish_directory_offset, async.layout->finish_leases_offset,
            async.layout->async_task_count, async.finish_cursor, index, owned
        ))
        return false;
    if (index == ASYNC_INVALID_INDEX) return true;
    auto *leases = scheduler_state_at<SchedulerAsyncLease>(base, async.layout->finish_leases_offset);
    auto *tasks = scheduler_state_at<SchedulerAsyncTaskData>(base, async.layout->task_data_offset);
    observe(&tasks[index], sizeof(SchedulerAsyncTaskData));
    const int64_t task_ref = tasks[index].task_ref;
    auto failed = [&]() __aicore__ {
        scheduler_async_lease_fail(leases[index], owned);
        return fail(async, task_ref);
    };
    if (!scheduler_task_ref_valid(graph, task_ref)) return failed();
    auto *joins = scheduler_state_at<SchedulerAsyncTaskJoin>(base, async.layout->joins_offset);
    const uint64_t joined = scheduler_gm_fetch_add(joins[index].word, UINT64_C(0));
    if (!scheduler_async_pending::ready_to_finish(joined)) return failed();
    auto *control = scheduler_task_control_at(base, local, task_ref);
    const uint64_t gate = scheduler_gm_fetch_add(control->completion_state, UINT64_C(0));
    if (!scheduler_completion_valid(gate) || scheduler_completion_remaining(gate) != 0 ||
        scheduler_completion_debt(gate) != scheduler_async_pending::registered(joined))
        return failed();
    scheduler_protocol_task_done(base, local, task_ref, true, true);
    scheduler_cache_barrier();
    if (scheduler_gm_compare_exchange(
            control->completion_state, gate, scheduler_completion_completed_state(scheduler_completion_total(gate))
        ) != gate)
        return failed();
    scheduler_protocol_task_done(base, local, task_ref, false, true);
    auto *directory = scheduler_state_at<SchedulerAsyncDirectoryWord>(base, async.layout->finish_directory_offset);
    if (scheduler_async_lease_finish(leases[index], owned).result != SchedulerAsyncLeaseResult::SUCCESS ||
        scheduler_async_directory_clear(directory[index / 64].bits, index % 64) !=
            SchedulerAsyncDirectoryResult::SUCCESS)
        return fail(async, task_ref);
    scheduler_cache_barrier();
    ++local->pending_completed;
    made_progress = true;
    return true;
}

static inline __aicore__ bool scheduler_service_async_slice(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    uint64_t profiling_level, bool &made_progress
) {
    made_progress = false;
    if (local->async == nullptr) return true;
    auto &async = *local->async;
    async.business_returns = 0;
    // An inactive slice owns no shared work. The resident loop still polls
    // run errors; an active slice checks them before claiming any responsibility.
    if (async.error_code == 0 && !scheduler_async_work_activated(&async)) return true;
    if (scheduler_read_worker_error(run, local->worker_id()) != 0) return false;
    const bool finish = async.finish_first;
    async.finish_first = !finish;
    SchedulerScalarScope scalar_step(
        local, finish ? SchedPhaseKind::ScalarAsyncFinish : SchedPhaseKind::ScalarAsyncCheck
    );
    const bool ok =
        finish ?
            scheduler_async_finish_step(graph, base, local, run, async, made_progress, profiling_level) :
            scheduler_async_check_step(
                async, UINT32_C(1) << local->config.self_core_type, 8, SchedulerAsyncCompletionObserver{}, made_progress
            );
    if (!ok)
        scheduler_record_error(
            run, async.failed_task_ref, scheduler_async_error_result(&async), &graph, local,
            finish ? SchedulerErrorSite::ASYNC_FINISH_FAILED : SchedulerErrorSite::ASYNC_CHECK_FAILED
        );
    return ok;
}
