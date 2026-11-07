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

#include "scheduler_async_layout.h"
#include "scheduler_ready.h"

struct SchedulerAsyncFinishProtocol {
    inline __aicore__ static void task_done(
        __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint64_t order, uint32_t flags, bool before
    ) {
        scheduler_protocol_task_actor(base, local, task, order, flags, true, before);
    }
};

namespace scheduler_async_finish_detail {
// A finishing core need not still participate in the task's admission. The
// immutable run record, rather than this core's admission frontier, owns order.
inline __aicore__ bool protocol_order(
    __gm__ void *base, const SchedulerLocalState *local, int64_t task, uint64_t epoch,
    const SchedulerTaskMetadata &metadata, uint64_t *order
) {
    *order = UINT64_MAX;
    if (!scheduler_protocol_enabled(local) || !scheduler_task_requires_admission(metadata)) return true;
    const auto *admission = local->admission;
    if (admission == nullptr || admission->records_offset == 0 ||
        admission->record_stride < sizeof(SchedulerAdmissionRecord) || admission->epoch != epoch)
        return false;
    auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
        base, admission->records_offset + static_cast<uint64_t>(task) * admission->record_stride
    );
    scheduler_observe_cache_line(record);
    if (record->run_epoch != epoch || record->task_ref != task || record->order_key >= admission->capacity)
        return false;
    *order = record->order_key;
    return true;
}

}  // namespace scheduler_async_finish_detail

template <
    typename QueueOps = SchedulerCompletionQueueOps, typename Link = SchedulerResolveLink,
    typename Protocol = SchedulerAsyncFinishProtocol>
inline __aicore__ SchedulerCompletionResult scheduler_async_finish_step_impl(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerAsyncFinishIdentity &identity, __gm__ SchedulerAsyncTaskJoin *join,
    SchedulerAsyncFinishProgress *progress, uint32_t *budget, SchedulerWakeStats *wake_stats = nullptr,
    SchedulerReadyStats *ready_stats = nullptr
) {
    using namespace scheduler_completion_detail;
    if (progress == nullptr || budget == nullptr || base == nullptr || local == nullptr || run == nullptr ||
        join == nullptr)
        return SchedulerCompletionResult::ERROR;
    if (progress->contributed) return SchedulerCompletionResult::DONE;
    auto &cursor = progress->completion;
    auto failed = [&](SchedulerErrorSite site = SchedulerErrorSite::COMPLETION_READY_PUBLISH_FAILED) __aicore__ {
        if (scheduler_read_error(run) != 0) {
            phase(cursor, SchedulerCompletionPhase::FAILED);
            return SchedulerCompletionResult::ERROR;
        }
        if (progress->failure_code == 0)
            progress->failure_code = static_cast<int64_t>(SchedulerGraphResult::INVALID_ARGUMENTS);
        return fail(graph, local, run, identity.task_ref, cursor, site);
    };
    if (scheduler_read_error(run) != 0 || scheduler_async_join_load(*join) != 0) return failed();
    if (phase(cursor) == SchedulerCompletionPhase::INITIALIZE && *budget != 0) {
        if (!scheduler_task_ref_valid(graph, identity.task_ref) || identity.run_epoch == 0 ||
            local->config.graph_summary_offset == 0 ||
            scheduler_state_at<SchedulerGraphSummary>(base, local->config.graph_summary_offset)->run_epoch !=
                identity.run_epoch)
            return failed();
        const auto metadata = scheduler_load_dispatch_metadata(base, local, identity.task_ref, 0);
        uint64_t order = UINT64_MAX;
        if (!scheduler_task_is_executable(metadata.flags) || scheduler_task_is_inline(metadata.flags) ||
            !scheduler_async_finish_detail::protocol_order(
                base, local, identity.task_ref, identity.run_epoch, metadata, &order
            ))
            return failed();
        auto *control = scheduler_task_control_at(base, local, identity.task_ref);
        Protocol::task_done(base, local, identity.task_ref, order, metadata.flags, true);
        if (scheduler_read_error(run) != 0) return failed();
        if (scheduler_gm_compare_exchange(
                control->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED),
                static_cast<int64_t>(SchedulerTaskState::DONE)
            ) != static_cast<int64_t>(SchedulerTaskState::BLOCKED))
            return failed(SchedulerErrorSite::ROUTE_INVALID_STATE);
        Protocol::task_done(base, local, identity.task_ref, order, metadata.flags, false);
    }
    const auto result = scheduler_resolve_completion_step<QueueOps, Link>(
        graph, base, local, run, identity.task_ref, &cursor, budget, wake_stats, ready_stats
    );
    return result;
}

struct SchedulerAsyncFinishClock {
    inline __aicore__ static uint64_t now() { return scheduler_cycles(); }
};

inline __aicore__ void scheduler_async_finish_accumulate(uint64_t &total, uint64_t added) {
    total = added > UINT64_MAX - total ? UINT64_MAX : total + added;
}

// Every measured interval starts and ends on the same core. The old synchronous
// completion_resolve span would attribute all handoffs and gaps to its first
// worker, so async FINISH never emits that pair. Host export can consume these
// owned counters independently; no common clock origin across cores is assumed.
template <
    typename QueueOps = SchedulerCompletionQueueOps, typename Link = SchedulerResolveLink,
    typename Clock = SchedulerAsyncFinishClock, typename Protocol = SchedulerAsyncFinishProtocol>
inline __aicore__ SchedulerCompletionResult scheduler_async_finish_step(
    const SchedulerGraphAccess &graph, __gm__ void *base, SchedulerLocalState *local, __gm__ SchedulerRunControl *run,
    const SchedulerAsyncFinishIdentity &identity, __gm__ SchedulerAsyncTaskJoin *join,
    SchedulerAsyncFinishProgress *progress, uint32_t *budget, SchedulerWakeStats *wake_stats = nullptr,
    SchedulerReadyStats *ready_stats = nullptr, uint64_t profiling_level = 0, SchedulerAsyncFinishStats *stats = nullptr
) {
    const bool measured = scheduler_phase_timing_enabled(profiling_level) && progress != nullptr && stats != nullptr;
    const uint64_t start = measured ? Clock::now() : 0;
    const uint32_t initial_budget = budget == nullptr ? 0 : *budget;
    const auto result = scheduler_async_finish_step_impl<QueueOps, Link, Protocol>(
        graph, base, local, run, identity, join, progress, budget, wake_stats, ready_stats
    );
    if (measured) {
        scheduler_async_finish_accumulate(stats->active_slice_cycles, Clock::now() - start);
        scheduler_async_finish_accumulate(stats->slice_count, 1);
        scheduler_async_finish_accumulate(stats->work_units, budget == nullptr ? 0 : initial_budget - *budget);
    }
    return result;
}

// The terminal owner calls this in the same non-yielding tail after publishing
// progress, transitioning the lease to TERMINAL, and clearing its directory bit.
// It then publishes CONTRIBUTED before returning. No terminal record is claimed
// again, and no local pending-publication descriptor participates in this tail.
inline __aicore__ bool scheduler_async_finish_contribute(
    SchedulerLocalState *local, __gm__ SchedulerRunControl *run, __gm__ SchedulerAsyncTaskJoin &join,
    __gm__ SchedulerAsyncLease &lease, __gm__ SchedulerAsyncDirectoryWord &directory, uint32_t bit,
    SchedulerAsyncFinishProgress *progress, SchedulerCompletionStats *stats = nullptr
) {
    using namespace scheduler_completion_detail;
    if (local == nullptr || run == nullptr || progress == nullptr || bit >= 64) return false;
    if (progress->contributed) return true;
    if (phase(progress->completion) != SchedulerCompletionPhase::DRAINED || scheduler_read_error(run) != 0 ||
        scheduler_async_join_load(join) != 0)
        return false;
    const auto word = scheduler_async_lease_load(lease);
    if (!scheduler_async_lease_valid(word) || scheduler_async_lease_phase(word) != SchedulerAsyncPhase::TERMINAL ||
        (scheduler_async_directory_load(directory) & (UINT64_C(1) << bit)) != 0)
        return false;
    ++local->pending_completed;
    if (stats != nullptr) ++stats->resolve_count;
    progress->contributed = true;
    return true;
}
