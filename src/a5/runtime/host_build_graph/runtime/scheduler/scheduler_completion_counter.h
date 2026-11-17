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

#include "scheduler_memory.h"
#include "scheduler_protocol.h"

inline __aicore__ const __gm__ SchedulerTaskCompletionCounter *
scheduler_completion_counter_at(__gm__ void *base, uint64_t offset, int64_t task) {
    if (offset == 0) return nullptr;
    const auto *counter = scheduler_state_at<SchedulerTaskCompletionCounter>(base, offset) + task;
    return counter->add_address == 0 ? nullptr : counter;
}

inline __aicore__ int scheduler_completion_counter_observe(const __gm__ SchedulerTaskCompletionCounter &counter) {
    const uint32_t value = scheduler_notify_query(*reinterpret_cast<__gm__ volatile uint32_t *>(counter.read_address));
    // A later task in this chain can legitimately exceed our threshold.
    if (value > counter.limit) return -1;
    return value >= counter.threshold ? 1 : 0;
}

inline __aicore__ void
scheduler_completion_counter_add(const __gm__ SchedulerTaskCompletionCounter &counter, uint32_t credits) {
    scheduler_notify_add(*reinterpret_cast<__gm__ volatile uint32_t *>(counter.add_address), credits);
}

struct SchedulerCounterProtocolActor {
    struct Profile {
        uint64_t protocol_layout_offset;
    } profile;
    Profile *profiling;
    uint32_t index;
    inline __aicore__ SchedulerCounterProtocolActor(uint64_t offset, uint32_t worker) :
        profile{offset},
        profiling(&profile),
        index(worker) {}
    inline __aicore__ uint32_t worker_id() const { return index; }
};

// Only async-capable empty MIX seals and successful CHECK owners publish Q.
// Known-synchronous executions have only P. An async-capable ordinary execution
// with no registered condition combines P and Q in its physical-return interval.
inline __aicore__ bool scheduler_counter_publish_condition(
    __gm__ void *base, const __gm__ SchedulerTaskCompletionCounter &counter, int64_t task, uint32_t block,
    uint32_t lane, uint64_t metadata_offset, uint64_t protocol_offset, uint32_t worker, bool acquired
) {
    SchedulerCounterProtocolActor actor(protocol_offset, worker);
    auto *layout = scheduler_protocol_layout_at(base, &actor);
    __gm__ SchedulerProtocolConditionContribution *event = nullptr;
    if (layout != nullptr) {
        const auto &metadata = scheduler_state_at<SchedulerTaskMetadata>(base, metadata_offset)[task];
        const uint32_t mask = scheduler_task_lanes(metadata).raw();
        if (lane >= 3 || (mask & (1U << lane)) == 0 || block >= metadata.logical_block_num) return false;
        const uint64_t index = metadata.trace_index_base + block * __builtin_popcount(mask) +
                               __builtin_popcount(mask & ((1U << lane) - 1));
        if (index >= layout->trace_count) return false;
        event =
            scheduler_state_at<SchedulerProtocolConditionContribution>(base, layout->condition_contributions_offset) +
            index;
        scheduler_protocol_observe_record(event);
        if (event->before.sequence != 0 || !scheduler_protocol_stamp(base, &actor, layout, &event->before))
            return false;
        event->run_epoch = layout->run_epoch;
        event->task_ref = task;
        event->worker_index = worker;
        event->block_idx = static_cast<uint16_t>(block);
        event->lane = static_cast<uint8_t>(lane);
        event->source = acquired ? 1 : 0;
        event->credits = 1;
    }
    scheduler_cache_barrier();
    scheduler_completion_counter_add(counter, 1);
    if (event != nullptr) {
        if (!scheduler_protocol_stamp(base, &actor, layout, &event->after)) return false;
        scheduler_protocol_publish_record(event);
    }
    return true;
}
