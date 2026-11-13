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

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "scheduler/scheduler_ready_pool.h"

struct SchedulerReadyPoolPlan {
    uint64_t header_offset{0};
    uint64_t total_size{0};
    SchedulerReadyPoolHeader header{};
    std::vector<SchedulerReadyPoolDescriptor> descriptors;
};

inline bool scheduler_plan_ready_pool(
    const std::vector<SchedulerTaskMetadata> &metadata, uint32_t clusters, uint64_t start_offset,
    SchedulerReadyPoolPlan *plan
) {
    if (plan == nullptr || metadata.size() > INT32_MAX || clusters == 0 || clusters > SCHEDULER_CLUSTER_CAPACITY ||
        start_offset == 0)
        return false;
    SchedulerReadyPoolPlan next{};
    auto &header = next.header;
    header.task_count = metadata.size();
    next.descriptors.resize(SCHEDULER_READY_QUEUE_COUNT);
    for (uint64_t ref = 0; ref < metadata.size(); ++ref) {
        const auto &task = metadata[ref];
        if (!scheduler_task_is_executable(task.flags)) continue;
        const uint32_t type = scheduler_task_ready_queue(task.flags, task.active_mask);
        if (type >= SCHEDULER_READY_QUEUE_COUNT) return false;
        ++next.descriptors[type].capacity;
        ++header.entry_count;
    }
    uint64_t cursor = start_offset;
    if (!scheduler_layout_reserve(&cursor, sizeof(header), SCHEDULER_STATE_ALIGNMENT, &next.header_offset) ||
        !scheduler_layout_reserve(
            &cursor, next.descriptors.size() * sizeof(SchedulerReadyPoolDescriptor), SCHEDULER_STATE_ALIGNMENT,
            &header.descriptors_offset
        ) ||
        !scheduler_layout_reserve(
            &cursor, next.descriptors.size() * sizeof(SchedulerTicketQueue), alignof(SchedulerTicketQueue),
            &header.controls_offset
        ))
        return false;
    for (auto &descriptor : next.descriptors)
        if (!scheduler_layout_reserve(
                &cursor, descriptor.capacity * sizeof(SchedulerTicketEntry), alignof(SchedulerTicketEntry),
                &descriptor.entries_offset
            ))
            return false;
    if (!scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size)) return false;
    header.storage_end_offset = next.total_size;
    *plan = std::move(next);
    return true;
}

// The caller owns a quiescent allocation already zeroed through total_size.
inline void scheduler_init_ready_pool(void *base, const SchedulerReadyPoolPlan &plan) {
    *scheduler_state_at<SchedulerReadyPoolHeader>(base, plan.header_offset) = plan.header;
    auto *descriptors = scheduler_state_at<SchedulerReadyPoolDescriptor>(base, plan.header.descriptors_offset);
    std::copy(plan.descriptors.begin(), plan.descriptors.end(), descriptors);
    for (const auto &descriptor : plan.descriptors) {
        auto *entries = scheduler_state_at<SchedulerTicketEntry>(base, descriptor.entries_offset);
        for (uint64_t index = 0; index < descriptor.capacity; ++index)
            entries[index].task_ref = SCHEDULER_TASK_REF_INVALID;
    }
}
