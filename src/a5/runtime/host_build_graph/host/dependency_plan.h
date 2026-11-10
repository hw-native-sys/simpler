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
#include <vector>

#include "scheduler/scheduler_dependency.h"
#include "scheduler/scheduler_graph.h"

struct SchedulerDependencyPlan {
    uint64_t header_offset{0};
    uint64_t total_size{0};
    SchedulerDependencyHeader header{};
};

inline bool scheduler_plan_dependencies(
    uint64_t task_count, uint64_t edge_capacity, uint64_t start_offset, SchedulerDependencyPlan *plan
) {
    if (plan == nullptr || task_count > INT32_MAX || start_offset == 0 || edge_capacity > UINT64_MAX / sizeof(int32_t))
        return false;
    SchedulerDependencyPlan next;
    next.total_size = start_offset;
    if (edge_capacity == 0) {
        *plan = next;
        return true;
    }
    auto &header = next.header;
    header.task_count = task_count;
    header.edge_capacity = edge_capacity;
    uint64_t cursor = start_offset;
    if (!scheduler_layout_reserve(&cursor, sizeof(header), SCHEDULER_STATE_ALIGNMENT, &next.header_offset) ||
        !scheduler_layout_reserve(
            &cursor, task_count * sizeof(uint64_t), SCHEDULER_STATE_ALIGNMENT, &header.remaining_offset
        ) ||
        !scheduler_layout_reserve(
            &cursor, (task_count + 1) * sizeof(uint64_t), SCHEDULER_STATE_ALIGNMENT, &header.fanout_offsets_offset
        ) ||
        !scheduler_layout_reserve(
            &cursor, edge_capacity * sizeof(int32_t), SCHEDULER_STATE_ALIGNMENT, &header.fanout_refs_offset
        ) ||
        !scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size))
        return false;
    header.storage_end_offset = next.total_size;
    *plan = next;
    return true;
}

// Input edges are validated through the same native/ordinary graph view used
// by the resident scheduler. Host-completed nodes do not issue device credits.
inline bool scheduler_init_dependencies(
    void *base, const SchedulerDependencyPlan &plan, const SchedulerGraphAccess &graph,
    const std::vector<int64_t> &completed
) {
    if (plan.header_offset == 0) return true;
    if (base == nullptr || plan.header.task_count != static_cast<uint64_t>(graph.task_count)) return false;
    auto *header = scheduler_state_at<SchedulerDependencyHeader>(base, plan.header_offset);
    *header = plan.header;
    auto *remaining = scheduler_state_at<uint64_t>(base, header->remaining_offset);
    auto *offsets = scheduler_state_at<uint64_t>(base, header->fanout_offsets_offset);
    auto *refs = scheduler_state_at<int32_t>(base, header->fanout_refs_offset);
    std::vector<uint8_t> active(graph.task_count, 1);
    for (int64_t ref : completed) {
        if (!scheduler_task_ref_valid(graph, ref)) return false;
        active[ref] = 0;
    }
    std::fill(remaining, remaining + graph.task_count, 0);
    std::fill(offsets, offsets + graph.task_count + 1, 0);
    std::vector<int64_t> fanins;
    auto read_fanins = [&](int64_t consumer) {
        fanins.clear();
        SchedulerFaninView view;
        if (scheduler_node_fanin_view(graph, consumer, &view) != SchedulerGraphResult::OK) return false;
        for (uint32_t i = 0; i < view.count; ++i) {
            int64_t producer;
            if (scheduler_fanin_ref(view, consumer, i, &producer) != SchedulerGraphResult::OK) return false;
            if (active[producer]) fanins.push_back(producer);
        }
        std::sort(fanins.begin(), fanins.end());
        fanins.erase(std::unique(fanins.begin(), fanins.end()), fanins.end());
        return true;
    };
    for (int64_t consumer = 0; consumer < graph.task_count; ++consumer) {
        if (!active[consumer]) continue;
        if (!read_fanins(consumer) || fanins.size() > header->edge_capacity - header->edge_count) return false;
        remaining[consumer] = fanins.size();
        header->edge_count += fanins.size();
        for (int64_t producer : fanins)
            ++offsets[producer + 1];
    }
    for (int64_t ref = 0; ref < graph.task_count; ++ref)
        offsets[ref + 1] += offsets[ref];
    std::vector<uint64_t> cursors(offsets, offsets + graph.task_count);
    for (int64_t consumer = 0; consumer < graph.task_count; ++consumer) {
        if (!active[consumer]) continue;
        if (!read_fanins(consumer)) return false;
        for (int64_t producer : fanins)
            refs[cursors[producer]++] = static_cast<int32_t>(consumer);
    }
    return offsets[graph.task_count] == header->edge_count;
}
