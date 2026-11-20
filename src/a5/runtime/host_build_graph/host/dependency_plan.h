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

#include <string>
#include <utility>
#include <vector>

#include "scheduler/scheduler_graph.h"

struct SchedulerDependencyPlan {
    std::vector<std::vector<int64_t>> fanin;
    std::vector<uint32_t> sinks;
    std::vector<bool> host_done;

    void collect_sinks() {
        std::vector<bool> consumed(fanin.size(), false);
        for (const auto &row : fanin)
            for (const auto ref : row)
                consumed[ref] = true;
        sinks.clear();
        for (uint32_t ref = 0; ref < fanin.size(); ++ref)
            if (!host_done[ref] && !consumed[ref]) sinks.push_back(ref);
    }
};

// Check the whole ordinary fan-in row against the pool before reading any entry.
struct SchedulerOrdinaryFaninPool {
    int32_t *data;
    uint64_t capacity;

    bool contains(const int32_t *row, uint32_t count) const {
        const uintptr_t base = reinterpret_cast<uintptr_t>(data);
        const uintptr_t address = reinterpret_cast<uintptr_t>(row);
        if (capacity > (UINTPTR_MAX - base) / sizeof(int32_t) ||
            (capacity != 0 && (data == nullptr || base % alignof(int32_t) != 0)))
            return false;
        return count == 0 ||
               (address >= base && address % alignof(int32_t) == 0 && (address - base) / sizeof(int32_t) <= capacity &&
                count <= capacity - (address - base) / sizeof(int32_t));
    }
};

inline SchedulerGraphResult scheduler_classify_host_task(
    const SchedulerGraphAccess &graph, int64_t ref, SchedulerTaskShape *shape, const SchedulerOrdinaryFaninPool &pool,
    std::vector<int64_t> *decoded
) {
    SchedulerFaninView view;
    const auto status = scheduler_node_fanin_view(graph, ref, &view);
    if (status != SchedulerGraphResult::OK) return status;
    if (!pool.contains(view.global_refs, view.count)) return SchedulerGraphResult::INVALID_FANIN_ID;
    if (decoded != nullptr) decoded->resize(view.count);
    return scheduler_classify_task_shape(graph, ref, shape, &view, decoded == nullptr ? nullptr : decoded->data());
}

// Only a device-live consumer waits for its incoming edges. This also makes
// a folded terminal join's producers sinks in the rewritten ordinary graph.
// The native fanin view exposes enter, body, exit and top-level edges together.
inline bool scheduler_plan_dependencies(
    const SchedulerGraphAccess &graph, const std::vector<SchedulerTaskMetadata> &metadata,
    const std::vector<int64_t> &completed, SchedulerDependencyPlan &plan, std::string *error = nullptr
) {
    const uint64_t count = graph.task_count;
    if (metadata.size() != count || count > INT32_MAX) return false;
    const auto fail_at = [&](const char *message, int64_t task) {
        if (error != nullptr) *error = message + std::to_string(task);
        return false;
    };
    std::vector<bool> host_done(count, false);
    for (int64_t ref : completed) {
        if (!scheduler_task_ref_valid(graph, ref) || host_done[ref]) return fail_at("invalid completed task ", ref);
        host_done[ref] = true;
    }
    SchedulerDependencyPlan next;
    next.fanin.resize(count);
    for (uint32_t consumer = 0; consumer < count; ++consumer) {
        if (host_done[consumer]) continue;
        SchedulerFaninView view;
        if (scheduler_node_fanin_view(graph, consumer, &view) != SchedulerGraphResult::OK)
            return fail_at("invalid fanin view at task ", consumer);
        auto &row = next.fanin[consumer];
        row.reserve(view.count);
        for (uint32_t edge = 0; edge < view.count; ++edge) {
            int64_t producer;
            if (scheduler_fanin_ref(view, consumer, edge, &producer) != SchedulerGraphResult::OK)
                return fail_at("invalid data predecessor at task ", consumer);
            row.push_back(producer);
        }
    }
    next.host_done = std::move(host_done);
    next.collect_sinks();
    plan = std::move(next);
    return true;
}
