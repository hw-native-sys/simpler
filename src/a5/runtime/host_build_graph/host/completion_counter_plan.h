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

#include <cstddef>
#include <cstdint>
#include <queue>
#include <string>
#include <utility>
#include <vector>

struct SchedulerCompletionCounterPlan {
    struct Chain {
        int64_t last_task{-1};
        uint32_t limit{0};
    };
    std::vector<Chain> chains;
    std::vector<uint32_t> task_chains;
    std::vector<uint32_t> thresholds;
};

// Only genuine ancestors share a monotonic counter. Independent tasks retain
// separate counters; the plan adds no execution ordering or device handoff.
// order is a complete topological permutation of [0, fanin.size()).
inline bool scheduler_plan_completion_counters(
    const std::vector<std::vector<int64_t>> &fanin, const std::vector<int64_t> &order,
    const std::vector<uint32_t> &credits, SchedulerCompletionCounterPlan *out, std::string *error = nullptr
) {
    const auto fail = [&](const char *message) {
        if (error != nullptr) *error = message;
        return false;
    };
    const size_t count = fanin.size();
    if (out == nullptr || count > INT32_MAX || order.size() != count || credits.size() != count)
        return fail("invalid completion counter DAG dimensions");
    SchedulerCompletionCounterPlan plan;
    plan.task_chains.assign(count, UINT32_MAX);
    plan.thresholds.assign(count, 0);
    std::vector<bool> seen(count, false);
    std::vector<size_t> ranks(count, 0);
    std::vector<size_t> visited(count, count);
    for (size_t rank = 0; rank < count; ++rank) {
        const int64_t task = order[rank];
        if (task < 0 || static_cast<uint64_t>(task) >= count || seen[task])
            return fail("invalid completion counter task order");
        for (const int64_t predecessor : fanin[task]) {
            if (predecessor < 0 || static_cast<uint64_t>(predecessor) >= count || !seen[predecessor])
                return fail("completion counter predecessor does not precede its consumer");
        }
        seen[task] = true;
        ranks[task] = rank;
        const uint32_t contribution = credits[task];
        if (contribution == 0) continue;
        uint32_t selected = UINT32_MAX;
        // Every predecessor has lower rank. Descending traversal therefore
        // finds the latest reusable chain tip before any older candidate.
        std::priority_queue<size_t> pending;
        const auto enqueue = [&](int64_t predecessor) {
            if (visited[predecessor] == rank) return;
            visited[predecessor] = rank;
            pending.push(ranks[predecessor]);
        };
        for (const int64_t predecessor : fanin[task])
            enqueue(predecessor);
        while (!pending.empty()) {
            const int64_t ancestor = order[pending.top()];
            pending.pop();
            const uint32_t index = plan.task_chains[ancestor];
            if (index != UINT32_MAX && plan.chains[index].last_task == ancestor &&
                static_cast<uint64_t>(plan.chains[index].limit) + contribution <= UINT32_MAX) {
                selected = index;
                break;
            }
            for (const int64_t predecessor : fanin[ancestor])
                enqueue(predecessor);
        }
        if (selected == UINT32_MAX) {
            selected = static_cast<uint32_t>(plan.chains.size());
            plan.chains.push_back({});
        }
        auto &chain = plan.chains[selected];
        chain.last_task = task;
        chain.limit += contribution;
        plan.task_chains[task] = selected;
        plan.thresholds[task] = chain.limit;
    }
    *out = std::move(plan);
    if (error != nullptr) error->clear();
    return true;
}
