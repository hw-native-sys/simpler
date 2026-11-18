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
#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <queue>
#include <set>
#include <tuple>
#include <utility>
#include <vector>

namespace simpler::percounter {

struct ResourceOrderTask {
    std::vector<uint32_t> predecessors;
    uint32_t blocks{1};
    uint8_t active_mask{0};
    bool sync{false};
    bool reserve_types{false};
    uint64_t duration{1};
    uint64_t depth{1};
};

struct ResourceOrderPlan {
    std::vector<uint32_t> tasks;
    std::vector<uint64_t> first_start;
    std::vector<uint64_t> finish;
    uint64_t duration{0};
    uint64_t depth{0};
    uint8_t active_mask{0};
};

// Times are virtual work units. Worker identities never leave this Host plan.
inline std::optional<ResourceOrderPlan>
make_resource_order(const std::vector<ResourceOrderTask> &tasks, uint32_t clusters) {
    const size_t n = tasks.size();
    if (clusters == 0 || clusters > UINT32_MAX / 3 || n > UINT32_MAX) return std::nullopt;
    ResourceOrderPlan plan;
    plan.tasks.reserve(n);
    plan.first_start.assign(n, UINT64_MAX);
    plan.finish.assign(n, UINT64_MAX);
    if (n == 0) return plan;
    std::vector<std::vector<uint32_t>> successors(n);
    std::vector<uint32_t> indegree(n), remaining(n), inflight(n, 0);
    std::vector<uint64_t> level(n, 0);
    for (uint32_t id = 0; id < n; ++id) {
        const auto &task = tasks[id];
        if (task.blocks == 0 || task.depth == 0 || task.active_mask > 7 || task.active_mask == 4 ||
            task.predecessors.size() > UINT32_MAX || (task.reserve_types && (task.blocks != 1 || task.sync)))
            return std::nullopt;
        if (!task.reserve_types && task.duration != (task.active_mask == 0 ? 0U : 1U)) return std::nullopt;
        const uint32_t width = task.active_mask == 2 ? clusters * 2 : clusters;
        if (task.sync && (task.active_mask == 0 || task.blocks > width)) return std::nullopt;
        remaining[id] = task.blocks;
        indegree[id] = static_cast<uint32_t>(task.predecessors.size());
        plan.active_mask |= task.active_mask;
        for (uint32_t producer : task.predecessors) {
            if (producer >= id) return std::nullopt;
            successors[producer].push_back(id);
        }
    }
    for (size_t id = n; id-- > 0;) {
        if (level[id] > UINT64_MAX - tasks[id].depth) return std::nullopt;
        level[id] += tasks[id].depth - 1;
        plan.depth = std::max(plan.depth, level[id] + 1);
        for (uint32_t producer : tasks[id].predecessors)
            level[producer] = std::max(level[producer], level[id] + 1);
    }
    struct Priority {
        const std::vector<uint64_t> *level;
        bool operator()(uint32_t a, uint32_t b) const {
            return (*level)[a] != (*level)[b] ? (*level)[a] > (*level)[b] : a < b;
        }
    };
    const Priority priority{&level};
    // A blocked wide sync task does not hide a narrower ready task of its type.
    using Key = std::tuple<uint8_t, bool, uint32_t>;
    using Buckets = std::map<Key, std::set<uint32_t, Priority>>;
    Buckets ready;
    std::vector<Buckets::iterator> bucket(n);
    for (uint32_t id = 0; id < n; ++id) {
        const auto &task = tasks[id];
        const Key key{task.active_mask, task.reserve_types, task.sync ? task.blocks : 0};
        bucket[id] = ready.try_emplace(key, priority).first;
        if (indegree[id] == 0) bucket[id]->second.insert(id);
    }
    std::vector<uint8_t> available(clusters, 7);
    const auto reserved_mask = [](uint8_t mask) {
        return static_cast<uint8_t>((mask & 1U) | ((mask & 6U) ? 6U : 0U));
    };
    const auto capacity = [&](const ResourceOrderTask &task) {
        if (task.active_mask == 0) return task.blocks;
        if (task.reserve_types) {
            const uint8_t mask = reserved_mask(task.active_mask);
            return std::all_of(
                       available.begin(), available.end(),
                       [mask](uint8_t free) {
                           return (free & mask) == mask;
                       }
                   ) ?
                       1U :
                       0U;
        }
        uint32_t count = 0;
        for (uint8_t free : available) {
            if (task.active_mask == 1) count += (free & 1U) != 0;
            else if (task.active_mask == 2) count += static_cast<uint32_t>(__builtin_popcount(free & 6U));
            else count += (free & task.active_mask) == task.active_mask;
        }
        return count;
    };
    const auto allocate = [&](const ResourceOrderTask &task, uint32_t blocks) {
        std::vector<uint32_t> workers;
        if (task.active_mask == 0) return workers;
        const auto take = [&](uint32_t cluster, uint32_t lane) {
            available[cluster] &= static_cast<uint8_t>(~(1U << lane));
            workers.push_back(cluster * 3 + lane);
        };
        if (task.reserve_types) {
            const uint8_t mask = reserved_mask(task.active_mask);
            for (uint32_t c = 0; c < clusters; ++c)
                for (uint32_t lane = 0; lane < 3; ++lane)
                    if (mask & (1U << lane)) take(c, lane);
        } else if (task.active_mask == 1 || task.active_mask == 2) {
            const uint32_t first = task.active_mask == 1 ? 0 : 1;
            const uint32_t end = task.active_mask == 1 ? 1 : 3;
            for (uint32_t lane = first; lane < end && blocks != 0; ++lane) {
                for (uint32_t c = 0; c < clusters && blocks != 0; ++c) {
                    if (!(available[c] & (1U << lane))) continue;
                    take(c, lane);
                    --blocks;
                }
            }
        } else {
            for (uint32_t c = 0; c < clusters && blocks != 0; ++c) {
                if ((available[c] & task.active_mask) != task.active_mask) continue;
                for (uint32_t lane = 0; lane < 3; ++lane)
                    if (task.active_mask & (1U << lane)) take(c, lane);
                --blocks;
            }
        }
        return workers;
    };
    struct Completion {
        uint64_t time;
        uint32_t task;
        uint32_t blocks;
        std::vector<uint32_t> workers;
    };
    struct EarlierCompletion {
        bool operator()(const Completion &a, const Completion &b) const { return a.time > b.time; }
    };
    std::priority_queue<Completion, std::vector<Completion>, EarlierCompletion> completions;
    uint64_t now = 0;
    size_t finished = 0;
    const auto complete = [&](uint32_t task) {
        ++finished;
        plan.finish[task] = now;
        for (uint32_t consumer : successors[task])
            if (--indegree[consumer] == 0) bucket[consumer]->second.insert(consumer);
    };
    while (finished < n) {
        uint32_t selected = UINT32_MAX;
        uint32_t count = 0;
        for (const auto &entry : ready) {
            if (entry.second.empty()) continue;
            const uint32_t id = *entry.second.begin();
            const uint32_t free = capacity(tasks[id]);
            if (free == 0 || (tasks[id].sync && free < remaining[id])) continue;
            if (selected == UINT32_MAX || priority(id, selected)) {
                selected = id;
                count = std::min(free, remaining[id]);
            }
        }
        if (selected == UINT32_MAX) {
            if (completions.empty()) return std::nullopt;
            now = completions.top().time;
            do {
                const Completion event = completions.top();
                completions.pop();
                for (uint32_t worker : event.workers)
                    available[worker / 3] |= static_cast<uint8_t>(1U << (worker % 3));
                inflight[event.task] -= event.blocks;
                if (remaining[event.task] == 0 && inflight[event.task] == 0) complete(event.task);
            } while (!completions.empty() && completions.top().time == now);
            continue;
        }
        const auto &task = tasks[selected];
        if (plan.first_start[selected] == UINT64_MAX) {
            plan.first_start[selected] = now;
            plan.tasks.push_back(selected);
        }
        auto workers = allocate(task, count);
        remaining[selected] -= count;
        if (remaining[selected] == 0) bucket[selected]->second.erase(selected);
        if (task.duration == 0) {
            for (uint32_t worker : workers)
                available[worker / 3] |= static_cast<uint8_t>(1U << (worker % 3));
            if (remaining[selected] == 0 && inflight[selected] == 0) complete(selected);
        } else {
            if (now > UINT64_MAX - task.duration) return std::nullopt;
            inflight[selected] += count;
            completions.push(Completion{now + task.duration, selected, count, std::move(workers)});
        }
    }
    plan.duration = now;
    return plan;
}

}  // namespace simpler::percounter
