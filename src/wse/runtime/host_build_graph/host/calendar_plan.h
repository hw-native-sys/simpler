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
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include "launch.h"

namespace wse {

struct CalendarTask {
    // Empty for Graph controls and inline-completed tasks.
    std::vector<uint32_t> block_cores;
    std::vector<uint32_t> waits;
    bool sync_start{false};
};

struct CoreCalendar {
    std::vector<uint32_t> indices;
    std::vector<CalendarRow> rows;
};

namespace calendar_detail {
struct ReadyBlock {
    uint32_t task, block, data_ready, priority;
};

struct LowerPriority {
    bool operator()(const ReadyBlock &a, const ReadyBlock &b) const {
        if (a.priority != b.priority) return a.priority < b.priority;
        if (a.task != b.task) return a.task > b.task;
        return a.block > b.block;
    }
};

struct LaterData {
    bool operator()(const ReadyBlock &a, const ReadyBlock &b) const {
        if (a.data_ready != b.data_ready) return a.data_ready > b.data_ready;
        return LowerPriority{}(a, b);
    }
};

struct ReadyQueue {
    std::priority_queue<ReadyBlock, std::vector<ReadyBlock>, LaterData> future;
    std::priority_queue<ReadyBlock, std::vector<ReadyBlock>, LowerPriority> available;

    const ReadyBlock *next(uint32_t free_at) {
        while (!future.empty() && future.top().data_ready <= free_at) {
            available.push(future.top());
            future.pop();
        }
        if (!available.empty()) return &available.top();
        return future.empty() ? nullptr : &future.top();
    }

    void pop() {
        if (available.empty()) future.pop();
        else available.pop();
    }
    bool empty() const { return future.empty() && available.empty(); }
};

struct CalendarCore {
    uint32_t free_at{0};
    ReadyQueue ordinary;
    std::vector<CalendarRow> rows;

    void append(uint32_t task, uint32_t block, uint32_t tick) {
        rows.push_back({task, block, tick});
        free_at = tick + 1;
    }
};

struct TaskState {
    std::vector<uint32_t> successors;
    uint32_t pending_waits{0}, pending_blocks{0};
    uint32_t data_ready{0}, finish{0}, priority{0};
};
}  // namespace calendar_detail

// Select the earliest runnable Block across Core calendars, breaking ties by
// the longest remaining dependency path. One unit occupies its assigned Core
// for the entire Block. A successor's data is ready after the final Block.
inline bool build_core_calendar(const std::vector<CalendarTask> &tasks, uint32_t core_count, CoreCalendar &out) {
    using namespace calendar_detail;
    if (!core_count || core_count > MAX_CORE_COUNT || tasks.size() > INT32_MAX) return false;
    std::vector<TaskState> states(tasks.size());
    std::vector<uint32_t> occupancy(core_count);
    uint32_t row_count = 0;
    for (uint32_t id = 0; id < tasks.size(); ++id) {
        const auto &task = tasks[id];
        if (task.block_cores.size() > UINT32_MAX - row_count || task.waits.size() > UINT32_MAX) return false;
        row_count += task.block_cores.size();
        auto &state = states[id];
        state.pending_waits = task.waits.size();
        state.pending_blocks = task.block_cores.size();
        if (task.sync_start && state.pending_blocks < 2) return false;
        for (uint32_t producer : task.waits) {
            if (producer >= id) return false;
            states[producer].successors.push_back(id);
        }
        // Blocks mapped to one Core serialize even if every input is available.
        for (uint32_t core : task.block_cores) {
            if (core >= core_count) return false;
            if (task.sync_start && occupancy[core]) return false;
            state.priority = std::max(state.priority, ++occupancy[core]);
        }
        for (uint32_t core : task.block_cores)
            occupancy[core] = 0;
    }
    for (size_t id = tasks.size(); id-- > 0;) {
        uint32_t tail = 0;
        for (uint32_t successor : states[id].successors)
            tail = std::max(tail, states[successor].priority);
        states[id].priority += tail;
    }

    std::vector<CalendarCore> cores(core_count);
    // Equal Core sets have the same resource availability, regardless of Block ID order.
    std::map<std::vector<uint32_t>, ReadyQueue> sync_queues;
    std::queue<uint32_t> ready;
    for (uint32_t id = 0; id < tasks.size(); ++id) {
        if (!states[id].pending_waits) ready.push(id);
    }
    auto release_successors = [&](uint32_t id) {
        for (uint32_t successor : states[id].successors) {
            auto &state = states[successor];
            state.data_ready = std::max(state.data_ready, states[id].finish);
            if (--state.pending_waits == 0) ready.push(successor);
        }
    };

    uint32_t scheduled = 0;
    while (!ready.empty() || scheduled < row_count) {
        while (!ready.empty()) {
            const uint32_t id = ready.front();
            ready.pop();
            auto &state = states[id];
            if (!state.pending_blocks) {
                state.finish = state.data_ready;
                release_successors(id);
                continue;
            }
            const auto &placements = tasks[id].block_cores;
            if (tasks[id].sync_start) {
                auto core_set = placements;
                std::sort(core_set.begin(), core_set.end());
                sync_queues[std::move(core_set)].future.push({id, 0, state.data_ready, state.priority});
            } else {
                for (uint32_t block = 0; block < placements.size(); ++block)
                    cores[placements[block]].ordinary.future.push({id, block, state.data_ready, state.priority});
            }
        }
        if (scheduled == row_count) break;
        ReadyQueue *selected = nullptr;
        ReadyBlock chosen{};
        uint32_t tick = UINT32_MAX;
        auto consider = [&](ReadyQueue &queue, uint32_t free_at) {
            const ReadyBlock *candidate = queue.next(free_at);
            if (!candidate) return false;
            const uint32_t start = std::max(free_at, candidate->data_ready);
            if (selected && (start > tick || (start == tick && !LowerPriority{}(chosen, *candidate)))) return false;
            selected = &queue;
            chosen = *candidate;
            tick = start;
            return true;
        };
        for (auto &core : cores)
            consider(core.ordinary, core.free_at);
        auto selected_sync = sync_queues.end();
        for (auto it = sync_queues.begin(); it != sync_queues.end(); ++it) {
            uint32_t free_at = 0;
            for (uint32_t core : it->first)
                free_at = std::max(free_at, cores[core].free_at);
            if (consider(it->second, free_at)) selected_sync = it;
        }
        if (!selected) return false;
        selected->pop();
        const auto &placements = tasks[chosen.task].block_cores;
        uint32_t placed = 1;
        if (selected_sync != sync_queues.end()) {
            // Commit all rows together so competing tasks have one order on their shared Cores.
            placed = placements.size();
            for (uint32_t block = 0; block < placed; ++block)
                cores[placements[block]].append(chosen.task, block, tick);
            if (selected->empty()) sync_queues.erase(selected_sync);
        } else {
            cores[placements[chosen.block]].append(chosen.task, chosen.block, tick);
        }
        auto &state = states[chosen.task];
        state.finish = std::max(state.finish, tick + 1);
        state.pending_blocks -= placed;
        if (!state.pending_blocks) release_successors(chosen.task);
        scheduled += placed;
    }

    CoreCalendar calendar;
    calendar.indices.reserve(core_count + 1);
    calendar.rows.reserve(row_count);
    for (const auto &core : cores) {
        calendar.indices.push_back(calendar.rows.size());
        calendar.rows.insert(calendar.rows.end(), core.rows.begin(), core.rows.end());
    }
    calendar.indices.push_back(row_count);
    out = std::move(calendar);
    return true;
}

}  // namespace wse
