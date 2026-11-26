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
#include <queue>
#include <utility>
#include <vector>

#include "launch.h"

namespace wse {

struct CalendarTask {
    // Empty for Graph controls and inline-completed tasks.
    std::vector<uint32_t> block_cores;
    std::vector<uint32_t> waits;
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

struct CalendarCore {
    uint32_t free_at{0};
    std::priority_queue<ReadyBlock, std::vector<ReadyBlock>, LaterData> future;
    std::priority_queue<ReadyBlock, std::vector<ReadyBlock>, LowerPriority> available;
    std::vector<CalendarRow> rows;

    const ReadyBlock *next() {
        while (!future.empty() && future.top().data_ready <= free_at) {
            available.push(future.top());
            future.pop();
        }
        if (!available.empty()) return &available.top();
        return future.empty() ? nullptr : &future.top();
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
        for (uint32_t producer : task.waits) {
            if (producer >= id) return false;
            states[producer].successors.push_back(id);
        }
        // Blocks mapped to one Core serialize even if every input is available.
        for (uint32_t core : task.block_cores) {
            if (core >= core_count) return false;
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
            }
            const auto &placements = tasks[id].block_cores;
            for (uint32_t block = 0; block < placements.size(); ++block)
                cores[placements[block]].future.push({id, block, state.data_ready, state.priority});
        }
        if (scheduled == row_count) break;
        CalendarCore *selected = nullptr;
        ReadyBlock chosen{};
        uint32_t tick = UINT32_MAX;
        for (auto &core : cores) {
            const ReadyBlock *candidate = core.next();
            if (!candidate) continue;
            const uint32_t start = std::max(core.free_at, candidate->data_ready);
            if (!selected || start < tick || (start == tick && LowerPriority{}(chosen, *candidate))) {
                selected = &core;
                chosen = *candidate;
                tick = start;
            }
        }
        if (!selected) return false;
        if (selected->available.empty()) selected->future.pop();
        else selected->available.pop();
        selected->rows.push_back({chosen.task, chosen.block, tick});
        selected->free_at = tick + 1;
        auto &state = states[chosen.task];
        state.finish = std::max(state.finish, selected->free_at);
        if (--state.pending_blocks == 0) release_successors(chosen.task);
        ++scheduled;
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
