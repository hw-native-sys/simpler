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
#include <numeric>
#include <queue>
#include <vector>

#include "calendar_phases.h"
#include "scheduler/calendar_wire.h"

namespace simpler::percounter {

struct CalendarCompletionPlan {
    struct TaskCounter {
        uint32_t slot{UINT32_MAX};
        uint32_t target{0};
        uint32_t defer_bit{0};
    };
    std::vector<TaskCounter> tasks;
    uint32_t slot_count{0};
};

namespace range_calendar_detail {

class CompletionChains {
public:
    explicit CompletionChains(CalendarCompletionPlan &plan) :
        plan_(plan) {}

    bool append_phase(const CalendarPhase &phase, uint32_t body_base, uint32_t shell, uint64_t sink_credits) {
        if (phase.tasks.empty() || phase.task_ids.size() != phase.tasks.size() ||
            phase.fanins.size() != phase.tasks.size())
            return false;
        available_.resize(chains_.size());
        std::iota(available_.begin(), available_.end(), uint32_t{0});
        if (shell != UINT32_MAX && !append(shell, UINT32_MAX, sink_credits, false, UINT32_MAX)) return false;

        std::vector<uint32_t> slots(phase.tasks.size(), UINT32_MAX);
        std::vector<uint32_t> visited(phase.tasks.size(), UINT32_MAX);
        for (uint32_t task = 0; task < phase.tasks.size(); ++task) {
            const auto &shape = phase.tasks[task];
            if (shape.block_num <= 0 || shape.subtask_count <= 0 || phase.task_ids[task] > UINT32_MAX - body_base)
                return false;
            for (int32_t pred : phase.fanins[task])
                if (pred < 0 || static_cast<uint32_t>(pred) >= task) return false;
            const uint64_t credits = uint64_t{static_cast<uint32_t>(shape.block_num)} * shape.subtask_count;
            const bool deferred = shape.active_mask != 0;
            const uint32_t slot = find_ancestor(phase, slots, visited, task, credits, deferred);
            const uint32_t global = body_base + phase.task_ids[task];
            if (!append(global, task, credits, deferred, slot)) return false;
            slots[task] = plan_.tasks[global].slot;
        }
        plan_.slot_count = static_cast<uint32_t>(chains_.size());
        return true;
    }

private:
    struct Chain {
        uint32_t tip{UINT32_MAX};
        uint32_t target{0};
        uint32_t deferred_count{0};
    };

    bool fits(uint32_t slot, uint64_t credits, bool deferred) const {
        const auto &chain = chains_[slot];
        return credits <= CALENDAR_COUNT_MASK - chain.target && (!deferred || chain.deferred_count < 8);
    }

    uint32_t find_ancestor(
        const CalendarPhase &phase, const std::vector<uint32_t> &slots, std::vector<uint32_t> &visited, uint32_t task,
        uint64_t credits, bool deferred
    ) const {
        // Each chain advances only along dependency edges. Reject unrelated
        // tips even if the current calendar happens to execute them in order.
        std::priority_queue<uint32_t> pending;
        const auto enqueue = [&](uint32_t predecessor) {
            if (visited[predecessor] == task) return;
            visited[predecessor] = task;
            pending.push(predecessor);
        };
        for (int32_t predecessor : phase.fanins[task])
            enqueue(static_cast<uint32_t>(predecessor));
        size_t edge_budget = 256;
        while (!pending.empty()) {
            const uint32_t ancestor = pending.top();
            pending.pop();
            const uint32_t slot = slots[ancestor];
            if (chains_[slot].tip == ancestor && fits(slot, credits, deferred)) return slot;
            const auto &row = phase.fanins[ancestor];
            if (row.size() > edge_budget) continue;
            edge_budget -= row.size();
            for (int32_t predecessor : row)
                enqueue(static_cast<uint32_t>(predecessor));
        }
        return UINT32_MAX;
    }

    bool append(uint32_t global, uint32_t task, uint64_t credits, bool deferred, uint32_t slot) {
        if (global >= plan_.tasks.size() || plan_.tasks[global].slot != UINT32_MAX || credits == 0 ||
            credits > CALENDAR_COUNT_MASK)
            return false;
        if (slot == UINT32_MAX) {
            // A following phase waits for every completion in the previous
            // phase. Reuse a slot without resetting its count or sticky bits.
            const auto available = std::find_if(available_.begin(), available_.end(), [&](uint32_t candidate) {
                return fits(candidate, credits, deferred);
            });
            if (available != available_.end()) {
                slot = *available;
                *available = available_.back();
                available_.pop_back();
            } else {
                slot = static_cast<uint32_t>(chains_.size());
                chains_.emplace_back();
            }
        }
        auto &chain = chains_[slot];
        chain.tip = task;
        chain.target += static_cast<uint32_t>(credits);
        const uint32_t bit = deferred ? uint32_t{1} << (24 + chain.deferred_count++) : 0;
        plan_.tasks[global] = CalendarCompletionPlan::TaskCounter{slot, chain.target, bit};
        return true;
    }

    CalendarCompletionPlan &plan_;
    std::vector<Chain> chains_;
    std::vector<uint32_t> available_;
};

}  // namespace range_calendar_detail
}  // namespace simpler::percounter
