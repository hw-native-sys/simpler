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
#include <utility>
#include <vector>

#include "region_plan.h"

namespace simpler::percounter {

struct CalendarPhase {
    CalendarPhase() = default;
    CalendarPhase(const CalendarPhase &) = delete;
    CalendarPhase &operator=(const CalendarPhase &) = delete;
    CalendarPhase(CalendarPhase &&) noexcept = default;
    CalendarPhase &operator=(CalendarPhase &&) noexcept = default;

    // Ordinary phases carry outer task IDs; Graph phases carry body task IDs.
    int32_t invocation{-1};
    std::vector<uint32_t> task_ids;
    std::vector<TaskInput> tasks;
    // Each TaskInput points into its row. Moving the phase transfers ownership
    // without relocating these buffers.
    std::vector<std::vector<int32_t>> fanins;
};

namespace calendar_phase_detail {

inline void bind_fanins(CalendarPhase &phase) {
    for (size_t id = 0; id < phase.tasks.size(); ++id) {
        const auto &row = phase.fanins[id];
        phase.tasks[id].fanin_count = static_cast<int32_t>(row.size());
        phase.tasks[id].fanin_ids = row.empty() ? nullptr : row.data();
    }
}

inline bool valid_outer(const std::vector<TaskInput> &outer, const GraphPlanInput &graph) {
    if (outer.size() > INT32_MAX || graph.invocation_count > outer.size() ||
        (graph.invocation_count != 0 && (graph.invocations == nullptr || graph.templates == nullptr)))
        return false;
    for (size_t k = 0; k < graph.invocation_count; ++k) {
        const auto &invocation = graph.invocations[k];
        if (invocation.template_index >= graph.template_count || invocation.shell_task_id < 0 ||
            static_cast<uint64_t>(invocation.shell_task_id) >= outer.size() ||
            outer[invocation.shell_task_id].graph_invocation != static_cast<int32_t>(k))
            return false;
    }
    for (size_t id = 0; id < outer.size(); ++id) {
        const auto &task = outer[id];
        if (task.fanin_count < 0 || (task.fanin_count != 0 && task.fanin_ids == nullptr) || task.graph_invocation < -1)
            return false;
        if (task.graph_invocation >= 0) {
            if (task.inline_completed || task.sync_start || task.has_predicate ||
                static_cast<size_t>(task.graph_invocation) >= graph.invocation_count ||
                graph.invocations[task.graph_invocation].shell_task_id != static_cast<int64_t>(id))
                return false;
        }
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            const int32_t producer = task.fanin_ids[e];
            if (producer < 0 || static_cast<size_t>(producer) >= id) return false;
        }
    }
    return true;
}

inline bool build_graph_phase(const HostGraphTemplate &body, int32_t invocation, std::vector<CalendarPhase> &phases) {
    const size_t count = body.tasks.size();
    if (count == 0 || count > INT32_MAX || body.fanin_indices.size() > INT32_MAX ||
        body.fanin_offsets.size() != count + 1 || body.fanin_offsets.front() != 0 ||
        body.fanin_offsets.back() != static_cast<int32_t>(body.fanin_indices.size()))
        return false;

    auto &phase = phases.emplace_back();
    phase.invocation = invocation;
    phase.task_ids.reserve(count);
    phase.tasks.reserve(count);
    phase.fanins.reserve(count);
    for (size_t id = 0; id < count; ++id) {
        const int32_t begin = body.fanin_offsets[id];
        const int32_t end = body.fanin_offsets[id + 1];
        if (begin > end || end > body.fanin_offsets.back()) return false;
        for (int32_t e = begin; e < end; ++e) {
            if (body.fanin_indices[e] < 0 || static_cast<size_t>(body.fanin_indices[e]) >= id) return false;
        }

        const auto &src = body.tasks[id];
        const uint32_t mask = src.active_mask;
        const uint32_t blocks = src.block_num;
        const bool sync = src.sync_start;
        uint8_t core_type = 1;
        const auto placement =
            mask == 0 ? TaskPlacement::AIV : task_placement_from_active_mask(static_cast<uint8_t>(mask), &core_type);
        const int32_t subtasks = mask == 0 ? 1 : __builtin_popcount(mask);
        if (mask > 7 || placement == TaskPlacement::UNSUPPORTED || blocks == 0 ||
            blocks > PERCOUNTER_ORDER_BLOCK_MASK || src.subtask_total != blocks * subtasks ||
            (mask == 0 && blocks != 1) || (sync && (mask == 0 || blocks < 2)))
            return false;

        TaskInput task{};
        task.core_type_index = core_type;
        task.is_mix = placement == TaskPlacement::MIX;
        task.block_num = static_cast<int32_t>(blocks);
        task.subtask_count = subtasks;
        task.sync_start = sync;
        task.has_predicate = src.has_predicate;
        task.active_mask = mask;
        phase.task_ids.push_back(static_cast<uint32_t>(id));
        phase.tasks.push_back(task);
        phase.fanins.emplace_back(body.fanin_indices.begin() + begin, body.fanin_indices.begin() + end);
    }
    bind_fanins(phase);
    return true;
}

inline void append_ordinary_phase(
    const std::vector<TaskInput> &outer, std::vector<uint32_t> ids, std::vector<int32_t> &local_ids,
    std::vector<CalendarPhase> &phases
) {
    if (ids.empty()) return;
    auto &phase = phases.emplace_back();
    phase.task_ids = std::move(ids);
    phase.tasks.reserve(phase.task_ids.size());
    phase.fanins.reserve(phase.task_ids.size());
    for (uint32_t id : phase.task_ids) {
        const auto &task = outer[id];
        auto &row = phase.fanins.emplace_back();
        for (int32_t e = 0; e < task.fanin_count; ++e) {
            const int32_t local = local_ids[task.fanin_ids[e]];
            if (local >= 0) row.push_back(local);
        }
        local_ids[id] = static_cast<int32_t>(phase.tasks.size());
        phase.tasks.push_back(task);
    }
    for (uint32_t id : phase.task_ids)
        local_ids[id] = -1;
    bind_fanins(phase);
}

}  // namespace calendar_phase_detail

// Only a Graph's ordinary ancestors must run before that Graph. Keep
// independent ordinary work available for later range and tail placement.
inline bool plan_calendar_phases(
    const std::vector<TaskInput> &outer, const GraphPlanInput &graph, std::vector<CalendarPhase> *out
) {
    using namespace calendar_phase_detail;
    if (out == nullptr || !valid_outer(outer, graph)) return false;

    std::vector<CalendarPhase> phases;
    std::vector<int32_t> local_ids(outer.size(), -1);
    std::vector<bool> scheduled(outer.size(), false);
    for (size_t id = 0; id < outer.size(); ++id)
        scheduled[id] = outer[id].inline_completed;

    // The outer table is topological. By the time we visit a Graph shell,
    // any other Graph in its predecessor closure has already been emitted.
    for (const auto &shell : outer) {
        if (shell.graph_invocation < 0) continue;

        std::vector<uint32_t> ancestors;
        const auto collect_producers = [&](const TaskInput &task) {
            for (int32_t e = 0; e < task.fanin_count; ++e) {
                const auto dep = static_cast<uint32_t>(task.fanin_ids[e]);
                if (!scheduled[dep]) {
                    scheduled[dep] = true;
                    ancestors.push_back(dep);
                }
            }
        };
        collect_producers(shell);
        for (size_t i = 0; i < ancestors.size(); ++i)
            collect_producers(outer[ancestors[i]]);
        std::sort(ancestors.begin(), ancestors.end());
        append_ordinary_phase(outer, std::move(ancestors), local_ids, phases);

        const auto &invocation = graph.invocations[shell.graph_invocation];
        if (!build_graph_phase(graph.templates[invocation.template_index], shell.graph_invocation, phases))
            return false;
        scheduled[invocation.shell_task_id] = true;
    }

    std::vector<uint32_t> tail;
    for (uint32_t id = 0; id < outer.size(); ++id) {
        if (!scheduled[id]) tail.push_back(id);
    }
    append_ordinary_phase(outer, std::move(tail), local_ids, phases);
    *out = std::move(phases);
    return true;
}

}  // namespace simpler::percounter
