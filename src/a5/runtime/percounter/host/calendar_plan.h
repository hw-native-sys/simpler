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
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <map>
#include <queue>
#include <utility>
#include <vector>

#include "calendar_waits.h"
#include "region_plan.h"
#include "scheduler/calendar_wire.h"

namespace simpler::percounter {

struct CalendarRowCost {
    uint64_t occupancy_cycles{1};
    uint64_t defer_tail_cycles{0};

    uint64_t output_latency() const { return occupancy_cycles + defer_tail_cycles; }
};

struct CalendarPlan {
    std::vector<std::byte> bytes;
    // Stable sample indices are grouped by task, block and logical role.
    std::vector<uint32_t> logical_row_by_packed;
    range_calendar_detail::CalendarWaitPlan waits;
    uint64_t predicted_cycles{0};
};

namespace calendar_detail {

constexpr uint32_t lane_count = 3;
constexpr uint32_t mask_count = 1U << lane_count;
static_assert(lane_count == PLATFORM_CORES_PER_BLOCKDIM);

struct TaskShape {
    uint32_t first_row{0};
    uint32_t first_group{0};
    uint32_t role_count{0};
    uint32_t role_mask{0};
    uint32_t group_count{0};
    uint64_t longest_output_latency{0};
};

struct Shape {
    std::vector<TaskShape> tasks;
    uint32_t row_count{0};
    uint32_t group_count{0};
};

inline bool describe(
    const std::vector<TaskInput> &tasks, uint32_t clusters, const std::vector<CalendarRowCost> &costs, Shape *shape
) {
    if (clusters == 0 || clusters > PLATFORM_MAX_BLOCKDIM || tasks.size() > INT32_MAX) return false;
    shape->tasks.resize(tasks.size());
    for (uint32_t id = 0; id < tasks.size(); ++id) {
        const auto &task = tasks[id];
        if (task.graph_invocation >= 0 || task.fanin_count < 0 || (task.fanin_count != 0 && task.fanin_ids == nullptr))
            return false;
        for (int32_t i = 0; i < task.fanin_count; ++i) {
            const int32_t dep = task.fanin_ids[i];
            if (dep < 0 || static_cast<uint32_t>(dep) >= id) return false;
        }
        auto &s = shape->tasks[id];
        s.first_row = shape->row_count;
        s.first_group = shape->group_count;
        if (task.inline_completed) continue;
        // Unfolded dependency-only nodes run the standard zero-kernel AIV completion path.
        if (task.active_mask == 0 && (task.block_num != 1 || task.sync_start)) return false;
        s.role_mask = task.active_mask == 0 ? 2U : task.active_mask;
        uint8_t core_type = 0;
        const auto placement = task_placement_from_active_mask(static_cast<uint8_t>(s.role_mask), &core_type);
        if (s.role_mask >= mask_count || placement == TaskPlacement::UNSUPPORTED ||
            task.is_mix != (placement == TaskPlacement::MIX) || (!task.is_mix && task.core_type_index != core_type) ||
            task.block_num <= 0 || static_cast<uint32_t>(task.block_num) > PERCOUNTER_ORDER_BLOCK_MASK ||
            task.subtask_count != __builtin_popcount(s.role_mask))
            return false;
        if (task.sync_start) {
            const uint32_t limit = placement == TaskPlacement::AIV ? 2U * clusters : clusters;
            if (task.block_num < 2 || static_cast<uint32_t>(task.block_num) > limit) return false;
        }
        s.role_count = static_cast<uint32_t>(task.subtask_count);
        s.group_count = task.sync_start ? 1 : static_cast<uint32_t>(task.block_num);
        const uint64_t added = static_cast<uint64_t>(task.block_num) * s.role_count;
        if (added > UINT32_MAX - shape->row_count) return false;
        shape->row_count += static_cast<uint32_t>(added);
        shape->group_count += s.group_count;
    }
    if (shape->row_count == 0) return false;
    if (costs.empty()) return true;
    if (costs.size() != shape->row_count) return false;

    // A completely serialized graph bounds both the resource paths and output paths.
    uint64_t sum = 0;
    for (size_t id = 0; id < tasks.size(); ++id) {
        auto &s = shape->tasks[id];
        if (s.role_count == 0) continue;
        const uint32_t count = static_cast<uint32_t>(tasks[id].block_num) * s.role_count;
        for (uint32_t row = s.first_row; row < s.first_row + count; ++row) {
            const auto &cost = costs[row];
            if (cost.occupancy_cycles == 0 || cost.defer_tail_cycles > UINT64_MAX - cost.occupancy_cycles) return false;
            const uint64_t latency = cost.output_latency();
            if (latency > UINT64_MAX - sum) return false;
            sum += latency;
            s.longest_output_latency = std::max(s.longest_output_latency, latency);
        }
    }
    return true;
}

inline uint32_t worker_index(uint32_t clusters, uint32_t cluster, uint32_t lane) {
    return lane == 0 ? cluster : clusters + 2U * cluster + lane - 1U;
}

inline uint32_t worker_lane(uint32_t clusters, uint32_t worker) {
    return worker < clusters ? 0 : 1U + (worker - clusters) % 2U;
}

inline uint32_t worker_cluster(uint32_t clusters, uint32_t worker) {
    return worker < clusters ? worker : (worker - clusters) / 2U;
}

inline uint32_t logical_row(const TaskShape &s, uint32_t block, uint32_t role) {
    const uint32_t preceding = s.role_mask & ((1U << role) - 1U);
    return s.first_row + block * s.role_count + static_cast<uint32_t>(__builtin_popcount(preceding));
}

inline CalendarRowCost row_cost(const std::vector<CalendarRowCost> &costs, uint32_t row) {
    return costs.empty() ? CalendarRowCost{} : costs[row];
}

struct Placement {
    uint32_t cluster;
    uint32_t mask;
    uint64_t free_at{0};
};

using Choices = std::array<std::vector<Placement>, mask_count>;

inline Choices make_choices(const Shape &shape, uint32_t clusters) {
    Choices choices;
    for (const auto &task : shape.tasks) {
        if (task.role_mask == 0 || !choices[task.role_mask].empty()) continue;
        auto &list = choices[task.role_mask];
        for (uint32_t cluster = 0; cluster < clusters; ++cluster) {
            list.push_back({cluster, task.role_mask, 0});
            // A standalone Vector's logical role is V0 on either physical Vector lane.
            if (task.role_mask == 2U) list.push_back({cluster, 4U, 0});
        }
    }
    return choices;
}

inline void update_choices(Choices *choices, const std::vector<uint64_t> &free_at, uint32_t clusters) {
    for (auto &list : *choices) {
        for (auto &place : list) {
            place.free_at = 0;
            for (uint32_t lane = 0; lane < lane_count; ++lane) {
                if ((place.mask & (1U << lane)) == 0) continue;
                place.free_at = std::max(place.free_at, free_at[worker_index(clusters, place.cluster, lane)]);
            }
        }
        std::sort(list.begin(), list.end(), [](const Placement &a, const Placement &b) {
            if (a.free_at != b.free_at) return a.free_at < b.free_at;
            if (a.cluster != b.cluster) return a.cluster < b.cluster;
            return a.mask < b.mask;
        });
    }
}

struct ReadyTask {
    uint32_t id;
    uint64_t data_ready;
    uint64_t priority;
};

struct LowerPriority {
    bool operator()(const ReadyTask &a, const ReadyTask &b) const {
        return a.priority != b.priority ? a.priority < b.priority : a.id > b.id;
    }
};

struct LaterData {
    bool operator()(const ReadyTask &a, const ReadyTask &b) const {
        return a.data_ready != b.data_ready ? a.data_ready > b.data_ready : LowerPriority{}(a, b);
    }
};

struct ReadyTasks {
    std::priority_queue<ReadyTask, std::vector<ReadyTask>, LaterData> future;
    std::priority_queue<ReadyTask, std::vector<ReadyTask>, LowerPriority> available;
};

struct PackedRow {
    CalendarRow row;
    uint32_t logical;
};

struct PlacementState {
    std::vector<std::vector<PackedRow>> workers;
    std::vector<uint64_t> free_at;
    std::vector<CalendarLocalJoin> joins;
};

inline uint64_t append_block(
    PlacementState *state, uint32_t clusters, const TaskInput &task, const TaskShape &shape, uint32_t id,
    uint32_t block, const Placement &placement, uint64_t start, const std::vector<CalendarRowCost> &costs
) {
    const uint64_t entry = percounter_order_entry(id, block, task.block_num);
    uint32_t join_index = CALENDAR_NO_JOIN;
    if (task.is_mix) {
        CalendarLocalJoin join{};
        join.leader_lane = static_cast<uint32_t>(__builtin_ctz(placement.mask));
        for (uint32_t lane = 0; lane < lane_count; ++lane) {
            if ((placement.mask & (1U << lane)) == 0) continue;
            const auto worker = worker_index(clusters, placement.cluster, lane);
            join.arrive[lane] = static_cast<uint32_t>(state->workers[worker].size()) + 1;
        }
        join_index = static_cast<uint32_t>(state->joins.size());
        state->joins.push_back(join);
    }
    uint64_t last = start;
    for (uint32_t lane = 0; lane < lane_count; ++lane) {
        if ((placement.mask & (1U << lane)) == 0) continue;
        const auto worker = worker_index(clusters, placement.cluster, lane);
        const auto logical = logical_row(shape, block, shape.role_mask == 2U ? 1U : lane);
        state->workers[worker].push_back({{entry, join_index, 0}, logical});
        const auto cost = row_cost(costs, logical);
        state->free_at[worker] = start + cost.occupancy_cycles;
        last = std::max(last, start + cost.output_latency());
    }
    return last;
}

inline bool pack(
    const PlacementState &state, const std::vector<TaskInput> &tasks, uint32_t row_count, uint64_t predicted,
    CalendarPlan *out
) {
    const uint32_t workers = static_cast<uint32_t>(state.workers.size());
    const uint64_t index_offset = sizeof(CalendarHeader);
    constexpr auto alignment = alignof(CalendarRow);
    const uint64_t index_end = index_offset + (workers + 1U) * sizeof(uint32_t);
    const uint64_t row_offset = (index_end + alignment - 1) & ~(alignment - 1);
    const uint64_t join_offset = row_offset + static_cast<uint64_t>(row_count) * sizeof(CalendarRow);
    const uint64_t bytes = join_offset + state.joins.size() * sizeof(CalendarLocalJoin);
    if (bytes > UINT32_MAX) return false;

    CalendarPlan result;
    result.bytes.resize(static_cast<size_t>(bytes));
    result.logical_row_by_packed.reserve(row_count);
    result.predicted_cycles = predicted;
    CalendarHeader header{
        workers,
        row_count,
        static_cast<uint32_t>(state.joins.size()),
        static_cast<uint32_t>(index_offset),
        static_cast<uint32_t>(row_offset),
        static_cast<uint32_t>(join_offset)
    };
    auto *base = result.bytes.data();
    std::memcpy(base, &header, sizeof(header));
    std::vector<CalendarRow> rows;
    std::vector<uint32_t> indexes;
    rows.reserve(row_count);
    indexes.reserve(workers + 1);
    for (const auto &lane : state.workers) {
        indexes.push_back(static_cast<uint32_t>(rows.size()));
        for (const auto &row : lane) {
            rows.push_back(row.row);
            result.logical_row_by_packed.push_back(row.logical);
        }
    }
    if (rows.size() != row_count) return false;
    indexes.push_back(row_count);
    if (!range_calendar_detail::plan_calendar_waits(tasks, rows, indexes, state.joins.size(), result.waits))
        return false;
    std::memcpy(base + index_offset, indexes.data(), indexes.size() * sizeof(uint32_t));
    std::memcpy(base + row_offset, rows.data(), rows.size() * sizeof(CalendarRow));
    if (!state.joins.empty())
        std::memcpy(base + join_offset, state.joins.data(), state.joins.size() * sizeof(CalendarLocalJoin));
    *out = std::move(result);
    return true;
}

template <class T>
inline bool read_table(const CalendarPlan &plan, uint32_t offset, uint32_t count, std::vector<T> *out) {
    const uint64_t table_bytes = static_cast<uint64_t>(count) * sizeof(T);
    if (offset > plan.bytes.size() || table_bytes > plan.bytes.size() - offset) return false;
    out->resize(count);
    if (count != 0) std::memcpy(out->data(), plan.bytes.data() + offset, table_bytes);
    return true;
}

struct Edge {
    size_t successor;
    uint64_t lag;
};

}  // namespace calendar_detail

inline bool build_static_calendar(
    const std::vector<TaskInput> &tasks, uint32_t cluster_count, const std::vector<CalendarRowCost> &costs,
    CalendarPlan *out
) {
    using namespace calendar_detail;
    Shape shape;
    if (out == nullptr || !describe(tasks, cluster_count, costs, &shape)) return false;

    std::vector<std::vector<uint32_t>> successors(tasks.size());
    std::vector<uint32_t> prerequisites(tasks.size(), 0);
    std::vector<uint32_t> next_block(tasks.size(), 0);
    std::vector<uint64_t> priority(tasks.size(), 0);
    std::vector<uint64_t> data_ready(tasks.size(), 0), task_complete(tasks.size(), 0);
    for (uint32_t id = 0; id < tasks.size(); ++id) {
        const auto &task = tasks[id];
        if (task.inline_completed) continue;
        for (int32_t edge = 0; edge < task.fanin_count; ++edge) {
            const auto dep = static_cast<uint32_t>(task.fanin_ids[edge]);
            if (tasks[dep].inline_completed) continue;
            successors[dep].push_back(id);
            ++prerequisites[id];
        }
    }
    // Use bottom-level ranks only after measurement. The cold run has a stable task-ID tie break.
    if (!costs.empty()) {
        for (size_t id = tasks.size(); id-- > 0;) {
            for (uint32_t child : successors[id])
                priority[id] = std::max(priority[id], priority[child]);
            priority[id] += shape.tasks[id].longest_output_latency;
        }
    }

    // Tasks in each class have the same earliest available placement time.
    std::map<std::pair<uint32_t, uint32_t>, ReadyTasks> ready;
    const auto add_ready = [&](uint32_t id) {
        const auto &task = tasks[id];
        const uint32_t width = task.sync_start ? static_cast<uint32_t>(task.block_num) : 1;
        ready[{shape.tasks[id].role_mask, width}].future.push({id, data_ready[id], priority[id]});
    };
    for (uint32_t id = 0; id < tasks.size(); ++id)
        if (!tasks[id].inline_completed && prerequisites[id] == 0) add_ready(id);

    Choices choices = make_choices(shape, cluster_count);
    const auto worker_count = static_cast<size_t>(cluster_count) * lane_count;
    PlacementState state{std::vector<std::vector<PackedRow>>(worker_count), std::vector<uint64_t>(worker_count, 0), {}};
    uint64_t predicted = 0;
    while (!ready.empty()) {
        update_choices(&choices, state.free_at, cluster_count);
        uint64_t start = UINT64_MAX;
        auto picked = ready.end();
        ReadyTask selected{};
        for (auto group = ready.begin(); group != ready.end(); ++group) {
            const auto [mask, width] = group->first;
            const uint64_t resource_ready = choices[mask][width - 1U].free_at;
            auto &queue = group->second;
            // Lane availability only advances. Tasks below the resource
            // frontier can stay ordered by priority for subsequent placements.
            while (!queue.future.empty() && queue.future.top().data_ready <= resource_ready) {
                queue.available.push(queue.future.top());
                queue.future.pop();
            }
            const auto &candidate = queue.available.empty() ? queue.future.top() : queue.available.top();
            const uint64_t earliest = std::max(candidate.data_ready, resource_ready);
            if (picked == ready.end() || earliest < start ||
                (earliest == start && LowerPriority{}(selected, candidate))) {
                start = earliest;
                picked = group;
                selected = candidate;
            }
        }
        const uint32_t id = selected.id;
        const auto &task = tasks[id];
        const auto [mask, width] = picked->first;
        const auto &choices_for_task = choices[mask];
        for (uint32_t i = 0; i < width; ++i) {
            const uint32_t block = next_block[id]++;
            const uint64_t complete = append_block(
                &state, cluster_count, task, shape.tasks[id], id, block, choices_for_task[i], start, costs
            );
            task_complete[id] = std::max(task_complete[id], complete);
        }
        if (next_block[id] != static_cast<uint32_t>(task.block_num)) continue;
        predicted = std::max(predicted, task_complete[id]);
        auto &queue = picked->second;
        if (queue.available.empty()) queue.future.pop();
        else queue.available.pop();
        if (queue.future.empty() && queue.available.empty()) ready.erase(picked);
        for (uint32_t successor : successors[id]) {
            data_ready[successor] = std::max(data_ready[successor], task_complete[id]);
            if (--prerequisites[successor] == 0) add_ready(successor);
        }
    }

    return pack(state, tasks, shape.row_count, predicted, out);
}

// Replay the packed order using the placement costs. Resource edges carry lane occupancy.
// Fanin uses the maximum output latency over all rows of its producer.
inline bool evaluate_static_calendar(
    const std::vector<TaskInput> &tasks, uint32_t cluster_count, const CalendarPlan &plan,
    const std::vector<CalendarRowCost> &costs, uint64_t *predicted_cycles
) {
    using namespace calendar_detail;
    Shape shape;
    if (predicted_cycles == nullptr || !describe(tasks, cluster_count, costs, &shape) ||
        plan.bytes.size() < sizeof(CalendarHeader) || plan.logical_row_by_packed.size() != shape.row_count)
        return false;

    CalendarHeader header{};
    std::memcpy(&header, plan.bytes.data(), sizeof(header));
    if (header.worker_count != cluster_count * lane_count || header.row_count != shape.row_count ||
        header.join_count > header.row_count)
        return false;
    std::vector<uint32_t> indices;
    std::vector<CalendarRow> rows;
    std::vector<CalendarLocalJoin> joins;
    if (!read_table(plan, header.index_offset, header.worker_count + 1, &indices) ||
        !read_table(plan, header.row_offset, header.row_count, &rows) ||
        !read_table(plan, header.join_offset, header.join_count, &joins) || indices.front() != 0 ||
        indices.back() != header.row_count || !std::is_sorted(indices.begin(), indices.end()))
        return false;

    const size_t graph_nodes = static_cast<size_t>(shape.group_count) + tasks.size();
    std::vector<std::vector<Edge>> next(graph_nodes);
    std::vector<size_t> pending(graph_nodes, 0);
    std::vector<bool> seen(shape.row_count, false);
    const auto edge = [&](size_t src, size_t dst, uint64_t lag) {
        next[src].push_back({dst, lag});
        ++pending[dst];
    };
    for (size_t id = 0; id < tasks.size(); ++id) {
        const auto &task = tasks[id];
        if (task.inline_completed) continue;
        const auto &s = shape.tasks[id];
        for (uint32_t g = s.first_group; g < s.first_group + s.group_count; ++g) {
            for (int32_t i = 0; i < task.fanin_count; ++i)
                edge(shape.group_count + static_cast<size_t>(task.fanin_ids[i]), g, 0);
        }
    }
    uint32_t joined_rows = 0;
    for (uint32_t worker = 0; worker < header.worker_count; ++worker) {
        size_t prev_group = graph_nodes;
        uint64_t prev_occupancy = 0;
        const uint32_t lane = worker_lane(cluster_count, worker);
        const uint32_t cluster = worker_cluster(cluster_count, worker);
        for (uint32_t i = indices[worker]; i < indices[worker + 1]; ++i) {
            const auto &row = rows[i];
            const auto id = static_cast<uint32_t>(percounter_order_task_id(row.entry));
            const auto block = static_cast<uint32_t>(percounter_order_block_idx(row.entry));
            if (id >= tasks.size()) return false;
            const auto &task = tasks[id];
            const auto &s = shape.tasks[id];
            const uint32_t role = s.role_mask == 2U && lane != 0 ? 1U : lane;
            if (task.inline_completed || block >= static_cast<uint32_t>(task.block_num) ||
                percounter_order_block_num(row.entry) != task.block_num || (s.role_mask & (1U << role)) == 0)
                return false;
            const uint32_t logical = logical_row(s, block, role);
            if (plan.logical_row_by_packed[i] != logical || seen[logical]) return false;
            seen[logical] = true;
            if (task.is_mix) {
                if (row.local_join >= joins.size()) return false;
                const auto &join = joins[row.local_join];
                if (join.arrive[lane] != i - indices[worker] + 1U || join.leader_lane >= lane_count ||
                    (task.active_mask & (1U << join.leader_lane)) == 0)
                    return false;
                ++joined_rows;
                for (uint32_t peer = 0; peer < lane_count; ++peer) {
                    const uint32_t ordinal = join.arrive[peer];
                    if ((ordinal != 0) != ((task.active_mask & (1U << peer)) != 0)) return false;
                    if (ordinal == 0 || peer == lane) continue;
                    const uint32_t sibling = worker_index(cluster_count, cluster, peer);
                    if (ordinal > indices[sibling + 1] - indices[sibling]) return false;
                    const auto &target = rows[indices[sibling] + ordinal - 1];
                    if (target.entry != row.entry || target.local_join != row.local_join) return false;
                }
            } else if (row.local_join != CALENDAR_NO_JOIN) {
                return false;
            }
            const size_t group = s.first_group + (task.sync_start ? 0 : block);
            const auto cost = row_cost(costs, logical);
            if (prev_group != graph_nodes) edge(prev_group, group, prev_occupancy);
            edge(group, shape.group_count + static_cast<size_t>(id), cost.output_latency());
            prev_group = group;
            prev_occupancy = cost.occupancy_cycles;
        }
    }
    if (header.join_count > joined_rows || std::any_of(seen.begin(), seen.end(), [](bool v) {
            return !v;
        }))
        return false;

    std::vector<size_t> ready;
    for (size_t node = 0; node < graph_nodes; ++node)
        if (pending[node] == 0) ready.push_back(node);
    std::vector<uint64_t> starts(graph_nodes, 0);
    uint64_t last = 0;
    for (size_t cursor = 0; cursor < ready.size(); ++cursor) {
        const size_t node = ready[cursor];
        last = std::max(last, starts[node]);
        for (const auto &arc : next[node]) {
            if (arc.lag > UINT64_MAX - starts[node]) return false;
            starts[arc.successor] = std::max(starts[arc.successor], starts[node] + arc.lag);
            if (--pending[arc.successor] == 0) ready.push_back(arc.successor);
        }
    }
    if (ready.size() != graph_nodes) return false;
    *predicted_cycles = last;
    return true;
}

}  // namespace simpler::percounter
