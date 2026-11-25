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
#include <iterator>
#include <map>
#include <queue>
#include <set>
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

struct EarlierPlacement {
    bool operator()(const Placement &a, const Placement &b) const {
        if (a.free_at != b.free_at) return a.free_at < b.free_at;
        if (a.cluster != b.cluster) return a.cluster < b.cluster;
        return a.mask < b.mask;
    }
};

class PlacementChoices {
    using Placements = std::set<Placement, EarlierPlacement>;
    static constexpr size_t max_width = 2U * PLATFORM_MAX_BLOCKDIM;

public:
    PlacementChoices(const Shape &shape, uint32_t clusters) :
        clusters_(clusters) {
        for (const auto &task : shape.tasks) {
            const auto mask = task.role_mask;
            if (mask == 0 || !ordered_[mask].empty()) continue;
            for (uint32_t cluster = 0; cluster < clusters_; ++cluster) {
                add(mask, {cluster, mask, 0});
                // A standalone Vector can use either physical Vector lane.
                if (mask == 2U) add(mask, {cluster, 4U, 0});
            }
        }
    }

    auto begin(uint32_t mask) const { return ordered_[mask].begin(); }

    uint64_t ready_at(uint32_t mask, uint32_t width) const { return std::next(begin(mask), width - 1)->free_at; }

    void update(uint32_t cluster, uint32_t changed, const std::vector<uint64_t> &free_at) {
        for (uint32_t mask = 1; mask < mask_count; ++mask) {
            if (ordered_[mask].empty()) continue;
            refresh(mask, cluster, mask, changed, free_at);
            if (mask == 2U) refresh(mask, cluster, 4U, changed, free_at);
        }
    }

private:
    uint32_t clusters_;
    std::array<Placements, mask_count> ordered_;
    std::array<std::array<Placements::iterator, max_width>, mask_count> positions_;

    static size_t position(uint32_t cluster, uint32_t logical, uint32_t physical) {
        return 2U * cluster + (logical != physical ? 1U : 0U);
    }

    void add(uint32_t mask, const Placement &choice) {
        positions_[mask][position(choice.cluster, mask, choice.mask)] = ordered_[mask].insert(choice).first;
    }

    void refresh(
        uint32_t mask, uint32_t cluster, uint32_t physical, uint32_t changed, const std::vector<uint64_t> &free_at
    ) {
        if ((physical & changed) == 0) return;
        auto &current = positions_[mask][position(cluster, mask, physical)];
        uint64_t updated = 0;
        for (uint32_t lane = 0; lane < lane_count; ++lane) {
            if ((physical & (1U << lane)) != 0)
                updated = std::max(updated, free_at[worker_index(clusters_, cluster, lane)]);
        }
        if (current->free_at == updated) return;
        auto node = ordered_[mask].extract(current);
        node.value().free_at = updated;
        current = ordered_[mask].insert(std::move(node)).position;
    }
};

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

struct CalendarLayout {
    CalendarHeader header{};
    std::vector<uint32_t> indices;
    std::vector<CalendarRow> rows;
    std::vector<CalendarLocalJoin> joins;
};

inline bool read_calendar_layout(
    const std::vector<TaskInput> &tasks, uint32_t cluster_count, const Shape &shape, const CalendarPlan &plan,
    CalendarLayout &layout
) {
    if (plan.bytes.size() < sizeof(CalendarHeader) || plan.logical_row_by_packed.size() != shape.row_count)
        return false;
    auto &header = layout.header;
    auto &indices = layout.indices;
    auto &rows = layout.rows;
    auto &joins = layout.joins;
    std::memcpy(&header, plan.bytes.data(), sizeof(header));
    if (header.worker_count != cluster_count * lane_count || header.row_count != shape.row_count ||
        header.join_count > header.row_count)
        return false;
    if (!read_table(plan, header.index_offset, header.worker_count + 1, &indices) ||
        !read_table(plan, header.row_offset, header.row_count, &rows) ||
        !read_table(plan, header.join_offset, header.join_count, &joins) || indices.front() != 0 ||
        indices.back() != header.row_count || !std::is_sorted(indices.begin(), indices.end()))
        return false;

    std::vector<bool> seen(shape.row_count, false);
    uint32_t joined_rows = 0;
    for (uint32_t worker = 0; worker < header.worker_count; ++worker) {
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
        }
    }
    if (header.join_count > joined_rows || std::any_of(seen.begin(), seen.end(), [](bool v) {
            return !v;
        }))
        return false;

    return true;
}

struct Edge {
    size_t successor;
    uint64_t lag;
};

}  // namespace calendar_detail

enum class CalendarBlockOrder { SUBMISSION, LONGEST_OUTPUT_FIRST };
enum class CalendarPriority { LONGEST_ROW, WORK_AND_WAIT };

struct CalendarPriorityCosts {
    uint64_t row_cycles{0};
    uint64_t observation_cycles{0};
    uint64_t credit_cycles{0};
};

inline bool calendar_placement_costs(
    const std::vector<TaskInput> &tasks, uint32_t clusters, const std::vector<CalendarRowCost> &samples,
    const CalendarPriorityCosts &overhead, std::vector<CalendarRowCost> *result
) {
    calendar_detail::Shape shape;
    if (samples.empty() || !calendar_detail::describe(tasks, clusters, samples, &shape)) return false;
    *result = samples;
    for (uint32_t id = 0; id < tasks.size(); ++id) {
        const auto &task = tasks[id];
        const auto &s = shape.tasks[id];
        if (s.role_count == 0) continue;
        uint64_t fanins = 0;
        for (int32_t i = 0; i < task.fanin_count; ++i)
            fanins += !tasks[task.fanin_ids[i]].inline_completed;
        const bool summarizes =
            fanins >= 2 && !task.is_mix && !task.sync_start && task.block_num >= 2 &&
            fanins <= CALENDAR_FANIN_TASKS_PER_LANE * std::min(static_cast<uint32_t>(task.block_num), 2U * clusters);
        const uint64_t probes = summarizes ? (fanins + task.block_num - 1) / task.block_num : fanins;
        if (overhead.row_cycles > UINT64_MAX - overhead.observation_cycles ||
            (probes != 0 &&
             overhead.credit_cycles > (UINT64_MAX - overhead.row_cycles - overhead.observation_cycles) / probes))
            return false;
        // Use local row and observation work to build the order. The complete
        // wait plan is subsequently scored using the original kernel samples.
        const auto delay =
            overhead.row_cycles + (fanins == 0 ? 0 : overhead.observation_cycles) + probes * overhead.credit_cycles;
        const auto row_count = static_cast<uint32_t>(task.block_num) * s.role_count;
        for (uint32_t row = s.first_row; row < s.first_row + row_count; ++row) {
            auto &cost = (*result)[row];
            if (cost.occupancy_cycles > UINT64_MAX - delay) return false;
            cost.occupancy_cycles += delay;
        }
    }
    return true;
}

inline bool build_static_calendar(
    const std::vector<TaskInput> &tasks, uint32_t cluster_count, const std::vector<CalendarRowCost> &costs,
    CalendarPlan *out, CalendarBlockOrder order = CalendarBlockOrder::SUBMISSION, uint64_t priority_window_cycles = 0,
    CalendarPriority ranking = CalendarPriority::LONGEST_ROW, bool fill_wait_gaps = false
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
            const auto &s = shape.tasks[id];
            uint64_t duration = s.longest_output_latency;
            if (ranking == CalendarPriority::WORK_AND_WAIT && s.role_count != 0) {
                const auto &task = tasks[id];
                const uint32_t capacity = s.role_mask == 2U ? 2U * cluster_count : cluster_count;
                for (uint32_t role = 0; role < s.role_count; ++role) {
                    uint64_t work = 0;
                    for (uint32_t block = 0; block < static_cast<uint32_t>(task.block_num); ++block) {
                        const auto row = s.first_row + block * s.role_count + role;
                        work += costs[row].occupancy_cycles;
                    }
                    const auto minimum = work / capacity + (work % capacity != 0);
                    duration = std::max(duration, minimum);
                }
            }
            priority[id] += duration;
        }
    }

    std::vector<std::vector<std::pair<uint64_t, uint32_t>>> block_order;
    if (!costs.empty() && order == CalendarBlockOrder::LONGEST_OUTPUT_FIRST) {
        block_order.resize(tasks.size());
        for (uint32_t id = 0; id < tasks.size(); ++id) {
            const auto &task = tasks[id];
            const auto &s = shape.tasks[id];
            if (task.inline_completed || task.sync_start || task.block_num < 2) continue;
            auto &blocks = block_order[id];
            blocks.reserve(static_cast<size_t>(task.block_num));
            for (uint32_t block = 0; block < static_cast<uint32_t>(task.block_num); ++block) {
                uint64_t latency = 0;
                for (uint32_t role = 0; role < s.role_count; ++role) {
                    const uint32_t row = s.first_row + block * s.role_count + role;
                    latency = std::max(latency, costs[row].output_latency());
                }
                blocks.emplace_back(latency, block);
            }
            std::sort(blocks.begin(), blocks.end(), [](const auto &a, const auto &b) {
                return a.first != b.first ? a.first > b.first : a.second < b.second;
            });
        }
    }

    // Tasks in each class have the same earliest available placement time.
    std::map<std::pair<uint32_t, uint32_t>, ReadyTasks> ready;
    const auto add_ready = [&](uint32_t id) {
        const auto &task = tasks[id];
        const uint32_t width = task.sync_start ? static_cast<uint32_t>(task.block_num) : 1;
        ready[{shape.tasks[id].role_mask, width}].future.push({id, data_ready[id], priority[id]});
    };
    uint64_t predicted = 0;
    const auto release_successors = [&](uint32_t id) {
        predicted = std::max(predicted, task_complete[id]);
        for (uint32_t successor : successors[id]) {
            data_ready[successor] = std::max(data_ready[successor], task_complete[id]);
            if (--prerequisites[successor] == 0) add_ready(successor);
        }
    };
    for (uint32_t id = 0; id < tasks.size(); ++id)
        if (!tasks[id].inline_completed && prerequisites[id] == 0) add_ready(id);

    PlacementChoices choices(shape, cluster_count);
    const auto worker_count = static_cast<size_t>(cluster_count) * lane_count;
    PlacementState state{std::vector<std::vector<PackedRow>>(worker_count), std::vector<uint64_t>(worker_count, 0), {}};
    struct GroupCandidate {
        decltype(ready)::iterator group;
        ReadyTask task;
        uint64_t earliest;
    };
    std::vector<GroupCandidate> candidates;
    while (!ready.empty()) {
        candidates.clear();
        uint64_t start = UINT64_MAX;
        uint64_t earliest_any = UINT64_MAX;
        auto picked = ready.end();
        ReadyTask selected{};
        for (auto group = ready.begin(); group != ready.end(); ++group) {
            const auto [mask, width] = group->first;
            const uint64_t resource_ready = choices.ready_at(mask, width);
            auto &queue = group->second;
            // Lane availability only advances. A priority window can admit a
            // near-ready task while the dependency still limits its start.
            const uint64_t ready_through =
                resource_ready + std::min(priority_window_cycles, UINT64_MAX - resource_ready);
            while (!queue.future.empty() && queue.future.top().data_ready <= ready_through) {
                queue.available.push(queue.future.top());
                queue.future.pop();
            }
            const auto &candidate = queue.available.empty() ? queue.future.top() : queue.available.top();
            const uint64_t earliest = std::max(candidate.data_ready, resource_ready);
            earliest_any = std::min(earliest_any, earliest);
            candidates.push_back({group, candidate, earliest});
        }
        for (const auto &candidate : candidates) {
            if (candidate.earliest - earliest_any > priority_window_cycles) continue;
            if (picked == ready.end() || LowerPriority{}(selected, candidate.task)) {
                start = candidate.earliest;
                picked = candidate.group;
                selected = candidate.task;
            }
        }
        const auto [mask, width] = picked->first;
        auto &queue = picked->second;
        const ReadyTask reserved = selected;
        const uint64_t reserved_start = start;
        bool backfilled = false;
        const uint64_t free_at = choices.ready_at(mask, width);
        if (fill_wait_gaps && start > free_at && !queue.available.empty()) {
            // Inspect a bounded number of the same placement class. They can
            // run before the selected task if all its lanes still meet the
            // original start time.
            std::array<ReadyTask, 4> deferred;
            size_t deferred_count = 0;
            while (deferred_count < deferred.size() && !queue.available.empty()) {
                const auto other = queue.available.top();
                queue.available.pop();
                const uint64_t other_start = std::max(free_at, other.data_ready);
                if (other.id != selected.id && other_start < start) {
                    const auto &other_shape = shape.tasks[other.id];
                    uint64_t occupancy = 0;
                    for (uint32_t i = 0; i < width; ++i) {
                        uint32_t block = next_block[other.id] + i;
                        if (!block_order.empty() && !block_order[other.id].empty())
                            block = block_order[other.id][block].second;
                        const auto first = other_shape.first_row + block * other_shape.role_count;
                        for (uint32_t role = 0; role < other_shape.role_count; ++role)
                            occupancy = std::max(occupancy, row_cost(costs, first + role).occupancy_cycles);
                    }
                    if (occupancy <= start - other_start) {
                        selected = other;
                        start = other_start;
                        backfilled = true;
                        break;
                    }
                }
                deferred[deferred_count++] = other;
            }
            for (size_t i = 0; i < deferred_count; ++i)
                queue.available.push(deferred[i]);
        }
        std::array<std::pair<uint32_t, uint32_t>, 2U * PLATFORM_MAX_BLOCKDIM> changed;
        const auto place = [&](uint32_t id, uint64_t tick) {
            auto choice = choices.begin(mask);
            const bool reorder = !block_order.empty() && !block_order[id].empty();
            for (uint32_t i = 0; i < width; ++i, ++choice) {
                const uint32_t index = next_block[id]++;
                const uint32_t block = reorder ? block_order[id][index].second : index;
                const uint64_t complete =
                    append_block(&state, cluster_count, tasks[id], shape.tasks[id], id, block, *choice, tick, costs);
                task_complete[id] = std::max(task_complete[id], complete);
                changed[i] = {choice->cluster, choice->mask};
            }
            return next_block[id] == static_cast<uint32_t>(tasks[id].block_num);
        };
        const bool selected_done = place(selected.id, start);
        // Commit the reservation on the same placements before allowing a
        // different resource class to compete for any overlapping lane.
        const bool reserved_done = backfilled ? place(reserved.id, reserved_start) : selected_done;
        for (uint32_t i = 0; i < width; ++i)
            choices.update(changed[i].first, changed[i].second, state.free_at);
        if (reserved_done) {
            if (queue.available.empty()) queue.future.pop();
            else queue.available.pop();
        }
        if (backfilled && !selected_done) queue.available.push(selected);
        if (queue.future.empty() && queue.available.empty()) ready.erase(picked);
        if (backfilled && selected_done) release_successors(selected.id);
        if (reserved_done) release_successors(reserved.id);
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
    CalendarLayout layout;
    if (predicted_cycles == nullptr || !describe(tasks, cluster_count, costs, &shape) ||
        !read_calendar_layout(tasks, cluster_count, shape, plan, layout))
        return false;

    const size_t graph_nodes = static_cast<size_t>(shape.group_count) + tasks.size();
    std::vector<std::vector<Edge>> next(graph_nodes);
    std::vector<size_t> pending(graph_nodes, 0);
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
    for (uint32_t worker = 0; worker < layout.header.worker_count; ++worker) {
        size_t prev_group = graph_nodes;
        uint64_t prev_occupancy = 0;
        for (uint32_t i = layout.indices[worker]; i < layout.indices[worker + 1]; ++i) {
            const auto &row = layout.rows[i];
            const auto id = static_cast<uint32_t>(percounter_order_task_id(row.entry));
            const auto block = static_cast<uint32_t>(percounter_order_block_idx(row.entry));
            const size_t group = shape.tasks[id].first_group + (tasks[id].sync_start ? 0 : block);
            const auto cost = row_cost(costs, plan.logical_row_by_packed[i]);
            if (prev_group != graph_nodes) edge(prev_group, group, prev_occupancy);
            edge(group, shape.group_count + static_cast<size_t>(id), cost.output_latency());
            prev_group = group;
            prev_occupancy = cost.occupancy_cycles;
        }
    }

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
