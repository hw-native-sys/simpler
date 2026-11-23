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
#include <string>
#include <utility>
#include <vector>

#include "scheduler/scheduler_task_catalog.h"
#include "dependency_plan.h"
#include "host/control_folding.h"
#include "task_catalog_deferral.h"

struct SchedulerTaskCatalogPlan {
    uint32_t cluster_count{0};
    uint64_t header_offset{0};
    uint64_t total_size{0};
    SchedulerTaskCatalogHeader header{};
    std::vector<SchedulerTaskCatalogOrder> order_rows;
    std::vector<SchedulerTaskCatalogDescriptor> descriptors;
    std::vector<std::vector<int64_t>> catalogs;
    std::vector<int64_t> task_order;
    std::vector<uint32_t> task_levels;
    std::vector<uint32_t> task_bottom_levels;
    std::vector<uint32_t> task_ranks;
};

struct SchedulerTaskOrder {
    std::vector<int64_t> task_order;
    std::vector<uint32_t> task_levels;
    std::vector<uint32_t> task_bottom_levels;
    std::vector<uint32_t> task_ranks;
};

// Sorting owns these scratch successor rows. Deduplicate in place and reuse them for the rank check.
inline bool scheduler_topological_task_order(
    std::vector<std::vector<int64_t>> &successors, const std::vector<SchedulerTaskMetadata> &metadata,
    uint32_t clusters, bool defer_side_tasks, SchedulerTaskOrder *order, std::string *error = nullptr
) {
    const auto fail = [&](const std::string &message) {
        if (error != nullptr) *error = message;
        return false;
    };
    const size_t count = successors.size();
    if (order == nullptr || count > INT32_MAX || metadata.size() != count || clusters == 0 ||
        clusters > SCHEDULER_CLUSTER_CAPACITY)
        return fail("invalid task ordering inputs");
    for (const auto &row : successors)
        for (const auto successor : row)
            if (successor < 0 || static_cast<uint64_t>(successor) >= count)
                return fail("invalid topological successor");
    std::vector<uint32_t> indegrees(count, 0);
    for (auto &edges : successors) {
        std::sort(edges.begin(), edges.end());
        edges.erase(std::unique(edges.begin(), edges.end()), edges.end());
        for (const int64_t successor : edges)
            ++indegrees[successor];
    }
    std::vector<int64_t> topological;
    topological.reserve(count);
    std::vector<uint32_t> levels(count, 0);
    for (size_t ref = 0; ref < count; ++ref)
        if (indegrees[ref] == 0) topological.push_back(static_cast<int64_t>(ref));
    for (size_t cursor = 0; cursor < topological.size(); ++cursor) {
        const int64_t predecessor = topological[cursor];
        for (const int64_t successor : successors[predecessor]) {
            levels[successor] = std::max(levels[successor], levels[predecessor] + 1);
            if (--indegrees[successor] == 0) topological.push_back(successor);
        }
    }
    if (topological.size() != count) {
        const auto blocked = std::find_if(indegrees.begin(), indegrees.end(), [](uint32_t value) {
            return value != 0;
        });
        return fail("task ordering cycle; blocked ref " + std::to_string(blocked - indegrees.begin()));
    }
    std::vector<uint32_t> bottom_levels(count, 0);
    for (auto it = topological.rbegin(); it != topological.rend(); ++it)
        for (const int64_t successor : successors[*it])
            bottom_levels[*it] = std::max(bottom_levels[*it], bottom_levels[successor] + 1);
    std::sort(topological.begin(), topological.end(), [&](int64_t left, int64_t right) {
        if (levels[left] != levels[right]) return levels[left] < levels[right];
        if (bottom_levels[left] != bottom_levels[right]) return bottom_levels[left] > bottom_levels[right];
        return left < right;
    });
    std::vector<uint32_t> ranks(count, 0);
    for (size_t rank = 0; rank < count; ++rank)
        ranks[topological[rank]] = static_cast<uint32_t>(rank);
    if (defer_side_tasks)
        scheduler_defer_side_tasks(successors, metadata, clusters, levels, bottom_levels, topological, ranks);
    *order = {std::move(topological), std::move(levels), std::move(bottom_levels), std::move(ranks)};
    if (error != nullptr) error->clear();
    return true;
}

// Maintain the last rank that could use each physical lane. AIV0's MIX
// publication duty conflicts even when that lane is absent from the kernel.
struct SchedulerCatalogOrderBuilder {
    uint32_t previous_mix_floor[PLATFORM_CORES_PER_BLOCKDIM]{};
    uint32_t previous_sync_floor[PLATFORM_CORES_PER_BLOCKDIM]{};
    uint32_t previous_mix_ref{UINT32_MAX};
    uint32_t cooperative_count{0};

    void append(
        uint32_t rank, uint32_t ref, const SchedulerTaskMetadata &task, std::vector<SchedulerTaskCatalogOrder> &rows,
        uint32_t &first_mix_ref
    ) {
        auto &row = rows[ref];
        row.rank = rank;
        if (!scheduler_task_is_executable(task.flags) || scheduler_task_is_inline(task.flags)) return;
        const bool mix = scheduler_task_is_mix(task.flags);
        if (mix || scheduler_task_requires_sync_start(task.flags)) row.cooperative_index = cooperative_count++;
        const auto lanes = scheduler_task_claim_lanes(task.flags, task.active_mask).merged_with(task.active_mask);
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            if (!lanes.has(static_cast<SubtaskSlot>(lane))) continue;
            row.required_floor[lane] = mix ? previous_sync_floor[lane] : previous_mix_floor[lane];
            if (mix) previous_mix_floor[lane] = rank + 1;
            else if (scheduler_task_requires_sync_start(task.flags)) previous_sync_floor[lane] = rank + 1;
        }
        if (mix) {
            if (previous_mix_ref == UINT32_MAX) first_mix_ref = ref;
            else rows[previous_mix_ref].next_mix_ref = ref;
            previous_mix_ref = ref;
        }
    }
};

// The ref-order form is for callers supplying an already ordered image.
// Host binding passes the full order of all expanded nodes, including controls.
inline bool scheduler_project_task_catalog(
    const std::vector<SchedulerTaskMetadata> &metadata, const SchedulerTaskOrder *order, SchedulerTaskCatalogPlan *plan
) {
    const size_t count = metadata.size();
    std::vector<std::vector<int64_t>> catalogs(SCHEDULER_EXECUTION_ENGINE_COUNT);
    std::vector<SchedulerTaskCatalogOrder> rows(count);
    SchedulerCatalogOrderBuilder builder;
    uint32_t first_mix = UINT32_MAX;
    for (uint32_t rank = 0; rank < count; ++rank) {
        const int64_t ref = order == nullptr ? static_cast<int64_t>(rank) : order->task_order[rank];
        if (ref < 0 || static_cast<uint64_t>(ref) >= count || (order != nullptr && order->task_ranks[ref] != rank))
            return false;
        const auto &task = metadata[ref];
        builder.append(rank, static_cast<uint32_t>(ref), task, rows, first_mix);
        if (!scheduler_task_is_executable(task.flags) || scheduler_task_is_inline(task.flags)) continue;
        const auto type = scheduler_task_execution_engine(task.flags, task.active_mask);
        catalogs[type].push_back(ref);
    }
    for (uint32_t engine = 0; engine < SCHEDULER_EXECUTION_ENGINE_COUNT; ++engine)
        if (catalogs[engine].size() != plan->descriptors[engine].capacity) return false;
    plan->header.first_mix_ref = first_mix;
    plan->order_rows = std::move(rows);
    plan->catalogs = std::move(catalogs);
    return true;
}

inline bool scheduler_apply_task_order(
    SchedulerTaskOrder ordered, const std::vector<SchedulerTaskMetadata> &metadata, SchedulerTaskCatalogPlan *plan,
    std::string *error = nullptr
) {
    const size_t count = metadata.size();
    if (plan == nullptr || ordered.task_order.size() != count || ordered.task_levels.size() != count ||
        ordered.task_bottom_levels.size() != count || ordered.task_ranks.size() != count ||
        plan->descriptors.size() != SCHEDULER_EXECUTION_ENGINE_COUNT ||
        !scheduler_project_task_catalog(metadata, &ordered, plan)) {
        if (error != nullptr) *error = "invalid projected task order";
        return false;
    }
    plan->task_order = std::move(ordered.task_order);
    plan->task_levels = std::move(ordered.task_levels);
    plan->task_bottom_levels = std::move(ordered.task_bottom_levels);
    plan->task_ranks = std::move(ordered.task_ranks);
    if (error != nullptr) error->clear();
    return true;
}

inline bool scheduler_can_defer_side_tasks(
    bool native_graph, bool has_deferred_work, uint64_t control_count,
    const std::vector<SchedulerTaskMetadata> &metadata
) {
    if (native_graph || has_deferred_work || control_count != 0) return false;
    return std::none_of(metadata.begin(), metadata.end(), [](const SchedulerTaskMetadata &task) {
        return scheduler_task_has_predicate(task.flags);
    });
}

inline bool scheduler_order_task_catalog(
    const SchedulerDependencyPlan &decoded, const std::vector<SchedulerTaskMetadata> &metadata,
    SchedulerTaskCatalogPlan *plan, std::string *error = nullptr, bool defer_side_tasks = false
) {
    const size_t count = metadata.size();
    if (plan == nullptr || plan->header.task_count != count || decoded.fanin.size() != count ||
        plan->descriptors.size() != SCHEDULER_EXECUTION_ENGINE_COUNT)
        return false;
    std::vector<std::vector<int64_t>> successors(count);
    for (size_t consumer = 0; consumer < count; ++consumer)
        for (const auto producer : decoded.fanin[consumer])
            successors[producer].push_back(static_cast<int64_t>(consumer));
    SchedulerTaskOrder ordered;
    if (!scheduler_topological_task_order(successors, metadata, plan->cluster_count, defer_side_tasks, &ordered, error))
        return false;
    for (size_t producer = 0; producer < count; ++producer)
        for (const auto consumer : successors[producer])
            if (ordered.task_ranks[producer] >= ordered.task_ranks[consumer]) {
                if (error != nullptr) *error = "task ordering violates a predecessor constraint";
                return false;
            }
    return scheduler_apply_task_order(std::move(ordered), metadata, plan, error);
}

// The original topology fixes issue order. Fold pure controls with at most one consumer within the fan-in limit.
inline std::vector<int64_t> scheduler_fold_ordinary_controls(
    std::vector<std::vector<int64_t>> &fanin, const std::vector<bool> &host_done,
    const std::vector<SchedulerTaskMetadata> &metadata, uint64_t capacity
) {
    std::vector<bool> active(fanin.size()), controls(fanin.size());
    bool has_live_control = false;
    for (size_t ref = 0; ref < fanin.size(); ++ref) {
        active[ref] = !host_done[ref];
        controls[ref] = active[ref] && scheduler_task_is_inline(metadata[ref].flags);
        has_live_control |= controls[ref];
    }
    if (!has_live_control) return {};

    uint64_t edges = 0;
    for (const auto &row : fanin)
        edges += row.size();
    if (edges <= capacity)
        return scheduler_fold_control_fanin(fanin, std::move(active), controls, SCHEDULER_GRAPH_MAX_FANIN);

    // Aliased rows can represent more logical edges than the pool holds.
    // Fold a copy when separate rows could exceed the pool capacity.
    auto packed = fanin;
    auto folded = scheduler_fold_control_fanin(packed, std::move(active), controls, SCHEDULER_GRAPH_MAX_FANIN);
    if (folded.empty()) return {};
    edges = 0;
    for (const auto &row : packed)
        edges += row.size();
    if (edges > capacity) return {};
    fanin = std::move(packed);
    return folded;
}

// Reduce transitive edges in the decoded rows and derive the final dependencies. Keep row addresses stable
// unless folding succeeds; repacked rows start at offsets determined by their lengths before reduction.
template <typename Storage>
inline bool scheduler_prepare_ordinary_fanin(
    Storage *storage, int32_t *pool, uint64_t capacity, std::vector<SchedulerTaskMetadata> &metadata,
    std::vector<int64_t> &completed, SchedulerTaskCatalogPlan &catalog, SchedulerDependencyPlan &dependencies,
    std::string *error = nullptr
) {
    auto &rows = dependencies.fanin;
    const auto folded = scheduler_fold_ordinary_controls(rows, dependencies.host_done, metadata, capacity);
    if (folded.size() > catalog.header.control_count) {
        if (error != nullptr) *error = "folded control count exceeds the task catalog";
        return false;
    }
    if (folded.empty()) {
        std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
        for (size_t ref = 0; ref < rows.size(); ++ref) {
            if (rows[ref].empty()) continue;
            const auto begin = reinterpret_cast<uintptr_t>(storage[ref].payload.fanin_data());
            ranges.emplace_back(begin, begin + rows[ref].size() * sizeof(int32_t));
        }
        std::sort(ranges.begin(), ranges.end());
        for (size_t i = 1; i < ranges.size(); ++i)
            if (ranges[i].first < ranges[i - 1].second) {
                dependencies.collect_sinks();
                return true;
            }
    } else {
        catalog.header.control_count -= folded.size();
        for (const auto ref : folded) {
            metadata[ref] = {};
            dependencies.host_done[ref] = true;
            completed.push_back(ref);
        }
        std::sort(completed.begin(), completed.end());
    }
    std::vector<uint64_t> ancestors(rows.size(), 0);
    uint64_t cursor = 0;
    for (size_t ref = 0; ref < rows.size(); ++ref) {
        auto &row = rows[ref];
        auto &payload = storage[ref].payload;
        if (!folded.empty()) {
            payload.fanin.set(row.empty() ? nullptr : pool + cursor);
            cursor += row.size();
            if (row.empty()) payload.fanin_count = 0;
        }
        if (row.empty()) continue;
        uint64_t direct = 0, implied = 0;
        for (const auto producer : row) {
            const size_t distance = ref - producer;
            if (distance <= 64) direct |= uint64_t{1} << (distance - 1);
            if (distance < 64) implied |= ancestors[producer] << distance;
        }
        ancestors[ref] = direct | implied;
        auto end = std::remove_if(row.begin(), row.end(), [&](int64_t producer) {
            const size_t distance = ref - producer;
            return distance <= 64 && (implied & (uint64_t{1} << (distance - 1))) != 0;
        });
        std::stable_sort(row.begin(), end, [&](int64_t left, int64_t right) {
            return catalog.task_ranks[left] < catalog.task_ranks[right];
        });
        std::transform(row.begin(), end, payload.fanin_data(), [](int64_t producer) {
            return static_cast<int32_t>(producer);
        });
        row.erase(end, row.end());
        payload.fanin_count = static_cast<int32_t>(row.size());
    }
    dependencies.collect_sinks();
    return true;
}

inline bool scheduler_size_task_catalog(
    const std::vector<SchedulerTaskMetadata> &metadata, uint32_t clusters, uint64_t start_offset,
    SchedulerTaskCatalogPlan *plan
) {
    if (plan == nullptr || metadata.size() > INT32_MAX || clusters == 0 || clusters > SCHEDULER_CLUSTER_CAPACITY ||
        start_offset == 0)
        return false;
    SchedulerTaskCatalogPlan next{};
    next.cluster_count = clusters;
    auto &header = next.header;
    header.task_count = metadata.size();
    next.descriptors.resize(SCHEDULER_EXECUTION_ENGINE_COUNT);
    for (const auto &task : metadata) {
        if (scheduler_task_is_inline(task.flags)) {
            constexpr uint8_t forbidden =
                SCHEDULER_TASK_MIX | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START | SCHEDULER_TASK_HAS_PREDICATE;
            if (!scheduler_task_is_executable(task.flags) || (task.flags & forbidden) != 0 ||
                task.logical_block_num != 1 || task.kernel_ids[0] != UINT16_MAX || task.kernel_ids[1] != UINT16_MAX ||
                task.kernel_ids[2] != UINT16_MAX)
                return false;
        }
        if (!scheduler_task_is_executable(task.flags) || scheduler_task_is_inline(task.flags)) {
            if (task.logical_block_num != 0) ++header.control_count;
            continue;
        }
        if (!task.active_mask.is_valid() || task.active_mask.is_empty() ||
            task.active_mask.is_mix() != scheduler_task_is_mix(task.flags))
            return false;
        const uint32_t type = scheduler_task_execution_engine(task.flags, task.active_mask);
        ++next.descriptors[type].capacity;
        ++header.entry_count;
    }
    uint64_t cursor = start_offset;
    if (!scheduler_layout_reserve(&cursor, sizeof(header), SCHEDULER_STATE_ALIGNMENT, &next.header_offset) ||
        !scheduler_layout_reserve(
            &cursor, next.descriptors.size() * sizeof(SchedulerTaskCatalogDescriptor), SCHEDULER_STATE_ALIGNMENT,
            &header.descriptors_offset
        ))
        return false;
    for (auto &descriptor : next.descriptors)
        if (!scheduler_layout_reserve(
                &cursor, descriptor.capacity * sizeof(int64_t), SCHEDULER_STATE_ALIGNMENT, &descriptor.entries_offset
            ))
            return false;
    if (!scheduler_layout_reserve(
            &cursor, metadata.size() * sizeof(SchedulerTaskCatalogOrder), SCHEDULER_STATE_ALIGNMENT,
            &header.order_offset
        ) ||
        !scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size))
        return false;
    header.storage_end_offset = next.total_size;
    *plan = std::move(next);
    return true;
}

inline bool scheduler_plan_task_catalog(
    const std::vector<SchedulerTaskMetadata> &metadata, uint32_t clusters, uint64_t start_offset,
    SchedulerTaskCatalogPlan *plan
) {
    SchedulerTaskCatalogPlan next;
    if (plan == nullptr || !scheduler_size_task_catalog(metadata, clusters, start_offset, &next) ||
        !scheduler_project_task_catalog(metadata, nullptr, &next))
        return false;
    *plan = std::move(next);
    return true;
}

// The caller owns a quiescent allocation already zeroed through total_size.
inline void scheduler_init_task_catalog(void *base, const SchedulerTaskCatalogPlan &plan) {
    *scheduler_state_at<SchedulerTaskCatalogHeader>(base, plan.header_offset) = plan.header;
    auto *rows = scheduler_state_at<SchedulerTaskCatalogOrder>(base, plan.header.order_offset);
    std::copy(plan.order_rows.begin(), plan.order_rows.end(), rows);
    auto *descriptors = scheduler_state_at<SchedulerTaskCatalogDescriptor>(base, plan.header.descriptors_offset);
    std::copy(plan.descriptors.begin(), plan.descriptors.end(), descriptors);
    for (uint32_t type = 0; type < plan.descriptors.size(); ++type) {
        auto *entries = scheduler_state_at<int64_t>(base, plan.descriptors[type].entries_offset);
        std::copy(plan.catalogs[type].begin(), plan.catalogs[type].end(), entries);
    }
}
