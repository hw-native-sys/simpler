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
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "scheduler/scheduler_task_catalog.h"
#include "dependency_plan.h"

struct SchedulerTaskOrderEdge {
    int64_t predecessor;
    int64_t successor;
};

struct SchedulerTaskCatalogPlan {
    uint64_t header_offset{0};
    uint64_t total_size{0};
    SchedulerTaskCatalogHeader header{};
    std::vector<SchedulerTaskCatalogDescriptor> descriptors;
    std::vector<std::vector<int64_t>> catalogs;
    std::vector<int64_t> task_order;
    std::vector<uint32_t> task_levels;
    std::vector<uint32_t> task_bottom_levels;
    std::vector<uint32_t> task_ranks;
    std::vector<SchedulerTaskOrderEdge> order_edges;
    std::vector<uint32_t> ordering_rows;
    std::vector<int64_t> ordering_predecessors;
};

struct SchedulerTaskOrder {
    std::vector<int64_t> task_order;
    std::vector<uint32_t> task_levels;
    std::vector<uint32_t> task_bottom_levels;
    std::vector<uint32_t> task_ranks;
};

inline bool scheduler_topological_task_order(
    std::vector<std::vector<int64_t>> successors, SchedulerTaskOrder *order, std::string *error = nullptr
) {
    const auto fail = [&](const std::string &message) {
        if (error != nullptr) *error = message;
        return false;
    };
    const size_t count = successors.size();
    if (order == nullptr || count > INT32_MAX) return fail("invalid topological task count");
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
    *order = {std::move(topological), std::move(levels), std::move(bottom_levels), std::move(ranks)};
    if (error != nullptr) error->clear();
    return true;
}

inline bool scheduler_apply_task_order(
    SchedulerTaskOrder ordered, const std::vector<SchedulerTaskMetadata> &metadata, SchedulerTaskCatalogPlan *plan,
    std::string *error = nullptr
) {
    const auto fail = [&](const std::string &message) {
        if (error != nullptr) *error = message;
        return false;
    };
    const size_t count = metadata.size();
    if (plan == nullptr || ordered.task_order.size() != count || ordered.task_levels.size() != count ||
        ordered.task_bottom_levels.size() != count || ordered.task_ranks.size() != count ||
        plan->descriptors.size() != SCHEDULER_EXECUTION_ENGINE_COUNT)
        return fail("invalid projected task order");
    std::vector<bool> seen(count, false);
    std::vector<std::vector<int64_t>> catalogs(SCHEDULER_EXECUTION_ENGINE_COUNT);
    for (const auto ref : ordered.task_order) {
        if (ref < 0 || static_cast<uint64_t>(ref) >= count || seen[ref]) return fail("invalid projected task identity");
        seen[ref] = true;
        const auto &task = metadata[ref];
        if (!scheduler_task_is_executable(task.flags) || scheduler_task_is_inline(task.flags)) continue;
        const auto type = scheduler_task_execution_engine(task.flags, task.active_mask);
        if (type >= SCHEDULER_EXECUTION_ENGINE_COUNT) return fail("invalid projected execution engine");
        catalogs[type].push_back(ref);
    }
    for (uint32_t engine = 0; engine < SCHEDULER_EXECUTION_ENGINE_COUNT; ++engine)
        if (catalogs[engine].size() != plan->descriptors[engine].capacity)
            return fail("projected catalog capacity mismatch");
    for (const auto &edge : plan->order_edges)
        if (edge.predecessor < 0 || edge.successor < 0 || static_cast<uint64_t>(edge.predecessor) >= count ||
            static_cast<uint64_t>(edge.successor) >= count ||
            ordered.task_ranks[edge.predecessor] >= ordered.task_ranks[edge.successor])
            return fail("projected order violates an issuance constraint");
    plan->catalogs = std::move(catalogs);
    plan->task_order = std::move(ordered.task_order);
    plan->task_levels = std::move(ordered.task_levels);
    plan->task_bottom_levels = std::move(ordered.task_bottom_levels);
    plan->task_ranks = std::move(ordered.task_ranks);
    if (error != nullptr) error->clear();
    return true;
}

inline bool scheduler_order_task_catalog(
    const SchedulerGraphAccess &graph, const std::vector<SchedulerTaskMetadata> &metadata,
    const std::vector<int64_t> &completed, SchedulerTaskCatalogPlan *plan, std::string *error = nullptr
) {
    const auto fail = [&](const std::string &message) {
        if (error != nullptr) *error = message;
        return false;
    };
    const size_t count = metadata.size();
    if (plan == nullptr || count > INT32_MAX || count != graph.task_count || plan->header.task_count != count ||
        plan->descriptors.size() != SCHEDULER_EXECUTION_ENGINE_COUNT)
        return fail("invalid task catalog dimensions");
    std::vector<bool> active(count, true);
    for (const int64_t ref : completed) {
        if (!scheduler_task_ref_valid(graph, ref)) return fail("invalid completed task " + std::to_string(ref));
        active[ref] = false;
    }
    std::vector<std::vector<int64_t>> successors(count);
    for (size_t consumer = 0; consumer < count; ++consumer) {
        if (!active[consumer]) continue;
        SchedulerFaninView view{};
        if (scheduler_node_fanin_view(graph, consumer, &view) != SchedulerGraphResult::OK)
            return fail("invalid fanin view at task " + std::to_string(consumer));
        for (uint32_t edge = 0; edge < view.count; ++edge) {
            int64_t producer = SCHEDULER_TASK_REF_INVALID;
            if (scheduler_fanin_ref(view, consumer, edge, &producer) != SchedulerGraphResult::OK)
                return fail("invalid data predecessor at task " + std::to_string(consumer));
            successors[producer].push_back(static_cast<int64_t>(consumer));
        }
    }
    // Ordering-only edges may point backwards in ref space; data fanin keeps
    // its original identities and completion semantics.
    for (const auto &edge : plan->order_edges) {
        if (!scheduler_task_ref_valid(graph, edge.predecessor) || !scheduler_task_ref_valid(graph, edge.successor) ||
            edge.predecessor == edge.successor)
            return fail(
                "invalid ordering edge " + std::to_string(edge.predecessor) + " -> " + std::to_string(edge.successor)
            );
        if (active[edge.successor]) successors[edge.predecessor].push_back(edge.successor);
    }
    SchedulerTaskOrder ordered;
    if (!scheduler_topological_task_order(successors, &ordered, error)) return false;
    auto &ranks = ordered.task_ranks;
    for (size_t predecessor = 0; predecessor < count; ++predecessor)
        for (const int64_t successor : successors[predecessor])
            if (ranks[predecessor] >= ranks[successor]) return fail("task ordering violates a predecessor constraint");
    return scheduler_apply_task_order(std::move(ordered), metadata, plan, error);
}

inline bool scheduler_plan_task_catalog(
    const std::vector<SchedulerTaskMetadata> &metadata, uint32_t clusters, uint64_t start_offset,
    SchedulerTaskCatalogPlan *plan, const std::vector<SchedulerTaskOrderEdge> &order_edges = {}
) {
    if (plan == nullptr || metadata.size() > INT32_MAX || clusters == 0 || clusters > SCHEDULER_CLUSTER_CAPACITY ||
        start_offset == 0)
        return false;
    SchedulerTaskCatalogPlan next{};
    auto &header = next.header;
    header.task_count = metadata.size();
    next.descriptors.resize(SCHEDULER_EXECUTION_ENGINE_COUNT);
    next.catalogs.resize(SCHEDULER_EXECUTION_ENGINE_COUNT);
    for (uint64_t ref = 0; ref < metadata.size(); ++ref) {
        const auto &task = metadata[ref];
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
        const uint32_t type = scheduler_task_execution_engine(task.flags, task.active_mask);
        if (type >= SCHEDULER_EXECUTION_ENGINE_COUNT) return false;
        ++next.descriptors[type].capacity;
        next.catalogs[type].push_back(static_cast<int64_t>(ref));
        ++header.entry_count;
    }
    next.order_edges = order_edges;
    std::sort(next.order_edges.begin(), next.order_edges.end(), [](const auto &left, const auto &right) {
        return left.successor < right.successor ||
               (left.successor == right.successor && left.predecessor < right.predecessor);
    });
    next.order_edges.erase(
        std::unique(
            next.order_edges.begin(), next.order_edges.end(),
            [](const auto &left, const auto &right) {
                return left.predecessor == right.predecessor && left.successor == right.successor;
            }
        ),
        next.order_edges.end()
    );
    std::vector<SchedulerTaskOrderEdge> device_edges;
    for (const auto &edge : next.order_edges) {
        if (edge.predecessor < 0 || edge.successor < 0 || static_cast<uint64_t>(edge.predecessor) >= metadata.size() ||
            static_cast<uint64_t>(edge.successor) >= metadata.size() || edge.predecessor == edge.successor)
            return false;
        const auto &producer = metadata[edge.predecessor];
        const auto &consumer = metadata[edge.successor];
        if (!scheduler_task_is_executable(consumer.flags) || scheduler_task_is_inline(consumer.flags)) return false;
        const bool same_engine = scheduler_task_is_executable(producer.flags) &&
                                 !scheduler_task_is_inline(producer.flags) &&
                                 scheduler_task_execution_engine(producer.flags, producer.active_mask) ==
                                     scheduler_task_execution_engine(consumer.flags, consumer.active_mask);
        if (!same_engine) device_edges.push_back(edge);
    }
    if (device_edges.size() > INT32_MAX) return false;
    header.ordering_edge_count = static_cast<uint32_t>(device_edges.size());
    if (!device_edges.empty()) {
        next.ordering_rows.resize(metadata.size() + 1, 0);
        next.ordering_predecessors.reserve(device_edges.size());
        for (const auto &edge : device_edges) {
            ++next.ordering_rows[edge.successor + 1];
            next.ordering_predecessors.push_back(edge.predecessor);
        }
        for (size_t ref = 1; ref < next.ordering_rows.size(); ++ref)
            next.ordering_rows[ref] += next.ordering_rows[ref - 1];
    }
    uint64_t cursor = start_offset;
    if (!scheduler_layout_reserve(&cursor, sizeof(header), SCHEDULER_STATE_ALIGNMENT, &next.header_offset) ||
        !scheduler_layout_reserve(
            &cursor, next.descriptors.size() * sizeof(SchedulerTaskCatalogDescriptor), SCHEDULER_STATE_ALIGNMENT,
            &header.descriptors_offset
        ) ||
        !scheduler_layout_reserve(
            &cursor, SCHEDULER_EXECUTION_ENGINE_COUNT * sizeof(SchedulerIssueCursor), SCHEDULER_GM_ATOMIC_SECTOR_BYTES,
            &header.issue_cursors_offset
        ))
        return false;
    for (auto &descriptor : next.descriptors)
        if (!scheduler_layout_reserve(
                &cursor, descriptor.capacity * sizeof(int64_t), SCHEDULER_STATE_ALIGNMENT, &descriptor.entries_offset
            ))
            return false;
    if (!next.ordering_predecessors.empty()) {
        uint64_t refs_offset = 0;
        if (!scheduler_layout_reserve(
                &cursor, next.ordering_rows.size() * sizeof(uint32_t), SCHEDULER_STATE_ALIGNMENT,
                &header.ordering_rows_offset
            ) ||
            !scheduler_layout_reserve(
                &cursor, next.ordering_predecessors.size() * sizeof(int64_t), alignof(int64_t), &refs_offset
            ) ||
            refs_offset != scheduler_ordering_refs_offset(header))
            return false;
    }
    if (!scheduler_layout_checked_align(cursor, SCHEDULER_STATE_ALIGNMENT, &next.total_size)) return false;
    header.storage_end_offset = next.total_size;
    *plan = std::move(next);
    return true;
}

// The caller owns a quiescent allocation already zeroed through total_size.
inline void scheduler_init_task_catalog(void *base, const SchedulerTaskCatalogPlan &plan) {
    if (!plan.order_edges.empty() && plan.task_order.size() != plan.header.task_count)
        throw std::logic_error("ordering edges require a validated task catalog order");
    *scheduler_state_at<SchedulerTaskCatalogHeader>(base, plan.header_offset) = plan.header;
    auto *descriptors = scheduler_state_at<SchedulerTaskCatalogDescriptor>(base, plan.header.descriptors_offset);
    std::copy(plan.descriptors.begin(), plan.descriptors.end(), descriptors);
    auto *issues = scheduler_state_at<SchedulerIssueCursor>(base, plan.header.issue_cursors_offset);
    for (uint32_t type = 0; type < plan.descriptors.size(); ++type) {
        issues[type].next_task = 0;
        auto *entries = scheduler_state_at<int64_t>(base, plan.descriptors[type].entries_offset);
        std::copy(plan.catalogs[type].begin(), plan.catalogs[type].end(), entries);
    }
    if (plan.header.ordering_edge_count != 0) {
        auto *rows = scheduler_state_at<uint32_t>(base, plan.header.ordering_rows_offset);
        auto *refs = scheduler_state_at<int64_t>(base, scheduler_ordering_refs_offset(plan.header));
        std::copy(plan.ordering_rows.begin(), plan.ordering_rows.end(), rows);
        std::copy(plan.ordering_predecessors.begin(), plan.ordering_predecessors.end(), refs);
    }
}
