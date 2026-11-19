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
#include <deque>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "scheduler/scheduler_task_catalog.h"
#include "dependency_plan.h"

struct SchedulerTaskCatalogPlan {
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
    SchedulerTaskOrder ordered;
    if (!scheduler_topological_task_order(successors, &ordered, error)) return false;
    auto &ranks = ordered.task_ranks;
    for (size_t predecessor = 0; predecessor < count; ++predecessor)
        for (const int64_t successor : successors[predecessor])
            if (ranks[predecessor] >= ranks[successor]) return fail("task ordering violates a predecessor constraint");
    return scheduler_apply_task_order(std::move(ordered), metadata, plan, error);
}

// Zero/single-consumer joins can move their dependencies to that consumer
// without increasing the edge count. The unpublished ordinary image owns the
// supplied pool; tensor/scalar regions and physical issuance order stay fixed.
// The caller has validated inline task shapes and fixed the issuance order.
template <typename Storage>
inline bool scheduler_fold_ordinary_controls(
    Storage *storage, int32_t *fanin_pool, uint64_t fanin_capacity, std::vector<SchedulerTaskMetadata> *metadata,
    std::vector<int64_t> *completed, std::string *error = nullptr
) {
    const auto fail = [&](const std::string &message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (metadata == nullptr || completed == nullptr || metadata->size() > INT32_MAX ||
        (!metadata->empty() && storage == nullptr) || (fanin_capacity != 0 && fanin_pool == nullptr))
        return fail("invalid ordinary control image");
    const size_t count = metadata->size();
    const uintptr_t pool_begin = reinterpret_cast<uintptr_t>(fanin_pool);
    if (fanin_capacity > (UINTPTR_MAX - pool_begin) / sizeof(int32_t)) return fail("invalid ordinary fanin pool");
    const uintptr_t pool_end = pool_begin + fanin_capacity * sizeof(int32_t);
    std::vector<bool> active(count, true), controls(count, false), queued(count, false);
    for (const int64_t ref : *completed) {
        if (ref < 0 || static_cast<uint64_t>(ref) >= count || !active[ref])
            return fail("invalid completed control task");
        active[ref] = false;
    }
    std::deque<uint32_t> pending;
    for (size_t ref = 0; ref < count; ++ref) {
        if (!active[ref] || !scheduler_task_is_inline((*metadata)[ref].flags)) continue;
        controls[ref] = true;
        queued[ref] = true;
        pending.push_back(static_cast<uint32_t>(ref));
    }
    if (pending.empty()) {
        if (error != nullptr) error->clear();
        return true;
    }
    std::vector<std::vector<int32_t>> fanin(count);
    std::vector<std::set<uint32_t>> consumers(count);
    uint64_t original_edges = 0;
    for (size_t ref = 0; ref < count; ++ref) {
        if (!active[ref]) continue;
        const auto &payload = storage[ref].payload;
        const int32_t edges = payload.fanin_count;
        const auto *refs = payload.fanin_data();
        const uintptr_t address = reinterpret_cast<uintptr_t>(refs);
        if (edges < 0 || edges > SCHEDULER_GRAPH_MAX_FANIN ||
            (edges != 0 && (refs == nullptr || address < pool_begin || address > pool_end ||
                            static_cast<uint64_t>(edges) > (pool_end - address) / sizeof(int32_t))))
            return fail("invalid control fanin row at task " + std::to_string(ref));
        for (int32_t edge = 0; edge < edges; ++edge) {
            const int32_t producer = refs[edge];
            if (producer < 0 || static_cast<uint64_t>(producer) >= ref)
                return fail("invalid control predecessor at task " + std::to_string(ref));
            fanin[ref].push_back(producer);
            consumers[producer].insert(static_cast<uint32_t>(ref));
        }
        original_edges += edges;
    }
    std::vector<int64_t> folded;
    const auto enqueue = [&](uint32_t ref) {
        if (active[ref] && controls[ref] && !queued[ref]) {
            queued[ref] = true;
            pending.push_back(ref);
        }
    };
    while (!pending.empty()) {
        const uint32_t ref = pending.front();
        pending.pop_front();
        queued[ref] = false;
        if (!active[ref] || consumers[ref].size() > 1) continue;
        const uint32_t consumer = consumers[ref].empty() ? UINT32_MAX : *consumers[ref].begin();
        std::vector<int32_t> merged;
        if (consumer != UINT32_MAX) {
            for (const int32_t producer : fanin[consumer])
                if (producer != static_cast<int32_t>(ref)) merged.push_back(producer);
            merged.insert(merged.end(), fanin[ref].begin(), fanin[ref].end());
            std::sort(merged.begin(), merged.end());
            merged.erase(std::unique(merged.begin(), merged.end()), merged.end());
            if (merged.size() > SCHEDULER_GRAPH_MAX_FANIN) continue;
        }
        std::vector<int32_t> affected = fanin[ref];
        for (const int32_t producer : fanin[ref])
            consumers[producer].erase(ref);
        if (consumer != UINT32_MAX) {
            affected.insert(affected.end(), fanin[consumer].begin(), fanin[consumer].end());
            for (const int32_t producer : fanin[consumer])
                consumers[producer].erase(consumer);
            fanin[consumer] = std::move(merged);
            for (const int32_t producer : fanin[consumer])
                consumers[producer].insert(consumer);
            enqueue(consumer);
        }
        active[ref] = false;
        fanin[ref].clear();
        folded.push_back(ref);
        for (const int32_t producer : affected)
            enqueue(static_cast<uint32_t>(producer));
    }
    uint64_t edges = 0;
    for (const auto &row : fanin)
        edges += row.size();
    if (edges > original_edges) return fail("control folding increased the edge count");
    // Aliased input rows can occupy less storage than their logical edge count.
    // Keep that valid image intact when a separate-row repack would not fit.
    if (folded.empty() || edges > fanin_capacity) {
        if (error != nullptr) error->clear();
        return true;
    }
    uint64_t cursor = 0;
    for (size_t ref = 0; ref < count; ++ref) {
        auto &payload = storage[ref].payload;
        payload.fanin_count = static_cast<int32_t>(fanin[ref].size());
        payload.fanin.set(fanin[ref].empty() ? nullptr : fanin_pool + cursor);
        if (!fanin[ref].empty()) std::copy(fanin[ref].begin(), fanin[ref].end(), fanin_pool + cursor);
        cursor += fanin[ref].size();
    }
    for (const int64_t ref : folded) {
        (*metadata)[ref] = {};
        completed->push_back(ref);
    }
    std::sort(completed->begin(), completed->end());
    if (error != nullptr) error->clear();
    return true;
}

// Resident ordinary tasks confirm all fanin DONE before execution or predicate
// skipping. Their logical DONE therefore certifies their transitive predecessors.
// The caller owns the unpublished image and has already fixed the issue order.
template <typename Storage>
inline bool scheduler_prepare_ordinary_fanin(
    Storage *storage, const std::vector<uint32_t> &task_ranks, const std::vector<int64_t> &completed,
    std::string *error = nullptr
) {
    const auto fail = [&](const std::string &message) {
        if (error != nullptr) *error = message;
        return false;
    };
    const size_t count = task_ranks.size();
    if (count > INT32_MAX || (count != 0 && storage == nullptr)) return fail("invalid ordinary fanin image");
    std::vector<bool> active(count, true);
    for (const int64_t ref : completed) {
        if (ref < 0 || static_cast<uint64_t>(ref) >= count) return fail("invalid completed fanin task");
        active[ref] = false;
    }
    // Validate all live rows before changing any of them. Completed nodes may
    // retain stale incoming references, as in task-catalog ordering.
    std::vector<std::pair<uintptr_t, uintptr_t>> ranges;
    ranges.reserve(count);
    for (size_t consumer = 0; consumer < count; ++consumer) {
        if (!active[consumer]) continue;
        const auto &payload = storage[consumer].payload;
        const int32_t edges = payload.fanin_count;
        const auto *refs = payload.fanin_data();
        if (task_ranks[consumer] >= count || edges < 0 || edges > SCHEDULER_GRAPH_MAX_FANIN ||
            (edges != 0 && refs == nullptr))
            return fail("invalid ordinary fanin row at task " + std::to_string(consumer));
        for (int32_t edge = 0; edge < edges; ++edge) {
            const int32_t producer = refs[edge];
            if (producer < 0 || static_cast<uint64_t>(producer) >= consumer || task_ranks[producer] >= count)
                return fail("invalid ordinary predecessor at task " + std::to_string(consumer));
        }
        if (edges != 0) {
            const uintptr_t begin = reinterpret_cast<uintptr_t>(refs);
            const uint64_t bytes = static_cast<uint64_t>(edges) * sizeof(int32_t);
            if (bytes > UINTPTR_MAX - begin) return fail("invalid ordinary fanin extent");
            ranges.emplace_back(begin, begin + bytes);
        }
    }
    // Both compaction and sorting overwrite the row. Aliased rows are valid,
    // but this helper owns no spare pool capacity to separate them safely.
    // Preserve the entire input before any mutation when live rows overlap.
    std::sort(ranges.begin(), ranges.end());
    for (size_t index = 1; index < ranges.size(); ++index) {
        if (ranges[index].first < ranges[index - 1].second) {
            if (error != nullptr) error->clear();
            return true;
        }
    }
    // Bit d-1 denotes a predecessor d refs earlier. The bounded window retains
    // distant edges and costs one Host word per task, independent of DAG depth.
    std::vector<uint64_t> ancestors(count, 0);
    for (size_t consumer = 0; consumer < count; ++consumer) {
        if (!active[consumer]) continue;
        auto &payload = storage[consumer].payload;
        auto *refs = payload.fanin_data();
        uint64_t direct = 0;
        uint64_t implied = 0;
        for (int32_t edge = 0; edge < payload.fanin_count; ++edge) {
            const size_t distance = consumer - refs[edge];
            if (distance > 64) continue;
            direct |= uint64_t{1} << (distance - 1);
            if (distance < 64) implied |= ancestors[refs[edge]] << distance;
        }
        // Host-completed nodes never acquire ancestors: their historical fanin
        // may be stale and was not observed by the resident scheduler.
        ancestors[consumer] = direct | implied;
        int32_t retained = 0;
        for (int32_t edge = 0; edge < payload.fanin_count; ++edge) {
            const size_t distance = consumer - refs[edge];
            if (distance > 64 || (implied & (uint64_t{1} << (distance - 1))) == 0) refs[retained++] = refs[edge];
        }
        payload.fanin_count = retained;
        if (retained < 2) continue;
        std::stable_sort(refs, refs + payload.fanin_count, [&](int32_t left, int32_t right) {
            return task_ranks[left] < task_ranks[right];
        });
    }
    if (error != nullptr) error->clear();
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
