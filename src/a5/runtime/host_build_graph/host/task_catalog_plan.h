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
#include <bitset>
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
    std::vector<int32_t> supply_inputs;
    std::vector<uint16_t> third_task_blocks;
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
            if (mix && task.active_mask.has(static_cast<SubtaskSlot>(lane)) && row.required_floor[lane] != 0)
                row.requires_peer_order = 1;
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

// An edge strictly decreases bottom level, so descending bottom level is topological.
// Only reorder consecutive synchronous ordinary tasks. Keep every resource or async
// boundary at its original rank, including when a dependency is external to this graph.
inline void scheduler_prioritize_ordinary_segments(
    const std::vector<SchedulerTaskMetadata> &metadata, const std::vector<LaneMask> &async_lanes,
    const SchedulerDependencyPlan &deps, const std::vector<std::vector<int64_t>> &successors, uint32_t clusters,
    SchedulerTaskOrder &order
) {
    const auto movable = [&](int64_t ref) {
        const auto flags = metadata[ref].flags;
        return scheduler_task_is_executable(flags) && !scheduler_task_is_inline(flags) &&
               !scheduler_task_is_mix(flags) && !scheduler_task_requires_sync_start(flags) &&
               !scheduler_task_has_predicate(flags) && async_lanes[ref].is_empty();
    };
    auto &refs = order.task_order;
    std::vector<uint8_t> pending(refs.size());
    std::vector<int64_t> packed;
    for (auto begin = refs.begin(); begin != refs.end();) {
        if (!movable(*begin)) {
            ++begin;
            continue;
        }
        const auto end = std::find_if_not(begin, refs.end(), movable);
        std::stable_sort(begin, end, [&](int64_t left, int64_t right) {
            return order.task_bottom_levels[left] > order.task_bottom_levels[right];
        });
        // Give the critical branch first choice. Independent siblings may use the rest of
        // this wave if every predecessor already precedes them in the new directory.
        for (auto it = begin; it != end; ++it)
            pending[*it] = 1;
        packed.clear();
        for (auto it = begin; it != end; ++it) {
            const auto first = *it;
            if (!pending[first]) continue;
            packed.push_back(first);
            pending[first] = 0;
            const auto &task = metadata[first];
            std::array<uint32_t, 2> free{clusters, clusters * PLATFORM_AIV_CORES_PER_BLOCKDIM};
            const size_t engine = task.active_mask.has_aic() ? 0 : 1;
            if (task.logical_block_num >= free[engine]) continue;
            free[engine] -= task.logical_block_num;
            // Use the latest executable predecessor; folded controls often fan out to the whole graph.
            int64_t fork = -1;
            for (const auto pred : deps.fanin[first]) {
                if (!scheduler_task_is_executable(metadata[pred].flags) ||
                    scheduler_task_is_inline(metadata[pred].flags))
                    continue;
                if (fork < 0 || order.task_ranks[pred] > order.task_ranks[fork]) fork = pred;
            }
            if (fork < 0) continue;
            // Packing is optional. Inspect at most one wave of candidate siblings per first task.
            // Successor rows are sorted by ref. On wide forks, start just after this child so
            // later waves can still pack their nearby siblings without rescanning the entire row.
            const auto &siblings = successors[fork];
            const size_t budget = clusters * PLATFORM_CORES_PER_BLOCKDIM;
            auto sibling_it =
                siblings.size() > budget ? std::upper_bound(siblings.begin(), siblings.end(), first) : siblings.begin();
            for (size_t checked = 0; checked < std::min(siblings.size(), budget); ++checked) {
                if (sibling_it == siblings.end()) sibling_it = siblings.begin();
                const auto next = *sibling_it++;
                if (!pending[next] || order.task_levels[next] > order.task_levels[first]) continue;
                const auto &sibling = metadata[next];
                auto &remaining = free[sibling.active_mask.has_aic() ? 0 : 1];
                if (sibling.logical_block_num > remaining ||
                    std::any_of(deps.fanin[next].begin(), deps.fanin[next].end(), [&](int64_t pred) {
                        return pending[pred] != 0;
                    }))
                    continue;
                remaining -= sibling.logical_block_num;
                packed.push_back(next);
                pending[next] = 0;
            }
        }
        std::copy(packed.begin(), packed.end(), begin);
        for (auto it = begin; it != end; ++it)
            order.task_ranks[*it] = static_cast<uint32_t>(it - refs.begin());
        begin = end;
    }
}

// A fixed-wave MIX can be prepared before an independent, later-submitted ordinary branch.
// Keep its predecessors and the other direct inputs to its consumers ahead of it.
inline void scheduler_advance_fixed_mix(
    const std::vector<std::vector<int64_t>> &successors, const std::vector<SchedulerTaskMetadata> &metadata,
    const std::vector<LaneMask> *async_lanes, SchedulerTaskOrder &order
) {
    auto &refs = order.task_order;
    size_t begin = 0;
    for (size_t rank = 0; rank < refs.size(); ++rank) {
        const auto ref = refs[rank];
        const auto flags = metadata[ref].flags;
        const bool synchronous = scheduler_task_is_executable(flags) && !scheduler_task_is_inline(flags) &&
                                 !scheduler_task_has_predicate(flags) &&
                                 (async_lanes == nullptr || (*async_lanes)[ref].is_empty());
        if (synchronous && !scheduler_task_is_mix(flags) && !scheduler_task_requires_sync_start(flags)) continue;
        if (synchronous && scheduler_task_is_mix(flags) && scheduler_task_has_fixed_owners(flags)) {
            size_t pos = rank;
            while (pos > begin && refs[pos - 1] > ref) {
                // Advancing the vector directory must not take a pure Cube branch's earlier execution opportunity.
                if (metadata[refs[pos - 1]].active_mask.has_aic()) break;
                const auto &consumers = successors[refs[pos - 1]];
                if (std::any_of(consumers.begin(), consumers.end(), [&](int64_t child) {
                        return child == ref ||
                               std::binary_search(successors[ref].begin(), successors[ref].end(), child);
                    }))
                    break;
                --pos;
            }
            std::rotate(refs.begin() + pos, refs.begin() + rank, refs.begin() + rank + 1);
            for (size_t i = pos; i <= rank; ++i)
                order.task_ranks[refs[i]] = static_cast<uint32_t>(i);
        }
        begin = rank + 1;
    }
}

inline bool scheduler_order_task_catalog(
    const SchedulerDependencyPlan &decoded, const std::vector<SchedulerTaskMetadata> &metadata,
    SchedulerTaskCatalogPlan *plan, std::string *error = nullptr, bool defer_side_tasks = false,
    const std::vector<LaneMask> *async_lanes = nullptr
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
    if (async_lanes != nullptr) {
        if (async_lanes->size() != count) return false;
        scheduler_prioritize_ordinary_segments(
            metadata, *async_lanes, decoded, successors, plan->cluster_count, ordered
        );
    }
    if (defer_side_tasks || async_lanes != nullptr)
        scheduler_advance_fixed_mix(successors, metadata, async_lanes, ordered);
    for (size_t producer = 0; producer < count; ++producer)
        for (const auto consumer : successors[producer])
            if (ordered.task_ranks[producer] >= ordered.task_ranks[consumer]) {
                if (error != nullptr) *error = "task ordering violates a predecessor constraint";
                return false;
            }
    return scheduler_apply_task_order(std::move(ordered), metadata, plan, error);
}

// A cursor has passed every earlier entry on the same engine. Fixed owners on the other engine
// are known at launch. Neither fact proves data completion.
inline int32_t scheduler_catalog_supply_input(
    const SchedulerDependencyPlan &deps, const std::vector<SchedulerTaskMetadata> &metadata, int64_t ref,
    uint32_t engine
) {
    int32_t input = SCHEDULER_SUPPLY_STATIC;
    for (int64_t producer : deps.fanin[ref]) {
        const auto &task = metadata[producer];
        if (deps.host_done[producer] || scheduler_task_has_fixed_owners(task.flags) ||
            (scheduler_task_is_executable(task.flags) && !scheduler_task_is_inline(task.flags) &&
             scheduler_task_execution_engine(task.flags, task.active_mask) == engine))
            continue;
        if (input != SCHEDULER_SUPPLY_STATIC) return SCHEDULER_SUPPLY_SCAN;
        input = static_cast<int32_t>(producer);
    }
    return input;
}

inline void scheduler_plan_static_supply(
    const SchedulerDependencyPlan &deps, const std::vector<SchedulerTaskMetadata> &metadata,
    SchedulerTaskCatalogPlan &catalog
) {
    catalog.supply_inputs.assign(metadata.size(), SCHEDULER_SUPPLY_SCAN);
    catalog.third_task_blocks.assign(metadata.size(), 0);
    for (uint32_t engine = 0; engine < catalog.catalogs.size(); ++engine) {
        const auto &entries = catalog.catalogs[engine];
        uint32_t ordinary = 0;
        for (size_t i = 0; i < entries.size(); ++i) {
            const auto ref = entries[i];
            catalog.supply_inputs[ref] = scheduler_catalog_supply_input(deps, metadata, ref, engine);
            const auto flags = metadata[ref].flags;
            if (scheduler_task_has_fixed_owners(flags) || scheduler_task_is_mix(flags) ||
                scheduler_task_requires_sync_start(flags) || scheduler_task_has_predicate(flags)) {
                ordinary = 0;
            } else if (++ordinary == 3) {
                // Every issuer of this dynamic ordinary task already passed the two preceding seals.
                catalog.third_task_blocks[entries[i - 2]] = metadata[ref].logical_block_num;
                ordinary = 0;
            }
        }
    }
}

// Fixed MIX entries in this suffix have no unresolved supply checks: their AIV
// predecessors precede the proxy cursor, and any AIC predecessor has a static owner.
// The members still acquire every predecessor's logical completion before entry.
inline uint32_t scheduler_terminal_mix_begin(
    const SchedulerDependencyPlan &deps, const std::vector<SchedulerTaskMetadata> &metadata,
    const SchedulerTaskCatalogPlan &catalog
) {
    constexpr auto aiv = static_cast<uint32_t>(SchedulerExecutionEngine::AIV);
    const auto &refs = catalog.catalogs[aiv];
    if (deps.sinks.size() != 1 || refs.empty() || refs.back() != deps.sinks.front()) return UINT32_MAX;
    auto begin = refs.size();
    while (begin != 0) {
        const auto ref = refs[begin - 1];
        const auto flags = metadata[ref].flags;
        if (!scheduler_task_is_mix(flags) || !scheduler_task_has_fixed_owners(flags)) break;
        if (begin < refs.size()) {
            const auto &successor_fanin = deps.fanin[refs[begin]];
            if (std::find(successor_fanin.begin(), successor_fanin.end(), ref) == successor_fanin.end()) break;
        }
        if (scheduler_catalog_supply_input(deps, metadata, ref, aiv) != SCHEDULER_SUPPLY_STATIC) break;
        --begin;
    }
    return refs.size() - begin >= 2 ? static_cast<uint32_t>(begin) : UINT32_MAX;
}

// The two terminal MIX Calls have one immutable identity per physical lane. Additional inputs
// into either MIX are allowed: their completion is still checked on the execution path.
inline void scheduler_plan_terminal_mix_calls(
    const SchedulerDependencyPlan &deps, const std::vector<SchedulerTaskMetadata> &metadata,
    SchedulerTaskCatalogPlan &catalog
) {
    const auto &refs = catalog.catalogs[static_cast<uint32_t>(SchedulerExecutionEngine::AIV)];
    if (refs.size() < 3 || deps.sinks.size() != 1 || refs.back() != deps.sinks[0]) return;
    const auto first = refs.end() - 3;
    for (unsigned i = 0; i < 3; ++i) {
        const auto &task = metadata[first[i]];
        const bool mix = i != 0;
        const LaneMask lanes = mix ? LaneMask{true, true, true} : LaneMask::aiv0();
        if (task.active_mask != lanes || !scheduler_task_has_fixed_owners(task.flags) ||
            scheduler_task_requires_sync_start(task.flags) || scheduler_task_has_predicate(task.flags) ||
            task.logical_block_num != catalog.cluster_count * (mix ? 1 : PLATFORM_AIV_CORES_PER_BLOCKDIM))
            return;
    }
    for (unsigned i = 0; i < 2; ++i) {
        bool linked = false;
        for (size_t consumer = 0; consumer < deps.fanin.size(); ++consumer) {
            if (deps.host_done[consumer]) continue;
            const auto &fanin = deps.fanin[consumer];
            if (std::find(fanin.begin(), fanin.end(), first[i]) == fanin.end()) continue;
            if (consumer != static_cast<size_t>(first[i + 1])) return;
            linked = true;
        }
        if (!linked) return;
    }
    catalog.header.terminal_mix_refs[0] = static_cast<uint32_t>(first[1]);
    catalog.header.terminal_mix_refs[1] = static_cast<uint32_t>(first[2]);
}

// A member observes every data predecessor's logical completion before it applies its MIX floor.
// Track only contiguous prefixes of ordinary work. A larger rank on an independent branch cannot
// certify an earlier task. An uncertain edge or task contributes no completion proof.
inline void scheduler_elide_proven_mix_floors(
    const SchedulerDependencyPlan &deps, const std::vector<SchedulerTaskMetadata> &metadata,
    const std::vector<LaneMask> &possible_async_lanes, SchedulerTaskCatalogPlan &catalog
) {
    using LaneFloors = std::array<uint32_t, PLATFORM_CORES_PER_BLOCKDIM>;
    std::vector<LaneFloors> completed_prefix(metadata.size());
    LaneFloors last_ordinary{};
    for (const int64_t ref : catalog.task_order) {
        const auto &task = metadata[ref];
        if (deps.host_done[ref] || !scheduler_task_is_executable(task.flags) || scheduler_task_is_inline(task.flags))
            continue;
        auto &proof = completed_prefix[ref];
        const bool known = !scheduler_task_has_predicate(task.flags) && possible_async_lanes[ref].is_empty();
        if (known)
            for (const int64_t pred : deps.fanin[ref])
                for (size_t lane = 0; lane < proof.size(); ++lane)
                    proof[lane] = std::max(proof[lane], completed_prefix[pred][lane]);

        auto &row = catalog.order_rows[ref];
        const auto lanes = scheduler_task_claim_lanes(task.flags, task.active_mask).merged_with(task.active_mask);
        if (scheduler_task_is_mix(task.flags)) {
            // Keep the full peer protocol if even one participating lane can owe earlier work.
            bool all_clear = known;
            for (size_t lane = 0; lane < proof.size(); ++lane)
                if (lanes.has(static_cast<SubtaskSlot>(lane)) && proof[lane] < last_ordinary[lane]) all_clear = false;
            if (all_clear) {
                std::fill_n(row.required_floor, proof.size(), 0u);
                row.requires_peer_order = 0;
            }
        } else {
            for (size_t lane = 0; lane < proof.size(); ++lane) {
                if (!lanes.has(static_cast<SubtaskSlot>(lane))) continue;
                if (known && proof[lane] >= last_ordinary[lane]) proof[lane] = row.rank + 1;
                last_ordinary[lane] = row.rank + 1;
            }
        }
    }
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
    using Ancestors = std::bitset<128>;
    std::vector<Ancestors> ancestors(rows.size());
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
        Ancestors direct, implied;
        for (const auto producer : row) {
            const size_t distance = ref - producer;
            if (distance <= direct.size()) direct.set(distance - 1);
            if (distance < implied.size()) implied |= ancestors[producer] << distance;
        }
        ancestors[ref] = direct | implied;
        auto end = std::remove_if(row.begin(), row.end(), [&](int64_t producer) {
            const size_t distance = ref - producer;
            return dependencies.host_done[producer] || (distance <= implied.size() && implied.test(distance - 1));
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
                &cursor, descriptor.capacity * sizeof(SchedulerCatalogEntry), SCHEDULER_STATE_ALIGNMENT,
                &descriptor.entries_offset
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
        auto *entries = scheduler_state_at<SchedulerCatalogEntry>(base, plan.descriptors[type].entries_offset);
        for (const auto ref : plan.catalogs[type])
            *entries++ = {
                static_cast<int32_t>(ref), plan.supply_inputs.empty() ? SCHEDULER_SUPPLY_SCAN : plan.supply_inputs[ref],
                plan.third_task_blocks.empty() ? uint16_t{0} : plan.third_task_blocks[ref]
            };
    }
}
