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
/**
 * Host-side Graph templates: what percounter keeps of one Graph Definition.
 *
 * A template is built once per Definition per bind, however many invocations
 * share it, and holds only what a lane needs to schedule the body: each task's
 * subtask total, its fanin rows, and the body's tasks in order. Materializing a
 * task -- its kernel ids, arguments and predicate -- stays with upstream, which
 * the AICPU calls per invocation.
 *
 * The Definition is read through upstream's own decoders over the bind's
 * staging block, the bytes the device later receives as its Definition section.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <utility>
#include <vector>

#include "graph_image_view.h"
#include "region_plan.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/wire_types.h"

namespace simpler::percounter {

/** One body task, as read from its SubTaskDefinition. */
struct BodyTaskShape {
    uint8_t active_mask;
    bool sync_start;
    bool has_predicate;
    int32_t block_num;
    int32_t total_required_subtasks;
};

enum class TemplateStatus : uint8_t {
    OK,
    MALFORMED,    // the Definition contradicts itself; fails the bind
    UNSUPPORTED,  // a body task percounter does not run; the bind fails
};

/** Why a template was not built, and on which body task (-1 when on none). */
struct TemplateResult {
    TemplateStatus status{TemplateStatus::OK};
    const char *reason{""};
    int32_t body_index{-1};
};

/**
 * Build a template from the body's shapes and fanin rows.
 *
 * Body tasks use a virtual ready/resource schedule, with b-level priority among
 * ready tasks. Every producer precedes its consumers. The body is
 * one range in each list, so this order and the outer graph's decide the whole
 * schedule; see build_tables for the outer half.
 *
 * An SPMD body task takes one boundary entry. Its private claim counter
 * allocates blocks independently of fanin readiness.
 */
inline TemplateResult build_graph_template(
    const BodyTaskShape *shapes, int32_t task_count, const int32_t *fanin_offsets, const int32_t *fanin_indices,
    int32_t edge_count, uint64_t worker_count, HostGraphTemplate *out
) {
    const auto fail = [](TemplateStatus status, const char *reason, int32_t index) {
        return TemplateResult{status, reason, index};
    };
    if (out == nullptr || shapes == nullptr || fanin_offsets == nullptr || task_count <= 0 || edge_count < 0 ||
        (edge_count != 0 && fanin_indices == nullptr))
        return fail(TemplateStatus::MALFORMED, "empty or null body", -1);
    if (fanin_offsets[0] != 0 || fanin_offsets[task_count] != edge_count)
        return fail(TemplateStatus::MALFORMED, "fanin rows do not cover the edges", -1);

    HostGraphTemplate next{};
    const size_t n = static_cast<size_t>(task_count);
    next.tasks.resize(n);
    next.fanin_offsets.assign(fanin_offsets, fanin_offsets + n + 1);
    next.fanin_indices.assign(fanin_indices, fanin_indices + edge_count);

    std::vector<bool> has_consumer(n, false);
    for (int32_t i = 0; i < task_count; ++i) {
        const int32_t begin = fanin_offsets[i];
        const int32_t end = fanin_offsets[i + 1];
        if (begin > end || end > edge_count) return fail(TemplateStatus::MALFORMED, "fanin row out of order", i);
        for (int32_t e = begin; e < end; ++e) {
            const int32_t producer = fanin_indices[e];
            if (producer < 0 || producer >= i)
                return fail(TemplateStatus::MALFORMED, "a fanin edge does not point backwards", i);
            has_consumer[static_cast<size_t>(producer)] = true;
        }
    }

    std::vector<int64_t> eligible[PERCOUNTER_LIST_COUNT];
    for (int32_t i = 0; i < task_count; ++i) {
        const BodyTaskShape &shape = shapes[i];
        // A dependency-only task runs no kernel but still orders its consumers
        // after its producers. It takes one entry in the AIV list, the list
        // every vector lane draws, and counts as one subtask: the lane that
        // reaches it publishes it once its producers have.
        const bool dummy = shape.active_mask == 0;
        uint8_t core_type = 1;
        const TaskPlacement placement =
            dummy ? TaskPlacement::AIV : task_placement_from_active_mask(shape.active_mask, &core_type);
        if (placement == TaskPlacement::UNSUPPORTED)
            return fail(TemplateStatus::UNSUPPORTED, "an AIV1-only body task", i);
        // A cohort of one would be released by its own claim, and the sync
        // sentinel must stay a value no real block index takes.
        if (shape.sync_start && (dummy || shape.block_num < 2 || shape.block_num >= PERCOUNTER_ORDER_BLOCK_IDX_SPMD))
            return fail(TemplateStatus::MALFORMED, "a sync-start body task with an impossible block count", i);
        // A cohort wider than the lanes that can serve it never fills: each of
        // them takes a block and waits for one nobody is left to take. The
        // entry gate requires every block. Body and top-level tasks use the same
        // worker-demand check and topology bound.
        SchedulerMinimumWorkerDemand demand{};
        if (shape.sync_start && !scheduler_resident_worker_demand_checked(
                                    lane_mask_from_active_mask(shape.active_mask),
                                    static_cast<uint32_t>(shape.block_num), true, worker_count, &demand
                                ))
            return fail(TemplateStatus::UNSUPPORTED, "a sync-start body task wider than the worker pool", i);
        const int32_t subtasks = dummy ? 1 : __builtin_popcount(shape.active_mask);
        if (dummy ? (shape.block_num != 1 || shape.total_required_subtasks != 0 || shape.has_predicate) :
                    (shape.block_num < 1 || shape.block_num > UINT16_MAX / subtasks ||
                     shape.total_required_subtasks != shape.block_num * subtasks))
            return fail(TemplateStatus::MALFORMED, "inconsistent block and subtask counts", i);
        // The same scope the task table has: a predicate on one block of one
        // subtask, which is all the lane evaluates.
        if (shape.has_predicate && (subtasks != 1 || shape.block_num != 1))
            return fail(TemplateStatus::UNSUPPORTED, "a predicate on a multi-block or MIX body task", i);

        HostBodyTask &task = next.tasks[static_cast<size_t>(i)];
        task.subtask_total = static_cast<uint32_t>(shape.block_num * subtasks);
        task.flags = 0;
        task.metadata = {
            0, static_cast<uint16_t>(shape.active_mask), static_cast<uint16_t>(shape.block_num),
            shape.sync_start ? 0U : PERCOUNTER_NO_SYNC, PERCOUNTER_NO_BOUNDARY
        };
        if (fanin_offsets[i] == fanin_offsets[i + 1]) task.flags |= PERCOUNTER_BODY_ROOT;
        if (!has_consumer[static_cast<size_t>(i)]) {
            task.flags |= PERCOUNTER_BODY_SINK;
            next.sink_subtask_total += task.subtask_total;
        }
        if (shape.has_predicate) {
            task.flags |= PERCOUNTER_BODY_PREDICATE;
            next.has_predicates = true;
        }
        next.executable_blocks += static_cast<uint64_t>(shape.block_num);
        if (shape.block_num > 1 || placement == TaskPlacement::MIX) ++next.boundary_task_count;
        if (shape.sync_start) {
            ++next.sync_cohort_count;
        }
        if (placement == TaskPlacement::MIX) {
            eligible[1].push_back(i);
            ++next.mix_task_count;
        } else {
            eligible[core_type].push_back(i);
        }
    }

    if (worker_count == 0 || worker_count % PLATFORM_CORES_PER_BLOCKDIM != 0 ||
        worker_count / PLATFORM_CORES_PER_BLOCKDIM > UINT32_MAX)
        return fail(TemplateStatus::MALFORMED, "invalid worker topology", -1);
    std::vector<ResourceOrderTask> order_tasks(n);
    for (int32_t i = 0; i < task_count; ++i) {
        auto &node = order_tasks[static_cast<size_t>(i)];
        node.active_mask = shapes[i].active_mask;
        node.blocks = static_cast<uint32_t>(shapes[i].block_num);
        node.sync = shapes[i].sync_start;
        if (node.active_mask == 0) node.duration = 0;
        for (int32_t e = fanin_offsets[i]; e < fanin_offsets[i + 1]; ++e)
            node.predecessors.push_back(static_cast<uint32_t>(fanin_indices[e]));
    }
    const auto schedule =
        make_resource_order(order_tasks, static_cast<uint32_t>(worker_count / PLATFORM_CORES_PER_BLOCKDIM));
    if (!schedule) return fail(TemplateStatus::MALFORMED, "invalid resource schedule", -1);
    next.depth = schedule->depth;
    next.virtual_duration = schedule->duration;
    next.active_mask = schedule->active_mask;
    const auto &global_order = schedule->tasks;
    std::vector<uint32_t> position(n);
    for (uint32_t rank = 0; rank < n; ++rank)
        position[global_order[rank]] = rank;
    const auto by_schedule = [&position](int64_t a, int64_t b) {
        return position[static_cast<size_t>(a)] < position[static_cast<size_t>(b)];
    };
    for (size_t rank = 0; rank < n; ++rank)
        next.tasks[global_order[rank]].metadata.global_order_id = static_cast<uint32_t>(rank);
    for (uint32_t l = 0; l < PERCOUNTER_LIST_COUNT; ++l) {
        std::stable_sort(eligible[l].begin(), eligible[l].end(), by_schedule);
        for (int64_t i : eligible[l]) {
            const int32_t blocks = shapes[i].block_num;
            if (blocks > 1) {
                next.orders[l].push_back(percounter_order_entry(i, PERCOUNTER_ORDER_BLOCK_IDX_SPMD, blocks));
                continue;
            }
            next.orders[l].push_back(percounter_order_entry(i, 0, 1));
        }
    }
    *out = std::move(next);
    return TemplateResult{};
}

/**
 * Read the Definition at `image_offset` of `view` and build its template.
 *
 * Every read goes through upstream's bounded decoders, so a Definition the
 * device would refuse to localize is refused here too, as MALFORMED.
 */
inline TemplateResult
read_graph_template(const GraphImageView &view, uint32_t image_offset, uint64_t worker_count, HostGraphTemplate *out) {
    GraphImageValue object{};
    if (!graph_definition_decode_framed(view, image_offset, &object))
        return TemplateResult{TemplateStatus::MALFORMED, "the Definition does not decode", -1};
    const Graph &definition = object.definition;
    if (definition.task_count <= 0 || definition.task_count > SUB_TASK_MAX_NUM || definition.edge_count < 0)
        return TemplateResult{TemplateStatus::MALFORMED, "the Definition's counts are out of range", -1};

    const size_t n = static_cast<size_t>(definition.task_count);
    std::vector<BodyTaskShape> shapes(n);
    for (int32_t i = 0; i < definition.task_count; ++i) {
        SubTaskDefinition source{};
        if (!graph_definition_load_element<SubTaskDefinition>(
                view, object, definition.off_sub_tasks, definition.task_count, i, &source
            ))
            return TemplateResult{TemplateStatus::MALFORMED, "a body task does not decode", i};
        const TaskAttrs attrs(source.task_attrs);
        shapes[static_cast<size_t>(i)] = BodyTaskShape{
            source.active_mask.raw(), attrs.requires_sync_start(), attrs.has_predicate(), source.logical_block_num,
            source.total_required_subtasks
        };
    }
    std::vector<int32_t> rows(n + 1, 0);
    std::vector<uint16_t> wire_indices(static_cast<size_t>(definition.edge_count), 0);
    if (!graph_definition_copy_array<int32_t>(
            view, object, definition.off_fanin_offsets, definition.task_count + 1, rows.data()
        ) ||
        (definition.edge_count != 0 &&
         !graph_definition_copy_array<uint16_t>(
             view, object, definition.off_fanin_indices, definition.edge_count, wire_indices.data()
         )))
        return TemplateResult{TemplateStatus::MALFORMED, "the fanin rows do not decode", -1};
    const std::vector<int32_t> indices(wire_indices.begin(), wire_indices.end());
    return build_graph_template(
        shapes.data(), definition.task_count, rows.data(), indices.data(), definition.edge_count, worker_count, out
    );
}

}  // namespace simpler::percounter
