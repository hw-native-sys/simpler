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
/** Read each Graph body shape and its dependency DAG from the recorded Graph. */

#pragma once

#include <algorithm>
#include <array>
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
    std::array<int32_t, PLATFORM_CORES_PER_BLOCKDIM> kernel_ids{};
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
    if (worker_count == 0 || worker_count % PLATFORM_CORES_PER_BLOCKDIM != 0)
        return fail(TemplateStatus::MALFORMED, "invalid worker topology", -1);
    if (fanin_offsets[0] != 0 || fanin_offsets[task_count] != edge_count)
        return fail(TemplateStatus::MALFORMED, "fanin rows do not cover the edges", -1);

    HostGraphTemplate next{};
    const size_t n = static_cast<size_t>(task_count);
    next.tasks.resize(n);
    next.fanin_offsets.assign(fanin_offsets, fanin_offsets + n + 1);
    if (edge_count != 0) next.fanin_indices.assign(fanin_indices, fanin_indices + edge_count);

    std::vector<bool> has_consumer(n, false);
    for (int32_t i = 0; i < task_count; ++i) {
        const int32_t begin = fanin_offsets[i];
        const int32_t end = fanin_offsets[i + 1];
        if (begin < 0 || begin > end || end > edge_count)
            return fail(TemplateStatus::MALFORMED, "fanin row out of order", i);
        for (int32_t e = begin; e < end; ++e) {
            const int32_t producer = fanin_indices[e];
            if (producer < 0 || producer >= i)
                return fail(TemplateStatus::MALFORMED, "a fanin edge does not point backwards", i);
            has_consumer[static_cast<size_t>(producer)] = true;
        }
    }

    for (int32_t i = 0; i < task_count; ++i) {
        const BodyTaskShape &shape = shapes[i];
        // A dependency-only task publishes one completion credit on an AIV lane.
        const bool dummy = shape.active_mask == 0;
        uint8_t core_type = 1;
        const TaskPlacement placement =
            dummy ? TaskPlacement::AIV : task_placement_from_active_mask(shape.active_mask, &core_type);
        if (placement == TaskPlacement::UNSUPPORTED)
            return fail(TemplateStatus::UNSUPPORTED, "an AIV1-only body task", i);
        if (shape.sync_start &&
            (dummy || shape.block_num < 2 || static_cast<uint32_t>(shape.block_num) > PERCOUNTER_ORDER_BLOCK_MASK))
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
        HostBodyTask &task = next.tasks[static_cast<size_t>(i)];
        task.subtask_total = static_cast<uint32_t>(shape.block_num * subtasks);
        task.active_mask = shape.active_mask;
        task.block_num = static_cast<uint32_t>(shape.block_num);
        task.sync_start = shape.sync_start;
        task.has_predicate = shape.has_predicate;
        task.kernel_ids = shape.kernel_ids;
        task.sink = !has_consumer[static_cast<size_t>(i)];
        if (task.sink) next.sink_subtask_total += task.subtask_total;
        if (!dummy) next.defer_rows += task.subtask_total;
        next.executable_blocks += task.block_num;
        next.mix_task_count += placement == TaskPlacement::MIX;
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
inline TemplateResult read_graph_template(
    const GraphImageView &view, const GraphImageValue &object, uint64_t worker_count, HostGraphTemplate *out
) {
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
        auto &shape = shapes[static_cast<size_t>(i)];
        shape = BodyTaskShape{
            source.active_mask.raw(), attrs.requires_sync_start(), attrs.has_predicate(), source.logical_block_num,
            source.total_required_subtasks
        };
        std::copy_n(source.kernel_id, shape.kernel_ids.size(), shape.kernel_ids.begin());
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
