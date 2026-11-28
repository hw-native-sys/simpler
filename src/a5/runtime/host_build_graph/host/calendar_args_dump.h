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

#include <cstdint>
#include <vector>

#include "aicpu/args_dump_aicpu.h"
#include "graph_image_view.h"
#include "host_build_graph/runtime.h"
#include "host_build_graph/shared_memory.h"
#include "image/image_pack.h"
#include "region_plan.h"
#include "scheduler/scheduler_args_dump.h"

namespace simpler::percounter {

struct CalendarArgsDumpPlan {
    SchedulerArgsDumpLayout layout;
    std::vector<SchedulerArgsDumpTask> tasks;
};

inline bool append_calendar_args_dump_task(
    CalendarArgsDumpPlan &plan, DumpArgsLevel level, SchedulerArgsDumpTask &task, int32_t tensors, int32_t scalars
) {
    if (tensors < 0 || tensors > CORE_MAX_TENSOR_ARGS || scalars < 0 || scalars > CORE_MAX_SCALAR_ARGS ||
        tensors + scalars > SPMD_LOCAL_CONTEXT_INDEX)
        return false;
    if ((tensors != 0 || scalars != 0) && dump_args_task_selected(level, task.dump_metadata.dump_arg_mask)) {
        task.tensor_count = static_cast<uint16_t>(tensors);
        task.scalar_count = static_cast<uint16_t>(scalars);
        plan.tasks.push_back(task);
    }
    return true;
}

inline bool plan_calendar_args_dump(
    Runtime *runtime, const SharedMemoryHandle &host_sm, const void *uploaded_sm,
    const sm_layout::SegmentOffsets &segments, const std::vector<SchedulerTaskMetadata> &metadata,
    const GraphPlanInput &graph, const GraphImageView &view, CalendarArgsDumpPlan *plan
) {
    const auto level = static_cast<DumpArgsLevel>(runtime->dev.dump_args_level);
    if (level == DumpArgsLevel::OFF) return true;
    const uint64_t device_sm = reinterpret_cast<uint64_t>(runtime->get_gm_sm_ptr());
    const auto image_address = [&](const void *address) {
        return device_sm +
               static_cast<uint64_t>(static_cast<const char *>(address) - static_cast<const char *>(uploaded_sm));
    };

    for (size_t id = 0; id < metadata.size(); ++id) {
        const auto &shape = metadata[id];
        if (shape.active_mask.is_empty() || shape.inline_task) continue;
        const auto &source = host_sm.header->tasks.storage_at(id);
        const auto &payload = task_image::payload(uploaded_sm, segments, id);
        SchedulerArgsDumpTask task{};
        task.task_ref = id;
        task.task_id = source.task.task_id;
        task.dump_metadata = source.payload.dump_metadata;
        task.active_mask = shape.active_mask.raw();
        task.expected_participants = static_cast<uint64_t>(shape.logical_block_num) * shape.active_mask.count();
        for (uint32_t role = 0; role < SUBTASK_SLOT_COUNT; ++role)
            task.kernel_id[role] = shape.kernel_ids[role];
        task.argument_region_base = device_sm;
        task.argument_region_bytes = runtime->dev.sm_image_bytes;
        if (payload.tensor_count > 0) task.tensors_address = image_address(payload.tensor_data());
        if (payload.scalar_count > 0) task.scalars_address = image_address(payload.scalar_data());
        if (!append_calendar_args_dump_task(*plan, level, task, payload.tensor_count, payload.scalar_count))
            return false;
    }

    uint64_t body_base = metadata.size();
    for (size_t index = 0; index < graph.invocation_count; ++index) {
        const auto &invocation = graph.invocations[index];
        const int32_t shell = static_cast<int32_t>(invocation.shell_task_id);
        const auto &outer = host_sm.header->tasks.storage_at(shell);
        GraphImageValue value{};
        if (!graph_definition_decode_framed(view, outer.task.graph_definition_offset, &value)) return false;
        const auto &definition = value.definition;
        for (int32_t id = 0; id < definition.task_count; ++id) {
            SubTaskDefinition child{};
            if (!graph_definition_load_element(
                    view, value, definition.off_sub_tasks, definition.task_count, id, &child
                ))
                return false;
            if (child.active_mask.is_empty()) continue;
            SchedulerArgsDumpTask task{};
            task.task_ref = body_base + id;
            task.task_id = TaskId::make_sub_task(shell, id);
            task.dump_metadata = child.dump_metadata;
            task.active_mask = child.active_mask.raw();
            task.expected_participants = child.total_required_subtasks;
            for (uint32_t role = 0; role < SUBTASK_SLOT_COUNT; ++role)
                task.kernel_id[role] = child.kernel_id[role];
            // The materializer binds the arguments before its calendar watermark.
            if (!append_calendar_args_dump_task(*plan, level, task, child.tensor_count, child.scalar_count))
                return false;
        }
        body_base += definition.task_count;
    }
    return scheduler_args_dump_plan_layout(
        body_base, plan->tasks.size(), runtime->get_worker_count(), !plan->tasks.empty(), &plan->layout
    );
}

inline bool initialize_calendar_args_dump_image(
    void *base, const CalendarArgsDumpPlan &plan, uint64_t run_epoch, DumpArgsLevel level
) {
    if (!scheduler_args_dump_init_image(base, plan.layout.total_size, plan.layout, run_epoch, level)) return false;
    for (const auto &task : plan.tasks)
        if (!scheduler_args_dump_select_task(base, task)) return false;
    return true;
}

}  // namespace simpler::percounter
