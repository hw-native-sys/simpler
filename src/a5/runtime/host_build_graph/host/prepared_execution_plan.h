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

#include "callable.h"
#include "scheduler/scheduler_async_layout.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_types.h"

enum class SchedulerCallableAddressSpace { DEVICE, HOST };

inline uint64_t scheduler_resolve_callable_entry(uint64_t object, SchedulerCallableAddressSpace address_space) {
    if (object == 0 || object % alignof(CoreCallable) != 0) return 0;
    if (address_space == SchedulerCallableAddressSpace::HOST)
        return reinterpret_cast<const CoreCallable *>(object)->resolved_addr();
    // Onboard registration patches resolved_addr to this binary address;
    // the device object itself is not readable through a Host pointer.
    constexpr uint64_t offset = CoreCallable::binary_data_offset();
    return object > UINT64_MAX - offset ? 0 : object + offset;
}

inline bool scheduler_prepare_host_calls(
    void *base, const AicoreSchedulerLayout &layout, uint64_t device_base, int64_t task_ref, const uint64_t *functions,
    uint64_t tensors_address, const uint64_t *scalars, uint32_t tensor_count, uint32_t scalar_count
) {
    if (base == nullptr || functions == nullptr || task_ref < 0 ||
        static_cast<uint64_t>(task_ref) >= layout.task_count || layout.prepared_layout_offset == 0 ||
        device_base == 0 || layout.total_size > UINT64_MAX - device_base || tensor_count > MAX_TENSOR_ARGS ||
        scalar_count > MAX_SCALAR_ARGS || tensor_count + scalar_count > SPMD_LOCAL_CONTEXT_INDEX ||
        (tensor_count != 0 &&
         (tensors_address == 0 || tensor_count > (UINT64_MAX - tensors_address) / TASKPAYLOAD_TENSOR_STRIDE)) ||
        (scalar_count != 0 && scalars == nullptr))
        return false;
    const auto &prepared = *scheduler_state_at<SchedulerPreparedLayout>(base, layout.prepared_layout_offset);
    const auto &plan = scheduler_state_at<SchedulerExecutionPlan>(base, prepared.plans_offset)[task_ref];
    auto *calls = scheduler_state_at<SchedulerPreparedCall>(base, prepared.calls_offset);
    auto *controls = scheduler_state_at<SchedulerPreparedControl>(base, prepared.controls_offset);
    if (prepared.version != SCHEDULER_PREPARED_VERSION || plan.execution_begin > prepared.execution_count ||
        plan.execution_count > prepared.execution_count - plan.execution_begin)
        return false;
    const SchedulerAsyncLayout *async = nullptr;
    const SchedulerAsyncTaskData *async_task = nullptr;
    if (layout.async_layout_offset != 0) {
        if (!scheduler_native_region_valid(
                layout.total_size, layout.async_layout_offset, 1, sizeof(SchedulerAsyncLayout),
                alignof(SchedulerAsyncLayout)
            ))
            return false;
        async = scheduler_state_at<SchedulerAsyncLayout>(base, layout.async_layout_offset);
        if (async->run_epoch != prepared.run_epoch || async->image_bytes != layout.total_size ||
            async->task_count != layout.task_count ||
            !scheduler_native_region_valid(
                layout.total_size, async->task_map_offset, async->task_count, sizeof(uint32_t), alignof(uint32_t)
            ) ||
            !scheduler_native_region_valid(
                layout.total_size, async->task_data_offset, async->async_task_count, sizeof(SchedulerAsyncTaskData),
                alignof(SchedulerAsyncTaskData)
            ) ||
            !scheduler_native_region_valid(
                layout.total_size, async->groups_offset, async->group_count, sizeof(pto2::resident::AsyncGroupData),
                alignof(pto2::resident::AsyncGroupData)
            ))
            return false;
        const uint32_t task_index = scheduler_state_at<uint32_t>(base, async->task_map_offset)[task_ref];
        if (task_index != pto2::resident::ASYNC_INVALID_INDEX) {
            if (task_index >= async->async_task_count) return false;
            async_task = scheduler_state_at<SchedulerAsyncTaskData>(base, async->task_data_offset) + task_index;
            if (async_task->task_ref != task_ref || async_task->block_count != plan.block_num ||
                async_task->async_mask.is_empty() || (async_task->async_mask.raw() & ~plan.active_mask_bits) != 0 ||
                async_task->group_base > async->group_count ||
                uint64_t{plan.block_num} * async_task->async_mask.count() > async->group_count - async_task->group_base)
                return false;
        }
    }
    auto *bindings = scheduler_state_at<SchedulerExecutionBinding>(base, prepared.bindings_offset);
    uint64_t index = plan.execution_begin;
    for (uint32_t block = 0; block < plan.block_num; ++block) {
        for (uint8_t lane = 0; lane < SUBTASK_SLOT_COUNT; ++lane) {
            if ((plan.active_mask_bits & (1U << lane)) == 0) continue;
            if (index >= plan.execution_begin + plan.execution_count || functions[lane] == 0 ||
                controls[index].state != 0)
                return false;
            auto &call = calls[index];
            call = {};
            call.function_bin_addr = functions[lane];
            call.run_epoch = prepared.run_epoch;
            call.execution_index = index;
            call.task_ref = task_ref;
            call.block_idx = block;
            call.num_args = static_cast<uint16_t>(tensor_count + scalar_count);
            call.subtask_slot = lane;
            for (uint32_t tensor = 0; tensor < tensor_count; ++tensor)
                call.args[tensor] = tensors_address + uint64_t{tensor} * TASKPAYLOAD_TENSOR_STRIDE;
            for (uint32_t scalar = 0; scalar < scalar_count; ++scalar)
                call.args[tensor_count + scalar] = scalars[scalar];
            const uint64_t binding = device_base + prepared.bindings_offset + index * sizeof(SchedulerExecutionBinding);
            call.args[SPMD_LOCAL_CONTEXT_INDEX] = binding + offsetof(SchedulerExecutionBinding, local_context);
            call.args[SPMD_GLOBAL_CONTEXT_INDEX] = binding + offsetof(SchedulerExecutionBinding, global_context);
            auto &global = bindings[index].global_context;
            global.async_group_index = pto2::resident::ASYNC_INVALID_INDEX;
            if (async_task != nullptr && async_task->async_mask.has(static_cast<SubtaskSlot>(lane))) {
                const uint32_t group = async_task->group_base + block * async_task->async_mask.count() +
                                       __builtin_popcount(async_task->async_mask.raw() & ((1U << lane) - 1));
                global.async_image_address = device_base;
                global.async_image_bytes = async->image_bytes;
                global.async_run_epoch = async->run_epoch;
                global.async_group_offset =
                    async->groups_offset + uint64_t{group} * sizeof(pto2::resident::AsyncGroupData);
                global.async_group_index = group;
            }
            controls[index].state = scheduler_prepared_state(SchedulerPreparedPhase::BUILT);
            ++index;
        }
    }
    return index == plan.execution_begin + plan.execution_count;
}
