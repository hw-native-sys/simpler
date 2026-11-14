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

#include "scheduler_native_graph.h"

// The caller supplies the allocation's trusted length. No header field may
// authorize reading the header itself, including when backing is reused.
inline __aicore__ bool
scheduler_native_region_valid(uint64_t bytes, uint64_t offset, uint64_t count, uint64_t stride, uint64_t alignment) {
    return offset != 0 && offset % alignment == 0 && offset <= bytes && count <= (bytes - offset) / stride;
}

inline __aicore__ bool scheduler_native_header_extent_valid(uint64_t base, uint64_t bytes, uint64_t offset) {
    return base != 0 && base % SCHEDULER_STATE_ALIGNMENT == 0 && bytes <= UINT64_MAX - base &&
           offset >= sizeof(SchedulerRunControl) &&
           scheduler_native_region_valid(
               bytes, offset, 1, sizeof(SchedulerNativeGraphHeader), alignof(SchedulerNativeGraphHeader)
           );
}

inline __aicore__ __gm__ const SchedulerNativeGraphHeader *scheduler_native_header_checked(
    uint64_t base, uint64_t bytes, uint64_t offset, uint64_t nodes, uint64_t original_tasks
) {
    if (!scheduler_native_header_extent_valid(base, bytes, offset)) return nullptr;
    auto *header = reinterpret_cast<__gm__ const SchedulerNativeGraphHeader *>(base + offset);
    if (header->magic != SCHEDULER_NATIVE_GRAPH_MAGIC || header->version != SCHEDULER_NATIVE_GRAPH_VERSION ||
        header->header_bytes != sizeof(SchedulerNativeGraphHeader) || header->image_bytes != bytes ||
        header->node_count != nodes || nodes == 0 || nodes > INT32_MAX ||
        header->original_task_count != original_tasks || original_tasks == 0 ||
        original_tasks > TaskId::GLOBAL_TASK_MAX_NUM || header->instance_count == 0 ||
        header->instance_count > original_tasks || header->definition_count == 0 ||
        header->definition_count > header->instance_count || header->run_epoch == 0 ||
        header->readonly_end_offset > bytes || header->readonly_end_offset % SCHEDULER_STATE_ALIGNMENT != 0)
        return nullptr;
    const uint64_t end = header->readonly_end_offset;
    if (!scheduler_native_region_valid(
            end, header->bindings_offset, nodes, sizeof(SchedulerNodeBinding), alignof(SchedulerNodeBinding)
        ) ||
        !scheduler_native_region_valid(
            end, header->completion_refs_offset, original_tasks, sizeof(int64_t), alignof(int64_t)
        ) ||
        !scheduler_native_region_valid(
            end, header->templates_offset, header->definition_count, sizeof(SchedulerGraphTemplate),
            alignof(SchedulerGraphTemplate)
        ) ||
        !scheduler_native_region_valid(
            end, header->instances_offset, header->instance_count, sizeof(SchedulerGraphInstance),
            alignof(SchedulerGraphInstance)
        ))
        return nullptr;
    if (header->bindings_offset < offset + sizeof(SchedulerNativeGraphHeader) ||
        header->completion_refs_offset < header->bindings_offset + nodes * sizeof(SchedulerNodeBinding) ||
        header->templates_offset < header->completion_refs_offset + original_tasks * sizeof(int64_t) ||
        header->instances_offset < header->templates_offset + header->definition_count * sizeof(SchedulerGraphTemplate))
        return nullptr;
    return header;
}

inline __aicore__ bool
scheduler_native_child_arguments_valid(__gm__ const SchedulerGraphChildPlan &child, uint64_t args_bytes) {
    return child.tensor_count <= MAX_TENSOR_ARGS && child.scalar_count <= MAX_SCALAR_ARGS &&
           child.tensors_offset % 64 == 0 && child.scalars_offset % 64 == 0 && child.tensors_offset <= args_bytes &&
           child.tensor_count <= (args_bytes - child.tensors_offset) / TASK_IMAGE_TENSOR_STRIDE &&
           child.scalars_offset <= args_bytes &&
           child.scalar_count <= (args_bytes - child.scalars_offset) / sizeof(uint64_t) &&
           (child.predicate_offset == SCHEDULER_NATIVE_NO_PREDICATE ||
            (child.predicate_offset % 64 == 0 && child.predicate_offset <= args_bytes &&
             args_bytes - child.predicate_offset >= 64));
}

inline __aicore__ __gm__ const SchedulerGraphTemplate *
scheduler_native_template_checked(uint64_t base, __gm__ const SchedulerNativeGraphHeader &header, uint32_t index) {
    if (index >= header.definition_count) return nullptr;
    auto *entry = reinterpret_cast<__gm__ const SchedulerGraphTemplate *>(base + header.templates_offset) + index;
    const uint64_t end = header.readonly_end_offset;
    if (entry->task_count == 0 || entry->task_count > SUB_TASK_MAX_NUM ||
        entry->edge_count > entry->task_count * (entry->task_count - 1) / 2 || entry->sink_count == 0 ||
        entry->sink_count > entry->task_count || entry->args_bytes % 64 != 0 ||
        entry->child_plans_offset < header.instances_offset + header.instance_count * sizeof(SchedulerGraphInstance) ||
        !scheduler_native_region_valid(
            end, entry->child_plans_offset, entry->task_count, sizeof(SchedulerGraphChildPlan),
            alignof(SchedulerGraphChildPlan)
        ) ||
        !scheduler_native_region_valid(
            end, entry->fanin_offsets_offset, entry->task_count + 1, sizeof(int32_t), alignof(int32_t)
        ) ||
        !scheduler_native_region_valid(
            end, entry->sinks_offset, entry->sink_count, sizeof(uint16_t), alignof(uint16_t)
        ) ||
        entry->fanin_offsets_offset < entry->child_plans_offset + entry->task_count * sizeof(SchedulerGraphChildPlan))
        return nullptr;
    const uint64_t rows_end = entry->fanin_offsets_offset + (entry->task_count + 1) * sizeof(int32_t);
    if (entry->edge_count == 0) {
        if (entry->fanin_indices_offset != 0 || entry->sinks_offset < rows_end) return nullptr;
    } else if (!scheduler_native_region_valid(
                   end, entry->fanin_indices_offset, entry->edge_count, sizeof(uint16_t), alignof(uint16_t)
               ) ||
               entry->fanin_indices_offset < rows_end ||
               entry->sinks_offset < entry->fanin_indices_offset + entry->edge_count * sizeof(uint16_t)) {
        return nullptr;
    }
    return entry;
}

inline __aicore__ __gm__ const SchedulerGraphInstance *
scheduler_native_instance_checked(uint64_t base, __gm__ const SchedulerNativeGraphHeader &header, uint32_t index) {
    if (index >= header.instance_count) return nullptr;
    auto *instance = reinterpret_cast<__gm__ const SchedulerGraphInstance *>(base + header.instances_offset) + index;
    if (instance->definition_index >= header.definition_count ||
        instance->original_shell_id >= header.original_task_count || instance->child_count == 0 ||
        instance->child_count > SUB_TASK_MAX_NUM ||
        static_cast<uint64_t>(instance->enter_ref) + 1 != instance->child_begin ||
        static_cast<uint64_t>(instance->child_begin) + instance->child_count != instance->exit_ref ||
        instance->exit_ref >= header.node_count)
        return nullptr;
    return instance;
}

inline __aicore__ __gm__ const SchedulerNodeBinding *
scheduler_native_binding_checked(uint64_t base, __gm__ const SchedulerNativeGraphHeader &header, int64_t ref) {
    if (ref < 0 || static_cast<uint64_t>(ref) >= header.node_count) return nullptr;
    auto *binding = reinterpret_cast<__gm__ const SchedulerNodeBinding *>(base + header.bindings_offset) + ref;
    const auto kind = binding->kind;
    if (binding->instance_index == SCHEDULER_NATIVE_NO_INSTANCE) {
        if (binding->source_index >= header.original_task_count ||
            (kind != SchedulerNodeKind::TOP_KERNEL && kind != SchedulerNodeKind::CONTROL_DUMMY &&
             kind != SchedulerNodeKind::HOST_DONE))
            return nullptr;
        return binding;
    }
    const auto *instance = scheduler_native_instance_checked(base, header, binding->instance_index);
    if (instance == nullptr) return nullptr;
    if (kind == SchedulerNodeKind::GRAPH_ENTER || kind == SchedulerNodeKind::GRAPH_EXIT) {
        if (binding->source_index != instance->original_shell_id ||
            ref != (kind == SchedulerNodeKind::GRAPH_ENTER ? instance->enter_ref : instance->exit_ref))
            return nullptr;
    } else if (kind == SchedulerNodeKind::CHILD_KERNEL || kind == SchedulerNodeKind::CONTROL_DUMMY) {
        if (binding->source_index >= instance->child_count ||
            static_cast<uint64_t>(instance->child_begin) + binding->source_index != static_cast<uint64_t>(ref))
            return nullptr;
    } else {
        return nullptr;
    }
    return binding;
}
