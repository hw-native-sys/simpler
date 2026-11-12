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

#include "scheduler_graph.h"
#include "scheduler_memory.h"

enum class SchedulerGraphBindingError : uint32_t {
    NONE = 0,
    NODE,
    DEFINITION,
    INSTANCE,
    TENSOR,
    SCALAR,
    PREDICATE,
};

inline __aicore__ SchedulerErrorSite scheduler_graph_binding_error_site(SchedulerGraphBindingError error) {
    return static_cast<SchedulerErrorSite>(
        static_cast<uint64_t>(SchedulerErrorSite::GRAPH_BIND_NODE) + static_cast<uint32_t>(error) - 1
    );
}

template <typename T>
inline __aicore__ __gm__ const T *
scheduler_definition_element(uint64_t base, uint64_t bytes, uint32_t offset, int32_t count, int32_t index) {
    if (count <= 0 || index < 0 || index >= count || offset < sizeof(GraphDefinition) ||
        !scheduler_native_region_valid(bytes, offset, static_cast<uint64_t>(count), sizeof(T), alignof(T)))
        return nullptr;
    return reinterpret_cast<__gm__ const T *>(base + offset) + index;
}

struct SchedulerTensorBinding {
    uint64_t address;
    uint64_t bytes;
    uint64_t origin;
    AddressSpace address_space;
};

inline __aicore__ uint32_t scheduler_graph_element_size(DataType dtype) {
    if (dtype == DataType::INT64 || dtype == DataType::UINT64) return 8;
    if (dtype == DataType::FLOAT32 || dtype == DataType::INT32 || dtype == DataType::UINT32) return 4;
    if (dtype == DataType::FLOAT16 || dtype == DataType::BFLOAT16 || dtype == DataType::INT16 ||
        dtype == DataType::UINT16)
        return 2;
    if (dtype == DataType::INT8 || dtype == DataType::UINT8 || dtype == DataType::BOOL ||
        dtype == DataType::FP8E4M3FN || dtype == DataType::FP8E8M0 || dtype == DataType::FP4E2M1 ||
        dtype == DataType::FP8E5M2)
        return 1;
    return 0;
}

inline __aicore__ bool scheduler_graph_relocate_tensor(
    const __gm__ simpler::hbg::TensorData &record, const SchedulerNativeNode &node, SchedulerTensorBinding *out
) {
    const auto &instance = *node.instance;
    const uint64_t owner = TaskId::to_uint64(record.owner_task_id);
    const uint64_t space = owner >> TaskId::SPACE_SHIFT;
    const uint32_t index = static_cast<uint32_t>(owner);
    if (record.ndims == 0 || record.ndims > MAX_TENSOR_DIMS ||
        (record.address_space != AddressSpace::HOST && record.address_space != AddressSpace::DEVICE))
        return false;
    out->address = record.buffer.addr;
    out->bytes = record.buffer.size;
    out->origin = record.start_offset;
    out->address_space = record.address_space;
    if (space == static_cast<uint64_t>(TaskId::Space::PARAM)) {
        if (index >= instance.boundary_tensor_count ||
            owner != ((static_cast<uint64_t>(TaskId::Space::PARAM) << TaskId::SPACE_SHIFT) | index) ||
            record.buffer.addr != 0 || instance.boundary_tensors_address == 0 ||
            instance.boundary_tensors_address % 64 != 0 ||
            instance.boundary_tensor_count >
                (UINT64_MAX - instance.boundary_tensors_address) / TASKPAYLOAD_TENSOR_STRIDE)
            return false;
        auto *boundary = reinterpret_cast<__gm__ const simpler::hbg::TensorData *>(
            instance.boundary_tensors_address + static_cast<uint64_t>(index) * TASKPAYLOAD_TENSOR_STRIDE
        );
        if (out->origin > UINT64_MAX - boundary->start_offset ||
            (boundary->address_space != AddressSpace::HOST && boundary->address_space != AddressSpace::DEVICE))
            return false;
        out->address = boundary->buffer.addr;
        out->bytes = boundary->buffer.size;
        out->origin += boundary->start_offset;
        out->address_space = boundary->address_space;
        const uint64_t element_bytes = scheduler_graph_element_size(record.dtype);
        if (element_bytes == 0) return false;
        const uint64_t capacity = out->bytes / element_bytes;
        if (out->origin > capacity || record.extent_elem_cache > capacity - out->origin) return false;
    } else if (space == static_cast<uint64_t>(TaskId::Space::SUB_TASK)) {
        if (index > node.binding->source_index || record.buffer.addr > node.definition->required_heap ||
            record.buffer.size > node.definition->required_heap - record.buffer.addr ||
            record.buffer.addr > UINT64_MAX - instance.heap_base)
            return false;
        out->address += instance.heap_base;
    } else {
        return false;
    }
    return out->bytes <= UINT64_MAX - out->address;
}

inline __aicore__ void scheduler_graph_store_tensor(
    __gm__ simpler::hbg::TensorData *dst, const __gm__ simpler::hbg::TensorData &src,
    const SchedulerTensorBinding &bound
) {
    dst->buffer.addr = bound.address;
    dst->buffer.size = bound.bytes;
    TaskId::assign(dst->owner_task_id, src.owner_task_id);
    dst->start_offset = bound.origin;
    dst->version = src.version;
    dst->ndims = src.ndims;
    dst->dtype = src.dtype;
    dst->manual_dep = src.manual_dep;
    dst->is_contiguous = src.is_contiguous;
    dst->address_space = bound.address_space;
    for (uint32_t i = 0; i < MAX_TENSOR_DIMS; ++i) {
        dst->shapes[i] = src.shapes[i];
        dst->strides[i] = src.strides[i];
    }
    dst->extent_elem_cache = src.extent_elem_cache;
}

static inline __aicore__ void scheduler_graph_argument_cache_lines(uint64_t address, uint64_t bytes, bool publish) {
    for (uint64_t offset = 0; offset < bytes; offset += 64) {
        auto *line = reinterpret_cast<__gm__ void *>(address + offset);
        if (publish) scheduler_writeback_cache_line(line);
        else scheduler_invalidate_cache_line(line);
    }
}

static inline __aicore__ void scheduler_graph_arguments_cache(
    const __gm__ SchedulerGraphInstance &instance, const __gm__ SchedulerGraphChildPlan &child, bool publish
) {
    scheduler_graph_argument_cache_lines(
        instance.args_base + child.tensors_offset,
        static_cast<uint64_t>(child.tensor_count) * TASKPAYLOAD_TENSOR_STRIDE, publish
    );
    scheduler_graph_argument_cache_lines(
        instance.args_base + child.scalars_offset, static_cast<uint64_t>(child.scalar_count) * sizeof(uint64_t), publish
    );
    if (child.predicate_offset != SCHEDULER_NATIVE_NO_PREDICATE)
        scheduler_graph_argument_cache_lines(instance.args_base + child.predicate_offset, 64, publish);
    scheduler_cache_barrier();
}

// Only the child's bootstrap partition owner writes its arguments. Publication
// completes before scan arrival; ordinary owners and admission members observe
// the immutable argument image after the bootstrap scan gate opens.
static inline __aicore__ SchedulerGraphBindingError
scheduler_bind_graph_arguments(const SchedulerGraphAccess &graph, int64_t ref) {
    if (graph.native_header_offset == 0) return SchedulerGraphBindingError::NONE;
    SchedulerNativeNode node{};
    if (!scheduler_native_node_checked(graph, ref, &node)) return SchedulerGraphBindingError::NODE;
    if (node.binding->kind == SchedulerNodeKind::TOP_KERNEL) return SchedulerGraphBindingError::NONE;
    if (node.binding->kind != SchedulerNodeKind::CHILD_KERNEL) return SchedulerGraphBindingError::NODE;
    auto *child = scheduler_native_child_plan_checked(graph, node);
    if (child == nullptr) return SchedulerGraphBindingError::DEFINITION;
    const auto &instance = *node.instance;
    const uint64_t definition_base = graph.scheduler_state_address + node.definition->definition_offset;
    auto *definition = reinterpret_cast<__gm__ const GraphDefinition *>(definition_base);
    if (instance.boundary_tensor_count != static_cast<uint32_t>(definition->boundary_tensor_count) ||
        instance.boundary_scalar_count != static_cast<uint32_t>(definition->boundary_scalar_count) ||
        node.definition->required_heap > UINT64_MAX - instance.heap_base ||
        instance.args_base != instance.heap_base + node.definition->required_heap ||
        (node.definition->args_bytes != 0 && (instance.args_base == 0 || instance.args_base % 64 != 0)) ||
        node.definition->args_bytes > UINT64_MAX - instance.args_base)
        return SchedulerGraphBindingError::INSTANCE;
    auto *task = scheduler_definition_element<SubTaskDefinition>(
        definition_base, node.definition->definition_bytes, definition->off_sub_tasks, definition->task_count,
        node.binding->source_index
    );
    if (task == nullptr || task->tensor_count != static_cast<int32_t>(child->tensor_count) ||
        task->scalar_count != static_cast<int32_t>(child->scalar_count) || task->tensor_offset < 0 ||
        task->tensor_offset > definition->tensor_arg_count ||
        task->tensor_count > definition->tensor_arg_count - task->tensor_offset || task->scalar_offset < 0 ||
        task->scalar_offset > definition->scalar_arg_count ||
        task->scalar_count > definition->scalar_arg_count - task->scalar_offset ||
        (task->predicate_slot == 0) != (child->predicate_offset == SCHEDULER_NATIVE_NO_PREDICATE))
        return SchedulerGraphBindingError::DEFINITION;
    for (uint32_t i = 0; i < child->tensor_count; ++i) {
        auto *record = scheduler_definition_element<simpler::hbg::TensorData>(
            definition_base, node.definition->definition_bytes, definition->off_tensors, definition->tensor_arg_count,
            task->tensor_offset + i
        );
        SchedulerTensorBinding bound{};
        if (record == nullptr || !scheduler_graph_relocate_tensor(*record, node, &bound))
            return SchedulerGraphBindingError::TENSOR;
        auto *dst = reinterpret_cast<__gm__ simpler::hbg::TensorData *>(
            instance.args_base + child->tensors_offset + static_cast<uint64_t>(i) * TASKPAYLOAD_TENSOR_STRIDE
        );
        scheduler_graph_store_tensor(dst, *record, bound);
    }
    for (uint32_t i = 0; i < child->scalar_count; ++i) {
        auto *inheritance = scheduler_definition_element<GraphScalarInheritance>(
            definition_base, node.definition->definition_bytes, definition->off_scalar_inheritance,
            definition->scalar_arg_count, task->scalar_offset + i
        );
        auto *value = scheduler_definition_element<uint64_t>(
            definition_base, node.definition->definition_bytes, definition->off_scalars, definition->scalar_arg_count,
            task->scalar_offset + i
        );
        if (inheritance == nullptr || value == nullptr) return SchedulerGraphBindingError::SCALAR;
        GraphScalarInheritance local{};
        auto *src = reinterpret_cast<__gm__ const uint8_t *>(inheritance);
        auto *dst = reinterpret_cast<uint8_t *>(&local);
        for (uint32_t j = 0; j < sizeof(local); ++j)
            dst[j] = src[j];
        if (!local.valid()) return SchedulerGraphBindingError::SCALAR;
        if (local.inherited()) {
            if (local.boundary_index() >= instance.boundary_scalar_count || instance.boundary_scalars_address == 0 ||
                instance.boundary_scalars_address % alignof(uint64_t) != 0 ||
                instance.boundary_scalar_count > (UINT64_MAX - instance.boundary_scalars_address) / sizeof(uint64_t))
                return SchedulerGraphBindingError::SCALAR;
            value =
                reinterpret_cast<__gm__ const uint64_t *>(instance.boundary_scalars_address) + local.boundary_index();
        }
        reinterpret_cast<__gm__ uint64_t *>(instance.args_base + child->scalars_offset)[i] = *value;
    }
    if (task->predicate_slot != 0) {
        auto *predicate = scheduler_definition_element<GraphPredicate>(
            definition_base, node.definition->definition_bytes, definition->off_predicates, definition->predicate_count,
            task->predicate_slot - 1
        );
        SchedulerTensorBinding bound{};
        if (predicate == nullptr || !scheduler_graph_relocate_tensor(predicate->operand, node, &bound))
            return SchedulerGraphBindingError::PREDICATE;
        const uint64_t width = scheduler_graph_element_size(predicate->operand.dtype);
        const uint64_t owner = TaskId::to_uint64(predicate->operand.owner_task_id);
        if (width == 0 || width != predicate->elem_size || predicate->op < 1 || predicate->op > 6 ||
            ((owner >> TaskId::SPACE_SHIFT) == static_cast<uint64_t>(TaskId::Space::SUB_TASK) &&
             static_cast<uint32_t>(owner) == node.binding->source_index) ||
            predicate->elem_offset >= predicate->operand.extent_elem_cache ||
            predicate->elem_offset > UINT64_MAX - bound.origin)
            return SchedulerGraphBindingError::PREDICATE;
        const uint64_t element = bound.origin + predicate->elem_offset;
        if (bound.bytes < width || element > (bound.bytes - width) / width)
            return SchedulerGraphBindingError::PREDICATE;
        auto *dst = reinterpret_cast<__gm__ SchedulerDispatchPredicate *>(instance.args_base + child->predicate_offset);
        dst->addr = bound.address + element * width;
        dst->target = predicate->target;
        dst->elem_size = predicate->elem_size;
        dst->op = predicate->op;
    }
    scheduler_graph_arguments_cache(instance, *child, true);
    return SchedulerGraphBindingError::NONE;
}

static inline __aicore__ SchedulerGraphBindingError
scheduler_observe_graph_arguments(const SchedulerGraphAccess &graph, int64_t ref) {
    if (graph.native_header_offset == 0) return SchedulerGraphBindingError::NONE;
    SchedulerNativeNode node{};
    if (!scheduler_native_node_checked(graph, ref, &node)) return SchedulerGraphBindingError::NODE;
    if (node.binding->kind == SchedulerNodeKind::TOP_KERNEL) return SchedulerGraphBindingError::NONE;
    auto *child = scheduler_native_child_plan_checked(graph, node);
    if (node.binding->kind != SchedulerNodeKind::CHILD_KERNEL || child == nullptr ||
        (node.definition->args_bytes != 0 && (node.instance->args_base == 0 || node.instance->args_base % 64 != 0)) ||
        node.definition->args_bytes > UINT64_MAX - node.instance->args_base)
        return SchedulerGraphBindingError::INSTANCE;
    scheduler_graph_arguments_cache(*node.instance, *child, false);
    return SchedulerGraphBindingError::NONE;
}
