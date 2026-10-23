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

#include <cstring>

#include "host_build_graph/kernel_external_tensor_wire.h"
#include "host_build_graph/kernel_graph_slot_registry.h"
#include "host_build_graph/runtime_core.h"

namespace hbg::detail {

template <typename T>
inline T read_value(const std::byte *p) noexcept {
    T value{};
    std::memcpy(&value, p, sizeof(value));
    return value;
}

inline bool same_layout(const RuntimeArenaLayout &a, const RuntimeArenaLayout &b) noexcept {
    if (a.off_sm_handle != b.off_sm_handle || a.off_scheduler != b.off_scheduler || a.off_runtime != b.off_runtime ||
        a.off_mailbox != b.off_mailbox || a.off_copied_begin != b.off_copied_begin ||
        a.off_copied_end != b.off_copied_end || a.task_capacity != b.task_capacity || a.arena_size != b.arena_size)
        return false;
    const auto &x = a.sched;
    const auto &y = b.sched;
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i)
        if (x.off_ready_queue_slots[i] != y.off_ready_queue_slots[i] ||
            x.off_ready_sync_queue_slots[i] != y.off_ready_sync_queue_slots[i] ||
            x.off_early_dispatch_queue_slots[i] != y.off_early_dispatch_queue_slots[i] ||
            x.capacities.ready[i] != y.capacities.ready[i] || x.capacities.ready_sync[i] != y.capacities.ready_sync[i])
            return false;
    return x.off_dummy_ready_queue_slots == y.off_dummy_ready_queue_slots &&
           x.off_graph_ready_queue_slots == y.off_graph_ready_queue_slots &&
           x.off_graph_prepare_queue_slots == y.off_graph_prepare_queue_slots &&
           x.off_early_sync_start_queue_slots == y.off_early_sync_start_queue_slots &&
           x.off_ed_publish_drain_queue_slots == y.off_ed_publish_drain_queue_slots &&
           x.capacities.dummy == y.capacities.dummy && x.capacities.graph_ready == y.capacities.graph_ready &&
           x.capacities.graph_prepare == y.capacities.graph_prepare;
}

inline bool relative_span(
    const std::byte *sm, uint64_t field, int32_t count, uint64_t element_bytes, uint64_t alignment, uint64_t pool_begin,
    uint64_t bytes, uint64_t &offset
) noexcept {
    const int32_t delta = read_value<int32_t>(sm + field);
    if (delta == 0) {
        offset = 0;
        return count == 0;
    }
    if (count < 0 || delta <= 0) return false;
    offset = field + static_cast<uint32_t>(delta);
    return offset >= pool_begin && offset % alignment == 0 &&
           graph_span_fits(offset, static_cast<uint64_t>(count) * element_bytes, bytes);
}

inline bool validate_definition(
    const std::byte *source, uint64_t capacity, uint64_t offset, int32_t boundary_count, int32_t boundary_scalar_count,
    uint64_t begin, uint64_t end
) noexcept {
    if (offset < sizeof(GraphDefinitionHeader) || offset % GRAPH_DEFINITION_OBJECT_ALIGN != 0 ||
        !graph_span_fits(offset, sizeof(GraphDefinition), capacity))
        return false;
    const auto framing = read_value<GraphDefinitionHeader>(source + offset - sizeof(GraphDefinitionHeader));
    const auto d = read_value<GraphDefinition>(source + offset);
    if (framing.magic != GRAPH_DEFINITION_OBJECT_MAGIC || framing.definition_bytes != d.total_bytes ||
        framing.full_key != d.full_key || !graph_span_fits(offset, d.total_bytes, capacity) ||
        d.total_bytes < sizeof(d) || d.task_count <= 0 || d.task_count > MAX_IN_GRAPH_TASKS || d.edge_count < 0 ||
        d.root_count <= 0 || d.root_count > d.task_count || d.boundary_count != boundary_count ||
        d.boundary_scalar_count != boundary_scalar_count)
        return false;
    const auto section = [&](uint64_t off, int32_t count, uint64_t stride, uint64_t align) {
        return count >= 0 &&
               (count == 0 || (off >= sizeof(d) && off % align == 0 &&
                               graph_span_fits(off, static_cast<uint64_t>(count) * stride, d.total_bytes)));
    };
    if (!section(d.off_fanin_offsets, d.task_count + 1, sizeof(int32_t), alignof(int32_t)) ||
        !section(d.off_fanout_offsets, d.task_count + 1, sizeof(int32_t), alignof(int32_t)) ||
        !section(d.off_fanin_indices, d.edge_count, sizeof(uint16_t), alignof(uint16_t)) ||
        !section(d.off_fanout_indices, d.edge_count, sizeof(uint16_t), alignof(uint16_t)) ||
        !section(d.off_root_indices, d.root_count, sizeof(uint16_t), alignof(uint16_t)) ||
        !section(d.off_in_graph_tasks, d.task_count, sizeof(InGraphTaskDefinition), alignof(InGraphTaskDefinition)) ||
        !section(d.off_in_graph_task_offsets, d.task_count, sizeof(uint64_t), alignof(uint64_t)) ||
        !section(d.off_tensors, d.tensor_arg_count, sizeof(GraphTensor), alignof(GraphTensor)) ||
        !section(
            d.off_tensor_sources, d.tensor_arg_count, sizeof(GraphTensorSourceRef), alignof(GraphTensorSourceRef)
        ) ||
        !section(d.off_scalars, d.scalar_arg_count, sizeof(uint64_t), alignof(uint64_t)) ||
        !section(
            d.off_scalar_inheritance, d.scalar_arg_count, sizeof(GraphScalarInheritance),
            alignof(GraphScalarInheritance)
        ) ||
        !section(
            d.off_boundary_signatures, d.boundary_count, sizeof(GraphBoundarySignature), alignof(GraphBoundarySignature)
        ) ||
        !section(d.off_predicates, d.predicate_count, sizeof(GraphPredicate), alignof(GraphPredicate)))
        return false;
    for (int32_t i = 0; i < d.tensor_arg_count; ++i) {
        const auto tensor = read_value<GraphTensor>(source + offset + d.off_tensors + i * sizeof(GraphTensor));
        if (!valid_kernel_graph_tensor(tensor)) return false;
    }
    for (int32_t i = 0; i < d.predicate_count; ++i) {
        const auto predicate =
            read_value<GraphPredicate>(source + offset + d.off_predicates + i * sizeof(GraphPredicate));
        if (!valid_kernel_graph_tensor(predicate.operand)) return false;
    }
    GraphExecutionStorageLayout storage{};
    return graph_execution_storage_layout(d.task_count, d.tensor_arg_count, d.scalar_arg_count, &storage) &&
           storage.total_bytes == d.execution_storage_bytes && end >= begin &&
           graph_span_fits(d.required_heap, storage.total_bytes, end - begin) &&
           (begin + d.required_heap) % alignof(ChipTaskStorage) == 0;
}

// Packet framing and resource binding are validated before this image-only read.
inline bool
validate_graph_images(const GraphRestoreView &view, RuntimeArenaLayout &layout, uint64_t &sm_bytes) noexcept {
    const auto &h = view.graph;
    DeviceArena reservations;
    uint64_t ready_capacity = 64;
    while (ready_capacity < h.task_window)
        ready_capacity <<= 1;
    layout = runtime_reserve_layout(reservations, h.task_window, ready_capacity);
    layout.sched.capacities = kernel_graph_queue_capacities(h.total_tasks, h.task_window, view.regions[2].bytes != 0);
    const auto &image = view.regions[1];
    if (h.runtime_offset != layout.off_runtime || h.sm_offset != layout.off_copied_end ||
        image.destination_offset != layout.off_runtime || h.sm_offset < image.destination_offset ||
        !graph_span_fits(0, sizeof(RuntimeContext), image.bytes) ||
        !graph_span_fits(h.sm_offset - image.destination_offset, sizeof(SharedMemoryHeader), image.bytes))
        return false;
    const auto *runtime_source = view.payload + image.source_offset;
    const auto pristine = read_value<RuntimeContext>(runtime_source);
    if (!same_layout(pristine.prebuilt_layout, layout) || pristine.mode != MODE_EXECUTE || pristine.ops != nullptr ||
        pristine.orchestrator != nullptr || pristine.tensor_access != nullptr || pristine.sm_handle != nullptr ||
        pristine.scheduler != nullptr || pristine.aicore_mailbox != nullptr || pristine.total_cycles != 0 ||
        pristine.inline_completed_tasks < 0 || pristine.inline_completed_tasks > h.total_tasks ||
        pristine.active_callable_hash != h.callable_hash)
        return false;
    const uint64_t sm_offset = h.sm_offset - image.destination_offset;
    sm_bytes = image.bytes - sm_offset;
    const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents({h.total_tasks, 0, 0, 0}));
    if (sm_bytes > INT32_MAX || offsets.fanin_pool > sm_bytes) return false;
    const auto *sm = runtime_source + sm_offset;
    if (read_value<uint64_t>(sm + offsetof(SharedMemoryTaskHeader, task_storage)) != 0 ||
        read_value<uint64_t>(sm + offsetof(SharedMemoryTaskHeader, task_states)) != 0 ||
        read_value<int32_t>(sm + offsetof(SharedMemoryTaskHeader, total_tasks)) != static_cast<int32_t>(h.total_tasks))
        return false;
    const auto &heap = view.slot.destinations[0];
    const auto &definitions = view.slot.destinations[2];
    const uint64_t definition_source = view.regions[2].source_offset;
    for (uint32_t i = 0; i < h.total_tasks; ++i) {
        const uint64_t storage_offset = offsets.storage + i * sizeof(ChipTaskStorage);
        const auto *descriptor = sm + storage_offset + offsetof(ChipTaskStorage, task);
        const auto *slot = sm + storage_offset + offsetof(ChipTaskStorage, slot);
        const auto task_id = read_value<TaskId>(descriptor + offsetof(TaskDescriptor, task_id));
        const auto task_kind = read_value<TaskKind>(slot + offsetof(ChipTaskSlotState, task_kind));
        if (task_id.raw != TaskId::make_global(i).raw ||
            (task_kind != TaskKind::GRAPH && task_kind != TaskKind::DUMMY && task_kind != TaskKind::KERNEL))
            return false;
        const bool graph = task_kind == TaskKind::GRAPH;
        const uint64_t payload_offset = storage_offset + offsetof(ChipTaskStorage, payload);
        const auto *payload = sm + payload_offset;
        const auto tensor_count = read_value<int32_t>(payload + offsetof(TaskPayload, tensor_count));
        const auto scalar_count = read_value<int32_t>(payload + offsetof(TaskPayload, scalar_count));
        const auto fanin_count = read_value<int32_t>(payload + offsetof(TaskPayload, fanin_count));
        if (tensor_count < 0 || tensor_count > (graph ? GRAPH_MAX_TENSOR_ARGS : MAX_TENSOR_ARGS) || scalar_count < 0 ||
            scalar_count > (graph ? GRAPH_MAX_SCALAR_ARGS : MAX_SCALAR_ARGS) || fanin_count < 0 ||
            fanin_count > CHIP_MAX_FANIN)
            return false;
        uint64_t tensors = 0, scalars = 0, fanin = 0;
        if (!relative_span(
                sm, payload_offset + offsetof(TaskPayload, tensors), tensor_count,
                graph ? sizeof(GraphTensor) : sizeof(simpler::hbg::Tensor),
                graph ? alignof(GraphTensor) : alignof(simpler::hbg::Tensor), offsets.fanin_pool, sm_bytes, tensors
            ) ||
            !relative_span(
                sm, payload_offset + offsetof(TaskPayload, scalars), scalar_count, sizeof(uint64_t), alignof(uint64_t),
                offsets.fanin_pool, sm_bytes, scalars
            ) ||
            !relative_span(
                sm, payload_offset + offsetof(TaskPayload, fanin), fanin_count, sizeof(int32_t), alignof(int32_t),
                offsets.fanin_pool, sm_bytes, fanin
            ))
            return false;
        for (int32_t j = 0; j < tensor_count; ++j) {
            if (graph) {
                if (!valid_restored_kernel_graph_tensor(
                        read_value<GraphTensor>(sm + tensors + j * sizeof(GraphTensor)), heap.address, h.heap_bytes
                    ))
                    return false;
            } else if (!valid_restored_kernel_graph_tensor(
                           read_value<simpler::hbg::Tensor>(sm + tensors + j * sizeof(simpler::hbg::Tensor)),
                           heap.address, h.heap_bytes
                       )) {
                return false;
            }
        }
        for (int32_t j = 0; j < fanin_count; ++j) {
            const int32_t dep = read_value<int32_t>(sm + fanin + j * sizeof(int32_t));
            if (dep < 0 || static_cast<uint32_t>(dep) >= i) return false;
        }
        const uint64_t begin =
            reinterpret_cast<uintptr_t>(read_value<void *>(descriptor + offsetof(TaskDescriptor, packed_buffer_base)));
        const uint64_t end =
            reinterpret_cast<uintptr_t>(read_value<void *>(descriptor + offsetof(TaskDescriptor, packed_buffer_end)));
        if ((begin != 0 || end != 0) &&
            (begin < heap.address || end < begin || !graph_span_fits(begin - heap.address, end - begin, h.heap_bytes)))
            return false;
        const uint64_t definition =
            reinterpret_cast<uintptr_t>(read_value<void *>(slot + offsetof(ChipTaskSlotState, graph_context)));
        if (graph) {
            if (definition < definitions.address ||
                !validate_definition(
                    view.payload + definition_source, view.regions[2].bytes, definition - definitions.address,
                    tensor_count, scalar_count, begin, end
                ))
                return false;
        } else if (definition != 0) return false;
    }
    return true;
}

}  // namespace hbg::detail
