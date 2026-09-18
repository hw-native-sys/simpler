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
#include "host_build_graph/kernel_graph_restore.h"

#include <cstring>

#include "aicpu/cache_maintenance.h"
#include "host_build_graph/kernel_external_tensor_wire.h"
#include "host_build_graph/runtime_core.h"

namespace hbg {
namespace {

template <typename T>
T read_value(const std::byte *p) noexcept {
    T value{};
    std::memcpy(&value, p, sizeof(value));
    return value;
}

bool same_layout(const RuntimeArenaLayout &a, const RuntimeArenaLayout &b) noexcept {
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

bool relative_span(
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

bool validate_definition(
    const std::byte *source, uint64_t capacity, uint64_t offset, const ChipTaskStorage &task
) noexcept {
    if (offset < sizeof(GraphDefinitionHeader) || offset % GRAPH_DEFINITION_OBJECT_ALIGN != 0 ||
        !graph_span_fits(offset, sizeof(GraphDefinition), capacity))
        return false;
    const auto framing = read_value<GraphDefinitionHeader>(source + offset - sizeof(GraphDefinitionHeader));
    const auto d = read_value<GraphDefinition>(source + offset);
    if (framing.magic != GRAPH_DEFINITION_OBJECT_MAGIC || framing.definition_bytes != d.total_bytes ||
        framing.full_key != d.full_key || !graph_span_fits(offset, d.total_bytes, capacity) ||
        d.total_bytes < sizeof(d) || d.task_count <= 0 || d.task_count > MAX_IN_GRAPH_TASKS || d.edge_count < 0 ||
        d.root_count <= 0 || d.root_count > d.task_count || d.boundary_count != task.payload.tensor_count ||
        d.boundary_scalar_count != task.payload.scalar_count)
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
    const auto begin = reinterpret_cast<uintptr_t>(task.task.packed_buffer_base);
    const auto end = reinterpret_cast<uintptr_t>(task.task.packed_buffer_end);
    return graph_execution_storage_layout(d.task_count, d.tensor_arg_count, d.scalar_arg_count, &storage) &&
           storage.total_bytes == d.execution_storage_bytes && end >= begin &&
           graph_span_fits(d.required_heap, storage.total_bytes, end - begin) &&
           (begin + d.required_heap) % alignof(ChipTaskStorage) == 0;
}

bool validate_images(const GraphRestoreView &view, RuntimeArenaLayout &layout, uint64_t &sm_bytes) noexcept {
    const auto &h = view.graph;
    DeviceArena reservations;
    uint64_t ready_capacity = 64;
    while (ready_capacity < h.task_window)
        ready_capacity <<= 1;
    layout = runtime_reserve_layout(reservations, h.task_window, ready_capacity);
    const uint64_t capacity = view.regions[1].bytes;
    if (h.runtime_offset != layout.off_runtime || h.sm_offset != layout.off_copied_end ||
        !graph_span_fits(h.runtime_offset, sizeof(RuntimeContext), capacity) ||
        !graph_span_fits(h.sm_offset, sizeof(SharedMemoryHeader), capacity))
        return false;
    const auto pristine = read_value<RuntimeContext>(view.payload + h.runtime_offset);
    if (!same_layout(pristine.prebuilt_layout, layout) || pristine.mode != MODE_EXECUTE || pristine.ops != nullptr ||
        pristine.orchestrator != nullptr || pristine.tensor_access != nullptr || pristine.sm_handle != nullptr ||
        pristine.scheduler != nullptr || pristine.aicore_mailbox != nullptr || pristine.total_cycles != 0 ||
        pristine.inline_completed_tasks < 0 || pristine.inline_completed_tasks > h.total_tasks ||
        pristine.active_callable_hash != h.callable_hash)
        return false;
    sm_bytes = capacity - h.sm_offset;
    const auto offsets = sm_layout::segment_offsets(sm_layout::image_extents({h.total_tasks, 0, 0, 0}));
    if (sm_bytes > INT32_MAX || offsets.fanin_pool > sm_bytes) return false;
    const auto *sm = view.payload + h.sm_offset;
    if (read_value<uint64_t>(sm + offsetof(SharedMemoryTaskHeader, task_storage)) != 0 ||
        read_value<uint64_t>(sm + offsetof(SharedMemoryTaskHeader, task_states)) != 0 ||
        read_value<int32_t>(sm + offsetof(SharedMemoryTaskHeader, total_tasks)) != static_cast<int32_t>(h.total_tasks))
        return false;
    const auto &heap = view.slot.destinations[0];
    const auto &definitions = view.slot.destinations[2];
    const uint64_t definition_source = view.regions[2].source_offset;
    for (uint32_t i = 0; i < h.total_tasks; ++i) {
        const uint64_t storage_offset = offsets.storage + i * sizeof(ChipTaskStorage);
        ChipTaskStorage task{};
        std::memcpy(static_cast<void *>(&task), sm + storage_offset, sizeof(task));
        if (task.task.task_id.raw != TaskId::make_global(i).raw ||
            (task.slot.task_kind != TaskKind::GRAPH && task.slot.task_kind != TaskKind::DUMMY &&
             task.slot.task_kind != TaskKind::KERNEL))
            return false;
        const bool graph = task.slot.task_kind == TaskKind::GRAPH;
        const auto &p = task.payload;
        if (p.tensor_count < 0 || p.tensor_count > (graph ? GRAPH_MAX_TENSOR_ARGS : MAX_TENSOR_ARGS) ||
            p.scalar_count < 0 || p.scalar_count > (graph ? GRAPH_MAX_SCALAR_ARGS : MAX_SCALAR_ARGS) ||
            p.fanin_count < 0 || p.fanin_count > CHIP_MAX_FANIN)
            return false;
        const uint64_t payload_offset = storage_offset + offsetof(ChipTaskStorage, payload);
        uint64_t tensors = 0, scalars = 0, fanin = 0;
        if (!relative_span(
                sm, payload_offset + offsetof(TaskPayload, tensors), p.tensor_count,
                graph ? sizeof(GraphTensor) : sizeof(simpler::hbg::Tensor),
                graph ? alignof(GraphTensor) : alignof(simpler::hbg::Tensor), offsets.fanin_pool, sm_bytes, tensors
            ) ||
            !relative_span(
                sm, payload_offset + offsetof(TaskPayload, scalars), p.scalar_count, sizeof(uint64_t),
                alignof(uint64_t), offsets.fanin_pool, sm_bytes, scalars
            ) ||
            !relative_span(
                sm, payload_offset + offsetof(TaskPayload, fanin), p.fanin_count, sizeof(int32_t), alignof(int32_t),
                offsets.fanin_pool, sm_bytes, fanin
            ))
            return false;
        for (int32_t j = 0; j < p.tensor_count; ++j) {
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
        for (int32_t j = 0; j < p.fanin_count; ++j) {
            const int32_t dep = read_value<int32_t>(sm + fanin + j * sizeof(int32_t));
            if (dep < 0 || static_cast<uint32_t>(dep) >= i) return false;
        }
        const uint64_t begin = reinterpret_cast<uintptr_t>(task.task.packed_buffer_base);
        const uint64_t end = reinterpret_cast<uintptr_t>(task.task.packed_buffer_end);
        if ((begin != 0 || end != 0) &&
            (begin < heap.address || end < begin || !graph_span_fits(begin - heap.address, end - begin, h.heap_bytes)))
            return false;
        const uint64_t definition = reinterpret_cast<uintptr_t>(task.slot.graph_context);
        if (graph) {
            if (definition < definitions.address ||
                !validate_definition(
                    view.payload + definition_source, view.regions[2].bytes, definition - definitions.address, task
                ))
                return false;
        } else if (definition != 0) return false;
    }
    return true;
}

bool copy_region(const GraphRestoreOps &ops, void *dst, const void *src, size_t bytes) noexcept {
    if (ops.copy) return ops.copy(ops.context, dst, src, bytes);
    std::memcpy(dst, src, bytes);
    return true;
}
bool zero_region(const GraphRestoreOps &ops, void *dst, size_t bytes) noexcept {
    if (ops.zero) return ops.zero(ops.context, dst, bytes);
    std::memset(dst, 0, bytes);
    return true;
}
bool flush_region(const GraphRestoreOps &ops, const void *dst, size_t bytes) noexcept {
    if (ops.flush) return ops.flush(ops.context, dst, bytes);
    cache_flush_range(dst, bytes);
    return true;
}
void publish(GraphRestoreControl &control, GraphRestorePhase phase, GraphRestoreStatus status) noexcept {
    control.status = static_cast<uint32_t>(status);
    cache_flush_range(&control, sizeof(control));
    __atomic_store_n(&control.phase, static_cast<uint32_t>(phase), __ATOMIC_RELEASE);
    cache_flush_range(&control, sizeof(control));
}
}  // namespace

GraphRestoreStatus restore_graph_packet(
    const void *packet, size_t bytes, int device_id, uint64_t runtime_binary_id,
    const simpler::kernel::PreparedInvocationView &trusted_callable, GraphRestoreResult &out, const GraphRestoreOps &ops
) noexcept {
    GraphRestoreView view{};
    const auto admission =
        admit_graph_packet_for_restore(packet, bytes, device_id, runtime_binary_id, trusted_callable, view);
    if (admission == GraphSlotStatus::Poisoned) return GraphRestoreStatus::Poisoned;
    if (admission != GraphSlotStatus::Ok) return GraphRestoreStatus::Rejected;
    RuntimeArenaLayout layout{};
    uint64_t sm_bytes = 0;
    if (!validate_images(view, layout, sm_bytes)) return GraphRestoreStatus::InvalidImage;
    auto *registry = reinterpret_cast<GraphSlotRegistry *>(view.slot.registry.address);
    auto &control = registry->restore;
    cache_invalidate_range(&control, sizeof(control));
    uint32_t phase = __atomic_load_n(&control.phase, __ATOMIC_ACQUIRE);
    if (phase == static_cast<uint32_t>(GraphRestorePhase::Restoring) ||
        phase == static_cast<uint32_t>(GraphRestorePhase::Ready))
        return GraphRestoreStatus::Busy;
    if (phase == static_cast<uint32_t>(GraphRestorePhase::Failed)) return GraphRestoreStatus::Quarantined;
    if (phase != static_cast<uint32_t>(GraphRestorePhase::Idle)) return GraphRestoreStatus::Rejected;
    if (!__atomic_compare_exchange_n(
            &control.phase, &phase, static_cast<uint32_t>(GraphRestorePhase::Restoring), false, __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE
        ))
        return GraphRestoreStatus::Busy;
    if (control.attempt == UINT64_MAX) {
        __atomic_store_n(&control.phase, phase, __ATOMIC_RELEASE);
        cache_flush_range(&control, sizeof(control));
        return GraphRestoreStatus::Exhausted;
    }
    ++control.attempt;
    cache_flush_range(&control, sizeof(control));
    const auto fail = [&]() {
        publish(control, GraphRestorePhase::Failed, GraphRestoreStatus::CopyFailed);
        return GraphRestoreStatus::CopyFailed;
    };
    const auto &heap = view.slot.destinations[0];
    if (!zero_region(ops, reinterpret_cast<void *>(heap.address), view.graph.heap_bytes)) return fail();
    for (size_t i = 1; i < 4; ++i) {
        const auto &dst = view.slot.destinations[i];
        const auto &region = view.regions[i];
        if (region.bytes == 0) continue;
        if (!copy_region(ops, reinterpret_cast<void *>(dst.address), view.payload + region.source_offset, region.bytes))
            return fail();
    }
    const auto &working = view.slot.destinations[1];
    DeviceArena arena;
    arena.attach(reinterpret_cast<void *>(working.address));
    auto *runtime = reinterpret_cast<RuntimeContext *>(working.address + layout.off_runtime);
    runtime_wire_arena_pointers(arena, layout, runtime);
    void *sm = reinterpret_cast<void *>(working.address + view.graph.sm_offset);
    if (!runtime->sm_handle->attach_populated(
            sm, sm_bytes, view.graph.task_window, sm_layout::live_slot_pitch(view.graph.total_tasks), sm_bytes
        ) ||
        !runtime->scheduler->init_data_from_layout(layout.sched, arena, sm))
        return fail();
    runtime->scheduler->seed_queue_slots();
    runtime->aicore_mailbox->init_empty();
    if (!flush_region(ops, reinterpret_cast<void *>(heap.address), view.graph.heap_bytes)) return fail();
    for (size_t i = 1; i < 4; ++i)
        if (view.regions[i].bytes &&
            !flush_region(ops, reinterpret_cast<void *>(view.slot.destinations[i].address), view.regions[i].bytes))
            return fail();
    control.runtime_address = reinterpret_cast<uintptr_t>(runtime);
    control.sm_bytes = sm_bytes;
    control.total_tasks = view.graph.total_tasks;
    control.live_bytes[0] = view.graph.heap_bytes;
    for (size_t i = 1; i < 4; ++i)
        control.live_bytes[i] = view.regions[i].bytes;
    if (!flush_region(ops, &control, sizeof(control))) return fail();
    control.committed_generation = control.attempt;
    publish(control, GraphRestorePhase::Ready, GraphRestoreStatus::Ok);
    out = {runtime, control.attempt, sm_bytes, view.graph.total_tasks};
    return GraphRestoreStatus::Ok;
}

GraphRestoreStatus
retire_graph_restore(GraphSlotRegistry *registry, uint64_t attempt, const GraphRestoreCompletion &completion) noexcept {
    if (registry == nullptr || attempt == 0 || reinterpret_cast<uintptr_t>(registry) % 1024 != 0)
        return GraphRestoreStatus::Rejected;
    GraphSlotRegistration slot{};
    const auto status = acquire_graph_execution_slot(registry, registry->device_id, registry->runtime_binary_id, slot);
    if (status == GraphSlotStatus::Poisoned) return GraphRestoreStatus::Poisoned;
    if (status != GraphSlotStatus::Ok) return GraphRestoreStatus::Rejected;
    auto &control = registry->restore;
    cache_invalidate_range(&control, sizeof(control));
    uint32_t phase = __atomic_load_n(&control.phase, __ATOMIC_ACQUIRE);
    if (phase == static_cast<uint32_t>(GraphRestorePhase::Restoring)) return GraphRestoreStatus::Busy;
    if (control.attempt != attempt || (phase != static_cast<uint32_t>(GraphRestorePhase::Ready) &&
                                       phase != static_cast<uint32_t>(GraphRestorePhase::Failed)))
        return GraphRestoreStatus::NotReady;
    const auto outcome = completion.outcome;
    if (completion.runtime_status != 0 || completion.unexpected_teardown_status != 0 ||
        outcome == GraphRestoreRetirement::FatalFailure) {
        return poison_graph_execution_slot(registry) == GraphSlotStatus::Ok ? GraphRestoreStatus::Poisoned :
                                                                              GraphRestoreStatus::Rejected;
    }
    const bool completed =
        outcome == GraphRestoreRetirement::Completed && phase == static_cast<uint32_t>(GraphRestorePhase::Ready) &&
        control.committed_generation == attempt && control.status == static_cast<uint32_t>(GraphRestoreStatus::Ok);
    const bool controlled = outcome == GraphRestoreRetirement::ControlledFailure &&
                            phase == static_cast<uint32_t>(GraphRestorePhase::Failed);
    if (!completed && !controlled) return GraphRestoreStatus::Rejected;
    if (!__atomic_compare_exchange_n(
            &control.phase, &phase, static_cast<uint32_t>(GraphRestorePhase::Idle), false, __ATOMIC_ACQ_REL,
            __ATOMIC_ACQUIRE
        ))
        return GraphRestoreStatus::Busy;
    cache_flush_range(&control, sizeof(control));
    return GraphRestoreStatus::Ok;
}

GraphRestoreStatus
acquire_graph_restore_result(const GraphSlotRegistry *registry, uint64_t generation, GraphRestoreResult &out) noexcept {
    if (registry == nullptr || generation == 0 || reinterpret_cast<uintptr_t>(registry) % 1024 != 0)
        return GraphRestoreStatus::NotReady;
    GraphSlotRegistration slot{};
    const auto status = acquire_graph_execution_slot(registry, registry->device_id, registry->runtime_binary_id, slot);
    if (status == GraphSlotStatus::Poisoned) return GraphRestoreStatus::Poisoned;
    if (status != GraphSlotStatus::Ok) return GraphRestoreStatus::NotReady;
    const auto &control = registry->restore;
    cache_invalidate_range(&control, sizeof(control));
    if (__atomic_load_n(&control.phase, __ATOMIC_ACQUIRE) != static_cast<uint32_t>(GraphRestorePhase::Ready) ||
        control.attempt != generation || control.committed_generation != generation ||
        control.status != static_cast<uint32_t>(GraphRestoreStatus::Ok) ||
        control.runtime_address < slot.destinations[1].address ||
        !graph_span_fits(
            control.runtime_address - slot.destinations[1].address, sizeof(RuntimeContext),
            slot.destinations[1].capacity
        ))
        return GraphRestoreStatus::NotReady;
    for (size_t i = 0; i < 4; ++i)
        if (control.live_bytes[i] > slot.destinations[i].capacity) return GraphRestoreStatus::NotReady;
    for (size_t i = 0; i < 4; ++i)
        if (control.live_bytes[i])
            cache_invalidate_range(reinterpret_cast<const void *>(slot.destinations[i].address), control.live_bytes[i]);
    out = {
        reinterpret_cast<RuntimeContext *>(control.runtime_address), generation, control.sm_bytes, control.total_tasks
    };
    return GraphRestoreStatus::Ok;
}
}  // namespace hbg
