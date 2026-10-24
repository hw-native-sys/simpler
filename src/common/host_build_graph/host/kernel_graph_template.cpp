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
#include "host_build_graph/kernel_graph_template.h"
#include "host_build_graph/kernel_graph_packet_parser.h"

#include <algorithm>
#include <new>
#include <unordered_map>

#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/kernel_graph_tensor_validation.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/kernel_resource_requirements.h"
#include "task_interface/kernel_invocation_validation.h"

namespace hbg {
namespace {

class AlignedImage {
public:
    explicit AlignedImage(size_t bytes) :
        storage_(bytes + CHIP_ALIGN_SIZE, std::byte{0}) {}
    std::byte *data() {
        return reinterpret_cast<std::byte *>(
            (reinterpret_cast<uintptr_t>(storage_.data()) + CHIP_ALIGN_SIZE - 1) & ~(uintptr_t{CHIP_ALIGN_SIZE} - 1)
        );
    }

private:
    std::vector<std::byte> storage_;
};

bool add_size(uint64_t bytes, uint64_t &cursor) {
    if (bytes > UINT32_MAX || cursor > UINT32_MAX - bytes) return false;
    cursor += bytes;
    return true;
}

bool valid_definition_external_tensors(const GraphDefinition &definition) {
    if (definition.tensor_arg_count < 0 || definition.predicate_count < 0) return false;
    if (definition.tensor_arg_count != 0) {
        const auto *tensors = graph_definition_array<simpler::hbg::TensorData>(
            definition, definition.off_tensors, definition.tensor_arg_count
        );
        if (tensors == nullptr) return false;
        for (int32_t i = 0; i < definition.tensor_arg_count; ++i)
            if (!is_valid_graph_tensor_descriptor(tensors[i])) return false;
    }
    if (definition.predicate_count != 0) {
        const auto *predicates =
            graph_definition_array<GraphPredicate>(definition, definition.off_predicates, definition.predicate_count);
        if (predicates == nullptr) return false;
        for (int32_t i = 0; i < definition.predicate_count; ++i)
            if (!is_valid_graph_tensor_descriptor(predicates[i].operand)) return false;
    }
    return true;
}

bool valid_build_external_tensors(const GraphBuild &build) {
    const auto offsets = sm_layout::segment_offsets(build.task_capacity);
    const auto *base = static_cast<const std::byte *>(build.sm_mirror);
    const auto *storage = reinterpret_cast<const ChipTaskStorage *>(base + offsets.storage);
    const uintptr_t tensor_pool_begin = reinterpret_cast<uintptr_t>(base + offsets.tensor_pool);
    const uintptr_t tensor_pool_end = reinterpret_cast<uintptr_t>(base + offsets.scalar_pool);
    for (int32_t i = 0; i < build.total_tasks; ++i) {
        const auto &task = storage[i];
        const bool graph = task.slot.task_kind == TaskKind::GRAPH;
        const int32_t count = task.payload.tensor_count;
        if (count < 0 || count > (graph ? GRAPH_MAX_TENSOR_ARGS : MAX_TENSOR_ARGS)) return false;
        if (count == 0) continue;
        const uintptr_t first = reinterpret_cast<uintptr_t>(task.payload.tensor_data());
        const uint64_t element_bytes = sizeof(simpler::hbg::Tensor);
        const uint64_t bytes = static_cast<uint64_t>(count) * element_bytes;
        if (first < tensor_pool_begin || first > tensor_pool_end || bytes > tensor_pool_end - first) return false;
        for (int32_t j = 0; j < count; ++j) {
            const auto *element = reinterpret_cast<const std::byte *>(first + static_cast<uint64_t>(j) * element_bytes);
            if (graph) {
                if (!is_valid_graph_tensor_descriptor(*reinterpret_cast<const simpler::hbg::TensorData *>(element)))
                    return false;
            } else if (!is_valid_graph_tensor_descriptor(*reinterpret_cast<const simpler::hbg::Tensor *>(element))) {
                return false;
            }
        }
    }
    for (const auto &[key, object] : build.definitions.objects) {
        if (object.definition == nullptr || object.definition->graph_key != key ||
            object.definition->total_bytes != object.image_bytes ||
            !valid_definition_external_tensors(*object.definition))
            return false;
    }
    return true;
}

int copy_definitions(const GraphBuild &build, std::byte *out, uint64_t capacity) {
    if (capacity != build.definitions.bytes || (capacity != 0 && out == nullptr))
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    for (const auto &[key, object] : build.definitions.objects) {
        const uint64_t bytes = sizeof(GraphDefinitionHeader) + object.image_bytes;
        if (!graph_span_fits(object.object_offset, bytes, capacity)) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        GraphDefinitionHeader framing{};
        framing.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
        framing.graph_key = key;
        framing.definition_bytes = static_cast<uint32_t>(object.image_bytes);
        std::memcpy(out + object.object_offset, &framing, sizeof(framing));
        std::memcpy(out + object.object_offset + sizeof(framing), object.definition, object.image_bytes);
    }
    return 0;
}

int bind_definition_images(const GraphBuild &build, std::byte *sm) {
    const auto source_offsets = sm_layout::segment_offsets(build.task_capacity);
    const auto target_offsets = sm_layout::segment_offsets(sm_layout::image_extents(build.bind_usage));
    const auto *source = reinterpret_cast<const ChipTaskStorage *>(
        static_cast<const std::byte *>(build.sm_mirror) + source_offsets.storage
    );
    auto *target = reinterpret_cast<ChipTaskStorage *>(sm + target_offsets.storage);
    // The Host mirror is retained across binds, so a slot can still carry the
    // previous bind's Graph identity. The canonical image starts every task
    // without one and then applies only this build's bindings.
    for (int32_t i = 0; i < build.total_tasks; ++i) {
        target[i].slot.graph_context = nullptr;
        target[i].task.graph_definition_offset = 0;
    }
    for (const auto &[slot, definition_offset] : build.definitions.bindings) {
        const uintptr_t first = reinterpret_cast<uintptr_t>(&source[0].slot);
        const uintptr_t address = reinterpret_cast<uintptr_t>(slot);
        if (address < first || (address - first) % sizeof(ChipTaskStorage) != 0 ||
            (address - first) / sizeof(ChipTaskStorage) >= static_cast<uint64_t>(build.total_tasks))
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        target[(address - first) / sizeof(ChipTaskStorage)].task.graph_definition_offset = definition_offset;
    }
    return 0;
}

}  // namespace

int make_kernel_graph_layout(uint64_t task_window, RuntimeArenaLayout &out) try {
    static_assert(READY_QUEUE_CAPACITY_LIMIT == 32768);
    if (task_window == 0 || task_window > READY_QUEUE_CAPACITY_LIMIT)
        return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
    DeviceArena arena;
    out = runtime_reserve_layout(arena, task_window);
    return 0;
} catch (const std::bad_alloc &) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int make_graph_launch_template(
    const GraphBuild &build, const RuntimeContext &runtime, const GraphSlotRegistration &registration,
    const GraphInvocationIdentity &identity, GraphLaunchTemplate &out
) try {
    if (!build.build_complete || !build.graph_state || build.sm_mirror == nullptr || build.total_tasks < 0)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (identity.callable_id < 0 || identity.callable_id >= MAX_REGISTERED_CALLABLE_IDS ||
        !simpler::kernel::valid_invocation_counts(identity.tensor_count, identity.scalar_count) ||
        identity.callable_hash == 0 || identity.argument_hash == 0 || identity.function_hash == 0 ||
        identity.generation == 0 || !is_valid_graph_slot_registration(registration))
        return PTO_RUNTIME_ERR_INTERNAL;
    RuntimeArenaLayout layout{};
    int rc = make_kernel_graph_layout(build.task_capacity, layout);
    if (rc != 0) return rc;
    layout.sched.capacities = build.ready_queue_capacities;
    const auto mirror = sm_layout::mirror_extents(build.task_capacity);
    if (static_cast<uint64_t>(build.total_tasks) > build.task_capacity ||
        build.bind_usage.submitted_tasks != static_cast<uint64_t>(build.total_tasks) ||
        build.bind_usage.fanin_elems > mirror.fanin_elems || build.bind_usage.tensor_elems > mirror.tensor_elems ||
        build.bind_usage.scalar_elems > mirror.scalar_elems ||
        build.image_bytes != sm_layout::segment_offsets(sm_layout::image_extents(build.bind_usage)).end)
        return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
    if (!valid_build_external_tensors(build)) return PTO_RUNTIME_ERR_INTERNAL;
    GraphResourceRequirements required{};
    rc = measure_graph_resources(build, layout, build.architecture, 0, required);
    if (rc != 0) return rc;
    struct {
        GraphDestination heap, runtime_image, definitions, scheduler;
    } binding{
        registration.destinations[0], registration.destinations[1], registration.destinations[2],
        registration.destinations[3]
    };
    if (required.gm_heap_bytes > binding.heap.capacity ||
        required.runtime_arena_bytes > binding.runtime_image.capacity ||
        required.graph_definition_bytes > binding.definitions.capacity)
        return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
    if (binding.heap.address >= HEAP_VIRTUAL_BASE || binding.heap.capacity > HEAP_VIRTUAL_BASE - binding.heap.address)
        return PTO_RUNTIME_ERR_INTERNAL;
    GraphPacketHeader header{};
    header.magic = GRAPH_PACKET_MAGIC;
    header.version = GRAPH_PACKET_VERSION;
    header.header_bytes = sizeof(header);
    header.slot_generation = registration.slot_generation;
    header.device_id = registration.device_id;
    header.runtime_binary_id = registration.runtime_binary_id;
    header.callable_hash = identity.callable_hash;
    header.argument_hash = identity.argument_hash;
    header.function_hash = identity.function_hash;
    header.runtime_offset = layout.off_runtime;
    header.sm_offset = layout.off_copied_end;
    header.task_window = build.task_capacity;
    header.total_tasks = build.total_tasks;
    header.heap_bytes = build.heap_bytes;
    header.destinations[0] = {binding.heap.address, binding.heap.capacity};
    header.destinations[1] = {binding.runtime_image.address, binding.runtime_image.capacity};
    header.destinations[2] = {binding.definitions.address, binding.definitions.capacity};
    header.destinations[3] = {binding.scheduler.address, binding.scheduler.capacity};
    std::vector<GraphImageRegion> regions;
    const uint64_t image_bytes[] = {
        0, required.runtime_arena_bytes - layout.off_copied_begin, required.graph_definition_bytes, 0
    };
    for (uint32_t i = 1; i <= 3; ++i) {
        if (image_bytes[i] == 0) continue;
        const uint64_t offset = (header.payload_bytes + 63) & ~uint64_t{63};
        header.payload_bytes = offset;
        if (!add_size(image_bytes[i], header.payload_bytes))
            return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
        regions.push_back(
            {static_cast<GraphImageKind>(i), 0, offset, image_bytes[i], i == 1 ? layout.off_copied_begin : 0}
        );
    }
    header.region_count = regions.size();
    header.payload_offset = (sizeof(header) + regions.size() * sizeof(GraphImageRegion) + 63) & ~uint64_t{63};
    header.total_bytes = header.payload_offset;
    if (!add_size(header.payload_bytes, header.total_bytes) ||
        header.total_bytes > UINT32_MAX - sizeof(SimplerKernelInvocationHeader))
        return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
    GraphLaunchTemplate next;
    next.size_ = sizeof(SimplerKernelInvocationHeader) + header.total_bytes;
    if (next.size_ > registration.max_packet_bytes) return PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
    next.storage_.resize((next.size_ + 7) / 8, 0);
    auto *packet = reinterpret_cast<std::byte *>(next.storage_.data());
    auto *payload = packet + sizeof(SimplerKernelInvocationHeader) + header.payload_offset;
    AlignedImage runtime_image(required.runtime_arena_bytes);
    auto *image = runtime_image.data();
    RuntimeContext pristine{};
    pristine.mode = MODE_EXECUTE;
    pristine.inline_completed_tasks = runtime.inline_completed_tasks;
    pristine.prebuilt_layout = layout;
    std::memcpy(image + layout.off_runtime, &pristine, sizeof(pristine));
    sm_layout::compact_live_image(
        static_cast<const char *>(build.sm_mirror), build.task_capacity, build.bind_usage,
        {binding.heap.address, build.heap_bytes}, reinterpret_cast<char *>(image + layout.off_copied_end)
    );
    reinterpret_cast<SharedMemoryHeader *>(image + layout.off_copied_end)->tasks.total_tasks = build.total_tasks;
    std::byte *definition_image = nullptr;
    for (const auto &region : regions)
        if (region.kind == GraphImageKind::Definitions) definition_image = payload + region.source_offset;
    rc = copy_definitions(build, definition_image, required.graph_definition_bytes);
    if (rc != 0) return rc;
    rc = bind_definition_images(build, image + layout.off_copied_end);
    if (rc != 0) return rc;
    std::memcpy(payload, image + layout.off_copied_begin, required.runtime_arena_bytes - layout.off_copied_begin);
    SimplerKernelInvocationHeader invocation{};
    invocation.mode = SIMPLER_MODE_KERNEL;
    invocation.callable_id = identity.callable_id;
    invocation.generation = identity.generation;
    invocation.tensor_count = identity.tensor_count;
    invocation.scalar_count = identity.scalar_count;
    invocation.host_copy_tensor_count = 0;
    invocation.payload_bytes = header.total_bytes;
    std::memcpy(packet, &invocation, sizeof(invocation));
    std::memcpy(packet + sizeof(invocation), &header, sizeof(header));
    std::memcpy(
        packet + sizeof(invocation) + sizeof(header), regions.data(), regions.size() * sizeof(GraphImageRegion)
    );
    header.checksum = graph_packet_checksum(packet, next.size_);
    std::memcpy(packet + sizeof(invocation), &header, sizeof(header));
    if (validate_graph_packet(packet, next.size_, GraphPacketAddress::HostTemplate) != GraphPacketStatus::Ok)
        return PTO_RUNTIME_ERR_INTERNAL;
    out = std::move(next);
    return 0;
} catch (const std::bad_alloc &) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int make_graph_host_args(const GraphLaunchTemplate &source, GraphHostArgs &out) {
    ParsedGraphPacket parsed;
    if (parse_graph_packet(source.data(), source.size(), GraphPacketAddress::HostTemplate, parsed) !=
        GraphPacketStatus::Ok)
        return PTO_RUNTIME_ERR_INTERNAL;
    try {
        GraphHostArgs next;
        next.bytes = source.size();
        next.storage.resize((next.bytes + 7) / 8);
        std::memcpy(next.storage.data(), source.data(), next.bytes);
        next.address_offset = sizeof(SimplerKernelInvocationHeader) + offsetof(GraphPacketHeader, inline_payload_addr);
        next.data_offset = sizeof(SimplerKernelInvocationHeader) + parsed.graph.payload_offset;
        out = std::move(next);
        return 0;
    } catch (const std::bad_alloc &) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int submit_graph_template(const GraphLaunchTemplate &source, const GraphHostLaunchOps &ops) {
    if (ops.launch == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    GraphHostArgs args;
    const int rc = make_graph_host_args(source, args);
    return rc == 0 ? ops.launch(ops.context, args) : rc;
}

}  // namespace hbg
