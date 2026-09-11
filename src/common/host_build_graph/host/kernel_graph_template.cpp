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

#include <algorithm>
#include <new>
#include <unordered_map>

#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/runtime_core.h"

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

struct DefinitionImage {
    uint64_t offset;
    const GraphDefinition *definition;
};

int copy_definitions(
    const GraphBuild &build, std::byte *out, uint64_t capacity, std::unordered_map<uint64_t, DefinitionImage> &images
) {
    auto definitions = graph_host_definitions(*build.graph_state);
    std::sort(definitions.entries.begin(), definitions.entries.end(), [](const auto &lhs, const auto &rhs) {
        return lhs.full_key < rhs.full_key;
    });
    uint64_t cursor = graph_host_arena_used(*build.graph_state);
    for (const auto &entry : definitions.entries) {
        const auto *source = graph_host_definition_data(*build.graph_state, entry.full_key);
        if (source == nullptr || entry.bytes < sizeof(GraphDefinition)) return PTO_RUNTIME_ERR_INTERNAL;
        const auto *definition = reinterpret_cast<const GraphDefinition *>(source);
        if (definition->total_bytes != entry.bytes || definition->full_key != entry.full_key ||
            definition->task_count <= 0 || definition->task_count > MAX_IN_GRAPH_TASKS)
            return PTO_RUNTIME_ERR_INTERNAL;
        const uint64_t object = entry.spill == nullptr ? entry.object_offset : cursor;
        const uint64_t framed = sizeof(GraphDefinitionHeader) + entry.bytes;
        const uint64_t padded =
            (framed + GRAPH_DEFINITION_OBJECT_ALIGN - 1) & ~(uint64_t{GRAPH_DEFINITION_OBJECT_ALIGN} - 1);
        if (!graph_span_fits(object, padded, capacity)) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
        GraphDefinitionHeader framing{};
        framing.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
        framing.full_key = entry.full_key;
        framing.definition_bytes = entry.bytes;
        std::memcpy(out + object, &framing, sizeof(framing));
        std::memcpy(out + object + sizeof(framing), source, entry.bytes);
        images.emplace(entry.full_key, DefinitionImage{object + sizeof(framing), definition});
        if (entry.spill != nullptr) cursor += padded;
    }
    return 0;
}

int bind_definition_images(
    const GraphBuild &build, std::byte *sm, uint64_t definition_base,
    const std::unordered_map<uint64_t, DefinitionImage> &images, uint64_t ready_capacity
) {
    const auto source_offsets = sm_layout::segment_offsets(build.task_capacity);
    const auto target_offsets = sm_layout::segment_offsets(sm_layout::image_extents(build.usage));
    const auto *source_storage = reinterpret_cast<const ChipTaskStorage *>(
        static_cast<const std::byte *>(build.host_sm) + source_offsets.storage
    );
    auto *target_storage = reinterpret_cast<ChipTaskStorage *>(sm + target_offsets.storage);
    ReadyQueuePopulations populations = build.ready_queue_populations;
    size_t graph_tasks = 0;
    for (int32_t i = 0; i < build.total_tasks; ++i) {
        // graph_context in a build can be a Host cache pointer or a previous
        // program upload's address. Neither is a source for the kernel image.
        target_storage[i].slot.graph_context = nullptr;
        graph_tasks += target_storage[i].slot.task_kind == TaskKind::GRAPH;
    }
    const size_t uploads = graph_host_upload_count(*build.graph_state);
    if (uploads != graph_tasks) return PTO_RUNTIME_ERR_INTERNAL;
    std::vector<bool> seen(static_cast<size_t>(build.total_tasks), false);
    for (size_t i = 0; i < uploads; ++i) {
        const auto upload = graph_host_upload(*build.graph_state, i);
        if (!upload || upload->outer_slot == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        const uintptr_t first = reinterpret_cast<uintptr_t>(&source_storage[0].slot);
        const uintptr_t address = reinterpret_cast<uintptr_t>(upload->outer_slot);
        if (address < first || (address - first) % sizeof(ChipTaskStorage) != 0) return PTO_RUNTIME_ERR_INTERNAL;
        const uint64_t task = (address - first) / sizeof(ChipTaskStorage);
        if (task >= static_cast<uint64_t>(build.total_tasks) || seen[task]) return PTO_RUNTIME_ERR_INTERNAL;
        seen[task] = true;
        const auto it = images.find(upload->full_key);
        if (it == images.end()) return PTO_RUNTIME_ERR_INTERNAL;
        const auto &definition = *it->second.definition;
        const auto &source = source_storage[task];
        GraphExecutionStorageLayout storage{};
        if (source.slot.task_kind != TaskKind::GRAPH || source.payload.tensor_count != definition.boundary_count ||
            source.payload.scalar_count != definition.boundary_scalar_count ||
            !graph_execution_storage_layout(
                definition.task_count, definition.tensor_arg_count, definition.scalar_arg_count, &storage
            ) ||
            storage.total_bytes != definition.execution_storage_bytes)
            return PTO_RUNTIME_ERR_INTERNAL;
        const uint64_t begin = reinterpret_cast<uintptr_t>(source.task.packed_buffer_base);
        const uint64_t end = reinterpret_cast<uintptr_t>(source.task.packed_buffer_end);
        if (begin < HEAP_VIRTUAL_BASE || end < begin ||
            !graph_span_fits(begin - HEAP_VIRTUAL_BASE, end - begin, build.heap_bytes) ||
            !graph_span_fits(definition.required_heap, storage.total_bytes, end - begin) ||
            (begin + definition.required_heap) % alignof(ChipTaskStorage) != 0)
            return PTO_RUNTIME_ERR_INTERNAL;
        const auto *tasks = graph_definition_array<InGraphTaskDefinition>(
            definition, definition.off_in_graph_tasks, definition.task_count
        );
        if (tasks == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        for (int32_t j = 0; j < definition.task_count; ++j) {
            const ActiveMask mask(tasks[j].active_mask);
            populations.add_task(
                mask, TaskAttrs(tasks[j].task_attrs), mask.is_dummy() ? TaskKind::DUMMY : TaskKind::KERNEL
            );
        }
        target_storage[task].slot.graph_context = reinterpret_cast<void *>(definition_base + it->second.offset);
    }
    ReadyQueueCapacities required{};
    if (derive_ready_queue_capacities(populations, &required) != 0) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i)
        if (required.ready[i] > ready_capacity || required.ready_sync[i] > ready_capacity)
            return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    if (required.dummy > ready_capacity || required.graph_ready > ready_capacity ||
        required.graph_prepare > ready_capacity)
        return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    return 0;
}

}  // namespace

int make_kernel_graph_layout(uint64_t task_window, RuntimeArenaLayout &out) try {
    static_assert(READY_QUEUE_CAPACITY_LIMIT == 32768);
    if (task_window == 0 || task_window > READY_QUEUE_CAPACITY_LIMIT) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    uint64_t ready_capacity = 64;
    while (ready_capacity < task_window)
        ready_capacity <<= 1;
    DeviceArena arena;
    out = runtime_reserve_layout(arena, task_window, ready_capacity);
    return 0;
} catch (const std::bad_alloc &) {
    return PTO_RUNTIME_ERR_INTERNAL;
}

int make_graph_launch_template(
    const GraphBuild &build, const RuntimeContext &runtime, const KernelExecutionState &context, int device_id,
    uint64_t slot_generation, const GraphInvocationIdentity &identity, GraphLaunchTemplate &out
) try {
    if (!build.ready || !build.graph_state || build.host_sm == nullptr || build.total_tasks < 0)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (identity.callable_id < 0 || identity.tensor_count < 0 || identity.scalar_count < 0 ||
        identity.tensor_count > CHIP_MAX_TENSOR_ARGS || identity.scalar_count > CHIP_MAX_SCALAR_ARGS ||
        identity.callable_generation == 0 || identity.callable_hash == 0 || identity.argument_hash == 0 ||
        identity.function_hash == 0)
        return PTO_RUNTIME_ERR_INTERNAL;
    RuntimeArenaLayout layout{};
    int rc = make_kernel_graph_layout(build.task_capacity, layout);
    if (rc != 0) return rc;
    const auto mirror = sm_layout::mirror_extents(build.task_capacity);
    if (static_cast<uint64_t>(build.total_tasks) >= build.task_capacity ||
        build.usage.submitted_tasks != static_cast<uint64_t>(build.total_tasks) ||
        build.usage.fanin_elems > mirror.fanin_elems || build.usage.tensor_elems > mirror.tensor_elems ||
        build.usage.scalar_elems > mirror.scalar_elems ||
        build.image_bytes != sm_layout::segment_offsets(sm_layout::image_extents(build.usage)).end)
        return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    GraphResourceRequirements required{};
    rc = get_graph_resource_requirements(build, layout, required);
    if (rc != 0) return rc;
    KernelWorkingBinding binding{};
    rc = bind_kernel_resources_for_launch(context, device_id, slot_generation, required, binding);
    if (rc != 0) return rc;
    if (binding.heap.address >= HEAP_VIRTUAL_BASE || binding.heap.capacity > HEAP_VIRTUAL_BASE - binding.heap.address)
        return PTO_RUNTIME_ERR_INTERNAL;
    GraphPacketHeader header{};
    header.magic = GRAPH_PACKET_MAGIC;
    header.version = GRAPH_PACKET_VERSION;
    header.header_bytes = sizeof(header);
    header.slot_generation = slot_generation;
    header.callable_hash = identity.callable_hash;
    header.argument_hash = identity.argument_hash;
    header.function_hash = identity.function_hash;
    header.runtime_offset = layout.off_runtime;
    header.sm_offset = layout.off_copied_end;
    header.task_window = build.task_capacity;
    header.total_tasks = build.total_tasks;
    header.destinations[0] = {binding.heap.address, binding.heap.capacity};
    header.destinations[1] = {binding.runtime_image.address, binding.runtime_image.capacity};
    header.destinations[2] = {binding.definitions.address, binding.definitions.capacity};
    header.destinations[3] = {binding.scheduler.address, binding.scheduler.capacity};
    std::vector<GraphImageRegion> regions;
    for (uint32_t i = 1; i <= 3; ++i) {
        const auto &destination = header.destinations[i];
        if (destination.capacity == 0) continue;
        const uint64_t offset = (header.payload_bytes + 63) & ~uint64_t{63};
        header.payload_bytes = offset;
        if (!add_size(destination.capacity, header.payload_bytes)) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
        regions.push_back({static_cast<GraphImageKind>(i), 0, offset, destination.capacity, 0});
    }
    header.region_count = regions.size();
    header.payload_offset = (sizeof(header) + regions.size() * sizeof(GraphImageRegion) + 63) & ~uint64_t{63};
    header.total_bytes = header.payload_offset;
    if (!add_size(header.payload_bytes, header.total_bytes) ||
        header.total_bytes > UINT32_MAX - sizeof(SimplerKernelInvocationHeader))
        return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    GraphLaunchTemplate next;
    next.size_ = sizeof(SimplerKernelInvocationHeader) + header.total_bytes;
    next.storage_.resize((next.size_ + 7) / 8, 0);
    auto *packet = reinterpret_cast<std::byte *>(next.storage_.data());
    auto *payload = packet + sizeof(SimplerKernelInvocationHeader) + header.payload_offset;
    AlignedImage runtime_image(binding.runtime_image.capacity);
    auto *image = runtime_image.data();
    RuntimeContext pristine{};
    pristine.mode = MODE_EXECUTE;
    pristine.inline_completed_tasks = runtime.inline_completed_tasks;
    pristine.active_callable_hash = identity.callable_hash;
    pristine.prebuilt_layout = layout;
    std::memcpy(image + layout.off_runtime, &pristine, sizeof(pristine));
    sm_layout::compact_live_image(
        static_cast<const char *>(build.host_sm), build.task_capacity, build.usage,
        {binding.heap.address, build.heap_bytes}, reinterpret_cast<char *>(image + layout.off_copied_end)
    );
    reinterpret_cast<SharedMemoryHeader *>(image + layout.off_copied_end)->tasks.total_tasks = build.total_tasks;
    std::unordered_map<uint64_t, DefinitionImage> definitions;
    std::byte *definition_image = nullptr;
    for (const auto &region : regions)
        if (region.kind == GraphImageKind::Definitions) definition_image = payload + region.source_offset;
    rc = copy_definitions(build, definition_image, binding.definitions.capacity, definitions);
    if (rc != 0) return rc;
    rc = bind_definition_images(
        build, image + layout.off_copied_end, binding.definitions.address, definitions, layout.sched.capacities.dummy
    );
    if (rc != 0) return rc;
    std::memcpy(payload, image, binding.runtime_image.capacity);
    SimplerKernelInvocationHeader invocation{};
    invocation.abi_version = SIMPLER_KERNEL_INVOCATION_ABI_VERSION;
    invocation.header_bytes = sizeof(invocation);
    invocation.mode = SIMPLER_MODE_KERNEL;
    invocation.callable_id = identity.callable_id;
    invocation.generation = identity.callable_generation;
    invocation.tensor_count = identity.tensor_count;
    invocation.scalar_count = identity.scalar_count;
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
    if (validate_graph_packet(source.data(), source.size(), GraphPacketAddress::HostTemplate) != GraphPacketStatus::Ok)
        return PTO_RUNTIME_ERR_INTERNAL;
    try {
        GraphHostArgs next;
        next.bytes = source.size();
        next.storage.resize((next.bytes + 7) / 8);
        std::memcpy(next.storage.data(), source.data(), next.bytes);
        GraphPacketHeader header{};
        std::memcpy(
            &header, static_cast<const std::byte *>(source.data()) + sizeof(SimplerKernelInvocationHeader),
            sizeof(header)
        );
        next.address_offset = sizeof(SimplerKernelInvocationHeader) + offsetof(GraphPacketHeader, inline_payload_addr);
        next.data_offset = sizeof(SimplerKernelInvocationHeader) + header.payload_offset;
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
