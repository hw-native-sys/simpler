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
#include "host_build_graph/kernel_graph_slot_registry.h"

#include <atomic>

#include "aicpu/cache_maintenance.h"

namespace hbg {
namespace {
std::atomic<GraphSlotRegistry *> current_registry{nullptr};

bool valid_registry(const GraphSlotRegistry *registry) {
    if (registry == nullptr || reinterpret_cast<uintptr_t>(registry) % 1024 != 0) return false;
    cache_invalidate_range(registry, offsetof(GraphSlotRegistry, registration));
    if (registry->magic != GRAPH_REGISTRY_MAGIC || registry->version != GRAPH_SLOT_VERSION ||
        registry->bytes != sizeof(*registry) || registry->device_id < 0 || registry->context_generation == 0 ||
        registry->runtime_binary_id == 0)
        return false;
    for (uint64_t word : registry->reserved)
        if (word != 0) return false;
    return true;
}

bool matches_registry(const GraphSlotRegistry &registry, const GraphSlotRegistration &registration) {
    return registration.registry.address == reinterpret_cast<uintptr_t>(&registry) &&
           registration.device_id == registry.device_id &&
           registration.slot_generation == registry.context_generation &&
           registration.runtime_binary_id == registry.runtime_binary_id;
}
}  // namespace

GraphSlotStatus initialize_graph_slot_registry(
    GraphSlotRegistry *registry, int device_id, uint64_t generation, uint64_t runtime_binary_id
) noexcept {
    if (registry == nullptr || reinterpret_cast<uintptr_t>(registry) % 1024 != 0 || device_id < 0 || generation == 0 ||
        runtime_binary_id == 0)
        return GraphSlotStatus::InvalidRegistry;
    if (current_registry.load(std::memory_order_acquire) == registry) return GraphSlotStatus::Conflict;
    GraphSlotRegistry initial{};
    initial.magic = GRAPH_REGISTRY_MAGIC;
    initial.version = GRAPH_SLOT_VERSION;
    initial.bytes = sizeof(initial);
    initial.phase = static_cast<uint32_t>(GraphSlotPhase::Empty);
    initial.device_id = device_id;
    initial.context_generation = generation;
    initial.runtime_binary_id = runtime_binary_id;
    std::memcpy(registry, &initial, sizeof(initial));
    cache_flush_range(registry, sizeof(*registry));
    return GraphSlotStatus::Ok;
}

GraphSlotStatus
register_graph_execution_slot(GraphSlotRegistry *registry, const void *registration, size_t bytes) noexcept {
    if (registration == nullptr || bytes != sizeof(GraphSlotRegistration)) return GraphSlotStatus::InvalidRegistration;
    GraphSlotRegistration candidate{};
    std::memcpy(&candidate, registration, sizeof(candidate));
    if (!valid_graph_slot_registration(candidate)) return GraphSlotStatus::InvalidRegistration;
    if (!valid_registry(registry)) return GraphSlotStatus::InvalidRegistry;
    if (!matches_registry(*registry, candidate)) return GraphSlotStatus::Conflict;
    for (;;) {
        uint32_t phase = __atomic_load_n(&registry->phase, __ATOMIC_ACQUIRE);
        if (phase == static_cast<uint32_t>(GraphSlotPhase::Ready)) {
            cache_invalidate_range(&registry->registration, sizeof(registry->registration));
            return std::memcmp(&registry->registration, &candidate, sizeof(candidate)) == 0 ? GraphSlotStatus::Ok :
                                                                                              GraphSlotStatus::Conflict;
        }
        if (phase == static_cast<uint32_t>(GraphSlotPhase::Publishing)) return GraphSlotStatus::Publishing;
        if (phase != static_cast<uint32_t>(GraphSlotPhase::Empty)) return GraphSlotStatus::InvalidRegistry;
        if (!__atomic_compare_exchange_n(
                &registry->phase, &phase, static_cast<uint32_t>(GraphSlotPhase::Publishing), false, __ATOMIC_ACQ_REL,
                __ATOMIC_ACQUIRE
            ))
            continue;
        registry->registration = candidate;
        cache_flush_range(&registry->registration, sizeof(registry->registration));
        __atomic_store_n(&registry->phase, static_cast<uint32_t>(GraphSlotPhase::Ready), __ATOMIC_RELEASE);
        cache_flush_range(registry, offsetof(GraphSlotRegistry, registration));
        return GraphSlotStatus::Ok;
    }
}

GraphSlotStatus acquire_graph_execution_slot(
    const GraphSlotRegistry *registry, int device_id, uint64_t runtime_binary_id, GraphSlotRegistration &out
) noexcept {
    if (!valid_registry(registry)) return GraphSlotStatus::InvalidRegistry;
    if (registry->device_id != device_id) return GraphSlotStatus::DeviceMismatch;
    if (registry->runtime_binary_id != runtime_binary_id) return GraphSlotStatus::BinaryMismatch;
    const uint32_t phase = __atomic_load_n(&registry->phase, __ATOMIC_ACQUIRE);
    if (phase == static_cast<uint32_t>(GraphSlotPhase::Empty)) return GraphSlotStatus::NotReady;
    if (phase == static_cast<uint32_t>(GraphSlotPhase::Publishing)) return GraphSlotStatus::Publishing;
    if (phase != static_cast<uint32_t>(GraphSlotPhase::Ready)) return GraphSlotStatus::InvalidRegistry;
    cache_invalidate_range(&registry->registration, sizeof(registry->registration));
    const GraphSlotRegistration candidate = registry->registration;
    if (!valid_graph_slot_registration(candidate) || !matches_registry(*registry, candidate))
        return GraphSlotStatus::InvalidRegistry;
    out = candidate;
    return GraphSlotStatus::Ok;
}

GraphSlotStatus
bind_graph_slot_registry(GraphSlotRegistry *registry, int device_id, uint64_t runtime_binary_id) noexcept {
    GraphSlotRegistration registration{};
    const auto status = acquire_graph_execution_slot(registry, device_id, runtime_binary_id, registration);
    if (status != GraphSlotStatus::Ok) return status;
    GraphSlotRegistry *expected = nullptr;
    if (!current_registry.compare_exchange_strong(
            expected, registry, std::memory_order_acq_rel, std::memory_order_acquire
        ) &&
        expected != registry)
        return GraphSlotStatus::Conflict;
    return GraphSlotStatus::Ok;
}

bool detach_graph_slot_registry(GraphSlotRegistry *registry) noexcept {
    if (registry == nullptr) return false;
    return current_registry.compare_exchange_strong(registry, nullptr, std::memory_order_acq_rel);
}

GraphSlotStatus admit_graph_packet_for_restore(
    const void *packet, size_t bytes, int device_id, uint64_t runtime_binary_id, GraphRestoreView &out
) noexcept {
    const GraphSlotRegistry *registry = current_registry.load(std::memory_order_acquire);
    if (registry == nullptr) return GraphSlotStatus::NotReady;
    GraphSlotRegistration registration{};
    const auto status = acquire_graph_execution_slot(registry, device_id, runtime_binary_id, registration);
    if (status != GraphSlotStatus::Ok) return status;
    if (packet == nullptr || bytes > registration.max_packet_bytes) return GraphSlotStatus::InvalidPacket;
    const GraphDestination source{reinterpret_cast<uintptr_t>(packet), bytes};
    if (!graph_span_fits(source.address, source.capacity, UINT64_MAX)) return GraphSlotStatus::InvalidPacket;
    if (graph_windows_overlap(source, registration.registry)) return GraphSlotStatus::SourceOverlap;
    for (const auto &destination : registration.destinations)
        if (graph_windows_overlap(source, destination)) return GraphSlotStatus::SourceOverlap;
    if (validate_graph_packet(packet, bytes, GraphPacketAddress::DeviceCopy) != GraphPacketStatus::Ok)
        return GraphSlotStatus::InvalidPacket;
    GraphPacketHeader header{};
    std::memcpy(
        &header, static_cast<const std::byte *>(packet) + sizeof(SimplerKernelInvocationHeader), sizeof(header)
    );
    if (header.device_id != registration.device_id) return GraphSlotStatus::DeviceMismatch;
    if (header.slot_generation != registration.slot_generation) return GraphSlotStatus::GenerationMismatch;
    if (header.runtime_binary_id != registration.runtime_binary_id) return GraphSlotStatus::BinaryMismatch;
    for (size_t i = 0; i < 4; ++i)
        if (header.destinations[i].address != registration.destinations[i].address ||
            header.destinations[i].capacity != registration.destinations[i].capacity)
            return GraphSlotStatus::BindingMismatch;
    out = {
        registration, header,
        static_cast<const std::byte *>(packet) + sizeof(SimplerKernelInvocationHeader) + header.payload_offset
    };
    return GraphSlotStatus::Ok;
}

}  // namespace hbg
