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

#include "host_build_graph/kernel_graph_wire.h"

namespace hbg {

inline constexpr uint32_t GRAPH_SLOT_MAGIC = 0x53474248;      // HBGS
inline constexpr uint32_t GRAPH_REGISTRY_MAGIC = 0x52474248;  // HBGR
inline constexpr uint16_t GRAPH_SLOT_VERSION = 2;
inline constexpr uint32_t GRAPH_SLOT_FROZEN_SERIAL = 3;

enum class GraphSlotPhase : uint32_t { Empty, Publishing, Ready, Poisoned };
enum class GraphSlotStatus : uint32_t {
    Ok,
    InvalidRegistration,
    InvalidRegistry,
    NotReady,
    Publishing,
    Conflict,
    DeviceMismatch,
    GenerationMismatch,
    BinaryMismatch,
    InvalidPacket,
    BindingMismatch,
    SourceOverlap,
    CallableMismatch,
    Poisoned,
    SourceUnavailable
};

// Sealed only from context-owned frozen allocations, never from a launch packet.
// The checksum detects corruption; the ordered prepare control path supplies authority.
struct GraphSlotRegistration {
    uint32_t magic;
    uint16_t version;
    uint16_t bytes;
    uint32_t flags;
    int32_t device_id;
    uint64_t slot_generation;
    uint64_t runtime_binary_id;
    uint64_t max_packet_bytes;
    GraphDestination destinations[4];
    GraphDestination registry;
    uint64_t checksum;
};

enum class GraphRestorePhase : uint32_t { Idle, Restoring, Ready, Failed };

// One leader writes this line; peers acquire phase after the entry's invocation
// barrier. attempt is device-owned and advances even when a restore fails.
struct alignas(64) GraphRestoreControl {
    uint64_t attempt;
    uint64_t committed_generation;
    uint64_t runtime_address;
    uint64_t sm_bytes;
    uint32_t phase;
    uint32_t status;
    uint32_t total_tasks;
    uint32_t reserved0;
    uint64_t reserved[2];
};
static_assert(sizeof(GraphRestoreControl) == 64);
static_assert(std::is_standard_layout_v<GraphRestoreControl> && std::is_trivially_copyable_v<GraphRestoreControl>);

// Context-owned device memory. Publication lives on a separate cache line from
// the immutable registration. phase is accessed through atomic builtins on AICPU.
struct alignas(64) GraphSlotRegistry {
    uint32_t magic;
    uint16_t version;
    uint16_t bytes;
    uint32_t phase;
    int32_t device_id;
    uint64_t context_generation;
    uint64_t runtime_binary_id;
    uint64_t reserved[4];
    GraphSlotRegistration registration;
    GraphRestoreControl restore;
};

static_assert(std::is_standard_layout_v<GraphSlotRegistration> && std::is_trivially_copyable_v<GraphSlotRegistration>);
static_assert(std::is_standard_layout_v<GraphSlotRegistry> && std::is_trivially_copyable_v<GraphSlotRegistry>);
static_assert(sizeof(GraphSlotRegistration) == 128 && alignof(GraphSlotRegistration) == 8);
static_assert(offsetof(GraphSlotRegistration, slot_generation) == 16);
static_assert(offsetof(GraphSlotRegistration, destinations) == 40);
static_assert(offsetof(GraphSlotRegistration, registry) == 104);
static_assert(offsetof(GraphSlotRegistration, checksum) == 120);
static_assert(sizeof(GraphSlotRegistry) == 256 && alignof(GraphSlotRegistry) == 64);
static_assert(offsetof(GraphSlotRegistry, phase) == 8);
static_assert(offsetof(GraphSlotRegistry, registration) == 64);
static_assert(offsetof(GraphSlotRegistry, restore) == 192);

// Both inputs must already have non-overflowing bounds.
inline bool graph_windows_overlap(GraphDestination a, GraphDestination b) noexcept {
    return a.capacity && b.capacity && a.address < b.address + b.capacity && b.address < a.address + a.capacity;
}

inline bool graph_slot_packet_size(const GraphDestination *destinations, uint64_t &bytes) noexcept {
    if (destinations == nullptr) return false;
    uint64_t payload = 0;
    uint64_t count = 0;
    for (size_t i = 1; i < 4; ++i) {
        const uint64_t capacity = destinations[i].capacity;
        if (capacity == 0) continue;
        if (payload > UINT32_MAX - 63) return false;
        payload = (payload + 63) & ~uint64_t{63};
        if (capacity > UINT32_MAX - payload) return false;
        payload += capacity;
        ++count;
    }
    const uint64_t prefix = sizeof(SimplerKernelInvocationHeader) +
                            ((sizeof(GraphPacketHeader) + count * sizeof(GraphImageRegion) + 63) & ~uint64_t{63});
    if (count == 0 || payload > UINT32_MAX - prefix) return false;
    bytes = prefix + payload;
    return true;
}

inline uint64_t graph_slot_checksum(const GraphSlotRegistration &registration) noexcept {
    return simpler::common::utils::fnv1a_64(&registration, offsetof(GraphSlotRegistration, checksum));
}

inline bool valid_graph_slot_registration(const GraphSlotRegistration &registration) noexcept {
    if (registration.magic != GRAPH_SLOT_MAGIC || registration.version != GRAPH_SLOT_VERSION ||
        registration.bytes != sizeof(registration) || registration.flags != GRAPH_SLOT_FROZEN_SERIAL ||
        registration.device_id < 0 || registration.slot_generation == 0 || registration.runtime_binary_id == 0 ||
        registration.registry.capacity != sizeof(GraphSlotRegistry))
        return false;
    for (size_t i = 0; i < 5; ++i) {
        const auto &region = i == 4 ? registration.registry : registration.destinations[i];
        if ((region.address == 0) != (region.capacity == 0) || ((i < 2 || i == 4) && region.capacity == 0) ||
            region.address % 1024 != 0 || !graph_span_fits(region.address, region.capacity, UINT64_MAX))
            return false;
        for (size_t j = 0; j < i; ++j)
            if (graph_windows_overlap(region, registration.destinations[j])) return false;
    }
    uint64_t packet_bytes = 0;
    return graph_slot_packet_size(registration.destinations, packet_bytes) &&
           packet_bytes == registration.max_packet_bytes && registration.checksum == graph_slot_checksum(registration);
}

}  // namespace hbg
