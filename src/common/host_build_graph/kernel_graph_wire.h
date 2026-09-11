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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <type_traits>

#include "task_interface/kernel_invocation_header.h"
#include "task_interface/arg_direction.h"
#include "utils/fnv1a_64.h"

namespace hbg {

inline constexpr uint32_t GRAPH_PACKET_MAGIC = 0x50474248;  // HBGP
inline constexpr uint16_t GRAPH_PACKET_VERSION = 2;

enum class GraphImageKind : uint32_t { Runtime = 1, Definitions = 2, Scheduler = 3 };
enum class GraphPacketAddress : uint32_t { HostTemplate, DeviceCopy };
enum class GraphPacketStatus : uint32_t {
    Ok,
    InvalidEnvelope,
    InvalidHeader,
    InvalidBinding,
    InvalidRegion,
    InvalidChecksum
};

struct GraphDestination {
    uint64_t address;
    uint64_t capacity;
};

// Offsets are relative to inline_payload_addr and the selected destination.
struct GraphImageRegion {
    GraphImageKind kind;
    uint32_t reserved;
    uint64_t source_offset;
    uint64_t bytes;
    uint64_t destination_offset;
};

// Follows the frozen 64-byte common invocation header. Device addresses name
// context-owned storage; they are claims to compare with the registry, not authority.
struct GraphPacketHeader {
    uint32_t magic;
    uint16_t version;
    uint16_t header_bytes;
    uint64_t total_bytes;
    uint64_t payload_offset;  // From this header, past the region table and padding.
    uint64_t payload_bytes;
    uint64_t inline_payload_addr;  // Zero in the template; one CANN placeholder.
    uint64_t checksum;             // Accidental-corruption check, not authentication.
    uint64_t slot_generation;      // Independent of callable residency generation in K9.
    uint64_t callable_hash;
    uint64_t argument_hash;
    uint64_t function_hash;
    uint64_t runtime_offset;
    uint64_t sm_offset;
    uint64_t task_window;
    uint32_t total_tasks;
    uint32_t region_count;
    GraphDestination destinations[4];  // heap, runtime/SM, Definitions, A5 scheduler
    int32_t device_id;
    uint32_t reserved;
    uint64_t runtime_binary_id;  // Runtime ABI/code identity, independent of callable identity.
};

static_assert(std::is_trivially_copyable_v<GraphDestination> && std::is_standard_layout_v<GraphDestination>);
static_assert(std::is_trivially_copyable_v<GraphImageRegion> && std::is_standard_layout_v<GraphImageRegion>);
static_assert(std::is_trivially_copyable_v<GraphPacketHeader> && std::is_standard_layout_v<GraphPacketHeader>);
static_assert(sizeof(GraphDestination) == 16);
static_assert(sizeof(GraphImageRegion) == 32);
static_assert(sizeof(GraphPacketHeader) == 192);
static_assert(alignof(GraphPacketHeader) == 8);
static_assert(offsetof(GraphPacketHeader, inline_payload_addr) == 32);
static_assert(offsetof(GraphPacketHeader, checksum) == 40);
static_assert(offsetof(GraphPacketHeader, slot_generation) == 48);
static_assert(offsetof(GraphPacketHeader, destinations) == 112);
static_assert(offsetof(GraphPacketHeader, device_id) == 176);
static_assert(offsetof(GraphPacketHeader, runtime_binary_id) == 184);
static_assert(offsetof(GraphImageRegion, source_offset) == 8);

inline bool graph_span_fits(uint64_t offset, uint64_t size, uint64_t capacity) noexcept {
    return offset <= capacity && size <= capacity - offset;
}

// The only excluded bytes are the runtime-patched address and this checksum.
// The envelope, binding, identity, region descriptors and padding are covered.
inline uint64_t graph_packet_checksum(const void *packet, size_t size) noexcept {
    constexpr size_t skip = sizeof(SimplerKernelInvocationHeader) + offsetof(GraphPacketHeader, inline_payload_addr);
    if (packet == nullptr || size < skip + 16) return 0;
    const auto *bytes = static_cast<const uint8_t *>(packet);
    auto hash = simpler::common::utils::fnv1a_64(bytes, skip);
    return simpler::common::utils::fnv1a_64_append(hash, bytes + skip + 16, size - skip - 16);
}

// Allocation-free framing validation. No device destination is read or written.
// Registry identity checks and semantic validation of restored images precede
// execution separately; a checksum-valid packet alone never authorizes restore.
inline GraphPacketStatus
validate_graph_packet(const void *packet, size_t size, GraphPacketAddress address_mode) noexcept {
    constexpr size_t prefix = sizeof(SimplerKernelInvocationHeader);
    if (packet == nullptr || size < prefix + sizeof(GraphPacketHeader) || size > UINT32_MAX)
        return GraphPacketStatus::InvalidEnvelope;
    const auto *bytes = static_cast<const uint8_t *>(packet);
    SimplerKernelInvocationHeader invocation{};
    std::memcpy(&invocation, bytes, sizeof(invocation));
    if (invocation.abi_version != SIMPLER_KERNEL_INVOCATION_ABI_VERSION || invocation.header_bytes != prefix ||
        invocation.mode != SIMPLER_MODE_KERNEL || invocation.callable_id < 0 || invocation.generation == 0 ||
        invocation.tensor_count < 0 || invocation.tensor_count > CHIP_MAX_TENSOR_ARGS || invocation.scalar_count < 0 ||
        invocation.scalar_count > CHIP_MAX_SCALAR_ARGS || invocation.host_copy_tensor_count != 0 ||
        invocation.reserved0 != 0 || invocation.reserved[0] != 0 || invocation.reserved[1] != 0 ||
        invocation.payload_bytes != size - prefix)
        return GraphPacketStatus::InvalidEnvelope;
    GraphPacketHeader header{};
    std::memcpy(&header, bytes + prefix, sizeof(header));
    if (header.magic != GRAPH_PACKET_MAGIC || header.version != GRAPH_PACKET_VERSION ||
        header.header_bytes != sizeof(header) || header.total_bytes != size - prefix || header.region_count < 1 ||
        header.region_count > 3 || header.slot_generation == 0 || header.callable_hash == 0 ||
        header.argument_hash == 0 || header.function_hash == 0 || header.task_window == 0 ||
        header.task_window > 32768 || header.total_tasks >= header.task_window || header.reserved != 0 ||
        header.device_id < 0 || header.runtime_binary_id == 0)
        return GraphPacketStatus::InvalidHeader;
    const uint64_t table_end = sizeof(header) + header.region_count * sizeof(GraphImageRegion);
    const uint64_t payload_offset = (table_end + 63) & ~uint64_t{63};
    if (header.payload_offset != payload_offset || payload_offset > header.total_bytes ||
        header.payload_bytes != header.total_bytes - payload_offset)
        return GraphPacketStatus::InvalidHeader;
    const uint64_t base = reinterpret_cast<uintptr_t>(packet);
    if (!graph_span_fits(base, size, UINT64_MAX)) return GraphPacketStatus::InvalidHeader;
    const uint64_t expected_address = base + prefix + payload_offset;
    if ((address_mode == GraphPacketAddress::HostTemplate && header.inline_payload_addr != 0) ||
        (address_mode == GraphPacketAddress::DeviceCopy && header.inline_payload_addr != expected_address) ||
        (address_mode != GraphPacketAddress::HostTemplate && address_mode != GraphPacketAddress::DeviceCopy))
        return GraphPacketStatus::InvalidHeader;
    for (uint64_t i = table_end; i < payload_offset; ++i)
        if (bytes[prefix + i] != 0) return GraphPacketStatus::InvalidHeader;
    for (size_t i = 0; i < 4; ++i) {
        const auto &dst = header.destinations[i];
        if ((dst.capacity == 0) != (dst.address == 0) || (i < 2 && dst.capacity == 0) || dst.address % 1024 != 0 ||
            !graph_span_fits(dst.address, dst.capacity, UINT64_MAX))
            return GraphPacketStatus::InvalidBinding;
        for (size_t j = 0; j < i; ++j) {
            const auto &other = header.destinations[j];
            if (dst.capacity && other.capacity && dst.address < other.address + other.capacity &&
                other.address < dst.address + dst.capacity)
                return GraphPacketStatus::InvalidBinding;
        }
    }
    if (header.runtime_offset % 64 != 0 || header.sm_offset % 64 != 0 || header.runtime_offset >= header.sm_offset ||
        header.sm_offset >= header.destinations[1].capacity)
        return GraphPacketStatus::InvalidBinding;
    uint64_t cursor = 0;
    uint32_t index = 0;
    for (uint32_t kind = 1; kind <= 3; ++kind) {
        const uint64_t capacity = header.destinations[kind].capacity;
        if (capacity == 0) continue;
        if (index >= header.region_count) return GraphPacketStatus::InvalidRegion;
        GraphImageRegion region{};
        std::memcpy(&region, bytes + prefix + sizeof(header) + index * sizeof(region), sizeof(region));
        const uint64_t aligned = (cursor + 63) & ~uint64_t{63};
        if (static_cast<uint32_t>(region.kind) != kind || region.reserved != 0 || region.source_offset != aligned ||
            region.destination_offset != 0 || region.bytes != capacity ||
            !graph_span_fits(region.source_offset, region.bytes, header.payload_bytes))
            return GraphPacketStatus::InvalidRegion;
        for (; cursor < aligned; ++cursor)
            if (bytes[prefix + payload_offset + cursor] != 0) return GraphPacketStatus::InvalidRegion;
        cursor = region.source_offset + region.bytes;
        ++index;
    }
    if (index != header.region_count || cursor != header.payload_bytes) return GraphPacketStatus::InvalidRegion;
    if (header.checksum != graph_packet_checksum(packet, size)) return GraphPacketStatus::InvalidChecksum;
    return GraphPacketStatus::Ok;
}

}  // namespace hbg
