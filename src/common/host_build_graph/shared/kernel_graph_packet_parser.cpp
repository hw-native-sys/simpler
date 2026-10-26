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
#include "host_build_graph/kernel_graph_packet_parser.h"

#include <cstring>

#include "host_build_graph/kernel_graph_packet_bounds.h"
#include "task_interface/kernel_invocation_validation.h"
#include "utils/fnv1a_64.h"

namespace hbg {

// The only excluded bytes are the runtime-patched address and this checksum.
// The envelope, binding, identity, region descriptors and padding are covered.
uint64_t graph_packet_checksum(const void *packet, size_t size) noexcept {
    constexpr size_t skip = sizeof(SimplerKernelInvocationHeader) + offsetof(GraphPacketHeader, inline_payload_addr);
    if (packet == nullptr || size < skip + 16) return 0;
    const auto *bytes = static_cast<const uint8_t *>(packet);
    auto hash = simpler::common::utils::fnv1a_64(bytes, skip);
    return simpler::common::utils::fnv1a_64_append(hash, bytes + skip + 16, size - skip - 16);
}

// Allocation-free framing parsing. No device destination is read or written.
// out is changed only after every format and checksum check succeeds.
// Registry identity checks and semantic validation of execution images precede
// execution separately; a checksum-valid packet alone never authorizes execution preparation.
GraphPacketStatus
parse_graph_packet(const void *packet, size_t size, GraphPacketAddress address_mode, ParsedGraphPacket &out) noexcept {
    constexpr size_t prefix = sizeof(SimplerKernelInvocationHeader);
    if (packet == nullptr || size < prefix + sizeof(GraphPacketHeader) || size > UINT32_MAX)
        return GraphPacketStatus::InvalidEnvelope;
    const auto *bytes = static_cast<const uint8_t *>(packet);
    SimplerKernelInvocationHeader invocation{};
    std::memcpy(&invocation, bytes, sizeof(invocation));
    if (invocation.mode != SIMPLER_MODE_KERNEL || invocation.callable_id < 0 ||
        invocation.callable_id >= MAX_REGISTERED_CALLABLE_IDS || invocation.generation == 0 ||
        !simpler::kernel::valid_invocation_counts(invocation.tensor_count, invocation.scalar_count) ||
        invocation.host_copy_tensor_count != 0 || invocation.reserved_ != 0 ||
        invocation.payload_bytes != size - prefix)
        return GraphPacketStatus::InvalidEnvelope;
    GraphPacketHeader header{};
    std::memcpy(&header, bytes + prefix, sizeof(header));
    if (header.magic != GRAPH_PACKET_MAGIC || header.version != GRAPH_PACKET_VERSION ||
        header.header_bytes != sizeof(header) || header.total_bytes != size - prefix || header.region_count < 1 ||
        header.region_count > 3 || header.slot_generation == 0 || header.callable_hash == 0 ||
        header.argument_hash == 0 || header.function_hash == 0 || header.task_window == 0 ||
        header.task_window > 32768 || header.total_tasks > header.task_window || header.reserved != 0 ||
        header.device_id < 0 || header.runtime_binary_id == 0 || header.heap_bytes > header.destinations[0].capacity)
        return GraphPacketStatus::InvalidHeader;
    const uint64_t table_end = sizeof(header) + header.region_count * sizeof(GraphImageRegion);
    const uint64_t payload_offset = (table_end + 63) & ~uint64_t{63};
    if (header.payload_offset != payload_offset || payload_offset > header.total_bytes ||
        header.payload_bytes != header.total_bytes - payload_offset)
        return GraphPacketStatus::InvalidHeader;
    const uint64_t base = reinterpret_cast<uintptr_t>(packet);
    if (!graph_span_fits(base, size, UINT64_MAX)) return GraphPacketStatus::InvalidHeader;
    const uint64_t expected_address = base + prefix + payload_offset;
    if (((address_mode == GraphPacketAddress::HostTemplate || address_mode == GraphPacketAddress::RelativeLaunch) &&
         header.inline_payload_addr != 0) ||
        (address_mode == GraphPacketAddress::DeviceCopy && header.inline_payload_addr != expected_address) ||
        (address_mode != GraphPacketAddress::HostTemplate && address_mode != GraphPacketAddress::DeviceCopy &&
         address_mode != GraphPacketAddress::RelativeLaunch))
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
    ParsedGraphPacket next{};
    next.invocation = invocation;
    next.graph = header;
    next.payload = reinterpret_cast<const std::byte *>(bytes + prefix + payload_offset);
    uint64_t cursor = 0;
    uint32_t previous_kind = 0;
    for (uint32_t index = 0; index < header.region_count; ++index) {
        GraphImageRegion region{};
        std::memcpy(&region, bytes + prefix + sizeof(header) + index * sizeof(region), sizeof(region));
        const uint32_t kind = static_cast<uint32_t>(region.kind);
        if (kind <= previous_kind || kind > 3 || (index == 0 && kind != 1)) return GraphPacketStatus::InvalidRegion;
        const uint64_t capacity = header.destinations[kind].capacity;
        const uint64_t aligned = (cursor + 63) & ~uint64_t{63};
        if (region.reserved != 0 || region.source_offset != aligned || (kind != 1 && region.destination_offset != 0) ||
            region.bytes == 0 || !graph_span_fits(region.destination_offset, region.bytes, capacity) ||
            (kind == 1 && (region.destination_offset > header.runtime_offset ||
                           region.destination_offset + region.bytes <= header.sm_offset || region.bytes % 64 != 0)) ||
            !graph_span_fits(region.source_offset, region.bytes, header.payload_bytes))
            return GraphPacketStatus::InvalidRegion;
        for (; cursor < aligned; ++cursor)
            if (bytes[prefix + payload_offset + cursor] != 0) return GraphPacketStatus::InvalidRegion;
        cursor = region.source_offset + region.bytes;
        previous_kind = kind;
        next.regions[kind] = region;
    }
    if (cursor != header.payload_bytes) return GraphPacketStatus::InvalidRegion;
    if (header.checksum != graph_packet_checksum(packet, size)) return GraphPacketStatus::InvalidChecksum;
    out = next;
    return GraphPacketStatus::Ok;
}

// Status-only callers share exactly the same parser and format rules.
GraphPacketStatus validate_graph_packet(const void *packet, size_t size, GraphPacketAddress address_mode) noexcept {
    ParsedGraphPacket parsed;
    return parse_graph_packet(packet, size, address_mode, parsed);
}

}  // namespace hbg
