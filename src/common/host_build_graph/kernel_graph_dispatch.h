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

#include "host_build_graph/kernel_callable_registration.h"
#include "host_build_graph/kernel_graph_wire.h"
#include "host_build_graph/kernel_graph_slot_wire.h"
#include "utils/fnv1a_64.h"

namespace hbg {

inline constexpr uint32_t GRAPH_DISPATCH_MAGIC = 0x44474248;  // HBGD
inline constexpr uint16_t GRAPH_DISPATCH_VERSION = 1;

// CANN deep-copies this prepare-owned authority with the canonical v5 packet.
// The packet never supplies its own registration. The checksum detects damage,
// not authorization; only the context owner constructs this transport prefix.
struct GraphDispatchHeader {
    SimplerKernelInvocationHeader invocation{};
    uint32_t magic{GRAPH_DISPATCH_MAGIC};
    uint16_t version{GRAPH_DISPATCH_VERSION};
    uint16_t header_bytes{sizeof(GraphDispatchHeader)};
    uint64_t packet_bytes{0};
    uint64_t checksum{0};
    HbgCallableRegistration registration{};
};

static_assert(std::is_standard_layout_v<GraphDispatchHeader>);
static_assert(std::is_trivially_copyable_v<GraphDispatchHeader>);
static_assert(sizeof(GraphDispatchHeader) == 144);
static_assert(offsetof(GraphDispatchHeader, checksum) == 48);

inline uint64_t graph_dispatch_checksum(const GraphDispatchHeader &header) noexcept {
    constexpr size_t skip = offsetof(GraphDispatchHeader, checksum);
    const auto *bytes = reinterpret_cast<const uint8_t *>(&header);
    return simpler::common::utils::fnv1a_64_append(
        simpler::common::utils::fnv1a_64(bytes, skip), bytes + skip + sizeof(header.checksum),
        sizeof(header) - skip - sizeof(header.checksum)
    );
}

// Reads only the fixed prefix. The CANN entry ABI has no independent allocation
// length, so the producer must still provide the complete declared allocation.
// In particular, never hash a variable length before checking this prefix.
inline bool read_graph_dispatch_header(const void *data, size_t bytes, GraphDispatchHeader &out) noexcept {
    if (data == nullptr || bytes < sizeof(GraphDispatchHeader)) return false;
    GraphDispatchHeader header{};
    std::memcpy(&header, data, sizeof(header));
    const auto &i = header.invocation;
    const auto &r = header.registration;
    if (header.magic != GRAPH_DISPATCH_MAGIC || header.version != GRAPH_DISPATCH_VERSION ||
        header.header_bytes != sizeof(header) ||
        header.packet_bytes < sizeof(SimplerKernelInvocationHeader) + sizeof(GraphPacketHeader) ||
        header.packet_bytes > UINT32_MAX - sizeof(header) ||
        i.payload_bytes != sizeof(header) - sizeof(i) + header.packet_bytes || i.mode != SIMPLER_MODE_KERNEL ||
        i.host_copy_tensor_count != 0 || i.reserved_ != 0 || !valid_hbg_callable_registration(r) ||
        i.callable_id != r.callable_id || i.tensor_count != r.tensor_count || i.scalar_count != r.scalar_count ||
        header.checksum != graph_dispatch_checksum(header))
        return false;
    out = header;
    return true;
}

inline bool graph_dispatch_source_fits(const void *data, size_t bytes, const GraphSlotRegistration &slot) noexcept {
    const GraphDestination source{reinterpret_cast<uintptr_t>(data), bytes};
    if (data == nullptr || !graph_span_fits(source.address, source.capacity, UINT64_MAX) ||
        graph_windows_overlap(source, slot.registry))
        return false;
    for (const auto &destination : slot.destinations)
        if (graph_windows_overlap(source, destination)) return false;
    return true;
}

inline bool graph_dispatch_matches_packet(
    const GraphDispatchHeader &header, const SimplerKernelInvocationHeader &invocation, const GraphPacketHeader &graph
) noexcept {
    const auto &r = header.registration;
    return graph.total_bytes <= UINT32_MAX - sizeof(invocation) &&
           header.packet_bytes == sizeof(invocation) + graph.total_bytes &&
           invocation.payload_bytes == graph.total_bytes && invocation.mode == header.invocation.mode &&
           invocation.callable_id == r.callable_id && invocation.tensor_count == r.tensor_count &&
           invocation.scalar_count == r.scalar_count && invocation.host_copy_tensor_count == 0 &&
           invocation.reserved_ == 0 && graph.slot_generation == r.context_generation &&
           graph.device_id == r.device_id && graph.runtime_binary_id == r.runtime_binary_id &&
           graph.callable_hash == r.callable_hash && graph.function_hash == r.function_hash;
}

inline bool graph_dispatch_matches_slot(
    const void *data, const GraphDispatchHeader &header, const GraphSlotRegistration &slot
) noexcept {
    return header.packet_bytes <= slot.max_packet_bytes &&
           header.registration.context_generation == slot.slot_generation &&
           header.registration.device_id == slot.device_id &&
           header.registration.runtime_binary_id == slot.runtime_binary_id &&
           graph_dispatch_source_fits(data, sizeof(header) + static_cast<size_t>(header.packet_bytes), slot);
}

}  // namespace hbg
