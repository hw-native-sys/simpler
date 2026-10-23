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

#include "host_build_graph/kernel_graph_dispatch.h"

namespace hbg {

inline constexpr uint32_t GRAPH_PREPARED_DISPATCH_MAGIC = 0x50444748;  // HGDP
inline constexpr uint16_t GRAPH_PREPARED_DISPATCH_VERSION = 1;

// Capture-external control binds execution roles independently of each replay's
// HostArgs. A malformed replay must still reach the same collective CANCEL.
struct GraphPreparedContext {
    GraphSlotRegistration slot{};
    uint64_t runtime_address{0};
};
static_assert(sizeof(GraphPreparedContext) == 136);
static_assert(std::is_standard_layout_v<GraphPreparedContext> && std::is_trivially_copyable_v<GraphPreparedContext>);

// The context owner validates the image on the Host and publishes it by ordered
// DMA before this task. Only this fixed record travels in CANN HostArgs. Its
// checksum covers metadata, not the separately published image's contents.
struct GraphPreparedDispatch {
    SimplerKernelInvocationHeader invocation{};
    uint32_t magic{GRAPH_PREPARED_DISPATCH_MAGIC};
    uint16_t version{GRAPH_PREPARED_DISPATCH_VERSION};
    uint16_t header_bytes{sizeof(GraphPreparedDispatch)};
    uint64_t checksum{0};
    HbgCallableRegistration registration{};
    GraphPacketHeader graph{};
    GraphImageRegion regions[3]{};
};

static_assert(std::is_standard_layout_v<GraphPreparedDispatch>);
static_assert(std::is_trivially_copyable_v<GraphPreparedDispatch>);
static_assert(sizeof(GraphPreparedDispatch) == 432);
static_assert(offsetof(GraphPreparedDispatch, magic) == sizeof(SimplerKernelInvocationHeader));
static_assert(offsetof(GraphPreparedDispatch, checksum) == 40);
static_assert(offsetof(GraphPreparedDispatch, graph) == 136);
static_assert(offsetof(GraphPreparedDispatch, regions) == 336);

inline uint64_t graph_prepared_dispatch_checksum(const GraphPreparedDispatch &header) noexcept {
    constexpr size_t skip = offsetof(GraphPreparedDispatch, checksum);
    const auto *bytes = reinterpret_cast<const uint8_t *>(&header);
    return simpler::common::utils::fnv1a_64_append(
        simpler::common::utils::fnv1a_64(bytes, skip), bytes + skip + sizeof(header.checksum),
        sizeof(header) - skip - sizeof(header.checksum)
    );
}

inline bool valid_graph_prepared_dispatch(const GraphPreparedDispatch &header) noexcept {
    const auto &i = header.invocation;
    const auto &r = header.registration;
    const auto &g = header.graph;
    if (header.magic != GRAPH_PREPARED_DISPATCH_MAGIC || header.version != GRAPH_PREPARED_DISPATCH_VERSION ||
        header.header_bytes != sizeof(header) || !valid_hbg_callable_registration(r) || i.mode != SIMPLER_MODE_KERNEL ||
        i.callable_id != r.callable_id || i.tensor_count != r.tensor_count || i.scalar_count != r.scalar_count ||
        i.host_copy_tensor_count != 0 || i.reserved_ != 0 || i.payload_bytes != sizeof(header) - sizeof(i) ||
        g.magic != GRAPH_PACKET_MAGIC || g.version != GRAPH_PACKET_VERSION || g.header_bytes != sizeof(g) ||
        g.inline_payload_addr != 0 || g.checksum != 0 || g.slot_generation != r.context_generation ||
        g.device_id != r.device_id || g.runtime_binary_id != r.runtime_binary_id ||
        g.callable_hash != r.callable_hash || g.function_hash != r.function_hash || g.argument_hash == 0 ||
        g.reserved != 0 || g.region_count < 1 || g.region_count > 3 || g.task_window == 0 || g.task_window > 32768 ||
        g.total_tasks >= g.task_window || g.heap_bytes > g.destinations[0].capacity || g.runtime_offset % 64 != 0 ||
        g.sm_offset % 64 != 0 || g.runtime_offset >= g.sm_offset || g.sm_offset >= g.destinations[1].capacity)
        return false;
    const uint64_t table_end = sizeof(g) + g.region_count * sizeof(GraphImageRegion);
    if (g.payload_offset != ((table_end + 63) & ~uint64_t{63}) || g.total_bytes > UINT32_MAX - sizeof(i) ||
        g.payload_offset > g.total_bytes || g.payload_bytes != g.total_bytes - g.payload_offset)
        return false;
    for (size_t n = 0; n < 4; ++n) {
        const auto &d = g.destinations[n];
        if ((d.capacity == 0) != (d.address == 0) || (n < 2 && d.capacity == 0) || d.address % 1024 != 0 ||
            !graph_span_fits(d.address, d.capacity, UINT64_MAX))
            return false;
        for (size_t p = 0; p < n; ++p)
            if (graph_windows_overlap(d, g.destinations[p])) return false;
    }
    uint64_t cursor = 0;
    uint32_t previous_kind = 0;
    for (uint32_t n = 0; n < 3; ++n) {
        const auto &region = header.regions[n];
        if (n >= g.region_count) {
            const GraphImageRegion empty{};
            if (std::memcmp(&region, &empty, sizeof(region)) != 0) return false;
            continue;
        }
        const auto kind = static_cast<uint32_t>(region.kind);
        if (kind <= previous_kind || kind > 3 || (n == 0 && kind != 1) || region.reserved != 0 || region.bytes == 0 ||
            region.source_offset != ((cursor + 63) & ~uint64_t{63}) ||
            region.destination_offset != (kind == 1 ? g.runtime_offset : 0) ||
            !graph_span_fits(region.source_offset, region.bytes, g.payload_bytes) ||
            !graph_span_fits(region.destination_offset, region.bytes, g.destinations[kind].capacity) ||
            (kind == 1 && (region.bytes <= g.sm_offset - g.runtime_offset || region.bytes % 64 != 0)))
            return false;
        cursor = region.source_offset + region.bytes;
        previous_kind = kind;
    }
    return cursor == g.payload_bytes && header.checksum == graph_prepared_dispatch_checksum(header);
}

inline bool read_graph_prepared_dispatch(const void *data, size_t bytes, GraphPreparedDispatch &out) noexcept {
    if (data == nullptr || bytes < sizeof(GraphPreparedDispatch)) return false;
    GraphPreparedDispatch header{};
    std::memcpy(&header, data, sizeof(header));
    if (!valid_graph_prepared_dispatch(header)) return false;
    out = header;
    return true;
}

inline bool
graph_prepared_dispatch_matches_slot(const GraphPreparedDispatch &header, const GraphSlotRegistration &slot) noexcept {
    const auto &g = header.graph;
    if (!valid_graph_prepared_dispatch(header) || g.device_id != slot.device_id ||
        g.slot_generation != slot.slot_generation || g.runtime_binary_id != slot.runtime_binary_id ||
        g.total_bytes + sizeof(SimplerKernelInvocationHeader) > slot.max_packet_bytes)
        return false;
    for (size_t n = 0; n < 4; ++n)
        if (g.destinations[n].address != slot.destinations[n].address ||
            g.destinations[n].capacity != slot.destinations[n].capacity)
            return false;
    return true;
}

}  // namespace hbg
