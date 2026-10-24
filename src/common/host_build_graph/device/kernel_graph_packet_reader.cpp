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
#include "host_build_graph/kernel_graph_packet_reader.h"
#include "host_build_graph/kernel_graph_packet_parser.h"

#include "aicpu/cache_maintenance.h"
#include "common/unified_log.h"
#include "host_build_graph/kernel_graph_slot_control.h"

namespace hbg {

GraphSlotStatus read_graph_execution_packet(
    const void *packet, size_t bytes, int device_id, uint64_t runtime_binary_id, GraphSlotRegistration &registration,
    ParsedGraphPacket &parsed, const GraphPacketReadOps &ops, GraphPacketAddress address_mode
) noexcept {
    const GraphSlotRegistry *registry = current_graph_slot_registry();
    if (registry == nullptr) return GraphSlotStatus::NotReady;
    GraphSlotRegistration snapshot{};
    const auto status = acquire_graph_execution_slot(registry, device_id, runtime_binary_id, snapshot);
    if (status != GraphSlotStatus::Ok) return status;
    if (packet == nullptr || bytes > snapshot.max_packet_bytes) return GraphSlotStatus::InvalidPacket;
    const GraphDestination source{reinterpret_cast<uintptr_t>(packet), bytes};
    if (!graph_span_fits(source.address, source.capacity, UINT64_MAX)) return GraphSlotStatus::InvalidPacket;
    if (graph_memory_regions_overlap(source, snapshot.registry)) return GraphSlotStatus::SourceOverlap;
    for (const auto &destination : snapshot.destinations)
        if (graph_memory_regions_overlap(source, destination)) return GraphSlotStatus::SourceOverlap;
    if (bytes < sizeof(SimplerKernelInvocationHeader) + sizeof(GraphPacketHeader))
        return GraphSlotStatus::InvalidPacket;
    if (ops.invalidate) {
        if (!ops.invalidate(ops.context, packet, bytes)) return GraphSlotStatus::SourceUnavailable;
    } else {
        cache_invalidate_range(packet, bytes);
    }
    ParsedGraphPacket next;
    const auto packet_status = parse_graph_packet(packet, bytes, address_mode, next);
    if (packet_status != GraphPacketStatus::Ok) {
        LOG_ERROR("HBG packet parsing failed: GraphPacketStatus=%u", static_cast<unsigned>(packet_status));
        return GraphSlotStatus::InvalidPacket;
    }
    registration = snapshot;
    parsed = next;
    return GraphSlotStatus::Ok;
}

}  // namespace hbg
