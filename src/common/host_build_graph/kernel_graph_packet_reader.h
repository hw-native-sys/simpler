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

#include "host_build_graph/kernel_graph_slot_protocol.h"

namespace hbg {

struct GraphPacketReadOps {
    void *context{nullptr};
    // Covers the complete task-owned packet after trusted size/overlap checks,
    // before its first byte is parsed. Null selects device cache invalidation.
    bool (*invalidate)(void *, const void *, size_t){nullptr};
};

// Read boundary: obtain the independently registered resources and reject size
// and source overlap before invalidating or parsing any packet bytes. Outputs
// change only together on success. The owner holds the context execution lease
// and immutable packet storage through parsing, binding and execution preparation.
GraphSlotStatus read_graph_execution_packet(
    const void *packet, size_t bytes, int device_id, uint64_t runtime_binary_id, GraphSlotRegistration &registration,
    ParsedGraphPacket &parsed, const GraphPacketReadOps &ops = {},
    GraphPacketAddress address_mode = GraphPacketAddress::DeviceCopy
) noexcept;

}  // namespace hbg
