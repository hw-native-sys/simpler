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

#include "host_build_graph/kernel_graph_slot_wire.h"

namespace hbg {

// AICPU control operations. Init only targets newly allocated, exclusively held
// registry storage. It cannot reset a live context. Register never changes Ready
// content; duplicates must match byte for byte. These are never launch-blob paths.
GraphSlotStatus initialize_graph_slot_registry(
    GraphSlotRegistry *registry, int device_id, uint64_t generation, uint64_t runtime_binary_id
) noexcept;
GraphSlotStatus
register_graph_execution_slot(GraphSlotRegistry *registry, const void *registration, size_t bytes) noexcept;
GraphSlotStatus acquire_graph_execution_slot(
    const GraphSlotRegistry *registry, int device_id, uint64_t runtime_binary_id, GraphSlotRegistration &out
) noexcept;

// The resident DSO retains only this context-owned registry's address. Bind and
// detach are serialized control operations; detach follows graph destruction and
// external quiescence, before context.close releases the underlying allocation.
GraphSlotStatus
bind_graph_slot_registry(GraphSlotRegistry *registry, int device_id, uint64_t runtime_binary_id) noexcept;
bool detach_graph_slot_registry(GraphSlotRegistry *registry) noexcept;

// Read-only admission result. The packet and registry stay immutable/alive while
// the caller uses it. Image semantics, restoration and dispatch are separate.
struct GraphRestoreView {
    GraphSlotRegistration slot{};
    GraphPacketHeader graph{};
    const std::byte *payload{nullptr};
};

// Uses the independently latched registry, never an address supplied by packet.
// Expected device/binary identity comes from trusted AICPU initialization, not packet fields.
// On failure: no slot write, generation publication, or output modification.
GraphSlotStatus admit_graph_packet_for_restore(
    const void *packet, size_t bytes, int device_id, uint64_t runtime_binary_id, GraphRestoreView &out
) noexcept;

}  // namespace hbg
