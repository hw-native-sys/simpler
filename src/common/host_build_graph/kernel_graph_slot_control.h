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

// AICPU control operations. Init only targets newly allocated, exclusively held
// registry storage. It cannot reset a live context. Register never changes Ready
// content; duplicates must match byte for byte. These are never launch-blob paths.
// This module controls registration and binding, not allocation or memory release.
// The context owner serializes install/bind/detach/poison with preparation,
// execution readers and allocation release. The atomic publication fields do
// not provide an allocation lease or make the whole install transaction atomic.
// Control records come only from that trusted owner: bounds/checksum validation
// cannot prove that a supplied address is mapped or still owned by the context.
// A new context must use a fresh generation, even when its allocation reuses an
// address. Detach alone never authorizes resetting the same generation.
GraphSlotStatus initialize_graph_slot_registry(
    GraphSlotRegistry *registry, int device_id, uint64_t generation, uint64_t runtime_binary_id
) noexcept;
GraphSlotStatus
register_graph_execution_slot(GraphSlotRegistry *registry, const void *registration, size_t bytes) noexcept;
// Kernel-mode control transaction. It initializes an unbound fresh-generation registry,
// publishes one immutable registration, then binds that registry for launch.
// An identical repeat is idempotent; any conflicting live registration fails
// before changing either registry.
// Reattaching the same generation preserves all execution state, including poison.
GraphSlotStatus install_graph_execution_slot(const void *registration, size_t bytes) noexcept;
GraphSlotStatus acquire_graph_execution_slot(
    const GraphSlotRegistry *registry, int device_id, uint64_t runtime_binary_id, GraphSlotRegistration &out
) noexcept;

// The resident DSO retains only this context-owned registry's address. Bind and
// detach are serialized control operations; detach follows graph destruction and
// external quiescence, before context.close releases the underlying allocation.
GraphSlotStatus
bind_graph_slot_registry(GraphSlotRegistry *registry, int device_id, uint64_t runtime_binary_id) noexcept;
bool detach_graph_slot_registry(GraphSlotRegistry *registry) noexcept;
// Borrowed only while the invocation owner holds the context execution lease.
GraphSlotRegistry *current_graph_slot_registry() noexcept;

// Serialized with control/preparation/retirement by the invocation owner. Terminal:
// no register, bind, admission or retirement may make this storage usable again.
GraphSlotStatus poison_graph_execution_slot(GraphSlotRegistry *registry) noexcept;

}  // namespace hbg

// Dedicated HBG kernel-mode AICPU entry. Program-mode symbol discovery does
// not report this symbol. CANN HostArgs may be unaligned, so the implementation
// copies the fixed-size record before inspecting it.
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_register_execution_slot(void *arg);
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_l1_hbg_detach_execution_slot(void *arg);
