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
#include "host_build_graph/kernel_graph_slot.h"

#include "host_build_graph/kernel_resource_requirements.h"

namespace hbg {

int seal_graph_execution_slot(
    const KernelExecutionState &context, int device_id, uint64_t generation, uint64_t runtime_binary_id,
    GraphSlotRegistration &out
) {
    if (runtime_binary_id == 0) return PTO_RUNTIME_ERR_INTERNAL;
    const uint64_t required[] = {0, 0, 0, 0, sizeof(GraphSlotRegistry)};
    KernelResourceBinding binding;
    const int rc = context.inspect_frozen_resources(
        device_id, generation, KernelResourcePlan::resource_schema, required, 5, binding
    );
    if (rc != 0) return rc;
    GraphSlotRegistration next{};
    next.magic = GRAPH_SLOT_MAGIC;
    next.version = GRAPH_SLOT_VERSION;
    next.bytes = sizeof(next);
    next.flags = GRAPH_SLOT_FROZEN_SERIAL;
    next.device_id = device_id;
    next.slot_generation = generation;
    next.runtime_binary_id = runtime_binary_id;
    for (size_t i = 0; i < 4; ++i)
        next.destinations[i] = {binding.regions[i].address, binding.regions[i].capacity};
    next.registry = {binding.regions[4].address, binding.regions[4].capacity};
    if (!graph_slot_packet_size(next.destinations, next.max_packet_bytes)) return PTO_RUNTIME_ERR_CAPACITY_EXCEEDED;
    next.checksum = graph_slot_checksum(next);
    if (!valid_graph_slot_registration(next)) return PTO_RUNTIME_ERR_INTERNAL;
    out = next;
    return 0;
}

int prepare_graph_execution_slot(
    const KernelExecutionState &context, int device_id, uint64_t generation, uint64_t runtime_binary_id,
    const GraphSlotPrepareOps &ops
) {
    if (ops.enqueue_registration == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    GraphSlotRegistration registration{};
    const int rc = seal_graph_execution_slot(context, device_id, generation, runtime_binary_id, registration);
    if (rc != 0) return rc;
    return ops.enqueue_registration(ops.context, registration, context.stream(KernelStreamKind::Aicpu));
}

}  // namespace hbg
