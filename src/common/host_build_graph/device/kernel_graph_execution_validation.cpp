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
#include "host_build_graph/kernel_graph_execution_validation.h"

namespace hbg {

GraphSlotStatus validate_graph_execution_binding(
    const ParsedGraphPacket &packet, const GraphSlotRegistration &registration,
    const simpler::kernel::PreparedInvocationView &trusted_callable, ValidatedGraphExecutionBinding &out
) noexcept {
    if (!simpler::kernel::valid_prepared_invocation(trusted_callable)) return GraphSlotStatus::InvalidPacket;
    const auto &invocation = packet.invocation;
    if (invocation.callable_id != trusted_callable.callable_id || invocation.generation != trusted_callable.generation)
        return GraphSlotStatus::CallableMismatch;
    if (invocation.tensor_count != trusted_callable.tensor_count ||
        invocation.scalar_count != trusted_callable.scalar_count)
        return GraphSlotStatus::InvalidPacket;
    const auto &header = packet.graph;
    if (header.device_id != registration.device_id) return GraphSlotStatus::DeviceMismatch;
    if (header.slot_generation != registration.slot_generation) return GraphSlotStatus::GenerationMismatch;
    if (header.runtime_binary_id != registration.runtime_binary_id) return GraphSlotStatus::BinaryMismatch;
    for (size_t i = 0; i < 4; ++i)
        if (header.destinations[i].address != registration.destinations[i].address ||
            header.destinations[i].capacity != registration.destinations[i].capacity)
            return GraphSlotStatus::BindingMismatch;
    out = {packet, registration};
    return GraphSlotStatus::Ok;
}

}  // namespace hbg
