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
#include "task_interface/kernel_invocation_validation.h"

namespace hbg {

// Validated resource binding; graph image semantics are checked during preparation.
// The caller retains the immutable packet and the context execution lease while
// consuming this borrowed view. Validation does not acquire resources or dispatch tasks.
struct ValidatedGraphExecutionBinding {
    ParsedGraphPacket packet;
    GraphSlotRegistration slot{};
};

// Compare an already parsed packet with the trusted snapshot from the read
// boundary and a live callable. Does not read packet bytes or the device registry.
// Callers must not fabricate the parsed view or derive trusted identity from it.
// The source and resource leases cover this comparison and its consumer.
// On failure, out is unchanged. Graph image semantics belong to preparation.
GraphSlotStatus validate_graph_execution_binding(
    const ParsedGraphPacket &packet, const GraphSlotRegistration &registration,
    const simpler::kernel::PreparedInvocationView &trusted_callable, ValidatedGraphExecutionBinding &out
) noexcept;

}  // namespace hbg
