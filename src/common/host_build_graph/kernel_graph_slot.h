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

class KernelExecutionState;

namespace hbg {

// Capture-external prepare: resources are frozen, but ReadyEnqueued is not required.
// Outputs own their bytes and remain unchanged on failure. The context/allocator
// lease spans sealing, enqueue, all graph uses and ordered registry detachment.
int seal_graph_execution_slot(
    const KernelExecutionState &context, int device_id, uint64_t generation, uint64_t runtime_binary_id,
    GraphSlotRegistration &out
);

struct GraphSlotPrepareOps {
    void *context{nullptr};
    // Deep-copies the record during this call and enqueues control work on the
    // dedicated AICPU stream. Device init/register/bind must complete in that
    // order before the enclosing prepare-tail event permits invocation work.
    int (*enqueue_registration)(void *, const GraphSlotRegistration &, void *aicpu_stream){nullptr};
};

int prepare_graph_execution_slot(
    const KernelExecutionState &context, int device_id, uint64_t generation, uint64_t runtime_binary_id,
    const GraphSlotPrepareOps &ops
);

}  // namespace hbg
