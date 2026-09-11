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

#include "worker/runtime_c_api.h"
#include "worker/pipeline_contract.h"

// Internal host-runtime hook, not a dlsym lifecycle API. Input is borrowed and
// immutable during the call; output is caller-exclusive and unchanged on error.
// No device resources are acquired or retained. Separate calls may run concurrently.
extern "C" int build_kernel_pipeline_contract_impl(const CallConfig *config, PipelineContract *out);

// Borrowed, call-local role bindings; neither stream is created or retained here.
// Caller, dedicated AICPU and hidden AICore are three distinct streams.
struct KernelStreamBinding {
    void *caller_stream{nullptr};
    void *aicpu_stream{nullptr};
    void *aicore_stream{nullptr};
};

inline int bind_kernel_stream_roles(
    const PipelineContract *contract, void *caller_stream, void *aicpu_stream, void *hidden_aicore_stream,
    KernelStreamBinding &out
) {
    if (!is_valid_pipeline_contract(contract, SIMPLER_MODE_KERNEL) || !has_serviceable_arena_topology(*contract) ||
        !has_serviceable_stream_topology(*contract) || caller_stream == nullptr || hidden_aicore_stream == nullptr ||
        aicpu_stream == nullptr || caller_stream == aicpu_stream || caller_stream == hidden_aicore_stream ||
        aicpu_stream == hidden_aicore_stream) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    out = {caller_stream, aicpu_stream, hidden_aicore_stream};
    return 0;
}
