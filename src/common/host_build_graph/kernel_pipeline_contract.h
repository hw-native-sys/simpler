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

#include "worker/pipeline_contract.h"

namespace simpler::hbg {

inline bool is_valid_kernel_pipeline_contract(const PipelineContract *contract) {
    if (!is_valid_pipeline_contract(contract, SIMPLER_MODE_KERNEL) || contract->pipeline_depth != 1 ||
        contract->resource_count != 4 || !has_serviceable_arena_topology(*contract) ||
        !has_serviceable_stream_topology(*contract)) {
        return false;
    }

    const auto *heap = find_pipeline_resource(*contract, PTO_PIPELINE_GM_HEAP);
    const auto *image = find_pipeline_resource(*contract, PTO_PIPELINE_RUNTIME_IMAGE);
    return heap != nullptr && image != nullptr && heap->resource_class == PTO_PIPELINE_HOST_PER_RUN &&
           image->resource_class == PTO_PIPELINE_HOST_PER_RUN &&
           find_pipeline_resource(*contract, PTO_PIPELINE_GM_SM) == nullptr &&
           find_pipeline_resource(*contract, PTO_PIPELINE_TASK_ARGS) == nullptr;
}

// Call-local roles. The framework owns caller_stream; the kernel context owns
// its dedicated non-hidden AICPU stream and hidden AICore stream.
struct KernelStreamBindings {
    void *caller_stream{nullptr};
    void *aicpu_stream{nullptr};
    void *aicore_stream{nullptr};
};

inline int bind_kernel_streams(
    const PipelineContract *contract, void *caller_stream, void *aicpu_stream, void *hidden_aicore_stream,
    KernelStreamBindings &out
) {
    if (!is_valid_kernel_pipeline_contract(contract) || caller_stream == nullptr || aicpu_stream == nullptr ||
        hidden_aicore_stream == nullptr || caller_stream == aicpu_stream || caller_stream == hidden_aicore_stream ||
        aicpu_stream == hidden_aicore_stream) {
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    }
    KernelStreamBindings next{caller_stream, aicpu_stream, hidden_aicore_stream};
    out = next;
    return 0;
}

}  // namespace simpler::hbg
