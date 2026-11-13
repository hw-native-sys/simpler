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

#include "completion_context.h"
#include "kernel_error.h"

static_assert(static_cast<int32_t>(KernelErrorCode::INVALID_OUTPUT) == SIMPLER_ERROR_KERNEL_INVALID_OUTPUT);
static_assert(static_cast<int32_t>(KernelErrorCode::INDEX_OUT_OF_RANGE) == SIMPLER_ERROR_KERNEL_INDEX_OUT_OF_RANGE);
static_assert(static_cast<int32_t>(KernelErrorCode::NON_FINITE_VALUE) == SIMPLER_ERROR_KERNEL_NON_FINITE_VALUE);

// The caller must return from kernel_entry after reporting. This operation does
// not cancel DMA requests or unblock a cooperating MIX lane.
inline __aicore__ bool report_kernel_error(AsyncCtx &ctx, KernelErrorCode code) {
    if (!ctx.task_token.is_valid() || ctx.completion_count == nullptr || ctx.completion_error_code == nullptr ||
        !is_kernel_error(static_cast<int32_t>(code))) {
        return false;
    }
    if (*ctx.completion_error_code != SIMPLER_ERROR_NONE) return false;
    *ctx.completion_error_code = static_cast<int32_t>(code);
    pto2::detail::defer_flush(ctx);
    return true;
}
