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

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "async_kernel_api.h"
#include "tensor.h"  // NOLINT(build/include_subdir)

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *counter = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *values = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset;
    AsyncCtx ctx = get_async_ctx(args);
    const bool valid = async_ctx_is_deferred(ctx) && ctx.completion_count != nullptr;
    const uint32_t before = valid ? *ctx.completion_count : 0;
    CompletionToken token{counter->buffer.addr, 7, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    const bool registered = register_completion_condition(ctx, token);
    const bool advanced = valid && *ctx.completion_count == before + 1;
    pto2::detail::defer_flush(ctx);
    // Each independent task owns a complete cache line; failure cannot look like a successful registration.
    const float marker = valid && registered && advanced ? static_cast<float>(args[2]) : -1.0f;
    for (int i = 0; i < 16; ++i)
        values[i] = marker;
    pto2::detail::defer_flush_range(values, 16 * sizeof(float));
    dsb(DSB_DDR);
    pipe_barrier(PIPE_ALL);
}
