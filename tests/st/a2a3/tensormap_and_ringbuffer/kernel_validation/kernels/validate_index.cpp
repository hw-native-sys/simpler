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
#include <pto/pto-inst.hpp>
#include "kernel_error_api.h"
#include "tensor.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const auto *src = reinterpret_cast<__gm__ int32_t *>(input->buffer.addr);
    const int64_t index = src[input->start_offset];
    if (index < 0 || index >= args[2]) {
        auto ctx = get_async_ctx(args);
        report_kernel_error(ctx, KernelErrorCode::INDEX_OUT_OF_RANGE, KernelErrorDetail{index, 0, args[2], 0});
        return;
    }
    const auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *dst = reinterpret_cast<__gm__ int32_t *>(output->buffer.addr) + output->start_offset;
    *dst = static_cast<int32_t>(index);
    pto2::detail::defer_flush_range(dst, sizeof(*dst));
}
