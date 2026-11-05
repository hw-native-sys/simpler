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
#include "tensor.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *src = reinterpret_cast<__gm__ int32_t *>(input->buffer.addr) + input->start_offset;
    auto *dst = reinterpret_cast<__gm__ int32_t *>(output->buffer.addr) + output->start_offset;
    dcci(src, SINGLE_CACHE_LINE);
    if (src[0] == -12345 || src[0] == -12346) {
        // A bounded kernel outlives the fault scene's 200ms scheduler budget.
        const uint64_t start = get_sys_cnt();
        while (get_sys_cnt() - start < 50000000ULL) {}
    }
    for (int i = 0; i < 16; ++i)
        dst[i] = src[i] + static_cast<int32_t>(args[2]);
    dcci(dst, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
