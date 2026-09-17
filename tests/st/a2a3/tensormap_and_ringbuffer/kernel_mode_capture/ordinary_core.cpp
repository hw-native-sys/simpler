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
#include "kernel_operator.h"

extern "C" __global__ __aicore__ void
ordinary_step_0_mix_aiv(__gm__ float *input, __gm__ float *output, uint64_t count, float scalar) {
    if (get_subblockid() != 0) return;
    const auto start = get_sys_cnt();
    while (get_sys_cnt() - start < 50000) {}
    for (uint64_t i = 0; i < count; i += 16) {
        dcci(input + i, SINGLE_CACHE_LINE);
        for (uint64_t j = i; j < count && j < i + 16; ++j)
            output[j] = -input[j] + scalar;
        dcci(output + i, SINGLE_CACHE_LINE, CACHELINE_OUT);
    }
}
