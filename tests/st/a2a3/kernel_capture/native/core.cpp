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

// The three operands arrive as integer device addresses, matching CaptureArgs;
// this entry is where they become pointers.
extern "C" __global__ __aicore__ void
capture_step_0(uint64_t first_addr, uint64_t second_addr, uint64_t output_addr, int64_t addend) {
    __gm__ int64_t *first = reinterpret_cast<__gm__ int64_t *>(first_addr);
    __gm__ int64_t *second = reinterpret_cast<__gm__ int64_t *>(second_addr);
    __gm__ int64_t *output = reinterpret_cast<__gm__ int64_t *>(output_addr);
    // Delayed predecessor and hidden completion expose missing entry/exit edges.
    uint64_t start = get_sys_cnt();
    while (get_sys_cnt() - start < 50000) {}
    dcci(first, SINGLE_CACHE_LINE);
    int64_t value = *first + addend;
    if (second) {
        dcci(second, SINGLE_CACHE_LINE);
        value += *second;
    }
    *output = value;
    dcci(output, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
