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

#include "intrinsic.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int64_t *state = reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset;
    int64_t task_id = args[1];
    uint64_t producer_mask = static_cast<uint64_t>(args[2]);
    const int64_t kernel_repeats = args[3];
    for (int64_t repeat = 0; repeat < kernel_repeats; ++repeat) {
        for (int64_t producer = 0; producer < 64; ++producer) {
            if ((producer_mask & (UINT64_C(1) << producer)) != 0)
                dcci(&state[producer * 8], cache_line_t::SINGLE_CACHE_LINE);
        }
        dsb((mem_dsb_t)0);
        for (int64_t producer = 0; producer < 64; ++producer) {
            if ((producer_mask & (UINT64_C(1) << producer)) != 0 && state[producer * 8] != producer + 1) {
                state[task_id * 8] = -(producer + 1);
                dcci(&state[task_id * 8], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
                dsb((mem_dsb_t)0);
                return;
            }
        }
        state[task_id * 8] = task_id + 1;
        dcci(&state[task_id * 8], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        dsb((mem_dsb_t)0);
    }
}
