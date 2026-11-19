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

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ int64_t *state = reinterpret_cast<__gm__ int64_t *>(tensor->buffer.addr) + tensor->start_offset;
    const int64_t base = args[1];
    const int64_t token = args[2];
    const int64_t local = args[3];
    const int64_t begin = args[4];
    const int64_t count = args[5];
    for (int64_t i = begin; i < begin + count; ++i)
        dcci(&state[(base + i) * 8], cache_line_t::SINGLE_CACHE_LINE);
    dsb((mem_dsb_t)0);
    int64_t value = token + local;
    for (int64_t i = begin; i < begin + count; ++i) {
        if (state[(base + i) * 8] != token + i) {
            value = -(i + 1);
            break;
        }
    }
    state[(base + local) * 8] = value;
    dcci(&state[(base + local) * 8], cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}
