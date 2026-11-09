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
#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "tensor.h"

namespace visibility_case {
inline __aicore__ __gm__ int32_t *data(__gm__ int64_t *args, uint32_t index) {
    auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
    return reinterpret_cast<__gm__ int32_t *>(tensor->buffer.addr) + tensor->start_offset;
}
inline __aicore__ void publish(__gm__ int32_t *data, uint32_t words) {
    pto2::detail::defer_flush_range(data, words * sizeof(int32_t));
    dsb((mem_dsb_t)0);
    pipe_barrier(PIPE_ALL);
}
inline __aicore__ void run(__gm__ int64_t *args) {
    auto *first = data(args, 0), *second = data(args, 1), *third = data(args, 2);
    const int32_t action = static_cast<int32_t>(args[3]);
    const int32_t tag = static_cast<int32_t>(args[4]);
    if (action == 0) {
        first[0] = 0;
        second[0] = 1 - (tag & 1);
        for (uint32_t i = 0; i < 32; ++i)
            third[i] = -17;
        publish(third, 32);
        publish(second, 1);
        publish(first, 1);
    } else if (action == 1) {
        const uint32_t block = get_block_idx(args);
        if (block >= 128) return;
        int32_t checksum = *reinterpret_cast<volatile __gm__ int32_t *>(first);
        for (uint32_t i = 0; i < 32; ++i)
            checksum += reinterpret_cast<volatile __gm__ int32_t *>(second)[i];
        third[128 + block * 32] = checksum;
        publish(third + 128 + block * 32, 1);
        if (block == 0) {
            third[64] = get_block_num(args);
            publish(third + 64, 1);
        }
    } else if (action == 2) {
        for (uint32_t i = 0; i < 32; ++i)
            third[i] = (tag >> 1) * 100 + static_cast<int32_t>(i);
        second[0] = tag & 1;
        publish(third, 32);
        publish(second, 1);
        first[0] = 1;
        publish(first, 1);
    } else if (action == 3) {
        for (uint32_t i = 0; i < 16; ++i)
            third[i] = first[i] + second[i];
        third[16] = 1;
        publish(third, 17);
    } else if (action == 4) {
        third[17] = 1;
        third[18] = first[0] + second[8];
        publish(third + 17, 2);
    }
}
}  // namespace visibility_case
