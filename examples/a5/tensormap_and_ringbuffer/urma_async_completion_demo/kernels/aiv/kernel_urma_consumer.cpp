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

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__
#endif

#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *tget_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *tput_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *tget_result_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *tput_result_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);

    __gm__ float *tget = reinterpret_cast<__gm__ float *>(tget_tensor->buffer.addr) + tget_tensor->start_offset;
    __gm__ float *tput = reinterpret_cast<__gm__ float *>(tput_tensor->buffer.addr) + tput_tensor->start_offset;
    __gm__ float *tget_result =
        reinterpret_cast<__gm__ float *>(tget_result_tensor->buffer.addr) + tget_result_tensor->start_offset;
    __gm__ float *tput_result =
        reinterpret_cast<__gm__ float *>(tput_result_tensor->buffer.addr) + tput_result_tensor->start_offset;

    uint32_t n = static_cast<uint32_t>(tget_result_tensor->shapes[0]);
    for (uint32_t i = 0; i < n; ++i) {
        tget_result[i] = tget[i];
        tput_result[i] = tput[i];
    }
}
