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

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *mailbox_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *mailbox =
        reinterpret_cast<__gm__ float *>(mailbox_tensor->buffer.addr) + mailbox_tensor->start_offset;
    __gm__ float *result = reinterpret_cast<__gm__ float *>(result_tensor->buffer.addr) + result_tensor->start_offset;

    uint32_t n = static_cast<uint32_t>(result_tensor->shapes[0]);
    for (uint32_t i = 0; i < n; ++i) {
        result[i] = mailbox[i];
    }
}
