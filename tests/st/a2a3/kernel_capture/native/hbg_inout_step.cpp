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
    auto *state = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *values = reinterpret_cast<__gm__ int32_t *>(state->buffer.addr) + state->start_offset;
    dcci(values, SINGLE_CACHE_LINE);
    for (int i = 0; i < 16; ++i)
        values[i] += static_cast<int32_t>(args[1]);
    dcci(values, SINGLE_CACHE_LINE, CACHELINE_OUT);
}
