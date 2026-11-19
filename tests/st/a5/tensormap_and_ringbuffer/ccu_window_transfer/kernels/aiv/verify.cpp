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
#include "platform_comm/comm_context.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *result = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[3]);
    auto *source = reinterpret_cast<__gm__ uint64_t *>(comm->windowsIn[comm->rankId]);
    auto *out = reinterpret_cast<__gm__ uint64_t *>(result->buffer.addr) + result->start_offset;
    dcci(source, cache_line_t::ENTIRE_DATA_CACHE);
    for (uint32_t i = 0; i < 4096; ++i)
        out[i] = source[i];
    dcci(out, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}
