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
#include "platform_comm/ccu/ccu_native_abi.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *b = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *a = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *result = reinterpret_cast<__gm__ Tensor *>(args[2]);
    const uint64_t value = args[4];
    auto *b_value = reinterpret_cast<volatile __gm__ uint64_t *>(b->buffer.addr) + b->start_offset;
    auto *a_value = reinterpret_cast<volatile __gm__ uint64_t *>(a->buffer.addr) + a->start_offset;
    auto *out = reinterpret_cast<__gm__ uint64_t *>(result->buffer.addr) + result->start_offset;
    dcci(const_cast<__gm__ uint64_t *>(b_value), cache_line_t::ENTIRE_DATA_CACHE);
    dcci(const_cast<__gm__ uint64_t *>(a_value), cache_line_t::ENTIRE_DATA_CACHE);
    out[0] = (*b_value == value + 2 && *a_value == value + 1) ? value : UINT64_MAX;
    dcci(out, cache_line_t::ENTIRE_DATA_CACHE);
    dsb(DSB_DDR);
}
