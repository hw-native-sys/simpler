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

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t kCount = 256;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *lhs_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *rhs_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ float *lhs = reinterpret_cast<__gm__ float *>(lhs_tensor->buffer.addr) + lhs_tensor->start_offset;
    __gm__ float *rhs = reinterpret_cast<__gm__ float *>(rhs_tensor->buffer.addr) + rhs_tensor->start_offset;
    __gm__ float *result = reinterpret_cast<__gm__ float *>(result_tensor->buffer.addr) + result_tensor->start_offset;

    using Shape = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, Shape, Stride, pto::Layout::ND>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 1, kCount, pto::BLayout::RowMajor, -1, -1>;

    Shape shape(1, 1, 1, 1, kCount);
    Stride stride(kCount, kCount, kCount, kCount, 1);
    Tile lhs_tile(1, kCount);
    Tile rhs_tile(1, kCount);
    Tile result_tile(1, kCount);
    TASSIGN(lhs_tile, 0x0);
    TASSIGN(rhs_tile, 0x10000);
    TASSIGN(result_tile, 0x20000);

    Global lhs_global(lhs, shape, stride);
    Global rhs_global(rhs, shape, stride);
    Global result_global(result, shape, stride);
    TLOAD(lhs_tile, lhs_global);
    TLOAD(rhs_tile, rhs_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(result_tile, lhs_tile, rhs_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(result_global, result_tile);
    pipe_barrier(PIPE_ALL);
}
