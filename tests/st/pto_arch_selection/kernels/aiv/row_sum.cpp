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

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    constexpr int kRows = 8;
    constexpr int kCols = 64;

    __gm__ Tensor *x_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ float *x = reinterpret_cast<__gm__ float *>(x_tensor->buffer.addr) + x_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    using GlobalInput = GlobalTensor<float, Shape<1, 1, 1, kRows, kCols>, Stride<1, 1, 1, kCols, 1>>;
    using GlobalOutput = GlobalTensor<float, Shape<1, 1, 1, kRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using InputTile = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, kRows, kCols>;
    using OutputTile = Tile<TileType::Vec, float, kRows, 1, BLayout::ColMajor, kRows, 1>;

    GlobalInput x_global(x);
    GlobalOutput out_global(out);
    InputTile x_tile;
    InputTile tmp_tile;
    OutputTile out_tile;

    TASSIGN(x_tile, 0);
    TASSIGN(out_tile, kRows * kCols * sizeof(float));
    TASSIGN(tmp_tile, 2 * kRows * kCols * sizeof(float));

    TLOAD(x_tile, x_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TROWSUM(out_tile, x_tile, tmp_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(out_global, out_tile);

    pipe_sync();
}
