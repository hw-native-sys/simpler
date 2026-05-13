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
/**
 * Elementwise affine compute: out = reduce_out * scale + bias.
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

static constexpr int kRows = 16;
static constexpr int kCols = 16;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *reduce_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *scale_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *bias_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);

    __gm__ float *reduce = reinterpret_cast<__gm__ float *>(reduce_tensor->buffer.addr) + reduce_tensor->start_offset;
    __gm__ float *scale = reinterpret_cast<__gm__ float *>(scale_tensor->buffer.addr) + scale_tensor->start_offset;
    __gm__ float *bias = reinterpret_cast<__gm__ float *>(bias_tensor->buffer.addr) + bias_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    using Shape2D = Shape<1, 1, 1, kRows, kCols>;
    using Stride2D = Stride<1, 1, 1, kCols, 1>;
    using Global = GlobalTensor<float, Shape2D, Stride2D, Layout::ND>;
    using TileData = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, -1, -1>;

    TileData reduce_tile(kRows, kCols);
    TileData scale_tile(kRows, kCols);
    TileData bias_tile(kRows, kCols);
    TASSIGN(reduce_tile, 0x0);
    TASSIGN(scale_tile, 0x10000);
    TASSIGN(bias_tile, 0x20000);

    Global reduce_g(reduce);
    Global scale_g(scale);
    Global bias_g(bias);
    Global out_g(out);

    TLOAD(reduce_tile, reduce_g);
    TLOAD(scale_tile, scale_g);
    TLOAD(bias_tile, bias_g);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMUL(reduce_tile, reduce_tile, scale_tile);
    TADD(reduce_tile, reduce_tile, bias_tile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(out_g, reduce_tile);
    pipe_sync();
}
