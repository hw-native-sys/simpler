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
 * Standalone Element-wise Multiplication Kernel (AIV1 slot)
 *
 * Implements: out[i] = src0[i] * src1[i]
 * Tile size: 128 x 128
 *
 * Reads args[3..5] — for AIV1 slot in AIV_X2 tasks where AIV0 uses args[0..2].
 *
 * Args (Tensor*):
 *   args[3] = src0 (INPUT)  - 128 x 128
 *   args[4] = src1 (INPUT)  - 128 x 128
 *   args[5] = out (OUTPUT)  - 128 x 128
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <int ROWS, int COLS>
static __aicore__ void mul_impl(__gm__ float *src0, __gm__ float *src1, __gm__ float *out) {
    using DynShapeDim5 = Shape<1, 1, 1, ROWS, COLS>;
    using DynStridDim5 = Stride<1, 1, 1, COLS, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, ROWS, COLS, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(ROWS, COLS);
    TileData src1Tile(ROWS, COLS);
    TileData dstTile(ROWS, COLS);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src0_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *src1_tensor = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[5]);

    __gm__ float *src0 = reinterpret_cast<__gm__ float *>(src0_tensor->buffer.addr) + src0_tensor->start_offset;
    __gm__ float *src1 = reinterpret_cast<__gm__ float *>(src1_tensor->buffer.addr) + src1_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    mul_impl<128, 128>(src0, src1, out);
}
