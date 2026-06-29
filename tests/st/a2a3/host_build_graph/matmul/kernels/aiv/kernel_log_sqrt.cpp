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
 * Element-wise log then sqrt kernel (submit_task / Tensor* ABI)
 *
 * Implements: out[i] = sqrt(log(src[i])).  Both input and output are half
 * precision for matmul compatibility.  Single 128x128 tile.
 *
 * Args (Tensor*):
 *   args[0] = src (INPUT, half)
 *   args[1] = out (OUTPUT, half)
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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);

    __gm__ half *src = reinterpret_cast<__gm__ half *>(src_tensor->buffer.addr) + src_tensor->start_offset;
    __gm__ half *out = reinterpret_cast<__gm__ half *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int kTRows_ = 128;
    constexpr int kTCols_ = 128;
    constexpr int vRows = 128;
    constexpr int vCols = 128;

    using DynShapeDim5Half = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5Half = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalDataHalf = GlobalTensor<half, DynShapeDim5Half, DynStridDim5Half>;
    using TileDataHalf = Tile<TileType::Vec, half, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileDataHalf srcTile(vRows, vCols);
    TileDataHalf tmpTile(vRows, vCols);
    TileDataHalf dstTile(vRows, vCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalDataHalf srcGlobal(src);
    GlobalDataHalf dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOG(tmpTile, srcTile);
    TSQRT(dstTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);

    pipe_sync();
}
