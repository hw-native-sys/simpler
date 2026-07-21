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
 * Matrix multiplication kernel (AIC, submit_task / Tensor* ABI)
 *
 * Implements: out = src0 @ src1.  Half precision inputs, float output.
 * Single 128x128 tile.  Flow: TLOAD -> TMOV -> TMATMUL -> TSTORE.
 *
 * Args (Tensor*):
 *   args[0] = src0 (INPUT, half)
 *   args[1] = src1 (INPUT, half)
 *   args[2] = out  (OUTPUT, float)
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

// Matrix dimensions
constexpr int validM = 128;
constexpr int validK = 128;
constexpr int validN = 128;

constexpr int blockAlign = 16;
constexpr int M = 128;
constexpr int K = 128;
constexpr int N = 128;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *src1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    __gm__ half *src0 = reinterpret_cast<__gm__ half *>(src0_tensor->buffer.addr) + src0_tensor->start_offset;
    __gm__ half *src1 = reinterpret_cast<__gm__ half *>(src1_tensor->buffer.addr) + src1_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    using GlobalDataSrc0 = GlobalTensor<
        half, Shape<1, 1, 1, validM, validK>, Stride<validM * validK, validM * validK, validM * validK, validK, 1>>;
    using GlobalDataSrc1 = GlobalTensor<
        half, Shape<1, 1, 1, validK, validN>, Stride<validK * validN, validK * validN, validK * validN, validN, 1>>;
    using GlobalDataOut = GlobalTensor<
        float, Shape<1, 1, 1, validM, validN>, Stride<validM * validN, validM * validN, validM * validN, validN, 1>>;

    GlobalDataSrc0 src0Global(src0);
    GlobalDataSrc1 src1Global(src1);
    GlobalDataOut dstGlobal(out);

    using TileMatAData = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, validM, validK, SLayout::RowMajor, 512>;
    using TileMatBData = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, validK, validN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<half, M, K, validM, validK>;
    using RightTile = TileRight<half, K, N, validK, validN>;
    using AccTile = TileAcc<float, M, N, validM, validN>;

    TileMatAData aMatTile;
    TileMatBData bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(dstGlobal, cTile);

    pipe_sync();
}
