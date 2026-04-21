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
 * Tile-based Element-wise Addition Kernel (Vector Core)
 *
 * Computes: output = input_a + input_b (64x64 tile addition)
 * Uses TADD instruction
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ float *input_a = reinterpret_cast<__gm__ float *>(args[0]);
    __gm__ float *input_b = reinterpret_cast<__gm__ float *>(args[1]);
    __gm__ float *output = reinterpret_cast<__gm__ float *>(args[2]);

    constexpr int TILE = 64;

    using DynShapeDim5 = Shape<1, 1, 1, TILE, TILE>;
    using DynStridDim5 = Stride<1, 1, 1, TILE, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, TILE, TILE, BLayout::RowMajor, -1, -1>;

    TileData aTile(TILE, TILE);
    TileData bTile(TILE, TILE);
    TileData outTile(TILE, TILE);
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x10000);
    TASSIGN(outTile, 0x20000);

    GlobalData aGlobal(input_a);
    GlobalData bGlobal(input_b);
    GlobalData outGlobal(output);

    TLOAD(aTile, aGlobal);
    TLOAD(bTile, bGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(outTile, aTile, bTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(outGlobal, outTile);

    pipe_sync();
}
