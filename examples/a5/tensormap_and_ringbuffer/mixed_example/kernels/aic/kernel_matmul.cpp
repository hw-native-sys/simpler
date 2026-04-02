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
 * Matrix Multiplication Kernel (Cube Core)
 *
 * Computes: C = A @ B (TILE x TILE x TILE matmul)
 * Uses TMATMUL instruction
 *
 * Args (Tensor*):
 *   args[0] = A (INPUT)  - TILE x TILE
 *   args[1] = B (INPUT)  - TILE x TILE
 *   args[2] = C (OUTPUT) - TILE x TILE
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

static __aicore__ inline int get_num_tiles(__gm__ Tensor *tensor, uint64_t tile_elems) {
    uint64_t total_elems = tensor->shapes[0];
    return static_cast<int>(total_elems / tile_elems);
}

template <int TILE>
static __aicore__ void matmul_impl(__gm__ float *input_a, __gm__ float *input_b, __gm__ float *output) {
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(float);
    constexpr int M = CeilAlign<int>(TILE, 16);
    constexpr int K = CeilAlign<int>(TILE, blockAlign);
    constexpr int N = CeilAlign<int>(TILE, blockAlign);

    using GlobalDataA = GlobalTensor<
        float, Shape<1, 1, 1, TILE, TILE>, pto::Stride<1 * TILE * TILE, 1 * TILE * TILE, TILE * TILE, TILE, 1>>;
    using GlobalDataB = GlobalTensor<
        float, Shape<1, 1, 1, TILE, TILE>, pto::Stride<1 * TILE * TILE, 1 * TILE * TILE, TILE * TILE, TILE, 1>>;
    using GlobalDataC = GlobalTensor<
        float, Shape<1, 1, 1, TILE, TILE>, pto::Stride<1 * TILE * TILE, 1 * TILE * TILE, TILE * TILE, TILE, 1>>;

    GlobalDataA src0Global(input_a);
    GlobalDataB src1Global(input_b);
    GlobalDataC dstGlobal(output);

    using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, TILE, TILE, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, TILE, TILE, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<float, M, K, TILE, TILE>;
    using RightTile = TileRight<float, K, N, TILE, TILE>;
    using AccTile = TileAcc<float, M, N, TILE, TILE>;

    TileMatA aMatTile;
    TileMatB bMatTile;
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
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_a = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *input_b = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *output = reinterpret_cast<__gm__ Tensor *>(args[2]);

    constexpr uint64_t TILE_ELEMS = 128 * 128;
    int num_tiles = get_num_tiles(input_a, TILE_ELEMS);

    __gm__ float *base_a = reinterpret_cast<__gm__ float *>(input_a->buffer.addr) + input_a->start_offset;
    __gm__ float *base_b = reinterpret_cast<__gm__ float *>(input_b->buffer.addr) + input_b->start_offset;
    __gm__ float *base_c = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        __gm__ float *a_ptr = base_a + (tile_idx * TILE_ELEMS);
        __gm__ float *b_ptr = base_b + (tile_idx * TILE_ELEMS);
        __gm__ float *c_ptr = base_c + (tile_idx * TILE_ELEMS);

        matmul_impl<128>(a_ptr, b_ptr, c_ptr);
    }
}
