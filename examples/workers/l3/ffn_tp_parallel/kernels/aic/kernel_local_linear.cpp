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
 * FFN local linear kernel (AIC) — partial_local = x_shard @ w_shard.
 *
 * Stage 1 of the 2-stage FFN tensor-parallel demo: each rank computes its
 * local matmul into per-rank device memory (partial_local).  Stage 2
 * (kernel_allreduce_sum.cpp) then sums partial_local across ranks into y.
 *
 * args layout (ChipStorageTaskArgs — see ffn_local_orch.cpp):
 *   tensor(0) = x_shard       INPUT           (M x K, host-backed)
 *   tensor(1) = w_shard       INPUT           (K x N, host-backed)
 *   tensor(2) = partial_local OUTPUT_EXISTING (M x N, per-rank device mem)
 */

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <cstdint>

#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>

#include "tensor.h"

using namespace pto;

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

static __aicore__ void local_linear_impl(__gm__ Tensor *x_tensor, __gm__ Tensor *w_tensor, __gm__ Tensor *out_tensor) {
    __gm__ float *x_ptr = reinterpret_cast<__gm__ float *>(x_tensor->buffer.addr) + x_tensor->start_offset;
    __gm__ float *w_ptr = reinterpret_cast<__gm__ float *>(w_tensor->buffer.addr) + w_tensor->start_offset;
    __gm__ float *out_ptr = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int TILE = 64;
    constexpr int block_align = C0_SIZE_BYTE / sizeof(float);
    constexpr int M = CeilAlign<int>(TILE, 16);
    constexpr int K = CeilAlign<int>(TILE, block_align);
    constexpr int N = CeilAlign<int>(TILE, block_align);

    using GlobalData =
        GlobalTensor<float, Shape<1, 1, 1, TILE, TILE>, Stride<1 * TILE * TILE, 1 * TILE * TILE, TILE * TILE, TILE, 1>>;
    using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, TILE, TILE, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, TILE, TILE, SLayout::RowMajor, 512>;
    using LeftTile = TileLeft<float, M, K, TILE, TILE>;
    using RightTile = TileRight<float, K, N, TILE, TILE>;
    using AccTile = TileAcc<float, M, N, TILE, TILE>;

    GlobalData x_global(x_ptr);
    GlobalData w_global(w_ptr);
    GlobalData out_global(out_ptr);

    TileMatA x_mat;
    TileMatB w_mat;
    TASSIGN(x_mat, 0x0);
    TASSIGN(w_mat, 0x20000);

    LeftTile x_tile;
    RightTile w_tile;
    AccTile out_tile;
    TASSIGN(x_tile, 0x0);
    TASSIGN(w_tile, 0x0);
    TASSIGN(out_tile, 0x0);

    TLOAD(x_mat, x_global);
    TLOAD(w_mat, w_global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(x_tile, x_mat);
    TMOV(w_tile, w_mat);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(out_tile, x_tile, w_tile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(out_global, out_tile);

    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *x_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *w_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    local_linear_impl(x_tensor, w_tensor, out_tensor);
}
