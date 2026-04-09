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
// QK Matmul Kernel: qi(M, K) @ kj.T(K, N) -> sij(M, N)
//
// Fixed tile size: (16, 16) @ (16, 16).T -> (16, 16)
//
// kj is stored as (N, K) = (block_size, head_dim) in row-major memory.
// This is equivalent to (K, N) in column-major (DN) layout.
// Using DN GlobalB + RowMajor/ColMajor TileMatB to handle the transposed B pattern.

#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static __aicore__ void qk_matmul_impl(__gm__ uint8_t *qi_raw, __gm__ uint8_t *kj_raw, __gm__ uint8_t *sij_raw) {
    constexpr int M = 16, K = 16, N = 16;

    __gm__ bfloat16_t *qi = reinterpret_cast<__gm__ bfloat16_t *>(qi_raw);
    __gm__ bfloat16_t *kj = reinterpret_cast<__gm__ bfloat16_t *>(kj_raw);
    __gm__ float *sij = reinterpret_cast<__gm__ float *>(sij_raw);

    // qi (M, K) bf16 in ND (row-major) layout
    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    // kj stored as (N, K) row-major = (K, N) column-major -> DN layout
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    GlobalA qiGlobal(qi);
    GlobalB kjGlobal(kj);
    GlobalOut sijGlobal(sij);

    // L1 Mat tiles: A is standard ND, B uses transposed-B pattern (RowMajor/ColMajor)
    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

    // L0 tiles
    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

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

    // Load qi and kj to L1
    TLOAD(aMatTile, qiGlobal);
    TLOAD(bMatTile, kjGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    // Move to L0A/L0B
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // Single matmul: (M,K) x (K,N) -> (M,N)
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(sijGlobal, cTile);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ uint8_t *qi = reinterpret_cast<__gm__ uint8_t *>(args[0]);
    __gm__ uint8_t *kj = reinterpret_cast<__gm__ uint8_t *>(args[1]);
    __gm__ uint8_t *sij = reinterpret_cast<__gm__ uint8_t *>(args[2]);

    qk_matmul_impl(qi, kj, sij);
}
