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
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128).T -> (16, 128)
//   Case2: (64, 128) @ (128,  64).T -> (64,  64)
//
// kj is stored as (N, K) = (block_size, head_dim) in row-major memory.
// This is equivalent to (K, N) in column-major (DN) layout.
// Using DN GlobalB + RowMajor/ColMajor TileMatB to handle the transposed B pattern.

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

template <int M, int K, int N>
static __aicore__ void qk_matmul_impl(__gm__ Tensor *qi, __gm__ Tensor *kj, __gm__ Tensor *sij) {
    __gm__ bfloat16_t *qi_addr = reinterpret_cast<__gm__ bfloat16_t *>(qi->buffer.addr);
    __gm__ bfloat16_t *kj_addr = reinterpret_cast<__gm__ bfloat16_t *>(kj->buffer.addr);
    __gm__ float *sij_addr = reinterpret_cast<__gm__ float *>(sij->buffer.addr);

    // qi (M, K) bf16 in ND (row-major) layout
    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    // kj stored as (N, K) row-major = (K, N) column-major -> DN layout
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    GlobalA qiGlobal(qi_addr + qi->start_offset);
    GlobalB kjGlobal(kj_addr + kj->start_offset);
    GlobalOut sijGlobal(sij_addr + sij->start_offset);

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

    // Load A and B to L1 with separate events for pipeline overlap
    TLOAD(aMatTile, qiGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);  // A load done
    TLOAD(bMatTile, kjGlobal);
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);  // B load done

    // Move A to L0A as soon as A load completes (B may still be loading)
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    TMOV(aTile, aMatTile);
    // Move B to L0B after B load completes
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // Matmul
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(sijGlobal, cTile);

    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *qi = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *kj = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *sij = reinterpret_cast<__gm__ Tensor *>(args[2]);
    uint64_t q_tile_size = static_cast<uint64_t>(qi->shapes[0]);
    // args[4] = head_dim (128), args[5] = block_size

    if (q_tile_size == 16 && qi->shapes[1] <= 16) {
        qk_matmul_impl<16, 16, 16>(qi, kj, sij);
    } else if (q_tile_size == 16) {
        qk_matmul_impl<16, 128, 128>(qi, kj, sij);
    } else {
        qk_matmul_impl<64, 128, 64>(qi, kj, sij);
    }
}
