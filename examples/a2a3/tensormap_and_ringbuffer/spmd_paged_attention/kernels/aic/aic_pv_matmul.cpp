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
// SPMD PV Matmul: pij(M, K) @ vj(K, N) -> oi_new(M, N)
//
// SPMD block_idx encodes (batch_idx, q_tile_idx).
// Each block computes one 16x16 matmul using paged V cache.
//
// Args:
//   args[0] = pij          Tensor* (spmd_blocks*Q_TILE, block_size) data_type
//   args[1] = value_cache  Tensor* (kv_total_rows, head_dim) bf16
//   args[2] = block_table  Tensor* (batch, max_blocks_per_req) int32
//   args[3] = context_lens Tensor* (batch,) int32
//   args[4] = oi_new       Tensor* (spmd_blocks*Q_TILE, head_dim) float32 [output]
//   args[5] = bn           scalar: current KV block index
//   args[6] = num_heads    scalar
//   args[7] = head_dim     scalar
//   args[8] = block_size   scalar
//   args[9] = max_num_blocks_per_req scalar
//   args[10] = q_loop      scalar

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

#include "intrinsic.h"

static constexpr int M = 16;
static constexpr int K = 16;
static constexpr int N = 16;

template <int TM, int TK, int TN>
static __aicore__ void pv_matmul_spmd(__gm__ bfloat16_t *pij_addr, __gm__ bfloat16_t *vj_addr, __gm__ float *oi_addr) {
    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, TM, TK>, Stride<TM * TK, TM * TK, TM * TK, TK, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, TK, TN>, Stride<TK * TN, TK * TN, TK * TN, TN, 1>>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, TM, TN>, Stride<TM * TN, TM * TN, TM * TN, TN, 1>>;

    GlobalA pijGlobal(pij_addr);
    GlobalB vjGlobal(vj_addr);
    GlobalOut oiGlobal(oi_addr);

    using TileMatA = Tile<TileType::Mat, bfloat16_t, TM, TK, BLayout::ColMajor, TM, TK, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, TK, TN, BLayout::ColMajor, TK, TN, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<bfloat16_t, TM, TK, TM, TK>;
    using RightTile = TileRight<bfloat16_t, TK, TN, TK, TN>;
    using AccTile = TileAcc<float, TM, TN, TM, TN>;

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

    TLOAD(aMatTile, pijGlobal);
    TLOAD(bMatTile, vjGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(oiGlobal, cTile);

    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *pij_t = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *value_cache_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *block_table_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *context_lens_t = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *oi_new_t = reinterpret_cast<__gm__ Tensor *>(args[4]);

    int64_t bn = static_cast<int64_t>(args[5]);
    int64_t num_heads = static_cast<int64_t>(args[6]);
    int64_t head_dim = static_cast<int64_t>(args[7]);
    int64_t block_size = static_cast<int64_t>(args[8]);
    int64_t max_blocks_per_req = static_cast<int64_t>(args[9]);
    int64_t q_loop = static_cast<int64_t>(args[10]);

    int32_t block_idx = get_block_idx(args);
    int64_t batch_idx = block_idx / q_loop;

    // Check if this batch has data at this KV block
    __gm__ int32_t *ctx_ptr =
        reinterpret_cast<__gm__ int32_t *>(context_lens_t->buffer.addr) + context_lens_t->start_offset;
    int64_t cur_seq = static_cast<int64_t>(ctx_ptr[batch_idx]);
    int64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

    // Output pointer for this block's oi_new slice
    __gm__ float *oi_addr =
        reinterpret_cast<__gm__ float *>(oi_new_t->buffer.addr) + oi_new_t->start_offset + block_idx * M * head_dim;

    if (bn >= bn_this_batch) {
        for (int i = 0; i < M * static_cast<int>(head_dim); i++) {
            oi_addr[i] = 0.0f;
        }
        return;
    }

    // Look up physical block index
    __gm__ int32_t *bt_ptr =
        reinterpret_cast<__gm__ int32_t *>(block_table_t->buffer.addr) + block_table_t->start_offset;
    int64_t phys_block = static_cast<int64_t>(bt_ptr[batch_idx * max_blocks_per_req + bn]);

    // pij offset: block_idx * Q_TILE * block_size
    int64_t pij_offset = block_idx * M * block_size;
    __gm__ bfloat16_t *pij_addr =
        reinterpret_cast<__gm__ bfloat16_t *>(pij_t->buffer.addr) + pij_t->start_offset + pij_offset;

    // Value offset: phys_block * block_size * head_dim
    int64_t v_offset = phys_block * block_size * head_dim;
    __gm__ bfloat16_t *vj_addr =
        reinterpret_cast<__gm__ bfloat16_t *>(value_cache_t->buffer.addr) + value_cache_t->start_offset + v_offset;

    pv_matmul_spmd<M, K, N>(pij_addr, vj_addr, oi_addr);
}
