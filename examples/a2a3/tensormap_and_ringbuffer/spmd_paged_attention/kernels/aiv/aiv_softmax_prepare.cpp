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
// SPMD Softmax Preparation Kernel (AIV) with partial block masking and
// dual-vector subvector split.
//
// SPMD block_idx encodes (batch_idx, q_tile_idx).
// The two AIV lanes in a cluster split the Q_TILE=16 rows 8/8 via
// get_sub_block_id(): AIV0 handles rows [0, 8), AIV1 handles rows [8, 16).
//
// Computes (per sub-slice of SUB_M=8 rows):
//   sij_masked = pad(sij, valid_len, -inf)
//   sij_scale = sij_masked * scale
//   mij = row_max(sij_scale)        -> (SUB_M, 1)
//   pij = exp(sij_scale - mij)      -> (SUB_M, N)
//   lij = row_sum(pij)              -> (SUB_M, 1)
//
// Args:
//   args[0] = sij          Tensor* (spmd_blocks*Q_TILE, block_size) float32 [input]
//   args[1] = context_lens Tensor* (batch,) int32
//   args[2] = pij          Tensor* (spmd_blocks*Q_TILE, block_size) bf16 [output]
//   args[3] = mij          Tensor* (spmd_blocks*Q_TILE,) float32 [output]
//   args[4] = lij          Tensor* (spmd_blocks*Q_TILE,) float32 [output]
//   args[5] = scale_value  scalar (as float bits in uint64)
//   args[6] = bn           scalar: current KV block index
//   args[7] = block_size   scalar
//   args[8] = q_loop       scalar

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

static constexpr int M = 16;      // Full Q tile rows (shared between both AIVs)
static constexpr int SUB_M = 8;   // Rows per AIV lane (M / 2)
static constexpr int N = 16;      // block_size

template <int TM, int TN>
static __aicore__ void softmax_prepare_spmd(
    __gm__ float *sij_addr, float scale_value, uint64_t valid_len, __gm__ bfloat16_t *pij_addr,
    __gm__ float *mij_addr, __gm__ float *lij_addr
) {
    constexpr int kAlignedRows = ((TM * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, TM, TN>, Stride<1, 1, 1, TN, 1>>;
    using GlobalDataMxN_bf16 = GlobalTensor<bfloat16_t, Shape<1, 1, 1, TM, TN>, Stride<1, 1, 1, TN, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataMxN sijGlobal(sij_addr);
    GlobalDataMxN_bf16 pijGlobal(pij_addr);
    GlobalScalarDN mijGlobal(mij_addr);
    GlobalScalarDN lijGlobal(lij_addr);

    using TileSijDyn = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, -1>;
    using TileSijPad =
        Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN, SLayout::NoneBox, 512, PadValue::Min>;

    using TileVecMxN = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;
    using TileVecMxN_bf16 = Tile<TileType::Vec, bfloat16_t, TM, TN, BLayout::RowMajor, TM, TN>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, TM, 1>;

    TileVecMxN sijTile;
    TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_bf16 pijBf16Tile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijDynTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, TM * TN * sizeof(float));
    TASSIGN(tmpTile, 2 * TM * TN * sizeof(float));
    TASSIGN(maxTile, 3 * TM * TN * sizeof(float));
    TASSIGN(sumTile, 3 * TM * TN * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijBf16Tile, 3 * TM * TN * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    TLOAD(sijTile, sijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TFILLPAD_INPLACE(sijPadTile, sijDynTile);
    pipe_barrier(PIPE_V);

    TMULS(sijTile, sijTile, scale_value);
    pipe_barrier(PIPE_V);
    TROWMAX(maxTile, sijTile, tmpTile);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(pijTile, sijTile, maxTile);
    pipe_barrier(PIPE_V);
    TEXP(pijTile, pijTile);
    TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
    TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);
    TROWSUM(sumTile, pijTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(mijGlobal, maxTile);
    TSTORE(lijGlobal, sumTile);
    TSTORE(pijGlobal, pijBf16Tile);

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *sij_t = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *context_lens_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *pij_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *mij_t = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *lij_t = reinterpret_cast<__gm__ Tensor *>(args[4]);
    float scale_value = from_u64<float>(static_cast<uint64_t>(args[5]));
    int64_t bn = static_cast<int64_t>(args[6]);
    int64_t block_size = static_cast<int64_t>(args[7]);
    int64_t q_loop = static_cast<int64_t>(args[8]);

    int32_t block_idx = get_block_idx(args);
    int32_t sub_block_id = get_sub_block_id(args);  // 0 = AIV0 (rows 0..7), 1 = AIV1 (rows 8..15)
    int64_t batch_idx = block_idx / q_loop;

    // Compute valid_len for this block: how many columns of sij are valid
    __gm__ int32_t *ctx_ptr =
        reinterpret_cast<__gm__ int32_t *>(context_lens_t->buffer.addr) + context_lens_t->start_offset;
    int64_t cur_seq = static_cast<int64_t>(ctx_ptr[batch_idx]);
    int64_t remaining = cur_seq - bn * block_size;
    uint64_t valid_len;
    if (remaining <= 0) {
        valid_len = 0;
    } else if (remaining >= block_size) {
        valid_len = static_cast<uint64_t>(block_size);
    } else {
        valid_len = static_cast<uint64_t>(remaining);
    }

    // Row offset for this AIV lane within the block_idx's Q_TILE slice
    int64_t row_offset = sub_block_id * SUB_M;

    // Pointers into this block's SUB_M-row sub-slice of the flat tensors
    int64_t data_row_offset = block_idx * M + row_offset;
    __gm__ float *sij_addr =
        reinterpret_cast<__gm__ float *>(sij_t->buffer.addr) + sij_t->start_offset + data_row_offset * block_size;
    __gm__ bfloat16_t *pij_addr =
        reinterpret_cast<__gm__ bfloat16_t *>(pij_t->buffer.addr) + pij_t->start_offset + data_row_offset * block_size;

    // Scalar layout: full M=16 rows pack to kAlignedRowsFull=16 floats per block_idx;
    // each AIV lane owns kAlignedRowsSub=8 contiguous floats inside that slab.
    constexpr int kAlignedRowsFull = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));
    constexpr int kAlignedRowsSub = ((SUB_M * sizeof(float) + 31) / 32) * (32 / sizeof(float));
    int64_t scalar_offset = block_idx * kAlignedRowsFull + sub_block_id * kAlignedRowsSub;
    __gm__ float *mij_addr =
        reinterpret_cast<__gm__ float *>(mij_t->buffer.addr) + mij_t->start_offset + scalar_offset;
    __gm__ float *lij_addr =
        reinterpret_cast<__gm__ float *>(lij_t->buffer.addr) + lij_t->start_offset + scalar_offset;

    if (valid_len == 0) {
        // No valid KV data — emit neutral values so online_update is a no-op:
        // mij = -1e30 (very negative so beta = exp(mij - mi_new) ≈ 0)
        // lij = 0 (no contribution to normalizer)
        // pij = 0 (no attention weight)
        for (int i = 0; i < kAlignedRowsSub; i++) {
            mij_addr[i] = -1e30f;
            lij_addr[i] = 0.0f;
        }
        for (int i = 0; i < SUB_M * static_cast<int>(block_size); i++) {
            pij_addr[i] = static_cast<bfloat16_t>(0.0f);
        }
        return;
    }

    softmax_prepare_spmd<SUB_M, N>(sij_addr, scale_value, valid_len, pij_addr, mij_addr, lij_addr);
}
