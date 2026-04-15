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
// SPMD Two-Pass Softmax Kernel (AIV) for n_blocks tiles with dual-vector split
//
// SPMD block_idx encodes (batch_idx, q_tile_idx).
// The two AIV lanes in a cluster split the Q_TILE rows via get_sub_block_id():
//   AIV0 (sub_block_id=0) handles rows [0, SUB_M)
//   AIV1 (sub_block_id=1) handles rows [SUB_M, Q_TILE)
//
// Memory layout: QK kernel writes packed (Q_TILE, block_size) tiles contiguously.
// Within each tile, AIV0 processes the first SUB_M rows and AIV1 the rest.
// Tile stride between consecutive tiles is Q_TILE * block_size elements.
//
// Two-pass softmax (same algorithm as paged_attention_unroll):
// Pass 1: Find global m = scale * max over all blocks of rowmax(S_i)
// Pass 2: Compute P_i = exp(S_i * scale - m) -> bf16, accumulate l = rowsum(P_i)
//
// Case1: SUB_M=8, TN=128 (Q_TILE=16, block_size=128)
// Case2: SUB_M=32, TN=64  (Q_TILE=64, block_size=64)
//
// Args:
//   args[0] = sij          Tensor* (spmd_blocks*Q_TILE, n_blocks*block_size) float32
//   args[1] = context_lens Tensor* (batch,) int32
//   args[2] = pij          Tensor* (spmd_blocks*Q_TILE, n_blocks*block_size) bf16 [output]
//   args[3] = mij          Tensor* (spmd_blocks*Q_TILE,) float32 [output]
//   args[4] = lij          Tensor* (spmd_blocks*Q_TILE,) float32 [output]
//   args[5] = scale_value  scalar (as float bits in uint64)
//   args[6] = bn_start     scalar: starting KV block index
//   args[7] = n_blocks     scalar: number of KV blocks in this group
//   args[8] = block_size   scalar
//   args[9] = q_loop       scalar

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

template <int TM, int TN, int TILE_STRIDE>
static __aicore__ void softmax_prepare_n_spmd(
    __gm__ float *sij_base, float scale_value, __gm__ bfloat16_t *pij_base, __gm__ float *mij_addr,
    __gm__ float *lij_addr, uint64_t n_blocks, uint64_t valid_len_last
) {
    constexpr int kAlignedRows = ((TM * sizeof(float) + 31) / 32) * (32 / sizeof(float));
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = TM / kScalarCols;

    // --- GlobalTensor types ---
    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, TM, TN>, Stride<1, 1, 1, TN, 1>>;
    using GlobalDataMxN_bf16 = GlobalTensor<bfloat16_t, Shape<1, 1, 1, TM, TN>, Stride<1, 1, 1, TN, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using GlobalScalarND =
        GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kScalarCols>, Stride<1, 1, 1, kScalarCols, 1>>;

    // --- Tile types ---
    using TileSijDyn = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, -1>;
    using TileSijPad =
        Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN, SLayout::NoneBox, 512, PadValue::Min>;
    using TileVecMxN = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;
    using TileVecMxN_bf16 = Tile<TileType::Vec, bfloat16_t, TM, TN, BLayout::RowMajor, TM, TN>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, TM, 1>;
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;
    using TileScalarRow = Tile<TileType::Vec, float, 1, TM, BLayout::RowMajor, 1, TM>;

    // --- UB memory layout (double-buffered sij) ---
    constexpr int kDataBytes = TM * TN * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);

    TileVecMxN sijTile_A;
    TileSijPad sijPadTile_A;
    TileVecMxN sijTile_B;
    TileSijPad sijPadTile_B;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileVecMxN sumAccTile;
    TileScalarDN localMaxDN;
    TileScalarDN globalMaxDN;
    TileScalarDN sumDN;
    TileVecMxN_bf16 pijBf16Tile;

    TileScalarRow localMaxRow;
    TileScalarRow globalMaxRow;
    TileScalarND globalMaxND;

    TASSIGN(sijTile_A, 0x0);
    TASSIGN(sijPadTile_A, 0x0);
    TASSIGN(sijTile_B, kDataBytes);
    TASSIGN(sijPadTile_B, kDataBytes);
    TASSIGN(pijTile, 2 * kDataBytes);
    TASSIGN(tmpTile, 3 * kDataBytes);
    TASSIGN(sumAccTile, 4 * kDataBytes);
    int scalarBase = 5 * kDataBytes;
    TASSIGN(localMaxDN, scalarBase);
    TASSIGN(localMaxRow, scalarBase);
    TASSIGN(globalMaxDN, scalarBase + kScalarDNBytes);
    TASSIGN(globalMaxRow, scalarBase + kScalarDNBytes);
    TASSIGN(globalMaxND, scalarBase + kScalarDNBytes);
    TASSIGN(sumDN, scalarBase + 2 * kScalarDNBytes);
    TASSIGN(pijBf16Tile, scalarBase + 3 * kScalarDNBytes);

    GlobalScalarND mijGlobalND(mij_addr);
    GlobalScalarDN lijGlobalDN(lij_addr);

    // ======== Pass 1: Find global row max (unscaled) ========
    // Tile stride between consecutive packed tiles = TILE_STRIDE elements
    GlobalDataMxN sijGlobal_p1_0(sij_base);
    TLOAD(sijTile_A, sijGlobal_p1_0);

    for (uint64_t i = 0; i < n_blocks; i++) {
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if (i == n_blocks - 1 && valid_len_last < static_cast<uint64_t>(TN)) {
            TileSijDyn sijDynTile(static_cast<size_t>(valid_len_last));
            if (i % 2 == 0) {
                TASSIGN(sijDynTile, 0x0);
                TFILLPAD_INPLACE(sijPadTile_A, sijDynTile);
            } else {
                TASSIGN(sijDynTile, static_cast<int>(kDataBytes));
                TFILLPAD_INPLACE(sijPadTile_B, sijDynTile);
            }
            pipe_barrier(PIPE_V);
        }

        if (i % 2 == 0) {
            TROWMAX(localMaxDN, sijTile_A, tmpTile);
        } else {
            TROWMAX(localMaxDN, sijTile_B, tmpTile);
        }
        pipe_barrier(PIPE_V);

        if (i + 1 < n_blocks) {
            GlobalDataMxN sijGlobal_next(sij_base + (i + 1) * TILE_STRIDE);
            if (i % 2 == 0) {
                TLOAD(sijTile_B, sijGlobal_next);
            } else {
                TLOAD(sijTile_A, sijGlobal_next);
            }
        }

        TRESHAPE(localMaxRow, localMaxDN);
        if (i == 0) {
            TMAX(globalMaxRow, localMaxRow, localMaxRow);
        } else {
            TMAX(globalMaxRow, globalMaxRow, localMaxRow);
        }
        pipe_barrier(PIPE_V);
    }

    TMULS(globalMaxRow, globalMaxRow, scale_value);
    pipe_barrier(PIPE_V);
    TRESHAPE(globalMaxDN, globalMaxRow);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(mijGlobalND, globalMaxND);

    // ======== Pass 2: Compute softmax ========
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    GlobalDataMxN sijGlobal_0(sij_base);
    TLOAD(sijTile_A, sijGlobal_0);

    for (uint64_t i = 0; i < n_blocks; i++) {
        GlobalDataMxN_bf16 pijGlobal(pij_base + i * TILE_STRIDE);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if (i == n_blocks - 1 && valid_len_last < static_cast<uint64_t>(TN)) {
            TileSijDyn curSijDyn(static_cast<size_t>(valid_len_last));
            if (i % 2 == 0) {
                TASSIGN(curSijDyn, 0x0);
                TFILLPAD_INPLACE(sijPadTile_A, curSijDyn);
            } else {
                TASSIGN(curSijDyn, static_cast<int>(kDataBytes));
                TFILLPAD_INPLACE(sijPadTile_B, curSijDyn);
            }
            pipe_barrier(PIPE_V);
        }

        if (i % 2 == 0) {
            TMULS(sijTile_A, sijTile_A, scale_value);
            pipe_barrier(PIPE_V);
            TROWEXPANDSUB(pijTile, sijTile_A, globalMaxDN);
        } else {
            TMULS(sijTile_B, sijTile_B, scale_value);
            pipe_barrier(PIPE_V);
            TROWEXPANDSUB(pijTile, sijTile_B, globalMaxDN);
        }
        pipe_barrier(PIPE_V);
        TEXP(pijTile, pijTile);
        pipe_barrier(PIPE_V);
        TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
        pipe_barrier(PIPE_V);
        TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);

        pipe_barrier(PIPE_V);
        if (i == 0) {
            TMULS(sumAccTile, pijTile, 1.0f);
        } else {
            TADD(sumAccTile, sumAccTile, pijTile);
        }

        pipe_barrier(PIPE_V);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(pijGlobal, pijBf16Tile);

        if (i + 1 < n_blocks) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            GlobalDataMxN sijGlobal_next(sij_base + (i + 1) * TILE_STRIDE);
            if (i % 2 == 0) {
                TLOAD(sijTile_B, sijGlobal_next);
            } else {
                TLOAD(sijTile_A, sijGlobal_next);
            }
        }
    }

    pipe_barrier(PIPE_V);
    TROWSUM(sumDN, sumAccTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(lijGlobalDN, sumDN);

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
    int64_t bn_start = static_cast<int64_t>(args[6]);
    int64_t n_blocks_total = static_cast<int64_t>(args[7]);
    int64_t block_size = static_cast<int64_t>(args[8]);
    int64_t q_loop = static_cast<int64_t>(args[9]);

    int32_t block_idx = get_block_idx(args);
    int32_t sub_block_id = get_sub_block_id(args);  // 0 = AIV0, 1 = AIV1
    int64_t batch_idx = block_idx / q_loop;

    int64_t q_tile = (block_size == 128) ? 16 : 64;
    int64_t sub_m = q_tile / 2;

    // Compute valid_len for the last block in this group
    __gm__ int32_t *ctx_ptr =
        reinterpret_cast<__gm__ int32_t *>(context_lens_t->buffer.addr) + context_lens_t->start_offset;
    int64_t cur_seq = static_cast<int64_t>(ctx_ptr[batch_idx]);
    int64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

    // Clamp n_blocks to valid range for this batch
    int64_t valid_blocks = bn_this_batch - bn_start;
    if (valid_blocks < 0) valid_blocks = 0;
    int64_t n_blocks = (valid_blocks < n_blocks_total) ? valid_blocks : n_blocks_total;

    // Compute valid_len for the last valid block
    uint64_t valid_len_last;
    if (n_blocks <= 0) {
        valid_len_last = 0;
    } else {
        int64_t last_block_seq_start = (bn_start + n_blocks - 1) * block_size;
        int64_t remaining = cur_seq - last_block_seq_start;
        if (remaining >= block_size) {
            valid_len_last = static_cast<uint64_t>(block_size);
        } else if (remaining > 0) {
            valid_len_last = static_cast<uint64_t>(remaining);
        } else {
            valid_len_last = 0;
        }
    }

    // Packed tile layout: SPMD block block_idx owns a contiguous region of
    // n_blocks_total packed (q_tile, block_size) tiles.
    // Packed base for this SPMD block:
    int64_t packed_base_offset = block_idx * q_tile * n_blocks_total * block_size;
    // AIV lane offset within each packed tile (sub_block_id selects the sub_m-row half):
    int64_t sub_offset = sub_block_id * sub_m * block_size;

    __gm__ float *sij_base =
        reinterpret_cast<__gm__ float *>(sij_t->buffer.addr) + sij_t->start_offset + packed_base_offset + sub_offset;
    __gm__ bfloat16_t *pij_base =
        reinterpret_cast<__gm__ bfloat16_t *>(pij_t->buffer.addr) + pij_t->start_offset + packed_base_offset +
        sub_offset;

    // Scalar layout: full q_tile rows pack to kAlignedRowsFull floats per block_idx;
    // each AIV lane owns kAlignedRowsSub contiguous floats inside that slab.
    int64_t kAlignedRowsFull = ((q_tile * static_cast<int64_t>(sizeof(float)) + 31) / 32) * (32 / static_cast<int64_t>(sizeof(float)));
    int64_t kAlignedRowsSub = ((sub_m * static_cast<int64_t>(sizeof(float)) + 31) / 32) * (32 / static_cast<int64_t>(sizeof(float)));
    int64_t scalar_offset = block_idx * kAlignedRowsFull + sub_block_id * kAlignedRowsSub;
    __gm__ float *mij_addr =
        reinterpret_cast<__gm__ float *>(mij_t->buffer.addr) + mij_t->start_offset + scalar_offset;
    __gm__ float *lij_addr =
        reinterpret_cast<__gm__ float *>(lij_t->buffer.addr) + lij_t->start_offset + scalar_offset;

    if (n_blocks <= 0) {
        // No valid KV data — emit neutral values
        for (int64_t i = 0; i < kAlignedRowsSub; i++) {
            mij_addr[i] = -1e30f;
            lij_addr[i] = 0.0f;
        }
        return;
    }

    // Tile stride = full packed tile size (q_tile * block_size), NOT sub_m * block_size
    if (q_tile == 16) {
        softmax_prepare_n_spmd<8, 128, 16 * 128>(
            sij_base, scale_value, pij_base, mij_addr, lij_addr,
            static_cast<uint64_t>(n_blocks), valid_len_last
        );
    } else {
        softmax_prepare_n_spmd<32, 64, 64 * 64>(
            sij_base, scale_value, pij_base, mij_addr, lij_addr,
            static_cast<uint64_t>(n_blocks), valid_len_last
        );
    }
}
