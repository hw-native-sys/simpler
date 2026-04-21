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
// Two-Pass Softmax Kernel (AIV) for n_blocks tiles
//
// Input:  sij_buf (n_blocks * M, N) fp32 — QK results stacked vertically
// Output: pij_buf (n_blocks * M, N) bf16 — attention weights per block
//         mij (M,) fp32 — global row max across all blocks
//         lij (M,) fp32 — total row sum across all blocks
//
// Pass 1: Iterate over n_blocks tiles, apply scale, mask last block,
//         find global m = max over all blocks of rowmax(S_i * scale)
//         Uses TRESHAPE for DN↔Row conversion to keep globalMax in UB
//         (eliminates 63 × 4 GM round-trip operations).
// Pass 2: Iterate again, compute P_i = exp(S_i * scale - m) -> bf16,
//         accumulate l = sum over all blocks of rowsum(P_i)
//         Uses double-buffered sij tiles to overlap TLOAD with computation.
//
// Two-pass ensures all P_i tiles share the same scale (global max),
// enabling direct TMATMUL_ACC accumulation in the PV kernel.
//
// Supports two tile configurations via runtime dispatch:
//   Case1: M=16, N=128 (q_tile=16, block_size=128)
//   Case2: M=64, N=64  (q_tile=64, block_size=64)

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

template <int M, int N>
static __aicore__ void softmax_prepare_n_impl(
    __gm__ float *sij_base, float scale_value, __gm__ bfloat16_t *pij_base, __gm__ float *mij_addr,
    __gm__ float *lij_addr, uint64_t n_blocks, uint64_t valid_len_last
) {
    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = M / kScalarCols;

    // --- GlobalTensor types ---
    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_bf16 = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;
    using GlobalScalarND =
        GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kScalarCols>, Stride<1, 1, 1, kScalarCols, 1>>;

    // --- Tile types ---
    using TileSijDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, -1>;
    using TileSijPad = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>;
    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_bf16 = Tile<TileType::Vec, bfloat16_t, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;
    // RowMajor (1, M) tile for element-wise arithmetic via TRESHAPE
    using TileScalarRow = Tile<TileType::Vec, float, 1, M, BLayout::RowMajor, 1, M>;

    // --- UB memory layout (double-buffered sij) ---
    constexpr int kDataBytes = M * N * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);

    // Double-buffered sij tiles
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

    // TRESHAPE aliases (same UB address as their DN counterparts)
    TileScalarRow localMaxRow;
    TileScalarRow globalMaxRow;

    // ND alias for storing globalMax to GM
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
    TASSIGN(localMaxRow, scalarBase);  // alias: same UB as localMaxDN
    TASSIGN(globalMaxDN, scalarBase + kScalarDNBytes);
    TASSIGN(globalMaxRow, scalarBase + kScalarDNBytes);  // alias: same UB as globalMaxDN
    TASSIGN(globalMaxND, scalarBase + kScalarDNBytes);   // alias: same UB as globalMaxDN
    TASSIGN(sumDN, scalarBase + 2 * kScalarDNBytes);
    TASSIGN(pijBf16Tile, scalarBase + 3 * kScalarDNBytes);

    // GM aliases (mij/lij output buffers)
    GlobalScalarND mijGlobalND(mij_addr);
    GlobalScalarDN lijGlobalDN(lij_addr);

    // ======== Pass 1: Find global row max via TRESHAPE (no GM round-trip) ========
    for (uint64_t i = 0; i < n_blocks; i++) {
        GlobalDataMxN sijGlobal(sij_base + i * M * N);
        TLOAD(sijTile_A, sijGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if (i == n_blocks - 1 && valid_len_last < static_cast<uint64_t>(N)) {
            TileSijDyn sijDynTile(static_cast<size_t>(valid_len_last));
            TASSIGN(sijDynTile, 0x0);
            TFILLPAD_INPLACE(sijPadTile_A, sijDynTile);
        }

        TMULS(sijTile_A, sijTile_A, scale_value);
        pipe_barrier(PIPE_V);
        TROWMAX(localMaxDN, sijTile_A, tmpTile);

        // TRESHAPE: ColMajor(M,1) → RowMajor(1,M) for element-wise TMAX
        TRESHAPE(localMaxRow, localMaxDN);
        if (i == 0) {
            pipe_barrier(PIPE_V);
            TMAX(globalMaxRow, localMaxRow, localMaxRow);
        } else {
            pipe_barrier(PIPE_V);
            TMAX(globalMaxRow, globalMaxRow, localMaxRow);
        }
    }

    // TRESHAPE back: RowMajor(1,M) → ColMajor(M,1) for Pass 2's TROWEXPANDSUB
    TRESHAPE(globalMaxDN, globalMaxRow);

    // Store final global max to mij for online_update to consume
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(mijGlobalND, globalMaxND);

    // ======== Pass 2: Compute softmax with double-buffered sij ========
    // globalMaxDN is already in UB from TRESHAPE — no reload needed.
    // Sync MTE3→MTE2 to ensure the mij TSTORE completed before first sij TLOAD.
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);

    // Pre-load first sij tile into buffer A
    GlobalDataMxN sijGlobal_0(sij_base);
    TLOAD(sijTile_A, sijGlobal_0);

    for (uint64_t i = 0; i < n_blocks; i++) {
        GlobalDataMxN_bf16 pijGlobal(pij_base + i * M * N);

        // Wait for current tile's TLOAD to complete
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // TFILLPAD on current buffer if last block with partial valid length
        if (i == n_blocks - 1 && valid_len_last < static_cast<uint64_t>(N)) {
            TileSijDyn curSijDyn(static_cast<size_t>(valid_len_last));
            if (i % 2 == 0) {
                TASSIGN(curSijDyn, 0x0);
                TFILLPAD_INPLACE(sijPadTile_A, curSijDyn);
            } else {
                TASSIGN(curSijDyn, static_cast<int>(kDataBytes));
                TFILLPAD_INPLACE(sijPadTile_B, curSijDyn);
            }
        }

        // Compute on current buffer (select A or B based on iteration parity)
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
        TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
        TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);

        if (i == 0) {
            TMULS(sumAccTile, pijTile, 1.0f);
        } else {
            TADD(sumAccTile, sumAccTile, pijTile);
        }

        // Store pij (must complete before next iteration's TCVT overwrites pijBf16Tile)
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(pijGlobal, pijBf16Tile);

        // Prefetch next sij into alternate buffer (after TSTORE to avoid UB race)
        if (i + 1 < n_blocks) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            GlobalDataMxN sijGlobal_next(sij_base + (i + 1) * M * N);
            if (i % 2 == 0) {
                TLOAD(sijTile_B, sijGlobal_next);
            } else {
                TLOAD(sijTile_A, sijGlobal_next);
            }
        }
    }

    // Compute final row sum from accumulated pij values
    pipe_barrier(PIPE_V);
    TROWSUM(sumDN, sumAccTile, tmpTile);

    // Store lij (total sum). mij already stored after Pass 1.
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(lijGlobalDN, sumDN);

    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ TensorData *sij_buf = reinterpret_cast<__gm__ TensorData *>(args[0]);
    __gm__ TensorData *pij_buf = reinterpret_cast<__gm__ TensorData *>(args[1]);
    __gm__ TensorData *mij = reinterpret_cast<__gm__ TensorData *>(args[2]);
    __gm__ TensorData *lij = reinterpret_cast<__gm__ TensorData *>(args[3]);
    union {
        uint64_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[4]);
    float scale_value = scale_conv.f;
    uint64_t n_blocks = static_cast<uint64_t>(args[5]);
    uint64_t valid_len_last = static_cast<uint64_t>(args[6]);

    __gm__ float *sij_base = reinterpret_cast<__gm__ float *>(sij_buf->buffer.addr) + sij_buf->start_offset;
    __gm__ bfloat16_t *pij_base = reinterpret_cast<__gm__ bfloat16_t *>(pij_buf->buffer.addr) + pij_buf->start_offset;
    __gm__ float *mij_addr = reinterpret_cast<__gm__ float *>(mij->buffer.addr) + mij->start_offset;
    __gm__ float *lij_addr = reinterpret_cast<__gm__ float *>(lij->buffer.addr) + lij->start_offset;

    uint64_t q_tile_size = static_cast<uint64_t>(sij_buf->shapes[0]);

    if (q_tile_size == 16) {
        softmax_prepare_n_impl<16, 128>(sij_base, scale_value, pij_base, mij_addr, lij_addr, n_blocks, valid_len_last);
    } else {
        softmax_prepare_n_impl<64, 64>(sij_base, scale_value, pij_base, mij_addr, lij_addr, n_blocks, valid_len_last);
    }
}
