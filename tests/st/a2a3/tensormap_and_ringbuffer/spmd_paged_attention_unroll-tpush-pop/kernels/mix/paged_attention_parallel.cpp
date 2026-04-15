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
 * Paged Attention Parallel Kernel — Combined AIC + AIV (TPUSH/TPOP)
 *
 * Single source compiled twice:
 *   - AIC (cube): __DAV_CUBE__ → QK matmul, PV matmul
 *   - AIV (vector): __DAV_VEC__ → online softmax, online update
 *
 * Per-block pipeline (all KV blocks processed in one invocation):
 *   AIC: QK matmul → TPUSH(sij) → TPOP(pij) → PV matmul → TPUSH(oi_new)
 *   AIV: TPOP(sij) → online softmax → TPUSH(pij) → TPOP(oi_new) → online update
 *
 * Three TPUSH/TPOP pipes:
 *   - sij_pipe (C2V): scores (Q_TILE, block_size) fp32, TILE_UP_DOWN split
 *   - pij_pipe (V2C): probabilities (Q_TILE, block_size) bf16, TILE_UP_DOWN
 *   - oi_pipe  (C2V): PV output (Q_TILE, head_dim) fp32, TILE_UP_DOWN split
 *
 * Q_TILE=16, SUB_QT=8. Supports block_size=64|128, head_dim=128.
 *
 * MixedKernels args:
 *   args[0]  = query         Tensor* (batch*num_heads, head_dim) bf16
 *   args[1]  = key_cache     Tensor* (kv_total_rows, head_dim) bf16
 *   args[2]  = value_cache   Tensor* (kv_total_rows, head_dim) bf16
 *   args[3]  = block_table   Tensor* (batch, max_blocks_per_req) int32
 *   args[4]  = context_lens  Tensor* (batch,) int32
 *   args[5]  = out           Tensor* (batch*num_heads, head_dim) float32 [output]
 *   args[6]  = sij_fifo      Tensor* GM ring buffer for sij pipe
 *   args[7]  = pij_fifo      Tensor* GM ring buffer for pij pipe
 *   args[8]  = oi_fifo       Tensor* GM ring buffer for oi_new pipe
 *   args[9]  = scale_value   scalar (float bits in uint64)
 *   args[10] = num_heads     scalar
 *   args[11] = head_dim      scalar
 *   args[12] = block_size    scalar
 *   args[13] = max_num_blocks_per_req scalar
 *   args[14] = q_loop        scalar
 */

#include <cstdint>
// NOLINTBEGIN(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

#include "tensor.h"

using pto::BLayout;
using pto::Direction;
using pto::GlobalTensor;
using pto::Layout;
using pto::PadValue;
using pto::RoundMode;
using pto::Shape;
using pto::SLayout;
using pto::Stride;
using pto::Tile;
using pto::TileAcc;
using pto::TileLeft;
using pto::TileRight;
using pto::TileSplitAxis;
using pto::TileType;
using pto::TPipe;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#include "intrinsic.h"

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

static constexpr int Q_TILE = 16;
static constexpr int SUB_QT = Q_TILE / 2;
static constexpr int HEAD_DIM = 128;

// TPUSH/TPOP pipe flag IDs (each consumes 2 consecutive IDs: data + backpressure)
static constexpr uint16_t SIJ_FLAG_ID = 0;
static constexpr uint16_t PIJ_FLAG_ID = 2;
static constexpr uint16_t OI_FLAG_ID = 4;
static constexpr uint8_t FIFO_DEPTH = 1;

// GM FIFO slot sizes (max case: block_size=128)
static constexpr uint32_t SIJ_SLOT_SIZE = Q_TILE * 128 * sizeof(float);       // 8192
static constexpr uint32_t PIJ_SLOT_SIZE = Q_TILE * 128 * sizeof(bfloat16_t);  // 4096
static constexpr uint32_t OI_SLOT_SIZE = Q_TILE * HEAD_DIM * sizeof(float);   // 8192

// Pipe types
using SijPipeT = TPipe<SIJ_FLAG_ID, Direction::DIR_C2V, SIJ_SLOT_SIZE, FIFO_DEPTH>;
using PijPipeT = TPipe<PIJ_FLAG_ID, Direction::DIR_V2C, PIJ_SLOT_SIZE, FIFO_DEPTH>;
using OiPipeT = TPipe<OI_FLAG_ID, Direction::DIR_C2V, OI_SLOT_SIZE, FIFO_DEPTH>;

// AIV UB consumer buffer layout (fixed, independent of template params)
// LocalSlotNum=2 by default, so each consumer needs 2 slots
// sij consumer: 2 * SUB_QT * 128 * 4 = 8192 bytes (max, block_size=128)
// oi  consumer: 2 * SUB_QT * HEAD_DIM * 4 = 8192 bytes
static constexpr uint32_t SIJ_UB_BASE = 0x0;
static constexpr uint32_t SIJ_UB_SIZE = 2 * SUB_QT * 128 * sizeof(float);      // 8192
static constexpr uint32_t OI_UB_BASE = SIJ_UB_BASE + SIJ_UB_SIZE;              // 0x2000
static constexpr uint32_t OI_UB_SIZE = 2 * SUB_QT * HEAD_DIM * sizeof(float);  // 8192
static constexpr uint32_t WORK_UB_BASE = OI_UB_BASE + OI_UB_SIZE;              // 0x4000

// AIC L1 consumer buffer for V2C pij pipe
// LocalSlotNum=2, pij consumer: 2 * Q_TILE * 128 * 2 = 8192 bytes
static constexpr uint32_t PIJ_L1_BASE = 0x40000;
static constexpr uint32_t PIJ_L1_SIZE = 2 * Q_TILE * 128 * sizeof(bfloat16_t);  // 8192

// ============================================================================
// AIC (Cube) side
// ============================================================================

template <int M, int K, int N>
static __aicore__ void aic_process_blocks(
    __gm__ bfloat16_t *qi_base, __gm__ bfloat16_t *key_base, __gm__ bfloat16_t *val_base, __gm__ int32_t *bt,
    uint64_t bt_offset, uint64_t n_blocks, SijPipeT &sij_pipe, PijPipeT &pij_pipe, OiPipeT &oi_pipe
) {
    // QK tile types
    using GlobalA_QK = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB_QK = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using TileMatA_QK = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB_QK = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    using LeftTile_QK = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile_QK = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile_QK = TileAcc<float, M, N, M, N>;

    // PV tile types
    using GlobalB_PV = GlobalTensor<bfloat16_t, Shape<1, 1, 1, N, K>, Stride<N * K, N * K, N * K, K, 1>>;
    using TileMatB_PV = Tile<TileType::Mat, bfloat16_t, N, K, BLayout::ColMajor, N, K, SLayout::RowMajor, 512>;
    using PijMatTile = Tile<TileType::Mat, bfloat16_t, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using LeftTile_PV = TileLeft<bfloat16_t, M, N, M, N>;
    using RightTile_PV = TileRight<bfloat16_t, N, K, N, K>;
    using AccTile_PV = TileAcc<float, M, K, M, K>;

    // L1 layout for QK:
    //   0x00000: qi (M*K*2)
    //   0x20000: kj double-buffer (2 * K*N*2)
    constexpr int kQKBBytes = K * N * static_cast<int>(sizeof(bfloat16_t));
    constexpr int kPVBBytes = N * K * static_cast<int>(sizeof(bfloat16_t));

    TileMatA_QK aMatTile_QK;
    TileMatB_QK bMatTile_QK_A, bMatTile_QK_B;
    TASSIGN(aMatTile_QK, 0x0);
    TASSIGN(bMatTile_QK_A, 0x20000);
    TASSIGN(bMatTile_QK_B, 0x20000 + kQKBBytes);

    LeftTile_QK aTile_QK;
    RightTile_QK bTile_QK;
    AccTile_QK cTile_QK;
    TASSIGN(aTile_QK, 0x0);
    TASSIGN(bTile_QK, 0x0);
    TASSIGN(cTile_QK, 0x0);

    // L1 layout for PV:
    //   PIJ_L1_BASE: pij from V2C TPOP (auto-assigned, 2 slots of M*N*2)
    //   PIJ_L1_BASE + PIJ_L1_SIZE: vj double-buffer (2 * N*K*2)
    PijMatTile pijMatTile;
    TileMatB_PV bMatTile_PV_A, bMatTile_PV_B;
    TASSIGN(bMatTile_PV_A, PIJ_L1_BASE + PIJ_L1_SIZE);
    TASSIGN(bMatTile_PV_B, PIJ_L1_BASE + PIJ_L1_SIZE + kPVBBytes);

    LeftTile_PV aTile_PV;
    RightTile_PV bTile_PV;
    AccTile_PV cTile_PV;
    TASSIGN(aTile_PV, 0x0);
    TASSIGN(bTile_PV, 0x0);
    TASSIGN(cTile_PV, 0x0);

    // Hoist qi TLOAD
    GlobalA_QK qiGlobal(qi_base);
    TLOAD(aMatTile_QK, qiGlobal);

    for (uint64_t i = 0; i < n_blocks; i++) {
        // ---- QK Matmul ----
        GlobalB_QK kjGlobal(key_base + bt[bt_offset + i] * N * K);
        if (i % 2 == 0) {
            TLOAD(bMatTile_QK_A, kjGlobal);
        } else {
            TLOAD(bMatTile_QK_B, kjGlobal);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        TMOV(aTile_QK, aMatTile_QK);
        if (i % 2 == 0) {
            TMOV(bTile_QK, bMatTile_QK_A);
        } else {
            TMOV(bTile_QK, bMatTile_QK_B);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(cTile_QK, aTile_QK, bTile_QK);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        // TPUSH sij (C2V): AccTile L0C -> GM -> AIV UB
        TPUSH<SijPipeT, AccTile_QK, TileSplitAxis::TILE_UP_DOWN>(sij_pipe, cTile_QK);

        // TPOP pij (V2C): AIV UB -> GM -> L1 (auto-assigned from PIJ_L1_BASE)
        TPOP<PijPipeT, PijMatTile, TileSplitAxis::TILE_NO_SPLIT>(pij_pipe, pijMatTile);

        // ---- PV Matmul ----
        GlobalB_PV vjGlobal(val_base + bt[bt_offset + i] * N * K);
        if (i % 2 == 0) {
            TLOAD(bMatTile_PV_A, vjGlobal);
        } else {
            TLOAD(bMatTile_PV_B, vjGlobal);
        }

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // pijMatTile address set by TPOP
        TMOV(aTile_PV, pijMatTile);
        if (i % 2 == 0) {
            TMOV(bTile_PV, bMatTile_PV_A);
        } else {
            TMOV(bTile_PV, bMatTile_PV_B);
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(cTile_PV, aTile_PV, bTile_PV);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        // TPUSH oi_new (C2V): AccTile L0C -> GM -> AIV UB
        TPUSH<OiPipeT, AccTile_PV, TileSplitAxis::TILE_UP_DOWN>(oi_pipe, cTile_PV);

        if (i + 1 < n_blocks) {
            pipe_barrier(PIPE_ALL);
        }
    }

    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

// ============================================================================
// AIV (Vector) side
// ============================================================================

template <int TM, int TN>
static __aicore__ void aiv_process_blocks(
    float scale_value, uint64_t n_blocks, uint64_t valid_len_last, __gm__ float *dst_ptr, SijPipeT &sij_pipe,
    PijPipeT &pij_pipe, OiPipeT &oi_pipe
) {
    constexpr int HD = HEAD_DIM;
    constexpr int kAlignedRows = ((TM * sizeof(float) + 31) / 32) * (32 / sizeof(float));
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = TM / kScalarCols;

    using SijVecTile = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;
    using PijVecBf16Tile = Tile<TileType::Vec, bfloat16_t, TM, TN, BLayout::RowMajor, TM, TN>;
    using OiVecTile = Tile<TileType::Vec, float, TM, HD, BLayout::RowMajor, TM, HD>;

    using TileVecMxN = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;
    using TileSijDyn = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, -1>;
    using TileSijPad =
        Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN, SLayout::NoneBox, 512, PadValue::Min>;
    // DN (ColMajor) for row-broadcast ops: TROWMAX, TROWSUM, TROWEXPAND*
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, TM, 1>;
    // ND (RowMajor) for element-wise ops: TMULS, TMUL, TSUB, TADD, TEXP, TMAX
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;
    // Row for max operations
    using TileScalarRow = Tile<TileType::Vec, float, 1, TM, BLayout::RowMajor, 1, TM>;
    using TileDataMxHD = Tile<TileType::Vec, float, TM, HD, BLayout::RowMajor, TM, HD>;
    using GlobalDataMxHD = GlobalTensor<float, Shape<1, 1, 1, TM, HD>, Stride<1, 1, 1, HD, 1>>;

    constexpr int kSijBytes = TM * TN * sizeof(float);
    constexpr int kPijBf16Bytes = TM * TN * sizeof(bfloat16_t);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);
    constexpr int kScalarNDBytes = kScalarRows * kScalarCols * sizeof(float);

    // Working tiles (after FIFO consumer buffers)
    SijVecTile sijTile;
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    PijVecBf16Tile pijBf16Tile;
    // DN tiles for row-broadcast
    TileScalarDN localMaxDN, globalMaxDN;
    TileScalarDN alphaDN_dn, llDN, glDN;
    // ND tiles for element-wise
    TileScalarND gmND, glND, alphaND, llND, dmND, miNewND, mijND;
    // Row tiles for max
    TileScalarRow localMaxRow, globalMaxRow, gmRow;
    OiVecTile oiNewTile;
    TileDataMxHD goTile;

    int ub = WORK_UB_BASE;
    TASSIGN(pijTile, ub);
    ub += kSijBytes;
    TASSIGN(pijBf16Tile, ub);
    ub += kPijBf16Bytes;
    TASSIGN(tmpTile, ub);
    ub += kSijBytes;

    int sb = ub;
    TASSIGN(localMaxDN, sb);
    TASSIGN(localMaxRow, sb);
    sb += kScalarDNBytes;
    TASSIGN(globalMaxDN, sb);
    TASSIGN(globalMaxRow, sb);
    sb += kScalarDNBytes;
    // gmND/gmRow alias at the same address
    TASSIGN(gmND, sb);
    TASSIGN(gmRow, sb);
    sb += kScalarDNBytes;
    // glND/glDN alias
    TASSIGN(glND, sb);
    TASSIGN(glDN, sb);
    sb += kScalarDNBytes;
    // alphaND/alphaDN_dn alias
    TASSIGN(alphaND, sb);
    TASSIGN(alphaDN_dn, sb);
    sb += kScalarDNBytes;
    // llND/llDN alias
    TASSIGN(llND, sb);
    TASSIGN(llDN, sb);
    sb += kScalarDNBytes;
    TASSIGN(dmND, sb);
    sb += kScalarNDBytes;
    TASSIGN(miNewND, sb);
    sb += kScalarNDBytes;
    TASSIGN(mijND, sb);
    sb += kScalarNDBytes;

    TASSIGN(goTile, sb);

    GlobalDataMxHD dstGlobal(dst_ptr);

    for (uint64_t i = 0; i < n_blocks; i++) {
        // ---- TPOP sij from AIC ----
        TPOP<SijPipeT, SijVecTile, TileSplitAxis::TILE_UP_DOWN>(sij_pipe, sijTile);

        // Pad last block if partial
        if (i == n_blocks - 1 && valid_len_last < static_cast<uint64_t>(TN)) {
            int sij_addr = SIJ_UB_BASE + static_cast<int>((i % 2) * TM * TN * sizeof(float));
            TASSIGN(sijPadTile, sij_addr);
            TileSijDyn sijDynTile(static_cast<size_t>(valid_len_last));
            TASSIGN(sijDynTile, sij_addr);
            TFILLPAD_INPLACE(sijPadTile, sijDynTile);
            pipe_barrier(PIPE_V);
        }

        // rowmax (produces DN tile)
        TROWMAX(localMaxDN, sijTile, tmpTile);
        pipe_barrier(PIPE_V);
        // Convert DN -> Row for element-wise max
        TRESHAPE(localMaxRow, localMaxDN);

        if (i == 0) {
            // First block: initialize globalMax = scale * localMax
            TMULS(globalMaxRow, localMaxRow, scale_value);
        } else {
            // Update: globalMax = max(gm_prev, scale * localMax)
            TMULS(localMaxRow, localMaxRow, scale_value);
            pipe_barrier(PIPE_V);
            TMAX(globalMaxRow, gmRow, localMaxRow);
        }
        pipe_barrier(PIPE_V);
        // Convert Row -> DN for row-broadcast
        TRESHAPE(globalMaxDN, globalMaxRow);

        // softmax: exp(sij * scale - globalMax)
        TMULS(sijTile, sijTile, scale_value);
        pipe_barrier(PIPE_V);
        TROWEXPANDSUB(pijTile, sijTile, globalMaxDN);
        pipe_barrier(PIPE_V);
        TEXP(pijTile, pijTile);
        pipe_barrier(PIPE_V);

        // fp32 -> bf16 -> fp32 for PV matmul precision matching
        TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
        pipe_barrier(PIPE_V);
        TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);
        pipe_barrier(PIPE_V);

        // rowsum -> DN tile, then reshape to ND for element-wise ops
        TROWSUM(llDN, pijTile, tmpTile);
        pipe_barrier(PIPE_V);
        TRESHAPE(llND, llDN);
        pipe_barrier(PIPE_V);

        // ---- TPUSH pij to AIC ----
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TPUSH<PijPipeT, PijVecBf16Tile, TileSplitAxis::TILE_UP_DOWN>(pij_pipe, pijBf16Tile);

        // ---- TPOP oi_new from AIC ----
        TPOP<OiPipeT, OiVecTile, TileSplitAxis::TILE_UP_DOWN>(oi_pipe, oiNewTile);

        // ---- Online update ----
        if (i == 0) {
            // First block: go = oi_new, gm = globalMax, gl = ll
            TMULS(goTile, oiNewTile, 1.0f);
            pipe_barrier(PIPE_V);
            // gmRow = globalMaxRow (copy)
            TMULS(gmRow, globalMaxRow, 1.0f);
            pipe_barrier(PIPE_V);
            // glND = llND (copy using ND tiles)
            TMULS(glND, llND, 1.0f);
        } else {
            // Compute alpha = exp(gm_prev - gm_new) using ND tiles
            // globalMaxRow contains the new max, gmRow contains the old max
            // Convert both to ND for element-wise ops
            TRESHAPE(mijND, globalMaxRow);
            TRESHAPE(dmND, gmRow);
            pipe_barrier(PIPE_V);
            TSUB(alphaND, dmND, mijND);
            pipe_barrier(PIPE_V);
            TEXP(alphaND, alphaND);
            pipe_barrier(PIPE_V);

            // go = go * alpha + oi_new (need alphaDN for TROWEXPANDMUL)
            TRESHAPE(alphaDN_dn, alphaND);
            pipe_barrier(PIPE_V);
            TROWEXPANDMUL(goTile, goTile, alphaDN_dn);
            pipe_barrier(PIPE_V);
            TADD(goTile, goTile, oiNewTile);
            pipe_barrier(PIPE_V);

            // gl = gl * alpha + ll (element-wise with ND tiles)
            TMUL(glND, glND, alphaND);
            pipe_barrier(PIPE_V);
            TADD(glND, glND, llND);
            pipe_barrier(PIPE_V);

            // Update gm = globalMax
            TMULS(gmRow, globalMaxRow, 1.0f);
        }

        pipe_barrier(PIPE_V);
    }

    // Normalize: go = go / gl (need glDN for TROWEXPANDDIV)
    TRESHAPE(glDN, glND);
    pipe_barrier(PIPE_V);
    TROWEXPANDDIV(goTile, goTile, glDN);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, goTile);

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

// ============================================================================
// Entry point
// ============================================================================

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *query_t = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *key_cache_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *value_cache_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *block_table_t = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *context_lens_t = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *out_t = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ Tensor *sij_fifo_t = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ Tensor *pij_fifo_t = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ Tensor *oi_fifo_t = reinterpret_cast<__gm__ Tensor *>(args[8]);

    float scale_value = from_u64<float>(static_cast<uint64_t>(args[9]));
    int64_t num_heads = static_cast<int64_t>(args[10]);
    int64_t head_dim = static_cast<int64_t>(args[11]);
    int64_t block_size = static_cast<int64_t>(args[12]);
    int64_t max_blocks_per_req = static_cast<int64_t>(args[13]);
    int64_t q_loop = static_cast<int64_t>(args[14]);

    int32_t block_idx = get_block_idx(args);
    int64_t batch_idx = block_idx / q_loop;
    int64_t q_tile_idx = block_idx % q_loop;

    __gm__ int32_t *ctx_ptr =
        reinterpret_cast<__gm__ int32_t *>(context_lens_t->buffer.addr) + context_lens_t->start_offset;
    int64_t cur_seq = static_cast<int64_t>(ctx_ptr[batch_idx]);
    int64_t n_blocks = (cur_seq + block_size - 1) / block_size;

    uint64_t valid_len_last = 0;
    if (n_blocks > 0) {
        int64_t last_block_seq = (n_blocks - 1) * block_size;
        int64_t remaining = cur_seq - last_block_seq;
        valid_len_last = (remaining >= block_size) ? static_cast<uint64_t>(block_size) :
                                                     (remaining > 0 ? static_cast<uint64_t>(remaining) : 0);
    }

    // GM FIFO buffer per SPMD block
    __gm__ void *sij_fifo_base = reinterpret_cast<__gm__ void *>(
        reinterpret_cast<__gm__ uint8_t *>(sij_fifo_t->buffer.addr) + block_idx * SIJ_SLOT_SIZE * FIFO_DEPTH
    );
    __gm__ void *pij_fifo_base = reinterpret_cast<__gm__ void *>(
        reinterpret_cast<__gm__ uint8_t *>(pij_fifo_t->buffer.addr) + block_idx * PIJ_SLOT_SIZE * FIFO_DEPTH
    );
    __gm__ void *oi_fifo_base = reinterpret_cast<__gm__ void *>(
        reinterpret_cast<__gm__ uint8_t *>(oi_fifo_t->buffer.addr) + block_idx * OI_SLOT_SIZE * FIFO_DEPTH
    );

    SijPipeT sij_pipe(sij_fifo_base, SIJ_UB_BASE, 0U);
    PijPipeT pij_pipe(pij_fifo_base, 0U, PIJ_L1_BASE);
    OiPipeT oi_pipe(oi_fifo_base, OI_UB_BASE, 0U);

    if constexpr (DAV_CUBE) {
        if (n_blocks <= 0) return;

        int64_t q_offset = (batch_idx * num_heads + q_tile_idx * Q_TILE) * head_dim;
        __gm__ bfloat16_t *qi_base =
            reinterpret_cast<__gm__ bfloat16_t *>(query_t->buffer.addr) + query_t->start_offset + q_offset;
        __gm__ bfloat16_t *key_base =
            reinterpret_cast<__gm__ bfloat16_t *>(key_cache_t->buffer.addr) + key_cache_t->start_offset;
        __gm__ bfloat16_t *val_base =
            reinterpret_cast<__gm__ bfloat16_t *>(value_cache_t->buffer.addr) + value_cache_t->start_offset;
        __gm__ int32_t *bt =
            reinterpret_cast<__gm__ int32_t *>(block_table_t->buffer.addr) + block_table_t->start_offset;
        uint64_t bt_offset = static_cast<uint64_t>(batch_idx * max_blocks_per_req);

        if (block_size == 128) {
            aic_process_blocks<Q_TILE, 128, 128>(
                qi_base, key_base, val_base, bt, bt_offset, static_cast<uint64_t>(n_blocks), sij_pipe, pij_pipe, oi_pipe
            );
        } else {
            aic_process_blocks<Q_TILE, 128, 64>(
                qi_base, key_base, val_base, bt, bt_offset, static_cast<uint64_t>(n_blocks), sij_pipe, pij_pipe, oi_pipe
            );
        }
    }

    if constexpr (DAV_VEC) {
        int32_t sub_block_id = get_sub_block_id(args);
        int64_t row_offset = sub_block_id * SUB_QT;

        int64_t out_offset = (batch_idx * num_heads + q_tile_idx * Q_TILE + row_offset) * head_dim;
        __gm__ float *dst = reinterpret_cast<__gm__ float *>(out_t->buffer.addr) + out_t->start_offset + out_offset;

        if (n_blocks <= 0) {
            for (int64_t j = 0; j < SUB_QT * head_dim; j++) {
                dst[j] = 0.0f;
            }
            return;
        }

        if (block_size == 128) {
            aiv_process_blocks<SUB_QT, 128>(
                scale_value, static_cast<uint64_t>(n_blocks), valid_len_last, dst, sij_pipe, pij_pipe, oi_pipe
            );
        } else {
            aiv_process_blocks<SUB_QT, 64>(
                scale_value, static_cast<uint64_t>(n_blocks), valid_len_last, dst, sij_pipe, pij_pipe, oi_pipe
            );
        }
    }
}
// NOLINTEND(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
