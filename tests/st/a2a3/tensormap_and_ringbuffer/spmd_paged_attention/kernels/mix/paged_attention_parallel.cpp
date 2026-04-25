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
 * Paged Attention MIX Kernel — AIC + AIV in single source via TPUSH/TPOP
 *
 * Hardware block_num is fixed at 24. Each hardware block strides over
 * total_logical_blocks = batch * q_loop logical work items:
 *   for (block_idx = hw_block_idx; block_idx < total_logical_blocks; block_idx += 24)
 * Each logical block_idx encodes one (batch_idx, q_tile_idx) position.
 *
 * q_tile adapts to num_heads at runtime: q_tile = min(num_heads, MAX_Q_TILE).
 * When num_heads <= MAX_Q_TILE, q_loop = 1 and each block processes all heads.
 * Two q_tile shapes are statically dispatched: 16 (default) and 64.
 *
 * Compiled twice: once with __DAV_CUBE__ (AIC), once with __DAV_VEC__ (AIV).
 * AIC and AIV cooperate via 3 GM-backed FIFO pipes (one set per hardware block,
 * reused across stride-loop iterations):
 *   - sij_pipe (C2V): QK scores    (Q_TILE, block_size) fp32, TILE_UP_DOWN
 *   - pij_pipe (V2C): softmax probs (Q_TILE, block_size) bf16, TILE_UP_DOWN
 *   - oi_pipe  (C2V): PV output    (Q_TILE, head_dim)   fp32, TILE_UP_DOWN
 *
 * Per-block pipeline:
 *   AIC: QK matmul → TPUSH(sij) → TPOP(pij) → PV matmul → TPUSH(oi_new)
 *   AIV: TPOP(sij) → online softmax → TPUSH(pij) → TPOP(oi_new) → online update
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
 *   args[15] = total_logical_blocks scalar (= batch * q_loop)
 *   args[16] = q_tile        scalar (16 or 64)
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

#include "intrinsic.h"

static constexpr int MAX_Q_TILE = 64;
static constexpr int HEAD_DIM = 128;
static constexpr int MAX_BLOCK_SIZE = 128;

// TPUSH/TPOP pipe flag IDs (each consumes 2 consecutive IDs: data + backpressure)
static constexpr uint16_t SIJ_FLAG_ID = 0;
static constexpr uint16_t PIJ_FLAG_ID = 2;
static constexpr uint16_t OI_FLAG_ID = 4;
static constexpr uint8_t FIFO_DEPTH = 2;

// Per-q_tile compile-time configuration: pipe types, slot sizes, UB/L1 layouts.
// QT must be 16 or 64. SUB_QT = QT / 2 (each of AIV0/AIV1 handles half the rows).
template <int QT>
struct PAConfig {
    static constexpr int Q_TILE = QT;
    static constexpr int SUB_QT = QT / 2;

    // GM FIFO slot sizes (full tile per slot, sized for max block_size to allow
    // the same FIFO to host both block_size=64 and block_size=128 cases).
    static constexpr uint32_t SIJ_SLOT_SIZE = QT * MAX_BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t PIJ_SLOT_SIZE = QT * MAX_BLOCK_SIZE * sizeof(bfloat16_t);
    static constexpr uint32_t OI_SLOT_SIZE = QT * HEAD_DIM * sizeof(float);

    using SijPipeT = TPipe<SIJ_FLAG_ID, Direction::DIR_C2V, SIJ_SLOT_SIZE, FIFO_DEPTH>;
    using PijPipeT = TPipe<PIJ_FLAG_ID, Direction::DIR_V2C, PIJ_SLOT_SIZE, FIFO_DEPTH>;
    using OiPipeT = TPipe<OI_FLAG_ID, Direction::DIR_C2V, OI_SLOT_SIZE, FIFO_DEPTH>;

    // AIV UB consumer buffer layout (sized for SUB_QT rows per AIV lane)
    static constexpr uint32_t SIJ_UB_BASE = 0x0;
    static constexpr uint32_t SIJ_UB_SIZE = 2 * SUB_QT * MAX_BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t OI_UB_BASE = SIJ_UB_BASE + SIJ_UB_SIZE;
    static constexpr uint32_t OI_UB_SIZE = 2 * SUB_QT * HEAD_DIM * sizeof(float);
    static constexpr uint32_t WORK_UB_BASE = OI_UB_BASE + OI_UB_SIZE;

    // AIC L1 consumer buffer for V2C pij pipe (full QT * MAX_BLOCK_SIZE rows)
    static constexpr uint32_t PIJ_L1_BASE = 0x40000;
    static constexpr uint32_t PIJ_L1_SIZE = 2 * QT * MAX_BLOCK_SIZE * sizeof(bfloat16_t);
};

// ============================================================================
// AIC (Cube) processing — QK-first offset-loop software pipeline
//
// QK-first order: each steady-state iteration does QK[i] then PV[i-1].
// This maximizes overlap by hiding AIV's softmax behind AIC's QK matmul:
// while AIC computes QK[i], AIV concurrently processes SF[i-1].
// By the time AIC finishes QK[i] and needs pij[i-1], SF[i-1] is done.
// FIFO_DEPTH=2 supports the 2-deep sij buffering (sij[i-1] + sij[i]).
//
// Timeline (steady state):
//   AIC:  QK[i] → TPUSH(sij[i]) → TPOP(pij[i-1]) → PV[i-1] → TPUSH(oi[i-1])
//   AIV:  TPOP(sij[i-1]) → SF[i-1] → TPUSH(pij[i-1]) → TPOP(oi[i-2]) → UP[i-2]
//   ──────────────────────────────────────────────────────────────────────────
//   QK[i] overlaps with SF[i-1]   (Cube compute ∥ Vector softmax)
//   PV[i-1] overlaps with UP[i-2] (Cube compute ∥ Vector online update)
// ============================================================================

// Helper: QK matmul for block i — load key, move to L0, matmul, TPUSH sij
template <
    int M, int K, int N, typename SijPipeT, typename GlobalB_QK, typename TileMatA_QK, typename TileMatB_QK,
    typename LeftTile_QK, typename RightTile_QK, typename AccTile_QK>
static __aicore__ void aic_qk_step(
    __gm__ bfloat16_t *key_base, __gm__ int32_t *bt, uint64_t bt_offset, uint64_t i, TileMatA_QK &aMatTile_QK,
    TileMatB_QK &bMatTile_QK_A, TileMatB_QK &bMatTile_QK_B, LeftTile_QK &aTile_QK, RightTile_QK &bTile_QK,
    AccTile_QK &cTile_QK, SijPipeT &sij_pipe
) {
    GlobalB_QK kjGlobal(key_base + static_cast<uint64_t>(bt[bt_offset + i]) * N * K);
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

    // TPUSH sij (C2V): AccTile L0C -> GM. Ensure prior MTE3 is done,
    // then push, then wait for MTE3 DMA to complete before signaling consumer.
    TPUSH<SijPipeT, AccTile_QK, TileSplitAxis::TILE_UP_DOWN>(sij_pipe, cTile_QK);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    sij_pipe.prod.record();
}

// Helper: PV matmul for block i — TPOP pij, load value, move to L0, matmul, TPUSH oi
template <
    int M, int K, int N, typename PijPipeT, typename OiPipeT, typename GlobalB_PV, typename PijMatTile,
    typename TileMatB_PV, typename LeftTile_PV, typename RightTile_PV, typename AccTile_PV>
static __aicore__ void aic_pv_step(
    __gm__ bfloat16_t *val_base, __gm__ int32_t *bt, uint64_t bt_offset, uint64_t i, PijMatTile &pijMatTile,
    TileMatB_PV &bMatTile_PV_A, TileMatB_PV &bMatTile_PV_B, LeftTile_PV &aTile_PV, RightTile_PV &bTile_PV,
    AccTile_PV &cTile_PV, PijPipeT &pij_pipe, OiPipeT &oi_pipe
) {
    GlobalB_PV vjGlobal(val_base + static_cast<uint64_t>(bt[bt_offset + i]) * N * K);
    if (i % 2 == 0) {
        TLOAD(bMatTile_PV_A, vjGlobal);
    } else {
        TLOAD(bMatTile_PV_B, vjGlobal);
    }

    TPOP<PijPipeT, PijMatTile, TileSplitAxis::TILE_NO_SPLIT>(pij_pipe, pijMatTile);

    // PV step uses EVENT_ID1 (QK step uses EVENT_ID0) to avoid flag aliasing
    // when pipe_barrier(PIPE_ALL) is removed between steps.
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

    TMOV(aTile_PV, pijMatTile);
    if (i % 2 == 0) {
        TMOV(bTile_PV, bMatTile_PV_A);
    } else {
        TMOV(bTile_PV, bMatTile_PV_B);
    }

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

    TMATMUL(cTile_PV, aTile_PV, bTile_PV);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID1);

    // TPUSH oi (C2V): AccTile L0C -> GM. Same manual record pattern as sij.
    TPUSH<OiPipeT, AccTile_PV, TileSplitAxis::TILE_UP_DOWN>(oi_pipe, cTile_PV);
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    oi_pipe.prod.record();
}

template <typename Cfg, int K, int N>
static __aicore__ void aic_process_blocks(
    __gm__ bfloat16_t *qi_base, __gm__ bfloat16_t *key_base, __gm__ bfloat16_t *val_base, __gm__ int32_t *bt,
    uint64_t bt_offset, uint64_t n_blocks, typename Cfg::SijPipeT &sij_pipe, typename Cfg::PijPipeT &pij_pipe,
    typename Cfg::OiPipeT &oi_pipe
) {
    constexpr int M = Cfg::Q_TILE;
    using SijPipeT = typename Cfg::SijPipeT;
    using PijPipeT = typename Cfg::PijPipeT;
    using OiPipeT = typename Cfg::OiPipeT;

    using GlobalA_QK = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB_QK = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using TileMatA_QK = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB_QK = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;
    using LeftTile_QK = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile_QK = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile_QK = TileAcc<float, M, N, M, N>;

    using GlobalB_PV = GlobalTensor<bfloat16_t, Shape<1, 1, 1, N, K>, Stride<N * K, N * K, N * K, K, 1>>;
    using TileMatB_PV = Tile<TileType::Mat, bfloat16_t, N, K, BLayout::ColMajor, N, K, SLayout::RowMajor, 512>;
    using PijMatTile = Tile<TileType::Mat, bfloat16_t, M, N, BLayout::ColMajor, M, N, SLayout::RowMajor, 512>;
    using LeftTile_PV = TileLeft<bfloat16_t, M, N, M, N>;
    using RightTile_PV = TileRight<bfloat16_t, N, K, N, K>;
    using AccTile_PV = TileAcc<float, M, K, M, K>;

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

    PijMatTile pijMatTile;
    TileMatB_PV bMatTile_PV_A, bMatTile_PV_B;
    TASSIGN(bMatTile_PV_A, Cfg::PIJ_L1_BASE + Cfg::PIJ_L1_SIZE);
    TASSIGN(bMatTile_PV_B, Cfg::PIJ_L1_BASE + Cfg::PIJ_L1_SIZE + kPVBBytes);

    LeftTile_PV aTile_PV;
    RightTile_PV bTile_PV;
    AccTile_PV cTile_PV;
    TASSIGN(aTile_PV, 0x0);
    TASSIGN(bTile_PV, 0x0);
    TASSIGN(cTile_PV, 0x0);

    GlobalA_QK qiGlobal(qi_base);
    TLOAD(aMatTile_QK, qiGlobal);

    if (n_blocks == 1) {
        // Degenerate case: no pipeline overlap possible
        aic_qk_step<M, K, N, SijPipeT, GlobalB_QK>(
            key_base, bt, bt_offset, 0, aMatTile_QK, bMatTile_QK_A, bMatTile_QK_B, aTile_QK, bTile_QK, cTile_QK,
            sij_pipe
        );
        aic_pv_step<M, K, N, PijPipeT, OiPipeT, GlobalB_PV>(
            val_base, bt, bt_offset, 0, pijMatTile, bMatTile_PV_A, bMatTile_PV_B, aTile_PV, bTile_PV, cTile_PV,
            pij_pipe, oi_pipe
        );
    } else {
        // Prologue: QK[0] — produces sij[0] for AIV to start SF[0]
        aic_qk_step<M, K, N, SijPipeT, GlobalB_QK>(
            key_base, bt, bt_offset, 0, aMatTile_QK, bMatTile_QK_A, bMatTile_QK_B, aTile_QK, bTile_QK, cTile_QK,
            sij_pipe
        );
        // Steady state: QK[i] then PV[i-1] (QK-first order).
        for (uint64_t i = 1; i < n_blocks; i++) {
            aic_qk_step<M, K, N, SijPipeT, GlobalB_QK>(
                key_base, bt, bt_offset, i, aMatTile_QK, bMatTile_QK_A, bMatTile_QK_B, aTile_QK, bTile_QK, cTile_QK,
                sij_pipe
            );
            aic_pv_step<M, K, N, PijPipeT, OiPipeT, GlobalB_PV>(
                val_base, bt, bt_offset, i - 1, pijMatTile, bMatTile_PV_A, bMatTile_PV_B, aTile_PV, bTile_PV, cTile_PV,
                pij_pipe, oi_pipe
            );
        }

        // Epilogue: PV[n-1] — consume last pij
        aic_pv_step<M, K, N, PijPipeT, OiPipeT, GlobalB_PV>(
            val_base, bt, bt_offset, n_blocks - 1, pijMatTile, bMatTile_PV_A, bMatTile_PV_B, aTile_PV, bTile_PV,
            cTile_PV, pij_pipe, oi_pipe
        );
    }
}

// ============================================================================
// AIV (Vector) processing — SF-first offset-loop software pipeline
//
// SF-first order: each steady-state iteration does SF[i] then UP[i-1].
// This ensures pij[i] is produced as early as possible so AIC's TPOP(pij)
// never stalls behind a pending UP computation. Combined with AIC's
// QK-first order, SF[i] overlaps with AIC's PV[i-1] Cube matmul.
// ============================================================================

// Helper: softmax step for block i — TPOP sij, compute softmax, TPUSH pij
//
// globalMaxRow is used as a running accumulator: on entry it holds the max
// from the previous iteration (or is undefined when i==0). SF updates it
// in-place to max(globalMaxRow, localMaxRow_i * scale). The caller must
// save globalMaxRow before calling SF if the old value is still needed.
template <
    typename Cfg, int TM, int TN, typename SijVecTile, typename TileSijPad, typename TileVecMxN,
    typename PijVecBf16Tile, typename TileScalarDN, typename TileScalarRow>
static __aicore__ void aiv_sf_step(
    uint64_t i, bool is_last_partial, uint64_t valid_len_last, float scale_value, SijVecTile &sijTile,
    TileSijPad &sijPadTile, TileVecMxN &pijTile, TileVecMxN &tmpTile, PijVecBf16Tile &pijBf16Tile,
    TileScalarDN &localMaxDN, TileScalarDN &globalMaxDN, TileScalarDN &llDN, TileScalarRow &localMaxRow,
    TileScalarRow &globalMaxRow, typename Cfg::SijPipeT &sij_pipe, typename Cfg::PijPipeT &pij_pipe
) {
    using TileSijDyn = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, -1>;
    using SijPipeT = typename Cfg::SijPipeT;
    using PijPipeT = typename Cfg::PijPipeT;

    TPOP<SijPipeT, SijVecTile, TileSplitAxis::TILE_UP_DOWN>(sij_pipe, sijTile);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (is_last_partial) {
        int sij_addr = Cfg::SIJ_UB_BASE + static_cast<int>((i % 2) * TM * TN * static_cast<int>(sizeof(float)));
        TASSIGN(sijPadTile, sij_addr);
        TileSijDyn sijDynTile(static_cast<size_t>(valid_len_last));
        TASSIGN(sijDynTile, sij_addr);
        TFILLPAD_INPLACE(sijPadTile, sijDynTile);
        pipe_barrier(PIPE_V);
    }

    TROWMAX(localMaxDN, sijTile, tmpTile);
    pipe_barrier(PIPE_V);
    TRESHAPE(localMaxRow, localMaxDN);

    if (i == 0) {
        TMULS(globalMaxRow, localMaxRow, scale_value);
    } else {
        TMULS(localMaxRow, localMaxRow, scale_value);
        pipe_barrier(PIPE_V);
        TMAX(globalMaxRow, globalMaxRow, localMaxRow);
    }
    TRESHAPE(globalMaxDN, globalMaxRow);

    TMULS(sijTile, sijTile, scale_value);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(pijTile, sijTile, globalMaxDN);
    pipe_barrier(PIPE_V);
    TEXP(pijTile, pijTile);
    pipe_barrier(PIPE_V);

    TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
    pipe_barrier(PIPE_V);
    TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);
    pipe_barrier(PIPE_V);

    TROWSUM(llDN, pijTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TPUSH<PijPipeT, PijVecBf16Tile, TileSplitAxis::TILE_UP_DOWN>(pij_pipe, pijBf16Tile);
}

// Helper: online update step for block i — TPOP oi, merge with accumulators
//
// curMaxRow  = M[i]   = running max over blocks 0..i   (mij in FlashAttention notation)
// prevMaxRow = M[i-1] = running max over blocks 0..i-1 (dm / old max)
// llDN_i     = row-sum of pij for block i
//
// alpha = exp(prevMaxRow - curMaxRow), used to rescale accumulated go and gl.
template <
    typename Cfg, int TM, int TN, typename OiVecTile, typename TileDataMxHD, typename TileScalarDN,
    typename TileScalarND, typename TileScalarRow>
static __aicore__ void aiv_up_step(
    uint64_t i, OiVecTile &oiNewTile, TileDataMxHD &goTile, TileScalarDN &alphaDN_dn, TileScalarDN &llDN_i,
    TileScalarND &glND, TileScalarND &alphaND, TileScalarND &llND, TileScalarND &dmND, TileScalarND &mijND,
    TileScalarRow &curMaxRow, TileScalarRow &prevMaxRow, typename Cfg::OiPipeT &oi_pipe
) {
    using OiPipeT = typename Cfg::OiPipeT;
    TPOP<OiPipeT, OiVecTile, TileSplitAxis::TILE_UP_DOWN>(oi_pipe, oiNewTile);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

    if (i == 0) {
        TMULS(goTile, oiNewTile, 1.0f);
        TRESHAPE(llND, llDN_i);
        pipe_barrier(PIPE_V);
        TMULS(glND, llND, 1.0f);
    } else {
        TRESHAPE(llND, llDN_i);
        TRESHAPE(mijND, curMaxRow);
        TRESHAPE(dmND, prevMaxRow);

        TSUB(alphaND, dmND, mijND);
        pipe_barrier(PIPE_V);
        TEXP(alphaND, alphaND);
        pipe_barrier(PIPE_V);

        TRESHAPE(alphaDN_dn, alphaND);
        TROWEXPANDMUL(goTile, goTile, alphaDN_dn);
        pipe_barrier(PIPE_V);
        TADD(goTile, goTile, oiNewTile);

        TMUL(glND, glND, alphaND);
        pipe_barrier(PIPE_V);
        TADD(glND, glND, llND);
    }

    pipe_barrier(PIPE_V);
}

template <typename Cfg, int TN>
static __aicore__ void aiv_process_blocks(
    float scale_value, uint64_t n_blocks, uint64_t valid_len_last, __gm__ float *dst_ptr,
    typename Cfg::SijPipeT &sij_pipe, typename Cfg::PijPipeT &pij_pipe, typename Cfg::OiPipeT &oi_pipe
) {
    constexpr int TM = Cfg::SUB_QT;
    constexpr int HD = HEAD_DIM;
    constexpr int kAlignedRows = ((TM * sizeof(float) + 31) / 32) * (32 / sizeof(float));
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = TM / kScalarCols;

    using SijVecTile = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;
    using PijVecBf16Tile = Tile<TileType::Vec, bfloat16_t, TM, TN, BLayout::RowMajor, TM, TN>;
    using OiVecTile = Tile<TileType::Vec, float, TM, HD, BLayout::RowMajor, TM, HD>;

    using TileVecMxN = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;
    using TileSijPad =
        Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN, SLayout::NoneBox, 512, PadValue::Min>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, TM, 1>;
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;
    using TileScalarRow = Tile<TileType::Vec, float, 1, TM, BLayout::RowMajor, 1, TM>;
    using TileDataMxHD = Tile<TileType::Vec, float, TM, HD, BLayout::RowMajor, TM, HD>;
    using GlobalDataMxHD = GlobalTensor<float, Shape<1, 1, 1, TM, HD>, Stride<1, 1, 1, HD, 1>>;

    constexpr int kSijBytes = TM * TN * sizeof(float);
    constexpr int kPijBf16Bytes = TM * TN * sizeof(bfloat16_t);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);
    constexpr int kScalarNDBytes = kScalarRows * kScalarCols * sizeof(float);

    SijVecTile sijTile;
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    PijVecBf16Tile pijBf16Tile;
    TileScalarDN localMaxDN, globalMaxDN;
    TileScalarDN alphaDN_dn, llDN, glDN;
    TileScalarDN savedLlDN;
    TileScalarND gmND, glND, alphaND, llND, dmND, miNewND, mijND;
    TileScalarRow localMaxRow, globalMaxRow;
    TileScalarRow savedMaxRow, prevMaxRow;
    OiVecTile oiNewTile;
    TileDataMxHD goTile;

    int ub = Cfg::WORK_UB_BASE;
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
    TASSIGN(gmND, sb);
    TASSIGN(savedMaxRow, sb);
    sb += kScalarDNBytes;
    TASSIGN(glND, sb);
    TASSIGN(glDN, sb);
    sb += kScalarDNBytes;
    TASSIGN(alphaND, sb);
    TASSIGN(alphaDN_dn, sb);
    sb += kScalarDNBytes;
    TASSIGN(llND, sb);
    TASSIGN(llDN, sb);
    sb += kScalarDNBytes;
    TASSIGN(dmND, sb);
    sb += kScalarNDBytes;
    TASSIGN(miNewND, sb);
    sb += kScalarNDBytes;
    TASSIGN(mijND, sb);
    sb += kScalarNDBytes;
    TASSIGN(prevMaxRow, sb);
    sb += kScalarDNBytes;
    TASSIGN(savedLlDN, sb);
    sb += kScalarDNBytes;

    TASSIGN(goTile, sb);

    GlobalDataMxHD dstGlobal(dst_ptr);

    bool last_partial = (valid_len_last < static_cast<uint64_t>(TN));

    if (n_blocks == 1) {
        aiv_sf_step<Cfg, TM, TN>(
            0, last_partial, valid_len_last, scale_value, sijTile, sijPadTile, pijTile, tmpTile, pijBf16Tile,
            localMaxDN, globalMaxDN, llDN, localMaxRow, globalMaxRow, sij_pipe, pij_pipe
        );
        aiv_up_step<Cfg, TM, TN>(
            0, oiNewTile, goTile, alphaDN_dn, llDN, glND, alphaND, llND, dmND, mijND, globalMaxRow, globalMaxRow,
            oi_pipe
        );
    } else {
        // Prologue: SF[0] — not the last block
        aiv_sf_step<Cfg, TM, TN>(
            0, false, valid_len_last, scale_value, sijTile, sijPadTile, pijTile, tmpTile, pijBf16Tile, localMaxDN,
            globalMaxDN, llDN, localMaxRow, globalMaxRow, sij_pipe, pij_pipe
        );

        // Steady state: SF[i] then UP[i-1] (SF-first order).
        for (uint64_t i = 1; i < n_blocks; i++) {
            // Shift max history: prevMaxRow ← savedMaxRow (M[i-2])
            // Save current: savedMaxRow ← globalMaxRow (M[i-1])
            TMULS(prevMaxRow, savedMaxRow, 1.0f);
            TMULS(savedMaxRow, globalMaxRow, 1.0f);
            TMULS(savedLlDN, llDN, 1.0f);
            pipe_barrier(PIPE_V);

            bool cur_last_partial = (i == n_blocks - 1) && last_partial;
            aiv_sf_step<Cfg, TM, TN>(
                i, cur_last_partial, valid_len_last, scale_value, sijTile, sijPadTile, pijTile, tmpTile, pijBf16Tile,
                localMaxDN, globalMaxDN, llDN, localMaxRow, globalMaxRow, sij_pipe, pij_pipe
            );

            aiv_up_step<Cfg, TM, TN>(
                i - 1, oiNewTile, goTile, alphaDN_dn, savedLlDN, glND, alphaND, llND, dmND, mijND, savedMaxRow,
                prevMaxRow, oi_pipe
            );
        }

        // Epilogue: UP[n-1] — uses live globalMaxRow (M[n-1]) and savedMaxRow (M[n-2])
        aiv_up_step<Cfg, TM, TN>(
            n_blocks - 1, oiNewTile, goTile, alphaDN_dn, llDN, glND, alphaND, llND, dmND, mijND, globalMaxRow,
            savedMaxRow, oi_pipe
        );
    }

    // Final normalization: output = goTile / glDN
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
// Per-config dispatch: builds pipes from per-hw-block FIFO bases, then runs
// the AIC or AIV stride loop over total_logical_blocks.
// ============================================================================

template <typename Cfg>
static __aicore__ void run_aic(
    __gm__ int64_t *args, __gm__ int32_t *ctx_ptr, int32_t hw_block_idx, int32_t hw_block_num,
    int64_t total_logical_blocks, int64_t num_heads, int64_t head_dim, int64_t block_size, int64_t max_blocks_per_req,
    int64_t q_loop, __gm__ void *sij_fifo_base, __gm__ void *pij_fifo_base, __gm__ void *oi_fifo_base
) {
    typename Cfg::SijPipeT sij_pipe(sij_fifo_base, Cfg::SIJ_UB_BASE, 0U);
    typename Cfg::PijPipeT pij_pipe(pij_fifo_base, 0U, Cfg::PIJ_L1_BASE);
    typename Cfg::OiPipeT oi_pipe(oi_fifo_base, Cfg::OI_UB_BASE, 0U);

    // Disable auto-record on C2V pipes: AccTile TSTORE goes through FIX → MTE3,
    // but auto-record fires on PIPE_FIX which may complete before MTE3 DMA writes
    // to GM. Manual pipe_barrier(PIPE_MTE3) + record() in each step ensures the
    // cross-core signal fires only after the GM write is visible.
    sij_pipe.prod.setRecordStatus(false);
    oi_pipe.prod.setRecordStatus(false);

    // Disable reverse-dependency sync (back-pressure). Forward dependency chain
    // (AIC: QK-first; AIV: SF-first; FIFO_DEPTH=2) guarantees producer is at
    // most SLOT_NUM=2 tiles ahead of consumer:
    //   sij: AIC pushes sij[i+1] only after TPOP(pij[i-1]), which requires
    //        AIV TPOP(sij[i-1]) — slot reuse safe.
    //   oi : AIC pushes oi[i+1]  only after TPOP(pij[i+1]), which requires
    //        AIV's iter i+1 SF, by which time AIV iter i finished UP[i-1]
    //        i.e. TPOP(oi[i-1]) — slot reuse safe.
    //   pij: AIV pushes pij[i+1] only after TPOP(sij[i+1]), which fires after
    //        AIC iter i+1 starts and AIC iter i has finished TPOP(pij[i-1])
    //        — slot reuse safe.
    // If the QK-first/SF-first interleaving or FIFO_DEPTH changes, restore
    // these flags.
    sij_pipe.prod.setAllocateStatus(false);
    oi_pipe.prod.setAllocateStatus(false);
    pij_pipe.cons.setFreeStatus(false);

    __gm__ Tensor *query_t = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *key_cache_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *value_cache_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *block_table_t = reinterpret_cast<__gm__ Tensor *>(args[3]);

    __gm__ bfloat16_t *query_base = reinterpret_cast<__gm__ bfloat16_t *>(query_t->buffer.addr) + query_t->start_offset;
    __gm__ bfloat16_t *key_base =
        reinterpret_cast<__gm__ bfloat16_t *>(key_cache_t->buffer.addr) + key_cache_t->start_offset;
    __gm__ bfloat16_t *val_base =
        reinterpret_cast<__gm__ bfloat16_t *>(value_cache_t->buffer.addr) + value_cache_t->start_offset;
    __gm__ int32_t *bt = reinterpret_cast<__gm__ int32_t *>(block_table_t->buffer.addr) + block_table_t->start_offset;

    for (int32_t block_idx = hw_block_idx; block_idx < total_logical_blocks; block_idx += hw_block_num) {
        int64_t batch_idx = block_idx / q_loop;
        int64_t q_tile_idx = block_idx % q_loop;

        int64_t cur_seq = static_cast<int64_t>(ctx_ptr[batch_idx]);
        int64_t n_blocks = (cur_seq + block_size - 1) / block_size;
        if (n_blocks <= 0) continue;

        int64_t q_offset = (batch_idx * num_heads + q_tile_idx * Cfg::Q_TILE) * head_dim;
        __gm__ bfloat16_t *qi_base = query_base + q_offset;
        uint64_t bt_offset = static_cast<uint64_t>(batch_idx * max_blocks_per_req);

        if (block_size == 128) {
            aic_process_blocks<Cfg, 128, 128>(
                qi_base, key_base, val_base, bt, bt_offset, static_cast<uint64_t>(n_blocks), sij_pipe, pij_pipe, oi_pipe
            );
        } else {
            aic_process_blocks<Cfg, 128, 64>(
                qi_base, key_base, val_base, bt, bt_offset, static_cast<uint64_t>(n_blocks), sij_pipe, pij_pipe, oi_pipe
            );
        }
    }
}

template <typename Cfg>
static __aicore__ void run_aiv(
    __gm__ int64_t *args, __gm__ int32_t *ctx_ptr, int32_t hw_block_idx, int32_t hw_block_num,
    int64_t total_logical_blocks, int64_t num_heads, int64_t head_dim, int64_t block_size, int64_t q_loop,
    __gm__ void *sij_fifo_base, __gm__ void *pij_fifo_base, __gm__ void *oi_fifo_base
) {
    typename Cfg::SijPipeT sij_pipe(sij_fifo_base, Cfg::SIJ_UB_BASE, 0U);
    typename Cfg::PijPipeT pij_pipe(pij_fifo_base, 0U, Cfg::PIJ_L1_BASE);
    typename Cfg::OiPipeT oi_pipe(oi_fifo_base, Cfg::OI_UB_BASE, 0U);

    __gm__ Tensor *out_t = reinterpret_cast<__gm__ Tensor *>(args[5]);
    float scale_value = from_u64<float>(static_cast<uint64_t>(args[9]));

    int32_t sub_block_id = get_sub_block_id(args);
    int64_t row_offset = sub_block_id * Cfg::SUB_QT;

    // Entry offsets depend on the actual tile width (block_size for sij/pij, HEAD_DIM for oi).
    // TILE_UP_DOWN splits Q_TILE rows into two SUB_QT halves; AIV1's data starts at
    // SUB_QT * tile_width * sizeof(element) within the contiguous TPUSH'd tile.
    int sij_sub_offset = sub_block_id * Cfg::SUB_QT * static_cast<int>(block_size) * static_cast<int>(sizeof(float));
    int pij_sub_offset =
        sub_block_id * Cfg::SUB_QT * static_cast<int>(block_size) * static_cast<int>(sizeof(bfloat16_t));
    int oi_sub_offset = sub_block_id * Cfg::SUB_QT * HEAD_DIM * static_cast<int>(sizeof(float));
    sij_pipe.cons.setEntryOffset(sij_sub_offset);
    pij_pipe.prod.setEntryOffset(pij_sub_offset);
    oi_pipe.cons.setEntryOffset(oi_sub_offset);

    // Mirror reverse-dependency disable on the AIV side (see run_aic for
    // the full forward-chain argument).
    pij_pipe.prod.setAllocateStatus(false);
    sij_pipe.cons.setFreeStatus(false);
    oi_pipe.cons.setFreeStatus(false);

    __gm__ float *out_base = reinterpret_cast<__gm__ float *>(out_t->buffer.addr) + out_t->start_offset;

    for (int32_t block_idx = hw_block_idx; block_idx < total_logical_blocks; block_idx += hw_block_num) {
        int64_t batch_idx = block_idx / q_loop;
        int64_t q_tile_idx = block_idx % q_loop;

        int64_t cur_seq = static_cast<int64_t>(ctx_ptr[batch_idx]);
        int64_t n_blocks = (cur_seq + block_size - 1) / block_size;

        int64_t out_offset = (batch_idx * num_heads + q_tile_idx * Cfg::Q_TILE + row_offset) * head_dim;
        __gm__ float *dst = out_base + out_offset;

        if (n_blocks <= 0) {
            using ZeroTile =
                Tile<TileType::Vec, float, Cfg::SUB_QT, HEAD_DIM, BLayout::RowMajor, Cfg::SUB_QT, HEAD_DIM>;
            using ZeroGlobal = GlobalTensor<float, Shape<1, 1, 1, Cfg::SUB_QT, HEAD_DIM>, Stride<1, 1, 1, HEAD_DIM, 1>>;
            ZeroTile zeroTile;
            TASSIGN(zeroTile, Cfg::WORK_UB_BASE);
            TEXPANDS(zeroTile, 0.0f);
            pipe_barrier(PIPE_V);
            ZeroGlobal dstZero(dst);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(dstZero, zeroTile);
            pipe_barrier(PIPE_MTE3);
            continue;
        }

        int64_t last_block_seq = (n_blocks - 1) * block_size;
        int64_t remaining = cur_seq - last_block_seq;
        uint64_t valid_len_last = (remaining >= block_size) ? static_cast<uint64_t>(block_size) :
                                                              (remaining > 0 ? static_cast<uint64_t>(remaining) : 0);

        if (block_size == 128) {
            aiv_process_blocks<Cfg, 128>(
                scale_value, static_cast<uint64_t>(n_blocks), valid_len_last, dst, sij_pipe, pij_pipe, oi_pipe
            );
        } else {
            aiv_process_blocks<Cfg, 64>(
                scale_value, static_cast<uint64_t>(n_blocks), valid_len_last, dst, sij_pipe, pij_pipe, oi_pipe
            );
        }
    }
}

// ============================================================================
// Entry point — shared by AIC and AIV via DAV_CUBE / DAV_VEC guards
// ============================================================================

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *context_lens_t = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *sij_fifo_t = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ Tensor *pij_fifo_t = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ Tensor *oi_fifo_t = reinterpret_cast<__gm__ Tensor *>(args[8]);

    int64_t num_heads = static_cast<int64_t>(args[10]);
    int64_t head_dim = static_cast<int64_t>(args[11]);
    int64_t block_size = static_cast<int64_t>(args[12]);
    int64_t max_blocks_per_req = static_cast<int64_t>(args[13]);
    int64_t q_loop = static_cast<int64_t>(args[14]);
    int64_t total_logical_blocks = static_cast<int64_t>(args[15]);
    int64_t q_tile = static_cast<int64_t>(args[16]);

    int32_t hw_block_idx = get_block_idx(args);
    int32_t hw_block_num = get_block_num(args);

    __gm__ int32_t *ctx_ptr =
        reinterpret_cast<__gm__ int32_t *>(context_lens_t->buffer.addr) + context_lens_t->start_offset;

    // GM FIFO buffer per hardware block (reused across stride-loop iterations).
    // Slot stride is sized for max(Q_TILE) so the same offset works for both q_tile=16 and 64.
    constexpr uint32_t SIJ_HW_STRIDE = PAConfig<MAX_Q_TILE>::SIJ_SLOT_SIZE * FIFO_DEPTH;
    constexpr uint32_t PIJ_HW_STRIDE = PAConfig<MAX_Q_TILE>::PIJ_SLOT_SIZE * FIFO_DEPTH;
    constexpr uint32_t OI_HW_STRIDE = PAConfig<MAX_Q_TILE>::OI_SLOT_SIZE * FIFO_DEPTH;

    __gm__ void *sij_fifo_base = reinterpret_cast<__gm__ void *>(
        reinterpret_cast<__gm__ uint8_t *>(sij_fifo_t->buffer.addr) + hw_block_idx * SIJ_HW_STRIDE
    );
    __gm__ void *pij_fifo_base = reinterpret_cast<__gm__ void *>(
        reinterpret_cast<__gm__ uint8_t *>(pij_fifo_t->buffer.addr) + hw_block_idx * PIJ_HW_STRIDE
    );
    __gm__ void *oi_fifo_base = reinterpret_cast<__gm__ void *>(
        reinterpret_cast<__gm__ uint8_t *>(oi_fifo_t->buffer.addr) + hw_block_idx * OI_HW_STRIDE
    );

    if constexpr (DAV_CUBE) {
        if (q_tile == 16) {
            run_aic<PAConfig<16>>(
                args, ctx_ptr, hw_block_idx, hw_block_num, total_logical_blocks, num_heads, head_dim, block_size,
                max_blocks_per_req, q_loop, sij_fifo_base, pij_fifo_base, oi_fifo_base
            );
        } else {
            run_aic<PAConfig<MAX_Q_TILE>>(
                args, ctx_ptr, hw_block_idx, hw_block_num, total_logical_blocks, num_heads, head_dim, block_size,
                max_blocks_per_req, q_loop, sij_fifo_base, pij_fifo_base, oi_fifo_base
            );
        }
    }

    if constexpr (DAV_VEC) {
        if (q_tile == 16) {
            run_aiv<PAConfig<16>>(
                args, ctx_ptr, hw_block_idx, hw_block_num, total_logical_blocks, num_heads, head_dim, block_size,
                q_loop, sij_fifo_base, pij_fifo_base, oi_fifo_base
            );
        } else {
            run_aiv<PAConfig<MAX_Q_TILE>>(
                args, ctx_ptr, hw_block_idx, hw_block_num, total_logical_blocks, num_heads, head_dim, block_size,
                q_loop, sij_fifo_base, pij_fifo_base, oi_fifo_base
            );
        }
    }
}
// NOLINTEND(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
