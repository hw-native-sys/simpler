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
 * Paged Attention MIX Kernel — AIC + AIV with FFTS cross-core sync
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
 * AIC and AIV cooperate via 3 GM workspace buffers + FFTS cross-core sync flags
 * (one set per hardware block, ×2 ping-pong to avoid data races):
 *   - s_ws: QK scores    (Q_TILE, block_size) fp32
 *   - p_ws: softmax probs (Q_TILE, block_size) bf16
 *   - o_ws: PV output    (Q_TILE, head_dim)   fp32
 *
 * Per-block pipeline (QK-first on AIC, SF-first on AIV):
 *   AIC: QK[i] → signal QK_READY → wait SF_READY → PV[i] → signal UP_READY
 *   AIV: wait QK_READY → SF[i] → signal SF_READY → wait UP_READY → UP[i]
 *
 * MixedKernels args:
 *   args[0]  = query         Tensor* (batch*num_heads, head_dim) bf16
 *   args[1]  = key_cache     Tensor* (kv_total_rows, head_dim) bf16
 *   args[2]  = value_cache   Tensor* (kv_total_rows, head_dim) bf16
 *   args[3]  = block_table   Tensor* (batch, max_blocks_per_req) int32
 *   args[4]  = context_lens  Tensor* (batch,) int32
 *   args[5]  = out           Tensor* (batch*num_heads, head_dim) float32 [output]
 *   args[6]  = s_ws          Tensor* GM workspace for QK scores
 *   args[7]  = p_ws          Tensor* GM workspace for softmax probs
 *   args[8]  = o_ws          Tensor* GM workspace for PV output
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

#include "tensor.h"

using pto::BLayout;
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

// FFTS cross-core sync flag IDs
static constexpr uint16_t QK_READY = 0;
static constexpr uint16_t SF_READY = 1;
static constexpr uint16_t UP_READY = 2;

// FFTS helpers (from highperf utils)
static __aicore__ inline void WaitFlagDev(uint16_t flagId) { wait_flag_dev(flagId); }

template <pipe_t pipe, uint8_t mode>
static __aicore__ inline void FftsCrossCoreSync(uint16_t flagId) {
    uint64_t config = 1ULL | (static_cast<uint64_t>(mode) << 4) | (static_cast<uint64_t>(flagId) << 8);
    ffts_cross_core_sync(pipe, config);
}

template <int V>
static constexpr int RoundUp16 = ((V) + 15) & ~15;

// Per-q_tile compile-time configuration: workspace sizes, UB/L1 layouts.
// QT must be 16 or 64. SUB_QT = QT / 2 (each of AIV0/AIV1 handles half the rows).
template <int QT>
struct PAConfig {
    static constexpr int Q_TILE = QT;
    static constexpr int SUB_QT = QT / 2;

    // GM workspace sizes per stage (sized for max block_size)
    static constexpr uint32_t S_WS_SIZE = QT * MAX_BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t P_WS_SIZE = QT * MAX_BLOCK_SIZE * sizeof(bfloat16_t);
    static constexpr uint32_t O_WS_SIZE = QT * HEAD_DIM * sizeof(float);

    // AIV UB buffer layout (sized for SUB_QT rows per AIV lane)
    static constexpr uint32_t SIJ_UB_BASE = 0x0;
    static constexpr uint32_t SIJ_UB_SIZE = 2 * SUB_QT * MAX_BLOCK_SIZE * sizeof(float);
    static constexpr uint32_t OI_UB_BASE = SIJ_UB_BASE + SIJ_UB_SIZE;
    static constexpr uint32_t OI_UB_SIZE = 2 * SUB_QT * HEAD_DIM * sizeof(float);
    static constexpr uint32_t WORK_UB_BASE = OI_UB_BASE + OI_UB_SIZE;

    // AIC L1 buffer for P (softmax probs from GM) — loaded via TLOAD from GM
    static constexpr uint32_t PIJ_L1_BASE = 0x40000;
    static constexpr uint32_t PIJ_L1_SIZE = 2 * QT * MAX_BLOCK_SIZE * sizeof(bfloat16_t);
};

// ============================================================================
// AIC (Cube) processing — QK-first offset-loop software pipeline with FFTS sync
//
// QK-first order: each steady-state iteration does QK[i] then PV[i-1].
// This maximizes overlap by hiding AIV's softmax behind AIC's QK matmul:
// while AIC computes QK[i], AIV concurrently processes SF[i-1].
//
// Timeline (steady state):
//   AIC:  QK[i] → signal(QK_READY) → wait(SF_READY) → PV[i-1] → signal(UP_READY)
//   AIV:  wait(QK_READY) → SF[i-1] → signal(SF_READY) → wait(UP_READY) → UP[i-2]
// ============================================================================

// Helper: begin K block load from GM→L1 into the ping or pong buffer (non-blocking)
template <int K, int N, typename GlobalB_QK, typename TileMatB_QK>
static __aicore__ void aic_start_k_load(
    __gm__ bfloat16_t *key_base, __gm__ int32_t *bt, uint64_t bt_offset, uint64_t i, TileMatB_QK &bMatTile_QK_ping,
    TileMatB_QK &bMatTile_QK_pong, uint64_t ping_flag
) {
    GlobalB_QK kjGlobal(key_base + static_cast<uint64_t>(bt[bt_offset + i]) * N * K);
    if (ping_flag == 0) {
        TLOAD(bMatTile_QK_ping, kjGlobal);
    } else {
        TLOAD(bMatTile_QK_pong, kjGlobal);
    }
}

// Helper: complete QK — wait for K load, move Q+K to L0, matmul, write L0C→GM via fixpipe.
template <
    int M, int K, int N, typename TileMatA_QK, typename TileMatB_QK, typename LeftTile_QK, typename RightTile_QK,
    typename AccTile_QK>
static __aicore__ void aic_qk_compute_to_gm(
    TileMatA_QK &aMatTile_QK, TileMatB_QK &bMatTile_QK_cur, LeftTile_QK &aTile_QK, RightTile_QK &bTile_QK,
    AccTile_QK &cTile_QK, __gm__ float *s_ws_dst, int block_size
) {
    // Wait for K GM→L1 (and any prior Q GM→L1) to complete
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    // Q L1→L0A: Q stays in L1, reused across all KV blocks
    TMOV(aTile_QK, aMatTile_QK);
    // K L1→L0B: from current ping-pong buffer
    TMOV(bTile_QK, bMatTile_QK_cur);

    // Gate matmul on both L1→L0 moves completing
    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID1);

    TMATMUL(cTile_QK, aTile_QK, bTile_QK);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID3);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID3);

    // Write L0C→GM via fixpipe (replaces TPUSH)
    set_nd_para(1ULL);
    pipe_barrier(PIPE_FIX);
    copy_matrix_cc_to_gm(
        s_ws_dst,
        reinterpret_cast<__cc__ float *>(0x0),  // L0C address (TASSIGN'd)
        0,                                      // sid
        static_cast<uint16_t>(block_size),      // nSize
        static_cast<uint16_t>(M),               // mSize
        static_cast<uint16_t>(block_size),      // dstStride
        static_cast<uint16_t>(RoundUp16<M>),    // srcStride (NZ format)
        0,                                      // unitFlag
        QuantMode_t::NoQuant, 0, false, true
    );

    // Ensure fixpipe DMA to GM is complete before signaling
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

// Helper: begin V block load from GM→L1 (non-blocking)
template <int K, int N, typename GlobalB_PV, typename TileMatB_PV>
static __aicore__ void aic_start_v_load(
    __gm__ bfloat16_t *val_base, __gm__ int32_t *bt, uint64_t bt_offset, uint64_t i, TileMatB_PV &bMatTile_PV_ping,
    TileMatB_PV &bMatTile_PV_pong, uint64_t ping_flag
) {
    GlobalB_PV vjGlobal(val_base + static_cast<uint64_t>(bt[bt_offset + i]) * N * K);
    if (ping_flag == 0) {
        TLOAD(bMatTile_PV_ping, vjGlobal);
    } else {
        TLOAD(bMatTile_PV_pong, vjGlobal);
    }
}

// Helper: complete PV — load p_ws from GM→L1, wait for V, matmul, write L0C→GM
template <
    int M, int K, int N, typename PijGlobal, typename PijMatTile, typename TileMatB_PV, typename LeftTile_PV,
    typename RightTile_PV, typename AccTile_PV>
static __aicore__ void aic_pv_compute_to_gm(
    PijMatTile &pijMatTile, TileMatB_PV &bMatTile_PV_cur, LeftTile_PV &aTile_PV, RightTile_PV &bTile_PV,
    AccTile_PV &cTile_PV, __gm__ bfloat16_t *p_ws_src, __gm__ float *o_ws_dst
) {
    // Load P from GM workspace into L1
    PijGlobal pijGlobal(p_ws_src);
    TLOAD(pijMatTile, pijGlobal);

    // Wait for V GM→L1 to complete
    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID4);

    // P L1→L0A
    TMOV(aTile_PV, pijMatTile);
    // V L1→L0B from current ping-pong buffer
    TMOV(bTile_PV, bMatTile_PV_cur);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID5);

    TMATMUL(cTile_PV, aTile_PV, bTile_PV);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID6);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID6);

    // Write L0C→GM via fixpipe (replaces TPUSH)
    set_nd_para(1ULL);
    pipe_barrier(PIPE_FIX);
    copy_matrix_cc_to_gm(
        o_ws_dst,
        reinterpret_cast<__cc__ float *>(0x0),  // L0C address (TASSIGN'd)
        0,                                      // sid
        static_cast<uint16_t>(K),               // nSize = HEAD_DIM
        static_cast<uint16_t>(M),               // mSize = Q_TILE
        static_cast<uint16_t>(K),               // dstStride
        static_cast<uint16_t>(RoundUp16<M>),    // srcStride (NZ format)
        0,                                      // unitFlag
        QuantMode_t::NoQuant, 0, false, true
    );

    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

template <typename Cfg, int K, int N>
static __aicore__ void aic_process_blocks(
    __gm__ bfloat16_t *qi_base, __gm__ bfloat16_t *key_base, __gm__ bfloat16_t *val_base, __gm__ int32_t *bt,
    uint64_t bt_offset, uint64_t n_blocks, int64_t block_size, __gm__ float *s_ws, __gm__ bfloat16_t *p_ws,
    __gm__ float *o_ws
) {
    constexpr int M = Cfg::Q_TILE;

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
    using PijGlobal = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;
    using LeftTile_PV = TileLeft<bfloat16_t, M, N, M, N>;
    using RightTile_PV = TileRight<bfloat16_t, N, K, N, K>;
    using AccTile_PV = TileAcc<float, M, K, M, K>;

    constexpr int kQKBBytes = K * N * static_cast<int>(sizeof(bfloat16_t));
    constexpr int kPVBBytes = N * K * static_cast<int>(sizeof(bfloat16_t));

    // L1 buffer layout: Q at 0x0, K ping/pong at 0x20000
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

    // L1 buffer layout: pij at PIJ_L1_BASE, V ping/pong after
    PijMatTile pijMatTile;
    TileMatB_PV bMatTile_PV_A, bMatTile_PV_B;
    TASSIGN(pijMatTile, Cfg::PIJ_L1_BASE);
    TASSIGN(bMatTile_PV_A, Cfg::PIJ_L1_BASE + Cfg::PIJ_L1_SIZE);
    TASSIGN(bMatTile_PV_B, Cfg::PIJ_L1_BASE + Cfg::PIJ_L1_SIZE + kPVBBytes);

    LeftTile_PV aTile_PV;
    RightTile_PV bTile_PV;
    AccTile_PV cTile_PV;
    TASSIGN(aTile_PV, 0x0);
    TASSIGN(bTile_PV, 0x0);
    TASSIGN(cTile_PV, 0x0);

    // Q loaded once to L1 — reused across all KV blocks within this logical block
    GlobalA_QK qiGlobal(qi_base);
    TLOAD(aMatTile_QK, qiGlobal);

    // Ping-pong GM workspace: consecutive blocks alternate between stage 0/1
    // to avoid data races (AIC writing QK[i] while AIV reads QK[i-1]).
    constexpr uint32_t S_STAGE_ELEMS = Cfg::S_WS_SIZE / sizeof(float);
    constexpr uint32_t P_STAGE_ELEMS = Cfg::P_WS_SIZE / sizeof(bfloat16_t);
    constexpr uint32_t O_STAGE_ELEMS = Cfg::O_WS_SIZE / sizeof(float);

    // Inline QK/PV steps — lambdas cannot be used because the AICore compiler
    // does not propagate __aicore__ context into lambda bodies.

// clang-format off
#define DO_QK(idx)                                                                                      \
    do {                                                                                                \
        uint64_t _i = (idx);                                                                            \
        __gm__ float *_s_ws = s_ws + (_i % 2) * S_STAGE_ELEMS;                                         \
        aic_start_k_load<K, N, GlobalB_QK>(key_base, bt, bt_offset, _i,                                \
                                           bMatTile_QK_A, bMatTile_QK_B, _i % 2);                      \
        if (_i % 2 == 0) {                                                                              \
            aic_qk_compute_to_gm<M, K, N>(aMatTile_QK, bMatTile_QK_A,                                  \
                                           aTile_QK, bTile_QK, cTile_QK, _s_ws,                        \
                                           static_cast<int>(block_size));                               \
        } else {                                                                                        \
            aic_qk_compute_to_gm<M, K, N>(aMatTile_QK, bMatTile_QK_B,                                  \
                                           aTile_QK, bTile_QK, cTile_QK, _s_ws,                        \
                                           static_cast<int>(block_size));                               \
        }                                                                                               \
        FftsCrossCoreSync<PIPE_FIX, 2>(QK_READY);                                                      \
    } while (0)

#define DO_PV(idx)                                                                                      \
    do {                                                                                                \
        uint64_t _i = (idx);                                                                            \
        __gm__ bfloat16_t *_p_ws = p_ws + (_i % 2) * P_STAGE_ELEMS;                                    \
        __gm__ float *_o_ws = o_ws + (_i % 2) * O_STAGE_ELEMS;                                         \
        WaitFlagDev(SF_READY);                                                                          \
        aic_start_v_load<K, N, GlobalB_PV>(val_base, bt, bt_offset, _i,                                \
                                           bMatTile_PV_A, bMatTile_PV_B, _i % 2);                      \
        if (_i % 2 == 0) {                                                                              \
            aic_pv_compute_to_gm<M, K, N, PijGlobal>(pijMatTile, bMatTile_PV_A,                        \
                                                      aTile_PV, bTile_PV, cTile_PV, _p_ws, _o_ws);     \
        } else {                                                                                        \
            aic_pv_compute_to_gm<M, K, N, PijGlobal>(pijMatTile, bMatTile_PV_B,                        \
                                                      aTile_PV, bTile_PV, cTile_PV, _p_ws, _o_ws);     \
        }                                                                                               \
        FftsCrossCoreSync<PIPE_FIX, 2>(UP_READY);                                                      \
    } while (0)
// clang-format on

    if (n_blocks == 1) {
        DO_QK(0);
        DO_PV(0);
    } else {
        // Prologue: QK[0]
        DO_QK(0);
        // Steady state: QK[i] then PV[i-1] (QK-first order)
        for (uint64_t i = 1; i < n_blocks; i++) {
            DO_QK(i);
            DO_PV(i - 1);
        }
        // Epilogue: PV[n-1]
        DO_PV(n_blocks - 1);
    }

#undef DO_QK
#undef DO_PV
}

// ============================================================================
// AIV (Vector) processing — SF-first offset-loop software pipeline with FFTS sync
//
// SF-first order: each steady-state iteration does SF[i] then UP[i-1].
// This ensures pij[i] is produced as early as possible so AIC's wait(SF_READY)
// never stalls behind a pending UP computation.
// ============================================================================

// Helper: softmax step for block i — read sij from GM, compute softmax, write pij to GM
template <
    typename Cfg, int TM, int TN, typename SijVecTile, typename TileSijPad, typename TileVecMxN,
    typename PijVecBf16Tile, typename TileScalarDN, typename TileScalarRow>
static __aicore__ void aiv_sf_step(
    uint64_t i, bool is_last_partial, uint64_t valid_len_last, float scale_value, SijVecTile &sijTile,
    TileSijPad &sijPadTile, TileVecMxN &pijTile, TileVecMxN &tmpTile, PijVecBf16Tile &pijBf16Tile,
    TileScalarDN &localMaxDN, TileScalarDN &globalMaxDN, TileScalarDN &llDN, TileScalarRow &localMaxRow,
    TileScalarRow &globalMaxRow, __gm__ float *s_ws_src, __gm__ bfloat16_t *p_ws_dst, int sij_ub_addr,
    int pij_bf16_ub_addr, int64_t block_size
) {
    using TileSijDyn = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, -1>;

    // Read sij from GM workspace to UB (replaces TPOP)
    uint16_t sij_nburst = static_cast<uint16_t>(TM);
    uint16_t sij_len_burst = static_cast<uint16_t>(block_size * sizeof(float) / 32);
    copy_gm_to_ubuf(reinterpret_cast<__ubuf__ float *>(sij_ub_addr), s_ws_src, 0, sij_nburst, sij_len_burst, 0, 0);

    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (is_last_partial) {
        int sij_addr = sij_ub_addr;
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

    // Write pij to GM workspace (replaces TPUSH)
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    uint16_t pij_nburst = static_cast<uint16_t>(TM);
    uint16_t pij_len_burst = static_cast<uint16_t>(block_size * sizeof(bfloat16_t) / 32);
    copy_ubuf_to_gm(
        p_ws_dst, reinterpret_cast<__ubuf__ bfloat16_t *>(pij_bf16_ub_addr), 0, pij_nburst, pij_len_burst, 0, 0
    );

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

// Helper: online update step for block i — read oi from GM, merge with accumulators
template <
    typename Cfg, int TM, int TN, typename OiVecTile, typename TileDataMxHD, typename TileScalarDN,
    typename TileScalarND, typename TileScalarRow>
static __aicore__ void aiv_up_step(
    uint64_t i, OiVecTile &oiNewTile, TileDataMxHD &goTile, TileScalarDN &alphaDN_dn, TileScalarDN &llDN_i,
    TileScalarND &glND, TileScalarND &alphaND, TileScalarND &llND, TileScalarND &dmND, TileScalarND &mijND,
    TileScalarRow &curMaxRow, TileScalarRow &prevMaxRow, __gm__ float *o_ws_src, int oi_ub_addr
) {
    // Read oi_new from GM workspace to UB (replaces TPOP)
    constexpr int HD = HEAD_DIM;
    uint16_t oi_nburst = static_cast<uint16_t>(TM);
    uint16_t oi_len_burst = static_cast<uint16_t>(HD * sizeof(float) / 32);
    copy_gm_to_ubuf(reinterpret_cast<__ubuf__ float *>(oi_ub_addr), o_ws_src, 0, oi_nburst, oi_len_burst, 0, 0);

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
    float scale_value, uint64_t n_blocks, uint64_t valid_len_last, __gm__ float *dst_ptr, __gm__ float *s_ws,
    __gm__ bfloat16_t *p_ws, __gm__ float *o_ws, int32_t sub_block_id, int64_t block_size
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

    // SIJ and OI use ping-pong UB addressing (reuse across iterations via i%2)
    int sij_ub_addr = Cfg::SIJ_UB_BASE;
    TASSIGN(sijTile, sij_ub_addr);
    TASSIGN(sijPadTile, sij_ub_addr);

    int oi_ub_addr = Cfg::OI_UB_BASE;
    TASSIGN(oiNewTile, oi_ub_addr);

    int ub = Cfg::WORK_UB_BASE;
    TASSIGN(pijTile, ub);
    ub += kSijBytes;
    int pij_bf16_ub_addr = ub;
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

    // AIV sub-block offset: AIV0 reads top half, AIV1 reads bottom half
    uint32_t sij_sub_offset = static_cast<uint32_t>(sub_block_id) * TM * static_cast<uint32_t>(block_size);
    uint32_t pij_sub_offset = static_cast<uint32_t>(sub_block_id) * TM * static_cast<uint32_t>(block_size);
    uint32_t oi_sub_offset = static_cast<uint32_t>(sub_block_id) * TM * HD;

    __gm__ float *s_ws_sub = s_ws + sij_sub_offset;
    __gm__ bfloat16_t *p_ws_sub = p_ws + pij_sub_offset;
    __gm__ float *o_ws_sub = o_ws + oi_sub_offset;

    // Ping-pong GM workspace stage offsets (in elements, for sub-block pointers)
    constexpr uint32_t S_STAGE_ELEMS = Cfg::S_WS_SIZE / sizeof(float);
    constexpr uint32_t P_STAGE_ELEMS = Cfg::P_WS_SIZE / sizeof(bfloat16_t);
    constexpr uint32_t O_STAGE_ELEMS = Cfg::O_WS_SIZE / sizeof(float);

    // Inline SF/UP steps — lambdas cannot be used because the AICore compiler
    // does not propagate __aicore__ context into lambda bodies.

// clang-format off
#define DO_SF(idx)                                                                                      \
    do {                                                                                                \
        uint64_t _i = (idx);                                                                            \
        bool cur_last_partial = (_i == n_blocks - 1) && last_partial;                                   \
        __gm__ float *_s = s_ws_sub + (_i % 2) * S_STAGE_ELEMS;                                        \
        __gm__ bfloat16_t *_p = p_ws_sub + (_i % 2) * P_STAGE_ELEMS;                                   \
        WaitFlagDev(QK_READY);                                                                          \
        aiv_sf_step<Cfg, TM, TN>(                                                                      \
            _i, cur_last_partial, valid_len_last, scale_value, sijTile, sijPadTile, pijTile, tmpTile,   \
            pijBf16Tile, localMaxDN, globalMaxDN, llDN, localMaxRow, globalMaxRow, _s, _p,              \
            sij_ub_addr, pij_bf16_ub_addr, block_size);                                                \
        FftsCrossCoreSync<PIPE_MTE3, 2>(SF_READY);                                                     \
    } while (0)

#define DO_UP(idx, curMax, prevMax, ll)                                                                 \
    do {                                                                                                \
        uint64_t _i = (idx);                                                                            \
        __gm__ float *_o = o_ws_sub + (_i % 2) * O_STAGE_ELEMS;                                        \
        WaitFlagDev(UP_READY);                                                                          \
        aiv_up_step<Cfg, TM, TN>(                                                                      \
            _i, oiNewTile, goTile, alphaDN_dn, ll, glND, alphaND, llND, dmND, mijND, curMax, prevMax,  \
            _o, oi_ub_addr);                                                                           \
    } while (0)
// clang-format on

    if (n_blocks == 1) {
        DO_SF(0);
        DO_UP(0, globalMaxRow, globalMaxRow, llDN);
    } else {
        // Prologue: SF[0]
        DO_SF(0);

        // Steady state: SF[i] then UP[i-1] (SF-first order)
        for (uint64_t i = 1; i < n_blocks; i++) {
            // Shift max history
            TMULS(prevMaxRow, savedMaxRow, 1.0f);
            TMULS(savedMaxRow, globalMaxRow, 1.0f);
            TMULS(savedLlDN, llDN, 1.0f);
            pipe_barrier(PIPE_V);

            DO_SF(i);
            DO_UP(i - 1, savedMaxRow, prevMaxRow, savedLlDN);
        }

        // Epilogue: UP[n-1]
        DO_UP(n_blocks - 1, globalMaxRow, savedMaxRow, llDN);
    }

#undef DO_SF
#undef DO_UP

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
// Per-config dispatch: computes per-hw-block workspace pointers, then runs
// the AIC or AIV stride loop over total_logical_blocks.
// ============================================================================

template <typename Cfg>
static __aicore__ void run_aic(
    __gm__ int64_t *args, __gm__ int32_t *ctx_ptr, int32_t hw_block_idx, int32_t hw_block_num,
    int64_t total_logical_blocks, int64_t num_heads, int64_t head_dim, int64_t block_size, int64_t max_blocks_per_req,
    int64_t q_loop, __gm__ float *s_ws_base, __gm__ bfloat16_t *p_ws_base, __gm__ float *o_ws_base
) {
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
                qi_base, key_base, val_base, bt, bt_offset, static_cast<uint64_t>(n_blocks), block_size, s_ws_base,
                p_ws_base, o_ws_base
            );
        } else {
            aic_process_blocks<Cfg, 128, 64>(
                qi_base, key_base, val_base, bt, bt_offset, static_cast<uint64_t>(n_blocks), block_size, s_ws_base,
                p_ws_base, o_ws_base
            );
        }
    }
}

template <typename Cfg>
static __aicore__ void run_aiv(
    __gm__ int64_t *args, __gm__ int32_t *ctx_ptr, int32_t hw_block_idx, int32_t hw_block_num,
    int64_t total_logical_blocks, int64_t num_heads, int64_t head_dim, int64_t block_size, int64_t q_loop,
    __gm__ float *s_ws_base, __gm__ bfloat16_t *p_ws_base, __gm__ float *o_ws_base
) {
    __gm__ Tensor *out_t = reinterpret_cast<__gm__ Tensor *>(args[5]);
    float scale_value = from_u64<float>(static_cast<uint64_t>(args[9]));

    int32_t sub_block_id = get_sub_block_id(args);
    int64_t row_offset = sub_block_id * Cfg::SUB_QT;

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
                scale_value, static_cast<uint64_t>(n_blocks), valid_len_last, dst, s_ws_base, p_ws_base, o_ws_base,
                sub_block_id, block_size
            );
        } else {
            aiv_process_blocks<Cfg, 64>(
                scale_value, static_cast<uint64_t>(n_blocks), valid_len_last, dst, s_ws_base, p_ws_base, o_ws_base,
                sub_block_id, block_size
            );
        }
    }
}

// ============================================================================
// Entry point — shared by AIC and AIV via DAV_CUBE / DAV_VEC guards
// ============================================================================

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *context_lens_t = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *s_ws_t = reinterpret_cast<__gm__ Tensor *>(args[6]);
    __gm__ Tensor *p_ws_t = reinterpret_cast<__gm__ Tensor *>(args[7]);
    __gm__ Tensor *o_ws_t = reinterpret_cast<__gm__ Tensor *>(args[8]);

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

    // Per-hw-block workspace (×2 for ping-pong between consecutive iterations)
    constexpr uint32_t S_WS_HW_STRIDE = PAConfig<MAX_Q_TILE>::S_WS_SIZE * 2;
    constexpr uint32_t P_WS_HW_STRIDE = PAConfig<MAX_Q_TILE>::P_WS_SIZE * 2;
    constexpr uint32_t O_WS_HW_STRIDE = PAConfig<MAX_Q_TILE>::O_WS_SIZE * 2;

    __gm__ float *s_ws_base = reinterpret_cast<__gm__ float *>(
        reinterpret_cast<__gm__ uint8_t *>(s_ws_t->buffer.addr) + hw_block_idx * S_WS_HW_STRIDE
    );
    __gm__ bfloat16_t *p_ws_base = reinterpret_cast<__gm__ bfloat16_t *>(
        reinterpret_cast<__gm__ uint8_t *>(p_ws_t->buffer.addr) + hw_block_idx * P_WS_HW_STRIDE
    );
    __gm__ float *o_ws_base = reinterpret_cast<__gm__ float *>(
        reinterpret_cast<__gm__ uint8_t *>(o_ws_t->buffer.addr) + hw_block_idx * O_WS_HW_STRIDE
    );

    if constexpr (DAV_CUBE) {
        if (q_tile == 16) {
            run_aic<PAConfig<16>>(
                args, ctx_ptr, hw_block_idx, hw_block_num, total_logical_blocks, num_heads, head_dim, block_size,
                max_blocks_per_req, q_loop, s_ws_base, p_ws_base, o_ws_base
            );
        } else {
            run_aic<PAConfig<MAX_Q_TILE>>(
                args, ctx_ptr, hw_block_idx, hw_block_num, total_logical_blocks, num_heads, head_dim, block_size,
                max_blocks_per_req, q_loop, s_ws_base, p_ws_base, o_ws_base
            );
        }
    }

    if constexpr (DAV_VEC) {
        if (q_tile == 16) {
            run_aiv<PAConfig<16>>(
                args, ctx_ptr, hw_block_idx, hw_block_num, total_logical_blocks, num_heads, head_dim, block_size,
                q_loop, s_ws_base, p_ws_base, o_ws_base
            );
        } else {
            run_aiv<PAConfig<MAX_Q_TILE>>(
                args, ctx_ptr, hw_block_idx, hw_block_num, total_logical_blocks, num_heads, head_dim, block_size,
                q_loop, s_ws_base, p_ws_base, o_ws_base
            );
        }
    }
}
// NOLINTEND(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
