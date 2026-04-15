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
// SPMD SplitK PV Matmul: Accumulated P @ V across n_blocks
//
// SPMD block_idx encodes (batch_idx, q_tile_idx).
// Each SPMD block processes n_blocks using SplitK accumulation:
//   Block 0: TMATMUL(C, A, B)       — initialize accumulator
//   Block i: TMATMUL_ACC(C, C, A, B) — accumulate into same C
//
// Per-block pij: contiguous packed (M, K) tiles in pij_buf
// Per-block vj: value_cache base + block_table lookup
// Single output: oi_new (M, N) fp32 = sum of P_i @ V_i across all blocks
//
// Case1: (16, 128) @ (128, 128) -> (16, 128)
// Case2: (64,  64) @ ( 64, 128) -> (64, 128)
//
// Args:
//   args[0] = pij          Tensor* (spmd_blocks*Q_TILE, n_blocks*block_size) bf16
//   args[1] = value_cache  Tensor* (kv_total_rows, head_dim) bf16
//   args[2] = block_table  Tensor* (batch, max_blocks_per_req) int32
//   args[3] = context_lens Tensor* (batch,) int32
//   args[4] = oi_new       Tensor* (spmd_blocks*Q_TILE, head_dim) float32 [output]
//   args[5] = bn_start     scalar: starting KV block index
//   args[6] = n_blocks     scalar: number of KV blocks to process
//   args[7] = num_heads    scalar
//   args[8] = head_dim     scalar
//   args[9] = block_size   scalar
//   args[10] = max_num_blocks_per_req scalar
//   args[11] = q_loop      scalar

#include <cstdint>
// NOLINTBEGIN(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-avoid-c-arrays,modernize-use-auto)
#include <pto/pto-inst.hpp>

#include "tensor.h"

// NOLINTNEXTLINE(build/namespaces)
using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

#include "intrinsic.h"

template <int M, int K, int N>
static __aicore__ void pv_matmul_n_spmd(
    __gm__ bfloat16_t *pij_base, __gm__ bfloat16_t *val_base, __gm__ float *oi_base, uint64_t n_blocks,
    __gm__ int32_t *bt, uint64_t bt_offset
) {
    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // L1 memory layout: double-buffered A and B tiles
    constexpr int kATileBytes = M * K * static_cast<int>(sizeof(bfloat16_t));
    constexpr int kBTileBytes = K * N * static_cast<int>(sizeof(bfloat16_t));

    TileMatA aMatTile[2];
    TileMatB bMatTile[2];
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], kATileBytes);
    TASSIGN(bMatTile[0], 2 * kATileBytes);
    TASSIGN(bMatTile[1], 2 * kATileBytes + kBTileBytes);

    // L0 memory layout: double-buffered L0A and L0B, single accumulator L0C
    LeftTile aTile[2];
    RightTile bTile[2];
    AccTile cTile;
    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], kATileBytes);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], kBTileBytes);
    TASSIGN(cTile, 0x0);

    GlobalOut oiGlobal(oi_base);

    // Seed reverse-dependency flags
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    for (uint64_t i = 0; i < n_blocks; i++) {
        int cur = static_cast<int>(i % 2);
        GlobalA pijGlobal(pij_base + i * M * K);
        GlobalB vjGlobal(val_base + bt[bt_offset + i] * K * N);

        // Stage 1: TLOAD (MTE2: GM -> L1[cur])
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)cur);
        TLOAD(aMatTile[cur], pijGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TLOAD(bMatTile[cur], vjGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);

        // Stage 2: TMOV (MTE1: L1[cur] -> L0[cur])
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)cur);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        TMOV(aTile[cur], aMatTile[cur]);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);
        TMOV(bTile[cur], bMatTile[cur]);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)cur);

        // Stage 3: TMATMUL (M-pipe: L0A[cur] x L0B[cur] -> L0C)
        set_flag(PIPE_MTE1, PIPE_M, (event_t)cur);
        wait_flag(PIPE_MTE1, PIPE_M, (event_t)cur);
        if (i == 0) {
            TMATMUL(cTile, aTile[cur], bTile[cur]);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile[cur], bTile[cur]);
        }
        set_flag(PIPE_M, PIPE_MTE1, (event_t)cur);
    }

    // Drain outstanding reverse-dependency flags
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

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

    int64_t bn_start = static_cast<int64_t>(args[5]);
    int64_t n_blocks_total = static_cast<int64_t>(args[6]);
    int64_t num_heads = static_cast<int64_t>(args[7]);
    int64_t head_dim = static_cast<int64_t>(args[8]);
    int64_t block_size = static_cast<int64_t>(args[9]);
    int64_t max_blocks_per_req = static_cast<int64_t>(args[10]);
    int64_t q_loop = static_cast<int64_t>(args[11]);

    int32_t block_idx = get_block_idx(args);
    int64_t batch_idx = block_idx / q_loop;

    int64_t q_tile = (block_size == 128) ? 16 : 64;

    // Check how many KV blocks this batch actually has
    __gm__ int32_t *ctx_ptr =
        reinterpret_cast<__gm__ int32_t *>(context_lens_t->buffer.addr) + context_lens_t->start_offset;
    int64_t cur_seq = static_cast<int64_t>(ctx_ptr[batch_idx]);
    int64_t bn_this_batch = (cur_seq + block_size - 1) / block_size;

    // Clamp n_blocks to valid range for this batch
    int64_t valid_blocks = bn_this_batch - bn_start;
    if (valid_blocks < 0) valid_blocks = 0;
    int64_t n_blocks = (valid_blocks < n_blocks_total) ? valid_blocks : n_blocks_total;

    // Output pointer for this SPMD block's oi_new slice
    __gm__ float *oi_base =
        reinterpret_cast<__gm__ float *>(oi_new_t->buffer.addr) + oi_new_t->start_offset + block_idx * q_tile * head_dim;

    if (n_blocks <= 0) {
        for (int64_t i = 0; i < q_tile * head_dim; i++) {
            oi_base[i] = 0.0f;
        }
        return;
    }

    // pij packed tiles: block_idx's region in pij tensor
    __gm__ bfloat16_t *pij_base =
        reinterpret_cast<__gm__ bfloat16_t *>(pij_t->buffer.addr) + pij_t->start_offset +
        block_idx * q_tile * n_blocks_total * block_size;

    // Value cache base
    __gm__ bfloat16_t *val_base =
        reinterpret_cast<__gm__ bfloat16_t *>(value_cache_t->buffer.addr) + value_cache_t->start_offset;

    // Block table pointer
    __gm__ int32_t *bt =
        reinterpret_cast<__gm__ int32_t *>(block_table_t->buffer.addr) + block_table_t->start_offset;
    uint64_t bt_offset = static_cast<uint64_t>(batch_idx * max_blocks_per_req + bn_start);

    if (q_tile == 16) {
        pv_matmul_n_spmd<16, 128, 128>(
            pij_base, val_base, oi_base, static_cast<uint64_t>(n_blocks), bt, bt_offset
        );
    } else {
        pv_matmul_n_spmd<64, 64, 128>(
            pij_base, val_base, oi_base, static_cast<uint64_t>(n_blocks), bt, bt_offset
        );
    }
}
// NOLINTEND(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-avoid-c-arrays,modernize-use-auto)
