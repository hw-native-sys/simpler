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
// SPMD Multi-block QK Matmul: qi(M, K) @ kj.T(K, N) -> sij(M, N) for n_blocks
//
// SPMD block_idx encodes (batch_idx, q_tile_idx).
// Each SPMD block processes n_blocks consecutive KV blocks starting at bn_start,
// using double-buffered L1 B tiles and hoisted qi TLOAD.
//
// Output: packed tiles [tile_0, tile_1, ..., tile_{n-1}] each (M, N) in row-major.
//
// Template: M=q_tile, K=head_dim, N=block_size
// Case1: (16, 128) @ (128, 128).T -> (16, 128)
// Case2: (64, 128) @ (128,  64).T -> (64,  64)
//
// Args:
//   args[0] = query       Tensor* (batch*num_heads, head_dim) bf16
//   args[1] = key_cache   Tensor* (kv_total_rows, head_dim) bf16
//   args[2] = block_table Tensor* (batch, max_blocks_per_req) int32
//   args[3] = context_lens Tensor* (batch,) int32
//   args[4] = sij         Tensor* (spmd_blocks*Q_TILE, n_blocks*block_size) float32 [output]
//   args[5] = bn_start    scalar: starting KV block index
//   args[6] = n_blocks    scalar: number of KV blocks to process
//   args[7] = num_heads   scalar
//   args[8] = head_dim    scalar
//   args[9] = block_size  scalar
//   args[10] = max_num_blocks_per_req scalar
//   args[11] = q_loop     scalar

#include <cstdint>
// NOLINTBEGIN(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
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
static __aicore__ void qk_matmul_n_spmd(
    __gm__ bfloat16_t *qi_base, __gm__ bfloat16_t *key_base, __gm__ float *sij_base, uint64_t n_blocks,
    __gm__ int32_t *bt, uint64_t bt_offset
) {
    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // Double-buffered L1 B tiles for kj prefetching
    constexpr int kBBytes = K * N * static_cast<int>(sizeof(bfloat16_t));
    TileMatA aMatTile;
    TileMatB bMatTile_A;
    TileMatB bMatTile_B;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile_A, 0x20000);
    TASSIGN(bMatTile_B, 0x20000 + kBBytes);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    // Hoist qi TLOAD before the loop (qi is constant across all blocks)
    GlobalA qiGlobal(qi_base);
    TLOAD(aMatTile, qiGlobal);

    // Pre-load first kj into buffer A
    GlobalB kjGlobal_0(key_base + bt[bt_offset + 0] * N * K);
    TLOAD(bMatTile_A, kjGlobal_0);

    for (uint64_t i = 0; i < n_blocks; i++) {
        GlobalOut sijGlobal(sij_base + i * M * N);

        // Wait for current kj TLOAD to complete
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // TMOV qi L1->L0A and kj L1->L0B from current buffer
        TMOV(aTile, aMatTile);
        if (i % 2 == 0) {
            TMOV(bTile, bMatTile_A);
        } else {
            TMOV(bTile, bMatTile_B);
        }

        // Prefetch next kj into alternate L1 buffer
        if (i + 1 < n_blocks) {
            GlobalB kjGlobal_next(key_base + bt[bt_offset + i + 1] * N * K);
            if (i % 2 == 0) {
                TLOAD(bMatTile_B, kjGlobal_next);
            } else {
                TLOAD(bMatTile_A, kjGlobal_next);
            }
        }

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(cTile, aTile, bTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        TSTORE(sijGlobal, cTile);

        if (i + 1 < n_blocks) {
            pipe_barrier(PIPE_ALL);
        }
    }
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *query_t = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *key_cache_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *block_table_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *context_lens_t = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *sij_t = reinterpret_cast<__gm__ Tensor *>(args[4]);

    int64_t bn_start = static_cast<int64_t>(args[5]);
    int64_t n_blocks_total = static_cast<int64_t>(args[6]);
    int64_t num_heads = static_cast<int64_t>(args[7]);
    int64_t head_dim = static_cast<int64_t>(args[8]);
    int64_t block_size = static_cast<int64_t>(args[9]);
    int64_t max_blocks_per_req = static_cast<int64_t>(args[10]);
    int64_t q_loop = static_cast<int64_t>(args[11]);

    int32_t block_idx = get_block_idx(args);

    // Decode (batch_idx, q_tile_idx) from block_idx
    int64_t batch_idx = block_idx / q_loop;
    int64_t q_tile_idx = block_idx % q_loop;

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

    // sij packed tile output: block_idx's region starts at block_idx * q_tile * n_blocks_total * block_size
    __gm__ float *sij_base = reinterpret_cast<__gm__ float *>(sij_t->buffer.addr) + sij_t->start_offset +
                             block_idx * q_tile * n_blocks_total * block_size;

    if (n_blocks <= 0) {
        for (int64_t i = 0; i < q_tile * n_blocks_total * block_size; i++) {
            sij_base[i] = 0.0f;
        }
        return;
    }

    // Zero out trailing invalid tiles
    if (n_blocks < n_blocks_total) {
        __gm__ float *trail = sij_base + n_blocks * q_tile * block_size;
        for (int64_t i = 0; i < (n_blocks_total - n_blocks) * q_tile * block_size; i++) {
            trail[i] = 0.0f;
        }
    }

    // Query offset: (batch_idx * num_heads + q_tile_idx * q_tile, 0)
    int64_t q_offset = (batch_idx * num_heads + q_tile_idx * q_tile) * head_dim;
    __gm__ bfloat16_t *qi_base =
        reinterpret_cast<__gm__ bfloat16_t *>(query_t->buffer.addr) + query_t->start_offset + q_offset;

    // Key cache base
    __gm__ bfloat16_t *key_base =
        reinterpret_cast<__gm__ bfloat16_t *>(key_cache_t->buffer.addr) + key_cache_t->start_offset;

    // Block table pointer
    __gm__ int32_t *bt =
        reinterpret_cast<__gm__ int32_t *>(block_table_t->buffer.addr) + block_table_t->start_offset;
    uint64_t bt_offset = static_cast<uint64_t>(batch_idx * max_blocks_per_req + bn_start);

    if (q_tile == 16) {
        qk_matmul_n_spmd<16, 128, 128>(
            qi_base, key_base, sij_base, static_cast<uint64_t>(n_blocks), bt, bt_offset
        );
    } else {
        qk_matmul_n_spmd<64, 128, 64>(
            qi_base, key_base, sij_base, static_cast<uint64_t>(n_blocks), bt, bt_offset
        );
    }
}
// NOLINTEND(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
