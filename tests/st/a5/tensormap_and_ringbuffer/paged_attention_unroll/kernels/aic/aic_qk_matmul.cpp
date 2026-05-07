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
// Multi-block QK Matmul Kernel: qi(M, K) @ kj.T(K, N) -> sij(M, N) for each block
//
// Processes n_blocks blocks in a single kernel invocation.
// Per-block kj addresses computed from key_cache base + block_indices lookup.
// qi is shared across all blocks (same query head against different key blocks).
//
// Output layout: n_blocks contiguous (M, N) tiles stacked vertically.
// Block i occupies sij[i*M : (i+1)*M, 0:N].
//
// Optimizations:
//   - qi TLOAD hoisted before the loop (constant across all iterations)
//   - Double-buffered L1 B tiles: prefetch next kj during current TMATMUL+TSTORE
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128).T -> (16, 128)
//   Case2: (64, 128) @ (128,  64).T -> (64,  64)
//
// Template: M=q_tile, K=head_dim, N=block_size

#include <cstdint>
// NOLINTBEGIN(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
#include <pto/pto-inst.hpp>

#include "tensor.h"

// NOLINTNEXTLINE(build/namespaces)
using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

template <int M, int K, int N>
static __aicore__ void qk_matmul_n_impl(
    __gm__ bfloat16_t *qi_base, __gm__ bfloat16_t *key_base, __gm__ float *sij_base, uint64_t n_blocks,
    __gm__ int32_t *bt, uint64_t bt_offset
) {
    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

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

        // TMOV qi L1→L0A and kj L1→L0B from current buffer
        TMOV(aTile, aMatTile);
        if (i % 2 == 0) {
            TMOV(bTile, bMatTile_A);
        } else {
            TMOV(bTile, bMatTile_B);
        }

        // Prefetch next kj into alternate L1 buffer (overlaps with MTE1→M→FIX)
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
            // Drain all pipes before next iteration:
            //   - FIX/MTE3: ensures TSTORE data path (L0C→UB→GM) fully completes
            //   - MTE2: prefetch TLOAD likely already done (ran during TMATMUL+TSTORE)
            // The prefetch TLOAD overlaps with compute, so barrier cost is minimal.
            pipe_barrier(PIPE_ALL);
        }
    }
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *qi = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *key_cache = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *block_table_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *sij_buf = reinterpret_cast<__gm__ Tensor *>(args[3]);
    uint64_t n_blocks = static_cast<uint64_t>(args[4]);
    uint64_t bt_offset = static_cast<uint64_t>(args[5]);

    __gm__ bfloat16_t *qi_base = reinterpret_cast<__gm__ bfloat16_t *>(qi->buffer.addr) + qi->start_offset;
    __gm__ bfloat16_t *key_base = reinterpret_cast<__gm__ bfloat16_t *>(key_cache->buffer.addr);
    __gm__ float *sij_base = reinterpret_cast<__gm__ float *>(sij_buf->buffer.addr) + sij_buf->start_offset;
    __gm__ int32_t *bt = reinterpret_cast<__gm__ int32_t *>(block_table_t->buffer.addr);

    uint64_t q_tile_size = static_cast<uint64_t>(qi->shapes[0]);

    if (q_tile_size == 16) {
        qk_matmul_n_impl<16, 128, 128>(qi_base, key_base, sij_base, n_blocks, bt, bt_offset);
    } else {
        qk_matmul_n_impl<64, 128, 64>(qi_base, key_base, sij_base, n_blocks, bt, bt_offset);
    }
}

// NOLINTEND(clang-diagnostic-error,bugprone-reserved-identifier,bugprone-easily-swappable-parameters,modernize-use-auto)
