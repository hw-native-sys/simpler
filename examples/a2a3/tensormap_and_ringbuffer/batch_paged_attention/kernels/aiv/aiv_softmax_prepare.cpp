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

// Batched Softmax Preparation Kernel (AIV)
//
// Processes batch_count batches in a single kernel invocation.
// For each batch b at block_idx bn:
//   valid_len = min(N, context_lens[b] - bn * N)
//   sij_masked = pad(sij[b], valid_len, -inf)
//   sij_scale  = sij_masked * scale
//   mij[b]     = row_max(sij_scale)
//   pij[b]     = exp(sij_scale - mij[b])  (truncated to bf16 then back)
//   lij[b]     = row_sum(pij[b])
//
// Supports three tile configurations via runtime dispatch:
//   Small: (16,  16)  -- q_tile=16, block_size=16   [truncate to fp16]
//   Case1: (16, 128)  -- q_tile=16, block_size=128  [truncate to bf16]
//   Case2: (64,  64)  -- q_tile=64, block_size=64   [truncate to bf16]

#include <cstdint>
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

template <typename T, int M, int N>
static __aicore__ void softmax_prepare_batch_impl(
    __gm__ Tensor *sij_batch, __gm__ Tensor *context_lens_t, __gm__ Tensor *pij_batch, __gm__ Tensor *mij_batch,
    __gm__ Tensor *lij_batch, float scale_value, uint64_t batch_count, uint64_t block_idx, uint64_t batch_start
) {
    __gm__ float *sij_base = reinterpret_cast<__gm__ float *>(sij_batch->buffer.addr);
    __gm__ T *pij_base = reinterpret_cast<__gm__ T *>(pij_batch->buffer.addr);
    __gm__ float *mij_base = reinterpret_cast<__gm__ float *>(mij_batch->buffer.addr);
    __gm__ float *lij_base = reinterpret_cast<__gm__ float *>(lij_batch->buffer.addr);
    __gm__ int32_t *ctx_lens = reinterpret_cast<__gm__ int32_t *>(context_lens_t->buffer.addr);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_T = GlobalTensor<T, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    using TileSijDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, -1>;
    using TileSijPad = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>;

    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_T = Tile<TileType::Vec, T, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    TileVecMxN sijTile;
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_T pijTruncTile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, M * N * sizeof(float));
    TASSIGN(tmpTile, 2 * M * N * sizeof(float));
    TASSIGN(maxTile, 3 * M * N * sizeof(float));
    TASSIGN(sumTile, 3 * M * N * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijTruncTile, 3 * M * N * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    for (uint64_t b = 0; b < batch_count; b++) {
        int32_t cur_seq = ctx_lens[batch_start + b];
        uint64_t start = block_idx * N;
        uint64_t valid_len = 0;
        if (start < static_cast<uint64_t>(cur_seq)) {
            uint64_t remaining = static_cast<uint64_t>(cur_seq) - start;
            valid_len = (remaining < N) ? remaining : N;
        }

        __gm__ float *sij_addr = sij_base + b * M * N;
        __gm__ T *pij_addr = pij_base + b * M * N;
        __gm__ float *mij_addr = mij_base + b * M;
        __gm__ float *lij_addr = lij_base + b * M;

        GlobalDataMxN sijGlobal(sij_addr);
        GlobalDataMxN_T pijGlobal(pij_addr);
        GlobalScalarDN mijGlobal(mij_addr);
        GlobalScalarDN lijGlobal(lij_addr);

        if (valid_len == 0) {
            // Block entirely beyond sequence: write mij=-1e30, lij=0, pij=0
            // Use -1e30 instead of -inf to avoid NaN in online_update (exp(-inf - (-inf)) = NaN)
            constexpr float NEG_LARGE = -1e30f;
            for (int i = 0; i < kAlignedRows; i++) {
                maxTile.SetValue(i, NEG_LARGE);
                sumTile.SetValue(i, 0.0f);
            }
            for (int i = 0; i < M * N; i++) {
                pijTruncTile.SetValue(i, static_cast<T>(0.0f));
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(mijGlobal, maxTile);
            TSTORE(lijGlobal, sumTile);
            TSTORE(pijGlobal, pijTruncTile);

            if (b + 1 < batch_count) {
                pipe_barrier(PIPE_ALL);
            }
            continue;
        }

        TLOAD(sijTile, sijGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
        TASSIGN(sijDynTile, 0x0);
        TFILLPAD_INPLACE(sijPadTile, sijDynTile);
        pipe_barrier(PIPE_V);

        TMULS(sijTile, sijTile, scale_value);
        pipe_barrier(PIPE_V);
        TROWMAX(maxTile, sijTile, tmpTile);
        pipe_barrier(PIPE_V);
        TROWEXPANDSUB(pijTile, sijTile, maxTile);
        pipe_barrier(PIPE_V);
        TEXP(pijTile, pijTile);
        pipe_barrier(PIPE_V);
        // Truncate pij to bf16 first, then compute lij from truncated values (matches golden)
        TCVT(pijTruncTile, pijTile, RoundMode::CAST_ROUND);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pipe_barrier(PIPE_V);
        TCVT(pijTile, pijTruncTile, RoundMode::CAST_ROUND);
        pipe_barrier(PIPE_V);
        TROWSUM(sumTile, pijTile, tmpTile);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);

        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(pijGlobal, pijTruncTile);
        TSTORE(mijGlobal, maxTile);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(lijGlobal, sumTile);

        if (b + 1 < batch_count) {
            pipe_barrier(PIPE_ALL);
        }
    }

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *sij_batch = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *context_lens_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *pij_batch = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *mij_batch = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *lij_batch = reinterpret_cast<__gm__ Tensor *>(args[4]);
    union {
        uint64_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[5]);
    float scale_value = scale_conv.f;
    uint64_t batch_count = static_cast<uint64_t>(args[6]);
    uint64_t block_idx = static_cast<uint64_t>(args[7]);
    uint64_t batch_start = static_cast<uint64_t>(args[8]);

    uint64_t q_tile_size = static_cast<uint64_t>(sij_batch->shapes[0] / batch_count);
    uint64_t block_size = static_cast<uint64_t>(sij_batch->shapes[1]);

    if (q_tile_size == 16 && block_size == 16) {
        softmax_prepare_batch_impl<half, 16, 16>(
            sij_batch, context_lens_t, pij_batch, mij_batch, lij_batch, scale_value, batch_count, block_idx, batch_start
        );
    } else if (q_tile_size == 16) {
        softmax_prepare_batch_impl<bfloat16_t, 16, 128>(
            sij_batch, context_lens_t, pij_batch, mij_batch, lij_batch, scale_value, batch_count, block_idx, batch_start
        );
    } else {
        softmax_prepare_batch_impl<bfloat16_t, 64, 64>(
            sij_batch, context_lens_t, pij_batch, mij_batch, lij_batch, scale_value, batch_count, block_idx, batch_start
        );
    }
}
