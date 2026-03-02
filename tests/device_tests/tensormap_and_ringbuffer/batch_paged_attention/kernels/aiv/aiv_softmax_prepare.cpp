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
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) -- q_tile=16, block_size=128
//   Case2: (64, 64)  -- q_tile=64, block_size=64

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

template <int M, int N>
static __aicore__ void softmax_prepare_batch_impl(
    __gm__ TensorData* sij_batch,
    __gm__ TensorData* pij_batch,
    __gm__ TensorData* mij_batch,
    __gm__ TensorData* lij_batch,
    float scale_value,
    uint64_t context_lens_ptr,
    uint64_t batch_count,
    uint64_t block_idx,
    uint64_t batch_start) {

    __gm__ float* sij_base = reinterpret_cast<__gm__ float*>(sij_batch->buffer.addr);
    __gm__ bfloat16_t* pij_base = reinterpret_cast<__gm__ bfloat16_t*>(pij_batch->buffer.addr);
    __gm__ float* mij_base = reinterpret_cast<__gm__ float*>(mij_batch->buffer.addr);
    __gm__ float* lij_base = reinterpret_cast<__gm__ float*>(lij_batch->buffer.addr);
    __gm__ int32_t* ctx_lens = reinterpret_cast<__gm__ int32_t*>(context_lens_ptr);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_bf16 = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    using TileSijDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, -1>;
    using TileSijPad = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>;

    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_bf16 = Tile<TileType::Vec, bfloat16_t, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    TileVecMxN sijTile;
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_bf16 pijBf16Tile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, M * N * sizeof(float));
    TASSIGN(tmpTile, 2 * M * N * sizeof(float));
    TASSIGN(maxTile, 3 * M * N * sizeof(float));
    TASSIGN(sumTile, 3 * M * N * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijBf16Tile, 3 * M * N * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    for (uint64_t b = 0; b < batch_count; b++) {
        int32_t cur_seq = ctx_lens[batch_start + b];
        uint64_t start = block_idx * N;
        uint64_t valid_len = 0;
        if (start < (uint64_t)cur_seq) {
            uint64_t remaining = (uint64_t)cur_seq - start;
            valid_len = (remaining < N) ? remaining : N;
        }

        __gm__ float* sij_addr = sij_base + b * M * N;
        __gm__ bfloat16_t* pij_addr = pij_base + b * M * N;
        __gm__ float* mij_addr = mij_base + b * M;
        __gm__ float* lij_addr = lij_base + b * M;

        GlobalDataMxN sijGlobal(sij_addr);
        GlobalDataMxN_bf16 pijGlobal(pij_addr);
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
                pijBf16Tile.SetValue(i, static_cast<bfloat16_t>(0.0f));
            }

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(mijGlobal, maxTile);
            TSTORE(lijGlobal, sumTile);
            TSTORE(pijGlobal, pijBf16Tile);

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

        TMULS(sijTile, sijTile, scale_value);
        pipe_barrier(PIPE_V);
        TROWMAX(maxTile, sijTile, tmpTile);
        pipe_barrier(PIPE_V);
        TROWEXPANDSUB(pijTile, sijTile, maxTile);
        pipe_barrier(PIPE_V);
        TEXP(pijTile, pijTile);
        // Truncate pij to bf16 first, then compute lij from truncated values (matches golden)
        TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
        TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);
        TROWSUM(sumTile, pijTile, tmpTile);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(mijGlobal, maxTile);
        TSTORE(lijGlobal, sumTile);
        TSTORE(pijGlobal, pijBf16Tile);

        if (b + 1 < batch_count) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* sij_batch = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* pij_batch = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* mij_batch = reinterpret_cast<__gm__ TensorData*>(args[2]);
    __gm__ TensorData* lij_batch = reinterpret_cast<__gm__ TensorData*>(args[3]);
    union { uint64_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[4]);
    float scale_value = scale_conv.f;
    uint64_t context_lens_ptr = static_cast<uint64_t>(args[5]);
    uint64_t batch_count = static_cast<uint64_t>(args[6]);
    uint64_t block_idx = static_cast<uint64_t>(args[7]);
    uint64_t batch_start = static_cast<uint64_t>(args[8]);

    uint64_t q_tile_size = static_cast<uint64_t>(sij_batch->shapes[0] / batch_count);

    if (q_tile_size == 16) {
        softmax_prepare_batch_impl<16, 128>(
            sij_batch, pij_batch, mij_batch, lij_batch,
            scale_value, context_lens_ptr, batch_count, block_idx, batch_start);
    } else {
        softmax_prepare_batch_impl<64, 64>(
            sij_batch, pij_batch, mij_batch, lij_batch,
            scale_value, context_lens_ptr, batch_count, block_idx, batch_start);
    }
}
