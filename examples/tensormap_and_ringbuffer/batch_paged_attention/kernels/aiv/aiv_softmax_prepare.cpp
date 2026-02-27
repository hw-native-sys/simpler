// Batched Softmax Preparation Kernel (AIV)
//
// Processes batch_count batches in a single kernel invocation.
// For each batch b at block_idx bn:
//   valid_len = min(N, context_lens[b] - bn * N)
//   sij_masked = pad(sij[b], valid_len, -inf)
//   sij_scale  = sij_masked * scale
//   mij[b]     = row_max(sij_scale)
//   pij[b]     = exp(sij_scale - mij[b])  (truncated to fp16 then back)
//   lij[b]     = row_sum(pij[b])

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
    uint64_t block_idx) {

    __gm__ float* sij_base = reinterpret_cast<__gm__ float*>(sij_batch->buffer.addr);
    __gm__ half* pij_base = reinterpret_cast<__gm__ half*>(pij_batch->buffer.addr);
    __gm__ float* mij_base = reinterpret_cast<__gm__ float*>(mij_batch->buffer.addr);
    __gm__ float* lij_base = reinterpret_cast<__gm__ float*>(lij_batch->buffer.addr);
    __gm__ int32_t* ctx_lens = reinterpret_cast<__gm__ int32_t*>(context_lens_ptr);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_f16 = GlobalTensor<half, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    using TileSijDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, -1>;
    using TileSijPad = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>;

    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_f16 = Tile<TileType::Vec, half, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    TileVecMxN sijTile;
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_f16 pijF16Tile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, M * N * sizeof(float));
    TASSIGN(tmpTile, 2 * M * N * sizeof(float));
    TASSIGN(maxTile, 3 * M * N * sizeof(float));
    TASSIGN(sumTile, 3 * M * N * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijF16Tile, 3 * M * N * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    for (uint64_t b = 0; b < batch_count; b++) {
        int32_t cur_seq = ctx_lens[b];
        uint64_t start = block_idx * N;
        uint64_t valid_len = 0;
        if (start < (uint64_t)cur_seq) {
            uint64_t remaining = (uint64_t)cur_seq - start;
            valid_len = (remaining < N) ? remaining : N;
        }

        __gm__ float* sij_addr = sij_base + b * M * N;
        __gm__ half* pij_addr = pij_base + b * M * N;
        __gm__ float* mij_addr = mij_base + b * M;
        __gm__ float* lij_addr = lij_base + b * M;

        GlobalDataMxN sijGlobal(sij_addr);
        GlobalDataMxN_f16 pijGlobal(pij_addr);
        GlobalScalarDN mijGlobal(mij_addr);
        GlobalScalarDN lijGlobal(lij_addr);

        TLOAD(sijTile, sijGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
        TASSIGN(sijDynTile, 0x0);
        // TFILLPAD_INPLACE alone is insufficient at small N (block_size<=32);
        // manually fill invalid columns with -inf as a workaround.
        TFILLPAD_INPLACE(sijPadTile, sijDynTile);
        if (valid_len < static_cast<uint64_t>(N)) {
            constexpr float NEG_INF = -__builtin_huge_valf();
            for (int r = 0; r < M; r++) {
                for (uint64_t c = valid_len; c < N; c++) {
                    sijTile.SetValue(static_cast<uint32_t>(r * N + c), NEG_INF);
                }
            }
        }

        TMULS(sijTile, sijTile, scale_value);
        pipe_barrier(PIPE_V);
        TROWMAX(maxTile, sijTile, tmpTile);
        pipe_barrier(PIPE_V);
        TROWEXPANDSUB(pijTile, sijTile, maxTile);
        pipe_barrier(PIPE_V);
        TEXP(pijTile, pijTile);
        TCVT(pijF16Tile, pijTile, RoundMode::CAST_ROUND);
        TCVT(pijTile, pijF16Tile, RoundMode::CAST_ROUND);
        TROWSUM(sumTile, pijTile, tmpTile);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(mijGlobal, maxTile);
        TSTORE(lijGlobal, sumTile);
        TSTORE(pijGlobal, pijF16Tile);

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

    softmax_prepare_batch_impl<16, 16>(
        sij_batch, pij_batch, mij_batch, lij_batch,
        scale_value, context_lens_ptr, batch_count, block_idx);
}
