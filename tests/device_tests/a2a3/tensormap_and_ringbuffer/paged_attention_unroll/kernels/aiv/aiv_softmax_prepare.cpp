// Two-Pass Softmax Kernel (AIV) for n_blocks tiles
//
// Input:  sij_buf (n_blocks * M, N) fp32 — QK results stacked vertically
// Output: pij_buf (n_blocks * M, N) bf16 — attention weights per block
//         mij (M,) fp32 — global row max across all blocks
//         lij (M,) fp32 — total row sum across all blocks
//
// Pass 1: Iterate over n_blocks tiles, apply scale, mask last block,
//         find global m = max over all blocks of rowmax(S_i * scale)
// Pass 2: Iterate again, compute P_i = exp(S_i * scale - m) -> bf16,
//         accumulate l = sum over all blocks of rowsum(P_i)
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

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <int M, int N>
static __aicore__ void softmax_prepare_n_impl(
    __gm__ float* sij_base,
    float scale_value,
    __gm__ bfloat16_t* pij_base,
    __gm__ float* mij_addr,
    __gm__ float* lij_addr,
    uint64_t n_blocks,
    uint64_t valid_len_last) {

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

    // --- UB memory layout ---
    constexpr int kDataBytes = M * N * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);
    constexpr int kScalarNDBytes = kScalarRows * kScalarCols * sizeof(float);

    TileVecMxN sijTile;
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileVecMxN sumAccTile;
    TileScalarDN localMaxDN;
    TileScalarDN globalMaxDN;
    TileScalarND maxND_a;
    TileScalarND maxND_b;
    TileScalarDN sumDN;
    TileVecMxN_bf16 pijBf16Tile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, kDataBytes);
    TASSIGN(tmpTile, 2 * kDataBytes);
    TASSIGN(sumAccTile, 3 * kDataBytes);
    int scalarBase = 4 * kDataBytes;
    TASSIGN(localMaxDN, scalarBase);
    TASSIGN(globalMaxDN, scalarBase + kScalarDNBytes);
    TASSIGN(maxND_a, scalarBase + 2 * kScalarDNBytes);
    TASSIGN(maxND_b, scalarBase + 2 * kScalarDNBytes + kScalarNDBytes);
    TASSIGN(sumDN, scalarBase + 2 * kScalarDNBytes + 2 * kScalarNDBytes);
    TASSIGN(pijBf16Tile, scalarBase + 2 * kScalarDNBytes + 2 * kScalarNDBytes + kScalarDNBytes);

    // GM scratch aliases (mij/lij output buffers double as scratch)
    GlobalScalarND mijGlobalND(mij_addr);
    GlobalScalarDN mijGlobalDN(mij_addr);
    GlobalScalarND lijGlobalND(lij_addr);
    GlobalScalarDN lijGlobalDN(lij_addr);

    // ======== Pass 1: Find global row max ========
    for (uint64_t i = 0; i < n_blocks; i++) {
        GlobalDataMxN sijGlobal(sij_base + i * M * N);
        TLOAD(sijTile, sijGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if (i == n_blocks - 1 && valid_len_last < static_cast<uint64_t>(N)) {
            TileSijDyn sijDynTile(static_cast<size_t>(valid_len_last));
            TASSIGN(sijDynTile, 0x0);
            TFILLPAD_INPLACE(sijPadTile, sijDynTile);
        }

        TMULS(sijTile, sijTile, scale_value);
        pipe_barrier(PIPE_V);
        TROWMAX(localMaxDN, sijTile, tmpTile);

        if (i == 0) {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(mijGlobalDN, localMaxDN);
        } else {
            // Store local max to lij buffer (scratch) as DN, reload as ND for TMAX
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(lijGlobalDN, localMaxDN);

            // Reload both as ND for TMAX
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            TLOAD(maxND_a, mijGlobalND);
            TLOAD(maxND_b, lijGlobalND);

            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            TMAX(maxND_a, maxND_a, maxND_b);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(mijGlobalND, maxND_a);
        }
    }

    // ======== Pass 2: Compute softmax with global max ========
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    TLOAD(globalMaxDN, mijGlobalDN);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    for (uint64_t i = 0; i < n_blocks; i++) {
        GlobalDataMxN sijGlobal(sij_base + i * M * N);
        GlobalDataMxN_bf16 pijGlobal(pij_base + i * M * N);

        TLOAD(sijTile, sijGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        if (i == n_blocks - 1 && valid_len_last < static_cast<uint64_t>(N)) {
            TileSijDyn sijDynTile(static_cast<size_t>(valid_len_last));
            TASSIGN(sijDynTile, 0x0);
            TFILLPAD_INPLACE(sijPadTile, sijDynTile);
        }

        TMULS(sijTile, sijTile, scale_value);
        pipe_barrier(PIPE_V);
        TROWEXPANDSUB(pijTile, sijTile, globalMaxDN);
        pipe_barrier(PIPE_V);
        TEXP(pijTile, pijTile);
        TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
        TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);

        if (i == 0) {
            TMULS(sumAccTile, pijTile, 1.0f);
        } else {
            TADD(sumAccTile, sumAccTile, pijTile);
        }

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(pijGlobal, pijBf16Tile);

        if (i + 1 < n_blocks) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        }
    }

    // Compute final row sum from accumulated pij values
    pipe_barrier(PIPE_V);
    TROWSUM(sumDN, sumAccTile, tmpTile);

    // Store mij (global max) and lij (total sum)
    // mij already contains the correct global max from Pass 1.
    // Reload and store as DN to ensure consistent format for online_update.
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(lijGlobalDN, sumDN);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* sij_buf = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* pij_buf = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* mij = reinterpret_cast<__gm__ TensorData*>(args[2]);
    __gm__ TensorData* lij = reinterpret_cast<__gm__ TensorData*>(args[3]);
    union {
        uint64_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[4]);
    float scale_value = scale_conv.f;
    uint64_t n_blocks = static_cast<uint64_t>(args[5]);
    uint64_t valid_len_last = static_cast<uint64_t>(args[6]);

    __gm__ float* sij_base = reinterpret_cast<__gm__ float*>(sij_buf->buffer.addr) + sij_buf->start_offset;
    __gm__ bfloat16_t* pij_base = reinterpret_cast<__gm__ bfloat16_t*>(pij_buf->buffer.addr) + pij_buf->start_offset;
    __gm__ float* mij_addr = reinterpret_cast<__gm__ float*>(mij->buffer.addr) + mij->start_offset;
    __gm__ float* lij_addr = reinterpret_cast<__gm__ float*>(lij->buffer.addr) + lij->start_offset;

    uint64_t q_tile_size = static_cast<uint64_t>(sij_buf->shapes[0]);

    if (q_tile_size == 16) {
        softmax_prepare_n_impl<16, 128>(sij_base, scale_value, pij_base, mij_addr, lij_addr, n_blocks, valid_len_last);
    } else {
        softmax_prepare_n_impl<64, 64>(sij_base, scale_value, pij_base, mij_addr, lij_addr, n_blocks, valid_len_last);
    }
}
