// Softmax Preparation Kernel (AIV)
//
// Fixed tile size: sij is (16, 16)
//
// Computes:
//   sij_scale = sij * scale
//   mij = row_max(sij_scale)        -> (M, 1)
//   pij = exp(sij_scale - mij)      -> (M, N)
//   lij = row_sum(pij)              -> (M, 1)

#include <cstdint>
#include <pto/pto-inst.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static __aicore__ void softmax_prepare_impl(__gm__ uint8_t* sij_raw, float scale_value,
                                 __gm__ uint8_t* pij_raw, __gm__ uint8_t* mij_raw,
                                 __gm__ uint8_t* lij_raw)
{
    constexpr int M = 16, N = 16;

    __gm__ float* sij = reinterpret_cast<__gm__ float*>(sij_raw);
    __gm__ half*  pij = reinterpret_cast<__gm__ half*>(pij_raw);
    __gm__ float* mij = reinterpret_cast<__gm__ float*>(mij_raw);
    __gm__ float* lij = reinterpret_cast<__gm__ float*>(lij_raw);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_f16 = GlobalTensor<half, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataMxN sijGlobal(sij);
    GlobalDataMxN_f16 pijGlobal(pij);
    GlobalScalarDN mijGlobal(mij);
    GlobalScalarDN lijGlobal(lij);

    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_f16 = Tile<TileType::Vec, half, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    TileVecMxN sijTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_f16 pijF16Tile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(pijTile, M * N * sizeof(float));
    TASSIGN(tmpTile, 2 * M * N * sizeof(float));
    TASSIGN(maxTile, 3 * M * N * sizeof(float));
    TASSIGN(sumTile, 3 * M * N * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijF16Tile, 3 * M * N * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    TLOAD(sijTile, sijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMULS(sijTile, sijTile, scale_value);
    TROWMAX(maxTile, sijTile, tmpTile);
    TROWEXPANDSUB(pijTile, sijTile, maxTile);
    TEXP(pijTile, pijTile);
    // Truncate pij to fp16 first, then compute lij from truncated values (matches golden)
    TCVT(pijF16Tile, pijTile, RoundMode::CAST_ROUND);
    TCVT(pijTile, pijF16Tile, RoundMode::CAST_ROUND);
    TROWSUM(sumTile, pijTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(mijGlobal, maxTile);
    TSTORE(lijGlobal, sumTile);
    TSTORE(pijGlobal, pijF16Tile);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ uint8_t* sij = reinterpret_cast<__gm__ uint8_t*>(args[0]);
    union { uint64_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[1]);
    float scale_value = scale_conv.f;
    __gm__ uint8_t* pij = reinterpret_cast<__gm__ uint8_t*>(args[2]);
    __gm__ uint8_t* mij = reinterpret_cast<__gm__ uint8_t*>(args[3]);
    __gm__ uint8_t* lij = reinterpret_cast<__gm__ uint8_t*>(args[4]);

    softmax_prepare_impl(sij, scale_value, pij, mij, lij);
}
