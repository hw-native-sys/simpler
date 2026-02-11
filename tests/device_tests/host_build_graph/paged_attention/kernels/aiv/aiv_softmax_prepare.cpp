// Softmax Preparation Kernel (AIV)
//
// Operates on full (M, N) tile where M=q_tile_size, N=block_size:
//   Case1: sij is (16, 128)
//   Case2: sij is (64, 64)
//
// Computes:
//   sij_scale = sij * scale
//   mij = row_max(sij_scale)        -> (M, 1)
//   pij = exp(sij_scale - mij)      -> (M, N)
//   lij = row_sum(pij)              -> (M, 1)

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <int M, int N>
static __aicore__ void softmax_prepare_impl(__gm__ uint8_t* sij_raw, float scale_value,
                                 __gm__ uint8_t* pij_raw, __gm__ uint8_t* mij_raw,
                                 __gm__ uint8_t* lij_raw)
{
    __gm__ float*      sij = reinterpret_cast<__gm__ float*>(sij_raw);
    __gm__ bfloat16_t* pij = reinterpret_cast<__gm__ bfloat16_t*>(pij_raw);
    __gm__ float*      mij = reinterpret_cast<__gm__ float*>(mij_raw);
    __gm__ float*      lij = reinterpret_cast<__gm__ float*>(lij_raw);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_bf16 = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataMxN sijGlobal(sij);
    GlobalDataMxN_bf16 pijGlobal(pij);
    GlobalScalarDN mijGlobal(mij);
    GlobalScalarDN lijGlobal(lij);

    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_bf16 = Tile<TileType::Vec, bfloat16_t, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    TileVecMxN sijTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_bf16 pijBf16Tile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(pijTile, M * N * sizeof(float));
    TASSIGN(tmpTile, 2 * M * N * sizeof(float));
    TASSIGN(maxTile, 3 * M * N * sizeof(float));
    TASSIGN(sumTile, 3 * M * N * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijBf16Tile, 3 * M * N * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    TLOAD(sijTile, sijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TMULS(sijTile, sijTile, scale_value);
    TROWMAX(maxTile, sijTile, tmpTile);
    TROWEXPANDSUB(pijTile, sijTile, maxTile);
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
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ uint8_t* sij = reinterpret_cast<__gm__ uint8_t*>(args[0]);
    union { uint64_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[1]);
    float scale_value = scale_conv.f;
    __gm__ uint8_t* pij = reinterpret_cast<__gm__ uint8_t*>(args[2]);
    __gm__ uint8_t* mij = reinterpret_cast<__gm__ uint8_t*>(args[3]);
    __gm__ uint8_t* lij = reinterpret_cast<__gm__ uint8_t*>(args[4]);
    int q_tile_size = static_cast<int>(args[5]);
    // args[6] = block_size

    if (q_tile_size == 16) {
        softmax_prepare_impl<16, 128>(sij, scale_value, pij, mij, lij);
    } else {
        softmax_prepare_impl<64, 64>(sij, scale_value, pij, mij, lij);
    }
}
