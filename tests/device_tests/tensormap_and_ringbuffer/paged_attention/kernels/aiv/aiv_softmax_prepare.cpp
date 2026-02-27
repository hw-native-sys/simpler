// Softmax Preparation Kernel (AIV) with partial block masking
//
// Operates on (M, N) tile where M=q_tile_size, N=block_size:
//   Case1: sij is (16, 128)
//   Case2: sij is (64, 64)
//
// For partial blocks (valid_len < N), positions [valid_len, N) in sij are
// filled with -inf via TFILLPAD_INPLACE before softmax, ensuring exp(-inf)=0
// so that invalid key positions contribute zero attention weight.
//
// Computes:
//   sij_masked = TFILLPAD(sij, valid_len, pad=-inf)
//   sij_scale = sij_masked * scale
//   mij = row_max(sij_scale)        -> (M, 1)
//   pij = exp(sij_scale - mij)      -> (M, N)
//   lij = row_sum(pij)              -> (M, 1)

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
static __aicore__ void softmax_prepare_impl(__gm__ TensorData* sij,
    float scale_value,
    __gm__ TensorData* pij,
    __gm__ TensorData* mij,
    __gm__ TensorData* lij) {
    uint64_t valid_len = static_cast<uint64_t>(sij->shapes[1]);
    __gm__ float* sij_addr = reinterpret_cast<__gm__ float*>(sij->buffer.addr);
    __gm__ bfloat16_t* pij_addr = reinterpret_cast<__gm__ bfloat16_t*>(pij->buffer.addr);
    __gm__ float* mij_addr = reinterpret_cast<__gm__ float*>(mij->buffer.addr);
    __gm__ float* lij_addr = reinterpret_cast<__gm__ float*>(lij->buffer.addr);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_bf16 = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataMxN sijGlobal(sij_addr + sij->start_offset);
    GlobalDataMxN_bf16 pijGlobal(pij_addr + pij->start_offset);
    GlobalScalarDN mijGlobal(mij_addr + mij->start_offset);
    GlobalScalarDN lijGlobal(lij_addr + lij->start_offset);

    // Dynamic-cols tile: marks which columns are valid for TFILLPAD boundary
    using TileSijDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, -1>;
    // Padded tile: TFILLPAD_INPLACE fills positions [valid_len, N) with -inf
    using TileSijPad = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>;

    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_bf16 = Tile<TileType::Vec, bfloat16_t, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    TileVecMxN sijTile;
    TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_bf16 pijBf16Tile;

    // All sij tiles share UB address 0x0 (in-place masking)
    TASSIGN(sijTile, 0x0);
    TASSIGN(sijDynTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, M * N * sizeof(float));
    TASSIGN(tmpTile, 2 * M * N * sizeof(float));
    TASSIGN(maxTile, 3 * M * N * sizeof(float));
    TASSIGN(sumTile, 3 * M * N * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijBf16Tile, 3 * M * N * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    // Load full sij (M, N) tile from GM - all N columns including garbage for partial blocks
    // printf("sij addr incore %x\n", sij->buffer.addr);
    TLOAD(sijTile, sijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Mask columns [valid_len, N) with -inf. sijDynTile provides the valid boundary,
    // sijPadTile provides PadValue::Min as the fill value. No-op when valid_len == N.
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
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* sij = reinterpret_cast<__gm__ TensorData*>(args[0]);
    union {
        uint64_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[1]);
    float scale_value = scale_conv.f;
    __gm__ TensorData* pij = reinterpret_cast<__gm__ TensorData*>(args[2]);
    __gm__ TensorData* mij = reinterpret_cast<__gm__ TensorData*>(args[3]);
    __gm__ TensorData* lij = reinterpret_cast<__gm__ TensorData*>(args[4]);
    uint64_t q_tile_size = static_cast<uint64_t>(sij->shapes[0]);

    if (q_tile_size == 16) {
        softmax_prepare_impl<16, 128>(sij, scale_value, pij, mij, lij);
    } else {
        softmax_prepare_impl<64, 64>(sij, scale_value, pij, mij, lij);
    }
}
