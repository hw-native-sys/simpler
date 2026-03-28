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

// Softmax Preparation Kernel (AIV) with partial block masking
//
// Fixed tile size: sij is (16, 16)
//
// For partial blocks (valid_len < N), positions [valid_len, N) in sij are
// filled with -inf before softmax, ensuring exp(-inf)=0 so that invalid
// key positions contribute zero attention weight.
//
// Computes:
//   sij_masked = pad(sij, valid_len, -inf)
//   sij_scale = sij_masked * scale
//   mij = row_max(sij_scale)        -> (M, 1)
//   pij = exp(sij_scale - mij)      -> (M, N)
//   lij = row_sum(pij)              -> (M, 1)

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"  // NOLINT(build/include_subdir)

// NOLINTNEXTLINE(build/namespaces)
using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]  // NOLINT(whitespace/braces)
#endif

template <int M, int N>
static __aicore__ void softmax_prepare_impl(
    __gm__ Tensor* sij, float scale_value, __gm__ Tensor* pij, __gm__ Tensor* mij, __gm__ Tensor* lij) {
    uint64_t valid_len = static_cast<uint64_t>(sij->shapes[1]);
    __gm__ float* sij_addr = reinterpret_cast<__gm__ float*>(sij->buffer.addr);
    __gm__ half* pij_addr = reinterpret_cast<__gm__ half*>(pij->buffer.addr);
    __gm__ float* mij_addr = reinterpret_cast<__gm__ float*>(mij->buffer.addr);
    __gm__ float* lij_addr = reinterpret_cast<__gm__ float*>(lij->buffer.addr);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_f16 = GlobalTensor<half, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataMxN sijGlobal(sij_addr + sij->start_offset);
    GlobalDataMxN_f16 pijGlobal(pij_addr + pij->start_offset);
    GlobalScalarDN mijGlobal(mij_addr + mij->start_offset);
    GlobalScalarDN lijGlobal(lij_addr + lij->start_offset);

    // Dynamic-cols tile: marks which columns are valid for TFILLPAD boundary
    using TileSijDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, -1>;
    // Padded tile: TFILLPAD_INPLACE fills positions [valid_len, N) with -inf
    using TileSijPad = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>;

    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_f16 = Tile<TileType::Vec, half, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    TileVecMxN sijTile;
    TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_f16 pijF16Tile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijDynTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, M * N * sizeof(float));
    TASSIGN(tmpTile, 2 * M * N * sizeof(float));
    TASSIGN(maxTile, 3 * M * N * sizeof(float));
    TASSIGN(sumTile, 3 * M * N * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijF16Tile, 3 * M * N * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    // Load full sij (M, N) tile from GM - all N columns including garbage for partial blocks
    TLOAD(sijTile, sijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // manually fill invalid columns with -inf as a workaround.
    TFILLPAD_INPLACE(sijPadTile, sijDynTile);
    pipe_barrier(PIPE_V);

    TMULS(sijTile, sijTile, scale_value);
    pipe_barrier(PIPE_V);
    TROWMAX(maxTile, sijTile, tmpTile);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(pijTile, sijTile, maxTile);
    pipe_barrier(PIPE_V);
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

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* sij = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* pij = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* mij = reinterpret_cast<__gm__ Tensor*>(args[2]);
    __gm__ Tensor* lij = reinterpret_cast<__gm__ Tensor*>(args[3]);
    float scale_value = from_u64<float>(static_cast<uint64_t>(args[4]));

    softmax_prepare_impl<16, 16>(sij, scale_value, pij, mij, lij);
}
