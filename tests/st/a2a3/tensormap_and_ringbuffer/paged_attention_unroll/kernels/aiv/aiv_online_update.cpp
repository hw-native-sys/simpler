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
// Online Softmax Update + Normalize Kernel (AIV)
//
// Operates on full tiles where M=q_tile_size, N=head_dim (128):
//   Case1: oi/oi_new are (16, 128), mij/lij/mi/li are 16-element vectors
//   Case2: oi/oi_new are (64, 128), mij/lij/mi/li are 64-element vectors
//
// Scalar layout strategy using TRESHAPE (zero-copy UB reshape):
//   Scalars loaded as DN ColMajor (M, 1) for TROWEXPANDMUL/TROWEXPANDDIV.
//   For element-wise ops (TMAX, TSUB, TEXP, etc.), TRESHAPE to RowMajor (1, M).
//   After arithmetic, TRESHAPE back to ColMajor (M, 1) for row-broadcast ops.
//   This eliminates the GM round-trip (TSTORE ND → TLOAD DN) used in the original.

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <int M, int N>
static __aicore__ void online_update_impl(
    __gm__ Tensor *mij, __gm__ Tensor *lij, __gm__ Tensor *oi_new, __gm__ Tensor *mi, __gm__ Tensor *li,
    __gm__ Tensor *oi, uint64_t is_first, uint64_t is_last, __gm__ Tensor *dst
) {
    __gm__ float *mij_ptr = reinterpret_cast<__gm__ float *>(mij->buffer.addr);
    __gm__ float *lij_ptr = reinterpret_cast<__gm__ float *>(lij->buffer.addr);
    __gm__ float *oi_new_ptr = reinterpret_cast<__gm__ float *>(oi_new->buffer.addr);
    __gm__ float *mi_ptr = reinterpret_cast<__gm__ float *>(mi->buffer.addr);
    __gm__ float *li_ptr = reinterpret_cast<__gm__ float *>(li->buffer.addr);
    __gm__ float *oi_ptr = reinterpret_cast<__gm__ float *>(oi->buffer.addr);
    __gm__ float *dst_ptr = reinterpret_cast<__gm__ float *>(dst->buffer.addr);

    // Aligned rows for ColMajor DN tiles (32-byte alignment)
    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    // --- GlobalTensor types ---

    // Data (M, N) RowMajor
    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;

    // Scalar DN: M contiguous floats as (kAlignedRows, 1) ColMajor for TROWEXPAND ops and loading
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    // Scalar ND: for storing mi_new and li_new back to GM
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = M / kScalarCols;
    using GlobalScalarND =
        GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kScalarCols>, Stride<1, 1, 1, kScalarCols, 1>>;

    // --- GlobalTensor instances ---

    GlobalDataMxN oiNewGlobal(oi_new_ptr + oi_new->start_offset);
    GlobalDataMxN oiGlobal(oi_ptr + oi->start_offset);
    GlobalDataMxN dstGlobal(dst_ptr + dst->start_offset);

    // DN globals for loading scalars as ColMajor
    GlobalScalarDN mijGlobalDN(mij_ptr + mij->start_offset);
    GlobalScalarDN lijGlobalDN(lij_ptr + lij->start_offset);
    GlobalScalarDN miGlobalDN(mi_ptr + mi->start_offset);
    GlobalScalarDN liGlobalDN(li_ptr + li->start_offset);

    // ND globals for storing scalar results
    GlobalScalarND miGlobalND(mi_ptr + mi->start_offset);
    GlobalScalarND liGlobalND(li_ptr + li->start_offset);

    // --- Tile types ---

    using TileDataMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    // RowMajor (1, M) tiles for element-wise arithmetic via TRESHAPE
    using TileScalarRow = Tile<TileType::Vec, float, 1, M, BLayout::RowMajor, 1, M>;

    // ND tile for storing back to GM
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;

    // --- UB memory layout ---

    constexpr int kDataBytes = M * N * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);

    // Data tiles
    TileDataMxN oiNewTile;
    TileDataMxN oiTile;

    // Scalar DN tiles loaded from GM (ColMajor)
    TileScalarDN mijDN, lijDN, miDN, liDN;

    // Temporary DN tiles for results
    TileScalarDN miNewDN, alphaDN, betaDN, liNewDN, tmpDN;

    TASSIGN(oiNewTile, 0);
    TASSIGN(oiTile, kDataBytes);
    TASSIGN(mijDN, 2 * kDataBytes);
    TASSIGN(lijDN, 2 * kDataBytes + kScalarDNBytes);
    TASSIGN(miDN, 2 * kDataBytes + 2 * kScalarDNBytes);
    TASSIGN(liDN, 2 * kDataBytes + 3 * kScalarDNBytes);
    TASSIGN(miNewDN, 2 * kDataBytes + 4 * kScalarDNBytes);
    TASSIGN(alphaDN, 2 * kDataBytes + 5 * kScalarDNBytes);
    TASSIGN(betaDN, 2 * kDataBytes + 6 * kScalarDNBytes);
    TASSIGN(liNewDN, 2 * kDataBytes + 7 * kScalarDNBytes);
    TASSIGN(tmpDN, 2 * kDataBytes + 8 * kScalarDNBytes);

    if (is_first) {
        // --- First block: copy inputs to accumulators ---
        TLOAD(oiNewTile, oiNewGlobal);
        TLOAD(mijDN, mijGlobalDN);
        TLOAD(lijDN, lijGlobalDN);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Store mi = mij, li = lij, oi = oi_new
        // Alias ND tiles to same UB as DN tiles for ND-format store
        TileScalarND mijND, lijND;
        TASSIGN(mijND, 2 * kDataBytes);                   // alias same UB as mijDN
        TASSIGN(lijND, 2 * kDataBytes + kScalarDNBytes);  // alias same UB as lijDN

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(miGlobalND, mijND);    // mi = mij
        TSTORE(liGlobalND, lijND);    // li = lij
        TSTORE(oiGlobal, oiNewTile);  // oi = oi_new

        if (is_last) {
            // Single block: normalize dst = oi_new / lij
            // lijDN already in ColMajor DN format, use directly for TROWEXPANDDIV
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
            TROWEXPANDDIV(oiNewTile, oiNewTile, lijDN);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(dstGlobal, oiNewTile);
        }
    } else {
        // --- Subsequent blocks: accumulate ---

        // Load all inputs as DN (ColMajor)
        TLOAD(oiNewTile, oiNewGlobal);
        TLOAD(oiTile, oiGlobal);
        TLOAD(mijDN, mijGlobalDN);
        TLOAD(lijDN, lijGlobalDN);
        TLOAD(miDN, miGlobalDN);
        TLOAD(liDN, liGlobalDN);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // TRESHAPE: ColMajor(M,1) → RowMajor(1,M) for element-wise arithmetic
        TileScalarRow miRow, mijRow, liRow, lijRow;
        TRESHAPE(miRow, miDN);
        TRESHAPE(mijRow, mijDN);
        TRESHAPE(liRow, liDN);
        TRESHAPE(lijRow, lijDN);

        // Scalar arithmetic in RowMajor (1, M) layout
        TileScalarRow miNewRow, alphaRow, betaRow, liNewRow, tmpRow;
        TASSIGN(miNewRow, 2 * kDataBytes + 4 * kScalarDNBytes);
        TASSIGN(alphaRow, 2 * kDataBytes + 5 * kScalarDNBytes);
        TASSIGN(betaRow, 2 * kDataBytes + 6 * kScalarDNBytes);
        TASSIGN(liNewRow, 2 * kDataBytes + 7 * kScalarDNBytes);
        TASSIGN(tmpRow, 2 * kDataBytes + 8 * kScalarDNBytes);

        TMAX(miNewRow, miRow, mijRow);  // mi_new = max(mi, mij)
        pipe_barrier(PIPE_V);
        // alphaRow and betaRow write to independent UB addresses; both only read miNewRow
        TSUB(alphaRow, miRow, miNewRow);  // alpha_exp = mi - mi_new
        TSUB(betaRow, mijRow, miNewRow);  // beta_exp = mij - mi_new
        pipe_barrier(PIPE_V);
        // TEXP on independent UB addresses
        TEXP(alphaRow, alphaRow);  // alpha = exp(mi - mi_new)
        TEXP(betaRow, betaRow);    // beta = exp(mij - mi_new)
        pipe_barrier(PIPE_V);
        // tmpRow and liNewRow write to independent UB addresses
        TMUL(tmpRow, alphaRow, liRow);    // alpha * li
        TMUL(liNewRow, betaRow, lijRow);  // beta * lij
        pipe_barrier(PIPE_V);
        TADD(liNewRow, tmpRow, liNewRow);  // li_new = alpha*li + beta*lij

        // TRESHAPE back: RowMajor(1,M) → ColMajor(M,1) for TROWEXPANDMUL
        pipe_barrier(PIPE_V);
        TRESHAPE(alphaDN, alphaRow);
        TRESHAPE(betaDN, betaRow);

        // Scale data tiles using row-broadcast multiply
        TROWEXPANDMUL(oiTile, oiTile, alphaDN);       // oi *= alpha
        TROWEXPANDMUL(oiNewTile, oiNewTile, betaDN);  // oi_new *= beta
        pipe_barrier(PIPE_V);
        TADD(oiTile, oiTile, oiNewTile);  // oi = alpha*oi + beta*oi_new

        // Store mi_new and li_new to GM (ND format)
        // Alias ND tiles to the same UB locations as miNewRow and liNewRow
        TileScalarND miNewND, liNewND;
        TASSIGN(miNewND, 2 * kDataBytes + 4 * kScalarDNBytes);
        TASSIGN(liNewND, 2 * kDataBytes + 7 * kScalarDNBytes);

        if (is_last) {
            // Normalize and output: dst = oi / li_new
            TRESHAPE(liNewDN, liNewRow);
            pipe_barrier(PIPE_V);
            TROWEXPANDDIV(oiTile, oiTile, liNewDN);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(miGlobalND, miNewND);  // persist mi_new
            TSTORE(liGlobalND, liNewND);  // persist li_new
            TSTORE(dstGlobal, oiTile);
        } else {
            // Store updated accumulators
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(miGlobalND, miNewND);  // persist mi_new
            TSTORE(liGlobalND, liNewND);  // persist li_new
            TSTORE(oiGlobal, oiTile);
        }
    }
    pipe_sync();
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *mij = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *lij = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *oi_new = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *mi = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *li = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *oi = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ Tensor *dst = reinterpret_cast<__gm__ Tensor *>(args[6]);
    uint64_t is_first = static_cast<uint64_t>(args[7]);
    uint64_t is_last = static_cast<uint64_t>(args[8]);
    uint64_t q_tile_size = static_cast<uint64_t>(mij->shapes[0]);
    // args[10] = head_dim (128)

    if (q_tile_size == 16) {
        online_update_impl<16, 128>(mij, lij, oi_new, mi, li, oi, is_first, is_last, dst);
    } else {
        online_update_impl<64, 128>(mij, lij, oi_new, mi, li, oi, is_first, is_last, dst);
    }
}
