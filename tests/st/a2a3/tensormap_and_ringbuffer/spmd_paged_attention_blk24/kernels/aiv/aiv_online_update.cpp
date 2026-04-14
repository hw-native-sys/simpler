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
// SPMD Online Softmax Update + Normalize Kernel (AIV) with dual-vector
// subvector split.
//
// Hardware block_num is fixed at 24. Each hardware block strides over
// total_blocks logical work items:
//   for (idx = hw_block_idx; idx < total_blocks; idx += block_num)
// Each logical block_idx encodes (batch_idx, q_tile_idx).
// The two AIV lanes in a cluster split the q_tile rows via
// get_sub_block_id(): AIV0 updates the first half, AIV1 the second half.
// q_tile is passed as a runtime scalar and dispatched to the matching template.
// The online softmax update is row-independent, so the two lanes never touch
// the same row of mi/li/oi accumulators or the output buffer.
//
// Hardware safety: TROWEXPANDMUL/DIV require a minimum tile height of 16 rows.
// When the sub-tile has fewer rows (e.g., 8), we use 16-row compute tiles with
// pad rows carrying -inf/0 values. Scalar tiles (mi, li, alpha, beta) are also
// padded to 16 elements per lane, stored contiguously in GM so that DN TLOAD
// can read them back with the correct stride.
//
// Scalar layout strategy:
//   M scalar floats stored contiguously in GM can be loaded as either:
//   - ND (kScalarRows, kScalarCols) RowMajor for element-wise ops
//   - DN (kAlignedRows, 1) ColMajor for row-broadcast ops (TROWEXPANDMUL/DIV)
//   Conversion between layouts uses GM round-trip: ND TSTORE -> DN TLOAD.
//
// Args:
//   args[0] = mij      Tensor* (padded scalar buf) float32
//   args[1] = lij      Tensor* (padded scalar buf) float32
//   args[2] = oi_new   Tensor* (total_blocks*q_tile, head_dim) float32
//   args[3] = mi_acc   Tensor* (padded scalar buf) float32 [inout]
//   args[4] = li_acc   Tensor* (padded scalar buf) float32 [inout]
//   args[5] = oi_acc   Tensor* (total_blocks*q_tile, head_dim) float32 [inout]
//   args[6] = out      Tensor* (batch*num_heads, head_dim) float32 [inout]
//   args[7] = is_first scalar
//   args[8] = is_last  scalar
//   args[9] = num_heads scalar
//   args[10] = head_dim scalar
//   args[11] = q_loop  scalar
//   args[12] = q_tile  scalar
//   args[13] = total_blocks scalar

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

#include "intrinsic.h"

static constexpr int HD = 128;     // Head dimension
static constexpr int MIN_TM = 16;  // minimum tile height for TROW* hw safety

// TM = actual number of valid rows this AIV lane owns.
// All TROW* instructions operate on padded PM(=16)-row tiles to avoid the
// hardware dstRptStride issue at TM=8. Data tiles (oi) and scalar tiles (mi, li)
// are padded accordingly; only TM valid rows are loaded from / stored to GM.
template <int TM, int TN>
static __aicore__ void online_update_spmd(
    __gm__ float *mij_ptr, __gm__ float *lij_ptr, __gm__ float *oi_new_ptr, __gm__ float *mi_ptr, __gm__ float *li_ptr,
    __gm__ float *oi_ptr, __gm__ float *dst_ptr, uint64_t is_first, uint64_t is_last
) {
    constexpr int PM = (TM < MIN_TM) ? MIN_TM : TM;  // padded tile height
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = PM / kScalarCols;
    constexpr int kAlignedRows = ((PM * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    // GM accessors for data: load/store only TM valid rows
    using GlobalDataTMxN = GlobalTensor<float, Shape<1, 1, 1, TM, TN>, Stride<1, 1, 1, TN, 1>>;
    // GM accessors for scalars: load/store padded kAlignedRows
    using GlobalScalarND =
        GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kScalarCols>, Stride<1, 1, 1, kScalarCols, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataTMxN oiNewGlobal(oi_new_ptr);
    GlobalDataTMxN oiGlobal(oi_ptr);
    GlobalDataTMxN dstGlobal(dst_ptr);

    GlobalScalarND mijGlobalND(mij_ptr);
    GlobalScalarND lijGlobalND(lij_ptr);
    GlobalScalarND miGlobalND(mi_ptr);
    GlobalScalarND liGlobalND(li_ptr);

    GlobalScalarDN mijGlobalDN(mij_ptr);
    GlobalScalarDN lijGlobalDN(lij_ptr);
    GlobalScalarDN liGlobalDN(li_ptr);

    // Compute tiles use PM(=16) rows
    using TileDataPMxN = Tile<TileType::Vec, float, PM, TN, BLayout::RowMajor, PM, TN>;
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, PM, 1>;

    // Load/store tiles (TM rows) aliased to start of padded tiles
    using TileLoadTMxN = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;

    constexpr int kDataBytes = PM * TN * sizeof(float);
    constexpr int kScalarNDBytes = kScalarRows * kScalarCols * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);

    TileDataPMxN oiNewTile;
    TileDataPMxN oiTile;
    TileLoadTMxN oiNewLoadTile;
    TileLoadTMxN oiLoadTile;
    TileLoadTMxN oiStoreTile;
    TileScalarND mijND, lijND, miND, liND;
    TileScalarND miNewND, alphaND, betaND, tmpND;
    TileScalarDN alphaDN, betaDN, liDN;

    TASSIGN(oiNewTile, 0);
    TASSIGN(oiNewLoadTile, 0);  // alias
    TASSIGN(oiTile, kDataBytes);
    TASSIGN(oiLoadTile, kDataBytes);   // alias
    TASSIGN(oiStoreTile, kDataBytes);  // alias
    TASSIGN(mijND, 2 * kDataBytes);
    TASSIGN(lijND, 2 * kDataBytes + kScalarNDBytes);
    TASSIGN(miND, 2 * kDataBytes + 2 * kScalarNDBytes);
    TASSIGN(liND, 2 * kDataBytes + 3 * kScalarNDBytes);
    TASSIGN(miNewND, 2 * kDataBytes + 4 * kScalarNDBytes);
    TASSIGN(alphaND, 2 * kDataBytes + 5 * kScalarNDBytes);
    TASSIGN(betaND, 2 * kDataBytes + 6 * kScalarNDBytes);
    TASSIGN(tmpND, 2 * kDataBytes + 7 * kScalarNDBytes);
    TASSIGN(alphaDN, 2 * kDataBytes + 8 * kScalarNDBytes);
    TASSIGN(betaDN, 2 * kDataBytes + 8 * kScalarNDBytes + kScalarDNBytes);
    TASSIGN(liDN, 2 * kDataBytes + 8 * kScalarNDBytes + 2 * kScalarDNBytes);

    if (is_first) {
        TLOAD(oiNewLoadTile, oiNewGlobal);
        TLOAD(mijND, mijGlobalND);
        TLOAD(lijND, lijGlobalND);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(miGlobalND, mijND);
        TSTORE(liGlobalND, lijND);
        TSTORE(oiGlobal, oiNewLoadTile);

        if (is_last) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            TLOAD(liDN, liGlobalDN);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TROWEXPANDDIV(oiNewTile, oiNewTile, liDN);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(dstGlobal, oiNewLoadTile);
        }
    } else {
        TLOAD(oiNewLoadTile, oiNewGlobal);
        TLOAD(oiLoadTile, oiGlobal);
        TLOAD(mijND, mijGlobalND);
        TLOAD(lijND, lijGlobalND);
        TLOAD(miND, miGlobalND);
        TLOAD(liND, liGlobalND);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        TMAX(miNewND, miND, mijND);
        pipe_barrier(PIPE_V);
        TSUB(alphaND, miND, miNewND);
        pipe_barrier(PIPE_V);
        TEXP(alphaND, alphaND);
        pipe_barrier(PIPE_V);
        TSUB(betaND, mijND, miNewND);
        pipe_barrier(PIPE_V);
        TEXP(betaND, betaND);
        pipe_barrier(PIPE_V);
        TMUL(liND, alphaND, liND);
        pipe_barrier(PIPE_V);
        TMUL(tmpND, betaND, lijND);
        pipe_barrier(PIPE_V);
        TADD(liND, liND, tmpND);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(miGlobalND, miNewND);
        TSTORE(liGlobalND, liND);
        TSTORE(mijGlobalND, alphaND);
        TSTORE(lijGlobalND, betaND);

        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(alphaDN, mijGlobalDN);
        TLOAD(betaDN, lijGlobalDN);
        if (is_last) {
            TLOAD(liDN, liGlobalDN);
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        TROWEXPANDMUL(oiTile, oiTile, alphaDN);
        TROWEXPANDMUL(oiNewTile, oiNewTile, betaDN);
        pipe_barrier(PIPE_V);
        TADD(oiTile, oiTile, oiNewTile);

        if (is_last) {
            pipe_barrier(PIPE_V);
            TROWEXPANDDIV(oiTile, oiTile, liDN);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(dstGlobal, oiStoreTile);
        } else {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(oiGlobal, oiStoreTile);
        }
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

template <int QT>
static __aicore__ void online_update_entry(
    __gm__ Tensor *mij_t, __gm__ Tensor *lij_t, __gm__ Tensor *oi_new_t, __gm__ Tensor *mi_acc_t,
    __gm__ Tensor *li_acc_t, __gm__ Tensor *oi_acc_t, __gm__ Tensor *out_t, uint64_t is_first, uint64_t is_last,
    int64_t num_heads, int64_t head_dim, int64_t q_loop, int64_t total_blocks, __gm__ int64_t *args
) {
    constexpr int SUB_QT = QT / 2;
    // Padded sub-tile height for hw-safe TROW* ops
    constexpr int PAD_SUB_QT = (SUB_QT < MIN_TM) ? MIN_TM : SUB_QT;

    int32_t hw_block_idx = get_block_idx(args);
    int32_t block_num = get_block_num(args);
    int32_t sub_block_id = get_sub_block_id(args);

    // Scalar layout uses padded kAlignedRows per sub-tile (based on PAD_SUB_QT)
    constexpr int kAlignedRowsFull = 2 * (((PAD_SUB_QT * sizeof(float) + 31) / 32) * (32 / sizeof(float)));
    constexpr int kAlignedRowsSub = ((PAD_SUB_QT * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    for (int32_t block_idx = hw_block_idx; block_idx < total_blocks; block_idx += block_num) {
        int64_t batch_idx = block_idx / q_loop;
        int64_t q_tile_idx = block_idx % q_loop;

        int64_t row_offset = sub_block_id * SUB_QT;

        // Accumulator offsets (each AIV lane owns its own sub-slice within the block_idx slab)
        int64_t scalar_offset = block_idx * kAlignedRowsFull + sub_block_id * kAlignedRowsSub;
        int64_t data_offset = (block_idx * QT + row_offset) * head_dim;

        __gm__ float *mij_ptr =
            reinterpret_cast<__gm__ float *>(mij_t->buffer.addr) + mij_t->start_offset + scalar_offset;
        __gm__ float *lij_ptr =
            reinterpret_cast<__gm__ float *>(lij_t->buffer.addr) + lij_t->start_offset + scalar_offset;
        __gm__ float *oi_new_ptr =
            reinterpret_cast<__gm__ float *>(oi_new_t->buffer.addr) + oi_new_t->start_offset + data_offset;
        __gm__ float *mi_ptr =
            reinterpret_cast<__gm__ float *>(mi_acc_t->buffer.addr) + mi_acc_t->start_offset + scalar_offset;
        __gm__ float *li_ptr =
            reinterpret_cast<__gm__ float *>(li_acc_t->buffer.addr) + li_acc_t->start_offset + scalar_offset;
        __gm__ float *oi_ptr =
            reinterpret_cast<__gm__ float *>(oi_acc_t->buffer.addr) + oi_acc_t->start_offset + data_offset;

        // Output offset: (batch_idx * num_heads + q_tile_idx * QT + row_offset, 0)
        int64_t out_offset = (batch_idx * num_heads + q_tile_idx * QT + row_offset) * head_dim;
        __gm__ float *dst_ptr = reinterpret_cast<__gm__ float *>(out_t->buffer.addr) + out_t->start_offset + out_offset;

        online_update_spmd<SUB_QT, HD>(
            mij_ptr, lij_ptr, oi_new_ptr, mi_ptr, li_ptr, oi_ptr, dst_ptr, is_first, is_last
        );
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    // Safety check: if called with null tensor args (misrouted hub invocation), return.
    if (args[0] == 0 || args[1] == 0 || args[2] == 0) {
        return;
    }

    __gm__ Tensor *mij_t = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *lij_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *oi_new_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *mi_acc_t = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *li_acc_t = reinterpret_cast<__gm__ Tensor *>(args[4]);
    __gm__ Tensor *oi_acc_t = reinterpret_cast<__gm__ Tensor *>(args[5]);
    __gm__ Tensor *out_t = reinterpret_cast<__gm__ Tensor *>(args[6]);
    uint64_t is_first = static_cast<uint64_t>(args[7]);
    uint64_t is_last = static_cast<uint64_t>(args[8]);
    int64_t num_heads = static_cast<int64_t>(args[9]);
    int64_t head_dim = static_cast<int64_t>(args[10]);
    int64_t q_loop = static_cast<int64_t>(args[11]);
    int64_t q_tile = static_cast<int64_t>(args[12]);
    int64_t total_blocks = static_cast<int64_t>(args[13]);

    if (q_tile == 16) {
        online_update_entry<16>(
            mij_t, lij_t, oi_new_t, mi_acc_t, li_acc_t, oi_acc_t, out_t, is_first, is_last, num_heads, head_dim, q_loop,
            total_blocks, args
        );
    } else {
        online_update_entry<64>(
            mij_t, lij_t, oi_new_t, mi_acc_t, li_acc_t, oi_acc_t, out_t, is_first, is_last, num_heads, head_dim, q_loop,
            total_blocks, args
        );
    }
}
