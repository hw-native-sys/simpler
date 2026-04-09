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
// SPMD block_idx encodes (batch_idx, q_tile_idx).
// The two AIV lanes in a cluster split the Q_TILE=16 rows 8/8 via
// get_sub_block_id(): AIV0 updates rows [0, 8), AIV1 updates rows [8, 16).
// The online softmax update is row-independent, so the two lanes never touch
// the same row of mi/li/oi accumulators or the output buffer.
//
// Scalar layout strategy (same as MPMD version):
//   M scalar floats stored contiguously in GM can be loaded as either:
//   - ND (kScalarRows, kScalarCols) RowMajor for element-wise ops
//   - DN (kAlignedRows, 1) ColMajor for row-broadcast ops (TROWEXPANDMUL/DIV)
//   Conversion between layouts uses GM round-trip: ND TSTORE -> DN TLOAD.
//
// Args:
//   args[0] = mij      Tensor* (spmd_blocks*Q_TILE,) float32
//   args[1] = lij      Tensor* (spmd_blocks*Q_TILE,) float32
//   args[2] = oi_new   Tensor* (spmd_blocks*Q_TILE, head_dim) float32
//   args[3] = mi_acc   Tensor* (spmd_blocks*Q_TILE,) float32 [inout]
//   args[4] = li_acc   Tensor* (spmd_blocks*Q_TILE,) float32 [inout]
//   args[5] = oi_acc   Tensor* (spmd_blocks*Q_TILE, head_dim) float32 [inout]
//   args[6] = out      Tensor* (batch*num_heads, head_dim) float32 [inout]
//   args[7] = is_first scalar
//   args[8] = is_last  scalar
//   args[9] = num_heads scalar
//   args[10] = head_dim scalar
//   args[11] = q_loop  scalar

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

static constexpr int QT = 16;     // Full Q tile rows (shared between both AIVs)
static constexpr int SUB_QT = 8;  // Rows per AIV lane (QT / 2)
static constexpr int HD = 16;     // Head dimension

template <int TM, int TN>
static __aicore__ void online_update_spmd(
    __gm__ float *mij_ptr, __gm__ float *lij_ptr, __gm__ float *oi_new_ptr, __gm__ float *mi_ptr,
    __gm__ float *li_ptr, __gm__ float *oi_ptr, __gm__ float *dst_ptr, uint64_t is_first, uint64_t is_last
) {
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = TM / kScalarCols;
    constexpr int kAlignedRows = ((TM * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, TM, TN>, Stride<1, 1, 1, TN, 1>>;
    using GlobalScalarND =
        GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kScalarCols>, Stride<1, 1, 1, kScalarCols, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataMxN oiNewGlobal(oi_new_ptr);
    GlobalDataMxN oiGlobal(oi_ptr);
    GlobalDataMxN dstGlobal(dst_ptr);

    GlobalScalarND mijGlobalND(mij_ptr);
    GlobalScalarND lijGlobalND(lij_ptr);
    GlobalScalarND miGlobalND(mi_ptr);
    GlobalScalarND liGlobalND(li_ptr);

    GlobalScalarDN mijGlobalDN(mij_ptr);
    GlobalScalarDN lijGlobalDN(lij_ptr);
    GlobalScalarDN liGlobalDN(li_ptr);

    using TileDataMxN = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, TM, 1>;

    constexpr int kDataBytes = TM * TN * sizeof(float);
    constexpr int kScalarNDBytes = kScalarRows * kScalarCols * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);

    TileDataMxN oiNewTile;
    TileDataMxN oiTile;
    TileScalarND mijND, lijND, miND, liND;
    TileScalarND miNewND, alphaND, betaND, tmpND;
    TileScalarDN alphaDN, betaDN, liDN;

    TASSIGN(oiNewTile, 0);
    TASSIGN(oiTile, kDataBytes);
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
        TLOAD(oiNewTile, oiNewGlobal);
        TLOAD(mijND, mijGlobalND);
        TLOAD(lijND, lijGlobalND);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(miGlobalND, mijND);
        TSTORE(liGlobalND, lijND);
        TSTORE(oiGlobal, oiNewTile);

        if (is_last) {
            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            TLOAD(liDN, liGlobalDN);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            TROWEXPANDDIV(oiNewTile, oiNewTile, liDN);
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(dstGlobal, oiNewTile);
        }
    } else {
        TLOAD(oiNewTile, oiNewGlobal);
        TLOAD(oiTile, oiGlobal);
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
            TSTORE(dstGlobal, oiTile);
        } else {
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(oiGlobal, oiTile);
        }
    }
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
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

    int32_t block_idx = get_block_idx(args);
    int32_t sub_block_id = get_sub_block_id(args);  // 0 = AIV0 (rows 0..7), 1 = AIV1 (rows 8..15)
    int64_t batch_idx = block_idx / q_loop;
    int64_t q_tile_idx = block_idx % q_loop;

    // Scalar layout: full QT=16 rows pack to kAlignedRowsFull=16 floats per block_idx;
    // each AIV lane owns kAlignedRowsSub=8 contiguous floats inside that slab.
    constexpr int kAlignedRowsFull = ((QT * sizeof(float) + 31) / 32) * (32 / sizeof(float));
    constexpr int kAlignedRowsSub = ((SUB_QT * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    int64_t row_offset = sub_block_id * SUB_QT;

    // Accumulator offsets (each AIV lane owns its own 8-row sub-slice within the block_idx slab)
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

    online_update_spmd<SUB_QT, HD>(mij_ptr, lij_ptr, oi_new_ptr, mi_ptr, li_ptr, oi_ptr, dst_ptr, is_first, is_last);
}
