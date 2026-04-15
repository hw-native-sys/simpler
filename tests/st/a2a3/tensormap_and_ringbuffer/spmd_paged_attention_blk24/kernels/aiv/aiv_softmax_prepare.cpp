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
// SPMD Softmax Preparation Kernel (AIV) with partial block masking and
// dual-vector subvector split.
//
// Hardware block_num is fixed at 24. Each hardware block strides over
// total_blocks logical work items:
//   for (idx = hw_block_idx; idx < total_blocks; idx += block_num)
// Each logical block_idx encodes (batch_idx, q_tile_idx).
// The two AIV lanes in a cluster split the q_tile rows via
// get_sub_block_id(): AIV0 handles the first half, AIV1 handles the second.
// q_tile is passed as a runtime scalar and dispatched to the matching template.
//
// Hardware safety: TROW* instructions (TROWMAX, TROWEXPANDSUB, TROWSUM) require
// a minimum tile height of 16 rows. When the sub-tile has fewer rows (e.g., 8),
// we allocate 16-row compute tiles and let the pad rows carry UB garbage.
// Row-independence of TROW* operations ensures valid rows are unaffected.
// Only the valid rows' results are stored to pij; scalars (mij/lij) are stored
// at padded width (16 per lane) so online_update can load them back consistently.
//
// Computes (per sub-slice of sub_m rows):
//   sij_masked = pad(sij, valid_len, -inf)
//   sij_scale = sij_masked * scale
//   mij = row_max(sij_scale)        -> (sub_m, 1)
//   pij = exp(sij_scale - mij)      -> (sub_m, N)
//   lij = row_sum(pij)              -> (sub_m, 1)
//
// Args:
//   args[0] = sij          Tensor* (total_blocks*q_tile, block_size) float32 [input]
//   args[1] = context_lens Tensor* (batch,) int32
//   args[2] = pij          Tensor* (total_blocks*q_tile, block_size) bf16 [output]
//   args[3] = mij          Tensor* (padded scalar buf) float32 [output]
//   args[4] = lij          Tensor* (padded scalar buf) float32 [output]
//   args[5] = scale_value  scalar (as float bits in uint64)
//   args[6] = bn           scalar: current KV block index
//   args[7] = block_size   scalar
//   args[8] = q_loop       scalar
//   args[9] = q_tile       scalar
//   args[10] = total_blocks scalar

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

static constexpr int N_128 = 128;  // block_size (Case1)
static constexpr int N_64 = 64;    // block_size (Case2)
static constexpr int MIN_TM = 16;  // minimum tile height for TROW* hw safety

// TM = actual number of valid rows this AIV lane owns (may be < 16).
// All TROW* instructions operate on padded 16-row tiles to avoid the hardware
// dstRptStride issue at TM=8. We load TM rows from GM into the first TM rows of
// the PM-row UB tile. Pad rows [TM,PM) contain UB garbage, but this is safe:
// all TROW* ops (TROWMAX, TROWEXPANDSUB, TROWSUM) are row-independent, so garbage
// in pad rows cannot affect valid-row results. Scalar outputs (mij/lij) are stored
// at padded width (PM per lane); only valid rows' pij data is stored to GM.
template <int TM, int TN>
static __aicore__ void softmax_prepare_spmd(
    __gm__ float *sij_addr, float scale_value, uint64_t valid_len, __gm__ bfloat16_t *pij_addr, __gm__ float *mij_addr,
    __gm__ float *lij_addr
) {
    constexpr int PM = (TM < MIN_TM) ? MIN_TM : TM;  // padded tile height
    constexpr int kAlignedRows = ((PM * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    // GM accessors: load/store only TM valid rows
    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, TM, TN>, Stride<1, 1, 1, TN, 1>>;
    using GlobalDataMxN_bf16 = GlobalTensor<bfloat16_t, Shape<1, 1, 1, TM, TN>, Stride<1, 1, 1, TN, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataMxN sijGlobal(sij_addr);
    GlobalDataMxN_bf16 pijGlobal(pij_addr);
    GlobalScalarDN mijGlobal(mij_addr);
    GlobalScalarDN lijGlobal(lij_addr);

    // Compute tiles are PM(=16) rows, matching the proven hardware path
    using TileSijDyn = Tile<TileType::Vec, float, PM, TN, BLayout::RowMajor, PM, -1>;
    using TileSijPad =
        Tile<TileType::Vec, float, PM, TN, BLayout::RowMajor, PM, TN, SLayout::NoneBox, 512, PadValue::Min>;

    using TileVecMxN = Tile<TileType::Vec, float, PM, TN, BLayout::RowMajor, PM, TN>;
    using TileVecMxN_bf16 = Tile<TileType::Vec, bfloat16_t, PM, TN, BLayout::RowMajor, PM, TN>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, PM, 1>;

    // GM load tiles (TM rows) aliased to the same UB offset as the padded tiles
    using TileLoadMxN = Tile<TileType::Vec, float, TM, TN, BLayout::RowMajor, TM, TN>;
    using TileStoreBf16 = Tile<TileType::Vec, bfloat16_t, TM, TN, BLayout::RowMajor, TM, TN>;

    TileVecMxN sijTile;
    TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_bf16 pijBf16Tile;

    // Load-size tiles aliased to the start of the padded tiles
    TileLoadMxN sijLoadTile;
    TileStoreBf16 pijStoreTile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijLoadTile, 0x0);  // alias: first TM rows of sijTile
    TASSIGN(sijDynTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, PM * TN * sizeof(float));
    TASSIGN(tmpTile, 2 * PM * TN * sizeof(float));
    TASSIGN(maxTile, 3 * PM * TN * sizeof(float));
    TASSIGN(sumTile, 3 * PM * TN * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijBf16Tile, 3 * PM * TN * sizeof(float) + 2 * kAlignedRows * sizeof(float));
    TASSIGN(pijStoreTile, 3 * PM * TN * sizeof(float) + 2 * kAlignedRows * sizeof(float));  // alias

    // Load only TM valid rows into the first TM rows of the PM-row tile.
    // Pad rows [TM, PM) contain UB garbage — this is safe because all TROW*
    // ops are row-independent: garbage in pad rows cannot affect valid rows.
    TLOAD(sijLoadTile, sijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Pad invalid columns [valid_len, N) with -inf for all PM rows
    TFILLPAD_INPLACE(sijPadTile, sijDynTile);
    pipe_barrier(PIPE_V);

    TMULS(sijTile, sijTile, scale_value);
    pipe_barrier(PIPE_V);
    TROWMAX(maxTile, sijTile, tmpTile);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(pijTile, sijTile, maxTile);
    pipe_barrier(PIPE_V);
    TEXP(pijTile, pijTile);
    TCVT(pijBf16Tile, pijTile, RoundMode::CAST_ROUND);
    TCVT(pijTile, pijBf16Tile, RoundMode::CAST_ROUND);
    TROWSUM(sumTile, pijTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(mijGlobal, maxTile);
    TSTORE(lijGlobal, sumTile);
    TSTORE(pijGlobal, pijStoreTile);  // store only TM valid rows

    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}

template <int Q_TILE, int BLOCK_SIZE>
static __aicore__ void softmax_prepare_entry(
    __gm__ Tensor *sij_t, __gm__ Tensor *context_lens_t, __gm__ Tensor *pij_t, __gm__ Tensor *mij_t,
    __gm__ Tensor *lij_t, float scale_value, int64_t bn, int64_t block_size, int64_t q_loop, int64_t total_blocks,
    __gm__ int64_t *args
) {
    constexpr int SUB_M = Q_TILE / 2;
    // Padded sub-tile height for hw-safe TROW* ops
    constexpr int PAD_SUB_M = (SUB_M < MIN_TM) ? MIN_TM : SUB_M;

    int32_t hw_block_idx = get_block_idx(args);
    int32_t block_num = get_block_num(args);
    int32_t sub_block_id = get_sub_block_id(args);

    for (int32_t block_idx = hw_block_idx; block_idx < total_blocks; block_idx += block_num) {
        int64_t batch_idx = block_idx / q_loop;

        // Compute valid_len for this block: how many columns of sij are valid
        __gm__ int32_t *ctx_ptr =
            reinterpret_cast<__gm__ int32_t *>(context_lens_t->buffer.addr) + context_lens_t->start_offset;
        int64_t cur_seq = static_cast<int64_t>(ctx_ptr[batch_idx]);
        int64_t remaining = cur_seq - bn * block_size;
        uint64_t valid_len;
        if (remaining <= 0) {
            valid_len = 0;
        } else if (remaining >= block_size) {
            valid_len = static_cast<uint64_t>(block_size);
        } else {
            valid_len = static_cast<uint64_t>(remaining);
        }

        // Row offset for this AIV lane within the block_idx's q_tile slice
        int64_t row_offset = sub_block_id * SUB_M;

        // Pointers into this block's SUB_M-row sub-slice of the flat tensors
        int64_t data_row_offset = block_idx * Q_TILE + row_offset;
        __gm__ float *sij_addr =
            reinterpret_cast<__gm__ float *>(sij_t->buffer.addr) + sij_t->start_offset + data_row_offset * block_size;
        __gm__ bfloat16_t *pij_addr = reinterpret_cast<__gm__ bfloat16_t *>(pij_t->buffer.addr) + pij_t->start_offset +
                                      data_row_offset * block_size;

        // Scalar layout uses padded kAlignedRows per sub-tile (based on PAD_SUB_M)
        constexpr int kAlignedRowsFull = 2 * (((PAD_SUB_M * sizeof(float) + 31) / 32) * (32 / sizeof(float)));
        constexpr int kAlignedRowsSub = ((PAD_SUB_M * sizeof(float) + 31) / 32) * (32 / sizeof(float));
        int64_t scalar_offset = block_idx * kAlignedRowsFull + sub_block_id * kAlignedRowsSub;
        __gm__ float *mij_addr =
            reinterpret_cast<__gm__ float *>(mij_t->buffer.addr) + mij_t->start_offset + scalar_offset;
        __gm__ float *lij_addr =
            reinterpret_cast<__gm__ float *>(lij_t->buffer.addr) + lij_t->start_offset + scalar_offset;

        if (valid_len == 0) {
            for (int i = 0; i < kAlignedRowsSub; i++) {
                mij_addr[i] = -1e30f;
                lij_addr[i] = 0.0f;
            }
            for (int i = 0; i < SUB_M * static_cast<int>(block_size); i++) {
                pij_addr[i] = static_cast<bfloat16_t>(0.0f);
            }
            continue;
        }

        softmax_prepare_spmd<SUB_M, BLOCK_SIZE>(sij_addr, scale_value, valid_len, pij_addr, mij_addr, lij_addr);
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *sij_t = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *context_lens_t = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *pij_t = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *mij_t = reinterpret_cast<__gm__ Tensor *>(args[3]);
    __gm__ Tensor *lij_t = reinterpret_cast<__gm__ Tensor *>(args[4]);
    float scale_value = from_u64<float>(static_cast<uint64_t>(args[5]));
    int64_t bn = static_cast<int64_t>(args[6]);
    int64_t block_size = static_cast<int64_t>(args[7]);
    int64_t q_loop = static_cast<int64_t>(args[8]);
    int64_t q_tile = static_cast<int64_t>(args[9]);
    int64_t total_blocks = static_cast<int64_t>(args[10]);

    if (q_tile == 16) {
        softmax_prepare_entry<16, 128>(
            sij_t, context_lens_t, pij_t, mij_t, lij_t, scale_value, bn, block_size, q_loop, total_blocks, args
        );
    } else {
        softmax_prepare_entry<64, 64>(
            sij_t, context_lens_t, pij_t, mij_t, lij_t, scale_value, bn, block_size, q_loop, total_blocks, args
        );
    }
}
