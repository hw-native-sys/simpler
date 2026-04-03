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
/**
 * Tile-based BGEMM Kernel — Combined Cube + Vector (TPUSH/TPOP)
 *
 * Computes one tile iteration: P = A[m,k] @ B[k,n], then C[m,n] += P.
 *
 * Single source compiled twice:
 *   - AIC (cube): __DAV_CUBE__ defined -> TLOAD, TMATMUL, TPUSH
 *   - AIV (vector): __DAV_VEC__ defined -> TPOP, TADD, TSTORE
 *
 * Intermediate result P is transferred via VEC_FIFO, bypassing GM.
 * The accumulator C is still read and written via GM.
 *
 * MixedKernels args:
 *   args[0] = input_a (input)
 *   args[1] = input_b (input)
 *   args[2] = C_tile (inout accumulator)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/fifo.hpp>

#include "tensor.h"

using pto::BLayout;
using pto::Direction;
using pto::GlobalTensor;
using pto::Shape;
using pto::SLayout;
using pto::Tile;
using pto::TileAcc;
using pto::TileLeft;
using pto::TileRight;
using pto::TileSplitAxis;
using pto::TileType;
using pto::TPipe;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__
#endif

#ifdef __DAV_CUBE__
constexpr bool DAV_CUBE = true;
#else
constexpr bool DAV_CUBE = false;
#endif

#ifdef __DAV_VEC__
constexpr bool DAV_VEC = true;
#else
constexpr bool DAV_VEC = false;
#endif

// Tile dimensions (must match golden.py)
constexpr int TILE = 64;
constexpr int M = TILE;
constexpr int K = TILE;
constexpr int N = TILE;

#define VEC_CORES 2
constexpr int VEC_M = M / VEC_CORES;  // each vector sub-core handles half the rows

// TPUSH/TPOP pipe configuration
constexpr uint16_t PP_FLAG_ID = 0;
constexpr uint8_t PP_FIFO_DEPTH = 2;

// Cube accumulator (full M×N tile in L0C)
using AccTileT = TileAcc<float, M, N, M, N>;
// Vector consumer tile (half tile: VEC_M×N in UB, split across 2 vector sub-cores)
using VecFifoTileT = Tile<TileType::Vec, float, VEC_M, N, BLayout::RowMajor, VEC_M, N>;

// Cube→Vector pipe via on-chip VEC_FIFO (bypasses global memory)
using PipeT = TPipe<PP_FLAG_ID, Direction::DIR_C2V, sizeof(float) * VEC_M * N, PP_FIFO_DEPTH>;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *input_a_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *input_b_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *c_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    // Pipe and FIFO tile are declared in common scope (both sides reference the type)
    VecFifoTileT vecFifoTile;
    PipeT mPipe(nullptr, 0U, 0U);

    // =========================================================================
    // Cube side: TLOAD A,B → TMATMUL → TPUSH result to vector via VEC_FIFO
    // =========================================================================
    if constexpr (DAV_CUBE) {
        __gm__ float *input_a =
            reinterpret_cast<__gm__ float *>(input_a_tensor->buffer.addr) + input_a_tensor->start_offset;
        __gm__ float *input_b =
            reinterpret_cast<__gm__ float *>(input_b_tensor->buffer.addr) + input_b_tensor->start_offset;

        using GlobalDataA = GlobalTensor<float, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
        using GlobalDataB = GlobalTensor<float, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;

        GlobalDataA src0Global(input_a);
        GlobalDataB src1Global(input_b);

        using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
        using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;
        using LeftTile = TileLeft<float, M, K, M, K>;
        using RightTile = TileRight<float, K, N, K, N>;

        TileMatA aMatTile;
        TileMatB bMatTile;
        TASSIGN(aMatTile, 0x0);
        TASSIGN(bMatTile, 0x20000);

        LeftTile aTile;
        RightTile bTile;
        AccTileT accTile;
        TASSIGN(aTile, 0x0);
        TASSIGN(bTile, 0x0);
        TASSIGN(accTile, 0x0);

        // Load A and B from GM to L1
        TLOAD(aMatTile, src0Global);
        TLOAD(bMatTile, src1Global);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // Move from L1 to L0A/L0B
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        // Matrix multiply
        TMATMUL(accTile, aTile, bTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        // Push result directly to vector core's UB (replaces TSTORE to GM)
        TPUSH<PipeT, AccTileT, TileSplitAxis::TILE_UP_DOWN>(mPipe, accTile);
    }

    // =========================================================================
    // Vector side: TPOP result from cube → TLOAD C from GM → TADD → TSTORE
    // =========================================================================
    if constexpr (DAV_VEC) {
        uint32_t subBlockIdx = get_subblockid();

        __gm__ float *c_ptr = reinterpret_cast<__gm__ float *>(c_tensor->buffer.addr) + c_tensor->start_offset;
        // Each vector sub-core handles its half: sub-core 0 → rows [0, VEC_M),
        //                                       sub-core 1 → rows [VEC_M, M)
        __gm__ float *c_sub = c_ptr + static_cast<size_t>(subBlockIdx) * VEC_M * N;

        using GlobalC =
            GlobalTensor<float, Shape<1, 1, 1, VEC_M, N>, pto::Stride<VEC_M * N, VEC_M * N, VEC_M * N, N, 1>>;

        GlobalC cGlobal(c_sub);
        GlobalC outGlobal(c_sub);  // write back to same location

        using VecTile = Tile<TileType::Vec, float, VEC_M, N, BLayout::RowMajor, VEC_M, N>;

        VecTile cTile;
        VecTile outTile;
        // Place after FIFO buffer: FIFO uses [0x0, FIFO_DEPTH * VEC_M * N * 4)
        // = [0x0, 2 * 32 * 64 * 4) = [0x0, 0x4000)
        TASSIGN(cTile, 0x4000);
        TASSIGN(outTile, 0x6000);

        // Pop matmul result from cube via VEC_FIFO (replaces TLOAD from GM)
        TPOP<PipeT, VecFifoTileT, TileSplitAxis::TILE_UP_DOWN>(mPipe, vecFifoTile);

        // Load current C tile from GM
        TLOAD(cTile, cGlobal);

        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Accumulate: C += P
        TADD(outTile, cTile, vecFifoTile);
        TFREE<PipeT, TileSplitAxis::TILE_UP_DOWN>(mPipe);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

        // Store result back to GM
        TSTORE(outGlobal, outTile);
    }
}
