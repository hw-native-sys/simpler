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
/*
 * Triangular matrix inverse kernel (recursive unrolled algorithm).
 *
 * Adapted from pto-kernels for the simpler framework:
 *   - tri_inv_utils.h content is inlined below
 *   - constants.h (static -I matrices) is replaced by a tensor arg passed
 *     from the Python test via the orchestration
 *   - The pto-kernels __global__ entry points are replaced by a single
 *     kernel_entry(__gm__ int64_t *args) as required by simpler
 *
 * Args layout (positional, via Tensor* packed in int64_t array):
 *   args[0] = M      (INPUT)  fp16 triangular matrices [num_matrices, N, N]
 *   args[1] = I_neg  (INPUT)  fp16 negative identity   [N, N]
 *   args[2] = M_inv  (OUTPUT) fp16 result              [num_matrices, N, N]
 *   args[3] = config (INPUT)  int64[4]: [matrix_size, num_matrices, is_lower, block_dim]
 */

#include <type_traits>
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

namespace tri_inv_utils {

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetWaitFlag(uint32_t id) {
    set_flag(SrcPipe, DstPipe, static_cast<::event_t>(id));
    wait_flag(SrcPipe, DstPipe, static_cast<::event_t>(id));
}

template <
    typename T1, typename T2,
    typename std::enable_if<std::is_integral<T1>::value && std::is_integral<T2>::value, int>::type = 0>
AICORE inline T1 CeilDiv(T1 value, T2 divisor) {
    return (value + divisor - 1) / divisor;
}

#define BSND_OFFSET(tile_id, N, S, D) (((tile_id) / (N)) * (S) * (N) * (D) + ((tile_id) % (N)) * (D))

AICORE inline uint32_t GetBSNDFixedTileOffset(uint32_t tile_id, uint32_t num_bsnd_heads, uint32_t matrix_size) {
    return BSND_OFFSET(tile_id, num_bsnd_heads, matrix_size, matrix_size);
}

struct BSNDVarlenTileInfo {
    uint32_t bsnd_offset;
    uint32_t valid_size;
};

AICORE inline BSNDVarlenTileInfo GetBSNDVarlenTileInfoFromCuSeqlens(
    uint32_t tile_id, uint32_t num_bsnd_heads, uint32_t matrix_size, __gm__ int32_t *cu_seqlens
) {
    const uint32_t head_idx = tile_id % num_bsnd_heads;
    const uint32_t chunk_idx = tile_id / num_bsnd_heads;

    uint32_t seq_start = static_cast<uint32_t>(cu_seqlens[0]);
    uint32_t accumulated_chunks = 0;
    for (uint32_t seq_idx = 0;; ++seq_idx) {
        const uint32_t seq_end = static_cast<uint32_t>(cu_seqlens[seq_idx + 1]);
        const uint32_t seq_len = seq_end - seq_start;
        const uint32_t seq_num_chunks = CeilDiv(seq_len, matrix_size);
        if (chunk_idx < accumulated_chunks + seq_num_chunks) {
            const uint32_t local_chunk_idx = chunk_idx - accumulated_chunks;
            const uint32_t row_start = seq_start + local_chunk_idx * matrix_size;
            const uint32_t valid_size = min(static_cast<uint32_t>(seq_end - row_start), matrix_size);
            return {row_start * num_bsnd_heads * matrix_size + head_idx * matrix_size, valid_size};
        }
        accumulated_chunks += seq_num_chunks;
        seq_start = seq_end;
    }
}
}  // namespace tri_inv_utils

using namespace tri_inv_utils;

// ---------------------------------------------------------------------------
// Kernel template code (verbatim from pto-kernels kernel_tri_inv_rec_unroll.cpp)
// ---------------------------------------------------------------------------

/**
 * @brief: Takes as input two matrices of size MatrixSize * MatrixSize each.
 * The src matrix lies in L1, while the dst matrix lies either in L0A or L0B.
 * This kernel copies only the diagonal blocks (fractals) of size FractalSize *
 * FractalSize from the src matrix to the dst matrix.
 *
 * @tparam InputT Input data type (fp16).
 * @tparam FractalSize Size of each fractal matrix (diagonal block).
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam SrcL1TileT The actual tile type of the src matrix.
 * @tparam DstL0TileT The actual tile type of the dst matrix.
 *
 * @param src Tile in L1 memory.
 * @param dst Tile in L0A or L0B memory.
 */
template <typename InputT, uint32_t FractalSize, uint32_t MatrixSize, typename SrcL1TileT, typename DstL0TileT>
AICORE inline void CopyDiagonalFractalsL1ToL0(SrcL1TileT src, DstL0TileT dst) {
    constexpr uint32_t NumFractals = MatrixSize / FractalSize;
    constexpr bool is_left = std::is_same_v<DstL0TileT, TileLeft<InputT, MatrixSize, MatrixSize>>;
    constexpr TileType LeftOrRight = is_left ? TileType::Left : TileType::Right;
    constexpr SLayout InnerLayout = is_left ? SLayout::RowMajor : SLayout::ColMajor;

    Tile<
        LeftOrRight, InputT, FractalSize, FractalSize, BLayout::RowMajor, FractalSize, FractalSize, InnerLayout,
        TileConfig::fractalABSize>
        fractals[NumFractals];
    const std::uintptr_t starting_address = reinterpret_cast<std::uintptr_t>(dst.data());
    for (uint32_t i = 0; i < NumFractals; ++i) {
        TASSIGN(fractals[i], starting_address + i * FractalSize * (MatrixSize + FractalSize) * sizeof(InputT));
        TEXTRACT(fractals[i], src, i * FractalSize, i * FractalSize);
    }
}

/**
 * @brief: Takes as input two matrices of size MatrixSize * MatrixSize each,
 * and an integer block_size. The src matrix lies in L1, while the dst matrix
 * either in L0A or L0B. This method copies some of the diagonal blocks from the
 * input to the output as follows:
 * - If dst is in L0A (left): copy even diagonal blocks 0, 2, 4, ...
 * - If dst is in L0B (right): copy odd blocks 1, 3, 5, ...
 * Important note: the dst matrix should be initialized to all-zeros before
 * calling this method
 *
 * @tparam InputT Input data type (fp16).
 * @tparam FractalSize Size of each fractal matrix (diagonal block).
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam SrcL1TileT The actual tile type of the src matrix.
 * @tparam DstL0TileT The actual tile type of the dst matrix.
 *
 * @param src Tile in L1 memory.
 * @param dst Tile in L0A or L0B memory.
 * @param block_size Size of diagonal blocks. Needs: block_size >= FractalSize.
 * @param swap_parity If true, then the parity of copied blocks is swapped: left
 * tile gets odd blocks, while right tile gets even blocks. This is used in the
 * unrolled recursion part of the algorithm, where we need to copy alternating
 * blocks of X in each iteration.
 */
template <typename InputT, uint32_t FractalSize, uint32_t MatrixSize, typename SrcL1TileT, typename DstL0TileT>
AICORE inline void
CopyOddOrEvenBlocksL1ToL0(SrcL1TileT src, DstL0TileT dst, uint32_t block_size, bool swap_parity = false) {
    constexpr bool is_left = std::is_same_v<DstL0TileT, TileLeft<InputT, MatrixSize, MatrixSize>>;
    constexpr TileType LeftOrRight = is_left ? TileType::Left : TileType::Right;
    constexpr SLayout InnerLayout = is_left ? SLayout::RowMajor : SLayout::ColMajor;

    // For left: copy even blocks 0, 2, 4, ... (starting_block=0)
    // For right: copy odd blocks 1, 3, 5, ... (starting_block=1)
    // Default: left→even(0), right→odd(1). swap_parity flips this.
    const uint32_t starting_block_index = (is_left ? 0u : 1u) ^ (swap_parity ? 1u : 0u);

    const uint32_t num_blocks = MatrixSize / block_size;
    const uint32_t num_fractals_per_block = block_size / FractalSize;

    // might need fewer fractals if block_size < FractalSize
    Tile<
        LeftOrRight, InputT, FractalSize, FractalSize, BLayout::RowMajor, FractalSize, FractalSize, InnerLayout,
        TileConfig::fractalABSize>
        fractals[MatrixSize / FractalSize];

    const std::uintptr_t starting_address = reinterpret_cast<std::uintptr_t>(dst.data());
    for (uint32_t i = 0; i < num_fractals_per_block; ++i) {
        for (uint32_t j = 0; j < num_fractals_per_block; ++j) {
            for (uint32_t b = starting_block_index; b < num_blocks; b += 2) {
                const uint32_t offset = b * (MatrixSize + FractalSize) * block_size /* block_offset */ +
                                        i * MatrixSize * FractalSize /* col_fractal_offset */ +
                                        j * FractalSize * FractalSize /* row_fractal_offset */;
                TASSIGN(fractals[b], starting_address + offset * sizeof(InputT));
                TEXTRACT(fractals[b], src, b * block_size + i * FractalSize, b * block_size + j * FractalSize);
            }
        }
    }
}

/**
 * @brief: Prepares Identity and Zeros matrix.
 *
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 *
 * @param I_neg_l1_tile Tile containing the -I (negative identity) matrix.
 * @param Zero_l1_tile Tile to store the all-zero matrix.
 * @param I_l1_tile Tile to store the identity matrix.
 * @param a_l0_tile Tile in L0A for matmuls.
 * @param b_l0_tile Tile in L0B for matmuls.
 * @param c_l0_tile Tile in L0C for matmuls.
 */
template <typename TileL1AB, typename TileL0A, typename TileL0B, typename TileL0C>
AICORE inline void PrepareAuxiliaryMatrices(
    TileL1AB I_neg_l1_tile, TileL1AB Zero_l1_tile, TileL1AB I_l1_tile, TileL0A a_l0_tile, TileL0B b_l0_tile,
    TileL0C c_l0_tile
) {
    TMOV(a_l0_tile, I_neg_l1_tile);  // a_l0 initialized with I_neg
    TMOV(b_l0_tile, I_neg_l1_tile);  // b_l0 initialized with I_neg
    set_flag(PIPE_MTE1, PIPE_M, static_cast<::event_t>(0));
    wait_flag(PIPE_MTE1, PIPE_M, static_cast<::event_t>(0));

    TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);  // c_l0 contains I
    set_flag(PIPE_M, PIPE_FIX, static_cast<::event_t>(0));
    wait_flag(PIPE_M, PIPE_FIX, static_cast<::event_t>(0));

    TMOV(I_l1_tile, c_l0_tile);  // I_l1 now contains I
    set_flag(PIPE_FIX, PIPE_MTE1, static_cast<::event_t>(0));
    wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<::event_t>(0));

    TMOV(b_l0_tile, I_l1_tile);  // b_l0 contains I
    set_flag(PIPE_MTE1, PIPE_M, static_cast<::event_t>(0));
    wait_flag(PIPE_MTE1, PIPE_M, static_cast<::event_t>(0));

    TMATMUL_ACC(c_l0_tile, c_l0_tile, a_l0_tile,
                b_l0_tile);  // c_l0 contains zeros
    set_flag(PIPE_M, PIPE_FIX, static_cast<::event_t>(0));
    wait_flag(PIPE_M, PIPE_FIX, static_cast<::event_t>(0));

    TMOV(Zero_l1_tile, c_l0_tile);  // Zeros_l1 now contains zeros
    set_flag(PIPE_FIX, PIPE_MTE1, static_cast<::event_t>(0));
    wait_flag(PIPE_FIX, PIPE_MTE1, static_cast<::event_t>(0));
}

/**
 * @brief: Inverts a single matrix / tile of the global tensor.
 * The first part of the algorithm inverts the FractalSize * FractalSize
 * diagonal blocks of the input matrix (inv_trick part). The second phase
 * assembles the partial inverses using the cube unig (recursive part).
 *
 * @tparam InputT The type of the input elements.
 * @tparam TileL1AB The type of the input tiles in L1.
 * @tparam TileL0A The type of the input tiles in L0A.
 * @tparam TileL0B The type of the input tiles in L0B.
 * @tparam TileL0C The type of the input tiles in L0C.
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam FractalSize Size of matrix fractals.
 * @tparam NumTilesPerCubeIter How many matrices to load and invert in a single
 * cube iteration.
 *
 * @param X_l1_tile Tile in L1 used for intermediate computations.
 * @param I_l1_tile Tile containing the identity matrix.
 * @param I_neg_l1_tile Tile containing the negative identity matrix.
 * @param M_neg_l1_tile Tile containing the negative input matrix.
 * @param Zero_l1_tile Tile containing the all-zero matrix.
 * @param Y_l1_tile Tile in L1 used for intermediate computations.
 * @param a_l0_tile* Array of two tiles in L0A (for double-buffering).
 * @param b_l0_tile* Array of two tiles in L0B (for double-buffering).
 * @param c_l0_tile* Tile in L0C for matmuls.
 * @param tile_id Index of the current tile (used for sync).
 * @param swap_parity If true, then the parity of copied blocks is swapped: left
 * tile gets odd blocks, while right tile gets even blocks. This is used in the
 * unrolled recursion part of the algorithm, where we need to copy alternating
 * blocks of X in each iteration.
 */
template <
    typename InputT, typename TileL1AB, typename TileL0A, typename TileL0B, typename TileL0C, uint32_t MatrixSize,
    uint32_t FractalSize, uint32_t NumTilesPerCubeIter>
AICORE inline void InvertSingleTile(
    TileL1AB X_l1_tile, TileL1AB I_l1_tile, TileL1AB I_neg_l1_tile, TileL1AB M_neg_l1_tile, TileL1AB Zero_l1_tile,
    TileL1AB Y_l1_tile, TileL0A *a_l0_tile, TileL0B *b_l0_tile, TileL0C *c_l0_tile, const uint32_t tile_id,
    const bool swap_parity = false
) {
    const ::event_t event_0 = static_cast<::event_t>(tile_id);
    const ::event_t event_1 = static_cast<::event_t>(tile_id + NumTilesPerCubeIter);

    TMOV(b_l0_tile[0], Y_l1_tile);      // b_l0[0] contains M
    TMOV(a_l0_tile[0], I_neg_l1_tile);  // a_l0[0] contains I_neg
    set_flag(PIPE_MTE1, PIPE_M, event_0);
    TMOV(a_l0_tile[1], Zero_l1_tile);
    TMOV(b_l0_tile[1], Zero_l1_tile);
    set_flag(PIPE_MTE1, PIPE_M, event_1);
    wait_flag(PIPE_MTE1, PIPE_M, event_1);
    set_flag(PIPE_M, PIPE_MTE1, event_1);
    wait_flag(PIPE_M, PIPE_MTE1, event_1);
    CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile, a_l0_tile[1]);  // a_l0[1] = diag_fractals(M)
    CopyDiagonalFractalsL1ToL0<InputT, FractalSize, MatrixSize>(Y_l1_tile, b_l0_tile[1]);  // b_l0[1] = diag_fractals(M)
    set_flag(PIPE_MTE1, PIPE_M, event_1);

    /* First Matmul: event_0 */
    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // c_l0[0] contains M_neg
    set_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_M, PIPE_MTE1, event_0);

    wait_flag(PIPE_M, PIPE_FIX, event_0);
    TMOV(M_neg_l1_tile, c_l0_tile[0]);  // M_neg_l1 now contains M_neg
    set_flag(PIPE_FIX, PIPE_M, event_0);

    /* Second Matmul: event_1 */
    wait_flag(PIPE_MTE1, PIPE_M, event_1);
    set_flag(PIPE_MTE1, PIPE_M, event_1);
    TMATMUL(c_l0_tile[1], a_l0_tile[1],
            b_l0_tile[1]);  // c_l0[1] contains diag_fractals(M)^2
    set_flag(PIPE_M, PIPE_FIX, event_1);
    wait_flag(PIPE_M, PIPE_FIX, event_1);
    TMOV(Y_l1_tile,
         c_l0_tile[1]);  // Y_l1 now contains diag_fractals(M)^2
    set_flag(PIPE_FIX, PIPE_M, event_1);
    wait_flag(PIPE_FIX, PIPE_M, event_1);

    /* Third Matmul: event_0*/
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    TMOV(b_l0_tile[0], I_neg_l1_tile);  // b_l0[0] contains I_neg
    TMOV(a_l0_tile[0], I_neg_l1_tile);  // a_l0[0] contains I_neg
    set_flag(PIPE_MTE1, PIPE_M, event_0);

    wait_flag(PIPE_MTE1, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);
    wait_flag(PIPE_MTE1, PIPE_M, event_1);
    TMATMUL(c_l0_tile[0], a_l0_tile[1],
            b_l0_tile[0]);  // c_l0[0] = diag_fractals(M_neg)
    set_flag(PIPE_M, PIPE_FIX, event_0);
    wait_flag(PIPE_M, PIPE_FIX, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);

    TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0],
                b_l0_tile[0]);  // c_l0[0] has I-diag_fractals(M)
    set_flag(PIPE_M, PIPE_FIX, event_1);
    wait_flag(PIPE_M, PIPE_FIX, event_1);
    TMOV(X_l1_tile, c_l0_tile[0]);  // X_l1 now contains I-diag_fractals(M)

    /*
     * Inv Trick part:
     * X = I - M
     * Y = M
     * block_size = 1
     * while block_size < FractalSize / 2:
     *     Y = Y @ Y
     *     X = X + X @ Y
     *     block_size *= 2
     */
    set_flag(PIPE_FIX, PIPE_M, event_0);   // store c
    set_flag(PIPE_M, PIPE_MTE1, event_0);  // load matrices for matmuls
    set_flag(PIPE_FIX, PIPE_MTE1, event_0);
    set_flag(PIPE_FIX, PIPE_M, event_1);     // only for update Y
    set_flag(PIPE_M, PIPE_MTE1, event_1);    // only for update Y
    set_flag(PIPE_FIX, PIPE_MTE1, event_1);  // only for update Y
    for (uint32_t block_size = 1; block_size < FractalSize / 2; block_size *= 2) {
        wait_flag(PIPE_M, PIPE_MTE1, event_0);
        TMOV(b_l0_tile[0], I_l1_tile);
        wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
        TMOV(a_l0_tile[0], X_l1_tile);
        set_flag(PIPE_MTE1, PIPE_M, event_0);

        wait_flag(PIPE_FIX, PIPE_MTE1, event_1);
        TMOV(b_l0_tile[1], Y_l1_tile);
        set_flag(PIPE_MTE1, PIPE_M, event_1);

        wait_flag(PIPE_FIX, PIPE_M, event_0);               // from previous iter
        wait_flag(PIPE_MTE1, PIPE_M, event_0);              // from loading a_l0[0], b_l0[0]
        TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // c_l0[0] contains X
        set_flag(PIPE_M, PIPE_FIX, event_0);
        wait_flag(PIPE_M, PIPE_FIX, event_0);
        set_flag(PIPE_FIX, PIPE_M, event_0);
        wait_flag(PIPE_FIX, PIPE_M, event_0);

        if (block_size < FractalSize / 4) {         // Update Y except in last iteration
            wait_flag(PIPE_M, PIPE_MTE1, event_1);  // from previous iter
            TMOV(a_l0_tile[1], Y_l1_tile);
            wait_flag(PIPE_MTE1, PIPE_M, event_1);
            set_flag(PIPE_MTE1, PIPE_M, event_1);

            wait_flag(PIPE_MTE1, PIPE_M, event_1);
            wait_flag(PIPE_FIX, PIPE_M, event_1);  // from previous iter
            TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[1]);
            set_flag(PIPE_M, PIPE_MTE1, event_1);  // for next iter
            set_flag(PIPE_M, PIPE_FIX, event_1);
            set_flag(PIPE_MTE1, PIPE_M, event_1);

            wait_flag(PIPE_M, PIPE_FIX, event_1);
            TMOV(Y_l1_tile, c_l0_tile[1]);
            set_flag(PIPE_FIX, PIPE_M, event_1);  // for next iter
        }
        set_flag(PIPE_FIX, PIPE_MTE1, event_1);  // for next iter

        wait_flag(PIPE_MTE1, PIPE_M, event_1);
        TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[0],
                    b_l0_tile[1]);  // c_l0[0] has X + X @ Y
        set_flag(PIPE_M, PIPE_MTE1, event_0);
        set_flag(PIPE_M, PIPE_FIX, event_0);

        wait_flag(PIPE_M, PIPE_FIX, event_0);
        TMOV(X_l1_tile, c_l0_tile[0]);
        set_flag(PIPE_FIX, PIPE_M, event_0);     // for next iter
        set_flag(PIPE_FIX, PIPE_MTE1, event_0);  // for next iter
    }
    wait_flag(PIPE_FIX, PIPE_MTE1, event_1);  // only for update Y
    wait_flag(PIPE_M, PIPE_MTE1, event_1);    // only for update Y
    wait_flag(PIPE_FIX, PIPE_M, event_1);     // only for update Y
    wait_flag(PIPE_FIX, PIPE_MTE1, event_0);
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    wait_flag(PIPE_FIX, PIPE_M, event_0);

    /*
     * Unrolled recursion part:
     * block_size = FractalSize
     * while block_size < MatrixSize:
     *     LX = even_blocks(X, block_size)
     *     RX = odd_blocks(X, block_size)
     *     Y = LX @ (-M) + I
     *     X = Y @ RX + LX
     *     block_size *= 2
     *
     * Comments:
     * Upper-tri (swap_parity=false):
     *   LX = even_blocks(X), RX = odd_blocks(X)
     *   Y = LX @ (-M) + I, X = Y @ RX + LX
     * Lower-tri (swap_parity=true):
     *   RX = even→L0A(odd via swap), LX = odd→L0B(even via swap)
     *   Y = RX @ (-M) + I, X = Y @ LX + RX
     */
    TMOV(b_l0_tile[1], M_neg_l1_tile);  // b_l0[1] contains M_neg
    TMOV(a_l0_tile[0], I_l1_tile);      // a_l0[0] contains I

    if constexpr (MatrixSize > FractalSize) {
        set_flag(PIPE_FIX, PIPE_M, event_1);
    }
    set_flag(PIPE_M, PIPE_MTE1, event_1);
    set_flag(PIPE_M, PIPE_MTE1, event_0);
    set_flag(PIPE_FIX, PIPE_MTE1, event_1);
    set_flag(PIPE_FIX, PIPE_M, event_0);
    for (uint32_t block_size = FractalSize; block_size < MatrixSize; block_size *= 2) {
        wait_flag(PIPE_M, PIPE_MTE1, event_0);  // Wait for last iter a_l0[1]
        TMOV(a_l0_tile[1], Zero_l1_tile);

        wait_flag(PIPE_M, PIPE_MTE1, event_1);
        TMOV(b_l0_tile[0], I_l1_tile);
        set_flag(PIPE_MTE1, PIPE_M, event_0);

        wait_flag(PIPE_FIX, PIPE_MTE1, event_1);  // Wait to write last X
        CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
            X_l1_tile, a_l0_tile[1], block_size,
            swap_parity
        );  // a_l0[1]: even(LX) or odd(RX)
        set_flag(PIPE_MTE1, PIPE_M, event_1);

        wait_flag(PIPE_MTE1, PIPE_M, event_0);
        wait_flag(PIPE_FIX, PIPE_M, event_0);               // Wait c_l0[0] from previous iter
        TMATMUL(c_l0_tile[0], a_l0_tile[0], b_l0_tile[0]);  // c_l0[0] has I

        wait_flag(PIPE_MTE1, PIPE_M, event_1);
        wait_flag(PIPE_FIX, PIPE_M, event_1);               // Wait c_l0[1] from previous iter
        TMATMUL(c_l0_tile[1], a_l0_tile[1], b_l0_tile[0]);  // c_l0[1] contains LX
        set_flag(PIPE_M, PIPE_MTE1, event_1);               // allow to load RX on b_l0[0]

        TMATMUL_ACC(c_l0_tile[0], c_l0_tile[0], a_l0_tile[1],
                    b_l0_tile[1]);  // c_l0[0] <- LX * M_neg + I
        set_flag(PIPE_M, PIPE_FIX, event_0);
        set_flag(PIPE_M, PIPE_MTE1, event_0);

        wait_flag(PIPE_M, PIPE_FIX, event_0);
        TMOV(Y_l1_tile, c_l0_tile[0]);  // Y_l1 contains LX * M_neg + I
        set_flag(PIPE_FIX, PIPE_MTE1, event_0);
        set_flag(PIPE_FIX, PIPE_M, event_0);

        /* Load complementary blocks of X in L0B. If swap_parity = fase, "Load Odd
         * Blocks Of X In L0B" */
        wait_flag(PIPE_M, PIPE_MTE1, event_1);
        TMOV(b_l0_tile[0], Zero_l1_tile);
        CopyOddOrEvenBlocksL1ToL0<InputT, FractalSize, MatrixSize>(
            X_l1_tile, b_l0_tile[0], block_size,
            swap_parity
        );  // b_l0[0]: odd(RX) or even(LX)

        wait_flag(PIPE_M, PIPE_MTE1, event_0);    // Wait for previous use of a_l0[1]
        wait_flag(PIPE_FIX, PIPE_MTE1, event_0);  // Wait for Y_l1
        TMOV(a_l0_tile[1], Y_l1_tile);            // a_l0[1] contains LX * M_neg + I
        set_flag(PIPE_MTE1, PIPE_M, event_0);

        wait_flag(PIPE_MTE1, PIPE_M, event_0);
        TMATMUL_ACC(c_l0_tile[1], c_l0_tile[1], a_l0_tile[1], b_l0_tile[0]);
        set_flag(PIPE_M, PIPE_MTE1, event_0);  // next iter can read on a_l0[1]
        set_flag(PIPE_M, PIPE_MTE1, event_1);  // next iter can read on b_l0[0]
        set_flag(PIPE_M, PIPE_FIX, event_0);
        wait_flag(PIPE_M, PIPE_FIX, event_0);

        if (block_size < MatrixSize / 2) {  // Update X_l1 except in last iteration
            TMOV(X_l1_tile, c_l0_tile[1]);
            set_flag(PIPE_FIX, PIPE_M, event_1);  // release c_l0[1] for next iter
        }
        set_flag(PIPE_FIX, PIPE_MTE1, event_1);
    }
    wait_flag(PIPE_M, PIPE_MTE1, event_0);
    wait_flag(PIPE_M, PIPE_MTE1, event_1);
    wait_flag(PIPE_FIX, PIPE_M, event_0);
    wait_flag(PIPE_FIX, PIPE_MTE1, event_1);  // Write c_l0[1] to X_l1
}

/**
 * @brief: Runs the main kernel (inverts all matrices in the tensor)
 *
 * @tparam InputT The type of the input elements. Supports fp16 and bf16.
 * @tparam OutputT The type of the output elements.
 * @tparam MatrixSize Size of the entire input/output matrices.
 * @tparam NumTilesPerCubeIter How many matrices to load and invert in a single
 * cube iteration.
 * @tparam IsBSND If IsBSND is false, then the last two dimensions represent a
 * 2D triangular matrix in row-major format, while the other dimensions are
 * batch dimensions. If IsBSND is true, then the dimensions represent in order:
 * B batch size, S sequence length (which is chunked in tiles of size D), N
 * number of heads (equivalent to a second batch dimension for this kernel), and
 * D chunk size. The inverse is over the dimensions S (chunked) and D, row-major
 * within each tile.
 *
 * @param M_inv pointer to the global memory to store the final inverse.
 * @param M Pointer to the global tensor matrix in global memory.
 * @param I_neg Pointer to global memory that contains the negative identity.
 * @param num_matrices The total number of matrices to invert.
 * @param num_bsnd_heads The number of heads, only for BSND format.
 * @param is_lower If input matrices are lower-triangular (is_lower == 1) or
 * upper-triangular (is_lower == 0). Default is upper triangular.
 * @param num_bsnd_heads The number of heads, only for BSND format.
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize, uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE inline void TriInvRecUnrollKernel(
    __gm__ OutputT *M_inv, __gm__ InputT *M, __gm__ InputT *I_neg, uint32_t block_dim, uint32_t num_matrices,
    uint32_t num_bsnd_heads = 0, uint32_t is_lower = 0, __gm__ int32_t *cu_seqlens = nullptr
) {
    /* Initializations */
    constexpr uint32_t TileLen = MatrixSize * MatrixSize;
    constexpr uint32_t FractalSize = 16;  // fractal size for half /bf16
    constexpr uint32_t NumFractalsRowWise = MatrixSize / FractalSize;
    constexpr uint32_t NumL0Buffers = 2;

    if (get_block_idx() * NumTilesPerCubeIter >= num_matrices) {
        return;
    }

    using GlobalTileShapeIn = TileShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
    using GlobalTileStridesIn = typename std::conditional<
        !IsBSND, BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>, Stride<1, 1, 1, -1, 1>>::type;
    using GlobalTileIn = GlobalTensor<InputT, GlobalTileShapeIn, GlobalTileStridesIn, Layout::ND>;
    using GlobalTileDynamicShape = Shape<1, 1, 1, DYNAMIC, DYNAMIC>;
    using GlobalTileDynamicStride = Stride<1, 1, 1, DYNAMIC, 1>;
    using GlobalTileDynamicIn = GlobalTensor<InputT, GlobalTileDynamicShape, GlobalTileDynamicStride, Layout::ND>;
    using GlobalTileStridesINeg = BaseShape2D<InputT, MatrixSize, MatrixSize, Layout::ND>;
    using GlobalTileINeg = GlobalTensor<InputT, GlobalTileShapeIn, GlobalTileStridesINeg, Layout::ND>;

    using GlobalTileShapeOut = TileShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>;
    using GlobalTileStridesOut = typename std::conditional<
        !IsBSND, BaseShape2D<OutputT, MatrixSize, MatrixSize, Layout::ND>, Stride<1, 1, 1, -1, 1>>::type;
    using GlobalTileOut = GlobalTensor<OutputT, GlobalTileShapeOut, GlobalTileStridesOut, Layout::ND>;
    using GlobalTileDynamicOut = GlobalTensor<OutputT, GlobalTileDynamicShape, GlobalTileDynamicStride, Layout::ND>;
    using TileL1AB = Tile<
        TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor, MatrixSize, MatrixSize, SLayout::RowMajor,
        512>;
    using TileL1ABDynamic = Tile<
        TileType::Mat, InputT, MatrixSize, MatrixSize, BLayout::ColMajor, DYNAMIC, DYNAMIC, SLayout::RowMajor, 512,
        PadValue::Zero>;

    // L0 Memory
    using TileL0A = TileLeft<InputT, MatrixSize, MatrixSize>;
    using TileL0B = TileRight<InputT, MatrixSize, MatrixSize>;
    using TileL0C = TileAcc<float, MatrixSize, MatrixSize>;
    using TileL0CDynamic = TileAcc<float, MatrixSize, MatrixSize, DYNAMIC, DYNAMIC>;

    GlobalTileINeg I_neg_global_in(I_neg);

    TileL1AB X_l1_tile;
    TileL1AB I_l1_tile;
    TileL1AB I_neg_l1_tile;
    TileL1AB M_neg_l1_tile;
    TileL1AB Zero_l1_tile;
    TileL1AB Y_l1_tile[NumTilesPerCubeIter];

    TileL0A a_l0_tile[NumL0Buffers];
    TileL0B b_l0_tile[NumL0Buffers];
    TileL0C c_l0_tile[NumL0Buffers];

    TASSIGN(I_l1_tile, 0x0);
    TASSIGN(I_neg_l1_tile, 0x0 + TileLen * sizeof(InputT));
    TASSIGN(Zero_l1_tile, 0x0 + 2 * TileLen * sizeof(InputT));
    TASSIGN(M_neg_l1_tile, 0x0 + 3 * TileLen * sizeof(InputT));
    TASSIGN(X_l1_tile, 0x0 + 4 * TileLen * sizeof(InputT));
    for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
        TASSIGN(Y_l1_tile[tile_id], 0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
    }

    for (uint32_t buffer_num = 0; buffer_num < NumL0Buffers; ++buffer_num) {
        TASSIGN(a_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
        TASSIGN(b_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(InputT));
        TASSIGN(c_l0_tile[buffer_num], 0x0 + buffer_num * TileLen * sizeof(float));
    }
    TLOAD(I_neg_l1_tile, I_neg_global_in);
    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(0));
    wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(0));

    PrepareAuxiliaryMatrices<TileL1AB, TileL0A, TileL0B, TileL0C>(
        I_neg_l1_tile, Zero_l1_tile, I_l1_tile, a_l0_tile[0], b_l0_tile[0], c_l0_tile[0]
    );

    const uint32_t max_iters_per_aic = CeilDiv(num_matrices, (uint32_t)(NumTilesPerCubeIter * block_dim));

    /* Main iteration - Compute all tiles */
    uint32_t bsnd_tile_offsets[NumTilesPerCubeIter] = {0};
    uint32_t bsnd_tile_valid_sizes[NumTilesPerCubeIter] = {0};
    uint32_t next_tile_id_that_waits_for_pipe_fix_pipe_m = 0;
    set_flag(PIPE_FIX, PIPE_M, static_cast<::event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
    for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
        set_flag(PIPE_M, PIPE_MTE2, static_cast<::event_t>(tile_id));
    }
    for (uint32_t cube_iter = 0; cube_iter < max_iters_per_aic; ++cube_iter) {
        const uint32_t global_index = (cube_iter * block_dim + get_block_idx()) * NumTilesPerCubeIter;
        if (global_index >= num_matrices) {
            break;
        }
        for (uint32_t tile_id = 0; (tile_id < NumTilesPerCubeIter) && (global_index + tile_id < num_matrices);
             ++tile_id) {
            if constexpr (IsBSND) {
                const uint32_t global_tile_id = global_index + tile_id;
                if (cu_seqlens != nullptr) {
                    const BSNDVarlenTileInfo tile_info = tri_inv_utils::GetBSNDVarlenTileInfoFromCuSeqlens(
                        global_tile_id, num_bsnd_heads, MatrixSize, cu_seqlens
                    );
                    bsnd_tile_offsets[tile_id] = tile_info.bsnd_offset;
                    bsnd_tile_valid_sizes[tile_id] = tile_info.valid_size;
                } else {
                    bsnd_tile_offsets[tile_id] =
                        tri_inv_utils::GetBSNDFixedTileOffset(global_tile_id, num_bsnd_heads, MatrixSize);
                    bsnd_tile_valid_sizes[tile_id] = MatrixSize;
                }
                const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
                const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
                const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
                wait_flag(PIPE_M, PIPE_MTE2, static_cast<::event_t>(tile_id));
                if (valid_size < MatrixSize) {
                    TileL1ABDynamic Y_dyn_l1_tile(valid_size, valid_size);
                    TASSIGN(Y_dyn_l1_tile, 0x0 + (5 + tile_id) * TileLen * sizeof(InputT));
                    GlobalTileDynamicIn M_global_in_dyn(
                        M + bsnd_offset, {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(valid_size)},
                        {1, 1, 1, row_stride, 1}
                    );
                    TLOAD(Y_dyn_l1_tile, M_global_in_dyn);
                    set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(tile_id));
                    wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(tile_id));
                    TFILLPAD(Y_dyn_l1_tile, Y_dyn_l1_tile);
                } else {
                    GlobalTileIn M_global_in(M + bsnd_offset, {}, {row_stride});
                    TLOAD(Y_l1_tile[tile_id], M_global_in);
                }
            } else {
                GlobalTileIn M_global_in(M + (global_index + tile_id) * TileLen);
                wait_flag(PIPE_M, PIPE_MTE2, static_cast<::event_t>(tile_id));
                TLOAD(Y_l1_tile[tile_id],
                      M_global_in);  // Copies NumTilesPerCubeIter tiles at once
            }
            set_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(tile_id));
        }

        constexpr uint32_t final_c_buffer_index = MatrixSize > FractalSize ? 1 : 0;
        for (uint32_t tile_id = 0; (tile_id < NumTilesPerCubeIter) && (global_index + tile_id < num_matrices);
             ++tile_id) {
            // Wait for previous cube iter to write result
            wait_flag(PIPE_FIX, PIPE_M, static_cast<::event_t>(tile_id));
            // Wait for loading new matrices from GM
            wait_flag(PIPE_MTE2, PIPE_MTE1, static_cast<::event_t>(tile_id));

            InvertSingleTile<InputT, TileL1AB, TileL0A, TileL0B, TileL0C, MatrixSize, FractalSize, NumTilesPerCubeIter>(
                X_l1_tile, I_l1_tile, I_neg_l1_tile, M_neg_l1_tile, Zero_l1_tile, Y_l1_tile[tile_id], a_l0_tile,
                b_l0_tile, c_l0_tile, tile_id, is_lower != 0
            );

            // Allow next cube_iter to proceed for this tile_id
            set_flag(PIPE_M, PIPE_MTE2, static_cast<::event_t>(tile_id));

            /* Store result */
            if constexpr (IsBSND) {
                const uint32_t bsnd_offset = bsnd_tile_offsets[tile_id];
                const uint32_t valid_size = bsnd_tile_valid_sizes[tile_id];
                const int row_stride = static_cast<int>(MatrixSize * num_bsnd_heads);
                if (valid_size < MatrixSize) {
                    TileL0CDynamic c_l0_tail_tile(valid_size, valid_size);
                    TASSIGN(c_l0_tail_tile, 0x0 + final_c_buffer_index * TileLen * sizeof(float));
                    GlobalTileDynamicOut M_inv_global_out_dyn(
                        M_inv + bsnd_offset, {1, 1, 1, static_cast<int>(valid_size), static_cast<int>(valid_size)},
                        {1, 1, 1, row_stride, 1}
                    );
                    TSTORE(M_inv_global_out_dyn, c_l0_tail_tile);
                } else {
                    GlobalTileOut M_inv_global_out(M_inv + bsnd_offset, {}, {row_stride});
                    TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
                }
            } else {
                GlobalTileOut M_inv_global_out(M_inv + (global_index + tile_id) * TileLen);
                TSTORE(M_inv_global_out, c_l0_tile[final_c_buffer_index]);
            }
            next_tile_id_that_waits_for_pipe_fix_pipe_m = (tile_id + 1) % NumTilesPerCubeIter;
            set_flag(PIPE_FIX, PIPE_M, static_cast<::event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
        }
    }
    for (uint32_t tile_id = 0; tile_id < NumTilesPerCubeIter; ++tile_id) {
        wait_flag(PIPE_M, PIPE_MTE2, static_cast<::event_t>(tile_id));
    }
    wait_flag(PIPE_FIX, PIPE_M, static_cast<::event_t>(next_tile_id_that_waits_for_pipe_fix_pipe_m));
}

/*
 * @brief: Computes the inverses of the blocks of tensor M
 */
template <typename InputT, typename OutputT, uint32_t MatrixSize, uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE __attribute__((always_inline)) void runKernelTriInvRecUnroll(
    __gm__ OutputT *M_inv, __gm__ InputT *M, __gm__ InputT *I_neg, uint32_t block_dim, uint32_t num_matrices,
    uint32_t num_bsnd_heads = 0, uint32_t is_lower = 0, __gm__ int32_t *cu_seqlens = nullptr
) {
#if defined(__DAV_CUBE__)  // Cube compilation

    TriInvRecUnrollKernel<InputT, OutputT, MatrixSize, NumTilesPerCubeIter, IsBSND>(
        M_inv, M, I_neg, block_dim, num_matrices, num_bsnd_heads, is_lower, cu_seqlens
    );
#else
// Nothing to do on AIV
#endif
}

template <typename InputT, typename OutputT, uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE __attribute__((always_inline)) void run_tri_inv_rec_unroll(
    __gm__ OutputT *tensor_out, __gm__ InputT *tensor_in, __gm__ InputT *minus_eye_in, uint32_t block_dim,
    uint32_t matrix_size, uint32_t num_matrices, uint32_t num_bsnd_heads, uint32_t is_lower = 0,
    __gm__ int32_t *cu_seqlens = nullptr
) {
    static_assert(
        std::is_same_v<InputT, half> or std::is_same_v<InputT, bfloat16_t>,
        "tri_inv_rec_unroll supports only fp16 or bf16."
    );

    static_assert(
        std::is_same_v<OutputT, half> or std::is_same_v<OutputT, bfloat16_t>,
        "tri_inv_rec_unroll supports only fp16 or bf16."
    );
    switch (matrix_size) {
    case 16:
        runKernelTriInvRecUnroll<InputT, OutputT, 16, NumTilesPerCubeIter, IsBSND>(
            tensor_out, tensor_in, minus_eye_in, block_dim, num_matrices, num_bsnd_heads, is_lower, cu_seqlens
        );
        break;
    case 32:
        runKernelTriInvRecUnroll<InputT, OutputT, 32, NumTilesPerCubeIter, IsBSND>(
            tensor_out, tensor_in, minus_eye_in, block_dim, num_matrices, num_bsnd_heads, is_lower, cu_seqlens
        );
        break;
    case 64:
        runKernelTriInvRecUnroll<InputT, OutputT, 64, NumTilesPerCubeIter, IsBSND>(
            tensor_out, tensor_in, minus_eye_in, block_dim, num_matrices, num_bsnd_heads, is_lower, cu_seqlens
        );
        break;
    case 128:
        runKernelTriInvRecUnroll<InputT, OutputT, 128, NumTilesPerCubeIter, IsBSND>(
            tensor_out, tensor_in, minus_eye_in, block_dim, num_matrices, num_bsnd_heads, is_lower, cu_seqlens
        );
        break;
    }
}

template <typename InputT, typename OutputT, uint32_t NumTilesPerCubeIter, bool IsBSND>
AICORE __attribute__((always_inline)) void run_tri_inv_rec_unroll_per_num_matrices(
    __gm__ OutputT *tensor_out, __gm__ InputT *tensor_in, __gm__ InputT *minus_eye_in, uint32_t block_dim,
    uint32_t matrix_size, uint32_t num_matrices, uint32_t num_bsnd_heads, uint32_t is_lower = 0,
    __gm__ int32_t *cu_seqlens = nullptr
) {
    if (num_bsnd_heads == 0) {
        if (num_matrices <= block_dim) {
            run_tri_inv_rec_unroll<InputT, OutputT, 1 /* NumTilesPerCubeIter */, false /* IsBSND */>(
                tensor_out, tensor_in, minus_eye_in, block_dim, matrix_size, num_matrices, num_bsnd_heads, is_lower,
                cu_seqlens
            );
        } else if (num_matrices <= 2 * block_dim) {
            run_tri_inv_rec_unroll<InputT, OutputT, 2 /* NumTilesPerCubeIter */, false /* IsBSND */>(
                tensor_out, tensor_in, minus_eye_in, block_dim, matrix_size, num_matrices, num_bsnd_heads, is_lower,
                cu_seqlens
            );
        } else {
            run_tri_inv_rec_unroll<InputT, OutputT, 4 /* NumTilesPerCubeIter */, false /* IsBSND */>(
                tensor_out, tensor_in, minus_eye_in, block_dim, matrix_size, num_matrices, num_bsnd_heads, is_lower,
                cu_seqlens
            );
        }
    } else {
        if (num_matrices <= block_dim) {
            run_tri_inv_rec_unroll<InputT, OutputT, 1 /* NumTilesPerCubeIter */, true /* IsBSND */>(
                tensor_out, tensor_in, minus_eye_in, block_dim, matrix_size, num_matrices, num_bsnd_heads, is_lower,
                cu_seqlens
            );
        } else if (num_matrices <= 2 * block_dim) {
            run_tri_inv_rec_unroll<InputT, OutputT, 2 /* NumTilesPerCubeIter */, true /* IsBSND */>(
                tensor_out, tensor_in, minus_eye_in, block_dim, matrix_size, num_matrices, num_bsnd_heads, is_lower,
                cu_seqlens
            );
        } else {
            run_tri_inv_rec_unroll<InputT, OutputT, 4 /* NumTilesPerCubeIter */, true /* IsBSND */>(
                tensor_out, tensor_in, minus_eye_in, block_dim, matrix_size, num_matrices, num_bsnd_heads, is_lower,
                cu_seqlens
            );
        }
    }
}

// ---------------------------------------------------------------------------
// simpler framework entry point
// ---------------------------------------------------------------------------

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *M_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *I_neg_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *M_inv_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *config_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);

    __gm__ int64_t *cfg = reinterpret_cast<__gm__ int64_t *>(config_tensor->buffer.addr);
    const uint32_t matrix_size = static_cast<uint32_t>(cfg[0]);
    const uint32_t num_matrices = static_cast<uint32_t>(cfg[1]);
    const uint32_t is_lower = static_cast<uint32_t>(cfg[2]);
    const uint32_t block_dim = static_cast<uint32_t>(cfg[3]);

    __gm__ half *M = reinterpret_cast<__gm__ half *>(M_tensor->buffer.addr) + M_tensor->start_offset;
    __gm__ half *I_neg = reinterpret_cast<__gm__ half *>(I_neg_tensor->buffer.addr) + I_neg_tensor->start_offset;
    __gm__ half *M_inv = reinterpret_cast<__gm__ half *>(M_inv_tensor->buffer.addr) + M_inv_tensor->start_offset;

    run_tri_inv_rec_unroll_per_num_matrices<half, half, 1, false>(
        M_inv, M, I_neg, block_dim, matrix_size, num_matrices, 0, is_lower, nullptr
    );
}
