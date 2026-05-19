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
 * Tile-based Matrix Multiplication Kernel (Cube Core)
 *
 * Computes: output = input_a @ input_b (tile_size x tile_size tile matmul)
 * Uses TMATMUL instruction
 *
 * Tile size is determined by golden.py configuration and passed through
 * tensor shapes from orchestration.
 *
 * Args (Tensor*):
 *   args[0] = input_a (INPUT)
 *   args[1] = input_b (INPUT)
 *   args[2] = output  (OUTPUT)
 *   args[3] = config  (INPUT) - int64_t[4]: [tile_size, grid_k, num_groups, incore_loop]
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>

#include "tensor.h"

using namespace pto;

#include "pipe_sync.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <typename T>
AICORE constexpr inline T CeilAlign(T num_1, T num_2) {
    if (num_2 == 0) {
        return 0;
    }
    return (num_1 + num_2 - 1) / num_2 * num_2;
}

using namespace pto;

template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void SetFlag(uint32_t id) {
    set_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}
template <pipe_t SrcPipe, pipe_t DstPipe>
AICORE inline void WaitFlag(uint32_t id) {
    wait_flag(SrcPipe, DstPipe, static_cast<event_t>(id));
}

template <typename InputT, typename OutputT, uint32_t matrix_size>
AICORE void runKernelSimpleMatMul(__gm__ InputT *a, __gm__ InputT *b, __gm__ OutputT *c) {
    constexpr uint32_t tile_len = matrix_size * matrix_size;

    /* Global Memory / Tensors */
    using TensorShapeIn = TileShape2D<InputT, matrix_size, matrix_size, Layout::ND>;
    using TensorStridesIn = BaseShape2D<InputT, matrix_size, matrix_size, Layout::ND>;
    using GlobalTensorIn = GlobalTensor<InputT, TensorShapeIn, TensorStridesIn, Layout::ND>;

    using TensorShapeOut = TileShape2D<OutputT, matrix_size, matrix_size, Layout::ND>;
    using TensorStridesOut = BaseShape2D<OutputT, matrix_size, matrix_size, Layout::ND>;
    using GlobalTensorOut = GlobalTensor<OutputT, TensorShapeOut, TensorStridesOut, Layout::ND>;

    /* L1 Memory */
    using TileL1AB = Tile<
        TileType::Mat, InputT, matrix_size, matrix_size, BLayout::ColMajor, matrix_size, matrix_size, SLayout::RowMajor,
        512>;

    /* L0 Memory */
    using TileL0A = TileLeft<InputT, matrix_size, matrix_size>;
    using TileL0B = TileRight<InputT, matrix_size, matrix_size>;
    using TileL0C = TileAcc<OutputT, matrix_size, matrix_size>;

    GlobalTensorIn a_global_in(a);
    GlobalTensorIn b_global_in(b);
    GlobalTensorOut c_global_out(c);
    TASSIGN(a_global_in, a);
    TASSIGN(b_global_in, b);
    TASSIGN(c_global_out, c);

    TileL1AB a_l1_tile;
    TileL1AB b_l1_tile;
    TASSIGN(a_l1_tile, 0x0);
    TASSIGN(b_l1_tile, 0x0 + tile_len * sizeof(InputT));

    TileL0A a_l0_tile;
    TileL0B b_l0_tile;
    TileL0C c_l0_tile;
    // L0A/L0B/L0C are distinct scratchpads
    TASSIGN(a_l0_tile, 0x0);
    TASSIGN(b_l0_tile, 0x0);
    TASSIGN(c_l0_tile, 0x0);

    // LOAD matrix A from GM -> L1 (MTE2)
    TLOAD(a_l1_tile, a_global_in);
    TLOAD(b_l1_tile, b_global_in);
    SetFlag<PIPE_MTE2, PIPE_MTE1>(0);
    WaitFlag<PIPE_MTE2, PIPE_MTE1>(0);

    // Copy A from L1 -> L0 (MTE1)
    // MatMul unit waits (using id:0) for MTE1 to load matrices into L0A/B
    TMOV(a_l0_tile, a_l1_tile);
    // Copy B from L1 -> L0B
    // MatMul unit waits (using id:1) for MTE1 to load matrices into L0A/B
    TMOV(b_l0_tile, b_l1_tile);
    SetFlag<PIPE_MTE1, PIPE_M>(0);   // MTE1 pipe sets flag for MM pipe
    WaitFlag<PIPE_MTE1, PIPE_M>(0);  // MM pipe waits for MTE1 pipe to set flag

    // MATMUL (M)
    TMATMUL(c_l0_tile, a_l0_tile, b_l0_tile);
    pipe_barrier(PIPE_ALL);
    SetFlag<PIPE_M, PIPE_FIX>(0);   // M pipe sets flag for FIX pipe
    WaitFlag<PIPE_M, PIPE_FIX>(0);  // FIX pipe waits for M pipe to set flag
    TSTORE(c_global_out, c_l0_tile);
}

template <typename T>
AICORE void run_simple_matmul(__gm__ T *a, __gm__ T *b, __gm__ float *c, uint32_t matrix_size) {
    static_assert(
        std::is_same_v<T, half> or std::is_same_v<T, bfloat16_t> or std::is_same_v<T, float>,
        "simple_matmul supports only fp16/bf16/fp32."
    );

    switch (matrix_size) {
    case 16:
        runKernelSimpleMatMul<T, float, 16>(a, b, c);
        break;
    case 32:
        runKernelSimpleMatMul<T, float, 32>(a, b, c);
        break;

    case 64:
        runKernelSimpleMatMul<T, float, 64>(a, b, c);
        break;

    case 96:
        runKernelSimpleMatMul<T, float, 96>(a, b, c);
        break;

    case 128:
        runKernelSimpleMatMul<T, float, 128>(a, b, c);
        break;
    default:
        // Unsupported tile size - can add more cases as needed
        break;
    }
}

/**
 * Element-wise multiplication kernel implementation
 *
 * Unified signature: all arguments passed via int64_t array
 * @param args  Argument array:
 *              args[0] = src0 pointer (first input tensor)
 *              args[1] = src1 pointer (second input tensor)
 *              args[2] = out pointer (output tensor)
 *              args[3] = size (number of elements)
 */
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    // Unpack arguments (Tensor* pointers from runtime)
    __gm__ Tensor *a = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *b = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *c = reinterpret_cast<__gm__ Tensor *>(args[2]);
    __gm__ Tensor *config = reinterpret_cast<__gm__ Tensor *>(args[3]);

    __gm__ int64_t *cfg = reinterpret_cast<__gm__ int64_t *>(config->buffer.addr);
    const uint64_t tile_size = static_cast<uint64_t>(cfg[0]);
    uint64_t tile_elems = tile_size * tile_size;
    const int num_tiles = static_cast<uint64_t>(cfg[3]);
}