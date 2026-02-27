/**
 * Tile-based Matrix Multiplication Kernel (Cube Core)
 *
 * Computes: output = input_a @ input_b (64x64 tile matmul)
 * Uses TMATMUL instruction
 *
 * Args (TensorData*):
 *   args[0] = input_a (INPUT)
 *   args[1] = input_b (INPUT)
 *   args[2] = output  (OUTPUT)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>
#include <pto/common/pto_tile.hpp>

#include "tensor.h"

using namespace pto;

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

static __aicore__ void gemm_tile_impl(
    __gm__ TensorData* input_a_tensor,
    __gm__ TensorData* input_b_tensor,
    __gm__ TensorData* output_tensor) {

    __gm__ float* input_a = reinterpret_cast<__gm__ float*>(input_a_tensor->buffer.addr) + input_a_tensor->start_offset;
    __gm__ float* input_b = reinterpret_cast<__gm__ float*>(input_b_tensor->buffer.addr) + input_b_tensor->start_offset;
    __gm__ float* output  = reinterpret_cast<__gm__ float*>(output_tensor->buffer.addr)  + output_tensor->start_offset;

    constexpr int TILE = 64;
    constexpr int blockAlign = C0_SIZE_BYTE / sizeof(float);
    constexpr int M = CeilAlign<int>(TILE, 16);
    constexpr int K = CeilAlign<int>(TILE, blockAlign);
    constexpr int N = CeilAlign<int>(TILE, blockAlign);

    using GlobalDataA = GlobalTensor<float, Shape<1, 1, 1, TILE, TILE>,
        Stride<1 * TILE * TILE, 1 * TILE * TILE, TILE * TILE, TILE, 1>>;
    using GlobalDataB = GlobalTensor<float, Shape<1, 1, 1, TILE, TILE>,
        Stride<1 * TILE * TILE, 1 * TILE * TILE, TILE * TILE, TILE, 1>>;
    using GlobalDataC = GlobalTensor<float, Shape<1, 1, 1, TILE, TILE>,
        Stride<1 * TILE * TILE, 1 * TILE * TILE, TILE * TILE, TILE, 1>>;

    GlobalDataA src0Global(input_a);
    GlobalDataB src1Global(input_b);
    GlobalDataC dstGlobal(output);

    using TileMatA = Tile<TileType::Mat, float, M, K, BLayout::ColMajor, TILE, TILE, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, float, K, N, BLayout::ColMajor, TILE, TILE, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<float, M, K, TILE, TILE>;
    using RightTile = TileRight<float, K, N, TILE, TILE>;
    using AccTile = TileAcc<float, M, N, TILE, TILE>;

    TileMatA aMatTile;
    TileMatB bMatTile;
    TASSIGN(aMatTile, 0x0);
    TASSIGN(bMatTile, 0x20000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    TLOAD(aMatTile, src0Global);
    TLOAD(bMatTile, src1Global);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(dstGlobal, cTile);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* input_a = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* input_b = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* output  = reinterpret_cast<__gm__ TensorData*>(args[2]);

    gemm_tile_impl(input_a, input_b, output);
}
