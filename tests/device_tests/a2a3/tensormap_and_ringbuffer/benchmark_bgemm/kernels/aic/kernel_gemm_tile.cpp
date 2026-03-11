/**
 * Tile-based Matrix Multiplication Kernel (Cube Core)
 *
 * Computes: output = input_a @ input_b (tile_size x tile_size tile matmul)
 * Uses TMATMUL instruction
 *
 * Tile size is determined by golden.py configuration and passed through
 * tensor shapes from orchestration.
 *
 * Args (TensorData*):
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

template <int TILE>
static __aicore__ void gemm_tile_impl(
    __gm__ float* input_a,
    __gm__ float* input_b,
    __gm__ float* output) {

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

    set_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_FIX, PIPE_MTE2, EVENT_ID0);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* input_a = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* input_b = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* output  = reinterpret_cast<__gm__ TensorData*>(args[2]);
    __gm__ TensorData* config  = reinterpret_cast<__gm__ TensorData*>(args[3]);

    __gm__ int64_t* cfg = reinterpret_cast<__gm__ int64_t*>(config->buffer.addr);
    uint64_t tile_size = static_cast<uint64_t>(cfg[0]);
    uint64_t tile_elems = tile_size * tile_size;
    int num_tiles = static_cast<uint64_t>(cfg[3]);

    __gm__ float* base_a = reinterpret_cast<__gm__ float*>(input_a->buffer.addr) + input_a->start_offset;
    __gm__ float* base_b = reinterpret_cast<__gm__ float*>(input_b->buffer.addr) + input_b->start_offset;
    __gm__ float* base_c = reinterpret_cast<__gm__ float*>(output->buffer.addr) + output->start_offset;

    for (int tile_idx = 0; tile_idx < num_tiles; tile_idx++) {
        __gm__ float* a_ptr = base_a + (tile_idx * tile_elems);
        __gm__ float* b_ptr = base_b + (tile_idx * tile_elems);
        __gm__ float* c_ptr = base_c + (tile_idx * tile_elems);

        switch (tile_size) {
            case 16:  gemm_tile_impl<16>(a_ptr, b_ptr, c_ptr);  break;
            case 32:  gemm_tile_impl<32>(a_ptr, b_ptr, c_ptr);  break;
            case 64:  gemm_tile_impl<64>(a_ptr, b_ptr, c_ptr);  break;
            case 128: gemm_tile_impl<128>(a_ptr, b_ptr, c_ptr); break;
            default: break;
        }
    }
}
