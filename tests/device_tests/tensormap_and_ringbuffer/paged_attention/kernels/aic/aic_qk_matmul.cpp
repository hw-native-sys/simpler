// QK Matmul Kernel: qi(M, K) @ kj.T(K, N) -> sij(M, N)
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128).T -> (16, 128)
//   Case2: (64, 128) @ (128,  64).T -> (64,  64)
//
// kj is stored as (N, K) = (block_size, head_dim) in row-major memory.
// This is equivalent to (K, N) in column-major (DN) layout.
// Using DN GlobalB + RowMajor/ColMajor TileMatB to handle the transposed B pattern.

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

template <int M, int K, int N>
static __aicore__ void qk_matmul_impl(__gm__ TensorData* qi, __gm__ TensorData* kj, __gm__ TensorData* sij) {
    __gm__ bfloat16_t* qi_addr = reinterpret_cast<__gm__ bfloat16_t*>(qi->buffer.addr);
    __gm__ bfloat16_t* kj_addr = reinterpret_cast<__gm__ bfloat16_t*>(kj->buffer.addr);
    __gm__ float* sij_addr = reinterpret_cast<__gm__ float*>(sij->buffer.addr);

    // qi (M, K) bf16 in ND (row-major) layout
    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    // kj stored as (N, K) row-major = (K, N) column-major -> DN layout
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    GlobalA qiGlobal(qi_addr + qi->start_offset);
    GlobalB kjGlobal(kj_addr + kj->start_offset);
    GlobalOut sijGlobal(sij_addr + sij->start_offset);

    // L1 Mat tiles: A is standard ND, B uses transposed-B pattern (RowMajor/ColMajor)
    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

    // L0 tiles
    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

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

    // // Load A and B to L1
    TLOAD(aMatTile, qiGlobal);
    TLOAD(bMatTile, kjGlobal);

    set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

    // Move from L1 to L0A/L0B
    TMOV(aTile, aMatTile);
    TMOV(bTile, bMatTile);

    set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

    // Matmul
    TMATMUL(cTile, aTile, bTile);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

    TSTORE(sijGlobal, cTile);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* qi = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* kj = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* sij = reinterpret_cast<__gm__ TensorData*>(args[2]);
    uint64_t q_tile_size = static_cast<uint64_t>(qi->shapes[0]);
    // args[4] = head_dim (128), args[5] = block_size

    if (q_tile_size == 16) {
        qk_matmul_impl<16, 128, 128>(qi, kj, sij);
    } else {
        qk_matmul_impl<64, 128, 64>(qi, kj, sij);
    }
}
