// QK Matmul Kernel: qi(M, K) @ kj.T(K, N) -> sij(M, N)
//
// Fixed tile size: (16, 16) @ (16, 16).T -> (16, 16)
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
static __aicore__ void qk_matmul_impl(__gm__ Tensor* qi, __gm__ Tensor* kj, __gm__ Tensor* sij) {
    __gm__ half* qi_addr = reinterpret_cast<__gm__ half*>(qi->buffer.addr);
    __gm__ half* kj_addr = reinterpret_cast<__gm__ half*>(kj->buffer.addr);
    __gm__ float* sij_addr = reinterpret_cast<__gm__ float*>(sij->buffer.addr);

    // qi (M, K) fp16 in ND (row-major) layout
    using GlobalA = GlobalTensor<half, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    // kj stored as (N, K) row-major = (K, N) column-major -> DN layout
    using GlobalB = GlobalTensor<half, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    GlobalA qiGlobal(qi_addr + qi->start_offset);
    GlobalB kjGlobal(kj_addr + kj->start_offset);
    GlobalOut sijGlobal(sij_addr + sij->start_offset);

    // L1 Mat tiles: A is standard ND, B uses transposed-B pattern (RowMajor/ColMajor)
    using TileMatA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, half, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

    // L0 tiles
    using LeftTile = TileLeft<half, M, K, M, K>;
    using RightTile = TileRight<half, K, N, K, N>;
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

    // Load A and B to L1
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
    __gm__ Tensor* qi = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* kj = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* sij = reinterpret_cast<__gm__ Tensor*>(args[2]);

    qk_matmul_impl<16, 16, 16>(qi, kj, sij);
}
