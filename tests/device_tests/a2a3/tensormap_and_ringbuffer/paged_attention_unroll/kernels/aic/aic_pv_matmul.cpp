// SplitK PV Matmul Kernel: Accumulated P @ V across n_blocks
//
// Processes n_blocks blocks using SplitK accumulation pattern:
//   Block 0: TMATMUL(C, A, B)       — initialize accumulator
//   Block i: TMATMUL_ACC(C, C, A, B) — accumulate into same C
//
// Per-block pij addresses: contiguous slices of pij_buf (n_blocks * M * K)
// Per-block vj addresses: value_cache base + block_indices lookup
// Single output: oi_new (M, N) fp32 = sum of P_i @ V_i across all blocks
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128) -> (16, 128)
//   Case2: (64,  64) @ ( 64, 128) -> (64, 128)
//
// pij is bfloat16 (from softmax_prepare TCVT).
// vj is stored as (K, N) = (block_size, head_dim) in row-major (ND) layout.

#include <cstdint>
#include <pto/pto-inst.hpp>

#define N_UNROLL 64

#include "tensor.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <int M, int K, int N>
static __aicore__ void pv_matmul_n_impl(
    __gm__ bfloat16_t* pij_base,
    __gm__ bfloat16_t* val_base,
    __gm__ float* oi_base,
    uint64_t n_blocks,
    uint64_t* block_indices) {

    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

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

    GlobalOut oiGlobal(oi_base);

    for (uint64_t i = 0; i < n_blocks; i++) {
        GlobalA pijGlobal(pij_base + i * M * K);
        GlobalB vjGlobal(val_base + block_indices[i] * K * N);

        TLOAD(aMatTile, pijGlobal);
        TLOAD(bMatTile, vjGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (i == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }

        set_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_MTE2, EVENT_ID0);
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(oiGlobal, cTile);

    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* pij_buf = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* value_cache = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* oi_new = reinterpret_cast<__gm__ TensorData*>(args[2]);
    uint64_t n_blocks = static_cast<uint64_t>(args[3]);
    uint64_t block_indices[N_UNROLL];
    for (uint64_t j = 0; j < n_blocks; j++) {
        block_indices[j] = static_cast<uint64_t>(args[4 + j]);
    }

    __gm__ bfloat16_t* pij_base = reinterpret_cast<__gm__ bfloat16_t*>(pij_buf->buffer.addr) + pij_buf->start_offset;
    __gm__ bfloat16_t* val_base = reinterpret_cast<__gm__ bfloat16_t*>(value_cache->buffer.addr);
    __gm__ float* oi_base = reinterpret_cast<__gm__ float*>(oi_new->buffer.addr) + oi_new->start_offset;

    uint64_t q_tile_size = static_cast<uint64_t>(pij_buf->shapes[0]);

    if (q_tile_size == 16) {
        pv_matmul_n_impl<16, 128, 128>(pij_base, val_base, oi_base, n_blocks, block_indices);
    } else {
        pv_matmul_n_impl<64, 64, 128>(pij_base, val_base, oi_base, n_blocks, block_indices);
    }
}
