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
// Optimizations:
//   - Double-buffered L1 tiles (ping/pong for A and B)
//   - TLOAD(next pij+vj) overlaps with TMATMUL_ACC(current) via MTE2/PIPE_M parallelism
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

    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, pto::Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, pto::Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, pto::Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // Double-buffered L1 tiles (ping/pong)
    TileMatA aMatTile_ping, aMatTile_pong;
    TileMatB bMatTile_ping, bMatTile_pong;
    TASSIGN(aMatTile_ping, 0x0);
    TASSIGN(aMatTile_pong, 0x10000);
    TASSIGN(bMatTile_ping, 0x20000);
    TASSIGN(bMatTile_pong, 0x30000);

    LeftTile aTile;
    RightTile bTile;
    AccTile cTile;
    TASSIGN(aTile, 0x0);
    TASSIGN(bTile, 0x0);
    TASSIGN(cTile, 0x0);

    GlobalOut oiGlobal(oi_base);

    // Pre-load first iteration's tiles into ping buffers
    GlobalA pijGlobal_0(pij_base);
    GlobalB vjGlobal_0(val_base + block_indices[0] * K * N);
    TLOAD(aMatTile_ping, pijGlobal_0);
    TLOAD(bMatTile_ping, vjGlobal_0);

    for (uint64_t i = 0; i < n_blocks; i++) {
        // Select current buffers based on iteration parity
        TileMatA& curA = (i % 2 == 0) ? aMatTile_ping : aMatTile_pong;
        TileMatB& curB = (i % 2 == 0) ? bMatTile_ping : bMatTile_pong;

        // Wait for current TLOAD to complete
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // Wait for previous matmul to complete (L0A/L0B safe to overwrite)
        if (i > 0) {
            wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
        }

        TMOV(aTile, curA);
        TMOV(bTile, curB);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        if (i == 0) {
            TMATMUL(cTile, aTile, bTile);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile, bTile);
        }

        // Prefetch next iteration's data (MTE2 overlaps with matmul completion)
        if (i + 1 < n_blocks) {
            // Signal matmul completion for next iteration's TMOV guard
            set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);
            TileMatA& nxtA = (i % 2 == 0) ? aMatTile_pong : aMatTile_ping;
            TileMatB& nxtB = (i % 2 == 0) ? bMatTile_pong : bMatTile_ping;
            GlobalA pijGlobal_next(pij_base + (i + 1) * M * K);
            GlobalB vjGlobal_next(val_base + block_indices[i + 1] * K * N);
            TLOAD(nxtA, pijGlobal_next);
            TLOAD(nxtB, vjGlobal_next);
        }
    }

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(oiGlobal, cTile);
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
