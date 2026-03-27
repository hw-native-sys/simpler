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
//   - Double-buffered L1 tiles (ping/pong for A and B via MTE2)
//   - Double-buffered L0 tiles (ping/pong for L0A and L0B via MTE1)
//   - TLOAD(next) overlaps with TMATMUL(current) via MTE2/M-pipe parallelism
//   - Canonical 3-stage pipeline: TLOAD(MTE2) → TMOV(MTE1) → TMATMUL(M)
//   - Reverse-dependency events ensure buffer safety across iterations
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128) -> (16, 128)
//   Case2: (64,  64) @ ( 64, 128) -> (64, 128)
//
// pij is bfloat16 (from softmax_prepare TCVT).
// vj is stored as (K, N) = (block_size, head_dim) in row-major (ND) layout.

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
static __aicore__ void pv_matmul_n_impl(
    __gm__ bfloat16_t* pij_base,
    __gm__ bfloat16_t* val_base,
    __gm__ float* oi_base,
    uint64_t n_blocks,
    __gm__ int32_t* block_table) {

    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

    using LeftTile = TileLeft<bfloat16_t, M, K, M, K>;
    using RightTile = TileRight<bfloat16_t, K, N, K, N>;
    using AccTile = TileAcc<float, M, N, M, N>;

    // L1 memory layout: double-buffered A and B tiles (tightly packed)
    constexpr int kATileBytes = M * K * static_cast<int>(sizeof(bfloat16_t));
    constexpr int kBTileBytes = K * N * static_cast<int>(sizeof(bfloat16_t));

    TileMatA aMatTile[2];
    TileMatB bMatTile[2];
    TASSIGN(aMatTile[0], 0x0);
    TASSIGN(aMatTile[1], kATileBytes);
    TASSIGN(bMatTile[0], 2 * kATileBytes);
    TASSIGN(bMatTile[1], 2 * kATileBytes + kBTileBytes);

    // L0 memory layout: double-buffered L0A and L0B, single accumulator L0C
    LeftTile aTile[2];
    RightTile bTile[2];
    AccTile cTile;
    TASSIGN(aTile[0], 0x0);
    TASSIGN(aTile[1], kATileBytes);
    TASSIGN(bTile[0], 0x0);
    TASSIGN(bTile[1], kBTileBytes);
    TASSIGN(cTile, 0x0);

    GlobalOut oiGlobal(oi_base);

    // Seed reverse-dependency flags: all ping/pong buffers initially free
    //   PIPE_MTE1 → PIPE_MTE2: L1 buffer [0/1] safe for TLOAD to overwrite
    //   PIPE_M    → PIPE_MTE1: L0 buffer [0/1] safe for TMOV to overwrite
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    set_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    for (uint64_t i = 0; i < n_blocks; i++) {
        int cur = static_cast<int>(i % 2);
        GlobalA pijGlobal(pij_base + i * M * K);
        GlobalB vjGlobal(val_base + block_table[i] * K * N);

        // Stage 1: TLOAD (MTE2: GM → L1[cur])
        // Wait for MTE1 to release L1[cur] (reverse dep from previous iteration)
        wait_flag(PIPE_MTE1, PIPE_MTE2, (event_t)cur);
        TLOAD(aMatTile[cur], pijGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);   // forward: A in L1 ready
        TLOAD(bMatTile[cur], vjGlobal);
        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);   // forward: B in L1 ready

        // Stage 2: TMOV (MTE1: L1[cur] → L0[cur])
        // Wait for M-pipe to release L0[cur] (reverse dep from previous iteration)
        wait_flag(PIPE_M, PIPE_MTE1, (event_t)cur);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);  // forward: wait A loaded
        TMOV(aTile[cur], aMatTile[cur]);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID1);  // forward: wait B loaded
        TMOV(bTile[cur], bMatTile[cur]);
        set_flag(PIPE_MTE1, PIPE_MTE2, (event_t)cur); // reverse: release L1[cur]

        // Stage 3: TMATMUL (M-pipe: L0A[cur] × L0B[cur] → L0C)
        set_flag(PIPE_MTE1, PIPE_M, (event_t)cur);   // forward: L0[cur] ready
        wait_flag(PIPE_MTE1, PIPE_M, (event_t)cur);
        if (i == 0) {
            TMATMUL(cTile, aTile[cur], bTile[cur]);
        } else {
            TMATMUL_ACC(cTile, cTile, aTile[cur], bTile[cur]);
        }
        set_flag(PIPE_M, PIPE_MTE1, (event_t)cur);   // reverse: release L0[cur]
    }

    // Drain outstanding reverse-dependency flags
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE1, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_MTE1, EVENT_ID1);

    set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
    TSTORE(oiGlobal, cTile);

    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* pij_buf = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* value_cache = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* oi_new = reinterpret_cast<__gm__ Tensor*>(args[2]);
    uint64_t n_blocks = static_cast<uint64_t>(args[3]);
    __gm__ int32_t* block_table = reinterpret_cast<__gm__ int32_t*>(args[4]);

    __gm__ bfloat16_t* pij_base = reinterpret_cast<__gm__ bfloat16_t*>(pij_buf->buffer.addr) + pij_buf->start_offset;
    __gm__ bfloat16_t* val_base = reinterpret_cast<__gm__ bfloat16_t*>(value_cache->buffer.addr);
    __gm__ float* oi_base = reinterpret_cast<__gm__ float*>(oi_new->buffer.addr) + oi_new->start_offset;

    uint64_t q_tile_size = static_cast<uint64_t>(pij_buf->shapes[0]);

    if (q_tile_size == 16) {
        pv_matmul_n_impl<16, 128, 128>(pij_base, val_base, oi_base, n_blocks, block_table);
    } else {
        pv_matmul_n_impl<64, 64, 128>(pij_base, val_base, oi_base, n_blocks, block_table);
    }
}
