// Multi-block QK Matmul Kernel: qi(M, K) @ kj.T(K, N) -> sij(M, N) for each block
//
// Processes n_blocks blocks in a single kernel invocation.
// Per-block kj addresses computed from key_cache base + block_indices lookup.
// qi is shared across all blocks (same query head against different key blocks).
//
// Output layout: n_blocks contiguous (M, N) tiles stacked vertically.
// Block i occupies sij[i*M : (i+1)*M, 0:N].
//
// Optimizations:
//   - qi TLOAD hoisted before the loop (constant across all iterations)
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) @ (128, 128).T -> (16, 128)
//   Case2: (64, 128) @ (128,  64).T -> (64,  64)
//
// Template: M=q_tile, K=head_dim, N=block_size

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

#define N_UNROLL 64

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

template <int M, int K, int N>
static __aicore__ void qk_matmul_n_impl(
    __gm__ bfloat16_t* qi_base,
    __gm__ bfloat16_t* key_base,
    __gm__ float* sij_base,
    uint64_t n_blocks,
    uint64_t* block_indices) {

    using GlobalA = GlobalTensor<bfloat16_t, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<bfloat16_t, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, 1, K>, Layout::DN>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, bfloat16_t, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, bfloat16_t, K, N, BLayout::RowMajor, K, N, SLayout::ColMajor, 512>;

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

    // Hoist qi TLOAD before the loop (qi is constant across all blocks)
    GlobalA qiGlobal(qi_base);
    TLOAD(aMatTile, qiGlobal);

    for (uint64_t i = 0; i < n_blocks; i++) {
        GlobalB kjGlobal(key_base + block_indices[i] * N * K);
        GlobalOut sijGlobal(sij_base + i * M * N);

        // Load only B each iteration (qi already in L1 from hoist)
        TLOAD(bMatTile, kjGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        // TMOV qi from L1→L0A (re-copy since TMATMUL consumed L0A) and kj from L1→L0B
        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(cTile, aTile, bTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        TSTORE(sijGlobal, cTile);

        if (i + 1 < n_blocks) {
            pipe_barrier(PIPE_ALL);
        }
    }
    set_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_FIX, PIPE_S, EVENT_ID7);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* qi = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* key_cache = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* sij_buf = reinterpret_cast<__gm__ TensorData*>(args[2]);
    uint64_t n_blocks = static_cast<uint64_t>(args[3]);
    uint64_t block_indices[N_UNROLL];
    for (uint64_t j = 0; j < n_blocks; j++) {
        block_indices[j] = static_cast<uint64_t>(args[4 + j]);
    }

    __gm__ bfloat16_t* qi_base = reinterpret_cast<__gm__ bfloat16_t*>(qi->buffer.addr) + qi->start_offset;
    __gm__ bfloat16_t* key_base = reinterpret_cast<__gm__ bfloat16_t*>(key_cache->buffer.addr);
    __gm__ float* sij_base = reinterpret_cast<__gm__ float*>(sij_buf->buffer.addr) + sij_buf->start_offset;

    uint64_t q_tile_size = static_cast<uint64_t>(qi->shapes[0]);

    if (q_tile_size == 16) {
        qk_matmul_n_impl<16, 128, 128>(qi_base, key_base, sij_base, n_blocks, block_indices);
    } else {
        qk_matmul_n_impl<64, 128, 64>(qi_base, key_base, sij_base, n_blocks, block_indices);
    }
}
