// Batched PV Matmul Kernel: for each batch b, pij(M, K) @ vj(K, N) -> oi_new(M, N)
//
// Processes batch_count batches in a single kernel invocation.
// Per-batch addresses are computed from global tensor bases + block_table lookup.
//
// Template: M=q_tile, K=block_size, N=head_dim

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
static __aicore__ void pv_matmul_batch_impl(
    __gm__ TensorData* pij_batch,
    __gm__ TensorData* value_cache,
    __gm__ TensorData* oi_new_batch,
    uint64_t block_table_ptr,
    uint64_t batch_count,
    uint64_t block_idx,
    uint64_t block_num) {

    __gm__ half* pij_base = reinterpret_cast<__gm__ half*>(pij_batch->buffer.addr);
    __gm__ half* val_base = reinterpret_cast<__gm__ half*>(value_cache->buffer.addr);
    __gm__ float* oi_base = reinterpret_cast<__gm__ float*>(oi_new_batch->buffer.addr);
    // Block table values are always non-negative (physical block indices)
    __gm__ int32_t* bt = reinterpret_cast<__gm__ int32_t*>(block_table_ptr);

    using GlobalA = GlobalTensor<half, Shape<1, 1, 1, M, K>, Stride<M * K, M * K, M * K, K, 1>>;
    using GlobalB = GlobalTensor<half, Shape<1, 1, 1, K, N>, Stride<K * N, K * N, K * N, N, 1>>;
    using GlobalOut = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<M * N, M * N, M * N, N, 1>>;

    using TileMatA = Tile<TileType::Mat, half, M, K, BLayout::ColMajor, M, K, SLayout::RowMajor, 512>;
    using TileMatB = Tile<TileType::Mat, half, K, N, BLayout::ColMajor, K, N, SLayout::RowMajor, 512>;

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

    for (uint64_t b = 0; b < batch_count; b++) {
        __gm__ half* pij_addr = pij_base + b * M * K;
        int32_t phys_block = bt[b * block_num + block_idx];
        __gm__ half* vj_addr = val_base + (uint64_t)phys_block * K * N;
        __gm__ float* oi_addr = oi_base + b * M * N;

        GlobalA pijGlobal(pij_addr);
        GlobalB vjGlobal(vj_addr);
        GlobalOut oiGlobal(oi_addr);

        TLOAD(aMatTile, pijGlobal);
        TLOAD(bMatTile, vjGlobal);

        set_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE1, EVENT_ID0);

        TMOV(aTile, aMatTile);
        TMOV(bTile, bMatTile);

        set_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);
        wait_flag(PIPE_MTE1, PIPE_M, EVENT_ID0);

        TMATMUL(cTile, aTile, bTile);

        set_flag(PIPE_M, PIPE_FIX, EVENT_ID0);
        wait_flag(PIPE_M, PIPE_FIX, EVENT_ID0);

        TSTORE(oiGlobal, cTile);

        if (b + 1 < batch_count) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* pij_batch = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* value_cache = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* oi_new_batch = reinterpret_cast<__gm__ TensorData*>(args[2]);
    uint64_t block_table_ptr = static_cast<uint64_t>(args[3]);
    uint64_t batch_count = static_cast<uint64_t>(args[4]);
    uint64_t block_idx = static_cast<uint64_t>(args[5]);
    uint64_t block_num = static_cast<uint64_t>(args[6]);

    pv_matmul_batch_impl<16, 16, 16>(
        pij_batch, value_cache, oi_new_batch,
        block_table_ptr, batch_count, block_idx, block_num);
}
