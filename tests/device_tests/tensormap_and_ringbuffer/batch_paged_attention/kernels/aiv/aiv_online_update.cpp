// Batched Online Softmax Update + Normalize Kernel (AIV)
//
// Processes batch_count batches in a single kernel invocation.
// For each batch b, updates accumulators mi/li/oi with new block's mij/lij/oi_new.
// On is_last, normalizes and writes to the output tensor at the correct batch offset.
//
// Supports two tile configurations via runtime dispatch:
//   Case1: (16, 128) -- q_tile=16, head_dim=128
//   Case2: (64, 128) -- q_tile=64, head_dim=128
//
// Scalar layout strategy (unchanged from unbatched version):
//   M scalar floats stored contiguously in GM can be loaded as either:
//   - ND (kScalarRows, kScalarCols) RowMajor for element-wise ops
//   - DN (kAlignedRows, 1) ColMajor for row-broadcast ops
//   Conversion between layouts uses GM round-trip: ND TSTORE -> DN TLOAD.

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

template <int M, int N>
static __aicore__ void online_update_batch_impl(
    __gm__ TensorData* mij_batch,
    __gm__ TensorData* lij_batch,
    __gm__ TensorData* oi_new_batch,
    __gm__ TensorData* mi_batch,
    __gm__ TensorData* li_batch,
    __gm__ TensorData* oi_batch,
    __gm__ TensorData* out,
    uint64_t is_first,
    uint64_t is_last,
    uint64_t batch_count,
    uint64_t q_offset,
    uint64_t num_heads,
    uint64_t batch_start) {

    __gm__ float* mij_base = reinterpret_cast<__gm__ float*>(mij_batch->buffer.addr);
    __gm__ float* lij_base = reinterpret_cast<__gm__ float*>(lij_batch->buffer.addr);
    __gm__ float* oi_new_base = reinterpret_cast<__gm__ float*>(oi_new_batch->buffer.addr);
    __gm__ float* mi_base = reinterpret_cast<__gm__ float*>(mi_batch->buffer.addr);
    __gm__ float* li_base = reinterpret_cast<__gm__ float*>(li_batch->buffer.addr);
    __gm__ float* oi_base = reinterpret_cast<__gm__ float*>(oi_batch->buffer.addr);
    __gm__ float* out_base = reinterpret_cast<__gm__ float*>(out->buffer.addr);

    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = M / kScalarCols;
    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarND =
        GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kScalarCols>, Stride<1, 1, 1, kScalarCols, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    using TileDataMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    constexpr int kDataBytes = M * N * sizeof(float);
    constexpr int kScalarNDBytes = kScalarRows * kScalarCols * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);

    TileDataMxN oiNewTile;
    TileDataMxN oiTile;

    TileScalarND mijND, lijND, miND, liND;
    TileScalarND miNewND, alphaND, betaND, tmpND;

    TileScalarDN alphaDN, betaDN, liDN;

    TASSIGN(oiNewTile, 0);
    TASSIGN(oiTile, kDataBytes);
    TASSIGN(mijND, 2 * kDataBytes);
    TASSIGN(lijND, 2 * kDataBytes + kScalarNDBytes);
    TASSIGN(miND, 2 * kDataBytes + 2 * kScalarNDBytes);
    TASSIGN(liND, 2 * kDataBytes + 3 * kScalarNDBytes);
    TASSIGN(miNewND, 2 * kDataBytes + 4 * kScalarNDBytes);
    TASSIGN(alphaND, 2 * kDataBytes + 5 * kScalarNDBytes);
    TASSIGN(betaND, 2 * kDataBytes + 6 * kScalarNDBytes);
    TASSIGN(tmpND, 2 * kDataBytes + 7 * kScalarNDBytes);
    TASSIGN(alphaDN, 2 * kDataBytes + 8 * kScalarNDBytes);
    TASSIGN(betaDN, 2 * kDataBytes + 8 * kScalarNDBytes + kScalarDNBytes);
    TASSIGN(liDN, 2 * kDataBytes + 8 * kScalarNDBytes + 2 * kScalarDNBytes);

    for (uint64_t b = 0; b < batch_count; b++) {
        __gm__ float* mij_ptr = mij_base + b * M;
        __gm__ float* lij_ptr = lij_base + b * M;
        __gm__ float* oi_new_ptr = oi_new_base + b * M * N;
        __gm__ float* mi_ptr = mi_base + b * M;
        __gm__ float* li_ptr = li_base + b * M;
        __gm__ float* oi_ptr = oi_base + b * M * N;
        __gm__ float* dst_ptr = out_base + ((batch_start + b) * num_heads + q_offset) * N;

        GlobalDataMxN oiNewGlobal(oi_new_ptr);
        GlobalDataMxN oiGlobal(oi_ptr);
        GlobalDataMxN dstGlobal(dst_ptr);

        GlobalScalarND mijGlobalND(mij_ptr);
        GlobalScalarND lijGlobalND(lij_ptr);
        GlobalScalarND miGlobalND(mi_ptr);
        GlobalScalarND liGlobalND(li_ptr);

        GlobalScalarDN mijGlobalDN(mij_ptr);
        GlobalScalarDN lijGlobalDN(lij_ptr);
        GlobalScalarDN liGlobalDN(li_ptr);

        if (is_first) {
            TLOAD(oiNewTile, oiNewGlobal);
            TLOAD(mijND, mijGlobalND);
            TLOAD(lijND, lijGlobalND);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(miGlobalND, mijND);
            TSTORE(liGlobalND, lijND);
            TSTORE(oiGlobal, oiNewTile);

            if (is_last) {
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                TLOAD(liDN, liGlobalDN);
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
                TROWEXPANDDIV(oiNewTile, oiNewTile, liDN);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(dstGlobal, oiNewTile);
            }
        } else {
            TLOAD(oiNewTile, oiNewGlobal);
            TLOAD(oiTile, oiGlobal);
            TLOAD(mijND, mijGlobalND);
            TLOAD(lijND, lijGlobalND);
            TLOAD(miND, miGlobalND);
            TLOAD(liND, liGlobalND);
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

            TMAX(miNewND, miND, mijND);
            pipe_barrier(PIPE_V);
            TSUB(alphaND, miND, miNewND);
            pipe_barrier(PIPE_V);
            TEXP(alphaND, alphaND);
            pipe_barrier(PIPE_V);
            TSUB(betaND, mijND, miNewND);
            pipe_barrier(PIPE_V);
            TEXP(betaND, betaND);
            pipe_barrier(PIPE_V);
            TMUL(liND, alphaND, liND);
            pipe_barrier(PIPE_V);
            TMUL(tmpND, betaND, lijND);
            pipe_barrier(PIPE_V);
            TADD(liND, liND, tmpND);

            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            TSTORE(miGlobalND, miNewND);
            TSTORE(liGlobalND, liND);
            TSTORE(mijGlobalND, alphaND);
            TSTORE(lijGlobalND, betaND);

            set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            TLOAD(alphaDN, mijGlobalDN);
            TLOAD(betaDN, lijGlobalDN);
            if (is_last) {
                TLOAD(liDN, liGlobalDN);
            }
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

            TROWEXPANDMUL(oiTile, oiTile, alphaDN);
            TROWEXPANDMUL(oiNewTile, oiNewTile, betaDN);
            pipe_barrier(PIPE_V);
            TADD(oiTile, oiTile, oiNewTile);

            if (is_last) {
                pipe_barrier(PIPE_V);
                TROWEXPANDDIV(oiTile, oiTile, liDN);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(dstGlobal, oiTile);
            } else {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
                TSTORE(oiGlobal, oiTile);
            }
        }

        if (b + 1 < batch_count) {
            pipe_barrier(PIPE_ALL);
        }
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* mij_batch = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* lij_batch = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* oi_new_batch = reinterpret_cast<__gm__ TensorData*>(args[2]);
    __gm__ TensorData* mi_batch = reinterpret_cast<__gm__ TensorData*>(args[3]);
    __gm__ TensorData* li_batch = reinterpret_cast<__gm__ TensorData*>(args[4]);
    __gm__ TensorData* oi_batch = reinterpret_cast<__gm__ TensorData*>(args[5]);
    __gm__ TensorData* out = reinterpret_cast<__gm__ TensorData*>(args[6]);
    uint64_t is_first = static_cast<uint64_t>(args[7]);
    uint64_t is_last = static_cast<uint64_t>(args[8]);
    uint64_t batch_count = static_cast<uint64_t>(args[9]);
    uint64_t q_offset = static_cast<uint64_t>(args[10]);
    uint64_t num_heads = static_cast<uint64_t>(args[11]);
    uint64_t batch_start = static_cast<uint64_t>(args[12]);

    uint64_t q_tile_size = static_cast<uint64_t>(mij_batch->shapes[0] / batch_count);

    if (q_tile_size == 16) {
        online_update_batch_impl<16, 128>(
            mij_batch, lij_batch, oi_new_batch,
            mi_batch, li_batch, oi_batch, out,
            is_first, is_last, batch_count, q_offset, num_heads, batch_start);
    } else {
        online_update_batch_impl<64, 128>(
            mij_batch, lij_batch, oi_new_batch,
            mi_batch, li_batch, oi_batch, out,
            is_first, is_last, batch_count, q_offset, num_heads, batch_start);
    }
}
