// Online Softmax Update + Normalize Kernel (AIV)
//
// Fixed tile size: oi/oi_new are (16, 16), mij/lij/mi/li are 16-element vectors
//
// Scalar layout strategy:
//   M scalar floats stored contiguously in GM can be loaded as either:
//   - ND (kScalarRows, kScalarCols) RowMajor for element-wise ops (TMAX, TSUB, TEXP, TMUL, TADD)
//   - DN (kAlignedRows, 1) ColMajor for row-broadcast ops (TROWEXPANDMUL, TROWEXPANDDIV)
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
static __aicore__ void online_update_impl(__gm__ TensorData* mij,
    __gm__ TensorData* lij,
    __gm__ TensorData* oi_new,
    __gm__ TensorData* mi,
    __gm__ TensorData* li,
    __gm__ TensorData* oi,
    uint64_t is_first,
    uint64_t is_last,
    __gm__ TensorData* dst) {
    __gm__ float* mij_ptr = reinterpret_cast<__gm__ float*>(mij->buffer.addr);
    __gm__ float* lij_ptr = reinterpret_cast<__gm__ float*>(lij->buffer.addr);
    __gm__ float* oi_new_ptr = reinterpret_cast<__gm__ float*>(oi_new->buffer.addr);
    __gm__ float* mi_ptr = reinterpret_cast<__gm__ float*>(mi->buffer.addr);
    __gm__ float* li_ptr = reinterpret_cast<__gm__ float*>(li->buffer.addr);
    __gm__ float* oi_ptr = reinterpret_cast<__gm__ float*>(oi->buffer.addr);
    __gm__ float* dst_ptr = reinterpret_cast<__gm__ float*>(dst->buffer.addr);

    // Scalar tile dimensions for RowMajor layout:
    // kScalarCols = 32 bytes / 4 bytes per float = 8 floats per row (one 32-byte block)
    // kScalarRows = M / 8 (M=16 -> 2 rows)
    constexpr int kScalarCols = 32 / sizeof(float);
    constexpr int kScalarRows = M / kScalarCols;
    // Aligned rows for ColMajor DN tiles (32-byte alignment)
    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    // --- GlobalTensor types ---

    // Data (M, N) RowMajor
    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;

    // Scalar ND: M contiguous floats as (kScalarRows, kScalarCols) RowMajor
    using GlobalScalarND =
        GlobalTensor<float, Shape<1, 1, 1, kScalarRows, kScalarCols>, Stride<1, 1, 1, kScalarCols, 1>>;

    // Scalar DN: same M contiguous floats as (kAlignedRows, 1) ColMajor
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    // --- GlobalTensor instances ---

    GlobalDataMxN oiNewGlobal(oi_new_ptr + oi_new->start_offset);
    GlobalDataMxN oiGlobal(oi_ptr + oi->start_offset);
    GlobalDataMxN dstGlobal(dst_ptr + dst->start_offset);

    // ND globals for scalar element-wise operations
    GlobalScalarND mijGlobalND(mij_ptr + mij->start_offset);
    GlobalScalarND lijGlobalND(lij_ptr + lij->start_offset);
    GlobalScalarND miGlobalND(mi_ptr + mi->start_offset);
    GlobalScalarND liGlobalND(li_ptr + li->start_offset);

    // DN globals aliased to same GM for ColMajor reload (used after ND TSTORE)
    GlobalScalarDN mijGlobalDN(mij_ptr + mij->start_offset);
    GlobalScalarDN lijGlobalDN(lij_ptr + lij->start_offset);
    GlobalScalarDN liGlobalDN(li_ptr + li->start_offset);

    // --- Tile types ---

    using TileDataMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileScalarND =
        Tile<TileType::Vec, float, kScalarRows, kScalarCols, BLayout::RowMajor, kScalarRows, kScalarCols>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    // --- UB memory layout ---

    constexpr int kDataBytes = M * N * sizeof(float);
    constexpr int kScalarNDBytes = kScalarRows * kScalarCols * sizeof(float);
    constexpr int kScalarDNBytes = kAlignedRows * sizeof(float);

    // Data tiles
    TileDataMxN oiNewTile;
    TileDataMxN oiTile;

    // Scalar ND tiles for element-wise arithmetic
    TileScalarND mijND, lijND, miND, liND;
    TileScalarND miNewND, alphaND, betaND, tmpND;

    // Scalar DN tiles for TROWEXPAND operations
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

    if (is_first) {
        // --- First block: copy inputs to accumulators ---
        TLOAD(oiNewTile, oiNewGlobal);
        TLOAD(mijND, mijGlobalND);
        TLOAD(lijND, lijGlobalND);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Passthrough to MTE3 (no V compute needed)
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(miGlobalND, mijND);    // mi = mij
        TSTORE(liGlobalND, lijND);    // li = lij
        TSTORE(oiGlobal, oiNewTile);  // oi = oi_new

        if (is_last) {
            // Single block: normalize dst = oi_new / lij
            // lij stored to li buffer in ND format; reload as DN for TROWEXPANDDIV
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
        // --- Subsequent blocks: accumulate ---

        // Phase 1: Load all inputs
        TLOAD(oiNewTile, oiNewGlobal);
        TLOAD(oiTile, oiGlobal);
        TLOAD(mijND, mijGlobalND);
        TLOAD(lijND, lijGlobalND);
        TLOAD(miND, miGlobalND);
        TLOAD(liND, liGlobalND);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

        // Phase 2: Scalar arithmetic in RowMajor (kScalarRows, kScalarCols)
        // pipe_barrier(PIPE_V) required between each dependent vector operation
        // to resolve RAW hazards on shared UB tiles.
        TMAX(miNewND, miND, mijND);  // mi_new = max(mi, mij)
        pipe_barrier(PIPE_V);
        TSUB(alphaND, miND, miNewND);  // alpha = mi - mi_new
        pipe_barrier(PIPE_V);
        TEXP(alphaND, alphaND);  // alpha = exp(mi - mi_new)
        pipe_barrier(PIPE_V);
        TSUB(betaND, mijND, miNewND);  // beta = mij - mi_new
        pipe_barrier(PIPE_V);
        TEXP(betaND, betaND);  // beta = exp(mij - mi_new)
        pipe_barrier(PIPE_V);
        TMUL(liND, alphaND, liND);  // li = alpha * li
        pipe_barrier(PIPE_V);
        TMUL(tmpND, betaND, lijND);  // tmp = beta * lij
        pipe_barrier(PIPE_V);
        TADD(liND, liND, tmpND);  // li = alpha * li + beta * lij (= li_new)

        // Phase 3: Store scalar results to GM (ND format)
        // mi_new -> mi accumulator, li_new -> li accumulator
        // alpha -> mij buffer (reuse), beta -> lij buffer (reuse)
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(miGlobalND, miNewND);   // persist mi_new
        TSTORE(liGlobalND, liND);      // persist li_new
        TSTORE(mijGlobalND, alphaND);  // temp: alpha to mij buffer
        TSTORE(lijGlobalND, betaND);   // temp: beta to lij buffer

        // Phase 4: Reload alpha, beta (and li if last) as ColMajor DN
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        TLOAD(alphaDN, mijGlobalDN);  // alpha from mij buffer as DN
        TLOAD(betaDN, lijGlobalDN);   // beta from lij buffer as DN
        if (is_last) {
            TLOAD(liDN, liGlobalDN);  // li_new from li buffer as DN
        }
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // Phase 5: Scale data tiles using row-broadcast multiply
        TROWEXPANDMUL(oiTile, oiTile, alphaDN);       // oi *= alpha
        TROWEXPANDMUL(oiNewTile, oiNewTile, betaDN);  // oi_new *= beta
        pipe_barrier(PIPE_V);
        TADD(oiTile, oiTile, oiNewTile);  // oi = alpha*oi + beta*oi_new

        if (is_last) {
            // Phase 6: Normalize and output
            pipe_barrier(PIPE_V);
            TROWEXPANDDIV(oiTile, oiTile, liDN);  // dst = oi / li_new
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(dstGlobal, oiTile);
        } else {
            // Phase 6: Store updated accumulators
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
            TSTORE(oiGlobal, oiTile);
        }
    }
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* mij = reinterpret_cast<__gm__ TensorData*>(args[0]);
    __gm__ TensorData* lij = reinterpret_cast<__gm__ TensorData*>(args[1]);
    __gm__ TensorData* oi_new = reinterpret_cast<__gm__ TensorData*>(args[2]);
    __gm__ TensorData* mi = reinterpret_cast<__gm__ TensorData*>(args[3]);
    __gm__ TensorData* li = reinterpret_cast<__gm__ TensorData*>(args[4]);
    __gm__ TensorData* oi = reinterpret_cast<__gm__ TensorData*>(args[5]);
    __gm__ TensorData* dst = reinterpret_cast<__gm__ TensorData*>(args[6]);
    uint64_t is_first = static_cast<uint64_t>(args[7]);
    uint64_t is_last = static_cast<uint64_t>(args[8]);

    online_update_impl<16, 16>(mij, lij, oi_new, mi, li, oi, is_first, is_last, dst);
}
