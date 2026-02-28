// Softmax Preparation Kernel (AIV) with partial block masking
//
// Fixed tile size: sij is (16, 16)
//
// For partial blocks (valid_len < N), positions [valid_len, N) in sij are
// filled with -inf before softmax, ensuring exp(-inf)=0 so that invalid
// key positions contribute zero attention weight.
//
// Uses TFILLPAD_INPLACE for vector pipeline state setup, then patches with
// scalar SetValue writes to fix a hardware bug in TFILLPAD's vcopy broadcast
// path at small N (N=16).
//
// Computes:
//   sij_masked = pad(sij, valid_len, -inf)
//   sij_scale = sij_masked * scale
//   mij = row_max(sij_scale)        -> (M, 1)
//   pij = exp(sij_scale - mij)      -> (M, N)
//   lij = row_sum(pij)              -> (M, 1)

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
static __aicore__ void softmax_prepare_impl(__gm__ TensorData* sij,
    float scale_value,
    __gm__ TensorData* pij,
    __gm__ TensorData* mij,
    __gm__ TensorData* lij) {
    uint64_t valid_len = static_cast<uint64_t>(sij->shapes[1]);
    __gm__ float* sij_addr = reinterpret_cast<__gm__ float*>(sij->buffer.addr);
    __gm__ half* pij_addr = reinterpret_cast<__gm__ half*>(pij->buffer.addr);
    __gm__ float* mij_addr = reinterpret_cast<__gm__ float*>(mij->buffer.addr);
    __gm__ float* lij_addr = reinterpret_cast<__gm__ float*>(lij->buffer.addr);

    constexpr int kAlignedRows = ((M * sizeof(float) + 31) / 32) * (32 / sizeof(float));

    using GlobalDataMxN = GlobalTensor<float, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalDataMxN_f16 = GlobalTensor<half, Shape<1, 1, 1, M, N>, Stride<1, 1, 1, N, 1>>;
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalDataMxN sijGlobal(sij_addr + sij->start_offset);
    GlobalDataMxN_f16 pijGlobal(pij_addr + pij->start_offset);
    GlobalScalarDN mijGlobal(mij_addr + mij->start_offset);
    GlobalScalarDN lijGlobal(lij_addr + lij->start_offset);

    // Dynamic-cols tile: marks which columns are valid for TFILLPAD boundary
    using TileSijDyn = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, -1>;
    // Padded tile: TFILLPAD_INPLACE fills positions [valid_len, N) with -inf
    using TileSijPad = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N, SLayout::NoneBox, 512, PadValue::Min>;

    using TileVecMxN = Tile<TileType::Vec, float, M, N, BLayout::RowMajor, M, N>;
    using TileVecMxN_f16 = Tile<TileType::Vec, half, M, N, BLayout::RowMajor, M, N>;
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, M, 1>;

    TileVecMxN sijTile;
    TileSijDyn sijDynTile(static_cast<size_t>(valid_len));
    TileSijPad sijPadTile;
    TileVecMxN pijTile;
    TileVecMxN tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;
    TileVecMxN_f16 pijF16Tile;

    TASSIGN(sijTile, 0x0);
    TASSIGN(sijDynTile, 0x0);
    TASSIGN(sijPadTile, 0x0);
    TASSIGN(pijTile, M * N * sizeof(float));
    TASSIGN(tmpTile, 2 * M * N * sizeof(float));
    TASSIGN(maxTile, 3 * M * N * sizeof(float));
    TASSIGN(sumTile, 3 * M * N * sizeof(float) + kAlignedRows * sizeof(float));
    TASSIGN(pijF16Tile, 3 * M * N * sizeof(float) + 2 * kAlignedRows * sizeof(float));

    // Load full sij (M, N) tile from GM - all N columns including garbage for partial blocks
    TLOAD(sijTile, sijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Mask columns [valid_len, N) with -inf.
    // Use TFILLPAD_INPLACE for the main fill, then patch with SetValue for
    // cases where TFILLPAD's vcopy broadcast path fails at small N.
    TFILLPAD_INPLACE(sijPadTile, sijDynTile);
    // Patch: SetValue ensures correctness for valid_len <= N/2 where
    // TFILLPAD's PadRightRemainingRows vcopy has a hardware issue.
    if (valid_len < static_cast<uint64_t>(N)) {
        // Cross-pipeline sync: wait for PIPE_V vcopy in TFILLPAD to complete
        // before PIPE_S scalar SetValue writes to the same UB addresses.
        // Without this, PIPE_V vcopy and PIPE_S SetValue race on UB memory,
        // causing sporadic FAIL when vcopy finishes after SetValue.
        // Pattern from TFillPad.hpp Handle32BAlignedPad_Byte (PtoSetWaitFlag).
        set_flag(PIPE_V, PIPE_S, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
        constexpr float NEG_INF = -__builtin_huge_valf();
        for (int r = 0; r < M; r++) {
            for (uint64_t c = valid_len; c < N; c++) {
                sijTile.SetValue(static_cast<uint32_t>(r * N + c), NEG_INF);
            }
        }
        // Ensure PIPE_S scalar UB writes are visible to subsequent PIPE_V ops.
        // dsb(DSB_UB) is a hardware-only intrinsic; in simulation there are no
        // real pipelines so the barrier is unnecessary and DSB_UB is undefined.
#ifdef DSB_UB
        dsb(DSB_UB);
#endif
    }

    TMULS(sijTile, sijTile, scale_value);
    pipe_barrier(PIPE_V);
    TROWMAX(maxTile, sijTile, tmpTile);
    pipe_barrier(PIPE_V);
    TROWEXPANDSUB(pijTile, sijTile, maxTile);
    pipe_barrier(PIPE_V);
    TEXP(pijTile, pijTile);
    // Truncate pij to fp16 first, then compute lij from truncated values (matches golden)
    TCVT(pijF16Tile, pijTile, RoundMode::CAST_ROUND);
    TCVT(pijTile, pijF16Tile, RoundMode::CAST_ROUND);
    TROWSUM(sumTile, pijTile, tmpTile);

    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(mijGlobal, maxTile);
    TSTORE(lijGlobal, sumTile);
    TSTORE(pijGlobal, pijF16Tile);
}

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ TensorData* sij = reinterpret_cast<__gm__ TensorData*>(args[0]);
    union {
        uint64_t u;
        float f;
    } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[1]);
    float scale_value = scale_conv.f;
    __gm__ TensorData* pij = reinterpret_cast<__gm__ TensorData*>(args[2]);
    __gm__ TensorData* mij = reinterpret_cast<__gm__ TensorData*>(args[3]);
    __gm__ TensorData* lij = reinterpret_cast<__gm__ TensorData*>(args[4]);

    softmax_prepare_impl<16, 16>(sij, scale_value, pij, mij, lij);
}
