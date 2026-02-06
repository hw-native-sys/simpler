/**
 * Softmax Preparation Kernel (AIV) - 16x16 Version with PTO Tile API
 *
 * Uses PTO Tile instructions for a2a3 hardware:
 *   sij_scale = sij * scale
 *   mij = row_max(sij_scale)
 *   pij = exp(sij_scale - mij)
 *   lij = row_sum(pij)
 */
#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/constants.hpp>

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

constexpr int kRows = 16;  // num_heads
constexpr int kCols = 16;  // block_size

// Aligned row count for scalar tile (32-byte alignment)
constexpr int kAlignedRows = ((kRows * sizeof(float) + 31) / 32) * (32 / sizeof(float));  // = 16

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* sij = reinterpret_cast<__gm__ float*>(args[0]);
    union { uint64_t u; float f; } scale_conv;
    scale_conv.u = static_cast<uint64_t>(args[1]);
    float scale_value = scale_conv.f;
    __gm__ float* pij = reinterpret_cast<__gm__ float*>(args[2]);
    __gm__ float* mij = reinterpret_cast<__gm__ float*>(args[3]);
    __gm__ float* lij = reinterpret_cast<__gm__ float*>(args[4]);

    // GlobalTensor definitions
    using GlobalData16x16 = GlobalTensor<float, Shape<1, 1, 1, kRows, kCols>, Stride<1, 1, 1, kCols, 1>>;
    // For scalar output: DN layout (column-major storage for row reduction results)
    using GlobalScalarDN = GlobalTensor<float, Shape<1, 1, 1, kAlignedRows, 1>, Stride<1, 1, 1, 1, 1>, Layout::DN>;

    GlobalData16x16 sijGlobal(sij);
    GlobalData16x16 pijGlobal(pij);
    GlobalScalarDN mijGlobal(mij);
    GlobalScalarDN lijGlobal(lij);

    // Vec tiles for 16x16 operations
    using TileVec16x16 = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor, kRows, kCols>;
    // DN scalar tile for row reduction output - ColMajor with aligned rows
    using TileScalarDN = Tile<TileType::Vec, float, kAlignedRows, 1, BLayout::ColMajor, kRows, 1>;

    TileVec16x16 sijTile;
    TileVec16x16 pijTile;
    TileVec16x16 tmpTile;
    TileScalarDN maxTile;
    TileScalarDN sumTile;

    // Allocate tiles in UB
    TASSIGN(sijTile, 0x0);
    TASSIGN(pijTile, 0x400);
    TASSIGN(tmpTile, 0x800);
    TASSIGN(maxTile, 0xC00);
    TASSIGN(sumTile, 0xC40);

    // Load sij (16x16)
    TLOAD(sijTile, sijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    // Scale: sij = sij * scale
    TMULS(sijTile, sijTile, scale_value);

    // Row max: get max of each row
    TROWMAX(maxTile, sijTile, tmpTile);

    // Subtract row max: pij = sij - rowmax (broadcast)
    TROWEXPANDSUB(pijTile, sijTile, maxTile);

    // Exp: pij = exp(pij)
    TEXP(pijTile, pijTile);

    // Row sum: get sum of each row
    TROWSUM(sumTile, pijTile, tmpTile);

    // Store results
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(mijGlobal, maxTile);
    TSTORE(lijGlobal, sumTile);
    TSTORE(pijGlobal, pijTile);
}
