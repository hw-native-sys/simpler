/**
 * Online Softmax Update + Normalize Kernel (AIV) - 16x16 Version with PTO Tile API
 *
 * For a2a3 hardware:
 * - TSUB/TMUL/TADD only support RowMajor layout
 * - TROWEXPANDMUL/TROWEXPANDDIV require ColMajor (DN) layout for scalar tile
 * 
 * Solution: Use (1, 16) RowMajor for scalar arithmetic, reshape to (16, 1) ColMajor for broadcast
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

constexpr int kNumHeads = 16;
constexpr int kHeadDim = 16;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* mij    = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* lij    = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* oi_new = reinterpret_cast<__gm__ float*>(args[2]);
    __gm__ float* mi     = reinterpret_cast<__gm__ float*>(args[3]);
    __gm__ float* li     = reinterpret_cast<__gm__ float*>(args[4]);
    __gm__ float* oi     = reinterpret_cast<__gm__ float*>(args[5]);
    int is_first  = static_cast<int>(args[6]);
    int is_last   = static_cast<int>(args[7]);
    __gm__ float* dst    = reinterpret_cast<__gm__ float*>(args[8]);

    // GlobalTensor definitions
    using GlobalData16x16 = GlobalTensor<float, Shape<1, 1, 1, kNumHeads, kHeadDim>, Stride<1, 1, 1, kHeadDim, 1>>;
    // For scalar: ND layout (row-major, 1 x 16)
    using GlobalScalarND = GlobalTensor<float, Shape<1, 1, 1, 1, kNumHeads>, Stride<1, 1, 1, kNumHeads, 1>>;

    GlobalData16x16 oiNewGlobal(oi_new);
    GlobalData16x16 oiGlobal(oi);
    GlobalData16x16 dstGlobal(dst);
    GlobalScalarND mijGlobal(mij);
    GlobalScalarND lijGlobal(lij);
    GlobalScalarND miGlobal(mi);
    GlobalScalarND liGlobal(li);

    // Tile types
    using TileData16x16 = Tile<TileType::Vec, float, kNumHeads, kHeadDim, BLayout::RowMajor, kNumHeads, kHeadDim>;
    // ND scalar tile for arithmetic (1 x 16, RowMajor) - TSUB/TMUL/TADD require RowMajor
    using TileScalarND = Tile<TileType::Vec, float, 1, kNumHeads, BLayout::RowMajor, 1, kNumHeads>;
    // DN scalar tile for row broadcast (16 x 1, ColMajor) - TROWEXPANDMUL/TROWEXPANDDIV require this
    using TileScalarDN = Tile<TileType::Vec, float, kNumHeads, 1, BLayout::ColMajor, kNumHeads, 1>;

    TileData16x16 oiNewTile;
    TileData16x16 oiTile;
    TileScalarND mijTileND, lijTileND, miTileND, liTileND;
    TileScalarND miNewTileND, alphaTileND, betaTileND, tmpScalarND;
    TileScalarDN alphaTileDN, betaTileDN, liTileDN;

    // Allocate tiles in UB
    TASSIGN(oiNewTile, 0x0);
    TASSIGN(oiTile, 0x400);
    TASSIGN(mijTileND, 0x800);
    TASSIGN(lijTileND, 0x840);
    TASSIGN(miTileND, 0x880);
    TASSIGN(liTileND, 0x8C0);
    TASSIGN(miNewTileND, 0x900);
    TASSIGN(alphaTileND, 0x940);
    TASSIGN(betaTileND, 0x980);
    TASSIGN(tmpScalarND, 0x9C0);
    TASSIGN(alphaTileDN, 0xA00);
    TASSIGN(betaTileDN, 0xA40);
    TASSIGN(liTileDN, 0xA80);

    // Load current block data
    TLOAD(oiNewTile, oiNewGlobal);
    TLOAD(mijTileND, mijGlobal);
    TLOAD(lijTileND, lijGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    if (is_first) {
        // First block: just copy
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(miGlobal, mijTileND);
        TSTORE(liGlobal, lijTileND);
        TSTORE(oiGlobal, oiNewTile);
    } else {
        // Load accumulated values
        TLOAD(oiTile, oiGlobal);
        TLOAD(miTileND, miGlobal);
        TLOAD(liTileND, liGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);

        // mi_new = max(mi, mij) - RowMajor OK for TMAX
        TMAX(miNewTileND, miTileND, mijTileND);

        // alpha = exp(mi - mi_new) - RowMajor for TSUB/TEXP
        TSUB(alphaTileND, miTileND, miNewTileND);
        TEXP(alphaTileND, alphaTileND);

        // beta = exp(mij - mi_new)
        TSUB(betaTileND, mijTileND, miNewTileND);
        TEXP(betaTileND, betaTileND);

        // li_new = alpha * li + beta * lij (element-wise on scalars) - RowMajor for TMUL/TADD
        TMUL(liTileND, alphaTileND, liTileND);
        TMUL(tmpScalarND, betaTileND, lijTileND);
        TADD(liTileND, liTileND, tmpScalarND);

        // Reshape alpha/beta from ND (1,16) to DN (16,1) for row broadcast
        TRESHAPE(alphaTileDN, alphaTileND);
        TRESHAPE(betaTileDN, betaTileND);

        // oi_scaled = oi * alpha (row broadcast) - ColMajor required for TROWEXPANDMUL
        TROWEXPANDMUL(oiTile, oiTile, alphaTileDN);

        // oi_new_scaled = oi_new * beta (row broadcast)
        TROWEXPANDMUL(oiNewTile, oiNewTile, betaTileDN);

        // oi = oi_scaled + oi_new_scaled - RowMajor OK
        TADD(oiTile, oiTile, oiNewTile);

        // Store updated values
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID1);
        TSTORE(miGlobal, miNewTileND);
        TSTORE(liGlobal, liTileND);
        TSTORE(oiGlobal, oiTile);
    }

    // Normalize on last block
    if (is_last) {
        // Reload oi and li for normalization
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID2);
        TLOAD(oiTile, oiGlobal);
        TLOAD(liTileND, liGlobal);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2);

        // Reshape li from ND (1,16) to DN (16,1) for row broadcast
        TRESHAPE(liTileDN, liTileND);

        // Normalize: oi = oi / li (row broadcast division)
        TROWEXPANDDIV(oiTile, oiTile, liTileDN);

        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
        TSTORE(dstGlobal, oiTile);
    }
}
