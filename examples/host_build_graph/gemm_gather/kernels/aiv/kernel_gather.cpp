/**
 * 1D Gather Kernel (AIV, a2a3)
 *
 * out = src0[src1] with comm-isa shape: src0 (32, 1024) float, src1 (16, 64) int32, out (16, 64) float.
 * Reference: pto-comm-isa tests/npu/a2a3/src/st/testcase/tgather (runTGather1D).
 * Uses TGATHER on Vector pipe (PIPE_V).
 * All logic in kernel_entry (same pattern as vector_example kernel_add.cpp) so set_flag/wait_flag
 * are used only in __aicore__ context.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tgather_common.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* out = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* src0 = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ int32_t* src1 = reinterpret_cast<__gm__ int32_t*>(args[2]);

    constexpr int kGRows0_ = GATHER_SRC0_ROWS;
    constexpr int kGCols0_ = GATHER_SRC0_COLS;
    constexpr int kGRows1_ = GATHER_SRC1_ROWS;
    constexpr int kGCols1_ = GATHER_SRC1_COLS;
    constexpr int src0_row = kGRows0_;
    constexpr int src0_col = kGCols0_;
    constexpr int src1_row = kGRows1_;
    constexpr int src1_col = kGCols1_;
    constexpr int dst_row = kGRows1_;
    constexpr int dst_col = kGCols1_;

    using DynShapeDim5_src0 = pto::Shape<1, 1, 1, kGRows0_, kGCols0_>;
    using DynStridDim5_src0 = pto::Stride<1, 1, 1, kGCols0_, 1>;
    using GlobalData_src0 = GlobalTensor<float, DynShapeDim5_src0, DynStridDim5_src0>;
    using DynShapeDim5_src1 = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_src1 = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_src1 = GlobalTensor<int32_t, DynShapeDim5_src1, DynStridDim5_src1>;
    using DynShapeDim5_dst = pto::Shape<1, 1, 1, kGRows1_, kGCols1_>;
    using DynStridDim5_dst = pto::Stride<1, 1, 1, kGCols1_, 1>;
    using GlobalData_dst = GlobalTensor<float, DynShapeDim5_dst, DynStridDim5_dst>;

    using TileData_src0 = Tile<TileType::Vec, float, kGRows0_, kGCols0_, BLayout::RowMajor, -1, -1>;
    using TileData_src1 = Tile<TileType::Vec, int32_t, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;
    using TileData_dst = Tile<TileType::Vec, float, kGRows1_, kGCols1_, BLayout::RowMajor, -1, -1>;

    TileData_src0 src0Tile(src0_row, src0_col);
    TileData_src1 src1Tile(src1_row, src1_col);
    TileData_dst dstTile(dst_row, dst_col);

    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x20000);
    TASSIGN(dstTile, 0x28000);

    GlobalData_src0 src0Global(src0);
    GlobalData_src1 src1Global(src1);
    GlobalData_dst dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TGATHER(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}
