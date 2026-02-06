/**
 * Element-wise Tensor Multiplication Kernel
 *
 * Implements: out[i] = src0[i] * src1[i]
 *
 * This kernel performs element-wise multiplication of two tensors. It's
 * compiled separately as a standalone kernel and linked with the dispatcher
 * using function pointers, demonstrating the separation pattern used in
 * production systems where kernel binaries are loaded dynamically.
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

/**
 * Element-wise multiplication kernel implementation
 *
 * Unified signature: all arguments passed via int64_t array
 * @param args  Argument array:
 *              args[0] = src0 pointer (first input tensor)
 *              args[1] = src1 pointer (second input tensor)
 *              args[2] = out pointer (output tensor)
 *              args[3] = size (number of elements)
 */
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    // Unpack arguments (order matches runtimemaker.cpp)
    __gm__ float* src0 = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* src1 = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ float* out = reinterpret_cast<__gm__ float*>(args[2]);
    int size = static_cast<int>(args[3]);

    // Configuration: float, 128, 128, 128, 128
    constexpr int kTRows_ = 128;
    constexpr int kTCols_ = 128;
    constexpr int vRows = 128;
    constexpr int vCols = 128;

    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData src0Tile(vRows, vCols);
    TileData src1Tile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(src0Tile, 0x0);
    TASSIGN(src1Tile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalData src0Global(src0);
    GlobalData src1Global(src1);
    GlobalData dstGlobal(out);

    TLOAD(src0Tile, src0Global);
    TLOAD(src1Tile, src1Global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TMUL(dstTile, src0Tile, src1Tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}
