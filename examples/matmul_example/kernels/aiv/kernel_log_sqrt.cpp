/**
 * Element-wise Log then Sqrt Kernel
 *
 * Implements: out[i] = sqrt(log(src[i]))
 *
 * This kernel performs element-wise natural logarithm followed by square root.
 * Both input and output are half precision for matmul compatibility.
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
 * Log + Sqrt kernel implementation (half precision in/out)
 *
 * Unified signature: all arguments passed via int64_t array
 * @param args  Argument array:
 *              args[0] = src pointer (input tensor, half)
 *              args[1] = out pointer (output tensor, half)
 *              args[2] = size (number of elements)
 */
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    // Unpack arguments
    __gm__ half* src = reinterpret_cast<__gm__ half*>(args[0]);
    __gm__ half* out = reinterpret_cast<__gm__ half*>(args[1]);
    int size = static_cast<int>(args[2]);

    // Configuration
    constexpr int kTRows_ = 128;
    constexpr int kTCols_ = 128;
    constexpr int vRows = 128;
    constexpr int vCols = 128;

    // Half types for input and output
    using DynShapeDim5Half = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5Half = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalDataHalf = GlobalTensor<half, DynShapeDim5Half, DynStridDim5Half>;
    using TileDataHalf = Tile<TileType::Vec, half, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileDataHalf srcTile(vRows, vCols);
    TileDataHalf tmpTile(vRows, vCols);
    TileDataHalf dstTile(vRows, vCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(tmpTile, 0x10000);
    TASSIGN(dstTile, 0x20000);

    GlobalDataHalf srcGlobal(src);
    GlobalDataHalf dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TLOG(tmpTile, srcTile);
    TSQRT(dstTile, tmpTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}
