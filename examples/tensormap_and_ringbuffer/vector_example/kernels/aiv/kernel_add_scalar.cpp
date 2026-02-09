/**
 * Scalar Addition Kernel
 *
 * Implements: out[i] = src[i] + scalar
 *
 * This kernel adds a scalar value to each element of a tensor. It's compiled
 * separately as a standalone kernel and linked with the dispatcher using
 * function pointers, demonstrating the separation pattern used in production
 * systems where kernel binaries are loaded dynamically.
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
 * Scalar addition kernel implementation
 *
 * Unified signature: all arguments passed via int64_t array
 * @param args  Argument array:
 *              args[0] = src pointer (input tensor)
 *              args[1] = scalar value (as uint64_t, needs conversion to float)
 *              args[2] = out pointer (output tensor)
 *              args[3] = size (number of elements)
 */
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    // Unpack arguments
    __gm__ float* src = reinterpret_cast<__gm__ float*>(args[0]);

    // Convert scalar from uint64_t to float
    union {
        uint64_t u64;
        float f32;
    } converter;
    converter.u64 = args[1];
    float scalar = converter.f32;

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

    TileData srcTile(vRows, vCols);
    TileData dstTile(vRows, vCols);
    TASSIGN(srcTile, 0x0);
    TASSIGN(dstTile, 0x10000);

    GlobalData srcGlobal(src);
    GlobalData dstGlobal(out);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADDS(dstTile, srcTile, scalar);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dstGlobal, dstTile);
}
