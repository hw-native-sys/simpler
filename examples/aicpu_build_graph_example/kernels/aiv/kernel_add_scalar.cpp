/**
 * Scalar Addition Kernel
 *
 * Implements: out[i] = src[i] + scalar
 */

#include <cstdint>
// clang-format off
#include <pto/pto-inst.hpp>          // defines CPU stubs (incl. __gm__) under __CPU_SIM
#include <pto/common/constants.hpp>  // uses __gm__ in some headers
// clang-format on

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// `a2a3sim` loads per-kernel binaries via dlopen+dlsym("kernel_entry").
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* src = reinterpret_cast<__gm__ float*>(args[0]);

    union {
        uint64_t u64;
        float f32;
    } converter;
    converter.u64 = args[1];
    float scalar = converter.f32;

    __gm__ float* out = reinterpret_cast<__gm__ float*>(args[2]);
    int size = static_cast<int>(args[3]);
    (void)size;

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
