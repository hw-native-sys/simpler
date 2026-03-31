/**
 * Async Notify Demo - Consumer Kernel (func_id=1)
 *
 * Implements: result[i] = src[i] + notify_counter[0]
 *
 * Depends on NotifyWait completing (via dummy tensor), guaranteeing
 * the local notification counter >= 1 before this kernel runs.
 *
 * Kernel args layout (packed by scheduler):
 *   args[0] = &Tensor(dummy_notify)   — input (dependency token from NotifyWait)
 *   args[1] = &Tensor(src)            — input tensor struct pointer (producer's output)
 *   args[2] = &Tensor(result)         — output tensor struct pointer
 *   args[3] = notify_counter_addr     — local notify counter (window memory)
 */

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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    // args[0] = dummy_notify tensor (dependency token, unused)
    __gm__ Tensor* src_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ Tensor* result_tensor = reinterpret_cast<__gm__ Tensor*>(args[2]);
    __gm__ int32_t* notify_counter = reinterpret_cast<__gm__ int32_t*>(args[3]);

    __gm__ float* src =
        reinterpret_cast<__gm__ float*>(src_tensor->buffer.addr) + src_tensor->start_offset;
    __gm__ float* result =
        reinterpret_cast<__gm__ float*>(result_tensor->buffer.addr) + result_tensor->start_offset;

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
    GlobalData dstGlobal(result);

    TLOAD(srcTile, srcGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    float notify_value = static_cast<float>(*notify_counter);
    TADDS(dstTile, srcTile, notify_value);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(dstGlobal, dstTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
}
