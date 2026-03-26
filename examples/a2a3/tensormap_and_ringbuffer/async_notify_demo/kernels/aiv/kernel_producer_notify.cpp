/**
 * Async Notify Demo - Hardware 2P Notify Producer Kernel (func_id=0)
 *
 * Implements:
 *   1. Local compute: out[i] = in[i] * 2.0
 *   2. Notify peer rank via TNOTIFY(AtomicAdd) on the peer's window counter
 *   3. Return normally (run-to-completion, no deferred completion)
 *
 * Rank 1 inserts a deliberate delay before notifying. This makes missing
 * launch-gating on the consumer side visible in the example output.
 *
 * Kernel args layout (packed by scheduler):
 *   args[0] = &Tensor(in)            — input tensor struct pointer
 *   args[1] = &Tensor(out)           — output tensor struct pointer
 *   args[2] = notify_counter_addr    — local notify counter (window memory)
 *   args[3] = CommDeviceContext*     — distributed communication context
 */

#include <cstdint>

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include "pto/common/pto_tile.hpp"

#include "common/comm_context.h"
#include "tensor.h"

using namespace pto;

#include "pto_notify_kernel_api.h"

template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* local_ptr,
                                      int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)local_ptr - local_base;
    return (__gm__ T*)(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* in_tensor = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ int32_t* local_counter = reinterpret_cast<__gm__ int32_t*>(args[2]);
    __gm__ CommDeviceContext* comm_ctx = reinterpret_cast<__gm__ CommDeviceContext*>(args[3]);

    __gm__ float* in_data =
        reinterpret_cast<__gm__ float*>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float* out_data =
        reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;

    int my_rank = static_cast<int>(comm_ctx->rankId);
    int nranks = static_cast<int>(comm_ctx->rankNum);
    if (nranks != 2) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    int peer_rank = 1 - my_rank;

    constexpr int kTRows_ = 128;
    constexpr int kTCols_ = 128;
    constexpr int vRows = 128;
    constexpr int vCols = 128;

    using DynShapeDim5 = Shape<1, 1, 1, vRows, vCols>;
    using DynStridDim5 = Stride<1, 1, 1, kTCols_, 1>;
    using GlobalData = GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = Tile<TileType::Vec, float, kTRows_, kTCols_, BLayout::RowMajor, -1, -1>;

    TileData inTile(vRows, vCols);
    TileData outTile(vRows, vCols);
    TASSIGN(inTile, 0x0);
    TASSIGN(outTile, 0x10000);

    GlobalData inGlobal(in_data);
    GlobalData outGlobal(out_data);

    TLOAD(inTile, inGlobal);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);

    TADD(outTile, inTile, inTile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);

    TSTORE(outGlobal, outTile);
    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);

    if (my_rank == 1) {
        for (volatile int i = 0; i < 2000000; ++i) {
        }
    }

    __gm__ int32_t* remote_counter = CommRemotePtr(comm_ctx, local_counter, peer_rank);
    pto2_send_notification(remote_counter, 1, PTO2NotifyOp::AtomicAdd);
}
