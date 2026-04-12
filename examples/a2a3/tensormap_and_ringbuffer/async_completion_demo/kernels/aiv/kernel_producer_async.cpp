/**
 * Async Completion Demo - Hardware 2P SDMA TGET Producer Kernel (func_id=2)
 *
 * Implements:
 *   1. Read peer rank's input buffer via TGET_ASYNC into local out
 *   2. Register the async event in the CQ
 *   3. Return immediately so the runtime completes the task asynchronously
 *
 * This kernel is only compiled for real hardware (a2a3), not for simulation.
 *
 * Kernel args layout (packed by scheduler):
 *   args[0] = &Tensor(in)            — input tensor struct pointer
 *   args[1] = &Tensor(out)           — output tensor struct pointer
 *   args[2] = CommDeviceContext*     — distributed communication context
 *   args[3] = sdma_context_addr      — SDMA async context
 *   args[4] = cq_addr                — completion queue (appended by submit_deferred)
 */

#include <cstdint>
#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>
#include "pto/comm/pto_comm_inst.hpp"
#include "pto/npu/comm/async/sdma/sdma_types.hpp"
#include "pto/common/pto_tile.hpp"

#include "common/comm_context.h"
#include "tensor.h"

using namespace pto;

#include "pto_async_kernel_api.h"

template <typename T>
AICORE inline __gm__ T* CommRemotePtr(__gm__ CommDeviceContext* ctx, __gm__ T* local_ptr,
                                      int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)local_ptr - local_base;
    return (__gm__ T*)(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ Tensor* in_tensor  = reinterpret_cast<__gm__ Tensor*>(args[0]);
    __gm__ Tensor* out_tensor = reinterpret_cast<__gm__ Tensor*>(args[1]);
    __gm__ CommDeviceContext* comm_ctx =
        reinterpret_cast<__gm__ CommDeviceContext*>(args[2]);
    uint64_t sdma_context     = static_cast<uint64_t>(args[3]);
    uint64_t cq_addr          = static_cast<uint64_t>(args[4]);

    __gm__ float* in_data  = reinterpret_cast<__gm__ float*>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float* out_data = reinterpret_cast<__gm__ float*>(out_tensor->buffer.addr) + out_tensor->start_offset;
    volatile __gm__ PTO2CompletionQueue* cq = pto2_cq_get(cq_addr);
    pto2_cq_reset(cq);

    int my_rank = static_cast<int>(comm_ctx->rankId);
    int nranks = static_cast<int>(comm_ctx->rankNum);
    if (nranks != 2) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    int peer_rank = 1 - my_rank;

    constexpr int kTotalElems = 128 * 128;

    using FlatShape = Shape<1, 1, 1, 1, kTotalElems>;
    using FlatStride = Stride<kTotalElems, kTotalElems, kTotalElems, kTotalElems, 1>;
    using FlatGlobalData = GlobalTensor<float, FlatShape, FlatStride>;
    FlatGlobalData outGlobalFlat(out_data);
    __gm__ float* remote_in_data = CommRemotePtr(comm_ctx, in_data, peer_rank);
    FlatGlobalData remoteInGlobalFlat(remote_in_data);

    using ScratchTile = pto::Tile<pto::TileType::Vec, uint8_t, 1, pto::comm::sdma::UB_ALIGN_SIZE>;
    ScratchTile scratchTile;
    TASSIGN(scratchTile, 0x20000);

    __gm__ uint8_t* context = reinterpret_cast<__gm__ uint8_t*>(static_cast<uintptr_t>(sdma_context));

    auto desc = pto2_remote_copy_tget_descriptor(outGlobalFlat, remoteInGlobalFlat, scratchTile, context);
    uint64_t tag = pto2_send_request_entry(PTO2_ENGINE_SDMA, PTO2_SQ_ID_AUTO, desc);
    pto2_save_expected_completion(PTO2_ENGINE_SDMA, cq, tag);

    pto2_cq_flush(cq);
}
