/**
 * TREDUCE kernel for simpler's kernel_entry signature.
 *
 * Performs collective reduce (Sum) across multiple NPU ranks using PTO comm
 * instructions. Each rank's input data resides in an RDMA window;
 * the root rank gathers and sums all inputs into the output buffer.
 *
 * PTO communication instructions access remote data through GVA addresses
 * (windowsIn[]) via MTE2 DMA over HCCS; no bound stream is required.
 *
 * args layout (all uint64_t, cast as needed):
 *   args[0] = __gm__ float* input   (device addr in RDMA window)
 *   args[1] = __gm__ float* output  (device addr, regular allocation)
 *   args[2] = int nranks
 *   args[3] = int root
 *   args[4] = __gm__ CommDeviceContext* ctx  (device addr)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "pto/comm/comm_types.hpp"
#include "pto/comm/pto_comm_inst.hpp"
#include "common/comm_context.h"

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t TREDUCE_COUNT = 256;
static constexpr int kMaxSupportedRanks = 16;

template <typename T>
AICORE inline __gm__ T *CommRemotePtr(
    __gm__ CommDeviceContext *ctx, __gm__ T *localPtr, int pe)
{
    uint64_t localBase = ctx->windowsIn[ctx->rankId];
    uint64_t offset = (uint64_t)localPtr - localBase;
    return (__gm__ T *)(ctx->windowsIn[pe] + offset);
}


extern "C" __aicore__ __attribute__((always_inline))
void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* input  = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* output = reinterpret_cast<__gm__ float*>(args[1]);
    int nranks = static_cast<int>(args[2]);
    int root   = static_cast<int>(args[3]);
    __gm__ CommDeviceContext* commCtx =
        reinterpret_cast<__gm__ CommDeviceContext*>(args[4]);

    using ShapeDyn  = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC,
                                  pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC,
                                   pto::DYNAMIC, pto::DYNAMIC>;
    using Global    = pto::GlobalTensor<float, ShapeDyn, StrideDyn,
                                         pto::Layout::ND>;
    using TileData  = pto::Tile<pto::TileType::Vec, float, 1, TREDUCE_COUNT,
                                 pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(commCtx->rankId);

    ShapeDyn shape(1, 1, 1, 1, TREDUCE_COUNT);
    StrideDyn stride(TREDUCE_COUNT, TREDUCE_COUNT, TREDUCE_COUNT,
                     TREDUCE_COUNT, 1);

    TileData accTile(1, TREDUCE_COUNT);
    TileData recvTile(1, TREDUCE_COUNT);
    TASSIGN(accTile, 0x0);
    TASSIGN(recvTile, 0x10000);

    if (nranks <= 0 || nranks > kMaxSupportedRanks || root < 0 || root >= nranks) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    if (my_rank == root) {
        Global outputG(output, shape, stride);
        Global tensors[kMaxSupportedRanks];
        for (int i = 0; i < nranks; ++i) {
            __gm__ float* remoteInput = CommRemotePtr(commCtx, input, i);
            tensors[i] = Global(remoteInput, shape, stride);
        }
        pto::comm::ParallelGroup<Global> pg(tensors, nranks, root);
        pto::comm::TREDUCE(pg, outputG, accTile, recvTile,
                           pto::comm::ReduceOp::Sum);
    }

    pipe_barrier(PIPE_ALL);
}
