/**
 * Manual AllGather kernel - direct RDMA reads, no TGATHER.
 *
 * Each rank independently reads from all ranks' win_src via HcclRemotePtr
 * and writes to local dst. No collective TGATHER call, so no deadlock.
 * All ranks can run in parallel (single kernel, single barrier).
 *
 * Args: dst, src, ctx, nranks, rank_id (rank_id unused, for API compatibility)
 */

#include <cstdint>
#include <pto/pto-inst.hpp>
#include <pto/common/pto_tile.hpp>
#include "hccl_context.h"
#include "hccl_helpers.h"

using namespace pto;

#ifndef __gm__
#define __gm__
#endif

#ifndef __aicore__
#define __aicore__ [aicore]
#endif

static constexpr size_t GATHER_COUNT = 64;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t* args) {
    __gm__ float* dst = reinterpret_cast<__gm__ float*>(args[0]);
    __gm__ float* src = reinterpret_cast<__gm__ float*>(args[1]);
    __gm__ HcclDeviceContext* hcclCtx = reinterpret_cast<__gm__ HcclDeviceContext*>(args[2]);
    int nranks = static_cast<int>(args[3]);
    (void)args[4]; /* rank_id unused */

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;
    using TileData = pto::Tile<pto::TileType::Vec, float, 1, GATHER_COUNT, pto::BLayout::RowMajor, -1, -1>;

    ShapeDyn sliceShape(1, 1, 1, 1, GATHER_COUNT);
    StrideDyn sliceStride(GATHER_COUNT, GATHER_COUNT, GATHER_COUNT, GATHER_COUNT, 1);

    TileData ubTile(1, GATHER_COUNT);
    TASSIGN(ubTile, 0x0);

    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int r = 0; r < actual_nranks; ++r) {
        __gm__ float* remote_src = HcclRemotePtr(hcclCtx, src, r);
        __gm__ float* local_dst = dst + static_cast<ptrdiff_t>(r) * GATHER_COUNT;

        Global srcG(remote_src, sliceShape, sliceStride);
        Global dstG(local_dst, sliceShape, sliceStride);

        TLOAD(ubTile, srcG);
        set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        TSTORE(dstG, ubTile);
        set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
        wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    }

    pipe_barrier(PIPE_ALL);
}
