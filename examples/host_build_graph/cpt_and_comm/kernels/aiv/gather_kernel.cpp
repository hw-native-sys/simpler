/**
 * TGATHER collective kernel - root gathers from all ranks.
 * Requires pto-comm-isa (PTO_ISA_ROOT or PTO_COMM_ISA_ROOT).
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
    int root = static_cast<int>(args[4]);

    using ShapeDyn = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideDyn = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeDyn, StrideDyn, pto::Layout::ND>;

    using TileData = pto::Tile<pto::TileType::Vec, float, 1, GATHER_COUNT, pto::BLayout::RowMajor, -1, -1>;

    int my_rank = static_cast<int>(hcclCtx->rankId);

    ShapeDyn srcShape(1, 1, 1, 1, GATHER_COUNT);
    StrideDyn srcStride(GATHER_COUNT, GATHER_COUNT, GATHER_COUNT, GATHER_COUNT, 1);

    ShapeDyn dstShape(1, 1, 1, nranks, GATHER_COUNT);
    StrideDyn dstStride(nranks * GATHER_COUNT, nranks * GATHER_COUNT, nranks * GATHER_COUNT, GATHER_COUNT, 1);
    Global dstG(dst, dstShape, dstStride);

    Global tensors[16];
    int actual_nranks = (nranks > 16) ? 16 : nranks;
    for (int i = 0; i < actual_nranks; ++i) {
        __gm__ float* remoteSrc = HcclRemotePtr(hcclCtx, src, i);
        tensors[i] = Global(remoteSrc, srcShape, srcStride);
    }

    pto::comm::ParallelGroup<Global> pg(tensors, actual_nranks, root);

    TileData ubTile(1, GATHER_COUNT);
    TASSIGN(ubTile, 0x0);

    if (my_rank == root) {
        pto::comm::TGATHER(pg, dstG, ubTile);
    }
}
