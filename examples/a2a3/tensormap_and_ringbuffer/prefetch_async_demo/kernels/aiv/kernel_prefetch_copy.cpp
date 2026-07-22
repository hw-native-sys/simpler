/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * Single-card TPREFETCH_ASYNC kernel: warm L2 with `in`, wait for the event,
 * then copy `in` to `out` through a tile.
 *
 * The SDMA workspace is obtained from the runtime via get_dma_workspace(args,
 * DMA_WORKSPACE_SDMA) -- injected into every kernel's GlobalContext by the
 * scheduler, the same way get_block_idx() is provided -- so no workspace is
 * threaded as a user arg. This callable declares SDMA as mandatory; a null
 * workspace returns without writing `out`, making broken injection visible to
 * the host golden instead of silently exercising only the copy.
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

// Must follow the __gm__ / __aicore__ definitions: intrinsic.h falls back to an
// empty __aicore__ if it is not already defined, which would strip the [aicore]
// attribute from kernel_entry below.
#include "intrinsic.h"
#include "pipe_sync.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *in_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);

    __gm__ float *in = reinterpret_cast<__gm__ float *>(in_tensor->buffer.addr) + in_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int kRows = 1;
    constexpr int kCols = 128;
    constexpr int kElems = kRows * kCols;

    // Flat contiguous logical-1D view: TPrefetchAsyncOp requires every dim but
    // the last to be 1.
    using FlatShape = Shape<1, 1, 1, 1, kElems>;
    using FlatStride = Stride<kElems, kElems, kElems, kElems, 1>;
    using GlobalData = GlobalTensor<float, FlatShape, FlatStride>;
    using TileData = Tile<TileType::Vec, float, kRows, kCols, BLayout::RowMajor>;

    GlobalData in_global(in);
    GlobalData out_global(out);

    // Runtime-injected SDMA workspace -- no user arg. The host refuses this
    // marked callable unless SDMA is supported and provisioning succeeded.
    __gm__ uint8_t *sdma_workspace = get_dma_workspace(args, DMA_WORKSPACE_SDMA);
    if (sdma_workspace == nullptr) {
        pipe_barrier(PIPE_ALL);
        return;
    }
    PrefetchAsyncContext ctx(sdma_workspace);
    comm::AsyncEvent evt = TPREFETCH_ASYNC(in_global, ctx);
    if (!evt.Wait(ctx.session)) {
        pipe_barrier(PIPE_ALL);
        return;
    }

    TileData tile;
    TASSIGN(tile, 0x0);

    TLOAD(tile, in_global);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(out_global, tile);

    pipe_sync();
}
