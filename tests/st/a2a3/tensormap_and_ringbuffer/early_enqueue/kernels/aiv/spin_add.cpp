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
 * One 128x128 float tile of addition, preceded by a bounded spin.
 *
 * Args (Tensor*, then scalars):
 *   args[0] = src     (INPUT)
 *   args[1] = addend  (INPUT)
 *   args[2] = dst     (INOUT)
 *   args[3] = spin iterations, 0 for none
 *   args[4] = accumulate flag
 *
 * accumulate == 0 writes `src + addend` into dst; a non-zero flag writes
 * `dst + addend` and leaves src unread. One kernel with both modes keeps every
 * task in a chain on the same three tensors, so the chain is ordered through
 * dst alone and needs no intermediate the ring heap would have to hold.
 *
 * The spin count is a caller-supplied loop bound, not a timer: it keeps the
 * task on-core long enough for a host observation window while staying below
 * the scheduler's no-progress timeout, and it terminates with no device-side
 * handshake for teardown to release.
 */

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"

#include "pipe_sync.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *addend_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *dst_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    const int32_t spin_iters = static_cast<int32_t>(args[3]);
    volatile int32_t spin_accumulator = 0;
    for (int32_t i = 0; i < spin_iters; ++i) {
        ++spin_accumulator;
    }
    (void)spin_accumulator;

    const bool accumulate = args[4] != 0;
    __gm__ Tensor *left_tensor = accumulate ? dst_tensor : src_tensor;

    __gm__ float *left = reinterpret_cast<__gm__ float *>(left_tensor->buffer.addr) + left_tensor->start_offset;
    __gm__ float *addend = reinterpret_cast<__gm__ float *>(addend_tensor->buffer.addr) + addend_tensor->start_offset;
    __gm__ float *dst = reinterpret_cast<__gm__ float *>(dst_tensor->buffer.addr) + dst_tensor->start_offset;

    constexpr int kRows = 128;
    constexpr int kCols = 128;
    using TileShape = pto::Shape<1, 1, 1, kRows, kCols>;
    using TileStride = pto::Stride<1, 1, 1, kCols, 1>;
    using GlobalData = pto::GlobalTensor<float, TileShape, TileStride>;
    using TileData = pto::Tile<pto::TileType::Vec, float, kRows, kCols, pto::BLayout::RowMajor, -1, -1>;

    TileData left_tile(kRows, kCols);
    TileData addend_tile(kRows, kCols);
    TileData dst_tile(kRows, kCols);
    TASSIGN(left_tile, 0x0);
    TASSIGN(addend_tile, 0x10000);
    TASSIGN(dst_tile, 0x20000);

    GlobalData left_global(left);
    GlobalData addend_global(addend);
    GlobalData dst_global(dst);

    TLOAD(left_tile, left_global);
    TLOAD(addend_tile, addend_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    TADD(dst_tile, left_tile, addend_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    TSTORE(dst_global, dst_tile);
    pipe_sync();
}
