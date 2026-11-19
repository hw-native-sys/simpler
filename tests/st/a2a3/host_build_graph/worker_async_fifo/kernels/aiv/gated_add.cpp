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

#include <cstdint>
#include <pto/pto-inst.hpp>

#include "tensor.h"
#include "pipe_sync.h"
#include "../device_gate.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *src0_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *src1_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ Tensor *out_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);

    __gm__ float *src0 = reinterpret_cast<__gm__ float *>(src0_tensor->buffer.addr) + src0_tensor->start_offset;
    __gm__ float *src1 = reinterpret_cast<__gm__ float *>(src1_tensor->buffer.addr) + src1_tensor->start_offset;
    __gm__ float *out = reinterpret_cast<__gm__ float *>(out_tensor->buffer.addr) + out_tensor->start_offset;

    constexpr int kTRows = 64;
    constexpr int kTCols = 128;
    constexpr int kHalfElements = kTRows * kTCols;
    using DynShapeDim5 = pto::Shape<1, 1, 1, kTRows, kTCols>;
    using DynStridDim5 = pto::Stride<1, 1, 1, kTCols, 1>;
    using GlobalData = pto::GlobalTensor<float, DynShapeDim5, DynStridDim5>;
    using TileData = pto::Tile<pto::TileType::Vec, float, kTRows, kTCols, pto::BLayout::RowMajor, -1, -1>;

    TileData src0_tile(kTRows, kTCols);
    TileData src1_tile(kTRows, kTCols);
    TileData dst_tile(kTRows, kTCols);
    TASSIGN(src0_tile, 0x0);
    TASSIGN(src1_tile, 0x10000);
    TASSIGN(dst_tile, 0x20000);

    // Publish the gate only after the first half's FP32 work has retired.
    for (int half = 0; half < 2; ++half) {
        if (half == 1 && !wait_device_gate(static_cast<uint64_t>(args[3]))) return;
        const int offset = half * kHalfElements;
        GlobalData src0_global(src0 + offset);
        GlobalData src1_global(src1 + offset);
        GlobalData dst_global(out + offset);

        TLOAD(src0_tile, src0_global);
        TLOAD(src1_tile, src1_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADD(dst_tile, src0_tile, src1_tile);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(dst_global, dst_tile);
        pipe_sync();
    }
}
