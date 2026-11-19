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

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include <pto/pto-inst.hpp>

#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *src_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *result_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *src = reinterpret_cast<__gm__ float *>(src_tensor->buffer.addr) + src_tensor->start_offset;
    auto *result = reinterpret_cast<__gm__ float *>(result_tensor->buffer.addr) + result_tensor->start_offset;
    const uint32_t elems = static_cast<uint32_t>(args[2]);

    constexpr int kRows = 64;
    constexpr int kCols = 128;
    constexpr int kChunkElems = kRows * kCols;
    using Shape = pto::Shape<1, 1, 1, kRows, kCols>;
    using Stride = pto::Stride<1, 1, 1, kCols, 1>;
    using GlobalData = pto::GlobalTensor<float, Shape, Stride>;
    using Tile = pto::Tile<pto::TileType::Vec, float, kRows, kCols, pto::BLayout::RowMajor, -1, -1>;
    Tile src_tile(kRows, kCols);
    Tile result_tile(kRows, kCols);
    TASSIGN(src_tile, 0x0);
    TASSIGN(result_tile, 0x10000);
    for (uint32_t offset = 0; offset < elems; offset += kChunkElems) {
        GlobalData src_global(src + offset);
        GlobalData result_global(result + offset);
        TLOAD(src_tile, src_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        TADDS(result_tile, src_tile, 1.0f);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        TSTORE(result_global, result_tile);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}
