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

#include "pipe_sync.h"
#include "tensor.h"

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

// Both 64x64 views are snapshotted before any write, including when their
// storage overlaps. Only destination elements are written; row padding is not.
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *source_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *target_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *source = reinterpret_cast<__gm__ float *>(source_tensor->buffer.addr) + source_tensor->start_offset;
    auto *target = reinterpret_cast<__gm__ float *>(target_tensor->buffer.addr) + target_tensor->start_offset;
    using ShapeType = pto::Shape<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using StrideType = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC>;
    using Global = pto::GlobalTensor<float, ShapeType, StrideType, pto::Layout::ND>;
    using Data = pto::Tile<pto::TileType::Vec, float, 64, 64, pto::BLayout::RowMajor, -1, -1>;
    ShapeType shape(1, 1, 1, source_tensor->shapes[0], source_tensor->shapes[1]);
    StrideType source_stride(1, 1, 1, source_tensor->strides[0], source_tensor->strides[1]);
    StrideType target_stride(1, 1, 1, target_tensor->strides[0], target_tensor->strides[1]);
    Global source_global(source, shape, source_stride);
    Global target_global(target, shape, target_stride);
    Data source_tile(source_tensor->shapes[0], source_tensor->shapes[1]);
    Data target_tile(target_tensor->shapes[0], target_tensor->shapes[1]);
    Data result(target_tensor->shapes[0], target_tensor->shapes[1]);
    pto::TASSIGN(source_tile, 0x0000);
    pto::TASSIGN(target_tile, 0x4000);
    pto::TASSIGN(result, 0x8000);
    pto::TLOAD(source_tile, source_global);
    pto::TLOAD(target_tile, target_global);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TADD(result, source_tile, target_tile);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(target_global, result);
    pipe_sync();
}
