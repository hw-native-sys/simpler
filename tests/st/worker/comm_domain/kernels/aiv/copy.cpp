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

AICORE inline __gm__ float *target_pointer(__gm__ Tensor *tensor) {
    uint64_t address = tensor->buffer.addr + tensor->start_offset * sizeof(float);
    return reinterpret_cast<__gm__ float *>(address);
}

extern "C" AICORE __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *source = target_pointer(reinterpret_cast<__gm__ Tensor *>(args[0]));
    auto *destination = target_pointer(reinterpret_cast<__gm__ Tensor *>(args[1]));
    using Shape = pto::Shape<1, 1, 1, 1, 256>;
    using Stride = pto::Stride<256, 256, 256, 256, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 1, 256, pto::BLayout::RowMajor, -1, -1>;
    Tile tile(1, 256);
    TASSIGN(tile, 0);
    Global input(source);
    Global output(destination);
    TLOAD(tile, input);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    TSTORE(output, tile);
    pipe_barrier(PIPE_ALL);
}
