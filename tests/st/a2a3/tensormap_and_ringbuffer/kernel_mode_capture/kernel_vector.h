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
#pragma once

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "tensor.h"
#include "pipe_sync.h"

template <bool Negate>
__aicore__ __attribute__((always_inline)) void vector_scalar(__gm__ int64_t *args) {
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    const int count = input->shapes[0];
    using Shape = pto::Shape<1, 1, 1, 1, -1>;
    using Stride = pto::Stride<1, 1, 1, 16384, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 1, 16384, pto::BLayout::RowMajor, -1, -1>;
    Tile source(1, count), result(1, count);
    pto::TASSIGN(source, 0x0);
    pto::TASSIGN(result, 0x10000);
    Global src(reinterpret_cast<__gm__ float *>(input->buffer.addr) + input->start_offset, Shape(1, 1, 1, 1, count));
    Global dst(reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset, Shape(1, 1, 1, 1, count));
    pto::TLOAD(source, src);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    if constexpr (Negate) pto::TMULS(source, source, -1.0f);
    pto::TADDS(result, source, from_u64<float>(static_cast<uint64_t>(args[2])));
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(dst, result);
    pipe_sync();
}
