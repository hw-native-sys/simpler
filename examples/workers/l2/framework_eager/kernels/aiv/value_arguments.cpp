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

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *source = reinterpret_cast<__gm__ float *>(input->buffer.addr) + input->start_offset;
    auto *target = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset;
    const float scale = from_u64<float>(static_cast<uint64_t>(args[2]));
    const float bias = from_u64<float>(static_cast<uint64_t>(args[3]));
    using Shape = pto::Shape<1, 1, 1, 64, 64>;
    using Stride = pto::Stride<1, 1, 1, 64, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 64, 64, pto::BLayout::RowMajor, -1, -1>;
    Tile initial(64, 64), scaled(64, 64), result(64, 64);
    pto::TASSIGN(initial, 0x0000);
    pto::TASSIGN(scaled, 0x4000);
    pto::TASSIGN(result, 0x8000);
    Global src(source), dst(target);
    pto::TLOAD(initial, src);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    pto::TMULS(scaled, initial, scale);
    pto::TADDS(result, scaled, bias);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    pto::TSTORE(dst, result);
    pipe_sync();
}
