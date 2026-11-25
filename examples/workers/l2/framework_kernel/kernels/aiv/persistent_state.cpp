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

// The fixed task consumes DEVICE controls; host orchestration never reads them.
// Only accepted cache rows are written. All padding and rejected rows survive.
extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *weights_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *cache_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *proposals_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *control_tensor = reinterpret_cast<__gm__ Tensor *>(args[3]);
    auto *weights = reinterpret_cast<__gm__ float *>(weights_tensor->buffer.addr) + weights_tensor->start_offset;
    auto *cache = reinterpret_cast<__gm__ float *>(cache_tensor->buffer.addr) + cache_tensor->start_offset;
    auto *proposals = reinterpret_cast<__gm__ float *>(proposals_tensor->buffer.addr) + proposals_tensor->start_offset;
    auto *control = reinterpret_cast<__gm__ int32_t *>(control_tensor->buffer.addr) + control_tensor->start_offset;
    const int32_t cursor = control[0];
    const int32_t accepted = control[control_tensor->strides[0]];
    if (cursor < 0 || cursor > 8 || accepted < 0 || accepted > 4 || cursor + accepted > 8) {
        return;
    }
    using Shape = pto::Shape<1, 1, 1, 1, 32>;
    using Stride = pto::Stride<1, 1, 1, 32, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    using Tile = pto::Tile<pto::TileType::Vec, float, 1, 32, pto::BLayout::RowMajor, -1, -1>;
    Tile weight(1, 32), previous(1, 32), proposal(1, 32), weighted(1, 32), result(1, 32);
    pto::TASSIGN(weight, 0x0000);
    pto::TASSIGN(previous, 0x0080);
    pto::TASSIGN(proposal, 0x0100);
    pto::TASSIGN(weighted, 0x0180);
    pto::TASSIGN(result, 0x0200);
    Global weight_global(weights);
    pto::TLOAD(weight, weight_global);
    pipe_sync();
    for (int32_t row = 0; row < accepted; ++row) {
        Global cache_global(cache + (cursor + row) * cache_tensor->strides[0]);
        Global proposal_global(proposals + row * proposals_tensor->strides[0]);
        pto::TLOAD(previous, cache_global);
        pto::TLOAD(proposal, proposal_global);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TMUL(weighted, proposal, weight);
        // TADD reads weighted, so TMUL must complete before the dependent instruction issues.
        pipe_barrier(PIPE_ALL);
        pto::TADD(result, previous, weighted);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pto::TSTORE(cache_global, result);
        // Tiles are reused only after both reads and the prior write complete.
        pipe_sync();
    }
}
