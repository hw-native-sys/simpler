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

#include "../device.h"

template <uint32_t Width>
inline __aicore__ void copy_and_add(__gm__ float *source, __gm__ float *output, uint32_t count) {
    using Shape = pto::Shape<1, 1, 1, 1, Width>;
    using Stride = pto::Stride<Width, Width, Width, Width, 1>;
    using Global = pto::GlobalTensor<float, Shape, Stride>;
    pto::Tile<pto::TileType::Vec, float, 1, Width> input, result;
    pto::TASSIGN(input, 0);
    pto::TASSIGN(result, 0x10000);
    for (uint32_t at = 0; at < count; at += Width) {
        Global src(source + at), dst(output + at);
        pto::TLOAD(input, src);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
        pto::TADDS(result, input, 7.0f);
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        pto::TSTORE(dst, result);
        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID7);
    }
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *metadata = sdma_reuse::data<uint64_t>(args[3]);
    const uint32_t lane = get_block_idx(args);
    const uint32_t posts = args[6];
    if (lane == 0) {
        auto *header = metadata + sdma_reuse::kMaxPosts * sdma_reuse::kWords;
        header[3] = get_block_num(args);
        sdma_reuse::flush(header);
    }
    for (uint32_t i = 0; i < posts; ++i) {
        const uint32_t index = lane * posts + i;
        auto *row = metadata + index * sdma_reuse::kWords;
        const uint64_t address = sdma_reuse::read(row + 1);
        const uint64_t expected = sdma_reuse::read(row + 2);
        if (sdma_reuse::read(row) != 1 || address == 0 || expected == 0 ||
            sdma_reuse::read(reinterpret_cast<__gm__ uint64_t *>(address)) < expected) {
            row[7] = 1;
            sdma_reuse::flush(row);
            return;
        }
        const uint32_t start = sdma_reuse::offset(index);
        auto *src = sdma_reuse::data<float>(args[1]) + start;
        auto *dst = sdma_reuse::data<float>(args[2]) + start;
        if (index == 0) copy_and_add<256>(src, dst, sdma_reuse::kLarge);
        else copy_and_add<8>(src, dst, 8);
        row[6] = 1;
        sdma_reuse::flush(row);
    }
}
