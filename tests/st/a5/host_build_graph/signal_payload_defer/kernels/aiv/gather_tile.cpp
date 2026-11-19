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
#include "../payload.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const uint32_t ranks = tensor->shapes[0], count = tensor->shapes[1];
    const uint32_t block = get_block_idx(args), blocks = get_block_num(args);
    if (block >= payload_case::kMaxBlocks) return;
    const uint32_t rank = static_cast<uint32_t>(args[9]);
    constexpr uint32_t kColumns = 128;
    using Global = pto::GlobalTensor<uint32_t, pto::Shape<1, 1, 1, 1, kColumns>, pto::Stride<1, 1, 1, kColumns, 1>>;
    using Tile = pto::Tile<pto::TileType::Vec, uint32_t, 1, kColumns, pto::BLayout::RowMajor, 1, kColumns>;
    Tile tile;
    pto::TASSIGN(tile, 0);
    for (uint32_t channel = 0; channel < 4; ++channel) {
        auto *source = payload_case::words(args, channel);
        auto *output = payload_case::words(args, channel + 4) + block * ranks * count;
        for (uint32_t peer = 0; peer < ranks; ++peer) {
            if (peer == rank) continue;
            for (uint32_t index = 0; index < count; index += kColumns) {
                Global input(source + peer * count + index);
                Global result(output + peer * count + index);
                pto::TLOAD(tile, input);
                set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
                pto::TSTORE(result, tile);
                set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
            }
        }
    }
    pipe_barrier(PIPE_ALL);
    auto *info = payload_case::words(args, 8) + block * payload_case::kInfoWords;
    info[0] = 1;
    info[1] = blocks;
    payload_case::publish(info);
}
