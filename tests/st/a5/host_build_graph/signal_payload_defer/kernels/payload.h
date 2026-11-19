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
#include "async_kernel_api.h"
#include "tensor.h"

namespace payload_case {
constexpr uint32_t kMaxBlocks = 128;
constexpr uint32_t kInfoWords = 32;
inline __aicore__ __gm__ uint32_t *words(__gm__ int64_t *args, uint32_t index) {
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
    return reinterpret_cast<__gm__ uint32_t *>(tensor->buffer.addr) + tensor->start_offset;
}
inline __aicore__ void publish(__gm__ void *address) {
    dcci(address, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    pipe_barrier(PIPE_ALL);
}
inline __aicore__ void warm(__gm__ int64_t *args) {
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const uint32_t ranks = tensor->shapes[0], count = tensor->shapes[1];
    const uint32_t block = get_block_idx(args), blocks = get_block_num(args);
    if (block >= kMaxBlocks) return;
    const uint32_t rank = static_cast<uint32_t>(args[5]);
    const bool dirty = args[6] != 0;
    const uint32_t field = block % 4;
    const uint32_t offset = ((rank + 1) % ranks) * count + (block / 4) * 16;
    if (dirty) {
        // Distinct cache lines have distinct writers. Release old dirty data
        // before any peer is allowed to replace this receive window.
        words(args, field)[offset] = UINT32_C(0xdead0000) + block;
        dcci(words(args, field) + offset, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
        dsb((mem_dsb_t)0);
    }
    uint64_t checksum = 0;
    for (uint32_t channel = 0; channel < 4; ++channel) {
        auto *source = reinterpret_cast<volatile __gm__ uint32_t *>(words(args, channel));
        for (uint32_t index = 0; index < ranks * count; ++index)
            checksum += source[index];
    }
    checksum += reinterpret_cast<volatile __gm__ uint32_t *>(words(args, field))[offset];
    auto *info = words(args, 4) + block * kInfoWords;
    info[2] = 1;
    info[3] = blocks;
    info[4] = static_cast<uint32_t>(checksum);
    publish(info);
}
inline __aicore__ void gather(__gm__ int64_t *args) {
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const uint32_t ranks = tensor->shapes[0], count = tensor->shapes[1];
    const uint32_t block = get_block_idx(args), blocks = get_block_num(args);
    if (block >= kMaxBlocks) return;
    const uint32_t rank = static_cast<uint32_t>(args[9]);
    for (uint32_t channel = 0; channel < 4; ++channel) {
        auto *source = reinterpret_cast<volatile __gm__ uint32_t *>(words(args, channel));
        auto *output = words(args, channel + 4) + block * ranks * count;
        for (uint32_t peer = 0; peer < ranks; ++peer) {
            if (peer == rank) continue;
            for (uint32_t index = 0; index < count; ++index)
                output[peer * count + index] = source[peer * count + index];
        }
    }
    auto *info = words(args, 8) + block * kInfoWords;
    info[0] = 1;
    info[1] = blocks;
    publish(info);
}
}  // namespace payload_case
