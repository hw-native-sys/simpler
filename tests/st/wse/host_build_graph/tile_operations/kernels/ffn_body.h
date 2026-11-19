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
#include "../../ffn_tiles.h"
#include "tensor.h"

inline FfnMemory ffn_view(int64_t *args, int index, size_t offset) {
    const auto &t = *reinterpret_cast<Tensor *>(args[index]);
    return wse::tile_memory<32, 32>(reinterpret_cast<float *>(t.buffer.addr) + t.start_offset, t.numel(), offset);
}

template <int Stage>
void ffn_compute(int64_t *args) {
    const auto block = get_block_idx(args);
    const size_t row = get_thread_idx(args), column = block;
    const size_t output_offset = column * 128 * 32 + row * 32 * 32;
    if constexpr (Stage == 1) {
        ffn_relu(ffn_view(args, 0, output_offset), ffn_view(args, 1, output_offset));
    } else {
        constexpr size_t chunks = Stage == 0 ? 1 : 40;
        ffn_project<Stage == 0 ? 0 : 1>(
            args, 0, chunks,
            [&](size_t k) {
                return ffn_view(args, 0, row * 1024 + k * 128 * 32);
            },
            [&](size_t k) {
                return ffn_view(args, 1, (column * chunks + k) * 1024);
            },
            ffn_view(args, 2, output_offset)
        );
        if constexpr (Stage == 0) {
            auto *t = reinterpret_cast<Tensor *>(args[3]);
            auto *ids = reinterpret_cast<float *>(t->buffer.addr) + t->start_offset;
            ids[(row * 40 + block) * 2] = get_core_id(args);
            ids[(row * 40 + block) * 2 + 1] = get_pe_id(args);
        }
    }
}
