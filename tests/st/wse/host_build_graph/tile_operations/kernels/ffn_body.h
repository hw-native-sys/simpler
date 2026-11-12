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
#include <wse/tile_ops.h>
#include "intrinsic.h"
#include "tensor.h"

inline wse::TensorView
ffn_view(int64_t *args, int index, size_t offset, size_t row_stride = 32, size_t col_stride = 1) {
    const auto &t = *reinterpret_cast<Tensor *>(args[index]);
    if (offset > t.numel()) throw std::out_of_range("FFN view offset");
    return {reinterpret_cast<float *>(t.buffer.addr) + t.start_offset + offset,
            t.numel() - offset,
            32,
            32,
            row_stride,
            col_stride};
}
inline void ffn_compute(int64_t *args, int stage) {
    const auto block = get_block_idx(args);
    const size_t row = get_thread_idx(args), column = block;
    const size_t output_offset = column * 128 * 32 + row * 32 * 32;
    using Tile = wse::Tile<32, 32>;
    Tile a, result;
    if (stage == 1) {
        wse::TLOAD(a, ffn_view(args, 0, output_offset));
        wse::TRELU(result, a);
        wse::TSTORE(ffn_view(args, 1, output_offset), result);
        return;
    }
    wse::SharedTile<Tile> weights(
        get_pe_id(args), stage == 0 ? 0 : 1, 0, 15, get_wse_block_scope(args), get_wse_operation_trace(args)
    );
    const size_t chunks = stage == 0 ? 1 : 40;
    for (size_t k = 0; k < chunks; ++k) {
        if (row == 0) {
            Tile weight;
            wse::TLOAD(weight, ffn_view(args, 1, (column * chunks + k) * 1024, 1, 32));
            weights.publish(k + 1, weight);
        }
        wse::TLOAD(a, ffn_view(args, 0, row * 1024 + (stage == 0 ? 0 : k * 128 * 32)));
        if (k == 0) wse::TMATMUL(result, a, weights, k + 1);
        else wse::TMATMUL_ACC(result, result, a, weights, k + 1);
    }
    wse::TSTORE(ffn_view(args, 2, output_offset), result);
    if (stage == 0) {
        auto *t = reinterpret_cast<Tensor *>(args[3]);
        auto *ids = reinterpret_cast<float *>(t->buffer.addr) + t->start_offset;
        ids[(row * 40 + block) * 2] = get_core_id(args);
        ids[(row * 40 + block) * 2 + 1] = get_pe_id(args);
    }
}
