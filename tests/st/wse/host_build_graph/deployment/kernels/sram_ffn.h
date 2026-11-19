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
#include "sram_args.h"

inline wse::SramView ffn_sram(int64_t *args, int index) {
    wse::SramView view;
    if (!wse::make_sram_view(args, *reinterpret_cast<Tensor *>(args[index]), view))
        throw std::invalid_argument("FFN requires live SRAM shards");
    return view;
}
inline wse::TensorView ffn_tile(int64_t *args, int index, size_t offset, bool transposed = false) {
    auto view = ffn_sram(args, index);
    if (offset > view.bytes / sizeof(float)) throw std::out_of_range("FFN shard offset");
    return {
        reinterpret_cast<float *>(view.address) + offset,
        view.bytes / sizeof(float) - offset,
        32,
        32,
        transposed ? 1u : 32u,
        transposed ? 32u : 1u
    };
}
inline void sram_ffn(int64_t *args, int stage) {
    const auto block = get_block_idx(args);
    const size_t row = get_thread_idx(args);
    const int tensors = stage == 1 ? 3 : 7;
    const auto publisher = static_cast<uint32_t>(args[tensors]);
    const uint64_t generation = 1, instance = args[tensors + 1];
    const auto record_view = ffn_sram(args, tensors - 1);
    const uint64_t record_offset = row * 20;
    if (record_offset + 20 > record_view.bytes / 8) throw std::out_of_range("FFN record extent");
    auto *record = reinterpret_cast<uint64_t *>(record_view.address) + record_offset;
    std::fill_n(record, 20, UINT64_MAX);
    const int output_index = stage == 1 ? 1 : 5;
    const auto output = ffn_sram(args, output_index);
    record[0] = get_core_id(args);
    record[1] = get_pe_id(args);
    record[2] = output.owner;
    record[3] = output.address + row * 4096;
    record[6] = publisher;
    record[7] = generation;
    record[10] = block;
    record[11] = instance;
    using Tile = wse::Tile<32, 32>;
    Tile a, result;
    if (stage == 1) {
        const auto input = ffn_sram(args, 0);
        record[4] = input.owner;
        record[5] = input.address + row * 4096;
        wse::TLOAD(a, ffn_tile(args, 0, row * 1024));
        wse::TRELU(result, a);
    } else {
        const auto weight = ffn_sram(args, 4);
        record[8] = weight.owner;
        record[9] = weight.address;
        wse::SharedTile<Tile> weights(
            get_pe_id(args), stage == 0 ? 0 : 1, publisher, 15, get_wse_block_scope(args), get_wse_operation_trace(args)
        );
        const size_t chunks = stage == 0 ? 1 : 4;
        for (size_t k = 0; k < chunks; ++k) {
            if (generation > UINT64_MAX - k) throw std::overflow_error("FFN generation overflow");
            if (row == 0) {
                Tile w;
                wse::TLOAD(w, ffn_tile(args, 4, k * 1024, true));
                weights.publish(generation + k, w);
            }
            const int input_index = stage == 0 ? row : k;
            const size_t input_offset = stage == 0 ? 0 : row * 1024;
            const auto input = ffn_sram(args, input_index);
            if (k == 0) {
                record[4] = input.owner;
                record[5] = input.address + input_offset * 4;
            }
            record[12 + k * 2] = input.owner;
            record[13 + k * 2] = input.address + input_offset * 4;
            wse::TLOAD(a, ffn_tile(args, input_index, input_offset));
            if (k == 0) wse::TMATMUL(result, a, weights, generation + k);
            else wse::TMATMUL_ACC(result, result, a, weights, generation + k);
        }
    }
    wse::TSTORE(ffn_tile(args, output_index, row * 1024), result);
}
