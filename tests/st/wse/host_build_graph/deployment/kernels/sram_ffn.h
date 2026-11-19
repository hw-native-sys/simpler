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
#include <algorithm>
#include "sram_args.h"

inline wse::SramView ffn_sram(int64_t *args, int index) {
    wse::SramView view;
    if (!wse::make_sram_view(args, *reinterpret_cast<Tensor *>(args[index]), view))
        throw std::invalid_argument("FFN requires live SRAM shards");
    return view;
}
inline FfnMemory ffn_tile(const wse::SramView &view, size_t offset) {
    return wse::tile_memory<32, 32>(reinterpret_cast<float *>(view.address), view.bytes / sizeof(float), offset);
}

template <int Stage>
void sram_ffn(int64_t *args) {
    const auto block = get_block_idx(args);
    const size_t row = get_thread_idx(args);
    const int tensors = Stage == 1 ? 3 : 7;
    const auto publisher = static_cast<uint32_t>(args[tensors]);
    const uint64_t generation = 1, instance = args[tensors + 1];
    const auto record_view = ffn_sram(args, tensors - 1);
    const uint64_t record_offset = row * 20;
    if (record_offset + 20 > record_view.bytes / 8) throw std::out_of_range("FFN record extent");
    auto *record = reinterpret_cast<uint64_t *>(record_view.address) + record_offset;
    std::fill_n(record, 20, UINT64_MAX);
    const int output_index = Stage == 1 ? 1 : 5;
    const auto output = ffn_sram(args, output_index);
    record[0] = get_core_id(args);
    record[1] = get_pe_id(args);
    record[2] = output.owner;
    record[3] = output.address + row * 4096;
    record[6] = publisher;
    record[7] = generation;
    record[10] = block;
    record[11] = instance;
    auto destination = ffn_tile(output, row * 1024);
    if constexpr (Stage == 1) {
        const auto input = ffn_sram(args, 0);
        record[4] = input.owner;
        record[5] = input.address + row * 4096;
        ffn_relu(ffn_tile(input, row * 1024), destination);
    } else {
        const auto weight = ffn_sram(args, 4);
        record[8] = weight.owner;
        record[9] = weight.address;
        constexpr size_t chunks = Stage == 0 ? 1 : 4;
        auto input_shard = [&](size_t k) {
            const int index = Stage == 0 ? row : k;
            const size_t offset = Stage == 0 ? 0 : row * 1024;
            const auto input = ffn_sram(args, index);
            if (k == 0) {
                record[4] = input.owner;
                record[5] = input.address + offset * sizeof(float);
            }
            record[12 + k * 2] = input.owner;
            record[13 + k * 2] = input.address + offset * sizeof(float);
            return ffn_tile(input, offset);
        };
        ffn_project<Stage == 0 ? 0 : 1>(
            args, publisher, chunks, input_shard,
            [&](size_t k) {
                return ffn_tile(weight, k * 1024);
            },
            destination
        );
    }
}
