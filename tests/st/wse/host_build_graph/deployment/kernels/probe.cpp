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
#include <wse/tile_ops.h>
#include "intrinsic.h"
#include "sram_args.h"

extern "C" void kernel_entry(int64_t *args) {
    wse::SramView input, output, records;
    if (!wse::make_sram_view(args, *reinterpret_cast<Tensor *>(args[0]), input) ||
        !wse::make_sram_view(args, *reinterpret_cast<Tensor *>(args[1]), output) ||
        !wse::make_sram_view(args, *reinterpret_cast<Tensor *>(args[2]), records))
        throw std::invalid_argument("deployment probe requires live SRAM views");
    const uint64_t stage = args[3], token = args[4];
    const uint64_t block = get_block_idx(args), blocks = get_block_num(args);
    const uint64_t index = stage * blocks + block;
    if ((index + 1) * 1024 * sizeof(float) > output.bytes || (index + 1) * 8 * sizeof(uint64_t) > records.bytes)
        throw std::out_of_range("deployment probe extent");
    wse::Tile<32, 32> tile, result;
    wse::TLOAD(tile, {reinterpret_cast<float *>(input.address), input.bytes / 4, 32, 32, 32, 1});
    wse::TRELU(result, tile);
    auto *destination = reinterpret_cast<float *>(output.address) + index * 1024;
    wse::TSTORE({destination, 1024, 32, 32, 32, 1}, result);
    auto *log = reinterpret_cast<uint64_t *>(records.address) + index * 8;
    log[0] = block;
    log[1] = get_core_id(args);
    log[2] = get_pe_id(args);
    log[3] = input.owner;
    log[4] = output.owner;
    log[5] = input.address;
    log[6] = reinterpret_cast<uint64_t>(destination);
    log[7] = token;
}
