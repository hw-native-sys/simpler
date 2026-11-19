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
#include "sram_args.h"
#include "intrinsic.h"
#include "tensor.h"

extern "C" void kernel_entry(int64_t *args) {
    auto &input = *reinterpret_cast<Tensor *>(args[0]);
    auto &output = *reinterpret_cast<Tensor *>(args[1]);
    auto &records = *reinterpret_cast<Tensor *>(args[2]);
    const auto stage = args[3];
    const auto b = get_block_idx(args);
    wse::SramView in, out, trace;
    if (!wse::make_sram_view(args, input, in) || !wse::make_sram_view(args, output, out) ||
        !wse::make_sram_view(args, records, trace))
        return;
    auto *x = reinterpret_cast<uint64_t *>(in.address);
    auto *y = reinterpret_cast<uint64_t *>(out.address);
    auto *log = reinterpret_cast<uint64_t *>(trace.address) + b * 8 + stage * 4;
    if (stage == 0) x[b * input.strides[0]] = 100 + b;
    else y[b * output.strides[0]] = x[b * input.strides[0]] + 1;
    log[0] = in.owner;
    log[1] = out.owner;
    log[2] = get_core_id(args);
    log[3] = get_pe_id(args);
    if (stage == 1 && b == 0) {
        Tensor bad = input;
        bad.start_offset = input.buffer.size / 8;
        wse::SramView rejected;
        if (wse::make_sram_view(args, bad, rejected) || wse::make_sram_view(args, input, rejected, (in.owner + 1) % 2))
            y[b] = 0;
        const auto *global = reinterpret_cast<const GlobalContext *>(args[SPMD_GLOBAL_CONTEXT_INDEX]);
        const auto &access = *reinterpret_cast<const wse::SramAccess *>(global->sram_base + global->sram_access_offset);
        const auto *regions = reinterpret_cast<const wse::SramRegion *>(global->sram_base + access.regions_offset);
        bad = input;
        bad.buffer.addr = regions[0].runtime_base;
        bad.buffer.size = 8;
        bad.start_offset = 0;
        bad.shapes[0] = bad.strides[0] = 1;
        if (wse::make_sram_view(args, bad, rejected)) y[b] = 0;
        // Adjacent Core application windows share a backing, but a Tensor still has one owner.
        bad = input;
        bad.buffer.addr = regions[0].base + regions[0].capacity - regions[0].reserve - 8;
        bad.buffer.size = 16;
        bad.start_offset = 0;
        bad.shapes[0] = 2;
        bad.strides[0] = 1;
        if (wse::make_sram_view(args, bad, rejected)) y[b] = 0;
        bad = input;
        bad.strides[0] = UINT32_MAX;
        if (input.shapes[0] > 1 && wse::make_sram_view(args, bad, rejected)) y[b] = 0;
    }
}
