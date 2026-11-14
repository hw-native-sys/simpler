// Copyright (c) PyPTO Contributors.
// This program is free software, you can redistribute it and/or modify it under the terms and conditions of
// CANN Open Software License Agreement Version 2.0 (the "License").
// Please refer to the License for details. You may not use this file except in compliance with the License.
// THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
// INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
// See LICENSE in the root of the software repository for the full text of the License.
// -----------------------------------------------------------------------------------------------------------
// Shared Q/KV input quantization with M16-packed E8M0 scales.
#include "mx_quant.hpp"
using namespace mx_quant;
namespace {
#if defined(__DAV_VEC__)
constexpr int width = 5120, columns = 256;
using Input = Tile<TileType::Vec, bfloat16_t, 16, columns, BLayout::RowMajor, DYNAMIC, columns>;
using Shape = TileShape2D<bfloat16_t, DYNAMIC, columns>;
using GlobalStride = pto::Stride<1, 1, 1, DYNAMIC, 1>;
__aicore__ inline int valid_rows(int active, int work) {
    int rows = active - work / (width / columns) * 16;
    return rows < 0 ? 0 : (rows > 16 ? 16 : rows);
}
__aicore__ inline void load_work(__gm__ int64_t* args, int work, int slot) {
    int rows = valid_rows(args[3], work);
    if (rows > 0) {
        Input input(rows);
        TASSIGN(input, slot * 16384);
        int t0 = work / (width / columns) * 16, k0 = work % (width / columns) * columns;
        GlobalTensor<bfloat16_t, Shape, GlobalStride> source(data<bfloat16_t>(args, 0) + t0 * stride(args, 0) + k0,
            Shape(rows, columns), GlobalStride(1, 1, 1, stride(args, 0), 1));
        TLOAD(input, source);
    }
}
#endif
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t* args) {
#if defined(__DAV_VEC__)
    set_mask_norm();
    set_vector_mask(-1, -1);
    int first = get_block_idx(args), workers = get_block_num(args);
    int total = args[4] / 16 * (width / columns);
    if (first >= total) return;
    load_work(args, first, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    // Each 16-KiB slot contains input, payload, and packed scales.
    // V-to-MTE2 credits protect input reuse; MTE3-to-V credits protect output reuse.
    int iteration = 0;
    for (int work = first; work < total; work += workers, ++iteration) {
        int slot = iteration % 2;
        event_t current = slot == 0 ? EVENT_ID0 : EVENT_ID1;
        event_t next = slot == 0 ? EVENT_ID1 : EVENT_ID0;
        if (work + workers < total) {
            if (iteration > 0) wait_flag(PIPE_V, PIPE_MTE2, next);
            load_work(args, work + workers, 1 - slot);
            set_flag(PIPE_MTE2, PIPE_V, next);
        }
        int t0 = work / (width / columns) * 16, k0 = work % (width / columns) * columns;
        int rows = valid_rows(args[3], work);
        Input input(rows);
        Vec<uint8_t, 16, columns> output;
        Vec<uint8_t, 1, 16 * columns / 32> scales;
        TASSIGN(input, slot * 16384);
        TASSIGN(output, slot * 16384 + 8192);
        TASSIGN(scales, slot * 16384 + 12288);
        wait_flag(PIPE_MTE2, PIPE_V, current);
        wait_flag(PIPE_MTE3, PIPE_V, current);
        quantize<columns, Input>(output.data(), scales.data(), input.data(), rows);
        set_flag(PIPE_V, PIPE_MTE2, current);
        set_flag(PIPE_V, PIPE_MTE3, current);
        wait_flag(PIPE_V, PIPE_MTE3, current);
        GlobalTensor<uint8_t, TileShape2D<uint8_t, 16, columns>, GlobalStride> target(
            data<uint8_t>(args, 1) + t0 * width + k0,
            TileShape2D<uint8_t, 16, columns>(), GlobalStride(1, 1, 1, width, 1));
        GlobalTensor<uint8_t, TileShape2D<uint8_t, 1, 128>, BaseShape2D<uint8_t, 1, 128>> scale_target(
            data<uint8_t>(args, 2) + t0 * (width / 32) + k0 / columns * 128);
        TSTORE(target, output);
        TSTORE(scale_target, scales);
        set_flag(PIPE_MTE3, PIPE_V, current);
    }
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    if (iteration > 1) wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
#endif
}
