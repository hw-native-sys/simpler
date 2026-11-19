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

#include <cstdint>
#include <pto/pto-inst.hpp>
#include "common/platform_config.h"
#include "intrinsic.h"
#include "tensor.h"
#include "pipe_sync.h"

using namespace pto;

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *source = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *target = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *input = reinterpret_cast<__gm__ float *>(source->buffer.addr) + source->start_offset;
    auto *output =
        reinterpret_cast<__gm__ float *>(target->buffer.addr) + target->start_offset + get_block_idx(args) * 32;
    using Global = GlobalTensor<float, Shape<1, 1, 1, 1, 16>, pto::Stride<16, 16, 16, 16, 1>>;
    using Vector = Tile<TileType::Vec, float, 1, 16, BLayout::RowMajor, -1, -1>;
    Vector tile(1, 16);
    TASSIGN(tile, 0);
    Global src(input);
    Global dst(output);
    TLOAD(tile, src);
    set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
    if (args[2] == 2) set_atomic_none();
    if (args[2] == 1) {
        set_atomic_f32();
        set_atomic_add();
    }
    // AtomicNone TSTORE preserves the incoming hardware mode. A value of 5
    // proves atomic-add was active; ordinary replacement must produce 1.
    TSTORE(dst, tile);
    pipe_sync();
    output[16] = static_cast<float>(static_cast<uint32_t>(get_coreid()) & AICORE_COREID_MASK);
    output[17] = static_cast<float>(get_block_num(args));
    output[18] = static_cast<float>(args[2]);
    dcci(output + 16, SINGLE_CACHE_LINE, CACHELINE_OUT);
    dsb((mem_dsb_t)0);
    // Leave the mode after all data and identity publications have finished.
    if (args[2] == 1) {
        set_atomic_f32();
        set_atomic_add();
    }
}
