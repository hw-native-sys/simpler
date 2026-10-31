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
#include "tensor.h"
#include "pipe_sync.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *table = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *indices = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *result = reinterpret_cast<__gm__ Tensor *>(args[2]);
    // Copy BF16 payload bits without imposing model arithmetic on the transport.
    auto *source = reinterpret_cast<__gm__ uint16_t *>(table->buffer.addr) + table->start_offset;
    auto *ids = reinterpret_cast<__gm__ uint32_t *>(indices->buffer.addr) + indices->start_offset;
    auto *output = reinterpret_cast<__gm__ uint16_t *>(result->buffer.addr) + result->start_offset;
    using Row = pto::GlobalTensor<uint16_t, pto::Shape<1, 1, 1, 1, 256>, pto::Stride<256, 256, 256, 256, 1>>;
    pto::Tile<pto::TileType::Vec, uint16_t, 1, 256> tile;
    pto::TASSIGN(tile, 0);
    for (uint32_t i = 0; i < 6; ++i) {
        uint32_t row = ids[i];
        if (row < 128) {
            Row input(source + row * 256);
            pto::TLOAD(tile, input);
            set_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_MTE3, EVENT_ID0);
        } else {
            // This test operator defines an out-of-range ID as a zero row.
            pto::TEXPANDS(tile, uint16_t{0});
            set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
            wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
        }
        Row destination(output + i * 256);
        pto::TSTORE(destination, tile);
        pipe_sync();
    }
}
