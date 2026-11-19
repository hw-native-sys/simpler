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

#include "kernel_operator.h"
#include "tensor.h"

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *table = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *ids = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *status = reinterpret_cast<__gm__ Tensor *>(args[3]);
    const uint64_t count = static_cast<uint64_t>(args[4]);
    const uint32_t element_bytes = table->dtype == DataType::BFLOAT16 ? 2 : 1;
    const uint32_t row_bytes = table->shapes[1] * element_bytes;
    auto *code = reinterpret_cast<__gm__ int32_t *>(status->buffer.addr) + status->start_offset;
    *code = 0;
    if ((table->dtype != DataType::BFLOAT16 && table->dtype != DataType::FP8E4M3FN) || table->dtype != output->dtype ||
        row_bytes == 0 || row_bytes > 4096 || row_bytes % 32 != 0 || count > ids->shapes[0] ||
        count > output->shapes[0]) {
        *code = 1;
    } else {
        auto *indices = reinterpret_cast<__gm__ int32_t *>(ids->buffer.addr) + ids->start_offset;
        AscendC::TPipe pipe;
        AscendC::TQueBind<AscendC::QuePosition::VECIN, AscendC::QuePosition::VECOUT, 1> queue;
        pipe.InitBuffer(queue, 1, row_bytes);
        AscendC::GlobalTensor<uint8_t> source, destination;
        source.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t *>(table->buffer.addr) + table->start_offset * element_bytes
        );
        destination.SetGlobalBuffer(
            reinterpret_cast<__gm__ uint8_t *>(output->buffer.addr) + output->start_offset * element_bytes
        );
        for (uint64_t i = 0; i < count; ++i) {
            const int32_t row = indices[i];
            if (row < 0 || static_cast<uint32_t>(row) >= table->shapes[0]) {
                *code = 2;
                continue;
            }
            auto tile = queue.AllocTensor<uint8_t>();
            AscendC::DataCopy(tile, source[static_cast<uint64_t>(row) * row_bytes], row_bytes);
            queue.EnQue(tile);
            tile = queue.DeQue<uint8_t>();
            AscendC::DataCopy(destination[i * row_bytes], tile, row_bytes);
            queue.FreeTensor(tile);
        }
        AscendC::PipeBarrier<PIPE_ALL>();
    }
    dcci(code, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb(DSB_DDR);
}
