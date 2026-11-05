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
#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif
#ifdef MEMORY_BASE
#undef MEMORY_BASE
#endif
#ifndef REGISTER_BASE
#define REGISTER_BASE
#endif
#include <pto/pto-inst.hpp>
#include "backend/urma/urma_completion_kernel.h"
#include "tensor.h"

template <int RowBytes>
static inline __aicore__ void gather_rows(
    AsyncCtx &ctx, __gm__ uint8_t *workspace, uint32_t peer, __gm__ Tensor *table, __gm__ Tensor *ids,
    __gm__ Tensor *output, __gm__ int32_t *status, uint64_t count, uint32_t element_bytes, uint64_t view_offset
) {
    using Shape = pto::Shape<1, 1, 1, 1, RowBytes>;
    using Stride = pto::Stride<RowBytes, RowBytes, RowBytes, RowBytes, 1>;
    using Row = pto::GlobalTensor<uint8_t, Shape, Stride>;
    auto *indices = reinterpret_cast<__gm__ int32_t *>(ids->buffer.addr) + ids->start_offset;
    auto *destination = reinterpret_cast<__gm__ uint8_t *>(output->buffer.addr) + output->start_offset * element_bytes;
    for (uint64_t i = 0; i < count; ++i) {
        const int32_t row = indices[i];
        if (row < 0 || static_cast<uint32_t>(row) >= table->shapes[0]) {
            *status = 2;
            continue;
        }
        const uint64_t offset = view_offset + static_cast<uint64_t>(row) * RowBytes;
        auto *source = pto2::urma_backend::peer_mr_ptr<uint8_t>(workspace, peer, offset);
        if (!send_request_entry(ctx, UrmaTget(Row(destination + i * RowBytes), Row(source), workspace, peer))) {
            *status = 3;
            break;
        }
    }
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    auto *table = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *ids = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *status = reinterpret_cast<__gm__ Tensor *>(args[3]);
    const uint64_t count = static_cast<uint64_t>(args[4]);
    const uint32_t peer = static_cast<uint32_t>(args[5]);
    const uint32_t local_rank = static_cast<uint32_t>(args[6]);
    auto *code = reinterpret_cast<__gm__ int32_t *>(status->buffer.addr) + status->start_offset;
    auto *workspace = get_dma_workspace(args, DMA_WORKSPACE_URMA);
    *code = 0;
    const bool bf16 = table->dtype == DataType::BFLOAT16 && table->shapes[1] == 256;
    const bool fp8 = table->dtype == DataType::FP8E4M3FN && table->shapes[1] == 64;
    if (!workspace || (!bf16 && !fp8) || table->dtype != output->dtype || count > ids->shapes[0] ||
        count > output->shapes[0]) {
        *code = 1;
    } else if (peer >= reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(workspace)->rankCount ||
               local_rank >= reinterpret_cast<__gm__ pto::comm::urma::UrmaInfo *>(workspace)->rankCount ||
               peer == local_rank) {
        *code = 1;
    } else {
        const uint32_t element_bytes = bf16 ? 2 : 1;
        const uint64_t local_base = pto2::urma_backend::peer_mr_base_addr(workspace, local_rank);
        // Materialization can fold the Tensor's byte offset into buffer.addr.
        const uint64_t view_address = table->buffer.addr + table->start_offset * element_bytes;
        if (local_base == 0 || view_address < local_base) {
            *code = 1;
        } else {
            const uint64_t view_offset = view_address - local_base;
            AsyncCtx context = get_async_ctx(args);
            if (bf16) {
                gather_rows<512>(context, workspace, peer, table, ids, output, code, count, 2, view_offset);
            } else {
                gather_rows<64>(context, workspace, peer, table, ids, output, code, count, 1, view_offset);
            }
        }
    }
    dcci(code, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb(DSB_DDR);
}
