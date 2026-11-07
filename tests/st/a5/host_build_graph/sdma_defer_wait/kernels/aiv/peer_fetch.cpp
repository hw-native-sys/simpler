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

#include "backend/sdma/sdma_completion_kernel.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[2]);
    auto ctx = get_async_ctx(args);
    if (!async_ctx_is_deferred(ctx) || comm->rankNum < 2) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    const uint64_t count = static_cast<uint64_t>(args[3]);
    const uint64_t capacity = output->shapes[0];
    const uint32_t peer = (comm->rankId + 1) % comm->rankNum;
    auto *local_input = reinterpret_cast<__gm__ float *>(input->buffer.addr) + input->start_offset;
    const uint64_t offset = reinterpret_cast<uint64_t>(local_input) - comm->windowsIn[comm->rankId];
    auto *remote = reinterpret_cast<__gm__ float *>(comm->windowsIn[peer] + offset);
    auto *destination = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset;
    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Data = pto::GlobalTensor<float, Shape, Stride>;
    Data source(remote, Shape(1, 1, 1, 1, count), Stride(count, count, count, count, 1));
    Data target(destination, Shape(1, 1, 1, 1, capacity), Stride(capacity, capacity, capacity, capacity, 1));
    using Scratch = pto::Tile<pto::TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT>;
    Scratch scratch;
    pto::TASSIGN(scratch, 0);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm).addr);
    (void)send_request_entry(ctx, SdmaTget(target, source, scratch, workspace));
}
