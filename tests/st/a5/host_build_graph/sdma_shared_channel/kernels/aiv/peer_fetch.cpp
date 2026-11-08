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
#include "common/platform_config.h"

inline __aicore__ void publish_stamp(__gm__ int64_t *p, int64_t value) {
    *p = value;
    dcci(p, cache_line_t::SINGLE_CACHE_LINE, dcci_dst_t::CACHELINE_OUT);
    dsb((mem_dsb_t)0);
}
inline __aicore__ int64_t observe_stamp(__gm__ int64_t *p) {
    return static_cast<int64_t>(__builtin_cce_ld_dev(reinterpret_cast<__gm__ uint64_t *>(p), 0));
}
extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *input = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *output = reinterpret_cast<__gm__ Tensor *>(args[1]);
    auto *stamps_tensor = reinterpret_cast<__gm__ Tensor *>(args[2]);
    auto *stamps = reinterpret_cast<__gm__ int64_t *>(stamps_tensor->buffer.addr) + stamps_tensor->start_offset;
    const int32_t lane = get_sub_block_id(args);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[3]);
    auto ctx = get_async_ctx(args);
    if (!async_ctx_is_deferred(ctx) || comm->rankNum < 2) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    const uint64_t count = static_cast<uint64_t>(args[4]) / 2;
    const uint64_t capacity = output->shapes[0] / 2;
    if (count == 0 || (capacity != count && capacity != 16 * count)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    const uint32_t peer = (comm->rankId + 1) % comm->rankNum;
    auto *local_input = reinterpret_cast<__gm__ float *>(input->buffer.addr) + input->start_offset + lane * capacity;
    const uint64_t offset = reinterpret_cast<uint64_t>(local_input) - comm->windowsIn[comm->rankId];
    auto *remote = reinterpret_cast<__gm__ float *>(comm->windowsIn[peer] + offset);
    auto *destination = reinterpret_cast<__gm__ float *>(output->buffer.addr) + output->start_offset + lane * capacity;
    using Shape = pto::Shape<1, 1, 1, 1, pto::DYNAMIC>;
    using Stride = pto::Stride<pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, pto::DYNAMIC, 1>;
    using Data = pto::GlobalTensor<float, Shape, Stride>;
    using Scratch = pto::Tile<pto::TileType::Vec, uint8_t, 1, SDMA_SCRATCH_ALIGNMENT>;
    Scratch scratch;
    pto::TASSIGN(scratch, 0);
    auto *workspace = reinterpret_cast<__gm__ uint8_t *>(comm_workspace<CommWorkspaceKind::SDMA>(*comm).addr);
    publish_stamp(stamps + lane * 16 + 1, get_block_idx());
    for (uint32_t iteration = 0; iteration < 16; ++iteration) {
        publish_stamp(stamps + lane * 16, iteration + 1);
        const uint64_t start = get_sys_cnt();
        while (observe_stamp(stamps + (1 - lane) * 16) < iteration + 1) {
            if (get_sys_cnt() - start > PLATFORM_PROF_SYS_CNT_FREQ) {
                pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
                return;
            }
        }
        // Separate intervals make every post observable: a later copy cannot
        // hide a dropped earlier transfer by writing the same payload again.
        const uint64_t iteration_offset = capacity == count ? 0 : iteration * count;
        Data source(remote + iteration_offset, Shape(1, 1, 1, 1, count), Stride(count, count, count, count, 1));
        Data target(destination + iteration_offset, Shape(1, 1, 1, 1, count), Stride(count, count, count, count, 1));
        const uint32_t before = *ctx.completion_count;
        if (!send_request_entry(ctx, SdmaTget(target, source, scratch, workspace, 0, static_cast<uint32_t>(args[6]))))
            return;
        const uint32_t registered = *ctx.completion_count;
        if (registered <= before || registered - before > 2 || registered > ctx.completion_capacity) {
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            return;
        }
        for (uint32_t record = 0; record < registered - before; ++record)
            publish_stamp(
                stamps + 32 + iteration * 32 + lane * 16 + record,
                ctx.completion_entries[before + record].backend_cookie
            );
        publish_stamp(stamps + lane * 16 + 3, registered - before);
        publish_stamp(stamps + lane * 16 + 2, registered);
    }
}
