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

#include "async_kernel_api.h"
#include "common/platform_config.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"

namespace {
template <typename T>
inline __aicore__ __gm__ T *data(__gm__ int64_t *args, uint32_t index) {
    const auto *tensor = reinterpret_cast<__gm__ Tensor *>(args[index]);
    return reinterpret_cast<__gm__ T *>(tensor->buffer.addr) + tensor->start_offset;
}

template <typename T>
inline __aicore__ __gm__ T *remote(__gm__ CommContext *comm, __gm__ T *local, uint32_t peer) {
    return reinterpret_cast<__gm__ T *>(
        comm->windowsIn[peer] + reinterpret_cast<uint64_t>(local) - comm->windowsIn[comm->rankId]
    );
}
}  // namespace

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto *input = data<float>(args, 0);
    auto *mailbox = data<float>(args, 1);
    auto *counters = data<int32_t>(args, 2);
    auto *result = data<float>(args, 3);
    const uint32_t role = static_cast<uint32_t>(args[4]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[5]);
    const auto *input_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    const uint32_t count = input_tensor->shapes[0];
    if (role == 0) {
        auto ctx = get_async_ctx(args);
        if (!async_ctx_is_deferred(ctx)) {
            pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
            return;
        }
        for (uint32_t peer = 0; peer < comm->rankNum; ++peer)
            if (peer != comm->rankId) save_expected_notification_counter(ctx, counters + peer * 32, 1);
    } else if (role == 1) {
        // Rank-dependent publication order leaves all waiters pending first.
        const uint64_t start = get_sys_cnt();
        const uint64_t delay = static_cast<uint64_t>(comm->rankId + 1) * PLATFORM_PROF_SYS_CNT_FREQ / 1000;
        while (get_sys_cnt() - start < delay) {}
        for (uint32_t peer = 0; peer < comm->rankNum; ++peer) {
            if (peer == comm->rankId) continue;
            auto *destination = remote(comm, mailbox, peer) + comm->rankId * count;
            for (uint32_t index = 0; index < count; ++index)
                destination[index] = input[index];
            // Keep the existing remote-store -> publish -> notify ordering.
            dcci(destination, cache_line_t::ENTIRE_DATA_CACHE, dcci_dst_t::CACHELINE_OUT);
            dsb((mem_dsb_t)0);
            pipe_barrier(PIPE_ALL);
            send_notification(remote(comm, counters, peer) + comm->rankId * 32, 1, pto::comm::NotifyOp::AtomicAdd);
        }
    } else if (role == 2) {
        for (uint32_t peer = 0; peer < comm->rankNum; ++peer)
            for (uint32_t index = 0; index < count; ++index)
                result[peer * count + index] = peer == comm->rankId ? -17.0F : mailbox[peer * count + index] + 1.0F;
    }
}
