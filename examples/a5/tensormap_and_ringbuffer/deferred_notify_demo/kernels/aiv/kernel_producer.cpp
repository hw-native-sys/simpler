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
#include <pto/comm/comm_types.hpp>
#include <pto/comm/pto_comm_inst.hpp>

#ifndef __gm__
#define __gm__
#endif
#ifndef __aicore__
#define __aicore__ [aicore]
#endif

#include "platform_comm/comm_context.h"
#include "tensor.h"

template <typename T>
static inline __aicore__ __gm__ T *comm_remote_ptr(__gm__ CommContext *ctx, __gm__ T *local_ptr, int peer_rank) {
    uint64_t local_base = ctx->windowsIn[ctx->rankId];
    uint64_t offset = reinterpret_cast<uint64_t>(local_ptr) - local_base;
    return reinterpret_cast<__gm__ T *>(ctx->windowsIn[peer_rank] + offset);
}

extern "C" __aicore__ __attribute__((always_inline)) void kernel_entry(__gm__ int64_t *args) {
    __gm__ Tensor *partial_tensor = reinterpret_cast<__gm__ Tensor *>(args[0]);
    __gm__ Tensor *mailbox_tensor = reinterpret_cast<__gm__ Tensor *>(args[1]);
    __gm__ int32_t *local_counter = reinterpret_cast<__gm__ int32_t *>(args[3]);
    __gm__ CommContext *ctx = reinterpret_cast<__gm__ CommContext *>(args[4]);

    __gm__ float *partial =
        reinterpret_cast<__gm__ float *>(partial_tensor->buffer.addr) + partial_tensor->start_offset;
    __gm__ float *mailbox =
        reinterpret_cast<__gm__ float *>(mailbox_tensor->buffer.addr) + mailbox_tensor->start_offset;

    int peer_rank = (static_cast<int>(ctx->rankId) + 1) % static_cast<int>(ctx->rankNum);
    __gm__ float *peer_mailbox = comm_remote_ptr(ctx, mailbox, peer_rank);
    uint32_t n = static_cast<uint32_t>(partial_tensor->shapes[0]);
    for (uint32_t i = 0; i < n; ++i) {
        peer_mailbox[i] = partial[i];
    }
#if defined(__CCE_KT_TEST__) || defined(__CCE_AICORE__) || defined(__DAV_C220__)
    dcci((__gm__ int32_t *)peer_mailbox, ENTIRE_DATA_CACHE, CACHELINE_OUT);
#if defined(__CPU_SIM)
    dsb(0);
#else
    dsb(DSB_DDR);
#endif
    pipe_barrier(PIPE_ALL);
#endif

    __gm__ int32_t *peer_counter = comm_remote_ptr(ctx, local_counter, peer_rank);
    pto::comm::Signal peer_signal(peer_counter);
    pto::comm::TNOTIFY(peer_signal, (int32_t)1, pto::comm::NotifyOp::AtomicAdd);
}
