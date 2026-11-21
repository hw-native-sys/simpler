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
#include <pto/pto-inst.hpp>
#include "async_kernel_api.h"
#include "kernel_error_api.h"
#include "kernel_cancellation_api.h"
#include "platform_comm/comm_context.h"
#include "tensor.h"
#if defined(__CPU_SIM)
#include <cstdio>
#endif

// Each signal has its own cache line; notification does not share a dirty line
// with a local writer. Both ranks own their domain until run retirement.
constexpr int kSignalWords = 16;

extern "C" __aicore__ void kernel_entry(__gm__ int64_t *args) {
    auto ctx = get_async_ctx(args);
    auto *out = reinterpret_cast<__gm__ Tensor *>(args[0]);
    auto *result = reinterpret_cast<__gm__ uint32_t *>(out->buffer.addr) + out->start_offset;
    auto *signals = reinterpret_cast<volatile __gm__ uint32_t *>(args[1]);
    auto *comm = reinterpret_cast<__gm__ CommContext *>(args[2]);
    const int mode = static_cast<int>(args[3]);
    const auto offset = reinterpret_cast<uint64_t>(signals) - comm->windowsIn[comm->rankId];
    auto *peer = reinterpret_cast<__gm__ uint32_t *>(comm->windowsIn[1 - comm->rankId] + offset);
#if NOTIFY_CONSUMER
    *result = 37;
    pto2::detail::defer_flush_range(result, sizeof(*result));
#if defined(__CPU_SIM)
    fprintf(stderr, "PEER_NOTIFY_CONSUMER_ENTERED\n");
    fflush(stderr);
#endif
#else
    if (comm->rankId == 0) {
        save_expected_notification_counter(ctx, signals, 1);
        if (mode != 0) report_kernel_error(ctx, KernelErrorCode::INDEX_OUT_OF_RANGE);
        // The peer cannot signal before registration and the requested injection.
        send_notification(peer + kSignalWords, 1, pto::comm::NotifyOp::Set);
    } else {
        while (true) {
            pto2::detail::defer_flush_range(signals + kSignalWords, sizeof(uint32_t));
            if (signals[kSignalWords] == 1) break;
            if (kernel_cancel_requested(ctx)) return;
        }
        if (mode != 2) send_notification(peer, 1, pto::comm::NotifyOp::Set);
        *result = 37;
        pto2::detail::defer_flush_range(result, sizeof(*result));
    }
#endif
}
