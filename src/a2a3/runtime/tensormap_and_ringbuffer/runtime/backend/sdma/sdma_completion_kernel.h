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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_KERNEL_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_KERNEL_H_

#include <stdint.h>

#include <pto/comm/async_common/async_event_impl.hpp>
#include <pto/npu/comm/async/sdma/sdma_async_intrin.hpp>

#include "pto_async_kernel_api.h"
#include "pto_completion_ingress.h"
#include "pto_runtime_status.h"

#ifndef __aicore__
#define __aicore__
#endif
#ifndef __gm__
#define __gm__
#endif

// SDMA backend kernel-side helpers. PR-D rewrites the example to use the
// merged send_request_entry overload instead of these lower-level helpers;
// they remain here for the transitional commit.
inline __aicore__ void defer_sdma_event_record(AsyncCtx &ctx, volatile __gm__ void *record_addr) {
    defer_condition(
        ctx, reinterpret_cast<uint64_t>(record_addr), 0, COMPLETION_ENGINE_SDMA,
        COMPLETION_TYPE_SDMA_EVENT_RECORD
    );
}

template <typename PtoAsyncEvent, typename PtoAsyncSession>
inline __aicore__ void
defer_pto_async_event(AsyncCtx &ctx, const PtoAsyncEvent &event, const PtoAsyncSession &session) {
    if (ctx.task_token.is_invalid() || ctx.completion_count == nullptr || ctx.completion_entries == nullptr) {
        (void)event.Wait(session);
        return;
    }
    if (event.handle == 0) {
        return;
    }

    const uint32_t engine = static_cast<uint32_t>(event.engine);
    if (engine != static_cast<uint32_t>(::pto::comm::DmaEngine::SDMA)) {
        defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }

    ::pto::comm::sdma::detail::UbTmpBuf tmp_buf;
    uint32_t sync_id = 0;
    __gm__ uint8_t *recv_workspace = nullptr;
    uint32_t queue_num = 0;
    if (!::pto::comm::sdma::detail::PrepareEventCheck(
            session.sdmaSession, tmp_buf, sync_id, recv_workspace, queue_num
        )) {
        defer_error(ctx, PTO2_ERROR_ASYNC_COMPLETION_INVALID);
        return;
    }
    for (uint32_t queue_id = 0; queue_id < queue_num; ++queue_id) {
        defer_sdma_event_record(ctx, ::pto::comm::sdma::detail::GetEventRecord(recv_workspace, queue_id));
    }
}

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_BACKEND_SDMA_SDMA_COMPLETION_KERNEL_H_
