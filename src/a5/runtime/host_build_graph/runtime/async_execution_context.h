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

#pragma once

#include "dispatch_payload.h"
#include "scheduler/scheduler_async_layout.h"
#include "host_build_graph/runtime_status.h"

// This view is valid only during the current kernel invocation. Mutable
// registration belongs to the shared writer, including across copied views.
inline __aicore__ bool async_execution_context_load(__gm__ DispatchPayload &payload, AsyncCtx *out) {
    if (out == nullptr) return false;
    auto &execution = payload.async_context;
    auto ctx = AsyncCtx::from_storage(payload.local_context.async_ctx);
    const bool deferred = ctx.task_token.is_valid();
    const bool writer_valid =
        deferred ?
            ctx.completion_count != nullptr &&
                execution.writer_state == reinterpret_cast<uint64_t>(ctx.completion_count) +
                                              offsetof(SchedulerAsyncGroupWriter, reservation) &&
                ctx.completion_error_code == reinterpret_cast<volatile __gm__ int32_t *>(
                                                 reinterpret_cast<uint64_t>(ctx.completion_count) +
                                                 offsetof(SchedulerAsyncGroupWriter, error_code)
                                             ) :
            execution.writer_state == 0 && ctx.completion_count == nullptr && ctx.completion_entries == nullptr &&
                ctx.completion_capacity == 0 && ctx.completion_error_code == &execution.error_code;
    if (payload.args[SPMD_LOCAL_CONTEXT_INDEX] != reinterpret_cast<uint64_t>(&payload.local_context) ||
        payload.args[SPMD_GLOBAL_CONTEXT_INDEX] != reinterpret_cast<uint64_t>(&payload.global_context) ||
        payload.async_context_address != reinterpret_cast<uint64_t>(&execution) ||
        execution.magic != ASYNC_EXECUTION_CONTEXT_MAGIC || execution.subtask_slot >= 3 || !writer_valid) {
        if (execution.error_code == 0) execution.error_code = SIMPLER_ERROR_ASYNC_COMPLETION_INVALID;
        *out = {};
        out->task_token = TaskId::invalid();
        out->completion_error_code = &execution.error_code;
        out->execution_context = &execution;
        return false;
    }
    ctx.execution_context = &execution;
    ctx.completion_reservation =
        reinterpret_cast<volatile __gm__ AsyncCompletionReservationState *>(execution.writer_state);
    *out = ctx;
    return true;
}

inline __aicore__ bool async_execution_context_load(__gm__ int64_t *args, AsyncCtx *out) {
    if (out == nullptr) return false;
    *out = {};
    out->task_token = TaskId::invalid();
    const uint64_t address = reinterpret_cast<uint64_t>(args);
    if (args == nullptr || address < offsetof(DispatchPayload, args)) return false;
    const uint64_t frame = address - offsetof(DispatchPayload, args);
    if (frame % alignof(DispatchPayload) != 0) return false;
    return async_execution_context_load(*reinterpret_cast<__gm__ DispatchPayload *>(frame), out);
}
