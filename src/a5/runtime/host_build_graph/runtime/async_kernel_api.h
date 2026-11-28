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

#include <pto/comm/comm_types.hpp>
#include <pto/comm/pto_comm_inst.hpp>

#include "async_context.h"
#include "platform_comm/ccu/ccu_device_submission.h"
#include "platform_comm/comm_context.h"

inline __aicore__ bool submit_ccu_mission(
    AsyncCtx &ctx, const __gm__ CommContext *domain, uint32_t ccu_id, const void *parameters, uint64_t parameter_size
) {
    const auto *workspace = simpler::ccu::ccu_workspace(domain);
    const auto *mission = simpler::ccu::ccu_control(workspace, ccu_id);
    auto register_ccu_completion = [&](uint64_t address, uint64_t mask) __aicore__ {
        const CompletionToken token{address, 0, COMPLETION_ENGINE_CCU, COMPLETION_TYPE_CCU, mask};
        return register_completion_condition(ctx, token);
    };
    const bool submitted = simpler::ccu::submit_mission(mission, parameters, parameter_size, register_ccu_completion);
    if (!submitted) pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    return submitted;
}

template <size_t N>
inline __aicore__ bool submit_ccu_mission(
    AsyncCtx &ctx, const __gm__ CommContext *domain, uint32_t ccu_id, const simpler::ccu::CcuCallArg (&call_args)[N]
) {
    uint64_t words[N * 2]{};
    uint64_t bytes = 0;
    if (!simpler::ccu::resolve_call_args(domain, call_args, words, bytes)) {
        pto2::detail::defer_error(ctx, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        return false;
    }
    return submit_ccu_mission(ctx, domain, ccu_id, words, bytes);
}

inline __aicore__ void
send_notification(volatile __gm__ void *remote_counter_addr, int32_t value, pto::comm::NotifyOp notify_op) {
    __gm__ int32_t *counter = reinterpret_cast<__gm__ int32_t *>(const_cast<__gm__ void *>(remote_counter_addr));
    pto::comm::Signal signal(counter);
    pto::comm::TNOTIFY(signal, value, notify_op);
}

inline __aicore__ void
save_expected_notification_counter(AsyncCtx &ctx, volatile __gm__ void *counter_addr, uint32_t expected_value) {
    CompletionToken token{
        reinterpret_cast<uint64_t>(counter_addr), expected_value, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0
    };
    (void)register_completion_condition(ctx, token);
}
