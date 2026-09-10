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

#include "tmr_kernel_invocation.h"

#include "device_runner_base.h"

namespace simpler::tmr {

int enqueue_tmr_invocation_aicpu(
    DeviceRunnerBase &runner, void *stream, int32_t aicpu_num, const TmrEncodingCandidate &candidate,
    const PreparedInvocationView &callable, const TmrExecutionBindingView &binding
) noexcept {
    if (stream == nullptr || aicpu_num <= 0) return PTO_RUNTIME_ERR_INTERNAL;
    const auto status = validate_tmr_submission(candidate, callable, binding);
    if (status == InvocationStatus::StaleCallable || status == InvocationStatus::InvalidBinding)
        return PTO_RUNTIME_ERR_INVALID_STATE;
    if (status != InvocationStatus::Ok) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        const auto packet = candidate.packet();
        return runner.launch_aicpu_payload(
            stream, const_cast<uint8_t *>(packet.data), packet.size, TmrKernelInvocationName, aicpu_num
        );
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

}  // namespace simpler::tmr
