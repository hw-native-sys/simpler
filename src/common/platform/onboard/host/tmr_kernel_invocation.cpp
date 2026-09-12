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
    const PreparedInvocationView &callable, const TmrExecutionBindingView &binding, uint64_t residency_address
) noexcept {
    if (stream == nullptr || aicpu_num <= 0) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        std::vector<uint8_t> packet;
        const auto status = make_tmr_dispatch_packet(candidate, callable, binding, residency_address, &packet);
        if (status == InvocationStatus::StaleCallable || status == InvocationStatus::InvalidBinding)
            return PTO_RUNTIME_ERR_INVALID_STATE;
        if (status != InvocationStatus::Ok) return PTO_RUNTIME_ERR_INTERNAL;
        return runner.launch_aicpu_payload(stream, packet.data(), packet.size(), TmrKernelInvocationName, aicpu_num);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

}  // namespace simpler::tmr
