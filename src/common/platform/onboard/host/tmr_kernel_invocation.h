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

#include "aicpu_loader/host/load_aicpu_op.h"
#include "host/tmr_dispatch_packet.h"
#include "worker/runtime_c_api.h"
#include "worker/tmr_kernel_invocation.h"

class DeviceRunnerBase;

namespace simpler::tmr {

inline constexpr char TmrKernelInvocationName[] = "simpler_aicpu_kernel_exec";

// The runner owns the initialized loader. The same borrowing and complete
// enqueue/commit contract below applies to this production transport adapter.
int enqueue_tmr_invocation_aicpu(
    DeviceRunnerBase &runner, void *stream, int32_t aicpu_num, const TmrEncodingCandidate &candidate,
    const PreparedInvocationView &callable, const TmrExecutionBindingView &binding, uint64_t residency_address
) noexcept;

// Called by the owner's KernelLaunchOps::launch_aicpu callback. The owner
// holds candidate, callable and binding live through this call and commits
// its encoding cache only after the complete enqueue sequence succeeds.
// stream is the owner's dedicated AICPU stream, not the borrowed caller
// stream. The owner establishes event ordering outside this transport call.
// The loader must have registered the kernel-mode entry, not the program
// KernelArgs entry. CPU transport copy/lifetime support is a platform
// integration precondition, verified with the native snapshot probe.
// residency_address is the descriptor from the issuing cache's resolve(),
// borrowed under the same owner protection as callable and execution binding.
inline int enqueue_tmr_invocation_aicpu(
    host::LoadAicpuOp &loader, void *stream, int32_t aicpu_num, const TmrEncodingCandidate &candidate,
    const PreparedInvocationView &callable, const TmrExecutionBindingView &binding, uint64_t residency_address
) noexcept {
    if (stream == nullptr || aicpu_num <= 0) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        std::vector<uint8_t> packet;
        const auto status = make_tmr_dispatch_packet(candidate, callable, binding, residency_address, &packet);
        if (status == InvocationStatus::StaleCallable || status == InvocationStatus::InvalidBinding)
            return PTO_RUNTIME_ERR_INVALID_STATE;
        if (status != InvocationStatus::Ok) return PTO_RUNTIME_ERR_INTERNAL;
        return loader.LaunchBuiltInOp(stream, packet.data(), packet.size(), aicpu_num, TmrKernelInvocationName);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

}  // namespace simpler::tmr
