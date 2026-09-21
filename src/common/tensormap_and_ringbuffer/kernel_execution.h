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

#include "kernel_round_storage.h"
#include "kernel_execution_inputs.h"
#include "kernel_round_gate.h"
#include "task_interface/kernel_dispatch_args.h"

namespace simpler::tmr {

int execute_kernel_task(void *arg) noexcept;

struct KernelThreadView {
    int32_t execution_index;
    int32_t execution_threads;
};

// Read-only provider inputs, valid through every CPU/core consumer. Framing or
// residency rejection still carries a trusted context/control view, so an
// already-launched AICore group participates in cancellation and retirement.
// These views are not constructed by dereferencing an arbitrary packet address.
struct KernelExecutionRequest {
    ByteSpan packet{};
    KernelCallableView callable{};
    // The image span the packet names, carried so the round leader can
    // establish residency for a callable the device has not seen. Every
    // launched thread builds its own request, so only the leader may write
    // the shared table.
    uint64_t image_address{0};
    uint64_t image_bytes{0};
    int32_t callable_id{-1};
    KernelBindingView binding{};
    KernelHandshakeView handshake{};
    const int32_t *allowed_cpus{nullptr};
    int32_t execution_threads{0};
    int32_t launched_threads{0};
    int32_t admission_status{0};
};

inline int32_t invocation_dispatch_status(InvocationStatus status) noexcept {
    switch (status) {
    case InvocationStatus::Ok:
        return 0;
    case InvocationStatus::InvalidBinding:
        return static_cast<int32_t>(KernelDispatchStatus::InvalidBinding);
    default:
        return static_cast<int32_t>(KernelDispatchStatus::InvalidArgs);
    }
}

// All native task threads call this entry; reported_cpu is an observation, not
// an execution index. The provider fixes the launch group before enqueue.
// Cleanup failure poisons the gate and retains storage until fatal recovery.
int32_t
execute_kernel_round(const KernelExecutionRequest &, int32_t reported_cpu, KernelFinalStatus *out = nullptr) noexcept;

}  // namespace simpler::tmr
