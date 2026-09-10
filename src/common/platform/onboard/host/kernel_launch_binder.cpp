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
#include "host/kernel_launch_binder.h"

#include "kernel_launch_sequence.h"

namespace simpler::kernel_launch {

KernelLaunchResult launch_bound_kernel(
    const KernelInvocationBinding &binding, void *caller_stream, const KernelLaunchGateOps &gate,
    const KernelLaunchOps &ops
) {
    KernelLaunchResult result;
    result.status = PTO_RUNTIME_ERR_INTERNAL;
    if (!gate.valid() || !ops.valid() || !caller_stream || !binding.packet || !binding.packet_bytes) return result;
    KernelLaunchAdmission admission;
    result.status = gate.acquire(gate.context, binding, caller_stream, &admission);
    if (result.status != 0) return result;
    auto finish = [&](KernelLaunchResult out) {
        gate.finish(gate.context, out);
        return out;
    };
    // The caller argument is authoritative; the owner cannot replace it.
    admission.handles.caller = caller_stream;
    if (!admission.handles.valid()) {
        result.status = PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
        return finish(result);
    }
    if (admission.previous_caller_identity != 0 &&
        admission.previous_caller_identity != reinterpret_cast<uintptr_t>(caller_stream)) {
        bool complete = false;
        result.failed_step = KernelLaunchStep::QueryTail;
        result.status = gate.query_tail(gate.context, admission.handles.serial_tail, &complete);
        if (result.status != 0) return finish(result);
        if (!complete) {
            result.status = PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
            return finish(result);
        }
    }
    return finish(enqueue_kernel_launch_sequence(ops, admission.handles));
}

}  // namespace simpler::kernel_launch
