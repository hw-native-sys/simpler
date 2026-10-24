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

#include "host/kernel_launch_binder.h"

namespace simpler::kernel_launch {

inline KernelLaunchResult
enqueue_kernel_launch_sequence(const KernelLaunchOps &ops, const KernelLaunchHandles &h) noexcept {
    KernelLaunchResult result;
    if (!ops.valid() || !h.valid()) {
        result.status = PTO_RUNTIME_ERR_INTERNAL;
        return result;
    }
    auto step = [&](KernelLaunchStep at, int rc) {
        if (rc == 0) return true;
        result.status = rc;
        result.failed_step = at;
        return false;
    };
    // A partial submission has no proven completion fence. The owner poisons
    // admission and retains resources until the caller establishes quiescence.
    result.enqueue_started = true;
    if (!step(KernelLaunchStep::Start, ops.record_event(ops.context, h.start, h.caller))) return result;
    if (!step(KernelLaunchStep::AicpuWait, ops.wait_event(ops.context, h.aicpu, h.start))) return result;
    if (!step(KernelLaunchStep::Clear, ops.memset_handshake(ops.context, h.aicpu))) return result;

    // AicoreStart is recorded on the AICPU stream before the AICPU launch. The
    // AICPU orchestrator spins on AICore's handshake report, so an AicoreStart
    // recorded after that launch could only fire once the AICPU task had
    // completed, which closes a cycle. Recording it first also keeps the caller
    // free of any edge to AICore: capture propagates caller to AICPU to AICore.
    if (!step(KernelLaunchStep::AicoreStart, ops.record_event(ops.context, h.aicore_start, h.aicpu))) return result;
    if (!step(KernelLaunchStep::AicoreWait, ops.wait_event(ops.context, h.aicore, h.aicore_start))) return result;
    if (!step(KernelLaunchStep::AicoreLaunch, ops.launch_aicore(ops.context, h.aicore))) return result;
    if (!step(KernelLaunchStep::AicoreDone, ops.record_event(ops.context, h.aicore_done, h.aicore))) {
        return result;
    }
    if (!step(KernelLaunchStep::AicpuLaunch, ops.launch_aicpu(ops.context, h.aicpu))) {
        return result;
    }
    // Completion/join errors stop enqueue and poison; there is no Host reset.
    if (!step(KernelLaunchStep::JoinAicore, ops.wait_event(ops.context, h.aicpu, h.aicore_done))) return result;
    if (!step(KernelLaunchStep::AicpuDone, ops.record_event(ops.context, h.aicpu_done, h.aicpu))) return result;
    if (!step(KernelLaunchStep::JoinAicpu, ops.wait_event(ops.context, h.caller, h.aicpu_done))) return result;
    if (!step(KernelLaunchStep::SerialTail, ops.record_event(ops.context, h.serial_tail, h.caller))) return result;
    result.tail_recorded = true;
    return result;
}

}  // namespace simpler::kernel_launch
