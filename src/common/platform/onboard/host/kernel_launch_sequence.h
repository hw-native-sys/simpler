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
    // Compensation never replaces the original error. A failed cancel or retry
    // leaves no provable caller join; external quiescence is then mandatory.
    auto compensate = [&](bool retry_core_done, bool join_cpu_wait) {
        auto cleanup = [&](int rc) {
            if (rc == 0) return true;
            result.cleanup_status = rc;
            return false;
        };
        if (!cleanup(ops.cancel_waiting_aicore(ops.context, h.caller))) return;
        if (retry_core_done && !cleanup(ops.record_event(ops.context, h.aicore_done, h.aicore))) return;
        if (join_cpu_wait) {
            if (!cleanup(ops.record_event(ops.context, h.aicpu_done, h.aicpu))) return;
            if (!cleanup(ops.wait_event(ops.context, h.caller, h.aicpu_done))) return;
        }
        if (!cleanup(ops.wait_event(ops.context, h.caller, h.aicore_done))) return;
        if (!cleanup(ops.record_event(ops.context, h.serial_tail, h.caller))) return;
        result.tail_recorded = true;
    };
    result.enqueue_started = true;
    if (h.consume_prepare_tail &&
        !step(KernelLaunchStep::PrepareWait, ops.wait_event(ops.context, h.caller, h.prepare_tail)))
        return result;
    if (!step(KernelLaunchStep::Clear, ops.memset_handshake(ops.context, h.caller))) return result;
    if (!step(KernelLaunchStep::Start, ops.record_event(ops.context, h.start, h.caller))) return result;

    // AICore-first avoids the scheduler cycle when an AICPU startup waiter
    // blocks a later AICore SQE. It also checks the hidden branch's submissions
    // before admitting AICPU, allowing cancel while no AICPU writes handshake.
    // Both device branches remain gated by the caller's Start event.
    if (!step(KernelLaunchStep::AicoreWait, ops.wait_event(ops.context, h.aicore, h.start))) return result;
    if (!step(KernelLaunchStep::AicoreLaunch, ops.launch_aicore(ops.context, h.aicore))) return result;
    if (!step(KernelLaunchStep::AicoreDone, ops.record_event(ops.context, h.aicore_done, h.aicore))) {
        compensate(true, false);
        return result;
    }
    if (!step(KernelLaunchStep::AicpuWait, ops.wait_event(ops.context, h.aicpu, h.start))) {
        compensate(false, false);
        return result;
    }
    if (!step(KernelLaunchStep::AicpuLaunch, ops.launch_aicpu(ops.context, h.aicpu))) {
        compensate(false, true);
        return result;
    }
    // Once AICPU is resident, Host cancel could overwrite its live handshake.
    // Completion/join errors stop enqueue and poison; there is no Host reset.
    if (!step(KernelLaunchStep::AicpuDone, ops.record_event(ops.context, h.aicpu_done, h.aicpu))) return result;
    if (!step(KernelLaunchStep::JoinAicpu, ops.wait_event(ops.context, h.caller, h.aicpu_done))) return result;
    if (!step(KernelLaunchStep::JoinAicore, ops.wait_event(ops.context, h.caller, h.aicore_done))) return result;
    if (!step(KernelLaunchStep::SerialTail, ops.record_event(ops.context, h.serial_tail, h.caller))) return result;
    result.tail_recorded = true;
    return result;
}

}  // namespace simpler::kernel_launch
