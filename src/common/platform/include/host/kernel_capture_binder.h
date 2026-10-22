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

#include <cstddef>

#include "native_run_execution.h"

struct KernelCaptureHandles {
    void *caller{nullptr};
    void *aicpu{nullptr};
    void *aicore{nullptr};
    void *start{nullptr};
    void *fork{nullptr};
    void *core_done{nullptr};
    void *cpu_done{nullptr};
    void *tail{nullptr};

    bool valid() const {
        const void *streams[] = {caller, aicpu, aicore};
        const void *events[] = {start, fork, core_done, cpu_done, tail};
        for (size_t i = 0; i < 3; ++i) {
            if (streams[i] == nullptr) return false;
            for (size_t j = 0; j < i; ++j)
                if (streams[i] == streams[j]) return false;
        }
        for (size_t i = 0; i < 5; ++i) {
            if (events[i] == nullptr) return false;
            for (size_t j = 0; j < i; ++j)
                if (events[i] == events[j]) return false;
        }
        return true;
    }
};

struct KernelCaptureOps {
    void *context{nullptr};
    int (*record)(void *, void *event, void *stream){nullptr};
    int (*wait)(void *, void *stream, void *event){nullptr};
    int (*admit)(void *, void *stream){nullptr};
    int (*launch_core)(void *, void *stream){nullptr};
    int (*launch_cpu)(void *, void *stream){nullptr};

    bool valid() const { return record && wait && admit && launch_core && launch_cpu; }
};

struct KernelCaptureRecordResult {
    int rc{PTO_RUNTIME_ERR_INVALID_ARGUMENT};
    LaunchProgress progress{LaunchProgress::NotStarted};
};

// All five events are recorded in this capture. Admission precedes the fork,
// and the fork precedes the scheduling task that needs AICore's handshake.
// No unconditional clear can run ahead of admission or after a failed replay.
inline KernelCaptureRecordResult record_kernel_capture(
    const KernelCaptureHandles &handles, const KernelCaptureOps &ops, LaunchProgressSink &sink
) noexcept {
    KernelCaptureRecordResult result;
    if (!handles.valid() || !ops.valid()) return result;
    sink.mark_submitted();
    result.progress = LaunchProgress::Partial;
    try {
        if ((result.rc = ops.record(ops.context, handles.start, handles.caller)) != 0) return result;
        if ((result.rc = ops.wait(ops.context, handles.aicpu, handles.start)) != 0) return result;
        if ((result.rc = ops.admit(ops.context, handles.aicpu)) != 0) return result;
        if ((result.rc = ops.record(ops.context, handles.fork, handles.aicpu)) != 0) return result;
        if ((result.rc = ops.wait(ops.context, handles.aicore, handles.fork)) != 0) return result;
        if ((result.rc = ops.launch_core(ops.context, handles.aicore)) != 0) return result;
        if ((result.rc = ops.record(ops.context, handles.core_done, handles.aicore)) != 0) return result;
        if ((result.rc = ops.launch_cpu(ops.context, handles.aicpu)) != 0) return result;
        if ((result.rc = ops.wait(ops.context, handles.aicpu, handles.core_done)) != 0) return result;
        if ((result.rc = ops.record(ops.context, handles.cpu_done, handles.aicpu)) != 0) return result;
        if ((result.rc = ops.wait(ops.context, handles.caller, handles.cpu_done)) != 0) return result;
        if ((result.rc = ops.record(ops.context, handles.tail, handles.caller)) != 0) return result;
        result.progress = LaunchProgress::Complete;
    } catch (...) {
        // Launch callbacks can allocate host storage after accepting earlier
        // submissions. Their progress must reach the owner on this exit too.
        result.rc = PTO_RUNTIME_ERR_INTERNAL;
    }
    return result;
}
