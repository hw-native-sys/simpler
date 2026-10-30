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

#include "aicpu/cache_maintenance.h"
#include "aicpu/device_run_result_aicpu.h"
#include "common/native_execution_control.h"

// Backend preparation may mark an existing device scheduler as stopped. It
// cannot dispatch work: the common stream fork has not released either kernel.
template <typename Runtime, typename PrepareBackend>
inline int native_execution_prepare_device(const NativeExecutionPrepareArgs &args, PrepareBackend prepare_backend) {
    if (args.control == 0 || args.runtime_address == 0 || args.result_address == 0 || args.launched_callbacks == 0 ||
        args.launched_callbacks > PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH || args.worker_count > RUNTIME_MAX_WORKER)
        return -1;
    auto &control = *reinterpret_cast<NativeExecutionControl *>(args.control);
    auto &runtime = *reinterpret_cast<Runtime *>(args.runtime_address);
    // Host publication precedes this command. Discard the previous descriptor
    // before modifying worker cache lines or following freshly uploaded pointers.
    cache_invalidate_range(&runtime, sizeof(runtime.dev));
    cache_invalidate_range(&control, sizeof(control));
    if (control.sequence != 0 && control.result_address != 0)
        cache_invalidate_range(reinterpret_cast<void *>(control.result_address), sizeof(DeviceRunResultRegion));
    if (args.first_error_address != 0)
        cache_invalidate_range(reinterpret_cast<void *>(args.first_error_address), sizeof(NativeExecutionFirstError));
    const int rc = native_execution_prepare(control, args);
    if (rc == 0) {
        for (uint32_t i = 0; i < args.worker_count; ++i) {
            runtime.dev.workers[i].aicore_done = 0;
            runtime.dev.workers[i].aicpu_ready = 0;
        }
        cache_flush_range(runtime.dev.workers, args.worker_count * sizeof(runtime.dev.workers[0]));
        const int backend_rc = prepare_backend(&runtime, control.decision, control.first_error);
        if (backend_rc != 0) {
            native_execution_latch_first(control, backend_rc);
            control.decision = NativeExecutionDecision::Skip;
        }
        cache_flush_range(reinterpret_cast<void *>(args.result_address), sizeof(DeviceRunResultRegion));
    }
    cache_flush_range(&control, sizeof(control));
    if (args.first_error_address != 0)
        cache_flush_range(reinterpret_cast<void *>(args.first_error_address), sizeof(NativeExecutionFirstError));
    return rc;
}

inline int native_execution_complete_device(NativeExecutionControl &control, int32_t status) {
    bool final_callback = false;
    const int rc = native_execution_finish(
        control, status,
        [&] {
            final_callback = true;
            cache_invalidate_range(reinterpret_cast<void *>(control.result_address), sizeof(DeviceRunResultRegion));
        },
        [&](int32_t selected) {
            (void)aicpu_publish_run_terminal(
                control.result_address, control.native_epoch, DeviceRunVerdict::Error, selected, control.failure_source,
                nullptr, 0
            );
        }
    );
    // Only the last callback owns the final cache publication. Earlier writers
    // have released their atomic contribution before the last acquire.
    if (final_callback) {
        cache_flush_range(reinterpret_cast<void *>(control.result_address), sizeof(DeviceRunResultRegion));
        cache_flush_range(&control, sizeof(control));
        if (control.first_error_address != 0)
            cache_flush_range(reinterpret_cast<void *>(control.first_error_address), sizeof(NativeExecutionFirstError));
    }
    return rc;
}
