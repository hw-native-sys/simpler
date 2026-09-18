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

#include "runtime.h"
#include "scheduler/scheduler_types.h"

inline bool aicore_scheduler_runtime_mode_is_resident(uint32_t mode) {
    return mode == SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING || mode == SCHEDULER_RUNTIME_MODE_RESIDENT_READY;
}

inline bool aicore_scheduler_runtime_mode_is_explicit_legacy(uint32_t mode) {
    return mode == SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH || mode == SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE;
}

inline bool aicore_scheduler_runtime_enabled(const Runtime *runtime) {
    return runtime != nullptr && runtime->get_worker_count() > 0 &&
           aicore_scheduler_runtime_mode_is_resident(runtime->dev.workers[0].aicpu_ready);
}

inline bool aicore_scheduler_explicit_legacy_enabled(const Runtime *runtime) {
    return runtime != nullptr && runtime->get_worker_count() > 0 &&
           aicore_scheduler_runtime_mode_is_explicit_legacy(runtime->dev.workers[0].aicpu_ready);
}

inline SchedulerWorkerContext *aicore_scheduler_bootstrap_context(Runtime *runtime) {
    if (!aicore_scheduler_runtime_enabled(runtime) || runtime->dev.workers[0].task == 0) return nullptr;
    return reinterpret_cast<SchedulerWorkerContext *>(runtime->dev.workers[0].task);
}

inline void *aicore_scheduler_state_base(SchedulerWorkerContext *context) {
    return context == nullptr ? nullptr : reinterpret_cast<void *>(context->scheduler_state_base_address);
}

inline SchedulerRunControl *aicore_scheduler_run_control(SchedulerWorkerContext *context) {
    void *state_base = aicore_scheduler_state_base(context);
    return state_base == nullptr ? nullptr :
                                   scheduler_state_at<SchedulerRunControl>(state_base, context->run_control_offset);
}
