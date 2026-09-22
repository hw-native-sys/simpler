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
           aicore_scheduler_runtime_mode_is_resident(runtime->dev.scheduler_bootstrap.runtime_mode);
}

inline bool aicore_scheduler_explicit_legacy_enabled(const Runtime *runtime) {
    return runtime != nullptr && runtime->get_worker_count() > 0 &&
           aicore_scheduler_runtime_mode_is_explicit_legacy(runtime->dev.scheduler_bootstrap.runtime_mode);
}

/**
 * Whether a legacy run reached the end of the path its terminal record may call
 * a success — what `LegacyAicpuExecutor::snapshot_run_terminal` passes to
 * `run_terminal_select` as `normal_path_completed`.
 *
 * Two conditions, and the first is this producer's alone. `aicpu_execute`
 * rejects a legacy run whose mode does not say legacy was chosen, and that
 * rejection reaches the host only as the kernel's return; a record calling such
 * a run Ok would contradict it. The second is the shared one: a participant
 * that never dispatched withholds its claim, so a short tally means some thread
 * cannot vouch for the path.
 *
 * Neither condition suppresses a failure — `run_terminal_select` reports a
 * header or participant error whatever this answers.
 */
inline bool aicore_legacy_run_completed_audited_path(const Runtime *runtime, int32_t claims, int32_t participants) {
    return aicore_scheduler_explicit_legacy_enabled(runtime) && claims == participants;
}

// Worker 0's context sits at the published base, so the bootstrap context is the
// base itself. A zero base is "no resident scheduler state", the same verdict a
// zero handshake task carried before.
inline SchedulerWorkerContext *aicore_scheduler_bootstrap_context(Runtime *runtime) {
    if (!aicore_scheduler_runtime_enabled(runtime) || runtime->dev.scheduler_bootstrap.worker_context_base == 0) {
        return nullptr;
    }
    return reinterpret_cast<SchedulerWorkerContext *>(runtime->dev.scheduler_bootstrap.worker_context_base);
}

/**
 * Whether this run may publish a context and READY to a worker.
 *
 * Two per-worker conditions. A null bootstrap context means this run has no
 * resident scheduler state to hand over: the host publishes
 * `scheduler_bootstrap.worker_context_base` before launch, so a null one is
 * `aicore_scheduler_bootstrap_context`'s mode-and-base verdict, not a report
 * that configuration finished. `worker_reg_addr` comes from `cores_`, which
 * `pre_handshake_init` clears and only an accepted report fills, so a zero one
 * means this worker did not report to this run — and on a stamped run that is
 * exactly what the epoch check withheld. Replying anyway would tell a core to
 * read a context it never asked for.
 *
 * Successful configuration is a separate outer gate this predicate does not
 * observe: `AicpuExecutor::init` calls `publish_context_partition` only after
 * `hs_config_done_` with `init_failed_` clear.
 */
inline bool aicore_context_reply_permitted(const SchedulerWorkerContext *bootstrap_context, uint64_t worker_reg_addr) {
    return bootstrap_context != nullptr && worker_reg_addr != 0;
}

inline void *aicore_scheduler_state_base(SchedulerWorkerContext *context) {
    return context == nullptr ? nullptr : reinterpret_cast<void *>(context->scheduler_state_base_address);
}

inline SchedulerRunControl *aicore_scheduler_run_control(SchedulerWorkerContext *context) {
    void *state_base = aicore_scheduler_state_base(context);
    return state_base == nullptr ? nullptr :
                                   scheduler_state_at<SchedulerRunControl>(state_base, context->run_control_offset);
}
