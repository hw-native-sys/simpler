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

/**
 * Runtime Status Codes
 *
 * Shared error-code contract used inside the tensormap_and_ringbuffer runtime.
 */

#pragma once

#include "../../../../common/runtime_status/error_codes.h"

// Sub-classification of a SIMPLER_ERROR_SCHEDULER_TIMEOUT (code 100). The top-level
// sched_error_code stays 100 for backward compatibility; this detail value tells
// the host which device error TYPE the AICPU no-progress watchdog observed, so a
// per-incident device-log dive becomes a glance at the host failure line.
#define SIMPLER_STALL_DETAIL_NONE 0             // not a timeout / no sub-class recorded
#define SIMPLER_STALL_DETAIL_RUNNING_STALLED 1  // S1: a task is on a core but never completes (AICore hang)
#define SIMPLER_STALL_DETAIL_READY_IDLE 3       // S3: all cores idle, a dispatch-ready task exists
#define SIMPLER_STALL_DETAIL_DEP_DEADLOCK 4     // S4: only WAIT tasks remain, fanin never resolves (dep cycle/wiring)
#define SIMPLER_STALL_DETAIL_ORCH_STARVATION 5  // S5: submitted tasks done, orchestrator not done, scheduler idle
#define SIMPLER_STALL_DETAIL_ASYNC_WAIT 6       // S6: AICore work finished, deferred completion pending
#define SIMPLER_STALL_DETAIL_UNKNOWN 99         // premise/bookkeeping invariant violated (accounting/corruption)

// Pure stall-priority decision: RUNNING > READY > ASYNC_WAIT > WAIT > (orch-not-done) > unknown.
// Reduces the multi-state snapshot to one dominant sub-class. Kept free-standing
// (no scheduler state) so it is unit-testable and identical on host and device.
static inline int32_t classify_stall_detail(
    int32_t cnt_running, int32_t cnt_ready, int32_t cnt_waiting, int32_t cnt_async_wait, int32_t orch_done
) {
    if (cnt_running > 0) return SIMPLER_STALL_DETAIL_RUNNING_STALLED;
    if (cnt_ready > 0) return SIMPLER_STALL_DETAIL_READY_IDLE;
    if (cnt_async_wait > 0) return SIMPLER_STALL_DETAIL_ASYNC_WAIT;
    if (cnt_waiting > 0) return SIMPLER_STALL_DETAIL_DEP_DEADLOCK;
    if (!orch_done) return SIMPLER_STALL_DETAIL_ORCH_STARVATION;
    return SIMPLER_STALL_DETAIL_UNKNOWN;
}

// Human-readable label for a SIMPLER_STALL_DETAIL_* value (host failure line).
static inline const char *stall_detail_name(int32_t detail) {
    switch (detail) {
    case SIMPLER_STALL_DETAIL_RUNNING_STALLED:
        return "S1:running-stalled";
    case SIMPLER_STALL_DETAIL_READY_IDLE:
        return "S3:ready-but-all-idle";
    case SIMPLER_STALL_DETAIL_DEP_DEADLOCK:
        return "S4:dependency-deadlock";
    case SIMPLER_STALL_DETAIL_ORCH_STARVATION:
        return "S5:orchestrator-starvation";
    case SIMPLER_STALL_DETAIL_ASYNC_WAIT:
        return "S6:async-completion-wait";
    case SIMPLER_STALL_DETAIL_UNKNOWN:
        return "unknown:accounting/corruption";
    case SIMPLER_STALL_DETAIL_NONE:
        return "none";
    default:
        return "invalid";
    }
}
