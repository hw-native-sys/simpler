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
 * PTO2 Runtime Status Helpers
 *
 * Shared error-code contract used inside the tensormap_and_ringbuffer runtime.
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_COMMON_PTO_RUNTIME_STATUS_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_COMMON_PTO_RUNTIME_STATUS_H_

#include <stdint.h>

// Orchestrator errors (1-99): detected in orchestrator thread
#define PTO2_ERROR_NONE 0  // Explicitly means "no error"; it is not an "unknown/unspecified" error code.
#define PTO2_ERROR_SCOPE_DEADLOCK 1
#define PTO2_ERROR_HEAP_RING_DEADLOCK 2
#define PTO2_ERROR_FLOW_CONTROL_DEADLOCK 3
#define PTO2_ERROR_DEP_POOL_OVERFLOW 4
#define PTO2_ERROR_INVALID_ARGS 5  // Arg construction error (invalid args)
// 6 retired: per-task fanin overflow is now reported as DEP_POOL_OVERFLOW (4).
#define PTO2_ERROR_REQUIRE_SYNC_START_INVALID 7
#define PTO2_ERROR_TENSOR_WAIT_TIMEOUT 8
#define PTO2_ERROR_EXPLICIT_ORCH_FATAL 9
#define PTO2_ERROR_SCOPE_TASKS_OVERFLOW 10  // scope_tasks buffer saturated (all rings full)
#define PTO2_ERROR_TENSORMAP_OVERFLOW 11    // tensormap entry pool wedged (last_task_alive not advancing)

// Scheduler errors (100+): detected in scheduler threads
#define PTO2_ERROR_SCHEDULER_TIMEOUT 100
#define PTO2_ERROR_ASYNC_COMPLETION_INVALID 101
#define PTO2_ERROR_ASYNC_WAIT_OVERFLOW 102
#define PTO2_ERROR_ASYNC_REGISTRATION_FAILED 103
#define PTO2_ERROR_DRAIN_ABA_TEST_NOT_EXERCISED \
    104  // SIMPLER_DRAIN_ABA_TEST: hook never observed the forced attempt change

// Sub-classification of a PTO2_ERROR_SCHEDULER_TIMEOUT (code 100). The top-level
// sched_error_code stays 100 for backward compatibility; this detail value tells
// the host which device error TYPE the AICPU no-progress watchdog observed, so a
// per-incident device-log dive becomes a glance at the host failure line.
#define PTO2_STALL_DETAIL_NONE 0             // not a timeout / no sub-class recorded
#define PTO2_STALL_DETAIL_RUNNING_STALLED 1  // S1: a task is on a core but never completes (AICore hang)
#define PTO2_STALL_DETAIL_READY_IDLE 3       // S3: all cores idle, a fanin-satisfied task exists, nothing dispatched
#define PTO2_STALL_DETAIL_DEP_DEADLOCK 4     // S4: only WAIT tasks remain, fanin never resolves (dep cycle/wiring)
#define PTO2_STALL_DETAIL_ORCH_STARVATION 5  // S5: submitted tasks done, orchestrator not done, scheduler idle
#define PTO2_STALL_DETAIL_UNKNOWN 99         // premise/bookkeeping invariant violated (accounting/corruption)

// Pure stall-priority decision: RUNNING > READY > WAIT > (orch-not-done) > unknown.
// Reduces the multi-state snapshot to one dominant sub-class. Kept free-standing
// (no scheduler state) so it is unit-testable and identical on host and device.
static inline int32_t
classify_stall_detail(int32_t cnt_running, int32_t cnt_ready, int32_t cnt_waiting, int32_t orch_done) {
    if (cnt_running > 0) return PTO2_STALL_DETAIL_RUNNING_STALLED;
    if (cnt_ready > 0) return PTO2_STALL_DETAIL_READY_IDLE;
    if (cnt_waiting > 0) return PTO2_STALL_DETAIL_DEP_DEADLOCK;
    if (!orch_done) return PTO2_STALL_DETAIL_ORCH_STARVATION;
    return PTO2_STALL_DETAIL_UNKNOWN;
}

// Human-readable label for a PTO2_STALL_DETAIL_* value (host failure line).
static inline const char *stall_detail_name(int32_t detail) {
    switch (detail) {
    case PTO2_STALL_DETAIL_RUNNING_STALLED:
        return "S1:running-stalled";
    case PTO2_STALL_DETAIL_READY_IDLE:
        return "S3:ready-but-all-idle";
    case PTO2_STALL_DETAIL_DEP_DEADLOCK:
        return "S4:dependency-deadlock";
    case PTO2_STALL_DETAIL_ORCH_STARVATION:
        return "S5:orchestrator-starvation";
    case PTO2_STALL_DETAIL_UNKNOWN:
        return "unknown:accounting/corruption";
    case PTO2_STALL_DETAIL_NONE:
        return "none";
    default:
        return "invalid";
    }
}

static inline int32_t runtime_status_from_error_codes(int32_t orch_error_code, int32_t sched_error_code) {
    if (orch_error_code != PTO2_ERROR_NONE) {
        return orch_error_code < 0 ? orch_error_code : -orch_error_code;
    }
    if (sched_error_code != PTO2_ERROR_NONE) {
        return sched_error_code < 0 ? sched_error_code : -sched_error_code;
    }
    return 0;
}

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_COMMON_PTO_RUNTIME_STATUS_H_
