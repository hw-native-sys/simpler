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
#define PTO2_ERROR_INVALID_ARGS 5         // Arg construction error (invalid args)
#define PTO2_ERROR_DEPENDENCY_OVERFLOW 6  // Too many unique fanin dependencies for one task
#define PTO2_ERROR_REQUIRE_SYNC_START_INVALID 7
#define PTO2_ERROR_TENSOR_WAIT_TIMEOUT 8
#define PTO2_ERROR_EXPLICIT_ORCH_FATAL 9

// Scheduler errors (100+): detected in scheduler threads
#define PTO2_ERROR_SCHEDULER_TIMEOUT 100
#define PTO2_ERROR_ASYNC_COMPLETION_INVALID 101
#define PTO2_ERROR_ASYNC_WAIT_OVERFLOW 102
#define PTO2_ERROR_ASYNC_REGISTRATION_FAILED 103

static inline int32_t pto2_runtime_status_from_error_codes(int32_t orch_error_code, int32_t sched_error_code) {
    if (orch_error_code != PTO2_ERROR_NONE) {
        return orch_error_code < 0 ? orch_error_code : -orch_error_code;
    }
    if (sched_error_code != PTO2_ERROR_NONE) {
        return sched_error_code < 0 ? sched_error_code : -sched_error_code;
    }
    return 0;
}

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_COMMON_PTO_RUNTIME_STATUS_H_
