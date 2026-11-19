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
 * Shared error-code contract used inside the host_build_graph runtime.
 */

#pragma once

#include <stdint.h>

// A latched code reaches the host as its own value negated, so the two families
// below land in -1..-99 and -100..-PTO_RUNTIME_LATCHED_CODE_MAX there. That
// ceiling lives in src/common/worker/runtime_c_api.h, and the host-side C
// API band begins below its negation — so the latched range is bounded, not
// open-ended downwards.

// Orchestrator errors (1-99): detected in orchestrator thread
#define SIMPLER_ERROR_NONE 0  // Explicitly means "no error"; it is not an "unknown/unspecified" error code.
#define SIMPLER_ERROR_SCOPE_DEADLOCK 1
#define SIMPLER_ERROR_HEAP_RING_DEADLOCK 2
#define SIMPLER_ERROR_FLOW_CONTROL_DEADLOCK 3
#define SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED 4
#define SIMPLER_ERROR_INVALID_ARGS 5  // Arg construction error (invalid args)
// 6 retired: per-task fanin overflow is reported as FANIN_CAPACITY_EXCEEDED (4).
#define SIMPLER_ERROR_REQUIRE_SYNC_START_INVALID 7
#define SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT 8  // Not raised here; the number stays reserved across runtimes
#define SIMPLER_ERROR_EXPLICIT_ORCH_FATAL 9
#define SIMPLER_ERROR_SCOPE_TASKS_OVERFLOW 10  // Not raised here; the number stays reserved across runtimes
#define SIMPLER_ERROR_TENSORMAP_OVERFLOW 11    // graph registers more outputs than the tensormap entry pool holds
// A submission needs more of a fixed capacity than the run has: task slots, heap bytes,
// the compact argument pools, or the Definition cache.
#define SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED 12

// Scheduler errors (100+): detected in scheduler threads
#define SIMPLER_ERROR_SCHEDULER_TIMEOUT 100
#define SIMPLER_ERROR_ASYNC_COMPLETION_INVALID 101
#define SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW 102
#define SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED 103
// push into a ready queue found no free slot (full, or window > capacity)
#define SIMPLER_ERROR_READY_QUEUE_OVERFLOW 104

// Maps one latched code onto the status the caller sees. Both families negate the
// same way, and a single call site only ever holds one of them: the orchestrator
// runs to completion on the host before the device starts, so an orchestrator code
// is latched during the bind and a scheduler code only after it.
static inline int32_t runtime_status_from_error_code(int32_t error_code) {
    if (error_code != SIMPLER_ERROR_NONE) {
        return error_code < 0 ? error_code : -error_code;
    }
    return 0;
}
