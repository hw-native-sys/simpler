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

#include <stdint.h>

// Public runtime status values. A recorded orchestration or scheduler code reaches
// the caller as its negative. Host interface statuses occupy a separate band.

// Orchestration errors (1-99).
#define SIMPLER_ERROR_NONE 0  // Explicitly means "no error"; it is not an "unknown/unspecified" error code.
#define SIMPLER_ERROR_SCOPE_DEADLOCK 1
#define SIMPLER_ERROR_HEAP_RING_DEADLOCK 2
#define SIMPLER_ERROR_FLOW_CONTROL_DEADLOCK 3
#define SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED 4
#define SIMPLER_ERROR_INVALID_ARGS 5  // Arg construction error (invalid args)
// 6 retired: per-task fanin overflow is reported as FANIN_CAPACITY_EXCEEDED (4).
#define SIMPLER_ERROR_REQUIRE_SYNC_START_INVALID 7
#define SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT 8  // Raised by tensormap_and_ringbuffer
#define SIMPLER_ERROR_EXPLICIT_ORCH_FATAL 9
#define SIMPLER_ERROR_SCOPE_TASKS_OVERFLOW 10  // Raised by tensormap_and_ringbuffer
#define SIMPLER_ERROR_TENSORMAP_OVERFLOW 11    // tensormap cannot register another output
// A submission needs more of a fixed capacity than the run has: task slots, heap bytes,
// the compact argument pools, or the Definition cache.
#define SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED 12

// Scheduler errors (100+), including capacity checks performed during bind.
#define SIMPLER_ERROR_SCHEDULER_TIMEOUT 100
#define SIMPLER_ERROR_ASYNC_COMPLETION_INVALID 101
#define SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW 102
#define SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED 103
// Required ready queue capacity exceeded, or enqueue found no free slot.
#define SIMPLER_ERROR_READY_QUEUE_OVERFLOW 104

// Only A2/A3 tensormap_and_ringbuffer exposes the kernel-validation macros.
#define SIMPLER_A2A3_TMR_KERNEL_INVALID_OUTPUT 105
#define SIMPLER_A2A3_TMR_KERNEL_INDEX_OUT_OF_RANGE 106
#define SIMPLER_A2A3_TMR_KERNEL_NON_FINITE_VALUE 107

enum {
    /* Ceiling on a device-latched code, and therefore the floor of the latched
       band once negated. */
    PTO_RUNTIME_LATCHED_CODE_MAX = 999,
    /* Highest host-side code; the band descends from here. */
    PTO_RUNTIME_ERR_BASE = -(PTO_RUNTIME_LATCHED_CODE_MAX + 1),
    /* A host-side failure whose diagnosis is the log line that preceded it. */
    PTO_RUNTIME_ERR_INTERNAL = PTO_RUNTIME_ERR_BASE,
    /* The request names a capability this platform/runtime does not implement. */
    PTO_RUNTIME_ERR_UNSUPPORTED = PTO_RUNTIME_ERR_BASE - 1,
    PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE = PTO_RUNTIME_ERR_BASE - 2,
    /* The call is structurally valid but conflicts with the context's
       execution mode or lifecycle (e.g. launch before kernel initialization). */
    PTO_RUNTIME_ERR_INVALID_STATE = PTO_RUNTIME_ERR_BASE - 3,
    /* A caller supplied an invalid pointer, id, size, or other argument. */
    PTO_RUNTIME_ERR_INVALID_ARGUMENT = PTO_RUNTIME_ERR_BASE - 4,
    /* BASE - 5 is reserved for diagnostic collector capacity refusal. */
    /* A prepare could not take the device resources it needs now, and a
       consumer that still holds some of them has a retirement this context will
       reach. The plan is kept and retried once that consumer finishes, so this
       is a wait rather than a failure and the caller must not discard the run. */
    SIMPLER_RUNTIME_ERR_RESOURCE_WAIT = PTO_RUNTIME_ERR_BASE - 6,
    /* A prepare could not take the device resources it needs and no consumer
       remains whose retirement could supply them: either none ever held any, or
       this plan has already waited for every one that did. Terminal. */
    SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE = PTO_RUNTIME_ERR_BASE - 7,
    /* Reserved for simpler_wait_run: this wait did not observe the result.
       The run remains live and may be waited/finalized again. Never store this
       operation result as the execution's terminal error. */
    SIMPLER_NATIVE_RUN_WAIT_INCOMPLETE = PTO_RUNTIME_ERR_BASE - 8,
    /* Host register/unregister succeeded, but restoring the caller context failed. */
    SIMPLER_HOST_DDR_CONTEXT_RESTORE_FAILED = PTO_RUNTIME_ERR_BASE - 9,
};

/**
 * A launch result that is not a failure of the run.
 *
 * Every diagnostic output slot or terminal bank some enabled owner needs is
 * still held by a run ahead. Nothing was submitted and nothing was admitted:
 * the run keeps its prepared state and may be launched again, unchanged, once
 * `simpler_diagnostic_credit_generation_ctx` advances. A caller that treats it
 * as a failure turns a run that is merely waiting into one that is lost.
 *
 * It takes a value in the host-side band so it cannot collide with a latched
 * device code, and sits in its own enum because the band's other members are
 * externally-contracted legacy names this one does not join.
 */
enum {
    SIMPLER_RUNTIME_ERR_DIAGNOSTIC_CAPACITY_FULL = PTO_RUNTIME_ERR_BASE - 5,
};

/** Return values from simpler_poll_run(). */
enum {
    SIMPLER_NATIVE_RUN_POLL_ERROR = PTO_RUNTIME_ERR_INTERNAL,
    SIMPLER_NATIVE_RUN_POLL_NOT_READY = 0,
    SIMPLER_NATIVE_RUN_POLL_COMPLETE = 1,
};

// Both records are retained. The orchestration code has priority for the caller.
static inline int32_t latched_error_code(int32_t orch_error_code, int32_t sched_error_code) {
    return orch_error_code != SIMPLER_ERROR_NONE ? orch_error_code : sched_error_code;
}

static inline int32_t runtime_status_from_error_code(int32_t code) { return code > 0 ? -code : code; }

static inline int32_t runtime_status_from_error_codes(int32_t orch_error_code, int32_t sched_error_code) {
    return runtime_status_from_error_code(latched_error_code(orch_error_code, sched_error_code));
}
