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
 * Runtime Error-Code Names and Triage Hints
 *
 * Turns a latched PTO2_ERROR_* code into text for the host failure line, so a caller
 * reading the log does not have to look the number up. Same role for the error code
 * that stall_detail_name() already plays for the stall sub-class.
 *
 * The switches key off the PTO2_ERROR_* macros rather than literals, so the code values
 * live in exactly one place (the per-runtime pto_runtime_status.h). Include this header
 * *after* that one; the #error below enforces it.
 *
 * Full triage steps per code: docs/troubleshooting/device-error-codes.md
 */

#ifndef SRC_COMMON_RUNTIME_STATUS_ERROR_NAMES_H_
#define SRC_COMMON_RUNTIME_STATUS_ERROR_NAMES_H_

#include <stdint.h>

#ifndef PTO2_ERROR_SCOPE_DEADLOCK
#error "include the runtime's pto_runtime_status.h before error_names.h"
#endif

// Symbolic name of a latched PTO2_ERROR_* code. "unknown" for a code this table does not
// cover -- callers use that to suppress the annotation rather than print a misleading one.
static inline const char *error_name(int32_t code) {
    switch (code) {
    case PTO2_ERROR_NONE:
        return "none";
    case PTO2_ERROR_SCOPE_DEADLOCK:
        return "SCOPE_DEADLOCK";
    case PTO2_ERROR_HEAP_RING_DEADLOCK:
        return "HEAP_RING_DEADLOCK";
    case PTO2_ERROR_FLOW_CONTROL_DEADLOCK:
        return "FLOW_CONTROL_DEADLOCK";
    case PTO2_ERROR_DEP_POOL_OVERFLOW:
        return "DEP_POOL_OVERFLOW";
    case PTO2_ERROR_INVALID_ARGS:
        return "INVALID_ARGS";
#ifdef PTO2_ERROR_DEPENDENCY_OVERFLOW
    case PTO2_ERROR_DEPENDENCY_OVERFLOW:
        return "DEPENDENCY_OVERFLOW (retired; now reported as DEP_POOL_OVERFLOW)";
#endif
    case PTO2_ERROR_REQUIRE_SYNC_START_INVALID:
        return "REQUIRE_SYNC_START_INVALID";
    case PTO2_ERROR_TENSOR_WAIT_TIMEOUT:
        return "TENSOR_WAIT_TIMEOUT";
    case PTO2_ERROR_EXPLICIT_ORCH_FATAL:
        return "EXPLICIT_ORCH_FATAL";
    case PTO2_ERROR_SCOPE_TASKS_OVERFLOW:
        return "SCOPE_TASKS_OVERFLOW";
    case PTO2_ERROR_TENSORMAP_OVERFLOW:
        return "TENSORMAP_OVERFLOW";
    case PTO2_ERROR_SCHEDULER_TIMEOUT:
        return "SCHEDULER_TIMEOUT";
    case PTO2_ERROR_ASYNC_COMPLETION_INVALID:
        return "ASYNC_COMPLETION_INVALID";
    case PTO2_ERROR_ASYNC_WAIT_OVERFLOW:
        return "ASYNC_WAIT_OVERFLOW";
    case PTO2_ERROR_ASYNC_REGISTRATION_FAILED:
        return "ASYNC_REGISTRATION_FAILED";
    default:
        return "unknown";
    }
}

// One-sentence meaning of a latched PTO2_ERROR_* code. Empty string for an uncovered code.
static inline const char *error_desc(int32_t code) {
    switch (code) {
    case PTO2_ERROR_NONE:
        return "no error";
    case PTO2_ERROR_SCOPE_DEADLOCK:
        return "tasks submitted in one scope reached the ring task-window cap; their fanout "
               "references are only released at scope_end, so no slot can be reclaimed";
    case PTO2_ERROR_HEAP_RING_DEADLOCK:
        return "the ring task allocator ran out of both task slots and heap bytes, so no further "
               "task can be admitted";
    case PTO2_ERROR_FLOW_CONTROL_DEADLOCK:
        return "the task window is blocked while the heap is not full -- typically nesting on the "
               "same ring, where an outer task waits on an inner task that cannot get a slot";
    case PTO2_ERROR_DEP_POOL_OVERFLOW:
        return "a task's explicit fanin edges overflowed the ring's dependency spill pool";
    case PTO2_ERROR_INVALID_ARGS:
        return "an orchestration API rejected its arguments (bad alloc_tensors info, illegal nested "
               "scope, unknown task id in set_dependencies, or an L0TaskArgs carrying an error flag)";
#ifdef PTO2_ERROR_DEPENDENCY_OVERFLOW
    case PTO2_ERROR_DEPENDENCY_OVERFLOW:
        return "retired code: per-task fanin overflow is now reported as DEP_POOL_OVERFLOW";
#endif
    case PTO2_ERROR_REQUIRE_SYNC_START_INVALID:
        return "require_sync_start() asked for more blocks than the target core type physically has, "
               "which can never be satisfied";
    case PTO2_ERROR_TENSOR_WAIT_TIMEOUT:
        return "waiting for tensor data timed out: the producing task never completed, or a consumer "
               "never released its fanout reference";
    case PTO2_ERROR_EXPLICIT_ORCH_FATAL:
        return "the orchestration code called rt_report_fatal() itself; every later orchestration API "
               "call short-circuits to a no-op";
    case PTO2_ERROR_SCOPE_TASKS_OVERFLOW:
        return "the scope task-record buffer saturated (runtime-internal; the rings normally fill "
               "first and latch SCOPE_DEADLOCK or FLOW_CONTROL_DEADLOCK)";
    case PTO2_ERROR_TENSORMAP_OVERFLOW:
        return "the tensormap entry pool is wedged and last_task_alive cannot advance "
               "(runtime-internal, or an extreme-scale workload)";
    case PTO2_ERROR_SCHEDULER_TIMEOUT:
        return "the scheduler saw no forward progress within its timeout; the sub_class= line below "
               "carries the device-classified reason and locators";
    case PTO2_ERROR_ASYNC_COMPLETION_INVALID:
        return "an async completion condition is malformed (bad completion_type, null counter address, "
               "or an invalid SDMA event record)";
    case PTO2_ERROR_ASYNC_WAIT_OVERFLOW:
        return "the async wait list filled up: in-flight async completions exceeded the per-task cap";
    case PTO2_ERROR_ASYNC_REGISTRATION_FAILED:
        return "the scheduler received an async completion message of an illegal kind (runtime-internal; "
               "ASYNC_WAIT_OVERFLOW normally intercepts this first)";
    default:
        return "";
    }
}

// What to do next about a latched PTO2_ERROR_* code. Empty string for an uncovered code.
static inline const char *error_hint(int32_t code) {
    switch (code) {
    case PTO2_ERROR_NONE:
        return "";
    case PTO2_ERROR_SCOPE_DEADLOCK:
        return "raise ring_task_window (PTO2_RING_TASK_WINDOW) or split the scope so slots are "
               "reclaimed sooner; enable CallConfig.enable_scope_stats to see which scope peaked";
    case PTO2_ERROR_HEAP_RING_DEADLOCK:
        return "raise ring_heap (PTO2_RING_HEAP) or shrink per-task args / intermediate tensors; the "
               "'Ring buffer sizes:' line above reports the configured capacities";
    case PTO2_ERROR_FLOW_CONTROL_DEADLOCK:
        return "check for self-dependent or nested submission on one ring; raise ring_task_window "
               "(PTO2_RING_TASK_WINDOW) or move the nested submission to another ring";
    case PTO2_ERROR_DEP_POOL_OVERFLOW:
        return "raise ring_dep_pool (PTO2_RING_DEP_POOL) or cut the fanin count of the offending "
               "set_dependencies() call; enable CallConfig.enable_scope_stats to locate it";
    case PTO2_ERROR_INVALID_ARGS:
        return "an orchestration bug, not a capacity problem -- resizing the rings will not help; "
               "recheck the arguments of the orchestration API calls listed above";
#ifdef PTO2_ERROR_DEPENDENCY_OVERFLOW
    case PTO2_ERROR_DEPENDENCY_OVERFLOW:
        return "see DEP_POOL_OVERFLOW";
#endif
    case PTO2_ERROR_REQUIRE_SYNC_START_INVALID:
        return "lower the task's block_num to at most the physical core count of the target core type "
               "(total AIV count, or total cluster count for MIX/AIC), or drop the sync-start request";
    case PTO2_ERROR_TENSOR_WAIT_TIMEOUT:
        return "find the producing kernel and check it for a hang (see SCHEDULER_TIMEOUT sub_class S1); "
               "verify the consumer declares the dependency and exits; raise PTO2_TENSOR_DATA_TIMEOUT_MS "
               "to tell a slow kernel apart from a stuck one";
    case PTO2_ERROR_EXPLICIT_ORCH_FATAL:
        return "self-inflicted -- follow the message passed to the rt_report_fatal() call site";
    case PTO2_ERROR_SCOPE_TASKS_OVERFLOW:
    case PTO2_ERROR_TENSORMAP_OVERFLOW:
    case PTO2_ERROR_ASYNC_REGISTRATION_FAILED:
        return "not expected in normal operation -- keep the device log and report it to the runtime "
               "maintainers; tuning the ring capacities will not help";
    case PTO2_ERROR_SCHEDULER_TIMEOUT:
        return "read the sub_class= line below first, then follow the S1-S5 table in the doc; raise "
               "PTO2_SCHEDULER_TIMEOUT_MS to tell a true deadlock apart from a merely slow kernel";
    case PTO2_ERROR_ASYNC_COMPLETION_INVALID:
        return "recheck the register_completion_condition() arguments in the kernel, in particular the "
               "completion type and the counter address";
    case PTO2_ERROR_ASYNC_WAIT_OVERFLOW:
        return "cut the number of in-flight async completions per task, and confirm the consumer side "
               "actually polls and retires them";
    default:
        return "";
    }
}

// At most one of the two codes is ever non-zero (orchestrator and scheduler each latch
// their own, first-writer-wins). These two pick the latched one and name the field it
// came from, so every failure site annotates the same way without repeating the choice.
static inline int32_t latched_error_code(int32_t orch_error_code, int32_t sched_error_code) {
    return orch_error_code != PTO2_ERROR_NONE ? orch_error_code : sched_error_code;
}

static inline const char *latched_error_field(int32_t orch_error_code, int32_t sched_error_code) {
    return orch_error_code != PTO2_ERROR_NONE ? "orch_error_code" : "sched_error_code";
}

#endif  // SRC_COMMON_RUNTIME_STATUS_ERROR_NAMES_H_
