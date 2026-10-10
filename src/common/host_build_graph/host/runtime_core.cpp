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
 * host_build_graph orchestration ops table
 *
 * The runtime entries the orchestration .so reaches through RuntimeContext::ops,
 * and the table itself. host_build_graph orchestrates on the host, so the AICPU
 * target compiles none of this and no device code reads the ops field.
 */

#include "host_build_graph/host_phase_trace.h"
#include "host_build_graph/runtime_core.h"

#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "common/unified_log.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/host_tensor_access.h"
#include "host_build_graph/task_id.h"

// =============================================================================
// Orchestration Ops Table (function-pointer dispatch for orchestration .so)
// =============================================================================

static TaskOutputTensors
submit_task_impl(RuntimeContext *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    return rt->orchestrator->submit_task(mixed_kernels, args);
}

static TaskOutputTensors alloc_tensors_impl(RuntimeContext *rt, const CoreTaskArgs &args) {
    return rt->orchestrator->alloc_tensors(args);
}

static TaskOutputTensors submit_dummy_task_impl(RuntimeContext *rt, const CoreTaskArgs &args) {
    return rt->orchestrator->submit_dummy_task(args);
}

static GraphScopeResult graph_begin_impl(RuntimeContext *rt, uint64_t graph_key, const GraphTaskArgs &args) {
    if (rt == nullptr) return GraphScopeResult{};
    return rt->orchestrator->graph_begin(graph_key, args, rt->active_callable_hash);
}

static GraphScopeResult graph_begin_with_config_impl(
    RuntimeContext *rt, uint64_t graph_id, uint64_t graph_key, GraphConfigView config, const GraphTaskArgs &args
) {
    if (rt == nullptr) return GraphScopeResult{};
    return rt->orchestrator->graph_begin(graph_id, graph_key, config, args, rt->active_callable_hash);
}

static bool graph_prepare_impl(RuntimeContext *rt, void *recording_handle, const GraphTaskArgs &args) {
    return rt != nullptr && rt->orchestrator->graph_prepare(recording_handle, args);
}

static void graph_abort_impl(RuntimeContext *rt, void *recording_handle) {
    if (rt != nullptr) rt->orchestrator->graph_abort(recording_handle);
}

static bool graph_end_impl(RuntimeContext *rt) { return rt != nullptr && rt->orchestrator->graph_end(); }

// The orchestration .so's phases arrive already bracketed, so this only forwards them
// to the same pool the runtime's own records go to. Defined only at SIMPLER_DFX=1: the
// slot stays in the struct either way, but a definition the table does not reference is
// an unused-function error on the AICPU build.
#if SIMPLER_DFX
static void record_orch_phase_impl(uint32_t kind, uint64_t start_ns, uint64_t end_ns, uint64_t detail) {
    host_phase_record(start_ns, end_ns, kind, detail, 0);
}
#endif

static void graph_commit_impl(RuntimeContext *rt) {
    if (rt != nullptr) rt->orchestrator->graph_commit();
}

void rt_scope_begin(RuntimeContext *rt) {
    ScopeMode mode = rt->pending_scope_mode;
    rt->pending_scope_mode = ScopeMode::AUTO;
    rt->orchestrator->begin_scope(mode);
}

void rt_scope_end(RuntimeContext *rt) { rt->orchestrator->end_scope(); }

void rt_orchestration_done(RuntimeContext *rt) {
    // Host orchestration calls this runtime entry directly rather than the
    // orchestration-SO wrapper. Commit here as well so an all-Graph entry has a
    // final synchronization point for its asynchronous recording and deferred
    // outer shells even when no later non-Graph task forced an earlier commit.
    rt->orchestrator->graph_commit();
    rt->orchestrator->mark_done();
    // The orchestrator itself never crosses to the device, so the count of tasks
    // it completed inline is published into the header that does.
    rt->inline_completed_tasks = rt->orchestrator->inline_completed_tasks;
}

static bool is_fatal_impl(RuntimeContext *rt) { return rt->orchestrator->is_fatal(); }

void rt_report_fatal(RuntimeContext *rt, int32_t error_code, const char *func, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    if (fmt == nullptr || fmt[0] == '\0') {
        rt->orchestrator->report_fatal(error_code, func, nullptr);
    } else {
        char message[1024];
        vsnprintf(message, sizeof(message), fmt, args);
        rt->orchestrator->report_fatal(error_code, func, "%s", message);
    }
    va_end(args);
}

static bool require_host_tensor(RuntimeContext *rt, const simpler::hbg::Tensor &tensor, const char *caller) {
    if (tensor.address_space == AddressSpace::HOST && !tensor.owner_task_id.is_valid()) return true;
    rt->orchestrator->report_fatal(
        SIMPLER_ERROR_INVALID_ARGS, caller,
        "host tensor access requires an explicit HOST/NONE argument; DEVICE values need an explicit copy before submit"
    );
    return false;
}

uint64_t
get_tensor_data(RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    // A refused access cannot replace an earlier orchestration failure.
    if (rt->orchestrator->is_fatal()) {
        return 0;
    }
    if (tensor.buffer.addr == 0) {
        rt->orchestrator->report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "get_tensor_data requires a nonempty allocated HOST view"
        );
        return 0;
    }

    if (!require_host_tensor(rt, tensor, __FUNCTION__)) {
        return 0;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    uint64_t elem_addr = tensor.buffer.addr + flat_offset * elem_size;
    uint64_t result = 0;
    if (!host_tensor_read(rt->tensor_access, elem_addr, &result, elem_size)) {
        rt->orchestrator->report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "HOST argument read is outside the bound view or violates its declared direction"
        );
        return 0;
    }
    return result;
}

void set_tensor_data(
    RuntimeContext *rt, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
) {
    // Preserve the first orchestration failure.
    if (rt->orchestrator->is_fatal()) {
        return;
    }
    if (tensor.buffer.addr == 0) {
        rt->orchestrator->report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "set_tensor_data requires a nonempty allocated HOST view"
        );
        return;
    }

    if (!require_host_tensor(rt, tensor, __FUNCTION__)) {
        return;
    }

    uint64_t flat_offset = tensor.compute_flat_offset(indices, ndims);
    uint64_t elem_size = get_element_size(tensor.dtype);
    uint64_t elem_addr = tensor.buffer.addr + flat_offset * elem_size;
    if (!host_tensor_write(rt->tensor_access, elem_addr, &value, elem_size)) {
        rt->orchestrator->report_fatal(
            SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__,
            "HOST argument write is outside the bound view or violates its declared direction"
        );
        return;
    }
}

static int32_t available_cluster_count_impl(RuntimeContext *rt) { return rt->orchestrator->total_cluster_count; }
static int32_t available_aiv_count_impl(RuntimeContext *rt) { return rt->orchestrator->total_aiv_count; }

static const RuntimeOps s_runtime_ops = {
    .submit_task = submit_task_impl,
    .scope_begin = rt_scope_begin,
    .scope_end = rt_scope_end,
    .orchestration_done = rt_orchestration_done,
    .is_fatal = is_fatal_impl,
    .report_fatal = rt_report_fatal,
    .log_error = unified_log_error,
    .log_warn = unified_log_warn,
    .log_timing = unified_log_timing,
    .log_info = unified_log_info,
    .log_debug = unified_log_debug,
    .get_tensor_data = get_tensor_data,
    .set_tensor_data = set_tensor_data,
    .alloc_tensors = alloc_tensors_impl,
    .submit_dummy_task = submit_dummy_task_impl,
    .available_cluster_count = available_cluster_count_impl,
    .available_aiv_count = available_aiv_count_impl,
    .graph_begin = graph_begin_impl,
    .graph_prepare = graph_prepare_impl,
    .graph_abort = graph_abort_impl,
    .graph_end = graph_end_impl,
    .graph_commit = graph_commit_impl,
#if SIMPLER_DFX
    .record_orch_phase = record_orch_phase_impl,
#else
    .record_orch_phase = nullptr,
#endif
    .graph_record_start = graph_record_start_impl,
    .graph_record_wait = graph_record_wait_impl,
    .graph_begin_with_config = graph_begin_with_config_impl,
};

// =============================================================================
// Runtime Lifecycle
// =============================================================================
//
// Layout / init_data / wire / destroy live in
// host_build_graph/shared/runtime_init.cpp so the host build can pre-populate the
// prebuilt arena image. Binding the ops table is host-only: the bind installs it
// for the orchestration .so and clears the field again before the RuntimeContext
// travels, so the device never receives a host address here.

void runtime_bind_ops(RuntimeContext *rt) { rt->ops = &s_runtime_ops; }

void runtime_set_mode(RuntimeContext *rt, RuntimeMode mode) {
    if (rt) {
        rt->mode = mode;
    }
}
