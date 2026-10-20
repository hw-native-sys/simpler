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
 * The runtime entry points an orchestration .so calls through.
 *
 * ChipWorker dlopens each runtime RTLD_LOCAL so two runtimes' identically-named
 * exports (simpler_init, simpler_run, ...) cannot collide when a process switches
 * between them, and so dlclose can actually unload one. A runtime's symbols
 * therefore never reach the global symbol table, and the orchestration .so — itself
 * dlopened RTLD_NOW|RTLD_LOCAL — has no way to bind them by name. This table is how
 * it reaches the runtime instead: the runtime fills it, the .so calls through it.
 *
 * The runtime and the orchestration .so include this same definition, which is what
 * makes the field order they agree on a single fact rather than two. Nothing would
 * catch a disagreement: a call dispatches through whatever pointer sits at the
 * offset the caller believes in.
 *
 * The header names only the argument types the signatures need, and no runtime
 * implementation types, for the same reason the orchestration .so links no runtime
 * .cpp.
 */

#pragma once

#include <stdint.h>

#include "host_build_graph/host/graph_cache.h"  // GraphScopeResult
#include "host_build_graph/submit_types.h"      // MixedKernels
#include "host_build_graph/tensor.h"            // simpler::hbg::Tensor
#include "host_build_graph/types.h"             // TaskOutputTensors, CoreTaskArgs, GraphTaskArgs

// Every op runs on the host and takes the orchestration context, so this table
// belongs to orchestration rather than to the device-side runtime. The
// orchestration .so sees OrchContext as an incomplete type plus the two-field
// partial definition in orchestration_api.h; the host sees the full struct in
// host/orchestrator.h. Both agree on the first two fields, which is what makes
// the partial definition well-defined.
struct OrchContext;

struct OrchOps {
    TaskOutputTensors (*submit_task)(OrchContext *ctx, const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    void (*scope_begin)(OrchContext *ctx);
    void (*scope_end)(OrchContext *ctx);
    void (*orchestration_done)(OrchContext *ctx);
    bool (*is_fatal)(OrchContext *ctx);
    void (*report_fatal)(OrchContext *ctx, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_timing)(const char *func, const char *fmt, ...);
    void (*log_info)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(
        OrchContext *ctx, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[]
    );
    void (*set_tensor_data)(
        OrchContext *ctx, const simpler::hbg::Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
    TaskOutputTensors (*alloc_tensors)(OrchContext *ctx, const CoreTaskArgs &args);
    TaskOutputTensors (*submit_dummy_task)(OrchContext *ctx, const CoreTaskArgs &args);

    // This-run core geometry latched by the host bind: MIX clusters
    // (one AIC each) and standalone AIV cores.
    int32_t (*available_cluster_count)(OrchContext *ctx);
    int32_t (*available_aiv_count)(OrchContext *ctx);
    GraphScopeResult (*graph_begin)(OrchContext *ctx, uint64_t graph_key, const GraphTaskArgs &args);
    bool (*graph_prepare)(OrchContext *ctx, void *recording_handle, const GraphTaskArgs &args);
    void (*graph_abort)(OrchContext *ctx, void *recording_handle);
    bool (*graph_end)(OrchContext *ctx);
    void (*graph_commit)(OrchContext *ctx);

    // Record one orchestration-side phase on the calling thread. The submission
    // segments this carries are measured in the orchestration .so and are invisible
    // to the runtime, which sees only what it is called for. Always present so the
    // layout does not move with SIMPLER_DFX; nullptr when DFX is off.
    void (*record_orch_phase)(uint32_t kind, uint64_t start_ns, uint64_t end_ns, uint64_t detail);
    // Queue one Graph body for asynchronous recording. Wait drains only this
    // RuntimeContext's jobs, including closure destruction.
    // `job` is a `std::function<void(const GraphTaskArgs &)> *` the pool moves out of --
    // whether or not it queues it, since start() takes the callable before it checks
    // capacity -- so the caller must not invoke it afterwards. Nothing is owned across
    // the boundary either way: the caller's std::function destructs normally, empty or
    // not, and rt_graph_submit's fallback re-runs its own copy of the body. The pool is
    // runtime-owned (host/graph_recorder_pool.h); a null or refusing start makes the
    // caller record the body inline instead.
    bool (*graph_record_start)(OrchContext *ctx, const GraphTaskArgs &args, void *job);
    void (*graph_record_wait)(OrchContext *ctx);

    // Publishes an EndpointFailureIdentity only when this call latches the run's first fatal.
    void (*report_endpoint_failure)(
        OrchContext *ctx, uint32_t kind, uint64_t session_bits, uint64_t transaction_id, const char *func,
        const char *message
    );
};
