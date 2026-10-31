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
 * PTO Orchestration API - Slim header for orchestration .so files
 *
 * This header provides everything an orchestration source needs without
 * pulling in runtime implementation headers.  The orchestration .so has
 * zero link dependencies on runtime .cpp files; all runtime calls go
 * through the RuntimeOps function-pointer table embedded in
 * RuntimeContext.
 *
 * Orchestration sources include ONLY this header:
 *   #include "orchestration_api.h"
 *
 * Runtime sources continue to use runtime_core.h (which defines the
 * full RuntimeContext struct with all internal fields).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <type_traits>

// Type headers needed by orchestration
#include "common.h"         // framework_bind_runtime / framework_current_runtime
#include "runtime_types.h"  // SIMPLER_ERROR_*
#include "submit_types.h"   // MixedKernels, INVALID_KERNEL_ID, subtask slots
#include "types.h"          // Arg, TaskOutputTensors, TensorArgType
#include "tensormap_and_ringbuffer/blocking_wait.h"
#include "task_args.h"  // ChipStorageTaskArgs, simpler::tmr::Tensor
#include "tensor.h"     // simpler::tmr::Tensor, TensorCreateInfo

// =============================================================================
// simpler::tmr::Tensor Factory Helpers
// =============================================================================

// simpler::tmr::make_tensor_external(...) — canonical factory for pre-allocated external
// memory — is defined in the unified tensor.h (common), so host and runtime
// build ChipTensors through the same controlled path.

// =============================================================================
// Ops Table and Opaque Runtime
// =============================================================================

/**
 * Forward declaration — the orchestration sees RuntimeContext as a partial
 * struct whose first field is the ops pointer.  The full definition
 * lives in runtime_core.h (used only by runtime .cpp files).
 */
typedef struct RuntimeContext RuntimeContext;

/**
 * Function-pointer table for runtime operations.
 * Populated by the runtime; called by orchestration through inline wrappers.
 */
typedef struct RuntimeOps {
    TaskOutputTensors (*submit_task)(RuntimeContext *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    void (*scope_begin)(RuntimeContext *rt);
    void (*scope_end)(RuntimeContext *rt);
    void (*orchestration_done)(RuntimeContext *rt);
    bool (*is_fatal)(RuntimeContext *rt);
    void (*report_fatal)(RuntimeContext *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_timing)(const char *func, const char *fmt, ...);
    void (*log_info)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(
        RuntimeContext *rt, const simpler::tmr::Tensor &tensor, uint32_t ndims, const uint32_t indices[]
    );
    void (*set_tensor_data)(
        RuntimeContext *rt, const simpler::tmr::Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
    TaskOutputTensors (*alloc_tensors)(RuntimeContext *rt, const CoreTaskArgs &args);
    TaskOutputTensors (*submit_dummy_task)(RuntimeContext *rt, const CoreTaskArgs &args);

    // This-run core geometry from runtime_finalize_after_wire: MIX clusters
    // (one AIC each) and standalone AIV cores.
    int32_t (*available_cluster_count)(RuntimeContext *rt);
    int32_t (*available_aiv_count)(RuntimeContext *rt);

    // Stash the call-site of the next ScopeGuard so the [ScopeStats]
    // collector can log it. Always present to keep ops-table layout stable
    // across SIMPLER_DFX settings; set to nullptr at SIMPLER_DFX=0.
    void (*scope_set_site)(const char *file, int line);
} RuntimeOps;

/**
 * Partial RuntimeContext definition for orchestration.
 *
 * Exposes the ops pointer (for runtime calls) and pending_scope_mode
 * (read directly by inline scope wrappers).  The real struct (in
 * runtime_core.h) has the same first fields, so accessing them through
 * this definition is well-defined (C struct layout guarantee).
 */
struct RuntimeContext {
    const RuntimeOps *ops;
    ScopeMode pending_scope_mode;
};

// =============================================================================
// Inline Convenience Wrappers (call through ops table)
// =============================================================================

static inline RuntimeContext *current_runtime() { return framework_current_runtime(); }

static inline TaskOutputTensors alloc_tensors(const CoreTaskArgs &args) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->alloc_tensors(rt, args);
}

static inline TaskOutputTensors alloc_tensors(const TensorCreateInfo create_infos[], uint32_t count) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    for (uint32_t i = 0; i < count; i++) {
        args.add_output(create_infos[i]);
    }
    if (args.has_error) {
        rt->ops->report_fatal(
            rt, SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

template <typename... CIs>
static inline TaskOutputTensors alloc_tensors(const CIs &...cis) {
    static_assert(sizeof...(cis) > 0, "alloc_tensors requires at least one TensorCreateInfo");
    static_assert(
        (std::is_same_v<std::decay_t<CIs>, TensorCreateInfo> && ...),
        "alloc_tensors only accepts TensorCreateInfo arguments"
    );
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    (args.add_output(cis), ...);
    if (args.has_error) {
        rt->ops->report_fatal(
            rt, SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_task(rt, mixed_kernels, args);
}

static inline TaskOutputTensors
rt_submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args, const BlockingWait &wait) {
    const BlockingWaitOps *ops = framework_current_blocking_wait_ops();
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) return {};
    if (ops == nullptr || ops->version != BLOCKING_WAIT_OPS_VERSION || ops->size < sizeof(BlockingWaitOps) ||
        ops->submit_task == nullptr) {
        rt->ops->report_fatal(
            rt, SIMPLER_ERROR_INVALID_ARGS, __FUNCTION__, "runtime does not support blocking-wait admission"
        );
        return {};
    }
    return ops->submit_task(rt, mixed_kernels, args, wait);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, const CoreTaskArgs &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

static inline TaskOutputTensors
rt_submit_aic_task(int32_t kernel_id, const CoreTaskArgs &args, const BlockingWait &wait) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task(mk, args, wait);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, const CoreTaskArgs &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

static inline TaskOutputTensors
rt_submit_aiv_task(int32_t kernel_id, const CoreTaskArgs &args, const BlockingWait &wait) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task(mk, args, wait);
}

/**
 * Submit a dependency-only task. Accepts the same Arg shape as rt_submit_task
 * (inputs, outputs, inouts, explicit_deps, scalars) but does not run any
 * AICore kernel. The task still participates in the dependency graph: it
 * waits on its fanin and notifies its fanout. Useful as a synchronization
 * barrier or as a placeholder producer for tests / dep-graph wiring.
 */
static inline TaskOutputTensors rt_submit_dummy_task(const CoreTaskArgs &args) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_dummy_task(rt, args);
}

static inline void rt_scope_begin(ScopeMode mode = ScopeMode::AUTO) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->pending_scope_mode = mode;
    rt->ops->scope_begin(rt);
}

static inline void rt_scope_end() {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->ops->scope_end(rt);
}

static inline void rt_orchestration_done() {
    RuntimeContext *rt = current_runtime();
    rt->ops->orchestration_done(rt);
}

/** This-run MIX cluster (= AIC) count. Do not hardcode 24/36; MIX cohorts use this. */
static inline int32_t rt_available_cluster_count() {
    RuntimeContext *rt = current_runtime();
    return rt->ops->available_cluster_count(rt);
}

/** This-run standalone AIV core count. AIV-only cohorts size themselves on this. */
static inline int32_t rt_available_aiv_count() {
    RuntimeContext *rt = current_runtime();
    return rt->ops->available_aiv_count(rt);
}

static inline bool rt_is_fatal() {
    RuntimeContext *rt = current_runtime();
    return rt->ops->is_fatal(rt);
}

#define rt_report_fatal(code, fmt, ...)                                          \
    do {                                                                         \
        RuntimeContext *_rt = current_runtime();                                 \
        _rt->ops->report_fatal(_rt, (code), __FUNCTION__, (fmt), ##__VA_ARGS__); \
    } while (0)

// =============================================================================
// Logging Macros for Orchestration (call through ops table)
// =============================================================================

#define LOG_ERROR(fmt, ...) current_runtime()->ops->log_error(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) current_runtime()->ops->log_warn(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_TIMING(fmt, ...) current_runtime()->ops->log_timing(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) current_runtime()->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) current_runtime()->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)

// =============================================================================
// Cross-Layer Data Access
// =============================================================================

/**
 * Read a value from a tensor at the given multi-dimensional indices.
 *
 * Default T = uint64_t preserves old behavior (raw bits).
 * Specify T to get automatic type conversion:
 *
 *   uint64_t raw = get_tensor_data(tensor, 1, idx);       // old usage unchanged
 *   float val = get_tensor_data<float>(tensor, 1, idx);   // typed read
 *
 * If the tensor has a producer in TensorMap, spin-waits until the producer
 * task completes before reading. External tensors (make_tensor_external)
 * are read immediately without waiting.
 */
template <typename T = uint64_t>
static inline T get_tensor_data(const simpler::tmr::Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return from_u64<T>(0);
    }
    return from_u64<T>(rt->ops->get_tensor_data(rt, tensor, ndims, indices));
}

/**
 * Write a value to a tensor at the given multi-dimensional indices.
 *
 * Type is deduced from value argument; uint64_t by default:
 *
 *   set_tensor_data(tensor, 1, idx, raw_u64);     // old usage unchanged
 *   set_tensor_data(tensor, 1, idx, 42.0f);       // typed write (T = float)
 *
 * If the tensor has a producer in TensorMap, spin-waits until the producer
 * and all its consumers complete before writing (WAW + WAR safety).
 * External tensors (make_tensor_external) with no TensorMap entry are
 * written immediately without waiting.
 *
 * Limitation: TensorMap only tracks producers (OUTPUT/INOUT), not consumers
 * that used the tensor as INPUT. If a kernel reads this tensor as INPUT
 * (not INOUT) and the tensor has no TensorMap producer entry, set_tensor_data
 * cannot detect the reader and may cause a data race.
 *
 * To ensure WAR safety for all access patterns, use add_inout() instead of
 * add_input() for kernel parameters that may later be written via
 * set_tensor_data. INOUT creates a TensorMap entry that enables automatic
 * consumer tracking via fanout_refcount.
 *
 * The tensor must already have an allocated buffer (addr != 0).
 * For runtime-created outputs, call this only on the simpler::tmr::Tensor returned by
 * add_output(TensorCreateInfo) after submit returns.
 */
template <typename T = uint64_t>
static inline void
set_tensor_data(const simpler::tmr::Tensor &tensor, uint32_t ndims, const uint32_t indices[], T value) {
    RuntimeContext *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->ops->set_tensor_data(rt, tensor, ndims, indices, to_u64(value));
}

// =============================================================================
// C++ Scope Guards and Macros
// =============================================================================

/**
 * RAII Scope Guard (calls through ops table)
 */
class ScopeGuard {
public:
    explicit ScopeGuard(
        ScopeMode mode = ScopeMode::AUTO, const char *file = __builtin_FILE(), int line = __builtin_LINE()
    ) :
        rt_(current_runtime()) {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->pending_scope_mode = mode;
            if (rt_->ops->scope_set_site) rt_->ops->scope_set_site(file, line);
            rt_->ops->scope_begin(rt_);
        }
    }
    ~ScopeGuard() {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->ops->scope_end(rt_);
        }
    }

private:
    RuntimeContext *rt_;
};

#define _SIMPLER_CONCATENATE_IMPL(x, y) x##y
#define _SIMPLER_CONCATENATE(x, y) _SIMPLER_CONCATENATE_IMPL(x, y)

#define SIMPLER_SCOPE_GUARD(...) \
    [[maybe_unused]] ScopeGuard _SIMPLER_CONCATENATE(scope_guard_, __COUNTER__) { __VA_ARGS__ }

/**
 * Scoped block macro:
 *   SIMPLER_SCOPE() {
 *       rt_submit_task(...);
 *   }
 */
#define SIMPLER_SCOPE(...) if (ScopeGuard _SIMPLER_CONCATENATE(scope_guard_, __COUNTER__){__VA_ARGS__}; true)

// =============================================================================
// Orchestration Config
// =============================================================================

/**
 * Configuration exported by orchestration .so via aicpu_orchestration_config().
 * The executor reads these values to set up shared memory and runtime.
 *
 * This struct is defined identically in runtime_core.h (with an include
 * guard) so the executor can use the same type without including this header.
 */
#ifndef ORCHESTRATION_CONFIG_DEFINED
#define ORCHESTRATION_CONFIG_DEFINED
struct OrchestrationConfig {
    int expected_arg_count;
};
#endif

// Convenience layer (CoreTaskArgsWithDeps<N> + matching rt_submit_*_task overloads).
// Pulled in at the bottom so the wrapper sees CoreTaskArgs, MixedKernels, and the
// rt_submit_*_task primitives defined above. Orchestration sources include
// only this single header to access both the primitive and convenience APIs.
#include "arg_with_deps.h"  // NOLINT(build/include_subdir)
