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
 * through the PTO2RuntimeOps function-pointer table embedded in
 * PTO2Runtime.
 *
 * Orchestration sources include ONLY this header:
 *   #include "pto_orchestration_api.h"
 *
 * Runtime sources continue to use pto_runtime2.h (which defines the
 * full PTO2Runtime struct with all internal fields).
 */

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <cstring>
#include <type_traits>

// Type headers needed by orchestration
#include "common.h"              // framework_bind_runtime / framework_current_runtime
#include "graph_cache.h"         // Graph Execution key and result helpers
#include "pto_runtime2_types.h"  // PTO2_ERROR_*
#include "pto_submit_types.h"    // MixedKernels, INVALID_KERNEL_ID, subtask slots
#include "pto_types.h"           // Arg, TaskOutputTensors, TensorArgType
#include "task_args.h"           // ChipStorageTaskArgs, ChipTensor
#include "tensor.h"              // ChipTensor, TensorCreateInfo

// =============================================================================
// ChipTensor Factory Helpers
// =============================================================================

// make_tensor_external(...) — canonical factory for pre-allocated external
// memory — is defined in the unified tensor.h (common), so host and runtime
// build ChipTensors through the same controlled path.

// =============================================================================
// Ops Table and Opaque Runtime
// =============================================================================

/**
 * Forward declaration — the orchestration sees PTO2Runtime as a partial
 * struct whose first field is the ops pointer.  The full definition
 * lives in pto_runtime2.h (used only by runtime .cpp files).
 */
typedef struct PTO2Runtime PTO2Runtime;

/**
 * Function-pointer table for runtime operations.
 * Populated by the runtime; called by orchestration through inline wrappers.
 */
typedef struct PTO2RuntimeOps {
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const CoreTaskArgs &args);
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);
    void (*report_fatal)(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_timing)(const char *func, const char *fmt, ...);
    void (*log_info)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[]);
    void (*set_tensor_data)(
        PTO2Runtime *rt, const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
    TaskOutputTensors (*alloc_tensors)(PTO2Runtime *rt, const CoreTaskArgs &args);
    TaskOutputTensors (*submit_dummy_task)(PTO2Runtime *rt, const CoreTaskArgs &args);

    // This-run core geometry from runtime_finalize_after_wire: MIX clusters
    // (one AIC each) and standalone AIV cores.
    int32_t (*available_cluster_count)(PTO2Runtime *rt);
    int32_t (*available_aiv_count)(PTO2Runtime *rt);
    GraphScopeResult (*graph_begin)(PTO2Runtime *rt, uint64_t graph_key, const CoreTaskArgs &args);
    void (*graph_end)(PTO2Runtime *rt);
    void (*graph_commit)(PTO2Runtime *rt);

    // Stash the call-site of the next PTO2ScopeGuard so the [ScopeStats]
    // collector can log it. Always present to keep ops-table layout stable
    // across SIMPLER_DFX settings; set to nullptr at SIMPLER_DFX=0.
    void (*scope_set_site)(const char *file, int line);
} PTO2RuntimeOps;

/**
 * Partial PTO2Runtime definition for orchestration.
 *
 * Exposes the ops pointer (for runtime calls) and pending_scope_mode
 * (read directly by inline scope wrappers).  The real struct (in
 * pto_runtime2.h) has the same first fields, so accessing them through
 * this definition is well-defined (C struct layout guarantee).
 */
struct PTO2Runtime {
    const PTO2RuntimeOps *ops;
    PTO2ScopeMode pending_scope_mode;
};

// =============================================================================
// Inline Convenience Wrappers (call through ops table)
// =============================================================================

static inline PTO2Runtime *current_runtime() { return framework_current_runtime(); }

static inline TaskOutputTensors alloc_tensors(const CoreTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->alloc_tensors(rt, args);
}

static inline TaskOutputTensors alloc_tensors(const TensorCreateInfo create_infos[], uint32_t count) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    for (uint32_t i = 0; i < count; i++) {
        args.add_output(create_infos[i]);
    }
    if (args.has_error) {
        rt->ops->report_fatal(
            rt, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
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
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    CoreTaskArgs args;
    (args.add_output(cis), ...);
    if (args.has_error) {
        rt->ops->report_fatal(
            rt, PTO2_ERROR_INVALID_ARGS, __FUNCTION__, "%s",
            args.error_msg ? args.error_msg : "alloc_tensors failed to construct output-only Arg"
        );
        return TaskOutputTensors{};
    }
    return alloc_tensors(args);
}

static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const CoreTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_task(rt, mixed_kernels, args);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, const CoreTaskArgs &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, const CoreTaskArgs &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task(mk, args);
}

/**
 * Submit a dependency-only task. Accepts the same Arg shape as rt_submit_task
 * (inputs, outputs, inouts, explicit_deps, scalars) but does not run any
 * AICore kernel. The task still participates in the dependency graph: it
 * waits on its fanin and notifies its fanout. Useful as a synchronization
 * barrier or as a placeholder producer for tests / dep-graph wiring.
 */
static inline TaskOutputTensors rt_submit_dummy_task(const CoreTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_dummy_task(rt, args);
}

static inline GraphScopeResult rt_graph_begin(uint64_t graph_key, const CoreTaskArgs &args) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt) || rt->ops->graph_begin == nullptr) {
        return GraphScopeResult{};
    }
    return rt->ops->graph_begin(rt, graph_key, args);
}

static inline void rt_graph_end() {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt) || rt->ops->graph_end == nullptr) {
        return;
    }
    rt->ops->graph_end(rt);
}

static inline void rt_graph_commit() {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt) || rt->ops->graph_commit == nullptr) {
        return;
    }
    rt->ops->graph_commit(rt);
}

static inline void rt_scope_begin(PTO2ScopeMode mode = PTO2ScopeMode::AUTO) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->pending_scope_mode = mode;
    rt->ops->scope_begin(rt);
}

static inline void rt_scope_end() {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return;
    }
    rt->ops->scope_end(rt);
}

static inline void rt_orchestration_done() {
    PTO2Runtime *rt = current_runtime();
    rt->ops->orchestration_done(rt);
}

/** This-run MIX cluster (= AIC) count. Do not hardcode 24/36; MIX cohorts use this. */
static inline int32_t rt_available_cluster_count() {
    PTO2Runtime *rt = current_runtime();
    return rt->ops->available_cluster_count(rt);
}

/** This-run standalone AIV core count. AIV-only cohorts size themselves on this. */
static inline int32_t rt_available_aiv_count() {
    PTO2Runtime *rt = current_runtime();
    return rt->ops->available_aiv_count(rt);
}

static inline bool rt_is_fatal() {
    PTO2Runtime *rt = current_runtime();
    return rt->ops->is_fatal(rt);
}

#define rt_report_fatal(code, fmt, ...)                                          \
    do {                                                                         \
        PTO2Runtime *_rt = current_runtime();                                    \
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
 * This API reads the registered host view used to stage an external tensor.
 * It is valid while host orchestration is building the graph, before device
 * scheduling starts. A tensor produced by a submitted task cannot become
 * readable during graph construction, and a runtime-created output has no
 * registered host view; either use is reported as an invalid argument.
 */
template <typename T = uint64_t>
static inline T get_tensor_data(const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    PTO2Runtime *rt = current_runtime();
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
 * This API updates the registered host view used to stage an external tensor.
 * The updated value becomes part of the graph's initial device data. It is not
 * a synchronization barrier for submitted readers or writers. A tensor with a
 * submitted producer, or a runtime-created output with no registered host view,
 * is rejected as an invalid argument.
 */
template <typename T = uint64_t>
static inline void set_tensor_data(const ChipTensor &tensor, uint32_t ndims, const uint32_t indices[], T value) {
    PTO2Runtime *rt = current_runtime();
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
class PTO2ScopeGuard {
public:
    explicit PTO2ScopeGuard(
        PTO2ScopeMode mode = PTO2ScopeMode::AUTO, const char *file = __builtin_FILE(), int line = __builtin_LINE()
    ) :
        rt_(current_runtime()) {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->pending_scope_mode = mode;
            if (rt_->ops->scope_set_site) rt_->ops->scope_set_site(file, line);
            rt_->ops->scope_begin(rt_);
        }
    }
    ~PTO2ScopeGuard() {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->ops->scope_end(rt_);
        }
    }

private:
    PTO2Runtime *rt_;
};

using GraphFunction = void (*)(const CoreTaskArgs &);

template <typename Function>
static inline uint64_t rt_graph_function_id(Function function) {
    static_assert(std::is_pointer_v<Function>, "Graph function identity requires a function pointer");
    static_assert(sizeof(function) <= sizeof(uint64_t), "Graph function pointer must fit in a 64-bit identity");
    uint64_t function_id = 0;
    std::memcpy(&function_id, &function, sizeof(function));
    return function_id;
}

template <typename Invoke>
static inline GraphSubmitResult rt_submit_graph_impl(uint64_t graph_key, const CoreTaskArgs &args, Invoke invoke) {
    debug_assert(!args.has_error && "Graph boundary CoreTaskArgs construction failed");
    debug_assert(
        args.tensor_count() <= static_cast<int32_t>(GRAPH_MAX_TENSOR_ARGS) &&
        "Graph boundary exceeds the step-1 tensor limit"
    );
    debug_assert(args.scalar_count() == 0 && "Dynamic Graph boundary scalars are not supported in step 1");
    debug_assert(
        args.explicit_dep_count() == 0 && "Explicit dependencies crossing the Graph boundary are not supported"
    );
    for (int32_t i = 0; i < args.tensor_count(); ++i) {
        debug_assert(
            args.tag(i) != TensorArgType::OUTPUT &&
            "Runtime-allocated TensorCreateInfo is not supported at the Graph boundary"
        );
    }
    if (!rt_graph_args_cacheable(args)) {
        invoke();
        rt_graph_commit();
        return GraphSubmitResult{};
    }
    GraphScopeResult result = rt_graph_begin(graph_key, args);
    if (result.execute_block) invoke();
    if (result.recording) {
        rt_graph_end();
    }
    rt_graph_commit();
    return result;
}

static inline GraphSubmitResult rt_submit_graph(uint64_t graph_id, GraphFunction function, const CoreTaskArgs &args) {
    debug_assert(function != nullptr && "Graph function must not be null");
    if (function == nullptr) return GraphSubmitResult{};
    return rt_submit_graph_impl(rt_graph_make_key(graph_id), args, [&]() {
        function(args);
    });
}

static inline GraphSubmitResult rt_submit_graph(GraphFunction function, const CoreTaskArgs &args) {
    return rt_submit_graph(rt_graph_function_id(function), function, args);
}

template <typename... Config>
using GraphFunctionWithConfig = void (*)(const CoreTaskArgs &, Config...);

template <typename... Config>
static inline GraphSubmitResult rt_submit_graph(
    uint64_t graph_id, GraphFunctionWithConfig<Config...> function, const CoreTaskArgs &args, Config... config
) {
    debug_assert(function != nullptr && "Graph function must not be null");
    if (function == nullptr) return GraphSubmitResult{};
    return rt_submit_graph_impl(rt_graph_make_key(graph_id, config...), args, [&]() {
        function(args, config...);
    });
}

template <typename... Config>
static inline GraphSubmitResult
rt_submit_graph(GraphFunctionWithConfig<Config...> function, const CoreTaskArgs &args, Config... config) {
    return rt_submit_graph(rt_graph_function_id(function), function, args, config...);
}

#define _PTO2_CONCATENATE_IMPL(x, y) x##y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)

#define PTO2_SCOPE_GUARD() [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__)

/**
 * Scoped block macro:
 *   PTO2_SCOPE() {
 *       rt_submit_task(...);
 *   }
 */
#define PTO2_SCOPE(...) if (PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__){__VA_ARGS__}; true)

// =============================================================================
// Orchestration Config
// =============================================================================

/**
 * Configuration exported by orchestration .so via aicpu_orchestration_config().
 * The executor reads these values to set up shared memory and runtime.
 *
 * This struct is defined identically in pto_runtime2.h (with an include
 * guard) so the executor can use the same type without including this header.
 */
#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int expected_arg_count;
};
#endif

// Convenience layer (CoreTaskArgsWithDeps<N> + matching rt_submit_*_task overloads).
// Pulled in at the bottom so the wrapper sees CoreTaskArgs, MixedKernels, and the
// rt_submit_*_task primitives defined above. Orchestration sources include
// only this single header to access both the primitive and convenience APIs.
#include "pto_arg_with_deps.h"  // NOLINT(build/include_subdir)
