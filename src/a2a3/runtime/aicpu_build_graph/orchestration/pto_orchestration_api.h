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

#ifndef SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_ORCHESTRATION_PTO_ORCHESTRATION_API_H_
#define SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_ORCHESTRATION_PTO_ORCHESTRATION_API_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Type headers needed by orchestration
#include "pto_runtime2_types.h"  // PTO2TaskId  // NOLINT(build/include_subdir)
#include "pto_submit_types.h"    // MixedKernels, INVALID_KERNEL_ID, subtask slots  // NOLINT(build/include_subdir)
#include "pto_types.h"           // Arg, PTOTensorEntry, TensorArgType  // NOLINT(build/include_subdir)
#include "task_args.h"           // ChipStorageTaskArgs, ContinuousTensor  // NOLINT(build/include_subdir)
#include "tensor.h"              // Tensor, TensorCreateInfo, make_tensor_external  // NOLINT(build/include_subdir)

// Convert ContinuousTensor to Tensor (needs make_tensor_external from tensor.h)
static_assert(
    CONTINUOUS_TENSOR_MAX_DIMS == RUNTIME_MAX_TENSOR_DIMS, "ContinuousTensor and runtime max dims must match"
);
inline Tensor from_tensor_arg(const ContinuousTensor &t, bool manual_dep = false, int32_t version = 0) {
    return make_tensor_external(
        reinterpret_cast<void *>(static_cast<uintptr_t>(t.data)), t.shapes, t.ndims, t.dtype, manual_dep, version
    );
}

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
    SubmitResult (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args);
    void (*add_dependency)(PTO2Runtime *rt, PTO2TaskId producer, PTO2TaskId consumer);
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char *func, const char *fmt, ...);
    void (*log_warn)(const char *func, const char *fmt, ...);
    void (*log_info)(const char *func, const char *fmt, ...);
    void (*log_debug)(const char *func, const char *fmt, ...);
    void (*log_always)(const char *func, const char *fmt, ...);
} PTO2RuntimeOps;

/**
 * Partial PTO2Runtime definition for orchestration.
 *
 * Only the ops pointer is visible.  The real struct (in pto_runtime2.h)
 * has the same first field, so accessing rt->ops through this definition
 * is well-defined (C struct layout guarantee).
 */
struct PTO2Runtime {
    const PTO2RuntimeOps *ops;
};

// =============================================================================
// Inline Convenience Wrappers (call through ops table)
// =============================================================================

static inline SubmitResult pto2_rt_submit_task(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args) {
    return rt->ops->submit_task(rt, mixed_kernels, args);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline SubmitResult pto2_rt_submit_aic_task(PTO2Runtime *rt, int32_t kernel_id, const Arg &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt->ops->submit_task(rt, mk, args);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline SubmitResult pto2_rt_submit_aiv_task(PTO2Runtime *rt, int32_t kernel_id, const Arg &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt->ops->submit_task(rt, mk, args);
}

/**
 * Add an explicit dependency: consumer waits for producer to complete.
 */
static inline void pto2_rt_add_dependency(PTO2Runtime *rt, PTO2TaskId producer, PTO2TaskId consumer) {
    rt->ops->add_dependency(rt, producer, consumer);
}

static inline void pto2_rt_scope_begin(PTO2Runtime *rt) { rt->ops->scope_begin(rt); }

static inline void pto2_rt_scope_end(PTO2Runtime *rt) { rt->ops->scope_end(rt); }

static inline void pto2_rt_orchestration_done(PTO2Runtime *rt) { rt->ops->orchestration_done(rt); }

static inline bool pto2_rt_is_fatal(PTO2Runtime *rt) { return rt->ops->is_fatal(rt); }

// =============================================================================
// Logging Macros for Orchestration (call through ops table)
// =============================================================================

#define LOG_ERROR(rt, fmt, ...) (rt)->ops->log_error(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_WARN(rt, fmt, ...) (rt)->ops->log_warn(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_INFO(rt, fmt, ...) (rt)->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(rt, fmt, ...) (rt)->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_ALWAYS(rt, fmt, ...) (rt)->ops->log_always(__FUNCTION__, fmt, ##__VA_ARGS__)

// =============================================================================
// C++ Scope Guards and Macros
// =============================================================================

/**
 * RAII Scope Guard (calls through ops table)
 */
class PTO2ScopeGuard {
public:  // NOLINT(whitespace/indent)
    explicit PTO2ScopeGuard(PTO2Runtime *rt) :
        rt_(rt) {
        rt_->ops->scope_begin(rt_);
    }
    ~PTO2ScopeGuard() { rt_->ops->scope_end(rt_); }

private:  // NOLINT(whitespace/indent)
    PTO2Runtime *rt_;
};

#define _PTO2_CONCATENATE_IMPL(x, y) x##y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)

#define PTO2_SCOPE_GUARD(rt) [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__)(rt)

/**
 * Scoped block macro:
 *   PTO2_SCOPE(rt) {
 *       pto2_rt_submit_task(rt, ...);
 *   }
 */
#define PTO2_SCOPE(rt) if (PTO2_SCOPE_GUARD(rt); true)

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

#endif  // SRC_A2A3_RUNTIME_AICPU_BUILD_GRAPH_ORCHESTRATION_PTO_ORCHESTRATION_API_H_
