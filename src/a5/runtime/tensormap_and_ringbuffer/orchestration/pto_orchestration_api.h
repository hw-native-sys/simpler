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

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_ORCHESTRATION_PTO_ORCHESTRATION_API_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_ORCHESTRATION_PTO_ORCHESTRATION_API_H_

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

// Type headers needed by orchestration
#include "pto_submit_types.h"  // MixedKernels, INVALID_KERNEL_ID, subtask slots  // NOLINT(build/include_subdir)
#include "pto_types.h"         // Arg, TaskOutputTensors, TensorArgType  // NOLINT(build/include_subdir)
#include "task_args.h"         // ChipStorageTaskArgs, ContinuousTensor  // NOLINT(build/include_subdir)
#include "tensor.h"            // Tensor, TensorCreateInfo  // NOLINT(build/include_subdir)

// =============================================================================
// Tensor Factory Helpers
// =============================================================================

/**
 * Create a Tensor for pre-allocated external memory.
 */
inline Tensor make_tensor_external(
    void *addr, const uint32_t shapes[], uint32_t ndims, DataType dtype = DataType::FLOAT32, bool manual_dep = false,
    int32_t version = 0
) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return {
        addr,
        total * get_element_size(dtype),
        shapes,
        shapes,
        zero_offsets,
        ndims,
        dtype,
        version,
        /*is_all_offset_zero=*/true,
        /*is_raw_eq_shapes=*/true,
        manual_dep
    };
}

// Convert ContinuousTensor to Tensor
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

#ifdef __cplusplus
extern "C" {
#endif

/**
 * Framework-internal TLS bridge.
 *
 * The executor binds the current thread's runtime before invoking
 * aicpu_orchestration_entry(), so orchestration helpers can fetch the
 * current PTO2Runtime without explicit parameter threading.
 */
PTO2Runtime *pto2_framework_current_runtime(void);
void pto2_framework_bind_runtime(PTO2Runtime *rt);

#ifdef __cplusplus
}
#endif

/**
 * Function-pointer table for runtime operations.
 * Populated by the runtime; called by orchestration through inline wrappers.
 */
typedef struct PTO2RuntimeOps {
    TaskOutputTensors (*submit_task)(PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args);
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

    // Cross-layer data access (orchestration reads/writes tensor values via runtime)
    // Placed after logging to avoid shifting hot-path field offsets.
    uint64_t (*get_tensor_data)(PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[]);
    void (*set_tensor_data)(
        PTO2Runtime *rt, const Tensor &tensor, uint32_t ndims, const uint32_t indices[], uint64_t value
    );
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

static inline PTO2Runtime *pto2_current_runtime() { return pto2_framework_current_runtime(); }

static inline TaskOutputTensors pto2_rt_submit_task(const MixedKernels &mixed_kernels, const Arg &args) {
    PTO2Runtime *rt = pto2_current_runtime();
    return rt->ops->submit_task(rt, mixed_kernels, args);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline TaskOutputTensors pto2_rt_submit_aic_task(int32_t kernel_id, const Arg &args) {
    PTO2Runtime *rt = pto2_current_runtime();
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt->ops->submit_task(rt, mk, args);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline TaskOutputTensors pto2_rt_submit_aiv_task(int32_t kernel_id, const Arg &args) {
    PTO2Runtime *rt = pto2_current_runtime();
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt->ops->submit_task(rt, mk, args);
}

static inline void pto2_rt_scope_begin() {
    PTO2Runtime *rt = pto2_current_runtime();
    rt->ops->scope_begin(rt);
}

static inline void pto2_rt_scope_end() {
    PTO2Runtime *rt = pto2_current_runtime();
    rt->ops->scope_end(rt);
}

static inline void pto2_rt_orchestration_done() {
    PTO2Runtime *rt = pto2_current_runtime();
    rt->ops->orchestration_done(rt);
}

static inline bool pto2_rt_is_fatal() {
    PTO2Runtime *rt = pto2_current_runtime();
    return rt->ops->is_fatal(rt);
}

// =============================================================================
// Logging Macros for Orchestration (call through ops table)
// =============================================================================

#define LOG_ERROR(fmt, ...) pto2_current_runtime()->ops->log_error(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...) pto2_current_runtime()->ops->log_warn(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...) pto2_current_runtime()->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) pto2_current_runtime()->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_ALWAYS(fmt, ...) pto2_current_runtime()->ops->log_always(__FUNCTION__, fmt, ##__VA_ARGS__)

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
static inline T get_tensor_data(const Tensor &tensor, uint32_t ndims, const uint32_t indices[]) {
    PTO2Runtime *rt = pto2_current_runtime();
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
 * For runtime-created outputs, call this only on the Tensor returned by
 * add_output(TensorCreateInfo) after submit returns.
 */
template <typename T = uint64_t>
static inline void set_tensor_data(const Tensor &tensor, uint32_t ndims, const uint32_t indices[], T value) {
    PTO2Runtime *rt = pto2_current_runtime();
    rt->ops->set_tensor_data(rt, tensor, ndims, indices, to_u64(value));
}

// =============================================================================
// C++ Scope Guards and Macros
// =============================================================================

/**
 * RAII Scope Guard (calls through ops table)
 */
class PTO2ScopeGuard {
public:  // NOLINT(whitespace/indent)
    PTO2ScopeGuard() :
        rt_(pto2_current_runtime()) {
        rt_->ops->scope_begin(rt_);
    }
    ~PTO2ScopeGuard() { rt_->ops->scope_end(rt_); }

private:  // NOLINT(whitespace/indent)
    PTO2Runtime *rt_;
};

#define _PTO2_CONCATENATE_IMPL(x, y) x##y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)

#define PTO2_SCOPE_GUARD() [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__)

/**
 * Scoped block macro:
 *   PTO2_SCOPE() {
 *       pto2_rt_submit_task(...);
 *   }
 */
#define PTO2_SCOPE() if (PTO2_SCOPE_GUARD(); true)

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

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_ORCHESTRATION_PTO_ORCHESTRATION_API_H_
