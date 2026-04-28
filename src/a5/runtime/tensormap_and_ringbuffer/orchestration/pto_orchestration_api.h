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

#include <type_traits>

// Type headers needed by orchestration
#include "pto_runtime2_types.h"  // PTO2_ERROR_*
#include "pto_submit_types.h"    // MixedKernels, INVALID_KERNEL_ID, subtask slots
#include "pto_types.h"           // Arg, TaskOutputTensors, TensorArgType
#include "task_args.h"           // ChipStorageTaskArgs, ContinuousTensor
#include "tensor.h"              // Tensor, TensorCreateInfo

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
PTO2Runtime *framework_current_runtime(void);
void framework_bind_runtime(PTO2Runtime *rt);

#ifdef __cplusplus
}
#endif

/**
 * Function-pointer table for runtime operations.
 * Populated by the runtime; called by orchestration through inline wrappers.
 */
typedef struct PTO2RuntimeOps {
    TaskOutputTensors (*submit_task)(
        PTO2Runtime *rt, const MixedKernels &mixed_kernels, const Arg &args, bool complete_in_future
    );
    void (*scope_begin)(PTO2Runtime *rt);
    void (*scope_end)(PTO2Runtime *rt);
    void (*orchestration_done)(PTO2Runtime *rt);
    bool (*is_fatal)(PTO2Runtime *rt);
    void (*report_fatal)(PTO2Runtime *rt, int32_t error_code, const char *func, const char *fmt, ...);

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
    TaskOutputTensors (*alloc_tensors)(PTO2Runtime *rt, const Arg &args);
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

static inline TaskOutputTensors alloc_tensors(const Arg &args) {
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
    Arg args;
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
    Arg args;
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

static inline TaskOutputTensors
rt_submit_task_impl(const MixedKernels &mixed_kernels, const Arg &args, bool complete_in_future) {
    PTO2Runtime *rt = current_runtime();
    if (rt->ops->is_fatal(rt)) {
        return TaskOutputTensors{};
    }
    return rt->ops->submit_task(rt, mixed_kernels, args, complete_in_future);
}

static inline TaskOutputTensors rt_submit_task(const MixedKernels &mixed_kernels, const Arg &args) {
    return rt_submit_task_impl(mixed_kernels, args, false);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline TaskOutputTensors rt_submit_aic_task(int32_t kernel_id, const Arg &args) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    return rt_submit_task_impl(mk, args, false);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline TaskOutputTensors rt_submit_aiv_task(int32_t kernel_id, const Arg &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task_impl(mk, args, false);
}

static inline TaskOutputTensors rt_submit_aiv_task_deferred(int32_t kernel_id, const Arg &args) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    return rt_submit_task_impl(mk, args, true);
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
#define LOG_INFO(fmt, ...) current_runtime()->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) current_runtime()->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_ALWAYS(fmt, ...) current_runtime()->ops->log_always(__FUNCTION__, fmt, ##__VA_ARGS__)

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
    explicit PTO2ScopeGuard(PTO2ScopeMode mode = PTO2ScopeMode::AUTO) :
        rt_(current_runtime()) {
        if (!rt_->ops->is_fatal(rt_)) {
            rt_->pending_scope_mode = mode;
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

#define _PTO2_CONCATENATE_IMPL(x, y) x##y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)

#define PTO2_SCOPE_GUARD(...) \
    [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__) { __VA_ARGS__ }

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

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_ORCHESTRATION_PTO_ORCHESTRATION_API_H_
