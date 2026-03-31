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

#ifndef PTO_ORCHESTRATION_API_H
#define PTO_ORCHESTRATION_API_H

#include <stdint.h>
#include <stdbool.h>
#include <stddef.h>

// Type headers needed by orchestration
#include "tensor.h"             // Tensor
#include "pto_types.h"          // PTOParam, PTOTensorEntry, PTOParamType, PTO2AsyncEngine
#include "pto_submit_types.h"   // MixedKernels, INVALID_KERNEL_ID, subtask slots

// =============================================================================
// Tensor Factory Helpers
// =============================================================================

/**
 * Create a Tensor for pre-allocated external memory.
 */
static inline Tensor make_tensor_external(void* addr,
    const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    bool manual_dep = false,
    int32_t version = 0) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return Tensor(addr, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version,
                  /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true, manual_dep);
}

/**
 * Create a Tensor for runtime-allocated output (addr=0).
 * NO memory allocation: only records dtype, shape, and buffer.size in the Tensor struct.
 * The runtime allocates from the heap ring and fills buffer.addr during pto2_submit_task
 * when this tensor is passed as OUTPUT param. No buffer content is ever copied.
 */
static inline Tensor make_tensor(const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype = DataType::FLOAT32,
    bool manual_dep = false,
    int32_t version = 0) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return Tensor(0, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version,
                  /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true, manual_dep);
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
PTO2Runtime* pto2_framework_current_runtime(void);
void pto2_framework_bind_runtime(PTO2Runtime* rt);

#ifdef __cplusplus
}
#endif

/**
 * Function-pointer table for runtime operations.
 * Populated by the runtime; called by orchestration through inline wrappers.
 */
typedef struct PTO2RuntimeOps {
    void (*submit_task)(PTO2Runtime* rt, const MixedKernels& mixed_kernels,
                        const PTOParam& params);
    uint64_t (*get_async_context)(PTO2Runtime* rt, PTO2AsyncEngine engine);
    uint64_t (*alloc_cq)(PTO2Runtime* rt);
    void (*scope_begin)(PTO2Runtime* rt);
    void (*scope_end)(PTO2Runtime* rt);
    void (*orchestration_done)(PTO2Runtime* rt);
    bool (*is_fatal)(PTO2Runtime* rt);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char* func, const char* fmt, ...);
    void (*log_warn)(const char* func, const char* fmt, ...);
    void (*log_info)(const char* func, const char* fmt, ...);
    void (*log_debug)(const char* func, const char* fmt, ...);
    void (*log_always)(const char* func, const char* fmt, ...);
} PTO2RuntimeOps;

/**
 * Partial PTO2Runtime definition for orchestration.
 *
 * Only the ops pointer is visible.  The real struct (in pto_runtime2.h)
 * has the same first field, so accessing rt->ops through this definition
 * is well-defined (C struct layout guarantee).
 */
struct PTO2Runtime {
    const PTO2RuntimeOps* ops;
};

// =============================================================================
// Inline Convenience Wrappers (call through ops table)
// =============================================================================

static inline PTO2Runtime* pto2_current_runtime() {
    return pto2_framework_current_runtime();
}

static inline void pto2_rt_submit_task(const MixedKernels& mixed_kernels,
                                       const PTOParam& params) {
    PTO2Runtime* rt = pto2_current_runtime();
    rt->ops->submit_task(rt, mixed_kernels, params);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline void pto2_rt_submit_aic_task(int32_t kernel_id, const PTOParam& params) {
    PTO2Runtime* rt = pto2_current_runtime();
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    rt->ops->submit_task(rt, mk, params);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline void pto2_rt_submit_aiv_task(int32_t kernel_id, const PTOParam& params) {
    PTO2Runtime* rt = pto2_current_runtime();
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    rt->ops->submit_task(rt, mk, params);
}
static inline uint64_t pto2_rt_get_async_context(PTO2AsyncEngine engine) {
    PTO2Runtime* rt = pto2_current_runtime();
    return rt->ops->get_async_context(rt, engine);
}

static inline uint64_t pto2_rt_get_async_context(PTO2Runtime* rt, PTO2AsyncEngine engine) {
    return rt->ops->get_async_context(rt, engine);
}

static inline uint64_t pto2_rt_get_sdma_context() {
    return pto2_rt_get_async_context(PTO2_ASYNC_ENGINE_SDMA);
}

static inline uint64_t pto2_rt_get_sdma_context(PTO2Runtime* rt) {
    return pto2_rt_get_async_context(rt, PTO2_ASYNC_ENGINE_SDMA);
}

// =============================================================================
// CQ Model: Deferred Completion Wrappers
// =============================================================================

/**
 * Allocate a zeroed per-task completion queue from the runtime pool.
 * Returns the GM address (cast to uint64_t) or 0 on failure.
 */
static inline uint64_t pto2_rt_alloc_cq() {
    PTO2Runtime* rt = pto2_current_runtime();
    return rt->ops->alloc_cq(rt);
}

static inline uint64_t pto2_rt_alloc_cq(PTO2Runtime* rt) {
    return rt->ops->alloc_cq(rt);
}

/**
 * Submit an AIV task with deferred completion (CQ model).
 *
 * The kernel decides at runtime how many async completions it has
 * and writes them into the completion queue. The scheduler discovers
 * completions by reading the CQ after all subtasks return.
 *
 * The CQ address is automatically appended as the last scalar
 * so the kernel can access it from args[].
 */
static inline void pto2_rt_submit_aiv_task_deferred(int32_t kernel_id,
                                                     PTOParam& params,
                                                     uint64_t cq_addr) {
    PTO2Runtime* rt = pto2_current_runtime();
    params.complete_in_future = true;
    params.cq_addr = cq_addr;
    params.add_scalar(cq_addr);
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    rt->ops->submit_task(rt, mk, params);
}

static inline void pto2_rt_submit_aiv_task_deferred(PTO2Runtime* rt,
                                                     int32_t kernel_id,
                                                     PTOParam& params,
                                                     uint64_t cq_addr) {
    params.complete_in_future = true;
    params.cq_addr = cq_addr;
    params.add_scalar(cq_addr);
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    rt->ops->submit_task(rt, mk, params);
}

static inline void pto2_rt_submit_aic_task_deferred(int32_t kernel_id,
                                                     PTOParam& params,
                                                     uint64_t cq_addr) {
    PTO2Runtime* rt = pto2_current_runtime();
    params.complete_in_future = true;
    params.cq_addr = cq_addr;
    params.add_scalar(cq_addr);
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    rt->ops->submit_task(rt, mk, params);
}

static inline void pto2_rt_submit_task_deferred(const MixedKernels& mixed_kernels,
                                                 PTOParam& params,
                                                 uint64_t cq_addr) {
    PTO2Runtime* rt = pto2_current_runtime();
    params.complete_in_future = true;
    params.cq_addr = cq_addr;
    params.add_scalar(cq_addr);
    rt->ops->submit_task(rt, mixed_kernels, params);
}

/**
 * Submit a notification-wait deferred task and return a dependency token.
 *
 * Encapsulates the boilerplate for creating a NotifyWait task:
 *   1. Allocate a CQ
 *   2. Create a 1-element dummy output tensor (dependency token)
 *   3. Submit a deferred AIV task with (counter_addr, expected_value, cq_addr)
 *
 * The returned token tensor should be added as an input to any downstream
 * task that depends on the notification completing.
 *
 * @param kernel_id      func_id of the NotifyWait kernel
 * @param counter_addr   GM address of the notification counter (int32*)
 * @param expected_value  threshold: task completes when *counter >= expected
 * @return dependency token tensor (add as input to downstream tasks)
 */
static inline Tensor pto2_rt_submit_notification_wait_task(
    int32_t kernel_id,
    uint64_t counter_addr,
    uint32_t expected_value) {
    uint64_t cq_addr = pto2_rt_alloc_cq();

    uint32_t dummy_shape[1] = { 1 };
    Tensor token = make_tensor(dummy_shape, 1, DataType::INT32);

    PTOParam params;
    params.add_output(token);
    params.add_scalar(counter_addr);
    params.add_scalar(static_cast<uint64_t>(expected_value));
    pto2_rt_submit_aiv_task_deferred(kernel_id, params, cq_addr);

    return token;
}

static inline void pto2_rt_scope_begin() {
    PTO2Runtime* rt = pto2_current_runtime();
    rt->ops->scope_begin(rt);
}

static inline void pto2_rt_scope_end() {
    PTO2Runtime* rt = pto2_current_runtime();
    rt->ops->scope_end(rt);
}

static inline void pto2_rt_orchestration_done() {
    PTO2Runtime* rt = pto2_current_runtime();
    rt->ops->orchestration_done(rt);
}

static inline bool pto2_rt_is_fatal() {
    PTO2Runtime* rt = pto2_current_runtime();
    return rt->ops->is_fatal(rt);
}

// =============================================================================
// Logging Macros for Orchestration (call through ops table)
// =============================================================================

#define LOG_ERROR(fmt, ...) pto2_current_runtime()->ops->log_error(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_WARN(fmt, ...)  pto2_current_runtime()->ops->log_warn(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_INFO(fmt, ...)  pto2_current_runtime()->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(fmt, ...) pto2_current_runtime()->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_ALWAYS(fmt, ...) pto2_current_runtime()->ops->log_always(__FUNCTION__, fmt, ##__VA_ARGS__)

// =============================================================================
// C++ Scope Guards and Macros
// =============================================================================

/**
 * RAII Scope Guard (calls through ops table)
 */
class PTO2ScopeGuard {
public:
    PTO2ScopeGuard() : rt_(pto2_current_runtime()) {
        rt_->ops->scope_begin(rt_);
    }
    ~PTO2ScopeGuard() {
        rt_->ops->scope_end(rt_);
    }
private:
    PTO2Runtime* rt_;
};

#define _PTO2_CONCATENATE_IMPL(x, y) x ## y
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
    int         expected_arg_count;
};
#endif

#endif // PTO_ORCHESTRATION_API_H
