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
#include "pto_types.h"          // PTOParam, PTOTensorEntry, PTOParamType
#include "tensor.h"             // Tensor struct
#include "pto_submit_types.h"   // MixedKernels, INVALID_KERNEL_ID, subtask slots

// Multi-ring: number of independent ring layers (HeapRing + TaskRing + DepPool per layer)
// Scope depth maps to ring index via: min(scope_depth, PTO2_MAX_RING_DEPTH - 1)
#define PTO2_MAX_RING_DEPTH 4

// Thread-local scope depth for tensor factory functions.
// Incremented/decremented by PTO2ScopeGuard and standalone scope wrappers.
// Tensor ring selection clamps this depth to the runtime's valid ring range.
static thread_local uint8_t __pto2_ring_id = 0;

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
    void (*submit_task)(PTO2Runtime* rt, const MixedKernels& mixed_kernels,
                        const PTOParam& params);
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

static inline void pto2_rt_submit_task(PTO2Runtime* rt, const MixedKernels& mixed_kernels,
                                        const PTOParam& params) {
    rt->ops->submit_task(rt, mixed_kernels, params);
}

/**
 * Convenience wrapper: submit an AIC-only task.
 */
static inline void pto2_rt_submit_aic_task(PTO2Runtime* rt, int32_t kernel_id,
                                            const PTOParam& params) {
    MixedKernels mk;
    mk.aic_kernel_id = kernel_id;
    rt->ops->submit_task(rt, mk, params);
}

/**
 * Convenience wrapper: submit an AIV-only task (uses AIV0 slot).
 */
static inline void pto2_rt_submit_aiv_task(PTO2Runtime* rt, int32_t kernel_id,
                                            const PTOParam& params) {
    MixedKernels mk;
    mk.aiv0_kernel_id = kernel_id;
    rt->ops->submit_task(rt, mk, params);
}

static inline void pto2_rt_scope_begin(PTO2Runtime* rt) {
    rt->ops->scope_begin(rt);
    __pto2_ring_id++;
}

static inline void pto2_rt_scope_end(PTO2Runtime* rt) {
    rt->ops->scope_end(rt);
    __pto2_ring_id--;
}

static inline void pto2_rt_orchestration_done(PTO2Runtime* rt) {
    rt->ops->orchestration_done(rt);
}

static inline bool pto2_rt_is_fatal(PTO2Runtime* rt) {
    return rt->ops->is_fatal(rt);
}

// =============================================================================
// Tensor Factory Functions
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
                  /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true, manual_dep,
                  TENSOR_RING_ID_NONE);
}

static inline Tensor make_tensor_with_ring(const uint32_t shapes[],
    uint32_t ndims,
    DataType dtype,
    bool manual_dep,
    int32_t version,
    uint8_t ring_id) {
    static uint32_t zero_offsets[RUNTIME_MAX_TENSOR_DIMS] = {};
    uint64_t total = 1;
    for (uint32_t i = 0; i < ndims; i++) {
        total *= shapes[i];
    }
    return Tensor(0, total * get_element_size(dtype), shapes, shapes, zero_offsets, ndims, dtype, version,
                  /*is_all_offset_zero=*/true, /*is_raw_eq_shapes=*/true, manual_dep, ring_id);
}

static inline uint8_t current_tensor_ring_id() {
    return __pto2_ring_id < PTO2_MAX_RING_DEPTH ? __pto2_ring_id : PTO2_MAX_RING_DEPTH - 1;
}

/**
 * Create a Tensor for runtime-allocated output (addr=0).
 * Uses the thread-local scope depth set by PTO2ScopeGuard, clamped to the
 * runtime ring range to match PTO2OrchestratorState::current_ring_id().
 */
static inline Tensor make_tensor(const uint32_t shapes[], uint32_t ndims,
    DataType dtype = DataType::FLOAT32, bool manual_dep = false,
    int32_t version = 0) {
    return make_tensor_with_ring(shapes, ndims, dtype, manual_dep, version, current_tensor_ring_id());
}

// =============================================================================
// Logging Macros for Orchestration (call through ops table)
// =============================================================================

#define LOG_ERROR(rt, fmt, ...) (rt)->ops->log_error(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_WARN(rt, fmt, ...)  (rt)->ops->log_warn(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_INFO(rt, fmt, ...)  (rt)->ops->log_info(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_DEBUG(rt, fmt, ...) (rt)->ops->log_debug(__FUNCTION__, fmt, ##__VA_ARGS__)
#define LOG_ALWAYS(rt, fmt, ...) (rt)->ops->log_always(__FUNCTION__, fmt, ##__VA_ARGS__)

// =============================================================================
// C++ Scope Guards and Macros
// =============================================================================

/**
 * RAII Scope Guard (calls through ops table)
 */
class PTO2ScopeGuard {
public:
    PTO2ScopeGuard(PTO2Runtime* rt) : rt_(rt) {
        pto2_rt_scope_begin(rt_);
    }
    ~PTO2ScopeGuard() {
        pto2_rt_scope_end(rt_);
    }
private:
    PTO2Runtime* rt_;
};

#define _PTO2_CONCATENATE_IMPL(x, y) x ## y
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
    int         expected_arg_count;
};
#endif

#endif // PTO_ORCHESTRATION_API_H
