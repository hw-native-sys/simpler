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
#include "pto_types.h"          // PTOParam, make_input_param, make_output_param, etc.
#include "tensor.h"             // Tensor, make_tensor, make_tensor_external

// Worker type constants (duplicated from pto_runtime2_types.h to avoid
// pulling in the full types header with its internal structures)
typedef enum {
    PTO2_WORKER_CUBE = 0,
    PTO2_WORKER_VECTOR = 1,
    PTO2_WORKER_AI_CPU = 2,
    PTO2_WORKER_ACCELERATOR = 3,
    PTO2_NUM_WORKER_TYPES = 4
} PTO2WorkerType;

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
    void (*submit_task)(PTO2Runtime* rt, int32_t kernel_id,
                        PTO2WorkerType worker_type,
                        PTOParam* params, int32_t num_params);
    void (*scope_begin)(PTO2Runtime* rt);
    void (*scope_end)(PTO2Runtime* rt);
    void (*orchestration_done)(PTO2Runtime* rt);

    // Logging (populated by runtime, called by orchestration)
    void (*log_error)(const char* func, const char* fmt, ...);
    void (*log_warn)(const char* func, const char* fmt, ...);
    void (*log_info)(const char* func, const char* fmt, ...);
    void (*log_debug)(const char* func, const char* fmt, ...);
    void (*log_always)(const char* func, const char* fmt, ...);

    TensorPool* (*get_tensor_pool)(PTO2Runtime* rt);
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

static inline void pto2_rt_submit_task(PTO2Runtime* rt, int32_t kernel_id,
                                        PTO2WorkerType worker_type,
                                        PTOParam* params, int32_t num_params) {
    rt->ops->submit_task(rt, kernel_id, worker_type, params, num_params);
}

static inline void pto2_rt_scope_begin(PTO2Runtime* rt) {
    rt->ops->scope_begin(rt);
}

static inline void pto2_rt_scope_end(PTO2Runtime* rt) {
    rt->ops->scope_end(rt);
}

static inline void pto2_rt_orchestration_done(PTO2Runtime* rt) {
    rt->ops->orchestration_done(rt);
}

static inline void pto2_rt_init_tensor_pool(PTO2Runtime* rt) {
    TensorPool::set_instance(rt->ops->get_tensor_pool(rt));
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
