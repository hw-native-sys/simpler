/**
 * PTO Runtime2 - Main Interface
 *
 * This is the main header for the PTO Runtime2 system.
 * It provides a unified API for task graph construction and execution.
 *
 * Key Features:
 * - Ring buffer based memory management (zero allocation overhead)
 * - Lazy invalidation TensorMap for dependency discovery
 * - Scope-based buffer lifecycle management
 * - Per-task spinlocks for concurrent fanout updates
 * - Orchestrator-Scheduler decoupling via shared memory
 *
 * Usage:
 *   1. Create runtime: pto2_runtime_create()
 *   2. Build task graph in orchestration function:
 *      - pto2_scope_begin() / pto2_scope_end()
 *      - pto2_submit_task()
 *   3. Mark orchestration complete: pto2_orchestrator_done()
 *   4. Destroy runtime: pto2_runtime_destroy()
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_RUNTIME2_H
#define PTO_RUNTIME2_H

#include "pto_runtime2_types.h"
#include "pto_submit_types.h"
#include "pto_shared_memory.h"
#include "pto_ring_buffer.h"
#include "pto_tensormap.h"
#include "pto_scheduler.h"
#include "pto_orchestrator.h"

// Maximum number of orchestrator threads supported
constexpr int PTO2_MAX_ORCH_THREADS = 4;

// =============================================================================
// Runtime Context
// =============================================================================

/**
 * Runtime execution mode
 */
enum PTO2RuntimeMode {
    PTO2_MODE_EXECUTE = 0,    // Execute tasks on workers
    PTO2_MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    PTO2_MODE_GRAPH_ONLY = 2  // Build graph only, no execution
};

/**
 * Function-pointer ops table for runtime operations.
 *
 * The orchestration .so calls runtime functions through this table
 * (via pto_orchestration_api.h inline wrappers), so it has zero link
 * dependencies on runtime .cpp files.
 */
typedef struct PTO2Runtime PTO2Runtime;  // forward declare for ops signatures

struct PTO2RuntimeOps {
    void (*submit_task)(PTO2Runtime* rt, const MixedKernels& mixed_kernels,
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
};

/**
 * PTO Runtime2 context
 *
 * Contains all state for orchestration and scheduling.
 * In simulated mode, runs in single process with shared address space.
 */
struct PTO2Runtime {
    // Ops table (first field — used by orchestration .so via function pointers)
    const PTO2RuntimeOps*   ops;

    // Components
    PTO2SharedMemoryHandle* sm_handle;
    PTO2OrchestratorState   orchestrators[PTO2_MAX_ORCH_THREADS];
    int                     orch_count;     // Number of active orchestrator states
    PTO2SchedulerState      scheduler;

    // GM Heap for output buffers
    void*                   gm_heap;
    uint64_t                  gm_heap_size;
    bool                    gm_heap_owned;  // True if we allocated it

    // Mode
    PTO2RuntimeMode         mode;

    // Statistics
    int64_t                 total_cycles;
};

// =============================================================================
// Runtime Lifecycle API
// =============================================================================

/**
 * Create a new runtime instance
 *
 * @param mode Execution mode
 * @return Runtime context, or NULL on failure
 */
PTO2Runtime* pto2_runtime_create(PTO2RuntimeMode mode);

/**
 * Create runtime with custom sizes
 *
 * @param mode             Execution mode
 * @param task_window_size Number of task slots
 * @param heap_size        Size of GM heap
 * @return Runtime context, or NULL on failure
 */
PTO2Runtime* pto2_runtime_create_custom(PTO2RuntimeMode mode,
                                         uint64_t task_window_size,
                                         uint64_t heap_size,
                                         int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE);

/**
 * Create runtime from existing shared memory and GM heap (e.g. on device).
 * Does not allocate sm_handle or gm_heap; caller owns them.
 *
 * @param mode      Execution mode
 * @param sm_handle Pre-created shared memory handle (e.g. from pto2_sm_create_from_buffer)
 * @param gm_heap   GM heap base for output buffers (or NULL if not used)
 * @param heap_size GM heap size in bytes
 * @return Runtime context, or NULL on failure
 */
PTO2Runtime* pto2_runtime_create_from_sm(PTO2RuntimeMode mode,
                                          PTO2SharedMemoryHandle* sm_handle,
                                          void* gm_heap,
                                          uint64_t heap_size,
                                          int orch_count = 1,
                                          int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE);

/**
 * Destroy runtime and free all resources
 */
void pto2_runtime_destroy(PTO2Runtime* rt);

/**
 * Set execution mode
 */
void pto2_runtime_set_mode(PTO2Runtime* rt, PTO2RuntimeMode mode);

/**
 * Set the orchestrator index for the current thread.
 * Must be called before any orchestration API calls on a given thread.
 */
void pto2_set_orch_thread_idx(int idx);

// =============================================================================
// Orchestration API (called by orchestration function)
// =============================================================================

/**
 * Begin a new scope
 *
 * All tasks submitted within this scope will have their lifetime
 * bounded by the scope. When scope_end() is called, the scope
 * releases its reference to all enclosed tasks.
 */
void pto2_rt_scope_begin(PTO2Runtime* rt);

/**
 * End current scope
 *
 * Releases scope reference for all tasks submitted since scope_begin().
 * Tasks whose refcount reaches zero will have their buffers released.
 */
void pto2_rt_scope_end(PTO2Runtime* rt);

/**
 * Mark orchestration as complete
 *
 * Signals that no more tasks will be submitted.
 */
void pto2_rt_orchestration_done(PTO2Runtime* rt);

/**
 * Scope helper macros for C
 *
 * These macros provide scope management for C code.
 * For C++, prefer using PTO2_SCOPE_GUARD or PTO2_SCOPE (see below).
 *
 * Usage (C):
 *   PTO2_SCOPE_BEGIN(rt);
 *   pto2_rt_submit_task(...);
 *   pto2_rt_submit_task(...);
 *   PTO2_SCOPE_END(rt);
 */
#define PTO2_SCOPE_BEGIN(rt) pto2_rt_scope_begin(rt)
#define PTO2_SCOPE_END(rt)   pto2_rt_scope_end(rt)

/**
 * RAII Scope Guard for C++
 *
 * PTO2ScopeGuard is a C++ RAII wrapper that automatically manages scope lifetime.
 * It calls pto2_rt_scope_begin() on construction and pto2_rt_scope_end() on destruction,
 * ensuring proper cleanup even in error paths.
 *
 * Usage Option 1 - Direct instantiation (recommended):
 *   PTO2ScopeGuard scope_guard(rt);
 *   pto2_rt_submit_task(...);
 *   pto2_rt_submit_task(...);
 *   // scope automatically ends here when scope_guard destructor is called
 *
 * Usage Option 2 - Macro for anonymous guard:
 *   PTO2_SCOPE_GUARD(rt);
 *   pto2_rt_submit_task(...);
 *   // scope automatically ends at end of current block
 *
 * Usage Option 3 - Scoped block with if statement:
 *   PTO2_SCOPE(rt) {
 *       pto2_rt_submit_task(...);
 *       pto2_rt_submit_task(...);
 *   } // scope automatically ends here
 *
 * Benefits:
 * - Exception-safe: scope ends even if exceptions are thrown
 * - Error-safe: no need to manually call PTO2_SCOPE_END in error paths
 * - Cleaner code: less boilerplate, automatic cleanup
 * - Less error-prone: impossible to forget scope cleanup
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

/**
 * Macro to create an anonymous scope guard with a unique name.
 * The [[maybe_unused]] attribute suppresses warnings if the guard
 * variable is not explicitly used.
 *
 * Example:
 *   PTO2_SCOPE_GUARD(rt);
 *   pto2_rt_submit_task(...);
 */
#define _PTO2_CONCATENATE_IMPL(x, y) x ## y
#define _PTO2_CONCATENATE(x, y) _PTO2_CONCATENATE_IMPL(x, y)
#define PTO2_SCOPE_GUARD(rt) [[maybe_unused]] PTO2ScopeGuard _PTO2_CONCATENATE(scope_guard_, __COUNTER__)(rt)

/**
 * Macro to create a scoped block with automatic scope management.
 * Uses if-statement initialization (C++17) to create guard and execute block.
 *
 * Example:
 *   PTO2_SCOPE(rt) {
 *       pto2_rt_submit_task(...);
 *   } // scope automatically ends here
 */
#define PTO2_SCOPE(rt) if (PTO2_SCOPE_GUARD(rt); true)

/**
 * Slim config struct exported by orchestration .so via aicpu_orchestration_config().
 * Shared definition with pto_orchestration_api.h (same layout, guarded).
 */
#ifndef PTO2_ORCHESTRATION_CONFIG_DEFINED
#define PTO2_ORCHESTRATION_CONFIG_DEFINED
struct PTO2OrchestrationConfig {
    int         expected_arg_count;
};
#endif

#endif // PTO_RUNTIME2_H
