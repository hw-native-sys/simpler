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
#include "pto_shared_memory.h"
#include "pto_ring_buffer.h"
#include "pto_tensormap.h"
#include "pto_scheduler.h"
#include "pto_orchestrator.h"

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
    PTO2OrchestratorState   orchestrator;
    PTO2SchedulerState      scheduler;

    // GM Heap for output buffers
    void*                   gm_heap;
    int32_t                 gm_heap_size;
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
 * @param dep_list_size    Size of dependency list pool
 * @return Runtime context, or NULL on failure
 */
PTO2Runtime* pto2_runtime_create_custom(PTO2RuntimeMode mode,
                                         int32_t task_window_size,
                                         int32_t heap_size,
                                         int32_t dep_list_size);

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
                                          int32_t heap_size);

/**
 * Destroy runtime and free all resources
 */
void pto2_runtime_destroy(PTO2Runtime* rt);

/**
 * Reset runtime for reuse (keeps allocations, clears state)
 */
void pto2_runtime_reset(PTO2Runtime* rt);

/**
 * Set execution mode
 */
void pto2_runtime_set_mode(PTO2Runtime* rt, PTO2RuntimeMode mode);

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
 * Configuration for orchestration entry point setup.
 *
 * Groups all parameters needed by PTO2OrchestrationGuard so the
 * PTO2_ORCHESTRATION macro stays concise.
 *
 * Example:
 *   PTO2OrchestrationBeginInfo begin_info{
 *       .sm_ptr             = sm_ptr,
 *       .args               = args,
 *       .arg_count          = arg_count,
 *       .expected_arg_count = 7,
 *       .task_window_size   = 16384,
 *       .dep_list_pool_size = 65536,
 *       .heap_size          = 256 * 1024,
 *       .gm_heap_ptr        = s_gm_heap_stub,
 *   };
 */
struct PTO2OrchestrationBeginInfo {
    void*       sm_ptr;
    uint64_t*   args;
    int         arg_count;
    int         expected_arg_count;
    int32_t     task_window_size;
    int32_t     dep_list_pool_size;
    int32_t     heap_size;
    void*       gm_heap_ptr = nullptr;
};

/**
 * RAII guard for orchestration entry/exit boilerplate.
 *
 * Handles validation, shared memory creation, GM heap extraction,
 * runtime creation (constructor) and orchestration-done signaling,
 * runtime destruction (destructor).
 *
 * On any init failure the destructor still signals orchestrator_done = 1.
 *
 * Usage with PTO2_ORCHESTRATION macro:
 *   PTO2_ORCHESTRATION(rt, begin_info) {
 *       pto2_rt_submit_task(rt, ...);  // implicitly inside outer scope
 *       PTO2_SCOPE(rt) { ... }        // nested inner scope
 *   }
 */
class PTO2OrchestrationGuard {
public:
    explicit PTO2OrchestrationGuard(const PTO2OrchestrationBeginInfo& begin_info)
        : header_(nullptr), rt_(nullptr)
    {
        if (!begin_info.sm_ptr || !begin_info.args || begin_info.arg_count < begin_info.expected_arg_count) {
            if (begin_info.sm_ptr) {
                header_ = static_cast<PTO2SharedMemoryHeader*>(begin_info.sm_ptr);
            }
            return;
        }
        header_ = static_cast<PTO2SharedMemoryHeader*>(begin_info.sm_ptr);

        int32_t sm_size = pto2_sm_calculate_size(begin_info.task_window_size,
                                                  begin_info.dep_list_pool_size);
        PTO2SharedMemoryHandle* sm_handle =
            pto2_sm_create_from_buffer(begin_info.sm_ptr, sm_size,
                                       begin_info.task_window_size,
                                       begin_info.heap_size,
                                       begin_info.dep_list_pool_size);
        if (!sm_handle) return;

        void*   gm_heap      = begin_info.gm_heap_ptr;
        int32_t gm_heap_size = begin_info.heap_size;
        if (begin_info.arg_count >= 2) {
            uint64_t heap_arg  = begin_info.args[begin_info.arg_count - 2];
            uint64_t size_arg  = begin_info.args[begin_info.arg_count - 1];
            if (heap_arg != 0 && size_arg != 0) {
                gm_heap      = reinterpret_cast<void*>(static_cast<uintptr_t>(heap_arg));
                gm_heap_size = static_cast<int32_t>(size_arg & 0x7FFFFFFF);
            }
        }

        rt_ = pto2_runtime_create_from_sm(PTO2_MODE_EXECUTE, sm_handle,
                                           gm_heap, gm_heap_size);
        if (!rt_) {
            pto2_sm_destroy(sm_handle);
            return;
        }
    }

    ~PTO2OrchestrationGuard() {
        if (rt_) {
            pto2_rt_orchestration_done(rt_);
            pto2_runtime_destroy(rt_);
        }
        if (header_) {
            header_->orchestrator_done = 1;
        }
    }

    bool valid() const { return rt_ != nullptr; }
    PTO2Runtime* runtime() const { return rt_; }

    PTO2OrchestrationGuard(const PTO2OrchestrationGuard&) = delete;
    PTO2OrchestrationGuard& operator=(const PTO2OrchestrationGuard&) = delete;

private:
    PTO2SharedMemoryHeader* header_;
    PTO2Runtime* rt_;
};

/**
 * Macro for orchestration entry with automatic setup/teardown.
 * Uses C++17 if-init to create the guard, expose the runtime pointer,
 * and implicitly open an outer scope (PTO2_SCOPE).
 * The block is skipped if initialization fails.
 *
 * Example:
 *   PTO2_ORCHESTRATION(rt, begin_info) {
 *       pto2_rt_submit_task(rt, ...);
 *       PTO2_SCOPE(rt) { ... }  // nested inner scope
 *   }
 *   // scope end + orchestrator_done + runtime destroy all automatic
 */
#define _PTO2_ORCHESTRATION_IMPL(rt_var, guard_name, begin_info)  \
    if ([[maybe_unused]] PTO2OrchestrationGuard guard_name(begin_info); \
        PTO2Runtime* rt_var = guard_name.runtime()) \
        PTO2_SCOPE(rt_var)

#define PTO2_ORCHESTRATION(rt_var, begin_info) \
    _PTO2_ORCHESTRATION_IMPL(rt_var, _PTO2_CONCATENATE(_pto2_orch_, __COUNTER__), begin_info)

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
