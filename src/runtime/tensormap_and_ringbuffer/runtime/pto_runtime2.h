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
 *   4. Execute or simulate: pto2_runtime_execute() / pto2_runtime_simulate()
 *   5. Destroy runtime: pto2_runtime_destroy()
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

#ifdef __cplusplus
extern "C" {
#endif

// =============================================================================
// Runtime Context
// =============================================================================

/**
 * Runtime execution mode
 */
typedef enum {
    PTO2_MODE_EXECUTE = 0,    // Execute tasks on workers
    PTO2_MODE_SIMULATE = 1,   // Simulate task execution with cycle counting
    PTO2_MODE_GRAPH_ONLY = 2  // Build graph only, no execution
} PTO2RuntimeMode;

/**
 * PTO Runtime2 context
 * 
 * Contains all state for orchestration and scheduling.
 * In simulated mode, runs in single process with shared address space.
 */
typedef struct PTO2Runtime {
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
    
} PTO2Runtime;

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
 * Submit a task
 * 
 * @param rt          Runtime context
 * @param kernel_id   InCore function ID
 * @param worker_type Target worker type
 * @param func_ptr    Function pointer (optional)
 * @param func_name   Function name (for debugging)
 * @param params      Array of task parameters
 * @param num_params  Number of parameters
 * @return Task ID, or -1 on failure
 */
int32_t pto2_rt_submit_task(PTO2Runtime* rt,
                             int32_t kernel_id,
                             PTO2WorkerType worker_type,
                             void* func_ptr,
                             const char* func_name,
                             PTO2TaskParam* params,
                             int32_t num_params);

/**
 * Simplified task submission (auto-detect worker type)
 */
int32_t pto2_rt_submit(PTO2Runtime* rt,
                        const char* func_name,
                        void* func_ptr,
                        PTO2TaskParam* params,
                        int32_t num_params);

/**
 * Mark orchestration as complete
 * 
 * Signals that no more tasks will be submitted.
 */
void pto2_rt_orchestration_done(PTO2Runtime* rt);

/**
 * Get output buffer pointer for a task
 */
void* pto2_rt_get_output(PTO2Runtime* rt, int32_t task_id, int32_t output_idx);

// =============================================================================
// Execution API
// =============================================================================

/**
 * Execute all submitted tasks
 * 
 * In EXECUTE mode, dispatches tasks to workers.
 * In SIMULATE mode, simulates execution with cycle counting.
 * In GRAPH_ONLY mode, does nothing (graph already built).
 * 
 * Blocks until all tasks complete.
 */
void pto2_runtime_execute(PTO2Runtime* rt);

/**
 * Signal task completion (called by worker)
 * 
 * @param rt      Runtime context
 * @param task_id Completed task ID
 */
void pto2_rt_task_complete(PTO2Runtime* rt, int32_t task_id);

/**
 * Get next ready task for worker type
 * 
 * @param rt          Runtime context
 * @param worker_type Worker type requesting task
 * @return Task ID, or -1 if no ready tasks
 */
int32_t pto2_rt_get_ready_task(PTO2Runtime* rt, PTO2WorkerType worker_type);

/**
 * Check if all tasks are complete
 */
bool pto2_runtime_is_done(PTO2Runtime* rt);

// =============================================================================
// Statistics and Debug API
// =============================================================================

/**
 * Print runtime statistics
 */
void pto2_runtime_print_stats(PTO2Runtime* rt);

/**
 * Get total simulated cycles
 */
int64_t pto2_runtime_get_cycles(PTO2Runtime* rt);

/**
 * Dump task graph to file
 * 
 * @param rt       Runtime context
 * @param filename Output file path
 * @return 0 on success, -1 on failure
 */
int pto2_runtime_dump_graph(PTO2Runtime* rt, const char* filename);

/**
 * Validate runtime state (for debugging)
 */
bool pto2_runtime_validate(PTO2Runtime* rt);

// =============================================================================
// Convenience Macros (if not already defined in pto_runtime2_types.h)
// =============================================================================

#ifndef PTO2_INPUT
/**
 * Create input parameter
 */
#define PTO2_INPUT(buf, tile, sz) \
    ((PTO2TaskParam){.type = PTO2_PARAM_INPUT, ._pad = {0}, .buffer = (buf), .tile_index = (tile), .size = (sz)})
#endif

#ifndef PTO2_OUTPUT
/**
 * Create output parameter
 */
#define PTO2_OUTPUT(buf, tile, sz) \
    ((PTO2TaskParam){.type = PTO2_PARAM_OUTPUT, ._pad = {0}, .buffer = (buf), .tile_index = (tile), .size = (sz)})
#endif

#ifndef PTO2_INOUT
/**
 * Create in-out parameter
 */
#define PTO2_INOUT(buf, tile, sz) \
    ((PTO2TaskParam){.type = PTO2_PARAM_INOUT, ._pad = {0}, .buffer = (buf), .tile_index = (tile), .size = (sz)})
#endif

/**
 * Scope helper macros
 */
#define PTO2_SCOPE_BEGIN(rt) pto2_rt_scope_begin(rt)
#define PTO2_SCOPE_END(rt)   pto2_rt_scope_end(rt)

/** Fill a single PTO2TaskParam at ABI-stable offsets (for C++ callers). */
void pto2_param_set_input(PTO2TaskParam* p, void* buf, int32_t tile_index, int32_t size_bytes);
void pto2_param_set_output(PTO2TaskParam* p, void* buf, int32_t tile_index, int32_t size_bytes);
void pto2_param_set_inout(PTO2TaskParam* p, void* buf, int32_t tile_index, int32_t size_bytes);

/** Force size at offset 20 in each 24-byte param slot. Call before submit to avoid C/C++ layout issues. */
void pto2_param_fix_sizes(void* params_base, int32_t num_params, int32_t size_bytes);

#ifdef __cplusplus
}
#endif

#endif // PTO_RUNTIME2_H
