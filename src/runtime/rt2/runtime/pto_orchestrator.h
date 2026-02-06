/**
 * PTO Runtime2 - Orchestrator Interface
 * 
 * The Orchestrator is responsible for:
 * 1. Executing the orchestration function (Turing-complete control flow)
 * 2. Allocating intermediate buffers from the heap
 * 3. Submitting tasks via async InCore function calls
 * 4. Building the dependency graph using TensorMap
 * 5. Managing buffer scopes for lifecycle control
 * 
 * The Orchestrator can run on either:
 * - Host CPU (lower latency for complex control, easier debugging)
 * - Device AI_CPU (lower latency for task submission)
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_ORCHESTRATOR_H
#define PTO_ORCHESTRATOR_H

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_ring_buffer.h"
#include "pto_tensormap.h"
#include "pto_scheduler.h"

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 * 
 * Contains all state needed for task graph construction and buffer management.
 */
typedef struct {
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHandle* sm_handle;
    
    // === RING BUFFERS ===
    PTO2HeapRing    heap_ring;      // Output buffer allocation
    PTO2TaskRing    task_ring;      // Task slot allocation
    PTO2DepListPool dep_pool;       // Dependency list allocation
    
    // === TENSOR MAP (Private) ===
    PTO2TensorMap   tensor_map;     // Producer lookup
    int32_t         tensormap_last_cleanup;  // Last cleanup threshold
    
    // === SCOPE STACK (Private) ===
    int32_t*        scope_stack;    // Stack of scope begin positions
    int32_t         scope_stack_top;// Current top of stack (-1 = empty)
    int32_t         scope_stack_capacity;
    
    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState* scheduler;  // For simulated mode only
    bool init_task_on_submit;       // If true, call scheduler_init_task on submit
    
    // === GM HEAP (for output buffers) ===
    void*           gm_heap_base;   // Base address of GM heap
    int32_t         gm_heap_size;   // Size of GM heap
    
    // === STATISTICS ===
    int64_t         tasks_submitted;
    int64_t         buffers_allocated;
    int64_t         bytes_allocated;
    int64_t         scope_depth_max;
    
} PTO2OrchestratorState;

// =============================================================================
// Orchestrator API
// =============================================================================

/**
 * Initialize orchestrator state
 * 
 * @param orch       Orchestrator state to initialize
 * @param sm_handle  Shared memory handle
 * @param gm_heap    GM heap memory for output buffers
 * @param heap_size  Size of GM heap
 * @return true on success
 */
bool pto2_orchestrator_init(PTO2OrchestratorState* orch,
                             PTO2SharedMemoryHandle* sm_handle,
                             void* gm_heap,
                             int32_t heap_size);

/**
 * Destroy orchestrator state and free resources
 */
void pto2_orchestrator_destroy(PTO2OrchestratorState* orch);

/**
 * Reset orchestrator state for reuse
 */
void pto2_orchestrator_reset(PTO2OrchestratorState* orch);

/**
 * Set scheduler reference (for simulated mode)
 */
void pto2_orchestrator_set_scheduler(PTO2OrchestratorState* orch,
                                      PTO2SchedulerState* scheduler);

/**
 * Set scheduler reference with mode control
 * 
 * @param orch           Orchestrator state
 * @param scheduler      Scheduler state
 * @param init_on_submit If true, init task on submit (single-threaded mode)
 *                       If false, scheduler thread polls for new tasks (multi-threaded)
 */
void pto2_orchestrator_set_scheduler_mode(PTO2OrchestratorState* orch,
                                           PTO2SchedulerState* scheduler,
                                           bool init_on_submit);

// =============================================================================
// Scope Management
// =============================================================================

/**
 * Begin a new scope
 * 
 * Pushes current task index to scope stack.
 * All buffers allocated within this scope will have their fanout_count
 * initialized to include this scope reference.
 */
void pto2_scope_begin(PTO2OrchestratorState* orch);

/**
 * End current scope
 * 
 * Pops scope stack and increments fanout_refcount for all tasks
 * in [scope_begin_pos, current_task_index).
 * May trigger buffer release for tasks that are now fully consumed.
 */
void pto2_scope_end(PTO2OrchestratorState* orch);

/**
 * Get current scope depth
 * @return Current nesting depth (0 = global scope)
 */
static inline int32_t pto2_get_scope_depth(PTO2OrchestratorState* orch) {
    return orch->scope_stack_top + 1;
}

// =============================================================================
// Task Submission
// =============================================================================

/**
 * Submit a task with InCore function and parameters
 * 
 * This is the main API for building the task graph:
 * 1. Allocates task slot from TaskRing (may stall)
 * 2. Allocates packed output buffer from HeapRing (may stall)
 * 3. Looks up inputs in TensorMap to find dependencies
 * 4. Updates producer's fanout_count/list (with spinlock)
 * 5. Registers outputs in TensorMap
 * 6. Initializes task state in scheduler
 * 
 * @param orch        Orchestrator state
 * @param kernel_id   InCore function ID
 * @param worker_type Target worker type (CUBE, VECTOR, AI_CPU, ACCELERATOR)
 * @param func_ptr    Function pointer (optional)
 * @param func_name   Function name (for debugging)
 * @param params      Array of task parameters
 * @param num_params  Number of parameters
 * @return Task ID, or -1 on failure
 */
int32_t pto2_submit_task(PTO2OrchestratorState* orch,
                          int32_t kernel_id,
                          PTO2WorkerType worker_type,
                          void* func_ptr,
                          const char* func_name,
                          PTO2TaskParam* params,
                          int32_t num_params);

/**
 * Get pointer to specific output of a task
 * 
 * @param orch       Orchestrator state
 * @param task_id    Task ID
 * @param output_idx Output index (0-based)
 * @return Pointer to output buffer
 */
void* pto2_task_get_output(PTO2OrchestratorState* orch, 
                            int32_t task_id, 
                            int32_t output_idx);

// =============================================================================
// Flow Control
// =============================================================================

/**
 * Mark orchestration as complete
 * 
 * Signals to scheduler that no more tasks will be submitted.
 */
void pto2_orchestrator_done(PTO2OrchestratorState* orch);

/**
 * Wait for all tasks to complete
 * 
 * Blocks until scheduler reports all tasks consumed.
 * Only valid in simulated mode or with shared address space.
 */
void pto2_orchestrator_wait_all(PTO2OrchestratorState* orch);

/**
 * Check if orchestrator has space for more tasks
 */
bool pto2_orchestrator_has_space(PTO2OrchestratorState* orch);

// =============================================================================
// TensorMap Synchronization
// =============================================================================

/**
 * Sync TensorMap validity threshold from shared memory
 * 
 * Called periodically to refresh the lazy invalidation threshold.
 * Also triggers cleanup if threshold has advanced significantly.
 */
void pto2_orchestrator_sync_tensormap(PTO2OrchestratorState* orch);

// =============================================================================
// Internal Helpers
// =============================================================================

/**
 * Add consumer to producer's fanout list (with spinlock)
 * Also checks if producer has already completed and updates consumer's fanin_refcount
 */
void pto2_add_consumer_to_producer(PTO2OrchestratorState* orch,
                                    PTO2TaskDescriptor* producer,
                                    int32_t producer_id,
                                    int32_t consumer_id);

/**
 * Allocate packed output buffer for a task
 */
void* pto2_alloc_packed_buffer(PTO2OrchestratorState* orch, int32_t total_size);

// =============================================================================
// Debug Utilities
// =============================================================================

/**
 * Print orchestrator statistics
 */
void pto2_orchestrator_print_stats(PTO2OrchestratorState* orch);

/**
 * Print scope stack state
 */
void pto2_orchestrator_print_scope_stack(PTO2OrchestratorState* orch);

#endif // PTO_ORCHESTRATOR_H
