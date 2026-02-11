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

#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_scheduler.h"
#include "pto_shared_memory.h"
#include "pto_tensormap.h"
#include "pto_types.h"

// =============================================================================
// Orchestrator State
// =============================================================================

/**
 * Orchestrator state structure (private to Orchestrator)
 *
 * Contains all state needed for task graph construction and buffer management.
 */
struct PTO2OrchestratorState {
    // === SHARED MEMORY ACCESS ===
    PTO2SharedMemoryHandle* sm_handle;

    // === RING BUFFERS ===
    PTO2HeapRing heap_ring;    // Output buffer allocation
    PTO2TaskRing task_ring;    // Task slot allocation
    PTO2DepListPool dep_pool;  // Dependency list allocation

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;        // Producer lookup
    int32_t tensormap_last_cleanup;  // Last cleanup threshold

    // === SCOPE STACK (Private) ===
    // Single contiguous buffer of task IDs, partitioned by scope level.
    // scope_begins[i] is the index into scope_tasks where scope i starts.
    // Tasks for the top scope occupy [scope_begins[top], scope_tasks_size).
    int32_t* scope_tasks;          // Flat buffer of task IDs (all scopes concatenated)
    int32_t scope_tasks_size;      // Number of task IDs currently in the buffer
    int32_t scope_tasks_capacity;  // Allocated capacity of scope_tasks
    int32_t* scope_begins;         // scope_begins[i] = start index of scope i in scope_tasks
    int32_t scope_stack_top;       // Current top of stack (-1 = no scope open)
    int32_t scope_stack_capacity;  // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState* scheduler;  // For simulated mode only
    bool init_task_on_submit;       // If true, call scheduler_init_task on submit

    // === GM HEAP (for output buffers) ===
    void* gm_heap_base;    // Base address of GM heap
    int32_t gm_heap_size;  // Size of GM heap

    // === STATISTICS ===
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
};

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
bool pto2_orchestrator_init(
    PTO2OrchestratorState* orch, PTO2SharedMemoryHandle* sm_handle, void* gm_heap, int32_t heap_size);

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
void pto2_orchestrator_set_scheduler(PTO2OrchestratorState* orch, PTO2SchedulerState* scheduler);

/**
 * Set scheduler reference with mode control
 *
 * @param orch           Orchestrator state
 * @param scheduler      Scheduler state
 * @param init_on_submit If true, init task on submit (single-threaded mode)
 *                       If false, scheduler thread polls for new tasks (multi-threaded)
 */
void pto2_orchestrator_set_scheduler_mode(
    PTO2OrchestratorState* orch, PTO2SchedulerState* scheduler, bool init_on_submit);

// =============================================================================
// Scope Management
// =============================================================================

/**
 * Begin a new scope
 *
 * Pushes a new empty task list onto the scope stack.
 * Tasks submitted while this scope is at the top of the stack are
 * owned by it and have their fanout_count initialized to 1.
 */
void pto2_scope_begin(PTO2OrchestratorState* orch);

/**
 * End current scope
 *
 * Pops the top scope and increments fanout_refcount for each task
 * directly owned by that scope.
 * May trigger buffer release for tasks that are now fully consumed.
 */
void pto2_scope_end(PTO2OrchestratorState* orch);

// =============================================================================
// Task Submission
// =============================================================================

/**
 * Submit a task with InCore function and parameters
 *
 * This is the main API for building the task graph:
 * 1. Allocates task slot from TaskRing (blocks until available)
 * 2. Allocates packed output buffer from HeapRing (blocks until available)
 * 3. Looks up inputs in TensorMap to find dependencies
 * 4. Updates producer's fanout_count/list (with spinlock)
 * 5. Registers outputs in TensorMap
 * 6. Initializes task state in scheduler
 *
 * @param orch        Orchestrator state
 * @param kernel_id   InCore function ID
 * @param worker_type Target worker type (CUBE, VECTOR, AI_CPU, ACCELERATOR)
 * @param func_name   Function name (for debugging)
 * @param params      Array of task parameters
 * @param num_params  Number of parameters
 */
void pto2_submit_task(PTO2OrchestratorState* orch,
    int32_t kernel_id,
    PTO2WorkerType worker_type,
    const char* func_name,
    PTOParam* params,
    int32_t num_params);

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
// Internal Helpers
// =============================================================================

/**
 * Add consumer to producer's fanout list (with spinlock)
 * Also checks if producer has already completed and updates consumer's fanin_refcount
 */
void pto2_add_consumer_to_producer(
    PTO2OrchestratorState* orch, PTO2TaskDescriptor* producer, int32_t producer_id, int32_t consumer_id);

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

#endif  // PTO_ORCHESTRATOR_H
