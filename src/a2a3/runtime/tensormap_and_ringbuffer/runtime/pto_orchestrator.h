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
#include "pto_submit_types.h"
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
    PTO2DepListPool dep_pool;  // Dependency list storage (per-orchestrator, no atomics needed)
    PTO2DepListEntry* dep_pool_cur_entry;
    int32_t dep_pool_last_reclaimed;  // last_task_alive value at last reclamation

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;        // Producer lookup
    int32_t tensormap_last_cleanup;  // Last cleanup threshold

    // === SCOPE STACK (Private) ===
    // Single contiguous buffer of task IDs, partitioned by scope level.
    // scope_begins[i] is the index into scope_tasks where scope i starts.
    // Tasks for the top scope occupy [scope_begins[top], scope_tasks_size).
    PTO2TaskSlotState** scope_tasks; // Flat buffer of taskSlotState (all scopes concatenated)
    int32_t scope_tasks_size;       // Number of task IDs currently in the buffer
    int32_t scope_tasks_capacity;   // Allocated capacity of scope_tasks
    int32_t* scope_begins;         // scope_begins[i] = start index of scope i in scope_tasks
    int32_t scope_stack_top;       // Current top of stack (-1 = no scope open)
    uint64_t scope_stack_capacity;   // Max nesting depth (PTO2_MAX_SCOPE_DEPTH)

    // === SCHEDULER REFERENCE ===
    // Note: In simulated mode, orchestrator and scheduler share address space
    // In real mode, they communicate via shared memory only
    PTO2SchedulerState* scheduler;  // For simulated mode only
    bool init_task_on_submit;       // If true, call scheduler_init_task on submit

    // === GM HEAP (for output buffers) ===
    void* gm_heap_base;    // Base address of GM heap
    uint64_t gm_heap_size;   // Size of GM heap

    // === STATISTICS ===
#if PTO2_PROFILING
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
#endif

    /**
     * Allocate packed output buffer for a task
     */
    void* pto2_alloc_packed_buffer(int32_t total_size) {
        if (total_size <= 0) {
            return NULL;
        }

        void* buffer = heap_ring.pto2_heap_ring_alloc(total_size);

#if PTO2_PROFILING
        buffers_allocated++;
        bytes_allocated += total_size;
#endif

        // heap_top is now updated atomically inside pto2_heap_ring_alloc via CAS

        return buffer;
    }
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
    PTO2OrchestratorState* orch, PTO2SharedMemoryHandle* sm_handle, void* gm_heap, uint64_t heap_size,
    int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE);

/**
 * Destroy orchestrator state and free resources
 */
void pto2_orchestrator_destroy(PTO2OrchestratorState* orch);

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
 * @param params      Array of task parameters
 * @param num_params  Number of parameters
 */
void pto2_submit_mixed_task(PTO2OrchestratorState* orch,
    const MixedKernels& mixed_kernels,
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

// =============================================================================
// Orchestrator Profiling Data
// =============================================================================

#if PTO2_ORCH_PROFILING
struct PTO2OrchProfilingData {
    uint64_t sync_cycle;
    uint64_t alloc_cycle;
    uint64_t params_cycle;
    uint64_t lookup_cycle;
    uint64_t heap_cycle;
    uint64_t insert_cycle;
    uint64_t fanin_cycle;
    uint64_t scope_end_cycle;
    int64_t  submit_count;
    // Wait time tracking for blocking phases
    uint64_t alloc_wait_cycle;      // Cycles spent waiting in task_ring_alloc
    uint64_t heap_wait_cycle;       // Cycles spent waiting in heap_ring_alloc
    uint64_t fanin_wait_cycle;      // Cycles spent waiting in fanout_lock
    uint64_t finalize_wait_cycle;   // Cycles spent in ready queue push CAS retries
    // Atomic operation counts per phase
    uint64_t alloc_atomic_count;
    uint64_t params_atomic_count;
    uint64_t heap_atomic_count;
    uint64_t fanin_atomic_count;
    uint64_t finalize_atomic_count;
    uint64_t scope_end_atomic_count;
};

/**
 * Get and reset orchestrator profiling data.
 * Returns accumulated profiling data and resets counters.
 */
PTO2OrchProfilingData pto2_orchestrator_get_profiling();
#endif

#endif  // PTO_ORCHESTRATOR_H
