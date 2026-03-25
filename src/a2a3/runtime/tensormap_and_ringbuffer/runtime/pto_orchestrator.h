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

    // === PER-RING RESOURCES ===
    PTO2RingSet rings[PTO2_MAX_RING_DEPTH];

    // === TENSOR MAP (Private) ===
    PTO2TensorMap tensor_map;        // Producer lookup

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
#if PTO2_PROFILING
    // Runtime profiling switch copied from Runtime::enable_profiling.
    bool enable_profiling;
#endif

    // === GM HEAP (for output buffers) ===
    void* gm_heap_base;    // Base address of GM heap
    uint64_t gm_heap_size;   // Total size of GM heap (all rings)

    // === VIRTUAL ADDRESS SEQUENCE ===
    // Monotonically increasing counter for assigning virtual addresses to
    // OUTPUT tensors. Used as TensorMap hash key; physical allocation is
    // deferred to Scheduler dispatch time.
    uint64_t next_vaddr_seq{1};

    // === FATAL ERROR ===
    // Fatal error flag (single-thread access by orchestrator, no atomic needed)
    // Cross-thread notification uses shared memory orch_error_code (atomic)
    bool fatal;

    // === STATISTICS ===
#if PTO2_PROFILING
    int64_t tasks_submitted;
    int64_t buffers_allocated;
    int64_t bytes_allocated;
#endif

    /**
     * Get current ring index from scope depth.
     * Maps scope depth to ring_id: min(scope_depth, PTO2_MAX_RING_DEPTH - 1)
     */
    uint8_t current_ring_id() const {
        int32_t depth = scope_stack_top;
        if (depth < 0) depth = 0;
        return depth < PTO2_MAX_RING_DEPTH ? static_cast<uint8_t>(depth) : PTO2_MAX_RING_DEPTH - 1;
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
 * @param mixed_kernels  Kernel IDs for AIC/AIV0/AIV1 slots
 * @param params      Aggregated tensor and scalar parameters
 */
void pto2_submit_mixed_task(PTO2OrchestratorState* orch,
    const MixedKernels& mixed_kernels,
    const PTOParam& params);

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
    uint64_t insert_cycle;
    uint64_t fanin_cycle;
    uint64_t scope_end_cycle;
    int64_t  submit_count;
    // Wait time tracking for blocking phases
    uint64_t alloc_wait_cycle;      // Cycles spent waiting in task_ring_alloc
    uint64_t fanin_wait_cycle;      // Cycles spent waiting in fanout_lock
    // Atomic operation counts per phase
    uint64_t alloc_atomic_count;
    uint64_t params_atomic_count;
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
