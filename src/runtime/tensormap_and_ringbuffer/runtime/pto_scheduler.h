/**
 * PTO Runtime2 - Scheduler Interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-worker-type ready queues
 * 2. Tracking task state (PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED)
 * 3. Managing fanin/fanout refcounts for dependency resolution
 * 4. Advancing last_task_alive for heap reclamation
 *
 * The Scheduler runs on Device AI_CPU and processes:
 * - Task state transitions based on fanin_refcount
 * - Buffer lifecycle based on fanout_refcount
 * - Ring pointer advancement for flow control
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#ifndef PTO_SCHEDULER_H
#define PTO_SCHEDULER_H

#include <atomic>

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_ring_buffer.h"

// =============================================================================
// Ready Queue Structure
// =============================================================================

/**
 * Per-worker-type ready queue
 * Circular buffer of task IDs
 */
struct PTO2ReadyQueue {
    int32_t* task_ids;    // Circular buffer of task IDs
    uint64_t head;        // Dequeue position
    uint64_t tail;        // Enqueue position
    uint64_t capacity;    // Queue capacity
    uint64_t count;       // Current number of tasks in queue
    std::atomic<int32_t> spinlock; // Spinlock for thread-safe push/pop
};

/**
 * Push task to ready queue
 * @return true if successful, false if queue is full
 */
bool pto2_ready_queue_push(PTO2ReadyQueue* queue, int32_t task_id);

// =============================================================================
// Scheduler State
// =============================================================================

/**
 * Scheduler state structure (private to Scheduler)
 *
 * Contains dynamic state updated during task execution.
 * Separated from shared memory for cache efficiency.
 */
struct PTO2SchedulerState {
    // Shared memory access
    PTO2SharedMemoryHandle* sm_handle;

    // Local copies of ring pointers (written to shared memory after update)
    int32_t last_task_alive;      // Task ring tail (advances on COMPLETED for slot reuse)
    int32_t last_heap_consumed;   // Heap watermark (advances on CONSUMED for buffer reuse)
    uint64_t heap_tail;           // Heap ring tail (offset from heap_base)

    // Heap base address (for converting absolute pointers to offsets)
    void* heap_base;

    // === DYNAMIC CONFIGURATION ===
    uint64_t task_window_size;    // Task window size (power of 2)
    uint64_t task_window_mask;    // task_window_size - 1 (for fast modulo)

    // === PRIVATE DATA (not in shared memory) ===

    // Per-task state arrays (dynamically allocated, indexed by task_id & task_window_mask)
    std::atomic<PTO2TaskState>* task_state; // PENDING/READY/RUNNING/COMPLETED/CONSUMED
    std::atomic<int32_t>* fanin_refcount;   // Dynamic: counts completed producers
    std::atomic<int32_t>* fanout_refcount;  // Dynamic: counts released references

    // Ready queues (one per worker type)
    PTO2ReadyQueue ready_queues[PTO2_NUM_WORKER_TYPES];

    // Dependency list pool reference
    PTO2DepListPool* dep_pool;

    // Statistics
    std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
    int64_t total_dispatch_cycles;



    /**
    * Calculate task slot from task_id
    * Uses runtime window_size instead of compile-time constant
    */
    inline int32_t pto2_task_slot(int32_t task_id) {
        return task_id & task_window_mask;
    }

    // =============================================================================
    // Task State Management
    // =============================================================================

    /**
     * Signal that one fanin dependency has been satisfied
     *
     * Atomically increments fanin_refcount. If the new count equals
     * fanin_count, CAS PENDING -> READY and enqueue.
     *
     * @param task_id Task ID
     * @param task    Task descriptor
     */
    void release_fanin_and_check_ready(int32_t task_id, PTO2TaskDescriptor* task) {
        int32_t slot = pto2_task_slot(task_id);

        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // release in init_task, making fanin_count visible — plain load suffices.
        int32_t new_refcount = fanin_refcount[slot].fetch_add(1, std::memory_order_acq_rel) + 1;

        // Check if all producers have completed
        if (new_refcount == task->fanin_count) {
            // CAS PENDING -> READY to prevent double-enqueue from concurrent threads
            PTO2TaskState expected = PTO2_TASK_PENDING;
            if (task_state[slot].compare_exchange_strong(
                    expected, PTO2_TASK_READY, std::memory_order_acq_rel, std::memory_order_acquire)) {
                pto2_ready_queue_push(&ready_queues[task->worker_type], task_id);
            }
        }
    }

    void init_task(int32_t task_id, PTO2TaskDescriptor* task) {
        int32_t slot = pto2_task_slot(task_id);

        task_state[slot].store(PTO2_TASK_PENDING, std::memory_order_relaxed); // Orchestrator is the unique owner

        // Reset fanout_refcount for new task lifecycle.
        // Do NOT reset fanin_refcount — it may have been incremented by
        // concurrent on_task_complete between Step 5 and Step 6.
        fanout_refcount[slot].store(0, std::memory_order_relaxed);

        release_fanin_and_check_ready(task_id, task);
    }
};

// =============================================================================
// Scheduler API
// =============================================================================

/**
 * Initialize scheduler state
 *
 * @param sched      Scheduler state to initialize
 * @param sm_handle  Shared memory handle
 * @param dep_pool   Dependency list pool
 * @param heap_base  Base address of GM heap (for pointer-to-offset conversion)
 * @return true on success
 */
bool pto2_scheduler_init(PTO2SchedulerState* sched,
                          PTO2SharedMemoryHandle* sm_handle,
                          PTO2DepListPool* dep_pool,
                          void* heap_base);

/**
 * Destroy scheduler state and free resources
 */
void pto2_scheduler_destroy(PTO2SchedulerState* sched);

/**
 * Reset scheduler state for reuse
 */
void pto2_scheduler_reset(PTO2SchedulerState* sched);

// =============================================================================
// Ready Queue Operations
// =============================================================================

/**
 * Initialize a ready queue
 */
bool pto2_ready_queue_init(PTO2ReadyQueue* queue, uint64_t capacity);

/**
 * Destroy ready queue
 */
void pto2_ready_queue_destroy(PTO2ReadyQueue* queue);

/**
 * Reset ready queue to empty
 */
void pto2_ready_queue_reset(PTO2ReadyQueue* queue);

/**
 * Pop task from ready queue
 * @return task_id, or -1 if queue is empty
 */
int32_t pto2_ready_queue_pop(PTO2ReadyQueue* queue);

/**
 * Check if ready queue is empty
 */
static inline bool pto2_ready_queue_empty(PTO2ReadyQueue* queue) {
    return queue->count == 0;
}

/**
 * Check if ready queue is full
 */
static inline bool pto2_ready_queue_full(PTO2ReadyQueue* queue) {
    return queue->count >= queue->capacity;
}

/**
 * Get ready queue count
 */
static inline uint64_t pto2_ready_queue_count(PTO2ReadyQueue* queue) {
    return queue->count;
}

// =============================================================================
// Task State Management
// =============================================================================

/**
 * Mark task as RUNNING (dispatched to worker)
 */
void pto2_scheduler_mark_running(PTO2SchedulerState* sched, int32_t task_id);

/**
 * Get next ready task from queue for worker type
 *
 * @param sched       Scheduler state
 * @param worker_type Worker type to get task for
 * @return task_id, or -1 if no ready tasks
 */
int32_t pto2_scheduler_get_ready_task(PTO2SchedulerState* sched,
                                       PTO2WorkerType worker_type);

// =============================================================================
// Task Completion Handling
// =============================================================================

/**
 * Handle task completion
 *
 * Called by worker when task execution finishes:
 * 1. Sets task state to COMPLETED
 * 2. Updates fanin_refcount of all consumers (may make them READY)
 * 3. Updates fanout_refcount of all producers (may make them CONSUMED)
 * 4. Checks if this task can transition to CONSUMED
 *
 * @param sched   Scheduler state
 * @param task_id Completed task ID
 */
void pto2_scheduler_on_task_complete(PTO2SchedulerState* sched, int32_t task_id);

/**
 * Handle scope end (called when orchestrator ends a scope)
 *
 * Increments fanout_refcount for each task in the provided list.
 * May transition some tasks to CONSUMED.
 *
 * @param sched     Scheduler state
 * @param task_ids  Array of task IDs owned by the ending scope
 * @param count     Number of task IDs in the array
 */
void pto2_scheduler_on_scope_end(PTO2SchedulerState* sched,
                                  const int32_t* task_ids, int32_t count);

// =============================================================================
// Ring Pointer Management
// =============================================================================

/**
 * Advance last_task_alive and heap_tail
 *
 * Called when a task transitions to CONSUMED.
 * Advances pointers if possible, updating shared memory.
 *
 * @param sched Scheduler state
 */
void pto2_scheduler_advance_ring_pointers(PTO2SchedulerState* sched);

/**
 * Sync scheduler state to shared memory
 * Writes last_task_alive and heap_tail
 */
void pto2_scheduler_sync_to_sm(PTO2SchedulerState* sched);

// =============================================================================
// Scheduler Main Loop
// =============================================================================

/**
 * Check if all tasks are done (orchestrator finished and all tasks consumed)
 */
bool pto2_scheduler_is_done(PTO2SchedulerState* sched);

// =============================================================================
// Debug Utilities
// =============================================================================

/**
 * Print scheduler statistics
 */
void pto2_scheduler_print_stats(PTO2SchedulerState* sched);

/**
 * Print ready queue states
 */
void pto2_scheduler_print_queues(PTO2SchedulerState* sched);

/**
 * Get task state as string
 */
const char* pto2_task_state_name(PTO2TaskState state);

#endif // PTO_SCHEDULER_H
