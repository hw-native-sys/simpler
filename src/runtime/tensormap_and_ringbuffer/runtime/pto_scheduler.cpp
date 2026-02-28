/**
 * PTO Runtime2 - Scheduler Implementation
 * 
 * Implements scheduler state management, ready queues, and task lifecycle.
 * 
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_scheduler.h"
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include "common/unified_log.h"

// =============================================================================
// Task State Names
// =============================================================================

const char* pto2_task_state_name(PTO2TaskState state) {
    switch (state) {
        case PTO2_TASK_PENDING:   return "PENDING";
        case PTO2_TASK_READY:     return "READY";
        case PTO2_TASK_RUNNING:   return "RUNNING";
        case PTO2_TASK_COMPLETED: return "COMPLETED";
        case PTO2_TASK_CONSUMED:  return "CONSUMED";
        default:                  return "UNKNOWN";
    }
}

// =============================================================================
// Ready Queue Implementation
// =============================================================================

bool pto2_ready_queue_init(PTO2ReadyQueue* queue, uint64_t capacity) {
    queue->task_ids = (int32_t*)malloc(capacity * sizeof(int32_t));
    if (!queue->task_ids) {
        return false;
    }

    queue->head = 0;
    queue->tail = 0;
    queue->capacity = capacity;
    queue->count = 0;

    return true;
}

void pto2_ready_queue_destroy(PTO2ReadyQueue* queue) {
    if (queue->task_ids) {
        free(queue->task_ids);
        queue->task_ids = NULL;
    }
}

void pto2_ready_queue_reset(PTO2ReadyQueue* queue) {
    queue->head = 0;
    queue->tail = 0;
    queue->count = 0;
}

bool pto2_ready_queue_push(PTO2ReadyQueue* queue, int32_t task_id) {
    if (pto2_ready_queue_full(queue)) {
        return false;
    }
    
    queue->task_ids[queue->tail] = task_id;
    queue->tail = (queue->tail + 1) % queue->capacity;
    queue->count++;
    
    return true;
}

int32_t pto2_ready_queue_pop(PTO2ReadyQueue* queue) {
    if (pto2_ready_queue_empty(queue)) {
        return -1;
    }
    
    int32_t task_id = queue->task_ids[queue->head];
    queue->head = (queue->head + 1) % queue->capacity;
    queue->count--;
    
    return task_id;
}

// =============================================================================
// Scheduler Initialization
// =============================================================================

bool pto2_scheduler_init(PTO2SchedulerState* sched,
                          PTO2SharedMemoryHandle* sm_handle,
                          PTO2DepListPool* dep_pool,
                          void* heap_base) {
    memset(sched, 0, sizeof(PTO2SchedulerState));

    sched->sm_handle = sm_handle;
    sched->dep_pool = dep_pool;
    sched->heap_base = heap_base;
    
    // Get runtime task_window_size from shared memory header
    uint64_t window_size = sm_handle->header->task_window_size;
    sched->task_window_size = window_size;
    sched->task_window_mask = window_size - 1;  // For fast modulo (window_size must be power of 2)
    
    // Initialize local copies of ring pointers
    sched->last_task_alive = 0;
    sched->heap_tail = 0;
    
    // Allocate per-task state arrays (dynamically sized based on runtime window_size)
    sched->task_state = (PTO2TaskState*)calloc(window_size, sizeof(PTO2TaskState));
    if (!sched->task_state) {
        return false;
    }
    
    sched->fanin_refcount = (int32_t*)calloc(window_size, sizeof(int32_t));
    if (!sched->fanin_refcount) {
        free(sched->task_state);
        return false;
    }
    
    sched->fanout_refcount = (int32_t*)calloc(window_size, sizeof(int32_t));
    if (!sched->fanout_refcount) {
        free(sched->fanin_refcount);
        free(sched->task_state);
        return false;
    }
    
    // Initialize ready queues
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        if (!pto2_ready_queue_init(&sched->ready_queues[i], PTO2_READY_QUEUE_SIZE)) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pto2_ready_queue_destroy(&sched->ready_queues[j]);
            }
            free(sched->fanout_refcount);
            free(sched->fanin_refcount);
            free(sched->task_state);
            return false;
        }
    }
    
    return true;
}

void pto2_scheduler_destroy(PTO2SchedulerState* sched) {
    if (sched->task_state) {
        free(sched->task_state);
        sched->task_state = NULL;
    }
    
    if (sched->fanin_refcount) {
        free(sched->fanin_refcount);
        sched->fanin_refcount = NULL;
    }
    
    if (sched->fanout_refcount) {
        free(sched->fanout_refcount);
        sched->fanout_refcount = NULL;
    }
    
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        pto2_ready_queue_destroy(&sched->ready_queues[i]);
    }
}

void pto2_scheduler_reset(PTO2SchedulerState* sched) {
    sched->last_task_alive = 0;
    sched->heap_tail = 0;

    memset(sched->task_state, 0, PTO2_TASK_WINDOW_SIZE * sizeof(PTO2TaskState));
    memset(sched->fanin_refcount, 0, PTO2_TASK_WINDOW_SIZE * sizeof(int32_t));
    memset(sched->fanout_refcount, 0, PTO2_TASK_WINDOW_SIZE * sizeof(int32_t));
    
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        pto2_ready_queue_reset(&sched->ready_queues[i]);
    }
    
    sched->tasks_completed = 0;
    sched->tasks_consumed = 0;
}

// =============================================================================
// Task State Management
// =============================================================================

void pto2_scheduler_init_task(PTO2SchedulerState* sched, int32_t task_id,
                               PTO2TaskDescriptor* task) {
    int32_t slot = pto2_task_slot(sched, task_id);
    
    // Initialize scheduler state for this task
    sched->task_state[slot] = PTO2_TASK_PENDING;
    sched->fanin_refcount[slot] = 0;
    sched->fanout_refcount[slot] = 0;
    
    // Check if task is immediately ready (no dependencies)
    if (task->fanin_count == 0) {
        sched->task_state[slot] = PTO2_TASK_READY;
        pto2_ready_queue_push(&sched->ready_queues[task->worker_type], task_id);
    }
}

void pto2_scheduler_check_ready(PTO2SchedulerState* sched, int32_t task_id,
                                 PTO2TaskDescriptor* task) {
    int32_t slot = pto2_task_slot(sched, task_id);
    
    // Only transition PENDING -> READY
    if (sched->task_state[slot] != PTO2_TASK_PENDING) {
        return;
    }
    
    // Check if all producers have completed
    if (sched->fanin_refcount[slot] == task->fanin_count) {
        sched->task_state[slot] = PTO2_TASK_READY;
        pto2_ready_queue_push(&sched->ready_queues[task->worker_type], task_id);
    }
}

void pto2_scheduler_mark_running(PTO2SchedulerState* sched, int32_t task_id) {
    int32_t slot = pto2_task_slot(sched, task_id);
    sched->task_state[slot] = PTO2_TASK_RUNNING;
}

int32_t pto2_scheduler_get_ready_task(PTO2SchedulerState* sched, 
                                       PTO2WorkerType worker_type) {
    return pto2_ready_queue_pop(&sched->ready_queues[worker_type]);
}

// =============================================================================
// Task Completion Handling
// =============================================================================

/**
 * Check if task can transition to CONSUMED and handle if so
 * 
 * NOTE: fanout_refcount is accessed atomically because it can be modified
 * by both orchestrator thread (via scope_end) and scheduler thread (via task_complete).
 */
static void check_and_handle_consumed(PTO2SchedulerState* sched, 
                                       int32_t task_id,
                                       PTO2TaskDescriptor* task) {
    int32_t slot = pto2_task_slot(sched, task_id);
    
    // Read fanout_count (set by orchestrator, only grows)
    int32_t fanout_count = __atomic_load_n(&task->fanout_count, __ATOMIC_ACQUIRE);
    
    // Read fanout_refcount atomically (modified by both orchestrator and scheduler threads)
    int32_t refcount = __atomic_load_n(&sched->fanout_refcount[slot], __ATOMIC_SEQ_CST);
    
    if (refcount != fanout_count) {
        return;  // Not all references released yet
    }

    // Use CAS to atomically transition COMPLETED -> CONSUMED
    // This prevents multiple threads from transitioning the same task
    PTO2TaskState expected = PTO2_TASK_COMPLETED;
    if (!__atomic_compare_exchange_n(&sched->task_state[slot], &expected, PTO2_TASK_CONSUMED,
                                      false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST)) {
        // CAS failed - either not COMPLETED or another thread already transitioned
        return;
    }
    
    // Successfully transitioned to CONSUMED
    __atomic_fetch_add(&sched->tasks_consumed, 1, __ATOMIC_SEQ_CST);
    
    // Reset refcounts for slot reuse (ring buffer will reuse this slot)
    // Use atomic store for fanout_refcount
    __atomic_store_n(&sched->fanout_refcount[slot], 0, __ATOMIC_SEQ_CST);
    __atomic_store_n(&sched->fanin_refcount[slot], 0, __ATOMIC_SEQ_CST);
    
    // Try to advance ring pointers
    if (task_id == sched->last_task_alive) {
        pto2_scheduler_advance_ring_pointers(sched);
    }
}

void pto2_scheduler_on_task_complete(PTO2SchedulerState* sched, int32_t task_id) {
    int32_t slot = pto2_task_slot(sched, task_id);
    PTO2TaskDescriptor* task = pto2_sm_get_task(sched->sm_handle, task_id);
    
    // Mark task as completed
    sched->task_state[slot] = PTO2_TASK_COMPLETED;
    sched->tasks_completed++;
    
    // === STEP 1: Update fanin_refcount of all consumers ===
    // Read fanout_list and increment each consumer's fanin_refcount
    int32_t fanout_head = PTO2_LOAD_ACQUIRE(&task->fanout_head);
    int32_t current = fanout_head;
    
    while (current > 0) {
        PTO2DepListEntry* entry = pto2_dep_pool_get(sched->dep_pool, current);
        if (!entry) break;
        
        int32_t consumer_id = entry->task_id;
        int32_t consumer_slot = pto2_task_slot(sched, consumer_id);
        PTO2TaskDescriptor* consumer = pto2_sm_get_task(sched->sm_handle, consumer_id);
        
        // Increment consumer's fanin_refcount
        sched->fanin_refcount[consumer_slot]++;
        
        // Check if consumer is now ready
        pto2_scheduler_check_ready(sched, consumer_id, consumer);
        
        current = entry->next_offset;
    }
    
    // === STEP 2: Update fanout_refcount of all producers ===
    // This task is a consumer of its fanin producers - release references
    current = task->fanin_head;
    
    while (current > 0) {
        PTO2DepListEntry* entry = pto2_dep_pool_get(sched->dep_pool, current);
        if (!entry) break;
        
        int32_t producer_id = entry->task_id;
        pto2_scheduler_release_producer(sched, producer_id);
        
        current = entry->next_offset;
    }
    
    // === STEP 3: Check if this task can transition to CONSUMED ===
    check_and_handle_consumed(sched, task_id, task);
}

void pto2_scheduler_on_scope_end(PTO2SchedulerState* sched,
                                  const int32_t* task_ids, int32_t count) {
    for (int32_t i = 0; i < count; i++) {
        pto2_scheduler_release_producer(sched, task_ids[i]);
    }
}

void pto2_scheduler_release_producer(PTO2SchedulerState* sched, int32_t producer_id) {
    int32_t slot = pto2_task_slot(sched, producer_id);
    PTO2TaskDescriptor* producer = pto2_sm_get_task(sched->sm_handle, producer_id);
    
    // Increment fanout_refcount atomically (called from both orchestrator and scheduler threads)
    __atomic_fetch_add(&sched->fanout_refcount[slot], 1, __ATOMIC_SEQ_CST);
    
    // Check if producer can transition to CONSUMED
    check_and_handle_consumed(sched, producer_id, producer);
}

// =============================================================================
// Ring Pointer Management
// =============================================================================

void pto2_scheduler_advance_ring_pointers(PTO2SchedulerState* sched) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;
    int32_t current_task_index = PTO2_LOAD_ACQUIRE(&header->current_task_index);
    
    // Advance last_task_alive while tasks at that position are CONSUMED
    while (sched->last_task_alive < current_task_index) {
        int32_t slot = pto2_task_slot(sched, sched->last_task_alive);
        
        if (sched->task_state[slot] != PTO2_TASK_CONSUMED) {
            break;  // Found non-consumed task, stop advancing
        }
        
        sched->last_task_alive++;
    }
    
    // Update heap_tail based on last consumed task's buffer
    if (sched->last_task_alive > 0) {
        int32_t last_consumed_id = sched->last_task_alive - 1;
        PTO2TaskDescriptor* last_consumed = pto2_sm_get_task(sched->sm_handle, last_consumed_id);
        
        if (last_consumed->packed_buffer_end != NULL) {
            sched->heap_tail = (uint64_t)((char*)last_consumed->packed_buffer_end - (char*)sched->heap_base);
        }
    }
    
    // Write to shared memory for orchestrator flow control
    pto2_scheduler_sync_to_sm(sched);
}

void pto2_scheduler_sync_to_sm(PTO2SchedulerState* sched) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;

    PTO2_STORE_RELEASE(&header->last_task_alive, sched->last_task_alive);
    PTO2_STORE_RELEASE(&header->heap_tail, sched->heap_tail);
    // Keep generation in sync so AICPU mode sees a consistent starting state
    PTO2_STORE_RELEASE(&header->heap_tail_gen, sched->last_task_alive);
}

// =============================================================================
// Scheduler Main Loop Helpers
// =============================================================================

bool pto2_scheduler_is_done(PTO2SchedulerState* sched) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;
    
    // Check if orchestrator has finished
    int32_t orch_done = PTO2_LOAD_ACQUIRE(&header->orchestrator_done);
    if (!orch_done) {
        return false;
    }
    
    // Check if all tasks have been consumed
    int32_t current_task_index = PTO2_LOAD_ACQUIRE(&header->current_task_index);
    return sched->last_task_alive >= current_task_index;
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState* sched) {
    LOG_INFO("=== Scheduler Statistics ===");
    LOG_INFO("last_task_alive:   %d", sched->last_task_alive);
    LOG_INFO("heap_tail:         %" PRIu64, sched->heap_tail);
    LOG_INFO("tasks_completed:   %lld", (long long)sched->tasks_completed);
    LOG_INFO("tasks_consumed:    %lld", (long long)sched->tasks_consumed);
    LOG_INFO("============================");
}

void pto2_scheduler_print_queues(PTO2SchedulerState* sched) {
    LOG_INFO("=== Ready Queues ===");
    
    const char* worker_names[] = {"CUBE", "VECTOR", "AI_CPU", "ACCELERATOR"};
    
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        LOG_INFO("  %s: count=%" PRIu64, worker_names[i],
                 pto2_ready_queue_count(&sched->ready_queues[i]));
    }
    
    LOG_INFO("====================");
}
