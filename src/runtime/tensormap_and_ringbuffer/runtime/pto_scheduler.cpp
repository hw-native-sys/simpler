/**
 * PTO Runtime2 - Scheduler Implementation
 *
 * Implements scheduler state management, ready queues, and task lifecycle.
 *
 * Based on: docs/runtime_buffer_manager_methods.md
 */

#include "pto_scheduler.h"
#include <inttypes.h>
#include <new>
#include <stdlib.h>
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
    queue->spinlock.store(0, std::memory_order_relaxed);

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
    while (queue->spinlock.exchange(1, std::memory_order_acquire)) {
        PTO2_SPIN_PAUSE_LIGHT();
    }

    bool result = false;
    if (!pto2_ready_queue_full(queue)) {
        queue->task_ids[queue->tail] = task_id;
        queue->tail = (queue->tail + 1) % queue->capacity;
        queue->count++;
        result = true;
    }

    queue->spinlock.store(0, std::memory_order_release);
    return result;
}

int32_t pto2_ready_queue_pop(PTO2ReadyQueue* queue) {
    while (queue->spinlock.exchange(1, std::memory_order_acquire)) {
        PTO2_SPIN_PAUSE_LIGHT();
    }

    int32_t task_id = -1;
    if (!pto2_ready_queue_empty(queue)) {
        task_id = queue->task_ids[queue->head];
        queue->head = (queue->head + 1) % queue->capacity;
        queue->count--;
    }

    queue->spinlock.store(0, std::memory_order_release);
    return task_id;
}

// =============================================================================
// Scheduler Initialization
// =============================================================================

bool pto2_scheduler_init(PTO2SchedulerState* sched,
                          PTO2SharedMemoryHandle* sm_handle,
                          PTO2DepListPool* dep_pool,
                          void* heap_base) {
    sched->sm_handle = sm_handle;
    sched->dep_pool = dep_pool;
    sched->heap_base = heap_base;
    sched->task_state = nullptr;
    sched->fanin_refcount = nullptr;
    sched->fanout_refcount = nullptr;
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
    sched->total_dispatch_cycles = 0;

    // Get runtime task_window_size from shared memory header
    uint64_t window_size = sm_handle->header->task_window_size;
    sched->task_window_size = window_size;
    sched->task_window_mask = window_size - 1;  // For fast modulo (window_size must be power of 2)

    // Initialize local copies of ring pointers
    sched->last_task_alive = 0;
    sched->last_heap_consumed = 0;
    sched->heap_tail = 0;

    // Allocate per-task state arrays (dynamically sized based on runtime window_size)
    sched->task_state = new (std::nothrow) std::atomic<PTO2TaskState>[window_size];
    if (!sched->task_state) {
        return false;
    }

    sched->fanin_refcount = new (std::nothrow) std::atomic<int32_t>[window_size];
    if (!sched->fanin_refcount) {
        delete[] sched->task_state;
        sched->task_state = nullptr;
        return false;
    }

    sched->fanout_refcount = new (std::nothrow) std::atomic<int32_t>[window_size];
    if (!sched->fanout_refcount) {
        delete[] sched->fanin_refcount;
        delete[] sched->task_state;
        sched->fanin_refcount = nullptr;
        sched->task_state = nullptr;
        return false;
    }

    // Initialize ready queues
    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        if (!pto2_ready_queue_init(&sched->ready_queues[i], PTO2_READY_QUEUE_SIZE)) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pto2_ready_queue_destroy(&sched->ready_queues[j]);
            }
            delete[] sched->fanout_refcount;
            delete[] sched->fanin_refcount;
            delete[] sched->task_state;
            sched->fanout_refcount = nullptr;
            sched->fanin_refcount = nullptr;
            sched->task_state = nullptr;
            return false;
        }
    }

    return true;
}

void pto2_scheduler_destroy(PTO2SchedulerState* sched) {
    if (sched->task_state) {
        delete[] sched->task_state;
        sched->task_state = nullptr;
    }

    if (sched->fanin_refcount) {
        delete[] sched->fanin_refcount;
        sched->fanin_refcount = nullptr;
    }

    if (sched->fanout_refcount) {
        delete[] sched->fanout_refcount;
        sched->fanout_refcount = nullptr;
    }

    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        pto2_ready_queue_destroy(&sched->ready_queues[i]);
    }
}

void pto2_scheduler_reset(PTO2SchedulerState* sched) {
    sched->last_task_alive = 0;
    sched->last_heap_consumed = 0;
    sched->heap_tail = 0;
    for (uint64_t i = 0; i < sched->task_window_size; i++) {
        sched->task_state[i].store(PTO2_TASK_PENDING, std::memory_order_relaxed);
        sched->fanin_refcount[i].store(0, std::memory_order_relaxed);
        sched->fanout_refcount[i].store(0, std::memory_order_relaxed);
    }

    for (int i = 0; i < PTO2_NUM_WORKER_TYPES; i++) {
        pto2_ready_queue_reset(&sched->ready_queues[i]);
    }

    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
}

void pto2_scheduler_mark_running(PTO2SchedulerState* sched, int32_t task_id) {
    int32_t slot = sched->pto2_task_slot(task_id);
    sched->task_state[slot].store(PTO2_TASK_RUNNING, std::memory_order_relaxed);
}

int32_t pto2_scheduler_get_ready_task(PTO2SchedulerState* sched,
                                       PTO2WorkerType worker_type) {
    return pto2_ready_queue_pop(&sched->ready_queues[worker_type]);
}

// =============================================================================
// Task Completion Handling
// =============================================================================

void pto2_scheduler_on_task_complete(PTO2SchedulerState* sched, int32_t task_id) {
    int32_t slot = sched->pto2_task_slot(task_id);
    PTO2TaskDescriptor* task = pto2_sm_get_task(sched->sm_handle, task_id);

    // === STEP 1: Mark COMPLETED and snapshot fanout_head under lock ===
    // Acquire fanout_lock to safely read fanout_head (orchestrator may be appending).
    // Release lock EARLY: once COMPLETED is visible, orchestrator's Step 5 will
    // skip this producer (prod_state >= COMPLETED), so no new entries can be
    // appended to the fanout list. Traversal outside the lock is safe.
    pto2_fanout_lock(task);
    sched->task_state[slot].store(PTO2_TASK_COMPLETED, std::memory_order_release);
    sched->tasks_completed.fetch_add(1, std::memory_order_relaxed);
    int32_t fanout_head = task->fanout_head.load(std::memory_order_acquire);
    pto2_fanout_unlock(task);

    // Traverse fanout chain OUTSIDE the lock to notify consumers
    int32_t current = fanout_head;
    while (current > 0) {
        PTO2DepListEntry* entry = pto2_dep_pool_get(sched->dep_pool, current);
        if (!entry) break;

        int32_t consumer_id = entry->task_id;
        PTO2TaskDescriptor* consumer = pto2_sm_get_task(sched->sm_handle, consumer_id);

        // Atomically increment consumer's fanin_refcount and check if consumer is now ready
        sched->release_fanin_and_check_ready(consumer_id, consumer);

        current = entry->next_offset;
    }

    // === STEP 2: Mark CONSUMED and CAS-advance ring pointers ===
    // Mark this task as fully processed. Once CONSUMED is visible, the CAS loop
    // below (or another thread's) can advance last_task_alive past this slot.
    sched->task_state[slot].store(PTO2_TASK_CONSUMED, std::memory_order_release);
    sched->tasks_consumed.fetch_add(1, std::memory_order_relaxed);

    // CAS-based lock-free advancement of last_task_alive (matches pre-migration logic).
    // Multiple threads race to advance; CAS serializes winners.
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;
    int32_t la = header->last_task_alive.load(std::memory_order_acquire);
    int32_t cti = header->current_task_index.load(std::memory_order_acquire);

    while (la < cti) {
        int32_t la_slot = la & sched->task_window_mask;
        if (sched->task_state[la_slot].load(std::memory_order_acquire) != PTO2_TASK_CONSUMED)
            break;

        // Reset fanin_refcount before exposing slot for reuse
        sched->fanin_refcount[la_slot].store(0, std::memory_order_release);

        // Atomically advance last_task_alive by 1
        int32_t expected = la;
        if (header->last_task_alive.compare_exchange_strong(
                expected, la + 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
            // Ticket-based heap_tail serialization: wait for our turn, then write
            while (header->heap_tail_gen.load(std::memory_order_acquire) != la) {
                PTO2_SPIN_PAUSE_LIGHT();
            }
            PTO2TaskDescriptor* consumed_t = pto2_sm_get_task(sched->sm_handle, la);
            if (consumed_t->packed_buffer_end != NULL) {
                uint64_t new_tail = (uint64_t)((char*)consumed_t->packed_buffer_end - (char*)sched->heap_base);
                header->heap_tail.store(new_tail, std::memory_order_release);
            }
            header->heap_tail_gen.store(la + 1, std::memory_order_release);
            la = la + 1;
        } else {
            break;
        }
    }
}

void pto2_scheduler_on_scope_end(PTO2SchedulerState* sched,
                                  const int32_t* task_ids, int32_t count) {
    // Scope references are no longer on the critical path for ring advancement.
    // Tasks transition to CONSUMED directly in on_task_complete.
    (void)sched;
    (void)task_ids;
    (void)count;
}

// =============================================================================
// Ring Pointer Management
// =============================================================================

void pto2_scheduler_advance_ring_pointers(PTO2SchedulerState* sched) {
    // Ring advancement is now handled inline by the CAS loop in on_task_complete.
    // This function is retained for API compatibility but is a no-op.
    (void)sched;
}

void pto2_scheduler_sync_to_sm(PTO2SchedulerState* sched) {
    // Sync is now handled inline by the CAS loop in on_task_complete.
    (void)sched;
}

// =============================================================================
// Scheduler Main Loop Helpers
// =============================================================================

bool pto2_scheduler_is_done(PTO2SchedulerState* sched) {
    PTO2SharedMemoryHeader* header = sched->sm_handle->header;

    // Check if orchestrator has finished
    int32_t orch_done = header->orchestrator_done.load(std::memory_order_acquire);
    if (!orch_done) {
        return false;
    }

    // Check if all tasks have been consumed (read directly from shared memory)
    int32_t current_task_index = header->current_task_index.load(std::memory_order_acquire);
    int32_t last_alive = header->last_task_alive.load(std::memory_order_acquire);
    return last_alive >= current_task_index;
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState* sched) {
    LOG_INFO("=== Scheduler Statistics ===");
    LOG_INFO("last_task_alive:   %d", sched->last_task_alive);
    LOG_INFO("heap_tail:         %" PRIu64, sched->heap_tail);
    LOG_INFO("tasks_completed:   %lld", (long long)sched->tasks_completed.load(std::memory_order_relaxed));
    LOG_INFO("tasks_consumed:    %lld", (long long)sched->tasks_consumed.load(std::memory_order_relaxed));
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
