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
#include <utility>
#include "common/unified_log.h"

// =============================================================================
// Scheduler Profiling Counters
// =============================================================================

#if PTO2_SCHED_PROFILING
#include "common/platform_config.h"

uint64_t g_sched_lock_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_consumed_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_push_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_complete_count[PLATFORM_MAX_AICPU_THREADS] = {};

PTO2SchedProfilingData pto2_scheduler_get_profiling(int thread_idx) {
    PTO2SchedProfilingData d;
    d.lock_cycle = std::exchange(g_sched_lock_cycle[thread_idx], 0);
    d.fanout_cycle = std::exchange(g_sched_fanout_cycle[thread_idx], 0);
    d.fanin_cycle = std::exchange(g_sched_fanin_cycle[thread_idx], 0);
    d.self_consumed_cycle = std::exchange(g_sched_self_consumed_cycle[thread_idx], 0);
    d.lock_wait_cycle = std::exchange(g_sched_lock_wait_cycle[thread_idx], 0);
    d.push_wait_cycle = std::exchange(g_sched_push_wait_cycle[thread_idx], 0);
    d.pop_wait_cycle = std::exchange(g_sched_pop_wait_cycle[thread_idx], 0);
    d.lock_atomic_count = std::exchange(g_sched_lock_atomic_count[thread_idx], 0);
    d.fanout_atomic_count = std::exchange(g_sched_fanout_atomic_count[thread_idx], 0);
    d.fanin_atomic_count = std::exchange(g_sched_fanin_atomic_count[thread_idx], 0);
    d.self_atomic_count = std::exchange(g_sched_self_atomic_count[thread_idx], 0);
    d.pop_atomic_count = std::exchange(g_sched_pop_atomic_count[thread_idx], 0);
    d.complete_count = std::exchange(g_sched_complete_count[thread_idx], 0);
    return d;
}
#endif

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
    queue->slots = (PTO2ReadyQueueSlot*)malloc(capacity * sizeof(PTO2ReadyQueueSlot));
    if (!queue->slots) {
        return false;
    }

    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++) {
        queue->slots[i].sequence.store((int64_t)i, std::memory_order_relaxed);
        queue->slots[i].task_id = -1;
    }

    return true;
}

void pto2_ready_queue_destroy(PTO2ReadyQueue* queue) {
    if (queue->slots) {
        free(queue->slots);
        queue->slots = NULL;
    }
}

// =============================================================================
// Scheduler Initialization
// =============================================================================

bool pto2_scheduler_init(PTO2SchedulerState* sched,
                          PTO2SharedMemoryHandle* sm_handle,
                          void* heap_base) {
    sched->sm_handle = sm_handle;
    sched->task_descriptors = sm_handle->task_descriptors;
    sched->heap_base = heap_base;
    sched->slot_states = nullptr;
#if PTO2_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif
    sched->ring_advance_lock.store(0, std::memory_order_relaxed);

    // Get runtime task_window_size from shared memory header
    uint64_t window_size = sm_handle->header->task_window_size;
    sched->task_window_size = window_size;
    sched->task_window_mask = window_size - 1;  // For fast modulo (window_size must be power of 2)

    // Initialize local copies of ring pointers
    sched->last_task_alive = 0;
    sched->last_heap_consumed = 0;
    sched->heap_tail = 0;

    // Allocate per-task slot state array (dynamically sized based on runtime window_size)
    sched->slot_states = new (std::nothrow) PTO2TaskSlotState[window_size];
    if (!sched->slot_states) {
        return false;
    }

    // Zero-initialize all per-task slot state fields.
    // new[] default-initializes std::atomic<T> which leaves values indeterminate.
    // Scheduler logic (e.g. fanin_refcount fetch_add in release_fanin_and_check_ready)
    // assumes slots start at zero before init_task writes them.
    for (uint64_t i = 0; i < window_size; i++) {
        sched->slot_states[i].fanout_lock.store(0, std::memory_order_relaxed);
        sched->slot_states[i].fanout_count = 0;
        sched->slot_states[i].fanout_head = nullptr;
        sched->slot_states[i].task_state.store(static_cast<PTO2TaskState>(0), std::memory_order_relaxed);
        sched->slot_states[i].fanin_refcount.store(0, std::memory_order_relaxed);
        sched->slot_states[i].fanin_count = 0;
        sched->slot_states[i].fanout_refcount.store(0, std::memory_order_relaxed);
    }

    // Initialize ready queues (one per resource shape)
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        if (!pto2_ready_queue_init(&sched->ready_queues[i], PTO2_READY_QUEUE_SIZE)) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pto2_ready_queue_destroy(&sched->ready_queues[j]);
            }
            delete[] sched->slot_states;
            sched->slot_states = nullptr;
            return false;
        }
    }

    return true;
}

void pto2_scheduler_destroy(PTO2SchedulerState* sched) {
    if (sched->slot_states) {
        delete[] sched->slot_states;
        sched->slot_states = nullptr;
    }

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        pto2_ready_queue_destroy(&sched->ready_queues[i]);
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState* sched) {
    LOG_INFO("=== Scheduler Statistics ===");
    LOG_INFO("last_task_alive:   %d", sched->last_task_alive);
    LOG_INFO("heap_tail:         %" PRIu64, sched->heap_tail);
#if PTO2_SCHED_PROFILING
    LOG_INFO("tasks_completed:   %lld", (long long)sched->tasks_completed.load(std::memory_order_relaxed));
    LOG_INFO("tasks_consumed:    %lld", (long long)sched->tasks_consumed.load(std::memory_order_relaxed));
#endif
    LOG_INFO("============================");
}

void pto2_scheduler_print_queues(PTO2SchedulerState* sched) {
    LOG_INFO("=== Ready Queues ===");

    const char* shape_names[] = {"AIC_ONLY", "AIV_X1", "AIV_X2", "AIC_AIV_X1", "AIC_AIV_X2"};

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        LOG_INFO("  %s: count=%" PRIu64, shape_names[i],
                 sched->ready_queues[i].size());
    }

    LOG_INFO("====================");
}
