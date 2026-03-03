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

// PTO2_PROFILING must be defined before inline methods that use it.
// Default ON; disable with -DPTO2_PROFILING=0.
#ifndef PTO2_PROFILING
#define PTO2_PROFILING 1
#endif

#include <atomic>

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "pto_ring_buffer.h"

// =============================================================================
// Ready Queue (Lock-free bounded MPMC — Vyukov design)
// =============================================================================

/**
 * Per-slot entry: sequence counter for ABA safety + task payload
 */
struct PTO2ReadyQueueSlot {
    std::atomic<int64_t> sequence;
    int32_t task_id;
    int32_t _pad;
};

/**
 * Lock-free bounded MPMC queue (Dmitry Vyukov design)
 *
 * Key properties:
 * - enqueue_pos and dequeue_pos on separate cache lines (no false sharing)
 * - Per-slot sequence counter prevents ABA problem
 * - Empty queue pop returns immediately (single atomic load, no lock)
 * - CAS contention is split: producers only touch enqueue_pos,
 *   consumers only touch dequeue_pos
 */
struct alignas(64) PTO2ReadyQueue {
    PTO2ReadyQueueSlot* slots;
    uint64_t capacity;
    uint64_t mask;                          // capacity - 1
    char _pad0[64 - 24];                   // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];     // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    char _pad2[64 - sizeof(std::atomic<uint64_t>)];     // Own cache line

    uint64_t size() {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }

    bool push(int32_t task_id) {
        uint64_t pos;
        PTO2ReadyQueueSlot* slot;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - (int64_t)pos;
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return false;  // Queue full
            }
        }

        slot->task_id = task_id;
        slot->sequence.store((int64_t)(pos + 1), std::memory_order_release);
        return true;
    }

    int32_t pop() {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return -1;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot* slot;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - (int64_t)(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return -1;  // Queue empty
            }
        }

        int32_t task_id = slot->task_id;
        slot->sequence.store((int64_t)(pos + mask + 1), std::memory_order_release);
        return task_id;
    }
};

// Cold-path ready queue operations (defined in pto_scheduler.cpp)
bool pto2_ready_queue_init(PTO2ReadyQueue* queue, uint64_t capacity);
void pto2_ready_queue_destroy(PTO2ReadyQueue* queue);
void pto2_ready_queue_reset(PTO2ReadyQueue* queue);

// =============================================================================
// Scheduler State
// =============================================================================

/**
 * Statistics returned by on_task_complete
 */
struct PTO2CompletionStats {
    int32_t fanout_edges;      // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;    // Number of consumers that became READY
    int32_t fanin_edges;       // Number of fanin edges traversed (release producers)
};

/**
 * Scheduler state structure
 *
 * Contains dynamic state updated during task execution.
 * Separated from shared memory for cache efficiency.
 * Hot-path methods are defined inline (implicitly inline as member functions).
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

    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    int32_t pto2_task_slot(int32_t task_id) {
        return task_id & task_window_mask;
    }

    void sync_to_sm() {
        PTO2SharedMemoryHeader* header = sm_handle->header;
        header->last_task_alive.store(last_task_alive, std::memory_order_release);
        header->heap_tail.store(heap_tail, std::memory_order_release);
        header->heap_tail_gen.store(last_task_alive, std::memory_order_release);
    }

    void advance_ring_pointers() {
        PTO2SharedMemoryHeader* header = sm_handle->header;
        int32_t current_task_index = header->current_task_index.load(std::memory_order_acquire);

        while (last_task_alive < current_task_index) {
            int32_t slot = pto2_task_slot(last_task_alive);
            if (task_state[slot].load(std::memory_order_acquire) != PTO2_TASK_CONSUMED) {
                break;
            }
            last_task_alive++;
        }

        if (last_task_alive > 0) {
            int32_t last_consumed_id = last_task_alive - 1;
            PTO2TaskDescriptor* last_consumed = pto2_sm_get_task(sm_handle, last_consumed_id);
            if (last_consumed->packed_buffer_end != NULL) {
                heap_tail = (uint64_t)((char*)last_consumed->packed_buffer_end - (char*)heap_base);
            }
        }

        sync_to_sm();
    }

    void check_and_handle_consumed(int32_t task_id, PTO2TaskDescriptor* task) {
        int32_t slot = pto2_task_slot(task_id);

        int32_t fc = task->fanout_count.load(std::memory_order_acquire);
        int32_t rc = fanout_refcount[slot].load(std::memory_order_acquire);

        if (rc != fc) return;

        PTO2TaskState expected = PTO2_TASK_COMPLETED;
        if (!task_state[slot].compare_exchange_strong(expected, PTO2_TASK_CONSUMED,
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
            return;
        }

        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
        fanout_refcount[slot].store(0, std::memory_order_release);
        fanin_refcount[slot].store(0, std::memory_order_release);

        if (task_id == last_task_alive) {
            advance_ring_pointers();
        }
    }

    void release_producer(int32_t producer_id) {
        int32_t slot = pto2_task_slot(producer_id);
        PTO2TaskDescriptor* producer = pto2_sm_get_task(sm_handle, producer_id);
        fanout_refcount[slot].fetch_add(1, std::memory_order_acq_rel);
        check_and_handle_consumed(producer_id, producer);
    }

    bool release_fanin_and_check_ready(int32_t task_id,
                                        PTO2TaskDescriptor* task) {
        int32_t slot = pto2_task_slot(task_id);

        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // release in init_task, making fanin_count visible — plain load suffices.
        int32_t new_refcount = fanin_refcount[slot].fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == task->fanin_count) {
            PTO2TaskState expected = PTO2_TASK_PENDING;
            if (task_state[slot].compare_exchange_strong(
                    expected, PTO2_TASK_READY, std::memory_order_acq_rel, std::memory_order_acquire)) {
                ready_queues[task->worker_type].push(task_id);
                return true;
            }
        }
        return false;
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

    void mark_running(int32_t task_id) {
        int32_t slot = pto2_task_slot(task_id);
        task_state[slot].store(PTO2_TASK_RUNNING, std::memory_order_relaxed);
    }

    int32_t get_ready_task(PTO2WorkerType worker_type) {
        return ready_queues[worker_type].pop();
    }

    bool is_done() {
        PTO2SharedMemoryHeader* header = sm_handle->header;
        int32_t orch_done = header->orchestrator_done.load(std::memory_order_acquire);
        if (!orch_done) return false;
        int32_t current_task_index = header->current_task_index.load(std::memory_order_acquire);
        return last_task_alive >= current_task_index;
    }

    void on_scope_end(const int32_t* task_ids, int32_t count) {
        for (int32_t i = 0; i < count; i++) {
            release_producer(task_ids[i]);
        }
    }

    PTO2CompletionStats on_task_complete(int32_t task_id) {
        PTO2CompletionStats stats = {0, 0, 0};
        int32_t slot = pto2_task_slot(task_id);
        PTO2TaskDescriptor* task = pto2_sm_get_task(sm_handle, task_id);

        tasks_completed.fetch_add(1, std::memory_order_relaxed);
        pto2_fanout_lock(task);
        task_state[slot].store(PTO2_TASK_COMPLETED, std::memory_order_release);
        PTO2DepListEntry* current = task->fanout_head;  // Protected by fanout_lock
        pto2_fanout_unlock(task);

        while (current != nullptr) {
            int32_t consumer_id = current->task_id;
            PTO2TaskDescriptor* consumer = pto2_sm_get_task(sm_handle, consumer_id);
#if PTO2_PROFILING
            stats.fanout_edges++;
            if (release_fanin_and_check_ready(consumer_id, consumer)) {
                stats.tasks_enqueued++;
            }
#else
            release_fanin_and_check_ready(consumer_id, consumer);
#endif
            current = current->next;
        }

        current = task->fanin_head;
        while (current != nullptr) {
            int32_t producer_id = current->task_id;
            release_producer(producer_id);
#if PTO2_PROFILING
            stats.fanin_edges++;
#endif
            current = current->next;
        }

        check_and_handle_consumed(task_id, task);
        return stats;
    }
};

// =============================================================================
// Scheduler API (cold path, defined in pto_scheduler.cpp)
// =============================================================================

bool pto2_scheduler_init(PTO2SchedulerState* sched,
                          PTO2SharedMemoryHandle* sm_handle,
                          PTO2DepListPool* dep_pool,
                          void* heap_base);
void pto2_scheduler_destroy(PTO2SchedulerState* sched);
void pto2_scheduler_reset(PTO2SchedulerState* sched);

// =============================================================================
// Debug Utilities (cold path, defined in pto_scheduler.cpp)
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState* sched);
void pto2_scheduler_print_queues(PTO2SchedulerState* sched);
const char* pto2_task_state_name(PTO2TaskState state);

#endif // PTO_SCHEDULER_H
