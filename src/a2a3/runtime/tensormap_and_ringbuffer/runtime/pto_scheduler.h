/**
 * PTO Runtime2 - Scheduler Interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-resource-shape ready queues
 * 2. Tracking task state (PENDING -> READY -> RUNNING -> COMPLETED -> RELEASED -> CONSUMED)
 * 3. Managing fanin/fanout refcounts for dependency resolution
 * 4. Dual waterline advancement (last_task_released for main ring, last_task_consumed for heap)
 * 5. Two-stage mixed-task completion (subtask done bits → mixed-task complete)
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

#include "common/core_type.h"

#if PTO2_SCHED_PROFILING
#include "aicpu/device_time.h"
#define PTO2_SCHED_CYCLE_START() uint64_t _st0 = get_sys_cnt_aicpu(), _st1
#define PTO2_SCHED_CYCLE_LAP(acc) do { _st1 = get_sys_cnt_aicpu(); acc += (_st1 - _st0); _st0 = _st1; } while(0)
#endif

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
 * Thread-local ready buffer for local-first dispatch optimization.
 *
 * Two buffers per scheduling thread, one per CoreType (AIC=0, AIV=1).
 * Initialized once before the scheduling loop; must be empty at
 * the start of each iteration (verified by always_assert).
 *
 * Phase 1 fills per-CoreType buffers via on_task_complete().
 * dispatch_ready_tasks_to_idle_cores drains them: local-first via
 * get_ready_task, then remaining tasks pushed to global readyQ.
 */
// Number of CoreType values eligible for local dispatch (AIC=0, AIV=1)
static constexpr int PTO2_LOCAL_DISPATCH_TYPE_NUM = 2;

struct PTO2LocalReadyBuffer {
    int32_t* task_ids = nullptr;  // Points to caller's stack array
    int count = 0;
    int capacity = 0;

    void reset(int32_t* buf, int cap) {
        task_ids = buf;
        count = 0;
        capacity = cap;
    }

    bool try_push(int32_t task_id) {
        if (task_ids && count < capacity) {
            task_ids[count++] = task_id;
            return true;
        }
        return false;
    }

    int32_t pop() {
        return (count > 0) ? task_ids[--count] : -1;  // LIFO: better cache locality
    }
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
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            }
        }

        slot->task_id = task_id;
        slot->sequence.store((int64_t)(pos + 1), std::memory_order_release);
        return true;
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool push(int32_t task_id, uint64_t& atomic_count, uint64_t& wait_cycle) {
        uint64_t pos;
        PTO2ReadyQueueSlot* slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - (int64_t)pos;
            atomic_ops += 2;  // enqueue_pos.load + sequence.load
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                return false;  // Queue full
            } else {
                contended = true;  // diff > 0: slot not yet released, spin
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        slot->task_id = task_id;
        slot->sequence.store((int64_t)(pos + 1), std::memory_order_release);
        return true;
    }
#endif

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

#if PTO2_SCHED_PROFILING
    int32_t pop(uint64_t& atomic_count, uint64_t& wait_cycle) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        atomic_count += 2;  // dequeue_pos.load + enqueue_pos.load
        if (d >= e) {
            return -1;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot* slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - (int64_t)(pos + 1);
            atomic_ops += 2;  // dequeue_pos.load + sequence.load
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(pos, pos + 1,
                        std::memory_order_relaxed, std::memory_order_relaxed)) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                atomic_count += atomic_ops;
                return -1;  // Queue empty
            } else {
                contended = true;
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        int32_t task_id = slot->task_id;
        slot->sequence.store((int64_t)(pos + mask + 1), std::memory_order_release);
        return task_id;
    }
#endif
};

// Cold-path ready queue operations (defined in pto_scheduler.cpp)
bool pto2_ready_queue_init(PTO2ReadyQueue* queue, uint64_t capacity);
void pto2_ready_queue_destroy(PTO2ReadyQueue* queue);

// =============================================================================
// Scheduler State
// =============================================================================

/**
 * Statistics returned by mixed-task completion processing
 */
struct PTO2CompletionStats {
    int32_t fanout_edges;      // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;    // Number of consumers that became READY
    int32_t fanin_edges;       // Number of fanin edges traversed (release producers)
    bool mixed_task_completed; // True only when this callback completed a mixed task
};

/**
 * Scheduler state structure
 *
 * Contains dynamic state updated during task execution.
 * Uses dual ring buffers: main ring (short-lived fanin state) and
 * consumed ring (long-lived fanout/task_state).
 * Hot-path methods are defined inline (implicitly inline as member functions).
 */
struct PTO2SchedulerState {
    // Shared memory access
    PTO2SharedMemoryHandle* sm_handle;
    PTO2TaskDescriptor*     task_descriptors;

    // Local copies of ring pointers (written to shared memory after update)
    int32_t last_task_consumed;   // Consumed ring tail (drives heap reclamation)
    int32_t last_task_released;   // Main ring tail (drives slot reclamation)
    int32_t last_heap_consumed;   // Heap watermark (advances on CONSUMED for buffer reuse)
    uint64_t heap_tail;           // Heap ring tail (offset from heap_base)

    // Heap base address (for converting absolute pointers to offsets)
    void* heap_base;

    // === DYNAMIC CONFIGURATION ===
    uint64_t task_window_size;    // Main ring capacity (power of 2)
    uint64_t task_window_mask;    // task_window_size - 1 (for fast modulo)
    uint64_t consumed_window_size;  // Consumed ring capacity (default = 4 × task_window_size)
    uint64_t consumed_window_mask;  // consumed_window_size - 1

    // === PRIVATE DATA (not in shared memory) ===

    // Consumed ring: long-lived per-task state (task_state, fanout tracking, heap ptr)
    // Indexed by task_id & consumed_window_mask
    PTO2ConsumedRingEntry* consumed_ring;

    // Main slot states: short-lived fanin tracking only
    // Indexed by task_id & task_window_mask
    PTO2MainSlotState* main_slot_states;

    // Ready queues (one per resource shape)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Statistics
#if PTO2_SCHED_PROFILING
    std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
#endif
    std::atomic<int32_t> ring_advance_lock{0};       // Try-lock for advance_consumed_ring_pointers
    std::atomic<int32_t> main_ring_advance_lock{0};  // Try-lock for advance_main_ring_pointers

    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    int32_t get_task_slot(int32_t task_id) {
        return task_id & task_window_mask;
    }

    PTO2ConsumedRingEntry& get_consumed_entry(int32_t task_id) {
        return consumed_ring[task_id & consumed_window_mask];
    }

    PTO2MainSlotState& get_main_slot(int32_t task_id) {
        return main_slot_states[task_id & task_window_mask];
    }

    void sync_to_sm() {
        PTO2SharedMemoryHeader* header = sm_handle->header;
        header->last_task_consumed.store(last_task_consumed, std::memory_order_release);
        header->last_task_released.store(last_task_released, std::memory_order_release);
        header->heap_tail.store(heap_tail, std::memory_order_release);
        header->heap_tail_gen.store(last_task_consumed, std::memory_order_release);
    }

    /**
     * Advance consumed ring watermark: scan CONSUMED tasks, update heap_tail.
     * Drives heap reclamation.
     */
    void advance_consumed_ring_pointers() {
        PTO2SharedMemoryHeader* header = sm_handle->header;
        int32_t current_task_index = header->current_task_index.load(std::memory_order_acquire);

        while (last_task_consumed < current_task_index) {
            PTO2ConsumedRingEntry& entry = get_consumed_entry(last_task_consumed);
            if (entry.task_state.load(std::memory_order_acquire) != PTO2_TASK_CONSUMED) {
                break;
            }
            last_task_consumed++;
        }

        if (last_task_consumed > 0) {
            PTO2ConsumedRingEntry& last_entry = get_consumed_entry(last_task_consumed - 1);
            if (last_entry.packed_buffer_end != nullptr) {
                heap_tail = (uint64_t)((char*)last_entry.packed_buffer_end - (char*)heap_base);
            }
        }

        sync_to_sm();
    }

    /**
     * Advance main ring watermark: scan RELEASED tasks.
     * Drives main ring slot (descriptor + payload) reclamation.
     */
    void advance_main_ring_pointers() {
        PTO2SharedMemoryHeader* header = sm_handle->header;
        int32_t current_task_index = header->current_task_index.load(std::memory_order_acquire);

        while (last_task_released < current_task_index) {
            PTO2ConsumedRingEntry& entry = get_consumed_entry(last_task_released);
            if (entry.task_state.load(std::memory_order_acquire) < PTO2_TASK_RELEASED) {
                break;
            }
            last_task_released++;
        }

        header->last_task_released.store(last_task_released, std::memory_order_release);
    }

    void check_and_handle_consumed(PTO2ConsumedRingEntry& entry) {
        if (entry.fanout_refcount.load(std::memory_order_acquire) != entry.fanout_count) return;

        // CAS from RELEASED→CONSUMED (not COMPLETED→CONSUMED)
        // If task is still COMPLETED (not yet RELEASED), CAS fails safely.
        PTO2TaskState expected = PTO2_TASK_RELEASED;
        if (!entry.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED,
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
            return;
        }

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        // Try-lock — if another thread is advancing, it will scan our CONSUMED task
        int32_t expected_lock = 0;
        if (ring_advance_lock.compare_exchange_strong(expected_lock, 1,
                std::memory_order_acquire, std::memory_order_relaxed)) {
            advance_consumed_ring_pointers();
            ring_advance_lock.store(0, std::memory_order_release);
        }
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void check_and_handle_consumed(PTO2ConsumedRingEntry& entry, uint64_t& atomic_count) {
        int32_t fc = entry.fanout_count;
        int32_t rc = entry.fanout_refcount.load(std::memory_order_acquire);

        atomic_count += 2;  // fanout_count.load + fanout_refcount.load

        if (rc != fc) return;

        PTO2TaskState expected = PTO2_TASK_RELEASED;
        if (!entry.task_state.compare_exchange_strong(expected, PTO2_TASK_CONSUMED,
                                          std::memory_order_acq_rel, std::memory_order_acquire)) {
            atomic_count += 1;  // failed CAS
            return;
        }

        atomic_count += 1;  // successful CAS

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        // Try-lock — if another thread is advancing, it will scan our CONSUMED task
        int32_t expected_lock = 0;
        if (ring_advance_lock.compare_exchange_strong(expected_lock, 1,
                std::memory_order_acquire, std::memory_order_relaxed)) {
            advance_consumed_ring_pointers();
            ring_advance_lock.store(0, std::memory_order_release);
            atomic_count += 2;  // try-lock CAS + unlock store
        } else {
            atomic_count += 1;  // failed try-lock CAS
        }
    }
#endif

    void release_producer(int32_t producer_id) {
        PTO2ConsumedRingEntry& entry = get_consumed_entry(producer_id);
        entry.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        check_and_handle_consumed(entry);
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void release_producer(int32_t producer_id, uint64_t& atomic_count) {
        PTO2ConsumedRingEntry& entry = get_consumed_entry(producer_id);
        entry.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        atomic_count += 1;  // fanout_refcount.fetch_add
        check_and_handle_consumed(entry, atomic_count);
    }
#endif

    bool release_fanin_and_check_ready(int32_t task_id,
                                        PTO2TaskDescriptor* task,
                                        PTO2LocalReadyBuffer* local_bufs = nullptr) {
        PTO2MainSlotState& main_slot = get_main_slot(task_id);

        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // release in init_task, making fanin_count visible — plain load suffices.
        int32_t new_refcount = main_slot.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == main_slot.fanin_count) {
            // Local-first: try per-CoreType thread-local buffer before global queue
            // Route by active_mask: AIC-containing tasks → buf[0], AIV-only → buf[1]
            PTO2ResourceShape shape = pto2_active_mask_to_shape(task->active_mask);
            bool pushed_local = false;
            if (local_bufs) {
                int32_t buf_idx = (task->active_mask & 0x01) ? 0 : 1;
                pushed_local = local_bufs[buf_idx].try_push(task_id);
            }
            if (!pushed_local) {
                ready_queues[static_cast<int32_t>(shape)].push(task_id);
            }
            return true;
        }
        return false;
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool release_fanin_and_check_ready(int32_t task_id, PTO2TaskDescriptor* task,
                                        uint64_t& atomic_count, uint64_t& push_wait,
                                        PTO2LocalReadyBuffer* local_bufs = nullptr) {
        PTO2MainSlotState& main_slot = get_main_slot(task_id);

        int32_t new_refcount = main_slot.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
        atomic_count += 1;  // fanin_refcount.fetch_add

        if (new_refcount == main_slot.fanin_count) {
            PTO2ConsumedRingEntry& entry = get_consumed_entry(task_id);
            PTO2TaskState expected = PTO2_TASK_PENDING;
            if (entry.task_state.compare_exchange_strong(
                    expected, PTO2_TASK_READY, std::memory_order_acq_rel, std::memory_order_acquire)) {
                atomic_count += 1;  // CAS(task_state PENDING→READY)
                // Local-first: try per-CoreType thread-local buffer before global queue
                PTO2ResourceShape shape = pto2_active_mask_to_shape(task->active_mask);
                bool pushed_local = false;
                if (local_bufs) {
                    int32_t buf_idx = (task->active_mask & 0x01) ? 0 : 1;
                    pushed_local = local_bufs[buf_idx].try_push(task_id);
                }
                if (!pushed_local) {
                    ready_queues[static_cast<int32_t>(shape)].push(task_id, atomic_count, push_wait);
                }
                return true;
            }
        }
        return false;
    }
#endif

    void init_task(int32_t task_id, PTO2TaskDescriptor* task) {
        PTO2ConsumedRingEntry& entry = get_consumed_entry(task_id);

        entry.task_state.store(PTO2_TASK_PENDING, std::memory_order_relaxed);

        // Reset fanout_refcount for new task lifecycle.
        // Do NOT reset fanin_refcount — it may have been incremented by
        // concurrent on_task_complete between Step 5 and Step 6.
        entry.fanout_refcount.store(0, std::memory_order_relaxed);

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
        extern uint64_t g_orch_finalize_atomic_count;
        extern uint64_t g_orch_finalize_wait_cycle;
        release_fanin_and_check_ready(task_id, task,
                                       g_orch_finalize_atomic_count, g_orch_finalize_wait_cycle);
#else
        release_fanin_and_check_ready(task_id, task);
#endif
    }

    int32_t get_ready_task(PTO2ResourceShape shape) {
        return ready_queues[static_cast<int32_t>(shape)].pop();
    }

    template<CoreType CT>
    int32_t get_ready_task(PTO2LocalReadyBuffer* local_bufs) {
        constexpr int ct = static_cast<int>(CT);
        if (local_bufs && local_bufs[ct].count > 0) {
            return local_bufs[ct].pop();
        }
        return ready_queues[ct].pop();
    }

#if PTO2_SCHED_PROFILING
    int32_t get_ready_task(PTO2ResourceShape shape, uint64_t& atomic_count, uint64_t& wait_cycle) {
        return ready_queues[static_cast<int32_t>(shape)].pop(atomic_count, wait_cycle);
    }

    template<CoreType CT>
    int32_t get_ready_task(PTO2LocalReadyBuffer* local_bufs,
                           uint64_t& atomic_count, uint64_t& wait_cycle) {
        constexpr int ct = static_cast<int>(CT);
        if (local_bufs && local_bufs[ct].count > 0) {
            return local_bufs[ct].pop();
        }
        return ready_queues[ct].pop(atomic_count, wait_cycle);
    }
#endif

    /**
     * Requeue a ready task that could not be dispatched (no suitable cluster).
     * Pushes the task back into its shape-based queue.
     */
    void requeue_ready_task(int32_t task_id) {
        int32_t slot = get_task_slot(task_id);
        PTO2TaskDescriptor& task = pto2_sm_get_task_by_slot(sm_handle, slot);
        PTO2ResourceShape shape = pto2_active_mask_to_shape(task.active_mask);
        ready_queues[static_cast<int32_t>(shape)].push(task_id);
    }

    void on_scope_end(const int32_t* task_ids, int32_t count) {
#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
        extern uint64_t g_orch_scope_end_atomic_count;
        for (int32_t i = 0; i < count; i++) {
            release_producer(task_ids[i], g_orch_scope_end_atomic_count);
        }
#else
        for (int32_t i = 0; i < count; i++) {
            release_producer(task_ids[i]);
        }
#endif
    }

    /**
     * Two-stage completion: first stage.
     * Called when a single subtask (AIC, AIV0, or AIV1) finishes.
     * Sets the corresponding done bit in subtask_done_mask.
     *
     * @return true if this subtask was the last one, completing the mixed task.
     */
    bool on_subtask_complete(int32_t mixed_task_id, PTO2SubtaskSlot subslot) {
        int32_t slot = get_task_slot(mixed_task_id);
        PTO2TaskDescriptor& task = pto2_sm_get_task_by_slot(sm_handle, slot);

        uint8_t done_bit = (1u << static_cast<uint8_t>(subslot));
        uint8_t prev_mask = task.subtask_done_mask.fetch_or(done_bit, std::memory_order_acq_rel);
        uint8_t new_mask = prev_mask | done_bit;

        return new_mask == task.active_mask;
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a mixed task are done
     * (i.e., on_subtask_complete returned true).
     * Handles fanout notification, fanin release, and self-consumption check.
     */
#if PTO2_SCHED_PROFILING
    PTO2CompletionStats on_mixed_task_complete(int32_t mixed_task_id, int thread_idx,
                                               PTO2LocalReadyBuffer* local_bufs = nullptr) {
        PTO2CompletionStats stats = {0, 0, 0, true};
#elif PTO2_PROFILING
    PTO2CompletionStats on_mixed_task_complete(int32_t mixed_task_id,
                                               PTO2LocalReadyBuffer* local_bufs = nullptr) {
        PTO2CompletionStats stats = {0, 0, 0, true};
#else
    void on_mixed_task_complete(int32_t mixed_task_id,
                                PTO2LocalReadyBuffer* local_bufs = nullptr) {
#endif
        PTO2ConsumedRingEntry& entry = get_consumed_entry(mixed_task_id);

#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_lock_cycle[], g_sched_fanout_cycle[];
        extern uint64_t g_sched_lock_atomic_count[], g_sched_lock_wait_cycle[];
        extern uint64_t g_sched_fanout_atomic_count[], g_sched_push_wait_cycle[];
        uint64_t lock_atomics = 0, lock_wait = 0;
        PTO2_SCHED_CYCLE_START();
#endif

#if PTO2_SCHED_PROFILING
        pto2_fanout_lock(entry, lock_atomics, lock_wait);
#else
        pto2_fanout_lock(entry);
#endif
        entry.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        PTO2DepListEntry* current = entry.fanout_head;  // Protected by fanout_lock
        pto2_fanout_unlock(entry);

#if PTO2_SCHED_PROFILING
        lock_atomics += 2;  // state.store + unlock.store
        g_sched_lock_atomic_count[thread_idx] += lock_atomics;
        g_sched_lock_wait_cycle[thread_idx] += lock_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_lock_cycle[thread_idx]);
#endif

        // Fanout: notify consumers
#if PTO2_SCHED_PROFILING
        uint64_t fanout_atomics = 0, push_wait = 0;
#endif
        while (current != nullptr) {
            int32_t consumer_id = current->task_id;
            PTO2TaskDescriptor* consumer = pto2_sm_get_task(sm_handle, consumer_id);
#if PTO2_PROFILING
            stats.fanout_edges++;
#endif
#if PTO2_SCHED_PROFILING
            if (release_fanin_and_check_ready(consumer_id, consumer,
                                               fanout_atomics, push_wait, local_bufs)) {
#if PTO2_PROFILING
                stats.tasks_enqueued++;
#endif
            }
#elif PTO2_PROFILING
            if (release_fanin_and_check_ready(consumer_id, consumer, local_bufs)) {
                stats.tasks_enqueued++;
            }
#else
            release_fanin_and_check_ready(consumer_id, consumer, local_bufs);
#endif
            current = current->next;
        }

#if PTO2_SCHED_PROFILING
        g_sched_fanout_atomic_count[thread_idx] += fanout_atomics;
        g_sched_push_wait_cycle[thread_idx] += push_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanout_cycle[thread_idx]);
#endif

#if PTO2_PROFILING
        return stats;
#endif
    }

    /**
     * Cold path: release producers (fanin traversal) + transition to RELEASED.
     *
     * SAFETY: payload->fanin_tasks must be fully read BEFORE setting RELEASED,
     * because advance_main_ring_pointers() may reclaim the main ring slot
     * once RELEASED is visible.
     *
     * Returns fanin edge count for profiling.
     */

#if PTO2_SCHED_PROFILING
    int32_t on_task_release(int32_t task_id, int32_t thread_idx) {
        PTO2_SCHED_CYCLE_START();
        extern uint64_t g_sched_fanin_cycle[], g_sched_fanin_atomic_count[];
        extern uint64_t g_sched_self_atomic_count[];
        extern uint64_t g_sched_self_consumed_cycle[];
        extern uint64_t g_sched_complete_count[];
        uint64_t fanin_atomics = 0;
#else
    int32_t on_task_release(int32_t task_id) {
#endif
        int32_t slot = get_task_slot(task_id);
        PTO2TaskPayload* payload = &sm_handle->task_payloads[slot];
        int32_t fanin_edges = payload->fanin_actual_count;

        // Read all fanin producer IDs from payload BEFORE marking RELEASED
        for (int32_t i = 0; i < fanin_edges; i++) {
#if PTO2_SCHED_PROFILING
            release_producer(payload->fanin_tasks[i], fanin_atomics);
#else
            release_producer(payload->fanin_tasks[i]);
#endif
        }
#if PTO2_SCHED_PROFILING
        g_sched_fanin_atomic_count[thread_idx] += fanin_atomics;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanin_cycle[thread_idx]);
#endif

        // Transition to RELEASED — payload is no longer needed
        PTO2ConsumedRingEntry& entry = get_consumed_entry(task_id);
        entry.task_state.store(PTO2_TASK_RELEASED, std::memory_order_release);

        // Try to advance main ring watermark
        int32_t expected_lock = 0;
        if (main_ring_advance_lock.compare_exchange_strong(expected_lock, 1,
                std::memory_order_acquire, std::memory_order_relaxed)) {
            advance_main_ring_pointers();
            main_ring_advance_lock.store(0, std::memory_order_release);
        }

        // Self consumed check — fanout_refcount may already equal fanout_count
#if PTO2_SCHED_PROFILING
        uint64_t self_atomics = 0;
        check_and_handle_consumed(entry, self_atomics);
        g_sched_self_atomic_count[thread_idx] += self_atomics;
        PTO2_SCHED_CYCLE_LAP(g_sched_self_consumed_cycle[thread_idx]);
        g_sched_complete_count[thread_idx]++;
#else
        check_and_handle_consumed(entry);
#endif
        return fanin_edges;
    }
};

// =============================================================================
// Scheduler API (cold path, defined in pto_scheduler.cpp)
// =============================================================================

bool pto2_scheduler_init(PTO2SchedulerState* sched,
                          PTO2SharedMemoryHandle* sm_handle,
                          void* heap_base,
                          uint64_t consumed_window_size);
void pto2_scheduler_destroy(PTO2SchedulerState* sched);

// =============================================================================
// Debug Utilities (cold path, defined in pto_scheduler.cpp)
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState* sched);
void pto2_scheduler_print_queues(PTO2SchedulerState* sched);
const char* pto2_task_state_name(PTO2TaskState state);

// =============================================================================
// Scheduler Profiling Data
// =============================================================================

#if PTO2_SCHED_PROFILING
struct PTO2SchedProfilingData {
    // Sub-phase cycle breakdown within on_mixed_task_complete
    uint64_t lock_cycle;           // pto2_fanout_lock + state store + unlock
    uint64_t fanout_cycle;         // fanout traversal
    uint64_t fanin_cycle;          // fanin traversal
    uint64_t self_consumed_cycle;  // self check_and_handle_consumed

    // Wait times
    uint64_t lock_wait_cycle;      // spin-wait in fanout_lock
    uint64_t push_wait_cycle;      // CAS contention in push()
    uint64_t pop_wait_cycle;       // CAS contention in pop()

    // Atomic counts per sub-phase
    uint64_t lock_atomic_count;
    uint64_t fanout_atomic_count;
    uint64_t fanin_atomic_count;
    uint64_t self_atomic_count;
    uint64_t pop_atomic_count;

    int64_t  complete_count;
};

/**
 * Get and reset scheduler profiling data for a specific thread.
 * Returns accumulated profiling data and resets counters.
 */
PTO2SchedProfilingData pto2_scheduler_get_profiling(int thread_idx);
#endif

#endif // PTO_SCHEDULER_H
