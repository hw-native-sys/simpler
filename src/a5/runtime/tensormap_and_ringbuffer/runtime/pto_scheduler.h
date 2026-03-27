/**
 * PTO Runtime2 - Scheduler Interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-resource-shape ready queues
 * 2. Tracking task state (PENDING -> READY -> RUNNING -> COMPLETED -> CONSUMED)
 * 3. Managing fanin/fanout refcounts for dependency resolution
 * 4. Advancing last_task_alive for heap reclamation
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
#define PTO2_SCHED_CYCLE_LAP(acc)   \
    do {                            \
        _st1 = get_sys_cnt_aicpu(); \
        acc += (_st1 - _st0);       \
        _st0 = _st1;                \
    } while (0)
#endif

// =============================================================================
// Ready Queue (Lock-free bounded MPMC — Vyukov design)
// =============================================================================

/**
 * Per-slot entry: sequence counter for ABA safety + task payload
 */
struct PTO2ReadyQueueSlot {
    std::atomic<int64_t> sequence;
    PTO2TaskSlotState* slot_state;
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
    PTO2TaskSlotState** slot_states = nullptr;
    int count = 0;
    int capacity = 0;

    void reset(PTO2TaskSlotState** buf, int cap) {
        slot_states = buf;
        count = 0;
        capacity = cap;
    }

    bool try_push(PTO2TaskSlotState* s) {
        if (slot_states && count < capacity) {
            slot_states[count++] = s;
            return true;
        }
        return false;
    }

    PTO2TaskSlotState* pop() { return (count > 0) ? slot_states[--count] : nullptr; }
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
    uint64_t mask;        // capacity - 1
    char _pad0[64 - 24];  // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    char _pad2[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    uint64_t size() {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }

    bool push(PTO2TaskSlotState* slot_state) {
        uint64_t pos;
        PTO2ReadyQueueSlot* slot;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - (int64_t)pos;
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            }
        }

        slot->slot_state = slot_state;
        slot->sequence.store((int64_t)(pos + 1), std::memory_order_release);
        return true;
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool push(PTO2TaskSlotState* slot_state, uint64_t& atomic_count, uint64_t& wait_cycle) {
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
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
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

        slot->slot_state = slot_state;
        slot->sequence.store((int64_t)(pos + 1), std::memory_order_release);
        return true;
    }
#endif

    PTO2TaskSlotState* pop() {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot* slot;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - (int64_t)(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed))
                    break;
            } else if (diff < 0) {
                return nullptr;  // Queue empty
            }
        }

        PTO2TaskSlotState* result = slot->slot_state;
        slot->sequence.store((int64_t)(pos + mask + 1), std::memory_order_release);
        return result;
    }

#if PTO2_SCHED_PROFILING
    PTO2TaskSlotState* pop(uint64_t& atomic_count, uint64_t& wait_cycle) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        atomic_count += 2;  // dequeue_pos.load + enqueue_pos.load
        if (d >= e) {
            return nullptr;
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
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed)) {
                    atomic_ops++;  // successful CAS
                    break;
                }
                contended = true;
                atomic_ops++;  // failed CAS
            } else if (diff < 0) {
                atomic_count += atomic_ops;
                return nullptr;  // Queue empty
            } else {
                contended = true;
            }
        }
        atomic_ops++;  // final sequence.store
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }

        PTO2TaskSlotState* result = slot->slot_state;
        slot->sequence.store((int64_t)(pos + mask + 1), std::memory_order_release);
        return result;
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
    int32_t fanout_edges;       // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;     // Number of consumers that became READY
    int32_t fanin_edges;        // Number of fanin edges traversed (release producers)
    bool mixed_task_completed;  // True only when this callback completed a mixed task
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

    // Per-ring state
    struct RingSchedState {
        PTO2TaskDescriptor* task_descriptors;
        PTO2TaskSlotState* slot_states;
        int32_t last_task_alive;
        int32_t last_heap_consumed;
        uint64_t heap_tail;
        void* heap_base;
        int32_t task_window_mask;
        uint64_t task_window_size;
        // Try-lock used to advance this ring's pointers (CONSUMED scanning + heap tail update).
        std::atomic<int32_t> advance_lock;

        bool init(PTO2SharedMemoryHandle* sm_handle, int32_t ring_id, void* gm_heap_base, uint64_t per_ring_heap_size);
        void destroy();

        PTO2TaskSlotState& get_slot_state_by_task_id(int32_t local_id) {
            return slot_states[local_id & task_window_mask];
        }
        PTO2TaskSlotState& get_slot_state_by_slot(int32_t slot) { return slot_states[slot]; }

        void sync_to_sm(PTO2SharedMemoryRingHeader& ring) {
            ring.fc.last_task_alive.store(last_task_alive, std::memory_order_release);
            ring.fc.heap_tail.store(heap_tail, std::memory_order_release);
        }

        void advance_ring_pointers(PTO2SharedMemoryRingHeader& ring) {
            int32_t current_task_index = ring.fc.current_task_index.load(std::memory_order_acquire);

            while (last_task_alive < current_task_index) {
                PTO2TaskSlotState& slot_state = get_slot_state_by_task_id(last_task_alive);
                if (slot_state.task_state.load(std::memory_order_acquire) != PTO2_TASK_CONSUMED) {
                    break;
                }
                last_task_alive++;
            }

            if (last_task_alive > 0) {
                int32_t last_consumed_id = last_task_alive - 1;
                PTO2TaskSlotState& slot_state = get_slot_state_by_task_id(last_consumed_id);
                PTO2TaskDescriptor& task = *slot_state.task;
                if (task.packed_buffer_end != NULL) {
                    heap_tail = (uint64_t)((char*)task.packed_buffer_end - (char*)heap_base);
                }
            }

            sync_to_sm(ring);
        }
    } ring_sched_states[PTO2_MAX_RING_DEPTH];

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Statistics
#if PTO2_SCHED_PROFILING
    std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
#endif
    // =========================================================================
    // Inline hot-path methods
    // =========================================================================
    PTO2TaskSlotState& get_slot_state(int32_t ring_id, int32_t local_id) {
        return ring_sched_states[ring_id].get_slot_state_by_task_id(local_id);
    }
    PTO2TaskSlotState& get_slot_state_by_slot(int32_t ring_id, int32_t slot) {
        return ring_sched_states[ring_id].get_slot_state_by_slot(slot);
    }

    void check_and_handle_consumed(PTO2TaskSlotState& slot_state) {
        if (slot_state.fanout_refcount.load(std::memory_order_acquire) != slot_state.fanout_count) return;

        PTO2TaskState expected = PTO2_TASK_COMPLETED;
        if (!slot_state.task_state.compare_exchange_strong(
                expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire)) {
            return;
        }

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        int32_t ring_id = slot_state.ring_id;
        // Try-lock — if another thread is advancing this ring, it will scan our CONSUMED task
        int32_t expected_lock = 0;
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(
                expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            ring_sched_states[ring_id].advance_ring_pointers(sm_handle->header->rings[ring_id]);
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
        }
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void check_and_handle_consumed(PTO2TaskSlotState& slot_state, uint64_t& atomic_count) {
        int32_t fc = slot_state.fanout_count;
        int32_t rc = slot_state.fanout_refcount.load(std::memory_order_acquire);

        atomic_count += 2;  // fanout_count.load + fanout_refcount.load

        if (rc != fc) return;

        PTO2TaskState expected = PTO2_TASK_COMPLETED;
        if (!slot_state.task_state.compare_exchange_strong(
                expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire)) {
            atomic_count += 1;  // failed CAS
            return;
        }

        atomic_count += 1;  // successful CAS

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        int32_t ring_id = slot_state.ring_id;
        // Try-lock — if another thread is advancing this ring, it will scan our CONSUMED task
        int32_t expected_lock = 0;
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(
                expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed)) {
            ring_sched_states[ring_id].advance_ring_pointers(sm_handle->header->rings[ring_id]);
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
            atomic_count += 2;  // try-lock CAS + unlock store
        } else {
            atomic_count += 1;  // failed try-lock CAS
        }
    }
#endif

    void release_producer(PTO2TaskSlotState& slot_state) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        check_and_handle_consumed(slot_state);
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void release_producer(PTO2TaskSlotState& slot_state, uint64_t& atomic_count) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        atomic_count += 1;  // fanout_refcount.fetch_add
        check_and_handle_consumed(slot_state, atomic_count);
    }
#endif

    bool release_fanin_and_check_ready(PTO2TaskSlotState& slot_state, PTO2LocalReadyBuffer* local_bufs = nullptr) {
        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // init release, making fanin_count visible — plain load suffices.
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == slot_state.fanin_count) {
            // Local-first: try per-CoreType thread-local buffer before global queue
            // Route by active_mask: AIC-containing tasks → buf[0], AIV-only → buf[1]
            PTO2ResourceShape shape = pto2_active_mask_to_shape(slot_state.active_mask);
            bool pushed_local = false;
            if (local_bufs) {
                int32_t buf_idx = (slot_state.active_mask & 0x01) ? 0 : 1;
                pushed_local = local_bufs[buf_idx].try_push(&slot_state);
            }
            if (!pushed_local) {
                ready_queues[static_cast<int32_t>(shape)].push(&slot_state);
            }
            return true;
        }
        return false;
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool release_fanin_and_check_ready(PTO2TaskSlotState& slot_state,
        uint64_t& atomic_count,
        uint64_t& push_wait,
        PTO2LocalReadyBuffer* local_bufs = nullptr) {
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
        atomic_count += 1;  // fanin_refcount.fetch_add

        if (new_refcount == slot_state.fanin_count) {
            PTO2TaskState expected = PTO2_TASK_PENDING;
            if (slot_state.task_state.compare_exchange_strong(
                    expected, PTO2_TASK_READY, std::memory_order_acq_rel, std::memory_order_acquire)) {
                atomic_count += 1;  // CAS(task_state PENDING→READY)
                // Local-first: try per-CoreType thread-local buffer before global queue
                PTO2ResourceShape shape = pto2_active_mask_to_shape(slot_state.active_mask);
                bool pushed_local = false;
                if (local_bufs) {
                    int32_t buf_idx = (slot_state.active_mask & 0x01) ? 0 : 1;
                    pushed_local = local_bufs[buf_idx].try_push(&slot_state);
                }
                if (!pushed_local) {
                    ready_queues[static_cast<int32_t>(shape)].push(&slot_state, atomic_count, push_wait);
                }
                return true;
            }
        }
        return false;
    }
#endif

    PTO2TaskSlotState* get_ready_task(PTO2ResourceShape shape) {
        return ready_queues[static_cast<int32_t>(shape)].pop();
    }

    template <CoreType CT>
    PTO2TaskSlotState* get_ready_task(PTO2LocalReadyBuffer* local_bufs) {
        constexpr int ct = static_cast<int>(CT);
        if (local_bufs && local_bufs[ct].count > 0) {
            return local_bufs[ct].pop();
        }
        return ready_queues[ct].pop();
    }

#if PTO2_SCHED_PROFILING
    PTO2TaskSlotState* get_ready_task(PTO2ResourceShape shape, uint64_t& atomic_count, uint64_t& wait_cycle) {
        return ready_queues[static_cast<int32_t>(shape)].pop(atomic_count, wait_cycle);
    }

    template <CoreType CT>
    PTO2TaskSlotState* get_ready_task(PTO2LocalReadyBuffer* local_bufs, uint64_t& atomic_count, uint64_t& wait_cycle) {
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
    void requeue_ready_task(PTO2TaskSlotState& slot_state) {
        PTO2ResourceShape shape = pto2_active_mask_to_shape(slot_state.active_mask);
        ready_queues[static_cast<int32_t>(shape)].push(&slot_state);
    }

    void on_scope_end(PTO2TaskSlotState** task_slot_states, int32_t count) {
#if PTO2_ORCH_PROFILING
        extern uint64_t g_orch_scope_end_atomic_count;
        for (int32_t i = 0; i < count; i++) {
            release_producer(*task_slot_states[i], g_orch_scope_end_atomic_count);
        }
#else
        for (int32_t i = 0; i < count; i++) {
            release_producer(*task_slot_states[i]);
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
    bool on_subtask_complete(PTO2TaskSlotState& slot_state, PTO2SubtaskSlot subslot) {
        uint8_t done_bit = (1u << static_cast<uint8_t>(subslot));
        uint8_t prev_mask = slot_state.subtask_done_mask.fetch_or(done_bit, std::memory_order_acq_rel);
        uint8_t new_mask = prev_mask | done_bit;

        return new_mask == slot_state.active_mask;
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a mixed task are done
     * (i.e., on_subtask_complete returned true).
     * Handles fanout notification, fanin release, and self-consumption check.
     */
#if PTO2_SCHED_PROFILING
    PTO2CompletionStats
#else
    void
#endif
    on_mixed_task_complete(PTO2TaskSlotState& slot_state,
#if PTO2_SCHED_PROFILING
        int thread_idx,
#endif

        PTO2LocalReadyBuffer* local_bufs = nullptr) {
#if PTO2_SCHED_PROFILING
        PTO2CompletionStats stats = {0, 0, 0, true};
#endif
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_lock_cycle[], g_sched_fanout_cycle[];
        extern uint64_t g_sched_lock_atomic_count[], g_sched_lock_wait_cycle[];
        extern uint64_t g_sched_fanout_atomic_count[], g_sched_push_wait_cycle[];
        uint64_t lock_atomics = 0, lock_wait = 0;
        PTO2_SCHED_CYCLE_START();
#endif

#if PTO2_SCHED_PROFILING
        pto2_fanout_lock(slot_state, lock_atomics, lock_wait);
#else
        pto2_fanout_lock(slot_state);
#endif
        slot_state.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        PTO2DepListEntry* current = slot_state.fanout_head;  // Protected by fanout_lock
        pto2_fanout_unlock(slot_state);

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
            PTO2TaskSlotState& consumer_slot = *current->slot_state;
#if PTO2_SCHED_PROFILING
            stats.fanout_edges++;
            if (release_fanin_and_check_ready(consumer_slot, fanout_atomics, push_wait, local_bufs)) {
                stats.tasks_enqueued++;
            }
#else
            release_fanin_and_check_ready(consumer_slot, local_bufs);
#endif
            current = current->next;
        }

#if PTO2_SCHED_PROFILING
        g_sched_fanout_atomic_count[thread_idx] += fanout_atomics;
        g_sched_push_wait_cycle[thread_idx] += push_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanout_cycle[thread_idx]);
        return stats;
#endif
    }

    /**
     * Cold path: release producers (fanin traversal) + check self for CONSUMED.
     * Returns fanin edge count for profiling.
     */

#if PTO2_SCHED_PROFILING
    int32_t on_task_release(PTO2TaskSlotState& slot_state, int32_t thread_idx) {
        PTO2_SCHED_CYCLE_START();
        extern uint64_t g_sched_fanin_cycle[], g_sched_fanin_atomic_count[];
        extern uint64_t g_sched_self_atomic_count[];
        extern uint64_t g_sched_self_consumed_cycle[];
        extern uint64_t g_sched_complete_count[];
        uint64_t fanin_atomics = 0;
#else
    int32_t on_task_release(PTO2TaskSlotState& slot_state) {
#endif
        PTO2TaskPayload* payload = slot_state.payload;
        int32_t fanin_edges = payload->fanin_actual_count;
        for (int32_t i = 0; i < fanin_edges; i++) {
#if PTO2_SCHED_PROFILING
            release_producer(*payload->fanin_slot_states[i], fanin_atomics);
#else
            release_producer(*payload->fanin_slot_states[i]);
#endif
        }
#if PTO2_SCHED_PROFILING
        g_sched_fanin_atomic_count[thread_idx] += fanin_atomics;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanin_cycle[thread_idx]);
#endif

        // Self consumed check
#if PTO2_SCHED_PROFILING
        uint64_t self_atomics = 0;
        check_and_handle_consumed(slot_state, self_atomics);
        g_sched_self_atomic_count[thread_idx] += self_atomics;
        PTO2_SCHED_CYCLE_LAP(g_sched_self_consumed_cycle[thread_idx]);
        g_sched_complete_count[thread_idx]++;
#else
        check_and_handle_consumed(slot_state);
#endif
        return fanin_edges;
    }
};

// =============================================================================
// Scheduler API (cold path, defined in pto_scheduler.cpp)
// =============================================================================

bool pto2_scheduler_init(
    PTO2SchedulerState* sched, PTO2SharedMemoryHandle* sm_handle, void* gm_heap_base, uint64_t per_ring_heap_size);
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
    uint64_t lock_wait_cycle;  // spin-wait in fanout_lock
    uint64_t push_wait_cycle;  // CAS contention in push()
    uint64_t pop_wait_cycle;   // CAS contention in pop()

    // Atomic counts per sub-phase
    uint64_t lock_atomic_count;
    uint64_t fanout_atomic_count;
    uint64_t fanin_atomic_count;
    uint64_t self_atomic_count;
    uint64_t pop_atomic_count;

    int64_t complete_count;
};

/**
 * Get and reset scheduler profiling data for a specific thread.
 * Returns accumulated profiling data and resets counters.
 */
PTO2SchedProfilingData pto2_scheduler_get_profiling(int thread_idx);
#endif

#endif  // PTO_SCHEDULER_H
