/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */

/**
 * PTO Runtime2 - Scheduler Interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-resource-shape ready queues
 * 2. Tracking task state (PENDING -> COMPLETED -> CONSUMED)
 * 3. Managing fanin/fanout refcounts for dependency resolution
 * 4. Advancing last_task_alive for heap reclamation
 * 5. Two-stage mixed-task completion (subtask done bits → mixed-task complete)
 *
 * The Scheduler runs on Device AI_CPU and processes:
 * - Task state transitions based on fanin_refcount
 * - Buffer lifecycle based on fanout_refcount
 * - Ring pointer advancement for flow control
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <atomic>

#include "common/core_type.h"
#include "utils/device_arena.h"
#include "pto_async_wait.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

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
    PTO2TaskSlotState *slot_state;
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
    PTO2ReadyQueueSlot *slots;
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

    void reset_for_reuse() {}

    bool push(PTO2TaskSlotState *slot_state) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
                    break;
                }
            } else if (diff < 0) {
                return false;  // Queue full
            }
        }

        slot->slot_state = slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }

    // Batch push: reserve count slots with a single CAS after confirming
    // every target slot is available under the usual Vyukov sequence check.
    void push_batch(PTO2TaskSlotState **items, int count) {
        if (count == 0) return;

        uint64_t pos;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            bool ready = true;
            for (int i = 0; i < count; i++) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + i);
                if (diff != 0) {
                    ready = false;
                    break;
                }
            }
            if (!ready) {
                continue;
            }
            if (enqueue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            slot->slot_state = items[i];
            slot->sequence.store(static_cast<int64_t>(pos + i + 1), std::memory_order_release);
        }
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool push(PTO2TaskSlotState *slot_state, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos);
            atomic_ops += 2;  // enqueue_pos.load + sequence.load
            if (diff == 0) {
                if (enqueue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
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
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }
#endif

    PTO2TaskSlotState *pop() {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    ))
                    break;
            } else if (diff < 0) {
                return nullptr;  // Queue empty
            }
        }

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }

#if PTO2_SCHED_PROFILING
    PTO2TaskSlotState *pop(uint64_t &atomic_count, uint64_t &wait_cycle) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        atomic_count += 2;  // dequeue_pos.load + enqueue_pos.load
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        PTO2ReadyQueueSlot *slot;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            slot = &slots[pos & mask];
            int64_t seq = slot->sequence.load(std::memory_order_acquire);
            int64_t diff = seq - static_cast<int64_t>(pos + 1);
            atomic_ops += 2;  // dequeue_pos.load + sequence.load
            if (diff == 0) {
                if (dequeue_pos.compare_exchange_weak(
                        pos, pos + 1, std::memory_order_relaxed, std::memory_order_relaxed
                    )) {
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

        PTO2TaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }
#endif

    // Batch pop: reserve a contiguous run of ready slots with a single CAS.
    // Returns actual number of items popped (may be less than max_count).
    int pop_batch(PTO2TaskSlotState **out, int max_count) {
        uint64_t pos;
        int count;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                count = -1;
                break;
            }
            if (count == 0) return 0;
            if (count < 0) continue;
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                break;
            }
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
        }
        return count;
    }

#if PTO2_SCHED_PROFILING
    int pop_batch(PTO2TaskSlotState **out, int max_count, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        int count;
        uint64_t t0 = get_sys_cnt_aicpu();
        bool contended = false;
        uint32_t atomic_ops = 0;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            atomic_ops++;  // dequeue_pos.load
            count = 0;
            while (count < max_count) {
                PTO2ReadyQueueSlot *slot = &slots[(pos + count) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + count + 1);
                atomic_ops++;  // sequence.load
                if (diff == 0) {
                    count++;
                    continue;
                }
                if (diff < 0) {
                    break;
                }
                contended = true;
                count = -1;
                break;
            }
            if (count == 0) {
                atomic_count += atomic_ops;
                return 0;
            }
            if (count < 0) {
                continue;
            }
            if (dequeue_pos.compare_exchange_weak(
                    pos, pos + count, std::memory_order_relaxed, std::memory_order_relaxed
                )) {
                atomic_ops++;  // successful CAS
                break;
            }
            contended = true;
            atomic_ops++;  // failed CAS
        }

        for (int i = 0; i < count; i++) {
            PTO2ReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
            atomic_ops++;  // sequence.store
        }
        atomic_count += atomic_ops;
        if (contended) {
            wait_cycle += (get_sys_cnt_aicpu() - t0);
        }
        return count;
    }
#endif
};

// Cold-path ready queue operations (defined in pto_scheduler.cpp). Declared
// as non-member so PTO2ReadyQueue stays a POD-like struct with cache-line
// alignment. Storage is owned by the caller-supplied arena.
//   reserve_layout: declare the slots[] region on the arena (must precede commit)
//   init_from_layout: bind slots pointer from arena.region_ptr(off) and
//                     initialize sequence counters
//   destroy: forget the slots pointer (arena owns the buffer)
size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity);
// Writes everything *except* the arena-internal `slots` pointer field
// (sequences/positions on the slot array, capacity, mask). Uses
// arena.region_ptr(slots_off) only to address the slot array for writes;
// does NOT store the pointer in `queue->slots`. Call
// `ready_queue_wire_arena_pointers` afterwards to set the field itself.
bool ready_queue_init_data_from_layout(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity);
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
void ready_queue_wire_arena_pointers(PTO2ReadyQueue *queue, DeviceArena &arena, size_t slots_off);
void ready_queue_destroy(PTO2ReadyQueue *queue);

/**
 * Statistics returned by mixed-task completion processing
 */
struct CompletionStats {
    int32_t fanout_edges;       // Number of fanout edges traversed (notify consumers)
    int32_t tasks_enqueued;     // Number of consumers that became READY
    int32_t fanin_edges;        // Number of fanin edges traversed (release producers)
    bool mixed_task_completed;  // True only when this callback completed a mixed task
};

/**
 * Layout descriptor produced by PTO2SchedulerState::reserve_layout(). Holds
 * the arena offsets of every sub-region the scheduler needs plus the
 * capacities used at layout time (init_from_layout reuses them).
 */
struct PTO2SchedulerLayout {
    size_t off_ready_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_dep_pool_entries[PTO2_MAX_RING_DEPTH];
    uint64_t ready_queue_capacity;
    int32_t dep_pool_capacities[PTO2_MAX_RING_DEPTH];
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
    PTO2SharedMemoryHeader *sm_header;

    // Per-ring state
    struct alignas(64) RingSchedState {
        // --- Cache Line 0: ring pointer (read-only) + hot path (read-write) ---
        PTO2SharedMemoryRingHeader *ring;
        int32_t last_task_alive;
        std::atomic<int32_t> advance_lock;  // multi-thread CAS

        // --- Cache Line 1+: Orch-side wiring dep_pool ---
        alignas(64) PTO2DepListPool dep_pool;
#if PTO2_PROFILING
        // Published only for scope_stats; orchestrator must not read dep_pool's non-atomic counters directly.
        alignas(64) std::atomic<int32_t> dep_pool_snapshot_tail;
        std::atomic<int32_t> dep_pool_snapshot_top;
#endif

        // Initialize arena-internal data + arena-external pointers; does NOT
        // store dep_pool.base (that lives in the runtime arena and is wired
        // by SchedulerState::wire_arena_pointers). The `ring` field stores
        // the device address of the SM ring header — computed via offset
        // arithmetic, no SM dereference.
        bool init_data_from_layout(void *sm_dev_base, int32_t ring_id);
        void reset_for_reuse(void *sm_dev_base, int32_t ring_id, std::atomic<int32_t> *orch_err);
        void destroy();

        void sync_to_sm() { ring->fc.last_task_alive.store(last_task_alive, std::memory_order_release); }

#if PTO2_PROFILING
        void publish_dep_pool_snapshot() {
            dep_pool_snapshot_tail.store(dep_pool.tail, std::memory_order_release);
            dep_pool_snapshot_top.store(dep_pool.top, std::memory_order_release);
        }

        void read_dep_pool_snapshot(int32_t &tail, int32_t &top) const {
            top = dep_pool_snapshot_top.load(std::memory_order_acquire);
            tail = dep_pool_snapshot_tail.load(std::memory_order_acquire);
            if (tail > top) tail = top;
        }
#endif

        void advance_ring_pointers() {
            int32_t current_task_index = ring->fc.current_task_index.load(std::memory_order_acquire);
            int32_t old_last_task_alive = last_task_alive;

            while (last_task_alive < current_task_index) {
                PTO2TaskSlotState &slot_state = ring->get_slot_state_by_task_id(last_task_alive);
                if (slot_state.task_state.load(std::memory_order_acquire) != PTO2_TASK_CONSUMED) {
                    break;
                }
                last_task_alive++;
            }

            // Eager reset: prepare reclaimed slots for reuse while still hot in cache.
            // Safe because last_task_alive has advanced past these slots but
            // sync_to_sm has not yet published — the orchestrator cannot reuse
            // them until the release store below.
            // Skips payload, task, ring_id — immutable after RingSchedState::init().
            for (int32_t id = old_last_task_alive; id < last_task_alive; id++) {
                ring->get_slot_state_by_task_id(id).reset_for_reuse();
            }

            sync_to_sm();
        }
    } ring_sched_states[PTO2_MAX_RING_DEPTH];

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
    // the dispatch loop and completed inline -- never goes to AICore.
    PTO2ReadyQueue dummy_ready_queue;

    alignas(64) AsyncWaitList async_wait_list;

    // Statistics (cold path, isolated from hot-path fields)
#if PTO2_SCHED_PROFILING
    alignas(64) std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
#endif
    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    // Route a ready slot to the right global queue. Dummy tasks (empty
    // active_mask) live in dummy_ready_queue; everything else goes to the
    // per-shape ready_queues[].
    void push_ready_routed(PTO2TaskSlotState *slot_state) {
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY) {
            dummy_ready_queue.push(slot_state);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(slot_state);
        }
    }

    void check_and_handle_consumed(PTO2TaskSlotState &slot_state) {
        // Read fanout_refcount/fanout_count and flip COMPLETED->CONSUMED under
        // fanout_lock. The orchestrator claims producers (fanout_count++) under the
        // same lock, so the consume decision is serialized against a concurrent
        // claim: either the ++ lands first (count then exceeds refcount, so we do
        // not consume and the producer stays pinned until released) or the consume
        // lands first (the orchestrator then observes CONSUMED and skips the
        // claim). Without this lock a claim racing the consume desyncs the slot's
        // refcount and wedges in-order reclaim.
        bool became_consumed = false;
        slot_state.lock_fanout();
        if (slot_state.fanout_refcount.load(std::memory_order_acquire) == slot_state.fanout_count) {
            PTO2TaskState expected = PTO2_TASK_COMPLETED;
            became_consumed = slot_state.task_state.compare_exchange_strong(
                expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            );
        }
        slot_state.unlock_fanout();
        if (!became_consumed) return;

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        int32_t ring_id = slot_state.ring_id;
        // advance_ring_pointers (and the reset_for_reuse it triggers) MUST run
        // outside fanout_lock: reset_for_reuse stores fanout_lock=0 and would
        // clobber a held lock. Safe here — the slot is CONSUMED and quiescent.
        // Try-lock — if another thread is advancing this ring, it will scan our CONSUMED task
        int32_t expected_lock = 0;
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(
                expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
            ring_sched_states[ring_id].advance_ring_pointers();
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
        }
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void check_and_handle_consumed(PTO2TaskSlotState &slot_state, uint64_t &atomic_count) {
        // See the non-profiling overload for why the read + COMPLETED->CONSUMED
        // flip is serialized against the orchestrator's claim under fanout_lock.
        bool became_consumed = false;
        slot_state.lock_fanout();
        atomic_count += 1;  // lock CAS
        uint32_t fc = slot_state.fanout_count;
        uint32_t rc = slot_state.fanout_refcount.load(std::memory_order_acquire);
        atomic_count += 1;  // fanout_refcount.load (fanout_count is a plain read under lock)
        if (rc == fc) {
            PTO2TaskState expected = PTO2_TASK_COMPLETED;
            became_consumed = slot_state.task_state.compare_exchange_strong(
                expected, PTO2_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            );
            atomic_count += 1;  // CAS
        }
        slot_state.unlock_fanout();
        atomic_count += 1;  // unlock store
        if (!became_consumed) return;

#if PTO2_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        int32_t ring_id = slot_state.ring_id;
        // advance_ring_pointers + reset_for_reuse run outside fanout_lock (reset
        // stores fanout_lock=0). Safe — the slot is CONSUMED and quiescent.
        // Try-lock — if another thread is advancing this ring, it will scan our CONSUMED task
        int32_t expected_lock = 0;
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(
                expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
            ring_sched_states[ring_id].advance_ring_pointers();
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
            atomic_count += 2;  // try-lock CAS + unlock store
        } else {
            atomic_count += 1;  // failed try-lock CAS
        }
    }
#endif

    void release_producer(PTO2TaskSlotState &slot_state) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        check_and_handle_consumed(slot_state);
    }

    // Scope-end release: sets bit31 (PTO2_FANOUT_SCOPE_BIT) instead of bumping a
    // consumer ref. Called exactly once per task from on_scope_end. Keeping it a
    // distinct add lets a consumer release leave the scope bit unset, so "all
    // consumers done but scope still open" stays distinguishable from "fully
    // consumed".
    void release_producer_scope(PTO2TaskSlotState &slot_state) {
        slot_state.fanout_refcount.fetch_add(PTO2_FANOUT_SCOPE_BIT, std::memory_order_acq_rel);
        check_and_handle_consumed(slot_state);
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    void release_producer(PTO2TaskSlotState &slot_state, uint64_t &atomic_count) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        atomic_count += 1;  // fanout_refcount.fetch_add
        check_and_handle_consumed(slot_state, atomic_count);
    }

    void release_producer_scope(PTO2TaskSlotState &slot_state, uint64_t &atomic_count) {
        slot_state.fanout_refcount.fetch_add(PTO2_FANOUT_SCOPE_BIT, std::memory_order_acq_rel);
        atomic_count += 1;  // fanout_refcount.fetch_add
        check_and_handle_consumed(slot_state, atomic_count);
    }
#endif

    bool release_fanin_and_check_ready(PTO2TaskSlotState &slot_state) {
        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // init release, making fanin_count visible — plain load suffices.
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == slot_state.fanin_count) {
            push_ready_routed(&slot_state);
            return true;
        }
        return false;
    }

#if PTO2_ORCH_PROFILING || PTO2_SCHED_PROFILING
    bool release_fanin_and_check_ready(PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &push_wait) {
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
        atomic_count += 1;  // fanin_refcount.fetch_add

        if (new_refcount == slot_state.fanin_count) {
            // Dummy slots go to dummy_ready_queue; everything else to the per-shape
            // ready_queues[]. Use the profiling-aware push so atomic_count / push_wait
            // stay consistent with the non-dummy path.
            PTO2ResourceShape shape = slot_state.active_mask.to_shape();
            if (shape == PTO2ResourceShape::DUMMY) {
                dummy_ready_queue.push(&slot_state, atomic_count, push_wait);
            } else {
                ready_queues[static_cast<int32_t>(shape)].push(&slot_state, atomic_count, push_wait);
            }
            return true;
        }
        return false;
    }
#endif

    int get_ready_tasks_batch(PTO2ResourceShape shape, PTO2TaskSlotState **out, int max_count) {
        return ready_queues[static_cast<int32_t>(shape)].pop_batch(out, max_count);
    }

#if PTO2_SCHED_PROFILING
    int get_ready_tasks_batch(
        PTO2ResourceShape shape, PTO2TaskSlotState **out, int max_count, uint64_t &atomic_count, uint64_t &wait_cycle
    ) {
        return ready_queues[static_cast<int32_t>(shape)].pop_batch(out, max_count, atomic_count, wait_cycle);
    }
#endif

    void on_scope_end(PTO2TaskSlotState **task_slot_states, int32_t count) {
#if PTO2_ORCH_PROFILING
        extern uint64_t g_orch_scope_end_atomic_count;
        if (count > 0) __builtin_prefetch(task_slot_states[0], 1, 0);
        for (int32_t i = 0; i < count; i++) {
            if (i + 1 < count) __builtin_prefetch(task_slot_states[i + 1], 1, 0);
            release_producer_scope(*task_slot_states[i], g_orch_scope_end_atomic_count);
        }
#else
        if (count > 0) __builtin_prefetch(task_slot_states[0], 1, 0);
        for (int32_t i = 0; i < count; i++) {
            if (i + 1 < count) __builtin_prefetch(task_slot_states[i + 1], 1, 0);
            release_producer_scope(*task_slot_states[i]);
        }
#endif
    }

    /**
     * Subtask completion: atomic counter model.
     * Called when a single subtask (AIC, AIV0, or AIV1) finishes on any block.
     * Atomically increments completed_subtasks and checks whether all subtasks
     * across all blocks are done.
     *
     * @return true if this was the last subtask, completing the entire task.
     */
    bool on_subtask_complete(PTO2TaskSlotState &slot_state) {
        int16_t prev = slot_state.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
        return (prev + 1) == slot_state.total_required_subtasks;
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a mixed task are done
     * (i.e., on_subtask_complete returned true).
     * Handles fanout notification, fanin release, and self-consumption check.
     */
#if PTO2_SCHED_PROFILING
    CompletionStats
#else
    void
#endif
    on_task_complete(
        PTO2TaskSlotState &slot_state
#if PTO2_SCHED_PROFILING
        ,
        int thread_idx
#endif
    ) {
#if PTO2_SCHED_PROFILING
        CompletionStats stats = {0, 0, 0, true};
#endif
#if PTO2_SCHED_PROFILING
        extern uint64_t g_sched_lock_cycle[], g_sched_fanout_cycle[];
        extern uint64_t g_sched_lock_atomic_count[], g_sched_lock_wait_cycle[];
        extern uint64_t g_sched_fanout_atomic_count[], g_sched_push_wait_cycle[];
        uint64_t lock_atomics = 0, lock_wait = 0;
        PTO2_SCHED_CYCLE_START();
#endif

#if PTO2_SCHED_PROFILING
        slot_state.lock_fanout(lock_atomics, lock_wait);
#else
        slot_state.lock_fanout();
#endif
        slot_state.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
        PTO2DepListEntry *current = slot_state.fanout_head;  // Protected by fanout_lock
        slot_state.unlock_fanout();

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
            PTO2TaskSlotState &consumer_slot = *current->slot_state;
#if PTO2_SCHED_PROFILING
            stats.fanout_edges++;
            if (release_fanin_and_check_ready(consumer_slot, fanout_atomics, push_wait)) {
                stats.tasks_enqueued++;
            }
#else
            release_fanin_and_check_ready(consumer_slot);
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
    int32_t on_task_release(PTO2TaskSlotState &slot_state, int32_t thread_idx) {
        PTO2_SCHED_CYCLE_START();
        extern uint64_t g_sched_fanin_cycle[], g_sched_fanin_atomic_count[];
        extern uint64_t g_sched_self_atomic_count[];
        extern uint64_t g_sched_self_consumed_cycle[];
        extern uint64_t g_sched_complete_count[];
        uint64_t fanin_atomics = 0;
#else
    int32_t on_task_release(PTO2TaskSlotState &slot_state) {
#endif
        PTO2TaskPayload *payload = slot_state.payload;
        for_each_fanin_slot_state(*payload, [&](PTO2TaskSlotState *producer_slot_state) {
#if PTO2_SCHED_PROFILING
            release_producer(*producer_slot_state, fanin_atomics);
#else
            release_producer(*producer_slot_state);
#endif
        });
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
        return payload->fanin_actual_count;
    }

    // === Cold-path API (defined in pto_scheduler.cpp) ===

    // Phase 1: declare every sub-region (ready_queue slots, dummy queue slots,
    // per-ring dep_pool entries) on the supplied arena.
    // Capacities are baked into the returned layout; init_data_from_layout uses
    // the same values.
    static PTO2SchedulerLayout reserve_layout(DeviceArena &arena, int32_t dep_pool_capacity = PTO2_DEP_LIST_POOL_SIZE);
    static PTO2SchedulerLayout
    reserve_layout(DeviceArena &arena, const int32_t dep_pool_capacities[PTO2_MAX_RING_DEPTH]);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // `sm_dev_base` is the device address of the SM (only stored, never
    // dereferenced here). Safe to call on a host arena that holds the
    // prebuilt image buffer. (The orchestrator counterpart takes
    // task_window_size for ring task_descriptors address arithmetic; the
    // scheduler only needs the SM header / ring header base addresses,
    // both window-size-independent.)
    bool init_data_from_layout(const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);
    void reset_for_reuse(const PTO2SchedulerLayout &layout, void *sm_dev_base);

    // Phase 3b: write the arena-internal pointer fields
    // (ready_queues[].slots, dummy_ready_queue.slots, dep_pool.base for each
    // ring). Called on both host and device sides.
    void wire_arena_pointers(const PTO2SchedulerLayout &layout, DeviceArena &arena);

    // Forget per-region pointers; arena owns the backing memory.
    void destroy();
    void print_stats();
    void print_queues();
};

// Scheduler cold-path API is declared as PTO2SchedulerState member functions.
// See init()/destroy()/print_stats()/print_queues() below the struct definition.

// Short-circuit NotDeferred completions seen during drain so they don't grow
// entries[]. Mirrors the a2a3 impl; see that mirror for the rationale.
inline bool
AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, PTO2TaskSlotState &slot_state) {
#if PTO2_SCHED_PROFILING
    sink.sched->on_task_complete(slot_state, sink.thread_idx);
#else
    sink.sched->on_task_complete(slot_state);
#endif
    if (*sink.deferred_release_count >= sink.deferred_release_capacity) {
        while (*sink.deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
            (void)sink.sched->on_task_release(
                *sink.deferred_release_slot_states[--(*sink.deferred_release_count)], sink.thread_idx
            );
#else
            sink.sched->on_task_release(*sink.deferred_release_slot_states[--(*sink.deferred_release_count)]);
#endif
        }
    }
    sink.deferred_release_slot_states[(*sink.deferred_release_count)++] = &slot_state;
    sink.inline_completed++;
    return true;
}

template <bool Profiling>
inline AsyncPollResult AsyncWaitList::poll_and_complete(
    AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched,
    PTO2TaskSlotState **deferred_release_slot_states, int32_t &deferred_release_count, int32_t deferred_release_capacity
#if PTO2_SCHED_PROFILING
    ,
    int thread_idx
#endif
) {
    AsyncPollResult result;
    if (!try_lock()) return result;

    AsyncWaitList::DrainCompletionSink sink{};
    sink.sched = sched;
    sink.deferred_release_slot_states = deferred_release_slot_states;
    sink.deferred_release_count = &deferred_release_count;
    sink.deferred_release_capacity = deferred_release_capacity;
#if PTO2_SCHED_PROFILING
    sink.thread_idx = thread_idx;
#endif

    int32_t drain_err = PTO2_ERROR_NONE;
    drain_aicore_completion_mailbox_locked(aicore_mailbox, sink, drain_err);
    if (drain_err != PTO2_ERROR_NONE) {
        result.error_code = drain_err;
        unlock();
        return result;
    }
    result.completed += sink.inline_completed;

    for (int32_t i = count - 1; i >= 0; --i) {
        AsyncWaitEntry &entry = entries[i];
        for (int32_t c = 0; c < entry.condition_count; c++) {
            CompletionCondition &cond = entry.conditions[c];
            if (cond.satisfied) continue;
            CompletionPollResult poll = cond.test();
            if (poll.state == CompletionPollState::FAILED) {
                result.error_code = poll.error_code;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
            if (poll.state == CompletionPollState::READY) {
                cond.satisfied = true;
                cond.retire();
                entry.waiting_completion_count--;
            }
        }

        if (entry.normal_done && entry.waiting_completion_count <= 0) {
#if PTO2_SCHED_PROFILING
            sched->on_task_complete(*entry.slot_state, thread_idx);
#else
            sched->on_task_complete(*entry.slot_state);
#endif
            if (deferred_release_count >= deferred_release_capacity) {
                while (deferred_release_count > 0) {
#if PTO2_SCHED_PROFILING
                    (void)sched->on_task_release(*deferred_release_slot_states[--deferred_release_count], thread_idx);
#else
                    sched->on_task_release(*deferred_release_slot_states[--deferred_release_count]);
#endif
                }
            }
            deferred_release_slot_states[deferred_release_count++] = entry.slot_state;
            result.completed++;

            int32_t last = count - 1;
            if (i != last) entries[i] = entries[last];
            count = last;
        }
    }

    unlock();
    return result;
}

// =============================================================================
// Scheduler Profiling Data
// =============================================================================

#if PTO2_SCHED_PROFILING
struct PTO2SchedProfilingData {
    // Sub-phase cycle breakdown within on_task_complete
    uint64_t lock_cycle;           // lock_fanout + state store + unlock
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
PTO2SchedProfilingData scheduler_get_profiling(int thread_idx);
#endif
