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
 * tensormap_and_ringbuffer scheduler interface
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
#include "async_wait.h"
#include "ring_buffer.h"
#include "runtime_types.h"
#include "shared_memory.h"
#include "aicpu/platform_regs.h"
#include "common/memory_barrier.h"

#if SIMPLER_SCHED_PROFILING
#include "aicpu/device_time.h"
#define SCHED_CYCLE_START() uint64_t _st0 = get_sys_cnt_aicpu(), _st1
#define SCHED_CYCLE_LAP(acc)        \
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
struct ChipReadyQueueSlot {
    std::atomic<int64_t> sequence;
    ChipTaskSlotState *slot_state;
    uint64_t task_id_snapshot;  // generation tag for early-dispatch queue entries
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
struct alignas(64) ChipReadyQueue {
    ChipReadyQueueSlot *slots;
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

    bool push(ChipTaskSlotState *slot_state) { return push_tagged(slot_state, 0); }

    bool push_tagged(ChipTaskSlotState *slot_state, uint64_t task_id_snapshot) {
        uint64_t pos;
        ChipReadyQueueSlot *slot;
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
        slot->task_id_snapshot = task_id_snapshot;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }

    // Batch push: reserve count slots with a single CAS after confirming
    // every target slot is available under the usual Vyukov sequence check.
    // Returns false without publishing anything when the queue cannot take all
    // `count` items. A target slot holding an older generation means full and
    // ends the call; a slot a peer has reserved but not yet published is
    // transient and retries, so this only spins while a peer is mid-publish.
    bool push_batch(ChipTaskSlotState **items, int count) { return push_batch_tagged(items, nullptr, count); }

    bool push_batch_tagged(ChipTaskSlotState **items, const uint64_t *task_id_snapshots, int count) {
        if (count == 0) return true;
        if (static_cast<uint64_t>(count) > capacity) return false;

        uint64_t pos;
        while (true) {
            pos = enqueue_pos.load(std::memory_order_relaxed);
            bool ready = true;
            for (int i = 0; i < count; i++) {
                ChipReadyQueueSlot *slot = &slots[(pos + i) & mask];
                int64_t seq = slot->sequence.load(std::memory_order_acquire);
                int64_t diff = seq - static_cast<int64_t>(pos + i);
                if (diff < 0) {
                    return false;  // Queue full
                }
                if (diff > 0) {
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
            ChipReadyQueueSlot *slot = &slots[(pos + i) & mask];
            slot->slot_state = items[i];
            slot->task_id_snapshot = task_id_snapshots == nullptr ? 0 : task_id_snapshots[i];
            slot->sequence.store(static_cast<int64_t>(pos + i + 1), std::memory_order_release);
        }
        return true;
    }

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
    bool push(ChipTaskSlotState *slot_state, uint64_t &atomic_count, uint64_t &wait_cycle) {
        uint64_t pos;
        ChipReadyQueueSlot *slot;
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
        slot->task_id_snapshot = 0;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }
#endif

    ChipTaskSlotState *pop() { return pop_tagged(nullptr); }

    ChipTaskSlotState *pop_tagged(uint64_t *task_id_snapshot) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        ChipReadyQueueSlot *slot;
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

        ChipTaskSlotState *result = slot->slot_state;
        if (task_id_snapshot != nullptr) *task_id_snapshot = slot->task_id_snapshot;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }

#if SIMPLER_SCHED_PROFILING
    ChipTaskSlotState *pop(uint64_t &atomic_count, uint64_t &wait_cycle) {
        // Fast-path: skip slot load when queue is clearly empty
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        atomic_count += 2;  // dequeue_pos.load + enqueue_pos.load
        if (d >= e) {
            return nullptr;
        }

        uint64_t pos;
        ChipReadyQueueSlot *slot;
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

        ChipTaskSlotState *result = slot->slot_state;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }
#endif

    // Batch pop: reserve a contiguous run of ready slots with a single CAS.
    // Returns actual number of items popped (may be less than max_count).
    int pop_batch(ChipTaskSlotState **out, int max_count) { return pop_batch_tagged(out, nullptr, max_count); }

    int pop_batch_tagged(ChipTaskSlotState **out, uint64_t *task_id_snapshots, int max_count) {
        uint64_t pos;
        int count;
        while (true) {
            pos = dequeue_pos.load(std::memory_order_relaxed);
            count = 0;
            while (count < max_count) {
                ChipReadyQueueSlot *slot = &slots[(pos + count) & mask];
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
            ChipReadyQueueSlot *slot = &slots[(pos + i) & mask];
            out[i] = slot->slot_state;
            if (task_id_snapshots != nullptr) task_id_snapshots[i] = slot->task_id_snapshot;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
        }
        return count;
    }

#if SIMPLER_SCHED_PROFILING
    int pop_batch(ChipTaskSlotState **out, int max_count, uint64_t &atomic_count, uint64_t &wait_cycle) {
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
                ChipReadyQueueSlot *slot = &slots[(pos + count) & mask];
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
            ChipReadyQueueSlot *slot = &slots[(pos + i) & mask];
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

// Cold-path ready queue operations (defined in scheduler.cpp). Declared
// as non-member so ChipReadyQueue stays a POD-like struct with cache-line
// alignment. Storage is owned by the caller-supplied arena.
//   reserve_layout: declare the slots[] region on the arena (must precede commit)
//   init_data_from_layout: initialize sequence counters and queue metadata
//   wire_arena_pointers: bind slots pointer from arena.region_ptr(off)
//   destroy: forget the slots pointer (arena owns the buffer)
size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity);
// Writes everything *except* the arena-internal `slots` pointer field
// (sequences/positions on the slot array, capacity, mask). Uses
// arena.region_ptr(slots_off) only to address the slot array for writes;
// does NOT store the pointer in `queue->slots`. Call
// `ready_queue_wire_arena_pointers` afterwards to set the field itself.
bool ready_queue_init_data_from_layout(ChipReadyQueue *queue, DeviceArena &arena, size_t slots_off, uint64_t capacity);
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
void ready_queue_wire_arena_pointers(ChipReadyQueue *queue, DeviceArena &arena, size_t slots_off);
void ready_queue_destroy(ChipReadyQueue *queue);

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
 * Layout descriptor produced by SchedulerState::reserve_layout(). Holds
 * the arena offsets of every sub-region the scheduler needs plus the
 * capacities used at layout time (init_data_from_layout reuses them).
 */
struct SchedulerLayout {
    size_t off_ready_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_ready_sync_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_early_dispatch_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_early_sync_start_queue_slots;
    size_t off_dep_pool_entries[CHIP_MAX_RING_DEPTH];
    uint64_t ready_queue_capacity;
    int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH];
};

/**
 * Scheduler state structure
 *
 * Contains dynamic state updated during task execution.
 * Separated from shared memory for cache efficiency.
 * Hot-path methods are defined inline (implicitly inline as member functions).
 */
struct SchedulerState {
    // Shared memory access
    SharedMemoryHeader *sm_header;

    // Per-ring state
    struct alignas(64) RingSchedState {
        static constexpr int32_t PUBLISH_INTERVAL_K = 16;

        // --- Cache Line 0: ring pointer (read-only) + hot path (read-write) ---
        SharedMemoryRingHeader *ring;
        int32_t last_task_alive;
        // Publication may trail local reclamation by PUBLISH_INTERVAL_K - 1
        // tasks only while no reclaim consumer is blocked on the watermark.
        int32_t last_published_to_sm{0};
        bool publication_batching_enabled{false};
        std::atomic<int32_t> advance_lock;  // multi-thread CAS

        // --- Cache Line 1+: Orch-side wiring dep_pool ---
        alignas(64) DepListPool dep_pool;
#if SIMPLER_DFX
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

        void sync_to_sm(bool force = false) {
            if (last_task_alive == last_published_to_sm) return;
            if (publication_batching_enabled && !force && last_task_alive - last_published_to_sm < PUBLISH_INTERVAL_K) {
                return;
            }
            ring->fc.last_task_alive.store(last_task_alive, std::memory_order_release);
            last_published_to_sm = last_task_alive;
        }

#if SIMPLER_DFX
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

        void advance_ring_pointers(bool force_publish = false) {
            int32_t current_task_index = ring->fc.current_task_index.load(std::memory_order_acquire);
            int32_t old_last_task_alive = last_task_alive;

            while (last_task_alive < current_task_index) {
                ChipTaskSlotState &slot_state = ring->get_slot_state_by_task_id(last_task_alive);
                if (slot_state.task_state.load(std::memory_order_acquire) != CHIP_TASK_CONSUMED) {
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

            sync_to_sm(force_publish || last_task_alive == current_task_index);
        }
    } ring_sched_states[CHIP_MAX_RING_DEPTH];

    alignas(64) std::atomic<uint32_t> advance_pending_mask;
    alignas(64) std::atomic<uint32_t> publication_request_mask;
    std::atomic<uint32_t> publication_ack_mask;

    // Ready queues remain global (scheduling is ring-agnostic)
    ChipReadyQueue ready_queues[NUM_RESOURCE_SHAPES];

    // Ready sync_start queues, one per shape. A ready sync_start cohort parks here
    // instead of ready_queues[] so the dispatch loop can drain it as a strict Tier-0
    // (sync_start > MIX > C/V) before any regular ready task takes a core, while
    // reusing the same per-shape dispatch_shape machinery (fits-local inline vs
    // stop-the-world drain, per-core MIX placement, head-start spacing).
    ChipReadyQueue ready_sync_queues[NUM_RESOURCE_SHAPES];

    // Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
    // the dispatch loop and completed inline -- never goes to AICore.
    ChipReadyQueue dummy_ready_queue;

    alignas(64) AsyncWaitList async_wait_list;

    // Statistics (cold path, isolated from hot-path fields)
#if SIMPLER_SCHED_PROFILING
    alignas(64) std::atomic<int64_t> tasks_completed;
    std::atomic<int64_t> tasks_consumed;
#endif
    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    // Route a ready slot to the right global queue. Dep-only tasks — DUMMY-shaped
    // (empty active_mask) or a task whose dispatch predicate fails — live in
    // dummy_ready_queue and are retired inline; a ready sync_start cohort goes to
    // the per-shape ready_sync_queues[] (drained as Tier-0); everything else to
    // ready_queues[].
    void push_ready_routed(ChipTaskSlotState *slot_state) {
        ResourceShape shape = slot_state->active_mask.to_shape();
        if (shape == ResourceShape::DUMMY ||
            (slot_state->task_attrs.has_predicate() && !slot_state->payload->predicate.pass())) {
            dummy_ready_queue.push(slot_state);
        } else if (slot_state->task_attrs.requires_sync_start()) {
            ready_sync_queues[static_cast<int32_t>(shape)].push(slot_state);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(slot_state);
        }
    }

    static uint32_t ring_advance_pending_bit(int32_t ring_id) { return ring_mask_bit(ring_id); }

    void set_publication_batching_enabled(bool enabled) {
        for (auto &ring_sched : ring_sched_states) {
            ring_sched.publication_batching_enabled = enabled;
        }
    }

    // Failed consumed-head advances remain deferred until a no-progress drain.
    void mark_ring_advance_pending(int32_t ring_id) {
        advance_pending_mask.fetch_or(ring_advance_pending_bit(ring_id), std::memory_order_release);
    }

    bool drain_pending_ring_advances() {
        uint32_t pending = advance_pending_mask.load(std::memory_order_acquire);
        if (pending == 0) return false;

        bool advanced = false;
        for (int32_t ring_id = 0; ring_id < CHIP_MAX_RING_DEPTH; ring_id++) {
            uint32_t bit = ring_advance_pending_bit(ring_id);
            if ((pending & bit) == 0) continue;

            auto &ring_sched = ring_sched_states[ring_id];
            int32_t expected_lock = 0;
            if (!ring_sched.advance_lock.compare_exchange_strong(
                    expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
                )) {
                continue;
            }

            advance_pending_mask.fetch_and(~bit, std::memory_order_acq_rel);
            int32_t before = ring_sched.last_task_alive;
            ring_sched.advance_ring_pointers(true);
            advanced = advanced || ring_sched.last_task_alive != before;
            ring_sched.advance_lock.store(0, std::memory_order_release);
        }
        return advanced;
    }

    bool drain_publication_requests() {
        uint32_t requests = publication_request_mask.load(std::memory_order_acquire);
        if (requests == 0) return false;

        bool advanced = false;
        for (int32_t ring_id = 0; ring_id < CHIP_MAX_RING_DEPTH; ring_id++) {
            uint32_t bit = ring_advance_pending_bit(ring_id);
            if ((requests & bit) == 0) continue;

            auto &ring_sched = ring_sched_states[ring_id];
            int32_t expected_lock = 0;
            if (!ring_sched.advance_lock.compare_exchange_strong(
                    expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
                )) {
                continue;
            }

            publication_request_mask.fetch_and(~bit, std::memory_order_acq_rel);
            int32_t before_alive = ring_sched.last_task_alive;
            int32_t before_published = ring_sched.last_published_to_sm;
            ring_sched.advance_ring_pointers(true);
            publication_ack_mask.fetch_or(bit, std::memory_order_release);
            advanced = advanced || ring_sched.last_task_alive != before_alive ||
                       ring_sched.last_published_to_sm != before_published;
            ring_sched.advance_lock.store(0, std::memory_order_release);
        }
        return advanced;
    }

    bool try_claim_ready_once(ChipTaskSlotState &slot_state) {
        uint8_t flags = slot_state.lifecycle_flags.load(std::memory_order_acquire);
        for (;;) {
            if ((flags & READY_CLAIMED) != 0) return false;
            uint8_t desired_flags = flags | READY_CLAIMED;
            if (slot_state.lifecycle_flags.compare_exchange_weak(
                    flags, desired_flags, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                return true;
            }
        }
    }

    void check_and_handle_consumed(ChipTaskSlotState &slot_state) {
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
            ChipTaskState expected = CHIP_TASK_COMPLETED;
            became_consumed = slot_state.task_state.compare_exchange_strong(
                expected, CHIP_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            );
        }
        slot_state.unlock_fanout();
        if (!became_consumed) return;

#if SIMPLER_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        int32_t ring_id = slot_state.ring_id;
        // Runs outside fanout_lock: reset_for_reuse stores fanout_lock=0.
        int32_t expected_lock = 0;
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(
                expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
            ring_sched_states[ring_id].advance_ring_pointers();
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
        } else {
            mark_ring_advance_pending(ring_id);
        }
    }

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
    void check_and_handle_consumed(ChipTaskSlotState &slot_state, uint64_t &atomic_count) {
        // See the non-profiling overload for why the read + COMPLETED->CONSUMED
        // flip is serialized against the orchestrator's claim under fanout_lock.
        bool became_consumed = false;
        slot_state.lock_fanout();
        atomic_count += 1;  // lock CAS
        uint32_t fc = slot_state.fanout_count;
        uint32_t rc = slot_state.fanout_refcount.load(std::memory_order_acquire);
        atomic_count += 1;  // fanout_refcount.load (fanout_count is a plain read under lock)
        if (rc == fc) {
            ChipTaskState expected = CHIP_TASK_COMPLETED;
            became_consumed = slot_state.task_state.compare_exchange_strong(
                expected, CHIP_TASK_CONSUMED, std::memory_order_acq_rel, std::memory_order_acquire
            );
            atomic_count += 1;  // CAS
        }
        slot_state.unlock_fanout();
        atomic_count += 1;  // unlock store
        if (!became_consumed) return;

#if SIMPLER_SCHED_PROFILING
        tasks_consumed.fetch_add(1, std::memory_order_relaxed);
#endif

        int32_t ring_id = slot_state.ring_id;
        // Runs outside fanout_lock: reset_for_reuse stores fanout_lock=0.
        int32_t expected_lock = 0;
        if (ring_sched_states[ring_id].advance_lock.compare_exchange_strong(
                expected_lock, 1, std::memory_order_acquire, std::memory_order_relaxed
            )) {
            ring_sched_states[ring_id].advance_ring_pointers();
            ring_sched_states[ring_id].advance_lock.store(0, std::memory_order_release);
            atomic_count += 2;  // try-lock CAS + unlock store
        } else {
            mark_ring_advance_pending(ring_id);
            atomic_count += 2;  // failed try-lock CAS + pending mark
        }
    }
#endif

    void release_producer(ChipTaskSlotState &slot_state) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        check_and_handle_consumed(slot_state);
    }

    // Scope-end release: sets bit31 (FANOUT_SCOPE_BIT) instead of bumping a
    // consumer ref. Called exactly once per task from on_scope_end. Keeping it a
    // distinct add lets a consumer release leave the scope bit unset, so "all
    // consumers done but scope still open" stays distinguishable from "fully
    // consumed".
    void release_producer_scope(ChipTaskSlotState &slot_state) {
        slot_state.fanout_refcount.fetch_add(FANOUT_SCOPE_BIT, std::memory_order_acq_rel);
        check_and_handle_consumed(slot_state);
    }

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
    void release_producer(ChipTaskSlotState &slot_state, uint64_t &atomic_count) {
        slot_state.fanout_refcount.fetch_add(1, std::memory_order_acq_rel);
        atomic_count += 1;  // fanout_refcount.fetch_add
        check_and_handle_consumed(slot_state, atomic_count);
    }

    void release_producer_scope(ChipTaskSlotState &slot_state, uint64_t &atomic_count) {
        slot_state.fanout_refcount.fetch_add(FANOUT_SCOPE_BIT, std::memory_order_acq_rel);
        atomic_count += 1;  // fanout_refcount.fetch_add
        check_and_handle_consumed(slot_state, atomic_count);
    }
#endif

    // =========================================================================
    // Early-dispatch (#1079 foundation + #1304 sync_start early queue)
    // =========================================================================

    struct EarlyDispatchDoorbell {
        uint64_t addr{0};
        uint32_t token{0};
    };
    EarlyDispatchDoorbell early_dispatch_doorbell_table[EARLY_DISPATCH_CORE_MASK_WORDS * 64]{};
    ChipReadyQueue early_dispatch_queues[NUM_RESOURCE_SHAPES];
    // Shape-agnostic sync_start candidates require an all-or-nothing owner
    // decision. The owner stages locally when one tracker can hold the complete
    // cohort and falls back to the generation-tagged global drain otherwise.
    ChipReadyQueue early_sync_start_queue;

    static inline void ring_one_doorbell(uint64_t reg_addr, uint32_t token) {
        volatile uint64_t *dmb = reinterpret_cast<volatile uint64_t *>(get_reg_ptr(reg_addr, RegId::DATA_MAIN_BASE));
        uint64_t tk = static_cast<uint64_t>(token);
        *dmb = (tk << 32) | tk;  // 64-bit STR: high=low=token releases the gated AICore
    }

    inline void ring_staged_doorbell_bits(int word, uint64_t bits) {
        while (bits != 0) {
            int core_id = word * 64 + __builtin_ctzll(bits);
            bits &= bits - 1;
            ring_one_doorbell(
                early_dispatch_doorbell_table[core_id].addr, early_dispatch_doorbell_table[core_id].token
            );
        }
    }

    static inline uint64_t claim_all_staged_doorbell_bits(std::atomic<uint64_t> &mask) {
        return mask.exchange(0, std::memory_order_seq_cst);
    }

    static inline uint64_t claim_late_staged_doorbell_bits(std::atomic<uint64_t> &mask, uint64_t candidates) {
        return mask.fetch_and(~candidates, std::memory_order_seq_cst) & candidates;
    }

    static inline bool should_gate_early_dispatch(bool force_gate, uint8_t early_dispatch_state) {
        return force_gate || early_dispatch_state == EARLY_DISPATCH_STAGING;
    }

    static inline bool
    ring_claimed_local_doorbell(uint64_t claimed_word, int core_id, uint64_t reg_addr, uint32_t token) {
        if ((claimed_word & (1ULL << (core_id & 63))) == 0) return false;
        ring_one_doorbell(reg_addr, token);
        return true;
    }

    static inline bool try_claim_early_dispatch_launch(TaskPayload &payload) {
        uint8_t expected = EARLY_DISPATCH_LAUNCH_NONE;
        return payload.early_dispatch_launch_state.compare_exchange_strong(
            expected, EARLY_DISPATCH_LAUNCH_RINGING, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
    }

    inline void record_published_blocks(ChipTaskSlotState &slot_state, int32_t count) {
        if (count <= 0 || !slot_state.task_attrs.allow_early_resolve()) return;
        slot_state.payload->published_block_count.fetch_add(static_cast<int16_t>(count), std::memory_order_seq_cst);
    }

    inline void ring_all_staged_doorbells(ChipTaskSlotState &slot_state) {
        for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
            uint64_t bits = slot_state.payload->staged_core_mask[w].load(std::memory_order_seq_cst);
            while (bits != 0) {
                int core_id = w * 64 + __builtin_ctzll(bits);
                bits &= bits - 1;
                ring_one_doorbell(
                    early_dispatch_doorbell_table[core_id].addr, early_dispatch_doorbell_table[core_id].token
                );
            }
        }
    }

    static inline bool try_claim_early_sync_drain(TaskPayload &payload) {
        uint8_t expected = EARLY_SYNC_DRAIN_NONE;
        return payload.early_sync_drain_state.compare_exchange_strong(
            expected, EARLY_SYNC_DRAIN_OWNER, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
    }

    static inline bool owns_early_sync_drain(const TaskPayload &payload) {
        return (payload.early_sync_drain_state.load(std::memory_order_acquire) & EARLY_SYNC_DRAIN_OWNER) != 0;
    }

    static inline void mark_early_sync_drain_armed(TaskPayload &payload) {
        payload.early_sync_drain_state.fetch_or(EARLY_SYNC_DRAIN_ARMED, std::memory_order_seq_cst);
    }

    static inline bool publish_ready_to_early_sync_drain(TaskPayload &payload) {
        uint8_t previous = payload.early_sync_drain_state.fetch_or(EARLY_SYNC_DRAIN_READY, std::memory_order_seq_cst);
        return (previous & EARLY_SYNC_DRAIN_OWNER) != 0;
    }

    inline void cancel_early_sync_drain(ChipTaskSlotState &slot_state) {
        uint8_t previous =
            slot_state.payload->early_sync_drain_state.exchange(EARLY_SYNC_DRAIN_NONE, std::memory_order_seq_cst);
        if ((previous & EARLY_SYNC_DRAIN_OWNER) == 0) return;
        if ((previous & EARLY_SYNC_DRAIN_READY) != 0) {
            push_ready_routed(&slot_state);
            return;
        }
        if (slot_state.payload->early_dispatch_state.load(std::memory_order_seq_cst) == EARLY_DISPATCH_STAGING) {
            early_sync_start_queue.push_tagged(&slot_state, static_cast<uint64_t>(slot_state.task->task_id.raw));
        }
    }

    static inline void finish_early_sync_drain(TaskPayload &payload) {
        uint8_t state = payload.early_sync_drain_state.load(std::memory_order_seq_cst);
        while ((state & EARLY_SYNC_DRAIN_OWNER) != 0 && (state & EARLY_SYNC_DRAIN_COMPLETE) == 0) {
            uint8_t desired = state | EARLY_SYNC_DRAIN_COMPLETE;
            if (payload.early_sync_drain_state.compare_exchange_weak(
                    state, desired, std::memory_order_seq_cst, std::memory_order_seq_cst
                )) {
                return;
            }
        }
    }

    inline bool maybe_rendezvous_ring(ChipTaskSlotState &slot_state) {
        // Staging publishes the complete mask before seeding running_slot_count.
        // Read the seed first: observing the final seq_cst seed then orders every
        // mask read after all of the stager's mask updates. Reading the mask first
        // could pair a partial mask with the later final seed and ring too early.
        int32_t running_cores = slot_state.payload->running_slot_count.load(std::memory_order_seq_cst);
        int32_t staged_cores = 0;
        for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++)
            staged_cores +=
                __builtin_popcountll(slot_state.payload->staged_core_mask[w].load(std::memory_order_seq_cst));
        if (staged_cores == 0) return false;
        if (running_cores != staged_cores) return false;
        if (slot_state.payload->early_dispatch_state.load(std::memory_order_seq_cst) != EARLY_DISPATCH_DISPATCHED)
            return false;
        if (!try_claim_early_dispatch_launch(*slot_state.payload)) return false;
        ring_all_staged_doorbells(slot_state);
        wmb();
        slot_state.payload->early_dispatch_launch_state.store(
            EARLY_DISPATCH_LAUNCH_COMPLETE, std::memory_order_release
        );
        return true;
    }

    inline bool retry_sync_start_rendezvous_after_staging(ChipTaskSlotState &slot_state) {
        if (!maybe_rendezvous_ring(slot_state)) return false;
        propagate_dispatch_fanin(slot_state);
        return true;
    }

    inline void try_enqueue_early_dispatch_candidate(ChipTaskSlotState &consumer) {
        if (consumer.task_attrs.has_predicate()) return;
        ResourceShape shape = consumer.active_mask.to_shape();
        if (shape != ResourceShape::AIC && shape != ResourceShape::AIV && shape != ResourceShape::MIX) {
            return;
        }

        uint8_t expected = EARLY_DISPATCH_NONE;
        if (!consumer.payload->early_dispatch_state.compare_exchange_strong(
                expected, EARLY_DISPATCH_STAGING, std::memory_order_seq_cst, std::memory_order_seq_cst
            )) {
            return;
        }

        uint64_t task_id = static_cast<uint64_t>(consumer.task->task_id.raw);
        bool queued = consumer.task_attrs.requires_sync_start() ?
                          early_sync_start_queue.push_tagged(&consumer, task_id) :
                          early_dispatch_queues[static_cast<int32_t>(shape)].push_tagged(&consumer, task_id);
        if (!queued) {
            // The candidate was never published to an early-dispatch drain, so
            // drop the speculative claim and let readiness take the ordinary
            // queue path. A concurrent producer release may already have moved
            // STAGING to DISPATCHED; that path then owns the transition.
            expected = EARLY_DISPATCH_STAGING;
            consumer.payload->early_dispatch_state.compare_exchange_strong(
                expected, EARLY_DISPATCH_NONE, std::memory_order_seq_cst, std::memory_order_seq_cst
            );
        }
    }

    // Only codegen-flagged (direct) producers propagate: a task's successors
    // early-dispatch off its DIRECT producers' marks, never an inherited chain
    // (a2a3 #1285 — a5 never had auto-chain / spec_chain_*).
    void propagate_dispatch_fanin(ChipTaskSlotState &p) {
        if (!p.task_attrs.allow_early_resolve()) return;
        if (p.payload->published_block_count.load(std::memory_order_seq_cst) < p.logical_block_num) return;
        if (p.payload->early_dispatch_state.load(std::memory_order_acquire) == EARLY_DISPATCH_STAGING) return;
        uint8_t launch_state = p.payload->early_dispatch_launch_state.load(std::memory_order_acquire);
        if (launch_state == EARLY_DISPATCH_LAUNCH_RINGING) return;
        if (p.task_attrs.requires_sync_start()) {
            bool was_pre_staged = false;
            for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
                was_pre_staged |= p.payload->staged_core_mask[w].load(std::memory_order_acquire) != 0;
            }
            if (was_pre_staged && launch_state != EARLY_DISPATCH_LAUNCH_COMPLETE) return;
        }
        if (p.has_dispatch_propagated()) return;
        p.lock_fanout();
        if (!p.try_mark_dispatch_propagated()) {
            p.unlock_fanout();
            return;
        }
        DepListEntry *edge = p.fanout_head;
        p.unlock_fanout();
        for (; edge != nullptr; edge = edge->next) {
            ChipTaskSlotState *c = edge->slot_state;
            if (c->task_attrs.has_predicate()) continue;
            // early_dispatch_target() is the WAIT-edge count plus any unit held
            // by a reduced-away unflagged producer: only DEP_WAIT producers link
            // onto fanout_head and bump dispatch_fanin, so RETAIN-only and
            // reduction-dropped edges (still counted by fanin_actual_count) stay
            // out of the count while still being able to hold the target short.
            int32_t nf = c->payload->dispatch_fanin.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (nf != c->payload->early_dispatch_target()) continue;
            try_enqueue_early_dispatch_candidate(*c);
        }
    }

    struct EarlyDispatchReleaseSink {
        static constexpr int CAP = 32;
        ChipTaskSlotState *items[CAP];
        int n = 0;
        inline bool push(ChipTaskSlotState *s) {
            if (n >= CAP) return false;
            items[n++] = s;
            return true;
        }
    };

    inline bool try_early_dispatch_release(ChipTaskSlotState &slot_state, EarlyDispatchReleaseSink *sink = nullptr) {
        if (slot_state.task_attrs.has_predicate()) return false;
        uint8_t expect = EARLY_DISPATCH_NONE;
        if (slot_state.payload->early_dispatch_state.compare_exchange_strong(
                expect, EARLY_DISPATCH_DISPATCHED, std::memory_order_seq_cst, std::memory_order_seq_cst
            )) {
            return false;
        }
        if (expect != EARLY_DISPATCH_STAGING) return true;
        bool sync_start = slot_state.task_attrs.requires_sync_start();
        if (!sync_start && !try_claim_early_dispatch_launch(*slot_state.payload)) return true;
        expect = EARLY_DISPATCH_STAGING;
        slot_state.payload->early_dispatch_state.compare_exchange_strong(
            expect, EARLY_DISPATCH_DISPATCHED, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
        bool launched = true;
        if (sync_start) {
            launched = maybe_rendezvous_ring(slot_state);
        } else {
            for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
                uint64_t owned = claim_all_staged_doorbell_bits(slot_state.payload->staged_core_mask[w]);
                ring_staged_doorbell_bits(w, owned);
            }
            wmb();
            slot_state.payload->early_dispatch_launch_state.store(
                EARLY_DISPATCH_LAUNCH_COMPLETE, std::memory_order_seq_cst
            );
        }
        if (launched) {
            if (sink == nullptr || !sink->push(&slot_state)) {
                propagate_dispatch_fanin(slot_state);
            }
        }
        return slot_state.next_block_idx.load(std::memory_order_seq_cst) >= slot_state.logical_block_num;
    }

    bool route_ready_once(ChipTaskSlotState &slot_state, EarlyDispatchReleaseSink *sink = nullptr) {
        if (!try_claim_ready_once(slot_state)) return false;

        // Early-dispatch: pre-staged tasks are released by doorbell
        // here, skipping the ready-queue round-trip entirely.
        bool early_handled = try_early_dispatch_release(slot_state, sink);
        if (slot_state.task_attrs.requires_sync_start()) {
            bool drain_owned = publish_ready_to_early_sync_drain(*slot_state.payload);
            if (early_handled || drain_owned) return true;
        } else if (early_handled) {
            return true;
        }

        ResourceShape shape = slot_state.active_mask.to_shape();
        if (shape == ResourceShape::DUMMY ||
            (slot_state.task_attrs.has_predicate() && !slot_state.payload->predicate.pass())) {
            dummy_ready_queue.push(&slot_state);
        } else if (slot_state.task_attrs.requires_sync_start()) {
            ready_sync_queues[static_cast<int32_t>(shape)].push(&slot_state);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(&slot_state);
        }
        return true;
    }

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
    bool route_ready_once(
        ChipTaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &push_wait,
        EarlyDispatchReleaseSink *sink = nullptr
    ) {
        uint8_t flags = slot_state.lifecycle_flags.load(std::memory_order_acquire);
        atomic_count += 1;
        for (;;) {
            if ((flags & READY_CLAIMED) != 0) return false;
            uint8_t desired_flags = flags | READY_CLAIMED;
            if (slot_state.lifecycle_flags.compare_exchange_weak(
                    flags, desired_flags, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                atomic_count += 1;
                break;
            }
            atomic_count += 1;
        }
        bool early_handled = try_early_dispatch_release(slot_state, sink);
        if (slot_state.task_attrs.requires_sync_start()) {
            bool drain_owned = publish_ready_to_early_sync_drain(*slot_state.payload);
            if (early_handled || drain_owned) return true;
        } else if (early_handled) {
            return true;
        }
        ResourceShape shape = slot_state.active_mask.to_shape();
        if (shape == ResourceShape::DUMMY ||
            (slot_state.task_attrs.has_predicate() && !slot_state.payload->predicate.pass())) {
            dummy_ready_queue.push(&slot_state, atomic_count, push_wait);
        } else if (slot_state.task_attrs.requires_sync_start()) {
            ready_sync_queues[static_cast<int32_t>(shape)].push(&slot_state, atomic_count, push_wait);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(&slot_state, atomic_count, push_wait);
        }
        return true;
    }
#endif

    bool release_fanin_and_check_ready(ChipTaskSlotState &slot_state, EarlyDispatchReleaseSink *sink = nullptr) {
        // Atomically increment fanin_refcount and check if all producers are done
        // ACQ_REL on fanin_refcount already synchronizes with the orchestrator's
        // init release, making fanin_count visible — plain load suffices.
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;

        if (new_refcount == slot_state.fanin_count) {
            return route_ready_once(slot_state, sink);
        }
        return false;
    }

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
    bool release_fanin_and_check_ready(
        ChipTaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &push_wait,
        EarlyDispatchReleaseSink *sink = nullptr
    ) {
        int32_t new_refcount = slot_state.fanin_refcount.fetch_add(1, std::memory_order_acq_rel) + 1;
        atomic_count += 1;  // fanin_refcount.fetch_add

        if (new_refcount == slot_state.fanin_count) {
            return route_ready_once(slot_state, atomic_count, push_wait, sink);
        }
        return false;
    }
#endif

    int get_ready_tasks_batch(ChipReadyQueue *queues, ResourceShape shape, ChipTaskSlotState **out, int max_count) {
        return queues[static_cast<int32_t>(shape)].pop_batch(out, max_count);
    }

#if SIMPLER_SCHED_PROFILING
    int get_ready_tasks_batch(
        ChipReadyQueue *queues, ResourceShape shape, ChipTaskSlotState **out, int max_count, uint64_t &atomic_count,
        uint64_t &wait_cycle
    ) {
        return queues[static_cast<int32_t>(shape)].pop_batch(out, max_count, atomic_count, wait_cycle);
    }
#endif

    void on_scope_end(ChipTaskSlotState **task_slot_states, int32_t count) {
#if SIMPLER_ORCH_PROFILING
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
    bool on_subtask_complete(ChipTaskSlotState &slot_state) {
        int16_t prev = slot_state.completed_subtasks.fetch_add(1, std::memory_order_acq_rel);
        return (prev + 1) == slot_state.total_required_subtasks;
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a mixed task are done
     * (i.e., on_subtask_complete returned true).
     * Handles fanout notification, fanin release, and self-consumption check.
     */
#if SIMPLER_SCHED_PROFILING
    CompletionStats
#else
    uint32_t
#endif
    on_task_complete(
        ChipTaskSlotState &slot_state
#if SIMPLER_SCHED_PROFILING
        ,
        int thread_idx
#endif
    ) {
#if SIMPLER_SCHED_PROFILING
        CompletionStats stats = {0, 0, 0, true};
#else
        uint32_t consumer_walk_count = 0;
#endif
#if SIMPLER_SCHED_PROFILING
        extern uint64_t g_sched_lock_cycle[], g_sched_fanout_cycle[];
        extern uint64_t g_sched_lock_atomic_count[], g_sched_lock_wait_cycle[];
        extern uint64_t g_sched_fanout_atomic_count[], g_sched_push_wait_cycle[];
        uint64_t lock_atomics = 0, lock_wait = 0;
        SCHED_CYCLE_START();
#endif

#if SIMPLER_SCHED_PROFILING
        slot_state.lock_fanout(lock_atomics, lock_wait);
#else
        slot_state.lock_fanout();
#endif
        slot_state.mark_completed();
        DepListEntry *current = slot_state.fanout_head;  // Protected by fanout_lock
        slot_state.unlock_fanout();

#if SIMPLER_SCHED_PROFILING
        lock_atomics += 3;  // task_state.store + lifecycle_flags.fetch_or + unlock.store
        g_sched_lock_atomic_count[thread_idx] += lock_atomics;
        g_sched_lock_wait_cycle[thread_idx] += lock_wait;
        SCHED_CYCLE_LAP(g_sched_lock_cycle[thread_idx]);
#endif

        // Fanout: notify consumers
#if SIMPLER_SCHED_PROFILING
        uint64_t fanout_atomics = 0, push_wait = 0;
#endif
        EarlyDispatchReleaseSink rel_sink;
        while (current != nullptr) {
            ChipTaskSlotState &consumer_slot = *current->slot_state;
#if SIMPLER_SCHED_PROFILING
            stats.fanout_edges++;
            if (release_fanin_and_check_ready(consumer_slot, fanout_atomics, push_wait, &rel_sink)) {
                stats.tasks_enqueued++;
            }
#else
            consumer_walk_count++;
            release_fanin_and_check_ready(consumer_slot, &rel_sink);
#endif
            current = current->next;
        }
        for (int i = 0; i < rel_sink.n; i++) {
            propagate_dispatch_fanin(*rel_sink.items[i]);
        }

#if SIMPLER_SCHED_PROFILING
        g_sched_fanout_atomic_count[thread_idx] += fanout_atomics;
        g_sched_push_wait_cycle[thread_idx] += push_wait;
        SCHED_CYCLE_LAP(g_sched_fanout_cycle[thread_idx]);
        return stats;
#else
        return consumer_walk_count;
#endif
    }

    /**
     * Cold path: release producers (fanin traversal) + check self for CONSUMED.
     * Returns the number of retained (DEP_RETAIN) producers actually released —
     * ordering-only edges dropped their pin at wiring and are skipped here.
     */

#if SIMPLER_SCHED_PROFILING
    int32_t on_task_release(ChipTaskSlotState &slot_state, int32_t thread_idx) {
        SCHED_CYCLE_START();
        extern uint64_t g_sched_fanin_cycle[], g_sched_fanin_atomic_count[];
        extern uint64_t g_sched_self_atomic_count[];
        extern uint64_t g_sched_self_consumed_cycle[];
        extern uint64_t g_sched_complete_count[];
        uint64_t fanin_atomics = 0;
#else
    int32_t on_task_release(ChipTaskSlotState &slot_state) {
#endif
        TaskPayload *payload = slot_state.payload;
        int32_t released = 0;
        // Only DEP_RETAIN edges still hold a fanout pin at completion: an
        // ordering-only edge released its submit->wire pin at wiring, so releasing
        // it again here would over-count fanout_refcount against fanout_count and
        // break the rc == fc consume invariant.
        for_each_fanin_slot_state(*payload, [&](ChipTaskSlotState *producer_slot_state, DepFlags flags) {
            if (!dep_has_retain(flags)) {
                return;
            }
            released++;
#if SIMPLER_SCHED_PROFILING
            release_producer(*producer_slot_state, fanin_atomics);
#else
            release_producer(*producer_slot_state);
#endif
        });
#if SIMPLER_SCHED_PROFILING
        g_sched_fanin_atomic_count[thread_idx] += fanin_atomics;
        SCHED_CYCLE_LAP(g_sched_fanin_cycle[thread_idx]);
#endif

        // Self consumed check
#if SIMPLER_SCHED_PROFILING
        uint64_t self_atomics = 0;
        check_and_handle_consumed(slot_state, self_atomics);
        g_sched_self_atomic_count[thread_idx] += self_atomics;
        SCHED_CYCLE_LAP(g_sched_self_consumed_cycle[thread_idx]);
        g_sched_complete_count[thread_idx]++;
#else
        check_and_handle_consumed(slot_state);
#endif
        return released;
    }

    // === Cold-path API (defined in scheduler.cpp) ===

    // Phase 1: declare every sub-region (ready_queue slots, dummy queue slots,
    // per-ring dep_pool entries) on the supplied arena.
    // Capacities are baked into the returned layout; init_data_from_layout uses
    // the same values.
    static SchedulerLayout reserve_layout(DeviceArena &arena, int32_t dep_pool_capacity = CHIP_DEP_LIST_POOL_SIZE);
    static SchedulerLayout reserve_layout(DeviceArena &arena, const int32_t dep_pool_capacities[CHIP_MAX_RING_DEPTH]);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // `sm_dev_base` is the device address of the SM (only stored, never
    // dereferenced here). Safe to call on a host arena that holds the
    // prebuilt image buffer. (The orchestrator counterpart takes
    // task_window_size for ring task_descriptors address arithmetic; the
    // scheduler only needs the SM header / ring header base addresses,
    // both window-size-independent.)
    bool init_data_from_layout(const SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);
    void reset_for_reuse(const SchedulerLayout &layout, void *sm_dev_base);

    // Phase 3b: write the arena-internal pointer fields
    // (ready_queues[].slots, dummy_ready_queue.slots, and dep_pool base plus
    // reclaim-publication pointers for each ring). Called on both host and
    // device sides.
    void wire_arena_pointers(const SchedulerLayout &layout, DeviceArena &arena);

    // Forget per-region pointers; arena owns the backing memory.
    void destroy();
    void print_stats();
    void print_queues();
};

// Scheduler cold-path API is declared as SchedulerState member functions.
// See init()/destroy()/print_stats()/print_queues() below the struct definition.

// Short-circuit NotDeferred completions seen during drain so they don't grow
// entries[]. Mirrors the a2a3 impl; see that mirror for the rationale.
inline bool
AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, ChipTaskSlotState &slot_state) {
#if SIMPLER_SCHED_PROFILING
    sink.sched->on_task_complete(slot_state, sink.thread_idx);
#else
    sink.sched->on_task_complete(slot_state);
#endif
    if (*sink.deferred_release_count >= sink.deferred_release_capacity) {
        while (*sink.deferred_release_count > 0) {
#if SIMPLER_SCHED_PROFILING
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
    AICoreCompletionMailbox *aicore_mailbox, SchedulerState *sched, ChipTaskSlotState **deferred_release_slot_states,
    int32_t &deferred_release_count, int32_t deferred_release_capacity
#if SIMPLER_SCHED_PROFILING
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
#if SIMPLER_SCHED_PROFILING
    sink.thread_idx = thread_idx;
#endif

    int32_t drain_err = SIMPLER_ERROR_NONE;
    drain_aicore_completion_mailbox_locked(aicore_mailbox, sink, drain_err);
    if (drain_err != SIMPLER_ERROR_NONE) {
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
#if SIMPLER_SCHED_PROFILING
            sched->on_task_complete(*entry.slot_state, thread_idx);
#else
            sched->on_task_complete(*entry.slot_state);
#endif
            if (deferred_release_count >= deferred_release_capacity) {
                while (deferred_release_count > 0) {
#if SIMPLER_SCHED_PROFILING
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

#if SIMPLER_SCHED_PROFILING
struct SchedProfilingData {
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
SchedProfilingData scheduler_get_profiling(int thread_idx);
#endif
