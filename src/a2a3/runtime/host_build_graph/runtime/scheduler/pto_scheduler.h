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
 * 2. Polling-completion dependency resolution: a task is ready when every
 *    producer named in its inline fanin has set its completion_flags byte;
 *    a producer publishes completion + drains its wake list on finish
 * 3. Publishing the host-visible task_state mirror (PENDING -> COMPLETED) and
 *    advancing the per-ring completed_watermark (consumer-retirement signal)
 * 4. Two-stage mixed-task completion (subtask done bits -> mixed-task complete)
 *
 * The Scheduler runs on Device AI_CPU. host_build_graph is scheduler-only (the
 * orchestrator runs to completion on the host); there is no on-device slot
 * reclaim (whole-graph-resident), so last_task_alive is not advanced here.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <atomic>

#include "common/core_type.h"
#include "common/memory_barrier.h"
#include "utils/device_arena.h"
#include "aicpu/platform_regs.h"  // get_reg_ptr / RegId for the early-dispatch doorbell
#include "pto_async_wait.h"
#include "pto_ring_buffer.h"
#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"

#include "aicpu/device_time.h"  // get_sys_cnt_aicpu (weak; used by early-dispatch doorbell timing too)
#if SIMPLER_SCHED_PROFILING
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
    uint64_t task_id_snapshot;
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

    bool push(PTO2TaskSlotState *slot_state) { return push_tagged(slot_state, 0); }

    bool push_tagged(PTO2TaskSlotState *slot_state, uint64_t task_id_snapshot) {
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
        slot->task_id_snapshot = task_id_snapshot;
        slot->sequence.store(static_cast<int64_t>(pos + 1), std::memory_order_release);
        return true;
    }

    // Batch push: reserve count slots with a single CAS after confirming
    // every target slot is available under the usual Vyukov sequence check.
    void push_batch(PTO2TaskSlotState **items, int count) { push_batch_tagged(items, nullptr, count); }

    void push_batch_tagged(PTO2TaskSlotState **items, const uint64_t *task_id_snapshots, int count) {
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
            slot->task_id_snapshot = task_id_snapshots == nullptr ? 0 : task_id_snapshots[i];
            slot->sequence.store(static_cast<int64_t>(pos + i + 1), std::memory_order_release);
        }
    }

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
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

    PTO2TaskSlotState *pop() { return pop_tagged(nullptr); }

    PTO2TaskSlotState *pop_tagged(uint64_t *task_id_snapshot) {
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
        if (task_id_snapshot != nullptr) *task_id_snapshot = slot->task_id_snapshot;
        slot->sequence.store(static_cast<int64_t>(pos + mask + 1), std::memory_order_release);
        return result;
    }

#if SIMPLER_SCHED_PROFILING
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
    int pop_batch(PTO2TaskSlotState **out, int max_count) { return pop_batch_tagged(out, nullptr, max_count); }

    int pop_batch_tagged(PTO2TaskSlotState **out, uint64_t *task_id_snapshots, int max_count) {
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
            if (task_id_snapshots != nullptr) task_id_snapshots[i] = slot->task_id_snapshot;
            slot->sequence.store(static_cast<int64_t>(pos + i + mask + 1), std::memory_order_release);
        }
        return count;
    }

#if SIMPLER_SCHED_PROFILING
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
    size_t off_ready_sync_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_early_dispatch_queue_slots[PTO2_NUM_RESOURCE_SHAPES];
    size_t off_early_sync_start_queue_slots;
    uint64_t ready_queue_capacity;
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

        // Polling: no per-ring dep_pool. Readiness is derived from the SM ring's
        // completion_flags; there is no arena-side wiring pool to reserve or wire.
        // The `ring` field stores the device address of the SM ring header —
        // computed via offset arithmetic, no SM dereference.
        bool init_data_from_layout(void *sm_dev_base);
        void destroy();
    } ring_sched_state;

    // Ready queues remain global (scheduling is ring-agnostic)
    PTO2ReadyQueue ready_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Ready sync_start queues, one per shape. A ready sync_start cohort parks here
    // instead of ready_queues[] so the dispatch loop can drain it as a strict Tier-0
    // (sync_start > MIX > C/V) before any regular ready task takes a core, while
    // reusing the same per-shape dispatch_shape machinery (fits-local inline vs
    // stop-the-world drain, per-core MIX placement, head-start spacing).
    PTO2ReadyQueue ready_sync_queues[PTO2_NUM_RESOURCE_SHAPES];

    // Dependency-only tasks (active_mask is empty, shape == DUMMY). Drained by
    // the dispatch loop and completed inline -- never goes to AICore.
    PTO2ReadyQueue dummy_ready_queue;

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
    void push_ready_routed(PTO2TaskSlotState *slot_state) {
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY ||
            (slot_state->task_attrs.has_predicate() && !slot_state->payload->predicate.pass())) {
            dummy_ready_queue.push(slot_state);
        } else if (slot_state->task_attrs.requires_sync_start()) {
            ready_sync_queues[static_cast<int32_t>(shape)].push(slot_state);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(slot_state);
        }
    }

    // ---- Polling completion primitives (single-ring hbg) ----------------------
    // Readiness: a task is ready iff every producer named in its inline fanin has
    // set its completion_flags byte. Single-ring: all producers are ring 0, so
    // there is no per-edge ring indirection.

    bool fanin_satisfied(const PTO2TaskSlotState *s) const {
        const PTO2TaskPayload &p = *s->payload;
        const PTO2SharedMemoryRingHeader &ring = *ring_sched_state.ring;
        for (int32_t i = 0; i < p.fanin_count; i++) {
            if (ring.completion_flags[p.fanin_local_ids[i] & ring.task_window_mask].load(std::memory_order_acquire) ==
                0)
                return false;
        }
        return true;
    }

    // First-unmet classification. Returns -1 (all fanins met -> route to ready)
    // or the index of the first unmet fanin (register on that producer's wake
    // list). The decision is terminal: tasks are never re-polled; a producer's
    // completion re-scans its waiters via on_mixed_task_complete's wake drain.
    int classify_fanin_state(const PTO2TaskSlotState *s) const {
        const PTO2TaskPayload &p = *s->payload;
        const PTO2SharedMemoryRingHeader &ring = *ring_sched_state.ring;
        for (int32_t i = 0; i < p.fanin_count; i++) {
            if (ring.completion_flags[p.fanin_local_ids[i] & ring.task_window_mask].load(std::memory_order_acquire) ==
                0)
                return i;
        }
        return -1;
    }

    // Register `consumer` on `producer`'s wake list. If the producer already
    // completed (head == SENTINEL), re-classify against ALL fanins: route to
    // ready only when every fanin is met, else re-target the next unmet producer
    // and retry. Monotonic completion_flags guarantee termination.
    void register_wake(PTO2TaskSlotState *producer, PTO2TaskSlotState *consumer) {
        PTO2SharedMemoryRingHeader &ring = *ring_sched_state.ring;
        while (true) {
            PTO2TaskSlotState *expected = producer->wake_list_head.load(std::memory_order_relaxed);
            while (expected != WAKE_LIST_SENTINEL) {
                consumer->next_in_wake_list = expected;
                if (producer->wake_list_head.compare_exchange_weak(
                        expected, consumer, std::memory_order_acq_rel, std::memory_order_relaxed
                    )) {
                    return;
                }
            }
            int32_t state = classify_fanin_state(consumer);
            if (state < 0) {
                push_ready_routed(consumer);
                return;
            }
            producer = &ring.get_slot_state_by_task_id(consumer->payload->fanin_local_ids[state]);
        }
    }

    // Producer completion under polling: publish the host-visible task_state
    // mirror + the device-visible completion_flags byte, drain the wake list
    // (route/re-register each waiter), then CAS-advance the monotonic
    // completed_watermark (load-bearing: the host wait_for_consumers gates on
    // watermark >= producer.last_consumer_local_id). Whole-graph-resident hbg
    // has no device slot reclaim, so no advance_ring_pointers here.
    void on_mixed_task_complete(PTO2TaskSlotState &slot_state) {
        const int32_t my_id = static_cast<int32_t>(slot_state.task->task_id.local());
        PTO2SharedMemoryRingHeader &ring = *ring_sched_state.ring;

        slot_state.mark_completed();  // host-visible mirror (task_state = COMPLETED)
        ring.completion_flags[my_id & ring.task_window_mask].store(1, std::memory_order_release);

        PTO2TaskSlotState *waiter = slot_state.wake_list_head.exchange(WAKE_LIST_SENTINEL, std::memory_order_acq_rel);
        while (waiter != nullptr && waiter != WAKE_LIST_SENTINEL) {
            PTO2TaskSlotState *next = waiter->next_in_wake_list;
            if (waiter->payload->fanin_count == 1) {
                push_ready_routed(waiter);  // single-fanin waiter was waiting only on us
                waiter = next;
                continue;
            }
            int state = classify_fanin_state(waiter);
            if (state < 0) {
                push_ready_routed(waiter);
            } else {
                register_wake(&ring.get_slot_state_by_task_id(waiter->payload->fanin_local_ids[state]), waiter);
            }
            waiter = next;
        }

        // completed_watermark = highest id such that every task in [0, watermark]
        // has its completion_flags byte set. The host wait_for_consumers gates on
        // watermark >= producer.last_consumer_local_id, so the walk must extend to
        // the full contiguous completed prefix — NOT cap at my_id. Capping at my_id
        // makes the final value order-dependent: a low-id task completing after a
        // higher one would leave the watermark stuck below the true prefix, hanging
        // any wait_for_consumers whose last_consumer sits in the gap.
        const int32_t submitted = ring.fc.current_task_index.load(std::memory_order_acquire);
        int32_t w = ring.completed_watermark.load(std::memory_order_acquire);
        while (w + 1 < submitted) {
            int32_t next = w + 1;
            if (ring.completion_flags[next & ring.task_window_mask].load(std::memory_order_acquire) == 0) break;
            if (ring.completed_watermark.compare_exchange_weak(
                    w, next, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                w = next;
            }
        }
    }

    // Polling: there is no ready-claim CAS (a producer routes each waiter exactly
    // once via the wake-list drain) and no per-producer consumer/scope refcount.
    // Consumer retirement is observed by the host through completed_watermark >=
    // producer.last_consumer_local_id, not by bumping a producer refcount.

    // Early-dispatch release. If the now-ready task was pre-staged
    // (gated on a core), ring its DATA_MAIN_BASE high-32 doorbell RIGHT HERE in
    // the completion path — the moment its last producer's FIN satisfies fanin —
    // instead of routing it through the ready queue and waiting for the dispatch
    // pass to pop it. Returns true if the task is fully handled (caller must NOT
    // push to the ready queue). Returns false when the caller must route C
    // normally: either it was never pre-staged, OR it is a SPMD consumer only
    // PARTIALLY pre-staged — the gated blocks are released by the doorbells rung
    // here, and the remaining (next_block_idx .. logical_block_num) blocks
    // dispatch normally off the ready queue. Lock-free claim shared with Hook 1
    // (the stager): CAS NONE->DISPATCHED wins => not pre-staged; otherwise flip
    // STAGING->DISPATCHED and destructively claim the published doorbell bits.

    // Per-core early-dispatch doorbell table. Hook 1 records each gated core's
    // (reg_addr, dispatch token) here at stage time; the completion-path release
    // reads it back for the cores set in the consumer's staged_core_mask. One
    // global table indexed by core_id (not per-task): gated cores in flight are
    // bounded by the chip's core count (no two-level pre-dispatch), so this is the
    // natural capacity and removes the old per-task 3-doorbell cap.
    struct EarlyDispatchDoorbell {
        uint64_t addr{0};
        uint32_t token{0};
    };
    EarlyDispatchDoorbell early_dispatch_doorbell_table[PTO2_EARLY_DISPATCH_CORE_MASK_WORDS * 64]{};

    // Cross-thread early-dispatch work queues, one PTO2ReadyQueue MPMC instance per
    // resource shape (AIC/AIV/MIX) — arena-backed, reserved/wired in pto_runtime2_init
    // alongside the per-shape ready queues, and indexed the same way. A candidate is
    // pushed to the queue for its own shape (active_mask.to_shape()) so the drain can
    // pop per shape and size the pop to that shape's free cores, exactly as normal
    // dispatch pops ready_queues[shape].
    //
    // A consumer's SPMD blocks span cores owned by several AICPU threads, but only a
    // thread RUNNING the consumer's producer discovers it (via the producer's
    // fanout). When that producer is thread-local (e.g. a 16-block AIV op filling one
    // thread's cores), the other threads never see the consumer and its blocks on
    // their cores can't pre-stage. The first claimer pushes the partially-staged
    // consumer here; every idle thread's early_dispatch pass pops one, stages a range
    // onto ITS OWN cores (range-claim via next_block_idx), and re-pushes if blocks
    // remain — exactly mirroring how a partially-dispatched SPMD task is re-pushed to
    // the ready queue (scheduler_dispatch: pop -> claim -> re-push). A stale/released
    // entry fails the STAGING check on pop and is dropped; a push that overflows is
    // logged and the consumer's blocks fall back to normal dispatch.
    PTO2ReadyQueue early_dispatch_queues[PTO2_NUM_RESOURCE_SHAPES];

    // sync_start early-dispatch candidates park here instead of early_dispatch_queues[]:
    // they need an atomic all-or-nothing stage via the drain barrier, not
    // early_dispatch_shape's per-thread partial range-claim. Shape-agnostic (the
    // rendezvous counts cores, not blocks), so a single queue serves all shapes; drained
    // as the highest occupancy tier at the top of try_early_dispatch.
    //
    // Deliberately single, vs the normal source's per-shape ready_sync_queues[]: a READY
    // sync cohort (producer done) can dispatch inline when it fits, so it reuses the
    // per-shape dispatch_shape; an EARLY sync cohort always carries a non-zero
    // src_payload gate and therefore always drains. The shape-agnostic rendezvous
    // makes one queue sufficient. Same drain, two sources.
    PTO2ReadyQueue early_sync_start_queue;

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
        return force_gate || early_dispatch_state == PTO2_EARLY_DISPATCH_STAGING;
    }

    static inline bool
    ring_claimed_local_doorbell(uint64_t claimed_word, int core_id, uint64_t reg_addr, uint32_t token) {
        if ((claimed_word & (1ULL << (core_id & 63))) == 0) return false;
        ring_one_doorbell(reg_addr, token);
        return true;
    }

    static inline bool try_claim_early_dispatch_launch(PTO2TaskPayload &payload) {
        uint8_t expected = PTO2_EARLY_DISPATCH_LAUNCH_NONE;
        return payload.early_dispatch_launch_state.compare_exchange_strong(
            expected, PTO2_EARLY_DISPATCH_LAUNCH_RINGING, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
    }

    inline void record_published_blocks(PTO2TaskSlotState &slot_state, int32_t count) {
        if (count <= 0 || !slot_state.task_attrs.allow_early_resolve()) return;
        slot_state.payload->published_block_count.fetch_add(static_cast<int16_t>(count), std::memory_order_seq_cst);
    }

    // Ring one sync_start cohort from its stable staged_core_mask. The caller owns
    // the NONE->RINGING launch latch and invokes this exactly once after drain
    // staging completes, while the corresponding per-core table entries are live.
    inline void ring_all_staged_doorbells(PTO2TaskSlotState &slot_state) {
        for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
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

    static inline bool try_claim_early_sync_drain(PTO2TaskPayload &payload) {
        uint8_t expected = PTO2_EARLY_SYNC_DRAIN_NONE;
        return payload.early_sync_drain_state.compare_exchange_strong(
            expected, PTO2_EARLY_SYNC_DRAIN_OWNER, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
    }

    static inline bool owns_early_sync_drain(const PTO2TaskPayload &payload) {
        return (payload.early_sync_drain_state.load(std::memory_order_acquire) & PTO2_EARLY_SYNC_DRAIN_OWNER) != 0;
    }

    static inline void mark_early_sync_drain_armed(PTO2TaskPayload &payload) {
        payload.early_sync_drain_state.fetch_or(PTO2_EARLY_SYNC_DRAIN_ARMED, std::memory_order_seq_cst);
    }

    static inline bool publish_ready_to_early_sync_drain(PTO2TaskPayload &payload) {
        uint8_t previous =
            payload.early_sync_drain_state.fetch_or(PTO2_EARLY_SYNC_DRAIN_READY, std::memory_order_seq_cst);
        return (previous & PTO2_EARLY_SYNC_DRAIN_OWNER) != 0;
    }

    inline void cancel_early_sync_drain(PTO2TaskSlotState &slot_state) {
        uint8_t previous =
            slot_state.payload->early_sync_drain_state.exchange(PTO2_EARLY_SYNC_DRAIN_NONE, std::memory_order_seq_cst);
        if ((previous & PTO2_EARLY_SYNC_DRAIN_OWNER) == 0) return;
        if ((previous & PTO2_EARLY_SYNC_DRAIN_READY) != 0) {
            push_ready_routed(&slot_state);
            return;
        }
        if (slot_state.payload->early_dispatch_state.load(std::memory_order_seq_cst) == PTO2_EARLY_DISPATCH_STAGING) {
            early_sync_start_queue.push_tagged(&slot_state, static_cast<uint64_t>(slot_state.task->task_id.raw));
        }
    }

    static inline void finish_early_sync_drain(PTO2TaskPayload &payload) {
        uint8_t state = payload.early_sync_drain_state.load(std::memory_order_seq_cst);
        while ((state & PTO2_EARLY_SYNC_DRAIN_OWNER) != 0 && (state & PTO2_EARLY_SYNC_DRAIN_COMPLETE) == 0) {
            uint8_t desired = state | PTO2_EARLY_SYNC_DRAIN_COMPLETE;
            if (payload.early_sync_drain_state.compare_exchange_weak(
                    state, desired, std::memory_order_seq_cst, std::memory_order_seq_cst
                )) {
                return;
            }
        }
    }
    // sync_start rendezvous: a sync_start consumer's gated cores launch as an atomic
    // cohort, so their doorbells are held until BOTH halves hold — every gated core
    // occupies a running slot (running_slot_count == popcount(staged_core_mask)) AND the
    // producer released (early_dispatch_state == DISPATCHED). Counting CORES (not blocks) makes
    // this shape-agnostic: an AIC/AIV block is one core, a MIX block is a cluster whose
    // cores promote pending->running independently. Called from both halves (the producer
    // release and each pending->running promotion); whichever observes the second half
    // wins the launch latch and rings exactly once. Returns true only to that winner,
    // which may then expose the cohort to its fanout.
    inline bool maybe_rendezvous_ring(PTO2TaskSlotState &slot_state) {
        int32_t staged_cores = 0;
        for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++)
            staged_cores +=
                __builtin_popcountll(slot_state.payload->staged_core_mask[w].load(std::memory_order_seq_cst));
        if (staged_cores == 0) return false;
        if (slot_state.payload->running_slot_count.load(std::memory_order_seq_cst) != staged_cores) return false;
        if (slot_state.payload->early_dispatch_state.load(std::memory_order_seq_cst) != PTO2_EARLY_DISPATCH_DISPATCHED)
            return false;
        if (!try_claim_early_dispatch_launch(*slot_state.payload)) return false;
        ring_all_staged_doorbells(slot_state);
        wmb();
        slot_state.payload->early_dispatch_launch_state.store(
            PTO2_EARLY_DISPATCH_LAUNCH_COMPLETE, std::memory_order_release
        );
        return true;
    }

    inline bool retry_sync_start_rendezvous_after_drain(PTO2TaskSlotState &slot_state) {
        if (!maybe_rendezvous_ring(slot_state)) return false;
        propagate_dispatch_fanin(slot_state);
        return true;
    }

    // Milestone 1: early-dispatch (predicated / allow_early_resolve) is stubbed.
    // This producer-push propagation walked the wiring fanout list bumping each
    // consumer's dispatch_fanin to pre-stage early-dispatch candidates — all of
    // which (fanout_head, dispatch_fanin, fanin_actual_count, dispatch_propagated)
    // are gone under polling. Nothing pre-stages into early_dispatch_queues /
    // early_sync_start_queue, so tasks reach cores only through the normal ready
    // path (wake drain -> push_ready_routed). Milestone 2 replaces this with the
    // consumer-pull publish_flags design. sync_start cohorts still launch via
    // ready_sync_queues (unaffected).
    void propagate_dispatch_fanin(PTO2TaskSlotState & /*p*/) {}

    int get_ready_tasks_batch(PTO2ReadyQueue *queues, PTO2ResourceShape shape, PTO2TaskSlotState **out, int max_count) {
        return queues[static_cast<int32_t>(shape)].pop_batch(out, max_count);
    }

#if SIMPLER_SCHED_PROFILING
    int get_ready_tasks_batch(
        PTO2ReadyQueue *queues, PTO2ResourceShape shape, PTO2TaskSlotState **out, int max_count, uint64_t &atomic_count,
        uint64_t &wait_cycle
    ) {
        return queues[static_cast<int32_t>(shape)].pop_batch(out, max_count, atomic_count, wait_cycle);
    }
#endif

    // Polling: scope-end takes no per-producer action. Under the wiring model
    // this bumped each task's scope refcount (PTO2_FANOUT_SCOPE_BIT); reclaim now
    // gates on completed_watermark >= last_consumer_local_id, which needs no
    // scope reference. Kept as a no-op so the orchestrator call site is unchanged.
    void on_scope_end(PTO2TaskSlotState ** /*task_slot_states*/, int32_t /*count*/) {}

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
     * Called exactly once when all subtasks of a task are done (i.e.,
     * on_subtask_complete returned true). Walks the consumer (fanout) list,
     * decrements each consumer's fanin, pushes newly-ready ones, and rings
     * doorbells for early-dispatch hits.
     *
     * Non-PROFILING returns the consumer-walk count (= edges traversed). The
     * Resolve swimlane bar reads it to label the bar with how many successors
     * actually got resolved. PROFILING returns the richer CompletionStats
     * whose `fanout_edges` carries the same number.
     */
#if SIMPLER_SCHED_PROFILING
    CompletionStats
#else
    uint32_t
#endif
    on_task_complete(
        PTO2TaskSlotState &slot_state
#if SIMPLER_SCHED_PROFILING
        ,
        int thread_idx
#endif
    ) {
        // Polling completion: publish the host-visible task_state mirror + the
        // device-visible completion_flags byte, drain the wake list (route or
        // re-register each waiter), and advance the watermark. Replaces the
        // fanout-list walk + fanin_refcount decrements of the wiring model.
        on_mixed_task_complete(slot_state);
#if SIMPLER_SCHED_PROFILING
        (void)thread_idx;
        // Resolved-successor accounting is not tracked on the polling path (the
        // producer no longer enumerates its consumers); report 0 for the DFX bar.
        return CompletionStats{0, 0, 0, true};
#else
        return 0;
#endif
    }

    // on_task_release is gone under polling. It existed to bump each producer's
    // fanout_refcount so the host wait_for_consumers could observe consumer
    // retirement; that signal is now the per-ring completed_watermark advanced by
    // on_mixed_task_complete. There is likewise no self CONSUMED flip (host-orch
    // never reclaimed slots on device).

    // === Cold-path API (defined in pto_scheduler.cpp) ===

    // Phase 1: declare every sub-region (ready_queue slots, dummy queue slots,
    // per-ring dep_pool entries) on the supplied arena.
    // Capacities are baked into the returned layout; init_data_from_layout uses
    // the same values.
    static PTO2SchedulerLayout reserve_layout(DeviceArena &arena);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // `sm_dev_base` is the device address of the SM (only stored, never
    // dereferenced here). Safe to call on a host arena that holds the
    // prebuilt image buffer. (The orchestrator counterpart takes
    // task_window_size for ring task_descriptors address arithmetic; the
    // scheduler only needs the SM header / ring header base addresses,
    // both window-size-independent.)
    bool init_data_from_layout(const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);

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

// try_inline_complete_locked: short-circuit NotDeferred completions seen during
// drain so they don't grow entries[]. Defined here (not in pto_async_wait.h)
// because PTO2SchedulerState's on_task_complete signature is only known
// after its full definition above.
//
// Polling: on_task_complete publishes completion + drains the wake list inline,
// so the async-drain path no longer buffers producer releases.
inline bool
AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, PTO2TaskSlotState &slot_state) {
    // Return value (CompletionStats / consumer-walk count) discarded:
    // async-wait drain path has no Resolve swimlane bar attached.
#if SIMPLER_SCHED_PROFILING
    (void)sink.sched->on_task_complete(slot_state, sink.thread_idx);
#else
    (void)sink.sched->on_task_complete(slot_state);
#endif
    sink.inline_completed++;
    return true;
}

template <bool Profiling>
inline AsyncPollResult AsyncWaitList::poll_and_complete(
    AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched
#if SIMPLER_SCHED_PROFILING
    ,
    int thread_idx
#endif
) {
    AsyncPollResult result;
    if (!try_lock()) return result;

    AsyncWaitList::DrainCompletionSink sink{};
    sink.sched = sched;
#if SIMPLER_SCHED_PROFILING
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
        uintptr_t last_invalidated_counter_line = static_cast<uintptr_t>(-1);
        for (int32_t c = 0; c < entry.condition_count; c++) {
            CompletionCondition &cond = entry.conditions[c];
            if (cond.satisfied) continue;
            if (cond.completion_type == COMPLETION_TYPE_COUNTER && cond.counter_addr != nullptr) {
                uintptr_t counter_line = mailbox_cache_line(cond.counter_addr);
                if (counter_line != last_invalidated_counter_line) {
                    cache_invalidate_range(reinterpret_cast<const void *>(counter_line), sizeof(uint32_t));
                    last_invalidated_counter_line = counter_line;
                }
            }
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
            // Return value (CompletionStats / consumer-walk count) discarded:
            // deferred-completion drain has no Resolve swimlane bar attached.
#if SIMPLER_SCHED_PROFILING
            (void)sched->on_task_complete(*entry.slot_state, thread_idx);
#else
            (void)sched->on_task_complete(*entry.slot_state);
#endif
            // Polling: completion is fully published inline; no deferred release.
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
struct PTO2SchedProfilingData {
    // Sub-phase cycle breakdown within on_task_complete
    uint64_t lock_cycle;           // lock_fanout + state store + unlock
    uint64_t fanout_cycle;         // fanout traversal
    uint64_t fanin_cycle;          // fanin traversal
    uint64_t self_consumed_cycle;  // self check_and_handle_consumed

    // Wait times
    uint64_t lock_wait_cycle;  // Legacy (wiring): fanout_lock spin-wait; polling has no such lock
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
