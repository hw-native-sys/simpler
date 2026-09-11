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
 * host_build_graph scheduler interface
 *
 * The Scheduler is responsible for:
 * 1. Maintaining per-resource-shape ready queues
 * 2. Polling-completion dependency resolution: a GLOBAL task is ready when every
 *    producer named in its inline fanin has a COMPLETED task_states byte, an
 *    IN_GRAPH one when every producer in its Graph's fanin wire does — the same
 *    array shape, held by the GraphExecution instead of the task header; a
 *    producer publishes completion + drains its wake list on finish
 * 3. Publishing completion (task_states byte PENDING -> COMPLETED)
 * 4. Two-stage mixed-task completion (subtask done bits -> mixed-task complete)
 *
 * The Scheduler runs on Device AI_CPU. host_build_graph is scheduler-only (the
 * orchestrator runs to completion on the host) and whole-graph-resident, so no
 * task slot or heap byte is reclaimed on device; the scheduler owns completion
 * state only.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <atomic>

#include "common/memory_barrier.h"
#include "utils/device_arena.h"
#include "aicpu/platform_regs.h"  // get_reg_ptr / RegId for the early-dispatch doorbell
#include "async_wait.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/task_id.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/runtime_types.h"
#include "host_build_graph/shared_memory.h"
// Defines the SIMPLER_*_PROFILING levels the conditionals below test. An #if on an
// undefined macro evaluates to 0, so a profiling block reached through a transitive
// include would switch itself off silently if that path ever went away.
#include "profiling_config.h"

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
#include "aicpu/device_time.h"  // get_sys_cnt_aicpu, used only by the timed blocks below
#endif
#if SIMPLER_SCHED_PROFILING
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
struct alignas(64) ChipReadyQueue {
    ChipReadyQueueSlot *slots;
    uint64_t capacity;
    uint64_t mask;        // capacity - 1
    char _pad0[64 - 24];  // Pad to own cache line

    std::atomic<uint64_t> enqueue_pos;
    char _pad1[64 - sizeof(std::atomic<uint64_t>)];  // Own cache line

    std::atomic<uint64_t> dequeue_pos;
    // Occupancy high-water, for teardown reporting only. Atomic because pushes
    // are concurrent; relaxed throughout, so it is ordered against nothing.
    std::atomic<uint64_t> max_occupancy;
    char _pad2[64 - 2 * sizeof(std::atomic<uint64_t>)];  // Own cache line

    // Bring the slots[] array to its empty state: slot i's sequence must equal i
    // for push_tagged's `diff == 0` claim test to accept the first pass, so an
    // empty queue is a 0..capacity-1 ramp rather than zeroed memory. Runs on the
    // device after wire_arena_pointers, before any push — the region is reserved
    // past the uploaded range, so nothing seeds it on the host.
    void seed_slots() {
        for (uint64_t i = 0; i < capacity; i++) {
            slots[i].sequence.store(static_cast<int64_t>(i), std::memory_order_relaxed);
            slots[i].slot_state = nullptr;
        }
    }

    uint64_t size() {
        uint64_t e = enqueue_pos.load(std::memory_order_relaxed);
        uint64_t d = dequeue_pos.load(std::memory_order_relaxed);
        return (e >= d) ? (e - d) : 0;
    }

    // Raise the high-water to the occupancy left by a push that published up to
    // `published_pos`. Off the fast path it is a load and a compare; the CAS runs
    // only on a new maximum, so a contended push never pays for it.
    void note_occupancy(uint64_t published_pos) {
        const uint64_t occ = published_pos - dequeue_pos.load(std::memory_order_relaxed);
        uint64_t observed = max_occupancy.load(std::memory_order_relaxed);
        while (occ > observed && !max_occupancy.compare_exchange_weak(
                                     observed, occ, std::memory_order_relaxed, std::memory_order_relaxed
                                 )) {}
    }

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
        note_occupancy(pos + 1);
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
        note_occupancy(pos + static_cast<uint64_t>(count));
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
//   init_from_layout: initialize the queue header (capacity, mask, positions)
//   seed_slots: establish the slot array's empty ramp, on the device only
//   destroy: forget the slots pointer (arena owns the buffer)
size_t ready_queue_reserve_layout(DeviceArena &arena, uint64_t capacity);
// Writes the header fields only: capacity, mask, positions, occupancy counter.
// The slot array is neither addressed nor written — it lives past the uploaded
// range and ChipReadyQueue::seed_slots() fills it on the device. Call
// `ready_queue_wire_arena_pointers` to set the `slots` pointer itself.
void ready_queue_init_data_from_layout(ChipReadyQueue *queue, uint64_t capacity);
// Stores queue->slots = arena.region_ptr(slots_off). Idempotent.
void ready_queue_wire_arena_pointers(ChipReadyQueue *queue, DeviceArena &arena, size_t slots_off);
void ready_queue_destroy(ChipReadyQueue *queue);

struct ReadyQueueCapacities {
    uint64_t ready[NUM_RESOURCE_SHAPES]{};
    uint64_t ready_sync[NUM_RESOURCE_SHAPES]{};
    uint64_t dummy{0};
    uint64_t graph_ready{0};
    uint64_t graph_prepare{0};
};

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
 * capacities used at layout time (init_from_layout reuses them).
 */
struct SchedulerLayout {
    size_t off_ready_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_ready_sync_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_dummy_ready_queue_slots;
    size_t off_graph_ready_queue_slots;
    size_t off_graph_prepare_queue_slots;
    size_t off_early_dispatch_queue_slots[NUM_RESOURCE_SHAPES];
    size_t off_early_sync_start_queue_slots;
    size_t off_ed_publish_drain_queue_slots;
    ReadyQueueCapacities capacities;
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

    // The task table's header, as a device address
    struct alignas(64) TaskHeaderView {
        SharedMemoryTaskHeader *tasks;

        // Polling: no dep_pool. Readiness is derived from the task table's
        // task_states; there is no arena-side wiring pool to reserve or wire.
        // The `tasks` field stores the device address of the SM task header —
        // computed via offset arithmetic, no SM dereference.
        bool init_data_from_layout(void *sm_dev_base);
        void destroy();
    } task_view;

    // Ready queues are global: scheduling does not partition by task id
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

    // An outer Graph is control work, never an AICore task. External dependency
    // readiness and bounded materialization progress independently and meet at
    // the submission's single atomic activation gate.
    ChipReadyQueue graph_ready_queue;
    ChipReadyQueue graph_prepare_queue;

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
    void latch_ready_queue_overflow(int32_t thread_idx = -1) {
        int32_t expected = SIMPLER_ERROR_NONE;
        const bool latched = sm_header->sched_error_code.compare_exchange_strong(
            expected, SIMPLER_ERROR_READY_QUEUE_OVERFLOW, std::memory_order_acq_rel, std::memory_order_acquire
        );
        if (latched && thread_idx >= 0) {
            sm_header->sched_error_thread.store(thread_idx, std::memory_order_release);
        }
        if (thread_idx >= 0 && thread_idx < 32) {
            sm_header->sched_error_bitmap.fetch_or(1U << static_cast<uint32_t>(thread_idx), std::memory_order_acq_rel);
        }
    }

    bool push_graph_prepare(ChipTaskSlotState *slot_state, uint64_t task_id, int32_t thread_idx) {
        if (graph_prepare_queue.push_tagged(slot_state, task_id)) return true;
        latch_ready_queue_overflow(thread_idx);
        return false;
    }

    void push_ready_routed(ChipTaskSlotState *slot_state) {
        // Early-dispatch release: a pre-staged candidate launches by doorbell
        // right here — the moment its readiness is decided — and skips the
        // queue round-trip. Only host-qualified candidates can hold a staging
        // claim, so every other task pays one hot-line flag test. A ready
        // sync_start candidate also publishes to its drain owner, which may
        // already hold it for an all-or-nothing stage.
        if ((slot_state->ed_flags & ED_FLAG_CANDIDATE) != 0) {
            const bool early_handled = try_early_dispatch_release(*slot_state);
            if (slot_state->task_attrs.requires_sync_start()) {
                const bool drain_owned = publish_ready_to_early_sync_drain(slot_state->to_payload());
                if (early_handled || drain_owned) return;
            } else if (early_handled) {
                return;
            }
        }
        bool pushed;
        if (slot_state->task_kind == TaskKind::GRAPH) {
            pushed = graph_ready_queue.push(slot_state);
        } else {
            ResourceShape shape = slot_state->active_mask.to_shape();
            if (shape == ResourceShape::DUMMY ||
                (slot_state->task_attrs.has_predicate() && !slot_state->to_payload().predicate.pass())) {
                pushed = dummy_ready_queue.push(slot_state);
            } else if (slot_state->task_attrs.requires_sync_start()) {
                pushed = ready_sync_queues[static_cast<int32_t>(shape)].push(slot_state);
            } else {
                pushed = ready_queues[static_cast<int32_t>(shape)].push(slot_state);
            }
        }
        // Every ready / sync / dummy / graph task routes to exactly one queue. A
        // false push means that queue's peak concurrent occupancy exceeded its
        // bind-time capacity — a capacity mis-sizing, not a normal condition.
        // Silently dropping the task would stall the run, so latch a named error
        // (surfaces as READY_QUEUE_OVERFLOW rather than an anonymous
        // forward-progress timeout). The graph_ready push is checked identically
        // so a graph task cannot be dropped either.
        if (!pushed) {
            latch_ready_queue_overflow();
        }
    }

    // ---- Polling completion primitives ---------------------------------------
    // Readiness: a task is ready iff every producer named in its inline fanin has
    // a COMPLETED task_states byte. A producer is named by its local id alone, so
    // there is no per-edge indirection.

    // Unmet-fanin classification. Returns -1 (all fanins met -> route to ready)
    // or the index of an unmet fanin (register on that producer's wake list).
    // Scan direction is load-bearing: the builder fills the fanin region in
    // submission order (and an early-dispatch candidate's row is sorted by
    // local id, making the order exact), so the last unmet entry is the
    // latest-submitted producer -- the one likeliest to complete last.
    // Targeting it minimises
    // wake-list transfers (a consumer re-registered onto a second producer once
    // its first one completes) and the CAS traffic those transfers put on the
    // lists. The decision is terminal: tasks are never re-polled; a producer's
    // completion re-scans its waiters via on_mixed_task_complete's wake drain.
    // Resumes at the wake-scan cursor: entries above it were completed at the
    // last classification and the task_states byte is monotone, so re-walking
    // them cannot change the verdict. An unmet verdict records itself as the
    // new cursor — the caller hangs the task exactly there, and this
    // classifier is the task's only owner at that moment.
    int classify_fanin_state(ChipTaskSlotState *s) const {
        const TaskPayload &p = s->to_payload();
        const SharedMemoryTaskHeader &tasks = *task_view.tasks;
        const int32_t *fanin = p.fanin_data();
        const int32_t start = s->wake_scan_cursor < p.fanin_count ? s->wake_scan_cursor : p.fanin_count - 1;
        for (int32_t i = start; i >= 0; i--) {
            if (!tasks.is_completed(fanin[i])) {
                s->wake_scan_cursor = static_cast<uint16_t>(i);
                return i;
            }
        }
        return -1;
    }

    // Register `consumer` on `producer`'s wake list. If the producer already
    // completed (head == SENTINEL), re-classify against ALL fanins: route to
    // ready only when every fanin is met, else re-target the next unmet producer
    // and retry. The monotone task_states bytes guarantee termination.
    void register_wake(ChipTaskSlotState *producer, ChipTaskSlotState *consumer) {
        SharedMemoryTaskHeader &tasks = *task_view.tasks;
        while (true) {
            ChipTaskSlotState *expected = producer->wake_list_head.load(std::memory_order_relaxed);
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
            producer = &tasks.get_slot_state_by_task_id(consumer->to_payload().fanin_data()[state]);
        }
    }

    // Producer completion under polling: publish the device-visible task_states
    // byte, then drain the wake list
    // (route/re-register each waiter). Whole-graph-resident hbg
    // has no device slot reclaim, so nothing advances a reclaim cursor here.
    void on_mixed_task_complete(ChipTaskSlotState &slot_state) {
        const int32_t task_id = slot_state.to_descriptor().task_id.local_id();
        SharedMemoryTaskHeader &tasks = *task_view.tasks;

        tasks.store_completed(task_id);
        // COMPLETED >= PUBLISHED, so a tracked producer that never published
        // (DUMMY, predicate-retired) still releases its publish-list waiters
        // here. Idempotent after a publish-time seal.
        if ((slot_state.ed_flags & ED_FLAG_TRACKED) != 0) seal_ed_publish_list(slot_state);

        ChipTaskSlotState *waiter = slot_state.wake_list_head.exchange(WAKE_LIST_SENTINEL, std::memory_order_acq_rel);
        while (waiter != nullptr && waiter != WAKE_LIST_SENTINEL) {
            ChipTaskSlotState *next = waiter->next_in_wake_list;
            if (waiter->to_payload().fanin_count == 1) {
                push_ready_routed(waiter);  // single-fanin waiter was waiting only on us
                waiter = next;
                continue;
            }
            int state = classify_fanin_state(waiter);
            if (state < 0) {
                push_ready_routed(waiter);
            } else {
                register_wake(&tasks.get_slot_state_by_task_id(waiter->to_payload().fanin_data()[state]), waiter);
            }
            waiter = next;
        }
    }

    // Polling: there is no ready-claim CAS (a producer routes each waiter exactly
    // once via the wake-list drain) and no per-producer consumer/scope refcount.

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
    EarlyDispatchDoorbell early_dispatch_doorbell_table[EARLY_DISPATCH_CORE_MASK_WORDS * 64]{};

    // Cross-thread early-dispatch work queues, one ChipReadyQueue MPMC instance per
    // resource shape (AIC/AIV/MIX) — arena-backed, reserved and wired by
    // SchedulerState::init_data_from_layout alongside the per-shape ready queues, and
    // indexed the same way. A candidate is pushed to the queue for its own shape
    // (active_mask.to_shape()) so the drain can pop per shape and size the pop to that
    // shape's free cores, exactly as normal dispatch pops ready_queues[shape].
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
    ChipReadyQueue early_dispatch_queues[NUM_RESOURCE_SHAPES];

    // sync_start early-dispatch candidates park here instead of early_dispatch_queues[]:
    // they need an atomic all-or-nothing stage, not early_dispatch_shape's
    // per-thread partial range-claim. Shape-agnostic (the rendezvous counts cores,
    // not blocks), so a single queue serves all shapes and one owner selects the
    // local stage or global-drain fallback.
    //
    // Deliberately single, vs the normal source's per-shape ready_sync_queues[]: a READY
    // sync cohort (producer done) can dispatch inline when it fits, so it reuses the
    // per-shape dispatch_shape. An EARLY sync cohort carries a non-zero src_payload
    // gate; its owner stages locally when one tracker fits and uses the global drain
    // otherwise. HBG producer propagation currently leaves this queue dormant.
    ChipReadyQueue early_sync_start_queue;

    // Pending-drain queue of the publish list. Each entry is the head of a
    // waiter chain a producer's publish event sealed and detached; idle
    // threads pop one per Phase 4b pass and rescan its waiters. A full queue
    // drops the chain — those candidates miss early dispatch and proceed on
    // the normal ready path; nothing blocks. The drop count surfaces at
    // teardown (no log on the dispatch path).
    ChipReadyQueue ed_publish_drain_queue;
    std::atomic<uint64_t> ed_publish_drain_drops{0};

#if SIMPLER_SCHED_PROFILING
    // Publish-list probe. Relaxed counters, read once at teardown.
    struct EdPublishListStats {
        std::atomic<uint64_t> registered{0};           // candidates hung at intake
        std::atomic<uint64_t> immediate_at_intake{0};  // all-published already at intake
        std::atomic<uint64_t> publish_seals{0};        // tracked producers fully published
        std::atomic<uint64_t> chains_detached{0};      // seals that handed off a non-empty chain
        std::atomic<uint64_t> waiters_rescanned{0};
        std::atomic<uint64_t> rehangs{0};           // bet missed: waiter moved to another producer
        std::atomic<uint64_t> candidates_ready{0};  // all-published verdicts (ED triggers)
    };
    EdPublishListStats ed_publish_stats;
#endif

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

    // All-published verdict -> ED queue. The exactly-once claim is the
    // NONE->STAGING CAS: a consumer that already became ready lost the state
    // to the release path's NONE->DISPATCHED and is dropped here. A failed
    // push returns the claim so readiness takes the ordinary queue path.
    inline void enqueue_early_dispatch_candidate(ChipTaskSlotState &consumer) {
        uint8_t expected = EARLY_DISPATCH_NONE;
        if (!consumer.to_payload().early_dispatch_state.compare_exchange_strong(
                expected, EARLY_DISPATCH_STAGING, std::memory_order_seq_cst, std::memory_order_seq_cst
            )) {
            return;
        }
        const uint64_t task_id = static_cast<uint64_t>(consumer.to_descriptor().task_id.raw);
        // A sync_start cohort parks in the shape-agnostic queue so one owner can
        // choose an all-or-nothing local stage or the global-drain fallback.
        const bool queued =
            consumer.task_attrs.requires_sync_start() ?
                early_sync_start_queue.push_tagged(&consumer, task_id) :
                early_dispatch_queues[static_cast<int32_t>(consumer.active_mask.to_shape())].push_tagged(
                    &consumer, task_id
                );
        if (!queued) {
            expected = EARLY_DISPATCH_STAGING;
            consumer.to_payload().early_dispatch_state.compare_exchange_strong(
                expected, EARLY_DISPATCH_NONE, std::memory_order_seq_cst, std::memory_order_seq_cst
            );
        }
    }

    // Early-dispatch release, run at the ready funnel the moment readiness is
    // decided. Returns true when the task is fully handled (every block
    // launches by doorbell) and must NOT be queued. Returns false to route
    // normally: never pre-staged, or a partially staged SPMD consumer whose
    // remaining blocks dispatch off the ready queue from next_block_idx.
    // Lock-free claim shared with the stagers: winning CAS NONE->DISPATCHED
    // means not pre-staged; otherwise flip STAGING->DISPATCHED and
    // destructively claim the published doorbell bits — a late stager rings
    // only the bits this pass did not take, so each gated core has exactly
    // one doorbell writer.
    inline bool try_early_dispatch_release(ChipTaskSlotState &slot_state) {
        TaskPayload &payload = slot_state.to_payload();
        uint8_t expect = EARLY_DISPATCH_NONE;
        if (payload.early_dispatch_state.compare_exchange_strong(
                expect, EARLY_DISPATCH_DISPATCHED, std::memory_order_seq_cst, std::memory_order_seq_cst
            )) {
            return false;
        }
        // Defensive: a duplicate that observes an in-progress launch must never
        // route the same partial task a second time.
        if (expect != EARLY_DISPATCH_STAGING) return true;
        const bool sync_start = slot_state.task_attrs.requires_sync_start();
        if (!sync_start && !try_claim_early_dispatch_launch(payload)) return true;
        expect = EARLY_DISPATCH_STAGING;
        payload.early_dispatch_state.compare_exchange_strong(
            expect, EARLY_DISPATCH_DISPATCHED, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
        if (sync_start) {
            // The flip to DISPATCHED is only the producer-released half of the
            // rendezvous; the ring fires once every gated core also occupies a
            // running slot, from whichever half completes second.
            try_launch_sync_start_cohort(slot_state);
        } else {
            for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
                uint64_t owned = claim_all_staged_doorbell_bits(payload.staged_core_mask[w]);
                ring_staged_doorbell_bits(w, owned);
            }
            wmb();
            payload.early_dispatch_launch_state.store(EARLY_DISPATCH_LAUNCH_COMPLETE, std::memory_order_seq_cst);
        }
        // A still-queued early-dispatch entry for this consumer is now
        // DISPATCHED and is dropped when a peer pops it.
        return slot_state.next_block_idx.load(std::memory_order_seq_cst) >= slot_state.logical_block_num;
    }

    // Publication accounting for the producers early dispatch watches. Only
    // ED_FLAG_TRACKED tasks pay: the host set that bit exactly on the
    // producers of at least one candidate. When the count reaches the task's
    // logical blocks, the state byte advances to PUBLISHED and the caller seals
    // the publish list — everything past that leaves the dispatch path.
    //
    // "Published" counts placement, not launch: a gated candidate's staged
    // blocks publish here too, so its consumers may pre-stage while it still
    // waits for its own doorbell. Such chains cannot deadlock on cores: a
    // consumer is detected only after this seal, hence after every producer
    // block already occupies its slot, so every waits-for edge between gated
    // tasks points from a later-staged task to an earlier-staged one — the
    // earliest unreleased gated task has only completed or released
    // producers, rings, and unwinds the chain in staging order. (The
    // reclaiming tensormap_and_ringbuffer runtime instead defers propagation
    // to launch visibility; whole-graph-resident hbg holds no reclaimable
    // resource a gated chain could pin, so placement is the sufficient gate.)
    // Called BEFORE the batch's MMIO token writes, with the count that batch is
    // about to publish. Returns true when this call reached the task's total,
    // i.e. this caller owns the seal and must call seal_ed_publish_list once its
    // tokens are flushed.
    //
    // The call site owes one precondition, and it is the one to re-check when
    // adding a publish site: every block counted here has its token written by
    // this same caller before it returns. A path that could reach the total and
    // then not emit those tokens would publish a task whose blocks never launch,
    // and its consumers would wait on a completion that cannot arrive.
    //
    // Accounting ahead of the tokens is what makes the state byte a plain
    // sequential store: it puts PUBLISHED before every token this task will
    // ever emit, so PUBLISHED ≺ token ≺ FIN ≺ COMPLETED holds no matter how
    // fast a core FINs or how sibling publishers interleave — the total-reaching
    // thread's store precedes its own tokens, and those tokens' FINs gate the
    // all-FIN completion. Cores are claimed at prepare time, before any of this,
    // so a consumer staged off the early PUBLISHED cannot occupy cores the
    // producer's in-flight blocks still need.
    inline bool account_published_blocks(ChipTaskSlotState &slot_state, int32_t count) {
        if (count <= 0 || (slot_state.ed_flags & ED_FLAG_TRACKED) == 0) return false;
        const int16_t total = static_cast<int16_t>(
            slot_state.to_payload().published_block_count.fetch_add(
                static_cast<int16_t>(count), std::memory_order_seq_cst
            ) +
            count
        );
        if (total != slot_state.logical_block_num) return false;
        task_view.tasks->store_published(slot_state.to_descriptor().task_id.local_id());
        return true;
    }

    // Seal event of a tracked producer, after its tokens are flushed. The
    // PUBLISHED store already happened in account_published_blocks, so a scanner
    // that meets the sentinel is guaranteed to read the byte as published. The
    // exchange makes the seal exactly-once (a second call gets the sentinel
    // back), so the completion path can call this too for tracked producers that
    // never publish (DUMMY / predicate-retired), whose COMPLETED byte already
    // compares as published. A full pending-drain queue drops the chain: those
    // candidates miss early dispatch, nothing blocks.
    inline void seal_ed_publish_list(ChipTaskSlotState &slot_state) {
        // The sentinel is monotone (set once, never cleared), so an
        // already-sealed list needs no exchange.
        if (slot_state.ed_publish_list_head.load(std::memory_order_acquire) == ED_PUBLISH_LIST_SENTINEL) return;
        ChipTaskSlotState *head =
            slot_state.ed_publish_list_head.exchange(ED_PUBLISH_LIST_SENTINEL, std::memory_order_acq_rel);
        if (head == ED_PUBLISH_LIST_SENTINEL) return;
#if SIMPLER_SCHED_PROFILING
        ed_publish_stats.publish_seals.fetch_add(1, std::memory_order_relaxed);
#endif
        if (head == nullptr) return;
#if SIMPLER_SCHED_PROFILING
        ed_publish_stats.chains_detached.fetch_add(1, std::memory_order_relaxed);
#endif
        if (!ed_publish_drain_queue.push(head)) {
            ed_publish_drain_drops.fetch_add(1, std::memory_order_relaxed);
        }
    }

    // Resume candidate `c`'s publish scan at its cursor. The row is sorted and
    // the state byte is monotone, so entries above the cursor stay published
    // forever and each entry is loaded once per life. Returns true when every
    // producer is published — the early-dispatch trigger; false when `c` was
    // re-hung on a still-unpublished producer's publish list. A hang CAS that
    // meets the sentinel treats the producer as published and advances.
    bool advance_ed_publish_scan(ChipTaskSlotState &c) {
        SharedMemoryTaskHeader &tasks = *task_view.tasks;
        const TaskPayload &p = c.to_payload();
        const int32_t *fanin = p.fanin_data();
        int32_t i = c.ed_publish_scan_cursor;
        while (i >= 0) {
            if (tasks.is_published(fanin[i])) {
                i--;
                continue;
            }
            ChipTaskSlotState &producer = tasks.get_slot_state_by_task_id(fanin[i]);
            ChipTaskSlotState *expected = producer.ed_publish_list_head.load(std::memory_order_acquire);
            bool hung = false;
            while (expected != ED_PUBLISH_LIST_SENTINEL) {
                c.next_in_ed_publish_list = expected;
                if (producer.ed_publish_list_head.compare_exchange_weak(
                        expected, &c, std::memory_order_acq_rel, std::memory_order_acquire
                    )) {
                    hung = true;
                    break;
                }
            }
            if (hung) {
                c.ed_publish_scan_cursor = static_cast<uint8_t>(i);
                return false;
            }
            // Sentinel: the producer published between the bit load and here.
            i--;
        }
        return true;
    }

    // Publish-list registration at intake. Only called for a candidate that
    // is not yet ready (its completion classification hung it on a wake list).
    // Returns true when every producer is already published at intake.
    bool register_on_ed_publish_list(ChipTaskSlotState &c) {
        c.ed_publish_scan_cursor = static_cast<uint8_t>(c.to_payload().fanin_count - 1);
#if SIMPLER_SCHED_PROFILING
        ed_publish_stats.registered.fetch_add(1, std::memory_order_relaxed);
#endif
        const bool all_published = advance_ed_publish_scan(c);
#if SIMPLER_SCHED_PROFILING
        if (all_published) ed_publish_stats.immediate_at_intake.fetch_add(1, std::memory_order_relaxed);
#endif
        return all_published;
    }

    // Ring one sync_start cohort from its stable staged_core_mask. The caller owns
    // the NONE->RINGING launch latch and invokes this exactly once after local or
    // global staging completes, while the corresponding per-core table entries are live.
    inline void ring_all_staged_doorbells(ChipTaskSlotState &slot_state) {
        for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
            uint64_t bits = slot_state.to_payload().staged_core_mask[w].load(std::memory_order_seq_cst);
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
            slot_state.to_payload().early_sync_drain_state.exchange(EARLY_SYNC_DRAIN_NONE, std::memory_order_seq_cst);
        if ((previous & EARLY_SYNC_DRAIN_OWNER) == 0) return;
        if ((previous & EARLY_SYNC_DRAIN_READY) != 0) {
            push_ready_routed(&slot_state);
            return;
        }
        if (slot_state.to_payload().early_dispatch_state.load(std::memory_order_seq_cst) == EARLY_DISPATCH_STAGING) {
            early_sync_start_queue.push_tagged(
                &slot_state, static_cast<uint64_t>(slot_state.to_descriptor().task_id.raw)
            );
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
    // sync_start rendezvous: a sync_start consumer's gated cores launch as an atomic
    // cohort, so their doorbells are held until BOTH halves hold — every gated core
    // occupies a running slot (running_slot_count == popcount(staged_core_mask)) AND the
    // producer released (early_dispatch_state == DISPATCHED). Counting CORES (not blocks) makes
    // this shape-agnostic: an AIC/AIV block is one core, a MIX block is a cluster whose
    // cores promote pending->running independently. Called from both halves (the producer
    // release and each pending->running promotion); whichever observes the second half
    // wins the launch latch and rings exactly once. Returns true only to that winner,
    // which may then expose the cohort to its fanout.
    inline bool try_launch_sync_start_cohort(ChipTaskSlotState &slot_state) {
        // running_slot_count is the publication seed: every staged_core_mask OR
        // happens-before its final store. Read the seed first, then the mask, so
        // observing the final count cannot be paired with a partially published
        // mask when producer release races staging.
        int32_t running_cores = slot_state.to_payload().running_slot_count.load(std::memory_order_seq_cst);
        int32_t staged_cores = 0;
        for (int w = 0; w < EARLY_DISPATCH_CORE_MASK_WORDS; w++)
            staged_cores +=
                __builtin_popcountll(slot_state.to_payload().staged_core_mask[w].load(std::memory_order_seq_cst));
        if (staged_cores == 0) return false;
        if (running_cores != staged_cores) return false;
        if (slot_state.to_payload().early_dispatch_state.load(std::memory_order_seq_cst) != EARLY_DISPATCH_DISPATCHED)
            return false;
        if (!try_claim_early_dispatch_launch(slot_state.to_payload())) return false;
        ring_all_staged_doorbells(slot_state);
        wmb();
        slot_state.to_payload().early_dispatch_launch_state.store(
            EARLY_DISPATCH_LAUNCH_COMPLETE, std::memory_order_release
        );
        return true;
    }

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

    // Orch owns dependency discovery and saves the immutable fanin wire.
    // Scheduler polling only chooses which already-wired producer a consumer
    // waits on at this instant; it never recomputes producer relationships.
    //
    // Returns -1 (every producer complete -> route to ready) or the consumer's
    // own CSR row index of an unmet producer, which graph_producer_at maps back
    // to that producer. Hanging on the producer that completes last minimises
    // wake-list transfers (a consumer re-registered onto a second producer once
    // its first one completes) and the CAS traffic those transfers put on the
    // lists, so the scan walks the row from its tail.
    //
    // What the tail names depends on the row. A row holds the consumer's
    // deduplicated operand order, which is a declaration order and carries no
    // relation to producer depth. Only an early-dispatch candidate's row is
    // sorted by producer index (graph_fill_definition), and a producer is
    // emitted before its consumers, so there the tail is the deepest producer
    // and the bet is exact. On an unsorted row the tail is the last-declared
    // operand and the scan is a heuristic, no worse-founded than picking the
    // head.
    //
    // Resumes at the wake-scan cursor: entries above it were complete at the
    // last classification and completion is monotonic, so re-walking them
    // cannot change the verdict.
    int32_t graph_first_unmet_producer(const GraphExecution &execution, ChipTaskSlotState &consumer) const {
        const int32_t task_index = consumer.in_graph_local_id;
        const int32_t begin = execution.fanin_offsets[task_index];
        const int32_t count = execution.fanin_offsets[task_index + 1] - begin;
        const int32_t start = consumer.wake_scan_cursor < count ? consumer.wake_scan_cursor : count - 1;
        for (int32_t row = start; row >= 0; --row) {
            if (!execution.is_completed(execution.fanin_indices[begin + row])) {
                consumer.wake_scan_cursor = static_cast<uint16_t>(row);
                return row;
            }
        }
        return -1;
    }

    // The producer a graph_first_unmet_producer row index names.
    ChipTaskSlotState &
    graph_producer_at(const GraphExecution &execution, const ChipTaskSlotState &consumer, int32_t row) const {
        const int32_t begin = execution.fanin_offsets[consumer.in_graph_local_id];
        return execution.task_at(execution.fanin_indices[begin + row]).slot;
    }

    void register_graph_wake(GraphExecution &execution, ChipTaskSlotState *producer, ChipTaskSlotState *consumer) {
        while (true) {
            ChipTaskSlotState *expected = producer->wake_list_head.load(std::memory_order_relaxed);
            while (expected != WAKE_LIST_SENTINEL) {
                consumer->next_in_wake_list = expected;
                if (producer->wake_list_head.compare_exchange_weak(
                        expected, consumer, std::memory_order_acq_rel, std::memory_order_relaxed
                    )) {
                    return;
                }
            }

            // The producer completed between fanin classification and the CAS.
            // Retry against acquire-loaded task states until cache coherence
            // exposes the completed producer, then route or retarget the waiter.
            const int32_t unmet_producer = graph_first_unmet_producer(execution, *consumer);
            if (unmet_producer < 0) {
                push_ready_routed(consumer);
                return;
            }
            producer = &graph_producer_at(execution, *consumer, unmet_producer);
        }
    }

    uint32_t drain_graph_wake_list(GraphExecution &execution, ChipTaskSlotState &producer) {
        uint32_t consumers_resolved = 0;
        ChipTaskSlotState *waiter = producer.wake_list_head.exchange(WAKE_LIST_SENTINEL, std::memory_order_acq_rel);
        while (waiter != nullptr && waiter != WAKE_LIST_SENTINEL) {
            ChipTaskSlotState *next = waiter->next_in_wake_list;
            const int32_t local_id = waiter->in_graph_local_id;
            if (execution.fanin_offsets[local_id + 1] - execution.fanin_offsets[local_id] == 1) {
                push_ready_routed(waiter);  // single-producer waiter was waiting only on us
                consumers_resolved++;
                waiter = next;
                continue;
            }
            const int32_t unmet_producer = graph_first_unmet_producer(execution, *waiter);
            if (unmet_producer < 0) {
                push_ready_routed(waiter);
            } else {
                register_graph_wake(execution, &graph_producer_at(execution, *waiter, unmet_producer), waiter);
            }
            consumers_resolved++;
            waiter = next;
        }
        return consumers_resolved;
    }

    // Push every materialized-and-published root that has not been routed yet,
    // once the outer Graph task's external dependencies are ready.
    // route_cursor makes this idempotent, so it composes across the per-slice
    // calls during materialization and the final call at the activation meet;
    // each root reaches the ready queue exactly once. Non-roots are never pushed
    // here — they reach the ready queue through their producers' wake list.
    int32_t graph_route_ready_roots(GraphExecution &execution) {
        if (execution.outer_slot == nullptr || !graph_execution_external_ready(execution)) return 0;
        const int32_t published = execution.published_tasks.load(std::memory_order_acquire);
        int32_t routed = 0;
        while (true) {
            int32_t i = execution.route_cursor.load(std::memory_order_relaxed);
            if (i >= published) break;
            if (!execution.route_cursor.compare_exchange_weak(
                    i, i + 1, std::memory_order_acq_rel, std::memory_order_relaxed
                )) {
                continue;
            }
            if (execution.fanin_offsets[i] == execution.fanin_offsets[i + 1]) {
                push_ready_routed(&execution.task_at(i).slot);
                routed++;
            }
        }
        return routed;
    }

    // Register each newly materialized in-graph task [first, last) on its first unmet
    // producer (or route it immediately when every producer already completed),
    // publish the range for routing, and route any roots the external gate now
    // admits. Runs single-owner per graph via the prepare-queue slot, so the
    // range never overlaps another thread's. register_graph_wake and
    // graph_first_unmet_producer are safe against a producer completing
    // concurrently, which is what lets a task dispatch before the whole graph is
    // materialized.
    void graph_incremental_publish(GraphExecution &execution, int32_t first, int32_t last) {
        for (int32_t i = first; i < last; ++i) {
            if (execution.fanin_offsets[i] == execution.fanin_offsets[i + 1]) continue;  // root
            ChipTaskSlotState &task = execution.task_at(i).slot;
            const int32_t unmet = graph_first_unmet_producer(execution, task);
            if (unmet < 0) {
                push_ready_routed(&task);
            } else {
                register_graph_wake(execution, &graph_producer_at(execution, task, unmet), &task);
            }
        }
        execution.published_tasks.store(last, std::memory_order_release);
        graph_route_ready_roots(execution);
    }

    int32_t activate_prepared_graph(GraphExecution &execution) {
        if (!graph_execution_transition(execution, GraphExecutionState::PREPARED, GraphExecutionState::ACTIVE)) {
            return 0;
        }
        return graph_route_ready_roots(execution);
    }

    GraphMaterializeResult prepare_graph_task(
        ChipTaskSlotState &outer_slot, int32_t max_tasks = GRAPH_MATERIALIZE_SLICE_TASKS,
        int32_t *tasks_materialized = nullptr
    ) {
        GraphExecution *execution = graph_execution_from_outer_slot(outer_slot);
        if (execution == nullptr) return GraphMaterializeResult::INVALID;
        const int32_t before = execution->materialized_tasks;
        const GraphMaterializeResult result =
            graph_execution_materialize_slice(outer_slot, *execution, max_tasks, tasks_materialized);
        if (result == GraphMaterializeResult::PENDING || result == GraphMaterializeResult::PREPARED) {
            graph_incremental_publish(*execution, before, execution->materialized_tasks);
        }
        if (result == GraphMaterializeResult::PREPARED && graph_execution_external_ready(*execution)) {
            activate_prepared_graph(*execution);
        }
        return result;
    }

    int32_t activate_graph_task(ChipTaskSlotState &outer_slot) {
        GraphExecution *execution = graph_execution_from_outer_slot(outer_slot);
        if (execution == nullptr) return 0;
        graph_execution_signal_external_ready(*execution);
        return activate_prepared_graph(*execution);
    }

    struct TaskCompletionOutcome {
        uint32_t fanout_edges{0};
        int32_t stream_tasks_completed{0};
        int32_t error_code{SIMPLER_ERROR_NONE};
    };

    TaskCompletionOutcome complete_task(
        ChipTaskSlotState &slot_state
#if SIMPLER_SCHED_PROFILING
        ,
        int thread_idx
#endif
    ) {
        TaskCompletionOutcome outcome;
        // A task in a Graph body retires into its execution's counters; everything else
        // — including the outer GRAPH shell — is a task of the run and releases its
        // fanout. graph_context is null for the common case, so this short-circuits
        // before the kind is read.
        if (slot_state.graph_context == nullptr || slot_state.task_kind == TaskKind::GRAPH) {
#if SIMPLER_SCHED_PROFILING
            CompletionStats stats = on_task_complete(slot_state, thread_idx);
            outcome.fanout_edges = static_cast<uint32_t>(stats.fanout_edges);
#else
            outcome.fanout_edges = on_task_complete(slot_state);
#endif
            outcome.stream_tasks_completed = 1;
            return outcome;
        }

        // Membership is established by the branch above: graph_context names this task's
        // execution, and the shell case has already returned.
        GraphExecution *execution = static_cast<GraphExecution *>(slot_state.graph_context);
        if (execution->definition == nullptr || execution->tasks == nullptr) {
            outcome.error_code = SIMPLER_ERROR_INVALID_ARGS;
            return outcome;
        }
        // Incremental activation routes an in-graph task before the graph reaches
        // ACTIVE, so one can legitimately complete while the graph is still
        // MATERIALIZING or PREPARED. Only SUBMITTED (execution not yet bound) and
        // COMPLETED (execution already retired) are invalid states for such a completion.
        const GraphExecutionState graph_state = graph_execution_state(*execution);
        if (graph_state < GraphExecutionState::MATERIALIZING || graph_state > GraphExecutionState::ACTIVE) {
            outcome.error_code = SIMPLER_ERROR_INVALID_ARGS;
            return outcome;
        }
        const int32_t task_index = slot_state.in_graph_local_id;
        if (task_index < 0 || task_index >= execution->task_count) {
            outcome.error_code = SIMPLER_ERROR_INVALID_ARGS;
            return outcome;
        }

        // Publish completion before closing the wake list. A consumer that
        // loses registration to the sentinel acquires this state when it
        // rescans the Orch-built fanin wire, so no wakeup can be lost.
        execution->store_completed(task_index);
        outcome.fanout_edges = drain_graph_wake_list(*execution, slot_state);

        const bool graph_completed = graph_execution_complete_in_graph_task(*execution);
        graph_execution_retire_in_graph_task(*execution);
        if (!graph_completed) return outcome;

        // Internal tasks count as zero stream tasks. The final in-graph task publishes
        // the outer task exactly once, waking external consumers and
        // contributing the one task the host actually submitted.
        if (execution->outer_slot != nullptr) {
            on_mixed_task_complete(*execution->outer_slot);
            outcome.stream_tasks_completed = 1;
        }
        graph_execution_mark_completed(*execution);
        return outcome;
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
        ChipTaskSlotState &slot_state
#if SIMPLER_SCHED_PROFILING
        ,
        int thread_idx
#endif
    ) {
        // Polling completion: publish the device-visible task_states
        // byte and drain the wake list (route or
        // re-register each waiter). Replaces the
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
    // fanout_refcount so a host-side consumer wait could observe consumer
    // retirement; no such wait exists — orchestration rejects scalar access to a
    // tensor with a producer instead. There is likewise no self CONSUMED flip
    // (host-orch never reclaimed slots on device).

    // === Cold-path API (defined in scheduler.cpp) ===

    // Phase 1: declare every sub-region (ready_queue slots, dummy queue slots) on
    // the supplied arena.
    // Capacities are baked into the returned layout; init_data_from_layout uses
    // the same values.
    static SchedulerLayout reserve_layout(DeviceArena &arena);
    static SchedulerLayout reserve_layout(DeviceArena &arena, uint64_t ready_capacity);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // `sm_dev_base` is the device address of the SM (only stored, never
    // dereferenced here). Safe to call on a host arena that holds the
    // prebuilt image buffer. (The orchestrator counterpart takes task_capacity
    // for its task_descriptors address arithmetic; the scheduler only needs the
    // SM header and task header base addresses, both capacity-independent.)
    bool init_data_from_layout(const SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);

    // Phase 3b: write the arena-internal pointer fields
    // (ready_queues[].slots, dummy_ready_queue.slots). Called on both host and
    // device sides.
    void wire_arena_pointers(const SchedulerLayout &layout, DeviceArena &arena);

    // Phase 3c, device only: bring every queue's slots[] to its empty ramp. The
    // slot arrays sit past the uploaded range, so this is the only thing that
    // establishes the sequence values push compares against. Runs once per arena
    // after wire_arena_pointers and before any push; a drained queue needs no
    // repeat, since a free slot's sequence tracks the position it serves.
    void seed_queue_slots();

    // Forget per-region pointers; arena owns the backing memory.
    void destroy();
    void print_stats();
    void print_queues();
};

// Scheduler cold-path API is declared as SchedulerState member functions.
// See init()/destroy()/print_stats()/print_queues() below the struct definition.

// try_inline_complete_locked: short-circuit NotDeferred completions seen during
// drain so they don't grow entries[]. Defined here (not in async_wait.h)
// because SchedulerState's on_task_complete signature is only known
// after its full definition above.
//
// Polling: on_task_complete publishes completion + drains the wake list inline,
// so the async-drain path no longer buffers producer releases.
inline bool
AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, ChipTaskSlotState &slot_state) {
    // Return value (CompletionStats / consumer-walk count) discarded:
    // async-wait drain path has no Resolve swimlane bar attached.
#if SIMPLER_SCHED_PROFILING
    SchedulerState::TaskCompletionOutcome outcome = sink.sched->complete_task(slot_state, sink.thread_idx);
#else
    SchedulerState::TaskCompletionOutcome outcome = sink.sched->complete_task(slot_state);
#endif
    if (outcome.error_code != SIMPLER_ERROR_NONE) {
        sink.error_code = outcome.error_code;
        return false;
    }
    sink.inline_completed += outcome.stream_tasks_completed;
    sink.inline_resolved++;
    return true;
}

template <bool Profiling>
inline AsyncPollResult AsyncWaitList::poll_and_complete(
    AICoreCompletionMailbox *aicore_mailbox, SchedulerState *sched
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

    int32_t drain_err = SIMPLER_ERROR_NONE;
    drain_aicore_completion_mailbox_locked(aicore_mailbox, sink, drain_err);
    if (drain_err != SIMPLER_ERROR_NONE) {
        result.error_code = drain_err;
        unlock();
        return result;
    }
    result.completed += sink.inline_completed;
    result.resolved += sink.inline_resolved;

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
            SchedulerState::TaskCompletionOutcome outcome = sched->complete_task(*entry.slot_state, thread_idx);
#else
            SchedulerState::TaskCompletionOutcome outcome = sched->complete_task(*entry.slot_state);
#endif
            if (outcome.error_code != SIMPLER_ERROR_NONE) {
                result.error_code = outcome.error_code;
                result.failed_slot_state = entry.slot_state;
                unlock();
                return result;
            }
            // Polling: completion is fully published inline; no deferred release.
            result.completed += outcome.stream_tasks_completed;
            result.resolved++;

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
SchedProfilingData scheduler_get_profiling(int thread_idx);
#endif
