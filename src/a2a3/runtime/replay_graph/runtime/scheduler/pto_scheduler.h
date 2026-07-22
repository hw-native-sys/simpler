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
 * 2. Tracking task state (PENDING -> COMPLETED)
 * 3. Managing fanin/fanout refcounts for dependency resolution
 * 4. Closing and traversing concurrent fanout stacks
 * 5. Two-stage mixed-task completion (subtask done bits → mixed-task complete)
 *
 * The Scheduler runs on Device AI_CPU and processes:
 * - Task state transitions based on fanin_refcount
 * - Graph-level task/heap arena completion accounting
 * - Ready and early-dispatch queue progress
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
#include "pto_graph_execution.h"
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

    void reset_for_reuse() {}

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
    size_t off_graph_ready_queue_slots;
    size_t off_graph_prepare_queue_slots;
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
    PTO2ReplayGraphPipelineState *graph_pipeline;

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

    // Graph tasks are control work. The scheduler expands them into their
    // GraphExecution nodes without assigning an AICore resource.
    PTO2ReadyQueue graph_ready_queue;

    // A Graph can be materialized as soon as its immutable Definition and
    // invocation Args are published, before its external data dependencies
    // are ready. Activation remains gated by graph_ready_queue.
    PTO2ReadyQueue graph_prepare_queue;

    alignas(64) AsyncWaitList async_wait_list;

    // Statistics (cold path, isolated from hot-path fields)
#if SIMPLER_SCHED_PROFILING
    alignas(64) std::atomic<int64_t> tasks_completed;
#endif
    // =========================================================================
    // Inline hot-path methods
    // =========================================================================

    // Route a ready slot to the right global queue. Dummy tasks (empty active_mask)
    // live in dummy_ready_queue; a ready sync_start cohort goes to the per-shape
    // ready_sync_queues[] (drained as Tier-0); everything else to ready_queues[].
    void push_ready_routed(PTO2TaskSlotState *slot_state) {
        if (slot_state->task->kind == PTO2TaskKind::GRAPH) {
            graph_ready_queue.push(slot_state);
            return;
        }
        PTO2ResourceShape shape = slot_state->active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY) {
            dummy_ready_queue.push(slot_state);
        } else if (slot_state->active_mask.requires_sync_start()) {
            ready_sync_queues[static_cast<int32_t>(shape)].push(slot_state);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(slot_state);
        }
    }

    bool try_claim_ready_once(PTO2TaskSlotState &slot_state) {
        uint8_t state = slot_state.ready_state.load(std::memory_order_acquire);
        for (;;) {
            if ((state & PTO2_READY_CLAIMED) != 0) return false;
            uint8_t desired = state | PTO2_READY_CLAIMED;
            if (slot_state.ready_state.compare_exchange_weak(
                    state, desired, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                return true;
            }
        }
    }

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
        if (count <= 0 || !slot_state.allow_early_resolve) return;
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

    // Event-driven candidate detection (the dual of fanin_refcount/ready). Called after a
    // FLAGGED producer `p` publishes blocks (normal dispatch, early-dispatch release, or the
    // sync_start drain), but no-ops until every logical block is launch-visible. Only then
    // does it walk p's fanout and bump each consumer's
    // dispatch_fanin. A consumer whose dispatch_fanin reaches fanin_count (= every
    // producer is flagged-and-fully-dispatched, or was already complete when the consumer was
    // wired) is an early-dispatch candidate: CAS NONE->STAGING (exactly-once) and push to
    // early_dispatch_queues[shape] (or early_sync_start_queue for a require_sync_start cohort)
    // for the drain to pre-stage. The full-publication gate is load-bearing: a consumer that
    // pre-occupies cores off a producer whose later blocks are not yet launch-visible would gate the
    // very cores those blocks need, starving the producer so it never completes and the
    // rendezvous never rings — a resource deadlock. published_block_count advances after
    // every placement path publishes its claimed range. Once-guarded per producer.
    // Only codegen-flagged producers propagate: a task's successors early-dispatch off its
    // DIRECT producers' marks, never an inherited chain.
    void propagate_dispatch_fanin(PTO2TaskSlotState &p) {
        if (!p.allow_early_resolve) return;  // only codegen-flagged (direct) producers propagate
        if (p.payload->published_block_count.load(std::memory_order_seq_cst) < p.logical_block_num) return;
        if (p.payload->early_dispatch_state.load(std::memory_order_acquire) == PTO2_EARLY_DISPATCH_STAGING) return;
        uint8_t launch_state = p.payload->early_dispatch_launch_state.load(std::memory_order_acquire);
        if (launch_state == PTO2_EARLY_DISPATCH_LAUNCH_RINGING) return;
        if (p.active_mask.requires_sync_start()) {
            bool was_pre_staged = false;
            for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
                was_pre_staged |= p.payload->staged_core_mask[w].load(std::memory_order_acquire) != 0;
            }
            if (was_pre_staged && launch_state != PTO2_EARLY_DISPATCH_LAUNCH_COMPLETE) return;
        }
        if (p.payload->dispatch_propagated.exchange(1, std::memory_order_acq_rel) != 0)
            return;  // already propagated once
        PTO2DepListEntry *edge = p.fanout_head.load(std::memory_order_acquire);
        if (pto2_is_fanout_closed(edge)) return;
        for (; edge != nullptr; edge = edge->next) {
            PTO2TaskSlotState *c = edge->slot_state;
            // fanin_count includes the graph publish dependency and every
            // pending producer edge.
            int32_t nf = c->payload->dispatch_fanin.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (nf != c->fanin_count) continue;
            PTO2ResourceShape shape = c->active_mask.to_shape();
            if (shape != PTO2ResourceShape::AIC && shape != PTO2ResourceShape::AIV && shape != PTO2ResourceShape::MIX)
                continue;
            // sync_start blocks launch as an atomic cohort. They ride the early-dispatch path
            // too, but through a dedicated shape-agnostic queue (early_sync_start_queue),
            // drained as the highest tier in try_early_dispatch via the gated drain +
            // running-slot rendezvous (enter_drain_mode pre-stages every block gated — MIX
            // with per-core split placement — and the rendezvous rings them together once
            // every gated core occupies a running slot and the producer released). They must
            // NOT enter early_dispatch_queues[shape]: that path's partial range-claim would
            // strand gated cohort blocks nobody rings, so the rendezvous never reaches
            // block_num. The rendezvous counts CORES (a MIX block spans a cluster), so one
            // shape-agnostic queue suffices.
            uint8_t expect = PTO2_EARLY_DISPATCH_NONE;  // exactly-once: only the CAS winner enqueues
            if (!c->payload->early_dispatch_state.compare_exchange_strong(
                    expect, PTO2_EARLY_DISPATCH_STAGING, std::memory_order_seq_cst, std::memory_order_seq_cst
                ))
                continue;
            uint64_t task_id = static_cast<uint64_t>(c->task->task_id.raw);
            if (c->active_mask.requires_sync_start()) early_sync_start_queue.push_tagged(c, task_id);
            else early_dispatch_queues[static_cast<int32_t>(shape)].push_tagged(c, task_id);
        }
    }

    // Collects consumers released via the early-dispatch-doorbell path during a
    // single on_task_complete fanout walk, so their dispatch_fanin
    // propagation runs AFTER the walk — never between two siblings' doorbells.
    struct EarlyDispatchReleaseSink {
        static constexpr int CAP = 32;
        PTO2TaskSlotState *items[CAP];
        int n = 0;
        inline bool push(PTO2TaskSlotState *s) {
            if (n >= CAP) return false;
            items[n++] = s;
            return true;
        }
    };

    inline bool try_early_dispatch_release(PTO2TaskSlotState &slot_state, EarlyDispatchReleaseSink *sink = nullptr) {
        // Never staged => CAS NONE->DISPATCHED wins => dispatch normally.
        uint8_t expect = PTO2_EARLY_DISPATCH_NONE;
        if (slot_state.payload->early_dispatch_state.compare_exchange_strong(
                expect, PTO2_EARLY_DISPATCH_DISPATCHED, std::memory_order_seq_cst, std::memory_order_seq_cst
            )) {
            return false;
        }
        // route_ready_once admits one caller, but keep the helper defensive: a
        // duplicate that observes or loses an in-progress launch must never
        // route the same partial task a second time.
        if (expect != PTO2_EARLY_DISPATCH_STAGING) return true;
        bool sync_start = slot_state.active_mask.requires_sync_start();
        if (!sync_start && !try_claim_early_dispatch_launch(*slot_state.payload)) return true;
        expect = PTO2_EARLY_DISPATCH_STAGING;
        slot_state.payload->early_dispatch_state.compare_exchange_strong(
            expect, PTO2_EARLY_DISPATCH_DISPATCHED, std::memory_order_seq_cst, std::memory_order_seq_cst
        );
        // Producer released: ring the gated cores. A non-sync_start consumer launches
        // each block the instant its doorbell fires. A sync_start consumer instead holds
        // for the rendezvous — every gated core must occupy a running slot first — so the
        // flip to DISPATCHED above is only the producer-released half; maybe_rendezvous_ring
        // rings now iff running_slot_count already reached popcount(staged_core_mask) (all
        // gated cores took idle running slots), else the last pending->running promotion rings.
        bool launched = true;
        if (sync_start) {
            launched = maybe_rendezvous_ring(slot_state);
        } else {
            // Destructively claim every published bit. A stager racing this pass
            // can claim only bits that land after the exchange, so each gated core
            // has exactly one doorbell writer.
            for (int w = 0; w < PTO2_EARLY_DISPATCH_CORE_MASK_WORDS; w++) {
                uint64_t owned = claim_all_staged_doorbell_bits(slot_state.payload->staged_core_mask[w]);
                ring_staged_doorbell_bits(w, owned);
            }
            wmb();
            slot_state.payload->early_dispatch_launch_state.store(
                PTO2_EARLY_DISPATCH_LAUNCH_COMPLETE, std::memory_order_seq_cst
            );
        }
        // This pre-staged consumer was just released by its doorbell — it starts
        // running NOW, so propagate dispatch_fanin to ITS consumers (only if it is
        // itself codegen-flagged; the gate inside no-ops otherwise). Defer it via the
        // sink so it runs after the whole fanout walk: doing it inline here would
        // delay the doorbells of later consumers in the same producer's fanout.
        // Fallback to inline if no sink / sink full.
        if (launched) {
            if (sink == nullptr || !sink->push(&slot_state)) {
                propagate_dispatch_fanin(slot_state);
            }
        }
        // No explicit removal from the cross-thread queue: a still-queued entry for
        // this consumer is now DISPATCHED and is dropped when a peer pops it.
        // Fully pre-staged => skip the ready queue. Partially staged SPMD consumer =>
        // fall through so the caller pushes C; dispatch resumes from next_block_idx.
        return slot_state.next_block_idx.load(std::memory_order_seq_cst) >= slot_state.logical_block_num;
    }

    bool route_ready_once(PTO2TaskSlotState &slot_state, EarlyDispatchReleaseSink *sink = nullptr) {
        if (!try_claim_ready_once(slot_state)) return false;

        if (slot_state.task->kind == PTO2TaskKind::GRAPH) {
            graph_ready_queue.push(&slot_state);
            return true;
        }

        // Early-dispatch: pre-staged tasks are released by doorbell
        // here, skipping the ready-queue round-trip entirely.
        bool early_handled = try_early_dispatch_release(slot_state, sink);
        if (slot_state.active_mask.requires_sync_start()) {
            bool drain_owned = publish_ready_to_early_sync_drain(*slot_state.payload);
            if (early_handled || drain_owned) return true;
        } else if (early_handled) {
            return true;
        }

        PTO2ResourceShape shape = slot_state.active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY) {
            dummy_ready_queue.push(&slot_state);
        } else if (slot_state.active_mask.requires_sync_start()) {
            ready_sync_queues[static_cast<int32_t>(shape)].push(&slot_state);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(&slot_state);
        }
        return true;
    }

#if SIMPLER_ORCH_PROFILING || SIMPLER_SCHED_PROFILING
    bool route_ready_once(
        PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &push_wait,
        EarlyDispatchReleaseSink *sink = nullptr
    ) {
        uint8_t state = slot_state.ready_state.load(std::memory_order_acquire);
        atomic_count += 1;  // ready_state load
        for (;;) {
            if ((state & PTO2_READY_CLAIMED) != 0) return false;
            uint8_t desired = state | PTO2_READY_CLAIMED;
            if (slot_state.ready_state.compare_exchange_weak(
                    state, desired, std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                atomic_count += 1;  // ready_state CAS
                break;
            }
            atomic_count += 1;  // failed ready_state CAS
        }

        if (slot_state.task->kind == PTO2TaskKind::GRAPH) {
            graph_ready_queue.push(&slot_state, atomic_count, push_wait);
            return true;
        }

        // Early-dispatch: pre-staged tasks are released by doorbell
        // here, skipping the ready-queue round-trip entirely.
        bool early_handled = try_early_dispatch_release(slot_state, sink);
        if (slot_state.active_mask.requires_sync_start()) {
            bool drain_owned = publish_ready_to_early_sync_drain(*slot_state.payload);
            if (early_handled || drain_owned) return true;
        } else if (early_handled) {
            return true;
        }

        PTO2ResourceShape shape = slot_state.active_mask.to_shape();
        if (shape == PTO2ResourceShape::DUMMY) {
            dummy_ready_queue.push(&slot_state, atomic_count, push_wait);
        } else if (slot_state.active_mask.requires_sync_start()) {
            ready_sync_queues[static_cast<int32_t>(shape)].push(&slot_state, atomic_count, push_wait);
        } else {
            ready_queues[static_cast<int32_t>(shape)].push(&slot_state, atomic_count, push_wait);
        }
        return true;
    }
#endif

    bool release_fanin_and_check_ready(PTO2TaskSlotState &slot_state, EarlyDispatchReleaseSink *sink = nullptr) {
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
        PTO2TaskSlotState &slot_state, uint64_t &atomic_count, uint64_t &push_wait,
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

    void record_graph_task_complete(PTO2TaskSlotState &slot_state) {
        if (graph_pipeline == nullptr || slot_state.task == nullptr) return;

        const int32_t task_id = static_cast<int32_t>(slot_state.task->task_id.local());
        for (int32_t i = 0; i < PTO2_REPLAY_GRAPH_BUFFER_COUNT; i++) {
            PTO2ReplayGraphBufferControl &buffer = graph_pipeline->buffers[i];
            const int32_t begin = buffer.task_begin;
            const int32_t end = begin + buffer.task_count;
            if (task_id < begin || task_id >= end) continue;

            const int32_t completed = buffer.completed_count.fetch_add(1, std::memory_order_acq_rel) + 1;
            if (completed >= buffer.task_count) {
                buffer.exec_done.store(1, std::memory_order_release);
                if (!pto2_try_release_graph_buffer(buffer)) {
                    buffer.state.store(PTO2ReplayGraphBufferState::DONE, std::memory_order_release);
                }
            }
            return;
        }
    }

    int32_t activate_prepared_graph_task(PTO2GraphExecution &execution) {
        if (!pto2_graph_execution_activate(execution)) return 0;

        int32_t roots = 0;
        for (int32_t i = 0; i < execution.topology.root_count; ++i) {
            uint16_t node_index = execution.topology.root_indices[i];
            if (node_index >= execution.node_count) continue;
            route_ready_once(execution.nodes[node_index].slot);
            roots++;
        }
        return roots;
    }

    PTO2GraphMaterializeResult prepare_graph_task(
        PTO2TaskSlotState &outer_slot, int32_t max_nodes = PTO2_GRAPH_MATERIALIZE_SLICE_NODES,
        int32_t *nodes_materialized = nullptr
    ) {
        if (outer_slot.task == nullptr || outer_slot.task->kind != PTO2TaskKind::GRAPH) {
            return PTO2GraphMaterializeResult::INVALID;
        }
        PTO2GraphExecution *execution = pto2_graph_execution_from_task(*outer_slot.task);
        if (execution == nullptr) return PTO2GraphMaterializeResult::INVALID;

        PTO2GraphMaterializeResult result =
            pto2_graph_execution_materialize_slice(outer_slot, max_nodes, nodes_materialized);
        PTO2GraphExecutionState state = execution->state.load(std::memory_order_acquire);
        if (result == PTO2GraphMaterializeResult::PREPARED && state == PTO2GraphExecutionState::PREPARED &&
            execution->activation_requested.load(std::memory_order_acquire) != 0) {
            activate_prepared_graph_task(*execution);
        }
        return result;
    }

    int32_t activate_graph_task(PTO2TaskSlotState &outer_slot) {
        if (outer_slot.task == nullptr || outer_slot.task->kind != PTO2TaskKind::GRAPH) return 0;
        PTO2GraphExecution *execution = pto2_graph_execution_from_task(*outer_slot.task);
        if (execution == nullptr) return 0;

        execution->activation_requested.store(1, std::memory_order_release);
        PTO2GraphExecutionState state = execution->state.load(std::memory_order_acquire);
        if (state != PTO2GraphExecutionState::PREPARED) return 0;
        return activate_prepared_graph_task(*execution);
    }

    /**
     * Two-stage completion: second stage.
     * Called exactly once when all subtasks of a task are done (i.e.,
     * on_subtask_complete returned true). Walks the consumer (fanout) list,
     * increments each consumer's completed-producer count, pushes newly-ready ones, and rings
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
#if SIMPLER_SCHED_PROFILING
        CompletionStats stats = {0, 0, 0, true};
#else
        uint32_t consumer_walk_count = 0;
#endif
#if SIMPLER_SCHED_PROFILING
        extern uint64_t g_sched_fanout_cycle[];
        extern uint64_t g_sched_fanout_atomic_count[], g_sched_push_wait_cycle[];
        PTO2_SCHED_CYCLE_START();
#endif

        slot_state.mark_completed();

        // Fanout: notify consumers. A pre-staged consumer that becomes ready has
        // its doorbell rung INLINE (db = nullptr) the moment its node is reached,
        // not batched to after the whole walk — so a flagged consumer near the
        // front of the list starts immediately and overlaps the remaining
        // release_fanin work for the other consumers, instead of waiting for the
        // full O(fanout-degree) walk (~5us for a 50-consumer producer).
        //
        // Safe on silicon: the producer's slot is already COMPLETED here — every
        // SPMD block has FIN'd AND dcci-flushed its output to HBM before
        // on_task_complete runs — so a released consumer never reads stale
        // producer output. (Batching used to align the released wave, but pushed
        // every doorbell to the end of the walk, defeating the whole point of
        // early-dispatch: minimal producer-end -> consumer-start.)
#if SIMPLER_SCHED_PROFILING
        uint64_t fanout_atomics = 0, push_wait = 0;
#endif
        // Doorbells for released pre-staged consumers fire INLINE in the walk
        // below; their dispatch_fanin propagation is collected here and replayed
        // after the walk, so no consumer's doorbell waits on a sibling's propagate.
        EarlyDispatchReleaseSink rel_sink;
        auto release_consumer = [&](PTO2TaskSlotState &consumer_slot) {
#if SIMPLER_SCHED_PROFILING
            stats.fanout_edges++;
            if (release_fanin_and_check_ready(consumer_slot, fanout_atomics, push_wait, &rel_sink)) {
                stats.tasks_enqueued++;
            }
#else
            consumer_walk_count++;
            release_fanin_and_check_ready(consumer_slot, &rel_sink);
#endif
        };
        bool graph_node = slot_state.task != nullptr && slot_state.task->kind == PTO2TaskKind::GRAPH_NODE;
        if (graph_node) {
            PTO2GraphExecution *execution = pto2_graph_execution_from_task(*slot_state.task);
            int32_t node_index = slot_state.task->graph_node_index;
            if (execution != nullptr && execution->nodes != nullptr && node_index >= 0 &&
                node_index < execution->node_count) {
                uint32_t begin = execution->topology.fanout_offsets[node_index];
                uint32_t end = execution->topology.fanout_offsets[node_index + 1];
                for (uint32_t edge_index = begin; edge_index < end; ++edge_index) {
                    uint16_t consumer_index = execution->topology.fanout_indices[edge_index];
                    release_consumer(execution->nodes[consumer_index].slot);
                }
            }
        } else {
            PTO2DepListEntry *current =
                slot_state.fanout_head.exchange(pto2_fanout_closed_sentinel(), std::memory_order_acq_rel);
            if (pto2_is_fanout_closed(current)) current = nullptr;
            while (current != nullptr) {
                release_consumer(*current->slot_state);
                current = current->next;
            }
        }
        for (int i = 0; i < rel_sink.n; i++) {
            propagate_dispatch_fanin(*rel_sink.items[i]);
        }
        if (!graph_node) {
            record_graph_task_complete(slot_state);
        }

#if SIMPLER_SCHED_PROFILING
        g_sched_fanout_atomic_count[thread_idx] += fanout_atomics;
        g_sched_push_wait_cycle[thread_idx] += push_wait;
        PTO2_SCHED_CYCLE_LAP(g_sched_fanout_cycle[thread_idx]);
        return stats;
#else
        return consumer_walk_count;
#endif
    }

    struct TaskCompletionOutcome {
        uint32_t fanout_edges{0};
        int32_t stream_tasks_completed{0};
    };

    TaskCompletionOutcome complete_task(
        PTO2TaskSlotState &slot_state
#if SIMPLER_SCHED_PROFILING
        ,
        int thread_idx
#endif
    ) {
        TaskCompletionOutcome outcome;
#if SIMPLER_SCHED_PROFILING
        CompletionStats stats = on_task_complete(slot_state, thread_idx);
        outcome.fanout_edges = static_cast<uint32_t>(stats.fanout_edges);
#else
        outcome.fanout_edges = on_task_complete(slot_state);
#endif
        if (slot_state.task == nullptr || slot_state.task->kind != PTO2TaskKind::GRAPH_NODE) {
            outcome.stream_tasks_completed = 1;
            return outcome;
        }

        PTO2GraphExecution *execution = pto2_graph_execution_from_task(*slot_state.task);
        if (execution == nullptr || !pto2_graph_execution_complete_node(*execution)) return outcome;

        PTO2TaskSlotState *outer = execution->outer_slot;
        if (outer != nullptr) {
#if SIMPLER_SCHED_PROFILING
            CompletionStats outer_stats = on_task_complete(*outer, thread_idx);
            outcome.fanout_edges += static_cast<uint32_t>(outer_stats.fanout_edges);
#else
            outcome.fanout_edges += on_task_complete(*outer);
#endif
            outcome.stream_tasks_completed = 1;
        }
        pto2_graph_execution_mark_completed(*execution);
        return outcome;
    }

    void retire_graph_node(PTO2TaskSlotState &slot_state) {
        if (slot_state.task == nullptr || slot_state.task->kind != PTO2TaskKind::GRAPH_NODE) return;
        PTO2GraphExecution *execution = pto2_graph_execution_from_task(*slot_state.task);
        if (execution != nullptr) pto2_graph_execution_retire_node(*execution);
    }

    // === Cold-path API (defined in pto_scheduler.cpp) ===

    static PTO2SchedulerLayout reserve_layout(DeviceArena &arena);

    // Phase 3a: write everything *except* arena-internal pointer fields.
    // `sm_dev_base` is the device address of the SM (only stored, never
    // dereferenced here). Safe to call on a host arena that holds the
    // prebuilt image buffer. The scheduler only needs the SM header base.
    bool init_data_from_layout(const PTO2SchedulerLayout &layout, DeviceArena &arena, void *sm_dev_base);
    void reset_for_reuse(const PTO2SchedulerLayout &layout, void *sm_dev_base);

    // Phase 3b: write the arena-internal pointer fields
    // (ready and early-dispatch queue slots). Called on host and device sides.
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
inline bool
AsyncWaitList::try_inline_complete_locked(AsyncWaitList::DrainCompletionSink &sink, PTO2TaskSlotState &slot_state) {
    // Return value (CompletionStats / consumer-walk count) discarded:
    // async-wait drain path has no Resolve swimlane bar attached.
#if SIMPLER_SCHED_PROFILING
    auto outcome = sink.sched->complete_task(slot_state, sink.thread_idx);
#else
    auto outcome = sink.sched->complete_task(slot_state);
#endif
    sink.inline_completed += outcome.stream_tasks_completed;
    sink.sched->retire_graph_node(slot_state);
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
            auto outcome = sched->complete_task(*entry.slot_state, thread_idx);
#else
            auto outcome = sched->complete_task(*entry.slot_state);
#endif
            result.completed += outcome.stream_tasks_completed;
            sched->retire_graph_node(*entry.slot_state);

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
    uint64_t fanout_cycle;

    // Wait times
    uint64_t push_wait_cycle;  // CAS contention in push()
    uint64_t pop_wait_cycle;   // CAS contention in pop()

    // Atomic counts per sub-phase
    uint64_t fanout_atomic_count;
    uint64_t pop_atomic_count;

    int64_t complete_count;
};

/**
 * Get and reset scheduler profiling data for a specific thread.
 * Returns accumulated profiling data and resets counters.
 */
PTO2SchedProfilingData scheduler_get_profiling(int thread_idx);
#endif
