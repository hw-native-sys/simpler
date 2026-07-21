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
 * PTO Runtime2 - Ring Buffer Data Structures
 *
 * Implements ring buffer designs for zero-overhead memory management:
 *
 * 1. TaskAllocator - Unified task slot + output buffer allocation
 *    - Combines task ring (slot allocation) and heap ring (output buffer allocation)
 *    - Single spin-wait loop with unified back-pressure and deadlock detection
 *    - O(1) bump allocation for both task slots and heap buffers
 *
 * 2. FaninPool - Fanin spill entry allocation
 *    - Ring buffer for spilled fanin entries
 *    - O(1) append allocation
 *    - Implicit reclamation with task ring
 *
 * 3. DepListPool - Dependency list entry allocation
 *    - Ring buffer for linked list entries
 *    - O(1) prepend operation
 *    - Implicit reclamation with task ring
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#ifndef PTO_RING_BUFFER_H
#define PTO_RING_BUFFER_H

#include <algorithm>
#include <inttypes.h>
#include <type_traits>
#include <utility>

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "aicpu/device_time.h"       // get_sys_cnt_aicpu (deadlock wall-clock backstop)
#include "common/platform_config.h"  // PLATFORM_PROF_SYS_CNT_FREQ (deadlock wall-clock)
#include "common/unified_log.h"

#if SIMPLER_DFX
// Heap-ring wrap reporting — the allocator is the only place each individual
// wrap is observable, so it notifies the scope_stats collector here. Gated:
// pays nothing (no include, no call) when profiling is compiled out.
#include "aicpu/scope_stats_collector_aicpu.h"
#endif

// Block notification interval (in spin counts)
#define PTO2_BLOCK_NOTIFY_INTERVAL 10000
// Heap/task deadlock is detected structurally (head task COMPLETED + all
// consumers released + scope still open -> only scope_end can free it, which a
// blocked orchestrator can never reach). This wall-clock value is only a
// backstop for the residual case the structural test can't prove locally; it is
// an ABSOLUTE TIME (not a spin count), so it is stable across chips/contention.
#define PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES (PLATFORM_PROF_SYS_CNT_FREQ / 2)  // 500 ms

// =============================================================================
// Task Allocator (unified task slot + heap buffer allocation)
// =============================================================================

/**
 * Unified task slot + heap buffer allocator.
 *
 * Since task and heap are always allocated together and the orchestrator is
 * single-threaded, both pointers (task index, heap top) are tracked locally
 * and published to shared memory via plain store — no fetch_add or CAS needed.
 *
 * The alloc() method checks both resources BEFORE committing to either,
 * eliminating the need for rollback on partial failure.
 */
class PTO2TaskAllocator {
public:
    /**
     * Initialize the allocator with task ring and heap ring resources.
     *
     * All pointer arguments are device addresses (live in SM / GM heap); this
     * function only stores them, no dereferences, so it is safe to invoke
     * from host code that constructs a prebuilt arena image.
     *
     * Production callers leave `initial_local_task_id` at 0: the SM ring
     * flow-control counters that current_index_ptr / last_alive_ptr point at
     * start at zero (PTO2RingFlowControl::init() runs on the AICPU during SM
     * reset), so we keep local_task_id_ aligned with that without reading the
     * SM. Tests that drive SM state directly may pass a non-zero seed to
     * exercise corner cases like task IDs near INT32_MAX.
     */
    void init(
        PTO2TaskDescriptor *descriptors, int32_t window_size, std::atomic<int32_t> *current_index_ptr,
        std::atomic<int32_t> *last_alive_ptr, void *heap_base, uint64_t heap_size, std::atomic<int32_t> *error_code_ptr,
        PTO2TaskSlotState *slot_states = nullptr, int32_t initial_local_task_id = 0, uint8_t ring_id = 0
    ) {
        descriptors_ = descriptors;
        slot_states_ = slot_states;
        ring_id_ = ring_id;
        window_size_ = window_size;
        window_mask_ = window_size - 1;
        current_index_ptr_ = current_index_ptr;
        last_alive_ptr_ = last_alive_ptr;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        error_code_ptr_ = error_code_ptr;
        local_task_id_ = initial_local_task_id;
        heap_top_ = 0;
        heap_tail_ = 0;
        last_alive_seen_ = 0;
    }

    /**
     * Allocate a task slot and its associated output buffer in one call.
     *
     * Both task index and heap top are maintained as local counters and
     * published to shared memory only on success. Since the orchestrator is
     * single-threaded, no CAS or fetch_add is needed — just check-then-commit.
     *
     * @param output_size  Total packed output size in bytes (0 = no heap needed)
     * @return Allocation result; check failed() for errors
     */
    PTO2TaskAllocResult alloc(int32_t output_size) {
        uint64_t aligned_size =
            output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;

        int spin_count = 0;
        int32_t prev_last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        int32_t last_alive = prev_last_alive;
        update_heap_tail(last_alive);
        bool blocked_on_heap = false;
        uint64_t block_cycle0 = 0;  // wall-clock anchor for the deadlock backstop
        bool block_timing = false;  // false until the first no-reclaim-progress spin
#if SIMPLER_ORCH_PROFILING
        uint64_t wait_start = 0;
        bool waiting = false;
#endif

        while (true) {
            // Check both resources; commit only if both available
            if (local_task_id_ - last_alive + 1 < window_size_) {
                void *heap_ptr = try_bump_heap(aligned_size);
                if (heap_ptr) {
                    int32_t task_id = commit_task();
#if SIMPLER_ORCH_PROFILING
                    record_wait(spin_count, wait_start, waiting);
#endif
                    return {task_id, task_id & window_mask_, heap_ptr, static_cast<char *>(heap_ptr) + aligned_size};
                }
                blocked_on_heap = true;
            } else {
                blocked_on_heap = false;
            }

            // Spin: wait for scheduler to advance last_task_alive
            spin_count++;
#if SIMPLER_ORCH_PROFILING
            if (!waiting) {
                wait_start = get_sys_cnt_aicpu();
                waiting = true;
            }
#endif
            last_alive = last_alive_ptr_->load(std::memory_order_acquire);
            update_heap_tail(last_alive);
            if (last_alive > prev_last_alive) {
                // Reclaim advanced -> productive backpressure, not a deadlock.
                spin_count = 0;
                prev_last_alive = last_alive;
                block_timing = false;
            } else if ((spin_count & 1023) == 0) {
                // A fatal latched elsewhere breaks this otherwise-unbounded spin; the
                // caller maps the failed alloc to orch_mark_fatal. Polled on the
                // cold path only -- error_code_ptr_ is orch_error_code.
                if (error_code_ptr_ != nullptr && error_code_ptr_->load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
                    return {-1, -1, nullptr, nullptr};
                }
                // Reclaim watermark is stuck. Run the deadlock checks only once
                // per 1024 spins to keep the hot reclaim loop tight:
                // get_sys_cnt_aicpu() is a cheap cntvct_el0 read, while this
                // block polls the fatal flag and head_blocked_on_scope_end()
                // walks the head slot (1024 spins is far below the wall-clock
                // timeout, so detection latency is unaffected).
                // (1) Structural, immediate: if the head task is COMPLETED with
                // every consumer released but its scope still open, only
                // scope_end can free it and a blocked orchestrator can never
                // call it -> provable deadlock now.
                if (head_blocked_on_scope_end(last_alive)) {
                    report_deadlock(output_size, blocked_on_heap, /*scope_gated=*/true);
                    return {-1, -1, nullptr, nullptr};
                }
                // (2) Wall-clock backstop for the residual case the local head
                // test can't prove (e.g. a closed sibling whose consumer is
                // deferred). Absolute time, not a spin count.
                uint64_t now = get_sys_cnt_aicpu();
                if (!block_timing) {
                    block_cycle0 = now;
                    block_timing = true;
                } else if (now - block_cycle0 >= PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES) {
                    report_deadlock(output_size, blocked_on_heap, /*scope_gated=*/false);
                    return {-1, -1, nullptr, nullptr};
                }
                if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0) {
                    LOG_WARN(
                        "[TaskAllocator ring=%u] BLOCKED: tasks=%d/%d, heap_used=%" PRIu64 "/%" PRIu64
                        ", heap_available=%" PRIu64 ", heap_cursor=%" PRIu64 ", on=%s, spins=%d",
                        static_cast<unsigned>(ring_id_), local_task_id_ - last_alive, window_size_, heap_used_bytes(),
                        heap_size_, heap_available(), heap_top_, blocked_on_heap ? "heap" : "task", spin_count
                    );
                }
            }
            SPIN_WAIT_HINT();
        }
    }

    // =========================================================================
    // State queries
    // =========================================================================

    int32_t active_count() const {
        int32_t last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        return local_task_id_ - last_alive;
    }

    // Task ring start/end: tail = oldest live task (last_task_alive), head =
    // next task id to allocate. head - tail == active_count().
    int32_t task_tail() const { return last_alive_ptr_->load(std::memory_order_acquire); }
    int32_t task_head() const { return local_task_id_; }

    int32_t window_size() const { return window_size_; }

    uint64_t heap_available() const {
        uint64_t tail = heap_tail_;
        if (heap_top_ >= tail) {
            uint64_t at_end = heap_size_ - heap_top_;
            uint64_t at_begin = tail;
            return at_end > at_begin ? at_end : at_begin;
        }
        return tail - heap_top_;
    }

    uint64_t heap_top() const { return heap_top_; }
    // Heap ring start: reclaim pointer (oldest byte still live). heap_top() is
    // the end (next allocation). heap_top - heap_tail == heap_used_bytes().
    uint64_t heap_tail() const { return heap_tail_; }
    uint64_t heap_capacity() const { return heap_size_; }
    uint64_t heap_used_bytes() const {
        if (heap_size_ == 0) return 0;
        return (heap_top_ + heap_size_ - heap_tail_) % heap_size_;
    }

private:
    // --- Task Ring ---
    PTO2TaskDescriptor *descriptors_ = nullptr;
    // Parallel to descriptors_, indexed by task_id & window_mask_. Read-only here,
    // used by the deadlock detector to inspect the head task's state + fanout.
    PTO2TaskSlotState *slot_states_ = nullptr;
    uint8_t ring_id_ = 0;
    int32_t window_size_ = 0;
    int32_t window_mask_ = 0;
    std::atomic<int32_t> *current_index_ptr_ = nullptr;
    std::atomic<int32_t> *last_alive_ptr_ = nullptr;

    // --- Heap ---
    void *heap_base_ = nullptr;
    uint64_t heap_size_ = 0;

    // --- Local state (single-writer, no atomics needed) ---
    int32_t local_task_id_ = 0;    // Next task ID to allocate
    uint64_t heap_top_ = 0;        // Current heap allocation pointer
    uint64_t heap_tail_ = 0;       // Heap reclamation pointer (derived from consumed tasks)
    int32_t last_alive_seen_ = 0;  // last_task_alive at last heap_tail derivation

    // --- Shared ---
    std::atomic<int32_t> *error_code_ptr_ = nullptr;

    // =========================================================================
    // Internal helpers
    // =========================================================================

    /**
     * Commit a task slot: bump local counter and publish to shared memory.
     * Must only be called after space check has passed.
     */
    int32_t commit_task() {
        int32_t task_id = local_task_id_++;
        current_index_ptr_->store(local_task_id_, std::memory_order_release);
        return task_id;
    }

    /**
     * Derive heap_tail_ from the last consumed task's packed_buffer_end.
     *
     * Every task has a valid packed_buffer_end (equal to packed_buffer_base
     * for zero-size allocations), so the last consumed task always determines
     * the correct heap_tail — no backward scan needed.
     */
    void update_heap_tail(int32_t last_alive) {
        if (last_alive <= last_alive_seen_) return;
        last_alive_seen_ = last_alive;

        PTO2TaskDescriptor &desc = descriptors_[(last_alive - 1) & window_mask_];
        uint64_t old_tail = heap_tail_;
        heap_tail_ =
            static_cast<uint64_t>(static_cast<char *>(desc.packed_buffer_end) - static_cast<char *>(heap_base_));
#if SIMPLER_DFX
        // Reclaim pointer moves forward monotonically in ring order; a decrease
        // means it wrapped past heap_size_ (occupancy < heap_size_ guarantees at
        // most one wrap per call). Report it so scope_stats can unroll.
        if (is_scope_stats_enabled() && heap_tail_ < old_tail) {
            scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_RECLAIM);
        }
#else
        (void)old_tail;
#endif
    }

    /**
     * Bump the heap pointer for the given allocation size.
     * Returns the allocated pointer, or nullptr if insufficient space.
     * When alloc_size == 0, returns current position without advancing.
     */
    void *try_bump_heap(uint64_t alloc_size) {
        uint64_t top = heap_top_;
        if (alloc_size == 0) {
            return static_cast<char *>(heap_base_) + top;
        }
        uint64_t tail = heap_tail_;
        void *result;

        if (top >= tail) {
            uint64_t space_at_end = heap_size_ - top;
            if (space_at_end >= alloc_size) {
                result = static_cast<char *>(heap_base_) + top;
                heap_top_ = top + alloc_size;
            } else if (tail > alloc_size) {
                LOG_DEBUG(
                    "try_bump_heap wrap-around alloc: top=%" PRIu64 ", tail=%" PRIu64 ", alloc=%" PRIu64, top, tail,
                    alloc_size
                );
                result = heap_base_;
                heap_top_ = alloc_size;
#if SIMPLER_DFX
                // Allocation pointer just wrapped past heap_size_; report it so
                // scope_stats can unroll the wrapping offset into a monotonic value.
                // The collector attributes the wrap to the current scope's ring.
                if (is_scope_stats_enabled()) scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_ALLOC);
#endif
            } else {
                LOG_DEBUG(
                    "try_bump_heap failed (top>=tail): top=%" PRIu64 ", tail=%" PRIu64 ", alloc=%" PRIu64
                    ", heap_size=%" PRIu64,
                    top, tail, alloc_size, heap_size_
                );
                return nullptr;
            }
        } else {
            if (tail - top > alloc_size) {
                result = static_cast<char *>(heap_base_) + top;
                heap_top_ = top + alloc_size;
            } else {
                LOG_DEBUG(
                    "try_bump_heap failed (top<tail): top=%" PRIu64 ", tail=%" PRIu64 ", alloc=%" PRIu64
                    ", free_gap=%" PRIu64,
                    top, tail, alloc_size, tail - top
                );
                return nullptr;
            }
        }

        return result;
    }

#if SIMPLER_ORCH_PROFILING
    void record_wait(int spin_count, uint64_t wait_start, bool waiting) {
        if (waiting) {
            extern uint64_t g_orch_alloc_wait_cycle;
            g_orch_alloc_wait_cycle += (get_sys_cnt_aicpu() - wait_start);
        }
        {
            extern uint64_t g_orch_alloc_atomic_count;
            g_orch_alloc_atomic_count += spin_count + 1;
        }
    }
#endif

    /**
     * Structural deadlock test on the reclaim head.
     *
     * The head (oldest un-CONSUMED task, at last_task_alive) gates all
     * reclamation. If it is COMPLETED and every consumer reference is released
     * (low bits of fanout_refcount == consumer count) but the scope reference
     * (bit31) is still unset, the only release left is its scope_end. Because
     * this is evaluated while the orchestrator is blocked in alloc(), scope_end
     * can never be reached -> provable deadlock, no timeout required.
     *
     * The COMPLETED guard is mandatory: a zero-consumer task has
     * refcount == 0 == (count & ~SCOPE_BIT) from birth, before it has run.
     */
    bool head_blocked_on_scope_end(int32_t head_task_id) const {
        if (slot_states_ == nullptr) return false;
        PTO2TaskSlotState &h = slot_states_[head_task_id & window_mask_];
        if (h.task_state.load(std::memory_order_acquire) != PTO2_TASK_COMPLETED) return false;
        uint32_t rc = h.fanout_refcount.load(std::memory_order_acquire);
        return rc == (h.fanout_count & ~PTO2_FANOUT_SCOPE_BIT);
    }

    /**
     * Report deadlock with targeted diagnostics. scope_gated == true means the
     * head-of-line structural test proved it (waiting only on scope_end);
     * false means the wall-clock backstop fired.
     */
    void report_deadlock(int32_t requested_output_size, bool heap_blocked, bool scope_gated) {
        int32_t last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        int32_t active_tasks = local_task_id_ - last_alive;
        uint64_t htail = heap_tail_;

        LOG_ERROR("========================================");
        if (heap_blocked) {
            LOG_ERROR("FATAL: Task Allocator Deadlock - Heap Exhausted! ring=%u", static_cast<unsigned>(ring_id_));
        } else {
            LOG_ERROR("FATAL: Task Allocator Deadlock - Task Ring Full! ring=%u", static_cast<unsigned>(ring_id_));
        }
        LOG_ERROR("========================================");
        if (scope_gated) {
            LOG_ERROR("Head task %d COMPLETED, all consumers released, scope still open ->", last_alive);
            LOG_ERROR("only scope_end can free it and the orchestrator is blocked here.");
            LOG_ERROR("Provable head-of-line deadlock.");
        } else {
            LOG_ERROR(
                "No reclaim progress for ~500 ms (%" PRIu64 " cycles wall clock).",
                (uint64_t)PTO2_ALLOC_DEADLOCK_TIMEOUT_CYCLES
            );
        }
        LOG_ERROR(
            "  Task ring %u: current=%d, last_alive=%d, active=%d/%d (%.1f%%)", static_cast<unsigned>(ring_id_),
            local_task_id_, last_alive, active_tasks, window_size_, 100.0 * active_tasks / window_size_
        );
        LOG_ERROR(
            "  Heap ring %u: top=%" PRIu64 ", tail=%" PRIu64 ", size=%" PRIu64 ", used=%" PRIu64 ", available=%" PRIu64,
            static_cast<unsigned>(ring_id_), heap_top_, htail, heap_size_, heap_used_bytes(), heap_available()
        );
        if (heap_blocked) {
            LOG_ERROR("  Requested:  %d bytes", requested_output_size);
        }
        // Head-task state dump: what the reclaim watermark is actually waiting on.
        if (slot_states_ != nullptr) {
            PTO2TaskSlotState &h = slot_states_[last_alive & window_mask_];
            uint32_t fc = h.fanout_count;
            uint32_t rc = h.fanout_refcount.load(std::memory_order_acquire);
            LOG_ERROR(
                "  Head task %d: state=%d, consumers=%u/%u, scope_released=%d", last_alive,
                static_cast<int>(h.task_state.load(std::memory_order_acquire)), rc & ~PTO2_FANOUT_SCOPE_BIT,
                fc & ~PTO2_FANOUT_SCOPE_BIT, (rc & PTO2_FANOUT_SCOPE_BIT) ? 1 : 0
            );
        }
        LOG_ERROR("Solution:");
        if (scope_gated) {
            LOG_ERROR("  The open scope's own allocation exceeds this ring. Either:");
            LOG_ERROR("  1. Split the scope / reduce per-scope allocation (reclaim sooner), or");
            LOG_ERROR("  2. Size the ring >= the scope's peak live-set (heap*2 may not be enough).");
        } else if (heap_blocked) {
            LOG_ERROR(
                "  Increase heap (current: %" PRIu64 "); env PTO2_RING_HEAP=<bytes> (e.g. %" PRIu64 ")", heap_size_,
                heap_size_ * 2
            );
            LOG_ERROR(
                "  If one increase completes, it was under-provisioned; otherwise debug the stuck head consumer."
            );
        } else {
            LOG_ERROR(
                "  Increase task window (current: %d); env PTO2_RING_TASK_WINDOW=<pow2> (e.g. %d)", window_size_,
                active_tasks * 2
            );
            LOG_ERROR(
                "  If one increase completes, it was under-provisioned; otherwise debug the stuck head consumer."
            );
        }
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            int32_t code = heap_blocked ? PTO2_ERROR_HEAP_RING_DEADLOCK : PTO2_ERROR_FLOW_CONTROL_DEADLOCK;
            error_code_ptr_->store(code, std::memory_order_release);
        }
    }
};

// =============================================================================
// Fanin Spill Pool
// =============================================================================

/**
 * Fanin spill pool structure
 *
 * True ring buffer for allocating spilled fanin entries.
 * Entries are reclaimed when their consumer tasks become CONSUMED.
 *
 * Linear counters (top, tail) grow monotonically; the physical index
 * is obtained via modulo: base[linear_index % capacity].
 */
struct PTO2FaninPool {
    PTO2FaninSpillEntry *base;       // Pool base address
    int32_t capacity;                // Total number of entries
    int32_t top;                     // Linear next-allocation counter (starts from 1)
    int32_t tail;                    // Linear first-alive counter (entries before this are dead)
    int32_t high_water;              // Peak concurrent usage (top - tail)
    int32_t reclaim_task_cursor{0};  // Last task id scanned for reclaim on this pool

    std::atomic<int32_t> *error_code_ptr = nullptr;

    void init(PTO2FaninSpillEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr) {
        base = in_base;
        capacity = in_capacity;
        top = 1;
        tail = 1;
        high_water = 0;
        reclaim_task_cursor = 0;
        base[0].slot_state = PTO2TaggedSlotPtr{};
        error_code_ptr = in_error_code_ptr;
    }

    void reset_for_reuse(std::atomic<int32_t> *in_error_code_ptr) {
        top = 1;
        tail = 1;
        high_water = 0;
        reclaim_task_cursor = 0;
        base[0].slot_state = PTO2TaggedSlotPtr{};
        error_code_ptr = in_error_code_ptr;
    }

    void reclaim(PTO2SharedMemoryRingHeader &ring, int32_t sm_last_task_alive);

    bool ensure_space(PTO2SharedMemoryRingHeader &ring, int32_t needed);

    PTO2FaninSpillEntry *alloc() {
        int32_t used = top - tail;
        if (used >= capacity) {
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Fanin Spill Pool Overflow!");
            LOG_ERROR("========================================");
            LOG_ERROR("Fanin spill pool exhausted: %d entries alive (capacity=%d).", used, capacity);
            LOG_ERROR("  - Pool top:      %d (linear)", top);
            LOG_ERROR("  - Pool tail:     %d (linear)", tail);
            LOG_ERROR("  - High water:    %d", high_water);
            LOG_ERROR("Solution:");
            LOG_ERROR("  Increase fanin spill pool capacity (current: %d, recommended: %d).", capacity, capacity * 2);
            LOG_ERROR("  Compile-time: PTO2_DEP_LIST_POOL_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Runtime env:  PTO2_RING_DEP_POOL=%d", capacity * 2);
            LOG_ERROR("========================================");
            if (error_code_ptr) {
                error_code_ptr->store(PTO2_ERROR_DEP_POOL_OVERFLOW, std::memory_order_release);
            }
            return nullptr;
        }
        int32_t idx = top % capacity;
        top++;
        used++;
        if (used > high_water) high_water = used;
        return &base[idx];
    }

    void advance_tail(int32_t new_tail) {
        if (new_tail > tail) {
            tail = new_tail;
        }
    }

    int32_t used() const { return top - tail; }

    int32_t available() const { return capacity - used(); }
};
template <typename Fn, typename = void>
struct PTO2FaninCallbackResultHelper {
    using type = std::invoke_result_t<Fn &, PTO2TaskSlotState *>;
};
template <typename Fn>
struct PTO2FaninCallbackResultHelper<Fn, std::void_t<std::invoke_result_t<Fn &, PTO2TaskSlotState *, PTO2DepFlags>>> {
    using type = std::invoke_result_t<Fn &, PTO2TaskSlotState *, PTO2DepFlags>;
};

template <typename Fn>
using PTO2FaninCallbackResult = typename PTO2FaninCallbackResultHelper<Fn>::type;

template <typename Fn>
using PTO2FaninForEachReturn = std::conditional_t<std::is_same_v<PTO2FaninCallbackResult<Fn>, void>, void, bool>;

// Walks fanin storage. Two callback forms, picked at compile time:
//   fn(slot_state, dep_flags) — unfiltered (pass select == PTO2_DEP_NONE):
//     every edge is reported with its flags, for callers that dispatch on
//     the kind per edge (e.g. wire_fanin_task handles both kinds in one
//     pass).
//   fn(slot_state)            — filtered: the callback fires only for edges
//     whose flags EXACTLY equal `select` and never sees the flags.
// For the filtered form the traversal strategy is internal: when the
// storage is retain-first (retain_count >= 0 marks the sorted prefix, see
// PTO2TaskPayload::fanin_retain_count) the two kinds are contiguous
// segments and the select resolves to a range with no per-entry check
// ([0, retain_count) for WAIT_RETAIN, [retain_count, fanin_count) for
// WAIT). Unsorted storage (retain_count < 0, e.g. the submit-time builder)
// and the spilled fallback (retain_count == fanin_count > inline cap) use
// a full walk with a per-entry exact-match test done here, never in the
// callback.
template <typename InlineSlots, typename Fn>
inline PTO2FaninForEachReturn<Fn> for_each_fanin_storage(
    InlineSlots &&inline_slot_states, int32_t fanin_count, int32_t retain_count, int32_t spill_start,
    PTO2FaninPool &spill_pool, PTO2DepFlags select, Fn &&fn
) {
    constexpr bool kWithFlags = std::is_invocable_v<Fn &, PTO2TaskSlotState *, PTO2DepFlags>;
    static_assert(
        kWithFlags || std::is_invocable_v<Fn &, PTO2TaskSlotState *>,
        "fanin callback must take (slot_state) or (slot_state, dep_flags)"
    );
    using FaninCallbackResult = PTO2FaninCallbackResult<Fn>;
    static_assert(
        std::is_same_v<FaninCallbackResult, void> || std::is_same_v<FaninCallbackResult, bool>,
        "fanin callback must return void or bool"
    );

    bool unsorted = retain_count < 0 || (retain_count == fanin_count && fanin_count > PTO2_FANIN_INLINE_CAP);
    int32_t lo = 0;
    int32_t hi = fanin_count;
    bool check = !kWithFlags && select != PTO2_DEP_NONE;
    if (!kWithFlags && !unsorted) {
        check = false;
        if (select == PTO2_DEP_WAIT_RETAIN) {
            hi = retain_count;
        } else if (select == PTO2_DEP_WAIT) {
            lo = retain_count;
        } else if (select == PTO2_DEP_RETAIN) {
            hi = lo;  // RETAIN-only edges: no segment exists today
        }
    }

    auto entry_at = [&](int32_t i) -> const PTO2TaggedSlotPtr & {
        if (i < PTO2_FANIN_INLINE_CAP) {
            return inline_slot_states[i];
        }
        return spill_pool.base[(spill_start + i - PTO2_FANIN_INLINE_CAP) % spill_pool.capacity].slot_state;
    };

    if constexpr (std::is_void_v<FaninCallbackResult>) {
        for (int32_t i = lo; i < hi; i++) {
            const PTO2TaggedSlotPtr &e = entry_at(i);
            if (check && e.flags() != select) continue;
            if constexpr (kWithFlags) fn(e.slot(), e.flags());
            else fn(e.slot());
        }
        return;
    } else {
        for (int32_t i = lo; i < hi; i++) {
            const PTO2TaggedSlotPtr &e = entry_at(i);
            if (check && e.flags() != select) continue;
            if constexpr (kWithFlags) {
                if (!fn(e.slot(), e.flags())) return false;
            } else {
                if (!fn(e.slot())) return false;
            }
        }
        return true;
    }
}

// Payload-owner wrapper: all storage parameters (counts, spill range, pool)
// come from the payload; callers only state which edges they want.
template <typename Fn>
inline PTO2FaninForEachReturn<Fn>
for_each_fanin_slot_state(const PTO2TaskPayload &payload, PTO2DepFlags select, Fn &&fn) {
    return for_each_fanin_storage(
        payload.fanin_inline_slot_states, payload.fanin_actual_count, payload.fanin_retain_count,
        payload.fanin_spill_start, *payload.fanin_spill_pool, select, std::forward<Fn>(fn)
    );
}

// =============================================================================
// Dependency List Pool
// =============================================================================

/**
 * Dependency list pool structure
 *
 * True ring buffer for allocating linked list entries.
 * Entries are reclaimed when their producer tasks become CONSUMED,
 * as tracked by the orchestrator via dep_pool_mark per task.
 *
 * Linear counters (top, tail) grow monotonically; the physical index
 * is obtained via modulo: base[linear_index % capacity].
 */
struct PTO2DepListPool {
    PTO2DepListEntry *base;     // Pool base address
    int32_t capacity;           // Total number of entries
    int32_t top;                // Linear next-allocation counter (starts from 1)
    int32_t tail;               // Linear first-alive counter (entries before this are dead)
    int32_t high_water;         // Peak concurrent usage (top - tail)
    int32_t last_reclaimed{0};  // last_task_alive at last successful reclamation

    // Error code pointer for fatal error reporting (→ sm_header->orch_error_code)
    std::atomic<int32_t> *error_code_ptr = nullptr;

    /**
     *
     * Initialize dependency list pool
     * @param base      Pool base address from shared memory
     * @param capacity  Total number of entries
     */
    void init(PTO2DepListEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr) {
        base = in_base;
        capacity = in_capacity;
        top = 1;   // Start from 1, 0 means NULL/empty
        tail = 1;  // Match initial top (no reclaimable entries yet)
        high_water = 0;
        last_reclaimed = 0;

        // Initialize entry 0 as NULL marker
        base[0].slot_state = PTO2TaggedSlotPtr{};
        base[0].next = nullptr;

        error_code_ptr = in_error_code_ptr;
    }

    void reset_for_reuse(std::atomic<int32_t> *in_error_code_ptr) {
        top = 1;
        tail = 1;
        high_water = 0;
        last_reclaimed = 0;
        base[0].slot_state = PTO2TaggedSlotPtr{};
        base[0].next = nullptr;
        error_code_ptr = in_error_code_ptr;
    }

    /**
     * Reclaim dead entries based on the slot state dep_pool_mark.
     * Safe to call multiple times — only advances tail forward.
     *
     * @param ring             Ring header (for reading slot dep_pool_mark)
     * @param sm_last_task_alive Current last_task_alive from shared memory
     */
    void reclaim(PTO2SharedMemoryRingHeader &ring, int32_t sm_last_task_alive);

    /**
     * Ensure dep pool for a specific ring has at least `needed` entries available.
     * Spin-waits for reclamation under pressure. The dep pool shares
     * last_task_alive with the heap and task rings, so it detects a wedged
     * reclaim watermark the same way PTO2TaskAllocator::alloc does: a structural
     * head-of-line check plus a wall-clock backstop, each emitting report_deadlock.
     */
    bool ensure_space(PTO2SharedMemoryRingHeader &ring, int32_t needed);

    /**
     * Structured dep-pool deadlock report, mirroring PTO2TaskAllocator::report_deadlock.
     * scope_gated marks the provable head-of-line case (head COMPLETED, all
     * consumers released, scope still open) as opposed to the wall-clock backstop.
     */
    void report_deadlock(PTO2SharedMemoryRingHeader &ring, int32_t needed, int32_t last_alive, bool scope_gated);

    /**
     * Allocate a single entry from the pool (single-thread per pool instance)
     *
     * @return Pointer to allocated entry, or nullptr on fatal error
     */
    PTO2DepListEntry *alloc() {
        int32_t used = top - tail;
        if (used >= capacity) {
            LOG_ERROR("========================================");
            LOG_ERROR("FATAL: Dependency Pool Overflow!");
            LOG_ERROR("========================================");
            LOG_ERROR("DepListPool exhausted: %d entries alive (capacity=%d).", used, capacity);
            LOG_ERROR("  - Pool top:      %d (linear)", top);
            LOG_ERROR("  - Pool tail:     %d (linear)", tail);
            LOG_ERROR("  - High water:    %d", high_water);
            LOG_ERROR("Solution:");
            LOG_ERROR("  Increase dep pool capacity (current: %d, recommended: %d).", capacity, capacity * 2);
            LOG_ERROR("  Compile-time: PTO2_DEP_LIST_POOL_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Runtime env:  PTO2_RING_DEP_POOL=%d", capacity * 2);
            LOG_ERROR("========================================");
            if (error_code_ptr) {
                error_code_ptr->store(PTO2_ERROR_DEP_POOL_OVERFLOW, std::memory_order_release);
            }
            return nullptr;
        }
        int32_t idx = top % capacity;
        top++;
        used++;
        if (used > high_water) high_water = used;
        return &base[idx];
    }

    /**
     * Advance the tail pointer, reclaiming dead entries.
     * Called by the orchestrator based on last_task_alive advancement.
     */
    void advance_tail(int32_t new_tail) {
        if (new_tail > tail) {
            tail = new_tail;
        }
    }

    /**
     * Prepend a task ID to a dependency list
     *
     * O(1) operation: allocates new entry and links to current head.
     *
     * @param current_head  Current list head offset (0 = empty list)
     * @param task_slot     Task slot to prepend
     * @return New head offset
     */
    PTO2DepListEntry *prepend(PTO2DepListEntry *cur, PTO2TaskSlotState *slot_state, PTO2DepFlags dep_flags) {
        PTO2DepListEntry *new_entry = alloc();
        if (!new_entry) return nullptr;
        new_entry->slot_state = PTO2TaggedSlotPtr::make(slot_state, dep_flags);
        new_entry->next = cur;
        return new_entry;
    }

    int32_t used() const { return top - tail; }

    int32_t available() const { return capacity - used(); }
};

// =============================================================================
// Ring Set (per-depth aggregate)
// =============================================================================

/**
 * Groups a TaskAllocator and DepPool into one per-depth unit.
 * PTO2_MAX_RING_DEPTH instances provide independent reclamation per scope depth.
 */
struct PTO2RingSet {
    PTO2TaskAllocator task_allocator;
    PTO2FaninPool fanin_pool;
};

#endif  // PTO_RING_BUFFER_H
