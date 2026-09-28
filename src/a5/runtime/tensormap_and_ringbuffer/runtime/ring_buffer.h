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
 * tensormap_and_ringbuffer ring buffer data structures
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

#include "runtime_types.h"
#include "tensormap_and_ringbuffer/task_id.h"
#include "shared_memory.h"
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
#define CHIP_BLOCK_NOTIFY_INTERVAL 10000
// Productive-loop publication is a liveness escape, not the normal pressure
// path. Require sustained no-progress before bypassing K-batched publication.
#define CHIP_PUBLICATION_REQUEST_TIMEOUT_CYCLES (PLATFORM_PROF_SYS_CNT_FREQ / 100)  // 10 ms
// Heap/task deadlock is detected structurally when the reclaim head is the
// oldest task owned by an open scope on the blocked ring. This wall-clock value
// is the backstop for all other cases; it is an ABSOLUTE TIME (not a spin
// count), so it is stable across chips/contention.
#define ALLOC_DEADLOCK_TIMEOUT_CYCLES (PLATFORM_PROF_SYS_CNT_FREQ / 2)  // 500 ms

constexpr uint32_t ring_mask_bit(int32_t ring_id) {
    static_assert(CHIP_MAX_RING_DEPTH <= 32, "ring masks use one uint32_t bit per ring");
    return 1u << static_cast<uint32_t>(ring_id);
}

inline bool
reclaim_head_matches_open_task(int32_t head_task_id, uint8_t ring_id, const ChipTaskSlotState *oldest_open_task) {
    return oldest_open_task != nullptr && oldest_open_task->task != nullptr &&
           oldest_open_task->task->task_id == TaskId::make(ring_id, head_task_id);
}

class ReclaimPublicationRequest {
public:
    ReclaimPublicationRequest(std::atomic<uint32_t> *request_mask, std::atomic<uint32_t> *ack_mask, uint8_t ring_id) :
        request_mask_(request_mask),
        ack_mask_(ack_mask),
        bit_(ring_mask_bit(ring_id)) {}

    bool enabled() const { return request_mask_ != nullptr && ack_mask_ != nullptr; }

    void request() {
        if (!enabled() || outstanding_) return;
        ack_mask_->fetch_and(~bit_, std::memory_order_acq_rel);
        request_mask_->fetch_or(bit_, std::memory_order_release);
        outstanding_ = true;
    }

    bool poll_acknowledged() {
        if (!enabled()) return true;
        if (!outstanding_ || (ack_mask_->load(std::memory_order_acquire) & bit_) == 0) return false;
        outstanding_ = false;
        return true;
    }

private:
    std::atomic<uint32_t> *request_mask_;
    std::atomic<uint32_t> *ack_mask_;
    uint32_t bit_;
    bool outstanding_{false};
};

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
class TaskAllocator {
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
     * start at zero (ChipRingFlowControl::init() runs on the AICPU during SM
     * reset), so we keep local_task_id_ aligned with that without reading the
     * SM. Tests that drive SM state directly may pass a non-zero seed to
     * exercise corner cases like task IDs near INT32_MAX.
     */
    void init(
        TaskDescriptor *descriptors, int32_t window_size, std::atomic<int32_t> *current_index_ptr,
        std::atomic<int32_t> *last_alive_ptr, void *heap_base, uint64_t heap_size, std::atomic<int32_t> *error_code_ptr,
        ChipTaskSlotState *slot_states = nullptr, int32_t initial_local_task_id = 0, uint8_t ring_id = 0
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
        heap_rebase_anchor_task_id_ = -1;
        reclaim_request_mask_ = nullptr;
        reclaim_ack_mask_ = nullptr;
    }

    void set_reclaim_publication_request(std::atomic<uint32_t> *request_mask, std::atomic<uint32_t> *ack_mask) {
        reclaim_request_mask_ = request_mask;
        reclaim_ack_mask_ = ack_mask;
    }

    bool reclaim_publication_is_wired_to(
        const std::atomic<uint32_t> *request_mask, const std::atomic<uint32_t> *ack_mask, uint8_t expected_ring_id
    ) const {
        return reclaim_request_mask_ == request_mask && reclaim_ack_mask_ == ack_mask && ring_id_ == expected_ring_id;
    }

    /**
     * Allocate a task slot and its associated output buffer in one call.
     *
     * Both task index and heap top are maintained as local counters and
     * published to shared memory only on success. Since the orchestrator is
     * single-threaded, no CAS or fetch_add is needed — just check-then-commit.
     *
     * @param output_size     Total packed output size in bytes (0 = no heap needed)
     * @param oldest_open_task Oldest task owned by any open scope on this ring
     * @return Allocation result; check failed() for errors
     */
    TaskAllocResult alloc(int32_t output_size, ChipTaskSlotState *oldest_open_task = nullptr) {
        uint64_t aligned_size =
            output_size > 0 ? CHIP_ALIGN_UP(static_cast<uint64_t>(output_size), CHIP_ALIGN_SIZE) : 0;

        int spin_count = 0;
        int32_t prev_last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        int32_t last_alive = prev_last_alive;
        update_heap_tail(last_alive);
        bool blocked_on_heap = false;
        uint64_t block_cycle0 = 0;  // wall-clock anchor for the deadlock backstop
        bool block_timing = false;  // false until the first no-reclaim-progress spin
        ReclaimPublicationRequest publication_request(reclaim_request_mask_, reclaim_ack_mask_, ring_id_);
        bool watermark_synchronized = !publication_request.enabled();
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
            if (!watermark_synchronized && publication_request.poll_acknowledged()) {
                last_alive = last_alive_ptr_->load(std::memory_order_acquire);
                update_heap_tail(last_alive);
                watermark_synchronized = true;
            }
            if (last_alive > prev_last_alive) {
                // Reclaim advanced -> productive backpressure, not a deadlock.
                spin_count = 0;
                prev_last_alive = last_alive;
                block_timing = false;
                // Any later shared progress can again trail scheduler-local
                // reclamation by K - 1, so a prior exact acknowledgment no
                // longer proves that the current watermark is exact.
                watermark_synchronized = !publication_request.enabled();
            } else if ((spin_count & 1023) == 0) {
                // A fatal latched elsewhere breaks this otherwise-unbounded spin; the
                // caller maps the failed alloc to orch_mark_fatal. Polled on the
                // cold path only -- error_code_ptr_ is orch_error_code.
                if (error_code_ptr_ != nullptr &&
                    error_code_ptr_->load(std::memory_order_acquire) != SIMPLER_ERROR_NONE) {
                    return {-1, -1, nullptr, nullptr};
                }
                // Reclaim watermark is stuck. Run the deadlock checks only once
                // per 1024 spins to keep the hot reclaim loop tight:
                // get_sys_cnt_aicpu() is a cheap cntvct_el0 read, while this
                // block polls the fatal flag and compares the reclaim head with
                // the oldest task pinned by an open scope on this ring.
                uint64_t now = get_sys_cnt_aicpu();
                if (!block_timing) {
                    block_cycle0 = now;
                    block_timing = true;
                } else if (!watermark_synchronized && now - block_cycle0 >= CHIP_PUBLICATION_REQUEST_TIMEOUT_CYCLES) {
                    publication_request.request();
                }
                // (1) Structural, after publication acknowledgment: no open
                // scope can end while this orchestrator is blocked here, so
                // the synchronized head cannot become CONSUMED.
                if (watermark_synchronized && head_is_oldest_open_task(last_alive, oldest_open_task)) {
                    report_deadlock(output_size, blocked_on_heap, /*scope_gated=*/true);
                    return {-1, -1, nullptr, nullptr};
                }
                // (2) Wall-clock backstop for the residual case the local head
                // test can't prove (e.g. a closed sibling whose consumer is
                // deferred). Absolute time, not a spin count.
                if (now - block_cycle0 >= ALLOC_DEADLOCK_TIMEOUT_CYCLES) {
                    report_deadlock(output_size, blocked_on_heap, /*scope_gated=*/false);
                    return {-1, -1, nullptr, nullptr};
                }
                if (spin_count % CHIP_BLOCK_NOTIFY_INTERVAL == 0) {
                    LOG_WARN(
                        "[TaskAllocator ring=%u] BLOCKED: tasks=%d/%d, heap_used=%" PRIu64 "/%" PRIu64
                        ", heap_available=%" PRIu64 ", heap_cursor=%" PRIu64 ", on=%s, spins=%d",
                        static_cast<unsigned>(ring_id_), local_task_id_ - last_alive, window_size_, heap_used_bytes(),
                        heap_size_, heap_available(), heap_top_, blocked_on_heap ? "heap" : "task", spin_count
                    );
                }
                watermark_synchronized = !publication_request.enabled();
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
        if (heap_top_ == tail) {
            return heap_size_;
        }
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
    TaskDescriptor *descriptors_ = nullptr;
    // Parallel to descriptors_, indexed by task_id & window_mask_. Read-only here,
    // used by the deadlock detector to identify the head task's slot.
    ChipTaskSlotState *slot_states_ = nullptr;
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
    // While last_alive has not advanced past this task ID, the latest
    // reclaimed descriptor still predates the empty-ring rebase.
    int32_t heap_rebase_anchor_task_id_ = -1;

    // --- Shared ---
    std::atomic<int32_t> *error_code_ptr_ = nullptr;
    std::atomic<uint32_t> *reclaim_request_mask_ = nullptr;
    std::atomic<uint32_t> *reclaim_ack_mask_ = nullptr;

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

        if (heap_rebase_anchor_task_id_ >= 0 && last_alive <= heap_rebase_anchor_task_id_) {
            return;
        }
        heap_rebase_anchor_task_id_ = -1;

        TaskDescriptor &desc = descriptors_[(last_alive - 1) & window_mask_];
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
            } else if (top == tail && alloc_size <= heap_size_) {
                result = heap_base_;
                heap_top_ = alloc_size;
                heap_tail_ = 0;
                heap_rebase_anchor_task_id_ = local_task_id_;
#if SIMPLER_DFX
                if (is_scope_stats_enabled()) {
                    scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_ALLOC);
                    scope_stats_note_heap_wrap(SCOPE_STATS_HEAP_SIDE_RECLAIM);
                }
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

    bool head_is_oldest_open_task(int32_t head_task_id, const ChipTaskSlotState *oldest_open_task) const {
        return slot_states_ != nullptr && reclaim_head_matches_open_task(head_task_id, ring_id_, oldest_open_task);
    }

    /**
     * Report deadlock with targeted diagnostics. scope_gated == true means the
     * head is pinned by an open scope on this ring; false means the wall-clock
     * backstop fired.
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
            LOG_ERROR("Head task %d is the oldest task owned by an open scope on this ring ->", last_alive);
            LOG_ERROR("no open scope can end while the orchestrator is blocked here.");
            LOG_ERROR("Provable head-of-line deadlock.");
        } else {
            LOG_ERROR(
                "No reclaim progress for ~500 ms (%" PRIu64 " cycles wall clock).",
                (uint64_t)ALLOC_DEADLOCK_TIMEOUT_CYCLES
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
            ChipTaskSlotState &h = slot_states_[last_alive & window_mask_];
            uint32_t fc = h.fanout_count;
            uint32_t rc = h.fanout_refcount.load(std::memory_order_acquire);
            LOG_ERROR(
                "  Head task %d: state=%d, consumers=%u/%u, scope_released=%d", last_alive,
                static_cast<int>(h.task_state.load(std::memory_order_acquire)), rc & ~FANOUT_SCOPE_BIT,
                fc & ~FANOUT_SCOPE_BIT, (rc & FANOUT_SCOPE_BIT) ? 1 : 0
            );
        }
        LOG_ERROR("Solution:");
        if (scope_gated) {
            LOG_ERROR("  The live-set retained behind this open-scope head cannot fit in the ring. Either:");
            LOG_ERROR("  1. Split scopes / reduce the live-set retained behind that head, or");
            LOG_ERROR("  2. Size the ring for the peak live-set retained behind the oldest open-scope task.");
        } else if (heap_blocked) {
            LOG_ERROR(
                "  Increase heap (current: %" PRIu64 "); CallConfig.runtime_env.ring_heap=<bytes> (e.g. %" PRIu64 ")",
                heap_size_, heap_size_ * 2
            );
            LOG_ERROR(
                "  If one increase completes, it was under-provisioned; otherwise debug the stuck head consumer."
            );
        } else {
            LOG_ERROR(
                "  Increase task window (current: %d); CallConfig.runtime_env.ring_task_window=<pow2> (e.g. %d)",
                window_size_, active_tasks * 2
            );
            LOG_ERROR(
                "  If one increase completes, it was under-provisioned; otherwise debug the stuck head consumer."
            );
        }
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            int32_t code = heap_blocked ? SIMPLER_ERROR_HEAP_RING_DEADLOCK : SIMPLER_ERROR_FLOW_CONTROL_DEADLOCK;
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
struct FaninPool {
    FaninSpillEntry *base;           // Pool base address
    int32_t capacity;                // Total number of entries
    int32_t top;                     // Linear next-allocation counter (starts from 1)
    int32_t tail;                    // Linear first-alive counter (entries before this are dead)
    int32_t high_water;              // Peak concurrent usage (top - tail)
    int32_t reclaim_task_cursor{0};  // Last task id scanned for reclaim on this pool

    std::atomic<int32_t> *error_code_ptr = nullptr;
    std::atomic<uint32_t> *reclaim_request_mask = nullptr;
    std::atomic<uint32_t> *reclaim_ack_mask = nullptr;
    uint8_t ring_id = 0;

    void init(FaninSpillEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr) {
        base = in_base;
        capacity = in_capacity;
        top = 1;
        tail = 1;
        high_water = 0;
        reclaim_task_cursor = 0;
        base[0].clear();
        error_code_ptr = in_error_code_ptr;
        reclaim_request_mask = nullptr;
        reclaim_ack_mask = nullptr;
        ring_id = 0;
    }

    void set_reclaim_publication_request(
        std::atomic<uint32_t> *request_mask, std::atomic<uint32_t> *ack_mask, uint8_t in_ring_id
    ) {
        reclaim_request_mask = request_mask;
        reclaim_ack_mask = ack_mask;
        ring_id = in_ring_id;
    }

    bool reclaim_publication_is_wired_to(
        const std::atomic<uint32_t> *request_mask, const std::atomic<uint32_t> *ack_mask, uint8_t expected_ring_id
    ) const {
        return reclaim_request_mask == request_mask && reclaim_ack_mask == ack_mask && ring_id == expected_ring_id;
    }

    void reset_for_reuse(std::atomic<int32_t> *in_error_code_ptr) {
        top = 1;
        tail = 1;
        high_water = 0;
        reclaim_task_cursor = 0;
        base[0].clear();
        error_code_ptr = in_error_code_ptr;
    }

    void reclaim(SharedMemoryRingHeader &ring, int32_t sm_last_task_alive);

    bool ensure_space(SharedMemoryRingHeader &ring, int32_t needed);

    FaninSpillEntry *alloc() {
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
            LOG_ERROR("  Compile-time: CHIP_DEP_LIST_POOL_SIZE in runtime_types.h");
            LOG_ERROR("  Per task:     CallConfig.runtime_env.ring_dep_pool=%d", capacity * 2);
            LOG_ERROR("========================================");
            if (error_code_ptr) {
                error_code_ptr->store(SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED, std::memory_order_release);
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

template <typename Fn>
using FaninCallbackResult = std::invoke_result_t<Fn &, ChipTaskSlotState *, DepFlags>;

template <typename Fn>
using FaninForEachReturn = std::conditional_t<std::is_same_v<FaninCallbackResult<Fn>, void>, void, bool>;

// Visit each fanin edge as (producer slot, DepFlags). Inline and spill entries
// share the packed FaninSpillEntry layout, so both are unpacked the same way.
template <typename InlineSlots, typename Fn>
inline FaninForEachReturn<Fn> for_each_fanin_storage(
    InlineSlots &&inline_edges, int32_t fanin_count, int32_t spill_start, FaninPool &spill_pool, Fn &&fn
) {
    using FaninCallbackResult = FaninCallbackResult<Fn>;
    static_assert(
        std::is_same_v<FaninCallbackResult, void> || std::is_same_v<FaninCallbackResult, bool>,
        "fanin callback must return void or bool"
    );

    if constexpr (std::is_void_v<FaninCallbackResult>) {
        int32_t inline_count = std::min(fanin_count, CHIP_FANIN_INLINE_CAP);
        for (int32_t i = 0; i < inline_count; i++) {
            fn(inline_edges[i].slot_state(), inline_edges[i].flags());
        }

        int32_t spill_count = fanin_count - inline_count;
        if (spill_count <= 0) {
            return;
        }

        int32_t start_idx = spill_start % spill_pool.capacity;
        int32_t first_count = std::min(spill_count, spill_pool.capacity - start_idx);
        FaninSpillEntry *first = spill_pool.base + start_idx;
        for (int32_t i = 0; i < first_count; i++) {
            fn(first[i].slot_state(), first[i].flags());
        }

        int32_t second_count = spill_count - first_count;
        for (int32_t i = 0; i < second_count; i++) {
            fn(spill_pool.base[i].slot_state(), spill_pool.base[i].flags());
        }
        return;
    } else {
        int32_t inline_count = std::min(fanin_count, CHIP_FANIN_INLINE_CAP);
        for (int32_t i = 0; i < inline_count; i++) {
            if (!fn(inline_edges[i].slot_state(), inline_edges[i].flags())) {
                return false;
            }
        }

        int32_t spill_count = fanin_count - inline_count;
        if (spill_count <= 0) {
            return true;
        }

        int32_t start_idx = spill_start % spill_pool.capacity;
        int32_t first_count = std::min(spill_count, spill_pool.capacity - start_idx);
        FaninSpillEntry *first = spill_pool.base + start_idx;
        for (int32_t i = 0; i < first_count; i++) {
            if (!fn(first[i].slot_state(), first[i].flags())) {
                return false;
            }
        }

        int32_t second_count = spill_count - first_count;
        for (int32_t i = 0; i < second_count; i++) {
            if (!fn(spill_pool.base[i].slot_state(), spill_pool.base[i].flags())) {
                return false;
            }
        }
        return true;
    }
}

template <typename Fn>
inline FaninForEachReturn<Fn> for_each_fanin_slot_state(const TaskPayload &payload, Fn &&fn) {
    return for_each_fanin_storage(
        payload.fanin_inline_edges, payload.fanin_actual_count, payload.fanin_spill_start, *payload.fanin_spill_pool,
        static_cast<Fn &&>(fn)
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
struct DepListPool {
    DepListEntry *base;         // Pool base address
    int32_t capacity;           // Total number of entries
    int32_t top;                // Linear next-allocation counter (starts from 1)
    int32_t tail;               // Linear first-alive counter (entries before this are dead)
    int32_t high_water;         // Peak concurrent usage (top - tail)
    int32_t last_reclaimed{0};  // last_task_alive at last successful reclamation

    // Error code pointer for fatal error reporting (→ sm_header->orch_error_code)
    std::atomic<int32_t> *error_code_ptr = nullptr;
    std::atomic<uint32_t> *reclaim_request_mask = nullptr;
    std::atomic<uint32_t> *reclaim_ack_mask = nullptr;
    uint8_t ring_id = 0;

    /**
     *
     * Initialize dependency list pool
     * @param base      Pool base address from shared memory
     * @param capacity  Total number of entries
     */
    void init(DepListEntry *in_base, int32_t in_capacity, std::atomic<int32_t> *in_error_code_ptr) {
        base = in_base;
        capacity = in_capacity;
        top = 1;   // Start from 1, 0 means NULL/empty
        tail = 1;  // Match initial top (no reclaimable entries yet)
        high_water = 0;
        last_reclaimed = 0;

        // Initialize entry 0 as NULL marker
        base[0].slot_state = nullptr;
        base[0].next = nullptr;

        error_code_ptr = in_error_code_ptr;
        reclaim_request_mask = nullptr;
        reclaim_ack_mask = nullptr;
        ring_id = 0;
    }

    void set_reclaim_publication_request(
        std::atomic<uint32_t> *request_mask, std::atomic<uint32_t> *ack_mask, uint8_t in_ring_id
    ) {
        reclaim_request_mask = request_mask;
        reclaim_ack_mask = ack_mask;
        ring_id = in_ring_id;
    }

    bool reclaim_publication_is_wired_to(
        const std::atomic<uint32_t> *request_mask, const std::atomic<uint32_t> *ack_mask, uint8_t expected_ring_id
    ) const {
        return reclaim_request_mask == request_mask && reclaim_ack_mask == ack_mask && ring_id == expected_ring_id;
    }

    void reset_for_reuse(std::atomic<int32_t> *in_error_code_ptr) {
        top = 1;
        tail = 1;
        high_water = 0;
        last_reclaimed = 0;
        base[0].slot_state = nullptr;
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
    void reclaim(SharedMemoryRingHeader &ring, int32_t sm_last_task_alive);

    /**
     * Ensure dep pool for a specific ring has at least `needed` entries available.
     * Spin-waits for reclamation under pressure. The dep pool shares
     * last_task_alive with the heap and task rings, so it detects a wedged
     * reclaim watermark the same way TaskAllocator::alloc does: request an
     * exact publication, then use a structural head-of-line check plus a
     * wall-clock backstop, each emitting report_deadlock.
     */
    bool ensure_space(SharedMemoryRingHeader &ring, int32_t needed, ChipTaskSlotState *oldest_open_task = nullptr);

    /**
     * Structured dep-pool deadlock report, mirroring TaskAllocator::report_deadlock.
     * scope_gated marks the provable head-of-line case where the head is pinned
     * by an open scope on this ring, as opposed to the wall-clock backstop.
     */
    void report_deadlock(SharedMemoryRingHeader &ring, int32_t needed, int32_t last_alive, bool scope_gated);

    /**
     * Allocate a single entry from the pool (single-thread per pool instance)
     *
     * @return Pointer to allocated entry, or nullptr on fatal error
     */
    DepListEntry *alloc() {
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
            LOG_ERROR("  Compile-time: CHIP_DEP_LIST_POOL_SIZE in runtime_types.h");
            LOG_ERROR("  Per task:     CallConfig.runtime_env.ring_dep_pool=%d", capacity * 2);
            LOG_ERROR("========================================");
            if (error_code_ptr) {
                error_code_ptr->store(SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED, std::memory_order_release);
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
    DepListEntry *prepend(DepListEntry *cur, ChipTaskSlotState *slot_state) {
        DepListEntry *new_entry = alloc();
        if (!new_entry) return nullptr;
        new_entry->slot_state = slot_state;
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
 * CHIP_MAX_RING_DEPTH instances provide independent reclamation per scope depth.
 */
struct ChipRingSet {
    TaskAllocator task_allocator;
    FaninPool fanin_pool;
};

#endif  // PTO_RING_BUFFER_H
