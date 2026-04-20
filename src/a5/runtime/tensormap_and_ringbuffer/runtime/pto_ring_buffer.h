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

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "common/unified_log.h"

struct PTO2SchedulerState;  // Forward declaration for dep_pool reclaim

// Set to 1 to enable periodic BLOCKED/Unblocked messages during spin-wait.
#ifndef PTO2_SPIN_VERBOSE_LOGGING
#define PTO2_SPIN_VERBOSE_LOGGING 1
#endif

// Block notification interval (in spin counts)
#define PTO2_BLOCK_NOTIFY_INTERVAL 10000
// Alloc spin limit - after this, report deadlock and exit
#define PTO2_ALLOC_SPIN_LIMIT 100000

// Dep pool spin limit - if exceeded, dep pool capacity too small for workload
#define PTO2_DEP_POOL_SPIN_LIMIT 100000

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
     */
    void init(
        PTO2TaskDescriptor *descriptors, int32_t window_size, std::atomic<int32_t> *current_index_ptr,
        std::atomic<int32_t> *last_alive_ptr, void *heap_base, uint64_t heap_size, std::atomic<int32_t> *error_code_ptr
    ) {
        descriptors_ = descriptors;
        window_size_ = window_size;
        window_mask_ = window_size - 1;
        current_index_ptr_ = current_index_ptr;
        last_alive_ptr_ = last_alive_ptr;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        error_code_ptr_ = error_code_ptr;
        local_task_id_ = current_index_ptr->load(std::memory_order_relaxed);
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
#if PTO2_ORCH_PROFILING
        uint64_t wait_start = 0;
        bool waiting = false;
#endif

        while (true) {
            // Check both resources; commit only if both available
            if (local_task_id_ - last_alive + 1 < window_size_) {
                void *heap_ptr = try_bump_heap(aligned_size);
                if (heap_ptr) {
                    int32_t task_id = commit_task();
#if PTO2_ORCH_PROFILING
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
#if PTO2_ORCH_PROFILING
            if (!waiting) {
                wait_start = get_sys_cnt_aicpu();
                waiting = true;
            }
#endif
            last_alive = last_alive_ptr_->load(std::memory_order_acquire);
            update_heap_tail(last_alive);
            if (last_alive > prev_last_alive) {
                spin_count = 0;
                prev_last_alive = last_alive;
            } else {
#if PTO2_SPIN_VERBOSE_LOGGING
                if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0) {
                    LOG_WARN(
                        "[TaskAllocator] BLOCKED: tasks=%d/%d, heap=%" PRIu64 "/%" PRIu64 ", on=%s, spins=%d",
                        local_task_id_ - last_alive, window_size_, heap_top_, heap_size_,
                        blocked_on_heap ? "heap" : "task", spin_count
                    );
                }
#endif
                if (spin_count >= PTO2_ALLOC_SPIN_LIMIT) {
                    report_deadlock(output_size, blocked_on_heap);
                    return {-1, -1, nullptr, nullptr};
                }
            }
            SPIN_WAIT_HINT();
        }
    }

    // =========================================================================
    // Task descriptor accessors
    // =========================================================================

    PTO2TaskDescriptor &task(int32_t task_id) const { return descriptors_[task_id & window_mask_]; }

    PTO2TaskDescriptor &task_by_slot(int32_t slot) const { return descriptors_[slot]; }

    // =========================================================================
    // State queries
    // =========================================================================

    int32_t active_count() const {
        int32_t last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        return local_task_id_ - last_alive;
    }

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
    uint64_t heap_capacity() const { return heap_size_; }

private:
    // --- Task Ring ---
    PTO2TaskDescriptor *descriptors_ = nullptr;
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
        heap_tail_ =
            static_cast<uint64_t>(static_cast<char *>(desc.packed_buffer_end) - static_cast<char *>(heap_base_));
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

#if PTO2_ORCH_PROFILING
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
     * Report deadlock with targeted diagnostics.
     */
    void report_deadlock(int32_t requested_output_size, bool heap_blocked) {
        int32_t last_alive = last_alive_ptr_->load(std::memory_order_acquire);
        int32_t active_tasks = local_task_id_ - last_alive;
        uint64_t htail = heap_tail_;

        LOG_ERROR("========================================");
        if (heap_blocked) {
            LOG_ERROR("FATAL: Task Allocator Deadlock - Heap Exhausted!");
        } else {
            LOG_ERROR("FATAL: Task Allocator Deadlock - Task Ring Full!");
        }
        LOG_ERROR("========================================");
        LOG_ERROR("No progress after %d spins.", PTO2_ALLOC_SPIN_LIMIT);
        LOG_ERROR(
            "  Task ring:  current=%d, last_alive=%d, active=%d/%d (%.1f%%)", local_task_id_, last_alive, active_tasks,
            window_size_, 100.0 * active_tasks / window_size_
        );
        LOG_ERROR(
            "  Heap ring:  top=%" PRIu64 ", tail=%" PRIu64 ", size=%" PRIu64 ", available=%" PRIu64, heap_top_, htail,
            heap_size_, heap_available()
        );
        if (heap_blocked) {
            LOG_ERROR("  Requested:  %d bytes", requested_output_size);
        }
        LOG_ERROR("Diagnosis:");
        LOG_ERROR("  last_task_alive is stuck at %d, meaning task %d", last_alive, last_alive);
        LOG_ERROR("  cannot transition to CONSUMED. Possible causes:");
        LOG_ERROR("  1. Task %d still executing (subtasks not complete)", last_alive);
        LOG_ERROR("  2. Task %d fanout not fully released (downstream not done)", last_alive);
        LOG_ERROR("  3. Scope reference not released (scope_end not called)");
        LOG_ERROR("  4. Orchestrator blocked here -> can't call scope_end -> circular wait");
        LOG_ERROR("Solution:");
        if (heap_blocked) {
            LOG_ERROR(
                "  Increase heap size (current: %" PRIu64 ", recommended: %" PRIu64 ")", heap_size_, heap_size_ * 2
            );
            LOG_ERROR("  Compile-time: PTO2_HEAP_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Runtime env:  PTO2_RING_HEAP=<power-of-2 bytes> (e.g. %" PRIu64 ")", heap_size_ * 2);
        } else {
            LOG_ERROR("  Increase task window size (current: %d, recommended: %d)", window_size_, active_tasks * 2);
            LOG_ERROR("  Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
            LOG_ERROR("  Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2> (e.g. %d)", active_tasks * 2);
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
        base[0].slot_state = nullptr;
        error_code_ptr = in_error_code_ptr;
    }

    void reclaim(PTO2SchedulerState &sched, uint8_t ring_id, int32_t sm_last_task_alive);

    void ensure_space(PTO2SchedulerState &sched, PTO2RingFlowControl &fc, uint8_t ring_id, int32_t needed);

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

template <typename Fn>
using PTO2FaninCallbackResult = std::invoke_result_t<Fn &, PTO2TaskSlotState *>;

template <typename Fn>
using PTO2FaninForEachReturn = std::conditional_t<std::is_same_v<PTO2FaninCallbackResult<Fn>, void>, void, bool>;

template <typename InlineSlots, typename Fn>
inline PTO2FaninForEachReturn<Fn> pto2_for_each_fanin_storage(
    InlineSlots &&inline_slot_states, int32_t fanin_count, int32_t spill_start, PTO2FaninPool *spill_pool, Fn &&fn
) {
    using FaninCallbackResult = PTO2FaninCallbackResult<Fn>;
    static_assert(
        std::is_same_v<FaninCallbackResult, void> || std::is_same_v<FaninCallbackResult, bool>,
        "fanin callback must return void or bool"
    );

    if constexpr (std::is_void_v<FaninCallbackResult>) {
        int32_t inline_count = std::min(fanin_count, PTO2_FANIN_INLINE_CAP);
        for (int32_t i = 0; i < inline_count; i++) {
            fn(inline_slot_states[i]);
        }

        int32_t spill_count = fanin_count - inline_count;
        if (spill_count <= 0) {
            return;
        }

        PTO2FaninPool *pool = spill_pool;
        int32_t start_idx = spill_start % pool->capacity;
        int32_t first_count = std::min(spill_count, pool->capacity - start_idx);
        PTO2FaninSpillEntry *first = pool->base + start_idx;
        for (int32_t i = 0; i < first_count; i++) {
            fn(first[i].slot_state);
        }

        int32_t second_count = spill_count - first_count;
        for (int32_t i = 0; i < second_count; i++) {
            fn(pool->base[i].slot_state);
        }
        return;
    } else {
        int32_t inline_count = std::min(fanin_count, PTO2_FANIN_INLINE_CAP);
        for (int32_t i = 0; i < inline_count; i++) {
            if (!fn(inline_slot_states[i])) {
                return false;
            }
        }

        int32_t spill_count = fanin_count - inline_count;
        if (spill_count <= 0) {
            return true;
        }

        PTO2FaninPool *pool = spill_pool;
        int32_t start_idx = spill_start % pool->capacity;
        int32_t first_count = std::min(spill_count, pool->capacity - start_idx);
        PTO2FaninSpillEntry *first = pool->base + start_idx;
        for (int32_t i = 0; i < first_count; i++) {
            if (!fn(first[i].slot_state)) {
                return false;
            }
        }

        int32_t second_count = spill_count - first_count;
        for (int32_t i = 0; i < second_count; i++) {
            if (!fn(pool->base[i].slot_state)) {
                return false;
            }
        }
        return true;
    }
}

template <typename Fn>
inline PTO2FaninForEachReturn<Fn> pto2_for_each_fanin_slot_state(const PTO2TaskPayload &payload, Fn &&fn) {
    return pto2_for_each_fanin_storage(
        payload.fanin_inline_slot_states, payload.fanin_actual_count, payload.fanin_spill_start,
        payload.fanin_spill_pool, static_cast<Fn &&>(fn)
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
        base[0].slot_state = nullptr;
        base[0].next = nullptr;

        error_code_ptr = in_error_code_ptr;
    }

    /**
     * Reclaim dead entries based on scheduler's slot state dep_pool_mark.
     * Safe to call multiple times — only advances tail forward.
     *
     * @param sched              Scheduler state (for reading slot dep_pool_mark)
     * @param ring_id            Ring layer index
     * @param sm_last_task_alive Current last_task_alive from shared memory
     */
    void reclaim(PTO2SchedulerState &sched, uint8_t ring_id, int32_t sm_last_task_alive);

    /**
     * Ensure dep pool for a specific ring has at least `needed` entries available.
     * Spin-waits for reclamation if under pressure. Detects deadlock if no progress.
     */
    void ensure_space(PTO2SchedulerState &sched, PTO2RingFlowControl &fc, uint8_t ring_id, int32_t needed);

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
    PTO2DepListEntry *prepend(PTO2DepListEntry *cur, PTO2TaskSlotState *slot_state) {
        PTO2DepListEntry *new_entry = alloc();
        if (!new_entry) return nullptr;
        new_entry->slot_state = slot_state;
        new_entry->next = cur;
        return new_entry;
    }

    /**
     * Get entry by offset
     */
    PTO2DepListEntry *pto2_dep_pool_get(int32_t offset) {
        if (offset <= 0) return NULL;
        return &base[offset];
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
