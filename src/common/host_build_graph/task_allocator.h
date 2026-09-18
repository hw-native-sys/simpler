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
 * TaskAllocator - Unified task slot + output buffer allocation
 *   - Combines task-table slot allocation and heap output-buffer allocation
 *   - O(1) forward bump allocation for both
 *   - Neither resource is reclaimed during a run, so exhaustion of either is a
 *     capacity error reported on the spot, never back-pressure to wait on
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <atomic>
#include <inttypes.h>

#include "host_build_graph/runtime_status.h"
#include "host_build_graph/runtime_types.h"
#include "common/unified_log.h"

// =============================================================================
// Task Allocator (unified task slot + heap buffer allocation)
// =============================================================================

/**
 * Unified task slot + heap buffer allocator.
 *
 * Task ids and heap bytes are handed out by two local bump counters. The
 * orchestrator is single-threaded and nothing outside it observes either
 * counter, so neither needs an atomic or a published copy: the run's task total
 * is read back through active_count() once orchestration ends.
 *
 * The alloc() method checks both resources BEFORE committing to either,
 * eliminating the need for rollback on partial failure.
 *
 * host_build_graph is whole-graph-resident: the device runs only after the host
 * has built the entire graph, so no task slot or heap byte is ever reclaimed
 * while allocation is in progress. Both counters are therefore forward-only, and
 * a request that does not fit can never become satisfiable by waiting — alloc()
 * reports the exhausted resource and fails on the spot.
 *
 * A task id is also its slot index: ids are capped at `capacity` and never
 * recycled, so the task table is a flat array indexed by id, with no wrap.
 */
class TaskAllocator {
public:
    /**
     * Initialize the allocator with its task capacity and heap resources.
     *
     * `heap_base` is a device address (the GM heap); this function only stores it,
     * no dereferences, so it is safe to invoke from host code that constructs a
     * prebuilt arena image. `error_code_ptr` is the host-side orchestrator's own
     * fatal_code, dereferenced only from the host as the allocator runs.
     *
     * `capacity` is the number of task slots the caller's task table holds — what
     * the bind resolved from runtime_env.ring_task_window, defaulting to
     * CHIP_DEFAULT_GRAPH_TASKS. It need not be a power of two: a task id indexes
     * its slot directly, so no slot lookup masks with it. It is bounded above by
     * TaskId::GLOBAL_TASK_MAX_NUM, because a sub-task's id carries its modular
     * task's local id in a fixed-width field — resolve_graph_task_capacity is
     * where that bound is enforced. Because ids are never reclaimed, alloc() caps
     * them at `capacity` — they cannot run away toward INT32_MAX.
     */
    void init(int32_t capacity, void *heap_base, uint64_t heap_size, std::atomic<int32_t> *error_code_ptr) {
        capacity_ = capacity;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        error_code_ptr_ = error_code_ptr;
        local_task_id_ = 0;
        heap_top_ = 0;
    }

    /**
     * Allocate a task slot and its associated output buffer in one call.
     *
     * Both the task id and the heap top are local counters, so this is a plain
     * check-then-commit with no atomic and no rollback.
     *
     * A fatal latched elsewhere short-circuits the allocation: the caller maps
     * the failed result to orch_mark_fatal without overwriting the first code.
     *
     * @param output_size  Total packed output size in bytes (0 = no heap needed)
     * @return Allocation result; check failed() for errors
     */
    TaskAllocResult alloc(int32_t output_size) {
        uint64_t aligned_size =
            output_size > 0 ? CHIP_ALIGN_UP(static_cast<uint64_t>(output_size), CHIP_ALIGN_SIZE) : 0;

        if (error_code_ptr_ != nullptr && error_code_ptr_->load(std::memory_order_acquire) != SIMPLER_ERROR_NONE) {
            return {-1, nullptr, nullptr};
        }

        // Check both resources; commit only if both are available.
        if (local_task_id_ >= capacity_) {
            report_capacity_exhausted(/*heap_blocked=*/false, aligned_size);
            return {-1, nullptr, nullptr};
        }
        void *heap_ptr = try_bump_heap(aligned_size);
        if (heap_ptr == nullptr) {
            report_capacity_exhausted(/*heap_blocked=*/true, aligned_size);
            return {-1, nullptr, nullptr};
        }
        int32_t task_id = local_task_id_++;
        return {task_id, heap_ptr, static_cast<char *>(heap_ptr) + aligned_size};
    }

    bool reserve_deferred_heap(int32_t output_size, void **packed_base, void **packed_end) {
        if (output_size < 0 || packed_base == nullptr || packed_end == nullptr) return false;
        if (error_code_ptr_ != nullptr && error_code_ptr_->load(std::memory_order_acquire) != SIMPLER_ERROR_NONE) {
            return false;
        }
        const uint64_t aligned_size =
            output_size > 0 ? CHIP_ALIGN_UP(static_cast<uint64_t>(output_size), CHIP_ALIGN_SIZE) : 0;
        void *base = try_bump_heap(aligned_size);
        if (base == nullptr) return false;
        *packed_base = base;
        *packed_end = static_cast<char *>(base) + aligned_size;
        return true;
    }

    // =========================================================================
    // State queries
    // =========================================================================

    // Nothing retires during a run, so every task allocated so far is still live
    // and the next id doubles as the occupancy. Once orchestration has ended this
    // is therefore also the run's task total, and the only place it is read from:
    // no copy of the count is published anywhere else.
    int32_t active_count() const { return local_task_id_; }

    int32_t capacity() const { return capacity_; }

    uint64_t heap_available() const { return heap_size_ - heap_top_; }

    uint64_t heap_top() const { return heap_top_; }
    uint64_t heap_capacity() const { return heap_size_; }
    uint64_t heap_used_bytes() const { return heap_top_; }

private:
    // --- Task table ---
    int32_t capacity_ = 0;

    // --- Heap ---
    void *heap_base_ = nullptr;
    uint64_t heap_size_ = 0;

    // --- Local state (single-writer, no atomics needed) ---
    int32_t local_task_id_ = 0;  // Next task ID to allocate
    uint64_t heap_top_ = 0;      // Current heap allocation pointer

    // --- Shared ---
    // The orchestrator's own fatal_code. Atomic for the same reason it is there: the
    // bind thread and a Graph recording worker both latch into it.
    std::atomic<int32_t> *error_code_ptr_ = nullptr;

    // =========================================================================
    // Internal helpers
    // =========================================================================

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
        if (heap_size_ - top < alloc_size) {
            LOG_DEBUG(
                "try_bump_heap failed: top=%" PRIu64 ", alloc=%" PRIu64 ", heap_size=%" PRIu64, top, alloc_size,
                heap_size_
            );
            return nullptr;
        }
        heap_top_ = top + alloc_size;
        return static_cast<char *>(heap_base_) + top;
    }

    /**
     * Report the exhausted resource and latch its error code.
     *
     * Nothing is reclaimed during a run, so this is a sizing verdict with no wait
     * that could still succeed.
     *
     * The task capacity is a configured size, so its branch is the one a real bind
     * reaches. The heap is not configured: a bind hands
     * this allocator the whole MAX_HEAP_CAPACITY span and commits the device
     * region afterwards, so a graph that does not fit the device fails at that
     * commit and not here. The heap branch stays because the allocator is also
     * constructed directly, against a small heap, by the unit tests that cover
     * this report.
     */
    void report_capacity_exhausted(bool heap_blocked, uint64_t requested_bytes) {
        LOG_ERROR("========================================");
        if (heap_blocked) {
            LOG_ERROR("FATAL: Graph Heap Exhausted!");
        } else {
            LOG_ERROR("FATAL: Graph Too Large!");
        }
        LOG_ERROR("========================================");
        LOG_ERROR("The whole graph must fit at once; nothing is reclaimed mid-run.");
        LOG_ERROR("  Tasks:      used=%d/%d", local_task_id_, capacity_);
        LOG_ERROR(
            "  Graph heap: used=%" PRIu64 "/%" PRIu64 ", available=%" PRIu64, heap_top_, heap_size_, heap_available()
        );
        LOG_ERROR("  Requested:  %" PRIu64 " bytes + 1 task slot", requested_bytes);
        LOG_ERROR("Solution:");
        if (heap_blocked) {
            LOG_ERROR("  Shrink the graph's intermediate tensors; this heap has no configuration knob");
        } else {
            LOG_ERROR(
                "  Raise the task capacity (current: %d) via CallConfig.runtime_env.ring_task_window, or shrink "
                "the graph",
                capacity_
            );
        }
        LOG_ERROR("========================================");
        // First-writer-wins, matching orch_mark_fatal, which latches the same field.
        // alloc() already declines once a code is latched, so in practice this is the
        // first writer -- but the rule is stated here rather than inherited from that
        // guard, so the two writers cannot drift apart.
        if (error_code_ptr_ != nullptr) {
            const int32_t code = heap_blocked ? SIMPLER_ERROR_HEAP_RING_DEADLOCK : SIMPLER_ERROR_FLOW_CONTROL_DEADLOCK;
            int32_t expected = SIMPLER_ERROR_NONE;
            error_code_ptr_->compare_exchange_strong(expected, code, std::memory_order_acq_rel);
        }
    }
};
