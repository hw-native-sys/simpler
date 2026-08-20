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
 * host_build_graph task and heap allocation.
 *
 * PTO2TaskAllocator combines task-slot and output-buffer allocation. Both are
 * forward-only bump allocators because the whole graph remains resident for
 * the run.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#pragma once

#include <algorithm>
#include <inttypes.h>
#include <type_traits>

#include "pto_runtime2_types.h"
#include "pto_shared_memory.h"
#include "common/unified_log.h"

// Base of the address range Graph recording hands to an internal node's packed
// outputs. Recorded addresses are never dereferenced: they exist so
// graph_classify_tensor can tell an internal producer's output from a boundary
// tensor by address-range containment alone, and the Definition stores them as
// offsets. That classification is only sound while the range is disjoint from
// every real heap address, which PTO2TaskAllocator::init() asserts.
inline constexpr uint64_t GRAPH_RECORD_VIRTUAL_BASE = 1ULL << 63;

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
 *
 * host_build_graph is whole-graph-resident: the device runs only after the host
 * has built the entire graph, so no task slot or heap byte is ever reclaimed
 * while allocation is in progress. Both allocators are therefore forward-only, and a
 * request that does not fit can never become satisfiable by waiting — alloc()
 * reports the exhausted resource and fails on the spot.
 */
class PTO2TaskAllocator {
public:
    /**
     * Initialize the allocator with task-capacity and heap resources.
     *
     * All pointer arguments are device addresses (live in SM / GM heap); this
     * function only stores them, no dereferences, so it is safe to invoke
     * from host code that constructs a prebuilt arena image.
     *
     * The ring starts at task id 0, matching the SM flow-control counter that
     * current_index_ptr points at (PTO2RingFlowControl::init() runs on the AICPU
     * during SM reset), so local_task_id_ stays in sync without reading the SM.
     * Because ids are never reclaimed, alloc() caps them at task_capacity — they
     * cannot run away toward INT32_MAX.
     */
    void init(
        int32_t task_capacity, std::atomic<int32_t> *current_index_ptr, void *heap_base, uint64_t heap_size,
        std::atomic<int32_t> *error_code_ptr
    ) {
        task_capacity_ = task_capacity;
        task_capacity_mask_ = task_capacity - 1;
        current_index_ptr_ = current_index_ptr;
        heap_base_ = heap_base;
        heap_size_ = heap_size;
        error_code_ptr_ = error_code_ptr;
        local_task_id_ = 0;
        heap_top_ = 0;
        // Every address this allocator hands out lies in
        // [heap_base_, heap_base_ + heap_size_), so checking the range once here
        // keeps it disjoint from GRAPH_RECORD_VIRTUAL_BASE for every allocation.
        const uint64_t heap_base_addr = reinterpret_cast<uint64_t>(heap_base);
        always_assert(
            heap_base_addr < GRAPH_RECORD_VIRTUAL_BASE && heap_size <= GRAPH_RECORD_VIRTUAL_BASE - heap_base_addr &&
            "Graph heap overlaps the Graph-recording virtual address range"
        );
    }

    /**
     * Allocate a task slot and its associated output buffer in one call.
     *
     * Both task index and heap top are maintained as local counters and
     * published to shared memory only on success. Since the orchestrator is
     * single-threaded, no CAS or fetch_add is needed — just check-then-commit.
     *
     * A fatal latched elsewhere short-circuits the allocation: the caller maps
     * the failed result to orch_mark_fatal without overwriting the first code.
     *
     * @param output_size  Total packed output size in bytes (0 = no heap needed)
     * @return Allocation result; check failed() for errors
     */
    PTO2TaskAllocResult alloc(int32_t output_size) {
        uint64_t aligned_size =
            output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;

        if (error_code_ptr_ != nullptr && error_code_ptr_->load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
            return {-1, -1, nullptr, nullptr};
        }

        // Check both resources; commit only if both are available.
        if (local_task_id_ >= task_capacity_) {
            report_capacity_exhausted(/*heap_blocked=*/false, aligned_size);
            return {-1, -1, nullptr, nullptr};
        }
        void *heap_ptr = try_bump_heap(aligned_size);
        if (heap_ptr == nullptr) {
            report_capacity_exhausted(/*heap_blocked=*/true, aligned_size);
            return {-1, -1, nullptr, nullptr};
        }
        int32_t task_id = commit_task();
        return {task_id, task_id & task_capacity_mask_, heap_ptr, static_cast<char *>(heap_ptr) + aligned_size};
    }

    bool reserve_deferred_heap(int32_t output_size, void **packed_base, void **packed_end) {
        if (output_size < 0 || packed_base == nullptr || packed_end == nullptr) return false;
        if (error_code_ptr_ != nullptr && error_code_ptr_->load(std::memory_order_acquire) != PTO2_ERROR_NONE) {
            return false;
        }
        const uint64_t aligned_size =
            output_size > 0 ? PTO2_ALIGN_UP(static_cast<uint64_t>(output_size), PTO2_ALIGN_SIZE) : 0;
        void *base = try_bump_heap(aligned_size);
        if (base == nullptr) return false;
        *packed_base = base;
        *packed_end = static_cast<char *>(base) + aligned_size;
        return true;
    }

    // =========================================================================
    // State queries
    // =========================================================================

    // Nothing retires during a run, so every task allocated so far is still
    // live and the ring's head doubles as its occupancy.
    int32_t active_count() const { return local_task_id_; }

    int32_t task_head() const { return local_task_id_; }

    int32_t task_capacity() const { return task_capacity_; }

    uint64_t heap_available() const { return heap_size_ - heap_top_; }

    uint64_t heap_top() const { return heap_top_; }
    uint64_t heap_capacity() const { return heap_size_; }
    uint64_t heap_used_bytes() const { return heap_top_; }

private:
    // --- Task capacity ---
    int32_t task_capacity_ = 0;
    int32_t task_capacity_mask_ = 0;
    std::atomic<int32_t> *current_index_ptr_ = nullptr;

    // --- Heap ---
    void *heap_base_ = nullptr;
    uint64_t heap_size_ = 0;

    // --- Local state (single-writer, no atomics needed) ---
    int32_t local_task_id_ = 0;  // Next task ID to allocate
    uint64_t heap_top_ = 0;      // Current heap allocation pointer

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
     * The graph does not fit the configured ring. Nothing is reclaimed during a
     * run, so this is a sizing problem with an immediate verdict, not a wait
     * that could still succeed.
     */
    void report_capacity_exhausted(bool heap_blocked, uint64_t requested_bytes) {
        LOG_ERROR("========================================");
        if (heap_blocked) {
            LOG_ERROR("FATAL: Graph Heap Exhausted!");
        } else {
            LOG_ERROR("FATAL: Task Capacity Exhausted!");
        }
        LOG_ERROR("========================================");
        LOG_ERROR("The whole graph must fit the configured capacities; nothing is reclaimed mid-run.");
        LOG_ERROR("  Task capacity: used=%d/%d", local_task_id_, task_capacity_);
        LOG_ERROR(
            "  Graph heap:  used=%" PRIu64 "/%" PRIu64 ", available=%" PRIu64, heap_top_, heap_size_, heap_available()
        );
        LOG_ERROR("  Requested:   %" PRIu64 " bytes + 1 task slot", requested_bytes);
        LOG_ERROR("Solution:");
        if (heap_blocked) {
            LOG_ERROR(
                "  Increase heap (current: %" PRIu64 "); env PTO2_RING_HEAP=<pow2> (e.g. %" PRIu64 ")", heap_size_,
                heap_size_ * 2
            );
        } else {
            LOG_ERROR(
                "  Increase task capacity (current: %d); env PTO2_RING_TASK_WINDOW=<pow2> (e.g. %d)", task_capacity_,
                task_capacity_ * 2
            );
        }
        LOG_ERROR("========================================");
        if (error_code_ptr_) {
            int32_t code = heap_blocked ? PTO2_ERROR_HEAP_RING_DEADLOCK : PTO2_ERROR_FLOW_CONTROL_DEADLOCK;
            error_code_ptr_->store(code, std::memory_order_release);
        }
    }
};
