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
 * 1. HeapRing - Output buffer allocation from GM Heap
 *    - O(1) bump allocation
 *    - Wrap-around at end, skip to beginning if buffer doesn't fit
 *    - Implicit reclamation via heap_tail advancement
 *    - Back-pressure: stalls when no space available
 *
 * 2. TaskRing - Task slot allocation
 *    - Fixed window size (TASK_WINDOW_SIZE)
 *    - Wrap-around modulo window size
 *    - Implicit reclamation via last_task_alive advancement
 *    - Back-pressure: stalls when window is full
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

#include <inttypes.h>

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
// Heap ring spin limit - after this, report deadlock and exit
#define PTO2_HEAP_SPIN_LIMIT 100000

// Flow control spin limit - if exceeded, likely deadlock due to scope/fanout_count
#define PTO2_FLOW_CONTROL_SPIN_LIMIT 100000

// Dep pool spin limit - if exceeded, dep pool capacity too small for workload
#define PTO2_DEP_POOL_SPIN_LIMIT 100000

// =============================================================================
// Heap Ring Buffer
// =============================================================================

/**
 * Heap ring buffer structure
 *
 * Allocates output buffers from a contiguous GM Heap.
 * Wrap-around design with implicit reclamation.
 */
struct PTO2HeapRing {
    void *base;                      // GM_Heap_Base pointer
    uint64_t size;                   // GM_Heap_Size (total heap size in bytes)
    std::atomic<uint64_t> *top_ptr;  // Allocation pointer (shared atomic in SM header)

    // Reference to shared memory tail (for back-pressure)
    std::atomic<uint64_t> *tail_ptr;  // Points to header->heap_tail

    // Error code pointer for fatal error reporting (→ sm_header->orch_error_code)
    std::atomic<int32_t> *error_code_ptr = nullptr;

    /**
     * Allocate memory from heap ring
     *
     * O(1) bump allocation with wrap-around.
     * May STALL (spin-wait) if insufficient space (back-pressure).
     * Never splits a buffer across the wrap-around boundary.
     *
     * @param size  Requested size in bytes
     * @return Pointer to allocated memory, or nullptr on fatal error
     */
    void *pto2_heap_ring_alloc(uint64_t size) {
        // Align size for DMA efficiency
        size = PTO2_ALIGN_UP(size, PTO2_ALIGN_SIZE);

        // Spin-wait if insufficient space (back-pressure from Scheduler)
        int spin_count = 0;
        uint64_t prev_tail = tail_ptr->load(std::memory_order_acquire);
#if PTO2_SPIN_VERBOSE_LOGGING
        bool notified = false;
#endif
#if PTO2_ORCH_PROFILING
        uint64_t wait_start = 0;
        bool waiting = false;
#endif

        while (1) {
            void *ptr = pto2_heap_ring_try_alloc(size);
            if (ptr != NULL) {
#if PTO2_SPIN_VERBOSE_LOGGING
                if (notified) {
                    LOG_INFO("[HeapRing] Unblocked after %d spins", spin_count);
                }
#endif
#if PTO2_ORCH_PROFILING
                if (waiting) {
                    extern uint64_t g_orch_heap_wait_cycle;
                    g_orch_heap_wait_cycle += (get_sys_cnt_aicpu() - wait_start);
                }
                {
                    extern uint64_t g_orch_heap_atomic_count;
                    g_orch_heap_atomic_count +=
                        spin_count + 1;  // spin_count retries + 1 success (each try_alloc = 1 load)
                }
#endif
                return ptr;
            }

            // No space available, spin-wait
            spin_count++;
#if PTO2_ORCH_PROFILING
            if (!waiting) {
                wait_start = get_sys_cnt_aicpu();
                waiting = true;
            }
#endif

            // Progress detection: reset spin counter if heap_tail advances
            uint64_t cur_tail = tail_ptr->load(std::memory_order_acquire);
            if (cur_tail != prev_tail) {
#if PTO2_SPIN_VERBOSE_LOGGING
                LOG_INFO(
                    "[HeapRing] Progress: tail %" PRIu64 " -> %" PRIu64 " (reset spin_count=%d)", prev_tail, cur_tail,
                    spin_count
                );
#endif
                spin_count = 0;
                prev_tail = cur_tail;
            }

#if PTO2_SPIN_VERBOSE_LOGGING
            // Periodic block notification
            if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0 && spin_count > 0 && spin_count < PTO2_HEAP_SPIN_LIMIT) {
                uint64_t top = top_ptr->load(std::memory_order_acquire);
                LOG_WARN(
                    "[HeapRing] BLOCKED: requesting %" PRIu64 " bytes"
                    ", top=%" PRIu64 ", tail=%" PRIu64 ", spins=%d",
                    size, top, cur_tail, spin_count
                );
                notified = true;
            }
#endif

            if (spin_count >= PTO2_HEAP_SPIN_LIMIT) {
                uint64_t top = top_ptr->load(std::memory_order_acquire);
                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: Heap Ring Deadlock Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("Orchestrator blocked waiting for heap space after %d spins (no tail progress).", spin_count);
                LOG_ERROR("  - Requested:     %" PRIu64 " bytes", size);
                LOG_ERROR("  - Heap top:      %" PRIu64, top);
                LOG_ERROR("  - Heap tail:     %" PRIu64 " (stuck here)", cur_tail);
                LOG_ERROR("  - Heap size:     %" PRIu64, this->size);
                LOG_ERROR("  - Available:     %" PRIu64 " bytes", pto2_heap_ring_available());
                LOG_ERROR("Diagnosis:");
                LOG_ERROR("  heap_tail is not advancing, which means last_task_alive");
                LOG_ERROR("  is stuck. Check TaskRing diagnostics for root cause.");
                LOG_ERROR("Solution: Increase heap size or investigate task stall.");
                LOG_ERROR("  Compile-time: PTO2_HEAP_SIZE in pto_runtime2_types.h");
                LOG_ERROR(
                    "  Runtime env:  PTO2_RING_HEAP=<power-of-2 bytes> (e.g. %lu)", (unsigned long)(this->size * 2)
                );
                LOG_ERROR("========================================");
                if (error_code_ptr) {
                    error_code_ptr->store(PTO2_ERROR_HEAP_RING_DEADLOCK, std::memory_order_release);
                }
                return nullptr;
            }

            SPIN_WAIT_HINT();
        }
    }

    /**
     * Try to allocate memory without stalling (thread-safe via CAS)
     *
     * @param size  Requested size in bytes
     * @return Pointer to allocated memory, or NULL if no space
     */
    void *pto2_heap_ring_try_alloc(uint64_t alloc_size) {
        // Align size for DMA efficiency
        alloc_size = PTO2_ALIGN_UP(alloc_size, PTO2_ALIGN_SIZE);

        while (true) {
            uint64_t top = top_ptr->load(std::memory_order_acquire);
            // Read latest tail from shared memory (Scheduler updates this)
            uint64_t tail = tail_ptr->load(std::memory_order_acquire);
            uint64_t new_top;
            void *result;

            if (top >= tail) {
                // Case 1: top is at or ahead of tail (normal case)
                uint64_t space_at_end = size - top;

                if (space_at_end >= alloc_size) {
                    new_top = top + alloc_size;
                    result = (char *)base + top;
                } else if (tail > alloc_size) {
                    // Wrap to beginning
                    new_top = alloc_size;
                    result = base;
                } else {
                    return NULL;
                }
            } else {
                // Case 2: top has wrapped, tail is ahead
                uint64_t gap = tail - top;
                if (gap >= alloc_size) {
                    new_top = top + alloc_size;
                    result = (char *)base + top;
                } else {
                    return NULL;
                }
            }

            if (top_ptr->compare_exchange_weak(top, new_top, std::memory_order_acq_rel, std::memory_order_acquire)) {
                return result;
            }
            // CAS failed, retry with updated top
        }
    }

    /**
     * Get available space in heap ring
     */
    uint64_t pto2_heap_ring_available() {
        uint64_t top = top_ptr->load(std::memory_order_acquire);
        uint64_t tail = tail_ptr->load(std::memory_order_acquire);

        if (top >= tail) {
            uint64_t at_end = size - top;
            uint64_t at_begin = tail;
            return at_end > at_begin ? at_end : at_begin;
        } else {
            return tail - top;
        }
    }
};

/**
 * Initialize heap ring buffer
 *
 * @param ring      Heap ring to initialize
 * @param base      Base address of heap memory
 * @param size      Total heap size in bytes
 * @param tail_ptr  Pointer to shared memory heap_tail
 */
void pto2_heap_ring_init(
    PTO2HeapRing *ring, void *base, uint64_t size, std::atomic<uint64_t> *tail_ptr, std::atomic<uint64_t> *top_ptr
);

// =============================================================================
// Task Ring Buffer
// =============================================================================

/**
 * Task ring buffer structure
 *
 * Fixed-size sliding window for task management.
 * Provides back-pressure when window is full.
 */
struct PTO2TaskRing {
    PTO2TaskDescriptor *descriptors;          // Task descriptor array (from shared memory)
    int32_t window_size;                      // Window size (power of 2)
    std::atomic<int32_t> *current_index_ptr;  // Shared atomic in SM header

    // Reference to shared memory last_task_alive (for back-pressure)
    std::atomic<int32_t> *last_alive_ptr;  // Points to header->last_task_alive

    // Error code pointer for fatal error reporting (→ sm_header->orch_error_code)
    std::atomic<int32_t> *error_code_ptr = nullptr;

    /**
     * Allocate a task slot from task ring
     *
     * May STALL (spin-wait) if window is full (back-pressure).
     * Initializes the task descriptor to default values.
     *
     * @return Allocated task ID (absolute, not wrapped)
     */
    int32_t pto2_task_ring_alloc() {
        // Spin-wait if window is full (back-pressure from Scheduler)
        int spin_count = 0;
        int32_t prev_last_alive = last_alive_ptr->load(std::memory_order_acquire);
#if PTO2_SPIN_VERBOSE_LOGGING
        bool notified = false;
#endif
#if PTO2_ORCH_PROFILING
        uint64_t wait_start = 0;
        bool waiting = false;
#endif

        while (1) {
            int32_t task_id = pto2_task_ring_try_alloc();
            if (task_id >= 0) {
#if PTO2_SPIN_VERBOSE_LOGGING
                if (notified) {
                    LOG_INFO("[TaskRing] Unblocked after %d spins, task_id=%d", spin_count, task_id);
                }
#endif
#if PTO2_ORCH_PROFILING
                if (waiting) {
                    extern uint64_t g_orch_alloc_wait_cycle;
                    g_orch_alloc_wait_cycle += (get_sys_cnt_aicpu() - wait_start);
                }
                {
                    extern uint64_t g_orch_alloc_atomic_count;
                    g_orch_alloc_atomic_count +=
                        spin_count + 1;  // spin_count retries + 1 success (each try_alloc = 1 load)
                }
#endif
                return task_id;
            }

            // Window is full, spin-wait (with yield to prevent CPU starvation)
            spin_count++;
#if PTO2_ORCH_PROFILING
            if (!waiting) {
                wait_start = get_sys_cnt_aicpu();
                waiting = true;
            }
#endif

            // Progress detection: reset spin counter if last_task_alive advances
            int32_t cur_last_alive = last_alive_ptr->load(std::memory_order_acquire);
            if (cur_last_alive > prev_last_alive) {
#if PTO2_SPIN_VERBOSE_LOGGING
                LOG_INFO(
                    "[TaskRing] Progress: last_alive %d -> %d (reset spin_count=%d)", prev_last_alive, cur_last_alive,
                    spin_count
                );
#endif
                spin_count = 0;
                prev_last_alive = cur_last_alive;
            }

#if PTO2_SPIN_VERBOSE_LOGGING
            // Periodic block notification
            if (spin_count % PTO2_BLOCK_NOTIFY_INTERVAL == 0 && spin_count > 0 &&
                spin_count < PTO2_FLOW_CONTROL_SPIN_LIMIT) {
                int32_t current = current_index_ptr->load(std::memory_order_acquire);
                int32_t active_count = current - cur_last_alive;
                LOG_WARN(
                    "[TaskRing] BLOCKED (Flow Control): current=%d, last_alive=%d, "
                    "active=%d/%d (%.1f%%), spins=%d",
                    current, cur_last_alive, active_count, window_size, 100.0 * active_count / window_size, spin_count
                );
                notified = true;
            }
#endif

            // Deadlock: no progress after SPIN_LIMIT spins
            if (spin_count >= PTO2_FLOW_CONTROL_SPIN_LIMIT) {
                int32_t current = current_index_ptr->load(std::memory_order_acquire);
                int32_t active_count = current - cur_last_alive;

                LOG_ERROR("========================================");
                LOG_ERROR("FATAL: Flow Control Deadlock Detected!");
                LOG_ERROR("========================================");
                LOG_ERROR("Task Ring is FULL and no progress after %d spins.", spin_count);
                LOG_ERROR("  - Current task index:  %d", current);
                LOG_ERROR("  - Last task alive:     %d (stuck here)", cur_last_alive);
                LOG_ERROR("  - Active tasks:        %d / %d", active_count, window_size);
                LOG_ERROR("  - Window utilization:  %.1f%%", 100.0 * active_count / window_size);
                LOG_ERROR("Diagnosis:");
                LOG_ERROR("  last_task_alive is stuck at %d, meaning task %d", cur_last_alive, cur_last_alive);
                LOG_ERROR("  cannot transition to CONSUMED. Possible causes:");
                LOG_ERROR("  1. Task %d still executing (subtasks not complete)", cur_last_alive);
                LOG_ERROR("  2. Task %d fanout not fully released (downstream not done)", cur_last_alive);
                LOG_ERROR("  3. Scope reference not released (scope_end not called)");
                LOG_ERROR("  4. Orchestrator blocked here -> can't call scope_end -> circular wait");
                LOG_ERROR("Solution:");
                LOG_ERROR("  Increase task window size (current: %d, recommended: %d)", window_size, active_count * 2);
                LOG_ERROR("  Compile-time: PTO2_TASK_WINDOW_SIZE in pto_runtime2_types.h");
                LOG_ERROR("  Runtime env:  PTO2_RING_TASK_WINDOW=<power-of-2> (e.g. %d)", active_count * 2);
                LOG_ERROR("========================================");
                if (error_code_ptr) {
                    error_code_ptr->store(PTO2_ERROR_FLOW_CONTROL_DEADLOCK, std::memory_order_release);
                }
                return -1;
            }

            SPIN_WAIT_HINT();
        }
    }

    /**
     * Try to allocate task slot without stalling (thread-safe via fetch_add)
     *
     * @return Task ID, or -1 if window is full
     */
    int32_t pto2_task_ring_try_alloc() {
        // Optimistically allocate a task ID
        int32_t task_id = current_index_ptr->fetch_add(1, std::memory_order_acq_rel);
        int32_t last_alive = last_alive_ptr->load(std::memory_order_acquire);
        int32_t active_count = task_id - last_alive;

        // Check if there's room (leave at least 1 slot empty)
        if (active_count < window_size - 1) {
            return task_id;
        }

        // Window is full — roll back the optimistic increment
        current_index_ptr->fetch_sub(1, std::memory_order_release);
        return -1;
    }

    int32_t get_task_slot(int32_t task_id) const { return task_id & (window_size - 1); }

    /**
     * Get task descriptor by ID
     */
    PTO2TaskDescriptor &get_task(int32_t task_id) { return descriptors[task_id & (window_size - 1)]; }

    /**
     * Get task descriptor by task slot
     */
    PTO2TaskDescriptor &get_task_by_slot(int32_t task_slot) { return descriptors[task_slot]; }
};

/**
 * Initialize task ring buffer
 *
 * @param ring            Task ring to initialize
 * @param descriptors     Task descriptor array from shared memory
 * @param window_size     Window size (must be power of 2)
 * @param last_alive_ptr  Pointer to shared memory last_task_alive
 */
void pto2_task_ring_init(
    PTO2TaskRing *ring, PTO2TaskDescriptor *descriptors, int32_t window_size, std::atomic<int32_t> *last_alive_ptr,
    std::atomic<int32_t> *current_index_ptr
);

/**
 * Get number of active tasks in window
 */
static inline int32_t pto2_task_ring_active_count(PTO2TaskRing *ring) {
    int32_t last_alive = ring->last_alive_ptr->load(std::memory_order_acquire);
    return ring->current_index_ptr->load(std::memory_order_acquire) - last_alive;
}

/**
 * Check if task ring has space for more tasks
 */
static inline bool pto2_task_ring_has_space(PTO2TaskRing *ring) {
    int32_t active = pto2_task_ring_active_count(ring);
    return active < ring->window_size - 1;
}

/**
 * Get task descriptor by ID
 */
static inline PTO2TaskDescriptor *pto2_task_ring_get(PTO2TaskRing *ring, int32_t task_id) {
    return &ring->descriptors[task_id & (ring->window_size - 1)];
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
     * Initialize dependency list pool
     *
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

    int32_t used() const { return top - tail; }

    int32_t available() const { return capacity - used(); }
};

// =============================================================================
// Ring Set (per-depth aggregate)
// =============================================================================

/**
 * Groups a HeapRing, TaskRing, and DepPool into one per-depth unit.
 * PTO2_MAX_RING_DEPTH instances provide independent reclamation per scope depth.
 */
struct PTO2RingSet {
    PTO2HeapRing heap_ring;
    PTO2TaskRing task_ring;
    PTO2DepListPool dep_pool;
};

#endif  // PTO_RING_BUFFER_H
