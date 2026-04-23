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
 * PTO Runtime2 - Scheduler Implementation
 *
 * Implements scheduler state management, ready queues, and task lifecycle.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "pto_scheduler.h"
#include <inttypes.h>
#include <stdlib.h>
#include "common/unified_log.h"

// =============================================================================
// Scheduler Profiling Counters
// =============================================================================

#if PTO2_SCHED_PROFILING
#include "common/platform_config.h"

uint64_t g_sched_lock_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_consumed_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_push_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_wait_cycle[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_lock_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanout_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_fanin_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_self_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_pop_atomic_count[PLATFORM_MAX_AICPU_THREADS] = {};
uint64_t g_sched_complete_count[PLATFORM_MAX_AICPU_THREADS] = {};

PTO2SchedProfilingData pto2_scheduler_get_profiling(int thread_idx) {
    PTO2SchedProfilingData d;
    d.lock_cycle = std::exchange(g_sched_lock_cycle[thread_idx], 0);
    d.fanout_cycle = std::exchange(g_sched_fanout_cycle[thread_idx], 0);
    d.fanin_cycle = std::exchange(g_sched_fanin_cycle[thread_idx], 0);
    d.self_consumed_cycle = std::exchange(g_sched_self_consumed_cycle[thread_idx], 0);
    d.lock_wait_cycle = std::exchange(g_sched_lock_wait_cycle[thread_idx], 0);
    d.push_wait_cycle = std::exchange(g_sched_push_wait_cycle[thread_idx], 0);
    d.pop_wait_cycle = std::exchange(g_sched_pop_wait_cycle[thread_idx], 0);
    d.lock_atomic_count = std::exchange(g_sched_lock_atomic_count[thread_idx], 0);
    d.fanout_atomic_count = std::exchange(g_sched_fanout_atomic_count[thread_idx], 0);
    d.fanin_atomic_count = std::exchange(g_sched_fanin_atomic_count[thread_idx], 0);
    d.self_atomic_count = std::exchange(g_sched_self_atomic_count[thread_idx], 0);
    d.pop_atomic_count = std::exchange(g_sched_pop_atomic_count[thread_idx], 0);
    d.complete_count = std::exchange(g_sched_complete_count[thread_idx], 0);
    return d;
}
#endif

// =============================================================================
// Task State Names
// =============================================================================

const char *pto2_task_state_name(PTO2TaskState state) {
    switch (state) {
    case PTO2_TASK_PENDING:
        return "PENDING";
    case PTO2_TASK_READY:
        return "READY";
    case PTO2_TASK_RUNNING:
        return "RUNNING";
    case PTO2_TASK_COMPLETED:
        return "COMPLETED";
    case PTO2_TASK_CONSUMED:
        return "CONSUMED";
    default:
        return "UNKNOWN";
    }
}

// =============================================================================
// Ready Queue Implementation
// =============================================================================

bool pto2_ready_queue_init(PTO2ReadyQueue *queue, uint64_t capacity) {
    queue->slots = (PTO2ReadyQueueSlot *)malloc(capacity * sizeof(PTO2ReadyQueueSlot));
    if (!queue->slots) {
        return false;
    }

    queue->capacity = capacity;
    queue->mask = capacity - 1;
    queue->enqueue_pos.store(0, std::memory_order_relaxed);
    queue->dequeue_pos.store(0, std::memory_order_relaxed);

    for (uint64_t i = 0; i < capacity; i++) {
        queue->slots[i].sequence.store((int64_t)i, std::memory_order_relaxed);
        queue->slots[i].slot_state = nullptr;
    }

    return true;
}

void pto2_ready_queue_destroy(PTO2ReadyQueue *queue) {
    if (queue->slots) {
        free(queue->slots);
        queue->slots = NULL;
    }
}

// =============================================================================
// Scheduler Initialization
// =============================================================================

bool PTO2SchedulerState::RingSchedState::init(PTO2SharedMemoryHeader *sm_header, int32_t ring_id) {
    ring = &sm_header->rings[ring_id];
    last_task_alive = 0;
    advance_lock.store(0, std::memory_order_relaxed);

    // Initialize all per-task slot state fields.
    // bind() sets payload, task, ring_id — immutable after init, bound once
    // to their fixed shared-memory addresses.
    // reset_for_reuse() sets dynamic fields to reclaim defaults (fanout_count=1,
    // rest zero) so the first submit needs no reset.
    for (uint64_t i = 0; i < ring->task_window_size; i++) {
        ring->slot_states[i].bind(&ring->task_payloads[i], &ring->task_descriptors[i], static_cast<uint8_t>(ring_id));
        ring->slot_states[i].reset_for_reuse();
        ring->slot_states[i].fanin_count = 0;
        ring->slot_states[i].active_mask = 0;
    }

    return true;
}

void PTO2SchedulerState::RingSchedState::destroy() { ring = nullptr; }

bool pto2_scheduler_init(PTO2SchedulerState *sched, PTO2SharedMemoryHeader *sm_header, int32_t dep_pool_capacity) {
    sched->sm_header = sm_header;
#if PTO2_SCHED_PROFILING
    sched->tasks_completed.store(0, std::memory_order_relaxed);
    sched->tasks_consumed.store(0, std::memory_order_relaxed);
#endif

    // Initialize per-ring state
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (!sched->ring_sched_states[r].init(sm_header, r)) {
            for (int j = 0; j < r; j++) {
                sched->ring_sched_states[j].destroy();
            }
            return false;
        }
    }

    // Initialize ready queues (one per resource shape, global)
    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        if (!pto2_ready_queue_init(&sched->ready_queues[i], PTO2_READY_QUEUE_SIZE)) {
            // Cleanup on failure
            for (int j = 0; j < i; j++) {
                pto2_ready_queue_destroy(&sched->ready_queues[j]);
            }
            for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
                sched->ring_sched_states[r].destroy();
            }
            return false;
        }
    }

    // Initialize per-ring wiring queues and dep pools (exclusively managed by scheduler thread 0)
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        PTO2DepListEntry *dep_entries =
            reinterpret_cast<PTO2DepListEntry *>(calloc(dep_pool_capacity, sizeof(PTO2DepListEntry)));
        if (!dep_entries) {
            for (int j = 0; j < r; j++) {
                free(sched->ring_sched_states[j].dep_pool.base);
            }
            for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
                pto2_ready_queue_destroy(&sched->ready_queues[i]);
            }
            sched->wiring.queue.destroy();
            for (int rr = 0; rr < PTO2_MAX_RING_DEPTH; rr++) {
                sched->ring_sched_states[rr].destroy();
            }
            return false;
        }
        sched->ring_sched_states[r].dep_pool.init(dep_entries, dep_pool_capacity, &sm_header->orch_error_code);
    }

    // Initialize global wiring queue (SPSC: orchestrator pushes, scheduler thread 0 drains)
    if (!sched->wiring.queue.init(PTO2_WRIRING_QUEUE_SIZE)) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            free(sched->ring_sched_states[r].dep_pool.base);
        }
        for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
            pto2_ready_queue_destroy(&sched->ready_queues[i]);
        }
        for (int rr = 0; rr < PTO2_MAX_RING_DEPTH; rr++) {
            sched->ring_sched_states[rr].destroy();
        }
        return false;
    }
    sched->wiring.batch_count = 0;
    sched->wiring.batch_index = 0;
    sched->wiring.backoff_counter = 0;

    return true;
}

void pto2_scheduler_destroy(PTO2SchedulerState *sched) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        sched->ring_sched_states[r].destroy();
        free(sched->ring_sched_states[r].dep_pool.base);
        sched->ring_sched_states[r].dep_pool.base = nullptr;
    }

    sched->wiring.queue.destroy();

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        pto2_ready_queue_destroy(&sched->ready_queues[i]);
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

void pto2_scheduler_print_stats(PTO2SchedulerState *sched) {
    LOG_INFO("=== Scheduler Statistics ===");
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (sched->ring_sched_states[r].last_task_alive > 0) {
            LOG_INFO("Ring %d:", r);
            LOG_INFO("  last_task_alive: %d", sched->ring_sched_states[r].last_task_alive);
            auto &dp = sched->ring_sched_states[r].dep_pool;
            if (dp.top > 0) {
                LOG_INFO(
                    "  dep_pool: top=%d tail=%d used=%d high_water=%d capacity=%d", dp.top, dp.tail, dp.top - dp.tail,
                    dp.high_water, dp.capacity
                );
            }
        }
    }
#if PTO2_SCHED_PROFILING
    LOG_INFO("tasks_completed:   %lld", (long long)sched->tasks_completed.load(std::memory_order_relaxed));
    LOG_INFO("tasks_consumed:    %lld", (long long)sched->tasks_consumed.load(std::memory_order_relaxed));
#endif
    LOG_INFO("============================");
}

void pto2_scheduler_print_queues(PTO2SchedulerState *sched) {
    LOG_INFO("=== Ready Queues ===");

    const char *shape_names[] = {"AIC", "AIV", "MIX"};

    for (int i = 0; i < PTO2_NUM_RESOURCE_SHAPES; i++) {
        LOG_INFO("  %s: count=%" PRIu64, shape_names[i], sched->ready_queues[i].size());
    }

    LOG_INFO("====================");
}
