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
 * host_build_graph scheduler implementation
 *
 * Implements scheduler state management, ready queues, and task lifecycle.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "scheduler.h"
#include <inttypes.h>
#include <stdlib.h>
#include "common/unified_log.h"

// =============================================================================
// Scheduler Profiling Counters
// =============================================================================

#if SIMPLER_SCHED_PROFILING
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

SchedProfilingData scheduler_get_profiling(int thread_idx) {
    SchedProfilingData d;
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
// Debug Utilities
// =============================================================================

void SchedulerState::print_stats() {
    LOG_DEBUG("=== Scheduler Statistics ===");
#if SIMPLER_SCHED_PROFILING
    LOG_DEBUG("tasks_completed:   %lld", (long long)tasks_completed.load(std::memory_order_relaxed));
    LOG_DEBUG("tasks_consumed:    %lld", (long long)tasks_consumed.load(std::memory_order_relaxed));
#endif
    LOG_DEBUG("============================");
}

void SchedulerState::print_queues() {
    SchedulerState *sched = this;
    const char *shape_names[] = {"AIC", "AIV", "MIX"};
    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        LOG_DEBUG(
            "QPROBE rq[%s] pushes=%llu maxocc=%llu cap=%llu", shape_names[i],
            (unsigned long long)sched->ready_queues[i].enqueue_pos.load(std::memory_order_relaxed),
            (unsigned long long)sched->ready_queues[i].max_occupancy.load(std::memory_order_relaxed),
            (unsigned long long)sched->ready_queues[i].capacity
        );
        LOG_DEBUG(
            "QPROBE rsq[%s] pushes=%llu maxocc=%llu", shape_names[i],
            (unsigned long long)sched->ready_sync_queues[i].enqueue_pos.load(std::memory_order_relaxed),
            (unsigned long long)sched->ready_sync_queues[i].max_occupancy.load(std::memory_order_relaxed)
        );
        LOG_DEBUG(
            "QPROBE edq[%s] pushes=%llu maxocc=%llu cap=%llu", shape_names[i],
            (unsigned long long)sched->early_dispatch_queues[i].enqueue_pos.load(std::memory_order_relaxed),
            (unsigned long long)sched->early_dispatch_queues[i].max_occupancy.load(std::memory_order_relaxed),
            (unsigned long long)sched->early_dispatch_queues[i].capacity
        );
    }
    LOG_DEBUG(
        "QPROBE dummy pushes=%llu maxocc=%llu | graph_pq pushes=%llu maxocc=%llu | ess pushes=%llu maxocc=%llu",
        (unsigned long long)sched->dummy_ready_queue.enqueue_pos.load(std::memory_order_relaxed),
        (unsigned long long)sched->dummy_ready_queue.max_occupancy.load(std::memory_order_relaxed),
        (unsigned long long)sched->graph_prepare_queue.enqueue_pos.load(std::memory_order_relaxed),
        (unsigned long long)sched->graph_prepare_queue.max_occupancy.load(std::memory_order_relaxed),
        (unsigned long long)sched->early_sync_start_queue.enqueue_pos.load(std::memory_order_relaxed),
        (unsigned long long)sched->early_sync_start_queue.max_occupancy.load(std::memory_order_relaxed)
    );
    const uint64_t pub_drops = sched->ed_publish_drain_drops.load(std::memory_order_relaxed);
    LOG_DEBUG(
        "QPROBE ed_pubdrain pushes=%llu maxocc=%llu cap=%llu drops=%llu",
        (unsigned long long)sched->ed_publish_drain_queue.enqueue_pos.load(std::memory_order_relaxed),
        (unsigned long long)sched->ed_publish_drain_queue.max_occupancy.load(std::memory_order_relaxed),
        (unsigned long long)sched->ed_publish_drain_queue.capacity, (unsigned long long)pub_drops
    );
    if (pub_drops > 0) {
        LOG_WARN(
            "[EARLY_DISPATCH] publish-drain queue dropped %llu chain(s); those candidates fell back to normal "
            "dispatch",
            (unsigned long long)pub_drops
        );
    }
#if SIMPLER_SCHED_PROFILING
    LOG_TIMING(
        "QPROBE ed_publist registered=%llu immediate=%llu seals=%llu detached=%llu rescanned=%llu rehangs=%llu "
        "ready=%llu",
        (unsigned long long)sched->ed_publish_stats.registered.load(std::memory_order_relaxed),
        (unsigned long long)sched->ed_publish_stats.immediate_at_intake.load(std::memory_order_relaxed),
        (unsigned long long)sched->ed_publish_stats.publish_seals.load(std::memory_order_relaxed),
        (unsigned long long)sched->ed_publish_stats.chains_detached.load(std::memory_order_relaxed),
        (unsigned long long)sched->ed_publish_stats.waiters_rescanned.load(std::memory_order_relaxed),
        (unsigned long long)sched->ed_publish_stats.rehangs.load(std::memory_order_relaxed),
        (unsigned long long)sched->ed_publish_stats.candidates_ready.load(std::memory_order_relaxed)
    );
#endif
}
