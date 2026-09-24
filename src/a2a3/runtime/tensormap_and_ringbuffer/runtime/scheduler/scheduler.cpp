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
 * tensormap_and_ringbuffer scheduler implementation
 *
 * Implements scheduler state management, ready queues, and task lifecycle.
 *
 * Based on: docs/RUNTIME_LOGIC.md
 */

#include "scheduler.h"
#include <inttypes.h>
#include <stdlib.h>
#include "common/unified_log.h"

#if SIMPLER_DFX
// Weak fallbacks for host/UT builds that don't link the scope_stats collector.
extern "C" __attribute__((weak, visibility("hidden"))) bool is_scope_stats_enabled() { return false; }
extern "C" __attribute__((weak, visibility("hidden"))) void scope_stats_note_heap_wrap(int) {}
#endif

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
// Async wait diagnostics
// =============================================================================

void AsyncWaitList::log_diagnostics(AICoreCompletionMailbox *aicore_mailbox, const char *reason, bool warn_details) {
    struct PendingDetail {
        uint64_t task_id;
        uint64_t addr;
        uint64_t backend_cookie;
        uint32_t expected_value;
        uint32_t observed_value;
        int32_t entry_idx;
        int32_t condition_idx;
        int32_t waiting_completion_count;
        int32_t condition_count;
        int32_t completion_type;
        AsyncEngine engine;
        bool normal_done;
        bool retired;
        bool observed_available;
    };

    constexpr int32_t kMaxDetails = 16;
    constexpr int32_t kRunningReserve = kMaxDetails / 2;
    PendingDetail details[kMaxDetails];
    bool entry_selected[MAX_ASYNC_WAITS] = {};
    int32_t first_selected_condition[MAX_ASYNC_WAITS];
    for (int32_t i = 0; i < MAX_ASYNC_WAITS; i++)
        first_selected_condition[i] = -1;

    uint64_t mailbox_head = 0;
    uint64_t mailbox_tail = 0;
    auto sample_mailbox = [&]() {
        if (aicore_mailbox == nullptr) return;
        mailbox_head = aicore_mailbox->head.load(std::memory_order_acquire);
        mailbox_tail = aicore_mailbox->tail.load(std::memory_order_acquire);
    };
    auto mailbox_pending = [&]() {
        return mailbox_head >= mailbox_tail ? mailbox_head - mailbox_tail : 0;
    };

    constexpr int32_t kLockRetries = 1024;
    bool locked = false;
    for (int32_t i = 0; i < kLockRetries && !locked; i++) {
        locked = try_lock();
        if (!locked) SPIN_WAIT_HINT();
    }
    if (!locked) {
        sample_mailbox();
        LOG_WARN(
            "[ASYNC_WAIT reason=%s] status=skipped cause=list_busy retries=%d mailbox_head=%llu mailbox_tail=%llu "
            "mailbox_pending=%llu mpsc_skipped=%llu",
            reason, kLockRetries, static_cast<unsigned long long>(mailbox_head),
            static_cast<unsigned long long>(mailbox_tail), static_cast<unsigned long long>(mailbox_pending()),
            static_cast<unsigned long long>(mpsc_skipped_count.load(std::memory_order_relaxed))
        );
        return;
    }

    // Keep the mailbox counters and wait-list entries in the same diagnostic
    // window. Sampling before the consumer lock can combine two different
    // states and make head-tail appear to underflow.
    sample_mailbox();

    int32_t entry_count = count;
    if (entry_count < 0) entry_count = 0;
    if (entry_count > MAX_ASYNC_WAITS) entry_count = MAX_ASYNC_WAITS;

    int32_t task_body_pending_entries = 0;
    int32_t completion_pending_entries = 0;
    int32_t settled_entries = 0;
    int32_t pending_conditions = 0;
    int32_t detail_candidates = entry_count;
    for (int32_t entry_idx = 0; entry_idx < entry_count; entry_idx++) {
        const AsyncWaitEntry &entry = entries[entry_idx];
        int32_t waiting = entries[entry_idx].waiting_completion_count;
        bool task_body_pending = !entry.normal_done;
        bool completion_pending = waiting > 0;
        if (task_body_pending) task_body_pending_entries++;
        if (completion_pending) {
            completion_pending_entries++;
            pending_conditions += waiting;
        }
        if (!task_body_pending && !completion_pending) {
            settled_entries++;
        }

        int32_t condition_count = entry.condition_count;
        if (condition_count < 0) condition_count = 0;
        if (condition_count > MAX_COMPLETIONS_PER_TASK) condition_count = MAX_COMPLETIONS_PER_TASK;
        int32_t unsatisfied = 0;
        for (int32_t condition_idx = 0; condition_idx < condition_count; condition_idx++) {
            if (!entry.conditions[condition_idx].satisfied) unsatisfied++;
        }
        if (unsatisfied > 1) detail_candidates += unsatisfied - 1;
    }

    int32_t detail_count = 0;
    auto capture = [&](int32_t entry_idx, const CompletionCondition *cond, int32_t condition_idx) {
        if (detail_count >= kMaxDetails) return;
        const AsyncWaitEntry &entry = entries[entry_idx];
        PendingDetail &detail = details[detail_count++];
        detail.task_id = static_cast<uint64_t>(entry.task_token.raw);
        detail.addr = cond != nullptr ? cond->addr : 0;
        detail.backend_cookie = cond != nullptr ? cond->backend_cookie : 0;
        detail.expected_value = cond != nullptr ? cond->expected_value : 0;
        detail.observed_value = 0;
        detail.entry_idx = entry_idx;
        detail.condition_idx = condition_idx;
        detail.waiting_completion_count = entry.waiting_completion_count;
        detail.condition_count = entry.condition_count;
        detail.completion_type = cond != nullptr ? cond->completion_type : -1;
        detail.engine = cond != nullptr ? cond->engine : ASYNC_ENGINE_SDMA;
        detail.normal_done = entry.normal_done;
        detail.retired = cond != nullptr && cond->retired;
        detail.observed_available =
            cond != nullptr && cond->completion_type == COMPLETION_TYPE_COUNTER && cond->counter_addr != nullptr;
        if (detail.observed_available) {
            uintptr_t counter_line = mailbox_cache_line(cond->counter_addr);
            cache_invalidate_range(reinterpret_cast<const void *>(counter_line), sizeof(uint32_t));
            detail.observed_value = *cond->counter_addr;
        }
    };

    auto capture_entry = [&](int32_t entry_idx) {
        if (entry_selected[entry_idx] || detail_count >= kMaxDetails) return;
        const AsyncWaitEntry &entry = entries[entry_idx];
        int32_t condition_count = entry.condition_count;
        if (condition_count < 0) condition_count = 0;
        if (condition_count > MAX_COMPLETIONS_PER_TASK) condition_count = MAX_COMPLETIONS_PER_TASK;
        for (int32_t condition_idx = 0; condition_idx < condition_count; condition_idx++) {
            const CompletionCondition &cond = entry.conditions[condition_idx];
            if (cond.satisfied) continue;
            first_selected_condition[entry_idx] = condition_idx;
            entry_selected[entry_idx] = true;
            capture(entry_idx, &cond, condition_idx);
            return;
        }
        entry_selected[entry_idx] = true;
        capture(entry_idx, nullptr, -1);
    };

    // Keep both classes visible when the list is larger than the detail cap:
    // before-normal-done entries include running candidates, while
    // normal-done entries are blocked only by their completion conditions.
    for (int32_t entry_idx = 0; entry_idx < entry_count && detail_count < kRunningReserve; entry_idx++) {
        if (!entries[entry_idx].normal_done) capture_entry(entry_idx);
    }
    for (int32_t entry_idx = 0; entry_idx < entry_count && detail_count < kMaxDetails; entry_idx++) {
        if (entries[entry_idx].normal_done && entries[entry_idx].waiting_completion_count > 0) {
            capture_entry(entry_idx);
        }
    }
    for (int32_t entry_idx = 0; entry_idx < entry_count && detail_count < kMaxDetails; entry_idx++) {
        capture_entry(entry_idx);
    }

    // If fewer than 16 tasks are listed, use the remaining rows for additional
    // pending conditions belonging to those tasks.
    for (int32_t entry_idx = 0; entry_idx < entry_count && detail_count < kMaxDetails; entry_idx++) {
        const AsyncWaitEntry &entry = entries[entry_idx];
        int32_t condition_count = entry.condition_count;
        if (condition_count < 0) condition_count = 0;
        if (condition_count > MAX_COMPLETIONS_PER_TASK) condition_count = MAX_COMPLETIONS_PER_TASK;
        for (int32_t condition_idx = 0; condition_idx < condition_count && detail_count < kMaxDetails;
             condition_idx++) {
            const CompletionCondition &cond = entry.conditions[condition_idx];
            if (cond.satisfied || condition_idx == first_selected_condition[entry_idx]) continue;
            capture(entry_idx, &cond, condition_idx);
        }
    }

    uint64_t mpsc_skipped = mpsc_skipped_count.load(std::memory_order_relaxed);
    unlock();

    LOG_WARN(
        "[ASYNC_WAIT reason=%s] status=captured wait_count=%d task_body_pending_entries=%d "
        "completion_pending_entries=%d settled_entries=%d pending_conditions=%d details=%d truncated=%d "
        "mailbox_head=%llu mailbox_tail=%llu mailbox_pending=%llu mpsc_skipped=%llu",
        reason, entry_count, task_body_pending_entries, completion_pending_entries, settled_entries, pending_conditions,
        detail_count, detail_candidates > detail_count ? 1 : 0, static_cast<unsigned long long>(mailbox_head),
        static_cast<unsigned long long>(mailbox_tail), static_cast<unsigned long long>(mailbox_pending()),
        static_cast<unsigned long long>(mpsc_skipped)
    );

    for (int32_t i = 0; i < detail_count; i++) {
        const PendingDetail &detail = details[i];
        if (detail.observed_available) {
            if (warn_details) {
                LOG_WARN(
                    "[ASYNC_WAIT reason=%s entry=%d cond=%d] task_token=%llu normal_done=%u waiting=%d conditions=%d "
                    "engine=%s type=%d addr=0x%llx cookie=0x%llx expected=%u observed=%u retired=%u",
                    reason, detail.entry_idx, detail.condition_idx, static_cast<unsigned long long>(detail.task_id),
                    static_cast<unsigned>(detail.normal_done), detail.waiting_completion_count, detail.condition_count,
                    async_engine_name(detail.engine), detail.completion_type,
                    static_cast<unsigned long long>(detail.addr),
                    static_cast<unsigned long long>(detail.backend_cookie), detail.expected_value,
                    detail.observed_value, static_cast<unsigned>(detail.retired)
                );
            } else {
                LOG_INFO(
                    "[ASYNC_WAIT reason=%s entry=%d cond=%d] task_token=%llu normal_done=%u waiting=%d conditions=%d "
                    "engine=%s type=%d addr=0x%llx cookie=0x%llx expected=%u observed=%u retired=%u",
                    reason, detail.entry_idx, detail.condition_idx, static_cast<unsigned long long>(detail.task_id),
                    static_cast<unsigned>(detail.normal_done), detail.waiting_completion_count, detail.condition_count,
                    async_engine_name(detail.engine), detail.completion_type,
                    static_cast<unsigned long long>(detail.addr),
                    static_cast<unsigned long long>(detail.backend_cookie), detail.expected_value,
                    detail.observed_value, static_cast<unsigned>(detail.retired)
                );
            }
        } else {
            if (warn_details) {
                LOG_WARN(
                    "[ASYNC_WAIT reason=%s entry=%d cond=%d] task_token=%llu normal_done=%u waiting=%d conditions=%d "
                    "engine=%s type=%d addr=0x%llx cookie=0x%llx expected=%u observed=unavailable retired=%u",
                    reason, detail.entry_idx, detail.condition_idx, static_cast<unsigned long long>(detail.task_id),
                    static_cast<unsigned>(detail.normal_done), detail.waiting_completion_count, detail.condition_count,
                    detail.condition_idx >= 0 ? async_engine_name(detail.engine) : "none", detail.completion_type,
                    static_cast<unsigned long long>(detail.addr),
                    static_cast<unsigned long long>(detail.backend_cookie), detail.expected_value,
                    static_cast<unsigned>(detail.retired)
                );
            } else {
                LOG_INFO(
                    "[ASYNC_WAIT reason=%s entry=%d cond=%d] task_token=%llu normal_done=%u waiting=%d conditions=%d "
                    "engine=%s type=%d addr=0x%llx cookie=0x%llx expected=%u observed=unavailable retired=%u",
                    reason, detail.entry_idx, detail.condition_idx, static_cast<unsigned long long>(detail.task_id),
                    static_cast<unsigned>(detail.normal_done), detail.waiting_completion_count, detail.condition_count,
                    detail.condition_idx >= 0 ? async_engine_name(detail.engine) : "none", detail.completion_type,
                    static_cast<unsigned long long>(detail.addr),
                    static_cast<unsigned long long>(detail.backend_cookie), detail.expected_value,
                    static_cast<unsigned>(detail.retired)
                );
            }
        }
    }
}

// =============================================================================
// Debug Utilities
// =============================================================================

void SchedulerState::print_stats() {
    SchedulerState *sched = this;
    LOG_DEBUG("=== Scheduler Statistics ===");
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        if (sched->ring_sched_states[r].last_task_alive > 0) {
            LOG_DEBUG("Ring %d:", r);
            LOG_DEBUG("  last_task_alive: %d", sched->ring_sched_states[r].last_task_alive);
            auto &dp = sched->ring_sched_states[r].dep_pool;
            if (dp.top > 0) {
                LOG_DEBUG(
                    "  dep_pool: top=%d tail=%d used=%d high_water=%d capacity=%d", dp.top, dp.tail, dp.top - dp.tail,
                    dp.high_water, dp.capacity
                );
            }
        }
    }
#if SIMPLER_SCHED_PROFILING
    LOG_DEBUG("tasks_completed:   %lld", (long long)sched->tasks_completed.load(std::memory_order_relaxed));
    LOG_DEBUG("tasks_consumed:    %lld", (long long)sched->tasks_consumed.load(std::memory_order_relaxed));
#endif
    LOG_DEBUG("============================");
}

void SchedulerState::print_queues() {
    SchedulerState *sched = this;
    LOG_DEBUG("=== Ready Queues ===");

    const char *shape_names[] = {"AIC", "AIV", "MIX"};

    for (int i = 0; i < NUM_RESOURCE_SHAPES; i++) {
        LOG_DEBUG("  %s: count=%" PRIu64, shape_names[i], sched->ready_queues[i].size());
    }
    LOG_DEBUG("  DUMMY: count=%" PRIu64, sched->dummy_ready_queue.size());

    LOG_DEBUG("====================");
}
