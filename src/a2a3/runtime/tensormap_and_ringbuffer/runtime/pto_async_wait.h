/**
 * PTO Runtime2 - Async Completion Wait List
 *
 * Lightweight watch-list abstraction for deferred task completion.
 *
 * The scheduler polls two logical protocols described in docs/runtime_async.md:
 *   - CQ protocol: poll *counter_addr >= expected_value (unified COUNTER type)
 *   - Notification protocol: poll a GM counter until it reaches expected_value
 *
 * All completion conditions use a single COUNTER type. Hardware event flags
 * (e.g. SDMA completion flags) are the special case where expected_value = 1.
 *
 * The scheduler polls this list each iteration (Phase 0) and triggers
 * on_mixed_task_complete for tasks whose conditions are all satisfied.
 *
 * Design reference: docs/runtime_async.md
 */

#ifndef PTO_ASYNC_WAIT_H
#define PTO_ASYNC_WAIT_H

#include <cstdint>
#include "pto_runtime2_types.h"
#include "pto_scheduler.h"

extern void cache_invalidate_range(const void* addr, size_t size);

inline constexpr int32_t PTO2_MAX_ASYNC_WAITS = 64;

enum class PTO2CompletionPollState : uint8_t {
    PENDING = 0,
    READY = 1,
    FAILED = 2,
};

struct PTO2CompletionPollResult {
    PTO2CompletionPollState state{PTO2CompletionPollState::PENDING};
    int32_t error_code{PTO2_ERROR_NONE};
};

struct PTO2CompletionCondition {
    PTO2AsyncEngine engine{PTO2_ASYNC_ENGINE_SDMA};
    bool satisfied{false};
    volatile uint32_t* counter_addr{nullptr};
    uint32_t expected_value{0};

    PTO2CompletionPollResult test() const {
        if (satisfied) {
            return {PTO2CompletionPollState::READY, PTO2_ERROR_NONE};
        }
        if (counter_addr == nullptr) {
            return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
        }
        return {*counter_addr >= expected_value ? PTO2CompletionPollState::READY
                                               : PTO2CompletionPollState::PENDING,
                PTO2_ERROR_NONE};
    }
};

template <bool Profiling>
#if PTO2_SCHED_PROFILING
static inline PTO2CompletionStats pto2_complete_task(
#else
static inline void pto2_complete_task(
#endif
        PTO2SchedulerState* sched,
        PTO2TaskSlotState& slot_state,
        PTO2LocalReadyBuffer* local_bufs,
        PTO2TaskSlotState** deferred_release_slot_states,
        int32_t& deferred_release_count
#if PTO2_SCHED_PROFILING
        , int thread_idx
#endif
        ) {
#if PTO2_SCHED_PROFILING
    PTO2CompletionStats stats = sched->on_mixed_task_complete(slot_state, thread_idx, local_bufs);
#else
    sched->on_mixed_task_complete(slot_state, local_bufs);
#endif
    deferred_release_slot_states[deferred_release_count++] = &slot_state;
#if PTO2_SCHED_PROFILING
    return stats;
#endif
}

// =============================================================================
// Async Wait Entry (one per deferred task)
// =============================================================================

struct PTO2AsyncWaitEntry {
    PTO2TaskSlotState* slot_state{nullptr};
    PTO2CompletionCondition conditions[PTO2_MAX_COMPLETIONS_PER_TASK];
    int32_t condition_count{0};
    int32_t waiting_completion_count{0};
};

struct PTO2AsyncPollResult {
    int32_t completed{0};
    int32_t error_code{PTO2_ERROR_NONE};
    PTO2TaskSlotState* failed_slot_state{nullptr};
};

// =============================================================================
// Name helpers (used by dump / diagnostics)
// =============================================================================

inline const char* pto2_async_engine_name(PTO2AsyncEngine engine) {
    switch (engine) {
        case PTO2_ASYNC_ENGINE_SDMA: return "SDMA";
        case PTO2_ASYNC_ENGINE_ROCE: return "ROCE";
        case PTO2_ASYNC_ENGINE_URMA: return "URMA";
        case PTO2_ASYNC_ENGINE_CCU: return "CCU";
        default: return "UNKNOWN";
    }
}

// =============================================================================
// Async Wait List (managed by scheduler thread)
// =============================================================================

struct PTO2AsyncWaitList {
    PTO2AsyncWaitEntry entries[PTO2_MAX_ASYNC_WAITS];
    int32_t count{0};

    /**
     * Find or create an entry for the given slot_state.
     * Returns pointer to the entry, or nullptr if full.
     */
    PTO2AsyncWaitEntry* find_or_create(PTO2TaskSlotState* slot_state) {
        for (int32_t i = 0; i < count; i++) {
            if (entries[i].slot_state == slot_state) {
                return &entries[i];
            }
        }
        if (count >= PTO2_MAX_ASYNC_WAITS) {
            return nullptr;
        }
        PTO2AsyncWaitEntry& e = entries[count++];
        e.slot_state = slot_state;
        e.condition_count = 0;
        e.waiting_completion_count = 0;
        return &e;
    }

    bool add_counter(PTO2TaskSlotState* slot_state,
                     volatile uint32_t* counter_addr,
                     uint32_t expected_value,
                     PTO2AsyncEngine engine = PTO2_ASYNC_ENGINE_SDMA) {
        PTO2AsyncWaitEntry* entry = find_or_create(slot_state);
        if (!entry || counter_addr == nullptr
            || entry->condition_count >= PTO2_MAX_COMPLETIONS_PER_TASK) {
            return false;
        }
        PTO2CompletionCondition& cond = entry->conditions[entry->condition_count++];
        cond.engine = engine;
        cond.satisfied = false;
        cond.counter_addr = counter_addr;
        cond.expected_value = expected_value;
        entry->waiting_completion_count++;
        return true;
    }

    /**
     * Poll all entries. For each satisfied condition, decrement waiting_completion_count.
     * When an entry's count reaches zero, call on_mixed_task_complete and add to
     * deferred_release. Remove completed entries by swap-with-last.
     *
     * Returns the number of tasks that completed this call.
     */
    template <bool Profiling>
    PTO2AsyncPollResult poll_and_complete(
            PTO2SchedulerState* sched,
            PTO2LocalReadyBuffer* local_bufs,
            PTO2TaskSlotState** deferred_release_slot_states,
            int32_t& deferred_release_count,
            int32_t deferred_release_capacity
#if PTO2_SCHED_PROFILING
            , int thread_idx
#endif
            ) {
        PTO2AsyncPollResult result;
        for (int32_t i = count - 1; i >= 0; --i) {
            PTO2AsyncWaitEntry& entry = entries[i];

            for (int32_t c = 0; c < entry.condition_count; c++) {
                PTO2CompletionCondition& cond = entry.conditions[c];
                if (!cond.satisfied) {
                    // RDMA-written counters (e.g. TNOTIFY) bypass AICPU data cache.
                    // Invalidate before reading to see the true memory value.
                    if (cond.counter_addr) {
                        cache_invalidate_range(
                            reinterpret_cast<const void*>(const_cast<const uint32_t*>(cond.counter_addr)),
                            sizeof(uint32_t));
                    }
                    PTO2CompletionPollResult poll = cond.test();
                    if (poll.state == PTO2CompletionPollState::FAILED) {
                        result.error_code = poll.error_code;
                        result.failed_slot_state = entry.slot_state;
                        return result;
                    }
                    if (poll.state == PTO2CompletionPollState::READY) {
                        cond.satisfied = true;
                        entry.waiting_completion_count--;
                    }
                }
            }

            if (entry.waiting_completion_count <= 0) {
                if (deferred_release_count >= deferred_release_capacity) {
                    result.error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
                    result.failed_slot_state = entry.slot_state;
                    return result;
                }
#if PTO2_SCHED_PROFILING
                auto stats = pto2_complete_task<Profiling>(
                    sched,
                    *entry.slot_state,
                    local_bufs,
                    deferred_release_slot_states,
                    deferred_release_count,
                    thread_idx
                );
                (void)stats;
#else
                pto2_complete_task<Profiling>(
                    sched,
                    *entry.slot_state,
                    local_bufs,
                    deferred_release_slot_states,
                    deferred_release_count
                );
#endif
                result.completed++;

                // Swap-remove: replace with last entry
                int32_t last = count - 1;
                if (i != last) {
                    entries[i] = entries[last];
                }
                count = last;
            }
        }
        return result;
    }
    /**
     * Register deferred completions for a task from its CQ.
     *
     * Reads the kernel-written PTO2CompletionQueue and registers each entry
     * as a COUNTER wait condition.  Returns true when at least one condition
     * was registered (task is now tracked by the wait list).  On error,
     * error_code is set to a non-zero PTO2_ERROR_* value.
     */
    bool register_deferred(PTO2TaskSlotState& slot_state,
                           int32_t thread_idx, int32_t& error_code) {
        (void)thread_idx;
        error_code = PTO2_ERROR_NONE;
        PTO2TaskPayload* payload = slot_state.payload;
        if (payload == nullptr || !payload->complete_in_future) return false;

        if (payload->cq_addr == 0) {
#ifdef DEV_ERROR
            DEV_ERROR("Thread %d: complete_in_future=true but no CQ entries for task %d",
                      thread_idx,
                      static_cast<int32_t>(slot_state.task->mixed_task_id.local()));
#endif
            error_code = PTO2_ERROR_ASYNC_COMPLETION_INVALID;
            return false;
        }

        volatile PTO2CompletionQueue* cq = reinterpret_cast<volatile PTO2CompletionQueue*>(
            static_cast<uintptr_t>(payload->cq_addr));
        // AICore kernel flushes its cache (dcci) before returning, but the
        // AICPU may still hold a stale cache line for this CQ.  Invalidate
        // before reading so we see the kernel's writes.
        cache_invalidate_range(
            const_cast<const void*>(reinterpret_cast<volatile void*>(cq)),
            sizeof(PTO2CompletionQueue));
        int32_t cq_count = cq->count;
        if (cq_count <= 0) {
#ifdef DEV_ALWAYS
            DEV_ALWAYS("Thread %d: task %d CQ addr=0x%lx count=0, completing immediately",
                       thread_idx,
                       static_cast<int32_t>(slot_state.task->mixed_task_id.local()),
                       payload->cq_addr);
#endif
            return false;
        }
        if (cq_count > PTO2_CQ_MAX_ENTRIES) {
#ifdef DEV_ERROR
            DEV_ERROR("Thread %d: CQ count=%d exceeds max %d for task %d",
                      thread_idx, cq_count, PTO2_CQ_MAX_ENTRIES,
                      static_cast<int32_t>(slot_state.task->mixed_task_id.local()));
#endif
            error_code = PTO2_ERROR_ASYNC_COMPLETION_INVALID;
            return false;
        }
#ifdef DEV_ALWAYS
        DEV_ALWAYS("Thread %d: task %d reading CQ addr=0x%lx count=%d",
                   thread_idx, static_cast<int32_t>(slot_state.task->mixed_task_id.local()),
                   payload->cq_addr, cq_count);
#endif
        for (int32_t i = 0; i < cq_count; ++i) {
            const volatile PTO2CQEntry& entry = cq->entries[i];
#ifdef DEV_ALWAYS
            DEV_ALWAYS("Thread %d: task %d CQ[%d] engine=%s(%d) addr=0x%lx expected=%u",
                       thread_idx,
                       static_cast<int32_t>(slot_state.task->mixed_task_id.local()),
                       i,
                       pto2_async_engine_name(static_cast<PTO2AsyncEngine>(entry.engine)),
                       static_cast<int32_t>(entry.engine),
                       entry.addr,
                       entry.expected_value);
#endif
            volatile uint32_t* counter_addr = reinterpret_cast<volatile uint32_t*>(
                static_cast<uintptr_t>(entry.addr));
            if (!add_counter(&slot_state, counter_addr, entry.expected_value,
                             static_cast<PTO2AsyncEngine>(entry.engine))) {
                error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
                return false;
            }
        }
        return true;
    }

    /**
     * Dump wait list state for stall diagnostics.
     */
    void dump(int32_t thread_idx, int32_t max_entries = 4) const {
#ifdef DEV_ALWAYS
        DEV_ALWAYS("Thread %d: async_wait_list pending entries=%d", thread_idx, count);
        int32_t dump_count = count < max_entries ? count : max_entries;
        for (int32_t i = 0; i < dump_count; ++i) {
            const PTO2AsyncWaitEntry& entry = entries[i];
            int32_t task_id = -1;
            if (entry.slot_state != nullptr && entry.slot_state->task != nullptr) {
                task_id = static_cast<int32_t>(entry.slot_state->task->mixed_task_id.local());
            }
            DEV_ALWAYS("Thread %d: async_wait[%d] task=%d waiting=%d conditions=%d",
                       thread_idx, i, task_id, entry.waiting_completion_count, entry.condition_count);
            for (int32_t c = 0; c < entry.condition_count; ++c) {
                const PTO2CompletionCondition& cond = entry.conditions[c];
                uint32_t value = cond.counter_addr == nullptr ? 0 : *cond.counter_addr;
                DEV_ALWAYS("Thread %d:   cond[%d] engine=%s satisfied=%d counter_addr=0x%lx value=%u expected=%u",
                           thread_idx, c, pto2_async_engine_name(cond.engine),
                           cond.satisfied ? 1 : 0,
                           static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cond.counter_addr)),
                           value, cond.expected_value);
            }
        }
#else
        (void)thread_idx;
        (void)max_entries;
#endif
    }
};

#endif // PTO_ASYNC_WAIT_H
