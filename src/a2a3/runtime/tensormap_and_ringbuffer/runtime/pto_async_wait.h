/**
 * PTO Runtime2 - Async Completion Wait List
 *
 * Lightweight completion-queue abstraction for deferred task completion.
 * Supports two completion condition types:
 *   - EVENT_FLAG: poll a GM flag address; complete when *addr != 0
 *                 (maps to SDMA SdmaEventRecord.flag)
 *   - COUNTER:    poll a GM counter address; complete when *addr >= expected
 *                 (maps to notification barriers, e.g. AllReduce)
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

inline constexpr int32_t PTO2_MAX_ASYNC_WAITS = 64;

struct PTO2CompletionCondition {
    PTO2CompletionType type;
    bool satisfied{false};
    union {
        struct { volatile uint32_t* flag_addr; } event_flag;
        struct { volatile uint32_t* counter_addr; uint32_t expected_value; } counter;
    };

    bool test() const {
        if (satisfied) return true;
        switch (type) {
            case PTO2CompletionType::EVENT_FLAG:
                return *event_flag.flag_addr != 0;
            case PTO2CompletionType::COUNTER:
                return *counter.counter_addr >= counter.expected_value;
        }
        return false;
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
    PTO2TaskSlotState* slot_state;
    PTO2CompletionCondition conditions[PTO2_MAX_COMPLETIONS_PER_TASK];
    int32_t condition_count{0};
    int32_t waiting_completion_count{0};
};

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

    /**
     * Register an event-flag completion condition for a task.
     * flag_addr points to SdmaEventRecord.flag in GM (or any uint32_t GM flag).
     */
    bool add_event_flag(PTO2TaskSlotState* slot_state, volatile uint32_t* flag_addr) {
        PTO2AsyncWaitEntry* entry = find_or_create(slot_state);
        if (!entry || entry->condition_count >= PTO2_MAX_COMPLETIONS_PER_TASK) {
            return false;
        }
        PTO2CompletionCondition& cond = entry->conditions[entry->condition_count++];
        cond.type = PTO2CompletionType::EVENT_FLAG;
        cond.satisfied = false;
        cond.event_flag.flag_addr = flag_addr;
        entry->waiting_completion_count++;
        return true;
    }

    /**
     * Register a counter-based completion condition for a task.
     * counter_addr points to a GM counter; completes when *counter_addr >= expected_value.
     */
    bool add_counter(PTO2TaskSlotState* slot_state, volatile uint32_t* counter_addr,
                     uint32_t expected_value) {
        PTO2AsyncWaitEntry* entry = find_or_create(slot_state);
        if (!entry || entry->condition_count >= PTO2_MAX_COMPLETIONS_PER_TASK) {
            return false;
        }
        PTO2CompletionCondition& cond = entry->conditions[entry->condition_count++];
        cond.type = PTO2CompletionType::COUNTER;
        cond.satisfied = false;
        cond.counter.counter_addr = counter_addr;
        cond.counter.expected_value = expected_value;
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
    int32_t poll_and_complete(
            PTO2SchedulerState* sched,
            PTO2LocalReadyBuffer* local_bufs,
            PTO2TaskSlotState** deferred_release_slot_states,
            int32_t& deferred_release_count
#if PTO2_SCHED_PROFILING
            , int thread_idx
#endif
            ) {
        int32_t completed = 0;
        for (int32_t i = count - 1; i >= 0; --i) {
            PTO2AsyncWaitEntry& entry = entries[i];

            for (int32_t c = 0; c < entry.condition_count; c++) {
                PTO2CompletionCondition& cond = entry.conditions[c];
                if (!cond.satisfied && cond.test()) {
                    cond.satisfied = true;
                    entry.waiting_completion_count--;
                }
            }

            if (entry.waiting_completion_count <= 0) {
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
                completed++;

                // Swap-remove: replace with last entry
                int32_t last = count - 1;
                if (i != last) {
                    entries[i] = entries[last];
                }
                count = last;
            }
        }
        return completed;
    }
};

#endif // PTO_ASYNC_WAIT_H
