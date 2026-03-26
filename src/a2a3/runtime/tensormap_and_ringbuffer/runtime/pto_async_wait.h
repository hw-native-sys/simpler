/**
 * PTO Runtime2 - Async Completion Wait List
 *
 * Lightweight watch-list abstraction for deferred task completion.
 *
 * The scheduler polls two logical protocols described in docs/runtime_async.md:
 *   - CQ protocol: dispatch by async engine, then by engine-specific sub-type
 *                  (today: SDMA EVENT_FLAG / EVENT_HANDLE_SLOT)
 *   - Notification protocol: poll a GM counter until it reaches expected_value
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

inline constexpr int32_t PTO2_MAX_ASYNC_WAITS = 64;
inline constexpr uint64_t PTO2_ASYNC_FAILURE_SENTINEL = UINT64_MAX;

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
    PTO2CompletionType type;
    bool satisfied{false};
    union {
        struct { volatile uint32_t* flag_addr; } event_flag;
        struct { volatile uint64_t* handle_slot_addr; } event_handle_slot;
        struct { volatile uint32_t* counter_addr; uint32_t expected_value; } counter;
    };

    PTO2CompletionPollResult test_notification() const {
        if (counter.counter_addr == nullptr) {
            return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
        }
        return {*counter.counter_addr >= counter.expected_value ? PTO2CompletionPollState::READY
                                                               : PTO2CompletionPollState::PENDING,
                PTO2_ERROR_NONE};
    }

    PTO2CompletionPollResult test_sdma() const {
        switch (type) {
            case PTO2CompletionType::EVENT_FLAG:
                if (event_flag.flag_addr == nullptr) {
                    return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
                }
                return {*event_flag.flag_addr != 0 ? PTO2CompletionPollState::READY
                                                  : PTO2CompletionPollState::PENDING,
                        PTO2_ERROR_NONE};
            case PTO2CompletionType::EVENT_HANDLE_SLOT:
                if (event_handle_slot.handle_slot_addr == nullptr) {
                    return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
                }
                {
                    const uint64_t handle = *event_handle_slot.handle_slot_addr;
                    if (handle == 0) {
                        return {PTO2CompletionPollState::PENDING, PTO2_ERROR_NONE};
                    }
                    if (handle == PTO2_ASYNC_FAILURE_SENTINEL) {
                        return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_FAILED};
                    }
                    volatile uint32_t* flag_addr = reinterpret_cast<volatile uint32_t*>(
                        static_cast<uintptr_t>(handle));
                    if (flag_addr == nullptr) {
                        return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
                    }
                    return {*flag_addr != 0 ? PTO2CompletionPollState::READY
                                            : PTO2CompletionPollState::PENDING,
                            PTO2_ERROR_NONE};
                }
            default:
                return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
        }
    }

    PTO2CompletionPollResult test_engine_cq() const {
        switch (engine) {
            case PTO2_ASYNC_ENGINE_SDMA:
                return test_sdma();
            case PTO2_ASYNC_ENGINE_ROCE:
            case PTO2_ASYNC_ENGINE_URMA:
            case PTO2_ASYNC_ENGINE_CCU:
            default:
                return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
        }
    }

    PTO2CompletionPollResult test() const {
        if (satisfied) {
            return {PTO2CompletionPollState::READY, PTO2_ERROR_NONE};
        }
        if (type == PTO2CompletionType::COUNTER) {
            return test_notification();
        }
        return test_engine_cq();
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

inline const char* pto2_completion_type_name(PTO2CompletionType type) {
    switch (type) {
        case PTO2CompletionType::EVENT_FLAG: return "EVENT_FLAG";
        case PTO2CompletionType::EVENT_HANDLE_SLOT: return "EVENT_HANDLE_SLOT";
        case PTO2CompletionType::COUNTER: return "COUNTER";
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

    bool add_engine_cq(PTO2TaskSlotState* slot_state,
                       PTO2AsyncEngine engine,
                       PTO2CompletionType type,
                       uint64_t addr) {
        PTO2AsyncWaitEntry* entry = find_or_create(slot_state);
        if (!entry || addr == 0 || entry->condition_count >= PTO2_MAX_COMPLETIONS_PER_TASK) {
            return false;
        }
        PTO2CompletionCondition& cond = entry->conditions[entry->condition_count++];
        cond.engine = engine;
        cond.type = type;
        cond.satisfied = false;
        switch (type) {
            case PTO2CompletionType::EVENT_FLAG:
                cond.event_flag.flag_addr = reinterpret_cast<volatile uint32_t*>(
                    static_cast<uintptr_t>(addr));
                break;
            case PTO2CompletionType::EVENT_HANDLE_SLOT:
                cond.event_handle_slot.handle_slot_addr = reinterpret_cast<volatile uint64_t*>(
                    static_cast<uintptr_t>(addr));
                break;
            case PTO2CompletionType::COUNTER:
            default:
                return false;
        }
        entry->waiting_completion_count++;
        return true;
    }

    bool add_notification(PTO2TaskSlotState* slot_state,
                          volatile uint32_t* counter_addr,
                          uint32_t expected_value) {
        PTO2AsyncWaitEntry* entry = find_or_create(slot_state);
        if (!entry || counter_addr == nullptr
            || entry->condition_count >= PTO2_MAX_COMPLETIONS_PER_TASK) {
            return false;
        }
        PTO2CompletionCondition& cond = entry->conditions[entry->condition_count++];
        cond.engine = PTO2_ASYNC_ENGINE_SDMA;
        cond.type = PTO2CompletionType::COUNTER;
        cond.satisfied = false;
        cond.counter.counter_addr = counter_addr;
        cond.counter.expected_value = expected_value;
        entry->waiting_completion_count++;
        return true;
    }

    /**
     * Register an event-flag completion condition for a task.
     * flag_addr points to SdmaEventRecord.flag in GM (or any uint32_t GM flag).
     */
    bool add_event_flag(PTO2TaskSlotState* slot_state, volatile uint32_t* flag_addr) {
        return add_engine_cq(slot_state, PTO2_ASYNC_ENGINE_SDMA,
                             PTO2CompletionType::EVENT_FLAG,
                             static_cast<uint64_t>(reinterpret_cast<uintptr_t>(flag_addr)));
    }

    bool add_event_handle_slot(PTO2TaskSlotState* slot_state, volatile uint64_t* handle_slot_addr) {
        return add_engine_cq(slot_state, PTO2_ASYNC_ENGINE_SDMA,
                             PTO2CompletionType::EVENT_HANDLE_SLOT,
                             static_cast<uint64_t>(reinterpret_cast<uintptr_t>(handle_slot_addr)));
    }

    /**
     * Register a counter-based completion condition for a task.
     * counter_addr points to a GM counter; completes when *counter_addr >= expected_value.
     */
    bool add_counter(PTO2TaskSlotState* slot_state, volatile uint32_t* counter_addr,
                     uint32_t expected_value) {
        return add_notification(slot_state, counter_addr, expected_value);
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
     * as an async wait condition.  Returns true when at least one condition
     * was registered (task is now tracked by the wait list).  On error,
     * error_code is set to a non-zero PTO2_ERROR_* value.
     */
    bool register_deferred(PTO2TaskSlotState& slot_state,
                           int32_t thread_idx, int32_t& error_code) {
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
            DEV_ALWAYS("Thread %d: task %d CQ[%d] engine=%s(%d) type=%s(%d) addr=0x%lx expected=%u",
                       thread_idx,
                       static_cast<int32_t>(slot_state.task->mixed_task_id.local()),
                       i,
                       pto2_async_engine_name(static_cast<PTO2AsyncEngine>(entry.engine)),
                       static_cast<int32_t>(entry.engine),
                       pto2_completion_type_name(static_cast<PTO2CompletionType>(entry.completion_type)),
                       static_cast<int32_t>(entry.completion_type),
                       entry.addr,
                       entry.expected_value);
#endif
            if (!register_one_cq_entry(slot_state, thread_idx, error_code,
                                       static_cast<PTO2AsyncEngine>(entry.engine),
                                       static_cast<PTO2CompletionType>(entry.completion_type),
                                       entry.addr, entry.expected_value)) {
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
                switch (cond.type) {
                    case PTO2CompletionType::EVENT_FLAG: {
                        uint32_t value = cond.event_flag.flag_addr == nullptr ? 0 : *cond.event_flag.flag_addr;
                        DEV_ALWAYS("Thread %d:   cond[%d] engine=%s type=%s satisfied=%d flag_addr=0x%lx flag=%u",
                                   thread_idx, c, pto2_async_engine_name(cond.engine),
                                   pto2_completion_type_name(cond.type), cond.satisfied ? 1 : 0,
                                   static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cond.event_flag.flag_addr)),
                                   value);
                        break;
                    }
                    case PTO2CompletionType::EVENT_HANDLE_SLOT: {
                        uint64_t handle = cond.event_handle_slot.handle_slot_addr == nullptr
                                              ? 0
                                              : *cond.event_handle_slot.handle_slot_addr;
                        uint32_t flag = 0;
                        if (handle != 0 && handle != PTO2_ASYNC_FAILURE_SENTINEL) {
                            volatile uint32_t* flag_addr =
                                reinterpret_cast<volatile uint32_t*>(static_cast<uintptr_t>(handle));
                            if (flag_addr != nullptr) {
                                flag = *flag_addr;
                            }
                        }
                        DEV_ALWAYS("Thread %d:   cond[%d] engine=%s type=%s satisfied=%d handle_slot=0x%lx handle=0x%lx flag=%u",
                                   thread_idx, c, pto2_async_engine_name(cond.engine),
                                   pto2_completion_type_name(cond.type), cond.satisfied ? 1 : 0,
                                   static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cond.event_handle_slot.handle_slot_addr)),
                                   handle, flag);
                        break;
                    }
                    case PTO2CompletionType::COUNTER: {
                        uint32_t value = cond.counter.counter_addr == nullptr ? 0 : *cond.counter.counter_addr;
                        DEV_ALWAYS("Thread %d:   cond[%d] engine=%s type=%s satisfied=%d counter_addr=0x%lx value=%u expected=%u",
                                   thread_idx, c, pto2_async_engine_name(cond.engine),
                                   pto2_completion_type_name(cond.type), cond.satisfied ? 1 : 0,
                                   static_cast<uint64_t>(reinterpret_cast<uintptr_t>(cond.counter.counter_addr)),
                                   value, cond.counter.expected_value);
                        break;
                    }
                    default:
                        DEV_ALWAYS("Thread %d:   cond[%d] engine=%s type=%d satisfied=%d",
                                   thread_idx, c, pto2_async_engine_name(cond.engine),
                                   static_cast<int32_t>(cond.type), cond.satisfied ? 1 : 0);
                        break;
                }
            }
        }
#else
        (void)thread_idx;
        (void)max_entries;
#endif
    }

private:
    bool register_one_cq_entry(PTO2TaskSlotState& slot_state,
                               int32_t thread_idx, int32_t& error_code,
                               PTO2AsyncEngine engine, PTO2CompletionType type,
                               uint64_t addr, uint32_t expected_value) {
        (void)thread_idx;
        bool added = false;
        switch (type) {
            case PTO2CompletionType::EVENT_FLAG:
                added = add_engine_cq(&slot_state, engine, PTO2CompletionType::EVENT_FLAG, addr);
                break;
            case PTO2CompletionType::EVENT_HANDLE_SLOT:
                added = add_engine_cq(&slot_state, engine, PTO2CompletionType::EVENT_HANDLE_SLOT, addr);
                break;
            case PTO2CompletionType::COUNTER: {
                volatile uint32_t* counter_addr = reinterpret_cast<volatile uint32_t*>(
                    static_cast<uintptr_t>(addr));
                added = add_notification(&slot_state, counter_addr, expected_value);
                break;
            }
            default:
                error_code = PTO2_ERROR_ASYNC_COMPLETION_INVALID;
                return false;
        }
        if (!added) {
            error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
            return false;
        }
        return true;
    }
};

// =============================================================================
// PTO2NotificationWaitList::poll_and_enqueue — deferred definition
// (struct declared in pto_scheduler.h, method body needs PTO2SchedulerState)
// =============================================================================

// Provided by platform: DC CIVAC + DSB + ISB on real hardware, no-op on sim.
extern void cache_invalidate_range(const void* addr, size_t size);

inline int32_t PTO2NotificationWaitList::poll_and_enqueue(
        PTO2SchedulerState* sched, PTO2LocalReadyBuffer* local_bufs) {
    // Called under try_lock_poll() — only one thread executes at a time.
    int32_t cur = count.load(std::memory_order_acquire);
    int32_t enqueued = 0;
    for (int32_t i = cur - 1; i >= 0; --i) {
        PTO2NotificationWaitEntry& e = entries[i];
        // TNOTIFY writes arrive via RDMA, bypassing AICPU data cache.
        // Invalidate the cache line to read the true memory value.
        cache_invalidate_range(reinterpret_cast<const void*>(
            const_cast<const uint32_t*>(e.counter_addr)), sizeof(uint32_t));
        if (*e.counter_addr >= e.expected_value) {
            PTO2ResourceShape shape = pto2_active_mask_to_shape(e.slot_state->active_mask);
            if (!local_bufs || !local_bufs[static_cast<int32_t>(shape)].try_push(e.slot_state)) {
                sched->ready_queues[static_cast<int32_t>(shape)].push(e.slot_state);
            }
            enqueued++;
            int32_t last = count.load(std::memory_order_acquire) - 1;
            if (i != last) {
                entries[i] = entries[last];
            }
            count.store(last, std::memory_order_release);
        }
    }
    return enqueued;
}

#endif // PTO_ASYNC_WAIT_H
