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

#ifndef PTO_ASYNC_WAIT_H
#define PTO_ASYNC_WAIT_H

#include <atomic>
#include <cstddef>
#include <cstdint>

#include "aicpu/platform_regs.h"
#include "backend/sdma/sdma_completion_scheduler.h"
#include "intrinsic.h"
#include "aicore_completion_mailbox.h"
#include "pto_completion_token.h"
#include "pto_runtime2_types.h"

struct PTO2SchedulerState;
struct PTO2LocalReadyBuffer;
struct CompletionStats;

inline constexpr int32_t MAX_ASYNC_WAITS = 64;

inline uintptr_t mailbox_cache_line(const volatile void *addr)
{
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
}

struct CompletionCondition;

using CompletionPollFn = CompletionPollResult (*)(const CompletionCondition &);
using CompletionRetireFn = void (*)(CompletionCondition &);

struct CompletionBackendOps
{
    CompletionPollFn poll;
    CompletionRetireFn retire;
};

struct CompletionCondition
{
    AsyncEngine engine{ASYNC_ENGINE_SDMA};
    int32_t completion_type{COMPLETION_TYPE_COUNTER};
    bool satisfied{false};
    bool retired{false};
    volatile uint32_t *counter_addr{nullptr};
    uint64_t addr{0};
    uint32_t expected_value{0};

    CompletionPollResult test() const;
    void retire();
};

inline CompletionPollResult counter_poll_op(const CompletionCondition &cond)
{
    if (cond.counter_addr == nullptr) return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    return {*cond.counter_addr >= cond.expected_value ? CompletionPollState::READY : CompletionPollState::PENDING, PTO2_ERROR_NONE};
}

inline void counter_retire_op(CompletionCondition &)
{}

inline CompletionPollResult sdma_event_record_poll_op(const CompletionCondition &cond)
{
    return poll_sdma_event_record(cond.addr);
}

inline void sdma_event_record_retire_op(CompletionCondition &cond)
{
    retire_sdma_event_record(cond.addr);
}

inline const CompletionBackendOps *completion_backend_ops_for(int completion_type)
{
    static const CompletionBackendOps kOps[] = {
        {counter_poll_op, counter_retire_op},                      // COMPLETION_TYPE_COUNTER = 0
        {sdma_event_record_poll_op, sdma_event_record_retire_op},  // COMPLETION_TYPE_SDMA_EVENT_RECORD = 1
    };
    constexpr int kOpsCount = static_cast<int>(sizeof(kOps) / sizeof(kOps[0]));
    if (completion_type < 0 || completion_type >= kOpsCount) return nullptr;
    return &kOps[completion_type];
}

inline CompletionPollResult CompletionCondition::test() const
{
    if (satisfied) return {CompletionPollState::READY, PTO2_ERROR_NONE};
    const CompletionBackendOps *ops = completion_backend_ops_for(completion_type);
    if (ops == nullptr || ops->poll == nullptr) return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    return ops->poll(*this);
}

inline void CompletionCondition::retire()
{
    if (retired) return;
    const CompletionBackendOps *ops = completion_backend_ops_for(completion_type);
    if (ops != nullptr && ops->retire != nullptr) ops->retire(*this);
    retired = true;
}

struct AsyncWaitEntry
{
    PTO2TaskSlotState *slot_state{nullptr};
    PTO2TaskId task_token{PTO2TaskId::invalid()};
    CompletionCondition conditions[MAX_COMPLETIONS_PER_TASK];
    int32_t condition_count{0};
    int32_t waiting_completion_count{0};
    bool normal_done{false};
};

struct AsyncPollResult
{
    int32_t completed{0};
    int32_t error_code{PTO2_ERROR_NONE};
    PTO2TaskSlotState *failed_slot_state{nullptr};
};

inline const char *async_engine_name(AsyncEngine engine)
{
    switch (engine)
    {
    case ASYNC_ENGINE_SDMA:
        return "SDMA";
    case ASYNC_ENGINE_ROCE:
        return "ROCE";
    case ASYNC_ENGINE_URMA:
        return "URMA";
    case ASYNC_ENGINE_CCU:
        return "CCU";
    default:
        return "UNKNOWN";
    }
}

struct AsyncWaitList
{
    std::atomic<int32_t> busy{0};
    AsyncWaitEntry entries[MAX_ASYNC_WAITS];
    int32_t count{0};
    std::atomic<uint64_t> mpsc_skipped_count{0};

    void reset_for_reuse()
    {
        busy.store(0, std::memory_order_relaxed);
        count = 0;
        mpsc_skipped_count.store(0, std::memory_order_relaxed);
    }

    bool try_lock()
    {
        int32_t expected = 0;
        return busy.compare_exchange_strong(expected, 1, std::memory_order_acquire, std::memory_order_relaxed);
    }

    void unlock()
    {
        busy.store(0, std::memory_order_release);
    }

    AsyncWaitEntry *find_entry_by_token(PTO2TaskId token)
    {
        for (int32_t i = 0; i < count; i++)
            if (entries[i].task_token == token) return &entries[i];
        return nullptr;
    }

    struct DrainCompletionSink
    {
        PTO2SchedulerState *sched{nullptr};
        int32_t inline_completed{0};

        bool can_inline_complete() const
        {
            return sched != nullptr;
        }
    };

    // Inline-complete a NotDeferred task during drain.
    bool try_inline_complete_locked(DrainCompletionSink &sink, PTO2TaskSlotState &slot_state);

    int32_t drain_aicore_completion_mailbox_locked(AICoreCompletionMailbox *aicore_mailbox, DrainCompletionSink &sink, int32_t &error_code)
    {
        error_code = PTO2_ERROR_NONE;
        if (aicore_mailbox == nullptr) return 0;

        int32_t drained = 0;
        AICoreCompletionMsgView msg;
        while (aicore_mailbox->try_pop(msg))
        {
            drained++;
            if (msg.kind == MSG_KIND_CONDITION)
            {
                AsyncWaitEntry *entry = find_entry_by_token(msg.task_token);
                if (entry == nullptr)
                {
                    if (count >= MAX_ASYNC_WAITS)
                    {
                        error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
                        return drained;
                    }
                    entry = &entries[count++];
                    entry->task_token = msg.task_token;
                    entry->slot_state = nullptr;
                    entry->condition_count = 0;
                    entry->waiting_completion_count = 0;
                    entry->normal_done = false;
                }
                if (!append_condition_locked(*entry, msg.addr, msg.expected_value, static_cast<AsyncEngine>(msg.engine), msg.completion_type, error_code)) return drained;
            }
            else if (msg.kind == MSG_KIND_TASK_NORMAL_DONE)
            {
                PTO2TaskSlotState *slot_state_ptr = reinterpret_cast<PTO2TaskSlotState *>(static_cast<uintptr_t>(msg.addr));
                AsyncWaitEntry *entry = find_entry_by_token(msg.task_token);
                if (entry == nullptr)
                {
                    if (sink.can_inline_complete())
                    {
                        (void)try_inline_complete_locked(sink, *slot_state_ptr);
                        continue;
                    }
                    if (count >= MAX_ASYNC_WAITS)
                    {
                        error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
                        return drained;
                    }
                    entry = &entries[count++];
                    entry->task_token = msg.task_token;
                    entry->slot_state = slot_state_ptr;
                    entry->condition_count = 0;
                    entry->waiting_completion_count = 0;
                    entry->normal_done = true;
                }
                else
                {
                    if (entry->slot_state == nullptr) entry->slot_state = slot_state_ptr;
                    entry->normal_done = true;
                }
            }
            else
            {
                error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
                return drained;
            }
        }
        return drained;
    }

    bool append_condition_locked(AsyncWaitEntry &entry, uint64_t addr, uint32_t expected_value, AsyncEngine engine, int32_t completion_type, int32_t &error_code)
    {
        if (entry.condition_count >= MAX_COMPLETIONS_PER_TASK)
        {
            error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
            return false;
        }
        CompletionCondition &cond = entry.conditions[entry.condition_count++];
        cond.engine = engine;
        cond.completion_type = completion_type;
        cond.satisfied = false;
        cond.retired = false;
        cond.addr = addr;
        cond.counter_addr = completion_type == COMPLETION_TYPE_COUNTER ? reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr)) : nullptr;
        cond.expected_value = expected_value;
        entry.waiting_completion_count++;
        return true;
    }

    template <bool Profiling>
    AsyncPollResult poll_and_complete(AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched);
};

#endif  // PTO_ASYNC_WAIT_H
