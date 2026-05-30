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

#include "intrinsic.h"
#include "aicore_completion_mailbox.h"
#include "pto_runtime2_types.h"

struct PTO2SchedulerState;
struct PTO2LocalReadyBuffer;
struct CompletionStats;

inline constexpr int32_t MAX_ASYNC_WAITS = 64;

// The mailbox transport (has_pending / try_push_condition /
// try_push_normal_done / try_pop) lives as AICoreCompletionMailbox member
// functions in aicore_completion_mailbox.h. This file only holds the
// application layer: translating drained messages into wait-list state.

enum class CompletionPollState : uint8_t {
    PENDING = 0,
    READY = 1,
    FAILED = 2,
};

struct CompletionPollResult {
    CompletionPollState state{CompletionPollState::PENDING};
    int32_t error_code{PTO2_ERROR_NONE};
};

struct CompletionCondition {
    AsyncEngine engine{ASYNC_ENGINE_SDMA};
    bool satisfied{false};
    volatile uint32_t *counter_addr{nullptr};
    uint32_t expected_value{0};

    CompletionPollResult test() const {
        if (satisfied) {
            return {CompletionPollState::READY, PTO2_ERROR_NONE};
        }
        if (counter_addr == nullptr) {
            return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
        }
        return {
            *counter_addr >= expected_value ? CompletionPollState::READY : CompletionPollState::PENDING, PTO2_ERROR_NONE
        };
    }
};

struct AsyncWaitEntry {
    PTO2TaskSlotState *slot_state{nullptr};
    PTO2TaskId task_token{PTO2TaskId::invalid()};
    CompletionCondition conditions[MAX_COMPLETIONS_PER_TASK];
    int32_t condition_count{0};
    int32_t waiting_completion_count{0};
    bool normal_done{false};
};

struct AsyncPollResult {
    int32_t completed{0};
    int32_t error_code{PTO2_ERROR_NONE};
    PTO2TaskSlotState *failed_slot_state{nullptr};
};

inline const char *async_engine_name(AsyncEngine engine) {
    switch (engine) {
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

struct AsyncWaitList {
    std::atomic<int32_t> busy{0};
    AsyncWaitEntry entries[MAX_ASYNC_WAITS];
    int32_t count{0};
    std::atomic<uint64_t> mpsc_skipped_count{0};

    bool try_lock() {
        int32_t expected = 0;
        return busy.compare_exchange_strong(expected, 1, std::memory_order_acquire, std::memory_order_relaxed);
    }

    void unlock() { busy.store(0, std::memory_order_release); }

    AsyncWaitEntry *find_entry_by_token(PTO2TaskId token) {
        for (int32_t i = 0; i < count; i++) {
            if (entries[i].task_token == token) return &entries[i];
        }
        return nullptr;
    }

    struct DrainCompletionSink {
        PTO2SchedulerState *sched{nullptr};
        PTO2LocalReadyBuffer *local_bufs{nullptr};
        PTO2TaskSlotState **deferred_release_slot_states{nullptr};
        int32_t *deferred_release_count{nullptr};
        int32_t deferred_release_capacity{0};
        int32_t inline_completed{0};
#if PTO2_SCHED_PROFILING
        int32_t thread_idx{0};
#endif
        bool can_inline_complete() const { return sched != nullptr; }
    };

    bool try_inline_complete_locked(DrainCompletionSink &sink, PTO2TaskSlotState &slot_state);

    // Single-consumer drain: pop each published message in tail order and
    // translate it into wait-list state. An empty sink (sched == nullptr) just
    // materializes entries; a sched-aware sink additionally inline-completes
    // lonely NotDeferred NORMAL_DONEs without ever growing entries[].
    int32_t drain_aicore_completion_mailbox_locked(
        AICoreCompletionMailbox *aicore_mailbox, DrainCompletionSink &sink, int32_t &error_code
    ) {
        error_code = PTO2_ERROR_NONE;
        if (aicore_mailbox == nullptr) return 0;

        int32_t drained = 0;
        AICoreCompletionMsgView msg;
        // try_pop is the transport layer (seq-gated, in-order dequeue); this
        // loop is the application layer (translate each message into wait-list
        // state). try_pop returns false at the first gap or when empty.
        while (aicore_mailbox->try_pop(msg)) {
            drained++;
            if (msg.kind == MSG_KIND_CONDITION) {
                AsyncWaitEntry *entry = find_entry_by_token(msg.task_token);
                if (entry == nullptr) {
                    if (count >= MAX_ASYNC_WAITS) {
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
                if (!append_condition_locked(
                        *entry, msg.addr, msg.expected_value, static_cast<AsyncEngine>(msg.engine), error_code
                    )) {
                    return drained;
                }
            } else if (msg.kind == MSG_KIND_TASK_NORMAL_DONE) {
                PTO2TaskSlotState *slot_state_ptr =
                    reinterpret_cast<PTO2TaskSlotState *>(static_cast<uintptr_t>(msg.addr));
                AsyncWaitEntry *entry = find_entry_by_token(msg.task_token);
                if (entry == nullptr) {
                    if (sink.can_inline_complete()) {
                        (void)try_inline_complete_locked(sink, *slot_state_ptr);
                        continue;
                    }
                    if (count >= MAX_ASYNC_WAITS) {
                        error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
                        return drained;
                    }
                    entry = &entries[count++];
                    entry->task_token = msg.task_token;
                    entry->slot_state = slot_state_ptr;
                    entry->condition_count = 0;
                    entry->waiting_completion_count = 0;
                    entry->normal_done = true;
                } else {
                    if (entry->slot_state == nullptr) {
                        entry->slot_state = slot_state_ptr;
                    }
                    entry->normal_done = true;
                }
            } else {
                error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
                return drained;
            }
        }
        return drained;
    }

    bool append_condition_locked(
        AsyncWaitEntry &entry, uint64_t addr, uint32_t expected_value, AsyncEngine engine, int32_t &error_code
    ) {
        if (entry.condition_count >= MAX_COMPLETIONS_PER_TASK) {
            error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
            return false;
        }
        CompletionCondition &cond = entry.conditions[entry.condition_count++];
        cond.engine = engine;
        cond.satisfied = false;
        cond.counter_addr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
        cond.expected_value = expected_value;
        entry.waiting_completion_count++;
        return true;
    }

    template <bool Profiling>
    AsyncPollResult poll_and_complete(
        AICoreCompletionMailbox *aicore_mailbox, PTO2SchedulerState *sched, PTO2LocalReadyBuffer *local_bufs,
        PTO2TaskSlotState **deferred_release_slot_states, int32_t &deferred_release_count,
        int32_t deferred_release_capacity
#if PTO2_SCHED_PROFILING
        ,
        int thread_idx
#endif
    );
};

#endif  // PTO_ASYNC_WAIT_H
