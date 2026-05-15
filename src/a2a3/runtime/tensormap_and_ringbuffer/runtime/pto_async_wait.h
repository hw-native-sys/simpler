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
#include "pto_completion_ingress.h"
#include "pto_completion_token.h"
#include "pto_runtime2_types.h"

struct PTO2SchedulerState;
struct PTO2LocalReadyBuffer;
struct CompletionStats;

inline constexpr int32_t MAX_ASYNC_WAITS = 64;
inline constexpr int32_t MAX_PENDING_COMPLETIONS = 128;

inline bool completion_ingress_has_pending(volatile CompletionIngressQueue *completion_ingress) {
    if (completion_ingress == nullptr) return false;
    uint64_t head = __atomic_load_n(&completion_ingress->head, __ATOMIC_ACQUIRE);
    uint64_t tail = __atomic_load_n(&completion_ingress->tail, __ATOMIC_ACQUIRE);
    return tail < head;
}

inline uintptr_t completion_ingress_cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
}

struct CompletionCondition;

using CompletionPollFn = CompletionPollResult (*)(const CompletionCondition &);
using CompletionRetireFn = void (*)(CompletionCondition &);

struct CompletionBackendOps {
    CompletionPollFn poll;
    CompletionRetireFn retire;
};

struct CompletionCondition {
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

// Per-completion-type ops. SDMA_EVENT_RECORD detail lives in
// backend/sdma/sdma_completion_scheduler.h; the op wrappers below are thin
// glue mapping CompletionCondition.addr into the backend's raw-addr helpers.
inline CompletionPollResult counter_poll_op(const CompletionCondition &cond) {
    if (cond.counter_addr == nullptr) {
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }
    return {
        *cond.counter_addr >= cond.expected_value ? CompletionPollState::READY : CompletionPollState::PENDING,
        PTO2_ERROR_NONE
    };
}

inline void counter_retire_op(CompletionCondition & /*cond*/) {}

inline CompletionPollResult sdma_event_record_poll_op(const CompletionCondition &cond) {
    return poll_sdma_event_record(cond.addr);
}

inline void sdma_event_record_retire_op(CompletionCondition &cond) {
    retire_sdma_event_record(cond.addr);
}

inline const CompletionBackendOps *completion_backend_ops_for(int completion_type) {
    static const CompletionBackendOps kOps[] = {
        {counter_poll_op, counter_retire_op},                      // COMPLETION_TYPE_COUNTER = 0
        {sdma_event_record_poll_op, sdma_event_record_retire_op},  // COMPLETION_TYPE_SDMA_EVENT_RECORD = 1
    };
    constexpr int kOpsCount = static_cast<int>(sizeof(kOps) / sizeof(kOps[0]));
    if (completion_type < 0 || completion_type >= kOpsCount) return nullptr;
    return &kOps[completion_type];
}

inline CompletionPollResult CompletionCondition::test() const {
    if (satisfied) {
        return {CompletionPollState::READY, PTO2_ERROR_NONE};
    }
    const CompletionBackendOps *ops = completion_backend_ops_for(completion_type);
    if (ops == nullptr || ops->poll == nullptr) {
        return {CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
    }
    return ops->poll(*this);
}

inline void CompletionCondition::retire() {
    if (retired) return;
    const CompletionBackendOps *ops = completion_backend_ops_for(completion_type);
    if (ops != nullptr && ops->retire != nullptr) {
        ops->retire(*this);
    }
    retired = true;
}

struct AsyncWaitEntry {
    PTO2TaskSlotState *slot_state{nullptr};
    PTO2TaskId task_token{PTO2TaskId::invalid()};
    CompletionCondition conditions[MAX_COMPLETIONS_PER_TASK];
    int32_t condition_count{0};
    int32_t waiting_completion_count{0};
    bool normal_done{false};
};

struct PendingCompletion {
    PTO2TaskId task_token{PTO2TaskId::invalid()};
    uint64_t addr{0};
    uint32_t expected_value{0};
    AsyncEngine engine{ASYNC_ENGINE_SDMA};
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
    PendingCompletion pending_completions[MAX_PENDING_COMPLETIONS];
    int32_t pending_completion_count{0};

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

    int32_t
    drain_completion_ingress_locked(volatile CompletionIngressQueue *completion_ingress, int32_t &error_code) {
        error_code = PTO2_ERROR_NONE;
        if (completion_ingress == nullptr) return 0;

        int32_t drained = 0;
        while (true) {
            uint64_t tail = __atomic_load_n(&completion_ingress->tail, __ATOMIC_ACQUIRE);
            uint64_t head_snapshot = __atomic_load_n(&completion_ingress->head, __ATOMIC_ACQUIRE);
            if (tail >= head_snapshot) break;

            while (tail < head_snapshot) {
                volatile CompletionIngressEntry *slot =
                    &completion_ingress->entries[tail & COMPLETION_INGRESS_MASK];
                uint64_t expected_seq = tail + 1;
                uint64_t seq = __atomic_load_n(&slot->seq, __ATOMIC_ACQUIRE);
                if (seq != expected_seq) return drained;

                PTO2TaskId token{slot->task_token.raw};
                uint64_t addr = slot->addr;
                uint32_t expected_value = slot->expected_value;
                AsyncEngine engine = static_cast<AsyncEngine>(slot->engine);

                __atomic_store_n(&slot->seq, 0, __ATOMIC_RELEASE);
                __atomic_store_n(&completion_ingress->tail, tail + 1, __ATOMIC_RELEASE);
                drained++;
                tail++;

                AsyncWaitEntry *entry = find_entry_by_token(token);
                if (entry != nullptr) {
                    if (entry->condition_count >= MAX_COMPLETIONS_PER_TASK) {
                        error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
                        return drained;
                    }
                    CompletionCondition &cond = entry->conditions[entry->condition_count++];
                    cond.engine = engine;
                    cond.completion_type = COMPLETION_TYPE_COUNTER;
                    cond.satisfied = false;
                    cond.retired = false;
                    cond.addr = addr;
                    cond.counter_addr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
                    cond.expected_value = expected_value;
                    entry->waiting_completion_count++;
                    continue;
                }

                if (pending_completion_count >= MAX_PENDING_COMPLETIONS) {
                    error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
                    return drained;
                }
                PendingCompletion &pending = pending_completions[pending_completion_count++];
                pending.task_token = token;
                pending.addr = addr;
                pending.expected_value = expected_value;
                pending.engine = engine;
            }
        }
        return drained;
    }

    void absorb_pending_completions_locked(AsyncWaitEntry &entry) {
        int32_t write = 0;
        for (int32_t i = 0; i < pending_completion_count; i++) {
            if (pending_completions[i].task_token == entry.task_token) {
                if (entry.condition_count < MAX_COMPLETIONS_PER_TASK) {
                    CompletionCondition &cond = entry.conditions[entry.condition_count++];
                    cond.engine = pending_completions[i].engine;
                    cond.completion_type = COMPLETION_TYPE_COUNTER;
                    cond.satisfied = false;
                    cond.retired = false;
                    cond.addr = pending_completions[i].addr;
                    cond.counter_addr =
                        reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(pending_completions[i].addr));
                    cond.expected_value = pending_completions[i].expected_value;
                    entry.waiting_completion_count++;
                }
            } else {
                if (write != i) pending_completions[write] = pending_completions[i];
                write++;
            }
        }
        pending_completion_count = write;
    }

    enum class RegisterResult { Registered, NotDeferred, Skipped, Error };

    bool append_condition_locked(
        AsyncWaitEntry &entry, uint64_t addr, uint32_t expected_value, AsyncEngine engine,
        int32_t completion_type, int32_t &error_code
    ) {
        if (entry.condition_count >= MAX_COMPLETIONS_PER_TASK) {
            error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
            return false;
        }
        CompletionCondition &cond = entry.conditions[entry.condition_count++];
        cond.engine = engine;
        cond.completion_type = completion_type;
        cond.satisfied = false;
        cond.retired = false;
        cond.addr = addr;
        cond.counter_addr = completion_type == COMPLETION_TYPE_COUNTER ?
                                reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr)) :
                                nullptr;
        cond.expected_value = expected_value;
        entry.waiting_completion_count++;
        return true;
    }

    RegisterResult
    register_deferred(PTO2TaskSlotState &slot_state, const AsyncCtx &async_ctx, bool normal_done, int32_t &error_code) {
        error_code = PTO2_ERROR_NONE;
        if (slot_state.payload == nullptr) {
            return RegisterResult::NotDeferred;
        }

        if (!try_lock()) return RegisterResult::Skipped;

        uint32_t deferred_count = 0;
        if (async_ctx.completion_count != nullptr) {
            if (async_ctx.completion_error_code != nullptr) {
                if (*async_ctx.completion_error_code != PTO2_ERROR_NONE) {
                    error_code = *async_ctx.completion_error_code;
                    unlock();
                    return RegisterResult::Error;
                }
            }
            deferred_count = *async_ctx.completion_count;
        }
        if (deferred_count > async_ctx.completion_capacity) {
            error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
            unlock();
            return RegisterResult::Error;
        }
        if (deferred_count > 0) {
            if (async_ctx.completion_entries == nullptr) {
                error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
                unlock();
                return RegisterResult::Error;
            }
        }
        AsyncWaitEntry *entry = find_entry_by_token(slot_state.task->task_id);
        if (entry == nullptr && deferred_count == 0) {
            unlock();
            return RegisterResult::NotDeferred;
        }
        if (entry == nullptr) {
            if (count >= MAX_ASYNC_WAITS) {
                error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
                unlock();
                return RegisterResult::Error;
            }
            entry = &entries[count++];
            entry->slot_state = &slot_state;
            entry->task_token = slot_state.task->task_id;
            entry->condition_count = 0;
            entry->waiting_completion_count = 0;
            entry->normal_done = false;
        }
        if (normal_done) {
            entry->normal_done = true;
        }

        for (uint32_t i = 0; i < deferred_count; ++i) {
            volatile DeferredCompletionEntry *deferred = &async_ctx.completion_entries[i];
            if (deferred->completion_type == COMPLETION_TYPE_COUNTER) {
                volatile uint32_t *counter =
                    reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(deferred->addr));
                cache_invalidate_range(
                    reinterpret_cast<const void *>(completion_ingress_cache_line(counter)), sizeof(uint32_t)
                );
            }
            if (!append_condition_locked(
                    *entry, deferred->addr, deferred->expected_value, static_cast<AsyncEngine>(deferred->engine),
                    deferred->completion_type, error_code
                )) {
                unlock();
                return RegisterResult::Error;
            }
        }

        absorb_pending_completions_locked(*entry);
        unlock();
        return RegisterResult::Registered;
    }

    template <bool Profiling>
    AsyncPollResult poll_and_complete(
        volatile CompletionIngressQueue *completion_ingress, PTO2SchedulerState *sched,
        PTO2LocalReadyBuffer *local_bufs, PTO2TaskSlotState **deferred_release_slot_states,
        int32_t &deferred_release_count, int32_t deferred_release_capacity
#if PTO2_SCHED_PROFILING
        ,
        int thread_idx
#endif
    );
};

#endif  // PTO_ASYNC_WAIT_H
