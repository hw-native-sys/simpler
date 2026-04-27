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
#include "common/memory_barrier.h"
#include "pto_completion_ingress.h"
#include "pto_runtime2_types.h"

struct PTO2SchedulerState;
struct PTO2LocalReadyBuffer;
struct PTO2CompletionStats;

inline constexpr int32_t PTO2_MAX_ASYNC_WAITS = 64;
inline constexpr int32_t PTO2_MAX_PENDING_COMPLETIONS = 128;

inline uintptr_t pto2_completion_ingress_cache_line(const volatile void *addr) {
    return reinterpret_cast<uintptr_t>(addr) & ~(uintptr_t(PTO2_ALIGN_SIZE) - 1u);
}

inline void pto2_completion_ingress_invalidate_entries(
    volatile PTO2CompletionIngressQueue *completion_ingress, uint64_t tail, uint64_t head
) {
    uint64_t active_count = head - tail;
    if (active_count == 0) return;

    uint64_t tail_index = tail & PTO2_COMPLETION_INGRESS_MASK;
    uint64_t first_span = active_count;
    uint64_t contiguous_capacity = PTO2_COMPLETION_INGRESS_CAPACITY - tail_index;
    if (first_span > contiguous_capacity) {
        first_span = contiguous_capacity;
    }

    cache_invalidate_range(
        const_cast<const void *>(reinterpret_cast<volatile void *>(&completion_ingress->entries[tail_index])),
        first_span * sizeof(PTO2CompletionIngressEntry)
    );

    uint64_t remaining = active_count - first_span;
    if (remaining == 0) return;

    cache_invalidate_range(
        const_cast<const void *>(reinterpret_cast<volatile void *>(&completion_ingress->entries[0])),
        remaining * sizeof(PTO2CompletionIngressEntry)
    );
}

inline bool pto2_completion_ingress_has_pending(volatile PTO2CompletionIngressQueue *completion_ingress) {
    if (completion_ingress == nullptr) return false;
    cache_invalidate_range(
        const_cast<const void *>(reinterpret_cast<volatile void *>(&completion_ingress->head)),
        sizeof(completion_ingress->head)
    );
    uint64_t head = __atomic_load_n(&completion_ingress->head, __ATOMIC_ACQUIRE);
    uint64_t tail = __atomic_load_n(&completion_ingress->tail, __ATOMIC_ACQUIRE);
    return tail < head;
}

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
    volatile uint32_t *counter_addr{nullptr};
    uint32_t expected_value{0};

    PTO2CompletionPollResult test() const {
        if (satisfied) {
            return {PTO2CompletionPollState::READY, PTO2_ERROR_NONE};
        }
        if (counter_addr == nullptr) {
            return {PTO2CompletionPollState::FAILED, PTO2_ERROR_ASYNC_COMPLETION_INVALID};
        }
        return {
            *counter_addr >= expected_value ? PTO2CompletionPollState::READY : PTO2CompletionPollState::PENDING,
            PTO2_ERROR_NONE
        };
    }
};

struct PTO2AsyncWaitEntry {
    PTO2TaskSlotState *slot_state{nullptr};
    PTO2TaskId task_token{PTO2TaskId::invalid()};
    PTO2CompletionCondition conditions[PTO2_MAX_COMPLETIONS_PER_TASK];
    int32_t condition_count{0};
    int32_t waiting_completion_count{0};
    bool normal_done{false};
};

struct PTO2PendingCompletion {
    PTO2TaskId task_token{PTO2TaskId::invalid()};
    uint64_t addr{0};
    uint32_t expected_value{0};
    PTO2AsyncEngine engine{PTO2_ASYNC_ENGINE_SDMA};
};

struct PTO2AsyncPollResult {
    int32_t completed{0};
    int32_t error_code{PTO2_ERROR_NONE};
    PTO2TaskSlotState *failed_slot_state{nullptr};
};

inline const char *pto2_async_engine_name(PTO2AsyncEngine engine) {
    switch (engine) {
    case PTO2_ASYNC_ENGINE_SDMA:
        return "SDMA";
    case PTO2_ASYNC_ENGINE_ROCE:
        return "ROCE";
    case PTO2_ASYNC_ENGINE_URMA:
        return "URMA";
    case PTO2_ASYNC_ENGINE_CCU:
        return "CCU";
    default:
        return "UNKNOWN";
    }
}

struct PTO2AsyncWaitList {
    std::atomic<int32_t> busy{0};
    PTO2AsyncWaitEntry entries[PTO2_MAX_ASYNC_WAITS];
    int32_t count{0};
    PTO2PendingCompletion pending_completions[PTO2_MAX_PENDING_COMPLETIONS];
    int32_t pending_completion_count{0};

    bool try_lock() {
        int32_t expected = 0;
        return busy.compare_exchange_strong(expected, 1, std::memory_order_acquire, std::memory_order_relaxed);
    }

    void unlock() { busy.store(0, std::memory_order_release); }

    PTO2AsyncWaitEntry *find_entry_by_token(PTO2TaskId token) {
        for (int32_t i = 0; i < count; i++) {
            if (entries[i].task_token == token) return &entries[i];
        }
        return nullptr;
    }

    int32_t
    drain_completion_ingress_locked(volatile PTO2CompletionIngressQueue *completion_ingress, int32_t &error_code) {
        error_code = PTO2_ERROR_NONE;
        if (completion_ingress == nullptr) return 0;

        int32_t drained = 0;
        while (true) {
            uint64_t tail = __atomic_load_n(&completion_ingress->tail, __ATOMIC_ACQUIRE);
            cache_invalidate_range(
                const_cast<const void *>(reinterpret_cast<volatile void *>(&completion_ingress->head)),
                sizeof(completion_ingress->head)
            );
            uint64_t head_snapshot = __atomic_load_n(&completion_ingress->head, __ATOMIC_ACQUIRE);
            if (tail >= head_snapshot) break;
            pto2_completion_ingress_invalidate_entries(completion_ingress, tail, head_snapshot);

            while (tail < head_snapshot) {
                volatile PTO2CompletionIngressEntry *slot =
                    &completion_ingress->entries[tail & PTO2_COMPLETION_INGRESS_MASK];
                uint64_t expected_seq = tail + 1;
                uint64_t seq = __atomic_load_n(&slot->seq, __ATOMIC_ACQUIRE);
                if (seq != expected_seq) return drained;

                PTO2TaskId token{slot->task_token.raw};
                uint64_t addr = slot->addr;
                uint32_t expected_value = slot->expected_value;
                PTO2AsyncEngine engine = static_cast<PTO2AsyncEngine>(slot->engine);

                slot->seq = 0;
                completion_ingress->tail = tail + 1;
                OUT_OF_ORDER_STORE_BARRIER();
                cache_flush_range(const_cast<const void *>(reinterpret_cast<volatile void *>(slot)), sizeof(*slot));
                cache_flush_range(
                    const_cast<const void *>(reinterpret_cast<volatile void *>(&completion_ingress->tail)),
                    sizeof(completion_ingress->tail)
                );
                drained++;
                tail++;

                PTO2AsyncWaitEntry *entry = find_entry_by_token(token);
                if (entry != nullptr) {
                    if (entry->condition_count >= PTO2_MAX_COMPLETIONS_PER_TASK) {
                        error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
                        return drained;
                    }
                    PTO2CompletionCondition &cond = entry->conditions[entry->condition_count++];
                    cond.engine = engine;
                    cond.satisfied = false;
                    cond.counter_addr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
                    cond.expected_value = expected_value;
                    entry->waiting_completion_count++;
                    continue;
                }

                if (pending_completion_count >= PTO2_MAX_PENDING_COMPLETIONS) {
                    error_code = PTO2_ERROR_ASYNC_WAIT_OVERFLOW;
                    return drained;
                }
                PTO2PendingCompletion &pending = pending_completions[pending_completion_count++];
                pending.task_token = token;
                pending.addr = addr;
                pending.expected_value = expected_value;
                pending.engine = engine;
            }
        }
        return drained;
    }

    void absorb_pending_completions_locked(PTO2AsyncWaitEntry &entry) {
        int32_t write = 0;
        for (int32_t i = 0; i < pending_completion_count; i++) {
            if (pending_completions[i].task_token == entry.task_token) {
                if (entry.condition_count < PTO2_MAX_COMPLETIONS_PER_TASK) {
                    PTO2CompletionCondition &cond = entry.conditions[entry.condition_count++];
                    cond.engine = pending_completions[i].engine;
                    cond.satisfied = false;
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
        PTO2AsyncWaitEntry &entry, uint64_t addr, uint32_t expected_value, PTO2AsyncEngine engine, int32_t &error_code
    ) {
        if (entry.condition_count >= PTO2_MAX_COMPLETIONS_PER_TASK) {
            error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
            return false;
        }
        PTO2CompletionCondition &cond = entry.conditions[entry.condition_count++];
        cond.engine = engine;
        cond.satisfied = false;
        cond.counter_addr = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(addr));
        cond.expected_value = expected_value;
        entry.waiting_completion_count++;
        return true;
    }

    RegisterResult register_deferred(
        PTO2TaskSlotState &slot_state, volatile PTO2DeferredCompletionIngressBuffer *ingress, bool normal_done,
        int32_t &error_code
    ) {
        error_code = PTO2_ERROR_NONE;
        PTO2TaskPayload *payload = slot_state.payload;
        if (payload == nullptr || !payload->complete_in_future) {
            return RegisterResult::NotDeferred;
        }

        if (!try_lock()) return RegisterResult::Skipped;

        uint32_t deferred_count = 0;
        if (ingress != nullptr) {
            cache_invalidate_range(
                const_cast<const void *>(reinterpret_cast<volatile void *>(ingress)), PTO2_ALIGN_SIZE
            );
            if (ingress->error_code != PTO2_ERROR_NONE) {
                error_code = ingress->error_code;
                unlock();
                return RegisterResult::Error;
            }
            deferred_count = ingress->count;
        }
        if (deferred_count > PTO2_MAX_COMPLETIONS_PER_TASK) {
            error_code = PTO2_ERROR_ASYNC_REGISTRATION_FAILED;
            unlock();
            return RegisterResult::Error;
        }
        if (deferred_count > 0 && ingress != nullptr) {
            cache_invalidate_range(
                const_cast<const void *>(reinterpret_cast<volatile void *>(ingress->entries)),
                deferred_count * sizeof(PTO2DeferredCompletionEntry)
            );
        }
        PTO2AsyncWaitEntry *entry = find_entry_by_token(slot_state.task->task_id);
        if (entry == nullptr) {
            if (!normal_done && deferred_count == 0) {
                unlock();
                return RegisterResult::Registered;
            }
            if (count >= PTO2_MAX_ASYNC_WAITS) {
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
            volatile PTO2DeferredCompletionEntry *deferred = &ingress->entries[i];
            volatile uint32_t *counter = reinterpret_cast<volatile uint32_t *>(static_cast<uintptr_t>(deferred->addr));
            cache_invalidate_range(
                reinterpret_cast<const void *>(pto2_completion_ingress_cache_line(counter)), sizeof(uint32_t)
            );
            if (!append_condition_locked(
                    *entry, deferred->addr, deferred->expected_value, static_cast<PTO2AsyncEngine>(deferred->engine),
                    error_code
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
    PTO2AsyncPollResult poll_and_complete(
        volatile PTO2CompletionIngressQueue *completion_ingress, PTO2SchedulerState *sched,
        PTO2LocalReadyBuffer *local_bufs, PTO2TaskSlotState **deferred_release_slot_states,
        int32_t &deferred_release_count, int32_t deferred_release_capacity
#if PTO2_SCHED_PROFILING
        ,
        int thread_idx
#endif
    );
};

#endif  // PTO_ASYNC_WAIT_H
