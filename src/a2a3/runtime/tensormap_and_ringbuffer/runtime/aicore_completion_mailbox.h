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

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_AICORE_COMPLETION_MAILBOX_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_AICORE_COMPLETION_MAILBOX_H_

#include <atomic>
#include <cstdint>

#include "aicore_completion_mailbox_types.h"
#include "pto_constants.h"
#include "pto_task_id.h"

#define AICORE_COMPLETION_MAILBOX_CAPACITY 4096u
#define AICORE_COMPLETION_MAILBOX_MASK (AICORE_COMPLETION_MAILBOX_CAPACITY - 1u)

static_assert((AICORE_COMPLETION_MAILBOX_CAPACITY & (AICORE_COMPLETION_MAILBOX_CAPACITY - 1u)) == 0, "AICORE_COMPLETION_MAILBOX_CAPACITY must be a power of two");

// Mailbox message discriminator. CONDITION carries one deferred-completion
// observation flattened from a DeferredCompletionEntry. TASK_NORMAL_DONE
// carries the slot_state pointer in `addr` so the consumer can finalize the
// AsyncWaitEntry.slot_state binding for tasks whose conditions arrived
// before the FIN thread saw task_complete. New kinds may be added in future
// without growing the message — the `_pad[5]` slack is reserved for
// kind-specific payload extension.
#define MSG_KIND_CONDITION 0u
#define MSG_KIND_TASK_NORMAL_DONE 1u

struct AICoreCompletionMailboxMessage
{
    std::atomic<uint64_t> seq;
    PTO2TaskId task_token;
    uint64_t addr;
    uint32_t expected_value;
    uint32_t engine;
    int32_t completion_type;
    uint32_t kind;
    uint32_t _pad[5];
};

static_assert(sizeof(AICoreCompletionMailboxMessage) == PTO2_ALIGN_SIZE, "AICoreCompletionMailboxMessage layout drift");
static_assert(sizeof(std::atomic<uint64_t>) == sizeof(uint64_t), "std::atomic<uint64_t> must be layout-compatible with uint64_t for the message slot layout to hold");
static_assert(std::atomic<uint64_t>::is_always_lock_free, "AICoreCompletionMailbox requires lock-free uint64_t atomics on every supported target");

struct AICoreCompletionMsgView
{
    PTO2TaskId task_token{PTO2TaskId::invalid()};
    uint64_t addr{0};
    uint32_t expected_value{0};
    uint32_t engine{0};
    int32_t completion_type{0};
    uint32_t kind{0};
};

struct AICoreCompletionMailbox
{
    // head and tail live on their own cache lines so producer CAS contention
    // on head can't false-share with the consumer's tail updates.
    alignas(PTO2_ALIGN_SIZE) std::atomic<uint64_t> head;
    uint8_t _head_pad[PTO2_ALIGN_SIZE - sizeof(uint64_t)];
    alignas(PTO2_ALIGN_SIZE) std::atomic<uint64_t> tail;
    uint8_t _tail_pad[PTO2_ALIGN_SIZE - sizeof(uint64_t)];
    alignas(PTO2_ALIGN_SIZE) AICoreCompletionMailboxMessage entries[AICORE_COMPLETION_MAILBOX_CAPACITY];

    // Cheap, lock-free pending hint. Callers may invoke this outside the
    // consumer lock; a stale answer only over/under-triggers a drain attempt.
    bool has_pending()
    {
        return tail.load(std::memory_order_acquire) < head.load(std::memory_order_acquire);
    }

    bool try_push_condition(PTO2TaskId task_token, uint64_t addr, uint32_t expected_value, uint32_t engine, int32_t completion_type)
    {
        while (true)
        {
            uint64_t h = head.load(std::memory_order_relaxed);
            uint64_t t = tail.load(std::memory_order_acquire);
            if (h - t >= AICORE_COMPLETION_MAILBOX_CAPACITY) return false;
            uint64_t new_head = h + 1;
            if (head.compare_exchange_weak(h, new_head, std::memory_order_relaxed, std::memory_order_relaxed))
            {
                AICoreCompletionMailboxMessage *slot = &entries[h & AICORE_COMPLETION_MAILBOX_MASK];
                slot->task_token.raw = task_token.raw;
                slot->addr = addr;
                slot->expected_value = expected_value;
                slot->engine = engine;
                slot->completion_type = completion_type;
                slot->kind = MSG_KIND_CONDITION;
                slot->seq.store(new_head, std::memory_order_release);
                return true;
            }
            // CAS lost: another producer claimed the slot, retry with refreshed head.
        }
    }

    bool try_push_normal_done(PTO2TaskId task_token, uint64_t slot_state_addr)
    {
        while (true)
        {
            uint64_t h = head.load(std::memory_order_relaxed);
            uint64_t t = tail.load(std::memory_order_acquire);
            if (h - t >= AICORE_COMPLETION_MAILBOX_CAPACITY) return false;
            uint64_t new_head = h + 1;
            if (head.compare_exchange_weak(h, new_head, std::memory_order_relaxed, std::memory_order_relaxed))
            {
                AICoreCompletionMailboxMessage *slot = &entries[h & AICORE_COMPLETION_MAILBOX_MASK];
                slot->task_token.raw = task_token.raw;
                slot->addr = slot_state_addr;
                slot->expected_value = 0;
                slot->engine = 0;
                slot->completion_type = 0;
                slot->kind = MSG_KIND_TASK_NORMAL_DONE;
                slot->seq.store(new_head, std::memory_order_release);
                return true;
            }
        }
    }

    bool try_pop(AICoreCompletionMsgView &out)
    {
        uint64_t t = tail.load(std::memory_order_relaxed);
        uint64_t h = head.load(std::memory_order_relaxed);
        if (t >= h) return false;
        AICoreCompletionMailboxMessage *slot = &entries[t & AICORE_COMPLETION_MAILBOX_MASK];
        if (slot->seq.load(std::memory_order_acquire) != t + 1) return false;
        out.task_token.raw = slot->task_token.raw;
        out.addr = slot->addr;
        out.expected_value = slot->expected_value;
        out.engine = slot->engine;
        out.completion_type = slot->completion_type;
        out.kind = slot->kind;
        tail.store(t + 1, std::memory_order_release);
        return true;
    }
};

static_assert(sizeof(AICoreCompletionMailbox) % PTO2_ALIGN_SIZE == 0, "AICoreCompletionMailbox size must be cache-line aligned");

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_AICORE_COMPLETION_MAILBOX_H_
