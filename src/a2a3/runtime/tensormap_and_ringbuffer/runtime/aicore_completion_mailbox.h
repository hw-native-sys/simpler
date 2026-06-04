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

// AICPU-only MPSC ring used to convey deferred-completion observations from
// FIN-handling scheduler threads to the dispatch thread. Producers push under
// CAS on `head`; the single consumer (dispatch thread, under AsyncWaitList::
// busy) drains in seq order. Kernel-side code never touches this struct —
// AICore writes go into DeferredCompletionSlab (see
// aicore_completion_mailbox_types.h), which the FIN thread reads, flattens
// into messages here, and forwards.

#define AICORE_COMPLETION_MAILBOX_CAPACITY 4096u
#define AICORE_COMPLETION_MAILBOX_MASK (AICORE_COMPLETION_MAILBOX_CAPACITY - 1u)

static_assert(
    (AICORE_COMPLETION_MAILBOX_CAPACITY & (AICORE_COMPLETION_MAILBOX_CAPACITY - 1u)) == 0,
    "AICORE_COMPLETION_MAILBOX_CAPACITY must be a power of two"
);

// Mailbox message discriminator. CONDITION carries one deferred-completion
// observation flattened from a DeferredCompletionEntry. TASK_NORMAL_DONE
// carries the slot_state pointer in `addr` so the consumer can finalize the
// AsyncWaitEntry.slot_state binding for tasks whose conditions arrived
// before the FIN thread saw mixed_complete. New kinds may be added in future
// without growing the message — the `_pad[5]` slack is reserved for
// kind-specific payload extension.
#define MSG_KIND_CONDITION 0u
#define MSG_KIND_TASK_NORMAL_DONE 1u

struct AICoreCompletionMailboxMessage {
    // Per-slot ready flag. Producer publishes `tail+1` after filling the rest
    // of the slot with a release store; consumer waits for the matching seq
    // value with an acquire load. The release-acquire pair publishes all
    // other fields below as a side effect, so they stay plain.
    std::atomic<uint64_t> seq;
    PTO2TaskId task_token;
    // CONDITION: completion observation addr (counter / SDMA event record).
    // TASK_NORMAL_DONE: PTO2TaskSlotState pointer carried over to the consumer
    //   so it can finalize the AsyncWaitEntry.slot_state binding.
    uint64_t addr;
    uint32_t expected_value;
    uint32_t engine;
    int32_t completion_type;
    uint32_t kind;
    uint32_t _pad[5];
};

static_assert(sizeof(AICoreCompletionMailboxMessage) == PTO2_ALIGN_SIZE, "AICoreCompletionMailboxMessage layout drift");
static_assert(
    sizeof(std::atomic<uint64_t>) == sizeof(uint64_t),
    "std::atomic<uint64_t> must be layout-compatible with uint64_t for the message slot layout to hold"
);
static_assert(
    std::atomic<uint64_t>::is_always_lock_free,
    "AICoreCompletionMailbox requires lock-free uint64_t atomics on every supported target"
);

// POD view of a drained message. `seq` is the ring's publication flag, not
// payload, so try_pop copies out only the fields below (and seq is not even
// copyable — it is a std::atomic).
struct AICoreCompletionMsgView {
    PTO2TaskId task_token{PTO2TaskId::invalid()};
    uint64_t addr{0};
    uint32_t expected_value{0};
    uint32_t engine{0};
    int32_t completion_type{0};
    uint32_t kind{0};
};

struct AICoreCompletionMailbox {
    // head and tail live on their own cache lines so producer CAS contention
    // on head can't false-share with the consumer's tail updates.
    alignas(PTO2_ALIGN_SIZE) std::atomic<uint64_t> head;
    uint8_t _head_pad[PTO2_ALIGN_SIZE - sizeof(uint64_t)];
    alignas(PTO2_ALIGN_SIZE) std::atomic<uint64_t> tail;
    uint8_t _tail_pad[PTO2_ALIGN_SIZE - sizeof(uint64_t)];
    alignas(PTO2_ALIGN_SIZE) AICoreCompletionMailboxMessage entries[AICORE_COMPLETION_MAILBOX_CAPACITY];

    // Cheap, lock-free pending hint. Callers may invoke this outside the
    // consumer lock; a stale answer only over/under-triggers a drain attempt.
    bool has_pending() { return tail.load(std::memory_order_acquire) < head.load(std::memory_order_acquire); }

    // MPSC push for a CONDITION message. Returns false when the ring is full
    // (head - tail >= CAPACITY); caller should SPIN_WAIT_HINT and retry.
    // Lock-free: CAS the shared head to claim a slot, write the fields, then
    // release-store seq so the single consumer observes the publication.
    //
    // The head CAS is relaxed: head is a pure ticket counter and carries no
    // data to the consumer — publication is solely the seq release-store, and
    // slot-reuse safety rests on the acquire load of tail. The relaxed failure
    // order is likewise sufficient since a lost CAS just re-reads head and
    // retries. compare_exchange_weak is used because this loop already re-reads
    // head and re-checks fullness, so masking LL/SC spurious failures (what
    // _strong adds on aarch64) would only be a redundant inner retry.
    //
    // Safe to call concurrently from any number of producers; structurally
    // independent of the AsyncWaitList::busy lock.
    bool try_push_condition(
        PTO2TaskId task_token, uint64_t addr, uint32_t expected_value, uint32_t engine, int32_t completion_type
    ) {
        while (true) {
            uint64_t h = head.load(std::memory_order_relaxed);
            uint64_t t = tail.load(std::memory_order_acquire);
            if (h - t >= AICORE_COMPLETION_MAILBOX_CAPACITY) return false;
            uint64_t new_head = h + 1;
            if (head.compare_exchange_weak(h, new_head, std::memory_order_relaxed, std::memory_order_relaxed)) {
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

    // MPSC push for a TASK_NORMAL_DONE sentinel. Carries the PTO2TaskSlotState
    // pointer in the `addr` field so the consumer can finish binding the
    // AsyncWaitEntry.slot_state without going back to the FIN-handling thread.
    bool try_push_normal_done(PTO2TaskId task_token, uint64_t slot_state_addr) {
        while (true) {
            uint64_t h = head.load(std::memory_order_relaxed);
            uint64_t t = tail.load(std::memory_order_acquire);
            if (h - t >= AICORE_COMPLETION_MAILBOX_CAPACITY) return false;
            uint64_t new_head = h + 1;
            if (head.compare_exchange_weak(h, new_head, std::memory_order_relaxed, std::memory_order_relaxed)) {
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

    // Single-consumer transport-level dequeue (caller holds the consumer lock).
    // Returns false at the first not-yet-published slot (gap) or when empty;
    // otherwise copies the next message in tail order into `out`, advances
    // tail, and returns true. tail is consumer-only-written (relaxed read);
    // head bounds the scan (relaxed); the seq acquire is the real publication
    // gate; the tail release publishes "slot free" to reusing producers.
    bool try_pop(AICoreCompletionMsgView &out) {
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

static_assert(
    sizeof(AICoreCompletionMailbox) % PTO2_ALIGN_SIZE == 0, "AICoreCompletionMailbox size must be cache-line aligned"
);

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_AICORE_COMPLETION_MAILBOX_H_
