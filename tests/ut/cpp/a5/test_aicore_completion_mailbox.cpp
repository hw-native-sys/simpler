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
 * Unit tests for the a5 AICoreCompletionMailbox MPSC push protocol.
 *
 * Mirrors tests/ut/cpp/a2a3/test_aicore_completion_mailbox.cpp for the a5
 * completion mailbox and async wait-list path.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <thread>
#include <unordered_map>
#include <vector>

#include "aicore_completion_mailbox.h"
#include "pto_async_wait.h"
#include "scheduler/pto_scheduler.h"

namespace {

PTO2TaskId make_token(uint32_t local) { return PTO2TaskId::make(/*ring=*/0, local); }

AICoreCompletionMailbox *fresh_mailbox() {
    void *raw = ::operator new(sizeof(AICoreCompletionMailbox));
    auto *mb = new (raw) AICoreCompletionMailbox{};
    std::memset(mb, 0, sizeof(*mb));
    return mb;
}

void destroy_mailbox(AICoreCompletionMailbox *mb) {
    mb->~AICoreCompletionMailbox();
    ::operator delete(mb);
}

}  // namespace

TEST(A5AICoreCompletionMailbox, PushConditionThenDrainCreatesEntry) {
    AICoreCompletionMailbox *mb = fresh_mailbox();
    AsyncWaitList wait_list{};

    PTO2TaskId token = make_token(42);
    constexpr uint64_t kAddr = 0xCAFEBABEDEADBEEFull;
    constexpr uint64_t kCookie = 0x123456789ABCDEF0ull;
    ASSERT_TRUE(mb->try_push_condition(
        token, kAddr, kCookie, /*expected=*/7, /*engine=*/COMPLETION_ENGINE_ROCE, COMPLETION_TYPE_COUNTER
    ));

    int32_t err = PTO2_ERROR_NONE;
    AsyncWaitList::DrainCompletionSink sink{};
    int32_t drained = wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err);
    EXPECT_EQ(drained, 1);
    EXPECT_EQ(err, PTO2_ERROR_NONE);

    ASSERT_EQ(wait_list.count, 1);
    EXPECT_EQ(wait_list.entries[0].task_token.raw, token.raw);
    EXPECT_EQ(wait_list.entries[0].slot_state, nullptr);
    ASSERT_EQ(wait_list.entries[0].condition_count, 1);
    EXPECT_EQ(wait_list.entries[0].conditions[0].expected_value, 7u);
    EXPECT_EQ(wait_list.entries[0].conditions[0].backend_cookie, kCookie);
    EXPECT_EQ(reinterpret_cast<uint64_t>(wait_list.entries[0].conditions[0].counter_addr), kAddr);
    EXPECT_EQ(wait_list.entries[0].waiting_completion_count, 1);
    EXPECT_FALSE(wait_list.entries[0].normal_done);

    destroy_mailbox(mb);
}

TEST(A5AICoreCompletionMailbox, PushNormalDoneCreatesEntryReadyToComplete) {
    AICoreCompletionMailbox *mb = fresh_mailbox();
    AsyncWaitList wait_list{};

    PTO2TaskId token = make_token(99);
    PTO2TaskSlotState dummy_slot{};
    uint64_t slot_addr = reinterpret_cast<uint64_t>(&dummy_slot);
    ASSERT_TRUE(mb->try_push_normal_done(token, slot_addr));

    int32_t err = PTO2_ERROR_NONE;
    AsyncWaitList::DrainCompletionSink sink{};
    ASSERT_EQ(wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err), 1);
    EXPECT_EQ(err, PTO2_ERROR_NONE);

    ASSERT_EQ(wait_list.count, 1);
    EXPECT_EQ(wait_list.entries[0].task_token.raw, token.raw);
    EXPECT_EQ(reinterpret_cast<uint64_t>(wait_list.entries[0].slot_state), slot_addr);
    EXPECT_EQ(wait_list.entries[0].condition_count, 0);
    EXPECT_EQ(wait_list.entries[0].waiting_completion_count, 0);
    EXPECT_TRUE(wait_list.entries[0].normal_done);

    destroy_mailbox(mb);
}

TEST(A5AICoreCompletionMailbox, NormalDoneAttachesToExistingEntry) {
    AICoreCompletionMailbox *mb = fresh_mailbox();
    AsyncWaitList wait_list{};

    PTO2TaskId token = make_token(7);
    wait_list.entries[0].task_token = token;
    wait_list.entries[0].slot_state = nullptr;
    wait_list.entries[0].condition_count = 0;
    wait_list.entries[0].waiting_completion_count = 0;
    wait_list.entries[0].normal_done = false;
    wait_list.count = 1;

    PTO2TaskSlotState dummy_slot{};
    uint64_t slot_addr = reinterpret_cast<uint64_t>(&dummy_slot);
    ASSERT_TRUE(mb->try_push_normal_done(token, slot_addr));

    int32_t err = PTO2_ERROR_NONE;
    AsyncWaitList::DrainCompletionSink sink{};
    ASSERT_EQ(wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err), 1);
    EXPECT_EQ(err, PTO2_ERROR_NONE);

    EXPECT_TRUE(wait_list.entries[0].normal_done);
    EXPECT_EQ(reinterpret_cast<uint64_t>(wait_list.entries[0].slot_state), slot_addr);

    destroy_mailbox(mb);
}

TEST(A5AICoreCompletionMailbox, ConditionAttachesToExistingEntry) {
    AICoreCompletionMailbox *mb = fresh_mailbox();
    AsyncWaitList wait_list{};

    PTO2TaskId token = make_token(15);
    wait_list.entries[0].task_token = token;
    wait_list.entries[0].slot_state = nullptr;
    wait_list.entries[0].condition_count = 0;
    wait_list.entries[0].waiting_completion_count = 0;
    wait_list.entries[0].normal_done = false;
    wait_list.count = 1;

    constexpr uint64_t kAddr1 = 0x1000;
    constexpr uint64_t kAddr2 = 0x2000;
    ASSERT_TRUE(
        mb->try_push_condition(token, kAddr1, 0, /*expected=*/3, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER)
    );
    ASSERT_TRUE(
        mb->try_push_condition(token, kAddr2, 0, /*expected=*/9, COMPLETION_ENGINE_ROCE, COMPLETION_TYPE_COUNTER)
    );

    int32_t err = PTO2_ERROR_NONE;
    AsyncWaitList::DrainCompletionSink sink{};
    ASSERT_EQ(wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err), 2);
    EXPECT_EQ(err, PTO2_ERROR_NONE);

    ASSERT_EQ(wait_list.entries[0].condition_count, 2);
    EXPECT_EQ(reinterpret_cast<uint64_t>(wait_list.entries[0].conditions[0].counter_addr), kAddr1);
    EXPECT_EQ(wait_list.entries[0].conditions[0].expected_value, 3u);
    EXPECT_EQ(reinterpret_cast<uint64_t>(wait_list.entries[0].conditions[1].counter_addr), kAddr2);
    EXPECT_EQ(wait_list.entries[0].conditions[1].expected_value, 9u);
    EXPECT_EQ(wait_list.entries[0].waiting_completion_count, 2);

    destroy_mailbox(mb);
}

TEST(A5AICoreCompletionMailbox, PushReturnsFalseWhenFull) {
    AICoreCompletionMailbox *mb = fresh_mailbox();

    constexpr uint32_t kCap = AICORE_COMPLETION_MAILBOX_CAPACITY;
    PTO2TaskId token = make_token(1);
    for (uint32_t i = 0; i < kCap; i++) {
        ASSERT_TRUE(mb->try_push_condition(
            token, /*addr=*/0x10000 + i, /*backend_cookie=*/0, /*expected=*/0, COMPLETION_ENGINE_SDMA,
            COMPLETION_TYPE_COUNTER
        ));
    }
    EXPECT_FALSE(mb->try_push_condition(
        token, /*addr=*/0x20000, /*backend_cookie=*/0, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER
    ));
    EXPECT_FALSE(mb->try_push_normal_done(token, /*slot_state_addr=*/0));

    destroy_mailbox(mb);
}

TEST(A5AICoreCompletionMailbox, MultiProducerNoLossAndPerProducerOrder) {
    AICoreCompletionMailbox *mb = fresh_mailbox();

    constexpr int kProducers = 8;
    constexpr int kPerProducer = 256;
    static_assert(kProducers * kPerProducer < AICORE_COMPLETION_MAILBOX_CAPACITY);

    std::vector<std::thread> producers;
    producers.reserve(kProducers);
    for (int p = 0; p < kProducers; p++) {
        producers.emplace_back([mb, p]() {
            PTO2TaskId token = make_token(static_cast<uint32_t>(p));
            for (int i = 0; i < kPerProducer; i++) {
                uint64_t addr = (static_cast<uint64_t>(p) << 32) | static_cast<uint64_t>(i);
                while (!mb->try_push_condition(
                    token, addr, /*backend_cookie=*/0, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER
                )) {
                    std::this_thread::yield();
                }
            }
        });
    }
    for (auto &t : producers)
        t.join();

    std::vector<std::pair<uint32_t, uint32_t>> observed;
    observed.reserve(kProducers * kPerProducer);
    uint64_t head_total = static_cast<uint64_t>(kProducers) * kPerProducer;
    for (uint64_t tail = 0; tail < head_total; tail++) {
        AICoreCompletionMailboxMessage *slot = &mb->entries[tail & AICORE_COMPLETION_MAILBOX_MASK];
        uint64_t expected_seq = tail + 1;
        uint64_t seq = slot->seq.load(std::memory_order_acquire);
        ASSERT_EQ(seq, expected_seq) << "missing publish at tail=" << tail;
        EXPECT_EQ(slot->kind, MSG_KIND_CONDITION);
        uint64_t addr = slot->addr;
        observed.emplace_back(static_cast<uint32_t>(addr >> 32), static_cast<uint32_t>(addr & 0xFFFFFFFFu));
    }

    EXPECT_EQ(observed.size(), static_cast<size_t>(kProducers) * kPerProducer);

    std::unordered_map<uint32_t, int> last_seen;
    for (auto &kv : observed) {
        auto it = last_seen.find(kv.first);
        if (it == last_seen.end()) {
            EXPECT_EQ(kv.second, 0u) << "producer " << kv.first << " first emit not index 0";
            last_seen[kv.first] = 0;
        } else {
            EXPECT_EQ(kv.second, static_cast<uint32_t>(it->second + 1)) << "producer " << kv.first << " out of order";
            it->second++;
        }
    }
    for (int p = 0; p < kProducers; p++) {
        EXPECT_EQ(last_seen[static_cast<uint32_t>(p)], kPerProducer - 1) << "producer " << p << " missing tail emits";
    }

    destroy_mailbox(mb);
}

// =============================================================================
// Push interleaved with drain (single producer thread + single drain thread,
// stress-tests seq publish/observe round-trips across capacity wrap).
// =============================================================================

TEST(A5AICoreCompletionMailbox, ProducerInterleavedWithDrain) {
    AICoreCompletionMailbox *mb = fresh_mailbox();
    AsyncWaitList wait_list{};
    // Pre-seed the entry for `token` so the drained CONDITIONs append to a
    // known slot the final assertions can read.
    PTO2TaskId token = make_token(5);
    wait_list.entries[0].task_token = token;
    wait_list.entries[0].slot_state = nullptr;
    wait_list.entries[0].condition_count = 0;
    wait_list.entries[0].waiting_completion_count = 0;
    wait_list.entries[0].normal_done = false;
    wait_list.count = 1;

    // entry.conditions[] is bounded at MAX_COMPLETIONS_PER_TASK = 64, so cap
    // the producer to that. Pin down: 64 pushes interleaved with drain do
    // not lose any condition and waiting_completion_count tracks the total.
    constexpr int kTotal = MAX_COMPLETIONS_PER_TASK;

    std::atomic<int> pushed{0};
    std::atomic<bool> drainer_stop{false};
    std::thread producer([&]() {
        for (int i = 0; i < kTotal; i++) {
            while (!mb->try_push_condition(
                token, /*addr=*/i, /*backend_cookie=*/0, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER
            )) {
                std::this_thread::yield();
            }
            pushed.fetch_add(1, std::memory_order_release);
        }
    });
    std::thread drainer([&]() {
        while (!drainer_stop.load(std::memory_order_acquire)) {
            int32_t err = PTO2_ERROR_NONE;
            AsyncWaitList::DrainCompletionSink sink{};
            (void)wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err);
            ASSERT_EQ(err, PTO2_ERROR_NONE);
            std::this_thread::yield();
        }
    });
    producer.join();
    // Stop and join the drain thread first so the final pass runs as the sole
    // consumer — drain_aicore_completion_mailbox_locked is single-consumer
    // (MPSC), so two concurrent drainers would race on tail/entries and could
    // double-append past MAX_COMPLETIONS_PER_TASK.
    drainer_stop.store(true, std::memory_order_release);
    drainer.join();
    // Final drain pass to consume any in-flight messages the drainer left
    // after it observed the stop flag.
    int32_t err = PTO2_ERROR_NONE;
    AsyncWaitList::DrainCompletionSink sink{};
    (void)wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err);

    EXPECT_EQ(pushed.load(), kTotal);
    EXPECT_EQ(wait_list.entries[0].condition_count, kTotal);
    EXPECT_EQ(wait_list.entries[0].waiting_completion_count, kTotal);

    destroy_mailbox(mb);
}

// =============================================================================
// Cross-run reuse without zeroing entries[]. The pooled-arena fast path keeps
// the mailbox's head/tail/seq monotonic across boots and resets per-boot with
// tail := head (NOT a 256KB memset). This pins down the two invariants that
// makes safe:
//   1. After a full capacity wrap every slot holds a STALE seq from a prior
//      generation; a fresh push's release-store of the new (larger) seq must be
//      the only value try_pop accepts — the stale seq must never false-match.
//   2. tail := head discards messages an error-aborted prior run left undrained
//      (head > tail) so the next consumer starts empty.
// =============================================================================

TEST(A5AICoreCompletionMailbox, MonotonicSeqSurvivesCapacityWrapWithoutZeroing) {
    AICoreCompletionMailbox *mb = fresh_mailbox();

    // Drive head/tail one full capacity past the wrap point so every physical
    // slot has been published once and now carries a stale seq in [1, CAPACITY].
    // Each push is drained immediately so the ring never reports full.
    PTO2TaskId token = make_token(7);
    AICoreCompletionMsgView msg;
    for (uint64_t i = 0; i < AICORE_COMPLETION_MAILBOX_CAPACITY + 17; i++) {
        ASSERT_TRUE(mb->try_push_condition(
            token, /*addr=*/i, /*backend_cookie=*/0, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER
        ));
        ASSERT_TRUE(mb->try_pop(msg));
        EXPECT_EQ(msg.addr, i);
    }
    ASSERT_FALSE(mb->try_pop(msg));  // drained: tail == head

    // Slot 0 now physically holds seq=1 (from the very first push). The next
    // push reuses it but publishes seq = head+1 (well past CAPACITY). try_pop
    // must accept only that fresh value — proving stale seq cannot be replayed.
    const uint64_t addr_after_wrap = 0xABCD1234ull;
    ASSERT_TRUE(mb->try_push_condition(
        token, addr_after_wrap, /*backend_cookie=*/0, /*expected=*/3, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER
    ));
    ASSERT_TRUE(mb->try_pop(msg));
    EXPECT_EQ(msg.addr, addr_after_wrap);
    EXPECT_EQ(msg.expected_value, 3u);
    ASSERT_FALSE(mb->try_pop(msg));

    destroy_mailbox(mb);
}

TEST(A5AICoreCompletionMailbox, TailEqualsHeadDiscardsUndrainedLeftovers) {
    AICoreCompletionMailbox *mb = fresh_mailbox();

    // Simulate a prior run that pushed messages but aborted before draining
    // them all (error path): head advances, tail lags.
    PTO2TaskId token = make_token(11);
    for (int i = 0; i < 5; i++) {
        ASSERT_TRUE(mb->try_push_condition(
            token, /*addr=*/static_cast<uint64_t>(i), /*backend_cookie=*/0, /*expected=*/0, COMPLETION_ENGINE_SDMA,
            COMPLETION_TYPE_COUNTER
        ));
    }
    ASSERT_TRUE(mb->has_pending());

    // Per-boot reset performed by the executor: tail := head (single-threaded,
    // no producers running). The leftovers are discarded, not replayed.
    mb->tail.store(mb->head.load(std::memory_order_acquire), std::memory_order_release);
    EXPECT_FALSE(mb->has_pending());
    AICoreCompletionMsgView msg;
    EXPECT_FALSE(mb->try_pop(msg));

    // The channel is fully usable afterward: a fresh push drains cleanly with
    // its own seq, unaffected by the discarded generation.
    ASSERT_TRUE(mb->try_push_normal_done(token, /*slot_state_addr=*/0xFEEDull));
    ASSERT_TRUE(mb->try_pop(msg));
    EXPECT_EQ(msg.kind, MSG_KIND_TASK_NORMAL_DONE);
    EXPECT_EQ(msg.addr, 0xFEEDull);

    destroy_mailbox(mb);
}
