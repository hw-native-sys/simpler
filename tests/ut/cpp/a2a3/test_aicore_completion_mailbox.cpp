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
 * Unit tests for the AICoreCompletionMailbox MPSC push protocol.
 *
 * Pins the lock-free producer/consumer contract used by the FIN-handling
 * scheduler thread to register deferred completions without taking the
 * AsyncWaitList::busy lock. The mailbox is the registration ingress
 * (complete_slot_task pushes here); this test stresses the push/drain path
 * under contention without booting the rest of the runtime.
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
    // ~256KB heap allocation — avoid stack/BSS pressure across tests.
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

// =============================================================================
// Basic push / drain round-trip
// =============================================================================

TEST(AICoreCompletionMailbox, PushConditionThenDrainCreatesEntry) {
    AICoreCompletionMailbox *mb = fresh_mailbox();
    AsyncWaitList wait_list{};

    PTO2TaskId token = make_token(42);
    constexpr uint64_t kAddr = 0xCAFEBABEDEADBEEFull;
    ASSERT_TRUE(
        mb->try_push_condition(token, kAddr, /*expected=*/7, /*engine=*/COMPLETION_ENGINE_ROCE, COMPLETION_TYPE_COUNTER)
    );

    int32_t err = PTO2_ERROR_NONE;
    AsyncWaitList::DrainCompletionSink sink{};
    int32_t drained = wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err);
    EXPECT_EQ(drained, 1);
    EXPECT_EQ(err, PTO2_ERROR_NONE);

    // A CONDITION for an unknown token materializes the entry directly
    // (slot_state stays null until a TASK_NORMAL_DONE sentinel arrives, but
    // the condition is already attached).
    ASSERT_EQ(wait_list.count, 1);
    EXPECT_EQ(wait_list.entries[0].task_token.raw, token.raw);
    EXPECT_EQ(wait_list.entries[0].slot_state, nullptr);
    ASSERT_EQ(wait_list.entries[0].condition_count, 1);
    EXPECT_EQ(wait_list.entries[0].conditions[0].addr, kAddr);
    EXPECT_EQ(wait_list.entries[0].conditions[0].expected_value, 7u);
    EXPECT_EQ(wait_list.entries[0].conditions[0].completion_type, COMPLETION_TYPE_COUNTER);
    EXPECT_EQ(wait_list.entries[0].waiting_completion_count, 1);
    EXPECT_FALSE(wait_list.entries[0].normal_done);

    destroy_mailbox(mb);
}

TEST(AICoreCompletionMailbox, PushNormalDoneCreatesEntryReadyToComplete) {
    AICoreCompletionMailbox *mb = fresh_mailbox();
    AsyncWaitList wait_list{};

    PTO2TaskId token = make_token(99);
    PTO2TaskSlotState dummy_slot{};
    uint64_t slot_addr = reinterpret_cast<uint64_t>(&dummy_slot);
    ASSERT_TRUE(mb->try_push_normal_done(token, slot_addr));

    int32_t err = PTO2_ERROR_NONE;
    AsyncWaitList::DrainCompletionSink sink{};
    int32_t drained = wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err);
    EXPECT_EQ(drained, 1);
    EXPECT_EQ(err, PTO2_ERROR_NONE);

    // A lone TASK_NORMAL_DONE means "no subtask of this task registered a
    // condition" — the consumer creates an entry with waiting_count=0 and
    // normal_done=true so the next poll iteration completes it inline.
    ASSERT_EQ(wait_list.count, 1);
    EXPECT_EQ(wait_list.entries[0].task_token.raw, token.raw);
    EXPECT_EQ(reinterpret_cast<uint64_t>(wait_list.entries[0].slot_state), slot_addr);
    EXPECT_EQ(wait_list.entries[0].condition_count, 0);
    EXPECT_EQ(wait_list.entries[0].waiting_completion_count, 0);
    EXPECT_TRUE(wait_list.entries[0].normal_done);

    destroy_mailbox(mb);
}

TEST(AICoreCompletionMailbox, NormalDoneAttachesToExistingEntry) {
    AICoreCompletionMailbox *mb = fresh_mailbox();
    AsyncWaitList wait_list{};

    // Pre-seed an entry so drain's TASK_NORMAL_DONE branch flips it in place
    // rather than stashing. Exercises the "entry exists already" arm.
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

TEST(AICoreCompletionMailbox, ConditionAttachesToExistingEntry) {
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
    ASSERT_TRUE(mb->try_push_condition(token, kAddr1, /*expected=*/3, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER));
    ASSERT_TRUE(
        mb->try_push_condition(token, kAddr2, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD)
    );

    int32_t err = PTO2_ERROR_NONE;
    AsyncWaitList::DrainCompletionSink sink{};
    ASSERT_EQ(wait_list.drain_aicore_completion_mailbox_locked(mb, sink, err), 2);
    EXPECT_EQ(err, PTO2_ERROR_NONE);

    ASSERT_EQ(wait_list.entries[0].condition_count, 2);
    EXPECT_EQ(wait_list.entries[0].conditions[0].addr, kAddr1);
    EXPECT_EQ(wait_list.entries[0].conditions[0].expected_value, 3u);
    EXPECT_EQ(wait_list.entries[0].conditions[0].completion_type, COMPLETION_TYPE_COUNTER);
    EXPECT_EQ(wait_list.entries[0].conditions[1].addr, kAddr2);
    EXPECT_EQ(wait_list.entries[0].conditions[1].completion_type, COMPLETION_TYPE_SDMA_EVENT_RECORD);
    // SDMA_EVENT_RECORD conditions don't bind counter_addr.
    EXPECT_EQ(wait_list.entries[0].conditions[1].counter_addr, nullptr);
    EXPECT_EQ(wait_list.entries[0].waiting_completion_count, 2);

    destroy_mailbox(mb);
}

// =============================================================================
// Capacity / overflow
// =============================================================================

TEST(AICoreCompletionMailbox, PushReturnsFalseWhenFull) {
    AICoreCompletionMailbox *mb = fresh_mailbox();

    constexpr uint32_t kCap = AICORE_COMPLETION_MAILBOX_CAPACITY;
    PTO2TaskId token = make_token(1);
    for (uint32_t i = 0; i < kCap; i++) {
        ASSERT_TRUE(
            mb->try_push_condition(token, /*addr=*/i, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER)
        );
    }
    EXPECT_FALSE(
        mb->try_push_condition(token, /*addr=*/kCap, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER)
    );
    EXPECT_FALSE(mb->try_push_normal_done(token, /*slot_state_addr=*/0));

    destroy_mailbox(mb);
}

// =============================================================================
// Multi-producer correctness
// =============================================================================

TEST(AICoreCompletionMailbox, MultiProducerNoLossAndPerProducerOrder) {
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
                // Encode (producer, index) into addr to check per-producer
                // ordering on the consumer side.
                uint64_t addr = (static_cast<uint64_t>(p) << 32) | static_cast<uint64_t>(i);
                while (!mb->try_push_condition(
                    token, addr, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER
                )) {
                    std::this_thread::yield();  // sized below capacity so this is defensive
                }
            }
        });
    }
    for (auto &t : producers)
        t.join();

    // Read slots in producer-emit order via the seq protocol (mirrors what
    // drain_aicore_completion_mailbox_locked does internally).
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

    // For each producer p, the i values must appear in increasing order in
    // `observed`. (Cross-producer interleaving is fine.)
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

TEST(AICoreCompletionMailbox, ProducerInterleavedWithDrain) {
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
                token, /*addr=*/i, /*expected=*/0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER
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
