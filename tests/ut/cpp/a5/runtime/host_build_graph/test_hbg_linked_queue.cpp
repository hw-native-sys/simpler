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

#include <gtest/gtest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <thread>
#include <vector>

#include "scheduler/scheduler_linked_queue.h"

namespace {
struct LinkedQueueFixture {
    SchedulerLinkedQueue control{};
    std::vector<int64_t> links;
    SchedulerLinkedQueueView queue;

    explicit LinkedQueueFixture(uint64_t tasks) :
        links(tasks + 1, SCHEDULER_TASK_REF_INVALID),
        queue{&control, links.data(), tasks, tasks + 1, static_cast<int64_t>(tasks)} {
        control.head = control.tail = queue.sentinel;
    }

    SchedulerLinkedBatch prepare(std::initializer_list<int64_t> refs) {
        SchedulerLinkedBatch batch{};
        EXPECT_EQ(scheduler_linked_prepare(queue, refs.begin(), refs.size(), &batch), SchedulerLinkedResult::SUCCESS);
        return batch;
    }

    SchedulerLinkedResult pop(int64_t *ref, uint32_t budget = 2) {
        int64_t refs[1]{};
        uint32_t count = 0;
        const auto result = scheduler_linked_try_pop_batch(queue, refs, 1, &count, budget);
        *ref = refs[0];
        return result;
    }
};

TEST(SchedulerLinkedQueueTest, PreservesOrderAcrossBatchesAndEmptyTransitions) {
    LinkedQueueFixture fixture(6);
    int64_t ref = 99;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
    auto first = fixture.prepare({4, 1, 3});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &first), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(first.count, 0u);
    for (int64_t expected : {4, 1, 3}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
    auto second = fixture.prepare({0, 5, 2});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &second), SchedulerLinkedResult::SUCCESS);
    for (int64_t expected : {0, 5, 2}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
    EXPECT_EQ(fixture.control.head, 2);
    EXPECT_EQ(fixture.links[fixture.queue.sentinel], 4);
}

TEST(SchedulerLinkedQueueTest, PausedPrivateProducerDoesNotBlockLaterPublication) {
    LinkedQueueFixture fixture(4);
    auto paused = fixture.prepare({0, 1});
    auto active = fixture.prepare({2, 3});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &active), SchedulerLinkedResult::SUCCESS);
    int64_t ref = -1;
    for (int64_t expected : {2, 3}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &paused), SchedulerLinkedResult::SUCCESS);
    for (int64_t expected : {0, 1}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
}

TEST(SchedulerLinkedQueueTest, ConsumerCanPassALaggingTailAndProducerCanHelpIt) {
    LinkedQueueFixture fixture(5);
    const auto paused = fixture.prepare({0, 1, 2});
    ASSERT_EQ(scheduler_gm_compare_exchange(fixture.links[5], int64_t{-1}, paused.first), -1);
    int64_t ref = -1;
    for (int64_t expected : {0, 1, 2}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(fixture.control.tail, 5);
    auto following = fixture.prepare({3, 4});
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &following, 2), SchedulerLinkedResult::RETRY);
    EXPECT_EQ(following.count, 2u);
    EXPECT_EQ(fixture.control.tail, 1);
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &following, 2), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(scheduler_gm_compare_exchange(fixture.control.tail, int64_t{5}, paused.last), 4);
    for (int64_t expected : {3, 4}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
}

TEST(SchedulerLinkedQueueTest, RetryRetainsExactlyThePreparedPrivateBatch) {
    LinkedQueueFixture fixture(2);
    auto batch = fixture.prepare({1, 0});
    const auto links = fixture.links;
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &batch, 0), SchedulerLinkedResult::RETRY);
    EXPECT_EQ(batch.first, 1);
    EXPECT_EQ(batch.last, 0);
    EXPECT_EQ(batch.count, 2u);
    EXPECT_EQ(fixture.links, links);
    EXPECT_EQ(fixture.control.head, 2);
    EXPECT_EQ(fixture.control.tail, 2);
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &batch), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(batch.count, 0u);
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &batch), SchedulerLinkedResult::INVALID);
}

TEST(SchedulerLinkedQueueTest, BatchClaimRespectsCreditAndIncludesOnlyPublishedPrefix) {
    LinkedQueueFixture fixture(3);
    auto first = fixture.prepare({2});
    auto pending = fixture.prepare({0, 1});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &first), SchedulerLinkedResult::SUCCESS);
    int64_t refs[2]{};
    uint32_t count = 9;
    EXPECT_EQ(scheduler_linked_try_pop_batch(fixture.queue, refs, 0, &count), SchedulerLinkedResult::NOT_READY);
    EXPECT_EQ(count, 0u);
    ASSERT_EQ(scheduler_linked_try_pop_batch(fixture.queue, refs, 2, &count), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(refs[0], 2);
    EXPECT_EQ(refs[1], -1);
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &pending), SchedulerLinkedResult::SUCCESS);
    ASSERT_EQ(scheduler_linked_try_pop_batch(fixture.queue, refs, 1, &count), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(refs[0], 0);
    ASSERT_EQ(scheduler_linked_try_pop_batch(fixture.queue, refs, 2, &count), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(refs[0], 1);
}

TEST(SchedulerLinkedQueueTest, StalePrefixCannotClaimAlreadyConsumedWork) {
    LinkedQueueFixture fixture(3);
    auto batch = fixture.prepare({0, 1, 2});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &batch), SchedulerLinkedResult::SUCCESS);
    const int64_t stale_head = scheduler_gm_query(fixture.control.head);
    const int64_t stale_first = scheduler_gm_query(fixture.links[stale_head]);
    int64_t refs[2]{};
    uint32_t count = 0;
    ASSERT_EQ(scheduler_linked_try_pop_batch(fixture.queue, refs, 2, &count), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(count, 2u);
    EXPECT_NE(scheduler_gm_compare_exchange(fixture.control.head, stale_head, stale_first), stale_head);
    int64_t ref = -1;
    ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(ref, 2);
}

TEST(SchedulerLinkedQueueTest, InvalidPreparationDoesNotModifyLinks) {
    LinkedQueueFixture fixture(3);
    const auto before = fixture.links;
    for (const std::vector<int64_t> refs : {std::vector<int64_t>{0, 3}, {0, 0}, {-1}, {}}) {
        SchedulerLinkedBatch batch{};
        EXPECT_EQ(
            scheduler_linked_prepare(fixture.queue, refs.data(), refs.size(), &batch), SchedulerLinkedResult::INVALID
        );
        EXPECT_EQ(batch.count, 0u);
        EXPECT_EQ(fixture.links, before);
    }
}

TEST(SchedulerLinkedQueueTest, ZeroCapacityAndInvalidViewsCannotAccessTaskStorage) {
    LinkedQueueFixture fixture(0);
    int64_t ref = 0;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
    EXPECT_EQ(ref, -1);
    fixture.queue.links = nullptr;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::INVALID);
    fixture.queue.links = fixture.links.data();
    fixture.queue.sentinel = 1;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::INVALID);
    fixture.queue.sentinel = 0;
    fixture.control.head = 1;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::INVALID);
}

TEST(SchedulerLinkedQueueTest, CorruptLinkDoesNotMoveHeadOrPublishPrivateBatch) {
    LinkedQueueFixture fixture(2);
    auto batch = fixture.prepare({0});
    fixture.links[2] = 3;
    int64_t ref = -1;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::INVALID);
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &batch), SchedulerLinkedResult::INVALID);
    EXPECT_EQ(fixture.control.head, 2);
    EXPECT_EQ(batch.count, 1u);
    fixture.links[2] = 2;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::INVALID);
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &batch), SchedulerLinkedResult::INVALID);
}

TEST(SchedulerLinkedQueueTest, LastTaskHasOneWinner) {
    LinkedQueueFixture fixture(1);
    auto batch = fixture.prepare({0});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &batch), SchedulerLinkedResult::SUCCESS);
    std::atomic<bool> start{false};
    std::atomic<uint32_t> wins{0}, failures{0};
    std::vector<std::thread> threads;
    for (uint32_t index = 0; index < 16; ++index) {
        threads.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {}
            int64_t ref = -1;
            const auto result = fixture.pop(&ref);
            if (result == SchedulerLinkedResult::SUCCESS) {
                ++wins;
                if (ref != 0) ++failures;
            } else if (result != SchedulerLinkedResult::NOT_READY && result != SchedulerLinkedResult::RETRY) {
                ++failures;
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(wins.load(), 1u);
    EXPECT_EQ(failures.load(), 0u);
}

TEST(SchedulerLinkedQueueTest, ConcurrentMixedBatchSizesPreservePayloadAndExactlyOnceClaims) {
    constexpr uint32_t producers = 8, consumers = 12, per_producer = 256;
    constexpr uint32_t tasks = producers * per_producer;
    for (uint32_t credit : {1u, 2u}) {
        LinkedQueueFixture fixture(tasks);
        std::vector<uint64_t> payload(tasks, 0);
        std::vector<std::atomic<uint32_t>> claims(tasks);
        for (auto &claim : claims)
            claim.store(0);
        std::atomic<bool> start{false};
        std::atomic<uint32_t> consumed{0}, failures{0};
        const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
        std::vector<std::thread> threads;
        for (uint32_t producer = 0; producer < producers; ++producer) {
            threads.emplace_back([&, producer] {
                while (!start.load(std::memory_order_acquire)) {}
                const uint32_t batch_size = 1u << (producer % 4);
                SchedulerLinkedBatch pending{};
                for (uint32_t offset = 0; offset < per_producer && failures.load() == 0; offset += batch_size) {
                    int64_t refs[8]{};
                    for (uint32_t index = 0; index < batch_size; ++index) {
                        const uint32_t ref = producer * per_producer + offset + index;
                        refs[index] = ref;
                        payload[ref] = UINT64_C(0xfeed0000) + ref;
                    }
                    SchedulerLinkedBatch batch{};
                    if (scheduler_linked_prepare(fixture.queue, refs, batch_size, &batch) !=
                        SchedulerLinkedResult::SUCCESS) {
                        ++failures;
                        return;
                    }
                    if (scheduler_linked_append_private(fixture.queue, &pending, &batch) !=
                        SchedulerLinkedResult::SUCCESS) {
                        ++failures;
                        return;
                    }
                    if ((offset / batch_size) % 2 == 0 && offset + batch_size < per_producer) continue;
                    while (failures.load() == 0) {
                        const auto result = scheduler_linked_try_push(fixture.queue, &pending, 2);
                        if (result == SchedulerLinkedResult::SUCCESS) break;
                        if (result != SchedulerLinkedResult::RETRY || std::chrono::steady_clock::now() >= deadline) {
                            ++failures;
                            return;
                        }
                    }
                }
            });
        }
        for (uint32_t consumer = 0; consumer < consumers; ++consumer) {
            threads.emplace_back([&] {
                while (!start.load(std::memory_order_acquire)) {}
                while (consumed.load() < tasks && failures.load() == 0) {
                    if (std::chrono::steady_clock::now() >= deadline) {
                        ++failures;
                        break;
                    }
                    int64_t refs[2]{};
                    uint32_t count = 0;
                    const auto result = scheduler_linked_try_pop_batch(fixture.queue, refs, credit, &count, 2);
                    if (result == SchedulerLinkedResult::NOT_READY || result == SchedulerLinkedResult::RETRY) continue;
                    if (result != SchedulerLinkedResult::SUCCESS || count == 0 || count > credit) {
                        ++failures;
                        break;
                    }
                    for (uint32_t index = 0; index < count; ++index) {
                        const int64_t ref = refs[index];
                        if (ref < 0 || ref >= tasks) {
                            ++failures;
                            break;
                        }
                        if (++claims[ref] != 1 || payload[ref] != UINT64_C(0xfeed0000) + ref) ++failures;
                        if (index != 0 && refs[index] / per_producer == refs[index - 1] / per_producer &&
                            refs[index] <= refs[index - 1])
                            ++failures;
                        ++consumed;
                    }
                }
            });
        }
        start.store(true, std::memory_order_release);
        for (auto &thread : threads)
            thread.join();
        EXPECT_EQ(failures.load(), 0u);
        EXPECT_EQ(consumed.load(), tasks);
        for (const auto &claim : claims)
            EXPECT_EQ(claim.load(), 1u);
        std::vector<int64_t> last(producers, -1);
        int64_t link = fixture.links[fixture.queue.sentinel];
        for (uint32_t visited = 0; visited < tasks; ++visited) {
            ASSERT_GE(link, 0);
            ASSERT_LT(link, tasks);
            const uint32_t producer = link / per_producer;
            EXPECT_EQ(link % per_producer, last[producer] + 1);
            last[producer] = link % per_producer;
            link = fixture.links[link];
        }
        EXPECT_EQ(link, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(SchedulerLinkedQueueTest, DeferredPublisherMergesNewWorkAfterRetryWithoutDuplicateOwnership) {
    LinkedQueueFixture fixture(12);
    SchedulerLinkedBatch pending{};
    auto first = fixture.prepare({0, 1, 2, 3, 4, 5, 6, 7});
    ASSERT_EQ(scheduler_linked_append_private(fixture.queue, &pending, &first), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(first.count, 0u);
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &pending, 0), SchedulerLinkedResult::RETRY);
    auto second = fixture.prepare({8, 9, 10, 11});
    ASSERT_EQ(scheduler_linked_append_private(fixture.queue, &pending, &second), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(second.count, 0u);
    EXPECT_EQ(pending.count, 12u);
    int64_t ref = -1;
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &pending), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(pending.count, 0u);
    for (int64_t expected = 0; expected < 12; ++expected) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
}

TEST(SchedulerLinkedQueueTest, DeferredPublisherDoesNotHoldUpUnrelatedPublishedWork) {
    LinkedQueueFixture fixture(4);
    auto pending = fixture.prepare({0, 1});
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &pending, 0), SchedulerLinkedResult::RETRY);
    auto active = fixture.prepare({2});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &active), SchedulerLinkedResult::SUCCESS);
    int64_t ref = -1;
    ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
    EXPECT_EQ(ref, 2);
    auto incoming = fixture.prepare({3});
    ASSERT_EQ(scheduler_linked_append_private(fixture.queue, &pending, &incoming), SchedulerLinkedResult::SUCCESS);
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &pending), SchedulerLinkedResult::SUCCESS);
    for (int64_t expected : {0, 1, 3}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
}

TEST(SchedulerLinkedQueueTest, RejectsInvalidPrivateTransferWithoutModifyingEitherOwner) {
    LinkedQueueFixture fixture(2);
    auto pending = fixture.prepare({0});
    auto incoming = fixture.prepare({1});
    const auto links = fixture.links;
    EXPECT_EQ(scheduler_linked_append_private(fixture.queue, &pending, &pending), SchedulerLinkedResult::INVALID);
    incoming.count = 2;
    EXPECT_EQ(scheduler_linked_append_private(fixture.queue, &pending, &incoming), SchedulerLinkedResult::INVALID);
    EXPECT_EQ(fixture.links, links);
    EXPECT_EQ(pending.first, 0);
    EXPECT_EQ(pending.count, 1u);
    EXPECT_EQ(incoming.first, 1);
    EXPECT_EQ(incoming.count, 2u);
}

TEST(SchedulerLinkedQueueTest, AbortDropsOnlyPrivateWorkAndResetRequiresQuiescence) {
    LinkedQueueFixture fixture(4);
    auto published = fixture.prepare({0, 1});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &published), SchedulerLinkedResult::SUCCESS);
    auto pending = fixture.prepare({2, 3});
    EXPECT_EQ(scheduler_linked_try_push(fixture.queue, &pending, 0), SchedulerLinkedResult::RETRY);
    int64_t ref = -1;
    for (int64_t expected : {0, 1}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    pending = {};
    EXPECT_EQ(fixture.pop(&ref), SchedulerLinkedResult::NOT_READY);
    std::fill(fixture.links.begin(), fixture.links.end(), SCHEDULER_TASK_REF_INVALID);
    fixture.control.head = fixture.control.tail = fixture.queue.sentinel;
    auto next_run = fixture.prepare({2, 0, 3, 1});
    ASSERT_EQ(scheduler_linked_try_push(fixture.queue, &next_run), SchedulerLinkedResult::SUCCESS);
    for (int64_t expected : {2, 0, 3, 1}) {
        ASSERT_EQ(fixture.pop(&ref), SchedulerLinkedResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
}
}  // namespace
