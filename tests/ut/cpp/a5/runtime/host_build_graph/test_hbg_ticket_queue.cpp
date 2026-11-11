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
#include <numeric>
#include <thread>
#include <vector>

#include "scheduler/scheduler_ticket_queue.h"

namespace {
struct TicketQueueFixture {
    SchedulerTicketQueue control{};
    std::vector<SchedulerTicketEntry> entries;
    SchedulerTicketQueueView queue;

    explicit TicketQueueFixture(uint64_t capacity) :
        entries(capacity),
        queue{&control, entries.data(), capacity, capacity} {}

    SchedulerTicketResult push(const std::vector<int64_t> &refs) {
        SchedulerTicketRange range{};
        const auto result = scheduler_ticket_reserve(queue, refs.size(), &range);
        if (result != SchedulerTicketResult::SUCCESS) return result;
        return scheduler_ticket_publish(queue, range, refs.data());
    }

    std::vector<int64_t> published_refs() const {
        std::vector<int64_t> refs;
        refs.reserve(entries.size());
        for (const auto &entry : entries)
            refs.push_back(static_cast<int64_t>(entry.task_ref));
        return refs;
    }
};

TEST(SchedulerTicketQueueTest, PublishedEntriesOccupyDistinctCacheLines) {
    TicketQueueFixture fixture(4);
    for (size_t index = 0; index < fixture.entries.size(); ++index) {
        const auto address = reinterpret_cast<uintptr_t>(&fixture.queue.entries[index]);
        EXPECT_EQ(address % 64, 0u);
        if (index != 0) {
            const auto previous = reinterpret_cast<uintptr_t>(&fixture.queue.entries[index - 1]);
            EXPECT_GE(address - previous, 64u);
        }
    }
}

TEST(SchedulerTicketQueueTest, PreservesBatchOrderThroughEmptyTransitionsWithoutReusingSlots) {
    TicketQueueFixture fixture(5);
    int64_t ref = 123;
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::NOT_READY);
    EXPECT_EQ(ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_EQ(fixture.push({3, 1}), SchedulerTicketResult::SUCCESS);
    for (int64_t expected : {3, 1}) {
        ASSERT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::NOT_READY);
    ASSERT_EQ(fixture.push({4, 0, 2}), SchedulerTicketResult::SUCCESS);
    for (int64_t expected : {4, 0, 2}) {
        ASSERT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::NOT_READY);
    EXPECT_EQ(fixture.control.head, 5u);
    EXPECT_EQ(fixture.control.reserved_tail, 5u);
    EXPECT_EQ(fixture.published_refs(), (std::vector<int64_t>{3, 1, 4, 0, 2}));
}

TEST(SchedulerTicketQueueTest, LaterPublicationCannotConsumeAnEarlierUnpublishedReservation) {
    TicketQueueFixture fixture(4);
    SchedulerTicketRange first{}, second{};
    ASSERT_EQ(scheduler_ticket_reserve(fixture.queue, 2, &first), SchedulerTicketResult::SUCCESS);
    ASSERT_EQ(scheduler_ticket_reserve(fixture.queue, 2, &second), SchedulerTicketResult::SUCCESS);
    const int64_t late[] = {2, 3};
    ASSERT_EQ(scheduler_ticket_publish(fixture.queue, second, late), SchedulerTicketResult::SUCCESS);
    int64_t ref = 123;
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::NOT_READY);
    EXPECT_EQ(fixture.control.head, 0u);
    EXPECT_EQ(ref, SCHEDULER_TASK_REF_INVALID);
    const int64_t early[] = {0, 1};
    ASSERT_EQ(scheduler_ticket_publish(fixture.queue, first, early), SchedulerTicketResult::SUCCESS);
    for (int64_t expected = 0; expected < 4; ++expected) {
        ASSERT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::SUCCESS);
        EXPECT_EQ(ref, expected);
    }
}

TEST(SchedulerTicketQueueTest, BatchClaimStopsAtAnUnpublishedSlotAndRespectsCandidateCredit) {
    TicketQueueFixture fixture(4);
    SchedulerTicketRange ranges[3]{};
    ASSERT_EQ(scheduler_ticket_reserve(fixture.queue, 1, &ranges[0]), SchedulerTicketResult::SUCCESS);
    ASSERT_EQ(scheduler_ticket_reserve(fixture.queue, 1, &ranges[1]), SchedulerTicketResult::SUCCESS);
    ASSERT_EQ(scheduler_ticket_reserve(fixture.queue, 2, &ranges[2]), SchedulerTicketResult::SUCCESS);
    const int64_t first[] = {0}, middle[] = {1}, last[] = {2, 3};
    ASSERT_EQ(scheduler_ticket_publish(fixture.queue, ranges[0], first), SchedulerTicketResult::SUCCESS);
    ASSERT_EQ(scheduler_ticket_publish(fixture.queue, ranges[2], last), SchedulerTicketResult::SUCCESS);
    int64_t refs[2]{};
    uint32_t count = 123;
    EXPECT_EQ(scheduler_ticket_try_pop_batch(fixture.queue, refs, 0, &count), SchedulerTicketResult::NOT_READY);
    EXPECT_EQ(count, 0u);
    EXPECT_EQ(fixture.control.head, 0u);
    ASSERT_EQ(scheduler_ticket_try_pop_batch(fixture.queue, refs, 2, &count), SchedulerTicketResult::SUCCESS);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(refs[0], 0);
    EXPECT_EQ(refs[1], SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(scheduler_ticket_try_pop_batch(fixture.queue, refs, 2, &count), SchedulerTicketResult::NOT_READY);
    EXPECT_EQ(count, 0u);
    ASSERT_EQ(scheduler_ticket_publish(fixture.queue, ranges[1], middle), SchedulerTicketResult::SUCCESS);
    ASSERT_EQ(scheduler_ticket_try_pop_batch(fixture.queue, refs, 1, &count), SchedulerTicketResult::SUCCESS);
    EXPECT_EQ(count, 1u);
    EXPECT_EQ(refs[0], 1);
    ASSERT_EQ(scheduler_ticket_try_pop_batch(fixture.queue, refs, 2, &count), SchedulerTicketResult::SUCCESS);
    EXPECT_EQ(count, 2u);
    EXPECT_EQ(refs[0], 2);
    EXPECT_EQ(refs[1], 3);
}

TEST(SchedulerTicketQueueTest, PollingAndExhaustedRetryBudgetNeverReserveFutureWork) {
    TicketQueueFixture fixture(2);
    uint64_t retries = 0;
    int64_t ref = 123;
    for (uint32_t index = 0; index < 100; ++index)
        EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref, 8, &retries), SchedulerTicketResult::NOT_READY);
    EXPECT_EQ(fixture.control.head, 0u);
    EXPECT_EQ(fixture.control.reserved_tail, 0u);
    EXPECT_EQ(retries, 0u);
    ASSERT_EQ(fixture.push({1}), SchedulerTicketResult::SUCCESS);
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref, 0, &retries), SchedulerTicketResult::RETRY);
    EXPECT_EQ(ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.control.head, 0u);
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::SUCCESS);
    EXPECT_EQ(ref, 1);
}

TEST(SchedulerTicketQueueTest, RejectsInvalidReferencesBeforePublishingAnyPartOfTheRange) {
    TicketQueueFixture fixture(3);
    SchedulerTicketRange range{};
    ASSERT_EQ(scheduler_ticket_reserve(fixture.queue, 3, &range), SchedulerTicketResult::SUCCESS);
    const int64_t refs[] = {0, 1, 3};
    EXPECT_EQ(scheduler_ticket_publish(fixture.queue, range, refs), SchedulerTicketResult::INVALID);
    EXPECT_EQ(fixture.published_refs(), (std::vector<int64_t>{-1, -1, -1}));
    int64_t ref = 123;
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::NOT_READY);
    EXPECT_EQ(fixture.control.head, 0u);
}

TEST(SchedulerTicketQueueTest, OverflowIsTerminalAndCannotWriteBeyondTheBacking) {
    TicketQueueFixture fixture(3);
    fixture.entries.push_back(SchedulerTicketEntry{0x12345});
    fixture.queue.entries = fixture.entries.data();
    SchedulerTicketRange range{};
    ASSERT_EQ(scheduler_ticket_reserve(fixture.queue, 2, &range), SchedulerTicketResult::SUCCESS);
    EXPECT_EQ(scheduler_ticket_reserve(fixture.queue, 2, &range), SchedulerTicketResult::INVALID);
    EXPECT_EQ(range.count, 0u);
    EXPECT_EQ(fixture.control.reserved_tail, 4u);
    const int64_t refs[] = {0, 1};
    EXPECT_EQ(scheduler_ticket_publish(fixture.queue, {2, 2}, refs), SchedulerTicketResult::INVALID);
    EXPECT_EQ(fixture.published_refs(), (std::vector<int64_t>{-1, -1, -1, 0x12345}));
}

TEST(SchedulerTicketQueueTest, ZeroCapacityAndInvalidViewsDoNotAccessEntries) {
    TicketQueueFixture fixture(0);
    fixture.queue.entries = nullptr;
    SchedulerTicketRange range{};
    int64_t ref = 123;
    EXPECT_EQ(scheduler_ticket_reserve(fixture.queue, 0, &range), SchedulerTicketResult::SUCCESS);
    EXPECT_EQ(scheduler_ticket_publish(fixture.queue, range, nullptr), SchedulerTicketResult::SUCCESS);
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::NOT_READY);
    EXPECT_EQ(scheduler_ticket_reserve(fixture.queue, 1, &range), SchedulerTicketResult::INVALID);
    fixture.queue.capacity = 1;
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::INVALID);
    fixture.queue = {};
    EXPECT_EQ(scheduler_ticket_reserve(fixture.queue, 1, &range), SchedulerTicketResult::INVALID);
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::INVALID);
    EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, nullptr), SchedulerTicketResult::INVALID);
}

TEST(SchedulerTicketQueueTest, RejectsCorruptPublishedReferenceWithoutClaimingIt) {
    TicketQueueFixture fixture(1);
    for (int64_t invalid : {-2, 1}) {
        __atomic_store_n(&fixture.entries[0].task_ref, invalid, __ATOMIC_RELEASE);
        int64_t ref = 123;
        EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::INVALID);
        EXPECT_EQ(ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.control.head, 0u);
    }
}

TEST(SchedulerTicketQueueTest, LastPublishedTaskHasExactlyOneWinner) {
    TicketQueueFixture fixture(1);
    ASSERT_EQ(fixture.push({0}), SchedulerTicketResult::SUCCESS);
    std::atomic<bool> start{false};
    std::atomic<uint32_t> successes{0}, failures{0};
    std::vector<std::thread> threads;
    for (uint32_t worker = 0; worker < 16; ++worker) {
        threads.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {}
            int64_t ref = -1;
            const auto result = scheduler_ticket_try_pop(fixture.queue, &ref);
            if (result == SchedulerTicketResult::SUCCESS) {
                ++successes;
                if (ref != 0) ++failures;
            } else if (result != SchedulerTicketResult::NOT_READY && result != SchedulerTicketResult::RETRY) {
                ++failures;
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(successes.load(), 1u);
    EXPECT_EQ(failures.load(), 0u);
    EXPECT_EQ(fixture.control.head, 1u);
}

TEST(SchedulerTicketQueueTest, ConcurrentBatchesPublishPayloadBeforeUniqueConsumption) {
    constexpr uint32_t producer_count = 8, consumer_count = 12, per_producer = 128;
    constexpr uint32_t task_count = producer_count * per_producer;
    TicketQueueFixture fixture(task_count);
    std::vector<uint64_t> payload(task_count, 0);
    std::vector<std::atomic<uint32_t>> claims(task_count);
    for (auto &count : claims)
        count.store(0);
    std::atomic<bool> start{false};
    std::atomic<uint32_t> consumed{0}, failures{0};
    std::vector<std::thread> threads;
    for (uint32_t producer = 0; producer < producer_count; ++producer) {
        threads.emplace_back([&, producer] {
            while (!start.load(std::memory_order_acquire)) {}
            for (uint32_t offset = 0; offset < per_producer; offset += 4) {
                SchedulerTicketRange range{};
                if (scheduler_ticket_reserve(fixture.queue, 4, &range) != SchedulerTicketResult::SUCCESS) {
                    ++failures;
                    return;
                }
                int64_t refs[4]{};
                for (uint32_t index = 0; index < 4; ++index) {
                    const uint32_t task = producer * per_producer + offset + index;
                    payload[task] = UINT64_C(0xfeed0000) + task;
                    refs[index] = task;
                }
                if (scheduler_ticket_publish(fixture.queue, range, refs) != SchedulerTicketResult::SUCCESS) {
                    ++failures;
                    return;
                }
            }
        });
    }
    for (uint32_t consumer = 0; consumer < consumer_count; ++consumer) {
        threads.emplace_back([&] {
            while (!start.load(std::memory_order_acquire)) {}
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
            while (consumed.load() < task_count && failures.load() == 0) {
                if (std::chrono::steady_clock::now() >= deadline) {
                    ++failures;
                    break;
                }
                int64_t refs[2]{};
                uint32_t count = 0;
                const auto result = scheduler_ticket_try_pop_batch(fixture.queue, refs, 2, &count, 2);
                if (result == SchedulerTicketResult::NOT_READY || result == SchedulerTicketResult::RETRY) continue;
                if (result != SchedulerTicketResult::SUCCESS || count == 0 || count > 2) {
                    ++failures;
                    break;
                }
                for (uint32_t index = 0; index < count; ++index) {
                    const int64_t ref = refs[index];
                    if (ref < 0 || ref >= task_count) {
                        ++failures;
                        break;
                    }
                    if (++claims[ref] != 1 || payload[ref] != UINT64_C(0xfeed0000) + ref) ++failures;
                    ++consumed;
                }
            }
        });
    }
    start.store(true, std::memory_order_release);
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(failures.load(), 0u);
    EXPECT_EQ(consumed.load(), task_count);
    for (const auto &count : claims)
        EXPECT_EQ(count.load(), 1u);
    EXPECT_EQ(fixture.control.head, task_count);
    EXPECT_EQ(fixture.control.reserved_tail, task_count);
}

TEST(SchedulerTicketQueueTest, ReinitializesTheSameBackingOnlyAfterQuiescence) {
    TicketQueueFixture fixture(2);
    for (uint32_t run = 0; run < 3; ++run) {
        fixture.control.head = 0;
        fixture.control.reserved_tail = 0;
        std::fill(fixture.entries.begin(), fixture.entries.end(), SchedulerTicketEntry{});
        int64_t ref = 123;
        EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::NOT_READY);
        const std::vector<int64_t> expected{static_cast<int64_t>(run % 2), static_cast<int64_t>((run + 1) % 2)};
        ASSERT_EQ(fixture.push(expected), SchedulerTicketResult::SUCCESS);
        for (const auto task_ref : expected) {
            ASSERT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::SUCCESS);
            EXPECT_EQ(ref, task_ref);
        }
        EXPECT_EQ(scheduler_ticket_try_pop(fixture.queue, &ref), SchedulerTicketResult::NOT_READY);
    }
}
}  // namespace
