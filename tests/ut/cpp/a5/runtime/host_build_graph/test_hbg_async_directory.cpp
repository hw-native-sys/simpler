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

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <thread>
#include <vector>

#include "scheduler/scheduler_async_directory.h"

namespace {

using Result = SchedulerAsyncDirectoryResult;

struct ScriptedAtomics {
    uint64_t word{0};
    std::vector<uint64_t> peer_changes;
    uint32_t queries{0};
    uint32_t attempts{0};
    uint32_t writes{0};
    bool use_hint{false};
    uint64_t hint{0};

    uint64_t query() {
        ++queries;
        return use_hint ? hint : word;
    }

    uint64_t compare_exchange(uint64_t expected, uint64_t desired) {
        if (attempts < peer_changes.size()) word ^= peer_changes[attempts];
        ++attempts;
        const uint64_t actual = word;
        if (actual == expected) {
            word = desired;
            ++writes;
        }
        return actual;
    }
};

TEST(SchedulerAsyncDirectoryTest, PublishesAndClearsBoundaryBitsWithoutChangingNeighbors) {
    constexpr uint64_t neighbors = UINT64_C(0x2040080010020040);
    volatile uint64_t word = neighbors;
    ASSERT_EQ(scheduler_async_directory_publish(word, 0), Result::SUCCESS);
    ASSERT_EQ(scheduler_async_directory_publish(word, 63), Result::SUCCESS);
    EXPECT_EQ(word, neighbors | UINT64_C(1) | (UINT64_C(1) << 63));
    ASSERT_EQ(scheduler_async_directory_clear(word, 0), Result::SUCCESS);
    EXPECT_EQ(word, neighbors | (UINT64_C(1) << 63));
    ASSERT_EQ(scheduler_async_directory_clear(word, 63), Result::SUCCESS);
    EXPECT_EQ(word, neighbors);
}

TEST(SchedulerAsyncDirectoryTest, DuplicatePublicationAndClearLeaveOtherRecordsUntouched) {
    volatile uint64_t word = UINT64_C(2);
    EXPECT_EQ(scheduler_async_directory_publish(word, 1), Result::UNEXPECTED_STATE);
    EXPECT_EQ(word, UINT64_C(2));
    EXPECT_EQ(scheduler_async_directory_clear(word, 0), Result::UNEXPECTED_STATE);
    EXPECT_EQ(word, UINT64_C(2));
    ASSERT_EQ(scheduler_async_directory_clear(word, 1), Result::SUCCESS);
    EXPECT_EQ(scheduler_async_directory_clear(word, 1), Result::UNEXPECTED_STATE);
    EXPECT_EQ(word, UINT64_C(0));
}

TEST(SchedulerAsyncDirectoryTest, RejectsInvalidBitBeforeAnyAtomicAccess) {
    for (uint32_t bit : {64u, 65u, UINT32_MAX}) {
        SCOPED_TRACE(bit);
        ScriptedAtomics atomics{UINT64_C(2)};
        EXPECT_EQ(scheduler_async_directory_detail::update<true>(atomics, bit), Result::INVALID_BIT);
        EXPECT_EQ(scheduler_async_directory_detail::update<false>(atomics, bit), Result::INVALID_BIT);
        EXPECT_EQ(atomics.word, UINT64_C(2));
        EXPECT_EQ(atomics.queries, 0u);
        EXPECT_EQ(atomics.attempts, 0u);
        EXPECT_EQ(atomics.writes, 0u);
    }
}

TEST(SchedulerAsyncDirectoryTest, CompetingSameBitIsRejectedWithoutASecondWrite) {
    constexpr uint64_t target = UINT64_C(1) << 63;
    ScriptedAtomics publish{UINT64_C(2), {target}};
    EXPECT_EQ(scheduler_async_directory_detail::update<true>(publish, 63), Result::UNEXPECTED_STATE);
    EXPECT_EQ(publish.word, UINT64_C(2) | target);
    EXPECT_EQ(publish.queries, 1u);
    EXPECT_EQ(publish.attempts, 1u);
    EXPECT_EQ(publish.writes, 0u);

    ScriptedAtomics clear{UINT64_C(2) | target, {target}};
    EXPECT_EQ(scheduler_async_directory_detail::update<false>(clear, 63), Result::UNEXPECTED_STATE);
    EXPECT_EQ(clear.word, UINT64_C(2));
    EXPECT_EQ(clear.queries, 1u);
    EXPECT_EQ(clear.attempts, 1u);
    EXPECT_EQ(clear.writes, 0u);
}

TEST(SchedulerAsyncDirectoryTest, StaleHintsCannotGrantOrRejectAChangeWithoutCasConfirmation) {
    for (uint32_t bit : {0u, 63u}) {
        const uint64_t target = UINT64_C(1) << bit;
        constexpr uint64_t neighbor = UINT64_C(2);
        for (bool publish : {false, true}) {
            for (bool already_changed : {false, true}) {
                const uint64_t word = neighbor | ((already_changed == publish) ? target : UINT64_C(0));
                for (uint64_t stale_bits : {UINT64_C(0), target, neighbor, target | neighbor}) {
                    SCOPED_TRACE(bit);
                    SCOPED_TRACE(publish);
                    SCOPED_TRACE(already_changed);
                    SCOPED_TRACE(stale_bits);
                    ScriptedAtomics atomics{word};
                    atomics.use_hint = true;
                    atomics.hint = word ^ stale_bits;
                    const auto result = publish ? scheduler_async_directory_detail::update<true>(atomics, bit) :
                                                  scheduler_async_directory_detail::update<false>(atomics, bit);
                    EXPECT_EQ(result, already_changed ? Result::UNEXPECTED_STATE : Result::SUCCESS);
                    EXPECT_EQ(atomics.word, publish ? word | target : word & ~target);
                    EXPECT_EQ(atomics.queries, 1u);
                    EXPECT_EQ(atomics.attempts, already_changed || stale_bits == 0 ? 1u : 2u);
                }
            }
        }
    }
}

std::vector<uint64_t> legal_neighbor_changes(uint32_t target) {
    std::vector<uint64_t> changes;
    for (uint32_t pass = 0; pass < 2; ++pass)
        for (uint32_t bit = 0; bit < 64; ++bit)
            if (bit != target) changes.push_back(UINT64_C(1) << bit);
    return changes;
}

TEST(SchedulerAsyncDirectoryTest, StaleHintAndEveryNeighborFitTheCasRetryBound) {
    for (uint32_t bit : {0u, 63u}) {
        for (bool stale : {false, true}) {
            SCOPED_TRACE(bit);
            SCOPED_TRACE(stale);
            const uint64_t target = UINT64_C(1) << bit;
            auto changes = legal_neighbor_changes(bit);
            ASSERT_EQ(changes.size(), 126u);
            if (stale) changes.insert(changes.begin(), UINT64_C(0));

            ScriptedAtomics publish{UINT64_C(0), changes};
            publish.use_hint = stale;
            publish.hint = target;
            EXPECT_EQ(scheduler_async_directory_detail::update<true>(publish, bit), Result::SUCCESS);
            EXPECT_EQ(publish.word, target);
            EXPECT_EQ(publish.queries, 1u);
            EXPECT_EQ(publish.attempts, stale ? 128u : 127u);
            EXPECT_EQ(publish.writes, 1u);

            ScriptedAtomics clear{target, changes};
            clear.use_hint = stale;
            clear.hint = 0;
            EXPECT_EQ(scheduler_async_directory_detail::update<false>(clear, bit), Result::SUCCESS);
            EXPECT_EQ(clear.word, UINT64_C(0));
            EXPECT_EQ(clear.queries, 1u);
            EXPECT_EQ(clear.attempts, stale ? 128u : 127u);
            EXPECT_EQ(clear.writes, 1u);
        }
    }
}

TEST(SchedulerAsyncDirectoryTest, ExceedingTheOneRunMutationContractIsTerminalCorruption) {
    const std::vector<uint64_t> changes(128, UINT64_C(1));
    ScriptedAtomics publish{UINT64_C(0), changes};
    EXPECT_EQ(scheduler_async_directory_detail::update<true>(publish, 63), Result::PROTOCOL_CORRUPTION);
    EXPECT_EQ(publish.word, UINT64_C(0));
    EXPECT_EQ(publish.queries, 1u);
    EXPECT_EQ(publish.attempts, 128u);
    EXPECT_EQ(publish.writes, 0u);

    constexpr uint64_t target = UINT64_C(1) << 63;
    ScriptedAtomics clear{target, changes};
    EXPECT_EQ(scheduler_async_directory_detail::update<false>(clear, 63), Result::PROTOCOL_CORRUPTION);
    EXPECT_EQ(clear.word, target);
    EXPECT_EQ(clear.queries, 1u);
    EXPECT_EQ(clear.attempts, 128u);
    EXPECT_EQ(clear.writes, 0u);
}

TEST(SchedulerAsyncDirectoryTest, ConcurrentOwnersPublishEveryBitAndClearOnlyTheirOwnBit) {
    volatile uint64_t word = 0;
    std::mutex mutex;
    std::condition_variable changed;
    bool start = false;
    bool clear = false;
    uint32_t published = 0;
    std::vector<Result> publish_results(64, Result::PROTOCOL_CORRUPTION);
    std::vector<Result> clear_results(64, Result::PROTOCOL_CORRUPTION);
    std::vector<std::thread> workers;
    for (uint32_t bit = 0; bit < 64; ++bit) {
        workers.emplace_back([&, bit] {
            {
                std::unique_lock<std::mutex> lock(mutex);
                changed.wait(lock, [&] {
                    return start;
                });
            }
            publish_results[bit] = scheduler_async_directory_publish(word, bit);
            {
                std::unique_lock<std::mutex> lock(mutex);
                ++published;
                changed.notify_all();
                changed.wait(lock, [&] {
                    return clear;
                });
            }
            clear_results[bit] = scheduler_async_directory_clear(word, bit);
        });
    }
    {
        std::lock_guard<std::mutex> lock(mutex);
        start = true;
    }
    changed.notify_all();
    uint64_t all_published = 0;
    {
        std::unique_lock<std::mutex> lock(mutex);
        changed.wait(lock, [&] {
            return published == 64;
        });
        all_published = scheduler_gm_fetch_add(word, UINT64_C(0));
        clear = true;
    }
    changed.notify_all();
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(all_published, UINT64_MAX);
    EXPECT_EQ(word, UINT64_C(0));
    for (uint32_t bit = 0; bit < 64; ++bit) {
        SCOPED_TRACE(bit);
        EXPECT_EQ(publish_results[bit], Result::SUCCESS);
        EXPECT_EQ(clear_results[bit], Result::SUCCESS);
    }
}

TEST(HbgAsyncDirectory, NotificationHalvesPreserveBitBoundariesAndGmFallback) {
    for (uint32_t mapped = 0; mapped < 4; ++mapped) {
        SCOPED_TRACE(mapped);
        SchedulerAsyncDirectoryWord word{};
        alignas(32) uint32_t notifications[2]{};
        for (uint32_t half = 0; half < 2; ++half)
            if ((mapped & (1U << half)) != 0)
                word.notification_addresses[half] = reinterpret_cast<uint64_t>(&notifications[half]);
        for (uint32_t bit : {0U, 31U, 32U, 63U})
            ASSERT_EQ(scheduler_async_directory_publish(word, bit), Result::SUCCESS);
        EXPECT_EQ(scheduler_async_directory_query(word, 0, 64), UINT64_C(0x8000000180000001));
        EXPECT_EQ(scheduler_async_directory_query(word, 32, 64), UINT64_C(0x8000000100000000));
        ASSERT_EQ(scheduler_async_directory_clear(word, 31), Result::SUCCESS);
        ASSERT_EQ(scheduler_async_directory_clear(word, 32), Result::SUCCESS);
        EXPECT_EQ(scheduler_async_directory_query(word, 0, 64), UINT64_C(0x8000000000000001));
        for (uint32_t half = 0; half < 2; ++half)
            if ((mapped & (1U << half)) != 0) EXPECT_EQ((word.bits >> (half * 32)) & UINT64_C(0xffffffff), 0u);
    }
    SchedulerAsyncDirectoryWord tail{};
    uint32_t notification = 1;
    tail.notification_addresses[0] = reinterpret_cast<uint64_t>(&notification);
    tail.notification_addresses[1] = UINT64_MAX;  // Absent tail half must never be read.
    EXPECT_EQ(scheduler_async_directory_query(tail, 0, 1), 1u);
}

TEST(HbgAsyncDirectory, ConcurrentMappedSetAndClearPreserveOtherGroups) {
    SchedulerAsyncDirectoryWord word{};
    alignas(32) uint32_t notification = UINT32_C(0xffff);
    word.notification_addresses[0] = reinterpret_cast<uint64_t>(&notification);
    std::vector<std::thread> workers;
    for (uint32_t lane = 0; lane < 16; ++lane)
        workers.emplace_back([&, lane] {
            EXPECT_EQ(scheduler_async_directory_clear(word, lane), Result::SUCCESS);
            EXPECT_EQ(scheduler_async_directory_publish(word, lane + 16), Result::SUCCESS);
        });
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(scheduler_async_directory_query(word, 0, 32), UINT64_C(0xffff0000));
    EXPECT_EQ(word.bits, 0u);
}

}  // namespace
