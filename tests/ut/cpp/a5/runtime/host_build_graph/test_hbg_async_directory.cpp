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
    uint32_t loads{0};
    uint32_t attempts{0};
    uint32_t writes{0};

    uint64_t load() {
        ++loads;
        return word;
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
        EXPECT_EQ(atomics.loads, 0u);
        EXPECT_EQ(atomics.attempts, 0u);
        EXPECT_EQ(atomics.writes, 0u);
    }
}

TEST(SchedulerAsyncDirectoryTest, CompetingSameBitIsRejectedWithoutASecondWrite) {
    constexpr uint64_t target = UINT64_C(1) << 63;
    ScriptedAtomics publish{UINT64_C(2), {target}};
    EXPECT_EQ(scheduler_async_directory_detail::update<true>(publish, 63), Result::UNEXPECTED_STATE);
    EXPECT_EQ(publish.word, UINT64_C(2) | target);
    EXPECT_EQ(publish.loads, 1u);
    EXPECT_EQ(publish.attempts, 1u);
    EXPECT_EQ(publish.writes, 0u);

    ScriptedAtomics clear{UINT64_C(2) | target, {target}};
    EXPECT_EQ(scheduler_async_directory_detail::update<false>(clear, 63), Result::UNEXPECTED_STATE);
    EXPECT_EQ(clear.word, UINT64_C(2));
    EXPECT_EQ(clear.loads, 1u);
    EXPECT_EQ(clear.attempts, 1u);
    EXPECT_EQ(clear.writes, 0u);
}

std::vector<uint64_t> legal_neighbor_changes(uint32_t target) {
    std::vector<uint64_t> changes;
    for (uint32_t pass = 0; pass < 2; ++pass)
        for (uint32_t bit = 0; bit < 64; ++bit)
            if (bit != target) changes.push_back(UINT64_C(1) << bit);
    return changes;
}

TEST(SchedulerAsyncDirectoryTest, EveryNeighborMayPublishAndClearBeforeTheLastAllowedCasSucceeds) {
    for (uint32_t bit : {0u, 63u}) {
        SCOPED_TRACE(bit);
        const uint64_t target = UINT64_C(1) << bit;
        const auto changes = legal_neighbor_changes(bit);
        ASSERT_EQ(changes.size(), 126u);

        ScriptedAtomics publish{UINT64_C(0), changes};
        EXPECT_EQ(scheduler_async_directory_detail::update<true>(publish, bit), Result::SUCCESS);
        EXPECT_EQ(publish.word, target);
        EXPECT_EQ(publish.loads, 1u);
        EXPECT_EQ(publish.attempts, 127u);
        EXPECT_EQ(publish.writes, 1u);

        ScriptedAtomics clear{target, changes};
        EXPECT_EQ(scheduler_async_directory_detail::update<false>(clear, bit), Result::SUCCESS);
        EXPECT_EQ(clear.word, UINT64_C(0));
        EXPECT_EQ(clear.loads, 1u);
        EXPECT_EQ(clear.attempts, 127u);
        EXPECT_EQ(clear.writes, 1u);
    }
}

TEST(SchedulerAsyncDirectoryTest, ExceedingTheOneRunMutationContractIsTerminalCorruption) {
    const std::vector<uint64_t> changes(128, UINT64_C(1));
    ScriptedAtomics publish{UINT64_C(0), changes};
    EXPECT_EQ(scheduler_async_directory_detail::update<true>(publish, 63), Result::PROTOCOL_CORRUPTION);
    EXPECT_EQ(publish.word, UINT64_C(1));
    EXPECT_EQ(publish.loads, 1u);
    EXPECT_EQ(publish.attempts, 127u);
    EXPECT_EQ(publish.writes, 0u);

    constexpr uint64_t target = UINT64_C(1) << 63;
    ScriptedAtomics clear{target, changes};
    EXPECT_EQ(scheduler_async_directory_detail::update<false>(clear, 63), Result::PROTOCOL_CORRUPTION);
    EXPECT_EQ(clear.word, target | UINT64_C(1));
    EXPECT_EQ(clear.loads, 1u);
    EXPECT_EQ(clear.attempts, 127u);
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

}  // namespace
