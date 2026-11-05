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

#include <atomic>
#include <thread>
#include <vector>

#include "scheduler/scheduler_async_directory.h"
#include "scheduler/scheduler_async_lease.h"

namespace {
using DirectoryResult = SchedulerAsyncDirectoryResult;
using LeaseResult = SchedulerAsyncLeaseResult;
using Phase = SchedulerAsyncPhase;

TEST(HbgAsyncRecords, EveryDirectoryBitPreservesNeighbours) {
    for (uint32_t bit = 0; bit < 64; ++bit) {
        SchedulerAsyncDirectoryWord directory;
        const uint64_t mask = UINT64_C(1) << bit;
        directory.bits = ~mask;
        EXPECT_EQ(scheduler_async_directory_publish_once(directory, bit), DirectoryResult::SUCCESS);
        EXPECT_EQ(scheduler_async_directory_load(directory), UINT64_MAX);
        EXPECT_EQ(scheduler_async_directory_clear_once(directory, bit), DirectoryResult::SUCCESS);
        EXPECT_EQ(scheduler_async_directory_load(directory), ~mask);
    }
}

TEST(HbgAsyncRecords, InvalidBitDoesNotMutateDirectory) {
    SchedulerAsyncDirectoryWord directory;
    directory.bits = UINT64_C(0x8000000000000001);
    for (uint32_t bit : {64u, 65u, UINT32_MAX}) {
        EXPECT_EQ(scheduler_async_directory_publish_once(directory, bit), DirectoryResult::INVALID_BIT);
        EXPECT_EQ(scheduler_async_directory_clear_once(directory, bit), DirectoryResult::INVALID_BIT);
        EXPECT_EQ(scheduler_async_directory_load(directory), UINT64_C(0x8000000000000001));
    }
}

TEST(HbgAsyncRecords, DuplicateContributionsReportCorruptionWithoutCompensation) {
    SchedulerAsyncDirectoryWord directory;
    ASSERT_EQ(scheduler_async_directory_publish_once(directory, 0), DirectoryResult::SUCCESS);
    EXPECT_EQ(scheduler_async_directory_publish_once(directory, 0), DirectoryResult::PROTOCOL_ERROR);
    // Arithmetic has already carried into bit 1. No caller may keep using
    // this corrupted directory or undo it while neighbours are changing.
    EXPECT_EQ(scheduler_async_directory_load(directory), 2u);

    SchedulerAsyncDirectoryWord empty;
    EXPECT_EQ(scheduler_async_directory_clear_once(empty, 0), DirectoryResult::PROTOCOL_ERROR);
    EXPECT_EQ(scheduler_async_directory_load(empty), UINT64_MAX);

    SchedulerAsyncDirectoryWord high;
    ASSERT_EQ(scheduler_async_directory_publish_once(high, 63), DirectoryResult::SUCCESS);
    EXPECT_EQ(scheduler_async_directory_publish_once(high, 63), DirectoryResult::PROTOCOL_ERROR);
    EXPECT_EQ(scheduler_async_directory_load(high), 0u);
}

TEST(HbgAsyncRecords, ConcurrentPublicationAndClearPreserveEveryContribution) {
    SchedulerAsyncDirectoryWord directory;
    std::atomic<uint32_t> published{0};
    std::atomic<uint32_t> failures{0};
    std::atomic<bool> clear{false};
    std::vector<std::thread> workers;
    for (uint32_t bit = 0; bit < 64; ++bit) {
        workers.emplace_back([&, bit] {
            if (scheduler_async_directory_publish_once(directory, bit) != DirectoryResult::SUCCESS) ++failures;
            ++published;
            while (!clear.load(std::memory_order_acquire)) {}
            if (scheduler_async_directory_clear_once(directory, bit) != DirectoryResult::SUCCESS) ++failures;
        });
    }
    while (published.load(std::memory_order_acquire) != 64) {}
    EXPECT_EQ(scheduler_async_directory_load(directory), UINT64_MAX);
    clear.store(true, std::memory_order_release);
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(failures.load(), 0u);
    EXPECT_EQ(scheduler_async_directory_load(directory), 0u);
}

TEST(HbgAsyncRecords, PublishingAndClearingAdjacentBitsMayInterleave) {
    for (int repeat = 0; repeat < 20; ++repeat) {
        SchedulerAsyncDirectoryWord directory;
        std::atomic<uint32_t> failures{0};
        std::vector<std::thread> workers;
        for (uint32_t bit = 0; bit < 64; bit += 2)
            ASSERT_EQ(scheduler_async_directory_publish_once(directory, bit), DirectoryResult::SUCCESS);
        for (uint32_t bit = 0; bit < 64; ++bit) {
            workers.emplace_back([&, bit] {
                const auto result = bit % 2 == 0 ? scheduler_async_directory_clear_once(directory, bit) :
                                                   scheduler_async_directory_publish_once(directory, bit);
                if (result != DirectoryResult::SUCCESS) ++failures;
            });
        }
        for (auto &worker : workers)
            worker.join();
        EXPECT_EQ(failures.load(), 0u);
        EXPECT_EQ(scheduler_async_directory_load(directory), UINT64_C(0xaaaaaaaaaaaaaaaa));
    }
}

TEST(HbgAsyncRecords, OnlyOneConcurrentClaimantGetsOwnership) {
    SchedulerAsyncLease lease;
    const auto published = scheduler_async_lease_publish(lease);
    ASSERT_EQ(published.result, LeaseResult::SUCCESS);
    std::atomic<int> winners{0};
    std::atomic<int> conflicts{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < 32; ++i) {
        workers.emplace_back([&] {
            const auto result = scheduler_async_lease_try_claim(lease, published.word).result;
            if (result == LeaseResult::SUCCESS) ++winners;
            else if (result == LeaseResult::CONTENDED) ++conflicts;
        });
    }
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(winners.load(), 1);
    EXPECT_EQ(conflicts.load(), 31);
    EXPECT_EQ(scheduler_async_lease_phase(scheduler_async_lease_load(lease)), Phase::OWNED);
}

TEST(HbgAsyncRecords, StaleClaimAndOwnerCannotAffectNewGeneration) {
    SchedulerAsyncLease lease;
    const auto initial = scheduler_async_lease_publish(lease);
    const auto first_owner = scheduler_async_lease_try_claim(lease, initial.word);
    ASSERT_EQ(first_owner.result, LeaseResult::SUCCESS);
    const auto released = scheduler_async_lease_handoff(lease, first_owner.word, Phase::RUNNABLE);
    ASSERT_EQ(released.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_generation(released.word), 2u);
    EXPECT_EQ(scheduler_async_lease_try_claim(lease, initial.word).result, LeaseResult::CONTENDED);
    const auto next_owner = scheduler_async_lease_try_claim(lease, released.word);
    ASSERT_EQ(next_owner.result, LeaseResult::SUCCESS);
    for (auto phase : {Phase::RUNNABLE, Phase::TERMINAL, Phase::FAILED})
        EXPECT_EQ(scheduler_async_lease_handoff(lease, first_owner.word, phase).result, LeaseResult::OWNERSHIP_LOST);
    EXPECT_EQ(scheduler_async_lease_load(lease), next_owner.word);
}

TEST(HbgAsyncRecords, RepeatedPendingRetainsTheOriginalDirectoryBit) {
    SchedulerAsyncDirectoryWord directory;
    SchedulerAsyncLease lease;
    ASSERT_EQ(scheduler_async_lease_publish(lease).result, LeaseResult::SUCCESS);
    ASSERT_EQ(scheduler_async_directory_publish_once(directory, 63), DirectoryResult::SUCCESS);
    for (int i = 0; i < 4096; ++i) {
        const auto claimed = scheduler_async_lease_try_claim(lease, scheduler_async_lease_load(lease));
        ASSERT_EQ(claimed.result, LeaseResult::SUCCESS);
        const auto released = scheduler_async_lease_handoff(lease, claimed.word, Phase::RUNNABLE);
        ASSERT_EQ(released.result, LeaseResult::SUCCESS);
        EXPECT_EQ(scheduler_async_directory_load(directory), UINT64_C(1) << 63);
    }
    const auto claimed = scheduler_async_lease_try_claim(lease, scheduler_async_lease_load(lease));
    ASSERT_EQ(claimed.result, LeaseResult::SUCCESS);
    ASSERT_EQ(scheduler_async_lease_handoff(lease, claimed.word, Phase::TERMINAL).result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_directory_clear_once(directory, 63), DirectoryResult::SUCCESS);
    EXPECT_EQ(scheduler_async_directory_load(directory), 0u);
    EXPECT_EQ(scheduler_async_lease_generation(scheduler_async_lease_load(lease)), 4097u);
}

TEST(HbgAsyncRecords, TerminalAndFailedRecordsCannotBeClaimedOrRepublished) {
    for (auto terminal : {Phase::TERMINAL, Phase::FAILED}) {
        SchedulerAsyncLease lease;
        const auto published = scheduler_async_lease_publish(lease);
        const auto claimed = scheduler_async_lease_try_claim(lease, published.word);
        ASSERT_EQ(claimed.result, LeaseResult::SUCCESS);
        const auto finished = scheduler_async_lease_handoff(lease, claimed.word, terminal);
        ASSERT_EQ(finished.result, LeaseResult::SUCCESS);
        EXPECT_EQ(scheduler_async_lease_try_claim(lease, finished.word).result, LeaseResult::UNAVAILABLE);
        EXPECT_EQ(scheduler_async_lease_try_claim(lease, published.word).result, LeaseResult::CONTENDED);
        EXPECT_EQ(scheduler_async_lease_publish(lease).result, LeaseResult::OWNERSHIP_LOST);
        EXPECT_EQ(scheduler_async_lease_load(lease), finished.word);
    }
}

TEST(HbgAsyncRecords, InvalidTransitionsLeaveWordUntouched) {
    for (auto phase : {Phase::DORMANT, Phase::RUNNABLE, Phase::OWNED, Phase::TERMINAL, Phase::FAILED}) {
        for (auto next : {Phase::DORMANT, Phase::RUNNABLE, Phase::OWNED, Phase::TERMINAL, Phase::FAILED}) {
            const bool legal = (phase == Phase::DORMANT && (next == Phase::RUNNABLE || next == Phase::FAILED)) ||
                               (phase == Phase::RUNNABLE && next == Phase::OWNED) ||
                               (phase == Phase::OWNED &&
                                (next == Phase::RUNNABLE || next == Phase::TERMINAL || next == Phase::FAILED));
            if (legal) continue;
            SchedulerAsyncLease lease;
            lease.word = scheduler_async_lease_word(1, phase);
            const uint64_t before = scheduler_async_lease_load(lease);
            EXPECT_EQ(
                scheduler_async_lease_try_transition(lease, before, next).result, LeaseResult::INVALID_TRANSITION
            );
            EXPECT_EQ(scheduler_async_lease_load(lease), before);
        }
    }
}

TEST(HbgAsyncRecords, InvalidEncodingDoesNotObtainOwnership) {
    for (uint64_t invalid : {UINT64_C(0), UINT64_C(1), UINT64_C(15), UINT64_C(16), UINT64_MAX}) {
        EXPECT_FALSE(scheduler_async_lease_valid(invalid));
        SchedulerAsyncLease lease;
        lease.word = invalid;
        EXPECT_EQ(scheduler_async_lease_try_claim(lease, invalid).result, LeaseResult::INVALID_STATE);
        EXPECT_EQ(scheduler_async_lease_load(lease), invalid);
    }
    EXPECT_EQ(scheduler_async_lease_word(0, Phase::RUNNABLE), 0u);
    EXPECT_EQ(scheduler_async_lease_word(SCHEDULER_ASYNC_GENERATION_MAX + 1, Phase::RUNNABLE), 0u);
}

TEST(HbgAsyncRecords, GenerationExhaustionCannotWrapToAnOldOwner) {
    SchedulerAsyncLease lease;
    lease.word = scheduler_async_lease_word(SCHEDULER_ASYNC_GENERATION_MAX, Phase::RUNNABLE);
    const auto claimed = scheduler_async_lease_try_claim(lease, scheduler_async_lease_load(lease));
    ASSERT_EQ(claimed.result, LeaseResult::SUCCESS);
    EXPECT_EQ(
        scheduler_async_lease_handoff(lease, claimed.word, Phase::RUNNABLE).result, LeaseResult::GENERATION_EXHAUSTED
    );
    EXPECT_EQ(scheduler_async_lease_load(lease), claimed.word);
    EXPECT_EQ(scheduler_async_lease_handoff(lease, claimed.word, Phase::FAILED).result, LeaseResult::SUCCESS);
}

TEST(HbgAsyncRecords, PublicationFailureDoesNotMakeAnActiveWaiter) {
    SchedulerAsyncLease lease;
    const auto failed = scheduler_async_lease_try_transition(lease, scheduler_async_lease_load(lease), Phase::FAILED);
    ASSERT_EQ(failed.result, LeaseResult::SUCCESS);
    EXPECT_EQ(scheduler_async_lease_try_claim(lease, failed.word).result, LeaseResult::UNAVAILABLE);
    EXPECT_EQ(scheduler_async_lease_publish(lease).result, LeaseResult::OWNERSHIP_LOST);
}
}  // namespace
