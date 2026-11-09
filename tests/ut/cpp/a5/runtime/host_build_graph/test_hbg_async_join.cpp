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

#include "scheduler/scheduler_async_join.h"

namespace {
using Result = SchedulerAsyncJoinResult;

void initialize(SchedulerAsyncTaskJoin &join, uint32_t blocks) {
    uint64_t word = 0;
    ASSERT_TRUE(scheduler_async_join_initial_word(blocks, word));
    join.word = word;
}

TEST(HbgAsyncJoin, InitializationCountsOnlyExecutionObligations) {
    uint64_t word = 123;
    EXPECT_FALSE(scheduler_async_join_initial_word(0, word));
    EXPECT_EQ(word, 123u);
    EXPECT_FALSE(scheduler_async_join_initial_word(SCHEDULER_ASYNC_JOIN_LIMIT + 1, word));
    EXPECT_EQ(word, 123u);
    EXPECT_FALSE(scheduler_async_join_initial_word(UINT32_MAX, word));
    ASSERT_TRUE(scheduler_async_join_initial_word(SCHEDULER_ASYNC_JOIN_LIMIT, word));
    EXPECT_EQ(scheduler_async_join_blocks(word), SCHEDULER_ASYNC_JOIN_LIMIT);
    EXPECT_EQ(scheduler_async_join_groups(word), 0u);
    EXPECT_TRUE(scheduler_async_join_valid(word));
}

TEST(HbgAsyncJoin, ZeroRegistrationAndSkippedExecutionNeedNoGroupSettlement) {
    SchedulerAsyncTaskJoin join;
    initialize(join, 3);
    EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::FINISH_OWNER);
    EXPECT_TRUE(scheduler_async_join_complete(scheduler_async_join_load(join)));
}

TEST(HbgAsyncJoin, LastConditionCanPrecedeOrFollowLastExecutionRelease) {
    for (bool condition_first : {false, true}) {
        SchedulerAsyncTaskJoin join;
        initialize(join, 1);
        ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
        EXPECT_EQ(scheduler_async_join_registered(scheduler_async_join_load(join)), 1u);
        if (condition_first) {
            EXPECT_EQ(scheduler_async_join_settle_group(join).result, Result::PENDING);
            EXPECT_EQ(scheduler_async_join_groups(scheduler_async_join_load(join)), 0u);
            EXPECT_EQ(scheduler_async_join_registered(scheduler_async_join_load(join)), 1u);
            EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::FINISH_OWNER);
        } else {
            EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::PENDING);
            EXPECT_EQ(scheduler_async_join_settle_group(join).result, Result::FINISH_OWNER);
        }
        EXPECT_TRUE(scheduler_async_join_complete(scheduler_async_join_load(join)));
    }
}

TEST(HbgAsyncJoin, MixedLanesCanSealAfterAnEarlierLaneAlreadySettled) {
    SchedulerAsyncTaskJoin join;
    initialize(join, 1);
    ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, Result::PENDING);
    ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
    ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, Result::FINISH_OWNER);
}

TEST(HbgAsyncJoin, FinalConditionAndBlockRaceGrantsExactlyOneFinishOwner) {
    for (int repeat = 0; repeat < 100; ++repeat) {
        SchedulerAsyncTaskJoin join;
        initialize(join, 1);
        ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
        std::atomic<bool> go{false};
        Result condition = Result::INVALID_STATE, block = Result::INVALID_STATE;
        std::thread first([&] {
            while (!go.load(std::memory_order_acquire)) {}
            condition = scheduler_async_join_settle_group(join).result;
        });
        std::thread second([&] {
            while (!go.load(std::memory_order_acquire)) {}
            block = scheduler_async_join_release_block(join).result;
        });
        go.store(true, std::memory_order_release);
        first.join();
        second.join();
        EXPECT_TRUE(
            (condition == Result::FINISH_OWNER && block == Result::PENDING) ||
            (block == Result::FINISH_OWNER && condition == Result::PENDING)
        );
        EXPECT_TRUE(scheduler_async_join_complete(scheduler_async_join_load(join)));
    }
}

TEST(HbgAsyncJoin, ConcurrentMixedPublicationAndSettlementDoNotLoseWakeup) {
    SchedulerAsyncTaskJoin join;
    initialize(join, 32);
    std::atomic<int> finish_owners{0}, errors{0};
    std::vector<std::thread> workers;
    auto settle = [&](SchedulerAsyncJoinOutcome outcome) {
        if (outcome.result == Result::FINISH_OWNER) ++finish_owners;
        else if (outcome.result != Result::PENDING) ++errors;
    };
    for (int i = 0; i < 32; ++i) {
        workers.emplace_back([&, i] {
            if (scheduler_async_join_add_group(join).result != Result::PENDING) ++errors;
            if (scheduler_async_join_add_group(join).result != Result::PENDING) ++errors;
            if (i % 2 == 0) {
                settle(scheduler_async_join_settle_group(join));
                settle(scheduler_async_join_release_block(join));
            } else {
                settle(scheduler_async_join_release_block(join));
                settle(scheduler_async_join_settle_group(join));
            }
            settle(scheduler_async_join_settle_group(join));
        });
    }
    for (auto &worker : workers)
        worker.join();
    EXPECT_EQ(errors.load(), 0);
    EXPECT_EQ(finish_owners.load(), 1);
    EXPECT_TRUE(scheduler_async_join_complete(scheduler_async_join_load(join)));
}

TEST(HbgAsyncJoin, RegisteringAfterExecutionReleaseIsAProtocolError) {
    SchedulerAsyncTaskJoin join;
    initialize(join, 1);
    ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_add_group(join).result, Result::INVALID_STATE);
}

TEST(HbgAsyncJoin, DuplicateSettlementNeverGrantsAnotherFinish) {
    SchedulerAsyncTaskJoin join;
    initialize(join, 1);
    ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, Result::UNDERFLOW);
    EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::INVALID_STATE);

    SchedulerAsyncTaskJoin completed;
    initialize(completed, 1);
    ASSERT_EQ(scheduler_async_join_release_block(completed).result, Result::FINISH_OWNER);
    EXPECT_EQ(scheduler_async_join_release_block(completed).result, Result::UNDERFLOW);
}

TEST(HbgAsyncJoin, GroupOverflowIsDiagnosedWithoutCounterCompensation) {
    SchedulerAsyncTaskJoin join;
    initialize(join, SCHEDULER_ASYNC_JOIN_LIMIT);
    for (uint32_t i = 0; i < SCHEDULER_ASYNC_JOIN_LIMIT; ++i)
        ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
    EXPECT_EQ(scheduler_async_join_add_group(join).result, Result::OVERFLOW);
    EXPECT_EQ(scheduler_async_join_settle_group(join).result, Result::INVALID_STATE);
}

TEST(HbgAsyncJoin, MaximumValidCountsSettleWithoutBorrowingBetweenFields) {
    SchedulerAsyncTaskJoin join;
    initialize(join, SCHEDULER_ASYNC_JOIN_LIMIT);
    for (uint32_t i = 0; i < SCHEDULER_ASYNC_JOIN_LIMIT; ++i)
        ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
    for (uint32_t i = 0; i < SCHEDULER_ASYNC_JOIN_LIMIT; ++i) {
        ASSERT_EQ(scheduler_async_join_settle_group(join).result, Result::PENDING);
        const auto expected = i + 1 == SCHEDULER_ASYNC_JOIN_LIMIT ? Result::FINISH_OWNER : Result::PENDING;
        ASSERT_EQ(scheduler_async_join_release_block(join).result, expected);
    }
    EXPECT_TRUE(scheduler_async_join_complete(scheduler_async_join_load(join)));
}

TEST(HbgAsyncJoin, UnknownHighBitsCannotGrantFinish) {
    SchedulerAsyncTaskJoin join;
    join.word = (UINT64_C(1) << 48) | (UINT64_C(1) << 16);
    EXPECT_FALSE(scheduler_async_join_valid(join.word));
    EXPECT_EQ(scheduler_async_join_release_block(join).result, Result::INVALID_STATE);
}

TEST(HbgAsyncJoin, AbortRemainsStickyAcrossTheMaximumRemainingContributions) {
    for (bool add_groups : {false, true}) {
        SchedulerAsyncTaskJoin join;
        initialize(join, SCHEDULER_ASYNC_JOIN_LIMIT);
        scheduler_async_join_abort(join);
        for (uint32_t i = 0; i < SCHEDULER_ASYNC_JOIN_LIMIT; ++i) {
            const auto result =
                add_groups ? scheduler_async_join_add_group(join) : scheduler_async_join_settle_group(join);
            ASSERT_EQ(result.result, Result::FAILED);
            ASSERT_EQ(scheduler_async_join_release_block(join).result, Result::FAILED);
        }
        const auto word = scheduler_async_join_load(join);
        EXPECT_NE(word & SCHEDULER_ASYNC_JOIN_FAILED, 0u);
        EXPECT_TRUE(scheduler_async_join_valid(word));
    }
}

TEST(HbgAsyncJoin, ConcurrentAbortsAndInFlightContributionsCannotClearFailure) {
    SchedulerAsyncTaskJoin join;
    initialize(join, 1024);
    for (int i = 0; i < 1024; ++i)
        ASSERT_EQ(scheduler_async_join_add_group(join).result, Result::PENDING);
    scheduler_async_join_abort(join);
    std::atomic<int> errors{0};
    std::vector<std::thread> workers;
    for (int i = 0; i < 8; ++i) {
        workers.emplace_back([&] {
            for (int j = 0; j < 128; ++j) {
                if (scheduler_async_join_settle_group(join).result != Result::FAILED) ++errors;
                if (scheduler_async_join_release_block(join).result != Result::FAILED) ++errors;
            }
        });
    }
    std::thread failer([&] {
        for (int i = 0; i < 64; ++i)
            scheduler_async_join_abort(join);
    });
    for (auto &worker : workers)
        worker.join();
    failer.join();
    EXPECT_EQ(errors.load(), 0);
    EXPECT_NE(scheduler_async_join_load(join) & SCHEDULER_ASYNC_JOIN_FAILED, 0u);
}
}  // namespace
