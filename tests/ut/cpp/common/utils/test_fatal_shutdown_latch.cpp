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

#include "utils/fatal_shutdown_latch.h"

namespace {

TEST(FatalShutdownLatchTest, CompletionNeverBecomesVisibleBeforeFatalState) {
    for (int iteration = 0; iteration < 1000; ++iteration) {
        std::atomic<bool> fatal_started{false};
        std::atomic<bool> completed{false};
        std::atomic<bool> observed_fatal{false};

        std::thread observer([&]() {
            while (!completed.load(std::memory_order_acquire)) {
                std::this_thread::yield();
            }
            observed_fatal.store(fatal_started.load(std::memory_order_acquire), std::memory_order_relaxed);
        });
        EXPECT_TRUE(publish_fatal_shutdown(fatal_started, completed));
        observer.join();

        EXPECT_TRUE(observed_fatal.load(std::memory_order_relaxed));
    }
}

TEST(FatalShutdownLatchTest, ExactlyOneCallerOwnsTheEmergencySignalBroadcast) {
    std::atomic<bool> fatal_started{false};
    std::atomic<bool> completed{false};
    std::atomic<int> leaders{0};
    std::vector<std::thread> publishers;

    for (int i = 0; i < 16; ++i) {
        publishers.emplace_back([&]() {
            if (publish_fatal_shutdown(fatal_started, completed)) {
                leaders.fetch_add(1, std::memory_order_relaxed);
            }
        });
    }
    for (auto &publisher : publishers) {
        publisher.join();
    }

    EXPECT_TRUE(fatal_started.load(std::memory_order_acquire));
    EXPECT_TRUE(completed.load(std::memory_order_acquire));
    EXPECT_EQ(leaders.load(std::memory_order_relaxed), 1);
}

TEST(FatalShutdownLatchTest, SdmaProvisionedCardGetsASingleResetAttempt) {
    int attempts = 0;
    int rc = attempt_fatal_reset(
        [&]() {
            ++attempts;
            return 507007;
        },
        1
    );

    EXPECT_EQ(rc, 507007);
    EXPECT_EQ(attempts, 1);
}

TEST(FatalShutdownLatchTest, OrdinaryPoisonRetriesUpToTheAttemptBudget) {
    int attempts = 0;
    int rc = attempt_fatal_reset(
        [&]() {
            ++attempts;
            return 507007;
        },
        3
    );

    EXPECT_EQ(rc, 507007);
    EXPECT_EQ(attempts, 3);
}

TEST(FatalShutdownLatchTest, RetryStopsOnTheFirstAttemptThatConfirmsTheCard) {
    int attempts = 0;
    int rc = attempt_fatal_reset(
        [&]() {
            return ++attempts == 2 ? 0 : 507007;
        },
        3
    );

    EXPECT_EQ(rc, 0);
    EXPECT_EQ(attempts, 2);
}

TEST(FatalShutdownLatchTest, AttemptBudgetBelowOneStillResetsOnce) {
    int attempts = 0;
    int rc = attempt_fatal_reset(
        [&]() {
            ++attempts;
            return 0;
        },
        0
    );

    EXPECT_EQ(rc, 0);
    EXPECT_EQ(attempts, 1);
}

}  // namespace
