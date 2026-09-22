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
#include <mutex>
#include <thread>

#include "utils/thread_completion_gate.h"

TEST(ThreadCompletionGateTest, CleanupCannotBeClaimedWhileFinalizerIsRunning) {
    simpler::ThreadCompletionGate gate;
    gate.arrive_and_finalize_if_last(2, [] {});

    std::mutex mutex;
    std::condition_variable condition;
    bool finalizer_started = false;
    bool allow_finalizer_to_finish = false;

    std::thread last_thread([&] {
        gate.arrive_and_finalize_if_last(2, [&] {
            std::unique_lock<std::mutex> lock(mutex);
            finalizer_started = true;
            condition.notify_one();
            condition.wait(lock, [&] {
                return allow_finalizer_to_finish;
            });
        });
    });

    {
        std::unique_lock<std::mutex> lock(mutex);
        condition.wait(lock, [&] {
            return finalizer_started;
        });
    }
    EXPECT_FALSE(gate.claim_cleanup());

    {
        std::lock_guard<std::mutex> lock(mutex);
        allow_finalizer_to_finish = true;
    }
    condition.notify_one();
    last_thread.join();

    EXPECT_TRUE(gate.claim_cleanup());
    EXPECT_FALSE(gate.claim_cleanup());
}

TEST(ThreadCompletionGateTest, ResetAllowsAnotherRun) {
    simpler::ThreadCompletionGate gate;
    int finalized = 0;

    gate.arrive_and_finalize_if_last(1, [&] {
        ++finalized;
    });
    ASSERT_TRUE(gate.claim_cleanup());

    gate.reset();
    gate.arrive_and_finalize_if_last(1, [&] {
        ++finalized;
    });
    EXPECT_TRUE(gate.claim_cleanup());
    EXPECT_EQ(finalized, 2);
}
