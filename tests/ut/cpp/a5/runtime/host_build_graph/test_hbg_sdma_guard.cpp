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

#include "backend/sdma/sdma_submission_guard.h"

namespace {
using namespace pto2::resident;

TEST(HbgSdmaGuard, SdmaGuardSerializesSessionStateAndRetainsFailure) {
    AsyncSdmaSubmissionGuard guard{};
    uint64_t post_id = 0;
    std::atomic<bool> failed{false};
    std::vector<std::thread> submitters;
    for (uint32_t lane = 0; lane < 2; ++lane) {
        submitters.emplace_back([&]() {
            for (uint32_t iteration = 0; iteration < 2000; ++iteration) {
                SdmaGuardResult result;
                do {
                    result = sdma_try_acquire(guard);
                } while (result == SdmaGuardResult::BUSY);
                if (result != SdmaGuardResult::ACQUIRED) {
                    failed = true;
                    return;
                }
                ++post_id;
                if (!sdma_release(guard, true)) failed = true;
            }
        });
    }
    for (auto &thread : submitters)
        thread.join();
    EXPECT_FALSE(failed);
    EXPECT_EQ(post_id, 4000u);
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::ACQUIRED);
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::BUSY);
    EXPECT_TRUE(sdma_release(guard, false));
    EXPECT_EQ(sdma_try_acquire(guard), SdmaGuardResult::FAILED);
    EXPECT_EQ(guard.word, 2u);
}
}  // namespace
