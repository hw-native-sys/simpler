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
#include <array>
#include <thread>
#include <vector>
#include "backend/sdma/sdma_submission_guard.h"

namespace {
using namespace pto2::detail;
struct GuardImage {
    AsyncSdmaGuardContext context;
    AsyncSdmaSubmissionGuard guards[2];
    uint32_t count{0};
    int32_t error{0};
    DeferredCompletionEntry entries[64]{};
    GuardImage() {
        context.run_epoch = 11;
        context.guard_count = 2;
        context.bytes = 3 * 128;
    }
    AsyncCtx ctx() {
        return {&count, &error, entries, 64, TaskId::make_global(1), reinterpret_cast<uint64_t>(&context)};
    }
};
struct FastClock {
    inline static uint64_t ticks{0};
    static uint64_t now() { return ticks += PLATFORM_PROF_SYS_CNT_FREQ; }
};
struct Publisher {
    AsyncSdmaSubmissionGuard *guard;
    bool *published;
    void publish(uint8_t *) const {
        EXPECT_EQ(guard->word, 1u);
        *published = true;
    }
};

TEST(HbgSdmaSubmissionGuard, PublicationPrecedesUnlockAndAnotherOwnerCanEnter) {
    GuardImage image;
    auto ctx = image.ctx();
    bool published = false;
    SdmaSubmissionScope owner;
    ASSERT_TRUE(owner.acquire(ctx, 0));
    EXPECT_EQ(sdma_guard_try_acquire(image.guards[0]), SdmaGuardResult::BUSY);
    ASSERT_TRUE(owner.release(true, nullptr, Publisher{&image.guards[0], &published}));
    EXPECT_TRUE(published);
    EXPECT_EQ(sdma_guard_try_acquire(image.guards[0]), SdmaGuardResult::ACQUIRED);
    EXPECT_TRUE(sdma_guard_release(image.guards[0], true));
}

TEST(HbgSdmaSubmissionGuard, FailedOrAbandonedSubmissionPoisonsOnlyItsChannelGroup) {
    for (bool explicit_failure : {false, true}) {
        GuardImage image;
        auto ctx = image.ctx();
        {
            SdmaSubmissionScope owner;
            ASSERT_TRUE(owner.acquire(ctx, 0));
            if (explicit_failure) EXPECT_TRUE(owner.release(false, nullptr));
        }
        EXPECT_EQ(sdma_guard_try_acquire(image.guards[0]), SdmaGuardResult::FAILED);
        EXPECT_EQ(sdma_guard_try_acquire(image.guards[1]), SdmaGuardResult::ACQUIRED);
        EXPECT_TRUE(sdma_guard_release(image.guards[1], true));
    }
}

TEST(HbgSdmaSubmissionGuard, ContentionTimeoutDoesNotUnlockTheOtherOwner) {
    GuardImage image;
    auto ctx = image.ctx();
    ASSERT_EQ(sdma_guard_try_acquire(image.guards[0]), SdmaGuardResult::ACQUIRED);
    FastClock::ticks = 0;
    {
        SdmaSubmissionScope contender;
        EXPECT_FALSE(contender.acquire<FastClock>(ctx, 0));
    }
    EXPECT_EQ(image.error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
    EXPECT_EQ(image.guards[0].word, 1u);
    EXPECT_TRUE(sdma_guard_release(image.guards[0], true));
}

TEST(HbgSdmaSubmissionGuard, PartialQueueSetTimeoutReleasesOnlyUnsubmittedOwnership) {
    GuardImage image;
    auto ctx = image.ctx();
    ASSERT_EQ(sdma_guard_try_acquire(image.guards[1]), SdmaGuardResult::ACQUIRED);
    FastClock::ticks = 0;
    SdmaSubmissionScope contender;
    EXPECT_FALSE(contender.acquire<FastClock>(ctx, 0, 2));
    EXPECT_EQ(image.guards[0].word, 0u);
    EXPECT_EQ(image.guards[1].word, 1u);
    EXPECT_TRUE(sdma_guard_release(image.guards[1], true));
}

TEST(HbgSdmaSubmissionGuard, MultiQueueSubmissionOwnsAndPoisonsTheEntireOverlap) {
    GuardImage image;
    auto ctx = image.ctx();
    SdmaSubmissionScope owner;
    ASSERT_TRUE(owner.acquire(ctx, 0, 2));
    EXPECT_EQ(sdma_guard_try_acquire(image.guards[0]), SdmaGuardResult::BUSY);
    EXPECT_EQ(sdma_guard_try_acquire(image.guards[1]), SdmaGuardResult::BUSY);
    ASSERT_TRUE(owner.release(false, nullptr));
    EXPECT_EQ(sdma_guard_try_acquire(image.guards[0]), SdmaGuardResult::FAILED);
    EXPECT_EQ(sdma_guard_try_acquire(image.guards[1]), SdmaGuardResult::FAILED);
}

TEST(HbgSdmaSubmissionGuard, InvalidMetadataAndOutOfRangeGroupFailBeforeAcquiring) {
    for (uint32_t fault = 0; fault < 6; ++fault) {
        GuardImage image;
        auto ctx = image.ctx();
        uint32_t group = 0;
        if (fault == 0) image.context.version = 0;
        if (fault == 1) image.context.run_epoch = 0;
        if (fault == 2) image.context.guard_count = 0;
        if (fault == 3) image.context.bytes = 128;
        if (fault == 4) group = 2;
        if (fault == 5) image.context.guard_count = ASYNC_SDMA_CHANNELS + 1;
        SdmaSubmissionScope owner;
        EXPECT_FALSE(owner.acquire(ctx, group));
        EXPECT_EQ(image.error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        EXPECT_EQ(image.guards[0].word, 0u);
    }
}

TEST(HbgSdmaSubmissionGuard, CompetingWritersProduceOneUniqueSequence) {
    AsyncSdmaSubmissionGuard guard;
    uint64_t next_post = 0;
    std::array<std::vector<uint64_t>, 2> ids;
    std::array<std::thread, 2> workers;
    for (uint32_t lane = 0; lane < 2; ++lane)
        workers[lane] = std::thread([&, lane] {
            for (uint32_t i = 0; i < 256; ++i) {
                while (sdma_guard_try_acquire(guard) == SdmaGuardResult::BUSY) {}
                ids[lane].push_back(++next_post);
                EXPECT_TRUE(sdma_guard_release(guard, true));
            }
        });
    for (auto &worker : workers)
        worker.join();
    auto all = ids[0];
    all.insert(all.end(), ids[1].begin(), ids[1].end());
    std::sort(all.begin(), all.end());
    ASSERT_EQ(all.size(), 512u);
    for (size_t i = 0; i < all.size(); ++i)
        EXPECT_EQ(all[i], i + 1);
}
}  // namespace
