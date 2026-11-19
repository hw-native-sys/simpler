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
#include <atomic>
#include <cstdint>
#include <thread>
#include <gtest/gtest.h>
#include "device_runner_helpers.h"

namespace {

constexpr uint64_t kMaxEpoch = UINT64_MAX >> 1;

// The start gate does not order the observation against the competing operation.
template <typename Operation>
void race_observation(EntryCacheQualification &receipt, uint64_t epoch, Operation operation) {
    std::atomic<bool> ready{false};
    std::atomic<bool> start{false};
    std::thread observer([&] {
        ready.store(true, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {}
        receipt.observe_success(epoch);
    });
    while (!ready.load(std::memory_order_acquire)) {}
    start.store(true, std::memory_order_release);
    operation();
    observer.join();
}

TEST(EntryCacheQualificationTest, ValidEpochsRequireMatchingObservationAndConsumeOnce) {
    for (uint64_t epoch : {uint64_t{1}, uint64_t{2}, kMaxEpoch - 1, kMaxEpoch}) {
        SCOPED_TRACE(epoch);
        EntryCacheQualification receipt;
        receipt.published(epoch);
        EXPECT_FALSE(receipt.eligible());
        receipt.observe_success(epoch == 1 ? 2 : epoch - 1);
        EXPECT_FALSE(receipt.eligible());
        receipt.observe_success(epoch);
        EXPECT_TRUE(receipt.consume());
        EXPECT_FALSE(receipt.consume());
        receipt.observe_success(epoch);
        EXPECT_FALSE(receipt.consume());
    }
}

TEST(EntryCacheQualificationTest, ZeroAndOverflowEpochsCannotAliasValidPublications) {
    for (uint64_t invalid : {uint64_t{0}, kMaxEpoch + 1, kMaxEpoch + 2, UINT64_MAX}) {
        SCOPED_TRACE(invalid);
        EntryCacheQualification receipt;
        receipt.published(invalid);
        receipt.observe_success(invalid);
        EXPECT_FALSE(receipt.consume());
        receipt.published(invalid);
        receipt.observe_success(1);
        receipt.observe_success(kMaxEpoch);
        EXPECT_FALSE(receipt.consume());
        for (uint64_t valid : {uint64_t{1}, kMaxEpoch}) {
            receipt.published(valid);
            receipt.observe_success(invalid);
            EXPECT_FALSE(receipt.eligible());
            receipt.observe_success(valid);
            EXPECT_TRUE(receipt.consume());
            receipt.published(valid);
            receipt.observe_success(valid);
            receipt.published(invalid);
            EXPECT_FALSE(receipt.consume());
        }
    }
}

TEST(EntryCacheQualificationTest, ConsumeAndClearRejectSubsequentOldObservations) {
    EntryCacheQualification receipt;
    receipt.published(1);
    EXPECT_FALSE(receipt.consume());
    receipt.observe_success(1);
    EXPECT_FALSE(receipt.consume());
    receipt.published(2);
    receipt.observe_success(2);
    receipt.clear();
    receipt.observe_success(2);
    EXPECT_FALSE(receipt.consume());
    receipt.published(3);
    receipt.clear();
    receipt.observe_success(3);
    EXPECT_FALSE(receipt.consume());
}

TEST(EntryCacheQualificationTest, SuccessorPublicationDiscardsOldQualification) {
    for (bool observed : {false, true}) {
        EntryCacheQualification receipt;
        receipt.published(1);
        if (observed) receipt.observe_success(1);
        receipt.published(kMaxEpoch);
        receipt.observe_success(1);
        EXPECT_FALSE(receipt.eligible());
        receipt.observe_success(kMaxEpoch);
        EXPECT_TRUE(receipt.consume());
    }
}

TEST(EntryCacheQualificationTest, ConcurrentConsumeLeavesNoLateQualification) {
    for (int iteration = 0; iteration < 256; ++iteration) {
        SCOPED_TRACE(iteration);
        EntryCacheQualification receipt;
        receipt.published(1);
        // Either consume result is legal; the receipt must be empty after both operations.
        race_observation(receipt, 1, [&] {
            (void)receipt.consume();
        });
        EXPECT_FALSE(receipt.eligible());
        EXPECT_FALSE(receipt.consume());
    }
}

TEST(EntryCacheQualificationTest, ConcurrentPublicationRejectsOldObserver) {
    for (int iteration = 0; iteration < 256; ++iteration) {
        SCOPED_TRACE(iteration);
        EntryCacheQualification receipt;
        receipt.published(1);
        race_observation(receipt, 1, [&] {
            receipt.published(kMaxEpoch);
        });
        EXPECT_FALSE(receipt.eligible());
        receipt.observe_success(kMaxEpoch);
        EXPECT_TRUE(receipt.consume());
        EXPECT_FALSE(receipt.consume());
    }
}

TEST(EntryCacheQualificationTest, ConcurrentClearLeavesNoLateQualification) {
    for (int iteration = 0; iteration < 256; ++iteration) {
        SCOPED_TRACE(iteration);
        EntryCacheQualification receipt;
        receipt.published(kMaxEpoch);
        race_observation(receipt, kMaxEpoch, [&] {
            receipt.clear();
        });
        EXPECT_FALSE(receipt.eligible());
        EXPECT_FALSE(receipt.consume());
    }
}

}  // namespace
