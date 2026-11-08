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

#include "platform_comm/ccu/ccu_run_lease.h"

namespace {
using namespace simpler::ccu;

class FakeLease final : public RunLease {
public:
    ReaderResult next{ReaderState::PENDING, 0};
    int poll_count = 0;
    int drain_count = 0;
    ReaderResult poll() noexcept override {
        ++poll_count;
        return next;
    }
    ReaderResult drain() noexcept override {
        ++drain_count;
        return next;
    }
};

TEST(CcuRunReaders, NoCcuDoesNotChangeOrdinaryCompletion) {
    RunReaders readers;
    EXPECT_TRUE(readers.stopped());
    EXPECT_EQ(readers.poll().state, ReaderState::STOPPED);
}

TEST(CcuRunReaders, UnconfirmedKeepsOwnerAndRequiresNewStopProof) {
    RunReaders readers;
    auto lease = std::make_shared<FakeLease>();
    const std::weak_ptr<RunLease> weak = lease;
    readers.lease = lease;
    EXPECT_FALSE(readers.stopped());
    EXPECT_EQ(readers.poll().state, ReaderState::PENDING);
    lease->next = {ReaderState::UNCONFIRMED, -106};
    EXPECT_EQ(readers.drain().error_code, -106);
    EXPECT_FALSE(readers.stopped());
    lease.reset();
    EXPECT_FALSE(weak.expired());
    EXPECT_EQ(readers.poll().state, ReaderState::UNCONFIRMED);
}

TEST(CcuRunReaders, ConfirmedFailureIsStickyAcrossMoveAndNeverBecomesSuccess) {
    RunReaders readers;
    auto lease = std::make_shared<FakeLease>();
    readers.lease = lease;
    lease->next = {ReaderState::STOPPED, -105};
    EXPECT_EQ(readers.drain().error_code, -105);
    EXPECT_TRUE(readers.stopped());
    RunReaders moved = std::move(readers);
    lease->next = {ReaderState::STOPPED, 0};
    EXPECT_EQ(moved.poll().error_code, -105);
    EXPECT_EQ(moved.drain().error_code, -105);
    EXPECT_EQ(lease->drain_count, 1);
    EXPECT_EQ(lease->poll_count, 0);
}

}  // namespace
