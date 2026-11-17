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
#include "host/scheduler_notification_cache.h"
#include "host/scheduler_notification_address.h"
#include <gtest/gtest.h>

namespace {
struct Counters {
    int created{0};
    int prepared{0};
    int released{0};
    int abandoned{0};
    bool fail_release{false};
    size_t capacity_limit{SIZE_MAX};
};
class FakePool final : public SchedulerNotificationStorage {
public:
    FakePool(void *context, SchedulerNotificationPurpose purpose, Counters &counts) :
        SchedulerNotificationStorage(context),
        counts_(counts),
        counter_(purpose == SchedulerNotificationPurpose::COMPLETION_COUNTER) {
        ++counts_.created;
    }
    size_t capacity() const override { return capacity_; }
    SchedulerDoneNotifications prepare(size_t count) override {
        ++counts_.prepared;
        capacity_ = std::min(std::max(capacity_, count), counts_.capacity_limit);
        SchedulerDoneNotifications result{std::vector<uint64_t>(count, 0), std::vector<uint64_t>(count, 0), {}};
        std::fill_n(result.addresses.begin(), std::min(capacity_, count), 32);
        if (counter_) std::fill_n(result.add_addresses.begin(), std::min(capacity_, count), 40);
        return result;
    }
    int release() override {
        ++counts_.released;
        return counts_.fail_release ? -7 : 0;
    }
    void abandon() noexcept override { ++counts_.abandoned; }

private:
    Counters &counts_;
    const bool counter_;
    size_t capacity_{0};
};
struct Fixture {
    SchedulerNotificationCache cache;
    Counters counts;
    int context{};
    int acquire(
        SchedulerDoneNotifications &out, size_t size = 8,
        SchedulerNotificationPurpose purpose = SchedulerNotificationPurpose::TASK_STATE
    ) {
        return cache.acquire(
            0, purpose, &context, size,
            [&](void *ctx, SchedulerNotificationPurpose purpose) {
                return std::make_shared<FakePool>(ctx, purpose, counts);
            },
            &out
        );
    }
};

TEST(SchedulerNotificationCache, CapturedLeaseForcesAnotherPoolAndBlocksRelease) {
    Fixture f;
    SchedulerDoneNotifications first, second;
    ASSERT_EQ(f.acquire(first), 0);
    auto captured = first.owner;
    first = {};
    ASSERT_EQ(f.acquire(second), 0);
    EXPECT_NE(second.owner, captured);
    EXPECT_EQ(f.counts.created, 2);
    EXPECT_NE(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 0);
    captured.reset();
    second = {};
    EXPECT_EQ(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 2);
}

TEST(SchedulerNotificationCache, ReusesAndGrowsUnleasedPoolWithIndependentPurpose) {
    Fixture f;
    SchedulerDoneNotifications first;
    ASSERT_EQ(f.acquire(first), 0);
    auto *identity = first.owner.get();
    first = {};
    ASSERT_EQ(f.acquire(first, 16), 0);
    EXPECT_EQ(first.owner.get(), identity);
    EXPECT_EQ(first.add_addresses[14], 0u);
    SchedulerDoneNotifications mix;
    ASSERT_EQ(f.acquire(mix, 6, SchedulerNotificationPurpose::MIX_ACK), 0);
    EXPECT_NE(mix.owner, first.owner);
    EXPECT_EQ(f.counts.created, 2);
    first = {};
    mix = {};
    EXPECT_EQ(f.cache.release(), 0);
}

TEST(SchedulerNotificationCache, ReleaseFailureRetainsHandleForRetry) {
    Fixture f;
    SchedulerDoneNotifications lease;
    ASSERT_EQ(f.acquire(lease), 0);
    lease = {};
    f.counts.fail_release = true;
    EXPECT_EQ(f.cache.release(), -7);
    EXPECT_EQ(f.counts.released, 1);
    f.counts.fail_release = false;
    EXPECT_EQ(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 2);
    EXPECT_EQ(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 2);
}

TEST(SchedulerNotificationCache, PostResetAbandonDoesNotCallDeviceRelease) {
    Fixture f;
    SchedulerDoneNotifications lease;
    ASSERT_EQ(f.acquire(lease), 0);
    f.cache.abandon();
    EXPECT_EQ(f.counts.abandoned, 1);
    EXPECT_EQ(f.counts.released, 0);
    lease = {};
    EXPECT_EQ(f.counts.released, 0);
    EXPECT_EQ(f.cache.release(), 0);
}

TEST(SchedulerNotificationCache, RefusesReuseAcrossContextGenerations) {
    Fixture f;
    SchedulerDoneNotifications lease;
    ASSERT_EQ(f.acquire(lease), 0);
    lease = {};
    int another_context = 1;
    EXPECT_NE(
        f.cache.acquire(
            0, SchedulerNotificationPurpose::TASK_STATE, &another_context, 8,
            [&](void *ctx, SchedulerNotificationPurpose purpose) {
                return std::make_shared<FakePool>(ctx, purpose, f.counts);
            },
            &lease
        ),
        0
    );
    EXPECT_EQ(f.counts.created, 1);
    EXPECT_EQ(f.cache.release(), 0);
}
TEST(SchedulerNotificationCache, CounterAliasesArePurposeSpecific) {
    Fixture f;
    SchedulerDoneNotifications counter;
    ASSERT_EQ(f.acquire(counter, 4, SchedulerNotificationPurpose::COMPLETION_COUNTER), 0);
    EXPECT_EQ(counter.addresses[0], 32u);
    EXPECT_EQ(counter.add_addresses[0], 40u);
    counter = {};
    EXPECT_EQ(f.cache.release(), 0);
}

TEST(SchedulerNotificationCache, CapacityFallbackKeepsWholeCounterPairsAndLease) {
    Fixture f;
    f.counts.capacity_limit = 2;
    SchedulerDoneNotifications counter;
    ASSERT_EQ(f.acquire(counter, 4, SchedulerNotificationPurpose::COMPLETION_COUNTER), 0);
    ASSERT_EQ(counter.addresses.size(), 4u);
    ASSERT_EQ(counter.add_addresses.size(), 4u);
    EXPECT_EQ(counter.addresses[0], 32u);
    EXPECT_EQ(counter.add_addresses[0], 40u);
    EXPECT_EQ(counter.addresses[2], 0u);
    EXPECT_EQ(counter.add_addresses[2], 0u);
    EXPECT_NE(counter.owner, nullptr);
    EXPECT_NE(f.cache.release(), 0);
    counter = {};
    EXPECT_EQ(f.cache.release(), 0);
    EXPECT_EQ(f.counts.released, 1);
}
}  // namespace

TEST(SchedulerNotificationAddress, UnalignedSetMappingsDoNotExposeAnIncrementAlias) {
    EXPECT_EQ(a5_cnt_notify_increment_address(0), 0u);
    EXPECT_EQ(a5_cnt_notify_increment_address(UINT64_C(0x1004)), 0u);
    EXPECT_EQ(a5_cnt_notify_increment_address(UINT64_C(0x101c)), 0u);
}

TEST(SchedulerNotificationAddress, AlignedSetMappingsKeepTheVerifiedSlotOffset) {
    EXPECT_EQ(a5_cnt_notify_increment_address(UINT64_C(0x1000)), UINT64_C(0x1008));
    EXPECT_EQ(a5_cnt_notify_increment_address(UINT64_C(0x1020)), UINT64_C(0x1028));
    EXPECT_EQ(a5_cnt_notify_increment_address(UINT64_MAX - 31), UINT64_MAX - 23);
}
