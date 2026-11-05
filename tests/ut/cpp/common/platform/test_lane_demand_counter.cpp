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
#include <array>
#include <thread>
#include <vector>
#include "common/tagged_ptr.h"
#include "common/lane_demand_counter.h"

namespace {
struct alignas(64) TestSlot {};
using LaneTaggedSlot = TaggedPtr<TestSlot, LaneMask, SUBTASK_SLOT_COUNT>;
static_assert(std::is_trivially_default_constructible_v<LaneTaggedSlot>);
static_assert(std::is_trivially_copyable_v<LaneTaggedSlot> && std::is_standard_layout_v<LaneTaggedSlot>);
constexpr LaneMask lanes_for(unsigned combination) {
    return {(combination & 1) != 0, (combination & 2) != 0, (combination & 4) != 0};
}
}  // namespace

TEST(LaneReservation, NamedUnionAndContainmentCoverAllCombinations) {
    for (unsigned left = 0; left < 8; ++left) {
        for (unsigned right = 0; right < 8; ++right) {
            const auto a = lanes_for(left);
            const auto b = lanes_for(right);
            EXPECT_EQ(a.merged_with(b), lanes_for(left | right));
            EXPECT_EQ(a.contains(b), (left & right) == right);
        }
    }
}

TEST(TaggedSlot, RoundTripsEveryTagAndRequiresExplicitPointerAccess) {
    TestSlot slot{};
    for (uint8_t bits = 0; bits < 8; ++bits) {
        const LaneMask tag = lanes_for(bits);
        LaneTaggedSlot handle(&slot, tag);
        EXPECT_EQ(handle.ptr(), &slot);
        EXPECT_EQ(handle.tag(), tag);
        EXPECT_FALSE(handle.is_null());
    }
    LaneTaggedSlot empty{};
    EXPECT_TRUE(empty.is_null());
    EXPECT_TRUE(empty.tag().is_empty());
}

TEST(TaggedSlot, TypedSlotArraysPreserveTagAlignment) {
    static_assert(alignof(TestSlot) >= (size_t{1} << SUBTASK_SLOT_COUNT));
    TestSlot slots[8]{};
    for (auto &slot : slots) {
        EXPECT_EQ(reinterpret_cast<uintptr_t>(&slot) % alignof(TestSlot), 0u);
        LaneTaggedSlot handle(&slot, LaneMask::all());
        EXPECT_EQ(handle.ptr(), &slot);
        EXPECT_EQ(handle.tag(), LaneMask::all());
    }
}

TEST(LaneDemand, AllEightDeltasAndReset) {
    LaneDemandCounter counter;
    const LaneMask lanes[] = {LaneMask::aic(), LaneMask::aiv0(), LaneMask::aiv1()};
    for (uint8_t bits = 0; bits < 8; ++bits) {
        const LaneMask demand = lanes_for(bits);
        counter.add(demand);
        EXPECT_EQ(counter.reserved(), demand);
        for (LaneMask lane : lanes)
            EXPECT_EQ(counter.count(lane), demand.contains(lane) ? 1u : 0u);
        counter.sub(demand);
        EXPECT_TRUE(counter.reserved().is_empty());
    }
    counter.add(LaneMask::all());
    counter.reset();
    EXPECT_TRUE(counter.reserved().is_empty());
    for (LaneMask lane : lanes)
        EXPECT_EQ(counter.count(lane), 0u);
}

TEST(LaneDemand, EachFieldReachesItsLimitWithoutChangingPeers) {
    const LaneMask lanes[] = {LaneMask::aic(), LaneMask::aiv0(), LaneMask::aiv1()};
    for (LaneMask lane : lanes) {
        LaneDemandCounter counter;
        for (uint32_t i = 0; i < LaneDemandCounter::kMaxCount; ++i)
            counter.add(lane);
        for (LaneMask peer : lanes)
            EXPECT_EQ(counter.count(peer), peer == lane ? LaneDemandCounter::kMaxCount : 0u);
        for (uint32_t i = 0; i < LaneDemandCounter::kMaxCount; ++i)
            counter.sub(lane);
        EXPECT_TRUE(counter.reserved().is_empty());
    }
}

TEST(LaneDemand, BatchAndSingleUpdatesAreEquivalent) {
    LaneDemandCounter batch, singles;
    LaneDemandCounter::Batch empty, aggregate;
    std::array<LaneMask, 8> demands;
    batch.add(empty);
    batch.sub(empty);
    for (uint8_t bits = 0; bits < 8; ++bits) {
        demands[bits] = lanes_for(bits);
        aggregate.add(demands[bits]);
    }
    batch.add(aggregate);
    for (LaneMask demand : demands)
        singles.add(demand);
    for (LaneMask lane : {LaneMask::aic(), LaneMask::aiv0(), LaneMask::aiv1()}) {
        EXPECT_EQ(batch.count(lane), 4u);
        EXPECT_EQ(batch.count(lane), singles.count(lane));
    }
    for (LaneMask demand : demands)
        batch.sub(demand);
    singles.sub(aggregate);
    EXPECT_TRUE(batch.reserved().is_empty());
    EXPECT_TRUE(singles.reserved().is_empty());
}

#ifndef NDEBUG
TEST(LaneDemand, RejectsInvalidCountLane) {
    LaneDemandCounter counter;
    EXPECT_ANY_THROW(counter.count(LaneMask::both_aiv()));
}
#endif

TEST(LaneDemand, ConcurrentPairedUpdatesReturnEveryFieldToZero) {
    LaneDemandCounter counter;
    std::vector<std::thread> threads;
    for (int worker = 0; worker < 8; ++worker)
        threads.emplace_back([&, worker] {
            const LaneMask demand = lanes_for(worker);
            for (int i = 0; i < 10000; ++i) {
                counter.add(demand);
                counter.sub(demand);
            }
        });
    for (auto &thread : threads)
        thread.join();
    EXPECT_TRUE(counter.reserved().is_empty());
    for (LaneMask lane : {LaneMask::aic(), LaneMask::aiv0(), LaneMask::aiv1()})
        EXPECT_EQ(counter.count(lane), 0u);
}

TEST(LaneDemand, BatchAtFieldLimitKeepsOtherLanesIndependent) {
    LaneDemandCounter counter;
    LaneDemandCounter::Batch batch;
    for (uint32_t i = 0; i < LaneDemandCounter::kMaxCount; ++i)
        batch.add(LaneMask::aiv1());
    counter.add(batch);
    EXPECT_EQ(counter.count(LaneMask::aic()), 0u);
    EXPECT_EQ(counter.count(LaneMask::aiv0()), 0u);
    EXPECT_EQ(counter.count(LaneMask::aiv1()), LaneDemandCounter::kMaxCount);
    counter.sub(batch);
    EXPECT_TRUE(counter.reserved().is_empty());
}
