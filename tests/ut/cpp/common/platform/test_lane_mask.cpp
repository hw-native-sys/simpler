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

#include "common/lane_mask.h"

static_assert(LaneMask::none().bits() == 0);
static_assert(LaneMask::aic().bits() == 1);
static_assert(LaneMask::aiv0().bits() == 2);
static_assert(LaneMask::aiv1().bits() == 4);
static_assert(LaneMask::both_aiv().bits() == 6);
static_assert(LaneMask::all().bits() == 7);
static_assert(LaneMask::aic().complement() == LaneMask::both_aiv());
static_assert(LaneMask::lanes_of(ResourceShape::AIC) == LaneMask::aic());
static_assert(LaneMask::lanes_of(ResourceShape::AIV) == LaneMask::both_aiv());
static_assert(LaneMask::lanes_of(ResourceShape::MIX) == LaneMask::all());
static_assert(LaneMask::lanes_of(ResourceShape::DUMMY).is_empty());
static_assert(LaneMask::aiv0().lane_demand() == LaneMask::both_aiv());
static_assert(LaneMask::aiv1().lane_demand() == LaneMask::both_aiv());
static_assert(LaneMask::from_bits(3).to_shape() == ResourceShape::MIX);

TEST(LaneMask, RawBoundaryRejectsEveryInvalidEncoding) {
    for (unsigned raw = 0; raw <= UINT8_MAX; ++raw) {
        SCOPED_TRACE(raw);
        const auto decoded = LaneMask::try_from_bits(static_cast<uint8_t>(raw));
        ASSERT_EQ(decoded.has_value(), raw < 8);
        if (decoded.has_value()) {
            EXPECT_EQ(decoded->bits(), raw);
            EXPECT_EQ(LaneMask::from_bits(static_cast<uint8_t>(raw)), *decoded);
        } else {
            EXPECT_THROW(LaneMask::from_bits(static_cast<uint8_t>(raw)), AssertionError);
        }
    }
}

TEST(LaneMask, ShapeQueriesAndDemandPreserveAllEightValues) {
    constexpr ResourceShape shapes[] = {
        ResourceShape::DUMMY, ResourceShape::AIC, ResourceShape::AIV, ResourceShape::MIX,
        ResourceShape::AIV,   ResourceShape::MIX, ResourceShape::MIX, ResourceShape::MIX,
    };
    constexpr uint8_t demand[] = {0, 1, 6, 3, 6, 5, 6, 7};
    for (uint8_t raw = 0; raw < 8; ++raw) {
        SCOPED_TRACE(raw);
        const auto lanes = LaneMask::from_bits(raw);
        EXPECT_EQ(lanes.to_shape(), shapes[raw]);
        EXPECT_EQ(lanes.lane_demand().bits(), demand[raw]);
        EXPECT_EQ(lanes.count(), __builtin_popcount(raw));
        EXPECT_EQ(lanes.is_mix(), __builtin_popcount(raw) >= 2);
        EXPECT_EQ(lanes.is_empty(), raw == 0);
        EXPECT_EQ(lanes.is_full(), raw == 7);
        EXPECT_EQ(lanes.has_aic(), (raw & 1) != 0);
        EXPECT_EQ(lanes.has_aiv0(), (raw & 2) != 0);
        EXPECT_EQ(lanes.has_aiv1(), (raw & 4) != 0);
        EXPECT_EQ(lanes.has_any_aiv(), (raw & 6) != 0);
        EXPECT_EQ(lanes.has_both_aiv(), (raw & 6) == 6);
        for (int slot = 0; slot < SUBTASK_SLOT_COUNT; ++slot) {
            const auto subtask = static_cast<SubtaskSlot>(slot);
            EXPECT_EQ(lanes.has(subtask), (raw & (1u << slot)) != 0);
            EXPECT_EQ(LaneMask::of(subtask).bits(), 1u << slot);
        }
    }
}

TEST(LaneMask, SetOperationsAreClosedOverValidLanes) {
    for (uint8_t left = 0; left < 8; ++left) {
        const auto a = LaneMask::from_bits(left);
        EXPECT_EQ(a.complement().bits(), uint8_t{7} ^ left);
        for (uint8_t right = 0; right < 8; ++right) {
            SCOPED_TRACE(testing::Message() << "left=" << int(left) << " right=" << int(right));
            const auto b = LaneMask::from_bits(right);
            EXPECT_EQ((a | b).bits(), left | right);
            EXPECT_EQ((a & b).bits(), left & right);
            EXPECT_EQ(a.without(b).bits(), left & ~right);
            EXPECT_EQ(a.contains(b), (left & right) == right);
            EXPECT_EQ(a.intersects(b), (left & right) != 0);
            auto joined = a;
            joined |= b;
            EXPECT_EQ(joined, a | b);
            auto common = a;
            common &= b;
            EXPECT_EQ(common, a & b);
        }
    }
}
