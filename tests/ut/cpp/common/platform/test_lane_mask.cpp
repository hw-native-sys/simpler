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

#include <cstring>
#include <type_traits>
#include <utility>

#include "common/lane_mask.h"

namespace {
constexpr LaneMask lanes(unsigned combination) {
    return LaneMask((combination & 1) != 0, (combination & 2) != 0, (combination & 4) != 0);
}
template <typename T, typename = void>
struct HasRawBits : std::false_type {};
template <typename T>
struct HasRawBits<T, std::void_t<decltype(std::declval<T>().bits())>> : std::true_type {};
template <typename T, typename = void>
struct HasBitwiseUnion : std::false_type {};
template <typename T>
struct HasBitwiseUnion<T, std::void_t<decltype(std::declval<T>() | std::declval<T>())>> : std::true_type {};
template <typename T, typename = void>
struct HasRawConstructor : std::false_type {};
template <typename T>
struct HasRawConstructor<T, std::void_t<decltype(T::from_bits(uint8_t{}))>> : std::true_type {};
template <typename T, typename = void>
struct HasBitConstant : std::false_type {};
template <typename T>
struct HasBitConstant<T, std::void_t<decltype(T::kAic)>> : std::true_type {};
}  // namespace

static_assert(!HasRawBits<LaneMask>::value && !HasBitwiseUnion<LaneMask>::value);
static_assert(!HasRawConstructor<LaneMask>::value && !HasBitConstant<LaneMask>::value);
static_assert(LaneMask::none().is_empty());
static_assert(LaneMask::aic().has_aic() && !LaneMask::aic().has_any_aiv());
static_assert(LaneMask::aiv0().has_aiv0() && !LaneMask::aiv0().has_aiv1());
static_assert(LaneMask::aiv1().has_aiv1() && !LaneMask::aiv1().has_aiv0());
static_assert(LaneMask::both_aiv().count() == 2 && LaneMask::all().is_full());
static_assert(LaneMask::lanes_of(ResourceShape::AIC) == LaneMask::aic());
static_assert(LaneMask::lanes_of(ResourceShape::AIV) == LaneMask::both_aiv());
static_assert(LaneMask::lanes_of(ResourceShape::MIX) == LaneMask::all());
static_assert(LaneMask::lanes_of(ResourceShape::DUMMY).is_empty());
static_assert(LaneMask::aiv0().lane_demand() == LaneMask::both_aiv());
static_assert(LaneMask::aiv1().lane_demand() == LaneMask::both_aiv());
static_assert(lanes(3).to_shape() == ResourceShape::MIX);
static_assert(lanes(5).rank_of(SubtaskSlot::AIV1) == 1);
static_assert(LaneMask::of(static_cast<SubtaskSlot>(3)).is_empty());
static_assert(!LaneMask::all().has(static_cast<SubtaskSlot>(3)));

TEST(LaneMask, WireValidationChecksEveryByteWithoutConstruction) {
    for (unsigned raw = 0; raw <= UINT8_MAX; ++raw) {
        SCOPED_TRACE(raw);
        const auto byte = static_cast<uint8_t>(raw);
        LaneMask wire;
        std::memcpy(&wire, &byte, sizeof(wire));
        ASSERT_EQ(wire.is_valid(), raw < 8);
        if (wire.is_valid()) EXPECT_EQ(wire, lanes(raw));
    }
}

TEST(LaneMask, SemanticQueriesAndRanksCoverAllEightCombinations) {
    constexpr ResourceShape shapes[] = {
        ResourceShape::DUMMY, ResourceShape::AIC, ResourceShape::AIV, ResourceShape::MIX,
        ResourceShape::AIV,   ResourceShape::MIX, ResourceShape::MIX, ResourceShape::MIX,
    };
    constexpr unsigned demand[] = {0, 1, 6, 3, 6, 5, 6, 7};
    for (unsigned combination = 0; combination < 8; ++combination) {
        SCOPED_TRACE(combination);
        const LaneMask value = lanes(combination);
        EXPECT_TRUE(value.is_valid());
        EXPECT_EQ(value.to_shape(), shapes[combination]);
        EXPECT_EQ(value.lane_demand(), lanes(demand[combination]));
        EXPECT_EQ(value.count(), __builtin_popcount(combination));
        EXPECT_EQ(value.is_mix(), __builtin_popcount(combination) >= 2);
        EXPECT_EQ(value.is_empty(), combination == 0);
        EXPECT_EQ(value.is_full(), combination == 7);
        EXPECT_EQ(value.has_aic(), (combination & 1) != 0);
        EXPECT_EQ(value.has_aiv0(), (combination & 2) != 0);
        EXPECT_EQ(value.has_aiv1(), (combination & 4) != 0);
        EXPECT_EQ(value.has_any_aiv(), (combination & 6) != 0);
        EXPECT_EQ(value.has_both_aiv(), (combination & 6) == 6);
        int rank = 0;
        for (unsigned index = 0; index < SUBTASK_SLOT_COUNT; ++index) {
            const auto slot = static_cast<SubtaskSlot>(index);
            const bool active = (combination & (1u << index)) != 0;
            EXPECT_EQ(value.has(slot), active);
            if (active) EXPECT_EQ(value.rank_of(slot), rank++);
            auto changed = value;
            changed.activate(slot);
            EXPECT_EQ(changed, lanes(combination | (1u << index)));
            changed.activate(slot);
            EXPECT_EQ(changed.count(), value.count() + !active);
            EXPECT_EQ(LaneMask::of(slot).count(), 1);
            EXPECT_TRUE(LaneMask::of(slot).has(slot));
        }
    }
}

TEST(LaneMask, InvalidSlotsNeverAliasOrActivateAValidLane) {
    for (unsigned index = SUBTASK_SLOT_COUNT; index <= UINT8_MAX; ++index) {
        const auto slot = static_cast<SubtaskSlot>(index);
        EXPECT_TRUE(LaneMask::of(slot).is_empty());
        for (unsigned combination = 0; combination < 8; ++combination) {
            auto value = lanes(combination);
            EXPECT_FALSE(value.has(slot));
            value.activate(slot);
            EXPECT_EQ(value, lanes(combination));
        }
    }
}

TEST(LaneMask, DifferencePreservesOnlyUnreservedPhysicalLanes) {
    for (unsigned left = 0; left < 8; ++left) {
        for (unsigned right = 0; right < 8; ++right) {
            LaneMask a{bool(left & 1), bool(left & 2), bool(left & 4)};
            LaneMask b{bool(right & 1), bool(right & 2), bool(right & 4)};
            const unsigned bits = left & ~right;
            EXPECT_EQ(a.without(b), (LaneMask{bool(bits & 1), bool(bits & 2), bool(bits & 4)}));
        }
    }
}
