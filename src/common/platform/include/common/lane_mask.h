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

#pragma once

#include <cstdint>
#include <type_traits>

#if !defined(__CCE_AICORE__)
#include <atomic>
#include <optional>

#include "assert_compat.h"
#endif

#if defined(__CCE_AICORE__)
#define SIMPLER_LANE_AICORE [aicore]
#else
#define SIMPLER_LANE_AICORE
#endif

inline constexpr int32_t SUBTASK_SLOT_COUNT = 3;

enum class SubtaskSlot : uint8_t {
    AIC = 0,
    AIV0 = 1,
    AIV1 = 2,
};

enum class ResourceShape : uint8_t {
    AIC = 0,
    AIV = 1,
    MIX = 2,
    DUMMY = 3,
};

// DUMMY completes inline and has a separate queue, without core allocation.
inline constexpr int32_t NUM_RESOURCE_SHAPES = 3;

// AIC/AIV0/AIV1 lanes. Construction and set operations preserve the low-three-bit invariant.
class LaneMask {
public:
    static constexpr int kBitWidth = 3;
    static constexpr uint8_t kAic = 1u << 0;
    static constexpr uint8_t kAiv0 = 1u << 1;
    static constexpr uint8_t kAiv1 = 1u << 2;

    SIMPLER_LANE_AICORE constexpr LaneMask() = default;

    SIMPLER_LANE_AICORE static constexpr LaneMask none() { return LaneMask(0); }
    SIMPLER_LANE_AICORE static constexpr LaneMask aic() { return LaneMask(kAic); }
    SIMPLER_LANE_AICORE static constexpr LaneMask aiv0() { return LaneMask(kAiv0); }
    SIMPLER_LANE_AICORE static constexpr LaneMask aiv1() { return LaneMask(kAiv1); }
    SIMPLER_LANE_AICORE static constexpr LaneMask both_aiv() { return LaneMask(kAiv0 | kAiv1); }
    SIMPLER_LANE_AICORE static constexpr LaneMask all() { return LaneMask(kAll); }
    SIMPLER_LANE_AICORE static constexpr LaneMask of(SubtaskSlot slot) {
        return slot == SubtaskSlot::AIC ? aic() : (slot == SubtaskSlot::AIV0 ? aiv0() : aiv1());
    }

    SIMPLER_LANE_AICORE static constexpr LaneMask lanes_of(ResourceShape shape) {
        switch (shape) {
        case ResourceShape::AIC:
            return aic();
        case ResourceShape::AIV:
            return both_aiv();
        case ResourceShape::MIX:
            return all();
        case ResourceShape::DUMMY:
            return none();
        }
        return none();
    }

#if !defined(__CCE_AICORE__)
    // Raw values enter at Host/AICPU boundaries; invalid high bits are rejected, never masked away.
    static constexpr LaneMask from_bits(uint8_t bits) {
        always_assert((bits & ~kAll) == 0 && "LaneMask has bits above AIV1");
        return LaneMask(bits);
    }

    static constexpr std::optional<LaneMask> try_from_bits(uint8_t bits) {
        return (bits & ~kAll) == 0 ? std::optional<LaneMask>(LaneMask(bits)) : std::nullopt;
    }
#endif

    SIMPLER_LANE_AICORE constexpr bool has_aic() const { return (bits_ & kAic) != 0; }
    SIMPLER_LANE_AICORE constexpr bool has_aiv0() const { return (bits_ & kAiv0) != 0; }
    SIMPLER_LANE_AICORE constexpr bool has_aiv1() const { return (bits_ & kAiv1) != 0; }
    SIMPLER_LANE_AICORE constexpr bool has_any_aiv() const { return (bits_ & (kAiv0 | kAiv1)) != 0; }
    SIMPLER_LANE_AICORE constexpr bool has_both_aiv() const { return (bits_ & (kAiv0 | kAiv1)) == (kAiv0 | kAiv1); }
    SIMPLER_LANE_AICORE constexpr bool has(SubtaskSlot slot) const { return contains(of(slot)); }
    SIMPLER_LANE_AICORE constexpr bool is_empty() const { return bits_ == 0; }
    SIMPLER_LANE_AICORE constexpr bool is_full() const { return bits_ == kAll; }
    SIMPLER_LANE_AICORE constexpr int count() const { return __builtin_popcount(bits_); }
    SIMPLER_LANE_AICORE constexpr bool is_mix() const { return count() >= 2; }

    SIMPLER_LANE_AICORE constexpr ResourceShape to_shape() const {
        return is_empty() ? ResourceShape::DUMMY :
                            (is_mix() ? ResourceShape::MIX : (has_aic() ? ResourceShape::AIC : ResourceShape::AIV));
    }

    // A single logical AIV can be placed on either physical V lane.
    SIMPLER_LANE_AICORE constexpr LaneMask lane_demand() const {
        return to_shape() == ResourceShape::AIV ? both_aiv() : *this;
    }

    SIMPLER_LANE_AICORE constexpr LaneMask operator|(LaneMask other) const { return LaneMask(bits_ | other.bits_); }
    SIMPLER_LANE_AICORE constexpr LaneMask operator&(LaneMask other) const { return LaneMask(bits_ & other.bits_); }
    SIMPLER_LANE_AICORE constexpr LaneMask &operator|=(LaneMask other) {
        bits_ |= other.bits_;
        return *this;
    }
    SIMPLER_LANE_AICORE constexpr LaneMask &operator&=(LaneMask other) {
        bits_ &= other.bits_;
        return *this;
    }
    SIMPLER_LANE_AICORE constexpr bool contains(LaneMask other) const { return (bits_ & other.bits_) == other.bits_; }
    SIMPLER_LANE_AICORE constexpr bool intersects(LaneMask other) const { return (bits_ & other.bits_) != 0; }
    SIMPLER_LANE_AICORE constexpr LaneMask without(LaneMask other) const { return LaneMask(bits_ & ~other.bits_); }
    SIMPLER_LANE_AICORE constexpr LaneMask complement() const { return LaneMask(~bits_ & kAll); }
    SIMPLER_LANE_AICORE constexpr bool operator==(LaneMask other) const { return bits_ == other.bits_; }
    SIMPLER_LANE_AICORE constexpr bool operator!=(LaneMask other) const { return bits_ != other.bits_; }

    // Serialization, diagnostics and packed indices use the validated representation directly.
    SIMPLER_LANE_AICORE constexpr uint8_t bits() const { return bits_; }

private:
    static constexpr uint8_t kAll = kAic | kAiv0 | kAiv1;
    SIMPLER_LANE_AICORE constexpr explicit LaneMask(uint8_t bits) :
        bits_(bits) {}

    uint8_t bits_{0};
};

static_assert(sizeof(LaneMask) == 1 && alignof(LaneMask) == 1);
static_assert(std::is_trivially_copyable_v<LaneMask> && std::is_standard_layout_v<LaneMask>);
#if !defined(__CCE_AICORE__)
static_assert(std::atomic<LaneMask>::is_always_lock_free);
#endif

#undef SIMPLER_LANE_AICORE
