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
#endif

#if defined(__CCE_AICORE__)
#define SIMPLER_LANE_AICORE [aicore]
#else
#define SIMPLER_LANE_AICORE
#endif

inline constexpr int32_t SUBTASK_SLOT_COUNT = 3;

enum class SubtaskSlot : uint8_t { AIC = 0, AIV0 = 1, AIV1 = 2 };

// Two or more active lanes form MIX. One cluster supplies the named lanes:
// each idle used core takes a running slot, and a busy used core takes its
// free pending slot. An occupied pending slot on a used core rejects placement;
// unused cores do not constrain it. DUMMY has no core work, uses a separate
// ready queue and completes inline in the scheduler.
enum class ResourceShape : uint8_t { AIC = 0, AIV = 1, MIX = 2, DUMMY = 3 };
inline constexpr int32_t NUM_RESOURCE_SHAPES = 3;

// Constructors and activate() produce only valid AIC/AIV0/AIV1 combinations.
// A memcpy-loaded wire value requires is_valid() before any scheduling use.
class LaneMask {
public:
    SIMPLER_LANE_AICORE constexpr LaneMask() = default;
    SIMPLER_LANE_AICORE constexpr LaneMask(bool aic, bool aiv0, bool aiv1) :
        bits_((aic ? kAic : 0) | (aiv0 ? kAiv0 : 0) | (aiv1 ? kAiv1 : 0)) {}

    SIMPLER_LANE_AICORE static constexpr LaneMask none() { return {}; }
    SIMPLER_LANE_AICORE static constexpr LaneMask aic() { return {true, false, false}; }
    SIMPLER_LANE_AICORE static constexpr LaneMask aiv0() { return {false, true, false}; }
    SIMPLER_LANE_AICORE static constexpr LaneMask aiv1() { return {false, false, true}; }
    SIMPLER_LANE_AICORE static constexpr LaneMask both_aiv() { return {false, true, true}; }
    SIMPLER_LANE_AICORE static constexpr LaneMask all() { return {true, true, true}; }
    SIMPLER_LANE_AICORE static constexpr LaneMask of(SubtaskSlot slot) {
        switch (slot) {
        case SubtaskSlot::AIC:
            return aic();
        case SubtaskSlot::AIV0:
            return aiv0();
        case SubtaskSlot::AIV1:
            return aiv1();
        }
        return none();
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

    SIMPLER_LANE_AICORE constexpr bool has_aic() const { return (bits_ & kAic) != 0; }
    SIMPLER_LANE_AICORE constexpr bool has_aiv0() const { return (bits_ & kAiv0) != 0; }
    SIMPLER_LANE_AICORE constexpr bool has_aiv1() const { return (bits_ & kAiv1) != 0; }
    SIMPLER_LANE_AICORE constexpr bool has_any_aiv() const { return (bits_ & (kAiv0 | kAiv1)) != 0; }
    SIMPLER_LANE_AICORE constexpr bool has_both_aiv() const { return (bits_ & (kAiv0 | kAiv1)) == (kAiv0 | kAiv1); }
    SIMPLER_LANE_AICORE constexpr bool has(SubtaskSlot slot) const {
        switch (slot) {
        case SubtaskSlot::AIC:
            return has_aic();
        case SubtaskSlot::AIV0:
            return has_aiv0();
        case SubtaskSlot::AIV1:
            return has_aiv1();
        }
        return false;
    }
    SIMPLER_LANE_AICORE constexpr bool is_empty() const { return bits_ == 0; }
    SIMPLER_LANE_AICORE constexpr bool is_aic() const { return bits_ == kAic; }
    SIMPLER_LANE_AICORE constexpr bool is_full() const { return bits_ == kAll; }
    SIMPLER_LANE_AICORE constexpr bool is_mix() const { return count() >= 2; }
    SIMPLER_LANE_AICORE constexpr bool is_valid() const { return (bits_ & ~kAll) == 0; }
    SIMPLER_LANE_AICORE constexpr int count() const { return __builtin_popcount(bits_); }
    SIMPLER_LANE_AICORE constexpr ResourceShape to_shape() const {
        return is_empty() ? ResourceShape::DUMMY :
                            (is_mix() ? ResourceShape::MIX : (has_aic() ? ResourceShape::AIC : ResourceShape::AIV));
    }
    // A single logical AIV can be placed on either physical V lane.
    SIMPLER_LANE_AICORE constexpr LaneMask lane_demand() const {
        return to_shape() == ResourceShape::AIV ? both_aiv() : *this;
    }
    SIMPLER_LANE_AICORE constexpr void activate(SubtaskSlot slot) { bits_ |= of(slot).bits_; }

    // Precondition: has(slot). Callers reject inactive slots before requesting
    // their compact index among active lanes in slot order.
    SIMPLER_LANE_AICORE constexpr int rank_of(SubtaskSlot slot) const {
        return (slot != SubtaskSlot::AIC && has_aic()) + (slot == SubtaskSlot::AIV1 && has_aiv0());
    }
    SIMPLER_LANE_AICORE constexpr LaneMask merged_with(LaneMask other) const {
        LaneMask merged = *this;
        merged.bits_ |= other.bits_;
        return merged;
    }
    SIMPLER_LANE_AICORE constexpr LaneMask without(LaneMask other) const { return from_low_bits(bits_ & ~other.bits_); }
    SIMPLER_LANE_AICORE constexpr bool contains(LaneMask other) const { return (bits_ & other.bits_) == other.bits_; }

    SIMPLER_LANE_AICORE constexpr bool operator==(LaneMask other) const { return bits_ == other.bits_; }
    SIMPLER_LANE_AICORE constexpr bool operator!=(LaneMask other) const { return bits_ != other.bits_; }

    // The byte wire structs already store and a reader memcpy's back, so this
    // is what a serialized lane set is. Scheduling decisions take the queries
    // above instead; this exists for carrying the value across a boundary that
    // cannot hold the type.
    SIMPLER_LANE_AICORE constexpr uint8_t raw() const { return bits_; }

private:
    static constexpr int kBitWidth = SUBTASK_SLOT_COUNT;
    SIMPLER_LANE_AICORE constexpr uint8_t bits() const { return bits_; }
    SIMPLER_LANE_AICORE static constexpr LaneMask from_low_bits(uintptr_t word) {
        LaneMask lanes;
        lanes.bits_ = static_cast<uint8_t>(word & kAll);
        return lanes;
    }
    template <typename, typename, int>
    friend class TaggedPtr;
    friend class LaneDemandCounter;

    static constexpr uint8_t kAic = 1u << 0;
    static constexpr uint8_t kAiv0 = 1u << 1;
    static constexpr uint8_t kAiv1 = 1u << 2;
    static constexpr uint8_t kAll = kAic | kAiv0 | kAiv1;
    uint8_t bits_{0};
};

static_assert(sizeof(LaneMask) == 1 && alignof(LaneMask) == 1);
static_assert(std::is_trivially_copyable_v<LaneMask> && std::is_standard_layout_v<LaneMask>);
#if !defined(__CCE_AICORE__)
static_assert(std::atomic<LaneMask>::is_always_lock_free);
#endif

#undef SIMPLER_LANE_AICORE
