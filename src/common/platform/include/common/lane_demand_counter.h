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

#include <atomic>
#include <cstdint>
#include <type_traits>

#include "assert_compat.h"
#include "common/lane_mask.h"

// Three independent 21-bit fields; each published cell owns one paired add/sub.
class LaneDemandCounter {
public:
    static constexpr int kFieldBits = 21;
    static constexpr uint32_t kMaxCount = (uint32_t{1} << kFieldBits) - 1;

private:
    static constexpr uint64_t kAicOne = 1;
    static constexpr uint64_t kAiv0One = uint64_t{1} << kFieldBits;
    static constexpr uint64_t kAiv1One = uint64_t{1} << (2 * kFieldBits);
    static constexpr uint64_t kDelta[8] = {
        0,
        kAicOne,
        kAiv0One,
        kAicOne | kAiv0One,
        kAiv1One,
        kAicOne | kAiv1One,
        kAiv0One | kAiv1One,
        kAicOne | kAiv0One | kAiv1One,
    };
    static_assert(LaneMask::kBitWidth == 3);
    static_assert(kDelta[LaneMask::kAic | LaneMask::kAiv0] == ((uint64_t{1} << 21) | 1));
    static_assert(kDelta[LaneMask::kAiv0 | LaneMask::kAiv1] == ((uint64_t{1} << 42) | (uint64_t{1} << 21)));

    static constexpr uint32_t field(uint64_t value, int lane) {
        return static_cast<uint32_t>((value >> (lane * kFieldBits)) & kMaxCount);
    }

public:
    // A local aggregate can be formed from either live slots or immutable cell tags.
    class Batch {
    public:
        void add(LaneMask demand) {
            const uint64_t delta = kDelta[demand.bits()];
            delta_ += delta;
        }

    private:
        friend class LaneDemandCounter;
        uint64_t delta_{0};
    };

    // Reset requires quiescent producers and consumers.
    void reset() { value_.store(0, std::memory_order_relaxed); }

    void add(LaneMask demand) { add_delta(kDelta[demand.bits()]); }
    void sub(LaneMask demand) { sub_delta(kDelta[demand.bits()]); }
    void add(const Batch &batch) { add_delta(batch.delta_); }
    void sub(const Batch &batch) { sub_delta(batch.delta_); }

    LaneMask reserved() const {
        const uint64_t value = value_.load(std::memory_order_relaxed);
        if (value == 0) return LaneMask::none();
        return {field(value, 0) != 0, field(value, 1) != 0, field(value, 2) != 0};
    }

    uint32_t count(LaneMask lane) const {
        debug_assert(lane.count() == 1 && "count requires exactly one lane");
        const int index = lane.has_aic() ? 0 : (lane.has_aiv0() ? 1 : 2);
        return field(value_.load(std::memory_order_relaxed), index);
    }

private:
    void add_delta(uint64_t delta) { value_.fetch_add(delta, std::memory_order_relaxed); }
    void sub_delta(uint64_t delta) { value_.fetch_sub(delta, std::memory_order_relaxed); }

    std::atomic<uint64_t> value_{0};
};
static_assert(std::atomic<uint64_t>::is_always_lock_free);
static_assert(sizeof(LaneDemandCounter) == sizeof(uint64_t) && alignof(LaneDemandCounter) == 8);
static_assert(std::is_standard_layout_v<LaneDemandCounter> && std::is_trivially_copyable_v<LaneDemandCounter>);
static_assert(3 * LaneDemandCounter::kFieldBits <= 64);
