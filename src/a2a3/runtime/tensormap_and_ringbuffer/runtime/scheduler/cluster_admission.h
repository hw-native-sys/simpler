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
#include "common/lane_mask.h"

// Ten complete hardware clusters per word. Free lanes and the current waiter's
// priority bits share one linearization point with ordinary placement.
template <int Owners, int Clusters>
class ReadyLanePool {
public:
    static constexpr int per_word = 10;
    static constexpr int width = 3 * per_word;
    static constexpr int words = (Clusters + per_word - 1) / per_word;
    static constexpr uint32_t valid_bits = (1u << width) - 1;
    static constexpr uint32_t cubes = valid_bits / 7;
    static uint32_t topology(int clusters, int word) {
        int n = clusters - word * per_word;
        if (n >= per_word) return valid_bits;
        return n <= 0 ? 0 : (1u << (3 * n)) - 1;
    }
    static uint32_t positions(uint32_t lanes, ResourceShape shape, LaneMask mask) {
        if (shape == ResourceShape::AIC) return lanes & cubes;
        if (shape == ResourceShape::AIV) return lanes & (valid_bits ^ cubes);
        uint32_t choices = cubes;
        if (mask.has_aic()) choices &= lanes;
        if (mask.has_aiv0()) choices &= lanes >> 1;
        if (mask.has_aiv1()) choices &= lanes >> 2;
        return choices;
    }
    static uint32_t expand(uint32_t starts, ResourceShape shape, LaneMask mask) {
        if (shape != ResourceShape::MIX) return starts;
        return (mask.has_aic() ? starts : 0) | (mask.has_aiv0() ? starts << 1 : 0) |
               (mask.has_aiv1() ? starts << 2 : 0);
    }
    static uint32_t first(uint32_t choices, int n) {
        if (__builtin_popcount(choices) <= n) return choices;
        uint32_t rest = choices;
        for (int i = 0; i < n; ++i)
            rest &= rest - 1;
        return choices ^ rest;
    }
    uint64_t snapshot(int owner, int w) const { return owners_[owner].word[w].load(std::memory_order_acquire); }
    void release(int owner, int w, uint32_t mask) {
        if (mask) owners_[owner].word[w].fetch_or(mask, std::memory_order_release);
    }
    void initialize(int owner, int clusters) {
        for (int w = 0; w < words; ++w)
            release(owner, w, topology(clusters, w));
    }
    void reset() {
        for (auto &owner : owners_)
            for (auto &v : owner.word)
                v.store(0, std::memory_order_relaxed);
    }
    void protect(int owner, int w, uint32_t mask) {
        if (mask) owners_[owner].word[w].fetch_or(uint64_t(mask) << 32, std::memory_order_acq_rel);
    }
    void unprotect(int owner, int w, uint32_t mask) {
        if (mask) owners_[owner].word[w].fetch_and(~(uint64_t(mask) << 32), std::memory_order_release);
    }
    uint32_t take_for_ss(int owner, int w, uint32_t mask) {
        return static_cast<uint32_t>(owners_[owner].word[w].fetch_and(~uint64_t(mask), std::memory_order_acq_rel)) &
               mask;
    }
    uint32_t acquire_local(
        int owner, int w, uint32_t choices, ResourceShape shape, LaneMask mask, int limit, uint32_t &held, bool &dirty
    ) {
        auto &word = owners_[owner].word[w];
        uint64_t old = word.load(std::memory_order_acquire);
        for (;;) {
            uint32_t available = (static_cast<uint32_t>(old) | held) & ~static_cast<uint32_t>(old >> 32);
            uint32_t starts = first(choices & positions(available, shape, mask), limit);
            uint32_t fresh = expand(starts, shape, mask) & ~held;
            if (!fresh) return starts;
            if (word.compare_exchange_weak(
                    old, old & ~uint64_t(fresh), std::memory_order_acq_rel, std::memory_order_acquire
                )) {
                held |= fresh;
                dirty = true;
                return starts;
            }
        }
    }

private:
    struct alignas(64) Owner {
        std::atomic<uint64_t> word[words]{};
    };
    Owner owners_[Owners]{};
};
