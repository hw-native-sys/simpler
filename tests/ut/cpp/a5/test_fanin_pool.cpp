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
/**
 * Unit tests for PTO2FaninPool and pto2_for_each_fanin_storage/slot_state
 * from pto_ring_buffer.h / pto_ring_buffer.cpp
 *
 * Tests:
 * 1. PTO2FaninPool — ring buffer allocation, overflow, tail advance,
 *    high-water tracking
 * 2. pto2_for_each_fanin_storage — inline-only, spill without wrap,
 *    spill with wrap, callback early return
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "pto_ring_buffer.h"
#include "pto_shared_memory.h"

// =============================================================================
// FaninPool fixture
// =============================================================================

class FaninPoolTest : public ::testing::Test {
protected:
    static constexpr int32_t POOL_CAP = 32;

    std::vector<PTO2FaninSpillEntry> entries;
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2FaninPool pool{};

    void SetUp() override {
        entries.assign(POOL_CAP, PTO2FaninSpillEntry{nullptr});
        error_code.store(PTO2_ERROR_NONE);
        pool.init(entries.data(), POOL_CAP, &error_code);
    }
};

// =============================================================================
// FaninPool: basic operations
// =============================================================================

TEST_F(FaninPoolTest, InitialState) {
    EXPECT_EQ(pool.used(), 0);
    EXPECT_EQ(pool.available(), POOL_CAP);
    EXPECT_EQ(pool.top, 1);
    EXPECT_EQ(pool.tail, 1);
    EXPECT_EQ(pool.high_water, 0);
}

TEST_F(FaninPoolTest, AllocReturnsCorrectModuloIndex) {
    // First alloc at index top%cap = 1%32 = 1
    auto *e1 = pool.alloc();
    EXPECT_EQ(e1, &entries[1]);

    auto *e2 = pool.alloc();
    EXPECT_EQ(e2, &entries[2]);
}

TEST_F(FaninPoolTest, AllocFillsPool) {
    for (int i = 0; i < POOL_CAP; i++) {
        auto *e = pool.alloc();
        ASSERT_NE(e, nullptr) << "Alloc failed at i=" << i;
    }
    EXPECT_EQ(pool.used(), POOL_CAP);
    EXPECT_EQ(pool.available(), 0);
}

TEST_F(FaninPoolTest, OverflowReturnsNullptr) {
    for (int i = 0; i < POOL_CAP; i++) {
        pool.alloc();
    }
    auto *overflow = pool.alloc();
    EXPECT_EQ(overflow, nullptr);
    EXPECT_EQ(error_code.load(), PTO2_ERROR_DEP_POOL_OVERFLOW);
}

TEST_F(FaninPoolTest, AdvanceTailFreesSpace) {
    for (int i = 0; i < 10; i++) {
        pool.alloc();
    }
    EXPECT_EQ(pool.used(), 10);

    pool.advance_tail(pool.tail + 5);
    EXPECT_EQ(pool.used(), 5);
    EXPECT_EQ(pool.available(), POOL_CAP - 5);
}

TEST_F(FaninPoolTest, AdvanceTailBackwardsIsNoop) {
    for (int i = 0; i < 10; i++) {
        pool.alloc();
    }
    int32_t old_tail = pool.tail;
    pool.advance_tail(old_tail - 1);
    EXPECT_EQ(pool.tail, old_tail);
    EXPECT_EQ(pool.used(), 10);
}

TEST_F(FaninPoolTest, HighWaterNeverDecreases) {
    for (int i = 0; i < 10; i++) {
        pool.alloc();
    }
    EXPECT_EQ(pool.high_water, 10);

    pool.advance_tail(pool.tail + 5);
    EXPECT_EQ(pool.high_water, 10) << "high_water must never decrease";
}

TEST_F(FaninPoolTest, WrapAroundAllocation) {
    // Fill and drain, then fill again to wrap
    for (int i = 0; i < POOL_CAP; i++) {
        pool.alloc();
    }
    pool.advance_tail(pool.top);
    EXPECT_EQ(pool.used(), 0);

    // New allocations wrap around
    for (int i = 0; i < 5; i++) {
        auto *e = pool.alloc();
        ASSERT_NE(e, nullptr);
        // Verify modulo indexing
        int32_t expected_idx = (pool.top - 1) % POOL_CAP;
        EXPECT_EQ(e, &entries[expected_idx]);
    }
    EXPECT_EQ(pool.used(), 5);
}

// =============================================================================
// pto2_for_each_fanin_storage: inline only
// =============================================================================

class ForEachFaninTest : public ::testing::Test {
protected:
    static constexpr int32_t POOL_CAP = 32;

    std::vector<PTO2FaninSpillEntry> spill_entries;
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2FaninPool spill_pool{};

    alignas(64) PTO2TaskSlotState slots[64];

    void SetUp() override {
        spill_entries.assign(POOL_CAP, PTO2FaninSpillEntry{nullptr});
        error_code.store(PTO2_ERROR_NONE);
        spill_pool.init(spill_entries.data(), POOL_CAP, &error_code);
        memset(slots, 0, sizeof(slots));
    }
};

TEST_F(ForEachFaninTest, InlineOnlyVoid) {
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < 5; i++) {
        inline_slots[i] = &slots[i];
    }

    std::vector<PTO2TaskSlotState *> visited;
    pto2_for_each_fanin_storage(inline_slots, 5, 0, spill_pool, [&](PTO2TaskSlotState *s) {
        visited.push_back(s);
    });

    ASSERT_EQ(visited.size(), 5u);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(visited[i], &slots[i]);
    }
}

TEST_F(ForEachFaninTest, InlineOnlyBoolEarlyReturn) {
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < 5; i++) {
        inline_slots[i] = &slots[i];
    }

    int count = 0;
    bool result = pto2_for_each_fanin_storage(inline_slots, 5, 0, spill_pool, [&](PTO2TaskSlotState *) -> bool {
        count++;
        return count < 3;  // stop after 3rd
    });

    EXPECT_FALSE(result) << "Should return false when callback returns false";
    EXPECT_EQ(count, 3);
}

TEST_F(ForEachFaninTest, InlineOnlyBoolAllTrue) {
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < 3; i++) {
        inline_slots[i] = &slots[i];
    }

    bool result = pto2_for_each_fanin_storage(inline_slots, 3, 0, spill_pool, [](PTO2TaskSlotState *) -> bool {
        return true;
    });

    EXPECT_TRUE(result);
}

TEST_F(ForEachFaninTest, ZeroFanin) {
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP] = {};
    int count = 0;
    pto2_for_each_fanin_storage(inline_slots, 0, 0, spill_pool, [&](PTO2TaskSlotState *) {
        count++;
    });
    EXPECT_EQ(count, 0);
}

// =============================================================================
// pto2_for_each_fanin_storage: spill without wrap
// =============================================================================

TEST_F(ForEachFaninTest, SpillNoWrap) {
    // 18 fanins = 16 inline + 2 spill
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < PTO2_FANIN_INLINE_CAP; i++) {
        inline_slots[i] = &slots[i];
    }

    // Allocate 2 spill entries
    auto *s0 = spill_pool.alloc();
    int32_t spill_start = spill_pool.top - 1;
    s0->slot_state = &slots[16];
    auto *s1 = spill_pool.alloc();
    s1->slot_state = &slots[17];

    std::vector<PTO2TaskSlotState *> visited;
    pto2_for_each_fanin_storage(inline_slots, 18, spill_start, spill_pool, [&](PTO2TaskSlotState *s) {
        visited.push_back(s);
    });

    ASSERT_EQ(visited.size(), 18u);
    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(visited[i], &slots[i]) << "Inline slot " << i;
    }
    EXPECT_EQ(visited[16], &slots[16]);
    EXPECT_EQ(visited[17], &slots[17]);
}

// =============================================================================
// pto2_for_each_fanin_storage: spill with wrap
// =============================================================================

TEST_F(ForEachFaninTest, SpillWithWrap) {
    // Push pool near end so spill wraps around
    // Pool cap = 32, advance top to 30 so next alloc is at index 30
    spill_pool.top = POOL_CAP - 2;
    spill_pool.tail = POOL_CAP - 2;

    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < PTO2_FANIN_INLINE_CAP; i++) {
        inline_slots[i] = &slots[i];
    }

    // 4 spill entries: indices 30, 31, 0, 1 (wraps around)
    int32_t spill_start = spill_pool.top;
    for (int i = 0; i < 4; i++) {
        auto *e = spill_pool.alloc();
        ASSERT_NE(e, nullptr);
        e->slot_state = &slots[16 + i];
    }

    std::vector<PTO2TaskSlotState *> visited;
    pto2_for_each_fanin_storage(inline_slots, 20, spill_start, spill_pool, [&](PTO2TaskSlotState *s) {
        visited.push_back(s);
    });

    ASSERT_EQ(visited.size(), 20u);
    // Inline
    for (int i = 0; i < 16; i++) {
        EXPECT_EQ(visited[i], &slots[i]);
    }
    // Spill (wrapped)
    for (int i = 0; i < 4; i++) {
        EXPECT_EQ(visited[16 + i], &slots[16 + i]);
    }
}

// =============================================================================
// pto2_for_each_fanin_storage: spill with bool callback early return
// =============================================================================

TEST_F(ForEachFaninTest, SpillBoolEarlyReturnInSpillRegion) {
    PTO2TaskSlotState *inline_slots[PTO2_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < PTO2_FANIN_INLINE_CAP; i++) {
        inline_slots[i] = &slots[i];
    }

    int32_t spill_start = spill_pool.top;
    for (int i = 0; i < 4; i++) {
        auto *e = spill_pool.alloc();
        e->slot_state = &slots[16 + i];
    }

    int count = 0;
    bool result =
        pto2_for_each_fanin_storage(inline_slots, 20, spill_start, spill_pool, [&](PTO2TaskSlotState *) -> bool {
            count++;
            return count < 17;  // stop on 17th (first spill entry)
        });

    EXPECT_FALSE(result);
    EXPECT_EQ(count, 17);
}
