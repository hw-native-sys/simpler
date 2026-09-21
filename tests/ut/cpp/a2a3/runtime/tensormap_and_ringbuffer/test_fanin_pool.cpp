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
 * Unit tests for FaninPool and for_each_fanin_storage/slot_state
 * from ring_buffer.h / ring_buffer.cpp
 *
 * Tests:
 * 1. FaninPool — ring buffer allocation, overflow, tail advance,
 *    high-water tracking
 * 2. for_each_fanin_storage — inline-only, spill without wrap,
 *    spill with wrap, callback early return
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "ring_buffer.h"
#include "shared_memory.h"

// =============================================================================
// FaninPool fixture
// =============================================================================

class FaninPoolTest : public ::testing::Test {
protected:
    static constexpr int32_t POOL_CAP = 32;

    std::vector<FaninSpillEntry> entries;
    std::atomic<int32_t> error_code{SIMPLER_ERROR_NONE};
    FaninPool pool{};

    void SetUp() override {
        entries.assign(POOL_CAP, FaninSpillEntry{});
        error_code.store(SIMPLER_ERROR_NONE);
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
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
}

TEST_F(FaninPoolTest, EnsureSpaceDeadlockReturnsFalseAndLatchesError) {
    for (int i = 0; i < POOL_CAP; i++) {
        ASSERT_NE(pool.alloc(), nullptr);
    }

    SharedMemoryRingHeader ring{};
    ring.fc.init();
    ring.fc.current_task_index.store(POOL_CAP + 1, std::memory_order_release);
    ring.fc.last_task_alive.store(0, std::memory_order_release);

    EXPECT_FALSE(pool.ensure_space(ring, 1));
    EXPECT_EQ(error_code.load(), SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
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
// for_each_fanin_storage: inline only
// =============================================================================

class ForEachFaninTest : public ::testing::Test {
protected:
    static constexpr int32_t POOL_CAP = 32;

    std::vector<FaninSpillEntry> spill_entries;
    std::atomic<int32_t> error_code{SIMPLER_ERROR_NONE};
    FaninPool spill_pool{};

    alignas(64) ChipTaskSlotState slots[64];

    void SetUp() override {
        spill_entries.assign(POOL_CAP, FaninSpillEntry{});
        error_code.store(SIMPLER_ERROR_NONE);
        spill_pool.init(spill_entries.data(), POOL_CAP, &error_code);
        memset(slots, 0, sizeof(slots));
    }
};

TEST_F(ForEachFaninTest, InlineOnlyVoid) {
    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < 5; i++) {
        inline_slots[i].set(&slots[i], DEP_WAIT);
    }

    std::vector<ChipTaskSlotState *> visited;
    for_each_fanin_storage(inline_slots, 5, 0, spill_pool, [&](ChipTaskSlotState *s, DepFlags) {
        visited.push_back(s);
    });

    ASSERT_EQ(visited.size(), 5u);
    for (int i = 0; i < 5; i++) {
        EXPECT_EQ(visited[i], &slots[i]);
    }
}

TEST_F(ForEachFaninTest, InlineOnlyBoolEarlyReturn) {
    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < 5; i++) {
        inline_slots[i].set(&slots[i], DEP_WAIT);
    }

    int count = 0;
    bool result = for_each_fanin_storage(inline_slots, 5, 0, spill_pool, [&](ChipTaskSlotState *, DepFlags) -> bool {
        count++;
        return count < 3;  // stop after 3rd
    });

    EXPECT_FALSE(result) << "Should return false when callback returns false";
    EXPECT_EQ(count, 3);
}

TEST_F(ForEachFaninTest, InlineOnlyBoolAllTrue) {
    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < 3; i++) {
        inline_slots[i].set(&slots[i], DEP_WAIT);
    }

    bool result = for_each_fanin_storage(inline_slots, 3, 0, spill_pool, [](ChipTaskSlotState *, DepFlags) -> bool {
        return true;
    });

    EXPECT_TRUE(result);
}

TEST_F(ForEachFaninTest, ZeroFanin) {
    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP] = {};
    int count = 0;
    for_each_fanin_storage(inline_slots, 0, 0, spill_pool, [&](ChipTaskSlotState *, DepFlags) {
        count++;
    });
    EXPECT_EQ(count, 0);
}

// =============================================================================
// for_each_fanin_storage: spill without wrap
// =============================================================================

TEST_F(ForEachFaninTest, SpillNoWrap) {
    // 18 fanins = 16 inline + 2 spill
    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < CHIP_FANIN_INLINE_CAP; i++) {
        inline_slots[i].set(&slots[i], DEP_WAIT);
    }

    // Allocate 2 spill entries
    auto *s0 = spill_pool.alloc();
    int32_t spill_start = spill_pool.top - 1;
    s0->set(&slots[16], DEP_WAIT | DEP_RETAIN);
    auto *s1 = spill_pool.alloc();
    s1->set(&slots[17], DEP_WAIT | DEP_RETAIN);

    std::vector<ChipTaskSlotState *> visited;
    for_each_fanin_storage(inline_slots, 18, spill_start, spill_pool, [&](ChipTaskSlotState *s, DepFlags) {
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
// for_each_fanin_storage: spill with wrap
// =============================================================================

TEST_F(ForEachFaninTest, SpillWithWrap) {
    // Push pool near end so spill wraps around
    // Pool cap = 32, advance top to 30 so next alloc is at index 30
    spill_pool.top = POOL_CAP - 2;
    spill_pool.tail = POOL_CAP - 2;

    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < CHIP_FANIN_INLINE_CAP; i++) {
        inline_slots[i].set(&slots[i], DEP_WAIT);
    }

    // 4 spill entries: indices 30, 31, 0, 1 (wraps around)
    int32_t spill_start = spill_pool.top;
    for (int i = 0; i < 4; i++) {
        auto *e = spill_pool.alloc();
        ASSERT_NE(e, nullptr);
        e->set(&slots[16 + i], DEP_WAIT | DEP_RETAIN);
    }

    std::vector<ChipTaskSlotState *> visited;
    for_each_fanin_storage(inline_slots, 20, spill_start, spill_pool, [&](ChipTaskSlotState *s, DepFlags) {
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
// for_each_fanin_storage: spill with bool callback early return
// =============================================================================

TEST_F(ForEachFaninTest, SpillBoolEarlyReturnInSpillRegion) {
    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP] = {};
    for (int i = 0; i < CHIP_FANIN_INLINE_CAP; i++) {
        inline_slots[i].set(&slots[i], DEP_WAIT);
    }

    int32_t spill_start = spill_pool.top;
    for (int i = 0; i < 4; i++) {
        auto *e = spill_pool.alloc();
        e->set(&slots[16 + i], DEP_WAIT);
    }

    int count = 0;
    bool result =
        for_each_fanin_storage(inline_slots, 20, spill_start, spill_pool, [&](ChipTaskSlotState *, DepFlags) -> bool {
            count++;
            return count < 17;  // stop on 17th (first spill entry)
        });

    EXPECT_FALSE(result);
    EXPECT_EQ(count, 17);
}

// =============================================================================
// FaninSpillEntry: DepFlags packing round-trips across inline and spill
// =============================================================================

TEST_F(ForEachFaninTest, DepFlagsRoundTripInlineAndSpill) {
    // Fill all 64 inline edges; slots 0..2 carry distinct flag combinations.
    FaninSpillEntry inline_slots[CHIP_FANIN_INLINE_CAP] = {};
    inline_slots[0].set(&slots[0], DEP_WAIT);
    inline_slots[1].set(&slots[1], DEP_RETAIN);
    inline_slots[2].set(&slots[2], DEP_WAIT | DEP_RETAIN);
    for (int i = 3; i < CHIP_FANIN_INLINE_CAP; i++) {
        inline_slots[i].set(&slots[i], DEP_WAIT);
    }

    // Two edges beyond the inline cap spill; they carry RETAIN-only and WAIT|RETAIN.
    auto *s0 = spill_pool.alloc();
    int32_t spill_start = spill_pool.top - 1;
    s0->set(&slots[0], DEP_RETAIN);
    auto *s1 = spill_pool.alloc();
    s1->set(&slots[1], DEP_WAIT | DEP_RETAIN);

    const int32_t total = CHIP_FANIN_INLINE_CAP + 2;  // 64 inline + 2 spill
    std::vector<DepFlags> flags;
    for_each_fanin_storage(inline_slots, total, spill_start, spill_pool, [&](ChipTaskSlotState *, DepFlags f) {
        flags.push_back(f);
    });

    ASSERT_EQ(flags.size(), static_cast<size_t>(total));
    // Inline flags survive.
    EXPECT_EQ(flags[0], DEP_WAIT);
    EXPECT_EQ(flags[1], DEP_RETAIN);
    EXPECT_EQ(flags[2], DEP_WAIT | DEP_RETAIN);
    // Spill flags survive.
    EXPECT_EQ(flags[CHIP_FANIN_INLINE_CAP + 0], DEP_RETAIN);
    EXPECT_EQ(flags[CHIP_FANIN_INLINE_CAP + 1], DEP_WAIT | DEP_RETAIN);
}
