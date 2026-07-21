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
 * Unit tests for PTO2DepListPool from pto_ring_buffer.h
 *
 * Tests dependency list pool allocation, prepend chaining, overflow detection,
 * tail advancement, and high-water mark tracking.
 *
 * Design contracts:
 *
 * - advance_tail(new_tail) only advances if new_tail > tail; it does
 *   not validate new_tail <= top.  Caller contract (monotonic,
 *   top-bounded).
 *
 * - The list terminator is literal nullptr.  base[0] is a normal pool entry;
 *   init clearing it is incidental, not an invariant.
 */

#include <gtest/gtest.h>

#include <atomic>
#include <cstring>
#include <vector>

#include "pto_ring_buffer.h"

// =============================================================================
// Fixture
// =============================================================================

class DepListPoolTest : public ::testing::Test {
protected:
    static constexpr int32_t POOL_CAP = 8;
    PTO2DepListEntry entries[POOL_CAP]{};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2DepListPool pool{};

    void SetUp() override {
        std::memset(entries, 0, sizeof(entries));
        error_code.store(PTO2_ERROR_NONE);
        pool.init(entries, POOL_CAP, &error_code);
    }
};

// =============================================================================
// Normal path
// =============================================================================

TEST_F(DepListPoolTest, InitialState) {
    EXPECT_EQ(pool.used(), 0);
    EXPECT_EQ(pool.available(), POOL_CAP);
}

TEST_F(DepListPoolTest, SingleAlloc) {
    PTO2DepListEntry *entry = pool.alloc();
    ASSERT_NE(entry, nullptr);
    EXPECT_EQ(pool.used(), 1);
    EXPECT_EQ(pool.available(), POOL_CAP - 1);
}

TEST_F(DepListPoolTest, OverflowDetection) {
    for (int i = 0; i < POOL_CAP; i++) {
        PTO2DepListEntry *e = pool.alloc();
        ASSERT_NE(e, nullptr) << "Unexpected failure at alloc " << i;
    }
    EXPECT_EQ(pool.used(), POOL_CAP);
    EXPECT_EQ(pool.available(), 0);

    PTO2DepListEntry *overflow = pool.alloc();
    EXPECT_EQ(overflow, nullptr);
    EXPECT_EQ(error_code.load(), PTO2_ERROR_DEP_POOL_OVERFLOW);
}

TEST_F(DepListPoolTest, EnsureSpaceDeadlockReturnsFalseAndLatchesError) {
    for (int i = 0; i < POOL_CAP; i++) {
        ASSERT_NE(pool.alloc(), nullptr);
    }

    PTO2SharedMemoryRingHeader ring{};
    ring.fc.init();
    ring.fc.current_task_index.store(POOL_CAP + 1, std::memory_order_release);
    ring.fc.last_task_alive.store(0, std::memory_order_release);

    EXPECT_FALSE(pool.ensure_space(ring, 1));
    EXPECT_EQ(error_code.load(), PTO2_ERROR_DEP_POOL_OVERFLOW);
}

// Prepend builds LIFO linked list: verify each slot_state pointer.
TEST_F(DepListPoolTest, PrependChainCorrectness) {
    PTO2TaskSlotState slots[5]{};
    PTO2DepListEntry *head = nullptr;

    for (int i = 0; i < 5; i++) {
        head = pool.prepend(head, &slots[i]);
        ASSERT_NE(head, nullptr);
    }

    // LIFO order: head -> slots[4] -> slots[3] -> ... -> slots[0] -> nullptr.
    PTO2DepListEntry *cur = head;
    for (int i = 4; i >= 0; i--) {
        ASSERT_NE(cur, nullptr);
        EXPECT_EQ(cur->slot_state, &slots[i]) << "Entry " << (4 - i) << " should point to slots[" << i << "]";
        cur = cur->next;
    }
    EXPECT_EQ(cur, nullptr) << "Chain should terminate with nullptr";
}

TEST_F(DepListPoolTest, AdvanceTail) {
    for (int i = 0; i < 4; i++) {
        pool.alloc();
    }
    EXPECT_EQ(pool.used(), 4);
    EXPECT_EQ(pool.available(), POOL_CAP - 4);

    pool.advance_tail(4);
    EXPECT_EQ(pool.used(), 1);
    EXPECT_EQ(pool.available(), POOL_CAP - 1);
}

TEST_F(DepListPoolTest, AdvanceTailBackwardsNoop) {
    pool.alloc();
    pool.alloc();
    pool.advance_tail(3);
    int32_t used_after = pool.used();

    pool.advance_tail(2);
    EXPECT_EQ(pool.used(), used_after);

    pool.advance_tail(3);
    EXPECT_EQ(pool.used(), used_after);
}

TEST_F(DepListPoolTest, HighWaterAccuracy) {
    for (int i = 0; i < 5; i++)
        pool.alloc();
    EXPECT_EQ(pool.high_water, 5);

    pool.advance_tail(4);
    EXPECT_EQ(pool.high_water, 5) << "High water never decreases";

    for (int i = 0; i < 3; i++)
        pool.alloc();
    EXPECT_GE(pool.high_water, 5);
}

// =============================================================================
// Boundary conditions
// =============================================================================

// Prepend chain integrity under pool exhaustion: chain must be walkable.
TEST_F(DepListPoolTest, PrependUnderExhaustion) {
    PTO2TaskSlotState slots[POOL_CAP]{};
    PTO2DepListEntry *head = nullptr;

    int count = 0;
    while (count < POOL_CAP + 5) {
        PTO2DepListEntry *new_head = pool.prepend(head, &slots[count % POOL_CAP]);
        if (!new_head) break;
        head = new_head;
        count++;
    }

    int walk = 0;
    PTO2DepListEntry *cur = head;
    while (cur) {
        walk++;
        cur = cur->next;
        if (walk > count + 1) {
            FAIL() << "Chain has cycle -- walked more entries than allocated";
            break;
        }
    }
    EXPECT_EQ(walk, count);
}
