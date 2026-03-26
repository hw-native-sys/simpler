/**
 * Unit tests for PTO2DepListPool — dependency list entry pool.
 *
 * Tests allocation, prepend (LIFO), null sentinel, exhaustion,
 * tail advance, used/available tracking, and high water mark.
 */

#include <gtest/gtest.h>
#include <cstring>
#include "pto_ring_buffer.h"

// =============================================================================
// Test fixture
// =============================================================================

class DepPoolTest : public ::testing::Test {
protected:
    static constexpr int32_t POOL_CAP = 32;

    PTO2DepListEntry entries[POOL_CAP]{};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2DepListPool pool{};

    void SetUp() override {
        memset(entries, 0, sizeof(entries));
        error_code.store(PTO2_ERROR_NONE);
        pool.init(entries, POOL_CAP, &error_code);
    }
};

// =============================================================================
// Basic alloc and prepend (LIFO order)
// =============================================================================

TEST_F(DepPoolTest, BasicAllocAndPrepend) {
    PTO2TaskSlotState slot_a{}, slot_b{}, slot_c{};

    // Build a linked list: prepend A, B, C → head should be C→B→A
    PTO2DepListEntry* head = nullptr;
    head = pool.prepend(head, &slot_a);
    ASSERT_NE(head, nullptr);
    head = pool.prepend(head, &slot_b);
    ASSERT_NE(head, nullptr);
    head = pool.prepend(head, &slot_c);
    ASSERT_NE(head, nullptr);

    // Verify LIFO: C is head, then B, then A
    EXPECT_EQ(head->slot_state, &slot_c);
    EXPECT_EQ(head->next->slot_state, &slot_b);
    EXPECT_EQ(head->next->next->slot_state, &slot_a);
    EXPECT_EQ(head->next->next->next, nullptr);
}

// =============================================================================
// Null sentinel — entry[0] is reserved
// =============================================================================

TEST_F(DepPoolTest, NullSentinel) {
    // After init, top starts at 1 (entry[0] is reserved as NULL marker)
    PTO2DepListEntry* first = pool.alloc();
    ASSERT_NE(first, nullptr);
    // First allocated entry should NOT be entries[0]
    EXPECT_NE(first, &entries[0]);
}

// =============================================================================
// Pool exhaustion
// =============================================================================

TEST_F(DepPoolTest, Exhaustion) {
    // Pool capacity is 32, top starts at 1.
    // Alloc returns nullptr when top - tail >= capacity
    int count = 0;
    while (count < POOL_CAP + 1) {
        PTO2DepListEntry* e = pool.alloc();
        if (e == nullptr) break;
        count++;
    }
    // Should exhaust at some point
    EXPECT_LE(count, POOL_CAP);
    // On overflow, alloc returns nullptr
    EXPECT_EQ(pool.alloc(), nullptr);
}

// =============================================================================
// Tail advance (batch reclaim)
// =============================================================================

TEST_F(DepPoolTest, TailAdvance) {
    // Allocate 10 entries
    for (int i = 0; i < 10; i++) {
        pool.alloc();
    }
    EXPECT_EQ(pool.used(), 10);

    // Advance tail by 5 (logical reclaim)
    pool.advance_tail(6);  // tail was 1, new tail = 6
    EXPECT_EQ(pool.used(), 5);  // 11 - 6 = 5
}

// =============================================================================
// Used / Available consistency
// =============================================================================

TEST_F(DepPoolTest, UsedAvailable) {
    EXPECT_EQ(pool.used(), 0);
    EXPECT_EQ(pool.available(), POOL_CAP);

    for (int i = 0; i < 5; i++) {
        pool.alloc();
    }
    EXPECT_EQ(pool.used(), 5);
    EXPECT_EQ(pool.available(), POOL_CAP - 5);

    // Advance tail
    pool.advance_tail(4);  // Reclaim entries 1..3
    EXPECT_EQ(pool.used(), 2);  // 6 - 4 = 2
    EXPECT_EQ(pool.available(), POOL_CAP - 2);
}

// =============================================================================
// High water mark tracking
// =============================================================================

TEST_F(DepPoolTest, HighWaterMark) {
    EXPECT_EQ(pool.high_water, 0);

    // Allocate 10 entries
    for (int i = 0; i < 10; i++) {
        pool.alloc();
    }
    EXPECT_EQ(pool.high_water, 10);

    // Reclaim 5
    pool.advance_tail(6);
    // High water should remain at 10
    EXPECT_EQ(pool.high_water, 10);

    // Allocate 8 more — peak should now be higher
    for (int i = 0; i < 8; i++) {
        pool.alloc();
    }
    EXPECT_GE(pool.high_water, 10);
}
