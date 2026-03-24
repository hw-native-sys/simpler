/**
 * Unit tests for PTO2TaskRing — task slot ring allocator.
 *
 * Tests basic allocation, monotonic IDs, slot masking, window full,
 * reclamation, and power-of-2 enforcement.
 */

#include <gtest/gtest.h>
#include <atomic>
#include <cstring>
#include "pto_ring_buffer.h"

// =============================================================================
// Test fixture
// =============================================================================

class TaskRingTest : public ::testing::Test {
protected:
    static constexpr int32_t WINDOW_SIZE = 64;

    PTO2TaskDescriptor descriptors[WINDOW_SIZE]{};
    std::atomic<int32_t> current_index{0};
    std::atomic<int32_t> last_alive{0};
    std::atomic<int32_t> error_code{PTO2_ERROR_NONE};
    PTO2TaskRing ring{};

    void SetUp() override {
        memset(descriptors, 0, sizeof(descriptors));
        current_index.store(0);
        last_alive.store(0);
        error_code.store(PTO2_ERROR_NONE);
        pto2_task_ring_init(&ring, descriptors, WINDOW_SIZE, &last_alive, &current_index);
        ring.error_code_ptr = &error_code;
    }
};

// =============================================================================
// Basic allocation
// =============================================================================

TEST_F(TaskRingTest, BasicAlloc) {
    int32_t id = ring.pto2_task_ring_try_alloc();
    EXPECT_EQ(id, 0);
}

// =============================================================================
// Monotonic IDs
// =============================================================================

TEST_F(TaskRingTest, MonotonicId) {
    for (int i = 0; i < 10; i++) {
        int32_t id = ring.pto2_task_ring_try_alloc();
        EXPECT_EQ(id, i);
    }
}

// =============================================================================
// Slot masking (modulo mapping)
// =============================================================================

TEST_F(TaskRingTest, SlotMasking) {
    // window_size = 64, so mask = 63
    // Allocate enough to reach task_id=10, then check slot
    for (int i = 0; i <= 10; i++) {
        ring.pto2_task_ring_try_alloc();
    }
    // task_id=10 should map to slot 10 (10 & 63 = 10)
    EXPECT_EQ(ring.get_task_slot(10), 10);

    // For a larger task_id: slot = task_id & (window_size - 1)
    EXPECT_EQ(ring.get_task_slot(65), 1);   // 65 & 63 = 1
    EXPECT_EQ(ring.get_task_slot(128), 0);  // 128 & 63 = 0
}

// =============================================================================
// Window full — try_alloc returns -1
// =============================================================================

TEST_F(TaskRingTest, WindowFull) {
    // Fill up to window_size - 1 (try_alloc keeps 1 slot empty)
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        int32_t id = ring.pto2_task_ring_try_alloc();
        EXPECT_GE(id, 0) << "Allocation " << i << " should succeed";
    }

    // Next allocation should fail (window full)
    int32_t id = ring.pto2_task_ring_try_alloc();
    EXPECT_EQ(id, -1);
}

// =============================================================================
// Reclaim by advancing last_alive
// =============================================================================

TEST_F(TaskRingTest, ReclaimByAdvance) {
    // Fill up the window
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        ring.pto2_task_ring_try_alloc();
    }
    EXPECT_EQ(ring.pto2_task_ring_try_alloc(), -1);  // Full

    // Advance last_alive to reclaim some slots
    last_alive.store(WINDOW_SIZE / 2);

    // Now allocation should succeed
    int32_t id = ring.pto2_task_ring_try_alloc();
    EXPECT_GE(id, 0);
}

// =============================================================================
// Active count tracking
// =============================================================================

TEST_F(TaskRingTest, ActiveCount) {
    EXPECT_EQ(pto2_task_ring_active_count(&ring), 0);

    for (int i = 0; i < 10; i++) {
        ring.pto2_task_ring_try_alloc();
    }
    EXPECT_EQ(pto2_task_ring_active_count(&ring), 10);

    // Advance last_alive
    last_alive.store(5);
    EXPECT_EQ(pto2_task_ring_active_count(&ring), 5);
}

// =============================================================================
// Has space check
// =============================================================================

TEST_F(TaskRingTest, HasSpace) {
    EXPECT_TRUE(pto2_task_ring_has_space(&ring));

    // Fill up
    for (int i = 0; i < WINDOW_SIZE - 1; i++) {
        ring.pto2_task_ring_try_alloc();
    }
    EXPECT_FALSE(pto2_task_ring_has_space(&ring));

    // Reclaim
    last_alive.store(1);
    EXPECT_TRUE(pto2_task_ring_has_space(&ring));
}
