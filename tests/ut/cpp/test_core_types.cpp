/**
 * Unit tests for core types in pto_runtime2_types.h
 *
 * Tests PTO2TaskId encoding, alignment assertions, and utility macros.
 */

#include <gtest/gtest.h>

#include "pto_runtime2_types.h"

// =============================================================================
// PTO2TaskId encoding/extraction
// =============================================================================

TEST(TaskId, DefaultIsZero) {
    PTO2TaskId id;
    EXPECT_EQ(id.raw, 0u);
    EXPECT_EQ(id.ring(), 0);
    EXPECT_EQ(id.local(), 0u);
}

TEST(TaskId, MakeAndExtract) {
    auto id = pto2_make_task_id(2, 42);
    EXPECT_EQ(id.ring(), 2);
    EXPECT_EQ(id.local(), 42u);
}

TEST(TaskId, RingInUpperBits) {
    auto id = pto2_make_task_id(3, 0);
    EXPECT_EQ(id.raw, static_cast<uint64_t>(3) << 32);
    EXPECT_EQ(id.ring(), 3);
    EXPECT_EQ(id.local(), 0u);
}

TEST(TaskId, MaxRingMaxLocal) {
    auto id = pto2_make_task_id(255, 0xFFFFFFFF);
    EXPECT_EQ(id.ring(), 255);
    EXPECT_EQ(id.local(), 0xFFFFFFFF);
}

TEST(TaskId, Roundtrip) {
    for (uint8_t ring = 0; ring < PTO2_MAX_RING_DEPTH; ring++) {
        for (uint32_t local : {0u, 1u, 100u, 0xFFFFu, 0xFFFFFFFFu}) {
            auto id = pto2_make_task_id(ring, local);
            EXPECT_EQ(id.ring(), ring);
            EXPECT_EQ(id.local(), local);
        }
    }
}

TEST(TaskId, Equality) {
    auto a = pto2_make_task_id(1, 42);
    auto b = pto2_make_task_id(1, 42);
    auto c = pto2_make_task_id(1, 43);
    auto d = pto2_make_task_id(2, 42);

    EXPECT_TRUE(a == b);
    EXPECT_FALSE(a != b);
    EXPECT_TRUE(a != c);
    EXPECT_TRUE(a != d);
}

TEST(TaskId, SizeIs8Bytes) {
    EXPECT_EQ(sizeof(PTO2TaskId), 8u);
}

// =============================================================================
// PTO2TaskSlotState size (cache-line aligned)
// =============================================================================

TEST(TaskSlotState, SizeIs64Bytes) {
    EXPECT_EQ(sizeof(PTO2TaskSlotState), 64u);
}

// =============================================================================
// PTO2_ALIGN_UP macro
// =============================================================================

TEST(AlignUp, Zero) {
    EXPECT_EQ(PTO2_ALIGN_UP(0, 64), 0u);
}

TEST(AlignUp, AlreadyAligned) {
    EXPECT_EQ(PTO2_ALIGN_UP(64, 64), 64u);
    EXPECT_EQ(PTO2_ALIGN_UP(128, 64), 128u);
}

TEST(AlignUp, NotAligned) {
    EXPECT_EQ(PTO2_ALIGN_UP(1, 64), 64u);
    EXPECT_EQ(PTO2_ALIGN_UP(63, 64), 64u);
    EXPECT_EQ(PTO2_ALIGN_UP(65, 64), 128u);
}

TEST(AlignUp, SmallAlignment) {
    EXPECT_EQ(PTO2_ALIGN_UP(5, 4), 8u);
    EXPECT_EQ(PTO2_ALIGN_UP(4, 4), 4u);
    EXPECT_EQ(PTO2_ALIGN_UP(3, 4), 4u);
}

// =============================================================================
// Task state enum values
// =============================================================================

TEST(TaskState, EnumValues) {
    EXPECT_EQ(PTO2_TASK_PENDING, 0);
    EXPECT_EQ(PTO2_TASK_READY, 1);
    EXPECT_EQ(PTO2_TASK_RUNNING, 2);
    EXPECT_EQ(PTO2_TASK_COMPLETED, 3);
    EXPECT_EQ(PTO2_TASK_CONSUMED, 4);
}

// =============================================================================
// Error code constants
// =============================================================================

TEST(ErrorCodes, Values) {
    EXPECT_EQ(PTO2_ERROR_NONE, 0);
    EXPECT_EQ(PTO2_ERROR_SCOPE_DEADLOCK, 1);
    EXPECT_EQ(PTO2_ERROR_HEAP_RING_DEADLOCK, 2);
    EXPECT_EQ(PTO2_ERROR_FLOW_CONTROL_DEADLOCK, 3);
    EXPECT_EQ(PTO2_ERROR_DEP_POOL_OVERFLOW, 4);
    EXPECT_EQ(PTO2_ERROR_INVALID_ARGS, 5);
    EXPECT_EQ(PTO2_ERROR_SCHEDULER_TIMEOUT, 100);
}

// =============================================================================
// Configuration constants
// =============================================================================

TEST(Config, TaskWindowSizeIsPowerOf2) {
    EXPECT_GT(PTO2_TASK_WINDOW_SIZE, 0);
    EXPECT_EQ(PTO2_TASK_WINDOW_SIZE & (PTO2_TASK_WINDOW_SIZE - 1), 0);
}

TEST(Config, MaxRingDepth) {
    EXPECT_EQ(PTO2_MAX_RING_DEPTH, 4);
}

TEST(Config, AlignSize) {
    EXPECT_EQ(PTO2_ALIGN_SIZE, 64);
}
