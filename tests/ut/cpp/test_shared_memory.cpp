/**
 * Unit tests for PTO2SharedMemory layout from pto_shared_memory.h
 */

#include <gtest/gtest.h>

#include "pto_shared_memory.h"

class SharedMemoryTest : public ::testing::Test {
protected:
    PTO2SharedMemoryHandle* handle = nullptr;

    void SetUp() override {
        handle = pto2_sm_create_default();
        ASSERT_NE(handle, nullptr);
    }

    void TearDown() override {
        if (handle) {
            pto2_sm_destroy(handle);
            handle = nullptr;
        }
    }
};

TEST_F(SharedMemoryTest, CreateDefaultReturnsNonNull) {
    EXPECT_NE(handle->sm_base, nullptr);
    EXPECT_GT(handle->sm_size, 0u);
}

TEST_F(SharedMemoryTest, IsOwner) {
    EXPECT_TRUE(handle->is_owner);
}

TEST_F(SharedMemoryTest, HeaderInitValues) {
    auto* hdr = handle->header;
    EXPECT_EQ(hdr->orchestrator_done.load(), 0);
    EXPECT_EQ(hdr->orch_error_code.load(), 0);
    EXPECT_EQ(hdr->sched_error_bitmap.load(), 0);
    EXPECT_EQ(hdr->sched_error_code.load(), 0);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto& fc = hdr->rings[r].fc;
        EXPECT_EQ(fc.current_task_index.load(), 0);
        EXPECT_EQ(fc.last_task_alive.load(), 0);
    }
}

TEST_F(SharedMemoryTest, Validate) {
    EXPECT_TRUE(pto2_sm_validate(handle));
}

TEST_F(SharedMemoryTest, PerRingIndependence) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        EXPECT_NE(handle->task_descriptors[r], nullptr) << "Ring " << r;
        EXPECT_NE(handle->task_payloads[r], nullptr) << "Ring " << r;
    }
    // Different rings should have different pointers
    for (int r = 1; r < PTO2_MAX_RING_DEPTH; r++) {
        EXPECT_NE(handle->task_descriptors[r], handle->task_descriptors[0]) << "Ring " << r;
    }
}

TEST_F(SharedMemoryTest, PointerAlignment) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto addr = reinterpret_cast<uintptr_t>(handle->task_descriptors[r]);
        EXPECT_EQ(addr % PTO2_ALIGN_SIZE, 0u) << "Ring " << r << " descriptors not aligned";
    }
}

TEST(SharedMemoryCalcSize, NonZero) {
    uint64_t size = pto2_sm_calculate_size(PTO2_TASK_WINDOW_SIZE);
    EXPECT_GT(size, 0u);
}

TEST(SharedMemoryCalcSize, LargerWindowGivesLargerSize) {
    uint64_t small_size = pto2_sm_calculate_size(64);
    uint64_t large_size = pto2_sm_calculate_size(256);
    EXPECT_GT(large_size, small_size);
}

TEST(SharedMemoryCalcSize, HeaderAligned) {
    EXPECT_EQ(sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE, 0u);
}
