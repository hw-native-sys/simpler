/**
 * Unit tests for PTO2 Shared Memory layout.
 *
 * Tests size calculation, alignment verification, per-ring isolation,
 * and offset consistency.
 */

#include <gtest/gtest.h>
#include <cstring>
#include "pto_shared_memory.h"

// =============================================================================
// Size calculation
// =============================================================================

TEST(SharedMemoryTest, SizeCalculation) {
    uint64_t size = pto2_sm_calculate_size(1024);
    EXPECT_GT(size, 0u);
    // Size must be at least: header + per-ring descriptors + payloads
    EXPECT_GT(size, sizeof(PTO2SharedMemoryHeader));
}

TEST(SharedMemoryTest, SizeIncreasesWithWindowSize) {
    uint64_t size_small = pto2_sm_calculate_size(256);
    uint64_t size_large = pto2_sm_calculate_size(4096);
    EXPECT_GT(size_large, size_small);
}

// =============================================================================
// Create and destroy
// =============================================================================

TEST(SharedMemoryTest, CreateAndDestroy) {
    PTO2SharedMemoryHandle* handle = pto2_sm_create(256, 4096);
    ASSERT_NE(handle, nullptr);
    EXPECT_NE(handle->sm_base, nullptr);
    EXPECT_GT(handle->sm_size, 0u);
    EXPECT_NE(handle->header, nullptr);
    EXPECT_TRUE(handle->is_owner);

    pto2_sm_destroy(handle);
}

// =============================================================================
// Alignment verification
// =============================================================================

TEST(SharedMemoryTest, AlignmentVerification) {
    PTO2SharedMemoryHandle* handle = pto2_sm_create(256, 4096);
    ASSERT_NE(handle, nullptr);

    // Header should be aligned
    EXPECT_EQ((uintptr_t)handle->header % PTO2_ALIGN_SIZE, 0u);

    // Per-ring task descriptors and payloads should be aligned
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        if (handle->task_descriptors[r] != nullptr) {
            EXPECT_EQ((uintptr_t)handle->task_descriptors[r] % PTO2_ALIGN_SIZE, 0u);
        }
        if (handle->task_payloads[r] != nullptr) {
            EXPECT_EQ((uintptr_t)handle->task_payloads[r] % PTO2_ALIGN_SIZE, 0u);
        }
    }

    pto2_sm_destroy(handle);
}

// =============================================================================
// Per-ring section isolation
// =============================================================================

TEST(SharedMemoryTest, PerRingSectionIsolation) {
    PTO2SharedMemoryHandle* handle = pto2_sm_create(256, 4096);
    ASSERT_NE(handle, nullptr);

    // Descriptor regions of different rings should not overlap
    for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
        if (handle->task_descriptors[r] != nullptr &&
            handle->task_descriptors[r + 1] != nullptr) {
            uintptr_t end_r = (uintptr_t)handle->task_descriptors[r] +
                              256 * sizeof(PTO2TaskDescriptor);
            uintptr_t start_next = (uintptr_t)handle->task_descriptors[r + 1];
            EXPECT_LE(end_r, start_next)
                << "Ring " << r << " descriptors overlap with ring " << r + 1;
        }
    }

    pto2_sm_destroy(handle);
}

// =============================================================================
// Flow control field initialization
// =============================================================================

TEST(SharedMemoryTest, FlowControlInit) {
    PTO2SharedMemoryHandle* handle = pto2_sm_create(256, 4096);
    ASSERT_NE(handle, nullptr);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto& fc = handle->header->rings[r].fc;
        EXPECT_EQ(fc.heap_top.load(), 0u);
        EXPECT_EQ(fc.heap_tail.load(), 0u);
        EXPECT_EQ(fc.current_task_index.load(), 0);
        EXPECT_EQ(fc.last_task_alive.load(), 0);
    }

    EXPECT_EQ(handle->header->orchestrator_done.load(), 0);

    pto2_sm_destroy(handle);
}

// =============================================================================
// Create from existing buffer
// =============================================================================

TEST(SharedMemoryTest, CreateFromBuffer) {
    uint64_t required_size = pto2_sm_calculate_size(256);
    void* buf = aligned_alloc(PTO2_ALIGN_SIZE, required_size);
    ASSERT_NE(buf, nullptr);
    memset(buf, 0, required_size);

    PTO2SharedMemoryHandle* handle =
        pto2_sm_create_from_buffer(buf, required_size, 256, 4096);
    ASSERT_NE(handle, nullptr);
    EXPECT_EQ(handle->sm_base, buf);
    EXPECT_FALSE(handle->is_owner);

    pto2_sm_destroy(handle);  // Should NOT free buf
    free(buf);
}
