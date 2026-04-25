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
 * Unit tests for PTO2SharedMemory layout from pto_shared_memory.h
 *
 * Tests creation, validation, per-ring independence, alignment, size
 * calculation, and error handling.
 *
 * Design contracts:
 *
 * - pto2_sm_validate checks `top > heap_size`.  top == heap_size is a
 *   legitimate "filled exactly to end" state, so strict > is correct.
 *
 * - Zero window size: if pto2_sm_calculate_size() is called with 0, all ring
 *   descriptors/payloads alias the same address.  Current entry path
 *   (pto2_sm_create) is called only with valid sizes, but there is no
 *   explicit guard.  pto2_sm_create should reject task_window_size==0.
 *
 * - Flow control heap_top validation: validate() does not verify
 *   heap_top <= heap_size.  After a corruption, heap_top could exceed
 *   heap_size without detection.  validate should check both bounds.
 */

#include <gtest/gtest.h>
#include <cstring>
#include "pto_shared_memory.h"

// =============================================================================
// Fixture (default-created handle)
// =============================================================================

class SharedMemoryTest : public ::testing::Test {
protected:
    PTO2SharedMemoryHandle *handle = nullptr;

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

// =============================================================================
// Normal path
// =============================================================================

TEST_F(SharedMemoryTest, CreateDefaultReturnsNonNull) {
    EXPECT_NE(handle->sm_base, nullptr);
    EXPECT_GT(handle->sm_size, 0u);
}

TEST_F(SharedMemoryTest, IsOwner) { EXPECT_TRUE(handle->is_owner); }

TEST_F(SharedMemoryTest, HeaderInitValues) {
    auto *hdr = handle->header;
    EXPECT_EQ(hdr->orchestrator_done.load(), 0);
    EXPECT_EQ(hdr->orch_error_code.load(), 0);
    EXPECT_EQ(hdr->sched_error_bitmap.load(), 0);
    EXPECT_EQ(hdr->sched_error_code.load(), 0);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto &fc = hdr->rings[r].fc;
        EXPECT_EQ(fc.current_task_index.load(), 0);
        EXPECT_EQ(fc.last_task_alive.load(), 0);
    }
}

TEST_F(SharedMemoryTest, Validate) { EXPECT_TRUE(pto2_sm_validate(handle)); }

TEST_F(SharedMemoryTest, PerRingIndependence) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        EXPECT_NE(handle->header->rings[r].task_descriptors, nullptr) << "Ring " << r;
        EXPECT_NE(handle->header->rings[r].task_payloads, nullptr) << "Ring " << r;
    }
    for (int r = 1; r < PTO2_MAX_RING_DEPTH; r++) {
        EXPECT_NE(handle->header->rings[r].task_descriptors, handle->header->rings[0].task_descriptors) << "Ring " << r;
    }
}

TEST_F(SharedMemoryTest, PointerAlignment) {
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        auto addr = reinterpret_cast<uintptr_t>(handle->header->rings[r].task_descriptors);
        EXPECT_EQ(addr % PTO2_ALIGN_SIZE, 0u) << "Ring " << r << " descriptors not aligned";
    }
}

TEST_F(SharedMemoryTest, HeaderAlignment) {
    uintptr_t header_addr = (uintptr_t)handle->header;
    EXPECT_EQ(header_addr % PTO2_ALIGN_SIZE, 0u) << "Header must be cache-line aligned";
}

// Descriptor and payload regions don't overlap within or across rings.
TEST_F(SharedMemoryTest, RegionsNonOverlapping) {
    uint64_t ws = 64;  // Use a known window size for byte arithmetic
    PTO2SharedMemoryHandle *h = pto2_sm_create(ws, 4096);
    ASSERT_NE(h, nullptr);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        uintptr_t desc_start = (uintptr_t)h->header->rings[r].task_descriptors;
        uintptr_t desc_end = desc_start + ws * sizeof(PTO2TaskDescriptor);
        uintptr_t payload_start = (uintptr_t)h->header->rings[r].task_payloads;

        EXPECT_GE(payload_start, desc_end) << "Ring " << r << ": payload region should not overlap descriptors";
    }

    for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
        uintptr_t this_payload_end = (uintptr_t)h->header->rings[r].task_payloads + ws * sizeof(PTO2TaskPayload);
        uintptr_t next_desc_start = (uintptr_t)h->header->rings[r + 1].task_descriptors;
        EXPECT_GE(next_desc_start, this_payload_end) << "Ring " << r << " and " << (r + 1) << " should not overlap";
    }

    pto2_sm_destroy(h);
}

// =============================================================================
// Size calculation
// =============================================================================

TEST(SharedMemoryCalcSize, NonZero) {
    uint64_t size = pto2_sm_calculate_size(PTO2_TASK_WINDOW_SIZE);
    EXPECT_GT(size, 0u);
}

TEST(SharedMemoryCalcSize, LargerWindowGivesLargerSize) {
    uint64_t small_size = pto2_sm_calculate_size(64);
    uint64_t large_size = pto2_sm_calculate_size(256);
    EXPECT_GT(large_size, small_size);
}

TEST(SharedMemoryCalcSize, HeaderAligned) { EXPECT_EQ(sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE, 0u); }

TEST(SharedMemoryCalcSize, PerRingDifferentSizes) {
    uint64_t ws[PTO2_MAX_RING_DEPTH] = {128, 256, 512, 1024};
    uint64_t size = pto2_sm_calculate_size_per_ring(ws);

    uint64_t uniform_size = pto2_sm_calculate_size(128);
    EXPECT_GT(size, uniform_size);
}

// =============================================================================
// Boundary conditions
// =============================================================================

// Zero window size: all ring descriptors collapse to same address.
TEST(SharedMemoryBoundary, ZeroWindowSize) {
    uint64_t size = pto2_sm_calculate_size(0);
    uint64_t header_size = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    EXPECT_EQ(size, header_size);

    PTO2SharedMemoryHandle *h = pto2_sm_create(0, 4096);
    if (h) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
            EXPECT_EQ(h->header->rings[r].task_descriptors, h->header->rings[r + 1].task_descriptors)
                << "Zero window: all rings' descriptor pointers collapse to same address";
        }
        pto2_sm_destroy(h);
    }
}

TEST(SharedMemoryBoundary, ValidateDetectsCorruption) {
    PTO2SharedMemoryHandle *h = pto2_sm_create(256, 4096);
    ASSERT_NE(h, nullptr);
    EXPECT_TRUE(pto2_sm_validate(h));

    h->header->rings[0].fc.current_task_index.store(-1);
    EXPECT_FALSE(pto2_sm_validate(h));

    pto2_sm_destroy(h);
}

TEST(SharedMemoryBoundary, ValidateNullHandle) { EXPECT_FALSE(pto2_sm_validate(nullptr)); }

TEST(SharedMemoryBoundary, CreateFromUndersizedBuffer) {
    char buf[64]{};
    PTO2SharedMemoryHandle *h = pto2_sm_create_from_buffer(buf, 64, 256, 4096);
    EXPECT_EQ(h, nullptr) << "Undersized buffer should fail";
}
