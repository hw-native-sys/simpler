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
 * calculation, and error handling under the DeviceArena-backed init model:
 *   - Wrapper and SM buffer both live in a caller-supplied DeviceArena.
 *   - handle->init(...) writes fields in place; arena.release() reclaims.
 */

#include <gtest/gtest.h>
#include <cstring>
#include "pto_shared_memory.h"

namespace {

// Reserve + commit a fresh handle + sm_base on `arena` and run init.
// Returns the wrapper pointer (arena-owned) or nullptr on init failure.
PTO2SharedMemoryHandle *make_handle(DeviceArena &arena, uint64_t task_window_size, uint64_t heap_size) {
    const uint64_t sm_size = PTO2SharedMemoryHandle::calculate_size(task_window_size);
    const size_t off_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
    const size_t off_buffer = arena.reserve(static_cast<size_t>(sm_size), PTO2_ALIGN_SIZE);
    if (arena.commit() == nullptr) return nullptr;

    auto *handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(off_handle));
    std::memset(handle, 0, sizeof(*handle));
    void *buffer = arena.region_ptr(off_buffer);
    std::memset(buffer, 0, static_cast<size_t>(sm_size));
    if (!handle->init(buffer, sm_size, task_window_size, heap_size)) return nullptr;
    return handle;
}

}  // namespace

// =============================================================================
// Fixture (default-sized, libc-backed arena)
// =============================================================================

class SharedMemoryTest : public ::testing::Test {
protected:
    DeviceArena arena;
    PTO2SharedMemoryHandle *handle = nullptr;

    void SetUp() override {
        handle = PTO2SharedMemoryHandle::create_and_init_default(arena);
        ASSERT_NE(handle, nullptr);
    }

    void TearDown() override {
        handle = nullptr;
        arena.release();
    }
};

// =============================================================================
// Normal path
// =============================================================================

TEST_F(SharedMemoryTest, CreateDefaultReturnsNonNull) {
    EXPECT_NE(handle->sm_base, nullptr);
    EXPECT_GT(handle->sm_size, 0u);
}

TEST_F(SharedMemoryTest, NotOwnerOfArenaBackedHandle) {
    // The arena owns both the wrapper and the SM buffer; the handle must
    // not try to free them in destroy().
    EXPECT_FALSE(handle->is_owner);
}

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

TEST_F(SharedMemoryTest, Validate) { EXPECT_TRUE(handle->validate()); }

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
TEST(SharedMemoryLayout, RegionsNonOverlapping) {
    DeviceArena arena;
    PTO2SharedMemoryHandle *h = make_handle(arena, /*ws=*/64, /*heap=*/4096);
    ASSERT_NE(h, nullptr);

    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
        uintptr_t desc_start = (uintptr_t)h->header->rings[r].task_descriptors;
        uintptr_t desc_end = desc_start + 64 * sizeof(PTO2TaskDescriptor);
        uintptr_t payload_start = (uintptr_t)h->header->rings[r].task_payloads;

        EXPECT_GE(payload_start, desc_end) << "Ring " << r << ": payload region should not overlap descriptors";
    }

    for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
        uintptr_t this_payload_end = (uintptr_t)h->header->rings[r].task_payloads + 64 * sizeof(PTO2TaskPayload);
        uintptr_t next_desc_start = (uintptr_t)h->header->rings[r + 1].task_descriptors;
        EXPECT_GE(next_desc_start, this_payload_end) << "Ring " << r << " and " << (r + 1) << " should not overlap";
    }
}

// =============================================================================
// Size calculation
// =============================================================================

TEST(SharedMemoryCalcSize, NonZero) {
    uint64_t size = PTO2SharedMemoryHandle::calculate_size(PTO2_TASK_WINDOW_SIZE);
    EXPECT_GT(size, 0u);
}

TEST(SharedMemoryCalcSize, LargerWindowGivesLargerSize) {
    uint64_t small_size = PTO2SharedMemoryHandle::calculate_size(64);
    uint64_t large_size = PTO2SharedMemoryHandle::calculate_size(256);
    EXPECT_GT(large_size, small_size);
}

TEST(SharedMemoryCalcSize, HeaderAligned) { EXPECT_EQ(sizeof(PTO2SharedMemoryHeader) % PTO2_ALIGN_SIZE, 0u); }

TEST(SharedMemoryCalcSize, PerRingDifferentSizes) {
    uint64_t ws[PTO2_MAX_RING_DEPTH] = {128, 256, 512, 1024};
    uint64_t size = PTO2SharedMemoryHandle::calculate_size_per_ring(ws);

    uint64_t uniform_size = PTO2SharedMemoryHandle::calculate_size(128);
    EXPECT_GT(size, uniform_size);
}

// =============================================================================
// Boundary conditions
// =============================================================================

// Zero window size: all ring descriptor pointers collapse to the same address.
TEST(SharedMemoryBoundary, ZeroWindowSize) {
    uint64_t size = PTO2SharedMemoryHandle::calculate_size(0);
    uint64_t header_size = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    EXPECT_EQ(size, header_size);

    DeviceArena arena;
    PTO2SharedMemoryHandle *h = make_handle(arena, /*ws=*/0, /*heap=*/4096);
    if (h) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
            EXPECT_EQ(h->header->rings[r].task_descriptors, h->header->rings[r + 1].task_descriptors)
                << "Zero window: all rings' descriptor pointers collapse to same address";
        }
    }
}

TEST(SharedMemoryBoundary, ValidateDetectsCorruption) {
    DeviceArena arena;
    PTO2SharedMemoryHandle *h = make_handle(arena, /*ws=*/256, /*heap=*/4096);
    ASSERT_NE(h, nullptr);
    EXPECT_TRUE(h->validate());

    h->header->rings[0].fc.current_task_index.store(-1);
    EXPECT_FALSE(h->validate());
}

TEST(SharedMemoryBoundary, InitRejectsUndersizedBuffer) {
    // init() must refuse an SM buffer smaller than calculate_size(window_size).
    PTO2SharedMemoryHandle handle{};
    char buf[64]{};
    EXPECT_FALSE(handle.init(buf, sizeof(buf), /*task_window_size=*/256, /*heap=*/4096));
}
