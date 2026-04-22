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
 * - validate() checks `top > heap_size`.  top == heap_size is a
 *   legitimate "filled exactly to end" state, so strict > is correct.
 *
 * - Zero window size: if calculate_size() is called with 0, all ring
 *   descriptors/payloads alias the same address.  Current entry path
 *   (create) is called only with valid sizes, but there is no
 *   explicit guard.  create should reject task_window_size==0.
 *
 * - Flow control heap_top validation: validate() does not verify
 *   heap_top <= heap_size.  After a corruption, heap_top could exceed
 *   heap_size without detection.  validate should check both bounds.
 */

#include <gtest/gtest.h>
#include <cstring>
#include <thread>
#include <vector>
#include "pto_shared_memory.h"

// =============================================================================
// Fixture (default-created handle)
// =============================================================================

class SharedMemoryTest : public ::testing::Test {
protected:
    PTO2SharedMemoryHandle *handle = nullptr;

    void SetUp() override {
        handle = PTO2SharedMemoryHandle::create_default();
        ASSERT_NE(handle, nullptr);
    }

    void TearDown() override {
        if (handle) {
            handle->destroy();
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
TEST_F(SharedMemoryTest, RegionsNonOverlapping) {
    uint64_t ws = 64;  // Use a known window size for byte arithmetic
    PTO2SharedMemoryHandle *h = PTO2SharedMemoryHandle::create(ws, 4096);
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

    h->destroy();
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

// Zero window size: all ring descriptors collapse to same address.
TEST(SharedMemoryBoundary, ZeroWindowSize) {
    uint64_t size = PTO2SharedMemoryHandle::calculate_size(0);
    uint64_t header_size = PTO2_ALIGN_UP(sizeof(PTO2SharedMemoryHeader), PTO2_ALIGN_SIZE);
    EXPECT_EQ(size, header_size);

    PTO2SharedMemoryHandle *h = PTO2SharedMemoryHandle::create(0, 4096);
    if (h) {
        for (int r = 0; r < PTO2_MAX_RING_DEPTH - 1; r++) {
            EXPECT_EQ(h->header->rings[r].task_descriptors, h->header->rings[r + 1].task_descriptors)
                << "Zero window: all rings' descriptor pointers collapse to same address";
        }
        h->destroy();
    }
}

TEST(SharedMemoryBoundary, ValidateDetectsCorruption) {
    PTO2SharedMemoryHandle *h = PTO2SharedMemoryHandle::create(256, 4096);
    ASSERT_NE(h, nullptr);
    EXPECT_TRUE(h->validate());

    h->header->rings[0].fc.current_task_index.store(-1);
    EXPECT_FALSE(h->validate());

    h->destroy();
}

TEST(SharedMemoryBoundary, CreateFromUndersizedBuffer) {
    char buf[64]{};
    PTO2SharedMemoryHandle *h = PTO2SharedMemoryHandle::create_from_buffer(buf, 64, 256, 4096);
    EXPECT_EQ(h, nullptr) << "Undersized buffer should fail";
}

// =============================================================================
// Concurrent read/write of per-ring flow control
// =============================================================================

class SharedMemoryConcurrentTest : public ::testing::Test {
protected:
    PTO2SharedMemoryHandle *handle = nullptr;

    void SetUp() override {
        handle = PTO2SharedMemoryHandle::create(256, 4096);
        ASSERT_NE(handle, nullptr);
    }

    void TearDown() override {
        if (handle) {
            handle->destroy();
            handle = nullptr;
        }
    }
};

TEST_F(SharedMemoryConcurrentTest, PerRingTaskIndexIsolation) {
    constexpr int kIterations = 10000;

    auto writer = [&](int ring) {
        auto &fc = handle->header->rings[ring].fc;
        int32_t base = ring * 100000;
        for (int i = 1; i <= kIterations; i++) {
            fc.current_task_index.store(base + i, std::memory_order_release);
        }
    };

    struct Observation {
        bool went_backward = false;
        bool saw_other_ring_range = false;
    };

    auto reader = [&](int ring, Observation *obs) {
        auto &fc = handle->header->rings[ring].fc;
        int32_t base = ring * 100000;
        int32_t prev = 0;
        for (int i = 0; i < kIterations; i++) {
            int32_t val = fc.current_task_index.load(std::memory_order_acquire);
            if (val < prev) {
                obs->went_backward = true;
            }
            if (val != 0 && (val <= base || val > base + kIterations)) {
                obs->saw_other_ring_range = true;
            }
            prev = val;
        }
    };

    Observation ring0;
    Observation ring1;

    std::thread w0(writer, 0);
    std::thread w1(writer, 1);
    std::thread r0(reader, 0, &ring0);
    std::thread r1(reader, 1, &ring1);

    w0.join();
    w1.join();
    r0.join();
    r1.join();

    EXPECT_FALSE(ring0.went_backward) << "Ring 0 current_task_index should be monotonic";
    EXPECT_FALSE(ring1.went_backward) << "Ring 1 current_task_index should be monotonic";
    EXPECT_FALSE(ring0.saw_other_ring_range) << "Ring 0 should not observe ring 1 values";
    EXPECT_FALSE(ring1.saw_other_ring_range) << "Ring 1 should not observe ring 0 values";

    EXPECT_EQ(handle->header->rings[0].fc.current_task_index.load(), static_cast<int32_t>(kIterations));
    EXPECT_EQ(handle->header->rings[1].fc.current_task_index.load(), static_cast<int32_t>(100000 + kIterations));
}

TEST_F(SharedMemoryConcurrentTest, TaskIndexAtomicIncrement) {
    constexpr int kIncrements = 5000;
    constexpr int kThreads = 4;

    auto &fc = handle->header->rings[0].fc;
    fc.current_task_index.store(0, std::memory_order_relaxed);

    auto incrementer = [&]() {
        for (int i = 0; i < kIncrements; i++) {
            fc.current_task_index.fetch_add(1, std::memory_order_acq_rel);
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back(incrementer);
    }
    for (auto &t : threads)
        t.join();

    EXPECT_EQ(fc.current_task_index.load(), kIncrements * kThreads) << "Concurrent increments should not lose updates";
}

TEST_F(SharedMemoryConcurrentTest, LastTaskAliveMonotonic) {
    constexpr int kIterations = 10000;
    constexpr int kThreads = 4;

    auto &fc = handle->header->rings[0].fc;
    fc.last_task_alive.store(0, std::memory_order_relaxed);

    auto advancer = [&](int id) {
        for (int i = 0; i < kIterations; i++) {
            int32_t desired = id * kIterations + i + 1;
            int32_t current = fc.last_task_alive.load(std::memory_order_acquire);
            while (current < desired) {
                if (fc.last_task_alive.compare_exchange_weak(
                        current, desired, std::memory_order_acq_rel, std::memory_order_acquire
                    )) {
                    break;
                }
            }
        }
    };

    std::vector<std::thread> threads;
    for (int i = 0; i < kThreads; i++) {
        threads.emplace_back(advancer, i);
    }
    for (auto &t : threads)
        t.join();

    int32_t final_val = fc.last_task_alive.load();
    EXPECT_EQ(final_val, kIterations * kThreads) << "last_task_alive should advance to the largest published value";
}

TEST_F(SharedMemoryConcurrentTest, ValidateAfterConcurrentWrites) {
    constexpr int kIterations = 1000;

    auto writer = [&](int ring) {
        auto &fc = handle->header->rings[ring].fc;
        for (int i = 0; i < kIterations; i++) {
            fc.current_task_index.store(static_cast<int32_t>(i % 256), std::memory_order_release);
        }
    };

    std::thread w0(writer, 0);
    std::thread w1(writer, 1);
    std::thread w2(writer, 2);
    std::thread w3(writer, 3);
    w0.join();
    w1.join();
    w2.join();
    w3.join();

    EXPECT_TRUE(handle->validate()) << "Valid current_task_index values should pass validation";

    handle->header->rings[2].fc.current_task_index.store(-1, std::memory_order_relaxed);
    EXPECT_FALSE(handle->validate()) << "Corrupted current_task_index should fail validation";
}
