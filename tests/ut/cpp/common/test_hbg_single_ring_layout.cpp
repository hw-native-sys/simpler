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

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <optional>

#include "host_build_graph/runtime_sizing.h"
#include "pto_shared_memory.h"

namespace {

struct SharedMemoryFixture {
    explicit SharedMemoryFixture(uint64_t task_capacity) :
        sm_size(PTO2SharedMemoryHandle::calculate_size(task_capacity)) {
        off_handle = arena.reserve(sizeof(PTO2SharedMemoryHandle), alignof(PTO2SharedMemoryHandle));
        off_buffer = arena.reserve(static_cast<size_t>(sm_size), PTO2_ALIGN_SIZE);
        EXPECT_NE(arena.commit(), nullptr);
        handle = static_cast<PTO2SharedMemoryHandle *>(arena.region_ptr(off_handle));
        buffer = arena.region_ptr(off_buffer);
        std::memset(handle, 0, sizeof(*handle));
        std::memset(buffer, 0, static_cast<size_t>(sm_size));
    }

    DeviceArena arena;
    uint64_t sm_size{0};
    size_t off_handle{0};
    size_t off_buffer{0};
    PTO2SharedMemoryHandle *handle{nullptr};
    void *buffer{nullptr};
};

TEST(HbgSingleRingLayout, ScalarConfigurationDefinesEntireSharedMemory) {
    constexpr uint64_t kTaskCapacity = 64;
    constexpr uint64_t kHeapSize = 4096;
    SharedMemoryFixture fixture(kTaskCapacity);

    ASSERT_TRUE(fixture.handle->init(fixture.buffer, fixture.sm_size, kTaskCapacity, kHeapSize));
    ASSERT_TRUE(fixture.handle->validate());

    const auto offsets = pto2_sm_layout::ring_segment_offsets(kTaskCapacity);
    auto *base = static_cast<char *>(fixture.buffer);
    const auto &ring = fixture.handle->header->ring;
    EXPECT_EQ(fixture.sm_size, offsets.end);
    EXPECT_EQ(ring.task_capacity, kTaskCapacity);
    EXPECT_EQ(ring.task_capacity_mask, static_cast<int32_t>(kTaskCapacity - 1));
    EXPECT_EQ(ring.heap_size, kHeapSize);
    EXPECT_EQ(ring.task_descriptors, reinterpret_cast<PTO2TaskDescriptor *>(base + offsets.descriptors));
    EXPECT_EQ(ring.task_payloads, reinterpret_cast<PTO2TaskPayload *>(base + offsets.payloads));
    EXPECT_EQ(ring.slot_states, reinterpret_cast<PTO2TaskSlotState *>(base + offsets.slot_states));
    EXPECT_EQ(ring.completion_flags, reinterpret_cast<std::atomic<uint8_t> *>(base + offsets.completion_flags));
}

TEST(HbgSingleRingLayout, AttachPopulatedPreservesGraphState) {
    constexpr uint64_t kTaskCapacity = 64;
    constexpr uint64_t kHeapSize = 4096;
    SharedMemoryFixture fixture(kTaskCapacity);
    ASSERT_TRUE(fixture.handle->init(fixture.buffer, fixture.sm_size, kTaskCapacity, kHeapSize));

    fixture.handle->header->ring.fc.current_task_index.store(7, std::memory_order_relaxed);
    fixture.handle->header->ring.completed_watermark.store(3, std::memory_order_relaxed);
    fixture.handle->header->orchestrator_done.store(1, std::memory_order_relaxed);

    PTO2SharedMemoryHandle attached{};
    ASSERT_TRUE(attached.attach_populated(fixture.buffer, fixture.sm_size, kTaskCapacity));
    EXPECT_EQ(attached.header->ring.fc.current_task_index.load(std::memory_order_relaxed), 7);
    EXPECT_EQ(attached.header->ring.completed_watermark.load(std::memory_order_relaxed), 3);
    EXPECT_EQ(attached.header->orchestrator_done.load(std::memory_order_relaxed), 1);
}

TEST(HbgSingleRingLayout, SharedMemoryRejectsInvalidTaskCapacityBeforePointerSetup) {
    alignas(PTO2_ALIGN_SIZE) std::array<std::byte, 4096> storage{};
    constexpr std::array<uint64_t, 3> kInvalidTaskCapacities{0, 3, 5};

    for (uint64_t task_capacity : kInvalidTaskCapacities) {
        SCOPED_TRACE(task_capacity);
        PTO2SharedMemoryHandle handle{};
        EXPECT_FALSE(handle.init(storage.data(), storage.size(), task_capacity, 4096));
        EXPECT_EQ(handle.sm_base, nullptr);
        EXPECT_FALSE(handle.attach_populated(storage.data(), storage.size(), task_capacity));
        EXPECT_EQ(handle.sm_base, nullptr);
    }
}

TEST(HbgSingleRingLayout, TaskIdKeepsSharedWireEncoding) {
    constexpr uint8_t kRingId = 3;
    constexpr uint32_t kLocalId = 7;
    const PTO2TaskId task_id = PTO2TaskId::make(kRingId, kLocalId);

    EXPECT_EQ(task_id.raw, (static_cast<uint64_t>(kRingId) << 32) | kLocalId);
    EXPECT_EQ(task_id.ring(), kRingId);
    EXPECT_EQ(task_id.local(), kLocalId);
}

TEST(HbgSingleRingLayout, RuntimeSizingUsesOnlySlotZeroWithTaskOverridePrecedence) {
    constexpr hbg_runtime_sizing::HbgRuntimeSizing kDefaults{16, 1024};
    const uint64_t task_capacity_overrides[4] = {128, 3, 3, 3};
    const uint64_t heaps[4] = {4096, 1, 1, 1};
    hbg_runtime_sizing::HbgRuntimeSizing sizing{};

    ASSERT_TRUE(hbg_runtime_sizing::resolve(kDefaults, 64, 2048, task_capacity_overrides, heaps, &sizing));
    EXPECT_EQ(sizing.task_capacity, 128u);
    EXPECT_EQ(sizing.heap_size, 4096u);
}

TEST(HbgSingleRingLayout, RuntimeSizingFallsBackFromEnvironmentToDefaults) {
    constexpr hbg_runtime_sizing::HbgRuntimeSizing kDefaults{16, 1024};
    const uint64_t unset[4] = {};
    hbg_runtime_sizing::HbgRuntimeSizing sizing{};

    ASSERT_TRUE(hbg_runtime_sizing::resolve(kDefaults, 64, 2048, unset, unset, &sizing));
    EXPECT_EQ(sizing.task_capacity, 64u);
    EXPECT_EQ(sizing.heap_size, 2048u);

    ASSERT_TRUE(hbg_runtime_sizing::resolve(kDefaults, std::nullopt, std::nullopt, nullptr, nullptr, &sizing));
    EXPECT_EQ(sizing.task_capacity, kDefaults.task_capacity);
    EXPECT_EQ(sizing.heap_size, kDefaults.heap_size);
}

TEST(HbgSingleRingLayout, RuntimeSizingReadsPackedOverridesWithoutAlignedLoads) {
    constexpr hbg_runtime_sizing::HbgRuntimeSizing kDefaults{16, 1024};
    constexpr uint64_t kTaskCapacity = 256;
    constexpr uint64_t kHeap = 8192;
    alignas(uint64_t) std::array<std::byte, 2 * sizeof(uint64_t) + 1> packed{};
    std::memcpy(packed.data() + 1, &kTaskCapacity, sizeof(kTaskCapacity));
    std::memcpy(packed.data() + 1 + sizeof(uint64_t), &kHeap, sizeof(kHeap));
    const void *task_capacity_override = packed.data() + 1;
    const void *heap_override = packed.data() + 1 + sizeof(uint64_t);
    hbg_runtime_sizing::HbgRuntimeSizing sizing{};

    ASSERT_TRUE(
        hbg_runtime_sizing::resolve(
            kDefaults, std::nullopt, std::nullopt, task_capacity_override, heap_override, &sizing
        )
    );
    EXPECT_EQ(sizing.task_capacity, kTaskCapacity);
    EXPECT_EQ(sizing.heap_size, kHeap);
}

}  // namespace
