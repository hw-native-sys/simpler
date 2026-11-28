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

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <memory>

#include "image_pack.h"
#include "task_image_status.h"

namespace {
class TaskImageTest : public ::testing::Test {
protected:
    static constexpr uint64_t kTasks = 3;
    using Memory = std::unique_ptr<void, decltype(&std::free)>;
    Memory mirror{nullptr, std::free};
    SharedMemoryHandle handle{};

    void SetUp() override {
        const uint64_t bytes = SharedMemoryHandle::calculate_size(kTasks);
        mirror.reset(std::aligned_alloc(CHIP_ALIGN_SIZE, bytes));
        ASSERT_NE(mirror, nullptr);
        ASSERT_TRUE(handle.init(mirror.get(), bytes, kTasks));
    }
    Memory image(uint64_t bytes) { return Memory(std::aligned_alloc(CHIP_ALIGN_SIZE, bytes), std::free); }
};

TEST_F(TaskImageTest, PublishesArgumentsAndPredicatesWithoutAicpuSchedulingState) {
    const auto from = sm_layout::segment_offsets(kTasks);
    auto *source = static_cast<char *>(mirror.get());
    auto *tensors = reinterpret_cast<simpler::hbg::Tensor *>(source + from.tensor_pool);
    auto *scalars = reinterpret_cast<uint64_t *>(source + from.scalar_pool);
    auto *fanins = reinterpret_cast<int32_t *>(source + from.fanin_pool);
    for (uint64_t i = 0; i < kTasks; ++i) {
        auto &entry = *new (&handle.header->tasks.task_storage[i]) ChipTaskStorage{};
        entry.task.task_id = TaskId::make_global(static_cast<int32_t>(i));
        entry.task.kernel_id[0] = 7;
        entry.task.kernel_id[1] = entry.task.kernel_id[2] = INVALID_KERNEL_ID;
        std::memset(&entry.slot, 0xa5, sizeof(entry.slot));
        entry.payload.tensor_count = 1;
        entry.payload.scalar_count = 1;
        entry.payload.fanin_count = i == 0 ? 0 : 1;
        entry.payload.bind_regions(tensors + i, scalars + i * 8, fanins + i * 16);
        fanins[i * 16] = 0;
        scalars[i * 8] = HEAP_VIRTUAL_BASE + 11;
        uint32_t shape[] = {4};
        new (&tensors[i]) simpler::hbg::Tensor{};
        tensors[i].init_external(
            reinterpret_cast<void *>(HEAP_VIRTUAL_BASE + i * 64), 16, shape, 1, DataType::UINT32, 0, false,
            AddressSpace::DEVICE
        );
        tensors[i].owner_task_id = TaskId::make_global(static_cast<int32_t>(i));
        entry.payload.predicate.addr = HEAP_VIRTUAL_BASE + i * 64;
        entry.payload.predicate.elem_size = 4;
        entry.payload.predicate.op = PredicateOp::EQ;
        entry.payload.predicate.target = static_cast<int64_t>(i);
    }
    const sm_layout::BindUsage usage{kTasks, kTasks * 16, kTasks, kTasks * 8};
    const auto layout = task_image::layout(source, kTasks, usage);
    auto packed = image(layout.end);
    auto relocated = image(layout.end);
    ASSERT_NE(packed, nullptr);
    ASSERT_NE(relocated, nullptr);
    std::memset(packed.get(), 0xa5, layout.end);
    std::memset(relocated.get(), 0x5a, layout.end);
    EXPECT_EQ(
        task_image::pack(source, kTasks, usage, {0x10000, 256}, static_cast<char *>(packed.get()), 0), layout.end
    );
    EXPECT_EQ(task_image_error_code(packed.get())->load(), SIMPLER_ERROR_NONE);
    std::memcpy(relocated.get(), packed.get(), layout.end);
    for (uint64_t i = 0; i < kTasks; ++i) {
        const auto &payload = task_image::payload(relocated.get(), layout, i);
        ASSERT_NE(payload.tensor_data(), nullptr);
        EXPECT_EQ(payload.tensor_data()->buffer.addr, 0x10000 + i * 64);
        EXPECT_EQ(payload.scalar_data()[0], HEAP_VIRTUAL_BASE + 11);
        EXPECT_EQ(payload.fanin_count, i == 0 ? 0 : 1);
        if (i != 0) EXPECT_EQ(payload.fanin_data()[0], 0);
        const auto *predicate = &payload.predicate;
        EXPECT_EQ(predicate->addr, 0x10000 + i * 64);
        EXPECT_EQ(predicate->target, static_cast<int64_t>(i));
    }
}

TEST_F(TaskImageTest, EmptyReusedImageResetsOnlyItsCurrentErrorContract) {
    const sm_layout::BindUsage usage{0, 0, 0, 0};
    const auto layout = task_image::layout(static_cast<const char *>(mirror.get()), kTasks, usage);
    auto packed = image(layout.end);
    ASSERT_NE(packed, nullptr);
    std::memset(packed.get(), 0xa5, layout.end);
    EXPECT_EQ(
        task_image::pack(
            static_cast<const char *>(mirror.get()), kTasks, usage, {0x10000, 0}, static_cast<char *>(packed.get()), 0
        ),
        layout.end
    );
    EXPECT_EQ(task_image_error_code(packed.get())->load(), SIMPLER_ERROR_NONE);
    EXPECT_EQ(layout.scalar_pool, layout.end);
}
}  // namespace

namespace {
class CalendarTaskImage : public ::testing::Test {
protected:
    static constexpr uint64_t capacity = GRAPH_MAX_SCALAR_ARGS / MAX_SCALAR_ARGS + 4;
    using Memory = std::unique_ptr<void, decltype(&std::free)>;
    Memory mirror{nullptr, std::free};
    SharedMemoryHandle handle;
    void SetUp() override {
        const uint64_t bytes = SharedMemoryHandle::calculate_size(capacity);
        mirror.reset(std::aligned_alloc(CHIP_ALIGN_SIZE, bytes));
        ASSERT_NE(mirror, nullptr);
        ASSERT_TRUE(handle.init(mirror.get(), bytes, capacity));
    }
    char *source() { return static_cast<char *>(mirror.get()); }
    Memory image(uint64_t bytes) { return Memory(std::aligned_alloc(CHIP_ALIGN_SIZE, bytes), std::free); }
};

TEST_F(CalendarTaskImage, ScalarsAndKernelArgumentsUseTheSameStorage) {
    const auto from = sm_layout::segment_offsets(capacity);
    auto *tensors = reinterpret_cast<simpler::hbg::Tensor *>(source() + from.tensor_pool);
    auto *scalars = reinterpret_cast<uint64_t *>(source() + from.scalar_pool);
    auto &first = *new (&handle.header->tasks.task_storage[0]) ChipTaskStorage{};
    auto &empty = *new (&handle.header->tasks.task_storage[1]) ChipTaskStorage{};
    first.payload.tensor_count = 2;
    first.payload.scalar_count = 3;
    first.payload.bind_regions(tensors, scalars, nullptr);
    empty.payload.bind_regions(nullptr, nullptr, nullptr);
    scalars[0] = 17;
    scalars[1] = UINT64_MAX;
    scalars[2] = 0x123456789abcdef0ULL;
    const uint32_t shape[]{4};
    for (int i = 0; i < 2; ++i) {
        new (&tensors[i]) simpler::hbg::Tensor{};
        tensors[i].init_external(
            reinterpret_cast<void *>(HEAP_VIRTUAL_BASE + i * 16), 16, shape, 1, DataType::UINT32, 0, false,
            AddressSpace::DEVICE
        );
    }
    const sm_layout::BindUsage usage{2, 0, 2, 8};
    const auto layout = task_image::layout(source(), capacity, usage);
    auto packed = image(layout.end);
    constexpr uint64_t device_base = 0x10000000;
    ASSERT_EQ(
        task_image::pack(source(), capacity, usage, {0x40000, 128}, static_cast<char *>(packed.get()), device_base),
        layout.end
    );
    auto *args = reinterpret_cast<DispatchPayload *>(static_cast<char *>(packed.get()) + layout.scalar_pool);
    const auto &payload = task_image::payload(packed.get(), layout, 0);
    EXPECT_EQ(payload.scalar_data(), args[0].args + 2);
    EXPECT_EQ(args[0].args[0], device_base + layout.tensor_pool);
    EXPECT_EQ(args[0].args[1], device_base + layout.tensor_pool + 128);
    EXPECT_EQ(args[0].args[2], 17U);
    EXPECT_EQ(args[0].args[3], UINT64_MAX);
    EXPECT_EQ(args[0].args[4], 0x123456789abcdef0ULL);
    EXPECT_EQ(payload.tensor_data()[0].buffer.addr, 0x40000U);
    EXPECT_EQ(payload.tensor_data()[1].buffer.addr, 0x40010U);
    args[0].args[2] = 93;
    EXPECT_EQ(payload.scalar_data()[0], 93U);
    EXPECT_EQ(task_image::payload(packed.get(), layout, 1).scalar_data(), args[1].args);
    EXPECT_EQ(layout.end, layout.scalar_pool + 2 * sizeof(DispatchPayload));
}

TEST_F(CalendarTaskImage, WideGraphBoundaryDoesNotConsumeAKernelVector) {
    const auto from = sm_layout::segment_offsets(capacity);
    auto *scalars = reinterpret_cast<uint64_t *>(source() + from.scalar_pool);
    auto &graph = *new (&handle.header->tasks.task_storage[0]) ChipTaskStorage{};
    graph.slot.task_kind = TaskKind::GRAPH;
    graph.payload.scalar_count = GRAPH_MAX_SCALAR_ARGS;
    graph.payload.bind_regions(nullptr, scalars, nullptr);
    auto &ordinary = *new (&handle.header->tasks.task_storage[1]) ChipTaskStorage{};
    ordinary.payload.scalar_count = 1;
    ordinary.payload.bind_regions(nullptr, scalars + GRAPH_MAX_SCALAR_ARGS, nullptr);
    for (uint32_t i = 0; i < GRAPH_MAX_SCALAR_ARGS + 1; ++i)
        scalars[i] = 100 + i;
    const sm_layout::BindUsage usage{2, 0, 0, GRAPH_MAX_SCALAR_ARGS + 8};
    const auto layout = task_image::layout(source(), capacity, usage);
    auto packed = image(layout.end);
    ASSERT_EQ(
        task_image::pack(
            source(), capacity, usage, {HEAP_VIRTUAL_BASE, 0}, static_cast<char *>(packed.get()), 0x10000000
        ),
        layout.end
    );
    auto *args = reinterpret_cast<DispatchPayload *>(static_cast<char *>(packed.get()) + layout.scalar_pool);
    const auto &boundary = task_image::payload(packed.get(), layout, 0);
    EXPECT_EQ(boundary.scalar_data(), reinterpret_cast<uint64_t *>(args + 2));
    for (uint32_t i = 0; i < GRAPH_MAX_SCALAR_ARGS; ++i)
        EXPECT_EQ(boundary.scalar_data()[i], 100 + i);
    EXPECT_EQ(task_image::payload(packed.get(), layout, 1).scalar_data(), args[1].args);
    EXPECT_EQ(args[1].args[0], 100 + GRAPH_MAX_SCALAR_ARGS);
}
}  // namespace
