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
#include "scheduler/scheduler_graph.h"

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
    handle.header->sched_error_code.store(SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    const sm_layout::BindUsage usage{kTasks, kTasks * 16, kTasks, kTasks * 8};
    const auto layout = task_image::segments(usage);
    const auto previous = sm_layout::segment_offsets(sm_layout::image_extents(usage));
    ASSERT_LT(layout.end, previous.end);
    auto packed = image(layout.end);
    auto relocated = image(layout.end);
    ASSERT_NE(packed, nullptr);
    ASSERT_NE(relocated, nullptr);
    EXPECT_EQ(task_image::pack(source, kTasks, usage, {0x10000, 256}, static_cast<char *>(packed.get())), layout.end);
    EXPECT_EQ(task_image_error_code(packed.get())->load(), SIMPLER_ERROR_NONE);
    std::memcpy(relocated.get(), packed.get(), layout.end);
    auto graph =
        scheduler_ordinary_graph_view(reinterpret_cast<uint64_t>(relocated.get()) + layout.storage, kTasks, kTasks - 1);
    ASSERT_TRUE(scheduler_graph_view_valid(graph));
    for (uint64_t i = 0; i < kTasks; ++i) {
        const auto &payload = task_image::payload(relocated.get(), layout, i);
        ASSERT_NE(payload.tensor_data(), nullptr);
        EXPECT_EQ(payload.tensor_data()->buffer.addr, 0x10000 + i * 64);
        EXPECT_EQ(payload.scalar_data()[0], HEAP_VIRTUAL_BASE + 11);
        EXPECT_EQ(payload.fanin_count, i == 0 ? 0 : 1);
        if (i != 0) EXPECT_EQ(payload.fanin_data()[0], 0);
        const auto *predicate = scheduler_node_predicate(graph, i);
        ASSERT_NE(predicate, nullptr);
        EXPECT_EQ(predicate->addr, 0x10000 + i * 64);
        EXPECT_EQ(predicate->target, static_cast<int64_t>(i));
    }
}

TEST_F(TaskImageTest, EmptyReusedImageResetsOnlyItsCurrentErrorContract) {
    const sm_layout::BindUsage usage{0, 0, 0, 0};
    const auto layout = task_image::segments(usage);
    auto packed = image(layout.end);
    ASSERT_NE(packed, nullptr);
    std::memset(packed.get(), 0xa5, layout.end);
    EXPECT_EQ(
        task_image::pack(
            static_cast<const char *>(mirror.get()), kTasks, usage, {0x10000, 0}, static_cast<char *>(packed.get())
        ),
        layout.end
    );
    EXPECT_EQ(task_image_error_code(packed.get())->load(), SIMPLER_ERROR_NONE);
    EXPECT_TRUE(scheduler_graph_view_valid(
        scheduler_ordinary_graph_view(reinterpret_cast<uint64_t>(packed.get()) + layout.storage, 0, 0)
    ));
}
}  // namespace
