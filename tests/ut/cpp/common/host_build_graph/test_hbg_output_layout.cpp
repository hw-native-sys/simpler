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
 * Packed-output layout totals: the sum of a task's 1024B-aligned outputs is the
 * size of its packed region, so it must keep its full value past 2 GiB. A
 * wrapped total (negative, or wrapped back to a small positive value)
 * under-allocates the region and silently aliases later tasks' outputs.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <sys/mman.h>

#include "host_build_graph/host/orchestrator_internal.h"
#include "host_build_graph/shared_memory.h"
#include "scheduler/scheduler.h"
#include "utils/device_arena.h"

namespace {

TEST(HbgOutputLayout, TwoOneGiBOutputsKeepFullTotal) {
    uint32_t shape[] = {1u << 28};  // FLOAT32 → 1 GiB per output, 1024-aligned
    TensorCreateInfo first(shape, 1, DataType::FLOAT32);
    TensorCreateInfo second(shape, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_output(first, second);

    OutputLayout layout = calculate_output_layout(args);

    EXPECT_EQ(layout.buffer_sizes[0], 1ull << 30);
    EXPECT_EQ(layout.buffer_sizes[1], 1ull << 30);
    EXPECT_EQ(layout.offsets[0], 0ull);
    EXPECT_EQ(layout.offsets[1], 1ull << 30);
    EXPECT_EQ(static_cast<uint64_t>(layout.total_output_size), 2ull << 30);
}

TEST(HbgOutputLayout, TotalsPastFourGiBDoNotWrapToASmallPositiveValue) {
    uint32_t shape_4g[] = {1u << 30};  // FLOAT32 → 4 GiB
    uint32_t shape_1g[] = {1u << 28};  // FLOAT32 → 1 GiB
    TensorCreateInfo big(shape_4g, 1, DataType::FLOAT32);
    TensorCreateInfo small(shape_1g, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_output(big, small);

    OutputLayout layout = calculate_output_layout(args);

    EXPECT_EQ(layout.offsets[1], 4ull << 30);
    EXPECT_EQ(static_cast<uint64_t>(layout.total_output_size), 5ull << 30);
}

class HbgLargeOutputSubmission : public ::testing::TestWithParam<uint32_t> {
protected:
    static constexpr uint64_t HEAP_BYTES = 6ull << 30;
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    void *heap = MAP_FAILED;

    void SetUp() override {
        // Submission only materializes tensor metadata; output storage is never touched.
        heap = mmap(nullptr, HEAP_BYTES, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        ASSERT_NE(heap, MAP_FAILED);
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        const auto layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(layout, runtime_arena);
        sched.seed_queue_slots();
        ASSERT_TRUE(orch.init(sm_handle->sm_base, heap, HEAP_BYTES, CHIP_DEFAULT_GRAPH_TASKS));
        orch.begin_scope();
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
        if (heap != MAP_FAILED) munmap(heap, HEAP_BYTES);
    }
};

TEST_P(HbgLargeOutputSubmission, ReservesFullPackedRegionBeforeTheNextTask) {
    uint32_t first_shape[] = {GetParam()};
    uint32_t second_shape[] = {1u << 28};
    TensorCreateInfo first(first_shape, 1, DataType::FLOAT32);
    TensorCreateInfo second(second_shape, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_output(first, second);
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;

    const auto outputs = orch.submit_task(kernels, args);

    ASSERT_TRUE(outputs.task_id().is_valid());
    ASSERT_EQ(outputs.size(), 2u);
    const uint64_t first_bytes = static_cast<uint64_t>(GetParam()) * sizeof(float);
    const uint64_t second_bytes = 1ull << 30;
    const uint64_t total_bytes = first_bytes + second_bytes;
    const uint64_t base = reinterpret_cast<uintptr_t>(heap);
    EXPECT_EQ(outputs.get_ref(0).buffer.addr, base);
    EXPECT_EQ(outputs.get_ref(0).buffer.size, first_bytes);
    EXPECT_EQ(outputs.get_ref(1).buffer.addr, base + first_bytes);
    EXPECT_EQ(outputs.get_ref(1).buffer.size, second_bytes);
    EXPECT_EQ(orch.task_allocator.heap_top(), total_bytes);
    const auto &task = sm_handle->header->tasks.get_task_by_task_id(outputs.task_id().local_id());
    EXPECT_EQ(reinterpret_cast<uintptr_t>(task.packed_buffer_end), base + total_bytes);

    uint32_t next_shape[] = {256};
    TensorCreateInfo next(next_shape, 1, DataType::FLOAT32);
    CoreTaskArgs next_args;
    next_args.add_output(next);
    const auto next_outputs = orch.submit_task(kernels, next_args);

    ASSERT_TRUE(next_outputs.task_id().is_valid());
    ASSERT_EQ(next_outputs.size(), 1u);
    EXPECT_EQ(next_outputs.get_ref(0).buffer.addr, base + total_bytes);
    EXPECT_EQ(orch.task_allocator.heap_top(), total_bytes + 1024);
    EXPECT_EQ(orch.fatal_code.load(), SIMPLER_ERROR_NONE);
    orch.end_scope();
}

INSTANTIATE_TEST_SUITE_P(TwoAndFiveGiB, HbgLargeOutputSubmission, ::testing::Values(1u << 28, 1u << 30));

}  // namespace
