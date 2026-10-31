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
 * Host scratch retention across binds, on both architectures.
 *
 * A bind hands the orchestrator the storage a previous bind left behind and
 * reconstructs every per-run field. These cases pin the pieces of that
 * contract a poisoned or repeated bind depends on: reinitialization reuses the
 * map and epoch arrays without carrying run state, the packed-output layout
 * writes only OUTPUT positions, a failed submission leaves nothing a successor
 * graph can resolve against, and the storage itself outlives the run that
 * owned it.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <cstring>
#include <vector>

#include "utils/device_arena.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/host/orchestrator_internal.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/task_id.h"

class HbgScratchRetentionTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096);

        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        // Same order the AICPU boots in: the slot arrays are not part of the
        // uploaded image, so nothing can push until they carry their ramp.
        sched.seed_queue_slots();
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), 4096, CHIP_DEFAULT_GRAPH_TASKS));
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }
};

TEST_F(HbgScratchRetentionTest, ReinitializationRetainsStorageButDropsRunState) {
    auto *entries = orch.tensor_map.entry_pool.get();
    auto *seen = orch.fanin_seen_epoch.get();
    orch.fatal_code.store(SIMPLER_ERROR_INVALID_ARGS);
    orch.tensor_pool_cursor = 17;
    orch.fanin_seen_epoch[7] = 99;
    orch.fanin_seen_current_epoch = 99;
    orch.inline_completed_tasks = 12;
    orch.scope_stack_top = 3;
    ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), 4096, CHIP_DEFAULT_GRAPH_TASKS));
    EXPECT_EQ(orch.tensor_map.entry_pool.get(), entries);
    EXPECT_EQ(orch.fanin_seen_epoch.get(), seen);
    EXPECT_FALSE(orch.is_fatal());
    EXPECT_EQ(orch.tensor_pool_cursor, 0);
    EXPECT_EQ(orch.fanin_seen_epoch[7], 0u);
    EXPECT_EQ(orch.fanin_seen_current_epoch, 1u);
    EXPECT_EQ(orch.inline_completed_tasks, 0);
    EXPECT_EQ(orch.scope_stack_top, -1);
    ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), 4096, 128));
    EXPECT_EQ(orch.tensor_map.max_tasks, 128);
    EXPECT_EQ(orch.fanin_seen_epoch.get(), seen);
    ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), 4096, CHIP_DEFAULT_GRAPH_TASKS));
    EXPECT_EQ(orch.tensor_map.max_tasks, CHIP_DEFAULT_GRAPH_TASKS);
    EXPECT_EQ(orch.tensor_map.current_used(), 0);
}

TEST_F(HbgScratchRetentionTest, OutputLayoutInitializesOnlyCreatedOutputPositions) {
    uint32_t shape[] = {16};
    auto input = simpler::hbg::make_tensor_external(gm_heap.data(), shape, 1, DataType::FLOAT32);
    TensorCreateInfo created(shape, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_input(input);
    args.add_output(created);
    args.add_input(input);
    args.add_output(created);
    OutputLayout layout;
    std::memset(&layout, 0xA5, sizeof(layout));
    calculate_output_layout(args, layout);
    const uint64_t bytes = CHIP_ALIGN_UP(created.buffer_size_bytes(), PACKED_OUTPUT_ALIGN);
    EXPECT_EQ(layout.offsets[1], 0u);
    EXPECT_EQ(layout.offsets[3], bytes);
    EXPECT_EQ(layout.buffer_sizes[1], bytes);
    EXPECT_EQ(layout.buffer_sizes[3], bytes);
    EXPECT_EQ(layout.total_output_size, 2 * bytes);
    EXPECT_EQ(layout.offsets[0], 0xA5A5A5A5A5A5A5A5ULL);
    EXPECT_EQ(layout.buffer_sizes[2], 0xA5A5A5A5A5A5A5A5ULL);
    CoreTaskArgs no_outputs;
    no_outputs.add_input(input);
    calculate_output_layout(no_outputs, layout);
    EXPECT_EQ(layout.total_output_size, 0);
}

TEST_F(HbgScratchRetentionTest, KernelTaskClearsDefinitionOffsetFromRetainedStorage) {
    sm_handle->header->tasks.storage_at(0).task.graph_definition_offset = 0xa5a5a5a5;
    uint32_t shape[] = {16};
    auto tensor = simpler::hbg::make_tensor_external(gm_heap.data(), shape, 1, DataType::FLOAT32);
    CoreTaskArgs args;
    args.add_inout(tensor);
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    orch.begin_scope();
    ASSERT_TRUE(orch.submit_task(kernels, args).task_id().is_valid());
    EXPECT_EQ(sm_handle->header->tasks.storage_at(0).task.graph_definition_offset, 0u);
    orch.end_scope();
}

TEST_F(HbgScratchRetentionTest, FailedSubmissionDoesNotContaminateTheNextGraph) {
    auto *entries = orch.tensor_map.entry_pool.get();
    auto *seen = orch.fanin_seen_epoch.get();
    uint32_t shape[] = {16};
    auto tensor = simpler::hbg::make_tensor_external(gm_heap.data(), shape, 1, DataType::FLOAT32);
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    CoreTaskArgs output_args;
    output_args.add_inout(tensor);

    // Repeat real failures after a producer has populated the map. Each new
    // graph reuses the same heap addresses and task ids, exposing stale edges.
    for (int round = 0; round < 8; ++round) {
        orch.begin_scope();
        const auto root = orch.submit_task(kernels, output_args);
        ASSERT_TRUE(root.task_id().is_valid());
        EXPECT_EQ(root.task_id().local_id(), 0);
        EXPECT_GT(orch.tensor_map.current_used(), 0);
        const auto &payload = sm_handle->header->tasks.storage_at(0).payload;
        EXPECT_EQ(payload.fanin_count, 0);

        TaskId invalid_dep{};
        CoreTaskArgs bad_args;
        bad_args.set_dependencies(&invalid_dep, 1);
        orch.submit_dummy_task(bad_args);
        ASSERT_TRUE(orch.is_fatal());
        EXPECT_EQ(orch.fatal_code.load(), SIMPLER_ERROR_INVALID_ARGS);

        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), gm_heap.size(), CHIP_DEFAULT_GRAPH_TASKS));
        EXPECT_EQ(orch.tensor_map.entry_pool.get(), entries);
        EXPECT_EQ(orch.fanin_seen_epoch.get(), seen);
        EXPECT_FALSE(orch.is_fatal());
        EXPECT_EQ(orch.task_allocator.active_count(), 0);
        EXPECT_EQ(orch.tensor_map.current_used(), 0);
        EXPECT_EQ(orch.ready_queue_populations.dummy, 0u);
        for (int i = 0; i < NUM_RESOURCE_SHAPES; ++i) {
            EXPECT_EQ(orch.ready_queue_populations.ready[i], 0u);
            EXPECT_EQ(orch.ready_queue_populations.ready_sync[i], 0u);
        }
    }
    orch.begin_scope();
    ASSERT_TRUE(orch.submit_task(kernels, output_args).task_id().is_valid());
    orch.end_scope();
    EXPECT_FALSE(orch.is_fatal());
    EXPECT_EQ(orch.task_allocator.active_count(), 1);
}

TEST_F(HbgScratchRetentionTest, StorageOutlivesTheRunWithoutRetainingItsState) {
    OrchestrationScratch storage;
    auto *entries = orch.tensor_map.entry_pool.get();
    auto *seen = orch.fanin_seen_epoch.get();
    orch.fatal_code.store(SIMPLER_ERROR_INVALID_ARGS);
    orch.tensor_pool_cursor = 17;
    orch.scope_stack_top = 2;
    storage.swap_storage(orch);
    EXPECT_EQ(orch.tensor_map.entry_pool, nullptr);
    EXPECT_EQ(orch.fanin_seen_epoch, nullptr);
    OrchestratorState next{};
    storage.swap_storage(next);
    ASSERT_TRUE(next.init(sm_handle->sm_base, gm_heap.data(), gm_heap.size(), CHIP_DEFAULT_GRAPH_TASKS));
    EXPECT_EQ(next.tensor_map.entry_pool.get(), entries);
    EXPECT_EQ(next.fanin_seen_epoch.get(), seen);
    EXPECT_FALSE(next.is_fatal());
    EXPECT_EQ(next.tensor_pool_cursor, 0);
    EXPECT_EQ(next.scope_stack_top, -1);
    EXPECT_EQ(next.graph_host_state, nullptr);
    storage.swap_storage(next);
    EXPECT_EQ(next.tensor_map.entry_pool, nullptr);
}
