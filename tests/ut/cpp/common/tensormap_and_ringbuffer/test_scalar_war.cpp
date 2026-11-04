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
#include <vector>
#include "runtime_core.h"
#include "utils/device_arena.h"

class ScalarWarTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    RuntimeContext rt{};
    OrchestratorState &orch = rt.orchestrator;
    SchedulerState sched{};
    OrchestratorLayout orch_layout{};
    SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096 * CHIP_MAX_RING_DEPTH);

        int32_t task_window_sizes[CHIP_MAX_RING_DEPTH];
        for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
            task_window_sizes[r] = static_cast<int32_t>(CHIP_TASK_WINDOW_SIZE);
        }

        orch_layout = OrchestratorState::reserve_layout(runtime_arena, task_window_sizes);
        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), 4096, CHIP_TASK_WINDOW_SIZE
        ));
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        orch.wire_arena_pointers(orch_layout, runtime_arena, &sched);
    }

    void TearDown() override {
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    uint32_t value = 17;
    simpler::tmr::Tensor tensor{};
    ChipTaskSlotState *reader = nullptr;

    void submit_reader() {
        uint32_t shape[] = {1};
        tensor = simpler::tmr::make_tensor_external(&value, shape, 1, DataType::UINT32);
        orch.begin_scope();
        CoreTaskArgs args;
        args.add_input(tensor);
        const auto submitted = orch.submit_dummy_task(args);
        ASSERT_TRUE(submitted.task_id().is_valid());
        reader = &sm_handle->header->rings[submitted.task_id().ring()].get_slot_state_by_task_id(
            submitted.task_id().local_id()
        );
    }
};

TEST_F(ScalarWarTest, AbortedWaitForAnIncompleteReaderLeavesStorageUntouched) {
    submit_reader();
    ASSERT_NE(reader, nullptr);
    reader->task_state.store(CHIP_TASK_PENDING, std::memory_order_release);
    // A real wait must observe the run's fatal and refuse the write. Skipping
    // the pure reader would perform the memcpy despite its incomplete access.
    sm_handle->header->orch_error_code.store(SIMPLER_ERROR_INVALID_ARGS, std::memory_order_release);
    uint32_t index[] = {0};
    set_tensor_data(&rt, tensor, 1, index, 99);
    EXPECT_EQ(value, 17u);
}

TEST_F(ScalarWarTest, CompletedReaderDoesNotRetainItsConsumersForScalarWrite) {
    submit_reader();
    ASSERT_NE(reader, nullptr);
    reader->task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
    reader->fanout_count = 1;
    reader->fanout_refcount.store(0, std::memory_order_release);
    // Waiting on this reader's consumers would observe fatal and skip the
    // write, although the reader itself has already finished using the data.
    sm_handle->header->orch_error_code.store(SIMPLER_ERROR_INVALID_ARGS, std::memory_order_release);
    uint32_t index[] = {0};
    set_tensor_data(&rt, tensor, 1, index, 99);
    EXPECT_EQ(value, 99u);
}

TEST_F(ScalarWarTest, ScalarReadDoesNotWaitForAnotherReader) {
    submit_reader();
    ASSERT_NE(reader, nullptr);
    reader->task_state.store(CHIP_TASK_PENDING, std::memory_order_release);
    sm_handle->header->orch_error_code.store(SIMPLER_ERROR_INVALID_ARGS, std::memory_order_release);
    uint32_t index[] = {0};
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index), 17u);
}
