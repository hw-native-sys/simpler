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

#include <cstdint>
#include <limits>
#include <string>
#include <vector>

#include "utils/device_arena.h"
#include "runtime_core.h"
#include <functional>
#include "shared_memory.h"
#include "tensormap_and_ringbuffer/task_id.h"

// Advance the simulated scheduler only after the accessor enters its spin loop.
// This makes a missing wait fail without depending on relative thread speed.
static std::function<void()> clock_hook;
static int clock_calls;
uint64_t get_sys_cnt_aicpu() {
    ++clock_calls;
    if (clock_hook) clock_hook();
    return 0;
}

class ScalarAccessDeps : public ::testing::Test {
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
        clock_hook = nullptr;
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }
};

TEST_F(ScalarAccessDeps, WriteWaitsForDeclaredOrderingOnlyReader) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs producer_args;
    TaskId producer = orch.submit_dummy_task(producer_args).task_id();
    CoreTaskArgs reader_args;
    DepFlags kind = DEP_WAIT;
    reader_args.set_dependencies_with_kinds(&producer, &kind, 1);
    TaskId reader = orch.submit_dummy_task(reader_args).task_id();
    ASSERT_TRUE(reader.is_valid());
    auto &slot = sm_handle->header->rings[reader.ring()].get_slot_state_by_task_id(reader.local_id());
    slot.task_state.store(CHIP_TASK_PENDING);
    float data = 5.0f;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1);
    float observed = 0;
    clock_calls = 0;
    clock_hook = [&] {
        if (clock_calls == 2) {
            observed = data;
            slot.task_state.store(CHIP_TASK_COMPLETED);
        }
    };
    uint64_t value = 0;
    float nine = 9.0f;
    memcpy(&value, &nine, sizeof(nine));
    set_tensor_data(&rt, tensor, 1, index, value, &reader, 1);
    EXPECT_FALSE(orch.fatal);
    EXPECT_GE(clock_calls, 2);
    EXPECT_EQ(observed, 5.0f);
    EXPECT_EQ(data, 9.0f);
}

TEST_F(ScalarAccessDeps, WriteWaitsForRetainedConsumer) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    TaskId dep = orch.submit_dummy_task(args).task_id();
    auto &slot = sm_handle->header->rings[dep.ring()].get_slot_state_by_task_id(dep.local_id());
    slot.task_state.store(CHIP_TASK_COMPLETED);
    slot.fanout_count = FANOUT_SCOPE_BIT | 1;
    slot.fanout_refcount.store(0);
    float data = 5.0f;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1);
    clock_calls = 0;
    bool waited = false;
    clock_hook = [&] {
        if (clock_calls == 3) {
            EXPECT_EQ(data, 5.0f);
            waited = true;
            slot.fanout_refcount.store(1);
        }
    };
    set_tensor_data(&rt, tensor, 1, index, 0, &dep, 1);
    EXPECT_FALSE(orch.fatal);
    EXPECT_TRUE(waited);
    EXPECT_EQ(data, 0.0f);
}

TEST_F(ScalarAccessDeps, InvalidWriteDoesNotWaitOrTouchMemory) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    TaskId deps[] = {orch.submit_dummy_task(args).task_id(), TaskId::make(0, 999)};
    float data = 5.0f;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1);
    clock_calls = 0;
    set_tensor_data(&rt, tensor, 1, index, 0, deps, 2);
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(clock_calls, 0);
    EXPECT_EQ(data, 5.0f);
}

TEST_F(ScalarAccessDeps, DefaultArgumentsLinkAndAccessExternalTensor) {
    float data = 5.0f;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1);
    EXPECT_NE(get_tensor_data(&rt, tensor, 1, index), 0U);
    set_tensor_data(&rt, tensor, 1, index, 0);
    EXPECT_EQ(data, 0.0f);
}

TEST_F(ScalarAccessDeps, NullWriteDependencyListDoesNotTouchMemory) {
    float data = 5.0f;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1);
    clock_calls = 0;
    set_tensor_data(&rt, tensor, 1, index, 0, nullptr, 1);
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(clock_calls, 0);
    EXPECT_EQ(data, 5.0f);
}
