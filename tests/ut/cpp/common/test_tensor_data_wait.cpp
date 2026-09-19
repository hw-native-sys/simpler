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

#include <functional>
#include <limits>
#include <string>
#include <vector>

#include "aicpu/aicpu_device_config.h"
#include "common/platform_config.h"
#include "common/tensor_data_timeout.h"
#include "runtime_core.h"
#include "shared_memory.h"

namespace {
uint64_t expected_seconds;
int configured_ms;
uint64_t clock_ticks;
uint64_t clock_step;
std::function<void()> on_tick;
}  // namespace

void set_test_clock(uint64_t (*clock)());

// The test platform clock drives the real runtime waits without sleeping.
static uint64_t tensor_wait_clock() {
    const uint64_t result = clock_ticks;
    clock_ticks += clock_step;
    if (on_tick) on_tick();
    return result;
}

class TensorDataWaitTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    RuntimeContext rt{};
    std::vector<char> gm_heap;
    simpler::tmr::Tensor tensor{};
    ChipTaskSlotState *slot = nullptr;
    uint32_t index[1] = {0};

    void SetUp() override {
        set_tensor_data_timeout_ms(configured_ms);
        set_test_clock(tensor_wait_clock);
        clock_ticks = 0;
        clock_step = 0;
        on_tick = {};
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096 * CHIP_MAX_RING_DEPTH);
        int32_t sizes[CHIP_MAX_RING_DEPTH];
        for (auto &size : sizes)
            size = CHIP_TASK_WINDOW_SIZE;
        auto orch_layout = OrchestratorState::reserve_layout(runtime_arena, sizes);
        auto sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(rt.orchestrator.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), 4096, CHIP_TASK_WINDOW_SIZE
        ));
        ASSERT_TRUE(rt.scheduler.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        rt.scheduler.wire_arena_pointers(sched_layout, runtime_arena);
        rt.orchestrator.wire_arena_pointers(orch_layout, runtime_arena, &rt.scheduler);
        rt.orchestrator.begin_scope();
        uint32_t shape[] = {1};
        TensorCreateInfo info(shape, 1, DataType::INT32);
        CoreTaskArgs args;
        args.add_output(info);
        auto outputs = rt.orchestrator.alloc_tensors(args);
        ASSERT_FALSE(rt.orchestrator.fatal);
        tensor = outputs.get_ref(0);
        auto owner = tensor.owner_task_id;
        slot = &sm_handle->header->rings[owner.ring()].get_slot_state_by_task_id(owner.local_id());
        *reinterpret_cast<int32_t *>(tensor.buffer.addr) = 7;
        clock_step = PLATFORM_PROF_SYS_CNT_FREQ;
    }

    void TearDown() override {
        on_tick = {};
        set_test_clock(nullptr);
        set_tensor_data_timeout_ms(0);
        rt.orchestrator.destroy();
        rt.scheduler.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    void expect_timeout(uint64_t start_ticks = 0) {
        EXPECT_TRUE(rt.orchestrator.fatal);
        EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT);
        const uint64_t elapsed_ticks = clock_ticks - start_ticks;
        EXPECT_GT(elapsed_ticks, expected_seconds * PLATFORM_PROF_SYS_CNT_FREQ);
        EXPECT_LE(elapsed_ticks, (expected_seconds + 4) * PLATFORM_PROF_SYS_CNT_FREQ);
    }

    void release_after(uint64_t seconds, bool consumers) {
        on_tick = [this, seconds, consumers]() {
            if (clock_ticks >= seconds * PLATFORM_PROF_SYS_CNT_FREQ) {
                if (consumers) slot->fanout_refcount.store(slot->fanout_count);
                else slot->task_state.store(CHIP_TASK_COMPLETED);
            }
        };
    }
};

TEST_F(TensorDataWaitTest, PlatformBudget) { EXPECT_EQ(get_tensor_data_timeout_ms(), configured_ms); }

TEST_F(TensorDataWaitTest, ShortProducerCompletes) {
    slot->task_state.store(CHIP_TASK_PENDING);
    release_after(2, false);
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index), 7);
    EXPECT_FALSE(rt.orchestrator.fatal);
}

TEST_F(TensorDataWaitTest, SlowProducerUsesPlatformBudget) {
    slot->task_state.store(CHIP_TASK_PENDING);
    release_after(18, false);
    const auto result = get_tensor_data(&rt, tensor, 1, index);
    if (expected_seconds > 18) {
        EXPECT_EQ(result, 7);
        EXPECT_FALSE(rt.orchestrator.fatal);
    } else {
        EXPECT_EQ(result, 0);
        expect_timeout();
    }
}

TEST_F(TensorDataWaitTest, StuckProducerTimesOut) {
    slot->task_state.store(CHIP_TASK_PENDING);
    testing::internal::CaptureStderr();
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index), 0);
    const auto diagnostic = testing::internal::GetCapturedStderr();
    EXPECT_NE(diagnostic.find("FATAL(code=8)"), std::string::npos);
    EXPECT_NE(diagnostic.find("budget_ms=" + std::to_string(expected_seconds * 1000)), std::string::npos);
    EXPECT_NE(diagnostic.find("elapsed_ms="), std::string::npos);
    EXPECT_NE(diagnostic.find("state="), std::string::npos);
    expect_timeout();
}

TEST_F(TensorDataWaitTest, ReadDoesNotWaitForConsumers) {
    slot->fanout_count = FANOUT_SCOPE_BIT + 1;
    slot->fanout_refcount.store(0);
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index), 7);
    EXPECT_FALSE(rt.orchestrator.fatal);
}

TEST_F(TensorDataWaitTest, SlowConsumerUsesPlatformBudget) {
    slot->fanout_count = FANOUT_SCOPE_BIT + 1;
    slot->fanout_refcount.store(FANOUT_SCOPE_BIT);
    release_after(18, true);
    set_tensor_data(&rt, tensor, 1, index, 9);
    if (expected_seconds > 18) {
        EXPECT_EQ(*reinterpret_cast<int32_t *>(tensor.buffer.addr), 9);
        EXPECT_FALSE(rt.orchestrator.fatal);
    } else {
        EXPECT_EQ(*reinterpret_cast<int32_t *>(tensor.buffer.addr), 7);
        expect_timeout();
    }
}

TEST_F(TensorDataWaitTest, StuckConsumerTimesOutWithoutWriting) {
    slot->fanout_count = FANOUT_SCOPE_BIT + 1;
    slot->fanout_refcount.store(FANOUT_SCOPE_BIT);
    set_tensor_data(&rt, tensor, 1, index, 9);
    EXPECT_EQ(*reinterpret_cast<int32_t *>(tensor.buffer.addr), 7);
    expect_timeout();
}

TEST_F(TensorDataWaitTest, ProducerAndConsumerHaveSeparateDeadlines) {
    slot->task_state.store(CHIP_TASK_PENDING);
    slot->fanout_count = FANOUT_SCOPE_BIT + 1;
    slot->fanout_refcount.store(FANOUT_SCOPE_BIT);
    on_tick = [this]() {
        if (clock_ticks >= (expected_seconds - 2) * PLATFORM_PROF_SYS_CNT_FREQ)
            slot->task_state.store(CHIP_TASK_COMPLETED);
        if (clock_ticks >= (expected_seconds * 2 - 4) * PLATFORM_PROF_SYS_CNT_FREQ)
            slot->fanout_refcount.store(slot->fanout_count);
    };
    set_tensor_data(&rt, tensor, 1, index, 9);
    EXPECT_EQ(*reinterpret_cast<int32_t *>(tensor.buffer.addr), 9);
    EXPECT_FALSE(rt.orchestrator.fatal);
}

TEST_F(TensorDataWaitTest, ProducerWaitPreservesExistingError) {
    slot->task_state.store(CHIP_TASK_PENDING);
    sm_handle->header->orch_error_code.store(SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index), 0);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_LT(clock_ticks, 3 * PLATFORM_PROF_SYS_CNT_FREQ);
}

TEST_F(TensorDataWaitTest, ConsumerWaitPreservesExistingError) {
    slot->fanout_count = 1;
    slot->fanout_refcount.store(0);
    sm_handle->header->orch_error_code.store(SIMPLER_ERROR_INVALID_ARGS);
    set_tensor_data(&rt, tensor, 1, index, 9);
    EXPECT_EQ(*reinterpret_cast<int32_t *>(tensor.buffer.addr), 7);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
}

TEST_F(TensorDataWaitTest, CounterWrapStillTimesOut) {
    clock_ticks = std::numeric_limits<uint64_t>::max() - 2 * PLATFORM_PROF_SYS_CNT_FREQ;
    const uint64_t start_ticks = clock_ticks;
    slot->task_state.store(CHIP_TASK_PENDING);
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index), 0);
    expect_timeout(start_ticks);
}

TEST_F(TensorDataWaitTest, ProducerDeadlineSampleObservesSchedulerFailure) {
    slot->task_state.store(CHIP_TASK_PENDING);
    clock_step = (expected_seconds + 1) * PLATFORM_PROF_SYS_CNT_FREQ;
    on_tick = [this]() {
        if (clock_ticks > (expected_seconds + 1) * PLATFORM_PROF_SYS_CNT_FREQ)
            sm_handle->header->sched_error_code.store(SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    };
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index), 0);
    EXPECT_TRUE(rt.orchestrator.fatal);
    EXPECT_EQ(sm_handle->header->sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_NONE);
}

TEST_F(TensorDataWaitTest, ConsumerDeadlineSampleObservesSchedulerFailure) {
    slot->fanout_count = 1;
    slot->fanout_refcount.store(0);
    clock_step = (expected_seconds + 1) * PLATFORM_PROF_SYS_CNT_FREQ;
    on_tick = [this]() {
        // Producer readiness takes the first clock sample; consumer wait takes the next.
        if (clock_ticks > 2 * (expected_seconds + 1) * PLATFORM_PROF_SYS_CNT_FREQ)
            sm_handle->header->sched_error_code.store(SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    };
    set_tensor_data(&rt, tensor, 1, index, 9);
    EXPECT_EQ(*reinterpret_cast<int32_t *>(tensor.buffer.addr), 7);
    EXPECT_TRUE(rt.orchestrator.fatal);
    EXPECT_EQ(sm_handle->header->sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_NONE);
}

TEST_F(TensorDataWaitTest, ProducerWaitStopsOnSchedulerFailure) {
    slot->task_state.store(CHIP_TASK_PENDING);
    on_tick = [this]() {
        if (clock_ticks >= 3 * PLATFORM_PROF_SYS_CNT_FREQ)
            sm_handle->header->sched_error_code.store(SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    };
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index), 0);
    EXPECT_TRUE(rt.orchestrator.fatal);
    EXPECT_EQ(sm_handle->header->sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_NONE);
    EXPECT_LT(clock_ticks, 5 * PLATFORM_PROF_SYS_CNT_FREQ);
}

TEST_F(TensorDataWaitTest, ConsumerWaitStopsOnSchedulerFailure) {
    slot->fanout_count = FANOUT_SCOPE_BIT + 1;
    slot->fanout_refcount.store(FANOUT_SCOPE_BIT);
    on_tick = [this]() {
        if (clock_ticks >= 3 * PLATFORM_PROF_SYS_CNT_FREQ)
            sm_handle->header->sched_error_code.store(SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    };
    set_tensor_data(&rt, tensor, 1, index, 9);
    EXPECT_EQ(*reinterpret_cast<int32_t *>(tensor.buffer.addr), 7);
    EXPECT_TRUE(rt.orchestrator.fatal);
    EXPECT_EQ(sm_handle->header->sched_error_code.load(), SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_NONE);
    EXPECT_LT(clock_ticks, 5 * PLATFORM_PROF_SYS_CNT_FREQ);
}

int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    configured_ms = argc > 1 ? std::stoi(argv[1]) * 1000 : SIM_TENSOR_DATA_TIMEOUT_MS;
    expected_seconds = configured_ms > 0 ? configured_ms / 1000 : TENSOR_DATA_TIMEOUT_MS / 1000;
    return RUN_ALL_TESTS();
}
