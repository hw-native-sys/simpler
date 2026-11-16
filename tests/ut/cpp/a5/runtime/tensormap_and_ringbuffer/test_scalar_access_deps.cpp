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

#include <type_traits>
#include <utility>
#include <cstdint>
#include <future>
#include <thread>
#include <limits>
#include <string>
#include <tuple>
#include <vector>

#include "utils/device_arena.h"
#include "runtime_core.h"
#include <functional>
#include "shared_memory.h"
#include "tensormap_and_ringbuffer/task_id.h"

namespace {

template <
    typename... Deps, typename = decltype(get_tensor_data(
                          std::declval<RuntimeContext *>(), std::declval<const simpler::tmr::Tensor &>(), uint32_t{},
                          std::declval<const uint32_t *>(), std::declval<Deps>()...
                      ))>
std::true_type accepts_scalar_read(int);
template <typename... Deps>
std::false_type accepts_scalar_read(...);

static_assert(decltype(accepts_scalar_read<>(0))::value);
static_assert(decltype(accepts_scalar_read<const TaskId *, uint32_t>(0))::value);
static_assert(decltype(accepts_scalar_read<TaskId>(0))::value);
static_assert(!decltype(accepts_scalar_read<const TaskId *>(0))::value);
static_assert(!decltype(accepts_scalar_read<TaskId *>(0))::value);
static_assert(!decltype(accepts_scalar_read<TaskId (&)[2]>(0))::value);
static_assert(!decltype(accepts_scalar_read<std::nullptr_t>(0))::value);

template <
    typename... Deps, typename = decltype(set_tensor_data(
                          std::declval<RuntimeContext *>(), std::declval<const simpler::tmr::Tensor &>(), uint32_t{},
                          std::declval<const uint32_t *>(), uint64_t{}, std::declval<Deps>()...
                      ))>
std::true_type accepts_scalar_write(int);
template <typename... Deps>
std::false_type accepts_scalar_write(...);

static_assert(decltype(accepts_scalar_write<>(0))::value);
static_assert(decltype(accepts_scalar_write<const TaskId *, uint32_t>(0))::value);
static_assert(decltype(accepts_scalar_write<TaskId>(0))::value);
static_assert(!decltype(accepts_scalar_write<const TaskId *>(0))::value);
static_assert(!decltype(accepts_scalar_write<TaskId *>(0))::value);
static_assert(!decltype(accepts_scalar_write<TaskId (&)[2]>(0))::value);
static_assert(!decltype(accepts_scalar_write<std::nullptr_t>(0))::value);

}  // namespace

// Advance the simulated scheduler only after the accessor enters its spin loop.
// This makes a missing wait fail without depending on relative thread speed.
static std::function<void()> clock_hook;
static int clock_calls;
static uint64_t clock_value;
uint64_t get_sys_cnt_aicpu() {
    ++clock_calls;
    if (clock_hook) clock_hook();
    return clock_value;
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
        clock_calls = 0;
        clock_value = 0;
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

TEST_F(ScalarAccessDeps, ReadWaitsForEveryDeclaredTask) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    TaskId deps[] = {orch.submit_dummy_task(args).task_id(), orch.submit_dummy_task(args).task_id()};
    auto &ring = sm_handle->header->rings[deps[0].ring()];
    auto &first = ring.get_slot_state_by_task_id(deps[0].local_id());
    auto &second = ring.get_slot_state_by_task_id(deps[1].local_id());
    first.task_state.store(CHIP_TASK_PENDING);
    second.task_state.store(CHIP_TASK_PENDING);
    int32_t data = 0;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1, DataType::INT32);
    clock_calls = 0;
    clock_hook = [&] {
        if (clock_calls == 2) {
            data = 5;
            first.task_state.store(CHIP_TASK_COMPLETED);
        } else if (clock_calls == 4) {
            data = 9;
            second.task_state.store(CHIP_TASK_COMPLETED);
        }
    };
    EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index, deps, 2), 9U);
    EXPECT_EQ(clock_calls, 4);
    EXPECT_FALSE(orch.fatal);
}

class ScalarAccessRetirement : public ScalarAccessDeps,
                               public ::testing::WithParamInterface<bool> {};

TEST_P(ScalarAccessRetirement, SchedulerReclaimsWhileAccessorWaits) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    TaskId dep = orch.submit_task(kernels, args).task_id();
    ASSERT_TRUE(dep.is_valid());
    orch.end_scope();
    auto &ring = sm_handle->header->rings[dep.ring()];
    auto &slot = ring.get_slot_state_by_task_id(dep.local_id());
    ASSERT_LT(slot.task_state.load(), CHIP_TASK_COMPLETED);
    int32_t data = 5;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1, DataType::INT32);
    std::promise<void> waiting, reclaimed;
    auto waiting_event = waiting.get_future();
    auto reclaimed_event = reclaimed.get_future();
    bool entered_wait = false;
    clock_calls = 0;
    clock_hook = [&] {
        if (clock_calls == 2) {
            entered_wait = true;
            waiting.set_value();
            reclaimed_event.wait();
        }
    };
    std::thread scheduler([&] {
        waiting_event.wait();
        EXPECT_EQ(data, 5);
        data = 7;
        sched.on_task_complete(slot);
        sched.on_task_release(slot);
        EXPECT_EQ(slot.task_state.load(), CHIP_TASK_CONSUMED);
        EXPECT_GT(ring.fc.last_task_alive.load(), dep.local_id());
        reclaimed.set_value();
    });
    if (GetParam()) {
        set_tensor_data(&rt, tensor, 1, index, 9, dep);
        EXPECT_EQ(data, 9);
    } else {
        EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index, dep), 7U);
    }
    // A missing wait still releases the scheduler so the regression fails rather than hangs.
    if (!entered_wait) waiting.set_value();
    scheduler.join();
    EXPECT_TRUE(entered_wait);
    EXPECT_FALSE(orch.fatal);
}

TEST_P(ScalarAccessRetirement, ConsumedTaskBelowWatermarkDoesNotWait) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    TaskId dep = orch.submit_dummy_task(args).task_id();
    auto &ring = sm_handle->header->rings[dep.ring()];
    auto &slot = ring.get_slot_state_by_task_id(dep.local_id());
    slot.task_state.store(CHIP_TASK_CONSUMED);
    slot.reset_for_reuse();
    ring.fc.last_task_alive.store(dep.local_id() + 1);
    int32_t data = 5;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1, DataType::INT32);
    clock_calls = 0;
    if (GetParam()) {
        set_tensor_data(&rt, tensor, 1, index, 9, &dep, 1);
        EXPECT_EQ(data, 9);
    } else {
        EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index, &dep, 1), 5U);
    }
    EXPECT_EQ(clock_calls, 0);
    EXPECT_FALSE(orch.fatal);
}

TEST_P(ScalarAccessRetirement, ResetBeforeWatermarkPublicationDoesNotSpin) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    TaskId dep = orch.submit_dummy_task(args).task_id();
    auto &ring = sm_handle->header->rings[dep.ring()];
    auto &slot = ring.get_slot_state_by_task_id(dep.local_id());
    slot.task_state.store(CHIP_TASK_CONSUMED);
    slot.reset_for_reuse();
    ASSERT_EQ(ring.fc.last_task_alive.load(), dep.local_id());
    int32_t data = 5;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1, DataType::INT32);
    clock_calls = 0;
    // Each wait reads its start time once; a second read means it spun.
    if (GetParam()) {
        set_tensor_data(&rt, tensor, 1, index, 9, &dep, 1);
        EXPECT_EQ(data, 9);
        EXPECT_EQ(clock_calls, 2);
    } else {
        EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index, &dep, 1), 5U);
        EXPECT_EQ(clock_calls, 1);
    }
    EXPECT_FALSE(orch.fatal);
}

TEST_P(ScalarAccessRetirement, ReboundSlotDoesNotWaitForNewGeneration) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    TaskId dep = orch.submit_dummy_task(args).task_id();
    auto &ring = sm_handle->header->rings[dep.ring()];
    auto &slot = ring.get_slot_state_by_task_id(dep.local_id());
    // Model the generation guard independently of the watermark fast path.
    slot.task->task_id = TaskId::make(dep.ring(), dep.local_id() + CHIP_TASK_WINDOW_SIZE);
    slot.task_state.store(CHIP_TASK_PENDING);
    int32_t data = 5;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1, DataType::INT32);
    clock_calls = 0;
    clock_hook = [&] {
        clock_value = clock_calls == 1 ? 0 : std::numeric_limits<uint64_t>::max();
    };
    if (GetParam()) {
        set_tensor_data(&rt, tensor, 1, index, 9, &dep, 1);
        EXPECT_EQ(data, 9);
    } else {
        EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index, &dep, 1), 5U);
    }
    EXPECT_EQ(clock_calls, 0);
    EXPECT_EQ(slot.task_state.load(), CHIP_TASK_PENDING);
    EXPECT_FALSE(orch.fatal);
}

INSTANTIATE_TEST_SUITE_P(ReadAndWrite, ScalarAccessRetirement, ::testing::Bool());

class ScalarAccessInvalidDeps : public ScalarAccessDeps,
                                public ::testing::WithParamInterface<std::tuple<TaskId, bool>> {};

TEST_P(ScalarAccessInvalidDeps, RejectsWholeListBeforeWaitingOrAccessingMemory) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    TaskId deps[] = {orch.submit_dummy_task(args).task_id(), std::get<0>(GetParam())};
    auto &slot = sm_handle->header->rings[deps[0].ring()].get_slot_state_by_task_id(deps[0].local_id());
    slot.task_state.store(CHIP_TASK_PENDING);
    int32_t data = 5;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1, DataType::INT32);
    clock_calls = 0;
    clock_hook = [&] {
        clock_value = clock_calls == 1 ? 0 : std::numeric_limits<uint64_t>::max();
    };
    if (std::get<1>(GetParam())) {
        set_tensor_data(&rt, tensor, 1, index, 9, deps, 2);
    } else {
        EXPECT_EQ(get_tensor_data(&rt, tensor, 1, index, deps, 2), 0U);
    }
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(clock_calls, 0);
    EXPECT_EQ(data, 5);
}

INSTANTIATE_TEST_SUITE_P(
    InvalidIds, ScalarAccessInvalidDeps,
    ::testing::Combine(
        ::testing::Values(
            TaskId::invalid(), TaskId::make(CHIP_MAX_RING_DEPTH, 0), TaskId::make(0, -1), TaskId::make(0, 999)
        ),
        ::testing::Bool()
    )
);

TEST_P(ScalarAccessRetirement, DeclaredTimeoutIdentifiesDependencyAndLeavesMemoryUntouched) {
    orch.begin_scope(ScopeMode::MANUAL);
    CoreTaskArgs args;
    TaskId deps[] = {orch.submit_dummy_task(args).task_id(), orch.submit_dummy_task(args).task_id()};
    auto &ring = sm_handle->header->rings[deps[0].ring()];
    ring.get_slot_state_by_task_id(deps[0].local_id()).task_state.store(CHIP_TASK_COMPLETED);
    auto &slot = ring.get_slot_state_by_task_id(deps[1].local_id());
    bool write = GetParam();
    slot.task_state.store(write ? CHIP_TASK_COMPLETED : CHIP_TASK_PENDING);
    slot.fanout_count = FANOUT_SCOPE_BIT | 1;
    slot.fanout_refcount.store(0);
    int32_t data = 5;
    uint32_t shape[] = {1}, index[] = {0};
    auto tensor = simpler::tmr::make_tensor_external(&data, shape, 1, DataType::INT32);
    clock_calls = 0;
    int timeout_call = write ? 5 : 3;
    clock_hook = [&] {
        if (clock_calls == timeout_call) clock_value = std::numeric_limits<uint64_t>::max();
    };
    testing::internal::CaptureStderr();
    uint64_t result = 0;
    if (write) {
        set_tensor_data(&rt, tensor, 1, index, 9, deps, 2);
    } else {
        result = get_tensor_data(&rt, tensor, 1, index, deps, 2);
    }
    std::string log = testing::internal::GetCapturedStderr();
    EXPECT_EQ(result, 0U);
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(), SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT);
    EXPECT_EQ(data, 5);
    EXPECT_NE(log.find("declared_dep_index=1"), std::string::npos) << log;
    EXPECT_NE(log.find("waited_task=0x1"), std::string::npos) << log;
}

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

TEST_F(ScalarAccessDeps, NoDependencyOverloadsLinkAndAccessExternalTensor) {
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
