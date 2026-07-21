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
#include <string>
#include <vector>

#include "utils/device_arena.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"

class OrchestratorFaninTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorLayout orch_layout{};
    PTO2SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096 * PTO2_MAX_RING_DEPTH);

        int32_t task_window_sizes[PTO2_MAX_RING_DEPTH];
        for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++) {
            task_window_sizes[r] = static_cast<int32_t>(PTO2_TASK_WINDOW_SIZE);
        }

        orch_layout = PTO2OrchestratorState::reserve_layout(runtime_arena, task_window_sizes);
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), 4096, PTO2_TASK_WINDOW_SIZE
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
};

static void
add_runtime_output_arg(L0TaskArgs &args, std::vector<TensorCreateInfo> &create_infos, uint32_t float_count) {
    uint32_t shape[] = {float_count};
    create_infos.emplace_back(shape, 1, DataType::FLOAT32);
    args.add_output(create_infos.back());
}

TEST_F(OrchestratorFaninTest, DuplicateExplicitProducerAddsOneFanin) {
    orch.begin_scope();

    L0TaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    PTO2TaskId deps[] = {producer.task_id(), producer.task_id()};
    L0TaskArgs consumer_args;
    consumer_args.set_dependencies(deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot =
        sm_handle->header->rings[producer.task_id().ring()].get_slot_state_by_task_id(producer.task_id().local());
    auto &consumer_slot =
        sm_handle->header->rings[consumer.task_id().ring()].get_slot_state_by_task_id(consumer.task_id().local());

    ASSERT_NE(consumer_slot.payload, nullptr);
    EXPECT_EQ(consumer_slot.payload->fanin_actual_count, 1);
    EXPECT_EQ(consumer_slot.payload->fanin_inline_slot_states[0], &producer_slot);
    // fanout_count is bit-packed: bit31 (PTO2_FANOUT_SCOPE_BIT) is the owning-scope
    // ref, low bits the consumer count. The duplicate explicit dep is deduped to a
    // single consumer, so this is scope + 1.
    EXPECT_EQ(producer_slot.fanout_count, PTO2_FANOUT_SCOPE_BIT + 1);
}

TEST_F(OrchestratorFaninTest, SubmitPathHeapDeadlockLogReportsRingAndRealHeapState) {
    std::vector<TensorCreateInfo> create_infos;
    create_infos.reserve(8);

    orch.begin_scope();
    orch.begin_scope();
    ASSERT_EQ(orch.current_ring_id(), 1);

    L0TaskArgs first_args;
    add_runtime_output_arg(first_args, create_infos, 1024);  // 4096 bytes
    TaskOutputTensors first = orch.submit_dummy_task(first_args);
    ASSERT_TRUE(first.task_id().is_valid());

    auto &ring = sm_handle->header->rings[1];
    ring.fc.last_task_alive.store(1, std::memory_order_release);

    L0TaskArgs wrap_args;
    add_runtime_output_arg(wrap_args, create_infos, 1);  // wraps, packed to 1024 bytes
    TaskOutputTensors wrapped = orch.submit_dummy_task(wrap_args);
    ASSERT_TRUE(wrapped.task_id().is_valid());

    L0TaskArgs fill_args;
    add_runtime_output_arg(fill_args, create_infos, 512);  // 2048 bytes
    TaskOutputTensors filled = orch.submit_dummy_task(fill_args);
    ASSERT_TRUE(filled.task_id().is_valid());
    ASSERT_EQ(orch.rings[1].task_allocator.heap_used_bytes(), 3072ULL);
    ASSERT_EQ(orch.rings[1].task_allocator.heap_available(), 1024ULL);

    auto &head = ring.get_slot_state_by_task_id(static_cast<int32_t>(wrapped.task_id().local()));
    head.fanout_count = PTO2_FANOUT_SCOPE_BIT;
    head.fanout_refcount.store(0, std::memory_order_release);
    head.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    L0TaskArgs blocked_args;
    add_runtime_output_arg(blocked_args, create_infos, 1);
    testing::internal::CaptureStderr();
    TaskOutputTensors blocked = orch.submit_dummy_task(blocked_args);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(blocked.task_id().is_valid());
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_NE(log.find("FATAL: Task Allocator Deadlock - Heap Exhausted! ring=1"), std::string::npos);
    EXPECT_NE(log.find("Heap ring 1:"), std::string::npos);
    EXPECT_NE(log.find("used=3072"), std::string::npos);
    EXPECT_NE(log.find("available=1024"), std::string::npos);
    EXPECT_EQ(log.find("PTO2_RING_HEAP=<pow2>"), std::string::npos);
}

// Regression for issue #1188: scope_tasks_cap must equal the real in-flight budget
// (sum of the runtime per-ring windows), not the compile-time PTO2_SCOPE_TASKS_CAP.
// reserve_layout only computes offsets, so no commit()/backing is needed here.
TEST(OrchestratorLayoutScopeTasksCap, FollowsRuntimeWindowSum) {
    auto cap_for = [](const int32_t windows[PTO2_MAX_RING_DEPTH]) {
        DeviceArena arena;
        int32_t cap = PTO2OrchestratorState::reserve_layout(arena, windows).scope_tasks_cap;
        arena.release();
        return cap;
    };

    int32_t windows[PTO2_MAX_RING_DEPTH];

    // Default window: cap == the old compile-time value (no behavior change).
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        windows[r] = PTO2_TASK_WINDOW_SIZE;
    EXPECT_EQ(cap_for(windows), PTO2_TASK_WINDOW_SIZE * PTO2_MAX_RING_DEPTH);
    EXPECT_EQ(cap_for(windows), PTO2_SCOPE_TASKS_CAP);

    // Shrunk window: cap shrinks to the real budget (no over-allocation).
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        windows[r] = 4;
    EXPECT_EQ(cap_for(windows), 4 * PTO2_MAX_RING_DEPTH);

    // Enlarged window past the compile default: cap grows to match the rings, so a
    // large scope no longer hits a premature SCOPE_TASKS_OVERFLOW (the bug fixed).
    const int32_t big = PTO2_TASK_WINDOW_SIZE * 2;
    for (int r = 0; r < PTO2_MAX_RING_DEPTH; r++)
        windows[r] = big;
    EXPECT_EQ(cap_for(windows), big * PTO2_MAX_RING_DEPTH);
    EXPECT_GT(cap_for(windows), PTO2_SCOPE_TASKS_CAP);
}
