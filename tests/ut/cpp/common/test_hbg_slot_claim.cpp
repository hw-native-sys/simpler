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
 * The shared-memory slot segments are init-on-write: nothing zeroes them, and a
 * slot's dynamic scheduling fields are established as the orchestrator claims it.
 * That makes the claim the only place the pristine state comes from, for every
 * kind of task — an ordinary submit and the outer task of a recorded Graph alike.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstdint>
#include <vector>

#include "graph_host_state.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"
#include "utils/device_arena.h"

class HbgSlotClaimTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    PTO2SharedMemoryHandle *sm_handle = nullptr;
    PTO2OrchestratorState orch{};
    PTO2SchedulerState sched{};
    PTO2OrchestratorLayout orch_layout{};
    PTO2SchedulerLayout sched_layout{};
    GraphHostStatePtr graph_state;
    std::vector<char> gm_heap;

    static constexpr size_t HEAP_BYTES = 64 * 1024;

    void SetUp() override {
        sm_handle = PTO2SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES * PTO2_MAX_RING_DEPTH);

        orch_layout = PTO2OrchestratorState::reserve_layout(runtime_arena, static_cast<int32_t>(PTO2_TASK_WINDOW_SIZE));
        sched_layout = PTO2SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), HEAP_BYTES, PTO2_TASK_WINDOW_SIZE
        ));
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        orch.wire_arena_pointers(orch_layout, runtime_arena, &sched);

        graph_state = make_graph_host_state();
        ASSERT_NE(graph_state, nullptr);
        orch.graph_host_state = graph_state.get();
    }

    void TearDown() override {
        orch.graph_host_state = nullptr;
        graph_state.reset();
        orch.destroy();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // The state a slot is in before it is claimed when the shared memory outlives
    // one pass. Nothing writes these bytes on the way in, so whatever the previous
    // pass left is what the claim has to overwrite. Every value here is one a
    // completed task really leaves behind.
    void poison_slot(int32_t slot) {
        PTO2TaskSlotState &state = sm_handle->header->ring.get_slot_state_by_slot(slot);
        state.wake_list_head.store(WAKE_LIST_SENTINEL, std::memory_order_relaxed);
        state.next_in_wake_list = &sm_handle->header->ring.get_slot_state_by_slot(slot + 1);
        state.any_subtask_deferred.store(true, std::memory_order_relaxed);
        state.completed_subtasks.store(7, std::memory_order_relaxed);
        state.next_block_idx.store(3, std::memory_order_relaxed);
        state.graph_node_index = 11;
        sm_handle->header->ring.completion_flags[slot].store(1, std::memory_order_relaxed);
    }

    void expect_slot_pristine(int32_t slot) {
        PTO2TaskSlotState &state = sm_handle->header->ring.get_slot_state_by_slot(slot);
        EXPECT_EQ(state.wake_list_head.load(std::memory_order_relaxed), nullptr)
            << "a stale SENTINEL closes the wake list, so no consumer can register on this producer";
        EXPECT_EQ(state.next_in_wake_list, nullptr);
        EXPECT_FALSE(state.has_any_subtask_deferred());
        EXPECT_EQ(state.completed_subtasks.load(std::memory_order_relaxed), 0);
        EXPECT_EQ(state.next_block_idx.load(std::memory_order_relaxed), 0);
        EXPECT_EQ(sm_handle->header->ring.completion_flags[slot].load(std::memory_order_relaxed), 0)
            << "a stale completion flag reports this task done before it has run";
    }
};

TEST_F(HbgSlotClaimTest, OrdinarySubmitClaimsAPoisonedSlot) {
    poison_slot(0);

    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    CoreTaskArgs args;
    args.add_input(boundary);
    ASSERT_TRUE(orch.submit_dummy_task(args).task_id().is_valid());

    expect_slot_pristine(0);
}

// The outer task of a recorded Graph takes a ring slot like any other task, and
// takes it through its own placement rather than through prepare_task — so it
// needs the same claim-time reset. It occupies one heap block and one slot for a
// whole sub-DAG, which is exactly why it is easy to overlook.
TEST_F(HbgSlotClaimTest, GraphOuterTaskClaimsAPoisonedSlot) {
    poison_slot(0);

    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    // A Graph boundary carries the wider arg type: graph_begin deep-copies it for the
    // recording, which the ordinary CoreTaskArgs cannot hold.
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x51ADC1A1, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    // graph_begin publishes the outer shell and creates the recording, handing back the
    // handle graph_prepare binds to the thread that will submit the body. The recorder is
    // normally a worker, so this test plays both roles on one thread.
    ASSERT_NE(graph.recording_handle, nullptr);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    ASSERT_EQ(orch.ring.task_allocator.active_count(), 1);

    PTO2TaskSlotState &outer = sm_handle->header->ring.get_slot_state_by_slot(0);
    ASSERT_EQ(outer.task_kind, TaskKind::GRAPH);
    expect_slot_pristine(0);
}

// A cached replay places its outer task the same way the recording pass did, so
// the second slot has to come out just as clean.
TEST_F(HbgSlotClaimTest, CachedGraphReplayClaimsAPoisonedSlot) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    // A Graph boundary carries the wider arg type: graph_begin deep-copies it for the
    // recording, which the ordinary CoreTaskArgs cannot hold.
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult recorded = orch.graph_begin(0x51ADC1A2, boundary_args, 0x1736);
    ASSERT_TRUE(recorded.recording);
    ASSERT_TRUE(orch.graph_prepare(recorded.recording_handle, boundary_args));
    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    poison_slot(1);
    const GraphScopeResult replay = orch.graph_begin(0x51ADC1A2, boundary_args, 0x1736);
    ASSERT_TRUE(replay.task_id.is_valid());
    ASSERT_EQ(replay.task_id.local(), 1u);

    expect_slot_pristine(1);
}
