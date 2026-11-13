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
 * Claim establishes initial state for ordinary and Graph tasks. Device entry
 * restores execution progress while retaining those prepared task definitions.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph_execution.h"
#include "graph_image_view.h"
#include "graph_host_state.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"
#include "host_build_graph/task_id.h"

class HbgSlotClaimTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    GraphHostStatePtr graph_state;
    std::vector<char> gm_heap;
    // Stands in for a bind's retained staging: the Definition objects are built
    // in here rather than in a buffer of their own.
    std::vector<std::byte> definition_staging;

    static constexpr size_t HEAP_BYTES = 64 * 1024;
    static constexpr size_t STAGING_BYTES = 256 * 1024;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES + 63);

        void *heap = reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(gm_heap.data()) + 63) & ~uintptr_t{63});
        ASSERT_TRUE(orch.init(sm_handle->sm_base, heap, HEAP_BYTES, CHIP_DEFAULT_GRAPH_TASKS));

        definition_staging.assign(STAGING_BYTES, std::byte{0});
        GraphDefinitionArena arena{};
        arena.base = definition_staging.data();
        arena.capacity = definition_staging.size();
        arena.object_prefix_bytes = sizeof(GraphDefinitionHeader);
        arena.object_align = GRAPH_DEFINITION_OBJECT_ALIGN;
        graph_state = make_graph_host_state(arena);
        ASSERT_NE(graph_state, nullptr);
        orch.graph_host_state = graph_state.get();
    }

    void TearDown() override {
        orch.graph_host_state = nullptr;
        graph_state.reset();
        sm_arena.release();
    }

    // The state a slot is in before it is claimed when the shared memory outlives
    // one pass. Nothing writes these bytes on the way in, so whatever the previous
    // pass left is what the claim has to overwrite. Every value here is one a
    // completed task really leaves behind.
    void poison_slot(int32_t slot) {
        ChipTaskSlotState &state = sm_handle->header->tasks.get_slot_state_by_task_id(slot);
        state.wake_list_head.store(WAKE_LIST_SENTINEL, std::memory_order_relaxed);
        state.next_in_wake_list = &sm_handle->header->tasks.get_slot_state_by_task_id(slot + 1);
        state.any_subtask_deferred.store(true, std::memory_order_relaxed);
        state.completed_subtasks.store(7, std::memory_order_relaxed);
        state.next_block_idx.store(3, std::memory_order_relaxed);
        state.sub_task_local_id = 11;
        sm_handle->header->tasks.store_completed(slot);
    }

    void expect_slot_pristine(int32_t slot) {
        ChipTaskSlotState &state = sm_handle->header->tasks.get_slot_state_by_task_id(slot);
        EXPECT_EQ(state.wake_list_head.load(std::memory_order_relaxed), nullptr)
            << "a stale SENTINEL closes the wake list, so no consumer can register on this producer";
        EXPECT_EQ(state.next_in_wake_list, nullptr);
        EXPECT_FALSE(state.has_any_subtask_deferred());
        EXPECT_EQ(state.completed_subtasks.load(std::memory_order_relaxed), 0);
        EXPECT_EQ(state.next_block_idx.load(std::memory_order_relaxed), 0);
        EXPECT_EQ(sm_handle->header->tasks.task_states[slot].load(std::memory_order_relaxed), CHIP_TASK_PENDING)
            << "a stale progress state reports this task done before it has run";
    }
};

TEST_F(HbgSlotClaimTest, OrdinarySubmitClaimsAPoisonedSlot) {
    poison_slot(0);

    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    CoreTaskArgs args;
    args.add_input(boundary);
    ASSERT_TRUE(orch.submit_dummy_task(args).task_id().is_valid());

    expect_slot_pristine(0);
}

// The outer task of a recorded Graph takes a table slot like any other task, and
// takes it through its own placement rather than through prepare_task — so it
// needs the same claim-time reset. It occupies one heap block and one slot for a
// whole sub-DAG, which is exactly why it is easy to overlook.
TEST_F(HbgSlotClaimTest, GraphOuterTaskClaimsAPoisonedSlot) {
    poison_slot(0);

    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    // A Graph boundary carries the wider arg type: graph_begin deep-copies it for the
    // recording, which the ordinary CoreTaskArgs cannot hold.
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x51ADC1A1, boundary_args);
    ASSERT_TRUE(graph.recording);
    // graph_begin publishes the outer shell and creates the recording, handing back the
    // handle graph_prepare binds to the thread that will submit the body. The recorder is
    // normally a worker, so this test plays both roles on one thread.
    ASSERT_NE(graph.recording_handle, nullptr);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs task_args;
    task_args.add_input(graph.params->tensor(0).ref());
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    ASSERT_EQ(orch.task_allocator.active_count(), 1);

    ChipTaskSlotState &outer = sm_handle->header->tasks.get_slot_state_by_task_id(0);
    ASSERT_EQ(outer.task_kind, TaskKind::GRAPH);
    expect_slot_pristine(0);
    orch.graph_commit();
    ASSERT_FALSE(orch.is_fatal());
    auto &tasks = sm_handle->header->tasks;
    tasks.total_tasks = 1;
    const auto &record = graph_state->definitions.at(0x51ADC1A1);
    const auto *definition = graph_record_definition(*graph_state, record);
    ASSERT_NE(definition, nullptr);
    GraphDefinitionHeader framing{};
    framing.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
    framing.graph_key = definition->graph_key;
    framing.definition_bytes = definition->total_bytes;
    std::vector<std::byte> section(sizeof(framing) + definition->total_bytes);
    std::memcpy(section.data(), &framing, sizeof(framing));
    std::memcpy(section.data() + sizeof(framing), definition, definition->total_bytes);
    outer.to_descriptor().graph_definition_offset = sizeof(framing);
    GraphImageView image{};
    image.section = section.data();
    image.bytes = static_cast<uint32_t>(section.size());
    const auto definition_offset = outer.to_descriptor().graph_definition_offset;
    for (int round = 0; round < 3; ++round) {
        auto &header = *sm_handle->header;
        header.sched_error_bitmap.store(1U << round);
        header.sched_error_code.store(100);
        header.sched_error_thread.store(round);
        header.reset_execution_state();
        EXPECT_EQ(header.sched_error_bitmap.load(), 0U);
        EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_NONE);
        EXPECT_EQ(header.sched_error_thread.load(), -1);
        EXPECT_EQ(outer.graph_context, nullptr);
        EXPECT_EQ(outer.task_kind, TaskKind::GRAPH);
        EXPECT_EQ(outer.to_descriptor().graph_definition_offset, definition_offset);
        auto *execution = graph_execution_localize(outer, image);
        ASSERT_NE(execution, nullptr);
        int32_t materialized = 0;
        graph_execution_materialize_slice(outer, *execution, image, 16, &materialized);
        EXPECT_EQ(materialized, 1);
        EXPECT_EQ(execution->retired_tasks.load(), 0);
        execution->retired_tasks.store(1);
        poison_slot(0);
    }
}

// A cached replay places its outer task the same way the recording pass did, so
// the second slot has to come out just as clean.
TEST_F(HbgSlotClaimTest, CachedGraphReplayClaimsAPoisonedSlot) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    // A Graph boundary carries the wider arg type: graph_begin deep-copies it for the
    // recording, which the ordinary CoreTaskArgs cannot hold.
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult recorded = orch.graph_begin(0x51ADC1A2, boundary_args);
    ASSERT_TRUE(recorded.recording);
    ASSERT_TRUE(orch.graph_prepare(recorded.recording_handle, boundary_args));
    CoreTaskArgs task_args;
    task_args.add_input(recorded.params->tensor(0).ref());
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    poison_slot(1);
    const GraphScopeResult replay = orch.graph_begin(0x51ADC1A2, boundary_args);
    ASSERT_TRUE(replay.task_id.is_valid());
    ASSERT_EQ(replay.task_id.local_id(), 1);

    expect_slot_pristine(1);
}

TEST_F(HbgSlotClaimTest, ReexecutionRestoresProgressAndPreservesPreparedArguments) {
    uint32_t shape[] = {16};
    orch.begin_scope();
    CoreTaskArgs allocation;
    TensorCreateInfo created(shape, 1, DataType::FLOAT32);
    allocation.add_output(created);
    const auto hidden = orch.alloc_tensors(allocation);
    ASSERT_TRUE(hidden.task_id().is_valid());
    CoreTaskArgs args;
    auto hidden_tensor = hidden.get_ref(0);
    args.add_input(hidden_tensor);
    args.add_scalar(uint64_t{73});
    args.set_allow_early_resolve(true);
    MixedKernels kernels{};
    kernels.aiv0_kernel_id = 0;
    const auto task = orch.submit_task(kernels, args);
    ASSERT_TRUE(task.task_id().is_valid());
    orch.end_scope();
    auto &tasks = sm_handle->header->tasks;
    tasks.total_tasks = orch.task_allocator.active_count();
    auto &entry = tasks.storage_at(task.task_id().local_id());
    const auto attrs = entry.slot.task_attrs;
    const auto flags = entry.slot.ed_flags;
    ASSERT_NE(flags & ED_FLAG_CANDIDATE, 0);
    const auto tensor = entry.payload.tensor_data();
    const auto fanin = entry.payload.fanin_data();
    ASSERT_EQ(entry.payload.fanin_count, 1);
    ASSERT_EQ(fanin[0], hidden.task_id().local_id());
    ASSERT_EQ(entry.slot.task_kind, TaskKind::KERNEL);
    // The unclaimed tail is neither initialized nor consumed.
    tasks.task_states[tasks.total_tasks].store(CHIP_TASK_PUBLISHED);
    for (int round = 0; round < 3; ++round) {
        SCOPED_TRACE(round);
        poison_slot(task.task_id().local_id());
        entry.slot.ed_publish_list_head.store(WAKE_LIST_SENTINEL);
        entry.slot.next_in_ed_publish_list = &entry.slot;
        entry.slot.wake_scan_cursor = 0;
        entry.slot.ed_publish_scan_cursor = 7;
        entry.payload.early_dispatch_state.store(EARLY_DISPATCH_DISPATCHED);
        entry.payload.early_dispatch_launch_state.store(2);
        entry.payload.early_sync_drain_state.store(3);
        entry.payload.published_block_count.store(4);
        entry.payload.running_slot_count.store(4);
        for (auto &mask : entry.payload.staged_core_mask)
            mask.store(UINT64_MAX);
        auto &header = *sm_handle->header;
        header.sched_error_bitmap.store(1U << round);
        header.sched_error_code.store(100);
        header.sched_error_thread.store(round);
        header.reset_execution_state();
        EXPECT_EQ(header.sched_error_bitmap.load(), 0U);
        EXPECT_EQ(header.sched_error_code.load(), SIMPLER_ERROR_NONE);
        EXPECT_EQ(header.sched_error_thread.load(), -1);
        expect_slot_pristine(task.task_id().local_id());
        EXPECT_TRUE(tasks.is_completed(hidden.task_id().local_id()));
        EXPECT_EQ(tasks.task_states[tasks.total_tasks].load(), CHIP_TASK_PUBLISHED);
        EXPECT_EQ(entry.slot.task_kind, TaskKind::KERNEL);
        EXPECT_EQ(entry.slot.ed_flags, flags);
        EXPECT_EQ(std::memcmp(&entry.slot.task_attrs, &attrs, sizeof(attrs)), 0);
        EXPECT_EQ(entry.slot.wake_scan_cursor, 0xFFFF);
        EXPECT_EQ(entry.slot.ed_publish_scan_cursor, 0);
        EXPECT_EQ(entry.slot.ed_publish_list_head.load(), nullptr);
        EXPECT_EQ(entry.slot.next_in_ed_publish_list, nullptr);
        EXPECT_EQ(entry.payload.tensor_data(), tensor);
        EXPECT_EQ(entry.payload.fanin_data(), fanin);
        EXPECT_EQ(entry.payload.scalar_data()[0], 73U);
        EXPECT_EQ(entry.payload.early_dispatch_state.load(), EARLY_DISPATCH_NONE);
        EXPECT_EQ(entry.payload.early_dispatch_launch_state.load(), EARLY_DISPATCH_LAUNCH_NONE);
        EXPECT_EQ(entry.payload.early_sync_drain_state.load(), EARLY_SYNC_DRAIN_NONE);
        EXPECT_EQ(entry.payload.published_block_count.load(), 0);
        EXPECT_EQ(entry.payload.running_slot_count.load(), 0);
        for (auto &mask : entry.payload.staged_core_mask)
            EXPECT_EQ(mask.load(), 0U);
    }
}
