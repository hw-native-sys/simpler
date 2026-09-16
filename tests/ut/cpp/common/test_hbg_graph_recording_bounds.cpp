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
 * A recorded task's id is what its entry in the recording's hazard map is keyed on,
 * and that map holds MAX_IN_GRAPH_TASKS task chains. An IN_GRAPH id's low field is the
 * task's index within its body, so the key is in range however far into a run the
 * Graph begins — which a GLOBAL id carrying the allocator's own local id would not
 * be.
 *
 * These tests place the Graph past that many ordinary tasks, where an id drawn from
 * the run's numbering would key the map outside its chains.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"
#include "host_build_graph/task_id.h"

class HbgGraphRecordingBoundsTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    GraphHostStatePtr graph_state;
    std::vector<char> gm_heap;
    std::vector<std::byte> definition_staging;

    static constexpr size_t HEAP_BYTES = 8 * 1024 * 1024;
    static constexpr size_t STAGING_BYTES = 256 * 1024;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES);

        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), HEAP_BYTES, CHIP_DEFAULT_GRAPH_TASKS));

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
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }
};

// A Graph recorded after MAX_IN_GRAPH_TASKS ordinary tasks. Its recorded task registers
// its outputs in the recording hazard map keyed on its own id, so that id decides
// whether the key lands inside the map's task chains.
TEST_F(HbgGraphRecordingBoundsTest, RecordedTaskIsKeyedByItsIndexNotByTheRunsNumbering) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    // Move the allocator's local-id counter past the recording map's task-chain
    // count, the way any run that submits a while before its first Graph does.
    for (uint32_t i = 0; i < MAX_IN_GRAPH_TASKS; ++i) {
        CoreTaskArgs filler;
        filler.add_input(boundary);
        ASSERT_TRUE(orch.submit_dummy_task(filler).task_id().is_valid()) << "filler task " << i;
    }
    ASSERT_EQ(orch.task_allocator.active_count(), MAX_IN_GRAPH_TASKS);

    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x6B0D5A1E, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_NE(graph.recording_handle, nullptr);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
    // What a body actually receives: the entry's own parameter list, whose tensors carry
    // recording-space addresses and PARAM provenance. Passing the caller's tensor instead
    // is the case the classifier rejects.
    const simpler::hbg::Tensor &param = graph.params->tensor(0).ref();

    // An INOUT operand is what makes the recorded task register an output: the
    // recording's hazard map exists for exactly the write-in-place shape.
    CoreTaskArgs task_args;
    task_args.add_inout(param);
    const TaskId in_graph_task_id = orch.submit_dummy_task(task_args).task_id();
    ASSERT_TRUE(in_graph_task_id.is_valid());
    EXPECT_EQ(in_graph_task_id.space(), TaskId::Space::IN_GRAPH)
        << "a recorded task must not take a GLOBAL id: nothing resolves it against the task table, and its low "
           "field is what keys the recording's hazard map";
    EXPECT_EQ(in_graph_task_id.local_id(), 0)
        << "the first recorded task's low field is task index 0, independent of how many tasks the run has "
           "already allocated";
    EXPECT_LT(in_graph_task_id.local_id(), MAX_IN_GRAPH_TASKS);

    ASSERT_TRUE(orch.graph_end());
}

// Channel the outer shell owns: a boundary tensor allocated before the Graph makes
// compute_task_fanin emit that pre-Graph producer for every task reading it. No edge
// in the Definition may name it — the shell was submitted through the ordinary path
// against the same boundary, so its own fanin orders the whole body behind it.
TEST_F(HbgGraphRecordingBoundsTest, PreGraphProducerOfABoundaryTensorContributesNoEdge) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor external = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    // The tensor the Graph takes as its boundary is this task's write target, so it
    // carries that task's id as its owner.
    CoreTaskArgs producer_args;
    producer_args.add_inout(external);
    const TaskId producer_id = orch.submit_dummy_task(producer_args).task_id();
    ASSERT_TRUE(producer_id.is_valid());
    simpler::hbg::Tensor boundary = external;
    boundary.owner_task_id = producer_id;

    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x6B0D5A1F, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
    const simpler::hbg::Tensor &param = graph.params->tensor(0).ref();

    CoreTaskArgs task_args;
    task_args.add_inout(param);
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    const GraphHostDefinitionList published = graph_host_definitions(*graph_state);
    ASSERT_EQ(published.entries.size(), 1u);
    const GraphHostDefinition &entry = published.entries.front();
    ASSERT_NE(entry.object_offset, GRAPH_NO_OBJECT_OFFSET) << "the staging arena is sized to hold this image";
    const auto *definition = reinterpret_cast<const GraphDefinition *>(
        definition_staging.data() + entry.object_offset + sizeof(GraphDefinitionHeader)
    );
    EXPECT_EQ(definition->task_count, 1u);
    EXPECT_EQ(definition->edge_count, 0u) << "the pre-Graph producer is reached through the shell, not through an "
                                             "edge the Definition carries";
    EXPECT_EQ(definition->root_count, 1u);
}

// Two parameters over one buffer are one alias partition and must land in one
// parameter window. The shadow tensor map infers WAR/WAW edges by buffer address, so
// giving them separate windows would tell the recorder they are unrelated memory and
// the edge between a write through one and a read through the other would vanish --
// silently, into a Definition that replays a DAG the body never had.
TEST_F(HbgGraphRecordingBoundsTest, TwoViewsOfOneBufferShareAWindowAndKeepTheEdgeBetweenThem) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor written = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    simpler::hbg::Tensor read = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_inout(written);
    boundary_args.add_input(read);
    const GraphScopeResult graph = orch.graph_begin(0x6B0D5A20, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    const simpler::hbg::Tensor &write_param = graph.params->tensor(0).ref();
    const simpler::hbg::Tensor &read_param = graph.params->tensor(1).ref();
    EXPECT_EQ(write_param.buffer.addr, read_param.buffer.addr) << "parameters over one buffer must share a window base";
    EXPECT_NE(write_param.buffer.addr, written.buffer.addr)
        << "a parameter is addressed in the recording's space, not the caller's";

    CoreTaskArgs writer_args;
    writer_args.add_inout(write_param);
    ASSERT_TRUE(orch.submit_dummy_task(writer_args).task_id().is_valid());

    CoreTaskArgs reader_args;
    reader_args.add_input(read_param);
    ASSERT_TRUE(orch.submit_dummy_task(reader_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    const GraphHostDefinitionList published = graph_host_definitions(*graph_state);
    ASSERT_EQ(published.entries.size(), 1u);
    const GraphHostDefinition &entry = published.entries.front();
    ASSERT_NE(entry.object_offset, GRAPH_NO_OBJECT_OFFSET);
    const auto *definition = reinterpret_cast<const GraphDefinition *>(
        definition_staging.data() + entry.object_offset + sizeof(GraphDefinitionHeader)
    );
    EXPECT_EQ(definition->task_count, 2u);
    EXPECT_EQ(definition->edge_count, 1u) << "the reader depends on the writer through the buffer they share";
}
