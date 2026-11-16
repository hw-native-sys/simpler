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
 * The sub-task twin of the early-dispatch qualification contract. A body is
 * recorded once per shape, so its verdicts are decided at Definition build time
 * rather than per submit:
 *
 *   ED_FLAG_CANDIDATE  every internal producer carries allow_early_resolve, no
 *                      dispatch predicate, dispatchable shape, internal fanin >= 1
 *   ED_FLAG_TRACKED    at least one candidate names this task as a producer
 *
 * and a candidate's fanin CSR row is sorted by ascending producer index. The
 * builder emits a producer before its consumers, so that order puts the deepest
 * producer at the row's tail, which is where the device's backward scan starts.
 * Non-candidate rows keep their record order.
 *
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"
#include "host_build_graph/task_id.h"

class HbgGraphEdQualificationTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    GraphHostStatePtr graph_state;
    std::vector<char> gm_heap;
    std::vector<std::byte> definition_staging;
    std::vector<TensorCreateInfo> create_infos;
    std::array<uint32_t, 16> boundary_storage{};

    static constexpr size_t HEAP_BYTES = 8 * 1024 * 1024;
    static constexpr size_t STAGING_BYTES = 256 * 1024;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES);
        create_infos.reserve(16);

        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), HEAP_BYTES, CHIP_DEFAULT_GRAPH_TASKS));

        definition_staging.assign(STAGING_BYTES, std::byte{0});
        GraphImageArena arena{};
        arena.base = definition_staging.data();
        arena.capacity = definition_staging.size();
        arena.object_prefix_bytes = sizeof(GraphImageHeader);
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

    simpler::hbg::Tensor boundary_tensor() {
        uint32_t shape[] = {static_cast<uint32_t>(boundary_storage.size())};
        return simpler::hbg::make_tensor_external(boundary_storage.data(), shape, 1);
    }

    // Opens a recording over one boundary tensor. Returns the parameter the body's tasks
    // read from -- the entry's own copy, which carries a recording-space address and PARAM
    // provenance, rather than the caller's tensor the recording was opened with.
    // `boundary` must outlive every use of `boundary_args`: GraphTaskArgs stores a
    // Tensor pointer, so a tensor built inside this helper would leave the caller
    // holding a dangling one.
    simpler::hbg::Tensor begin_body(uint64_t key, GraphTaskArgs &boundary_args, const simpler::hbg::Tensor &boundary) {
        orch.begin_scope();
        boundary_args.add_input(boundary);
        const GraphScopeResult graph = orch.graph_begin(key, boundary_args);
        EXPECT_TRUE(graph.recording);
        EXPECT_NE(graph.recording_handle, nullptr);
        EXPECT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
        return graph.params->tensor(0).ref();
    }

    // One AIV task reading `input` and writing its own output; `flagged` sets
    // allow_early_resolve. Returns the output, which is how a later task in the
    // body names this one as its producer.
    simpler::hbg::Tensor record_task(const simpler::hbg::Tensor &input, bool flagged) {
        uint32_t shape[] = {16};
        CoreTaskArgs args;
        args.add_input(input);
        create_infos.emplace_back(shape, 1, DataType::FLOAT32);
        args.add_output(create_infos.back());
        args.set_allow_early_resolve(flagged);
        MixedKernels mixed{};
        mixed.aiv0_kernel_id = 0;
        TaskOutputTensors out = orch.submit_task(mixed, args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.get_ref(0);
    }

    // One allocation recorded inside the body: a kernel-less producer whose
    // output is ready at creation. Returns the buffer, which is how a later task
    // names this allocation as its producer.
    simpler::hbg::Tensor record_alloc() {
        uint32_t shape[] = {16};
        CoreTaskArgs args;
        create_infos.emplace_back(shape, 1, DataType::FLOAT32);
        args.add_output(create_infos.back());
        TaskOutputTensors out = orch.alloc_tensors(args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.get_ref(0);
    }

    // One AIV task reading both producers' outputs, in the order given.
    TaskId record_consumer(const simpler::hbg::Tensor &first, const simpler::hbg::Tensor &second) {
        uint32_t shape[] = {16};
        CoreTaskArgs args;
        args.add_input(first, second);
        create_infos.emplace_back(shape, 1, DataType::FLOAT32);
        args.add_output(create_infos.back());
        MixedKernels mixed{};
        mixed.aiv0_kernel_id = 0;
        TaskOutputTensors out = orch.submit_task(mixed, args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.task_id();
    }

    // One top-level task writing the boundary the body reads, so a shell
    // submitted for that body names it as a producer. `flagged` sets
    // allow_early_resolve, the term the shell's conjunction turns on.
    TaskId submit_boundary_producer(const simpler::hbg::Tensor &boundary, bool flagged) {
        CoreTaskArgs args;
        args.add_output(boundary);
        args.set_allow_early_resolve(flagged);
        MixedKernels mixed{};
        mixed.aiv0_kernel_id = 0;
        TaskOutputTensors out = orch.submit_task(mixed, args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.task_id();
    }

    // Replays a recorded body by its key. The cache hit is what submits the
    // outer shell, which is where the shell's own verdict is decided.
    TaskId submit_shell(uint64_t key, GraphTaskArgs &boundary_args) {
        const GraphScopeResult replay = orch.graph_begin(key, boundary_args);
        EXPECT_FALSE(replay.recording) << "the body is already recorded, so this must be a cache hit";
        EXPECT_TRUE(replay.task_id.is_valid());
        return replay.task_id;
    }

    const ChipTaskSlotState &slot_of(TaskId id) {
        return orch.sm_header->tasks.get_slot_state_by_task_id(id.local_id());
    }

    const Graph *published_definition() {
        const GraphHostImageList published = graph_host_definitions(*graph_state);
        EXPECT_EQ(published.entries.size(), 1u);
        if (published.entries.empty()) return nullptr;
        const GraphHostImage &entry = published.entries.front();
        EXPECT_NE(entry.object_offset, GRAPH_NO_OBJECT_OFFSET) << "the staging arena is sized to hold this image";
        return reinterpret_cast<const Graph *>(
            definition_staging.data() + entry.object_offset + sizeof(GraphImageHeader)
        );
    }
};

TEST_F(HbgGraphEdQualificationTest, AllFlaggedProducersMakeCandidateAndSortItsRow) {
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor caller = boundary_tensor();
    const simpler::hbg::Tensor boundary = begin_body(0x6ED0A001, boundary_args, caller);

    const simpler::hbg::Tensor t0 = record_task(boundary, /*flagged=*/true);
    const simpler::hbg::Tensor t1 = record_task(boundary, /*flagged=*/true);
    // Producers named in reverse record order: the sorted row must not depend on
    // the order the consumer's operands were declared in.
    ASSERT_TRUE(record_consumer(t1, t0).is_valid());
    ASSERT_TRUE(orch.graph_end());

    const Graph *definition = published_definition();
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 3);
    const auto *tasks = graph_definition_array<SubTaskDefinition>(*definition, definition->off_sub_tasks, 3);
    const auto *fanin_offsets = graph_definition_array<int32_t>(*definition, definition->off_fanin_offsets, 4);
    const auto *fanin_indices =
        graph_definition_array<uint16_t>(*definition, definition->off_fanin_indices, definition->edge_count);
    ASSERT_NE(tasks, nullptr);
    ASSERT_NE(fanin_offsets, nullptr);
    ASSERT_NE(fanin_indices, nullptr);

    EXPECT_NE(tasks[2].ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_NE(tasks[0].ed_flags & ED_FLAG_TRACKED, 0);
    EXPECT_NE(tasks[1].ed_flags & ED_FLAG_TRACKED, 0);
    // A body root's real gate is the outer shell's activation, not this CSR.
    EXPECT_EQ(tasks[0].ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_EQ(tasks[1].ed_flags & ED_FLAG_CANDIDATE, 0);
    // No sub-task reaches the device carrying the early-resolve bit.
    EXPECT_EQ(TaskAttrs{tasks[0].task_attrs}.allow_early_resolve(), false);

    ASSERT_EQ(fanin_offsets[3] - fanin_offsets[2], 2);
    EXPECT_EQ(fanin_indices[fanin_offsets[2]], 0);
    EXPECT_EQ(fanin_indices[fanin_offsets[2] + 1], 1);
}

TEST_F(HbgGraphEdQualificationTest, OneUnflaggedProducerDisqualifiesAndLeavesItsRowInRecordOrder) {
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor caller = boundary_tensor();
    const simpler::hbg::Tensor boundary = begin_body(0x6ED0A002, boundary_args, caller);

    const simpler::hbg::Tensor t0 = record_task(boundary, /*flagged=*/true);
    const simpler::hbg::Tensor t1 = record_task(boundary, /*flagged=*/false);
    ASSERT_TRUE(record_consumer(t1, t0).is_valid());
    ASSERT_TRUE(orch.graph_end());

    const Graph *definition = published_definition();
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 3);
    const auto *tasks = graph_definition_array<SubTaskDefinition>(*definition, definition->off_sub_tasks, 3);
    const auto *fanin_offsets = graph_definition_array<int32_t>(*definition, definition->off_fanin_offsets, 4);
    const auto *fanin_indices =
        graph_definition_array<uint16_t>(*definition, definition->off_fanin_indices, definition->edge_count);
    ASSERT_NE(tasks, nullptr);
    ASSERT_NE(fanin_offsets, nullptr);
    ASSERT_NE(fanin_indices, nullptr);

    EXPECT_EQ(tasks[2].ed_flags & ED_FLAG_CANDIDATE, 0);
    // No candidate, so neither producer is tracked.
    EXPECT_EQ(tasks[0].ed_flags & ED_FLAG_TRACKED, 0);
    EXPECT_EQ(tasks[1].ed_flags & ED_FLAG_TRACKED, 0);

    // Record order (here producer 1 before producer 0), so this fails if a
    // non-candidate row were sorted too.
    ASSERT_EQ(fanin_offsets[3] - fanin_offsets[2], 2);
    EXPECT_EQ(fanin_indices[fanin_offsets[2]], 1);
    EXPECT_EQ(fanin_indices[fanin_offsets[2] + 1], 0);
}

// An allocation is a transparent producer, not a barrier: its output is ready at
// creation, so it must not suppress a consumer's early dispatch. The top-level
// submit path holds the same contract by marking the alloc slot; a body holds it
// through the recorded attrs.
TEST_F(HbgGraphEdQualificationTest, HiddenAllocProducerDoesNotDisqualifyItsConsumer) {
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor caller = boundary_tensor();
    const simpler::hbg::Tensor boundary = begin_body(0x6ED0A003, boundary_args, caller);

    const simpler::hbg::Tensor flagged = record_task(boundary, /*flagged=*/true);
    const simpler::hbg::Tensor allocated = record_alloc();
    ASSERT_TRUE(record_consumer(flagged, allocated).is_valid());
    ASSERT_TRUE(orch.graph_end());

    const Graph *definition = published_definition();
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 3);
    const auto *tasks = graph_definition_array<SubTaskDefinition>(*definition, definition->off_sub_tasks, 3);
    const auto *fanin_offsets = graph_definition_array<int32_t>(*definition, definition->off_fanin_offsets, 4);
    ASSERT_NE(tasks, nullptr);
    ASSERT_NE(fanin_offsets, nullptr);

    // Both producers reach the consumer, and the allocation among them does not
    // cost it the verdict.
    ASSERT_EQ(fanin_offsets[3] - fanin_offsets[2], 2);
    EXPECT_NE(tasks[2].ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_NE(tasks[0].ed_flags & ED_FLAG_TRACKED, 0);
    EXPECT_NE(tasks[1].ed_flags & ED_FLAG_TRACKED, 0);
}

// The outer shell's own verdict, decided per submit against its inline row of
// GLOBAL producers rather than against the body's CSR. A shell occupies no core,
// so its conjunction is the top-level one minus the dispatch-shape terms: it
// qualifies on producers alone. What its release does instead is admit the
// body's roots, which is why the verdict below is also what decides whether
// materialization may flag any of them.
TEST_F(HbgGraphEdQualificationTest, FlaggedProducerMakesTheShellACandidate) {
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor caller = boundary_tensor();
    const simpler::hbg::Tensor param = begin_body(0x6ED0B001, boundary_args, caller);
    record_task(param, /*flagged=*/true);
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();

    // Outside the Graph the caller's own tensor is what an ordinary task takes: a
    // parameter's owner is PARAM, which names no task of the run, so passing one here
    // would ask the fanin walk to resolve it against the task table.
    const TaskId producer = submit_boundary_producer(boundary_tensor(), /*flagged=*/true);
    const TaskId shell = submit_shell(0x6ED0B001, boundary_args);

    EXPECT_EQ(slot_of(shell).task_kind, TaskKind::GRAPH);
    EXPECT_NE(slot_of(shell).ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_NE(slot_of(producer).ed_flags & ED_FLAG_TRACKED, 0);
}

TEST_F(HbgGraphEdQualificationTest, OneUnflaggedProducerDisqualifiesTheShell) {
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor caller = boundary_tensor();
    const simpler::hbg::Tensor param = begin_body(0x6ED0B002, boundary_args, caller);
    record_task(param, /*flagged=*/true);
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();

    const TaskId producer = submit_boundary_producer(boundary_tensor(), /*flagged=*/false);
    const TaskId shell = submit_shell(0x6ED0B002, boundary_args);

    EXPECT_EQ(slot_of(shell).ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_EQ(slot_of(producer).ed_flags & ED_FLAG_TRACKED, 0);
}
