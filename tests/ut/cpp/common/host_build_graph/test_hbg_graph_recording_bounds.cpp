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
 * and that map holds SUB_TASK_MAX_NUM task chains. A SUB_TASK id's low field is the
 * task's index within its body, so the key is in range however far into a run the
 * Graph begins — which a GLOBAL id carrying the allocator's own local id would not
 * be.
 *
 * These tests place the Graph past that many ordinary tasks, where an id drawn from
 * the run's numbering would key the map outside its chains.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "graph_submit.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"
#include "host_build_graph/task_id.h"

class HbgGraphRecordingBoundsTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    GraphHostStatePtr graph_state;
    std::vector<char> gm_heap;
    std::vector<std::byte> definition_staging;

    static constexpr size_t HEAP_BYTES = 8 * 1024 * 1024;
    static constexpr size_t STAGING_BYTES = 256 * 1024;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES);

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
};

// A Graph recorded after SUB_TASK_MAX_NUM ordinary tasks. Its recorded task registers
// its outputs in the recording hazard map keyed on its own id, so that id decides
// whether the key lands inside the map's task chains.
TEST_F(HbgGraphRecordingBoundsTest, RecordedTaskIsKeyedByItsIndexNotByTheRunsNumbering) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    // Move the allocator's local-id counter past the recording map's task-chain
    // count, the way any run that submits a while before its first Graph does.
    for (uint32_t i = 0; i < SUB_TASK_MAX_NUM; ++i) {
        CoreTaskArgs filler;
        filler.add_input(boundary);
        ASSERT_TRUE(orch.submit_dummy_task(filler).task_id().is_valid()) << "filler task " << i;
    }
    ASSERT_EQ(orch.task_allocator.active_count(), SUB_TASK_MAX_NUM);

    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x6B0D5A1E, boundary_args);
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
    const TaskId sub_task_id = orch.submit_dummy_task(task_args).task_id();
    ASSERT_TRUE(sub_task_id.is_valid());
    EXPECT_EQ(sub_task_id.space(), TaskId::Space::SUB_TASK)
        << "a recorded task must not take a GLOBAL id: nothing resolves it against the task table, and its low "
           "field is what keys the recording's hazard map";
    EXPECT_EQ(
        sub_task_id.local_id(), 0
    ) << "the first recorded task's low field is task index 0, independent of how many tasks the run has "
         "already allocated";
    EXPECT_LT(sub_task_id.local_id(), SUB_TASK_MAX_NUM);

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
    const GraphScopeResult graph = orch.graph_begin(0x6B0D5A1F, boundary_args);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
    const simpler::hbg::Tensor &param = graph.params->tensor(0).ref();

    CoreTaskArgs task_args;
    task_args.add_inout(param);
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    const GraphHostImageList published = graph_host_definitions(*graph_state);
    ASSERT_EQ(published.entries.size(), 1u);
    const GraphHostImage &entry = published.entries.front();
    ASSERT_NE(entry.object_offset, GRAPH_NO_OBJECT_OFFSET) << "the staging arena is sized to hold this image";
    const auto *definition =
        reinterpret_cast<const Graph *>(definition_staging.data() + entry.object_offset + sizeof(GraphImageHeader));
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
    const GraphScopeResult graph = orch.graph_begin(0x6B0D5A20, boundary_args);
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

    const GraphHostImageList published = graph_host_definitions(*graph_state);
    ASSERT_EQ(published.entries.size(), 1u);
    const GraphHostImage &entry = published.entries.front();
    ASSERT_NE(entry.object_offset, GRAPH_NO_OBJECT_OFFSET);
    const auto *definition =
        reinterpret_cast<const Graph *>(definition_staging.data() + entry.object_offset + sizeof(GraphImageHeader));
    EXPECT_EQ(definition->task_count, 2u);
    EXPECT_EQ(definition->edge_count, 1u) << "the reader depends on the writer through the buffer they share";
}

// A boundary of no tensors records and publishes. This is the shape a Config-driven Graph has:
// the body reaches its buffers through handles its Config carries, so nothing crosses the
// boundary as a tensor and the boundary holds scalars alone.
//
// The whole path, through the real recorder rather than a hand-built image: the admission
// `rt_graph_args_cacheable` decides, then graph_prepare, a recorded sub-task, graph_end, and the
// published Definition. Entering at replay would prove nothing here — the two ends disagreeing
// about a zero-tensor boundary is exactly the defect this pins, and a submission admitted at one
// end and refused at the other records a whole body before failing at graph_end.
TEST_F(HbgGraphRecordingBoundsTest, ABoundaryOfNoTensorsRecordsAndPublishes) {
    GraphTaskArgs boundary_args;
    boundary_args.add_scalar(uint64_t{17});
    ASSERT_EQ(boundary_args.tensor_count(), 0);
    ASSERT_TRUE(rt_graph_args_cacheable(boundary_args)) << "the submission end admits a boundary of no tensors";

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x20E50A11, boundary_args);
    ASSERT_FALSE(graph.has_error) << "a boundary of no tensors opens a Graph";
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    // The body's own tensor, allocated inside the Graph. With no boundary tensor there is
    // nothing for it to inherit, which is the case a Config-driven body is in: every tensor it
    // touches it produces itself, and the buffers its Config names reach the kernel as scalars
    // rather than as boundary parameters.
    uint32_t shape[] = {16};
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    CoreTaskArgs task_args;
    task_args.add_output(recorded_output);
    task_args.add_scalar(graph.params->scalar(0));
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());

    ASSERT_TRUE(orch.graph_end()) << "the recording end must publish what the submission end admitted";

    const GraphHostImageList published = graph_host_definitions(*graph_state);
    ASSERT_EQ(published.entries.size(), 1u) << "a zero-tensor boundary yields a Definition like any other";
    const GraphHostImage &entry = published.entries.front();
    ASSERT_NE(entry.object_offset, GRAPH_NO_OBJECT_OFFSET);
    const auto *definition =
        reinterpret_cast<const Graph *>(definition_staging.data() + entry.object_offset + sizeof(GraphImageHeader));
    EXPECT_EQ(definition->boundary_tensor_count, 0);
    EXPECT_EQ(definition->boundary_scalar_count, 1);
    EXPECT_EQ(definition->task_count, 1u);
}

// A manual-scope producer has no tensor-map output edge. The Graph shell must
// preserve its explicit predecessor on the first recording, an in-flight reuse,
// and a published cache hit. Those invocation edges never enter the Definition.
TEST_F(HbgGraphRecordingBoundsTest, ExplicitPredecessorsBelongToEachOuterShell) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    auto boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    orch.begin_scope(ScopeMode::MANUAL);
    std::array<TaskId, 3> producers{};
    for (auto &producer : producers) {
        CoreTaskArgs args;
        producer = orch.submit_dummy_task(args).task_id();
        ASSERT_TRUE(producer.is_valid());
    }
    const uint64_t key = 0x20E50A13;
    const std::array<TaskId, 2> repeated_dep{producers[0], producers[0]};
    GraphTaskArgs first;
    first.add_input(boundary);
    first.set_dependencies(repeated_dep.data(), repeated_dep.size());
    const auto cold = orch.graph_begin(key, first);
    ASSERT_FALSE(cold.has_error);
    ASSERT_TRUE(cold.recording);
    ASSERT_TRUE(cold.task_id.is_valid()) << "cold recording already owns an outer shell";
    ASSERT_NE(cold.params, nullptr);
    EXPECT_EQ(cold.params->explicit_dep_count(), 0U)
        << "the recording body must not retain this invocation's GLOBAL dependencies";

    GraphTaskArgs second;
    second.add_input(boundary);
    second.set_dependencies(&producers[1], 1);
    const auto inflight = orch.graph_begin(key, second);
    ASSERT_FALSE(inflight.has_error);
    EXPECT_FALSE(inflight.recording);
    ASSERT_TRUE(inflight.task_id.is_valid());

    ASSERT_TRUE(orch.graph_prepare(cold.recording_handle, first));
    CoreTaskArgs inner;
    inner.add_input(cold.params->tensor(0).ref());
    ASSERT_TRUE(orch.submit_dummy_task(inner).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    GraphTaskArgs third;
    third.add_input(boundary);
    third.set_dependencies(producers.data() + 1, 2);
    const auto cached = orch.graph_begin(key, third);
    ASSERT_FALSE(cached.has_error);
    EXPECT_FALSE(cached.recording);
    ASSERT_TRUE(cached.task_id.is_valid());

    const std::array<TaskId, 3> shells{cold.task_id, inflight.task_id, cached.task_id};
    const std::array<std::vector<int32_t>, 3> want{
        std::vector<int32_t>{static_cast<int32_t>(producers[0].local_id())},
        std::vector<int32_t>{static_cast<int32_t>(producers[1].local_id())},
        std::vector<int32_t>{
            static_cast<int32_t>(producers[1].local_id()), static_cast<int32_t>(producers[2].local_id())
        }
    };
    for (size_t i = 0; i < shells.size(); ++i) {
        const auto &payload = orch.sm_header->tasks.storage_at(shells[i].local_id()).payload;
        std::vector<int32_t> actual(payload.fanin_data(), payload.fanin_data() + payload.fanin_count);
        std::sort(actual.begin(), actual.end());
        EXPECT_EQ(actual, want[i]) << "shell " << i;
    }
    const auto published = graph_host_definitions(*graph_state);
    ASSERT_EQ(published.entries.size(), 1U);
    const auto &entry = published.entries.front();
    const auto *definition =
        reinterpret_cast<const Graph *>(definition_staging.data() + entry.object_offset + sizeof(GraphImageHeader));
    EXPECT_EQ(definition->task_count, 1U);
    EXPECT_EQ(definition->edge_count, 0U);
}

class HbgGraphExplicitDependencyRejectionTest : public HbgGraphRecordingBoundsTest,
                                                public ::testing::WithParamInterface<int> {};

TEST_P(HbgGraphExplicitDependencyRejectionTest, RejectsSelfAndFuturePredecessors) {
    const int phase = GetParam() / 2;  // cold, in-flight, cached
    const int future_offset = GetParam() % 2;
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    auto boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs initial;
    initial.add_input(boundary);
    orch.begin_scope();
    constexpr uint64_t key = 0x20E50A14;
    if (phase != 0) {
        const auto cold = orch.graph_begin(key, initial);
        ASSERT_TRUE(cold.recording);
        ASSERT_FALSE(cold.has_error);
        if (phase == 2) {
            ASSERT_TRUE(orch.graph_prepare(cold.recording_handle, initial));
            CoreTaskArgs inner;
            inner.add_input(cold.params->tensor(0).ref());
            ASSERT_TRUE(orch.submit_dummy_task(inner).task_id().is_valid());
            ASSERT_TRUE(orch.graph_end());
        }
    }
    // The next shell is claimed before explicit fanins are appended. That must
    // not turn its predicted id into a valid predecessor of itself.
    const auto invalid_predecessor = TaskId::make_global(orch.task_allocator.active_count() + future_offset);
    GraphTaskArgs invocation;
    invocation.add_input(boundary);
    invocation.set_dependencies(&invalid_predecessor, 1);
    const auto result = orch.graph_begin(key, invocation);
    EXPECT_TRUE(result.has_error);
    EXPECT_FALSE(result.task_id.is_valid());
    EXPECT_TRUE(orch.is_fatal());
    EXPECT_EQ(orch.fatal_code.load(), SIMPLER_ERROR_INVALID_ARGS);
}

INSTANTIATE_TEST_SUITE_P(ColdInflightCached, HbgGraphExplicitDependencyRejectionTest, ::testing::Range(0, 6));
TEST_F(HbgGraphRecordingBoundsTest, FallbackScratchIsReusedAndDuplicateAbortIsHarmless) {
    GraphTaskArgs args;
    orch.begin_scope();
    const auto first = orch.graph_begin(0xFA110001, args);
    ASSERT_TRUE(first.recording);
    ASSERT_TRUE(orch.graph_prepare(first.recording_handle, args));
    auto *scratch = static_cast<GraphInflightRecording *>(first.recording_handle)->fallback_scratch.get();
    ASSERT_NE(scratch, nullptr);
    const auto *tensors = scratch->task_tensor_pool.get();
    orch.graph_abort(first.recording_handle);
    EXPECT_EQ(scratch->boundary, nullptr);
    orch.graph_abort(first.recording_handle);
    EXPECT_EQ(active_graph_recording(&orch), nullptr);
    const auto second = orch.graph_begin(0xFA110002, args);
    ASSERT_TRUE(second.recording);
    ASSERT_TRUE(orch.graph_prepare(second.recording_handle, args));
    EXPECT_EQ(static_cast<GraphInflightRecording *>(second.recording_handle)->fallback_scratch.get(), scratch);
    EXPECT_EQ(scratch->task_tensor_pool.get(), tensors);
    // Retiring an old handle must not clear the new recording's identity.
    orch.graph_abort(first.recording_handle);
    EXPECT_EQ(active_graph_recording(&orch), scratch);
    orch.graph_abort(second.recording_handle);
    orch.end_scope();
}

#if defined(__linux__)
#include <cerrno>
#include <pthread.h>
namespace {
int fail_recording_bind_at = 0;
}
extern "C" int __real_pthread_setspecific(pthread_key_t, const void *);
extern "C" int __wrap_pthread_setspecific(pthread_key_t key, const void *value) {
    if (fail_recording_bind_at != 0 && --fail_recording_bind_at == 0) return ENOMEM;
    return __real_pthread_setspecific(key, value);
}
TEST_F(HbgGraphRecordingBoundsTest, FailedIdentityBindingClearsBorrowedBoundary) {
    GraphRecording scratch;
    ScopedRecorderScratchBinding storage(scratch);
    GraphTaskArgs args;
    orch.begin_scope();
    for (int word : {1, 2, 3}) {
        const auto graph = orch.graph_begin(0xFA120000 + word, args);
        ASSERT_TRUE(graph.recording);
        fail_recording_bind_at = word;
        EXPECT_FALSE(orch.graph_prepare(graph.recording_handle, args));
        EXPECT_EQ(fail_recording_bind_at, 0);
        EXPECT_EQ(scratch.boundary, nullptr);
        EXPECT_EQ(active_graph_recording(&orch), nullptr);
        orch.graph_abort(graph.recording_handle);
    }
    orch.end_scope();
}
#endif
