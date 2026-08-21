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

#include <array>
#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "graph_host_state.h"
#include "pto_orchestrator.h"
#include "pto_shared_memory.h"
#include "task_interface/assert_compat.h"
#include "utils/device_arena.h"

class HbgGraphSubmitFailureTest : public ::testing::Test {
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

    // A Graph task's heap allocation covers its nodes' packed outputs *and* the
    // execution storage the device materializes into, so the pool has to hold a
    // GraphExecution header plus one GraphNodeStorage (~5 KB) on top of the
    // outputs. 4 KB used to be enough when the storage came from a separate
    // device allocation.
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
};

TEST_F(HbgGraphSubmitFailureTest, InFlightGraphInvocationsReserveHeapOnlyAtCommit) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult first = orch.graph_begin(0x1715, boundary_args, 0x1736);
    ASSERT_TRUE(first.recording);
    ASSERT_TRUE(first.task_id.is_valid());
    const GraphScopeResult second = orch.graph_begin(0x1715, boundary_args, 0x1736);
    EXPECT_FALSE(second.recording);
    EXPECT_FALSE(second.execute_block);
    ASSERT_TRUE(second.task_id.is_valid());
    EXPECT_EQ(second.task_id.local(), first.task_id.local() + 1);
    EXPECT_EQ(orch.ring.task_allocator.heap_top(), 0u);
    EXPECT_EQ(graph_host_upload_count(*graph_state), 2u);

    ASSERT_TRUE(orch.graph_prepare(first.recording_handle, boundary_args));
    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    node_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    EXPECT_EQ(orch.ring.task_allocator.heap_top(), 0u);

    orch.graph_commit();
    EXPECT_FALSE(orch.fatal);
    EXPECT_GT(orch.ring.task_allocator.heap_top(), 0u);
    const std::optional<GraphHostUpload> first_upload = graph_host_upload(*graph_state, 0);
    const std::optional<GraphHostUpload> second_upload = graph_host_upload(*graph_state, 1);
    ASSERT_TRUE(first_upload.has_value());
    ASSERT_TRUE(second_upload.has_value());
    EXPECT_NE(first_upload->definition_hash, 0u);
    EXPECT_EQ(second_upload->definition_hash, first_upload->definition_hash);
    // Distinct bases alone would still pass if finalization handed out a wrong
    // extent, so pin the length the Definition asks for and the disjointness two
    // shells of one Graph must have.
    const auto *first_base = static_cast<const char *>(first_upload->outer_slot->task->packed_buffer_base);
    const auto *first_end = static_cast<const char *>(first_upload->outer_slot->task->packed_buffer_end);
    const auto *second_base = static_cast<const char *>(second_upload->outer_slot->task->packed_buffer_base);
    const auto *second_end = static_cast<const char *>(second_upload->outer_slot->task->packed_buffer_end);
    const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
    ASSERT_EQ(definitions.entries.size(), 1u);
    ASSERT_EQ(definitions.entries[0].full_key, first_upload->full_key);
    const auto *definition = reinterpret_cast<const GraphDefinition *>(definitions.entries[0].data);
    const uint64_t expected_extent =
        PTO2_ALIGN_UP(definition->required_heap + definition->execution_storage_bytes, PTO2_ALIGN_SIZE);
    EXPECT_EQ(static_cast<uint64_t>(first_end - first_base), expected_extent);
    EXPECT_EQ(static_cast<uint64_t>(second_end - second_base), expected_extent);
    EXPECT_TRUE(first_end <= second_base || second_end <= first_base) << "two shells must not share heap bytes";
}

// The one combination the other two tests miss: real orchestrator state driven
// by two real threads. test_hbg_graph_async_submit exercises the worker handoff
// against a fake ops table, and every case here otherwise calls prepare/record/
// end on the test thread, so nothing covers a worker recording *while* the main
// thread submits same-hash shells.
//
// That overlap is held together only by field partitioning: under
// recording_mutex the main thread reads boundary_tensors / boundary_types /
// boundary_scalar_count, while the worker writes boundary_args / nodes /
// next_virtual_offset / unsupported without it (graph_prepare skips the mutex on
// purpose, so a submit burst cannot starve it). Nothing enforces that split, so
// this pins the functional contract that depends on it — and gives TSAN a window
// to report the split being broken.
//
// The handshake is deterministic rather than timing-based: the worker is proven
// to be between graph_prepare and graph_end while the main thread runs its
// in-flight graph_begin calls.
TEST_F(HbgGraphSubmitFailureTest, WorkerRecordsWhileMainThreadSubmitsSameHashShells) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult first = orch.graph_begin(0x171a, boundary_args, 0x1736);
    ASSERT_TRUE(first.recording);
    ASSERT_TRUE(first.task_id.is_valid());

    std::mutex gate_mutex;
    std::condition_variable gate_cv;
    bool prepared = false;
    bool main_done_submitting = false;
    bool prepare_ok = false;
    bool node_ok = false;
    bool end_ok = false;

    std::thread worker([&]() {
        // Worker-owned boundary copy, alive until graph_end: graph_prepare
        // anchors scalar sources into it and stores its address.
        GraphTaskArgs worker_args;
        worker_args.add_input(boundary);
        prepare_ok = orch.graph_prepare(first.recording_handle, worker_args);
        {
            std::lock_guard<std::mutex> lock(gate_mutex);
            prepared = true;
        }
        gate_cv.notify_all();
        if (!prepare_ok) return;

        {
            std::unique_lock<std::mutex> lock(gate_mutex);
            gate_cv.wait(lock, [&]() {
                return main_done_submitting;
            });
        }

        CoreTaskArgs node_args;
        node_args.add_input(boundary);
        TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
        node_args.add_output(recorded_output);
        node_ok = orch.submit_dummy_task(node_args).task_id().is_valid();
        end_ok = orch.graph_end();
    });

    {
        std::unique_lock<std::mutex> lock(gate_mutex);
        gate_cv.wait(lock, [&]() {
            return prepared;
        });
    }

    // The worker is now inside the recording. These two go through the in-flight
    // branch, which reads the boundary signature under recording_mutex.
    const GraphScopeResult second = orch.graph_begin(0x171a, boundary_args, 0x1736);
    const GraphScopeResult third = orch.graph_begin(0x171a, boundary_args, 0x1736);
    {
        std::lock_guard<std::mutex> lock(gate_mutex);
        main_done_submitting = true;
    }
    gate_cv.notify_all();
    worker.join();

    ASSERT_TRUE(prepare_ok);
    ASSERT_TRUE(node_ok);
    ASSERT_TRUE(end_ok);
    EXPECT_FALSE(second.recording);
    EXPECT_FALSE(second.execute_block);
    EXPECT_FALSE(third.execute_block);
    ASSERT_TRUE(second.task_id.is_valid());
    ASSERT_TRUE(third.task_id.is_valid());
    EXPECT_EQ(second.task_id.local(), first.task_id.local() + 1);
    EXPECT_EQ(third.task_id.local(), first.task_id.local() + 2);
    EXPECT_EQ(orch.ring.task_allocator.heap_top(), 0u) << "no shell may take heap before commit";

    orch.graph_commit();
    ASSERT_FALSE(orch.fatal);
    ASSERT_EQ(graph_host_upload_count(*graph_state), 3u);

    const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
    ASSERT_EQ(definitions.entries.size(), 1u);
    const auto *definition = reinterpret_cast<const GraphDefinition *>(definitions.entries[0].data);
    const uint64_t expected_extent =
        PTO2_ALIGN_UP(definition->required_heap + definition->execution_storage_bytes, PTO2_ALIGN_SIZE);

    std::vector<std::pair<const char *, const char *>> ranges;
    for (size_t i = 0; i < 3; ++i) {
        const std::optional<GraphHostUpload> upload = graph_host_upload(*graph_state, i);
        ASSERT_TRUE(upload.has_value());
        EXPECT_EQ(upload->definition_hash, definition->content_hash) << "shell " << i;
        const auto *base = static_cast<const char *>(upload->outer_slot->task->packed_buffer_base);
        const auto *end = static_cast<const char *>(upload->outer_slot->task->packed_buffer_end);
        EXPECT_EQ(static_cast<uint64_t>(end - base), expected_extent) << "shell " << i;
        ranges.emplace_back(base, end);
    }
    for (size_t i = 0; i < ranges.size(); ++i) {
        for (size_t j = i + 1; j < ranges.size(); ++j) {
            EXPECT_TRUE(ranges[i].second <= ranges[j].first || ranges[j].second <= ranges[i].first)
                << "shells " << i << " and " << j << " share heap bytes";
        }
    }
}

// An outer Graph shell enters the task and dependency sequence before the
// worker has recorded the body, so a construct the recording cannot represent
// can no longer be answered by re-running the body on the ordinary path — the
// shell's task id and TensorMap producers are already published. Commit
// therefore has to latch a fatal rather than leave a shell that can never
// complete.
TEST_F(HbgGraphSubmitFailureTest, AbortedRecordingLatchesFatalAtCommit) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x1717, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(graph.task_id.is_valid());
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    node_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());

    orch.graph_abort(graph.recording_handle);
    ASSERT_FALSE(orch.fatal) << "Abort alone must not latch; the shell is still finalizable in principle";

    orch.graph_commit();
    EXPECT_TRUE(orch.fatal) << "A shell whose Definition never arrived cannot be completed";
}

// The ordinary path reports PTO2_ERROR_INVALID_ARGS for an auto scope opened
// inside a manual one. The recording pass keeps a scope depth of its own — the
// manual flag has to reach compute_task_fanin, which suppresses inference inside
// a manual scope — so it has to refuse the same nesting. Accepting it would let a
// Graph record and replay a body ordinary submission rejects outright.
TEST_F(HbgGraphSubmitFailureTest, AutoScopeNestedInManualScopeRefusesTheRecording) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x171d, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    orch.begin_scope(PTO2ScopeMode::MANUAL);
    orch.begin_scope(PTO2ScopeMode::AUTO);

    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    node_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());

    orch.end_scope();
    orch.end_scope();

    EXPECT_THROW(orch.graph_end(), AssertionError) << "an auto scope inside a manual one must not publish";
    orch.graph_abort(graph.recording_handle);
    orch.graph_commit();
    EXPECT_TRUE(orch.fatal) << "a shell whose Definition never arrived cannot be completed";
}

// A Graph body may allocate. The allocation records as a kernel-less node, the
// same shape submit_dummy_task records, so the recording stays publishable and
// the commit latches no fatal.
TEST_F(HbgGraphSubmitFailureTest, RuntimeAllocationInsideTheBodyRecordsAKernellessNode) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x1718, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs alloc_args;
    TensorCreateInfo allocated(shape, 1, DataType::UINT32);
    alloc_args.add_output(allocated);
    const TaskOutputTensors outputs = orch.alloc_tensors(alloc_args);
    EXPECT_TRUE(outputs.task_id().is_valid());

    EXPECT_TRUE(orch.graph_end());

    orch.graph_commit();
    EXPECT_FALSE(orch.fatal);
}

TEST_F(HbgGraphSubmitFailureTest, FaninFailureLatchesFatalWithoutPartialUpload) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x1715, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    node_args.add_output(recorded_output);
    const uint64_t heap_top_before_record = orch.ring.task_allocator.heap_top();
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());
    EXPECT_EQ(orch.ring.task_allocator.heap_top(), heap_top_before_record);
    ASSERT_TRUE(orch.graph_end());
    EXPECT_EQ(orch.ring.task_allocator.heap_top(), heap_top_before_record);
    orch.graph_commit();
    EXPECT_GT(orch.ring.task_allocator.heap_top(), heap_top_before_record);
    ASSERT_FALSE(orch.fatal);
    const size_t uploads_before_failure = graph_host_upload_count(*graph_state);

    CoreTaskArgs producer_args;
    producer_args.add_output(boundary);
    for (int32_t i = 0; i < PTO2_MAX_FANIN + 1; ++i) {
        ASSERT_TRUE(orch.submit_dummy_task(producer_args).task_id().is_valid());
    }

    const GraphScopeResult replay = orch.graph_begin(0x1715, boundary_args, 0x1736);

    EXPECT_TRUE(replay.execute_block);
    EXPECT_FALSE(replay.recording);
    EXPECT_FALSE(replay.task_id.is_valid());
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_DEP_POOL_OVERFLOW);
    EXPECT_EQ(graph_host_upload_count(*graph_state), uploads_before_failure);
}

TEST_F(HbgGraphSubmitFailureTest, CachedGraphUsesFinalTaskWindowSlot) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x1716, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    ASSERT_EQ(orch.ring.task_allocator.active_count(), 1);

    PTO2TaskAllocator &allocator = orch.ring.task_allocator;
    while (allocator.active_count() < allocator.window_size() - 1) {
        ASSERT_FALSE(allocator.alloc(0).failed());
    }

    const GraphScopeResult replay = orch.graph_begin(0x1716, boundary_args, 0x1736);

    EXPECT_FALSE(replay.execute_block);
    ASSERT_TRUE(replay.task_id.is_valid());
    EXPECT_EQ(replay.task_id.local(), static_cast<uint32_t>(allocator.window_size() - 1));
    EXPECT_EQ(allocator.active_count(), allocator.window_size());
    EXPECT_EQ(sm_handle->header->ring.fc.current_task_index.load(std::memory_order_acquire), allocator.window_size());
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_NONE);
}

// The constructs a predicate can present that no Definition can express. Each is
// discovered while recording, after the outer shell is already in the task
// sequence, so the contract is the one AbortedRecordingLatchesFatalAtCommit
// states: the recording cannot be published and the commit latches a fatal.
// There is no re-run on the ordinary path to fall back to.
//
// This build keeps assertions enabled, so the unsupported construct surfaces as
// the throwing debug_assert graph_end() fires on its way out. That assert
// precedes graph_end()'s own abort, so the abort has to be issued here instead —
// otherwise this thread's recording stays bound and the next test records into
// it.
class HbgGraphPredicateRejectionTest : public HbgGraphSubmitFailureTest {
protected:
    // Records one predicated node into a fresh Graph and asserts the recording
    // refused it. `build_predicate` receives the boundary tensor.
    template <typename BuildPredicate>
    void expect_recording_refused(uint64_t graph_key, BuildPredicate build_predicate) {
        std::array<uint32_t, 16> storage{};
        uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
        ChipTensor boundary = make_tensor_external(storage.data(), shape, 1, DataType::INT32);
        GraphTaskArgs boundary_args;
        boundary_args.add_input(boundary);

        orch.begin_scope();
        const GraphScopeResult graph = orch.graph_begin(graph_key, boundary_args, 0x1736);
        EXPECT_TRUE(graph.recording);
        EXPECT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

        CoreTaskArgs node_args;
        node_args.add_input(boundary);
        TensorCreateInfo recorded_output(shape, 1, DataType::INT32);
        node_args.add_output(recorded_output);
        MixedKernels mixed{};
        mixed.aiv0_kernel_id = 0;
        node_args.set_predicate(build_predicate(boundary));
        EXPECT_TRUE(orch.submit_task(mixed, node_args).task_id().is_valid());

        EXPECT_THROW(orch.graph_end(), AssertionError) << "an unrecordable predicate must not publish";
        orch.graph_abort(graph.recording_handle);
        orch.graph_commit();
        EXPECT_TRUE(orch.fatal) << "a shell whose Definition never arrived cannot be completed";
    }

    static CoreTaskPredicate predicate_on(const ChipTensor &operand, uint32_t index) {
        CoreTaskPredicate pred;
        pred.operand.tensor = &operand;
        pred.operand.ndims = 1;
        pred.operand.indices[0] = index;
        pred.op = PredicateOp::GT;
        pred.target = 0;
        return pred;
    }
};

TEST_F(HbgGraphPredicateRejectionTest, OperandIndexOutsideTheExtentAbortsTheRecording) {
    // Index 16 on a 16-element operand: the flat offset is one element past the
    // extent, so the address it names belongs to whatever follows the buffer.
    expect_recording_refused(0x2001, [](const ChipTensor &boundary) {
        return predicate_on(boundary, 16);
    });
}

TEST_F(HbgGraphPredicateRejectionTest, OperandOnAnUnclassifiableTensorAbortsTheRecording) {
    // Neither a boundary tensor nor any recorded node's output, so the recorder
    // cannot name a base the replay could rebind against.
    std::array<uint32_t, 16> foreign_storage{};
    uint32_t shape[] = {static_cast<uint32_t>(foreign_storage.size())};
    const ChipTensor foreign = make_tensor_external(foreign_storage.data(), shape, 1, DataType::INT32);
    expect_recording_refused(0x2002, [&foreign](const ChipTensor &) {
        return predicate_on(foreign, 0);
    });
}

// A kernel-less node never dispatches, so submit_dummy_task and alloc_tensors
// drop the caller's predicate exactly as they do on the ordinary path. Recording
// must drop it too: a node whose Definition claimed a predicate its own attribute
// denies is rejected by materialize, on the device, for a value the scheduler was
// never going to read.
TEST_F(HbgGraphPredicateRejectionTest, PredicateOnAKernellessNodeIsNotRecorded) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1, DataType::INT32);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x2003, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    TensorCreateInfo recorded_output(shape, 1, DataType::INT32);
    node_args.add_output(recorded_output);
    // Out of extent, which a recorded predicate would reject — proving the
    // predicate never reached the recorder rather than merely passing its checks.
    node_args.set_predicate(predicate_on(boundary, 16));
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());

    EXPECT_TRUE(orch.graph_end()) << "a dropped predicate must not make the body unrecordable";
    orch.graph_commit();
    EXPECT_FALSE(orch.fatal);
}

// Distinct Graph keys record concurrently. A Definition the run has not seen
// before must open its own recording even while another is in flight — the
// alternative is that it is turned away, replays nothing for the rest of the run,
// and every occurrence of it submits its whole body as ordinary tasks.
TEST_F(HbgGraphSubmitFailureTest, ASecondKeyRecordsAlongsideTheFirst) {
    std::array<uint32_t, 16> storage_a{};
    std::array<uint32_t, 16> storage_b{};
    uint32_t shape[] = {static_cast<uint32_t>(storage_a.size())};
    ChipTensor boundary_a = make_tensor_external(storage_a.data(), shape, 1);
    ChipTensor boundary_b = make_tensor_external(storage_b.data(), shape, 1);
    GraphTaskArgs args_a;
    args_a.add_input(boundary_a);
    GraphTaskArgs args_b;
    args_b.add_input(boundary_b);

    orch.begin_scope();
    const GraphScopeResult first = orch.graph_begin(0x1901, args_a, 0x1736);
    ASSERT_TRUE(first.recording);
    ASSERT_NE(first.recording_handle, nullptr);

    const GraphScopeResult second = orch.graph_begin(0x1902, args_b, 0x1736);
    EXPECT_TRUE(second.recording) << "a distinct key must not be demoted by a busy recorder";
    EXPECT_FALSE(second.execute_block);
    ASSERT_NE(second.recording_handle, nullptr);
    EXPECT_NE(second.recording_handle, first.recording_handle);

    // Record both from this thread; concurrency of the threads is the pool's
    // concern, and interleaving the two recordings is what the runtime must
    // tolerate. Each bind goes through its own handle.
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);

    ASSERT_TRUE(orch.graph_prepare(first.recording_handle, args_a));
    CoreTaskArgs node_a;
    node_a.add_input(boundary_a);
    node_a.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(node_a).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    ASSERT_TRUE(orch.graph_prepare(second.recording_handle, args_b));
    CoreTaskArgs node_b;
    node_b.add_input(boundary_b);
    node_b.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(node_b).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    // One commit drains and back-patches both keys' deferred shells.
    orch.graph_commit();
    EXPECT_FALSE(orch.fatal);

    const GraphScopeResult replay_a = orch.graph_begin(0x1901, args_a, 0x1736);
    EXPECT_FALSE(replay_a.execute_block) << "the first key's Definition must be cached";
    EXPECT_FALSE(replay_a.recording);
    const GraphScopeResult replay_b = orch.graph_begin(0x1902, args_b, 0x1736);
    EXPECT_FALSE(replay_b.execute_block) << "the second key's Definition must be cached";
    EXPECT_FALSE(replay_b.recording);
}

// Recording completion order belongs to the worker pool; heap reservation order
// belongs to the main-thread program. Finalizing one Definition at a time walks
// an unordered key map and makes the heap layout depend on hash iteration, so
// finish these four recordings in reverse and require commit to preserve the
// original shell order.
TEST_F(HbgGraphSubmitFailureTest, ConcurrentDefinitionsFinalizeInSubmissionOrder) {
    constexpr size_t kGraphCount = 4;
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs args;
    args.add_input(boundary);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);

    orch.begin_scope();
    std::array<GraphScopeResult, kGraphCount> graphs;
    for (size_t i = 0; i < kGraphCount; ++i) {
        graphs[i] = orch.graph_begin(0x1910 + i, args, 0x1736);
        ASSERT_TRUE(graphs[i].recording) << "Graph " << i;
        ASSERT_TRUE(graphs[i].task_id.is_valid()) << "Graph " << i;
    }

    for (size_t i = kGraphCount; i-- > 0;) {
        ASSERT_TRUE(orch.graph_prepare(graphs[i].recording_handle, args)) << "Graph " << i;
        CoreTaskArgs node_args;
        node_args.add_input(boundary);
        node_args.add_output(recorded_output);
        ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid()) << "Graph " << i;
        ASSERT_TRUE(orch.graph_end()) << "Graph " << i;
    }

    orch.graph_commit();
    ASSERT_FALSE(orch.fatal);
    ASSERT_EQ(graph_host_upload_count(*graph_state), kGraphCount);

    const char *previous_end = nullptr;
    for (size_t i = 0; i < kGraphCount; ++i) {
        const std::optional<GraphHostUpload> upload = graph_host_upload(*graph_state, i);
        ASSERT_TRUE(upload.has_value()) << "Graph " << i;
        EXPECT_NE(upload->definition_hash, 0u) << "Graph " << i;
        const auto *base = static_cast<const char *>(upload->outer_slot->task->packed_buffer_base);
        const auto *end = static_cast<const char *>(upload->outer_slot->task->packed_buffer_end);
        ASSERT_NE(base, nullptr) << "Graph " << i;
        ASSERT_GT(end, base) << "Graph " << i;
        if (previous_end != nullptr) {
            EXPECT_LE(previous_end, base) << "Graph " << i << " was finalized ahead of an earlier shell";
        }
        previous_end = end;
    }
}

// A published Definition is immutable, so replaying it needs nothing from an
// unrelated recording. Gating the cache lookup on an idle recorder made an
// already-built Graph wait for a Definition it has no relationship with.
TEST_F(HbgGraphSubmitFailureTest, ACachedGraphReplaysWhileAnotherKeyRecords) {
    std::array<uint32_t, 16> storage_a{};
    std::array<uint32_t, 16> storage_b{};
    uint32_t shape[] = {static_cast<uint32_t>(storage_a.size())};
    ChipTensor boundary_a = make_tensor_external(storage_a.data(), shape, 1);
    ChipTensor boundary_b = make_tensor_external(storage_b.data(), shape, 1);
    GraphTaskArgs args_a;
    args_a.add_input(boundary_a);
    GraphTaskArgs args_b;
    args_b.add_input(boundary_b);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);

    orch.begin_scope();
    const GraphScopeResult first = orch.graph_begin(0x1903, args_a, 0x1736);
    ASSERT_TRUE(first.recording);
    ASSERT_TRUE(orch.graph_prepare(first.recording_handle, args_a));
    CoreTaskArgs node_a;
    node_a.add_input(boundary_a);
    node_a.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(node_a).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();
    ASSERT_FALSE(orch.fatal);

    // Key B is now recording and stays that way for the rest of the test.
    const GraphScopeResult second = orch.graph_begin(0x1904, args_b, 0x1736);
    ASSERT_TRUE(second.recording);

    const GraphScopeResult replay = orch.graph_begin(0x1903, args_a, 0x1736);
    EXPECT_FALSE(replay.execute_block) << "a cache hit must not wait for an unrelated recording";
    EXPECT_FALSE(replay.recording);
    ASSERT_TRUE(replay.task_id.is_valid());
    // A replay off the cache carries its own heap, unlike the zero-heap shells
    // key B is still deferring.
    EXPECT_GT(orch.ring.task_allocator.heap_top(), 0u);

    ASSERT_TRUE(orch.graph_prepare(second.recording_handle, args_b));
    CoreTaskArgs node_b;
    node_b.add_input(boundary_b);
    node_b.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(node_b).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();
    EXPECT_FALSE(orch.fatal);
}

// An ordinary task submitted while a recording is in flight takes its heap
// immediately, so the shell's deferred block lands after it and heap-address order
// stops matching task-id order. Nothing depends on that correspondence — each
// reservation is an independent bump and HBG retires nothing during a run — which
// is what lets an ordinary submission proceed without joining the recorders.
TEST_F(HbgGraphSubmitFailureTest, AnOrdinaryAllocationInterleavesWithADeferredShell) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    ChipTensor boundary = make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x1905, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    EXPECT_EQ(orch.ring.task_allocator.heap_top(), 0u) << "the shell defers its heap";

    // The ordinary task goes first, from the base of the heap.
    CoreTaskArgs ordinary_args;
    TensorCreateInfo ordinary_output(shape, 1, DataType::UINT32);
    ordinary_args.add_output(ordinary_output);
    const TaskOutputTensors ordinary = orch.alloc_tensors(ordinary_args);
    ASSERT_TRUE(ordinary.task_id().is_valid());
    const uint64_t heap_after_ordinary = orch.ring.task_allocator.heap_top();
    EXPECT_GT(heap_after_ordinary, 0u);

    // Only then does the recording finish and the shell claim its block.
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
    CoreTaskArgs node_args;
    node_args.add_input(boundary);
    node_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(node_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();

    EXPECT_FALSE(orch.fatal);
    EXPECT_GT(orch.ring.task_allocator.heap_top(), heap_after_ordinary)
        << "the shell's block sits above the ordinary task's, not before it";
    PTO2SharedMemoryRingHeader &ring = sm_handle->header->ring;
    const int32_t shell_slot = ring.get_slot_by_task_id(static_cast<int32_t>(graph.task_id.local()));
    const PTO2TaskDescriptor *shell = ring.slot_states[shell_slot].task.get();
    ASSERT_NE(shell, nullptr);
    ASSERT_NE(shell->packed_buffer_base, nullptr);
    EXPECT_GE(
        reinterpret_cast<uintptr_t>(shell->packed_buffer_base),
        reinterpret_cast<uintptr_t>(gm_heap.data()) + heap_after_ordinary
    ) << "the two reservations must be disjoint";
}
