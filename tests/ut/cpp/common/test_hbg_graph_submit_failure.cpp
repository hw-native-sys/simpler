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
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <mutex>
#include <optional>
#include <thread>
#include <utility>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "task_interface/assert_compat.h"
#include "utils/device_arena.h"
#include "host_build_graph/task_id.h"

class HbgGraphSubmitFailureTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    GraphHostStatePtr graph_state;
    std::vector<char> gm_heap;
    // The Definition objects are built in here, as a bind's retained staging.
    // vector<std::byte>::data() is aligned for any fundamental type, which is what
    // an object base has to carry.
    std::vector<std::byte> definition_staging;
    GraphDefinitionArena arena{};

    // A Graph task's heap allocation covers its tasks' packed outputs *and* the
    // execution storage the device materializes into, so the pool has to hold a
    // GraphExecution header plus one ChipTaskStorage (~5 KB) on top of the
    // outputs. 4 KB used to be enough when the storage came from a separate
    // device allocation.
    static constexpr size_t HEAP_BYTES = 64 * 1024;
    static constexpr size_t STAGING_BYTES = 256 * 1024;

    // Where an entry's image is: in the arena at the offset it claimed, or in the
    // buffer it spilled to.
    const GraphDefinition *definition_image(const GraphHostDefinition &entry) const {
        const std::byte *image =
            entry.spill != nullptr ? entry.spill : arena.base + entry.object_offset + arena.object_prefix_bytes;
        return reinterpret_cast<const GraphDefinition *>(image);
    }

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

TEST_F(HbgGraphSubmitFailureTest, InFlightGraphInvocationsReserveHeapOnlyAtCommit) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
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
    EXPECT_EQ(second.task_id.local_id(), first.task_id.local_id() + 1);
    EXPECT_EQ(orch.task_allocator.heap_top(), 0u);
    EXPECT_EQ(graph_host_upload_count(*graph_state), 2u);

    ASSERT_TRUE(orch.graph_prepare(first.recording_handle, boundary_args));
    // What a body actually receives: the entry's own parameter list. Its tensors carry
    // recording-space addresses and PARAM provenance, which is what the classifier
    // resolves against -- the caller's own tensor is an object the body never saw.
    const simpler::hbg::Tensor &param = first.params->tensor(0).ref();
    CoreTaskArgs task_args;
    task_args.add_input(param);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    task_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    EXPECT_EQ(orch.task_allocator.heap_top(), 0u);

    orch.graph_commit();
    EXPECT_FALSE(orch.is_fatal());
    EXPECT_GT(orch.task_allocator.heap_top(), 0u);
    const std::optional<GraphHostUpload> first_upload = graph_host_upload(*graph_state, 0);
    const std::optional<GraphHostUpload> second_upload = graph_host_upload(*graph_state, 1);
    ASSERT_TRUE(first_upload.has_value());
    ASSERT_TRUE(second_upload.has_value());
    EXPECT_NE(first_upload->full_key, 0u);
    EXPECT_EQ(second_upload->full_key, first_upload->full_key) << "both shells replay one Graph";
    // Distinct bases alone would still pass if finalization handed out a wrong
    // extent, so pin the length the Definition asks for and the disjointness two
    // shells of one Graph must have.
    const auto *first_base = static_cast<const char *>(first_upload->outer_slot->to_descriptor().packed_buffer_base);
    const auto *first_end = static_cast<const char *>(first_upload->outer_slot->to_descriptor().packed_buffer_end);
    const auto *second_base = static_cast<const char *>(second_upload->outer_slot->to_descriptor().packed_buffer_base);
    const auto *second_end = static_cast<const char *>(second_upload->outer_slot->to_descriptor().packed_buffer_end);
    const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
    ASSERT_EQ(definitions.entries.size(), 1u);
    ASSERT_EQ(definitions.entries[0].full_key, first_upload->full_key);
    const GraphDefinition *definition = definition_image(definitions.entries[0]);
    const uint64_t expected_extent =
        CHIP_ALIGN_UP(definition->required_heap + definition->execution_storage_bytes, CHIP_ALIGN_SIZE);
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
// boundary_scalar_count, while the worker writes boundary_args / tasks /
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
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
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
    bool task_ok = false;
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

        CoreTaskArgs task_args;
        task_args.add_input(first.params->tensor(0).ref());
        TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
        task_args.add_output(recorded_output);
        task_ok = orch.submit_dummy_task(task_args).task_id().is_valid();
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
    ASSERT_TRUE(task_ok);
    ASSERT_TRUE(end_ok);
    EXPECT_FALSE(second.recording);
    EXPECT_FALSE(second.execute_block);
    EXPECT_FALSE(third.execute_block);
    ASSERT_TRUE(second.task_id.is_valid());
    ASSERT_TRUE(third.task_id.is_valid());
    EXPECT_EQ(second.task_id.local_id(), first.task_id.local_id() + 1);
    EXPECT_EQ(third.task_id.local_id(), first.task_id.local_id() + 2);
    EXPECT_EQ(orch.task_allocator.heap_top(), 0u) << "no shell may take heap before commit";

    orch.graph_commit();
    ASSERT_FALSE(orch.is_fatal());
    ASSERT_EQ(graph_host_upload_count(*graph_state), 3u);

    const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
    ASSERT_EQ(definitions.entries.size(), 1u);
    const GraphDefinition *definition = definition_image(definitions.entries[0]);
    const uint64_t expected_extent =
        CHIP_ALIGN_UP(definition->required_heap + definition->execution_storage_bytes, CHIP_ALIGN_SIZE);

    std::vector<std::pair<const char *, const char *>> ranges;
    for (size_t i = 0; i < 3; ++i) {
        const std::optional<GraphHostUpload> upload = graph_host_upload(*graph_state, i);
        ASSERT_TRUE(upload.has_value());
        EXPECT_EQ(upload->full_key, definition->full_key) << "shell " << i;
        const auto *base = static_cast<const char *>(upload->outer_slot->to_descriptor().packed_buffer_base);
        const auto *end = static_cast<const char *>(upload->outer_slot->to_descriptor().packed_buffer_end);
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
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x1717, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(graph.task_id.is_valid());
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs task_args;
    task_args.add_input(graph.params->tensor(0).ref());
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    task_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());

    orch.graph_abort(graph.recording_handle);
    ASSERT_FALSE(orch.is_fatal()) << "Abort alone must not latch; the shell is still finalizable in principle";

    orch.graph_commit();
    EXPECT_TRUE(orch.is_fatal()) << "A shell whose Definition never arrived cannot be completed";
}

// A recording worker reaches report_fatal for anything the recording cannot answer
// locally: every submit entry validates its arguments ahead of its recording branch,
// and the public rt_report_fatal is callable from a body. Two things have to hold
// afterwards, and neither is about the code that was latched.
//
// The entry has to leave RECORDING. graph_commit's drain blocks until every in-flight
// entry has, and on this path graph_end is the only thing the worker calls that can
// perform the transition — so a graph_end that returns early on the fatal turns a
// reported error into a hang on the bind thread.
//
// And the worker's recorder thread_locals have to be released. The recorder pool
// outlives the run, so a thread that keeps them bound fails graph_prepare's
// already-recording guard for every later recording it is handed.
TEST_F(HbgGraphSubmitFailureTest, AFatalDuringRecordingRetiresTheEntryAndFreesTheRecorderThread) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    // Two keys, so the worker has a second recording to prove its thread_locals came
    // back. Both open before any fatal is latched.
    const GraphScopeResult first = orch.graph_begin(0x1720, boundary_args, 0x1736);
    const GraphScopeResult second = orch.graph_begin(0x1721, boundary_args, 0x1736);
    ASSERT_TRUE(first.recording);
    ASSERT_TRUE(second.recording);

    bool first_prepare_ok = false;
    bool first_end_ok = true;
    bool second_prepare_ok = false;
    bool second_end_ok = true;
    bool retired_entry_refuses_prepare = false;

    std::thread worker([&]() {
        // Worker-owned boundary copies, alive until each recording ends.
        GraphTaskArgs first_args;
        first_args.add_input(boundary);
        first_prepare_ok = orch.graph_prepare(first.recording_handle, first_args);
        if (!first_prepare_ok) return;

        // The body reports a fatal. This is the cross-thread write fatal_code is
        // atomic for: the bind thread reads it through is_fatal() at every entry.
        orch.report_fatal(SIMPLER_ERROR_EXPLICIT_ORCH_FATAL, "recorded_body", "%s", "the body reported a fatal");
        first_end_ok = orch.graph_end();

        GraphTaskArgs second_args;
        second_args.add_input(boundary);
        second_prepare_ok = orch.graph_prepare(second.recording_handle, second_args);
        if (!second_prepare_ok) return;
        second_end_ok = orch.graph_end();

        // With this thread's thread_locals proven clear by the prepare above, the only
        // reason left to refuse the first handle is that its entry is no longer
        // RECORDING -- which is the transition graph_commit's drain waits for.
        GraphTaskArgs retry_args;
        retry_args.add_input(boundary);
        retired_entry_refuses_prepare = !orch.graph_prepare(first.recording_handle, retry_args);
    });
    worker.join();

    ASSERT_TRUE(first_prepare_ok);
    EXPECT_FALSE(first_end_ok) << "a fatal publishes no Definition, so end must decline";
    EXPECT_TRUE(second_prepare_ok) << "graph_end must release the recorder thread_locals it bound";
    EXPECT_FALSE(second_end_ok);
    EXPECT_TRUE(retired_entry_refuses_prepare) << "a fatal must leave the entry out of RECORDING";

    // Returns rather than blocking: the drain's predicate is already satisfied, because
    // both entries left RECORDING above.
    orch.graph_commit();

    EXPECT_TRUE(orch.is_fatal());
    EXPECT_EQ(orch.fatal_code.load(std::memory_order_acquire), SIMPLER_ERROR_EXPLICIT_ORCH_FATAL)
        << "first-writer-wins: commit's own SIMPLER_ERROR_INVALID_ARGS must not displace the body's code";
    EXPECT_EQ(graph_host_definitions(*graph_state).entries.size(), 0u) << "no Definition may be published";
}

// The ordinary path reports SIMPLER_ERROR_INVALID_ARGS for an auto scope opened
// inside a manual one. The recording pass keeps a scope depth of its own — the
// manual flag has to reach compute_task_fanin, which suppresses inference inside
// a manual scope — so it has to refuse the same nesting. Accepting it would let a
// Graph record and replay a body ordinary submission rejects outright.
TEST_F(HbgGraphSubmitFailureTest, AutoScopeNestedInManualScopeRefusesTheRecording) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x171d, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    orch.begin_scope(ScopeMode::MANUAL);
    orch.begin_scope(ScopeMode::AUTO);

    CoreTaskArgs task_args;
    task_args.add_input(graph.params->tensor(0).ref());
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    task_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());

    orch.end_scope();
    orch.end_scope();

    EXPECT_THROW(orch.graph_end(), AssertionError) << "an auto scope inside a manual one must not publish";
    orch.graph_abort(graph.recording_handle);
    orch.graph_commit();
    EXPECT_TRUE(orch.is_fatal()) << "a shell whose Definition never arrived cannot be completed";
}

// A Graph body may allocate. The allocation records as a kernel-less in-graph task,
// the same shape submit_dummy_task records, so the recording stays publishable and
// the commit latches no fatal.
TEST_F(HbgGraphSubmitFailureTest, RuntimeAllocationInsideTheBodyRecordsAKernellessInGraphTask) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
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
    EXPECT_FALSE(orch.is_fatal());
}

// A parameter's recording-space window is reserved once per buffer address and sized from
// the first parameter to claim it, so the parameters at one address have to agree on how
// wide that is. Two that present one address at two sizes do not, and the boundary is
// refused -- the invocation takes the ordinary path, which is always correct.
//
// This is one of the two properties the alias partition checks; the other is that no
// buffer is empty. That *different* addresses name non-overlapping memory is not checked
// but assumed: every buffer a boundary can name comes from the one allocator, which hands
// out disjoint blocks, and proving it per invocation would cost an ordering of the
// addresses where grouping alone needs only a hash.
TEST_F(HbgGraphSubmitFailureTest, ABoundaryPresentingOneAddressAtTwoSizesTakesTheOrdinaryPath) {
    std::array<uint32_t, 32> storage{};
    uint32_t wide_shape[] = {32};
    uint32_t narrow_shape[] = {16};
    // One address, two buffer sizes -- `buffer.size` comes from the tensor's own shape, so
    // two shapes over one base is all it takes.
    simpler::hbg::Tensor wide = simpler::hbg::make_tensor_external(storage.data(), wide_shape, 1);
    simpler::hbg::Tensor narrow = simpler::hbg::make_tensor_external(storage.data(), narrow_shape, 1);
    ASSERT_EQ(wide.buffer.addr, narrow.buffer.addr);
    ASSERT_NE(wide.buffer.size, narrow.buffer.size);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_input(wide);
    boundary_args.add_inout(narrow);
    const GraphScopeResult graph = orch.graph_begin(0x1723, boundary_args, 0x1736);

    EXPECT_FALSE(graph.recording) << "an unrepresentable boundary opens no recording";
    EXPECT_TRUE(graph.execute_block) << "the caller runs the body itself instead";
    EXPECT_EQ(graph_host_upload_count(*graph_state), 0u) << "and no shell was submitted for it";
    orch.graph_commit();
    EXPECT_FALSE(orch.is_fatal()) << "declining to record is not a failure";
}

// A body may only use what came through its boundary, or what one of its own tasks
// produced. A tensor that entered any other way -- a global, or an upstream task's
// output the caller never declared as a parameter -- has no place in a Definition:
// replay would rebind it against whatever the boundary holds that time, so it is
// refused and the recording abandoned.
//
// The refusal is by provenance, not by address. This tensor's address is a real one,
// and a recording's own space starts just above zero, so the two ranges overlap: an
// address test could attribute this to a parameter instead of rejecting it.
TEST_F(HbgGraphSubmitFailureTest, ATensorThatSkippedTheBoundaryIsRefused) {
    std::array<uint32_t, 16> storage{};
    std::array<uint32_t, 16> foreign_storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    simpler::hbg::Tensor foreign = simpler::hbg::make_tensor_external(foreign_storage.data(), shape, 1);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x1722, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs task_args;
    task_args.add_input(foreign);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    task_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());

    // Assertions are enabled in this build, so the unsupported construct surfaces as
    // the debug_assert graph_end() fires on its way out -- which precedes its own
    // abort, so the abort is issued here instead.
    EXPECT_THROW(orch.graph_end(), AssertionError) << "a tensor that skipped the boundary must not publish";
    orch.graph_abort(graph.recording_handle);
    orch.graph_commit();
    EXPECT_TRUE(orch.is_fatal()) << "a shell whose Definition never arrived cannot be completed";
}

// A parameter's view origin is the one field of the boundary contract that may move. Every
// recorded tensor stores its own origin relative to the parameter it came from, so replay
// adds this invocation's origin and the whole body follows the parameter -- which is what
// lets a sliding-window caller keep one Definition instead of recording per slice.
//
// Reuse alone would not prove that: a Definition recorded against origin 0 and replayed
// against origin 0 reuses too. What has to reach the device is the *new* origin, so the
// outer payload is read for it.
TEST_F(HbgGraphSubmitFailureTest, ASlidingBoundaryOriginKeepsReusingItsDefinition) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor base = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    // Same buffer, same shape, same stride, same contiguity -- only the origin differs, so
    // every field graph_boundary_param_matches compares is equal across the two. Named
    // locals because GraphTaskArgs stores a pointer to what it is given.
    simpler::hbg::Tensor recorded_slice = base.slice(0, 0, 8);
    simpler::hbg::Tensor slid_slice = base.slice(0, 4, 12);
    ASSERT_EQ(recorded_slice.start_offset, 0u);
    ASSERT_EQ(slid_slice.start_offset, 4u);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_input(recorded_slice);
    const GraphScopeResult graph = orch.graph_begin(0x1724, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs task_args;
    task_args.add_input(graph.params->tensor(0).ref());
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    GraphTaskArgs slid_args;
    slid_args.add_input(slid_slice);
    const GraphScopeResult replay = orch.graph_begin(0x1724, slid_args, 0x1736);

    EXPECT_FALSE(replay.execute_block) << "a slid origin is inside the boundary contract";
    ASSERT_TRUE(replay.task_id.is_valid());
    EXPECT_FALSE(orch.is_fatal());

    // The boundary the device rebinds against is this invocation's argument, held in the
    // outer task's own tensor pool. Its origin is what a recorded tensor's relative origin
    // is added to, so a Definition reused under a slid argument is only correct while this
    // carries the slid value rather than the recorded one.
    const ChipTaskStorage *slots = sm_handle->header->tasks.task_storage;
    ASSERT_NE(slots, nullptr);
    const simpler::hbg::Tensor *replay_boundary = slots[replay.task_id.local_id()].payload.tensor_data();
    ASSERT_NE(replay_boundary, nullptr);
    EXPECT_EQ(replay_boundary[0].start_offset, slid_slice.start_offset);
}

// What a partition's members may not do is slide against each other. Recording inferred
// the body's WAR/WAW edges from where their views sat in the buffer they share, and those
// edges are fixed in the Definition, so the distance between two members is part of the
// reuse condition even though neither member's absolute origin is.
//
// Both sides of the rule are one test: the uniform slide has to still reuse, or the
// refusal below would be explained by the slide rather than by the differential.
TEST_F(HbgGraphSubmitFailureTest, PartitionMembersSlidingAgainstEachOtherDoNotReuseTheDefinition) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor base = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    // Three presentations of one buffer, so one alias partition throughout, whose
    // representative is parameter 0. Named locals because GraphTaskArgs stores a pointer to
    // what it is given, and every pair has to outlive the graph_begin call that reads it.
    simpler::hbg::Tensor recorded_first = base.slice(0, 0, 4);
    simpler::hbg::Tensor recorded_second = base.slice(0, 4, 8);
    simpler::hbg::Tensor uniform_first = base.slice(0, 2, 6);
    simpler::hbg::Tensor uniform_second = base.slice(0, 6, 10);
    simpler::hbg::Tensor differential_first = base.slice(0, 0, 4);
    simpler::hbg::Tensor differential_second = base.slice(0, 8, 12);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_inout(recorded_first);
    boundary_args.add_input(recorded_second);
    const GraphScopeResult graph = orch.graph_begin(0x1725, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs writer_args;
    writer_args.add_inout(graph.params->tensor(0).ref());
    ASSERT_TRUE(orch.submit_dummy_task(writer_args).task_id().is_valid());
    CoreTaskArgs reader_args;
    reader_args.add_input(graph.params->tensor(1).ref());
    ASSERT_TRUE(orch.submit_dummy_task(reader_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    const size_t uploads_after_recording = graph_host_upload_count(*graph_state);

    // Both members moved by 2, so their distance is the recorded 4 and every recorded
    // tensor's relative origin still lands where it did.
    GraphTaskArgs uniform_args;
    uniform_args.add_inout(uniform_first);
    uniform_args.add_input(uniform_second);
    const GraphScopeResult uniform = orch.graph_begin(0x1725, uniform_args, 0x1736);
    EXPECT_FALSE(uniform.execute_block) << "a uniform slide of the whole partition is reusable";
    EXPECT_TRUE(uniform.task_id.is_valid());

    // Only the second member moved, so the distance is 8 where the Definition's edges were
    // inferred from 4. The arrangement check refuses it, and this build's assertions turn
    // that refusal into a throw.
    GraphTaskArgs differential_args;
    differential_args.add_inout(differential_first);
    differential_args.add_input(differential_second);
    EXPECT_THROW(orch.graph_begin(0x1725, differential_args, 0x1736), AssertionError)
        << "a differential slide changes the overlap geometry the Definition baked in";
    EXPECT_EQ(graph_host_upload_count(*graph_state), uploads_after_recording + 1)
        << "only the uniform slide got a shell; the refused one submitted nothing";
    EXPECT_FALSE(orch.is_fatal()) << "declining to reuse is not a failure";
}

TEST_F(HbgGraphSubmitFailureTest, FaninFailureLatchesFatalWithoutPartialUpload) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x1715, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs task_args;
    task_args.add_input(graph.params->tensor(0).ref());
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
    task_args.add_output(recorded_output);
    const uint64_t heap_top_before_record = orch.task_allocator.heap_top();
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    EXPECT_EQ(orch.task_allocator.heap_top(), heap_top_before_record);
    ASSERT_TRUE(orch.graph_end());
    EXPECT_EQ(orch.task_allocator.heap_top(), heap_top_before_record);
    orch.graph_commit();
    EXPECT_GT(orch.task_allocator.heap_top(), heap_top_before_record);
    ASSERT_FALSE(orch.is_fatal());
    const size_t uploads_before_failure = graph_host_upload_count(*graph_state);

    CoreTaskArgs producer_args;
    producer_args.add_output(boundary);
    for (int32_t i = 0; i < CHIP_MAX_FANIN + 1; ++i) {
        ASSERT_TRUE(orch.submit_dummy_task(producer_args).task_id().is_valid());
    }

    const GraphScopeResult replay = orch.graph_begin(0x1715, boundary_args, 0x1736);

    EXPECT_TRUE(replay.execute_block);
    EXPECT_FALSE(replay.recording);
    EXPECT_FALSE(replay.task_id.is_valid());
    EXPECT_TRUE(orch.is_fatal());
    EXPECT_EQ(orch.fatal_code.load(std::memory_order_acquire), SIMPLER_ERROR_FANIN_CAPACITY_EXCEEDED);
    EXPECT_EQ(graph_host_upload_count(*graph_state), uploads_before_failure);
}

TEST_F(HbgGraphSubmitFailureTest, CachedGraphUsesFinalTaskWindowSlot) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    orch.begin_scope();
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    const GraphScopeResult graph = orch.graph_begin(0x1716, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

    CoreTaskArgs task_args;
    task_args.add_input(graph.params->tensor(0).ref());
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    ASSERT_EQ(orch.task_allocator.active_count(), 1);

    TaskAllocator &allocator = orch.task_allocator;
    while (allocator.active_count() < allocator.capacity() - 1) {
        ASSERT_FALSE(allocator.alloc(0).failed());
    }

    const GraphScopeResult replay = orch.graph_begin(0x1716, boundary_args, 0x1736);

    EXPECT_FALSE(replay.execute_block);
    ASSERT_TRUE(replay.task_id.is_valid());
    EXPECT_EQ(replay.task_id.local_id(), allocator.capacity() - 1);
    EXPECT_EQ(allocator.active_count(), allocator.capacity());
    EXPECT_EQ(allocator.active_count(), allocator.capacity());
    EXPECT_EQ(orch.fatal_code.load(std::memory_order_acquire), SIMPLER_ERROR_NONE);
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
    // Records one predicated in-graph task into a fresh Graph and asserts the recording
    // refused it. `build_predicate` receives the boundary parameter the body reads.
    template <typename BuildPredicate>
    void expect_recording_refused(uint64_t graph_key, BuildPredicate build_predicate) {
        std::array<uint32_t, 16> storage{};
        uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
        simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1, DataType::INT32);
        GraphTaskArgs boundary_args;
        boundary_args.add_input(boundary);

        orch.begin_scope();
        const GraphScopeResult graph = orch.graph_begin(graph_key, boundary_args, 0x1736);
        EXPECT_TRUE(graph.recording);
        EXPECT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
        const simpler::hbg::Tensor &param = graph.params->tensor(0).ref();

        CoreTaskArgs task_args;
        task_args.add_input(param);
        TensorCreateInfo recorded_output(shape, 1, DataType::INT32);
        task_args.add_output(recorded_output);
        MixedKernels mixed{};
        mixed.aiv0_kernel_id = 0;
        task_args.set_predicate(build_predicate(param));
        EXPECT_TRUE(orch.submit_task(mixed, task_args).task_id().is_valid());

        EXPECT_THROW(orch.graph_end(), AssertionError) << "an unrecordable predicate must not publish";
        orch.graph_abort(graph.recording_handle);
        orch.graph_commit();
        EXPECT_TRUE(orch.is_fatal()) << "a shell whose Definition never arrived cannot be completed";
    }

    static CoreTaskPredicate predicate_on(const simpler::hbg::Tensor &operand, uint32_t index) {
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
    expect_recording_refused(0x2001, [](const simpler::hbg::Tensor &boundary) {
        return predicate_on(boundary, 16);
    });
}

TEST_F(HbgGraphPredicateRejectionTest, OperandOnAnUnclassifiableTensorAbortsTheRecording) {
    // Neither a boundary tensor nor any recorded task's output, so the recorder
    // cannot name a base the replay could rebind against.
    std::array<uint32_t, 16> foreign_storage{};
    uint32_t shape[] = {static_cast<uint32_t>(foreign_storage.size())};
    const simpler::hbg::Tensor foreign =
        simpler::hbg::make_tensor_external(foreign_storage.data(), shape, 1, DataType::INT32);
    expect_recording_refused(0x2002, [&foreign](const simpler::hbg::Tensor &) {
        return predicate_on(foreign, 0);
    });
}

// A kernel-less in-graph task never dispatches, so submit_dummy_task and alloc_tensors
// drop the caller's predicate exactly as they do on the ordinary path. Recording
// must drop it too: a task whose Definition claimed a predicate its own attribute
// denies is rejected by materialize, on the device, for a value the scheduler was
// never going to read.
TEST_F(HbgGraphPredicateRejectionTest, PredicateOnAKernellessInGraphTaskIsNotRecorded) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1, DataType::INT32);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x2003, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
    const simpler::hbg::Tensor &param = graph.params->tensor(0).ref();

    CoreTaskArgs task_args;
    task_args.add_input(param);
    TensorCreateInfo recorded_output(shape, 1, DataType::INT32);
    task_args.add_output(recorded_output);
    // Out of extent, which a recorded predicate would reject — proving the
    // predicate never reached the recorder rather than merely passing its checks.
    task_args.set_predicate(predicate_on(param, 16));
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());

    EXPECT_TRUE(orch.graph_end()) << "a dropped predicate must not make the body unrecordable";
    orch.graph_commit();
    EXPECT_FALSE(orch.is_fatal());
}

// Distinct Graph keys record concurrently. A Definition the run has not seen
// before must open its own recording even while another is in flight — the
// alternative is that it is turned away, replays nothing for the rest of the run,
// and every occurrence of it submits its whole body as ordinary tasks.
TEST_F(HbgGraphSubmitFailureTest, ASecondKeyRecordsAlongsideTheFirst) {
    std::array<uint32_t, 16> storage_a{};
    std::array<uint32_t, 16> storage_b{};
    uint32_t shape[] = {static_cast<uint32_t>(storage_a.size())};
    simpler::hbg::Tensor boundary_a = simpler::hbg::make_tensor_external(storage_a.data(), shape, 1);
    simpler::hbg::Tensor boundary_b = simpler::hbg::make_tensor_external(storage_b.data(), shape, 1);
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
    CoreTaskArgs task_a;
    task_a.add_input(first.params->tensor(0).ref());
    task_a.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_a).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    ASSERT_TRUE(orch.graph_prepare(second.recording_handle, args_b));
    CoreTaskArgs task_b;
    task_b.add_input(second.params->tensor(0).ref());
    task_b.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_b).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());

    // One commit drains and back-patches both keys' deferred shells.
    orch.graph_commit();
    EXPECT_FALSE(orch.is_fatal());

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
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
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
        CoreTaskArgs task_args;
        task_args.add_input(graphs[i].params->tensor(0).ref());
        task_args.add_output(recorded_output);
        ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid()) << "Graph " << i;
        ASSERT_TRUE(orch.graph_end()) << "Graph " << i;
    }

    orch.graph_commit();
    ASSERT_FALSE(orch.is_fatal());
    ASSERT_EQ(graph_host_upload_count(*graph_state), kGraphCount);

    const char *previous_end = nullptr;
    for (size_t i = 0; i < kGraphCount; ++i) {
        const std::optional<GraphHostUpload> upload = graph_host_upload(*graph_state, i);
        ASSERT_TRUE(upload.has_value()) << "Graph " << i;
        EXPECT_NE(upload->full_key, 0u) << "Graph " << i;
        const auto *base = static_cast<const char *>(upload->outer_slot->to_descriptor().packed_buffer_base);
        const auto *end = static_cast<const char *>(upload->outer_slot->to_descriptor().packed_buffer_end);
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
    simpler::hbg::Tensor boundary_a = simpler::hbg::make_tensor_external(storage_a.data(), shape, 1);
    simpler::hbg::Tensor boundary_b = simpler::hbg::make_tensor_external(storage_b.data(), shape, 1);
    GraphTaskArgs args_a;
    args_a.add_input(boundary_a);
    GraphTaskArgs args_b;
    args_b.add_input(boundary_b);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);

    orch.begin_scope();
    const GraphScopeResult first = orch.graph_begin(0x1903, args_a, 0x1736);
    ASSERT_TRUE(first.recording);
    ASSERT_TRUE(orch.graph_prepare(first.recording_handle, args_a));
    CoreTaskArgs task_a;
    task_a.add_input(first.params->tensor(0).ref());
    task_a.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_a).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();
    ASSERT_FALSE(orch.is_fatal());

    // Key B is now recording and stays that way for the rest of the test.
    const GraphScopeResult second = orch.graph_begin(0x1904, args_b, 0x1736);
    ASSERT_TRUE(second.recording);

    const GraphScopeResult replay = orch.graph_begin(0x1903, args_a, 0x1736);
    EXPECT_FALSE(replay.execute_block) << "a cache hit must not wait for an unrelated recording";
    EXPECT_FALSE(replay.recording);
    ASSERT_TRUE(replay.task_id.is_valid());
    // A replay off the cache carries its own heap, unlike the zero-heap shells
    // key B is still deferring.
    EXPECT_GT(orch.task_allocator.heap_top(), 0u);

    ASSERT_TRUE(orch.graph_prepare(second.recording_handle, args_b));
    CoreTaskArgs task_b;
    task_b.add_input(second.params->tensor(0).ref());
    task_b.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_b).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();
    EXPECT_FALSE(orch.is_fatal());
}

// An ordinary task submitted while a recording is in flight takes its heap
// immediately, so the shell's deferred block lands after it and heap-address order
// stops matching task-id order. Nothing depends on that correspondence — each
// reservation is an independent bump and HBG retires nothing during a run — which
// is what lets an ordinary submission proceed without joining the recorders.
TEST_F(HbgGraphSubmitFailureTest, AnOrdinaryAllocationInterleavesWithADeferredShell) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);
    GraphTaskArgs boundary_args;
    boundary_args.add_input(boundary);
    TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);

    orch.begin_scope();
    const GraphScopeResult graph = orch.graph_begin(0x1905, boundary_args, 0x1736);
    ASSERT_TRUE(graph.recording);
    EXPECT_EQ(orch.task_allocator.heap_top(), 0u) << "the shell defers its heap";

    // The ordinary task goes first, from the base of the heap.
    CoreTaskArgs ordinary_args;
    TensorCreateInfo ordinary_output(shape, 1, DataType::UINT32);
    ordinary_args.add_output(ordinary_output);
    const TaskOutputTensors ordinary = orch.alloc_tensors(ordinary_args);
    ASSERT_TRUE(ordinary.task_id().is_valid());
    const uint64_t heap_after_ordinary = orch.task_allocator.heap_top();
    EXPECT_GT(heap_after_ordinary, 0u);

    // Only then does the recording finish and the shell claim its block.
    ASSERT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
    CoreTaskArgs task_args;
    task_args.add_input(graph.params->tensor(0).ref());
    task_args.add_output(recorded_output);
    ASSERT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();

    EXPECT_FALSE(orch.is_fatal());
    EXPECT_GT(orch.task_allocator.heap_top(), heap_after_ordinary)
        << "the shell's block sits above the ordinary task's, not before it";
    SharedMemoryTaskHeader &tasks = sm_handle->header->tasks;
    const int32_t shell_slot = graph.task_id.local_id();
    const TaskDescriptor *shell = &tasks.storage_at(shell_slot).task;
    ASSERT_NE(shell, nullptr);
    ASSERT_NE(shell->packed_buffer_base, nullptr);
    EXPECT_GE(
        reinterpret_cast<uintptr_t>(shell->packed_buffer_base),
        reinterpret_cast<uintptr_t>(gm_heap.data()) + heap_after_ordinary
    ) << "the two reservations must be disjoint";
}

// A Graph's boundary tensor can be an upstream task's output, which lives in the
// graph heap and therefore carries an address out of HEAP_VIRTUAL_BASE's window
// while recording. Recording must still classify such a tensor as a boundary: a
// captured boundary is moved into the recording's own address space and stamped
// with PARAM provenance, so where the caller's buffer lived does not reach the
// classifier at all. If it fell through to the recorded-output resolution instead,
// the task would be marked unsupported and the whole Graph would silently drop to
// the ordinary path.
//
// The Definition describes a tensor that came from a parameter by its shape,
// strides, buffer size and provenance, and never by an address -- a parameter's is
// dropped, because replay takes the buffer from the invocation's own argument. So
// the same body over a heap-resident boundary must record that tensor exactly as
// one over a caller-owned boundary of the same shape. That is what is compared
// here; comparing whole images would also fold in full_key, which differs between
// two recordings under different graph_keys whatever their boundaries are.
struct BoundaryRecording {
    uint64_t full_key;
    simpler::hbg::TensorData recorded;
};

TEST_F(HbgGraphSubmitFailureTest, RecordsAGraphWhoseBoundaryLivesInTheHeapWindow) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    // Never dereferenced: recording is bookkeeping over the tensor's descriptor.
    auto *heap_resident = reinterpret_cast<void *>(HEAP_VIRTUAL_BASE + 0x2000);
    const uint64_t nbytes = storage.size() * sizeof(uint32_t);

    auto record_with = [&](void *boundary_addr, uint64_t graph_key) -> std::optional<BoundaryRecording> {
        const size_t definitions_before = graph_host_definitions(*graph_state).entries.size();
        const size_t uploads_before = graph_host_upload_count(*graph_state);
        simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(boundary_addr, shape, 1);
        GraphTaskArgs boundary_args;
        boundary_args.add_input(boundary);

        const GraphScopeResult graph = orch.graph_begin(graph_key, boundary_args, 0x1736);
        EXPECT_TRUE(graph.recording);
        EXPECT_TRUE(graph.task_id.is_valid());
        EXPECT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));

        CoreTaskArgs task_args;
        task_args.add_input(graph.params->tensor(0).ref());
        TensorCreateInfo recorded_output(shape, 1, DataType::UINT32);
        task_args.add_output(recorded_output);
        EXPECT_TRUE(orch.submit_dummy_task(task_args).task_id().is_valid());
        EXPECT_TRUE(orch.graph_end());
        orch.graph_commit();
        EXPECT_FALSE(orch.is_fatal());

        // Each call uses its own graph_key, so it publishes exactly one Definition
        // and appends exactly one upload. That upload names this call's full_key
        // (graph_key combined with the callable hash), which is how the Definition
        // is selected: graph_host_definitions walks an unordered_map, so the order
        // of `entries` says nothing about which call published which.
        const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
        if (definitions.entries.size() != definitions_before + 1) {
            ADD_FAILURE() << "graph_key " << graph_key << " published "
                          << (definitions.entries.size() - definitions_before) << " Definitions, expected 1";
            return std::nullopt;
        }
        if (graph_host_upload_count(*graph_state) != uploads_before + 1) {
            ADD_FAILURE() << "graph_key " << graph_key << " appended "
                          << (graph_host_upload_count(*graph_state) - uploads_before) << " uploads, expected 1";
            return std::nullopt;
        }
        const std::optional<GraphHostUpload> upload = graph_host_upload(*graph_state, uploads_before);
        if (!upload.has_value()) {
            ADD_FAILURE() << "graph_key " << graph_key << " has no upload at index " << uploads_before;
            return std::nullopt;
        }
        const GraphHostDefinition *published = nullptr;
        for (const GraphHostDefinition &entry : definitions.entries) {
            if (entry.full_key == upload->full_key) {
                published = &entry;
                break;
            }
        }
        if (published == nullptr) {
            ADD_FAILURE() << "graph_key " << graph_key << " published no Definition under its own full_key";
            return std::nullopt;
        }
        const GraphDefinition *def = definition_image(*published);
        if (def->boundary_tensor_count != 1 || def->tensor_arg_count != 2) {
            ADD_FAILURE() << "graph_key " << graph_key << " recorded " << def->boundary_tensor_count
                          << " boundaries and " << def->tensor_arg_count << " tensor args, expected 1 and 2";
            return std::nullopt;
        }
        // off_tensors is an offset into the Definition image, whose base is what
        // definition_image resolves to. The body's one task takes the parameter first and
        // its own output second, so entry 0 is the tensor under test.
        const auto *tensors = reinterpret_cast<const simpler::hbg::TensorData *>(
            reinterpret_cast<const std::byte *>(def) + def->off_tensors
        );
        if (tensors[0].owner_task_id.space() != TaskId::Space::PARAM) {
            ADD_FAILURE() << "graph_key " << graph_key << " did not record its first tensor arg as a parameter";
            return std::nullopt;
        }
        return BoundaryRecording{def->full_key, tensors[0]};
    };

    orch.begin_scope();
    const std::optional<BoundaryRecording> heap_recording = record_with(heap_resident, 0x1801);
    const std::optional<BoundaryRecording> caller_recording = record_with(storage.data(), 0x1802);
    ASSERT_TRUE(heap_recording.has_value());
    ASSERT_TRUE(caller_recording.has_value());

    // Two distinct Definitions, so the comparison below is between two recordings
    // rather than one Definition against itself.
    EXPECT_NE(heap_recording->full_key, caller_recording->full_key);
    // Field-wise rather than byte-wise: a Tensor writes only the shape and stride slots its
    // ndims covers, so the trailing ones hold whatever the view before it left there.
    const simpler::hbg::TensorData &from_heap = heap_recording->recorded;
    const simpler::hbg::TensorData &from_caller = caller_recording->recorded;
    EXPECT_EQ(from_heap.buffer.addr, 0u) << "a parameter's recorded address is dropped, not kept";
    EXPECT_EQ(from_caller.buffer.addr, 0u) << "a parameter's recorded address is dropped, not kept";
    const bool same =
        from_heap.buffer.size == from_caller.buffer.size && from_heap.owner_task_id == from_caller.owner_task_id &&
        from_heap.start_offset == from_caller.start_offset &&
        from_heap.extent_elem_cache == from_caller.extent_elem_cache && from_heap.version == from_caller.version &&
        from_heap.ndims == from_caller.ndims && from_heap.dtype == from_caller.dtype &&
        from_heap.manual_dep == from_caller.manual_dep && from_heap.is_contiguous == from_caller.is_contiguous &&
        from_heap.address_space == from_caller.address_space &&
        std::equal(
            std::begin(from_heap.shapes), std::begin(from_heap.shapes) + from_heap.ndims, std::begin(from_caller.shapes)
        ) &&
        std::equal(
            std::begin(from_heap.strides), std::begin(from_heap.strides) + from_heap.ndims,
            std::begin(from_caller.strides)
        );
    EXPECT_TRUE(same) << "a recorded parameter must not depend on where its buffer lives";
    // The address the recording saw stayed in the heap window, i.e. the test really
    // exercised a heap-resident boundary rather than a coincidentally-real address.
    EXPECT_GE(reinterpret_cast<uint64_t>(heap_resident), HEAP_VIRTUAL_BASE);
    EXPECT_LT(reinterpret_cast<uint64_t>(heap_resident) + nbytes, HEAP_VIRTUAL_BASE + MAX_HEAP_CAPACITY);
}

// A hidden-alloc task's payload passes through TaskPayload::init() and nothing
// else — unlike an ordinary task, no dispatch-predicate assignment follows it,
// and unlike an outer GRAPH task, no graph_reset_outer_payload precedes it. So
// init() is where its predicate has to acquire a defined value: the ring's payload
// storage is reused raw memory that no constructor runs over, and compact_live_image
// translates every submitted slot's predicate.addr as a graph-heap address.
TEST_F(HbgGraphSubmitFailureTest, AHiddenAllocTaskLeavesItsDispatchPredicateDefined) {
    ChipTaskStorage *storage = sm_handle->header->tasks.task_storage;
    ASSERT_NE(storage, nullptr);
    // 0x4A repeated has 01 as its top two bits, so read as an address it lands at or
    // above HEAP_VIRTUAL_BASE — the quarter of the 64-bit range that the rebase would
    // mistake for a graph-heap allocation. The mirror arrives zeroed here, so an
    // undefined field is only observable once the slot is poisoned the way a reused one
    // would be.
    constexpr int kPoisonedSlots = 8;
    for (int i = 0; i < kPoisonedSlots; ++i) {
        std::memset(&storage[i].payload.predicate, 0x4A, sizeof(storage[i].payload.predicate));
    }
    ASSERT_GE(storage[0].payload.predicate.addr, HEAP_VIRTUAL_BASE);

    orch.begin_scope();
    uint32_t shape[] = {16};
    TensorCreateInfo output(shape, 1, DataType::UINT32);
    CoreTaskArgs args;
    args.add_output(output);
    const TaskOutputTensors outputs = orch.alloc_tensors(args);
    ASSERT_TRUE(outputs.task_id().is_valid());
    ASSERT_FALSE(orch.is_fatal());

    const int32_t slot = outputs.task_id().local_id();
    ASSERT_LT(slot, kPoisonedSlots) << "the submitted slot must be one this test poisoned";
    EXPECT_EQ(storage[slot].payload.predicate.op, PredicateOp::NONE);
    EXPECT_EQ(storage[slot].payload.predicate.addr, 0u);
}
