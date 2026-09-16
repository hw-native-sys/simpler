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
 * The in-graph twin of the early-dispatch qualification contract. A body is
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
 * The second fixture covers the device side of the same row: the tail-first
 * scan, the wake-scan cursor that resumes it, the row-index-to-producer mapping,
 * and the drain's single-producer fast path.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "utils/device_arena.h"
#include "host_build_graph/task_id.h"

class HbgGraphEdQualificationTest : public ::testing::Test {
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
    std::vector<TensorCreateInfo> create_infos;
    std::array<uint32_t, 16> boundary_storage{};

    static constexpr size_t HEAP_BYTES = 8 * 1024 * 1024;
    static constexpr size_t STAGING_BYTES = 256 * 1024;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES);
        create_infos.reserve(16);

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

    simpler::hbg::Tensor boundary_tensor() {
        uint32_t shape[] = {static_cast<uint32_t>(boundary_storage.size())};
        return simpler::hbg::make_tensor_external(boundary_storage.data(), shape, 1);
    }

    // Opens a recording over one boundary tensor. Returns the boundary the body's
    // tasks read from.
    simpler::hbg::Tensor begin_body(uint64_t key, GraphTaskArgs &boundary_args) {
        orch.begin_scope();
        const simpler::hbg::Tensor boundary = boundary_tensor();
        boundary_args.add_input(boundary);
        const GraphScopeResult graph = orch.graph_begin(key, boundary_args, 0);
        EXPECT_TRUE(graph.recording);
        EXPECT_NE(graph.recording_handle, nullptr);
        EXPECT_TRUE(orch.graph_prepare(graph.recording_handle, boundary_args));
        return boundary;
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
        const GraphScopeResult replay = orch.graph_begin(key, boundary_args, 0);
        EXPECT_FALSE(replay.recording) << "the body is already recorded, so this must be a cache hit";
        EXPECT_TRUE(replay.task_id.is_valid());
        return replay.task_id;
    }

    const ChipTaskSlotState &slot_of(TaskId id) {
        return orch.sm_header->tasks.get_slot_state_by_task_id(id.local_id());
    }

    const GraphDefinition *published_definition() {
        const GraphHostDefinitionList published = graph_host_definitions(*graph_state);
        EXPECT_EQ(published.entries.size(), 1u);
        if (published.entries.empty()) return nullptr;
        const GraphHostDefinition &entry = published.entries.front();
        EXPECT_NE(entry.object_offset, GRAPH_NO_OBJECT_OFFSET) << "the staging arena is sized to hold this image";
        return reinterpret_cast<const GraphDefinition *>(
            definition_staging.data() + entry.object_offset + sizeof(GraphDefinitionHeader)
        );
    }
};

TEST_F(HbgGraphEdQualificationTest, AllFlaggedProducersMakeCandidateAndSortItsRow) {
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor boundary = begin_body(0x6ED0A001, boundary_args);

    const simpler::hbg::Tensor t0 = record_task(boundary, /*flagged=*/true);
    const simpler::hbg::Tensor t1 = record_task(boundary, /*flagged=*/true);
    // Producers named in reverse record order: the sorted row must not depend on
    // the order the consumer's operands were declared in.
    ASSERT_TRUE(record_consumer(t1, t0).is_valid());
    ASSERT_TRUE(orch.graph_end());

    const GraphDefinition *definition = published_definition();
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 3);
    const auto *tasks = graph_definition_array<InGraphTaskDefinition>(*definition, definition->off_in_graph_tasks, 3);
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
    // No in-graph task reaches the device carrying the early-resolve bit.
    EXPECT_EQ(TaskAttrs{tasks[0].task_attrs}.allow_early_resolve(), false);

    ASSERT_EQ(fanin_offsets[3] - fanin_offsets[2], 2);
    EXPECT_EQ(fanin_indices[fanin_offsets[2]], 0);
    EXPECT_EQ(fanin_indices[fanin_offsets[2] + 1], 1);
}

TEST_F(HbgGraphEdQualificationTest, OneUnflaggedProducerDisqualifiesAndLeavesItsRowInRecordOrder) {
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor boundary = begin_body(0x6ED0A002, boundary_args);

    const simpler::hbg::Tensor t0 = record_task(boundary, /*flagged=*/true);
    const simpler::hbg::Tensor t1 = record_task(boundary, /*flagged=*/false);
    ASSERT_TRUE(record_consumer(t1, t0).is_valid());
    ASSERT_TRUE(orch.graph_end());

    const GraphDefinition *definition = published_definition();
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 3);
    const auto *tasks = graph_definition_array<InGraphTaskDefinition>(*definition, definition->off_in_graph_tasks, 3);
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
    const simpler::hbg::Tensor boundary = begin_body(0x6ED0A003, boundary_args);

    const simpler::hbg::Tensor flagged = record_task(boundary, /*flagged=*/true);
    const simpler::hbg::Tensor allocated = record_alloc();
    ASSERT_TRUE(record_consumer(flagged, allocated).is_valid());
    ASSERT_TRUE(orch.graph_end());

    const GraphDefinition *definition = published_definition();
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 3);
    const auto *tasks = graph_definition_array<InGraphTaskDefinition>(*definition, definition->off_in_graph_tasks, 3);
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
    const simpler::hbg::Tensor boundary = begin_body(0x6ED0B001, boundary_args);
    record_task(boundary, /*flagged=*/true);
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();

    const TaskId producer = submit_boundary_producer(boundary, /*flagged=*/true);
    const TaskId shell = submit_shell(0x6ED0B001, boundary_args);

    EXPECT_EQ(slot_of(shell).task_kind, TaskKind::GRAPH);
    EXPECT_NE(slot_of(shell).ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_NE(slot_of(producer).ed_flags & ED_FLAG_TRACKED, 0);
}

TEST_F(HbgGraphEdQualificationTest, OneUnflaggedProducerDisqualifiesTheShell) {
    GraphTaskArgs boundary_args;
    const simpler::hbg::Tensor boundary = begin_body(0x6ED0B002, boundary_args);
    record_task(boundary, /*flagged=*/true);
    ASSERT_TRUE(orch.graph_end());
    orch.graph_commit();

    const TaskId producer = submit_boundary_producer(boundary, /*flagged=*/false);
    const TaskId shell = submit_shell(0x6ED0B002, boundary_args);

    EXPECT_EQ(slot_of(shell).ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_EQ(slot_of(producer).ed_flags & ED_FLAG_TRACKED, 0);
}

// The device side of a body's fanin CSR row. The scheduler reads only the CSR
// pair and the slots it indexes, so a topology can be stated directly here
// rather than materialized from a Definition.
class HbgGraphWakeScanTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    GraphExecution execution{};
    // ChipTaskStorage holds atomics, so it is neither copyable nor movable and
    // cannot back a vector; this constructs each element in place.
    std::unique_ptr<ChipTaskStorage[]> storage;
    // The execution's readiness array, allocated with the storage it indexes.
    std::unique_ptr<std::atomic<ChipTaskState>[]> states;
    std::vector<int32_t> fanin_offsets;
    std::vector<uint16_t> fanin_indices;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        sched.seed_queue_slots();
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // `rows[i]` holds task i's producers in row order. Every task starts as a
    // pending AIV leaf, the state materialization leaves it in.
    void build(const std::vector<std::vector<uint16_t>> &rows) {
        storage = std::make_unique<ChipTaskStorage[]>(rows.size());
        states = std::make_unique<std::atomic<ChipTaskState>[]>(rows.size());
        execution.task_states = states.get();
        fanin_offsets.assign(1, 0);
        for (size_t i = 0; i < rows.size(); ++i) {
            fanin_indices.insert(fanin_indices.end(), rows[i].begin(), rows[i].end());
            fanin_offsets.push_back(static_cast<int32_t>(fanin_indices.size()));
            ChipTaskSlotState &s = storage[i].slot;
            s.reset_for_reuse();
            s.in_graph_local_id = static_cast<int32_t>(i);
            s.active_mask = ActiveMask(SUBTASK_MASK_AIV0);
            s.graph_context = &execution;
            execution.reset_task_state(static_cast<int32_t>(i));
        }
        execution.task_count = static_cast<int32_t>(rows.size());
        execution.task_storage = storage.get();
        execution.fanin_offsets = fanin_offsets.data();
        execution.fanin_indices = fanin_indices.data();
    }

    ChipTaskSlotState &slot(size_t i) { return storage[i].slot; }

    ChipTaskSlotState *pop_ready() {
        ChipTaskSlotState *out = nullptr;
        const int popped = sched.ready_queues[static_cast<int32_t>(ResourceShape::AIV)].pop_batch(&out, 1);
        return popped == 1 ? out : nullptr;
    }
};

// A wide in-graph row exceeds what a byte cursor can index. The scan must
// report the row unfinished until every producer has published: a truncated
// cursor would wrap to a low index, find that one entry published, and stage a
// candidate whose remaining producers have not.
TEST_F(HbgGraphWakeScanTest, WideRowCursorDoesNotTruncate) {
    constexpr int32_t kProducers = 300;  // > 0xFF, the old cursor width
    std::vector<std::vector<uint16_t>> rows(kProducers);
    std::vector<uint16_t> consumer_row;
    consumer_row.reserve(kProducers);
    for (int32_t i = 0; i < kProducers; ++i)
        consumer_row.push_back(static_cast<uint16_t>(i));
    rows.push_back(consumer_row);
    build(rows);

    ChipTaskSlotState &consumer = slot(kProducers);
    consumer.ed_flags |= ED_FLAG_CANDIDATE;
    for (int32_t i = 0; i < kProducers; ++i)
        slot(i).ed_flags |= ED_FLAG_TRACKED;

    // Intake hangs it on the row's tail, and the cursor must name that entry
    // rather than a wrapped one.
    ASSERT_FALSE(sched.register_on_ed_publish_list(consumer));
    EXPECT_EQ(consumer.ed_publish_scan_cursor, kProducers - 1);

    // Publishing every producer but the first leaves the row unfinished.
    for (int32_t i = kProducers - 1; i > 0; --i) {
        execution.store_published(i);
        sched.seal_ed_publish_list(slot(i));
    }
    ChipTaskSlotState *detached = nullptr;
    while (sched.ed_publish_drain_queue.pop_batch(&detached, 1) == 1) {
        if (sched.advance_ed_publish_scan(*detached)) FAIL() << "staged while producer 0 is unpublished";
    }

    execution.store_published(0);
    sched.seal_ed_publish_list(slot(0));
    ASSERT_EQ(sched.ed_publish_drain_queue.pop_batch(&detached, 1), 1);
    EXPECT_TRUE(sched.advance_ed_publish_scan(*detached));
}

// Each classification resumes where the last one hung and never re-walks the
// row's completed tail. Completion is monotone, so the resume reaches the same
// verdict a full rescan would.
TEST_F(HbgGraphWakeScanTest, CursorResumesAndNeverRewalks) {
    build({{}, {}, {}, {0, 1, 2}});
    ChipTaskSlotState &consumer = slot(3);

    // Never hung: the sentinel folds the scan start to the row's tail.
    EXPECT_EQ(consumer.wake_scan_cursor, 0xFFFF);
    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), 2);
    EXPECT_EQ(consumer.wake_scan_cursor, 2);

    // The hung-at producer completes: the rescan resumes at the cursor and walks
    // down to the next unmet row entry.
    execution.store_completed(2);
    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), 1);
    EXPECT_EQ(consumer.wake_scan_cursor, 1);

    // An entry above the cursor completing does not move it — nothing re-walks
    // that tail — and clearing the rest yields the ready verdict.
    execution.store_completed(0);
    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), 1);
    EXPECT_EQ(consumer.wake_scan_cursor, 1);
    execution.store_completed(1);
    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), -1);
}

// The classifier names a row index, not a producer index; graph_producer_at is
// what maps one to the other. A row that is not in producer order separates them.
TEST_F(HbgGraphWakeScanTest, ClassifierNamesARowIndexNotAProducerIndex) {
    build({{}, {}, {}, {2, 0, 1}});
    ChipTaskSlotState &consumer = slot(3);

    EXPECT_EQ(sched.graph_first_unmet_producer(execution, consumer), 2);
    EXPECT_EQ(&sched.graph_producer_at(execution, consumer, 2), &slot(1));
    EXPECT_EQ(&sched.graph_producer_at(execution, consumer, 0), &slot(2));
}

// A waiter with exactly one producer is by construction waiting only on the
// producer draining it, so the drain routes it without a classification pass.
TEST_F(HbgGraphWakeScanTest, SingleProducerWaiterIsRoutedByTheDrain) {
    build({{}, {0}});
    ChipTaskSlotState &producer = slot(0);
    ChipTaskSlotState &consumer = slot(1);

    sched.register_graph_wake(execution, &producer, &consumer);
    ASSERT_EQ(producer.wake_list_head.load(std::memory_order_relaxed), &consumer);

    execution.store_completed(0);
    EXPECT_EQ(sched.drain_graph_wake_list(execution, producer), 1u);
    EXPECT_EQ(pop_ready(), &consumer);
    // Decided from the row's length alone, so the never-hung sentinel stands.
    EXPECT_EQ(consumer.wake_scan_cursor, 0xFFFF);
}

// A multi-producer waiter takes the classification path instead: it re-registers
// on the row's next unmet entry and reaches the queue only once that clears too.
TEST_F(HbgGraphWakeScanTest, MultiProducerWaiterReRegistersUntilItsRowClears) {
    build({{}, {}, {0, 1}});
    ChipTaskSlotState &consumer = slot(2);

    sched.register_graph_wake(execution, &slot(1), &consumer);
    execution.store_completed(1);
    EXPECT_EQ(sched.drain_graph_wake_list(execution, slot(1)), 1u);

    // Producer 0 is still pending, so the waiter moved rather than became ready.
    EXPECT_EQ(pop_ready(), nullptr);
    EXPECT_EQ(slot(0).wake_list_head.load(std::memory_order_relaxed), &consumer);
    EXPECT_EQ(consumer.wake_scan_cursor, 0);

    execution.store_completed(0);
    EXPECT_EQ(sched.drain_graph_wake_list(execution, slot(0)), 1u);
    EXPECT_EQ(pop_ready(), &consumer);
}
