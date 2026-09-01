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
 * Acceptance for the orchestrator's transitive reduction of fanin, on both paths
 * that build a dependency edge.
 *
 * An edge P -> C carries readiness only. When another producer Q of C already
 * reaches P, the chain P -> ... -> Q -> C orders C behind P by itself and the
 * direct edge is redundant, so it is dropped: the device scans a shorter fanin
 * region and moves the consumer between fewer wake lists.
 *
 * The two paths reduce at different resolutions, on purpose. The global submit
 * path runs per task on an unbounded table, so it carries one word of ancestors
 * and proves coverage within FANIN_REACH_WINDOW ids. A recorded Graph body is
 * capped at MAX_IN_GRAPH_TASKS and recorded once before any number of replays, so
 * it carries an exact closure and has no window at all.
 *
 * Every case drives the real path and reads the published edges back — out of
 * shared memory for a global task, out of the committed Definition's CSR for a
 * recorded one — so what is asserted is what the device would see. The properties
 * that must hold across all of them:
 *
 *   - reachability is preserved (a dropped producer is still ordered before the
 *     consumer through a surviving edge), so no task can start early;
 *   - retention is untouched — last_consumer_local_id still names the consumer of
 *     a dropped edge, which is what gates the host's overwrite of that buffer;
 *   - a global producer beyond the window keeps its edge, because neither it nor
 *     its ancestors are representable in one word.
 */

#include <gtest/gtest.h>

#include <array>
#include <cstddef>
#include <cstdint>
#include <vector>

#include "graph_execution.h"
#include "graph_host_state.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/task_id_encoding.h"
#include "utils/device_arena.h"

class HbgFaninReductionTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    static constexpr size_t HEAP_BYTES = 64 * 1024;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES);

        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        sched.seed_queue_slots();
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), HEAP_BYTES, CHIP_DEFAULT_GRAPH_TASKS, &sched));
        orch.begin_scope();
    }

    void TearDown() override {
        orch.end_scope();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // A kernel-less task with exactly the given producers. Explicit dependencies
    // are the only fanin source here, so each case states its DAG outright rather
    // than through tensor aliasing.
    TaskId submit(const std::vector<TaskId> &deps) {
        CoreTaskArgs args;
        if (!deps.empty()) {
            args.set_dependencies(deps.data(), static_cast<uint32_t>(deps.size()));
        }
        const TaskOutputTensors result = orch.submit_dummy_task(args);
        EXPECT_TRUE(result.task_id().is_valid());
        return result.task_id();
    }

    static int32_t local(TaskId id) { return static_cast<int32_t>(simpler::hbg::task_local_id(id)); }

    // The fanin region as the device boot scan would read it.
    std::vector<int32_t> fanin_of(TaskId id) const {
        const TaskPayload &payload = sm_handle->header->tasks.task_payloads[local(id)];
        const int32_t *slots = payload.fanin_data();
        return std::vector<int32_t>(slots, slots + payload.fanin_count);
    }

    int32_t last_consumer_of(TaskId id) const {
        return sm_handle->header->tasks.get_slot_state_by_task_id(local(id)).last_consumer_local_id;
    }
};

// The base case: C depends on A and on B, and B already depends on A. B alone
// orders C behind A, so the direct A -> C edge goes.
TEST_F(HbgFaninReductionTest, DiamondDropsTheCoveredEdge) {
    const TaskId a = submit({});
    const TaskId b = submit({a});
    const TaskId c = submit({a, b});

    EXPECT_EQ(fanin_of(c), std::vector<int32_t>{local(b)});
}

// Coverage is transitive, not one hop: the ancestor word a producer publishes is
// already its own closure, so a chain of any length inside the window collapses.
TEST_F(HbgFaninReductionTest, MultiHopChainCoversTheDirectEdge) {
    const TaskId a = submit({});
    const TaskId b = submit({a});
    const TaskId c = submit({b});
    const TaskId d = submit({a, b, c});

    EXPECT_EQ(fanin_of(d), std::vector<int32_t>{local(c)});
}

// Nothing is dropped without a proof: two producers with no path between them
// both gate the consumer and both survive, in the order they were appended.
TEST_F(HbgFaninReductionTest, IndependentProducersBothSurvive) {
    const TaskId a = submit({});
    const TaskId b = submit({});
    const TaskId c = submit({a, b});

    const std::vector<int32_t> expected{local(a), local(b)};
    EXPECT_EQ(fanin_of(c), expected);
}

// A dropped edge is a readiness edge only. The producer's reclaim gate still
// names the consumer, so the host cannot overwrite that buffer until the consumer
// that reads it has retired.
TEST_F(HbgFaninReductionTest, DroppedEdgeKeepsTheProducerReclaimGate) {
    const TaskId a = submit({});
    const TaskId b = submit({a});
    const TaskId c = submit({a, b});

    ASSERT_EQ(fanin_of(c), std::vector<int32_t>{local(b)});
    EXPECT_EQ(last_consumer_of(a), local(c));
}

// Every producer covered by another still leaves one maximal element, so a task
// that had producers never becomes a root. C's three producers form a chain, and
// only its head survives.
TEST_F(HbgFaninReductionTest, ChainedCoverageStillLeavesOneEdge) {
    const TaskId a = submit({});
    const TaskId b = submit({a});
    const TaskId c = submit({b});
    const TaskId d = submit({c});
    const TaskId e = submit({a, b, c, d});

    EXPECT_EQ(fanin_of(e), std::vector<int32_t>{local(d)});
}

// Reduction is bounded by one word of ancestors. A producer further back than
// FANIN_REACH_WINDOW is unrepresentable, so its edge is kept even though a chain
// covers it — conservative, never wrong.
TEST_F(HbgFaninReductionTest, ProducerBeyondTheWindowKeepsItsEdge) {
    const TaskId root = submit({});
    TaskId prev = root;
    // Push root exactly FANIN_REACH_WINDOW + 1 ids back from the consumer, while
    // keeping an unbroken chain from it.
    for (int32_t i = 0; i < FANIN_REACH_WINDOW; ++i) {
        prev = submit({prev});
    }
    const TaskId consumer = submit({root, prev});

    ASSERT_EQ(local(consumer) - local(root), FANIN_REACH_WINDOW + 1);
    const std::vector<int32_t> expected{local(root), local(prev)};
    EXPECT_EQ(fanin_of(consumer), expected);
}

// The same chain one task shorter puts the root at exactly the window edge, where
// it is representable and the chain does cover it.
TEST_F(HbgFaninReductionTest, ProducerAtTheWindowEdgeIsStillReduced) {
    const TaskId root = submit({});
    TaskId prev = root;
    for (int32_t i = 0; i < FANIN_REACH_WINDOW - 1; ++i) {
        prev = submit({prev});
    }
    const TaskId consumer = submit({root, prev});

    ASSERT_EQ(local(consumer) - local(root), FANIN_REACH_WINDOW);
    EXPECT_EQ(fanin_of(consumer), std::vector<int32_t>{local(prev)});
}

// Surviving edges keep their append order: classify_fanin_state scans the region
// back-to-front for the latest-submitted unmet producer, so compaction must not
// reorder what it leaves behind.
TEST_F(HbgFaninReductionTest, CompactionPreservesAppendOrder) {
    const TaskId a = submit({});
    const TaskId b = submit({});
    const TaskId covered = submit({a});
    const TaskId c = submit({b, covered, a});

    // `a` is covered by `covered`; `b` is independent. Both survivors stay in the
    // order set_dependencies listed them.
    const std::vector<int32_t> expected{local(b), local(covered)};
    EXPECT_EQ(fanin_of(c), expected);
}

// ---------------------------------------------------------------------------
// Recorded Graph bodies
// ---------------------------------------------------------------------------

// A recorded body reduces into the Definition's own fanin CSR, which is what every
// replay of that Graph reads. The recording is driven end to end and the committed
// Definition is read back, so these assert the shipped edge set rather than the
// recorder's scratch.
class HbgRecordedFaninReductionTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    GraphHostStatePtr graph_state;
    GraphDefinitionArena arena{};
    std::vector<char> gm_heap;
    std::vector<std::byte> staging;

    static constexpr size_t HEAP_BYTES = 256 * 1024;
    static constexpr size_t STAGING_BYTES = 512 * 1024;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(HEAP_BYTES);

        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);
        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        sched.seed_queue_slots();
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), HEAP_BYTES, CHIP_DEFAULT_GRAPH_TASKS, &sched));

        staging.assign(STAGING_BYTES, std::byte{0});
        arena = GraphDefinitionArena{};
        arena.base = staging.data();
        arena.capacity = staging.size();
        arena.object_prefix_bytes = sizeof(GraphDefinitionHeader);
        arena.object_align = GRAPH_DEFINITION_OBJECT_ALIGN;
        graph_state = make_graph_host_state(arena);
        ASSERT_NE(graph_state, nullptr);
        orch.graph_host_state = graph_state.get();
        orch.begin_scope();
    }

    void TearDown() override {
        orch.end_scope();
        orch.graph_host_state = nullptr;
        graph_state.reset();
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // Record one body under `graph_key`, driven by `body`, then commit so the
    // Definition is built. The recorder is normally a worker thread; this plays
    // both roles on one thread, as the other Graph tests do.
    template <typename Fn>
    void record(uint64_t graph_key, const simpler::hbg::Tensor &boundary, Fn &&body) {
        GraphTaskArgs boundary_args;
        boundary_args.add_input(boundary);
        const GraphScopeResult scope = orch.graph_begin(graph_key, boundary_args, 0x1736);
        ASSERT_TRUE(scope.recording);
        ASSERT_TRUE(orch.graph_prepare(scope.recording_handle, boundary_args));
        body();
        ASSERT_TRUE(orch.graph_end());
        orch.graph_commit();
        ASSERT_FALSE(orch.fatal);
    }

    const GraphDefinition *only_definition() const {
        const GraphHostDefinitionList definitions = graph_host_definitions(*graph_state);
        if (definitions.entries.size() != 1) return nullptr;
        const GraphHostDefinition &entry = definitions.entries[0];
        const std::byte *image =
            entry.spill != nullptr ? entry.spill : arena.base + entry.object_offset + arena.object_prefix_bytes;
        return reinterpret_cast<const GraphDefinition *>(image);
    }

    // One in-graph task's producers, straight out of the CSR a replay walks.
    static std::vector<uint16_t> csr_fanin_of(const GraphDefinition &definition, uint32_t task) {
        const auto *offsets =
            graph_definition_array<uint32_t>(definition, definition.off_fanin_offsets, definition.task_count + 1);
        const auto *indices =
            graph_definition_array<uint16_t>(definition, definition.off_fanin_indices, definition.edge_count);
        if (offsets == nullptr || (definition.edge_count != 0 && indices == nullptr)) return {};
        return std::vector<uint16_t>(indices + offsets[task], indices + offsets[task + 1]);
    }

    // A kernel-less in-graph task depending on the given in-graph tasks.
    static TaskId body_task(OrchestratorState &orch, const std::vector<TaskId> &deps) {
        CoreTaskArgs args;
        if (!deps.empty()) {
            args.set_dependencies(deps.data(), static_cast<uint32_t>(deps.size()));
        }
        const TaskOutputTensors result = orch.submit_dummy_task(args);
        EXPECT_TRUE(result.task_id().is_valid());
        return result.task_id();
    }
};

// The diamond again, this time inside a body: the Definition ships one edge into
// the consumer, so every replay of this Graph walks the shorter CSR.
TEST_F(HbgRecordedFaninReductionTest, DiamondInsideABodyShipsTheReducedCsr) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    const simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    record(0xFA1EDA01, boundary, [&] {
        const TaskId a = body_task(orch, {});
        const TaskId b = body_task(orch, {a});
        body_task(orch, {a, b});
    });

    const GraphDefinition *definition = only_definition();
    ASSERT_NE(definition, nullptr);
    ASSERT_EQ(definition->task_count, 3u);
    EXPECT_EQ(csr_fanin_of(*definition, 2), std::vector<uint16_t>{1});
    // Two edges recorded (a->b, b->c); the direct a->c is gone, so the CSR is not
    // merely reordered — it is smaller.
    EXPECT_EQ(definition->edge_count, 2u);
}

// The recording path carries an exact closure, not a window, so a producer at any
// distance inside the body is reduced. A chain longer than FANIN_REACH_WINDOW would
// defeat the global path's one word and is covered here.
TEST_F(HbgRecordedFaninReductionTest, AChainLongerThanTheGlobalWindowIsStillReduced) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    const simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    constexpr int32_t CHAIN = FANIN_REACH_WINDOW + 8;
    record(0xFA1EDA02, boundary, [&] {
        const TaskId root = body_task(orch, {});
        TaskId prev = root;
        for (int32_t i = 0; i < CHAIN; ++i) {
            prev = body_task(orch, {prev});
        }
        body_task(orch, {root, prev});
    });

    const GraphDefinition *definition = only_definition();
    ASSERT_NE(definition, nullptr);
    const uint32_t consumer = definition->task_count - 1;
    ASSERT_EQ(definition->task_count, static_cast<uint32_t>(CHAIN) + 2u);
    // The root sits CHAIN + 1 tasks back — past what one word could represent — and
    // the chain still covers it.
    EXPECT_EQ(csr_fanin_of(*definition, consumer), std::vector<uint16_t>{static_cast<uint16_t>(consumer - 1)});
}

// Independent producers inside a body both survive, and in recording order.
TEST_F(HbgRecordedFaninReductionTest, IndependentBodyProducersBothSurvive) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    const simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    record(0xFA1EDA03, boundary, [&] {
        const TaskId a = body_task(orch, {});
        const TaskId b = body_task(orch, {});
        body_task(orch, {a, b});
    });

    const GraphDefinition *definition = only_definition();
    ASSERT_NE(definition, nullptr);
    const std::vector<uint16_t> expected{0, 1};
    EXPECT_EQ(csr_fanin_of(*definition, 2), expected);
}

// A body root has no internal producers — its ordering comes from the outer shell's
// own fanin — and reduction leaves that untouched, so the Definition still names it
// a root.
TEST_F(HbgRecordedFaninReductionTest, BodyRootsAreLeftAlone) {
    std::array<uint32_t, 16> storage{};
    uint32_t shape[] = {static_cast<uint32_t>(storage.size())};
    const simpler::hbg::Tensor boundary = simpler::hbg::make_tensor_external(storage.data(), shape, 1);

    record(0xFA1EDA04, boundary, [&] {
        const TaskId a = body_task(orch, {});
        body_task(orch, {a});
    });

    const GraphDefinition *definition = only_definition();
    ASSERT_NE(definition, nullptr);
    EXPECT_TRUE(csr_fanin_of(*definition, 0).empty());
    EXPECT_EQ(definition->root_count, 1u);
}
