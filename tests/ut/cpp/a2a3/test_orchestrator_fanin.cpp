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
add_runtime_output_arg(CoreTaskArgs &args, std::vector<TensorCreateInfo> &create_infos, uint32_t float_count) {
    uint32_t shape[] = {float_count};
    create_infos.emplace_back(shape, 1, DataType::FLOAT32);
    args.add_output(create_infos.back());
}

TEST_F(OrchestratorFaninTest, DuplicateExplicitProducerAddsOneFanin) {
    orch.begin_scope();

    CoreTaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    PTO2TaskId deps[] = {producer.task_id(), producer.task_id()};
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies(deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot =
        sm_handle->header->rings[producer.task_id().ring()].get_slot_state_by_task_id(producer.task_id().local());
    auto &consumer_slot =
        sm_handle->header->rings[consumer.task_id().ring()].get_slot_state_by_task_id(consumer.task_id().local());

    ASSERT_NE(consumer_slot.payload, nullptr);
    EXPECT_EQ(consumer_slot.payload->fanin_actual_count, 1);
    EXPECT_EQ(consumer_slot.payload->fanin_inline_edges[0].slot_state(), &producer_slot);
    // A plain set_dependencies() dep is conservative RETAIN: DEP_WAIT|DEP_RETAIN.
    EXPECT_EQ(consumer_slot.payload->fanin_inline_edges[0].flags(), DEP_WAIT | DEP_RETAIN);
    // fanout_count is bit-packed: bit31 (PTO2_FANOUT_SCOPE_BIT) is the owning-scope
    // ref, low bits the consumer count. The duplicate explicit dep is deduped to a
    // single consumer, so this is scope + 1.
    EXPECT_EQ(producer_slot.fanout_count, PTO2_FANOUT_SCOPE_BIT + 1);
}

// An explicit ordering-only dep (the primitive add_dep_wait() lowers to) yields a
// DEP_WAIT edge, not the conservative DEP_WAIT|DEP_RETAIN default.
TEST_F(OrchestratorFaninTest, ExplicitWaitDepProducesWaitOnlyEdge) {
    orch.begin_scope();

    CoreTaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    PTO2TaskId deps[] = {producer.task_id()};
    DepFlags kinds[] = {DEP_WAIT};
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies_with_kinds(deps, kinds, 1);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &consumer_slot =
        sm_handle->header->rings[consumer.task_id().ring()].get_slot_state_by_task_id(consumer.task_id().local());
    ASSERT_NE(consumer_slot.payload, nullptr);
    ASSERT_EQ(consumer_slot.payload->fanin_actual_count, 1);
    EXPECT_EQ(consumer_slot.payload->fanin_inline_edges[0].flags(), DEP_WAIT);
}

// The same producer reached with different kinds OR-accumulates into one edge:
// WAIT-only first, then WAIT|RETAIN folds RETAIN in, claiming exactly one pin.
TEST_F(OrchestratorFaninTest, DuplicateProducerOrAccumulatesFlags) {
    orch.begin_scope();

    CoreTaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    PTO2TaskId deps[] = {producer.task_id(), producer.task_id()};
    DepFlags kinds[] = {DEP_WAIT, DEP_WAIT | DEP_RETAIN};
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies_with_kinds(deps, kinds, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot =
        sm_handle->header->rings[producer.task_id().ring()].get_slot_state_by_task_id(producer.task_id().local());
    auto &consumer_slot =
        sm_handle->header->rings[consumer.task_id().ring()].get_slot_state_by_task_id(consumer.task_id().local());
    ASSERT_NE(consumer_slot.payload, nullptr);
    ASSERT_EQ(consumer_slot.payload->fanin_actual_count, 1);
    EXPECT_EQ(consumer_slot.payload->fanin_inline_edges[0].flags(), DEP_WAIT | DEP_RETAIN);
    EXPECT_EQ(producer_slot.fanout_count, PTO2_FANOUT_SCOPE_BIT + 1);
}

// The duplicate lands in the spill region (>64 fanin), exercising
// or_flags_into_existing's spill lookup: the dup folds (65 edges, not 66), claims
// exactly one pin, and OR-accumulates its flags into the spilled edge.
TEST_F(OrchestratorFaninTest, DuplicateProducerInSpillRegionDedups) {
    orch.begin_scope();

    constexpr int kProducers = PTO2_FANIN_INLINE_CAP + 1;  // 65: the last one spills
    std::vector<TaskOutputTensors> producers;
    producers.reserve(kProducers);
    for (int i = 0; i < kProducers; i++) {
        CoreTaskArgs a;
        producers.push_back(orch.submit_dummy_task(a));
        ASSERT_TRUE(producers.back().task_id().is_valid());
    }

    std::vector<PTO2TaskId> deps;
    std::vector<DepFlags> kinds;
    deps.reserve(kProducers + 1);
    kinds.reserve(kProducers + 1);
    for (auto &p : producers) {
        deps.push_back(p.task_id());
        kinds.push_back(DEP_WAIT);  // the 65th (first spill edge) starts WAIT-only
    }
    deps.push_back(producers.back().task_id());  // duplicate the spilled 65th ...
    kinds.push_back(DEP_WAIT | DEP_RETAIN);      // ... contributing RETAIN via the fold

    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies_with_kinds(deps.data(), kinds.data(), static_cast<uint32_t>(deps.size()));
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &consumer_slot =
        sm_handle->header->rings[consumer.task_id().ring()].get_slot_state_by_task_id(consumer.task_id().local());
    ASSERT_NE(consumer_slot.payload, nullptr);
    PTO2TaskPayload *payload = consumer_slot.payload;
    EXPECT_EQ(payload->fanin_actual_count, kProducers);  // duplicate folded, not 66

    PTO2TaskId dup = producers.back().task_id();
    auto &dup_slot = sm_handle->header->rings[dup.ring()].get_slot_state_by_task_id(dup.local());
    EXPECT_EQ(dup_slot.fanout_count, PTO2_FANOUT_SCOPE_BIT + 1);  // one pin, not two

    // The first spilled edge is the duplicated producer; its flags OR-folded to
    // WAIT|RETAIN across the two discovery kinds.
    ASSERT_NE(payload->fanin_spill_pool, nullptr);
    PTO2FaninPool &spill_pool = *payload->fanin_spill_pool;
    PTO2FaninSpillEntry &spill_edge = spill_pool.base[payload->fanin_spill_start % spill_pool.capacity];
    EXPECT_EQ(spill_edge.slot_state(), &dup_slot);
    EXPECT_EQ(spill_edge.flags(), DEP_WAIT | DEP_RETAIN);
}

// The all-completed fast path (wire_fanin_task skipped) still drops an
// ordering-only producer's submit->wire pin.
TEST_F(OrchestratorFaninTest, AllCompletedFastPathReleasesWaitOnlyPin) {
    orch.begin_scope();

    CoreTaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());
    auto &producer_slot =
        sm_handle->header->rings[producer.task_id().ring()].get_slot_state_by_task_id(producer.task_id().local());
    // COMPLETED but not consumed (the open scope still pins it): the consumer takes
    // the all-completed fast path.
    producer_slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    int32_t rc_before = producer_slot.fanout_refcount.load();

    PTO2TaskId deps[] = {producer.task_id()};
    DepFlags kinds[] = {DEP_WAIT};  // ordering-only
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies_with_kinds(deps, kinds, 1);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    // The fast path released the ordering-only pin.
    EXPECT_EQ(producer_slot.fanout_refcount.load(), rc_before + 1);
}

// 1-hop transitive reduction: A -> B -> C plus a direct A -> C (both retention).
// The direct A -> C WAIT is ordered via A -> B -> C, so reduction clears it,
// leaving A -> C RETAIN-only while B -> C stays WAIT|RETAIN.
TEST_F(OrchestratorFaninTest, DiamondReducesRedundantWaitToRetainOnly) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());

    PTO2TaskId b_deps[] = {a.task_id()};  // A -> B
    CoreTaskArgs b_args;
    b_args.set_dependencies(b_deps, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    PTO2TaskId c_deps[] = {a.task_id(), b.task_id()};  // A -> C, B -> C
    CoreTaskArgs c_args;
    c_args.set_dependencies(c_deps, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = sm_handle->header->rings[a.task_id().ring()].get_slot_state_by_task_id(a.task_id().local());
    auto &b_slot = sm_handle->header->rings[b.task_id().ring()].get_slot_state_by_task_id(b.task_id().local());
    auto &c_slot = sm_handle->header->rings[c.task_id().ring()].get_slot_state_by_task_id(c.task_id().local());
    ASSERT_NE(c_slot.payload, nullptr);
    EXPECT_EQ(c_slot.payload->fanin_actual_count, 2);  // both edges kept in storage
    EXPECT_EQ(c_slot.payload->fanin_wait_count, 1);    // only B gates readiness after reduction

    DepFlags a_flags = DEP_NONE, b_flags = DEP_NONE;
    for (int i = 0; i < c_slot.payload->fanin_actual_count; i++) {
        auto &e = c_slot.payload->fanin_inline_edges[i];
        if (e.slot_state() == &a_slot) a_flags = e.flags();
        else if (e.slot_state() == &b_slot) b_flags = e.flags();
    }
    EXPECT_EQ(a_flags, DEP_RETAIN);             // A -> C reduced to RETAIN-only (lifetime kept)
    EXPECT_EQ(b_flags, DEP_WAIT | DEP_RETAIN);  // B -> C unchanged
}

// Same diamond but the direct A -> C is ordering-only (add_dep_wait). Reduction
// clears its only flag, dropping the edge to DEP_NONE.
TEST_F(OrchestratorFaninTest, DiamondDropsRedundantWaitOnlyEdge) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());

    PTO2TaskId b_deps[] = {a.task_id()};  // A -> B (WAIT|RETAIN)
    CoreTaskArgs b_args;
    b_args.set_dependencies(b_deps, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    PTO2TaskId c_deps[] = {a.task_id(), b.task_id()};
    DepFlags c_kinds[] = {DEP_WAIT, DEP_WAIT | DEP_RETAIN};  // A -> C ordering-only, B -> C retain
    CoreTaskArgs c_args;
    c_args.set_dependencies_with_kinds(c_deps, c_kinds, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = sm_handle->header->rings[a.task_id().ring()].get_slot_state_by_task_id(a.task_id().local());
    auto &c_slot = sm_handle->header->rings[c.task_id().ring()].get_slot_state_by_task_id(c.task_id().local());
    ASSERT_NE(c_slot.payload, nullptr);
    EXPECT_EQ(c_slot.payload->fanin_wait_count, 1);  // only B gates readiness

    DepFlags a_flags = DEP_WAIT;  // sentinel != NONE
    for (int i = 0; i < c_slot.payload->fanin_actual_count; i++) {
        auto &e = c_slot.payload->fanin_inline_edges[i];
        if (e.slot_state() == &a_slot) a_flags = e.flags();
    }
    EXPECT_EQ(a_flags, DEP_NONE);  // A -> C dropped entirely
}

// A -> C with no bypassing producer is not reduced (no A -> ? -> C path).
TEST_F(OrchestratorFaninTest, IndependentProducersAreNotReduced) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    CoreTaskArgs b_args;
    TaskOutputTensors b = orch.submit_dummy_task(b_args);  // independent of A
    ASSERT_TRUE(b.task_id().is_valid());

    PTO2TaskId c_deps[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(c_deps, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &c_slot = sm_handle->header->rings[c.task_id().ring()].get_slot_state_by_task_id(c.task_id().local());
    ASSERT_NE(c_slot.payload, nullptr);
    EXPECT_EQ(c_slot.payload->fanin_actual_count, 2);
    EXPECT_EQ(c_slot.payload->fanin_wait_count, 2);  // no reduction: both keep WAIT
}

// The covering edge A -> B must carry RETAIN for A -> C to be reduced. With A -> B
// ordering-only, A's pin is released at B's wiring and its slot may be rebound, so
// the (ring, slot) identity check cannot prove A -> B -> C still holds. A -> C keeps
// its WAIT — a redundant edge conservatively retained, never an ordering edge lost.
TEST_F(OrchestratorFaninTest, WaitOnlyCoveringEdgeDoesNotReduce) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());

    PTO2TaskId b_deps[] = {a.task_id()};
    DepFlags b_kinds[] = {DEP_WAIT};  // A -> B ordering-only (no RETAIN)
    CoreTaskArgs b_args;
    b_args.set_dependencies_with_kinds(b_deps, b_kinds, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    PTO2TaskId c_deps[] = {a.task_id(), b.task_id()};  // A -> C, B -> C
    CoreTaskArgs c_args;
    c_args.set_dependencies(c_deps, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = sm_handle->header->rings[a.task_id().ring()].get_slot_state_by_task_id(a.task_id().local());
    auto &c_slot = sm_handle->header->rings[c.task_id().ring()].get_slot_state_by_task_id(c.task_id().local());
    ASSERT_NE(c_slot.payload, nullptr);
    EXPECT_EQ(c_slot.payload->fanin_wait_count, 2);  // A -> C not reduced away

    DepFlags a_flags = DEP_NONE;
    for (int i = 0; i < c_slot.payload->fanin_actual_count; i++) {
        auto &e = c_slot.payload->fanin_inline_edges[i];
        if (e.slot_state() == &a_slot) a_flags = e.flags();
    }
    EXPECT_EQ(a_flags, DEP_WAIT | DEP_RETAIN);  // A -> C keeps its WAIT
}

// A completed covering producer B is not used for reduction: once B has run
// on_task_release its fanin pointer to A may resolve to a reused slot, so the
// (ring, slot) identity check is no longer generation-safe. A -> C keeps its WAIT.
TEST_F(OrchestratorFaninTest, CompletedCoveringProducerIsNotUsed) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());

    PTO2TaskId b_deps[] = {a.task_id()};  // A -> B (WAIT|RETAIN)
    CoreTaskArgs b_args;
    b_args.set_dependencies(b_deps, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    // Drive B past execution so reduce_wait_edges must treat its fanin as stale.
    auto &b_slot = sm_handle->header->rings[b.task_id().ring()].get_slot_state_by_task_id(b.task_id().local());
    b_slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);

    PTO2TaskId c_deps[] = {a.task_id(), b.task_id()};  // A -> C, B -> C
    CoreTaskArgs c_args;
    c_args.set_dependencies(c_deps, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = sm_handle->header->rings[a.task_id().ring()].get_slot_state_by_task_id(a.task_id().local());
    auto &c_slot = sm_handle->header->rings[c.task_id().ring()].get_slot_state_by_task_id(c.task_id().local());
    ASSERT_NE(c_slot.payload, nullptr);

    DepFlags a_flags = DEP_NONE;
    for (int i = 0; i < c_slot.payload->fanin_actual_count; i++) {
        auto &e = c_slot.payload->fanin_inline_edges[i];
        if (e.slot_state() == &a_slot) a_flags = e.flags();
    }
    EXPECT_EQ(a_flags, DEP_WAIT | DEP_RETAIN);  // completed B not used as cover; A -> C keeps WAIT
}

// Acceptance #2 lifetime: a reduced RETAIN-only A -> C keeps A pinned until C
// releases it, not merely until wiring. A WAIT-only edge would have dropped its
// pin at wiring; here A's pin survives and is dropped only by on_task_release(C).
TEST_F(OrchestratorFaninTest, ReducedRetainOnlyEdgeHoldsProducerUntilConsumerRelease) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());

    PTO2TaskId b_deps[] = {a.task_id()};  // A -> B (WAIT|RETAIN)
    CoreTaskArgs b_args;
    b_args.set_dependencies(b_deps, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    PTO2TaskId c_deps[] = {a.task_id(), b.task_id()};  // A -> C, B -> C
    CoreTaskArgs c_args;
    c_args.set_dependencies(c_deps, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = sm_handle->header->rings[a.task_id().ring()].get_slot_state_by_task_id(a.task_id().local());
    auto &c_slot = sm_handle->header->rings[c.task_id().ring()].get_slot_state_by_task_id(c.task_id().local());

    // A -> C was reduced to RETAIN-only, so C's submit->wire pin on A is still held
    // after wiring — a WAIT-only edge would have released it already.
    DepFlags a_flags = DEP_NONE;
    for (int i = 0; i < c_slot.payload->fanin_actual_count; i++) {
        auto &e = c_slot.payload->fanin_inline_edges[i];
        if (e.slot_state() == &a_slot) a_flags = e.flags();
    }
    ASSERT_EQ(a_flags, DEP_RETAIN);

    int32_t rc_before = a_slot.fanout_refcount.load();
    c_slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    sched.on_task_release(c_slot);
    // on_task_release(C) is what finally drops the retained pin on A — proving A
    // outlived wiring and stays alive until its consumer releases it.
    EXPECT_EQ(a_slot.fanout_refcount.load(), rc_before + 1);
}

TEST_F(OrchestratorFaninTest, SubmitPathHeapDeadlockLogReportsRingAndRealHeapState) {
    std::vector<TensorCreateInfo> create_infos;
    create_infos.reserve(8);

    orch.begin_scope();
    orch.begin_scope();
    ASSERT_EQ(orch.current_ring_id(), 1);

    CoreTaskArgs first_args;
    add_runtime_output_arg(first_args, create_infos, 1024);  // 4096 bytes
    TaskOutputTensors first = orch.submit_dummy_task(first_args);
    ASSERT_TRUE(first.task_id().is_valid());

    auto &ring = sm_handle->header->rings[1];
    auto &first_slot = ring.get_slot_state_by_task_id(static_cast<int32_t>(first.task_id().local()));
    orch.end_scope();
    first_slot.task_state.store(PTO2_TASK_COMPLETED, std::memory_order_release);
    sched.check_and_handle_consumed(first_slot);
    ASSERT_EQ(ring.fc.last_task_alive.load(std::memory_order_acquire), 1);

    orch.begin_scope();
    ASSERT_EQ(orch.current_ring_id(), 1);

    CoreTaskArgs wrap_args;
    add_runtime_output_arg(wrap_args, create_infos, 1);  // wraps, packed to 1024 bytes
    TaskOutputTensors wrapped = orch.submit_dummy_task(wrap_args);
    ASSERT_TRUE(wrapped.task_id().is_valid());

    CoreTaskArgs fill_args;
    add_runtime_output_arg(fill_args, create_infos, 512);  // 2048 bytes
    TaskOutputTensors filled = orch.submit_dummy_task(fill_args);
    ASSERT_TRUE(filled.task_id().is_valid());
    ASSERT_EQ(orch.rings[1].task_allocator.heap_used_bytes(), 3072ULL);
    ASSERT_EQ(orch.rings[1].task_allocator.heap_available(), 1024ULL);

    CoreTaskArgs blocked_args;
    add_runtime_output_arg(blocked_args, create_infos, 1);
    testing::internal::CaptureStderr();
    TaskOutputTensors blocked = orch.submit_dummy_task(blocked_args);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(blocked.task_id().is_valid());
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_NE(log.find("FATAL: Task Allocator Deadlock - Heap Exhausted! ring=1"), std::string::npos);
    EXPECT_NE(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
    EXPECT_NE(log.find("Heap ring 1:"), std::string::npos);
    EXPECT_NE(log.find("used=3072"), std::string::npos);
    EXPECT_NE(log.find("available=1024"), std::string::npos);
    EXPECT_EQ(log.find("PTO2_RING_HEAP=<pow2>"), std::string::npos);
}

TEST_F(OrchestratorFaninTest, StructuralCheckRejectsOpenAncestorWhenNestedScopesShareRing) {
    std::vector<TensorCreateInfo> create_infos;
    create_infos.reserve(2);

    for (int32_t depth = 0; depth < PTO2_MAX_RING_DEPTH; ++depth) {
        orch.begin_scope();
    }
    ASSERT_EQ(orch.current_ring_id(), PTO2_MAX_RING_DEPTH - 1);

    CoreTaskArgs parent_args;
    add_runtime_output_arg(parent_args, create_infos, 1024);
    TaskOutputTensors parent = orch.submit_dummy_task(parent_args);
    ASSERT_TRUE(parent.task_id().is_valid());

    orch.begin_scope();
    ASSERT_EQ(orch.current_ring_id(), PTO2_MAX_RING_DEPTH - 1);

    CoreTaskArgs child_args;
    add_runtime_output_arg(child_args, create_infos, 1);
    testing::internal::CaptureStderr();
    TaskOutputTensors child = orch.submit_dummy_task(child_args);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(child.task_id().is_valid());
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_NE(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
}

TEST_F(OrchestratorFaninTest, ClosedChildHeadUsesTimeoutWithOpenParentOnSharedRing) {
    std::vector<TensorCreateInfo> create_infos;
    create_infos.reserve(3);

    for (int32_t depth = 0; depth < PTO2_MAX_RING_DEPTH; ++depth) {
        orch.begin_scope();
    }
    orch.begin_scope();
    ASSERT_EQ(orch.current_ring_id(), PTO2_MAX_RING_DEPTH - 1);

    CoreTaskArgs child_args;
    add_runtime_output_arg(child_args, create_infos, 768);
    TaskOutputTensors child = orch.submit_dummy_task(child_args);
    ASSERT_TRUE(child.task_id().is_valid());

    orch.end_scope();
    ASSERT_EQ(orch.current_ring_id(), PTO2_MAX_RING_DEPTH - 1);

    CoreTaskArgs parent_args;
    add_runtime_output_arg(parent_args, create_infos, 256);
    TaskOutputTensors parent = orch.submit_dummy_task(parent_args);
    ASSERT_TRUE(parent.task_id().is_valid());

    CoreTaskArgs blocked_args;
    add_runtime_output_arg(blocked_args, create_infos, 1);
    testing::internal::CaptureStderr();
    TaskOutputTensors blocked = orch.submit_dummy_task(blocked_args);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(blocked.task_id().is_valid());
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), PTO2_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_NE(log.find("No reclaim progress for ~500 ms"), std::string::npos);
    EXPECT_EQ(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
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
