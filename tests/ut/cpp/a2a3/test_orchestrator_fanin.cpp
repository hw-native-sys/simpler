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
#include <limits>
#include <string>
#include <vector>

#include "utils/device_arena.h"
#include "orchestrator.h"
#include "shared_memory.h"
#include "tensormap_and_ringbuffer/task_id_encoding.h"

class OrchestratorFaninTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    OrchestratorLayout orch_layout{};
    SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(4096 * CHIP_MAX_RING_DEPTH);

        int32_t task_window_sizes[CHIP_MAX_RING_DEPTH];
        for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
            task_window_sizes[r] = static_cast<int32_t>(CHIP_TASK_WINDOW_SIZE);
        }

        orch_layout = OrchestratorState::reserve_layout(runtime_arena, task_window_sizes);
        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(orch.init_data_from_layout(
            orch_layout, runtime_arena, sm_handle->sm_base, gm_heap.data(), 4096, CHIP_TASK_WINDOW_SIZE
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

    TaskId deps[] = {producer.task_id(), producer.task_id()};
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies(deps, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot =
        sm_handle->header->rings[simpler::tmr::task_ring(producer.task_id())].get_slot_state_by_task_id(
            simpler::tmr::task_local_id(producer.task_id())
        );
    auto &consumer_slot =
        sm_handle->header->rings[simpler::tmr::task_ring(consumer.task_id())].get_slot_state_by_task_id(
            simpler::tmr::task_local_id(consumer.task_id())
        );

    ASSERT_NE(consumer_slot.payload, nullptr);
    EXPECT_EQ(consumer_slot.payload->fanin_actual_count, 1);
    EXPECT_EQ(consumer_slot.payload->fanin_inline_edges[0].slot_state(), &producer_slot);
    // A plain set_dependencies() dep is conservative RETAIN: DEP_WAIT|DEP_RETAIN.
    EXPECT_EQ(consumer_slot.payload->fanin_inline_edges[0].flags(), DEP_WAIT | DEP_RETAIN);
    // fanout_count is bit-packed: bit31 (FANOUT_SCOPE_BIT) is the owning-scope
    // ref, low bits the consumer count. The duplicate explicit dep is deduped to a
    // single consumer, so this is scope + 1.
    EXPECT_EQ(producer_slot.fanout_count, FANOUT_SCOPE_BIT + 1);
}

// An explicit ordering-only dep (the primitive add_dep_wait() lowers to) yields a
// DEP_WAIT edge, not the conservative DEP_WAIT|DEP_RETAIN default.
TEST_F(OrchestratorFaninTest, ExplicitWaitDepProducesWaitOnlyEdge) {
    orch.begin_scope();

    CoreTaskArgs producer_args;
    TaskOutputTensors producer = orch.submit_dummy_task(producer_args);
    ASSERT_TRUE(producer.task_id().is_valid());

    TaskId deps[] = {producer.task_id()};
    DepFlags kinds[] = {DEP_WAIT};
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies_with_kinds(deps, kinds, 1);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &consumer_slot =
        sm_handle->header->rings[simpler::tmr::task_ring(consumer.task_id())].get_slot_state_by_task_id(
            simpler::tmr::task_local_id(consumer.task_id())
        );
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

    TaskId deps[] = {producer.task_id(), producer.task_id()};
    DepFlags kinds[] = {DEP_WAIT, DEP_WAIT | DEP_RETAIN};
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies_with_kinds(deps, kinds, 2);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    auto &producer_slot =
        sm_handle->header->rings[simpler::tmr::task_ring(producer.task_id())].get_slot_state_by_task_id(
            simpler::tmr::task_local_id(producer.task_id())
        );
    auto &consumer_slot =
        sm_handle->header->rings[simpler::tmr::task_ring(consumer.task_id())].get_slot_state_by_task_id(
            simpler::tmr::task_local_id(consumer.task_id())
        );
    ASSERT_NE(consumer_slot.payload, nullptr);
    ASSERT_EQ(consumer_slot.payload->fanin_actual_count, 1);
    EXPECT_EQ(consumer_slot.payload->fanin_inline_edges[0].flags(), DEP_WAIT | DEP_RETAIN);
    EXPECT_EQ(producer_slot.fanout_count, FANOUT_SCOPE_BIT + 1);
}

// The duplicate lands in the spill region (>64 fanin), exercising
// or_flags_into_existing's spill lookup: the dup folds (65 edges, not 66), claims
// exactly one pin, and OR-accumulates its flags into the spilled edge.
TEST_F(OrchestratorFaninTest, DuplicateProducerInSpillRegionDedups) {
    orch.begin_scope();

    constexpr int kProducers = CHIP_FANIN_INLINE_CAP + 1;  // 65: the last one spills
    std::vector<TaskOutputTensors> producers;
    producers.reserve(kProducers);
    for (int i = 0; i < kProducers; i++) {
        CoreTaskArgs a;
        producers.push_back(orch.submit_dummy_task(a));
        ASSERT_TRUE(producers.back().task_id().is_valid());
    }

    std::vector<TaskId> deps;
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
        sm_handle->header->rings[simpler::tmr::task_ring(consumer.task_id())].get_slot_state_by_task_id(
            simpler::tmr::task_local_id(consumer.task_id())
        );
    ASSERT_NE(consumer_slot.payload, nullptr);
    TaskPayload *payload = consumer_slot.payload;
    EXPECT_EQ(payload->fanin_actual_count, kProducers);  // duplicate folded, not 66

    TaskId dup = producers.back().task_id();
    auto &dup_slot = sm_handle->header->rings[simpler::tmr::task_ring(dup)].get_slot_state_by_task_id(
        simpler::tmr::task_local_id(dup)
    );
    EXPECT_EQ(dup_slot.fanout_count, FANOUT_SCOPE_BIT + 1);  // one pin, not two

    // The first spilled edge is the duplicated producer; its flags OR-folded to
    // WAIT|RETAIN across the two discovery kinds.
    ASSERT_NE(payload->fanin_spill_pool, nullptr);
    FaninPool &spill_pool = *payload->fanin_spill_pool;
    FaninSpillEntry &spill_edge = spill_pool.base[payload->fanin_spill_start % spill_pool.capacity];
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
        sm_handle->header->rings[simpler::tmr::task_ring(producer.task_id())].get_slot_state_by_task_id(
            simpler::tmr::task_local_id(producer.task_id())
        );
    // COMPLETED but not consumed (the open scope still pins it): the consumer takes
    // the all-completed fast path.
    producer_slot.task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
    int32_t rc_before = producer_slot.fanout_refcount.load();

    TaskId deps[] = {producer.task_id()};
    DepFlags kinds[] = {DEP_WAIT};  // ordering-only
    CoreTaskArgs consumer_args;
    consumer_args.set_dependencies_with_kinds(deps, kinds, 1);
    TaskOutputTensors consumer = orch.submit_dummy_task(consumer_args);
    ASSERT_TRUE(consumer.task_id().is_valid());

    // The fast path released the ordering-only pin.
    EXPECT_EQ(producer_slot.fanout_refcount.load(), rc_before + 1);
}

// Bounded reachability bitmap reduction (issue #1376)
// ---------------------------------------------------------------------------

// Helper: fetch a task's slot state from the SM handle.
static ChipTaskSlotState &slot_of(SharedMemoryHandle *sm, const TaskOutputTensors &t) {
    return sm->header->rings[simpler::tmr::task_ring(t.task_id())].get_slot_state_by_task_id(
        static_cast<int32_t>(simpler::tmr::task_local_id(t.task_id()))
    );
}

// Diamond A -> B -> C plus direct A -> C, all conservative RETAIN edges: the
// direct A -> C WAIT is covered by the transitive path, so it demotes to
// RETAIN-only and drops out of the readiness count.
TEST_F(OrchestratorFaninTest, DiamondReducesRedundantWaitToRetainOnly) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    auto &b_slot = slot_of(sm_handle, b);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_actual_count, 2);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    bool saw_retain_only = false, saw_wait_retain = false;
    for (int i = 0; i < payload->fanin_actual_count; i++) {
        ChipTaskSlotState *p = payload->fanin_inline_edges[i].slot_state();
        DepFlags f = payload->fanin_inline_edges[i].flags();
        if (p == &a_slot) {
            EXPECT_EQ(f, DEP_RETAIN);
            saw_retain_only = true;
        } else if (p == &b_slot) {
            EXPECT_EQ(f, DEP_WAIT | DEP_RETAIN);
            saw_wait_retain = true;
        }
    }
    EXPECT_TRUE(saw_retain_only);
    EXPECT_TRUE(saw_wait_retain);
}

// Same diamond but the direct A -> C is ordering-only: the cleared edge becomes
// DEP_NONE and stays in storage (fanin_actual_count unchanged) so its
// submit-claim pin is still released by the !DEP_RETAIN paths.
TEST_F(OrchestratorFaninTest, DiamondDropsRedundantWaitOnlyEdge) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId ac[] = {b.task_id(), a.task_id()};
    DepFlags kinds[] = {DEP_WAIT | DEP_RETAIN, DEP_WAIT};
    CoreTaskArgs c_args;
    c_args.set_dependencies_with_kinds(ac, kinds, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_actual_count, 2);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    EXPECT_EQ(payload->fanin_inline_edges[1].slot_state(), &a_slot);
    EXPECT_EQ(payload->fanin_inline_edges[1].flags(), DEP_NONE);
}

// Early-dispatch accounting survives reduction. submit_dummy_task tasks do not
// allow early resolve, so the diamond's reduced A -> C edge points at an
// unflagged producer: C keeps the unit that producer held in fanin_wait_count,
// leaving early_dispatch_target() unreachable exactly as it was before
// reduction.
TEST_F(OrchestratorFaninTest, ReducedEdgeToUnflaggedProducerBlocksEarlyDispatch) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId ac[] = {b.task_id(), a.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_actual_count, 2);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    EXPECT_EQ(payload->early_dispatch_blocked, 1);
    EXPECT_EQ(payload->early_dispatch_target(), 2);
}

// The same diamond with a producer that DOES allow early resolve: reduction
// costs the consumer nothing, because that producer would have propagated.
TEST_F(OrchestratorFaninTest, ReducedEdgeToFlaggedProducerLeavesEarlyDispatchOpen) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    slot_of(sm_handle, a).task_attrs.set_early_resolve(true);

    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId ac[] = {b.task_id(), a.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    EXPECT_EQ(payload->early_dispatch_blocked, 0);
    EXPECT_EQ(payload->early_dispatch_target(), 1);
}

// A -> B -> C -> D plus direct A -> D (no A -> C, no B -> D): the covering path
// is longer than one hop, so only the transitive bitmap can prove A redundant.
TEST_F(OrchestratorFaninTest, Depth3ChainReducesBeyondOneHop) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());
    TaskId bc[] = {b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(bc, 1);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    TaskId ad[] = {a.task_id(), c.task_id()};
    CoreTaskArgs d_args;
    d_args.set_dependencies(ad, 2);
    TaskOutputTensors d = orch.submit_dummy_task(d_args);
    ASSERT_TRUE(d.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    TaskPayload *payload = slot_of(sm_handle, d).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    for (int i = 0; i < payload->fanin_actual_count; i++) {
        if (payload->fanin_inline_edges[i].slot_state() == &a_slot) {
            EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_RETAIN);
        }
    }
}

// Two producers with no path between them: nothing to prove, both WAITs stay.
TEST_F(OrchestratorFaninTest, IndependentProducersAreNotReduced) {
    orch.begin_scope();

    CoreTaskArgs a_args, b_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(a.task_id().is_valid());
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId deps[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(deps, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_actual_count, 2);
    EXPECT_EQ(payload->fanin_wait_count, 2);
    EXPECT_EQ(payload->fanin_inline_edges[0].flags(), DEP_WAIT | DEP_RETAIN);
    EXPECT_EQ(payload->fanin_inline_edges[1].flags(), DEP_WAIT | DEP_RETAIN);
}

// A -> B ordered by a WAIT-only (ordering-only) edge still puts A in R[B]: any
// WAIT edge carries ordering, so the covering path proves reachability and the
// direct A -> C is reduced. (The reduction's reachability semantics differ
// from a fanin-pointer walk here — WAIT-only cover is a valid witness.)
TEST_F(OrchestratorFaninTest, WaitOnlyCoveringEdgeStillProvesReachability) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    DepFlags wait_kind[] = {DEP_WAIT};
    CoreTaskArgs b_args;
    b_args.set_dependencies_with_kinds(ab, wait_kind, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    for (int i = 0; i < payload->fanin_actual_count; i++) {
        if (payload->fanin_inline_edges[i].slot_state() == &a_slot) {
            EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_RETAIN);
        }
    }
}

// A producer farther back than WAIT_REACH_WINDOW submissions keeps its WAIT:
// the window cannot represent it, so the edge is retained conservatively.
TEST_F(OrchestratorFaninTest, WindowMissBeyondBlKeepsWait) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    // Fillers push seq(A) out of the window: seq distance to C exceeds 64.
    CoreTaskArgs filler_args;
    for (int i = 0; i < WAIT_REACH_WINDOW; i++) {
        TaskOutputTensors f = orch.submit_dummy_task(filler_args);
        ASSERT_TRUE(f.task_id().is_valid());
    }

    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 2);
    for (int i = 0; i < payload->fanin_actual_count; i++) {
        if (payload->fanin_inline_edges[i].slot_state() == &a_slot) {
            EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_WAIT | DEP_RETAIN);
        }
    }
}

// d(A -> C) == WAIT_REACH_WINDOW exactly: the direct bit is the last window
// bit, and the close covering producer B (d == 1, with A at bit 62 of R[B])
// still shifts A's bit onto it, so the edge reduces. Proves the
// d == WAIT_REACH_WINDOW guard does not block valid boundary reduction.
TEST_F(OrchestratorFaninTest, BoundaryAtBlStillReducesViaCloseProducer) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    // 62 fillers: seq(A)=0, seq(B)=1, fillers 2..63, C=64 -> d(A->C)=64.
    CoreTaskArgs filler_args;
    for (int i = 0; i < WAIT_REACH_WINDOW - 2; i++) {
        TaskOutputTensors f = orch.submit_dummy_task(filler_args);
        ASSERT_TRUE(f.task_id().is_valid());
    }

    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    for (int i = 0; i < payload->fanin_actual_count; i++) {
        if (payload->fanin_inline_edges[i].slot_state() == &a_slot) {
            EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_RETAIN);
        }
    }
}

// Unsigned sequence subtraction preserves recent distances across uint64 wrap.
TEST_F(OrchestratorFaninTest, SequenceWrapPreservesRecentReachability) {
    orch.submit_seq = std::numeric_limits<uint64_t>::max() - 1;
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    for (int i = 0; i < payload->fanin_actual_count; i++) {
        if (payload->fanin_inline_edges[i].slot_state() == &a_slot) {
            EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_RETAIN);
        }
    }
}

// Nested scopes move tasks onto different rings; the global submission
// sequence makes cross-ring candidates participate in the same window.
TEST_F(OrchestratorFaninTest, CrossRingCandidateUsesGlobalSequence) {
    orch.begin_scope();  // ring 0

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    ASSERT_EQ(simpler::tmr::task_ring(a.task_id()), 0);

    orch.begin_scope();  // ring 1
    ASSERT_EQ(orch.current_ring_id(), 1);
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());
    ASSERT_EQ(simpler::tmr::task_ring(b.task_id()), 1);

    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    for (int i = 0; i < payload->fanin_actual_count; i++) {
        if (payload->fanin_inline_edges[i].slot_state() == &a_slot) {
            EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_RETAIN);
        }
    }
}

// alloc_tensors' hidden task never enters submit_task_common; it must still
// publish an empty reachability bitmap so a consumer reading it as a creator
// producer sees a conservative entry, never a stale slot generation.
TEST_F(OrchestratorFaninTest, AllocTensorProducerPublishesEmptyReach) {
    orch.begin_scope();

    std::vector<TensorCreateInfo> create_infos;
    CoreTaskArgs alloc_args;
    add_runtime_output_arg(alloc_args, create_infos, 4);
    TaskOutputTensors alloc = orch.alloc_tensors(alloc_args);
    ASSERT_TRUE(alloc.task_id().is_valid());

    // Consumer depends on the alloc task explicitly; a second consumer chain
    // through the first proves distances stay correct across the alloc entry.
    TaskId deps[] = {alloc.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(deps, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());
    ASSERT_FALSE(orch.fatal);

    TaskId cd[] = {alloc.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(cd, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());
    ASSERT_FALSE(orch.fatal);

    auto &alloc_slot = slot_of(sm_handle, alloc);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    for (int i = 0; i < payload->fanin_actual_count; i++) {
        if (payload->fanin_inline_edges[i].slot_state() == &alloc_slot) {
            EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_RETAIN);
        }
    }
}

// Runtime reuse leaves the side array uncleared; publication by the new slot
// owner replaces stale bits before a later consumer can read them.
TEST_F(OrchestratorFaninTest, RuntimeReuseOverwritesStaleSlotBitmap) {
    orch.wait_reach[0][0].ancestors = std::numeric_limits<uint64_t>::max();
    orch.wait_reach[0][0].seq = 1234;

    ASSERT_TRUE(sm_handle->init(sm_handle->sm_base, sm_handle->sm_size, CHIP_TASK_WINDOW_SIZE, 4096));
    uint64_t heap_sizes[CHIP_MAX_RING_DEPTH];
    uint64_t task_window_sizes[CHIP_MAX_RING_DEPTH];
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++) {
        heap_sizes[r] = 4096;
        task_window_sizes[r] = CHIP_TASK_WINDOW_SIZE;
    }
    ASSERT_TRUE(orch.reset_for_reuse(orch_layout, sm_handle->sm_base, gm_heap.data(), heap_sizes, task_window_sizes));
    sched.reset_for_reuse(sched_layout, sm_handle->sm_base);

    EXPECT_EQ(orch.wait_reach[0][0].ancestors, std::numeric_limits<uint64_t>::max());
    orch.begin_scope();
    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    ASSERT_EQ(simpler::tmr::task_local_id(a.task_id()), 0u);
    EXPECT_EQ(orch.wait_reach[0][0].ancestors, 0);
    EXPECT_EQ(orch.wait_reach[0][0].seq, 0);

    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());
    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());
    EXPECT_EQ(slot_of(sm_handle, c).payload->fanin_wait_count, 1);
}

// Acceptance #5: candidate discovery order must not change the reduced graph.
// Same diamond with the dependency arrays in both orders.
TEST_F(OrchestratorFaninTest, DiscoveryOrderDoesNotChangeReduction) {
    for (int reverse = 0; reverse < 2; reverse++) {
        orch.begin_scope();

        CoreTaskArgs a_args;
        TaskOutputTensors a = orch.submit_dummy_task(a_args);
        ASSERT_TRUE(a.task_id().is_valid());
        TaskId ab[] = {a.task_id()};
        CoreTaskArgs b_args;
        b_args.set_dependencies(ab, 1);
        TaskOutputTensors b = orch.submit_dummy_task(b_args);
        ASSERT_TRUE(b.task_id().is_valid());

        TaskId ac[] = {a.task_id(), b.task_id()};
        TaskId ca[] = {b.task_id(), a.task_id()};
        CoreTaskArgs c_args;
        c_args.set_dependencies(reverse ? ca : ac, 2);
        TaskOutputTensors c = orch.submit_dummy_task(c_args);
        ASSERT_TRUE(c.task_id().is_valid());

        auto &a_slot = slot_of(sm_handle, a);
        auto &b_slot = slot_of(sm_handle, b);
        TaskPayload *payload = slot_of(sm_handle, c).payload;
        ASSERT_NE(payload, nullptr);
        EXPECT_EQ(payload->fanin_actual_count, 2);
        EXPECT_EQ(payload->fanin_wait_count, 1);
        for (int i = 0; i < payload->fanin_actual_count; i++) {
            ChipTaskSlotState *p = payload->fanin_inline_edges[i].slot_state();
            if (p == &a_slot) {
                EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_RETAIN);
            } else if (p == &b_slot) {
                EXPECT_EQ(payload->fanin_inline_edges[i].flags(), DEP_WAIT | DEP_RETAIN);
            }
        }

        orch.end_scope();
    }
}

// The reduction covers spill-region candidates too — no inline cap. The
// redundant pair lands past CHIP_FANIN_INLINE_CAP so the cleared edge lives in
// the spill pool.
TEST_F(OrchestratorFaninTest, SpillRegionCandidatesAreReduced) {
    orch.begin_scope();

    constexpr int kOldProducers = CHIP_FANIN_INLINE_CAP + 1;
    std::vector<TaskOutputTensors> old_producers;
    old_producers.reserve(kOldProducers);
    for (int i = 0; i < kOldProducers; i++) {
        CoreTaskArgs args;
        old_producers.push_back(orch.submit_dummy_task(args));
        ASSERT_TRUE(old_producers.back().task_id().is_valid());
    }

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    std::vector<TaskId> deps;
    std::vector<DepFlags> kinds;
    deps.reserve(kOldProducers + 2);
    kinds.reserve(kOldProducers + 2);
    for (auto &producer : old_producers) {
        deps.push_back(producer.task_id());
        kinds.push_back(DEP_WAIT | DEP_RETAIN);
    }
    deps.push_back(b.task_id());
    kinds.push_back(DEP_WAIT | DEP_RETAIN);
    deps.push_back(a.task_id());  // redundant, lands in the spill region
    kinds.push_back(DEP_WAIT | DEP_RETAIN);

    CoreTaskArgs c_args;
    c_args.set_dependencies_with_kinds(deps.data(), kinds.data(), static_cast<uint32_t>(deps.size()));
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_actual_count, kOldProducers + 2);
    EXPECT_EQ(payload->fanin_wait_count, kOldProducers + 1);
    ASSERT_NE(payload->fanin_spill_pool, nullptr);
    bool found = false;
    auto check = [&](ChipTaskSlotState *p, DepFlags f) {
        if (p == &a_slot) {
            EXPECT_EQ(f, DEP_RETAIN);
            found = true;
        }
    };
    FaninPool &pool = *payload->fanin_spill_pool;
    int32_t spill_count = payload->fanin_actual_count - CHIP_FANIN_INLINE_CAP;
    ASSERT_GT(spill_count, 0);
    for (int i = 0; i < spill_count; i++) {
        FaninSpillEntry &e = pool.base[(payload->fanin_spill_start % pool.capacity + i) % pool.capacity];
        check(e.slot_state(), e.flags());
    }
    EXPECT_TRUE(found);
}

// A reduction-dropped (DEP_NONE) edge on the all-completed fast path releases
// its submit-claim pin exactly once — there, not again at on_task_release.
TEST_F(OrchestratorFaninTest, AllCompletedFastPathReleasesDroppedEdgePin) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    DepFlags wait_kind[] = {DEP_WAIT};
    CoreTaskArgs b_args;
    b_args.set_dependencies_with_kinds(ab, wait_kind, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    auto &b_slot = slot_of(sm_handle, b);
    // Both completed: the consumer takes the all-completed fast path.
    a_slot.task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
    b_slot.task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
    int32_t a_rc_before = a_slot.fanout_refcount.load();

    // Direct A -> C is ordering-only so the reduction drops it to DEP_NONE.
    TaskId ac[] = {b.task_id(), a.task_id()};
    DepFlags kinds[] = {DEP_WAIT | DEP_RETAIN, DEP_WAIT};
    CoreTaskArgs c_args;
    c_args.set_dependencies_with_kinds(ac, kinds, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
    // The dropped edge's pin was released by the fast path.
    EXPECT_EQ(a_slot.fanout_refcount.load(), a_rc_before + 1);
}

// A RETAIN-only survivor of the reduction keeps its producer pinned past
// wiring: only the consumer's on_task_release drops that pin.
TEST_F(OrchestratorFaninTest, ReducedRetainOnlyEdgeHoldsProducerUntilConsumerRelease) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());

    auto &a_slot = slot_of(sm_handle, a);
    auto &c_slot = slot_of(sm_handle, c);
    TaskPayload *payload = c_slot.payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);

    int32_t a_rc_before = a_slot.fanout_refcount.load();
    sched.on_task_release(c_slot);
    // The RETAIN-only edge's pin is released by on_task_release.
    EXPECT_EQ(a_slot.fanout_refcount.load(), a_rc_before + 1);
}

// Zero-fanin and single-fanin tasks still publish their (mostly empty)
// bitmaps; a later chain consuming them must reduce normally.
TEST_F(OrchestratorFaninTest, ZeroFaninTaskPublishesEmptyBitmap) {
    orch.begin_scope();

    CoreTaskArgs a_args;
    TaskOutputTensors a = orch.submit_dummy_task(a_args);  // zero fanin -> R=0
    ASSERT_TRUE(a.task_id().is_valid());
    TaskId ab[] = {a.task_id()};
    CoreTaskArgs b_args;
    b_args.set_dependencies(ab, 1);  // single fanin
    TaskOutputTensors b = orch.submit_dummy_task(b_args);
    ASSERT_TRUE(b.task_id().is_valid());

    TaskId ac[] = {a.task_id(), b.task_id()};
    CoreTaskArgs c_args;
    c_args.set_dependencies(ac, 2);
    TaskOutputTensors c = orch.submit_dummy_task(c_args);
    ASSERT_TRUE(c.task_id().is_valid());
    ASSERT_FALSE(orch.fatal);

    TaskPayload *payload = slot_of(sm_handle, c).payload;
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(payload->fanin_wait_count, 1);
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
    auto &first_slot =
        ring.get_slot_state_by_task_id(static_cast<int32_t>(simpler::tmr::task_local_id(first.task_id())));
    orch.end_scope();
    first_slot.task_state.store(CHIP_TASK_COMPLETED, std::memory_order_release);
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
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), SIMPLER_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_NE(log.find("FATAL: Task Allocator Deadlock - Heap Exhausted! ring=1"), std::string::npos);
    EXPECT_NE(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
    EXPECT_NE(log.find("Heap ring 1:"), std::string::npos);
    EXPECT_NE(log.find("used=3072"), std::string::npos);
    EXPECT_NE(log.find("available=1024"), std::string::npos);
    EXPECT_EQ(log.find("runtime_env.ring_heap=<bytes>"), std::string::npos);
}

TEST_F(OrchestratorFaninTest, StructuralCheckRejectsOpenAncestorWhenNestedScopesShareRing) {
    std::vector<TensorCreateInfo> create_infos;
    create_infos.reserve(2);

    for (int32_t depth = 0; depth < CHIP_MAX_RING_DEPTH; ++depth) {
        orch.begin_scope();
    }
    ASSERT_EQ(orch.current_ring_id(), CHIP_MAX_RING_DEPTH - 1);

    CoreTaskArgs parent_args;
    add_runtime_output_arg(parent_args, create_infos, 1024);
    TaskOutputTensors parent = orch.submit_dummy_task(parent_args);
    ASSERT_TRUE(parent.task_id().is_valid());

    orch.begin_scope();
    ASSERT_EQ(orch.current_ring_id(), CHIP_MAX_RING_DEPTH - 1);

    CoreTaskArgs child_args;
    add_runtime_output_arg(child_args, create_infos, 1);
    testing::internal::CaptureStderr();
    TaskOutputTensors child = orch.submit_dummy_task(child_args);
    std::string log = testing::internal::GetCapturedStderr();

    EXPECT_FALSE(child.task_id().is_valid());
    EXPECT_TRUE(orch.fatal);
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), SIMPLER_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_NE(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
}

TEST_F(OrchestratorFaninTest, ClosedChildHeadUsesTimeoutWithOpenParentOnSharedRing) {
    std::vector<TensorCreateInfo> create_infos;
    create_infos.reserve(3);

    for (int32_t depth = 0; depth < CHIP_MAX_RING_DEPTH; ++depth) {
        orch.begin_scope();
    }
    orch.begin_scope();
    ASSERT_EQ(orch.current_ring_id(), CHIP_MAX_RING_DEPTH - 1);

    CoreTaskArgs child_args;
    add_runtime_output_arg(child_args, create_infos, 768);
    TaskOutputTensors child = orch.submit_dummy_task(child_args);
    ASSERT_TRUE(child.task_id().is_valid());

    orch.end_scope();
    ASSERT_EQ(orch.current_ring_id(), CHIP_MAX_RING_DEPTH - 1);

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
    EXPECT_EQ(sm_handle->header->orch_error_code.load(std::memory_order_acquire), SIMPLER_ERROR_HEAP_RING_DEADLOCK);
    EXPECT_NE(log.find("No reclaim progress for ~500 ms"), std::string::npos);
    EXPECT_EQ(log.find("oldest task owned by an open scope on this ring"), std::string::npos);
}

// Regression for issue #1188: scope_tasks_cap must equal the real in-flight budget
// (sum of the runtime per-ring windows), not the compile-time CHIP_SCOPE_TASKS_CAP.
// reserve_layout only computes offsets, so no commit()/backing is needed here.
TEST(OrchestratorLayoutScopeTasksCap, FollowsRuntimeWindowSum) {
    auto cap_for = [](const int32_t windows[CHIP_MAX_RING_DEPTH]) {
        DeviceArena arena;
        int32_t cap = OrchestratorState::reserve_layout(arena, windows).scope_tasks_cap;
        arena.release();
        return cap;
    };

    int32_t windows[CHIP_MAX_RING_DEPTH];

    // Default window: cap == the old compile-time value (no behavior change).
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++)
        windows[r] = CHIP_TASK_WINDOW_SIZE;
    EXPECT_EQ(cap_for(windows), CHIP_TASK_WINDOW_SIZE * CHIP_MAX_RING_DEPTH);
    EXPECT_EQ(cap_for(windows), CHIP_SCOPE_TASKS_CAP);

    // Shrunk window: cap shrinks to the real budget (no over-allocation).
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++)
        windows[r] = 4;
    EXPECT_EQ(cap_for(windows), 4 * CHIP_MAX_RING_DEPTH);

    // Enlarged window past the compile default: cap grows to match the rings, so a
    // large scope no longer hits a premature SCOPE_TASKS_OVERFLOW (the bug fixed).
    const int32_t big = CHIP_TASK_WINDOW_SIZE * 2;
    for (int r = 0; r < CHIP_MAX_RING_DEPTH; r++)
        windows[r] = big;
    EXPECT_EQ(cap_for(windows), big * CHIP_MAX_RING_DEPTH);
    EXPECT_GT(cap_for(windows), CHIP_SCOPE_TASKS_CAP);
}
