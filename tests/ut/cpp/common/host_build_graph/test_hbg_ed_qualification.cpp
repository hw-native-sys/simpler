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
 * Early-dispatch qualification contract, decided by the host at submit:
 *
 *   ED_FLAG_CANDIDATE  every producer carries allow_early_resolve, no dispatch
 *                      predicate, dispatchable shape, fanin >= 1
 *   ED_FLAG_TRACKED    at least one candidate names this task as a producer
 *
 * and a candidate's fanin row is sorted by ascending producer local id
 * (= submission order, ids are bump-allocated), so the device's backward
 * fanin scans target the latest-submitted producer; non-candidate rows keep
 * their fill order.
 */

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "utils/device_arena.h"
#include "scheduler/scheduler.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/task_id.h"

class HbgEdQualificationTest : public ::testing::Test {
protected:
    DeviceArena sm_arena;
    DeviceArena runtime_arena;
    SharedMemoryHandle *sm_handle = nullptr;
    OrchestratorState orch{};
    SchedulerState sched{};
    SchedulerLayout sched_layout{};
    std::vector<char> gm_heap;
    std::vector<TensorCreateInfo> create_infos;

    void SetUp() override {
        sm_handle = SharedMemoryHandle::create_and_init_default(sm_arena);
        ASSERT_NE(sm_handle, nullptr);
        gm_heap.resize(1 << 16);
        create_infos.reserve(16);

        sched_layout = SchedulerState::reserve_layout(runtime_arena);
        ASSERT_NE(runtime_arena.commit(), nullptr);

        ASSERT_TRUE(sched.init_data_from_layout(sched_layout, runtime_arena, sm_handle->sm_base));
        sched.wire_arena_pointers(sched_layout, runtime_arena);
        sched.seed_queue_slots();
        ASSERT_TRUE(orch.init(sm_handle->sm_base, gm_heap.data(), gm_heap.size(), CHIP_DEFAULT_GRAPH_TASKS));
        orch.begin_scope();
    }

    void TearDown() override {
        sched.destroy();
        runtime_arena.release();
        sm_arena.release();
    }

    // A whole publish event as the dispatch path performs it: account for the
    // blocks before their tokens go out, then seal once they have.
    void publish_blocks(ChipTaskSlotState &slot_state, int32_t count) {
        if (sched.account_published_blocks(slot_state, count)) sched.seal_ed_publish_list(slot_state);
    }

    // One AIV producer with an output tensor; `flagged` sets allow_early_resolve.
    TaskId submit_producer(bool flagged) {
        uint32_t shape[] = {16};
        CoreTaskArgs args;
        create_infos.emplace_back(shape, 1, DataType::FLOAT32);
        args.add_output(create_infos.back());
        args.set_allow_early_resolve(flagged);
        MixedKernels mixed{};
        mixed.aiv0_kernel_id = 0;
        TaskOutputTensors out = orch.submit_task(mixed, args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.task_id();
    }

    // One AIV consumer depending on `deps`, in the given order.
    TaskId submit_consumer(const TaskId *deps, uint32_t dep_count) {
        uint32_t shape[] = {16};
        CoreTaskArgs args;
        create_infos.emplace_back(shape, 1, DataType::FLOAT32);
        args.add_output(create_infos.back());
        args.set_dependencies(deps, dep_count);
        MixedKernels mixed{};
        mixed.aiv0_kernel_id = 0;
        TaskOutputTensors out = orch.submit_task(mixed, args);
        EXPECT_TRUE(out.task_id().is_valid());
        return out.task_id();
    }

    const ChipTaskSlotState &slot_of(TaskId id) {
        return sm_handle->header->tasks.get_slot_state_by_task_id(id.local_id());
    }

    const TaskPayload &payload_of(TaskId id) { return sm_handle->header->tasks.storage_at(id.local_id()).payload; }
};

TEST_F(HbgEdQualificationTest, AllFlaggedProducersMakeCandidateAndTrackProducers) {
    TaskId p1 = submit_producer(/*flagged=*/true);
    TaskId p2 = submit_producer(/*flagged=*/true);
    // Dependencies handed over in reverse submission order: the sorted row must
    // not depend on declaration order.
    TaskId deps[] = {p2, p1};
    TaskId c = submit_consumer(deps, 2);

    EXPECT_NE(slot_of(c).ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_NE(slot_of(p1).ed_flags & ED_FLAG_TRACKED, 0);
    EXPECT_NE(slot_of(p2).ed_flags & ED_FLAG_TRACKED, 0);

    const TaskPayload &pl = payload_of(c);
    ASSERT_EQ(pl.fanin_count, 2);
    EXPECT_EQ(pl.fanin_data()[0], p1.local_id());
    EXPECT_EQ(pl.fanin_data()[1], p2.local_id());
}

TEST_F(HbgEdQualificationTest, OneUnflaggedProducerDisqualifies) {
    TaskId p1 = submit_producer(/*flagged=*/true);
    TaskId p2 = submit_producer(/*flagged=*/false);
    // Reverse submission order: a fill-order row must keep p2 before p1, so
    // this assertion fails if non-candidate rows were sorted too.
    TaskId deps[] = {p2, p1};
    TaskId c = submit_consumer(deps, 2);

    EXPECT_EQ(slot_of(c).ed_flags & ED_FLAG_CANDIDATE, 0);
    // No candidate, so neither producer is tracked.
    EXPECT_EQ(slot_of(p1).ed_flags & ED_FLAG_TRACKED, 0);
    EXPECT_EQ(slot_of(p2).ed_flags & ED_FLAG_TRACKED, 0);

    // A non-candidate row keeps its fill order (here p2 then p1).
    const TaskPayload &pl = payload_of(c);
    ASSERT_EQ(pl.fanin_count, 2);
    EXPECT_EQ(pl.fanin_data()[0], p2.local_id());
    EXPECT_EQ(pl.fanin_data()[1], p1.local_id());
}

TEST_F(HbgEdQualificationTest, DummyConsumerIsNotCandidate) {
    TaskId p1 = submit_producer(/*flagged=*/true);
    TaskId deps[] = {p1};
    CoreTaskArgs args;
    args.set_dependencies(deps, 1);
    TaskOutputTensors dummy = orch.submit_dummy_task(args);
    ASSERT_TRUE(dummy.task_id().is_valid());

    EXPECT_EQ(slot_of(dummy.task_id()).ed_flags & ED_FLAG_CANDIDATE, 0);
    EXPECT_EQ(slot_of(p1).ed_flags & ED_FLAG_TRACKED, 0);
}

// The wake-scan cursor: each classification resumes where the last one hung
// and never re-walks the row's completed tail. The monotone state byte makes
// the resume equivalent to a full rescan, so hang decisions are unchanged.
TEST_F(HbgEdQualificationTest, WakeScanCursorResumesAndNeverRewalks) {
    TaskId p1 = submit_producer(/*flagged=*/false);
    TaskId p2 = submit_producer(/*flagged=*/false);
    TaskId p3 = submit_producer(/*flagged=*/false);
    TaskId deps[] = {p1, p2, p3};
    TaskId c = submit_consumer(deps, 3);

    ChipTaskSlotState &c_slot = sm_handle->header->tasks.get_slot_state_by_task_id(c.local_id());
    auto &tasks = sm_handle->header->tasks;
    const int32_t *row = c_slot.to_payload().fanin_data();

    // Never hung: cursor is the 0xFFFF sentinel and the first scan starts at the
    // row's tail.
    EXPECT_EQ(c_slot.wake_scan_cursor, 0xFFFF);
    EXPECT_EQ(sched.classify_fanin_state(&c_slot), 2);
    EXPECT_EQ(c_slot.wake_scan_cursor, 2);

    // The hung-at producer completes: the rescan resumes at the cursor and
    // walks down to the next unmet entry.
    tasks.store_completed(row[2]);
    EXPECT_EQ(sched.classify_fanin_state(&c_slot), 1);
    EXPECT_EQ(c_slot.wake_scan_cursor, 1);

    // Entries above the cursor completing do not move it (nothing rescans
    // them); completing the remaining tail yields the ready verdict.
    tasks.store_completed(row[0]);
    EXPECT_EQ(sched.classify_fanin_state(&c_slot), 1);
    EXPECT_EQ(c_slot.wake_scan_cursor, 1);
    tasks.store_completed(row[1]);
    EXPECT_EQ(sched.classify_fanin_state(&c_slot), -1);
}

TEST_F(HbgEdQualificationTest, RootIsNotCandidate) {
    TaskId root = submit_producer(/*flagged=*/true);
    EXPECT_EQ(slot_of(root).ed_flags & ED_FLAG_CANDIDATE, 0);
}

// The publish-list lifecycle, driven at the SchedulerState level: register ->
// hang on the latest-submitted producer -> publish seals and detaches the
// chain -> cursor rescan re-hangs on the next unpublished producer -> the
// last publish yields the all-published verdict and the ED-queue claim.
TEST_F(HbgEdQualificationTest, PublishListDetectsAllPublished) {
    TaskId p1 = submit_producer(/*flagged=*/true);
    TaskId p2 = submit_producer(/*flagged=*/true);
    TaskId deps[] = {p1, p2};
    TaskId c = submit_consumer(deps, 2);

    ChipTaskSlotState &c_slot = sm_handle->header->tasks.get_slot_state_by_task_id(c.local_id());
    ChipTaskSlotState &p1_slot = sm_handle->header->tasks.get_slot_state_by_task_id(p1.local_id());
    ChipTaskSlotState &p2_slot = sm_handle->header->tasks.get_slot_state_by_task_id(p2.local_id());

    // Intake: nothing published -> hangs on p2, the row's tail.
    ASSERT_FALSE(sched.register_on_ed_publish_list(c_slot));
    EXPECT_EQ(p2_slot.ed_publish_list_head.load(std::memory_order_relaxed), &c_slot);
    EXPECT_EQ(c_slot.ed_publish_scan_cursor, 1);

    // p2 publishes its only block: chain seals, detached head reaches the
    // pending-drain queue.
    publish_blocks(p2_slot, 1);
    EXPECT_EQ(p2_slot.ed_publish_list_head.load(std::memory_order_relaxed), ED_PUBLISH_LIST_SENTINEL);
    ChipTaskSlotState *detached = nullptr;
    ASSERT_EQ(sched.ed_publish_drain_queue.pop_batch(&detached, 1), 1);
    ASSERT_EQ(detached, &c_slot);

    // Rescan: p1 still unpublished -> re-hang on p1, cursor advanced.
    ASSERT_FALSE(sched.advance_ed_publish_scan(*detached));
    EXPECT_EQ(p1_slot.ed_publish_list_head.load(std::memory_order_relaxed), &c_slot);
    EXPECT_EQ(c_slot.ed_publish_scan_cursor, 0);

    // p1 publishes: rescan reaches the all-published verdict; the claim CAS
    // moves the candidate to STAGING and its shape queue.
    publish_blocks(p1_slot, 1);
    ASSERT_EQ(sched.ed_publish_drain_queue.pop_batch(&detached, 1), 1);
    ASSERT_EQ(detached, &c_slot);
    ASSERT_TRUE(sched.advance_ed_publish_scan(*detached));
    sched.enqueue_early_dispatch_candidate(*detached);
    EXPECT_EQ(c_slot.to_payload().early_dispatch_state.load(std::memory_order_relaxed), EARLY_DISPATCH_STAGING);
    ChipTaskSlotState *staged = nullptr;
    ASSERT_EQ(sched.early_dispatch_queues[static_cast<int32_t>(ResourceShape::AIV)].pop_batch(&staged, 1), 1);
    EXPECT_EQ(staged, &c_slot);
}

// The publish event is two calls around the caller's MMIO token writes, and the
// state byte belongs to the first: a scanner that meets the sentinel therefore
// always reads the producer as published, never the reverse. The window between
// them is where a caller's tokens go, so it is entered here deliberately.
TEST_F(HbgEdQualificationTest, PublishedStateIsVisibleBeforeTheChainSeals) {
    TaskId p = submit_producer(/*flagged=*/true);
    TaskId deps[] = {p};
    TaskId c = submit_consumer(deps, 1);

    auto &tasks = sm_handle->header->tasks;
    ChipTaskSlotState &p_slot = tasks.get_slot_state_by_task_id(p.local_id());
    ChipTaskSlotState &c_slot = tasks.get_slot_state_by_task_id(c.local_id());
    const int32_t p_local = p.local_id();

    ASSERT_FALSE(sched.register_on_ed_publish_list(c_slot));
    ASSERT_FALSE(tasks.is_published(p_local));

    // Accounting reaches the total: the byte says published, the chain is still
    // open, and the waiter has not been detached yet.
    ASSERT_TRUE(sched.account_published_blocks(p_slot, p_slot.logical_block_num));
    EXPECT_TRUE(tasks.is_published(p_local));
    EXPECT_EQ(p_slot.ed_publish_list_head.load(std::memory_order_relaxed), &c_slot);
    ChipTaskSlotState *detached = nullptr;
    EXPECT_EQ(sched.ed_publish_drain_queue.pop_batch(&detached, 1), 0);

    // The seal closes the chain and hands the waiter over.
    sched.seal_ed_publish_list(p_slot);
    EXPECT_EQ(p_slot.ed_publish_list_head.load(std::memory_order_relaxed), ED_PUBLISH_LIST_SENTINEL);
    ASSERT_EQ(sched.ed_publish_drain_queue.pop_batch(&detached, 1), 1);
    EXPECT_EQ(detached, &c_slot);
    EXPECT_TRUE(sched.advance_ed_publish_scan(*detached));
}

// A tracked producer that never publishes (DUMMY, predicate-retired) reaches
// COMPLETED without passing through PUBLISHED, and the ordered comparison is
// what still releases its publish-list waiters.
TEST_F(HbgEdQualificationTest, CompletionAloneCountsAsPublished) {
    TaskId p = submit_producer(/*flagged=*/true);
    TaskId deps[] = {p};
    TaskId c = submit_consumer(deps, 1);

    auto &tasks = sm_handle->header->tasks;
    const int32_t p_local = p.local_id();
    ChipTaskSlotState &c_slot = tasks.get_slot_state_by_task_id(c.local_id());

    ASSERT_FALSE(sched.register_on_ed_publish_list(c_slot));
    ASSERT_FALSE(tasks.is_published(p_local));

    tasks.store_completed(p_local);
    EXPECT_EQ(tasks.task_states[p_local].load(std::memory_order_relaxed), CHIP_TASK_COMPLETED);
    EXPECT_TRUE(tasks.is_published(p_local));
    EXPECT_TRUE(tasks.is_completed(p_local));
}

// A two-level ED chain: the middle task is both a candidate (its own producer
// is flagged) and a tracked producer. Its seal must fire on full publication
// even while it is itself STAGING — gated, doorbell not yet rung — so its
// consumer's verdict and claim follow while the whole chain is still gated.
TEST_F(HbgEdQualificationTest, GatedCandidateSealsWhileStagingAndReleasesItsConsumer) {
    TaskId p0 = submit_producer(/*flagged=*/true);
    TaskId deps0[] = {p0};
    TaskId p1 = submit_consumer(deps0, 1);  // candidate AND tracked (C depends on it)
    ChipTaskSlotState &p1_slot =
        sm_handle->header->tasks.get_slot_state_by_task_id(static_cast<int32_t>(p1.local_id()));
    // p1 must be flagged for C to qualify; the host wrote CANDIDATE from p0's
    // flag, and TRACKED lands below when C qualifies against p1's flag.
    p1_slot.task_attrs.set_early_resolve(true);
    TaskId deps1[] = {p1};
    TaskId c = submit_consumer(deps1, 1);
    ChipTaskSlotState &c_slot = sm_handle->header->tasks.get_slot_state_by_task_id(static_cast<int32_t>(c.local_id()));
    ASSERT_NE(p1_slot.ed_flags & ED_FLAG_CANDIDATE, 0);
    ASSERT_NE(p1_slot.ed_flags & ED_FLAG_TRACKED, 0);
    ASSERT_NE(c_slot.ed_flags & ED_FLAG_CANDIDATE, 0);

    // Intake: p1 hangs on p0's publish list, C hangs on p1's.
    ASSERT_FALSE(sched.register_on_ed_publish_list(p1_slot));
    ASSERT_FALSE(sched.register_on_ed_publish_list(c_slot));

    // p0 publishes: p1's rescan reaches the verdict and p1 is claimed STAGING
    // (pre-staged, doorbell not rung — the gated state).
    ChipTaskSlotState &p0_slot =
        sm_handle->header->tasks.get_slot_state_by_task_id(static_cast<int32_t>(p0.local_id()));
    publish_blocks(p0_slot, 1);
    ChipTaskSlotState *detached = nullptr;
    ASSERT_EQ(sched.ed_publish_drain_queue.pop_batch(&detached, 1), 1);
    ASSERT_EQ(detached, &p1_slot);
    ASSERT_TRUE(sched.advance_ed_publish_scan(*detached));
    sched.enqueue_early_dispatch_candidate(*detached);
    ASSERT_EQ(p1_slot.to_payload().early_dispatch_state.load(std::memory_order_relaxed), EARLY_DISPATCH_STAGING);

    // p1's (gated) blocks publish while it is STAGING: the seal fires anyway —
    // publication counts placement, not launch — and hands C to the drain.
    publish_blocks(p1_slot, 1);
    EXPECT_EQ(p1_slot.ed_publish_list_head.load(std::memory_order_relaxed), ED_PUBLISH_LIST_SENTINEL);
    ASSERT_EQ(sched.ed_publish_drain_queue.pop_batch(&detached, 1), 1);
    ASSERT_EQ(detached, &c_slot);
    ASSERT_TRUE(sched.advance_ed_publish_scan(*detached));
    sched.enqueue_early_dispatch_candidate(*detached);
    EXPECT_EQ(c_slot.to_payload().early_dispatch_state.load(std::memory_order_relaxed), EARLY_DISPATCH_STAGING);
}

// Untracked producers must not pay the publish accounting, and the release
// path's NONE->DISPATCHED claim keeps a never-staged task on the normal route.
TEST_F(HbgEdQualificationTest, ReleaseClaimsNeverStagedTaskForNormalRoute) {
    TaskId p1 = submit_producer(/*flagged=*/false);
    ChipTaskSlotState &p1_slot = sm_handle->header->tasks.get_slot_state_by_task_id(p1.local_id());
    // Untracked: publication leaves no publish state behind.
    publish_blocks(p1_slot, 1);
    EXPECT_EQ(p1_slot.to_payload().published_block_count.load(std::memory_order_relaxed), 0);
    EXPECT_EQ(p1_slot.ed_publish_list_head.load(std::memory_order_relaxed), nullptr);

    TaskId p2 = submit_producer(/*flagged=*/true);
    TaskId deps[] = {p2};
    TaskId c = submit_consumer(deps, 1);
    ChipTaskSlotState &c_slot = sm_handle->header->tasks.get_slot_state_by_task_id(c.local_id());
    // Never staged: release wins NONE->DISPATCHED and reports "route normally".
    EXPECT_FALSE(sched.try_early_dispatch_release(c_slot));
    EXPECT_EQ(c_slot.to_payload().early_dispatch_state.load(std::memory_order_relaxed), EARLY_DISPATCH_DISPATCHED);
    // A later enqueue attempt (stale drain verdict) is dropped by the claim.
    sched.enqueue_early_dispatch_candidate(c_slot);
    EXPECT_EQ(c_slot.to_payload().early_dispatch_state.load(std::memory_order_relaxed), EARLY_DISPATCH_DISPATCHED);
}
