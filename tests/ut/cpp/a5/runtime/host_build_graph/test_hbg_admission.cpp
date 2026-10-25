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

#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <numeric>
#include <thread>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_admission.h"

namespace {

struct AdmissionShape {
    uint16_t blocks{2};
    uint8_t mask{1};
    uint8_t extra_flags{0};
};

SchedulerSpmdRequirements admission_requirements(const std::vector<AdmissionShape> &shapes, uint64_t workers) {
    SchedulerSpmdRequirements requirements{workers, workers / 3, 0, 0};
    for (const auto &shape : shapes) {
        const bool mix = __builtin_popcount(shape.mask) > 1;
        if (shape.blocks > 1 || mix || (shape.extra_flags & SCHEDULER_TASK_SYNC_START) != 0)
            ++requirements.cooperative_task_count;
        if (mix) ++requirements.mix_task_count;
    }
    return requirements;
}

struct AdmissionFixture {
    explicit AdmissionFixture(
        const std::vector<AdmissionShape> &shapes, uint64_t workers = 6, uint64_t epoch = 17,
        std::vector<uint64_t> permutation = {}, bool protocol_profiling = false
    ) :
        storage(shapes.size(), workers, 3, admission_requirements(shapes, workers), protocol_profiling),
        graph(shapes.size()),
        admissions(workers),
        locals(workers) {
        for (uint64_t task = 0; task < shapes.size(); ++task) {
            const auto &shape = shapes[task];
            graph.mixed(task, shape.mask);
            auto &metadata = storage.metadata[task];
            metadata.logical_block_num = shape.blocks;
            metadata.active_mask = shape.mask;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE | shape.extra_flags;
            if (shape.blocks > 1) metadata.flags |= SCHEDULER_TASK_SPMD;
            if (__builtin_popcount(shape.mask) > 1) metadata.flags |= SCHEDULER_TASK_MIX;
            metadata.trace_index_base = task * 3;
            metadata.timing_slot = -1;
            for (uint8_t lane = 0; lane < 3; ++lane)
                metadata.kernel_ids[lane] = (shape.mask & (1U << lane)) != 0 ? 1 : UINT16_MAX;
        }
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, epoch));
        if (permutation.empty()) {
            permutation.resize(workers);
            std::iota(permutation.begin(), permutation.end(), UINT64_C(0));
        }
        EXPECT_EQ(permutation.size(), workers);
        storage.run_control->active_worker_count = workers;
        storage.run_control->aic_active_worker_count = workers / 3;
        storage.run_control->aiv_active_worker_count = workers / 3 * 2;
        for (uint64_t position = 0; position < workers; ++position) {
            const uint64_t worker = permutation[position];
            const uint64_t cluster = position / 3;
            const uint64_t lane = position % 3;
            auto &context = storage.contexts[worker];
            context.core_type = static_cast<int32_t>(lane == 0 ? CoreType::AIC : CoreType::AIV);
            context.physical_core_id = static_cast<int32_t>(1000 + worker * 113);
            context.cluster_index = cluster;
            context.cluster_count = workers / 3;
            context.type_rank = static_cast<int32_t>(lane == 0 ? cluster : cluster * 2 + lane - 1);
            context.worker_index = worker;
            context.scheduler_index = worker;
            context.scheduler_count = workers;
            context.is_scheduler = 1;
            for (uint64_t peer = 0; peer < 3; ++peer)
                context.cluster_worker_ids[peer] = permutation[cluster * 3 + peer];
            storage.test_graphs[worker] = graph.graph();
        }
        for (uint64_t worker = 0; worker < workers; ++worker) {
            locals[worker] = storage.local_context(&storage.contexts[worker]);
            EXPECT_TRUE(scheduler_initialize_admission(base(), locals[worker], &admissions[worker]));
        }
    }

    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    SchedulerAdmissionControl *control() const {
        return scheduler_state_at<SchedulerAdmissionControl>(base(), storage.layout.admission_control_offset);
    }
    SchedulerAdmissionRecord *record(uint64_t task) const {
        return scheduler_state_at<SchedulerAdmissionRecord>(
            base(), storage.layout.admission_records_offset + task * storage.layout.admission_record_stride
        );
    }
    SchedulerTaskControl *task_control(uint64_t task) const {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    uint64_t *members(uint64_t task) const {
        return scheduler_state_at<uint64_t>(base(), record(task)->member_words_offset);
    }
    int64_t *log() const { return scheduler_state_at<int64_t>(base(), storage.layout.admission_log_offset); }

    bool ready_pop(uint64_t task, uint64_t worker, SchedulerReadyClaim *claim) {
        auto *local = locals[worker];
        if (scheduler_route_task(graph.graph(), base(), local, run(), task, nullptr) !=
            SchedulerRouteResult::READY_TO_ENQUEUE)
            return false;
        const auto &metadata = storage.metadata[task];
        const uint32_t queue = scheduler_task_ready_queue(metadata.flags, metadata.active_mask);
        if (!scheduler_ready_queue_holder(queue, local->config.self_core_type)) return false;
        const uint64_t target = scheduler_ready_push_target(local, queue, static_cast<int64_t>(task));
        SchedulerReadyBatch batch{};
        if (!scheduler_ready_batch_append(base(), local, task, &batch, nullptr) ||
            !scheduler_ready_batch_push(base(), local, queue, &batch, nullptr))
            return false;
        *claim = {};
        claim->inbox_index = target;
        return scheduler_ready_pop_from_inbox(
                   graph.graph(), base(), local, run(), queue, target, &claim->task_ref, nullptr
               ) &&
               claim->task_ref == static_cast<int64_t>(task);
    }

    SchedulerAdmissionResult claim(uint64_t worker, const SchedulerReadyClaim &ready) {
        return scheduler_try_claim_admission(graph.graph(), base(), locals[worker], run(), &admissions[worker], ready);
    }
    bool prepare(uint64_t worker) {
        return scheduler_prepare_admission_record(graph.graph(), base(), locals[worker], run(), &admissions[worker]);
    }
    SchedulerAdmissionResult commit(uint64_t worker) {
        return scheduler_try_commit_admission(graph.graph(), base(), locals[worker], run(), &admissions[worker]);
    }
    uint64_t publisher(uint64_t task) const {
        const auto &metadata = storage.metadata[task];
        const uint32_t queue = scheduler_task_ready_queue(metadata.flags, metadata.active_mask);
        for (uint64_t worker = 0; worker < locals.size(); ++worker)
            if (scheduler_ready_queue_holder(queue, locals[worker]->config.self_core_type)) return worker;
        return UINT64_MAX;
    }
    bool publish(uint64_t task, uint64_t worker = UINT64_MAX) {
        if (worker == UINT64_MAX) worker = publisher(task);
        if (worker >= locals.size()) return false;
        SchedulerReadyClaim ready{};
        return ready_pop(task, worker, &ready) && claim(worker, ready) == SchedulerAdmissionResult::PROGRESS &&
               prepare(worker) && commit(worker) == SchedulerAdmissionResult::PROGRESS;
    }
    bool refresh(uint64_t worker) {
        return scheduler_refresh_local_obligation(graph.graph(), base(), locals[worker], run(), &admissions[worker]);
    }
    bool ordinary_permit(uint64_t worker, bool *permit) {
        return scheduler_can_enter_ordinary(graph.graph(), base(), locals[worker], run(), &admissions[worker], permit);
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::GraphBuffer graph;
    std::vector<SchedulerAdmissionLocalState> admissions;
    std::vector<SchedulerLocalState *> locals;
};

TEST(HbgAdmission, PrepareDoesNotPublishTailOrLogAndCommitTransfersPendingOwnership) {
    AdmissionFixture fixture({{5, 1}});
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
    EXPECT_EQ(
        scheduler_gm_fetch_add(fixture.task_control(0)->admission_state, UINT64_C(0)),
        static_cast<uint64_t>(SchedulerAdmissionState::CLAIMED)
    );
    ASSERT_TRUE(fixture.prepare(0));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.record(0)->order_key, UINT64_MAX);
    EXPECT_EQ(fixture.record(0)->run_epoch, 17u);
    EXPECT_EQ(fixture.record(0)->member_count, 2u);
    EXPECT_EQ(fixture.members(0)[0], UINT64_C(0x9));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->next_block, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->completed_blocks, UINT64_C(0)), 0u);
    EXPECT_EQ(
        scheduler_gm_fetch_add(fixture.task_control(0)->admission_state, UINT64_C(0)),
        static_cast<uint64_t>(SchedulerAdmissionState::PREPARED)
    );
    ASSERT_EQ(fixture.commit(0), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), 0);
    EXPECT_EQ(fixture.record(0)->order_key, 0u);
    EXPECT_EQ(
        scheduler_gm_fetch_add(fixture.task_control(0)->admission_state, UINT64_C(0)),
        static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
    );
}

TEST(HbgAdmission, ContendedCommitRetainsPreparedCandidateWithoutCreatingALogHole) {
    AdmissionFixture fixture({{3, 1}, {4, 1}});
    SchedulerReadyClaim first{};
    SchedulerReadyClaim second{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &first));
    ASSERT_TRUE(fixture.ready_pop(1, 3, &second));
    ASSERT_EQ(fixture.claim(0, first), SchedulerAdmissionResult::PROGRESS);
    ASSERT_TRUE(fixture.prepare(0));
    ASSERT_EQ(fixture.claim(3, second), SchedulerAdmissionResult::PROGRESS);
    ASSERT_TRUE(fixture.prepare(3));
    scheduler_gm_store(fixture.control()->commit_lock, UINT64_C(1));
    for (uint32_t attempt = 0; attempt < 3; ++attempt) {
        EXPECT_EQ(fixture.commit(0), SchedulerAdmissionResult::RETRY);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
        EXPECT_EQ(fixture.record(0)->order_key, UINT64_MAX);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
        EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), SCHEDULER_TASK_REF_INVALID);
    }
    scheduler_gm_store(fixture.control()->commit_lock, UINT64_C(0));
    ASSERT_EQ(fixture.commit(3), SchedulerAdmissionResult::PROGRESS);
    ASSERT_EQ(fixture.commit(0), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), 1);
    EXPECT_EQ(scheduler_gm_load(fixture.log()[1]), 0);
    EXPECT_EQ(fixture.record(1)->order_key, 0u);
    EXPECT_EQ(fixture.record(0)->order_key, 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_lock, UINT64_C(0)), 0u);
}

TEST(HbgAdmission, ExistingPendingCandidateDoesNotConsumeAnotherReadyOwnership) {
    AdmissionFixture fixture({{2, 1}, {3, 1}});
    SchedulerReadyClaim first{};
    SchedulerReadyClaim second{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &first));
    ASSERT_TRUE(fixture.ready_pop(1, 0, &second));
    ASSERT_EQ(fixture.claim(0, first), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.claim(0, second), SchedulerAdmissionResult::RETRY);
    EXPECT_EQ(second.task_ref, 1);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
    EXPECT_EQ(
        scheduler_gm_fetch_add(fixture.task_control(1)->admission_state, UINT64_C(0)),
        static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    ASSERT_TRUE(fixture.prepare(0));
    ASSERT_EQ(fixture.commit(0), SchedulerAdmissionResult::PROGRESS);
    ASSERT_EQ(fixture.claim(0, second), SchedulerAdmissionResult::PROGRESS);
    ASSERT_TRUE(fixture.prepare(0));
    ASSERT_EQ(fixture.commit(0), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 2u);
}

TEST(HbgAdmission, DuplicateTaskOwnershipIsRejectedAcrossWorkers) {
    AdmissionFixture fixture({{2, 1}});
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.claim(3, ready), SchedulerAdmissionResult::ERROR);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
    EXPECT_EQ(fixture.admissions[3].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgAdmission, ReadyTaskBMayCommitBeforeTaskAWhoseDependencyIsNotReady) {
    AdmissionFixture fixture({{1, 1}, {3, 1}, {2, 1}});
    fixture.graph.executable(1, 0, {0});
    fixture.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 17));
    ASSERT_EQ(
        scheduler_route_task(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1, nullptr),
        SchedulerRouteResult::WAITING
    );
    ASSERT_TRUE(fixture.publish(2));
    EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), 2);
    EXPECT_EQ(fixture.record(2)->order_key, 0u);
    EXPECT_EQ(fixture.record(1)->order_key, UINT64_MAX);
    EXPECT_EQ(
        scheduler_gm_fetch_add(fixture.task_control(1)->admission_state, UINT64_C(0)),
        static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
    );
    ASSERT_TRUE(fixture.refresh(3));
    EXPECT_EQ(fixture.admissions[3].current_task_ref, 2);
}

class HbgAdmissionMembership : public ::testing::TestWithParam<uint8_t> {};

TEST_P(HbgAdmissionMembership, FreezesCompactWorkerMembershipFromTopologyNotPhysicalIds) {
    const uint8_t mask = GetParam();
    AdmissionFixture fixture({{3, mask}}, 6, 19, {4, 0, 3, 1, 5, 2});
    ASSERT_TRUE(fixture.publish(0));
    const uint64_t expected = mask == 1                     ? UINT64_C(0x12) :
                              __builtin_popcount(mask) == 1 ? UINT64_C(0x2d) :
                                                              UINT64_C(0x3f);
    EXPECT_EQ(fixture.members(0)[0], expected);
    EXPECT_EQ(fixture.record(0)->member_count, static_cast<uint32_t>(__builtin_popcountll(expected)));
    for (uint64_t worker = 0; worker < 6; ++worker) {
        SCOPED_TRACE(worker);
        ASSERT_TRUE(fixture.refresh(worker));
        const bool member = (expected & (UINT64_C(1) << worker)) != 0;
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, member ? 0 : SCHEDULER_TASK_REF_INVALID);
        bool permit = true;
        ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
        EXPECT_EQ(permit, !member);
        if (member) {
            EXPECT_EQ(fixture.admissions[worker].current_block_num, 3u);
            EXPECT_EQ(fixture.admissions[worker].current_active_mask, mask);
        }
    }
}

INSTANTIATE_TEST_SUITE_P(PureAndMixMasks, HbgAdmissionMembership, ::testing::Values<uint8_t>(1, 2, 4, 3, 5, 6, 7));

TEST(HbgAdmission, MembershipAcrossTheSixtyFourthWorkerUsesASecondWord) {
    AdmissionFixture fixture({{3, 1}}, 66);
    ASSERT_TRUE(fixture.publish(0));
    uint64_t expected_first = 0;
    for (uint32_t worker = 0; worker < 64; worker += 3)
        expected_first |= UINT64_C(1) << worker;
    EXPECT_EQ(fixture.storage.layout.admission_member_word_count, 2u);
    EXPECT_EQ(fixture.members(0)[0], expected_first);
    EXPECT_EQ(fixture.members(0)[1], 0u);
    EXPECT_EQ(fixture.record(0)->member_count, 22u);
    ASSERT_TRUE(fixture.refresh(63));
    EXPECT_EQ(fixture.admissions[63].current_task_ref, 0);
    ASSERT_TRUE(fixture.refresh(64));
    EXPECT_EQ(fixture.admissions[64].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    bool permit = false;
    ASSERT_TRUE(fixture.ordinary_permit(64, &permit));
    EXPECT_TRUE(permit);
}

TEST(HbgAdmission, VectorMembershipSetsWorkerBitsBeyondTheFirstWord) {
    AdmissionFixture fixture({{3, 2}}, 66);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.members(0)[1], UINT64_C(3));
    EXPECT_EQ(fixture.record(0)->member_count, 44u);
    for (uint64_t worker : {64u, 65u}) {
        ASSERT_TRUE(fixture.refresh(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        bool permit = true;
        ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
        EXPECT_FALSE(permit);
    }
}

TEST(HbgAdmission, MixMaskSixIncludesTheNonExecutingAICCoordinator) {
    AdmissionFixture fixture({{4, 6}});
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.members(0)[0], UINT64_C(0x3f));
    EXPECT_EQ(fixture.record(0)->member_count, 6u);
    for (uint64_t worker : {0u, 3u}) {
        ASSERT_TRUE(fixture.refresh(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[worker].current_active_mask, 6u);
        bool permit = true;
        ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
        EXPECT_FALSE(permit);
    }
}

TEST(HbgAdmission, EpochZeroAndMaximumAreExactIdentitiesNotSentinels) {
    for (uint64_t epoch : {UINT64_C(0), UINT64_C(1), UINT64_MAX - 1, UINT64_MAX}) {
        SCOPED_TRACE(epoch);
        AdmissionFixture fixture({{2, 1}}, 6, epoch);
        EXPECT_EQ(fixture.admissions[0].epoch, epoch);
        ASSERT_TRUE(fixture.publish(0));
        EXPECT_EQ(fixture.record(0)->run_epoch, epoch);
        ASSERT_TRUE(fixture.refresh(3));
        EXPECT_EQ(fixture.admissions[3].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[3].epoch, epoch);
    }
}

TEST(HbgAdmission, UninitializedZeroEpochStateCannotBypassAPublishedObligation) {
    AdmissionFixture fixture({{2, 1}}, 6, 0);
    ASSERT_TRUE(fixture.publish(0));
    SchedulerAdmissionLocalState uninitialized{};
    bool permit = true;
    EXPECT_FALSE(scheduler_can_enter_ordinary(
        fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run(), &uninitialized, &permit
    ));
    EXPECT_FALSE(permit);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgAdmission, OrdinaryPermitRejectsAPrefixMadeStaleByAnotherWorkerCommit) {
    AdmissionFixture fixture({{2, 1}});
    ASSERT_TRUE(fixture.refresh(3));
    ASSERT_TRUE(fixture.refresh(1));
    for (uint64_t worker : {3u, 1u}) {
        bool permit = false;
        ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
        EXPECT_TRUE(permit);
    }
    ASSERT_TRUE(fixture.publish(0));
    for (uint64_t worker : {3u, 1u}) {
        bool permit = true;
        ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
        EXPECT_FALSE(permit);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        ASSERT_TRUE(fixture.refresh(worker));
        ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
        EXPECT_EQ(permit, worker == 1);
    }
}

TEST(HbgAdmission, PendingClaimAndPreparedRecordCannotBeCarriedIntoAnOrdinaryKernel) {
    AdmissionFixture fixture({{2, 1}});
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
    bool permit = true;
    ASSERT_TRUE(fixture.ordinary_permit(0, &permit));
    EXPECT_FALSE(permit);
    ASSERT_TRUE(fixture.prepare(0));
    permit = true;
    ASSERT_TRUE(fixture.ordinary_permit(0, &permit));
    EXPECT_FALSE(permit);
    ASSERT_EQ(fixture.commit(0), SchedulerAdmissionResult::PROGRESS);
    permit = true;
    ASSERT_TRUE(fixture.ordinary_permit(0, &permit));
    EXPECT_FALSE(permit);
    ASSERT_TRUE(fixture.refresh(0));
    ASSERT_TRUE(fixture.ordinary_permit(0, &permit));
    EXPECT_FALSE(permit);
}

TEST(HbgAdmission, GraphWithoutCooperativeTasksNeedsNoProtocolAllocationForOrdinaryPermit) {
    AdmissionFixture fixture({{1, 1}, {1, 2}});
    EXPECT_EQ(fixture.storage.layout.admission_control_offset, 0u);
    for (uint64_t worker = 0; worker < fixture.locals.size(); ++worker) {
        EXPECT_EQ(fixture.admissions[worker].capacity, 0u);
        ASSERT_TRUE(fixture.refresh(worker));
        bool permit = false;
        ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
        EXPECT_TRUE(permit);
    }
}

TEST(HbgAdmission, LegacyZeroSummaryWithoutProtocolRegionsStillInitializes) {
    scheduler_test::FixtureStorage storage(1, 3);
    auto *local = storage.local_context(&storage.contexts[0]);
    SchedulerAdmissionLocalState admission{};
    ASSERT_TRUE(scheduler_initialize_admission(storage.scheduler_state->base(), local, &admission));
    EXPECT_TRUE(admission.initialized);
    EXPECT_EQ(admission.capacity, 0u);
    EXPECT_EQ(admission.control_offset, 0u);
    bool permit = false;
    ASSERT_TRUE(scheduler_can_enter_ordinary(
        *local->config.graph, storage.scheduler_state->base(), local, storage.run_control, &admission, &permit
    ));
    EXPECT_TRUE(permit);
}

TEST(HbgAdmission, RefreshStopsAtFirstMemberAndDoesNotTreatTaskDoneAsLocalDone) {
    AdmissionFixture fixture({{2, 1}, {3, 1}, {4, 1}});
    ASSERT_TRUE(fixture.publish(2));
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.publish(1));
    ASSERT_TRUE(fixture.refresh(3));
    EXPECT_EQ(fixture.admissions[3].current_task_ref, 2);
    const uint64_t cursor = fixture.admissions[3].scan_cursor;
    scheduler_gm_store(fixture.task_control(2)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    ASSERT_TRUE(fixture.refresh(3));
    EXPECT_EQ(fixture.admissions[3].current_task_ref, 2);
    EXPECT_EQ(fixture.admissions[3].scan_cursor, cursor);
    bool permit = true;
    ASSERT_TRUE(fixture.ordinary_permit(3, &permit));
    EXPECT_FALSE(permit);
}

TEST(HbgAdmission, BoundedRefreshCannotGrantPermitWithUnprocessedNonmemberPrefix) {
    AdmissionFixture fixture(std::vector<AdmissionShape>(19, AdmissionShape{2, 1}));
    for (uint64_t task = 0; task < 19; ++task)
        ASSERT_TRUE(fixture.publish(task));
    ASSERT_TRUE(fixture.refresh(1));
    EXPECT_GT(fixture.admissions[1].scan_cursor, 0u);
    EXPECT_LE(fixture.admissions[1].scan_cursor, 8u);
    bool permit = true;
    ASSERT_TRUE(fixture.ordinary_permit(1, &permit));
    EXPECT_FALSE(permit);
    for (uint32_t step = 0; step < 4 && fixture.admissions[1].scan_cursor < 19; ++step)
        ASSERT_TRUE(fixture.refresh(1));
    EXPECT_EQ(fixture.admissions[1].scan_cursor, 19u);
    EXPECT_EQ(fixture.admissions[1].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(fixture.ordinary_permit(1, &permit));
    EXPECT_TRUE(permit);
}

struct SyncCapacityCase {
    uint8_t mask;
    uint16_t blocks;
    bool accepted;
};

class HbgAdmissionSyncCapacity : public ::testing::TestWithParam<SyncCapacityCase> {};

TEST_P(HbgAdmissionSyncCapacity, PreparationCountsDistinctExecutionPositionsNotPendingSlots) {
    const auto parameters = GetParam();
    AdmissionFixture fixture({{parameters.blocks, parameters.mask, SCHEDULER_TASK_SYNC_START}});
    const uint64_t publisher = fixture.publisher(0);
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, publisher, &ready));
    ASSERT_EQ(fixture.claim(publisher, ready), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.prepare(publisher), parameters.accepted);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), SCHEDULER_TASK_REF_INVALID);
}

INSTANTIATE_TEST_SUITE_P(
    DistinctCoreAndClusterBounds, HbgAdmissionSyncCapacity,
    ::testing::Values(
        SyncCapacityCase{1, 2, true}, SyncCapacityCase{1, 3, false}, SyncCapacityCase{2, 4, true},
        SyncCapacityCase{2, 5, false}, SyncCapacityCase{4, 4, true}, SyncCapacityCase{4, 5, false},
        SyncCapacityCase{3, 2, true}, SyncCapacityCase{3, 3, false}, SyncCapacityCase{5, 2, true},
        SyncCapacityCase{5, 3, false}, SyncCapacityCase{6, 2, true}, SyncCapacityCase{6, 3, false},
        SyncCapacityCase{7, 2, true}, SyncCapacityCase{7, 3, false}
    )
);

TEST(HbgAdmission, OrdinaryMultiBlockPreparationDoesNotRequireAllBlocksToFitAtOnce) {
    for (uint8_t mask : {1u, 2u, 4u, 3u, 5u, 6u, 7u}) {
        SCOPED_TRACE(static_cast<uint32_t>(mask));
        AdmissionFixture fixture({{512, mask}});
        ASSERT_TRUE(fixture.publish(0));
        EXPECT_EQ(fixture.record(0)->logical_block_num, 512u);
    }
}

TEST(HbgAdmission, NonzeroBlockCountersAreRejectedWithoutResettingTheirEvidence) {
    for (const auto &counts : std::vector<std::array<uint64_t, 2>>{{1, 0}, {0, 1}, {2, 2}, {UINT64_MAX, UINT64_MAX}}) {
        SCOPED_TRACE(counts[0]);
        SCOPED_TRACE(counts[1]);
        AdmissionFixture fixture({{3, 1}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        scheduler_gm_store(fixture.task_control(0)->next_block, counts[0]);
        scheduler_gm_store(fixture.task_control(0)->completed_blocks, counts[1]);
        EXPECT_FALSE(fixture.prepare(0));
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->next_block, UINT64_C(0)), counts[0]);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->completed_blocks, UINT64_C(0)), counts[1]);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
        EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    }
}

TEST(HbgAdmission, InvalidTaskReferenceAndNoncooperativeTaskCannotAcquireOwnership) {
    for (int64_t task_ref : {INT64_C(-2), INT64_C(-1), INT64_C(1), INT64_MAX}) {
        SCOPED_TRACE(task_ref);
        AdmissionFixture fixture({{2, 1}});
        SchedulerReadyClaim invalid{};
        invalid.task_ref = task_ref;
        EXPECT_EQ(fixture.claim(0, invalid), SchedulerAdmissionResult::ERROR);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    }
    AdmissionFixture fixture({{2, 1}, {1, 2}});
    SchedulerReadyClaim ordinary{};
    ASSERT_TRUE(fixture.ready_pop(1, 1, &ordinary));
    EXPECT_EQ(fixture.claim(1, ordinary), SchedulerAdmissionResult::ERROR);
    EXPECT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(HbgAdmission, FullCommitLogIsAnErrorAndNeverWrapsOrDropsPendingCandidate) {
    for (uint64_t tail : {UINT64_C(1), UINT64_C(2), UINT64_MAX}) {
        SCOPED_TRACE(tail);
        AdmissionFixture fixture({{2, 1}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        ASSERT_TRUE(fixture.prepare(0));
        scheduler_gm_store(fixture.control()->commit_tail, tail);
        EXPECT_EQ(fixture.commit(0), SchedulerAdmissionResult::ERROR);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), tail);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_lock, UINT64_C(0)), 0u);
        EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(HbgAdmission, EpochMismatchBeforeCommitDoesNotPublishARecord) {
    for (uint64_t epoch : {UINT64_C(0), UINT64_MAX}) {
        AdmissionFixture fixture({{2, 1}}, 6, epoch);
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        ASSERT_TRUE(fixture.prepare(0));
        fixture.record(0)->run_epoch = epoch == 0 ? UINT64_MAX : 0;
        EXPECT_EQ(fixture.commit(0), SchedulerAdmissionResult::ERROR);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
        EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), SCHEDULER_TASK_REF_INVALID);
    }
}

enum class RecordCorruption {
    TASK_REFERENCE,
    ORDER_KEY,
    BLOCK_COUNT,
    ACTIVE_MASK,
    FLAGS,
    MEMBER_OFFSET,
    MEMBER_COUNT,
    OUT_OF_RANGE_MEMBER,
    MISSING_MEMBER,
    WRONG_CORE_TYPE_MEMBER,
    SEAL_OFFSET,
};

void corrupt_record(AdmissionFixture &fixture, RecordCorruption corruption, bool prepared) {
    auto *record = fixture.record(0);
    switch (corruption) {
    case RecordCorruption::TASK_REFERENCE:
        record->task_ref = SCHEDULER_TASK_REF_INVALID;
        break;
    case RecordCorruption::ORDER_KEY:
        record->order_key = prepared ? 1 : 0;
        break;
    case RecordCorruption::BLOCK_COUNT:
        ++record->logical_block_num;
        break;
    case RecordCorruption::ACTIVE_MASK:
        record->active_mask = 2;
        break;
    case RecordCorruption::FLAGS:
        record->flags ^= SCHEDULER_TASK_SPMD;
        break;
    case RecordCorruption::MEMBER_OFFSET:
        record->member_words_offset = 0;
        break;
    case RecordCorruption::MEMBER_COUNT:
        ++record->member_count;
        break;
    case RecordCorruption::OUT_OF_RANGE_MEMBER:
        fixture.members(0)[0] |= UINT64_C(1) << 63;
        break;
    case RecordCorruption::MISSING_MEMBER:
        fixture.members(0)[0] &= ~UINT64_C(1);
        if (record->member_count != 0) --record->member_count;
        break;
    case RecordCorruption::WRONG_CORE_TYPE_MEMBER:
        fixture.members(0)[0] = UINT64_C(0x12);
        record->member_count = 2;
        break;
    case RecordCorruption::SEAL_OFFSET:
        record->mix_close_seals_offset = fixture.storage.layout.admission_records_offset;
        break;
    }
}

class HbgAdmissionPreparedCorruption : public ::testing::TestWithParam<RecordCorruption> {};

TEST_P(HbgAdmissionPreparedCorruption, CorruptPreparedRecordCannotAdvanceCommitTail) {
    AdmissionFixture fixture({{3, 1}});
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
    ASSERT_TRUE(fixture.prepare(0));
    corrupt_record(fixture, GetParam(), true);
    EXPECT_EQ(fixture.commit(0), SchedulerAdmissionResult::ERROR);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
}

TEST_P(HbgAdmissionPreparedCorruption, CorruptPublishedRecordCannotBecomeLocalObligation) {
    AdmissionFixture fixture({{3, 1}});
    ASSERT_TRUE(fixture.publish(0));
    corrupt_record(fixture, GetParam(), true);
    EXPECT_FALSE(fixture.refresh(3));
    EXPECT_EQ(fixture.admissions[3].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    IdentityShapeAndFrozenMembership, HbgAdmissionPreparedCorruption,
    ::testing::Values(
        RecordCorruption::TASK_REFERENCE, RecordCorruption::ORDER_KEY, RecordCorruption::BLOCK_COUNT,
        RecordCorruption::ACTIVE_MASK, RecordCorruption::FLAGS, RecordCorruption::MEMBER_OFFSET,
        RecordCorruption::MEMBER_COUNT, RecordCorruption::OUT_OF_RANGE_MEMBER, RecordCorruption::MISSING_MEMBER,
        RecordCorruption::WRONG_CORE_TYPE_MEMBER, RecordCorruption::SEAL_OFFSET
    )
);

TEST(HbgAdmission, DirtyUnpublishedRecordIsRejectedBeforePreparation) {
    for (RecordCorruption corruption : {
             RecordCorruption::TASK_REFERENCE,
             RecordCorruption::ORDER_KEY,
             RecordCorruption::BLOCK_COUNT,
             RecordCorruption::ACTIVE_MASK,
             RecordCorruption::FLAGS,
             RecordCorruption::MEMBER_OFFSET,
             RecordCorruption::MEMBER_COUNT,
             RecordCorruption::OUT_OF_RANGE_MEMBER,
             RecordCorruption::SEAL_OFFSET,
         }) {
        SCOPED_TRACE(static_cast<uint32_t>(corruption));
        AdmissionFixture fixture({{3, 1}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        corrupt_record(fixture, corruption, false);
        EXPECT_FALSE(fixture.prepare(0));
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
        EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    }
}

TEST(HbgAdmission, InvalidPublishedLogReferenceFailsWithoutAdvancingWorkerFrontier) {
    for (int64_t task_ref : {INT64_C(-2), INT64_C(-1), INT64_C(1), INT64_MAX}) {
        AdmissionFixture fixture({{2, 1}});
        ASSERT_TRUE(fixture.publish(0));
        scheduler_gm_store(fixture.log()[0], task_ref);
        EXPECT_FALSE(fixture.refresh(3));
        EXPECT_EQ(fixture.admissions[3].scan_cursor, 0u);
        EXPECT_EQ(fixture.admissions[3].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(HbgAdmission, CorruptPublishedEpochCannotBeAcceptedByAMemberOrNonmember) {
    for (uint64_t worker : {1u, 3u}) {
        AdmissionFixture fixture({{2, 1}}, 6, UINT64_MAX);
        ASSERT_TRUE(fixture.publish(0));
        fixture.record(0)->run_epoch = 0;
        EXPECT_FALSE(fixture.refresh(worker));
        EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(HbgAdmission, InvalidTailOrCursorFailsClosedBeforeGrantingOrdinaryPermit) {
    for (bool corrupt_tail : {false, true}) {
        AdmissionFixture fixture({{2, 1}});
        if (corrupt_tail) scheduler_gm_store(fixture.control()->commit_tail, UINT64_C(2));
        else fixture.admissions[1].scan_cursor = 1;
        bool permit = false;
        EXPECT_FALSE(fixture.ordinary_permit(1, &permit));
        EXPECT_FALSE(permit);
        EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    }
}

TEST(HbgAdmission, InvalidTopologyCannotFreezeMembership) {
    for (bool corrupt_index : {false, true}) {
        AdmissionFixture fixture({{3, 1}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        if (corrupt_index) fixture.storage.contexts[3].worker_index = 0;
        else fixture.storage.contexts[3].core_type = static_cast<int32_t>(CoreType::AIV);
        EXPECT_FALSE(fixture.prepare(0));
        EXPECT_EQ(fixture.record(0)->member_count, 0u);
        EXPECT_EQ(fixture.members(0)[0], 0u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    }
}

TEST(HbgAdmission, InitializationRejectsInvalidLayoutWithoutChangingOutput) {
    for (uint32_t corruption = 0; corruption < 10; ++corruption) {
        SCOPED_TRACE(corruption);
        AdmissionFixture fixture({{2, 1}});
        auto *summary =
            scheduler_state_at<SchedulerGraphSummary>(fixture.base(), fixture.storage.layout.graph_summary_offset);
        switch (corruption) {
        case 0:
            ++summary->worker_count;
            break;
        case 1:
            ++summary->cluster_count;
            break;
        case 2:
            summary->cooperative_task_count = 2;
            break;
        case 3:
            summary->mix_task_count = 2;
            break;
        case 4:
            summary->admission_member_word_count = 0;
            break;
        case 5:
            ++summary->admission_control_offset;
            break;
        case 6:
            summary->admission_record_stride = 64;
            break;
        case 7:
            summary->admission_records_offset = summary->admission_control_offset;
            break;
        case 8:
            summary->admission_log_offset = summary->admission_records_offset;
            break;
        case 9:
            summary->admission_record_stride = UINT64_MAX - 127;
            break;
        }
        SchedulerAdmissionLocalState output{};
        output.epoch = 991;
        output.pending_task_ref = 441;
        std::array<uint8_t, sizeof(output)> before{};
        std::memcpy(before.data(), &output, sizeof(output));
        EXPECT_FALSE(scheduler_initialize_admission(fixture.base(), fixture.locals[0], &output));
        EXPECT_EQ(std::memcmp(before.data(), &output, sizeof(output)), 0);
    }
}

TEST(HbgAdmission, InitializationRejectsMissingOrUnalignedInputsWithoutChangingOutput) {
    AdmissionFixture fixture({{2, 1}});
    SchedulerAdmissionLocalState output{};
    output.epoch = 883;
    std::array<uint8_t, sizeof(output)> before{};
    std::memcpy(before.data(), &output, sizeof(output));
    EXPECT_FALSE(scheduler_initialize_admission(nullptr, fixture.locals[0], &output));
    EXPECT_FALSE(scheduler_initialize_admission(fixture.base(), nullptr, &output));
    EXPECT_FALSE(scheduler_initialize_admission(fixture.base(), fixture.locals[0], nullptr));
    EXPECT_FALSE(
        scheduler_initialize_admission(static_cast<uint8_t *>(fixture.base()) + 1, fixture.locals[0], &output)
    );
    auto invalid_local = *fixture.locals[0];
    invalid_local.config.graph = nullptr;
    EXPECT_FALSE(scheduler_initialize_admission(fixture.base(), &invalid_local, &output));
    EXPECT_EQ(std::memcmp(before.data(), &output, sizeof(output)), 0);
}

TEST(HbgAdmission, ConcurrentRealCommitsPublishUniqueCompleteContiguousLogEntries) {
    constexpr uint64_t thread_count = 12;
    constexpr uint64_t tasks_per_thread = 4;
    constexpr uint64_t task_count = thread_count * tasks_per_thread;
    constexpr uint64_t iteration_budget = 10000000;
    AdmissionFixture fixture(std::vector<AdmissionShape>(task_count, AdmissionShape{2, 1}), thread_count * 3, 71);
    std::vector<SchedulerReadyClaim> claims(task_count);
    for (uint64_t task = 0; task < task_count; ++task)
        ASSERT_TRUE(fixture.ready_pop(task, (task / tasks_per_thread) * 3, &claims[task]));
    std::atomic<uint64_t> at_start{0};
    std::atomic<uint64_t> errors{0};
    std::atomic<bool> start{false};
    std::atomic<bool> finished{false};
    std::atomic<bool> reader_observed_prefix{false};
    uint64_t reader_refreshes = 0;
    std::thread reader([&] {
        at_start.fetch_add(1, std::memory_order_release);
        while (!start.load(std::memory_order_acquire)) {}
        for (uint64_t attempt = 0; attempt < iteration_budget; ++attempt) {
            if (errors.load(std::memory_order_acquire) != 0) return;
            const uint64_t previous = fixture.admissions[1].scan_cursor;
            if (!fixture.refresh(1)) {
                errors.fetch_add(1, std::memory_order_release);
                return;
            }
            ++reader_refreshes;
            const uint64_t cursor = fixture.admissions[1].scan_cursor;
            if (cursor < previous || cursor > task_count ||
                fixture.admissions[1].current_task_ref != SCHEDULER_TASK_REF_INVALID) {
                errors.fetch_add(1, std::memory_order_release);
                return;
            }
            if (cursor != 0 && cursor < task_count) reader_observed_prefix.store(true, std::memory_order_release);
            if (finished.load(std::memory_order_acquire) && cursor == task_count) return;
        }
        errors.fetch_add(1, std::memory_order_release);
    });
    std::vector<std::thread> threads;
    for (uint64_t thread = 0; thread < thread_count; ++thread) {
        threads.emplace_back([&, thread] {
            const uint64_t worker = thread * 3;
            at_start.fetch_add(1, std::memory_order_release);
            while (!start.load(std::memory_order_acquire)) {}
            for (uint64_t index = 0; index < tasks_per_thread; ++index) {
                const uint64_t task = thread * tasks_per_thread + index;
                if (fixture.claim(worker, claims[task]) != SchedulerAdmissionResult::PROGRESS ||
                    !fixture.prepare(worker)) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                bool committed = false;
                for (uint64_t attempt = 0; attempt < 1000000; ++attempt) {
                    const auto result = fixture.commit(worker);
                    if (result == SchedulerAdmissionResult::PROGRESS) {
                        committed = true;
                        break;
                    }
                    if (result == SchedulerAdmissionResult::ERROR) break;
                }
                if (!committed) {
                    errors.fetch_add(1, std::memory_order_relaxed);
                    return;
                }
                if (index == 0) {
                    uint64_t attempt = 0;
                    for (; attempt < iteration_budget; ++attempt) {
                        if (errors.load(std::memory_order_acquire) != 0) return;
                        if (reader_observed_prefix.load(std::memory_order_acquire)) break;
                    }
                    if (attempt == iteration_budget) {
                        errors.fetch_add(1, std::memory_order_release);
                        return;
                    }
                }
            }
        });
    }
    while (at_start.load(std::memory_order_acquire) != thread_count + 1) {}
    start.store(true, std::memory_order_release);
    for (auto &thread : threads)
        thread.join();
    finished.store(true, std::memory_order_release);
    reader.join();
    ASSERT_EQ(errors.load(), 0u);
    EXPECT_TRUE(reader_observed_prefix.load());
    EXPECT_GT(reader_refreshes, 0u);
    EXPECT_EQ(fixture.admissions[1].scan_cursor, task_count);
    EXPECT_EQ(fixture.admissions[1].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    bool permit = false;
    ASSERT_TRUE(fixture.ordinary_permit(1, &permit));
    EXPECT_TRUE(permit);
    ASSERT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    ASSERT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), task_count);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_lock, UINT64_C(0)), 0u);
    std::vector<bool> seen(task_count, false);
    for (uint64_t order = 0; order < task_count; ++order) {
        SCOPED_TRACE(order);
        const int64_t task = scheduler_gm_load(fixture.log()[order]);
        ASSERT_GE(task, 0);
        ASSERT_LT(static_cast<uint64_t>(task), task_count);
        EXPECT_FALSE(seen[task]);
        seen[task] = true;
        EXPECT_EQ(fixture.record(task)->order_key, order);
        EXPECT_EQ(fixture.record(task)->run_epoch, 71u);
        EXPECT_EQ(fixture.record(task)->member_count, thread_count);
        EXPECT_EQ(
            scheduler_gm_fetch_add(fixture.task_control(task)->admission_state, UINT64_C(0)),
            static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
        );
    }
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) {
        return value;
    }));
    for (uint64_t thread = 0; thread < thread_count; ++thread)
        EXPECT_EQ(fixture.admissions[thread * 3].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(HbgAdmissionProtocol, SuccessfulPublicationRecordsActualPublisherOrderAndEpoch) {
    for (uint64_t epoch : {UINT64_C(0), UINT64_MAX}) {
        AdmissionFixture fixture({{2, 1}, {2, 2}}, 6, epoch, {}, true);
        auto *tasks = scheduler_state_at<SchedulerProtocolTaskTrace>(
            fixture.base(), fixture.storage.layout.protocol.tasks_offset
        );
        ASSERT_TRUE(fixture.publish(1, 1));
        ASSERT_TRUE(fixture.publish(0, 3));
        EXPECT_EQ(tasks[1].admission.before.sequence, 1u);
        EXPECT_EQ(tasks[1].admission.after.sequence, 2u);
        EXPECT_EQ(tasks[1].admission.task_ref, 1);
        EXPECT_EQ(tasks[1].admission.order_key, 0u);
        EXPECT_EQ(tasks[1].admission.worker_index, 1u);
        EXPECT_EQ(tasks[1].admission.run_epoch, epoch);
        EXPECT_EQ(tasks[0].admission.before.sequence, 1u);
        EXPECT_EQ(tasks[0].admission.after.sequence, 2u);
        EXPECT_EQ(tasks[0].admission.order_key, 1u);
        EXPECT_EQ(tasks[0].admission.worker_index, 3u);
        EXPECT_EQ(tasks[0].completion.before.sequence, 0u);
        EXPECT_EQ(tasks[1].completion.after.sequence, 0u);
    }
}

TEST(HbgAdmissionProtocol, CommitRetryDoesNotInventASequenceOrPublishAnInterval) {
    AdmissionFixture fixture({{2, 1}}, 6, 17, {}, true);
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
    ASSERT_TRUE(fixture.prepare(0));
    auto *task =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    scheduler_gm_store(fixture.control()->commit_lock, UINT64_C(4));
    EXPECT_EQ(fixture.commit(0), SchedulerAdmissionResult::RETRY);
    EXPECT_EQ(task->admission.before.sequence, 0u);
    EXPECT_EQ(task->admission.after.sequence, 0u);
    EXPECT_EQ(workers[0].next_sequence, 0u);
    scheduler_gm_store(fixture.control()->commit_lock, UINT64_C(0));
    EXPECT_EQ(fixture.commit(0), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(task->admission.before.sequence, 1u);
    EXPECT_EQ(task->admission.after.sequence, 2u);
    EXPECT_EQ(workers[0].next_sequence, 2u);
}

TEST(HbgAdmissionProtocol, MemberObservationIsUniqueAndNonMembersProduceNoObligation) {
    AdmissionFixture fixture({{2, 1}, {2, 2}}, 6, 17, {}, true);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.publish(1));
    ASSERT_TRUE(fixture.refresh(3));
    ASSERT_TRUE(fixture.refresh(1));
    auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    EXPECT_EQ(members[3].observed.sequence, 1u);
    EXPECT_EQ(members[3].task_ref, 0);
    EXPECT_EQ(members[3].observed_tail, 2u);
    EXPECT_EQ(members[3].worker_index, 3u);
    EXPECT_EQ(members[1].observed.sequence, 0u);
    EXPECT_EQ(members[6 + 1].observed.sequence, 3u);
    EXPECT_EQ(members[6 + 1].task_ref, 1);
    EXPECT_EQ(members[6 + 1].order_key, 1u);
    EXPECT_EQ(members[6 + 1].observed_tail, 2u);
    const auto observed = members[3].observed.sequence;
    ASSERT_TRUE(fixture.refresh(3));
    EXPECT_EQ(members[3].observed.sequence, observed);
}

TEST(HbgAdmissionProtocol, ZeroWorkMemberClosesAtCursorExhaustionBeforeTaskDone) {
    AdmissionFixture fixture({{2, 1}}, 6, 17, {}, true);
    ASSERT_TRUE(fixture.publish(0));
    for (uint64_t worker : {UINT64_C(0), UINT64_C(3)}) {
        fixture.locals[worker]->admission = &fixture.admissions[worker];
        ASSERT_TRUE(fixture.refresh(worker));
    }
    auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    bool closed = true;
    ASSERT_TRUE(scheduler_close_ordered_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run(), &closed
    ));
    EXPECT_FALSE(closed);
    EXPECT_EQ(members[3].closed.sequence, 0u);
    for (uint32_t index = 0; index < 2; ++index) {
        uint32_t block = UINT32_MAX;
        ASSERT_EQ(
            scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &block
            ),
            SchedulerBlockClaimResult::CLAIMED
        );
        EXPECT_EQ(block, index);
    }
    ASSERT_TRUE(scheduler_close_ordered_obligation(
        fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run(), &closed
    ));
    EXPECT_TRUE(closed);
    EXPECT_EQ(members[3].closed.sequence, 2u);
    EXPECT_EQ(members[3].closed_cursor, 2u);
    EXPECT_EQ(members[3].pending_mask, 0u);
    EXPECT_EQ(members[3].close_seal, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->completed_blocks, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.admissions[3].scan_cursor, 1u);
}

TEST(HbgAdmissionProtocol, DuplicateAndWrongIdentityKeepTheFirstRecordAndSetStickyErrors) {
    AdmissionFixture fixture({{2, 1}}, 6, 0, {}, true);
    ASSERT_TRUE(fixture.publish(0));
    auto *task =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    const auto first = task->admission;
    scheduler_protocol_admission(fixture.base(), fixture.locals[0], 0, 0, 255, true);
    scheduler_protocol_admission(fixture.base(), fixture.locals[0], 0, 0, 255, false);
    scheduler_protocol_admission(fixture.base(), fixture.locals[0], -1, 0, 0, true);
    EXPECT_EQ(std::memcmp(&task->admission, &first, sizeof(first)), 0);
    auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_DUPLICATE | SCHEDULER_PROTOCOL_INVALID_IDENTITY);
    EXPECT_EQ(worker->next_sequence, 2u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgAdmissionProtocol, SequenceNeverWrapsAndAnIncompletePublicationStaysIncomplete) {
    AdmissionFixture fixture({{2, 1}}, 6, UINT64_MAX, {}, true);
    auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    worker->next_sequence = UINT64_MAX - 1;
    ASSERT_TRUE(fixture.publish(0));
    const auto *task =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    EXPECT_EQ(task->admission.before.sequence, UINT64_MAX);
    EXPECT_EQ(task->admission.after.sequence, 0u);
    EXPECT_EQ(worker->next_sequence, UINT64_MAX);
    EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_SEQUENCE_OVERFLOW);
    EXPECT_EQ(task->admission.run_epoch, UINT64_MAX);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 1u);
}

TEST(HbgAdmissionProtocol, FinishedWorkerRejectsLateEventsAndFailedExitIsNotSuccessful) {
    for (bool success : {false, true}) {
        AdmissionFixture fixture({{2, 1}}, 6, 17, {}, true);
        scheduler_protocol_finish(fixture.base(), fixture.locals[0], success);
        auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
            fixture.base(), fixture.storage.layout.protocol.workers_offset
        );
        EXPECT_EQ(worker->finished, success ? 1u : 2u);
        scheduler_protocol_admission(fixture.base(), fixture.locals[0], 0, 0, 0, true);
        EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_AFTER_FINISH);
        EXPECT_EQ(worker->next_sequence, 0u);
        scheduler_protocol_finish(fixture.base(), fixture.locals[0], !success);
        EXPECT_EQ(worker->finished, success ? 1u : 2u);
        EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_AFTER_FINISH | SCHEDULER_PROTOCOL_DUPLICATE);
    }
}

TEST(HbgAdmissionProtocol, DisabledCaptureDoesNotAllocateOrTouchProtocolState) {
    AdmissionFixture fixture({{2, 1}});
    EXPECT_EQ(fixture.storage.layout.protocol_layout_offset, 0u);
    EXPECT_EQ(fixture.locals[0]->profiling->protocol_layout_offset, 0u);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.refresh(0));
    scheduler_protocol_finish(fixture.base(), fixture.locals[0], true);
    EXPECT_EQ(fixture.locals[0]->profiling->protocol_layout_offset, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

class HbgProtocolLayoutCorruption : public ::testing::TestWithParam<int> {};

TEST(HbgAdmissionProtocol, CompactWorkerBeyondFirstBitmapWordKeepsItsActualIdentity) {
    AdmissionFixture fixture({{2, 2}}, 66, 0, {}, true);
    ASSERT_TRUE(fixture.publish(0, 65));
    ASSERT_TRUE(fixture.refresh(65));
    const auto *tasks =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    const auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    EXPECT_EQ(tasks[0].admission.worker_index, 65u);
    EXPECT_EQ(tasks[0].admission.after.sequence, 2u);
    EXPECT_EQ(members[65].worker_index, 65u);
    EXPECT_EQ(members[65].observed.sequence, 3u);
    EXPECT_EQ(members[65].run_epoch, 0u);
    EXPECT_EQ(members[65].observed_tail, 1u);
    EXPECT_NE(fixture.storage.contexts[65].physical_core_id, 65);
}

TEST_P(HbgProtocolLayoutCorruption, BootstrapRejectsBadLayoutAndRetainedWorkerState) {
    AdmissionFixture fixture({{2, 1}}, 6, 17, {}, true);
    auto *layout =
        scheduler_state_at<SchedulerProtocolLayout>(fixture.base(), fixture.storage.layout.protocol_layout_offset);
    auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(fixture.base(), layout->workers_offset);
    switch (GetParam()) {
    case 0:
        layout->magic ^= 1;
        break;
    case 1:
        layout->run_epoch ^= 1;
        break;
    case 2:
        ++layout->task_count;
        break;
    case 3:
        --layout->worker_count;
        break;
    case 4:
        ++layout->cooperative_task_count;
        break;
    case 5:
        ++layout->member_count;
        break;
    case 6:
        layout->tasks_offset = layout->members_offset;
        break;
    case 7:
        layout->members_offset = layout->blocks_offset;
        break;
    case 8:
        layout->blocks_offset = layout->executions_offset;
        break;
    case 9:
        layout->executions_offset = layout->workers_offset;
        break;
    case 10:
        ++layout->workers_offset;
        break;
    case 11:
        --layout->storage_end_offset;
        break;
    case 12:
        layout->trace_count = UINT64_MAX;
        break;
    case 13:
        worker->run_epoch ^= 1;
        break;
    case 14:
        ++worker->worker_index;
        break;
    case 15:
        worker->next_sequence = 1;
        break;
    case 16:
        worker->error_flags = 1;
        break;
    case 17:
        worker->finished = 1;
        break;
    case 18:
        fixture.locals[0]->profiling = nullptr;
        break;
    }
    SchedulerAdmissionLocalState admission{};
    EXPECT_FALSE(scheduler_initialize_admission(fixture.base(), fixture.locals[0], &admission));
    EXPECT_FALSE(admission.initialized);
    if (fixture.locals[0]->profiling != nullptr) EXPECT_EQ(fixture.locals[0]->profiling->protocol_layout_offset, 0u);
}

INSTANTIATE_TEST_SUITE_P(InvalidProtocolStorage, HbgProtocolLayoutCorruption, ::testing::Range(0, 19));

}  // namespace
