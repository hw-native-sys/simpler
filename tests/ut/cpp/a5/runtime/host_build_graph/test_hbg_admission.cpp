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
#include <string>
#include <thread>
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_admission.h"

namespace {

struct AdmissionShape {
    uint16_t blocks{2};
    LaneMask mask{LaneMask::aic()};
    uint8_t extra_flags{0};
};

constexpr LaneMask kAicAiv0{/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false};
constexpr LaneMask kAicAiv1{/*aic=*/true, /*aiv0=*/false, /*aiv1=*/true};
// Every non-empty lane set a task can carry.
constexpr std::array<LaneMask, 7> kAllMasks{LaneMask::aic(), LaneMask::aiv0(),     LaneMask::aiv1(), kAicAiv0,
                                            kAicAiv1,        LaneMask::both_aiv(), LaneMask::all()};

// gtest cannot stream a LaneMask, so SCOPED_TRACE names the lanes instead.
std::string lane_names(LaneMask lanes) {
    std::string name;
    if (lanes.has_aic()) name += "aic";
    if (lanes.has_aiv0()) name += name.empty() ? "aiv0" : "+aiv0";
    if (lanes.has_aiv1()) name += name.empty() ? "aiv1" : "+aiv1";
    return name.empty() ? "none" : name;
}

SchedulerSpmdRequirements admission_requirements(const std::vector<AdmissionShape> &shapes, uint64_t workers) {
    SchedulerSpmdRequirements requirements{workers, workers / 3, 0, 0};
    for (const auto &shape : shapes) {
        const bool mix = shape.mask.is_mix();
        if (shape.blocks > 1 || mix || (shape.extra_flags & SCHEDULER_TASK_SYNC_START) != 0)
            ++requirements.cooperative_task_count;
        if (mix) ++requirements.mix_task_count;
    }
    return requirements;
}

struct AdmissionFixture {
    explicit AdmissionFixture(
        const std::vector<AdmissionShape> &shapes, uint64_t workers = 6, uint64_t epoch = 17,
        std::vector<uint64_t> permutation = {}, bool protocol_profiling = false,
        SchedulerTopologyLayout topology_layout = SchedulerTopologyLayout::GENERAL
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
            if (shape.mask.is_mix()) metadata.flags |= SCHEDULER_TASK_MIX;
            metadata.trace_index_base = task * 3;
            metadata.timing_slot = -1;
            for (uint8_t lane = 0; lane < 3; ++lane)
                metadata.kernel_ids[lane] = shape.mask.has(static_cast<SubtaskSlot>(lane)) ? 1 : UINT16_MAX;
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
        scheduler_state_at<SchedulerGraphSummary>(base(), storage.layout.graph_summary_offset)->topology_layout =
            topology_layout;
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
    SchedulerAdmissionResult publish_pending(uint64_t worker) {
        locals[worker]->admission = &admissions[worker];
        return scheduler_ordered_publish_pending(graph.graph(), base(), locals[worker], run());
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
    AdmissionFixture fixture({{5, LaneMask::aic()}});
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
    EXPECT_EQ(
        scheduler_gm_query(fixture.task_control(0)->admission_state),
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
        scheduler_gm_query(fixture.task_control(0)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::PREPARED)
    );
    ASSERT_EQ(fixture.commit(0), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.log()[0]), 0);
    EXPECT_EQ(fixture.record(0)->order_key, 0u);
    EXPECT_EQ(
        scheduler_gm_query(fixture.task_control(0)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
    );
}

TEST(HbgAdmission, ContendedCommitRetainsPreparedCandidateWithoutCreatingALogHole) {
    AdmissionFixture fixture({{3, LaneMask::aic()}, {4, LaneMask::aic()}});
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
    AdmissionFixture fixture({{2, LaneMask::aic()}, {3, LaneMask::aic()}});
    SchedulerReadyClaim first{};
    SchedulerReadyClaim second{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &first));
    ASSERT_TRUE(fixture.ready_pop(1, 0, &second));
    ASSERT_EQ(fixture.claim(0, first), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.claim(0, second), SchedulerAdmissionResult::RETRY);
    EXPECT_EQ(second.task_ref, 1);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
    EXPECT_EQ(
        scheduler_gm_query(fixture.task_control(1)->admission_state),
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

TEST(HbgAdmission, PublishedRunErrorRejectsReadyOwnershipWithoutChangingTheCandidate) {
    AdmissionFixture fixture({{2, LaneMask::aic()}});
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[3],
        SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
    );
    EXPECT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::ERROR);
    EXPECT_EQ(ready.task_ref, 0);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.task_control(0)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED));
    EXPECT_EQ(fixture.control()->commit_tail, 0u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ORDERED_EXIT_TIMEOUT));
}

TEST(HbgAdmission, PublishedRunErrorPreventsPreparedCandidateFromCommitting) {
    AdmissionFixture fixture({{2, LaneMask::aic()}});
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
    ASSERT_TRUE(fixture.prepare(0));
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[3],
        SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
    );
    EXPECT_EQ(fixture.commit(0), SchedulerAdmissionResult::ERROR);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
    EXPECT_EQ(fixture.task_control(0)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::PREPARED));
    EXPECT_EQ(fixture.control()->commit_tail, 0u);
    EXPECT_EQ(fixture.control()->commit_lock, 0u);
    EXPECT_EQ(fixture.log()[0], SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.record(0)->order_key, UINT64_MAX);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ORDERED_EXIT_TIMEOUT));
}

TEST(HbgAdmission, DuplicateTaskOwnershipIsRejectedAcrossWorkers) {
    AdmissionFixture fixture({{2, LaneMask::aic()}});
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
    ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
    EXPECT_EQ(fixture.claim(3, ready), SchedulerAdmissionResult::ERROR);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
    EXPECT_EQ(fixture.admissions[3].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgAdmission, BusyClaimLockNeverTransfersPendingOwnershipOrReleasesAnotherOwner) {
    for (uint64_t owner : {UINT64_C(1), UINT64_C(64), UINT64_MAX}) {
        AdmissionFixture fixture({{2, LaneMask::aic()}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        scheduler_gm_store(fixture.task_control(0)->admission_claim_lock, owner);
        EXPECT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::ERROR);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->admission_claim_lock, UINT64_C(0)), owner);
        EXPECT_EQ(scheduler_gm_query(fixture.task_control(0)->admission_state), 0u);
        EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ADMISSION_INVALID_STATE));
    }
}

TEST(HbgAdmission, LateClaimRechecksPhaseUnderLockAndReleasesItsOwnLockOnRejection) {
    for (uint64_t phase : {UINT64_C(1), UINT64_C(2), UINT64_C(3), UINT64_MAX}) {
        AdmissionFixture fixture({{2, LaneMask::aic()}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        scheduler_publish_signal(fixture.task_control(0)->admission_state, phase);
        EXPECT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::ERROR);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->admission_claim_lock, UINT64_C(0)), 0u);
        EXPECT_EQ(scheduler_gm_query(fixture.task_control(0)->admission_state), phase);
        EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ADMISSION_INVALID_STATE));
    }
}

TEST(HbgAdmission, SuccessfulClaimPublishesPhaseBeforeReturningWithoutHoldingTheLock) {
    for (LaneMask mask : kAllMasks) {
        AdmissionFixture fixture({{2, mask}});
        const uint64_t worker = fixture.publisher(0);
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, worker, &ready));
        ASSERT_EQ(fixture.claim(worker, ready), SchedulerAdmissionResult::PROGRESS);
        EXPECT_EQ(scheduler_gm_query(fixture.task_control(0)->admission_state), 1u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->admission_claim_lock, UINT64_C(0)), 0u);
        ASSERT_TRUE(fixture.prepare(worker));
        EXPECT_EQ(scheduler_gm_query(fixture.task_control(0)->admission_state), 2u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->admission_claim_lock, UINT64_C(0)), 0u);
        ASSERT_EQ(fixture.commit(worker), SchedulerAdmissionResult::PROGRESS);
        EXPECT_EQ(scheduler_gm_query(fixture.task_control(0)->admission_state), 3u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->admission_claim_lock, UINT64_C(0)), 0u);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgAdmission, ConcurrentDuplicateClaimCannotAcquireTwoPendingOwnersOrLeakTheClaimLock) {
    for (uint32_t repeat = 0; repeat < 32; ++repeat) {
        AdmissionFixture fixture({{2, LaneMask::aic()}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        std::atomic<uint32_t> arrived{0};
        std::array<SchedulerAdmissionResult, 2> results{};
        std::vector<std::thread> claimers;
        for (uint32_t index = 0; index < 2; ++index)
            claimers.emplace_back([&, index] {
                arrived.fetch_add(1, std::memory_order_release);
                while (arrived.load(std::memory_order_acquire) != 2) {}
                results[index] = fixture.claim(index * 3, ready);
            });
        for (auto &thread : claimers)
            thread.join();
        uint32_t owners = 0;
        for (uint32_t index = 0; index < 2; ++index) {
            const bool owns = fixture.admissions[index * 3].pending_task_ref == 0;
            EXPECT_EQ(results[index] == SchedulerAdmissionResult::PROGRESS, owns);
            owners += owns;
        }
        EXPECT_LE(owners, 1u);
        EXPECT_EQ(scheduler_gm_query(fixture.task_control(0)->admission_state), owners == 0 ? 0u : 1u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.task_control(0)->admission_claim_lock, UINT64_C(0)), 0u);
        EXPECT_EQ(scheduler_gm_query(fixture.control()->commit_tail), 0u);
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgAdmission, PublishedTailCannotHideAnInvalidPhaseForMembersOrNonmembers) {
    for (uint64_t worker : {UINT64_C(0), UINT64_C(1)}) {
        for (uint64_t phase : {UINT64_C(0), UINT64_C(1), UINT64_C(2), UINT64_MAX}) {
            AdmissionFixture fixture({{2, LaneMask::aic()}});
            ASSERT_TRUE(fixture.publish(0));
            scheduler_publish_signal(fixture.task_control(0)->admission_state, phase);
            EXPECT_FALSE(fixture.refresh(worker));
            EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
            EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
            EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ADMISSION_INVALID_RECORD));
        }
    }
}

TEST(HbgAdmission, ReadyTaskBMayCommitBeforeTaskAWhoseDependencyIsNotReady) {
    AdmissionFixture fixture({{1, LaneMask::aic()}, {3, LaneMask::aic()}, {2, LaneMask::aic()}});
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
        scheduler_gm_query(fixture.task_control(1)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED)
    );
    ASSERT_TRUE(fixture.refresh(3));
    EXPECT_EQ(fixture.admissions[3].current_task_ref, 2);
}

class HbgAdmissionMembership : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgAdmissionMembership, FreezesCompactWorkerMembershipFromTopologyNotPhysicalIds) {
    const LaneMask mask = GetParam();
    AdmissionFixture fixture({{3, mask}}, 6, 19, {4, 0, 3, 1, 5, 2});
    ASSERT_TRUE(fixture.publish(0));
    const uint64_t expected = mask == LaneMask::aic() ? UINT64_C(0x12) :
                              mask.count() == 1       ? UINT64_C(0x2d) :
                              mask == kAicAiv0        ? UINT64_C(0x33) :
                              mask == kAicAiv1        ? UINT64_C(0x1e) :
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

INSTANTIATE_TEST_SUITE_P(PureAndMixMasks, HbgAdmissionMembership, ::testing::ValuesIn(kAllMasks));

TEST(HbgAdmission, MembershipAcrossTheSixtyFourthWorkerUsesASecondWord) {
    AdmissionFixture fixture({{3, LaneMask::aic()}}, 66);
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
    AdmissionFixture fixture({{3, LaneMask::aiv0()}}, 66);
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

TEST(HbgAdmission, PublishedSecondBitmapWordDistinguishesMembersAndNonmembers) {
    for (LaneMask mask :
         {LaneMask::aic(), LaneMask::aiv0(), kAicAiv0, kAicAiv1, LaneMask::both_aiv(), LaneMask::all()}) {
        SCOPED_TRACE(lane_names(mask));
        AdmissionFixture fixture({{3, mask}}, 69);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_EQ(fixture.admissions[0].member_word_count, 2u);
        for (uint64_t worker : {63u, 64u, 65u, 66u, 67u, 68u}) {
            SCOPED_TRACE(worker);
            const bool member = mask.count() > 1 ?
                                    mask.merged_with(LaneMask::aic()).has(static_cast<SubtaskSlot>(worker % 3)) :
                                    (mask == LaneMask::aic() ? worker % 3 == 0 : worker % 3 != 0);
            ASSERT_TRUE(fixture.refresh(worker));
            const auto &admission = fixture.admissions[worker];
            EXPECT_EQ(admission.current_task_ref, member ? 0 : SCHEDULER_TASK_REF_INVALID);
            EXPECT_EQ(admission.scan_cursor, member ? 0u : 1u);
            if (member) {
                EXPECT_EQ(admission.current_block_num, 3u);
                EXPECT_EQ(admission.current_active_mask, mask);
            }
            bool permit = true;
            ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
            EXPECT_EQ(permit, !member);
        }
    }
}

TEST(HbgAdmission, CorruptSecondBitmapWordIsRejectedByMembersAndNonmembersWithColdOrWarmTopology) {
    for (bool warm_topology : {false, true}) {
        SCOPED_TRACE(warm_topology);
        for (uint64_t worker : {64u, 66u}) {
            SCOPED_TRACE(worker);
            for (uint32_t corruption = 0; corruption < 3; ++corruption) {
                SCOPED_TRACE(corruption);
                AdmissionFixture fixture({{3, LaneMask::aic()}}, 69);
                ASSERT_TRUE(fixture.publish(0));
                ASSERT_EQ(fixture.members(0)[1], UINT64_C(4));
                if (warm_topology) {
                    uint64_t expected[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
                    uint32_t members = 0;
                    ASSERT_TRUE(scheduler_admission_expected_members(
                        fixture.base(), fixture.locals[worker], &fixture.admissions[worker], fixture.record(0),
                        expected, &members
                    ));
                    ASSERT_EQ(fixture.admissions[worker].topology_worker_count, 69u);
                }
                if (corruption == 0) fixture.members(0)[1] = 0;
                else if (corruption == 1) fixture.members(0)[1] = UINT64_C(1);
                else fixture.members(0)[1] |= UINT64_C(1) << 63;
                EXPECT_FALSE(fixture.refresh(worker));
                EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
                EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
                EXPECT_EQ(
                    fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ADMISSION_INVALID_RECORD)
                );
            }
        }
    }
}

TEST(HbgAdmission, InvalidPublishedMemberOffsetsFailClosedForMembersAndNonmembers) {
    for (uint64_t worker : {64u, 66u}) {
        SCOPED_TRACE(worker);
        for (uint32_t corruption = 0; corruption < 7; ++corruption) {
            SCOPED_TRACE(corruption);
            AdmissionFixture fixture({{3, LaneMask::aic()}, {3, LaneMask::aic()}}, 69);
            ASSERT_TRUE(fixture.publish(0));
            const uint64_t expected = fixture.record(0)->member_words_offset;
            const uint64_t offsets[] = {
                0,
                1,
                UINT64_MAX,
                UINT64_MAX - 63,
                expected + 8,
                expected - sizeof(SchedulerAdmissionRecord),
                fixture.record(1)->member_words_offset,
            };
            fixture.record(0)->member_words_offset = offsets[corruption];
            EXPECT_FALSE(fixture.refresh(worker));
            EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
            EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
            EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ADMISSION_INVALID_RECORD));
            EXPECT_EQ(scheduler_gm_query(fixture.control()->commit_tail), 1u);
            EXPECT_EQ(scheduler_gm_query(fixture.log()[0]), 0);
        }
    }
}

TEST(HbgAdmission, MixMaskSixIncludesTheNonExecutingAICCoordinator) {
    AdmissionFixture fixture({{4, LaneMask::both_aiv()}});
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.members(0)[0], UINT64_C(0x3f));
    EXPECT_EQ(fixture.record(0)->member_count, 6u);
    for (uint64_t worker : {0u, 3u}) {
        ASSERT_TRUE(fixture.refresh(worker));
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[worker].current_active_mask, LaneMask::both_aiv());
        bool permit = true;
        ASSERT_TRUE(fixture.ordinary_permit(worker, &permit));
        EXPECT_FALSE(permit);
    }
}

TEST(HbgAdmission, EpochZeroAndMaximumAreExactIdentitiesNotSentinels) {
    for (uint64_t epoch : {UINT64_C(0), UINT64_C(1), UINT64_MAX - 1, UINT64_MAX}) {
        SCOPED_TRACE(epoch);
        AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, epoch);
        EXPECT_EQ(fixture.admissions[0].epoch, epoch);
        ASSERT_TRUE(fixture.publish(0));
        EXPECT_EQ(fixture.record(0)->run_epoch, epoch);
        ASSERT_TRUE(fixture.refresh(3));
        EXPECT_EQ(fixture.admissions[3].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[3].epoch, epoch);
    }
}

TEST(HbgAdmission, UninitializedZeroEpochStateCannotBypassAPublishedObligation) {
    AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, 0);
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
    AdmissionFixture fixture({{2, LaneMask::aic()}});
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
    AdmissionFixture fixture({{2, LaneMask::aic()}});
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
    AdmissionFixture fixture({{1, LaneMask::aic()}, {1, LaneMask::aiv0()}});
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
    AdmissionFixture fixture({{2, LaneMask::aic()}, {3, LaneMask::aic()}, {4, LaneMask::aic()}});
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
    AdmissionFixture fixture(std::vector<AdmissionShape>(19, AdmissionShape{2, LaneMask::aic()}));
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
    LaneMask mask;
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
        SyncCapacityCase{LaneMask::aic(), 2, true}, SyncCapacityCase{LaneMask::aic(), 3, false},
        SyncCapacityCase{LaneMask::aiv0(), 4, true}, SyncCapacityCase{LaneMask::aiv0(), 5, false},
        SyncCapacityCase{LaneMask::aiv1(), 4, true}, SyncCapacityCase{LaneMask::aiv1(), 5, false},
        SyncCapacityCase{kAicAiv0, 2, true}, SyncCapacityCase{kAicAiv0, 3, false}, SyncCapacityCase{kAicAiv1, 2, true},
        SyncCapacityCase{kAicAiv1, 3, false}, SyncCapacityCase{LaneMask::both_aiv(), 2, true},
        SyncCapacityCase{LaneMask::both_aiv(), 3, false}, SyncCapacityCase{LaneMask::all(), 2, true},
        SyncCapacityCase{LaneMask::all(), 3, false}
    )
);

TEST(HbgAdmission, OrdinaryMultiBlockPreparationDoesNotRequireAllBlocksToFitAtOnce) {
    for (LaneMask mask : kAllMasks) {
        SCOPED_TRACE(lane_names(mask));
        AdmissionFixture fixture({{512, mask}});
        ASSERT_TRUE(fixture.publish(0));
        EXPECT_EQ(fixture.record(0)->logical_block_num, 512u);
    }
}

TEST(HbgAdmission, OrdinaryBlocksCanStayOnOneWorkerOrCrossDirectoryShards) {
    constexpr uint64_t workers = 3 * (SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD + 1);
    constexpr uint64_t first = 1;
    for (bool cross_shard : {false, true}) {
        SCOPED_TRACE(cross_shard);
        AdmissionFixture fixture({{3, LaneMask::aiv0()}}, workers, 17, {}, true);
        uint64_t remote = workers;
        for (uint64_t worker = 0; worker < workers; ++worker) {
            const auto &context = fixture.storage.contexts[worker];
            if (context.core_type == static_cast<int32_t>(CoreType::AIV) &&
                context.type_rank >= static_cast<int32_t>(SCHEDULER_READY_DIRECTORY_OWNERS_PER_SHARD)) {
                remote = worker;
                break;
            }
        }
        ASSERT_LT(remote, workers);
        ASSERT_EQ(fixture.storage.contexts[first].type_rank, 0);
        ASSERT_TRUE(fixture.publish(0, first));
        for (uint64_t worker : {first, remote}) {
            fixture.locals[worker]->admission = &fixture.admissions[worker];
            ASSERT_TRUE(fixture.refresh(worker));
            ASSERT_EQ(fixture.admissions[worker].current_task_ref, 0);
        }

        const std::array<uint64_t, 3> owners{first, cross_shard ? remote : first, first};
        const auto *blocks = scheduler_state_at<SchedulerProtocolBlockTrace>(
            fixture.base(), fixture.storage.layout.protocol.blocks_offset
        );
        for (uint32_t expected = 0; expected < owners.size(); ++expected) {
            uint32_t block = UINT32_MAX;
            ASSERT_EQ(
                scheduler_claim_ordered_cursor(
                    fixture.graph.graph(), fixture.base(), fixture.locals[owners[expected]], fixture.run(), &block
                ),
                SchedulerBlockClaimResult::CLAIMED
            );
            ASSERT_EQ(block, expected);
            EXPECT_EQ(blocks[block].owner.worker_index, owners[expected]);
            EXPECT_EQ(blocks[block].owner.task_ref, 0);
            EXPECT_EQ(blocks[block].owner.block_idx, block);
            EXPECT_EQ(blocks[block].owner.logical_block_num, owners.size());
        }
        for (uint64_t worker : {first, remote}) {
            uint32_t block = UINT32_MAX;
            EXPECT_EQ(
                scheduler_claim_ordered_cursor(
                    fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
                ),
                SchedulerBlockClaimResult::EXHAUSTED
            );
            EXPECT_EQ(block, UINT32_MAX);
        }
        EXPECT_EQ(fixture.task_control(0)->next_block, owners.size());
        EXPECT_EQ(fixture.task_control(0)->completed_blocks, 0u);
        EXPECT_EQ(fixture.run()->scheduler_error, 0u);
    }
}

TEST(HbgAdmission, NonzeroBlockCountersAreRejectedWithoutResettingTheirEvidence) {
    for (const auto &counts : std::vector<std::array<uint64_t, 2>>{{1, 0}, {0, 1}, {2, 2}, {UINT64_MAX, UINT64_MAX}}) {
        SCOPED_TRACE(counts[0]);
        SCOPED_TRACE(counts[1]);
        AdmissionFixture fixture({{3, LaneMask::aic()}});
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

TEST(HbgAdmission, PendingPublicationMatchesSeparatePreparationAndCommitAcrossShapesAndEpochs) {
    for (uint64_t epoch : {UINT64_C(0), UINT64_C(17), UINT64_MAX}) {
        for (uint64_t workers : {UINT64_C(6), UINT64_C(69)}) {
            for (const auto &shape : std::vector<AdmissionShape>{
                     {2, LaneMask::aic()},
                     {3, LaneMask::aiv0()},
                     {1, kAicAiv0},
                     {512, LaneMask::all()},
                     {2, LaneMask::both_aiv(), SCHEDULER_TASK_SYNC_START},
                     {4, LaneMask::aiv1(), SCHEDULER_TASK_SYNC_START},
                 }) {
                SCOPED_TRACE(epoch);
                SCOPED_TRACE(workers);
                SCOPED_TRACE(shape.mask.raw());
                AdmissionFixture pending({shape}, workers, epoch, {}, true);
                AdmissionFixture separate({shape}, workers, epoch, {}, true);
                const uint64_t worker = pending.publisher(0);
                SchedulerReadyClaim ready{};
                ASSERT_TRUE(pending.ready_pop(0, worker, &ready));
                ASSERT_EQ(pending.claim(worker, ready), SchedulerAdmissionResult::PROGRESS);
                ASSERT_EQ(pending.publish_pending(worker), SchedulerAdmissionResult::PROGRESS);
                ASSERT_TRUE(separate.publish(0, worker));
                EXPECT_EQ(pending.control()->commit_tail, 1u);
                EXPECT_EQ(pending.control()->commit_lock, 0u);
                EXPECT_EQ(pending.log()[0], 0);
                EXPECT_EQ(pending.admissions[worker].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
                EXPECT_EQ(pending.record(0)->run_epoch, epoch);
                EXPECT_EQ(pending.record(0)->order_key, 0u);
                EXPECT_EQ(pending.record(0)->member_count, separate.record(0)->member_count);
                for (uint64_t word = 0; word < pending.admissions[worker].member_word_count; ++word)
                    EXPECT_EQ(pending.members(0)[word], separate.members(0)[word]);
                EXPECT_EQ(pending.task_control(0)->claim_tickets, 0u);
                EXPECT_EQ(pending.task_control(0)->next_block, 0u);
                EXPECT_EQ(pending.task_control(0)->completed_blocks, 0u);
                EXPECT_EQ(
                    pending.task_control(0)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
                );
                const auto *trace = scheduler_state_at<SchedulerProtocolTaskTrace>(
                    pending.base(), pending.storage.layout.protocol.tasks_offset
                );
                EXPECT_EQ(trace->admission.before.sequence, 1u);
                EXPECT_EQ(trace->admission.after.sequence, 2u);
                EXPECT_EQ(pending.publish_pending(worker), SchedulerAdmissionResult::PROGRESS);
                EXPECT_EQ(pending.control()->commit_tail, 1u);
                EXPECT_EQ(trace->admission.after.sequence, 2u);
                for (uint64_t member = 0; member < workers; ++member) {
                    ASSERT_TRUE(pending.refresh(member));
                    ASSERT_TRUE(separate.refresh(member));
                    EXPECT_EQ(
                        pending.admissions[member].current_task_ref, separate.admissions[member].current_task_ref
                    );
                    EXPECT_EQ(pending.admissions[member].scan_cursor, separate.admissions[member].scan_cursor);
                }
                EXPECT_EQ(scheduler_read_error(pending.run()), 0u);
            }
        }
    }
}

TEST(HbgAdmission, PendingPublicationRejectsNonzeroCountersBeforePreparationAndAfterCommitRetry) {
    for (bool retry : {false, true}) {
        for (uint32_t counter = 0; counter < 3; ++counter) {
            for (uint64_t value : {UINT64_C(1), UINT64_MAX}) {
                SCOPED_TRACE(retry);
                SCOPED_TRACE(counter);
                SCOPED_TRACE(value);
                AdmissionFixture fixture({{3, LaneMask::all()}});
                SchedulerReadyClaim ready{};
                ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
                ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
                if (retry) {
                    scheduler_gm_store(fixture.control()->commit_lock, UINT64_C(4));
                    ASSERT_EQ(fixture.publish_pending(0), SchedulerAdmissionResult::RETRY);
                    EXPECT_EQ(fixture.control()->commit_lock, 4u);
                    scheduler_gm_store(fixture.control()->commit_lock, UINT64_C(0));
                }
                auto *task = fixture.task_control(0);
                volatile uint64_t *counts[] = {&task->claim_tickets, &task->next_block, &task->completed_blocks};
                scheduler_gm_store(*counts[counter], value);
                EXPECT_EQ(fixture.publish_pending(0), SchedulerAdmissionResult::ERROR);
                EXPECT_EQ(*counts[counter], value);
                EXPECT_EQ(fixture.control()->commit_tail, 0u);
                EXPECT_EQ(fixture.control()->commit_lock, 0u);
                EXPECT_EQ(fixture.log()[0], SCHEDULER_TASK_REF_INVALID);
                EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
                EXPECT_EQ(
                    fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ADMISSION_INVALID_STATE)
                );
            }
        }
    }
}

TEST(HbgAdmission, PendingPublicationKeepsFreshErrorAndFullLogGates) {
    for (uint32_t failure = 0; failure < 3; ++failure) {
        SCOPED_TRACE(failure);
        AdmissionFixture fixture({{2, LaneMask::all()}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        if (failure == 0) {
            scheduler_record_error(
                fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
                SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
            );
        } else {
            scheduler_gm_store(fixture.control()->commit_tail, failure == 1 ? UINT64_C(1) : UINT64_MAX);
        }
        const uint64_t tail = fixture.control()->commit_tail;
        EXPECT_EQ(fixture.publish_pending(0), SchedulerAdmissionResult::ERROR);
        EXPECT_EQ(fixture.control()->commit_tail, tail);
        EXPECT_EQ(fixture.control()->commit_lock, 0u);
        EXPECT_EQ(fixture.log()[0], SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
        EXPECT_NE(fixture.task_control(0)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
        EXPECT_EQ(
            fixture.run()->error_site,
            static_cast<uint64_t>(
                failure == 0 ? SchedulerErrorSite::ORDERED_EXIT_TIMEOUT : SchedulerErrorSite::ADMISSION_LOG_CAPACITY
            )
        );
    }
}

TEST(HbgAdmission, InvalidTaskReferenceAndNoncooperativeTaskCannotAcquireOwnership) {
    for (int64_t task_ref : {INT64_C(-2), INT64_C(-1), INT64_C(1), INT64_MAX}) {
        SCOPED_TRACE(task_ref);
        AdmissionFixture fixture({{2, LaneMask::aic()}});
        SchedulerReadyClaim invalid{};
        invalid.task_ref = task_ref;
        EXPECT_EQ(fixture.claim(0, invalid), SchedulerAdmissionResult::ERROR);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    }
    AdmissionFixture fixture({{2, LaneMask::aic()}, {1, LaneMask::aiv0()}});
    SchedulerReadyClaim ordinary{};
    ASSERT_TRUE(fixture.ready_pop(1, 1, &ordinary));
    EXPECT_EQ(fixture.claim(1, ordinary), SchedulerAdmissionResult::ERROR);
    EXPECT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(HbgAdmission, FullCommitLogIsAnErrorAndNeverWrapsOrDropsPendingCandidate) {
    for (uint64_t tail : {UINT64_C(1), UINT64_C(2), UINT64_MAX}) {
        SCOPED_TRACE(tail);
        AdmissionFixture fixture({{2, LaneMask::aic()}});
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
        AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, epoch);
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
        record->active_mask = LaneMask::aiv0();
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

TEST(HbgAdmission, MixCloseSealOffsetMustNameACompleteTaskRowNotAnInteriorCell) {
    for (uint64_t delta : {UINT64_C(1), UINT64_C(8), UINT64_C(16), UINT64_C(128), UINT64_C(512)}) {
        SCOPED_TRACE(delta);
        AdmissionFixture fixture({{1, kAicAiv0}, {1, LaneMask::all()}}, 6);
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        fixture.record(0)->mix_close_seals_offset = fixture.storage.layout.mix_close_seals_offset + delta;
        EXPECT_FALSE(fixture.prepare(0));
        EXPECT_EQ(scheduler_gm_query(fixture.control()->commit_tail), 0u);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
        EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ADMISSION_INVALID_RECORD));
    }
}

TEST_P(HbgAdmissionPreparedCorruption, CorruptPreparedRecordCannotAdvanceCommitTail) {
    AdmissionFixture fixture({{3, LaneMask::aic()}});
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

TEST_P(HbgAdmissionPreparedCorruption, PendingPublicationRevalidatesExplicitPreparationAndRetriedRecords) {
    for (bool retry : {false, true}) {
        SCOPED_TRACE(retry);
        AdmissionFixture fixture({{3, LaneMask::aic()}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        if (retry) {
            scheduler_gm_store(fixture.control()->commit_lock, UINT64_C(4));
            ASSERT_EQ(fixture.publish_pending(0), SchedulerAdmissionResult::RETRY);
            EXPECT_EQ(fixture.control()->commit_lock, 4u);
            scheduler_gm_store(fixture.control()->commit_lock, UINT64_C(0));
        } else {
            ASSERT_TRUE(fixture.prepare(0));
        }
        corrupt_record(fixture, GetParam(), true);
        EXPECT_EQ(fixture.publish_pending(0), SchedulerAdmissionResult::ERROR);
        EXPECT_EQ(fixture.control()->commit_tail, 0u);
        EXPECT_EQ(fixture.control()->commit_lock, 0u);
        EXPECT_EQ(fixture.log()[0], SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, 0);
    }
}

TEST_P(HbgAdmissionPreparedCorruption, CorruptPublishedRecordCannotBecomeLocalObligation) {
    AdmissionFixture fixture({{3, LaneMask::aic()}});
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
        AdmissionFixture fixture({{3, LaneMask::aic()}});
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
        AdmissionFixture fixture({{2, LaneMask::aic()}});
        ASSERT_TRUE(fixture.publish(0));
        scheduler_gm_store(fixture.log()[0], task_ref);
        EXPECT_FALSE(fixture.refresh(3));
        EXPECT_EQ(fixture.admissions[3].scan_cursor, 0u);
        EXPECT_EQ(fixture.admissions[3].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(HbgAdmission, CorruptPublishedEpochCannotBeAcceptedByAMemberOrNonmember) {
    for (uint64_t worker : {1u, 3u}) {
        AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, UINT64_MAX);
        ASSERT_TRUE(fixture.publish(0));
        fixture.record(0)->run_epoch = 0;
        EXPECT_FALSE(fixture.refresh(worker));
        EXPECT_EQ(fixture.admissions[worker].scan_cursor, 0u);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(HbgAdmission, InvalidTailOrCursorFailsClosedBeforeGrantingOrdinaryPermit) {
    for (bool corrupt_tail : {false, true}) {
        AdmissionFixture fixture({{2, LaneMask::aic()}});
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
        AdmissionFixture fixture({{3, LaneMask::aic()}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        if (corrupt_index) fixture.storage.contexts[3].worker_index = 0;
        else fixture.storage.contexts[3].core_type = static_cast<int32_t>(CoreType::AIV);
        EXPECT_FALSE(fixture.prepare(0));
        EXPECT_EQ(fixture.record(0)->member_count, 0u);
        EXPECT_EQ(fixture.members(0)[0], 0u);
        EXPECT_EQ(fixture.admissions[0].topology_worker_count, 0u);
        for (uint64_t word : fixture.admissions[0].topology_cube_words)
            EXPECT_EQ(word, 0u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    }
}

TEST(HbgAdmission, FrozenTopologyCacheIsReusedForDifferentShapesButNotAcrossRuns) {
    AdmissionFixture fixture(
        {{2, LaneMask::aic()}, {2, LaneMask::aiv0()}, {2, kAicAiv0}, {2, kAicAiv1}, {2, LaneMask::all()}}
    );
    auto &admission = fixture.admissions[0];
    const uint64_t expected[] = {0x9, 0x36, 0x1b, 0x2d, 0x3f};
    for (uint64_t task = 0; task < 5; ++task) {
        SchedulerAdmissionRecord record{};
        record.flags = fixture.storage.metadata[task].flags;
        record.active_mask = fixture.storage.metadata[task].active_mask;
        record.logical_block_num = 2;
        uint64_t words[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
        uint32_t members = 0;
        ASSERT_TRUE(scheduler_admission_expected_members(
            fixture.base(), fixture.locals[0], &admission, &record, words, &members
        ));
        EXPECT_EQ(words[0], expected[task]);
        EXPECT_EQ(words[1], 0u);
        EXPECT_EQ(members, static_cast<uint32_t>(__builtin_popcountll(expected[task])));
        EXPECT_EQ(admission.topology_worker_count, 6u);
        EXPECT_EQ(admission.topology_cube_words[0], 0x9u);
        EXPECT_EQ(admission.topology_vector0_words[0], 0x12u);
    }
    ASSERT_TRUE(scheduler_init_spmd_task_records(fixture.base(), fixture.storage.layout, 18));
    // A new run may publish a different compact-worker topology.
    std::swap(fixture.storage.contexts[0].core_type, fixture.storage.contexts[1].core_type);
    for (uint32_t worker : {0u, 1u, 2u}) {
        fixture.storage.contexts[worker].cluster_worker_ids[0] = 1;
        fixture.storage.contexts[worker].cluster_worker_ids[1] = 0;
        fixture.storage.contexts[worker].cluster_worker_ids[2] = 2;
    }
    ASSERT_TRUE(scheduler_initialize_admission(fixture.base(), fixture.locals[0], &admission));
    EXPECT_EQ(admission.epoch, 18u);
    EXPECT_EQ(admission.topology_worker_count, 0u);
    EXPECT_EQ(admission.topology_cube_words[0], 0u);
    EXPECT_EQ(admission.topology_vector0_words[0], 0u);
    SchedulerAdmissionRecord record{};
    record.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD;
    record.active_mask = LaneMask::aic();
    record.logical_block_num = 2;
    uint64_t words[(SCHEDULER_WORKER_CAPACITY + 63) / 64]{};
    uint32_t members = 0;
    ASSERT_TRUE(
        scheduler_admission_expected_members(fixture.base(), fixture.locals[0], &admission, &record, words, &members)
    );
    EXPECT_EQ(words[0], 0xau);
    EXPECT_EQ(members, 2u);
    EXPECT_EQ(admission.topology_vector0_words[0], 0x11u);
}

TEST(HbgAdmission, InvalidClusterLaneMappingsCannotFreezeAResourceSubset) {
    for (uint32_t corruption = 0; corruption < 5; ++corruption) {
        SCOPED_TRACE(corruption);
        AdmissionFixture fixture({{2, kAicAiv0}});
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        auto &peers = fixture.storage.contexts[0].cluster_worker_ids;
        if (corruption == 0) peers[0] = 3;
        else if (corruption == 1) peers[1] = 6;
        else if (corruption == 2) peers[1] = 0;
        else if (corruption == 3) peers[2] = peers[1];
        else peers[1] = 4;
        EXPECT_FALSE(fixture.prepare(0));
        EXPECT_EQ(fixture.record(0)->member_count, 0u);
        EXPECT_EQ(fixture.admissions[0].topology_worker_count, 0u);
        EXPECT_EQ(fixture.admissions[0].topology_vector0_words[0], 0u);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control()->commit_tail, UINT64_C(0)), 0u);
    }
}

TEST(HbgAdmission, CachedTopologyStillRejectsWrongMembersInALaterPublishedRecord) {
    AdmissionFixture fixture({{2, LaneMask::aic()}, {2, LaneMask::aic()}});
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.refresh(1));
    ASSERT_EQ(fixture.admissions[1].scan_cursor, 1u);
    ASSERT_EQ(fixture.admissions[1].topology_worker_count, 6u);
    ASSERT_TRUE(fixture.publish(1));
    // Same population, wrong types: comparing only member_count would miss it.
    fixture.members(1)[0] = 0x12;
    EXPECT_FALSE(fixture.refresh(1));
    EXPECT_EQ(fixture.admissions[1].scan_cursor, 1u);
    EXPECT_EQ(fixture.admissions[1].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_NE(fixture.run()->scheduler_error, 0u);
}

TEST(HbgAdmission, InitializationRejectsInvalidLayoutWithoutChangingOutput) {
    for (uint32_t corruption = 0; corruption < 10; ++corruption) {
        SCOPED_TRACE(corruption);
        AdmissionFixture fixture({{2, LaneMask::aic()}});
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
    AdmissionFixture fixture({{2, LaneMask::aic()}});
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
    AdmissionFixture fixture(
        std::vector<AdmissionShape>(task_count, AdmissionShape{2, LaneMask::aic()}), thread_count * 3, 71
    );
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
            scheduler_gm_query(fixture.task_control(task)->admission_state),
            static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
        );
    }
    EXPECT_TRUE(std::all_of(seen.begin(), seen.end(), [](bool value) {
        return value;
    }));
    for (uint64_t thread = 0; thread < thread_count; ++thread)
        EXPECT_EQ(fixture.admissions[thread * 3].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(HbgAdmission, DiscoveryProvenCompactTopologyMatchesTheGeneralScanForEveryMask) {
    for (uint64_t clusters : {1u, 2u, 22u, 32u, 36u}) {
        const uint64_t workers = clusters * 3;
        std::vector<uint64_t> compact(workers);
        for (uint64_t cluster = 0; cluster < clusters; ++cluster) {
            compact[3 * cluster] = cluster;
            compact[3 * cluster + 1] = clusters + 2 * cluster;
            compact[3 * cluster + 2] = clusters + 2 * cluster + 1;
        }
        for (const LaneMask mask : kAllMasks) {
            for (const bool sync : {false, true}) {
                if (sync && scheduler_block_claim_positions(mask, clusters) < 2) continue;
                SCOPED_TRACE(clusters);
                SCOPED_TRACE(lane_names(mask));
                SCOPED_TRACE(sync);
                const AdmissionShape shape{2, mask, static_cast<uint8_t>(sync ? SCHEDULER_TASK_SYNC_START : 0)};
                AdmissionFixture general({shape}, workers, 17, compact);
                AdmissionFixture proven({shape}, workers, 17, compact, false, SchedulerTopologyLayout::COMPACT_LAUNCH);
                for (uint64_t worker = 0; worker < workers; ++worker) {
                    EXPECT_EQ(proven.admissions[worker].topology_worker_count, workers);
                    EXPECT_EQ(general.admissions[worker].topology_worker_count, 0u);
                }
                ASSERT_TRUE(general.publish(0));
                ASSERT_TRUE(proven.publish(0));
                EXPECT_EQ(proven.record(0)->member_count, general.record(0)->member_count);
                for (uint64_t word = 0; word < (workers + 63) / 64; ++word)
                    EXPECT_EQ(proven.members(0)[word], general.members(0)[word]);
                for (uint64_t worker = 0; worker < workers; ++worker) {
                    ASSERT_TRUE(general.refresh(worker));
                    ASSERT_TRUE(proven.refresh(worker));
                    EXPECT_EQ(proven.admissions[worker].current_task_ref, general.admissions[worker].current_task_ref);
                    EXPECT_EQ(proven.admissions[worker].topology_worker_count, workers);
                    EXPECT_EQ(proven.admissions[worker].scan_cursor, general.admissions[worker].scan_cursor);
                }
            }
        }
    }
}

TEST(HbgAdmission, UnknownTopologyProofAndInvalidLocalBindingCannotInitializeAdmission) {
    for (uint32_t failure = 0; failure < 5; ++failure) {
        AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, 17, {0, 2, 3, 1, 4, 5});
        auto *summary =
            scheduler_state_at<SchedulerGraphSummary>(fixture.base(), fixture.storage.layout.graph_summary_offset);
        summary->topology_layout = SchedulerTopologyLayout::COMPACT_LAUNCH;
        if (failure == 0) summary->topology_layout = static_cast<SchedulerTopologyLayout>(2);
        if (failure == 1) summary->worker_count = 3;
        if (failure == 2) fixture.locals[0]->config.worker_ids[2] = 4;
        if (failure == 3) fixture.locals[0]->config.self_lane = PLATFORM_CORES_PER_BLOCKDIM;
        if (failure == 4) fixture.locals[0]->config.self_core_type = 1;
        SchedulerAdmissionLocalState rejected{};
        EXPECT_FALSE(scheduler_initialize_admission(fixture.base(), fixture.locals[0], &rejected));
        EXPECT_FALSE(rejected.initialized);
        EXPECT_EQ(rejected.topology_worker_count, 0u);
        EXPECT_EQ(fixture.control()->commit_tail, 0u);
        EXPECT_EQ(fixture.record(0)->run_epoch, 0u);
        EXPECT_EQ(fixture.record(0)->member_count, 0u);
        EXPECT_EQ(fixture.members(0)[0], 0u);
        EXPECT_EQ(fixture.admissions[0].topology_worker_count, 0u);
    }
}

TEST(HbgAdmissionProtocol, SuccessfulPublicationRecordsActualPublisherOrderAndEpoch) {
    for (uint64_t epoch : {UINT64_C(0), UINT64_MAX}) {
        AdmissionFixture fixture({{2, LaneMask::aic()}, {2, LaneMask::aiv0()}}, 6, epoch, {}, true);
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

TEST(HbgAdmissionProtocol, ExplicitNoRecordPathPreservesAdmissionAndEntryErrorGate) {
    for (const bool prepare_first : {false, true}) {
        AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, 17, {}, true);
        auto *local = fixture.locals[0];
        local->admission = &fixture.admissions[0];
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, 0, &ready));
        ASSERT_EQ(fixture.claim(0, ready), SchedulerAdmissionResult::PROGRESS);
        if (prepare_first) ASSERT_TRUE(fixture.prepare(0));
        ASSERT_EQ(
            scheduler_ordered_publish_pending<false>(fixture.graph.graph(), fixture.base(), local, fixture.run()),
            SchedulerAdmissionResult::PROGRESS
        );
        EXPECT_EQ(fixture.control()->commit_tail, 1u);
        EXPECT_EQ(fixture.record(0)->run_epoch, 17u);
        EXPECT_EQ(fixture.record(0)->member_count, 2u);
        EXPECT_EQ(fixture.task_control(0)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
        ASSERT_TRUE(
            scheduler_refresh_local_obligation<false>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), local->admission
            )
        );
        EXPECT_EQ(local->admission->current_task_ref, 0);
        ASSERT_EQ(
            scheduler_claim_ordered_block<false>(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0),
            SchedulerBlockClaimResult::CLAIMED
        );
        local->slots[0].state = SchedulerDispatchSlotState::READY;
        bool permit = false;
        ASSERT_TRUE(
            scheduler_ordered_can_enter_slot<false>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &permit
            )
        );
        ASSERT_TRUE(permit);
        const auto *task = scheduler_state_at<SchedulerProtocolTaskTrace>(
            fixture.base(), fixture.storage.layout.protocol.tasks_offset
        );
        const auto *member = scheduler_state_at<SchedulerProtocolMemberTrace>(
            fixture.base(), fixture.storage.layout.protocol.members_offset
        );
        const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
            fixture.base(), fixture.storage.layout.protocol.workers_offset
        );
        EXPECT_EQ(task->admission.before.sequence, 0u);
        EXPECT_EQ(task->admission.after.sequence, 0u);
        EXPECT_EQ(member->observed.sequence, 0u);
        EXPECT_EQ(worker->next_sequence, 0u);
        EXPECT_EQ(worker->error_flags, 0u);
        scheduler_gm_store(fixture.run()->scheduler_error, UINT64_C(1));
        EXPECT_FALSE(
            scheduler_ordered_can_enter_slot<false>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &permit
            )
        );
        EXPECT_FALSE(permit);
        EXPECT_EQ(fixture.task_control(0)->next_block, 1u);
    }
}

TEST(HbgAdmissionProtocol, CommitRetryDoesNotInventASequenceOrPublishAnInterval) {
    AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, 17, {}, true);
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
    AdmissionFixture fixture({{2, LaneMask::aic()}, {2, LaneMask::aiv0()}}, 6, 17, {}, true);
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
    AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, 17, {}, true);
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
    AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, 0, {}, true);
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
    AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, UINT64_MAX, {}, true);
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
        AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, 17, {}, true);
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
    AdmissionFixture fixture({{2, LaneMask::aic()}});
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
    AdmissionFixture fixture({{2, LaneMask::aiv0()}}, 66, 0, {}, true);
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

TEST(HbgAdmissionProtocol, DeferredFinishUsesPublishedOrderWithoutLocalCohort) {
    AdmissionFixture fixture({{2, LaneMask::aiv0()}}, 6, 17, {}, true);
    ASSERT_TRUE(fixture.publish(0, 1));
    ASSERT_EQ(fixture.locals[2]->admission, nullptr);
    ASSERT_EQ(fixture.admissions[2].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    scheduler_protocol_task_done(fixture.base(), fixture.locals[2], 0, true, true);
    scheduler_protocol_task_done(fixture.base(), fixture.locals[2], 0, false, true);
    const auto *tasks =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(tasks[0].completion.worker_index, 2u);
    EXPECT_EQ(tasks[0].completion.order_key, tasks[0].admission.order_key);
    EXPECT_EQ(tasks[0].completion.run_epoch, 17u);
    EXPECT_EQ(tasks[0].completion.flags, tasks[0].admission.flags | SCHEDULER_PROTOCOL_DEFERRED_FINISH);
    EXPECT_GT(tasks[0].completion.after.sequence, tasks[0].completion.before.sequence);
    EXPECT_EQ(workers[2].error_flags, 0u);
}

TEST(HbgAdmissionProtocol, DeferredFinishRejectsForeignAdmissionEpoch) {
    AdmissionFixture fixture({{2, LaneMask::aiv0()}}, 6, 17, {}, true);
    ASSERT_TRUE(fixture.publish(0, 1));
    auto *tasks =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    ++tasks[0].admission.run_epoch;
    scheduler_protocol_task_done(fixture.base(), fixture.locals[2], 0, true, true);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(tasks[0].completion.before.sequence, 0u);
    EXPECT_EQ(workers[2].error_flags, SCHEDULER_PROTOCOL_INVALID_IDENTITY);
}

TEST_P(HbgProtocolLayoutCorruption, BootstrapRejectsBadLayoutAndRetainedWorkerState) {
    AdmissionFixture fixture({{2, LaneMask::aic()}}, 6, 17, {}, true);
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

std::vector<AdmissionShape> claim_seal_shapes() {
    std::vector<AdmissionShape> shapes;
    for (LaneMask mask : {LaneMask::aic(), LaneMask::aiv0(), kAicAiv0, kAicAiv1, LaneMask::both_aiv(), LaneMask::all()})
        for (uint16_t blocks : {1, 2, 3, 4, 17, 65}) {
            if (blocks == 1 && mask.count() == 1) continue;
            shapes.push_back({blocks, mask, 0});
            if (blocks > 1 && blocks <= 4) shapes.push_back({blocks, mask, SCHEDULER_TASK_SYNC_START});
        }
    return shapes;
}

class HbgClaimSeal : public ::testing::TestWithParam<AdmissionShape> {};

TEST_P(HbgClaimSeal, OnlyFinalValidClaimPublishesAndDoesNotCompleteTheTask) {
    const auto shape = GetParam();
    AdmissionFixture fixture({shape}, 12);
    ASSERT_EQ(fixture.task_control(0)->claim_seal, 0u);
    ASSERT_TRUE(fixture.publish(0));
    std::vector<uint32_t> claimants;
    for (uint32_t worker = 0; worker < 12; ++worker) {
        auto *local = fixture.locals[worker];
        local->admission = &fixture.admissions[worker];
        ASSERT_TRUE(fixture.refresh(worker));
        if (fixture.admissions[worker].current_task_ref != 0) continue;
        if (shape.mask == LaneMask::aiv0() || worker % 3 == 0) claimants.push_back(worker);
    }
    ASSERT_FALSE(claimants.empty());
    std::vector<bool> seen(shape.blocks, false);
    for (uint32_t index = 0; index < shape.blocks; ++index) {
        const uint32_t worker = claimants[index % claimants.size()];
        uint32_t block = UINT32_MAX;
        ASSERT_EQ(
            scheduler_claim_ordered_cursor(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
            ),
            SchedulerBlockClaimResult::CLAIMED
        );
        ASSERT_LT(block, shape.blocks);
        EXPECT_FALSE(seen[block]);
        seen[block] = true;
        EXPECT_EQ(fixture.task_control(0)->claim_seal, index + 1 == shape.blocks ? shape.blocks : 0u);
        uint64_t cursor = UINT64_MAX;
        bool sealed = false;
        ASSERT_TRUE(scheduler_read_claim_cursor(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &cursor, &sealed
        ));
        EXPECT_EQ(cursor, index + 1u);
        EXPECT_EQ(sealed, index + 1 == shape.blocks);
        EXPECT_EQ(fixture.task_control(0)->completed_blocks, 0u);
        EXPECT_NE(fixture.task_control(0)->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    }
    for (const uint32_t worker : claimants) {
        uint32_t block = UINT32_MAX;
        const auto result = scheduler_claim_ordered_cursor(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &block
        );
        EXPECT_TRUE(result == SchedulerBlockClaimResult::EXHAUSTED || result == SchedulerBlockClaimResult::RETRY);
        EXPECT_EQ(block, UINT32_MAX);
        EXPECT_EQ(fixture.task_control(0)->claim_seal, shape.blocks);
    }
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST_P(HbgClaimSeal, MissingSignalUsesStrongCursorAndEarlyOrMalformedSignalIsRejected) {
    const auto shape = GetParam();
    for (uint32_t scenario = 0; scenario < 9; ++scenario) {
        SCOPED_TRACE(scenario);
        AdmissionFixture fixture({shape}, 12);
        ASSERT_TRUE(fixture.publish(0));
        const uint64_t worker = fixture.publisher(0);
        fixture.locals[worker]->admission = &fixture.admissions[worker];
        ASSERT_TRUE(fixture.refresh(worker));
        auto *control = fixture.task_control(0);
        control->next_block = shape.blocks;
        if (scenario == 1) control->next_block = shape.blocks - 1;
        if (scenario >= 2) control->claim_seal = shape.blocks;
        if (scenario == 2) control->next_block = shape.blocks - 1;
        if (scenario == 3) control->claim_seal = shape.blocks + 1;
        if (scenario == 4) control->next_block = shape.blocks + 1;
        if (scenario == 5) ++fixture.admissions[worker].epoch;
        if (scenario == 6) fixture.run()->scheduler_error = static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT);
        if (scenario == 7) control->claim_seal = (UINT64_C(1) << 32) + shape.blocks;
        if (scenario == 8) control->claim_seal = UINT64_MAX;
        uint64_t cursor = UINT64_MAX;
        bool sealed = true;
        const bool valid = scheduler_read_claim_cursor(
            fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &cursor, &sealed
        );
        EXPECT_EQ(valid, scenario < 2);
        EXPECT_FALSE(sealed);
        if (scenario < 2) EXPECT_EQ(cursor, shape.blocks - scenario);
        else EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
        EXPECT_EQ(control->completed_blocks, 0u);
        EXPECT_NE(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    }
}

TEST_P(HbgClaimSeal, AdmissionRejectsPreexistingSealAndRunInitializationClearsIt) {
    const auto shape = GetParam();
    for (bool after_prepare : {false, true}) {
        AdmissionFixture fixture({shape}, 12);
        const uint64_t worker = fixture.publisher(0);
        SchedulerReadyClaim ready{};
        ASSERT_TRUE(fixture.ready_pop(0, worker, &ready));
        ASSERT_EQ(fixture.claim(worker, ready), SchedulerAdmissionResult::PROGRESS);
        if (after_prepare) ASSERT_TRUE(fixture.prepare(worker));
        fixture.task_control(0)->claim_seal = shape.blocks;
        if (after_prepare) EXPECT_EQ(fixture.commit(worker), SchedulerAdmissionResult::ERROR);
        else EXPECT_FALSE(fixture.prepare(worker));
        EXPECT_EQ(fixture.control()->commit_tail, 0u);
        EXPECT_EQ(fixture.task_control(0)->next_block, 0u);
        ASSERT_TRUE(scheduler_init_data_from_layout(
            fixture.base(), fixture.storage.layout, reinterpret_cast<uint64_t>(fixture.base())
        ));
        EXPECT_EQ(fixture.task_control(0)->claim_seal, 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(MasksSizesAndSyncModes, HbgClaimSeal, ::testing::ValuesIn(claim_seal_shapes()));

}  // namespace
