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

#include "scheduler/scheduler_scalar_policy.h"

namespace {
template <>
struct SchedulerScalarCompilation<void> {
    static constexpr bool enabled = true;
};
}  // namespace

#include "inner_kernel.h"

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <functional>
#include <map>
#include <set>
#include <string>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

#include "hbg_scheduler_test_support.h"

namespace {

thread_local std::function<void(volatile uint64_t *)> gm_query_observer;

template <typename T>
T observe_gm_query(volatile T &word, int order = __ATOMIC_ACQUIRE) {
    if constexpr (std::is_same_v<T, uint64_t>)
        if (gm_query_observer) gm_query_observer(&word);
    return scheduler_gm_query(word, order);
}

class ScopedGmQueryObserver {
public:
    explicit ScopedGmQueryObserver(std::function<void(volatile uint64_t *)> observer) {
        gm_query_observer = std::move(observer);
    }
    ~ScopedGmQueryObserver() { gm_query_observer = {}; }
};

thread_local std::function<void(SchedulerLocalState *)> gm_prepare_observer;
enum class GmAtomicOperation : uint8_t { FETCH_ADD, COMPARE_EXCHANGE, STORE };
thread_local std::function<void(volatile uint64_t *, uint64_t, GmAtomicOperation)> gm_atomic_observer;

uint64_t observe_gm_atomic_add(volatile uint64_t &word, uint64_t delta, int order = __ATOMIC_ACQ_REL) {
    if (gm_atomic_observer) gm_atomic_observer(&word, delta, GmAtomicOperation::FETCH_ADD);
    return scheduler_gm_fetch_add(word, delta, order);
}

uint64_t observe_gm_atomic_cas(
    volatile uint64_t &word, uint64_t expected, uint64_t desired, int success_order = __ATOMIC_ACQ_REL,
    int failure_order = __ATOMIC_ACQUIRE
) {
    if (gm_atomic_observer) gm_atomic_observer(&word, desired, GmAtomicOperation::COMPARE_EXCHANGE);
    return scheduler_gm_compare_exchange(word, expected, desired, success_order, failure_order);
}

void observe_gm_atomic_store(volatile uint64_t &word, uint64_t desired, int order = __ATOMIC_RELEASE) {
    if (gm_atomic_observer) gm_atomic_observer(&word, desired, GmAtomicOperation::STORE);
    scheduler_gm_store(word, desired, order);
}

class ScopedGmAtomicObserver {
public:
    explicit ScopedGmAtomicObserver(std::function<void(volatile uint64_t *, uint64_t, GmAtomicOperation)> observer) {
        gm_atomic_observer = std::move(observer);
    }
    ~ScopedGmAtomicObserver() { gm_atomic_observer = {}; }
};

bool observe_gm_prepare(
    const SchedulerGraphAccess &graph, void *base, SchedulerLocalState *local, SchedulerRunControl *run,
    uint32_t slot_index, const SchedulerReadyClaim &ready, const SchedulerTaskMetadata &metadata, uint64_t profiling,
    uint8_t subtask, uint64_t *prepare_start
) {
    const auto result = scheduler_prepare_dispatch_slot(
        graph, base, local, run, slot_index, ready, metadata, profiling, subtask, prepare_start
    );
    if (result && gm_prepare_observer) gm_prepare_observer(local);
    return result;
}

class ScopedGmPrepareObserver {
public:
    explicit ScopedGmPrepareObserver(std::function<void(SchedulerLocalState *)> observer) {
        gm_prepare_observer = std::move(observer);
    }
    ~ScopedGmPrepareObserver() { gm_prepare_observer = {}; }
};

}  // namespace

#define scheduler_prepare_dispatch_slot observe_gm_prepare
#define scheduler_gm_query observe_gm_query
#define scheduler_gm_fetch_add observe_gm_atomic_add
#define scheduler_gm_compare_exchange observe_gm_atomic_cas
#define scheduler_gm_store observe_gm_atomic_store
#include "scheduler/scheduler_gm_mix.h"
#undef scheduler_gm_store
#undef scheduler_gm_compare_exchange
#undef scheduler_gm_fetch_add
#include "scheduler/scheduler_cluster_mix.h"
#undef scheduler_prepare_dispatch_slot
#include "scheduler/scheduler_dispatch.h"
#undef scheduler_gm_query
#include "scheduler/scheduler_graph_control.h"
#include "scheduler/scheduler_protocol_execute.h"
#include "scheduler/scheduler_args_dump.h"

#include "hbg_gm_mix_test_support.h"

namespace {

using namespace scheduler_test::gm_mix;

TEST(HbgGmMix, OrderedObservationRetainsArrivalsAndReleasesWhileAnotherLaneWaits) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    ASSERT_TRUE(fixture.arrive(0, &progress));
    ASSERT_TRUE(fixture.observe(1, &progress));
    ASSERT_TRUE(fixture.arrive(1, &progress));
    auto *mailbox = fixture.mailbox(0);
    std::map<volatile uint64_t *, uint32_t> reads;
    ScopedGmQueryObserver observe([&](volatile uint64_t *word) {
        ++reads[word];
    });
    auto poll = [&]() {
        return scheduler_observe_ordered_progress(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()
        );
    };
    for (uint32_t visit = 0; visit < 3; ++visit)
        EXPECT_EQ(poll(), SchedulerOrderedPollResult::WAIT);
    EXPECT_EQ(fixture.admissions[0].gm_mix.arrivals_confirmed, 3u);
    EXPECT_EQ(reads[&mailbox->lanes[1].arrived_seq], 1u);
    ASSERT_TRUE(fixture.observe(2, &progress));
    ASSERT_TRUE(fixture.arrive(2, &progress));
    EXPECT_EQ(poll(), SchedulerOrderedPollResult::READY);
    const auto arrival_reads = reads[&mailbox->lanes[2].arrived_seq];
    ASSERT_TRUE(fixture.rendezvous(0));
    EXPECT_EQ(reads[&mailbox->lanes[1].arrived_seq], 1u);
    EXPECT_EQ(reads[&mailbox->lanes[2].arrived_seq], arrival_reads);
    for (uint32_t lane : {0u, 1u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    for (uint32_t visit = 0; visit < 3; ++visit)
        EXPECT_EQ(poll(), SchedulerOrderedPollResult::WAIT);
    EXPECT_EQ(fixture.admissions[0].gm_mix.releases_confirmed, 3u);
    EXPECT_EQ(reads[&mailbox->lanes[1].released_seq], 1u);
    GmExecution last{};
    ASSERT_TRUE(fixture.begin(2, &last));
    ASSERT_TRUE(fixture.complete(last));
    EXPECT_EQ(poll(), SchedulerOrderedPollResult::READY);
    const auto release_reads = reads[&mailbox->lanes[2].released_seq];
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(reads[&mailbox->lanes[1].released_seq], 1u);
    EXPECT_EQ(reads[&mailbox->lanes[2].released_seq], release_reads);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 3u);
}

TEST(HbgGmMix, AssignmentSnapshotRequiresMatchingLocalRunTaskOrderGroupAndPhase) {
    for (uint32_t changed = 0; changed < 6; ++changed) {
        SCOPED_TRACE(changed);
        GmMixFixture fixture({{2, LaneMask::all(), true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        auto *local = fixture.locals[1];
        uint32_t reads = 0;
        ScopedGmQueryObserver observe([&](volatile uint64_t *word) {
            reads += word == &fixture.mailbox(0)->lanes[1].assigned_seq;
        });
        SchedulerGmMixAssignmentObservation snapshot{};
        ASSERT_EQ(
            scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run(), &snapshot),
            SchedulerOrderedPollResult::READY
        );
        ASSERT_EQ(reads, 1u);
        if (changed == 1) ++snapshot.epoch;
        if (changed == 2) ++snapshot.task_ref;
        if (changed == 3) ++snapshot.order_key;
        if (changed == 4) ++snapshot.group_seq;
        if (changed == 5) snapshot.phase = SchedulerGmMixPhase::PUBLISHED;
        bool progress = false;
        ASSERT_TRUE(
            scheduler_gm_mix_observe(fixture.graph.graph(), fixture.base(), local, fixture.run(), &progress, &snapshot)
        );
        EXPECT_TRUE(progress);
        EXPECT_EQ(reads, changed == 0 ? 1u : 2u);
        EXPECT_EQ(snapshot.assigned_seq, 0u);
        EXPECT_EQ(local->admission->gm_mix.group_seq, 1u);
    }
}

TEST(HbgGmMix, AssignmentAndAckObservationsDoNotUseGmAtomics) {
    for (const auto mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask, true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        uint32_t assigned_reads = 0;
        uint32_t arrival_reads = 0;
        uint32_t release_reads = 0;
        uint32_t self_ack_stores = 0;
        ScopedGmAtomicObserver observer([&](volatile uint64_t *word, uint64_t delta, GmAtomicOperation operation) {
            if (operation == GmAtomicOperation::STORE && (word == &fixture.mailbox(0)->lanes[0].arrived_seq ||
                                                          word == &fixture.mailbox(0)->lanes[0].released_seq))
                ++self_ack_stores;
            if (delta != 0 || operation != GmAtomicOperation::FETCH_ADD) return;
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (word == &fixture.mailbox(0)->lanes[lane].assigned_seq) ++assigned_reads;
                if (word == &fixture.mailbox(0)->lanes[lane].arrived_seq) ++arrival_reads;
                if (word == &fixture.mailbox(0)->lanes[lane].released_seq) ++release_reads;
            }
        });
        ASSERT_TRUE(fixture.rendezvous(0));
        EXPECT_EQ(assigned_reads, 0u);
        EXPECT_EQ(arrival_reads, 0u);
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        EXPECT_EQ(release_reads, 0u);
        EXPECT_EQ(self_ack_stores, 0u);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgGmMix, CoordinatorRetiresThroughItsOwnMonotoneFrontierWithoutGmAtomics) {
    for (const auto mask : kMixMasks) {
        for (const uint32_t initial : {0u, 7u}) {
            SCOPED_TRACE(lane_names(mask));
            SCOPED_TRACE(initial);
            GmMixFixture fixture({{2, mask, true}});
            ASSERT_TRUE(fixture.prepare_calls());
            ASSERT_TRUE(fixture.admit(0));
            ASSERT_TRUE(fixture.refresh_all());
            ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
            ASSERT_TRUE(fixture.publish(0));
            ASSERT_TRUE(fixture.rendezvous(0));
            ASSERT_TRUE(fixture.finish_lanes(0));
            ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
            auto *signal = fixture.signal(0);
            scheduler_cluster_store_field(&signal->sync_frontier[0], initial);
            uint32_t operations = 0;
            ScopedGmAtomicObserver observer([&](volatile uint64_t *, uint64_t, GmAtomicOperation) {
                ++operations;
            });
            bool closed = false;
            EXPECT_EQ(fixture.close(0, &closed), initial == 0);
            EXPECT_EQ(closed, initial == 0);
            EXPECT_EQ(scheduler_cluster_load_field(&signal->sync_frontier[0]), initial == 0 ? 1u : initial);
            if (initial == 0) {
                EXPECT_EQ(operations, 0u);
                EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
            } else {
                EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_CLOSE_INVALID));
            }
        }
    }
}

TEST(HbgClusterDiscovery, MixCannotBypassAnUnissuedOrdinaryOrSyncHead) {
    GmMixFixture fixture({{1, LaneMask::aic()}, {2, LaneMask::all(), true}, {1, LaneMask::all()}});
    ASSERT_TRUE(fixture.prepare_calls());
    bool progress = false;
    ASSERT_TRUE(scheduler_cluster_mix_discover_task(
        fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run(), 2, &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
    ASSERT_TRUE(fixture.claim_ordinary(0, 0));
    ASSERT_TRUE(scheduler_cluster_mix_discover_next(
        fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run(), &progress
    ));
    EXPECT_EQ(fixture.locals[3]->cluster_pending.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
    EXPECT_EQ(fixture.signal(1)->prepare_generation, 0u);
    EXPECT_EQ(fixture.signal(1)->armed_generation, 0u);
}

TEST(HbgClusterDiscovery, IssuedGroupsAdvanceOneSharedHeadAcrossClusters) {
    GmMixFixture fixture({{1, LaneMask::all()}, {1, LaneMask::all()}, {1, LaneMask::all()}});
    ASSERT_TRUE(fixture.prepare_calls());
    bool progress = false;
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_TRUE(scheduler_cluster_mix_discover_next(
            fixture.graph.graph(), fixture.base(), fixture.locals[cluster * 3], fixture.run(), &progress
        ));
        ASSERT_TRUE(progress);
        EXPECT_EQ(fixture.locals[cluster * 3]->cluster_pending.task_ref, cluster);
    }
    const auto held = fixture.locals[0]->cluster_pending.anchor;
    ASSERT_TRUE(scheduler_cluster_mix_discover_next(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.locals[0]->cluster_pending.anchor.claimed_state, held.claimed_state);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
}

class HbgClusterEarlyPreparation : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgClusterEarlyPreparation, ProtocolClaimFollowsInterleavedProducerCompletion) {
    const auto mask = GetParam();
    GmMixFixture fixture({{1, LaneMask::aic()}, {1, mask}}, true, 1);
    fixture.graph.executable(1, mask.has_aic() ? 0 : 1, {0});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.claim_ordinary(0, 0));
    ASSERT_TRUE(fixture.progress_ordinary(0));
    GmExecution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_EQ(producer.task, 0);
    bool progress = false;
    ASSERT_TRUE(scheduler_cluster_mix_discover_task(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1, &progress
    ));
    ASSERT_EQ(fixture.locals[0]->cluster_pending.task_ref, 1);
    ASSERT_TRUE(fixture.complete(producer));
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(1, 0, &armed));
    ASSERT_TRUE(armed);
    const auto *tasks =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    const auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    const auto &completion = tasks[0].completion.after;
    const auto &owner = blocks[fixture.storage.metadata[1].trace_index_base].owner;
    ASSERT_GT(completion.sequence, 0u);
    EXPECT_GT(owner.staged.sequence, completion.sequence);
    EXPECT_GE(owner.staged.cycles, completion.cycles);
    EXPECT_GT(owner.claim_before.sequence, owner.staged.sequence);
    EXPECT_GE(owner.claim_before.cycles, completion.cycles);
    EXPECT_GE(owner.claim_before.cycles, owner.staged.cycles);
    EXPECT_GT(owner.claim_after.sequence, owner.claim_before.sequence);
    EXPECT_GE(owner.claim_after.cycles, owner.claim_before.cycles);
    EXPECT_GT(owner.publish_before.sequence, owner.claim_after.sequence);
    EXPECT_GE(owner.publish_before.cycles, owner.claim_after.cycles);
}

TEST_P(HbgClusterEarlyPreparation, FutureGroupKeepsItsClusterUntilProducerCompletion) {
    const auto mask = GetParam();
    GmMixFixture fixture({{1, LaneMask::aic()}, {2, mask}, {1, mask}});
    fixture.graph.executable(1, mask.has_aic() ? 0 : 1, {0});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.claim_ordinary(0, 0));
    ASSERT_TRUE(fixture.progress_ordinary(0));
    GmExecution producer{};
    ASSERT_TRUE(fixture.begin(0, &producer));
    ASSERT_EQ(producer.task, 0);
    auto *coordinator = fixture.locals[3];
    bool progress = false;
    ASSERT_TRUE(scheduler_cluster_mix_discover_next(
        fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), &progress
    ));
    ASSERT_TRUE(progress);
    ASSERT_EQ(coordinator->cluster_pending.task_ref, 1);
    const auto held = coordinator->cluster_pending.anchor;
    const auto prepared = fixture.signal(1)->prepare_generation;
    for (uint32_t worker : {3u, 4u, 5u}) {
        ASSERT_TRUE(scheduler_prepare_cluster_lane(fixture.base(), fixture.locals[worker]));
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
        for (const auto &slot : fixture.locals[worker]->slots)
            EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    }
    ASSERT_TRUE(scheduler_cluster_mix_progress_pending(
        fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), &progress
    ));
    EXPECT_FALSE(progress);
    ASSERT_TRUE(scheduler_cluster_mix_discover_next(
        fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(coordinator->cluster_pending.task_ref, 1);
    EXPECT_EQ(coordinator->cluster_pending.anchor.execution_index, held.execution_index);
    EXPECT_EQ(coordinator->cluster_pending.anchor.claimed_state, held.claimed_state);
    EXPECT_EQ(fixture.signal(1)->prepare_generation, prepared);
    EXPECT_EQ(fixture.signal(1)->armed_generation, 0u);
    EXPECT_EQ(fixture.signal(1)->staged_mask, 0u);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);

    const uint64_t issue_head = fixture.locals[0]->issue_cursor;
    ASSERT_EQ(issue_head, 1u);
    ASSERT_TRUE(scheduler_cluster_mix_discover_task(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 2, &progress
    ));
    EXPECT_TRUE(progress);
    EXPECT_EQ(fixture.locals[0]->issue_cursor, issue_head);
    EXPECT_EQ(fixture.locals[0]->cluster_pending.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
    ASSERT_TRUE(fixture.complete(producer));
    std::set<std::tuple<int64_t, uint32_t, uint8_t>> executions;
    for (uint32_t pass = 0; pass < 128 && !(fixture.task_done(1) && fixture.task_done(2)); ++pass)
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_TRUE(execution.task == 1 || execution.task == 2);
            if (execution.task == 1 && execution.block == 0) EXPECT_EQ(worker / 3, 1u);
            EXPECT_TRUE(executions.emplace(execution.task, execution.block, execution.subtask).second);
            ASSERT_TRUE(fixture.complete(execution));
        }
    EXPECT_EQ(executions.size(), 3 * mask.count());
    EXPECT_TRUE(fixture.task_done(1));
    EXPECT_TRUE(fixture.task_done(2));
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgClusterEarlyPreparation, ::testing::ValuesIn(kMixMasks));

TEST(HbgClusterEarlyPreparation, ResourceWaitRetainsReservationAndPreparation) {
    GmMixFixture fixture({{1, LaneMask::all()}, {1, LaneMask::all()}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *coordinator = fixture.locals[0];
    bool progress = false;
    ASSERT_TRUE(scheduler_cluster_mix_discover_task(
        fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), 0, &progress
    ));
    ASSERT_TRUE(progress);
    const auto held = coordinator->cluster_pending.anchor;
    const auto generation = fixture.signal(0)->prepare_generation;
    fixture.signal(0)->sync_hold[0] = 1;
    for (uint32_t observation = 0; observation < 8; ++observation) {
        ASSERT_TRUE(scheduler_cluster_mix_progress_pending(
            fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), &progress
        ));
        EXPECT_FALSE(progress);
        ASSERT_TRUE(scheduler_cluster_mix_discover_next(
            fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), &progress
        ));
        EXPECT_FALSE(progress);
        EXPECT_EQ(coordinator->cluster_pending.task_ref, 0);
        EXPECT_EQ(coordinator->cluster_pending.anchor.claimed_state, held.claimed_state);
        EXPECT_EQ(fixture.signal(0)->prepare_generation, generation);
        EXPECT_EQ(fixture.control(1)->next_block, 0u);
        EXPECT_EQ(fixture.signal(0)->armed_generation, 0u);
    }
    fixture.signal(0)->sync_hold[0] = 0;
    ASSERT_TRUE(scheduler_cluster_mix_progress_pending(
        fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), &progress
    ));
    EXPECT_TRUE(progress);
    EXPECT_TRUE(scheduler_cluster_mix_armed(fixture.signal(0)->armed_generation));
    EXPECT_EQ(fixture.signal(0)->task_ref, 0);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
}

TEST(HbgClusterEarlyPreparation, WaitingFutureGroupPreservesAnArmedCurrentGroup) {
    GmMixFixture fixture(
        {{1, LaneMask::aic()}, {1, LaneMask::aic()}, {2, LaneMask::all()}, {1, LaneMask::all()}, {1, LaneMask::all()}}
    );
    fixture.graph.executable(3, 0, {0});
    fixture.graph.executable(4, 0, {1});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.claim_ordinary(0, 3));
    ASSERT_TRUE(fixture.claim_ordinary(1, 3));
    bool peer_progress = false;
    for (uint32_t slice = 0; slice < 10 && scheduler_task_ref_is_null(fixture.locals[3]->cluster_pending.task_ref);
         ++slice) {
        ASSERT_TRUE(scheduler_cluster_mix_discover_task(
            fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run(), 2, &peer_progress
        ));
        ASSERT_TRUE(peer_progress);
        EXPECT_EQ(fixture.control(3)->next_block, 0u);
    }
    ASSERT_EQ(fixture.locals[3]->cluster_pending.task_ref, 2);
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(2, 0, &armed));
    ASSERT_TRUE(armed);
    const uint32_t generation = fixture.signal(0)->armed_generation;
    const uint32_t staged = fixture.signal(0)->staged_mask;
    auto *coordinator = fixture.locals[0];
    bool progress = false;
    for (uint32_t slice = 0; slice < 10 && scheduler_task_ref_is_null(coordinator->cluster_pending.task_ref); ++slice) {
        ASSERT_TRUE(scheduler_cluster_mix_discover_task(
            fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), 3, &progress
        ));
        ASSERT_TRUE(progress);
        EXPECT_EQ(fixture.control(4)->next_block, 0u);
    }
    ASSERT_EQ(coordinator->cluster_pending.task_ref, 3);
    const auto held = coordinator->cluster_pending.anchor;
    ASSERT_TRUE(scheduler_cluster_mix_discover_next(
        fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(coordinator->cluster_pending.task_ref, 3);
    EXPECT_EQ(coordinator->cluster_pending.anchor.claimed_state, held.claimed_state);
    EXPECT_EQ(fixture.control(2)->next_block, 2u);
    EXPECT_EQ(fixture.control(4)->next_block, 0u);

    fixture.control(1)->completion_state = scheduler_completion_completed_state(1);
    ASSERT_TRUE(scheduler_cluster_mix_discover_next(
        fixture.graph.graph(), fixture.base(), coordinator, fixture.run(), &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(coordinator->cluster_pending.task_ref, 3);
    EXPECT_EQ(coordinator->cluster_pending.anchor.claimed_state, held.claimed_state);
    EXPECT_EQ(fixture.control(4)->next_block, 0u);
    EXPECT_EQ(fixture.signal(0)->armed_generation, generation);
    EXPECT_EQ(fixture.signal(0)->staged_mask, staged);
    EXPECT_EQ(fixture.signal(0)->task_ref, 2);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(fixture.signal(0)->arrived[lane], 0u);
        EXPECT_EQ(fixture.signal(0)->finished[lane], 0u);
    }
}

class HbgClusterServicePriority : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgClusterServicePriority, FormalArrivalPrecedesNewOrdinaryOwnership) {
    const auto mask = GetParam();
    const uint32_t lane = mask.has_aiv0() ? 1 : 2;
    GmMixFixture fixture({{1, mask}, {2, LaneMask::aiv0()}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    ASSERT_TRUE(fixture.fill(lane));
    EXPECT_EQ(fixture.locals[lane]->cluster_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    EXPECT_EQ(fixture.locals[lane]->local_ready_mask, 0u);

    bool progress = false;
    for (uint32_t pass = 0; pass < 2; ++pass)
        for (uint32_t worker = 0; worker < 3; ++worker)
            ASSERT_TRUE(fixture.ordinary_service(worker, &progress));
    GmExecution member{};
    ASSERT_TRUE(fixture.begin(lane, &member));
    ASSERT_EQ(member.task, 0);
    ASSERT_TRUE(fixture.complete(member));
    ASSERT_TRUE(fixture.fill(lane));
    // This lane finished, although its partners still own the same formal group.
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(fixture.signal(0)->finished[lane], fixture.signal(0)->armed_generation);
}

TEST_P(HbgClusterServicePriority, PublishedGroupStillDrainsAnExistingReadySlot) {
    const auto mask = GetParam();
    const uint32_t lane = mask.has_aiv0() ? 1 : 2;
    GmMixFixture fixture({{1, mask}, {1, LaneMask::aiv0()}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.claim_ordinary(1, lane));
    ASSERT_TRUE(fixture.progress_ordinary(lane));
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    ASSERT_TRUE(fixture.fill(lane));
    ASSERT_TRUE(fixture.service_cooperative(lane));
    ASSERT_EQ(fixture.locals[lane]->cluster_mix.phase, SchedulerGmMixPhase::PUBLISHED);
    EXPECT_EQ(fixture.signal(0)->arrived[lane], 0u);
    GmExecution ordinary{};
    ASSERT_TRUE(fixture.begin(lane, &ordinary));
    ASSERT_EQ(ordinary.task, 1);
    ASSERT_TRUE(fixture.complete(ordinary));
    ASSERT_TRUE(fixture.fill(lane));
    EXPECT_EQ(fixture.locals[lane]->cluster_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.signal(0)->arrived[lane], fixture.signal(0)->armed_generation);
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgClusterServicePriority, ::testing::ValuesIn(kMixMasks));

TEST(HbgClusterEarlyPreparation, FreshAdmissionServicesAnArmedGroupBeforeResumingDeferredPreparation) {
    GmMixFixture fixture({{1, LaneMask::all()}, {2, LaneMask::aiv0(), true}, {1, LaneMask::all()}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    auto *local = fixture.locals[1];
    const uint32_t generation = fixture.signal(0)->prepare_generation;
    ASSERT_GT(generation, 0u);
    ASSERT_EQ(local->cluster_prepare_seen, 0u);

    // The peer can publish an ARM after this worker's earlier service pass.
    bool failed = false;
    ASSERT_TRUE(
        scheduler_fill_cooperative_slots(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed)
    );
    ASSERT_FALSE(failed);
    EXPECT_EQ(local->cluster_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(local->cluster_mix.task_ref, 0);
    EXPECT_EQ(local->admission->pending_task_ref, 1);
    EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
    EXPECT_EQ(local->cluster_prepare_seen, 0u);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);

    bool progress = false;
    for (uint32_t pass = 0; pass < 2; ++pass)
        for (uint32_t worker = 0; worker < 3; ++worker)
            ASSERT_TRUE(fixture.ordinary_service(worker, &progress));
    for (uint32_t worker = 0; worker < 3; ++worker) {
        GmExecution member{};
        ASSERT_TRUE(fixture.begin(worker, &member));
        ASSERT_EQ(member.task, 0);
        ASSERT_TRUE(fixture.complete(member));
    }
    EXPECT_FALSE(fixture.task_done(0));
    ASSERT_TRUE(fixture.ordinary_service(0, &progress));
    ASSERT_TRUE(fixture.task_done(0));

    ASSERT_TRUE(
        scheduler_fill_cooperative_slots(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed)
    );
    ASSERT_FALSE(failed);
    EXPECT_EQ(local->cluster_prepare_seen, generation);
    EXPECT_EQ(fixture.signal(0)->prepared[1], generation);
    EXPECT_EQ(local->admission->pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));

    std::set<std::tuple<int64_t, uint32_t, uint8_t>> executions;
    for (uint32_t pass = 0; pass < 128 && !(fixture.task_done(1) && fixture.task_done(2)); ++pass)
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_TRUE(execution.task == 1 || execution.task == 2);
            EXPECT_TRUE(executions.emplace(execution.task, execution.block, execution.subtask).second);
            ASSERT_TRUE(fixture.complete(execution));
        }
    EXPECT_EQ(executions.size(), 5u);
    EXPECT_TRUE(fixture.task_done(1));
    EXPECT_TRUE(fixture.task_done(2));
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgDirectDispatch, CapturesOneOwnedSlotWithoutPublishingItToTheLocalQueue) {
    GmMixFixture fixture({{2, LaneMask::aiv0()}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *local = fixture.locals[1];
    uint32_t direct = UINT32_MAX;
    bool failed = false;
    EXPECT_TRUE(
        scheduler_fill_own_slots(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, &direct)
    );
    ASSERT_FALSE(failed);
    ASSERT_LT(direct, SCHEDULER_PENDING_SLOT_COUNT);
    EXPECT_EQ(local->direct_slot, nullptr);
    EXPECT_EQ(local->local_ready_mask, 0u);
    EXPECT_EQ(local->ready_head, UINT8_MAX);
    EXPECT_EQ(local->slots[direct].state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(scheduler_prepared_execute_owner(local->slots[direct].prepared.claimed_state), 1u);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(local->slots[1 - direct].state, SchedulerDispatchSlotState::FREE);
}

TEST(HbgDirectDispatch, ExistingQueuedWorkWinsBeforeAnotherDirectCapture) {
    GmMixFixture fixture({{1, LaneMask::aiv0()}, {1, LaneMask::aiv0()}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *local = fixture.locals[1];
    ASSERT_TRUE(fixture.fill(1));
    const auto queued_mask = local->local_ready_mask;
    ASSERT_NE(queued_mask, 0u);
    const auto queued_head = local->ready_head;
    uint32_t direct = 0;
    bool failed = false;
    EXPECT_FALSE(
        scheduler_fill_own_slots(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, &direct)
    );
    EXPECT_FALSE(failed);
    EXPECT_EQ(direct, UINT32_MAX);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    EXPECT_EQ(local->direct_slot, nullptr);
    EXPECT_EQ(local->local_ready_mask, queued_mask);
    EXPECT_EQ(local->ready_head, queued_head);
}

TEST(HbgDirectDispatch, LaterCooperativeCommitDeniesEntryWithoutLosingTheCapturedCall) {
    GmMixFixture fixture({{1, LaneMask::aiv0()}, {2, kAicAiv0, true}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *local = fixture.locals[1];
    uint32_t direct = UINT32_MAX;
    bool failed = false;
    ASSERT_TRUE(
        scheduler_fill_own_slots(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, &direct)
    );
    ASSERT_FALSE(failed);
    ASSERT_LT(direct, SCHEDULER_PENDING_SLOT_COUNT);
    const auto handle = local->slots[direct].prepared;
    ASSERT_TRUE(fixture.admit(1));
    bool permit = true;
    ASSERT_TRUE(
        scheduler_ordered_can_enter_slot(fixture.graph.graph(), fixture.base(), local, fixture.run(), direct, &permit)
    );
    ASSERT_FALSE(permit);
    scheduler_local_ready_publish(local, direct);
    EXPECT_EQ(local->slots[direct].state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(local->slots[direct].prepared.execution_index, handle.execution_index);
    EXPECT_EQ(local->slots[direct].prepared.claimed_state, handle.claimed_state);
    EXPECT_EQ(local->local_ready_mask, 1U << direct);
    std::set<std::tuple<int64_t, uint32_t, uint8_t>> completed;
    for (uint32_t pass = 0; pass < 64 && completed.size() != 5; ++pass) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) {
                ASSERT_TRUE(fixture.service_cooperative(worker));
                continue;
            }
            EXPECT_TRUE(completed.emplace(execution.task, execution.block, execution.subtask).second);
            ASSERT_TRUE(fixture.complete(execution));
        }
    }
    EXPECT_EQ(completed.size(), 5u);
    EXPECT_EQ(
        scheduler_completion_task_state(fixture.control(0)->completion_state),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
}

void service_cluster_members(GmMixFixture &fixture, uint32_t cluster = 0, uint64_t profiling = 0) {
    for (uint32_t pass = 0; pass < 2; ++pass)
        for (uint32_t lane = 0; lane < 3; ++lane) {
            bool progress = false;
            ASSERT_TRUE(fixture.ordinary_service(cluster * 3 + lane, &progress, profiling));
        }
}

TEST(HbgClusterGroupOwnership, VectorGroupExecutesOnlyOnItsAssignedCluster) {
    GmMixFixture fixture({{1, LaneMask::aic()}, {1, LaneMask::both_aiv()}, {1, LaneMask::both_aiv()}});
    fixture.graph.executable(1, 1, {0});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.claim_ordinary(0, 3));
    ASSERT_TRUE(fixture.progress_ordinary(3));
    GmExecution producer{};
    ASSERT_TRUE(fixture.begin(3, &producer));
    auto *first = fixture.locals[0];
    bool progress = false;
    ASSERT_TRUE(
        scheduler_cluster_mix_discover_task(fixture.graph.graph(), fixture.base(), first, fixture.run(), 1, &progress)
    );
    ASSERT_TRUE(progress);
    const auto reserved = first->cluster_pending.anchor;
    auto *second = fixture.locals[3];
    EXPECT_EQ(first->issue_cursor, 2u);
    EXPECT_EQ(second->issue_cursor, 1u);
    ASSERT_TRUE(
        scheduler_cluster_mix_discover_task(fixture.graph.graph(), fixture.base(), second, fixture.run(), 1, &progress)
    );
    EXPECT_TRUE(progress);
    EXPECT_EQ(first->issue_cursor, 2u);
    EXPECT_EQ(second->issue_cursor, 2u);
    EXPECT_EQ(fixture.control(1)->claim_seal, 1u);
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
    EXPECT_EQ(second->cluster_pending.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(first->cluster_pending.anchor.claimed_state, reserved.claimed_state);
    EXPECT_EQ(scheduler_prepared_owner(reserved.claimed_state), 0u);
    ASSERT_TRUE(fixture.complete(producer));
    ASSERT_TRUE(
        scheduler_cluster_mix_progress_pending(fixture.graph.graph(), fixture.base(), first, fixture.run(), &progress)
    );
    ASSERT_TRUE(progress);
    service_cluster_members(fixture);
    const auto anchor =
        scheduler_prepared_control_at(fixture.base(), first->config.prepared, reserved.execution_index)->state;
    EXPECT_EQ(scheduler_prepared_phase(anchor), SchedulerPreparedPhase::CLAIMED);
    EXPECT_EQ(scheduler_prepared_execute_owner(anchor), 1u);
    GmExecution coordinator{};
    EXPECT_FALSE(fixture.begin(0, &coordinator));
    for (uint32_t worker : {1u, 2u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.task, 1);
        EXPECT_EQ(execution.subtask, worker);
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_FALSE(fixture.task_done(1));
    ASSERT_TRUE(fixture.ordinary_service(0, &progress));
    EXPECT_TRUE(fixture.task_done(1));
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(1)->completion_state), 2u);
}

TEST(HbgClusterMixArgsDump, LastSnapshotHoldsExistingCompletionAcrossThreeWaves) {
    GmMixFixture fixture({{3, LaneMask::all()}, {1, LaneMask::aiv0()}}, false, 1);
    fixture.graph.executable(1, 1, {0});
    auto *local = fixture.locals[0];
    EXPECT_FALSE(fixture.task_done(1));
    ASSERT_TRUE(fixture.prepare_calls());
    SchedulerArgsDumpTask dump{};
    dump.task_ref = 0;
    dump.active_mask = 7;
    dump.expected_participants = 9;
    ASSERT_EQ(scheduler_args_dump_enter(&dump), SchedulerArgsDumpArrival::Publisher);
    dump.publisher_before = scheduler_args_dump_publisher(0);
    SchedulerArgsDumpAck before{11, 0, 0, 1, 1, 0, 0, 0};
    before.status = scheduler_args_dump_ack_commit(0, 0, SchedulerArgsDumpStatus::Sealed);
    ASSERT_TRUE(scheduler_args_dump_seal(&dump, ArgsDumpStage::BEFORE_DISPATCH, before, 11, 1));
    std::array<uint32_t, 9> visible{};
    GmExecution held{};
    for (uint32_t block = 0; block < 3; ++block) {
        bool armed = false;
        ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
        ASSERT_TRUE(armed);
        ASSERT_NO_FATAL_FAILURE(service_cluster_members(fixture));
        for (uint32_t worker = 0; worker < 3; ++worker) {
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(worker, &execution));
            ASSERT_EQ(scheduler_args_dump_enter(&dump), SchedulerArgsDumpArrival::Ready);
            const uint32_t index = block * 3 + worker;
            const auto arrival = scheduler_args_dump_return(&dump, [&] {
                visible[index] = index + 1;
                return true;
            });
            if (index == 8) {
                ASSERT_EQ(arrival, SchedulerArgsDumpArrival::Publisher);
                held = execution;
            } else {
                ASSERT_EQ(arrival, SchedulerArgsDumpArrival::Waiting);
                ASSERT_TRUE(fixture.complete(execution));
            }
        }
        bool progress = false;
        ASSERT_TRUE(fixture.ordinary_service(0, &progress));
        EXPECT_EQ(scheduler_cluster_mix_armed(fixture.signal(0)->armed_generation), block == 2);
        EXPECT_FALSE(fixture.task_done(0));
        EXPECT_EQ(fixture.control(1)->next_block, 0u);
        EXPECT_EQ(local->pending_completed, 0u);
    }
    EXPECT_EQ(visible, (std::array<uint32_t, 9>{1, 2, 3, 4, 5, 6, 7, 8, 9}));
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 6u);
    dump.publisher_after = scheduler_args_dump_publisher(held.worker);
    SchedulerArgsDumpAck after{11, 0, 1, scheduler_args_dump_publisher(held.worker), 1, 0, 0, 0};
    after.status = scheduler_args_dump_ack_commit(0, 1, SchedulerArgsDumpStatus::Sealed);
    ASSERT_TRUE(scheduler_args_dump_seal(&dump, ArgsDumpStage::AFTER_COMPLETION, after, 11, 1));
    ASSERT_TRUE(fixture.complete(held));
    bool progress = false;
    ASSERT_TRUE(fixture.ordinary_service(0, &progress));
    EXPECT_EQ(fixture.signal(0)->staged_mask, 0u);
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_EQ(fixture.locals[0]->pending_completed, 1u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 9u);
    uint32_t successor_entries = 0;
    for (uint32_t step = 0; step < 8; ++step) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_EQ(execution.task, 1);
            ++successor_entries;
            ASSERT_TRUE(fixture.complete(execution));
        }
    }
    EXPECT_EQ(successor_entries, 1u);
    EXPECT_TRUE(fixture.task_done(1));
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

class HbgClusterMixMembers : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgClusterMixMembers, LateArrivalAndLateReturnPreventEntryAndRecordReuse) {
    const auto mask = GetParam();
    const uint32_t late_lane = mask.has_aiv1() ? 2 : 1;
    GmMixFixture fixture({{2, mask}}, true, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    EXPECT_EQ(fixture.locals[0]->admission, nullptr);
    uint32_t previous_generation = 0;
    std::set<std::pair<uint32_t, uint8_t>> completed;
    for (uint32_t block = 0; block < 2; ++block) {
        bool armed = false;
        ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
        ASSERT_TRUE(armed);
        auto *signal = fixture.signal(0);
        const auto generation = signal->armed_generation;
        EXPECT_GT(generation, previous_generation);
        EXPECT_TRUE(scheduler_cluster_mix_armed(generation));
        EXPECT_EQ(signal->block_idx, block);
        bool progress = false;
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (lane == late_lane) continue;
            ASSERT_TRUE(fixture.ordinary_service(lane, &progress));
            GmExecution premature{};
            EXPECT_FALSE(fixture.begin(lane, &premature));
        }
        EXPECT_NE(signal->arrived[late_lane], generation);
        ASSERT_TRUE(fixture.ordinary_service(late_lane, &progress));
        ASSERT_NO_FATAL_FAILURE(service_cluster_members(fixture));
        GmExecution late{};
        for (uint32_t lane = 0; lane < 3; ++lane) {
            GmExecution execution{};
            const bool entered = fixture.begin(lane, &execution);
            ASSERT_EQ(entered, mask.has(static_cast<SubtaskSlot>(lane)));
            if (!entered) continue;
            EXPECT_EQ(execution.block, block);
            EXPECT_EQ(execution.subtask, lane);
            EXPECT_EQ(execution.physical_subblock, lane == 0 ? 0u : lane - 1);
            if (lane == late_lane) late = execution;
            else {
                EXPECT_TRUE(completed.emplace(execution.block, execution.subtask).second);
                ASSERT_TRUE(fixture.complete(execution));
            }
        }
        ASSERT_TRUE(fixture.ordinary_service(0, &progress));
        EXPECT_EQ(signal->armed_generation, generation);
        EXPECT_NE(signal->finished[late_lane], generation);
        ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
        EXPECT_FALSE(armed);
        EXPECT_EQ(fixture.control(0)->next_block, block + 1);
        EXPECT_TRUE(completed.emplace(late.block, late.subtask).second);
        ASSERT_TRUE(fixture.complete(late));
        ASSERT_TRUE(fixture.ordinary_service(0, &progress));
        EXPECT_FALSE(scheduler_cluster_mix_armed(signal->armed_generation));
        EXPECT_EQ(signal->staged_mask, 0u);
        previous_generation = generation;
    }
    EXPECT_EQ(completed.size(), 2 * mask.count());
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_EQ(fixture.control(0)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED));
    EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
    const auto *executions = scheduler_state_at<SchedulerProtocolExecutionTrace>(
        fixture.base(), fixture.storage.layout.protocol.executions_offset
    );
    for (uint32_t index = 0; index < 2 * mask.count(); ++index) {
        EXPECT_NE(executions[index].returned.sequence, 0u);
        EXPECT_EQ(executions[index].contribution_after.sequence, 0u);
        EXPECT_NE(executions[index].completion_consumed.sequence, 0u);
    }
    for (uint32_t worker = 0; worker < 3; ++worker) {
        bool done = false;
        ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
    }
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST_P(HbgClusterMixMembers, ExistingExecutionMustReturnBeforeItsLaneArrives) {
    const auto mask = GetParam();
    const uint32_t busy = mask.has_aiv0() ? 1 : 2;
    GmMixFixture fixture({{1, LaneMask::aiv0()}, {1, mask}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.fill(busy));
    GmExecution ordinary{};
    ASSERT_TRUE(fixture.begin(busy, &ordinary));
    ASSERT_EQ(ordinary.task, 0);
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(1, 0, &armed));
    ASSERT_TRUE(armed);
    const auto generation = fixture.signal(0)->armed_generation;
    ASSERT_NO_FATAL_FAILURE(service_cluster_members(fixture));
    EXPECT_NE(fixture.signal(0)->arrived[busy], generation);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        GmExecution premature{};
        EXPECT_FALSE(fixture.begin(lane, &premature));
    }
    ASSERT_TRUE(fixture.complete(ordinary));
    ASSERT_NO_FATAL_FAILURE(service_cluster_members(fixture));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        GmExecution execution{};
        if (!mask.has(static_cast<SubtaskSlot>(lane))) {
            EXPECT_FALSE(fixture.begin(lane, &execution));
            continue;
        }
        ASSERT_TRUE(fixture.begin(lane, &execution));
        EXPECT_EQ(execution.task, 1);
        ASSERT_TRUE(fixture.complete(execution));
    }
    bool progress = false;
    ASSERT_TRUE(fixture.ordinary_service(0, &progress));
    EXPECT_EQ(fixture.signal(0)->staged_mask, 0u);
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_TRUE(fixture.task_done(1));
}

TEST_P(HbgClusterMixMembers, PublishedErrorBlocksEntryAndRecordReuse) {
    const auto mask = GetParam();
    GmMixFixture fixture({{2, mask}, {1, mask}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    ASSERT_NO_FATAL_FAILURE(service_cluster_members(fixture));
    const auto generation = fixture.signal(0)->armed_generation;
    scheduler_publish_error(fixture.run(), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
        auto *local = fixture.locals[lane];
        const auto slot = local->cluster_mix.slot_index;
        ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
        bool permit = true;
        EXPECT_FALSE(scheduler_can_enter_dispatch_slot(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, &permit
        ));
        EXPECT_FALSE(permit);
        EXPECT_EQ(local->slots[slot].state, SchedulerDispatchSlotState::READY);
    }
    EXPECT_FALSE(scheduler_cluster_mix_try_arm(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &armed
    ));
    bool discovered = true;
    EXPECT_FALSE(scheduler_cluster_mix_discover_task(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 1, &discovered
    ));
    EXPECT_FALSE(discovered);
    EXPECT_EQ(fixture.locals[0]->cluster_pending.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    EXPECT_EQ(fixture.signal(0)->armed_generation, generation);
    EXPECT_EQ(fixture.control(0)->next_block, 1u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgClusterMixMembers, ::testing::ValuesIn(kMixMasks));

TEST(HbgClusterMix, UnusedVectorDoesNotDelayAnIndependentPair) {
    for (const auto mask : {kAicAiv0, kAicAiv1}) {
        const uint32_t active = mask.has_aiv0() ? 1 : 2;
        const uint32_t unused = 3 - active;
        GmMixFixture fixture({{1, LaneMask::aiv0()}, {1, mask}}, false, 1);
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.fill(unused));
        GmExecution held{};
        ASSERT_TRUE(fixture.begin(unused, &held));
        ASSERT_EQ(held.task, 0);
        bool armed = false;
        ASSERT_TRUE(fixture.ordinary_arm(1, 0, &armed));
        ASSERT_TRUE(armed);
        ASSERT_NO_FATAL_FAILURE(service_cluster_members(fixture));
        for (uint32_t lane : {0u, active}) {
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(lane, &execution));
            EXPECT_EQ(execution.task, 1);
            ASSERT_TRUE(fixture.complete(execution));
        }
        bool progress = false;
        ASSERT_TRUE(fixture.ordinary_service(0, &progress));
        EXPECT_TRUE(fixture.task_done(1));
        EXPECT_EQ(fixture.locals[unused]->slots[held.slot].state, SchedulerDispatchSlotState::EXECUTING);
        EXPECT_EQ(fixture.signal(0)->arrived[unused], 0u);
        EXPECT_EQ(fixture.signal(0)->finished[unused], 0u);
        ASSERT_TRUE(fixture.complete(held));
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgClusterMix, SkippedLanesRejoinLaterGenerationsAcrossChangingMasks) {
    const std::vector<GmShape> shapes{
        {2, kAicAiv0}, {3, kAicAiv1}, {1, kAicAiv0}, {2, LaneMask::all()}, {2, LaneMask::both_aiv()}
    };
    GmMixFixture fixture(shapes, false, 1);
    for (uint32_t task = 1; task < shapes.size(); ++task) {
        fixture.graph.executable(task, shapes[task].mask.has_aic() ? 0 : 1, {static_cast<int32_t>(task - 1)});
    }
    ASSERT_TRUE(fixture.prepare_calls());
    std::set<std::tuple<int64_t, uint32_t, uint8_t>> visits;
    for (uint32_t step = 0; step < 100; ++step) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_GE(execution.task, 0);
            ASSERT_LT(execution.task, shapes.size());
            EXPECT_LT(execution.block, shapes[execution.task].blocks);
            EXPECT_TRUE(shapes[execution.task].mask.has(static_cast<SubtaskSlot>(execution.subtask)));
            if (execution.task != 0) EXPECT_TRUE(fixture.task_done(execution.task - 1));
            EXPECT_TRUE(visits.emplace(execution.task, execution.block, execution.subtask).second);
            ASSERT_TRUE(fixture.complete(execution));
        }
    }
    for (uint32_t task = 0; task < shapes.size(); ++task) {
        EXPECT_TRUE(fixture.task_done(task));
        for (uint32_t block = 0; block < shapes[task].blocks; ++block)
            for (uint8_t lane = 0; lane < 3; ++lane)
                EXPECT_EQ(visits.count({task, block, lane}), shapes[task].mask.has(static_cast<SubtaskSlot>(lane)));
    }
    for (uint32_t worker = 0; worker < 3; ++worker) {
        bool done = false;
        ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
    }
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST(HbgClusterMix, UnobservedSyncTailAndAdoptedHoldBothPreventArming) {
    GmMixFixture fixture({{2, LaneMask::aiv0(), true}, {1, kAicAiv0}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0, 1));
    bool armed = true;
    ASSERT_TRUE(fixture.ordinary_arm(1, 0, &armed));
    EXPECT_FALSE(armed);
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    const auto reserved = fixture.locals[0]->cluster_pending.anchor;
    EXPECT_EQ(scheduler_prepared_phase(reserved.claimed_state), SchedulerPreparedPhase::GROUP_RESERVED);
    EXPECT_EQ(fixture.signal(0)->staged_mask, 0u);
    EXPECT_EQ(fixture.signal(0)->armed_generation, 0u);
    ASSERT_TRUE(fixture.refresh_all());
    EXPECT_NE(fixture.signal(0)->sync_hold[1], 0u);
    ASSERT_TRUE(fixture.ordinary_arm(1, 0, &armed));
    EXPECT_FALSE(armed);
    EXPECT_EQ(fixture.signal(0)->staged_mask, 0u);
    EXPECT_EQ(fixture.signal(0)->armed_generation, 0u);
    EXPECT_EQ(fixture.control(1)->next_block, 1u);
    EXPECT_EQ(fixture.locals[0]->cluster_pending.anchor.execution_index, reserved.execution_index);
    EXPECT_EQ(fixture.locals[0]->cluster_pending.anchor.claimed_state, reserved.claimed_state);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(fixture.signal(0)->arrived[lane], 0u);
        EXPECT_EQ(fixture.locals[lane]->local_ready_mask, 0u);
        for (const auto &slot : fixture.locals[lane]->slots)
            EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    }
}

TEST(HbgClusterMix, StagedOrdinaryGroupBlocksNewSyncTicketBeforeSlotOwnership) {
    GmMixFixture fixture({{2, LaneMask::aiv0(), true}, {1, kAicAiv0}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0, 1));
    // Pause the coordinator after its staged-mask publication, before the
    // second hold read that either retracts the stage or publishes armed.
    scheduler_cluster_store_release(&fixture.signal(0)->staged_mask, UINT32_C(3));
    ASSERT_TRUE(fixture.refresh(1));
    EXPECT_NE(fixture.signal(0)->sync_hold[1], 0u);
    uint32_t block = UINT32_MAX;
    EXPECT_EQ(
        scheduler_claim_ordered_cursor(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), &block),
        SchedulerBlockClaimResult::RETRY
    );
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    for (const auto &slot : fixture.locals[1]->slots)
        EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    scheduler_cluster_store_release(&fixture.signal(0)->staged_mask, UINT32_C(0));
    EXPECT_EQ(
        scheduler_claim_ordered_cursor(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), &block),
        SchedulerBlockClaimResult::CLAIMED
    );
    EXPECT_EQ(block, 0u);
}

TEST(HbgClusterMix, ArmedVectorOnlyGroupKeepsCoordinatorOutOfLaterSyncUntilRetirement) {
    GmMixFixture fixture({{1, LaneMask::both_aiv()}, {2, LaneMask::aic(), true}}, false, 2);
    ASSERT_TRUE(fixture.prepare_calls());
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    ASSERT_TRUE(fixture.admit(1));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(service_cluster_members(fixture));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    uint32_t block = UINT32_MAX;
    EXPECT_EQ(
        scheduler_claim_ordered_cursor(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &block),
        SchedulerBlockClaimResult::RETRY
    );
    EXPECT_EQ(fixture.control(1)->next_block, 0u);
    for (uint32_t lane : {1u, 2u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        EXPECT_EQ(execution.task, 0);
        ASSERT_TRUE(fixture.complete(execution));
    }
    bool progress = false;
    ASSERT_TRUE(fixture.ordinary_service(0, &progress));
    EXPECT_EQ(fixture.signal(0)->staged_mask, 0u);
    EXPECT_EQ(
        scheduler_claim_ordered_cursor(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &block),
        SchedulerBlockClaimResult::CLAIMED
    );
    EXPECT_EQ(block, 0u);
}

class HbgClusterMixMultiWave : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};

TEST_P(HbgClusterMixMultiWave, EveryReturnPrecedesExactlyOneSuccessor) {
    const auto [mask, blocks] = GetParam();
    GmMixFixture fixture({{blocks, mask}, {1, LaneMask::aiv0()}}, false, 2);
    fixture.graph.executable(1, 1, {0});
    ASSERT_TRUE(fixture.prepare_calls());

    std::set<std::pair<uint32_t, uint8_t>> visits;
    uint32_t successor_entries = 0;
    for (uint32_t step = 0; step < blocks * 100 + 100; ++step) {
        for (uint32_t index = 0; index < 6; ++index) {
            const uint32_t worker = (index + step) % 6;
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            if (execution.task == 0) {
                EXPECT_LT(execution.block, blocks);
                EXPECT_TRUE(visits.emplace(execution.block, execution.subtask).second);
            } else {
                ASSERT_EQ(execution.task, 1);
                EXPECT_TRUE(fixture.task_done(0));
                EXPECT_EQ(visits.size(), blocks * mask.count());
                ++successor_entries;
            }
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (fixture.task_done(1)) break;
    }
    EXPECT_EQ(successor_entries, 1u);
    for (uint32_t block = 0; block < blocks; ++block)
        for (uint8_t lane = 0; lane < 3; ++lane)
            EXPECT_EQ(visits.count({block, lane}), mask.has(static_cast<SubtaskSlot>(lane)) ? 1u : 0u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), blocks * mask.count());
    EXPECT_TRUE(fixture.task_done(1));
    for (uint32_t worker = 0; worker < 6; ++worker) {
        bool done = false;
        for (uint32_t pass = 0; pass < 4 && !done; ++pass)
            ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
        scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
    }
    EXPECT_EQ(fixture.run()->resolved_task_count, 2u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndWaves, HbgClusterMixMultiWave,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values<uint16_t>(3, 17, 65))
);

class HbgClusterMixShapes : public ::testing::TestWithParam<std::tuple<LaneMask, uint16_t>> {};

TEST_P(HbgClusterMixShapes, RealOwnDispatchAndCompletionKeepEveryLogicalBlockAndSubtaskExactlyOnce) {
    const auto [mask, blocks] = GetParam();
    GmMixFixture fixture({{blocks, mask}});

    ASSERT_TRUE(fixture.prepare_calls());

    std::map<std::pair<uint32_t, uint8_t>, uint32_t> visits;
    std::set<uint32_t> single_vector_lanes;
    const uint64_t expected = blocks * static_cast<uint32_t>(mask.count());
    uint64_t executions = 0;
    for (uint32_t step = 0; step < blocks * 100 + 100; ++step) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            ASSERT_EQ(execution.task, 0);
            ASSERT_EQ(execution.blocks, blocks);
            ASSERT_LT(execution.block, blocks);
            ASSERT_LT(execution.subtask, 3);
            ASSERT_TRUE(mask.has(static_cast<SubtaskSlot>(execution.subtask)));
            EXPECT_TRUE(fixture.locals[worker]->slots[execution.slot].cluster_mix);
            EXPECT_EQ(worker % 3, execution.subtask);
            if (execution.subtask != 0) {
                EXPECT_EQ(worker % 3, execution.subtask);
                EXPECT_EQ(execution.physical_subblock, worker % 3 - 1);
                if (mask.has_aic() && !mask.has_both_aiv()) single_vector_lanes.insert(worker % 3);
            }
            ++visits[{execution.block, execution.subtask}];
            ++executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (scheduler_completion_done(scheduler_gm_query(fixture.control(0)->completion_state))) break;
    }
    EXPECT_EQ(executions, expected);
    for (uint32_t block = 0; block < blocks; ++block)
        for (uint8_t subtask = 0; subtask < 3; ++subtask)
            EXPECT_EQ(visits[std::make_pair(block, subtask)], mask.has(static_cast<SubtaskSlot>(subtask)) ? 1u : 0u);
    if (mask.has_aic() && !mask.has_both_aiv())
        EXPECT_EQ(single_vector_lanes, (std::set<uint32_t>{mask.has_aiv0() ? 1U : 2U}));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), blocks);
    EXPECT_EQ(fixture.control(0)->claim_tickets, 0u);
    EXPECT_EQ(fixture.control(0)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED));
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), expected
    );
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    std::array<bool, 6> done{};
    for (uint32_t step = 0; step < 16; ++step)
        for (uint32_t worker = 0; worker < 6; ++worker)
            ASSERT_TRUE(fixture.drain(worker, &done[worker]));
    EXPECT_TRUE(std::all_of(done.begin(), done.end(), [](bool value) {
        return value;
    }));
    for (uint32_t worker = 0; worker < 6; ++worker)
        scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndClusterCapacity, HbgClusterMixShapes,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values<uint16_t>(1, 2, 3, 4, 5))
);

// GM descriptors, acknowledgements and retained close certificates belong
// to sync_start; ordinary MIX above uses the cluster SSBUF protocol.

void finish_sync_cluster(GmMixFixture &fixture, uint32_t cluster) {
    ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(cluster));
    ASSERT_TRUE(fixture.rendezvous(cluster));
    ASSERT_TRUE(fixture.finish_lanes(cluster));
    ASSERT_EQ(fixture.retire(cluster), SchedulerGmMixResult::PROGRESS);
}

class HbgGmMixSyncStart : public ::testing::TestWithParam<LaneMask> {};

class HbgGmMixIdlePoll : public ::testing::TestWithParam<std::tuple<LaneMask, uint32_t>> {};

void expect_admitted_task_owned(GmMixFixture &fixture, uint32_t task = 0) {
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(task)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
    );
    const auto *record = scheduler_admission_record_at(fixture.base(), &fixture.admissions[0], task);
    const auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    ASSERT_LT(record->order_key, shared->commit_tail);
    const auto *log = scheduler_state_at<int64_t>(fixture.base(), fixture.storage.layout.admission_log_offset);
    EXPECT_EQ(log[record->order_key], task);
}

void depend_on_inline_gate(GmMixFixture &fixture, uint32_t task, uint32_t gate = 1) {
    const auto &metadata = fixture.storage.metadata[task];
    uint8_t lane = metadata.active_mask.has_aic() ? 0 : metadata.active_mask.has_aiv0() ? 1 : 2;
    fixture.graph.executable(task, lane, {static_cast<int32_t>(gate)});
    ASSERT_EQ(fixture.run()->bootstrap_arrived_count, 0u);
}

void complete_inline_gate(GmMixFixture &fixture, uint32_t worker, uint32_t gate = 1) {
    auto *local = fixture.locals[worker];
    local->control_peer_cursor = gate;
    bool progress = false;
    ASSERT_TRUE(scheduler_service_controls(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress));
    ASSERT_TRUE(progress);
    ASSERT_TRUE(fixture.task_done(gate));
}

TEST_P(HbgGmMixIdlePoll, UnpublishedClaimIsIdleButDescriptorAndRetainedSealWakeTheVector) {
    const auto [mask, lane] = GetParam();
    GmMixFixture fixture({{2, mask, true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_admitted_task_owned(fixture));
    auto *local = fixture.locals[lane];
    const bool member = fixture.participant(lane);
    EXPECT_FALSE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    EXPECT_FALSE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
    ASSERT_TRUE(fixture.fill(lane));
    EXPECT_EQ(fixture.admissions[lane].current_task_ref, member ? 0 : SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(fixture.admissions[lane].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
         SchedulerOrderedPollResult::READY),
        member
    );
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    EXPECT_FALSE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
         SchedulerOrderedPollResult::READY),
        member
    );
    ASSERT_TRUE(fixture.fill(lane));
    EXPECT_EQ(fixture.admissions[lane].current_task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_FALSE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
}

TEST_P(HbgGmMixIdlePoll, WaitingForPartnersDoesNotHideAnOrdinaryReadyTask) {
    const auto [mask, lane] = GetParam();
    GmMixFixture fixture({{2, mask, true}, {1, LaneMask::aiv0(), false, true}, {1, LaneMask::aiv0()}});
    depend_on_inline_gate(fixture, 2);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_admitted_task_owned(fixture));
    auto *local = fixture.locals[lane];
    EXPECT_FALSE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
    EXPECT_FALSE(fixture.claim_ordinary(2, lane));
    EXPECT_EQ(fixture.control(2)->next_block, 0u);
    ASSERT_TRUE(fixture.progress_ordinary(lane));
    EXPECT_EQ(local->local_ready_mask, 0u);
    GmExecution execution{};
    EXPECT_FALSE(fixture.begin(lane, &execution));
    EXPECT_EQ(scheduler_completion_count(fixture.control(2)->completion_state), 0u);
    ASSERT_NO_FATAL_FAILURE(complete_inline_gate(fixture, lane));
    ASSERT_TRUE(fixture.claim_ordinary(2, lane));
    ASSERT_TRUE(fixture.progress_ordinary(lane));
    const bool entered = fixture.begin(lane, &execution);
    EXPECT_EQ(entered, !fixture.participant(lane));
    if (entered) {
        EXPECT_EQ(execution.task, 2);
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_EQ(scheduler_completion_count(fixture.control(2)->completion_state), entered ? 1u : 0u);
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndVectors, HbgGmMixIdlePoll,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values<uint32_t>(1, 2))
);

TEST_P(HbgGmMixSyncStart, IdlePollRequiresAllArrivalAndReleaseAcknowledgements) {
    GmMixFixture fixture({{2, GetParam(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_admitted_task_owned(fixture));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    ASSERT_TRUE(fixture.arrive(0, &progress));
    EXPECT_FALSE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
    for (uint32_t lane : {1u, 2u}) {
        if (!fixture.participant(lane)) continue;
        EXPECT_FALSE(
            (scheduler_observe_ordered_progress(
                 fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()
             ) == SchedulerOrderedPollResult::READY)
        );
        ASSERT_TRUE(fixture.observe(lane, &progress));
        ASSERT_TRUE(fixture.arrive(lane, &progress));
    }
    EXPECT_TRUE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
    ASSERT_TRUE(fixture.rendezvous(0));
    if (GetParam().has_aic()) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(0, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_FALSE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
    for (uint32_t lane = 1; lane < 3; ++lane) {
        if (!GetParam().has(static_cast<SubtaskSlot>(lane))) continue;
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(lane, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_TRUE(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()) ==
         SchedulerOrderedPollResult::READY)
    );
    ASSERT_TRUE(fixture.fill(0));
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), GetParam().count());
}

struct GmServiceCase {
    uint32_t worker;
    SchedulerGmMixPhase phase;
    bool record_protocol{true};
};

void prepare_service_phase(GmMixFixture &fixture, const GmServiceCase &scenario) {
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    if (scenario.phase == SchedulerGmMixPhase::IDLE) return;
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    if (scenario.phase == SchedulerGmMixPhase::CLAIMED) return;
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    if (scenario.worker != 0) ASSERT_TRUE(fixture.observe(scenario.worker, &progress));
    if (scenario.phase == SchedulerGmMixPhase::PUBLISHED) return;
    if (scenario.phase == SchedulerGmMixPhase::ARRIVED) {
        ASSERT_TRUE(fixture.arrive(scenario.worker, &progress));
        return;
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    if (scenario.phase == SchedulerGmMixPhase::RELEASED) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(scenario.worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
}

class HbgGmMixServiceError : public ::testing::TestWithParam<GmServiceCase> {};

TEST_P(HbgGmMixServiceError, ServiceRejectsPublishedErrorBeforeAnyStateSpecificWork) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    const auto scenario = GetParam();
    if (!scenario.record_protocol)
        for (auto *local : fixture.locals)
            local->profiling = nullptr;
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, scenario));
    ASSERT_EQ(fixture.admissions[scenario.worker].gm_mix.phase, scenario.phase);
    const auto before = fixture.admissions[scenario.worker].gm_mix;
    const uint64_t cursor = fixture.control(0)->next_block;
    const uint64_t published = fixture.mailbox(0)->descriptor.group_seq;
    const uint8_t ready_mask = fixture.locals[scenario.worker]->local_ready_mask;
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
        SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
    );
    bool progress = true;
    const auto service = [&](auto record) {
        return scheduler_service_gm_mix<decltype(record)::value>(
            fixture.graph.graph(), fixture.base(), fixture.locals[scenario.worker], fixture.run(), 0, &progress
        );
    };
    EXPECT_FALSE(scenario.record_protocol ? service(std::true_type{}) : service(std::false_type{}));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.phase, before.phase);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.group_seq, before.group_seq);
    EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.slot_index, before.slot_index);
    EXPECT_EQ(fixture.control(0)->next_block, cursor);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, published);
    EXPECT_EQ(fixture.locals[scenario.worker]->local_ready_mask, ready_mask);
    EXPECT_EQ(fixture.frontier(0), 0u);
    EXPECT_EQ(fixture.run()->error_worker_index, 4u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ORDERED_EXIT_TIMEOUT));
}

TEST_P(HbgGmMixServiceError, AHealthyServicePassDoesNotCacheTheNextPassEpochOrErrorPermission) {
    const auto scenario = GetParam();
    for (uint32_t fault = 0; fault < 3; ++fault) {
        SCOPED_TRACE(fault);
        GmMixFixture fixture({{2, LaneMask::all(), true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, scenario));
        auto *local = fixture.locals[scenario.worker];
        bool progress = false;
        const auto service = [&](auto record) {
            return scheduler_service_gm_mix<decltype(record)::value>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
            );
        };
        ASSERT_TRUE(scenario.record_protocol ? service(std::true_type{}) : service(std::false_type{}));
        const auto before = fixture.admissions[scenario.worker].gm_mix;
        const uint64_t cursor = fixture.control(0)->next_block;
        const uint64_t completed = scheduler_completion_count(fixture.control(0)->completion_state);
        const uint64_t published = fixture.mailbox(0)->descriptor.group_seq;
        const uint8_t ready = local->local_ready_mask;
        if (fault == 0) {
            ++fixture.admissions[scenario.worker].epoch;
        } else if (fault == 1) {
            ++scheduler_state_at<SchedulerGraphSummary>(fixture.base(), local->config.graph_summary_offset)->run_epoch;
        } else {
            scheduler_record_error(
                fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
                SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
            );
        }
        progress = true;
        EXPECT_FALSE(scenario.record_protocol ? service(std::true_type{}) : service(std::false_type{}));
        EXPECT_FALSE(progress);
        EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.phase, before.phase);
        EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.group_seq, before.group_seq);
        EXPECT_EQ(fixture.admissions[scenario.worker].gm_mix.slot_index, before.slot_index);
        EXPECT_EQ(fixture.control(0)->next_block, cursor);
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), completed);
        EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, published);
        EXPECT_EQ(local->local_ready_mask, ready);
        EXPECT_EQ(fixture.frontier(0), 0u);
        EXPECT_EQ(
            fixture.run()->error_site,
            static_cast<uint64_t>(
                fault == 2 ? SchedulerErrorSite::ORDERED_EXIT_TIMEOUT : SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH
            )
        );
    }
}

TEST_P(HbgGmMixServiceError, ServiceValidatesTheWholeContextBeforePhaseSpecificWork) {
    const auto scenario = GetParam();
    for (uint32_t invalid = 0; invalid < 5; ++invalid) {
        SCOPED_TRACE(invalid);
        GmMixFixture fixture({{2, LaneMask::all(), true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_NO_FATAL_FAILURE(prepare_service_phase(fixture, scenario));
        auto *local = fixture.locals[scenario.worker];
        auto &admission = fixture.admissions[scenario.worker];
        const auto before = admission.gm_mix;
        const uint64_t cursor = fixture.control(0)->next_block;
        const uint64_t published = fixture.mailbox(0)->descriptor.group_seq;
        const uint8_t ready = local->local_ready_mask;
        if (invalid == 0) admission.current_task_ref = SCHEDULER_TASK_REF_INVALID;
        else if (invalid == 1) admission.current_flags &= ~SCHEDULER_TASK_MIX;
        else if (invalid == 2) admission.gm_mix.mailbox_offset = 0;
        else if (invalid == 3) admission.gm_mix.cluster_index = local->config.cluster_count;
        else local->config.self_lane = PLATFORM_CORES_PER_BLOCKDIM;
        bool progress = true;
        const auto service = [&](auto record) {
            return scheduler_service_gm_mix<decltype(record)::value>(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
            );
        };
        EXPECT_FALSE(scenario.record_protocol ? service(std::true_type{}) : service(std::false_type{}));
        EXPECT_FALSE(progress);
        EXPECT_EQ(admission.gm_mix.phase, before.phase);
        EXPECT_EQ(admission.gm_mix.group_seq, before.group_seq);
        EXPECT_EQ(admission.gm_mix.slot_index, before.slot_index);
        EXPECT_EQ(fixture.control(0)->next_block, cursor);
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
        EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, published);
        EXPECT_EQ(local->local_ready_mask, ready);
        EXPECT_EQ(fixture.frontier(0), 0u);
        EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
    }
}

std::vector<GmServiceCase> gm_service_cases() {
    std::vector<GmServiceCase> cases{
        GmServiceCase{0, SchedulerGmMixPhase::IDLE},      GmServiceCase{0, SchedulerGmMixPhase::CLAIMED},
        GmServiceCase{0, SchedulerGmMixPhase::PUBLISHED}, GmServiceCase{0, SchedulerGmMixPhase::ARRIVED},
        GmServiceCase{0, SchedulerGmMixPhase::RUNNING},   GmServiceCase{0, SchedulerGmMixPhase::RELEASED},
        GmServiceCase{1, SchedulerGmMixPhase::IDLE},      GmServiceCase{1, SchedulerGmMixPhase::PUBLISHED},
        GmServiceCase{1, SchedulerGmMixPhase::ARRIVED},   GmServiceCase{1, SchedulerGmMixPhase::RUNNING},
        GmServiceCase{1, SchedulerGmMixPhase::RELEASED}
    };
    const auto recorded = cases;
    for (auto scenario : recorded) {
        scenario.record_protocol = false;
        cases.push_back(scenario);
    }
    return cases;
}

INSTANTIATE_TEST_SUITE_P(CoordinatorAndVectorPhases, HbgGmMixServiceError, ::testing::ValuesIn(gm_service_cases()));

class HbgGmMixBoundedValidation : public ::testing::TestWithParam<std::tuple<uint32_t, uint32_t, bool>> {};

TEST_P(HbgGmMixBoundedValidation, AValidatedStaticContextNeverSuppressesFreshErrorAndEpochChecks) {
    const auto [operation, fault, record_protocol] = GetParam();
    for (LaneMask mask : kMixMasks) {
        for (uint32_t worker = 0; worker < 3; ++worker) {
            if (!scheduler_mix_resource_mask(mask).has(static_cast<SubtaskSlot>(worker))) continue;
            SCOPED_TRACE(lane_names(mask));
            SCOPED_TRACE(worker);
            GmMixFixture fixture({{2, mask, true}});
            ASSERT_TRUE(fixture.prepare_calls());
            ASSERT_TRUE(fixture.admit(0));
            ASSERT_TRUE(fixture.refresh_all());
            auto *local = fixture.locals[worker];
            ASSERT_TRUE(scheduler_gm_mix_context_valid(fixture.graph.graph(), fixture.base(), local, fixture.run()));
            if (fault == 0) {
                ++fixture.admissions[worker].epoch;
            } else if (fault == 1) {
                ++scheduler_state_at<SchedulerGraphSummary>(fixture.base(), local->config.graph_summary_offset)
                      ->run_epoch;
            } else {
                scheduler_record_error(
                    fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
                    SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
                );
            }
            bool progress = true;
            const auto apply = [&](auto record) {
                constexpr bool enabled = decltype(record)::value;
                if (operation == 0)
                    return scheduler_gm_mix_close<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), &progress
                    );
                if (operation == 1)
                    return scheduler_gm_mix_claim<enabled, true>(
                               fixture.graph.graph(), fixture.base(), local, fixture.run()
                           ) != SchedulerBlockClaimResult::ERROR;
                if (operation == 2)
                    return scheduler_gm_mix_publish<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run()
                    );
                if (operation == 3)
                    return scheduler_gm_mix_observe<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), &progress
                    );
                if (operation == 4)
                    return scheduler_gm_mix_arrive<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), &progress
                    );
                if (operation == 5)
                    return scheduler_gm_mix_prepare_lane<enabled, true>(
                        fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress
                    );
                return scheduler_gm_mix_retire<enabled, true>(
                           fixture.graph.graph(), fixture.base(), local, fixture.run()
                       ) != SchedulerGmMixResult::ERROR;
            };
            EXPECT_FALSE(record_protocol ? apply(std::true_type{}) : apply(std::false_type{}));
            EXPECT_EQ(fixture.admissions[worker].gm_mix.phase, SchedulerGmMixPhase::IDLE);
            EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 0u);
            EXPECT_EQ(fixture.control(0)->next_block, 0u);
            EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
            EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 0u);
            EXPECT_EQ(local->local_ready_mask, 0u);
            EXPECT_EQ(fixture.frontier(0), 0u);
            EXPECT_EQ(
                fixture.run()->error_site,
                static_cast<uint64_t>(
                    fault == 2 ? SchedulerErrorSite::ORDERED_EXIT_TIMEOUT : SchedulerErrorSite::ADMISSION_EPOCH_MISMATCH
                )
            );
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
    StageAndHealth, HbgGmMixBoundedValidation,
    ::testing::Combine(::testing::Range(0u, 7u), ::testing::Range(0u, 3u), ::testing::Bool())
);

TEST(HbgGmMix, ServiceCarriesNewPublicationThroughArrivalWithoutAnExtraPoll) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask, true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        for (uint32_t worker = 0; worker < 3; ++worker) {
            if (!fixture.participant(worker)) continue;
            bool progress = false;
            ASSERT_TRUE(scheduler_service_gm_mix(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, &progress
            ));
            EXPECT_TRUE(progress);
            EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 1u);
            EXPECT_EQ(fixture.mailbox(0)->lanes[worker].arrived_seq, worker == 0 ? 0u : 1u);
            EXPECT_EQ(fixture.admissions[worker].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
        }
        for (uint32_t worker = 0; worker < 3; ++worker) {
            if (!fixture.participant(worker) || fixture.admissions[worker].gm_mix.phase != SchedulerGmMixPhase::ARRIVED)
                continue;
            bool progress = false;
            ASSERT_TRUE(scheduler_service_gm_mix(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, &progress
            ));
            EXPECT_TRUE(progress);
            EXPECT_EQ(
                fixture.admissions[worker].gm_mix.phase, mask.has(static_cast<SubtaskSlot>(worker)) ?
                                                             SchedulerGmMixPhase::RUNNING :
                                                             SchedulerGmMixPhase::RELEASED
            );
        }
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
    }
}

TEST(HbgClusterMix, ArrivalDuringPreparationPublishesInTheSameService) {
    GmMixFixture fixture({{1, LaneMask::all()}}, false, 1);
    ASSERT_TRUE(fixture.prepare_calls());
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    const uint32_t generation = fixture.signal(0)->armed_generation;
    ScopedGmPrepareObserver observer([&](SchedulerLocalState *local) {
        if (local != fixture.locals[0]) return;
        EXPECT_NE(fixture.signal(0)->arrived[1], generation);
        for (uint32_t worker : {1u, 2u}) {
            bool progress = false;
            ASSERT_TRUE(fixture.ordinary_service(worker, &progress));
            ASSERT_EQ(fixture.signal(0)->arrived[worker], generation);
        }
    });
    bool progress = false;
    ASSERT_TRUE(fixture.ordinary_service(0, &progress));
    EXPECT_EQ(fixture.locals[0]->cluster_mix.phase, SchedulerGmMixPhase::RUNNING);
    EXPECT_NE(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
}

TEST(HbgGmMix, ArrivalDuringPreparationPublishesImmediatelyAndWaitingHandlesAreRetained) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    bool progress = false;
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        for (uint32_t lane : {1u, 2u})
            ASSERT_TRUE(fixture.observe(cluster * 3 + lane, &progress));
        ASSERT_TRUE(fixture.arrive(cluster * 3, &progress));
    }
    uint32_t prepared_calls = 0;
    ScopedGmPrepareObserver observer([&](SchedulerLocalState *local) {
        ++prepared_calls;
        if (local != fixture.locals[0]) return;
        EXPECT_EQ(fixture.mailbox(0)->lanes[1].arrived_seq, 0u);
        for (uint32_t worker : {1u, 2u}) {
            bool arrived = false;
            ASSERT_TRUE(fixture.arrive(worker, &arrived));
            ASSERT_TRUE(arrived);
        }
    });

    ASSERT_TRUE(fixture.prepare(0, &progress));
    EXPECT_EQ(prepared_calls, 1u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RUNNING);
    EXPECT_NE(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);

    ASSERT_TRUE(fixture.prepare(3, &progress));
    EXPECT_EQ(prepared_calls, 2u);
    EXPECT_EQ(fixture.admissions[3].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    const uint32_t slot = fixture.admissions[3].gm_mix.slot_index;
    ASSERT_LT(slot, SCHEDULER_PENDING_SLOT_COUNT);
    const auto held = fixture.locals[3]->slots[slot].prepared;
    ASSERT_NE(held.claimed_state, 0u);
    EXPECT_EQ(fixture.locals[3]->slots[slot].state, SchedulerDispatchSlotState::FILLING);
    ASSERT_TRUE(fixture.prepare(3, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(prepared_calls, 2u);
    for (uint32_t worker : {4u, 5u})
        ASSERT_TRUE(fixture.arrive(worker, &progress));
    ASSERT_TRUE(fixture.prepare(3, &progress));
    EXPECT_EQ(prepared_calls, 2u);
    EXPECT_EQ(fixture.admissions[3].gm_mix.phase, SchedulerGmMixPhase::RUNNING);
    EXPECT_EQ(fixture.locals[3]->slots[slot].prepared.claimed_state, held.claimed_state);
    EXPECT_EQ(fixture.locals[3]->slots[slot].prepared.execution_index, held.execution_index);
    for (uint32_t worker : {1u, 2u, 4u, 5u})
        ASSERT_TRUE(fixture.prepare(worker, &progress));
    for (uint32_t worker = 0; worker < 6; ++worker) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        EXPECT_EQ(execution.task, 0);
        EXPECT_EQ(execution.block, worker / 3);
        ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_FALSE(fixture.task_done(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgGmMix, PreparedArrivalExecutesBeforeDiscoveringTheNextSyncAdmission) {
    for (const bool direct_output : {false, true}) {
        SCOPED_TRACE(direct_output);
        GmMixFixture fixture({{2, LaneMask::all(), true}, {1, LaneMask::aiv0()}, {2, LaneMask::aiv0(), true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        bool progress = false;
        for (uint32_t worker : {0u, 1u}) {
            ASSERT_TRUE(scheduler_service_gm_mix(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), 0, &progress
            ));
            ASSERT_EQ(fixture.admissions[worker].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
        }
        auto *local = fixture.locals[1];
        const uint32_t held_slot = fixture.admissions[1].gm_mix.slot_index;
        ASSERT_LT(held_slot, SCHEDULER_PENDING_SLOT_COUNT);
        const auto held = local->slots[held_slot].prepared;
        ASSERT_NE(held.claimed_state, 0u);
        ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.claim_ordinary(1, 1));
        EXPECT_EQ(fixture.mailbox(0)->lanes[2].arrived_seq, 0u);

        // Advance only the issuance frontier; the next cohort has no proposal.
        int64_t head = SCHEDULER_TASK_REF_INVALID;
        for (uint32_t slice = 0; slice < 3 && scheduler_task_ref_is_null(head); ++slice) {
            ASSERT_TRUE(scheduler_query_issue_head(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), &head, &progress
            ));
        }
        ASSERT_EQ(head, 2);
        ASSERT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        ASSERT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);

        ASSERT_TRUE(scheduler_service_gm_mix(
            fixture.graph.graph(), fixture.base(), fixture.locals[2], fixture.run(), 0, &progress
        ));
        ASSERT_EQ(fixture.admissions[2].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
        ASSERT_TRUE(scheduler_service_gm_mix(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), 0, &progress
        ));
        EXPECT_EQ(fixture.signal(0)->gm_arrival_seq, 1u);
        ASSERT_TRUE(scheduler_service_gm_mix(
            fixture.graph.graph(), fixture.base(), fixture.locals[2], fixture.run(), 0, &progress
        ));
        ASSERT_EQ(fixture.admissions[2].gm_mix.phase, SchedulerGmMixPhase::RUNNING);
        bool failed = false;
        uint32_t direct = UINT32_MAX;
        EXPECT_TRUE(scheduler_fill_own_slots(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, direct_output ? &direct : nullptr
        ));
        EXPECT_FALSE(failed);
        EXPECT_EQ(fixture.admissions[1].gm_mix.phase, SchedulerGmMixPhase::RUNNING);
        EXPECT_EQ(local->slots[held_slot].state, SchedulerDispatchSlotState::READY);
        EXPECT_EQ(local->slots[held_slot].prepared.claimed_state, held.claimed_state);
        EXPECT_EQ(local->slots[held_slot].prepared.execution_index, held.execution_index);
        EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        EXPECT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.control(2)->next_block, 0u);
        EXPECT_EQ(direct, direct_output ? held_slot : UINT32_MAX);
        if (!direct_output) EXPECT_NE(local->local_ready_mask, 0u);
    }
}

TEST(HbgGmMix, PublishedRunErrorRejectsNewClaimAndDescriptorPublication) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask, true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        EXPECT_EQ(fixture.claim(1), SchedulerBlockClaimResult::ERROR);
        EXPECT_FALSE(fixture.publish(0));
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 0u);
        EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::CLAIMED);
        EXPECT_EQ(fixture.admissions[3].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    }
}

TEST(HbgGmMix, PublishedRunErrorRejectsAlreadyPreparedKernelEntryOnEveryActiveLane) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask, true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
            auto *local = fixture.locals[lane];
            const auto slot = fixture.admissions[lane].gm_mix.slot_index;
            ASSERT_EQ(local->slots[slot].state, SchedulerDispatchSlotState::READY);
            bool permit = true;
            EXPECT_FALSE(scheduler_ordered_can_enter_slot(
                fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, &permit
            ));
            EXPECT_FALSE(permit);
            EXPECT_EQ(local->slots[slot].state, SchedulerDispatchSlotState::READY);
        }
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    }
}

TEST(HbgGmMix, InvalidDiscoveredFaninRejectsNewAdmissionAndPreviouslyPermittedKernelEntry) {
    GmMixFixture fixture(
        {{2, LaneMask::all(), true},
         {1, LaneMask::aic()},
         {2, LaneMask::all(), true},
         {2, LaneMask::all(), true},
         {2, LaneMask::all(), true}}
    );
    for (uint32_t task = 2; task <= 4; ++task) {
        fixture.graph.executable(task, 0, {1});
    }
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto slot = fixture.admissions[lane].gm_mix.slot_index;
        bool permit = false;
        ASSERT_TRUE(scheduler_ordered_can_enter_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[lane], fixture.run(), slot, &permit
        ));
        ASSERT_TRUE(permit);
    }

    auto *resolver = fixture.locals[3];
    for (uint32_t task : {4u, 3u, 2u})
        EXPECT_FALSE(fixture.task_done(task));
    scheduler_gm_store(
        fixture.control(1)->completion_state,
        scheduler_completion_completed_state(scheduler_completion_total(fixture.control(1)->completion_state))
    );

    ASSERT_TRUE(fixture.claim_pending(2, 3));
    ASSERT_EQ(fixture.admissions[3].pending_task_ref, 2);
    fixture.graph.executable(2, 0, {1, 2});
    ASSERT_EQ(fixture.graph.payload(2).fanin_count, 2);
    auto *bad_observer = fixture.locals[0];
    bool discovered = false;
    bool healthy = true;
    for (uint32_t slice = 0; slice < 10 && healthy; ++slice) {
        healthy = scheduler_discover_cooperative_task(
            fixture.graph.graph(), fixture.base(), bad_observer, fixture.run(), &discovered
        );
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
    }
    ASSERT_FALSE(healthy);
    EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_EQ(scheduler_read_error(fixture.run()), static_cast<uint64_t>(SchedulerGraphResult::INVALID_FANIN_ID));
    ASSERT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ROUTE_INVALID_STATE));
    const uint64_t first_error = scheduler_read_error(fixture.run());
    const uint64_t first_site = fixture.run()->error_site;
    const uint64_t first_task = fixture.run()->error_task_ref;

    // A staged candidate cannot publish after a different observer latches a failure.
    EXPECT_EQ(
        scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), resolver, fixture.run()),
        SchedulerAdmissionResult::ERROR
    );
    EXPECT_EQ(fixture.admissions[3].pending_task_ref, 2);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(2)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
    );
    const auto *shared =
        scheduler_state_at<SchedulerAdmissionControl>(fixture.base(), fixture.storage.layout.admission_control_offset);
    EXPECT_EQ(shared->commit_tail, 1u);
    EXPECT_EQ(shared->commit_lock, 0u);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto slot = fixture.admissions[lane].gm_mix.slot_index;
        bool permit = true;
        EXPECT_FALSE(scheduler_ordered_can_enter_slot(
            fixture.graph.graph(), fixture.base(), fixture.locals[lane], fixture.run(), slot, &permit
        ));
        EXPECT_FALSE(permit);
        EXPECT_EQ(fixture.locals[lane]->slots[slot].state, SchedulerDispatchSlotState::READY);
    }
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), first_error);
    EXPECT_EQ(fixture.run()->error_site, first_site);
    EXPECT_EQ(fixture.run()->error_task_ref, first_task);
}

TEST(HbgGmMix, PublishedRunErrorDoesNotTurnReleasedLanesIntoSuccessfulBlockCompletion) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask, true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
        EXPECT_EQ(fixture.frontier(0), 0u);
    }
}

TEST(HbgGmMix, FatalRunStillRetiresRealReturnsWithoutCompletingTheTask) {
    for (bool sync_start : {false, true}) {
        for (bool seal_failed : {false, true}) {
            for (LaneMask mask : kMixMasks) {
                SCOPED_TRACE(lane_names(mask));
                SCOPED_TRACE(sync_start);
                SCOPED_TRACE(seal_failed);
                GmMixFixture fixture({{2, mask, sync_start}});
                ASSERT_TRUE(fixture.prepare_calls());
                if (sync_start) {
                    ASSERT_TRUE(fixture.admit(0));
                    ASSERT_TRUE(fixture.refresh_all());
                    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
                    ASSERT_TRUE(fixture.publish(0));
                    ASSERT_TRUE(fixture.rendezvous(0));
                } else {
                    bool armed = false;
                    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
                    ASSERT_TRUE(armed);
                    for (uint32_t pass = 0; pass < 2; ++pass)
                        for (uint32_t worker = 0; worker < 3; ++worker) {
                            bool progress = false;
                            ASSERT_TRUE(fixture.ordinary_service(worker, &progress));
                        }
                }
                std::array<GmExecution, 3> executions{};
                std::array<uint64_t, 3> indices{};
                for (uint32_t lane = 0; lane < 3; ++lane) {
                    if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
                    ASSERT_TRUE(fixture.begin(lane, &executions[lane]));
                    indices[lane] = fixture.locals[lane]->slots[executions[lane].slot].prepared.execution_index;
                }
                const uint64_t completion_before = fixture.control(0)->completion_state;
                scheduler_record_error(
                    fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
                    SchedulerErrorSite::ASYNC_SEAL_FAILED
                );
                for (uint32_t lane = 0; lane < 3; ++lane) {
                    if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
                    auto *local = fixture.locals[lane];
                    const auto &execution = executions[lane];
                    if (seal_failed) {
                        EXPECT_TRUE((scheduler_complete_own_slot<false, true, false>(
                            fixture.graph.graph(), fixture.base(), local, fixture.run(), execution.slot, 0
                        )));
                    } else {
                        EXPECT_FALSE((scheduler_complete_own_slot<true, true>(
                            fixture.graph.graph(), fixture.base(), local, fixture.run(), execution.slot, 0
                        )));
                    }
                    EXPECT_EQ(local->slots[execution.slot].state, SchedulerDispatchSlotState::FREE);
                    EXPECT_EQ(
                        scheduler_prepared_phase(
                            scheduler_prepared_control_at(fixture.base(), local->config.prepared, indices[lane])->state
                        ),
                        SchedulerPreparedPhase::RETURNED
                    );
                    if (sync_start)
                        EXPECT_EQ(
                            fixture.mailbox(0)->lanes[lane].released_seq,
                            lane == 0 ? 0u : SCHEDULER_GROUP_SEQUENCE_LIMIT | 1u
                        );
                    else EXPECT_EQ(fixture.signal(0)->finished[lane], fixture.signal(0)->armed_generation + 1);
                    EXPECT_EQ(local->pending_completed, 0u);
                }
                EXPECT_EQ(fixture.control(0)->completion_state, completion_before);
                EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
                EXPECT_FALSE(fixture.task_done(0));
                EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::ASYNC_SEAL_FAILED));
            }
        }
    }
}

TEST(HbgGmMix, CancelUnstartedTicketsAndOwnedCallsNeverContributesOrReturnsThem) {
    for (auto phase :
         {SchedulerPreparedPhase::BUILT, SchedulerPreparedPhase::CLAIMED, SchedulerPreparedPhase::FAILED}) {
        GmMixFixture fixture({{1, LaneMask::aic()}});
        ASSERT_TRUE(fixture.prepare_calls());
        auto *local = fixture.locals[0];
        if (phase == SchedulerPreparedPhase::BUILT) {
            auto &token = local->execution_tokens[0];
            token.task_ref = 0;
            token.block_idx = scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(1));
            token.subtask_slot = 0;
            token.phase = SchedulerExecutionTokenPhase::TICKETED;
            ASSERT_TRUE(scheduler_prepared_execution_index(
                fixture.base(), local->config.prepared, 0, token.block_idx, 0, &token.prepared.execution_index
            ));
        } else {
            ASSERT_TRUE(fixture.claim_ordinary(0, 0));
        }
        auto &token = local->execution_tokens[0];
        const uint64_t index = token.prepared.execution_index;
        if (phase == SchedulerPreparedPhase::FAILED) {
            auto *call = scheduler_prepared_call_at(fixture.base(), local->config.prepared, index);
            ++call->run_epoch;
            ASSERT_FALSE(
                scheduler_receive_execution_token(fixture.base(), local->config.prepared, 0, 0, 0, &token.prepared)
            );
        }
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, local, SchedulerErrorSite::EXIT_WAIT_TIMEOUT
        );
        EXPECT_TRUE(scheduler_cancel_unstarted_work(fixture.base(), local));
        EXPECT_TRUE(scheduler_cancel_unstarted_work(fixture.base(), local));
        EXPECT_FALSE(scheduler_has_execution_tokens(local));
        EXPECT_EQ(
            scheduler_prepared_phase(
                scheduler_prepared_control_at(fixture.base(), local->config.prepared, index)->state
            ),
            SchedulerPreparedPhase::FAILED
        );
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
        EXPECT_EQ(fixture.control(0)->next_block, 1u);
        EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
    }
}

TEST(HbgClusterMix, CancelFutureReservationDoesNotAcknowledgeAnArmedGroup) {
    GmMixFixture fixture({{1, kAicAiv0}, {1, kAicAiv0}});
    ASSERT_TRUE(fixture.prepare_calls());
    auto *local = fixture.locals[0];
    bool armed = false;
    ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
    ASSERT_TRUE(armed);
    const uint32_t generation = fixture.signal(0)->armed_generation;
    const uint64_t armed_anchor = fixture.signal(0)->anchor_state;
    bool progress = false;
    ASSERT_TRUE(
        scheduler_cluster_mix_discover_task(fixture.graph.graph(), fixture.base(), local, fixture.run(), 1, &progress)
    );
    const uint64_t future_index = local->cluster_pending.anchor.execution_index;
    ASSERT_EQ(local->cluster_pending.task_ref, 1);
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, local, SchedulerErrorSite::EXIT_WAIT_TIMEOUT
    );
    EXPECT_TRUE(scheduler_cancel_unstarted_work(fixture.base(), local));
    EXPECT_EQ(local->cluster_pending.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(
        scheduler_prepared_phase(
            scheduler_prepared_control_at(fixture.base(), local->config.prepared, future_index)->state
        ),
        SchedulerPreparedPhase::FAILED
    );
    EXPECT_EQ(fixture.signal(0)->prepare_mask, 0u);
    EXPECT_EQ(fixture.signal(0)->armed_generation, generation);
    EXPECT_EQ(fixture.signal(0)->anchor_state, armed_anchor);
    for (uint32_t lane = 0; lane < 3; ++lane)
        EXPECT_EQ(fixture.signal(0)->finished[lane], 0u);
    EXPECT_EQ(fixture.run()->resolved_task_count, 0u);
}

TEST(HbgGmMix, PublishedRunErrorDoesNotSealALocallyCompletedSyncTask) {
    for (LaneMask mask : kMixMasks) {
        SCOPED_TRACE(lane_names(mask));
        GmMixFixture fixture({{2, mask, true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        ASSERT_TRUE(fixture.rendezvous(0));
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        scheduler_record_error(
            fixture.run(), 0, SchedulerGraphResult::TIMEOUT, nullptr, fixture.locals[4],
            SchedulerErrorSite::ORDERED_EXIT_TIMEOUT
        );
        bool closed = true;
        EXPECT_FALSE(fixture.close(0, &closed));
        EXPECT_FALSE(closed);
        EXPECT_EQ(fixture.frontier(0), 0u);
        EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
        EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), mask.count());
    }
}

TEST_P(HbgGmMixSyncStart, LocalFrontierIsIndependentButNextTaskWaitsForAllClaims) {
    const LaneMask mask = GetParam();
    GmMixFixture fixture({{2, mask, true}, {2, mask, true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    bool closed = true;
    ASSERT_TRUE(fixture.close(0, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 0u);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::RETRY);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 1u);
    EXPECT_FALSE(fixture.admit(1));
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.admit(1));
    // The closed prefix remains valid after the next task replaces the mailbox.
    ASSERT_TRUE(fixture.refresh(0));
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 2u);
    for (uint32_t worker : {1u, 2u}) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.close(worker, &closed));
        ASSERT_TRUE(closed);
        ASSERT_TRUE(fixture.refresh(worker));
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        mask.count()
    );
    ASSERT_TRUE(fixture.publish(1));
    EXPECT_EQ(fixture.mailbox(1)->descriptor.task_ref, 0);
    EXPECT_EQ(fixture.mailbox(1)->descriptor.block_idx, 1u);
    ASSERT_TRUE(fixture.rendezvous(1));
    ASSERT_TRUE(fixture.finish_lanes(1));
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST_P(HbgGmMixSyncStart, ZeroWorkClusterClosesOnlyAfterExhaustionWhileLastPublisherStillOwesItsGroup) {
    GmMixFixture fixture({{2, GetParam(), true}}, false, 3);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    bool closed = true;
    ASSERT_TRUE(fixture.close(6, &closed));
    EXPECT_FALSE(closed);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.close(6, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(2)), 1u);
    for (uint32_t worker : {7u, 8u}) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
    }
    for (uint32_t worker : {0u, 1u, 2u, 3u, 4u, 5u}) {
        if (!fixture.participant(worker)) continue;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_FALSE(closed);
    }
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmMixSyncStart, ::testing::ValuesIn(kMixMasks));

TEST(HbgGmMix, ClaimedLastBlockWithoutDescriptorCannotLetItsVectorMembersClose) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::CLAIMED);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 0u);
    for (uint32_t worker : {1u, 2u}) {
        bool progress = true;
        ASSERT_TRUE(fixture.observe(worker, &progress));
        EXPECT_FALSE(progress);
        bool closed = true;
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_FALSE(closed);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, 0);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
    }
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 0u);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 1u);
}

class HbgGmMixInactiveLane : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixInactiveLane, OnlyDeclaredResourceLanesArriveAndInactiveCoordinatorNeverRunsAKernel) {
    GmMixFixture fixture({{2, GetParam(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_EQ(fixture.admissions[0].current_resource_mask, GetParam().merged_with(LaneMask::aic()));
    uint32_t inactive = UINT32_MAX;
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (!GetParam().has(static_cast<SubtaskSlot>(lane))) inactive = lane;
    ASSERT_LT(inactive, 3u);
    if (!fixture.participant(inactive)) {
        EXPECT_EQ(fixture.admissions[inactive].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        ASSERT_TRUE(fixture.rendezvous(0));
        EXPECT_EQ(fixture.mailbox(0)->lanes[inactive].arrived_seq, 0u);
        EXPECT_EQ(fixture.mailbox(0)->lanes[inactive].released_seq, 0u);
        ASSERT_TRUE(fixture.finish_lanes(0));
        ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), GetParam().count());
        return;
    }
    bool progress = false;
    for (uint32_t lane = 1; lane < 3; ++lane)
        ASSERT_TRUE(fixture.observe(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (lane != inactive) ASSERT_TRUE(fixture.arrive(lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (lane == inactive) continue;
        ASSERT_TRUE(fixture.prepare(lane, &progress));
        EXPECT_EQ(fixture.locals[lane]->local_ready_mask, 0u);
    }
    ASSERT_TRUE(fixture.arrive(inactive, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        ASSERT_TRUE(fixture.prepare(lane, &progress));
    EXPECT_EQ(fixture.admissions[inactive].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(fixture.locals[inactive]->local_ready_mask, 0u);
    EXPECT_EQ(fixture.mailbox(0)->lanes[inactive].arrived_seq, inactive == 0 ? 0u : 1u);
    EXPECT_NE(fixture.admissions[inactive].gm_mix.arrivals_confirmed & (1U << inactive), 0u);
    EXPECT_EQ(fixture.mailbox(0)->lanes[inactive].released_seq, inactive == 0 ? 0u : 1u);
    EXPECT_NE(fixture.admissions[inactive].gm_mix.releases_confirmed & (1U << inactive), 0u);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        GmExecution execution{};
        if (lane == inactive) {
            EXPECT_FALSE(fixture.begin(lane, &execution));
            continue;
        }
        ASSERT_TRUE(fixture.begin(lane, &execution));
        EXPECT_EQ(execution.subtask, lane);
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))),
        GetParam().count()
    );
}

INSTANTIATE_TEST_SUITE_P(
    MasksWithInactiveResourceMember, HbgGmMixInactiveLane, ::testing::Values(kAicAiv0, kAicAiv1, LaneMask::both_aiv())
);

TEST(HbgGmMix, KernelReturnWithoutCompletionReleaseDoesNotRetireTheGroup) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    std::array<GmExecution, 3> executions{};
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.begin(worker, &executions[worker]));
    ASSERT_TRUE(fixture.complete(executions[0]));
    ASSERT_TRUE(fixture.complete(executions[1]));
    const auto &late = executions[2];
    EXPECT_EQ(fixture.locals[2]->slots[late.slot].state, SchedulerDispatchSlotState::EXECUTING);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[2].released_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    ASSERT_TRUE(fixture.complete(late));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    EXPECT_FALSE(fixture.task_done(0));
    ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 1));
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(fixture.locals[3]->pending_completed, 1u);
    EXPECT_EQ(scheduler_cached_task_proof(fixture.locals[3], 0), SchedulerTaskProof::DONE);
}

TEST(HbgGmMixProtocol, ZeroWorkMembersRecordTheActualFrontierWithoutInventingCursorReads) {
    GmMixFixture fixture({{2, LaneMask::both_aiv(), true}}, true, 3);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    bool closed = false;
    ASSERT_TRUE(fixture.close(7, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(members[7].closed.sequence, 0u);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    for (uint32_t worker = 6; worker < 9; ++worker) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(members[worker].observed.sequence, 1u);
        EXPECT_EQ(members[worker].closed.sequence, 2u);
        EXPECT_EQ(members[worker].close_frontier, 1u);
        EXPECT_EQ(members[worker].last_group_seq, 0u);
        EXPECT_EQ(members[worker].closed_cursor, worker == 6 ? UINT64_C(2) : UINT64_MAX);
        EXPECT_EQ(members[worker].pending_mask, 0u);
        EXPECT_EQ(members[worker].worker_index, worker);
    }
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    EXPECT_EQ(members[0].closed.sequence, 0u);
    EXPECT_EQ(members[1].closed.sequence, 0u);
}

TEST(HbgGmMix, AZeroWorkClusterClosesWhileAnotherClusterStillOwnsTheLastBlock) {
    GmMixFixture fixture({{2, LaneMask::all(), true}}, false, 3);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_EQ(fixture.claim(2), SchedulerBlockClaimResult::EXHAUSTED);
    bool closed = false;
    ASSERT_TRUE(fixture.close(6, &closed));
    EXPECT_TRUE(closed);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(2)), 1u);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 0u);
    for (uint32_t worker : {7u, 8u}) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.locals[worker]->local_ready_mask, 0u);
    }
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
}

TEST(HbgGmMix, FastClusterPublishesBWhileRemoteAIsRunningAndLateVectorConsumesTheClosedPrefix) {
    GmMixFixture fixture({{2, LaneMask::all(), true}, {2, kAicAiv0, true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        ASSERT_TRUE(fixture.rendezvous(cluster));
    }
    ASSERT_TRUE(fixture.admit(1, 3));
    GmExecution remote{};
    ASSERT_TRUE(fixture.begin(5, &remote));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 1u);
    ASSERT_TRUE(fixture.refresh(0));
    ASSERT_EQ(fixture.admissions[0].current_task_ref, 1);
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.task_ref, 1);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 2u);
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::BLOCKED)
    );
    EXPECT_EQ(fixture.admissions[5].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[5].gm_mix.phase, SchedulerGmMixPhase::RUNNING);
    for (uint32_t worker : {1u, 2u}) {
        bool progress = false;
        ASSERT_EQ(fixture.admissions[worker].current_task_ref, 0);
        if (fixture.participant(worker, 1)) ASSERT_TRUE(fixture.observe(worker, &progress));
        else {
            EXPECT_EQ(fixture.mailbox(0)->lanes[worker].assigned_seq, 1u);
            ASSERT_TRUE(fixture.close(worker, &progress));
        }
        EXPECT_TRUE(progress);
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 1u);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.task_last_seq, 0u);
        ASSERT_TRUE(fixture.refresh(worker));
        if (!fixture.participant(worker, 1)) {
            EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
            EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 1u);
            continue;
        }
        ASSERT_EQ(fixture.admissions[worker].current_task_ref, 1);
        ASSERT_TRUE(fixture.observe(worker, &progress));
        EXPECT_TRUE(progress);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, 2u);
    }
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_FALSE(fixture.task_done(1));
    EXPECT_EQ(scheduler_completion_count(fixture.control(1)->completion_state), 2u);
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::BLOCKED)
    );
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 1u);
    for (uint32_t worker : {3u, 4u}) {
        GmExecution remaining{};
        ASSERT_TRUE(fixture.begin(worker, &remaining));
        ASSERT_TRUE(fixture.complete(remaining));
    }
    ASSERT_TRUE(fixture.complete(remote));
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
    for (uint32_t worker : {3u, 4u, 5u}) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        ASSERT_TRUE(closed);
        ASSERT_TRUE(fixture.refresh(worker));
    }
    ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 1));
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_TRUE(fixture.task_done(1));
}

TEST(HbgGmMix, RetiredSlotCannotReleaseOrContributeTwice) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution first{};
    ASSERT_TRUE(fixture.begin(1, &first));
    ASSERT_TRUE(fixture.complete(first));
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].released_seq, UINT64_C(0)), 1u);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    for (uint32_t worker : {0u, 2u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 3u
    );
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    EXPECT_FALSE(scheduler_complete_own_slot(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), first.slot, 0
    ));
    EXPECT_EQ(fixture.mailbox(0)->lanes[1].released_seq, 1u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 3u);
}

TEST(HbgGmMix, ReleaseBeforeRealSlotRetirementFailsWithoutPublishingAnAcknowledgement) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution execution{};
    ASSERT_TRUE(fixture.begin(1, &execution));
    EXPECT_FALSE(scheduler_gm_mix_release_lane(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(),
        fixture.locals[1]->slots[execution.slot], execution.slot
    ));
    EXPECT_EQ(fixture.locals[1]->slots[execution.slot].state, SchedulerDispatchSlotState::EXECUTING);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].released_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
}

class HbgGmMixHighSequence : public ::testing::TestWithParam<uint64_t> {};

TEST_P(HbgGmMixHighSequence, GroupSequenceIsIndependentOfTheNarrowOwnSlotGeneration) {
    const uint64_t previous = GetParam();
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());

    for (uint32_t worker = 0; worker < 3; ++worker) {
        fixture.admissions[worker].gm_mix.group_seq = previous;
        scheduler_gm_store(fixture.mailbox(0)->lanes[worker].arrived_seq, previous);
        scheduler_gm_store(fixture.mailbox(0)->lanes[worker].released_seq, previous);
    }
    for (auto &sequence : fixture.admissions[0].gm_mix.vector_last_published_seq)
        sequence = previous;
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, previous + 1);
    for (uint32_t worker = 0; worker < 3; ++worker) {
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, previous + 1);
        EXPECT_LT(fixture.admissions[worker].gm_mix.slot_index, SCHEDULER_PENDING_SLOT_COUNT);
    }
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_FALSE(fixture.task_done(0));
    ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 1));
    ASSERT_TRUE(fixture.task_done(0));
    bool closed = false;
    for (uint32_t worker = 0; worker < 3; ++worker) {
        ASSERT_TRUE(fixture.close(worker, &closed));
        EXPECT_TRUE(closed);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.group_seq, previous + 1);
        EXPECT_EQ(fixture.admissions[worker].gm_mix.task_last_seq, 0u);
    }
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 1u);
}

INSTANTIATE_TEST_SUITE_P(
    WideCounterBoundaries, HbgGmMixHighSequence,
    ::testing::Values(UINT64_C(1) << 32, SCHEDULER_GROUP_SEQUENCE_LIMIT - 2)
);

TEST(HbgGmMix, SequenceExhaustionFailsBeforeClaimingAnUnpublishableBlock) {
    for (uint64_t sequence : {SCHEDULER_GROUP_SEQUENCE_LIMIT - 1, SCHEDULER_GROUP_SEQUENCE_LIMIT}) {
        GmMixFixture fixture({{2, LaneMask::all(), true}});
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        fixture.admissions[0].gm_mix.group_seq = sequence;
        EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::ERROR);
        EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 0u);
        EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 0u);
        EXPECT_EQ(fixture.admissions[0].gm_mix.group_seq, sequence);
        EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    }
}

enum class DescriptorFault {
    EPOCH,
    TASK,
    ORDER,
    SEQUENCE,
    BLOCK,
    CLUSTER,
    SKIPPED_SEQUENCE,
    OWNER_PREDECESSOR,
    VECTOR_PREDECESSOR,
    FUTURE_PREDECESSOR,
};

class HbgGmMixDescriptorFault : public ::testing::TestWithParam<DescriptorFault> {};

TEST_P(HbgGmMixDescriptorFault, CorruptPublishedDescriptorCannotAcquireAVectorLane) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    auto &descriptor = fixture.mailbox(0)->descriptor;
    switch (GetParam()) {
    case DescriptorFault::EPOCH:
        ++descriptor.run_epoch;
        break;
    case DescriptorFault::TASK:
        descriptor.task_ref = SCHEDULER_TASK_REF_INVALID;
        break;
    case DescriptorFault::ORDER:
        ++descriptor.order_key;
        break;
    case DescriptorFault::SEQUENCE:
        ++descriptor.group_seq;
        break;
    case DescriptorFault::BLOCK:
        descriptor.block_idx = fixture.admissions[1].current_block_num;
        break;
    case DescriptorFault::CLUSTER:
        ++descriptor.cluster_index;
        break;
    case DescriptorFault::SKIPPED_SEQUENCE:
        descriptor.group_seq = 3;
        descriptor.previous_lane_seq[0] = 2;
        descriptor.previous_lane_seq[1] = 1;

        scheduler_gm_store(fixture.mailbox(0)->lanes[1].assigned_seq, UINT64_C(3));
        break;
    case DescriptorFault::OWNER_PREDECESSOR:
        ++descriptor.previous_lane_seq[0];
        break;
    case DescriptorFault::VECTOR_PREDECESSOR:
        ++descriptor.previous_lane_seq[1];
        break;
    case DescriptorFault::FUTURE_PREDECESSOR:
        descriptor.previous_lane_seq[2] = UINT64_MAX;
        break;
    }
    bool progress = true;
    EXPECT_FALSE(fixture.observe(1, &progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(fixture.admissions[1].gm_mix.group_seq, 0u);
    EXPECT_EQ(fixture.admissions[1].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.mailbox(0)->lanes[1].arrived_seq, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.locals[1]->local_ready_mask, 0u);
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    IdentityAndPlacement, HbgGmMixDescriptorFault,
    ::testing::Values(
        DescriptorFault::EPOCH, DescriptorFault::TASK, DescriptorFault::ORDER, DescriptorFault::SEQUENCE,
        DescriptorFault::BLOCK, DescriptorFault::CLUSTER, DescriptorFault::SKIPPED_SEQUENCE,
        DescriptorFault::OWNER_PREDECESSOR, DescriptorFault::VECTOR_PREDECESSOR, DescriptorFault::FUTURE_PREDECESSOR
    )
);

struct AckPrefixCase {
    bool releases;
    uint64_t sequence;
    std::array<int8_t, 3> offsets;
};

std::vector<AckPrefixCase> ack_prefix_cases() {
    std::vector<AckPrefixCase> result;
    for (bool releases : {false, true})
        for (uint64_t sequence : {UINT64_C(1), (UINT64_C(1) << 32) + 9, SCHEDULER_GROUP_SEQUENCE_LIMIT - 2})
            for (int8_t first : {-1, 0, 1})
                for (int8_t second : {-1, 0, 1})
                    for (int8_t third : {-1, 0, 1})
                        result.push_back({releases, sequence, {first, second, third}});
    return result;
}

class HbgGmAckPrefix : public ::testing::TestWithParam<AckPrefixCase> {};

TEST_P(HbgGmAckPrefix, WaitsOnOldPrefixAndRejectsFuturePrefixBeforePublishingEvidence) {
    const auto scenario = GetParam();
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    fixture.admissions[0].gm_mix.group_seq = scenario.sequence;
    int8_t first_difference = 0;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const int8_t delta = scenario.offsets[lane];
        if (first_difference == 0 && delta != 0) first_difference = delta;
        const uint64_t value = delta < 0 ? scenario.sequence - 1 : scenario.sequence + delta;
        auto &ack = scenario.releases ? fixture.mailbox(0)->lanes[lane].released_seq :
                                        fixture.mailbox(0)->lanes[lane].arrived_seq;
        scheduler_gm_store(ack, value);
    }
    bool all = true;
    const bool valid = scheduler_gm_mix_check_acks(
        fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), scenario.releases, &all
    );
    EXPECT_EQ(valid, first_difference <= 0);
    EXPECT_EQ(all, first_difference == 0);
    EXPECT_EQ(scheduler_read_error(fixture.run()) != 0, first_difference > 0);
    if (first_difference > 0)
        EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_ACK_INVALID));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
}

INSTANTIATE_TEST_SUITE_P(OldCurrentFutureAndFullWidth, HbgGmAckPrefix, ::testing::ValuesIn(ack_prefix_cases()));

class HbgGmMixPollAckHint : public ::testing::TestWithParam<std::tuple<LaneMask, uint32_t>> {};

TEST_P(HbgGmMixPollAckHint, ArrivalObservationRejectsUnconfirmedFutureAcknowledgement) {
    const auto [mask, worker] = GetParam();
    const uint32_t cluster = worker / 3;
    GmMixFixture fixture({{2, mask, true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_admitted_task_owned(fixture));
    ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(cluster));
    bool progress = false;
    for (uint32_t lane = 1; lane < 3; ++lane)
        if (fixture.participant(lane)) ASSERT_TRUE(fixture.observe(cluster * 3 + lane, &progress));
    for (uint32_t lane = 0; lane < 3; ++lane)
        if (fixture.participant(lane)) ASSERT_TRUE(fixture.arrive(cluster * 3 + lane, &progress));
    auto *local = fixture.locals[worker];
    if (!fixture.participant(worker)) {
        EXPECT_FALSE(
            (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
             SchedulerOrderedPollResult::READY)
        );
        EXPECT_EQ(fixture.admissions[worker].current_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(local->local_ready_mask, 0u);
        return;
    }
    if (!local->is_mix_coordinator()) {
        EXPECT_FALSE(
            (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
             SchedulerOrderedPollResult::READY)
        );
        bool all = false;
        ASSERT_TRUE(scheduler_gm_mix_arrivals_ready(
            fixture.graph.graph(), fixture.base(), fixture.locals[cluster * 3], fixture.run(), &all
        ));
        ASSERT_TRUE(all);
    }
    const uint32_t checked_lane = mask.has_aiv1() ? 2 : 1;
    auto &gate = local->is_mix_coordinator() ? fixture.mailbox(cluster)->lanes[checked_lane].arrived_seq :
                                               fixture.signal(cluster)->gm_arrival_seq;
    gate = 0;
    EXPECT_EQ(
        scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()),
        SchedulerOrderedPollResult::WAIT
    );
    gate = 2;
    if (local->is_mix_coordinator()) {
        bool failed = false;
        EXPECT_FALSE(
            scheduler_fill_ordered_slots(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed)
        );
        EXPECT_TRUE(failed);
    } else {
        EXPECT_EQ(
            scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()),
            SchedulerOrderedPollResult::READY
        );
        EXPECT_FALSE(
            scheduler_gm_mix_prepare_lane(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress)
        );
    }
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(local->local_ready_mask, 0u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_ACK_INVALID));
}

TEST_P(HbgGmMixPollAckHint, ReleaseHintOnlyBelongsToCoordinatorAndStillRejectsLateError) {
    const auto [mask, worker] = GetParam();
    const uint32_t cluster = worker / 3;
    GmMixFixture fixture({{2, mask, true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_NO_FATAL_FAILURE(expect_admitted_task_owned(fixture));
    ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(cluster));
    ASSERT_TRUE(fixture.rendezvous(cluster));
    ASSERT_TRUE(fixture.finish_lanes(cluster));
    auto *local = fixture.locals[worker];
    EXPECT_EQ(
        (scheduler_observe_ordered_progress(fixture.graph.graph(), fixture.base(), local, fixture.run()) ==
         SchedulerOrderedPollResult::READY),
        worker % 3 == 0
    );
    scheduler_publish_error(fixture.run(), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    bool progress = false;
    EXPECT_FALSE(scheduler_service_gm_mix(fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &progress));
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
    EXPECT_EQ(
        fixture.admissions[worker].current_task_ref, fixture.participant(worker) ? 0 : SCHEDULER_TASK_REF_INVALID
    );
}

INSTANTIATE_TEST_SUITE_P(
    MasksAndWorkers, HbgGmMixPollAckHint,
    ::testing::Combine(::testing::ValuesIn(kMixMasks), ::testing::Values(0u, 1u, 2u, 3u, 4u, 5u))
);

TEST(HbgGmMix, FutureArrivedAcknowledgementIsRejectedInsteadOfSatisfyingTheBarrier) {
    GmMixFixture fixture({{2, LaneMask::all(), true}}, true);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    for (uint32_t worker : {1u, 2u})
        ASSERT_TRUE(fixture.observe(worker, &progress));
    for (uint32_t worker = 0; worker < 3; ++worker)
        ASSERT_TRUE(fixture.arrive(worker, &progress));
    scheduler_gm_store(fixture.mailbox(0)->lanes[2].arrived_seq, UINT64_C(2));
    EXPECT_FALSE(fixture.prepare(0, &progress));
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->lanes[0].arrivals_observed.sequence, 0u);
    EXPECT_EQ(record->owner.completion_before.sequence, 0u);
}

TEST(HbgGmMix, FutureReleasedAcknowledgementIsRejectedBeforeResourceRetirement) {
    GmMixFixture fixture({{2, LaneMask::all(), true}}, true);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    scheduler_gm_store(fixture.mailbox(0)->lanes[2].released_seq, UINT64_C(2));
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->owner.releases_observed.sequence, 0u);
    EXPECT_EQ(record->owner.completion_before.sequence, 0u);
    EXPECT_NE(record->lanes[2].release_after.sequence, 0u);
}

TEST(HbgGmMix, InvalidCoordinatorFrontierCannotRetireALateVector) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    bool closed = false;
    ASSERT_TRUE(fixture.close(0, &closed));
    ASSERT_TRUE(closed);
    scheduler_cluster_store_field(&fixture.frontier(0), UINT32_MAX);
    EXPECT_FALSE(fixture.close(1, &closed));
    EXPECT_FALSE(closed);
    EXPECT_EQ(fixture.admissions[1].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[1].scan_cursor, 0u);
}

TEST(HbgGmMix, ExitDrainNeverClaimsOrPreparesUnexecutedWork) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    bool done = true;
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 0u);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 0u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::IDLE);
    EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
}

TEST(HbgGmMix, ExitDrainConsumesClosedFrontiersWithoutPublishingAnotherGroup) {
    GmMixFixture fixture({{2, LaneMask::all(), true}}, false, 3);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 1));
    ASSERT_TRUE(fixture.task_done(0));
    bool done = true;
    ASSERT_TRUE(fixture.drain(1, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 0u);
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
        done = false;
        for (uint32_t step = 0; step < 3 && !done; ++step)
            ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
    }
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 1u);
    EXPECT_EQ(fixture.mailbox(1)->descriptor.group_seq, 1u);
    EXPECT_EQ(fixture.mailbox(2)->descriptor.group_seq, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 2u);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 1u);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(1)), 1u);
    EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(2)), 1u);
}

TEST(HbgGmMix, ExitDrainPublishesOutstandingGroupBeforeLogicalCompletion) {
    for (bool protocol : {false, true}) {
        SCOPED_TRACE(protocol);
        GmMixFixture fixture({{2, LaneMask::both_aiv(), true}}, protocol, 2);
        ASSERT_TRUE(fixture.prepare_calls());
        ASSERT_TRUE(fixture.admit(0));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(0));
        bool progress = false;
        ASSERT_TRUE(fixture.arrive(0, &progress));
        ASSERT_TRUE(fixture.prepare(0, &progress));
        ASSERT_FALSE(progress);
        ASSERT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
        ASSERT_EQ(fixture.admissions[0].gm_mix.subtask_slot, UINT8_MAX);
        for (uint32_t worker : {1u, 2u}) {
            ASSERT_TRUE(fixture.observe(worker, &progress));
            ASSERT_TRUE(fixture.arrive(worker, &progress));
        }
        bool all = false;
        ASSERT_TRUE(scheduler_gm_mix_arrivals_ready(
            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run(), &all
        ));
        ASSERT_TRUE(all);
        EXPECT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 0u);
        for (uint32_t worker : {1u, 2u}) {
            ASSERT_TRUE(fixture.prepare(worker, &progress));
            GmExecution execution{};
            ASSERT_TRUE(fixture.begin(worker, &execution));
            ASSERT_TRUE(fixture.complete(execution));
            scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
        }
        ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 1));
        for (uint32_t worker : {3u, 4u, 5u})
            scheduler_flush_completions(fixture.run(), fixture.locals[worker]);
        ASSERT_FALSE(fixture.task_done(0));
        ASSERT_EQ(fixture.run()->resolved_task_count, 0u);
        ASSERT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 0u);

        bool done = true;
        ASSERT_TRUE(fixture.drain(0, &done));
        EXPECT_FALSE(done);
        EXPECT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 0u);
        for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
            done = false;
            for (uint32_t step = 0; step < 3 && !done; ++step)
                ASSERT_TRUE(fixture.drain(worker, &done));
            EXPECT_TRUE(done);
        }
        ASSERT_TRUE(fixture.task_done(0));
        scheduler_flush_completions(fixture.run(), fixture.locals[0]);
        EXPECT_EQ(fixture.run()->resolved_task_count, 1u);
        EXPECT_EQ(scheduler_cluster_load_field(&fixture.frontier(0)), 1u);
        EXPECT_EQ(fixture.locals[0]->local_ready_mask, 0u);
        EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 4u);
        EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 1u);
        EXPECT_EQ(scheduler_gm_query(fixture.control(0)->next_block), 2u);
        EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    }
}

TEST(HbgGmMix, ExitDrainInactiveCoordinatorWaitsForArrivalsAndRejectsFutureAcks) {
    GmMixFixture fixture({{2, LaneMask::both_aiv(), true}}, true, 2);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    bool progress = false;
    ASSERT_TRUE(fixture.arrive(0, &progress));
    bool done = true;
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::ARRIVED);
    EXPECT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
    for (uint32_t worker : {1u, 2u}) {
        ASSERT_TRUE(fixture.observe(worker, &progress));
        ASSERT_TRUE(fixture.arrive(worker, &progress));
    }
    scheduler_gm_store(fixture.mailbox(0)->lanes[1].arrived_seq, UINT64_C(2));
    EXPECT_FALSE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(fixture.mailbox(0)->lanes[0].released_seq, 0u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::GROUP_ACK_INVALID));
}

TEST(HbgGmMix, ExitDrainWaitsForLateReleaseThenClosesTheObligation) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    GmExecution late{};
    ASSERT_TRUE(fixture.begin(2, &late));
    for (uint32_t worker : {0u, 1u}) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        ASSERT_TRUE(fixture.complete(execution));
    }
    ASSERT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    bool done = true;
    ASSERT_TRUE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(fixture.admissions[0].current_task_ref, 0);
    EXPECT_EQ(fixture.admissions[0].scan_cursor, 0u);
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, SchedulerGmMixPhase::RELEASED);
    EXPECT_EQ(fixture.mailbox(0)->lanes[2].released_seq, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);

    ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 1));
    ASSERT_TRUE(fixture.complete(late));
    for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
        done = false;
        for (uint32_t step = 0; step < 3 && !done; ++step)
            ASSERT_TRUE(fixture.drain(worker, &done));
        EXPECT_TRUE(done);
    }
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 6u);
    EXPECT_EQ(fixture.mailbox(0)->descriptor.group_seq, 1u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST(HbgGmMix, MixFaninWaitsForLastSubtaskAndCoordinatorDiscoversSuccessorOnce) {
    GmMixFixture fixture({{2, LaneMask::both_aiv(), true}, {2, kAicAiv0, true}});
    fixture.graph.executable(1, 0, {0});
    EXPECT_FALSE(fixture.task_done(1));
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0)->completion_state)),
        static_cast<int64_t>(SchedulerTaskState::BLOCKED)
    );
    EXPECT_FALSE(fixture.task_done(0));
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(1)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
    );
    EXPECT_EQ(fixture.locals[0]->pending_completed, 0u);
    ASSERT_EQ(fixture.claim(1), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(1));
    ASSERT_TRUE(fixture.rendezvous(1));
    ASSERT_TRUE(fixture.finish_lanes(1));
    ASSERT_EQ(fixture.retire(1), SchedulerGmMixResult::PROGRESS);
    ASSERT_TRUE(fixture.task_done(0));
    EXPECT_TRUE(fixture.task_done(0));
    EXPECT_EQ(fixture.locals[3]->pending_completed, 1u);
    EXPECT_EQ(
        scheduler_gm_query(fixture.control(1)->admission_state),
        static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
    );
    EXPECT_EQ(scheduler_gm_query(fixture.control(1)->next_block), 0u);
    uint32_t successor_executions = 0;
    for (uint32_t step = 0; step < 100; ++step) {
        for (uint32_t worker = 0; worker < 6; ++worker) {
            ASSERT_TRUE(fixture.fill(worker));
            GmExecution execution{};
            if (!fixture.begin(worker, &execution)) continue;
            EXPECT_EQ(execution.task, 1);
            EXPECT_LT(execution.block, 2u);
            EXPECT_EQ(execution.blocks, 2u);
            ++successor_executions;
            ASSERT_TRUE(fixture.complete(execution));
        }
        if (scheduler_completion_done(scheduler_gm_query(fixture.control(1)->completion_state))) break;
    }
    EXPECT_EQ(successor_executions, 4u);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 4u
    );
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(1)->completion_state, UINT64_C(0))), 4u
    );
    for (auto *local : fixture.locals)
        scheduler_flush_completions(fixture.run(), local);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 2u);
}

class HbgGmMixExitInFlight : public ::testing::TestWithParam<SchedulerGmMixPhase> {};

TEST_P(HbgGmMixExitInFlight, ExitDrainRejectsAnOutstandingGroupWithoutClaimingMoreWork) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    if (GetParam() != SchedulerGmMixPhase::CLAIMED) ASSERT_TRUE(fixture.publish(0));
    if (GetParam() == SchedulerGmMixPhase::ARRIVED || GetParam() == SchedulerGmMixPhase::RUNNING) {
        bool progress = false;
        for (uint32_t worker : {1u, 2u})
            ASSERT_TRUE(fixture.observe(worker, &progress));
        for (uint32_t worker = 0; worker < 3; ++worker)
            ASSERT_TRUE(fixture.arrive(worker, &progress));
        if (GetParam() == SchedulerGmMixPhase::RUNNING)
            for (uint32_t worker = 0; worker < 3; ++worker)
                ASSERT_TRUE(fixture.prepare(worker, &progress));
    }
    ASSERT_EQ(fixture.admissions[0].gm_mix.phase, GetParam());
    bool done = true;
    EXPECT_FALSE(fixture.drain(0, &done));
    EXPECT_FALSE(done);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.control(0)->next_block, UINT64_C(0)), 1u);
    EXPECT_EQ(
        scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(0)->completion_state, UINT64_C(0))), 0u
    );
    EXPECT_EQ(fixture.admissions[0].gm_mix.phase, GetParam());
}

INSTANTIATE_TEST_SUITE_P(
    UnretiredGroupPhases, HbgGmMixExitInFlight,
    ::testing::Values(
        SchedulerGmMixPhase::CLAIMED, SchedulerGmMixPhase::PUBLISHED, SchedulerGmMixPhase::ARRIVED,
        SchedulerGmMixPhase::RUNNING
    )
);

class HbgGmMixOwnPrepare : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmMixOwnPrepare, HasMixDoesNotRequireSsbufForAnOwnPayloadPreparation) {
    const LaneMask mask = GetParam();
    GmMixFixture fixture({{1, mask}, {1, LaneMask::all()}});
    ASSERT_TRUE(fixture.prepare_calls());
    for (uint8_t subtask = 0; subtask < 3; ++subtask) {
        if (!mask.has(static_cast<SubtaskSlot>(subtask))) continue;
        auto *local = fixture.locals[subtask];
        local->cluster_signal = nullptr;

        local->slots[0].state = SchedulerDispatchSlotState::FILLING;
        SchedulerReadyClaim ready{};
        ready.task_ref = 0;
        ASSERT_TRUE(scheduler_prepare_dispatch_slot(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), {0}, ready, fixture.storage.metadata[0], 0,
            subtask
        ));
        const auto index = local->slots[0].prepared.execution_index;
        const auto *call = scheduler_prepared_call_at(fixture.base(), local->config.prepared, index);
        const auto *binding = scheduler_execution_binding_at(fixture.base(), local->config.prepared, index);
        EXPECT_NE(call->function_bin_addr, 0u);
        EXPECT_EQ(binding->local_context.block_num, 1);
    }
}

INSTANTIATE_TEST_SUITE_P(
    PureAndMixedPayloads, HbgGmMixOwnPrepare,
    ::testing::Values(LaneMask::aic(), LaneMask::aiv0(), kAicAiv0, kAicAiv1, LaneMask::both_aiv(), LaneMask::all())
);

class HbgGmProtocolEvents : public ::testing::TestWithParam<LaneMask> {};

TEST_P(HbgGmProtocolEvents, CapturesPublicationDescriptorActualAcksAndUniqueBlockCompletion) {
    const LaneMask mask = GetParam();
    GmMixFixture fixture({{2, mask, true}}, true);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    auto &record = blocks[0];
    ASSERT_NE(record.owner.claim_after.sequence, 0u);
    EXPECT_EQ(record.owner.publish_before.sequence, 0u);
    EXPECT_EQ(record.lanes[1].descriptor_ready.sequence, 0u);
    bool progress = true;
    ASSERT_TRUE(fixture.observe(mask.has_aiv0() ? 1 : 2, &progress));
    EXPECT_FALSE(progress);
    ASSERT_TRUE(fixture.publish(0));
    EXPECT_GT(record.owner.publish_before.sequence, record.owner.claim_after.sequence);
    EXPECT_GT(record.owner.publish_after.sequence, record.owner.publish_before.sequence);
    EXPECT_EQ(record.owner.group_seq, 1u);
    EXPECT_EQ(record.owner.active_mask, mask);
    EXPECT_EQ(record.owner.logical_block_num, 2u);
    EXPECT_GT(record.lanes[0].descriptor_ready.sequence, record.owner.publish_after.sequence);
    EXPECT_EQ(record.lanes[1].descriptor_ready.sequence, 0u);
    ASSERT_TRUE(fixture.rendezvous(0));
    for (uint32_t lane = 0; lane < 3; ++lane) {
        const auto &event = record.lanes[lane];
        if (!fixture.participant(lane)) {
            EXPECT_EQ(event.descriptor_ready.sequence, 0u);
            EXPECT_EQ(event.arrive_after.sequence, 0u);
            EXPECT_EQ(event.release_after.sequence, 0u);
            continue;
        }
        EXPECT_EQ(event.worker_index, lane);
        EXPECT_EQ(event.lane, lane);
        EXPECT_EQ(event.run_epoch, 47u);
        EXPECT_EQ(event.group_seq, 1u);
        EXPECT_EQ(event.active_mask, mask);
        EXPECT_EQ(event.resource_mask, mask.merged_with(LaneMask::aic()));
        EXPECT_EQ(event.logical_block_num, 2u);
        EXPECT_GT(event.arrive_before.sequence, event.descriptor_ready.sequence);
        EXPECT_GT(event.arrive_after.sequence, event.arrive_before.sequence);
        EXPECT_GT(event.arrivals_observed.sequence, event.arrive_after.sequence);
        for (uint32_t peer = 0; peer < 3; ++peer) {
            EXPECT_EQ(event.arrived_seq[peer], fixture.participant(peer) ? 1u : 0u);
            EXPECT_EQ(event.previous_lane_seq[peer], 0u);
            EXPECT_EQ(event.lane_subtasks[peer], mask.has(static_cast<SubtaskSlot>(peer)) ? peer : UINT8_MAX);
        }
        if (mask.has(static_cast<SubtaskSlot>(lane))) EXPECT_EQ(event.release_before.sequence, 0u);
        else EXPECT_GT(event.release_before.sequence, event.arrivals_observed.sequence);
    }
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::WAIT);
    EXPECT_EQ(record.owner.releases_observed.sequence, 0u);
    EXPECT_EQ(record.owner.completion_before.sequence, 0u);
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        if (!fixture.participant(lane)) {
            EXPECT_EQ(record.owner.released_seq[lane], 0u);
            continue;
        }
        EXPECT_GT(record.lanes[lane].release_after.sequence, record.lanes[lane].release_before.sequence);
        EXPECT_EQ(record.owner.released_seq[lane], 1u);
    }
    EXPECT_GT(record.owner.releases_observed.sequence, record.lanes[0].release_after.sequence);
    EXPECT_GT(record.owner.completion_before.sequence, record.owner.releases_observed.sequence);
    EXPECT_GT(record.owner.completion_after.sequence, record.owner.completion_before.sequence);
    EXPECT_EQ(record.owner.completion_ticket, 0u);
    EXPECT_EQ(record.owner.worker_index, 0u);
    EXPECT_FALSE(fixture.task_done(0));
    ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 1));
    ASSERT_TRUE(fixture.task_done(0));
    const auto *tasks =
        scheduler_state_at<SchedulerProtocolTaskTrace>(fixture.base(), fixture.storage.layout.protocol.tasks_offset);
    const auto &last_group = blocks[mask.count()].owner;
    EXPECT_NE(tasks[0].completion.flags & SCHEDULER_PROTOCOL_COMPLETION_FROM_GROUP, 0u);
    EXPECT_EQ(tasks[0].completion.before.sequence, last_group.completion_before.sequence);
    EXPECT_EQ(tasks[0].completion.after.sequence, last_group.completion_after.sequence);
    EXPECT_EQ(last_group.completion_ticket, mask.count());
    EXPECT_GT(tasks[0].completion.after.sequence, tasks[0].completion.before.sequence);
    EXPECT_EQ(tasks[0].completion.order_key, 0u);
    EXPECT_EQ(tasks[0].completion.worker_index, 3u);
    EXPECT_EQ(blocks[1].owner.claim_after.sequence, 0u);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    for (uint32_t lane = 0; lane < 3; ++lane)
        EXPECT_EQ(workers[lane].error_flags, 0u);
}

INSTANTIATE_TEST_SUITE_P(AllMixMasks, HbgGmProtocolEvents, ::testing::ValuesIn(kMixMasks));

TEST(HbgGmMixProtocol, SameSequenceInDifferentClustersDoesNotAliasBlockOrLaneRecords) {
    GmMixFixture fixture({{2, LaneMask::both_aiv(), true}}, true);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
        ASSERT_TRUE(fixture.publish(cluster));
        ASSERT_TRUE(fixture.rendezvous(cluster));
    }
    const auto *blocks =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    for (uint32_t cluster = 0; cluster < 2; ++cluster) {
        ASSERT_TRUE(fixture.finish_lanes(cluster));
        ASSERT_EQ(fixture.retire(cluster), SchedulerGmMixResult::PROGRESS);
        const auto &record = blocks[cluster * 2];
        EXPECT_EQ(record.owner.cluster_index, cluster);
        EXPECT_EQ(record.owner.worker_index, cluster * 3);
        EXPECT_EQ(record.owner.block_idx, cluster);
        EXPECT_EQ(record.owner.group_seq, 1u);
        EXPECT_EQ(record.owner.completion_ticket, cluster * 2);
        for (uint32_t lane = 0; lane < 3; ++lane) {
            EXPECT_EQ(record.lanes[lane].worker_index, cluster * 3 + lane);
            EXPECT_EQ(record.lanes[lane].cluster_index, cluster);
            EXPECT_EQ(record.lanes[lane].block_idx, cluster);
            EXPECT_EQ(record.lanes[lane].group_seq, 1u);
            EXPECT_NE(record.lanes[lane].release_after.sequence, 0u);
        }
    }
    EXPECT_EQ(blocks[1].owner.claim_after.sequence, 0u);
}

TEST(HbgGmMixProtocol, FullWidthGroupSequenceIsPreservedInObservedAckValues) {
    GmMixFixture fixture({{2, LaneMask::all(), true}}, true);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    const uint64_t previous = SCHEDULER_GROUP_SEQUENCE_LIMIT - 2;
    for (uint32_t lane = 0; lane < 3; ++lane) {
        fixture.admissions[lane].gm_mix.group_seq = previous;
        scheduler_gm_store(fixture.mailbox(0)->lanes[lane].arrived_seq, previous);
        scheduler_gm_store(fixture.mailbox(0)->lanes[lane].released_seq, previous);
    }
    for (auto &sequence : fixture.admissions[0].gm_mix.vector_last_published_seq)
        sequence = previous;

    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    ASSERT_TRUE(fixture.finish_lanes(0));
    ASSERT_EQ(fixture.retire(0), SchedulerGmMixResult::PROGRESS);
    const auto *record =
        scheduler_state_at<SchedulerProtocolBlockTrace>(fixture.base(), fixture.storage.layout.protocol.blocks_offset);
    EXPECT_EQ(record->owner.group_seq, SCHEDULER_GROUP_SEQUENCE_LIMIT - 1);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(record->owner.released_seq[lane], SCHEDULER_GROUP_SEQUENCE_LIMIT - 1);
        EXPECT_EQ(record->lanes[lane].group_seq, SCHEDULER_GROUP_SEQUENCE_LIMIT - 1);
        for (uint32_t peer = 0; peer < 3; ++peer)
            EXPECT_EQ(record->lanes[lane].arrived_seq[peer], SCHEDULER_GROUP_SEQUENCE_LIMIT - 1);
    }
}

TEST(HbgGmMixProtocol, ReplayedPublicationCannotOverwriteItsFirstInterval) {
    for (const bool sync : {false, true}) {
        SCOPED_TRACE(sync ? "sync GM" : "ordinary SSBUF");
        GmMixFixture fixture({{2, LaneMask::all(), sync}}, true);
        ASSERT_TRUE(fixture.prepare_calls());
        bool progress = false;
        if (sync) {
            ASSERT_TRUE(fixture.admit(0));
            ASSERT_TRUE(fixture.refresh_all());
            ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
            ASSERT_TRUE(fixture.publish(0));
            for (uint32_t lane : {1u, 2u}) {
                ASSERT_TRUE(fixture.observe(lane, &progress));
                ASSERT_TRUE(fixture.arrive(lane, &progress));
            }
        } else {
            bool armed = false;
            ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
            ASSERT_TRUE(armed);
            for (uint32_t lane : {1u, 2u})
                ASSERT_TRUE(fixture.ordinary_service(lane, &progress));
        }
        // The actual direct-entry path publishes a payload without a READY
        // queue bit or consumed permit that could independently reject replay.
        auto *local = fixture.locals[0];
        uint32_t direct_slot = UINT32_MAX;
        bool failed = false;
        ASSERT_TRUE(scheduler_fill_own_slots(
            fixture.graph.graph(), fixture.base(), local, fixture.run(), 0, &failed, &direct_slot
        ));
        ASSERT_FALSE(failed);
        ASSERT_LT(direct_slot, SCHEDULER_PENDING_SLOT_COUNT);
        ASSERT_EQ(local->local_ready_mask, 0u);
        ASSERT_EQ(local->slots[direct_slot].state, SchedulerDispatchSlotState::READY);
        ASSERT_EQ(local->slots[direct_slot].cluster_mix, !sync);
        const auto *execution = scheduler_state_at<SchedulerProtocolExecutionTrace>(
            fixture.base(), fixture.storage.layout.protocol.executions_offset
        );
        ASSERT_EQ(execution->publication_count, 1u);
        ASSERT_EQ(execution->permit.sequence, 0u);
        ASSERT_EQ(execution->entry.sequence, 0u);
        const auto first_execution = *execution;
        const auto *group = scheduler_state_at<SchedulerProtocolBlockTrace>(
            fixture.base(), fixture.storage.layout.protocol.blocks_offset
        );
        const auto first_group = group->owner;
        SchedulerReadyClaim ready{};
        ready.task_ref = 0;
        ready.source = SchedulerReadySource::COOPERATIVE;
        scheduler_protocol_payload_publication(fixture.base(), local, direct_slot, ready, true);
        scheduler_protocol_payload_publication(fixture.base(), local, direct_slot, ready, false);
        EXPECT_EQ(execution->publication_count, first_execution.publication_count);
        EXPECT_EQ(execution->publication_before_cycles, first_execution.publication_before_cycles);
        EXPECT_EQ(execution->publication_after_cycles, first_execution.publication_after_cycles);
        EXPECT_EQ(execution->pending_slot, first_execution.pending_slot);
        EXPECT_EQ(execution->flags, first_execution.flags);
        if (sync) {
            scheduler_protocol_group_publish(fixture.base(), local, true);
            scheduler_protocol_group_publish(fixture.base(), local, false);
        }
        EXPECT_EQ(group->owner.publish_before.sequence, first_group.publish_before.sequence);
        EXPECT_EQ(group->owner.publish_after.sequence, first_group.publish_after.sequence);
        EXPECT_EQ(group->owner.group_seq, first_group.group_seq);
        const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(
            fixture.base(), fixture.storage.layout.protocol.workers_offset
        );
        EXPECT_EQ(worker->error_flags, SCHEDULER_PROTOCOL_DUPLICATE);
    }
}

// Admission ownership is independent of the type of the core that resolves
// the last dependency. Consumer discovery must not allocate an execution slot
// until the predecessor is logically DONE.
TEST(HbgConsumerAdmission, ReadyCooperativeSuccessorIsDiscoveredAfterCompletion) {
    {
        for (const auto mask : {LaneMask::aic(), LaneMask::aiv0(), LaneMask::both_aiv(), LaneMask::all()}) {
            for (uint32_t owner : {0u, 1u, 2u}) {
                SCOPED_TRACE(lane_names(mask));
                SCOPED_TRACE(owner);
                GmMixFixture fixture({{2, LaneMask::aic(), true}, {2, mask, true}});
                fixture.graph.executable(1, 0, {0});
                ASSERT_TRUE(fixture.prepare_calls());
                EXPECT_FALSE(fixture.task_done(1));
                ASSERT_TRUE(fixture.admit(0));
                fixture.control(0)->next_block = 2;
                fixture.control(0)->claim_seal = 2;
                fixture.control(0)->completion_state = scheduler_completion_completed_state(
                    scheduler_completion_total(fixture.control(0)->completion_state)
                );
                ASSERT_TRUE(fixture.task_done(0));
                auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
                    fixture.base(), fixture.storage.layout.admission_control_offset
                );
                EXPECT_EQ(shared->commit_tail, 1u);
                EXPECT_EQ(
                    fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
                );
                const uint32_t consumer = mask == LaneMask::aiv0() ? 1 : 0;
                ASSERT_TRUE(fixture.admit(1, consumer));
                EXPECT_EQ(shared->commit_tail, 2u);
                EXPECT_EQ(
                    fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED)
                );
                EXPECT_EQ(fixture.control(1)->next_block, 0u);
                EXPECT_EQ(scheduler_completion_count(fixture.control(1)->completion_state), 0u);
                for (auto *local : fixture.locals)
                    EXPECT_EQ(local->local_ready_mask, 0u);
                for (uint32_t engine = 0; engine < 2; ++engine) {
                    auto *holder = fixture.locals[engine == 0 ? 0 : 1];
                    for (uint32_t step = 0; step < 2; ++step) {
                        bool progress = false;
                        ASSERT_TRUE(scheduler_discover_cooperative_task(
                            fixture.graph.graph(), fixture.base(), holder, fixture.run(), &progress
                        ));
                        EXPECT_EQ(holder->admission->pending_task_ref, SCHEDULER_TASK_REF_INVALID);
                    }
                }
                EXPECT_EQ(shared->commit_tail, 2u);
            }
        }
    }
}

TEST(HbgConsumerAdmission, ContendedCommitRetainsCandidatesButPublishesOnce) {
    {
        GmMixFixture fixture({{2, LaneMask::aic(), true}, {2, LaneMask::all(), true}});
        fixture.graph.executable(1, 0, {0});
        {
            ASSERT_TRUE(fixture.prepare_calls());
            EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        }
        EXPECT_FALSE(fixture.task_done(1));
        ASSERT_TRUE(fixture.admit(0));
        auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
            fixture.base(), fixture.storage.layout.admission_control_offset
        );
        shared->commit_lock = 1;
        fixture.control(0)->next_block = 2;
        fixture.control(0)->claim_seal = 2;
        fixture.control(0)->completion_state =
            scheduler_completion_completed_state(scheduler_completion_total(fixture.control(0)->completion_state));
        ASSERT_TRUE(fixture.task_done(0));
        EXPECT_EQ(shared->commit_tail, 1u);
        EXPECT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        ASSERT_TRUE(fixture.claim_pending(1, 0));
        ASSERT_TRUE(fixture.claim_pending(1, 3));
        EXPECT_EQ(
            scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()),
            SchedulerAdmissionResult::RETRY
        );
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, 1);
        EXPECT_EQ(
            scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run()),
            SchedulerAdmissionResult::RETRY
        );
        EXPECT_EQ(fixture.admissions[3].pending_task_ref, 1);
        EXPECT_EQ(fixture.control(1)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        EXPECT_EQ(fixture.admissions[1].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(fixture.control(1)->next_block, 0u);
        shared->commit_lock = 0;
        EXPECT_EQ(
            scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()),
            SchedulerAdmissionResult::PROGRESS
        );
        EXPECT_EQ(shared->commit_tail, 2u);
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(
            scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), fixture.locals[3], fixture.run()),
            SchedulerAdmissionResult::PROGRESS
        );
        EXPECT_EQ(fixture.admissions[3].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        EXPECT_EQ(shared->commit_tail, 2u);
        const auto *log = scheduler_state_at<int64_t>(fixture.base(), fixture.storage.layout.admission_log_offset);
        EXPECT_EQ(log[1], 1);
    }
}

TEST(HbgConsumerAdmission, UnfinishedSecondFaninCannotBeAdmitted) {
    {
        GmMixFixture fixture({{2, LaneMask::aic(), true}, {2, LaneMask::aiv0(), true}, {2, LaneMask::all(), true}});
        fixture.graph.executable(2, 0, {0, 1});
        ASSERT_TRUE(fixture.prepare_calls());
        EXPECT_FALSE(fixture.task_done(2));
        ASSERT_TRUE(fixture.admit(0));
        fixture.control(0)->next_block = 2;
        fixture.control(0)->claim_seal = 2;
        fixture.control(0)->completion_state =
            scheduler_completion_completed_state(scheduler_completion_total(fixture.control(0)->completion_state));
        ASSERT_TRUE(fixture.task_done(0));
        auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
            fixture.base(), fixture.storage.layout.admission_control_offset
        );
        EXPECT_EQ(shared->commit_tail, 1u);
        EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED));
        auto *observer = fixture.locals[0];
        bool progress = false;
        ASSERT_TRUE(scheduler_discover_cooperative_task(
            fixture.graph.graph(), fixture.base(), observer, fixture.run(), &progress
        ));
        EXPECT_EQ(fixture.admissions[0].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
        ASSERT_TRUE(fixture.admit(1, 1));
        fixture.control(1)->next_block = 2;
        fixture.control(1)->claim_seal = 2;
        fixture.control(1)->completion_state =
            scheduler_completion_completed_state(scheduler_completion_total(fixture.control(1)->completion_state));
        ASSERT_TRUE(fixture.task_done(1));
        EXPECT_EQ(shared->commit_tail, 2u);
        ASSERT_TRUE(fixture.admit(2, 0));
        EXPECT_EQ(shared->commit_tail, 3u);
        EXPECT_EQ(fixture.control(2)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::ADMITTED));
    }
}

TEST(HbgConsumerAdmission, ConfirmedPrefixSurvivesDeferredFaninAndResetsForAnotherCandidate) {
    GmMixFixture fixture(
        {{1, LaneMask::aic()}, {1, LaneMask::aiv0()}, {2, LaneMask::all(), true}, {2, LaneMask::all(), true}}
    );
    fixture.graph.executable(2, 0, {0, 1});
    fixture.graph.executable(3, 0, {2});
    for (uint32_t task : {2u, 3u})
        ASSERT_TRUE(fixture.prepare_calls());
    fixture.control(0)->next_block = 1;
    fixture.control(1)->next_block = 1;
    fixture.control(0)->claim_seal = 1;
    fixture.control(1)->claim_seal = 1;
    fixture.control(0)->completion_state = scheduler_completion_completed_state(1);
    // Physical return is insufficient while an asynchronous debt remains.
    fixture.control(1)->completion_state = scheduler_completion_encode(1, 0, 1);
    auto *observer = fixture.locals[0];
    bool progress = false;
    ASSERT_TRUE(
        scheduler_discover_cooperative_task(fixture.graph.graph(), fixture.base(), observer, fixture.run(), &progress)
    );
    EXPECT_EQ(observer->cooperative_fanin_task_ref, 2);
    EXPECT_EQ(observer->cooperative_next_fanin, 1u);
    EXPECT_EQ(observer->admission->pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(
        scheduler_discover_cooperative_task(fixture.graph.graph(), fixture.base(), observer, fixture.run(), &progress)
    );
    EXPECT_FALSE(progress);
    EXPECT_EQ(observer->cooperative_next_fanin, 1u);
    EXPECT_EQ(observer->admission->pending_task_ref, SCHEDULER_TASK_REF_INVALID);

    fixture.control(1)->completion_state = scheduler_completion_completed_state(1);
    ASSERT_TRUE(
        scheduler_discover_cooperative_task(fixture.graph.graph(), fixture.base(), observer, fixture.run(), &progress)
    );
    EXPECT_EQ(observer->cooperative_next_fanin, 2u);
    ASSERT_EQ(observer->admission->pending_task_ref, 2);
    ASSERT_EQ(
        scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), observer, fixture.run()),
        SchedulerAdmissionResult::PROGRESS
    );
    // The next candidate may be supplied after all member tickets, before
    // this cohort computes. Its shorter fanin must not inherit the old prefix.
    fixture.control(2)->next_block = 2;
    fixture.control(2)->claim_seal = 2;
    ASSERT_TRUE(
        scheduler_discover_cooperative_task(fixture.graph.graph(), fixture.base(), observer, fixture.run(), &progress)
    );
    EXPECT_EQ(observer->cooperative_fanin_task_ref, 3);
    EXPECT_EQ(observer->cooperative_next_fanin, 0u);
    EXPECT_EQ(observer->admission->pending_task_ref, SCHEDULER_TASK_REF_INVALID);
    fixture.control(2)->completion_state = scheduler_completion_completed_state(6);
    ASSERT_TRUE(
        scheduler_discover_cooperative_task(fixture.graph.graph(), fixture.base(), observer, fixture.run(), &progress)
    );
    EXPECT_EQ(observer->cooperative_next_fanin, 1u);
    EXPECT_EQ(observer->admission->pending_task_ref, 3);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgConsumerAdmission, IndependentSuccessorsRemainDiscoverableExactlyOnce) {
    {
        GmMixFixture fixture({{2, LaneMask::aic(), true}, {2, LaneMask::all(), true}, {2, LaneMask::all(), true}});
        for (uint32_t task : {1u, 2u}) {
            fixture.graph.executable(task, 0, {0});
        }
        ASSERT_TRUE(fixture.prepare_calls());
        for (uint32_t task : {1u, 2u})
            EXPECT_FALSE(fixture.task_done(task));
        ASSERT_TRUE(fixture.admit(0));
        fixture.control(0)->next_block = 2;
        fixture.control(0)->claim_seal = 2;
        fixture.control(0)->completion_state =
            scheduler_completion_completed_state(scheduler_completion_total(fixture.control(0)->completion_state));
        ASSERT_TRUE(fixture.task_done(0));
        for (uint32_t task : {1u, 2u})
            EXPECT_EQ(
                fixture.control(task)->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::INITIALIZED)
            );
        std::set<int64_t> discovered;
        auto *observer = fixture.locals[0];
        for (uint32_t count = 0; count < 2; ++count) {
            for (uint32_t step = 0; step < 3 && fixture.admissions[0].pending_task_ref == SCHEDULER_TASK_REF_INVALID;
                 ++step) {
                bool progress = false;
                ASSERT_TRUE(scheduler_discover_cooperative_task(
                    fixture.graph.graph(), fixture.base(), observer, fixture.run(), &progress
                ));
            }
            const int64_t task = fixture.admissions[0].pending_task_ref;
            ASSERT_TRUE(task == 1 || task == 2);
            EXPECT_TRUE(discovered.insert(task).second);
            ASSERT_EQ(
                scheduler_ordered_publish_pending(fixture.graph.graph(), fixture.base(), observer, fixture.run()),
                SchedulerAdmissionResult::PROGRESS
            );
            if (count == 0) {
                ASSERT_TRUE(fixture.refresh_all());
                for (uint32_t worker = 0; worker < fixture.locals.size(); ++worker) {
                    if (fixture.admissions[worker].current_task_ref != 0) continue;
                    bool closed = false;
                    ASSERT_TRUE(scheduler_close_ordered_obligation(
                        fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), &closed
                    ));
                    ASSERT_TRUE(closed);
                }
                ASSERT_TRUE(fixture.refresh_all());
                for (uint32_t cluster = 0; cluster < 2; ++cluster)
                    ASSERT_EQ(fixture.claim(cluster), SchedulerBlockClaimResult::CLAIMED);
            }
        }
        for (uint32_t worker : {0u, 3u})
            for (uint32_t step = 0; step < 3; ++step) {
                auto *local = fixture.locals[worker];
                const uint64_t head = local->issue_cursor;
                const int64_t issue_task = local->issue_task_ref;
                const uint32_t fanin = local->issue_next_fanin;
                const uint64_t first_cursor = fixture.control(1)->next_block;
                const uint64_t second_cursor = fixture.control(2)->next_block;
                bool progress = false;
                ASSERT_TRUE(scheduler_discover_cooperative_task(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(), &progress
                ));
                if (progress)
                    EXPECT_TRUE(
                        local->issue_cursor > head ||
                        (local->issue_next_fanin != 0 &&
                         (local->issue_task_ref != issue_task || local->issue_next_fanin > fanin))
                    );
                EXPECT_EQ(fixture.admissions[worker].pending_task_ref, SCHEDULER_TASK_REF_INVALID);
                EXPECT_EQ(fixture.control(1)->next_block, first_cursor);
                EXPECT_EQ(fixture.control(2)->next_block, second_cursor);
            }
        const auto *shared = scheduler_state_at<SchedulerAdmissionControl>(
            fixture.base(), fixture.storage.layout.admission_control_offset
        );
        EXPECT_EQ(shared->commit_tail, 3u);
        const auto *log = scheduler_state_at<int64_t>(fixture.base(), fixture.storage.layout.admission_log_offset);
        EXPECT_EQ((std::set<int64_t>{log[1], log[2]}), discovered);
    }
}
TEST(HbgGmMix, FailedPhysicalReturnReleasesResourcesButCannotGrantGroupCredit) {
    for (bool sync_start : {false, true}) {
        for (LaneMask mask : kMixMasks) {
            SCOPED_TRACE(sync_start);
            SCOPED_TRACE(lane_names(mask));
            GmMixFixture fixture({{2, mask, sync_start}});
            ASSERT_TRUE(fixture.prepare_calls());
            if (sync_start) {
                ASSERT_TRUE(fixture.admit(0));
                ASSERT_TRUE(fixture.refresh_all());
                ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
                ASSERT_TRUE(fixture.publish(0));
                ASSERT_TRUE(fixture.rendezvous(0));
            } else {
                bool armed = false;
                ASSERT_TRUE(fixture.ordinary_arm(0, 0, &armed));
                ASSERT_TRUE(armed);
                service_cluster_members(fixture);
            }
            bool failed = false;
            for (uint32_t lane = 0; lane < 3; ++lane) {
                if (!mask.has(static_cast<SubtaskSlot>(lane))) continue;
                GmExecution execution{};
                ASSERT_TRUE(fixture.begin(lane, &execution));
                if (!failed) {
                    ASSERT_TRUE((scheduler_complete_own_slot<false, true, false>(
                        fixture.graph.graph(), fixture.base(), fixture.locals[lane], fixture.run(), execution.slot, 0
                    )));
                    failed = true;
                } else {
                    ASSERT_TRUE(fixture.complete(execution));
                }
                if (sync_start && fixture.admissions[0].gm_mix.phase == SchedulerGmMixPhase::RELEASED) {
                    EXPECT_NE(
                        scheduler_observe_ordered_progress(
                            fixture.graph.graph(), fixture.base(), fixture.locals[0], fixture.run()
                        ),
                        SchedulerOrderedPollResult::ERROR
                    );
                    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
                }
            }
            EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
            EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
            if (sync_start) {
                EXPECT_TRUE(fixture.admissions[0].gm_mix.release_failed);
                EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
            } else {
                bool progress = false;
                EXPECT_FALSE(fixture.ordinary_service(0, &progress));
            }
            EXPECT_NE(scheduler_read_error(fixture.run()), 0u);
            EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
            EXPECT_FALSE(fixture.task_done(0));
        }
    }
}

TEST(HbgCompletionPacking, GroupCreditsLeaveDeferredDebtForTheUniqueFinish) {
    for (uint64_t credits : {2u, 3u}) {
        SchedulerTaskControl control{};
        control.completion_state = scheduler_completion_encode(2 * credits, 2 * credits, 2 * credits + 1);
        uint64_t previous = UINT64_MAX;
        EXPECT_EQ(scheduler_contribute_returns(&control, credits, &previous), SchedulerSubtaskFinishResult::PARTIAL);
        EXPECT_EQ(previous, 0u);
        EXPECT_EQ(scheduler_completion_remaining(control.completion_state), credits);
        EXPECT_EQ(scheduler_completion_debt(control.completion_state), credits + 1);
        EXPECT_EQ(
            scheduler_contribute_returns(&control, credits, &previous), SchedulerSubtaskFinishResult::TASK_LAST_DEFERRED
        );
        EXPECT_EQ(previous, credits);
        EXPECT_EQ(scheduler_completion_remaining(control.completion_state), 0u);
        EXPECT_EQ(scheduler_completion_debt(control.completion_state), 1u);
        EXPECT_FALSE(scheduler_completion_done(control.completion_state));
    }
}

TEST(HbgCompletionPacking, MaximumSupportedShapesIgnoreUnrelatedAsyncActivityUntilTheUniqueLastReturn) {
    for (const auto mask : {LaneMask::aic(), LaneMask::aiv0(), LaneMask::all()}) {
        const auto blocks = static_cast<uint16_t>(INT16_MAX / mask.count());
        const uint32_t total = blocks * mask.count();
        std::vector<uint8_t> subtasks;
        for (uint8_t lane = 0; lane < 3; ++lane)
            if (mask.has(static_cast<SubtaskSlot>(lane))) subtasks.push_back(lane);
        for (const bool asynchronous : {false, true}) {
            GmMixFixture fixture({{blocks, mask}});
            auto *local = fixture.locals[0];
            SchedulerAsyncLocalState async{};
            local->async = &async;
            SchedulerLocalSlotState slot{};
            slot.task_ref = 0;
            for (uint32_t index = 0; index < total; ++index) {
                slot.block_idx = index / subtasks.size();
                slot.subtask_slot = subtasks[index % subtasks.size()];
                async.work_activated = asynchronous && index % 3 == 0;
                const auto result = scheduler_contribute_subtask<false>(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, 0
                );
                const auto expected = index + 1 != total ? SchedulerSubtaskFinishResult::PARTIAL :
                                                           SchedulerSubtaskFinishResult::TASK_LAST;
                ASSERT_EQ(result, expected);
            }
            const uint64_t state = fixture.control(0)->completion_state;
            EXPECT_EQ(scheduler_completion_count(state), total);
            EXPECT_EQ(scheduler_completion_total(state), total);
            EXPECT_EQ(scheduler_completion_remaining(state), 0u);
            EXPECT_EQ(scheduler_completion_debt(state), 0u);
            EXPECT_TRUE(scheduler_completion_done(state));
            EXPECT_EQ(
                (scheduler_contribute_subtask<false>(
                    fixture.graph.graph(), fixture.base(), local, fixture.run(), slot, 0
                )),
                SchedulerSubtaskFinishResult::ERROR
            );
        }
    }
}

// Addressed words model SDK-owned SRAM values, separate from all GM ACK lines.
void bind_mix_notifications(SchedulerGroupMailbox *mailbox, std::array<uint32_t, 6> &values) {
    for (uint32_t lane = 0; lane < 2; ++lane)
        for (uint32_t kind = 0; kind < 3; ++kind)
            mailbox->notification_addresses[lane][kind] = reinterpret_cast<uint64_t>(&values[lane * 3 + kind]);
}

TEST(HbgGmMix, CntNotifyReusesLaneSequencesAcrossChangingMembershipWithoutGmAckOperations) {
    GmMixFixture fixture({{2, LaneMask::all(), true}, {2, kAicAiv0, true}, {2, LaneMask::all(), true}});
    std::array<uint32_t, 6> signals{};
    auto *mailbox = fixture.mailbox(0);
    bind_mix_notifications(mailbox, signals);
    uint32_t gm_reads = 0, gm_atomics = 0;
    auto remote_ack = [&](volatile uint64_t *word) {
        for (uint32_t lane = 1; lane < 3; ++lane)
            if (word == &mailbox->lanes[lane].assigned_seq || word == &mailbox->lanes[lane].arrived_seq ||
                word == &mailbox->lanes[lane].released_seq)
                return true;
        return false;
    };
    ScopedGmQueryObserver observe_reads([&](volatile uint64_t *word) {
        gm_reads += remote_ack(word);
    });
    ScopedGmAtomicObserver observe_atomics([&](volatile uint64_t *word, uint64_t, GmAtomicOperation) {
        gm_atomics += remote_ack(word);
    });
    ASSERT_TRUE(fixture.prepare_calls());
    for (uint32_t task = 0; task < 3; ++task) {
        ASSERT_TRUE(fixture.admit(task));
        ASSERT_TRUE(fixture.refresh_all());
        ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 0));
        ASSERT_NO_FATAL_FAILURE(finish_sync_cluster(fixture, 1));
        EXPECT_TRUE(fixture.task_done(task));
        bool closed = false;
        for (uint32_t worker = 0; worker < 6; ++worker) {
            if (!fixture.participant(worker, task)) continue;
            ASSERT_TRUE(fixture.close(worker, &closed));
            ASSERT_TRUE(closed);
        }
    }
    EXPECT_EQ(signals[0], 3u);
    EXPECT_EQ(signals[3], 3u);
    EXPECT_EQ(signals[5], 3u);
    EXPECT_EQ(gm_reads, 0u);
    EXPECT_EQ(gm_atomics, 0u);
    EXPECT_EQ(scheduler_read_error(fixture.run()), 0u);
}

TEST(HbgGmMix, CntNotifyFailureReleasePreservesFailureAndPreventsGroupContribution) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    std::array<uint32_t, 6> signals{};
    bind_mix_notifications(fixture.mailbox(0), signals);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    ASSERT_EQ(fixture.claim(0), SchedulerBlockClaimResult::CLAIMED);
    ASSERT_TRUE(fixture.publish(0));
    ASSERT_TRUE(fixture.rendezvous(0));
    for (uint32_t worker = 0; worker < 3; ++worker) {
        GmExecution execution{};
        ASSERT_TRUE(fixture.begin(worker, &execution));
        if (worker == 2) {
            ASSERT_TRUE((scheduler_complete_own_slot<false, true, false>(
                fixture.graph.graph(), fixture.base(), fixture.locals[worker], fixture.run(), execution.slot, 0
            )));
        } else ASSERT_TRUE(fixture.complete(execution));
    }
    EXPECT_EQ(signals[5], (UINT32_C(1) << 31) | 1u);
    EXPECT_EQ(
        scheduler_gm_mix_query_signal(fixture.mailbox(0), 2, SchedulerGmMixSignal::RELEASED),
        SCHEDULER_GROUP_SEQUENCE_LIMIT | 1u
    );
    EXPECT_EQ(fixture.retire(0), SchedulerGmMixResult::ERROR);
    EXPECT_EQ(scheduler_completion_count(fixture.control(0)->completion_state), 0u);
}

TEST(HbgGmMix, CntNotifySequenceExhaustionFailsBeforeClaiming) {
    GmMixFixture fixture({{2, LaneMask::all(), true}});
    std::array<uint32_t, 6> signals{};
    bind_mix_notifications(fixture.mailbox(0), signals);
    ASSERT_TRUE(fixture.prepare_calls());
    ASSERT_TRUE(fixture.admit(0));
    ASSERT_TRUE(fixture.refresh_all());
    fixture.admissions[0].gm_mix.group_seq = SCHEDULER_GROUP_NOTIFY_SEQUENCE_LIMIT - 1;
    EXPECT_EQ(fixture.claim(0), SchedulerBlockClaimResult::ERROR);
    EXPECT_EQ(fixture.control(0)->next_block, 0u);
    EXPECT_EQ(signals[0], 0u);
}

}  // namespace
