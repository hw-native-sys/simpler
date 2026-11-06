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

#include "inner_kernel.h"

#include <gtest/gtest.h>
#include <array>
#include <cstring>
#include <memory>
#include <set>
#include <tuple>
#include <vector>

#include "async_plan.h"
#include "submission_bundle_plan.h"
#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_submission_bundle.h"

namespace {
constexpr uint64_t kEpoch = 93;
constexpr uint64_t kBudget = 64 * 1024 * 1024;
using Result = SchedulerSubmissionResult;
using Provisional = SchedulerSubmissionProvisionalResult;

SchedulerTaskMetadata task(uint32_t mask, uint32_t blocks = 1, bool sync = false) {
    SchedulerTaskMetadata meta{};
    meta.active_mask = LaneMask((mask & 1) != 0, (mask & 2) != 0, (mask & 4) != 0);
    meta.logical_block_num = blocks;
    meta.flags = SCHEDULER_TASK_EXECUTABLE | (blocks > 1 ? SCHEDULER_TASK_SPMD : 0) |
                 (meta.active_mask.is_mix() ? SCHEDULER_TASK_MIX : 0) | (sync ? SCHEDULER_TASK_SYNC_START : 0);
    meta.timing_slot = -1;
    for (uint32_t lane = 0; lane < 3; ++lane)
        meta.kernel_ids[lane] = (mask & (1u << lane)) != 0 ? 1 : UINT16_MAX;
    return meta;
}

std::vector<SchedulerAsyncCapabilities> explicit_capabilities(const std::vector<SchedulerTaskMetadata> &metadata) {
    std::vector<SchedulerAsyncCapabilities> caps(metadata.size());
    for (size_t task = 0; task < metadata.size(); ++task)
        for (uint32_t lane = 0; lane < 3; ++lane) {
            const bool active = metadata[task].active_mask.has(static_cast<SubtaskSlot>(lane));
            caps[task][lane] = {
                CORE_ASYNC_CAPABILITY_VERSION, active ? CoreAsyncMode::DEFERRED : CoreAsyncMode::SYNCHRONOUS,
                active ? 1u : 0u, active ? CORE_ASYNC_CONDITION_TYPES : 0u
            };
        }
    return caps;
}

bool plan_bundles(
    const std::vector<SchedulerTaskMetadata> &metadata,
    const std::vector<SchedulerSubmissionWorkspaceInput> &workspaces,
    const std::vector<SchedulerSubmissionExecutionUse> &uses, uint64_t epoch, uint64_t budget,
    SchedulerSubmissionHostPlan *plan, std::string *error
) {
    std::vector<SchedulerSubmissionExecutionUseSet> sets;
    for (size_t task = 0; task < metadata.size(); ++task)
        for (uint32_t block = 0; block < metadata[task].logical_block_num; ++block)
            for (uint32_t lane = 0; lane < 3; ++lane)
                if (metadata[task].active_mask.has(static_cast<SubtaskSlot>(lane)))
                    sets.push_back({static_cast<int64_t>(task), block, lane, 0, 1});
    return scheduler_plan_submission_bundles(
        metadata, explicit_capabilities(metadata), workspaces, uses, sets, epoch, budget, plan, error
    );
}

struct WorkspaceInputs {
    alignas(128) std::array<std::array<uint8_t, 256>, 16> storage{};
    SchedulerSubmissionWorkspaceInput at(uint32_t index, uint32_t queues = 1) {
        return {
            reinterpret_cast<uint64_t>(storage[index].data()),
            storage[index].size(),
            kEpoch,
            queues,
            SCHEDULER_SUBMISSION_SDMA_CHANNELS,
            SchedulerSubmissionPoolQualification::EXCLUSIVE_RUN
        };
    }
};

struct ActorBackend {};
struct Actor {
    SchedulerLocalState local{};
    SchedulerAdmissionLocalState admission{};
    SchedulerAsyncExecutionState<ActorBackend> state{};
    AsyncSubmissionEvidence evidence{};
};

struct Fixture {
    Fixture(
        const std::vector<SchedulerTaskMetadata> &metadata,
        const std::vector<SchedulerSubmissionWorkspaceInput> &workspaces,
        const std::vector<SchedulerSubmissionExecutionUse> &uses
    ) :
        image(nullptr, &std::free),
        cpu_image(nullptr, &std::free) {
        std::string error;
        if (!plan_bundles(metadata, workspaces, uses, kEpoch, kBudget, &plan, &error)) throw std::runtime_error(error);
        image.reset(std::aligned_alloc(128, plan.image.size()));
        if (!image) throw std::bad_alloc();
        std::memcpy(image.get(), plan.image.data(), plan.image.size());
        EXPECT_TRUE(scheduler_submission_mount(
            static_cast<uint8_t *>(image.get()), plan.image.size(), kEpoch, metadata.size(), &view
        ));
        const auto caps = explicit_capabilities(metadata);
        if (!scheduler_plan_async_state(metadata, caps, kEpoch, kBudget, {}, &cpu_plan, &error))
            throw std::runtime_error(error);
        cpu_image.reset(std::aligned_alloc(128, cpu_plan.image.size()));
        if (!cpu_image) throw std::bad_alloc();
        std::memcpy(cpu_image.get(), cpu_plan.image.data(), cpu_plan.image.size());
        actors.resize(plan.header.execution_count);
        for (uint64_t i = 0; i < actors.size(); ++i) {
            auto &actor = actors[i];
            const auto execution = row<SchedulerSubmissionExecution>(view.header.executions_offset, i);
            const auto &meta = metadata[execution.task_ref];
            auto &slot = actor.local.slots[0];
            slot.state = SchedulerDispatchSlotState::READY;
            slot.generation = 1;
            slot.task_ref = execution.task_ref;
            slot.block_idx = execution.block_idx;
            slot.subtask_slot = execution.subtask_slot;
            slot.cooperative = scheduler_task_requires_admission(meta);
            slot.gm_mix = scheduler_task_is_mix(meta.flags);
            if (slot.cooperative) {
                actor.local.admission = &actor.admission;
                actor.admission.initialized = true;
                actor.admission.epoch = kEpoch;
                actor.admission.sync_claimed = scheduler_task_requires_sync_start(meta.flags) && !slot.gm_mix;
                actor.admission.current_task_ref = slot.task_ref;
                actor.admission.current_block_num = meta.logical_block_num;
                actor.admission.current_flags = meta.flags;
                actor.admission.current_active_mask = meta.active_mask;
                actor.admission.gm_mix.phase = SchedulerGmMixPhase::RUNNING;
                actor.admission.gm_mix.slot_index = 0;
                actor.admission.gm_mix.slot_generation = slot.generation;
                actor.admission.gm_mix.block_idx = slot.block_idx;
                actor.admission.gm_mix.subtask_slot = slot.subtask_slot;
            }
            actor.state.view = {static_cast<uint8_t *>(cpu_image.get()), cpu_plan.layout};
            auto &e = actor.evidence;
            e.run_epoch = kEpoch;
            e.task_ref = execution.task_ref;
            e.execution_index = i;
            e.dispatch_generation = 1;
            e.block_idx = execution.block_idx;
            e.subtask_slot = execution.subtask_slot;
        }
    }
    template <typename T>
    T row(uint64_t offset, uint64_t index = 0) {
        T result{};
        scheduler_submission_detail::load_into(scheduler_submission_detail::at<T>(view, offset, index), &result);
        return result;
    }
    uint64_t busy(uint64_t workspace = 0) {
        return scheduler_gm_fetch_add(
            scheduler_submission_detail::at<SchedulerSubmissionPool>(view, view.header.pools_offset, workspace)
                ->busy_groups,
            UINT64_C(0)
        );
    }
    void occupy(uint64_t workspace, uint64_t mask) {
        scheduler_gm_store(
            scheduler_submission_detail::at<SchedulerSubmissionPool>(view, view.header.pools_offset, workspace)
                ->busy_groups,
            mask
        );
    }
    void bind(uint64_t index) {
        auto &actor = actors[index];
        actor.state.bound_generations[0] = actor.local.slots[0].generation;
    }
    void bind_all() {
        for (uint64_t i = 0; i < actors.size(); ++i)
            bind(i);
    }
    SchedulerSubmissionCpuGate<ActorBackend> gate(uint64_t index) {
        return {&actors[index].state, &actors[index].local, 0};
    }
    Result slice(uint64_t actor, uint32_t work = 8) {
        auto cpu = gate(actor);
        SchedulerSubmissionBudget budget{1, 1, work};
        return scheduler_submission_acquire(view, actor, &run, cpu, &budget);
    }
    void grant_cooperative(uint64_t actor) {
        for (uint32_t i = 0; i < 128; ++i) {
            const auto result = slice(actor, 3);
            ASSERT_NE(result, Result::ERROR);
            if (result == Result::READY) return;
        }
        FAIL() << "cooperative bundle did not progress within its finite test bound";
    }
    Provisional provisional(
        uint64_t index, SchedulerSubmissionProvisional *transaction,
        SchedulerSubmissionTransactionStats *stats = nullptr
    ) {
        auto cpu = gate(index);
        return scheduler_submission_try_ordinary(view, index, &run, cpu, transaction, stats);
    }
    void grant_ordinary(uint64_t index) {
        SchedulerSubmissionProvisional transaction;
        ASSERT_EQ(provisional(index, &transaction), Provisional::RESERVED);
        ASSERT_TRUE(scheduler_submission_commit_provisional(view, &transaction, &run));
    }
    bool release(uint64_t index) {
        const auto execution = row<SchedulerSubmissionExecution>(view.header.executions_offset, index);
        SchedulerSubmissionExecutionIdentity identity{kEpoch, execution.task_ref,  index,
                                                      1,      execution.block_idx, execution.subtask_slot};
        return scheduler_submission_release_all(view, identity, &actors[index].evidence, &run);
    }
    void sealed(uint64_t index) { actors[index].evidence.flags = ASYNC_SUBMISSION_FLAGS; }

    SchedulerSubmissionHostPlan plan;
    SchedulerAsyncHostPlan cpu_plan;
    std::unique_ptr<void, decltype(&std::free)> image, cpu_image;
    SchedulerSubmissionView view{};
    SchedulerRunControl run{};
    std::vector<Actor> actors;
};

TEST(HbgSubmissionBundle, PhysicalQueueCapacityRejectsWholeOverlargeSyncCohortBeforeLaunch) {
    WorkspaceInputs workspace;
    for (auto [blocks, queues, accepted] :
         {std::tuple{48u, 1u, true}, std::tuple{64u, 1u, false}, std::tuple{24u, 2u, true},
          std::tuple{25u, 2u, false}}) {
        std::vector<SchedulerSubmissionExecutionUse> uses;
        for (uint32_t block = 0; block < blocks; ++block)
            uses.push_back({0, block, 1, 0, 0});
        SchedulerSubmissionHostPlan plan;
        EXPECT_EQ(
            plan_bundles({task(2, blocks, true)}, {workspace.at(0, queues)}, uses, kEpoch, kBudget, &plan, nullptr),
            accepted
        );
        if (!accepted) EXPECT_TRUE(plan.image.empty());
    }
}

TEST(HbgSubmissionBundle, WorkspaceAliasesNormalizeAndConflictingQueueOrScalarClaimsReject) {
    WorkspaceInputs workspace;
    const auto owner = workspace.at(0);
    SchedulerSubmissionHostPlan plan;
    ASSERT_TRUE(plan_bundles(
        {task(2)}, {owner, owner}, {{0, 0, 1, 0, 0}, {0, 0, 1, 1, 1}, {0, 0, 1, 0, 0}}, kEpoch, kBudget, &plan, nullptr
    ));
    EXPECT_EQ(plan.header.workspace_count, 1u);
    EXPECT_EQ(plan.header.grant_count, 1u);
    EXPECT_EQ(plan.workspace_remap[0], plan.workspace_remap[1]);
    auto conflicting = owner;
    conflicting.queue_num = 2;
    EXPECT_FALSE(plan_bundles(
        {task(2)}, {owner, conflicting}, {{0, 0, 1, 0, 0}, {0, 0, 1, 1, 1}}, kEpoch, kBudget, &plan, nullptr
    ));
    EXPECT_FALSE(plan_bundles(
        {task(2)}, {owner, workspace.at(1)}, {{0, 0, 1, 0, 0}, {0, 0, 1, 0, 1}}, kEpoch, kBudget, &plan, nullptr
    ));
    EXPECT_EQ(plan.header.grant_count, 1u);
}

TEST(HbgSubmissionBundle, UnknownOrUnimplementedSharedAuthorityCannotAllocateAPool) {
    WorkspaceInputs workspace;
    for (auto authority :
         {SchedulerSubmissionPoolQualification::UNKNOWN, SchedulerSubmissionPoolQualification::OWNER_SHARED}) {
        auto input = workspace.at(0);
        input.qualification = authority;
        SchedulerSubmissionHostPlan plan;
        EXPECT_FALSE(plan_bundles({task(2)}, {input}, {{0, 0, 1, 0, 0}}, kEpoch, kBudget, &plan, nullptr));
        EXPECT_TRUE(plan.image.empty());
    }
    auto input = workspace.at(0);
    --input.qualification_epoch;
    SchedulerSubmissionHostPlan plan;
    EXPECT_FALSE(plan_bundles({task(2)}, {input}, {{0, 0, 1, 0, 0}}, kEpoch, kBudget, &plan, nullptr));
}

TEST(HbgSubmissionBundle, ContextBindingPrecedesReservationAndOnlySdmaLanesConsume) {
    WorkspaceInputs workspace;
    Fixture fixture({task(3)}, {workspace.at(0)}, {{0, 0, 1, 0, 0}});
    EXPECT_EQ(fixture.slice(1), Result::PENDING);
    EXPECT_EQ(fixture.busy(), 0u);
    fixture.bind_all();
    const auto request = fixture.row<SchedulerSubmissionRequest>(fixture.view.header.requests_offset);
    EXPECT_EQ(request.grant_count, 1u);
    fixture.grant_cooperative(0);
    EXPECT_EQ(fixture.busy(), 1u);
    EXPECT_EQ(fixture.row<SchedulerSubmissionExecution>(fixture.view.header.executions_offset, 0).grant_count, 0u);
    EXPECT_EQ(fixture.row<SchedulerSubmissionExecution>(fixture.view.header.executions_offset, 1).grant_count, 1u);
}

TEST(HbgSubmissionBundle, InvalidExistingCooperativeReservationCannotClaimChannels) {
    WorkspaceInputs workspace;
    for (uint32_t scenario = 0; scenario < 7; ++scenario) {
        Fixture fixture({task(3)}, {workspace.at(0)}, {{0, 0, 1, 0, 0}});
        fixture.bind_all();
        auto &actor = fixture.actors[1];
        if (scenario == 0) actor.admission.initialized = false;
        if (scenario == 1) ++actor.admission.epoch;
        if (scenario == 2) ++actor.admission.current_task_ref;
        if (scenario == 3) ++actor.admission.current_block_num;
        if (scenario == 4) actor.admission.current_active_mask = LaneMask::all();
        if (scenario == 5) actor.admission.gm_mix.phase = SchedulerGmMixPhase::ARRIVED;
        if (scenario == 6) ++actor.admission.gm_mix.slot_generation;
        EXPECT_EQ(fixture.slice(1), Result::ERROR) << scenario;
        EXPECT_EQ(fixture.busy(), 0u) << scenario;
    }
}

TEST(HbgSubmissionBundle, NonMixSyncStartRequiresItsExistingBlockClaim) {
    WorkspaceInputs workspace;
    Fixture fixture({task(2, 2, true)}, {workspace.at(0)}, {{0, 0, 1, 0, 0}, {0, 1, 1, 0, 0}});
    fixture.bind_all();
    fixture.actors[0].admission.sync_claimed = false;
    EXPECT_EQ(fixture.slice(0), Result::ERROR);
    EXPECT_EQ(fixture.busy(), 0u);
}

TEST(HbgSubmissionBundle, CooperativePrefixIsOrderedAndNoPeerCanEnterAPartialBundle) {
    WorkspaceInputs workspace;
    Fixture fixture(
        {task(6)}, {workspace.at(1), workspace.at(0)},
        {{0, 0, 1, 0, 0}, {0, 0, 1, 1, 1}, {0, 0, 2, 0, 0}, {0, 0, 2, 1, 1}}
    );
    fixture.bind_all();
    fixture.occupy(1, (UINT64_C(1) << 48) - 1);
    for (uint32_t i = 0; i < 12; ++i)
        EXPECT_EQ(fixture.slice(i % 2, 3), Result::PENDING);
    EXPECT_EQ(__builtin_popcountll(fixture.busy(0)), 2);
    const auto progress = fixture.row<SchedulerSubmissionProgress>(fixture.view.header.bundle_progress_offset);
    EXPECT_EQ(progress.request_index, 1u);
    SchedulerSubmissionGrant grant;
    SchedulerSubmissionWorkspace selected;
    uint64_t index = 0;
    EXPECT_FALSE(scheduler_submission_lookup_grant(fixture.view, 0, 0, &grant, &selected, &index));
    fixture.occupy(1, 0);
    fixture.grant_cooperative(1);
    EXPECT_EQ(__builtin_popcountll(fixture.busy(1)), 2);
    EXPECT_TRUE(scheduler_submission_lookup_grant(fixture.view, 0, 0, &grant, &selected, &index));
    EXPECT_EQ(selected.address, workspace.at(0).address);
}

struct FailFirstClaim : scheduler_submission_detail::Atomics {
    uint32_t *calls;
    uint64_t claim(volatile uint64_t &word, uint64_t expected, uint64_t desired) {
        ++*calls;
        return *calls == 1 ? expected ^ 1 : scheduler_gm_compare_exchange(word, expected, desired);
    }
};

TEST(HbgSubmissionBundle, ContendedCooperativeCasYieldsAndAnotherCoreResumesWithoutDuplicateReservation) {
    WorkspaceInputs workspace;
    Fixture fixture({task(6)}, {workspace.at(0)}, {{0, 0, 1, 0, 0}, {0, 0, 2, 0, 0}});
    fixture.bind_all();
    auto cpu = fixture.gate(0);
    SchedulerSubmissionBudget budget;
    uint32_t calls = 0;
    EXPECT_EQ(
        scheduler_submission_acquire(fixture.view, 0, &fixture.run, cpu, &budget, FailFirstClaim{{}, &calls}),
        Result::PENDING
    );
    EXPECT_EQ(calls, 1u);
    EXPECT_EQ(fixture.busy(), 0u);
    fixture.grant_cooperative(1);
    EXPECT_EQ(__builtin_popcountll(fixture.busy()), 2);
    EXPECT_EQ(fixture.slice(0), Result::READY);
    EXPECT_EQ(__builtin_popcountll(fixture.busy()), 2);
}

TEST(HbgSubmissionBundle, OrdinaryPartialClaimFailureReturnsEveryAcquiredWorkspaceBeforeYield) {
    WorkspaceInputs workspace;
    Fixture fixture({task(2)}, {workspace.at(0), workspace.at(1)}, {{0, 0, 1, 0, 0}, {0, 0, 1, 1, 1}});
    fixture.bind_all();
    fixture.occupy(1, (UINT64_C(1) << 48) - 1);
    SchedulerSubmissionProvisional transaction;
    SchedulerSubmissionTransactionStats stats;
    EXPECT_EQ(fixture.provisional(0, &transaction, &stats), Provisional::PENDING);
    EXPECT_FALSE(transaction.active);
    EXPECT_EQ(fixture.busy(0), 0u);
    EXPECT_EQ(stats.pool_returns, 1u);
    EXPECT_LE(stats.pool_probes, 2u);
    fixture.occupy(1, 0);
    fixture.grant_ordinary(0);
    EXPECT_EQ(fixture.busy(0), 1u);
    EXPECT_EQ(fixture.busy(1), 1u);
}

TEST(HbgSubmissionBundle, OriginalOrderedAdmissionDenialRollsBackSoCooperativeWorkCanClaimTheSameChannel) {
    WorkspaceInputs workspace;
    const std::vector<SchedulerTaskMetadata> metadata{task(2), task(2, 2, true)};
    Fixture fixture(metadata, {workspace.at(0)}, {{0, 0, 1, 0, 0}, {1, 0, 1, 0, 0}, {1, 1, 1, 0, 0}});
    fixture.bind_all();
    SchedulerSubmissionProvisional transaction;
    ASSERT_EQ(fixture.provisional(0, &transaction), Provisional::RESERVED);
    ASSERT_EQ(fixture.busy(), 1u);

    scheduler_test::FixtureStorage storage(2, 6, 2, {6, 2, 1, 0});
    scheduler_test::BasicGraphBuffer<2> graph(2);
    graph.executable(0, 1);
    graph.executable(1, 1);
    std::copy(metadata.begin(), metadata.end(), storage.metadata);
    ASSERT_TRUE(scheduler_init_spmd_task_records(storage.scheduler_state->base(), storage.layout, kEpoch));
    for (uint32_t worker = 0; worker < 6; ++worker) {
        storage.contexts[worker].is_scheduler = 1;
        storage.contexts[worker].scheduler_count = 6;
        for (uint32_t lane = 0; lane < 3; ++lane)
            storage.contexts[worker].cluster_worker_ids[lane] = (worker / 3) * 3 + lane;
        storage.test_graphs[worker] = graph.graph();
    }
    auto *local = storage.local_context(&storage.contexts[1]);
    SchedulerAdmissionLocalState admission;
    ASSERT_TRUE(scheduler_initialize_admission(storage.scheduler_state->base(), local, &admission));
    local->admission = &admission;
    local->slots[0].state = SchedulerDispatchSlotState::READY;
    local->slots[0].task_ref = 0;
    local->slots[0].generation = 1;
    local->slots[0].candidate_index = 0;
    admission.candidates[0].ready.task_ref = 0;
    admission.candidates[0].slot_index = 0;
    auto *control =
        scheduler_state_at<SchedulerAdmissionControl>(storage.scheduler_state->base(), admission.control_offset);
    scheduler_gm_store(control->commit_tail, UINT64_C(1));
    bool permit = true;
    ASSERT_TRUE(scheduler_ordered_can_enter_slot(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, &permit
    ));
    EXPECT_FALSE(permit);
    ASSERT_TRUE(scheduler_submission_abort_provisional(fixture.view, &transaction, &fixture.run));
    EXPECT_EQ(fixture.busy(), 0u);
    fixture.grant_cooperative(1);
    EXPECT_EQ(fixture.busy(), 3u);
}

TEST(HbgSubmissionBundle, MetadataCommitAndRuntimeSealEvidenceAreRequiredBeforeRelease) {
    WorkspaceInputs workspace;
    for (uint32_t scenario = 0; scenario < 5; ++scenario) {
        Fixture fixture({task(2)}, {workspace.at(0)}, {{0, 0, 1, 0, 0}});
        fixture.bind_all();
        fixture.grant_ordinary(0);
        auto &e = fixture.actors[0].evidence;
        fixture.sealed(0);
        if (scenario == 0) e.flags = ASYNC_SUBMISSION_METADATA_PUBLISHED;
        if (scenario == 1) e.flags |= 4;
        if (scenario == 2) {
            e.attempted = 2;
            e.committed = 1;
        }
        if (scenario == 3) ++e.dispatch_generation;
        if (scenario == 4) e.error_code = -7;
        EXPECT_FALSE(fixture.release(0));
        EXPECT_EQ(fixture.busy(), 1u);
    }
}

TEST(HbgSubmissionBundle, WriterReleaseIsExactlyOnceWhileItsEventMayRemainPending) {
    WorkspaceInputs workspace;
    Fixture fixture({task(6)}, {workspace.at(0)}, {{0, 0, 1, 0, 0}, {0, 0, 2, 0, 0}});
    fixture.bind_all();
    fixture.grant_cooperative(0);
    fixture.sealed(0);
    fixture.actors[0].evidence.attempted = fixture.actors[0].evidence.committed = 1;
    ASSERT_TRUE(fixture.release(0));
    const uint64_t remaining = fixture.busy();
    EXPECT_EQ(__builtin_popcountll(remaining), 1);
    EXPECT_TRUE(fixture.release(0));
    EXPECT_EQ(fixture.busy(), remaining);
    fixture.sealed(1);
    ASSERT_TRUE(fixture.release(1));
    EXPECT_EQ(fixture.busy(), 0u);
    // No event/condition record is read or cleared by submit-writer retirement.
    EXPECT_EQ(fixture.actors[0].evidence.committed, 1u);
}

TEST(HbgSubmissionBundle, ZeroBudgetDoesNotAcquireLeaseOrChannels) {
    WorkspaceInputs workspace;
    Fixture fixture({task(6)}, {workspace.at(0)}, {{0, 0, 1, 0, 0}, {0, 0, 2, 0, 0}});
    fixture.bind_all();
    EXPECT_EQ(fixture.slice(0, 0), Result::PENDING);
    EXPECT_EQ(fixture.busy(), 0u);
    const auto lease = scheduler_async_lease_load(
        *scheduler_submission_detail::at<SchedulerAsyncLease>(fixture.view, fixture.view.header.bundle_leases_offset)
    );
    EXPECT_EQ(lease, scheduler_async_lease_word(1, SchedulerAsyncPhase::RUNNABLE));
}
}  // namespace
