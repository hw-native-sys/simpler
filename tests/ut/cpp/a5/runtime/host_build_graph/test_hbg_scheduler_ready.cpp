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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>

#include "scheduler/scheduler_ready.h"
#include "runtime_types.h"
#include "hbg_scheduler_test_support.h"

namespace {

using scheduler_test::FixtureStorage;
using scheduler_test::GraphBuffer;
using scheduler_test::kFixtureCallableCount;
using scheduler_test::SchedulerStateBuffer;

// Put one task in a queue's inbox at `owner` and mark the directory for it, the
// state a push from that owner's cluster leaves behind. Cases about where a
// thief may look use this so they do not also depend on push routing.
void seed_ready_inbox(
    FixtureStorage &storage, SchedulerLocalState *local, uint32_t core_type_index, uint64_t owner, int64_t task_ref
) {
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), local, task_ref);
    control->next_waiter = SCHEDULER_INBOX_EMPTY;
    scheduler_ready_inbox_at(storage.scheduler_state->base(), local, core_type_index, owner)->endpoints =
        scheduler_ready_endpoints_pack(task_ref, task_ref);
    scheduler_ready_directory_set(
        scheduler_ready_directory_at(storage.scheduler_state->base(), local), core_type_index, owner
    );
}

// The AIV queue of a five-cluster run: ten owners, so ranks 0-6 fall in the
// first directory shard and 7-9 in the second. An AIV rank maps back to the
// worker that holds it as 3 * (rank / 2) + 1 + rank % 2.
constexpr uint32_t kAivQueue = 1;
constexpr uint64_t kAivWorkers = 15;

uint64_t aiv_rank_worker(uint64_t rank) {
    return 3 * (rank / PLATFORM_AIV_CORES_PER_BLOCKDIM) + 1 + rank % PLATFORM_AIV_CORES_PER_BLOCKDIM;
}

TEST(SchedulerActivityBuffer, IsAllocatedOnlyWhenRequestedAndNeverWraps) {
    AicoreSchedulerLayout disabled{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &disabled));
    EXPECT_EQ(disabled.activity_buffers_offset, 0u);

    AicoreSchedulerLayout enabled{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &enabled, true));
    ASSERT_NE(enabled.activity_buffers_offset, 0u);
    EXPECT_EQ(
        enabled.total_size - disabled.total_size,
        static_cast<uint64_t>(SCHEDULER_WORKER_CAPACITY) * sizeof(SchedulerActivityBuffer)
    );
    SchedulerStateBuffer storage(enabled);
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(storage.base(), enabled.worker_contexts_offset);
    auto *buffers = scheduler_state_at<SchedulerActivityBuffer>(storage.base(), enabled.activity_buffers_offset);
    contexts[0].worker_index = SCHEDULER_WORKER_CAPACITY - 1;
    contexts[0].is_scheduler = 1;
    contexts[0].scheduler_index = SCHEDULER_WORKER_CAPACITY - 1;
    SchedulerLocalState local{};
    SchedulerLocalProfilingState profile{};
    local.profiling = &profile;
    local.profiling->activity_buffers_offset = contexts[0].activity_buffers_offset;
    local.config.scheduler_index = contexts[0].scheduler_index;
    local.config.self_lane = 0;
    local.config.scheduler_lane = 0;
    local.profiling->loop_iter = 17;
    SchedulerActivityBuffer &buffer = buffers[SCHEDULER_WORKER_CAPACITY - 1];
    buffer.committed = SCHEDULER_ACTIVITY_CAPACITY - 1;

    scheduler_append_idle_activity(storage.base(), &local, 10, 20);
    scheduler_append_idle_activity(storage.base(), &local, 30, 40);

    EXPECT_EQ(buffer.committed, SCHEDULER_ACTIVITY_CAPACITY);
    EXPECT_EQ(buffer.dropped, 1u);
    const SchedulerIdleRecord &last = buffer.records[SCHEDULER_ACTIVITY_CAPACITY - 1];
    EXPECT_EQ(last.start_time, 10u);
    EXPECT_EQ(last.end_time, 20u);
    EXPECT_EQ(last.loop_iter, 17u);
}

TEST(SchedulerActivityBuffer, RejectsCorruptCommittedCountOnHost) {
    EXPECT_TRUE(scheduler_activity_record_count_valid(SCHEDULER_ACTIVITY_CAPACITY));
    EXPECT_FALSE(scheduler_activity_record_count_valid(SCHEDULER_ACTIVITY_CAPACITY + 1));
}

TEST(SchedulerProfilingLevel, EnablesOnlyTheRequestedGranularity) {
    EXPECT_FALSE(scheduler_task_timing_enabled(0));
    EXPECT_TRUE(scheduler_task_timing_enabled(SCHEDULER_PROFILING_TASK_TIMING_LEVEL));
    EXPECT_FALSE(scheduler_schedule_timing_enabled(SCHEDULER_PROFILING_TASK_TIMING_LEVEL));
    EXPECT_TRUE(scheduler_schedule_timing_enabled(SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL));
    EXPECT_FALSE(scheduler_phase_timing_enabled(SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL));
    EXPECT_TRUE(scheduler_phase_timing_enabled(SCHEDULER_PROFILING_SCHED_PHASES_LEVEL));
}

TEST(SchedulerProfilingLevel, DispatchWritesTaskIdentityBeforePhaseDetails) {
    for (uint64_t level = 0; level <= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL; ++level) {
        FixtureStorage storage(1, 2);
        GraphBuffer graph(1);
        graph.executable(0, 0);
        storage.contexts[1].core_type = static_cast<int32_t>(CoreType::AIC);
        SchedulerReadyClaim ready_claim{};
        ready_claim.task_ref = 0;
        ready_claim.source = SchedulerReadySource::STOLEN;
        ready_claim.state_probe_start_cycles = 100;
        ready_claim.state_probe_end_cycles = 123;

        ASSERT_TRUE(scheduler_fill_dispatch_slot(
            graph.graph(), storage.scheduler_state->base(),
            storage.local_context(&storage.contexts[1], &storage.scheduler_local_state), storage.run_control,
            SchedulerFreeSlotClaim{1, 0, 0, 0}, ready_claim, level, storage.ssbuf_region
        ));
        auto *traces =
            scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
        EXPECT_EQ(traces[0].worker_id, level == 0 ? 0u : storage.contexts[1].worker_index);
        EXPECT_EQ(traces[0].task_ref, 0u);
        EXPECT_EQ(
            traces[0].state_probe_scheduler_worker_id,
            level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? storage.contexts[1].worker_index : 0u
        );
        EXPECT_EQ(traces[0].state_probe_start_cycles, level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? 100u : 0u);
        EXPECT_EQ(traces[0].state_probe_end_cycles, level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? 123u : 0u);
        EXPECT_EQ(
            traces[0].ready_source,
            level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ? static_cast<uint64_t>(SchedulerReadySource::STOLEN) : 0u
        );
        EXPECT_EQ(
            traces[0].publication_mode, level >= SCHEDULER_PROFILING_SCHED_PHASES_LEVEL ?
                                            static_cast<uint64_t>(SchedulerPublicationMode::DISPATCH) :
                                            0u
        );
    }
}

TEST(SchedulerBootstrap, RegistersOnlyOnFirstExecutableProducer) {
    FixtureStorage storage(4, 2);
    GraphBuffer graph(4);
    graph.executable(0, 0);
    graph.executable(1, 1, {0});
    graph.executable(3, 1, {2, 1});
    storage.metadata[1].active_mask = LaneMask::aiv0();
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    storage.metadata[2].flags = 0;
    storage.metadata[3].active_mask = LaneMask::aiv0();
    storage.metadata[3].flags |= SCHEDULER_TASK_HAS_FANIN;
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[2].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    controls[2].wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;

    SchedulerWakeStats stats{};
    EXPECT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 1, &stats
        ),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(controls[0].wake_list_head, 1);
    EXPECT_EQ(controls[1].next_waiter, SCHEDULER_WAKE_LIST_OPEN);
    EXPECT_EQ(controls[1].waiting_producer, 0);

    EXPECT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[1]),
            storage.run_control, 3, &stats
        ),
        SchedulerRouteResult::WAITING
    );
    EXPECT_EQ(controls[1].wake_list_head, 3);
    EXPECT_EQ(controls[3].next_fanin_index, 1);
    EXPECT_EQ(controls[3].waiting_producer, 1);
    EXPECT_EQ(stats.wake_register_count, 2u);
    EXPECT_EQ(stats.fanin_state_load_count, 0u);
    EXPECT_EQ(stats.wake_cas_retry_count, 0u);
}

// Bootstrap publishes through the same type-routed push the steady state uses,
// so a core must not leave work of another type in its own inbox: an AIC task
// an AIV lane found during its graph scan belongs to the cluster's AIC.
TEST(SchedulerBootstrap, PushesAppendedBatchToTheTypeOwnersInbox) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    for (uint64_t worker = 0; worker < 3; ++worker)
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
            storage.contexts[worker].cluster_worker_ids[lane] = lane;
    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    auto *publisher = storage.local_context(&storage.contexts[1], &storage.owner_states[1]);
    ASSERT_EQ(publisher->config.self_lane, 1u);
    ASSERT_TRUE(scheduler_bootstrap_ready_batch_append(storage.scheduler_state->base(), publisher, 0, &batch, &stats));
    ASSERT_TRUE(scheduler_bootstrap_ready_batch_append(storage.scheduler_state->base(), publisher, 1, &batch, &stats));
    ASSERT_TRUE(scheduler_ready_batch_push(storage.scheduler_state->base(), publisher, 0, &batch, &stats));

    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    EXPECT_EQ(controls[0].next_waiter, 1);
    EXPECT_EQ(controls[1].next_waiter, SCHEDULER_INBOX_EMPTY);
    const uint64_t owner_endpoints =
        scheduler_ready_inbox_at(storage.scheduler_state->base(), publisher, 0, 0)->endpoints;
    EXPECT_EQ(scheduler_ready_endpoints_head(owner_endpoints), 0);
    EXPECT_EQ(scheduler_ready_endpoints_tail(owner_endpoints), 1);
    EXPECT_EQ(
        scheduler_ready_inbox_at(storage.scheduler_state->base(), publisher, 0, 1)->endpoints,
        SCHEDULER_READY_ENDPOINTS_EMPTY
    );
    auto *directory = scheduler_ready_directory_at(storage.scheduler_state->base(), publisher);
    EXPECT_EQ(directory->core_types[0][0].bits, 1u);
    EXPECT_EQ(directory->core_types[1][0].bits, 0u);
    EXPECT_EQ(stats.enqueue_count, 2u);
    EXPECT_EQ(stats.batch_count, 1u);
}

TEST(SchedulerReadyInbox, RejectsReadyQueueBoundary) {
    FixtureStorage storage(1, 1);
    SchedulerReadyStats stats{};
    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), 0, &batch, &stats
    ));

    // A queue is addressed by its holders' rank within their own core type, so
    // what bounds every access is that queue's live owner count -- not a worker
    // id and not this core's Scheduler identity.
    auto *local = storage.local_context(&storage.contexts[0], &storage.owner_states[0]);
    EXPECT_TRUE(scheduler_refresh_ready_inbox_type(storage.scheduler_state->base(), local, 0));
    local->config.cluster_count = 0;
    EXPECT_FALSE(scheduler_refresh_ready_inbox_type(storage.scheduler_state->base(), local, 0));
    EXPECT_FALSE(scheduler_ready_batch_push(storage.scheduler_state->base(), local, 0, &batch, &stats));

    local = storage.local_context(&storage.contexts[0], &storage.owner_states[0]);
    EXPECT_FALSE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), local, SCHEDULER_READY_QUEUE_COUNT, &batch, &stats)
    );
    // An AIC holds the AIC and Mix queues; the AIV queue is not its to refresh.
    EXPECT_FALSE(scheduler_refresh_ready_inbox_type(storage.scheduler_state->base(), local, 1));
    EXPECT_TRUE(scheduler_refresh_ready_inbox_type(storage.scheduler_state->base(), local, SCHEDULER_MIX_QUEUE));
}

TEST(SchedulerReadyInbox, RequiresOwnerStateForPublishAndClaim) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerReadyStats stats{};
    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), 0, &batch, &stats
    ));

    EXPECT_FALSE(scheduler_ready_batch_push(storage.scheduler_state->base(), nullptr, 0, &batch, &stats));
    uint64_t cursor = 0;
    SchedulerReadyClaim claim{};
    EXPECT_FALSE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), nullptr, storage.run_control, 1, 0, &cursor, &stats, &claim
    ));
    EXPECT_FALSE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), nullptr, storage.run_control, 0, nullptr, nullptr, nullptr, 0,
        true, nullptr, UINT32_MAX
    ));
}

TEST(SchedulerReadyInbox, BatchPushAndInboxRefreshPreserveFifoAndDirectory) {
    constexpr uint64_t kTasks = 4;
    FixtureStorage storage(kTasks, 1);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerReadyBatch batch{};
    SchedulerLocalState &owner_state = storage.owner_states[0];
    SchedulerReadyStats stats{};
    for (uint64_t task = 0; task < kTasks; ++task)
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), task, &batch, &stats
        ));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state), 0, &batch, &stats
    ));

    auto *directory = scheduler_state_at<SchedulerReadyDirectory>(
        storage.scheduler_state->base(), storage.layout.ready_directory_offset
    );
    EXPECT_NE(directory->core_types[0][0].bits & 1, 0u);
    for (uint64_t index = 0; index < kTasks; ++index) {
        int64_t task = SCHEDULER_TASK_REF_INVALID;
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 0, 0, &task, &stats
        ));
        EXPECT_EQ(task, static_cast<int64_t>(index));
    }
    int64_t task = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
        storage.run_control, 0, 0, &task, &stats
    ));
    EXPECT_EQ(task, SCHEDULER_TASK_REF_INVALID);
    EXPECT_NE(directory->core_types[0][0].bits & 1, 0u);
    ASSERT_TRUE(scheduler_refresh_ready_inbox_type(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state), 0
    ));
    EXPECT_EQ(directory->core_types[0][0].bits & 1, 0u);
    EXPECT_EQ(stats.pop_count, kTasks);
}

TEST(SchedulerReadyInbox, RefreshesOnlyLocallyActiveQueueTypes) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerLocalState scheduler_local_state{};
    auto *inactive_inbox = scheduler_ready_inbox_at(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), SCHEDULER_MIX_QUEUE, 0
    );
    // One endpoint naming a task while the other is empty is corruption, which
    // a refresh of that type must reject rather than schedule against. Worker 0
    // is its cluster's AIC, so the Mix queue is one it does hold.
    inactive_inbox->endpoints = scheduler_ready_endpoints_pack(SCHEDULER_INBOX_EMPTY, 0);

    EXPECT_TRUE(scheduler_refresh_ready_inbox(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &scheduler_local_state)
    ));

    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), 0, &batch, &stats
    ));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &scheduler_local_state), 0, &batch,
        &stats
    ));
    // A push marks the shared directory for the target, which may be another
    // core, so it cannot touch the target's local mask. Only the owner's own
    // refresh mirrors the bit locally.
    auto *directory = scheduler_ready_directory_at(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &scheduler_local_state)
    );
    EXPECT_NE(directory->core_types[0][0].bits & 1, 0u);
    EXPECT_EQ(scheduler_local_state.owner_ready_queue_mask, UINT32_C(0));
    ASSERT_TRUE(scheduler_refresh_ready_inbox_type(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &scheduler_local_state), 0
    ));
    EXPECT_EQ(scheduler_local_state.owner_ready_queue_mask, UINT32_C(1));

    int64_t task = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
        storage.run_control, 0, 0, &task, &stats
    ));
    ASSERT_EQ(task, 0);
    ASSERT_TRUE(scheduler_refresh_ready_inbox(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &scheduler_local_state)
    ));
    EXPECT_EQ(scheduler_local_state.owner_ready_queue_mask, UINT32_C(0));

    // A queue this core cannot hold is skipped rather than refreshed, so
    // activating the AIV queue on an AIC changes nothing.
    scheduler_owner_queue_activate(&scheduler_local_state, 1);
    EXPECT_TRUE(scheduler_refresh_ready_inbox(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &scheduler_local_state)
    ));
    scheduler_owner_queue_deactivate(&scheduler_local_state, 1);

    scheduler_owner_queue_activate(&scheduler_local_state, SCHEDULER_MIX_QUEUE);
    EXPECT_FALSE(scheduler_refresh_ready_inbox(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &scheduler_local_state)
    ));
}

TEST(SchedulerReadyInbox, StealsOnlyFromMarkedVictim) {
    // Two clusters, so the AIC queue has two owners and a steal has somewhere
    // to come from: within one cluster an AIC task routes to the stealer itself.
    FixtureStorage storage(1, 6);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats stats{};
    auto *publisher = storage.local_context(&storage.contexts[3], &storage.owner_states[3]);
    ASSERT_EQ(publisher->self_type_rank(), 1u);
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), publisher, 0, &batch, &stats));
    ASSERT_TRUE(scheduler_ready_batch_push(storage.scheduler_state->base(), publisher, 0, &batch, &stats));

    uint64_t cursor = 1;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.owner_states[0]), storage.run_control, 6, 0, &cursor,
        &stats, &claim
    ));
    EXPECT_EQ(claim.task_ref, 0);
    EXPECT_EQ(claim.inbox_index, 1u);
    EXPECT_EQ(claim.source, SchedulerReadySource::STOLEN);
    EXPECT_EQ(stats.steal_count, 1u);
}

TEST(SchedulerReadyInbox, DirectoryShardIgnoresSchedulerTail) {
    FixtureStorage storage(1, 9);
    auto *directory =
        scheduler_ready_directory_at(storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]));
    directory->core_types[0][1].bits = UINT64_C(1) << 6;
    EXPECT_EQ(scheduler_load_ready_directory_shard(directory, 9, 0, 7), 0u);

    directory->core_types[0][1].bits = UINT64_C(1) << 1;
    EXPECT_EQ(scheduler_load_ready_directory_shard(directory, 9, 0, 7), UINT64_C(1) << 1);
}

TEST(SchedulerReadyInbox, SparseDirectoryWrapsWithinShard) {
    FixtureStorage storage(2, kAivWorkers);
    GraphBuffer graph(2);
    graph.executable(0, 1);
    graph.executable(1, 1);
    SchedulerReadyStats stats{};
    auto *thief = storage.local_context(&storage.contexts[aiv_rank_worker(7)], &storage.owner_states[0]);
    ASSERT_EQ(thief->self_type_rank(), 7u);
    seed_ready_inbox(storage, thief, kAivQueue, 9, 0);
    seed_ready_inbox(storage, thief, kAivQueue, 8, 1);

    uint64_t cursor = 9;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), thief, storage.run_control, kAivWorkers, kAivQueue, &cursor,
        &stats, &claim
    ));
    EXPECT_EQ(claim.task_ref, 0);
    EXPECT_EQ(claim.inbox_index, 9u);
    // Past the shard's last owner, so the cursor wraps to its first.
    EXPECT_EQ(cursor, 7u);

    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), thief, storage.run_control, kAivWorkers, kAivQueue, &cursor,
        &stats, &claim
    ));
    EXPECT_EQ(claim.task_ref, 1);
    EXPECT_EQ(claim.inbox_index, 8u);
    EXPECT_EQ(cursor, 9u);
}

TEST(SchedulerReadyInbox, DoesNotStealAcrossDirectoryShards) {
    FixtureStorage storage(1, kAivWorkers);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    SchedulerReadyStats stats{};
    auto *owner = storage.local_context(&storage.contexts[aiv_rank_worker(8)], &storage.owner_states[0]);
    ASSERT_EQ(owner->self_type_rank(), 8u);
    seed_ready_inbox(storage, owner, kAivQueue, 8, 0);

    // Rank 0 is in the first shard, so the seeded owner is outside its reach.
    uint64_t cursor = 1;
    SchedulerReadyClaim claim{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[aiv_rank_worker(0)], &storage.owner_states[1]), storage.run_control,
        kAivWorkers, kAivQueue, &cursor, &stats, &claim
    ));
    EXPECT_EQ(claim.task_ref, SCHEDULER_TASK_REF_INVALID);

    // Rank 7 shares the second shard with it, so the steal lands.
    cursor = 8;
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[aiv_rank_worker(7)], &storage.owner_states[2]), storage.run_control,
        kAivWorkers, kAivQueue, &cursor, &stats, &claim
    ));
    EXPECT_EQ(claim.task_ref, 0);
    EXPECT_EQ(claim.inbox_index, 8u);
    EXPECT_EQ(claim.source, SchedulerReadySource::STOLEN);
}

TEST(SchedulerDispatch, RejectsKernelIdBeforeCallableTableAccess) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].kernel_ids[0] = static_cast<uint16_t>(kFixtureCallableCount);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
}

TEST(SchedulerDispatch, RejectsInvalidSingleSubtaskShapeBeforeMetadataIndex) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].active_mask = LaneMask::none();
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, RejectsCoreTypeMismatch) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIV);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, RejectsUnknownTargetCoreType) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    storage.contexts[0].core_type = 2;
    storage.metadata[0].kernel_ids[1] = 1;
    storage.metadata[0].active_mask = LaneMask::aiv0();
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerDispatch, DispatchSequenceIsIndependentOfSlotGenerationAndRejectsZeroCallable) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.scheduler_local_state.slots[0][0].generation, 1u);
    uint32_t pending_slot = UINT32_MAX;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(&storage.scheduler_local_state, &pending_slot, &publication));
    EXPECT_EQ(pending_slot, 0u);
    EXPECT_EQ(publication, scheduler_dispatch_publication(1, SchedulerDispatchSlotState::READY));

    storage.callable_addresses[1] = 0;
    storage.run_control->error_claimed = 0;
    storage.run_control->scheduler_error = 0;
    slot_claim.slot_index = 1;
    ready_claim.task_ref = 1;
    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
}

// The bound is the count this worker context publishes, not the capacity the
// scheduler state used to reserve. Both ids below are mapped to a non-zero
// address, so a zero entry cannot be what refuses the second one: only the
// length can. An implementation that kept the former fixed 1024 bound accepts
// it and fails here.
TEST(SchedulerDispatch, RejectsAKernelIdPastThePublishedCallableCount) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.contexts[0].callable_addresses_count = 2;
    storage.callable_addresses[1] = UINT64_C(0x1000);
    storage.callable_addresses[2] = UINT64_C(0x3000);
    storage.metadata[0].kernel_ids[0] = 1;
    storage.metadata[1].kernel_ids[0] = 2;

    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    auto *payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset
    );
    EXPECT_EQ(payload->function_bin_addr, UINT64_C(0x1000)) << "the last id inside the count still dispatches";

    slot_claim.slot_index = 1;
    ready_claim.task_ref = 1;
    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_CALLABLE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
}

TEST(SchedulerDispatch, AcceptsLastCallableAndInlineSentinel) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].kernel_ids[0] = static_cast<uint16_t>(kFixtureCallableCount - 1);
    storage.callable_addresses[kFixtureCallableCount - 1] = UINT64_C(0x2000);
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    auto *payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset
    );
    EXPECT_EQ(payload->function_bin_addr, UINT64_C(0x2000));

    storage.metadata[1].kernel_ids[0] = UINT16_MAX;
    storage.metadata[1].flags |= SCHEDULER_TASK_INLINE;
    slot_claim.slot_index = 1;
    ready_claim.task_ref = 1;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset + sizeof(DispatchPayload)
    );
    EXPECT_EQ(payload->function_bin_addr, 0u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerPredicate, DistinguishesFailedAndMalformedPredicates) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    alignas(8) int64_t value = 5;
    struct PredicateCase {
        PredicateOp op;
        int64_t pass_target;
        int64_t fail_target;
    };
    constexpr std::array<PredicateCase, 6> cases{{
        {PredicateOp::EQ, 5, 6},
        {PredicateOp::NE, 6, 5},
        {PredicateOp::GT, 4, 5},
        {PredicateOp::LT, 6, 5},
        {PredicateOp::GE, 5, 6},
        {PredicateOp::LE, 5, 4},
    }};
    constexpr std::array<uint8_t, 4> element_sizes{1, 2, 4, 8};
    for (uint8_t element_size : element_sizes) {
        for (const PredicateCase &predicate_case : cases) {
            graph.predicate(
                0, reinterpret_cast<uint64_t>(&value), element_size, static_cast<uint8_t>(predicate_case.op),
                predicate_case.pass_target
            );
            EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::PASS);
            graph.predicate(
                0, reinterpret_cast<uint64_t>(&value), element_size, static_cast<uint8_t>(predicate_case.op),
                predicate_case.fail_target
            );
            EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::FAIL);
        }
    }

    graph.predicate(0, 0, 0, static_cast<uint8_t>(PredicateOp::NONE));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::PASS);

    graph.predicate(0, 0, 8, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 3, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 8, UINT8_C(0xff));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
    graph.predicate(0, reinterpret_cast<uint64_t>(&value) + 1, 8, static_cast<uint8_t>(PredicateOp::GT));
    EXPECT_EQ(scheduler_evaluate_task_predicate(graph.graph(), 0), SchedulerPredicateResult::MALFORMED);
}

TEST(SchedulerPredicate, MalformedPredicateStopsDispatchWithoutPublishingSlot) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    graph.predicate(0, 0, 4, static_cast<uint8_t>(PredicateOp::GT));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    EXPECT_FALSE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_PREDICATE));
    EXPECT_NE(storage.scheduler_local_state.slots[0][0].state, SchedulerDispatchSlotState::READY);
}

TEST(SchedulerPredicate, FailedPredicatePublishesDependencyOnlyDispatch) {
    FixtureStorage storage(1, 1);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    alignas(4) int32_t value = 0;
    graph.predicate(0, reinterpret_cast<uint64_t>(&value), 4, static_cast<uint8_t>(PredicateOp::GT));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[0].flags |= SCHEDULER_TASK_HAS_PREDICATE;
    SchedulerFreeSlotClaim slot_claim{0, 0, 0, 0};
    SchedulerReadyClaim ready_claim{};
    ready_claim.task_ref = 0;

    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.scheduler_local_state), storage.run_control, slot_claim,
        ready_claim, 0, storage.ssbuf_region
    ));
    const auto *payload = scheduler_state_at<DispatchPayload>(
        storage.scheduler_state->base(), storage.contexts[0].dispatch_payload_offset
    );
    EXPECT_EQ(payload->function_bin_addr, 0u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerReadyInbox, ConcurrentConsumersNeverDuplicateTask) {
    constexpr uint64_t kTasks = 8192;
    constexpr uint64_t kConsumerCount = 8;
    FixtureStorage storage(kTasks, kConsumerCount);
    GraphBuffer graph(kTasks);
    SchedulerReadyBatch batch{};
    for (uint64_t task = 0; task < kTasks; ++task) {
        graph.executable(task, 0);
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), task, &batch, nullptr
        ));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &storage.owner_states[0]), 0,
        &batch, nullptr
    ));
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    std::atomic<bool> failed{false};
    auto consume = [&](uint64_t worker) {
        while (!failed.load(std::memory_order_relaxed) && claimed.load(std::memory_order_relaxed) < kTasks) {
            int64_t task = SCHEDULER_TASK_REF_INVALID;
            if (!scheduler_ready_pop_from_inbox(
                    graph.graph(), storage.scheduler_state->base(),
                    storage.local_context(&storage.contexts[worker], nullptr), storage.run_control, 0, 0, &task, nullptr
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    };
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumerCount);
    for (uint64_t worker = 0; worker < kConsumerCount; ++worker)
        consumers.emplace_back(consume, worker);
    for (auto &consumer : consumers)
        consumer.join();
    EXPECT_FALSE(failed.load());
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(SchedulerReadyInbox, OwnerPushRacesThiefWithoutLosingTasks) {
    constexpr uint64_t kTasks = 2048;
    constexpr uint64_t kSpinLimit = 10000000;
    FixtureStorage storage(kTasks, 2);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    SchedulerLocalState &owner_state = storage.owner_states[0];
    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    std::atomic<bool> failed{false};

    std::thread owner([&] {
        for (uint64_t task = 0; task < kTasks && !failed.load(std::memory_order_relaxed); ++task) {
            SchedulerReadyBatch batch{};
            if (!scheduler_ready_batch_append(
                    storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), task, &batch, nullptr
                ) ||
                !scheduler_ready_batch_push(
                    storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state), 0,
                    &batch, nullptr
                ) ||
                !scheduler_refresh_ready_inbox_type(
                    storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state), 0
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }
        for (uint64_t spin = 0; spin < kSpinLimit && claimed.load(std::memory_order_relaxed) < kTasks; ++spin) {
            if (!scheduler_refresh_ready_inbox_type(
                    storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state), 0
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            std::this_thread::yield();
        }
        if (claimed.load(std::memory_order_relaxed) != kTasks) failed.store(true, std::memory_order_relaxed);
    });
    std::thread thief([&] {
        for (uint64_t spin = 0; spin < kSpinLimit && !failed.load(std::memory_order_relaxed) &&
                                claimed.load(std::memory_order_relaxed) < kTasks;
             ++spin) {
            int64_t task = SCHEDULER_TASK_REF_INVALID;
            if (!scheduler_ready_pop_from_inbox(
                    graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[1]),
                    storage.run_control, 0, 0, &task, nullptr
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
            std::this_thread::yield();
        }
    });
    owner.join();
    thief.join();

    EXPECT_FALSE(failed.load());
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

// The inbox head is a Treiber stack any core may push onto, so the directory
// bit cannot simply follow the owner's view of its own head. That bit is the
// only thing that tells a consumer -- the owner's own slot fill included -- to
// look at the inbox at all, so a push landing between the owner's head load and
// its clear must still leave the bit set once both have run.
TEST(SchedulerReadyInbox, ConcurrentPushesSurviveOwnerDirectoryClear) {
    constexpr uint64_t kTasks = 2048;
    constexpr uint64_t kPusherCount = 2;
    constexpr uint64_t kWorkers = 3;
    constexpr uint64_t kSpinLimit = 20000000;
    FixtureStorage storage(kTasks, kWorkers);
    GraphBuffer graph(kTasks);
    for (uint64_t task = 0; task < kTasks; ++task)
        graph.executable(task, 0);
    for (uint64_t worker = 0; worker < kWorkers; ++worker)
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
            storage.contexts[worker].cluster_worker_ids[lane] = lane;

    std::vector<std::atomic<uint32_t>> seen(kTasks);
    std::atomic<uint64_t> claimed{0};
    std::atomic<bool> failed{false};

    std::vector<std::thread> pushers;
    pushers.reserve(kPusherCount);
    for (uint64_t pusher = 0; pusher < kPusherCount; ++pusher) {
        pushers.emplace_back([&, pusher] {
            auto *local = storage.local_context(&storage.contexts[1 + pusher], &storage.owner_states[1 + pusher]);
            for (uint64_t task = pusher; task < kTasks && !failed.load(std::memory_order_relaxed);
                 task += kPusherCount) {
                SchedulerReadyBatch batch{};
                if (!scheduler_ready_batch_append(
                        storage.scheduler_state->base(), local, static_cast<int64_t>(task), &batch, nullptr
                    ) ||
                    !scheduler_ready_batch_push(storage.scheduler_state->base(), local, 0, &batch, nullptr)) {
                    failed.store(true, std::memory_order_relaxed);
                    return;
                }
                std::this_thread::yield();
            }
        });
    }
    std::thread owner([&] {
        auto *local = storage.local_context(&storage.contexts[0], &storage.owner_states[0]);
        for (uint64_t spin = 0; spin < kSpinLimit && !failed.load(std::memory_order_relaxed) &&
                                claimed.load(std::memory_order_relaxed) < kTasks;
             ++spin) {
            if (!scheduler_refresh_ready_inbox_type(storage.scheduler_state->base(), local, 0)) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            // Gate on the directory exactly as a slot fill does: a bit lost to
            // the clear hides the inbox from its owner and strands the work.
            if (!scheduler_ready_directory_nonempty(storage.scheduler_state->base(), local, kWorkers, 0)) {
                std::this_thread::yield();
                continue;
            }
            int64_t task = SCHEDULER_TASK_REF_INVALID;
            if (!scheduler_ready_pop_from_inbox(
                    graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &task, nullptr
                )) {
                failed.store(true, std::memory_order_relaxed);
                return;
            }
            if (task >= 0) {
                seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
                claimed.fetch_add(1, std::memory_order_relaxed);
            }
        }
    });
    for (auto &pusher : pushers)
        pusher.join();
    owner.join();

    EXPECT_FALSE(failed.load());
    EXPECT_EQ(claimed.load(), kTasks);
    for (const auto &count : seen)
        EXPECT_EQ(count.load(), 1u);
}

TEST(SchedulerReadyWake, ConcurrentRegistrationAndCloseResolveEveryConsumerExactlyOnce) {
    constexpr uint64_t kConsumers = 64;
    FixtureStorage storage(kConsumers + 1, 1);
    GraphBuffer graph(kConsumers + 1);
    graph.executable(0, 0);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        graph.executable(task, 0, {0});
        storage.metadata[task].flags |= SCHEDULER_TASK_HAS_FANIN;
    }
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    std::vector<std::atomic<uint32_t>> seen(kConsumers + 1);
    std::atomic<uint64_t> started{0};
    std::atomic<bool> failed{false};
    SchedulerLocalState *reader = storage.local_context(&storage.contexts[0]);
    SchedulerLocalState *owner = storage.local_context(&storage.contexts[0], &storage.owner_states[0]);
    std::vector<std::thread> consumers;
    consumers.reserve(kConsumers);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        consumers.emplace_back([&, task] {
            started.fetch_add(1, std::memory_order_release);
            const SchedulerRouteResult route = scheduler_route_task(
                graph.graph(), storage.scheduler_state->base(), reader, storage.run_control, static_cast<int64_t>(task),
                nullptr
            );
            if (route == SchedulerRouteResult::READY_TO_ENQUEUE) {
                seen[task].fetch_add(1, std::memory_order_relaxed);
            } else if (route != SchedulerRouteResult::WAITING) {
                failed.store(true, std::memory_order_relaxed);
            }
        });
    }
    std::thread closer([&] {
        while (started.load(std::memory_order_acquire) != kConsumers)
            std::this_thread::yield();
        scheduler_gm_store(controls[0].state, static_cast<int64_t>(SchedulerTaskState::DONE));
        if (!scheduler_resolve_completion(
                graph.graph(), storage.scheduler_state->base(), owner, storage.run_control, 0, nullptr, nullptr,
                nullptr, 0, true, nullptr, UINT32_MAX
            ))
            failed.store(true, std::memory_order_relaxed);
    });
    for (auto &consumer : consumers)
        consumer.join();
    closer.join();

    while (!failed.load(std::memory_order_relaxed)) {
        int64_t task = SCHEDULER_TASK_REF_INVALID;
        ASSERT_TRUE(scheduler_ready_pop_from_inbox(
            graph.graph(), storage.scheduler_state->base(), reader, storage.run_control, 0, 0, &task, nullptr
        ));
        if (task == SCHEDULER_TASK_REF_INVALID) break;
        seen[static_cast<size_t>(task)].fetch_add(1, std::memory_order_relaxed);
    }
    EXPECT_FALSE(failed.load());
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
    for (uint64_t task = 1; task <= kConsumers; ++task) {
        EXPECT_EQ(seen[task].load(), 1u) << "consumer=" << task;
        EXPECT_EQ(controls[task].next_fanin_index, 0) << "consumer=" << task;
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(SchedulerTaskState::BLOCKED)) << "consumer=" << task;
    }
}

TEST(SchedulerReadyWake, WakeResolvePublishesConsumerToSchedulerLocalInbox) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    SchedulerWakeStats wake{};
    SchedulerReadyStats ready{};
    SchedulerCompletionStats completion{};
    EXPECT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 1, &wake
        ),
        SchedulerRouteResult::WAITING
    );
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.owner_states[0]), storage.run_control, 0, &wake, &ready,
        &completion, 0, true, nullptr, UINT32_MAX
    ));
    EXPECT_EQ(completion.resolve_count, 1u);
    int64_t task = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
        storage.run_control, 0, 0, &task, &ready
    ));
    EXPECT_EQ(task, 1);
    EXPECT_EQ(wake.wake_register_count, 1u);
    EXPECT_EQ(wake.wake_migrate_count, 1u);
}

// A push appends, so work a completion resolves is consumed after work that was
// already queued: the queue is one FIFO per core and type, with no second bank
// to promote between the two pops.
TEST(SchedulerReadyWake, WakeResolveQueuesBehindOlderPublishedWork) {
    FixtureStorage storage(3, 1);
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0);
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    SchedulerWakeStats wake{};
    SchedulerReadyStats ready{};
    SchedulerCompletionStats completion{};
    SchedulerLocalState &owner_state = storage.owner_states[0];
    EXPECT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 1, &wake
        ),
        SchedulerRouteResult::WAITING
    );
    SchedulerReadyBatch older{};
    ASSERT_TRUE(scheduler_ready_batch_append(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]), 2, &older, &ready
    ));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state), 0, &older, &ready
    ));
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);
    ASSERT_TRUE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0], &owner_state),
        storage.run_control, 0, &wake, &ready, &completion, 0, true, nullptr, UINT32_MAX
    ));
    EXPECT_EQ(completion.resolve_count, 1u);

    int64_t task = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
        storage.run_control, 0, 0, &task, &ready
    ));
    EXPECT_EQ(task, 2);
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
        storage.run_control, 0, 0, &task, &ready
    ));
    EXPECT_EQ(task, 1);
}

TEST(SchedulerReadyWake, RejectsInvalidWaiterShapeBeforeReadyBatchIndex) {
    FixtureStorage storage(2, 1);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].active_mask = LaneMask::none();
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_EQ(
        scheduler_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&storage.contexts[0]),
            storage.run_control, 1, nullptr
        ),
        SchedulerRouteResult::WAITING
    );
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(storage.scheduler_state->base(), storage.layout.task_controls_offset);
    controls[0].state = static_cast<int64_t>(SchedulerTaskState::DONE);

    EXPECT_FALSE(scheduler_resolve_completion(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&storage.contexts[0], &storage.owner_states[0]), storage.run_control, 0, nullptr, nullptr,
        nullptr, 0, true, nullptr, UINT32_MAX
    ));
    EXPECT_EQ(storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::UNSUPPORTED_SHAPE));
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_INVALID_SHAPE));
}

}  // namespace
