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
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "scheduler/scheduler_mix.h"
// Not only for the cases below: this is the one build that compiles the SPMD
// claim path for the host toolchain at all. Without it the device build is the
// first thing to see a change here, and a failure there is a wheel that does
// not build rather than a test that fails.
#include "scheduler/scheduler_spmd.h"
#include "runtime_types.h"
#include "hbg_scheduler_test_support.h"

namespace {

using scheduler_test::FixtureStorage;
using scheduler_test::GraphBuffer;
using scheduler_test::SchedulerStateBuffer;

uint32_t test_cluster_lane(const SchedulerWorkerContext &scheduler, uint64_t worker_id) {
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        if (scheduler.cluster_worker_ids[lane] == worker_id) return lane;
    }
    return static_cast<uint32_t>(worker_id);
}

// The position this core runs from. A core has one, so the worker and the slot
// index a caller passes are only there to say which core it means -- and in
// this fixture there is one local state, so they say nothing further.
SchedulerLocalSlotState *
test_slot(FixtureStorage &storage, const SchedulerWorkerContext &scheduler, uint64_t worker_id, uint32_t pending_slot) {
    (void)scheduler;
    (void)worker_id;
    (void)pending_slot;
    return &storage.scheduler_local_state.running;
}

void configure_normal_aiv_cluster(FixtureStorage &storage, uint64_t task_count) {
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    storage.run_control->scheduler_count = 1;
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;
    scheduler_initialize_free_slot(&storage.scheduler_local_state.running);
    for (uint64_t task = 0; task < task_count; ++task) {
        storage.metadata[task].kernel_ids[0] = UINT16_MAX;
        storage.metadata[task].kernel_ids[1] = 1;
        storage.metadata[task].active_mask = 2;
        storage.metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
    }
}

void enqueue_normal_aiv_tasks(
    FixtureStorage &storage, SchedulerWorkerContext &scheduler, uint64_t task_begin, uint64_t task_end
) {
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    for (uint64_t task = task_begin; task < task_end; ++task) {
        auto *control = scheduler_task_control_at(
            storage.scheduler_state->base(), storage.local_context(&scheduler), static_cast<int64_t>(task)
        );
        control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), storage.local_context(&scheduler), static_cast<int64_t>(task), &batch,
            &ready_stats
        ));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state),
        TaskType::AIV, &batch, &ready_stats
    ));
}

void occupy_normal_slot(
    FixtureStorage &storage, SchedulerWorkerContext &scheduler, uint64_t worker_id, uint32_t pending_slot,
    int64_t task_ref
) {
    SchedulerLocalSlotState &local_slot = *test_slot(storage, scheduler, worker_id, pending_slot);
    local_slot.task_ref = task_ref;
    local_slot.subtask_slot = 1;
    local_slot.state = SchedulerDispatchSlotState::READY;
    local_slot.generation = ++storage.scheduler_local_state.dispatch_sequence;
    scheduler_local_ready_publish(&storage.scheduler_local_state);
}

// A completed task in one of this core's own slots. A core fills and retires
// only the slots it executes from, so the completion arrives in local state and
// nothing about it crosses the cluster's SSBUF. `scheduler` must therefore be
// the lane-0 context, which is the lane the slot belongs to.
SchedulerLocalSlotState *prepare_completed_normal_slot(FixtureStorage &storage, SchedulerWorkerContext &scheduler) {
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        scheduler.cluster_worker_ids[lane] = lane;
    storage.run_control->scheduler_count = 1;
    SchedulerLocalSlotState &local_slot = storage.scheduler_local_state.running;
    scheduler_initialize_free_slot(&local_slot);
    local_slot.task_ref = 0;
    local_slot.subtask_slot = 0;
    local_slot.state = SchedulerDispatchSlotState::READY;
    local_slot.generation = ++storage.scheduler_local_state.dispatch_sequence;
    storage.scheduler_local_state.local_completed_generation = local_slot.generation;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    return &local_slot;
}

TEST(SchedulerSsbufCompletionInbox, PacksBothGenerationSlotsInOneDeviceWord) {
    SchedulerSsbufCompletionInbox inbox{};
    uint64_t shadow = scheduler_ssbuf_update_completion(0, 0, UINT32_C(0x11223344));
    shadow = scheduler_ssbuf_update_completion(shadow, 1, UINT32_C(0x55667788));
    scheduler_cache_barrier();
    scheduler_ssbuf_store_relaxed(&inbox.publication, shadow);
    EXPECT_EQ(scheduler_ssbuf_load_relaxed(&inbox.publication), UINT64_C(0x5566778811223344));
    shadow = scheduler_ssbuf_update_completion(shadow, 0, 1);
    scheduler_cache_barrier();
    scheduler_ssbuf_store_relaxed(&inbox.publication, shadow);
    EXPECT_EQ(scheduler_ssbuf_load_relaxed(&inbox.publication), UINT64_C(0x5566778800000001));
}

// A lane's own completions and the Mix subtask completions it reports are two
// sequences, tracked apart: consuming one must not make the other look stale.
TEST(SchedulerSsbufCompletionInbox, TracksOwnAndMixConsumedGenerationsApart) {
    SchedulerLocalState scheduler_local_state{};
    EXPECT_FALSE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 0));
    EXPECT_TRUE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 1));
    scheduler_mark_completion_consumed(&scheduler_local_state, 0, 1);
    EXPECT_FALSE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 1));
    EXPECT_TRUE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 2));
    // The same generation on the Mix side is still unseen.
    EXPECT_TRUE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 1, true));
    scheduler_mark_completion_consumed(&scheduler_local_state, 0, 1, true);
    EXPECT_FALSE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 1, true));
    EXPECT_FALSE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 1));
}

TEST(SchedulerSsbufCompletionInbox, ReadyWordPreservesSignedTimingSlotAndGeneration) {
    for (int32_t timing_slot : {-1, 0, 7}) {
        const uint64_t word = scheduler_ssbuf_pack_ready(UINT32_MAX, timing_slot);
        EXPECT_EQ(static_cast<uint32_t>(word), UINT32_MAX);
        EXPECT_EQ(static_cast<int32_t>(word >> 32), timing_slot);
    }
}

// A core services one own slot per pass, so the three outcomes are driven one
// at a time: a clean completion, a resolve that fails, and a generation the
// slot never published.
TEST(SchedulerClusterCompletion, CountsACompletionAndTakesBothKindsOfErrorExit) {
    for (int failure = 0; failure < 3; ++failure) {
        SCOPED_TRACE(failure);
        FixtureStorage storage(2, 2);
        GraphBuffer graph(2);
        graph.executable(0, 0);
        graph.executable(1, 0);
        auto &scheduler = storage.contexts[0];
        prepare_completed_normal_slot(storage, scheduler);
        auto *control =
            scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
        if (failure == 1) control->wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;
        if (failure == 2) storage.scheduler_local_state.local_completed_generation += 1;
        EXPECT_EQ(
            scheduler_service_cluster_completions(
                graph.graph(), storage.scheduler_state->base(),
                storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, nullptr,
                nullptr, nullptr, 0, nullptr, storage.ssbuf_region
            ),
            failure == 0
        );
        // A resolve that fails still counts the task, so resolved_task_count
        // stays consistent with what the run dispatched; a generation the slot
        // never published is rejected before that and counts nothing.
        EXPECT_EQ(storage.scheduler_local_state.pending_completed, failure == 2 ? 0u : 1u);
        EXPECT_EQ(storage.run_control->scheduler_error == 0, failure == 0);
        if (failure == 0) EXPECT_EQ(scheduler_consumed_completion_generation(&storage.scheduler_local_state, 0, 0), 1u);
    }
}

TEST(SchedulerClusterCompletion, SpscGenerationCompletesNormalTask) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = UINT64_MAX;
    auto *slot = test_slot(storage, scheduler, 0, 0);
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.running;
    scheduler_initialize_free_slot(local_slot);
    slot->task_ref = 0;
    local_slot->task_ref = 0;
    local_slot->subtask_slot = 0;
    storage.scheduler_local_state.set_timing_slot(0, 0);
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->generation = ++storage.scheduler_local_state.dispatch_sequence;
    // A core's own slot keeps its executor trace and its completion in local
    // state; neither crosses the cluster's SSBUF.
    auto *executor_trace = &storage.scheduler_local_state.profiling->executor_trace;
    executor_trace->kernel_start_cycles = 100;
    executor_trace->kernel_end_cycles = 200;
    storage.metadata[0].timing_slot = 0;
    storage.scheduler_local_state.local_completed_generation = local_slot->generation;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, 0, nullptr, storage.ssbuf_region
    ));
    // The generation is consumed from local state, and the slot is released.
    EXPECT_EQ(scheduler_consumed_completion_generation(&storage.scheduler_local_state, 0, 0), slot->generation);
    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    EXPECT_FALSE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, 0, nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[0].kernel_start_cycles, 100u);
    EXPECT_EQ(traces[0].kernel_end_cycles, 200u);
    EXPECT_EQ(traces[0].valid, 0u);
}

TEST(SchedulerClusterCompletion, UnprofiledSelfHandoffNeedsNoProfilingStorage) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    storage.metadata[0].timing_slot = -1;
    auto *local = storage.local_context(&storage.contexts[1], &storage.scheduler_local_state);
    local->profiling = nullptr;
    auto &slot = local->running;
    scheduler_initialize_free_slot(&slot);
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control,
        SchedulerFreeSlotClaim{1, slot.generation, 1}, ready, 0, storage.ssbuf_region
    ));
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(local, &publication));
    EXPECT_EQ(local->timing_slot(1), -1);
    local->local_completed_generation = scheduler_dispatch_generation(publication);
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, nullptr, nullptr, nullptr, 0,
        nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(local->profiling, nullptr);
    EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    scheduler_flush_completions(storage.run_control, local);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

// A core runs one task at a time, so consecutive tasks reuse its one position.
// Each must still carry its own sampled trace, and the sampling slot must be
// released between them rather than leaking into the next.
TEST(SchedulerClusterCompletion, ConsecutiveSelfTasksKeepDistinctSampledTraces) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        scheduler.cluster_worker_ids[lane] = lane;
    auto *local = storage.local_context(&scheduler, &storage.scheduler_local_state);
    for (uint32_t index = 0; index < 2; ++index) {
        graph.executable(index, 1);
        auto &slot = local->running;
        scheduler_initialize_free_slot(&slot);
        slot.task_ref = index;
        slot.subtask_slot = 1;
        local->set_timing_slot(1, index);
        slot.state = SchedulerDispatchSlotState::READY;
        slot.generation = ++local->dispatch_sequence;
        storage.metadata[index].active_mask = 2;
        local->profiling->executor_trace.kernel_start_cycles = 100 + index;
        local->profiling->executor_trace.kernel_end_cycles = 200 + index;
        local->local_completed_generation = slot.generation;
        auto *control = scheduler_task_control_at(storage.scheduler_state->base(), local, index);
        control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        storage.metadata[index].timing_slot = index;
        ASSERT_TRUE(scheduler_service_cluster_completions(
            graph.graph(), storage.scheduler_state->base(), local, storage.run_control, nullptr, nullptr, nullptr, 0,
            nullptr, storage.ssbuf_region
        ));
        EXPECT_FALSE(local->sampled_task_timing(1));
    }
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    for (uint32_t index = 0; index < 2; ++index) {
        EXPECT_EQ(traces[index].kernel_start_cycles, 100 + index);
        EXPECT_EQ(traces[index].kernel_end_cycles, 200 + index);
    }
}

TEST(SchedulerClusterCompletion, RejectsStaleCompletionGenerationAtNamedSite) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.cluster_worker_ids[0] = 0;
    auto *slot = test_slot(storage, scheduler, 0, 0);
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.running;
    scheduler_initialize_free_slot(local_slot);
    slot->task_ref = 0;
    local_slot->task_ref = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->generation = ++storage.scheduler_local_state.dispatch_sequence;

    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, slot->generation + 1,
        &wake_stats, &ready_stats, &completion_stats, false, nullptr, nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH)
    );
}

TEST(SchedulerClusterCompletion, AccountsCompletedTaskWhenResolveFails) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    SchedulerLocalSlotState *slot = prepare_completed_normal_slot(storage, scheduler);
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;

    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, slot->generation,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    EXPECT_NE(storage.run_control->scheduler_error, 0u);
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED)
    );
}

TEST(SchedulerClusterCompletion, CompletionIgnoresTheReadyQueueEntirely) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    SchedulerLocalSlotState *slot = prepare_completed_normal_slot(storage, scheduler);
    // One endpoint naming a task while the other is empty is a corrupt inbox.
    // A completion no longer claims a replacement from the ready queue, so a
    // queue it could not read is not its problem: the slot is released and the
    // task accounted regardless, and the next fill pass is what meets the
    // corruption.
    scheduler_ready_inbox_at(
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state),
        TaskType::AIC, 0
    )
        ->endpoints = scheduler_ready_endpoints_pack(SCHEDULER_INBOX_EMPTY, 0);

    EXPECT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, slot->generation,
        nullptr, nullptr, nullptr, 0, nullptr, nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    EXPECT_EQ(slot->state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
}

TEST(SchedulerClusterCompletion, AccountsCompletedTaskWhenRefillDispatchFails) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    SchedulerLocalSlotState *slot = prepare_completed_normal_slot(storage, scheduler);
    SchedulerReadyClaim replacement{};
    replacement.task_ref = 1;

    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, slot->generation,
        nullptr, nullptr, nullptr, 0, &replacement, nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    EXPECT_NE(storage.run_control->scheduler_error, 0u);
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_REFILL_DISPATCH_FAILED)
    );
}

TEST(SchedulerClusterCompletion, PropagatesTraceToCompletionAndWokenTask) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;
    auto *producer = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    producer->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerWakeStats wake_stats{};
    ASSERT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler), storage.run_control, 1,
            &wake_stats
        ),
        SchedulerRouteResult::WAITING
    );
    auto *slot = test_slot(storage, scheduler, 0, 0);
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.running;
    scheduler_initialize_free_slot(local_slot);
    slot->task_ref = 0;
    local_slot->task_ref = 0;
    local_slot->subtask_slot = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->generation = ++storage.scheduler_local_state.dispatch_sequence;
    auto *executor_trace = &storage.scheduler_local_state.profiling->executor_trace;
    executor_trace->kernel_start_cycles = 100;
    executor_trace->kernel_end_cycles = 200;
    storage.scheduler_local_state.local_completed_generation = local_slot->generation;
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    producer->completion_resolve_start_cycles = 1;
    producer->completion_resolve_end_cycles = 1;
    producer->scheduler_worker_id = UINT64_MAX;
    traces[1].ready_transition_cycles = 1;

    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(producer->completion_resolve_start_cycles, 0u);
    EXPECT_EQ(producer->completion_resolve_end_cycles, 0u);
    EXPECT_EQ(producer->scheduler_worker_id, scheduler.worker_index);
    EXPECT_EQ(traces[1].ready_transition_cycles, 0u);
    EXPECT_EQ(traces[0].valid, 1u);
    EXPECT_EQ(traces[0].kernel_start_cycles, 100u);
    EXPECT_EQ(traces[0].kernel_end_cycles, 200u);
    auto *waiter = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1);
    EXPECT_EQ(waiter->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    // The core owns the slot it just freed, so the successor it woke goes
    // straight back into it instead of round-tripping through the ready queue.
    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(local_slot->task_ref, 1);
    EXPECT_EQ(ready_stats.enqueue_count, 0u);
    EXPECT_EQ(wake_stats.fanin_state_load_count, 0u);
}

// Completing a slot releases it and leaves queued work in the queue. A core
// fills only slots it executes from, so there is no peer slot to refill and
// nothing claims a replacement on the completion path; the next fill pass is
// what takes the queued task.
TEST(SchedulerClusterCompletion, ReleasesCompletedSlotAndLeavesReadyTaskQueued) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;

    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.running;
    scheduler_initialize_free_slot(local_slot);
    local_slot->generation = ++storage.scheduler_local_state.dispatch_sequence;
    const uint32_t completed_generation = local_slot->generation;
    local_slot->task_ref = 0;
    local_slot->subtask_slot = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    storage.scheduler_local_state.local_completed_generation = completed_generation;
    auto *completed_control =
        scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completed_control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    auto *ready_control =
        scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1);
    ready_control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(
        storage.scheduler_state->base(), storage.local_context(&scheduler), 1, &batch, &ready_stats
    ));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state),
        TaskType::AIC, &batch, &ready_stats
    ));

    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    bool direct_refilled = false;
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, &direct_refilled, storage.ssbuf_region
    ));

    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(local_slot->task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(local_slot->generation, completed_generation);
    EXPECT_EQ(completed_control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    EXPECT_FALSE(direct_refilled);
    // Task 1 is still where it was pushed, claimable by the next fill pass.
    int64_t queued = SCHEDULER_TASK_REF_INVALID;
    ASSERT_TRUE(scheduler_ready_pop_from_inbox(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, TaskType::AIC, 0,
        &queued, &ready_stats
    ));
    EXPECT_EQ(queued, 1);
}

TEST(SchedulerClusterCompletion, UsesSchedulerLocalSlotStateWithoutRereadingDispatchMetadata) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;

    SchedulerLocalState scheduler_local_state{};
    SchedulerLocalSlotState *local_slot = &scheduler_local_state.running;
    scheduler_initialize_free_slot(local_slot);
    SchedulerReadyClaim producer_ready{};
    producer_ready.task_ref = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, SchedulerFreeSlotClaim{0, local_slot->generation, 0}, producer_ready,
        SCHEDULER_PROFILING_TASK_TIMING_LEVEL, storage.ssbuf_region
    ));
    ASSERT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler), storage.run_control, 1,
            nullptr
        ),
        SchedulerRouteResult::WAITING
    );

    const uint32_t completed_generation = local_slot->generation;
    auto *executor_trace = &scheduler_local_state.profiling->executor_trace;
    executor_trace->kernel_start_cycles = 100;
    executor_trace->kernel_end_cycles = 200;
    scheduler_local_state.local_completed_generation = completed_generation;

    SchedulerReadyStats ready_stats{};
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, nullptr, &ready_stats, nullptr, SCHEDULER_PROFILING_TASK_TIMING_LEVEL, nullptr,
        storage.ssbuf_region
    ));

    // The woken successor refills the slot directly; the poisoned SSBUF
    // metadata above was never consulted, which is what this covers.
    EXPECT_EQ(local_slot->task_ref, 1);
    EXPECT_EQ(local_slot->generation, completed_generation + 1);
    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(ready_stats.enqueue_count, 0u);
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[0].kernel_start_cycles, 100u);
    EXPECT_EQ(traces[0].kernel_end_cycles, 200u);
}

TEST(SchedulerClusterCompletion, CachesStableWorkerTraceAfterFirstCompletion) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    SchedulerWorkerContext &target = storage.contexts[0];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;
    target.trace_aicore_entry_cycles = 11;
    target.trace_handshake_publish_cycles = 12;
    target.trace_register_release_cycles = 13;
    target.trace_descriptor_cache_observed_cycles = 14;

    // A core has one slot, so the two completions are two rounds on it. The
    // cache is what the second round must read: the target reports new numbers
    // by then, and the trace still has to carry the first round's.
    SchedulerLocalState scheduler_local_state{};
    SchedulerLocalSlotState *local_slot = &scheduler_local_state.running;
    for (int64_t task_ref = 0; task_ref < 2; ++task_ref) {
        if (task_ref == 1) {
            target.trace_aicore_entry_cycles = 21;
            target.trace_handshake_publish_cycles = 22;
            target.trace_register_release_cycles = 23;
            target.trace_descriptor_cache_observed_cycles = 24;
        }
        scheduler_initialize_free_slot(local_slot);
        SchedulerReadyClaim ready{};
        ready.task_ref = task_ref;
        ASSERT_TRUE(scheduler_fill_dispatch_slot(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
            storage.run_control, SchedulerFreeSlotClaim{0, local_slot->generation, 0}, ready,
            SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, storage.ssbuf_region
        ));
        auto *executor_trace = &scheduler_local_state.profiling->executor_trace;
        executor_trace->kernel_start_cycles = 100 + task_ref;
        executor_trace->kernel_end_cycles = 200 + task_ref;

        ASSERT_TRUE(scheduler_service_cluster_completion_slot(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
            storage.run_control, 0, local_slot->generation, nullptr, nullptr, nullptr,
            SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, nullptr, nullptr, storage.ssbuf_region
        ));
        ASSERT_TRUE((scheduler_local_state.profiling->worker_trace_valid_mask & 1U) != 0);
    }

    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[1].aicore_entry_cycles, 11u);
    EXPECT_EQ(traces[1].handshake_publish_cycles, 12u);
    EXPECT_EQ(traces[1].register_release_cycles, 13u);
    EXPECT_EQ(traces[1].descriptor_cache_observed_cycles, 14u);
}

TEST(SchedulerClusterCompletion, RefillWithSuppliedClaimPreservesStateProbeAndReadySource) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    SchedulerLocalSlotState *slot = prepare_completed_normal_slot(storage, scheduler);
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;

    SchedulerReadyClaim replacement{};
    replacement.task_ref = 1;
    replacement.source = SchedulerReadySource::STOLEN;
    replacement.state_probe_start_cycles = 123;
    replacement.state_probe_end_cycles = 456;
    bool refilled = false;

    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, slot->generation,
        nullptr, nullptr, nullptr, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, &replacement, &refilled, storage.ssbuf_region
    ));

    EXPECT_TRUE(refilled);
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[1].ready_source, static_cast<uint64_t>(SchedulerReadySource::STOLEN));
    EXPECT_EQ(traces[1].publication_mode, static_cast<uint64_t>(SchedulerPublicationMode::REFILL));
    EXPECT_EQ(traces[1].state_probe_start_cycles, 123u);
    EXPECT_EQ(traces[1].state_probe_end_cycles, 456u);
}

TEST(SchedulerNormalDispatch, FillsItsOwnFreshAicSlot) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.core_type = static_cast<int32_t>(CoreType::AIC);
    scheduler.is_scheduler = 1;
    scheduler.scheduler_count = 1;
    scheduler.scheduler_index = 0;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        scheduler.cluster_worker_ids[lane] = lane;
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;
    auto *local = storage.local_context(&scheduler, &storage.scheduler_local_state);
    ASSERT_EQ(local->config.self_lane, 0u);
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), local, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, 0, &batch, &ready_stats));
    ASSERT_TRUE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), local, TaskType::AIC, &batch, &ready_stats)
    );

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    bool failed = true;
    ASSERT_TRUE(scheduler_fill_own_slots(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, victim_cursors, &ready_stats, 0, 0,
        &failed, storage.ssbuf_region
    ));
    EXPECT_FALSE(failed);
    EXPECT_EQ(local->running.task_ref, 0);
    EXPECT_EQ(local->running.state, SchedulerDispatchSlotState::READY);
}

// A publisher sets an owner's directory bit, and only that owner clears it,
// after observing its own inbox empty. This pins what the owner does when the
// bit is clear while its inbox holds work -- the state a clear latched from a
// stale read of the endpoints would leave behind.
TEST(SchedulerNormalDispatch, FindsOwnQueuedWorkWithItsDirectoryBitClear) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.core_type = static_cast<int32_t>(CoreType::AIC);
    scheduler.is_scheduler = 1;
    scheduler.scheduler_count = 1;
    scheduler.scheduler_index = 0;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        scheduler.cluster_worker_ids[lane] = lane;
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;
    auto *local = storage.local_context(&scheduler, &storage.scheduler_local_state);
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), local, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, 0, &batch, &ready_stats));
    ASSERT_TRUE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), local, TaskType::AIC, &batch, &ready_stats)
    );

    // The work is queued and reachable; now drop only the directory bit and
    // the local active mask, leaving the inbox exactly as it is.
    const TaskType task_type = scheduler_self_task_type(local->config.self_core_type);
    scheduler_ready_directory_clear(
        scheduler_ready_directory_at(storage.scheduler_state->base(), local), task_type, local->self_type_rank()
    );
    local->owner_ready_queue_mask = 0;

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    bool failed = true;
    const bool progress = scheduler_fill_own_slots(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, victim_cursors, &ready_stats, 0, 0,
        &failed, storage.ssbuf_region
    );
    EXPECT_FALSE(failed);
    EXPECT_TRUE(progress);
    EXPECT_EQ(local->running.task_ref, 0);
    EXPECT_EQ(local->running.state, SchedulerDispatchSlotState::READY);
}

// A core has one dispatch slot, so a pass fills at most one task and there is
// no "later fill" to preserve earlier progress across. What survives is the
// reporting: a fill that fails names its site and publishes nothing.
TEST(SchedulerNormalDispatch, AFailedFillNamesItsSiteAndPublishesNothing) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.mixed(1, 3);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.core_type = static_cast<int32_t>(CoreType::AIC);
    scheduler.is_scheduler = 1;
    scheduler.scheduler_count = 1;
    scheduler.scheduler_index = 0;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        scheduler.cluster_worker_ids[lane] = lane;
    storage.metadata[1].active_mask = 3;
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;
    auto *local = storage.local_context(&scheduler, &storage.scheduler_local_state);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    // Only the Mix-shaped task is queued: an AIC queue holding it is the shape
    // error this exercises, and with one slot a well-formed task ahead of it
    // would simply end the pass before it was reached.
    {
        auto *control = scheduler_task_control_at(storage.scheduler_state->base(), local, 1);
        control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, 1, &batch, &ready_stats));
    }
    ASSERT_TRUE(
        scheduler_ready_batch_push(storage.scheduler_state->base(), local, TaskType::AIC, &batch, &ready_stats)
    );

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    bool failed = false;
    EXPECT_FALSE(scheduler_fill_own_slots(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, victim_cursors, &ready_stats, 0, 0,
        &failed, storage.ssbuf_region
    ));
    EXPECT_TRUE(failed);
    EXPECT_EQ(local->running.state, SchedulerDispatchSlotState::FILLING);
    // The slot was taken out of FREE but never reached READY, so nothing is
    // published for the executor to pick up.
    EXPECT_FALSE(local->local_ready);
    EXPECT_NE(storage.run_control->scheduler_error, 0u);
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerLocalReady, InvalidSlotStateIsNotReconstructedAsReady) {
    SchedulerLocalState local{};
    local.config.self_lane = 1;
    local.running.generation = 7;
    local.running.state = SchedulerDispatchSlotState::FREE;
    scheduler_local_ready_publish(&local);
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(&local, &publication));
    EXPECT_NE(scheduler_dispatch_state(publication), SchedulerDispatchSlotState::READY);
    EXPECT_FALSE(scheduler_local_ready_pop(&local, &publication));
}

// Model READY's fully published physical cluster, including all peers. Every
// core schedules itself, and the cluster's Mix coordinator is its AIC on lane 0.
void configure_cached_cluster(FixtureStorage &storage) {
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        auto &context = storage.contexts[lane];
        context.core_type = static_cast<int32_t>(lane == 0 ? CoreType::AIC : CoreType::AIV);
        context.scheduler_count = 1;
        context.scheduler_index = 0;
        context.is_scheduler = 1;
        for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer)
            context.cluster_worker_ids[peer] = peer;
    }
}

TEST(SchedulerLocalConfig, SnapshotsReadyPublicationAndReloadsOnNextRun) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph_buffer(2);
    const SchedulerGraphView graph = graph_buffer.graph();
    for (uint32_t self_lane : {1u, 2u}) {
        configure_cached_cluster(storage);
        SchedulerLocalState local{};
        auto &context = storage.contexts[self_lane];
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        EXPECT_EQ(local.config.self_lane, self_lane);
        EXPECT_EQ(local.config.worker_ids[0], 0u);
        EXPECT_EQ(scheduler_completion_id(&local, 7), 7 * 3 + self_lane);
        const uint64_t ready_offset = local.config.ready_directory_offset;
        const uint64_t payload_offset = local.dispatch_payload_offset(2);
        context.ready_directory_offset = ready_offset + 128;
        context.scheduler_count = 2;
        context.scheduler_index = 1;
        for (auto lane = 0u; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
            storage.contexts[lane].dispatch_payload_offset += 128;
        EXPECT_EQ(local.config.ready_directory_offset, ready_offset);
        EXPECT_EQ(local.config.scheduler_count, 1u);
        EXPECT_EQ(local.config.scheduler_index, 0u);
        EXPECT_EQ(local.dispatch_payload_offset(2), payload_offset);
        SchedulerLocalState next_run{};
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &next_run));
        EXPECT_EQ(next_run.config.ready_directory_offset, ready_offset + 128);
        EXPECT_EQ(next_run.config.scheduler_count, 2u);
        EXPECT_EQ(next_run.config.scheduler_index, 1u);
        EXPECT_EQ(next_run.dispatch_payload_offset(2), payload_offset + 128);
        EXPECT_EQ(next_run.profiling, nullptr);
        EXPECT_EQ(next_run.pending_completed, 0u);
    }
}

TEST(SchedulerLocalConfig, DetectsMixFromTheGraphSummary) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    const SchedulerGraphView graph = graph_buffer.graph();
    configure_cached_cluster(storage);
    auto *summary =
        scheduler_state_at<SchedulerGraphSummary>(storage.scheduler_state->base(), storage.layout.graph_summary_offset);
    summary->mix_task_count = 0;
    SchedulerLocalState without_mix{};
    ASSERT_TRUE(
        scheduler_initialize_local_config(storage.scheduler_state->base(), &storage.contexts[1], &graph, &without_mix)
    );
    EXPECT_FALSE(without_mix.has_mix);

    summary->mix_task_count = 1;
    SchedulerLocalState with_mix{};
    ASSERT_TRUE(
        scheduler_initialize_local_config(storage.scheduler_state->base(), &storage.contexts[1], &graph, &with_mix)
    );
    EXPECT_TRUE(with_mix.has_mix);
}

TEST(SchedulerLocalConfig, RejectsTruncatedOffsetsAndInconsistentPayloadRoutes) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    const auto graph = graph_buffer.graph();
    configure_cached_cluster(storage);
    auto &context = storage.contexts[2];
    SchedulerLocalState local{};
    EXPECT_EQ(local.worker_id(), UINT64_MAX);
    for (auto field :
         {&SchedulerWorkerContext::worker_contexts_offset, &SchedulerWorkerContext::callable_addresses_count,
          &SchedulerWorkerContext::task_metadata_offset}) {
        const uint64_t original = context.*field;
        context.*field = UINT64_C(1) << 32;
        EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        context.*field = original;
    }
    const uint64_t original = context.dispatch_payload_offset;
    context.dispatch_payload_offset = (UINT64_C(1) << 32) + 2 * sizeof(DispatchPayload);
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.dispatch_payload_offset = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.dispatch_payload_offset = original;
    ++storage.contexts[0].dispatch_payload_offset;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    --storage.contexts[0].dispatch_payload_offset;
    context.worker_reserved = 1;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.worker_reserved = 0;
    for (auto field :
         {&SchedulerWorkerContext::scheduler_ssbuf_reserved0, &SchedulerWorkerContext::scheduler_ssbuf_reserved1,
          &SchedulerWorkerContext::scheduler_ssbuf_reserved2}) {
        context.*field = 1;
        EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        context.*field = 0;
    }
    context.topology_reserved[0] = 1;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.topology_reserved[0] = 0;
    for (auto field : {&SchedulerWorkerContext::mix_payloads_offset, &SchedulerWorkerContext::mix_traces_offset}) {
        const uint64_t saved = context.*field;
        context.*field = 0;
        EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        context.*field = saved;
    }
    storage.contexts[0].core_type = 2;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    // This context is worker 2, and a core has one payload buffer, so the
    // offset the config derives for lane 2 is the one the context carries.
    EXPECT_EQ(local.dispatch_payload_offset(2), original);
}

TEST(SchedulerLocalConfig, RejectsInvalidTopologyBeforeMailboxAccess) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    const SchedulerGraphView graph = graph_buffer.graph();
    configure_cached_cluster(storage);
    auto &context = storage.contexts[2];
    SchedulerLocalState local{};
    context.runtime_worker_count = SCHEDULER_WORKER_CAPACITY + 1;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.runtime_worker_count = 3;
    context.cluster_worker_ids[1] = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.cluster_worker_ids[1] = 1;
    context.is_scheduler = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.is_scheduler = 1;
    context.scheduler_count = SCHEDULER_WORKER_CAPACITY;
    context.scheduler_index = SCHEDULER_WORKER_CAPACITY - 1;
    EXPECT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.scheduler_index = SCHEDULER_WORKER_CAPACITY;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
}

TEST(SchedulerClusterCompletion, DefersGlobalCountUntilFlushAndNeverPublishesTwice) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    graph_buffer.executable(0, 0);
    configure_normal_aiv_cluster(storage, 1);
    auto &context = storage.contexts[0];
    prepare_completed_normal_slot(storage, context);
    SchedulerLocalState *local = storage.local_context(&context, &storage.scheduler_local_state);
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph_buffer.graph(), storage.scheduler_state->base(), local, storage.run_control, nullptr, nullptr, nullptr, 0,
        nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(local->pending_completed, 1u);
    EXPECT_EQ(storage.run_control->resolved_task_count, 0u);
    EXPECT_FALSE(scheduler_service_cluster_completions(
        graph_buffer.graph(), storage.scheduler_state->base(), local, storage.run_control, nullptr, nullptr, nullptr, 0,
        nullptr, storage.ssbuf_region
    ));
    EXPECT_EQ(local->pending_completed, 1u);
    scheduler_flush_completions(storage.run_control, local);
    EXPECT_EQ(local->pending_completed, 0u);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    scheduler_flush_completions(storage.run_control, local);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

TEST(SchedulerNormalDispatch, NoUsableCapacityDoesNotAccessDirectory) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    configure_normal_aiv_cluster(storage, 1);
    auto *local = storage.local_context(&storage.contexts[1], &storage.scheduler_local_state);
    uint64_t cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    // The invalid directory offset makes an accidental directory load fail.
    local->config.ready_directory_offset = UINT64_MAX;
    local->running.state = SchedulerDispatchSlotState::FILLING;
    bool failed = true;
    EXPECT_FALSE(scheduler_fill_own_slots(
        graph_buffer.graph(), storage.scheduler_state->base(), local, storage.run_control, cursors, nullptr, 0, 0,
        &failed, storage.ssbuf_region
    ));
    EXPECT_FALSE(failed);
    const uint32_t own_type = local->config.self_core_type;
    EXPECT_FALSE(scheduler_has_usable_slot(local, own_type, 0));

    // A free slot is capacity unless a completion pass already refilled it.
    local->running.state = SchedulerDispatchSlotState::FREE;
    EXPECT_TRUE(scheduler_has_usable_slot(local, own_type, 0));
    EXPECT_FALSE(scheduler_has_usable_slot(local, own_type, UINT64_C(1) << 0));
    // Work of another type is never this core's capacity.
    EXPECT_FALSE(scheduler_has_usable_slot(local, own_type == 0 ? 1 : 0, 0));
}

}  // namespace

namespace {
// A Mix task is placed by its cluster's AIC, so the Scheduler under test here is
// lane 0. Its own subtask takes one of its own slots; the AIV subtasks take Mix
// slots on lanes 1 and 2, which only this core writes.
struct MixFixture {
    FixtureStorage storage{6, 3, 3};
    GraphBuffer graph{6};
    SchedulerLocalState *local;
    uint64_t cursor{0};
    MixFixture() {
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            auto &context = storage.contexts[lane];
            context.is_scheduler = 1;
            context.scheduler_index = lane;
            context.scheduler_count = PLATFORM_CORES_PER_BLOCKDIM;
            for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer)
                context.cluster_worker_ids[peer] = peer;
        }
        storage.run_control->scheduler_count = PLATFORM_CORES_PER_BLOCKDIM;
        storage.callable_addresses[1] = 0x1000;
        local = storage.local_context(&storage.contexts[0], &storage.scheduler_local_state);
        local->has_mix = true;
        scheduler_initialize_free_slot(&local->running);
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            // Start the rolling index at 1, so the slot a coordinator may fill
            // is 0 and the first placement lands where these cases name it.
            scheduler_ssbuf_store_relaxed(&storage.ssbuf_region->lanes[lane].current, 1);
            // A slot state defaults to EMPTY, so the records of what this
            // coordinator has placed have to be opened explicitly, exactly as
            // it does at bootstrap.
            for (uint32_t slot = 0; slot < SCHEDULER_LANE_SLOT_COUNT; ++slot)
                scheduler_initialize_free_slot(&local->mix_placed[lane][slot]);
        }
        for (uint32_t task = 0; task < 6; ++task)
            set_task(task, 7);
    }
    void set_task(uint32_t task, uint8_t mask) {
        graph.mixed(task, mask);
        auto &metadata = storage.metadata[task];
        metadata.active_mask = mask;
        metadata.flags = SCHEDULER_TASK_EXECUTABLE | (__builtin_popcount(mask) > 1 ? SCHEDULER_TASK_MIX : 0);
        metadata.trace_index_base = task * 3;
        metadata.timing_slot = -1;
        for (uint8_t subtask = 0; subtask < 3; ++subtask)
            metadata.kernel_ids[subtask] = (mask & (1U << subtask)) != 0 ? 1 : UINT16_MAX;
    }
    void enqueue(uint32_t task) {
        SchedulerReadyBatch batch{};
        ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, task, &batch, nullptr));
        ASSERT_TRUE(scheduler_ready_batch_push(
            storage.scheduler_state->base(), local,
            scheduler_task_ready_queue(storage.metadata[task].flags, storage.metadata[task].active_mask), &batch,
            nullptr
        ));
    }
    bool dispatch(bool steal = true) {
        return scheduler_fill_cluster_mix_slots(
            graph.graph(), storage.scheduler_state->base(), local, storage.run_control, &cursor, nullptr, 0, steal,
            storage.ssbuf_region, nullptr
        );
    }
    // Lane 0 is the coordinator, which seats its own subtask in the position it
    // runs from and so has no second slot; an AIV lane has two, and `index`
    // says which. The coordinator's record is indexed the same way the lane
    // indexes the slots themselves.
    SchedulerLocalSlotState &slot(uint32_t lane, uint32_t index) {
        return lane == 0 ? local->running : local->mix_placed[lane][index];
    }
    uint64_t payload_offset(uint32_t lane, uint32_t index) {
        return lane == 0 ? local->dispatch_payload_offset(lane) : local->mix_payload_offset(lane, index != 0);
    }
    uint32_t sequence(uint32_t lane) {
        return lane == 0 ? local->dispatch_sequence : local->mix_dispatch_sequences[lane];
    }
    SchedulerSsbufDispatchControl *control(uint32_t lane, uint32_t index) {
        return &storage.ssbuf_region->lanes[lane].dispatch[index];
    }
    // Stand in for the lane's executor: taking a dispatch up is the flip, and
    // it is what frees the other slot for the coordinator to fill again.
    void take(uint32_t lane, uint32_t index) {
        scheduler_ssbuf_store_relaxed(&storage.ssbuf_region->lanes[lane].current, index);
    }
    bool complete(uint32_t lane, uint32_t index, SchedulerRefillCandidates *candidates = nullptr) {
        if (lane == local->config.self_lane) local->local_ready = false;
        return scheduler_service_cluster_completion_slot(
            graph.graph(), storage.scheduler_state->base(), local, storage.run_control, lane,
            slot(lane, index).generation, nullptr, nullptr, nullptr, 0, nullptr, nullptr, storage.ssbuf_region,
            candidates, lane != 0, static_cast<uint8_t>(index)
        );
    }
};
}  // namespace

TEST(SchedulerMix, MasksHaveIndependentPayloadsAndStablePhysicalContext) {
    for (uint8_t mask : {3, 5, 6, 7}) {
        MixFixture f;
        f.set_task(0, mask);
        f.enqueue(0);
        ASSERT_TRUE(f.dispatch());
        uint32_t published = 0;
        uint64_t payload_addresses[3]{};
        for (uint32_t lane = 0; lane < 3; ++lane) {
            const auto &slot = f.slot(lane, 0);
            if (slot.state != SchedulerDispatchSlotState::READY) continue;
            ++published;
            EXPECT_EQ(slot.task_ref, 0);
            EXPECT_EQ(slot.generation, 1u);
            auto *payload =
                scheduler_state_at<DispatchPayload>(f.storage.scheduler_state->base(), f.payload_offset(lane, 0));
            payload_addresses[lane] = reinterpret_cast<uint64_t>(payload);
            EXPECT_EQ(payload->function_bin_addr, 0x1000u);
            EXPECT_EQ(payload->global_context.sub_block_id, lane == 2 ? 1 : 0);
        }
        EXPECT_EQ(published, static_cast<uint32_t>(__builtin_popcount(mask)));
        for (uint32_t a = 0; a < 3; ++a)
            for (uint32_t b = a + 1; b < 3; ++b)
                if (payload_addresses[a] && payload_addresses[b]) EXPECT_NE(payload_addresses[a], payload_addresses[b]);
        // A lone AIV subtask takes the first AIV lane with a free Mix slot, so
        // lane 2 is the one left untouched.
        if (mask == 3 || mask == 5) EXPECT_EQ(f.slot(2, 0).state, SchedulerDispatchSlotState::FREE);
    }
}

TEST(SchedulerMix, TwoPublishedMixesReleaseEachLaneBeforeWholeTask) {
    MixFixture f;
    // AIV-only Mix for both. The coordinator has one own slot, so two Mix tasks
    // with an AIC subtask cannot be in the air at once and the second would
    // never be published -- which is a different property from the one here.
    f.set_task(0, 6);
    f.set_task(1, 6);
    f.enqueue(0);
    f.enqueue(1);
    ASSERT_TRUE(f.dispatch());
    // The second dispatch can only go in once the lanes have taken the first
    // up: a lane has two slots, and the coordinator fills the one it is not
    // running. Flipping the index is what the executor does for real.
    for (uint32_t lane = 1; lane < 3; ++lane)
        f.take(lane, 0);
    ASSERT_TRUE(f.dispatch());
    for (uint32_t lane = 1; lane < 3; ++lane) {
        EXPECT_EQ(f.slot(lane, 0).task_ref, 0);
        EXPECT_EQ(f.slot(lane, 1).task_ref, 1);
        EXPECT_EQ(f.slot(lane, 1).generation, 2u);
    }
    ASSERT_TRUE(f.complete(1, 0));
    EXPECT_EQ(f.slot(1, 0).state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(f.local->pending_completed, 0u);
    ASSERT_TRUE(f.complete(2, 0));
    EXPECT_EQ(f.local->pending_completed, 1u);
    EXPECT_EQ(f.local->mix_trackers[0].task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(f.local->mix_trackers[1].task_ref, 1);
    EXPECT_FALSE(f.complete(2, 0));
    EXPECT_EQ(f.local->pending_completed, 1u);
}

TEST(SchedulerMix, CapacityCheckLeavesTaskAvailableAndPreservesReservations) {
    MixFixture f;
    // The coordinator's one own slot is busy, so a Mix task with an AIC subtask
    // has nowhere to go and the candidate must survive the probe untouched.
    f.local->running.state = SchedulerDispatchSlotState::FILLING;
    SchedulerReadyClaim candidate{};
    candidate.task_ref = 0;
    candidate.source = SchedulerReadySource::DIRECT_RESOLVE;
    bool progress = false;
    ASSERT_TRUE(scheduler_publish_direct_mix_candidate(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, &candidate, nullptr, 0,
        f.storage.ssbuf_region, &progress
    ));
    EXPECT_FALSE(progress);
    EXPECT_EQ(candidate.task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(f.dispatch());
    EXPECT_EQ(f.sequence(1), 0u);
    EXPECT_EQ(f.slot(0, 0).state, SchedulerDispatchSlotState::FILLING);
    // Freeing it is what lets the queued task place, and the reservation it
    // made on the AIV lanes is still the one it gets.
    scheduler_initialize_free_slot(&f.local->running);
    ASSERT_TRUE(f.dispatch());
    EXPECT_EQ(f.slot(0, 0).task_ref, 0);
    EXPECT_EQ(f.sequence(1), 1u);
}

TEST(SchedulerMix, FailedPreparationPublishesNothingAndConsumesNoSequence) {
    MixFixture f;
    f.storage.metadata[0].kernel_ids[2] = 2;
    f.enqueue(0);
    EXPECT_FALSE(f.dispatch());
    EXPECT_NE(f.storage.run_control->scheduler_error, 0u);
    for (uint32_t lane = 0; lane < 3; ++lane) {
        EXPECT_EQ(f.sequence(lane), 0u);
        EXPECT_EQ(f.slot(lane, 0).state, SchedulerDispatchSlotState::FREE);
        // Only a Mix subtask publishes through SSBUF; lane 0 is the
        // coordinator's own slot, whose publication is the local ready mask
        // asserted below.
        if (lane != 0) EXPECT_EQ(f.control(lane, 0)->publication, 0u);
    }
    EXPECT_FALSE(f.local->local_ready);
    EXPECT_EQ(f.local->mix_trackers[0].task_ref, SCHEDULER_TASK_REF_INVALID);
}

TEST(SchedulerMix, DirectCandidateDoesNotOvertakeQueuedMix) {
    MixFixture f;
    // AIV-only Mix for both: the coordinator has one own slot, so two Mix tasks
    // carrying an AIC subtask cannot be in the air together, and the ordering
    // this pins would never get a second task to order.
    f.set_task(1, 6);
    f.set_task(2, 6);
    f.graph.executable(2, 0, {0});
    f.storage.metadata[2].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_EQ(
        scheduler_route_task(
            f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 2, nullptr
        ),
        SchedulerRouteResult::WAITING
    );
    f.enqueue(1);
    SchedulerReadyClaim candidate{};
    ASSERT_TRUE(scheduler_resolve_completion(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0, nullptr, nullptr,
        nullptr, 0, false, nullptr, TaskType::MIX, &candidate
    ));
    EXPECT_EQ(candidate.task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(f.dispatch());
    // The lanes take the first dispatch up, which is what frees their other
    // slot for the second.
    for (uint32_t lane = 1; lane < 3; ++lane)
        f.take(lane, 0);
    ASSERT_TRUE(f.dispatch());
    for (uint32_t lane = 1; lane < 3; ++lane) {
        EXPECT_EQ(f.slot(lane, 0).task_ref, 1);
        EXPECT_EQ(f.slot(lane, 1).task_ref, 2);
    }
}

TEST(SchedulerMix, DirectCandidatePrecedesMixQueuedAfterResolve) {
    MixFixture f;
    f.set_task(1, 6);
    f.set_task(2, 6);
    f.graph.executable(1, 0, {0});
    f.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_EQ(
        scheduler_route_task(
            f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 1, nullptr
        ),
        SchedulerRouteResult::WAITING
    );
    SchedulerReadyClaim candidate{};
    ASSERT_TRUE(scheduler_resolve_completion(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0, nullptr, nullptr,
        nullptr, 0, false, nullptr, TaskType::MIX, &candidate
    ));
    ASSERT_EQ(candidate.task_ref, 1);
    f.enqueue(2);

    bool progress = false;
    ASSERT_TRUE(scheduler_fill_mix_after_completions(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, &candidate, &f.cursor,
        nullptr, 0, true, f.storage.ssbuf_region, &progress
    ));
    EXPECT_TRUE(progress);
    EXPECT_EQ(candidate.task_ref, SCHEDULER_TASK_REF_INVALID);
    // The direct candidate went in first; the queued one needs the lanes to
    // take that up before its slot is free.
    for (uint32_t lane = 1; lane < 3; ++lane)
        f.take(lane, 0);
    ASSERT_TRUE(f.dispatch());
    for (uint32_t lane = 1; lane < 3; ++lane) {
        EXPECT_EQ(f.slot(lane, 0).task_ref, 1);
        EXPECT_EQ(f.slot(lane, 1).task_ref, 2);
    }
}

TEST(SchedulerMix, DirectCandidateQueuesWhenRequiredLaneIsBusy) {
    MixFixture f;
    // The direct candidate is AIV-only, so what refuses it is the busy lane
    // rather than the coordinator's one own slot, which task 2 takes below.
    f.set_task(1, 6);
    f.graph.executable(1, 0, {0});
    f.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_EQ(
        scheduler_route_task(
            f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 1, nullptr
        ),
        SchedulerRouteResult::WAITING
    );
    f.set_task(2, 3);
    for (auto &slot : f.local->mix_placed[2])
        slot.state = SchedulerDispatchSlotState::FILLING;
    SchedulerReadyClaim candidate{};
    ASSERT_TRUE(scheduler_resolve_completion(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0, nullptr, nullptr,
        nullptr, 0, false, nullptr, TaskType::MIX, &candidate
    ));
    ASSERT_EQ(candidate.task_ref, 1);
    f.enqueue(2);

    bool progress = false;
    ASSERT_TRUE(scheduler_fill_mix_after_completions(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, &candidate, &f.cursor,
        nullptr, 0, true, f.storage.ssbuf_region, &progress
    ));
    EXPECT_TRUE(progress);
    EXPECT_EQ(candidate.task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(f.slot(0, 0).task_ref, 2);
    EXPECT_EQ(f.slot(1, 0).task_ref, 2);
    for (auto &slot : f.local->mix_placed[2])
        slot.state = SchedulerDispatchSlotState::FREE;
    // Lane 1 already holds task 2 in slot 0, so task 1's AIV subtask needs it
    // taken up before the other slot is the one a placement may fill.
    f.take(1, 0);
    ASSERT_TRUE(f.dispatch());
    EXPECT_EQ(f.slot(1, 1).task_ref, 1);
    EXPECT_EQ(f.slot(2, 0).task_ref, 1);
}

TEST(SchedulerMix, CompletedMixPassesUnlockedMixToDirectCandidate) {
    MixFixture f;
    f.graph.executable(1, 0, {0});
    f.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    ASSERT_EQ(
        scheduler_route_task(
            f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 1, nullptr
        ),
        SchedulerRouteResult::WAITING
    );
    f.enqueue(0);
    ASSERT_TRUE(f.dispatch());
    SchedulerRefillCandidates candidates;
    ASSERT_TRUE(f.complete(0, 0, &candidates));
    ASSERT_TRUE(f.complete(1, 0, &candidates));
    EXPECT_EQ(candidates.mix_ready.task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(f.complete(2, 0, &candidates));
    EXPECT_EQ(candidates.mix_ready.task_ref, 1);
    ASSERT_TRUE(scheduler_publish_direct_mix_candidate(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, &candidates.mix_ready,
        nullptr, 0, f.storage.ssbuf_region, nullptr
    ));
    for (uint32_t lane = 0; lane < 3; ++lane)
        EXPECT_EQ(f.slot(lane, 0).task_ref, 1);
}

TEST(SchedulerMix, MultipleUnlockedMixesKeepWakeOrder) {
    MixFixture f;
    // Both successors are AIV-only Mix, so the wake order is what decides which
    // lane slot each takes rather than the coordinator's single own slot.
    f.set_task(1, 6);
    f.set_task(2, 6);
    for (uint32_t task : {1U, 2U}) {
        f.graph.executable(task, 0, {0});
        f.storage.metadata[task].flags |= SCHEDULER_TASK_HAS_FANIN;
        ASSERT_EQ(
            scheduler_route_task(
                f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, task, nullptr
            ),
            SchedulerRouteResult::WAITING
        );
    }
    f.enqueue(0);
    ASSERT_TRUE(f.dispatch());
    SchedulerRefillCandidates candidates;
    ASSERT_TRUE(f.complete(0, 0, &candidates));
    ASSERT_TRUE(f.complete(1, 0, &candidates));
    ASSERT_TRUE(f.complete(2, 0, &candidates));
    EXPECT_EQ(candidates.mix_ready.task_ref, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(f.dispatch());
    for (uint32_t lane = 1; lane < 3; ++lane)
        f.take(lane, 0);
    ASSERT_TRUE(f.dispatch());
    for (uint32_t lane = 1; lane < 3; ++lane) {
        EXPECT_EQ(f.slot(lane, 0).task_ref, 2);
        EXPECT_EQ(f.slot(lane, 1).task_ref, 1);
    }
}

TEST(SchedulerMix, LastLaneAloneResolvesDependency) {
    MixFixture f;
    f.set_task(1, 1);
    f.graph.executable(1, 0, {0});
    f.storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    EXPECT_EQ(
        scheduler_route_task(
            f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 1, nullptr
        ),
        SchedulerRouteResult::WAITING
    );
    f.enqueue(0);
    ASSERT_TRUE(f.dispatch());
    ASSERT_TRUE(f.complete(0, 0));
    ASSERT_TRUE(f.complete(1, 0));
    auto *control = scheduler_task_control_at(f.storage.scheduler_state->base(), f.local, 0);
    EXPECT_NE(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    SchedulerRefillCandidates candidates;
    ASSERT_TRUE(f.complete(2, 0, &candidates));
    EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    SchedulerReadyClaim claim;
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 1, TaskType::AIC, &f.cursor,
        nullptr, &claim
    ));
    EXPECT_EQ(claim.task_ref, 1);
}

// A dispatch stays live until every one of its subtasks reports, so completing
// one lane's slot frees capacity without closing the task. More Mix tasks can
// therefore be in the air than a cluster has slots on any single lane.
TEST(SchedulerMix, PartialCompletionsAllowMoreThanTwoLiveMixes) {
    MixFixture f;
    f.set_task(0, 3);
    f.set_task(1, 6);
    f.set_task(3, 6);
    f.enqueue(0);
    ASSERT_TRUE(f.dispatch());
    // Task 0 takes the coordinator's own position and lane 1's slot 0. The
    // lane taking it up is what frees its other slot.
    f.take(1, 0);
    f.enqueue(1);
    ASSERT_TRUE(f.dispatch());
    // Task 1 takes lane 1's slot 1 and lane 2's slot 0, which fills lane 1.
    f.take(1, 1);
    f.take(2, 0);
    // Lane 1 reports task 0's AIV subtask. Task 0 stays live -- its AIC subtask
    // has not finished -- but lane 1 has a slot back.
    ASSERT_TRUE(f.complete(1, 0));
    EXPECT_EQ(f.local->pending_completed, 0u);
    f.enqueue(3);
    ASSERT_TRUE(f.dispatch());
    uint32_t live = 0;
    for (const auto &tracker : f.local->mix_trackers)
        live += tracker.task_ref >= 0;
    EXPECT_EQ(live, 3u);
    EXPECT_EQ(f.slot(0, 0).task_ref, 0) << "task 0 is live through the subtask the coordinator runs";
}

TEST(SchedulerMix, NoStealPolicyStillAdmitsLocalMix) {
    MixFixture f;
    // AIV-only Mix for both, so the two land on lane slots: the coordinator's
    // single own slot could not hold a second Mix subtask.
    f.set_task(0, 6);
    f.set_task(1, 6);
    // Two clusters, so the Mix queue has an owner this core can only reach by
    // stealing: inside one cluster a Mix task routes to this very Scheduler.
    f.local->config.cluster_count = 2;
    f.local->config.scheduler_count = 2;
    auto *control = scheduler_task_control_at(f.storage.scheduler_state->base(), f.local, 0);
    control->next_waiter = SCHEDULER_INBOX_EMPTY;
    scheduler_ready_inbox_at(f.storage.scheduler_state->base(), f.local, TaskType::MIX, 1)->endpoints =
        scheduler_ready_endpoints_pack(0, 0);
    scheduler_ready_directory_set(
        scheduler_ready_directory_at(f.storage.scheduler_state->base(), f.local), TaskType::MIX, 1
    );

    ASSERT_TRUE(f.dispatch(false));
    EXPECT_EQ(f.sequence(1), 0u);
    f.enqueue(1);
    ASSERT_TRUE(f.dispatch(false));
    EXPECT_EQ(f.slot(1, 0).task_ref, 1);
    for (uint32_t lane = 1; lane < 3; ++lane)
        f.take(lane, 0);
    ASSERT_TRUE(f.dispatch(true));
    EXPECT_EQ(f.slot(1, 1).task_ref, 0);
}

TEST(SchedulerMix, CompletionTracesUseLogicalSubtaskRatherThanPhysicalLane) {
    MixFixture f;
    // Subtasks 0 and 2, so the AIV subtask is logical 2 while the lane holding
    // it is physical 1 -- the two must not be confused when indexing traces.
    f.set_task(0, 5);
    f.enqueue(0);
    bool progress = false;
    ASSERT_TRUE(scheduler_fill_cluster_mix_slots(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, &f.cursor, nullptr,
        SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, true, f.storage.ssbuf_region, &progress
    ));
    // The coordinator reads its own slot's trace from local profiling state and
    // a Mix slot's from global memory; neither comes from the lane's SSBUF cells.
    f.local->profiling->executor_trace.kernel_start_cycles = 100;
    f.local->profiling->executor_trace.kernel_end_cycles = 200;
    auto *mix_trace = scheduler_state_at<SchedulerExecutorTaskTrace>(
        f.storage.scheduler_state->base(), f.local->mix_trace_offset(1, false)
    );
    mix_trace->kernel_start_cycles = 102;
    mix_trace->kernel_end_cycles = 202;
    for (uint32_t lane : {0, 1}) {
        ASSERT_TRUE(scheduler_service_cluster_completion_slot(
            f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, lane,
            f.slot(lane, 0).generation, nullptr, nullptr, nullptr, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, nullptr,
            nullptr, f.storage.ssbuf_region, nullptr, lane != 0
        ));
    }
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(f.storage.scheduler_state->base(), f.storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[0].kernel_start_cycles, 100u);
    EXPECT_EQ(traces[1].kernel_start_cycles, 102u);
    // The trace cell follows the logical subtask (2, second among those active),
    // while worker_id records the physical lane that ran it (1).
    EXPECT_EQ(traces[1].worker_id, 1u);
    EXPECT_EQ(traces[1].valid, 1u);
    EXPECT_EQ(traces[2].valid, 0u);
    EXPECT_EQ(f.local->pending_completed, 1u);
}

TEST(SchedulerDispatch, SequenceExhaustionDoesNotPublishOrWrap) {
    MixFixture f;
    f.set_task(0, 1);
    f.local->dispatch_sequence = UINT32_MAX;
    f.local->running.state = SchedulerDispatchSlotState::FILLING;
    SchedulerReadyClaim ready;
    ready.task_ref = 0;
    const auto metadata =
        scheduler_load_dispatch_metadata(f.storage.scheduler_state->base(), f.local, ready.task_ref, 0);
    EXPECT_FALSE(scheduler_prepare_dispatch_slot(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, {0, 0, 0, 0}, ready,
        metadata, 0, f.storage.ssbuf_region, 0
    ));
    EXPECT_EQ(f.local->dispatch_sequence, UINT32_MAX);
    EXPECT_FALSE(f.local->local_ready);
}

// One cluster, with the SPMD queue holding whatever a case puts there. The
// cluster's AIC is the core under test: it is both an ordinary claimer and the
// coordinator a Mix claim has to go through.
struct SpmdFixture {
    static constexpr uint32_t kTasks = 6;
    FixtureStorage storage{kTasks, PLATFORM_CORES_PER_BLOCKDIM, 16};
    GraphBuffer graph{kTasks};
    SchedulerLocalState *local;

    SpmdFixture() {
        for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            auto &context = storage.contexts[lane];
            context.is_scheduler = 1;
            context.scheduler_index = lane;
            context.scheduler_count = PLATFORM_CORES_PER_BLOCKDIM;
            for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer)
                context.cluster_worker_ids[peer] = peer;
        }
        storage.run_control->scheduler_count = PLATFORM_CORES_PER_BLOCKDIM;
        storage.callable_addresses[1] = 0x1000;
        local = storage.local_context(&storage.contexts[0], &storage.scheduler_local_state);
        local->has_spmd = true;
        scheduler_initialize_free_slot(&local->running);
        for (uint32_t lane = 1; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
            scheduler_ssbuf_store_relaxed(&storage.ssbuf_region->lanes[lane].current, 1);
            for (uint32_t slot = 0; slot < SCHEDULER_LANE_SLOT_COUNT; ++slot)
                scheduler_initialize_free_slot(&local->mix_placed[lane][slot]);
        }
    }

    // `mask` is the task's shape; `blocks` its block count, which also decides
    // whether the SPMD flag is set at all.
    void set_task(uint32_t task, uint8_t mask, uint16_t blocks, bool sync_start = false) {
        graph.mixed(task, mask);
        auto &metadata = storage.metadata[task];
        metadata.active_mask = mask;
        metadata.flags = static_cast<uint8_t>(
            SCHEDULER_TASK_EXECUTABLE | (__builtin_popcount(mask) > 1 ? SCHEDULER_TASK_MIX : 0) |
            (blocks > 1 ? SCHEDULER_TASK_SPMD : 0) | (sync_start ? SCHEDULER_TASK_SYNC_START : 0)
        );
        metadata.trace_index_base = task * 16;
        metadata.timing_slot = -1;
        for (uint8_t subtask = 0; subtask < 3; ++subtask)
            metadata.kernel_ids[subtask] = (mask & (1U << subtask)) != 0 ? 1 : UINT16_MAX;
        control(task)->block_num = blocks;
    }

    void enqueue(uint32_t task) {
        SchedulerReadyBatch batch{};
        ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, task, &batch, nullptr));
        ASSERT_TRUE(scheduler_ready_batch_push(
            storage.scheduler_state->base(), local,
            scheduler_task_ready_queue(storage.metadata[task].flags, storage.metadata[task].active_mask), &batch,
            nullptr
        ));
    }

    __gm__ SchedulerTaskControl *control(uint32_t task) {
        return scheduler_task_control_at(storage.scheduler_state->base(), local, task);
    }

    int64_t head() {
        const uint64_t packed =
            scheduler_ready_inbox_at(storage.scheduler_state->base(), local, TaskType::SPMD, 0)->endpoints;
        return scheduler_ready_endpoints_empty(packed) ? SCHEDULER_TASK_REF_INVALID :
                                                         scheduler_ready_endpoints_head(packed);
    }

    bool claim(bool *progress = nullptr) {
        return scheduler_claim_spmd_block(
            graph.graph(), storage.scheduler_state->base(), local, storage.run_control, nullptr, 0,
            storage.ssbuf_region, progress
        );
    }

    // Whether this core's position is holding the task. A core runs one at a
    // time, so there is nothing to search: the question is only whether the
    // block landed here.
    bool is_running(int64_t task_ref) {
        return local->running.state == SchedulerDispatchSlotState::READY && local->running.task_ref == task_ref;
    }
    // Stand in for a lane's executor taking its dispatch up, which is what
    // moves the slot a placement may fill.
    void take(uint32_t lane, uint32_t index) {
        scheduler_ssbuf_store_relaxed(&storage.ssbuf_region->lanes[lane].current, index);
    }
};

// The head stays put while its blocks are handed out, which is what lets a task
// of many blocks reach cores its publisher's queue shard never could.
TEST(SchedulerSpmdClaim, HandsOutEveryBlockExactlyOnceThenYieldsTheHead) {
    SpmdFixture f;
    f.set_task(0, 0b001, 4);
    f.set_task(1, 0b001, 2);
    f.enqueue(0);
    f.enqueue(1);

    // One slot per core, so four blocks take four rounds with a completion in
    // between. The head must not move until the last one is taken.
    std::vector<uint16_t> seen;
    for (uint32_t round = 0; round < 4; ++round) {
        bool progress = false;
        ASSERT_TRUE(f.claim(&progress));
        ASSERT_TRUE(progress) << "round " << round;
        seen.push_back(f.local->running.block_idx);
        EXPECT_EQ(f.local->running.block_num, 4u);
        if (round < 3) {
            EXPECT_EQ(f.head(), 0) << "the head must not yield with blocks left";
            scheduler_initialize_free_slot(&f.local->running);
        }
    }
    std::sort(seen.begin(), seen.end());
    EXPECT_EQ(seen, (std::vector<uint16_t>{0, 1, 2, 3}));
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 4u);
    // Claimed clean, so the next SPMD task is the head.
    EXPECT_EQ(f.head(), 1);
}

// The cursor cannot be put back, so a block taken with nowhere to run is lost
// for the rest of the run and the task's completed count never reaches its
// total. Probing first is what makes "it did not fit" cost nothing.
TEST(SchedulerSpmdClaim, LeavesTheCursorAloneWhenThereIsNowhereToPutTheBlock) {
    SpmdFixture f;
    f.set_task(0, 0b001, 4);
    f.enqueue(0);
    f.local->running.state = SchedulerDispatchSlotState::READY;

    bool progress = false;
    ASSERT_TRUE(f.claim(&progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 0u);
    EXPECT_EQ(f.head(), 0);
}

// A core whose type the head does not match leaves it alone and goes on to its
// own inbox. Nothing about the task changes.
TEST(SchedulerSpmdClaim, IgnoresAHeadThisCoreCannotRun) {
    SpmdFixture f;
    f.set_task(0, 0b010, 4);  // AIV-only, and this core is the cluster's AIC.
    f.enqueue(0);

    bool progress = false;
    ASSERT_TRUE(f.claim(&progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 0u);
    EXPECT_EQ(f.head(), 0);
    EXPECT_EQ(f.local->running.state, SchedulerDispatchSlotState::FREE);
}

// A core still holding the old head finds the task already claimed clean. The
// authority is the task's own cursor, so that costs one failed comparison
// rather than needing a protocol to prevent it -- and the core then leaves the
// head alone.
//
// Leaving it alone is the point. The cursor is spent when the last block is
// *claimed*, which is before that block's payload is published, so a core that
// advanced the head on this signal would let the next task be claimed while the
// previous one still had an undispatched block. The core that took the final
// block advances the head instead, after its own dispatch succeeds.
TEST(SchedulerSpmdClaim, AClaimAgainstAnExhaustedHeadLeavesTheHeadAlone) {
    SpmdFixture f;
    f.set_task(0, 0b001, 2);
    f.set_task(1, 0b001, 2);
    f.enqueue(0);
    f.enqueue(1);
    scheduler_gm_store(f.control(0)->next_block, UINT64_C(2));

    bool progress = false;
    ASSERT_TRUE(f.claim(&progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 2u);
    EXPECT_EQ(f.head(), 0);
}

// A graph with no SPMD task must not pay for the queue at all: the gate is what
// keeps an endpoints read off every core's every pass.
TEST(SchedulerSpmdClaim, ReadsNothingWhenTheGraphHasNoSpmdTask) {
    SpmdFixture f;
    f.local->has_spmd = false;
    f.set_task(0, 0b001, 4);
    f.enqueue(0);

    bool progress = false;
    ASSERT_TRUE(f.claim(&progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 0u);
    EXPECT_TRUE(scheduler_task_ref_is_null(f.local->spmd_head_ref));
}

// Two blocks of one cohort on one core self-lock: the second waits in a slot
// the first cannot release, because the first is spinning in a barrier with its
// core's whole scheduling loop stopped behind it.
//
// An AIV-only Mix cohort is the shape that reaches the rule. A core has one own
// slot, so for every other shape a block of this cohort already occupies it and
// the placement probe refuses the second block before the scan is consulted.
// This one takes no AIC slot at all and leaves a free Mix slot on both lanes,
// so the scan is the only thing standing in the way.
TEST(SchedulerSpmdClaim, RefusesASecondBlockOfTheSameCohortOnOneCluster) {
    SpmdFixture f;
    f.local->has_mix = true;
    f.set_task(0, 0b110, 4, /*sync_start=*/true);
    f.enqueue(0);

    bool progress = false;
    ASSERT_TRUE(f.claim(&progress));
    ASSERT_TRUE(progress);
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 1u);

    // The lanes take the first block up, so their other slot is free and
    // capacity is not what refuses the second: the cohort scan is.
    f.take(1, 0);
    f.take(2, 0);
    progress = false;
    ASSERT_TRUE(f.claim(&progress));
    EXPECT_FALSE(progress);
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 1u);

    // An ordinary SPMD task of the same shape has no such restriction: its
    // blocks wait on nothing, and the lanes' second Mix slots take them.
    f.set_task(0, 0b110, 4);
    f.local->spmd_head_ref = SCHEDULER_TASK_REF_INVALID;
    progress = false;
    ASSERT_TRUE(f.claim(&progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 2u);
}

// The head's shape is read once and remembered, so a pass that finds the head
// unchanged costs the endpoints load alone. Every core pays this on every pass
// for as long as one task holds the head.
TEST(SchedulerSpmdClaim, RemembersTheHeadsShapeUntilTheHeadMoves) {
    SpmdFixture f;
    f.set_task(0, 0b001, 3);
    f.enqueue(0);

    bool progress = false;
    ASSERT_TRUE(f.claim(&progress));
    ASSERT_TRUE(progress);
    EXPECT_EQ(f.local->spmd_head_ref, 0);
    EXPECT_EQ(f.local->spmd_head_block_num, 3u);
    EXPECT_EQ(f.local->spmd_head_active_mask, 0b001);

    // Taking the last block is what moves the head, and the core that took it
    // is the one that moves it -- so this core sees the queue drain and clears
    // the memo rather than leaving it to describe a task that no longer leads.
    // The slot the first block took has to come back first: a core with no free
    // slot never reaches the cursor at all.
    scheduler_initialize_free_slot(&f.local->running);
    scheduler_gm_store(f.control(0)->next_block, UINT64_C(2));
    ASSERT_TRUE(f.claim(&progress));
    ASSERT_TRUE(scheduler_ready_endpoints_empty(scheduler_gm_query(
        scheduler_ready_inbox_at(f.storage.scheduler_state->base(), f.local, TaskType::SPMD, 0)->endpoints
    )));
    scheduler_initialize_free_slot(&f.local->running);
    ASSERT_TRUE(f.claim(&progress));
    EXPECT_TRUE(scheduler_task_ref_is_null(f.local->spmd_head_ref));
}

// Only the last block of a task resolves its dependents. A block that reported
// completion as if it were the whole task would release successors onto data
// its siblings had not written yet.
TEST(SchedulerSpmdCompletion, HoldsTheTaskOpenUntilItsLastBlockFinishes) {
    SpmdFixture f;
    f.set_task(0, 0b001, 4);
    f.enqueue(0);

    for (uint32_t block = 0; block < 4; ++block) {
        bool progress = false;
        ASSERT_TRUE(f.claim(&progress));
        ASSERT_TRUE(progress) << "block " << block;
        ASSERT_TRUE(f.is_running(0)) << "block " << block;
        f.local->local_ready = false;
        ASSERT_TRUE(scheduler_service_cluster_completion_slot(
            f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0,
            f.local->running.generation, nullptr, nullptr, nullptr, 0, nullptr, nullptr, f.storage.ssbuf_region,
            nullptr, false
        )) << "block "
           << block;
        const bool last = block == 3;
        EXPECT_EQ(scheduler_gm_query(f.control(0)->completed_blocks), block + 1);
        EXPECT_EQ(scheduler_gm_query(f.control(0)->state) == static_cast<int64_t>(SchedulerTaskState::DONE), last)
            << "block " << block;
        EXPECT_EQ(f.local->pending_completed, last ? 1u : 0u) << "block " << block;
    }
}

// A task of one block counts nothing: its completion must not touch the shared
// counter, which is what keeps the ordinary path free of an SPMD cost.
TEST(SchedulerSpmdCompletion, ASingleBlockTaskCountsNoBlocks) {
    SpmdFixture f;
    f.set_task(0, 0b001, 1);
    f.enqueue(0);

    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    const SchedulerFreeSlotClaim claim{0, 0, f.local->running.generation, 0};
    f.local->running.state = SchedulerDispatchSlotState::FILLING;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, claim, ready, 0,
        f.storage.ssbuf_region
    ));
    f.local->local_ready = false;
    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0,
        f.local->running.generation, nullptr, nullptr, nullptr, 0, nullptr, nullptr, f.storage.ssbuf_region, nullptr,
        false
    ));
    EXPECT_EQ(scheduler_gm_query(f.control(0)->completed_blocks), 0u);
    EXPECT_EQ(scheduler_gm_query(f.control(0)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
}

// Yielding the head and placing a block are unrelated, and conflating them
// deadlocks at the worst moment: a Mix head whose last block was just taken has
// its own blocks sitting in the cluster's Mix slots, so a placement probe
// cannot succeed. A pop handed a placement would leave the head where it is and
// every later SPMD task would starve behind it.
TEST(SchedulerSpmdClaim, YieldsAMixHeadWhoseOwnBlocksFillTheCluster) {
    SpmdFixture f;
    f.local->has_mix = true;
    // AIV-only Mix, so both blocks fit the cluster: a shape with an AIC subtask
    // would need the coordinator's one own slot twice and never place the
    // second block at all.
    f.set_task(0, 0b110, 2);
    f.set_task(1, 0b110, 2);
    f.enqueue(0);
    f.enqueue(1);

    for (uint32_t block = 0; block < 2; ++block) {
        bool progress = false;
        ASSERT_TRUE(f.claim(&progress));
        ASSERT_TRUE(progress) << "block " << block;
        // The lanes take each block up, which is what leaves room for the next
        // and, after the second, fills the cluster.
        f.take(1, block);
        f.take(2, block);
    }
    EXPECT_EQ(scheduler_gm_query(f.control(0)->next_block), 2u);

    // Both blocks are resident, so nothing in the cluster can be placed.
    SchedulerMixPlacement placement{};
    ASSERT_FALSE(
        scheduler_plan_mix_placement(f.storage.scheduler_state->base(), f.local, 0, &placement, f.storage.ssbuf_region)
    );
    EXPECT_EQ(f.head(), 1) << "the head must yield on its cursor, not on whether it could be placed";
}

// A direct refill bypasses the claim pass, so a core that resolves a successor
// on every completion is never seen with a free slot. The cohort then assembles
// only once the rest of the reachable graph drains, while its already-claimed
// blocks spin in a barrier holding cores whose scheduling loops are stopped.
TEST(SchedulerSpmdClaim, HoldsASlotOpenForACohortInsteadOfRefillingIt) {
    SpmdFixture f;
    f.set_task(0, 0b001, 1);
    f.set_task(1, 0b001, 1);
    // Task 1 waits on task 0, so completing task 0 resolves a successor of this
    // core's own type -- exactly what the refill path takes.
    f.graph.executable(1, 0, {0});
    f.set_task(2, 0b001, 4, /*sync_start=*/true);
    f.enqueue(2);

    auto dispatch_task_zero = [&] {
        SchedulerReadyClaim ready{};
        ready.task_ref = 0;
        const SchedulerFreeSlotClaim claim{0, 0, f.local->running.generation, 0};
        f.local->running.state = SchedulerDispatchSlotState::FILLING;
        ASSERT_TRUE(scheduler_fill_dispatch_slot(
            f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, claim, ready, 0,
            f.storage.ssbuf_region
        ));
        f.local->local_ready = false;
        scheduler_gm_store(f.control(0)->wake_list_head, INT64_C(1));
        scheduler_gm_store(f.control(1)->next_waiter, SCHEDULER_TASK_REF_INVALID);
    };

    // The head is read from this core's memo, which the claim pass maintains.
    SchedulerSpmdHeadVerdict verdict = SchedulerSpmdHeadVerdict::EMPTY;
    ASSERT_TRUE(scheduler_observe_spmd_head(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, &verdict
    ));
    ASSERT_EQ(verdict, SchedulerSpmdHeadVerdict::CLAIMABLE);
    ASSERT_TRUE(scheduler_spmd_head_is_own_cohort(f.local));

    dispatch_task_zero();
    bool refilled = true;
    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0,
        f.local->running.generation, nullptr, nullptr, nullptr, 0, nullptr, &refilled, f.storage.ssbuf_region, nullptr,
        false
    ));
    EXPECT_FALSE(refilled) << "the slot belongs to the cohort at the head";
    EXPECT_EQ(f.local->running.state, SchedulerDispatchSlotState::FREE);
    // The successor is not dropped: it joins the queue the ordinary fill reads.
    const uint64_t aic_endpoints =
        scheduler_ready_inbox_at(f.storage.scheduler_state->base(), f.local, TaskType::AIC, 0)->endpoints;
    EXPECT_EQ(scheduler_ready_endpoints_head(aic_endpoints), 1);

    // Without a cohort at the head the same completion refills the slot, which
    // is what says the suppression is the cohort's doing and not an accident of
    // this setup.
    SpmdFixture g;
    g.set_task(0, 0b001, 1);
    g.set_task(1, 0b001, 1);
    g.graph.executable(1, 0, {0});
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    const SchedulerFreeSlotClaim claim{0, 0, g.local->running.generation, 0};
    g.local->running.state = SchedulerDispatchSlotState::FILLING;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        g.graph.graph(), g.storage.scheduler_state->base(), g.local, g.storage.run_control, claim, ready, 0,
        g.storage.ssbuf_region
    ));
    g.local->local_ready = false;
    scheduler_gm_store(g.control(0)->wake_list_head, INT64_C(1));
    scheduler_gm_store(g.control(1)->next_waiter, SCHEDULER_TASK_REF_INVALID);
    ASSERT_TRUE(scheduler_task_ref_is_null(g.local->spmd_head_ref));

    refilled = false;
    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        g.graph.graph(), g.storage.scheduler_state->base(), g.local, g.storage.run_control, 0,
        g.local->running.generation, nullptr, nullptr, nullptr, 0, nullptr, &refilled, g.storage.ssbuf_region, nullptr,
        false
    ));
    EXPECT_TRUE(refilled);
    EXPECT_EQ(g.local->running.task_ref, 1);
}

// Two cores deciding at once that the head owes nothing is ordinary, not a
// corner: every core racing for the last block ends its claim loop with the
// cursor past the end, and every one of them then tries to advance. The second
// must not carry off the task that just became the head -- its blocks are all
// still unclaimed, and off the queue nothing can find them.
//
// An unguarded pop does exactly that. Its CAS makes the update atomic, not the
// decision: the loser re-reads the endpoints, finds the successor, and pops it.
TEST(SchedulerSpmdClaim, ASecondAdvanceDoesNotCarryOffTheNewHead) {
    SpmdFixture f;
    f.set_task(0, 0b001, 2);
    f.set_task(1, 0b001, 4);
    f.enqueue(0);
    f.enqueue(1);
    scheduler_gm_store(f.control(0)->next_block, UINT64_C(2));

    // The first core observes the exhausted head and yields it.
    ASSERT_TRUE(scheduler_advance_spmd_head(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0, nullptr
    ));
    EXPECT_EQ(f.head(), 1);

    // The second arrives with the same conclusion about the same task.
    ASSERT_TRUE(scheduler_advance_spmd_head(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0, nullptr
    ));
    EXPECT_EQ(f.head(), 1) << "the late advance took a head that still had every block to give out";
    EXPECT_EQ(scheduler_gm_query(f.control(1)->next_block), 0u);

    // And the claim path still reaches task 1's blocks.
    bool progress = false;
    ASSERT_TRUE(f.claim(&progress));
    EXPECT_TRUE(progress);
    EXPECT_EQ(scheduler_gm_query(f.control(1)->next_block), 1u);
}

namespace {

// Dispatch task 0 into slot 0 and register task 1 as its waiter, so completing
// task 0 resolves a successor this core could take straight back.
void arrange_completion_with_successor(SpmdFixture &f) {
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    const SchedulerFreeSlotClaim claim{0, 0, f.local->running.generation, 0};
    f.local->running.state = SchedulerDispatchSlotState::FILLING;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, claim, ready, 0,
        f.storage.ssbuf_region
    ));
    f.local->local_ready = false;
    scheduler_gm_store(f.control(0)->wake_list_head, INT64_C(1));
    scheduler_gm_store(f.control(1)->next_waiter, SCHEDULER_TASK_REF_INVALID);
}

bool complete_slot_zero(SpmdFixture &f, bool *refilled) {
    return scheduler_service_cluster_completion_slot(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, 0,
        f.local->running.generation, nullptr, nullptr, nullptr, 0, nullptr, refilled, f.storage.ssbuf_region, nullptr,
        false
    );
}

int64_t aic_inbox_head(SpmdFixture &f) {
    const uint64_t packed =
        scheduler_ready_inbox_at(f.storage.scheduler_state->base(), f.local, TaskType::AIC, 0)->endpoints;
    return scheduler_ready_endpoints_empty(packed) ? SCHEDULER_TASK_REF_INVALID :
                                                     scheduler_ready_endpoints_head(packed);
}

}  // namespace

// The completion path consults this core's memo of the SPMD head rather than
// the queue, which costs it no GM access and makes its view up to one round
// old. Stale in the direction of "a cohort is waiting" when none is: the slot
// is held open for nothing, and the successor has to still be somewhere.
TEST(SchedulerSpmdClaim, AStaleCohortMemoCostsAHoldButNeverTheWork) {
    SpmdFixture f;
    f.set_task(0, 0b001, 1);
    f.set_task(1, 0b001, 1);
    f.graph.executable(1, 0, {0});
    // A cohort this core could run, as the memo last saw it -- and nothing on
    // the queue, as the queue is now.
    f.local->spmd_head_ref = 2;
    f.local->spmd_head_flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
    f.local->spmd_head_active_mask = 0b001;
    f.local->spmd_head_block_num = 4;
    ASSERT_TRUE(scheduler_spmd_head_is_own_cohort(f.local));

    arrange_completion_with_successor(f);
    bool refilled = true;
    ASSERT_TRUE(complete_slot_zero(f, &refilled));

    EXPECT_FALSE(refilled);
    EXPECT_EQ(f.local->running.state, SchedulerDispatchSlotState::FREE);
    // The hold is the whole cost: the successor is queued, so the ordinary
    // fill takes it on this same pass.
    EXPECT_EQ(aic_inbox_head(f), 1) << "a hold that drops the successor is not a hold, it is a loss";

    // And a look at the real queue retires the memo rather than leaving it to
    // hold slots open forever.
    SchedulerSpmdHeadVerdict verdict = SchedulerSpmdHeadVerdict::CLAIMABLE;
    ASSERT_TRUE(scheduler_observe_spmd_head(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, &verdict
    ));
    EXPECT_EQ(verdict, SchedulerSpmdHeadVerdict::EMPTY);
    EXPECT_FALSE(scheduler_spmd_head_is_own_cohort(f.local));
}

// Stale the other way: a cohort reached the head after this core last looked.
// The refill goes ahead, which costs the cohort a round -- and the next look
// is what ends that, so the staleness has to be bounded by the claim pass
// refreshing the memo rather than by anything the completion does.
TEST(SchedulerSpmdClaim, AMemoThatHasNotSeenTheCohortYetRefreshesOnTheNextLook) {
    SpmdFixture f;
    f.set_task(0, 0b001, 1);
    f.set_task(1, 0b001, 1);
    f.graph.executable(1, 0, {0});
    f.set_task(2, 0b001, 4, /*sync_start=*/true);
    f.enqueue(2);
    ASSERT_TRUE(scheduler_task_ref_is_null(f.local->spmd_head_ref));

    arrange_completion_with_successor(f);
    bool refilled = false;
    ASSERT_TRUE(complete_slot_zero(f, &refilled));
    EXPECT_TRUE(refilled) << "nothing told this core a cohort had arrived, so the slot is refilled";
    EXPECT_EQ(f.local->running.task_ref, 1);

    // The claim pass looks at the queue every round, so the window is one.
    SchedulerSpmdHeadVerdict verdict = SchedulerSpmdHeadVerdict::EMPTY;
    ASSERT_TRUE(scheduler_observe_spmd_head(
        f.graph.graph(), f.storage.scheduler_state->base(), f.local, f.storage.run_control, &verdict
    ));
    EXPECT_EQ(verdict, SchedulerSpmdHeadVerdict::CLAIMABLE);
    EXPECT_TRUE(scheduler_spmd_head_is_own_cohort(f.local));
}
