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

#include <iterator>

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "scheduler/scheduler_dispatch.h"
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

SchedulerLocalSlotState *
test_slot(FixtureStorage &storage, const SchedulerWorkerContext &scheduler, uint64_t worker_id, uint32_t pending_slot) {
    (void)scheduler;
    (void)worker_id;
    return &storage.scheduler_local_state.slots[pending_slot];
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
    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
        scheduler_initialize_free_slot(&storage.scheduler_local_state.slots[slot]);
    for (uint64_t task = 0; task < task_count; ++task) {
        storage.metadata[task].kernel_ids[0] = UINT16_MAX;
        storage.metadata[task].kernel_ids[1] = 1;
        storage.metadata[task].active_mask = LaneMask::aiv0();
        storage.metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
    }
    storage.prepare_ready_pool();
}

void enqueue_normal_aiv_tasks(
    FixtureStorage &storage, SchedulerWorkerContext &scheduler, uint64_t task_begin, uint64_t task_end
) {
    SchedulerReadyBatch batch{};

    for (uint64_t task = task_begin; task < task_end; ++task) {
        auto *control = scheduler_task_control_at(
            storage.scheduler_state->base(), storage.local_context(&scheduler), static_cast<int64_t>(task)
        );
        control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), storage.local_context(&scheduler), static_cast<int64_t>(task), &batch
        ));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state), 1, &batch
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
    scheduler_local_ready_publish(&storage.scheduler_local_state, pending_slot);
}

// A completed task in one of this core's own slots. A core fills and retires
// only the slots it executes from. `scheduler` must be the lane-0 context,
// which is the lane the slot belongs to.
SchedulerLocalSlotState *prepare_completed_normal_slot(FixtureStorage &storage, SchedulerWorkerContext &scheduler) {
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        scheduler.cluster_worker_ids[lane] = lane;
    storage.run_control->scheduler_count = 1;
    SchedulerLocalSlotState &local_slot = storage.scheduler_local_state.slots[0];
    scheduler_initialize_free_slot(&local_slot);
    local_slot.task_ref = 0;
    local_slot.subtask_slot = 0;
    local_slot.state = SchedulerDispatchSlotState::READY;
    local_slot.generation = ++storage.scheduler_local_state.dispatch_sequence;
    storage.scheduler_local_state.local_completed_generations[0] = local_slot.generation;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    return &local_slot;
}

TEST(SchedulerLocalCompletion, TracksConsumedGenerationsPerSlot) {
    SchedulerLocalState scheduler_local_state{};
    EXPECT_FALSE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 0, 0));
    EXPECT_TRUE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 0, 1));
    scheduler_mark_completion_consumed(&scheduler_local_state, 0, 0, 1);
    EXPECT_FALSE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 0, 1));
    EXPECT_TRUE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 0, 2));
    EXPECT_TRUE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 1, 1));
    scheduler_mark_completion_consumed(&scheduler_local_state, 0, 1, UINT32_MAX);
    EXPECT_TRUE(scheduler_completion_generation_is_new(&scheduler_local_state, 0, 1, 1));
}

TEST(SchedulerClusterCompletion, AccumulatesBatchOnSuccessAndBothKindsOfErrorExit) {
    for (int failure = 0; failure < 3; ++failure) {
        FixtureStorage storage(2, 2);
        GraphBuffer graph(2);
        graph.executable(0, 0);
        graph.executable(1, 0);
        auto &scheduler = storage.contexts[0];
        prepare_completed_normal_slot(storage, scheduler);
        auto &second = storage.scheduler_local_state.slots[1];
        scheduler_initialize_free_slot(&second);
        second.task_ref = 1;
        second.subtask_slot = 0;
        second.state = SchedulerDispatchSlotState::READY;
        second.generation = ++storage.scheduler_local_state.dispatch_sequence;
        auto *control =
            scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1);
        if (failure == 1) control->wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;
        storage.scheduler_local_state.local_completed_generations[1] = second.generation + (failure == 2 ? 1 : 0);
        EXPECT_EQ(
            scheduler_service_cluster_completions(
                graph.graph(), storage.scheduler_state->base(),
                storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, nullptr
            ),
            failure == 0
        );
        EXPECT_EQ(storage.scheduler_local_state.pending_completed, failure == 2 ? 1u : 2u);
        EXPECT_EQ(storage.run_control->scheduler_error == 0, failure == 0);
        EXPECT_EQ(scheduler_consumed_completion_generation(&storage.scheduler_local_state, 0, 0), 1u);
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
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.slots[0];
    scheduler_initialize_free_slot(local_slot);
    slot->task_ref = 0;
    local_slot->task_ref = 0;
    local_slot->subtask_slot = 0;
    storage.scheduler_local_state.set_timing_slot(0, 0, 0);
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->generation = ++storage.scheduler_local_state.dispatch_sequence;
    auto *executor_trace = &storage.scheduler_local_state.profiling->executor_traces[0];
    executor_trace->kernel_start_cycles = 100;
    executor_trace->kernel_end_cycles = 200;
    storage.metadata[0].timing_slot = 0;
    storage.scheduler_local_state.local_completed_generations[0] = local_slot->generation;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);

    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, nullptr
    ));
    // The generation is consumed from local state, and the slot is released.
    EXPECT_EQ(scheduler_consumed_completion_generation(&storage.scheduler_local_state, 0, 0), slot->generation);
    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    EXPECT_FALSE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, nullptr
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
    auto &slot = local->slots[0];
    scheduler_initialize_free_slot(&slot);
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control,
        SchedulerFreeSlotClaim{1, 0, slot.generation, 1}, ready, 0
    ));
    uint32_t selected = UINT32_MAX;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(local, &selected, &publication));
    ASSERT_EQ(selected, 0u);
    EXPECT_EQ(local->timing_slot(1, selected), -1);
    local->local_completed_generations[selected] = scheduler_dispatch_generation(publication);
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, nullptr
    ));
    EXPECT_EQ(local->profiling, nullptr);
    EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
    scheduler_flush_completions(storage.run_control, local);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

TEST(SchedulerClusterCompletion, SelfSlotsKeepDistinctSampledTraces) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        scheduler.cluster_worker_ids[lane] = lane;
    auto *local = storage.local_context(&scheduler, &storage.scheduler_local_state);
    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
        graph.executable(index, 1);
        auto &slot = local->slots[index];
        scheduler_initialize_free_slot(&slot);
        slot.task_ref = index;
        slot.subtask_slot = 1;
        local->set_timing_slot(1, index, index);
        slot.state = SchedulerDispatchSlotState::READY;
        slot.generation = ++local->dispatch_sequence;
        storage.metadata[index].active_mask = LaneMask::aiv0();
        local->profiling->executor_traces[index].kernel_start_cycles = 100 + index;
        local->profiling->executor_traces[index].kernel_end_cycles = 200 + index;
        local->local_completed_generations[index] = slot.generation;
        auto *control = scheduler_task_control_at(storage.scheduler_state->base(), local, index);
        control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        storage.metadata[index].timing_slot = index;
    }
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, nullptr
    ));
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
        EXPECT_EQ(traces[index].kernel_start_cycles, 100 + index);
        EXPECT_EQ(traces[index].kernel_end_cycles, 200 + index);
        EXPECT_FALSE(local->sampled_task_timing(1, index));
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
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.slots[0];
    scheduler_initialize_free_slot(local_slot);
    slot->task_ref = 0;
    local_slot->task_ref = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->generation = ++storage.scheduler_local_state.dispatch_sequence;

    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0,
        slot->generation + 1, false, nullptr, nullptr
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
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0, slot->generation,
        0, nullptr, nullptr
    ));
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    EXPECT_NE(storage.run_control->scheduler_error, 0u);
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED)
    );
}

TEST(SchedulerClusterCompletion, CompletionWithoutSuccessorsIgnoresQueueStorage) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    SchedulerLocalSlotState *slot = prepare_completed_normal_slot(storage, scheduler);
    storage.prepare_ready_pool();
    auto *local = storage.local_context(&scheduler, &storage.scheduler_local_state);
    auto *pool =
        scheduler_state_at<SchedulerReadyPoolHeader>(storage.scheduler_state->base(), local->config.ready_pool_offset);
    // Completion reads dependency metadata but does not access queue storage
    // when the task has no successors.
    pool->dependencies_offset = 0;
    pool->descriptors_offset = UINT64_MAX;
    pool->controls_offset = UINT64_MAX;

    EXPECT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, slot->generation, 0, nullptr,
        nullptr
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
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0, slot->generation,
        0, &replacement, nullptr
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

    ASSERT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler), storage.run_control, 1
        ),
        SchedulerRouteResult::WAITING
    );
    auto *slot = test_slot(storage, scheduler, 0, 0);
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.slots[0];
    scheduler_initialize_free_slot(local_slot);
    slot->task_ref = 0;
    local_slot->task_ref = 0;
    local_slot->subtask_slot = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    local_slot->generation = ++storage.scheduler_local_state.dispatch_sequence;
    auto *executor_trace = &storage.scheduler_local_state.profiling->executor_traces[0];
    executor_trace->kernel_start_cycles = 100;
    executor_trace->kernel_end_cycles = 200;
    storage.scheduler_local_state.local_completed_generations[0] = local_slot->generation;
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    producer->completion_resolve_start_cycles = 1;
    producer->completion_resolve_end_cycles = 1;
    producer->scheduler_worker_id = UINT64_MAX;
    traces[1].kernel_start_cycles = 1;

    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control,
        SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, nullptr
    ));
    EXPECT_EQ(producer->completion_resolve_start_cycles, 0u);
    EXPECT_EQ(producer->completion_resolve_end_cycles, 0u);
    EXPECT_EQ(producer->scheduler_worker_id, scheduler.worker_index);
    EXPECT_EQ(traces[1].kernel_start_cycles, 1u);
    EXPECT_EQ(traces[0].valid, 1u);
    EXPECT_EQ(traces[0].kernel_start_cycles, 100u);
    EXPECT_EQ(traces[0].kernel_end_cycles, 200u);
    auto *waiter = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1);
    EXPECT_EQ(waiter->state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
    // The core owns the slot it just freed, so the successor it woke goes
    // straight back into it instead of round-tripping through the ready queue.
    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(local_slot->task_ref, 1);
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

    storage.prepare_ready_pool();
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.slots[0];
    scheduler_initialize_free_slot(local_slot);
    local_slot->generation = ++storage.scheduler_local_state.dispatch_sequence;
    const uint32_t completed_generation = local_slot->generation;
    local_slot->task_ref = 0;
    local_slot->subtask_slot = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    storage.scheduler_local_state.local_completed_generations[0] = completed_generation;
    auto *completed_control =
        scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completed_control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    auto *ready_control =
        scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1);
    ready_control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerReadyBatch batch{};

    ASSERT_TRUE(
        scheduler_ready_batch_append(storage.scheduler_state->base(), storage.local_context(&scheduler), 1, &batch)
    );
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state), 0, &batch
    ));

    uint64_t direct_refilled_slot_mask = 0;
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control,
        SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, &direct_refilled_slot_mask
    ));

    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(local_slot->task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(local_slot->generation, completed_generation);
    EXPECT_EQ(completed_control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.scheduler_local_state.pending_completed, 1u);
    EXPECT_EQ(direct_refilled_slot_mask, 0u);
    // Task 1 is still where it was pushed, claimable by the next fill pass.
    SchedulerReadyClaim queued{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 1, 0, &queued
    ));
    EXPECT_EQ(queued.task_ref, 1);
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
    SchedulerLocalSlotState *local_slot = &scheduler_local_state.slots[0];
    scheduler_initialize_free_slot(local_slot);
    SchedulerReadyClaim producer_ready{};
    producer_ready.task_ref = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, SchedulerFreeSlotClaim{0, 0, local_slot->generation, 0}, producer_ready,
        SCHEDULER_PROFILING_TASK_TIMING_LEVEL
    ));
    ASSERT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler), storage.run_control, 1
        ),
        SchedulerRouteResult::WAITING
    );

    const uint32_t completed_generation = local_slot->generation;
    auto *executor_trace = &scheduler_local_state.profiling->executor_traces[0];
    executor_trace->kernel_start_cycles = 100;
    executor_trace->kernel_end_cycles = 200;
    scheduler_local_state.local_completed_generations[0] = completed_generation;

    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, SCHEDULER_PROFILING_TASK_TIMING_LEVEL, nullptr
    ));

    EXPECT_EQ(local_slot->task_ref, 1);
    EXPECT_EQ(local_slot->generation, completed_generation + 1);
    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::READY);

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

    SchedulerLocalState scheduler_local_state{};
    for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
        SchedulerLocalSlotState *local_slot = &scheduler_local_state.slots[pending_slot];
        scheduler_initialize_free_slot(local_slot);
        SchedulerReadyClaim ready{};
        ready.task_ref = pending_slot;
        ASSERT_TRUE(scheduler_fill_dispatch_slot(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
            storage.run_control, SchedulerFreeSlotClaim{0, pending_slot, local_slot->generation, 0}, ready,
            SCHEDULER_PROFILING_SCHED_PHASES_LEVEL
        ));
        auto *executor_trace = &scheduler_local_state.profiling->executor_traces[pending_slot];
        executor_trace->kernel_start_cycles = 100 + pending_slot;
        executor_trace->kernel_end_cycles = 200 + pending_slot;
    }

    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, 0, 0, scheduler_local_state.slots[0].generation, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL,
        nullptr, nullptr
    ));
    ASSERT_TRUE((scheduler_local_state.profiling->worker_trace_valid_mask & 1U) != 0);

    target.trace_aicore_entry_cycles = 21;
    target.trace_handshake_publish_cycles = 22;
    target.trace_register_release_cycles = 23;
    target.trace_descriptor_cache_observed_cycles = 24;
    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, 0, 1, scheduler_local_state.slots[1].generation, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL,
        nullptr, nullptr
    ));

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
    replacement.source = SchedulerReadySource::POOL;
    replacement.state_probe_start_cycles = 123;
    replacement.state_probe_end_cycles = 456;
    bool refilled = false;

    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0, slot->generation,
        SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, &replacement, &refilled
    ));

    EXPECT_TRUE(refilled);
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[1].ready_source, static_cast<uint64_t>(SchedulerReadySource::POOL));
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
    storage.prepare_ready_pool();
    auto *local = storage.local_context(&scheduler, &storage.scheduler_local_state);
    ASSERT_EQ(local->config.self_lane, 0u);
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), local, 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerReadyBatch batch{};

    ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, 0, &batch));
    ASSERT_TRUE(scheduler_ready_batch_push(storage.scheduler_state->base(), local, 0, &batch));

    bool failed = true;
    ASSERT_TRUE(scheduler_fill_own_slots(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &failed
    ));
    EXPECT_FALSE(failed);
    EXPECT_EQ(local->own_slots()[0].task_ref, 0);
    EXPECT_EQ(local->own_slots()[0].state, SchedulerDispatchSlotState::READY);
}

TEST(SchedulerNormalDispatch, LeavesLaterWorkSharedWhileOneExecutionIsQueued) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.mixed(1, LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false));
    SchedulerWorkerContext &scheduler = storage.contexts[0];
    scheduler.core_type = static_cast<int32_t>(CoreType::AIC);
    scheduler.is_scheduler = 1;
    scheduler.scheduler_count = 1;
    scheduler.scheduler_index = 0;
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
        scheduler.cluster_worker_ids[lane] = lane;
    storage.metadata[1].flags |= SCHEDULER_TASK_MIX;
    storage.metadata[1].active_mask = (LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false));
    uint64_t *callables = storage.callable_addresses;
    callables[1] = 0x1000;
    storage.prepare_ready_pool();
    auto *local = storage.local_context(&scheduler, &storage.scheduler_local_state);
    SchedulerReadyBatch batch{};

    for (int64_t task_ref = 0; task_ref < 2; ++task_ref) {
        auto *control = scheduler_task_control_at(storage.scheduler_state->base(), local, task_ref);
        control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        ASSERT_TRUE(scheduler_ready_batch_append(storage.scheduler_state->base(), local, task_ref, &batch));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(storage.scheduler_state->base(), local, 0, &batch));

    bool failed = false;
    EXPECT_TRUE(scheduler_fill_own_slots(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &failed
    ));
    EXPECT_FALSE(failed);
    EXPECT_EQ(local->own_slots()[0].task_ref, 0);
    EXPECT_EQ(local->own_slots()[0].state, SchedulerDispatchSlotState::READY);
    EXPECT_EQ(local->own_slots()[1].task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(local->own_slots()[1].state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(local->local_ready_mask, 1u);
    EXPECT_EQ(storage.run_control->scheduler_error, 0u);
    EXPECT_FALSE(scheduler_fill_own_slots(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &failed
    ));
    EXPECT_FALSE(failed);
    // The later task remains in the shared pool for another execution core.
    SchedulerReadyClaim remaining{};
    ASSERT_TRUE(scheduler_claim_ready_for_slot(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, 1, 0, &remaining
    ));
    EXPECT_EQ(remaining.task_ref, 1);
}

TEST(SchedulerLocalReady, OlderDispatchPrecedesReusedLowerSlot) {
    SchedulerLocalState scheduler_local_state{};
    scheduler_local_state.config.self_lane = 1;
    scheduler_local_state.slots[0].generation = 1;
    scheduler_local_state.slots[1].generation = 2;
    scheduler_local_state.slots[0].state = SchedulerDispatchSlotState::READY;
    scheduler_local_state.slots[1].state = SchedulerDispatchSlotState::READY;
    const uint64_t slot_0_publication = scheduler_dispatch_publication(1, SchedulerDispatchSlotState::READY);
    const uint64_t slot_1_publication = scheduler_dispatch_publication(2, SchedulerDispatchSlotState::READY);
    scheduler_local_ready_publish(&scheduler_local_state, 0);
    scheduler_local_ready_publish(&scheduler_local_state, 1);

    uint32_t pending_slot = UINT32_MAX;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(&scheduler_local_state, &pending_slot, &publication));
    EXPECT_EQ(pending_slot, 0u);
    EXPECT_EQ(publication, slot_0_publication);

    scheduler_local_state.slots[0].generation = 3;
    scheduler_local_ready_publish(&scheduler_local_state, 0);
    ASSERT_TRUE(scheduler_local_ready_pop(&scheduler_local_state, &pending_slot, &publication));
    EXPECT_EQ(pending_slot, 1u);
    EXPECT_EQ(publication, slot_1_publication);

    ASSERT_TRUE(scheduler_local_ready_pop(&scheduler_local_state, &pending_slot, &publication));
    EXPECT_EQ(pending_slot, 0u);
    EXPECT_EQ(publication, scheduler_dispatch_publication(3, SchedulerDispatchSlotState::READY));
    EXPECT_EQ(scheduler_local_state.local_ready_mask, 0u);
}

TEST(SchedulerLocalReady, InvalidSlotStateIsNotReconstructedAsReady) {
    SchedulerLocalState local{};
    local.config.self_lane = 1;
    local.slots[0].generation = 7;
    local.slots[0].state = SchedulerDispatchSlotState::FREE;
    scheduler_local_ready_publish(&local, 0);
    uint32_t slot = UINT32_MAX;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(&local, &slot, &publication));
    EXPECT_NE(scheduler_dispatch_state(publication), SchedulerDispatchSlotState::READY);
    EXPECT_FALSE(scheduler_local_ready_pop(&local, &slot, &publication));
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

TEST(SchedulerLocalConfig, SnapshotsImmutableConfigurationAndReloadsOnNextRun) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph_buffer(2);
    storage.prepare_ready_pool();
    const SchedulerGraphView graph = graph_buffer.graph();
    for (uint32_t self_lane : {1u, 2u}) {
        configure_cached_cluster(storage);
        SchedulerLocalState local{};
        auto &context = storage.contexts[self_lane];
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        EXPECT_EQ(local.config.self_lane, self_lane);
        EXPECT_EQ(local.config.worker_ids[0], 0u);
        EXPECT_EQ(scheduler_completion_id(&local, 7), 7 * 3 + self_lane);
        const uint64_t payload_offset = local.dispatch_payload_offset(2, 0);
        context.scheduler_count = 2;
        context.scheduler_index = 1;
        for (auto lane = 0u; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
            storage.contexts[lane].dispatch_payload_offset += 128;
        EXPECT_EQ(local.config.scheduler_count, 1u);
        EXPECT_EQ(local.config.scheduler_index, 0u);
        EXPECT_EQ(local.dispatch_payload_offset(2, 0), payload_offset);
        SchedulerLocalState next_run{};
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &next_run));
        EXPECT_EQ(next_run.config.scheduler_count, 2u);
        EXPECT_EQ(next_run.config.scheduler_index, 1u);
        EXPECT_EQ(next_run.dispatch_payload_offset(2, 0), payload_offset + 128);
        EXPECT_EQ(next_run.profiling, nullptr);
        EXPECT_EQ(next_run.pending_completed, 0u);
    }
}

TEST(SchedulerLocalConfig, RejectsTruncatedOffsetsAndInconsistentPayloadRoutes) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    storage.prepare_ready_pool();
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
    context.dispatch_payload_offset = (UINT64_C(1) << 32) + 2 * SCHEDULER_PENDING_SLOT_COUNT * sizeof(DispatchPayload);
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.dispatch_payload_offset = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.dispatch_payload_offset = original;
    ++storage.contexts[0].dispatch_payload_offset;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    --storage.contexts[0].dispatch_payload_offset;
    storage.contexts[0].core_type = 2;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    EXPECT_EQ(local.dispatch_payload_offset(2, 1), original + sizeof(DispatchPayload));
}

TEST(SchedulerLocalConfig, RejectsInvalidTopologyBeforeMailboxAccess) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    storage.prepare_ready_pool();
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
        graph_buffer.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, nullptr
    ));
    EXPECT_EQ(local->pending_completed, 1u);
    EXPECT_EQ(storage.run_control->resolved_task_count, 0u);
    EXPECT_FALSE(scheduler_service_cluster_completions(
        graph_buffer.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, nullptr
    ));
    EXPECT_EQ(local->pending_completed, 1u);
    scheduler_flush_completions(storage.run_control, local);
    EXPECT_EQ(local->pending_completed, 0u);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    scheduler_flush_completions(storage.run_control, local);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

TEST(SchedulerNormalDispatch, NoUsableCapacityDoesNotAccessPool) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    configure_normal_aiv_cluster(storage, 1);
    auto *local = storage.local_context(&storage.contexts[1], &storage.scheduler_local_state);
    // The invalid pool offset makes an accidental ready load fail.
    local->config.ready_pool_offset = UINT64_MAX;
    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
        local->own_slots()[slot].state = SchedulerDispatchSlotState::FILLING;
    bool failed = true;
    EXPECT_FALSE(scheduler_fill_own_slots(
        graph_buffer.graph(), storage.scheduler_state->base(), local, storage.run_control, 0, 0, &failed
    ));
    EXPECT_FALSE(failed);
    const uint32_t own_type = local->config.self_core_type;
    EXPECT_FALSE(scheduler_has_usable_slot(local, own_type, 0));

    // A free slot is capacity unless a completion pass already refilled it.
    local->own_slots()[1].state = SchedulerDispatchSlotState::FREE;
    EXPECT_TRUE(scheduler_has_usable_slot(local, own_type, 0));
    EXPECT_FALSE(scheduler_has_usable_slot(local, own_type, UINT64_C(1) << 1));
    // Work of another type is never this core's capacity.
    EXPECT_FALSE(scheduler_has_usable_slot(local, own_type == 0 ? 1 : 0, 0));
}

TEST(SchedulerDispatch, SequenceExhaustionDoesNotPublishOrWrap) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    auto *local = storage.local_context(&storage.contexts[0]);
    local->dispatch_sequence = UINT32_MAX;
    local->slots[0].state = SchedulerDispatchSlotState::FILLING;
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    const auto metadata = scheduler_load_dispatch_metadata(storage.scheduler_state->base(), local, 0, 0);
    EXPECT_FALSE(scheduler_prepare_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), local, storage.run_control, {0, 0, 0, 0}, ready, metadata, 0, 0
    ));
    EXPECT_EQ(local->dispatch_sequence, UINT32_MAX);
    EXPECT_EQ(local->local_ready_mask, 0u);
}

}  // namespace
