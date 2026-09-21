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

#include "scheduler/scheduler_dispatch.h"
#include "runtime_types.h"
#include "hbg_scheduler_test_support.h"

namespace {

using scheduler_test::FixtureStorage;
using scheduler_test::GraphBuffer;
using scheduler_test::SchedulerStateBuffer;

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
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    for (uint64_t worker = 0; worker < 3; ++worker) {
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
            scheduler_initialize_free_slot(
                scheduler_dispatch_slot_at(
                    storage.scheduler_state->base(), storage.local_context(&scheduler), worker, slot
                ),
                &storage.scheduler_local_state.slots[worker][slot]
            );
    }
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
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state), 1, &batch,
        &ready_stats
    ));
}

void occupy_normal_slot(
    FixtureStorage &storage, SchedulerWorkerContext &scheduler, uint64_t worker_id, uint32_t pending_slot,
    int64_t task_id
) {
    auto *slot = scheduler_dispatch_slot_at(
        storage.scheduler_state->base(), storage.local_context(&scheduler), worker_id, pending_slot
    );
    SchedulerLocalSlotState &local_slot = storage.scheduler_local_state.slots[worker_id][pending_slot];
    slot->task_id = task_id;
    local_slot.task_id = task_id;
    local_slot.subtask_slot = 1;
    local_slot.state = SchedulerDispatchSlotState::READY;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(local_slot.generation, SchedulerDispatchSlotState::READY)
    );
}

SchedulerDispatchSlot *
prepare_completed_normal_slot(FixtureStorage &storage, SchedulerWorkerContext &scheduler, uint64_t worker_id = 0) {
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = worker_id;
    scheduler.cluster_worker_ids[1] = scheduler.worker_index;
    scheduler.cluster_worker_ids[2] = UINT64_MAX;
    storage.run_control->scheduler_count = 1;
    auto *slot =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), worker_id, 0);
    SchedulerLocalSlotState &local_slot = storage.scheduler_local_state.slots[0][0];
    scheduler_initialize_free_slot(slot, &local_slot);
    slot->task_id = 0;
    local_slot.task_id = 0;
    local_slot.subtask_slot = 0;
    local_slot.state = SchedulerDispatchSlotState::READY;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(local_slot.generation, SchedulerDispatchSlotState::READY)
    );
    auto *completion_line =
        scheduler_completion_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), worker_id);
    completion_line->completed_generations[0] = local_slot.generation;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    return slot;
}

TEST(SchedulerCompletionInbox, PacksBothGenerationSlotsInOneDeviceWord) {
    alignas(uint64_t) volatile uint32_t generations[SCHEDULER_PENDING_SLOT_COUNT] = {
        UINT32_C(0x11223344), UINT32_C(0x55667788)
    };
    EXPECT_EQ(scheduler_gm_query_u32_pair(generations), UINT64_C(0x5566778811223344));
    scheduler_gm_store(generations[1], UINT32_C(0));
    EXPECT_EQ(scheduler_gm_query_u32_pair(generations), UINT64_C(0x0000000011223344));
}

TEST(SchedulerCompletionInbox, TracksConsumedGenerationsPerSlot) {
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

TEST(SchedulerClusterCompletion, SpscGenerationCompletesNormalTask) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = UINT64_MAX;
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0, 0);
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.slots[0][0];
    scheduler_initialize_free_slot(slot, local_slot);
    slot->task_id = 0;
    local_slot->task_id = 0;
    local_slot->subtask_slot = 0;
    local_slot->sampled_task_timing = 1;
    local_slot->state = SchedulerDispatchSlotState::READY;
    slot->executor_trace.kernel_start_cycles = 100;
    slot->executor_trace.kernel_end_cycles = 200;
    storage.metadata[0].timing_slot = 0;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
    );
    auto *completion_line =
        scheduler_completion_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completion_line->completed_generations[0] = slot->generation;
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, nullptr, 0, nullptr
    ));
    EXPECT_EQ(completion_line->completed_generations[0], slot->generation);
    EXPECT_EQ(local_slot->state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::READY);
    EXPECT_EQ(control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(control->wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    EXPECT_FALSE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, nullptr, 0, nullptr
    ));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[0].kernel_start_cycles, 100u);
    EXPECT_EQ(traces[0].kernel_end_cycles, 200u);
    EXPECT_EQ(traces[0].valid, 0u);
}

TEST(SchedulerClusterCompletion, RejectsStaleCompletionGenerationAtNamedSite) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.cluster_worker_ids[0] = 0;
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0, 0);
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.slots[0][0];
    scheduler_initialize_free_slot(slot, local_slot);
    slot->task_id = 0;
    local_slot->task_id = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
    );

    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0,
        slot->generation + 1, &wake_stats, &ready_stats, &completion_stats, nullptr, false, nullptr, nullptr
    ));
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_GENERATION_MISMATCH)
    );
}

TEST(SchedulerClusterCompletion, AccountsCompletedTaskWhenResolveFails) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    SchedulerDispatchSlot *slot = prepare_completed_normal_slot(storage, scheduler);

    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;
    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0, slot->generation,
        nullptr, nullptr, nullptr, nullptr, 0, nullptr, nullptr
    ));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    EXPECT_NE(storage.run_control->scheduler_error, 0u);
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_WAKE_ALREADY_CLOSED)
    );
}

TEST(SchedulerClusterCompletion, AccountsCompletedTaskWhenRefillClaimFails) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    SchedulerDispatchSlot *slot = prepare_completed_normal_slot(storage, scheduler);
    SchedulerLocalState &owner_state = storage.scheduler_local_state;
    owner_state.owner_pending_endpoints[0] = scheduler_ready_pending_pack(SCHEDULER_INBOX_EMPTY, 0);
    uint64_t ready_victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};

    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0, slot->generation,
        nullptr, nullptr, nullptr, ready_victim_cursors, 0, nullptr, nullptr
    ));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    EXPECT_NE(storage.run_control->scheduler_error, 0u);
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::COMPLETION_REFILL_CLAIM_FAILED)
    );
}

TEST(SchedulerClusterCompletion, AccountsCompletedTaskWhenRefillDispatchFails) {
    FixtureStorage storage(1, 2);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    SchedulerDispatchSlot *slot = prepare_completed_normal_slot(storage, scheduler);
    SchedulerReadyClaim replacement{};
    replacement.task_id = 1;

    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0, slot->generation,
        nullptr, nullptr, nullptr, nullptr, 0, &replacement, nullptr
    ));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
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
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
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
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0, 0);
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.slots[0][0];
    scheduler_initialize_free_slot(slot, local_slot);
    slot->task_id = 0;
    local_slot->task_id = 0;
    local_slot->subtask_slot = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    slot->executor_trace.kernel_start_cycles = 100;
    slot->executor_trace.kernel_end_cycles = 200;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::READY)
    );
    auto *completion_line =
        scheduler_completion_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completion_line->completed_generations[0] = slot->generation;
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
        &ready_stats, &completion_stats, nullptr, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, nullptr
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
    EXPECT_EQ(slot->task_id, 1);
    EXPECT_EQ(traces[1].ready_source, static_cast<uint64_t>(SchedulerReadySource::DIRECT_RESOLVE));
    EXPECT_EQ(traces[1].state_probe_scheduler_worker_id, UINT64_MAX);
    EXPECT_EQ(ready_stats.enqueue_count, 0u);
    EXPECT_EQ(wake_stats.fanin_state_load_count, 0u);
}

TEST(SchedulerClusterCompletion, DirectlyRefillsCompletedSlotWhenReadyTaskExists) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;

    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0, 0);
    SchedulerLocalSlotState *local_slot = &storage.scheduler_local_state.slots[0][0];
    scheduler_initialize_free_slot(slot, local_slot);
    const uint32_t completed_generation = local_slot->generation;
    slot->task_id = 0;
    local_slot->task_id = 0;
    local_slot->subtask_slot = 0;
    local_slot->state = SchedulerDispatchSlotState::READY;
    scheduler_gm_store(
        slot->publication, scheduler_dispatch_publication(completed_generation, SchedulerDispatchSlotState::READY)
    );
    auto *completion_line =
        scheduler_completion_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completion_line->completed_generations[0] = completed_generation;
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
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state), 0, &batch,
        &ready_stats
    ));

    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    uint64_t ready_victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t direct_refilled_slot_mask = 0;
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, ready_victim_cursors, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL,
        &direct_refilled_slot_mask
    ));

    EXPECT_EQ(completion_line->completed_generations[0], completed_generation);
    EXPECT_EQ(slot->task_id, 1);
    EXPECT_EQ(slot->generation, completed_generation + 1);
    EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::READY);
    EXPECT_EQ(completed_control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    EXPECT_EQ(direct_refilled_slot_mask, 1u);
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[1].ready_source, static_cast<uint64_t>(SchedulerReadySource::LOCAL));
    EXPECT_EQ(traces[1].publication_mode, static_cast<uint64_t>(SchedulerPublicationMode::REFILL));
    EXPECT_LE(traces[1].state_probe_start_cycles, traces[1].state_probe_end_cycles);
    EXPECT_EQ(traces[1].state_probe_end_cycles, traces[0].refill_start_cycles);

    const uint32_t refilled_generation = local_slot->generation;
    completion_line->completed_generations[0] = refilled_generation;
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, ready_victim_cursors, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, nullptr
    ));
    EXPECT_EQ(completion_line->completed_generations[0], refilled_generation);
    EXPECT_EQ(storage.run_control->resolved_task_count, 2u);
}

TEST(SchedulerClusterCompletion, UsesSchedulerLocalSlotStateWithoutRereadingDispatchMetadata) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_index = 0;
    scheduler.scheduler_count = 1;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;

    SchedulerLocalState scheduler_local_state{};
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0, 0);
    SchedulerLocalSlotState *local_slot = &scheduler_local_state.slots[0][0];
    scheduler_initialize_free_slot(slot, local_slot);
    SchedulerReadyClaim producer_ready{};
    producer_ready.task_id = 0;
    ASSERT_TRUE(scheduler_fill_dispatch_slot(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, SchedulerFreeSlotClaim{0, 0, local_slot->generation, 0}, producer_ready,
        SCHEDULER_PROFILING_TASK_TIMING_LEVEL
    ));
    ASSERT_EQ(
        scheduler_bootstrap_route_task(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler), storage.run_control, 1,
            nullptr
        ),
        SchedulerRouteResult::WAITING
    );

    const uint32_t completed_generation = local_slot->generation;
    slot->executor_trace.kernel_start_cycles = 100;
    slot->executor_trace.kernel_end_cycles = 200;
    slot->task_id = 999;
    slot->generation = completed_generation + 99;
    slot->publication = scheduler_dispatch_publication(slot->generation, SchedulerDispatchSlotState::EMPTY);
    auto *completion_line =
        scheduler_completion_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completion_line->completed_generations[0] = completed_generation;

    SchedulerReadyStats ready_stats{};
    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, nullptr, &ready_stats, nullptr, victim_cursors, SCHEDULER_PROFILING_TASK_TIMING_LEVEL,
        nullptr
    ));

    EXPECT_EQ(local_slot->task_id, 1);
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
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    target.trace_aicore_entry_cycles = 11;
    target.trace_handshake_publish_cycles = 12;
    target.trace_register_release_cycles = 13;
    target.trace_descriptor_cache_observed_cycles = 14;

    SchedulerLocalState scheduler_local_state{};
    for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
        auto *slot = scheduler_dispatch_slot_at(
            storage.scheduler_state->base(), storage.local_context(&scheduler), 0, pending_slot
        );
        SchedulerLocalSlotState *local_slot = &scheduler_local_state.slots[0][pending_slot];
        scheduler_initialize_free_slot(slot, local_slot);
        SchedulerReadyClaim ready{};
        ready.task_id = pending_slot;
        ASSERT_TRUE(scheduler_fill_dispatch_slot(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
            storage.run_control, SchedulerFreeSlotClaim{0, pending_slot, local_slot->generation, 0}, ready,
            SCHEDULER_PROFILING_SCHED_PHASES_LEVEL
        ));
        slot->executor_trace.kernel_start_cycles = 100 + pending_slot;
        slot->executor_trace.kernel_end_cycles = 200 + pending_slot;
    }

    auto *completion_line =
        scheduler_completion_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completion_line->completed_generations[0] = scheduler_local_state.slots[0][0].generation;
    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, 0, 0, scheduler_local_state.slots[0][0].generation, nullptr, nullptr, nullptr, nullptr,
        SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, nullptr, nullptr
    ));
    ASSERT_TRUE((scheduler_local_state.worker_trace_valid_mask & 1U) != 0);

    target.trace_aicore_entry_cycles = 21;
    target.trace_handshake_publish_cycles = 22;
    target.trace_register_release_cycles = 23;
    target.trace_descriptor_cache_observed_cycles = 24;
    completion_line->completed_generations[1] = scheduler_local_state.slots[0][1].generation;
    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, 0, 1, scheduler_local_state.slots[0][1].generation, nullptr, nullptr, nullptr, nullptr,
        SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, nullptr, nullptr
    ));

    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[1].aicore_entry_cycles, 11u);
    EXPECT_EQ(traces[1].handshake_publish_cycles, 12u);
    EXPECT_EQ(traces[1].register_release_cycles, 13u);
    EXPECT_EQ(traces[1].descriptor_cache_observed_cycles, 14u);
}

TEST(SchedulerClusterCompletion, DeferredRefillPreservesOriginalStateProbeAndReadySource) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0);
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    SchedulerDispatchSlot *slot = prepare_completed_normal_slot(storage, scheduler);
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;

    SchedulerReadyClaim replacement{};
    replacement.task_id = 1;
    replacement.source = SchedulerReadySource::STOLEN;
    replacement.state_probe_start_cycles = 123;
    replacement.state_probe_end_cycles = 456;
    bool refilled = false;

    ASSERT_TRUE(scheduler_service_cluster_completion_slot(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, 0, 0, slot->generation,
        nullptr, nullptr, nullptr, nullptr, SCHEDULER_PROFILING_SCHED_PHASES_LEVEL, &replacement, &refilled
    ));

    EXPECT_TRUE(refilled);
    auto *traces =
        scheduler_state_at<SchedulerTaskTrace>(storage.scheduler_state->base(), storage.layout.trace_cells_offset);
    EXPECT_EQ(traces[1].ready_source, static_cast<uint64_t>(SchedulerReadySource::STOLEN));
    EXPECT_EQ(traces[1].publication_mode, static_cast<uint64_t>(SchedulerPublicationMode::REFILL));
    EXPECT_EQ(traces[1].state_probe_start_cycles, 123u);
    EXPECT_EQ(traces[1].state_probe_end_cycles, 456u);
}

TEST(SchedulerNormalDispatch, FillsFreshAicSlot) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 0);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_count = 1;
    scheduler.scheduler_index = 0;
    scheduler.cluster_worker_ids[0] = UINT64_MAX;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 0;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    auto *slot = scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0, 0);
    scheduler_initialize_free_slot(slot, &storage.scheduler_local_state.slots[2][0]);
    auto *control = scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    ASSERT_TRUE(scheduler_ready_batch_append(
        storage.scheduler_state->base(), storage.local_context(&scheduler), 0, &batch, &ready_stats
    ));
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state), 0, &batch,
        &ready_stats
    ));

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, nullptr, nullptr
    ));
    EXPECT_EQ(slot->task_id, 0);
    EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::READY);
}

TEST(SchedulerNormalDispatch, PreservesProgressWhenALaterFillFails) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.mixed(1, 3);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.is_scheduler = 1;
    scheduler.scheduler_count = 1;
    scheduler.scheduler_index = 0;
    scheduler.cluster_worker_ids[0] = 0;
    scheduler.cluster_worker_ids[1] = 1;
    scheduler.cluster_worker_ids[2] = 2;
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    storage.metadata[1].active_mask = 3;
    auto *callables =
        scheduler_state_at<uint64_t>(storage.scheduler_state->base(), storage.layout.callable_addresses_offset);
    callables[1] = 0x1000;
    for (uint32_t slot_index = 0; slot_index < SCHEDULER_PENDING_SLOT_COUNT; ++slot_index)
        scheduler_initialize_free_slot(
            scheduler_dispatch_slot_at(
                storage.scheduler_state->base(), storage.local_context(&scheduler), 0, slot_index
            ),
            &storage.scheduler_local_state.slots[0][slot_index]
        );
    SchedulerReadyBatch batch{};
    SchedulerReadyStats ready_stats{};
    for (int64_t task_id = 0; task_id < 2; ++task_id) {
        auto *control =
            scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), task_id);
        control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
        ASSERT_TRUE(scheduler_ready_batch_append(
            storage.scheduler_state->base(), storage.local_context(&scheduler), task_id, &batch, &ready_stats
        ));
    }
    ASSERT_TRUE(scheduler_ready_batch_push(
        storage.scheduler_state->base(), storage.local_context(&scheduler, &storage.scheduler_local_state), 0, &batch,
        &ready_stats
    ));

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    bool failed = false;
    EXPECT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, nullptr, &failed
    ));
    EXPECT_TRUE(failed);
    auto *first_slot =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0, 0);
    EXPECT_EQ(first_slot->task_id, 0);
    EXPECT_EQ(scheduler_dispatch_state(first_slot->publication), SchedulerDispatchSlotState::READY);
    auto *failed_slot =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0, 1);
    EXPECT_EQ(failed_slot->task_id, SCHEDULER_TASK_ID_INVALID);
    EXPECT_EQ(storage.scheduler_local_state.slots[0][1].state, SchedulerDispatchSlotState::FILLING);
    EXPECT_EQ(scheduler_dispatch_state(failed_slot->publication), SchedulerDispatchSlotState::FREE);
    EXPECT_NE(storage.run_control->scheduler_error, 0u);
    EXPECT_EQ(storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_SHAPE));
}

TEST(SchedulerNormalDispatch, RejectsExcessAivWorkersAtNamedSite) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIV);
    enqueue_normal_aiv_tasks(storage, scheduler, 0, 1);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    bool failed = false;
    EXPECT_FALSE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, &deferred, &failed
    ));
    EXPECT_TRUE(failed);
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::NORMAL_DISPATCH_INVALID_TOPOLOGY)
    );
}

TEST(SchedulerNormalDispatch, PublishesOrdinaryAivToPeerBeforeScheduler) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    enqueue_normal_aiv_tasks(storage, scheduler, 0, 1);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, &deferred, nullptr
    ));
    auto *peer_slot =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 2, 0);
    EXPECT_EQ(peer_slot->task_id, 0);
    EXPECT_EQ(scheduler_dispatch_state(peer_slot->publication), SchedulerDispatchSlotState::READY);
    EXPECT_EQ(deferred.count, 0u);
}

TEST(SchedulerDeferredAiv, ReservesOnlyAvailableSchedulerSlotsBeforeClaiming) {
    FixtureStorage storage(3, 3);
    GraphBuffer graph(3);
    for (uint64_t task = 0; task < 3; ++task)
        graph.executable(task, 1);
    configure_normal_aiv_cluster(storage, 3);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    occupy_normal_slot(storage, scheduler, 2, 0, SCHEDULER_TASK_ID_INVALID);
    occupy_normal_slot(storage, scheduler, 2, 1, SCHEDULER_TASK_ID_INVALID);
    enqueue_normal_aiv_tasks(storage, scheduler, 0, 3);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, &deferred, nullptr
    ));

    ASSERT_EQ(deferred.count, SCHEDULER_PENDING_SLOT_COUNT);
    for (uint32_t slot_index = 0; slot_index < SCHEDULER_PENDING_SLOT_COUNT; ++slot_index) {
        auto *slot = scheduler_dispatch_slot_at(
            storage.scheduler_state->base(), storage.local_context(&scheduler), 1, slot_index
        );
        EXPECT_EQ(storage.scheduler_local_state.slots[1][slot_index].state, SchedulerDispatchSlotState::FILLING);
        EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::FREE);
        EXPECT_EQ(slot->task_id, SCHEDULER_TASK_ID_INVALID);
    }
    auto *ready_inbox =
        scheduler_ready_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1, 0);
    EXPECT_NE(ready_inbox->head, SCHEDULER_INBOX_EMPTY);
}

TEST(SchedulerDeferredAiv, RejectsInvalidReservationBeforeSlotAccess) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    SchedulerDeferredAivQueue deferred{};
    deferred.count = 1;
    deferred.entries[0].ready.task_id = 0;
    deferred.entries[0].reserved_slot = {UINT64_MAX, UINT32_MAX, 1};

    uint32_t published_slot = 0;
    EXPECT_FALSE(scheduler_publish_deferred_aiv_local(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &deferred, false,
        &published_slot
    ));
    EXPECT_EQ(published_slot, UINT32_MAX);
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DEFERRED_PUBLISH_INVALID_RESERVATION)
    );
}

TEST(SchedulerDeferredAiv, RejectsReservationWithMaterializedTask) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    SchedulerLocalSlotState &local_slot = storage.scheduler_local_state.slots[1][0];
    local_slot.task_id = 0;
    local_slot.state = SchedulerDispatchSlotState::FILLING;
    SchedulerDeferredAivQueue deferred{};
    deferred.count = 1;
    deferred.entries[0].ready.task_id = 0;
    deferred.entries[0].reserved_slot = {scheduler.worker_index, 0, local_slot.generation, 1};

    uint32_t published_slot = 0;
    EXPECT_FALSE(scheduler_publish_deferred_aiv_local(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &deferred, false,
        &published_slot
    ));
    EXPECT_EQ(published_slot, UINT32_MAX);
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DEFERRED_PUBLISH_INVALID_RESERVATION)
    );
}

TEST(SchedulerDeferredAiv, RejectsReservationOwnedByPeerAtNamedSite) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];

    EXPECT_FALSE(scheduler_release_deferred_aiv_reservation(
        graph.graph(), storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control,
        SchedulerFreeSlotClaim{2, 0, 1, 0}
    ));
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DEFERRED_RESERVATION_INVALID_OWNER)
    );
}

TEST(SchedulerDeferredAiv, RejectsReservationWithInvalidStateAtNamedSite) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    const SchedulerLocalSlotState &local_slot = storage.scheduler_local_state.slots[1][0];

    EXPECT_FALSE(scheduler_release_deferred_aiv_reservation(
        graph.graph(), storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control,
        SchedulerFreeSlotClaim{scheduler.worker_index, 0, local_slot.generation, 1}
    ));
    EXPECT_EQ(
        storage.run_control->error_site, static_cast<uint64_t>(SchedulerErrorSite::DEFERRED_RESERVATION_INVALID_STATE)
    );
}

TEST(SchedulerDeferredAiv, DoesNotClaimWithoutSchedulerReservation) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    for (uint64_t worker = 1; worker <= 2; ++worker) {
        for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot)
            occupy_normal_slot(storage, scheduler, worker, slot, SCHEDULER_TASK_ID_INVALID);
    }
    enqueue_normal_aiv_tasks(storage, scheduler, 0, 1);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    bool failed = true;
    EXPECT_FALSE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, &deferred, &failed
    ));
    EXPECT_FALSE(failed);
    EXPECT_EQ(deferred.count, 0u);
    auto *ready_inbox =
        scheduler_ready_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1, 0);
    EXPECT_EQ(ready_inbox->head, 0);
}

TEST(SchedulerDeferredAiv, KeepsReservationForSchedulerWhenNoPeerIsActive) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph(1);
    graph.executable(0, 1);
    configure_normal_aiv_cluster(storage, 1);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.cluster_worker_ids[2] = UINT64_MAX;
    enqueue_normal_aiv_tasks(storage, scheduler, 0, 1);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, &deferred, nullptr
    ));
    ASSERT_EQ(deferred.count, 1u);

    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    EXPECT_TRUE(scheduler_drain_deferred_aiv_to_peer(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &deferred, &wake_stats,
        &ready_stats, &completion_stats, 0
    ));
    EXPECT_EQ(deferred.count, 1u);

    uint32_t self_slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_publish_deferred_aiv_local(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &deferred, 0, &self_slot
    ));
    EXPECT_EQ(deferred.count, 0u);
    EXPECT_LT(self_slot, SCHEDULER_PENDING_SLOT_COUNT);
}

TEST(SchedulerDeferredAiv, LocalReservationsAndReadyHandoffsDoNotPollSharedState) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 1);
    graph.executable(1, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    scheduler.cluster_worker_ids[2] = UINT64_MAX;
    storage.metadata[0].timing_slot = 3;
    storage.metadata[1].timing_slot = 4;

    SchedulerLocalState scheduler_local_state{};
    for (uint32_t cluster_lane = 0; cluster_lane < PLATFORM_CORES_PER_BLOCKDIM; ++cluster_lane) {
        const uint64_t worker_id = scheduler.cluster_worker_ids[cluster_lane];
        if (worker_id >= scheduler.runtime_worker_count) continue;
        for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
            auto *slot = scheduler_dispatch_slot_at(
                storage.scheduler_state->base(), storage.local_context(&scheduler), worker_id, pending_slot
            );
            scheduler_initialize_free_slot(slot, &scheduler_local_state.slots[cluster_lane][pending_slot]);
        }
    }
    enqueue_normal_aiv_tasks(storage, scheduler, 0, 2);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
        storage.run_control, victim_cursors, &ready_stats, 0, 0, &deferred, nullptr
    ));
    ASSERT_EQ(deferred.count, 2u);
    for (uint32_t index = 0; index < deferred.count; ++index) {
        const SchedulerFreeSlotClaim &reservation = deferred.entries[index].reserved_slot;
        const SchedulerLocalSlotState &local_slot =
            scheduler_local_state.slots[reservation.cluster_lane][reservation.slot_index];
        auto *slot = scheduler_dispatch_slot_at(
            storage.scheduler_state->base(), storage.local_context(&scheduler), reservation.worker_id,
            reservation.slot_index
        );
        EXPECT_EQ(local_slot.state, SchedulerDispatchSlotState::FILLING);
        EXPECT_EQ(scheduler_dispatch_state(slot->publication), SchedulerDispatchSlotState::FREE);
    }

    for (uint32_t index = 0; index < SCHEDULER_PENDING_SLOT_COUNT; ++index) {
        uint32_t published_slot = UINT32_MAX;
        ASSERT_TRUE(scheduler_publish_deferred_aiv_local(
            graph.graph(), storage.scheduler_state->base(), storage.local_context(&scheduler, &scheduler_local_state),
            storage.run_control, &deferred, 0, &published_slot
        ));
        ASSERT_LT(published_slot, SCHEDULER_PENDING_SLOT_COUNT);
    }
    EXPECT_EQ(scheduler_local_state.local_ready_mask, UINT32_C(3));
    // Leave both notifications pending across Scheduler passes. Neither scanning
    // completions nor looking for free capacity may overwrite their generations.
    const auto first_generation = scheduler_local_state.slots[1][0].generation;
    const auto second_generation = scheduler_local_state.slots[1][1].generation;
    for (uint32_t pass = 0; pass < 3; ++pass) {
        EXPECT_FALSE(scheduler_service_cluster_completions(
            graph.graph(), storage.scheduler_state->base(), &scheduler_local_state, storage.run_control, nullptr,
            nullptr, nullptr, victim_cursors, 0, nullptr
        ));
        bool failed = false;
        EXPECT_FALSE(scheduler_fill_cluster_normal_slots(
            graph.graph(), storage.scheduler_state->base(), &scheduler_local_state, storage.run_control, victim_cursors,
            &ready_stats, 0, 0, &deferred, &failed
        ));
        EXPECT_FALSE(failed);
        EXPECT_EQ(scheduler_local_state.local_ready_mask, 3u);
        EXPECT_EQ(scheduler_local_state.slots[1][0].generation, first_generation);
        EXPECT_EQ(scheduler_local_state.slots[1][1].generation, second_generation);
    }

    uint64_t expected_publications[SCHEDULER_PENDING_SLOT_COUNT]{};
    for (uint32_t pending_slot = 0; pending_slot < SCHEDULER_PENDING_SLOT_COUNT; ++pending_slot) {
        auto *slot = scheduler_dispatch_slot_at(
            storage.scheduler_state->base(), storage.local_context(&scheduler), scheduler.worker_index, pending_slot
        );
        expected_publications[pending_slot] = slot->publication;
        EXPECT_EQ(slot->timing_slot, static_cast<int32_t>(pending_slot + 3));
        slot->publication = scheduler_dispatch_publication(99, SchedulerDispatchSlotState::EMPTY);
    }
    for (uint32_t expected_slot = 0; expected_slot < SCHEDULER_PENDING_SLOT_COUNT; ++expected_slot) {
        uint32_t pending_slot = UINT32_MAX;
        uint64_t publication = 0;
        ASSERT_TRUE(scheduler_local_ready_pop(&scheduler_local_state, expected_slot, &pending_slot, &publication));
        EXPECT_EQ(pending_slot, expected_slot);
        EXPECT_EQ(publication, expected_publications[expected_slot]);
    }
    EXPECT_EQ(scheduler_local_state.local_ready_mask, 0u);
}

TEST(SchedulerLocalReady, RotatesPastRepublishedLowerSlot) {
    SchedulerLocalState scheduler_local_state{};
    scheduler_local_state.config.self_lane = 1;
    scheduler_local_state.slots[1][0].generation = 1;
    scheduler_local_state.slots[1][1].generation = 2;
    scheduler_local_state.slots[1][0].state = SchedulerDispatchSlotState::READY;
    scheduler_local_state.slots[1][1].state = SchedulerDispatchSlotState::READY;
    const uint64_t slot_0_publication = scheduler_dispatch_publication(1, SchedulerDispatchSlotState::READY);
    const uint64_t slot_1_publication = scheduler_dispatch_publication(2, SchedulerDispatchSlotState::READY);
    scheduler_local_ready_publish(&scheduler_local_state, 0);
    scheduler_local_ready_publish(&scheduler_local_state, 1);

    uint32_t pending_slot = UINT32_MAX;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(&scheduler_local_state, 0, &pending_slot, &publication));
    EXPECT_EQ(pending_slot, 0u);
    EXPECT_EQ(publication, slot_0_publication);

    scheduler_local_ready_publish(&scheduler_local_state, 0);
    ASSERT_TRUE(scheduler_local_ready_pop(&scheduler_local_state, 1, &pending_slot, &publication));
    EXPECT_EQ(pending_slot, 1u);
    EXPECT_EQ(publication, slot_1_publication);

    ASSERT_TRUE(scheduler_local_ready_pop(&scheduler_local_state, 0, &pending_slot, &publication));
    EXPECT_EQ(pending_slot, 0u);
    EXPECT_EQ(publication, slot_0_publication);
    EXPECT_EQ(scheduler_local_state.local_ready_mask, 0u);
}

TEST(SchedulerLocalReady, InvalidSlotStateIsNotReconstructedAsReady) {
    SchedulerLocalState local{};
    local.config.self_lane = 1;
    local.slots[1][0].generation = 7;
    local.slots[1][0].state = SchedulerDispatchSlotState::FREE;
    scheduler_local_ready_publish(&local, 0);
    uint32_t slot = UINT32_MAX;
    uint64_t publication = 0;
    ASSERT_TRUE(scheduler_local_ready_pop(&local, 0, &slot, &publication));
    EXPECT_NE(scheduler_dispatch_state(publication), SchedulerDispatchSlotState::READY);
    EXPECT_FALSE(scheduler_local_ready_pop(&local, 0, &slot, &publication));
}

TEST(SchedulerDeferredAiv, PrefersNewPeerCapacityAndSelfPublishesOnlyOne) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    for (uint64_t task = 0; task < 2; ++task)
        graph.executable(task, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    occupy_normal_slot(storage, scheduler, 2, 0, SCHEDULER_TASK_ID_INVALID);
    occupy_normal_slot(storage, scheduler, 2, 1, SCHEDULER_TASK_ID_INVALID);
    enqueue_normal_aiv_tasks(storage, scheduler, 0, 2);
    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, &deferred, nullptr
    ));
    ASSERT_EQ(deferred.count, 2u);
    const SchedulerFreeSlotClaim first_reservation = deferred.entries[0].reserved_slot;

    auto *peer_slot =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 2, 0);
    peer_slot->task_id = SCHEDULER_TASK_ID_INVALID;
    SchedulerLocalSlotState &peer_local_slot = storage.scheduler_local_state.slots[2][0];
    peer_local_slot.task_id = SCHEDULER_TASK_ID_INVALID;
    peer_local_slot.state = SchedulerDispatchSlotState::FREE;
    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(scheduler_drain_deferred_aiv_to_peer(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &deferred, &wake_stats,
        &ready_stats, &completion_stats, 0
    ));
    ASSERT_EQ(deferred.count, 1u);
    EXPECT_EQ(scheduler_dispatch_state(peer_slot->publication), SchedulerDispatchSlotState::READY);
    EXPECT_EQ(
        storage.scheduler_local_state.slots[first_reservation.cluster_lane][first_reservation.slot_index].state,
        SchedulerDispatchSlotState::FREE
    );

    uint32_t self_slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_publish_deferred_aiv_local(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &deferred, 0, &self_slot
    ));
    EXPECT_EQ(deferred.count, 0u);
    ASSERT_LT(self_slot, SCHEDULER_PENDING_SLOT_COUNT);
    auto *published =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1, self_slot);
    EXPECT_EQ(scheduler_dispatch_state(published->publication), SchedulerDispatchSlotState::READY);
}

TEST(SchedulerDeferredAiv, KeepsSecondReservationAfterOneSelfPublish) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    for (uint64_t task = 0; task < 2; ++task)
        graph.executable(task, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    occupy_normal_slot(storage, scheduler, 2, 0, SCHEDULER_TASK_ID_INVALID);
    occupy_normal_slot(storage, scheduler, 2, 1, SCHEDULER_TASK_ID_INVALID);
    enqueue_normal_aiv_tasks(storage, scheduler, 0, 2);
    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, &deferred, nullptr
    ));
    ASSERT_EQ(deferred.count, 2u);

    uint32_t self_slot = UINT32_MAX;
    ASSERT_TRUE(scheduler_publish_deferred_aiv_local(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &deferred, 0, &self_slot
    ));
    ASSERT_EQ(deferred.count, 1u);
    auto *published =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1, self_slot);
    EXPECT_EQ(scheduler_dispatch_state(published->publication), SchedulerDispatchSlotState::READY);
    const SchedulerFreeSlotClaim remaining = deferred.entries[0].reserved_slot;
    const SchedulerLocalSlotState &reserved =
        storage.scheduler_local_state.slots[remaining.cluster_lane][remaining.slot_index];
    EXPECT_EQ(reserved.state, SchedulerDispatchSlotState::FILLING);
    EXPECT_EQ(reserved.task_id, SCHEDULER_TASK_ID_INVALID);
}

TEST(SchedulerDeferredAiv, RetiresCompletedPeerAndRefillsWithoutFreeDecision) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 1);
    graph.executable(1, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    occupy_normal_slot(storage, scheduler, 2, 0, 0);
    occupy_normal_slot(storage, scheduler, 2, 1, SCHEDULER_TASK_ID_INVALID);
    occupy_normal_slot(storage, scheduler, 1, 1, SCHEDULER_TASK_ID_INVALID);
    auto *completed_control =
        scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completed_control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    enqueue_normal_aiv_tasks(storage, scheduler, 1, 2);

    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    SchedulerReadyStats ready_stats{};
    SchedulerDeferredAivQueue deferred{};
    ASSERT_TRUE(scheduler_fill_cluster_normal_slots(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, victim_cursors,
        &ready_stats, 0, 0, &deferred, nullptr
    ));
    ASSERT_EQ(deferred.count, 1u);
    auto *peer_slot =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 2, 0);
    const uint32_t completed_generation = storage.scheduler_local_state.slots[2][0].generation;
    auto *completion_line =
        scheduler_completion_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 2);
    completion_line->completed_generations[0] = completed_generation;

    SchedulerWakeStats wake_stats{};
    SchedulerCompletionStats completion_stats{};
    ASSERT_TRUE(scheduler_drain_deferred_aiv_to_peer(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &deferred, &wake_stats,
        &ready_stats, &completion_stats, 0
    ));
    EXPECT_EQ(deferred.count, 0u);
    EXPECT_EQ(completion_line->completed_generations[0], completed_generation);
    EXPECT_EQ(completed_control->state, static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
    EXPECT_EQ(peer_slot->task_id, 1);
    EXPECT_EQ(peer_slot->generation, completed_generation + 1);
    EXPECT_EQ(scheduler_dispatch_state(peer_slot->publication), SchedulerDispatchSlotState::READY);
    EXPECT_FALSE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, victim_cursors, 0, nullptr
    ));
    EXPECT_EQ(storage.run_control->resolved_task_count, 1u);
}

TEST(SchedulerDeferredAiv, SchedulerCompletionDoesNotDirectRefillItself) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph(2);
    graph.executable(0, 1);
    graph.executable(1, 1);
    configure_normal_aiv_cluster(storage, 2);
    SchedulerWorkerContext &scheduler = storage.contexts[1];
    occupy_normal_slot(storage, scheduler, 1, 0, 0);
    auto *completed_slot =
        scheduler_dispatch_slot_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1, 0);
    auto *completion_line =
        scheduler_completion_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1);
    completion_line->completed_generations[0] = completed_slot->generation;
    auto *completed_control =
        scheduler_task_control_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 0);
    completed_control->state = static_cast<int64_t>(SchedulerTaskState::BLOCKED);
    enqueue_normal_aiv_tasks(storage, scheduler, 1, 2);

    SchedulerWakeStats wake_stats{};
    SchedulerReadyStats ready_stats{};
    SchedulerCompletionStats completion_stats{};
    uint64_t victim_cursors[SCHEDULER_CORE_TYPE_COUNT]{};
    uint64_t direct_refilled_slot_mask = 0;
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), storage.scheduler_state->base(),
        storage.local_context(&scheduler, &storage.scheduler_local_state), storage.run_control, &wake_stats,
        &ready_stats, &completion_stats, victim_cursors, 0, &direct_refilled_slot_mask
    ));
    EXPECT_EQ(direct_refilled_slot_mask, 0u);
    EXPECT_EQ(storage.scheduler_local_state.slots[1][0].task_id, SCHEDULER_TASK_ID_INVALID);
    EXPECT_EQ(storage.scheduler_local_state.slots[1][0].state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(completed_slot->task_id, 0);
    EXPECT_EQ(scheduler_dispatch_state(completed_slot->publication), SchedulerDispatchSlotState::READY);
    auto *ready_inbox =
        scheduler_ready_inbox_at(storage.scheduler_state->base(), storage.local_context(&scheduler), 1, 0);
    EXPECT_EQ(ready_inbox->head, 1);
}

// Model READY's fully published physical cluster, including inactive peers.
void configure_cached_cluster(FixtureStorage &storage, uint32_t scheduler_lane) {
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        auto &context = storage.contexts[lane];
        context.core_type = static_cast<int32_t>(lane == 0 ? CoreType::AIC : CoreType::AIV);
        context.scheduler_count = 1;
        context.scheduler_index = 0;
        context.scheduler_worker_id = scheduler_lane;
        context.is_scheduler = lane == scheduler_lane;
        for (uint32_t peer = 0; peer < PLATFORM_CORES_PER_BLOCKDIM; ++peer)
            context.cluster_worker_ids[peer] = peer;
    }
}

TEST(SchedulerLocalConfig, SnapshotsReadyPublicationAndReloadsOnNextRun) {
    FixtureStorage storage(2, 3);
    GraphBuffer graph_buffer(2);
    const SchedulerGraphView graph = graph_buffer.graph();
    for (uint32_t scheduler_lane : {1u, 2u}) {
        configure_cached_cluster(storage, scheduler_lane);
        storage.contexts[0].active = 0;
        SchedulerLocalState local{};
        auto &context = storage.contexts[scheduler_lane];
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
        EXPECT_EQ(local.config.self_lane, scheduler_lane);
        EXPECT_EQ(local.config.scheduler_lane, scheduler_lane);
        EXPECT_TRUE(local.is_scheduler());
        EXPECT_EQ(local.config.worker_ids[0], 0u);
        EXPECT_EQ(storage.contexts[0].active, 0u);
        EXPECT_EQ(scheduler_completion_id(&local, 7), 7 * 3 + scheduler_lane);
        const uint64_t ready_offset = local.config.ready_directory_offset;
        const uint64_t payload_offset = local.dispatch_payload_offset(2, 0);
        context.ready_directory_offset = ready_offset + 128;
        context.scheduler_count = 2;
        context.scheduler_index = 1;
        for (auto lane = 0u; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane)
            storage.contexts[lane].dispatch_payload_offset += 128;
        EXPECT_EQ(local.config.ready_directory_offset, ready_offset);
        EXPECT_EQ(local.config.scheduler_count, 1u);
        EXPECT_EQ(local.config.scheduler_index, 0u);
        EXPECT_EQ(local.dispatch_payload_offset(2, 0), payload_offset);
        SchedulerLocalState next_run{};
        ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &next_run));
        EXPECT_EQ(next_run.config.ready_directory_offset, ready_offset + 128);
        EXPECT_EQ(next_run.config.scheduler_count, 2u);
        EXPECT_EQ(next_run.config.scheduler_index, 1u);
        EXPECT_EQ(next_run.dispatch_payload_offset(2, 0), payload_offset + 128);
        EXPECT_EQ(next_run.loop_iter, 0u);
    }
}

TEST(SchedulerLocalConfig, RejectsTruncatedOffsetsAndInconsistentPayloadRoutes) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    const auto graph = graph_buffer.graph();
    configure_cached_cluster(storage, 2);
    auto &context = storage.contexts[2];
    SchedulerLocalState local{};
    EXPECT_EQ(local.worker_id(), UINT64_MAX);
    for (auto field :
         {&SchedulerWorkerContext::worker_contexts_offset, &SchedulerWorkerContext::callable_addresses_offset,
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
    context.active = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.active = 1;
    context.scheduler_ssbuf_reserved1 = 1;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.scheduler_ssbuf_reserved1 = 0;
    context.topology_reserved[2] = 1;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.topology_reserved[2] = 0;
    storage.contexts[0].core_type = 2;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    storage.contexts[0].core_type = static_cast<int32_t>(CoreType::AIC);
    ASSERT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    EXPECT_EQ(local.dispatch_payload_offset(2, 1), original + sizeof(DispatchPayload));
}

TEST(SchedulerLocalState, CompactLayoutAndPendingEndpointBoundaries) {
    EXPECT_EQ(sizeof(SchedulerLocalConfig), 96u);
    EXPECT_EQ(sizeof(SchedulerLocalSlotState), 16u);
    EXPECT_EQ(sizeof(SchedulerLocalState), 336u);
    SchedulerLocalState local{};
    for (auto endpoints : local.owner_pending_endpoints)
        EXPECT_EQ(endpoints, UINT64_MAX);
    const uint64_t endpoints = scheduler_ready_pending_pack(INT32_MAX, INT32_MAX);
    local.owner_pending_endpoints[0] = endpoints;
    EXPECT_EQ(scheduler_ready_pending_head(local.owner_pending_endpoints[0]), INT32_MAX);
    EXPECT_EQ(scheduler_ready_pending_tail(local.owner_pending_endpoints[0]), INT32_MAX);
    EXPECT_EQ(scheduler_ready_pending_pack(SCHEDULER_INBOX_EMPTY, SCHEDULER_INBOX_EMPTY), UINT64_MAX);
}

TEST(SchedulerLocalConfig, RejectsInvalidTopologyBeforeMailboxAccess) {
    FixtureStorage storage(1, 3);
    GraphBuffer graph_buffer(1);
    const SchedulerGraphView graph = graph_buffer.graph();
    configure_cached_cluster(storage, 2);
    auto &context = storage.contexts[2];
    SchedulerLocalState local{};
    context.runtime_worker_count = SCHEDULER_WORKER_CAPACITY + 1;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.runtime_worker_count = 3;
    context.cluster_worker_ids[1] = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.cluster_worker_ids[1] = 1;
    context.scheduler_worker_id = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.scheduler_worker_id = 2;
    context.is_scheduler = 0;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.is_scheduler = 1;
    context.scheduler_count = SCHEDULER_CAPACITY;
    context.scheduler_index = SCHEDULER_CAPACITY - 1;
    EXPECT_TRUE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
    context.scheduler_index = SCHEDULER_CAPACITY;
    EXPECT_FALSE(scheduler_initialize_local_config(storage.scheduler_state->base(), &context, &graph, &local));
}

}  // namespace
