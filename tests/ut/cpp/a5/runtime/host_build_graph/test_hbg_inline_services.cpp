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

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_dispatch.h"

namespace {

struct InlineFixture {
    explicit InlineFixture(bool ordered = true, bool mix = false, bool protocol = false) :
        storage(3, 3, 4, SchedulerSpmdRequirements{3, 1, 1, mix ? 1u : 0u}, protocol) {
        graph.mixed(0, mix ? 3 : 2);
        graph.executable(2, 1);
        for (uint32_t task = 0; task < 3; ++task) {
            auto &metadata = storage.metadata[task];
            metadata.active_mask = task == 0 && mix ? 3 : 2;
            metadata.logical_block_num = task == 0 ? 2 : 1;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE;
            if (task == 0) metadata.flags |= SCHEDULER_TASK_SPMD | (mix ? SCHEDULER_TASK_MIX : 0);
            if (task == 1) metadata.flags |= SCHEDULER_TASK_INLINE;
            metadata.trace_index_base = task == 0 ? 0 : (mix ? 4 : 2);
            metadata.timing_slot = -1;
            for (uint32_t lane = 0; lane < 3; ++lane)
                metadata.kernel_ids[lane] = task != 1 && (metadata.active_mask & (1U << lane)) ? 1 : UINT16_MAX;
        }
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 61));
        run()->active_worker_count = 3;
        run()->aic_active_worker_count = 1;
        run()->aiv_active_worker_count = 2;
        run()->expected_task_count = 3;
        for (uint32_t worker = 0; worker < 3; ++worker) {
            auto &context = storage.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = 3;
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = lane;
            storage.test_graphs[worker] = graph.graph();
            locals[worker] = storage.local_context(&context);
            locals[worker]->has_mix = mix;
            if (!ordered && protocol) EXPECT_TRUE(scheduler_protocol_bind(base(), locals[worker]));
            if (ordered) {
                EXPECT_TRUE(scheduler_initialize_admission(base(), locals[worker], &admissions[worker]));
                locals[worker]->admission = &admissions[worker];
            }
        }
    }

    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    SchedulerTaskControl &control(uint32_t task) {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset)[task];
    }
    SchedulerTaskTrace &trace(uint32_t index) {
        return scheduler_state_at<SchedulerTaskTrace>(base(), storage.layout.trace_cells_offset)[index];
    }
    void dependency(uint32_t consumer, uint32_t producer) {
        graph.payload(consumer).fanin_count = 1;
        graph.payload(consumer).fanin_data()[0] = static_cast<int32_t>(producer);
        storage.metadata[consumer].flags |= SCHEDULER_TASK_HAS_FANIN;
    }
    SchedulerRouteResult route(uint32_t task, uint32_t worker = 1) {
        return scheduler_route_task(graph.graph(), base(), locals[worker], run(), task, nullptr);
    }
    bool enqueue(uint32_t task, uint32_t worker = 1) {
        if (route(task, worker) != SchedulerRouteResult::READY_TO_ENQUEUE) return false;
        SchedulerReadyBatch batch{};
        const auto &metadata = storage.metadata[task];
        return scheduler_ready_batch_append(base(), locals[worker], task, &batch, nullptr, 4) &&
               scheduler_ready_batch_push(
                   base(), locals[worker], scheduler_task_ready_queue(metadata.flags, metadata.active_mask), &batch,
                   nullptr
               );
    }
    bool fill(uint32_t worker = 1) {
        bool failed = false;
        scheduler_fill_own_slots(
            graph.graph(), base(), locals[worker], run(), cursors[worker].data(), nullptr, 4, 0, &failed, nullptr
        );
        return !failed;
    }
    bool complete(uint32_t worker, uint32_t slot) {
        auto *local = locals[worker];
        bool permit = false;
        if (local->admission != nullptr &&
            (!scheduler_ordered_can_enter_slot(graph.graph(), base(), local, run(), slot, &permit) || !permit))
            return false;
        local->local_ready_mask &= static_cast<uint8_t>(~(1U << slot));
        local->local_completed_generations[slot] = local->slots[slot].generation;
        return scheduler_service_cluster_completions(
            graph.graph(), base(), local, run(), nullptr, nullptr, nullptr, 4, &refilled, nullptr
        );
    }
    void expect_no_dispatch(uint32_t worker) {
        EXPECT_EQ(locals[worker]->local_ready_mask, 0u);
        for (const auto &slot : locals[worker]->slots) {
            EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
            EXPECT_EQ(slot.generation, 0u);
        }
        for (const auto &candidate : admissions[worker].candidates)
            EXPECT_TRUE(scheduler_task_ref_is_null(candidate.ready.task_ref));
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<4> graph{3};
    std::array<SchedulerLocalState *, 3> locals{};
    std::array<SchedulerAdmissionLocalState, 3> admissions{};
    std::array<std::array<uint64_t, SCHEDULER_READY_QUEUE_COUNT>, 3> cursors{};
    uint64_t refilled{0};
};

class HbgInlineDispatch : public ::testing::TestWithParam<bool> {};

TEST_P(HbgInlineDispatch, CompletesWithoutPayloadSlotGenerationOrKernelTrace) {
    InlineFixture fixture(GetParam());
    auto *local = fixture.locals[1];
    auto *payload =
        scheduler_state_at<DispatchPayload>(fixture.base(), local->dispatch_payload_offset(local->config.self_lane, 0));
    std::array<uint8_t, sizeof(DispatchPayload)> previous{};
    std::memset(payload, 0x5a, sizeof(*payload));
    std::memcpy(previous.data(), payload, previous.size());
    fixture.trace(2).ready_transition_cycles = 123456;
    ASSERT_TRUE(fixture.enqueue(1));
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(scheduler_gm_load(fixture.control(1).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(local->pending_completed, 1u);
    fixture.expect_no_dispatch(1);
    EXPECT_EQ(std::memcmp(previous.data(), payload, previous.size()), 0);
    EXPECT_EQ(fixture.trace(2).ready_transition_cycles, 123456u);
    EXPECT_EQ(fixture.trace(2).valid, 0u);
    ASSERT_TRUE(fixture.fill());
    scheduler_flush_completions(fixture.run(), local);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 1u);
}

INSTANTIATE_TEST_SUITE_P(OrdinaryAndOrdered, HbgInlineDispatch, ::testing::Bool());

TEST(HbgInlineService, ProgressesWhileAnUnrelatedMixWaitsForAnotherLane) {
    InlineFixture fixture(true, true);
    ASSERT_TRUE(fixture.enqueue(0, 0));
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_TRUE(fixture.enqueue(1));
    ASSERT_TRUE(fixture.fill(1));
    EXPECT_EQ(scheduler_gm_load(fixture.control(1).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_NE(scheduler_gm_load(fixture.control(0).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.admissions[1].current_task_ref, 0);
    fixture.expect_no_dispatch(1);
}

TEST(HbgInlineService, BootstrapAndOrdinaryReadyDoNotWriteTheNextKernelTrace) {
    for (bool bootstrap : {false, true}) {
        InlineFixture fixture;
        fixture.trace(2).ready_transition_cycles = 987654;
        SchedulerReadyBatch batch{};
        const bool appended =
            bootstrap ?
                scheduler_bootstrap_ready_batch_append(fixture.base(), fixture.locals[1], 1, &batch, nullptr, 4) :
                scheduler_ready_batch_append(fixture.base(), fixture.locals[1], 1, &batch, nullptr, 4);
        ASSERT_TRUE(appended);
        EXPECT_EQ(batch.count, 1u);
        EXPECT_EQ(fixture.trace(2).ready_transition_cycles, 987654u);
    }
}

TEST(HbgInlineService, DirectReadyTransitionDoesNotWriteTheNextKernelTrace) {
    InlineFixture fixture;
    fixture.dependency(1, 0);
    ASSERT_EQ(fixture.route(1), SchedulerRouteResult::WAITING);
    fixture.trace(2).ready_transition_cycles = 987654;
    scheduler_gm_store(fixture.control(0).state, static_cast<int64_t>(SchedulerTaskState::DONE));
    SchedulerReadyClaim ready{};
    ASSERT_TRUE(scheduler_resolve_completion(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, nullptr, nullptr, nullptr, 4, true,
        &ready, fixture.locals[1]->config.self_core_type
    ));
    EXPECT_EQ(ready.task_ref, 1);
    EXPECT_EQ(fixture.trace(2).ready_transition_cycles, 987654u);
}

TEST(HbgInlineService, DirectCompletionServicesDummyWithoutReportingKernelRefill) {
    InlineFixture fixture(false);
    fixture.storage.metadata[0].logical_block_num = 1;
    fixture.storage.metadata[0].flags = SCHEDULER_TASK_EXECUTABLE;
    fixture.dependency(1, 0);
    ASSERT_EQ(fixture.route(1), SchedulerRouteResult::WAITING);
    ASSERT_TRUE(fixture.enqueue(0));
    ASSERT_TRUE(fixture.fill());
    ASSERT_TRUE(fixture.complete(1, 0));
    EXPECT_EQ(fixture.refilled, 0u);
    EXPECT_EQ(fixture.locals[1]->local_ready_mask, 0u);
    EXPECT_EQ(fixture.locals[1]->slots[0].state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(fixture.locals[1]->slots[0].generation, 1u);
    EXPECT_EQ(scheduler_gm_load(fixture.control(1).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.locals[1]->pending_completed, 2u);
    EXPECT_EQ(fixture.trace(2).valid, 0u);
    EXPECT_EQ(fixture.trace(0).refill_start_cycles, 0u);
}

TEST(HbgInlineService, TwoBlocksThenDummyReleaseTheSuccessorExactlyOnce) {
    InlineFixture fixture;
    fixture.dependency(1, 0);
    fixture.dependency(2, 1);
    ASSERT_EQ(fixture.route(1), SchedulerRouteResult::WAITING);
    ASSERT_EQ(fixture.route(2), SchedulerRouteResult::WAITING);
    ASSERT_TRUE(fixture.enqueue(0));
    ASSERT_TRUE(fixture.fill());
    ASSERT_EQ(fixture.locals[1]->slots[0].task_ref, 0);
    ASSERT_EQ(fixture.locals[1]->slots[0].block_idx, 0u);
    ASSERT_EQ(fixture.locals[1]->slots[1].state, SchedulerDispatchSlotState::FREE);
    ASSERT_TRUE(fixture.complete(1, 0));
    EXPECT_EQ(fixture.locals[1]->pending_completed, 0u);
    EXPECT_NE(scheduler_gm_load(fixture.control(1).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.control(2).waiting_producer, 1);
    ASSERT_TRUE(fixture.fill());
    ASSERT_EQ(fixture.locals[1]->slots[0].task_ref, 0);
    ASSERT_EQ(fixture.locals[1]->slots[0].block_idx, 1u);
    ASSERT_TRUE(fixture.complete(1, 0));
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(scheduler_gm_load(fixture.control(1).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_EQ(fixture.locals[1]->pending_completed, 2u);
    ASSERT_TRUE(fixture.fill());
    uint32_t successor_slots = 0;
    for (uint32_t slot = 0; slot < SCHEDULER_PENDING_SLOT_COUNT; ++slot) {
        if (fixture.locals[1]->slots[slot].state != SchedulerDispatchSlotState::READY) continue;
        ASSERT_EQ(fixture.locals[1]->slots[slot].task_ref, 2);
        ++successor_slots;
        ASSERT_TRUE(fixture.complete(1, slot));
    }
    EXPECT_EQ(successor_slots, 1u);
    ASSERT_TRUE(fixture.fill());
    scheduler_flush_completions(fixture.run(), fixture.locals[1]);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->resolved_task_count, UINT64_C(0)), 3u);
    EXPECT_EQ(fixture.trace(0).valid, 1u);
    EXPECT_EQ(fixture.trace(1).valid, 1u);
    EXPECT_EQ(fixture.trace(2).valid, 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgInlineService, AServiceChainIsBoundedByCandidateCreditsAndDoesNotRecurse) {
    InlineFixture fixture;
    for (uint32_t task = 0; task < 3; ++task) {
        auto &metadata = fixture.storage.metadata[task];
        metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_INLINE;
        metadata.logical_block_num = 1;
        for (auto &kernel : metadata.kernel_ids)
            kernel = UINT16_MAX;
    }
    fixture.dependency(1, 0);
    fixture.dependency(2, 1);
    ASSERT_EQ(fixture.route(1), SchedulerRouteResult::WAITING);
    ASSERT_EQ(fixture.route(2), SchedulerRouteResult::WAITING);
    ASSERT_TRUE(fixture.enqueue(0));
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(fixture.locals[1]->pending_completed, 2u);
    EXPECT_NE(scheduler_gm_load(fixture.control(2).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    fixture.expect_no_dispatch(1);
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(fixture.locals[1]->pending_completed, 3u);
    fixture.expect_no_dispatch(1);
}

TEST(HbgInlineService, HostCompletedServiceDoesNotCompleteAgainOnDevice) {
    InlineFixture fixture;
    fixture.storage.metadata[1].flags = 0;
    scheduler_gm_store(fixture.control(1).state, static_cast<int64_t>(SchedulerTaskState::DONE));
    ASSERT_EQ(fixture.route(1), SchedulerRouteResult::COMPLETED);
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(fixture.locals[1]->pending_completed, 0u);
    fixture.expect_no_dispatch(1);
}

TEST(HbgInlineService, RepeatedServiceAndFailedResolutionDoNotAddSuccessfulCompletions) {
    for (bool already_closed : {false, true}) {
        InlineFixture fixture;
        if (already_closed) scheduler_gm_store(fixture.control(1).wake_list_head, SCHEDULER_WAKE_LIST_CLOSED);
        EXPECT_EQ(
            scheduler_complete_inline_task(
                fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 1, 4
            ),
            !already_closed
        );
        EXPECT_FALSE(scheduler_complete_inline_task(
            fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 1, 4
        ));
        EXPECT_EQ(fixture.locals[1]->pending_completed, already_closed ? 0u : 1u);
        EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
        fixture.expect_no_dispatch(1);
    }
}

class HbgInvalidInlineMetadata : public ::testing::TestWithParam<uint32_t> {};

TEST_P(HbgInvalidInlineMetadata, RefusesKernelCooperativeAndPredicateShapesWithoutCompleting) {
    InlineFixture fixture;
    auto &metadata = fixture.storage.metadata[1];
    switch (GetParam()) {
    case 0:
        metadata.flags |= SCHEDULER_TASK_SYNC_START;
        break;
    case 1:
        metadata.flags |= SCHEDULER_TASK_SPMD;
        break;
    case 2:
        metadata.flags |= SCHEDULER_TASK_MIX;
        break;
    case 3:
        metadata.flags |= SCHEDULER_TASK_HAS_PREDICATE;
        break;
    case 4:
        metadata.kernel_ids[1] = 1;
        break;
    case 5:
        metadata.logical_block_num = 2;
        break;
    case 6:
        metadata.logical_block_num = 0;
        break;
    case 7:
        metadata.flags = SCHEDULER_TASK_INLINE;
        break;
    }
    EXPECT_FALSE(
        scheduler_complete_inline_task(fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 1, 4)
    );
    EXPECT_EQ(fixture.locals[1]->pending_completed, 0u);
    EXPECT_NE(scheduler_gm_load(fixture.control(1).state), static_cast<int64_t>(SchedulerTaskState::DONE));
    EXPECT_NE(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
    fixture.expect_no_dispatch(1);
}

INSTANTIATE_TEST_SUITE_P(InvalidServiceWireShapes, HbgInvalidInlineMetadata, ::testing::Range<uint32_t>(0, 8));

TEST(HbgInlineProtocol, RecordsOnlyRealLogicalCompletionWithoutAdmissionOrKernel) {
    for (bool ordered : {false, true}) {
        InlineFixture fixture(ordered, false, true);
        ASSERT_TRUE(fixture.enqueue(1));
        ASSERT_TRUE(fixture.fill());
        const auto *tasks = scheduler_state_at<SchedulerProtocolTaskTrace>(
            fixture.base(), fixture.storage.layout.protocol.tasks_offset
        );
        const auto &done = tasks[1].completion;
        EXPECT_EQ(done.before.sequence, 1u);
        EXPECT_EQ(done.after.sequence, 2u);
        EXPECT_EQ(done.task_ref, 1);
        EXPECT_EQ(done.run_epoch, 61u);
        EXPECT_EQ(done.worker_index, 1u);
        EXPECT_EQ(done.order_key, UINT64_MAX);
        EXPECT_NE(done.flags & SCHEDULER_TASK_INLINE, 0u);
        EXPECT_EQ(tasks[1].admission.before.sequence, 0u);
        EXPECT_EQ(tasks[0].completion.before.sequence, 0u);
        const auto *executions = scheduler_state_at<SchedulerProtocolExecutionTrace>(
            fixture.base(), fixture.storage.layout.protocol.executions_offset
        );
        for (uint64_t index = 0; index < fixture.storage.layout.protocol.trace_count; ++index)
            EXPECT_EQ(executions[index].entry.sequence, 0u);
        fixture.expect_no_dispatch(1);
    }
}

}  // namespace
