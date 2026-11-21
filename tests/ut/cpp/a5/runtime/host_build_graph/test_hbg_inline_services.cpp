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
#include <vector>

#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_dispatch.h"
#include "scheduler/scheduler_graph_control.h"
#include "test_hbg_owned_execution.h"

namespace {

struct InlineFixture {
    explicit InlineFixture(
        bool ordered = true, bool mix = false, bool protocol = false, LaneMask mix_mask = LaneMask(true, true, false),
        bool host_completed_service = false, uint16_t producer_blocks = 0, uint32_t workers = 3,
        bool service_chain = false
    ) :
        storage(
            3, workers, 4,
            SchedulerSpmdRequirements{
                workers, workers / 3, ordered && mix ? 1u : 0u, mix ? 1u : 0u, mix && ordered ? 1u : 0u
            },
            protocol
        ),
        locals(workers),
        signals((workers + 2) / 3) {
        for (uint32_t task = 0; task < 3; ++task) {
            auto &metadata = storage.metadata[task];
            metadata.active_mask = LaneMask::aiv0();
            metadata.logical_block_num = 1;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_INLINE;
            metadata.trace_index_base = task == 0 ? 0 : 2;
            metadata.timing_slot = -1;
            if (task == 0 && (mix || producer_blocks != 0)) {
                metadata.active_mask = mix ? mix_mask : LaneMask::aiv0();
                metadata.logical_block_num = mix ? 2 : producer_blocks;
                metadata.flags = SCHEDULER_TASK_EXECUTABLE;
                if (metadata.logical_block_num > 1) metadata.flags |= SCHEDULER_TASK_SPMD;
                if (mix) metadata.flags |= SCHEDULER_TASK_MIX;
                if (mix && ordered) metadata.flags |= SCHEDULER_TASK_SYNC_START;
                graph.mixed(task, metadata.active_mask);
            }
            if (task == 2 && producer_blocks > 1) {
                metadata.flags = SCHEDULER_TASK_EXECUTABLE;
                graph.executable(task, 1, {1});
            }
            if (!service_chain &&
                    ((task == 0 && !mix && producer_blocks == 0) || (task == 2 && producer_blocks <= 1)) ||
                (task == 1 && host_completed_service)) {
                metadata.flags = 0;
                metadata.logical_block_num = 0;
                metadata.active_mask = LaneMask::none();
            }
            for (uint32_t lane = 0; lane < 3; ++lane)
                metadata.kernel_ids[lane] = scheduler_task_is_executable(metadata.flags) &&
                                                    !scheduler_task_is_inline(metadata.flags) &&
                                                    metadata.active_mask.has(static_cast<SubtaskSlot>(lane)) ?
                                                1 :
                                                UINT16_MAX;
        }
        storage.prepare_catalog();
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 61));
        EXPECT_TRUE(storage.prepare_calls(graph.graph()));
        run()->active_worker_count = workers;

        for (uint32_t worker = 0; worker < workers; ++worker) {
            auto &context = storage.contexts[worker];
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = worker / 3 * 3 + lane;
            storage.test_graphs[worker] = graph.graph();
        }
        for (auto &signal : signals)
            scheduler_cluster_signal_clear(&signal, 61);
        for (uint32_t worker = 0; worker < workers; ++worker) {
            auto &context = storage.contexts[worker];
            locals[worker] = storage.local_context(&context);
            locals[worker]->cluster_signal = &signals[worker / 3];

            if (protocol) EXPECT_TRUE(scheduler_protocol_bind(base(), locals[worker]));
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
    }
    bool service(uint32_t worker = 1) {
        bool progress = false;
        return scheduler_progress_graph_controls(graph.graph(), base(), locals[worker], run(), 4, &progress);
    }
    bool fill(uint32_t worker = 1) {
        if (!service(worker)) return false;
        bool failed = false;
        scheduler_find_local_execution(graph.graph(), base(), locals[worker], run(), 4, &failed);
        return !failed;
    }
    bool service_until_done(uint32_t task, uint32_t worker = 1) {
        for (uint32_t probe = 0; probe < 2 * graph.graph().task_count; ++probe) {
            if (scheduler_completion_done(scheduler_gm_query(control(task).completion_state))) return true;
            const uint64_t before = storage.completed_gm_records();
            if (!fill(worker)) return false;
            EXPECT_LE(storage.completed_gm_records() - before, SCHEDULER_CONTROL_NODE_BUDGET);
            expect_no_dispatch(worker);
        }
        return scheduler_completion_done(scheduler_gm_query(control(task).completion_state));
    }
    bool complete(uint32_t worker, uint32_t slot) {
        auto *local = locals[worker];
        bool permit = false;
        if (!scheduler_check_entry_permit(graph.graph(), base(), local, run(), slot, &permit) || !permit) return false;
        if (!hbg_test_begin_execution(local, slot)) return false;
        return scheduler_complete_lane_execution(graph.graph(), base(), local, run(), slot, 4);
    }
    void expect_no_dispatch(uint32_t worker) {
        EXPECT_EQ(hbg_test_eligible_mask(locals[worker]), 0u);
        for (uint32_t index = 0; index < SCHEDULER_LOCAL_EXECUTION_COUNT; ++index) {
            const auto &slot = locals[worker]->execution_at(index);
            EXPECT_EQ(slot.state, SchedulerDispatchSlotState::FREE);
            EXPECT_EQ(slot.task_ref, SCHEDULER_TASK_REF_INVALID);
        }
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<4> graph{3};
    std::vector<SchedulerLocalState *> locals;
    std::vector<SchedulerClusterSignal> signals;
};

class HbgInlineDispatch : public ::testing::TestWithParam<bool> {};

TEST_P(HbgInlineDispatch, CompletesWithoutPayloadSlotGenerationOrKernelTrace) {
    InlineFixture fixture(GetParam());
    auto *local = fixture.locals[1];
    EXPECT_EQ(fixture.control(1).completion_state, scheduler_completion_initial_state(1));
    EXPECT_FALSE(scheduler_completion_done(fixture.control(1).completion_state));
    auto *calls = scheduler_prepared_call_at(fixture.base(), local->config.prepared, 0);
    std::vector<uint8_t> previous(local->config.prepared->execution_count * sizeof(SchedulerPreparedCall));
    if (!previous.empty()) std::memcpy(previous.data(), calls, previous.size());
    fixture.trace(2).kernel_start_cycles = 123456;
    ASSERT_TRUE(fixture.service_until_done(1));
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(1).completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(fixture.storage.completed_gm_records(), 1u);
    EXPECT_EQ(scheduler_cached_task_proof(local, 1), SchedulerTaskProof::DONE);
    fixture.expect_no_dispatch(1);
    if (!previous.empty()) EXPECT_EQ(std::memcmp(previous.data(), calls, previous.size()), 0);
    EXPECT_EQ(fixture.trace(2).kernel_start_cycles, 123456u);
    EXPECT_EQ(fixture.trace(2).valid, 0u);
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(fixture.storage.completed_gm_records(), 1u);
}

INSTANTIATE_TEST_SUITE_P(OrdinaryAndOrdered, HbgInlineDispatch, ::testing::Bool());

TEST(HbgInlineService, CancellationIsObservedBeforePublishingTaskDone) {
    InlineFixture fixture;
    scheduler_publish_error(fixture.run(), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_FALSE(fixture.service());
    EXPECT_EQ(
        scheduler_completion_task_state(fixture.control(1).completion_state),
        static_cast<int64_t>(SchedulerTaskState::BLOCKED)
    );
    EXPECT_EQ(fixture.storage.completed_gm_records(), 0u);
}

TEST(HbgInlineService, ProgressesWhileAnUnrelatedMixWaitsForAnotherLane) {
    InlineFixture fixture(true, true, false, LaneMask::all(), false, 0, 6);
    ASSERT_TRUE(fixture.fill(0));
    ASSERT_TRUE(fixture.fill(1));
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(1).completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_NE(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(0).completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    const auto &group = fixture.locals[1]->mix_groups[0];
    EXPECT_EQ(group.task_ref, 0);
    EXPECT_EQ(group.phase, SchedulerGmMixPhase::RUNNING);
    EXPECT_EQ(hbg_test_eligible_mask(fixture.locals[1]), UINT32_C(1) << SCHEDULER_CLUSTER_MIX_EXECUTION);
    EXPECT_EQ(fixture.locals[1]->execution_at(SCHEDULER_CLUSTER_MIX_EXECUTION).task_ref, 0);
    EXPECT_EQ(scheduler_execution_token_count(fixture.locals[1]), 0u);
}

TEST(HbgInlineService, FaninCompletionDoesNotWriteTheNextKernelTrace) {
    InlineFixture fixture(false, false, false, LaneMask::aiv0(), false, 1);
    fixture.dependency(1, 0);
    fixture.trace(2).kernel_start_cycles = 987654;
    ASSERT_TRUE(fixture.service());
    EXPECT_FALSE(scheduler_completion_done(fixture.control(1).completion_state));
    scheduler_gm_store(fixture.control(0).completion_state, scheduler_completion_completed_state(1));
    ASSERT_TRUE(fixture.service());
    EXPECT_TRUE(scheduler_completion_done(fixture.control(1).completion_state));
    EXPECT_EQ(fixture.trace(2).kernel_start_cycles, 987654u);
    fixture.expect_no_dispatch(1);
}

TEST(HbgInlineService, ConsumerPollingServicesDummyWithoutReportingKernelRefill) {
    InlineFixture fixture(false, false, false, LaneMask::aiv0(), false, 1);
    fixture.dependency(1, 0);
    ASSERT_TRUE(fixture.fill());
    ASSERT_TRUE(fixture.complete(1, 0));
    ASSERT_TRUE(fixture.service());
    EXPECT_EQ(hbg_test_eligible_mask(fixture.locals[1]), 0u);
    EXPECT_EQ(fixture.locals[1]->execution_at(0).state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(fixture.locals[1]->execution_at(0).task_ref, SCHEDULER_TASK_REF_INVALID);
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(1).completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(fixture.storage.completed_gm_records(), 2u);
    EXPECT_EQ(fixture.trace(2).valid, 0u);
}

TEST(HbgInlineService, TwoBlocksThenDummyReleaseTheSuccessorExactlyOnce) {
    InlineFixture fixture(false, false, false, LaneMask::aiv0(), false, 2);
    fixture.dependency(1, 0);
    fixture.dependency(2, 1);
    ASSERT_TRUE(fixture.fill());
    ASSERT_EQ(fixture.locals[1]->execution_at(0).task_ref, 0);
    const auto first = fixture.locals[1]->execution_at(0).block_idx;
    ASSERT_LT(first, 2u);
    ASSERT_EQ(scheduler_execution_token_count(fixture.locals[1]), 1u);
    ASSERT_TRUE(fixture.complete(1, 0));
    EXPECT_EQ(fixture.storage.completed_gm_records(), 0u);
    EXPECT_NE(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(1).completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    ASSERT_TRUE(fixture.fill());
    ASSERT_EQ(fixture.locals[1]->execution_at(0).task_ref, 0);
    const auto second = fixture.locals[1]->execution_at(0).block_idx;
    ASSERT_LT(second, 2u);
    ASSERT_NE(first, second);
    ASSERT_TRUE(fixture.complete(1, 0));
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(
        scheduler_completion_task_state(scheduler_gm_query(fixture.control(1).completion_state)),
        static_cast<int64_t>(SchedulerTaskState::DONE)
    );
    EXPECT_EQ(fixture.storage.completed_gm_records(), 2u);
    ASSERT_TRUE(fixture.fill());
    uint32_t successor_slots = 0;
    for (uint32_t slot = 0; slot < SCHEDULER_LOCAL_EXECUTION_COUNT; ++slot) {
        if (fixture.locals[1]->execution_at(slot).state != SchedulerDispatchSlotState::READY) continue;
        ASSERT_EQ(fixture.locals[1]->execution_at(slot).task_ref, 2);
        ++successor_slots;
        ASSERT_TRUE(fixture.complete(1, slot));
    }
    EXPECT_EQ(successor_slots, 1u);
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(fixture.storage.completed_gm_records(), 3u);
    EXPECT_EQ(fixture.trace(0).valid, 1u);
    EXPECT_EQ(fixture.trace(1).valid, 1u);
    EXPECT_EQ(fixture.trace(2).valid, 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.run()->scheduler_error, UINT64_C(0)), 0u);
}

TEST(HbgInlineService, AServiceChainCompletesWithinTheBoundedControlSlice) {
    InlineFixture fixture(false, false, false, LaneMask::aiv0(), false, 0, 3, true);
    fixture.dependency(1, 0);
    fixture.dependency(2, 1);
    ASSERT_TRUE(fixture.fill());
    EXPECT_GE(fixture.storage.completed_gm_records(), 1u);
    EXPECT_LE(fixture.storage.completed_gm_records(), SCHEDULER_CONTROL_NODE_BUDGET);
    fixture.expect_no_dispatch(1);
    ASSERT_TRUE(fixture.service_until_done(2));
    EXPECT_EQ(fixture.storage.completed_gm_records(), 3u);
    fixture.expect_no_dispatch(1);
}

TEST(HbgInlineService, HostCompletedServiceDoesNotCompleteAgainOnDevice) {
    InlineFixture fixture(true, false, false, LaneMask(true, true, false), true);
    const auto &pool =
        *scheduler_state_at<SchedulerTaskCatalogHeader>(fixture.base(), fixture.storage.task_catalog_offset);
    EXPECT_EQ(pool.entry_count, 0u);
    EXPECT_EQ(pool.control_count, 0u);
    scheduler_gm_store(
        fixture.control(1).completion_state,
        scheduler_completion_completed_state(scheduler_completion_total(fixture.control(1).completion_state))
    );
    const auto before = fixture.storage.completed_gm_records();
    ASSERT_TRUE(fixture.fill());
    EXPECT_EQ(fixture.storage.completed_gm_records(), before);
    fixture.expect_no_dispatch(1);
}

TEST(HbgInlineService, RepeatedServiceAndInvalidCompletionDoNotAddSuccessfulCompletions) {
    for (bool invalid_state : {false, true}) {
        InlineFixture fixture;
        if (invalid_state) fixture.control(1).completion_state = 0;
        EXPECT_EQ(fixture.service(), !invalid_state);
        EXPECT_EQ(fixture.service(), !invalid_state);
        EXPECT_EQ(fixture.storage.completed_gm_records(), invalid_state ? 0u : 1u);
        EXPECT_EQ(scheduler_read_error(fixture.run()) != 0, invalid_state);
        fixture.expect_no_dispatch(1);
    }
}

class HbgInvalidInlineMetadata : public ::testing::TestWithParam<uint32_t> {};

TEST_P(HbgInvalidInlineMetadata, HostRejectsKernelCooperativeAndPredicateShapesWithoutPublication) {
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
    SchedulerTaskCatalogPlan plan{};
    plan.total_size = 123;
    const std::vector<SchedulerTaskMetadata> metadata_rows(fixture.storage.metadata, fixture.storage.metadata + 3);
    EXPECT_FALSE(scheduler_plan_task_catalog(metadata_rows, 1, fixture.storage.catalog_begin, &plan));
    EXPECT_EQ(plan.total_size, 123u);
    EXPECT_EQ(fixture.storage.completed_gm_records(), 0u);
    EXPECT_FALSE(scheduler_completion_done(fixture.control(1).completion_state));
    fixture.expect_no_dispatch(1);
}

INSTANTIATE_TEST_SUITE_P(InvalidServiceWireShapes, HbgInvalidInlineMetadata, ::testing::Range<uint32_t>(0, 8));

TEST(HbgInlineProtocol, RecordsTheLogicalCompletionAndRankWithoutKernelEntry) {
    for (bool ordered : {false, true}) {
        InlineFixture fixture(ordered, false, true);
        ASSERT_TRUE(fixture.service_until_done(1));
        const auto *tasks = scheduler_state_at<SchedulerProtocolTaskTrace>(
            fixture.base(), fixture.storage.layout.protocol.tasks_offset
        );
        const auto &done = tasks[1].completion;
        EXPECT_EQ(done.before.sequence, 1u);
        EXPECT_EQ(done.after.sequence, 2u);
        EXPECT_EQ(done.task_ref, 1);
        EXPECT_EQ(done.run_epoch, 61u);
        EXPECT_EQ(done.worker_index, 1u);
        EXPECT_EQ(done.order_key, scheduler_order_for(fixture.base(), fixture.locals[1], 1).rank);
        EXPECT_NE(done.flags & SCHEDULER_TASK_INLINE, 0u);
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
