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
#include <cstdint>
#include <tuple>

#include "hbg_scheduler_test_support.h"

// The startup branch and dispatch-loop branch share this production translation unit.
#include "aicore/aicore_executor.cpp"

namespace {

thread_local volatile uint8_t *startup_reg_base = nullptr;
uint64_t startup_kernel_calls = 0;
uint64_t startup_ssbuf_calls = 0;
DispatchPayload *identity_test_payload = nullptr;

void count_kernel_entry(__gm__ int64_t *) { ++startup_kernel_calls; }

void protocol_chain_kernel(__gm__ int64_t *) {
    if (++startup_kernel_calls == 2) write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
}

void identity_test_kernel(__gm__ int64_t *) {
    ++startup_kernel_calls;
    identity_test_payload->local_context.block_idx = 77;
    identity_test_payload->local_context.block_num = 99;
    write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
}

struct StartupFixture {
    explicit StartupFixture(const std::array<uint8_t, 2> &masks, bool protocol_profiling = false) :
        storage(2, 6, 3, SchedulerSpmdRequirements{6, 2, 2, 2}, protocol_profiling) {
        startup_reg_base = registers.data();
        graph_view = graph.graph();
        uint32_t trace = 0;
        for (uint32_t task = 0; task < 2; ++task) {
            graph.mixed(task, masks[task]);
            auto &metadata = storage.metadata[task];
            metadata.active_mask = masks[task];
            metadata.logical_block_num = 1;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX;
            metadata.timing_slot = -1;
            metadata.trace_index_base = trace;
            trace += static_cast<uint32_t>(__builtin_popcount(masks[task]));
            for (uint32_t lane = 0; lane < 3; ++lane)
                metadata.kernel_ids[lane] = (masks[task] & (1U << lane)) != 0 ? 1 : UINT16_MAX;
        }
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 59));
        storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&count_kernel_entry);
        storage.ssbuf_region = nullptr;
        auto *run = storage.run_control;
        run->active_worker_count = 6;
        run->expected_task_count = 2;
        run->aic_active_worker_count = 2;
        run->aiv_active_worker_count = 4;
        run->scheduler_timeout_cycles = PLATFORM_PROF_SYS_CNT_FREQ;
        for (uint32_t worker = 0; worker < 6; ++worker) {
            auto &context = storage.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = 6;
            context.physical_core_id = static_cast<int32_t>(worker);
            context.graph_storage_address = graph_view.storage_address;
            context.graph_reserved_address = 0;
            context.graph_task_count = graph_view.task_count;
            context.task_window_last_index = graph_view.task_window_last_index;
            context.scheduler_state_base_address = reinterpret_cast<uint64_t>(base());
            context.run_control_offset = storage.layout.run_control_offset;
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = worker / 3 * 3 + lane;
        }
    }

    void *base() const { return storage.scheduler_state->base(); }
    SchedulerTaskControl *control(uint64_t task) const {
        return scheduler_state_at<SchedulerTaskControl>(base(), storage.layout.task_controls_offset) + task;
    }
    uint64_t &seal(uint64_t task, uint32_t cluster) const {
        const auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
            base(), storage.layout.admission_records_offset + task * storage.layout.admission_record_stride
        );
        return scheduler_state_at<uint64_t>(base(), record->mix_close_seals_offset)[cluster];
    }

    void finish_both_tasks_on_cluster_zero() {
        std::array<SchedulerLocalState, 3> locals{};
        std::array<SchedulerLocalProfilingState, 3> profiles{};
        std::array<SchedulerAdmissionLocalState, 3> admissions{};
        for (uint32_t lane = 0; lane < 3; ++lane) {
            if (storage.layout.protocol_layout_offset != 0) locals[lane].profiling = &profiles[lane];
            ASSERT_TRUE(scheduler_initialize_local_config(base(), &storage.contexts[lane], &graph_view, &locals[lane]));
            ASSERT_TRUE(scheduler_initialize_admission(base(), &locals[lane], &admissions[lane]));
            locals[lane].admission = &admissions[lane];
            for (auto &slot : locals[lane].slots)
                scheduler_initialize_free_slot(&slot);
        }
        auto *run = storage.run_control;
        for (int64_t task = 0; task < 2; ++task) {
            ASSERT_EQ(
                scheduler_route_task(graph_view, base(), &locals[0], run, task, nullptr),
                SchedulerRouteResult::READY_TO_ENQUEUE
            );
            SchedulerReadyBatch batch{};
            ASSERT_TRUE(scheduler_ready_batch_append(base(), &locals[0], task, &batch, nullptr));
            ASSERT_TRUE(scheduler_ready_batch_push(base(), &locals[0], SCHEDULER_MIX_QUEUE, &batch, nullptr));
            SchedulerReadyClaim ready{};
            ASSERT_TRUE(scheduler_ready_pop_from_inbox(
                graph_view, base(), &locals[0], run, SCHEDULER_MIX_QUEUE, 0, &ready.task_ref, nullptr
            ));
            ASSERT_EQ(ready.task_ref, task);
            ASSERT_EQ(
                scheduler_try_claim_admission(graph_view, base(), &locals[0], run, &admissions[0], ready),
                SchedulerAdmissionResult::PROGRESS
            );
            ASSERT_TRUE(scheduler_prepare_admission_record(graph_view, base(), &locals[0], run, &admissions[0]));
            ASSERT_EQ(
                scheduler_try_commit_admission(graph_view, base(), &locals[0], run, &admissions[0]),
                SchedulerAdmissionResult::PROGRESS
            );
            for (uint32_t lane = 0; lane < 3; ++lane)
                ASSERT_TRUE(
                    scheduler_refresh_local_obligation(graph_view, base(), &locals[lane], run, &admissions[lane])
                );
            ASSERT_EQ(scheduler_gm_mix_claim(graph_view, base(), &locals[0], run), SchedulerBlockClaimResult::CLAIMED);
            ASSERT_TRUE(scheduler_gm_mix_publish(graph_view, base(), &locals[0], run));
            bool progress = false;
            for (uint32_t lane : {1u, 2u})
                ASSERT_TRUE(scheduler_gm_mix_observe(graph_view, base(), &locals[lane], run, &progress));
            for (uint32_t lane = 0; lane < 3; ++lane)
                ASSERT_TRUE(scheduler_gm_mix_arrive(graph_view, base(), &locals[lane], run, &progress));
            for (uint32_t lane = 0; lane < 3; ++lane)
                ASSERT_TRUE(scheduler_gm_mix_prepare_lane(graph_view, base(), &locals[lane], run, 0, &progress));
            for (uint32_t lane = 0; lane < 3; ++lane) {
                auto &local = locals[lane];
                uint32_t slot = UINT32_MAX;
                uint64_t publication = 0;
                const bool active = (storage.metadata[task].active_mask & (1U << lane)) != 0;
                if (!scheduler_local_ready_pop(&local, &slot, &publication)) {
                    EXPECT_FALSE(active);
                    continue;
                }
                bool permit = false;
                ASSERT_TRUE(scheduler_ordered_can_enter_slot(graph_view, base(), &local, run, slot, &permit));
                ASSERT_TRUE(permit);
                auto *payload = scheduler_state_at<DispatchPayload>(base(), local.dispatch_payload_offset(lane, slot));
                if (scheduler_protocol_enabled(&local)) scheduler_execute_protocol_task(base(), &local, slot, payload);
                else execute_task(payload);
                local.local_completed_generations[slot] = scheduler_dispatch_generation(publication);
                ASSERT_TRUE(scheduler_service_cluster_completions(
                    graph_view, base(), &local, run, nullptr, nullptr, nullptr, 0, nullptr, nullptr
                ));
            }
            ASSERT_EQ(scheduler_gm_mix_retire(graph_view, base(), &locals[0], run), SchedulerGmMixResult::TASK_LAST);
            ASSERT_TRUE(
                scheduler_complete_ordered_task(graph_view, base(), &locals[0], run, task, 0, nullptr, nullptr, nullptr)
            );
            for (uint32_t lane = 0; lane < 3; ++lane) {
                bool closed = false;
                ASSERT_TRUE(scheduler_gm_mix_close(graph_view, base(), &locals[lane], run, &closed));
                ASSERT_TRUE(closed);
            }
        }
        scheduler_flush_completions(run, &locals[0]);
    }

    void run_late_worker(uint32_t worker = 3) {
        startup_kernel_calls = 0;
        startup_ssbuf_calls = 0;
        scheduler_gm_store(storage.run_control->bootstrap_scan_arrived_count, UINT64_C(5));
        scheduler_gm_store(storage.run_control->bootstrap_arrived_count, UINT64_C(5));
        write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        write_reg(RegId::COND, 0);
        if (storage.layout.protocol_layout_offset != 0)
            run_resident_scheduler<true>(&storage.contexts[worker], storage.run_control, base(), 4, 0, 0);
        else run_resident_scheduler<false>(&storage.contexts[worker], storage.run_control, base(), 0, 0, 0);
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<2> graph{2};
    SchedulerGraphView graph_view{};
    alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> registers{};
};

class HbgGmMixStartupExit : public ::testing::TestWithParam<std::array<uint8_t, 2>> {};

TEST_P(HbgGmMixStartupExit, LateZeroWorkCoordinatorClosesEveryAdmittedMixBeforeExitAck) {
    StartupFixture fixture(GetParam());
    fixture.finish_both_tasks_on_cluster_zero();
    ASSERT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->resolved_task_count, UINT64_C(0)), 2u);
    ASSERT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->scheduler_error, UINT64_C(0)), 0u);
    for (uint64_t task = 0; task < 2; ++task) {
        ASSERT_EQ(scheduler_gm_fetch_add(fixture.control(task)->next_block, UINT64_C(0)), 1u);
        ASSERT_EQ(scheduler_gm_fetch_add(fixture.control(task)->completed_blocks, UINT64_C(0)), 1u);
        ASSERT_EQ(scheduler_gm_load(fixture.control(task)->state), static_cast<int64_t>(SchedulerTaskState::DONE));
        ASSERT_EQ(scheduler_gm_fetch_add(fixture.seal(task, 1), UINT64_C(0)), 0u);
    }
    fixture.run_late_worker();
    EXPECT_EQ(read_reg(RegId::COND), AICORE_EXITED_VALUE);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(1, 1), UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->bootstrap_scan_arrived_count, UINT64_C(0)), 6u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->bootstrap_arrived_count, UINT64_C(0)), 6u);
    EXPECT_EQ(startup_kernel_calls, 0u);
    EXPECT_EQ(startup_ssbuf_calls, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    CompletedMixShapes, HbgGmMixStartupExit,
    ::testing::Values(std::array<uint8_t, 2>{3, 6}, std::array<uint8_t, 2>{6, 6}, std::array<uint8_t, 2>{6, 7})
);

TEST(HbgGmMixStartupExit, FailedRunAcknowledgesExitWithoutFabricatingCloseSeals) {
    StartupFixture fixture({6, 6});
    fixture.finish_both_tasks_on_cluster_zero();
    ASSERT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->resolved_task_count, UINT64_C(0)), 2u);
    scheduler_record_error(
        fixture.storage.run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS,
        &fixture.graph_view, &fixture.storage.contexts[3], SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
    );
    fixture.run_late_worker();
    EXPECT_EQ(read_reg(RegId::COND), AICORE_EXITED_VALUE);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(1, 1), UINT64_C(0)), 0u);
    EXPECT_EQ(startup_kernel_calls, 0u);
    EXPECT_EQ(startup_ssbuf_calls, 0u);
    EXPECT_EQ(
        scheduler_gm_fetch_add(fixture.storage.run_control->scheduler_error, UINT64_C(0)),
        static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
    );
}

TEST(HbgGmMixStartupExit, MissingCoordinatorSealTimeoutIdentifiesTheWaitingTaskAndWorker) {
    StartupFixture fixture({6, 6});
    fixture.finish_both_tasks_on_cluster_zero();
    fixture.storage.run_control->scheduler_timeout_cycles = PLATFORM_PROF_SYS_CNT_FREQ / 1000;
    fixture.run_late_worker(4);
    const auto &run = *fixture.storage.run_control;
    EXPECT_EQ(run.scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_EQ(run.error_site, static_cast<uint64_t>(SchedulerErrorSite::ORDERED_EXIT_TIMEOUT));
    EXPECT_EQ(run.error_task_ref, 0u);
    EXPECT_EQ(run.error_run_epoch, 59u);
    EXPECT_EQ(run.error_order_key, 0u);
    EXPECT_EQ(run.error_block_idx, UINT64_MAX);
    EXPECT_EQ(run.error_group_seq, UINT64_MAX);
    EXPECT_EQ(run.error_worker_index, 4u);
    EXPECT_EQ(run.error_cluster_lane, 1u);
    EXPECT_EQ(run.error_identity_fields, 51u);
    EXPECT_EQ(read_reg(RegId::COND), AICORE_EXITED_VALUE);
    EXPECT_EQ(startup_kernel_calls, 0u);
    EXPECT_EQ(startup_ssbuf_calls, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 0u);
}

TEST(HbgProtocolStartupExit, RealExecutorFinishesLateZeroWorkMembersAfterClosingEveryObligation) {
    StartupFixture fixture({6, 7}, true);
    fixture.finish_both_tasks_on_cluster_zero();
    auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    for (uint32_t worker = 3; worker < 6; ++worker) {
        fixture.run_late_worker(worker);
        ASSERT_EQ(fixture.storage.run_control->scheduler_error, 0u);
        EXPECT_EQ(read_reg(RegId::COND), AICORE_EXITED_VALUE);
        EXPECT_EQ(workers[worker].finished, 1u);
        EXPECT_EQ(workers[worker].error_flags, 0u);
        EXPECT_EQ(workers[worker].next_sequence, 4u);
        EXPECT_EQ(members[worker].observed.sequence, 1u);
        EXPECT_EQ(members[worker].closed.sequence, 2u);
        EXPECT_EQ(members[6 + worker].observed.sequence, 3u);
        EXPECT_EQ(members[6 + worker].closed.sequence, 4u);
        EXPECT_EQ(startup_kernel_calls, 0u);
        EXPECT_EQ(startup_ssbuf_calls, 0u);
    }
}

TEST(HbgProtocolStartupExit, RealExecutorErrorExitDoesNotMarkUnfinishedProtocolSuccessful) {
    StartupFixture fixture({6, 6}, true);
    fixture.finish_both_tasks_on_cluster_zero();
    scheduler_record_error(
        fixture.storage.run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS,
        &fixture.graph_view, &fixture.storage.contexts[3], SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
    );
    fixture.run_late_worker();
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    EXPECT_EQ(workers[3].finished, 2u);
    EXPECT_EQ(workers[3].next_sequence, 0u);
    EXPECT_EQ(fixture.seal(0, 1), 0u);
    EXPECT_EQ(fixture.seal(1, 1), 0u);
    EXPECT_EQ(read_reg(RegId::COND), AICORE_EXITED_VALUE);
}

TEST(HbgProtocolStartupExit, RealExecutorSealTimeoutPreservesObservedButUnclosedMember) {
    StartupFixture fixture({6, 6}, true);
    fixture.finish_both_tasks_on_cluster_zero();
    fixture.storage.run_control->scheduler_timeout_cycles = PLATFORM_PROF_SYS_CNT_FREQ / 1000;
    fixture.run_late_worker(4);
    const auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    const auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    EXPECT_EQ(workers[4].finished, 2u);
    EXPECT_EQ(workers[4].next_sequence, 1u);
    EXPECT_EQ(members[4].observed.sequence, 1u);
    EXPECT_EQ(members[4].closed.sequence, 0u);
    EXPECT_EQ(fixture.storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_EQ(read_reg(RegId::COND), AICORE_EXITED_VALUE);
}

class HbgExecutionIdentityEntry : public ::testing::TestWithParam<std::tuple<uint32_t, uint64_t>> {};

TEST_P(HbgExecutionIdentityEntry, RealExecutorCapturesEntryPayloadBeforeKernelCanChangeIt) {
    const auto [worker, level] = GetParam();
    scheduler_test::FixtureStorage storage(1, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0}, level == 4);
    scheduler_test::BasicGraphBuffer<1> graph(1);
    graph.executable(0, worker == 0 ? 0 : 1);
    auto &metadata = storage.metadata[0];
    metadata.active_mask = worker == 0 ? 1 : 2;
    metadata.kernel_ids[0] = worker == 0 ? 1 : UINT16_MAX;
    metadata.kernel_ids[1] = worker == 0 ? UINT16_MAX : 1;
    metadata.timing_slot = -1;
    auto *base = storage.scheduler_state->base();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, 59));
    storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&identity_test_kernel);
    storage.test_graphs[worker] = graph.graph();
    for (uint32_t lane = 0; lane < 3; ++lane)
        storage.contexts[worker].cluster_worker_ids[lane] = lane;
    auto *local = storage.local_context(&storage.contexts[worker]);
    ASSERT_TRUE(scheduler_protocol_bind(base, local));
    local->config.scheduler_count = 3;
    auto *run = storage.run_control;
    run->expected_task_count = 1;
    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(base, local, 0, &batch, nullptr));
    ASSERT_TRUE(scheduler_ready_batch_push(base, local, worker == 0 ? 0 : 1, &batch, nullptr));
    std::array<uint8_t, 64> registers{};
    startup_reg_base = registers.data();
    startup_kernel_calls = 0;
    identity_test_payload =
        scheduler_state_at<DispatchPayload>(base, local->dispatch_payload_offset(local->config.self_lane, 0));
    write_reg(RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    SchedulerWorkerStats stats{};
    ASSERT_TRUE(run_ready_dispatch_loop(graph.graph(), base, local, run, &stats, level));
    ASSERT_EQ(startup_kernel_calls, 1u);
    ASSERT_EQ(identity_test_payload->local_context.block_idx, 77);
    uint64_t refilled = 0;
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), base, local, run, nullptr, nullptr, nullptr, level, &refilled, nullptr
    ));
    const auto *trace = scheduler_state_at<SchedulerTaskTrace>(base, storage.layout.trace_cells_offset);
    if (level == 0) {
        EXPECT_EQ(trace->execution_identity_flags, 0u);
    } else {
        EXPECT_EQ(trace->valid, 1u);
        EXPECT_EQ(trace->execution_run_epoch, 59u);
        EXPECT_EQ(trace->execution_order_key, UINT64_MAX);
        EXPECT_EQ(trace->execution_group_seq, 0u);
        EXPECT_EQ(trace->execution_block_idx, 0u);
        EXPECT_EQ(trace->execution_block_num, 1u);
        EXPECT_EQ(trace->execution_subtask_slot, worker == 0 ? 0u : 1u);
        EXPECT_EQ(trace->execution_pending_slot, 0u);
        EXPECT_EQ(trace->execution_generation, 1u);
        EXPECT_EQ(trace->execution_cluster_index, 0u);
        EXPECT_EQ(trace->execution_cluster_lane, worker);
        EXPECT_EQ(trace->execution_identity_flags, SCHEDULER_TRACE_IDENTITY_CAPTURED | SCHEDULER_TRACE_EXECUTED);
    }
    startup_reg_base = nullptr;
    identity_test_payload = nullptr;
    if (level == 4) {
        const auto *tasks = scheduler_state_at<SchedulerProtocolTaskTrace>(base, storage.layout.protocol.tasks_offset);
        EXPECT_NE(tasks[0].completion.before.sequence, 0u);
        EXPECT_GT(tasks[0].completion.after.sequence, tasks[0].completion.before.sequence);
        EXPECT_EQ(tasks[0].completion.worker_index, worker);
        EXPECT_EQ(tasks[0].completion.order_key, UINT64_MAX);
        EXPECT_EQ(tasks[0].completion.run_epoch, 59u);
        const auto *event =
            scheduler_state_at<SchedulerProtocolExecutionTrace>(base, storage.layout.protocol.executions_offset);
        EXPECT_EQ(event->publication_count, 1u);
        EXPECT_LE(event->publication_before_cycles, event->publication_after_cycles);
        EXPECT_LE(event->publication_after_cycles, event->permit.cycles);
        EXPECT_NE(event->permit.sequence, 0u);
        EXPECT_GT(event->entry.sequence, event->permit.sequence);
        EXPECT_GT(event->returned.sequence, event->entry.sequence);
        EXPECT_GT(event->visible.sequence, event->returned.sequence);
        EXPECT_GT(event->completion_consumed.sequence, event->visible.sequence);
        EXPECT_GT(tasks[0].completion.before.sequence, event->completion_consumed.sequence);
        EXPECT_GT(event->trace_published.sequence, event->completion_consumed.sequence);
        EXPECT_LE(event->returned.cycles, event->visible.cycles);
        EXPECT_EQ(event->run_epoch, 59u);
        EXPECT_EQ(event->worker_index, worker);
        EXPECT_EQ(event->block_idx, 0u);
        EXPECT_EQ(event->logical_block_num, 1u);
        EXPECT_EQ(event->pending_slot, 0u);
        EXPECT_EQ(event->generation, 1u);
        EXPECT_EQ(event->cluster_lane, worker);
        EXPECT_EQ(
            event->flags, SCHEDULER_PROTOCOL_ENTRY | SCHEDULER_PROTOCOL_EXECUTED | SCHEDULER_PROTOCOL_PERMIT_NO_LOG |
                              SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED
        );
    }
}

INSTANTIATE_TEST_SUITE_P(
    LanesAndProfilingLevels, HbgExecutionIdentityEntry,
    ::testing::Combine(::testing::Values(0u, 1u, 2u), ::testing::Values<uint64_t>(0, 1, 2, 4))
);

TEST(HbgProtocolExecution, DirectRefillKeepsTheCompletedGenerationSeparateFromItsSuccessor) {
    scheduler_test::FixtureStorage storage(2, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0}, true);
    scheduler_test::BasicGraphBuffer<2> graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    auto *base = storage.scheduler_state->base();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, 73));
    storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&protocol_chain_kernel);
    storage.test_graphs[0] = graph.graph();
    for (uint32_t lane = 0; lane < 3; ++lane)
        storage.contexts[0].cluster_worker_ids[lane] = lane;
    auto *local = storage.local_context(&storage.contexts[0]);
    local->config.scheduler_count = 3;
    ASSERT_TRUE(scheduler_protocol_bind(base, local));
    auto *run = storage.run_control;
    run->expected_task_count = 2;
    ASSERT_EQ(scheduler_route_task(graph.graph(), base, local, run, 1, nullptr), SchedulerRouteResult::WAITING);
    SchedulerReadyBatch batch{};
    ASSERT_TRUE(scheduler_ready_batch_append(base, local, 0, &batch, nullptr));
    ASSERT_TRUE(scheduler_ready_batch_push(base, local, 0, &batch, nullptr));
    std::array<uint8_t, 64> registers{};
    startup_reg_base = registers.data();
    startup_kernel_calls = 0;
    write_reg(RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    SchedulerWorkerStats stats{};
    ASSERT_TRUE(run_ready_dispatch_loop(graph.graph(), base, local, run, &stats, 4));
    ASSERT_EQ(startup_kernel_calls, 2u);
    ASSERT_TRUE(scheduler_service_cluster_completions(
        graph.graph(), base, local, run, nullptr, nullptr, nullptr, 4, nullptr, nullptr
    ));
    const auto *events =
        scheduler_state_at<SchedulerProtocolExecutionTrace>(base, storage.layout.protocol.executions_offset);
    EXPECT_EQ(events[0].task_ref, 0);
    EXPECT_EQ(events[1].task_ref, 1);
    EXPECT_EQ(events[0].pending_slot, events[1].pending_slot);
    EXPECT_GT(events[1].generation, events[0].generation);
    EXPECT_NE(events[0].trace_published.sequence, 0u);
    EXPECT_GT(events[1].entry.sequence, events[0].trace_published.sequence);
    EXPECT_GT(events[1].trace_published.sequence, events[1].completion_consumed.sequence);
    EXPECT_EQ(events[1].ready_source, static_cast<uint32_t>(SchedulerReadySource::DIRECT_RESOLVE));
    EXPECT_EQ(events[1].publication_mode, static_cast<uint32_t>(SchedulerPublicationMode::REFILL));
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(base, storage.layout.protocol.workers_offset);
    EXPECT_EQ(worker->error_flags, 0u);
    startup_reg_base = nullptr;
}

}  // namespace

volatile uint8_t *sim_get_reg_base() { return startup_reg_base; }
uint32_t sim_get_physical_core_id() { return 3; }
uint64_t platform_ssbuf_hardware_base_address() {
    ++startup_ssbuf_calls;
    return 0;
}
