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

#include "hbg_scheduler_test_support.h"

// The startup branch and dispatch-loop branch share this production translation unit.
#include "aicore/aicore_executor.cpp"

namespace {

thread_local volatile uint8_t *startup_reg_base = nullptr;
uint64_t startup_kernel_calls = 0;
uint64_t startup_ssbuf_calls = 0;

void count_kernel_entry(__gm__ int64_t *) { ++startup_kernel_calls; }

struct StartupFixture {
    explicit StartupFixture(const std::array<uint8_t, 2> &masks) :
        storage(2, 6, 3, SchedulerSpmdRequirements{6, 2, 2, 2}) {
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
        std::array<SchedulerAdmissionLocalState, 3> admissions{};
        for (uint32_t lane = 0; lane < 3; ++lane) {
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
                execute_task(payload);
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
        run_resident_scheduler<false>(&storage.contexts[worker], storage.run_control, base(), 0, 0, 0);
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

}  // namespace

volatile uint8_t *sim_get_reg_base() { return startup_reg_base; }
uint32_t sim_get_physical_core_id() { return 3; }
uint64_t platform_ssbuf_hardware_base_address() {
    ++startup_ssbuf_calls;
    return 0;
}
