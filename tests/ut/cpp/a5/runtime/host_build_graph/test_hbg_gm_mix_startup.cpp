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

#include <array>
#include <cstdint>
#include <functional>
#include <memory>
#include <tuple>
#include <utility>

#include "hbg_scheduler_test_support.h"

#include "aicore/resident_scheduler_impl.h"

namespace {

thread_local volatile uint8_t *startup_reg_base = nullptr;
thread_local uint8_t *startup_ssbuf_base = nullptr;
thread_local std::function<void()> startup_register_observer;
thread_local bool startup_observing_register = false;
uint64_t startup_kernel_calls = 0;
SchedulerExecutionBinding *identity_test_payload = nullptr;

void count_kernel_entry(__gm__ int64_t *) { ++startup_kernel_calls; }

void idle_ready_kernel(__gm__ int64_t *) {
    ++startup_kernel_calls;
    write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
}

class ScopedStartupRegisterObserver {
public:
    explicit ScopedStartupRegisterObserver(std::function<void()> observer) {
        startup_register_observer = std::move(observer);
    }
    ~ScopedStartupRegisterObserver() {
        startup_register_observer = {};
        startup_observing_register = false;
        startup_reg_base = nullptr;
    }
};

void protocol_chain_kernel(__gm__ int64_t *) { ++startup_kernel_calls; }

void identity_test_kernel(__gm__ int64_t *) {
    ++startup_kernel_calls;
    identity_test_payload->local_context.block_idx = 77;
    identity_test_payload->local_context.block_num = 99;
}

struct StartupFixture {
    explicit StartupFixture(const std::array<LaneMask, 2> &masks, bool protocol_profiling = false) :
        storage(2, 9, 6, SchedulerSpmdRequirements{9, 3, 2, 2, 2}, protocol_profiling) {
        startup_reg_base = registers.data();
        startup_ssbuf_base = reinterpret_cast<uint8_t *>(cluster_storage[0].data());
        graph_view = graph.graph();
        uint32_t trace = 0;
        for (uint32_t task = 0; task < 2; ++task) {
            graph.mixed(task, masks[task]);
            auto &metadata = storage.metadata[task];
            metadata.active_mask = masks[task];
            metadata.logical_block_num = 2;
            metadata.flags =
                SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_MIX | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START;
            metadata.timing_slot = -1;
            metadata.trace_index_base = trace;
            trace += 2 * static_cast<uint32_t>(masks[task].count());
            for (uint32_t lane = 0; lane < 3; ++lane)
                metadata.kernel_ids[lane] = masks[task].has(static_cast<SubtaskSlot>(lane)) ? 1 : UINT16_MAX;
        }
        storage.prepare_catalog();
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 59));
        storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&count_kernel_entry);
        EXPECT_TRUE(storage.prepare_calls(graph_view));

        auto *run = storage.run_control;
        run->active_worker_count = 9;
        run->expected_task_count = 2;
        run->aic_active_worker_count = 3;
        run->aiv_active_worker_count = 6;
        run->scheduler_timeout_cycles = PLATFORM_PROF_SYS_CNT_FREQ;
        for (uint32_t worker = 0; worker < 9; ++worker) {
            auto &context = storage.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = 9;
            context.physical_core_id = static_cast<int32_t>(worker);
            context.graph_storage_address = graph_view.storage_address;
            context.graph_task_count = graph_view.task_count;
            context.original_task_count = graph_view.original_task_count;
            context.graph_view_version = graph_view.view_version;
            context.graph_view_bytes = graph_view.view_bytes;
            context.scheduler_state_bytes = storage.layout.total_size;
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
    volatile uint64_t &seal(uint64_t task, uint32_t cluster) const {
        const auto *record = scheduler_state_at<SchedulerAdmissionRecord>(
            base(), storage.layout.admission_records_offset + task * storage.layout.admission_record_stride
        );
        return scheduler_state_at<SchedulerMixCloseSeal>(base(), record->mix_close_seals_offset)[cluster].value;
    }

    void finish_both_tasks_on_early_clusters() {
        // SchedulerLocalState's default constructor is explicit, which a
        // braced initializer here would reach by copy-initialization. These
        // elements are default-initialized instead; every member of that
        // struct carries its own initializer, so the state is the same.
        std::array<SchedulerLocalState, 6> locals;
        std::array<SchedulerLocalProfilingState, 6> profiles{};
        std::array<SchedulerAdmissionLocalState, 6> admissions{};
        // Clusters 0 and 2 execute the two blocks; cluster 1 arrives only at EXIT.
        for (uint32_t lane = 0; lane < 6; ++lane) {
            const uint32_t worker = lane < 3 ? lane : lane + 3;
            if (storage.layout.protocol_layout_offset != 0) locals[lane].profiling = &profiles[lane];
            ASSERT_TRUE(
                scheduler_initialize_local_config(base(), &storage.contexts[worker], &graph_view, &locals[lane])
            );
            ASSERT_TRUE(scheduler_initialize_admission(base(), &locals[lane], &admissions[lane]));
            locals[lane].admission = &admissions[lane];
            for (auto &slot : locals[lane].slots)
                scheduler_initialize_free_slot(&slot);
        }
        auto *run = storage.run_control;
        for (int64_t task = 0; task < 2; ++task)
            ASSERT_TRUE(
                scheduler_bootstrap_prepare_admission(graph_view, base(), &locals[0], run, &admissions[0], task)
            );
        run->bootstrap_scan_complete = 1;
        for (int64_t task = 0; task < 2; ++task) {
            const LaneMask resources = scheduler_mix_resource_mask(storage.metadata[task].active_mask);
            ASSERT_EQ(
                scheduler_stage_admission_candidate(graph_view, base(), &locals[0], run, &admissions[0], task),
                SchedulerAdmissionResult::PROGRESS
            );

            ASSERT_EQ(
                scheduler_try_commit_admission(graph_view, base(), &locals[0], run, &admissions[0]),
                SchedulerAdmissionResult::PROGRESS
            );
            for (uint32_t cluster = 0; cluster < 2; ++cluster) {
                const uint32_t first = cluster * 3;
                for (uint32_t lane = 0; lane < 3; ++lane)
                    ASSERT_TRUE(scheduler_refresh_local_obligation(
                        graph_view, base(), &locals[first + lane], run, &admissions[first + lane]
                    ));
                ASSERT_EQ(
                    scheduler_gm_mix_claim(graph_view, base(), &locals[first], run), SchedulerBlockClaimResult::CLAIMED
                );
                ASSERT_TRUE(scheduler_gm_mix_publish(graph_view, base(), &locals[first], run));
                bool progress = false;
                for (uint32_t lane : {1u, 2u})
                    if (resources.has(static_cast<SubtaskSlot>(lane)))
                        ASSERT_TRUE(
                            scheduler_gm_mix_observe(graph_view, base(), &locals[first + lane], run, &progress)
                        );
                for (uint32_t lane = 0; lane < 3; ++lane)
                    if (resources.has(static_cast<SubtaskSlot>(lane)))
                        ASSERT_TRUE(scheduler_gm_mix_arrive(graph_view, base(), &locals[first + lane], run, &progress));
                for (uint32_t lane = 0; lane < 3; ++lane)
                    if (resources.has(static_cast<SubtaskSlot>(lane)))
                        ASSERT_TRUE(
                            scheduler_gm_mix_prepare_lane(graph_view, base(), &locals[first + lane], run, 0, &progress)
                        );
                for (uint32_t lane = 0; lane < 3; ++lane) {
                    auto &local = locals[first + lane];
                    uint32_t slot = UINT32_MAX;
                    const bool active = storage.metadata[task].active_mask.has(static_cast<SubtaskSlot>(lane));
                    if (!scheduler_local_ready_peek(&local, &slot)) {
                        EXPECT_FALSE(active);
                        continue;
                    }
                    bool permit = false;
                    ASSERT_TRUE(scheduler_ordered_can_enter_slot(graph_view, base(), &local, run, slot, &permit));
                    ASSERT_TRUE(permit);
                    ASSERT_TRUE(scheduler_local_ready_execute(&local, slot));
                    const auto index = local.slots[slot].prepared.execution_index;
                    auto *call = scheduler_prepared_call_at(base(), local.config.prepared, index);
                    auto *binding = scheduler_execution_binding_at(base(), local.config.prepared, index);
                    if (scheduler_protocol_enabled(&local))
                        scheduler_execute_protocol_task(
                            base(), &local, slot, call->function_bin_addr, call->args, binding->local_context
                        );
                    else execute_task(call->function_bin_addr, call->args);
                    ASSERT_TRUE(scheduler_complete_own_slot(graph_view, base(), &local, run, slot, 0));
                }
                ASSERT_EQ(
                    scheduler_gm_mix_retire(graph_view, base(), &locals[first], run), SchedulerGmMixResult::PROGRESS
                );
                for (uint32_t lane = 0; lane < 3; ++lane) {
                    if (!resources.has(static_cast<SubtaskSlot>(lane))) continue;
                    bool closed = false;
                    ASSERT_TRUE(scheduler_gm_mix_close(graph_view, base(), &locals[first + lane], run, &closed));
                    ASSERT_TRUE(closed);
                }
            }
            EXPECT_EQ(
                scheduler_completion_task_state(scheduler_task_control_at(base(), &locals[0], task)->completion_state),
                static_cast<int64_t>(SchedulerTaskState::DONE)
            );
        }
        for (uint32_t lane = 0; lane < 6; ++lane)
            scheduler_flush_completions(run, &locals[lane]);
    }

    void run_late_worker(uint32_t worker = 3) {
        startup_kernel_calls = 0;
        startup_ssbuf_base = reinterpret_cast<uint8_t *>(cluster_storage[worker / 3].data());
        if (worker % 3 != 0)
            scheduler_cluster_signal_clear(
                scheduler_cluster_signal_at(reinterpret_cast<uint64_t>(startup_ssbuf_base)), 59
            );
        scheduler_gm_store(storage.run_control->bootstrap_scan_arrived_count, UINT64_C(8));
        scheduler_gm_store(storage.run_control->bootstrap_arrived_count, UINT64_C(8));
        write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
        write_reg(RegId::COND, 0);
        if (storage.layout.protocol_layout_offset != 0)
            run_resident_scheduler<true, false>(&storage.contexts[worker], storage.run_control, base(), 4, 0, 0);
        else run_resident_scheduler<true, false>(&storage.contexts[worker], storage.run_control, base(), 0, 0, 0);
    }

    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<2> graph{2};
    SchedulerGraphView graph_view{};
    alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> registers{};
    std::array<std::array<uint64_t, PLATFORM_SSBUF_SIZE / sizeof(uint64_t)>, 3> cluster_storage{};
};

class HbgGmMixStartupExit : public ::testing::TestWithParam<std::array<LaneMask, 2>> {};

TEST(HbgGmMixStartupExit, RejectsMalformedPublishedGraphBeforeBootstrap) {
    for (uint32_t field = 0; field < 3; ++field) {
        SCOPED_TRACE(field);
        StartupFixture fixture({LaneMask::both_aiv(), LaneMask::both_aiv()});
        auto &context = fixture.storage.contexts[3];
        if (field == 0) context.graph_view_version = 0;
        else if (field == 1) context.graph_view_bytes = 32;
        else context.original_task_count = 0;
        fixture.run_late_worker();
        EXPECT_EQ(startup_kernel_calls, 0u);
        EXPECT_EQ(fixture.storage.run_control->bootstrap_scan_arrived_count, 8u);
        EXPECT_EQ(
            fixture.storage.run_control->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
        );
    }
}

TEST_P(HbgGmMixStartupExit, LateZeroWorkCoordinatorClosesEveryAdmittedMixBeforeExitAck) {
    StartupFixture fixture(GetParam());
    fixture.finish_both_tasks_on_early_clusters();
    ASSERT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->resolved_task_count, UINT64_C(0)), 2u);
    ASSERT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->scheduler_error, UINT64_C(0)), 0u);
    for (uint64_t task = 0; task < 2; ++task) {
        ASSERT_EQ(scheduler_gm_fetch_add(fixture.control(task)->next_block, UINT64_C(0)), 2u);
        ASSERT_EQ(
            scheduler_completion_count(scheduler_gm_fetch_add(fixture.control(task)->completion_state, UINT64_C(0))),
            2 * fixture.storage.metadata[task].active_mask.count()
        );
        ASSERT_EQ(
            scheduler_completion_task_state(scheduler_gm_query(fixture.control(task)->completion_state)),
            static_cast<int64_t>(SchedulerTaskState::DONE)
        );
        ASSERT_EQ(scheduler_gm_fetch_add(fixture.seal(task, 1), UINT64_C(0)), 0u);
    }
    fixture.run_late_worker();
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(1, 1), UINT64_C(0)), 1u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->bootstrap_scan_arrived_count, UINT64_C(0)), 9u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->bootstrap_arrived_count, UINT64_C(0)), 9u);
    EXPECT_EQ(startup_kernel_calls, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->scheduler_error, UINT64_C(0)), 0u);
}

INSTANTIATE_TEST_SUITE_P(
    CompletedMixShapes, HbgGmMixStartupExit,
    ::testing::Values(
        std::array<LaneMask, 2>{LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false), LaneMask::both_aiv()},
        std::array<LaneMask, 2>{LaneMask::both_aiv(), LaneMask::both_aiv()},
        std::array<LaneMask, 2>{LaneMask::both_aiv(), LaneMask::all()}
    )
);

TEST(HbgGmMixStartupExit, FailedRunAcknowledgesExitWithoutFabricatingCloseSeals) {
    StartupFixture fixture({LaneMask::both_aiv(), LaneMask::both_aiv()});
    fixture.finish_both_tasks_on_early_clusters();
    ASSERT_EQ(scheduler_gm_fetch_add(fixture.storage.run_control->resolved_task_count, UINT64_C(0)), 2u);
    scheduler_record_error(
        fixture.storage.run_control, SCHEDULER_TASK_REF_INVALID, SchedulerGraphResult::INVALID_ARGUMENTS,
        &fixture.graph_view, &fixture.storage.contexts[3], SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
    );
    fixture.run_late_worker();
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(1, 1), UINT64_C(0)), 0u);
    EXPECT_EQ(startup_kernel_calls, 0u);
    EXPECT_EQ(
        scheduler_gm_fetch_add(fixture.storage.run_control->scheduler_error, UINT64_C(0)),
        static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS)
    );
}

TEST(HbgGmMixStartupExit, MissingCoordinatorSealTimeoutIdentifiesTheWaitingTaskAndWorker) {
    StartupFixture fixture({LaneMask::both_aiv(), LaneMask::both_aiv()});
    fixture.finish_both_tasks_on_early_clusters();
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
    EXPECT_EQ(startup_kernel_calls, 0u);
    EXPECT_EQ(scheduler_gm_fetch_add(fixture.seal(0, 1), UINT64_C(0)), 0u);
}

TEST(HbgProtocolStartupExit, RealExecutorFinishesLateZeroWorkMembersAfterClosingEveryObligation) {
    StartupFixture fixture({LaneMask::both_aiv(), LaneMask::all()}, true);
    ASSERT_EQ(fixture.storage.layout.activity_buffers_offset, 0u);
    fixture.finish_both_tasks_on_early_clusters();
    auto *workers = scheduler_state_at<SchedulerProtocolWorkerTrace>(
        fixture.base(), fixture.storage.layout.protocol.workers_offset
    );
    auto *members = scheduler_state_at<SchedulerProtocolMemberTrace>(
        fixture.base(), fixture.storage.layout.protocol.members_offset
    );
    for (uint32_t worker = 3; worker < 6; ++worker) {
        fixture.run_late_worker(worker);
        EXPECT_EQ(scheduler_scalar_current(), nullptr);
        ASSERT_EQ(fixture.storage.run_control->scheduler_error, 0u);
        EXPECT_EQ(workers[worker].finished, 1u);
        EXPECT_EQ(workers[worker].error_flags, 0u);
        EXPECT_EQ(workers[worker].next_sequence, 4u);
        EXPECT_EQ(members[worker].observed.sequence, 1u);
        EXPECT_EQ(members[worker].closed.sequence, 2u);
        EXPECT_EQ(members[9 + worker].observed.sequence, 3u);
        EXPECT_EQ(members[9 + worker].closed.sequence, 4u);
        EXPECT_EQ(startup_kernel_calls, 0u);
    }
}

TEST(HbgProtocolStartupExit, RealExecutorErrorExitDoesNotMarkUnfinishedProtocolSuccessful) {
    StartupFixture fixture({LaneMask::both_aiv(), LaneMask::both_aiv()}, true);
    fixture.finish_both_tasks_on_early_clusters();
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
}

TEST(HbgProtocolStartupExit, RealExecutorSealTimeoutPreservesObservedButUnclosedMember) {
    StartupFixture fixture({LaneMask::both_aiv(), LaneMask::both_aiv()}, true);
    fixture.finish_both_tasks_on_early_clusters();
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
}

uint32_t scalar_record_count(const SchedulerScalarTrace &trace, SchedPhaseKind kind) {
    uint32_t cursor = 0;
    uint32_t count = 0;
    while (cursor < trace.storage_slots) {
        SchedulerScalarRecord record{};
        if (!scheduler_decode_scalar_storage(trace.records, trace.storage_slots, cursor, record)) {
            ADD_FAILURE() << "Invalid Scalar storage";
            break;
        }
        if ((record.kind & 0xffffU) == static_cast<uint32_t>(kind)) ++count;
    }
    return count;
}

TEST(HbgScalarObserverLifetime, BootstrapPreservesTheResidentBinding) {
    scheduler_test::FixtureStorage storage(1, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0});
    scheduler_test::BasicGraphBuffer<1> graph(1);
    graph.executable(0, 0);
    auto *base = storage.scheduler_state->base();
    storage.prepare_catalog();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, 59));
    ASSERT_TRUE(storage.prepare_calls(graph.graph()));
    auto *local = storage.local_context(&storage.contexts[0]);
    local->config.scheduler_count = 1;
    storage.run_control->expected_task_count = 1;
    storage.test_graphs[0] = graph.graph();
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = local->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    {
        SchedulerScalarBinding binding;
        binding.attach(&trace);
        SchedulerWorkerStats stats{};
        ASSERT_TRUE(
            (bootstrap_ready_graph<false, true>(graph.graph(), base, local, storage.run_control, 1, &stats, 3))
        );
        EXPECT_EQ(scheduler_scalar_current(), &trace);
        EXPECT_GT(scalar_record_count(trace, SchedPhaseKind::ScalarAtomicFetchAdd), 0u);
        EXPECT_GT(scalar_record_count(trace, SchedPhaseKind::ScalarBypassLoad), 0u);
    }
    EXPECT_EQ(scheduler_scalar_current(), nullptr);
}

TEST(HbgScalarObserverLifetime, DispatchLoopRecordsMemoryPrimitives) {
    scheduler_test::FixtureStorage storage(1, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0});
    scheduler_test::BasicGraphBuffer<1> graph(1);
    graph.executable(0, 0);
    auto *base = storage.scheduler_state->base();
    storage.prepare_catalog();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, 59));
    storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&idle_ready_kernel);
    ASSERT_TRUE(storage.prepare_calls(graph.graph()));
    storage.test_graphs[0] = graph.graph();
    for (uint32_t lane = 0; lane < 3; ++lane)
        storage.contexts[0].cluster_worker_ids[lane] = lane;
    auto *local = storage.local_context(&storage.contexts[0]);
    local->config.scheduler_count = 3;
    auto *run = storage.run_control;
    run->expected_task_count = 1;

    alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> registers{};
    ScopedStartupRegisterObserver observer([] {});
    startup_reg_base = registers.data();
    startup_kernel_calls = 0;
    write_reg(RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    auto buffer = std::make_unique<SchedulerActivityBuffer>();
    auto records = std::make_unique<SchedulerScalarStorageSlot[]>(2 * SCHEDULER_SCALAR_RECORD_CAPACITY);
    auto &trace = local->profiling->scalar;
    trace.buffer = buffer.get();
    trace.records = records.get();
    trace.window_start = scheduler_cycles();
    {
        SchedulerScalarBinding binding;
        binding.attach(&trace);
        SchedulerWorkerStats stats{};
        ASSERT_TRUE((run_ready_dispatch_loop<false, true>(graph.graph(), base, local, run, &stats, 3)));
        EXPECT_EQ(startup_kernel_calls, 1u);
        EXPECT_EQ(scalar_record_count(trace, SchedPhaseKind::ScalarExecutorScan), 0u);
        EXPECT_EQ(scheduler_scalar_current(), &trace);
        EXPECT_GT(scalar_record_count(trace, SchedPhaseKind::ScalarAtomicCas), 0u);
        EXPECT_GT(scalar_record_count(trace, SchedPhaseKind::ScalarBypassLoad), 0u);
        EXPECT_GT(scalar_record_count(trace, SchedPhaseKind::ScalarDcciOut), 0u);
        EXPECT_GT(scalar_record_count(trace, SchedPhaseKind::ScalarDsbAll), 0u);
    }
    EXPECT_EQ(scheduler_scalar_current(), nullptr);
}

class HbgExecutionIdentityEntry : public ::testing::TestWithParam<std::tuple<uint32_t, uint64_t, bool>> {};

TEST_P(HbgExecutionIdentityEntry, RealExecutorCapturesEntryPayloadBeforeKernelCanChangeIt) {
    const auto [worker, level, sampled] = GetParam();
    scheduler_test::FixtureStorage storage(1, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0}, level == 4);
    scheduler_test::BasicGraphBuffer<1> graph(1);
    graph.executable(0, worker == 0 ? 0 : 1);
    auto &metadata = storage.metadata[0];
    metadata.active_mask = worker == 0 ? LaneMask::aic() : LaneMask::aiv0();
    metadata.kernel_ids[0] = worker == 0 ? 1 : UINT16_MAX;
    metadata.kernel_ids[1] = worker == 0 ? UINT16_MAX : 1;
    metadata.timing_slot = sampled ? 0 : -1;
    storage.prepare_catalog();
    auto *base = storage.scheduler_state->base();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, 59));
    storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&identity_test_kernel);
    ASSERT_TRUE(storage.prepare_calls(graph.graph()));
    storage.test_graphs[worker] = graph.graph();
    for (uint32_t lane = 0; lane < 3; ++lane)
        storage.contexts[worker].cluster_worker_ids[lane] = lane;
    auto *local = storage.local_context(&storage.contexts[worker]);
    const bool with_profiling = level != 0 || sampled;
    if (!with_profiling) local->profiling = nullptr;
    ASSERT_TRUE(scheduler_protocol_bind(base, local));
    local->config.scheduler_count = 3;
    auto *run = storage.run_control;
    run->expected_task_count = 1;

    alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> registers{};
    startup_reg_base = registers.data();
    startup_kernel_calls = 0;
    identity_test_payload = scheduler_execution_binding_at(base, local->config.prepared, 0);
    write_reg(RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    ScopedStartupRegisterObserver stop_after_completion([&] {
        if (local->pending_completed + run->resolved_task_count == run->expected_task_count)
            write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    });
    SchedulerWorkerStats stats{};
    if (with_profiling) {
        ASSERT_TRUE((run_ready_dispatch_loop<false, true>(graph.graph(), base, local, run, &stats, level)));
    } else {
        ASSERT_TRUE((run_ready_dispatch_loop<false, false>(graph.graph(), base, local, run, &stats, level)));
    }
    ASSERT_EQ(startup_kernel_calls, 1u);
    ASSERT_EQ(identity_test_payload->local_context.block_idx, 77);
    EXPECT_EQ(local->slots[0].state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(scheduler_read_error(run), 0u);
    EXPECT_EQ(scheduler_completion_count(scheduler_task_control_at(base, local, 0)->completion_state), 1u);
    const auto *trace = scheduler_state_at<SchedulerTaskTrace>(base, storage.layout.trace_cells_offset);
    if (level == 0) {
        EXPECT_EQ(trace->execution_identity_flags, 0u);
        EXPECT_EQ(trace->valid, 0u);
        if (sampled) {
            EXPECT_GT(trace->kernel_start_cycles, 0u);
            EXPECT_GE(trace->kernel_end_cycles, trace->kernel_start_cycles);
        } else {
            EXPECT_EQ(trace->kernel_start_cycles, 0u);
            EXPECT_EQ(trace->kernel_end_cycles, 0u);
        }
    } else {
        EXPECT_EQ(trace->valid, 1u);
        EXPECT_EQ(trace->execution_run_epoch, 59u);
        EXPECT_EQ(trace->execution_order_key, UINT64_MAX);
        EXPECT_EQ(trace->execution_group_seq, 0u);
        EXPECT_EQ(trace->execution_block_idx, 0u);
        EXPECT_EQ(trace->execution_block_num, 1u);
        EXPECT_EQ(trace->execution_subtask_slot, worker == 0 ? 0u : 1u);
        EXPECT_EQ(trace->execution_pending_slot, 0u);
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
        EXPECT_EQ(event->cluster_lane, worker);
        EXPECT_EQ(
            event->flags, SCHEDULER_PROTOCOL_ENTRY | SCHEDULER_PROTOCOL_EXECUTED | SCHEDULER_PROTOCOL_PERMIT_NO_LOG |
                              SCHEDULER_PROTOCOL_PAYLOAD_PUBLISHED
        );
    }
}

INSTANTIATE_TEST_SUITE_P(
    LanesAndProfilingLevels, HbgExecutionIdentityEntry,
    ::testing::Combine(::testing::Values(0u, 1u, 2u), ::testing::Values<uint64_t>(0, 1, 2, 3, 4), ::testing::Bool())
);

enum class IdleDispatchArrival : uint8_t { READY, EXIT, ERROR };

class HbgIdleDispatchArrival
    : public ::testing::TestWithParam<std::tuple<uint32_t, uint64_t, bool, IdleDispatchArrival>> {};

TEST_P(HbgIdleDispatchArrival, EmptyPollsObserveNewWorkAndTerminationWithoutSpuriousOwnership) {
    const auto [worker, level, ordered, arrival] = GetParam();
    scheduler_test::FixtureStorage storage(
        4, 3, 3, SchedulerSpmdRequirements{3, 1, ordered ? 1u : 0u, 0, 0}, level == 4
    );
    scheduler_test::BasicGraphBuffer<4> graph(4);
    // Two externally driven control predecessors keep all physical kernels
    // discoverable but blocked until the observed event really completes.
    storage.metadata[0].flags = 0;
    storage.metadata[1].flags = 0;
    graph.executable(2, worker == 0 ? 0 : 1, {0});
    auto &metadata = storage.metadata[2];
    metadata.flags |= SCHEDULER_TASK_HAS_FANIN;
    metadata.active_mask = worker == 0 ? LaneMask::aic() : LaneMask::aiv0();
    metadata.kernel_ids[0] = worker == 0 ? 1 : UINT16_MAX;
    metadata.kernel_ids[1] = worker == 0 ? UINT16_MAX : 1;
    metadata.kernel_ids[2] = UINT16_MAX;
    metadata.timing_slot = -1;
    if (ordered) {
        graph.executable(3, 1, {1});
        storage.metadata[3].active_mask = LaneMask::aiv0();
        storage.metadata[3].logical_block_num = 2;
        storage.metadata[3].flags =
            SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_SYNC_START | SCHEDULER_TASK_HAS_FANIN;
        storage.metadata[3].kernel_ids[0] = storage.metadata[3].kernel_ids[2] = UINT16_MAX;
        storage.metadata[3].kernel_ids[1] = 1;
    } else {
        graph.executable(3, 0, {1});
        storage.metadata[3].flags |= SCHEDULER_TASK_HAS_FANIN;
    }
    storage.prepare_catalog();
    auto *base = storage.scheduler_state->base();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, 83));
    storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&idle_ready_kernel);
    ASSERT_TRUE(storage.prepare_calls(graph.graph()));
    storage.test_graphs[worker] = graph.graph();
    for (uint32_t member = 0; member < 3; ++member)
        for (uint32_t lane = 0; lane < 3; ++lane)
            storage.contexts[member].cluster_worker_ids[lane] = lane;
    auto *local = storage.local_context(&storage.contexts[worker]);
    local->config.scheduler_count = 3;
    if (level == 0) local->profiling = nullptr;
    ASSERT_TRUE(scheduler_protocol_bind(base, local));
    SchedulerAdmissionLocalState admission{};
    if (ordered) {
        ASSERT_TRUE(scheduler_initialize_admission(base, local, &admission));
        local->admission = &admission;
    }
    auto *run = storage.run_control;
    run->expected_task_count = 2;
    alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> registers{};
    startup_reg_base = registers.data();
    startup_kernel_calls = 0;
    write_reg(RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    uint32_t observations = 0;
    bool queued = false;
    ScopedStartupRegisterObserver observer([&] {
        ++observations;
        if (observations == 5) {
            if (arrival == IdleDispatchArrival::READY) {
                auto *completed = scheduler_task_control_at(base, local, 0);
                scheduler_gm_store(
                    completed->completion_state,
                    scheduler_completion_completed_state(scheduler_completion_total(completed->completion_state))
                );
                queued = true;
            } else if (arrival == IdleDispatchArrival::EXIT) {
                write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
            } else {
                scheduler_record_error(
                    run, 0, SchedulerGraphResult::INVALID_ARGUMENTS, nullptr, local,
                    SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT
                );
            }
        }
        if (observations == 128) write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    });
    SchedulerWorkerStats stats{};
    const bool result = level == 0 ?
                            run_ready_dispatch_loop<false, false>(graph.graph(), base, local, run, &stats, level) :
                            run_ready_dispatch_loop<false, true>(graph.graph(), base, local, run, &stats, level);
    EXPECT_EQ(result, arrival != IdleDispatchArrival::ERROR);
    EXPECT_LT(observations, 128u);
    EXPECT_GE(observations, 5u);
    EXPECT_EQ(startup_kernel_calls, arrival == IdleDispatchArrival::READY ? 1u : 0u);
    EXPECT_EQ(queued, arrival == IdleDispatchArrival::READY);
    if (ordered) {
        EXPECT_EQ(admission.scan_cursor, 0u);
        EXPECT_EQ(admission.current_task_ref, SCHEDULER_TASK_REF_INVALID);
        auto *control = scheduler_task_control_at(base, local, 3);
        EXPECT_EQ(control->next_block, 0u);
        EXPECT_EQ(scheduler_completion_count(control->completion_state), 0u);
        EXPECT_EQ(control->admission_state, static_cast<uint64_t>(SchedulerAdmissionState::UNCLAIMED));
    }
    if (arrival == IdleDispatchArrival::ERROR) {
        EXPECT_EQ(run->error_site, static_cast<uint64_t>(SchedulerErrorSite::ORDERED_EXIT_IN_FLIGHT));
    } else {
        EXPECT_EQ(scheduler_read_error(run), 0u);
    }
}

INSTANTIATE_TEST_SUITE_P(
    LanesLevelsAdmissionAndArrival, HbgIdleDispatchArrival,
    ::testing::Combine(
        ::testing::Values(0u, 1u, 2u), ::testing::Values<uint64_t>(0, 1, 2, 3, 4), ::testing::Bool(),
        ::testing::Values(IdleDispatchArrival::READY, IdleDispatchArrival::EXIT, IdleDispatchArrival::ERROR)
    )
);

TEST(HbgProtocolExecution, ConsumerDiscoveryKeepsTheCompletedGenerationSeparateFromItsSuccessor) {
    scheduler_test::FixtureStorage storage(2, 3, 1, SchedulerSpmdRequirements{3, 1, 0, 0}, true);
    scheduler_test::BasicGraphBuffer<2> graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    storage.metadata[1].flags |= SCHEDULER_TASK_HAS_FANIN;
    storage.prepare_catalog();
    auto *base = storage.scheduler_state->base();
    ASSERT_TRUE(scheduler_init_spmd_task_records(base, storage.layout, 73));
    storage.callable_addresses[1] = reinterpret_cast<uint64_t>(&protocol_chain_kernel);
    ASSERT_TRUE(storage.prepare_calls(graph.graph()));
    storage.test_graphs[0] = graph.graph();
    for (uint32_t lane = 0; lane < 3; ++lane)
        storage.contexts[0].cluster_worker_ids[lane] = lane;
    auto *local = storage.local_context(&storage.contexts[0]);
    local->config.scheduler_count = 3;
    ASSERT_TRUE(scheduler_protocol_bind(base, local));
    auto *run = storage.run_control;
    run->expected_task_count = 2;

    alignas(64) std::array<uint8_t, SIM_REG_TOTAL_SIZE> registers{};
    startup_reg_base = registers.data();
    startup_kernel_calls = 0;
    write_reg(RegId::DATA_MAIN_BASE, AICPU_IDLE_TASK_ID);
    ScopedStartupRegisterObserver stop_after_completion([&] {
        if (local->pending_completed + run->resolved_task_count == run->expected_task_count)
            write_reg(RegId::DATA_MAIN_BASE, AICORE_EXIT_SIGNAL);
    });
    SchedulerWorkerStats stats{};
    ASSERT_TRUE((run_ready_dispatch_loop<false, true>(graph.graph(), base, local, run, &stats, 4)));
    ASSERT_EQ(startup_kernel_calls, 2u);
    EXPECT_EQ(local->slots[0].state, SchedulerDispatchSlotState::FREE);
    EXPECT_EQ(scheduler_read_error(run), 0u);
    EXPECT_EQ(scheduler_completion_count(scheduler_task_control_at(base, local, 0)->completion_state), 1u);
    EXPECT_EQ(scheduler_completion_count(scheduler_task_control_at(base, local, 1)->completion_state), 1u);
    const auto *events =
        scheduler_state_at<SchedulerProtocolExecutionTrace>(base, storage.layout.protocol.executions_offset);
    EXPECT_EQ(events[0].task_ref, 0);
    EXPECT_EQ(events[1].task_ref, 1);
    EXPECT_EQ(events[0].pending_slot, events[1].pending_slot);
    EXPECT_GE(events[1].publication_before_cycles, events[0].completion_consumed.cycles);
    EXPECT_NE(events[0].trace_published.sequence, 0u);
    EXPECT_GT(events[1].entry.sequence, events[0].trace_published.sequence);
    EXPECT_GT(events[1].trace_published.sequence, events[1].completion_consumed.sequence);
    EXPECT_EQ(events[1].ready_source, static_cast<uint32_t>(SchedulerReadySource::POOL));
    EXPECT_EQ(events[1].publication_mode, static_cast<uint32_t>(SchedulerPublicationMode::DISPATCH));
    const auto *worker = scheduler_state_at<SchedulerProtocolWorkerTrace>(base, storage.layout.protocol.workers_offset);
    EXPECT_EQ(worker->error_flags, 0u);
    startup_reg_base = nullptr;
}

}  // namespace

volatile uint8_t *sim_get_reg_base() {
    if (startup_register_observer && !startup_observing_register) {
        startup_observing_register = true;
        startup_register_observer();
        startup_observing_register = false;
    }
    return startup_reg_base;
}
uint32_t sim_get_physical_core_id() { return 3; }
uint64_t platform_ssbuf_hardware_base_address() { return reinterpret_cast<uint64_t>(startup_ssbuf_base); }
// A zero register base leaves the executor's PMU staging off, which is what
// these startup cases observe: they drive the dispatch loop, not the counters.
uint32_t get_aicore_profiling_flag() { return 0; }
uint64_t get_aicore_pmu_reg_base() { return 0; }
