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
#include "runtime_types.h"

uint64_t get_aicore_report_epoch();
#include "lane_loop.h"
#include "calendar_cache.h"
#include "calendar_plan.h"

#include <gtest/gtest.h>
#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstring>
#include <thread>
#include <vector>

namespace {
using namespace simpler::percounter;

alignas(64) std::array<uint8_t, 4096> registers{};
alignas(64) std::array<uint8_t, PLATFORM_SSBUF_SIZE> startup_ssbuf{};

TaskInput task(uint32_t mask, int blocks = 1, bool sync = false) {
    TaskInput result{};
    result.active_mask = mask;
    result.is_mix = __builtin_popcount(mask) > 1;
    result.core_type_index = mask == 1 ? 0 : 1;
    result.subtask_count = __builtin_popcount(mask);
    result.block_num = blocks;
    result.sync_start = sync;
    return result;
}

struct CalendarDeferTest : testing::Test {
    static constexpr uint32_t defer_bit = 1U << 24;
    alignas(32) uint32_t done{0};
    alignas(32) uint32_t shell_done{0};
    alignas(32) uint32_t lifecycle{0};
    uint32_t signal{0};
    CalendarTiming sample{100, 110, 0, 0};
    PercounterControl control{};
    PercounterCounter counter{};
    pto2::resident::AsyncGroupData scratch{};
    std::array<DeferredCompletionEntry, 1> entries{};
    std::array<PercounterAsyncCompletion, 2> groups{};
    LaneConfig producer{};
    CalendarExecutionExtra extra{};
    CalendarWait wait{};

    CalendarDeferTest() {
        const uint64_t set = reinterpret_cast<uint64_t>(&done);
        counter.completion_notify_address = set;
        counter.calendar_defer_bit = defer_bit;
        extra.shell_add_cmd = percounter_notify_add_address(reinterpret_cast<uint64_t>(&shell_done));
        wait = {set | CALENDAR_WAIT_DEFER, (uint64_t{24} << 24) | 1};
        producer.lifecycle_notify = &lifecycle;
        producer.control = &control;
        producer.counters = &counter;
        producer.task_count = 1;
        producer.async_group = &scratch;
        producer.async_entries = entries.data();
        producer.async_completions = groups.data();
        producer.async_completion_capacity = groups.size();
        entries[0].addr = reinterpret_cast<uint64_t>(&signal);
        entries[0].expected_value = 1;
        entries[0].engine = COMPLETION_ENGINE_SDMA;
        entries[0].completion_type = COMPLETION_TYPE_COUNTER;
    }

    void seal(CalendarTiming &timing) {
        scratch.count = 1;
        bool deferred = false;
        ASSERT_TRUE(percounter_calendar_seal(producer, &extra, true, &timing, 0, &deferred));
        ASSERT_TRUE(deferred);
    }
};

TEST(CalendarLifecycle, EarlyAndLateWorkersAcknowledgeOnlyTheirOwnRejectedExitAcrossWrap) {
    PercounterControl control{};
    alignas(32) uint32_t signal = 1;
    for (uint32_t first_sample : {UINT32_MAX, signal}) {
        control.rejected_exit[1] = signal;
        EXPECT_EQ(percounter_wait_for_admission(&control, 1, signal, first_sample), PercounterAdmission::Rejected);
        EXPECT_EQ(control.rejected_exit[1], 0u);
    }
    // An accepted peer seeing the identical broadcast proceeds to COND retirement.
    EXPECT_EQ(percounter_wait_for_admission(&control, 0, signal, UINT32_MAX), PercounterAdmission::Retire);
    // With only a retained EXIT the worker waits for this launch's START.
    const uint32_t retained = signal;
    signal = 2;
    EXPECT_EQ(percounter_wait_for_admission(&control, 1, signal, retained), PercounterAdmission::Start);
    EXPECT_EQ(control.lane_error, 0u);
}

TEST(CalendarLifecycle, WatchdogStillWaitsForThisLaunchToRejectTheWorker) {
    PercounterControl control{};
    control.scheduler_timeout_cycles = 1;
    alignas(32) uint32_t signal = 3;
    std::thread supervisor([&] {
        while (percounter_gm_read(control.lane_error) == 0)
            std::this_thread::yield();
        percounter_gm_store(control.rejected_exit[0], uint32_t{5});
        __atomic_store_n(&signal, 5u, __ATOMIC_RELEASE);
    });
    EXPECT_EQ(percounter_wait_for_admission(&control, 0, signal, 3), PercounterAdmission::Rejected);
    supervisor.join();
    EXPECT_EQ(control.lane_error, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_EQ(control.rejected_exit[0], 0u);
}

TEST_F(CalendarDeferTest, DirectWaitOnlyBacksOffWhenItHasNoRegisteredDeferToService) {
    const auto many_credits = (uint64_t{24} << 24) | 24;
    uint64_t cursor = 0;
    uint32_t remaining = 0;
    EXPECT_EQ(
        percounter_calendar_poll_completion(producer, wait.read_cmd, many_credits, &cursor, &remaining),
        CompletionPollState::PENDING
    );
    EXPECT_EQ(remaining, 24U);
    EXPECT_EQ(percounter_calendar_retry_scale(remaining), 3U);

    seal(sample);
    EXPECT_EQ(
        percounter_calendar_poll_completion(producer, wait.read_cmd, many_credits, &cursor, &remaining),
        CompletionPollState::PENDING
    );
    EXPECT_EQ(remaining, 0U);
}

TEST_F(CalendarDeferTest, CounterReadyAtSealDoesNotRegisterADefer) {
    signal = 1;
    scratch.count = 1;
    bool deferred = true;
    ASSERT_TRUE(percounter_calendar_seal(producer, &extra, true, &sample, 0, &deferred));
    EXPECT_FALSE(deferred);
    EXPECT_EQ(scratch.count, 0U);
    EXPECT_EQ(counter.calendar_defer_head, 0U);
    EXPECT_EQ(control.async_registered, 0U);
    EXPECT_EQ(control.async_settled, 0U);
    EXPECT_EQ(sample.defer_register, 0U);
    EXPECT_EQ(sample.defer_completion, 0U);
    EXPECT_EQ(done, 0U);
    EXPECT_EQ(shell_done, 0U);
}

TEST_F(CalendarDeferTest, RegistrationSetsFlagWithoutReleasingTheTaskOrGraph) {
    seal(sample);
    EXPECT_EQ(done, defer_bit);
    EXPECT_EQ(shell_done, 0U);
    EXPECT_EQ(control.async_registered, 1U);
    EXPECT_NE(sample.defer_register, 0U);
    EXPECT_EQ(sample.defer_completion, 0U);

    LaneConfig successor = producer;
    successor.cluster_lane = 1;
    uint64_t cursor = 0;
    ASSERT_TRUE(percounter_calendar_service_defer(successor, 0, &cursor));
    EXPECT_EQ(done, defer_bit);
    EXPECT_EQ(shell_done, 0U);

    signal = 1;
    ASSERT_TRUE(percounter_calendar_wait_list(successor, &wait, 1));
    EXPECT_EQ(done, defer_bit | 1);
    EXPECT_EQ(shell_done, 1U);
    EXPECT_EQ(control.async_settled, 1U);
    EXPECT_GE(sample.defer_completion, sample.defer_register);
}

TEST_F(CalendarDeferTest, TwoSuccessorsSettleEveryDeferredGroupOnce) {
    CalendarTiming second_sample{100, 110, 0, 0};
    seal(sample);
    seal(second_sample);
    wait.target = (uint64_t{24} << 24) | 2;
    signal = 1;
    auto consume = [&](uint32_t lane) {
        LaneConfig successor = producer;
        successor.cluster_lane = lane;
        EXPECT_TRUE(percounter_calendar_wait_list(successor, &wait, 1));
    };
    std::thread first(consume, 1);
    std::thread second(consume, 2);
    first.join();
    second.join();
    EXPECT_EQ(done, defer_bit | 2);
    EXPECT_EQ(shell_done, 2U);
    EXPECT_EQ(control.async_registered, 2U);
    EXPECT_EQ(control.async_settled, 2U);
    ASSERT_GT(sample.defer_register, 0U);
    ASSERT_GT(second_sample.defer_register, 0U);
    EXPECT_GE(sample.defer_completion, sample.defer_register);
    EXPECT_GE(second_sample.defer_completion, second_sample.defer_register);
}

TEST_F(CalendarDeferTest, SharesADeferredCompletionOnlyAfterTheOwnerHasObservedIt) {
    struct Program {
        CalendarRangeHeader header;
        CalendarExecutionRow rows[2];
        CalendarDispatch binding;
        CalendarWait wait;
        CalendarExecutionExtra extra;
    } program{};

    constexpr uint32_t phase_base = 8;
    constexpr uint32_t owner_ordinal = 1;
    constexpr uint32_t token = phase_base + owner_ordinal;
    program.rows[0] = {0, owner_ordinal, 0, 2, 1, 1 | CALENDAR_EXEC_INPUT_OWNER};
    program.rows[1] = {0, owner_ordinal, 1, 2, 0, 1 | CALENDAR_EXEC_INPUT_FOLLOWER};
    program.wait = wait;
    DispatchPayload arguments{};
    const PercounterKernelFunc kernel = [](int64_t *args) {
        __atomic_fetch_add(args, INT64_C(1), __ATOMIC_RELAXED);
    };
    program.binding = {
        reinterpret_cast<uint64_t>(kernel), reinterpret_cast<uint64_t>(&arguments), 0,
        offsetof(Program, wait) - offsetof(Program, binding), offsetof(Program, extra) - offsetof(Program, binding)
    };

    seal(sample);
    PercounterSsbufRegion cluster{};
    // A previous phase's publication cannot admit the follower.
    percounter_ssbuf_store_field(&cluster.mix.release[1], token - 1);
    LaneConfig owner = producer;
    owner.cluster_lane = 1;
    owner.ssbuf = &cluster;
    LaneConfig follower = owner;
    follower.cluster_lane = 2;
    const auto run = [&](const LaneConfig &lane, uint32_t row) {
        CalendarRange range{};
        range.rows_offset = offsetof(Program, rows) + row * sizeof(CalendarExecutionRow);
        range.row_count = 1;
        range.binding_offset = offsetof(Program, binding);
        range.sequence_base = phase_base;
        return percounter_calendar_execute_range(
            lane, reinterpret_cast<const uint8_t *>(&program), sizeof(program), range, 0, 1, 0
        );
    };
    percounter_gm_store(control.lane_error, uint64_t{1});
    EXPECT_FALSE(run(follower, 1));
    EXPECT_EQ(arguments.args[0], 0U);
    percounter_gm_store(control.lane_error, uint64_t{0});

    // Only the owner may service the defer. It must be able to reach the
    // kernel even though the follower has not entered this phase.
    bool owner_ok = false;
    std::thread first([&] {
        owner_ok = run(owner, 0);
    });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (percounter_gm_read(groups[0].calendar_defer_owner) == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    EXPECT_EQ(percounter_gm_read(groups[0].calendar_defer_owner), 2U);
    EXPECT_EQ(percounter_ssbuf_load_field(&cluster.mix.release[1]), token - 1);
    EXPECT_EQ(percounter_gm_read(done), defer_bit);
    percounter_gm_store(signal, uint32_t{1});
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (percounter_ssbuf_load_field(&cluster.mix.release[1]) != token && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    if (percounter_ssbuf_load_field(&cluster.mix.release[1]) != token)
        percounter_gm_store(control.lane_error, uint64_t{1});
    first.join();
    EXPECT_TRUE(owner_ok);
    EXPECT_EQ(done, defer_bit | 1);
    EXPECT_EQ(control.async_settled, 1U);
    EXPECT_EQ(percounter_ssbuf_load_field(&cluster.mix.release[1]), token);
    EXPECT_EQ(arguments.args[0], 1U);

    ASSERT_TRUE(run(follower, 1));
    EXPECT_EQ(arguments.args[0], 2U);
    EXPECT_EQ(control.async_settled, 1U);
    EXPECT_EQ(percounter_ssbuf_load_field(&cluster.mix.release[2]), 0U);
}

TEST_F(CalendarDeferTest, FollowerCanProceedWhenDoneWasPublishedButTheOwnerIsDelayed) {
    done = 1;
    const CalendarWait direct{reinterpret_cast<uint64_t>(&done) | CALENDAR_WAIT_NOTIFY, 1};
    PercounterSsbufRegion cluster{};
    LaneConfig follower = producer;
    follower.cluster_lane = 2;
    follower.ssbuf = &cluster;
    EXPECT_TRUE(percounter_calendar_wait_cluster_input(follower, direct, 1, 0));
    EXPECT_EQ(percounter_ssbuf_load_field(&cluster.mix.release[1]), 0U);
}

TEST_F(CalendarDeferTest, FollowerCanSettleItsOwnInputWhenTheOwnerCannotPublish) {
    seal(sample);
    signal = 1;
    PercounterSsbufRegion cluster{};
    LaneConfig follower = producer;
    follower.cluster_lane = 2;
    follower.ssbuf = &cluster;
    ASSERT_TRUE(percounter_calendar_wait_cluster_input(follower, wait, 1, 0));
    EXPECT_EQ(done, defer_bit | 1);
    EXPECT_EQ(control.async_settled, 1U);
    EXPECT_EQ(percounter_ssbuf_load_field(&cluster.mix.release[1]), 0U);
}

TEST_F(CalendarDeferTest, SummaryFollowersCanUseTheCounterWithoutAnOwnerToken) {
    PercounterSsbufRegion cluster{};
    LaneConfig follower = producer;
    follower.cluster_lane = 2;
    follower.ssbuf = &cluster;
    const auto address = reinterpret_cast<uint64_t>(&done);
    const CalendarWait summary{address | CALENDAR_WAIT_NOTIFY, 2};
    const CalendarWait fused{address | CALENDAR_WAIT_FUSED_FANIN, (uint64_t{2} << 32) | 2};

    for (const auto &header : {summary, fused}) {
        // A sticky defer bit and one credit never satisfy a two-credit input gate.
        done = defer_bit | 1;
        control.lane_error = 1;
        EXPECT_FALSE(percounter_calendar_wait_cluster_input<true>(follower, header, 1, 0));
        control.lane_error = 0;
        done = defer_bit | 2;
        EXPECT_TRUE(percounter_calendar_wait_cluster_input<true>(follower, header, 1, 0));
        EXPECT_EQ(percounter_ssbuf_load_field(&cluster.mix.release[1]), 0U);
        EXPECT_EQ(control.async_settled, 0U);
    }
}

TEST_F(CalendarDeferTest, ASharedFusedOwnerKeepsItsOwnDeferredHelperAndRangeOrdinal) {
    struct Program {
        CalendarRangeHeader header;
        CalendarExecutionRow rows[2];
        CalendarDispatch binding;
        CalendarWait waits[3];
        CalendarExecutionExtra extra;
    } program{};
    constexpr uint32_t phase_base = 8;
    constexpr uint32_t owner_phase_offset = 4;
    constexpr uint32_t owner_ordinal = owner_phase_offset + 1;
    constexpr uint32_t token = phase_base + owner_ordinal;
    const auto address = reinterpret_cast<uint64_t>(&done);
    counter.completion_notify_target = 2;
    counter.calendar_defer_bit = 1U << 25;

    // Helper begin=1 must survive packing and must not become the release token.
    program.rows[0] = {0, 1, 0, 2, 1, 1 | CALENDAR_EXEC_FANIN | CALENDAR_EXEC_INPUT_OWNER};
    program.rows[1] = {0, owner_ordinal, 1, 2, 0, 1 | CALENDAR_EXEC_FANIN | CALENDAR_EXEC_INPUT_FOLLOWER};
    program.waits[0] = {address | CALENDAR_WAIT_FUSED_FANIN, (uint64_t{2} << 32) | 2};
    program.waits[2] = {address | CALENDAR_WAIT_DEFER, (uint64_t{25} << 24) | 2};
    DispatchPayload arguments{};
    const PercounterKernelFunc kernel = [](int64_t *args) {
        __atomic_fetch_add(args, INT64_C(1), __ATOMIC_RELAXED);
    };
    program.binding = {
        reinterpret_cast<uint64_t>(kernel), reinterpret_cast<uint64_t>(&arguments), 0,
        offsetof(Program, waits) - offsetof(Program, binding), offsetof(Program, extra) - offsetof(Program, binding)
    };
    // Writer zero is already done. The owner's writer still needs the real signal.
    done = 1;
    seal(sample);
    PercounterSsbufRegion cluster{};
    percounter_ssbuf_store_field(&cluster.mix.release[1], token - 1);
    LaneConfig owner = producer;
    owner.cluster_lane = 1;
    owner.ssbuf = &cluster;
    LaneConfig follower = owner;
    follower.cluster_lane = 2;

    const auto run = [&](const LaneConfig &lane, uint32_t row, uint32_t phase_offset) {
        CalendarRange range{};
        range.rows_offset = offsetof(Program, rows) + row * sizeof(CalendarExecutionRow);
        range.row_count = 1;
        range.binding_offset = offsetof(Program, binding);
        range.sequence_base = phase_base;
        range.phase_row_offset = phase_offset;
        return percounter_calendar_execute_range(
            lane, reinterpret_cast<const uint8_t *>(&program), sizeof(program), range, 0, 1, 0
        );
    };
    percounter_gm_store(control.lane_error, uint64_t{1});
    EXPECT_FALSE(run(follower, 1, 2));
    EXPECT_EQ(arguments.args[0], 0U);
    percounter_gm_store(control.lane_error, uint64_t{0});

    bool owner_ok = false;
    std::thread first([&] {
        owner_ok = run(owner, 0, owner_phase_offset);
    });
    auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (percounter_gm_read(groups[0].calendar_defer_owner) == 0 && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    EXPECT_EQ(percounter_gm_read(groups[0].calendar_defer_owner), 2U);
    EXPECT_EQ(percounter_gm_read(done), (1U << 25) | 1);
    EXPECT_EQ(percounter_ssbuf_load_field(&cluster.mix.release[1]), token - 1);
    percounter_gm_store(signal, uint32_t{1});
    deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (percounter_ssbuf_load_field(&cluster.mix.release[1]) != token && std::chrono::steady_clock::now() < deadline)
        std::this_thread::yield();
    if (percounter_ssbuf_load_field(&cluster.mix.release[1]) != token)
        percounter_gm_store(control.lane_error, uint64_t{1});
    first.join();

    EXPECT_TRUE(owner_ok);
    EXPECT_EQ(done, (1U << 25) | 2);
    EXPECT_EQ(control.async_settled, 1U);
    EXPECT_EQ(percounter_ssbuf_load_field(&cluster.mix.release[1]), token);
    EXPECT_EQ(arguments.args[0], 1U);
    EXPECT_TRUE(run(follower, 1, 2));
    EXPECT_EQ(arguments.args[0], 2U);
    EXPECT_EQ(control.async_settled, 1U);
}

TEST(PercounterCalendar, FusedFaninsWaitForAllIndependentDeferredProducersInEveryOrder) {
    constexpr uint32_t width = CALENDAR_FUSED_FANIN_MAX_PRODUCERS;
    for (uint32_t producers = 2; producers <= width; ++producers) {
        std::array<uint32_t, width> order{0, 1, 2, 3};
        do {
            alignas(32) uint32_t done = 0;
            const auto address = reinterpret_cast<uint64_t>(&done);
            PercounterControl control{};
            std::array<PercounterCounter, width> counters{};
            std::array<PercounterAsyncCompletion, width> groups{};
            std::array<CalendarTiming, width> samples{};
            std::array<uint32_t, width> signals{};
            std::array<CalendarWait, width + 1> waits{};
            waits[0] = {address | CALENDAR_WAIT_FUSED_FANIN, (uint64_t{producers} << 32) | producers};
            pto2::resident::AsyncGroupData scratch{};
            DeferredCompletionEntry registration{};
            registration.expected_value = 1;
            registration.engine = COMPLETION_ENGINE_SDMA;
            registration.completion_type = COMPLETION_TYPE_COUNTER;
            alignas(32) uint32_t lifecycle = 0;
            LaneConfig lane{};
            lane.lifecycle_notify = &lifecycle;
            lane.control = &control;
            lane.counters = counters.data();
            lane.task_count = producers;
            lane.async_group = &scratch;
            lane.async_entries = &registration;
            lane.async_completions = groups.data();
            lane.async_completion_capacity = groups.size();
            for (uint32_t task = 0; task < producers; ++task) {
                counters[task].completion_notify_address = address;
                counters[task].calendar_defer_bit = 1U << (24 + task);
                counters[task].completion_notify_target = producers;
                waits[1 + task] = {
                    address | CALENDAR_WAIT_DEFER, (uint64_t{task} << 32) | (uint64_t{24 + task} << 24) | producers
                };
                registration.addr = reinterpret_cast<uint64_t>(&signals[task]);
                scratch.count = 1;
                CalendarExecutionExtra extra{};
                extra.task_id = task;
                bool deferred = false;
                ASSERT_TRUE(percounter_calendar_seal(lane, &extra, false, &samples[task], 0, &deferred));
                ASSERT_TRUE(deferred);
            }
            const uint32_t sticky_bits = ((1U << producers) - 1) << 24;
            EXPECT_EQ(done, sticky_bits);
            EXPECT_EQ(control.async_registered, producers);
            std::array<uint64_t, width> cursors{};
            const auto poll = [&](uint32_t task) {
                lane.cluster_index = task / PLATFORM_CORES_PER_BLOCKDIM;
                lane.cluster_lane = task % PLATFORM_CORES_PER_BLOCKDIM;
                return percounter_calendar_poll_fused_fanin(
                    lane, address, producers, task, counters[task].calendar_defer_bit, &cursors[task]
                );
            };
            for (uint32_t task = 0; task < producers; ++task)
                EXPECT_EQ(poll(task), CompletionPollState::PENDING);
            for (uint32_t settled = 0; settled < producers; ++settled) {
                const uint32_t task = order[settled];
                signals[task] = 1;
                EXPECT_EQ(poll(task), CompletionPollState::PENDING);
                EXPECT_EQ(done, sticky_bits | (settled + 1));
                EXPECT_EQ(control.async_settled, settled + 1);
                const auto expected =
                    settled + 1 == producers ? CompletionPollState::READY : CompletionPollState::PENDING;
                EXPECT_EQ(percounter_calendar_poll_fused_fanin(lane, address, producers, 0, 0, nullptr), expected);
            }
            EXPECT_TRUE(percounter_calendar_wait_fanin(lane, waits.data(), 0, 0, 0));
            for (uint32_t task = 0; task < producers; ++task)
                EXPECT_TRUE(percounter_calendar_wait_fanin(lane, waits.data(), task, 1, 0));
            EXPECT_EQ(control.async_settled, producers);
        } while (std::next_permutation(order.begin(), order.begin() + producers));
    }
}

TEST(PercounterCalendar, SeparatesCoreOccupancyFromDeferredTail) {
    const auto ordinary = calendar_sample_cost({100, 104, 0, 0});
    ASSERT_TRUE(ordinary.has_value());
    EXPECT_EQ(ordinary->occupancy_cycles, 4U);
    EXPECT_EQ(ordinary->defer_tail_cycles, 0U);

    const auto deferred = calendar_sample_cost({100, 104, 108, 119});
    ASSERT_TRUE(deferred.has_value());
    EXPECT_EQ(deferred->occupancy_cycles, 8U);
    EXPECT_EQ(deferred->defer_tail_cycles, 11U);
    EXPECT_FALSE(calendar_sample_cost({100, 104, 108, 0}).has_value());
    EXPECT_FALSE(calendar_sample_cost({100, 104, 108, 105}).has_value());
}

TEST(PercounterCalendar, SyncMixReservesDistinctClustersAndWaitsForWholePredecessor) {
    std::vector<TaskInput> inputs{task(1), task(5, 2, true), task(2)};
    const int32_t first[] = {0};
    const int32_t second[] = {1};
    inputs[1].fanin_count = inputs[2].fanin_count = 1;
    inputs[1].fanin_ids = first;
    inputs[2].fanin_ids = second;

    CalendarPlan plan;
    const std::vector<CalendarRowCost> durations{{5, 0}, {2, 0}, {3, 0}, {4, 0}, {7, 0}, {2, 0}};
    ASSERT_TRUE(build_static_calendar(inputs, 2, durations, &plan));

    CalendarHeader header{};
    std::memcpy(&header, plan.bytes.data(), sizeof(header));
    ASSERT_EQ(header.worker_count, 6);
    ASSERT_EQ(header.row_count, 6);
    EXPECT_EQ(header.join_count, 2);
    std::array<uint32_t, 7> indices{};
    std::array<CalendarRow, 6> rows{};
    ASSERT_LE(header.index_offset + sizeof(indices), plan.bytes.size());
    ASSERT_LE(header.row_offset + sizeof(rows), plan.bytes.size());
    std::memcpy(indices.data(), plan.bytes.data() + header.index_offset, sizeof(indices));
    std::memcpy(rows.data(), plan.bytes.data() + header.row_offset, sizeof(rows));
    ASSERT_TRUE(std::is_sorted(indices.begin(), indices.end()));
    ASSERT_EQ(indices.back(), rows.size());
    std::array<int32_t, 2> cubes{-1, -1}, vectors{-1, -1};
    for (uint32_t worker = 0; worker < header.worker_count; ++worker) {
        for (uint32_t i = indices[worker]; i < indices[worker + 1]; ++i) {
            const auto entry = rows[i].entry;
            if (percounter_order_task_id(entry) != 1) continue;
            const auto block = percounter_order_block_idx(entry);
            ASSERT_GE(block, 0);
            ASSERT_LT(block, 2);
            if (worker < 2) {
                EXPECT_EQ(cubes[block], -1);
                cubes[block] = static_cast<int32_t>(worker);
            } else {
                EXPECT_EQ((worker - 2) % 2, 1);  // This MIX uses V1.
                EXPECT_EQ(vectors[block], -1);
                vectors[block] = static_cast<int32_t>((worker - 2) / 2);
            }
        }
    }
    EXPECT_GE(cubes[0], 0);
    EXPECT_GE(cubes[1], 0);
    EXPECT_NE(cubes[0], cubes[1]);
    EXPECT_EQ(cubes, vectors);

    uint64_t replay = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 2, plan, durations, &replay));
    EXPECT_EQ(replay, 14);  // Predecessor 5, slowest co-starting role 7, successor 2.
}

TEST(PercounterCalendar, PriorityWindowReservesCubeForAReadyChain) {
    std::vector<TaskInput> inputs{task(2), task(1), task(1), task(2)};
    const int32_t vector_producer[] = {0};
    const int32_t cube_producer[] = {1};
    inputs[1].fanin_count = inputs[3].fanin_count = 1;
    inputs[1].fanin_ids = vector_producer;
    inputs[3].fanin_ids = cube_producer;
    const std::vector<CalendarRowCost> costs{{2}, {2}, {5}, {5}};
    CalendarPlan earliest;
    CalendarPlan windowed;
    ASSERT_TRUE(build_static_calendar(inputs, 1, costs, &earliest));
    ASSERT_TRUE(build_static_calendar(inputs, 1, costs, &windowed, CalendarBlockOrder::SUBMISSION, 3));

    uint64_t earliest_cycles = 0;
    uint64_t windowed_cycles = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, earliest, costs, &earliest_cycles));
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, windowed, costs, &windowed_cycles));
    EXPECT_EQ(earliest_cycles, 12U);
    EXPECT_EQ(windowed_cycles, 9U);
}

TEST(PercounterCalendar, BackfillsTheGapBeforeACubeInputBecomesReady) {
    std::vector<TaskInput> inputs{task(2), task(1), task(1), task(2), task(1)};
    const int32_t vector_producer[] = {0};
    const int32_t cube_producer[] = {1};
    inputs[1].fanin_count = inputs[3].fanin_count = 1;
    inputs[1].fanin_ids = vector_producer;
    inputs[3].fanin_ids = cube_producer;
    const std::vector<CalendarRowCost> costs{{2}, {2}, {5}, {5}, {1}};
    CalendarPlan windowed;
    CalendarPlan filled;
    ASSERT_TRUE(build_static_calendar(inputs, 1, costs, &windowed, CalendarBlockOrder::SUBMISSION, 3));
    ASSERT_TRUE(build_static_calendar(
        inputs, 1, costs, &filled, CalendarBlockOrder::SUBMISSION, 3, CalendarPriority::LONGEST_ROW, true
    ));

    uint64_t windowed_cycles = 0;
    uint64_t filled_cycles = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, windowed, costs, &windowed_cycles));
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, filled, costs, &filled_cycles));
    EXPECT_EQ(windowed_cycles, 10U);
    EXPECT_EQ(filled_cycles, 9U);
}

TEST(PercounterCalendar, WholeTaskWorkMovesBlocksAheadOfIndependentCube) {
    std::vector<TaskInput> inputs{task(1), task(1, 2), task(2)};
    const int32_t predecessor[] = {1};
    inputs[2].fanin_count = 1;
    inputs[2].fanin_ids = predecessor;
    const std::vector<CalendarRowCost> costs{{6}, {4}, {4}, {1}};
    CalendarPlan longest_row;
    CalendarPlan whole_task;
    ASSERT_TRUE(build_static_calendar(inputs, 1, costs, &longest_row));
    ASSERT_TRUE(build_static_calendar(
        inputs, 1, costs, &whole_task, CalendarBlockOrder::SUBMISSION, 0, CalendarPriority::WORK_AND_WAIT
    ));

    uint64_t longest_row_cycles = 0;
    uint64_t whole_task_cycles = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, longest_row, costs, &longest_row_cycles));
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, whole_task, costs, &whole_task_cycles));
    EXPECT_EQ(longest_row_cycles, 15U);
    EXPECT_EQ(whole_task_cycles, 14U);
}

TEST(PercounterCalendar, RejectsSyncMixWiderThanAvailableClusters) {
    CalendarPlan plan;
    EXPECT_FALSE(build_static_calendar({task(5, 2, true)}, 1, {}, &plan));
}

TEST(PercounterCalendar, MeasuredPriorityAdvancesCubeThatUnblocksVector) {
    std::vector<TaskInput> inputs{task(1), task(1), task(2)};
    const int32_t predecessor[] = {1};
    inputs[2].fanin_count = 1;
    inputs[2].fanin_ids = predecessor;

    CalendarPlan cold;
    const std::vector<CalendarRowCost> durations{{5, 0}, {7, 0}, {10, 0}};
    ASSERT_TRUE(build_static_calendar(inputs, 1, {}, &cold));
    uint64_t cold_cycles = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, cold, durations, &cold_cycles));
    EXPECT_EQ(cold_cycles, 22);  // Cold Cube order runs both tasks before Vector can start.

    CalendarPlan measured;
    ASSERT_TRUE(build_static_calendar(inputs, 1, durations, &measured));
    uint64_t measured_cycles = 0;
    ASSERT_TRUE(evaluate_static_calendar(inputs, 1, measured, durations, &measured_cycles));
    EXPECT_EQ(measured_cycles, 17);  // Vector overlaps the shorter Cube task.
    EXPECT_EQ(measured.logical_row_by_packed.front(), 1);
    EXPECT_EQ(measured.predicted_cycles, measured_cycles);
}

}  // namespace

volatile uint8_t *sim_get_reg_base() { return registers.data(); }
uint32_t sim_get_physical_core_id() { return 0; }
uint64_t platform_ssbuf_hardware_base_address() { return reinterpret_cast<uint64_t>(startup_ssbuf.data()); }
uint32_t get_aicore_profiling_flag() { return 0; }
ChipSwimlaneActiveHead *get_chip_swimlane_aicore_head() { return nullptr; }

TEST(PercounterScalarStorage, PreservesMixedIntervalsWithoutLogicalCount) {
    std::array<uint64_t, 8> slots{};
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.depth = 2;
    trace.append_memory(ProfilePhase::SsbufLoad, 100, 117);
    trace.write({200, uint64_t{UINT32_MAX} + 300, 42, static_cast<uint32_t>(ProfilePhase::BindContext), 7});
    trace.append_memory(ProfilePhase::DsbAll, 500, 500);

    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 100);
    EXPECT_EQ(record.end_time, 117);
    EXPECT_EQ(record.task_ref, -1);
    EXPECT_EQ(record.count, 1);
    EXPECT_EQ(record.kind, static_cast<uint32_t>(ProfilePhase::SsbufLoad) | (3U << 16));
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 200);
    EXPECT_EQ(record.end_time, uint64_t{UINT32_MAX} + 300);
    EXPECT_EQ(record.task_ref, 42);
    EXPECT_EQ(record.count, 7);
    ASSERT_TRUE(
        percounter_decode_scalar_storage(slots.data(), trace.storage_words, cursor, trace.window_start, record)
    );
    EXPECT_EQ(record.start_time, 500);
    EXPECT_EQ(record.end_time, 500);
    EXPECT_EQ(cursor, trace.storage_words);
    EXPECT_EQ(trace.dropped, 0);
}

TEST(PercounterScalarStorage, RejectsWholeExtendedRecordAtCapacityBoundary) {
    constexpr uint32_t capacity = PERCOUNTER_SCALAR_WORD_CAPACITY;
    std::vector<uint64_t> slots(capacity + 1);
    slots.back() = 0xfeed;
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.storage_words = capacity - 1;
    trace.write({100, 200, 42, static_cast<uint32_t>(ProfilePhase::BindContext), 1});
    EXPECT_EQ(trace.storage_words, capacity - 1);
    EXPECT_EQ(trace.dropped, 1);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 0);
    trace.append_memory(ProfilePhase::SsbufLoad, 300, 400);
    EXPECT_EQ(trace.storage_words, capacity);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 300);
    trace.append_memory(ProfilePhase::SsbufLoad, 500, 600);
    EXPECT_EQ(trace.storage_words, capacity);
    EXPECT_EQ(trace.dropped, 2);
    EXPECT_EQ(static_cast<uint32_t>(slots[capacity - 1]), 300);
    EXPECT_EQ(slots.back(), 0xfeed);
}

TEST(PercounterScalarStorage, InvalidIntervalsNeverConsumeStorage) {
    std::array<uint64_t, 4> slots{};
    PercounterScalarTrace trace{};
    trace.records = slots.data();
    trace.append_memory(ProfilePhase::SsbufLoad, 200, 100);
    EXPECT_EQ(trace.storage_words, 0);
    EXPECT_EQ(trace.dropped, 1);
    trace.dropped = UINT32_MAX;
    trace.append_memory(ProfilePhase::SsbufLoad, 200, 100);
    EXPECT_EQ(trace.dropped, UINT32_MAX);
}

TEST(PercounterScalarStorage, CompactWordsPreserveTicksAndUseFullWidthForLongIntervals) {
    std::array<uint64_t, 24> words{};
    PercounterScalarTrace trace{};
    trace.records = words.data();
    trace.window_start = 1000;
    trace.depth = 2;
    trace.append_memory(ProfilePhase::SsbufLoad, 1007, 1012);
    EXPECT_EQ(words[0], UINT64_C(0x031b000500000007));
    trace.append_memory(ProfilePhase::SsbufLoad, 2000, 2000 + uint64_t{UINT16_MAX} + 1);
    trace.append_memory(ProfilePhase::SsbufLoad, uint64_t{UINT32_MAX} + 1001, uint64_t{UINT32_MAX} + 1002);
    trace.depth = 127;
    trace.append_memory(ProfilePhase::SsbufLoad, 3000, 3010);
    EXPECT_EQ(trace.storage_words, 16);
    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.start_time, 1007);
    EXPECT_EQ(record.end_time, 1012);
    EXPECT_EQ(record.kind >> 16, 3);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.end_time, 2000 + uint64_t{UINT16_MAX} + 1);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.start_time, uint64_t{UINT32_MAX} + 1001);
    ASSERT_TRUE(percounter_decode_scalar_storage(words.data(), trace.storage_words, cursor, 1000, record));
    EXPECT_EQ(record.kind >> 16, 128);
    EXPECT_EQ(cursor, trace.storage_words);
}

TEST(PercounterScalarStorage, RejectsPartialExtendedRowsAndOverflowingAbsoluteTime) {
    const std::array<uint64_t, 4> partial{PERCOUNTER_SCALAR_EXTENDED, 1, 2, 3};
    uint32_t cursor = 0;
    SchedulerScalarRecord record{};
    EXPECT_FALSE(percounter_decode_scalar_storage(partial.data(), partial.size(), cursor, 0, record));
    const uint64_t offset = 1;
    cursor = 0;
    EXPECT_FALSE(percounter_decode_scalar_storage(&offset, 1, cursor, UINT64_MAX, record));
    const uint64_t duration = UINT64_C(1) << 32;
    cursor = 0;
    EXPECT_FALSE(percounter_decode_scalar_storage(&duration, 1, cursor, UINT64_MAX, record));
}
