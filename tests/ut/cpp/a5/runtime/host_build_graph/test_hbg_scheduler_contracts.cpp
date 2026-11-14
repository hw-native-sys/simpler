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
#include <atomic>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <vector>

#include "host_build_graph/runtime_types.h"
#include "aicore_scheduler_error.h"
#include "aicore_scheduler_state.h"
#include "scheduler_ready_watchdog.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_topology.h"
#include "scheduler/scheduler_types.h"
#include "scheduler/scheduler_watchdog.h"

#include "hbg_scheduler_test_support.h"

namespace {

TEST(AicoreSchedulerError, MapsInternalFailureToExistingHostStatus) {
    EXPECT_EQ(aicore_scheduler_runtime_error_code(0), SIMPLER_ERROR_NONE);
    EXPECT_EQ(aicore_scheduler_runtime_error_code(1), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)),
        SIMPLER_ERROR_SCHEDULER_TIMEOUT
    );
    EXPECT_EQ(aicore_scheduler_runtime_error_code(UINT64_MAX), SIMPLER_ERROR_INVALID_ARGS);
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::ASYNC_WAIT_OVERFLOW)),
        SIMPLER_ERROR_ASYNC_WAIT_OVERFLOW
    );
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::ASYNC_COMPLETION_INVALID)),
        SIMPLER_ERROR_ASYNC_COMPLETION_INVALID
    );
    EXPECT_EQ(
        aicore_scheduler_runtime_error_code(static_cast<uint64_t>(SchedulerGraphResult::ASYNC_REGISTRATION_FAILED)),
        SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED
    );
    EXPECT_STREQ(
        aicore_scheduler_error_name(static_cast<uint64_t>(SchedulerGraphResult::ASYNC_WAIT_OVERFLOW)),
        "ASYNC_WAIT_OVERFLOW"
    );
    EXPECT_STREQ(
        aicore_scheduler_error_name(static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)), "SCHEDULER_TIMEOUT"
    );
}

TEST(AicoreSchedulerError, LatchesFirstHostVisibleFailure) {
    std::atomic<int32_t> status{SIMPLER_ERROR_NONE};
    EXPECT_FALSE(latch_aicore_scheduler_runtime_error(&status, 0));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_NONE);

    EXPECT_TRUE(latch_aicore_scheduler_runtime_error(&status, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT)));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_SCHEDULER_TIMEOUT);

    status.store(SIMPLER_ERROR_READY_QUEUE_OVERFLOW, std::memory_order_relaxed);
    EXPECT_FALSE(latch_aicore_scheduler_runtime_error(&status, 2));
    EXPECT_EQ(status.load(std::memory_order_relaxed), SIMPLER_ERROR_READY_QUEUE_OVERFLOW);
}

TEST(AicoreSchedulerState, AcceptsOnlyCurrentResidentModes) {
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_resident(0));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_RESIDENT_READY));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_resident(UINT32_MAX));
}

TEST(AicoreSchedulerState, ResidentAcceptsOrdinaryPureAndMixMultiBlock) {
    for (uint32_t active : {1U, 2U, 3U}) {
        SCOPED_TRACE(active);
        const uint32_t max_blocks = static_cast<uint32_t>(INT16_MAX) / active;
        for (uint32_t blocks : {1U, 2U, 3U, 31U, 32U, 33U, 64U, 65U, 97U, 512U, max_blocks}) {
            SCOPED_TRACE(blocks);
            uint32_t subtasks = 0;
            ASSERT_TRUE(scheduler_subtask_count_checked(blocks, active, &subtasks));
            EXPECT_EQ(subtasks, blocks * active);
            EXPECT_TRUE(scheduler_resident_task_shape_supported(active, blocks, false));
        }
    }
}

TEST(AicoreSchedulerState, ResidentShapeRejectsSubtaskProductOverflowAndInvalidDimensions) {
    EXPECT_FALSE(scheduler_resident_task_shape_supported(0, 1, false));
    EXPECT_FALSE(scheduler_resident_task_shape_supported(4, 1, false));
    EXPECT_FALSE(scheduler_resident_task_shape_supported(UINT32_MAX, 1, false));
    for (uint32_t active : {1U, 2U, 3U}) {
        SCOPED_TRACE(active);
        const uint32_t max_blocks = static_cast<uint32_t>(INT16_MAX) / active;
        for (uint32_t blocks : {0U, max_blocks + 1, static_cast<uint32_t>(INT16_MAX) + 1, UINT32_MAX}) {
            SCOPED_TRACE(blocks);
            EXPECT_FALSE(scheduler_resident_task_shape_supported(active, blocks, false));
        }
    }
}

TEST(AicoreSchedulerState, ResidentSyncStartShapeRequiresMultipleBlocksAndSeparateCapacityCheck) {
    for (uint32_t active : {1U, 2U, 3U}) {
        for (uint32_t blocks : {2U, 32U, 33U, 65U, 512U}) {
            SCOPED_TRACE(active);
            SCOPED_TRACE(blocks);
            EXPECT_TRUE(scheduler_resident_task_shape_supported(active, blocks, true));
        }
        EXPECT_FALSE(scheduler_resident_task_shape_supported(active, 1, true));
        EXPECT_FALSE(scheduler_resident_task_shape_supported(active, 0, true));
        EXPECT_FALSE(scheduler_resident_task_shape_supported(active, UINT32_MAX, true));
    }
}

TEST(AicoreSchedulerState, SyncDemandCountsDistinctCoresInTheDeclaredResourceScope) {
    for (uint64_t workers : {3U, 6U, 9U, 96U, 108U}) {
        for (LaneMask mask :
             {LaneMask::aic(), LaneMask::aiv0(), LaneMask::aiv1(),
              LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false),
              LaneMask(/*aic=*/true, /*aiv0=*/false, /*aiv1=*/true), LaneMask::both_aiv(), LaneMask::all()}) {
            SCOPED_TRACE(workers);
            SCOPED_TRACE(mask.count());
            const uint32_t positions =
                static_cast<uint32_t>(workers / 3) * (mask.to_shape() == ResourceShape::AIV ? 2 : 1);
            for (uint32_t blocks : {1U, 2U, positions, positions + 1, 512U, UINT32_MAX}) {
                SCOPED_TRACE(blocks);
                SchedulerMinimumWorkerDemand demand{};
                const bool valid = blocks > 1 && blocks <= positions;
                EXPECT_EQ(scheduler_resident_worker_demand_checked(mask, blocks, true, workers, &demand), valid);
                if (valid) {
                    const bool mix = mask.is_mix();
                    EXPECT_EQ(demand.aic, (mix || mask.is_aic()) ? blocks : 0u);
                    EXPECT_EQ(
                        demand.aiv, mix ? blocks * (static_cast<uint32_t>(mask.has_aiv0()) +
                                                    static_cast<uint32_t>(mask.has_aiv1())) :
                                          (mask.is_aic() ? 0 : blocks)
                    );
                }
            }
            SchedulerMinimumWorkerDemand ordinary{};
            ASSERT_TRUE(scheduler_resident_worker_demand_checked(mask, 512, false, workers, &ordinary));
            const auto expected = scheduler_minimum_worker_demand(mask);
            EXPECT_EQ(ordinary.aic, expected.aic);
            EXPECT_EQ(ordinary.aiv, expected.aiv);
        }
    }
}

TEST(AicoreSchedulerState, ResidentDemandRejectsInvalidTopologyMaskAndOutput) {
    SchedulerMinimumWorkerDemand demand{};
    for (uint64_t workers : {UINT64_C(0), UINT64_C(1), UINT64_C(5), UINT64_C(109), UINT64_C(111), UINT64_MAX})
        EXPECT_FALSE(scheduler_resident_worker_demand_checked(LaneMask::aic(), 2, true, workers, &demand));
    // An empty lane set is the only mask the demand check can still be asked
    // about and must refuse: a byte with bits outside the three lanes is not a
    // LaneMask any caller can construct.
    EXPECT_FALSE(scheduler_resident_worker_demand_checked(LaneMask::none(), 2, true, 96, &demand));
    EXPECT_FALSE(scheduler_resident_worker_demand_checked(LaneMask::aic(), 2, true, 96, nullptr));
}

TEST(AicoreSchedulerState, OrdinaryMixMasksCanExecuteMoreBlocksThanPhysicalClusters) {
    for (LaneMask mask :
         {LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false), LaneMask(/*aic=*/true, /*aiv0=*/false, /*aiv1=*/true),
          LaneMask::both_aiv(), LaneMask::all()}) {
        SCOPED_TRACE(mask.count());
        const uint32_t active = static_cast<uint32_t>(mask.count());
        for (uint32_t blocks : {SCHEDULER_CLUSTER_CAPACITY + 1, 512U}) {
            SCOPED_TRACE(blocks);
            EXPECT_TRUE(scheduler_resident_task_shape_supported(active, blocks, false));
            const auto demand = scheduler_minimum_worker_demand(mask);
            EXPECT_EQ(demand.aic, 1u);
            EXPECT_EQ(demand.aiv, mask.has_both_aiv() ? 2u : 1u);
        }
    }
}

TEST(AicoreSchedulerState, OrdinaryPureAndMixShapeValidationIsSubmissionOrderIndependent) {
    for (uint32_t mix_subtasks : {2U, 3U}) {
        for (uint32_t mix_blocks : {1U, 2U, 33U, 512U}) {
            for (bool mix_first : {false, true}) {
                SCOPED_TRACE(mix_subtasks);
                SCOPED_TRACE(mix_blocks);
                SCOPED_TRACE(mix_first);
                for (uint32_t index = 0; index < 2; ++index) {
                    const bool mix = (index == 0) == mix_first;
                    const uint32_t active = mix ? mix_subtasks : 1U;
                    const uint32_t blocks = mix ? mix_blocks : 512U;
                    EXPECT_TRUE(scheduler_resident_task_shape_supported(active, blocks, false));
                }
            }
        }
    }
}

TEST(AicoreSchedulerState, OrdinaryDemandCountsOneExecutionPositionOrMixGroup) {
    struct DemandCase {
        LaneMask mask;
        uint32_t aic;
        uint32_t aiv;
    };
    for (const DemandCase &test : {
             DemandCase{LaneMask::aic(), 1, 0},
             DemandCase{LaneMask::aiv0(), 0, 1},
             DemandCase{LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false), 1, 1},
             DemandCase{LaneMask(/*aic=*/true, /*aiv0=*/false, /*aiv1=*/true), 1, 1},
             DemandCase{LaneMask::both_aiv(), 1, 2},
             DemandCase{LaneMask::all(), 1, 2},
         }) {
        SCOPED_TRACE(test.mask.count());
        const auto demand = scheduler_minimum_worker_demand(test.mask);
        EXPECT_EQ(demand.aic, test.aic);
        EXPECT_EQ(demand.aiv, test.aiv);
    }
}

TEST(AicoreSchedulerWatchdog, UsesElapsedWallClockBudget) {
    EXPECT_FALSE(scheduler_watchdog_expired(100, UINT64_MAX, 0));
    EXPECT_FALSE(scheduler_watchdog_expired(100, 199, 100));
    EXPECT_TRUE(scheduler_watchdog_expired(100, 200, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(UINT64_MAX - 10, 3, 15));
    EXPECT_TRUE(scheduler_watchdog_expired(UINT64_MAX - 10, 4, 15));
}

TEST(AicoreSchedulerWatchdog, RejectsBackwardAndHalfRangeSamples) {
    constexpr uint64_t half_range = uint64_t{1} << 63;
    EXPECT_TRUE(scheduler_watchdog_expired(0, half_range - 1, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(0, half_range, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(0, UINT64_MAX, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(1000, 800, 0));
}

TEST(AicpuReadyWatchdog, SupervisorSeedAheadOfFirstSampleKeepsDeadline) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(1000);
    const uint64_t last_progress = watchdog.last_progress();
    EXPECT_FALSE(scheduler_watchdog_expired(last_progress, 800, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(last_progress, 1099, 100));
    EXPECT_TRUE(scheduler_watchdog_expired(last_progress, 1100, 100));
    EXPECT_EQ(watchdog.last_progress(), 1000);
}

TEST(AicpuReadyWatchdog, RealProgressAllowsTotalExecutionBeyondBudget) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(0);
    uint64_t now = 0;
    int advances = 0;
    EXPECT_TRUE(watchdog.wait_until_ready(
        [&] {
            return advances == 4;
        },
        [&] {
            return now;
        },
        [&] {
            now += 90;
            watchdog.record_progress(now);
            ++advances;
        },
        100
    ));
    EXPECT_EQ(now, 360u);
    EXPECT_TRUE(scheduler_watchdog_expired(0, now, 100));
}

TEST(AicpuReadyWatchdog, UnchangedProgressExpiresAtOriginalBoundary) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(100);
    uint64_t now = 100;
    int polls = 0;
    EXPECT_FALSE(watchdog.wait_until_ready(
        [] {
            return false;
        },
        [&] {
            return now;
        },
        [&] {
            now += 25;
            ++polls;
        },
        100
    ));
    EXPECT_EQ(now, 200u);
    EXPECT_EQ(polls, 4);
    EXPECT_EQ(watchdog.last_progress(), 100u);
}

TEST(AicpuReadyWatchdog, SupervisorNeverStartsRemainsBounded) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(100);
    EXPECT_FALSE(watchdog.wait_until_ready(
        [] {
            return false;
        },
        [] {
            return 200u;
        },
        [] {
            ADD_FAILURE() << "expired wait must not spin";
        },
        100
    ));
}

TEST(AicpuReadyWatchdog, SupervisorLossAfterProgressRemainsBounded) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(100);
    watchdog.record_progress(150);
    uint64_t now = 200;
    int unrelated_activity = 0;
    EXPECT_FALSE(watchdog.wait_until_ready(
        [] {
            return false;
        },
        [&] {
            return now;
        },
        [&] {
            ++unrelated_activity;
            now += 25;
        },
        100
    ));
    EXPECT_EQ(now, 250u);
    EXPECT_EQ(unrelated_activity, 2);
    EXPECT_EQ(watchdog.last_progress(), 150u);
}

TEST(AicpuReadyWatchdog, ReadyWinsBeforeClockAndAtExpiryRecheck) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(100);
    EXPECT_TRUE(watchdog.wait_until_ready(
        [] {
            return true;
        },
        [] {
            ADD_FAILURE() << "ready wait must not sample the clock";
            return 0u;
        },
        [] {
            ADD_FAILURE() << "ready wait must not spin";
        },
        100
    ));
    bool ready = false;
    EXPECT_TRUE(watchdog.wait_until_ready(
        [&] {
            return ready;
        },
        [&] {
            ready = true;
            return 200u;
        },
        [] {
            ADD_FAILURE() << "ready publication must win expiry";
        },
        100
    ));
}

TEST(AicpuReadyWatchdog, RefreshDuringExpiryUsesNewClockSample) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(100);
    int samples = 0;
    bool ready = false;
    EXPECT_TRUE(watchdog.wait_until_ready(
        [&] {
            return ready;
        },
        [&] {
            if (++samples == 1) {
                watchdog.record_progress(205);
                return 200u;
            }
            return 206u;
        },
        [&] {
            if (samples == 2) ready = true;
        },
        100
    ));
    EXPECT_EQ(samples, 2);
}

TEST(AicpuReadyWatchdog, ProgressPublicationAheadOfSampleDoesNotExpire) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(101);
    uint64_t now = 100;
    bool ready = false;
    int relaxations = 0;
    EXPECT_TRUE(watchdog.wait_until_ready(
        [&] {
            return ready;
        },
        [&] {
            return now;
        },
        [&] {
            now = 102;
            ready = true;
            ++relaxations;
        },
        2000
    ));
    EXPECT_EQ(relaxations, 1);
}

TEST(AicpuReadyWatchdog, OlderClockSampleStillExpiresAtProgressDeadline) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(101);
    uint64_t now = 100;
    int samples = 0;
    EXPECT_FALSE(watchdog.wait_until_ready(
        [] {
            return false;
        },
        [&] {
            ++samples;
            return now;
        },
        [&] {
            now = now == 100 ? 102 : 2101;
        },
        2000
    ));
    EXPECT_EQ(now, 2101u);
    EXPECT_EQ(samples, 3);
    EXPECT_EQ(watchdog.last_progress(), 101u);
}

TEST(AicpuReadyWatchdog, InvocationResetDropsPriorProgress) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(100);
    watchdog.record_progress(900);
    watchdog.reset();
    EXPECT_EQ(watchdog.last_progress(), 0u);
    watchdog.reset(1000);
    uint64_t now = 1099;
    EXPECT_FALSE(watchdog.wait_until_ready(
        [] {
            return false;
        },
        [&] {
            return now;
        },
        [&] {
            ++now;
        },
        100
    ));
    EXPECT_EQ(now, 1100u);
}

TEST(AicpuReadyWatchdog, SignalingBarrierKeepsFixedBudget) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(100);
    const uint64_t signaling_start = 100;
    watchdog.record_progress(199);
    EXPECT_TRUE(scheduler_watchdog_expired(signaling_start, 200, 100));
    EXPECT_FALSE(scheduler_watchdog_expired(watchdog.last_progress(), 200, 100));
}

TEST(AicpuReadyWatchdog, WrapAndDisabledBudgetMatchElapsedWatchdog) {
    AicpuReadyWatchdog watchdog;
    watchdog.reset(UINT64_MAX - 10);
    EXPECT_FALSE(watchdog.wait_until_ready(
        [] {
            return false;
        },
        [] {
            return 4u;
        },
        [] {
            ADD_FAILURE() << "wrapped deadline expired";
        },
        15
    ));
    bool ready = false;
    EXPECT_TRUE(watchdog.wait_until_ready(
        [&] {
            return ready;
        },
        [] {
            return UINT64_MAX;
        },
        [&] {
            ready = true;
        },
        0
    ));
}

using scheduler_test::SchedulerStateBuffer;
using GraphBuffer = scheduler_test::BasicGraphBuffer<16>;

TEST(SchedulerState, PlansAndInitializesReadyState) {
    // The reverse lookup for the wire constants AICore addresses with. They are
    // literals there because the AICore .o cannot see these types; here it can.
    EXPECT_EQ(sizeof(ChipTaskStorage), SCHEDULER_GRAPH_TASK_STORAGE_STRIDE);
    EXPECT_EQ(offsetof(ChipTaskStorage, task), SCHEDULER_GRAPH_DESCRIPTOR_OFFSET);
    EXPECT_EQ(offsetof(ChipTaskStorage, payload), SCHEDULER_GRAPH_PAYLOAD_OFFSET);
    EXPECT_EQ(offsetof(TaskPayload, predicate), SCHEDULER_GRAPH_PREDICATE_OFFSET);
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(5, 3, 2, &layout));
    EXPECT_EQ(layout.total_size % SCHEDULER_STATE_ALIGNMENT, 0u);
    EXPECT_EQ(layout.task_metadata_offset % alignof(SchedulerTaskMetadata), 0u);
    EXPECT_EQ(
        layout.worker_contexts_offset - layout.aicpu_lifecycle_traces_offset,
        static_cast<uint64_t>(PLATFORM_MAX_AICPU_THREADS) * sizeof(AicpuThreadLifecycleTrace)
    );

    std::vector<SchedulerTaskMetadata> metadata(layout.task_count);
    for (uint32_t task = 0; task < metadata.size(); ++task) {
        metadata[task].flags = SCHEDULER_TASK_EXECUTABLE;
        metadata[task].active_mask = task < 3 ? LaneMask::aic() : LaneMask::aiv0();
    }
    metadata[1].flags |= SCHEDULER_TASK_MIX;
    metadata[1].active_mask = LaneMask::all();
    metadata[2].flags |= SCHEDULER_TASK_MIX;
    metadata[2].active_mask = LaneMask::both_aiv();
    SchedulerReadyPoolPlan ready_plan{};
    ASSERT_TRUE(scheduler_plan_ready_pool(metadata, 1, layout.total_size, &ready_plan));
    EXPECT_GE(ready_plan.header_offset, layout.total_size);
    EXPECT_EQ(ready_plan.header_offset % SCHEDULER_STATE_ALIGNMENT, 0u);
    EXPECT_EQ(ready_plan.header.descriptors_offset % SCHEDULER_STATE_ALIGNMENT, 0u);
    EXPECT_EQ(ready_plan.header.controls_offset % SCHEDULER_STATE_ALIGNMENT, 0u);
    ASSERT_EQ(ready_plan.descriptors.size(), 2u);
    EXPECT_EQ(ready_plan.descriptors[0].capacity, 3u);
    EXPECT_EQ(ready_plan.descriptors[1].capacity, 2u);
    layout.total_size = ready_plan.total_size;

    SchedulerStateBuffer storage(layout);
    scheduler_init_ready_pool(storage.base(), ready_plan);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(storage.base(), layout.task_controls_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
        EXPECT_EQ(controls[task].wake_list_head, SCHEDULER_WAKE_LIST_OPEN);
    }
    const auto &pool = *scheduler_state_at<SchedulerReadyPoolHeader>(storage.base(), ready_plan.header_offset);
    EXPECT_EQ(pool.version, SCHEDULER_READY_POOL_VERSION);
    EXPECT_EQ(pool.entry_count, layout.task_count);
    EXPECT_TRUE(scheduler_ready_pool_validate(storage.base(), layout.total_size, ready_plan.header_offset, 5, 1));
    for (uint32_t type = 0; type < SCHEDULER_READY_QUEUE_COUNT; ++type) {
        const auto queue = scheduler_ready_pool_queue(storage.base(), pool, type);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(queue.control) % SCHEDULER_STATE_ALIGNMENT, 0u);
        EXPECT_EQ(reinterpret_cast<uintptr_t>(queue.entries) % SCHEDULER_STATE_ALIGNMENT, 0u);
        EXPECT_EQ(queue.control->head, 0u);
        EXPECT_EQ(queue.control->reserved_tail, 0u);
        for (uint64_t entry = 0; entry < queue.capacity; ++entry)
            EXPECT_EQ(queue.entries[entry].task_ref, SCHEDULER_TASK_REF_INVALID);
    }
}

TEST(SchedulerState, RejectsA64ByteOnlyAllocationBeforeWritingState) {
    AicoreSchedulerLayout layout{};
    ASSERT_TRUE(scheduler_plan_layout(1, 1, 0, &layout));
    scheduler_test::SchedulerStateBuffer storage(layout);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(storage.base(), layout.task_controls_offset);
    controls[0].state = 123;
    EXPECT_FALSE(
        scheduler_init_data_from_layout(static_cast<uint8_t *>(storage.base()) + 64, layout, UINT64_C(0x400000000))
    );
    EXPECT_FALSE(scheduler_init_data_from_layout(storage.base(), layout, UINT64_C(0x400000040)));
    EXPECT_EQ(controls[0].state, 123);
}

TEST(SchedulerState, PreservesCacheLineAlignmentAndArrayStride) {
    EXPECT_EQ(alignof(SchedulerTaskControl), 128u);
    EXPECT_EQ(alignof(SchedulerExecutorTaskTrace), alignof(uint64_t));
    EXPECT_EQ(sizeof(SchedulerExecutorTaskTrace), 48u);
    EXPECT_EQ(alignof(SchedulerRunControl), 64u);
    EXPECT_EQ(alignof(SchedulerWorkerContext), 64u);
    EXPECT_EQ(alignof(SchedulerTaskTrace), 64u);

    std::array<SchedulerTaskControl, 2> controls{};
    std::array<SchedulerWorkerContext, 2> contexts{};
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&controls[1]) - reinterpret_cast<uintptr_t>(&controls[0]), 640u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&contexts[1]) - reinterpret_cast<uintptr_t>(&contexts[0]), 320u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, state) / 64, offsetof(SchedulerTaskControl, wake_list_head) / 64);
    // Both chain links belong to the atomic line: they are written as single
    // device words, and nothing publishes that line as a whole, so no producer's
    // writeback can restore a stale copy of another's link.
    EXPECT_EQ(offsetof(SchedulerTaskControl, state) / 64, offsetof(SchedulerTaskControl, next_waiter) / 64);
    EXPECT_NE(
        offsetof(SchedulerTaskControl, state) / 64, offsetof(SchedulerTaskControl, completion_resolve_start_cycles) / 64
    );

    EXPECT_EQ(
        offsetof(SchedulerTaskTrace, dispatch_start_cycles) / 64, offsetof(SchedulerTaskTrace, complete_loop_iter) / 64
    );
    EXPECT_NE(
        offsetof(SchedulerTaskTrace, dispatch_start_cycles) / 64,
        offsetof(SchedulerTaskTrace, descriptor_cache_observed_cycles) / 64
    );
    EXPECT_NE(
        offsetof(SchedulerTaskTrace, refill_scheduler_worker_id) / 64,
        offsetof(SchedulerTaskTrace, descriptor_cache_observed_cycles) / 64
    );
}

TEST(SchedulerMetadata, ProjectsExistingSubmitTypesWithoutChangingTheirSemantics) {
    TaskAttrs attrs{};
    attrs.set_sync_start();
    attrs.set_predicate();
    attrs.set_timing_slot(7);
    const uint8_t flags = scheduler_task_metadata_flags_from_submit_state(
        LaneMask(/*aic=*/true, /*aiv0=*/true, /*aiv1=*/false), attrs, 2, true, false
    );
    EXPECT_TRUE(scheduler_task_is_executable(flags));
    EXPECT_TRUE(scheduler_task_has_fanin(flags));
    EXPECT_TRUE(scheduler_task_is_mix(flags));
    EXPECT_TRUE(scheduler_task_is_spmd(flags));
    EXPECT_TRUE(scheduler_task_requires_sync_start(flags));
    EXPECT_TRUE(scheduler_task_has_predicate(flags));
    EXPECT_FALSE(scheduler_task_is_inline(flags));
    EXPECT_EQ(attrs.timing_slot(), 7);

    const uint8_t inline_flags =
        scheduler_task_metadata_flags_from_submit_state(LaneMask::aiv0(), TaskAttrs{}, 1, false, true);
    EXPECT_EQ(inline_flags, SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_INLINE);
}

// AICore addresses a task's records by literal offsets, because its .o cannot see
// the types. The reverse lookup in PlansAndInitializesReadyState only proves the
// constants carry the right values; it cannot catch a helper that strides by the
// wrong one. That failure is invisible at task 0 — every candidate stride agrees
// there — so this walks several tasks and checks both records of each against the
// entry they belong to.
TEST(SchedulerGraph, AddressesEveryTaskAtTheStorageStride) {
    constexpr size_t kTasks = 4;
    struct alignas(64) GraphImage {
        std::array<ChipTaskStorage, kTasks> storage{};
    } image;
    auto &storage = image.storage;
    const auto graph = scheduler_ordinary_graph_view(reinterpret_cast<uint64_t>(storage.data()), kTasks, kTasks - 1);

    for (size_t task = 0; task < kTasks; ++task) {
        SCOPED_TRACE(testing::Message() << "task " << task);
        EXPECT_EQ(
            scheduler_graph_descriptor(graph, static_cast<int64_t>(task)),
            reinterpret_cast<uint8_t *>(&storage[task].task)
        );
        EXPECT_EQ(
            scheduler_graph_payload(graph, static_cast<int64_t>(task)),
            reinterpret_cast<uint8_t *>(&storage[task].payload)
        );
    }

    // Consecutive tasks are one whole storage apart, not one record apart. The
    // pre-merge strides (64 for a descriptor, 192 for a payload) satisfy the loop
    // above at task 0 and fail it everywhere else; this states the step directly.
    const auto *first = scheduler_graph_descriptor(graph, 0);
    const auto *second = scheduler_graph_descriptor(graph, 1);
    EXPECT_EQ(second - first, SCHEDULER_GRAPH_TASK_STORAGE_STRIDE);
    EXPECT_EQ(scheduler_graph_payload(graph, 0) - scheduler_graph_descriptor(graph, 0), SCHEDULER_GRAPH_PAYLOAD_OFFSET);
}

TEST(SchedulerGraph, NonPowerOfTwoWindowKeepsDirectTaskIdIndexing) {
    struct alignas(64) GraphImage {
        std::array<ChipTaskStorage, 3> storage{};
        std::array<std::array<int32_t, SCHEDULER_GRAPH_MAX_FANIN>, 3> fanins{};
    } image;
    auto &storage = image.storage;
    auto &fanins = image.fanins;
    for (size_t task = 0; task < storage.size(); ++task) {
        storage[task].task.task_id = TaskId::make_global(static_cast<int32_t>(task));
        for (int slot = 0; slot < 3; ++slot)
            storage[task].task.kernel_id[slot] = INVALID_KERNEL_ID;
        storage[task].payload.bind_regions(nullptr, nullptr, fanins[task].data());
    }
    storage[1].task.kernel_id[0] = 1;
    const auto graph = scheduler_ordinary_graph_view(reinterpret_cast<uint64_t>(storage.data()), 3, 2);

    // Both records of task 1 resolve from the one base, at that entry's stride.
    EXPECT_EQ(scheduler_graph_descriptor(graph, 1), reinterpret_cast<uint8_t *>(&storage[1].task));
    EXPECT_EQ(scheduler_graph_payload(graph, 1), reinterpret_cast<uint8_t *>(&storage[1].payload));
    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(graph, 1, &shape), SchedulerGraphResult::OK);
    EXPECT_EQ(shape.task_ref, 1);
}

TEST(SchedulerGraph, ValidatesNonEmptyFanins) {
    GraphBuffer graph(3);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.executable(2, 0, {1, 1});

    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(graph.graph(), 1, &shape), SchedulerGraphResult::OK);
    EXPECT_EQ(scheduler_classify_task_shape(graph.graph(), 2, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
}

TEST(SchedulerGraph, RejectsUnboundNonEmptyFaninRegion) {
    GraphBuffer graph(2);
    graph.executable(0, 0);
    graph.executable(1, 0, {0});
    graph.payload(1).bind_regions(nullptr, nullptr, nullptr);

    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(graph.graph(), 1, &shape), SchedulerGraphResult::INVALID_FANIN_ID);
}

TEST(SchedulerGraph, RejectsAViewWithAnUnsupportedVersion) {
    GraphBuffer graph(1);
    graph.executable(0, 0);

    SchedulerGraphView dirty = graph.graph();
    dirty.view_version = SCHEDULER_GRAPH_VIEW_VERSION + 1;

    SchedulerTaskShape shape{};
    EXPECT_EQ(scheduler_classify_task_shape(dirty, 0, &shape), SchedulerGraphResult::INVALID_ARGUMENTS);

    DispatchPayload payload{};
    SchedulerTaskInfo task{0, 1, 0, CoreType::AIC};
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(dirty, task, 0x1000, &payload),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );
}

TEST(SchedulerDispatchPayload, PreservesInitializedSlotContextDuringMaterialization) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    DispatchPayload payload{};
    payload.global_context.sub_block_id = 7;
    payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.local_context);
    payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.global_context);
    SchedulerTaskInfo task{0, 1, 0, CoreType::AIC};

    ASSERT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload), SchedulerGraphResult::OK
    );
    EXPECT_EQ(payload.global_context.sub_block_id, 7);
    EXPECT_EQ(payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX], reinterpret_cast<uint64_t>(&payload.local_context));
    EXPECT_EQ(payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX], reinterpret_cast<uint64_t>(&payload.global_context));

    task.subtask_slot = 2;
    task.core_type = CoreType::AIV;
    ASSERT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload), SchedulerGraphResult::OK
    );
    EXPECT_EQ(payload.global_context.sub_block_id, 7);
}

TEST(SchedulerDispatchPayload, RejectsInvalidGraphBoundsBeforeReadingPayload) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    DispatchPayload payload{};

    SchedulerTaskInfo out_of_range{1, 1, 0, CoreType::AIC};
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), out_of_range, 0x1000, &payload),
        SchedulerGraphResult::INVALID_TASK_COUNT
    );

    SchedulerGraphView missing_storage = graph.graph();
    missing_storage.storage_address = 0;
    SchedulerTaskInfo valid_task{0, 1, 0, CoreType::AIC};
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(missing_storage, valid_task, 0x1000, &payload),
        SchedulerGraphResult::INVALID_TASK_COUNT
    );
}

TEST(SchedulerDispatchPayload, RejectsUnboundNonEmptyArgumentRegions) {
    GraphBuffer graph(1);
    graph.executable(0, 0);
    DispatchPayload payload{};
    SchedulerTaskInfo task{0, 1, 0, CoreType::AIC};

    graph.payload(0).tensor_count = 1;
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );

    graph.payload(0).tensor_count = 0;
    graph.payload(0).scalar_count = 1;
    EXPECT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload),
        SchedulerGraphResult::INVALID_ARGUMENTS
    );
}

TEST(SchedulerClusterTopology, UsesMixedKernelLaunchCoordinate) {
    constexpr int32_t kClusters = 4;
    SchedulerClusterCoordinate coordinate{};
    ASSERT_TRUE(scheduler_cluster_coordinate_from_worker(2, true, kClusters, 2, &coordinate));
    EXPECT_EQ(coordinate.cluster_index, 2);
    EXPECT_EQ(coordinate.cluster_lane, 0);
    ASSERT_TRUE(scheduler_cluster_coordinate_from_worker(9, false, kClusters, 2, &coordinate));
    EXPECT_EQ(coordinate.cluster_index, 2);
    EXPECT_EQ(coordinate.cluster_lane, 2);
    EXPECT_FALSE(scheduler_cluster_coordinate_from_worker(12, false, kClusters, 2, &coordinate));
}

}  // namespace
