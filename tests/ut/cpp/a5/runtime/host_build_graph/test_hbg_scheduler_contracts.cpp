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
#include "scheduler/scheduler_ssbuf.h"
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

TEST(AicoreSchedulerState, DistinguishesResidentAndExplicitLegacyModes) {
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_resident(0));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_RESIDENT_READY));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_resident(SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE));

    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_explicit_legacy(0));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_explicit_legacy(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING));
    EXPECT_FALSE(aicore_scheduler_runtime_mode_is_explicit_legacy(SCHEDULER_RUNTIME_MODE_RESIDENT_READY));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_explicit_legacy(SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH));
    EXPECT_TRUE(aicore_scheduler_runtime_mode_is_explicit_legacy(SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE));
}

TEST(AicoreSchedulerState, ResidentAcceptsSingleBlockOrdinaryMix) {
    EXPECT_TRUE(scheduler_resident_v0_task_shape_supported(1, 1, false));
    EXPECT_TRUE(scheduler_resident_v0_task_shape_supported(2, 1, false));
    EXPECT_FALSE(scheduler_resident_v0_task_shape_supported(1, 2, false));
    EXPECT_FALSE(scheduler_resident_v0_task_shape_supported(1, 1, true));
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
    EXPECT_EQ(layout.ready_inboxes_offset % alignof(SchedulerReadyInbox), 0u);
    EXPECT_EQ(layout.ready_directory_offset % alignof(SchedulerReadyDirectory), 0u);
    EXPECT_EQ(
        layout.worker_contexts_offset - layout.aicpu_lifecycle_traces_offset,
        static_cast<uint64_t>(PLATFORM_MAX_AICPU_THREADS) * sizeof(AicpuThreadLifecycleTrace)
    );

    SchedulerStateBuffer storage(layout);
    auto *controls = scheduler_state_at<SchedulerTaskControl>(storage.base(), layout.task_controls_offset);
    for (uint64_t task = 0; task < layout.task_count; ++task) {
        EXPECT_EQ(controls[task].state, static_cast<int64_t>(SchedulerTaskState::BLOCKED));
        EXPECT_EQ(controls[task].wake_list_head, SCHEDULER_WAKE_LIST_OPEN);
    }
    auto *ready = scheduler_state_at<SchedulerReadyInbox>(storage.base(), layout.ready_inboxes_offset);
    for (uint64_t inbox = 0; inbox < SCHEDULER_READY_INBOX_CAPACITY; ++inbox)
        EXPECT_EQ(ready[inbox].endpoints, SCHEDULER_READY_ENDPOINTS_EMPTY);
    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(storage.base(), layout.worker_contexts_offset);
    EXPECT_EQ(contexts[0].scheduler_ssbuf_reserved0, 0u);
    EXPECT_EQ(contexts[0].scheduler_ssbuf_reserved1, 0u);
    EXPECT_EQ(contexts[0].scheduler_ssbuf_reserved2, 0u);

    auto *directory = scheduler_state_at<SchedulerReadyDirectory>(storage.base(), layout.ready_directory_offset);
    auto shard0 = reinterpret_cast<uintptr_t>(&directory->core_types[0][0]);
    auto shard1 = reinterpret_cast<uintptr_t>(&directory->core_types[0][1]);
    auto aiv_shard0 = reinterpret_cast<uintptr_t>(&directory->core_types[1][0]);
    EXPECT_EQ(shard1 - shard0, 64u);
    EXPECT_EQ(aiv_shard0 - shard0, SCHEDULER_READY_DIRECTORY_SHARD_COUNT * 64u);
}

TEST(SchedulerState, PreservesCacheLineAlignmentAndArrayStride) {
    EXPECT_EQ(alignof(SchedulerTaskControl), 128u);
    EXPECT_EQ(alignof(SchedulerExecutorTaskTrace), alignof(uint64_t));
    EXPECT_EQ(sizeof(SchedulerExecutorTaskTrace), 48u);
    EXPECT_EQ(alignof(SchedulerSsbufRegion), 64u);
    EXPECT_EQ(alignof(SchedulerRunControl), 128u);
    EXPECT_EQ(alignof(SchedulerWorkerContext), 128u);
    EXPECT_EQ(alignof(SchedulerTaskTrace), 128u);

    std::array<SchedulerTaskControl, 2> controls{};
    std::array<SchedulerWorkerContext, 2> contexts{};
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&controls[1]) - reinterpret_cast<uintptr_t>(&controls[0]), 128u);
    EXPECT_EQ(reinterpret_cast<uintptr_t>(&contexts[1]) - reinterpret_cast<uintptr_t>(&contexts[0]), 1024u);
    EXPECT_EQ(offsetof(SchedulerTaskControl, state) / 64, offsetof(SchedulerTaskControl, wake_list_head) / 64);
    // Both chain links belong to the atomic line: they are written as single
    // device words, and nothing publishes that line as a whole, so no producer's
    // writeback can restore a stale copy of another's link.
    EXPECT_EQ(offsetof(SchedulerTaskControl, state) / 64, offsetof(SchedulerTaskControl, next_waiter) / 64);
    EXPECT_NE(offsetof(SchedulerTaskControl, state) / 64, offsetof(SchedulerTaskControl, next_fanin_index) / 64);
    EXPECT_EQ(offsetof(SchedulerSsbufLane, mix_completion), 0x00u);
    EXPECT_EQ(sizeof(SchedulerSsbufLane), 0x40u);
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

TEST(SchedulerSsbuf, InitializesDirtyMailboxWithoutTouchingUserStorage) {
    alignas(64) std::array<uint8_t, SCHEDULER_SSBUF_HARDWARE_SIZE> hardware{};
    hardware.fill(0xa5);
    auto *region = scheduler_ssbuf_region(reinterpret_cast<uint64_t>(hardware.data()));
    scheduler_ssbuf_initialize(region);

    EXPECT_EQ(reinterpret_cast<uint8_t *>(region) - hardware.data(), SCHEDULER_SSBUF_REGION_OFFSET);
    EXPECT_TRUE(scheduler_ssbuf_is_initialized(region));
    for (uint32_t lane = 0; lane < PLATFORM_CORES_PER_BLOCKDIM; ++lane) {
        for (uint32_t slot = 0; slot < SCHEDULER_MIX_SLOT_COUNT; ++slot) {
            EXPECT_EQ(scheduler_ssbuf_load_relaxed(&region->mix_dispatch[lane][slot].publication), 0u);
        }
        EXPECT_EQ(scheduler_ssbuf_load_relaxed(&region->lanes[lane].mix_completion.publication), 0u);
    }
    EXPECT_EQ(hardware[SCHEDULER_SSBUF_REGION_OFFSET - 1], 0xa5);
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
    SchedulerGraphView graph{reinterpret_cast<uint64_t>(storage.data()), 0, kTasks, kTasks - 1};

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
    SchedulerGraphView graph{reinterpret_cast<uint64_t>(storage.data()), 0, 3, 2};

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

// The view's reserved word exists so its wire size survived the collapse from two
// addresses to one. A non-zero read means the producer is writing a field this
// build does not know about, so both entry points refuse rather than proceed on a
// view they cannot fully interpret.
TEST(SchedulerGraph, RejectsAViewWithANonZeroReservedWord) {
    GraphBuffer graph(1);
    graph.executable(0, 0);

    SchedulerGraphView dirty = graph.graph();
    dirty.reserved = 1;

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
    payload.local_context.async_ctx.task_token = TaskId::invalid();
    payload.global_context.sub_block_id = 7;
    payload.args[PAYLOAD_LOCAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.local_context);
    payload.args[PAYLOAD_GLOBAL_CONTEXT_INDEX] = reinterpret_cast<uint64_t>(&payload.global_context);
    SchedulerTaskInfo task{0, 1, 0, CoreType::AIC};

    ASSERT_EQ(
        scheduler_materialize_task_payload_resolved(graph.graph(), task, 0x1000, &payload), SchedulerGraphResult::OK
    );
    EXPECT_FALSE(payload.local_context.async_ctx.task_token.is_valid());
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
