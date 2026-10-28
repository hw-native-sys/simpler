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
#include <atomic>
#include <thread>
#include <vector>

#include "aicore_scheduler_error.h"
#include "hbg_scheduler_test_support.h"
#include "scheduler/scheduler_completion.h"
#include "scheduler/scheduler_gm_mix.h"

namespace {

struct ErrorCacheEvent {
    const void *address;
    size_t bytes;
    bool flush;
    uint64_t status;
};

struct ErrorCacheTrace;
thread_local ErrorCacheTrace *active_cache_trace = nullptr;

struct ErrorCacheTrace {
    explicit ErrorCacheTrace(SchedulerRunControl *value) :
        run(value) {
        active_cache_trace = this;
    }
    ~ErrorCacheTrace() { active_cache_trace = nullptr; }
    SchedulerRunControl *run;
    std::vector<ErrorCacheEvent> events;
};

void record_cache_access(const void *address, size_t bytes, bool flush) {
    if (active_cache_trace == nullptr) return;
    active_cache_trace->events.push_back(
        {address, bytes, flush, __atomic_load_n(&active_cache_trace->run->scheduler_error, __ATOMIC_ACQUIRE)}
    );
}

}  // namespace

// Host-only cache-operation recording; device visibility has separate native probes.
namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *address, size_t bytes) { record_cache_access(address, bytes, false); }
void flush_range_impl(const void *address, size_t bytes) { record_cache_access(address, bytes, true); }
}  // namespace aicpu_cache_maintenance

namespace {

using Identity = std::array<uint64_t, 6>;
constexpr Identity kUnknown{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX};
constexpr size_t kIdentityOffset = offsetof(SchedulerRunControl, error_site) + sizeof(uint64_t);

Identity read_identity(const SchedulerRunControl &run) {
    const auto *words =
        reinterpret_cast<const volatile uint64_t *>(reinterpret_cast<const uint8_t *>(&run) + kIdentityOffset);
    Identity result{};
    for (size_t index = 0; index < result.size() - 1; ++index)
        result[index] = __atomic_load_n(words + index, __ATOMIC_RELAXED);
    const uint32_t lane = __atomic_load_n(&run.error_cluster_lane, __ATOMIC_RELAXED);
    result[5] = lane == UINT32_MAX ? UINT64_MAX : lane;
    return result;
}

struct ErrorFixture {
    ErrorFixture() :
        storage(2, 3, 8, SchedulerSpmdRequirements{3, 1, 2, 2}) {
        for (uint32_t task = 0; task < 2; ++task) {
            graph.mixed(task, 3);
            auto &metadata = storage.metadata[task];
            metadata.active_mask = 3;
            metadata.logical_block_num = 4;
            metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD | SCHEDULER_TASK_MIX;
            metadata.kernel_ids[0] = 1;
            metadata.kernel_ids[1] = 1;
            metadata.kernel_ids[2] = UINT16_MAX;
        }
        EXPECT_TRUE(scheduler_init_spmd_task_records(base(), storage.layout, 61));
        for (uint32_t worker = 0; worker < 3; ++worker) {
            auto &context = storage.contexts[worker];
            context.is_scheduler = 1;
            context.scheduler_count = 3;
            context.physical_core_id = static_cast<int32_t>(100 + worker);
            for (uint32_t lane = 0; lane < 3; ++lane)
                context.cluster_worker_ids[lane] = lane;
            storage.test_graphs[worker] = graph.graph();
            locals[worker] = storage.local_context(&context);
            EXPECT_TRUE(scheduler_initialize_admission(base(), locals[worker], &admissions[worker]));
            locals[worker]->admission = &admissions[worker];
            auto &admission = admissions[worker];
            admission.current_task_ref = 0;
            admission.current_block_num = 4;
            admission.current_active_mask = 3;
            admission.current_flags = storage.metadata[0].flags;
            admission.scan_cursor = 0;
            admission.gm_mix.phase = SchedulerGmMixPhase::RUNNING;
            admission.gm_mix.block_idx = 2;
            admission.gm_mix.group_seq = 7;
            admission.gm_mix.task_last_seq = 7;
            admission.gm_mix.subtask_slot = static_cast<uint8_t>(worker);
        }
    }
    void *base() const { return storage.scheduler_state->base(); }
    SchedulerRunControl *run() const { return storage.run_control; }
    void record(uint32_t worker = 1, int64_t task = 0) {
        const auto view = graph.graph();
        scheduler_record_error(
            run(), task, SchedulerGraphResult::INVALID_ARGUMENTS, &view, locals[worker],
            SchedulerErrorSite::GROUP_INVALID_STATE
        );
    }
    scheduler_test::FixtureStorage storage;
    scheduler_test::BasicGraphBuffer<2> graph{2};
    std::array<SchedulerLocalState *, 3> locals{};
    std::array<SchedulerAdmissionLocalState, 3> admissions{};
};

TEST(HbgErrorIdentity, DedicatedStatusLinePreservesDiagnosticIdentityOffsets) {
    EXPECT_EQ(sizeof(SchedulerRunControl), 512u);
    EXPECT_EQ(offsetof(SchedulerRunControl, error_claimed), 256u);
    EXPECT_EQ(offsetof(SchedulerRunControl, error_site), 328u);
    EXPECT_EQ(kIdentityOffset, 336u);
    EXPECT_EQ(kIdentityOffset + sizeof(Identity), offsetof(SchedulerRunControl, scheduler_error));
}

TEST(HbgErrorIdentity, PublishedStatusHasNoAtomicWriterOnItsCacheLine) {
    constexpr size_t status_offset = offsetof(SchedulerRunControl, scheduler_error);
    EXPECT_EQ(status_offset % 128, 0u);
    EXPECT_NE(status_offset / 64, offsetof(SchedulerRunControl, error_claimed) / 64);
    EXPECT_NE(status_offset / 64, offsetof(SchedulerRunControl, error_task_ref) / 64);
    EXPECT_NE(status_offset / 64, offsetof(SchedulerRunControl, error_identity_fields) / 64);
    EXPECT_EQ(status_offset, 384u);
}

TEST(HbgErrorIdentity, AicpuFlushesIdentityBeforePublishingAndFlushingStatus) {
    ErrorFixture fixture;
    ErrorCacheTrace trace(fixture.run());
    ASSERT_TRUE(record_aicore_scheduler_runtime_error(
        fixture.run(), SchedulerGraphResult::TIMEOUT, SchedulerErrorSite::EXECUTION_PROGRESS_TIMEOUT
    ));
    ASSERT_EQ(trace.events.size(), 2u);
    const auto *base = reinterpret_cast<const uint8_t *>(fixture.run());
    EXPECT_EQ(trace.events[0].address, base + offsetof(SchedulerRunControl, error_claimed));
    EXPECT_EQ(trace.events[0].bytes, 128u);
    EXPECT_TRUE(trace.events[0].flush);
    EXPECT_EQ(trace.events[0].status, 0u);
    EXPECT_EQ(trace.events[1].address, base + offsetof(SchedulerRunControl, scheduler_error));
    EXPECT_EQ(trace.events[1].bytes, 128u);
    EXPECT_TRUE(trace.events[1].flush);
    EXPECT_EQ(trace.events[1].status, static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
    EXPECT_FALSE(record_aicore_scheduler_runtime_error(
        fixture.run(), SchedulerGraphResult::INVALID_ARGUMENTS, SchedulerErrorSite::GROUP_INVALID_STATE
    ));
    EXPECT_EQ(trace.events.size(), 2u);
}

TEST(HbgErrorIdentity, ColdLogObservesTheSeparateStatusLineBeforeReloadingIdentity) {
    ErrorFixture fixture;
    fixture.record();
    ErrorCacheTrace trace(fixture.run());
    testing::internal::CaptureStderr();
    log_aicore_scheduler_error_identity(fixture.run(), "terminal");
    const auto message = testing::internal::GetCapturedStderr();
    EXPECT_FALSE(message.empty());
    ASSERT_EQ(trace.events.size(), 2u);
    const auto *base = reinterpret_cast<const uint8_t *>(fixture.run());
    EXPECT_EQ(trace.events[0].address, base + offsetof(SchedulerRunControl, error_claimed));
    EXPECT_EQ(trace.events[0].bytes, 256u);
    EXPECT_FALSE(trace.events[0].flush);
    EXPECT_EQ(trace.events[1].address, base + offsetof(SchedulerRunControl, error_claimed));
    EXPECT_EQ(trace.events[1].bytes, 128u);
    EXPECT_FALSE(trace.events[1].flush);
}

TEST(HbgErrorIdentity, InitializationAndRetainedResetUseUnknownInsteadOfValidZeroIdentities) {
    ErrorFixture fixture;
    EXPECT_EQ(read_identity(*fixture.run()), kUnknown);
    fixture.record();
    ASSERT_TRUE(scheduler_init_data_from_layout(
        fixture.base(), fixture.storage.layout, reinterpret_cast<uint64_t>(fixture.base())
    ));
    EXPECT_EQ(read_identity(*fixture.run()), kUnknown);
    EXPECT_EQ(fixture.run()->error_claimed, 0u);
    EXPECT_EQ(fixture.run()->scheduler_error, 0u);
}

TEST(HbgErrorIdentity, EpochZeroIsKnownWhenTheAdmissionContextWasInitialized) {
    ErrorFixture fixture;
    fixture.admissions[1].epoch = 0;
    fixture.record();
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{0, 0, 2, 7, 1, 1}));
    EXPECT_EQ(fixture.run()->error_core_id, 101u);
}

uint32_t read_valid_fields(const SchedulerRunControl &run) {
    const auto *word = reinterpret_cast<const volatile uint32_t *>(
        reinterpret_cast<const uint8_t *>(&run) + offsetof(SchedulerRunControl, error_identity_fields)
    );
    return __atomic_load_n(word, __ATOMIC_RELAXED);
}

TEST(HbgErrorIdentity, MaximumEpochHasAnExplicitValidityBitWhileUnknownDoesNot) {
    ErrorFixture fixture;
    EXPECT_EQ(read_valid_fields(*fixture.run()), 0u);
    fixture.admissions[1].epoch = UINT64_MAX;
    fixture.record();
    EXPECT_EQ(read_identity(*fixture.run())[0], UINT64_MAX);
    EXPECT_EQ(read_valid_fields(*fixture.run()), 63u);
}

TEST(HbgErrorIdentity, SharedContextFailureRetainsKnownWorkerAndLaneWithoutGuessingEpoch) {
    ErrorFixture fixture;
    const auto graph = fixture.graph.graph();
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::INVALID_ARGUMENTS, &graph, &fixture.storage.contexts[1],
        SchedulerErrorSite::BOOTSTRAP_FAILED
    );
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, 1, 1}));
}

TEST(HbgErrorIdentity, PurePreparationFailureNamesTheExactFillingBlockAmongTwoOwnedSlots) {
    ErrorFixture fixture;
    auto *local = fixture.locals[1];
    auto &admission = fixture.admissions[1];
    auto &metadata = fixture.storage.metadata[0];
    metadata.active_mask = 2;
    metadata.flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD;
    metadata.kernel_ids[0] = UINT16_MAX;
    admission.current_active_mask = metadata.active_mask;
    admission.current_flags = metadata.flags;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        local->slots[slot].task_ref = 0;
        local->slots[slot].cooperative = true;
        local->slots[slot].block_idx = slot + 1;
        local->slots[slot].state = slot == 0 ? SchedulerDispatchSlotState::READY : SchedulerDispatchSlotState::FILLING;
    }
    fixture.storage.callable_addresses[1] = 0;
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    EXPECT_FALSE(scheduler_prepare_dispatch_slot(
        fixture.graph.graph(), fixture.base(), local, fixture.run(), {1, 1, 0, 1}, ready, metadata, 0, nullptr, 1
    ));
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, 0, 2, UINT64_MAX, 1, 1}));
}

class HbgErrorGroupPhase : public ::testing::TestWithParam<SchedulerGmMixPhase> {};

TEST_P(HbgErrorGroupPhase, PreservesOwnedGroupIdentityIncludingInactiveCoordinatorRelease) {
    for (uint32_t worker : {0u, 1u, 2u}) {
        ErrorFixture fixture;
        auto &admission = fixture.admissions[worker];
        admission.current_active_mask = GetParam() == SchedulerGmMixPhase::RUNNING ? 7 : 6;
        admission.gm_mix.phase = GetParam();
        admission.gm_mix.subtask_slot = worker == 0 && admission.current_active_mask == 6 ? UINT8_MAX : worker;
        fixture.record(worker);
        EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, 0, 2, 7, worker, worker}));
    }
}

INSTANTIATE_TEST_SUITE_P(
    OwnedPhases, HbgErrorGroupPhase,
    ::testing::Values(
        SchedulerGmMixPhase::CLAIMED, SchedulerGmMixPhase::PUBLISHED, SchedulerGmMixPhase::ARRIVED,
        SchedulerGmMixPhase::RUNNING, SchedulerGmMixPhase::RELEASED
    )
);

TEST(HbgErrorIdentity, IdleGroupDoesNotAttributeThePreviousTasksBlockOrSequenceToTheNextTask) {
    ErrorFixture fixture;
    auto &admission = fixture.admissions[1];
    admission.current_task_ref = 1;
    admission.scan_cursor = 1;
    admission.gm_mix.phase = SchedulerGmMixPhase::IDLE;
    admission.gm_mix.task_last_seq = 0;
    fixture.record(1, 1);
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, 1, UINT64_MAX, UINT64_MAX, 1, 1}));
}

TEST(HbgErrorIdentity, UninitializedAdmissionDoesNotInventEpochOrderOrGroupIdentity) {
    ErrorFixture fixture;
    fixture.admissions[1].initialized = false;
    fixture.record();
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{UINT64_MAX, UINT64_MAX, UINT64_MAX, UINT64_MAX, 1, 1}));
}

TEST(HbgErrorIdentity, AFailureForAnotherTaskDoesNotInheritTheCurrentObligationsOrderOrGroup) {
    ErrorFixture fixture;
    fixture.record(1, 1);
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, UINT64_MAX, UINT64_MAX, UINT64_MAX, 1, 1}));
}

TEST(HbgErrorIdentity, AdmissionFailureKeepsLocalIdentityRatherThanOnlyTheSharedWorkerContext) {
    ErrorFixture fixture;
    EXPECT_FALSE(scheduler_admission_fail(
        fixture.graph.graph(), fixture.locals[1], fixture.run(), 0, SchedulerErrorSite::GROUP_ACK_INVALID
    ));
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, 0, 2, 7, 1, 1}));
}

TEST(HbgErrorIdentity, ActualMixPreparationFailureRetainsTheClaimedBlock) {
    ErrorFixture fixture;
    auto &admission = fixture.admissions[1];
    admission.gm_mix.phase = SchedulerGmMixPhase::ARRIVED;
    auto *mailbox = scheduler_state_at<SchedulerGroupMailbox>(fixture.base(), admission.gm_mix.mailbox_offset);
    for (auto &lane : mailbox->lanes)
        scheduler_gm_store(lane.arrived_seq, UINT64_C(7));
    fixture.storage.callable_addresses[1] = 0;
    bool progress = false;
    EXPECT_FALSE(scheduler_gm_mix_prepare_lane(
        fixture.graph.graph(), fixture.base(), fixture.locals[1], fixture.run(), 0, &progress
    ));
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::DISPATCH_INVALID_CALLABLE));
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, 0, 2, 7, 1, 1}));
    EXPECT_EQ(fixture.locals[1]->pending_completed, 0u);
}

TEST(HbgErrorIdentity, AicoreFirstWriterSurvivesALaterAicpuTimeoutAndAnotherWorker) {
    ErrorFixture fixture;
    fixture.record();
    const auto first = read_identity(*fixture.run());
    EXPECT_FALSE(record_aicore_scheduler_runtime_error(
        fixture.run(), SchedulerGraphResult::TIMEOUT, SchedulerErrorSite::AICPU_SHUTDOWN_BARRIER_TIMEOUT
    ));
    fixture.record(2, 1);
    EXPECT_EQ(read_identity(*fixture.run()), first);
    EXPECT_EQ(first, (Identity{61, 0, 2, 7, 1, 1}));
    EXPECT_EQ(fixture.run()->error_task_ref, 0u);
}

TEST(HbgErrorIdentity, AicpuFirstWriterPublishesUnknownRatherThanStaleAicoreIdentity) {
    ErrorFixture fixture;
    fixture.run()->error_task_ref = 8;
    fixture.run()->error_core_id = 101;
    fixture.run()->error_core_type = 1;
    auto *words = reinterpret_cast<uint64_t *>(reinterpret_cast<uint8_t *>(fixture.run()) + kIdentityOffset);
    for (uint32_t index = 0; index < 6; ++index)
        words[index] = 900 + index;
    ASSERT_TRUE(record_aicore_scheduler_runtime_error(
        fixture.run(), SchedulerGraphResult::TIMEOUT, SchedulerErrorSite::AICPU_INIT_BARRIER_TIMEOUT
    ));
    fixture.record();
    EXPECT_EQ(read_identity(*fixture.run()), kUnknown);
    EXPECT_EQ(fixture.run()->error_task_ref, UINT64_MAX);
    EXPECT_EQ(fixture.run()->error_core_id, UINT64_MAX);
    EXPECT_EQ(fixture.run()->error_core_type, UINT64_MAX);
    EXPECT_EQ(read_valid_fields(*fixture.run()), 0u);
    EXPECT_EQ(fixture.run()->error_site, static_cast<uint64_t>(SchedulerErrorSite::AICPU_INIT_BARRIER_TIMEOUT));
    EXPECT_EQ(scheduler_read_error(fixture.run()), static_cast<uint64_t>(SchedulerGraphResult::TIMEOUT));
}

TEST(HbgErrorIdentity, OrdinarySingleBlockPreparationHasKnownBlockWithoutClaimingAnAdmissionOrder) {
    ErrorFixture fixture;
    auto *local = fixture.locals[1];
    local->admission = nullptr;
    local->slots[0].state = SchedulerDispatchSlotState::FILLING;
    auto &metadata = fixture.storage.metadata[0];
    metadata.active_mask = 2;
    metadata.logical_block_num = 1;
    metadata.flags = SCHEDULER_TASK_EXECUTABLE;
    metadata.kernel_ids[0] = UINT16_MAX;
    fixture.storage.callable_addresses[1] = 0;
    SchedulerReadyClaim ready{};
    ready.task_ref = 0;
    EXPECT_FALSE(scheduler_prepare_dispatch_slot(
        fixture.graph.graph(), fixture.base(), local, fixture.run(), {1, 0, 0, 1}, ready, metadata, 0, nullptr, 1
    ));
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{UINT64_MAX, UINT64_MAX, 0, UINT64_MAX, 1, 1}));
    EXPECT_EQ(read_valid_fields(*fixture.run()), 52u);
}

TEST(HbgErrorIdentity, PureGenerationMismatchNamesTheCompletedSlotWithoutGuessingFromTheOtherSlot) {
    ErrorFixture fixture;
    auto *local = fixture.locals[1];
    fixture.admissions[1].current_flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD;
    for (uint32_t slot = 0; slot < 2; ++slot) {
        local->slots[slot].task_ref = 0;
        local->slots[slot].block_idx = slot + 1;
        local->slots[slot].cooperative = true;
        local->slots[slot].state = SchedulerDispatchSlotState::READY;
        local->slots[slot].generation = 3;
    }
    EXPECT_FALSE(scheduler_service_cluster_completion_slot(
        fixture.graph.graph(), fixture.base(), local, fixture.run(), 1, 1, 4, nullptr, nullptr, nullptr, 0, nullptr,
        nullptr, nullptr
    ));
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, 0, 2, UINT64_MAX, 1, 1}));
    EXPECT_EQ(read_valid_fields(*fixture.run()), 55u);
    EXPECT_EQ(local->pending_completed, 0u);
}

TEST(HbgErrorIdentity, PureCompletionOverflowRetainsItsBlockAndDoesNotCountSuccess) {
    ErrorFixture fixture;
    auto *local = fixture.locals[1];
    fixture.admissions[1].current_flags = SCHEDULER_TASK_EXECUTABLE | SCHEDULER_TASK_SPMD;
    auto &slot = local->slots[1];
    slot.task_ref = 0;
    slot.block_idx = 3;
    slot.cooperative = true;
    slot.state = SchedulerDispatchSlotState::READY;
    auto *controls =
        scheduler_state_at<SchedulerTaskControl>(fixture.base(), fixture.storage.layout.task_controls_offset);
    scheduler_gm_store(controls[0].completed_blocks, UINT64_C(4));
    EXPECT_EQ(
        scheduler_finish_ordered_block(fixture.graph.graph(), fixture.base(), local, fixture.run(), 1),
        SchedulerBlockFinishResult::ERROR
    );
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, 0, 3, UINT64_MAX, 1, 1}));
    EXPECT_EQ(local->pending_completed, 0u);
}

TEST(HbgErrorIdentity, RetiredGroupKeepsItsLastIdentityOnlyWhileItBelongsToTheCurrentTask) {
    ErrorFixture fixture;
    fixture.admissions[1].gm_mix.phase = SchedulerGmMixPhase::IDLE;
    fixture.record();
    EXPECT_EQ(read_identity(*fixture.run()), (Identity{61, 0, 2, 7, 1, 1}));
}

TEST(HbgErrorIdentity, MissingContextCannotInventWorkerAndNullDestinationDoesNotCrash) {
    ErrorFixture fixture;
    scheduler_record_error(nullptr, 0, SchedulerGraphResult::INVALID_ARGUMENTS);
    const auto view = fixture.graph.graph();
    scheduler_record_error(
        fixture.run(), 0, SchedulerGraphResult::INVALID_ARGUMENTS, &view,
        static_cast<const SchedulerLocalState *>(nullptr)
    );
    EXPECT_EQ(read_identity(*fixture.run()), kUnknown);
    EXPECT_EQ(read_valid_fields(*fixture.run()), 0u);
}

TEST(HbgErrorIdentity, ACleanStatusDoesNotEmitAnErrorIdentityLog) {
    ErrorFixture fixture;
    testing::internal::CaptureStderr();
    log_aicore_scheduler_error_identity(fixture.run(), "terminal");
    EXPECT_TRUE(testing::internal::GetCapturedStderr().empty());
}

TEST(HbgErrorIdentity, ASuccessStatusMustNotConsumeTheFirstErrorClaim) {
    ErrorFixture fixture;
    const auto graph = fixture.graph.graph();
    scheduler_record_error(fixture.run(), 0, SchedulerGraphResult::OK, &graph, fixture.locals[1]);
    EXPECT_EQ(fixture.run()->error_claimed, 0u);
    fixture.record();
    EXPECT_EQ(fixture.run()->scheduler_error, static_cast<uint64_t>(SchedulerGraphResult::INVALID_ARGUMENTS));
}

TEST(HbgErrorIdentity, PublishedStatusAlwaysCarriesOneCoherentWinningIdentity) {
    for (uint32_t round = 0; round < 32; ++round) {
        ErrorFixture fixture;
        fixture.admissions[2].current_task_ref = 1;
        fixture.admissions[2].scan_cursor = 1;
        fixture.admissions[2].gm_mix.block_idx = 3;
        fixture.admissions[2].gm_mix.group_seq = 8;
        std::atomic<uint32_t> ready{0};
        std::atomic<bool> go{false};
        const auto publish = [&](uint32_t worker, int64_t task) {
            ready.fetch_add(1, std::memory_order_release);
            while (!go.load(std::memory_order_acquire)) {}
            fixture.record(worker, task);
        };
        std::thread a(publish, 1, 0);
        std::thread b(publish, 2, 1);
        while (ready.load(std::memory_order_acquire) != 2) {}
        go.store(true, std::memory_order_release);
        while (scheduler_read_error(fixture.run()) == 0) {}
        const uint64_t task = __atomic_load_n(&fixture.run()->error_task_ref, __ATOMIC_RELAXED);
        const auto identity = read_identity(*fixture.run());
        a.join();
        b.join();
        EXPECT_EQ(identity, task == 0 ? (Identity{61, 0, 2, 7, 1, 1}) : (Identity{61, 1, 3, 8, 2, 2}));
        EXPECT_EQ(fixture.run()->error_claimed, 1u);
    }
}

}  // namespace
