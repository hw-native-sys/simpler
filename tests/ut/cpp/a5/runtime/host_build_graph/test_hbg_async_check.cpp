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

#include <array>
#include <limits>
#include <numeric>
#include <vector>

#include "scheduler/scheduler_async_check.h"

namespace {
constexpr uint64_t kEpoch = 73;
constexpr uint64_t kGeneration = 19;
constexpr uint64_t kBase = UINT64_C(0x10000000000);
constexpr uint64_t kCookie = UINT64_C(0xfedcba9876543210);

struct BackendState {
    std::array<SchedulerAsyncConditionState, 64> observations{};
    std::array<uint32_t, 64> observed{};
    std::array<uint32_t, 64> acquired{};
    std::array<uint32_t, 64> validated{};
    std::array<uint32_t, 64> acquire_steps{};
    std::array<bool, 64> registered{};
    std::array<bool, 64> run_scoped{};
    std::array<uint64_t, 64> retained_run_epochs{};
    std::array<bool, 64> stalled{};
    std::array<bool, 64> acquire_fails{};
    std::vector<uint32_t> observe_order;
    std::vector<uint32_t> acquire_workers;
    std::vector<uint64_t> received_cookies;
    bool invalid_observe_result{false};
    bool invalid_acquire_result{false};

    BackendState() {
        acquire_steps.fill(1);
        registered.fill(true);
        retained_run_epochs.fill(kEpoch);
    }
};

// A synthetic authoritative registry, not a permissive address predicate. Each
// side reference must match the existing run/domain qualification and record.
// No method dereferences the fake condition address or performs a DMA atomic.
struct RegistryBackend {
    BackendState &state;
    uint32_t worker;

    bool validate(
        const SchedulerAsyncGroupIdentity &identity, uint32_t index, const DeferredCompletionEntry &entry,
        const SchedulerAsyncConditionBinding &binding
    ) {
        ++state.validated[index];
        if (identity.run_epoch != kEpoch || identity.task_ref != 0 || binding.binding_index >= 64 ||
            !state.registered[binding.binding_index])
            return false;
        const uint64_t resource = binding.binding_index;
        const bool retained = state.run_scoped[resource] ? (binding.owner_generation == 0 &&
                                                            state.retained_run_epochs[resource] == identity.run_epoch) :
                                                           binding.owner_generation == kGeneration;
        return retained && entry.addr == kBase + resource * 64 && entry.engine == COMPLETION_ENGINE_SDMA;
    }
    SchedulerAsyncConditionOutcome observe_once(
        const SchedulerAsyncGroupIdentity &, uint32_t index, const DeferredCompletionEntry &entry,
        const SchedulerAsyncConditionBinding &
    ) {
        ++state.observed[index];
        state.observe_order.push_back(index);
        state.received_cookies.push_back(entry.backend_cookie);
        if (state.invalid_observe_result) return {SchedulerAsyncConditionState::READY, -300};
        return {
            state.observations[index], state.observations[index] == SchedulerAsyncConditionState::FAILED ? -701 : 0
        };
    }
    SchedulerAsyncConditionOutcome acquire_completed(
        const SchedulerAsyncGroupIdentity &, uint32_t index, const DeferredCompletionEntry &entry,
        const SchedulerAsyncConditionBinding &, uint64_t &stage, uint64_t &offset
    ) {
        ++state.acquired[index];
        state.acquire_workers.push_back(worker);
        state.received_cookies.push_back(entry.backend_cookie);
        if (state.invalid_acquire_result) return {static_cast<SchedulerAsyncConditionState>(99), 0};
        if (state.acquire_fails[index]) return {SchedulerAsyncConditionState::FAILED, -702};
        if (offset != stage * 16 || stage >= state.acquire_steps[index])
            return {SchedulerAsyncConditionState::FAILED, -703};
        if (state.stalled[index]) return {SchedulerAsyncConditionState::PENDING, 0};
        ++stage;
        offset += 16;
        return {
            stage == state.acquire_steps[index] ? SchedulerAsyncConditionState::READY :
                                                  SchedulerAsyncConditionState::PENDING,
            0
        };
    }
};

struct CheckFixture {
    SchedulerAsyncGroupIdentity identity{};
    SchedulerAsyncGroupWriter writer{};
    SchedulerAsyncGroupProgress saved{};
    std::array<DeferredCompletionEntry, 64> entries{};
    std::array<SchedulerAsyncConditionBinding, 64> bindings{};
    BackendState backend;
    uint32_t remaining_budget{0};

    explicit CheckFixture(uint32_t count, bool initialize = true) {
        identity.run_epoch = kEpoch;
        identity.task_ref = 0;
        identity.finish_index = 0;
        identity.capacity = 64;
        identity.condition_types = CORE_ASYNC_CONDITION_TYPES;
        writer.count = count;
        writer.sealed = 1;
        for (uint32_t index = 0; index < 64; ++index) {
            entries[index] = {
                kBase + index * 64,
                kCookie + index,
                UINT32_MAX - index,
                COMPLETION_ENGINE_SDMA,
                index % 2 == 0 ? COMPLETION_TYPE_COUNTER : COMPLETION_TYPE_SDMA_EVENT_RECORD,
                0
            };
            bindings[index] = {index, kGeneration};
        }
        if (initialize)
            EXPECT_EQ(
                scheduler_async_check_initialize(identity, writer, &saved),
                count == 0 ? SchedulerAsyncCheckResult::DONE : SchedulerAsyncCheckResult::PENDING
            );
    }
    SchedulerAsyncCheckView view() { return {entries.data(), bindings.data(), entries.size(), bindings.size()}; }

    SchedulerAsyncCheckResult slice(uint32_t worker, uint32_t budget = 8) {
        RegistryBackend observer{backend, worker};
        auto progress = saved;
        remaining_budget = budget;
        const auto result =
            scheduler_async_check_step(identity, writer, view(), observer, &progress, &remaining_budget);
        saved = progress;
        return result;
    }
};

TEST(HbgAsyncCheck, SixtyFourPendingConditionsCompleteOneRotationInEightSlices) {
    CheckFixture fixture(64);
    for (uint32_t slice = 0; slice < 8; ++slice) {
        ASSERT_EQ(fixture.slice(slice % 3), SchedulerAsyncCheckResult::PENDING);
        EXPECT_EQ(fixture.remaining_budget, 0u);
        EXPECT_EQ(fixture.saved.next_condition, (slice + 1) * 8 % 64);
        EXPECT_EQ(fixture.saved.observation_count, (slice + 1) * 8);
        EXPECT_EQ(fixture.saved.work_units, (slice + 1) * 8);
        EXPECT_EQ(fixture.saved.acquire_count, 0u);
        EXPECT_EQ(fixture.saved.remaining_mask, UINT64_MAX);
    }
    std::vector<uint32_t> expected(64);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(fixture.backend.observe_order, expected);
    EXPECT_EQ(fixture.saved.slice_count, 8u);
    for (uint32_t count : fixture.backend.observed)
        EXPECT_EQ(count, 1u);
}

TEST(HbgAsyncCheck, PendingConditionIsObservedOnlyOnceEvenWithOversizedCallerBudget) {
    CheckFixture one(1);
    EXPECT_EQ(one.slice(0, 100), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(one.remaining_budget, 99u);
    EXPECT_EQ(one.backend.observed[0], 1u);
    CheckFixture many(64);
    EXPECT_EQ(many.slice(0, 100), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(many.remaining_budget, 92u);
    EXPECT_EQ(many.saved.observation_count, 8u);
}

TEST(HbgAsyncCheck, RoundRobinPositionPersistsAcrossDifferentCoreOwners) {
    CheckFixture fixture(3);
    EXPECT_EQ(fixture.slice(0, 2), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.slice(2, 2), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.slice(1, 2), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.backend.observe_order, (std::vector<uint32_t>{0, 1, 2, 0, 1, 2}));
    EXPECT_EQ(fixture.saved.next_condition, 0u);
}

TEST(HbgAsyncCheck, ReadyBitsClearOnlyAfterAcquisitionAndNeverReadAgain) {
    CheckFixture fixture(4);
    fixture.backend.observations[0] = fixture.backend.observations[2] = SchedulerAsyncConditionState::READY;
    ASSERT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.saved.remaining_mask, UINT64_C(0xa));
    EXPECT_EQ(fixture.saved.observation_count, 4u);
    EXPECT_EQ(fixture.saved.acquire_count, 2u);
    EXPECT_EQ(fixture.saved.work_units, 6u);
    // A cleared entry and side reference become unusable. Neither generic
    // validation nor either backend operation may inspect that index again.
    fixture.entries[0].completion_type = fixture.entries[2].completion_type = -55;
    fixture.bindings[0] = fixture.bindings[2] = {};
    const auto validated = fixture.backend.validated;
    EXPECT_EQ(fixture.slice(2), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.saved.observation_count, 6u);
    EXPECT_EQ(fixture.backend.validated[0], validated[0]);
    EXPECT_EQ(fixture.backend.validated[2], validated[2]);
    EXPECT_EQ(fixture.backend.observe_order, (std::vector<uint32_t>{0, 1, 2, 3, 1, 3}));
}

TEST(HbgAsyncCheck, AcquisitionContinuationMovesAcrossCoresWithoutReobservingReady) {
    CheckFixture fixture(1);
    fixture.backend.observations[0] = SchedulerAsyncConditionState::READY;
    fixture.backend.acquire_steps[0] = 4;
    ASSERT_EQ(fixture.slice(0, 1), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.saved.acquire_condition, 0u);
    EXPECT_EQ(fixture.saved.acquire_stage, 0u);
    EXPECT_EQ(fixture.saved.remaining_mask, 1u);
    // READY is cached monotonically. A later observer must resume acquisition,
    // not poll the condition again even if the test source changes its answer.
    fixture.backend.observations[0] = SchedulerAsyncConditionState::PENDING;
    for (uint32_t step = 1; step <= 4; ++step) {
        EXPECT_EQ(
            fixture.slice(step % 3, 1), step == 4 ? SchedulerAsyncCheckResult::DONE : SchedulerAsyncCheckResult::PENDING
        );
        if (step != 4) {
            EXPECT_EQ(fixture.saved.acquire_stage, step);
            EXPECT_EQ(fixture.saved.acquire_byte_offset, step * 16);
            EXPECT_EQ(fixture.saved.remaining_mask, 1u);
        }
    }
    EXPECT_EQ(fixture.backend.observed[0], 1u);
    EXPECT_EQ(fixture.backend.acquire_workers, (std::vector<uint32_t>{1, 2, 0, 1}));
    EXPECT_EQ(fixture.saved.observation_count, 1u);
    EXPECT_EQ(fixture.saved.acquire_count, 4u);
    EXPECT_EQ(fixture.saved.work_units, 5u);
    EXPECT_EQ(fixture.saved.acquire_condition, SCHEDULER_ASYNC_INDEX_NONE);
    EXPECT_EQ(fixture.saved.remaining_mask, 0u);
    EXPECT_EQ(fixture.saved.acquire_stage, 0u);
    EXPECT_EQ(fixture.saved.acquire_byte_offset, 0u);
    for (auto cookie : fixture.backend.received_cookies)
        EXPECT_EQ(cookie, kCookie);
}

TEST(HbgAsyncCheck, StalledAcquisitionYieldsWithoutSpendingTheRestOfTheSlice) {
    CheckFixture fixture(1);
    fixture.backend.observations[0] = SchedulerAsyncConditionState::READY;
    fixture.backend.stalled[0] = true;
    ASSERT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.saved.work_units, 2u);
    EXPECT_EQ(fixture.remaining_budget, 6u);
    ASSERT_EQ(fixture.slice(1), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.saved.work_units, 3u);
    EXPECT_EQ(fixture.remaining_budget, 7u);
    fixture.backend.stalled[0] = false;
    EXPECT_EQ(fixture.slice(2), SchedulerAsyncCheckResult::DONE);
    EXPECT_EQ(fixture.saved.observation_count, 1u);
}

TEST(HbgAsyncCheck, SealedWriterAndCapacityAreValidatedBeforeInitialization) {
    for (uint32_t scenario = 0; scenario < 10; ++scenario) {
        CheckFixture fixture(1, false);
        switch (scenario) {
        case 0:
            fixture.identity.capacity = 0;
            break;
        case 1:
            fixture.identity.capacity = 65;
            break;
        case 2:
            fixture.writer.count = 65;
            break;
        case 3:
            fixture.writer.sealed = 0;
            break;
        case 4:
            fixture.writer.reserved_count = 1;
            break;
        case 5:
            fixture.writer.error_code = -900;
            break;
        case 6:
            fixture.identity.condition_types = 0;
            break;
        case 7:
            fixture.identity.condition_types = 4;
            break;
        case 8:
            fixture.identity.run_epoch = 0;
            break;
        case 9:
            fixture.identity.finish_index = SCHEDULER_ASYNC_INDEX_NONE;
            break;
        }
        EXPECT_EQ(
            scheduler_async_check_initialize(fixture.identity, fixture.writer, &fixture.saved),
            SchedulerAsyncCheckResult::ERROR
        );
        EXPECT_EQ(fixture.saved.phase, static_cast<uint64_t>(SchedulerAsyncCheckPhase::FAILED));
        if (scenario == 5) EXPECT_EQ(fixture.saved.failure_code, -900);
        EXPECT_EQ(fixture.saved.observation_count, 0u);
    }
}

TEST(HbgAsyncCheck, InvalidMaskCursorAndAcquisitionIdentityFailWithoutBackendWork) {
    for (uint32_t scenario = 0; scenario < 8; ++scenario) {
        CheckFixture fixture(2);
        switch (scenario) {
        case 0:
            fixture.saved.remaining_mask = 4;
            break;
        case 1:
            fixture.saved.next_condition = 2;
            break;
        case 2:
            fixture.saved.acquire_condition = 2;
            break;
        case 3:
            fixture.saved.acquire_condition = 1;
            fixture.saved.remaining_mask = 1;
            break;
        case 4:
            fixture.saved.acquire_stage = 1;
            break;
        case 5:
            fixture.saved.acquire_byte_offset = 16;
            break;
        case 6:
            fixture.saved.phase = static_cast<uint64_t>(SchedulerAsyncCheckPhase::DRAINED);
            break;
        case 7:
            fixture.saved.phase = static_cast<uint64_t>(SchedulerAsyncCheckPhase::DRAINED);
            fixture.saved.remaining_mask = 0;
            fixture.saved.next_condition = 2;
            break;
        }
        EXPECT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::ERROR);
        EXPECT_EQ(fixture.saved.work_units, 0u);
        EXPECT_TRUE(fixture.backend.observe_order.empty());
    }
}

TEST(HbgAsyncCheck, MissingMismatchedAndUnqualifiedBindingsFailClosed) {
    for (uint32_t scenario = 0; scenario < 6; ++scenario) {
        CheckFixture fixture(1);
        switch (scenario) {
        case 0:
            fixture.bindings[0].binding_index = SCHEDULER_ASYNC_INDEX_NONE;
            break;
        case 1:
            fixture.bindings[0].owner_generation = 0;
            break;
        case 2:
            fixture.bindings[0].binding_index = 64;
            break;
        case 3:
            fixture.bindings[0].owner_generation = kGeneration + 1;
            break;
        case 4:
            fixture.backend.registered[0] = false;
            break;
        case 5:
            fixture.entries[0].addr = kBase + 1;
            break;
        }
        EXPECT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::ERROR);
        EXPECT_EQ(fixture.saved.work_units, 1u);
        EXPECT_EQ(fixture.saved.observation_count, 0u);
        EXPECT_TRUE(fixture.backend.observe_order.empty());
    }
}

TEST(HbgAsyncCheck, UnknownUndeclaredAndMalformedEntryTypesFailBeforeObservation) {
    for (uint32_t scenario = 0; scenario < 4; ++scenario) {
        CheckFixture fixture(1);
        if (scenario == 0) fixture.entries[0].completion_type = -1;
        if (scenario == 1) fixture.entries[0].completion_type = 2;
        if (scenario == 2) fixture.identity.condition_types = CORE_ASYNC_CONDITION_SDMA_POST_DONE;
        if (scenario == 3) fixture.entries[0]._pad = 1;
        EXPECT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::ERROR);
        EXPECT_EQ(fixture.saved.failure_code, static_cast<int64_t>(SchedulerAsyncCheckError::INVALID_ENTRY));
        EXPECT_EQ(fixture.backend.validated[0], 0u);
        EXPECT_TRUE(fixture.backend.observe_order.empty());
    }
}

TEST(HbgAsyncCheck, BackendFailureIsStickyAndKeepsNativeErrorCode) {
    for (bool acquire_failure : {false, true}) {
        CheckFixture fixture(1);
        fixture.backend.observations[0] =
            acquire_failure ? SchedulerAsyncConditionState::READY : SchedulerAsyncConditionState::FAILED;
        fixture.backend.acquire_fails[0] = acquire_failure;
        EXPECT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::ERROR);
        EXPECT_EQ(fixture.saved.failure_code, acquire_failure ? -702 : -701);
        EXPECT_EQ(fixture.saved.remaining_mask, 1u);
        const auto calls = fixture.backend.validated[0];
        fixture.backend.observations[0] = SchedulerAsyncConditionState::READY;
        fixture.backend.acquire_fails[0] = false;
        EXPECT_EQ(fixture.slice(2), SchedulerAsyncCheckResult::ERROR);
        EXPECT_EQ(fixture.backend.validated[0], calls);
        EXPECT_EQ(fixture.saved.failure_code, acquire_failure ? -702 : -701);
    }
}

TEST(HbgAsyncCheck, InvalidBackendStatusesAreNotSuccessfulCompletion) {
    for (bool acquire : {false, true}) {
        CheckFixture fixture(1);
        fixture.backend.observations[0] = SchedulerAsyncConditionState::READY;
        fixture.backend.invalid_observe_result = !acquire;
        fixture.backend.invalid_acquire_result = acquire;
        EXPECT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::ERROR);
        EXPECT_EQ(fixture.saved.failure_code, static_cast<int64_t>(SchedulerAsyncCheckError::INVALID_BACKEND_RESULT));
        EXPECT_EQ(fixture.saved.remaining_mask, 1u);
    }
}

TEST(HbgAsyncCheck, ZeroBudgetDoesNotValidateOrObserveAnyEntry) {
    CheckFixture fixture(1);
    fixture.bindings[0] = {};
    EXPECT_EQ(fixture.slice(0, 0), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.backend.validated[0], 0u);
    EXPECT_EQ(fixture.saved.work_units, 0u);
    EXPECT_EQ(fixture.saved.observation_count, 0u);
    EXPECT_EQ(fixture.saved.slice_count, 1u);
    EXPECT_EQ(fixture.saved.remaining_mask, 1u);
}

TEST(HbgAsyncCheck, EmptyAndDrainedGroupsNeedNoEntryAccessAndCannotBeReinitialized) {
    CheckFixture empty(0);
    RegistryBackend backend{empty.backend, 0};
    uint32_t budget = 8;
    EXPECT_EQ(
        scheduler_async_check_step(empty.identity, empty.writer, {}, backend, &empty.saved, &budget),
        SchedulerAsyncCheckResult::DONE
    );
    EXPECT_EQ(budget, 8u);
    CheckFixture fixture(1);
    fixture.backend.observations[0] = SchedulerAsyncConditionState::READY;
    ASSERT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::DONE);
    fixture.entries[0].completion_type = -100;
    fixture.bindings[0] = {};
    EXPECT_EQ(fixture.slice(2), SchedulerAsyncCheckResult::DONE);
    EXPECT_EQ(fixture.backend.observed[0], 1u);
    EXPECT_EQ(
        scheduler_async_check_initialize(fixture.identity, fixture.writer, &fixture.saved),
        SchedulerAsyncCheckResult::ERROR
    );
    EXPECT_EQ(fixture.saved.remaining_mask, 0u);
}

TEST(HbgAsyncCheck, UninitializedProgressAndTruncatedSideTablesFailClosed) {
    CheckFixture uninitialized(1, false);
    EXPECT_EQ(uninitialized.slice(0), SchedulerAsyncCheckResult::ERROR);
    for (bool missing : {false, true}) {
        CheckFixture fixture(1);
        auto view = fixture.view();
        if (missing) view.bindings = nullptr;
        else view.binding_capacity = 63;
        RegistryBackend backend{fixture.backend, 0};
        uint32_t budget = 8;
        EXPECT_EQ(
            scheduler_async_check_step(fixture.identity, fixture.writer, view, backend, &fixture.saved, &budget),
            SchedulerAsyncCheckResult::ERROR
        );
        EXPECT_EQ(budget, 8u);
        EXPECT_EQ(fixture.saved.work_units, 0u);
    }
}

TEST(HbgAsyncCheck, ExistingRunScopedBindingDoesNotRequireAnotherAllocationGeneration) {
    CheckFixture fixture(1);
    fixture.backend.run_scoped[0] = true;
    fixture.bindings[0].owner_generation = 0;
    fixture.backend.observations[0] = SchedulerAsyncConditionState::READY;
    EXPECT_EQ(fixture.slice(1), SchedulerAsyncCheckResult::DONE);
    EXPECT_EQ(fixture.saved.observation_count, 1u);
    EXPECT_EQ(fixture.saved.acquire_count, 1u);
    EXPECT_EQ(fixture.saved.work_units, 2u);

    CheckFixture wrong_run(1);
    wrong_run.backend.run_scoped[0] = true;
    wrong_run.bindings[0].owner_generation = 0;
    wrong_run.backend.retained_run_epochs[0] = kEpoch + 1;
    EXPECT_EQ(wrong_run.slice(0), SchedulerAsyncCheckResult::ERROR);
    EXPECT_TRUE(wrong_run.backend.observe_order.empty());
}

TEST(HbgAsyncCheck, BackendValidatesItsOpaqueAcquisitionCursorBeforeResuming) {
    CheckFixture fixture(1);
    fixture.backend.observations[0] = SchedulerAsyncConditionState::READY;
    ASSERT_EQ(fixture.slice(0, 1), SchedulerAsyncCheckResult::PENDING);
    fixture.saved.acquire_byte_offset = 8;
    EXPECT_EQ(fixture.slice(2), SchedulerAsyncCheckResult::ERROR);
    EXPECT_EQ(fixture.saved.failure_code, -703);
    EXPECT_EQ(fixture.saved.remaining_mask, 1u);
    EXPECT_EQ(fixture.saved.observation_count, 1u);
}

TEST(HbgAsyncCheck, DiagnosticCountersSaturateWithoutChangingProgress) {
    CheckFixture fixture(1);
    fixture.saved.slice_count = fixture.saved.work_units = fixture.saved.observation_count = UINT64_MAX;
    EXPECT_EQ(fixture.slice(0, 1), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.saved.slice_count, UINT64_MAX);
    EXPECT_EQ(fixture.saved.work_units, UINT64_MAX);
    EXPECT_EQ(fixture.saved.observation_count, UINT64_MAX);
    EXPECT_EQ(fixture.backend.observed[0], 1u);
    EXPECT_EQ(fixture.saved.remaining_mask, 1u);
}
TEST(HbgAsyncCheck, NewOwnerRevalidatesTheCurrentSideReferenceInsteadOfKeepingAnOldSnapshot) {
    CheckFixture fixture(1);
    ASSERT_EQ(fixture.slice(0, 1), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.backend.observed[0], 1u);
    // Corruption after a handoff must not be masked by the previous owner's
    // cached qualification. Real sealed references are immutable for the run.
    fixture.bindings[0].owner_generation = kGeneration + 1;
    fixture.backend.observations[0] = SchedulerAsyncConditionState::READY;
    EXPECT_EQ(fixture.slice(2, 1), SchedulerAsyncCheckResult::ERROR);
    EXPECT_EQ(fixture.backend.observed[0], 1u);
    EXPECT_EQ(fixture.saved.remaining_mask, 1u);
}

TEST(HbgAsyncCheck, SealedMetadataReadsPreserveBothThirtyTwoBitPairsAndFullCookie) {
    CheckFixture fixture(1);
    fixture.entries[0] = {kBase, UINT64_MAX, UINT32_MAX, UINT32_C(0x87654321), -11, UINT32_C(0xfedcba98)};
    const auto value = scheduler_async_check_detail::entry(fixture.view(), 0);
    EXPECT_EQ(value.addr, kBase);
    EXPECT_EQ(value.backend_cookie, UINT64_MAX);
    EXPECT_EQ(value.expected_value, UINT32_MAX);
    EXPECT_EQ(value.engine, UINT32_C(0x87654321));
    EXPECT_EQ(value.completion_type, -11);
    EXPECT_EQ(value._pad, UINT32_C(0xfedcba98));
}
}  // namespace
