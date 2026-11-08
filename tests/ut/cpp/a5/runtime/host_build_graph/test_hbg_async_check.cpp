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
#include <numeric>
#include <vector>

#include "scheduler/scheduler_async_check.h"

namespace {
constexpr uint64_t kEpoch = 73;
constexpr uint64_t kBase = UINT64_C(0x10000000000);
constexpr uint64_t kCookie = UINT64_C(0xfedcba9876543210);

struct BackendState {
    std::array<SchedulerAsyncConditionState, 64> observations{};
    std::array<uint32_t, 64> observed{};
    std::vector<uint32_t> observe_order;
    std::vector<uint64_t> received_cookies;
    uint32_t invalid_result{0};
};

struct Observer {
    BackendState &state;
    SchedulerAsyncConditionOutcome
    observe_once(const SchedulerAsyncGroupIdentity &, uint32_t index, const DeferredCompletionEntry &entry) {
        ++state.observed[index];
        state.observe_order.push_back(index);
        state.received_cookies.push_back(entry.backend_cookie);
        if (state.invalid_result == 1) return {SchedulerAsyncConditionState::READY, -300};
        if (state.invalid_result == 2) return {static_cast<SchedulerAsyncConditionState>(99), 0};
        return {
            state.observations[index], state.observations[index] == SchedulerAsyncConditionState::FAILED ? -701 : 0
        };
    }
};

struct CheckFixture {
    SchedulerAsyncGroupIdentity identity{};
    SchedulerAsyncGroupWriter writer{};
    SchedulerAsyncGroupProgress saved{};
    std::array<DeferredCompletionEntry, 64> entries{};
    BackendState backend;
    uint32_t remaining_budget{0};
    explicit CheckFixture(uint32_t count, bool initialize = true) {
        identity.run_epoch = kEpoch;
        identity.task_ref = 0;
        identity.finish_index = 0;
        identity.capacity = 64;
        writer.count = count;
        writer.reservation.sealed = 1;
        for (uint32_t index = 0; index < 64; ++index) {
            const bool counter = index % 2 == 0;
            entries[index] = {
                kBase + index * 64,
                counter ? 0 : kCookie + index,
                counter ? index : 0,
                COMPLETION_ENGINE_SDMA,
                counter ? COMPLETION_TYPE_COUNTER : COMPLETION_TYPE_SDMA_EVENT_RECORD,
                0
            };
        }
        if (initialize)
            EXPECT_EQ(
                scheduler_async_check_initialize(identity, writer, &saved),
                count == 0 ? SchedulerAsyncCheckResult::DONE : SchedulerAsyncCheckResult::PENDING
            );
    }
    SchedulerAsyncCheckView view() { return {entries.data(), entries.size()}; }
    SchedulerAsyncCheckResult slice(uint32_t, uint32_t budget = 8) {
        Observer observer{backend};
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
        EXPECT_EQ(fixture.saved.remaining_mask, UINT64_MAX);
    }
    std::vector<uint32_t> expected(64);
    std::iota(expected.begin(), expected.end(), 0);
    EXPECT_EQ(fixture.backend.observe_order, expected);
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
}

TEST(HbgAsyncCheck, RoundRobinPositionPersistsAcrossDifferentCoreOwners) {
    CheckFixture fixture(3);
    EXPECT_EQ(fixture.slice(0, 2), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.slice(2, 2), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.slice(1, 2), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(fixture.backend.observe_order, (std::vector<uint32_t>{0, 1, 2, 0, 1, 2}));
    EXPECT_EQ(fixture.saved.next_condition, 0u);
}

TEST(HbgAsyncCheck, SealedWriterAndCapacityAreValidatedBeforeInitialization) {
    for (uint32_t scenario = 0; scenario < 8; ++scenario) {
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
            fixture.writer.reservation.sealed = 0;
            break;
        case 4:
            fixture.writer.reservation.limit = 1;
            break;
        case 5:
            fixture.writer.error_code = -900;
            break;
        case 6:
            fixture.identity.run_epoch = 0;
            break;
        case 7:
            fixture.identity.finish_index = SCHEDULER_ASYNC_INDEX_NONE;
            break;
        }
        EXPECT_EQ(
            scheduler_async_check_initialize(fixture.identity, fixture.writer, &fixture.saved),
            SchedulerAsyncCheckResult::ERROR
        );
        EXPECT_EQ(fixture.saved.phase, static_cast<uint64_t>(SchedulerAsyncCheckPhase::FAILED));
        if (scenario == 5) EXPECT_EQ(fixture.saved.failure_code, -900);
    }
}

TEST(HbgAsyncCheck, UnknownAndMalformedEntryTypesFailBeforeObservation) {
    for (uint32_t scenario = 0; scenario < 3; ++scenario) {
        CheckFixture fixture(1);
        if (scenario == 0) fixture.entries[0].completion_type = -1;
        if (scenario == 1) fixture.entries[0].completion_type = 2;
        if (scenario == 2) fixture.entries[0]._pad = 1;
        EXPECT_EQ(fixture.slice(0), SchedulerAsyncCheckResult::ERROR);
        EXPECT_EQ(fixture.saved.failure_code, static_cast<int64_t>(SchedulerAsyncCheckError::INVALID_ENTRY));
        EXPECT_TRUE(fixture.backend.observe_order.empty());
    }
}

TEST(HbgAsyncCheck, EmptyAndDrainedGroupsNeedNoEntryAccessAndCannotBeReinitialized) {
    CheckFixture empty(0);
    Observer backend{empty.backend};
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
    EXPECT_EQ(fixture.slice(2), SchedulerAsyncCheckResult::DONE);
    EXPECT_EQ(fixture.backend.observed[0], 1u);
    EXPECT_EQ(
        scheduler_async_check_initialize(fixture.identity, fixture.writer, &fixture.saved),
        SchedulerAsyncCheckResult::ERROR
    );
    EXPECT_EQ(fixture.saved.remaining_mask, 0u);
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

TEST(HbgAsyncCheck, ReadyBitsAreClearedAndNeverReadAgainAcrossOwners) {
    CheckFixture f(4);
    f.backend.observations[0] = f.backend.observations[2] = SchedulerAsyncConditionState::READY;
    ASSERT_EQ(f.slice(0), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(f.saved.remaining_mask, UINT64_C(0xa));
    f.entries[0].completion_type = f.entries[2].completion_type = -55;
    ASSERT_EQ(f.slice(2), SchedulerAsyncCheckResult::PENDING);
    EXPECT_EQ(f.backend.observe_order, (std::vector<uint32_t>{0, 1, 2, 3, 1, 3}));
    f.backend.observations[1] = f.backend.observations[3] = SchedulerAsyncConditionState::READY;
    EXPECT_EQ(f.slice(1), SchedulerAsyncCheckResult::DONE);
    EXPECT_EQ(f.saved.remaining_mask, 0u);
    EXPECT_EQ(f.backend.received_cookies[1], kCookie + 1);
}

TEST(HbgAsyncCheck, CompletionOrderingIsPartOfOneObservationUnit) {
    CheckFixture f(1);
    f.backend.observations[0] = SchedulerAsyncConditionState::READY;
    EXPECT_EQ(f.slice(0, 1), SchedulerAsyncCheckResult::DONE);
    EXPECT_EQ(f.remaining_budget, 0u);
    f.backend.observations[0] = SchedulerAsyncConditionState::PENDING;
    EXPECT_EQ(f.slice(2, 1), SchedulerAsyncCheckResult::DONE);
    EXPECT_EQ(f.backend.observed[0], 1u);
}

TEST(HbgAsyncCheck, AllSixtyFourReadyBitsDrainWithinEightBoundedSlices) {
    CheckFixture f(64);
    f.backend.observations.fill(SchedulerAsyncConditionState::READY);
    for (uint32_t slice = 0; slice < 8; ++slice)
        EXPECT_EQ(
            f.slice(slice % 3), slice == 7 ? SchedulerAsyncCheckResult::DONE : SchedulerAsyncCheckResult::PENDING
        );
    for (auto count : f.backend.observed)
        EXPECT_EQ(count, 1u);
}

TEST(HbgAsyncCheck, InvalidProgressFailsBeforeBackendWork) {
    for (uint32_t scenario = 0; scenario < 6; ++scenario) {
        CheckFixture f(2);
        if (scenario == 0) f.saved.remaining_mask = 4;
        if (scenario == 1) f.saved.next_condition = 2;
        if (scenario == 2) f.saved.remaining_mask = 0;
        if (scenario == 3) f.saved.phase = 99;
        if (scenario == 4) f.saved.failure_code = -1;
        if (scenario == 5) f.saved.phase = static_cast<uint64_t>(SchedulerAsyncCheckPhase::DRAINED);
        EXPECT_EQ(f.slice(0), SchedulerAsyncCheckResult::ERROR);
        EXPECT_TRUE(f.backend.observe_order.empty());
    }
}

TEST(HbgAsyncCheck, BackendFailureIsStickyAndKeepsNativeErrorCode) {
    CheckFixture f(1);
    f.backend.observations[0] = SchedulerAsyncConditionState::FAILED;
    EXPECT_EQ(f.slice(0), SchedulerAsyncCheckResult::ERROR);
    EXPECT_EQ(f.saved.failure_code, -701);
    EXPECT_EQ(f.saved.remaining_mask, 1u);
    f.backend.observations[0] = SchedulerAsyncConditionState::READY;
    EXPECT_EQ(f.slice(2), SchedulerAsyncCheckResult::ERROR);
    EXPECT_EQ(f.backend.observed[0], 1u);
}

TEST(HbgAsyncCheck, InvalidBackendStatusesNeverClearAReadyBit) {
    for (uint32_t invalid : {1u, 2u}) {
        CheckFixture f(1);
        f.backend.invalid_result = invalid;
        EXPECT_EQ(f.slice(0), SchedulerAsyncCheckResult::ERROR);
        EXPECT_EQ(f.saved.failure_code, static_cast<int64_t>(SchedulerAsyncCheckError::INVALID_BACKEND_RESULT));
        EXPECT_EQ(f.saved.remaining_mask, 1u);
    }
}

TEST(HbgAsyncCheck, ZeroBudgetAndTruncatedEntryStorageNeverObserve) {
    CheckFixture zero(1);
    EXPECT_EQ(zero.slice(0, 0), SchedulerAsyncCheckResult::PENDING);
    EXPECT_TRUE(zero.backend.observe_order.empty());
    for (bool missing : {false, true}) {
        CheckFixture f(1);
        auto view = f.view();
        if (missing) view.entries = nullptr;
        else view.entry_capacity = 63;
        Observer observer{f.backend};
        uint32_t budget = 8;
        EXPECT_EQ(
            scheduler_async_check_step(f.identity, f.writer, view, observer, &f.saved, &budget),
            SchedulerAsyncCheckResult::ERROR
        );
        EXPECT_EQ(budget, 8u);
        EXPECT_TRUE(f.backend.observe_order.empty());
    }
    CheckFixture unsealed(1, false);
    EXPECT_EQ(unsealed.slice(0), SchedulerAsyncCheckResult::ERROR);
}

TEST(HbgAsyncCheck, HandoffReadsTheCurrentSealedTokenInsteadOfAnOldSnapshot) {
    CheckFixture f(1);
    EXPECT_EQ(f.slice(0, 1), SchedulerAsyncCheckResult::PENDING);
    f.entries[0].completion_type = -1;
    f.backend.observations[0] = SchedulerAsyncConditionState::READY;
    EXPECT_EQ(f.slice(2, 1), SchedulerAsyncCheckResult::ERROR);
    EXPECT_EQ(f.backend.observed[0], 1u);
    EXPECT_EQ(f.saved.remaining_mask, 1u);
}
}  // namespace
