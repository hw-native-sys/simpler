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
#include <cstdint>
#include <cstdlib>
#include <vector>

#include "host/execution_mode_latch.h"
#include "host/kernel_execution_state.h"
#include "host/kernel_launch_binder.h"

namespace {

/**
 * Fake ops backing the restricted context vocabulary. Handles are small
 * integers cast to pointers; create/destroy counters prove balance, and the
 * failure knobs drive the CLOSING-retry and partial-init paths.
 */
struct FakeContextOps {
    int current_device{3};
    int get_device_rc{0};
    void *current_context{reinterpret_cast<void *>(0x1000)};
    int get_context_rc{0};
    int create_stream_rc_after{-1};  // fail the Nth create_hidden_stream (0-based); -1 = never
    int create_event_rc_after{-1};
    int destroy_failures_remaining{0};

    int streams_created{0};
    int streams_destroyed{0};
    int events_created{0};
    int events_destroyed{0};
    uintptr_t next_handle{1};
    void *destroy_failure_handle{nullptr};
    std::vector<void *> destroyed_handles;
    static FakeContextOps *self(void *context) { return static_cast<FakeContextOps *>(context); }

    static int get_current_device(void *context, int *device_id) noexcept {
        auto *ops = self(context);
        if (ops->get_device_rc != 0) return ops->get_device_rc;
        *device_id = ops->current_device;
        return 0;
    }
    static int get_current_context(void *context, void **current) noexcept {
        auto *ops = self(context);
        if (ops->get_context_rc != 0) return ops->get_context_rc;
        *current = ops->current_context;
        return 0;
    }
    static int create_hidden_stream(void *context, void **stream) noexcept {
        auto *ops = self(context);
        if (ops->create_stream_rc_after >= 0 && ops->streams_created == ops->create_stream_rc_after) return -42;
        ops->streams_created++;
        *stream = reinterpret_cast<void *>(ops->next_handle++);
        return 0;
    }
    static int destroy_hidden_stream(void *context, void *handle) noexcept {
        auto *ops = self(context);
        if (handle == ops->destroy_failure_handle) {
            ops->destroy_failure_handle = nullptr;
            return -43;
        }
        if (ops->destroy_failures_remaining > 0) {
            ops->destroy_failures_remaining--;
            return -43;
        }
        ops->streams_destroyed++;
        ops->destroyed_handles.push_back(handle);
        return 0;
    }
    static int create_event(void *context, void **event) noexcept {
        auto *ops = self(context);
        if (ops->create_event_rc_after >= 0 && ops->events_created == ops->create_event_rc_after) return -44;
        ops->events_created++;
        *event = reinterpret_cast<void *>(ops->next_handle++);
        return 0;
    }
    static int destroy_event(void *context, void *handle) noexcept {
        auto *ops = self(context);
        if (handle == ops->destroy_failure_handle) {
            ops->destroy_failure_handle = nullptr;
            return -43;
        }
        if (ops->destroy_failures_remaining > 0) {
            ops->destroy_failures_remaining--;
            return -43;
        }
        ops->events_destroyed++;
        ops->destroyed_handles.push_back(handle);
        return 0;
    }

    KernelContextOps table() {
        return KernelContextOps{
            this,          &get_current_device, &get_current_context, &create_hidden_stream, &destroy_hidden_stream,
            &create_event, &destroy_event
        };
    }
};

constexpr size_t kStreamCount = static_cast<size_t>(KernelStreamKind::Count);
constexpr size_t kEventCount = static_cast<size_t>(KernelEventKind::Count);

TEST(KernelExecutionState, NoBorrowedIdentityNeedsNoCallerContext) {
    KernelExecutionState state;
    EXPECT_EQ(state.validate_current_context(), 0);
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.validate_current_context(), 0);
}

TEST(KernelExecutionState, SameDeviceDifferentContextRefusesCloseWithoutReleasingOwners) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    void *owner = fake.current_context;
    fake.current_context = reinterpret_cast<void *>(0x2000);
    EXPECT_EQ(state.close(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.phase(), KernelContextPhase::Collecting);
    EXPECT_EQ(fake.events_destroyed, 0);
    EXPECT_EQ(fake.streams_destroyed, 0);
    fake.current_context = owner;
    EXPECT_EQ(state.close(), 0);
    EXPECT_FALSE(state.has_live_resources());
}

TEST(KernelExecutionState, MissingContextOrQueryFailureCreatesNoResources) {
    FakeContextOps fake;
    KernelExecutionState state;
    fake.current_context = nullptr;
    EXPECT_EQ(state.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    fake.get_context_rc = -45;
    EXPECT_EQ(state.initialize(3, fake.table()), -45);
    EXPECT_EQ(fake.streams_created, 0);
    EXPECT_EQ(fake.events_created, 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    fake.get_context_rc = 0;
    fake.current_context = reinterpret_cast<void *>(0x1000);
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.close(), 0);
}

TEST(KernelExecutionState, ContextValidationPreservesStateOnDeviceAndQueryErrors) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.validate_current_context(), 0);
    fake.current_device = 4;
    EXPECT_EQ(state.validate_current_context(), PTO_RUNTIME_ERR_INVALID_STATE);
    fake.current_device = 3;
    fake.get_context_rc = -45;
    EXPECT_EQ(state.validate_current_context(), -45);
    EXPECT_EQ(state.close(), -45);
    EXPECT_EQ(state.phase(), KernelContextPhase::Collecting);
    EXPECT_EQ(fake.streams_destroyed, 0);
    EXPECT_EQ(fake.events_destroyed, 0);
    fake.get_context_rc = 0;
    EXPECT_EQ(state.close(), 0);
}

TEST(KernelExecutionState, BorrowedIdentitySurvivesPartialAndCompletedClose) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    fake.destroy_failures_remaining = 1;
    EXPECT_EQ(state.close(), -43);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_TRUE(state.has_live_resources());
    const int destroyed = fake.events_destroyed;
    fake.current_context = reinterpret_cast<void *>(0x2000);
    EXPECT_EQ(state.close(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(fake.events_destroyed, destroyed);
    fake.current_context = reinterpret_cast<void *>(0x1000);
    EXPECT_EQ(state.close(), 0);
    EXPECT_FALSE(state.has_live_resources());
    fake.current_context = reinterpret_cast<void *>(0x2000);
    EXPECT_EQ(state.validate_current_context(), PTO_RUNTIME_ERR_INVALID_STATE);
    fake.current_context = reinterpret_cast<void *>(0x1000);
    EXPECT_EQ(state.validate_current_context(), 0);
}

class KernelContextCreationFailure : public ::testing::TestWithParam<int> {};

TEST_P(KernelContextCreationFailure, EveryCreationPointRollsBackAndCanRetry) {
    FakeContextOps fake;
    const int point = GetParam();
    if (point < static_cast<int>(kStreamCount)) {
        fake.create_stream_rc_after = point;
    } else {
        fake.create_event_rc_after = point - kStreamCount;
    }
    KernelExecutionState state;
    EXPECT_NE(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_created, fake.streams_destroyed);
    EXPECT_EQ(fake.events_created, fake.events_destroyed);
    EXPECT_EQ(fake.streams_created + fake.events_created, point);
    fake.get_context_rc = -45;
    EXPECT_EQ(state.validate_current_context(), 0);
    fake.get_context_rc = 0;
    fake.create_stream_rc_after = -1;
    fake.create_event_rc_after = -1;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(fake.streams_created, fake.streams_destroyed);
    EXPECT_EQ(fake.events_created, fake.events_destroyed);
}

INSTANTIATE_TEST_SUITE_P(
    AllStreamsAndEvents, KernelContextCreationFailure, ::testing::Range(0, static_cast<int>(kStreamCount + kEventCount))
);

class KernelContextCloseFailure : public ::testing::TestWithParam<int> {};

TEST_P(KernelContextCloseFailure, OnlyTheFailedHandleSurvivesAndIsDestroyedOnceOnRetry) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    std::vector<void *> handles;
    for (size_t i = 0; i < kStreamCount; ++i)
        handles.push_back(state.hidden_stream(static_cast<KernelStreamKind>(i)));
    for (size_t i = 0; i < kEventCount; ++i)
        handles.push_back(state.event(static_cast<KernelEventKind>(i)));
    void *failed = handles.at(GetParam());
    fake.destroy_failure_handle = failed;
    EXPECT_EQ(state.close(), -43);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_FALSE(state.accepts_dispatch());
    ASSERT_EQ(fake.destroyed_handles.size(), handles.size() - 1);
    for (size_t i = 0; i < handles.size(); ++i) {
        void *remaining = i < kStreamCount ? state.hidden_stream(static_cast<KernelStreamKind>(i)) :
                                             state.event(static_cast<KernelEventKind>(i - kStreamCount));
        EXPECT_EQ(remaining, handles[i] == failed ? failed : nullptr);
    }
    fake.current_context = reinterpret_cast<void *>(0x2000);
    EXPECT_EQ(state.close(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(fake.destroyed_handles.size(), handles.size() - 1);
    fake.current_context = reinterpret_cast<void *>(0x1000);
    EXPECT_EQ(state.close(), 0);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(state.close(), 0);
    ASSERT_EQ(fake.destroyed_handles.size(), handles.size());
    for (void *handle : handles) {
        EXPECT_EQ(std::count(fake.destroyed_handles.begin(), fake.destroyed_handles.end(), handle), 1);
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllStreamsAndEvents, KernelContextCloseFailure, ::testing::Range(0, static_cast<int>(kStreamCount + kEventCount))
);

TEST(KernelExecutionState, DestructionWithoutCloseMakesNoRuntimeCalls) {
    FakeContextOps fake;
    {
        KernelExecutionState state;
        ASSERT_EQ(state.initialize(3, fake.table()), 0);
    }
    EXPECT_EQ(fake.streams_destroyed, 0);
    EXPECT_EQ(fake.events_destroyed, 0);
}

TEST(KernelContextOpsVocabulary, IncompleteTableIsInvalid) {
    FakeContextOps fake;
    KernelContextOps ops = fake.table();
    EXPECT_TRUE(ops.valid());
    ops.create_event = nullptr;
    EXPECT_FALSE(ops.valid());
    ops = fake.table();
    ops.get_current_context = nullptr;
    EXPECT_FALSE(ops.valid());
}

TEST(ExecutionModeLatch, StartsUnlatched) {
    ExecutionModeLatch latch;
    EXPECT_FALSE(latch.is_latched());
    EXPECT_FALSE(latch.is_kernel());
}

TEST(ExecutionModeLatch, LatchingIsIdempotentForTheHeldMode) {
    ExecutionModeLatch latch;
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_TRUE(latch.is_latched());
    EXPECT_EQ(latch.latched_mode(), SIMPLER_MODE_PROGRAM);
    EXPECT_FALSE(latch.is_kernel());
}

TEST(ExecutionModeLatch, ModesAreMutuallyExclusiveInBothDirections) {
    ExecutionModeLatch program;
    EXPECT_EQ(program.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(program.latch(SIMPLER_MODE_KERNEL), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_FALSE(program.is_kernel());

    ExecutionModeLatch kernel;
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_KERNEL), 0);
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_KERNEL), 0);
    EXPECT_EQ(kernel.latch(SIMPLER_MODE_PROGRAM), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(kernel.is_kernel());
}

// finalize resets a runner's device state but not its identity, so the
// init -> finalize -> init sequence the program path already supports must
// still latch cleanly the second time.
TEST(ExecutionModeLatch, SameModeRelatchesAfterAFinalizedLifetime) {
    ExecutionModeLatch latch;
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_PROGRAM), 0);
    EXPECT_EQ(latch.latched_mode(), SIMPLER_MODE_PROGRAM);
    EXPECT_EQ(latch.latch(SIMPLER_MODE_KERNEL), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(KernelExecutionState, EmptyContextInitThenCloseIsClean) {
    FakeContextOps fake;
    KernelExecutionState state;
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Collecting);
    EXPECT_TRUE(state.accepts_dispatch());
    EXPECT_EQ(state.device_id(), 3);
    EXPECT_EQ(fake.streams_created, static_cast<int>(kStreamCount));
    EXPECT_EQ(fake.events_created, static_cast<int>(kEventCount));
    for (size_t i = 0; i < kStreamCount; ++i) {
        EXPECT_NE(state.hidden_stream(static_cast<KernelStreamKind>(i)), nullptr);
    }
    for (size_t i = 0; i < kEventCount; ++i) {
        EXPECT_NE(state.event(static_cast<KernelEventKind>(i)), nullptr);
    }

    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_destroyed, fake.events_created);
    // Idempotent close.
    EXPECT_EQ(state.close(), 0);
}

TEST(KernelExecutionState, CloseOnNewContextIsANoOpSuccess) {
    KernelExecutionState state;
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
}

TEST(KernelExecutionState, RepeatedInitializeRejected) {
    FakeContextOps fake;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.phase(), KernelContextPhase::Collecting);
}

TEST(KernelExecutionState, PreEnqueueValidationLeavesStateUnchanged) {
    FakeContextOps fake;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(-1, fake.table()), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    KernelContextOps incomplete = fake.table();
    incomplete.destroy_event = nullptr;
    EXPECT_EQ(state.initialize(3, incomplete), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(fake.streams_created, 0);
    EXPECT_EQ(fake.events_created, 0);
    // The context stays usable after the clean rejections.
    EXPECT_EQ(state.initialize(3, fake.table()), 0);
}

TEST(KernelExecutionState, DeviceMismatchRejectedBeforeAnyCreation) {
    FakeContextOps fake;
    fake.current_device = 7;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(fake.streams_created, 0);
    EXPECT_EQ(fake.events_created, 0);
}

TEST(KernelExecutionState, PartialInitFailureRollsBackCleanly) {
    FakeContextOps fake;
    fake.create_event_rc_after = 1;  // second event creation fails
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), -44);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_destroyed, fake.events_created);
    // The rolled-back context is reusable.
    fake.create_event_rc_after = -1;
    EXPECT_EQ(state.initialize(3, fake.table()), 0);
}

TEST(KernelExecutionState, StreamCreateFailureRollsBackCleanly) {
    FakeContextOps fake;
    fake.create_stream_rc_after = 1;  // second hidden-stream creation fails
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), -42);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_created, 0);
}

TEST(KernelExecutionState, GetCurrentDeviceFailurePropagates) {
    FakeContextOps fake;
    fake.get_device_rc = -51;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, fake.table()), -51);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(fake.streams_created, 0);
    EXPECT_EQ(fake.events_created, 0);
}

TEST(KernelExecutionState, InitFailureWithFailedCleanupLandsInClosing) {
    FakeContextOps fake;
    fake.create_event_rc_after = 0;       // first event creation fails
    fake.destroy_failures_remaining = 1;  // the rollback's first destroy also fails
    KernelExecutionState state;
    // The reported error is the create failure; the cleanup failure is
    // latched in the separate teardown slot.
    EXPECT_EQ(state.initialize(3, fake.table()), -44);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_EQ(state.unexpected_teardown_error(), -43);
    EXPECT_TRUE(state.has_live_resources());
    EXPECT_FALSE(state.accepts_dispatch());
    // The kept ops table lets an explicit close retry release the remainder.
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
    EXPECT_FALSE(state.has_live_resources());
}

TEST(KernelExecutionState, ReadyEnqueuedKeepsAdmissionOpen) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    EXPECT_EQ(state.mark_ready_enqueued(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::ReadyEnqueued);
    EXPECT_EQ(state.mark_ready_enqueued(), 0);  // append admission stays open
    EXPECT_TRUE(state.accepts_dispatch());
    EXPECT_TRUE(state.accepts_launch());  // no prepared resources to freeze
}

TEST(KernelExecutionState, MarkReadyEnqueuedRejectedOffPath) {
    KernelExecutionState state;
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
}

TEST(KernelExecutionState, PoisonRejectsDispatchButAllowsClose) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    ASSERT_EQ(state.mark_ready_enqueued(), 0);
    state.poison(-7);
    EXPECT_EQ(state.phase(), KernelContextPhase::Poisoned);
    EXPECT_FALSE(state.accepts_dispatch());
    EXPECT_EQ(state.last_runtime_error(), -7);
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
    // First poison cause wins; later ones do not overwrite it.
    state.poison(-8);
    EXPECT_EQ(state.last_runtime_error(), -7);
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
}

TEST(KernelExecutionState, PoisonOutsideDispatchPhasesIsIgnored) {
    KernelExecutionState state;
    state.poison(-9);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(state.last_runtime_error(), 0);
}

TEST(KernelExecutionState, PoisonDuringClosingIsIgnored) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    fake.destroy_failures_remaining = 1;
    ASSERT_EQ(state.close(), -43);
    ASSERT_EQ(state.phase(), KernelContextPhase::Closing);
    state.poison(-9);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_EQ(state.last_runtime_error(), 0);
}

TEST(KernelExecutionState, PoisonAfterCloseIsIgnored) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    ASSERT_EQ(state.close(), 0);
    state.poison(-9);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
    EXPECT_EQ(state.last_runtime_error(), 0);
}

TEST(KernelExecutionState, InitializeRejectedFromEveryNonNewPhase) {
    FakeContextOps fake;

    KernelExecutionState ready;
    ASSERT_EQ(ready.initialize(3, fake.table()), 0);
    ASSERT_EQ(ready.mark_ready_enqueued(), 0);
    EXPECT_EQ(ready.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(ready.phase(), KernelContextPhase::ReadyEnqueued);

    KernelExecutionState poisoned;
    ASSERT_EQ(poisoned.initialize(3, fake.table()), 0);
    poisoned.poison(-7);
    EXPECT_EQ(poisoned.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(poisoned.phase(), KernelContextPhase::Poisoned);

    KernelExecutionState closing;
    ASSERT_EQ(closing.initialize(3, fake.table()), 0);
    fake.destroy_failures_remaining = 1;
    ASSERT_EQ(closing.close(), -43);
    ASSERT_EQ(closing.phase(), KernelContextPhase::Closing);
    EXPECT_EQ(closing.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(closing.phase(), KernelContextPhase::Closing);

    KernelExecutionState closed;
    EXPECT_EQ(closed.close(), 0);
    EXPECT_EQ(closed.initialize(3, fake.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(closed.phase(), KernelContextPhase::Closed);
}

TEST(KernelExecutionState, ClosingIsStickyAndRetriable) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    fake.destroy_failures_remaining = 2;
    const int rc = state.close();
    EXPECT_EQ(rc, -43);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_FALSE(state.accepts_dispatch());
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(state.has_live_resources());
    EXPECT_EQ(state.unexpected_teardown_error(), -43);

    // Retry succeeds and releases only what is still live.
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closed);
    EXPECT_FALSE(state.has_live_resources());
    EXPECT_EQ(fake.streams_destroyed, fake.streams_created);
    EXPECT_EQ(fake.events_destroyed, fake.events_created);
}

TEST(KernelExecutionState, TeardownErrorSlotIsSeparateFromRuntimeError) {
    FakeContextOps fake;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, fake.table()), 0);
    state.poison(-7);
    fake.destroy_failures_remaining = 1;
    EXPECT_EQ(state.close(), -43);
    // The controlled poison cause and the real teardown failure are reported
    // through separate slots; neither masks the other.
    EXPECT_EQ(state.last_runtime_error(), -7);
    EXPECT_EQ(state.unexpected_teardown_error(), -43);
    EXPECT_EQ(state.close(), 0);
    // The first teardown failure stays latched after a successful retry.
    EXPECT_EQ(state.unexpected_teardown_error(), -43);
}

struct FakeResourceOps {
    int allocations{0};
    int releases{0};
    int release_failures{0};
    int allocation_failure_at{0};

    KernelResourceOps table() {
        return {
            this,
            [](void *context, size_t bytes) -> void * {
                auto &fake = *static_cast<FakeResourceOps *>(context);
                ++fake.allocations;
                if (fake.allocations == fake.allocation_failure_at) return nullptr;
                return std::malloc(bytes);
            },
            [](void *context, void *ptr) {
                auto &fake = *static_cast<FakeResourceOps *>(context);
                if (fake.release_failures > 0) {
                    --fake.release_failures;
                    return -71;
                }
                ++fake.releases;
                std::free(ptr);
                return 0;
            },
        };
    }
};

KernelResourceLayout resource_layout() {
    KernelResourceLayout layout;
    layout.schema = 41;
    layout.contract.abi_version = PTO_PIPELINE_CONTRACT_ABI_VERSION;
    layout.contract.pipeline_depth = 1;
    layout.contract.resource_count = 3;
    layout.contract.resources[0] = {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_DEVICE_SCRATCH, 4096};
    layout.contract.resources[1] = {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    layout.contract.resources[2] = {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0};
    layout.regions = {{PTO_PIPELINE_GM_HEAP, 0, 4096}};
    return layout;
}

TEST(KernelExecutionStateResources, InvalidLayoutRejectsBeforeAllocation) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    auto layout = resource_layout();
    layout.regions[0].offset = 4096;
    EXPECT_EQ(state.prepare_resources(layout, memory.table()), PTO_RUNTIME_ERR_INTERNAL);
    layout = resource_layout();
    layout.regions.push_back(layout.regions[0]);
    EXPECT_EQ(state.prepare_resources(layout, memory.table()), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(memory.allocations, 0);
    EXPECT_EQ(state.close(), 0);
}

TEST(KernelExecutionStateResources, FrozenCapacityCannotGrowOrChangeSchemaAndReusesOriginalAddress) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    ASSERT_EQ(state.freeze_resources(), 0);
    auto layout = resource_layout();
    ++layout.schema;
    EXPECT_EQ(state.prepare_resources(layout, memory.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    layout = resource_layout();
    layout.regions[0].bytes += 1;
    layout.contract.resources[0].bytes_per_copy += 1;
    EXPECT_EQ(state.prepare_resources(layout, memory.table()), PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE);
    EXPECT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    EXPECT_EQ(memory.allocations, 1);
    EXPECT_EQ(state.close(), 0);
}

TEST(KernelExecutionStateResources, RepeatedPrepareRejectsChangedRegionOffset) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    auto layout = resource_layout();
    layout.contract.resources[0].bytes_per_copy = 8192;
    ASSERT_EQ(state.prepare_resources(layout, memory.table()), 0);
    ASSERT_EQ(state.freeze_resources(), 0);
    layout.regions[0].offset = 1024;
    EXPECT_EQ(state.prepare_resources(layout, memory.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(memory.allocations, 1);
    EXPECT_EQ(state.phase(), KernelContextPhase::Collecting);
    EXPECT_EQ(state.close(), 0);
}

TEST(KernelExecutionStateResources, PartialAllocationFailureWithFailedRollbackRetainsBothResourceOwners) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    auto layout = resource_layout();
    layout.contract.resource_count = 4;
    layout.contract.resources[3] = {PTO_PIPELINE_GM_SM, PTO_PIPELINE_DEVICE_SCRATCH, 4096};
    layout.regions.push_back({PTO_PIPELINE_GM_SM, 0, 4096});
    memory.allocation_failure_at = 2;
    memory.release_failures = 1;
    EXPECT_EQ(state.prepare_resources(layout, memory.table()), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_EQ(state.unexpected_teardown_error(), -71);
    EXPECT_EQ(memory.releases, 0);
    EXPECT_EQ(handles.streams_destroyed, 0);
    EXPECT_EQ(handles.events_destroyed, 0);
    EXPECT_TRUE(state.has_live_resources());
    EXPECT_EQ(state.close(), 0);
    EXPECT_EQ(memory.releases, 1);
    EXPECT_EQ(handles.streams_destroyed, 2);
    EXPECT_EQ(handles.events_destroyed, 5);
}

TEST(KernelExecutionStateResources, ZeroGenerationIsRejectedBeforeCreatingHandles) {
    FakeContextOps handles;
    KernelExecutionState state;
    EXPECT_EQ(state.initialize(3, handles.table(), 0), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(state.phase(), KernelContextPhase::New);
    EXPECT_EQ(handles.streams_created, 0);
    EXPECT_EQ(handles.events_created, 0);
}

TEST(KernelExecutionStateResources, FrozenBindingChecksGenerationAndPreservesAddressAfterRefusal) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    ASSERT_TRUE(state.resources_prepared());
    ASSERT_EQ(state.freeze_resources(), 0);
    ASSERT_TRUE(state.resources_frozen());
    const uint64_t required[] = {4096};
    KernelResourceBinding binding;
    EXPECT_EQ(state.bind_resources_for_launch(3, 19, 41, required, 1, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(state.inspect_frozen_resources(3, 19, 41, required, 1, binding), 0);
    const auto address = binding.regions[0].address;
    ASSERT_NE(address, 0u);
    ASSERT_EQ(state.mark_ready_enqueued(), 0);
    EXPECT_EQ(state.bind_resources_for_launch(3, 18, 41, required, 1, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(state.bind_resources_for_launch(4, 19, 41, required, 1, binding), PTO_RUNTIME_ERR_INVALID_STATE);
    const uint64_t too_large[] = {4097};
    EXPECT_EQ(state.bind_resources_for_launch(3, 19, 41, too_large, 1, binding), PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE);
    ASSERT_EQ(state.bind_resources_for_launch(3, 19, 41, required, 1, binding), 0);
    EXPECT_EQ(binding.regions[0].address, address);
    EXPECT_EQ(memory.allocations, 1);
    EXPECT_EQ(memory.releases, 0);
    ASSERT_EQ(state.close(), 0);
    EXPECT_EQ(memory.releases, 1);
    // The binding owns its snapshot, so closing the state cannot dangle it.
    EXPECT_EQ(binding.regions.size(), 1U);
    EXPECT_EQ(binding.regions[0].address, address);
}

TEST(KernelExecutionStateResources, MemoryCloseFailureKeepsHandlesForRetry) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    memory.release_failures = 1;
    EXPECT_EQ(state.close(), -71);
    EXPECT_EQ(state.phase(), KernelContextPhase::Closing);
    EXPECT_TRUE(state.has_live_resources());
    EXPECT_EQ(state.unexpected_teardown_error(), -71);
    EXPECT_EQ(handles.streams_destroyed, 0);
    EXPECT_EQ(handles.events_destroyed, 0);
    EXPECT_EQ(state.prepare_resources(resource_layout(), memory.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(state.close(), 0);
    EXPECT_EQ(memory.releases, 1);
    EXPECT_EQ(handles.streams_destroyed, handles.streams_created);
    EXPECT_EQ(handles.events_destroyed, handles.events_created);
    EXPECT_FALSE(state.has_live_resources());
}

TEST(KernelExecutionStateResources, RepeatedPrepareFollowsRequestedRegionSizes) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    auto layout = resource_layout();
    ASSERT_EQ(state.prepare_resources(layout, memory.table()), 0);
    ASSERT_EQ(state.freeze_resources(), 0);
    const uint64_t full[] = {4096};
    KernelResourceBinding binding;
    ASSERT_EQ(state.inspect_frozen_resources(3, 19, 41, full, 1, binding), 0);
    const auto address = binding.regions[0].address;
    ASSERT_NE(address, 0u);
    EXPECT_EQ(binding.regions[0].capacity, 4096U);

    layout.regions[0].bytes = 2048;
    ASSERT_EQ(state.prepare_resources(layout, memory.table()), 0);
    const uint64_t half[] = {2048};
    ASSERT_EQ(state.inspect_frozen_resources(3, 19, 41, half, 1, binding), 0);
    EXPECT_EQ(binding.regions[0].address, address);
    EXPECT_EQ(binding.regions[0].capacity, 2048U);

    layout.regions[0].bytes = 0;
    ASSERT_EQ(state.prepare_resources(layout, memory.table()), 0);
    const uint64_t none[] = {0};
    ASSERT_EQ(state.inspect_frozen_resources(3, 19, 41, none, 1, binding), 0);
    EXPECT_EQ(binding.regions[0].address, 0U);
    EXPECT_EQ(binding.regions[0].capacity, 0U);

    layout.regions[0].bytes = 4096;
    ASSERT_EQ(state.prepare_resources(layout, memory.table()), 0);
    ASSERT_EQ(state.inspect_frozen_resources(3, 19, 41, full, 1, binding), 0);
    EXPECT_EQ(binding.regions[0].address, address);
    EXPECT_EQ(binding.regions[0].capacity, 4096U);
    EXPECT_EQ(memory.allocations, 1);
    EXPECT_EQ(memory.releases, 0);
    ASSERT_EQ(state.close(), 0);
    EXPECT_EQ(memory.releases, 1);
}

TEST(KernelExecutionStateResources, LaunchReadinessRequiresReadyAndFrozenResources) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    EXPECT_TRUE(state.accepts_dispatch());
    EXPECT_FALSE(state.accepts_launch());
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    // Prepared but not frozen: mark_ready_enqueued must refuse to make the
    // context launch-ready.
    EXPECT_EQ(state.mark_ready_enqueued(), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(state.freeze_resources(), 0);
    EXPECT_FALSE(state.accepts_launch());
    ASSERT_EQ(state.mark_ready_enqueued(), 0);
    EXPECT_TRUE(state.accepts_launch());
    EXPECT_TRUE(state.accepts_dispatch());
    const uint64_t required[] = {4096};
    KernelResourceBinding binding;
    ASSERT_EQ(state.bind_resources_for_launch(3, 19, 41, required, 1, binding), 0);
    ASSERT_EQ(state.close(), 0);
}

TEST(KernelExecutionStateResources, FirstPreparationAfterLaunchReadyIsRejected) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    ASSERT_EQ(state.mark_ready_enqueued(), 0);
    EXPECT_TRUE(state.accepts_launch());
    EXPECT_EQ(state.prepare_resources(resource_layout(), memory.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(memory.allocations, 0);
    ASSERT_EQ(state.close(), 0);
}

TEST(KernelExecutionStateResources, LaunchBindingAndBinderShareTheReadyState) {
    namespace kl = simpler::kernel_launch;
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    ASSERT_EQ(state.freeze_resources(), 0);
    ASSERT_EQ(state.mark_ready_enqueued(), 0);
    ASSERT_TRUE(state.accepts_launch());
    const uint64_t required[] = {4096};
    KernelResourceBinding binding;
    ASSERT_EQ(state.bind_resources_for_launch(3, 19, 41, required, 1, binding), 0);

    struct GateState {
        int finishes{0};
    } gate_state;
    kl::KernelLaunchGateOps gate;
    gate.context = &gate_state;
    gate.acquire = [](void *, const kl::KernelInvocationBinding &, void *, kl::KernelLaunchAdmission *out) noexcept {
        out->handles = {
            nullptr,
            reinterpret_cast<void *>(1),
            reinterpret_cast<void *>(2),
            reinterpret_cast<void *>(3),
            reinterpret_cast<void *>(4),
            reinterpret_cast<void *>(5),
            reinterpret_cast<void *>(6),
            reinterpret_cast<void *>(7)
        };
        out->previous_caller_identity = 0;
        return 0;
    };
    gate.finish = [](void *p, const kl::KernelLaunchResult &result) noexcept {
        if (result.status == 0) ++static_cast<GateState *>(p)->finishes;
    };
    gate.query_tail = [](void *, void *, bool *complete) noexcept {
        *complete = true;
        return 0;
    };
    kl::KernelLaunchOps ops;
    ops.wait_event = [](void *, void *, void *) noexcept {
        return 0;
    };
    ops.memset_handshake = [](void *, void *) noexcept {
        return 0;
    };
    ops.record_event = [](void *, void *, void *) noexcept {
        return 0;
    };
    ops.launch_aicpu = [](void *, void *) noexcept {
        return 0;
    };
    ops.launch_aicore = [](void *, void *) noexcept {
        return 0;
    };
    std::vector<uint8_t> packet(1, 0);
    const auto result =
        kl::launch_bound_kernel({packet.data(), packet.size(), nullptr, 0}, reinterpret_cast<void *>(100), gate, ops);
    EXPECT_EQ(result.status, 0);
    EXPECT_EQ(gate_state.finishes, 1);
    ASSERT_EQ(state.close(), 0);
}

TEST(KernelExecutionStateResources, PreparationChecksTheBorrowedCurrentDeviceBeforeAllocating) {
    FakeContextOps handles;
    FakeResourceOps memory;
    KernelExecutionState state;
    ASSERT_EQ(state.initialize(3, handles.table(), 19), 0);
    handles.current_device = 4;
    EXPECT_EQ(state.prepare_resources(resource_layout(), memory.table()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(memory.allocations, 0);
    handles.current_device = 3;
    ASSERT_EQ(state.prepare_resources(resource_layout(), memory.table()), 0);
    EXPECT_EQ(memory.allocations, 1);
    EXPECT_EQ(state.close(), 0);
}

}  // namespace
