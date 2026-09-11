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

#include <cstdint>

#include "host/execution_mode_latch.h"
#include "host/kernel_execution_state.h"

namespace {

/**
 * Fake ops backing the restricted context vocabulary. Handles are small
 * integers cast to pointers; create/destroy counters prove balance, and the
 * failure knobs drive the CLOSING-retry and partial-init paths.
 */
struct FakeContextOps {
    int current_device{3};
    int get_device_rc{0};
    int create_stream_rc_after{-1};  // fail the Nth create_hidden_stream (0-based); -1 = never
    int create_event_rc_after{-1};
    int destroy_failures_remaining{0};

    int streams_created{0};
    int streams_destroyed{0};
    int events_created{0};
    int events_destroyed{0};
    uintptr_t next_handle{1};

    static FakeContextOps *self(void *context) { return static_cast<FakeContextOps *>(context); }

    static int get_current_device(void *context, int *device_id) noexcept {
        auto *ops = self(context);
        if (ops->get_device_rc != 0) return ops->get_device_rc;
        *device_id = ops->current_device;
        return 0;
    }
    static int create_hidden_stream(void *context, void **stream) noexcept {
        auto *ops = self(context);
        if (ops->create_stream_rc_after >= 0 && ops->streams_created == ops->create_stream_rc_after) return -42;
        ops->streams_created++;
        *stream = reinterpret_cast<void *>(ops->next_handle++);
        return 0;
    }
    static int destroy_hidden_stream(void *context, void *) noexcept {
        auto *ops = self(context);
        if (ops->destroy_failures_remaining > 0) {
            ops->destroy_failures_remaining--;
            return -43;
        }
        ops->streams_destroyed++;
        return 0;
    }
    static int create_event(void *context, void **event) noexcept {
        auto *ops = self(context);
        if (ops->create_event_rc_after >= 0 && ops->events_created == ops->create_event_rc_after) return -44;
        ops->events_created++;
        *event = reinterpret_cast<void *>(ops->next_handle++);
        return 0;
    }
    static int destroy_event(void *context, void *) noexcept {
        auto *ops = self(context);
        if (ops->destroy_failures_remaining > 0) {
            ops->destroy_failures_remaining--;
            return -43;
        }
        ops->events_destroyed++;
        return 0;
    }

    KernelContextOps table() {
        return KernelContextOps{this,          &get_current_device, &create_hidden_stream, &destroy_hidden_stream,
                                &create_event, &destroy_event};
    }
};

constexpr size_t kStreamCount = static_cast<size_t>(KernelStreamKind::Count);
constexpr size_t kEventCount = static_cast<size_t>(KernelEventKind::Count);

TEST(KernelContextOpsVocabulary, IncompleteTableIsInvalid) {
    FakeContextOps fake;
    KernelContextOps ops = fake.table();
    EXPECT_TRUE(ops.valid());
    ops.create_event = nullptr;
    EXPECT_FALSE(ops.valid());
}

TEST(KernelLaunchOpsVocabulary, IncompleteTableIsInvalid) {
    KernelLaunchOps ops{};
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
    EXPECT_TRUE(state.has_live_resources());

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

}  // namespace
