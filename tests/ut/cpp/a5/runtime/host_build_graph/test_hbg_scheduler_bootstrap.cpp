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
// The host's scheduler selection and its bootstrap address live in
// DeviceRuntimeLaunchDesc::scheduler_bootstrap, not on any worker's handshake.
// These cases cover what that separation is for: a selection that survives a
// mode change, the address each tier derives from the published base, and the
// pre-READY view staying distinct from the one the AICPU publishes at hand-off.
//
// This is logic only. Nothing here observes a cache, so no case states anything
// about device visibility.
#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

#include "aicore_scheduler_state.h"
#include "runtime.h"
#include "scheduler/scheduler_types.h"

namespace {

constexpr uint64_t kStateBase = 0x7f0000000000ULL;
constexpr uint64_t kWorkerContextsOffset = 4096;
constexpr uint64_t kContextBase = kStateBase + kWorkerContextsOffset;

class SchedulerBootstrapFields : public ::testing::Test {
protected:
    void SetUp() override { runtime.set_worker_count(8); }

    // A resident selection, as the host publishes it once its scheduler state is
    // allocated.
    void select_resident() {
        runtime.publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, kContextBase);
    }

    Runtime runtime;
};

TEST_F(SchedulerBootstrapFields, AFreshRuntimePublishesNoSelection) {
    EXPECT_EQ(runtime.dev.scheduler_bootstrap.runtime_mode, 0u);
    EXPECT_EQ(runtime.dev.scheduler_bootstrap.worker_context_base, 0u);
    EXPECT_FALSE(aicore_scheduler_runtime_enabled(&runtime));
    EXPECT_EQ(aicore_scheduler_bootstrap_context(&runtime), nullptr);
}

TEST_F(SchedulerBootstrapFields, ReleaseLeavesNoModeAndNoBase) {
    select_resident();
    runtime.publish_scheduler_bootstrap(0, 0);

    EXPECT_EQ(runtime.dev.scheduler_bootstrap.runtime_mode, 0u);
    EXPECT_EQ(runtime.dev.scheduler_bootstrap.worker_context_base, 0u);
    EXPECT_FALSE(aicore_scheduler_runtime_enabled(&runtime));
}

TEST_F(SchedulerBootstrapFields, ResidentSelectionAfterReleaseUsesTheCurrentBase) {
    select_resident();
    EXPECT_EQ(aicore_scheduler_bootstrap_context(&runtime), reinterpret_cast<SchedulerWorkerContext *>(kContextBase));

    runtime.publish_scheduler_bootstrap(0, 0);
    ASSERT_EQ(aicore_scheduler_bootstrap_context(&runtime), nullptr);

    // A second bind allocates its own state, so the base moves. Nothing of the
    // first resident run may survive into it.
    const uint64_t second_base = kContextBase + 0x100000;
    runtime.publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, second_base);
    EXPECT_TRUE(aicore_scheduler_runtime_enabled(&runtime));
    EXPECT_EQ(aicore_scheduler_bootstrap_context(&runtime), reinterpret_cast<SchedulerWorkerContext *>(second_base));
}

TEST_F(SchedulerBootstrapFields, RepeatedResidentSelectionRepublishesTheCurrentBase) {
    for (uint64_t run = 0; run < 4; ++run) {
        const uint64_t base = kContextBase + run * 0x10000;
        runtime.publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, base);
        ASSERT_TRUE(aicore_scheduler_runtime_enabled(&runtime)) << "run " << run;
        ASSERT_EQ(aicore_scheduler_bootstrap_context(&runtime), reinterpret_cast<SchedulerWorkerContext *>(base))
            << "run " << run << " read a base from an earlier run";
    }
}

// The AICPU accepts a resident mode only with a base to go with it. A mode
// published without one is the case the host's paired write exists to prevent,
// and it must not produce a context pointer.
TEST_F(SchedulerBootstrapFields, AResidentModeWithoutABaseYieldsNoContext) {
    runtime.publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, 0);
    EXPECT_FALSE(aicore_scheduler_runtime_enabled(&runtime));
    EXPECT_EQ(aicore_scheduler_bootstrap_context(&runtime), nullptr);
}

// A base with no resident mode is the mirror case: the AICPU must not follow it
// just because it is non-zero.
TEST_F(SchedulerBootstrapFields, ABaseWithoutAResidentModeYieldsNoContext) {
    runtime.publish_scheduler_bootstrap(UINT32_MAX, kContextBase);
    EXPECT_EQ(aicore_scheduler_bootstrap_context(&runtime), nullptr);
}

TEST_F(SchedulerBootstrapFields, InvalidBaseExtentNeverNamesAContext) {
    for (uint64_t base : {UINT64_C(1), kContextBase + 1, UINT64_MAX - UINT64_C(63)}) {
        SCOPED_TRACE(base);
        runtime.publish_scheduler_bootstrap(SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING, base);
        EXPECT_FALSE(aicore_scheduler_runtime_enabled(&runtime));
        EXPECT_EQ(aicore_scheduler_bootstrap_context(&runtime), nullptr);
    }
}

TEST_F(SchedulerBootstrapFields, AnOversizedWorkerArrayNeverNamesAContext) {
    select_resident();
    runtime.set_worker_count(SCHEDULER_WORKER_CAPACITY + 1);
    EXPECT_FALSE(aicore_scheduler_runtime_enabled(&runtime));
    EXPECT_EQ(aicore_scheduler_bootstrap_context(&runtime), nullptr);
}

TEST_F(SchedulerBootstrapFields, NoSelectionIsHonouredWithNoActiveWorkers) {
    select_resident();
    runtime.set_worker_count(0);
    EXPECT_FALSE(aicore_scheduler_runtime_enabled(&runtime));
    EXPECT_EQ(aicore_scheduler_bootstrap_context(&runtime), nullptr);
}

// A worker's handshake line and the bootstrap line are distinct storage, so an
// AICore whole-line write-back over its own handshake cannot reach the host's
// selection. The layout is what makes that true, so it is asserted here rather
// than assumed.
TEST_F(SchedulerBootstrapFields, TheBootstrapLineIsDisjointFromEveryHandshakeLine) {
    const auto base_of = [](const void *p) {
        return reinterpret_cast<uintptr_t>(p) / 64;
    };
    const uintptr_t bootstrap_line = base_of(&runtime.dev.scheduler_bootstrap);

    ASSERT_EQ(sizeof(SchedulerBootstrapInputs), 64u);
    ASSERT_EQ(reinterpret_cast<uintptr_t>(&runtime.dev.scheduler_bootstrap) % 64, 0u);

    for (int32_t worker = 0; worker < RUNTIME_MAX_WORKER; ++worker) {
        ASSERT_NE(base_of(&runtime.dev.workers[worker]), bootstrap_line) << "worker " << worker;
    }
}

}  // namespace
