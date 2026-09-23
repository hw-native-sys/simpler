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
#include <atomic>
#include <memory>
#include <thread>

#include "runtime.h"
#include "scheduler/scheduler_context.h"

namespace {
constexpr int kCores = 6;
std::array<std::atomic<int>, kCores> retirements{};
}  // namespace

uint64_t platform_aicore_exit_deadline() { return 1; }
uint64_t read_reg(uint64_t, RegId) { return AICORE_EXITED_VALUE; }
int32_t platform_retire_aicore_group(const uint64_t *addresses, size_t count, uint64_t, bool *released) {
    for (size_t i = 0; i < count; ++i) {
        EXPECT_GE(addresses[i], 1U);
        EXPECT_LE(addresses[i], kCores);
        if (addresses[i] >= 1 && addresses[i] <= kCores) ++retirements[addresses[i] - 1];
        if (released != nullptr) released[i] = true;
    }
    return 0;
}

class SchedulerRetirementTestPeer {
public:
    static void open_cores(SchedulerContext &context) {
        for (int i = 0; i < kCores; ++i)
            context.core_exec_states_[i].reg_addr = i + 1;
    }
    static void start_assignment(SchedulerContext &context, int owner) { context.core_trackers_[owner].init(1); }
    static bool fatal(const SchedulerContext &context) { return context.fatal_shutdown_started_.load(); }
    static void fail_handshake(SchedulerContext &context) { context.handshake_failed_.store(true); }
    static void set_core_types(SchedulerContext &context) {
        for (int i = 0; i < kCores; ++i) {
            context.core_type_compact_[i] = static_cast<uint8_t>(i < 2 ? CoreType::AIC : CoreType::AIV);
        }
    }
};

class SchedulerRetirement : public testing::Test {
protected:
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    std::unique_ptr<Runtime> runtime = std::make_unique<Runtime>();

    void SetUp() override {
        for (auto &count : retirements)
            count.store(0);
        runtime->dev.worker_count = kCores;
        runtime->dev.serial_orch_sched = false;
        ASSERT_EQ(context->pre_handshake_init(runtime.get(), 3, 2, 0), 0);
    }
    void expect_once() {
        for (int i = 0; i < kCores; ++i)
            EXPECT_EQ(retirements[i].load(), 1) << "core " << i;
    }
};

TEST_F(SchedulerRetirement, EmergencyBeforeInitializationRetiresLateCores) {
    context->abort_and_shutdown(runtime.get());
    ASSERT_TRUE(context->is_completed());
    ASSERT_TRUE(SchedulerRetirementTestPeer::fatal(*context));
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    // Failed init need not reach run()/shutdown(): assignment must honor the request.
    expect_once();
    EXPECT_EQ(context->shutdown(0), 0);
    EXPECT_EQ(context->shutdown(1), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, EmergencyDoesNotReadPartiallyAssignedTrackers) {
    SchedulerRetirementTestPeer::open_cores(*context);
    SchedulerRetirementTestPeer::start_assignment(*context, 1);
    context->abort_and_shutdown(runtime.get());
    for (auto &count : retirements)
        EXPECT_EQ(count.load(), 0);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    expect_once();
}

TEST_F(SchedulerRetirement, NormalAndEmergencyRetireEachReadyCoreOnce) {
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    std::thread normal([&] {
        context->shutdown(0);
    });
    std::thread emergency([&] {
        context->abort_and_shutdown(runtime.get());
    });
    context->shutdown(1);
    normal.join();
    emergency.join();
    expect_once();
}

TEST_F(SchedulerRetirement, AssignmentAndEmergencyCanPublishConcurrently) {
    SchedulerRetirementTestPeer::open_cores(*context);
    std::thread first([&] {
        context->assign_own_clusters(0);
    });
    std::thread second([&] {
        context->assign_own_clusters(1);
    });
    context->abort_and_shutdown(runtime.get());
    first.join();
    second.join();
    expect_once();
}

TEST_F(SchedulerRetirement, SerialHandshakeFailureUsesOneFallbackOwner) {
    runtime->dev.serial_orch_sched = true;
    ASSERT_EQ(context->pre_handshake_init(runtime.get(), 3, 2, 0), 0);
    SchedulerRetirementTestPeer::open_cores(*context);
    SchedulerRetirementTestPeer::fail_handshake(*context);
    EXPECT_EQ(context->post_handshake_init(runtime.get()), -1);
    expect_once();
    context->retire_all_cores();
    expect_once();
}

TEST_F(SchedulerRetirement, SerialInitializationPublishesAssignedGroups) {
    runtime->dev.serial_orch_sched = true;
    ASSERT_EQ(context->pre_handshake_init(runtime.get(), 3, 2, 0), 0);
    SchedulerRetirementTestPeer::open_cores(*context);
    SchedulerRetirementTestPeer::set_core_types(*context);
    ASSERT_EQ(context->post_handshake_init(runtime.get()), 0);
    context->shutdown(0);
    context->shutdown(1);
    context->shutdown(2);
    expect_once();
}

TEST_F(SchedulerRetirement, NewGenerationDoesNotInheritPendingRetirement) {
    context->abort_and_shutdown(runtime.get());
    context->deinit();
    ASSERT_EQ(context->pre_handshake_init(runtime.get(), 3, 2, 0), 0);
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    for (auto &count : retirements)
        EXPECT_EQ(count.load(), 0);
    EXPECT_FALSE(context->is_completed());
    EXPECT_FALSE(SchedulerRetirementTestPeer::fatal(*context));
    context->shutdown(0);
    context->shutdown(1);
    expect_once();
}

TEST_F(SchedulerRetirement, CompletionObserverSeesFatalPublication) {
    std::thread emergency([&] {
        context->abort_and_shutdown(runtime.get());
    });
    while (!context->is_completed()) {}
    EXPECT_TRUE(SchedulerRetirementTestPeer::fatal(*context));
    emergency.join();
}
