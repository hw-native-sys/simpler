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
#include <chrono>
#include <condition_variable>
#include <memory>
#include <mutex>
#include <thread>

#include "runtime.h"
#include "scheduler/scheduler_context.h"

namespace {
constexpr int kCores = 6;
std::array<std::atomic<int>, kCores> opened{};
std::array<std::atomic<int>, kCores> retired{};
std::array<uint32_t, kCores> conditions{};
std::atomic<int> batches{0};
std::atomic<int> deadlines{0};
std::mutex init_mutex;
std::condition_variable init_cv;
bool pause_open = false;
bool open_paused = false;
bool resume_open = false;
bool silent_first_core = false;

template <typename Context>
auto initialize(Context &context, Runtime *runtime, uint64_t regs, int)
    -> decltype(context.pre_handshake_init(runtime, 3, 2, regs)) {
    return context.pre_handshake_init(runtime, 3, 2, regs);
}
template <typename Context>
auto initialize(Context &context, Runtime *runtime, uint64_t regs, long)
    -> decltype(context.pre_handshake_init(runtime, 3, regs)) {
    return context.pre_handshake_init(runtime, 3, regs);
}

template <typename Context>
auto handshake_owned(Context &context, Runtime *runtime, int owner, int)
    -> decltype(context.handshake_owned_clusters(runtime, owner, 2), bool()) {
    context.handshake_owned_clusters(runtime, owner, 2);
    context.assign_own_clusters(runtime, owner);
    return true;
}
template <typename Context>
bool handshake_owned(Context &, Runtime *, int, long) {
    return false;
}

template <typename Context>
auto handshake_owned_only(Context &context, Runtime *runtime, int owner, int)
    -> decltype(context.handshake_owned_clusters(runtime, owner, 2), bool()) {
    context.handshake_owned_clusters(runtime, owner, 2);
    return true;
}

template <typename Context>
bool handshake_owned_only(Context &, Runtime *, int, long) {
    return false;
}

template <typename Context>
auto assign_owned(Context &context, Runtime *runtime, int owner, int)
    -> decltype(context.assign_own_clusters(runtime, owner), bool()) {
    context.assign_own_clusters(runtime, owner);
    return true;
}

template <typename Context>
bool assign_owned(Context &, Runtime *, int, long) {
    return false;
}
}  // namespace

uint32_t platform_get_physical_cores_count() { return kCores; }
extern "C" uint64_t get_platform_run_result_epoch() { return 0; }
uint64_t platform_aicore_exit_deadline() { return ++deadlines; }
uint64_t read_reg(uint64_t, RegId) { return AICORE_IDLE_VALUE; }
volatile uint32_t *get_reg_ptr(uint64_t address, RegId) { return &conditions.at(address - 1); }

#ifdef PTO_ASYNC_WAIT_H
void AsyncWaitList::log_diagnostics(AICoreCompletionMailbox *, const char *, bool) {}
#endif

void platform_init_aicore_regs(uint64_t address) {
    ++opened.at(address - 1);
    std::unique_lock<std::mutex> lock(init_mutex);
    if (pause_open && address == kCores) {
        open_paused = true;
        init_cv.notify_all();
        init_cv.wait(lock, [] {
            return resume_open;
        });
    }
}

int32_t platform_retire_aicore_group(const AicoreExitTarget *targets, size_t count, uint64_t, bool *released) {
    ++batches;
    bool timed_out = false;
    for (size_t i = 0; i < count; ++i) {
        const auto core = targets[i].reg_addr - 1;
        EXPECT_LT(core, kCores);
        if (core >= kCores) continue;
        EXPECT_EQ(opened[core].load(), 1);
        EXPECT_NE(targets[i].teardown, nullptr);
        ++retired[core];
        const bool success = !(silent_first_core && core == 0);
        if (released != nullptr) released[i] = success;
        if (success && targets[i].teardown != nullptr) {
            __atomic_store_n(&targets[i].teardown->post_close_release, AICORE_POST_CLOSE_RELEASE, __ATOMIC_RELEASE);
        }
        timed_out |= !success;
    }
    return timed_out ? -1 : 0;
}

class SchedulerContextTestPeer {
public:
    static int32_t request_all(SchedulerContext &context, Runtime *runtime) {
        return context.retire_all_cores(runtime);
    }
    static void emergency(SchedulerContext &context, Runtime *runtime) { context.emergency_shutdown(runtime); }
    static bool handshake_failed(const SchedulerContext &context) {
        return context.handshake_failed_.load(std::memory_order_acquire);
    }
    static int32_t request(SchedulerContext &context, Runtime *runtime, const int32_t *ids, int count) {
        return context.retire_cores(runtime, ids, count);
    }
    static void clear_register_address(SchedulerContext &context, int core) {
        context.core_exec_states_[core].reg_addr = 0;
    }
    static void clear_ownership(SchedulerContext &context) {
        context.core_trackers_[0].init(0);
        context.core_trackers_[1].init(0);
    }
    static void force_assignment_failure(SchedulerContext &context) {
        // Fault-injected discovery geometry stays within the allocated tables.
        context.cores_total_num_ = PLATFORM_MAX_CORES;
        context.aicpu_thread_num_ = 2;
        for (int i = 0; i < PLATFORM_MAX_CORES; ++i)
            context.core_type_compact_[i] = static_cast<uint8_t>(CoreType::AIC);
    }
    static void assign(SchedulerContext &context) {
        context.core_trackers_[0].init(1);
        context.core_trackers_[0].set_cluster(0, 0, 2, 3);
        context.core_trackers_[1].init(1);
        context.core_trackers_[1].set_cluster(0, 1, 4, 5);
    }
};

class SchedulerRetirement : public testing::Test {
protected:
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    std::unique_ptr<Runtime> runtime = std::make_unique<Runtime>();
    std::array<uint64_t, kCores> regs{};

    void SetUp() override {
        batches = 0;
        deadlines = 0;
        pause_open = false;
        open_paused = false;
        resume_open = false;
        silent_first_core = false;
        for (int i = 0; i < kCores; ++i) {
            opened[i] = 0;
            retired[i] = 0;
            regs[i] = i + 1;
        }
        runtime->set_worker_count(kCores);
        ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
        for (int i = 0; i < kCores; ++i) {
            auto &handshake = runtime->dev.workers[i];
            handshake.physical_core_id = i;
            handshake.core_type = i < 2 ? CoreType::AIC : CoreType::AIV;
            handshake.aicore_done = 1;
        }
        SchedulerContextTestPeer::assign(*context);
    }
    void handshake() {
        context->handshake_partition(runtime.get(), 0, 1);
        const bool failed = SchedulerContextTestPeer::handshake_failed(*context);
        EXPECT_EQ(context->post_handshake_init(runtime.get()), failed ? -1 : 0);
    }
    int32_t request_all() { return SchedulerContextTestPeer::request_all(*context, runtime.get()); }
    void emergency() { SchedulerContextTestPeer::emergency(*context, runtime.get()); }
    void expect_once() {
        for (int i = 0; i < kCores; ++i) {
            EXPECT_EQ(retired[i].load(), 1) << "core " << i;
            EXPECT_EQ(
                runtime->get_teardown_gates()[i].post_close_release,
                silent_first_core && i == 0 ? 0U : AICORE_POST_CLOSE_RELEASE
            ) << "core "
              << i;
        }
    }
};

TEST_F(SchedulerRetirement, RequestBeforeHandshakeIsServicedByPublisher) {
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    handshake();
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, RequestAfterHandshakeUsesOneBatchAndDeadline) {
    handshake();
    EXPECT_EQ(batches.load(), 0);
    EXPECT_EQ(request_all(), 0);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
}

TEST_F(SchedulerRetirement, RequestAfterHandshakeWaitsForInitialization) {
    SchedulerContextTestPeer::clear_ownership(*context);
    context->handshake_partition(runtime.get(), 0, 1);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(opened[i].load(), 1);
        EXPECT_EQ(retired[i].load(), 0);
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0U);
    }
    EXPECT_EQ(context->post_handshake_init(runtime.get()), 0);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, OwnedHandshakeWaitsForAssignment) {
    SchedulerContextTestPeer::clear_ownership(*context);
    if (!handshake_owned_only(*context, runtime.get(), 0, 0)) {
        GTEST_SKIP() << "HBG has only the partition handshake";
    }
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    ASSERT_TRUE(assign_owned(*context, runtime.get(), 0, 0));
    const std::array<int, 3> owned = {0, 2, 3};
    for (int core : owned)
        EXPECT_EQ(retired[core].load(), 1);
    EXPECT_EQ(retired[1].load(), 0);
    EXPECT_EQ(retired[4].load(), 0);
    EXPECT_EQ(retired[5].load(), 0);
    EXPECT_EQ(batches.load(), 1);
    ASSERT_TRUE(handshake_owned(*context, runtime.get(), 1, 0));
    expect_once();
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, AssignmentFailurePublishesOpenedWindows) {
    SchedulerContextTestPeer::clear_ownership(*context);
    context->handshake_partition(runtime.get(), 0, 1);
    SchedulerContextTestPeer::force_assignment_failure(*context);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    EXPECT_EQ(context->post_handshake_init(runtime.get()), -1);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, EmergencyBeforeHandshakeIsServicedByPublisher) {
    emergency();
    EXPECT_EQ(batches.load(), 0);
    handshake();
    expect_once();
    emergency();
    EXPECT_EQ(request_all(), 0);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
}

TEST_F(SchedulerRetirement, RequestWhileWindowsOpenBeforeStatePublicationIsRetained) {
    pause_open = true;
    std::thread publisher([&] {
        handshake();
    });
    bool reached;
    {
        std::unique_lock<std::mutex> lock(init_mutex);
        reached = init_cv.wait_for(lock, std::chrono::seconds(5), [] {
            return open_paused;
        });
    }
    EXPECT_TRUE(reached);
    if (reached) {
        EXPECT_EQ(request_all(), 0);
        EXPECT_EQ(batches.load(), 0);
    }
    {
        std::lock_guard<std::mutex> lock(init_mutex);
        resume_open = true;
    }
    init_cv.notify_all();
    publisher.join();
    expect_once();
}

TEST_F(SchedulerRetirement, OwnedClusterPublishersServiceEarlyRequest) {
    EXPECT_EQ(request_all(), 0);
    if (!handshake_owned(*context, runtime.get(), 0, 0)) {
        GTEST_SKIP() << "HBG has only the partition handshake";
    }
    ASSERT_TRUE(handshake_owned(*context, runtime.get(), 1, 0));
    expect_once();
    EXPECT_EQ(batches.load(), 2);
    EXPECT_EQ(deadlines.load(), 2);
}

TEST_F(SchedulerRetirement, OverlappingSetsAndDuplicateIdsRetireOnlyOnce) {
    handshake();
    const int32_t subset[] = {-1, 0, 3, 0, kCores};
    EXPECT_EQ(SchedulerContextTestPeer::request(*context, runtime.get(), subset, 5), 0);
    EXPECT_EQ(request_all(), 0);
    expect_once();
    EXPECT_EQ(batches.load(), 2);
}

TEST_F(SchedulerRetirement, NormalAndEmergencyRetireReadyCoresOnce) {
    handshake();
    std::thread normal([&] {
        context->shutdown(0, runtime.get());
    });
    std::thread fatal([&] {
        emergency();
    });
    context->shutdown(1, runtime.get());
    normal.join();
    fatal.join();
    expect_once();
}

TEST_F(SchedulerRetirement, PublicationAndEmergencyCanRace) {
    std::thread publisher([&] {
        handshake();
    });
    std::thread fatal([&] {
        emergency();
    });
    publisher.join();
    fatal.join();
    expect_once();
}

TEST_F(SchedulerRetirement, PartialTimeoutDoesNotReleaseSilentCoreOrRetryClaim) {
    handshake();
    silent_first_core = true;
    EXPECT_EQ(request_all(), -1);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, EarlyRequestPreservesPartialTimeout) {
    silent_first_core = true;
    EXPECT_EQ(request_all(), 0);
    handshake();
    expect_once();
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}

TEST_F(SchedulerRetirement, InvalidPhysicalCoreIsNotPublishedOrRetired) {
    runtime->dev.workers[0].physical_core_id = kCores;
    EXPECT_EQ(request_all(), 0);
    handshake();
    EXPECT_TRUE(SchedulerContextTestPeer::handshake_failed(*context));
    EXPECT_EQ(retired[0].load(), 0);
    EXPECT_EQ(runtime->get_teardown_gates()[0].post_close_release, 0U);
    for (int i = 1; i < kCores; ++i)
        EXPECT_EQ(retired[i].load(), 1);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(retired[0].load(), 0);
}

TEST_F(SchedulerRetirement, UnmappedWindowIsNotOpenedOrPublished) {
    regs[0] = 0;
    EXPECT_EQ(request_all(), 0);
    handshake();
    EXPECT_TRUE(SchedulerContextTestPeer::handshake_failed(*context));
    EXPECT_EQ(opened[0].load(), 0);
    EXPECT_EQ(retired[0].load(), 0);
    EXPECT_EQ(runtime->get_teardown_gates()[0].post_close_release, 0U);
    for (int i = 1; i < kCores; ++i)
        EXPECT_EQ(retired[i].load(), 1);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(retired[0].load(), 0);
}

TEST_F(SchedulerRetirement, OwnedClusterPublishersExcludeUnmappedWindow) {
    regs[0] = 0;
    EXPECT_EQ(request_all(), 0);
    if (!handshake_owned(*context, runtime.get(), 0, 0)) {
        GTEST_SKIP() << "HBG has only the partition handshake";
    }
    ASSERT_TRUE(handshake_owned(*context, runtime.get(), 1, 0));
    EXPECT_TRUE(SchedulerContextTestPeer::handshake_failed(*context));
    EXPECT_EQ(opened[0].load(), 0);
    EXPECT_EQ(retired[0].load(), 0);
    EXPECT_EQ(runtime->get_teardown_gates()[0].post_close_release, 0U);
    for (int i = 1; i < kCores; ++i)
        EXPECT_EQ(retired[i].load(), 1);
    EXPECT_EQ(batches.load(), 2);
    EXPECT_EQ(request_all(), 0);
}

TEST_F(SchedulerRetirement, PublishedZeroAddressDoesNotRetireOrReleaseTarget) {
    handshake();
    // Fault injection after READY exercises the retirement boundary guard.
    SchedulerContextTestPeer::clear_register_address(*context, 3);
    silent_first_core = true;
    EXPECT_EQ(request_all(), -1);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(retired[i].load(), i == 3 ? 0 : 1);
        EXPECT_EQ(
            runtime->get_teardown_gates()[i].post_close_release, i == 0 || i == 3 ? 0U : AICORE_POST_CLOSE_RELEASE
        );
    }
    EXPECT_EQ(batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 1);
}

TEST_F(SchedulerRetirement, EmptyValidatedBatchDoesNotAllocateDeadline) {
    handshake();
    for (int i = 0; i < kCores; ++i)
        SchedulerContextTestPeer::clear_register_address(*context, i);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(batches.load(), 0);
    EXPECT_EQ(deadlines.load(), 0);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(retired[i].load(), 0);
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0U);
    }
}

TEST_F(SchedulerRetirement, NewGenerationClearsPendingRequestsAndReturnGates) {
    EXPECT_EQ(request_all(), 0);
    for (int i = 0; i < kCores; ++i)
        runtime->get_teardown_gates()[i].post_close_release = AICORE_POST_CLOSE_RELEASE;
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    handshake();
    EXPECT_EQ(batches.load(), 0);
    for (int i = 0; i < kCores; ++i)
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0U);
    EXPECT_EQ(request_all(), 0);
    expect_once();
}
