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
std::atomic<int> retirement_batches{0};
std::atomic<int> deadlines{0};
bool silent_core = false;
}  // namespace

bool platform_aicpu_affinity_exact_match() { return false; }
extern "C" uint64_t get_sched_thread_die_bits() { return 0; }
uint32_t platform_get_physical_cores_count() { return 64; }
void platform_init_aicore_regs(uint64_t) {}
uint64_t platform_aicore_exit_deadline() { return ++deadlines; }
uint64_t read_reg(uint64_t, RegId) { return AICORE_EXITED_VALUE; }
int32_t platform_retire_aicore_group(const AicoreExitTarget *targets, size_t count, uint64_t, bool *released) {
    ++retirement_batches;
    for (size_t i = 0; i < count; ++i) {
        EXPECT_GE(targets[i].reg_addr, 1U);
        EXPECT_LE(targets[i].reg_addr, kCores);
        if (targets[i].reg_addr >= 1 && targets[i].reg_addr <= kCores) ++retirements[targets[i].reg_addr - 1];
        if (released != nullptr) released[i] = !(silent_core && targets[i].reg_addr == 1);
        EXPECT_NE(targets[i].teardown, nullptr);
        if (targets[i].teardown != nullptr && !(silent_core && targets[i].reg_addr == 1)) {
            __atomic_store_n(&targets[i].teardown->post_close_release, AICORE_POST_CLOSE_RELEASE, __ATOMIC_RELAXED);
        }
    }
    return silent_core ? -1 : 0;
}

class SchedulerRetirementTestPeer {
public:
    static void publish_orchestration(SchedulerContext &context, int total, int completed, bool done) {
        context.total_tasks_ = total;
        context.completed_tasks_.store(completed, std::memory_order_relaxed);
        context.orchestrator_done_.store(done, std::memory_order_release);
    }
    static LoopAction check_exit(SchedulerContext &context, SharedMemoryHeader &header, Runtime *runtime) {
        int32_t count = -1;
        return context.handle_orchestrator_exit(0, &header, runtime, count);
    }
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
    static void set_first_thread_die_ownership(SchedulerContext &context) {
        context.owned_clusters_[0][0] = 1;
        context.owned_cluster_count_[0] = 1;
        context.die_affinity_active_[0] = true;
    }
    static int cluster_count(const SchedulerContext &context, int owner) {
        return context.core_trackers_[owner].get_cluster_count();
    }
    static int first_aic(const SchedulerContext &context, int owner) {
        return context.core_trackers_[owner].get_aic_core_id(0);
    }
    static uint32_t physical_core_id(const SchedulerContext &context, int worker) {
        return context.physical_core_ids_[worker];
    }
};

class SchedulerRetirement : public testing::Test {
protected:
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    std::unique_ptr<Runtime> runtime = std::make_unique<Runtime>();

    void SetUp() override {
        retirement_batches = 0;
        deadlines = 0;
        silent_core = false;
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

TEST_F(SchedulerRetirement, EmergencyUsesOneBatchAndDeadlineAcrossOwners) {
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    silent_core = true;
    EXPECT_EQ(context->retire_all_cores(), -1);
    expect_once();
    EXPECT_EQ(retirement_batches.load(), 1);
    EXPECT_EQ(deadlines.load(), 1);
}

TEST_F(SchedulerRetirement, OverlappingCoreSetsAreClaimedIndependently) {
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    const int32_t subset[] = {0, 3, 0};
    EXPECT_EQ(context->retire_cores(subset, 3), 0);
    EXPECT_EQ(context->retire_all_cores(), 0);
    expect_once();
    EXPECT_EQ(retirement_batches.load(), 2);
}

TEST_F(SchedulerRetirement, NewGenerationClearsEveryReturnGateBeforePublication) {
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    ASSERT_EQ(context->retire_all_cores(), 0);
    for (int i = 0; i < kCores; ++i)
        ASSERT_EQ(runtime->dev.teardown_gates[i].post_close_release, AICORE_POST_CLOSE_RELEASE);
    context->deinit();
    ASSERT_EQ(context->pre_handshake_init(runtime.get(), 3, 2, 0), 0);
    for (int i = 0; i < kCores; ++i)
        EXPECT_EQ(runtime->dev.teardown_gates[i].post_close_release, 0U);
}

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

TEST_F(SchedulerRetirement, OneThreadDieOwnershipDoesNotChangeAnotherThreadFallback) {
    SchedulerRetirementTestPeer::set_first_thread_die_ownership(*context);
    context->assign_own_clusters(1);
    EXPECT_EQ(SchedulerRetirementTestPeer::cluster_count(*context, 1), 1);
}

TEST_F(SchedulerRetirement, ReinitializationClearsDieOwnership) {
    SchedulerRetirementTestPeer::set_first_thread_die_ownership(*context);
    context->deinit();
    ASSERT_EQ(context->pre_handshake_init(runtime.get(), 3, 2, 0), 0);
    context->assign_own_clusters(0);
    EXPECT_EQ(SchedulerRetirementTestPeer::cluster_count(*context, 0), 1);
    EXPECT_EQ(SchedulerRetirementTestPeer::first_aic(*context, 0), 0);
}

TEST_F(SchedulerRetirement, NonDfxHandshakePublishesPhysicalIdsForSerialAssignment) {
    runtime->dev.serial_orch_sched = true;
    std::array<uint64_t, 64> regs{};
    for (int i = 0; i < kCores; ++i) {
        const uint32_t pcid = static_cast<uint32_t>(i + 2);
        regs[pcid] = static_cast<uint64_t>(i + 1);
        runtime->dev.workers[i].physical_core_id = pcid;
        runtime->dev.workers[i].core_type = i < 2 ? CoreType::AIC : CoreType::AIV;
        runtime->dev.workers[i].aicore_done = pcid + 1;
    }
    ASSERT_EQ(context->pre_handshake_init(runtime.get(), 3, 2, reinterpret_cast<uint64_t>(regs.data())), 0);
    context->handshake_partition(runtime.get(), 0, 1);
    for (int i = 0; i < kCores; ++i)
        EXPECT_EQ(SchedulerRetirementTestPeer::physical_core_id(*context, i), static_cast<uint32_t>(i + 2));
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

// Claim is per core, not per caller: concurrent calls naming arbitrary
// overlapping sets must each win only the cores no other call has claimed. A
// per-caller claim lets two overlapping callers both retire a shared core, which
// shows up below as a count above 1 for that core.
TEST_F(SchedulerRetirement, ConcurrentOverlappingSetsClaimEachCoreOnce) {
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);

    const int32_t sets[][6] = {
        {0, 1, 2, 3, 4, 5}, {5, 4, 3, 2, 1, 0}, {0, 2, 4, 0, 2, 4}, {1, 3, 5, 1, 3, 5}, {2, 3, 0, 5, 4, 1},
    };
    constexpr int kCallers = static_cast<int>(sizeof(sets) / sizeof(sets[0]));

    // A start line, so the calls contend on the per-core claim instead of being
    // serialized by thread creation order.
    std::atomic<bool> go{false};
    std::array<std::thread, kCallers> callers;
    for (int t = 0; t < kCallers; ++t) {
        callers[t] = std::thread([&, t] {
            while (!go.load(std::memory_order_acquire))
                std::this_thread::yield();
            context->retire_cores(sets[t], 6);
        });
    }
    go.store(true, std::memory_order_release);
    for (auto &caller : callers)
        caller.join();

    // Exactly once per core is the property: a bare total would hide a double
    // claim offset by a miss.
    expect_once();
}

TEST_F(SchedulerRetirement, EmptyGraphCompletesOnlyAfterOrchestrationPublication) {
    auto header = std::make_unique<SharedMemoryHeader>();
    SchedulerRetirementTestPeer::publish_orchestration(*context, 0, 0, false);
    EXPECT_EQ(SchedulerRetirementTestPeer::check_exit(*context, *header, runtime.get()), LoopAction::NONE);
    SchedulerRetirementTestPeer::publish_orchestration(*context, 0, 0, true);
    EXPECT_EQ(SchedulerRetirementTestPeer::check_exit(*context, *header, runtime.get()), LoopAction::BREAK_LOOP);
    EXPECT_EQ(SchedulerRetirementTestPeer::check_exit(*context, *header, runtime.get()), LoopAction::BREAK_LOOP);
    EXPECT_FALSE(SchedulerRetirementTestPeer::fatal(*context));
}

TEST_F(SchedulerRetirement, NonemptyGraphStillWaitsForItsLastTask) {
    auto header = std::make_unique<SharedMemoryHeader>();
    SchedulerRetirementTestPeer::publish_orchestration(*context, 1, 0, true);
    EXPECT_EQ(SchedulerRetirementTestPeer::check_exit(*context, *header, runtime.get()), LoopAction::NONE);
    SchedulerRetirementTestPeer::publish_orchestration(*context, 1, 1, true);
    EXPECT_EQ(SchedulerRetirementTestPeer::check_exit(*context, *header, runtime.get()), LoopAction::BREAK_LOOP);
}
