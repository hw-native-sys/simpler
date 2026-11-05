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
#include <string>
#include <thread>

#include "aicpu/device_run_result_base_aicpu.h"
#include "runtime.h"
#include "scheduler/scheduler_context.h"

namespace {
constexpr int kCores = 6;
std::array<std::atomic<int>, kCores> retirements{};
std::atomic<int> retirement_batches{0};
std::atomic<int> deadlines{0};
bool silent_core = false;
std::atomic<int> nonce_batches{0};
bool diagnostic_registers = false;
uint64_t diagnostic_cond = 0;
uint64_t diagnostic_dmb = 0;
int diagnostic_reads = 0;
}  // namespace

uint64_t platform_aicore_exit_deadline() { return ++deadlines; }
uint64_t read_reg(uint64_t address, RegId reg) {
    if (!diagnostic_registers) return AICORE_EXITED_VALUE;
    EXPECT_EQ(address, 1u);
    ++diagnostic_reads;
    if (reg == RegId::COND) return diagnostic_cond;
    EXPECT_EQ(reg, RegId::DATA_MAIN_BASE);
    return diagnostic_dmb;
}
int32_t platform_retire_aicore_group(const AicoreExitTarget *targets, size_t count, uint64_t, bool *released) {
    ++nonce_batches;
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
    static void configure_diagnostic(
        SchedulerContext &context, ChipTaskSlotState *running, int32_t running_token, int32_t pending_token,
        bool busy = true
    ) {
        auto &tracker = context.core_trackers_[0];
        tracker.init(1);
        tracker.set_cluster(0, 0, 1, 2);
        if (busy) tracker.change_core_state(0);
        auto &core = context.core_exec_states_[0];
        core.reg_addr = 1;
        core.running_slot_state = running;
        core.running_subslot = SubtaskSlot::AIC;
        core.running_reg_task_id = running_token;
        core.pending_reg_task_id = pending_token;
        // A raw-token snapshot does not inspect pending task memory.
        core.pending_slot_state = reinterpret_cast<ChipTaskSlotState *>(uintptr_t{1});
    }
    static void log_diagnostic(SchedulerContext &context) {
        context.log_stall_diagnostics(0, 42, StallDumpReport::Shutdown);
    }
};

class SchedulerRetirement : public testing::Test {
protected:
    std::unique_ptr<SchedulerContext> context = std::make_unique<SchedulerContext>();
    std::unique_ptr<Runtime> runtime = std::make_unique<Runtime>();

    void SetUp() override {
        set_platform_run_result(0, 0);
        nonce_batches = 0;
        retirement_batches = 0;
        deadlines = 0;
        silent_core = false;
        diagnostic_registers = false;
        diagnostic_reads = 0;
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

TEST_F(SchedulerRetirement, StallSnapshotPreservesDualSlotRegisterIdentity) {
    struct Observation {
        int32_t token;
        int32_t state;
        int32_t pending;
        const char *match;
    };
    const Observation observations[] = {
        {41, TASK_FIN_STATE, 42, "running"},
        {42, TASK_ACK_STATE, 42, "pending"},
        {42, TASK_FIN_STATE, 42, "pending"},
        {41, TASK_ACK_STATE, 42, "running"},
        {99, TASK_ACK_STATE, 42, "other"},
        {99, TASK_FIN_STATE, 42, "other"},
        {41, TASK_FIN_STATE, AICPU_TASK_INVALID, "running"},
        {AICPU_TASK_INVALID, TASK_FIN_STATE, AICPU_TASK_INVALID, "other"},
    };
    TaskDescriptor task{};
    task.task_id = TaskId::make(3, 67);
    task.kernel_id[0] = 25;
    ChipTaskSlotState slot{};
    slot.task = &task;
    diagnostic_registers = true;
    diagnostic_dmb = 42;
    for (const auto &observation : observations) {
        SCOPED_TRACE(observation.token);
        SchedulerRetirementTestPeer::configure_diagnostic(*context, &slot, 41, observation.pending);
        diagnostic_cond =
            observation.state == TASK_ACK_STATE ? MAKE_ACK_VALUE(observation.token) : MAKE_FIN_VALUE(observation.token);
        testing::internal::CaptureStderr();
        SchedulerRetirementTestPeer::log_diagnostic(*context);
        const std::string output = testing::internal::GetCapturedStderr();
        EXPECT_EQ(output.find("ANOMALY"), std::string::npos);
        EXPECT_NE(
            output.find(observation.state == TASK_ACK_STATE ? "cond_reg_state=ack" : "cond_reg_state=fin"),
            std::string::npos
        );
        EXPECT_NE(output.find("cond_tok=" + std::to_string(EXTRACT_TASK_ID(diagnostic_cond))), std::string::npos);
        EXPECT_NE(output.find("running_tok=41 pending_tok=" + std::to_string(observation.pending)), std::string::npos);
        EXPECT_NE(output.find(std::string("cond_match=") + observation.match), std::string::npos);
        EXPECT_NE(output.find("dmb_low=0x2a"), std::string::npos);
        EXPECT_NE(output.find("snapshot=best_effort"), std::string::npos);
    }
}

TEST_F(SchedulerRetirement, StallSnapshotKeepsMissingMetadataAndWideRegisterValues) {
    SchedulerRetirementTestPeer::configure_diagnostic(*context, nullptr, INT32_MAX - 1, INT32_MAX - 2);
    diagnostic_registers = true;
    diagnostic_cond = UINT32_MAX;
    diagnostic_dmb = UINT32_MAX;
    testing::internal::CaptureStderr();
    SchedulerRetirementTestPeer::log_diagnostic(*context);
    const std::string output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("kernel=-1 task=0x"), std::string::npos);
    EXPECT_NE(output.find("cond_raw=0xffffffff"), std::string::npos);
    EXPECT_NE(output.find("running_tok=2147483646 pending_tok=2147483645"), std::string::npos);
    EXPECT_NE(output.find("cond_match=other dmb_low=0xffffffff)"), std::string::npos);
    EXPECT_EQ(diagnostic_reads, 2);
}

TEST_F(SchedulerRetirement, IdleStallSnapshotDoesNotReadRegisters) {
    SchedulerRetirementTestPeer::configure_diagnostic(*context, nullptr, AICPU_TASK_INVALID, AICPU_TASK_INVALID, false);
    diagnostic_registers = true;
    testing::internal::CaptureStderr();
    SchedulerRetirementTestPeer::log_diagnostic(*context);
    const std::string output = testing::internal::GetCapturedStderr();
    EXPECT_NE(output.find("core0(idle)"), std::string::npos);
    EXPECT_EQ(diagnostic_reads, 0);
}

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

TEST_F(SchedulerRetirement, InitializationClearsRequestAndReleaseForEveryCore) {
    for (int i = 0; i < kCores; ++i) {
        auto &control = runtime->dev.teardown_gates[i];
        control.post_close_release = AICORE_POST_CLOSE_RELEASE;
        control.exit_request = 9;
    }
    ASSERT_EQ(context->pre_handshake_init(runtime.get(), 3, 2, 0), 0);
    for (int i = 0; i < kCores; ++i) {
        const auto &control = runtime->dev.teardown_gates[i];
        EXPECT_EQ(control.post_close_release, 0u);
        EXPECT_EQ(control.exit_request, 0u);
    }
}

TEST_F(SchedulerRetirement, NativeRetirementUsesNonceAfterAssignment) {
    set_platform_run_result(0, 42);
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    EXPECT_EQ(context->retire_all_cores(), 0);
    expect_once();
    EXPECT_EQ(nonce_batches.load(), 1);
}

TEST_F(SchedulerRetirement, UnstampedRetirementUsesNonce) {
    SchedulerRetirementTestPeer::open_cores(*context);
    context->assign_own_clusters(0);
    context->assign_own_clusters(1);
    EXPECT_EQ(context->retire_all_cores(), 0);
    expect_once();
    EXPECT_EQ(nonce_batches.load(), 1);
}
