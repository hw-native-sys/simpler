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
#include <cstdlib>
#include <memory>
#include <mutex>
#include <thread>

#include "runtime.h"
#include "scheduler/scheduler_context.h"
#ifdef PTO_ASYNC_WAIT_H
#include "runtime_core.h"
#endif

namespace {
constexpr int kCores = 6;
std::array<std::atomic<int>, kCores> opened{};
std::array<std::atomic<int>, kCores> retired{};
std::array<uint32_t, kCores> conditions{};
std::array<Handshake *, kCores> expected_reports{};
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
static uint64_t test_report_epoch = 0;
uint64_t test_exit_epoch() {
#ifdef SIMPLER_TEST_EPOCH_RETIREMENT
    return test_report_epoch;
#else
    return 0;
#endif
}

static std::atomic<int> nonce_batches{0};
static std::atomic<int> epoch_batches{0};
extern "C" uint64_t get_platform_run_result_epoch() { return test_report_epoch; }
uint64_t platform_aicore_exit_deadline() { return ++deadlines; }
uint64_t read_reg(uint64_t, RegId) { return AICORE_IDLE_VALUE; }
volatile uint32_t *get_reg_ptr(uint64_t address, RegId) { return &conditions.at(address - 1); }

#ifdef PTO_ASYNC_WAIT_H
void AsyncWaitList::log_diagnostics(AICoreCompletionMailbox *, const char *) {}
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
    if (test_exit_epoch() == 0) ++nonce_batches;
    else ++epoch_batches;
    ++batches;
    bool timed_out = false;
    for (size_t i = 0; i < count; ++i) {
        const auto core = targets[i].reg_addr - 1;
        EXPECT_LT(core, kCores);
        if (core >= kCores) continue;
        EXPECT_EQ(opened[core].load(), 1);
        EXPECT_NE(targets[i].teardown, nullptr);
        EXPECT_EQ(targets[i].expected_epoch, test_exit_epoch());
#ifdef SIMPLER_TEST_EPOCH_RETIREMENT
        if (test_exit_epoch() != 0) EXPECT_EQ(targets[i].retired_epoch, &expected_reports[core]->retired_epoch);
#else
        EXPECT_EQ(targets[i].retired_epoch, nullptr);
#endif
        ++retired[core];
        const bool success = !(silent_first_core && core == 0);
        if (released != nullptr) released[i] = success;
        if (success && targets[i].teardown != nullptr) {
            if (targets[i].expected_epoch != 0) {
                __atomic_store_n(targets[i].retired_epoch, targets[i].expected_epoch, __ATOMIC_RELEASE);
                __atomic_store_n(&targets[i].teardown->release_epoch, targets[i].expected_epoch, __ATOMIC_RELEASE);
            } else {
                __atomic_store_n(&targets[i].teardown->post_close_release, AICORE_POST_CLOSE_RELEASE, __ATOMIC_RELEASE);
            }
        }
        timed_out |= !success;
    }
    return timed_out ? -1 : 0;
}

class SchedulerContextTestPeer {
public:
#ifdef PTO_ASYNC_WAIT_H
    static void bind_async(SchedulerContext &context, RuntimeContext &runtime) {
        context.rt_ = &runtime;
        context.sched_ = &runtime.scheduler;
    }
    static DeferredCompletionSlab &slab(SchedulerContext &context, TaskId task) {
        context.payload_per_core_[0][0].local_context.async_ctx.task_token = task;
        return context.deferred_slab_per_core_[0][0];
    }
#endif
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
        test_report_epoch = 0;
        nonce_batches = 0;
        epoch_batches = 0;
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
            expected_reports[i] = &handshake;
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
                runtime->get_teardown_gates()[i].release_epoch, silent_first_core && i == 0 ? 0U : test_exit_epoch()
            );
            EXPECT_EQ(
                runtime->get_teardown_gates()[i].post_close_release,
                silent_first_core && i == 0 || test_exit_epoch() != 0 ? 0U : AICORE_POST_CLOSE_RELEASE
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

#ifdef PTO_ASYNC_WAIT_H
TEST_F(SchedulerRetirement, ShutdownPreservesEmergencyRetirementFailure) {
    handshake();
    silent_first_core = true;
    emergency();
    EXPECT_EQ(context->shutdown(0, runtime.get()), -1);
    EXPECT_EQ(context->shutdown(1, runtime.get()), -1);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
}

TEST_F(SchedulerRetirement, ShutdownPreservesPublisherRetirementFailure) {
    silent_first_core = true;
    EXPECT_EQ(request_all(), 0);
    handshake();
    EXPECT_EQ(context->shutdown(0, runtime.get()), -1);
    EXPECT_EQ(context->shutdown(1, runtime.get()), -1);
    expect_once();
    EXPECT_EQ(batches.load(), 1);
}

TEST_F(SchedulerRetirement, NewRunClearsRetirementFailure) {
    handshake();
    silent_first_core = true;
    emergency();
    EXPECT_EQ(context->retirement_error(), -1);
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    EXPECT_EQ(context->retirement_error(), 0);
}

TEST_F(SchedulerRetirement, CancelledDrainCoversAllOwnersWithoutCompletingTasks) {
    handshake();
    ASSERT_EQ(request_all(), 0);
    auto rt = std::make_unique<RuntimeContext>();
    auto mailbox = std::make_unique<AICoreCompletionMailbox>();
    rt->aicore_mailbox = mailbox.get();
    SchedulerContextTestPeer::bind_async(*context, *rt);
    const TaskId task = TaskId::make(0, 19);
    uint32_t counter = 3;
    uint64_t sdma_record = 9;
    auto &waits = rt->scheduler.async_wait_list;
    waits.count.store(1);
    waits.entries[0].task_token = task;
    waits.entries[0].normal_done = true;
    // Any accidental success callback would dereference this invalid slot.
    waits.entries[0].slot_state = reinterpret_cast<ChipTaskSlotState *>(1);
    int32_t error = 0;
    ASSERT_TRUE(waits.append_condition_locked(
        waits.entries[0], reinterpret_cast<uint64_t>(&counter), 0, 3, ASYNC_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, error
    ));
    ASSERT_TRUE(mailbox->try_push_condition(
        task, reinterpret_cast<uint64_t>(&sdma_record), 9, 0, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_SDMA_EVENT_RECORD
    ));
    ASSERT_TRUE(mailbox->try_push_normal_done(task, 1));
    auto &slab = SchedulerContextTestPeer::slab(*context, task);
    slab.count = 1;
    slab.entries[0] = {reinterpret_cast<uint64_t>(&counter), 0, 3, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    KernelErrorCleanup result{};
    context->drain_cancelled_run(UINT64_MAX, result);
    EXPECT_EQ(result.complete, 1u);
    EXPECT_EQ(result.released_cores, kCores);
    EXPECT_EQ(result.drained_conditions, 3u);
    EXPECT_TRUE(waits.entries[0].conditions[0].retired);
    EXPECT_EQ(waits.count.load(), 1);
    EXPECT_EQ(sdma_record, 9u);
}

TEST_F(SchedulerRetirement, CancelledDrainRejectsPendingSlabAndPreservesIdentity) {
    handshake();
    ASSERT_EQ(request_all(), 0);
    auto rt = std::make_unique<RuntimeContext>();
    SchedulerContextTestPeer::bind_async(*context, *rt);
    const TaskId task = TaskId::make(1, 20);
    uint32_t counter = 0;
    auto &slab = SchedulerContextTestPeer::slab(*context, task);
    slab.count = 1;
    slab.entries[0] = {reinterpret_cast<uint64_t>(&counter), 0, 1, COMPLETION_ENGINE_SDMA, COMPLETION_TYPE_COUNTER, 0};
    KernelErrorCleanup result{};
    context->drain_cancelled_run(0, result);
    EXPECT_EQ(result.complete, 0u);
    EXPECT_EQ(result.async_error, SIMPLER_ERROR_SCHEDULER_TIMEOUT);
    EXPECT_EQ(result.source, 4u);
    EXPECT_EQ(result.task_token, TaskId::to_uint64(task));
    EXPECT_EQ(slab.count, 1u);
    counter = 1;
    context->drain_cancelled_run(UINT64_MAX, result);
    EXPECT_EQ(result.complete, 1u);
}

TEST_F(SchedulerRetirement, CancelledDrainRequiresEveryCoreRelease) {
    handshake();
    silent_first_core = true;
    emergency();
    KernelErrorCleanup result{};
    context->drain_cancelled_run(UINT64_MAX, result);
    EXPECT_EQ(result.complete, 0u);
    EXPECT_EQ(result.core_error, -1);
    EXPECT_EQ(result.source, 1u);
    EXPECT_EQ(result.core, 0);
    EXPECT_EQ(result.released_cores, kCores - 1);
}
#endif

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

TEST_F(SchedulerRetirement, InitializationClearsRequestAndReleaseForEveryCore) {
    for (int i = 0; i < kCores; ++i) {
        auto &control = runtime->dev.teardown_gates[i];
        control.post_close_release = AICORE_POST_CLOSE_RELEASE;
        control.exit_request = 9;
        control.release_epoch = UINT64_MAX;
    }
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    for (int i = 0; i < kCores; ++i) {
        const auto &control = runtime->dev.teardown_gates[i];
        EXPECT_EQ(control.post_close_release, 0u);
        EXPECT_EQ(control.exit_request, 0u);
        EXPECT_EQ(control.release_epoch, 0U);
    }
}

TEST_F(SchedulerRetirement, NativeModeClaimsOnlyAfterCurrentReportAndUsesMatchingProtocol) {
    test_report_epoch = 42;
    ASSERT_EQ(initialize(*context, runtime.get(), reinterpret_cast<uint64_t>(regs.data()), 0), 0);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(runtime->get_teardown_gates()[i].exit_request, 0U);
        EXPECT_EQ(runtime->get_teardown_gates()[i].post_close_release, 0U);
        runtime->dev.workers[i].report_epoch = 42;
    }
    SchedulerContextTestPeer::assign(*context);
    EXPECT_EQ(request_all(), 0);
    EXPECT_EQ(nonce_batches.load(), 0);
    handshake();
    expect_once();
    EXPECT_EQ(nonce_batches.load(), test_exit_epoch() == 0 ? 1 : 0);
    EXPECT_EQ(epoch_batches.load(), test_exit_epoch() == 0 ? 0 : 1);
    for (int i = 0; i < kCores; ++i) {
        EXPECT_EQ(runtime->get_teardown_gates()[i].release_epoch, test_exit_epoch());
        EXPECT_EQ(runtime->dev.workers[i].aicore_done, 0U);
    }
    test_report_epoch = 0;
}
TEST_F(SchedulerRetirement, UnstampedModeUsesNonceAfterOwnershipPublication) {
    handshake();
    EXPECT_EQ(request_all(), 0);
    expect_once();
    EXPECT_EQ(nonce_batches.load(), 1);
}

TEST_F(SchedulerRetirement, ReportsAreInvalidatedOnlyAfterTheirOwnExitAcknowledgement) {
    handshake();
    for (int i = 0; i < kCores; ++i) {
        runtime->dev.workers[i].aicpu_ready = 2;
        runtime->dev.workers[i].report_epoch = 42;
    }
    silent_first_core = true;
    EXPECT_NE(request_all(), 0);
    for (int i = 0; i < kCores; ++i) {
        const auto &report = runtime->dev.workers[i];
        EXPECT_EQ(report.aicore_done, i == 0 ? 1u : 0u);
        EXPECT_EQ(report.aicpu_ready, i == 0 ? 2u : 0u);
        EXPECT_EQ(report.report_epoch, 42u);
        EXPECT_EQ(aicore_report_accepted(&report, 42), i == 0);
    }
    const int previous_batches = batches.load();
    emergency();
    EXPECT_EQ(batches.load(), previous_batches);
    EXPECT_EQ(runtime->dev.workers[0].aicore_done, 1u);
}

TEST_F(SchedulerRetirement, MissingReportDoesNotTurnAcceptedRetirementIntoExitProof) {
    EXPECT_EXIT(
        {
            SchedulerContextTestPeer::clear_ownership(*context);
            runtime->dev.workers[kCores - 1].aicore_done = 0;
            context->handshake_partition(runtime.get(), 0, 2);
            for (int i = 0; i < kCores / 2; ++i)
                if (opened[i].load() != 1) std::_Exit(1);

            std::atomic<bool> entered{false};
            std::atomic<bool> returned{false};
            std::thread missing_report([&] {
                entered.store(true, std::memory_order_release);
                context->handshake_partition(runtime.get(), 1, 2);
                returned.store(true, std::memory_order_release);
            });
            const auto startup_deadline = std::chrono::steady_clock::now() + std::chrono::seconds(3);
            while (!entered.load(std::memory_order_acquire)) {
                if (std::chrono::steady_clock::now() >= startup_deadline) std::_Exit(2);
            }
            if (request_all() != 0) std::_Exit(3);
            const auto observation_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(100);
            while (std::chrono::steady_clock::now() < observation_deadline) {
                if (returned.load(std::memory_order_acquire)) std::_Exit(4);
            }
            // READY is published only after the handshake join. An accepted
            // request is still pending, even for the already-open first slice.
            if (batches.load() != 0 || deadlines.load() != 0) std::_Exit(5);
            for (int i = 0; i < kCores; ++i) {
                if (retired[i].load() != 0 || runtime->get_teardown_gates()[i].post_close_release != 0) std::_Exit(6);
            }
            if (opened[kCores - 1].load() != 0) std::_Exit(7);
            // Only the fixture process ends here; no physical-exit fact is minted.
            std::_Exit(0);
        },
        testing::ExitedWithCode(0), ""
    );
}
