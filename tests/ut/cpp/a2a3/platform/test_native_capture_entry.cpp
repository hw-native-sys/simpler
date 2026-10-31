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
#include <atomic>
#include <thread>
#include <vector>

#include "aicpu/device_run_result_aicpu.h"
#include "aicpu/device_run_result_base_aicpu.h"
#include "aicpu/platform_entry_args.h"
#include "common/kernel_args.h"
#include "common/native_execution_control.h"
#include "runtime.h"

extern "C" int simpler_aicpu_exec(void *);
extern "C" int simpler_aicpu_prepare_execution(void *);
bool native_capture_test_affinity = true;

namespace {
constexpr uint64_t kEpoch = 73;
int execution_calls = 0;
int local_status = 0;
int terminal_status = 0;
bool publish_result = true;
bool wrong_epoch = false;
uint64_t expected_result_base = 0;

class NativeExecutionEntry : public ::testing::Test {
protected:
    void SetUp() override {
        native_capture_test_affinity = true;
        execution_calls = local_status = terminal_status = 0;
        publish_result = true;
        wrong_epoch = false;
        runtime.set_aicpu_allowed_cpu_count(1);
        runtime.set_aicpu_launch_count(1);
        args.runtime_args = &runtime;
        args.run_result_data_base = reinterpret_cast<uint64_t>(&result);
        expected_result_base = args.run_result_data_base;
        args.run_result_epoch = kEpoch;
        args.execution_control = reinterpret_cast<uint64_t>(&execution);
    }
    int prepare(uint32_t callbacks = 1, uint32_t workers = 0) {
        runtime.set_aicpu_launch_count(static_cast<int>(callbacks));
        NativeExecutionPrepareArgs gate{
            args.execution_control,
            reinterpret_cast<uint64_t>(&runtime),
            args.run_result_data_base,
            args.run_result_epoch,
            reinterpret_cast<uint64_t>(&context_failure),
            callbacks,
            workers
        };
        return simpler_aicpu_prepare_execution(&gate);
    }
    NativeExecutionControl execution{};
    NativeExecutionFirstError context_failure{};
    int32_t &context_error{context_failure.code};
    Runtime runtime;
    KernelArgs args{};
    DeviceRunResultRegion result{};
};
}  // namespace

// The native platform entry remains production code. Stand in only for the
// task executor, including its independent structured terminal selection.
extern "C" int aicpu_execute(Runtime *) {
    ++execution_calls;
    EXPECT_EQ(get_platform_run_result_epoch(), kEpoch);
    EXPECT_EQ(get_platform_run_result_base(), expected_result_base);
    EXPECT_TRUE(get_platform_entry_args(0).unstamped_reports);
    if (get_platform_entry_args(0).dispatch_error != 0) return get_platform_entry_args(0).dispatch_error;
    if (publish_result) {
        EXPECT_TRUE(aicpu_publish_run_terminal(
            get_platform_run_result_base(), wrong_epoch ? kEpoch + 1 : kEpoch,
            terminal_status == 0 ? DeviceRunVerdict::Ok : DeviceRunVerdict::Error, terminal_status,
            terminal_status == 0 ? DeviceRunCodeSource::None : DeviceRunCodeSource::ShutdownRc, nullptr, 0
        ));
    }
    return local_status;
}

TEST_F(NativeExecutionEntry, AllLaunchedCallbacksIncludeFilteredEntries) {
    ASSERT_EQ(prepare(2), 0);
    native_capture_test_affinity = false;
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    EXPECT_EQ(execution.remaining_callbacks, 1u);
    native_capture_test_affinity = true;
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    EXPECT_EQ(execution.remaining_callbacks, 0u);
    EXPECT_EQ(execution_calls, 1);
}

TEST_F(NativeExecutionEntry, NonLastCallbackDefersItsErrorUntilFinalContribution) {
    ASSERT_EQ(prepare(2), 0);
    local_status = -71;
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    local_status = 0;
    EXPECT_EQ(simpler_aicpu_exec(&args), -71);
    EXPECT_EQ(execution.remaining_callbacks, 0u);
    EXPECT_EQ(context_error, -71);
    EXPECT_EQ(device_run_result_terminal(result, kEpoch).code, -71);
}

TEST_F(NativeExecutionEntry, EntryFailureBeforeAffinityStillContributes) {
    ASSERT_EQ(prepare(2), 0);
    args.runtime_args = nullptr;
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    EXPECT_EQ(execution.remaining_callbacks, 1u);
    args.runtime_args = &runtime;
    native_capture_test_affinity = false;
    EXPECT_EQ(simpler_aicpu_exec(&args), -1);
    EXPECT_EQ(execution.remaining_callbacks, 0u);
    EXPECT_EQ(execution_calls, 0);
    EXPECT_EQ(device_run_result_terminal(result, kEpoch).state, DeviceRunTerminalState::Failed);
}

TEST_F(NativeExecutionEntry, FilteredLastCallbackPublishesEarlierRuntimeFailure) {
    ASSERT_EQ(prepare(2), 0);
    terminal_status = -81;
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    native_capture_test_affinity = false;
    EXPECT_EQ(simpler_aicpu_exec(&args), -81);
    EXPECT_EQ(context_error, -81);
    EXPECT_EQ(result.code_source, static_cast<uint32_t>(DeviceRunCodeSource::ShutdownRc));
}

TEST_F(NativeExecutionEntry, DecisionDoesNotChangeWhenFirstErrorChangesAfterPreparation) {
    ASSERT_EQ(prepare(), 0);
    context_error = -81;
    EXPECT_EQ(execution.decision, NativeExecutionDecision::Execute);
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    ASSERT_EQ(prepare(), 0);
    EXPECT_EQ(execution.decision, NativeExecutionDecision::Skip);
    EXPECT_EQ(simpler_aicpu_exec(&args), -81);
    EXPECT_EQ(get_platform_entry_args(0).dispatch_error, -81);
    EXPECT_EQ(context_error, -81);
}

TEST_F(NativeExecutionEntry, MissingTerminalIsLatchedBeforeNextPublicationIsCleared) {
    ASSERT_EQ(prepare(), 0);
    publish_result = false;
    EXPECT_EQ(simpler_aicpu_exec(&args), -1);
    EXPECT_EQ(device_run_result_terminal(result, kEpoch).state, DeviceRunTerminalState::Undecided);
    EXPECT_EQ(result.published, 0u);
    ASSERT_EQ(prepare(), 0);
    EXPECT_EQ(execution.decision, NativeExecutionDecision::Skip);
    EXPECT_EQ(context_error, -1);
    EXPECT_EQ(result.published, 0u);
    EXPECT_EQ(simpler_aicpu_exec(&args), -1);
    EXPECT_EQ(device_run_result_terminal(result, kEpoch).state, DeviceRunTerminalState::Undecided);
    // A newly owned descriptor must inherit Unknown from context storage, too.
    execution = NativeExecutionControl{};
    ASSERT_EQ(prepare(), 0);
    EXPECT_EQ(simpler_aicpu_exec(&args), -1);
    EXPECT_EQ(device_run_result_terminal(result, kEpoch).state, DeviceRunTerminalState::Undecided);
}

TEST_F(NativeExecutionEntry, FirstErrorSourceSurvivesSuccessorSkip) {
    ASSERT_EQ(prepare(), 0);
    terminal_status = -81;
    EXPECT_EQ(simpler_aicpu_exec(&args), -81);
    execution = NativeExecutionControl{};
    ASSERT_EQ(prepare(), 0);
    terminal_status = 0;
    EXPECT_EQ(simpler_aicpu_exec(&args), -81);
    EXPECT_EQ(result.code_source, static_cast<uint32_t>(DeviceRunCodeSource::ShutdownRc));
}

TEST_F(NativeExecutionEntry, IncompletePredecessorCannotResetItsEvidence) {
    ASSERT_EQ(prepare(2), 0);
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    const uint64_t published = result.published;
    EXPECT_NE(prepare(2), 0);
    EXPECT_EQ(result.published, published);
    EXPECT_EQ(execution.remaining_callbacks, 1u);
    EXPECT_EQ(execution.sequence, 1u);
}

TEST_F(NativeExecutionEntry, PreparationResetsOnlyItsWorkerRangeBeforeExecution) {
    runtime.dev.workers[0].physical_core_id = 37;
    runtime.dev.workers[0].report_epoch = 91;
    runtime.dev.workers[0].aicore_done = 71;
    runtime.dev.workers[0].aicpu_ready = 72;
    runtime.dev.workers[1].aicore_done = 81;
    ASSERT_EQ(prepare(1, 1), 0);
    EXPECT_EQ(runtime.dev.workers[0].physical_core_id, 37u);
    EXPECT_EQ(runtime.dev.workers[0].report_epoch, 91u);
    EXPECT_EQ(runtime.dev.workers[0].aicore_done, 0u);
    EXPECT_EQ(runtime.dev.workers[0].aicpu_ready, 0u);
    EXPECT_EQ(runtime.dev.workers[1].aicore_done, 81u);
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
}

TEST_F(NativeExecutionEntry, OrdinaryRunIdentitiesChangeWithoutLookingInTheNewResultForTheOldOne) {
    ASSERT_EQ(prepare(), 0);
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    // The executor fixture fixes its own identity; the gate must still inspect
    // the old identity before rebinding the next ordinary run.
    args.run_result_epoch = kEpoch + 1;
    ASSERT_EQ(prepare(), 0);
    EXPECT_EQ(execution.decision, NativeExecutionDecision::Execute);
    EXPECT_EQ(execution.native_epoch, kEpoch + 1);
    EXPECT_EQ(result.published, 0u);
}

TEST_F(NativeExecutionEntry, CountsMaximumRTSLaunchIncludingEveryFilteredCallback) {
    ASSERT_EQ(prepare(PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH), 0);
    native_capture_test_affinity = false;
    for (uint32_t i = 1; i < PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH; ++i) {
        EXPECT_EQ(simpler_aicpu_exec(&args), 0);
        EXPECT_EQ(execution.remaining_callbacks, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH - i);
    }
    native_capture_test_affinity = true;
    EXPECT_EQ(simpler_aicpu_exec(&args), 0);
    EXPECT_EQ(execution.remaining_callbacks, 0u);
    EXPECT_EQ(execution_calls, 1);
}

TEST_F(NativeExecutionEntry, ConcurrentContributionsHaveExactlyOneFinalPublisher) {
    constexpr uint32_t count = PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    ASSERT_EQ(prepare(count), 0);
    std::atomic<uint32_t> started{0};
    std::atomic<uint32_t> final_count{0};
    std::vector<int> returned(count);
    std::vector<std::thread> callbacks;
    for (uint32_t i = 0; i < count; ++i) {
        callbacks.emplace_back([&, i] {
            started.fetch_add(1);
            while (started.load() != count)
                std::this_thread::yield();
            returned[i] = native_execution_finish(
                execution, i == 0 ? -71 : 0, [] {},
                [&](int32_t status) {
                    final_count.fetch_add(1);
                    EXPECT_EQ(status, -71);
                    EXPECT_TRUE(aicpu_publish_run_terminal(
                        args.run_result_data_base, kEpoch, DeviceRunVerdict::Error, status,
                        DeviceRunCodeSource::ThreadRc, nullptr, 0
                    ));
                }
            );
        });
    }
    for (auto &callback : callbacks)
        callback.join();
    uint32_t failures = 0;
    for (int status : returned) {
        EXPECT_TRUE(status == 0 || status == -71);
        failures += status != 0;
    }
    EXPECT_EQ(failures, 1u);
    EXPECT_EQ(final_count.load(), 1u);
    EXPECT_EQ(execution.remaining_callbacks, 0u);
    EXPECT_EQ(context_error, -71);
}

TEST_F(NativeExecutionEntry, WrongNativeIdentityCannotBecomeSuccessfulCompletion) {
    ASSERT_EQ(prepare(), 0);
    wrong_epoch = true;
    EXPECT_NE(simpler_aicpu_exec(&args), 0);
    EXPECT_EQ(context_failure.kind, NativeExecutionFailure::Unknown);
    EXPECT_EQ(result.published, 0u);
}
