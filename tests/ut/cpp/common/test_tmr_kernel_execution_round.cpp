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
#include <condition_variable>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

#include "tensormap_and_ringbuffer/kernel_execution_round.h"

namespace {
using namespace simpler::tmr;

class Signal {
public:
    void set() {
        std::lock_guard<std::mutex> lock(mutex_);
        ready_ = true;
        condition_.notify_all();
    }
    void wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        condition_.wait(lock, [&] {
            return ready_;
        });
    }

private:
    std::mutex mutex_;
    std::condition_variable condition_;
    bool ready_{false};
};

// The production coordinator and gate run unchanged. Only executor work and
// core retirement are modeled, so cleanup failures need no live device or
// permanent poisoning of the real executor singleton.
struct ExecutorModel {
    static constexpr int32_t kExecutionThreads = 3;
    static constexpr int32_t kLaunchedThreads = 4;
    static constexpr uint64_t kStorageValue = 91;

    struct Invocation {
        std::atomic<uint64_t> storage{0};
        std::atomic<bool> active{false};
        ExecutionInputs input{};
        Runtime *resident() const { return nullptr; }
        const ExecutionInputs &inputs() const { return input; }
    } kernel_invocation_;

    struct CoreGroup {
        int32_t finish_status{0};
        std::atomic<int32_t> finishes{0};
        std::atomic<int32_t> publications{0};
        KernelFinalStatus published{};
        std::function<void()> on_publish;

        bool attach(KernelHandshakeView) { return true; }
        int32_t finish() {
            ++finishes;
            return finish_status;
        }
        void publish_status(int32_t runtime, int32_t cleanup) {
            published = {runtime, cleanup};
            ++publications;
            if (on_publish) on_publish();
        }
    } kernel_cores_;

    KernelRoundGate kernel_gate_;
    bool kernel_control_attached_{false};
    TmrLaunchControl control{};
    std::array<TmrCoreReport, 3> reports{};
    const std::array<int32_t, kExecutionThreads> allowed{10, 11, 12};
    int32_t complete_status{0};
    int32_t finalize_status{0};
    int32_t failing_run_index{-1};
    std::atomic<int32_t> preparations{0};
    std::atomic<int32_t> initializers{0};
    std::atomic<int32_t> completions{0};
    std::atomic<int32_t> runs{0};
    std::atomic<int32_t> cancellations{0};
    std::atomic<int32_t> finalizations{0};
    std::atomic<int32_t> clears{0};
    std::atomic<bool> init_published{false};

    KernelExecutionRequest request() {
        KernelExecutionRequest request;
        request.handshake = {&control, reports.data(), static_cast<int32_t>(reports.size()), 0};
        request.allowed_cpus = allowed.data();
        request.execution_threads = kExecutionThreads;
        request.launched_threads = kLaunchedThreads;
        return request;
    }

    int32_t prepare_kernel_round(const KernelExecutionRequest &) {
        ++preparations;
        initializers = 0;
        init_published = false;
        kernel_invocation_.storage = kStorageValue;
        kernel_invocation_.active = true;
        return 0;
    }
    int32_t initialize_kernel_thread(const KernelThreadView &) {
        ++initializers;
        return 0;
    }
    int32_t complete_kernel_init() {
        EXPECT_EQ(initializers.load(), kExecutionThreads);
        ++completions;
        init_published = true;
        return complete_status;
    }
    int32_t run(Runtime *, const ExecutionInputs &, const KernelThreadView *thread) {
        EXPECT_TRUE(init_published.load());
        EXPECT_EQ(complete_status, 0);
        EXPECT_TRUE(kernel_invocation_.active.load());
        EXPECT_EQ(kernel_invocation_.storage.load(), kStorageValue);
        ++runs;
        return thread->execution_index == failing_run_index ? -33 : 0;
    }
    void cancel_kernel_round() { ++cancellations; }
    int32_t kernel_status() const { return 0; }
    int32_t finalize_kernel_round() {
        EXPECT_TRUE(kernel_invocation_.active.load());
        ++finalizations;
        return finalize_status;
    }
    void clear_kernel_round() noexcept {
        ++clears;
        kernel_invocation_.storage = 0;
        kernel_invocation_.active = false;
        kernel_control_attached_ = false;
    }
};

std::array<int32_t, ExecutorModel::kLaunchedThreads> run_round(ExecutorModel &executor) {
    const auto request = executor.request();
    std::array<int32_t, ExecutorModel::kLaunchedThreads> results{};
    Signal start;
    std::vector<std::thread> threads;
    for (size_t i = 0; i < results.size(); ++i) {
        threads.emplace_back([&, i] {
            start.wait();
            results[i] = execute_kernel_round_impl(executor, request, static_cast<int32_t>(10 + i));
        });
    }
    start.set();
    for (auto &thread : threads)
        thread.join();
    return results;
}

TEST(TmrKernelExecutionRoundTest, CompleteInitFailureSkipsAllExecutionAndNextRoundCanReuse) {
    ExecutorModel executor;
    executor.complete_status = -27;
    for (int32_t result : run_round(executor))
        EXPECT_EQ(result, -27);
    EXPECT_EQ(executor.runs.load(), 0);
    EXPECT_EQ(executor.completions.load(), 1);
    EXPECT_EQ(executor.kernel_cores_.published.runtime_status, -27);
    EXPECT_EQ(executor.kernel_cores_.published.cleanup_status, 0);
    EXPECT_EQ(executor.finalizations.load(), 1);
    EXPECT_EQ(executor.clears.load(), 1);
    EXPECT_TRUE(executor.kernel_gate_.idle());
    EXPECT_FALSE(executor.kernel_invocation_.active.load());

    executor.complete_status = 0;
    for (int32_t result : run_round(executor))
        EXPECT_EQ(result, 0);
    EXPECT_EQ(executor.runs.load(), ExecutorModel::kExecutionThreads);
    EXPECT_EQ(executor.completions.load(), 2);
    EXPECT_EQ(executor.clears.load(), 2);
    EXPECT_TRUE(executor.kernel_gate_.idle());
}

TEST(TmrKernelExecutionRoundTest, ThreadFailureCancelsAndEveryThreadReadsTheSameFinalVerdict) {
    ExecutorModel executor;
    executor.failing_run_index = 1;
    for (int32_t result : run_round(executor))
        EXPECT_EQ(result, -33);
    EXPECT_EQ(executor.runs.load(), ExecutorModel::kExecutionThreads);
    EXPECT_EQ(executor.cancellations.load(), 1);
    EXPECT_EQ(executor.kernel_cores_.finishes.load(), 1);
    EXPECT_EQ(executor.kernel_cores_.published.runtime_status, -33);
    EXPECT_EQ(executor.kernel_cores_.published.cleanup_status, 0);
    EXPECT_EQ(executor.clears.load(), 1);
    EXPECT_TRUE(executor.kernel_gate_.idle());
}

TEST(TmrKernelExecutionRoundTest, CleanupFailureRetainsStorageAndRejectsAnotherRound) {
    for (bool finish_failure : {false, true}) {
        SCOPED_TRACE(finish_failure);
        ExecutorModel executor;
        executor.kernel_cores_.finish_status = finish_failure ? -51 : 0;
        executor.finalize_status = finish_failure ? 0 : -52;
        const int32_t expected = finish_failure ? -51 : -52;
        for (int32_t result : run_round(executor))
            EXPECT_EQ(result, expected);
        EXPECT_EQ(executor.kernel_cores_.finishes.load(), 1);
        EXPECT_EQ(executor.kernel_cores_.publications.load(), 1);
        EXPECT_EQ(executor.kernel_cores_.published.runtime_status, 0);
        EXPECT_EQ(executor.kernel_cores_.published.cleanup_status, expected);
        EXPECT_EQ(executor.finalizations.load(), finish_failure ? 0 : 1);
        EXPECT_EQ(executor.clears.load(), 0);
        EXPECT_TRUE(executor.kernel_invocation_.active.load());
        EXPECT_EQ(executor.kernel_invocation_.storage.load(), ExecutorModel::kStorageValue);
        EXPECT_FALSE(executor.kernel_gate_.idle());
        EXPECT_EQ(execute_kernel_round_impl(executor, executor.request(), 10), -1);
        EXPECT_EQ(executor.preparations.load(), 1);
        EXPECT_EQ(executor.clears.load(), 0);
        EXPECT_EQ(executor.kernel_invocation_.storage.load(), ExecutorModel::kStorageValue);
    }
}

TEST(TmrKernelExecutionRoundTest, SlowReportPublisherBlocksNativeReturnAndOverwrite) {
    ExecutorModel executor;
    executor.complete_status = -47;
    Signal publishing;
    Signal release_finalizer;
    std::atomic<bool> report_complete{false};
    executor.kernel_cores_.on_publish = [&] {
        publishing.set();
        release_finalizer.wait();
        report_complete.store(true, std::memory_order_release);
    };
    const auto request = executor.request();
    std::array<int32_t, ExecutorModel::kLaunchedThreads> results{};
    std::atomic<int32_t> returned{0};
    std::vector<std::thread> threads;
    for (size_t i = 0; i < results.size(); ++i) {
        threads.emplace_back([&, i] {
            results[i] = execute_kernel_round_impl(executor, request, static_cast<int32_t>(10 + i));
            EXPECT_TRUE(report_complete.load(std::memory_order_acquire));
            ++returned;
        });
    }
    publishing.wait();
    EXPECT_EQ(returned.load(), 0);
    EXPECT_EQ(executor.clears.load(), 0);
    EXPECT_TRUE(executor.kernel_invocation_.active.load());
    EXPECT_EQ(executor.kernel_invocation_.storage.load(), ExecutorModel::kStorageValue);
    EXPECT_FALSE(executor.kernel_gate_.idle());
    EXPECT_EQ(execute_kernel_round_impl(executor, request, 10), -1);
    EXPECT_EQ(executor.preparations.load(), 1);
    release_finalizer.set();
    for (auto &thread : threads)
        thread.join();
    EXPECT_EQ(returned.load(), ExecutorModel::kLaunchedThreads);
    for (int32_t result : results)
        EXPECT_EQ(result, -47);
    EXPECT_EQ(executor.clears.load(), 1);
    EXPECT_FALSE(executor.kernel_invocation_.active.load());
    EXPECT_EQ(executor.kernel_invocation_.storage.load(), 0u);
    EXPECT_TRUE(executor.kernel_gate_.idle());
}

}  // namespace
