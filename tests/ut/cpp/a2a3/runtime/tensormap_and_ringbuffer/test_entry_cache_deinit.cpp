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
#include <utility>
#include <vector>
#include <future>
#include <memory>
#include "host/raii_scope_guard.h"
#include "aicpu/native_invocation_completion.h"

// Include the implementation to exercise the private executor, without a
// production test hook or an alternative teardown implementation.
// NOLINTNEXTLINE(bugprone-suspicious-include)
#include "aicpu/aicpu_executor.cpp"

namespace {
std::vector<std::pair<const void *, size_t>> invalidations;
thread_local std::promise<void> *pending_poll = nullptr;
thread_local std::shared_future<void> release_pending;
NativeInvocationCompletion *native_completion = nullptr;
}  // namespace
// A clock read in drain follows a real PENDING poll of the SDMA record.
uint64_t get_sys_cnt_aicpu() {
    if (pending_poll != nullptr) {
        auto *signal = pending_poll;
        pending_poll = nullptr;
        signal->set_value();
        release_pending.wait();
    }
    return 1;
}
uint64_t platform_aicore_exit_deadline() { return UINT64_MAX; }
extern "C" uint64_t get_platform_run_result_epoch() { return 41; }
extern "C" bool aicpu_accept_runtime_terminal(RunTerminalSelection selection, const void *payload, size_t bytes) {
    return native_completion != nullptr && native_completion->accept_runtime_terminal(selection, payload, bytes);
}

class SchedulerContextTestPeer {
public:
    static DeferredCompletionSlab &prepare(SchedulerContext &context, RuntimeContext &runtime, TaskId task) {
        context.rt_ = &runtime;
        context.sched_ = &runtime.scheduler;
        context.cores_total_num_ = 1;
        context.retirement_state_[0] = SchedulerContext::RETIREMENT_RELEASED;
        context.completed_.store(true);
        context.payload_per_core_[0][0].local_context.async_ctx.task_token = task;
        return context.deferred_slab_per_core_[0][0];
    }
};
namespace aicpu_cache_maintenance {
void invalidate_range_impl(const void *address, size_t bytes) { invalidations.emplace_back(address, bytes); }
void flush_range_impl(const void *, size_t) {}
}  // namespace aicpu_cache_maintenance

TEST(EntryCacheDeinit, FailedSnapshotFallsBackToFullThroughExecutor) {
    Runtime runtime;
    runtime.dev.entry_args_source_ = static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope);
    runtime.dev.entry_tensor_count_ = 4;
    PlatformEntryArgs view{
        &runtime, LAUNCH_ENVELOPE_HEADER_BYTES, 4, 0, static_cast<uint32_t>(EntryArgsSource::LaunchEnvelope)
    };
    view.cache_maintenance = static_cast<uint32_t>(EntryCacheMaintenance::Live);
    // Direct executor tests provide the affinity cell normally bound by the wrapper.
    int32_t affinity_index = 0;
    ASSERT_EQ(platform_aicpu_affinity_bind_thread(&affinity_index), 0);
    auto unbind = RAIIScopeGuard([] {
        platform_aicpu_affinity_bind_thread(nullptr);
    });
    set_platform_entry_args(0, view);
    AicpuExecutor executor;
    for (auto verdict : {DeviceRunVerdict::Ok, DeviceRunVerdict::Error, DeviceRunVerdict::None, DeviceRunVerdict::Ok}) {
        const bool success = verdict == DeviceRunVerdict::Ok;
        executor.terminal_publisher_.take(
            RunTerminalSelection{
                verdict, success ? 0 : -7, success ? DeviceRunCodeSource::None : DeviceRunCodeSource::ThreadRc
            },
            nullptr, 0
        );
        invalidations.clear();
        executor.deinit(&runtime);
        if (success) {
            ASSERT_EQ(invalidations.size(), 2u);
            EXPECT_EQ(invalidations[0].first, &runtime);
            EXPECT_EQ(invalidations[0].second, simpler::tmr::entry_tensor_offset + 4 * sizeof(Tensor));
            EXPECT_EQ(
                invalidations[1].first,
                reinterpret_cast<const char *>(&runtime) + offsetof(DeviceRuntimeLaunchDesc, workers)
            );
        } else {
            ASSERT_EQ(invalidations.size(), 1u);
            EXPECT_EQ(invalidations[0].first, &runtime);
            EXPECT_EQ(invalidations[0].second, sizeof(runtime.dev));
        }
        EXPECT_EQ(executor.terminal_publisher_.snapshot_is_ok(), success);
    }
    set_platform_entry_args(0, {});
}

TEST(EntryCacheDeinit, AsyncSubmissionFailureDrainsPendingRecordsAndReturnsOriginalError) {
    for (const int code : {101, 102}) {
        for (const int primary : {code, SIMPLER_ERROR_TENSOR_WAIT_TIMEOUT}) {
            SCOPED_TRACE(primary);
            for (const bool failed_with_records : {false, true}) {
                if (code == 102 && failed_with_records) continue;
                auto executor = std::make_unique<AicpuExecutor>();
                auto runtime = std::make_unique<Runtime>();
                auto rt_context = std::make_unique<RuntimeContext>();
                auto header = std::make_unique<SharedMemoryHeader>();
                runtime->set_gm_sm_ptr(header.get());
                rt_context->scheduler.sm_header = header.get();
                const auto task = TaskId::make(0, 19);
                auto &slab = SchedulerContextTestPeer::prepare(executor->sched_ctx_, *rt_context, task);
                alignas(CHIP_ALIGN_SIZE) uint64_t record = 0;
                // Replace the device submit boundary; consume its real slab wire
                // format through the production scheduler and terminal publisher.
                slab.entries[0] = {reinterpret_cast<uint64_t>(&record), 7, 0, COMPLETION_ENGINE_SDMA,
                                   COMPLETION_TYPE_SDMA_EVENT_RECORD,   0};
                slab.count = 1;
                ASSERT_EQ(slab.count, 1u);
                ASSERT_EQ(poll_sdma_post_done_record(slab.entries[0].addr, 7).state, CompletionPollState::PENDING);
                // The post boundary has accepted A; B fails while A is provably pending.
                if (failed_with_records) {
                    slab.entries[1] = slab.entries[0];
                    slab.count = 2;
                }
                slab.error_code = code;
                ASSERT_EQ(slab.error_code, code);
                header->orch_error_code.store(primary);
                header->sched_error_code.store(code);
                std::promise<void> polled, release;
                auto polled_future = polled.get_future();
                auto release_future = release.get_future().share();
                auto snapshot = std::async(std::launch::async, [&] {
                    pending_poll = &polled;
                    release_pending = release_future;
                    executor->snapshot_run_terminal(runtime.get());
                    pending_poll = nullptr;
                });
                EXPECT_EQ(polled_future.wait_for(std::chrono::seconds(2)), std::future_status::ready);
                EXPECT_EQ(snapshot.wait_for(std::chrono::seconds(0)), std::future_status::timeout);
                __atomic_store_n(&record, 7, __ATOMIC_RELEASE);
                release.set_value();
                snapshot.get();
                NativeInvocationCompletion completion;
                ASSERT_TRUE(completion.begin(1));
                native_completion = &completion;
                ASSERT_TRUE(executor->terminal_publisher_.handoff());
                native_completion = nullptr;
                EXPECT_EQ(
                    completion.finish(
                        0, 0,
                        [&](RunTerminalSelection selection, const void *, size_t) {
                            EXPECT_EQ(selection.code, -primary);
                            return NativeTerminalPublication::Published;
                        }
                    ),
                    -primary
                );
                KernelErrorCleanup cleanup{};
                executor->sched_ctx_.drain_cancelled_run(UINT64_MAX, cleanup);
                EXPECT_EQ(cleanup.drained_conditions, failed_with_records ? 2u : 1u);
                // A submission failure can leave unknown request identities even
                // after its valid prefix has drained. It never grants reuse.
                EXPECT_EQ(cleanup.complete, 0u);
                EXPECT_EQ(slab.error_code, code);
                EXPECT_EQ(slab.count, failed_with_records ? 2u : 1u);
            }
        }
    }
}

TEST(EntryCacheDeinit, AsyncFailureRetainsPendingAndInvalidRecords) {
    for (const int code : {101, 102}) {
        auto executor = std::make_unique<AicpuExecutor>();
        auto runtime = std::make_unique<RuntimeContext>();
        auto &slab = SchedulerContextTestPeer::prepare(executor->sched_ctx_, *runtime, TaskId::make(0, 19));
        alignas(CHIP_ALIGN_SIZE) uint64_t record = 0;
        slab.count = 1;
        slab.error_code = code;
        slab.entries[0] = {reinterpret_cast<uint64_t>(&record), 7, 0, COMPLETION_ENGINE_SDMA,
                           COMPLETION_TYPE_SDMA_EVENT_RECORD,   0};
        KernelErrorCleanup cleanup{};
        executor->sched_ctx_.drain_cancelled_run(0, cleanup);
        EXPECT_EQ(cleanup.complete, 0u);
        EXPECT_EQ(cleanup.async_error, SIMPLER_ERROR_SCHEDULER_TIMEOUT);
        EXPECT_EQ(slab.count, 1u);
        EXPECT_EQ(slab.entries[0].backend_cookie, 7u);
        slab.entries[0].addr = 0;
        executor->sched_ctx_.drain_cancelled_run(UINT64_MAX, cleanup);
        EXPECT_EQ(cleanup.async_error, SIMPLER_ERROR_ASYNC_COMPLETION_INVALID);
        EXPECT_EQ(cleanup.complete, 0u);
        EXPECT_EQ(cleanup.drained_conditions, 0u);
        EXPECT_EQ(slab.error_code, code);
    }
}

TEST(EntryCacheDeinit, WaitListTrackingFailureCannotCertifyCompleteRetirement) {
    for (const int code : {101, 102}) {
        auto executor = std::make_unique<AicpuExecutor>();
        auto runtime = std::make_unique<RuntimeContext>();
        auto header = std::make_unique<SharedMemoryHeader>();
        header->sched_error_code.store(code);
        runtime->scheduler.sm_header = header.get();
        auto &slab = SchedulerContextTestPeer::prepare(executor->sched_ctx_, *runtime, TaskId::make(0, 19));
        alignas(CHIP_ALIGN_SIZE) uint64_t record = 7;
        slab.count = 1;
        slab.entries[0] = {reinterpret_cast<uint64_t>(&record), 7, 0, COMPLETION_ENGINE_SDMA,
                           COMPLETION_TYPE_SDMA_EVENT_RECORD,   0};
        KernelErrorCleanup cleanup{};
        executor->sched_ctx_.drain_cancelled_run(UINT64_MAX, cleanup);
        EXPECT_EQ(cleanup.drained_conditions, 1u);
        EXPECT_EQ(cleanup.complete, 0u);
        EXPECT_EQ(cleanup.async_error, SIMPLER_ERROR_ASYNC_REGISTRATION_FAILED);
        EXPECT_EQ(header->sched_error_code.load(), code);
    }
}
