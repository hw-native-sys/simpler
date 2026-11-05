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

#include "fixture.h"
#include <sstream>
#include <unistd.h>
#define private public
#include "device_runner.h"
#undef private
#include "async_plan.h"
#include "common/device_run_result.h"
#include "scheduler/scheduler_types.h"

extern "C" int runtime_async_last_use_requirement(uint32_t, const void *, size_t, uint64_t) noexcept;

TEST(NativeAsyncLastUse, LegacyModeDoesNotParseAnUnrelatedRetainedImage) {
    EXPECT_EQ(runtime_async_last_use_requirement(3, nullptr, 0, 17), 0);
    EXPECT_NE(runtime_async_last_use_requirement(UINT32_MAX, nullptr, 0, 17), 0);
}

TEST(NativeAsyncLastUse, PreparationModeTravelsWithTheExistingResourceOwner) {
    auto resource = std::make_shared<PreparedRunResources>(
        nullptr, nullptr, nullptr, std::vector<WorkspaceManager::BackingLease>{}, SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH
    );
    PreparedRunUse retained(resource);
    resource.reset();
    EXPECT_EQ(retained.resources()->scheduler_runtime_mode(), SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH);
    PreparedRunResources unspecified(nullptr, nullptr, nullptr, {});
    EXPECT_EQ(unspecified.scheduler_runtime_mode(), UINT32_MAX);
    EXPECT_NE(runtime_async_last_use_requirement(unspecified.scheduler_runtime_mode(), nullptr, 0, 17), 0);
}

namespace native_admission_test {
void *empty_entry_points();
namespace {
enum class Result { SUCCESS, ERROR, OLD_EPOCH, ABSENT };
enum class Image { SYNC, DEFERRED, BAD_VERSION, OLD_EPOCH, TRUNCATED, LEGACY };

// Only submission and the device-produced bytes are faked. Retirement, drain,
// native poll/wait/finalize, resource facts and the raw fatal-close gate below
// are the production A5 and common implementations.
class AsyncRunner : public DeviceRunner {
public:
    explicit AsyncRunner(Image image = Image::DEFERRED, Result result = Result::SUCCESS) :
        image_(image),
        result_(result) {
        device_id_ = 0;
        EXPECT_EQ(execution_mode_latch().latch(SIMPLER_MODE_PROGRAM), 0);
        timeout_configuration_rc_ = 0;
        max_block_dim_ = block_dim_ = 1;
        EXPECT_EQ(install_staged_workspace(), 0);
        EXPECT_TRUE(set_run_resource_grant(6));
        enable_automatic_admission();
        CallableState callable;
        callable.hash = 1;
        callable.chip_buffer_hash = 1;
        callable.signature = {ArgDirection::IN};
        callable.host_orch_func_ptr = primary_entry_points();
        callables_[1] = std::move(callable);
        chip_callable_buffers_[1] = ChipCallableBuffer{};
    }
    ~AsyncRunner() override {
        EXPECT_EQ(finalize(), 0);
        device_id_ = -1;
    }
    int prepare_execution_streams(PreparedExecution &prepared, std::shared_ptr<const void> *lease) override {
        static int streams[2];
        prepared.streams = {&streams[0], &streams[1]};
        lease->reset();
        return 0;
    }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        auto prepared = std::make_unique<PreparedExecution>(identity, runtime, config, slot);
        prepared->scheduler_runtime_mode =
            image_ == Image::LEGACY ? SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH : SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING;
        prepared->resources_owned = true;
        // Model a published async run even while public admission still refuses
        // unqualified backends. Its image comes from the real Host planner.
        SchedulerTaskMetadata task{};
        task.flags = SCHEDULER_TASK_EXECUTABLE;
        task.active_mask = LaneMask::aiv0();
        task.logical_block_num = 1;
        SchedulerAsyncCapabilities capabilities{};
        capabilities[1] = image_ == Image::SYNC ?
                              CoreAsyncCapability{1, CoreAsyncMode::SYNCHRONOUS, 0, 0} :
                              CoreAsyncCapability{1, CoreAsyncMode::DEFERRED, 1, CORE_ASYNC_CONDITION_COUNTER};
        SchedulerAsyncHostPlan plan;
        if (!scheduler_plan_async_state({task}, {capabilities}, identity.run_epoch, 1 << 20, {}, &plan, nullptr))
            return PTO_RUNTIME_ERR_INTERNAL;
        const uint64_t summary_offset = sizeof(SchedulerRunControl);
        const uint64_t async_offset = summary_offset + sizeof(SchedulerGraphSummary);
        const uint64_t bytes = async_offset + plan.image.size();
        void *device = nullptr;
        void *host = nullptr;
        if (acquire_scheduler_state_storage(slot, bytes, 128, &device, &host) != 0) return PTO_RUNTIME_ERR_INTERNAL;
        std::memset(host, 0, bytes);
        auto *run = static_cast<SchedulerRunControl *>(host);
        run->expected_task_count = 1;
        run->graph_summary_offset = summary_offset;
        auto *summary = reinterpret_cast<SchedulerGraphSummary *>(static_cast<std::byte *>(host) + summary_offset);
        summary->run_epoch = identity.run_epoch;
        summary->async_layout_offset = plan.image.empty() ? 0 : async_offset;
        if (!plan.image.empty()) {
            std::memcpy(static_cast<std::byte *>(host) + async_offset, plan.image.data(), plan.image.size());
            auto *layout = reinterpret_cast<SchedulerAsyncLayout *>(static_cast<std::byte *>(host) + async_offset);
            if (image_ == Image::BAD_VERSION) ++layout->version;
            if (image_ == Image::OLD_EPOCH) --layout->run_epoch;
            if (image_ == Image::TRUNCATED) layout->total_bytes = UINT64_MAX;
        }
        submission_storage_[slot]->retain_replay_evidence = true;
        const int rc = ensure_device_run_result_region(slot, identity.run_epoch, prepared->kernel_args);
        if (rc != 0) return rc;
        *out = std::move(prepared);
        return 0;
    }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        auto transaction = exact_launch_transaction(
            prepared.identity, std::move(permit),
            [](LaunchProgressSink &sink) {
                sink.mark_submitted();
                return 0;
            },
            [](LaunchProgressSink &sink) {
                sink.mark_submitted();
                return 0;
            }
        );
        run_progress_.claim(prepared.identity);
        (void)run_progress_.publish(prepared.identity, RunProgressState::Submitted);
        EXPECT_EQ(run_fence(prepared.pipeline_slot).prepare_resources(), 0);
        EXPECT_EQ(arm_run_fence(prepared), 0);
        EXPECT_EQ(record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicore), 0);
        EXPECT_EQ(record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicpu), 0);
        auto *result = static_cast<DeviceRunResultRegion *>(submission_storage_[prepared.pipeline_slot]->device_result);
        *result = {};
        if (result_ != Result::ABSENT) {
            result->published = prepared.identity.run_epoch - (result_ == Result::OLD_EPOCH ? 1 : 0);
            result->verdict =
                static_cast<uint32_t>(result_ == Result::ERROR ? DeviceRunVerdict::Error : DeviceRunVerdict::Ok);
            result->completion_code = result_ == Result::ERROR ? -17 : 0;
            result->code_source = static_cast<uint32_t>(
                result_ == Result::ERROR ? DeviceRunCodeSource::ThreadRc : DeviceRunCodeSource::None
            );
        }
        return transaction;
    }
    // Healthy teardown never needs hardware recovery in this fake transport.
    // The raw poisoned path uses the real A5 fatal gate before any reset.
    int finalize_impl() override { return can_accept_run() ? finalize_common() : DeviceRunner::finalize_impl(); }
    bool holds_epoch(uint64_t epoch) const { return workspace_.holds_references(epoch); }
    bool fatal_reset_blocked() const { return prepared_resources_block_fatal_reset(); }
    void add_unconsumed_cross_wait(PreparedExecution &prepared) {
        const NativeRunIdentity waiter{prepared.identity.run_epoch + 1, 1, 1, 2};
        void *boundary = nullptr;
        ASSERT_EQ(
            queued_waits_->open(
                QueuedStreamWaits::Shape::CrossRun, waiter, prepared.identity, run_fence(prepared.pipeline_slot),
                RunCompletionFence::StreamRole::Aicpu, RunCompletionFence::StreamRole::Aicore, &boundary
            ),
            0
        );
        ASSERT_EQ(queued_waits_->commit(QueuedStreamWaits::Shape::CrossRun, waiter), 0);
        ASSERT_EQ(queued_waits_->record_proof(waiter, prepared.streams.aicore), 0);
        control().incomplete_event = control().last_recorded_event;
    }
    std::shared_ptr<const PreparedRunResources> retained(uint64_t epoch) const {
        for (const auto &entry : prepared_uses_)
            if (entry.identity.run_epoch == epoch) return entry.use.resources();
        return {};
    }

private:
    Image image_;
    Result result_;
};
}  // namespace

TEST_F(NativeAdmissionTest, AsyncSuccessUsesActualA5PollRetirementAndReleasesNativeOwner) {
    AsyncRunner runner;
    Attempt attempt(runner, 1, 20);
    ASSERT_EQ(attempt.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, attempt.storage), 0);
    EXPECT_EQ(simpler_poll_run(&runner, attempt.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    const auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    EXPECT_TRUE(state->physical_complete);
    EXPECT_TRUE(state->execution_retired);
    EXPECT_EQ(
        state->active_execution->prepared->prepared_resources->scheduler_runtime_mode(),
        SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING
    );
    EXPECT_EQ(simpler_wait_run(&runner, attempt.storage), 0);
    uint32_t proof = SIMPLER_NATIVE_RUN_LAST_USE_UNKNOWN;
    EXPECT_EQ(simpler_finalize_run_with_last_use_v1(&runner, attempt.storage, 1, &proof), 0);
    EXPECT_EQ(proof, SIMPLER_NATIVE_RUN_LAST_USE_PROVED);
    EXPECT_FALSE(runner.holds_epoch(20));
}

TEST_F(NativeAdmissionTest, AsyncLastUseProofIsCapturedWhenFinalizePerformsTheFirstDrain) {
    AsyncRunner runner;
    Attempt attempt(runner, 1, 20);
    ASSERT_EQ(attempt.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, attempt.storage), 0);
    uint32_t proof = 99;
    EXPECT_EQ(simpler_finalize_run_with_last_use_v1(&runner, attempt.storage, 0, &proof), 0);
    EXPECT_EQ(proof, SIMPLER_NATIVE_RUN_LAST_USE_PROVED);
    EXPECT_TRUE(attempt.empty());
}

TEST_F(NativeAdmissionTest, AsyncSuccessUsesActualA5BlockingDrainAndReleasesNativeOwner) {
    AsyncRunner runner;
    Attempt attempt(runner, 1, 20);
    ASSERT_EQ(attempt.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, attempt.storage), 0);
    EXPECT_EQ(simpler_wait_run(&runner, attempt.storage), 0);
    EXPECT_TRUE(static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage)->physical_complete);
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(runner.holds_epoch(20));
}

TEST_F(NativeAdmissionTest, StreamSuccessWithoutThisEpochsSuccessfulAsyncTerminalRetainsOwners) {
    for (bool poll_first : {false, true}) {
        for (Result result : {Result::ERROR, Result::OLD_EPOCH, Result::ABSENT}) {
            EXPECT_EXIT(
                {
                    AsyncRunner runner(Image::DEFERRED, result);
                    Attempt attempt(runner, 1, 20);
                    if (attempt.prepare() != 0 || simpler_launch_run(&runner, attempt.storage) != 0) _exit(2);
                    if (poll_first && simpler_poll_run(&runner, attempt.storage) != SIMPLER_NATIVE_RUN_POLL_ERROR)
                        _exit(3);
                    if (simpler_wait_run(&runner, attempt.storage) == 0) _exit(4);
                    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
                    if (state->physical_complete || state->execution_retired ||
                        !state->active_execution->prepared->resources_owned)
                        _exit(5);
                    uint32_t proof = 99;
                    if (simpler_finalize_run_with_last_use_v1(&runner, attempt.storage, 1, &proof) == 0 ||
                        proof != SIMPLER_NATIVE_RUN_LAST_USE_UNKNOWN || attempt.empty() || !runner.holds_epoch(20))
                        _exit(6);
                    if (!runner.fatal_reset_blocked() || finalize_device(&runner) == 0 || control().retired) _exit(7);
                    const auto retained = runner.retained(20);
                    if (!retained || retained->scheduler_runtime_mode() != SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING)
                        _exit(8);
                    _exit(0);
                },
                ::testing::ExitedWithCode(0), ""
            );
        }
    }
}

TEST_F(NativeAdmissionTest, DamagedAsyncDeclarationCannotAllowNativeRetirementOrRawReset) {
    for (Image image : {Image::BAD_VERSION, Image::OLD_EPOCH, Image::TRUNCATED}) {
        EXPECT_EXIT(
            {
                AsyncRunner runner(image);
                Attempt attempt(runner, 1, 20);
                if (attempt.prepare() != 0 || simpler_launch_run(&runner, attempt.storage) != 0) _exit(2);
                if (simpler_wait_run(&runner, attempt.storage) == 0 || !runner.fatal_reset_blocked()) _exit(3);
                if (finalize_device(&runner) == 0 || attempt.empty() || control().retired) _exit(4);
                _exit(0);
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, SynchronousAndLegacyRunsDoNotRequireAnAsyncTerminal) {
    for (Image image : {Image::SYNC, Image::LEGACY}) {
        AsyncRunner runner(image, Result::ABSENT);
        Attempt attempt(runner, 1, 20);
        ASSERT_EQ(attempt.prepare(), 0);
        ASSERT_EQ(simpler_launch_run(&runner, attempt.storage), 0);
        EXPECT_EQ(simpler_wait_run(&runner, attempt.storage), 0);
        EXPECT_FALSE(runner.fatal_reset_blocked());
        uint32_t proof = 99;
        EXPECT_EQ(simpler_finalize_run_with_last_use_v1(&runner, attempt.storage, 1, &proof), 0);
        EXPECT_EQ(proof, SIMPLER_NATIVE_RUN_LAST_USE_UNKNOWN);
    }
}

TEST_F(NativeAdmissionTest, AsyncPollWithUnconsumedWaitNeverLeaksACompleteLastUseFact) {
    for (bool query_error : {false, true}) {
        EXPECT_EXIT(
            {
                AsyncRunner runner;
                Attempt attempt(runner, 1, 20);
                if (attempt.prepare() != 0 || simpler_launch_run(&runner, attempt.storage) != 0) _exit(2);
                auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
                runner.add_unconsumed_cross_wait(*state->active_execution->prepared);
                control().incomplete_query_rc = query_error ? -47 : 0;
                const int expected = query_error ? SIMPLER_NATIVE_RUN_POLL_ERROR : SIMPLER_NATIVE_RUN_POLL_NOT_READY;
                if (simpler_poll_run(&runner, attempt.storage) != expected) _exit(3);
                if (state->physical_complete || state->active_execution->prepared->physical_complete ||
                    !state->active_execution->prepared->resources_owned || !runner.holds_epoch(20))
                    _exit(4);
                _exit(0);
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, AsyncStreamAndTerminalSuccessStillRequireQueuedWaitDischarge) {
    EXPECT_EXIT(
        {
            AsyncRunner runner;
            Attempt attempt(runner, 1, 20);
            if (attempt.prepare() != 0 || simpler_launch_run(&runner, attempt.storage) != 0) _exit(2);
            auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
            runner.add_unconsumed_cross_wait(*state->active_execution->prepared);
            control().event_wait_rc = -47;
            control().event_wait_hook = [] {
                control().stream_sync_rc = -47;
            };
            if (simpler_wait_run(&runner, attempt.storage) == 0) _exit(3);
            if (state->physical_complete || !state->active_execution->prepared->resources_owned ||
                !runner.fatal_reset_blocked() || simpler_finalize_run(&runner, attempt.storage) == 0 ||
                attempt.empty() || !runner.holds_epoch(20))
                _exit(4);
            if (finalize_device(&runner) == 0 || control().retired) _exit(5);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, CompletionOnlyWaitFailureCannotPublishExternalLastUse) {
    EXPECT_EXIT(
        {
            AsyncRunner runner;
            Attempt attempt(runner, 1, 20);
            if (attempt.prepare() != 0 || simpler_launch_run(&runner, attempt.storage) != 0) _exit(2);
            auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
            runner.add_unconsumed_cross_wait(*state->active_execution->prepared);
            control().event_wait_rc = -47;
            if (simpler_wait_run_completion(&runner, attempt.storage) == 0) _exit(3);
            if (state->physical_complete || state->active_execution->prepared->physical_complete ||
                !state->active_execution->prepared->resources_owned || !runner.holds_epoch(20))
                _exit(4);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, AsyncDrainExceptionRetainsTheExecutionAndCommunicationOwners) {
    EXPECT_EXIT(
        {
            AsyncRunner runner;
            Attempt attempt(runner, 1, 20);
            if (attempt.prepare() != 0 || simpler_launch_run(&runner, attempt.storage) != 0) _exit(2);
            control().stream_sync_hook = [] {
                throw std::runtime_error("transport failed before drain proof");
            };
            if (simpler_wait_run(&runner, attempt.storage) == 0) _exit(3);
            auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
            if (state->physical_complete || state->active_execution->prepared->physical_complete ||
                !state->active_execution->prepared->resources_owned || !runner.holds_epoch(20) ||
                !runner.fatal_reset_blocked())
                _exit(4);
            if (finalize_device(&runner) == 0 || control().retired) _exit(5);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, EmptyHbgGraphPublishesAnExplicitResidentMode) {
    class EmptyRunner : public Runner {
    public:
        EmptyRunner() { callables_.at(1).host_orch_func_ptr = empty_entry_points(); }
    } runner;
    Attempt attempt(runner, 1, 20);
    ASSERT_EQ(attempt.prepare(), 0);
    EXPECT_EQ(runner.last_scheduler_runtime_mode(), SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING);
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
}
}  // namespace native_admission_test
