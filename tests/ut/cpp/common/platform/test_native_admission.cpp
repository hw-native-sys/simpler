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
/** Real onboard workspace adapters and native prepare ownership over CPU transport. */
#include <cstdlib>
#include <new>
#include "callable.h"
#include "fixture.h"

namespace native_admission_test {
thread_local bool refuse_next_host_allocation = false;
}  // namespace native_admission_test

// Only the throwing, unaligned allocation is replaced. Its ordinary delete
// accepts malloc storage; aligned allocations retain their own allocator pair.
void *operator new(std::size_t bytes) {
    if (std::exchange(native_admission_test::refuse_next_host_allocation, false)) throw std::bad_alloc();
    if (void *storage = std::malloc(bytes == 0 ? 1 : bytes)) return storage;
    throw std::bad_alloc();
}

class NativeRunExecutionTestPeer {
public:
    static LaunchPermit mint(const NativeRunIdentity &identity) { return LaunchPermit(identity); }
};

namespace native_admission_test {
namespace {
int fake_current_device(void *, int *device_id) noexcept {
    *device_id = 0;
    return 0;
}
int fake_current_context(void *, void **context) noexcept {
    static int token;
    *context = &token;
    return 0;
}
int fake_create_hidden_stream(void *, void **stream) noexcept {
    static int tokens[2];
    static int next;
    *stream = &tokens[next++ % 2];
    return 0;
}
int fake_destroy_hidden_stream(void *, void *) noexcept { return 0; }
int fake_create_event(void *, void **event) noexcept {
    static int tokens[8];
    static int next;
    *event = &tokens[next++ % 8];
    return 0;
}
int fake_destroy_event(void *, void *) noexcept { return 0; }
}  // namespace

TEST_F(NativeAdmissionTest, RegistrationRefusesWhileANativeRunIsOutstanding) {
    Runner runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
    KernelContextOps ops;
    ops.context = &runner;
    ops.get_current_device = fake_current_device;
    ops.get_current_context = fake_current_context;
    ops.create_hidden_stream = fake_create_hidden_stream;
    ops.destroy_hidden_stream = fake_destroy_hidden_stream;
    ops.create_event = fake_create_event;
    ops.destroy_event = fake_destroy_event;
    ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    ASSERT_TRUE(runner.kernel_execution_state().accepts_dispatch());

    int owner = 0;
    ASSERT_TRUE(runner.try_reserve_native_run(&owner, 1, arena_bank(1), true));
    ASSERT_TRUE(runner.native_runs_outstanding());

    const uint8_t binary[] = {0x01};
    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    int32_t auto_id = 99;
    EXPECT_EQ(simpler_register_callable(&runner, 2, image.data()), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(simpler_kernel_mode_prepare_callable(&runner, 3, image.data(), image.size()), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(simpler_register_callable_v2(&runner, image.data(), image.size(), &auto_id), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(auto_id, -1);
    runner.release_native_run_reservation(&owner);
}

TEST_F(NativeAdmissionTest, AutomaticRegistrationDoesNotRetryFailedProgramInitialization) {
    class FreshRunner : public Runner {
    public:
        FreshRunner() {
            device_id_ = -1;
            callables_.clear();
            chip_callable_buffers_.clear();
        }
    } runner;
    const uint8_t binary[] = {0x01};
    control().fail_stream_create = 1;
    ASSERT_EQ(
        simpler_init(&runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0),
        -4242
    );
    ASSERT_EQ(control().stream_create_calls, 1u);
    ASSERT_EQ(runner.device_id(), 0);
    ASSERT_EQ(runner.execution_mode_latch().latched_mode(), SIMPLER_MODE_PROGRAM);
    ASSERT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);

    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    const auto allocations = control().allocations;
    for (int attempt = 0; attempt < 2; ++attempt) {
        int32_t id = 99;
        EXPECT_EQ(
            simpler_register_callable_v2(&runner, image.data(), image.size(), &id), PTO_RUNTIME_ERR_INVALID_STATE
        );
        EXPECT_EQ(id, -1);
        EXPECT_EQ(control().stream_create_calls, 1u);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_FALSE(runner.has_callable(0));
    }
    EXPECT_EQ(finalize_device(&runner), 0);
    int32_t id = 99;
    EXPECT_EQ(simpler_register_callable_v2(&runner, image.data(), image.size(), &id), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(id, -1);
}

TEST_F(NativeAdmissionTest, EarlyPrepareFailureDoesNotBlockTheNormalRunComposition) {
    EXPECT_EQ(simpler_prepare_run(nullptr, nullptr, 0, nullptr, nullptr, nullptr), PTO_RUNTIME_ERR_INTERNAL);
    Runner runner;
    Attempt failed(runner, 0, 10);
    failed.config.aicpu_thread_num = 1;
    EXPECT_EQ(failed.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(failed.consume_release());
    EXPECT_TRUE(failed.empty());
    Attempt next(runner, 0, 20);
    EXPECT_EQ(simpler_run(&runner, next.storage, 1, &next.args, &next.config, &next.descriptor), 0);
    EXPECT_TRUE(next.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_FALSE(runner.holds_epoch(20));
}

TEST_F(NativeAdmissionTest, OwnerAllocationFailureReturnsPreparedWithoutEnteringTheBackend) {
    Runner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    refuse_next_host_allocation = true;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    const bool allocation_attempted = !std::exchange(refuse_next_host_allocation, false);
    EXPECT_TRUE(allocation_attempted);
    EXPECT_EQ(outcome.rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(outcome.progress, LaunchProgress::NotStarted);
    EXPECT_EQ(outcome.prepared.get(), original);
    EXPECT_EQ(outcome.active, nullptr);
    EXPECT_FALSE(outcome.receipt.valid());
    EXPECT_EQ(runner.launch_calls, 0);
    EXPECT_EQ(runner.recovery_calls, 0);
    EXPECT_TRUE(runner.can_accept_run());
}

TEST_F(NativeAdmissionTest, FailurePolicyCannotReplaceTheResultOrLoseTheOwner) {
    Runner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    runner.launch_refusal_rc = -4321;
    runner.partial_launch = true;
    runner.recovery_throws = true;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    EXPECT_EQ(outcome.rc, -4321);
    EXPECT_EQ(outcome.progress, LaunchProgress::Partial);
    ASSERT_NE(outcome.active, nullptr);
    EXPECT_EQ(outcome.active->prepared.get(), original);
    EXPECT_FALSE(outcome.receipt.valid());
    EXPECT_FALSE(runner.can_accept_run());
    EXPECT_EQ(runner.recovery_calls, 1);
    EXPECT_EQ(runner.recovery_rc, -4321);
}

TEST_F(NativeAdmissionTest, CompleteSubmissionHandsOffItsOwnerWithoutAnyLaterAllocation) {
    Runner runner;
    Runtime runtime;
    const NativeRunIdentity identity{10, 10, 10, 0};
    auto prepared = std::make_unique<Runner::PreparedExecution>(identity, runtime, CallConfig{}, 0);
    auto *original = prepared.get();
    runner.forbid_post_submit_allocation = true;
    auto outcome = runner.launch_execution(std::move(prepared), NativeRunExecutionTestPeer::mint(identity));
    const bool nothing_allocated = std::exchange(refuse_next_host_allocation, false);
    EXPECT_TRUE(nothing_allocated);
    EXPECT_EQ(outcome.rc, 0);
    EXPECT_EQ(outcome.progress, LaunchProgress::Complete);
    ASSERT_NE(outcome.active, nullptr);
    EXPECT_EQ(outcome.active->prepared.get(), original);
    EXPECT_EQ(outcome.active->progress, LaunchProgress::Complete);
    EXPECT_EQ(outcome.prepared, nullptr);
    EXPECT_TRUE(outcome.receipt.matches(identity));
}

TEST_F(NativeAdmissionTest, UnstartedLaunchCleanupPublishesOnlyItsOwnReleaseProof) {
    Runner runner;
    Attempt rejected(runner, 0, 10);
    ASSERT_EQ(rejected.prepare(), 0);
    runner.launch_refusal_rc = -4321;
    EXPECT_EQ(simpler_launch_run(&runner, rejected.storage), -4321);
    EXPECT_EQ(simpler_poll_run(&runner, rejected.storage), SIMPLER_NATIVE_RUN_POLL_COMPLETE);
    EXPECT_EQ(simpler_finalize_run(&runner, rejected.storage), 0);
    EXPECT_TRUE(rejected.consume_release());
    EXPECT_TRUE(rejected.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_FALSE(runner.holds_epoch(10));
    runner.launch_refusal_rc = 0;
    Attempt next(runner, 0, 20);
    EXPECT_EQ(simpler_run(&runner, next.storage, 1, &next.args, &next.config, &next.descriptor), 0);
    EXPECT_TRUE(next.empty());
}

TEST_F(NativeAdmissionTest, PartialLaunchNeverPublishesAnUnstartedReleaseProof) {
    Runner runner;
    Attempt rejected(runner, 0, 10);
    ASSERT_EQ(rejected.prepare(), 0);
    runner.launch_refusal_rc = -4321;
    runner.partial_launch = true;
    EXPECT_EQ(simpler_launch_run(&runner, rejected.storage), -4321);
    EXPECT_EQ(simpler_wait_run(&runner, rejected.storage), -4321);
    EXPECT_EQ(simpler_finalize_run(&runner, rejected.storage), -4321);
    EXPECT_FALSE(rejected.consume_release());
    EXPECT_TRUE(rejected.empty());
}

TEST_F(NativeAdmissionTest, ThrowingBackendRetainsTheOwnerAndNeverPublishesNoSubmission) {
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(candidate.prepare(), 0);
    runner.launch_throws = true;
    EXPECT_EQ(simpler_launch_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.recovery_calls, 1);
    EXPECT_EQ(runner.recovery_rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(runner.can_accept_run());
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(candidate.storage);
    ASSERT_NE(state->active_execution, nullptr);
    EXPECT_EQ(state->active_execution->progress, LaunchProgress::Partial);
    EXPECT_TRUE(runner.holds_epoch(10));
    EXPECT_EQ(runner.drain_calls, 0);
    Attempt blocked(runner, 1, 20);
    const auto allocations = control().allocations;
    EXPECT_EQ(blocked.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(blocked.empty());
    EXPECT_EQ(control().allocations, allocations);

    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.drain_calls, 1);
    EXPECT_FALSE(candidate.consume_release());
    EXPECT_TRUE(candidate.empty());
}

TEST_F(NativeAdmissionTest, ThrowAfterSubmissionRetainsTheCarrierAndAllBindingsUntilDrain) {
    Runner runner;
    Attempt candidate(runner, 0, 10);
    volatile int32_t accepted = 73;
    candidate.descriptor.accepted_state = &accepted;
    candidate.descriptor.accepted_value = 1;
    ASSERT_EQ(candidate.prepare(), 0);
    const auto bytes = control().bytes;
    runner.launch_throws_after_submission = true;
    EXPECT_EQ(simpler_launch_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.recovery_calls, 1);
    EXPECT_EQ(runner.recovery_rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(runner.can_accept_run());
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(candidate.storage);
    ASSERT_NE(state->active_execution, nullptr);
    ASSERT_NE(state->active_execution->prepared, nullptr);
    EXPECT_EQ(state->active_execution->prepared->runtime, &state->runtime);
    EXPECT_EQ(state->active_execution->progress, LaunchProgress::Partial);
    EXPECT_EQ(accepted, 73);
    EXPECT_EQ(simpler_poll_run(&runner, candidate.storage), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
    EXPECT_EQ(runner.reservations(), 1u);
    EXPECT_TRUE(runner.holds_epoch(10));
    EXPECT_EQ(control().bytes, bytes);
    EXPECT_EQ(runner.drain_calls, 0);
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.drain_calls, 1);
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_FALSE(candidate.consume_release());
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_FALSE(runner.holds_epoch(10));
    EXPECT_EQ(runner.drain_calls, 1);
}

TEST_F(NativeAdmissionTest, UnstartedLaunchWithFailedCleanupPublishesNoReleaseProof) {
    // Failed cleanup deliberately retains workspace; subprocess exit owns its
    // final disposal rather than fabricating device-completion evidence.
    EXPECT_EXIT(
        {
            Runner runner;
            Attempt rejected(runner, 0, 10);
            if (rejected.prepare() != 0) std::_Exit(1);
            runner.launch_refusal_rc = -4321;
            if (simpler_launch_run(&runner, rejected.storage) != -4321) std::_Exit(2);
            control().attach_rc = -77;
            if (simpler_finalize_run(&runner, rejected.storage) != -77) std::_Exit(3);
            if (rejected.consume_release() || !rejected.empty()) std::_Exit(4);
            std::_Exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

TEST_F(NativeAdmissionTest, ActualStagingAdapterTrimsIdleStorageThenRetriesOnce) {
    Runner runner;
    control().ceiling = 50 * kUnit;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    runner.retire(0, 10);
    const auto calls = control().allocations;
    ASSERT_EQ(staging(runner, 1, 20, 40 * kUnit), 0);
    EXPECT_EQ(control().allocations - calls, 2u);
    EXPECT_EQ(runner.retained_temp_addr(0), 0u);
    EXPECT_EQ(runner.charged(), 40 * kUnit);
    EXPECT_EQ(runner.abort_run_plan(1, 20), 0);
}

TEST_F(NativeAdmissionTest, FortyWithThirtyFreeWaitsOnlyForItsNamedTwentyByteUnitsConsumer) {
    Runner runner;
    control().ceiling = 50 * kUnit;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    ASSERT_EQ(staging(runner, 1, 20, 40 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(runner.plan(1).staging_snapshot, std::vector<uint64_t>({10}));
    const auto calls = control().allocations;
    runner.retire(2, 99);
    EXPECT_EQ(staging(runner, 1, 21, 40 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(control().allocations, calls);
    runner.retire(0, 10);
    EXPECT_EQ(staging(runner, 1, 22, 40 * kUnit), 0);
    EXPECT_EQ(runner.charged(), 40 * kUnit);
    EXPECT_EQ(runner.abort_run_plan(1, 22), 0);
}

TEST_F(NativeAdmissionTest, ExhaustedConsumersCannotRecaptureALaterRun) {
    Runner runner;
    control().ceiling = 50 * kUnit;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    ASSERT_EQ(staging(runner, 1, 20, 60 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    runner.retire(0, 10);
    EXPECT_EQ(staging(runner, 1, 21, 60 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE);
    EXPECT_EQ(runner.abort_run_plan(1, 21), 0);
}

TEST_F(NativeAdmissionTest, NoncapacityAdmissionFailureNeverParks) {
    Runner runner;
    runner.stop_serving();
    EXPECT_EQ(staging(runner, 0, 10, 20 * kUnit), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(runner.plan(0).parked, AutoAdmissionState::Park::None);
    EXPECT_EQ(control().allocations, 0u);
    EXPECT_EQ(runner.abort_run_plan(0, 10), 0);
}

TEST_F(NativeAdmissionTest, MultiregionFailureKeepsEveryPublishedBackingAndRollsBackOnlyNewClaims) {
    Runner runner;
    {
        DeviceRunnerBase::WorkspacePlanScope scope(0, 10);
        ASSERT_EQ(runner.setup_static_arena(0, 2 * kUnit, 2 * kUnit, 2 * kUnit), 0);
        runner.note_run_resources_proven(0, 10);
    }
    runner.note_workspace_run_fact(0, 10, Fact::NoDeviceSubmission);
    runner.retire(0, 10);
    const std::vector<void *> old{
        runner.arena_region_backing(0, 0), runner.arena_region_backing(0, 1), runner.arena_region_backing(0, 2)
    };
    const size_t charged = runner.charged();
    control().ceiling = control().bytes + 4 * kUnit + DeviceArena::kDefaultBaseAlign - 1;
    for (uint64_t epoch : {20u, 30u, 40u}) {
        const auto staged = control().successful_allocations;
        {
            DeviceRunnerBase::WorkspacePlanScope scope(0, epoch);
            EXPECT_EQ(
                runner.setup_static_arena(0, 4 * kUnit, 4 * kUnit, 4 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE
            );
        }
        EXPECT_EQ(control().successful_allocations - staged, 2u);  // first region staged on both attempts
        for (size_t i = 0; i < old.size(); ++i)
            EXPECT_EQ(runner.arena_region_backing(0, i), old[i]);
        // Abort may follow attach failure, so it relinquishes claims without
        // freeing on an unproven thread. The final staged region stays charged.
        EXPECT_EQ(runner.abort_run_plan(0, epoch), 0);
        EXPECT_EQ(runner.charged(), charged + 4 * kUnit + DeviceArena::kDefaultBaseAlign - 1);
        const auto allocations = control().allocations;
        Attempt next(runner, 0, epoch + 1);
        next.config.aicpu_thread_num = 1;
        // The real C API must reclaim after attach and before launch-shape
        // validation or allocation, even when this next prepare also fails.
        EXPECT_EQ(next.prepare(), PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_TRUE(next.consume_release());
        EXPECT_TRUE(next.empty());
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(runner.charged(), charged);
        EXPECT_EQ(control().bytes, charged);
        for (size_t i = 0; i < old.size(); ++i)
            EXPECT_EQ(runner.arena_region_backing(0, i), old[i]);
    }
    // No further prepare: close is the final attached reclamation boundary.
    {
        DeviceRunnerBase::WorkspacePlanScope scope(0, 50);
        EXPECT_EQ(
            runner.setup_static_arena(0, 4 * kUnit, 4 * kUnit, 4 * kUnit), SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE
        );
    }
    EXPECT_EQ(runner.abort_run_plan(0, 50), 0);
    EXPECT_GT(runner.charged(), charged);
    EXPECT_EQ(runner.finalize(), 0);
    EXPECT_EQ(control().bytes, 0u);
}

TEST_F(NativeAdmissionTest, NativePrepareParksAndDestroysTheAttemptBeforeAnEarlyRetryAttachFailure) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20, 40 * kUnit);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_TRUE(runner.plan(1).holds_token);
    EXPECT_EQ(runner.plan(1).staging_snapshot, std::vector<uint64_t>({10}));
    const auto allocations = control().allocations;
    candidate.descriptor.run_epoch = 21;
    EXPECT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(control().allocations, allocations);
    control().attach_rc = -77;
    candidate.descriptor.run_epoch = 22;
    EXPECT_EQ(candidate.prepare(), -77);
    control().attach_rc = 0;
    EXPECT_TRUE(candidate.consume_release());
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_EQ(runner.plan(1).epoch, 0u);
    EXPECT_FALSE(runner.plan(1).holds_token);
    runner.retire(0, 10);
}

TEST_F(NativeAdmissionTest, NativePrepareResumesAfterItsLaunchedConsumerRetires) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    runner.retire(0, 10);
    control().ceiling = std::numeric_limits<size_t>::max();
    candidate.descriptor.run_epoch = 21;
    ASSERT_EQ(candidate.prepare(), 0);
    EXPECT_EQ(runner.reservations(), 1u);
    EXPECT_EQ(runner.plan(1).epoch, 0u);
    ASSERT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.reservations(), 0u);
}

TEST_F(NativeAdmissionTest, NativeAbortAndCloseReleaseParkedPlanAndGrowthPermission) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(simpler_abort_run_plan_ctx(&runner, 1, 19), 0);
    EXPECT_EQ(runner.plan(1).epoch, 20u);  // stale caller cannot settle this plan
    EXPECT_EQ(simpler_abort_run_plan_ctx(&runner, 1, 20), 0);
    EXPECT_FALSE(runner.plan(1).holds_token);
    EXPECT_EQ(runner.reservations(), 0u);
    runner.retire(0, 10);
    EXPECT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(runner.charged(), 0u);
}
TEST_F(NativeAdmissionTest, CloseOwnsAParkedAttemptBeforeItsCallback) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(callbacks, 0);
    runner.retire(0, 10);
    ASSERT_EQ(finalize_device(&runner), 0);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.get_prepare_plan(1), nullptr);
    EXPECT_EQ(runner.charged(), 0u);
    EXPECT_FALSE(runner.plan(1).holds_token);
    EXPECT_EQ(runner.plan(1).epoch, 0u);
    EXPECT_EQ(runner.reservations(), 0u);
}

TEST_F(NativeAdmissionTest, AWaitingPlanCannotReleaseBorrowsWhileItsNativeEpochStillOwnsExecution) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    control().ceiling = control().bytes;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    control().ceiling = std::numeric_limits<size_t>::max();
    void *caller = runner.allocate_caller_buffer(64);
    ASSERT_NE(caller, nullptr);
    CallerDeviceBuffers::Span span{reinterpret_cast<uint64_t>(caller), 64};
    ASSERT_TRUE(runner.borrow_caller_buffers(20, &span, 1));
    // Inject only stop evidence: a native owner still names this epoch. The
    // abort guard and caller allocation table are the production implementations.
    int owner = 0;
    NativeRunIdentity identity{20, 20, 20, 1};
    LaunchPermit permit;
    ASSERT_TRUE(runner.try_reserve_native_run(&owner, 1, candidate.descriptor.arena_bank, true));
    ASSERT_TRUE(runner.try_acquire_native_run(&owner, identity, &permit));
    const auto charged = runner.charged();
    EXPECT_NE(finalize_device(&runner), 0);
    EXPECT_EQ(runner.charged(), charged);
    EXPECT_NE(simpler_abort_run_plan_ctx(&runner, 1, 20), 0);
    EXPECT_NE(runner.free_caller_buffer(caller), 0);
    EXPECT_TRUE(runner.plan(1).holds_token);
    runner.release_native_run(&owner);
    runner.release_native_run_reservation(&owner);
    ASSERT_EQ(simpler_abort_run_plan_ctx(&runner, 1, 20), 0);
    runner.release_caller_buffers(20, false);
    EXPECT_EQ(runner.free_caller_buffer(caller), 0);
    runner.retire(0, 10);
}

}  // namespace native_admission_test
