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
#include <unistd.h>
#include <array>
#include <map>
#include <optional>
#include <string>
#include <tuple>
#include "callable.h"
#include "fixture.h"

extern "C" void simpler_bind_device_fault_monitor(void *);
extern "C" int simpler_execution_environment_releasable_ctx(DeviceContextHandle);

namespace native_admission_test {
thread_local bool refuse_next_host_allocation = false;
thread_local int host_allocations_until_failure = -1;
namespace {
int initialize_capture_context(Runner &runner);
}

static int prepare_kernel(Attempt &candidate) {
    candidate.args.tensor(0).address_space = AddressSpace::DEVICE;
    candidate.args.tensor(0).transfer = TensorTransfer::NONE;
    int rc = candidate.runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL);
    if (rc == 0) rc = initialize_capture_context(candidate.runner);
    return rc != 0 ? rc : candidate.prepare();
}

TEST_F(NativeAdmissionTest, NullCallerIsRefusedBeforeSdkQueryOrPreparedStateMutation) {
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(candidate.prepare(), 0);
    auto &prepared = *candidate.state()->prepared_execution;
    void *const caller = prepared.caller_stream;
    const auto calls = control().capture_info_calls;
    control().capture_info_rc = 100000;
    EXPECT_EQ(
        simpler_launch_run_on_stream(&runner, candidate.storage, nullptr, nullptr), PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_EQ(runner.validate_execution_stream(prepared), PTO_RUNTIME_ERR_INVALID_STATE);
    control().capture_info_rc = 0;
    EXPECT_EQ(control().capture_info_calls, calls);
    EXPECT_EQ(prepared.caller_stream, caller);
    EXPECT_FALSE(prepared.recorded_submission);
    EXPECT_EQ(candidate.state()->phase.load(), NativeRunPhase::Prepared);
    EXPECT_EQ(runner.launch_calls, 0);
    EXPECT_EQ(simpler_launch_run(&runner, candidate.storage), 0);
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), 0);
}

TEST_F(NativeAdmissionTest, CaptureRefusalPreservesProgramEagerLaunchAndCompletion) {
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(candidate.prepare(), 0);
    auto &prepared = *candidate.state()->prepared_execution;
    ASSERT_EQ(prepared.caller_stream, nullptr);
    ASSERT_FALSE(prepared.recorded_submission);
    control().capture_active = true;
    EXPECT_EQ(
        simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr),
        PTO_RUNTIME_ERR_INVALID_STATE
    );
    control().capture_active = false;
    EXPECT_EQ(prepared.caller_stream, nullptr);
    EXPECT_FALSE(prepared.recorded_submission);
    EXPECT_EQ(candidate.state()->phase.load(), NativeRunPhase::Prepared);
    EXPECT_EQ(runner.launch_calls, 0);
    ASSERT_EQ(simpler_launch_run(&runner, candidate.storage), 0);
    EXPECT_EQ(simpler_poll_run(&runner, candidate.storage), SIMPLER_NATIVE_RUN_POLL_NOT_READY);
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), 0);
    EXPECT_EQ(runner.drain_calls, 1);
}

TEST_F(NativeAdmissionTest, FailedStreamQueryPreservesCallerAndPriorRecordingState) {
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(candidate.prepare(), 0);
    auto &prepared = *candidate.state()->prepared_execution;
    prepared.recorded_submission = true;
    control().capture_info_rc = -4246;
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), -4246);
    control().capture_info_rc = 0;
    EXPECT_EQ(prepared.caller_stream, nullptr);
    EXPECT_TRUE(prepared.recorded_submission);
    EXPECT_EQ(candidate.state()->phase.load(), NativeRunPhase::Prepared);
    EXPECT_EQ(runner.launch_calls, 0);
}

TEST_F(NativeAdmissionTest, RetainHostAllocationFailuresPreservePreparedEagerState) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    for (int fail_after : {0, 1}) {
        SCOPED_TRACE(fail_after);
        Runner runner;
        Attempt candidate(runner, 0, 10);
        ASSERT_EQ(prepare_kernel(candidate), 0);
        auto *state = candidate.state();
        auto *const prepared = state->prepared_execution.get();
        void *const caller = prepared->caller_stream;
        const auto claims = runner.native_run_claim_count();
        const auto reservations = runner.reservations();
        const auto allocations = control().allocations;
        RuntimeHandle reference = candidate.storage;
        host_allocations_until_failure = fail_after;
        const int rc = simpler_retain_run_resources(&runner, candidate.storage, candidate.caller_stream(), &reference);
        const int remaining = std::exchange(host_allocations_until_failure, -1);
        EXPECT_EQ(remaining, -1);
        EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(reference, nullptr);
        EXPECT_EQ(state->prepared_execution.get(), prepared);
        EXPECT_EQ(prepared->caller_stream, caller);
        EXPECT_FALSE(prepared->recorded_submission);
        EXPECT_EQ(state->phase.load(), NativeRunPhase::Prepared);
        EXPECT_FALSE(state->runner_claimed);
        EXPECT_EQ(state->graph_reference_count.load(), 0u);
        EXPECT_EQ(state->kernel_execution, nullptr);
        EXPECT_EQ(state->capture_result, nullptr);
        EXPECT_EQ(runner.kernel_capture_owner(), nullptr);
        EXPECT_EQ(runner.native_run_claim_count(), claims);
        EXPECT_EQ(runner.reservations(), reservations);
        EXPECT_EQ(control().allocations, allocations);
        ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), 0);
        EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), 0);
        EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    }
}

TEST_F(NativeAdmissionTest, UnsupportedRetentionPreservesPreparedEagerAdmission) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "host_build_graph") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_EQ(simpler_kernel_capture_supported(&runner), 0);
    auto *state = candidate.state();
    ASSERT_NE(state, nullptr);
    auto *prepared = state->prepared_execution.get();
    ASSERT_NE(prepared, nullptr);
    void *const caller = prepared->caller_stream;
    ASSERT_NE(caller, candidate.caller_stream());
    const auto claims = runner.native_run_claim_count();
    const auto reservations = runner.reservations();
    const auto allocations = control().allocations;
    ASSERT_EQ(claims, 0u);
    for (int attempt = 0; attempt < 2; ++attempt) {
        SCOPED_TRACE(attempt);
        RuntimeHandle reference = candidate.storage;
        EXPECT_EQ(
            simpler_retain_run_resources(&runner, candidate.storage, candidate.caller_stream(), &reference),
            PTO_RUNTIME_ERR_UNSUPPORTED
        );
        EXPECT_EQ(reference, nullptr);
        EXPECT_EQ(state->prepared_execution.get(), prepared);
        EXPECT_EQ(prepared->caller_stream, caller);
        EXPECT_EQ(state->phase.load(), NativeRunPhase::Prepared);
        EXPECT_FALSE(state->runner_claimed);
        EXPECT_EQ(runner.native_run_claim_count(), claims);
        EXPECT_EQ(runner.reservations(), reservations);
        EXPECT_EQ(state->graph_reference_count.load(), 0u);
        EXPECT_EQ(state->kernel_execution, nullptr);
        EXPECT_EQ(state->capture_result, nullptr);
        EXPECT_EQ(runner.kernel_capture_owner(), nullptr);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(runner.launch_calls, 0);
    }
    Attempt successor(runner, 1, 20);
    successor.args.tensor(0).address_space = AddressSpace::DEVICE;
    successor.args.tensor(0).transfer = TensorTransfer::NONE;
    EXPECT_EQ(successor.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(successor.empty());
    EXPECT_EQ(runner.reservations(), reservations);
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), 0);
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), 0);
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.native_run_claim_count(), 0u);
    EXPECT_EQ(runner.reservations(), 0u);
}

static simpler::tmr::KernelCaptureControl *inject_execution_resources(Attempt &candidate, uint64_t epoch) {
    auto &runner = candidate.runner;
    auto *state = candidate.state();
    state->kernel_execution = std::make_unique<KernelCaptureOwner>();
    auto &owner = *state->kernel_execution;
    owner.graph_reference_count = &state->graph_reference_count;
    owner.caller_stream = candidate.caller_stream();
    auto *control = static_cast<simpler::tmr::KernelCaptureControl *>(
        runner.allocate_tensor(sizeof(simpler::tmr::KernelCaptureControl))
    );
    if (control == nullptr) return nullptr;
    *control = {};
    // Device evidence is injected; reference acquisition, common retirement and
    // cleanup execute through their production C API and workspace ledger.
    owner.reference = {reinterpret_cast<uint64_t>(control), 17, 23};
    owner.registration.reference = owner.reference;
    owner.registration.binding_epoch = candidate.descriptor.run_epoch;
    auto &kernel_args = state->prepared_execution->kernel_args;
    if (runner.ensure_device_run_result_region(
            candidate.descriptor.pipeline_slot, candidate.descriptor.run_epoch, kernel_args
        ) != 0)
        return nullptr;
    owner.registration.run_result_address = kernel_args.args.run_result_data_base;
    owner.control_initialized = true;
    owner.ready = true;
    control->capture_generation = 17;
    control->context_generation = 23;
    control->binding_epoch = candidate.descriptor.run_epoch;
    control->decision = static_cast<uint32_t>(simpler::tmr::CaptureDecision::Admitted);
    control->replay_epoch = epoch == 0 ? candidate.descriptor.run_epoch - 1 : candidate.descriptor.run_epoch + epoch;
    control->completed_epoch = epoch == 0 ? 0 : control->replay_epoch;
    state->prepared_execution->kernel_resources = &owner;
    state->retained_device_use.retain(state->identity());
    runner.note_workspace_run_fact(
        candidate.descriptor.pipeline_slot, candidate.descriptor.run_epoch,
        WorkspaceManager::RunFact::DeviceReferencesRetained
    );
    state->prepared_execution->streams = {
        runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu),
        runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicore)
    };
    return control;
}

static simpler::tmr::KernelCaptureControl *retain_capture(Attempt &candidate, uint64_t epoch) {
    auto *control = inject_execution_resources(candidate, epoch);
    if (control == nullptr ||
        simpler_retain_run_resources(
            &candidate.runner, candidate.storage, candidate.caller_stream(), &candidate.resource_reference
        ) != 0)
        return nullptr;
    return control;
}

static int release_reference(Attempt &candidate) {
    if (!candidate.empty()) {
        const int rc = simpler_finalize_run(&candidate.runner, candidate.storage);
        if (!candidate.empty()) return rc;
    }
    const int rc = simpler_release_run_resources(&candidate.runner, candidate.resource_reference);
    if (rc == 0) candidate.resource_reference = nullptr;
    return rc;
}

TEST_F(NativeAdmissionTest, RefusedGraphRetentionDoesNotReplaceTheOrdinaryResultView) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(inject_execution_resources(candidate, 0), nullptr);
    int other_caller = 0;
    RuntimeHandle reference = nullptr;
    EXPECT_EQ(
        simpler_retain_run_resources(&runner, candidate.storage, &other_caller, &reference),
        PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_EQ(reference, nullptr);
    auto *state = candidate.state();
    EXPECT_EQ(state->capture_result, nullptr);
    DeviceRunResultRegion cached{};
    cached.published = candidate.descriptor.run_epoch;
    cached.payload_bytes = 1;
    cached.payload[0] = 0x34;
    runner.inject_cached_result(0, cached);
    size_t bytes = 0;
    const auto *payload = static_cast<const uint8_t *>(state->host_api.run_result(&bytes));
    ASSERT_NE(payload, nullptr);
    EXPECT_EQ(bytes, 1u);
    EXPECT_EQ(payload[0], 0x34);
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
}

TEST_F(NativeAdmissionTest, CompletedEagerRetiresWithoutSynchronizingQueuedSuccessors) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    auto *device = inject_execution_resources(candidate, 1);
    ASSERT_NE(device, nullptr);
    device->launched_threads = device->arrived_threads = device->finished_threads = 2;
    device->replay_epoch = device->completed_epoch = candidate.descriptor.run_epoch;
    auto &owner = *candidate.state()->kernel_execution;
    owner.registered = true;
    owner.registration.action = static_cast<uint32_t>(simpler::tmr::CaptureRegistrationAction::RegisterOnce);
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), 0);
    ASSERT_EQ(simpler_wait_run(&runner, candidate.storage), 0);
    const auto syncs = control().capture_sync_calls;
    control().stream_sync_rc = -4321;
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(control().capture_sync_calls, syncs);
    control().stream_sync_rc = 0;
    if (!candidate.empty()) candidate.state()->kernel_execution->registered = false;
}

static void graph_result_does_not_poison_ordinary_run(int transfer_error) {
    Runner runner;
    runner.use_common_submission = true;
    Attempt graph(runner, 0, 1);
    ASSERT_EQ(prepare_kernel(graph), 0);
    ASSERT_NE(retain_capture(graph, 1), nullptr);
    auto *result =
        reinterpret_cast<DeviceRunResultRegion *>(graph.state()->kernel_execution->registration.run_result_address);
    ASSERT_EQ(runner.prewarm_common_submission(*graph.state()->prepared_execution), 0);
    control().capture_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, graph.storage, graph.caller_stream(), nullptr), 0);
    control().capture_active = false;
    *result = {};
    result->published = graph.descriptor.run_epoch;
    result->verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    ASSERT_EQ(simpler_finalize_run(&runner, graph.storage), 0);
    control().run_result_copy_rc = transfer_error;
    const int released = simpler_release_run_resources(&runner, graph.resource_reference);
    control().run_result_copy_rc = 0;
    ASSERT_EQ(released, 0);
    graph.resource_reference = nullptr;

    Attempt ordinary(runner, 0, 2);
    ordinary.args.tensor(0).address_space = AddressSpace::DEVICE;
    ordinary.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(ordinary.prepare(), 0);
    auto &args = ordinary.state()->prepared_execution->kernel_args;
    ASSERT_EQ(runner.ensure_device_run_result_region(0, ordinary.descriptor.run_epoch, args), 0);
    auto *next = reinterpret_cast<DeviceRunResultRegion *>(args.args.run_result_data_base);
    *next = {};
    next->published = ordinary.descriptor.run_epoch;
    next->verdict = static_cast<uint32_t>(DeviceRunVerdict::Ok);
    const auto copies = control().run_result_copies;
    EXPECT_EQ(runner.read_device_run_result(0, ordinary.descriptor.run_epoch), 0);
    EXPECT_EQ(control().run_result_copies, copies + 1);
    EXPECT_EQ(runner.device_run_terminal(0, ordinary.descriptor.run_epoch).state, DeviceRunTerminalState::Succeeded);
    EXPECT_EQ(simpler_finalize_run(&runner, ordinary.storage), 0);
}

TEST_F(NativeAdmissionTest, GraphReplaySequenceCannotReuseAnOrdinaryResultsCachedBytes) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    graph_result_does_not_poison_ordinary_run(0);
}

TEST_F(NativeAdmissionTest, GraphReadFailureCannotBecomeTheNextOrdinaryRunsTransferError) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    graph_result_does_not_poison_ordinary_run(-4321);
}

static void graph_observation_survives_cleanup_retry(int runtime_error) {
    Runner runner;
    Attempt graph(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(graph), 0);
    auto *device = retain_capture(graph, 1);
    ASSERT_NE(device, nullptr);
    device->first_error = runtime_error;
    auto *state = graph.state();
    control().capture_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, graph.storage, graph.caller_stream(), nullptr), 0);
    control().capture_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, graph.storage), 0);
    state->kernel_execution->registered = true;
    control().capture_registration_throws = true;
    control().run_result_copy_rc = -4321;
    const int failed_revoke = simpler_release_run_resources(&runner, graph.resource_reference);
    control().capture_registration_throws = false;
    control().run_result_copy_rc = 0;
    ASSERT_EQ(failed_revoke, PTO_RUNTIME_ERR_INTERNAL);
    ASSERT_NE(state->kernel_execution, nullptr);
    const auto first_result_copies = control().run_result_copies;
    state->runner_resources_owned = true;
    runner.abandon_rc = -77;
    const int failed_cleanup = simpler_release_run_resources(&runner, graph.resource_reference);
    ASSERT_EQ(failed_cleanup, -77);
    EXPECT_EQ(control().run_result_copies, first_result_copies);
    ASSERT_NE(state->capture_result, nullptr);
    EXPECT_EQ(state->kernel_execution, nullptr);
    EXPECT_EQ(state->completion_rc, runtime_error);
    EXPECT_EQ(state->capture_result->transfer_rc, -4321);
    EXPECT_TRUE(state->capture_result->read_attempted);
    EXPECT_TRUE(state->capture_result->boundaries_complete);
    size_t payload_bytes = 17;
    EXPECT_EQ(state->host_api.run_result(&payload_bytes), nullptr);
    EXPECT_EQ(payload_bytes, 0u);
    const auto copies = control().run_result_copies;
    runner.abandon_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, graph.resource_reference), 0);
    graph.resource_reference = nullptr;
    EXPECT_EQ(control().run_result_copies, copies);
}

TEST_F(NativeAdmissionTest, GraphObservationKeepsItsFirstTransferErrorAcrossCleanupRetries) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    graph_observation_survives_cleanup_retry(-9);
}

TEST_F(NativeAdmissionTest, GraphSnapshotReadFailureDoesNotRewriteACompletedRuntimeResult) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    graph_observation_survives_cleanup_retry(0);
}

TEST_F(NativeAdmissionTest, CommonEagerWaitReadsTheRuntimeFaultAfterAProvedDrain) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    auto *device = inject_execution_resources(candidate, 1);
    ASSERT_NE(device, nullptr);
    device->replay_epoch = device->completed_epoch = candidate.descriptor.run_epoch;
    device->launched_threads = device->arrived_threads = device->finished_threads = 2;
    device->first_error = -9;
    candidate.state()->kernel_execution->registration.action =
        static_cast<uint32_t>(simpler::tmr::CaptureRegistrationAction::RegisterOnce);
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), 0);
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), -9);
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), -9);
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), -9);
    EXPECT_TRUE(candidate.empty());
}

TEST_F(NativeAdmissionTest, UnexecutedEagerCleanupPreservesItsEarlierSubmissionError) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(inject_execution_resources(candidate, 0), nullptr);
    candidate.state()->kernel_execution->registration.action =
        static_cast<uint32_t>(simpler::tmr::CaptureRegistrationAction::RegisterOnce);
    runner.partial_launch = true;
    runner.launch_refusal_rc = -6543;
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), -6543);
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), -6543);
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), -6543);
    EXPECT_TRUE(candidate.empty());
}

TEST_F(NativeAdmissionTest, RuntimeFailureInspectionDoesNotRetainAReleasedGraph) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    auto *device = retain_capture(candidate, 1);
    ASSERT_NE(device, nullptr);
    device->first_error = -9;
    control().capture_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), 0);
    control().capture_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    DeviceRunResultRegion result{};
    inject_runtime_failure_result(result, candidate.descriptor.run_epoch);
    auto *result_state = resolve_native_run_storage<DeviceRunnerBase>(candidate.resource_reference);
    ASSERT_NE(result_state, nullptr);
    *reinterpret_cast<DeviceRunResultRegion *>(result_state->kernel_execution->registration.run_result_address) =
        result;
    ASSERT_EQ(simpler_wait_run(&runner, candidate.resource_reference), -9);
    const int released = simpler_release_run_resources(&runner, candidate.resource_reference);
    EXPECT_EQ(released, 0);
    if (released != 0) {
        runner.inject_cached_result(0, {});
        EXPECT_EQ(simpler_release_run_resources(&runner, candidate.resource_reference), 0);
    }
    candidate.resource_reference = nullptr;
    EXPECT_EQ(runner.reservations(), 0u);
}

TEST_F(NativeAdmissionTest, FailedGraphCleanupKeepsItsRuntimeResultAndReportsItsCleanupError) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    auto *device = retain_capture(candidate, 1);
    ASSERT_NE(device, nullptr);
    device->first_error = -9;
    auto *state = candidate.state();
    control().capture_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), 0);
    control().capture_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    DeviceRunResultRegion result{};
    inject_runtime_failure_result(result, candidate.descriptor.run_epoch);
    auto *result_state = resolve_native_run_storage<DeviceRunnerBase>(candidate.resource_reference);
    ASSERT_NE(result_state, nullptr);
    auto *source_result =
        reinterpret_cast<DeviceRunResultRegion *>(result_state->kernel_execution->registration.run_result_address);
    *source_result = result;
    ASSERT_GT(result.payload_bytes, 0u);
    auto stale = result;
    stale.payload[0] ^= 0xff;
    runner.inject_cached_result(0, stale);
    ASSERT_EQ(simpler_wait_run(&runner, candidate.resource_reference), -9);
    state->runner_resources_owned = true;
    runner.abandon_rc = -4321;
    EXPECT_EQ(simpler_release_run_resources(&runner, candidate.resource_reference), -4321);
    EXPECT_EQ(runner.reservations(), 1u);
    EXPECT_EQ(state->kernel_execution, nullptr);
    size_t payload_bytes = 0;
    const auto *payload = static_cast<const uint8_t *>(state->host_api.run_result(&payload_bytes));
    EXPECT_NE(payload, nullptr);
    EXPECT_EQ(payload_bytes, result.payload_bytes);
    if (payload != nullptr && payload_bytes == result.payload_bytes)
        EXPECT_EQ(std::memcmp(payload, result.payload, payload_bytes), 0);
    *source_result = {};
    runner.inject_cached_result(0, stale);
    payload = static_cast<const uint8_t *>(state->host_api.run_result(&payload_bytes));
    if (payload != nullptr && payload_bytes == result.payload_bytes)
        EXPECT_EQ(std::memcmp(payload, result.payload, payload_bytes), 0);
    const auto result_copies = control().run_result_copies;
    EXPECT_EQ(simpler_wait_run(&runner, candidate.resource_reference), -9);
    runner.abandon_rc = 0;
    const int released = simpler_release_run_resources(&runner, candidate.resource_reference);
    EXPECT_EQ(released, 0);
    EXPECT_EQ(control().run_result_copies, result_copies);
    if (released != 0) {
        runner.inject_cached_result(0, {});
        EXPECT_EQ(simpler_release_run_resources(&runner, candidate.resource_reference), 0);
    }
    candidate.resource_reference = nullptr;
    EXPECT_EQ(runner.reservations(), 0u);
}

TEST_F(NativeAdmissionTest, EagerAndRecordedLaunchUseTheSameProductionCommandSequence) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    using Op = native_admission_transport::SubmissionOp;
    using Step = std::tuple<Op, int, int>;
    std::array<std::vector<Step>, 2> traces;
    for (int recorded = 0; recorded < 2; ++recorded) {
        Runner runner;
        runner.use_common_submission = true;
        Attempt candidate(runner, 0, 10);
        ASSERT_EQ(prepare_kernel(candidate), 0);
        auto *device = retain_capture(candidate, 0);
        ASSERT_NE(device, nullptr);
        auto *prepared = candidate.state()->prepared_execution.get();
        ASSERT_EQ(runner.prewarm_common_submission(*prepared), 0);
        auto &transport = control();
        transport.submissions.clear();
        transport.submissions.reserve(16);
        transport.trace_submission = true;
        transport.capture_active = recorded != 0;
        const size_t events = transport.event_create_calls;
        ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr), 0);
        EXPECT_EQ(transport.event_create_calls, events);
        EXPECT_EQ(runner.facts(0, candidate.descriptor.run_epoch).launched, recorded == 0);
        std::map<void *, int> event_ids;
        for (const auto &step : transport.submissions) {
            const int stream = step.stream == candidate.caller_stream() ? 0 :
                               step.stream == prepared->streams.aicpu   ? 1 :
                                                                          2;
            int event = -1;
            if (step.event != nullptr) {
                const auto [it, inserted] = event_ids.emplace(step.event, static_cast<int>(event_ids.size()));
                (void)inserted;
                event = it->second;
            }
            traces[recorded].emplace_back(step.operation, stream, event);
        }
        transport.trace_submission = false;
        transport.capture_active = false;
        if (!recorded) device->replay_epoch = device->completed_epoch = candidate.descriptor.run_epoch;
        EXPECT_EQ(release_reference(candidate), 0);
    }
    const std::vector<Step> expected{{Op::Record, 0, 0}, {Op::Wait, 1, 0}, {Op::Admission, 1, -1},
                                     {Op::Record, 1, 1}, {Op::Wait, 2, 1}, {Op::Core, 2, -1},
                                     {Op::Record, 2, 2}, {Op::Cpu, 1, -1}, {Op::Wait, 1, 2},
                                     {Op::Record, 1, 3}, {Op::Wait, 0, 3}, {Op::Record, 0, 4}};
    EXPECT_EQ(traces[0], expected);
    EXPECT_EQ(traces[1], traces[0]);
}

TEST_F(NativeAdmissionTest, FailedCommonPayloadRetainsPartialGraphSubmissionUntilQuiescentRelease) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    using Op = native_admission_transport::SubmissionOp;
    for (Op failed : {Op::Admission, Op::Core, Op::Cpu}) {
        Runner runner;
        runner.use_common_submission = true;
        Attempt candidate(runner, 0, 10);
        ASSERT_EQ(prepare_kernel(candidate), 0);
        ASSERT_NE(retain_capture(candidate, 0), nullptr);
        auto *state = candidate.state();
        ASSERT_EQ(runner.prewarm_common_submission(*state->prepared_execution), 0);
        control().submissions.clear();
        control().submissions.reserve(16);
        control().trace_submission = true;
        control().capture_active = true;
        control().fail_submission = failed;
        EXPECT_EQ(
            simpler_launch_run_on_stream(&runner, candidate.storage, candidate.caller_stream(), nullptr),
            control().submission_rc
        );
        ASSERT_NE(state->active_execution, nullptr);
        EXPECT_EQ(state->active_execution->progress, LaunchProgress::Partial);
        EXPECT_NE(candidate.resource_reference, nullptr);
        EXPECT_EQ(runner.reservations(), 1u);
        control().capture_active = false;
        control().trace_submission = false;
        control().fail_submission = Op::None;
        EXPECT_EQ(release_reference(candidate), 0);
        EXPECT_EQ(runner.reservations(), 0u);
    }
}

TEST_F(NativeAdmissionTest, CaptureReleaseReportsObservedExecutionInsteadOfNoSubmission) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(retain_capture(candidate, 3), nullptr);
    EXPECT_EQ(release_reference(candidate), 0);
    EXPECT_TRUE(candidate.empty());
    const auto facts = runner.facts(0, 10);
    EXPECT_TRUE(facts.launched);
    EXPECT_TRUE(facts.drain_proved);
    EXPECT_FALSE(facts.no_submission);
}

TEST_F(NativeAdmissionTest, UnexecutedCaptureRetiresOnlyWithAQuiescenceProof) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(retain_capture(candidate, 0), nullptr);
    auto *state = candidate.state();
    void *caller = candidate.caller_stream();
    state->kernel_execution->caller_stream = caller;
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    EXPECT_TRUE(candidate.empty());
    ASSERT_TRUE(runner.can_accept_run());
    control().stream_sync_rc = -4321;
    runner.recovery_throws = true;
    EXPECT_EQ(release_reference(candidate), -4321);
    EXPECT_FALSE(runner.can_accept_run());
    EXPECT_EQ(runner.recovery_rc, -4321);
    EXPECT_NE(candidate.resource_reference, nullptr);
    auto facts = runner.facts(0, 10);
    EXPECT_FALSE(facts.launched);
    EXPECT_FALSE(facts.no_submission);
    EXPECT_FALSE(facts.drain_proved);
    EXPECT_TRUE(runner.holds_epoch(10));
    runner.recovery_throws = false;
    control().stream_sync_rc = 0;
    EXPECT_EQ(release_reference(candidate), 0);
    facts = runner.facts(0, 10);
    EXPECT_FALSE(facts.launched);
    EXPECT_FALSE(facts.no_submission);
    EXPECT_TRUE(facts.drain_proved);
    EXPECT_FALSE(runner.can_accept_run());
}

TEST_F(NativeAdmissionTest, CaptureConsumesLateFaultWithoutRewritingItsRuntimeResult) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    DeviceFaultMonitor monitor(
        {[] {
             return 0;
         },
         [] {
             return 0;
         },
         [] {
             return static_cast<long>(getpid());
         }}
    );
    struct Binding {
        explicit Binding(DeviceFaultMonitor &m) { simpler_bind_device_fault_monitor(&m); }
        ~Binding() { simpler_bind_device_fault_monitor(nullptr); }
    } binding(monitor);
    Runner runner;
    ASSERT_EQ(runner.monitor_faults(), 0);
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(retain_capture(candidate, 3), nullptr);
    auto *state = candidate.state();
    void *caller = candidate.caller_stream();
    state->kernel_execution->caller_stream = caller;
    runner.note_run_stream_identity(caller);
    monitor.report(DeviceFaultNotice{0, 1, 7, 507001, 0});
    control().capture_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, caller, nullptr), 0);
    control().capture_active = false;
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), 0);
    EXPECT_EQ(state->observed_replay_sequence, 13u);
    EXPECT_FALSE(runner.accepts_new_run());
    EXPECT_EQ(release_reference(candidate), 0);
}

TEST_F(NativeAdmissionTest, GraphResourcesSurviveOriginalHandleRetirementAndStorageReuse) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(retain_capture(candidate, 0), nullptr);
    auto *state = candidate.state();
    const size_t close_calls = control().loader_finalize_calls;
    ASSERT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    EXPECT_TRUE(candidate.empty());
    EXPECT_TRUE(runner.holds_epoch(10));
    EXPECT_EQ(resolve_native_run_storage<DeviceRunnerBase>(candidate.resource_reference), state);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    EXPECT_NE(finalize_device(&runner), 0);
    EXPECT_EQ(control().loader_finalize_calls, close_calls);
    std::memset(candidate.storage, 0xa5, get_runtime_size());
    EXPECT_EQ(simpler_release_run_resources(&runner, candidate.resource_reference), 0);
    candidate.resource_reference = nullptr;
    uint64_t overwritten = 0;
    std::memcpy(&overwritten, candidate.storage, sizeof(overwritten));
    EXPECT_EQ(overwritten, UINT64_C(0xa5a5a5a5a5a5a5a5));
    EXPECT_FALSE(runner.holds_epoch(10));
    std::memset(candidate.storage, 0, get_runtime_size());
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    ASSERT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(control().loader_finalize_calls, close_calls + 1);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 1);
}

TEST_F(NativeAdmissionTest, CommonWaitReadsLaterReplayInsteadOfCachedCompletion) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    auto *device = retain_capture(candidate, 3);
    ASSERT_NE(device, nullptr);
    void *caller = candidate.caller_stream();
    control().capture_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, candidate.storage, caller, nullptr), 0);
    control().capture_active = false;
    EXPECT_EQ(simpler_wait_run(&runner, candidate.storage), 0);
    EXPECT_EQ(candidate.state()->phase.load(), NativeRunPhase::Complete);
    device->replay_epoch = device->completed_epoch = 14;
    device->first_error = -77;
    EXPECT_EQ(simpler_wait_run(&runner, candidate.resource_reference), -77);
    EXPECT_EQ(candidate.state()->observed_replay_sequence, 14u);
    EXPECT_EQ(release_reference(candidate), 0);
}

TEST_F(NativeAdmissionTest, CaptureControlReadFailureRetainsItsLeaseForRetry) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(retain_capture(candidate, 2), nullptr);
    control().copy_rc = -4321;
    EXPECT_EQ(release_reference(candidate), -4321);
    EXPECT_NE(candidate.resource_reference, nullptr);
    EXPECT_TRUE(runner.holds_epoch(10));
    control().copy_rc = 0;
    EXPECT_EQ(release_reference(candidate), 0);
    EXPECT_FALSE(runner.holds_epoch(10));
}

}  // namespace native_admission_test

// Only the throwing, unaligned allocation is replaced. Its ordinary delete
// accepts malloc storage; aligned allocations retain their own allocator pair.
void *operator new(std::size_t bytes) {
    const bool refuse_next = std::exchange(native_admission_test::refuse_next_host_allocation, false);
    auto &remaining = native_admission_test::host_allocations_until_failure;
    const bool refuse_counted = remaining == 0;
    if (remaining >= 0) --remaining;
    if (refuse_next || refuse_counted) throw std::bad_alloc();
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
    if (control().attach_rc != 0) return control().attach_rc;
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

class TimeoutEnvironment {
public:
    explicit TimeoutEnvironment(const std::array<const char *, 3> &values) {
        for (size_t i = 0; i < names_.size(); ++i) {
            if (const char *old = std::getenv(names_[i])) saved_[i] = old;
            if (values[i] != nullptr) setenv(names_[i], values[i], 1);
            else unsetenv(names_[i]);
        }
    }
    ~TimeoutEnvironment() {
        for (size_t i = 0; i < names_.size(); ++i) {
            if (saved_[i]) setenv(names_[i], saved_[i]->c_str(), 1);
            else unsetenv(names_[i]);
        }
    }

private:
    static constexpr std::array<const char *, 3> names_{
        SIMPLER_OP_EXECUTE_TIMEOUT_US_ENV, SIMPLER_STREAM_SYNC_TIMEOUT_MS_ENV, SIMPLER_SCHEDULER_TIMEOUT_MS_ENV
    };
    std::array<std::optional<std::string>, 3> saved_;
};

class TimeoutRunner : public Runner {
public:
    TimeoutRunner() {
        device_id_ = -1;
        callables_.clear();
        chip_callable_buffers_.clear();
    }
    HostRuntimeTimeoutConfig timeouts() const { return timeout_config_; }
};

int initialize_capture_context(Runner &runner) {
    KernelContextOps ops;
    ops.get_current_device = fake_current_device;
    ops.get_current_context = fake_current_context;
    ops.create_hidden_stream = fake_create_hidden_stream;
    ops.destroy_hidden_stream = fake_destroy_hidden_stream;
    ops.create_event = fake_create_event;
    ops.destroy_event = fake_destroy_event;
    const int rc = runner.kernel_execution_state().initialize(0, ops);
    return rc != 0 ? rc : runner.kernel_execution_state().mark_ready_enqueued();
}
}  // namespace

TEST_F(NativeAdmissionTest, ProgramColdCleanupDebtAbandonsOnlyAfterResetAttempt) {
    class CloseRunner : public Runner {
    public:
        void own_sources() {
            ASSERT_NE(mem_alloc_.alloc(64), nullptr);
            ASSERT_EQ(rtStreamCreate(&stream_aicpu_, 0), 0);
        }
        bool pending() const { return common_cleanup_pending_; }
        void retire(bool reset_ok) { retire_loader_after_device_teardown(reset_ok); }
    };
    for (int reset_error : {0, -4245}) {
        SCOPED_TRACE(reset_error);
        CloseRunner runner;
        runner.own_sources();
        control().loader_finalize_error = -4243;
        EXPECT_EQ(finalize_device(&runner), -4243);
        EXPECT_TRUE(runner.pending());
        EXPECT_EQ(control().bytes, 64u);
        EXPECT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);
        control().reset_error = reset_error;
        EXPECT_EQ(rtDeviceReset(0), reset_error);
        runner.retire(reset_error == 0);
        EXPECT_FALSE(runner.pending());
        EXPECT_EQ(control().calls_after_retire, 0u);
        EXPECT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);
        EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
        // Provider fixture cleanup is external to every per-resource RTS path.
        native_admission_transport::release_provider_state();
        control().retired = false;
        control().loader_finalize_error = 0;
    }
}

TEST_F(NativeAdmissionTest, NativeReleasableProofReadsRealSuccessfulCloseAndOutstandingUsers) {
    Runner runner;
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    ASSERT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 1);
    int owner = 0;
    ASSERT_TRUE(runner.try_reserve_native_run(&owner, 1, arena_bank(1), true));
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    const size_t close_calls = control().loader_finalize_calls;
    EXPECT_NE(finalize_device(&runner), 0);
    EXPECT_EQ(control().loader_finalize_calls, close_calls);
    runner.release_native_run_reservation(&owner);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 0);
    ASSERT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(simpler_execution_environment_releasable_ctx(&runner), 1);
}

TEST_F(NativeAdmissionTest, KernelColdCleanupFailureStopsAdmissionAndRetainsItsControlStream) {
    Runner runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
    KernelContextOps ops{&runner,
                         fake_current_device,
                         fake_current_context,
                         fake_create_hidden_stream,
                         fake_destroy_hidden_stream,
                         fake_create_event,
                         fake_destroy_event};
    ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
    ASSERT_TRUE(runner.kernel_execution_state().accepts_dispatch());
    void *stream = runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu);
    control().loader_finalize_error = -4243;
    ASSERT_EQ(finalize_device(&runner), -4243);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Closing);
    EXPECT_FALSE(runner.kernel_execution_state().accepts_dispatch());
    EXPECT_EQ(runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu), stream);
    const uint8_t binary[] = {1};
    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    EXPECT_EQ(
        simpler_kernel_mode_prepare_callable(&runner, 3, image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_STATE
    );
    control().loader_finalize_error = 0;
    EXPECT_EQ(finalize_device(&runner), 0);
    EXPECT_EQ(runner.kernel_execution_state().phase(), KernelContextPhase::Closed);
    EXPECT_FALSE(runner.kernel_execution_state().has_live_resources());
}

TEST_F(NativeAdmissionTest, RoundedTimeoutContinuesAndPreservesStreamCreationFailure) {
    TimeoutEnvironment env({"3000000", "4000", "2000"});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = 4294967;
    control().fail_stream_create = 1;
    const uint8_t binary[] = {1};
    EXPECT_EQ(
        simpler_init(&runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0),
        -4242
    );
    EXPECT_EQ(control().stream_create_calls, 1u);
    EXPECT_EQ(control().op_timeout_calls, 1u);
    EXPECT_EQ(runner.device_id(), 0);
    EXPECT_EQ(runner.timeouts().op_execute_timeout_us, 3000000u);
    EXPECT_EQ(runner.timeouts().actual_op_execute_timeout_us, 4294967u);
    EXPECT_EQ(runner.timeouts().stream_sync_timeout_ms, 4000);
    EXPECT_EQ(runner.timeouts().scheduler_timeout_ms, 2000);
    EXPECT_EQ(runner.initialization_error(), nullptr);
    EXPECT_FALSE(runner.program_initialized());
    EXPECT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);
    for (int attempt = 0; attempt != 2; ++attempt) {
        EXPECT_EQ(runner.attach_current_thread(0), 0);
    }
    EXPECT_EQ(control().stream_create_calls, 1u);
    EXPECT_EQ(control().op_timeout_calls, 1u);
    EXPECT_EQ(
        simpler_init(&runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0),
        PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_EQ(control().op_timeout_calls, 1u);
}

TEST_F(NativeAdmissionTest, OperatorTimeoutSetterFailureIsPreservedBeforeStreams) {
    TimeoutEnvironment env({nullptr, nullptr, nullptr});
    TimeoutRunner runner;
    control().op_timeout_rc = -4243;
    control().fail_stream_create = 1;
    const uint8_t binary[] = {1};
    EXPECT_EQ(
        simpler_init(&runner, 0, binary, sizeof(binary), binary, sizeof(binary), nullptr, 0, nullptr, 0, nullptr, 0),
        -4243
    );
    EXPECT_EQ(control().stream_create_calls, 0u);
    EXPECT_EQ(runner.device_id(), 0);
    EXPECT_FALSE(runner.program_initialized());
    EXPECT_EQ(runner.teardown_proof(), TeardownProof::Unresolved);
    for (int attempt = 0; attempt != 2; ++attempt) {
        EXPECT_EQ(runner.attach_current_thread(0), -4243);
        EXPECT_EQ(runner.ensure_device_initialized(), -4243);
    }
    EXPECT_EQ(control().stream_create_calls, 0u);
    EXPECT_EQ(control().op_timeout_calls, 1u);
}

TEST_F(NativeAdmissionTest, ActualTimeoutAtDefaultSchedulerDoesNotRejectAttachment) {
    TimeoutEnvironment env({nullptr, nullptr, nullptr});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = static_cast<uint64_t>(PLATFORM_SCHEDULER_TIMEOUT_MS) * 1000;
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().scheduler_timeout_ms, 0);
    EXPECT_EQ(runner.timeouts().op_execute_timeout_us, PLATFORM_OP_EXECUTE_TIMEOUT_US);
    EXPECT_EQ(runner.device_id(), 0);
}

TEST_F(NativeAdmissionTest, ActualTimeoutAtStreamBudgetDoesNotRejectAttachment) {
    TimeoutEnvironment env({"3000000", "5000", "1000"});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = 5000000;
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().stream_sync_timeout_ms, 5000);
    EXPECT_EQ(runner.initialization_error(), nullptr);
}

TEST_F(NativeAdmissionTest, ZeroReportedTimeoutDoesNotRejectSuccessfulSetter) {
    TimeoutEnvironment env({nullptr, nullptr, nullptr});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().actual_op_execute_timeout_us, 0u);
    EXPECT_EQ(runner.initialization_error(), nullptr);
}

TEST_F(NativeAdmissionTest, ValidAdjustedTimeoutPreservesRequestAndIsConfiguredOnce) {
    TimeoutEnvironment env({"3000000", "8000", "1000"});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = 4294967;
    ASSERT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().op_execute_timeout_us, 3000000u);
    EXPECT_EQ(runner.timeouts().actual_op_execute_timeout_us, 4294967u);
    EXPECT_EQ(control().requested_op_timeout_us, 3000000u);
    EXPECT_EQ(runner.timeouts().stream_sync_timeout_ms, 8000);
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(control().op_timeout_calls, 1u);
}

TEST_F(NativeAdmissionTest, ValidDownwardAdjustmentIsAccepted) {
    TimeoutEnvironment env({nullptr, nullptr, nullptr});
    TimeoutRunner runner;
    control().replace_op_timeout = true;
    control().actual_op_timeout_us = 30000000;
    EXPECT_EQ(runner.attach_current_thread(0), 0);
    EXPECT_EQ(runner.timeouts().op_execute_timeout_us, PLATFORM_OP_EXECUTE_TIMEOUT_US);
}

TEST_F(NativeAdmissionTest, BorrowedDeviceDoesNotConfigureTheGlobalOperatorTimeout) {
    TimeoutEnvironment env({"3000000", "4000", "2000"});
    TimeoutRunner runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
    KernelContextOps ops;
    ops.get_current_device = fake_current_device;
    ops.get_current_context = fake_current_context;
    ops.create_hidden_stream = fake_create_hidden_stream;
    ops.destroy_hidden_stream = fake_destroy_hidden_stream;
    ops.create_event = fake_create_event;
    ops.destroy_event = fake_destroy_event;
    ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
    control().op_timeout_rc = -4243;
    EXPECT_EQ(runner.adopt_borrowed_device(0), 0);
    EXPECT_EQ(control().op_timeout_calls, 0u);
    EXPECT_EQ(runner.timeouts().actual_op_execute_timeout_us, 0u);
}

TEST_F(NativeAdmissionTest, CoreLaunchPreservesMainlineAbiAndProgramReportIdentity) {
    Runner runner;
    runner.seed_cold_callable();
    control().capture_active = true;
    KernelArgs args{};
    args.run_result_epoch = 73;
    for (const auto source :
         {EntryArgsSource::KernelInvocation, EntryArgsSource::LaunchEnvelope, EntryArgsSource::Descriptor}) {
        SCOPED_TRACE(static_cast<uint32_t>(source));
        args.entry_args_source = static_cast<uint32_t>(source);
        ASSERT_EQ(runner.launch_common_core(args), 0);
        EXPECT_EQ(control().core_payload_bytes, sizeof(AicoreLaunchArgs));
        EXPECT_EQ(control().core_report_epoch, source == EntryArgsSource::KernelInvocation ? 0u : 73u);
        EXPECT_EQ(args.run_result_epoch, 73u);
    }
    control().capture_active = false;
}

TEST_F(NativeAdmissionTest, UnknownColdRegistrationIsRevokedBeforeControlRelease) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    runner.use_common_submission = true;
    runner.seed_cold_callable();
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    auto &prepared = *candidate.state()->prepared_execution;
    prepared.runtime->set_orch_args(candidate.args);
    prepared.launch_aicpu_num = 2;
    prepared.num_aicore = 1;
    DeviceRunResultRegion terminal{};
    prepared.kernel_args.args.run_result_data_base = reinterpret_cast<uint64_t>(&terminal);
    prepared.streams = {
        runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicpu),
        runner.kernel_execution_state().hidden_stream(KernelStreamKind::Aicore)
    };
    ASSERT_EQ(runner.prewarm_common_submission(prepared, true), 0);
    control().capture_registration_throws = true;
    RuntimeHandle reference = nullptr;
    EXPECT_EQ(
        simpler_retain_run_resources(&runner, candidate.storage, candidate.caller_stream(), &reference),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(reference, nullptr);
    ASSERT_NE(candidate.state()->kernel_execution, nullptr);
    EXPECT_NE(candidate.state()->kernel_execution->reference.control_address, 0u);
    EXPECT_TRUE(candidate.state()->kernel_execution->cold_pending);
    control().capture_registration_throws = false;
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    EXPECT_TRUE(candidate.empty());
    // An enqueue exception carries no proof that registration was not queued.
    // Quiescence must be followed by revoke before freeing its control.
    EXPECT_EQ(control().capture_registration_calls, 2u);
}

TEST_F(NativeAdmissionTest, CaptureReleaseLoaderExceptionRetainsRegistrationForRetry) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(retain_capture(candidate, 0), nullptr);
    auto &owner = *candidate.state()->kernel_execution;
    owner.registered = true;
    const auto control_address = owner.reference.control_address;
    control().capture_registration_throws = true;
    bool escaped = false;
    int rc = 0;
    try {
        rc = release_reference(candidate);
    } catch (...) {
        escaped = true;
    }
    control().capture_registration_throws = false;
    EXPECT_FALSE(escaped);
    EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_NE(candidate.resource_reference, nullptr);
    EXPECT_TRUE(owner.registered);
    EXPECT_EQ(owner.reference.control_address, control_address);
    EXPECT_FALSE(runner.kernel_execution_state().accepts_dispatch());
    EXPECT_EQ(control().capture_registration_calls, 1u);
    EXPECT_EQ(release_reference(candidate), 0);
    EXPECT_EQ(control().capture_registration_calls, 2u);
    EXPECT_TRUE(candidate.empty());
}

TEST_F(NativeAdmissionTest, CommonLaunchExceptionsRetainGraphResourcesForQuiescentRetry) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    for (const bool throw_after_submission : {true, false}) {
        SCOPED_TRACE(throw_after_submission);
        Runner runner;
        Attempt candidate(runner, 0, 10);
        ASSERT_EQ(prepare_kernel(candidate), 0);
        ASSERT_NE(retain_capture(candidate, 0), nullptr);
        control().capture_active = true;
        runner.launch_throws_after_submission = throw_after_submission;
        runner.partial_launch = !throw_after_submission;
        runner.launch_refusal_rc = throw_after_submission ? 0 : PTO_RUNTIME_ERR_INTERNAL;
        void *caller = candidate.caller_stream();
        int rc = 0;
        bool escaped = false;
        try {
            rc = simpler_launch_run_on_stream(&runner, candidate.storage, caller, nullptr);
        } catch (...) {
            escaped = true;
        }
        EXPECT_FALSE(escaped);
        EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(runner.launch_calls, 1);
        ASSERT_NE(candidate.state()->active_execution, nullptr);
        EXPECT_EQ(candidate.state()->active_execution->progress, LaunchProgress::Partial);
        control().capture_active = false;
        control().stream_sync_rc = -4321;
        EXPECT_EQ(release_reference(candidate), -4321);
        EXPECT_NE(candidate.resource_reference, nullptr);
        control().stream_sync_rc = 0;
        EXPECT_EQ(release_reference(candidate), 0);
        EXPECT_TRUE(candidate.empty());
    }
}

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
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(runner.has_callable(1));
    runner.release_native_run_reservation(&owner);
}

TEST_F(NativeAdmissionTest, WrongContextDoesNotConsumeCaptureFaultNotices) {
    if (std::strcmp(SIMPLER_RUNTIME_NAME, "tensormap_and_ringbuffer") != 0) GTEST_SKIP();
    DeviceFaultMonitor monitor(
        {[] {
             return 0;
         },
         [] {
             return 0;
         },
         [] {
             return static_cast<long>(getpid());
         }}
    );
    struct Binding {
        explicit Binding(DeviceFaultMonitor &m) { simpler_bind_device_fault_monitor(&m); }
        ~Binding() { simpler_bind_device_fault_monitor(nullptr); }
    } binding(monitor);
    Runner runner;
    ASSERT_EQ(runner.monitor_faults(), 0);
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(prepare_kernel(candidate), 0);
    ASSERT_NE(retain_capture(candidate, 0), nullptr);
    auto *state = candidate.state();
    state->kernel_execution->registered = true;
    void *caller = candidate.caller_stream();
    runner.note_run_stream_identity(caller);
    monitor.report(DeviceFaultNotice{0, 1, 7, 507001, 0});
    control().attach_rc = -4321;
    EXPECT_NE(simpler_launch_run_on_stream(&runner, candidate.storage, caller, nullptr), 0);
    EXPECT_TRUE(runner.accepts_new_run());
    EXPECT_EQ(runner.launch_calls, 0);
    EXPECT_EQ(state->phase.load(), NativeRunPhase::Prepared);
    EXPECT_EQ(state->kernel_execution->caller_stream, caller);
    EXPECT_NE(simpler_release_run_resources(&runner, candidate.resource_reference), 0);
    EXPECT_TRUE(runner.accepts_new_run());
    control().attach_rc = 0;
    EXPECT_EQ(runner.consume_device_fault_notices(), 1u);
    state->kernel_execution->registered = false;
    EXPECT_EQ(release_reference(candidate), 0);
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

TEST_F(NativeAdmissionTest, InvalidCaptureOwnerCannotMutateTheCallableRegistry) {
    Runner runner;
    Attempt candidate(runner, 0, 10);
    ASSERT_EQ(candidate.prepare(), 0);
    runner.set_kernel_capture_owner(candidate.storage);
    const uint8_t binary[] = {0x01};
    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    const auto allocations = control().allocations;
    EXPECT_EQ(simpler_register_callable(&runner, 2, image.data()), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(
        simpler_kernel_mode_prepare_callable(&runner, 2, image.data(), image.size()), PTO_RUNTIME_ERR_INVALID_STATE
    );
    int32_t auto_id = 99;
    EXPECT_EQ(
        simpler_register_callable_v2(&runner, image.data(), image.size(), &auto_id), PTO_RUNTIME_ERR_INVALID_STATE
    );
    EXPECT_EQ(auto_id, -1);
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(runner.has_callable(1));
    EXPECT_FALSE(runner.has_callable(2));
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_TRUE(runner.native_runs_outstanding());
    runner.set_kernel_capture_owner(nullptr);
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
    auto *state = candidate.state();
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
    auto *state = candidate.state();
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

TEST_F(NativeAdmissionTest, LostDrainAttachmentStopsAdmissionWithoutReleasingSubmittedWork) {
    for (bool finalize_without_wait : {false, true}) {
        SCOPED_TRACE(finalize_without_wait);
        EXPECT_EXIT(
            {
                Runner runner;
                Attempt active(runner, 0, 10);
                if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
                control().attach_rc = -77;
                const int rc = finalize_without_wait ? simpler_finalize_run(&runner, active.storage) :
                                                       simpler_wait_run(&runner, active.storage);
                if (rc != -77) std::_Exit(2);
                if (runner.accepts_new_run()) std::_Exit(3);
                SimplerWorkspaceReport report{};
                if (!runner.workspace_report(&report) || active.consume_release()) std::_Exit(4);
                if (finalize_without_wait ? report.quarantined_blocks == 0 : !runner.holds_epoch(10)) std::_Exit(4);
                if (runner.drain_calls != 0) std::_Exit(5);
                if (runner.quarantine_rc != -77 || runner.recovery_calls != 0) std::_Exit(6);
                control().attach_rc = 0;
                Attempt next(runner, 1, 20);
                if (next.prepare() == 0 || runner.launch_calls != 1) std::_Exit(7);
                std::_Exit(0);
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, EscapingDrainExceptionStopsAdmissionAndPreservesResources) {
    for (bool finalize_without_wait : {false, true}) {
        SCOPED_TRACE(finalize_without_wait);
        EXPECT_EXIT(
            {
                Runner runner;
                Attempt active(runner, 0, 10);
                if (active.prepare() != 0 || simpler_launch_run(&runner, active.storage) != 0) std::_Exit(1);
                runner.drain_throws = true;
                runner.quarantine_throws = true;
                const int rc = finalize_without_wait ? simpler_finalize_run(&runner, active.storage) :
                                                       simpler_wait_run(&runner, active.storage);
                if (rc != PTO_RUNTIME_ERR_INTERNAL) std::_Exit(2);
                if (runner.accepts_new_run() || active.consume_release()) std::_Exit(3);
                SimplerWorkspaceReport report{};
                if (!runner.workspace_report(&report)) std::_Exit(3);
                if (finalize_without_wait ? report.quarantined_blocks == 0 : !runner.holds_epoch(10)) std::_Exit(3);
                if (runner.drain_calls != 1 || runner.quarantine_calls != 1) std::_Exit(4);
                Attempt next(runner, 1, 20);
                if (next.prepare() == 0 || runner.launch_calls != 1) std::_Exit(5);
                std::_Exit(0);
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, DiagnosticsFailureAfterProvedCompletionDoesNotQuarantine) {
    Runner runner;
    runner.allocate_callable();
    Attempt active(runner, 0, 10);
    ASSERT_EQ(active.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, active.storage), 0);
    runner.drain_result = DrainOutcome::device_complete(-59);
    EXPECT_EQ(simpler_wait_run(&runner, active.storage), -59);
    EXPECT_TRUE(runner.accepts_new_run());
    EXPECT_EQ(runner.quarantine_calls, 0);
    EXPECT_EQ(simpler_finalize_run(&runner, active.storage), -59);
    EXPECT_TRUE(active.empty());
    EXPECT_TRUE(runner.accepts_new_run());
    EXPECT_EQ(runner.quarantine_calls, 0);
    SimplerWorkspaceReport report{};
    ASSERT_TRUE(runner.workspace_report(&report));
    EXPECT_EQ(report.quarantined_blocks, 0u);
    const size_t bytes = control().bytes;
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), 0);
    EXPECT_FALSE(runner.has_callable(1));
    EXPECT_EQ(control().bytes, bytes - 64);
}

TEST_F(NativeAdmissionTest, UndrainedFinalizationRetainsCallableCodeAndTables) {
    for (int failure = 0; failure < 4; ++failure) {
        SCOPED_TRACE(failure);
        EXPECT_EXIT(
            {
                Runner runner;
                runner.allocate_callable();
                Attempt active(runner, 0, 10);
                if (active.prepare() != 0) std::_Exit(1);
                if (failure == 3) {
                    runner.partial_launch = true;
                    runner.launch_refusal_rc = -4321;
                }
                if (simpler_launch_run(&runner, active.storage) != (failure == 3 ? -4321 : 0)) std::_Exit(1);
                if (failure < 2) control().attach_rc = -77;
                else if (failure == 2) runner.drain_throws = true;
                const int rc = failure < 2 ? -77 : failure == 2 ? PTO_RUNTIME_ERR_INTERNAL : -4321;
                if (failure == 0 && simpler_wait_run(&runner, active.storage) != rc) std::_Exit(2);
                if (simpler_finalize_run(&runner, active.storage) != rc) std::_Exit(3);
                control().attach_rc = 0;
                if (runner.accepts_new_run() || runner.native_runs_outstanding()) std::_Exit(4);
                if (failure == 3 && (runner.recovery_calls != 1 || runner.quarantine_calls != 0)) std::_Exit(4);
                const size_t bytes = control().bytes;
                if (simpler_unregister_callable(&runner, 1) != PTO_RUNTIME_ERR_INVALID_STATE) std::_Exit(5);
                if (runner.unregister_callable(1) != PTO_RUNTIME_ERR_INVALID_STATE) std::_Exit(6);
                if (!runner.has_callable(1) || runner.callable_refcount() != 1 || control().bytes != bytes)
                    std::_Exit(7);
                std::_Exit(0);
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

TEST_F(NativeAdmissionTest, DeviceHealthSuspicionDoesNotBlockProvedCallableRelease) {
    Runner runner;
    runner.allocate_callable();
    Attempt completed(runner, 0, 10);
    ASSERT_EQ(simpler_run(&runner, completed.storage, 1, &completed.args, &completed.config, &completed.descriptor), 0);
    runner.note_device_suspicion();
    ASSERT_FALSE(runner.accepts_new_run());
    ASSERT_TRUE(runner.can_accept_run());
    const size_t bytes = control().bytes;
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), 0);
    EXPECT_FALSE(runner.has_callable(1));
    EXPECT_EQ(control().bytes, bytes - 64);
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
