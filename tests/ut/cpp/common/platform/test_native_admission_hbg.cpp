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
#include <cstdio>
#include <cstdlib>
#include "fixture.h"
#include "kernel_platform_ops.h"
#include "host/raii_scope_guard.h"
#include "host_build_graph/host/dep_gen_host_graph.h"

extern "C" int args_dump_capture_supported_impl(void);
extern "C" int publish_run_image_impl(Runtime *, const HostApi *);

#if defined(SIMPLER_TEST_LINKER_WRAP)
extern "C" void set_binding_release_result(void *, int);
extern "C" void set_publication_result(void *, int);
extern "C" int publication_call_count();
extern "C" int binding_release_call_count();
extern "C" int actual_publication_rc();
extern "C" int actual_binding_release_rc();
#endif

namespace native_admission_test {
TEST_F(NativeAdmissionTest, CallbackRejectsAnotherContextBeforeChangingItsRunState) {
    Runner outer_runner;
    Runner inner_runner;
    inner_runner.use_other_entry();
    Attempt outer(outer_runner, 1, 20);
    Attempt inner(inner_runner, 2, 30);
    volatile int32_t accepted = 73;
    inner.descriptor.accepted_state = &accepted;
    inner.descriptor.accepted_value = 1;
    const std::vector<uint8_t> untouched(get_runtime_size(), 0);
    bool entered = false;
    callback_hook = [&] {
        entered = true;
        const auto allocations = control().allocations;
        EXPECT_EQ(inner.prepare(), PTO_RUNTIME_ERR_INVALID_STATE);
        EXPECT_EQ(simpler_prepare_run(nullptr, nullptr, 0, nullptr, nullptr, nullptr), PTO_RUNTIME_ERR_INVALID_STATE);
        EXPECT_EQ(std::memcmp(inner.storage, untouched.data(), untouched.size()), 0);
        EXPECT_EQ(accepted, 73);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(inner_runner.reservations(), 0u);
        EXPECT_EQ(inner_runner.charged(), 0u);
        EXPECT_EQ(inner_runner.plan(2).epoch, 0u);
        EXPECT_EQ(inner_runner.plan(2).phase, AutoAdmissionState::Phase::None);
        EXPECT_EQ(inner_runner.get_prepare_plan(2), nullptr);
        EXPECT_EQ(outer_runner.reservations(), 1u);
        EXPECT_EQ(outer_runner.plan(1).epoch, 20u);
        EXPECT_TRUE(outer_runner.holds_epoch(20));
    };
    ASSERT_EQ(outer.prepare(), 0);
    EXPECT_TRUE(entered);
    EXPECT_EQ(callbacks, 1);
    EXPECT_NE(outer_runner.pooled_heap(1), nullptr);
    EXPECT_TRUE(outer_runner.arena_referenced(1, 2));
    EXPECT_TRUE(outer_runner.holds_epoch(20));
    EXPECT_FALSE(outer_runner.holds_epoch(30));
    ASSERT_EQ(simpler_finalize_run(&outer_runner, outer.storage), 0);
    EXPECT_FALSE(outer_runner.holds_epoch(20));
    // The same request remains usable after the outer prepare has returned.
    ASSERT_EQ(inner.prepare(), 0);
    EXPECT_EQ(callbacks, 2);
    EXPECT_TRUE(inner_runner.holds_epoch(30));
    ASSERT_EQ(simpler_finalize_run(&inner_runner, inner.storage), 0);
    EXPECT_FALSE(inner_runner.holds_epoch(30));
    EXPECT_EQ(accepted, 73);  // neither request launched
}

TEST_F(NativeAdmissionTest, CallbackRejectsSameContextBeforeConstructingAnotherRun) {
    Runner runner;
    Attempt outer(runner, 1, 20);
    Attempt inner(runner, 2, 30);
    const std::vector<uint8_t> untouched(get_runtime_size(), 0);
    bool entered = false;
    callback_hook = [&] {
        entered = true;
        EXPECT_EQ(
            simpler_run(&runner, inner.storage, 1, &inner.args, &inner.config, &inner.descriptor),
            PTO_RUNTIME_ERR_INVALID_STATE
        );
        EXPECT_EQ(std::memcmp(inner.storage, untouched.data(), untouched.size()), 0);
        EXPECT_EQ(runner.reservations(), 1u);
        EXPECT_EQ(runner.plan(2).phase, AutoAdmissionState::Phase::None);
        EXPECT_EQ(runner.plan(1).epoch, 20u);
    };
    ASSERT_EQ(outer.prepare(), 0);
    EXPECT_TRUE(entered);
    EXPECT_EQ(callbacks, 1);
    EXPECT_TRUE(runner.arena_referenced(1, 2));
}

TEST_F(NativeAdmissionTest, CallbackExceptionUnwindsOuterOwnershipAndAllowsTheNextPrepare) {
    Runner outer_runner;
    Runner inner_runner;
    inner_runner.use_other_entry();
    Attempt outer(outer_runner, 1, 20);
    Attempt inner(inner_runner, 2, 30);
    callback_hook = [&] {
        EXPECT_EQ(inner.prepare(), PTO_RUNTIME_ERR_INVALID_STATE);
        throw std::runtime_error("host callback failed after nested prepare refusal");
    };
    EXPECT_EQ(outer.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_TRUE(outer.consume_release());
    EXPECT_TRUE(outer.empty());
    EXPECT_TRUE(inner.empty());
    EXPECT_EQ(outer_runner.reservations(), 0u);
    EXPECT_EQ(inner_runner.reservations(), 0u);
    EXPECT_FALSE(outer_runner.holds_epoch(20));
    EXPECT_FALSE(outer_runner.plan(1).holds_token);
    EXPECT_EQ(outer_runner.get_prepare_plan(1), nullptr);
    ASSERT_EQ(inner.prepare(), 0);
    ASSERT_EQ(simpler_finalize_run(&inner_runner, inner.storage), 0);
    outer.descriptor.run_epoch = 21;
    ASSERT_EQ(outer.prepare(), 0);
    EXPECT_TRUE(outer_runner.holds_epoch(21));
    EXPECT_FALSE(outer_runner.holds_epoch(20));
}

TEST_F(NativeAdmissionTest, ReservedPrepareExcludesAWarmConsumerBeforeTrimmingItsIdleBank) {
    Runner runner;
    Attempt warmup(runner, 2, 10);
    ASSERT_EQ(warmup.prepare(), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, warmup.storage), 0);
    void *const idle_heap = runner.pooled_heap(2);
    void *const idle_image = runner.arena_region_backing(2, 2);
    ASSERT_NE(idle_heap, nullptr);
    ASSERT_NE(idle_image, nullptr);
    Attempt outer(runner, 1, 20);
    Attempt warm_consumer(runner, 2, 30);
    bool entered = false;
    callback_hook = [&] {
        entered = true;
        const auto allocations = control().allocations;
        const auto charged = runner.charged();
        auto contender = std::async(std::launch::async, [&] {
            return warm_consumer.prepare();
        });
        const int rc = contender.get();
        // A different thread passes the reentry guard; the unclaimed native
        // reservation must still exclude even a non-growing bank consumer.
        EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_TRUE(warm_consumer.empty());
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(runner.charged(), charged);
        EXPECT_EQ(runner.reservations(), 1u);
        EXPECT_EQ(runner.pooled_heap(2), idle_heap);
        EXPECT_EQ(runner.arena_region_backing(2, 2), idle_image);
        if (rc == 0) EXPECT_EQ(simpler_finalize_run(&runner, warm_consumer.storage), 0);
        // Scheduler storage is per pipeline slot and survives idle arena trim.
        // Reserve one additional aligned backing while keeping arena growth capped.
        const auto scheduler_bytes = runner.scheduler_capacity(2);
        control().ceiling = control().bytes + (scheduler_bytes == 0 ? 0 : scheduler_bytes + 127);
    };
    ASSERT_EQ(outer.prepare(), 0);
    EXPECT_TRUE(entered);
    EXPECT_EQ(runner.arena_region_backing(2, 0), nullptr);
    EXPECT_EQ(runner.arena_region_backing(2, 2), nullptr);
    EXPECT_EQ(runner.retained_temp_addr(2), 0u);
    EXPECT_TRUE(runner.arena_referenced(1, 2));
    ASSERT_EQ(simpler_finalize_run(&runner, outer.storage), 0);
    control().ceiling = std::numeric_limits<size_t>::max();
    const auto allocations = control().allocations;
    ASSERT_EQ(warm_consumer.prepare(), 0);
    EXPECT_GT(control().allocations, allocations);
    EXPECT_TRUE(runner.arena_referenced(2, 2));
    EXPECT_TRUE(runner.holds_epoch(30));
}

TEST_F(NativeAdmissionTest, AnotherThreadCanPrepareADifferentContextWhileTheCallbackWaits) {
    Runner outer_runner;
    Runner other_runner;
    other_runner.use_other_entry();
    Attempt outer(outer_runner, 1, 20);
    bool entered = false;
    callback_hook = [&] {
        entered = true;
        auto other = std::async(std::launch::async, [&] {
            Attempt attempt(other_runner, 2, 30);
            const int rc = attempt.prepare();
            if (rc == 0) {
                EXPECT_TRUE(other_runner.holds_epoch(30));
                EXPECT_EQ(simpler_finalize_run(&other_runner, attempt.storage), 0);
            }
            return rc;
        });
        EXPECT_EQ(other.get(), 0);
        EXPECT_EQ(outer_runner.plan(1).epoch, 20u);
    };
    ASSERT_EQ(outer.prepare(), 0);
    EXPECT_TRUE(entered);
    EXPECT_EQ(callbacks, 2);
    EXPECT_TRUE(outer_runner.arena_referenced(1, 2));
    EXPECT_FALSE(other_runner.holds_epoch(30));
}

TEST_F(NativeAdmissionTest, RealHbgCallbackAndOwnedInputsSurviveNativeParkExactlyOnce) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    refuse_after_callback = true;
    Attempt candidate(runner, 1, 20);
    diagnostics(candidate);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    ASSERT_EQ(callbacks, 1);
    ASSERT_TRUE(candidate.empty());
    const auto records_before = runner.phases(1).total_records();
    const auto dropped_before = runner.phases(1).dropped_records();
    ASSERT_TRUE(runner.phases(1).finished());
    EXPECT_GT(records_before, 0u);
    EXPECT_FALSE(std::filesystem::exists(diagnostic_path + "/deps.json"));
    std::weak_ptr<void> saved = runner.get_prepare_plan(1);
    ASSERT_FALSE(saved.expired());
    const auto address = runner.retained_temp_addr(1);
    ASSERT_NE(address, 0u);
    const auto *owned = reinterpret_cast<const uint8_t *>(RetainedTempBump::align_up(address));
    const std::vector<uint8_t> original(owned, owned + candidate.bytes.size());
    EXPECT_EQ(original, candidate.bytes);
    std::fill(candidate.bytes.begin(), candidate.bytes.end(), 0x42);
    runner.retire(0, 10);
    control().ceiling = std::numeric_limits<size_t>::max();
    candidate.descriptor.run_epoch = 21;
    auto resumed = std::async(std::launch::async, [&] {
        return candidate.prepare();
    });
    ASSERT_EQ(resumed.get(), 0);
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(runner.retained_temp_addr(1), address);
    EXPECT_EQ(std::vector<uint8_t>(owned, owned + original.size()), original);
    EXPECT_TRUE(saved.expired());
    EXPECT_GE(runner.phases(1).total_records(), records_before);
    EXPECT_GE(runner.phases(1).dropped_records(), dropped_before);
    EXPECT_TRUE(runner.phases(1).finished());
    EXPECT_EQ(
        runner.phases(1).invocation_id(),
        static_cast<NativeRunContext<DeviceRunnerBase> *>(candidate.storage)->trace_inv
    );
    EXPECT_TRUE(std::filesystem::exists(diagnostic_path + "/deps.json"));
    ASSERT_EQ(simpler_finalize_run(&runner, candidate.storage), 0);
    EXPECT_TRUE(candidate.empty());
}

TEST_F(NativeAdmissionTest, RealHbgCallbackSurvivesNativeParkAndLaunchPublicationFailureExactlyOnce) {
    auto owned_runner = make_fenced_completion_runner();
    Runner &runner = *owned_runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    refuse_after_callback = true;
    Attempt candidate(runner, 1, 20);
    diagnostics(candidate);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    EXPECT_EQ(callbacks, 1);
    EXPECT_TRUE(candidate.empty());
    ASSERT_NE(runner.get_prepare_plan(1), nullptr);
    std::weak_ptr<void> saved = runner.get_prepare_plan(1);
    const auto staging_address = runner.retained_temp_addr(1);
    EXPECT_TRUE(runner.plan(1).plan_stamped);
    EXPECT_EQ(runner.reservations(), 0u);
    runner.retire(0, 10);
    control().ceiling = std::numeric_limits<size_t>::max();
    candidate.descriptor.run_epoch = 21;
    ASSERT_EQ(candidate.prepare(), 0);
    control().copy_rc = -88;
    EXPECT_EQ(simpler_launch_run(&runner, candidate.storage), -88);
    control().copy_rc = 0;
    EXPECT_EQ(simpler_finalize_run(&runner, candidate.storage), -88);
    EXPECT_TRUE(std::filesystem::exists(diagnostic_path + "/deps.json"));
    simpler::dfx::host_graph::HostGraphExport graph;
    EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(simpler::dfx::host_graph::TakeOutcome::NotCaptured));
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(runner.retained_temp_addr(1), staging_address);
    EXPECT_TRUE(candidate.empty());
    EXPECT_EQ(runner.get_prepare_plan(1), nullptr);
    EXPECT_TRUE(saved.expired());
    EXPECT_FALSE(runner.plan(1).holds_token);
    EXPECT_EQ(runner.reservations(), 0u);
}

TEST_F(NativeAdmissionTest, CloseOwnsTheSavedGraphAfterItsCallback) {
    Runner runner;
    ASSERT_EQ(staging(runner, 0, 10, 20 * kUnit), 0);
    launched(runner, 0, 10);
    refuse_after_callback = true;
    Attempt candidate(runner, 1, 20);
    ASSERT_EQ(candidate.prepare(), SIMPLER_RUNTIME_ERR_RESOURCE_WAIT);
    std::weak_ptr<void> owner = runner.get_prepare_plan(1);
    ASSERT_FALSE(owner.expired());
    EXPECT_EQ(callbacks, 1);
    runner.retire(0, 10);
    ASSERT_EQ(finalize_device(&runner), 0);
    EXPECT_TRUE(owner.expired());
    EXPECT_EQ(runner.charged(), 0u);
}

TEST_F(NativeAdmissionTest, UnsupportedArgsDumpReleasesExactAttemptBeforeValidRunFlushClose) {
    if (args_dump_capture_supported_impl() != 0) GTEST_SKIP() << "This runtime supports ArgsDump";
    Runner runner;
    Attempt rejected(runner, 0, 10);
    rejected.config.enable_dump_args = 1;
    const auto allocations = control().allocations;
    ASSERT_EQ(rejected.prepare(), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(callbacks, 0);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_EQ(runner.reservations(), 0u);
    EXPECT_FALSE(runner.holds_epoch(10));
    EXPECT_EQ(runner.plan(0).phase, AutoAdmissionState::Phase::None);
    EXPECT_FALSE(consume_released_preparation(rejected.storage, get_runtime_size(), 11));
    ASSERT_TRUE(rejected.consume_release());
    EXPECT_FALSE(rejected.consume_release());
    EXPECT_TRUE(rejected.empty());
    Attempt next(runner, 0, 20);
    ASSERT_EQ(simpler_run(&runner, next.storage, 1, &next.args, &next.config, &next.descriptor), 0);
    EXPECT_EQ(callbacks, 1);
    EXPECT_TRUE(next.empty());
    EXPECT_FALSE(runner.holds_epoch(20));
    char error[256]{};
    EXPECT_EQ(simpler_flush_diagnostics_ctx(&runner, 1000, error, sizeof(error)), 0) << error;
    EXPECT_EQ(finalize_device(&runner), 0);
}

TEST_F(NativeAdmissionTest, UnsupportedArgsDumpCannotReplaceLiveCarrierOrItsEpoch) {
    if (args_dump_capture_supported_impl() != 0) GTEST_SKIP() << "This runtime supports ArgsDump";
    Runner runner;
    Attempt live(runner, 0, 10);
    ASSERT_EQ(live.prepare(), 0);
    std::vector<unsigned char> before(get_runtime_size());
    std::memcpy(before.data(), live.storage, before.size());
    live.config.enable_dump_args = 1;
    live.descriptor.run_epoch = 20;
    EXPECT_EQ(live.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(std::memcmp(before.data(), live.storage, before.size()), 0);
    EXPECT_FALSE(live.consume_release());
    EXPECT_TRUE(runner.holds_epoch(10));
    EXPECT_FALSE(runner.holds_epoch(20));
    EXPECT_EQ(runner.reservations(), 1u);
    EXPECT_EQ(simpler_finalize_run(&runner, live.storage), 0);
}

TEST_F(NativeAdmissionTest, RecursiveUnsupportedArgsDumpDoesNotAcquireOrCertifyAnAttempt) {
    if (args_dump_capture_supported_impl() != 0) GTEST_SKIP() << "This runtime supports ArgsDump";
    Runner runner;
    Attempt outer(runner, 0, 10);
    Attempt inner(runner, 1, 20);
    inner.config.enable_dump_args = 1;
    bool entered = false;
    callback_hook = [&] {
        entered = true;
        EXPECT_EQ(inner.prepare(), PTO_RUNTIME_ERR_INVALID_STATE);
        EXPECT_TRUE(inner.empty());
        EXPECT_FALSE(inner.consume_release());
        EXPECT_EQ(runner.reservations(), 1u);
        EXPECT_FALSE(runner.holds_epoch(20));
    };
    ASSERT_EQ(outer.prepare(), 0);
    EXPECT_TRUE(entered);
    EXPECT_EQ(simpler_finalize_run(&runner, outer.storage), 0);
}

namespace {
class HbgImageCaptureRunner : public Runner {
public:
    HbgImageCaptureRunner() {
        static int binary;
        aicore_bin_handle_ = &binary;
    }
    int initialize_kernel() {
        int rc = execution_mode_latch().latch(SIMPLER_MODE_KERNEL);
        if (rc == 0) rc = kernel_execution_state().initialize(0, make_onboard_kernel_context_ops());
        return rc == 0 ? kernel_execution_state().mark_ready_enqueued() : rc;
    }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        int rc = Runner::prepare_execution(runtime, config, slot, identity, out);
        if (rc == 0) rc = ensure_device_run_result_region(identity, (*out)->kernel_args, (*out)->dfx);
        return rc == 0 ? (*out)->kernel_args.prepare_runtime_args(runtime, mem_alloc_, slot_persistent_args(slot)) : rc;
    }
    LaunchTransactionResult launch_execution_impl(PreparedExecution &prepared, LaunchPermit permit) override {
        prepared.streams = borrowed_run_streams();
        prepared.streams.aicpu = prepared.caller_stream;
        const int rc = arm_run_fence(prepared);
        if (rc != 0) return {rc, LaunchProgress::NotStarted, {}};
        return exact_launch_transaction(
            prepared.identity, std::move(permit),
            [&](LaunchProgressSink &sink) {
                const int entry_rc = queue_run_entry(prepared, sink);
                if (entry_rc != 0) return entry_rc;
                copies_before_entry = control().deferred_copies.size();
                sink.mark_submitted();
                return record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicore);
            },
            [&](LaunchProgressSink &sink) {
                sink.mark_submitted();
                const int boundary_rc = record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicpu);
                return boundary_rc != 0 ? boundary_rc : queue_caller_exit(prepared);
            }
        );
    }
    int prepare_upload(Runtime &runtime, KernelArgsHelper &helper) {
        return helper.prepare_runtime_args(runtime, mem_alloc_, slot_persistent_args(0));
    }
    size_t copies_before_entry{0};
};
}  // namespace

TEST_F(NativeAdmissionTest, CapturedHbgImageUsesPinnedLaunchSourceThroughOriginalRunRetirement) {
    HbgImageCaptureRunner runner;
    ASSERT_EQ(runner.initialize_kernel(), 0);
    Attempt attempt(runner, 0, 9990);
    attempt.descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
    attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
    attempt.args.tensor(0).transfer = TensorTransfer::NONE;
    auto reset_recording = RAIIScopeGuard([] {
        control().recording_active = false;
    });
    ASSERT_EQ(attempt.prepare(), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    auto publication = state->runtime->prepared_publication();
    ASSERT_NE(publication, nullptr);
    ASSERT_GT(publication->bytes, 0u);
    const auto synchronous_images = std::count_if(control().uploads.begin(), control().uploads.end(), [&](auto &copy) {
        return copy.first == publication->device_target;
    });
    EXPECT_EQ(synchronous_images, 0) << "the image must be staged without a synchronous device copy";
    const auto *source = static_cast<const uint8_t *>(publication->source);
    const std::vector<uint8_t> expected(source, source + publication->bytes);
    std::memset(const_cast<void *>(publication->source), 0xe7, publication->bytes);
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    control().recording_active = true;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_finalize_run(&runner, attempt.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    const void *descriptor_target = state->prepared_execution->kernel_args.args.runtime_args;
    const std::weak_ptr<const PreparedRunResources> backing = state->prepared_execution->prepared_resources;
    control().defer_async_copies = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    auto image = std::find_if(control().deferred_copies.begin(), control().deferred_copies.end(), [&](auto &copy) {
        return copy.destination == publication->device_target;
    });
    ASSERT_NE(image, control().deferred_copies.end());
    EXPECT_NE(image->source, publication->source);
    EXPECT_EQ(std::memcmp(image->source, expected.data(), expected.size()), 0);
    auto descriptor = std::find_if(control().deferred_copies.begin(), control().deferred_copies.end(), [&](auto &copy) {
        return copy.destination == descriptor_target;
    });
    ASSERT_NE(descriptor, control().deferred_copies.end());
    EXPECT_LT(image, descriptor);
    EXPECT_EQ(runner.copies_before_entry, control().deferred_copies.size());
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(backing.expired());
    EXPECT_EQ(std::memcmp(image->source, expected.data(), expected.size()), 0);
    for (const auto &copy : control().deferred_copies)
        std::memcpy(copy.destination, copy.source, copy.bytes);
    EXPECT_EQ(std::memcmp(publication->device_target, expected.data(), expected.size()), 0);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_TRUE(backing.expired());
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, HbgPrerequisitesAndImagePublishBeforeTheDescriptorFromOwnedSnapshots) {
    HbgImageCaptureRunner runner;
    auto runtime = std::make_unique<Runtime>();
    std::array<uint8_t, 4> first_source{1, 2, 3, 4}, second_source{5, 6, 7, 8}, image_source{9, 10, 11, 12};
    std::array<uint8_t, 4> first_target{}, second_target{}, image_target{};
    const auto expected_first = first_source, expected_second = second_source, expected_image = image_source;
    runtime->add_pending_metadata(
        first_target.data(), first_source.data(), first_source.size(), HostPhaseKind::BindArenaH2d, {}
    );
    runtime->add_pending_metadata(
        second_target.data(), second_source.data(), second_source.size(), HostPhaseKind::BindArenaH2d, {}
    );
    runtime->set_pending_publication(image_target.data(), image_source.data(), image_source.size(), 0, 0, 0);
    HostApiOps ops{};
    ops.stage_launch_metadata = [](void *context, uint32_t slot, const HostMetadataRegion *regions, size_t count) {
        return static_cast<HbgImageCaptureRunner *>(context)->stage_launch_metadata(slot, regions, count);
    };
    ops.copy_to_device = [](void *, void *, const void *, size_t) {
        ADD_FAILURE() << "unexpected synchronous image copy";
        return -4291;
    };
    HostApi api(&runner, 0, 0, 0, &ops);
    ASSERT_EQ(publish_run_image_impl(runtime.get(), &api), 0);
    first_source.fill(0xe1);
    second_source.fill(0xe2);
    image_source.fill(0xe3);
    KernelArgsHelper helper;
    ASSERT_EQ(runner.prepare_upload(*runtime, helper), 0);
    int caller;
    LaunchProgressSink sink;
    control().defer_async_copies = true;
    ASSERT_EQ(publish_for_launch(helper, &caller, sink, true), 0);
    ASSERT_EQ(control().deferred_copies.size(), 4u);
    EXPECT_EQ(control().deferred_copies[0].destination, first_target.data());
    EXPECT_EQ(control().deferred_copies[1].destination, second_target.data());
    EXPECT_EQ(control().deferred_copies[2].destination, image_target.data());
    EXPECT_EQ(control().deferred_copies[3].destination, helper.args.runtime_args);
    EXPECT_EQ(first_target, (std::array<uint8_t, 4>{}));
    EXPECT_EQ(second_target, (std::array<uint8_t, 4>{}));
    EXPECT_EQ(image_target, (std::array<uint8_t, 4>{}));
    for (const auto &copy : control().deferred_copies)
        std::memcpy(copy.destination, copy.source, copy.bytes);
    EXPECT_EQ(first_target, expected_first);
    EXPECT_EQ(second_target, expected_second);
    EXPECT_EQ(image_target, expected_image);
}

TEST_F(NativeAdmissionTest, SynchronousHbgBackendKeepsPrerequisiteAndImagePublicationOrder) {
    auto runtime = std::make_unique<Runtime>();
    uint32_t first = 1, second = 2, image = 3;
    uint32_t first_target = 0, second_target = 0, image_target = 0;
    runtime->add_pending_metadata(&first_target, &first, sizeof(first), HostPhaseKind::BindArenaH2d, {});
    runtime->add_pending_metadata(&second_target, &second, sizeof(second), HostPhaseKind::BindArenaH2d, {});
    runtime->set_pending_publication(&image_target, &image, sizeof(image), 0, 0, 0);
    HostApiOps ops{};
    ops.copy_to_device = [](void *, void *target, const void *source, size_t bytes) {
        return rtMemcpy(target, bytes, source, bytes, RT_MEMCPY_HOST_TO_DEVICE);
    };
    HostApi api(nullptr, 0, 0, 0, &ops);
    ASSERT_EQ(publish_run_image_impl(runtime.get(), &api), 0);
    ASSERT_EQ(control().uploads.size(), 3u);
    EXPECT_EQ(control().uploads[0].first, &first_target);
    EXPECT_EQ(control().uploads[1].first, &second_target);
    EXPECT_EQ(control().uploads[2].first, &image_target);
    EXPECT_EQ(first_target, first);
    EXPECT_EQ(second_target, second);
    EXPECT_EQ(image_target, image);
}

TEST_F(NativeAdmissionTest, FailedCapturedHbgImageUploadPreservesBackingThroughCheckedRelease) {
    HbgImageCaptureRunner runner;
    ASSERT_EQ(runner.initialize_kernel(), 0);
    Attempt attempt(runner, 0, 9991);
    attempt.descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
    attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
    attempt.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(attempt.prepare(), 0);
    auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
    const auto publication = state->runtime->prepared_publication();
    ASSERT_NE(publication, nullptr);
    const std::weak_ptr<const PreparedRunResources> backing = state->prepared_execution->prepared_resources;
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        control().stream_sync_rc = 0;
        (void)simpler_finalize_run(&runner, attempt.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    control().recording_active = true;
    control().defer_async_copies = true;
    control().failed_async_destination = publication->device_target;
    control().async_copy_rc = -4288;
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), -4288);
    EXPECT_EQ(runner.copies_before_entry, 0u);
    ASSERT_TRUE(std::any_of(control().deferred_copies.begin(), control().deferred_copies.end(), [&](auto &copy) {
        return copy.destination == publication->device_target;
    }));
    control().recording_active = false;
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), -4288);
    EXPECT_FALSE(backing.expired());
    control().stream_sync_rc = -4290;
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), -4290);
    EXPECT_FALSE(backing.expired());
    control().stream_sync_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_TRUE(backing.expired());
    cleanup.dismiss();
}

#if defined(SIMPLER_TEST_LINKER_WRAP)
TEST_F(NativeAdmissionTest, FailedHbgPublicationPreservesTheUnprovenBindingReleaseVerdict) {
    ASSERT_EXIT(
        {
            auto check_failed_prepare = [] {
                Runner runner;
                auto reset_faults = RAIIScopeGuard([] {
                    control().copy_rc = 0;
                    set_publication_result(nullptr, 0);
                    set_binding_release_result(nullptr, 0);
                });
                Attempt attempt(runner, 0, 9992);
                attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
                attempt.args.tensor(0).transfer = TensorTransfer::NONE;
                ASSERT_EQ(attempt.prepare(), 0);
                ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
                ++attempt.descriptor.run_epoch;
                const size_t uploads = control().uploads.size();
                control().copy_rc = -4321;
                ASSERT_EQ(attempt.prepare(), 0);
                EXPECT_EQ(control().uploads.size(), uploads);
                control().copy_rc = 0;
                ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
                ++attempt.descriptor.run_epoch;
                set_binding_release_result(nullptr, -4324);
                set_publication_result(nullptr, -4321);
                EXPECT_EQ(attempt.prepare(), -4324);
                EXPECT_EQ(publication_call_count(), 1);
                EXPECT_EQ(actual_publication_rc(), 0);
                EXPECT_EQ(binding_release_call_count(), 1);
                EXPECT_EQ(actual_binding_release_rc(), 0);
                const std::vector<uint8_t> empty_receipt(8, 0);
                EXPECT_EQ(std::memcmp(attempt.storage, empty_receipt.data(), empty_receipt.size()), 0);
                EXPECT_FALSE(runner.native_run_active());
            };
            check_failed_prepare();
            EXPECT_GT(control().bytes, 0u);
            std::fflush(stdout);
            std::fflush(stderr);
            std::_Exit(::testing::Test::HasFailure() ? 1 : 0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}
#endif

}  // namespace native_admission_test
