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
#include "callable.h"
#include "host_build_graph/host/dep_gen_host_graph.h"

extern "C" int args_dump_capture_supported_impl(void);

namespace native_admission_test {
TEST_F(NativeAdmissionTest, RegistrationRefusesInsideBorrowedPreparation) {
    Runner runner;
    ASSERT_EQ(runner.execution_mode_latch().latch(SIMPLER_MODE_KERNEL), 0);
    KernelContextOps ops;
    ops.get_current_device = [](void *, int *id) noexcept {
        *id = 0;
        return 0;
    };
    ops.get_current_context = [](void *, void **ctx) noexcept {
        static int token;
        *ctx = &token;
        return 0;
    };
    ops.create_hidden_stream = ops.get_current_context;
    ops.create_event = ops.get_current_context;
    ops.destroy_hidden_stream = [](void *, void *) noexcept {
        return 0;
    };
    ops.destroy_event = ops.destroy_hidden_stream;
    ASSERT_EQ(runner.kernel_execution_state().initialize(0, ops), 0);
    ASSERT_EQ(runner.kernel_execution_state().mark_ready_enqueued(), 0);
    const uint8_t binary[] = {0x01};
    const auto image = make_chip_callable(nullptr, 0, "orch", binary, sizeof(binary), nullptr, nullptr, 0, "");
    Attempt outer(runner, 1, 20);
    outer.args.tensor(0).transfer = TensorTransfer::NONE;
    outer.args.tensor(0).address_space = AddressSpace::DEVICE;
    callback_hook = [&] {
        ASSERT_TRUE(runner.native_runs_outstanding());
        int32_t id = 99;
        const int rc = simpler_register_callable_v2(&runner, image.data(), image.size(), &id);
        EXPECT_EQ(rc, PTO_RUNTIME_ERR_INTERNAL);
        EXPECT_EQ(id, -1);
    };
    ASSERT_EQ(outer.prepare(), 0);
    EXPECT_EQ(callbacks, 1);
}

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
    EXPECT_TRUE(outer_runner.arena_referenced(1, 0));
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
    void *const idle_heap = runner.arena_region_backing(2, 0);
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
        EXPECT_EQ(runner.arena_region_backing(2, 0), idle_heap);
        EXPECT_EQ(runner.arena_region_backing(2, 2), idle_image);
        if (rc == 0) EXPECT_EQ(simpler_finalize_run(&runner, warm_consumer.storage), 0);
        control().ceiling = control().bytes;
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

TEST_F(NativeAdmissionTest, RealHbgCallbackSurvivesNativeParkAndPublicationFailureExactlyOnce) {
    Runner runner;
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
    control().copy_rc = -88;
    candidate.descriptor.run_epoch = 21;
    EXPECT_EQ(candidate.prepare(), PTO_RUNTIME_ERR_INTERNAL);
    control().copy_rc = 0;
    EXPECT_FALSE(std::filesystem::exists(diagnostic_path + "/deps.json"));
    simpler::dfx::host_graph::HostGraphExport graph;
    EXPECT_EQ(dep_gen_host_graph_take(&graph), static_cast<int>(simpler::dfx::host_graph::TakeOutcome::NotCaptured));
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(runner.retained_temp_addr(1), staging_address);
    EXPECT_TRUE(candidate.consume_release());
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

}  // namespace native_admission_test
