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
#include <acl/acl.h>
#include "callable.h"
#include "host_build_graph/host/dep_gen_host_graph.h"

extern "C" int args_dump_capture_supported_impl(void);

namespace native_admission_test {
TEST_F(NativeAdmissionTest, CallerCaptureRefusalPrecedesBindAndPreservesRetry) {
    Runner runner;
    int stream;
    Attempt attempt(runner, 1, 20);
    attempt.descriptor.caller_stream = &stream;
    volatile int32_t accepted = 73;
    attempt.descriptor.accepted_state = &accepted;
    attempt.descriptor.accepted_value = 1;
    for (int status : {ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE, ACL_MODEL_RI_CAPTURE_STATUS_INVALIDATED}) {
        control().capture_status = status;
        const auto allocations = control().allocations;
        EXPECT_EQ(attempt.prepare(), PTO_RUNTIME_ERR_UNSUPPORTED);
        EXPECT_EQ(callbacks, 0);
        EXPECT_EQ(control().allocations, allocations);
        EXPECT_EQ(runner.reservations(), 0u);
        EXPECT_FALSE(runner.holds_epoch(20));
        EXPECT_EQ(accepted, 73);
        EXPECT_TRUE(attempt.consume_release());
    }
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    ASSERT_EQ(attempt.prepare(), 0);
    EXPECT_EQ(callbacks, 1);
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
}

TEST_F(NativeAdmissionTest, CallerQueryFailurePreservesPreparationRetry) {
    Runner runner;
    int stream;
    Attempt attempt(runner, 1, 20);
    attempt.descriptor.caller_stream = &stream;
    control().capture_query_rc = -4321;
    const auto allocations = control().allocations;
    EXPECT_EQ(attempt.prepare(), -4321);
    EXPECT_EQ(callbacks, 0);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_TRUE(attempt.consume_release());
    control().capture_query_rc = 0;
    ASSERT_EQ(attempt.prepare(), 0);
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
}

TEST_F(NativeAdmissionTest, CaptureStartedAfterPrepareRefusesPlainLaunchWithoutConsumingRun) {
    Runner runner;
    int stream;
    Attempt attempt(runner, 1, 20);
    attempt.descriptor.caller_stream = &stream;
    ASSERT_EQ(attempt.prepare(), 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    EXPECT_EQ(simpler_launch_run(&runner, attempt.storage), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(runner.launch_calls, 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    ASSERT_EQ(simpler_launch_run(&runner, attempt.storage), 0);
    EXPECT_EQ(runner.launch_calls, 1);
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
}

TEST_F(NativeAdmissionTest, PreparedCallerCannotBeReboundBeforeLaunch) {
    Runner runner;
    int stream, other;
    Attempt attempt(runner, 1, 20);
    attempt.descriptor.caller_stream = &stream;
    ASSERT_EQ(attempt.prepare(), 0);
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &other, nullptr), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(runner.launch_calls, 0);
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &stream, nullptr), 0);
    EXPECT_EQ(runner.launch_calls, 1);
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
}

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

namespace {
class CapturedWorkspaceRunner : public Runner {
public:
    int capture_arena_request(uint32_t slot, uint32_t bank_id, uint64_t epoch, size_t extra) {
        WorkspacePlanScope scope(slot, epoch);
        auto &bank = arena_bank(bank_id);
        slot_persistent_args_[slot].capture_preparing = true;
        const int rc = setup_static_arena(
            bank_id, bank.cached_gm_heap_size + extra, bank.cached_gm_sm_size, bank.cached_runtime_arena_size
        );
        slot_persistent_args_[slot].capture_preparing = false;
        return rc;
    }
    int capture_temp_request(uint32_t slot, uint64_t epoch, size_t extra) {
        WorkspacePlanScope scope(slot, epoch);
        const auto old = retained_temp_addrs_[slot];
        const auto capacity = retained_temp_sizes_[slot];
        slot_persistent_args_[slot].capture_preparing = true;
        void *out = nullptr;
        size_t size = 0;
        const int rc = acquire_retained_temp(slot, capacity + extra, &out, &size);
        slot_persistent_args_[slot].capture_preparing = false;
        EXPECT_EQ(retained_temp_addrs_[slot], old);
        EXPECT_EQ(retained_temp_sizes_[slot], capacity);
        return rc;
    }
    void (*notification)(void *){nullptr};
    void *notification_data{nullptr};
    int callback_rc{0};
    int callback_calls{0};
    void *captured_backing{nullptr};

    int retain_prepared_workspace(void *model, const NativeRunDescriptor &descriptor) {
        std::vector<void *> blocks;
        for (size_t region = 0; region < 3; ++region) {
            void *base = arena_region_backing(descriptor.arena_bank, region);
            if (base != nullptr) blocks.push_back(base);
        }
        if (!blocks.empty()) captured_backing = blocks.back();
        if (retained_temp_addrs_[descriptor.pipeline_slot] != nullptr)
            blocks.push_back(retained_temp_addrs_[descriptor.pipeline_slot]);
        return captured_graph_resources_.retain(
            model, {},
            [&](void *, void (*notify)(void *), void *data) {
                ++callback_calls;
                if (callback_rc == 0) {
                    notification = notify;
                    notification_data = data;
                }
                return callback_rc;
            },
            &workspace_, blocks
        );
    }
    int retire_capture_after_drain() { return captured_graph_resources_.retire_after_drain(); }
    bool cap_at_current_workspace() { return workspace_.set_limit(workspace_.reserved_bytes()); }
    bool probe_cached_arena() const {
        const uint64_t key = 1;
        void *heap = nullptr, *sm = nullptr, *arena = nullptr;
        size_t offset = 0, bytes = 0;
        const void *image = nullptr;
        return lookup_prebuilt_runtime_arena_cache(
            0, 1, &key, sizeof(key), &heap, &sm, &arena, &offset, &image, &bytes
        );
    }
    void seed_cached_arena() {
        const uint64_t key = 1;
        mark_prebuilt_runtime_arena_cached(
            0, 1, &key, sizeof(key), nullptr, nullptr, arena_region_backing(0, 2), 0, &key, sizeof(key)
        );
    }
    bool captured_block_referenced() const {
        return workspace_.block_state(captured_backing) == WorkspaceManager::BlockState::Referenced;
    }
    std::vector<unsigned char> captured_contents() const {
        auto *base = static_cast<unsigned char *>(captured_backing);
        return {base, base + workspace_.block_bytes(base)};
    }
};
}  // namespace

TEST_F(NativeAdmissionTest, CaptureWorkspaceRefusesGrowthBeforeAllocating) {
    CapturedWorkspaceRunner runner;
    Attempt attempt(runner, 1, 20);
    ASSERT_EQ(attempt.prepare(), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    const auto allocations = control().allocations;
    const auto charged = runner.charged();
    EXPECT_EQ(runner.capture_arena_request(1, attempt.descriptor.arena_bank, 21, 4096), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(runner.capture_temp_request(1, 21, 4096), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_EQ(runner.charged(), charged);
}

TEST_F(NativeAdmissionTest, CaptureWorkspaceRefusesReplacementOfGraphHeldBacking) {
    CapturedWorkspaceRunner runner;
    Attempt attempt(runner, 1, 20);
    ASSERT_EQ(attempt.prepare(), 0);
    int model;
    ASSERT_EQ(runner.retain_prepared_workspace(&model, attempt.descriptor), 0);
    const auto contents = runner.captured_contents();
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    const auto allocations = control().allocations;
    EXPECT_EQ(runner.capture_arena_request(1, attempt.descriptor.arena_bank, 21, 0), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_EQ(runner.captured_contents(), contents);
    runner.notification(runner.notification_data);
    EXPECT_EQ(runner.retire_capture_after_drain(), 0);
}

TEST_F(NativeAdmissionTest, CapturedArenaReferenceSurvivesNativeFinalizeUntilModelAndReadersRetire) {
    CapturedWorkspaceRunner runner;
    Attempt attempt(runner, 1, 20);
    ASSERT_EQ(attempt.prepare(), 0);
    int model;
    ASSERT_EQ(runner.retain_prepared_workspace(&model, attempt.descriptor), 0);
    ASSERT_NE(runner.captured_backing, nullptr);
    ASSERT_EQ(runner.retain_prepared_workspace(&model, attempt.descriptor), 0);
    EXPECT_EQ(runner.callback_calls, 1);
    // Exercise the real native finalizer with an executed run. This test pins
    // its workspace only; it does not claim to record a native execution graph.
    ASSERT_EQ(simpler_launch_run(&runner, attempt.storage), 0);
    ASSERT_EQ(simpler_wait_run(&runner, attempt.storage), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_TRUE(attempt.empty());
    EXPECT_FALSE(runner.holds_epoch(20));
    EXPECT_TRUE(runner.captured_block_referenced());
    EXPECT_TRUE(runner.workspace_capture_references_live());
    const auto contents = runner.captured_contents();
    {
        // The existing grant supplies another ordinary slot/bank. Capturing a
        // reference does not create a new pool or globally forbid eager work.
        Attempt independent(runner, 2, 21);
        ASSERT_EQ(independent.prepare(), 0);
        ASSERT_EQ(simpler_launch_run(&runner, independent.storage), 0);
        ASSERT_EQ(simpler_wait_run(&runner, independent.storage), 0);
        ASSERT_EQ(simpler_finalize_run(&runner, independent.storage), 0);
        EXPECT_EQ(runner.captured_contents(), contents);
    }
    {
        Attempt replacement(runner, 1, 22);
        ASSERT_EQ(replacement.prepare(), 0);
        EXPECT_EQ(runner.captured_contents(), contents);
        ASSERT_EQ(simpler_launch_run(&runner, replacement.storage), 0);
        ASSERT_EQ(simpler_wait_run(&runner, replacement.storage), 0);
        ASSERT_EQ(simpler_finalize_run(&runner, replacement.storage), 0);
        EXPECT_EQ(runner.captured_contents(), contents);
    }
    const auto bytes = control().bytes;
    EXPECT_EQ(runner.finalize(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(control().bytes, bytes);
    EXPECT_EQ(runner.retire_capture_after_drain(), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_NE(runner.notification, nullptr);
    runner.notification(runner.notification_data);
    EXPECT_TRUE(runner.workspace_capture_references_live());
    EXPECT_EQ(runner.finalize(), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(runner.retire_capture_after_drain(), 0);
    EXPECT_FALSE(runner.workspace_capture_references_live());
    EXPECT_FALSE(runner.captured_block_referenced());
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, RefusedModelRegistrationRollsBackOnlyItsWorkspaceReferences) {
    CapturedWorkspaceRunner runner;
    Attempt attempt(runner, 1, 20);
    ASSERT_EQ(attempt.prepare(), 0);
    int model;
    runner.callback_rc = -4331;
    EXPECT_EQ(runner.retain_prepared_workspace(&model, attempt.descriptor), -4331);
    EXPECT_FALSE(runner.workspace_capture_references_live());
    EXPECT_TRUE(runner.holds_epoch(20));
    EXPECT_TRUE(runner.captured_block_referenced());
    EXPECT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(runner.holds_epoch(20));
    EXPECT_EQ(runner.finalize(), 0);
}

TEST_F(NativeAdmissionTest, CapturedWorkspaceQuotaRefusalPreservesBackingAndRetriesAfterRetirement) {
    CapturedWorkspaceRunner runner;
    Attempt original(runner, 0, 30);
    ASSERT_EQ(original.prepare(), 0);
    runner.seed_cached_arena();
    ASSERT_TRUE(runner.probe_cached_arena());
    int model;
    ASSERT_EQ(runner.retain_prepared_workspace(&model, original.descriptor), 0);
    EXPECT_EQ(simpler_finalize_run(&runner, original.storage), 0);
    EXPECT_FALSE(runner.probe_cached_arena());
    const auto contents = runner.captured_contents();
    ASSERT_TRUE(runner.cap_at_current_workspace());
    Attempt next(runner, 0, 31);
    EXPECT_NE(next.prepare(), 0);
    EXPECT_TRUE(next.consume_release());
    EXPECT_EQ(runner.captured_contents(), contents);
    EXPECT_TRUE(runner.workspace_capture_references_live());
    runner.notification(runner.notification_data);
    ASSERT_EQ(runner.retire_capture_after_drain(), 0);
    EXPECT_TRUE(runner.probe_cached_arena());
    ASSERT_EQ(next.prepare(), 0);
    EXPECT_EQ(simpler_finalize_run(&runner, next.storage), 0);
}

namespace {
class NativeGraphResourceRunner : public Runner {
public:
    using DeviceRunnerBase::finalize_common;
    bool fail_descriptor_once{false};
    int adopt_kernel_context() {
        int rc = execution_mode_latch().latch(SIMPLER_MODE_KERNEL);
        if (rc != 0) return rc;
        KernelContextOps ops;
        ops.get_current_device = [](void *, int *id) noexcept {
            *id = 0;
            return 0;
        };
        ops.get_current_context = [](void *, void **ctx) noexcept {
            static int context;
            *ctx = &context;
            return 0;
        };
        ops.create_hidden_stream = ops.get_current_context;
        ops.create_event = ops.get_current_context;
        ops.destroy_hidden_stream = [](void *, void *) noexcept {
            return 0;
        };
        ops.destroy_event = ops.destroy_hidden_stream;
        rc = kernel_execution_state().initialize(0, ops);
        return rc == 0 ? kernel_execution_state().mark_ready_enqueued() : rc;
    }

    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        const int rc = Runner::prepare_execution(runtime, config, slot, identity, out);
        if (rc != 0) return rc;
        if (std::exchange(fail_descriptor_once, false)) control().fail_allocation = control().allocations + 1;
        int args_rc = (*out)->kernel_args.prepare_runtime_args(runtime, mem_alloc_, slot_persistent_args(slot));
        control().fail_allocation = 0;
        if (args_rc != 0) return args_rc;
        args_rc = run_fence(slot).arm(identity);
        if (args_rc != 0) return args_rc;
        return caller_orders_[slot]->arm(identity, this, &stream_aicpu_, &stream_aicore_);
    }
    void abandon_prepared_execution(PreparedExecution &prepared) noexcept override {
        retire_run_fence(prepared);
        prepared.kernel_args.release_run_view();
    }
    void add_independent_callable(int32_t id) {
        CallableState callable;
        callable.chip_buffer_hash = static_cast<uint64_t>(id);
        callable.host_orch_func_ptr = primary_entry_points();
        callables_[id] = std::move(callable);
        chip_callable_buffers_[id] = ChipCallableBuffer{};
    }
    int retain_recording(Attempt &attempt, void *model) {
        auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        auto &prepared = *state->prepared_execution;
        prepared.capture_model = model;
        prepared.streams = {&stream_aicpu_, &stream_aicore_};
        return retain_recording_resources(prepared, prepared.arena_bank);
    }
    void *descriptor(uint32_t slot) { return slot_persistent_args(slot).runtime_args; }
    void *source(uint32_t slot) { return slot_persistent_args(slot).runtime_upload.data(); }
    const void *fence(uint32_t slot) { return run_fences_[slot].get(); }
    const void *order(uint32_t slot) { return caller_orders_[slot].get(); }
    int publish(Attempt &attempt) {
        auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(attempt.storage);
        LaunchProgressSink sink;
        return state->prepared_execution->kernel_args.publish_runtime_args(false, this, sink);
    }
};
class ColdReservationRunner : public NativeGraphResourceRunner {
public:
    int stream_reservation_rc{0};
    bool supports_native_capture() const override { return true; }
    int reserve_native_capture_streams() override { return stream_reservation_rc; }
    uint64_t upload_capacity() { return slot_persistent_args(0).runtime_upload.capacity(); }
    void leave_retired_graph_backing() {
        const auto region = WorkspaceManager::arena_region(5, WorkspaceManager::ArenaRegion::GmHeap);
        void *old = workspace_.acquire(region, 0, 4096);
        ASSERT_NE(old, nullptr);
        int graph;
        ASSERT_TRUE(workspace_.retain_capture(&graph, {old}));
        void *replacement = workspace_.acquire(region, 0, 8192);
        ASSERT_NE(replacement, nullptr);
        ASSERT_NE(replacement, old);
        workspace_.note_published(region, replacement);
        // Graph retirement drops its claim; physical reclaim is a separate boundary.
        workspace_.release_capture(&graph);
        ASSERT_TRUE(workspace_.has_reclaimable());
    }
};

void destroy_native_model() {
    ASSERT_NE(control().capture_destroy_callback, nullptr);
    control().capture_destroy_callback(control().capture_destroy_data);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
}
}  // namespace

TEST_F(NativeAdmissionTest, CaptureFinalizeDefersRetiredWorkspaceToOutsideReservation) {
    for (const auto phase : {NativeRunPhase::Recorded, NativeRunPhase::Prepared}) {
        ColdReservationRunner runner;
        ASSERT_EQ(runner.adopt_kernel_context(), 0);
        int stream, model;
        Attempt run(runner, 0, 20);
        run.descriptor.caller_stream = &stream;
        run.args.tensor(0).transfer = TensorTransfer::NONE;
        run.args.tensor(0).address_space = AddressSpace::DEVICE;
        ASSERT_EQ(run.prepare(), 0);
        runner.leave_retired_graph_backing();
        const auto bytes = control().bytes;
        // Inject only the capture receipt: exercise real C ABI finalize and the
        // real workspace ledger, without pretending to execute an RTS graph.
        auto *state = static_cast<NativeRunContext<DeviceRunnerBase> *>(run.storage);
        state->capture_model = &model;
        state->phase.store(phase);
        if (phase == NativeRunPhase::Recorded) state->completion_rc = 0;
        control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
        ASSERT_EQ(simpler_finalize_run(&runner, run.storage), 0);
        EXPECT_EQ(control().bytes, bytes);
        control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
        ASSERT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), 0);
        EXPECT_LT(control().bytes, bytes);
    }
}

TEST_F(NativeAdmissionTest, CaptureReservationReportsReclaimFailureWithoutGrantingReadiness) {
    ColdReservationRunner runner;
    ASSERT_EQ(runner.adopt_kernel_context(), 0);
    int stream;
    Attempt warm(runner, 0, 20);
    warm.descriptor.caller_stream = &stream;
    warm.args.tensor(0).transfer = TensorTransfer::NONE;
    warm.args.tensor(0).address_space = AddressSpace::DEVICE;
    ASSERT_EQ(warm.prepare(), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, warm.storage), 0);
    runner.leave_retired_graph_backing();
    const auto bytes = control().bytes;
    control().initialize = true;
    control().fail_free_once = true;
    EXPECT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), -4327);
    EXPECT_EQ(control().bytes, bytes);
    EXPECT_EQ(runner.begin_native_capture_prepare(0, true), PTO_RUNTIME_ERR_UNSUPPORTED);
    // The existing ledger retains an unconfirmed free until terminal cleanup;
    // reservation retry must not silently retry or lose that charged allocation.
    ASSERT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), 0);
    EXPECT_EQ(control().bytes, bytes);
}

TEST_F(NativeAdmissionTest, CaptureReservationRequiresWarmFinalizedSlotAndIsIdempotent) {
    ColdReservationRunner runner;
    ASSERT_EQ(runner.adopt_kernel_context(), 0);
    int stream;
    EXPECT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), PTO_RUNTIME_ERR_UNSUPPORTED);
    Attempt warm(runner, 0, 20);
    warm.descriptor.caller_stream = &stream;
    warm.args.tensor(0).transfer = TensorTransfer::NONE;
    warm.args.tensor(0).address_space = AddressSpace::DEVICE;
    ASSERT_EQ(warm.prepare(), 0);
    EXPECT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(simpler_finalize_run(&runner, warm.storage), 0);
    EXPECT_EQ(runner.begin_native_capture_prepare(0, true), PTO_RUNTIME_ERR_UNSUPPORTED);
    const auto initial_capacity = runner.upload_capacity();
    runner.stream_reservation_rc = -771;
    EXPECT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), -771);
    EXPECT_EQ(runner.begin_native_capture_prepare(0, true), PTO_RUNTIME_ERR_UNSUPPORTED);
    runner.stream_reservation_rc = 0;
    ASSERT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), 0);
    EXPECT_GT(runner.upload_capacity(), initial_capacity);
    const auto allocations = control().allocations;
    const auto events = control().event_creates;
    const auto capacity = runner.upload_capacity();
    ASSERT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), 0);
    EXPECT_EQ(runner.upload_capacity(), capacity);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_EQ(control().event_creates, events);
    EXPECT_EQ(runner.begin_native_capture_prepare(0, true), 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    EXPECT_EQ(simpler_reserve_capture_resources(&runner, &stream, 0, 0), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(control().allocations, allocations);
    EXPECT_EQ(control().event_creates, events);
    EXPECT_EQ(runner.begin_native_capture_prepare(0, true), 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    EXPECT_EQ(runner.begin_native_capture_prepare(0, false), 0);
    EXPECT_EQ(runner.begin_native_capture_prepare(0, true), PTO_RUNTIME_ERR_UNSUPPORTED);
}

TEST_F(NativeAdmissionTest, CommonGraphResourceEntryRetainsRealPreparedGenerationsAcrossFinalize) {
    NativeGraphResourceRunner runner;
    ASSERT_EQ(runner.adopt_kernel_context(), 0);
    int stream, model;
    Attempt original(runner, 0, 20);
    original.descriptor.caller_stream = &stream;
    original.args.tensor(0).transfer = TensorTransfer::NONE;
    original.args.tensor(0).address_space = AddressSpace::DEVICE;
    ASSERT_EQ(original.prepare(), 0);
    void *old_descriptor = runner.descriptor(0);
    void *old_source = runner.source(0);
    const void *old_fence = runner.fence(0), *old_order = runner.order(0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    control().capture_model = &model;
    ASSERT_EQ(runner.retain_native_run_resources(&model, &stream, 0, 0), 0);
    ASSERT_EQ(runner.retain_native_run_resources(&model, &stream, 0, 0), 0);
    EXPECT_EQ(control().capture_callback_calls, 1u);
    ASSERT_EQ(runner.publish(original), 0);
    const auto *bytes = static_cast<const unsigned char *>(old_descriptor);
    std::vector<unsigned char> saved(bytes, bytes + 32);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    ASSERT_EQ(simpler_finalize_run(&runner, original.storage), 0);
    EXPECT_TRUE(original.empty());
    EXPECT_TRUE(runner.captured_resources_live());
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(runner.has_callable(1));
    EXPECT_EQ(finalize_device(&runner), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(runner.finalize_common(), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_EQ(simpler_retire_capture_resources(&runner), PTO_RUNTIME_ERR_INVALID_STATE);

    Attempt next(runner, 0, 21);
    next.args.tensor(0).transfer = TensorTransfer::NONE;
    next.args.tensor(0).address_space = AddressSpace::DEVICE;
    ASSERT_EQ(next.prepare(), 0);
    EXPECT_NE(runner.descriptor(0), old_descriptor);
    EXPECT_NE(runner.source(0), old_source);
    EXPECT_NE(runner.fence(0), old_fence);
    EXPECT_NE(runner.order(0), old_order);
    EXPECT_EQ(control().event_creates, 10u);
    EXPECT_EQ(control().event_destroys, 0u);
    ASSERT_EQ(runner.publish(next), 0);
    EXPECT_EQ(std::memcmp(saved.data(), old_descriptor, saved.size()), 0);
    destroy_native_model();
    EXPECT_EQ(simpler_retire_capture_resources(&runner), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(simpler_finalize_run(&runner, next.storage), 0);
    ASSERT_EQ(simpler_retire_capture_resources(&runner), 0);
    EXPECT_FALSE(runner.captured_resources_live());
    EXPECT_EQ(control().event_destroys, 5u);
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), 0);
    EXPECT_EQ(runner.finalize_common(), 0);
}

TEST_F(NativeAdmissionTest, GraphBlocksOnlyItsCallableUntilDestroyedAndDrained) {
    NativeGraphResourceRunner runner;
    ASSERT_EQ(runner.adopt_kernel_context(), 0);
    runner.add_independent_callable(2);
    int stream, model;
    Attempt original(runner, 0, 20);
    original.descriptor.caller_stream = &stream;
    original.args.tensor(0).transfer = TensorTransfer::NONE;
    original.args.tensor(0).address_space = AddressSpace::DEVICE;
    ASSERT_EQ(original.prepare(), 0);
    ASSERT_EQ(runner.retain_recording(original, &model), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, original.storage), 0);
    EXPECT_EQ(simpler_unregister_callable(&runner, 2), 0);
    EXPECT_FALSE(runner.has_callable(2));
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), PTO_RUNTIME_ERR_INVALID_STATE);
    destroy_native_model();
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(simpler_retire_capture_resources(&runner), 0);
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), 0);
}

TEST_F(NativeAdmissionTest, NativeGraphKeepsCallerAllocationAfterRunBorrowIsReleased) {
    NativeGraphResourceRunner runner;
    int stream, model;
    Attempt run(runner, 0, 20);
    ASSERT_EQ(run.prepare(), 0);
    void *input = runner.allocate_caller_buffer(64);
    ASSERT_NE(input, nullptr);
    CallerDeviceBuffers::Span span{reinterpret_cast<uint64_t>(input), 64};
    ASSERT_TRUE(runner.borrow_caller_buffers(1, &span, 1));
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    control().capture_model = &model;
    ASSERT_EQ(runner.retain_native_run_resources(&model, &stream, 0, 0), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, run.storage), 0);
    runner.release_caller_buffers(1, false);
    EXPECT_EQ(runner.free_caller_buffer(input), PTO_RUNTIME_ERR_INVALID_STATE);
    destroy_native_model();
    EXPECT_EQ(runner.free_caller_buffer(input), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(simpler_retire_capture_resources(&runner), 0);
    EXPECT_EQ(runner.free_caller_buffer(input), 0);
}

TEST_F(NativeAdmissionTest, CommonGraphRegistrationRefusalLeavesPreparedResourcesReusable) {
    NativeGraphResourceRunner runner;
    int stream, model;
    Attempt original(runner, 0, 20);
    ASSERT_EQ(original.prepare(), 0);
    void *descriptor = runner.descriptor(0), *source = runner.source(0);
    const void *fence = runner.fence(0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    control().capture_model = &model;
    control().capture_callback_rc = -4331;
    EXPECT_EQ(runner.retain_native_run_resources(&model, &stream, 0, 0), -4331);
    EXPECT_FALSE(runner.captured_resources_live());
    EXPECT_TRUE(runner.holds_epoch(20));
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    ASSERT_EQ(simpler_finalize_run(&runner, original.storage), 0);
    Attempt retry(runner, 0, 21);
    ASSERT_EQ(retry.prepare(), 0);
    EXPECT_EQ(runner.descriptor(0), descriptor);
    EXPECT_EQ(runner.source(0), source);
    EXPECT_EQ(runner.fence(0), fence);
    ASSERT_EQ(simpler_finalize_run(&runner, retry.storage), 0);
    ASSERT_EQ(runner.finalize_common(), 0);
}

TEST_F(NativeAdmissionTest, CommonGraphRetirementKeepsCodeAndDependenciesAcrossDrainAndFreeFailures) {
    NativeGraphResourceRunner runner;
    runner.set_program_initialized(true);
    int stream, model;
    Attempt original(runner, 0, 20);
    ASSERT_EQ(original.prepare(), 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    control().capture_model = &model;
    ASSERT_EQ(runner.retain_native_run_resources(&model, &stream, 0, 0), 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    ASSERT_EQ(simpler_finalize_run(&runner, original.storage), 0);
    Attempt next(runner, 0, 21);
    ASSERT_EQ(next.prepare(), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, next.storage), 0);
    destroy_native_model();
    control().initialize = true;
    control().fail_sync_call = control().sync_calls + 1;
    EXPECT_EQ(simpler_retire_capture_resources(&runner), -4328);
    EXPECT_TRUE(runner.captured_resources_live());
    EXPECT_EQ(finalize_device(&runner), PTO_RUNTIME_ERR_INVALID_STATE);
    EXPECT_TRUE(runner.program_initialized());
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), PTO_RUNTIME_ERR_INVALID_STATE);
    control().fail_sync_call = 0;
    control().fail_free_once = true;
    EXPECT_EQ(simpler_retire_capture_resources(&runner), -4327);
    EXPECT_TRUE(runner.captured_resources_live());
    EXPECT_EQ(runner.finalize_common(), PTO_RUNTIME_ERR_INVALID_STATE);
    control().fail_event_destroy_once = true;
    EXPECT_EQ(simpler_retire_capture_resources(&runner), -4332);
    EXPECT_TRUE(runner.captured_resources_live());
    EXPECT_EQ(simpler_unregister_callable(&runner, 1), PTO_RUNTIME_ERR_INVALID_STATE);
    ASSERT_EQ(simpler_retire_capture_resources(&runner), 0);
    EXPECT_FALSE(runner.captured_resources_live());
    control().initialize = false;
    ASSERT_EQ(runner.finalize_common(), 0);
}

TEST_F(NativeAdmissionTest, GraphRetirementCallerCallbackRefusesCrossThreadMutationAndPreparation) {
    NativeGraphResourceRunner runner;
    int stream, model;
    Attempt original(runner, 0, 20);
    ASSERT_EQ(original.prepare(), 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    control().capture_model = &model;
    ASSERT_EQ(runner.retain_native_run_resources(&model, &stream, 0, 0), 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    ASSERT_EQ(simpler_finalize_run(&runner, original.storage), 0);
    destroy_native_model();
    int callbacks_seen = 0;
    control().sync_hook = [&] {
        ++callbacks_seen;
        auto reentry = std::async(std::launch::async, [&] {
            EXPECT_EQ(simpler_unregister_callable(&runner, 1), PTO_RUNTIME_ERR_INVALID_STATE);
            EXPECT_EQ(simpler_retire_capture_resources(&runner), PTO_RUNTIME_ERR_INVALID_STATE);
            EXPECT_EQ(finalize_device(&runner), PTO_RUNTIME_ERR_INVALID_STATE);
            Attempt nested(runner, 0, 21);
            EXPECT_NE(nested.prepare(), 0);
            EXPECT_TRUE(nested.empty());
        });
        reentry.get();
    };
    ASSERT_EQ(simpler_retire_capture_resources(&runner), 0);
    EXPECT_EQ(callbacks_seen, 1);
    control().sync_hook = {};
    Attempt next(runner, 0, 22);
    ASSERT_EQ(next.prepare(), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, next.storage), 0);
    ASSERT_EQ(runner.finalize_common(), 0);
}

TEST_F(NativeAdmissionTest, CapturedDescriptorReplacementRefusalPreservesOriginalAndRetry) {
    NativeGraphResourceRunner runner;
    int stream, model;
    Attempt original(runner, 0, 20);
    ASSERT_EQ(original.prepare(), 0);
    void *descriptor = runner.descriptor(0), *source = runner.source(0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_ACTIVE;
    control().capture_model = &model;
    ASSERT_EQ(runner.retain_native_run_resources(&model, &stream, 0, 0), 0);
    control().capture_status = ACL_MODEL_RI_CAPTURE_STATUS_NONE;
    ASSERT_EQ(simpler_finalize_run(&runner, original.storage), 0);
    runner.fail_descriptor_once = true;
    Attempt failed(runner, 0, 21);
    EXPECT_NE(failed.prepare(), 0);
    EXPECT_TRUE(failed.consume_release());
    EXPECT_EQ(runner.descriptor(0), descriptor);
    EXPECT_EQ(runner.source(0), source);
    EXPECT_TRUE(runner.captured_resources_live());
    EXPECT_FALSE(runner.holds_epoch(21));
    Attempt retry(runner, 0, 22);
    ASSERT_EQ(retry.prepare(), 0);
    EXPECT_NE(runner.descriptor(0), descriptor);
    EXPECT_NE(runner.source(0), source);
    ASSERT_EQ(simpler_finalize_run(&runner, retry.storage), 0);
    destroy_native_model();
    ASSERT_EQ(simpler_retire_capture_resources(&runner), 0);
    ASSERT_EQ(runner.finalize_common(), 0);
}

}  // namespace native_admission_test
