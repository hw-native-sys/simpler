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
#include <cstring>
#include "aicpu/hbg_execution_image.h"
#include "host/platform_compile_info.h"
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
namespace {
class HeapLeaseRunner : public Runner {
public:
    explicit HeapLeaseRunner(AdmissionMode mode) :
        Runner(true, mode) {}
    size_t heap_leases() const { return heap_pool_->live_leases(); }
    size_t heap_allocations() const { return heap_pool_->current_base() != nullptr ? 1u : 0u; }
    bool fail_preparation{false};
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        if (fail_preparation) return -4311;
        return Runner::prepare_execution(runtime, config, slot, identity, out);
    }
};
}  // namespace

TEST_F(NativeAdmissionTest, OversizeGraphFailsBeforeHeapAllocationAndNextPrepareRecovers) {
    HeapLeaseRunner runner(AdmissionMode::FixedGrant);
    Attempt oversized(runner, 0, 10);
    oversized.config.runtime_env.ring_heap[0] = 1024;
    EXPECT_EQ(oversized.prepare(), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
    EXPECT_EQ(runner.pooled_heap_base(), nullptr);
    EXPECT_EQ(runner.heap_leases(), 0u);
    EXPECT_TRUE(oversized.consume_release());
    Attempt valid(runner, 0, 11);
    ASSERT_EQ(valid.prepare(), 0);
    EXPECT_NE(runner.prepared_heap_base(0, 11), nullptr);
    EXPECT_EQ(simpler_finalize_run(&runner, valid.storage), 0);
}

TEST_F(NativeAdmissionTest, ManualSharedHeapQuotaRefusalReportsResourceUnavailable) {
    HeapLeaseRunner runner(AdmissionMode::FixedGrant);
    ASSERT_TRUE(runner.set_workspace_limit(64 * kUnit));
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(0, 10);
        EXPECT_EQ(
            runner.setup_static_arena(0, {0, 0, 0}, true, 128 * kUnit, 0, kUnit),
            SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE
        );
    }
    EXPECT_EQ(runner.heap_leases(), 0u);
    EXPECT_EQ(runner.heap_allocations(), 0u);
    EXPECT_EQ(runner.charged(), 0u);
    EXPECT_EQ(runner.abort_run_plan(0, 10), 0);
}

TEST_F(NativeAdmissionTest, FixedHeapCapacityChangePreservesLivePeer) {
    HeapLeaseRunner runner(AdmissionMode::FixedGrant);
    ASSERT_TRUE(runner.set_workspace_limit(160 * kUnit));
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(0, 10);
        ASSERT_EQ(runner.setup_static_arena(0, {0, 0, 0}, true, 64 * kUnit, 0, kUnit), 0);
        runner.note_run_resources_proven(0, 10);
    }
    void *old = runner.pooled_heap_base();
    std::memset(old, 0x5a, 64 * kUnit);
    runner.note_workspace_run_fact(NativeRunIdentity{10, 10, 10, 0}, Fact::Launched);
    const auto charged = runner.charged();
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(1, 20);
        EXPECT_EQ(
            runner.setup_static_arena(1, {0, 0, 1}, true, 128 * kUnit, 0, kUnit), PTO_RUNTIME_ERR_INVALID_ARGUMENT
        );
    }
    EXPECT_EQ(runner.pooled_heap_base(), old);
    EXPECT_EQ(runner.heap_leases(), 1u);
    EXPECT_EQ(runner.heap_allocations(), 1u);
    EXPECT_EQ(runner.charged(), charged);
    const std::vector<unsigned char> expected(64 * kUnit, 0x5a);
    EXPECT_EQ(std::memcmp(old, expected.data(), expected.size()), 0);
    EXPECT_EQ(runner.abort_run_plan(1, 20), 0);
    runner.release_heap_lease(0);
    runner.retire(0, 10);
}

TEST_F(NativeAdmissionTest, IdleSharedHeapCannotBeResizedOnAnotherSlot) {
    HeapLeaseRunner runner(AdmissionMode::Automatic);
    ASSERT_TRUE(runner.set_workspace_limit(160 * kUnit));
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(0, 10);
        ASSERT_EQ(runner.setup_static_arena(0, {0, 0, 0}, true, 64 * kUnit, 0, kUnit), 0);
        runner.note_run_resources_proven(0, 10);
    }
    runner.release_heap_lease(0);
    runner.note_workspace_run_fact(NativeRunIdentity{10, 10, 10, 0}, Fact::NoDeviceSubmission);
    runner.retire(0, 10);
    ASSERT_EQ(runner.heap_leases(), 0u);
    ASSERT_GT(runner.charged(), 64 * kUnit);
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(1, 20);
        EXPECT_EQ(
            runner.setup_static_arena(1, {0, 0, 1}, true, 128 * kUnit, 0, kUnit), PTO_RUNTIME_ERR_INVALID_ARGUMENT
        );
    }
    EXPECT_EQ(runner.heap_leases(), 0u);
    EXPECT_EQ(runner.heap_allocations(), 1u);
    EXPECT_LE(runner.charged(), 160 * kUnit);
    EXPECT_EQ(runner.abort_run_plan(1, 20), 0);
    runner.note_workspace_run_fact(NativeRunIdentity{20, 20, 20, 1}, Fact::NoDeviceSubmission);
    runner.retire(1, 20);
}

TEST_F(NativeAdmissionTest, IdleFixedHeapSurvivesTemporaryAllocationPressure) {
    HeapLeaseRunner runner(AdmissionMode::Automatic);
    ASSERT_TRUE(runner.set_workspace_limit(160 * kUnit));
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(0, 10);
        ASSERT_EQ(runner.setup_static_arena(0, {0, 0, 0}, true, 64 * kUnit, 0, kUnit), 0);
        runner.note_run_resources_proven(0, 10);
    }
    void *base = runner.pooled_heap_base();
    runner.release_heap_lease(0);
    runner.note_workspace_run_fact(NativeRunIdentity{10, 10, 10, 0}, Fact::NoDeviceSubmission);
    runner.retire(0, 10);
    ASSERT_EQ(runner.heap_leases(), 0u);
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(1, 20);
        void *temporary = nullptr;
        size_t capacity = 0;
        EXPECT_NE(runner.acquire_retained_temp(1, 128 * kUnit, &temporary, &capacity), 0);
    }
    EXPECT_EQ(runner.pooled_heap_base(), base);
    EXPECT_EQ(runner.heap_allocations(), 1u);
    EXPECT_EQ(runner.abort_run_plan(1, 20), 0);
}

TEST_F(NativeAdmissionTest, FixedCapacityPersistsAfterLastPeerRetires) {
    HeapLeaseRunner runner(AdmissionMode::Automatic);
    ASSERT_TRUE(runner.set_workspace_limit(160 * kUnit));
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(0, 10);
        ASSERT_EQ(runner.setup_static_arena(0, {0, 0, 0}, true, 64 * kUnit, 0, kUnit), 0);
        runner.note_run_resources_proven(0, 10);
    }
    void *old = runner.pooled_heap_base();
    std::memset(old, 0x5a, 64 * kUnit);
    runner.note_workspace_run_fact(NativeRunIdentity{10, 10, 10, 0}, Fact::Launched);
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(1, 20);
        EXPECT_NE(runner.setup_static_arena(1, {0, 0, 1}, true, 128 * kUnit, 0, kUnit), 0);
    }
    EXPECT_EQ(runner.pooled_heap_base(), old);
    EXPECT_EQ(runner.heap_leases(), 1u);
    EXPECT_EQ(runner.heap_allocations(), 1u);
    const std::vector<unsigned char> expected(64 * kUnit, 0x5a);
    EXPECT_EQ(std::memcmp(old, expected.data(), expected.size()), 0);
    EXPECT_EQ(runner.abort_run_plan(1, 20), 0);
    runner.release_heap_lease(0);
    runner.retire(0, 10);
    {
        DeviceRunnerBase::PrepareWorkspaceScope scope(1, 21);
        EXPECT_EQ(
            runner.setup_static_arena(1, {0, 0, 1}, true, 128 * kUnit, 0, kUnit), PTO_RUNTIME_ERR_INVALID_ARGUMENT
        );
    }
    EXPECT_EQ(runner.heap_allocations(), 1u);
    EXPECT_EQ(runner.abort_run_plan(1, 21), 0);
    runner.note_workspace_run_fact(NativeRunIdentity{21, 21, 21, 1}, Fact::NoDeviceSubmission);
    runner.retire(1, 21);
}

TEST_F(NativeAdmissionTest, FailedPrepareReturnsHeapLeaseBesideLivePeer) {
    HeapLeaseRunner runner(AdmissionMode::FixedGrant);
    Attempt peer(runner, 0, 10);
    ASSERT_EQ(peer.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, peer.storage), 0);
    ASSERT_EQ(runner.heap_leases(), 1u);
    runner.fail_preparation = true;
    for (uint64_t epoch : {20u, 21u}) {
        Attempt failed(runner, 1, epoch);
        EXPECT_EQ(failed.prepare(), -4311);
        EXPECT_TRUE(failed.consume_release());
        EXPECT_EQ(runner.heap_leases(), 1u);
    }
    ASSERT_EQ(simpler_finalize_run(&runner, peer.storage), 0);
    EXPECT_EQ(runner.heap_leases(), 0u);
    ASSERT_EQ(runner.setup_static_arena(1, {0, 0, 1}, true, test_heap_capacity(), 0, 4096), 0);
    EXPECT_EQ(runner.heap_allocations(), 1u);
    EXPECT_EQ(runner.abort_run_plan(1, 0), 0);
    EXPECT_EQ(runner.heap_leases(), 0u);
}

TEST_F(NativeAdmissionTest, PlanlessAbortReturnsOnlyTheUnpublishedHeapLease) {
    HeapLeaseRunner runner(AdmissionMode::FixedGrant);
    Attempt peer(runner, 0, 10);
    ASSERT_EQ(peer.prepare(), 0);
    ASSERT_EQ(simpler_launch_run(&runner, peer.storage), 0);
    const void *peer_heap = runner.prepared_heap_base(0, 10);
    ASSERT_NE(peer_heap, nullptr);
    ASSERT_EQ(runner.setup_static_arena(1, {0, 0, 1}, true, test_heap_capacity(), 0, 4096), 0);
    ASSERT_EQ(runner.pooled_heap_base(), peer_heap);
    EXPECT_EQ(runner.prepared_heap_base(0, 10), peer_heap);
    EXPECT_EQ(runner.prepared_heap_base(0, 11), nullptr);
    runner.enable_automatic_admission();
    ASSERT_EQ(runner.plan(1).phase, AutoAdmissionState::Phase::None);
    ASSERT_EQ(runner.abort_run_plan(1, 20), 0);
    EXPECT_EQ(runner.heap_leases(), 1u);
    EXPECT_EQ(runner.prepared_heap_base(0, 10), peer_heap);
    ASSERT_EQ(simpler_finalize_run(&runner, peer.storage), 0);
    EXPECT_EQ(runner.heap_leases(), 0u);
    EXPECT_EQ(runner.heap_allocations(), 1u);
    EXPECT_EQ(runner.prepared_heap_base(0, 10), nullptr);
}

TEST_F(NativeAdmissionTest, ReplacingSlotHeapLeaseKeepsFixedAllocation) {
    HeapLeaseRunner runner(AdmissionMode::FixedGrant);
    ASSERT_EQ(runner.setup_static_arena(0, {0, 0, 0}, true, 4096, 0, 4096), 0);
    ASSERT_EQ(runner.heap_leases(), 1u);
    ASSERT_EQ(runner.setup_static_arena(0, {0, 0, 0}, true, 4096, 0, 4096), 0);
    EXPECT_EQ(runner.heap_leases(), 1u);
    EXPECT_EQ(runner.heap_allocations(), 1u);
    EXPECT_EQ(runner.abort_run_plan(0, 0), 0);
    EXPECT_EQ(runner.heap_leases(), 0u);
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
    EXPECT_NE(outer_runner.pooled_heap(1), nullptr);
    EXPECT_TRUE(outer_runner.arena_referenced(outer.descriptor.arena_bank, 2));
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
    EXPECT_TRUE(runner.arena_referenced(outer.descriptor.arena_bank, 2));
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
    void *const idle_image = runner.arena_region_backing(warmup.descriptor.arena_bank, 2);
    ASSERT_NE(idle_heap, nullptr);
    ASSERT_NE(idle_image, nullptr);
    Attempt outer(runner, 1, 20);
    Attempt warm_consumer(runner, 2, 30);
    bool entered = false;
    ASSERT_NE(runner.retained_temp_addr(2), 0u);
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
        EXPECT_EQ(runner.arena_region_backing(warmup.descriptor.arena_bank, 2), idle_image);
        if (rc == 0) EXPECT_EQ(simpler_finalize_run(&runner, warm_consumer.storage), 0);
        // Exercise pressure reclamation while the outer run reserves bank zero.
        EXPECT_GT(runner.trim_idle_run_resources(outer.descriptor.pipeline_slot), 0u);
    };
    ASSERT_EQ(outer.prepare(), 0);
    EXPECT_TRUE(entered);
    EXPECT_EQ(runner.arena_region_backing(2, 0), nullptr);
    EXPECT_EQ(runner.arena_region_backing(outer.descriptor.arena_bank, 2), idle_image);
    EXPECT_EQ(runner.retained_temp_addr(2), 0u);
    EXPECT_TRUE(runner.arena_referenced(outer.descriptor.arena_bank, 2));
    ASSERT_EQ(simpler_finalize_run(&runner, outer.storage), 0);
    const auto allocations = control().allocations;
    ASSERT_EQ(warm_consumer.prepare(), 0);
    EXPECT_GT(control().allocations, allocations);
    EXPECT_TRUE(runner.arena_referenced(warm_consumer.descriptor.arena_bank, 2));
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
    EXPECT_TRUE(outer_runner.arena_referenced(outer.descriptor.arena_bank, 2));
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

TEST_F(NativeAdmissionTest, RealHbgCallbackSurvivesNativeParkAndAicpuSubmissionFailureExactlyOnce) {
    auto owned_runner = make_fenced_completion_runner(-88);
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
    EXPECT_EQ(simpler_launch_run(&runner, candidate.storage), -88);
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
    std::shared_ptr<const void> shared_resource;
    size_t heap_leases() const { return heap_pool_->live_leases(); }
    size_t heap_allocations() const { return heap_pool_->current_base() != nullptr ? 1u : 0u; }
    void trim_heap() { (void)reclaim_workspace_obsolete(); }
    HbgImageCaptureRunner() {
        static int binary;
        aicore_bin_handle_ = &binary;
    }
    int initialize_kernel() {
        control().loader_start_rc = 0;
        int rc = execution_mode_latch().latch(SIMPLER_MODE_KERNEL);
        if (rc == 0) rc = kernel_execution_state().initialize(0, make_onboard_kernel_context_ops());
        return rc == 0 ? kernel_execution_state().mark_ready_enqueued() : rc;
    }
    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *out
    ) override {
        int rc = Runner::prepare_execution(runtime, config, slot, identity, out);
        if (rc == 0 && shared_resource) {
            auto publication = *runtime.prepared_publication();
            publication.resources.push_back(std::move(shared_resource));
            runtime.retain_prepared_publication(std::move(publication));
        }
        if (rc == 0) rc = ensure_device_run_result_region(identity, (*out)->kernel_args, (*out)->dfx);
        return rc == 0 ? init_runtime_args_with_metadata(runtime, (*out)->kernel_args, slot_persistent_args(slot)) : rc;
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
                sink.mark_submitted();
                return record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicore);
            },
            [&](LaunchProgressSink &sink) {
                auto &args = prepared.kernel_args;
                const int launch_rc = launch_aicpu_payload(
                    prepared.streams.aicpu, args.launch_payload(), args.launch_payload_bytes(),
                    host::KernelNames::RunName, 1, args.host_inputs(), args.host_input_count()
                );
                if (launch_rc != 0) return launch_rc;
                sink.mark_submitted();
                const int boundary_rc = record_run_boundary(prepared, RunCompletionFence::StreamRole::Aicpu);
                return boundary_rc != 0 ? boundary_rc : queue_caller_exit(prepared);
            }
        );
    }
    int prepare_upload(Runtime &runtime, KernelArgsHelper &helper) {
        return helper.prepare_runtime_args(runtime, mem_alloc_, slot_persistent_args(0));
    }
};
}  // namespace

TEST_F(NativeAdmissionTest, RetainedRunSharesDeviceWorkspaceAcrossSlots) {
    HbgImageCaptureRunner runner;
    ASSERT_EQ(runner.initialize_kernel(), 0);
    auto capture_args = [](Attempt &attempt) {
        attempt.descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
        attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
        attempt.args.tensor(0).transfer = TensorTransfer::NONE;
    };
    Attempt warm(runner, 0, 10030);
    capture_args(warm);
    ASSERT_EQ(warm.prepare(), 0);
    ASSERT_EQ(simpler_finalize_run(&runner, warm.storage), 0);
    void *heap0 = runner.pooled_heap_base();
    void *image0 = runner.arena_region_backing(0, 2);
    ASSERT_NE(heap0, nullptr);
    ASSERT_NE(image0, nullptr);
    Attempt captured(runner, 1, 10031);
    capture_args(captured);
    ASSERT_EQ(captured.prepare(), 0);
    EXPECT_EQ(runner.prepared_heap_base(1, 10031), heap0);
    const auto publication = [](const Attempt &run) {
        return static_cast<NativeRunContext<DeviceRunnerBase> *>(run.storage)->runtime->prepared_publication();
    };
    auto first = publication(captured);
    ASSERT_NE(first, nullptr);
    RuntimeHandle reference = nullptr;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_finalize_run(&runner, captured.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    int caller;
    ASSERT_EQ(simpler_retain_run_resources(&runner, captured.storage, &caller, &reference), 0);
    control().recording_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, captured.storage, &caller, nullptr), 0);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, captured.storage), 0);
    ASSERT_TRUE(runner.set_workspace_limit(runner.charged()));
    Attempt reuse(runner, 0, 10032);
    capture_args(reuse);
    ASSERT_EQ(reuse.prepare(), 0);
    EXPECT_EQ(runner.prepared_heap_base(0, 10032), heap0);
    EXPECT_EQ(runner.arena_region_backing(0, 2), image0);
    EXPECT_EQ(runner.prepared_heap_base(1, 10031), heap0);
    auto second = publication(reuse);
    ASSERT_NE(second, nullptr);
    EXPECT_EQ(first->device_target, second->device_target);
    EXPECT_NE(first->source, second->source);
    ASSERT_EQ(simpler_finalize_run(&runner, reuse.storage), 0);
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
}

TEST_F(NativeAdmissionTest, CapturedHbgImageUsesRtsSnapshotThroughOriginalRunRetirement) {
    HbgImageCaptureRunner runner;
    ASSERT_EQ(runner.initialize_kernel(), 0);
    runner.shared_resource = std::make_shared<int>(0);
    const std::weak_ptr<const void> shared_resource = runner.shared_resource;
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
    void *const working_arena = publication->device_target;
    publication.reset();
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, attempt.storage, &caller, &reference), 0);
    control().recording_active = true;
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_finalize_run(&runner, attempt.storage);
        if (reference != nullptr) (void)simpler_release_run_resources(&runner, reference);
    });
    const std::weak_ptr<const PreparedRunResources> backing = state->prepared_execution->prepared_resources;
    control().defer_async_copies = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), 0);
    EXPECT_FALSE(std::any_of(control().deferred_copies.begin(), control().deferred_copies.end(), [&](auto &copy) {
        return copy.destination == working_arena;
    })) << "RTS owns the H2D of the launch packet";
    ASSERT_EQ(control().aicpu_argument_copies.size(), 1u);
    const auto &packet = control().aicpu_argument_copies.front();
    ASSERT_GE(packet.size(), sizeof(KernelArgs));
    KernelArgs header{};
    std::memcpy(&header, packet.data(), sizeof(header));
    const auto packet_begin = reinterpret_cast<uintptr_t>(packet.data());
    const auto image_address = reinterpret_cast<uintptr_t>(header.execution_image);
    ASSERT_GE(image_address, packet_begin);
    ASSERT_LE(image_address - packet_begin, packet.size());
    EXPECT_EQ((image_address - packet_begin + sizeof(HbgExecutionImage)) % 64, 0u);
    ASSERT_GE(packet.size() - (image_address - packet_begin), sizeof(HbgExecutionImage) + expected.size());
    HbgExecutionImage image{};
    std::memcpy(&image, header.execution_image, sizeof(image));
    ASSERT_EQ(image.working_address, reinterpret_cast<uint64_t>(working_arena));
    ASSERT_EQ(image.bytes, expected.size());
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
    EXPECT_FALSE(backing.expired());
    EXPECT_FALSE(shared_resource.expired());
    for (const auto &copy : control().deferred_copies)
        std::memcpy(copy.destination, copy.source, copy.bytes);
    // The transport stub records raw bytes in a vector. Replay from an aligned RTS allocation.
    struct alignas(64) CacheLine { unsigned char data[64]; };
    std::vector<CacheLine> replay_storage((packet.size() + 63) / 64);
    std::memcpy(replay_storage.data(), packet.data(), packet.size());
    const auto *image_packet = reinterpret_cast<const unsigned char *>(replay_storage.data()) +
                               (image_address - packet_begin);
    const auto *image_data = image_packet + sizeof(image);
    ASSERT_LE(image.mutable_prefix_bytes, expected.size());
    ASSERT_EQ(std::memcmp(image_data, expected.data(), expected.size()), 0);
    for (int replay = 0; replay < 2; ++replay) {
        std::memset(working_arena, 0xa5, image.mutable_prefix_bytes);
        ASSERT_EQ(
            restore_hbg_execution_image(
                image_packet, [](const void *, size_t) {}, [](const void *, size_t) {}
            ),
            0
        );
        for (uint64_t i = 0; i < image.reference_count; ++i) {
            HbgImageReference ref{};
            std::memcpy(&ref, image_data + image.bytes + i * sizeof(ref), sizeof(ref));
            EXPECT_EQ(*reinterpret_cast<const uint64_t *>(ref.device_field_address),
                      reinterpret_cast<uint64_t>(image_data + ref.image_offset));
        }
        if (image.reference_count == 0)
            EXPECT_EQ(std::memcmp(working_arena, expected.data(), image.mutable_prefix_bytes), 0);
    }
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_TRUE(backing.expired());
    EXPECT_TRUE(shared_resource.expired());
    cleanup.dismiss();
}

TEST_F(NativeAdmissionTest, SchedulerScratchIsSharedAcrossSlotsAndIsolatesProfilingReadback) {
    Runner runner;
    DfxRunConfig profiled;
    profiled.chip_swimlane_level = ChipSwimlaneLevel::SCHED_PHASES;
    void *targets[4]{};
    void *sources[4]{};
    // Ordinary, profiled, profiled, ordinary. Keep all four preparations alive.
    for (uint32_t slot = 0; slot < 4; ++slot) {
        runner.begin_host_phase_run(slot, slot == 1 || slot == 2 ? profiled : DfxRunConfig{});
        ASSERT_EQ(runner.acquire_scheduler_state_storage(slot, 256, 64, &targets[slot], &sources[slot]), 0);
        std::memset(sources[slot], static_cast<int>(slot + 1), 256);
    }
    EXPECT_EQ(targets[0], targets[3]);
    EXPECT_NE(targets[0], targets[1]);
    EXPECT_NE(targets[0], targets[2]);
    EXPECT_NE(targets[1], targets[2]);
    for (uint32_t slot = 0; slot < 4; ++slot)
        EXPECT_EQ(static_cast<unsigned char *>(sources[slot])[255], slot + 1);
}

TEST_F(NativeAdmissionTest, CapturePreparationAcquiresHeapAndSchedulerStorage) {
    HbgImageCaptureRunner runner;
    ASSERT_EQ(runner.initialize_kernel(), 0);
    Attempt attempt(runner, 0, 7003);
    attempt.descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
    attempt.args.tensor(0).address_space = AddressSpace::DEVICE;
    attempt.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(attempt.prepare(), 0);
    EXPECT_NE(runner.prepared_heap_base(0, 7003), nullptr);
    EXPECT_EQ(runner.heap_leases(), 1u);
    if (std::strcmp(get_platform(), "a5sim") == 0) EXPECT_GT(runner.scheduler_capacity(0), 0u);
    ASSERT_EQ(simpler_finalize_run(&runner, attempt.storage), 0);
}

TEST_F(NativeAdmissionTest, RetainedCaptureHeapSurvivesEagerPrepareTrimAndFailedRelease) {
    HbgImageCaptureRunner runner;
    ASSERT_EQ(runner.initialize_kernel(), 0);
    Attempt capture(runner, 0, 7010);
    capture.descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
    capture.args.tensor(0).address_space = AddressSpace::DEVICE;
    capture.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(capture.prepare(), 0);
    void *heap = runner.prepared_heap_base(0, 7010);
    ASSERT_NE(heap, nullptr);
    std::memset(heap, 0x5a, 16);
    const std::array<unsigned char, 16> expected{0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a,
                                                 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a, 0x5a};
    int caller;
    RuntimeHandle reference = nullptr;
    RuntimeHandle successor_reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, capture.storage, &caller, &reference), 0);
    auto cleanup = RAIIScopeGuard([&] {
        control().recording_active = false;
        control().stream_sync_rc = 0;
        (void)simpler_finalize_run(&runner, capture.storage);
        if (reference) (void)simpler_release_run_resources(&runner, reference);
        if (successor_reference) (void)simpler_release_run_resources(&runner, successor_reference);
    });
    control().recording_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, capture.storage, &caller, nullptr), 0);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, capture.storage), 0);
    ASSERT_EQ(runner.heap_leases(), 1u);
    runner.trim_heap();
    EXPECT_EQ(runner.pooled_heap_base(), heap);

    Attempt successor(runner, 0, 7012);
    successor.descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
    successor.args.tensor(0).address_space = AddressSpace::DEVICE;
    successor.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(successor.prepare(), 0);
    EXPECT_EQ(runner.prepared_heap_base(0, 7012), heap);
    ASSERT_EQ(simpler_retain_run_resources(&runner, successor.storage, &caller, &successor_reference), 0);
    control().recording_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, successor.storage, &caller, nullptr), 0);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, successor.storage), 0);
    EXPECT_EQ(runner.heap_leases(), 2u);

    Attempt eager(runner, 0, 7013);
    eager.args.tensor(0).address_space = AddressSpace::DEVICE;
    eager.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(eager.prepare(), 0);
    EXPECT_EQ(runner.prepared_heap_base(0, 7013), heap);
    EXPECT_EQ(runner.heap_leases(), 3u);
    EXPECT_EQ(runner.heap_allocations(), 1u);
    EXPECT_EQ(std::memcmp(heap, expected.data(), expected.size()), 0)
        << "preparation must not overwrite another execution's shared scratch";
    ASSERT_EQ(simpler_finalize_run(&runner, eager.storage), 0);
    EXPECT_EQ(runner.heap_leases(), 2u);

    control().stream_sync_rc = -4312;
    EXPECT_EQ(simpler_release_run_resources(&runner, reference), -4312);
    runner.trim_heap();
    EXPECT_EQ(runner.heap_leases(), 2u);
    EXPECT_EQ(runner.pooled_heap_base(), heap);
    control().stream_sync_rc = 0;
    ASSERT_EQ(simpler_release_run_resources(&runner, reference), 0);
    reference = nullptr;
    EXPECT_EQ(runner.heap_leases(), 1u);
    runner.trim_heap();
    EXPECT_EQ(runner.pooled_heap_base(), heap);
    ASSERT_EQ(simpler_release_run_resources(&runner, successor_reference), 0);
    successor_reference = nullptr;
    EXPECT_EQ(runner.heap_leases(), 0u);
    runner.trim_heap();
    EXPECT_EQ(runner.heap_allocations(), 1u);
}

TEST_F(NativeAdmissionTest, RetainedRunSharesWritableDeviceStorageWithSameSlotSuccessor) {
    HbgImageCaptureRunner runner;
    ASSERT_EQ(runner.initialize_kernel(), 0);
    Attempt capture(runner, 0, 7004);
    capture.descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
    capture.args.tensor(0).address_space = AddressSpace::DEVICE;
    capture.args.tensor(0).transfer = TensorTransfer::NONE;
    auto reset_recording = RAIIScopeGuard([] {
        control().recording_active = false;
    });
    ASSERT_EQ(capture.prepare(), 0);
    auto *capture_state = static_cast<NativeRunContext<DeviceRunnerBase> *>(capture.storage);
    const auto capture_publication = capture_state->runtime->prepared_publication();
    ASSERT_NE(capture_publication, nullptr);
    const void *const capture_target = capture_publication->device_target;
    const uint64_t capture_state_base = capture_state->runtime->dev.scheduler_bootstrap.worker_context_base;
    int caller;
    RuntimeHandle reference = nullptr;
    ASSERT_EQ(simpler_retain_run_resources(&runner, capture.storage, &caller, &reference), 0);
    auto release = RAIIScopeGuard([&] {
        control().recording_active = false;
        (void)simpler_release_run_resources(&runner, reference);
    });
    control().recording_active = true;
    ASSERT_EQ(simpler_launch_run_on_stream(&runner, capture.storage, &caller, nullptr), 0);
    control().recording_active = false;
    ASSERT_EQ(simpler_finalize_run(&runner, capture.storage), 0);

    // Preparing the successor must leave the previous device contents intact.
    unsigned char sentinel = 0xa5;
    std::memcpy(capture_publication->device_target, &sentinel, sizeof(sentinel));
    Attempt successor(runner, 0, 7005);
    successor.args.tensor(0).address_space = AddressSpace::DEVICE;
    successor.args.tensor(0).transfer = TensorTransfer::NONE;
    ASSERT_EQ(successor.prepare(), 0);
    auto *successor_state = static_cast<NativeRunContext<DeviceRunnerBase> *>(successor.storage);
    const auto successor_publication = successor_state->runtime->prepared_publication();
    ASSERT_NE(successor_publication, nullptr);
    EXPECT_EQ(successor_publication->device_target, capture_target);
    EXPECT_EQ(*static_cast<const unsigned char *>(capture_target), sentinel);
    if (std::strcmp(get_platform(), "a5sim") == 0)
        EXPECT_EQ(successor_state->runtime->dev.scheduler_bootstrap.worker_context_base, capture_state_base);
    ASSERT_EQ(simpler_finalize_run(&runner, successor.storage), 0);
}

TEST_F(NativeAdmissionTest, HbgPrerequisitesAndRtsImageAreDeliveredFromOwnedSnapshots) {
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
    ASSERT_EQ(control().deferred_copies.size(), 2u);
    EXPECT_EQ(control().deferred_copies[0].destination, first_target.data());
    EXPECT_EQ(control().deferred_copies[1].destination, second_target.data());
    EXPECT_EQ(first_target, (std::array<uint8_t, 4>{}));
    EXPECT_EQ(second_target, (std::array<uint8_t, 4>{}));
    EXPECT_EQ(image_target, (std::array<uint8_t, 4>{}));
    for (const auto &copy : control().deferred_copies)
        std::memcpy(copy.destination, copy.source, copy.bytes);
    EXPECT_EQ(first_target, expected_first);
    EXPECT_EQ(second_target, expected_second);
    ASSERT_EQ(helper.host_input_count(), 2);
    EXPECT_EQ(helper.host_inputs()[1].addrOffset, offsetof(KernelArgs, execution_image));
    const auto *packet = static_cast<const unsigned char *>(helper.launch_payload());
    ASSERT_NE(packet, nullptr);
    const auto *image = packet + helper.host_inputs()[1].dataOffset;
    EXPECT_EQ(image_target, (std::array<uint8_t, 4>{}));
    ASSERT_EQ(restore_hbg_execution_image(image, [](const void *, size_t) {}, [](const void *, size_t) {}), 0);
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

TEST_F(NativeAdmissionTest, FailedCapturedHbgRtsSubmissionPreservesBackingThroughCheckedRelease) {
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
    control().aicpu_payload = [](void *, void *, size_t, const std::string &) {
        return -4288;
    };
    EXPECT_EQ(simpler_launch_run_on_stream(&runner, attempt.storage, &caller, nullptr), -4288);
    ASSERT_EQ(control().aicpu_argument_copies.size(), 1u);
    EXPECT_FALSE(std::any_of(control().deferred_copies.begin(), control().deferred_copies.end(), [&](auto &copy) {
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
