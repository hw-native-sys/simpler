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
/*
 * The native-run operation lease and the finalization protocol, driven directly against ChipWorker.
 *
 * These are the properties no hardware run can show, because a run that happens to be fast enough
 * satisfies them by accident: which thread owns a finalization, what a second one is told, whether a
 * completed wait is repeated, and what the worker does with a run whose teardown left an ambiguous
 * tag. The runtime entries are stubs here, so each of those can be produced on demand rather than
 * waited for.
 */
#include <unistd.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <utility>
#include <unordered_map>
#include <vector>

#include <dlfcn.h>

#include <gtest/gtest.h>

#define private public
#define protected public
// ChipWorker holds its run lane by unique_ptr, so the lane's definition has to be visible for the
// worker to be destructible here.
#include "chip_run_lane.h"
#include "chip_worker.h"
#include "runtime_module.h"
#undef private
#undef protected

#include "native_run_storage_tag.h"

namespace {

TEST(NativeRunOperationLeaseTest, RunStorageRetainsItsModuleAcrossMoves) {
    auto module = std::make_shared<RuntimeModule>(nullptr);
    std::weak_ptr<RuntimeModule> weak = module;
    ChipWorker::RuntimeStorage first(64, alignof(std::max_align_t), module);
    void *address = first.data();
    module.reset();
    EXPECT_FALSE(weak.expired());
    ChipWorker::RuntimeStorage second(std::move(first));
    EXPECT_EQ(first.data(), nullptr);
    EXPECT_EQ(second.data(), address);
    EXPECT_FALSE(weak.expired());
    first = std::move(second);
    EXPECT_EQ(second.data(), nullptr);
    EXPECT_EQ(first.data(), address);
    first = ChipWorker::RuntimeStorage{};
    EXPECT_TRUE(weak.expired());
}

// What each stub should do on its next call. Plain globals because one case drives one worker.
std::mutex g_mu;
std::condition_variable g_cv;
uint64_t g_prepare_epoch = 0;
bool g_prepare_entered = false;
bool g_prepare_released = true;
bool g_prepare_in_progress = false;
bool g_finalize_during_prepare = false;
bool g_wait_entered = false;
bool g_wait_released = true;
int g_wait_rc = 0;
bool g_wait_throws = false;
int g_finalize_rc = 0;
uint32_t g_last_use_proof = 0;
bool g_finalize_throws = false;
bool g_finalize_destroys = true;
bool g_finalize_corrupts = false;
int g_wait_calls = 0;
int g_finalize_calls = 0;
int g_poll_rc = SIMPLER_NATIVE_RUN_POLL_NOT_READY;
int g_launch_rc = 0;
// The joined launch and the capability query both dereference two carriers, so both can be held
// open here while another thread tries to destroy one of them.
std::mutex g_join_mu;
std::condition_variable g_join_cv;
bool g_join_entered = false;
bool g_join_released = true;
int g_join_rc = 0;
int g_join_calls = 0;
bool g_query_entered = false;
bool g_query_released = true;
int g_query_calls = 0;
int g_device_finalize_rc = 0;
int g_device_finalize_calls = 0;
int g_destroy_context_calls = 0;
int g_comm_free_calls = 0;
int g_context_rc = 0;
struct StubCarrier {
    uint64_t magic;
    uint64_t epoch;
};
std::unordered_map<uint64_t, int> g_per_run_wait_rc;
std::vector<uint64_t> g_waited_epochs;

int validate_context(void *) { return g_context_rc; }
int g_comm_rc = 0;
int g_comm_stream_free_calls = 0;
int g_comm_stream_rc = 0;
int g_comm_alloc_calls = 0;
int g_comm_abort_calls = 0;

int stub_release_domain(void *, uint64_t, size_t, uint32_t) {
    ++g_comm_free_calls;
    return g_comm_rc;
}
int stub_release_global_domain(uint64_t) {
    ++g_comm_free_calls;
    return 0;
}
int stub_abort_domain(void *, uint64_t, size_t, uint32_t) {
    ++g_comm_abort_calls;
    return g_comm_rc;
}
int stub_destroy_comm(void *) {
    ++g_comm_free_calls;
    return g_comm_rc;
}

int stub_destroy_comm_stream(void *, void *) {
    ++g_comm_stream_free_calls;
    return g_comm_stream_rc;
}

int stub_alloc_domain(
    void *, uint64_t, const uint32_t *, size_t, uint32_t, size_t, uint32_t, uint64_t *, uint64_t *, size_t *
) {
    ++g_comm_alloc_calls;
    return g_comm_rc;
}

int prepare_run(
    void *, void *runtime, int32_t, const void *, const CallConfig *, const NativeRunDescriptor *descriptor
) {
    {
        std::unique_lock<std::mutex> lk(g_mu);
        g_prepare_epoch = descriptor->run_epoch;
        g_prepare_entered = true;
        g_prepare_in_progress = true;
        g_cv.notify_all();
        g_cv.wait(lk, [] {
            return g_prepare_released;
        });
        g_prepare_in_progress = false;
    }
    // The carrier the platform would place here; only its tag matters to the worker.
    const StubCarrier carrier{kNativeRunContextMagic, descriptor->run_epoch};
    std::memcpy(runtime, &carrier, sizeof(carrier));
    return 0;
}

int launch_run(void *, void *) { return g_launch_rc; }

int poll_run(void *, void *) { return g_poll_rc; }

int wait_run(void *, void *runtime) {
    {
        std::unique_lock<std::mutex> lk(g_mu);
        ++g_wait_calls;
        g_wait_entered = true;
        g_cv.notify_all();
        g_cv.wait(lk, [] {
            return g_wait_released;
        });
    }
    if (g_wait_throws) throw std::runtime_error("stub wait failed");
    StubCarrier carrier{};
    std::memcpy(&carrier, runtime, sizeof(carrier));
    std::lock_guard<std::mutex> lk(g_mu);
    g_waited_epochs.push_back(carrier.epoch);
    const auto found = g_per_run_wait_rc.find(carrier.epoch);
    if (found != g_per_run_wait_rc.end()) return found->second;
    if (g_wait_rc == 0) g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    return g_wait_rc;
}

int wait_run_completion(void *, void *) {
    std::unique_lock<std::mutex> lk(g_mu);
    g_wait_entered = true;
    g_cv.notify_all();
    g_cv.wait(lk, [] {
        return g_wait_released;
    });
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    return 0;
}

int finalize_run(void *, void *runtime, int) {
    ++g_finalize_calls;
    if (g_finalize_destroys) {
        uint64_t cleared = 0;
        std::memcpy(runtime, &cleared, sizeof(cleared));
    } else if (g_finalize_corrupts) {
        uint64_t garbage = kNativeRunContextMagic ^ UINT64_C(0xff);
        std::memcpy(runtime, &garbage, sizeof(garbage));
    }
    if (g_finalize_throws) throw std::runtime_error("stub finalize failed");
    return g_finalize_rc;
}

int finalize_run_with_last_use(void *context, void *runtime, int delivery, uint32_t *out_proof) {
    *out_proof = SIMPLER_NATIVE_RUN_LAST_USE_UNKNOWN;
    const int rc = finalize_run(context, runtime, delivery);
    *out_proof = g_last_use_proof;
    return rc;
}

int supports_successor(void *) { return 1; }

int launch_run_joined(void *, void *, void *predecessor_runtime) {
    {
        std::unique_lock<std::mutex> lk(g_join_mu);
        ++g_join_calls;
        g_join_entered = true;
        g_join_cv.notify_all();
        g_join_cv.wait(lk, [] {
            return g_join_released;
        });
    }
    // Read after the release, which is the dereference a destruction underneath this call would
    // have invalidated: the predecessor's carrier must still be the one this join was handed.
    uint64_t tag = 0;
    std::memcpy(&tag, predecessor_runtime, sizeof(tag));
    EXPECT_EQ(tag, kNativeRunContextMagic) << "the predecessor's carrier was destroyed under a joined launch";
    return g_join_rc;
}

int supports_joined_diagnostic_launch(void *, void *, void *predecessor_runtime) {
    {
        std::unique_lock<std::mutex> lk(g_join_mu);
        ++g_query_calls;
        g_query_entered = true;
        g_join_cv.notify_all();
        g_join_cv.wait(lk, [] {
            return g_query_released;
        });
    }
    uint64_t tag = 0;
    std::memcpy(&tag, predecessor_runtime, sizeof(tag));
    EXPECT_EQ(tag, kNativeRunContextMagic) << "the predecessor's carrier was destroyed under a capability query";
    return 1;
}

int stub_finalize_device(void *) {
    std::lock_guard<std::mutex> lk(g_mu);
    g_finalize_during_prepare |= g_prepare_in_progress;
    ++g_device_finalize_calls;
    return g_device_finalize_rc;
}

void stub_destroy_device_context(void *) { ++g_destroy_context_calls; }

void prime(ChipWorker &worker, uint32_t depth = 2) {
    {
        std::lock_guard<std::mutex> lk(g_mu);
        g_prepare_entered = false;
        g_prepare_released = true;
        g_prepare_in_progress = g_finalize_during_prepare = false;
        g_wait_entered = false;
        g_wait_released = true;
    }
    {
        std::lock_guard<std::mutex> lk(g_join_mu);
        g_join_entered = false;
        g_join_released = true;
        g_query_entered = false;
        g_query_released = true;
    }
    g_wait_rc = 0;
    g_wait_throws = false;
    g_finalize_rc = 0;
    g_last_use_proof = SIMPLER_NATIVE_RUN_LAST_USE_UNKNOWN;
    g_finalize_throws = false;
    g_finalize_destroys = true;
    g_finalize_corrupts = false;
    g_wait_calls = 0;
    g_finalize_calls = 0;
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_NOT_READY;
    g_launch_rc = 0;
    g_join_rc = 0;
    g_join_calls = 0;
    g_query_calls = 0;
    g_device_finalize_rc = 0;
    g_device_finalize_calls = 0;
    g_destroy_context_calls = 0;
    g_comm_free_calls = 0;
    g_context_rc = 0;
    g_per_run_wait_rc.clear();
    g_waited_epochs.clear();
    g_comm_rc = 0;
    g_comm_stream_free_calls = 0;
    g_comm_stream_rc = 0;
    g_comm_alloc_calls = 0;
    g_comm_abort_calls = 0;
    worker.initialized_ = true;
    // The stubs ignore it, but teardown refuses to run without one, which is the production
    // guard against closing a worker that never reached a device.
    static int device_context_placeholder = 0;
    worker.device_ctx_ = &device_context_placeholder;
    worker.destroy_device_context_fn_ = stub_destroy_device_context;
    worker.validate_execution_context_fn_ = validate_context;
    // A reference to this executable supplies the runtime owner without another DSO.
    if (worker.runtime_module_ == nullptr) worker.runtime_module_ = std::make_shared<RuntimeModule>(nullptr);
    EXPECT_NE(worker.runtime_module_, nullptr);
    worker.pipeline_contract_ = {PTO_PIPELINE_CONTRACT_ABI_VERSION, 0, depth, {}};
    // One generation per granted set, as `init` sizes it: an unsized filter has no entry for
    // slot 0, and `is_admissible` refuses a lease outside it, so every preparation would be
    // refused as stale before reaching the stubs below.
    worker.pipeline_generations_.resize(depth);
    for (uint32_t slot = 0; slot < depth; ++slot) {
        worker.runtime_bufs_.emplace_back(64, alignof(std::max_align_t), worker.runtime_module_);
    }
    worker.prepare_run_fn_ = prepare_run;
    worker.launch_run_fn_ = launch_run;
    worker.poll_run_fn_ = poll_run;
    worker.wait_run_fn_ = wait_run;
    worker.wait_run_completion_fn_ = wait_run_completion;
    worker.finalize_run_fn_ = finalize_run;
    worker.supports_concurrent_native_prepare_fn_ = supports_successor;
}

// What the joined launch and the diagnostic capability query additionally need: both are gated on
// a launch depth above one and on the backend answering that it can order two runs.
void prime_joined(ChipWorker &worker) {
    worker.launch_depth_ = 2;
    worker.launch_run_joined_fn_ = launch_run_joined;
    worker.supports_joined_native_launch_fn_ = supports_successor;
    worker.supports_joined_diagnostic_launch_fn_ = supports_joined_diagnostic_launch;
    worker.retained_epoch_limit_granted_ = 3;
}

NativeRunDomainProvider g_domain_provider{};
CommDomainSnapshot g_domain_answer{};
uint32_t g_domain_qualification = 0;
int g_domain_backend_queries = 0;

int domain_snapshot_backend(CommHandle handle, uint64_t id, CommDomainSnapshot *out) {
    ++g_domain_backend_queries;
    *out = {};
    if (handle == nullptr || id != 31) return -1;
    out->use = {CommDomainKind::LOCAL, 0, 31, 0x1000};
    out->context.rankNum = 1;
    out->context.winSize = 4096;
    out->context.windowsIn[0] = 0x2000;
    return 0;
}

int prepare_domain_run(
    void *device, void *runtime, int32_t callable, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor, size_t descriptor_bytes, const NativeRunDomainProvider *provider,
    size_t provider_bytes
) {
    EXPECT_EQ(descriptor_bytes, sizeof(NativeRunDescriptor));
    EXPECT_EQ(provider_bytes, sizeof(NativeRunDomainProvider));
    EXPECT_EQ(provider->version, 1u);
    EXPECT_EQ(provider->count, 1u);
    g_domain_provider = *provider;
    const int rc = provider->query(
        provider->owner, descriptor->run_epoch, descriptor->pipeline_slot, 0, &g_domain_answer, &g_domain_qualification
    );
    if (rc != 0) {
        publish_released_preparation(runtime, descriptor->run_epoch);
        return rc;
    }
    return prepare_run(device, runtime, callable, args, config, descriptor);
}

TEST(NativeRunOperationLeaseTest, ActualLaneDomainProviderUsesTheCurrentOwnerAndBlocksRelease) {
    ChipWorker worker;
    prime(worker);
    worker.execution_environment_ = ChipWorker::ExecutionEnvironment{static_cast<uint64_t>(getpid()), 1, 1, 0, false};
    worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
    worker.prepare_run_with_domains_fn_ = prepare_domain_run;
    worker.comm_query_domain_snapshot_fn_ = domain_snapshot_backend;
    worker.comm_release_domain_windows_fn_ = stub_release_domain;
    worker.comm_destroy_fn_ = stub_destroy_comm;
    static int comm = 0;
    ASSERT_NE(worker.create_comm_session(&comm, nullptr, true), nullptr);
    worker.base_comm_handle_ = reinterpret_cast<uint64_t>(&comm);
    g_domain_backend_queries = 0;
    g_domain_provider = {};
    g_domain_answer = {};
    g_domain_qualification = 0;
    ChipStorageTaskArgs args;
    const auto run = worker.submit_chip_run(
        1, args, CallConfig{}, PipelineSlotLease{0, 0, 1}, 8, 9, nullptr, 0, true, {},
        {{CommDomainKind::LOCAL, 0, 31, 0x1000}}
    );
    EXPECT_EQ(g_domain_backend_queries, 1);
    EXPECT_EQ(g_domain_answer.use.id, 31u);
    EXPECT_EQ(g_domain_qualification, static_cast<uint32_t>(CommDomainExecution::EXCLUSIVE_RUN));
    EXPECT_THROW(worker.comm_release_domain_windows(worker.base_comm_handle_, 31, 1, 0), std::runtime_error);
    EXPECT_EQ(g_comm_free_calls, 0);
    CommDomainSnapshot stale;
    uint32_t qualification = 99;
    ASSERT_NE(g_domain_provider.query, nullptr);
    EXPECT_NE(g_domain_provider.query(&worker, g_prepare_epoch, 0, 0, &stale, &qualification), 0);
    EXPECT_EQ(qualification, 0u);  // The owner exists, but preparation has ended.
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    auto completed = run;
    EXPECT_TRUE(completed.wait_until(ChipRun::Clock::now() + std::chrono::seconds(1)));
    EXPECT_NE(g_domain_provider.query(&worker, g_prepare_epoch, 0, 0, &stale, &qualification), 0);
    EXPECT_EQ(stale.use.device_context, 0u);
    EXPECT_NO_THROW(worker.comm_release_domain_windows(worker.base_comm_handle_, 31, 1, 0));
    EXPECT_EQ(g_comm_free_calls, 1);
    EXPECT_NO_THROW(worker.comm_destroy_all());
}

NativeRunDomainProviderV2 g_sdma_provider{};
CommSdmaOwnerInfoV1 g_sdma_answer{};

int sdma_domain_snapshot_backend(CommHandle owner, uint64_t id, CommDomainSnapshot *out) {
    const int rc = domain_snapshot_backend(owner, id, out);
    if (rc == 0) set_comm_workspace(out->context, CommWorkspaceKind::SDMA, 0x8000, 4096);
    return rc;
}
int sdma_owner_backend(CommHandle owner, uint64_t id, CommSdmaOwnerInfoV1 *out, size_t bytes) {
    if (owner == nullptr || id != 31 || bytes != sizeof(*out)) return -1;
    *out = {0x8000, 4096, CommSdmaAccess::SDK_ONLY, 3};
    return 0;
}
int prepare_sdma_domain_run(
    void *device, void *runtime, int32_t callable, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor, size_t descriptor_bytes, const NativeRunDomainProviderV2 *provider,
    size_t provider_bytes
) {
    EXPECT_EQ(descriptor_bytes, sizeof(NativeRunDescriptor));
    EXPECT_EQ(provider_bytes, sizeof(NativeRunDomainProviderV2));
    EXPECT_EQ(provider->base.version, 2u);
    g_sdma_provider = *provider;
    const auto &base = provider->base;
    CommDomainSnapshot snapshot;
    uint32_t execution = 0;
    int rc = base.query(base.owner, descriptor->run_epoch, descriptor->pipeline_slot, 0, &snapshot, &execution);
    if (rc == 0)
        rc = provider->query_sdma_owner(
            base.owner, descriptor->run_epoch, descriptor->pipeline_slot, 0, &g_sdma_answer, sizeof(g_sdma_answer)
        );
    CommSdmaOwnerInfoV1 wrong{1, 1, CommSdmaAccess::SDK_ONLY, 1};
    EXPECT_NE(
        provider->query_sdma_owner(
            base.owner, descriptor->run_epoch, descriptor->pipeline_slot + 1, 0, &wrong, sizeof(wrong)
        ),
        0
    );
    EXPECT_EQ(wrong.workspace_addr, 0u);
    EXPECT_NE(
        provider->query_sdma_owner(
            base.owner, descriptor->run_epoch, descriptor->pipeline_slot, 1, &wrong, sizeof(wrong)
        ),
        0
    );
    if (rc != 0) {
        publish_released_preparation(runtime, descriptor->run_epoch);
        return rc;
    }
    return prepare_run(device, runtime, callable, args, config, descriptor);
}

TEST(NativeRunOperationLeaseTest, ActualLaneV2QueriesOnlyItsSelectedLiveOwnerDuringPrepare) {
    for (bool supported : {false, true}) {
        ChipWorker worker;
        prime(worker);
        worker.execution_environment_ =
            ChipWorker::ExecutionEnvironment{static_cast<uint64_t>(getpid()), 1, 1, 0, false};
        worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
        worker.prepare_run_with_domains_fn_ = prepare_domain_run;
        worker.prepare_run_with_domains_v2_fn_ = prepare_sdma_domain_run;
        worker.comm_query_domain_snapshot_fn_ = sdma_domain_snapshot_backend;
        worker.comm_query_domain_sdma_owner_fn_ = supported ? sdma_owner_backend : nullptr;
        worker.comm_release_domain_windows_fn_ = stub_release_domain;
        worker.comm_destroy_fn_ = stub_destroy_comm;
        static int comm = 0;
        ASSERT_NE(worker.create_comm_session(&comm, nullptr, true), nullptr);
        worker.base_comm_handle_ = reinterpret_cast<uint64_t>(&comm);
        g_sdma_answer = {};
        g_sdma_provider = {};
        ChipStorageTaskArgs args;
        auto run = worker.submit_chip_run(
            1, args, CallConfig{}, PipelineSlotLease{0, 0, 1}, 8, 9, nullptr, 0, true, {},
            {{CommDomainKind::LOCAL, 0, 31, 0x1000}}
        );
        ASSERT_NE(g_sdma_provider.query_sdma_owner, nullptr);
        EXPECT_EQ(g_sdma_answer.access, supported ? CommSdmaAccess::SDK_ONLY : CommSdmaAccess::UNKNOWN);
        EXPECT_EQ(g_sdma_answer.queue_num, supported ? 3u : 0u);
        CommSdmaOwnerInfoV1 stale{1, 1, CommSdmaAccess::SDK_ONLY, 1};
        EXPECT_NE(g_sdma_provider.query_sdma_owner(&worker, g_prepare_epoch, 0, 0, &stale, sizeof(stale)), 0);
        EXPECT_EQ(stale.workspace_addr, 0u);
        EXPECT_THROW(worker.comm_release_domain_windows(worker.base_comm_handle_, 31, 1, 0), std::runtime_error);
        g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        EXPECT_TRUE(run.wait_until(ChipRun::Clock::now() + std::chrono::seconds(1)));
        EXPECT_NE(g_sdma_provider.query_sdma_owner(&worker, g_prepare_epoch, 0, 0, &stale, sizeof(stale)), 0);
        EXPECT_NO_THROW(worker.comm_release_domain_windows(worker.base_comm_handle_, 31, 1, 0));
        EXPECT_NO_THROW(worker.comm_destroy_all());
    }
}

TEST(NativeRunOperationLeaseTest, ManagedAllocationNeverFallsBackToTheLegacySymbol) {
    ChipWorker worker;
    prime(worker);
    worker.comm_alloc_domain_windows_fn_ = stub_alloc_domain;
    EXPECT_THROW(worker.comm_alloc_domain_windows_sdma_sdk_v1(1, 1, {0, 1}, 0, 4096, 3), std::runtime_error);
    EXPECT_EQ(g_comm_alloc_calls, 0);
}

TEST(NativeRunOperationLeaseTest, SelectedDomainsCannotFallBackToAnOldPrepareEntry) {
    ChipWorker worker;
    prime(worker);
    ChipStorageTaskArgs args;
    EXPECT_THROW(
        worker.prepare_native_run_for_lane(
            1, &args, CallConfig{}, PipelineSlotLease{0, 0, 1}, 1, 1, nullptr, 0, true, {},
            {{CommDomainKind::LOCAL, 0, 31, 0x1000}}
        ),
        RunPreparationRefused
    );
    EXPECT_FALSE(g_prepare_entered);
    EXPECT_EQ(worker.unproven_native_run_count(), 0u);
}

ChipWorkerNativeRun prepare(ChipWorker &worker, uint32_t slot, uint64_t generation) {
    ChipStorageTaskArgs args{};
    return worker.prepare_native_run_on_slot(
        1, &args, CallConfig{}, slot, generation, generation, generation, nullptr, 0, false, NativeRunProvenance{}
    );
}

void await_wait_entered() {
    std::unique_lock<std::mutex> lk(g_mu);
    ASSERT_TRUE(g_cv.wait_for(lk, std::chrono::seconds(5), [] {
        return g_wait_entered;
    }));
}

void release_wait() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_wait_released = true;
    g_cv.notify_all();
}

void hold_wait() {
    std::lock_guard<std::mutex> lk(g_mu);
    g_wait_released = false;
    g_wait_entered = false;
}

void hold_join() {
    std::lock_guard<std::mutex> lk(g_join_mu);
    g_join_released = false;
    g_join_entered = false;
    g_query_released = false;
    g_query_entered = false;
}

void await_join_entered() {
    std::unique_lock<std::mutex> lk(g_join_mu);
    ASSERT_TRUE(g_join_cv.wait_for(lk, std::chrono::seconds(5), [] {
        return g_join_entered;
    }));
}

void await_query_entered() {
    std::unique_lock<std::mutex> lk(g_join_mu);
    ASSERT_TRUE(g_join_cv.wait_for(lk, std::chrono::seconds(5), [] {
        return g_query_entered;
    }));
}

void release_join() {
    std::lock_guard<std::mutex> lk(g_join_mu);
    g_join_released = true;
    g_query_released = true;
    g_join_cv.notify_all();
}

/**
 * A parked waiter thread that is always released and joined, however the case ends.
 *
 * A case that holds a waiter inside the stub's `wait` and then fails a fatal assertion returns
 * from its body with that thread still joinable, which calls `std::terminate` and destroys the
 * finding it was about to report. An exception thrown by anything between the two — a refused
 * preparation, for one — does the same. Both are cleanup, so both belong in a destructor.
 *
 * The waiter's own exception is carried out rather than escaping its thread, and `rethrow` is
 * what a case calls to make it the case's failure.
 */
class ParkedWaiter {
public:
    template <typename Fn>
    explicit ParkedWaiter(Fn &&body) {
        // Started in the body, not in a member initializer. A thread launched from an initializer
        // begins running while the members declared after it are still unconstructed, and the
        // very first thing this one may do is assign `failure_` — so the only safe order is every
        // member first, the thread last.
        thread_ = std::thread([this, body = std::forward<Fn>(body)] {
            try {
                body();
            } catch (...) {
                failure_ = std::current_exception();
            }
        });
    }

    ParkedWaiter(const ParkedWaiter &) = delete;
    ParkedWaiter &operator=(const ParkedWaiter &) = delete;

    ~ParkedWaiter() { settle(); }

    /** Release the stub and join, once. Safe to call again from the destructor. */
    void settle() {
        release_wait();
        if (thread_.joinable()) thread_.join();
    }

    /** The waiter's own exception, if it had one. Call after `settle`: the join is what makes
     *  the thread's write to it visible here. */
    void rethrow() const {
        if (failure_) std::rethrow_exception(failure_);
    }

private:
    std::exception_ptr failure_;
    std::thread thread_;
};

}  // namespace

TEST(NativeRunOperationLeaseTest, CompletedPollDoesNotReplaceTheExplicitObserver) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    EXPECT_TRUE(worker.poll_native_run(run));
    EXPECT_TRUE(worker.poll_native_run(run));
    EXPECT_EQ(g_wait_calls, 0);
    EXPECT_EQ(g_finalize_calls, 0);

    g_wait_rc = -731;
    worker.wait_native_run(run);
    EXPECT_EQ(g_wait_calls, 1);
    g_wait_rc = -732;
    worker.wait_native_run(run);
    EXPECT_EQ(g_wait_calls, 1);
    try {
        worker.finalize_native_run(run);
        FAIL() << "the explicit observer's failure was replaced by a completed poll";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-731"), std::string::npos);
    }
    EXPECT_EQ(g_wait_calls, 1);
    EXPECT_EQ(g_finalize_calls, 1);
}

TEST(NativeRunOperationLeaseTest, CloseBeforeWaitObservesEvenAfterCompletedPoll) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    ASSERT_TRUE(worker.poll_native_run(run));
    g_wait_rc = -733;
    try {
        worker.finalize_native_run(run);
        FAIL() << "closing an unobserved call omitted the explicit observer";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-733"), std::string::npos);
    }
    EXPECT_EQ(g_wait_calls, 1);
    EXPECT_EQ(g_finalize_calls, 1);
}

TEST(NativeRunOperationLeaseTest, CompletedPollKeepsAnUnobservedRunRetryableAfterContextRefusal) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    ASSERT_TRUE(worker.poll_native_run(run));
    g_context_rc = PTO_RUNTIME_ERR_INVALID_STATE;
    EXPECT_THROW(worker.wait_native_run(run), ExecutionContextRefused);
    EXPECT_THROW(worker.finalize_native_run(run), ExecutionContextRefused);
    EXPECT_EQ(g_wait_calls, 0);
    EXPECT_EQ(g_finalize_calls, 0);
    g_context_rc = 0;
    g_wait_rc = -734;
    worker.wait_native_run(run);
    EXPECT_EQ(g_wait_calls, 1);
    try {
        worker.finalize_native_run(run);
        FAIL() << "context refusal lost the pending explicit observation";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-734"), std::string::npos);
    }
    EXPECT_EQ(g_wait_calls, 1);
}

TEST(NativeRunOperationLeaseTest, SlotReuseKeepsEachUnobservedCallAndDoesNotForceAnObserver) {
    ChipWorker worker;
    prime(worker);
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    std::vector<ChipWorkerNativeRun> runs;
    for (uint64_t generation = 1; generation <= 6; ++generation) {
        std::optional<ChipWorkerNativeRun> current;
        EXPECT_NO_THROW(current = prepare(worker, (generation - 1) % 2, generation));
        if (!current) break;
        runs.push_back(*current);
        g_per_run_wait_rc[current->run_epoch] = -740 - static_cast<int>(generation);
        worker.launch_native_run(*current);
        ASSERT_TRUE(worker.poll_native_run(*current));
        EXPECT_EQ(g_wait_calls, 0);
        EXPECT_EQ(g_finalize_calls, 0);
    }
    ASSERT_EQ(runs.size(), 6u) << "unobserved calls are not executing occupants of the pipeline slots";
    for (auto it = runs.rbegin(); it != runs.rend(); ++it) {
        const int expected = g_per_run_wait_rc.at(it->run_epoch);
        EXPECT_TRUE(worker.poll_native_run(*it));
        worker.wait_native_run(*it);
        worker.wait_native_run(*it);
        try {
            worker.finalize_native_run(*it);
            FAIL() << "the old handle lost its own observer error";
        } catch (const std::runtime_error &error) {
            EXPECT_NE(std::string(error.what()).find(std::to_string(expected)), std::string::npos);
        }
    }
    EXPECT_EQ(g_wait_calls, 6);
    EXPECT_EQ(g_finalize_calls, 6);
    std::vector<uint64_t> expected_epochs;
    for (auto it = runs.rbegin(); it != runs.rend(); ++it)
        expected_epochs.push_back(it->run_epoch);
    EXPECT_EQ(g_waited_epochs, expected_epochs);
}

TEST(NativeRunOperationLeaseTest, ObserverErrorDoesNotProvePhysicalCompletionOrReleaseTheSlot) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_NOT_READY;
    g_wait_rc = -751;
    worker.wait_native_run(run);
    EXPECT_EQ(g_wait_calls, 1);
    EXPECT_FALSE(worker.poll_native_run(run)) << "an observer error does not prove that device readers stopped";
    EXPECT_THROW(prepare(worker, 0, 2), RunPreparationRefused);
    EXPECT_EQ(g_finalize_calls, 0);
    g_finalize_destroys = false;
    g_finalize_rc = -752;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_THROW(prepare(worker, 0, 3), RunPreparationRefused);
    EXPECT_EQ(g_wait_calls, 1);
}

// A refused finalization must not undo the completion its own wait established, or the next
// attempt waits again and reports whatever that second wait says instead of the first error.
TEST(NativeRunOperationLeaseTest, RefusedFinalizeKeepsItsWaitOutcomeAndDoesNotWaitTwice) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    // The teardown refuses without destroying the carrier, which is the context-refusal shape.
    g_wait_rc = -7;
    g_finalize_rc = -5;
    g_finalize_destroys = false;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 1);
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_EQ(it->second->phase, ChipWorker::NativeRunPhase::LAUNCHED);
        EXPECT_TRUE(it->second->result_observed);
        EXPECT_TRUE(it->second->pending_rc_set);
        EXPECT_EQ(it->second->pending_rc, -7);
        EXPECT_FALSE(it->second->finalize_reserved);
    }

    // The retry succeeds at the teardown and reports the error the first wait established.
    g_finalize_rc = 0;
    g_finalize_destroys = true;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 1) << "the completed wait was repeated";
    EXPECT_EQ(g_finalize_calls, 2);
}

// Two finalizations of one run: one owns it, the other is told so and never reaches the runtime.
TEST(NativeRunOperationLeaseTest, SecondFinalizerIsRefusedAndNeverEntersTheRuntime) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    hold_wait();
    std::atomic<bool> first_threw{false};
    std::thread owner([&] {
        try {
            worker.finalize_native_run(run);
        } catch (...) {
            first_threw = true;
        }
    });
    await_wait_entered();

    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_finalize_calls, 0) << "the refused finalizer reached the runtime";

    release_wait();
    owner.join();
    EXPECT_FALSE(first_threw.load());
    EXPECT_EQ(g_finalize_calls, 1);
    EXPECT_EQ(g_wait_calls, 1);
}

// A poll while another operation holds the lease reports no progress rather than entering
// concurrently or blocking; a wait on the same run waits for the lease instead.
TEST(NativeRunOperationLeaseTest, PollReportsNoProgressWhileAnotherOperationHoldsTheLease) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    hold_wait();
    std::thread waiter([&] {
        worker.wait_native_run(run);
    });
    await_wait_entered();

    EXPECT_FALSE(worker.poll_native_run(run));

    release_wait();
    waiter.join();
    EXPECT_EQ(g_wait_calls, 1);
    // Completion is established, so a later poll answers from the record without a device call.
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_ERROR;
    EXPECT_TRUE(worker.poll_native_run(run));
}

// §2.3.6(f) / contract row 24: later admissions grow the run table while a waiter sleeps inside
// the runtime, and the waiter is unaffected.
//
// This is a property of *where* the entry lives, so the case makes the rehash actually happen
// rather than arguing that it would be harmless. `native_runs_` maps epoch to a `unique_ptr`, and
// a waiter holds the pointee for the whole of its wait; admitting enough later runs forces the
// container to grow and relocate every node. The assertion is that the address the waiter is
// holding is still the entry its epoch resolves to afterwards, and that the wait it was in the
// middle of still returns its own outcome.
//
// The waiter is parked inside the backend's `wait` for the whole window, which is the state the
// row is about — a waiter between the lease and the completion, not one that happens to be
// scheduled elsewhere. The admissions are made from this thread, which does not weaken that: the
// waiter is genuinely inside the runtime for every one of them.
//
// `ParkedWaiter` owns the release and the join, so a fatal assertion or a throwing preparation
// below still unparks the thread instead of reaching the end of the case with it joinable.
TEST(NativeRunOperationLeaseTest, AdmittingLaterRunsRehashesTheTableUnderASleepingWaiter) {
    // One slot per run this case admits, since a preparation is refused on a slot whose
    // generation filter has no entry for it.
    constexpr uint32_t kDepth = 24;
    ChipWorker worker;
    prime(worker, kDepth);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    hold_wait();
    // Set before the waiter is released, which is anywhere from here on: the stub reads it on the
    // way out of its park.
    g_wait_rc = -7;
    ParkedWaiter waiter([&] {
        worker.wait_native_run(run);
    });
    await_wait_entered();

    const ChipWorker::NativeRunEntry *entry_before = nullptr;
    size_t buckets_before = 0;
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        entry_before = it->second.get();
        buckets_before = worker.native_runs_.bucket_count();
    }

    // Enough later admissions that the table must grow: libstdc++ keeps its load factor at one,
    // so a bucket count that never moved would mean no rehash happened and this case proved
    // nothing.
    //
    // Each one is launched as well as prepared, because that is the lifecycle the product
    // requires: `prepare_native_run_on_slot` admits a successor only while every run already in
    // the table is `LAUNCHED` or `REAPED`, so a queue of merely-prepared runs is refused at the
    // second one. Launching each is what makes these admissions real rather than a fixture that
    // the admission check would have to be relaxed to accept.
    std::vector<ChipWorkerNativeRun> later;
    for (uint32_t slot = 1; slot < kDepth; ++slot) {
        later.push_back(prepare(worker, slot, slot + 1));
        worker.launch_native_run(later.back());
    }
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        ASSERT_GT(worker.native_runs_.bucket_count(), buckets_before)
            << "the run table never rehashed, so this case did not reach the state row 24 is about";
        ASSERT_EQ(worker.native_runs_.size(), later.size() + 1);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end()) << "the sleeping waiter's run left the table";
        EXPECT_EQ(it->second.get(), entry_before)
            << "a rehash moved the entry itself, so every waiter holding one is holding freed storage";
        EXPECT_EQ(it->second->ops_in_flight, 1u) << "the waiter's lease was lost across the rehash";
    }

    waiter.settle();
    ASSERT_NO_THROW(waiter.rethrow()) << "the parked waiter failed rather than returning its own outcome";
    EXPECT_EQ(g_wait_calls, 1) << "the waiter was made to wait a second time";
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_EQ(it->second.get(), entry_before);
        EXPECT_EQ(it->second->phase, ChipWorker::NativeRunPhase::LAUNCHED);
        EXPECT_TRUE(it->second->result_observed) << "the waiter's own observation did not land on its own entry";
        EXPECT_TRUE(it->second->pending_rc_set);
        EXPECT_EQ(it->second->pending_rc, -7) << "the waiter's own outcome was not the one it recorded";
        EXPECT_EQ(it->second->ops_in_flight, 0u);
    }

    // The launched runs behind it are finalized here rather than left to teardown, so the
    // destructor has nothing to report and this case's own finding is the only thing in the log.
    // Their waits are the stub's, and the outcome under test has already been read above.
    g_wait_rc = 0;
    for (const ChipWorkerNativeRun &launched : later) {
        EXPECT_NO_THROW(worker.finalize_native_run(launched)) << "a later admission could not be retired";
    }
}

// A joined launch dereferences the predecessor's carrier too, so a finalization of that
// predecessor must wait for the launch to return rather than destroying it underneath. The join is
// held open inside the runtime while the other thread tries exactly that.
TEST(NativeRunOperationLeaseTest, JoinedLaunchHoldsThePredecessorAgainstConcurrentFinalize) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    hold_join();
    std::atomic<bool> joined{false};
    std::thread launcher([&] {
        joined = worker.launch_native_run_joined(behind, ahead);
    });
    await_join_entered();

    // The predecessor's finalization has to drain this join first. It is started here and must
    // still be outside the runtime when the join is released.
    std::atomic<bool> finalize_returned{false};
    std::thread closer([&] {
        try {
            worker.finalize_native_run(ahead);
        } catch (...) {}
        finalize_returned = true;
    });
    // Nothing proves a negative by waiting forever, so this waits long enough that a finalization
    // free to proceed would have destroyed the carrier, then checks it did not.
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(g_finalize_calls, 0) << "a finalization entered the runtime under a live joined launch";
    EXPECT_FALSE(finalize_returned.load());

    release_join();
    launcher.join();
    closer.join();
    EXPECT_TRUE(joined.load());
    EXPECT_EQ(g_join_calls, 1);
    EXPECT_EQ(g_finalize_calls, 1) << "the drained finalization never ran";
}

// A predecessor that is no longer executing is a decline, not a failure, and never reaches the
// backend.
TEST(NativeRunOperationLeaseTest, JoinedLaunchDeclinesARetiredPredecessor) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    worker.note_native_run_completion(ahead.run_epoch, 0);
    EXPECT_FALSE(worker.launch_native_run_joined(behind, ahead));
    EXPECT_EQ(g_join_calls, 0) << "a declined join still entered the backend";
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(behind.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_EQ(it->second->phase, ChipWorker::NativeRunPhase::PREPARED) << "a decline changed the run's state";
        EXPECT_EQ(it->second->ops_in_flight, 0u);
    }
}

// The diagnostic capability query dereferences both carriers, so it holds both leases for the
// duration of the backend call — a finalization cannot destroy either one under it.
TEST(NativeRunOperationLeaseTest, CapabilityQueryHoldsBothCarriersAgainstConcurrentFinalize) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    hold_join();
    std::atomic<bool> answered{false};
    std::thread asker([&] {
        answered = worker.supports_joined_diagnostic_launch(behind, ahead);
    });
    await_query_entered();

    std::atomic<bool> finalize_returned{false};
    std::thread closer([&] {
        try {
            worker.finalize_native_run(ahead);
        } catch (...) {}
        finalize_returned = true;
    });
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    EXPECT_EQ(g_finalize_calls, 0) << "a finalization entered the runtime under a live capability query";
    EXPECT_FALSE(finalize_returned.load());

    release_join();
    asker.join();
    closer.join();
    EXPECT_TRUE(answered.load());
    EXPECT_EQ(g_query_calls, 1);
    EXPECT_EQ(g_finalize_calls, 1);
}

// The query is a scheduling question asked once a round: a run another operation is inside is a
// refusal for this round, never a wait, and never a backend call.
TEST(NativeRunOperationLeaseTest, CapabilityQueryRefusesRatherThanWaiting) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    hold_wait();
    std::thread waiter([&] {
        worker.wait_native_run(ahead);
    });
    await_wait_entered();

    const auto asked_at = std::chrono::steady_clock::now();
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(behind, ahead));
    EXPECT_LT(std::chrono::steady_clock::now() - asked_at, std::chrono::seconds(1)) << "the query waited for a lease";
    EXPECT_EQ(g_query_calls, 0) << "a refused query still entered the backend";

    release_wait();
    waiter.join();
}

// Stale identity is refused by the same validation every operation uses, so a recycled slot or a
// reused epoch cannot be handed to the backend as if it named this run.
TEST(NativeRunOperationLeaseTest, CapabilityQueryRefusesStaleIdentity) {
    ChipWorker worker;
    prime(worker);
    prime_joined(worker);
    const ChipWorkerNativeRun ahead = prepare(worker, 0, 1);
    worker.launch_native_run(ahead);
    const ChipWorkerNativeRun behind = prepare(worker, 1, 2);

    ChipWorkerNativeRun stale_generation = behind;
    stale_generation.generation = behind.generation + 1;
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(stale_generation, ahead));

    ChipWorkerNativeRun stale_slot = behind;
    stale_slot.slot_id = ahead.slot_id;
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(stale_slot, ahead));

    ChipWorkerNativeRun unknown_epoch = behind;
    unknown_epoch.run_epoch = behind.run_epoch + 1000;
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(unknown_epoch, ahead));

    ChipWorkerNativeRun predecessor_stale = ahead;
    predecessor_stale.generation = ahead.generation + 1;
    EXPECT_FALSE(worker.supports_joined_diagnostic_launch(behind, predecessor_stale));

    EXPECT_EQ(g_query_calls, 0) << "a stale pair reached the backend";
    // The same pair, correctly named, is admitted — so the refusals above are about identity and
    // not about the fixture simply never being able to answer true.
    EXPECT_TRUE(worker.supports_joined_diagnostic_launch(behind, ahead));
    EXPECT_EQ(g_query_calls, 1);
}

// A teardown that leaves neither a destroyed nor a canonical tag is isolated, and a success is
// never reported for it.
TEST(NativeRunOperationLeaseTest, UnknownTeardownIsIsolatedAndNeverReportsSuccess) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    g_finalize_destroys = false;
    g_finalize_corrupts = true;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_TRUE(it->second->destruction_unknown);
    }
    // Nothing further reaches an isolated run, including another finalization.
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_finalize_calls, 1);
}

// A teardown that returns success without destroying the carrier is self-contradictory: it is
// reported as a cleanup failure and the run is isolated rather than offered for another attempt.
TEST(NativeRunOperationLeaseTest, NonDestroyingSuccessIsACleanupFailure) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);

    g_finalize_destroys = false;
    g_finalize_rc = 0;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    std::lock_guard<std::mutex> lk(worker.native_run_mu_);
    const auto it = worker.native_runs_.find(run.run_epoch);
    ASSERT_NE(it, worker.native_runs_.end());
    EXPECT_TRUE(it->second->destruction_unknown);
}

// A wait that throws never reaches the teardown, so the carrier is intact and the run may be
// finalized again.
TEST(NativeRunOperationLeaseTest, WaitThrowLeavesTheRunFinalizableAgain) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    g_wait_throws = true;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_finalize_calls, 0) << "the teardown ran after its wait threw";
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_FALSE(it->second->destruction_unknown);
        EXPECT_FALSE(it->second->finalize_reserved);
    }

    g_wait_throws = false;
    EXPECT_NO_THROW(worker.finalize_native_run(run));
    EXPECT_EQ(g_finalize_calls, 1);
}

// Teardown closes every run it can and isolates the rest, without letting a stub's exception
// leave a noexcept path and without leaving a lease behind that would block it.
TEST(NativeRunOperationLeaseTest, TeardownClosesEveryRunAndSwallowsForeignExceptions) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun first = prepare(worker, 0, 1);
    worker.launch_native_run(first);
    const ChipWorkerNativeRun second = prepare(worker, 1, 2);

    g_finalize_throws = true;
    g_finalize_destroys = false;
    EXPECT_NO_THROW(worker.cleanup_native_runs_noexcept());
    EXPECT_EQ(g_finalize_calls, 2) << "teardown skipped a run it owned";
    std::lock_guard<std::mutex> lk(worker.native_run_mu_);
    for (const uint64_t run_epoch : {first.run_epoch, second.run_epoch}) {
        const auto it = worker.native_runs_.find(run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_TRUE(it->second->destruction_unknown);
        EXPECT_FALSE(it->second->finalize_reserved);
        EXPECT_EQ(it->second->ops_in_flight, 0u);
    }
}

TEST(NativeRunOperationLeaseTest, WholeCloseCannotDetachAHostPreparationInProgress) {
    EXPECT_EXIT(
        {
            ChipWorker worker;
            prime(worker);
            worker.finalize_device_fn_ = stub_finalize_device;
            worker.destroy_device_context_fn_ = stub_destroy_device_context;
            {
                std::lock_guard<std::mutex> lk(g_mu);
                g_prepare_released = false;
            }
            std::thread preparer([&] {
                try {
                    (void)prepare(worker, 0, 1);
                } catch (...) {}
            });
            uint64_t preparing_epoch = 0;
            {
                std::unique_lock<std::mutex> lk(g_mu);
                if (!g_cv.wait_for(lk, std::chrono::seconds(5), [] {
                        return g_prepare_entered;
                    }))
                    _exit(2);
                preparing_epoch = g_prepare_epoch;
            }
            bool close_started = false;
            bool close_finished = false;
            std::thread closer([&] {
                {
                    std::lock_guard<std::mutex> lk(g_mu);
                    close_started = true;
                    g_cv.notify_all();
                }
                try {
                    worker.finalize();
                } catch (...) {}
                {
                    std::lock_guard<std::mutex> lk(g_mu);
                    close_finished = true;
                    g_cv.notify_all();
                }
            });
            {
                std::unique_lock<std::mutex> lk(g_mu);
                g_cv.wait(lk, [&] {
                    return close_started;
                });
                (void)g_cv.wait_for(lk, std::chrono::milliseconds(200), [&] {
                    return close_finished;
                });
            }
            bool retained = false;
            {
                std::lock_guard<std::mutex> lk(worker.native_run_mu_);
                const auto found = worker.native_runs_.find(preparing_epoch);
                retained = found != worker.native_runs_.end() && found->second->storage.data() != nullptr;
            }
            {
                std::lock_guard<std::mutex> lk(g_mu);
                g_prepare_released = true;
                g_cv.notify_all();
            }
            preparer.join();
            closer.join();
            if (!retained) _exit(3);
            if (g_finalize_during_prepare) _exit(4);
            _exit(0);
        },
        ::testing::ExitedWithCode(0), ""
    );
}

// The whole close, not just its sweep: a worker whose every run closed cleanly reaches the device
// teardown and the context destruction, and gives up its run storage.
TEST(NativeRunOperationLeaseTest, WholeCloseReleasesEverythingWhenEveryRunIsProvedDestroyed) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.destroy_device_context_fn_ = stub_destroy_device_context;
    const ChipWorkerNativeRun first = prepare(worker, 0, 1);
    worker.launch_native_run(first);
    const ChipWorkerNativeRun second = prepare(worker, 1, 2);
    (void)second;

    EXPECT_NO_THROW(worker.finalize());
    EXPECT_EQ(g_finalize_calls, 2);
    EXPECT_EQ(g_device_finalize_calls, 1);
    EXPECT_EQ(g_destroy_context_calls, 1);
    EXPECT_TRUE(worker.runtime_bufs_.empty());
    std::lock_guard<std::mutex> lk(worker.native_run_mu_);
    EXPECT_TRUE(worker.native_runs_.empty());
}

TEST(NativeRunOperationLeaseTest, UnknownRunWithCommunicationCannotUseLocalResetAsPeerDrain) {
    for (bool communication_owner : {false, true}) {
        EXPECT_EXIT(
            {
                ChipWorker worker;
                prime(worker);
                worker.finalize_device_fn_ = stub_finalize_device;
                worker.destroy_device_context_fn_ = stub_destroy_device_context;
                worker.comm_destroy_fn_ = stub_destroy_comm;
                static int comm = 0;
                if (communication_owner && worker.create_comm_session(&comm, nullptr, true) == nullptr) _exit(2);
                const auto run = prepare(worker, 0, 1);
                worker.launch_native_run(run);
                g_wait_rc = -23;
                g_finalize_rc = -23;
                g_finalize_destroys = false;
                bool refused = false;
                try {
                    worker.finalize();
                } catch (const std::exception &) {
                    refused = true;
                }
                if (communication_owner) {
                    if (!refused || !worker.comm_cleanup_unconfirmed_ || g_device_finalize_calls != 0 ||
                        g_destroy_context_calls != 0 || g_comm_free_calls != 0 || worker.device_ctx_ == nullptr ||
                        worker.runtime_module_->unload_allowed_ || worker.unproven_native_run_count() != 1)
                        _exit(3);
                    try {
                        worker.comm_destroy_all();
                        _exit(4);
                    } catch (const std::exception &) {}
                } else if (refused || g_device_finalize_calls != 1 || g_destroy_context_calls != 1) {
                    _exit(5);
                }
                _exit(0);  // The failed owner intentionally survives; do not invoke test-process destructors.
            },
            ::testing::ExitedWithCode(0), ""
        );
    }
}

// A destroyed carrier without physical completion proof cannot authorize comm free.
TEST(NativeRunOperationLeaseTest, UnprovedDestroyedRunRetainsCommOwnersAcrossDeviceTeardownRetry) {
    for (bool failed : {false, true}) {
        SCOPED_TRACE(failed);
        ChipWorker worker;
        prime(worker);
        worker.finalize_device_fn_ = stub_finalize_device;
        worker.comm_release_domain_windows_fn_ = stub_release_domain;
        worker.comm_global_domain_release_fn_ = stub_release_global_domain;
        worker.comm_destroy_fn_ = stub_destroy_comm;
        static int comm_placeholder = 0;
        const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
        ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
        worker.base_comm_handle_ = comm;
        worker.global_domain_ids_.insert(7);
        const auto module = worker.runtime_module_;
        worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
        ChipStorageTaskArgs args{};
        ChipRun run = worker.run_lane_->submit(1, args, CallConfig{});
        g_wait_rc = failed ? -5 : 0;
        if (failed) g_poll_rc = SIMPLER_NATIVE_RUN_POLL_ERROR;
        if (failed) {
            EXPECT_THROW(run.wait_until(ChipRun::Deadline::max()), std::runtime_error);
        } else {
            EXPECT_TRUE(run.wait_until(ChipRun::Deadline::max()));
        }
        EXPECT_EQ(worker.run_lane_->poisoned(), failed);
        EXPECT_EQ(run.owner_recovery_required(), failed);
        EXPECT_EQ(g_finalize_calls, 1);
        EXPECT_TRUE(worker.native_runs_.empty());
        EXPECT_EQ(worker.unproven_native_run_count(), 0u);
        uint64_t tag = kNativeRunContextMagic;
        std::memcpy(&tag, worker.runtime_bufs_[0].data(), sizeof(tag));
        EXPECT_EQ(tag, 0u);
        if (!failed) {
            EXPECT_NO_THROW(worker.finalize());
            EXPECT_EQ(g_comm_free_calls, 2);
            EXPECT_EQ(g_device_finalize_calls, 1);
            EXPECT_TRUE(module->unload_allowed_);
            EXPECT_TRUE(worker.comm_sessions_.empty());
            EXPECT_TRUE(worker.global_domain_ids_.empty());
            continue;
        }
        const auto expect_refused = [&] {
            EXPECT_THROW(worker.comm_release_domain_windows(comm, 1, 2, 0), std::runtime_error);
            EXPECT_THROW(worker.comm_global_domain_release(7), std::runtime_error);
            EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
            EXPECT_THROW(worker.comm_destroy_all(), std::runtime_error);
            EXPECT_EQ(g_comm_free_calls, 0);
            ASSERT_EQ(worker.comm_sessions_.size(), 1u);
            EXPECT_EQ(worker.comm_sessions_[0].handle, &comm_placeholder);
            EXPECT_EQ(worker.global_domain_ids_.count(7), 1u);
        };
        expect_refused();
        g_device_finalize_rc = -9;
        EXPECT_THROW(worker.finalize(), std::runtime_error);
        EXPECT_EQ(g_device_finalize_calls, 1);
        EXPECT_EQ(g_destroy_context_calls, 0);
        EXPECT_FALSE(module->unload_allowed_);
        expect_refused();
        g_device_finalize_rc = 0;
        EXPECT_NO_THROW(worker.finalize());
        EXPECT_EQ(g_device_finalize_calls, 2);
        EXPECT_EQ(g_destroy_context_calls, 1);
        expect_refused();
        EXPECT_NO_THROW(worker.finalize());
        EXPECT_EQ(g_device_finalize_calls, 2);
        EXPECT_EQ(g_comm_free_calls, 0);
    }
}

TEST(NativeRunOperationLeaseTest, CompletedSemanticFailureReleasesCommOwnersWithoutClearingItsError) {
    for (bool explicit_release : {false, true}) {
        SCOPED_TRACE(explicit_release);
        ChipWorker worker;
        prime(worker);
        worker.finalize_run_with_last_use_fn_ = finalize_run_with_last_use;
        g_last_use_proof = SIMPLER_NATIVE_RUN_LAST_USE_PROVED;
        worker.finalize_device_fn_ = stub_finalize_device;
        worker.comm_release_domain_windows_fn_ = stub_release_domain;
        worker.comm_abort_domain_windows_fn_ = stub_abort_domain;
        worker.comm_global_domain_release_fn_ = stub_release_global_domain;
        worker.comm_destroy_fn_ = stub_destroy_comm;
        int base_placeholder = 0;
        int child_placeholder = 0;
        const uint64_t base = reinterpret_cast<uint64_t>(&base_placeholder);
        const uint64_t child = reinterpret_cast<uint64_t>(&child_placeholder);
        ASSERT_NE(worker.create_comm_session(&base_placeholder, nullptr, true), nullptr);
        ASSERT_NE(worker.create_comm_session(&child_placeholder, nullptr, false), nullptr);
        worker.base_comm_handle_ = base;
        worker.global_domain_ids_.insert(7);
        const auto module = worker.runtime_module_;
        worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
        ChipStorageTaskArgs args{};
        ChipRun run = worker.run_lane_->submit(1, args, CallConfig{});
        g_wait_rc = -5;
        const auto expect_original_error = [&] {
            try {
                (void)run.wait_until(ChipRun::Deadline::max());
                FAIL() << "the semantic error was discarded";
            } catch (const std::runtime_error &error) {
                EXPECT_NE(std::string(error.what()).find("-5"), std::string::npos);
            }
        };
        // The blocking completion entry proves END before the semantic observer fails.
        expect_original_error();
        EXPECT_EQ(g_poll_rc, SIMPLER_NATIVE_RUN_POLL_COMPLETE);
        EXPECT_TRUE(run.lane_poisoned());
        EXPECT_FALSE(run.owner_recovery_required());
        EXPECT_TRUE(worker.native_runs_.empty());
        EXPECT_FALSE(worker.comm_reclamation_unconfirmed_);
        EXPECT_EQ(g_wait_calls, 1);
        EXPECT_EQ(g_finalize_calls, 1);

        if (explicit_release) {
            EXPECT_NO_THROW(worker.comm_release_domain_windows(base, 1, 2, 0));
            EXPECT_NO_THROW(worker.comm_abort_domain_windows(base, 2, 2, 0));
            EXPECT_NO_THROW(worker.comm_global_domain_release(7));
            EXPECT_NO_THROW(worker.comm_destroy(child));
            EXPECT_NO_THROW(worker.comm_destroy_all());
            EXPECT_EQ(g_comm_free_calls, 4);
            EXPECT_EQ(g_comm_abort_calls, 1);
        }
        EXPECT_TRUE(run.lane_poisoned());
        EXPECT_NO_THROW(worker.finalize());
        EXPECT_EQ(g_comm_free_calls, explicit_release ? 4 : 3);
        EXPECT_TRUE(worker.comm_sessions_.empty());
        EXPECT_TRUE(worker.global_domain_ids_.empty());
        EXPECT_EQ(g_device_finalize_calls, 1);
        EXPECT_EQ(g_destroy_context_calls, 1);
        EXPECT_TRUE(module->unload_allowed_);
        EXPECT_TRUE(run.lane_poisoned());
        expect_original_error();
        EXPECT_NO_THROW(worker.finalize());
        EXPECT_EQ(g_wait_calls, 1);
        EXPECT_EQ(g_finalize_calls, 1);
        EXPECT_EQ(g_device_finalize_calls, 1);
        EXPECT_EQ(g_comm_free_calls, explicit_release ? 4 : 3);
    }
}

TEST(NativeRunOperationLeaseTest, FinalizerOwnWaitPublishesLatestRetirementProofDespiteSemanticFailure) {
    for (bool poll_proves_retirement : {false, true}) {
        SCOPED_TRACE(poll_proves_retirement);
        ChipWorker worker;
        prime(worker);
        worker.finalize_run_with_last_use_fn_ = finalize_run_with_last_use;
        g_last_use_proof = SIMPLER_NATIVE_RUN_LAST_USE_PROVED;
        worker.finalize_device_fn_ = stub_finalize_device;
        worker.comm_release_domain_windows_fn_ = stub_release_domain;
        worker.comm_destroy_fn_ = stub_destroy_comm;
        int comm_placeholder = 0;
        const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
        ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
        worker.base_comm_handle_ = comm;
        if (poll_proves_retirement)
            worker.wait_run_fn_ = [](void *device, void *runtime) {
                const int rc = wait_run(device, runtime);
                g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
                return rc;
            };
        worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
        ChipStorageTaskArgs args{};
        ChipRun run = worker.run_lane_->submit(1, args, CallConfig{});
        ASSERT_EQ(worker.native_runs_.size(), 1u);
        EXPECT_EQ(worker.native_runs_.begin()->second->phase, ChipWorker::NativeRunPhase::LAUNCHED);
        g_poll_rc = SIMPLER_NATIVE_RUN_POLL_ERROR;
        g_wait_rc = -5;
        // The failed initial poll enters finalization while the entry is still LAUNCHED.
        // Proof first arrives either from its own follow-up poll or the v1 finalizer.
        EXPECT_THROW(run.wait_until(ChipRun::Deadline::max()), std::runtime_error);
        EXPECT_EQ(g_wait_calls, 1);
        EXPECT_EQ(g_finalize_calls, 1);
        EXPECT_TRUE(worker.native_runs_.empty());
        EXPECT_TRUE(run.lane_poisoned());
        EXPECT_FALSE(run.owner_recovery_required());
        EXPECT_FALSE(worker.comm_reclamation_unconfirmed_);
        EXPECT_NO_THROW(worker.comm_release_domain_windows(comm, 1, 2, 0));
        EXPECT_EQ(g_comm_free_calls, 1);
        EXPECT_NO_THROW(worker.finalize());
        EXPECT_EQ(g_comm_free_calls, 2);
        EXPECT_EQ(g_device_finalize_calls, 1);
        EXPECT_FALSE(run.owner_recovery_required());
    }
}

TEST(NativeRunOperationLeaseTest, DetachedLaneKeepsCommunicationDebtWithoutChangingTheRunError) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.comm_destroy_fn_ = stub_destroy_comm;
    int comm_placeholder = 0;
    ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
    worker.base_comm_handle_ = reinterpret_cast<uint64_t>(&comm_placeholder);
    worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
    ChipStorageTaskArgs args{};
    ChipRun run = worker.run_lane_->submit(1, args, CallConfig{});
    g_wait_rc = -5;
    EXPECT_THROW(run.wait_until(ChipRun::Deadline::max()), std::runtime_error);
    EXPECT_TRUE(worker.native_runs_.empty());
    EXPECT_TRUE(run.owner_recovery_required());
    EXPECT_NO_THROW(worker.finalize());
    EXPECT_EQ(g_comm_free_calls, 0);
    EXPECT_TRUE(run.owner_recovery_required());
    EXPECT_NO_THROW(worker.finalize());
    EXPECT_TRUE(run.owner_recovery_required());
    try {
        (void)run.wait_until(ChipRun::Deadline::max());
        FAIL() << "the semantic error was discarded after detach";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-5"), std::string::npos);
    }
}

TEST(NativeRunOperationLeaseTest, LocalRetirementCannotReplaceMissingUnknownOrInvalidBackendProof) {
    for (int provider : {-1, static_cast<int>(SIMPLER_NATIVE_RUN_LAST_USE_UNKNOWN), 2}) {
        SCOPED_TRACE(provider);
        ChipWorker worker;
        prime(worker);
        worker.finalize_device_fn_ = stub_finalize_device;
        worker.comm_release_domain_windows_fn_ = stub_release_domain;
        worker.comm_destroy_fn_ = stub_destroy_comm;
        if (provider >= 0) {
            worker.finalize_run_with_last_use_fn_ = finalize_run_with_last_use;
            g_last_use_proof = static_cast<uint32_t>(provider);
        }
        int comm_placeholder = 0;
        const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
        ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
        worker.base_comm_handle_ = comm;
        const auto run = prepare(worker, 0, 1);
        worker.launch_native_run(run);
        g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        g_wait_rc = -5;
        EXPECT_NO_THROW(worker.wait_native_run(run));
        EXPECT_EQ(worker.native_runs_.at(run.run_epoch)->phase, ChipWorker::NativeRunPhase::REAPED);
        EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
        EXPECT_TRUE(worker.native_runs_.empty());
        EXPECT_TRUE(worker.comm_reclamation_unconfirmed());
        EXPECT_THROW(worker.comm_release_domain_windows(comm, 1, 2, 0), std::runtime_error);
        EXPECT_THROW(worker.comm_destroy_all(), std::runtime_error);
        EXPECT_NO_THROW(worker.finalize());
        EXPECT_EQ(g_comm_free_calls, 0);
        EXPECT_EQ(worker.comm_sessions_.size(), 1u);
    }
}

TEST(NativeRunOperationLeaseTest, UnknownPrepareRollbackWithoutNativeCarrierStillProtectsCommOwners) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.comm_release_domain_windows_fn_ = stub_release_domain;
    worker.comm_abort_domain_windows_fn_ = stub_abort_domain;
    worker.comm_global_domain_release_fn_ = stub_release_global_domain;
    worker.comm_destroy_fn_ = stub_destroy_comm;
    int comm_placeholder = 0;
    const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
    ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
    worker.base_comm_handle_ = comm;
    worker.global_domain_ids_.insert(7);
    worker.prepare_run_fn_ = [](void *, void *runtime, int32_t, const void *, const CallConfig *,
                                const NativeRunDescriptor *) {
        // An empty carrier is not the released-preparation receipt for this epoch.
        const uint64_t destroyed = 0;
        std::memcpy(runtime, &destroyed, sizeof(destroyed));
        return -37;
    };
    worker.run_lane_ = std::make_unique<ChipRunLane>(worker);
    ChipStorageTaskArgs args{};
    ChipRun run = worker.run_lane_->submit(1, args, CallConfig{});
    EXPECT_THROW(run.wait_until(ChipRun::Deadline::max()), std::runtime_error);
    EXPECT_FALSE(run.launched());
    EXPECT_TRUE(run.lane_poisoned());
    EXPECT_TRUE(run.owner_recovery_required());
    EXPECT_TRUE(worker.native_runs_.empty());
    EXPECT_EQ(g_wait_calls, 0);
    EXPECT_EQ(g_finalize_calls, 0);
    EXPECT_TRUE(worker.comm_reclamation_unconfirmed_);
    EXPECT_THROW(worker.comm_release_domain_windows(comm, 1, 2, 0), std::runtime_error);
    EXPECT_THROW(worker.comm_abort_domain_windows(comm, 2, 2, 0), std::runtime_error);
    EXPECT_THROW(worker.comm_global_domain_release(7), std::runtime_error);
    EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
    EXPECT_THROW(worker.comm_destroy_all(), std::runtime_error);
    EXPECT_EQ(g_comm_free_calls, 0);
    EXPECT_EQ(g_comm_abort_calls, 0);
    EXPECT_NO_THROW(worker.finalize());
    EXPECT_NO_THROW(worker.finalize());
    EXPECT_EQ(g_comm_free_calls, 0);
    EXPECT_EQ(g_device_finalize_calls, 1);
    EXPECT_EQ(worker.comm_sessions_.size(), 1u);
    EXPECT_EQ(worker.global_domain_ids_.count(7), 1u);
}

TEST(NativeRunOperationLeaseTest, UnpublishedDomainAbortUsesLocalBackendAndHonorsReclamationGuard) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.comm_destroy_fn_ = stub_destroy_comm;
    worker.comm_release_domain_windows_fn_ = stub_release_domain;
    worker.comm_abort_domain_windows_fn_ = stub_abort_domain;
    int comm_placeholder = 0;
    const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
    ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
    EXPECT_NO_THROW(worker.comm_abort_domain_windows(comm, 1, 2, 0));
    EXPECT_EQ(g_comm_abort_calls, 1);
    EXPECT_EQ(g_comm_free_calls, 0);

    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    EXPECT_THROW(worker.comm_abort_domain_windows(comm, 2, 2, 0), std::runtime_error);
    EXPECT_EQ(g_comm_abort_calls, 1);
    worker.finalize_native_run(run);

    g_comm_rc = COMM_RECLAMATION_UNCONFIRMED;
    EXPECT_THROW(worker.comm_abort_domain_windows(comm, 2, 2, 0), std::runtime_error);
    EXPECT_EQ(g_comm_abort_calls, 2);
    ASSERT_NE(worker.find_comm_session(comm), nullptr);
    EXPECT_THROW(worker.comm_abort_domain_windows(comm, 2, 2, 0), std::runtime_error);
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_comm_abort_calls, 2);
    EXPECT_EQ(g_comm_free_calls, 0);
    EXPECT_EQ(g_device_finalize_calls, 0);
    EXPECT_EQ(g_destroy_context_calls, 0);
}

TEST(NativeRunOperationLeaseTest, DomainAllocationFailurePublishesCleanupOwnership) {
    const int results[] = {
        -1,
        COMM_DOMAIN_WORKSPACE_UNSUPPORTED,
        COMM_DOMAIN_URMA_CONFIG_INVALID,
        COMM_DOMAIN_URMA_INIT_FAILED,
        COMM_DOMAIN_URMA_WORKSPACE_INVALID,
        COMM_INITIALIZATION_ABORT_REQUIRED,
        COMM_RECLAMATION_UNCONFIRMED
    };
    for (int rc : results) {
        SCOPED_TRACE(rc);
        ChipWorker worker;
        prime(worker);
        worker.comm_destroy_fn_ = stub_destroy_comm;
        worker.comm_alloc_domain_windows_fn_ = stub_alloc_domain;
        worker.comm_abort_domain_windows_fn_ = stub_abort_domain;
        int comm_placeholder = 0;
        const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
        ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
        uint64_t ownership = 0;
        g_comm_rc = rc;
        EXPECT_THROW(worker.comm_alloc_domain_windows(comm, 1, {0, 1}, 0, 4096, 2, &ownership), std::runtime_error);
        EXPECT_EQ(
            __atomic_load_n(&ownership, __ATOMIC_ACQUIRE),
            (rc == COMM_INITIALIZATION_ABORT_REQUIRED || rc == COMM_RECLAMATION_UNCONFIRMED) ? 1u : 0u
        );
        ASSERT_NE(worker.find_comm_session(comm), nullptr);
        EXPECT_EQ(worker.comm_cleanup_unconfirmed_, rc == COMM_RECLAMATION_UNCONFIRMED);
        g_comm_rc = 0;
        if (rc == COMM_INITIALIZATION_ABORT_REQUIRED) {
            EXPECT_NO_THROW(worker.comm_abort_domain_windows(comm, 1, 2, 0));
            EXPECT_EQ(g_comm_abort_calls, 1);
            EXPECT_NO_THROW(worker.finalize());
        } else if (rc == COMM_RECLAMATION_UNCONFIRMED) {
            EXPECT_THROW(worker.comm_abort_domain_windows(comm, 1, 2, 0), std::runtime_error);
            EXPECT_EQ(g_comm_abort_calls, 0);
            EXPECT_THROW(worker.finalize(), std::runtime_error);
        }
    }
}

TEST(NativeRunOperationLeaseTest, DomainAllocationOwnershipPrecedesOutputValidation) {
    ChipWorker worker;
    prime(worker);
    worker.comm_destroy_fn_ = stub_destroy_comm;
    worker.comm_alloc_domain_windows_fn_ = stub_alloc_domain;
    worker.comm_abort_domain_windows_fn_ = stub_abort_domain;
    int comm_placeholder = 0;
    const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
    ASSERT_NE(worker.create_comm_session(&comm_placeholder, nullptr, true), nullptr);
    uint64_t ownership = 0;
    EXPECT_THROW(worker.comm_alloc_domain_windows(comm, 1, {0, 1}, 0, 4096, 2, &ownership), std::runtime_error);
    EXPECT_EQ(__atomic_load_n(&ownership, __ATOMIC_ACQUIRE), 1u);
    EXPECT_NO_THROW(worker.comm_abort_domain_windows(comm, 1, 2, 0));

    worker.comm_alloc_domain_windows_fn_ = [](void *, uint64_t, const uint32_t *, size_t, uint32_t, size_t, uint32_t,
                                              uint64_t *ctx, uint64_t *window, size_t *size) {
        *ctx = 0x100;
        *window = 0x200;
        *size = 4096;
        return 0;
    };
    ownership = 0;
    const auto result = worker.comm_alloc_domain_windows(comm, 2, {0, 1}, 0, 4096, 2, &ownership);
    EXPECT_EQ(std::get<0>(result), 0x100u);
    EXPECT_EQ(std::get<1>(result), 0x200u);
    EXPECT_EQ(__atomic_load_n(&ownership, __ATOMIC_ACQUIRE), 1u);
}

TEST(NativeRunOperationLeaseTest, FailedCommCleanupRetainsOwnerStreamAndDeviceAcrossClose) {
    for (int operation = 0; operation < 3; ++operation) {
        SCOPED_TRACE(operation);
        const auto module = std::make_shared<RuntimeModule>(nullptr);
        {
            ChipWorker worker;
            worker.runtime_module_ = module;
            prime(worker);
            worker.finalize_device_fn_ = stub_finalize_device;
            worker.comm_destroy_fn_ = stub_destroy_comm;
            worker.comm_release_domain_windows_fn_ = stub_release_domain;
            worker.comm_alloc_domain_windows_fn_ = stub_alloc_domain;
            worker.destroy_comm_stream_fn_ = stub_destroy_comm_stream;
            int comm_placeholder = 0;
            int stream_placeholder = 0;
            const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
            ASSERT_NE(worker.create_comm_session(&comm_placeholder, &stream_placeholder, true), nullptr);
            worker.base_comm_handle_ = comm;
            g_comm_rc = COMM_RECLAMATION_UNCONFIRMED;
            if (operation == 0) {
                EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
            } else if (operation == 1) {
                EXPECT_THROW(worker.comm_release_domain_windows(comm, 1, 2, 0), std::runtime_error);
            } else {
                EXPECT_THROW(worker.comm_alloc_domain_windows(comm, 1, {0, 1}, 0, 4096, 2), std::runtime_error);
            }
            ASSERT_NE(worker.find_comm_session(comm), nullptr);
            EXPECT_EQ(worker.find_comm_session(comm)->handle, &comm_placeholder);
            EXPECT_EQ(worker.find_comm_session(comm)->stream, &stream_placeholder);
            EXPECT_EQ(worker.base_comm_handle_, comm);
            EXPECT_EQ(g_comm_stream_free_calls, 0);
            const int cleanup_calls = g_comm_free_calls;
            const int allocation_calls = g_comm_alloc_calls;
            EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
            EXPECT_THROW(worker.comm_release_domain_windows(comm, 1, 2, 0), std::runtime_error);
            EXPECT_THROW(worker.comm_alloc_domain_windows(comm, 2, {0, 1}, 0, 4096, 2), std::runtime_error);
            EXPECT_THROW(prepare(worker, 0, 1), std::runtime_error);
            EXPECT_THROW(worker.finalize(), std::runtime_error);
            EXPECT_THROW(worker.finalize(), std::runtime_error);
            EXPECT_EQ(g_comm_free_calls, cleanup_calls);
            EXPECT_EQ(g_comm_alloc_calls, allocation_calls);
            EXPECT_EQ(g_device_finalize_calls, 0);
            EXPECT_EQ(g_destroy_context_calls, 0);
            EXPECT_FALSE(module->unload_allowed_);
        }
        EXPECT_EQ(g_comm_stream_free_calls, 0);
        EXPECT_EQ(g_device_finalize_calls, 0);
        EXPECT_EQ(g_destroy_context_calls, 0);
        EXPECT_FALSE(module->unload_allowed_);
    }
}

TEST(NativeRunOperationLeaseTest, FinalizeStopsBeforeBaseWhenChildCommCleanupIsUnconfirmed) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.comm_destroy_fn_ = stub_destroy_comm;
    worker.destroy_comm_stream_fn_ = stub_destroy_comm_stream;
    int base = 0;
    int child = 0;
    int stream = 0;
    ASSERT_NE(worker.create_comm_session(&base, &stream, true), nullptr);
    ASSERT_NE(worker.create_comm_session(&child, &stream, false), nullptr);
    worker.base_comm_handle_ = reinterpret_cast<uint64_t>(&base);
    g_comm_rc = COMM_RECLAMATION_UNCONFIRMED;
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_comm_free_calls, 1);
    EXPECT_EQ(worker.comm_sessions_.size(), 2u);
    EXPECT_EQ(worker.comm_session_index_.size(), 2u);
    EXPECT_EQ(g_comm_stream_free_calls, 0);
    EXPECT_EQ(g_device_finalize_calls, 0);
    EXPECT_EQ(g_destroy_context_calls, 0);
}

TEST(NativeRunOperationLeaseTest, ConsumedCommErrorStillReleasesSessionAndStream) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.comm_destroy_fn_ = stub_destroy_comm;
    worker.destroy_comm_stream_fn_ = stub_destroy_comm_stream;
    int comm_placeholder = 0;
    int stream_placeholder = 0;
    const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
    ASSERT_NE(worker.create_comm_session(&comm_placeholder, &stream_placeholder, true), nullptr);
    worker.base_comm_handle_ = comm;
    g_comm_rc = -1;
    EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
    EXPECT_EQ(worker.find_comm_session(comm), nullptr);
    EXPECT_EQ(worker.base_comm_handle_, 0u);
    EXPECT_EQ(g_comm_stream_free_calls, 1);
    EXPECT_NO_THROW(worker.finalize());
    EXPECT_EQ(g_comm_free_calls, 1);
    EXPECT_EQ(g_device_finalize_calls, 1);
    EXPECT_EQ(g_destroy_context_calls, 1);
}

TEST(NativeRunOperationLeaseTest, FailedCommStreamDestroyRetainsStreamWithoutRetryingConsumedHandle) {
    for (int backend_rc : {0, -1}) {
        SCOPED_TRACE(backend_rc);
        ChipWorker worker;
        prime(worker);
        worker.finalize_device_fn_ = stub_finalize_device;
        worker.comm_destroy_fn_ = stub_destroy_comm;
        worker.destroy_comm_stream_fn_ = stub_destroy_comm_stream;
        int comm_placeholder = 0;
        int stream_placeholder = 0;
        const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
        ASSERT_NE(worker.create_comm_session(&comm_placeholder, &stream_placeholder, true), nullptr);
        worker.base_comm_handle_ = comm;
        g_comm_rc = backend_rc;
        g_comm_stream_rc = -7;
        EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
        ASSERT_EQ(worker.comm_sessions_.size(), 1u);
        EXPECT_EQ(worker.comm_sessions_[0].handle, nullptr);
        EXPECT_EQ(worker.comm_sessions_[0].stream, &stream_placeholder);
        EXPECT_EQ(worker.find_comm_session(comm), nullptr);
        EXPECT_EQ(worker.base_comm_handle_, 0u);
        EXPECT_EQ(g_comm_free_calls, 1);
        EXPECT_EQ(g_comm_stream_free_calls, 1);
        EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
        EXPECT_THROW(worker.finalize(), std::runtime_error);
        EXPECT_EQ(g_comm_free_calls, 1);
        EXPECT_EQ(g_comm_stream_free_calls, 1);
        EXPECT_EQ(g_device_finalize_calls, 0);
        EXPECT_EQ(g_destroy_context_calls, 0);
        EXPECT_FALSE(worker.runtime_module_->unload_allowed_);
    }
}

TEST(NativeRunOperationLeaseTest, UntrackedCommCleanupFailureRegistersRetainedOwner) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.comm_destroy_fn_ = stub_destroy_comm;
    int comm_placeholder = 0;
    const uint64_t comm = reinterpret_cast<uint64_t>(&comm_placeholder);
    g_comm_rc = COMM_RECLAMATION_UNCONFIRMED;
    EXPECT_THROW(worker.comm_destroy(comm), std::runtime_error);
    ASSERT_NE(worker.find_comm_session(comm), nullptr);
    EXPECT_EQ(worker.find_comm_session(comm)->handle, &comm_placeholder);
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_comm_free_calls, 1);
    EXPECT_EQ(g_device_finalize_calls, 0);
}

TEST(NativeRunOperationLeaseTest, FailedCommInitRetainsStreamWhenRollbackFails) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.destroy_comm_stream_fn_ = stub_destroy_comm_stream;
    static int stream_placeholder = 0;
    worker.ensure_acl_ready_fn_ = [](void *, int) {
        return 0;
    };
    worker.create_comm_stream_fn_ = [](void *) -> void * {
        return &stream_placeholder;
    };
    worker.comm_init_fn_ = [](int, int, void *, const char *) -> void * {
        return nullptr;
    };
    g_comm_stream_rc = -7;
    EXPECT_THROW(worker.create_base_comm(0, 2, "unused"), std::runtime_error);
    ASSERT_EQ(worker.comm_sessions_.size(), 1u);
    EXPECT_EQ(worker.comm_sessions_[0].handle, nullptr);
    EXPECT_EQ(worker.comm_sessions_[0].stream, &stream_placeholder);
    EXPECT_EQ(g_comm_stream_free_calls, 1);
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_comm_stream_free_calls, 1);
    EXPECT_EQ(g_device_finalize_calls, 0);
    EXPECT_EQ(g_destroy_context_calls, 0);
}

TEST(NativeRunOperationLeaseTest, DuplicateCommCleanupFailureRetainsBothStreams) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.comm_destroy_fn_ = stub_destroy_comm;
    worker.destroy_comm_stream_fn_ = stub_destroy_comm_stream;
    static int comm_placeholder = 0;
    static int new_stream_placeholder = 0;
    int old_stream_placeholder = 0;
    ASSERT_NE(worker.create_comm_session(&comm_placeholder, &old_stream_placeholder, false), nullptr);
    worker.ensure_acl_ready_fn_ = [](void *, int) {
        return 0;
    };
    worker.create_comm_stream_fn_ = [](void *) -> void * {
        return &new_stream_placeholder;
    };
    worker.comm_init_fn_ = [](int, int, void *, const char *) -> void * {
        return &comm_placeholder;
    };
    g_comm_rc = COMM_RECLAMATION_UNCONFIRMED;
    EXPECT_THROW(worker.create_base_comm(0, 2, "unused"), std::runtime_error);
    EXPECT_EQ(g_comm_free_calls, 1);
    EXPECT_EQ(g_comm_stream_free_calls, 0);
    ASSERT_EQ(worker.comm_sessions_.size(), 2u);
    EXPECT_EQ(worker.comm_sessions_[0].stream, &old_stream_placeholder);
    EXPECT_EQ(worker.comm_sessions_[1].stream, &new_stream_placeholder);
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_comm_free_calls, 1);
    EXPECT_EQ(g_device_finalize_calls, 0);
}

// A run the sweep could not prove destroyed still owns whatever its carrier holds, and that
// carrier still lives in this worker's run storage. The close therefore refuses the whole
// reclamation below it rather than freeing those resources twice or unloading their owner.
TEST(NativeRunOperationLeaseTest, WholeCloseRetainsAnUnprovenRunWhenDeviceTeardownRefuses) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    g_device_finalize_rc = -9;
    worker.destroy_device_context_fn_ = stub_destroy_device_context;
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);
    worker.launch_native_run(run);

    g_finalize_destroys = false;
    g_finalize_corrupts = true;
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_device_finalize_calls, 1);
    EXPECT_EQ(g_destroy_context_calls, 0) << "the context was destroyed behind an undestroyed run";
    EXPECT_NE(worker.device_ctx_, nullptr);
    EXPECT_FALSE(worker.runtime_bufs_.empty()) << "the carrier's storage was released under it";
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_TRUE(it->second->destruction_unknown);
    }
    // The refusal is not a one-shot: a second close sees the same unproven run and refuses again
    // without re-entering the partially executed destructor. The common close may retry its proof.
    EXPECT_THROW(worker.finalize(), std::runtime_error);
    EXPECT_EQ(g_device_finalize_calls, 2);
    EXPECT_EQ(g_finalize_calls, 1) << "an isolated run was handed to the runtime again";
    // The fixture's worker is destroyed holding the isolated run, which is what production does
    // too; its destructor absorbs the same refusal.
}

// This zero models the strengthened common-close contract, not the business
// verdict: the real architecture/probe consumer separately proves reset.
TEST(NativeRunOperationLeaseTest, ConfirmedTerminalCloseNeverReentersAPartialDestructor) {
    ChipWorker worker;
    prime(worker);
    worker.finalize_device_fn_ = stub_finalize_device;
    worker.destroy_device_context_fn_ = stub_destroy_device_context;
    const auto run = prepare(worker, 0, 1);
    worker.launch_native_run(run);
    void *carrier = worker.native_runs_.at(run.run_epoch)->storage.data();
    g_finalize_destroys = false;
    g_finalize_corrupts = true;
    EXPECT_NO_THROW(worker.finalize());
    EXPECT_EQ(g_finalize_calls, 1);
    EXPECT_EQ(g_device_finalize_calls, 1);
    EXPECT_EQ(g_destroy_context_calls, 1);
    EXPECT_EQ(worker.device_ctx_, nullptr);
    EXPECT_TRUE(worker.native_runs_.empty());
    EXPECT_NO_THROW(worker.finalize());
    EXPECT_EQ(g_finalize_calls, 1);
    EXPECT_EQ(g_device_finalize_calls, 1);
    std::free(carrier);  // This stub carrier owns no real resources.
}

// Retaining a map entry preserves nothing once the owner is gone: the entry and
// the carrier's storage both belong to the worker, so its destructor would free
// those bytes whatever the run table said. What has to survive is the block, and
// the proof is that the allocator never gets it back.
//
// The carrier is given a known word before the close and read back after the
// worker is destroyed. That read is only meaningful because the block was never
// freed — which is exactly the claim — so the case also asserts the worker
// reported abandoning it, and nothing else in this fixture writes there.
TEST(NativeRunOperationLeaseTest, AnUnprovenRunsStorageOutlivesTheWorkerThatHeldIt) {
    static constexpr uint64_t kCarrierWitness = UINT64_C(0xA5A5C0FFEE5151A5);
    void *carrier = nullptr;
    {
        ChipWorker worker;
        prime(worker);
        worker.finalize_device_fn_ = stub_finalize_device;
        g_device_finalize_rc = -9;
        worker.destroy_device_context_fn_ = stub_destroy_device_context;
        const ChipWorkerNativeRun run = prepare(worker, 0, 1);
        worker.launch_native_run(run);
        carrier = worker.native_runs_.at(run.run_epoch)->storage.data();
        ASSERT_NE(carrier, nullptr);

        // A teardown that leaves the tag unreadable: the run is isolated, and
        // whether anything still reaches the carrier is exactly what the host
        // cannot tell.
        g_finalize_destroys = false;
        g_finalize_corrupts = true;
        EXPECT_THROW(worker.finalize(), std::runtime_error);

        // Detached from the owner, so the block is no longer this object's to
        // free — the run table still names the set, but that is bookkeeping and
        // this is the ownership.
        EXPECT_EQ(worker.native_runs_.at(run.run_epoch)->storage.data(), nullptr)
            << "the carrier's storage is still owned by a worker that is about to be destroyed";
        std::memcpy(carrier, &kCarrierWitness, sizeof(kCarrierWitness));
    }

    // The worker and its run table are gone. The block is not.
    uint64_t witness = 0;
    std::memcpy(&witness, carrier, sizeof(witness));
    EXPECT_EQ(witness, kCarrierWitness) << "the abandoned run storage did not survive its owner";
    // This case owns the leak it just proved, so it is the one that frees it.
    std::free(carrier);
}

// A launch error outranks the later explicit observer's error, independently of
// which operation first established physical completion.
TEST(NativeRunOperationLeaseTest, FailedLaunchThatCompletedKeepsItsOwnError) {
    ChipWorker worker;
    prime(worker);
    const ChipWorkerNativeRun run = prepare(worker, 0, 1);

    g_launch_rc = -11;
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    EXPECT_THROW(worker.launch_native_run(run), std::runtime_error);
    {
        std::lock_guard<std::mutex> lk(worker.native_run_mu_);
        const auto it = worker.native_runs_.find(run.run_epoch);
        ASSERT_NE(it, worker.native_runs_.end());
        EXPECT_EQ(it->second->phase, ChipWorker::NativeRunPhase::REAPED);
        EXPECT_EQ(it->second->pending_rc, -11);
    }
    g_wait_rc = -12;
    try {
        worker.finalize_native_run(run);
        FAIL() << "the launch error was discarded";
    } catch (const std::runtime_error &error) {
        EXPECT_NE(std::string(error.what()).find("-11"), std::string::npos);
    }
    EXPECT_EQ(g_wait_calls, 1);
}

// Model the checked sim wait's early binding failure using the existing backend
// stubs, while exercising the production ChipWorker state machine.
TEST(NativeRunOperationLeaseTest, IncompleteWaitRemainsRetryable) {
    ChipWorker worker;
    prime(worker);
    const auto run = prepare(worker, 0, 1);
    worker.launch_native_run(run);
    g_wait_rc = SIMPLER_NATIVE_RUN_WAIT_INCOMPLETE;
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_NOT_READY;
    EXPECT_THROW(worker.wait_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 1);
    EXPECT_FALSE(worker.native_runs_.at(run.run_epoch)->pending_rc_set);
    EXPECT_FALSE(worker.native_runs_.at(run.run_epoch)->result_observed);
    EXPECT_EQ(worker.native_runs_.at(run.run_epoch)->phase, ChipWorker::NativeRunPhase::LAUNCHED);
    EXPECT_FALSE(worker.poll_native_run(run));
    // A second caller must reach the backend while it still owes a drain.
    EXPECT_THROW(worker.wait_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 2);
    // A later successful wait must complete without retaining the transient error.
    g_wait_rc = 0;
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    EXPECT_NO_THROW(worker.wait_native_run(run));
    EXPECT_EQ(g_wait_calls, 3);
    EXPECT_TRUE(worker.poll_native_run(run));
    EXPECT_NO_THROW(worker.finalize_native_run(run));
}

TEST(NativeRunOperationLeaseTest, FinalizeIncompleteWaitRemainsRetryable) {
    ChipWorker worker;
    prime(worker);
    const auto run = prepare(worker, 0, 1);
    worker.launch_native_run(run);
    g_wait_rc = SIMPLER_NATIVE_RUN_WAIT_INCOMPLETE;
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_NOT_READY;
    g_finalize_rc = PTO_RUNTIME_ERR_INTERNAL;
    g_finalize_destroys = false;
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 1);
    EXPECT_FALSE(worker.native_runs_.at(run.run_epoch)->pending_rc_set);
    EXPECT_FALSE(worker.native_runs_.at(run.run_epoch)->result_observed);
    EXPECT_EQ(worker.native_runs_.at(run.run_epoch)->phase, ChipWorker::NativeRunPhase::LAUNCHED);
    EXPECT_FALSE(worker.poll_native_run(run));
    EXPECT_EQ(g_finalize_calls, 0);
    EXPECT_FALSE(worker.native_runs_.at(run.run_epoch)->finalize_reserved);
    EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
    EXPECT_EQ(g_wait_calls, 2);
    g_wait_rc = 0;
    g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    g_finalize_rc = 0;
    g_finalize_destroys = true;
    EXPECT_NO_THROW(worker.finalize_native_run(run));
    EXPECT_EQ(g_wait_calls, 3);
    EXPECT_EQ(g_finalize_calls, 1);
}

TEST(NativeRunOperationLeaseTest, CompletedExecutionErrorStaysComplete) {
    for (int error : {-7, static_cast<int>(PTO_RUNTIME_ERR_INTERNAL)}) {
        ChipWorker worker;
        prime(worker);
        const auto run = prepare(worker, 0, 1);
        worker.launch_native_run(run);
        g_wait_rc = error;
        g_poll_rc = SIMPLER_NATIVE_RUN_POLL_COMPLETE;
        EXPECT_NO_THROW(worker.wait_native_run(run));
        EXPECT_EQ(worker.native_runs_.at(run.run_epoch)->phase, ChipWorker::NativeRunPhase::REAPED);
        EXPECT_TRUE(worker.poll_native_run(run));
        EXPECT_NO_THROW(worker.wait_native_run(run));
        EXPECT_EQ(g_wait_calls, 1);
        EXPECT_THROW(worker.finalize_native_run(run), std::runtime_error);
        EXPECT_EQ(g_finalize_calls, 1);
    }
}
