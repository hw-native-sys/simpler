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

#include <atomic>
#include <cstring>
#include <memory>
#include <thread>

#include "device_runner_base.h"
#include "native_run_context.h"
#include "runtime_c_api.h"

namespace {
int release_bindings_rc = 0;
int copyback_calls = 0;
}  // namespace

// Runtime IO is the collaborator here. The real sim C API owns finalization,
// stop/result separation, claim release and placement-carrier destruction.
extern "C" int copy_back_run_outputs_impl(const Runtime *, const HostApi *, int execution_rc, int) {
    ++copyback_calls;
    return execution_rc;
}
extern "C" int release_run_bindings_impl(Runtime *, const HostApi *) { return release_bindings_rc; }

namespace {
class JoiningRunner : public SimDeviceRunnerBase {
public:
    JoiningRunner() {
        EXPECT_TRUE(set_run_resource_grant(1));
        EXPECT_EQ(attach_current_thread(0), 0);
    }
    ~JoiningRunner() override {
        if (producer.joinable()) producer.join();
    }
    int prepare_execution(
        Runtime &, const CallConfig &, uint32_t, const NativeRunIdentity &, std::unique_ptr<PreparedExecution> *
    ) override {
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    void abandon_prepared_execution(PreparedExecution &) noexcept override {}
    int poll_execution(const ActiveExecution &) override { return SIMPLER_NATIVE_RUN_POLL_NOT_READY; }
    DrainOutcome drain_execution(ActiveExecution &active) override {
        ++drains;
        if (refuse_stop) return DrainOutcome::device_error(semantic_rc);
        stop_run_producers(active.prepared->identity.run_epoch, [&] {
            producer.join();
        });
        return DrainOutcome::device_error(semantic_rc, true);
    }
    int finalize() override { return 0; }
    std::thread producer;
    int semantic_rc{-100};
    int drains{0};
    bool refuse_stop{false};

protected:
    LaunchTransactionResult launch_execution_impl(PreparedExecution &, LaunchPermit) override { return {}; }
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs &) override { return 0; }
};

struct Invocation {
    explicit Invocation(JoiningRunner &owner) :
        runner(owner) {
        release_bindings_rc = 0;
        copyback_calls = 0;
        storage = ::operator new(get_runtime_size(), std::align_val_t(get_runtime_alignment()));
        std::memset(storage, 0, get_runtime_size());
        NativeRunDescriptor descriptor{};
        descriptor.run_epoch = descriptor.generation = descriptor.dispatch_id = 17;
        state = new (storage) NativeRunContext<SimDeviceRunnerBase>(&runner, CallConfig{}, 0, descriptor, &ops);
        EXPECT_TRUE(runner.try_acquire_native_run(state, state->identity(), &state->launch_permit));
        state->runner_claimed = true;
        auto prepared = std::make_unique<SimDeviceRunnerBase::PreparedExecution>(
            state->identity(), *state->runtime, state->config, 0
        );
        state->active_execution =
            std::make_unique<SimDeviceRunnerBase::ActiveExecution>(std::move(prepared), LaunchProgress::Complete);
        state->completion_rc = 0;
        state->phase.store(NativeRunPhase::Running);
        runner.producer = std::thread([this] {
            produced.store(true, std::memory_order_release);
        });
    }
    ~Invocation() {
        if (!destroyed()) {
            release_bindings_rc = 0;
            runner.refuse_stop = false;
            (void)simpler_finalize_run_with_delivery(&runner, storage, 0);
        }
        ::operator delete(storage, std::align_val_t(get_runtime_alignment()));
    }
    bool destroyed() const {
        uint64_t tag = UINT64_MAX;
        std::memcpy(&tag, storage, sizeof(tag));
        return tag == 0;
    }
    JoiningRunner &runner;
    HostApiOps ops{};
    void *storage{nullptr};
    NativeRunContext<SimDeviceRunnerBase> *state{nullptr};
    std::atomic<bool> produced{false};
};
}  // namespace

TEST(SimNativeLastUse, SemanticErrorKeepsItsCodeAfterARealJoinAndCarrierDestruction) {
    for (int semantic_rc : {-100, -102}) {
        JoiningRunner runner;
        runner.semantic_rc = semantic_rc;
        Invocation invocation(runner);

        EXPECT_EQ(simpler_finalize_run_with_delivery(&runner, invocation.storage, 0), semantic_rc);

        EXPECT_TRUE(invocation.produced.load(std::memory_order_acquire));
        EXPECT_TRUE(runner.run_producers_stopped(17));
        EXPECT_TRUE(invocation.destroyed());
        EXPECT_FALSE(runner.native_run_active());
        EXPECT_EQ(runner.drains, 1);
        EXPECT_EQ(copyback_calls, 1);
    }
}

TEST(SimNativeLastUse, InvalidDeliveryDoesNotStartFinalization) {
    JoiningRunner runner;
    Invocation invocation(runner);

    EXPECT_EQ(simpler_finalize_run_with_delivery(&runner, invocation.storage, 2), PTO_RUNTIME_ERR_INVALID_ARGUMENT);

    EXPECT_EQ(runner.drains, 0);
    EXPECT_FALSE(invocation.destroyed());
}

TEST(SimNativeLastUse, UnprovedStopRetainsTheCarrierAndAProvedRetryCanFinalize) {
    JoiningRunner runner;
    runner.refuse_stop = true;
    Invocation invocation(runner);

    EXPECT_EQ(simpler_finalize_run_with_delivery(&runner, invocation.storage, 0), runner.semantic_rc);

    EXPECT_FALSE(invocation.destroyed());
    runner.refuse_stop = false;
    EXPECT_EQ(simpler_finalize_run_with_delivery(&runner, invocation.storage, 0), runner.semantic_rc);

    EXPECT_TRUE(invocation.destroyed());
}

TEST(SimNativeLastUse, FailedBindingCleanupPreservesTheExecutionError) {
    JoiningRunner runner;
    Invocation invocation(runner);
    release_bindings_rc = -61;

    EXPECT_EQ(simpler_finalize_run_with_delivery(&runner, invocation.storage, 0), runner.semantic_rc);

    EXPECT_TRUE(invocation.destroyed());
}
