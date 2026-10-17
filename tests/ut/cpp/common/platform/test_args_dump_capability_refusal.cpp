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

/**
 * What the public preparation entry does with an argument dump the runtime
 * beneath it has no producer for.
 *
 * The decision has two inputs — the request's `enable_dump_args`, and the
 * runtime's own `args_dump_capture_supported_impl` — and one place it is taken:
 * `simpler_prepare_run`, after carrier validation and ownership. These cases
 * pin: the request is refused as `PTO_RUNTIME_ERR_UNSUPPORTED` with no live carrier
 * left in the caller's storage and an exact-epoch release receipt, the four other owners are not terms in it, and
 * the same request on a runtime that answers yes prepares. A refusal that
 * instead arrived from the device would arrive after the run had executed and
 * held a diagnostic bank, and would leave a manifest that cannot be told from a
 * capture which found nothing.
 *
 * **Scope: the sim entry, and the runtime's answer is this target's.** Every
 * target here builds `sim/host/c_api_shared.cpp`, and `tests/ut/cpp` links no
 * runtime maker (it is a CANN-free build), so the answer is the strong
 * definition below rather than a production runtime's. Which answer each of the
 * four runtimes actually gives is stated in its own `runtime_maker.cpp` and
 * qualified by that backend's `test_dfx_joined_launch_attribution.py`: the
 * three that capture assert the records' content, and a5 `host_build_graph`
 * asserts this refusal over the real public API.
 */

#include "runtime_c_api.h"

#include <gtest/gtest.h>

#include "device_runner_base.h"
#include "native_run_context.h"
#include "native_run_execution.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <vector>

/* ===========================================================================
 * The runtime-maker half, which this target does not link. @see
 * tests/ut/cpp/support/weak_link_placeholders.cpp — strong definitions here
 * override the weak ones for this target alone, and each reports success
 * because a preparation that reaches them is the subject of its own suite.
 * =========================================================================== */
static int g_release_rc = 0;

extern "C" {

int bind_callable_to_runtime_impl(
    Runtime * /*runtime*/, const HostApi * /*api*/, const ChipStorageTaskArgs * /*orch_args*/,
    void * /*host_orch_func_ptr*/, const ArgDirection * /*signature*/, int /*sig_count*/,
    const uint64_t * /*ring_task_window*/, const uint64_t * /*ring_heap*/, const uint64_t * /*ring_dep_pool*/
) {
    return 0;
}

int publish_run_image_impl(Runtime * /*runtime*/, const HostApi * /*api*/) { return 0; }

int copy_in_run_inputs_impl(const Runtime * /*runtime*/, const HostApi * /*api*/) { return 0; }

int copy_back_run_outputs_impl(
    const Runtime * /*runtime*/, const HostApi * /*api*/, int /*execution_rc*/, int /*launched*/
) {
    return 0;
}

int release_run_bindings_impl(Runtime * /*runtime*/, const HostApi * /*api*/) { return g_release_rc; }

}  // extern "C"

/** The answer this target's "runtime" gives, which each case chooses. */
static int g_capture_supported = 0;

extern "C" int args_dump_capture_supported_impl(void) { return g_capture_supported; }

namespace {

constexpr int32_t kCallableId = 23;
constexpr int kSimDeviceId = 0;

/**
 * A sim runner that can be prepared against and never launched.
 *
 * Everything the entry reads — the callable registry, the run-resource grant,
 * the execution claim and the native-run carrier — is the production base's.
 * The launch side is never reached by these cases and refuses, so a case that
 * accidentally launched would fail rather than pass on a double's leniency.
 */
class PreparableRunner : public SimDeviceRunnerBase {
public:
    PreparableRunner() {
        // The two registration facts a real preparation checks: the callable is
        // registered, and the retained block its hash names exists.
        callables_[kCallableId] = CallableState{};
        chip_callable_buffers_[CallableState{}.chip_buffer_hash] = ChipCallableBuffer{};
    }

    bool stand_up() {
        if (attach_current_thread(kSimDeviceId) != 0) return false;
        return set_run_resource_grant(1);
    }

    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *prepared
    ) override {
        *prepared = std::make_unique<PreparedExecution>(identity, runtime, config, pipeline_slot);
        prepare_calls++;
        return 0;
    }

    LaunchOutcome launch_execution(std::unique_ptr<PreparedExecution> prepared, LaunchPermit /*permit*/) override {
        LaunchOutcome outcome;
        outcome.rc = PTO_RUNTIME_ERR_INTERNAL;
        outcome.progress = LaunchProgress::NotStarted;
        outcome.prepared = std::move(prepared);
        return outcome;
    }

    void abandon_prepared_execution(PreparedExecution & /*prepared*/) noexcept override {}
    int poll_execution(const ActiveExecution & /*active*/) override { return SIMPLER_NATIVE_RUN_POLL_COMPLETE; }
    DrainOutcome drain_execution(ActiveExecution & /*active*/) override { return DrainOutcome::device_complete(0); }
    int finalize() override { return 0; }
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs & /*reg_args*/) override { return 0; }

    int prepare_calls{0};
};

/** Runtime storage with the alignment and zeroing `simpler_prepare_run` requires. */
struct RuntimeStorage {
    std::vector<unsigned char> bytes;

    RuntimeStorage() :
        bytes(get_runtime_size() + get_runtime_alignment(), 0) {}

    void *handle() {
        auto address = reinterpret_cast<uintptr_t>(bytes.data());
        const size_t align = get_runtime_alignment();
        const uintptr_t aligned = (address + align - 1) & ~static_cast<uintptr_t>(align - 1);
        return reinterpret_cast<void *>(aligned);
    }
};

class ArgsDumpCapabilityRefusalTest : public ::testing::Test {
protected:
    void SetUp() override {
        g_capture_supported = 0;
        g_release_rc = 0;
        ASSERT_TRUE(runner.stand_up());
    }

    void TearDown() override {
        g_capture_supported = 0;
        g_release_rc = 0;
    }

    /** Drive the real sim `simpler_prepare_run` with this diagnostic request. */
    int prepare(const CallConfig &config, uint64_t epoch = 7701) {
        NativeRunDescriptor descriptor{};
        descriptor.pipeline_slot = 0;
        descriptor.arena_bank = 0;
        descriptor.run_id = 7700;
        descriptor.generation = 1;
        descriptor.dispatch_id = 3;
        descriptor.run_epoch = epoch;
        return simpler_prepare_run(
            static_cast<DeviceContextHandle>(&runner), storage.handle(), kCallableId, nullptr, &config, &descriptor
        );
    }

    int finalize() { return simpler_finalize_run(static_cast<DeviceContextHandle>(&runner), storage.handle()); }

    /** Whether the caller's storage holds a prepared run at all. */
    bool carrier_present() const {
        uint64_t magic = 0;
        std::memcpy(&magic, const_cast<RuntimeStorage &>(storage).handle(), sizeof(magic));
        return magic == kNativeRunContextMagic;
    }

    static CallConfig config_with(int dump_args_level, bool other_owners) {
        CallConfig config;
        config.enable_dump_args = dump_args_level;
        if (other_owners) {
            config.enable_chip_swimlane = 1;
            config.enable_pmu = 2;
            config.enable_dep_gen = 1;
            config.enable_scope_stats = 1;
        }
        std::snprintf(config.output_prefix, sizeof(config.output_prefix), "%s", "args-dump-capability");
        return config;
    }

    PreparableRunner runner;
    RuntimeStorage storage;
};

// The request names a capture this runtime has no producer for, so it is
// refused as unsupported rather than prepared. No live carrier remains in the
// storage and the runner was never asked to prepare an execution: a code is not
// enough on its own, because a refusal that had already built a carrier would
// leave the slot owning a run nobody finalizes.
TEST_F(ArgsDumpCapabilityRefusalTest, RefusesAnArgumentDumpNoProducerCanCollect) {
    for (const int level : {1, 2, 3}) {
        const int rc = prepare(config_with(level, /*other_owners=*/false));
        EXPECT_EQ(rc, PTO_RUNTIME_ERR_UNSUPPORTED) << "enable_dump_args=" << level << " was not refused as unsupported";
        EXPECT_FALSE(carrier_present()) << "enable_dump_args=" << level << " left a prepared run behind";
        EXPECT_EQ(runner.prepare_calls, 0) << "enable_dump_args=" << level << " reached the runner";
    }
}

// Only this owner is a term. The same runtime serves a request that enables the
// other four, so an unavailable capture does not make a backend's remaining
// diagnostics unavailable with it.
TEST_F(ArgsDumpCapabilityRefusalTest, ServesTheOtherOwnersOnTheSameRuntime) {
    ASSERT_EQ(prepare(config_with(/*dump_args_level=*/0, /*other_owners=*/true)), 0);
    EXPECT_TRUE(carrier_present());
    EXPECT_EQ(runner.prepare_calls, 1);
    EXPECT_EQ(finalize(), 0);
}

// And the refusal is the runtime's answer rather than the flag's: the identical
// request prepares where a producer exists, so a backend that captures keeps its
// argument dump.
TEST_F(ArgsDumpCapabilityRefusalTest, PreparesTheSameRequestWhereAProducerExists) {
    g_capture_supported = 1;
    ASSERT_EQ(prepare(config_with(/*dump_args_level=*/3, /*other_owners=*/true)), 0);
    EXPECT_TRUE(carrier_present());
    EXPECT_EQ(runner.prepare_calls, 1);
    EXPECT_EQ(finalize(), 0);
}

// A run with no diagnostics at all never consults the answer, so an unsupported
// capture costs the ordinary path nothing.
TEST_F(ArgsDumpCapabilityRefusalTest, LeavesTheNoDiagnosticsPathAlone) {
    CallConfig quiet;
    NativeRunDescriptor descriptor{};
    descriptor.pipeline_slot = 0;
    descriptor.arena_bank = 0;
    descriptor.run_id = 7800;
    descriptor.generation = 1;
    descriptor.dispatch_id = 4;
    descriptor.run_epoch = 7801;
    ASSERT_EQ(
        simpler_prepare_run(
            static_cast<DeviceContextHandle>(&runner), storage.handle(), kCallableId, nullptr, &quiet, &descriptor
        ),
        0
    );
    EXPECT_TRUE(carrier_present());
    EXPECT_EQ(finalize(), 0);
}

TEST_F(ArgsDumpCapabilityRefusalTest, UnsupportedReceiptNamesOnlyThisEpochAndAllowsPrepareFlushClose) {
    ASSERT_EQ(prepare(config_with(1, false)), PTO_RUNTIME_ERR_UNSUPPORTED);
    EXPECT_FALSE(carrier_present());
    EXPECT_EQ(runner.prepare_calls, 0);
    EXPECT_FALSE(consume_released_preparation(storage.handle(), get_runtime_size(), 7702));
    ASSERT_TRUE(consume_released_preparation(storage.handle(), get_runtime_size(), 7701));
    EXPECT_FALSE(consume_released_preparation(storage.handle(), get_runtime_size(), 7701));
    ASSERT_EQ(prepare(CallConfig{}, 7702), 0);
    EXPECT_EQ(runner.prepare_calls, 1);
    ASSERT_EQ(finalize(), 0);
    char error[256]{};
    EXPECT_EQ(simpler_flush_diagnostics_ctx(&runner, 1000, error, sizeof(error)), 0) << error;
    EXPECT_EQ(finalize_device(&runner), 0);
}

TEST_F(ArgsDumpCapabilityRefusalTest, UnsupportedRequestPreservesLiveCarrierWithoutReleaseProof) {
    ASSERT_EQ(prepare(CallConfig{}), 0);
    const auto before = storage.bytes;
    EXPECT_EQ(prepare(config_with(1, false), 7702), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(storage.bytes, before);
    EXPECT_FALSE(consume_released_preparation(storage.handle(), get_runtime_size(), 7702));
    EXPECT_TRUE(carrier_present());
    RuntimeStorage other;
    NativeRunDescriptor descriptor{};
    descriptor.run_epoch = 7702;
    const auto unsupported = config_with(1, false);
    EXPECT_EQ(
        simpler_prepare_run(&runner, other.handle(), kCallableId, nullptr, &unsupported, &descriptor),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_FALSE(consume_released_preparation(other.handle(), get_runtime_size(), 7702));
    EXPECT_EQ(storage.bytes, before);
    EXPECT_EQ(finalize(), 0);
}

TEST_F(ArgsDumpCapabilityRefusalTest, UnsupportedRequestDoesNotCertifyInvalidStorageOrDescriptor) {
    const auto config = config_with(1, false);
    NativeRunDescriptor descriptor{};
    descriptor.run_epoch = 7701;
    const auto untouched = storage.bytes;
    EXPECT_EQ(
        simpler_prepare_run(nullptr, storage.handle(), kCallableId, nullptr, &config, &descriptor),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(
        simpler_prepare_run(
            &runner, static_cast<unsigned char *>(storage.handle()) + 1, kCallableId, nullptr, &config, &descriptor
        ),
        PTO_RUNTIME_ERR_INTERNAL
    );
    descriptor.pipeline_slot = 1;  // outside this runner's one-set grant
    EXPECT_EQ(
        simpler_prepare_run(&runner, storage.handle(), kCallableId, nullptr, &config, &descriptor),
        PTO_RUNTIME_ERR_INTERNAL
    );
    EXPECT_EQ(storage.bytes, untouched);
    EXPECT_FALSE(consume_released_preparation(storage.handle(), get_runtime_size(), 7701));
    const uint64_t foreign = 0x12345678;
    std::memcpy(storage.handle(), &foreign, sizeof(foreign));
    const auto foreign_storage = storage.bytes;
    EXPECT_EQ(prepare(config), PTO_RUNTIME_ERR_INTERNAL);
    EXPECT_EQ(storage.bytes, foreign_storage);
    EXPECT_FALSE(consume_released_preparation(storage.handle(), get_runtime_size(), 7701));
    EXPECT_EQ(runner.prepare_calls, 0);
}

TEST_F(ArgsDumpCapabilityRefusalTest, UnsupportedCleanupFailureDoesNotPublishReleaseProof) {
    g_release_rc = -77;
    EXPECT_EQ(prepare(config_with(1, false)), -77);
    EXPECT_FALSE(carrier_present());
    EXPECT_FALSE(consume_released_preparation(storage.handle(), get_runtime_size(), 7701));
    EXPECT_EQ(runner.prepare_calls, 0);
}

}  // namespace
