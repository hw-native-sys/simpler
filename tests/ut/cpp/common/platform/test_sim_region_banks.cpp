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
#include "device_runner_base.h"
#include "native_run_context.h"
#include "worker/pipeline_contract.h"

#include <algorithm>
#include <memory>
#include <vector>

namespace {
PipelineContract contract{};
constexpr int32_t kCallableId = 23;
constexpr int kSimDeviceId = 0;
}  // namespace

extern "C" const PipelineContract *get_pipeline_contract() { return &contract; }
extern "C" int bind_callable_to_runtime_impl(
    Runtime *, const HostApi *api, const ChipStorageTaskArgs *, void *, const ArgDirection *, int, const uint64_t *,
    const uint64_t *, const uint64_t *
) {
    return api->setup_static_arena(
        find_pipeline_resource(contract, PTO_PIPELINE_GM_HEAP) ? 64 : 0,
        find_pipeline_resource(contract, PTO_PIPELINE_GM_SM) ? 64 : 0, 64
    );
}
extern "C" int publish_run_image_impl(Runtime *, const HostApi *) { return 0; }
extern "C" int copy_in_run_inputs_impl(const Runtime *, const HostApi *) { return 0; }
extern "C" int copy_back_run_outputs_impl(const Runtime *, const HostApi *, int, int) { return 0; }
extern "C" int release_run_bindings_impl(Runtime *, const HostApi *) { return 0; }

namespace {
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
        return set_run_resource_grant(2);
    }

    int prepare_execution(
        Runtime &runtime, const CallConfig &config, uint32_t pipeline_slot, const NativeRunIdentity &identity,
        std::unique_ptr<PreparedExecution> *prepared
    ) override {
        *prepared = std::make_unique<PreparedExecution>(identity, runtime, config, pipeline_slot);
        return 0;
    }

    LaunchTransactionResult launch_execution_impl(PreparedExecution & /*prepared*/, LaunchPermit /*permit*/) override {
        LaunchTransactionResult result;
        result.rc = PTO_RUNTIME_ERR_INTERNAL;
        return result;
    }

    void abandon_prepared_execution(PreparedExecution & /*prepared*/) noexcept override {}
    int poll_execution(const ActiveExecution & /*active*/) override { return SIMPLER_NATIVE_RUN_POLL_COMPLETE; }
    DrainOutcome drain_execution(ActiveExecution & /*active*/) override { return DrainOutcome::device_complete(0); }
    int finalize() override { return 0; }
    int ensure_binaries_loaded() override { return 0; }
    int invoke_device_register(const RegisterCallableArgs & /*reg_args*/) override { return 0; }
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

TEST(SimRegionBanks, PrepareRoutesEachRegionFromTheContract) {
    // Includes absent heap and SM declarations, whose banks remain zero.
    const std::vector<std::vector<PipelineResource>> shapes = {
        {{PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_DELIVERED, 0}},
        {{PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_SCRATCH, 0},
         {PTO_PIPELINE_GM_SM, PTO_PIPELINE_DELIVERED, 0},
         {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_DELIVERED, 0}},
        {{PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_DELIVERED, 0},
         {PTO_PIPELINE_GM_SM, PTO_PIPELINE_SCRATCH, 0},
         {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_SCRATCH, 0}},
    };
    for (size_t shape = 0; shape < shapes.size(); ++shape) {
        SCOPED_TRACE(shape);
        contract = {PTO_PIPELINE_CONTRACT_ABI_VERSION, static_cast<uint32_t>(shapes[shape].size()), 2, {}};
        std::copy(shapes[shape].begin(), shapes[shape].end(), contract.resources);
        ASSERT_TRUE(is_valid_pipeline_contract(&contract));
        ASSERT_TRUE(has_serviceable_arena_topology(contract));
        PreparableRunner runner;
        ASSERT_TRUE(runner.stand_up());
        RuntimeStorage storage;
        CallConfig config{};
        NativeRunDescriptor descriptor{};
        descriptor.pipeline_slot = 1;
        descriptor.arena_bank = pipeline_resource_banks_for_slot(contract, 1).heap;
        descriptor.run_epoch = shape + 1;
        ASSERT_EQ(simpler_prepare_run(&runner, storage.handle(), kCallableId, nullptr, &config, &descriptor), 0);
        const auto *state = static_cast<const NativeRunContext<SimDeviceRunnerBase> *>(storage.handle());
        const RegionBanks expected[] = {{0, 0, 1}, {0, 1, 1}, {1, 0, 0}};
        const auto banks = expected[shape];
        EXPECT_EQ(state->host_api.acquire_pooled_gm_heap() != nullptr, shape != 0);
        EXPECT_EQ(state->host_api.acquire_pooled_gm_sm() != nullptr, shape != 0);
        EXPECT_NE(state->host_api.acquire_pooled_runtime_arena(), nullptr);
        EXPECT_EQ(state->host_api.acquire_pooled_gm_heap(), runner.acquire_pooled_gm_heap(banks.heap));
        EXPECT_EQ(state->host_api.acquire_pooled_gm_sm(), runner.acquire_pooled_gm_sm(banks.sm));
        EXPECT_EQ(state->host_api.acquire_pooled_runtime_arena(), runner.acquire_pooled_runtime_arena(banks.image));
        EXPECT_EQ(runner.acquire_pooled_gm_heap(1 - banks.heap), nullptr);
        EXPECT_EQ(runner.acquire_pooled_gm_sm(1 - banks.sm), nullptr);
        EXPECT_EQ(runner.acquire_pooled_runtime_arena(1 - banks.image), nullptr);
        EXPECT_EQ(simpler_finalize_run(&runner, storage.handle()), 0);
    }
}
}  // namespace
