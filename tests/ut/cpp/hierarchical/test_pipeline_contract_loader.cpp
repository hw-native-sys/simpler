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
#include <dlfcn.h>
#include <stdexcept>
#include <string>
#include <gtest/gtest.h>

#include "chip_worker.h"
#include "chip_run_lane.h"
#include "pipeline_contract.h"

namespace {

template <typename T>
T symbol(void *handle, const char *name) {
    dlerror();
    void *result = dlsym(handle, name);
    const char *error = dlerror();
    if (error != nullptr || result == nullptr) {
        throw std::runtime_error(error != nullptr ? error : name);
    }
    return reinterpret_cast<T>(result);
}

class LoadedRuntime {
public:
    explicit LoadedRuntime(const char *path, int flags = RTLD_NOW | RTLD_LOCAL) {
        handle = dlopen(path, flags);
        if (!handle) throw std::runtime_error(dlerror());
    }
    ~LoadedRuntime() { dlclose(handle); }
    LoadedRuntime(const LoadedRuntime &) = delete;
    LoadedRuntime &operator=(const LoadedRuntime &) = delete;
    void *handle;
};

PipelineContract program_contract() {
    return {
        PTO_PIPELINE_CONTRACT_ABI_VERSION,
        6,
        2,
        {
            {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_DEVICE_SCRATCH, 0},
            {PTO_PIPELINE_GM_SM, PTO_PIPELINE_DEVICE_SCRATCH, 0},
            {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_DEVICE_SCRATCH, 0},
            {PTO_PIPELINE_TASK_ARGS, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
        }
    };
}

void expect_factory_admission(const PipelineContract &contract, int expected_creates, const char *message) {
    LoadedRuntime fixture(PIPELINE_FIXTURE_PATH);
    auto set_contract = symbol<void (*)(const PipelineContract *)>(fixture.handle, "test_set_pipeline_contract");
    auto create_count = symbol<int (*)()>(fixture.handle, "test_context_create_count");
    set_contract(&contract);
    {
        ChipWorker worker;
        try {
            worker.init(PIPELINE_FIXTURE_PATH, "", "", "", 0);
            FAIL() << "The fixture factory must fail after admission";
        } catch (const std::runtime_error &error) {
            EXPECT_STREQ(error.what(), message);
        }
    }
    EXPECT_EQ(create_count(), expected_creates);
}

}  // namespace

TEST(PipelineContractLoader, MissingOrMisclassifiedStreamsRejectBeforeCreatingContext) {
    for (uint32_t index : {4u, 5u}) {
        auto contract = program_contract();
        contract.resources[index] = contract.resources[3];
        ASSERT_TRUE(is_valid_pipeline_contract(&contract));
        expect_factory_admission(contract, 0, "host runtime returned a PipelineContract this build cannot accept");
        contract = program_contract();
        contract.resources[index].resource_class = PTO_PIPELINE_HOST_PER_RUN;
        expect_factory_admission(contract, 0, "host runtime returned a PipelineContract this build cannot accept");
    }
}

TEST(PipelineContractLoader, LegalProgramDeclarationReachesRealFactoryCall) {
    expect_factory_admission(program_contract(), 1, "create_device_context returned null");
}

TEST(KernelPipelineEntry, RealSimVariantsValidateWithoutClaimingOrCommitting) {
    // Sim hooks must remain global for every host runtime loaded beneath them.
    LoadedRuntime sim_context(SIM_CONTEXT_PATH, RTLD_NOW | RTLD_GLOBAL);
    for (const char *arch : {"a2a3", "a5"}) {
        for (const char *runtime : {"tensormap_and_ringbuffer", "host_build_graph"}) {
            const std::string path =
                std::string(RUNTIME_LIBRARY_ROOT) + "/" + arch + "/sim/" + runtime + "/libhost_runtime.so";
            SCOPED_TRACE(path);
            LoadedRuntime library(path.c_str());
            auto create = symbol<decltype(&create_device_context)>(library.handle, "create_device_context");
            auto destroy = symbol<decltype(&destroy_device_context)>(library.handle, "destroy_device_context");
            auto init = symbol<decltype(&simpler_kernel_mode_init)>(library.handle, "simpler_kernel_mode_init");
            auto supported =
                symbol<decltype(&simpler_kernel_mode_supported)>(library.handle, "simpler_kernel_mode_supported");
            auto launch = symbol<decltype(&simpler_kernel_mode_launch)>(library.handle, "simpler_kernel_mode_launch");
            auto committed =
                symbol<decltype(&committed_device_memory_ctx)>(library.handle, "committed_device_memory_ctx");
            DeviceContextHandle ctx = create();
            ASSERT_NE(ctx, nullptr);
            CallConfig config;
            auto invoke = [&](const CallConfig *input, uint64_t generation = 1) {
                return init(ctx, 0, nullptr, 0, nullptr, 0, nullptr, 0, input, generation);
            };
            EXPECT_EQ(invoke(nullptr), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
            EXPECT_EQ(invoke(&config, 0), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
            const uint8_t binary = 0;
            EXPECT_EQ(init(ctx, 0, &binary, 0, nullptr, 0, nullptr, 0, &config, 1), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
            EXPECT_EQ(init(ctx, 0, nullptr, 0, &binary, 0, nullptr, 0, &config, 1), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
            EXPECT_EQ(init(ctx, 0, nullptr, 0, nullptr, 0, &binary, 0, &config, 1), PTO_RUNTIME_ERR_INVALID_ARGUMENT);
            config.runtime_env.ring_task_window[0] = 3;
            EXPECT_EQ(
                invoke(&config), std::string(runtime) == "tensormap_and_ringbuffer" ? PTO_RUNTIME_ERR_INVALID_ARGUMENT :
                                                                                      PTO_RUNTIME_ERR_UNSUPPORTED
            );
            config.runtime_env.ring_task_window[0] = 0;
            EXPECT_EQ(invoke(&config), PTO_RUNTIME_ERR_UNSUPPORTED);
            EXPECT_EQ(supported(ctx), 0);
            EXPECT_EQ(committed(ctx), 0u);
            // A structurally valid launch still cannot run after unsupported init.
            int caller_stream_token = 0;
            EXPECT_EQ(launch(ctx, 0, &config, &caller_stream_token), PTO_RUNTIME_ERR_INVALID_STATE);
            EXPECT_EQ(invoke(&config), PTO_RUNTIME_ERR_UNSUPPORTED);
            EXPECT_EQ(committed(ctx), 0u);
            destroy(ctx);
        }
    }
}
