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

#include <acl/acl.h>
#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "call_config.h"
#include "runtime_c_api.h"
#include "task_args.h"

namespace {
constexpr size_t elements = 16;
constexpr size_t bytes = elements * sizeof(int32_t);
using State = std::array<int32_t, elements>;

void check(int rc, const char *stage) {
    std::printf("stage=%s rc=%d\n", stage, rc);
    std::fflush(stdout);
    if (rc != 0) std::exit(1);
}
void require(bool condition, const char *stage) { check(condition ? 0 : -1, stage); }
std::vector<uint8_t> read(const char *path) {
    std::ifstream input(path, std::ios::binary);
    require(input.good(), path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
}  // namespace

int main(int argc, char **argv) {
    require(argc == 6 || argc == 8, "usage: device callable cpu core dispatcher [pmu-output replay]");
    const bool pmu = argc == 8;
    const bool execute_models = !pmu || std::atoi(argv[7]) != 0;
    const int device = std::atoi(argv[1]);
    const auto callable = read(argv[2]), cpu = read(argv[3]), core = read(argv[4]), dispatcher = read(argv[5]);
    check(aclInit(nullptr), "acl-init");
    check(aclrtSetDevice(device), "set-device");
    aclrtContext owner = nullptr;
    check(aclrtCreateContext(&owner, device), "owner-context");
    aclrtStream caller = nullptr;
    check(aclrtCreateStream(&caller), "caller-stream");
    std::array<void *, 3> buffers{};
    std::array<State, 3> expected{};
    const std::array<int32_t, 3> initial{10, 1000, -24680};
    for (size_t tensor = 0; tensor < buffers.size(); ++tensor) {
        check(aclrtMalloc(&buffers[tensor], bytes, ACL_MEM_MALLOC_HUGE_FIRST), "allocate-state");
        for (size_t element = 0; element < elements; ++element)
            expected[tensor][element] = initial[tensor] + static_cast<int32_t>(element * 7);
        check(
            aclrtMemcpy(buffers[tensor], bytes, expected[tensor].data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "seed-state"
        );
        std::printf("state-address index=%zu address=%p\n", tensor, buffers[tensor]);
    }
    require(
        buffers[0] != buffers[1] && buffers[0] != buffers[2] && buffers[1] != buffers[2], "distinct-state-addresses"
    );
    auto verify = [&](const char *stage) {
        for (size_t tensor = 0; tensor < buffers.size(); ++tensor) {
            State actual{};
            check(aclrtMemcpy(actual.data(), bytes, buffers[tensor], bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read-state");
            for (size_t element = 0; element < elements; ++element) {
                if (actual[element] != expected[tensor][element]) {
                    std::fprintf(
                        stderr, "state=%s tensor=%zu element=%zu expected=%d actual=%d\n", stage, tensor, element,
                        expected[tensor][element], actual[element]
                    );
                    std::exit(1);
                }
            }
        }
        std::printf("state=%s verified=48\n", stage);
    };
    auto ctx = create_device_context();
    require(ctx != nullptr, "native-context");
    CallConfig config{};
    config.aicpu_thread_num = 2;
    check(
        simpler_kernel_mode_init(
            ctx, device, cpu.data(), cpu.size(), core.data(), core.size(), dispatcher.data(), dispatcher.size(),
            &config, 1
        ),
        "kernel-init"
    );
    check(simpler_kernel_mode_prepare_callable(ctx, 0, callable.data(), callable.size()), "register-a");
    check(simpler_kernel_mode_prepare_callable(ctx, 1, callable.data(), callable.size()), "register-b");
    const size_t runtime_size = get_runtime_size(), alignment = get_runtime_alignment();
    std::vector<uint8_t> storage(runtime_size + alignment);
    auto run =
        reinterpret_cast<void *>((reinterpret_cast<uintptr_t>(storage.data()) + alignment - 1) & ~(alignment - 1));
    const auto original_run = run;
    uint64_t identity = 0;
    auto prepare = [&](int callable_id, size_t tensor, int32_t increment, bool capture) {
        if (pmu) {
            config.enable_pmu = capture ? (callable_id == 0 ? 2 : 4) : 7;
            const char *leaf = capture ? (callable_id == 0 ? "a" : "b") : "eager";
            std::snprintf(config.output_prefix, sizeof(config.output_prefix), "%s/%s", argv[6], leaf);
        }
        std::memset(run, 0, runtime_size);
        uint32_t shape[]{elements};
        ChipStorageTaskArgs args;
        args.add_tensor(
            make_tensor_external(buffers[tensor], shape, 1, DataType::INT32, AddressSpace::DEVICE, TensorTransfer::NONE)
        );
        args.add_scalar(increment);
        NativeRunDescriptor descriptor{};
        descriptor.generation = descriptor.run_epoch = descriptor.dispatch_id = ++identity;
        descriptor.prepare_purpose = capture ? SIMPLER_PREPARE_CAPTURE : SIMPLER_PREPARE_EAGER;
        require(run == original_run && descriptor.pipeline_slot == 0, "same-carrier-and-slot");
        check(simpler_prepare_run(ctx, run, callable_id, &args, &config, &descriptor), "prepare");
    };
    auto record = [&](RuntimeHandle &reference) {
        check(simpler_retain_run_resources(ctx, run, caller, &reference), "retain");
        require(reference != nullptr, "retained-reference");
        aclmdlRI model = nullptr;
        check(aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture-begin");
        check(simpler_launch_run_on_stream(ctx, run, caller, nullptr), "capture-launch");
        check(aclmdlRICaptureEnd(caller, &model), "capture-end");
        require(model != nullptr, "captured-model");
        check(simpler_finalize_run(ctx, run), "finalize-original");
        std::memset(run, 0xA5, runtime_size);
        return model;
    };
    auto replay = [&](aclmdlRI model, size_t tensor, int32_t increment, const char *stage) {
        check(aclmdlRIExecuteAsync(model, caller), "model-execute");
        check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "model-sync");
        for (auto &value : expected[tensor])
            value += increment;
        verify(stage);
    };
    RuntimeHandle reference_a = nullptr, reference_b = nullptr;
    prepare(0, 0, 3, true);
    verify("a-prepared");
    const auto model_a = record(reference_a);
    verify("a-recorded");
    require(simpler_unregister_callable(ctx, 0) == PTO_RUNTIME_ERR_INVALID_STATE, "a-retains-callable");
    require(finalize_device(ctx) == PTO_RUNTIME_ERR_INVALID_STATE, "a-retains-context");
    prepare(1, 1, 5, true);
    verify("b-prepared");
    const auto model_b = record(reference_b);
    require(model_a != model_b && reference_a != reference_b, "independent-models-and-references");
    verify("b-recorded");
    if (execute_models) {
        replay(model_b, 1, 5, "b1");
        replay(model_a, 0, 3, "a1");
        replay(model_b, 1, 5, "b2");
        replay(model_a, 0, 3, "a2");
        replay(model_b, 1, 5, "b3");
    }
    check(aclmdlRIDestroy(model_a), "destroy-a");
    check(simpler_release_run_resources(ctx, reference_a), "release-a");
    check(simpler_unregister_callable(ctx, 0), "unregister-a");
    require(simpler_unregister_callable(ctx, 1) == PTO_RUNTIME_ERR_INVALID_STATE, "b-retains-callable");
    require(finalize_device(ctx) == PTO_RUNTIME_ERR_INVALID_STATE, "b-retains-context");
    if (execute_models) replay(model_b, 1, 5, "b-after-a-release");
    check(aclmdlRIDestroy(model_b), "destroy-b");
    check(simpler_release_run_resources(ctx, reference_b), "release-b");
    check(simpler_unregister_callable(ctx, 1), "unregister-b");
    check(simpler_kernel_mode_prepare_callable(ctx, 0, callable.data(), callable.size()), "register-eager");
    prepare(0, 2, 17, false);
    check(simpler_launch_run_on_stream(ctx, run, caller, nullptr), "eager-launch");
    check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "eager-sync");
    check(simpler_finalize_run(ctx, run), "eager-finalize");
    for (auto &value : expected[2])
        value += 17;
    verify("eager");
    if (pmu) {
        std::array<char, 4096> error{};
        const int flush_rc = simpler_flush_diagnostics_ctx(ctx, 60000, error.data(), error.size());
        if (flush_rc != 0) std::fprintf(stderr, "PMU flush: %s\n", error.data());
        check(flush_rc, "pmu-flush");
    }
    check(simpler_unregister_callable(ctx, 0), "unregister-eager");
    check(finalize_device(ctx), "finalize-context");
    destroy_device_context(ctx);
    for (auto buffer : buffers)
        check(aclrtFree(buffer), "free-state");
    check(aclrtDestroyStream(caller), "destroy-caller");
    check(aclrtDestroyContext(owner), "destroy-owner");
    check(aclrtResetDevice(device), "reset-device");
    check(aclFinalize(), "acl-finalize");
    std::printf("hbg_retained_models PASS models=2 references=2 replays=6 eager=1 elements=16\n");
}
