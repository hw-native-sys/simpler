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
#include <acl/error_codes/rt_error_codes.h>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "runtime_c_api.h"
#include "call_config.h"
#include "task_args.h"

static void require(bool condition, const char *stage) {
    if (!condition) {
        std::fprintf(stderr, "kernel-stream failed: %s\n", stage);
        std::fflush(nullptr);
        std::_Exit(1);
    }
}

static void check(int rc, const char *stage) {
    if (rc != 0) std::fprintf(stderr, "kernel-stream %s rc=%d\n", stage, rc);
    require(rc == 0, stage);
}

static std::vector<uint8_t> read(const char *path) {
    std::ifstream in(path, std::ios::binary);
    require(in.good(), path);
    return {std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>()};
}

int main(int argc, char **argv) {
    require(argc == 8, "usage: device capture error callable cpu core dispatcher");
    const int device = std::atoi(argv[1]);
    const bool capture = std::atoi(argv[2]) != 0;
    const int code = std::atoi(argv[3]);
    auto callable = read(argv[4]), cpu = read(argv[5]), core = read(argv[6]), dispatcher = read(argv[7]);
    check(aclInit(nullptr), "acl-init");
    check(aclrtSetDevice(device), "set-device");
    aclrtStream caller = nullptr;
    check(aclrtCreateStream(&caller), "caller-stream");
    void *output = nullptr;
    check(aclrtMalloc(&output, sizeof(int32_t), ACL_MEM_MALLOC_HUGE_FIRST), "output");
    const int32_t poison = 77;
    check(aclrtMemcpy(output, sizeof(poison), &poison, sizeof(poison), ACL_MEMCPY_HOST_TO_DEVICE), "poison");
    auto ctx = create_device_context();
    require(ctx != nullptr, "context");
    CallConfig config{};
    config.aicpu_thread_num = 2;
    config.enable_chip_swimlane = 0;
    check(
        simpler_kernel_mode_init(
            ctx, device, cpu.data(), cpu.size(), core.data(), core.size(), dispatcher.data(), dispatcher.size(),
            &config, 1
        ),
        "kernel-init"
    );
    check(simpler_kernel_mode_prepare_callable(ctx, 0, callable.data(), callable.size()), "register");
    std::vector<uint8_t> storage(get_runtime_size() + get_runtime_alignment());
    void *run = reinterpret_cast<void *>(
        (reinterpret_cast<uintptr_t>(storage.data()) + get_runtime_alignment() - 1) & ~(get_runtime_alignment() - 1)
    );
    const uint32_t shape[]{1};
    ChipStorageTaskArgs args;
    args.add_tensor(
        make_tensor_external(output, shape, 1, DataType::INT32, AddressSpace::DEVICE, TensorTransfer::NONE)
    );
    NativeRunDescriptor descriptor{};
    descriptor.generation = descriptor.run_epoch = 1;
    descriptor.prepare_purpose = capture ? SIMPLER_PREPARE_CAPTURE : SIMPLER_PREPARE_EAGER;
    check(simpler_prepare_run(ctx, run, 0, &args, &config, &descriptor), "prepare");
    RuntimeHandle reference = nullptr;
    aclmdlRI model = nullptr;
    if (capture) {
        check(simpler_retain_run_resources(ctx, run, caller, &reference), "retain");
        check(aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture-begin");
    }
    check(simpler_launch_run_on_stream(ctx, run, caller, nullptr), "launch");
    if (capture) {
        check(aclmdlRICaptureEnd(caller, &model), "capture-end");
        check(simpler_finalize_run(ctx, run), "finalize-recording");
        int32_t observed = 0;
        check(
            aclrtMemcpy(&observed, sizeof(observed), output, sizeof(observed), ACL_MEMCPY_DEVICE_TO_HOST),
            "read-recording"
        );
        require(observed == poison, "recording-does-not-execute");
        check(aclmdlRIExecuteAsync(model, caller), "replay");
    }
    // The caller observes its stream before invoking any simpler wait/finalize
    // that could interpret a private computational result.
    const int stream_rc = aclrtSynchronizeStreamWithTimeout(caller, 15000);
    std::printf("KERNEL_STREAM_RESULT capture=%d code=%d stream_rc=%d\n", capture, code, stream_rc);
    if (code != 0) {
        require(
            stream_rc == (capture ? ACL_ERROR_RT_MODEL_EXECUTE : ACL_ERROR_RT_AICPU_EXCEPTION),
            "failed-computation-must-fail-caller-stream"
        );
        // A failed stream supplies no last-use proof. All device and model
        // owners stay alive until this isolated process exits.
        std::puts("KERNEL_STREAM_PASS owners=retained observer=caller-stream");
        std::fflush(nullptr);
        std::_Exit(0);
    }
    check(stream_rc, "success-sync");
    int32_t observed = 0;
    check(aclrtMemcpy(&observed, sizeof(observed), output, sizeof(observed), ACL_MEMCPY_DEVICE_TO_HOST), "read-output");
    require(observed == 1, "success-consumer-output");
    if (capture) {
        check(aclmdlRIDestroy(model), "destroy-model");
        check(simpler_release_run_resources(ctx, reference), "release-model-resources");
    } else {
        check(simpler_wait_run(ctx, run), "wait");
        check(simpler_finalize_run(ctx, run), "finalize");
    }
    check(simpler_unregister_callable(ctx, 0), "unregister");
    check(finalize_device(ctx), "finalize-device");
    destroy_device_context(ctx);
    check(aclrtFree(output), "free-output");
    check(aclrtDestroyStream(caller), "destroy-stream");
    check(aclrtResetDevice(device), "reset-device");
    check(aclFinalize(), "acl-finalize");
    std::puts("KERNEL_STREAM_PASS owners=released output=1");
}
