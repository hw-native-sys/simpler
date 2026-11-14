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

#pragma once
#include <acl/acl.h>
#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <vector>
#include "call_config.h"
#include "runtime_c_api.h"
#include "task_args.h"
#include "capture_capacity_observer.h"

namespace simpler::capture_test {
constexpr uint64_t workspace_budget = 256ULL << 20;
constexpr size_t backing_elements = 64;
constexpr size_t elements = 16;
constexpr size_t bytes = backing_elements * sizeof(int32_t);
using Values = std::array<int32_t, backing_elements>;
inline void check(int rc, const char *stage) {
    std::printf("stage=%s rc=%d\n", stage, rc);
    std::fflush(stdout);
    if (rc != 0) std::exit(1);
}
inline void require(bool value, const char *stage) { check(value ? 0 : -1, stage); }
inline std::vector<uint8_t> read(const char *path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), "binary-input");
    return {std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};
}
struct Model {
    aclmdlRI model = nullptr;
    RuntimeHandle reference = nullptr;
    void *output = nullptr;
    size_t offset = 0;
};
class CaptureFixture {
public:
    DeviceContextHandle context = nullptr;
    aclrtStream caller = nullptr;
    CallConfig config{};
    std::vector<Model> models;
    RuntimeHandle run = nullptr;
    volatile int32_t accepted = 0;
    uint64_t epoch = 0;
    bool tmr;
    void *input = nullptr;
    aclrtContext owner = nullptr;
    std::vector<uint8_t> storage;

    CaptureFixture(int argc, char **argv, size_t model_count = 3) :
        models(model_count),
        tmr(false) {
        require(argc == 7, "usage-device-callable-cpu-core-dispatcher-runtime");
        tmr = std::strcmp(argv[6], "tensormap_and_ringbuffer") == 0;
        const auto callable = read(argv[2]), cpu = read(argv[3]), core = read(argv[4]), dispatcher = read(argv[5]);
        config.aicpu_thread_num = 2;
        if (tmr) {
            for (int ring = 0; ring < RUNTIME_ENV_RING_COUNT; ++ring) {
                config.runtime_env.ring_task_window[ring] = 64;
                config.runtime_env.ring_heap[ring] = 1ULL << 20;
                config.runtime_env.ring_dep_pool[ring] = 256;
            }
        }
        const int device = std::atoi(argv[1]);
        check(aclInit(nullptr), "acl-init");
        check(aclrtSetDevice(device), "set-device");
        check(aclrtCreateContext(&owner, device), "caller-context");
        check(aclrtCreateStream(&caller), "caller-stream");
        check(aclrtMalloc(&input, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "input");
        context = create_device_context();
        require(context != nullptr, "context");
        check(simpler_set_workspace_budget_ctx(context, workspace_budget), "finite-workspace-budget");
        check(simpler_set_run_resource_grant_ctx(context, 2), "grant-two");
        check(
            simpler_kernel_mode_init(
                context, device, cpu.data(), cpu.size(), core.data(), core.size(), dispatcher.data(), dispatcher.size(),
                &config, 1
            ),
            "kernel-init"
        );
        check(simpler_kernel_mode_prepare_callable(context, 0, callable.data(), callable.size()), "register");
        storage.resize(get_runtime_size() + get_runtime_alignment());
        run = reinterpret_cast<void *>(
            (reinterpret_cast<uintptr_t>(storage.data()) + get_runtime_alignment() - 1) & ~(get_runtime_alignment() - 1)
        );
        for (size_t i = 0; i < models.size(); ++i) {
            models[i].offset = i % 2 == 1 ? 32 : 0;
            check(aclrtMalloc(&models[i].output, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "output");
            poison(models[i]);
        }
    }
    void poison(const Model &model) {
        Values values{};
        values.fill(-999);
        check(aclrtMemcpy(model.output, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "poison-output");
    }
    Values output(const Model &model) {
        Values values{};
        check(aclrtMemcpy(values.data(), bytes, model.output, bytes, ACL_MEMCPY_DEVICE_TO_HOST), "read-output");
        return values;
    }
    void unchanged(const Model &model) {
        const auto values = output(model);
        const bool clean = std::all_of(values.begin(), values.end(), [](int32_t value) {
            return value == -999;
        });
        require(clean, "unexecuted-output");
    }
    void seed(int round) {
        Values values{};
        for (size_t i = 0; i < values.size(); ++i)
            values[i] = round * 100 + static_cast<int32_t>(i);
        check(aclrtMemcpy(input, bytes, values.data(), bytes, ACL_MEMCPY_HOST_TO_DEVICE), "seed-input");
    }
    void verify(const Model &model, int round) {
        const auto values = output(model);
        bool correct = true;
        for (size_t i = 0; i < values.size(); ++i) {
            const auto expected =
                i >= model.offset && i < model.offset + elements ? round * 100 + static_cast<int32_t>(i) + 10 : -999;
            if (values[i] != expected) {
                std::fprintf(stderr, "element=%zu expected=%d actual=%d\n", i, expected, values[i]);
                correct = false;
            }
        }
        require(correct, "full-values-and-view-guards");
    }
    int prepare(Model &model, uint64_t workspace_bytes = 0) {
        std::memset(run, 0, get_runtime_size());
        uint32_t shape[]{elements};
        ChipTensor in{}, out{};
        in.init_external(input, bytes, shape, 1, DataType::INT32, AddressSpace::DEVICE, TensorTransfer::NONE);
        out.init_external(model.output, bytes, shape, 1, DataType::INT32, AddressSpace::DEVICE, TensorTransfer::NONE);
        in.start_offset = out.start_offset = model.offset;
        ChipStorageTaskArgs args;
        args.add_tensor(in);
        args.add_tensor(out);
        args.add_scalar(workspace_bytes != 0 && !tmr ? workspace_bytes / sizeof(int32_t) : elements);
        auto current = config;
        if (workspace_bytes != 0 && tmr) {
            for (int ring = 0; ring < RUNTIME_ENV_RING_COUNT; ++ring)
                current.runtime_env.ring_heap[ring] = workspace_bytes / RUNTIME_ENV_RING_COUNT;
        }
        NativeRunDescriptor descriptor{};
        descriptor.generation = descriptor.run_epoch = descriptor.dispatch_id = ++epoch;
        descriptor.prepare_purpose = SIMPLER_PREPARE_CAPTURE;
        descriptor.accepted_state = &accepted;
        descriptor.accepted_value = 1;
        accepted = 0;
        return simpler_prepare_run(context, run, 0, &args, &current, &descriptor);
    }
    void record(Model &model) {
        check(simpler_retain_run_resources(context, run, caller, &model.reference), "retain");
        check(aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture-begin");
        check(simpler_launch_run_on_stream(context, run, caller, nullptr), "capture-submit");
        check(aclmdlRICaptureEnd(caller, &model.model), "capture-end");
        require(model.model != nullptr, "model");
        check(simpler_finalize_run(context, run), "retire-host-carrier");
        unchanged(model);
    }
    void replay(Model &model, int round) {
        seed(round);
        poison(model);
        check(aclmdlRIExecuteAsync(model.model, caller), "replay-enqueue");
        check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "replay-complete");
        verify(model, round);
    }
    void release(Model &model) {
        check(aclmdlRIDestroy(model.model), "destroy-model");
        model.model = nullptr;
        check(simpler_release_run_resources(context, model.reference), "release-reference");
        model.reference = nullptr;
    }
    SimplerWorkspaceReport report(const char *phase) {
        SimplerWorkspaceReport report{};
        check(simpler_get_workspace_report_ctx(context, &report, sizeof(report)), "workspace-report");
        require(
            report.schema == WORKSPACE_REPORT_SCHEMA && report.budget_enforced && report.coverage_is_partial &&
                report.limit_bytes == workspace_budget && report.reserved_bytes <= report.limit_bytes &&
                !report.proof_unavailable && report.relinquished_bytes == 0 && report.quarantined_blocks == 0 &&
                report.quarantined_mapped_bytes == 0 && report.release_unconfirmed_blocks == 0,
            "budget-accounting"
        );
        std::printf(
            "workspace phase=%s limit=%llu reserved=%llu partial=1 live_blocked=%u\n", phase,
            static_cast<unsigned long long>(report.limit_bytes), static_cast<unsigned long long>(report.reserved_bytes),
            report.live_blocked
        );
        return report;
    }
    void close() {
        for (const auto &model : models)
            require(model.model == nullptr && model.reference == nullptr, "no-live-model");
        check(simpler_unregister_callable(context, 0), "unregister");
        check(finalize_device(context), "context-finalize");
        require(report("finalized").reserved_bytes == 0, "workspace-reclaimed");
        destroy_device_context(context);
        for (const auto &model : models)
            check(aclrtFree(model.output), "free-output");
        check(aclrtFree(input), "free-input");
        check(aclrtDestroyStream(caller), "destroy-stream");
        check(aclrtDestroyContext(owner), "destroy-context");
        check(aclFinalize(), "acl-finalize");
    }
};
}  // namespace simpler::capture_test
