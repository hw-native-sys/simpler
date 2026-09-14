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
#include <runtime/rt.h>

#include <array>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iterator>
#include <vector>

#include "args.h"
#include "host/kernel_execution_state.h"
#include "kernel_platform_ops.h"
#include "load_aicpu_op.h"

namespace {
int forbidden_calls = 0;

void check(int rc, const char *stage) {
    if (rc == 0) return;
    std::fprintf(stderr, "capture probe: %s rc=%d\n", stage, rc);
    std::_Exit(1);
}

void require(bool valid, const char *message) { check(valid ? 0 : -1, message); }

std::vector<char> read_binary(const char *path) {
    std::ifstream file(path, std::ios::binary);
    require(file.good(), path);
    std::vector<char> bytes{std::istreambuf_iterator<char>(file), {}};
    require(!bytes.empty(), "empty binary");
    return bytes;
}

void core_launch(void *binary, aclrtStream stream, CaptureArgs args) {
    rtArgsEx_t rt_args{};
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);
    rtTaskCfgInfo_t cfg{};
    cfg.schemMode = RT_SCHEM_MODE_BATCH;
    check(rtKernelLaunchWithHandleV2(binary, 0, 1, &rt_args, nullptr, stream, &cfg), "AICore launch");
}

void record(KernelExecutionState &state, KernelEventKind kind, aclrtStream stream) {
    check(aclrtRecordEvent(state.event(kind), stream), "record event");
}

void wait(KernelExecutionState &state, KernelEventKind kind, aclrtStream stream) {
    check(aclrtStreamWaitEvent(stream, state.event(kind)), "wait event");
}

// Each scalar occupies its own cache line: both kernels write independently.
struct Buffers {
    std::array<int64_t *, 5> slots{};
    Buffers() {
        for (auto &slot : slots) {
            void *allocation = nullptr;
            check(aclrtMalloc(&allocation, 64, ACL_MEM_MALLOC_HUGE_FIRST), "malloc");
            slot = static_cast<int64_t *>(allocation);
        }
    }
    ~Buffers() {
        for (auto slot : slots)
            check(aclrtFree(slot), "free");
    }
    void seed(int64_t value) {
        check(aclrtMemcpy(slots[0], 64, &value, sizeof(value), ACL_MEMCPY_HOST_TO_DEVICE), "seed");
        for (size_t i = 1; i < slots.size(); ++i)
            check(aclrtMemset(slots[i], 64, 0xA5, 64), "poison output");
    }
    void verify(int64_t value) {
        const std::array<int64_t, 5> expected{value, value + 3, value + 10, value + 14, 2 * value + 24};
        for (size_t i = 0; i < slots.size(); ++i) {
            int64_t actual = 0;
            check(
                aclrtMemcpy(&actual, sizeof(actual), slots[i], sizeof(actual), ACL_MEMCPY_DEVICE_TO_HOST), "readback"
            );
            if (actual != expected[i]) {
                std::fprintf(stderr, "slot=%zu expected=%ld actual=%ld\n", i, expected[i], actual);
                std::_Exit(1);
            }
        }
    }
};

void sequence(
    KernelExecutionState &state, host::LoadAicpuOp &loader, void *binary, aclrtStream caller, const Buffers &buffers
) {
    auto cpu = state.hidden_stream(KernelStreamKind::Aicpu);
    auto core = state.hidden_stream(KernelStreamKind::Aicore);
    const auto &p = buffers.slots;
    core_launch(binary, caller, {p[0], nullptr, p[1], 3});
    record(state, KernelEventKind::Start, caller);
    wait(state, KernelEventKind::Start, cpu);
    record(state, KernelEventKind::AicoreStart, cpu);
    wait(state, KernelEventKind::AicoreStart, core);
    core_launch(binary, core, {p[1], nullptr, p[3], 11});
    record(state, KernelEventKind::AicoreDone, core);
    CaptureArgs cpu_args{p[1], nullptr, p[2], 7};
    check(loader.LaunchBuiltInOp(cpu, &cpu_args, sizeof(cpu_args), 1, host::KernelNames::RunName), "AICPU launch");
    wait(state, KernelEventKind::AicoreDone, cpu);
    record(state, KernelEventKind::AicpuDone, cpu);
    wait(state, KernelEventKind::AicpuDone, caller);
    record(state, KernelEventKind::SerialTail, caller);
    core_launch(binary, caller, {p[2], p[3], p[4], 0});
}
}  // namespace

extern "C" rtError_t __wrap_rtStreamAddToModel(rtStream_t, rtModel_t) {
    ++forbidden_calls;
    return -70001;
}
extern "C" rtError_t __wrap_rtStreamGetCaptureInfo(rtStream_t, rtStreamCaptureStatus *, rtModel_t *) {
    ++forbidden_calls;
    return -70001;
}
extern "C" aclError __wrap_aclmdlRICaptureGetInfo(aclrtStream, aclmdlRICaptureStatus *, aclmdlRI *) {
    ++forbidden_calls;
    return -70001;
}

int main(int argc, char **argv) {
    require(argc == 5, "usage: driver device dispatcher.so cpu.so core.o");
    const int device = std::atoi(argv[1]);
    auto dispatcher = read_binary(argv[2]);
    auto cpu_binary = read_binary(argv[3]);
    auto core_binary = read_binary(argv[4]);
    require(rtStreamAddToModel(nullptr, nullptr) == -70001, "attachment shim self-test");
    require(rtStreamGetCaptureInfo(nullptr, nullptr, nullptr) == -70001, "RTS query shim self-test");
    require(aclmdlRICaptureGetInfo(nullptr, nullptr, nullptr) == -70001, "ACL query shim self-test");
    require(forbidden_calls == 3, "shim count");
    forbidden_calls = 0;

    check(aclInit(nullptr), "aclInit");
    check(aclrtSetDevice(device), "set device");
    aclrtStream warmup = nullptr, caller = nullptr;
    check(aclrtCreateStream(&warmup), "warmup stream");
    check(aclrtCreateStream(&caller), "caller stream");
    {
        KernelExecutionState state;
        check(state.initialize(device, make_onboard_kernel_context_ops()), "context initialize");
        require(state.hidden_stream(KernelStreamKind::Aicpu) != caller, "private AICPU stream");
        require(state.hidden_stream(KernelStreamKind::Aicore) != caller, "hidden AICore stream");
        require(
            state.hidden_stream(KernelStreamKind::Aicpu) != state.hidden_stream(KernelStreamKind::Aicore),
            "distinct internal streams"
        );
        host::LoadAicpuOp loader;
        check(
            loader.BootstrapDispatcher(
                dispatcher.data(), dispatcher.size(), cpu_binary.data(), cpu_binary.size(),
                state.hidden_stream(KernelStreamKind::Aicpu), device
            ),
            "bootstrap"
        );
        check(loader.Init({}), "loader init");
        check(state.mark_ready_enqueued(), "ready");
        rtDevBinary_t bin{};
        bin.magic = RT_DEV_BINARY_MAGIC_ELF;
        bin.data = core_binary.data();
        bin.length = core_binary.size();
        void *core_handle = nullptr;
        check(rtRegisterAllKernel(&bin, &core_handle), "register AICore");
        {
            Buffers buffers;
            buffers.seed(5);
            sequence(state, loader, core_handle, warmup, buffers);
            check(aclrtSynchronizeStreamWithTimeout(warmup, 10000), "warmup sync");
            buffers.verify(5);

            aclmdlRI graph = nullptr;
            check(aclmdlRICaptureBegin(caller, ACL_MODEL_RI_CAPTURE_MODE_GLOBAL), "capture begin");
            sequence(state, loader, core_handle, caller, buffers);
            check(aclmdlRICaptureEnd(caller, &graph), "capture end");
            require(graph != nullptr, "non-null captured graph");
            for (int64_t i = 0; i < 100; ++i) {
                buffers.seed(i);
                check(aclmdlRIExecuteAsync(graph, caller), "graph replay");
                check(aclrtSynchronizeStreamWithTimeout(caller, 10000), "replay sync");
                buffers.verify(i);
            }
            check(aclmdlRIDestroy(graph), "graph destroy");
        }
        check(state.close(), "context close after graph destroy");
        require(!state.has_live_resources(), "context handles released");
        loader.Finalize();
        check(rtDevBinaryUnRegister(core_handle), "unregister AICore");
    }
    check(aclrtDestroyStream(caller), "destroy caller");
    check(aclrtDestroyStream(warmup), "destroy warmup");
    check(aclrtResetDevice(device), "reset device");
    check(aclFinalize(), "aclFinalize");
    require(forbidden_calls == 0, "launcher used forbidden capture API");
    std::puts("capture_probe PASS replays=100 internal_streams=2 forbidden_calls=0");
    return 0;
}
