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

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

#include "../protocol.h"
#include "platform/onboard/host/tmr_kernel_invocation.h"

namespace {
void check(int code, const char *operation) {
    if (code != 0) throw std::runtime_error(std::string(operation) + ": " + std::to_string(code));
}
std::vector<uint8_t> read_binary(const char *path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error(std::string("cannot open ") + path);
    return {std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
}
}  // namespace

// Standalone process: owns its ACL setup and result buffer. No TMR execution
// resources or production registry are created by this transport-only probe.
int main(int argc, char **argv) {
    if (argc != 4) {
        std::fprintf(stderr, "usage: snapshot_probe DEVICE DISPATCHER_SO PROBE_SO\n");
        return 2;
    }
    try {
        const int device = std::stoi(argv[1]);
        const auto dispatcher = read_binary(argv[2]);
        const auto binary = read_binary(argv[3]);
        check(aclInit(nullptr), "aclInit");
        check(aclrtSetDevice(device), "aclrtSetDevice");
        aclrtStream stream = nullptr;
        check(aclrtCreateStream(&stream), "create stream");
        aclrtStream release_stream = nullptr;
        check(aclrtCreateStream(&release_stream), "create release stream");
        constexpr uint32_t count = 24;
        void *results = nullptr;
        check(aclrtMalloc(&results, count * sizeof(SnapshotProbeResult), ACL_MEM_MALLOC_HUGE_FIRST), "alloc results");
        check(
            aclrtMemset(results, count * sizeof(SnapshotProbeResult), 0xff, count * sizeof(SnapshotProbeResult)),
            "initialize result sentinels"
        );
        void *gate = nullptr;
        check(aclrtMalloc(&gate, sizeof(uint32_t), ACL_MEM_MALLOC_HUGE_FIRST), "alloc test gate");
        check(aclrtMemset(gate, sizeof(uint32_t), 0, sizeof(uint32_t)), "close test gate");
        {
            host::LoadAicpuOp loader;
            check(
                loader.BootstrapDispatcher(
                    dispatcher.data(), dispatcher.size(), binary.data(), binary.size(), stream, device
                ),
                "bootstrap"
            );
            check(loader.Init({simpler::tmr::TmrKernelInvocationName}), "register entries");
            SnapshotProbeInit init{reinterpret_cast<uint64_t>(results), 13, reinterpret_cast<uint64_t>(gate), count};
            check(loader.LaunchBuiltInOp(stream, &init, sizeof(init), 1, "simpler_aicpu_init"), "init fixture");
            check(aclrtSynchronizeStream(stream), "sync prepare");

            std::vector<SnapshotProbeResult> expected;
            simpler::tmr::TmrEncodingCache caches[3];
            const simpler::tmr::TmrExecutionBindingView binding{init.results_addr, init.context_generation};
            for (uint32_t i = 0; i < count; ++i) {
                const auto callable = snapshot_probe_callable(static_cast<int32_t>(i % 3));
                ChipStorageTaskArgs args{};
                uint64_t sum = 0;
                for (int t = 0; t < callable.tensor_count; ++t) {
                    const uint32_t shape[] = {1};
                    const uint64_t addr = init.results_addr + (i % count) * sizeof(SnapshotProbeResult);
                    args.add_tensor(make_tensor_external(
                        reinterpret_cast<void *>(addr), shape, 1, DataType::FLOAT32, AddressSpace::DEVICE
                    ));
                    sum += addr + 1;
                }
                if (callable.scalar_count != 0) {
                    args.add_scalar(i * 17 + 5);
                    sum += i * 17 + 5;
                }
                simpler::tmr::TmrEncodingCandidate candidate;
                auto status = simpler::tmr::encode_tmr_invocation(args, callable, binding, caches[i % 3], &candidate);
                if (status != simpler::kernel::InvocationStatus::Ok) throw std::runtime_error("encode failed");
                check(
                    simpler::tmr::enqueue_tmr_invocation_aicpu(loader, stream, 1, candidate, callable, binding),
                    "enqueue snapshot"
                );
                caches[i % 3].commit(std::move(candidate));
                const auto packet = candidate.packet();
                std::memset(const_cast<uint8_t *>(packet.data), 0xa5, packet.size);
                args = {};
                expected.push_back({sum, callable.callable_id, 0});
            }
            check(aclrtMemsetAsync(gate, sizeof(uint32_t), 1, sizeof(uint32_t), release_stream), "release test gate");
            check(aclrtSynchronizeStream(stream), "sync invocations");
            check(aclrtSynchronizeStream(release_stream), "sync release");
            std::vector<SnapshotProbeResult> actual(count);
            check(
                aclrtMemcpy(
                    actual.data(), actual.size() * sizeof(actual[0]), results, actual.size() * sizeof(actual[0]),
                    ACL_MEMCPY_DEVICE_TO_HOST
                ),
                "read results"
            );
            for (uint32_t i = 0; i < count; ++i) {
                if (actual[i].sum != expected[i].sum || actual[i].callable_id != expected[i].callable_id ||
                    actual[i].status != 0)
                    throw std::runtime_error("snapshot mismatch at " + std::to_string(i));
            }
            std::printf(
                "PASS: %u gated asynchronous snapshots, minimum/maximum packets, host overwrite+release\n", count
            );
        }
        check(aclrtFree(results), "free results");
        check(aclrtFree(gate), "free test gate");
        check(aclrtDestroyStream(release_stream), "destroy release stream");
        check(aclrtDestroyStream(stream), "destroy stream");
        check(aclrtResetDevice(device), "reset device");
        check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what());
        return 1;
    }
}
