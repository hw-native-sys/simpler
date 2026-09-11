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

#include <cstring>
#include <ctime>
#include "../protocol.h"
#include "tensormap_and_ringbuffer/kernel_invocation_args.h"

namespace {
SnapshotProbeInit fixture{};
uint32_t next_result = 0;
}  // namespace

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *args) {
    if (args == nullptr) return 1;
    std::memcpy(&fixture, args, sizeof(fixture));
    next_result = 0;
    return 0;
}
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *) { return 1; }

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_kernel_exec(void *args) {
    using namespace simpler::tmr;
    if (args == nullptr || fixture.results_addr == 0 || next_result >= fixture.capacity) return 1;
    if (fixture.gate_addr == 0) return 1;
    timespec started{};
    if (clock_gettime(CLOCK_MONOTONIC, &started) != 0) return 1;
    // The host releases this test-only gate after destroying every packet.
    // No invocation bytes are read while the gate is closed.
    while (*reinterpret_cast<volatile uint32_t *>(fixture.gate_addr) == 0) {
        timespec now{};
        if (clock_gettime(CLOCK_MONOTONIC, &now) != 0 || now.tv_sec - started.tv_sec > 30) return 1;
    }
    SimplerKernelInvocationHeader header{};
    std::memcpy(&header, args, sizeof(header));
    const auto callable = snapshot_probe_callable(header.callable_id);
    size_t bytes = 0;
    auto status = tmr_invocation_size(callable, &bytes);
    TmrInvocationView view;
    const TmrExecutionBindingView binding{fixture.results_addr, fixture.context_generation};
    // The formal host adapter submits exactly the independently derived size.
    // This CPU entry has no API for observing CANN's actual readable byte count.
    if (status == InvocationStatus::Ok)
        status = decode_tmr_invocation({static_cast<const uint8_t *>(args), bytes}, callable, binding, &view);
    uint64_t sum = 0;
    if (status == InvocationStatus::Ok) {
        EntryArgsStorage storage{};
        status = materialize_tmr_entry_args(view, &storage);
        for (int i = 0; i < storage.tensor_count(); ++i)
            sum += storage.tensor(i).buffer.addr + storage.tensor(i).numel();
        for (int i = 0; i < storage.scalar_count(); ++i)
            sum += storage.scalar(i);
    }
    auto *results = reinterpret_cast<volatile SnapshotProbeResult *>(fixture.results_addr);
    const auto index = next_result++;
    results[index].sum = sum;
    results[index].callable_id = header.callable_id;
    results[index].status = static_cast<int32_t>(status);
    return status == InvocationStatus::Ok ? 0 : 1;
}
