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

#include <cstddef>
#include <cstdint>

#include "worker/runtime_c_api.h"

namespace simpler::kernel_launch {

struct KernelInvocationPlaceholder {
    uint32_t address_offset;
    uint32_t data_offset;
};

// Runtime wire validation belongs to the owner. Packet and placeholder storage
// remain exclusively leased until finish; task-owned copies outlive submission.
struct KernelInvocationBinding {
    const uint8_t *packet{nullptr};
    size_t packet_bytes{0};
    const KernelInvocationPlaceholder *placeholders{nullptr};
    size_t placeholder_count{0};
};

struct KernelLaunchOps {
    void *context{nullptr};
    int (*wait_event)(void *, void *stream, void *event) noexcept {nullptr};
    int (*memset_handshake)(void *, void *stream) noexcept {nullptr};
    int (*record_event)(void *, void *event, void *stream) noexcept {nullptr};
    int (*launch_aicpu)(void *, void *stream) noexcept {nullptr};
    int (*launch_aicore)(void *, void *stream) noexcept {nullptr};
    int (*cancel_waiting_aicore)(void *, void *stream) noexcept {nullptr};
    bool valid() const {
        return wait_event && memset_handshake && record_event && launch_aicpu && launch_aicore && cancel_waiting_aicore;
    }
};

struct KernelLaunchHandles {
    void *caller{nullptr};
    void *aicpu{nullptr};
    void *aicore{nullptr};
    void *prepare_tail{nullptr};
    void *start{nullptr};
    void *aicore_done{nullptr};
    void *aicpu_done{nullptr};
    void *serial_tail{nullptr};
    bool consume_prepare_tail{false};

    bool valid() const {
        if (!caller || !aicpu || !aicore || caller == aicpu || caller == aicore || aicpu == aicore) return false;
        const void *events[] = {prepare_tail, start, aicore_done, aicpu_done, serial_tail};
        for (size_t i = 0; i < 5; ++i) {
            if (!events[i]) return false;
            for (size_t j = 0; j < i; ++j)
                if (events[i] == events[j]) return false;
        }
        return true;
    }
};

enum class KernelLaunchStep : uint8_t {
    Validate,
    QueryTail,
    PrepareWait,
    Clear,
    Start,
    AicoreWait,
    AicoreLaunch,
    AicoreDone,
    AicpuWait,
    AicpuLaunch,
    AicpuDone,
    JoinAicpu,
    JoinAicore,
    SerialTail
};

struct KernelLaunchResult {
    int status{0};
    int cleanup_status{0};
    KernelLaunchStep failed_step{KernelLaunchStep::Validate};
    bool enqueue_started{false};
    bool tail_recorded{false};
};

struct KernelLaunchAdmission {
    KernelLaunchHandles handles;
    uintptr_t previous_caller_identity{0};
};

// acquire is read-only apart from acquiring an exclusive owner submission lease:
// it checks context phase, device/registration affinity, frozen capacity, packet
// ABI and runtime bindings, and returns handles plus prior submission state.
// Failure retains no lease. Success is paired with exactly one finish call.
// finish commits caller/tail/prepare state on success, poisons on enqueue failure,
// preserves clean rejections, and releases the lease. Neither callback enqueues,
// allocates, synchronizes, or queries capture. The owner also serializes close.
struct KernelLaunchGateOps {
    void *context{nullptr};
    int (*acquire)(void *, const KernelInvocationBinding &, void *caller, KernelLaunchAdmission *) noexcept {nullptr};
    void (*finish)(void *, const KernelLaunchResult &) noexcept {nullptr};
    int (*query_tail)(void *, void *event, bool *complete) noexcept {nullptr};
    bool valid() const { return acquire && finish && query_tail; }
};

KernelLaunchResult launch_bound_kernel(
    const KernelInvocationBinding &binding, void *caller_stream, const KernelLaunchGateOps &gate,
    const KernelLaunchOps &ops
);

}  // namespace simpler::kernel_launch
