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
#include <type_traits>

#include "common/launch_entry_args.h"

namespace simpler::tmr {

// A capture owns one fixed execution image. Its model executions must enter the
// same caller stream: the whole-operator join orders the next admission behind
// every AICPU and AICore return of its predecessor. Host submission mutexes do
// not establish that device order.
enum class CaptureDecision : uint32_t { Idle = 0, Admitted = 1, Rejected = 2 };

struct alignas(64) KernelCaptureControl {
    uint64_t context_generation{0};
    uint64_t capture_generation{0};
    uint64_t runtime_address{0};
    uint64_t binding_epoch{0};
    uint64_t callable_generation{0};
    int32_t callable_id{-1};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    uint32_t launched_threads{0};
    uint32_t worker_count{0};
    uint32_t reserved{0};

    // Only the admission task writes a new decision/epoch. A failure arriving
    // from an execution thread latches first_error without changing that
    // decision: late participants of the admitted group must still enter its
    // ordinary handshake and retirement paths.
    uint64_t replay_epoch{0};
    uint64_t completed_epoch{0};
    int32_t first_error{0};
    uint32_t decision{static_cast<uint32_t>(CaptureDecision::Idle)};
    uint32_t finished_threads{0};
    uint32_t arrived_threads{0};
    uint32_t reserved_tail[8]{};
};

// The first object in a KernelCaptureInvocation payload, after the shared
// launch header and before its immutable invocation. It is checked against
// independent registration before either processor accesses the execution image.
struct KernelCaptureReference {
    uint64_t control_address{0};
    uint64_t capture_generation{0};
    uint64_t context_generation{0};
};

inline constexpr size_t KERNEL_CAPTURE_INVOCATION_OFFSET =
    LAUNCH_ENVELOPE_HEADER_BYTES + sizeof(KernelCaptureReference);

enum class CaptureRegistrationAction : uint32_t { Register = 0, Revoke = 1, RevokeCallable = 2 };

struct KernelCaptureRegistrationArgs {
    KernelCaptureReference reference{};
    uint64_t runtime_address{0};
    uint64_t binding_epoch{0};
    uint64_t callable_generation{0};
    uint64_t run_result_address{0};
    uint32_t packet_bytes{0};
    uint32_t tensor_count{0};
    uint32_t scalar_count{0};
    uint32_t launched_threads{0};
    uint32_t worker_count{0};
    int32_t callable_id{-1};
    uint32_t action{static_cast<uint32_t>(CaptureRegistrationAction::Register)};
    uint32_t reserved{0};
};

static_assert(std::is_standard_layout_v<KernelCaptureControl> && std::is_trivially_copyable_v<KernelCaptureControl>);
static_assert(sizeof(KernelCaptureControl) == 128 && alignof(KernelCaptureControl) == 64);
static_assert(offsetof(KernelCaptureControl, replay_epoch) == 64);
static_assert(sizeof(KernelCaptureReference) == 24);
static_assert(
    std::is_standard_layout_v<KernelCaptureReference> && std::is_trivially_copyable_v<KernelCaptureReference>
);
static_assert(
    std::is_standard_layout_v<KernelCaptureRegistrationArgs> &&
    std::is_trivially_copyable_v<KernelCaptureRegistrationArgs>
);

}  // namespace simpler::tmr
