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

#include <cstdint>
#include <type_traits>

#include "common/device_run_result.h"

enum class NativeExecutionDecision : uint32_t { Execute = 0, Skip = 1 };
enum class NativeExecutionFailure : uint32_t { None = 0, Error = 1, Unknown = 2 };

// Context-owned storage lies outside every restored image. A missing terminal
// is distinguishable from a genuine signed -1 error, including across banks.
struct alignas(128) NativeExecutionFirstError {
    int32_t code{0};
    NativeExecutionFailure kind{NativeExecutionFailure::None};
    DeviceRunCodeSource source{DeviceRunCodeSource::ThreadRc};
};

// Never part of a restored Runtime/image. The preparation command owns writes
// before the fork; callbacks only contribute status and retire one count each.
struct alignas(64) NativeExecutionControl {
    uint64_t sequence{0};
    uint64_t native_epoch{0};
    uint64_t result_address{0};
    uint64_t first_error_address{0};
    uint32_t expected_callbacks{0};
    uint32_t remaining_callbacks{0};
    NativeExecutionDecision decision{NativeExecutionDecision::Execute};
    int32_t first_error{0};
    int32_t entry_error{0};
    NativeExecutionFailure failure_kind{NativeExecutionFailure::None};
    DeviceRunCodeSource failure_source{DeviceRunCodeSource::ThreadRc};
    uint32_t padding{0};
};

struct NativeExecutionPrepareArgs {
    uint64_t control;
    uint64_t runtime_address;
    uint64_t result_address;
    uint64_t native_epoch;
    uint64_t first_error_address;
    uint32_t launched_callbacks;
    uint32_t worker_count;
};

inline void native_execution_latch(int32_t &destination, int32_t status) {
    if (status == 0) return;
    int32_t clean = 0;
    (void)__atomic_compare_exchange_n(&destination, &clean, status, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);
}

inline void native_execution_latch_first(
    NativeExecutionControl &control, int32_t status, NativeExecutionFailure kind = NativeExecutionFailure::Error,
    DeviceRunCodeSource source = DeviceRunCodeSource::ThreadRc
) {
    if (status == 0) return;
    int32_t clean = 0;
    if (__atomic_compare_exchange_n(&control.first_error, &clean, status, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        control.failure_kind = kind;
        control.failure_source = source;
    }
    if (control.first_error_address != 0) {
        auto &first = *reinterpret_cast<NativeExecutionFirstError *>(control.first_error_address);
        clean = 0;
        if (__atomic_compare_exchange_n(&first.code, &clean, status, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
            first.kind = kind;
            first.source = source;
        }
    }
    // Only preparation or the final callback writes these associated fields.
    // The next preparation reads them after the whole-operator stream join.
}

// Called only after the prior whole-operator join. Inspect the previous result
// before changing its identity or clearing a publication which could be evidence
// of an incomplete run. Cache maintenance is supplied by the platform entry.
inline int native_execution_prepare(NativeExecutionControl &control, const NativeExecutionPrepareArgs &args) {
    if (args.launched_callbacks == 0 || args.result_address == 0 || args.native_epoch == 0) return -1;
    if (control.remaining_callbacks != 0) {
        native_execution_latch_first(control, -1, NativeExecutionFailure::Unknown);
        return -1;
    }
    if (control.sequence != 0 && control.result_address == 0) return -1;
    if (control.sequence != 0) {
        const auto *previous = reinterpret_cast<const DeviceRunResultRegion *>(control.result_address);
        const auto terminal = device_run_result_terminal(*previous, control.native_epoch);
        if (terminal.state != DeviceRunTerminalState::Succeeded)
            native_execution_latch_first(
                control, terminal.state == DeviceRunTerminalState::Failed ? terminal.code : -1,
                terminal.state == DeviceRunTerminalState::Failed ? NativeExecutionFailure::Error :
                                                                   NativeExecutionFailure::Unknown,
                static_cast<DeviceRunCodeSource>(previous->code_source)
            );
    }
    control.first_error_address = args.first_error_address;
    if (args.first_error_address != 0) {
        const auto &first = *reinterpret_cast<const NativeExecutionFirstError *>(args.first_error_address);
        const int32_t status = __atomic_load_n(&first.code, __ATOMIC_ACQUIRE);
        if (status != 0 && control.first_error == 0) {
            control.first_error = status;
            control.failure_kind =
                first.kind == NativeExecutionFailure::None ? NativeExecutionFailure::Error : first.kind;
            control.failure_source = first.source;
        }
    }
    control.decision = control.first_error == 0 ? NativeExecutionDecision::Execute : NativeExecutionDecision::Skip;
    control.native_epoch = args.native_epoch;
    control.result_address = args.result_address;
    control.expected_callbacks = args.launched_callbacks;
    control.entry_error = 0;
    control.remaining_callbacks = args.launched_callbacks;
    if (control.sequence != UINT64_MAX) ++control.sequence;
    auto &result = *reinterpret_cast<DeviceRunResultRegion *>(args.result_address);
    __atomic_store_n(&result.published, uint64_t{0}, __ATOMIC_RELEASE);
    return 0;
}

// Every RTS callback contributes after its own work and destructors have ended,
// including filtered callbacks and controllable entry failures. This counter is
// semantic aggregation, not evidence that the RTS tasks have physically exited.
template <typename RefreshResult, typename PublishResult>
inline int32_t native_execution_finish(
    NativeExecutionControl &control, int32_t status, RefreshResult refresh_result, PublishResult publish_result
) {
    native_execution_latch(control.entry_error, status);
    uint32_t remaining = __atomic_load_n(&control.remaining_callbacks, __ATOMIC_ACQUIRE);
    do {
        if (remaining == 0) {
            native_execution_latch_first(control, -1);
            return __atomic_load_n(&control.first_error, __ATOMIC_ACQUIRE);
        }
    } while (!__atomic_compare_exchange_n(
        &control.remaining_callbacks, &remaining, remaining - 1, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
    ));
    if (remaining != 1) return 0;
    refresh_result();
    auto &result = *reinterpret_cast<DeviceRunResultRegion *>(control.result_address);
    const auto terminal = device_run_result_terminal(result, control.native_epoch);
    const int32_t entry_error = __atomic_load_n(&control.entry_error, __ATOMIC_ACQUIRE);
    const bool missing_terminal = terminal.state == DeviceRunTerminalState::Undecided && entry_error == 0 &&
                                  __atomic_load_n(&control.first_error, __ATOMIC_ACQUIRE) == 0;
    int32_t selected = terminal.state == DeviceRunTerminalState::Failed ? terminal.code : entry_error;
    if (selected == 0 && terminal.state != DeviceRunTerminalState::Succeeded) selected = -1;
    native_execution_latch_first(
        control, selected, missing_terminal ? NativeExecutionFailure::Unknown : NativeExecutionFailure::Error,
        terminal.state == DeviceRunTerminalState::Failed ? static_cast<DeviceRunCodeSource>(result.code_source) :
                                                           DeviceRunCodeSource::ThreadRc
    );
    selected = __atomic_load_n(&control.first_error, __ATOMIC_ACQUIRE);
    // Keep the runtime's payload/source when it already published the selected
    // error. An entry error must replace a successful or absent runtime result.
    if (control.failure_kind == NativeExecutionFailure::Unknown) {
        // A successor's refusal does not upgrade the original missing evidence
        // into a known error. Clear any runtime-produced Skip result as well.
        __atomic_store_n(&result.published, uint64_t{0}, __ATOMIC_RELEASE);
    } else if (selected != 0 && (terminal.state != DeviceRunTerminalState::Failed || terminal.code != selected ||
                                 result.code_source != static_cast<uint32_t>(control.failure_source))) {
        publish_result(selected);
    }
    return selected;
}

static_assert(sizeof(NativeExecutionFirstError) == 128);
static_assert(sizeof(NativeExecutionControl) == 64);
static_assert(sizeof(NativeExecutionPrepareArgs) == 48);
static_assert(
    std::is_trivially_copyable_v<NativeExecutionControl> && std::is_standard_layout_v<NativeExecutionControl>
);
