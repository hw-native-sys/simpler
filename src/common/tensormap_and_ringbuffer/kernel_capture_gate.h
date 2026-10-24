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

#include <limits>

#include "task_interface/kernel_capture.h"

namespace simpler::tmr {

inline void capture_latch_error(KernelCaptureControl &control, int32_t status) noexcept {
    if (status == 0) return;
    int32_t expected = 0;
    (void)__atomic_compare_exchange_n(
        &control.first_error, &expected, status, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
    );
}

// Called once by the admission task, before its event releases either processor.
// Validation is supplied by the registered packet consumer; a rejected request
// changes only this capture's control record, never Runtime or arena state.
inline bool capture_admit(KernelCaptureControl &control, bool valid, int32_t refusal) noexcept {
    if (control.launched_threads == 0 || control.capture_generation == 0 || control.context_generation == 0 ||
        control.runtime_address == 0 || control.binding_epoch == 0)
        return false;
    const uint64_t previous = __atomic_load_n(&control.replay_epoch, __ATOMIC_ACQUIRE);
    const uint64_t complete = __atomic_load_n(&control.completed_epoch, __ATOMIC_ACQUIRE);
    if (previous == std::numeric_limits<uint64_t>::max() || complete != previous) return false;

    __atomic_store_n(&control.finished_threads, 0U, __ATOMIC_RELAXED);
    __atomic_store_n(&control.arrived_threads, 0U, __ATOMIC_RELAXED);
    if (!valid) capture_latch_error(control, refusal);
    const bool accepted = valid && __atomic_load_n(&control.first_error, __ATOMIC_ACQUIRE) == 0;
    __atomic_store_n(&control.replay_epoch, previous + 1, __ATOMIC_RELAXED);
    __atomic_store_n(
        &control.decision, static_cast<uint32_t>(accepted ? CaptureDecision::Admitted : CaptureDecision::Rejected),
        __ATOMIC_RELEASE
    );
    return true;
}

// Every native AICPU participant, including an affinity-filtered thread and a
// rejected request, reports its own return. This record covers CPU arrivals;
// only the native whole-operator join proves the AICore kernels returned.
template <typename Finalize>
inline bool capture_finish(KernelCaptureControl &control, uint64_t epoch, int32_t status, Finalize finalize) noexcept {
    if (epoch == 0 || epoch != __atomic_load_n(&control.replay_epoch, __ATOMIC_ACQUIRE)) return false;
    uint32_t arrived = __atomic_load_n(&control.arrived_threads, __ATOMIC_ACQUIRE);
    do {
        if (arrived >= control.launched_threads) return false;
    } while (!__atomic_compare_exchange_n(
        &control.arrived_threads, &arrived, arrived + 1, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE
    ));
    capture_latch_error(control, status);
    const uint32_t finished = __atomic_fetch_add(&control.finished_threads, 1U, __ATOMIC_ACQ_REL);
    if (finished + 1 == control.launched_threads) {
        finalize(control, epoch);
        __atomic_store_n(&control.completed_epoch, epoch, __ATOMIC_RELEASE);
    }
    return true;
}

inline bool capture_finish(KernelCaptureControl &control, uint64_t epoch, int32_t status) noexcept {
    return capture_finish(control, epoch, status, [](KernelCaptureControl &, uint64_t) {});
}

}  // namespace simpler::tmr
