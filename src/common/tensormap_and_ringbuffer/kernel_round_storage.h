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

#include "aicpu/cache_maintenance.h"
#include "aicpu/platform_regs.h"
#include "tmr_kernel_control.h"

namespace simpler::tmr {

struct KernelHandshakeView {
    TmrLaunchControl *control{nullptr};
    TmrCoreReport *reports{nullptr};
    int32_t worker_count{0};
};

// Borrowed round storage. Register handshake and retirement belong to SchedulerContext.
class KernelRoundStorage {
public:
    bool attach(KernelHandshakeView view) noexcept {
        if (view.control == nullptr || view.reports == nullptr || view.worker_count <= 0 ||
            view.worker_count > PLATFORM_MAX_CORES ||
            reinterpret_cast<uintptr_t>(view.control) % alignof(TmrLaunchControl) != 0 ||
            reinterpret_cast<uintptr_t>(view.reports) % alignof(TmrCoreReport) != 0)
            return false;
        const uintptr_t control = reinterpret_cast<uintptr_t>(view.control);
        const uintptr_t reports = reinterpret_cast<uintptr_t>(view.reports);
        const size_t reports_bytes = static_cast<size_t>(view.worker_count) * sizeof(TmrCoreReport);
        if (sizeof(TmrLaunchControl) > UINTPTR_MAX - control || reports_bytes > UINTPTR_MAX - reports ||
            !(control + sizeof(TmrLaunchControl) <= reports || reports + reports_bytes <= control))
            return false;
        cache_invalidate_range(view.control, sizeof(TmrLaunchControl));
        const uint64_t completed_epoch = view.control->round_epoch;
        if (completed_epoch == UINT64_MAX) return false;
        cache_invalidate_range(view.reports, reports_bytes);
        view_ = view;
        expected_epoch_ = completed_epoch + 1;
        return true;
    }

    TmrCoreReport *reports() const noexcept { return view_.reports; }
    uint64_t expected_epoch() const noexcept { return expected_epoch_; }

    void publish_status(int32_t runtime, int32_t cleanup) noexcept {
        auto &control = *view_.control;
        control.runtime_status = runtime;
        control.cleanup_status = cleanup;
        if (runtime == 0 && cleanup == 0) control.round_epoch = expected_epoch_;
        __atomic_store_n(&control.completion, uint32_t{1}, __ATOMIC_RELEASE);
        cache_flush_range(&control, sizeof(control));
    }

private:
    KernelHandshakeView view_{};
    uint64_t expected_epoch_{0};
};

}  // namespace simpler::tmr
