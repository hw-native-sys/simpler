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

#include <condition_variable>
#include <mutex>
#include "aicpu/platform_aicpu_affinity.h"

// Each simulated run owns its rendezvous. No counter is shared across runners.
class SimNativeEntryGate {
public:
    explicit SimNativeEntryGate(int32_t callbacks) :
        callbacks_(callbacks) {}
    AicpuEntryGateResult enter(int32_t logical, AicpuEntryGateInitializer initialize) {
        std::unique_lock<std::mutex> lock(mutex_);
        if (callbacks_ <= 0 || callbacks_ > MAX_GATE_THREADS || arrived_ >= callbacks_) return {};
        const int32_t index = arrived_++;
        if (index == 0) logical_ = logical;
        if (logical <= 0 || logical > callbacks_ || logical != logical_ || initialize.initialize == nullptr)
            status_ = -1;
        if (arrived_ == callbacks_) {
            if (initialize.initialize != nullptr) {
                const int32_t rc = initialize.initialize(initialize.context);
                if (rc != 0) status_ = rc;
            }
            ready_ = true;
            ready_cv_.notify_all();
        } else {
            ready_cv_.wait(lock, [this] {
                return ready_;
            });
        }
        return {index, status_, status_ == 0 && index < logical_};
    }

private:
    std::mutex mutex_;
    std::condition_variable ready_cv_;
    const int32_t callbacks_;
    int32_t arrived_{0};
    int32_t logical_{0};
    int32_t status_{0};
    bool ready_{false};
};
