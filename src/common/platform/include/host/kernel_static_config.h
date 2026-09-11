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

#include <cstring>

#include "common/platform_config.h"
#include "task_interface/call_config.h"
#include "worker/runtime_c_api.h"

// Host-only request snapshot. Topology resolution precedes freeze; neither
// callable registration order nor launch can replace a frozen configuration.
class KernelStaticConfig {
public:
    static int validate(const CallConfig *config) {
        if (config == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
        CallConfig value;
        std::memcpy(&value, config, sizeof(value));
        if (value.aicpu_thread_num < 0 || value.aicpu_thread_num == 1 ||
            value.aicpu_thread_num > PLATFORM_MAX_AICPU_THREADS)
            return PTO_RUNTIME_ERR_INTERNAL;
        if (value.diagnostics_any() || value.capture_clock_anchors != 0) return PTO_RUNTIME_ERR_UNSUPPORTED;
        return 0;
    }

    int initialize(const CallConfig *config, uint64_t generation, bool serial_orch_sched) {
        if (initialized_) return PTO_RUNTIME_ERR_INVALID_STATE;
        const int rc = validate(config);
        if (rc != 0) return rc;
        if (generation == 0) return PTO_RUNTIME_ERR_INTERNAL;
        std::memcpy(&request_, config, sizeof(request_));
        generation_ = generation;
        serial_orch_sched_ = serial_orch_sched;
        initialized_ = true;
        return 0;
    }

    bool initialized() const { return initialized_; }
    bool frozen() const { return frozen_; }
    int freeze() {
        if (!initialized_) return PTO_RUNTIME_ERR_INVALID_STATE;
        frozen_ = true;
        return 0;
    }
    const CallConfig &request() const { return request_; }
    uint64_t generation() const { return generation_; }
    bool serial_orch_sched() const { return serial_orch_sched_; }

private:
    CallConfig request_{};
    uint64_t generation_{0};
    bool serial_orch_sched_{false};
    bool initialized_{false};
    bool frozen_{false};
};
