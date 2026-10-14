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

namespace simpler::hbg {

struct AsyncPollPhaseSummary {
    uint64_t start_time{0};
    uint64_t end_time{0};
    uint32_t resolved{0};
};

// Accumulates only the CPU time spent polling, excluding the gaps between
// scheduler-loop iterations. The caller chooses the flush point, so a run of
// empty polls and its terminating resolved/error poll becomes one compact bar.
class AsyncPollPhaseAccumulator {
public:
    bool active() const { return active_; }

    void begin() { active_ = true; }

    bool add_poll(uint64_t start_time, uint64_t end_time, uint32_t resolved, bool failed = false) {
        if (!active_) begin();
        accumulated_cycles_ += end_time - start_time;
        resolved_ += resolved;
        return resolved > 0 || failed;
    }

    AsyncPollPhaseSummary flush(uint64_t end_time) {
        AsyncPollPhaseSummary summary;
        summary.start_time = end_time >= accumulated_cycles_ ? end_time - accumulated_cycles_ : 0;
        summary.end_time = end_time;
        summary.resolved = resolved_;
        active_ = false;
        accumulated_cycles_ = 0;
        resolved_ = 0;
        return summary;
    }

private:
    bool active_{false};
    uint64_t accumulated_cycles_{0};
    uint32_t resolved_{0};
};

}  // namespace simpler::hbg
