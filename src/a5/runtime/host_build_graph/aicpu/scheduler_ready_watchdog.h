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

#include <atomic>
#include <cstdint>

#include "scheduler/scheduler_watchdog.h"

// The init owner seeds this clock before releasing run participants. Only the
// supervisor publishes actual bootstrap/resolved-task progress during execution.
class AicpuReadyWatchdog {
public:
    void reset(uint64_t now = 0) { last_progress_.store(now, std::memory_order_release); }

    void record_progress(uint64_t now) { last_progress_.store(now, std::memory_order_release); }

    uint64_t last_progress() const { return last_progress_.load(std::memory_order_acquire); }

    template <typename Ready, typename Clock, typename Relax>
    bool wait_until_ready(Ready ready, Clock clock, Relax relax, uint64_t timeout_cycles) const {
        while (!ready()) {
            const uint64_t observed_progress = last_progress();
            const uint64_t now = clock();
            // A shared publication can be newer than this clock sample.
            // Forward elapsed intervals occupy the lower half of the counter range.
            const uint64_t elapsed = now - observed_progress;
            if (elapsed < (uint64_t{1} << 63) && scheduler_watchdog_expired(observed_progress, now, timeout_cycles)) {
                if (ready()) return true;
                // A new publication requires a fresh clock sample: its timestamp
                // may be later than `now`. Polling alone never extends the budget.
                if (last_progress() == observed_progress) return false;
            }
            relax();
        }
        return true;
    }

private:
    std::atomic<uint64_t> last_progress_{0};
};
