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
// supervisor and dump services publish new observations during execution.
class AicpuReadyWatchdog {
public:
    void reset(uint64_t now = 0) { last_progress_.store(now, std::memory_order_release); }

    void record_progress(uint64_t now) { last_progress_.store(now, std::memory_order_release); }

    bool publish_sample(uint64_t previous, uint64_t now) {
        return last_progress_.compare_exchange_strong(previous, now, std::memory_order_acq_rel);
    }

    uint64_t last_progress() const { return last_progress_.load(std::memory_order_acquire); }

    template <typename Ready, typename Clock, typename Relax>
    bool wait_until_ready(Ready ready, Clock clock, Relax relax, uint64_t timeout_cycles) const {
        while (!ready()) {
            const uint64_t observed_progress = last_progress();
            const uint64_t now = clock();
            if (scheduler_watchdog_expired(observed_progress, now, timeout_cycles)) {
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

// A sink can stay pending while its entire ancestor chain advances. Take a
// baseline after half a quiet interval; compare only after the other half.
// Publish each accepted sample to the peers waiting for shutdown, so they
// cannot time out a working supervisor at the same instant it probes again.
class AicpuSinkWatchdog {
public:
    template <typename Sample>
    bool stalled(AicpuReadyWatchdog &ready, uint64_t now, uint64_t timeout, Sample sample) {
        const uint64_t progress = ready.last_progress();
        if (progress != last_sample_) have_sample_ = false;
        const uint64_t interval = timeout / 2 + ((!have_sample_ || timeout == 1) ? timeout % 2 : 0);
        if (!scheduler_watchdog_expired(progress, now, interval)) return false;
        const uint64_t current = sample();
        if (ready.last_progress() != progress) return false;  // A dump copy completed during the scan.
        if (have_sample_ && current == fingerprint_) return true;
        if (!ready.publish_sample(progress, now)) return false;
        fingerprint_ = current;
        have_sample_ = true;
        last_sample_ = now;
        return false;
    }

private:
    bool have_sample_{};
    uint64_t last_sample_{};
    uint64_t fingerprint_{};
};
