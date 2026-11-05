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
#include <cassert>
#include <cstddef>

#include "runtime_c_api.h"

namespace simpler::common::sim_host {

/**
 * Sticky completion state shared by a simulated run's kernel threads and its
 * nonblocking poller.
 *
 * reset() is called before any kernel thread starts. On a successful launch,
 * each released thread calls task_finished() exactly once. If startup is
 * cancelled, the launch owner joins every created thread and calls abandon(rc);
 * cancelled or uncreated threads do not decrement the remaining task count.
 * The final task of a successful launch publishes COMPLETE only
 * after every earlier task's writes and any recorded error are visible to a
 * poller that observes completion. abandon() publishes sticky ERROR when
 * enqueue rollback wins; a late task completion cannot overwrite it.
 */
class SimRunCompletion {
public:
    void reset(size_t submitted_tasks) noexcept {
        assert(submitted_tasks > 0);
        first_error_.store(0, std::memory_order_relaxed);
        remaining_.store(submitted_tasks, std::memory_order_relaxed);
        state_.store(SIMPLER_NATIVE_RUN_POLL_NOT_READY, std::memory_order_release);
    }

    void abandon(int rc = 0) noexcept {
        if (rc != 0) {
            int expected = 0;
            first_error_.compare_exchange_strong(expected, rc, std::memory_order_relaxed);
        }
        state_.store(SIMPLER_NATIVE_RUN_POLL_ERROR, std::memory_order_release);
    }

    void task_finished(int rc = 0) noexcept {
        if (rc != 0) {
            int expected = 0;
            first_error_.compare_exchange_strong(expected, rc, std::memory_order_relaxed);
        }
        const size_t previous = remaining_.fetch_sub(1, std::memory_order_acq_rel);
        assert(previous > 0);
        if (previous == 1) {
            int expected = SIMPLER_NATIVE_RUN_POLL_NOT_READY;
            (void)state_.compare_exchange_strong(
                expected, SIMPLER_NATIVE_RUN_POLL_COMPLETE, std::memory_order_release, std::memory_order_relaxed
            );
        }
    }

    int poll() const noexcept { return state_.load(std::memory_order_acquire); }
    int first_error() const noexcept { return first_error_.load(std::memory_order_acquire); }

private:
    std::atomic<size_t> remaining_{0};
    std::atomic<int> first_error_{0};
    std::atomic<int> state_{SIMPLER_NATIVE_RUN_POLL_ERROR};
};

}  // namespace simpler::common::sim_host
