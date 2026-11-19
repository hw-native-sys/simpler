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
#include <cstddef>
#include <mutex>

// The run owns this gate until all core threads have joined. Only a complete
// cohort may enter the executors; partial creation is cancelled before join.
class SimThreadStartGate {
public:
    // Called by the launch owner with no live users from an earlier attempt.
    void reset() {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = State::Waiting;
        ready_ = 0;
        error_ = 0;
    }
    bool wait() {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] {
            return state_ != State::Waiting;
        });
        return state_ == State::Released;
    }
    static bool report_ready_and_wait(void *context, int error) {
        return static_cast<SimThreadStartGate *>(context)->arrive_and_wait(error);
    }
    bool arrive_and_wait(int error) {
        std::unique_lock<std::mutex> lock(mutex_);
        ++ready_;
        if (error_ == 0) error_ = error;
        changed_.notify_all();
        changed_.wait(lock, [&] {
            return state_ != State::Waiting;
        });
        return state_ == State::Released;
    }
    int wait_for_arrivals_or_error(std::size_t expected) {
        std::unique_lock<std::mutex> lock(mutex_);
        changed_.wait(lock, [&] {
            return ready_ == expected || error_ != 0;
        });
        return error_;
    }
    void release() { publish(State::Released); }
    void cancel() { publish(State::Cancelled); }

private:
    enum class State { Waiting, Released, Cancelled };
    void publish(State state) {
        std::lock_guard<std::mutex> lock(mutex_);
        state_ = state;
        changed_.notify_all();
    }
    std::mutex mutex_;
    std::condition_variable changed_;
    State state_{State::Waiting};
    std::size_t ready_{0};
    int error_{0};
};
