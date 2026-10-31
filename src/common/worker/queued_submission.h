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
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <unistd.h>

// The action and Run contain native values only. Python owns storage lifetimes
// separately, until a completion verdict permits retirement.
template <class Run>
class QueuedSubmission : public std::enable_shared_from_this<QueuedSubmission<Run>> {
public:
    using Deadline = typename Run::Deadline;

    explicit QueuedSubmission(std::function<Run()> action) :
        action_(std::move(action)) {}

    void submit() {
        validate_process();
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (phase_ != Phase::Pending) throw std::logic_error("submission token is already consumed or cancelled");
            phase_ = Phase::Submitting;
        }
        std::optional<Run> result;
        std::exception_ptr error;
        try {
            result.emplace(action_());
        } catch (...) {
            error = std::current_exception();
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            run_ = std::move(result);
            error_ = error;
            action_ = {};
            phase_ = Phase::Submitted;
        }
        changed_.notify_all();
        if (error) std::rethrow_exception(error);
    }

    bool cancel(std::exception_ptr reason) {
        validate_process();
        if (!reason) throw std::invalid_argument("cancellation requires a reason");
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (phase_ != Phase::Pending) return false;
            error_ = reason;
            action_ = {};
            phase_ = Phase::Submitted;
        }
        changed_.notify_all();
        return true;
    }

    bool done() {
        validate_process();
        std::optional<Run> run;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (phase_ == Phase::Retired || error_) return true;
            if (phase_ != Phase::Submitted) return false;
            run = run_;
        }
        return run->done();
    }

    bool wait_until(Deadline deadline) {
        validate_process();
        std::optional<Run> run;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            auto ready = [&] {
                return phase_ == Phase::Submitted || phase_ == Phase::Retired;
            };
            if (deadline == Deadline::max()) {
                changed_.wait(lock, ready);
            } else if (!changed_.wait_until(lock, deadline, ready)) {
                return false;
            }
            if (error_) std::rethrow_exception(error_);
            if (phase_ == Phase::Retired) return true;
            run = run_;
        }
        return run->wait_until(deadline);
    }

    // Called by the lifetime owner only after wait establishes safe completion.
    void retire() {
        validate_process();
        std::lock_guard<std::mutex> lock(mutex_);
        if (phase_ == Phase::Pending || phase_ == Phase::Submitting)
            throw std::logic_error("cannot retire a pending submission");
        run_.reset();
        phase_ = Phase::Retired;
    }

private:
    void validate_process() const {
        if (getpid() != process_) throw std::logic_error("submission token belongs to another process");
    }
    enum class Phase { Pending, Submitting, Submitted, Retired };
    const pid_t process_ = getpid();
    std::mutex mutex_;
    std::condition_variable changed_;
    Phase phase_ = Phase::Pending;
    std::function<Run()> action_;
    std::optional<Run> run_;
    std::exception_ptr error_;
};
