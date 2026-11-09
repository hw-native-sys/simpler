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
        action_([action = std::move(action)](bool) {
            return action();
        }) {}

    explicit QueuedSubmission(std::function<Run(bool)> action, bool framework_eligible = true) :
        framework_eligible_(framework_eligible),
        action_(std::move(action)) {}

    void claim_framework() {
        validate_process();
        std::lock_guard<std::mutex> lock(mutex_);
        if (phase_ != Phase::Pending || exported_ || public_observation_ || framework_claimed_)
            throw std::logic_error("framework ownership requires an unexported, unconsumed native token");
        if (!framework_eligible_)
            throw std::logic_error("native token does not support framework observation transfer");
        framework_claimed_ = true;
    }

    void note_export() {
        validate_process();
        std::lock_guard<std::mutex> lock(mutex_);
        exported_ = true;
    }

    bool framework_claimed() {
        validate_process();
        std::lock_guard<std::mutex> lock(mutex_);
        return framework_claimed_;
    }

    bool framework_transferred() {
        validate_process();
        std::lock_guard<std::mutex> lock(mutex_);
        return framework_transferred_;
    }

    void note_framework_transfer() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!framework_claimed_) throw std::logic_error("native submission has no framework owner");
        framework_transferred_ = true;
    }

    template <typename Operation>
    bool with_framework_run(bool wait_for_publication, Operation operation, Deadline deadline = Deadline::max()) {
        validate_process();
        std::optional<Run> run;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (!framework_claimed_) throw std::logic_error("native submission has no framework owner");
            if (wait_for_publication) {
                auto ready = [&] {
                    return phase_ == Phase::Submitted || phase_ == Phase::Retired;
                };
                if (deadline == Deadline::max()) changed_.wait(lock, ready);
                else if (!changed_.wait_until(lock, deadline, ready)) return false;
            }
            if (error_) std::rethrow_exception(error_);
            if (framework_transferred_) return true;
            if (phase_ != Phase::Submitted) return false;
            if (!run_) throw std::logic_error("framework submission has no published native run");
            run = run_;
        }
        return operation(*run);
    }

    void submit() {
        validate_process();
        bool framework = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (phase_ != Phase::Pending) throw std::logic_error("submission token is already consumed or cancelled");
            framework = framework_claimed_;
            phase_ = Phase::Submitting;
        }
        std::optional<Run> result;
        std::exception_ptr error;
        try {
            result.emplace(action_(framework));
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
            if (framework_claimed_) throw std::logic_error("framework-owned token cannot be publicly observed");
            public_observation_ = true;
            if (phase_ == Phase::Retired || error_) return true;
            if (phase_ != Phase::Submitted) return false;
            run = run_;
        }
        return run->done();
    }

    bool wait_until(Deadline deadline) { return wait_until_impl(deadline, false); }
    bool wait_framework_until(Deadline deadline) { return wait_until_impl(deadline, true); }

private:
    bool wait_until_impl(Deadline deadline, bool framework) {
        validate_process();
        std::optional<Run> run;
        {
            std::unique_lock<std::mutex> lock(mutex_);
            if (framework != framework_claimed_)
                throw std::logic_error("native submission observation belongs to a different owner");
            if (!framework) public_observation_ = true;
            auto ready = [&] {
                return phase_ == Phase::Submitted || phase_ == Phase::Retired;
            };
            if (deadline == Deadline::max()) {
                changed_.wait(lock, ready);
            } else if (!changed_.wait_until(lock, deadline, ready)) {
                return false;
            }
            if (error_) std::rethrow_exception(error_);
            if (phase_ == Phase::Retired || (framework && framework_transferred_)) return true;
            run = run_;
        }
        return run->wait_until(deadline);
    }

public:
    bool owner_recovery_required() {
        validate_process();
        std::optional<Run> run;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            run = run_;
        }
        // No returned run supplies no cleanup proof for an arbitrary submit error.
        return !run || run->owner_recovery_required();
    }

    // Internal lifetime-owner operation after safe wait retirement or successful
    // owner-close recovery. done() alone supplies no cleanup proof; an error
    // without a returned Run requires the owner's recovery proof.
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
    bool exported_{false};
    bool public_observation_{false};
    bool framework_claimed_{false};
    bool framework_transferred_{false};
    const bool framework_eligible_{true};
    std::function<Run(bool)> action_;
    std::optional<Run> run_;
    std::exception_ptr error_;
};
