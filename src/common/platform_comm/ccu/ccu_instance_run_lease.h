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

#include "ccu_registry.h"
#include "ccu_run_lease.h"

namespace simpler::ccu {

// One NativeRunContext owns this lease until CCU stops and all task/output readers exit.
class InstanceRunLease final : public RunLease {
public:
    // The attaching runtime supplies its canonical Host error code, never a device error code.
    InstanceRunLease(std::vector<std::shared_ptr<Instance>> instances, int host_error_code) :
        instances_(std::move(instances)),
        host_error_code_(host_error_code) {
        if (host_error_code_ == 0) throw std::invalid_argument("CCU run lease requires a nonzero Host error code");
        size_t retained = 0;
        try {
            for (const auto &instance : instances_) {
                if (!instance) throw std::invalid_argument("CCU run lease requires live instances");
                instance->retain_run();
                ++retained;
            }
        } catch (...) {
            while (retained)
                instances_[--retained]->rollback_run_reference();
            throw;
        }
    }

    ~InstanceRunLease() override {
        // The final NativeRunContext/PreparedExecution owner disappears only after its readers exit.
        if (!released_ && completion_proven_) {
            try {
                (void)release_after_readers(!failed_);
            } catch (...) {}
        }
        if (!released_) {
            // Neither object destruction nor a stream fence proves CCU's last access.
            auto &debt = quarantine();
            std::lock_guard<std::mutex> lock(debt.mutex);
            debt.leases.push_back(std::move(instances_));
        }
    }
    InstanceRunLease(const InstanceRunLease &) = delete;
    InstanceRunLease &operator=(const InstanceRunLease &) = delete;

    // Native retirement calls these overrides only after ordinary producers have stopped.
    ReaderResult poll() noexcept override {
        try {
            const auto result = poll_completion(true);
            if (result == PollResult::COMPLETE) return {ReaderState::STOPPED, failed() ? host_error_code_ : 0};
            if (result == PollResult::PENDING) return {ReaderState::PENDING, failed() ? host_error_code_ : 0};
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
        }
        return {ReaderState::UNCONFIRMED, host_error_code_};
    }

    ReaderResult drain() noexcept override {
        try {
            if (drain_completion(true)) return {ReaderState::STOPPED, failed() ? host_error_code_ : 0};
        } catch (...) {
            std::lock_guard<std::mutex> lock(mutex_);
            failed_ = true;
        }
        return {ReaderState::UNCONFIRMED, host_error_code_};
    }

    PollResult poll_completion(bool ordinary_stream_complete) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completion_proven_) return PollResult::COMPLETE;
        if (!ordinary_stream_complete) return PollResult::PENDING;
        bool complete = true;
        bool failed = false;
        for (const auto &instance : instances_) {
            const auto result = instance->poll_run_completion(true);
            if (result != PollResult::COMPLETE) complete = false;
            if (result == PollResult::FAILED) failed = true;
        }
        failed_ = failed_ || failed;
        completion_proven_ = complete;
        return failed ? PollResult::FAILED : complete ? PollResult::COMPLETE : PollResult::PENDING;
    }

    bool drain_completion(bool ordinary_stream_complete) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (completion_proven_) return true;
        if (!ordinary_stream_complete) return false;
        bool confirmed = true;
        for (const auto &instance : instances_) {
            if (!instance->drain_run_completion(true)) confirmed = false;
        }
        if (!confirmed) failed_ = true;
        completion_proven_ = confirmed;
        return confirmed;
    }

    // The caller proves Host/task/output readers are gone independently of CCU completion.
    bool release_after_readers(bool success) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (released_) return true;
        if (!completion_proven_) return false;
        while (released_count_ < instances_.size()) {
            instances_[released_count_]->release_run(success && !failed_);
            ++released_count_;
        }
        instances_.clear();
        released_ = true;
        return true;
    }

    bool failed() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return failed_;
    }

private:
    struct Quarantine {
        std::mutex mutex;
        std::vector<std::vector<std::shared_ptr<Instance>>> leases;
    };
    static Quarantine &quarantine() {
        static auto *debt = new Quarantine;
        return *debt;
    }

    mutable std::mutex mutex_;
    std::vector<std::shared_ptr<Instance>> instances_;
    const int host_error_code_;
    size_t released_count_ = 0;
    bool completion_proven_ = false;
    bool failed_ = false;
    bool released_ = false;
};

}  // namespace simpler::ccu
