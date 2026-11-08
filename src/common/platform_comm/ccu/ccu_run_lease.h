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

#include <memory>

namespace simpler::ccu {

// Host-only ownership bridge. The provider retains every selected instance,
// module, domain, window and directory version for this run. STOPPED means no
// CCU can access run memory, including requests lost before AIV FIN. An ordinary
// AIV/AICPU stream fence alone never supplies that proof.
enum class ReaderState { PENDING, STOPPED, UNCONFIRMED };
struct ReaderResult {
    ReaderState state;
    int error_code;
};

class RunLease {
public:
    virtual ~RunLease() = default;
    virtual ReaderResult poll() noexcept = 0;
    // Called only after the ordinary producer stopped. Refuses new submissions,
    // attempts a bounded drain, and quarantines the provider on UNCONFIRMED.
    virtual ReaderResult drain() noexcept = 0;
    // Only for a preparation whose ordinary producer provably never launched.
    // This cancels this run's ownership, without stopping other runs of the instance.
    virtual ReaderResult cancel_unlaunched() noexcept { return drain(); }
};

class RunReaders {
public:
    // Set during prepare, before any device launch. A declaring callable with
    // no provider must be rejected by Host registration; null means no CCU use.
    std::shared_ptr<RunLease> lease;

    ReaderResult poll() noexcept { return observe(false); }
    ReaderResult drain() noexcept { return observe(true); }
    ReaderResult cancel_unlaunched() noexcept {
        if (!lease || stopped_) return {ReaderState::STOPPED, error_code_};
        const auto result = lease->cancel_unlaunched();
        if (result.state == ReaderState::STOPPED) {
            stopped_ = true;
            error_code_ = result.error_code;
        }
        return result;
    }
    bool stopped() const noexcept { return !lease || stopped_; }

private:
    ReaderResult observe(bool draining) noexcept {
        if (!lease) return {ReaderState::STOPPED, 0};
        if (stopped_) return {ReaderState::STOPPED, error_code_};
        const auto result = draining ? lease->drain() : lease->poll();
        if (result.state == ReaderState::STOPPED) {
            stopped_ = true;
            error_code_ = result.error_code;
        }
        return result;
    }
    bool stopped_{false};
    int error_code_{0};
};

}  // namespace simpler::ccu
