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
#include "native_run_execution.h"

class WseLaunchPermit {
public:
    WseLaunchPermit() = default;
    WseLaunchPermit(const WseLaunchPermit &) = delete;
    WseLaunchPermit &operator=(const WseLaunchPermit &) = delete;

    WseLaunchPermit(WseLaunchPermit &&other) noexcept :
        identity_(other.identity_),
        valid_(other.valid_) {
        other.valid_ = false;
    }

    WseLaunchPermit &operator=(WseLaunchPermit &&other) noexcept {
        if (this == &other) return *this;
        identity_ = other.identity_;
        valid_ = other.valid_;
        other.valid_ = false;
        return *this;
    }

    bool valid() const { return valid_; }

private:
    // Only the owning runner can authorize a single WSE executor submission.
    friend class WseHostRunner;
    template <typename Submit>
    friend struct WseSingleLaunchTransaction;

    explicit WseLaunchPermit(const NativeRunIdentity &identity) :
        identity_(identity),
        valid_(true) {}

    bool consume(const NativeRunIdentity &identity) {
        if (!valid_ || identity_ != identity) return false;
        valid_ = false;
        return true;
    }

    NativeRunIdentity identity_{};
    bool valid_{false};
};

class WseLaunchReceipt {
public:
    WseLaunchReceipt() = default;
    WseLaunchReceipt(const WseLaunchReceipt &) = delete;
    WseLaunchReceipt &operator=(const WseLaunchReceipt &) = delete;

    WseLaunchReceipt(WseLaunchReceipt &&other) noexcept :
        identity_(other.identity_),
        valid_(other.valid_) {
        other.valid_ = false;
    }

    WseLaunchReceipt &operator=(WseLaunchReceipt &&other) noexcept {
        if (this == &other) return *this;
        identity_ = other.identity_;
        valid_ = other.valid_;
        other.valid_ = false;
        return *this;
    }

    bool valid() const { return valid_; }
    bool matches(const NativeRunIdentity &identity) const { return valid_ && identity_ == identity; }

private:
    template <typename Submit>
    friend struct WseSingleLaunchTransaction;

    explicit WseLaunchReceipt(const NativeRunIdentity &identity) :
        identity_(identity),
        valid_(true) {}

    NativeRunIdentity identity_{};
    bool valid_{false};
};

struct WseLaunchTransactionResult {
    int rc{-1};
    LaunchProgress progress{LaunchProgress::NotStarted};
    WseLaunchReceipt receipt{};

    bool poisoned() const { return progress == LaunchProgress::Partial; }
};

// A single executor submission with the same identity/receipt ownership as multi-component launches.
template <typename Submit>
struct WseSingleLaunchTransaction {
    static WseLaunchTransactionResult run(const NativeRunIdentity &identity, WseLaunchPermit permit, Submit &&submit) {
        WseLaunchTransactionResult result;
        if (!permit.consume(identity)) return result;
        LaunchProgressSink sink;
        try {
            result.rc = std::forward<Submit>(submit)(sink);
        } catch (...) {
            result.rc = -1;
            result.progress = LaunchProgress::Partial;
            return result;
        }
        if (result.rc != 0) {
            if (sink.submitted()) result.progress = LaunchProgress::Partial;
            return result;
        }
        result.progress = LaunchProgress::Complete;
        result.receipt = WseLaunchReceipt(identity);
        return result;
    }
};

template <typename Submit>
WseLaunchTransactionResult
wse_launch_transaction(const NativeRunIdentity &identity, WseLaunchPermit permit, Submit &&submit) {
    return WseSingleLaunchTransaction<Submit>::run(identity, std::move(permit), std::forward<Submit>(submit));
}
