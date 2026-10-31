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

// A graph can retain device references without having executed. Only its
// explicit release, after graph destruction and stream quiescence, settles
// future use. Observed execution is separate from that physical stop proof.
class RetainedDeviceUse {
public:
    enum class State { Empty, Retained, Quiescent };

    void retain(const NativeRunIdentity &identity) {
        identity_ = identity;
        state_ = State::Retained;
    }
    bool observe(uint64_t epoch) {
        if (state_ != State::Retained || epoch < observed_epoch_) return false;
        observed_epoch_ = epoch;
        return true;
    }
    bool settle(const NativeRunIdentity &identity) {
        if (state_ != State::Retained || identity != identity_) return false;
        state_ = State::Quiescent;
        return true;
    }
    bool retained() const { return state_ != State::Empty; }
    bool observed_execution() const { return observed_epoch_ != 0; }
    bool can_finalize(const NativeRunIdentity &identity) const {
        return state_ == State::Empty || (state_ == State::Quiescent && identity == identity_);
    }

private:
    NativeRunIdentity identity_{};
    State state_{State::Empty};
    uint64_t observed_epoch_{0};
};
