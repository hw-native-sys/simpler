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

#include "host/kernel_execution_state.h"

int KernelExecutionState::initialize(int requested_device_id, const KernelContextOps &ops) {
    std::scoped_lock lock(mutex_);
    if (phase_ != KernelContextPhase::New) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (requested_device_id < 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    if (!ops.valid()) return PTO_RUNTIME_ERR_INTERNAL;

    int current_device = -1;
    int rc = ops.get_current_device(ops.context, &current_device);
    if (rc != 0) return rc;
    if (current_device != requested_device_id) return PTO_RUNTIME_ERR_INVALID_STATE;

    phase_ = KernelContextPhase::Initializing;
    ops_ = ops;
    device_id_ = requested_device_id;

    for (auto &stream : hidden_streams_) {
        rc = ops_.create_hidden_stream(ops_.context, &stream);
        if (rc != 0) break;
    }
    if (rc == 0) {
        for (auto &event : events_) {
            rc = ops_.create_event(ops_.context, &event);
            if (rc != 0) break;
        }
    }
    if (rc != 0) {
        const int cleanup_rc = cleanup_owned_resources_locked();
        if (cleanup_rc != 0) {
            if (unexpected_teardown_error_ == 0) unexpected_teardown_error_ = cleanup_rc;
            phase_ = KernelContextPhase::Closing;
        } else {
            phase_ = KernelContextPhase::New;
            device_id_ = -1;
            ops_ = {};
        }
        return rc;
    }

    phase_ = KernelContextPhase::Collecting;
    return 0;
}

int KernelExecutionState::mark_ready_enqueued() {
    std::scoped_lock lock(mutex_);
    if (phase_ != KernelContextPhase::Collecting && phase_ != KernelContextPhase::ReadyEnqueued) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    phase_ = KernelContextPhase::ReadyEnqueued;
    return 0;
}

void KernelExecutionState::poison(int runtime_error) {
    std::scoped_lock lock(mutex_);
    if (phase_ != KernelContextPhase::Collecting && phase_ != KernelContextPhase::ReadyEnqueued) return;
    phase_ = KernelContextPhase::Poisoned;
    if (last_runtime_error_ == 0) last_runtime_error_ = runtime_error;
}

int KernelExecutionState::close() {
    std::scoped_lock lock(mutex_);
    switch (phase_) {
    case KernelContextPhase::New:
        phase_ = KernelContextPhase::Closed;
        return 0;
    case KernelContextPhase::Closed:
        return 0;
    case KernelContextPhase::Initializing:
        return PTO_RUNTIME_ERR_INVALID_STATE;
    case KernelContextPhase::Collecting:
    case KernelContextPhase::ReadyEnqueued:
    case KernelContextPhase::Poisoned:
    case KernelContextPhase::Closing:
        break;
    }
    phase_ = KernelContextPhase::Closing;
    const int rc = cleanup_owned_resources_locked();
    if (rc != 0) {
        if (unexpected_teardown_error_ == 0) unexpected_teardown_error_ = rc;
        return rc;
    }
    phase_ = KernelContextPhase::Closed;
    return 0;
}

int KernelExecutionState::cleanup_owned_resources_locked() {
    int first_error = 0;
    for (size_t i = events_.size(); i > 0; --i) {
        void *&event = events_[i - 1];
        if (event == nullptr) continue;
        const int rc = ops_.destroy_event(ops_.context, event);
        if (rc != 0) {
            if (first_error == 0) first_error = rc;
            continue;
        }
        event = nullptr;
    }
    for (size_t i = hidden_streams_.size(); i > 0; --i) {
        void *&stream = hidden_streams_[i - 1];
        if (stream == nullptr) continue;
        const int rc = ops_.destroy_hidden_stream(ops_.context, stream);
        if (rc != 0) {
            if (first_error == 0) first_error = rc;
            continue;
        }
        stream = nullptr;
    }
    return first_error;
}

KernelContextPhase KernelExecutionState::phase() const {
    std::scoped_lock lock(mutex_);
    return phase_;
}

bool KernelExecutionState::accepts_dispatch() const {
    std::scoped_lock lock(mutex_);
    return phase_ == KernelContextPhase::Collecting || phase_ == KernelContextPhase::ReadyEnqueued;
}

int KernelExecutionState::device_id() const {
    std::scoped_lock lock(mutex_);
    return device_id_;
}

int KernelExecutionState::last_runtime_error() const {
    std::scoped_lock lock(mutex_);
    return last_runtime_error_;
}

int KernelExecutionState::unexpected_teardown_error() const {
    std::scoped_lock lock(mutex_);
    return unexpected_teardown_error_;
}

bool KernelExecutionState::has_live_resources() const {
    std::scoped_lock lock(mutex_);
    return has_live_resources_locked();
}

bool KernelExecutionState::has_live_resources_locked() const {
    for (void *stream : hidden_streams_) {
        if (stream != nullptr) return true;
    }
    for (void *event : events_) {
        if (event != nullptr) return true;
    }
    return false;
}
