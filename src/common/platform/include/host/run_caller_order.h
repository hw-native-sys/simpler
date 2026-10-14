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

#include <array>
#include <functional>
#include <mutex>
#include <utility>

#include "run_completion_fence.h"

/**
 * One slot's caller fork and exit-wait proof. Kernel completion belongs to
 * RunCompletionFence; this owner holds its AICPU boundary until the caller
 * has consumed the wait. No caller stream is owned or destroyed here.
 */
class RunCallerOrder {
public:
    using StreamWait = std::function<int(void *, void *)>;
    using StreamSync = std::function<int(void *, int)>;
    using Completion = RunCompletionFence::Completion;
    using Role = RunCompletionFence::StreamRole;

    RunCallerOrder(RunCompletionFence::DeviceEventOps ops, StreamWait stream_wait, StreamSync sync) :
        ops_(std::move(ops)),
        stream_wait_(std::move(stream_wait)),
        sync_(std::move(sync)) {}

    int arm(const NativeRunIdentity &id, void *caller, void *aicpu, void *aicore) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_ || caller == nullptr || aicpu == nullptr || aicore == nullptr || caller == aicpu ||
            caller == aicore)
            return PTO_RUNTIME_ERR_INVALID_STATE;
        for (void *&event : events_) {
            if (event != nullptr) continue;
            int rc = ops_.create(&event);
            if (rc != 0) return rc;
        }
        identity_ = id;
        streams_ = {caller, aicpu, aicore};
        active_ = true;
        return 0;
    }

    int fork(const NativeRunIdentity &id, LaunchProgressSink &sink) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(id) || submitted_) return PTO_RUNTIME_ERR_INVALID_STATE;
        // An error return does not prove that an attempted record/wait was
        // rejected. From this point rollback requires a completion proof.
        submitted_ = true;
        sink.mark_submitted();
        int rc = ops_.record(events_[Start], streams_[Caller]);
        if (rc != 0) return rc;
        rc = stream_wait_(streams_[Aicpu], events_[Start]);
        if (rc != 0) return rc;
        rc = ops_.record(events_[Fork], streams_[Aicpu]);
        if (rc != 0) return rc;
        return stream_wait_(streams_[Aicore], events_[Fork]);
    }

    // Called after the AICPU boundary is recorded behind its own AICore wait.
    int join(const NativeRunIdentity &id, RunCompletionFence &fence) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!owns(id) || !submitted_ || reference_.valid()) return PTO_RUNTIME_ERR_INVALID_STATE;
        int rc = fence.reserve_wait_reference(id, Role::Aicpu, Role::Caller, &reference_);
        if (rc != 0) return rc;
        fence_ = &fence;
        rc = stream_wait_(streams_[Caller], fence.boundary_event(id, Role::Aicpu));
        const int commit_rc = fence.commit_wait_reference(reference_);
        if (rc != 0) return rc;
        if (commit_rc != 0) return commit_rc;
        rc = ops_.record(events_[Tail], streams_[Caller]);
        tail_recorded_ = rc == 0;
        return rc;
    }

    Completion poll(const NativeRunIdentity &id) {
        std::unique_lock<std::mutex> lock(mutex_, std::try_to_lock);
        if (!lock.owns_lock()) return Completion::Pending;
        if (!owns(id)) return Completion::Error;
        if (!submitted_ || consumed_) return Completion::Complete;
        if (!tail_recorded_) return Completion::Unfenced;
        bool complete = false;
        if (ops_.query(events_[Tail], &complete) != 0) return Completion::Error;
        return complete ? Completion::Complete : Completion::Pending;
    }

    int discharge(const NativeRunIdentity &id, int timeout_ms) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_) return 0;
        if (!owns(id)) return PTO_RUNTIME_ERR_INVALID_STATE;
        if (submitted_ && !consumed_) {
            if (tail_recorded_) {
                int rc = ops_.wait(events_[Tail], timeout_ms);
                if (rc != 0) return rc;
            } else {
                // Partial enqueue has no tail proof. Both private streams and
                // the borrowed caller may still name a fork or completion event.
                int first = 0;
                for (void *stream : streams_) {
                    const int rc = sync_(stream, timeout_ms);
                    if (first == 0) first = rc;
                }
                if (first != 0) return first;
            }
            consumed_ = true;
        }
        if (reference_.valid()) {
            int rc = reference_.committed() ? fence_->release_wait_reference(reference_, Role::Caller) :
                                              fence_->revoke_wait_reference(reference_);
            if (rc != 0) return rc;
        }
        return 0;
    }

    int retire(const NativeRunIdentity &id) {
        std::lock_guard<std::mutex> lock(mutex_);
        if (!active_) return 0;
        if (!owns(id) || reference_.valid() || (submitted_ && !consumed_)) return PTO_RUNTIME_ERR_INVALID_STATE;
        active_ = submitted_ = tail_recorded_ = consumed_ = false;
        streams_ = {};
        fence_ = nullptr;
        return 0;
    }

    int release() {
        std::lock_guard<std::mutex> lock(mutex_);
        if (active_) return PTO_RUNTIME_ERR_INVALID_STATE;
        int first = 0;
        for (void *&event : events_) {
            if (event == nullptr) continue;
            int rc = ops_.destroy(event);
            if (rc == 0) event = nullptr;
            else if (first == 0) first = rc;
        }
        return first;
    }

    // No device calls: the caller has ended or quarantined the generation.
    void abandon() {
        std::lock_guard<std::mutex> lock(mutex_);
        events_ = {};
        streams_ = {};
        active_ = submitted_ = tail_recorded_ = consumed_ = false;
        if (fence_ != nullptr) fence_->discard_stale_wait_reference(reference_);
        fence_ = nullptr;
    }

private:
    static constexpr size_t Start = 0, Fork = 1, Tail = 2;
    static constexpr size_t Caller = 0, Aicpu = 1, Aicore = 2;
    bool owns(const NativeRunIdentity &id) const { return active_ && identity_ == id; }
    std::mutex mutex_;
    RunCompletionFence::DeviceEventOps ops_;
    StreamWait stream_wait_;
    StreamSync sync_;
    std::array<void *, 3> events_{};
    std::array<void *, 3> streams_{};
    NativeRunIdentity identity_{};
    RunCompletionFence *fence_{nullptr};
    RunCompletionFence::WaitReference reference_;
    bool active_{false};
    bool submitted_{false};
    bool tail_recorded_{false};
    bool consumed_{false};
};
