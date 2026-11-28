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

#include <acl/acl.h>
#include <array>
#include <cstring>
#include <mutex>
#include <new>
#include <optional>

#include "common/execution_gate.h"
#include "host/producer_never_started.h"
#include "worker/native_run_execution.h"

// Device storage remains owned until both entry streams and the release stream
// have proved last use. A failed ACL operation is never retried implicitly.
class NativeExecutionGate {
public:
    explicit NativeExecutionGate(NativeRunIdentity identity, int32_t timeout_ms) :
        identity_(identity),
        timeout_ms_(timeout_ms) {}
    NativeExecutionGate(const NativeExecutionGate &) = delete;
    NativeExecutionGate &operator=(const NativeExecutionGate &) = delete;

    ~NativeExecutionGate() {
        if (!wait_attempted_ && !cleanup_failed_) (void)retire(false);
    }

    const NativeRunIdentity &identity() const { return identity_; }
    uint64_t device_address() const { return reinterpret_cast<uint64_t>(device_); }
    aclrtStream aicpu_stream() const { return entry_streams_[0]; }
    aclrtStream aicore_stream() const { return entry_streams_[1]; }

    int initialize() {
        int rc = aclrtCreateStream(&release_stream_);
        if (rc != 0) return fail(rc);
        for (auto &stream : entry_streams_) {
            rc = aclrtCreateStream(&stream);
            if (rc != 0) return fail(rc);
        }
        rc = aclrtMallocHost(reinterpret_cast<void **>(&host_), sizeof(ExecutionGateRecord));
        if (rc != 0) return fail(rc);
        new (host_) ExecutionGateRecord{};
        host_->header.identity = identity_.run_epoch;
        host_->header.generation = identity_.generation;
        rc = aclrtMalloc(&device_, sizeof(ExecutionGateRecord), ACL_MEM_MALLOC_NORMAL_ONLY);
        if (rc != 0) return fail(rc);
        copy_attempted_ = true;
        rc = aclrtMemcpyAsync(
            device_, sizeof(ExecutionGateRecord), host_, sizeof(ExecutionGateRecord), ACL_MEMCPY_HOST_TO_DEVICE,
            release_stream_
        );
        if (rc == 0) rc = aclrtSynchronizeStreamWithTimeout(release_stream_, timeout_ms_);
        if (rc != 0) return fail(rc);
        for (auto &notify : notifications_) {
            rc = aclrtCreateNotify(&notify, ACL_NOTIFY_DEFAULT);
            if (rc != 0) return fail(rc);
        }
        return 0;
    }

    int enqueue_waits(void *aicpu, void *aicore, LaunchProgressSink &sink) {
        std::lock_guard<std::mutex> lock(mu_);
        if (wait_attempted_ || failure_ != 0 || device_ == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
        // A failed wait submission has unknown acceptance. The launch
        // transaction must retain this owner even when no entry was launched.
        wait_attempted_ = true;
        sink.mark_submitted();
        int rc = aclrtWaitAndResetNotify(notifications_[0], aicpu, static_cast<uint32_t>(timeout_ms_));
        if (rc == 0) rc = aclrtWaitAndResetNotify(notifications_[1], aicore, static_cast<uint32_t>(timeout_ms_));
        if (rc != 0) return fail(rc);
        return 0;
    }

    int release(ExecutionGateDecision decision) {
        std::lock_guard<std::mutex> lock(mu_);
        if (decision != ExecutionGateDecision::Go && decision != ExecutionGateDecision::Cancel)
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        if (release_attempted_) {
            if (decision != decision_) return PTO_RUNTIME_ERR_INVALID_STATE;
            return release_rc_;
        }
        if (host_ == nullptr || release_stream_ == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
        release_attempted_ = true;
        decision_ = decision;
        host_->header.decision = static_cast<uint32_t>(decision);
        int rc = aclrtMemcpyAsync(
            device_, sizeof(ExecutionGateHeader), &host_->header, sizeof(ExecutionGateHeader),
            ACL_MEMCPY_HOST_TO_DEVICE, release_stream_
        );
        // Even a failed publication must unblock every possibly queued waiter.
        // Its entry rejects a still-pending/invalid header without user work.
        for (auto notify : notifications_) {
            if (notify == nullptr) continue;
            const int notify_rc = aclrtRecordNotify(notify, release_stream_);
            if (rc == 0) rc = notify_rc;
        }
        release_rc_ = rc;
        if (rc != 0) failure_ = rc;
        return rc;
    }

    bool never_submitted() const {
        std::lock_guard<std::mutex> lock(mu_);
        return !wait_attempted_;
    }

private:
    friend class DeviceRunnerBase;
    std::optional<ProducerNeverStartedProof>
    prove_producer_never_started(const NativeRunIdentity &identity, bool expect_aic, bool expect_aiv) {
        std::lock_guard<std::mutex> lock(mu_);
        if (identity != identity_ || !wait_attempted_ || !release_attempted_ ||
            decision_ != ExecutionGateDecision::Cancel || release_rc_ != 0 || failure_ != 0)
            return std::nullopt;
        // Do not accept a caller-supplied "complete" boolean as proof.
        // These private streams contain only this run's waits and entries.
        for (auto stream : entry_streams_) {
            if (stream == nullptr) return std::nullopt;
            const int rc = aclrtSynchronizeStreamWithTimeout(stream, timeout_ms_);
            if (rc != 0) {
                fail(rc);
                return std::nullopt;
            }
        }
        // Fence publication and read every expected role under the same owner
        // lock, so concurrent retirement cannot remove the record in between.
        if (validate_completed_locked(expect_aic, expect_aiv) != 0) return std::nullopt;
        return ProducerNeverStartedProof(identity_);
    }

public:
    bool pending() const {
        std::lock_guard<std::mutex> lock(mu_);
        return !release_attempted_;
    }

    bool cancelled() const {
        std::lock_guard<std::mutex> lock(mu_);
        return release_attempted_ && decision_ == ExecutionGateDecision::Cancel;
    }

    int validate_completed(bool expect_aic, bool expect_aiv) {
        std::lock_guard<std::mutex> lock(mu_);
        return validate_completed_locked(expect_aic, expect_aiv);
    }

    int retire(bool complete) {
        std::lock_guard<std::mutex> lock(mu_);
        if (cleanup_failed_ || (wait_attempted_ && (failure_ != 0 || !(complete && completed_))))
            return PTO_RUNTIME_ERR_INVALID_STATE;
        if ((release_attempted_ || copy_attempted_) && release_stream_ != nullptr) {
            const int rc = aclrtSynchronizeStreamWithTimeout(release_stream_, timeout_ms_);
            if (rc != 0) {
                cleanup_failed_ = true;
                return fail(rc);
            }
        }
        for (auto &stream : entry_streams_) {
            if (stream == nullptr) continue;
            const int rc = aclrtDestroyStream(stream);
            if (rc != 0) {
                cleanup_failed_ = true;
                return fail(rc);
            }
            stream = nullptr;
        }
        for (auto &notify : notifications_) {
            if (notify == nullptr) continue;
            const int rc = aclrtDestroyNotify(notify);
            if (rc != 0) {
                cleanup_failed_ = true;
                return fail(rc);
            }
            notify = nullptr;
        }
        if (device_ != nullptr) {
            const int rc = aclrtFree(device_);
            if (rc != 0) {
                cleanup_failed_ = true;
                return fail(rc);
            }
            device_ = nullptr;
        }
        if (host_ != nullptr) {
            const int rc = aclrtFreeHost(host_);
            if (rc != 0) {
                cleanup_failed_ = true;
                return fail(rc);
            }
            host_ = nullptr;
        }
        if (release_stream_ != nullptr) {
            const int rc = aclrtDestroyStream(release_stream_);
            if (rc != 0) {
                cleanup_failed_ = true;
                return fail(rc);
            }
            release_stream_ = nullptr;
        }
        failure_ = 0;
        return 0;
    }

private:
    int validate_completed_locked(bool expect_aic, bool expect_aiv) {
        if (!release_attempted_ || release_rc_ != 0 || failure_ != 0) return PTO_RUNTIME_ERR_INVALID_STATE;
        int rc = aclrtMemcpyAsync(
            host_, sizeof(ExecutionGateRecord), device_, sizeof(ExecutionGateRecord), ACL_MEMCPY_DEVICE_TO_HOST,
            release_stream_
        );
        if (rc == 0) rc = aclrtSynchronizeStreamWithTimeout(release_stream_, timeout_ms_);
        if (rc != 0) return fail(rc);
        const auto outcome = static_cast<uint32_t>(
            decision_ == ExecutionGateDecision::Go ? ExecutionGateOutcome::Go : ExecutionGateOutcome::Cancel
        );
        if (host_->header.version != EXECUTION_GATE_VERSION ||
            host_->header.decision != static_cast<uint32_t>(decision_) ||
            host_->header.identity != identity_.run_epoch || host_->header.generation != identity_.generation ||
            host_->aicpu.outcome != outcome || (expect_aic && host_->aic.outcome != outcome) ||
            (expect_aiv && host_->aiv.outcome != outcome))
            return fail(PTO_RUNTIME_ERR_INVALID_STATE);
        completed_ = true;
        return 0;
    }

    int fail(int rc) {
        failure_ = rc;
        return rc;
    }
    const NativeRunIdentity identity_;
    const int32_t timeout_ms_;
    mutable std::mutex mu_;
    aclrtStream release_stream_{nullptr};
    std::array<aclrtNotify, 2> notifications_{};
    std::array<aclrtStream, 2> entry_streams_{};
    ExecutionGateRecord *host_{nullptr};
    void *device_{nullptr};
    ExecutionGateDecision decision_{ExecutionGateDecision::Pending};
    bool wait_attempted_{false};
    bool copy_attempted_{false};
    bool cleanup_failed_{false};
    bool release_attempted_{false};
    bool completed_{false};
    int failure_{0};
    int release_rc_{0};
};
