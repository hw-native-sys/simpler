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

#include <cstdint>
#include <utility>

class DeviceRunnerBase;
class NativeRunExecutionTestPeer;
class RunRetentionProbePeer;
class SimDeviceRunnerBase;

struct NativeRunIdentity {
    uint64_t run_epoch{0};
    uint64_t generation{0};
    uint64_t dispatch_id{0};
    uint32_t pipeline_slot{0};

    bool operator==(const NativeRunIdentity &other) const {
        return run_epoch == other.run_epoch && generation == other.generation && dispatch_id == other.dispatch_id &&
               pipeline_slot == other.pipeline_slot;
    }
    bool operator!=(const NativeRunIdentity &other) const { return !(*this == other); }
};

/**
 * A successor's statement that it has been ordered behind one named predecessor
 * on the device.
 *
 * Both halves are load-bearing. The owner pointer is what the execution claim
 * matches against its newest holder, so a join can only ever name the run
 * immediately ahead of this one; the identity is what makes that holder the run
 * the caller meant rather than whichever run happens to occupy the slot, since a
 * pipeline slot is reused and only `run_epoch` is unique for the process
 * lifetime.
 *
 * Carrying no device handle is deliberate: the predecessor's completion
 * boundary is owned by its own slot's fence, and the successor reaches it
 * through the identity rather than through a copied event.
 */
struct NativeRunJoin {
    const void *predecessor_owner{nullptr};
    NativeRunIdentity predecessor_identity{};
};

class LaunchPermit {
public:
    LaunchPermit() = default;
    LaunchPermit(const LaunchPermit &) = delete;
    LaunchPermit &operator=(const LaunchPermit &) = delete;

    LaunchPermit(LaunchPermit &&other) noexcept :
        identity_(other.identity_),
        valid_(other.valid_) {
        other.valid_ = false;
    }

    LaunchPermit &operator=(LaunchPermit &&other) noexcept {
        if (this == &other) return *this;
        identity_ = other.identity_;
        valid_ = other.valid_;
        other.valid_ = false;
        return *this;
    }

    bool valid() const { return valid_; }

private:
    friend class DeviceRunnerBase;
    friend class NativeRunExecutionTestPeer;
    // #2267's retention probe launches a run that holds no C API execution
    // claim, so it mints its permit instead of receiving one.
    friend class RunRetentionProbePeer;
    friend class SimDeviceRunnerBase;
    template <typename AicoreSubmit, typename AicpuSubmit>
    friend struct ExactLaunchTransaction;

    explicit LaunchPermit(const NativeRunIdentity &identity) :
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

class LaunchReceipt {
public:
    LaunchReceipt() = default;
    LaunchReceipt(const LaunchReceipt &) = delete;
    LaunchReceipt &operator=(const LaunchReceipt &) = delete;

    LaunchReceipt(LaunchReceipt &&other) noexcept :
        identity_(other.identity_),
        valid_(other.valid_) {
        other.valid_ = false;
    }

    LaunchReceipt &operator=(LaunchReceipt &&other) noexcept {
        if (this == &other) return *this;
        identity_ = other.identity_;
        valid_ = other.valid_;
        other.valid_ = false;
        return *this;
    }

    bool valid() const { return valid_; }
    bool matches(const NativeRunIdentity &identity) const { return valid_ && identity_ == identity; }

private:
    template <typename AicoreSubmit, typename AicpuSubmit>
    friend struct ExactLaunchTransaction;

    explicit LaunchReceipt(const NativeRunIdentity &identity) :
        identity_(identity),
        valid_(true) {}

    NativeRunIdentity identity_{};
    bool valid_{false};
};

enum class LaunchProgress : uint8_t {
    NotStarted,
    Partial,
    Complete,
};

/**
 * A submit callback's channel for reporting that its submission reached the
 * device queue.
 *
 * A callback does more than one thing — it submits a kernel and then records
 * that kernel's completion boundary — and only the first of those is what
 * grades the run. Without this, a boundary-record failure returned from the
 * callback would be read as a failure before any submission and the run graded
 * `NotStarted`, i.e. an already-submitted kernel reported as never launched.
 * Callbacks therefore mark this the instant the device accepts a submission,
 * ahead of anything that can still fail.
 */
class LaunchProgressSink {
public:
    void mark_submitted() { submitted_ = true; }
    bool submitted() const { return submitted_; }

private:
    bool submitted_{false};
};

struct LaunchTransactionResult {
    int rc{-1};
    LaunchProgress progress{LaunchProgress::NotStarted};
    LaunchReceipt receipt{};

    bool poisoned() const { return progress == LaunchProgress::Partial; }
};

/**
 * The exact two-submission launch transaction.
 *
 * Consumes an identity-bound permit, then submits AICore and AICPU in that
 * order. Only both submissions succeeding produces a `LaunchReceipt`.
 *
 * A failure before the first execution-visible stream submission is safe: the
 * run touched no stream, so it rolls back and admission stays clean. Once the
 * AICore submission succeeds, an AICPU failure leaves execution uncertain, so
 * the run is `Partial` — resources stay retained and admission is poisoned
 * until a later drain or teardown proves quiescence.
 *
 * Callback contract: a submit callback owns everything it does before it
 * attempts its real device submission, and must report such a failure as a
 * non-zero return rather than an exception. An escaping exception is therefore
 * evidence that the submission itself was attempted, and grades the run
 * `Partial`. Returning non-zero from `submit_aicore` is by contrast a
 * before-first-submission failure, so it stays `NotStarted` — unless the
 * callback marked its `LaunchProgressSink`, which says the device already
 * accepted a submission and the failure came from a later step.
 */
template <typename AicoreSubmit, typename AicpuSubmit>
struct ExactLaunchTransaction {
    static LaunchTransactionResult
    run(const NativeRunIdentity &identity, LaunchPermit permit, AicoreSubmit &&submit_aicore,
        AicpuSubmit &&submit_aicpu) {
        LaunchTransactionResult result;
        if (!permit.consume(identity)) return result;

        LaunchProgressSink sink;
        try {
            result.rc = std::forward<AicoreSubmit>(submit_aicore)(sink);
        } catch (...) {
            result.rc = -1;
            result.progress = LaunchProgress::Partial;
            return result;
        }
        if (result.rc != 0) {
            if (sink.submitted()) result.progress = LaunchProgress::Partial;
            return result;
        }

        result.progress = LaunchProgress::Partial;
        try {
            result.rc = std::forward<AicpuSubmit>(submit_aicpu)(sink);
        } catch (...) {
            result.rc = -1;
        }
        if (result.rc != 0) return result;

        result.progress = LaunchProgress::Complete;
        result.receipt = LaunchReceipt(identity);
        return result;
    }
};

template <typename AicoreSubmit, typename AicpuSubmit>
LaunchTransactionResult exact_launch_transaction(
    const NativeRunIdentity &identity, LaunchPermit permit, AicoreSubmit &&submit_aicore, AicpuSubmit &&submit_aicpu
) {
    return ExactLaunchTransaction<AicoreSubmit, AicpuSubmit>::run(
        identity, std::move(permit), std::forward<AicoreSubmit>(submit_aicore), std::forward<AicpuSubmit>(submit_aicpu)
    );
}
