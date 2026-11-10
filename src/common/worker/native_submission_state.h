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

#include <atomic>
#include <exception>
#include <memory>

#include "chip_run_lane.h"
#include "native_submission.h"
#include "run_resource_arena.h"

struct AclCaptureModelNotice;
class CallArgumentSnapshot;
class ChipWorker;
struct CallConfig;

enum class NativeSubmissionDisposition : int { Pending, Submitting, Submitted, Cancelled, Retired };

struct NativeSubmissionCleanup {
    NativeSubmissionDisposition disposition{NativeSubmissionDisposition::Pending};
    bool carrier_retired{false};
};

// Native invocation storage shared by every view of the same capsule token.
// The framework graph remains the model owner.
class NativeSubmissionState : public std::enable_shared_from_this<NativeSubmissionState> {
public:
    NativeSubmissionState(
        ChipWorker &worker, std::shared_ptr<ChipWorker> owner, int32_t callable_id, const CallArgumentSnapshot &values,
        const CallConfig &config, void *caller_stream, bool framework_eligible
    );
    ~NativeSubmissionState() noexcept;
    NativeSubmissionState(const NativeSubmissionState &) = delete;
    NativeSubmissionState &operator=(const NativeSubmissionState &) = delete;

    void submit();
    // The stock graph's first token anchors its exact model generation. Other
    // tokens from that scope reuse the notice, never a bare reused handle.
    void retain_model_resources(int &registration_started, const NativeSubmissionState *model_anchor = nullptr);
    bool model_destroyed() const;
    void cancel_pending(NativeSubmissionCleanup &outcome);
    void release_resources();

    bool cancel(std::exception_ptr reason);
    bool done();
    bool wait_until(ChipRun::Deadline deadline);
    void claim_framework();
    void note_export();
    bool framework_claimed();
    bool framework_transferred();
    bool retire_framework(bool wait_for_own_completion, ChipRun::Deadline deadline);
    bool wait_framework_until(ChipRun::Deadline deadline);
    bool owner_recovery_required();
    void retire();

private:
    struct Backing;
    std::unique_ptr<Backing> backing_;

    void *active_acl_model() const;
    bool carrier_retired() const;
    void validate_process() const;
    void finish_operation() noexcept;
};

// The holder's intrusive count pins the same genuine native State control block;
// neither its callbacks nor its final destruction need a Python owner/caster.
struct NativeSubmissionCapsule {
    explicit NativeSubmissionCapsule(std::shared_ptr<NativeSubmissionState> state);
    static void retain(void *) noexcept;
    static void release(void *) noexcept;
    static int submit(void *, char *, size_t) noexcept;

    SimplerNativeSubmission api{};
    std::atomic<size_t> references{1};
    std::shared_ptr<NativeSubmissionState> state;
};
