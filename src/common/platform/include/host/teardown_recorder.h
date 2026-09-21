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
/**
 * What a device teardown actually did, recorded as it happens.
 *
 * The facts this keeps are the ones a host-side return code destroys by
 * folding together: whether a reset API ran at all, which one, what it
 * returned, whether a recovery wrapper with a post-reset probe sat above it,
 * and what the teardown as a whole returned. Every one of those is separately
 * observable inside `finalize()` and unrecoverable afterwards.
 *
 * Observation only. Nothing here performs a reset, a synchronize or a probe,
 * and no combination of fields asserts that device work has stopped.
 *
 * Two scopes coexist. `note_stage` is the *last* attempt's stage, so a later
 * attempt overwrites an earlier one; the invocation and attempt counters and
 * the last reset-API return value are cumulative over the whole teardown. That
 * split is what keeps a final PREAMBLE_FAILED — an attempt that never reached
 * a reset call — from implying that no attempt ever did.
 *
 * `child_pid` is left zero here: it is a publication-side generation key the
 * process that owns the transport fills, not something the runner observes.
 */

#pragma once

#include <cstdint>
#include <type_traits>

#include "worker/runtime_c_api.h"

static_assert(
    std::is_trivially_copyable_v<SimplerTeardownReport> && std::is_standard_layout_v<SimplerTeardownReport>,
    "SimplerTeardownReport crosses a dlopen boundary and a shared-memory frame, so it must be POD"
);
static_assert(
    sizeof(SimplerTeardownReport) == SIMPLER_TEARDOWN_REPORT_BYTES,
    "SimplerTeardownReport has a fixed wire size mirrored by the mailbox trailer and the Python reader"
);

class TeardownRecorder {
public:
    /**
     * Start recording one teardown on `path`.
     *
     * `NOT_ATTEMPTED` is the starting stage because it is true until an arm is
     * reached, and it stays true for a teardown that returns before one.
     */
    void begin(TeardownPath path, int32_t device_id) {
        report_ = SimplerTeardownReport{};
        report_.path = static_cast<uint8_t>(path);
        report_.reset_stage = static_cast<uint8_t>(TEARDOWN_STAGE_NOT_ATTEMPTED);
        report_.device_id = device_id;
        started_ = true;
        finished_ = false;
    }

    /** The stage the attempt now running reached. Later attempts overwrite. */
    void note_stage(TeardownResetStage stage) {
        if (!started_) return;
        report_.reset_stage = static_cast<uint8_t>(stage);
    }

    /** One actual reset-API call and what it returned. */
    void note_reset_api(TeardownResetApi api, int rc) {
        if (!started_) return;
        report_.reset_api = static_cast<uint8_t>(api);
        report_.last_reset_api_rc = rc;
        report_.flags |= static_cast<uint8_t>(TEARDOWN_FLAG_LAST_RESET_API_RC_VALID);
        if (report_.reset_api_invocations_total != UINT16_MAX) ++report_.reset_api_invocations_total;
    }

    /** One iteration of the recovery wrapper, whether or not it reaches a reset API. */
    void note_recovery_attempt() {
        if (!started_) return;
        if (report_.recovery_attempts_total != UINT16_MAX) ++report_.recovery_attempts_total;
    }

    /** What the recovery wrapper returned: the probe's verdict, not the API's. */
    void note_recovery_sequence(int rc) {
        if (!started_) return;
        report_.recovery_sequence_rc = rc;
        report_.flags |= static_cast<uint8_t>(TEARDOWN_FLAG_RECOVERY_SEQUENCE_RC_VALID);
    }

    /** The post-reset probe ran and passed. */
    void note_probe_confirmed() {
        if (!started_) return;
        report_.flags |= static_cast<uint8_t>(TEARDOWN_FLAG_PROBE_CONFIRMED);
    }

    /**
     * Close the record with the teardown's own return value.
     *
     * The first close wins: `finalize()` has several returns and the earliest
     * one reached is the one that describes this teardown.
     */
    void finish(int teardown_rc) {
        if (!started_ || finished_) return;
        report_.teardown_rc = teardown_rc;
        report_.schema = TEARDOWN_REPORT_SCHEMA;
        finished_ = true;
    }

    /** False when no teardown was recorded, which is not the same as one that did nothing. */
    bool copy_to(SimplerTeardownReport *out) const {
        if (out == nullptr || !finished_) return false;
        *out = report_;
        return true;
    }

private:
    SimplerTeardownReport report_{};
    bool started_{false};
    bool finished_{false};
};
