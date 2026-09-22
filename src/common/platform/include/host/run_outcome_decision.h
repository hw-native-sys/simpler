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

#include "common/device_run_result.h"
#include "host/run_completion_fence.h"

/**
 * What a run's two evidence channels, together, say about that run's execution.
 *
 * The channels are independent and neither subsumes the other. The completion
 * fence reports whether the work this run submitted has *ended*; the record its
 * device side published reports whether that work *succeeded*. A run can end
 * without publishing, and can publish a failure and keep tearing down, so the
 * two are combined by rule rather than by preferring one.
 *
 * This decides execution only. It says nothing about whether the device is
 * healthy, whether this run's resources may be retired, or whether copy-back
 * succeeded — those are separate axes with separate evidence, and folding them
 * in here would let a delivery failure be reported as a kernel error.
 *
 * Pure and device-free: everything it reads is already in hand, so the rule is
 * exercisable without hardware and both the poll and the wait path can share it
 * rather than each growing its own version.
 */

/** Whether this run's result region was read back, and with what outcome. */
enum class RunRecordRead : uint8_t {
    /** No read was attempted — a run that never launched, or one still running. */
    NotAttempted,
    /** A read was attempted and the copy failed, so the host copy holds nothing. */
    Failed,
    /** The region was copied back; `DeviceRunTerminal` says what it contains. */
    Ok,
};

enum class RunExecutionState : uint8_t {
    /** Still in progress: no channel has produced a terminal observation yet. */
    Pending,
    Succeeded,
    Failed,
    /** A terminal observation was attempted and did not decide the run. */
    Undecided,
};

/** The evidence in hand about one run, from both channels. */
struct RunOutcomeEvidence {
    RunCompletionFence::Completion boundaries{RunCompletionFence::Completion::Pending};
    RunRecordRead record_read{RunRecordRead::NotAttempted};
    /** Meaningful only when `record_read` is Ok. */
    DeviceRunTerminal terminal{};
};

/**
 * `reason` is set exactly when the state is Undecided, and names which
 * observation failed to decide. A caller reports it rather than collapsing
 * every undecided path into one silence — the paths differ in what recovery
 * they need.
 */
struct RunExecutionOutcome {
    RunExecutionState state{RunExecutionState::Undecided};
    int32_t code{0};
    DeviceRunCodeSource source{DeviceRunCodeSource::None};
    const char *reason{"no evidence"};
};

/**
 * Combine both channels into this run's execution outcome.
 *
 * Three rules carry the weight, and each exists because the obvious reading is
 * wrong:
 *
 *  - **A published failure decides the run whatever the boundaries say.** The
 *    code is attributable to this run because its device side wrote it before
 *    its kernel returned, and a failing run is precisely the one whose
 *    boundaries may never complete.
 *  - **A published success does not decide the run until both boundaries
 *    complete.** The record is written before the kernel returns, so a valid Ok
 *    read proves the orchestration finished, not that the submitted work ended.
 *    Retiring on it would free resources the device still holds.
 *  - **Submitted work no boundary covers is Undecided, never success.** An
 *    Unfenced or Error completion means the run's own events cannot speak for
 *    the work; a success record alongside it is not the missing proof.
 */
inline RunExecutionOutcome decide_run_execution(const RunOutcomeEvidence &evidence) {
    RunExecutionOutcome out;
    const bool read_ok = evidence.record_read == RunRecordRead::Ok;
    // An unread record says the same thing as an unpublished one: nothing. Both
    // collapse to Undecided here so the record channel is consulted once, and
    // `record_read` then only has to name *which* silence it was.
    const DeviceRunTerminalState published = read_ok ? evidence.terminal.state : DeviceRunTerminalState::Undecided;

    if (published == DeviceRunTerminalState::Failed) {
        out.state = RunExecutionState::Failed;
        out.code = evidence.terminal.code;
        out.source = evidence.terminal.source;
        out.reason = nullptr;
        return out;
    }

    switch (evidence.boundaries) {
    case RunCompletionFence::Completion::Error:
        out.reason = "a boundary query or wait failed";
        return out;
    case RunCompletionFence::Completion::Unfenced:
        out.reason = "submitted work no recorded boundary covers";
        return out;
    case RunCompletionFence::Completion::Pending:
    case RunCompletionFence::Completion::Complete:
        break;
    }

    if (!read_ok) {
        // Boundaries alone never decide execution: they prove the work ended,
        // and a run that ended without publishing is exactly what the record
        // channel exists to report on.
        if (evidence.record_read == RunRecordRead::Failed) {
            out.reason = "result record read failed";
            return out;
        }
        if (evidence.boundaries == RunCompletionFence::Completion::Pending) {
            out.state = RunExecutionState::Pending;
            out.reason = nullptr;
            return out;
        }
        out.reason = "no result record was read";
        return out;
    }

    if (published == DeviceRunTerminalState::Succeeded) {
        out.state = evidence.boundaries == RunCompletionFence::Completion::Complete ? RunExecutionState::Succeeded :
                                                                                      RunExecutionState::Pending;
        out.reason = nullptr;
        return out;
    }

    out.reason = evidence.terminal.reason != nullptr ? evidence.terminal.reason : "record decided nothing";
    return out;
}

/** Stable name for logs and probe reports. */
inline const char *run_execution_state_name(RunExecutionState state) {
    switch (state) {
    case RunExecutionState::Pending:
        return "pending";
    case RunExecutionState::Succeeded:
        return "succeeded";
    case RunExecutionState::Failed:
        return "failed";
    case RunExecutionState::Undecided:
        return "undecided";
    }
    return "unknown";
}

/** What a fenced drain does once this run's completion boundaries have settled. */
enum class RunDrainAction : uint8_t {
    /**
     * The record transfer reported a status of its own. That is an SDK error
     * this host thread has already observed, and it is what the run returns.
     */
    ReportTransferError,
    /**
     * This run's own evidence decides success: no whole-stream wait is taken.
     */
    AcceptRecordedSuccess,
    /**
     * Every other shape. The device's verdict comes from the stream
     * synchronize, which is the only call measured to produce one.
     */
    Synchronize,
};

/**
 * Choose the drain's action from this run's own evidence.
 *
 * Two rules carry it, and the first exists because the read is itself an SDK
 * call on the drain path:
 *
 *  - **An observed transfer error outranks everything.** A non-zero
 *    `record_transfer_rc` is a code the transport already reported to this
 *    thread. Whatever a later call answers — including zero — does not annul
 *    it, so the decision cannot fall through to a branch that would replace
 *    it. A caller that still synchronizes to converge the device must keep
 *    returning this code.
 *  - **Success needs this run's own boundaries and its own record.**
 *    `decide_run_execution` already requires both completed boundaries and a
 *    valid `Ok` read for this run's identity; nothing weaker reaches
 *    `AcceptRecordedSuccess`, and `Failed`, `Pending` and every `Undecided`
 *    shape synchronize as before.
 *
 * `AcceptRecordedSuccess` asserts what those channels observed about this run.
 * It does not assert that the device raised no exception for other work on the
 * same stream: a fault no participant recorded reaches the caller only through
 * a later API call or the device-health channel.
 */
inline RunDrainAction decide_run_drain(int record_transfer_rc, const RunOutcomeEvidence &evidence) {
    if (record_transfer_rc != 0) return RunDrainAction::ReportTransferError;
    if (decide_run_execution(evidence).state == RunExecutionState::Succeeded) {
        return RunDrainAction::AcceptRecordedSuccess;
    }
    return RunDrainAction::Synchronize;
}
