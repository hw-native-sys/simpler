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
 * @file run_retention_probe.cpp
 * @brief #2267's late-read retention measurement.
 */

#include "run_retention_probe.h"

#include <chrono>
#include <cstring>
#include <vector>

#include "common/unified_log.h"
#include "host/raii_scope_guard.h"
#include "host/run_outcome_decision.h"
#include "host_log.h"

namespace {

using Clock = std::chrono::steady_clock;

uint64_t elapsed_ns(Clock::time_point since) {
    return static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - since).count());
}

void set_reason(RunRetentionProbeReport &report, const char *reason) {
    if (reason == nullptr) {
        report.execution_reason[0] = '\0';
        return;
    }
    std::strncpy(report.execution_reason, reason, sizeof(report.execution_reason) - 1);
    report.execution_reason[sizeof(report.execution_reason) - 1] = '\0';
}

/**
 * Whether any AICore of `prepared`'s run has reported itself started.
 *
 * The AICore publishes its report on kernel entry, gated by nothing, so a
 * report this run's epoch accepts is a device-written fact that the kernel is
 * executing — which is what the probe needs, because a launch call returning
 * says only that the driver accepted a submission.
 *
 * The epoch is what makes the answer this run's. The block is not reset per
 * run: its handshake region is published once per allocation and carries the
 * previous run's reports afterwards, so a bare `aicore_done != 0` would report
 * a predecessor as this successor. Read the same way the handshake dump reads
 * it, by copying the region back rather than dereferencing it in place.
 */
bool successor_is_running(const DeviceRunnerBase::PreparedExecution &prepared) {
    const Runtime *device_runtime = prepared.kernel_args.args.runtime_args;
    if (device_runtime == nullptr || prepared.num_aicore <= 0) return false;
    const size_t count = static_cast<size_t>(prepared.num_aicore);
    std::vector<Handshake> workers(count);
    const size_t bytes = sizeof(Handshake) * count;
    if (rtMemcpy(workers.data(), bytes, device_runtime->get_workers(), bytes, RT_MEMCPY_DEVICE_TO_HOST) != 0) {
        return false;
    }
    for (const Handshake &worker : workers) {
        if (aicore_report_accepted(&worker, prepared.identity.run_epoch)) return true;
    }
    return false;
}

}  // namespace

int run_retention_probe(
    DeviceRunnerBase &runner, DeviceRunnerBase::ActiveExecution &active,
    std::unique_ptr<DeviceRunnerBase::PreparedExecution> &prepared_successor, const RunRetentionProbeConfig &config,
    RunRetentionProbeReport *report, std::unique_ptr<DeviceRunnerBase::ActiveExecution> *active_successor_out
) {
    if (report == nullptr || active_successor_out == nullptr || active.prepared == nullptr) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const bool want_successor = config.launch_successor != 0;
    if (want_successor && prepared_successor == nullptr) return PTO_RUNTIME_ERR_INTERNAL;

    *report = RunRetentionProbeReport{};
    DeviceRunnerBase::PreparedExecution &prepared = *active.prepared;
    const uint32_t slot = prepared.pipeline_slot;
    const uint64_t epoch = prepared.identity.run_epoch;
    if (want_successor && prepared_successor->pipeline_slot == slot) {
        LOG_ERROR("run_retention_probe: both runs name pipeline slot %u; the fixture needs two", slot);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Step 1 — the predecessor's own two boundaries, and nothing else. This is
    // `wait_run_fence`'s boundary wait without any `sync_stream_pair`, which is
    // the call that would wait for the successor once one is queued.
    const int boundary_timeout_ms = config.boundary_timeout_ms != 0 ?
                                        static_cast<int>(config.boundary_timeout_ms) :
                                        RunRetentionProbePeer::stream_sync_timeout_ms(runner);
    const Clock::time_point boundary_start = Clock::now();
    report->boundary_wait_rc = RunRetentionProbePeer::fence(runner, slot).wait(prepared.identity, boundary_timeout_ms);
    report->boundary_wait_ns = elapsed_ns(boundary_start);
    if (report->boundary_wait_rc != 0) {
        LOG_ERROR("run_retention_probe: the predecessor's boundaries did not complete: %d", report->boundary_wait_rc);
        return 0;
    }

    // Step 2 — release only what the successor's launch waits for, on that
    // boundary evidence. The run keeps everything else it owns.
    report->pair_retire_rc = RunRetentionProbePeer::retire_predecessor_ownership(runner, prepared);
    if (report->pair_retire_rc != 0) {
        LOG_ERROR("run_retention_probe: retiring the predecessor's ownership failed: %d", report->pair_retire_rc);
        return 0;
    }

    // Everything below can stop early on a device failure, and every one of
    // those exits owes the same two things: a launched successor has to be
    // drained, or its diagnostics and device resources are never released, and
    // the predecessor has to be made drainable again, or on a5 it is stranded by
    // the successor having taken the runner's only poll slot.
    auto teardown = RAIIScopeGuard([&]() {
        if (*active_successor_out != nullptr) {
            // Timed for cross-arm comparison only. This is not evidence that the
            // successor still had work: a drain costs time on an already-finished
            // run too, which the retained-sync arm measures directly. What the
            // two numbers together indicate is how much of the successor the
            // synchronize had already absorbed.
            const Clock::time_point drain_start = Clock::now();
            report->successor_drain_rc = runner.drain_execution(**active_successor_out);
            report->successor_drain_ns = elapsed_ns(drain_start);
        }
        RunRetentionProbePeer::adopt_drain_ownership(runner, prepared);
    });

    DeviceRunnerBase::PreparedExecution *successor = prepared_successor.get();
    if (want_successor) {
        // Step 3 — the one admission decision the fixture replaces.
        const Clock::time_point launch_start = Clock::now();
        DeviceRunnerBase::LaunchOutcome launch = runner.launch_execution(
            std::move(prepared_successor), RunRetentionProbePeer::mint_permit(successor->identity)
        );
        report->successor_launch_rc = launch.rc;
        *active_successor_out = std::move(launch.active);
        // A launch that never reached the device hands the prepared run back
        // instead of an execution. Returning it to the caller is what keeps its
        // device resources owned by something that will release them.
        prepared_successor = std::move(launch.prepared);
        if (launch.rc != 0 || *active_successor_out == nullptr) {
            LOG_ERROR("run_retention_probe: the successor did not launch: %d", launch.rc);
            return 0;
        }

        // Step 4 — a launch call returning is not the successor executing. Wait
        // for the device to say so itself before the measurement starts, or the
        // read being fast would prove nothing about overlap.
        const uint64_t start_budget_ns =
            static_cast<uint64_t>(config.successor_start_timeout_ms != 0 ? config.successor_start_timeout_ms : 5000) *
            1000000ULL;
        while (elapsed_ns(launch_start) < start_budget_ns) {
            if (successor_is_running(*successor)) {
                report->successor_started = 1;
                break;
            }
        }
        report->successor_start_ns = elapsed_ns(launch_start);
        if (report->successor_started == 0) {
            report->successor_start_rc = PTO_RUNTIME_ERR_INTERNAL;
            LOG_ERROR("run_retention_probe: no AICore of the successor reported itself started");
            return 0;
        }
        report->successor_completion_before_read = static_cast<uint32_t>(
            RunRetentionProbePeer::fence(runner, successor->pipeline_slot).poll(successor->identity)
        );
    }

    // The control arm. This is the call `wait_run_fence` falls back to after
    // the boundaries: with a successor in flight it waits for that successor,
    // which is exactly what the record read must not do. Running it here,
    // against the same sequence, is what turns the Pending reading below into
    // evidence.
    if (config.use_retained_sync != 0) {
        rtStream_t aicpu = nullptr;
        rtStream_t aicore = nullptr;
        RunRetentionProbePeer::run_streams(runner, &aicpu, &aicore);
        const Clock::time_point sync_start = Clock::now();
        report->retained_sync_rc = RunRetentionProbePeer::sync_streams(runner, aicpu, aicore);
        report->reference_sync_ns = elapsed_ns(sync_start);
        if (report->retained_sync_rc != 0) {
            // A synchronize that failed proves nothing about waiting, so the
            // arm stops here rather than reporting a boundary reading this run
            // did not earn. Teardown still runs.
            LOG_ERROR("run_retention_probe: the control synchronize failed: %d", report->retained_sync_rc);
            return 0;
        }
    }

    // Step 5 — the measurement. The predecessor's result, read while the
    // successor is mid-execution.
    const Clock::time_point read_start = Clock::now();
    runner.read_device_run_result(slot, epoch);
    report->record_read_ns = elapsed_ns(read_start);

    if (want_successor) {
        // The one-sided observation. Pending here can only mean the read did not
        // wait for the successor. Complete has two causes — the read waited, or
        // the successor finished on its own — and nothing recorded here separates
        // them, so a caller must treat it as inconclusive rather than as a
        // verdict. See the report's own documentation.
        report->successor_completion_after_read = static_cast<uint32_t>(
            RunRetentionProbePeer::fence(runner, successor->pipeline_slot).poll(successor->identity)
        );
    }

    const Clock::time_point decide_start = Clock::now();
    RunOutcomeEvidence evidence;
    evidence.boundaries = RunCompletionFence::Completion::Complete;
    // Reported, not assumed: a failed copy-back has to reach the rule as a failed
    // read, so the rule takes its own read-failure path rather than inferring one
    // from a record it was told was valid.
    evidence.record_read =
        RunRetentionProbePeer::result_read_ok(runner, slot, epoch) ? RunRecordRead::Ok : RunRecordRead::Failed;
    evidence.terminal = runner.device_run_terminal(slot, epoch);
    const RunExecutionOutcome outcome = decide_run_execution(evidence);
    report->decision_ns = elapsed_ns(decide_start);
    report->execution_state = static_cast<int32_t>(outcome.state);
    report->execution_code = outcome.code;
    report->execution_source = static_cast<uint32_t>(outcome.source);
    set_reason(*report, outcome.reason);

    report->candidate_drain_ns = report->boundary_wait_ns + report->record_read_ns + report->decision_ns;
    return 0;
}
