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

#include "common/chip_swimlane_profiling.h"
#include "host/dfx_run_config.h"
#include "worker/native_run_phase.h"

/**
 * The two decisions a joined diagnostic launch is refused by, as values.
 *
 * Both are read from a `DeviceRunnerBase` and an `OnboardNativeRunContext` that
 * only a live device session owns, and the answer they produce is a bare bool
 * at the C boundary. Stating each refusal as a named verdict is what lets the
 * composition be decided in one place, and lets a caller's log say which term
 * declined rather than only that something did.
 *
 * Nothing here reads device or collector state. The resource terms -- the
 * collector's latched shape and its retained capacity -- stay with the runner
 * that owns them, so the configuration decision below never takes the
 * collector's lock for a pair its levels have already disqualified.
 */
namespace simpler::dfx {

/** Why a prepared successor may not be ordered behind a running predecessor. */
enum class JoinedDiagnosticVerdict : uint8_t {
    kConfigurationAgrees,
    kKernelExecution,
    kLevelDisagrees,
    kLevelNotAdmitted,
    kChannelDisagrees,
    kSamePipelineSlot,
};

/** The verdict as a reason a caller can report. */
inline const char *joined_diagnostic_verdict_name(JoinedDiagnosticVerdict verdict) {
    switch (verdict) {
    case JoinedDiagnosticVerdict::kConfigurationAgrees:
        return "the pair's configuration agrees";
    case JoinedDiagnosticVerdict::kKernelExecution:
        return "kernel-mode execution orders its own launches";
    case JoinedDiagnosticVerdict::kLevelDisagrees:
        return "the pair asked for different diagnostic settings";
    case JoinedDiagnosticVerdict::kLevelNotAdmitted:
        return "this diagnostic level admits no joined successor";
    case JoinedDiagnosticVerdict::kChannelDisagrees:
        return "the pair enabled different diagnostic channels";
    case JoinedDiagnosticVerdict::kSamePipelineSlot:
        return "both runs hold the same pipeline slot";
    }
    return "an unnamed configuration verdict";
}

/** One run's configuration, as the pair decision reads it. */
struct JoinedDiagnosticRun {
    ChipSwimlaneLevel chip_swimlane_level{ChipSwimlaneLevel::DISABLED};
    // The four queue-pair owners' settings, compared rather than refused. Each
    // publishes its own per-run accounting into a bank this run's identity
    // reserved, so two open runs are told apart by what their producers
    // stamped. What is still shared is the *setting*: level and event mode are
    // one word in each device header, latched when a producer initializes, so
    // a pair that disagrees would silently give one run the other's setting.
    DumpArgsLevel dump_args_level{DumpArgsLevel::OFF};
    bool pmu_enabled{false};
    PmuEventType pmu_event_type{};
    bool dep_gen_enabled{false};
    bool scope_stats_enabled{false};
    uint32_t pipeline_slot{0};

    /** The five settings this run asked for, plus the slot it holds. */
    static JoinedDiagnosticRun from(const DfxRunConfig &dfx, uint32_t pipeline_slot) {
        JoinedDiagnosticRun out;
        out.chip_swimlane_level = dfx.chip_swimlane_level;
        out.dump_args_level = dfx.dump_args_level;
        out.pmu_enabled = dfx.pmu_enabled;
        out.pmu_event_type = dfx.pmu_event_type;
        out.dep_gen_enabled = dfx.dep_gen_enabled;
        out.scope_stats_enabled = dfx.scope_stats_enabled;
        out.pipeline_slot = pipeline_slot;
        return out;
    }
};

/**
 * The configuration half of the pair decision, for a successor that would be
 * ordered behind `pred`.
 *
 * `kernel_execution` is the runner's execution-mode latch: the kernel path
 * prepares and finalizes through the shared run lifecycle but launches through
 * its own, and its producers' ordering is not established.
 *
 * The levels must be *equal*, not merely each supported, and the same holds for
 * every other owner's setting. A level is one shared word in the device header,
 * written by whichever run arms, and the device latches it once when its
 * producer initializes; a queued predecessor may not have reached that latch
 * when its successor arms, so a 1-and-2 pair would silently give one run the
 * other's level, with no counter, no verdict and no log to say so.
 *
 * Equality is also what makes an owner that is *off* on both sides free: it has
 * no device state for either run to latch, so it constrains nothing.
 */
inline JoinedDiagnosticVerdict classify_joined_diagnostic_pair(
    bool kernel_execution, const JoinedDiagnosticRun &succ, const JoinedDiagnosticRun &pred
) {
    if (kernel_execution) return JoinedDiagnosticVerdict::kKernelExecution;
    if (succ.chip_swimlane_level != pred.chip_swimlane_level) return JoinedDiagnosticVerdict::kLevelDisagrees;
    // A swimlane level either run admits but the pair has not been shown safe
    // for. Checked after equality so the message names the narrower fact.
    if (succ.chip_swimlane_level != ChipSwimlaneLevel::DISABLED &&
        succ.chip_swimlane_level != ChipSwimlaneLevel::TASK_TIMING &&
        succ.chip_swimlane_level != ChipSwimlaneLevel::SCHEDULE_TIMING) {
        return JoinedDiagnosticVerdict::kLevelNotAdmitted;
    }
    if (succ.dump_args_level != pred.dump_args_level || succ.pmu_enabled != pred.pmu_enabled ||
        succ.dep_gen_enabled != pred.dep_gen_enabled || succ.scope_stats_enabled != pred.scope_stats_enabled) {
        return JoinedDiagnosticVerdict::kChannelDisagrees;
    }
    // Only when PMU is on for both: the event mode selects the counter group
    // the device programs, and a run measuring one group under columns naming
    // another cannot be told apart by any later count.
    if (succ.pmu_enabled && succ.pmu_event_type != pred.pmu_event_type) {
        return JoinedDiagnosticVerdict::kChannelDisagrees;
    }
    // Two runs in one pipeline slot are one run's state reused, not a pair:
    // the successor would arm over the predecessor's own slot-indexed device
    // buffers while it is still producing into them.
    if (succ.pipeline_slot == pred.pipeline_slot) return JoinedDiagnosticVerdict::kSamePipelineSlot;
    return JoinedDiagnosticVerdict::kConfigurationAgrees;
}

/**
 * What one native run context says about a handle the pair gate was given.
 *
 * `has_execution` is the phase's own owner: a running context's configuration
 * lives on its active execution, because its `prepared_execution` was moved out
 * when the launch took it and reading that would read a hollow object; a
 * prepared context has no active execution yet and owns its own.
 *
 * The three `*_matches` terms are read off that execution against the context
 * that produced it. They are not redundant with the handle lookup: a handle
 * resolves to a context, and a context can be holding an execution left by an
 * earlier run on the same slot, so the identity is what says this execution is
 * the one this caller named.
 */
struct NativeRunHandleFacts {
    NativeRunPhase phase{NativeRunPhase::Prepared};
    bool runner_claimed{false};
    bool runner_reserved{false};
    bool has_execution{false};
    bool runtime_matches{false};
    bool identity_matches{false};
    bool slot_matches{false};
};

/** Whether `facts` describe an execution this caller may read at `required`. */
inline bool prepared_handle_is_owned(const NativeRunHandleFacts &facts, NativeRunPhase required) {
    if (facts.phase != required) return false;
    if (required == NativeRunPhase::Running) {
        if (!facts.runner_claimed) return false;
    } else if (!facts.runner_reserved) {
        return false;
    }
    if (!facts.has_execution) return false;
    return facts.runtime_matches && facts.identity_matches && facts.slot_matches;
}

}  // namespace simpler::dfx
