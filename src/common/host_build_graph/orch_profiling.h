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

#include <stdint.h>
#include <time.h>

// Every declaration and macro below is gated on SIMPLER_ORCH_PROFILING / SIMPLER_DFX, so
// this header defines nothing at all unless the gates are already in scope. A translation
// unit that includes it before whatever else would have supplied them -- its own .cpp,
// which includes it first by convention -- would otherwise take the whole file as empty
// and fail at the first use rather than here. The gate definitions are not a reference an
// include-cleaner can see.
#include "profiling_config.h"

#include "host_build_graph/host_phase_trace.h"

// The host orchestrator's own instrumentation, on two independent channels:
//
//   ORCH_STEP_*  accumulates per-sub-step nanoseconds into the `g_orch_*_ns` cumulatives
//                that orchestrator_get_profiling() exports. SIMPLER_ORCH_PROFILING only.
//   ORCH_PHASE_* emits one span per event to the host phase trace. SIMPLER_DFX, a
//                different clock, and a different consumer.
//
// Both are macros rather than functions so a build with the channel off expands to
// nothing at all, leaving no argument evaluation behind. That is also why the counters
// they touch are declared here instead of being hidden behind accessors: a macro body
// names them directly in every translation unit that submits.
//
// The counters sit in simpler::hbg for the reason orchestrator_internal.h's functions do:
// the tensormap_and_ringbuffer orchestrator has a file-static g_orch_submit_idx and
// g_orch_submit_count of its own, so an unqualified definition here would put a name that
// generic in host_runtime.so. The using-declarations below keep every macro body and call
// site spelling them unqualified.

namespace simpler::hbg {

#if SIMPLER_ORCH_PROFILING
// Accumulated nanoseconds per sub-step, reported and reset by
// orchestrator_get_profiling().
extern uint64_t g_orch_alloc_ns;   // unified task+heap alloc
extern uint64_t g_orch_args_ns;    // param copy
extern uint64_t g_orch_lookup_ns;  // tensormap lookup + dep building
extern uint64_t g_orch_insert_ns;  // tensormap insert
extern uint64_t g_orch_fanin_ns;   // fanin list + early-return check
extern int64_t g_orch_submit_count;
#endif

#if SIMPLER_ORCH_PROFILING || SIMPLER_DFX
// Position of the current submission in the orchestration's submit order, which tags a
// phase record with where in that order it belongs. Both submit paths advance it -- the
// ordinary one per task, the Graph one per outer shell -- so a record files itself under
// the submission it sits in regardless of which path produced it.
extern uint32_t g_orch_submit_idx;
#endif

}  // namespace simpler::hbg

#if SIMPLER_ORCH_PROFILING
using simpler::hbg::g_orch_alloc_ns;
using simpler::hbg::g_orch_args_ns;
using simpler::hbg::g_orch_fanin_ns;
using simpler::hbg::g_orch_insert_ns;
using simpler::hbg::g_orch_lookup_ns;
using simpler::hbg::g_orch_submit_count;
#endif

#if SIMPLER_ORCH_PROFILING || SIMPLER_DFX
using simpler::hbg::g_orch_submit_idx;
#endif

#if SIMPLER_ORCH_PROFILING
// The orchestrator runs on the host, so its sub-steps are timed by the host's own
// monotonic clock. Inline, so the timing costs a call to clock_gettime and no symbol
// resolution.
inline uint64_t orch_now_ns() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ull + static_cast<uint64_t>(ts.tv_nsec);
}

// Accumulation is unconditional under SIMPLER_ORCH_PROFILING (that's what the flag
// is for) and feeds the per-sub-step `g_orch_*_ns` cumulatives printed in the
// cold-path log. Per-event records are a separate channel on a separate clock —
// see ORCH_PHASE_END below.
#define ORCH_STEP_START()              \
    uint64_t _t0 = orch_now_ns(), _t1; \
    (void)_t1
#define ORCH_STEP_LAP(acc)   \
    do {                     \
        _t1 = orch_now_ns(); \
        acc += (_t1 - _t0);  \
        _t0 = _t1;           \
    } while (0)
#else
// The per-sub-step accumulators exist only in an ORCH_PROFILING build, so below that
// level there is nothing to time.
#define ORCH_STEP_START()
#define ORCH_STEP_LAP(acc)
#endif

// Returns the submit order to its start for the next orchestration. An ORCH_PROFILING
// build rewinds it in orchestrator_get_profiling() instead, which the bind reads once it
// is done, so the rewind here would be a second one before that read.
inline void orch_profiling_mark_done() {
#if !SIMPLER_ORCH_PROFILING && SIMPLER_DFX
    g_orch_submit_idx = 0;
#endif
}

#if SIMPLER_DFX
// Only the host orchestrator reaches these sites, so only the Orch* half of HostPhaseKind
// appears at them; the bind kinds never do.
#define ORCH_PHASE_START() const uint64_t _orch_phase_t0 = host_phase_now_ns()
#define ORCH_PHASE_END(phase, detail)                                                                         \
    do {                                                                                                      \
        host_phase_record(                                                                                    \
            _orch_phase_t0, host_phase_now_ns(), static_cast<uint32_t>(phase), static_cast<uint64_t>(detail), \
            g_orch_submit_idx                                                                                 \
        );                                                                                                    \
    } while (0)
// For a phase that spans a submission rather than sitting inside one: the index
// advances during the span, so the group is taken at the start or the record
// files itself under the next submission.
#define ORCH_PHASE_START_SPANNING() \
    ORCH_PHASE_START();             \
    const uint32_t _orch_phase_group = g_orch_submit_idx
#define ORCH_PHASE_END_SPANNING(phase, detail)                                                                \
    do {                                                                                                      \
        host_phase_record(                                                                                    \
            _orch_phase_t0, host_phase_now_ns(), static_cast<uint32_t>(phase), static_cast<uint64_t>(detail), \
            _orch_phase_group                                                                                 \
        );                                                                                                    \
    } while (0)
#else
#define ORCH_PHASE_START()
#define ORCH_PHASE_END(phase, detail) \
    do {                              \
    } while (0)
#define ORCH_PHASE_START_SPANNING()
#define ORCH_PHASE_END_SPANNING(phase, detail) \
    do {                                       \
    } while (0)
#endif
