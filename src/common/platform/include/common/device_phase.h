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
 * @file device_phase.h
 * @brief Fixed-cardinality AICPU run-phase timing, shared by host + AICPU + sim.
 *
 * The AICPU cannot write to the host log on its hot path (logging perturbs the
 * very timing we measure, and the AICore side has no log path at all). Instead
 * it stamps raw `get_sys_cnt_aicpu()` cycles into a host-allocated buffer whose
 * address rides on `KernelArgs::device_wall_data_base`; the host reads the
 * buffer back after the run and converts cycles → ns.
 *
 * This buffer holds a FIXED set of phases — each fires exactly once per run per
 * AICPU thread, so a plain indexed store (no ring, no rotation, no recycle)
 * suffices. That is the same model the original single run-wall pair used; this
 * just generalizes "one {start,end} pair" to "N_FIXED {start,end} pairs", with
 * AicpuPhase::RunWall keeping the original whole-run wall as slot 0.
 *
 * Variable-cardinality phases (per-submit, per-scheduler-loop, arbitrary
 * nesting) do NOT belong here — they need a rotating ring and live in the
 * L2 swimlane orch/sched pools instead.
 *
 * Layout per AICPU thread: AicpuPhaseRecord[NUM_AICPU_PHASES]. The buffer is
 * thread-major: thread t occupies records [t*NUM_AICPU_PHASES, (t+1)*N).
 */

#ifndef PLATFORM_COMMON_DEVICE_PHASE_H_
#define PLATFORM_COMMON_DEVICE_PHASE_H_

#include <cstdint>

// Fixed AICPU run phases. Each fires once per run per thread. RunWall (slot 0)
// is the whole-run wall preserved from the original device_wall buffer; the
// rest subdivide the on-NPU portion of run_prepared's blocking wait. Append new
// fixed phases before Count (this is a small, closed set by design — variable
// phases go to the L2 swimlane ring, not here).
enum class AicpuPhase : uint32_t {
    RunWall = 0,  // whole-run AICPU wall (legacy device_wall pair)
    Preamble,     // init + affinity gate before orchestration
    SoLoad,       // orchestration SO dlopen
    GraphBuild,   // orchestrate(): submit tasks (scheduler dispatches concurrently)
    PostOrch,     // AICore exec tail + drain after orchestration returns
    OrchWindow,   // orchestrator thread's submit window (former device-log orch_start/end)
    SchedWindow,  // scheduler thread's dispatch window (former device-log sched_start/end)
    // graph_build front-matter sub-phases (orchestrator thread only, before the
    // OrchWindow opens): the per-run prep the scheduler threads spin-wait on. New
    // entries appended (not reordered) so existing slot indices stay stable.
    ConfigValidate,  // config_func() + arg-count validate
    ArenaWire,       // attach prebuilt runtime arena + wire device pointers
    SmReset,         // SM/ring reset + finalize + bind, up to releasing the schedulers
    Count,
};

constexpr int NUM_AICPU_PHASES = static_cast<int>(AicpuPhase::Count);

// One phase's start/end in raw sys-counter cycles. start == kPhaseUnset marks a
// slot that was never stamped (host skips it on readback).
struct AicpuPhaseRecord {
    uint64_t start_cycle;
    uint64_t end_cycle;
};

constexpr uint64_t kPhaseUnset = UINT64_MAX;

// Total record count for a buffer sized to `max_threads` AICPU threads.
constexpr int aicpu_phase_buffer_slots(int max_threads) { return max_threads * NUM_AICPU_PHASES; }

// Reduce a thread-major phase buffer: for each phase, report the reduced
// `min(start)` in raw cycles (kPhaseUnset when no thread completed it) and span
// = max(end) - min(start). Only threads that **completed** the phase
// (`end_cycle > start_cycle`) are considered, so a thread that stamped a start
// but never an end (early exit / error / timeout, leaving end_cycle == 0) does
// not pull `min_start` down and inflate the span. The start cycles let the host
// place each phase on a common device-clock timeline — needed to compute the
// orch∪sched merged window and to position the sub-phases relative to each
// other. `out_start` and `out_span` each hold NUM_AICPU_PHASES entries.
inline void
reduce_aicpu_phase_windows(const AicpuPhaseRecord *buf, int threads, uint64_t *out_start, uint64_t *out_span) {
    for (int p = 0; p < NUM_AICPU_PHASES; ++p) {
        uint64_t min_start = kPhaseUnset;
        uint64_t max_end = 0;
        for (int t = 0; t < threads; ++t) {
            const AicpuPhaseRecord &r = buf[t * NUM_AICPU_PHASES + p];
            if (r.start_cycle != kPhaseUnset && r.end_cycle > r.start_cycle) {
                if (r.start_cycle < min_start) min_start = r.start_cycle;
                if (r.end_cycle > max_end) max_end = r.end_cycle;
            }
        }
        const bool valid = (min_start != kPhaseUnset && max_end > min_start);
        out_start[p] = valid ? min_start : kPhaseUnset;
        out_span[p] = valid ? (max_end - min_start) : 0;
    }
}

#endif  // PLATFORM_COMMON_DEVICE_PHASE_H_
