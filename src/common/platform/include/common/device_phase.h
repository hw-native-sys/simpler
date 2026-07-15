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

#include <cstddef>
#include <cstdint>

// Fixed AICPU run phases. Each fires once per run per thread. RunWall (slot 0)
// is the whole-run wall preserved from the original device_wall buffer; the
// rest subdivide the on-NPU portion of simpler_run's blocking wait. Append new
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

// =============================================================================
// Selective task-timing slots
// =============================================================================
//
// A fixed tail appended after the AicpuPhaseRecord region in the SAME device
// buffer (same base pointer, same per-run H2D reset and post-sync D2H copy).
// Orchestration tags selected tasks with a slot id 0..NUM_TASK_TIMING_SLOTS-1;
// the scheduler folds each tagged task's AICPU dispatch/finish cycles into that
// slot. Unlike AicpuPhase (one {start,end} per run per thread), a slot receives
// min(dispatch)/max(finish) across every tagged task and block/subtask on a
// thread, so tail records are a distinct type reduced by min/max, not by the
// phase window reducer above.

constexpr int NUM_TASK_TIMING_SLOTS = 16;

// A slot untagged on a task descriptor. Valid slot ids are 0..15; anything else
// is rejected on the invalid-argument path and never stamps a record.
constexpr int32_t TASK_TIMING_SLOT_NONE = -1;

// One slot in raw sys-counter cycles. dispatch_cycle == kPhaseUnset marks a
// slot no thread dispatched this run; finish_cycle == 0 marks one no thread
// observed finished. A complete slot has dispatch != kPhaseUnset && finish >
// dispatch; incomplete slots are skipped on readback.
struct TaskTimingRecord {
    uint64_t dispatch_cycle;  // min across tagged dispatches
    uint64_t finish_cycle;    // max across tagged finishes
};

static_assert(sizeof(TaskTimingRecord) == 16, "TaskTimingRecord layout must stay 16 bytes for the fixed tail");

// Task-timing tail is thread-major, same as the phase region: thread t occupies
// records [t*NUM_TASK_TIMING_SLOTS, (t+1)*N).
constexpr int task_timing_buffer_slots(int max_threads) { return max_threads * NUM_TASK_TIMING_SLOTS; }

// Byte offset of the task-timing tail from the device buffer base: it sits
// immediately after the AicpuPhaseRecord region sized for `max_threads`.
constexpr size_t task_timing_tail_offset(int max_threads) {
    return static_cast<size_t>(aicpu_phase_buffer_slots(max_threads)) * sizeof(AicpuPhaseRecord);
}

// Total device buffer bytes: phase region + task-timing tail, both sized for
// `max_threads` AICPU threads.
constexpr size_t device_phase_buffer_bytes(int max_threads) {
    return task_timing_tail_offset(max_threads) +
           static_cast<size_t>(task_timing_buffer_slots(max_threads)) * sizeof(TaskTimingRecord);
}

// Reduce a thread-major task-timing tail: for each slot, report the reduced
// `min(dispatch)` (kPhaseUnset when no thread dispatched it) and `max(finish)`
// (0 when none finished). Only threads with a complete slot (dispatch !=
// kPhaseUnset && finish > dispatch) contribute, so a slot that was dispatched
// but never observed finished (short/failed run) does not surface a bogus span.
// `out_dispatch` and `out_finish` each hold NUM_TASK_TIMING_SLOTS entries.
inline void
reduce_task_timing_slots(const TaskTimingRecord *buf, int threads, uint64_t *out_dispatch, uint64_t *out_finish) {
    for (int s = 0; s < NUM_TASK_TIMING_SLOTS; ++s) {
        uint64_t min_dispatch = kPhaseUnset;
        uint64_t max_finish = 0;
        for (int t = 0; t < threads; ++t) {
            const TaskTimingRecord &r = buf[t * NUM_TASK_TIMING_SLOTS + s];
            if (r.dispatch_cycle != kPhaseUnset && r.finish_cycle > r.dispatch_cycle) {
                if (r.dispatch_cycle < min_dispatch) min_dispatch = r.dispatch_cycle;
                if (r.finish_cycle > max_finish) max_finish = r.finish_cycle;
            }
        }
        const bool valid = (min_dispatch != kPhaseUnset && max_finish > min_dispatch);
        out_dispatch[s] = valid ? min_dispatch : kPhaseUnset;
        out_finish[s] = valid ? max_finish : 0;
    }
}

// Resolve a task-timing tail into per-slot dispatch/finish ns on a shared
// timeline (the platform-independent half of the host readback; the D2H copy
// and the cycle→ns conversion stay platform-side). Steps: reduce across threads,
// pick a timeline origin, and convert the surviving slots.
//
// `phase_origin` anchors slots to the phase timeline so slots and phases are
// comparable; when it is kPhaseUnset — no sub-phase was stamped, e.g.
// host_build_graph runs orchestration on the host and the device stamps no
// orch/sched phases — the earliest tagged dispatch becomes the origin so tagged
// slots are still emitted on a self-consistent timeline. `cyc_to_ns` converts a
// raw cycle delta to ns using the platform sys-counter frequency. Untagged or
// incomplete slots (no dispatch, or finish <= dispatch) yield 0/0. `out_*` each
// hold NUM_TASK_TIMING_SLOTS entries.
inline void resolve_task_timing_slots_ns(
    const TaskTimingRecord *tail, int threads, uint64_t phase_origin, uint64_t (*cyc_to_ns)(uint64_t),
    uint64_t *out_dispatch_ns, uint64_t *out_finish_ns
) {
    uint64_t slot_dispatch[NUM_TASK_TIMING_SLOTS];
    uint64_t slot_finish[NUM_TASK_TIMING_SLOTS];
    reduce_task_timing_slots(tail, threads, slot_dispatch, slot_finish);

    uint64_t origin = phase_origin;
    if (origin == kPhaseUnset) {
        for (int s = 0; s < NUM_TASK_TIMING_SLOTS; ++s) {
            if (slot_dispatch[s] != kPhaseUnset && slot_dispatch[s] < origin) origin = slot_dispatch[s];
        }
    }
    for (int s = 0; s < NUM_TASK_TIMING_SLOTS; ++s) {
        out_dispatch_ns[s] = 0;
        out_finish_ns[s] = 0;
        if (slot_dispatch[s] != kPhaseUnset && slot_finish[s] > slot_dispatch[s] && origin != kPhaseUnset &&
            slot_dispatch[s] >= origin) {
            out_dispatch_ns[s] = cyc_to_ns(slot_dispatch[s] - origin);
            out_finish_ns[s] = cyc_to_ns(slot_finish[s] - origin);
        }
    }
}

#endif  // PLATFORM_COMMON_DEVICE_PHASE_H_
