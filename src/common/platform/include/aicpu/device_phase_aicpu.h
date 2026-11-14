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
 * @file device_phase_aicpu.h
 * @brief AICPU-side stamping into the fixed device-phase buffer.
 *
 * The buffer (header + AicpuPhaseRecord[NUM_AICPU_PHASES] per AICPU thread +
 * optional task-timing tail) is host-allocated; its base address is published
 * into the AICPU SO via
 * `set_platform_phase_base()` (onboard: kernel.cpp from KernelArgs; sim: the
 * host dlsym's the setter), exactly like the dump / chip_swimlane / pmu bases.
 * The per-thread slot is resolved from `platform_aicpu_affinity_thread_idx()`
 * (POSIX pthread-key TLS) — no C++ `thread_local`, per docs/dynamic-linking.md,
 * so the base survives the dlopen boundary on sim.
 *
 * Each surviving thread stamps its OWN slot with raw get_sys_cnt_aicpu() cycles
 * — plain stores, no atomics, no rotation (each fixed phase fires once per run).
 * The host reduces across threads on readback (see read_device_wall_ns).
 *
 * All helpers are no-ops when the base is unset (capture disabled) or the
 * thread slot is out of range, so call sites need no extra guards.
 */

#pragma once

#include <cstdint>

#include "aicpu/device_time.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "common/device_phase.h"
#include "common/platform_config.h"

// Published by the host (onboard: kernel.cpp from KernelArgs::device_wall_data_base;
// sim: dlsym'd setter), mirroring set_platform_dump_base / set_platform_chip_swimlane_base.
// Defined in src/common/platform/shared/aicpu/device_phase_aicpu.cpp.
extern "C" void set_platform_phase_base(uint64_t phase_data_base);
extern "C" uint64_t get_platform_phase_base();

/**
 * Resolve this thread's phase-record array within the buffer, or nullptr if
 * capture is disabled / the slot is out of range.
 *
 * @param buffer_base  the phase buffer base (get_platform_phase_base()).
 * @param thread_idx   platform_aicpu_affinity_thread_idx() for this thread.
 */
inline AicpuPhaseRecord *aicpu_phase_records(uint64_t buffer_base, int thread_idx) {
    if (buffer_base == 0 || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH) {
        return nullptr;
    }
    AicpuPhaseRecord *records = device_phase_records(reinterpret_cast<void *>(buffer_base));
    return records + thread_idx * NUM_AICPU_PHASES;
}

/** This thread's phase records, resolved from the published base + affinity idx. */
inline AicpuPhaseRecord *aicpu_phase_self_records() {
    return aicpu_phase_records(get_platform_phase_base(), platform_aicpu_affinity_thread_idx());
}

/** Stamp the start cycle of `phase` for this thread. No-op if capture is off. */
inline void aicpu_phase_start(AicpuPhase phase) {
    AicpuPhaseRecord *records = aicpu_phase_self_records();
    if (records == nullptr) return;
    records[static_cast<int>(phase)].start_cycle = get_sys_cnt_aicpu();
}

/** Stamp the end cycle of `phase` for this thread. No-op if capture is off. */
inline void aicpu_phase_end(AicpuPhase phase) {
    AicpuPhaseRecord *records = aicpu_phase_self_records();
    if (records == nullptr) return;
    records[static_cast<int>(phase)].end_cycle = get_sys_cnt_aicpu();
}

/**
 * Store an already-captured {start, end} cycle window into `phase` for this
 * thread. Used for the orch/sched windows, whose timestamps are measured by the
 * orchestrator / scheduler themselves and just need to ride home to the host
 * buffer rather than be re-stamped here. No-op if capture is off.
 */
inline void aicpu_phase_set_window(AicpuPhase phase, uint64_t start_cycle, uint64_t end_cycle) {
    AicpuPhaseRecord *records = aicpu_phase_self_records();
    if (records == nullptr) return;
    records[static_cast<int>(phase)].start_cycle = start_cycle;
    records[static_cast<int>(phase)].end_cycle = end_cycle;
}

// One terminal writer, after all AICores retired. Reuses the ordinary device
// timing readback, independently of chip-swimlane collection.
inline void device_phase_set_aicore_work_window(const AicoreWorkWindow &window) {
    const uint64_t base = get_platform_phase_base();
    if (base == 0) return;
    device_phase_buffer_header(reinterpret_cast<void *>(base))->aicore_work = window;
}

// =============================================================================
// Selective task-timing slots (fixed tail after the phase region)
// =============================================================================
//
// The tail lives in the same published buffer at task_timing_tail_offset(), so
// it reuses get_platform_phase_base() and the same per-thread affinity slot. A
// tagged task's scheduler folds its dispatch/finish cycles into slot `id`;
// untagged tasks (TASK_TIMING_SLOT_NONE) never reach these helpers — the caller
// gates on the sentinel so the hot path stays a single cache-hot compare.

/**
 * Resolve this thread's task-timing slot array within the buffer, or nullptr if
 * capture is disabled / the slot array is out of range.
 */
inline TaskTimingRecord *aicpu_task_timing_records(uint64_t buffer_base, int thread_idx) {
    if (buffer_base == 0 || thread_idx < 0 || thread_idx >= PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH) {
        return nullptr;
    }
    TaskTimingRecord *tail =
        task_timing_tail_records(reinterpret_cast<void *>(buffer_base), PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH);
    return tail + thread_idx * NUM_TASK_TIMING_SLOTS;
}

/**
 * Publish whether any task-timing record was dispatched this run. Call only
 * from the last surviving AICPU thread, after an acquire/release completion
 * gate has made every scheduler thread's record writes visible. The scan stays
 * out of the dispatch hot path and runs once per capture-enabled run.
 */
inline void aicpu_publish_task_timing_tail_usage(int active_threads) {
    uint64_t buffer_base = get_platform_phase_base();
    if (buffer_base == 0 || active_threads <= 0 || active_threads > PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH) {
        return;
    }

    void *buffer = reinterpret_cast<void *>(buffer_base);
    const TaskTimingRecord *tail = task_timing_tail_records(buffer, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH);
    device_phase_buffer_header(buffer)->task_timing_tail_used = task_timing_tail_used(tail, active_threads) ? 1 : 0;
}

/**
 * Fold a dispatch cycle into `slot` on `thread_idx`'s record as a min. No-op if
 * capture is off, `slot` is out of range (0..NUM_TASK_TIMING_SLOTS-1), or
 * `thread_idx` is out of range — an out-of-range id never writes out of bounds.
 *
 * The caller passes the Scheduler's own thread index (not
 * platform_aicpu_affinity_thread_idx()): host_build_graph hands its scheduler
 * threads a local index because the sim affinity gate leaves the affinity idx
 * unset (-1), so resolving via affinity there would drop every write. The host
 * reduces across all thread records with min/max, so any distinct valid index
 * per thread yields the same result.
 */
inline void aicpu_task_timing_dispatch(int slot, int thread_idx) {
    if (slot < 0 || slot >= NUM_TASK_TIMING_SLOTS) return;
    TaskTimingRecord *records = aicpu_task_timing_records(get_platform_phase_base(), thread_idx);
    if (records == nullptr) return;
    uint64_t cycle = get_sys_cnt_aicpu();
    if (cycle < records[slot].dispatch_cycle) records[slot].dispatch_cycle = cycle;
}

/** Fold a finish cycle into `slot` on `thread_idx`'s record as a max. No-op on
 * out-of-range slot/thread_idx or when capture is off. See the dispatch variant
 * for why the caller supplies the Scheduler's own thread index. */
inline void aicpu_task_timing_finish(int slot, int thread_idx) {
    if (slot < 0 || slot >= NUM_TASK_TIMING_SLOTS) return;
    TaskTimingRecord *records = aicpu_task_timing_records(get_platform_phase_base(), thread_idx);
    if (records == nullptr) return;
    uint64_t cycle = get_sys_cnt_aicpu();
    if (cycle > records[slot].finish_cycle) records[slot].finish_cycle = cycle;
}

/**
 * RAII guard: stamp `phase`'s start on construction, end on destruction. The
 * device analogue of the host `StraceScope` — but it only writes the per-thread
 * cycle slot (no logging: the AICPU hot path must not log, per the codestyle
 * rule; the host re-emits the readback as the marker). Bracket a phase region
 * with its own `{}` scope so the start/end are visually obvious and every exit
 * path (including an early `return`) records the end:
 *
 *     {
 *         AicpuPhaseScope _g(AicpuPhase::ArenaWire);
 *         ...                              // end stamped at the closing }
 *     }
 *
 * Each AicpuPhase has its own fixed slot, so nested scopes (e.g. RunWall ⊃
 * GraphBuild ⊃ SmReset, each its own guard) record independently and never
 * interfere; a phase must still fire at most once per run (a second guard for
 * the same phase would overwrite the slot).
 */
class AicpuPhaseScope {
public:
    explicit AicpuPhaseScope(AicpuPhase phase) :
        phase_(phase) {
        aicpu_phase_start(phase_);
    }
    ~AicpuPhaseScope() { aicpu_phase_end(phase_); }
    AicpuPhaseScope(const AicpuPhaseScope &) = delete;
    AicpuPhaseScope &operator=(const AicpuPhaseScope &) = delete;

private:
    AicpuPhase phase_;
};
