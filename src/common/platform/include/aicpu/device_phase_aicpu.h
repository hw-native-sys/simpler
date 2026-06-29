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
 * The buffer (AicpuPhaseRecord[NUM_AICPU_PHASES] per AICPU thread, thread-major)
 * is host-allocated; its base address is published into the AICPU SO via
 * `set_platform_phase_base()` (onboard: kernel.cpp from KernelArgs; sim: the
 * host dlsym's the setter), exactly like the dump / l2_swimlane / pmu bases.
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

#ifndef PLATFORM_AICPU_DEVICE_PHASE_AICPU_H_
#define PLATFORM_AICPU_DEVICE_PHASE_AICPU_H_

#include <cstdint>

#include "aicpu/device_time.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "common/device_phase.h"
#include "common/platform_config.h"

// Published by the host (onboard: kernel.cpp from KernelArgs::device_wall_data_base;
// sim: dlsym'd setter), mirroring set_platform_dump_base / set_platform_l2_swimlane_base.
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
    return reinterpret_cast<AicpuPhaseRecord *>(buffer_base) + thread_idx * NUM_AICPU_PHASES;
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

#endif  // PLATFORM_AICPU_DEVICE_PHASE_AICPU_H_
