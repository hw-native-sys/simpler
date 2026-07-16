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
 * @file dfx_backpressure_device.h
 * @brief Unified DFX backpressure (block-on-contention + global-sync freeze)
 *        coordination — shared layout + AICPU-side primitives.
 *
 * `DfxBackpressureHeader` is the per-subsystem coordination block. Every DFX
 * subsystem's DataHeader embeds ONE `DfxBackpressureHeader backpressure;` member
 * instead of re-declaring the three fields. It is physically per-subsystem (each
 * subsystem has its own shared-memory region) — that is the per-subsystem peer
 * group: swimlane's freeze is independent of pmu's. The struct *definition* is
 * shared here; the *instances* are necessarily separate.
 *
 * The two device-side reactions (park at the gate / raise the leader signal)
 * live here too. The host side of the mechanism (the freeze open/release state
 * machine) lives once in `ProfilerAlgorithms::update_backpressure_freeze`
 * (host/profiler_base.h). Every DFX subsystem carries the header block and runs
 * the machinery; block-on-contention is the only behavior — it is idle at zero
 * cost until a lane raises `contended`.
 */

#ifndef SRC_COMMON_PLATFORM_INCLUDE_COMMON_DFX_BACKPRESSURE_DEVICE_H_
#define SRC_COMMON_PLATFORM_INCLUDE_COMMON_DFX_BACKPRESSURE_DEVICE_H_

#include <cstdint>

#include "aicpu/device_time.h"
#include "common/memory_barrier.h"

// SPIN_WAIT_HINT() is the repo-wide AICPU spin-wait relax used by every
// resource-wait spin (ring/dep-pool full, lock_fanout ticket locks, scheduler
// dispatch). It is platform-tiered: onboard silicon expands to a no-op (AICPU
// owns its A55 core), sim adds sched_yield() so the oversubscribed host cores
// don't starve the AICore threads running real kernels.
//
// This header must NEVER (re)define SPIN_WAIT_HINT once a translation unit
// already has it. It is pulled — via each DFX DataHeader — into always-on
// scheduler / orchestrator / aicore TUs that already got the platform's
// sched_yield() definition (from inner_kernel.h / pto_runtime2_types.h). An
// unguarded `#define SPIN_WAIT_HINT() ((void)0)` here would clobber that on
// include order, turning those TUs' yielding spins into busy spins and starving
// the AICore threads (sim scheduler no-progress timeout). So: reuse whatever the
// platform already provides; only supply the fallback — for pure host,
// struct-only builds where spin_hint.h is off the path and the spin templates
// are never instantiated — when nobody else has defined it.
#ifndef SPIN_WAIT_HINT
#if __has_include("spin_hint.h")
#include "spin_hint.h"
#else
#define SPIN_WAIT_HINT() ((void)0)
#endif
#endif

// Per-subsystem backpressure coordination block, embedded once in every DFX
// DataHeader. Two independent global freezes, one per gate — together they
// deliver #997's "block until RQ fully drained AND FQ fully refilled":
//   rq_* : ready-queue-full (push gate). Host opens rq_freeze on rq_contended;
//          every lane parks at its push gate; host releases once RQ is drained.
//   fq_* : free-queue-empty (pop gate). Host opens fq_freeze on fq_contended;
//          every lane parks at its pop gate; host releases once FQ is refilled to
//          its initial upper limit min(kSlotCount, BUFFERS).
// freeze_active are host→device; contended are device→host (leader signal, host
// consumes + clears). Block-on-contention is the only behavior — no opt-out gate.
struct DfxBackpressureHeader {
    volatile uint32_t rq_freeze_active;  // host → device (push gate)
    volatile uint32_t rq_contended;      // device → host (ready-queue-full leader)
    volatile uint32_t fq_freeze_active;  // host → device (pop gate)
    volatile uint32_t fq_contended;      // device → host (free-queue-empty leader)
};

namespace dfx_backpressure {

// Global-sync peer freeze at the push gate: while the host has opened rq_freeze,
// park here so every lane stops at its push gate (common-mode, lane-aligned gap)
// instead of one lane sparsifying and misleading bottleneck reading. Relaxes via
// SPIN_WAIT_HINT() (platform-tiered). Null-safe. Timeout-protected to prevent infinite spin on
// host crash or hardware failure. Returns false on timeout, true when gate opens normally.
template <typename Header>
inline bool push_freeze_barrier(const Header *header, uint64_t timeout_cycles) {
    if (header == nullptr) {
        return true;
    }
    const uint64_t start = get_sys_cnt_aicpu();
    while (header->backpressure.rq_freeze_active != 0) {
        if (get_sys_cnt_aicpu() - start >= timeout_cycles) {
            return false;  // timeout: gate failed to open
        }
        SPIN_WAIT_HINT();
    }
    rmb();  // acquire: order host's pre-release queue writes before the reads that follow the gate
    return true;
}

// Global-sync peer freeze at the pop gate: while the host has opened fq_freeze,
// park here so every lane stops at its pop gate. Symmetric to
// push_freeze_barrier for the free-queue side. Null-safe. Timeout-protected to prevent
// infinite spin on host crash or hardware failure. Returns false on timeout, true when
// gate opens normally.
template <typename Header>
inline bool pop_freeze_barrier(const Header *header, uint64_t timeout_cycles) {
    if (header == nullptr) {
        return true;
    }
    const uint64_t start = get_sys_cnt_aicpu();
    while (header->backpressure.fq_freeze_active != 0) {
        if (get_sys_cnt_aicpu() - start >= timeout_cycles) {
            return false;  // timeout: gate failed to open
        }
        SPIN_WAIT_HINT();
    }
    rmb();  // acquire: order host's pre-release queue writes before the reads that follow the gate
    return true;
}

// Leader signals (once per wait, via the `signalled` guard) when a lane hits
// real contention, so the host opens the matching global freeze that parks the
// peer lanes too. Idempotent sticky flags; the host consumes + clears them.
template <typename Header>
inline void mark_rq_contended(Header *header, bool *signalled) {
    if (!*signalled && header != nullptr) {
        header->backpressure.rq_contended = 1;
        *signalled = true;
    }
}
template <typename Header>
inline void mark_fq_contended(Header *header, bool *signalled) {
    if (!*signalled && header != nullptr) {
        header->backpressure.fq_contended = 1;
        *signalled = true;
    }
}

// Leader park for a lane whose own reclaim depends on the host completing an
// open→drain→release cycle on the FREE-queue (pop) side — only tensor_dump's
// arena barrier today (engine-gate leaders instead spin on a real free slot).
// Blocks until the host has both opened fq_freeze covering this contention AND
// released it. Uses the DISJUNCTION (fq_contended || fq_freeze_active): the host
// opens fq_freeze before consuming fq_contended, so the predicate stays
// continuously true from mark_fq_contended() to release — no (0,0) escape
// window. Relaxes via SPIN_WAIT_HINT() (platform-tiered). Null-safe.
template <typename Header>
inline void wait_for_release(const Header *header) {
    if (header == nullptr) {
        return;
    }
    while (header->backpressure.fq_contended != 0 || header->backpressure.fq_freeze_active != 0) {
        SPIN_WAIT_HINT();
    }
    rmb();  // acquire: order host's pre-release writes before the leader's post-release reads
}

}  // namespace dfx_backpressure

#endif  // SRC_COMMON_PLATFORM_INCLUDE_COMMON_DFX_BACKPRESSURE_DEVICE_H_
