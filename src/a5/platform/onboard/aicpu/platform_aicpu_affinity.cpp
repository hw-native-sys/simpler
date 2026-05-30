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
#include "aicpu/platform_aicpu_affinity.h"

#include <atomic>
#include <cstdint>
#ifdef __linux__
#include <sched.h>
#endif

#include "common/unified_log.h"

// =============================================================================
// A5 AICPU affinity gate — 1 orch + 1 wiring + 4 sched placement
// =============================================================================
//
// AICPU topology (A5 / DAV_3510, Full SKU verified on device 0, OCCUPY=0x7ffe):
//   chip = 2 dies; each die has 2 AICPU clusters x 2 phy_cpu = 4 phy_cpu (total 8)
//   phy_cpu 0 is OS-reserved on die 0 (only 7 phy_cpu schedulable as AICPU)
//   SMT enabled on phy 1..7 -> 2 logical CPUs per phy_cpu (ht 0 + ht 1)
//   AICPU cluster mapping: cluster_id = phy_cpu_id / 2, die_id = phy_cpu_id / 4
//
// Placement intent (wiring-thread split, single-die, all on die 1):
//
//   cluster 2 (orch + wiring; ring/graph state stays cluster-local):
//     cpu  9 (phy 5 ht 0) → orch          — alone on phy 5
//     cpu  7 (phy 4 ht 0) → wiring        — alone on phy 4
//     (cpu  8 unused; cpu 10 not in CANN dispatch set under launch budget 7)
//
//   cluster 3 (4 schedulers, 2 SMT pairs):
//     cpu 11 (phy 6 ht 0) → sched 0       — SMT pair w/ sched 1
//     cpu 12 (phy 6 ht 1) → sched 1
//     cpu 13 (phy 7 ht 0) → sched 2       — SMT pair w/ sched 3
//     cpu 14 (phy 7 ht 1) → sched 3
//
// Rationale:
//   * Graph state (wiring_queue, current_task_index, last_task_alive, dep_pool)
//     is touched only by orch + wiring → keep both in cluster 2 so those cache
//     lines stay cluster-local; orch's stores reach wiring without cluster snoop.
//   * Per-task scheduling state (tracker bitmaps, has_idle peer reads, async
//     wait list lock) is touched only by the 4 sched threads → cluster 3
//     internal snoop, no cross-cluster atomic chain.
//   * The two SMT pairs in cluster 3 cost ~40% per thread under contention,
//     so 4 sched ≈ 3.2× single-thread throughput rather than 4×. Trade-off
//     accepted: gives 4 polling threads for AIC COND-register fan-out while
//     keeping all sched coherence traffic inside one cluster.
//
// Single-producer chains created by this layout:
//   * orch       → wiring_queue → wiring         (cluster-2 internal)
//   * wiring     → ready_queues                  (single producer → MPSC for sched)
//   * wiring     → last_task_alive (SM)          (single writer; no advance_lock CAS)
//
// ALLOWED_CPUS ordering convention:
//   indices 0..N-3 are scheduler slots (thread_idx assigned in this order),
//   index  N-2   is the wiring slot,
//   index  N-1   is the orchestrator slot.
//
// PG variants (3-cluster SKUs): NOT auto-detected. This layout assumes two
// live clusters on die 1. Re-derive ALLOWED_CPUS by running
// ~/simpler/basics/00-aicpu-num/launcher on the target device and reading
// the AICPU+OCCUPY bitmap. cpu 12 / cpu 14 must be in CANN's dispatch set
// under PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH=7 — if not, this layout
// silently loses sched threads and falls back to the index-based survivor
// rule.
// =============================================================================
// 1 orch + 1 wiring + 4 sched, all on die 1.
// Last entry is the orchestrator; second-last is the wiring thread; earlier
// entries are schedulers in thread_idx order. The gate identifies each
// surviving thread by its CANN-dispatched cpu_id (we do NOT call
// sched_setaffinity), so cpu 12 / cpu 14 must already be in CANN's launch
// set for this layout to populate all 4 sched slots.
static constexpr int32_t ALLOWED_CPUS[] = {7, 8, 3, 4, 5, 6};
static constexpr int32_t ALLOWED_CPU_COUNT = sizeof(ALLOWED_CPUS) / sizeof(ALLOWED_CPUS[0]);

// Slot-claim counter: each gate call fetch_adds to get its slot index.
// Reset to 0 when all total_launched threads have completed the gate.
static std::atomic<int32_t> s_reported{0};

static thread_local int32_t tl_exec_idx = -1;

// Per-thread state for the gate (filter-style — survive iff sched_getcpu() ∈ ALLOWED_CPUS).
static constexpr int32_t MAX_GATE_THREADS = 16;
static std::atomic<int32_t> s_cpu_written{0};
static std::atomic<int32_t> s_gate_init{0};
static std::atomic<int32_t> s_gate_ready{0};
static int32_t s_thread_cpu[MAX_GATE_THREADS];
static bool s_thread_survive[MAX_GATE_THREADS];
static int32_t s_thread_exec_idx[MAX_GATE_THREADS];

bool platform_aicpu_affinity_gate(int32_t /*logical_count*/, int32_t total_launched) {
    if (ALLOWED_CPU_COUNT >= total_launched) {
        tl_exec_idx = -1;
        return true;
    }

    int32_t idx = s_reported.fetch_add(1, std::memory_order_acq_rel);
#if defined(__aarch64__) || defined(__x86_64__)
    int32_t cpu = sched_getcpu();
#else
    int32_t cpu = -1;
#endif
    LOG_INFO_V0("AICPU affinity gate: thread idx=%d sched_getcpu=%d", idx, cpu);

    if (idx < MAX_GATE_THREADS) s_thread_cpu[idx] = cpu;
    s_cpu_written.fetch_add(1, std::memory_order_release);
    while (s_cpu_written.load(std::memory_order_acquire) < total_launched) {}

    int32_t expected = 0;
    if (s_gate_init.compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        for (int32_t i = 0; i < total_launched; ++i) {
            s_thread_survive[i] = false;
            s_thread_exec_idx[i] = -1;
        }
        int32_t allowed_cnt = 0;
        for (int32_t tid = 0; tid < total_launched; ++tid) {
            int32_t c = s_thread_cpu[tid];
            if (c < 0) continue;
            for (int32_t a = 0; a < ALLOWED_CPU_COUNT; ++a) {
                if (c == ALLOWED_CPUS[a]) {
                    s_thread_survive[tid] = true;
                    s_thread_exec_idx[tid] = a;
                    allowed_cnt++;
                    break;
                }
            }
        }
        LOG_INFO_V0(
            "AICPU affinity gate: allowed_cnt=%d total_launched=%d orch_cpu=%d", allowed_cnt, total_launched,
            ALLOWED_CPUS[ALLOWED_CPU_COUNT - 1]
        );
        s_gate_ready.store(1, std::memory_order_release);
    }

    while (s_gate_ready.load(std::memory_order_acquire) == 0) {}

    bool survive = (idx < total_launched) ? s_thread_survive[idx] : false;
    tl_exec_idx = (idx < total_launched) ? s_thread_exec_idx[idx] : -1;

    static std::atomic<int32_t> s_cleanup{0};
    if (s_cleanup.fetch_add(1, std::memory_order_acq_rel) + 1 == total_launched) {
        s_reported.store(0, std::memory_order_release);
        s_cpu_written.store(0, std::memory_order_release);
        s_gate_init.store(0, std::memory_order_release);
        s_gate_ready.store(0, std::memory_order_release);
        s_cleanup.store(0, std::memory_order_release);
    }

    if (!survive) {
        LOG_INFO_V0("AICPU affinity gate: thread idx=%d cpu=%d DROPPED", idx, cpu);
    } else {
        const char *role;
        if (tl_exec_idx == ALLOWED_CPU_COUNT - 1) {
            role = "ACTIVE(orch)";
        } else if (tl_exec_idx == ALLOWED_CPU_COUNT - 2) {
            role = "ACTIVE(wiring)";
        } else {
            role = "ACTIVE(sched)";
        }
        LOG_INFO_V0(
            "AICPU affinity gate: thread idx=%d cpu=%d exec_idx=%d %s", idx, cpu, tl_exec_idx, role
        );
    }
    return survive;
}

int32_t platform_aicpu_affinity_thread_idx() {
    return tl_exec_idx;
}
