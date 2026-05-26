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
// A5 AICPU affinity gate — 1 orch + 4 sched placement
// =============================================================================
//
// AICPU topology (A5 / DAV_3510, Full SKU verified on device 0, OCCUPY=0x7ffe):
//   chip = 2 dies; each die has 2 AICPU clusters x 2 phy_cpu = 4 phy_cpu (total 8)
//   phy_cpu 0 is OS-reserved on die 0 (only 7 phy_cpu schedulable as AICPU)
//   SMT enabled on phy 1..7 -> 2 logical CPUs per phy_cpu (ht 0 + ht 1)
//   AICPU cluster mapping: cluster_id = phy_cpu_id / 2, die_id = phy_cpu_id / 4
//
// Placement intent (directa-stage1: single-die 1 orch + 4 sched, all on die 1):
//   orch_die       = 1 (die 0 carries reserved phy 0 -> avoid it entirely)
//   orch_cluster   = 2 (cpu 7..10)
//   remote_cluster = 3 (cpu 11..14)  — within-die cross-cluster, NOT cross-die
//
//   orch    = cpu_id  9  (cluster 2 / phy 5 ht 0)            — alone on phy 5
//   sched 0 = cpu_id  7  (cluster 2 / phy 4 ht 0)            — local wiring sched
//   sched 1 = cpu_id  8  (cluster 2 / phy 4 ht 1)            — local SMT sibling of sched 0
//   sched 2 = cpu_id 11  (cluster 3 / phy 6 ht 0)            — same-die remote-cluster sched
//   sched 3 = cpu_id 13  (cluster 3 / phy 7 ht 0)            — same-die remote-cluster sched
//
// Rationale: the baseline 1+4 had sched 2/3 on die 0 cluster 1 — cross-die
// from orch. Hot sync paths (SM `current_task_index`, `last_task_alive`,
// `aicore_mailbox`) were all cross-die polls/stores in the dispatch loop,
// blocking forward progress every iteration. directa-stage1 moves both
// remote scheds onto die-1 cluster 3 so every sched↔orch sync line stays
// within die 1. AICore mapping is NOT remapped: sched 2/3 will now dispatch
// cross-die to die-0 AICores (via MMIO, write-buffered → async), and
// AICore→sched mailbox reads may go cross-die depending on mailbox placement.
// The trade is sync-coherence cross-die → async-MMIO cross-die.
//
// NOTE: orch_phy and sched_pair_phy could be either {phy 4, phy 5}. We use
// phy 4 for the sched pair (cpu_id 7/8) because CANN's worker dispatch hits
// both SMT siblings reliably on phy 4; cpu_id 10 (= phy 5 ht 1) is NOT in
// CANN's observed dispatch set on this device under the current
// PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH=7 launch budget.
//
// ALLOWED_CPUS ordering convention:
//   indices 0..N-2 are scheduler slots (thread_idx assigned in this order),
//   index  N-1   is the orchestrator slot.
//
// PG variants (3-cluster SKUs): NOT auto-detected. Single-die placement only
// works on SKUs where two clusters are alive on die 1. Re-derive ALLOWED_CPUS
// by running ~/simpler/basics/00-aicpu-num/launcher on the target device and
// reading the AICPU+OCCUPY bitmap. Expected tables:
//   Full   (OCCUPY=0x7ffe):  {7, 8, 11, 13, 9}   die-1 cluster 2 + 3 (this file)
//   PG-a   (cluster 0 dead, OCCUPY=0x7ff8): same as Full
//   PG-b   (cluster 2 dead, OCCUPY=0x79fe): single-die infeasible (only one
//                                           cluster alive on die 1)
//   PG-c   (cluster 3 dead, OCCUPY=0x1ffe): single-die infeasible (only one
//                                           cluster alive on die 1)
// =============================================================================
// directa-stage1: 1 orch + 4 sched single-die (cluster 2 = orch local, cluster 3 = same-die remote).
// Last entry is the orchestrator; earlier entries are schedulers in thread_idx order.
// The gate forces each surviving thread onto its slot's cpu_id via
// sched_setaffinity, so CANN's worker→cpu dispatch order doesn't matter.
static constexpr int32_t ALLOWED_CPUS[] = {7, 8, 11, 13, 9};
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
        LOG_INFO_V0(
            "AICPU affinity gate: thread idx=%d cpu=%d exec_idx=%d %s", idx, cpu, tl_exec_idx,
            tl_exec_idx == ALLOWED_CPU_COUNT - 1 ? "ACTIVE(orch)" : "ACTIVE(sched)"
        );
    }
    return survive;
}

int32_t platform_aicpu_affinity_thread_idx() {
    return tl_exec_idx;
}
