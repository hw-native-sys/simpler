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

// 16 = headroom for a5's launch budget (14 logical user cpus on the
// 0x7ffe SKU) + a small over-launch margin. a2a3 only ever launches 6
// threads and never approaches this bound.
static constexpr int32_t MAX_GATE_THREADS = 16;

// Per-thread exec/slot index set by the filter gate:
// -1 = dropped, otherwise this thread's index in allowed_cpus[].
// Read via platform_aicpu_affinity_thread_idx(); used as the run-wall buffer
// slot and the sched/orch role id.
static thread_local int32_t tl_exec_idx = -1;

// =============================================================================
// Filter-style gate (onboard).
// =============================================================================
//
// All `total_launched` threads enter, each reads sched_getcpu() and reports
// to s_filter_thread_cpu[]. After the barrier, the CAS-winner classifies:
// for each report, look up cpu_id in `allowed_cpus[]`; if found, the thread
// survives and gets exec_idx = index in allowed_cpus[]. Misses are dropped.

// Per-launch barrier + classification state.
// Two counters: s_filter_claim hands out a unique slot via fetch_add so each
// thread writes to a distinct s_filter_thread_cpu[idx]. s_filter_published
// is bumped (release) AFTER the cpu write — the classification barrier
// waits on the publish counter (acquire), so when it equals total_launched
// every thread's cpu write is visible. A single counter cannot do this:
// if the barrier waits on the same counter that fetch_add already moved,
// the cpu store between fetch_add and the barrier check is unordered.
static std::atomic<int32_t> s_filter_claim{0};
static std::atomic<int32_t> s_filter_published{0};
static std::atomic<int32_t> s_filter_classify_init{0};
static std::atomic<int32_t> s_filter_classify_ready{0};
static std::atomic<int32_t> s_filter_cleanup{0};
static int32_t s_filter_thread_cpu[MAX_GATE_THREADS];
static int32_t s_filter_thread_exec_idx[MAX_GATE_THREADS];

bool platform_aicpu_affinity_gate_filter(const int32_t *allowed_cpus, int32_t allowed_count, int32_t total_launched) {
    tl_exec_idx = -1;

    // Bound-check both inputs against the static slot buffers
    // (s_filter_thread_cpu[MAX_GATE_THREADS] etc.) before any indexing.
    // Without this, allowed_count or total_launched > MAX_GATE_THREADS
    // would silently truncate the classification loop and let the
    // diagnostic dump read past `allowed_cpus[]`.
    if (allowed_cpus == nullptr || allowed_count <= 0 || allowed_count > MAX_GATE_THREADS || total_launched <= 0 ||
        total_launched > MAX_GATE_THREADS) {
        LOG_ERROR(
            "AICPU filter gate: invalid config allowed_count=%d total_launched=%d (max=%d) — dropping all threads",
            allowed_count, total_launched, MAX_GATE_THREADS
        );
        return false;
    }

    int32_t idx = s_filter_claim.fetch_add(1, std::memory_order_acq_rel);
#if defined(__aarch64__) || defined(__x86_64__)
    int32_t cpu = sched_getcpu();
#else
    int32_t cpu = -1;
#endif

    if (idx < MAX_GATE_THREADS) s_filter_thread_cpu[idx] = cpu;

    // Publish: release-ordered increment ensures the s_filter_thread_cpu[idx]
    // store above is visible to any thread that observes the new published
    // value via acquire load.
    s_filter_published.fetch_add(1, std::memory_order_release);
    // Barrier: wait until every launched thread has published its cpu.
    while (s_filter_published.load(std::memory_order_acquire) < total_launched) {}

    // One thread classifies for everyone.
    int32_t expected = 0;
    if (s_filter_classify_init.compare_exchange_strong(
            expected, 1, std::memory_order_acq_rel, std::memory_order_acquire
        )) {
        for (int32_t i = 0; i < total_launched && i < MAX_GATE_THREADS; ++i)
            s_filter_thread_exec_idx[i] = -1;

        // For each reporting thread, see if its cpu is in allowed_cpus.
        // O(total_launched * allowed_count) — both ≤ ~16, fine.
        // We DO allow duplicate cpu_id landings (CANN over-subscribes the
        // sink cpu when launch_count >= popcount(OCCUPY)). The first thread
        // that lands on each allowed cpu wins; later duplicates are dropped.
        bool slot_filled[MAX_GATE_THREADS] = {false};
        int32_t filled_count = 0;
        for (int32_t tid = 0; tid < total_launched && tid < MAX_GATE_THREADS; ++tid) {
            int32_t my_cpu = s_filter_thread_cpu[tid];
            if (my_cpu < 0) continue;
            for (int32_t a = 0; a < allowed_count && a < MAX_GATE_THREADS; ++a) {
                if (allowed_cpus[a] == my_cpu && !slot_filled[a]) {
                    s_filter_thread_exec_idx[tid] = a;
                    slot_filled[a] = true;
                    ++filled_count;
                    break;
                }
            }
        }

        // Forward-progress guard: launch_count=popcount(OCCUPY) is expected
        // to give one representative for each host-selected cpu_id, but some
        // runtime dispatch patterns can duplicate a cpu and miss another one.
        // If that happens, keep the exact-match assignments and fill the
        // missing exec slots by reported thread order so sched/orch roles are
        // still all present instead of timing out the AICore side.
        if (filled_count < allowed_count) {
            LOG_WARN(
                "AICPU filter gate: only matched %d/%d allowed cpus; filling missing exec slots by report order",
                filled_count, allowed_count
            );
            int32_t next_slot = 0;
            for (int32_t tid = 0; tid < total_launched && tid < MAX_GATE_THREADS && filled_count < allowed_count;
                 ++tid) {
                if (s_filter_thread_exec_idx[tid] >= 0) continue;
                while (next_slot < allowed_count && slot_filled[next_slot])
                    ++next_slot;
                if (next_slot >= allowed_count) break;
                s_filter_thread_exec_idx[tid] = next_slot;
                slot_filled[next_slot] = true;
                ++filled_count;
            }
        }

        // Diagnostic: dump the allowed table once.
        // (Lower-volume than a per-thread line; cheaper at INFO.)
        LOG_INFO_V0("AICPU filter gate: allowed_count=%d total_launched=%d", allowed_count, total_launched);
        for (int32_t a = 0; a < allowed_count; ++a) {
            const char *role = (a == allowed_count - 1) ? "orch" : "sched";
            LOG_INFO_V0("AICPU filter gate:   allowed[%d] = cpu_id %d  role=%s", a, allowed_cpus[a], role);
        }

        s_filter_classify_ready.store(1, std::memory_order_release);
    }

    while (s_filter_classify_ready.load(std::memory_order_acquire) == 0) {}

    bool survive;
    if (idx < total_launched && idx < MAX_GATE_THREADS) {
        tl_exec_idx = s_filter_thread_exec_idx[idx];
        survive = (tl_exec_idx >= 0);
    } else {
        tl_exec_idx = -1;
        survive = false;
    }

    // Reset gate state after the last thread has read its result.
    if (s_filter_cleanup.fetch_add(1, std::memory_order_acq_rel) + 1 == total_launched) {
        s_filter_claim.store(0, std::memory_order_release);
        s_filter_published.store(0, std::memory_order_release);
        s_filter_classify_init.store(0, std::memory_order_release);
        s_filter_classify_ready.store(0, std::memory_order_release);
        s_filter_cleanup.store(0, std::memory_order_release);
    }

    if (survive) {
        const char *role = (tl_exec_idx == allowed_count - 1) ? "orch" : "sched";
        LOG_INFO_V0("AICPU filter gate: thread idx=%d cpu=%d exec_idx=%d ACTIVE(%s)", idx, cpu, tl_exec_idx, role);
    } else {
        LOG_INFO_V0("AICPU filter gate: thread idx=%d cpu=%d DROPPED", idx, cpu);
    }
    return survive;
}

int32_t platform_aicpu_affinity_thread_idx() { return tl_exec_idx; }
