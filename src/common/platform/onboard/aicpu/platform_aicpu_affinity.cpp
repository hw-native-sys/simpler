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

static constexpr int32_t AICPU_CORES_PER_CHIP = 8;
static constexpr int32_t MAX_CLUSTERS = 2;
static constexpr int32_t CPUS_PER_CLUSTER = 4;
// 16 = headroom for a5's launch budget (14 logical user cpus on the
// 0x7ffe SKU) + a small over-launch margin. a2a3 only ever launches 6
// threads and never approaches this bound.
static constexpr int32_t MAX_GATE_THREADS = 16;

static std::atomic<uint64_t> s_cpumask{0};
static std::atomic<int32_t> s_reported{0};
static std::atomic<int32_t> s_gate_init{0};
static std::atomic<int32_t> s_gate_ready{0};

static int32_t s_thread_cpu[MAX_GATE_THREADS];
static bool s_thread_survive[MAX_GATE_THREADS];

static inline int32_t popcount64(uint64_t v) { return __builtin_popcountll(static_cast<unsigned long long>(v)); }

bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched) {
    if (logical_count >= total_launched) {
        return true;
    }

    // Assign thread index
    int32_t idx = s_reported.fetch_add(1, std::memory_order_acq_rel);

    // Report CPU
#if defined(__aarch64__)
    int32_t cpu = sched_getcpu();
#elif defined(__x86_64__)
    int32_t cpu = sched_getcpu();
#else
    int32_t cpu = -1;
#endif

    int32_t normalized_cpu = -1;
    if (cpu >= 0) {
        if (cpu < 63) {
            s_cpumask.fetch_or(1ULL << cpu, std::memory_order_release);
        }
        normalized_cpu = cpu % AICPU_CORES_PER_CHIP;
    }
    if (idx < MAX_GATE_THREADS) {
        s_thread_cpu[idx] = normalized_cpu;
    }

    // Barrier: wait until all total_launched threads have reported
    while (popcount64(s_cpumask.load(std::memory_order_acquire)) < total_launched &&
           s_reported.load(std::memory_order_acquire) < total_launched) {}

    // CAS winner does cluster classification
    int32_t expected = 0;
    if (s_gate_init.compare_exchange_strong(expected, 1, std::memory_order_acq_rel, std::memory_order_acquire)) {
        // Initialize survive flags
        for (int32_t i = 0; i < total_launched; ++i) {
            s_thread_survive[i] = false;
        }

        struct ClusterInfo {
            int32_t count{0};
            int32_t tids[MAX_GATE_THREADS];
        };
        ClusterInfo clusters[MAX_CLUSTERS];

        for (int32_t tid = 0; tid < total_launched; ++tid) {
            int32_t c = s_thread_cpu[tid];
            if (c < 0) continue;
            int32_t cluster_id = c / CPUS_PER_CLUSTER;
            if (cluster_id < 0 || cluster_id >= MAX_CLUSTERS) continue;
            ClusterInfo &info = clusters[cluster_id];
            if (info.count < MAX_GATE_THREADS) info.tids[info.count++] = tid;
        }

        int32_t major_id = (clusters[0].count >= clusters[1].count) ? 0 : 1;
        int32_t minor_id = 1 - major_id;
        int32_t major_cnt = clusters[major_id].count;
        int32_t minor_cnt = clusters[minor_id].count;

        LOG_INFO_V0(
            "AICPU affinity gate: major=%d(cnt=%d) minor=%d(cnt=%d) logical=%d", major_id, major_cnt, minor_id,
            minor_cnt, logical_count
        );

        if (major_cnt == logical_count && minor_cnt == (total_launched - logical_count)) {
            // Expected topology: major cluster threads survive
            for (int32_t i = 0; i < clusters[major_id].count; ++i) {
                s_thread_survive[clusters[major_id].tids[i]] = true;
            }
        } else {
            // Unexpected topology: fall back to first logical_count threads
            LOG_WARN(
                "AICPU affinity gate: unexpected topology (major=%d minor=%d), "
                "falling back to index-based cutoff",
                major_cnt, minor_cnt
            );
            for (int32_t i = 0; i < logical_count && i < total_launched; ++i) {
                s_thread_survive[i] = true;
            }
        }

        s_gate_ready.store(1, std::memory_order_release);
    }

    // Wait for classification to complete
    while (s_gate_ready.load(std::memory_order_acquire) == 0) {}

    bool survive = (idx < total_launched) ? s_thread_survive[idx] : false;

    // Last thread resets state for next invocation
    int32_t finished = s_reported.load(std::memory_order_acquire);
    (void)finished;
    // Reset is deferred: the statics persist but are re-initialized by the CAS winner
    // on next call. We reset the atomics after all threads have read their result.
    // Use a second atomic counter for cleanup.
    static std::atomic<int32_t> s_cleanup{0};
    int32_t cleanup_idx = s_cleanup.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
        s_cpumask.store(0, std::memory_order_release);
        s_reported.store(0, std::memory_order_release);
        s_gate_init.store(0, std::memory_order_release);
        s_gate_ready.store(0, std::memory_order_release);
        s_cleanup.store(0, std::memory_order_release);
    }

    if (!survive) {
        LOG_INFO_V0("AICPU affinity gate: thread idx=%d cpu=%d DROPPED", idx, normalized_cpu);
    } else {
        LOG_INFO_V0("AICPU affinity gate: thread idx=%d cpu=%d ACTIVE", idx, normalized_cpu);
    }

    return survive;
}

// =============================================================================
// Filter-style gate (a5 onboard).
// =============================================================================
//
// All `total_launched` threads enter, each reads sched_getcpu() and reports
// to s_filter_thread_cpu[]. After the barrier, the CAS-winner classifies:
// for each report, look up cpu_id in `allowed_cpus[]`; if found, the thread
// survives and gets exec_idx = index in allowed_cpus[]. Misses are dropped.
//
// State is shared with the legacy gate where harmless (s_reported,
// s_gate_init, s_gate_ready, s_cleanup) — only one variant runs in any
// given build (a2a3 uses the legacy gate; a5 onboard uses this one).

// Per-thread output of the filter gate. -1 = dropped (this thread was not
// in allowed_cpus when the CAS-winner classified). Otherwise = position in
// allowed_cpus[0..allowed_count-1], used downstream as sched/orch role id.
static thread_local int32_t tl_filter_exec_idx = -1;

// Per-launch barrier + classification state, parallel to the legacy gate.
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
    tl_filter_exec_idx = -1;

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
        for (int32_t tid = 0; tid < total_launched && tid < MAX_GATE_THREADS; ++tid) {
            int32_t my_cpu = s_filter_thread_cpu[tid];
            if (my_cpu < 0) continue;
            for (int32_t a = 0; a < allowed_count && a < MAX_GATE_THREADS; ++a) {
                if (allowed_cpus[a] == my_cpu && !slot_filled[a]) {
                    s_filter_thread_exec_idx[tid] = a;
                    slot_filled[a] = true;
                    break;
                }
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
        tl_filter_exec_idx = s_filter_thread_exec_idx[idx];
        survive = (tl_filter_exec_idx >= 0);
    } else {
        tl_filter_exec_idx = -1;
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
        const char *role = (tl_filter_exec_idx == allowed_count - 1) ? "orch" : "sched";
        LOG_INFO_V0(
            "AICPU filter gate: thread idx=%d cpu=%d exec_idx=%d ACTIVE(%s)", idx, cpu, tl_filter_exec_idx, role
        );
    } else {
        LOG_INFO_V0("AICPU filter gate: thread idx=%d cpu=%d DROPPED", idx, cpu);
    }
    return survive;
}

int32_t platform_aicpu_affinity_thread_idx() { return tl_filter_exec_idx; }
