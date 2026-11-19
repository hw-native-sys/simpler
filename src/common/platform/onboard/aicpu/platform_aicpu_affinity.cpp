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
#include <cerrno>
#include <cstdio>
#include <exception>
#include "aicpu/affinity_thread_state.h"
#include <cstdint>
#ifdef __linux__
#include <sched.h>
#endif

#include "common/unified_log.h"

// MAX_GATE_THREADS (= 16) is defined in aicpu/platform_aicpu_affinity.h, the
// single source of truth shared with the Runtime ABI. 16 = headroom for a5's
// launch budget (14 logical user cpus on the 0x7ffe SKU) + a small over-launch
// margin. a2a3 only ever launches 6 threads and never approaches this bound.

// Module-owned classification slots published to each participant: the filter
// gate assigns the per-thread exec/slot index.
// -1 = dropped, otherwise this thread's index in allowed_cpus[].
// The per-thread binding record snapshots the slot value; invalidating the
// record never writes the slot. Read via platform_aicpu_affinity_thread_idx();
// used as the run-wall buffer slot and the sched/orch role id.
namespace {
simpler::AffinityTlsKey &affinity_tls_key() {
    static simpler::AffinityTlsKey value;
    return value;
}
simpler::AffinityBindingCohort binding_cohort;
}  // namespace
extern "C" __attribute__((visibility("default"))) int platform_aicpu_affinity_bind_thread(int32_t *cell) {
    const int error = affinity_tls_key().bind(cell);
    if (error == ENOSPC)
        LOG_ERROR(
            "AICPU affinity record pool exhausted: all %d records are checked out", simpler::kAffinityRecordPoolSize
        );
    else if (error != 0)
        LOG_ERROR("AICPU affinity TLS binding failed: pthread error=%d cell=%p", error, static_cast<void *>(cell));
    return error;
}

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
static int32_t s_filter_logical_counts[MAX_GATE_THREADS];
static int32_t s_filter_entry_status{0};

static bool filter_gate_impl(
    const int32_t *allowed_cpus, int32_t allowed_count, int32_t total_launched, AicpuEntryInitializer initialize,
    const AicpuEntryGateInitializer *native, AicpuEntryGateResult *result, int *error_out = nullptr
) {
    if (error_out) *error_out = 0;
    // Rejection must not expose a previous execution identity. The binding
    // record is thread-exclusive; the previously bound cell is left untouched.
    affinity_tls_key().invalidate();

    const bool valid = allowed_cpus != nullptr && allowed_count > 0 && allowed_count <= MAX_GATE_THREADS &&
                       allowed_count <= total_launched;
    const bool legacy_valid = allowed_cpus != nullptr && allowed_count > 0 && allowed_count <= MAX_GATE_THREADS &&
                              (initialize == nullptr || allowed_count <= total_launched);
    if (total_launched <= 0 || total_launched > MAX_GATE_THREADS || (native == nullptr && !legacy_valid)) return false;

    int32_t idx = s_filter_claim.fetch_add(1, std::memory_order_acq_rel);
    if (idx >= total_launched) {
        LOG_ERROR("AICPU filter gate: excess participant idx=%d total=%d", idx, total_launched);
        if (error_out) *error_out = EINVAL;
        return false;
    }
    binding_cohort.record(platform_aicpu_affinity_bind_thread(nullptr));
#ifdef __linux__
    int32_t cpu = sched_getcpu();
#else
    int32_t cpu = -1;
#endif

    if (idx < MAX_GATE_THREADS) {
        s_filter_thread_cpu[idx] = cpu;
        s_filter_logical_counts[idx] = valid ? allowed_count : -1;
    }

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
        s_filter_entry_status = 0;
        if (native != nullptr) {
            for (int32_t i = 0; i < total_launched; ++i) {
                if (s_filter_logical_counts[i] <= 0 || s_filter_logical_counts[i] != allowed_count)
                    s_filter_entry_status = -1;
            }
        }
        // Suppress classification on rejection, including reads of an invalid table.
        if (s_filter_entry_status != 0) allowed_count = 0;

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
        LOG_INFO("AICPU filter gate: allowed_count=%d total_launched=%d", allowed_count, total_launched);
        for (int32_t a = 0; a < allowed_count; ++a) {
            const char *role = (a == allowed_count - 1) ? "orch" : "sched";
            LOG_INFO("AICPU filter gate:   allowed[%d] = cpu_id %d  role=%s", a, allowed_cpus[a], role);
        }

        if (initialize != nullptr) initialize();
        if (native != nullptr) {
            const int32_t init_status = native->initialize(native->context);
            if (init_status != 0) s_filter_entry_status = init_status;
        }
        s_filter_classify_ready.store(1, std::memory_order_release);
    }

    while (s_filter_classify_ready.load(std::memory_order_acquire) == 0) {}

    binding_cohort.record(platform_aicpu_affinity_bind_thread(&s_filter_thread_exec_idx[idx]));
    bool survive = affinity_tls_key().value() >= 0 && (native == nullptr || s_filter_entry_status == 0);

    // Every participant copies the error before the last participant resets it.
    // Native initialization still runs on failure so every callback can retire.
    const int error = binding_cohort.arrive_and_read(total_launched);
    if (error_out) *error_out = error;
    survive = survive && error == 0;
    if (!survive) affinity_tls_key().invalidate();
    if (result != nullptr) *result = {idx, error != 0 ? -1 : s_filter_entry_status, survive};

    // Reset gate state after the last thread has read its result.
    if (s_filter_cleanup.fetch_add(1, std::memory_order_acq_rel) + 1 == total_launched) {
        binding_cohort.reset_after_readers();
        s_filter_claim.store(0, std::memory_order_release);
        s_filter_published.store(0, std::memory_order_release);
        s_filter_classify_init.store(0, std::memory_order_release);
        s_filter_classify_ready.store(0, std::memory_order_release);
        s_filter_cleanup.store(0, std::memory_order_release);
    }

    if (survive) {
        const char *role = (platform_aicpu_affinity_thread_idx() == allowed_count - 1) ? "orch" : "sched";
        LOG_INFO(
            "AICPU filter gate: thread idx=%d cpu=%d exec_idx=%d ACTIVE(%s)", idx, cpu,
            platform_aicpu_affinity_thread_idx(), role
        );
    } else {
        LOG_INFO("AICPU filter gate: thread idx=%d cpu=%d DROPPED", idx, cpu);
    }
    return survive;
}

int32_t platform_aicpu_affinity_thread_idx() { return affinity_tls_key().value(); }

void platform_aicpu_affinity_set_thread_idx(int32_t idx) {
    if (!affinity_tls_key().publish(idx)) {
        // Fatal diagnostics bypass the asynchronous logger so abort cannot lose them.
        std::fprintf(stderr, "AICPU affinity: set_thread_idx(%d) requires a bound cell\n", idx);
        std::terminate();
    }
}

bool platform_aicpu_affinity_gate_filter(
    const int32_t *allowed_cpus, int32_t allowed_count, int32_t total_launched, AicpuEntryInitializer initialize,
    int *error_out
) {
    return filter_gate_impl(allowed_cpus, allowed_count, total_launched, initialize, nullptr, nullptr, error_out);
}

AicpuEntryGateResult platform_aicpu_entry_gate_filter(
    const int32_t *allowed_cpus, int32_t allowed_count, int32_t total_launched, AicpuEntryGateInitializer initialize
) {
    AicpuEntryGateResult result;
    if (initialize.initialize == nullptr) return result;
    filter_gate_impl(allowed_cpus, allowed_count, total_launched, nullptr, &initialize, &result);
    return result;
}
