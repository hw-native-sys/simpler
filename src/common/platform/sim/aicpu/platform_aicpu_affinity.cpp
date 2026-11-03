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

#include "common/unified_log.h"

static std::atomic<int32_t> s_thread_counter{0};
static std::atomic<int32_t> s_cleanup_counter{0};
static std::atomic<bool> s_entry_ready{false};
static int32_t s_entry_counts[MAX_GATE_THREADS];
static int32_t s_entry_status{0};
static std::atomic<int32_t> s_entry_published{0};

static bool basic_gate_impl(
    int32_t logical_count, int32_t total_launched, AicpuEntryInitializer initialize,
    const AicpuEntryGateInitializer *native, AicpuEntryGateResult *result
) {
    if (initialize == nullptr && native == nullptr && logical_count >= total_launched) {
        return true;
    }

    if (initialize != nullptr && (logical_count <= 0 || total_launched <= 0 || logical_count > total_launched ||
                                  total_launched > MAX_GATE_THREADS))
        return false;

    if (native != nullptr && (total_launched <= 0 || total_launched > MAX_GATE_THREADS)) return false;
    int32_t idx = s_thread_counter.fetch_add(1, std::memory_order_acq_rel);
    if (native != nullptr && idx >= total_launched) return false;
    if (native != nullptr) {
        s_entry_counts[idx] = logical_count;
        s_entry_published.fetch_add(1, std::memory_order_release);
    }
    if (initialize != nullptr || native != nullptr) {
        while (s_thread_counter.load(std::memory_order_acquire) < total_launched) {}
        if (native != nullptr) {
            while (s_entry_published.load(std::memory_order_acquire) < total_launched) {}
        }
        if (idx == 0) {
            s_entry_status = 0;
            if (initialize != nullptr) initialize();
            if (native != nullptr) {
                for (int32_t i = 0; i < total_launched; ++i) {
                    if (s_entry_counts[i] <= 0 || s_entry_counts[i] > total_launched ||
                        s_entry_counts[i] != logical_count)
                        s_entry_status = -1;
                }
                const int32_t rc = native->initialize(native->context);
                if (rc != 0) s_entry_status = rc;
            }
            s_entry_ready.store(true, std::memory_order_release);
        } else {
            while (!s_entry_ready.load(std::memory_order_acquire)) {}
        }
    }
    bool survive = (idx < logical_count) && (native == nullptr || s_entry_status == 0);
    if (result != nullptr) *result = {idx, s_entry_status, survive};

    if (!survive) {
        LOG_INFO(
            "AICPU affinity gate (sim): thread idx=%d DROPPED (logical=%d, launched=%d)", idx, logical_count,
            total_launched
        );
    }

    // Last thread resets state for next invocation
    int32_t cleanup_idx = s_cleanup_counter.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
        s_entry_ready.store(false, std::memory_order_release);
        s_entry_published.store(0, std::memory_order_relaxed);
        s_thread_counter.store(0, std::memory_order_release);
        s_cleanup_counter.store(0, std::memory_order_release);
    }

    return survive;
}

// =============================================================================
// Filter-style gate (sim stub).
//
// Sim threads run on arbitrary host cores; the cpu_id pinning the onboard
// filter gate reasons about does not exist. We approximate the surviving
// semantics by mapping the first `allowed_count` arriving threads to
// exec_idx 0..allowed_count-1 (which downstream interprets as sched 0..N-2
// + orch N-1) and dropping the rest. Role classification still works; the
// actual SMT/cluster locality the onboard gate provides is not applicable.
// =============================================================================

static thread_local int32_t tl_filter_exec_idx = -1;
static std::atomic<int32_t> s_filter_counter{0};
static std::atomic<int32_t> s_filter_cleanup{0};
static std::atomic<bool> s_filter_ready{false};
static int32_t s_filter_counts[MAX_GATE_THREADS];
static int32_t s_filter_status{0};
static std::atomic<int32_t> s_filter_published{0};

static bool filter_gate_impl(
    const int32_t *allowed_cpus, int32_t allowed_count, int32_t total_launched, AicpuEntryInitializer initialize,
    const AicpuEntryGateInitializer *native, AicpuEntryGateResult *result
) {
    tl_filter_exec_idx = -1;
    if (native == nullptr && allowed_count <= 0) return false;
    if (initialize != nullptr && (allowed_cpus == nullptr || allowed_count > total_launched || total_launched <= 0 ||
                                  total_launched > MAX_GATE_THREADS))
        return false;

    if (native != nullptr && (total_launched <= 0 || total_launched > MAX_GATE_THREADS)) return false;
    int32_t idx = s_filter_counter.fetch_add(1, std::memory_order_acq_rel);
    if (native != nullptr && idx >= total_launched) return false;
    if (native != nullptr) {
        s_filter_counts[idx] = allowed_cpus != nullptr ? allowed_count : -1;
        s_filter_published.fetch_add(1, std::memory_order_release);
    }
    if (initialize != nullptr || native != nullptr) {
        while (s_filter_counter.load(std::memory_order_acquire) < total_launched) {}
        if (native != nullptr) {
            while (s_filter_published.load(std::memory_order_acquire) < total_launched) {}
        }
        if (idx == 0) {
            s_filter_status = 0;
            if (initialize != nullptr) initialize();
            if (native != nullptr) {
                for (int32_t i = 0; i < total_launched; ++i) {
                    if (s_filter_counts[i] <= 0 || s_filter_counts[i] > total_launched ||
                        s_filter_counts[i] != allowed_count)
                        s_filter_status = -1;
                }
                const int32_t rc = native->initialize(native->context);
                if (rc != 0) s_filter_status = rc;
            }
            s_filter_ready.store(true, std::memory_order_release);
        } else {
            while (!s_filter_ready.load(std::memory_order_acquire)) {}
        }
    }
    bool survive = (idx < allowed_count) && (native == nullptr || s_filter_status == 0);
    if (result != nullptr) *result = {idx, s_filter_status, survive};
    tl_filter_exec_idx = survive ? idx : -1;

    if (!survive) {
        LOG_INFO(
            "AICPU filter gate (sim): thread idx=%d DROPPED (allowed=%d, launched=%d)", idx, allowed_count,
            total_launched
        );
    }

    int32_t cleanup_idx = s_filter_cleanup.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
        s_filter_ready.store(false, std::memory_order_release);
        s_filter_published.store(0, std::memory_order_relaxed);
        s_filter_counter.store(0, std::memory_order_release);
        s_filter_cleanup.store(0, std::memory_order_release);
    }
    return survive;
}

int32_t platform_aicpu_affinity_thread_idx() { return tl_filter_exec_idx; }

void platform_aicpu_affinity_set_thread_idx(int32_t idx) { tl_filter_exec_idx = idx; }

bool platform_aicpu_affinity_gate(int32_t logical, int32_t launched, AicpuEntryInitializer initialize) {
    return basic_gate_impl(logical, launched, initialize, nullptr, nullptr);
}

bool platform_aicpu_affinity_gate_filter(
    const int32_t *cpus, int32_t logical, int32_t launched, AicpuEntryInitializer initialize
) {
    return filter_gate_impl(cpus, logical, launched, initialize, nullptr, nullptr);
}

AicpuEntryGateResult
platform_aicpu_entry_gate(int32_t logical, int32_t launched, AicpuEntryGateInitializer initialize) {
    AicpuEntryGateResult result;
    if (initialize.initialize != nullptr) basic_gate_impl(logical, launched, nullptr, &initialize, &result);
    return result;
}

AicpuEntryGateResult platform_aicpu_entry_gate_filter(
    const int32_t *cpus, int32_t logical, int32_t launched, AicpuEntryGateInitializer initialize
) {
    AicpuEntryGateResult result;
    if (initialize.initialize != nullptr) filter_gate_impl(cpus, logical, launched, nullptr, &initialize, &result);
    return result;
}
