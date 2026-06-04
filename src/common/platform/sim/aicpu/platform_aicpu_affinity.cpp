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

bool platform_aicpu_affinity_gate(int32_t logical_count, int32_t total_launched) {
    if (logical_count >= total_launched) {
        return true;
    }

    int32_t idx = s_thread_counter.fetch_add(1, std::memory_order_acq_rel);
    bool survive = (idx < logical_count);

    if (!survive) {
        LOG_INFO_V0(
            "AICPU affinity gate (sim): thread idx=%d DROPPED (logical=%d, launched=%d)", idx, logical_count,
            total_launched
        );
    }

    // Last thread resets state for next invocation
    int32_t cleanup_idx = s_cleanup_counter.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
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

bool platform_aicpu_affinity_gate_filter(
    const int32_t * /*allowed_cpus*/, int32_t allowed_count, int32_t total_launched
) {
    tl_filter_exec_idx = -1;
    if (allowed_count <= 0) return false;

    int32_t idx = s_filter_counter.fetch_add(1, std::memory_order_acq_rel);
    bool survive = (idx < allowed_count);
    tl_filter_exec_idx = survive ? idx : -1;

    if (!survive) {
        LOG_INFO_V0(
            "AICPU filter gate (sim): thread idx=%d DROPPED (allowed=%d, launched=%d)", idx, allowed_count,
            total_launched
        );
    }

    int32_t cleanup_idx = s_filter_cleanup.fetch_add(1, std::memory_order_acq_rel);
    if (cleanup_idx + 1 == total_launched) {
        s_filter_counter.store(0, std::memory_order_release);
        s_filter_cleanup.store(0, std::memory_order_release);
    }
    return survive;
}

int32_t platform_aicpu_affinity_thread_idx() { return tl_filter_exec_idx; }
