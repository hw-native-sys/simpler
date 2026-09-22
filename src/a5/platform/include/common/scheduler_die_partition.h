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

#pragma once

#include <cstdint>

/**
 * Die-affinity ownership of AICore clusters by AICPU scheduler threads.
 *
 * A scheduler thread polls each of its cores' COND register once per completion
 * sweep, and that MMIO read costs measurably more when the core sits on the
 * other die: 139 ns cross-die against 90 ns same-die, measured on an
 * Ascend950DT_9581 (three runs, under 5% spread; the same-die figure matches the
 * ~95 ns recorded in .claude/rules/ascend.md). Ownership that keeps a thread on
 * its own die therefore shortens its sweep, which is what bounds how quickly it
 * observes completions and re-dispatches.
 *
 * Two inputs meet here, and neither is available to the other side alone:
 *
 *   - which die a *scheduler thread* sits on. Derived host-side from
 *     ALLOWED_CPUS and CPU_TOPO, then passed down packed (see the bit helpers
 *     below), because the device sees only cpu_ids and never phy_cpu_id.
 *   - which die a *cluster* sits on. Derived device-side from the
 *     physical_core_id each AICore reports at handshake, because CANN assigns
 *     block indices to physical cores per launch and the host cannot predict
 *     the result.
 *
 * Balance is a hard constraint and affinity a soft one: every thread ends up
 * with exactly its quota. A greedy "same die until the die runs out" pass
 * without the quota would hand one thread every cluster on an under-subscribed
 * die and leave its peers sharing the rest.
 */
namespace pto::a5 {

// Mirrors MAX_GATE_THREADS; kept local so this header stays dependency-light
// and usable from a host unit test without the platform config.
constexpr int32_t SCHED_DIE_MAX_THREADS = 16;

// =============================================================================
// Packed host->device encoding
// =============================================================================
//
// bit t        die of scheduler thread t (0 or 1)
// bit 32 + t   that die is known
//
// A thread whose die is unknown makes the whole vector unusable, so the host
// publishes 0 (nothing known) rather than a partially filled word.

constexpr uint64_t sched_die_bits_with(uint64_t bits, int32_t thread_idx, int32_t die) {
    if (thread_idx < 0 || thread_idx >= SCHED_DIE_MAX_THREADS || (die != 0 && die != 1)) return bits;
    const uint64_t slot = static_cast<uint64_t>(1) << thread_idx;
    const uint64_t known = static_cast<uint64_t>(1) << (32 + thread_idx);
    return (bits & ~slot) | (die == 1 ? slot : 0) | known;
}

// Returns 0 / 1, or -1 when this thread's die was never published.
constexpr int32_t sched_die_bits_get(uint64_t bits, int32_t thread_idx) {
    if (thread_idx < 0 || thread_idx >= SCHED_DIE_MAX_THREADS) return -1;
    if ((bits & (static_cast<uint64_t>(1) << (32 + thread_idx))) == 0) return -1;
    return (bits & (static_cast<uint64_t>(1) << thread_idx)) != 0 ? 1 : 0;
}

// =============================================================================
// Ownership
// =============================================================================

/**
 * Assign every cluster to exactly one scheduler thread, preferring same-die.
 *
 * @param cluster_count  number of clusters (1 AIC + 2 AIV each)
 * @param thread_count   number of scheduler threads that own cores
 * @param cluster_die    cluster_die[ci] is 0 or 1, from physical_core_id
 * @param sched_die_bits packed per-thread die, as published by the host
 * @param out_owner      filled with the owning thread index per cluster
 *
 * @return false when the inputs cannot support a die-aware partition — the
 *         caller keeps its existing round-robin ownership. This is the single
 *         fallback switch: an unknown thread die, an out-of-range die, or a
 *         degenerate shape all land here rather than producing a partition that
 *         is subtly wrong.
 *
 * Deterministic: threads are filled in index order and clusters scanned in
 * index order, so every thread that runs this independently reaches the same
 * answer without exchanging anything. The barrier-free scheduler init relies on
 * that.
 */
inline bool compute_cluster_owners(
    int32_t cluster_count, int32_t thread_count, const int8_t *cluster_die, uint64_t sched_die_bits, int32_t *out_owner
) {
    if (cluster_count <= 0 || thread_count <= 0 || thread_count > SCHED_DIE_MAX_THREADS || cluster_die == nullptr ||
        out_owner == nullptr) {
        return false;
    }

    int32_t thread_die[SCHED_DIE_MAX_THREADS];
    for (int32_t t = 0; t < thread_count; t++) {
        thread_die[t] = sched_die_bits_get(sched_die_bits, t);
        if (thread_die[t] < 0) return false;
    }
    for (int32_t ci = 0; ci < cluster_count; ci++) {
        if (cluster_die[ci] != 0 && cluster_die[ci] != 1) return false;
        out_owner[ci] = -1;
    }

    // Exactly cluster_count/thread_count each, with the first
    // cluster_count%thread_count threads taking one more.
    int32_t quota[SCHED_DIE_MAX_THREADS];
    int32_t taken[SCHED_DIE_MAX_THREADS];
    for (int32_t t = 0; t < thread_count; t++) {
        quota[t] = cluster_count / thread_count + (t < cluster_count % thread_count ? 1 : 0);
        taken[t] = 0;
    }

    // Pass 1 — same die, still bounded by the quota. A thread that cannot fill
    // up here is one whose die is over-subscribed; pass 2 gives it the rest.
    for (int32_t t = 0; t < thread_count; t++) {
        for (int32_t ci = 0; ci < cluster_count && taken[t] < quota[t]; ci++) {
            if (out_owner[ci] < 0 && cluster_die[ci] == thread_die[t]) {
                out_owner[ci] = t;
                taken[t]++;
            }
        }
    }

    // Pass 2 — cross die, to the quota.
    for (int32_t t = 0; t < thread_count; t++) {
        for (int32_t ci = 0; ci < cluster_count && taken[t] < quota[t]; ci++) {
            if (out_owner[ci] < 0) {
                out_owner[ci] = t;
                taken[t]++;
            }
        }
    }

    for (int32_t ci = 0; ci < cluster_count; ci++) {
        if (out_owner[ci] < 0) return false;
    }
    return true;
}

}  // namespace pto::a5
