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
 * Unit tests for a2a3 host-side AICPU affinity selection
 * (compute_allowed_cpus from src/a2a3/platform/onboard/host/aicpu_topology_probe).
 *
 * This is the regression barrier for issue #1045: AICPU survivors must all
 * land in one NUMA cluster. Moving selection host-side (PR #1119) made it a
 * pure, deterministic function, so the property is testable without hardware.
 *
 * Covered:
 * - the standard 0xfc pool (cpu_id 2..7): active=4 (tensormap) and active=3
 *   (host_build_graph) both land entirely in the high cluster (cpu 4..7);
 * - determinism (same input → same output);
 * - the cross-cluster fallback when no single cluster fits;
 * - invalid-input rejection.
 */

#include <gtest/gtest.h>

#include <vector>

#include "aicpu_topology_probe.h"

// Minimal logger stubs so aicpu_topology_probe.cpp links without pulling in
// HostLogger. compute_allowed_cpus only emits LOG_WARN; the probe path (not
// exercised here) also uses LOG_INFO. The unified_log symbols have C
// linkage (see common/unified_log.h), so match it. No-ops — these tests
// assert on return values, not log text.
extern "C" {
void unified_log_error(const char *, const char *, ...) {}
void unified_log_warn(const char *, const char *, ...) {}
void unified_log_timing(const char *, const char *, ...) {}
void unified_log_info(const char *, const char *, ...) {}
void unified_log_debug(const char *, const char *, ...) {}
}

using pto::a2a3::AicpuLogicalCpu;
using pto::a2a3::compute_allowed_cpus;
using pto::a2a3::resolve_aicpu_cpu_id_base;

namespace {

// a2a3: cluster_id = (cpu_id % 8) / 4 — no SMT, cpu_id == physical core.
AicpuLogicalCpu cpu(int32_t cpu_id) { return AicpuLogicalCpu{cpu_id, (cpu_id % 8) / 4}; }

// The OCCUPY pool observed on a2a3 silicon: 0xfc ⇒ cpu_id 2,3,4,5,6,7
// (0 and 1 reserved for the OS). Cluster 0 = {2,3}, cluster 1 = {4,5,6,7}.
std::vector<AicpuLogicalCpu> standard_pool() { return {cpu(2), cpu(3), cpu(4), cpu(5), cpu(6), cpu(7)}; }

int32_t cluster_of(int32_t cpu_id) { return (cpu_id % 8) / 4; }

}  // namespace

// tensormap_and_ringbuffer needs 4 active threads (3 sched + 1 orch); all four
// must come from the single cluster that can hold them — cpu_id 4..7.
TEST(A2a3AffinitySelect, TensormapFourThreadsLandInHighCluster) {
    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_allowed_cpus(standard_pool(), /*active_count=*/4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 5, 6, 7}));
    for (int32_t id : allowed)
        EXPECT_EQ(cluster_of(id), 1) << "cpu " << id << " crossed a cluster";
}

// host_build_graph needs only 3 active threads (2 sched + 1 orch). They must
// still all share one cluster — the high cluster, picked deterministically.
TEST(A2a3AffinitySelect, HbgThreeThreadsLandInOneCluster) {
    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_allowed_cpus(standard_pool(), /*active_count=*/3, allowed));
    ASSERT_EQ(allowed.size(), 3u);
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 5, 6}));
    const int32_t c = cluster_of(allowed[0]);
    for (int32_t id : allowed)
        EXPECT_EQ(cluster_of(id), c) << "cpu " << id << " crossed a cluster";
}

TEST(A2a3AffinitySelect, A2UsesDieLocalCpuIds) {
    int32_t base = -1;
    ASSERT_TRUE(resolve_aicpu_cpu_id_base("Ascend910B3", /*phy_die_id=*/-1, base));
    EXPECT_EQ(base, 0);
}

TEST(A2a3AffinitySelect, A3OffsetsTheSecondDieInTheSharedAicpuOs) {
    int32_t die0_base = -1;
    int32_t die1_base = -1;
    ASSERT_TRUE(resolve_aicpu_cpu_id_base("Ascend910_9392", /*phy_die_id=*/0, die0_base));
    ASSERT_TRUE(resolve_aicpu_cpu_id_base("Ascend910_9392", /*phy_die_id=*/1, die1_base));
    EXPECT_EQ(die0_base, 0);
    EXPECT_EQ(die1_base, 8);

    auto die1_pool = standard_pool();
    for (auto &entry : die1_pool)
        entry.cpu_id += die1_base;
    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_allowed_cpus(die1_pool, /*active_count=*/4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{12, 13, 14, 15}));
}

TEST(A2a3AffinitySelect, RejectsUnknownSocAndInvalidA3Die) {
    int32_t base = -1;
    EXPECT_FALSE(resolve_aicpu_cpu_id_base(nullptr, /*phy_die_id=*/0, base));
    EXPECT_FALSE(resolve_aicpu_cpu_id_base("Ascend950PR_9599", /*phy_die_id=*/0, base));
    EXPECT_FALSE(resolve_aicpu_cpu_id_base("Ascend910_9392", /*phy_die_id=*/2, base));
}

// Selection is a pure function of its input — repeated calls are identical.
TEST(A2a3AffinitySelect, Deterministic) {
    std::vector<int32_t> a, b;
    ASSERT_TRUE(compute_allowed_cpus(standard_pool(), 4, a));
    ASSERT_TRUE(compute_allowed_cpus(standard_pool(), 4, b));
    EXPECT_EQ(a, b);
}

// When the request fits inside a cluster smaller than the largest, the chosen
// cluster is the highest-id one that can hold it.
TEST(A2a3AffinitySelect, PrefersHighestClusterThatFits) {
    // Both clusters full (2,3 | 4,5,6,7). active=2 fits in either; pick high.
    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_allowed_cpus(standard_pool(), /*active_count=*/2, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 5}));
}

// No single cluster can hold the request (3 + 3 split, active=4): the function
// keeps deterministic behavior and falls back to a cross-cluster selection by
// ascending cpu_id rather than failing.
TEST(A2a3AffinitySelect, CrossClusterFallbackWhenNoClusterFits) {
    // cluster 0 = {1,2,3}, cluster 1 = {4,5,6} — neither holds 4.
    std::vector<AicpuLogicalCpu> pool = {cpu(1), cpu(2), cpu(3), cpu(4), cpu(5), cpu(6)};
    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_allowed_cpus(pool, /*active_count=*/4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{1, 2, 3, 4}));  // first 4 by cpu_id
}

// Requesting more threads than the pool has must fail (not truncate).
TEST(A2a3AffinitySelect, RejectsRequestLargerThanPool) {
    std::vector<int32_t> allowed;
    EXPECT_FALSE(compute_allowed_cpus(standard_pool(), /*active_count=*/7, allowed));
    EXPECT_TRUE(allowed.empty());
}

// Non-positive active_count is rejected.
TEST(A2a3AffinitySelect, RejectsNonPositiveCount) {
    std::vector<int32_t> allowed;
    EXPECT_FALSE(compute_allowed_cpus(standard_pool(), 0, allowed));
    EXPECT_FALSE(compute_allowed_cpus(standard_pool(), -1, allowed));
}
