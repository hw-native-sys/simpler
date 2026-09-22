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

#include <gtest/gtest.h>

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include "aicpu_topology_probe.h"
#include "common/platform_config.h"

extern "C" {
void unified_log_error(const char *, const char *, ...) {}
void unified_log_warn(const char *, const char *, ...) {}
void unified_log_info(const char *, const char *, ...) {}
void unified_log_debug(const char *, const char *, ...) {}
}

namespace {

using pto::a5::AicpuLaunchPlan;
using pto::a5::AicpuLogicalCpu;
using pto::a5::AicpuScenarioType;
using pto::a5::AicpuSelectionPolicy;
using pto::a5::AicpuTopology;
using pto::a5::AicpuTopologySource;
using pto::a5::build_aicpu_launch_plan;
using pto::a5::classify_aicpu_scenario;
using pto::a5::compute_allowed_cpus;
using pto::a5::compute_scenario_allowed_cpus;
using pto::a5::compute_unknown_allowed_cpus;
using pto::a5::derive_topology_from_occupy;
using pto::a5::enumerate_cpus_from_occupy;
using pto::a5::format_aicpu_topology_json;
using pto::a5::load_cpu_topo_from_json;

std::vector<AicpuLogicalCpu> make_physical_range(int32_t first_phy, int32_t last_phy) {
    std::vector<AicpuLogicalCpu> cpus;
    for (int32_t phy = first_phy; phy <= last_phy; ++phy) {
        cpus.push_back({2 * phy, phy, 0, phy / 2, phy / 4});
        cpus.push_back({2 * phy + 1, phy, 1, phy / 2, phy / 4});
    }
    return cpus;
}

std::vector<AicpuLogicalCpu> primaries(const std::vector<AicpuLogicalCpu> &cpus) {
    std::vector<AicpuLogicalCpu> result;
    for (const auto &cpu : cpus)
        if (cpu.hyperthread_id == 0) result.push_back(cpu);
    return result;
}

void set_device_occupy(AicpuTopology &topology, uint64_t occupy) {
    topology.device_occupancy.occupy = occupy;
    topology.device_occupancy.occupy_valid = true;
}

TEST(A5AicpuTopologyFallback, EnumeratesBoundaryCpuIdsAndRejectsEmptyMask) {
    std::vector<AicpuLogicalCpu> cpus;

    ASSERT_TRUE(enumerate_cpus_from_occupy((1ULL << 0) | (1ULL << 17) | (1ULL << 63), cpus));
    ASSERT_EQ(cpus.size(), 3U);

    const std::vector<int32_t> expected_cpu_ids = {0, 17, 63};
    for (size_t i = 0; i < expected_cpu_ids.size(); ++i) {
        EXPECT_EQ(cpus[i].cpu_id, expected_cpu_ids[i]);
        EXPECT_EQ(cpus[i].phy_cpu_id, -1);
        EXPECT_EQ(cpus[i].hyperthread_id, -1);
        EXPECT_EQ(cpus[i].cluster_id, -1);
        EXPECT_EQ(cpus[i].die_id, -1);
    }

    cpus = {{1, 1, 0, 0, 0}};
    EXPECT_FALSE(enumerate_cpus_from_occupy(0, cpus));
    EXPECT_TRUE(cpus.empty());
}

#if defined(__x86_64__)
TEST(A5AicpuTopologyFallback, PreservesLegacyVerified9579Fallback) {
    std::vector<AicpuLogicalCpu> cpus;
    ASSERT_TRUE(derive_topology_from_occupy("Ascend950PR_9579", 0x3e, cpus));
    ASSERT_EQ(cpus.size(), 5U);
    EXPECT_EQ(cpus.front().cpu_id, 1);
    EXPECT_EQ(cpus.back().cpu_id, 5);
}
#else
TEST(A5AicpuTopologyFallback, LegacyVerified9579FallbackIsX86Only) {
    std::vector<AicpuLogicalCpu> cpus;
    EXPECT_FALSE(derive_topology_from_occupy("Ascend950PR_9579", 0x3e, cpus));
}
#endif

TEST(A5AicpuTopologyFallback, LegacyFallbackRejectsUnverifiedSignature) {
    std::vector<AicpuLogicalCpu> cpus = {{1, 1, 0, 0, 0}};
    EXPECT_FALSE(derive_topology_from_occupy("Ascend950PR_9599", 0x3e, cpus));
    EXPECT_TRUE(cpus.empty());
}

TEST(A5AicpuTopologyJsonFallback, Loads9599LayoutFromPackagedJson) {
    std::vector<AicpuLogicalCpu> cpus;
    ASSERT_TRUE(load_cpu_topo_from_json("Ascend950PR_9599", 0x1f8, cpus));
    ASSERT_EQ(cpus.size(), 9U);

    EXPECT_EQ(cpus[0].cpu_id, 0);
    EXPECT_EQ(cpus[0].phy_cpu_id, 0);
    EXPECT_EQ(cpus[0].hyperthread_id, 0);

    EXPECT_EQ(cpus[1].cpu_id, 1);
    EXPECT_EQ(cpus[1].phy_cpu_id, 1);
    EXPECT_EQ(cpus[1].hyperthread_id, 0);
    EXPECT_EQ(cpus[2].cpu_id, 2);
    EXPECT_EQ(cpus[2].phy_cpu_id, 1);
    EXPECT_EQ(cpus[2].hyperthread_id, 1);

    for (int32_t i = 3; i <= 8; ++i) {
        EXPECT_EQ(cpus[i].cpu_id, i);
        EXPECT_EQ(cpus[i].phy_cpu_id, i - 1);
        EXPECT_EQ(cpus[i].hyperthread_id, 0);
        EXPECT_EQ(cpus[i].cluster_id, cpus[i].phy_cpu_id / 2);
        EXPECT_EQ(cpus[i].die_id, cpus[i].phy_cpu_id / 4);
    }

    // phy 1 is the only SMT pair; remaining phys are unique.
    int phy1_count = 0;
    std::vector<int32_t> other_phys;
    for (const auto &cpu : cpus) {
        if (cpu.phy_cpu_id == 1) {
            ++phy1_count;
        } else {
            other_phys.push_back(cpu.phy_cpu_id);
        }
    }
    EXPECT_EQ(phy1_count, 2);
    std::sort(other_phys.begin(), other_phys.end());
    other_phys.erase(std::unique(other_phys.begin(), other_phys.end()), other_phys.end());
    EXPECT_EQ(other_phys.size(), 7U);
}

#if defined(__x86_64__)
TEST(A5AicpuTopologyJsonFallback, PreservesVerifiedFallbackSignatureAndSelection) {
    std::vector<AicpuLogicalCpu> cpus;
    bool generic_selection_only = false;
    ASSERT_TRUE(load_cpu_topo_from_json("Ascend950PR_9579", 0x3e, cpus, &generic_selection_only));
    ASSERT_TRUE(generic_selection_only);
    ASSERT_EQ(cpus.size(), 5U);
    for (int32_t i = 0; i < 5; ++i) {
        EXPECT_EQ(cpus[i].cpu_id, i + 1);
        EXPECT_EQ(cpus[i].phy_cpu_id, i + 1);
        EXPECT_EQ(cpus[i].hyperthread_id, 0);
        EXPECT_EQ(cpus[i].cluster_id, (i + 1) / 2);
        EXPECT_EQ(cpus[i].die_id, (i + 1) / 4);
    }

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_allowed_cpus(cpus, 2, 1, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 5, 1}));
}
#else
TEST(A5AicpuTopologyJsonFallback, RejectsVerifiedFallbackOnNonX86Host) {
    std::vector<AicpuLogicalCpu> cpus;
    EXPECT_FALSE(load_cpu_topo_from_json("Ascend950PR_9579", 0x3e, cpus));
    EXPECT_TRUE(cpus.empty());
}
#endif

TEST(A5AicpuTopologyJsonFallback, RejectsUnverifiedSignatures) {
    auto expect_rejected = [](const char *soc_name, uint64_t occupy) {
        SCOPED_TRACE(
            ::testing::Message() << "soc=" << (soc_name == nullptr ? "(null)" : soc_name) << " occupy=" << occupy
        );
        std::vector<AicpuLogicalCpu> cpus = {{1, 1, 0, 0, 0}};
        EXPECT_FALSE(load_cpu_topo_from_json(soc_name, occupy, cpus));
        EXPECT_TRUE(cpus.empty());
    };

    expect_rejected("Ascend950PR_9599", 0x1f0);
    expect_rejected("Ascend950PR_9579", 0x1f8);
    expect_rejected("Ascend950PR_unknown", 0);
    expect_rejected("Ascend950PR_missing", 0);
    expect_rejected(nullptr, 0);
}

TEST(A5AicpuTopologySelection, ClassifiesKnownScenarios) {
    const auto fg_all = make_physical_range(0, 7);
    // FG classification is independent of Scheduler SMT availability.
    EXPECT_EQ(classify_aicpu_scenario(16, fg_all, primaries(fg_all)), AicpuScenarioType::kFg);
    EXPECT_EQ(classify_aicpu_scenario(16, fg_all, fg_all), AicpuScenarioType::kFg);
    const auto pg1_all = make_physical_range(0, 5);
    EXPECT_EQ(classify_aicpu_scenario(12, pg1_all, pg1_all), AicpuScenarioType::kPg1);

    auto pg2_all = make_physical_range(0, 1);
    auto right_die = make_physical_range(4, 5);
    pg2_all.insert(pg2_all.end(), right_die.begin(), right_die.end());
    EXPECT_EQ(classify_aicpu_scenario(8, pg2_all, pg2_all), AicpuScenarioType::kPg2);
    // logical_cpu_count must still match all_logical_cpus.size(); a mismatched
    // count is Unknown. The non-16/12/8 case is covered by
    // Classifies9599StyleNineLogicalAsFg.
    EXPECT_EQ(classify_aicpu_scenario(9, pg2_all, pg2_all), AicpuScenarioType::kUnknown);
}

TEST(A5AicpuTopologySelection, Classifies9599StyleNineLogicalAsFg) {
    std::vector<AicpuLogicalCpu> all;
    ASSERT_TRUE(load_cpu_topo_from_json("Ascend950PR_9599", 0x1f8, all));
    std::vector<AicpuLogicalCpu> sched;
    for (const auto &cpu : all) {
        if (cpu.cpu_id >= 3 && cpu.cpu_id <= 8) sched.push_back(cpu);
    }
    EXPECT_EQ(classify_aicpu_scenario(static_cast<uint32_t>(all.size()), all, sched), AicpuScenarioType::kFg);
}

TEST(A5AicpuTopologySelection, ComputesFgPolicy) {
    const auto all = make_physical_range(0, 7);
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kFg;
    topology.os_schedulable_cpus = primaries(all);
    std::reverse(topology.os_schedulable_cpus.begin(), topology.os_schedulable_cpus.end());

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 2, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{12, 14}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 3, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{12, 8, 14}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{12, 8, 10, 14}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{12, 8, 10, 0, 14}));
}

TEST(A5AicpuTopologySelection, ComputesPg1MinimisesSmtSharing) {
    // Die0 has 2 clusters, die1 has 1, so O is on die0. Scheduler
    // primaries are exhausted before any SMT sibling is selected.
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kPg1;
    topology.os_schedulable_cpus = make_physical_range(0, 5);

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 2, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 6}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 3, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 0, 6}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 0, 2, 6}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 0, 2, 8, 6}));
}

TEST(A5AicpuTopologySelection, Pg1UsesAvailablePrimaryBeforeSibling) {
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kPg1;
    topology.os_schedulable_cpus = make_physical_range(0, 2);
    topology.os_schedulable_cpus.erase(
        std::remove_if(
            topology.os_schedulable_cpus.begin(), topology.os_schedulable_cpus.end(),
            [](const AicpuLogicalCpu &cpu) {
                return cpu.phy_cpu_id == 1 && cpu.hyperthread_id == 1;
            }
        ),
        topology.os_schedulable_cpus.end()
    );

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 3, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{0, 2, 4}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{0, 2, 1, 4}));
}

TEST(A5AicpuTopologySelection, ComputesPg1SpillsWhenPreferredDieCannotFill) {
    // Tie on thread count → higher die_id (die1). O=phy7 ht0. Select the
    // remaining three physical primaries before phy6's sibling.
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kPg1;
    topology.os_schedulable_cpus = make_physical_range(0, 1);  // die0 pairs
    auto die1 = make_physical_range(6, 7);                     // phy6+phy7 on die1
    topology.os_schedulable_cpus.insert(topology.os_schedulable_cpus.end(), die1.begin(), die1.end());

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{12, 0, 2, 13, 14}));
}

TEST(A5AicpuTopologySelection, ComputesPg1DamagedFirstDiePacksOnSecondDie) {
    // First die lost a cluster: phys 2..7 → die0 has 1 cluster, die1 has 2.
    // O is on die1; keep schedulers on distinct physical CPUs.
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kPg1;
    topology.os_schedulable_cpus = make_physical_range(2, 7);

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{12, 8, 10, 4, 14}));
}

TEST(A5AicpuTopologySelection, ComputesPg2SharingPolicy) {
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kPg2;
    topology.os_schedulable_cpus = make_physical_range(0, 1);
    auto right_die = make_physical_range(4, 5);
    topology.os_schedulable_cpus.insert(topology.os_schedulable_cpus.end(), right_die.begin(), right_die.end());

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 2, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{8, 11}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 3, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{8, 9, 11}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{8, 9, 10, 11}));
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{8, 9, 10, 0, 11}));
}

TEST(A5AicpuTopologySelection, MatchesFgComputeOnlyDiagram) {
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kFg;
    topology.os_schedulable_cpus = {
        {4, 2, 0, 1, 0}, {6, 3, 0, 1, 0}, {8, 4, 0, 2, 1}, {10, 5, 0, 2, 1}, {12, 6, 0, 3, 1}, {14, 7, 0, 3, 1},
    };
    std::reverse(topology.os_schedulable_cpus.begin(), topology.os_schedulable_cpus.end());

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{12, 8, 10, 4, 14}));
}

TEST(A5AicpuTopologySelection, MatchesPg1ComputeOnlyDiagramWithMinimumSmtSharing) {
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kPg1;
    topology.scheduler_smt_enabled = true;
    topology.os_schedulable_cpus = {
        {4, 2, 0, 1, 0}, {6, 3, 0, 1, 0},  {7, 3, 1, 1, 0},  {8, 4, 0, 2, 1},
        {9, 4, 1, 2, 1}, {10, 5, 0, 2, 1}, {11, 5, 1, 2, 1},
    };
    std::reverse(topology.os_schedulable_cpus.begin(), topology.os_schedulable_cpus.end());

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 5, allowed));
    // S=T8,T4,T6,T9; O=T10. T8/T9 is the only scheduler SMT pair.
    EXPECT_EQ(allowed, (std::vector<int32_t>{8, 4, 6, 9, 10}));
}

TEST(A5AicpuTopologySelection, MatchesPg2ComputeOnlyDiagram) {
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kPg2;
    topology.scheduler_smt_enabled = true;
    topology.os_schedulable_cpus = {
        {3, 1, 1, 0, 0}, {4, 2, 0, 2, 1}, {5, 2, 1, 2, 1}, {6, 3, 0, 2, 1}, {7, 3, 1, 2, 1},
    };
    std::reverse(topology.os_schedulable_cpus.begin(), topology.os_schedulable_cpus.end());

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_scenario_allowed_cpus(topology, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{4, 5, 6, 3, 7}));
}

TEST(A5AicpuLaunchPlan, AutoShrinksAndWarnsBelowFiveStableCpus) {
    AicpuTopology topology;
    topology.source = AicpuTopologySource::kDriver;
    topology.scenario_type = AicpuScenarioType::kUnknown;
    topology.os_schedulable_cpus = {{1, -1, -1, -1, -1}, {3, -1, -1, -1, -1}, {5, -1, -1, -1, -1}, {7, -1, -1, -1, -1}};
    set_device_occupy(topology, (1ULL << 1) | (1ULL << 3) | (1ULL << 5) | (1ULL << 7));
    AicpuLaunchPlan plan;
    std::string error;

    ASSERT_TRUE(build_aicpu_launch_plan(topology, /*requested_active_count=*/0, plan, error)) << error;
    EXPECT_EQ(plan.effective_active_count, 4);
    EXPECT_EQ(plan.launch_count, 4);
    EXPECT_TRUE(plan.warn_stable_reachable_below_default);
    EXPECT_FALSE(plan.warn_cpu_topology_unavailable);
    EXPECT_EQ(plan.allowed_cpus, (std::vector<int32_t>{1, 3, 5, 7}));
}

TEST(A5AicpuLaunchPlan, AssignsRolesFromFiveCpuOccupyMask) {
    AicpuTopology topology;
    topology.source = AicpuTopologySource::kOccupyFallback;
    topology.scenario_type = AicpuScenarioType::kUnknown;
    ASSERT_TRUE(enumerate_cpus_from_occupy(0x3eU, topology.os_schedulable_cpus));
    set_device_occupy(topology, 0x3eU);
    std::reverse(topology.os_schedulable_cpus.begin(), topology.os_schedulable_cpus.end());
    AicpuLaunchPlan plan;
    std::string error;

    ASSERT_TRUE(build_aicpu_launch_plan(topology, 0, plan, error)) << error;
    EXPECT_EQ(plan.stable_reachable_count, 5);
    EXPECT_EQ(plan.effective_active_count, 5);
    EXPECT_EQ(plan.launch_count, 5);
    ASSERT_EQ(plan.allowed_cpus.size(), 5U);
    EXPECT_EQ(
        std::vector<int32_t>(plan.allowed_cpus.begin(), plan.allowed_cpus.end() - 1), (std::vector<int32_t>{1, 2, 3, 4})
    );
    EXPECT_EQ(plan.allowed_cpus.back(), 5);  // Orchestrator is always the last slot.
    EXPECT_TRUE(plan.warn_cpu_topology_unavailable);
    EXPECT_FALSE(plan.warn_stable_reachable_below_default);
}

TEST(A5AicpuLaunchPlan, WarnsWhenCpuTopologyFallsBackToPackagedJson) {
    std::vector<AicpuLogicalCpu> all_cpus;
    ASSERT_TRUE(load_cpu_topo_from_json("Ascend950PR_9599", 0x1f8, all_cpus));

    AicpuTopology topology;
    topology.source = AicpuTopologySource::kJsonFallback;
    topology.scenario_type = AicpuScenarioType::kFg;
    set_device_occupy(topology, 0x1f8U);
    for (const auto &cpu : all_cpus) {
        if (cpu.cpu_id >= 3 && cpu.cpu_id <= 8) topology.os_schedulable_cpus.push_back(cpu);
    }

    AicpuLaunchPlan plan;
    std::string error;
    ASSERT_TRUE(build_aicpu_launch_plan(topology, 0, plan, error)) << error;
    EXPECT_TRUE(plan.warn_cpu_topology_unavailable);
    EXPECT_FALSE(plan.warn_stable_reachable_below_default);
    EXPECT_EQ(plan.effective_active_count, 5);
    EXPECT_EQ(plan.launch_count, 6);
}

TEST(A5AicpuLaunchPlan, EnforcesManualAndLaunchCapacityWithoutClamping) {
    AicpuTopology topology;
    topology.source = AicpuTopologySource::kOccupyFallback;
    topology.scenario_type = AicpuScenarioType::kUnknown;
    ASSERT_TRUE(enumerate_cpus_from_occupy((1ULL << 4) - 1, topology.os_schedulable_cpus));
    set_device_occupy(topology, (1ULL << 4) - 1);
    AicpuLaunchPlan plan;
    std::string error;
    EXPECT_FALSE(build_aicpu_launch_plan(topology, 5, plan, error));

    ASSERT_TRUE(enumerate_cpus_from_occupy((1ULL << 14) - 1, topology.os_schedulable_cpus));
    set_device_occupy(topology, (1ULL << 14) - 1);
    ASSERT_TRUE(build_aicpu_launch_plan(topology, 5, plan, error)) << error;
    EXPECT_EQ(plan.launch_count, 14);

    ASSERT_TRUE(enumerate_cpus_from_occupy((1ULL << 15) - 1, topology.os_schedulable_cpus));
    set_device_occupy(topology, (1ULL << 15) - 1);
    EXPECT_FALSE(build_aicpu_launch_plan(topology, 5, plan, error));
    EXPECT_NE(error.find("launch capacity"), std::string::npos);
}

TEST(A5AicpuLaunchPlan, LaunchesFullOccupyPopulationWhenCpuTopoIsIncomplete) {
    AicpuTopology topology;
    topology.source = AicpuTopologySource::kDriver;
    topology.scenario_type = AicpuScenarioType::kUnknown;
    topology.os_schedulable_cpus = primaries(make_physical_range(1, 5));
    uint64_t occupy = 1ULL << 12;  // OCCUPY also exposes a CPU missing from CPU_TOPO metadata.
    for (const auto &cpu : topology.os_schedulable_cpus)
        occupy |= 1ULL << cpu.cpu_id;
    set_device_occupy(topology, occupy);

    AicpuLaunchPlan plan;
    std::string error;
    ASSERT_TRUE(build_aicpu_launch_plan(topology, 0, plan, error)) << error;
    EXPECT_EQ(plan.stable_reachable_count, 5);
    EXPECT_EQ(plan.effective_active_count, 5);
    EXPECT_EQ(plan.launch_count, 6);
    EXPECT_EQ(plan.allowed_cpus.size(), 5U);
}

TEST(A5AicpuTopologyDiagnostic, JsonIncludesEscapedTopologyAndLaunchDecision) {
    AicpuTopology topology;
    topology.soc_name = "Ascend950PR_\"test\"\\line\n";
    topology.source = AicpuTopologySource::kJsonFallback;
    topology.scenario_type = AicpuScenarioType::kUnknown;
    topology.logical_cpu_count = 6;
    topology.surviving_cluster_ids = {0, 2};
    topology.os_schedulable_cpus = {{1, 1, 0, 0, 0}, {2, 2, 0, 1, 0}};
    set_device_occupy(topology, 0x3eU);

    AicpuLaunchPlan plan;
    plan.requested_active_count = 0;
    plan.effective_active_count = 2;
    plan.stable_reachable_count = 2;
    plan.launch_count = 5;
    plan.allowed_cpus = {1, 2};
    const std::string json = format_aicpu_topology_json(topology, AicpuSelectionPolicy::kSequentialFallback, plan);

    EXPECT_NE(json.find("\"soc_name\": \"Ascend950PR_\\\"test\\\"\\\\line\\n\""), std::string::npos);
    EXPECT_NE(json.find("\"topology_source\": \"json_fallback\""), std::string::npos);
    EXPECT_NE(json.find("\"selection_policy\": \"sequential_fallback\""), std::string::npos);
    EXPECT_NE(json.find("\"launch_count\": 5"), std::string::npos);
    EXPECT_NE(json.find("\"allowed_cpus\": [1, 2]"), std::string::npos);
}

TEST(A5AicpuTopologySelection, UnknownFallbackUsesTopologyOrderAndExactCount) {
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kUnknown;
    topology.os_schedulable_cpus = {
        {50, 4, 1, 2, 1}, {7, 0, 0, 0, 0}, {8, 0, 1, 0, 0}, {12, 1, 0, 0, 0}, {2, 4, 0, 2, 1}, {99, 5, 0, 2, 1},
    };

    std::vector<int32_t> allowed;
    ASSERT_TRUE(compute_unknown_allowed_cpus(topology, 4, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{7, 8, 12, 2}));
    ASSERT_TRUE(compute_unknown_allowed_cpus(topology, 5, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{7, 8, 12, 2, 50}));
}

TEST(A5AicpuTopologySelection, UnknownFallbackRejectsInvalidOrInsufficientCounts) {
    AicpuTopology topology;
    topology.os_schedulable_cpus = {{7, 0, 0, 0, 0}};
    std::vector<int32_t> allowed = {123};
    EXPECT_FALSE(compute_unknown_allowed_cpus(topology, 2, allowed));
    EXPECT_TRUE(allowed.empty());

    topology.os_schedulable_cpus = {{1, -1, -1, -1, -1}, {3, -1, -1, -1, -1}, {5, -1, -1, -1, -1}};
    EXPECT_FALSE(compute_unknown_allowed_cpus(topology, 1, allowed));
    EXPECT_FALSE(compute_unknown_allowed_cpus(topology, 4, allowed));
    EXPECT_FALSE(compute_unknown_allowed_cpus(topology, 6, allowed));
    ASSERT_TRUE(compute_unknown_allowed_cpus(topology, 3, allowed));
    EXPECT_EQ(allowed, (std::vector<int32_t>{1, 3, 5}));
}

TEST(A5AicpuTopologySelection, KnownScenarioRejectsUnsupportedActiveCount) {
    AicpuTopology topology;
    topology.scenario_type = AicpuScenarioType::kFg;
    topology.os_schedulable_cpus = primaries(make_physical_range(0, 7));
    std::vector<int32_t> allowed = {123};

    EXPECT_FALSE(compute_scenario_allowed_cpus(topology, 1, allowed));
    EXPECT_TRUE(allowed.empty());
    EXPECT_FALSE(compute_scenario_allowed_cpus(topology, 6, allowed));
    EXPECT_TRUE(allowed.empty());
}

}  // namespace
