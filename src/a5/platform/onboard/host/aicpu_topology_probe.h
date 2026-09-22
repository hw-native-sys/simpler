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
#include <string>
#include <vector>

namespace pto::a5 {

// Per-cpu_id metadata used by the packing algorithm. Filled from CPU_TOPO
// data when available. Topology fields are -1 in the OCCUPY-only fallback.
struct AicpuLogicalCpu {
    int32_t cpu_id;
    int32_t phy_cpu_id;
    int32_t hyperthread_id;  // 0 or 1; 0 for non-SMT phys
    int32_t cluster_id;      // phy_cpu_id / 2
    int32_t die_id;          // phy_cpu_id / 4
};

enum class AicpuScenarioType {
    kNotApplicable,
    kFg,
    kPg1,
    kPg2,
    kUnknown,
};

enum class AicpuTopologySource {
    kDriver,
    kJsonFallback,
    kOccupyFallback,
};

enum class AicpuSelectionPolicy {
    kScenario,
    kGeneric,
    kSequentialFallback,
};

struct AicpuDeviceOccupancy {
    uint64_t occupy{0};
    uint64_t pf_occupy{0};
    uint64_t os_sched{0};
    bool occupy_valid{false};
    bool pf_occupy_valid{false};
    bool os_sched_valid{false};
};

struct AicpuTopology {
    std::string soc_name;
    AicpuTopologySource source{AicpuTopologySource::kDriver};
    AicpuScenarioType scenario_type{AicpuScenarioType::kUnknown};
    bool scheduler_smt_enabled{false};
    uint32_t logical_cpu_count{0};
    std::vector<int32_t> surviving_cluster_ids;
    std::vector<AicpuLogicalCpu> os_schedulable_cpus;
    AicpuDeviceOccupancy device_occupancy;
    bool generic_selection_only{false};
};

// Complete host decision for one AICPU launch. The affinity convention is
// [scheduler..., orchestrator], so allowed_cpus.back() always carries O.
struct AicpuLaunchPlan {
    int32_t requested_active_count{0};  // 0 means automatic
    int32_t effective_active_count{0};
    int32_t stable_reachable_count{0};
    int32_t launch_count{0};
    std::vector<int32_t> allowed_cpus;
    bool warn_stable_reachable_below_default{false};
    bool warn_cpu_topology_unavailable{false};
};

// Preserve the original verified x86 standard-card fallback contract.
bool derive_topology_from_occupy(const char *soc_name, uint64_t occupy, std::vector<AicpuLogicalCpu> &out_user_cpus);

// Merge host CPU_TOPO metadata with the authoritative device-side scheduler
// pool and classify the resulting A5 topology. Returns false only when the
// probe itself is unusable; an unrecognised but internally valid shape is
// returned successfully with scenario_type == kUnknown.
bool probe_aicpu_topology(
    uint32_t device_id, const AicpuDeviceOccupancy &device_occupancy, AicpuTopology &out_topology
);

// Enumerate OCCUPY set bits without inferring any topology relationships.
// Returns false when the mask is empty.
bool enumerate_cpus_from_occupy(uint64_t occupy, std::vector<AicpuLogicalCpu> &out_user_cpus);

// Load the full logical CPU_TOPO for a packaged fallback whose SoC and every
// constraint declared by that entry (host architecture and/or OCCUPY) match.
// Output is not OCCUPY-filtered. `out_generic_selection_only` reports entries
// that must keep using compute_allowed_cpus() instead of scenario selection.
// Returns false when the signature is absent, mismatched, or unusable.
bool load_cpu_topo_from_json(
    const char *soc_name, uint64_t occupy, std::vector<AicpuLogicalCpu> &out_all_cpus,
    bool *out_generic_selection_only = nullptr
);

// Compute the `ALLOWED_CPUS` selection for the surviving threads.
//
// Inputs:
//   * `user_cpus`  — the user-schedulable pool from `probe_aicpu_topology`
//   * `n_sched`    — number of scheduler threads (sched 0..n_sched-1)
//   * `n_orch`     — number of orchestrator threads (currently always 1)
//
// Output:
//   * `out_allowed_cpus` — n_sched + n_orch cpu_ids, ordered as
//     [sched 0..n_sched-1, orch 0..n_orch-1].  The on-device gate uses
//     this as `ALLOWED_CPUS[]`; the index in this array IS the deterministic
//     `exec_idx` the surviving thread receives, so the role assignment in
//     `aicpu_executor.cpp` (sched / orch) is fully driven by the order here.
//
// Placement policy:
//   Step 1 (sched): smallest containing unit wins —
//     1.1 a single cluster with >= n_sched logical cpus, else
//     1.2 a single die with  >= n_sched user logical cpus, else
//     1.3 spread across dies.
//     Within the chosen unit, fill phys round-robin (ht 0 of each phy
//     before doubling up SMT siblings) to minimise pairwise SMT contention.
//     Tiebreaker between candidate units: highest cluster_id / die_id
//     (= farthest from cpu_id 0 / AICPU OS).
//
//   Step 2 (orch): placed AFTER sched, in the sched-relative priority
//     order  same cluster > same die > different die.  Within the chosen
//     unit, the lowest free cpu_id wins.
//
// Returns true iff `out_allowed_cpus.size() == n_sched + n_orch`.
bool compute_allowed_cpus(
    const std::vector<AicpuLogicalCpu> &user_cpus, int32_t n_sched, int32_t n_orch,
    std::vector<int32_t> &out_allowed_cpus
);

// Classify an internally valid A5 topology from raw logical topology and the
// driver-filtered scheduler pool. FG / PG1 / PG2 come from the surviving
// cluster/die layout. Scheduler SMT availability is recorded separately in
// AicpuTopology::scheduler_smt_enabled and does not define another scenario.
// Logical CPU count is not used as a scenario gate.
AicpuScenarioType classify_aicpu_scenario(
    uint32_t logical_cpu_count, const std::vector<AicpuLogicalCpu> &all_logical_cpus,
    const std::vector<AicpuLogicalCpu> &os_schedulable_cpus
);

// Compute the documented topology policy for a known A5 scenario with an
// active count in [2, 5]. Output order is [S0, ..., S(active_count-2), O].
// The output is empty on failure.
bool compute_scenario_allowed_cpus(
    const AicpuTopology &topology, int32_t active_count, std::vector<int32_t> &out_allowed_cpus
);

// Unknown-topology fallback: select exactly active_count CPUs in [2, 5]. Use
// topology order when metadata is valid; OCCUPY-only metadata naturally
// degenerates to cpu_id order. The last selected CPU is O and all preceding
// CPUs are S. The output is empty when capacity is insufficient.
bool compute_unknown_allowed_cpus(
    const AicpuTopology &topology, int32_t active_count, std::vector<int32_t> &out_allowed_cpus
);

// Resolve automatic/manual active count, affinity and physical launch count.
// Automatic mode shrinks when fewer than PLATFORM_DEFAULT_AICPU_THREAD_NUM CPUs are
// stably reachable. Manual mode is exact. Launch coverage is never clamped.
bool build_aicpu_launch_plan(
    const AicpuTopology &topology, int32_t requested_active_count, AicpuLaunchPlan &out_plan, std::string &out_error
);

const char *aicpu_scenario_name(AicpuScenarioType scenario);
const char *aicpu_topology_source_name(AicpuTopologySource source);

// Serialize the complete topology and launch decision for diagnostic tooling.
// The result is a standalone JSON object terminated by a newline.
std::string format_aicpu_topology_json(
    const AicpuTopology &topology, AicpuSelectionPolicy policy, const AicpuLaunchPlan &launch_plan
);

}  // namespace pto::a5
