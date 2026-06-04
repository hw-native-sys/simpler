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

#ifndef SRC_A5_PLATFORM_ONBOARD_HOST_AICPU_TOPOLOGY_PROBE_H_
#define SRC_A5_PLATFORM_ONBOARD_HOST_AICPU_TOPOLOGY_PROBE_H_

#include <cstdint>
#include <vector>

namespace pto::a5 {

// Per-cpu_id metadata used by the packing algorithm. Filled from DSMI
// CPU_TOPO + halGetDeviceInfo(AICPU, OCCUPY). cluster/die ids derive from
// phy_cpu_id via the a5 mapping (cluster = phy/2, die = phy/4).
struct AicpuLogicalCpu {
    int32_t cpu_id;
    int32_t phy_cpu_id;
    int32_t hyperthread_id;  // 0 or 1; 0 for non-SMT phys
    int32_t cluster_id;      // phy_cpu_id / 2
    int32_t die_id;          // phy_cpu_id / 4
};

// Probe device-side AICPU topology. Returns true iff the user pool was
// successfully resolved (at least one entry in `out_user_cpus`). The output
// only contains cpu_ids that are in the device-side OCCUPY bitmap (i.e.
// user-schedulable), sorted by cpu_id ascending.
//
// This function performs three driver calls:
//   * halGetDeviceInfo(AICPU, OCCUPY) — user-schedulable bitmap
//   * halGetDeviceInfoByBuff(SYSTEM, CPU_TOPO)  (primary)
//   * dsmi_get_device_info(SOC_INFO, CPU_TOPO)  (fallback)
//
// All driver entry points are dlsym'd from the host process (CANN is
// expected to be already loaded by the surrounding `aclInit` path).
bool probe_aicpu_topology(uint32_t device_id, std::vector<AicpuLogicalCpu> &out_user_cpus);

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

}  // namespace pto::a5

#endif  // SRC_A5_PLATFORM_ONBOARD_HOST_AICPU_TOPOLOGY_PROBE_H_
