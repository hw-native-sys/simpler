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

#include "aicpu_topology_probe.h"

#include <algorithm>

#include "common/unified_log.h"

namespace pto::a2a3 {

bool resolve_aicpu_cpu_id_base(int64_t phy_die_id, int32_t &out_cpu_id_base) {
    out_cpu_id_base = 0;
    if (phy_die_id < 0 || phy_die_id > 1) {
        LOG_ERROR("a2a3_aicpu_topology_probe: invalid PHY_DIE_ID %lld", static_cast<long long>(phy_die_id));
        return false;
    }
    out_cpu_id_base = static_cast<int32_t>(phy_die_id) * kAicpuCoresPerDie;
    return true;
}

bool compute_allowed_cpus(
    const std::vector<AicpuLogicalCpu> &user_cpus, int32_t active_count, std::vector<int32_t> &out_allowed_cpus
) {
    out_allowed_cpus.clear();
    if (active_count <= 0 || static_cast<int32_t>(user_cpus.size()) < active_count) return false;

    int32_t max_cluster = -1;
    for (const auto &c : user_cpus)
        max_cluster = std::max(max_cluster, c.cluster_id);
    if (max_cluster < 0) return false;

    std::vector<std::vector<int32_t>> buckets(max_cluster + 1);
    for (int32_t i = 0; i < static_cast<int32_t>(user_cpus.size()); ++i) {
        if (user_cpus[i].cluster_id >= 0 && user_cpus[i].cluster_id <= max_cluster) {
            buckets[user_cpus[i].cluster_id].push_back(i);
        }
    }

    // Prefer one cluster that can hold all active AICPU threads. On the
    // observed 0xfc pool this selects cpu_id 4..7 for active_count=4.
    int32_t chosen_cluster = -1;
    for (int32_t cluster = max_cluster; cluster >= 0; --cluster) {
        if (static_cast<int32_t>(buckets[cluster].size()) >= active_count) {
            chosen_cluster = cluster;
            break;
        }
    }

    if (chosen_cluster >= 0) {
        auto ordered = buckets[chosen_cluster];
        std::sort(ordered.begin(), ordered.end(), [&](int32_t a, int32_t b) {
            return user_cpus[a].cpu_id < user_cpus[b].cpu_id;
        });
        for (int32_t i = 0; i < active_count; ++i) {
            out_allowed_cpus.push_back(user_cpus[ordered[i]].cpu_id);
        }
        return true;
    }

    // No single cluster fits the request (for example a pathological 3+3 pool
    // with active_count=4). Keep deterministic behavior rather than doing
    // device-side majority classification again. This crosses a NUMA boundary
    // and reintroduces the cross-cluster penalty issue #1045 is about, so warn
    // loudly — the only reason to be here is active_count exceeding a single
    // cluster, which is outside the supported topology.
    LOG_WARN(
        "A2A3 AICPU: no single cluster holds %d active threads (user_cpus=%zu); "
        "falling back to cross-cluster selection — expect NUMA-crossing slowdown",
        active_count, user_cpus.size()
    );
    std::vector<AicpuLogicalCpu> ordered = user_cpus;
    std::sort(ordered.begin(), ordered.end(), [](const AicpuLogicalCpu &a, const AicpuLogicalCpu &b) {
        return a.cpu_id < b.cpu_id;
    });
    for (int32_t i = 0; i < active_count; ++i) {
        out_allowed_cpus.push_back(ordered[i].cpu_id);
    }
    return true;
}

}  // namespace pto::a2a3
