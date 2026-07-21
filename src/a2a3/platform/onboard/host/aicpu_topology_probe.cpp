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

#include <dlfcn.h>

#include <algorithm>
#include <mutex>
#include <unordered_map>

#include "common/unified_log.h"

namespace pto::a2a3 {

namespace {

// halGetDeviceInfo module/info selectors (CANN driver ABI). Provenance:
// driver/ascend_hal.h — replicated here to keep the runtime .so free of a
// CANN header dependency (see tools/cann-examples/query for the reference use).
constexpr int32_t kModuleAicpu = 1;
constexpr int32_t kInfoOccupy = 8;
// a2a3 AICPU has NO SMT: each logical cpu_id maps 1:1 to a physical core, so
// the cluster can be derived arithmetically from cpu_id alone (no DSMI CPU_TOPO
// probe is needed, unlike a5). 8 cores/die, 4 cores/cluster ⇒ 2 clusters/die.
constexpr int32_t kAicpuCoresPerDie = 8;
constexpr int32_t kCpusPerCluster = 4;

using HalGetDeviceInfoFn = int (*)(uint64_t deviceId, int32_t moduleType, int32_t infoType, int64_t *value);

HalGetDeviceInfoFn load_hal_get_device_info() {
    static HalGetDeviceInfoFn cached_fn = []() -> HalGetDeviceInfoFn {
        auto fn = reinterpret_cast<HalGetDeviceInfoFn>(dlsym(nullptr, "halGetDeviceInfo"));
        if (fn != nullptr) return fn;
        static const char *const kHalLibs[] = {
            "libascend_hal.so",
            "/usr/local/Ascend/driver/lib64/driver/libascend_hal.so",
        };
        for (const char *path : kHalLibs) {
            if (dlopen(path, RTLD_LAZY | RTLD_GLOBAL) == nullptr) continue;
            fn = reinterpret_cast<HalGetDeviceInfoFn>(dlsym(nullptr, "halGetDeviceInfo"));
            if (fn != nullptr) return fn;
        }
        LOG_WARN("a2a3_aicpu_topology_probe: halGetDeviceInfo not found after dlopen fallback");
        return nullptr;
    }();
    return cached_fn;
}

bool query_occupy(uint32_t device_id, uint64_t &out_mask) {
    auto fn = load_hal_get_device_info();
    if (fn == nullptr) return false;

    int64_t v = 0;
    int rc = fn(static_cast<uint64_t>(device_id), kModuleAicpu, kInfoOccupy, &v);
    if (rc != 0) {
        LOG_WARN("a2a3_aicpu_topology_probe: halGetDeviceInfo(AICPU,OCCUPY) rc=%d", rc);
        return false;
    }

    out_mask = static_cast<uint64_t>(v);
    return true;
}

std::mutex s_topo_cache_mu;
std::unordered_map<uint32_t, std::vector<AicpuLogicalCpu>> s_topo_cache;

bool probe_aicpu_topology_uncached(uint32_t device_id, std::vector<AicpuLogicalCpu> &out_user_cpus) {
    out_user_cpus.clear();

    uint64_t occupy = 0;
    if (!query_occupy(device_id, occupy)) return false;

    for (int32_t cpu_id = 0; cpu_id < 64; ++cpu_id) {
        if (((occupy >> cpu_id) & 1ULL) == 0) continue;
        AicpuLogicalCpu e{};
        e.cpu_id = cpu_id;
        e.cluster_id = (cpu_id % kAicpuCoresPerDie) / kCpusPerCluster;
        out_user_cpus.push_back(e);
    }

    std::sort(out_user_cpus.begin(), out_user_cpus.end(), [](const AicpuLogicalCpu &a, const AicpuLogicalCpu &b) {
        return a.cpu_id < b.cpu_id;
    });
    return !out_user_cpus.empty();
}

}  // namespace

bool probe_aicpu_topology(uint32_t device_id, std::vector<AicpuLogicalCpu> &out_user_cpus) {
    {
        std::lock_guard<std::mutex> lk(s_topo_cache_mu);
        auto it = s_topo_cache.find(device_id);
        if (it != s_topo_cache.end()) {
            out_user_cpus = it->second;
            return !out_user_cpus.empty();
        }
    }

    std::vector<AicpuLogicalCpu> probed;
    bool ok = probe_aicpu_topology_uncached(device_id, probed);
    if (!ok) {
        out_user_cpus.clear();
        return false;
    }

    LOG_INFO_V0("A2A3 AICPU topology probed for device %u: %zu user-schedulable cpu_ids", device_id, probed.size());

    {
        std::lock_guard<std::mutex> lk(s_topo_cache_mu);
        s_topo_cache[device_id] = probed;
    }
    out_user_cpus = std::move(probed);
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
