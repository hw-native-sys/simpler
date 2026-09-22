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

#include "ascend_hal.h"
#include "common/unified_log.h"
#include "common/acl_hal_device.h"

namespace pto::a2a3 {

namespace {

// Four CPUs per cluster, so a die's eight IDs form two clusters. a2a3 AICPU has
// no SMT, which is what lets the cluster be derived from the ID arithmetically —
// no DSMI CPU_TOPO probe is needed here, unlike a5.
constexpr int32_t kCpusPerCluster = 4;

using HalGetDeviceInfoFn = decltype(&halGetDeviceInfo);

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
    int rc = fn(static_cast<uint32_t>(pto::acl_to_hal_device_id(device_id)), MODULE_TYPE_AICPU, INFO_TYPE_OCCUPY, &v);
    if (rc != 0) {
        LOG_WARN("a2a3_aicpu_topology_probe: halGetDeviceInfo(AICPU,OCCUPY) rc=%d", rc);
        return false;
    }

    out_mask = static_cast<uint64_t>(v);
    return true;
}

bool query_phy_die_id(uint32_t device_id, int64_t &out_phy_die_id) {
    auto fn = load_hal_get_device_info();
    if (fn == nullptr) return false;

    int64_t value = -1;
    int rc =
        fn(static_cast<uint32_t>(pto::acl_to_hal_device_id(device_id)), MODULE_TYPE_SYSTEM, INFO_TYPE_PHY_DIE_ID,
           &value);
    if (rc != 0) {
        LOG_ERROR("a2a3_aicpu_topology_probe: halGetDeviceInfo(SYSTEM,PHY_DIE_ID) rc=%d", rc);
        return false;
    }
    out_phy_die_id = value;
    return true;
}

std::mutex s_topo_cache_mu;
std::unordered_map<uint32_t, std::vector<AicpuLogicalCpu>> s_topo_cache;

bool probe_aicpu_topology_uncached(uint32_t device_id, std::vector<AicpuLogicalCpu> &out_user_cpus) {
    out_user_cpus.clear();

    uint64_t occupy = 0;
    if (!query_occupy(device_id, occupy)) return false;

    if ((occupy >> kAicpuCoresPerDie) != 0) {
        LOG_ERROR(
            "a2a3_aicpu_topology_probe: OCCUPY 0x%llx has bits outside the %d-bit die-local namespace",
            static_cast<unsigned long long>(occupy), kAicpuCoresPerDie
        );
        return false;
    }

    int64_t phy_die_id = -1;
    if (!query_phy_die_id(device_id, phy_die_id)) return false;

    int32_t cpu_id_base = 0;
    if (!resolve_aicpu_cpu_id_base(phy_die_id, cpu_id_base)) return false;

    for (int32_t local_cpu_id = 0; local_cpu_id < kAicpuCoresPerDie; ++local_cpu_id) {
        if (((occupy >> local_cpu_id) & 1ULL) == 0) continue;
        AicpuLogicalCpu e{};
        e.cpu_id = cpu_id_base + local_cpu_id;
        e.cluster_id = local_cpu_id / kCpusPerCluster;
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

    LOG_INFO("A2A3 AICPU topology probed for device %u: %zu user-schedulable cpu_ids", device_id, probed.size());

    {
        std::lock_guard<std::mutex> lk(s_topo_cache_mu);
        s_topo_cache[device_id] = probed;
    }
    out_user_cpus = std::move(probed);
    return true;
}

}  // namespace pto::a2a3
