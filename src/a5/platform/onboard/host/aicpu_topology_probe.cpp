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
#include <cstring>

#include "aicpu_topology_driver.h"
#include "common/acl_hal_device.h"
#include "common/unified_log.h"

namespace pto::a5 {
namespace {

using pto::driver::CpuTopology;

// dlsym helpers — keep error reporting at WARN, callers fall back.
using HalGetDeviceInfoByBuffFn = decltype(&halGetDeviceInfoByBuff);
using DsmiGetDeviceInfoFn = decltype(&dsmi_get_device_info);
using AclrtGetSocNameFn = const char *(*)();

HalGetDeviceInfoByBuffFn load_hal_get_device_info_by_buff() {
    return reinterpret_cast<HalGetDeviceInfoByBuffFn>(dlsym(nullptr, "halGetDeviceInfoByBuff"));
}

DsmiGetDeviceInfoFn load_dsmi_get_device_info() {
    // First try the global namespace — works if some other component
    // already loaded libdrvdsmi_host.so. The simpler runtime doesn't, so
    // explicitly dlopen the driver library before re-trying. RTLD_GLOBAL
    // makes the symbols visible to future dlsym(nullptr,...) calls.
    auto fn = reinterpret_cast<DsmiGetDeviceInfoFn>(dlsym(nullptr, "dsmi_get_device_info"));
    if (fn != nullptr) return fn;
    static const char *const kDsmiLibs[] = {
        "libdrvdsmi_host.so",
        "/usr/local/Ascend/driver/lib64/driver/libdrvdsmi_host.so",
    };
    for (const char *path : kDsmiLibs) {
        if (dlopen(path, RTLD_LAZY | RTLD_GLOBAL) != nullptr) break;
    }
    fn = reinterpret_cast<DsmiGetDeviceInfoFn>(dlsym(nullptr, "dsmi_get_device_info"));
    if (fn == nullptr) LOG_WARN("aicpu_topology_probe: dsmi_get_device_info not found after dlopen fallback");
    return fn;
}

const char *query_soc_name() {
    auto fn = reinterpret_cast<AclrtGetSocNameFn>(dlsym(nullptr, "aclrtGetSocName"));
    if (fn == nullptr) {
        LOG_WARN("aicpu_topology_probe: aclrtGetSocName not found via dlsym");
        return nullptr;
    }
    return fn();
}

bool query_cpu_topo(uint32_t device_id, CpuTopology &out) {
    std::memset(&out, 0, sizeof(out));
    if (auto fn = load_hal_get_device_info_by_buff(); fn != nullptr) {
        int32_t sz = static_cast<int32_t>(sizeof(out));
        int rc =
            fn(static_cast<uint32_t>(pto::acl_to_hal_device_id(device_id)), MODULE_TYPE_SYSTEM,
               pto::driver::kCpuTopoHalInfoType, &out, &sz);
        if (rc == 0 && out.total_nums > 0 && out.total_nums <= pto::driver::kCpuTopoCapacity) return true;
        LOG_WARN("aicpu_topology_probe: halGetDeviceInfoByBuff(CPU_TOPO) rc=%d total=%u", rc, out.total_nums);
    }
    if (auto fn = load_dsmi_get_device_info(); fn != nullptr) {
        unsigned int sz = static_cast<unsigned int>(sizeof(out));
        // DSMI is a driver-level API (bypasses ACL), so like the hal* calls it needs the
        // driver-visible id. Device 0 confirms this on a5; other ids are unverified.
        int rc =
            fn(static_cast<uint32_t>(pto::acl_to_hal_device_id(device_id)), DSMI_MAIN_CMD_SOC_INFO,
               pto::driver::kCpuTopoDsmiSubcommand, &out, &sz);
        if (rc == 0 && out.total_nums > 0 && out.total_nums <= pto::driver::kCpuTopoCapacity) return true;
        LOG_WARN("aicpu_topology_probe: dsmi_get_device_info(CPU_TOPO) rc=%d total=%u", rc, out.total_nums);
    }
    return false;
}

}  // namespace

bool probe_aicpu_topology(
    uint32_t device_id, const AicpuDeviceOccupancy &device_occupancy, AicpuTopology &out_topology
) {
    out_topology = {};
    out_topology.device_occupancy = device_occupancy;
    if (!device_occupancy.occupy_valid || device_occupancy.occupy == 0) return false;

    const char *soc_name = query_soc_name();
    CpuTopology driver_topo{};
    const bool available =
        query_cpu_topo(device_id, driver_topo) && driver_topo.total_nums <= detail::kCpuOccupancyBits;
    detail::CpuTopologyData topo{};
    if (available) {
        topo.total_nums = driver_topo.total_nums;
        for (uint32_t i = 0; i < topo.total_nums; ++i) {
            const auto &cpu = driver_topo.single_cpu_topo_info[i];
            topo.cpus[i] = {cpu.cpu_mask, cpu.cpu_id, cpu.is_share, cpu.phy_cpu_id, cpu.hyperthread_id};
        }
    }
    return detail::build_aicpu_topology(device_occupancy, soc_name, available, topo, out_topology);
}

}  // namespace pto::a5
