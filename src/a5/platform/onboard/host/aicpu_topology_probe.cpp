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
#include <cctype>
#include <cstddef>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <set>
#include <sstream>
#include <string>
#include <tuple>
#include <unordered_map>

#include "common/platform_config.h"
#include "common/unified_log.h"
#include "common/acl_hal_device.h"

namespace pto::a5 {

namespace {

// HAL constants — replicated locally so we don't need to pull
// driver/ascend_hal_base.h into the public header.  Values match
// `tools/cann-examples/query/query.cpp`'s usage.
constexpr int32_t kModuleSystem = 0;
constexpr int32_t kInfoCpuTopo = 59;

// DSMI constants for SOC_INFO + CPU_TOPO. Verified against
// $ASCEND_HOME_PATH/aarch64-linux/include/driver/dsmi_common_interface.h:
// the dsmi_main_cmd enum jumps from CAN=3 to UPGRADE=5, so SOC_INFO = 14
// (0x0e), NOT 0x10. Reference tools/cann-examples/query/query.cpp picks
// it up via the header symbol; we hardcode here to keep the runtime SO
// dependency-free.
constexpr unsigned int kDsmiSocInfoMainCmd = 14;       // DSMI_MAIN_CMD_SOC_INFO
constexpr unsigned int kDsmiSocInfoSubCmdCpuTopo = 2;  // SUB_CMD_CPU_TOPO
constexpr unsigned int kCpuTopoMaxLogical = 64;        // headroom for any a5 SKU
constexpr char kVerifiedFallbackSoc[] = "Ascend950PR_9579";
constexpr uint64_t kVerifiedFallbackOccupy = 0x3e;

// Natural-alignment layout (no pack pragma). Mirrors
// tools/cann-examples/query/query.cpp's struct; the HAL/DSMI driver
// expects this byte layout — packing breaks the size check inside the
// driver and yields rc=65534.
struct DsmiSingleCpu {
    uint64_t cpu_mask;
    uint8_t cpu_id;
    uint8_t is_share;
    uint8_t phy_cpu_id;
    uint8_t hyperthread_id;
};
struct DsmiCpuTopo {
    uint32_t total_nums;
    DsmiSingleCpu cpus[kCpuTopoMaxLogical];
};

static_assert(sizeof(DsmiSingleCpu) == 16, "DsmiSingleCpu ABI size drift");
static_assert(offsetof(DsmiSingleCpu, cpu_id) == 8, "DsmiSingleCpu::cpu_id offset drift");
static_assert(offsetof(DsmiSingleCpu, hyperthread_id) == 11, "DsmiSingleCpu::hyperthread_id offset drift");
static_assert(offsetof(DsmiCpuTopo, cpus) == 8, "DsmiCpuTopo::cpus offset drift");

bool fill_dsmi_topo_from_json(
    const char *soc_name, uint64_t occupy, DsmiCpuTopo &out, bool &out_generic_selection_only
);

// dlsym helpers — keep error reporting at WARN, callers fall back.
using HalGetDeviceInfoByBuffFn =
    int (*)(uint64_t deviceId, int32_t moduleType, int32_t infoType, void *buf, int32_t *size);
using DsmiGetDeviceInfoFn =
    int (*)(uint32_t device_id, unsigned int main_cmd, unsigned int sub_cmd, void *buf, unsigned int *size);
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

bool query_cpu_topo(uint32_t device_id, DsmiCpuTopo &out) {
    std::memset(&out, 0, sizeof(out));
    if (auto fn = load_hal_get_device_info_by_buff(); fn != nullptr) {
        int32_t sz = static_cast<int32_t>(sizeof(out));
        int rc =
            fn(static_cast<uint64_t>(pto::acl_to_hal_device_id(device_id)), kModuleSystem, kInfoCpuTopo, &out, &sz);
        if (rc == 0 && out.total_nums > 0 && out.total_nums <= kCpuTopoMaxLogical) return true;
        LOG_WARN("aicpu_topology_probe: halGetDeviceInfoByBuff(CPU_TOPO) rc=%d total=%u", rc, out.total_nums);
    }
    if (auto fn = load_dsmi_get_device_info(); fn != nullptr) {
        unsigned int sz = static_cast<unsigned int>(sizeof(out));
        // DSMI is a driver-level API (bypasses ACL), so like the hal* calls it needs the
        // driver-visible id. NOTE (unverified — no a5 hardware): this assumes
        // dsmi_get_device_info indexes the same driver-visible id space as the HAL; if DSMI
        // uses its own logic-id numbering this is wrong. Revisit once an a5 node is available.
        int rc =
            fn(static_cast<uint32_t>(pto::acl_to_hal_device_id(device_id)), kDsmiSocInfoMainCmd,
               kDsmiSocInfoSubCmdCpuTopo, &out, &sz);
        if (rc == 0 && out.total_nums > 0 && out.total_nums <= kCpuTopoMaxLogical) return true;
        LOG_WARN("aicpu_topology_probe: dsmi_get_device_info(CPU_TOPO) rc=%d total=%u", rc, out.total_nums);
    }
    return false;
}

}  // namespace

bool derive_topology_from_occupy(const char *soc_name, uint64_t occupy, std::vector<AicpuLogicalCpu> &out_user_cpus) {
    out_user_cpus.clear();
#if defined(__aarch64__)
    (void)soc_name;
    (void)occupy;
    return false;
#elif defined(__x86_64__)
    if (soc_name == nullptr || std::strcmp(soc_name, kVerifiedFallbackSoc) != 0 || occupy != kVerifiedFallbackOccupy) {
        return false;
    }
#else
    (void)soc_name;
    (void)occupy;
    return false;
#endif
    for (int32_t cpu_id = 0; cpu_id < static_cast<int32_t>(kCpuTopoMaxLogical); ++cpu_id) {
        if (((occupy >> cpu_id) & 1ULL) == 0) continue;
        AicpuLogicalCpu cpu{};
        cpu.cpu_id = cpu_id;
        cpu.phy_cpu_id = cpu_id;
        cpu.hyperthread_id = 0;
        cpu.cluster_id = cpu.phy_cpu_id / 2;
        cpu.die_id = cpu.phy_cpu_id / 4;
        out_user_cpus.push_back(cpu);
    }
    return !out_user_cpus.empty();
}

bool enumerate_cpus_from_occupy(uint64_t occupy, std::vector<AicpuLogicalCpu> &out_user_cpus) {
    out_user_cpus.clear();
    for (int32_t cpu_id = 0; cpu_id < static_cast<int32_t>(kCpuTopoMaxLogical); ++cpu_id) {
        if (((occupy >> cpu_id) & 1ULL) == 0) continue;
        AicpuLogicalCpu cpu{};
        cpu.cpu_id = cpu_id;
        cpu.phy_cpu_id = -1;
        cpu.hyperthread_id = -1;
        cpu.cluster_id = -1;
        cpu.die_id = -1;
        out_user_cpus.push_back(cpu);
    }
    return !out_user_cpus.empty();
}

namespace {
bool validate_cpu_topology(const std::vector<AicpuLogicalCpu> &cpus);
bool validate_cpu_ids(const std::vector<AicpuLogicalCpu> &cpus);
std::vector<int32_t> clusters_of(const std::vector<AicpuLogicalCpu> &cpus);
bool has_schedulable_smt_pair(const std::vector<AicpuLogicalCpu> &cpus);
}  // namespace

bool probe_aicpu_topology(
    uint32_t device_id, const AicpuDeviceOccupancy &device_occupancy, AicpuTopology &out_topology
) {
    out_topology = {};
    out_topology.device_occupancy = device_occupancy;
    if (!device_occupancy.occupy_valid || device_occupancy.occupy == 0) return false;

    const char *soc_name = query_soc_name();
    if (soc_name != nullptr) out_topology.soc_name = soc_name;

    DsmiCpuTopo topo{};
    std::vector<AicpuLogicalCpu> all_cpus;
    const char *source = "driver";
    bool occupy_only_fallback = false;
    if (!query_cpu_topo(device_id, topo)) {
        if (load_cpu_topo_from_json(
                soc_name, device_occupancy.occupy, all_cpus, &out_topology.generic_selection_only
            )) {
            source = "json_fallback";
            out_topology.source = AicpuTopologySource::kJsonFallback;
            out_topology.logical_cpu_count = static_cast<uint32_t>(all_cpus.size());
        } else if (enumerate_cpus_from_occupy(device_occupancy.occupy, all_cpus)) {
            source = "occupy_fallback";
            out_topology.source = AicpuTopologySource::kOccupyFallback;
            occupy_only_fallback = true;
            out_topology.logical_cpu_count = static_cast<uint32_t>(all_cpus.size());
        } else {
            return false;
        }
    } else {
        out_topology.logical_cpu_count = topo.total_nums;
        all_cpus.reserve(topo.total_nums);
        for (uint32_t i = 0; i < topo.total_nums; ++i) {
            const DsmiSingleCpu &cpu = topo.cpus[i];
            if (cpu.cpu_id >= kCpuTopoMaxLogical) return false;
            AicpuLogicalCpu entry{};
            entry.cpu_id = static_cast<int32_t>(cpu.cpu_id);
            entry.phy_cpu_id = static_cast<int32_t>(cpu.phy_cpu_id);
            entry.hyperthread_id = static_cast<int32_t>(cpu.hyperthread_id);
            entry.cluster_id = entry.phy_cpu_id / 2;
            entry.die_id = entry.phy_cpu_id / 4;
            all_cpus.push_back(entry);
        }
    }
    LOG_INFO(
        "aicpu_topology_probe: CPU_TOPO source=%s soc=%s logical=%u", source,
        out_topology.soc_name.empty() ? "(unknown)" : out_topology.soc_name.c_str(), out_topology.logical_cpu_count
    );

    std::sort(all_cpus.begin(), all_cpus.end(), [](const auto &a, const auto &b) {
        return a.cpu_id < b.cpu_id;
    });
    for (const auto &cpu : all_cpus) {
        if (cpu.cpu_id < 64 && ((device_occupancy.occupy >> cpu.cpu_id) & 1ULL) != 0) {
            out_topology.os_schedulable_cpus.push_back(cpu);
        }
    }
    const bool valid_cpus = occupy_only_fallback ? validate_cpu_ids(all_cpus) : validate_cpu_topology(all_cpus);
    const bool valid_schedulable = occupy_only_fallback ? validate_cpu_ids(out_topology.os_schedulable_cpus) :
                                                          validate_cpu_topology(out_topology.os_schedulable_cpus);
    if (!valid_cpus || !valid_schedulable || out_topology.os_schedulable_cpus.empty()) {
        out_topology = {};
        return false;
    }
    if (!occupy_only_fallback) {
        out_topology.surviving_cluster_ids = clusters_of(all_cpus);
        out_topology.scheduler_smt_enabled = has_schedulable_smt_pair(out_topology.os_schedulable_cpus);
        out_topology.scenario_type =
            classify_aicpu_scenario(out_topology.logical_cpu_count, all_cpus, out_topology.os_schedulable_cpus);
    }

    {
        std::ostringstream dies_os, clusters_os, smt_os, sched_os;
        std::set<int32_t> dies;
        for (const auto &cpu : all_cpus)
            dies.insert(cpu.die_id);
        for (int32_t d : dies) {
            if (dies_os.tellp() > 0) dies_os << ',';
            dies_os << d;
        }
        for (size_t i = 0; i < out_topology.surviving_cluster_ids.size(); ++i) {
            if (i) clusters_os << ',';
            clusters_os << out_topology.surviving_cluster_ids[i];
        }
        std::unordered_map<int32_t, std::vector<int32_t>> phy_to_cpus;
        for (const auto &cpu : all_cpus)
            phy_to_cpus[cpu.phy_cpu_id].push_back(cpu.cpu_id);
        for (auto &entry : phy_to_cpus) {
            auto &ids = entry.second;
            if (ids.size() < 2) continue;
            std::sort(ids.begin(), ids.end());
            if (smt_os.tellp() > 0) smt_os << ',';
            smt_os << "phy" << entry.first << ':';
            for (size_t i = 0; i < ids.size(); ++i) {
                if (i) smt_os << '+';
                smt_os << ids[i];
            }
        }
        for (size_t i = 0; i < out_topology.os_schedulable_cpus.size(); ++i) {
            if (i) sched_os << ',';
            sched_os << out_topology.os_schedulable_cpus[i].cpu_id;
        }
        LOG_DEBUG(
            "aicpu_topology: scenario=%s dies=[%s] clusters=[%s] smt_pairs=[%s] "
            "schedulable=[%s] scheduler_smt_enabled=%d logical=%u",
            aicpu_scenario_name(out_topology.scenario_type), dies_os.str().c_str(), clusters_os.str().c_str(),
            smt_os.str().c_str(), sched_os.str().c_str(), out_topology.scheduler_smt_enabled ? 1 : 0,
            out_topology.logical_cpu_count
        );
    }
    return true;
}

namespace {

// Step 1 — return indices into `user_cpus` for n_sched threads placed
// inside the tightest unit available.
//
// The unit is identified by a predicate over `user_cpus` (same cluster,
// same die). For each candidate unit, we count its logical cpus; if any
// has >= n_sched we pick the highest-id unit (tiebreaker: closer to die
// 1 / cluster 3).
//
// "Fill phys round-robin": within the chosen unit, sort entries by
// (phy_cpu_id ASC, hyperthread_id ASC), then pick the first n_sched in
// the order [ph_a ht_0, ph_b ht_0, ..., ph_a ht_1, ph_b ht_1, ...].
// This consumes all ht=0 logical cpus first before doubling onto SMT
// siblings.
template <class GetGroup>
bool try_fit_in_one(
    const std::vector<AicpuLogicalCpu> &user_cpus, int32_t n_sched, GetGroup get_group,
    std::vector<int32_t> &out_sched_indices
) {
    // Build group buckets, indexed by group id; group ids are dense small
    // ints (cluster_id ∈ [0, ~3], die_id ∈ [0, 1]).
    int32_t max_group = -1;
    for (const auto &c : user_cpus)
        max_group = std::max(max_group, get_group(c));
    if (max_group < 0) return false;

    std::vector<std::vector<int32_t>> buckets(max_group + 1);
    for (int32_t i = 0; i < static_cast<int32_t>(user_cpus.size()); ++i) {
        buckets[get_group(user_cpus[i])].push_back(i);
    }

    // Pick the highest-id group with enough logical cpus.
    int32_t chosen = -1;
    for (int32_t g = max_group; g >= 0; --g) {
        if (static_cast<int32_t>(buckets[g].size()) >= n_sched) {
            chosen = g;
            break;
        }
    }
    if (chosen < 0) return false;

    // Within the chosen unit, lay out by (phy_cpu_id ASC, hyperthread_id ASC,
    // cpu_id ASC). This pairs SMT siblings adjacently in the output:
    //   sched 0 = phy_a ht 0
    //   sched 1 = phy_a ht 1   (SMT sibling of sched 0)
    //   sched 2 = phy_b ht 0
    //   sched 3 = phy_b ht 1   (SMT sibling of sched 2)
    // Matches the confirmed user-facing layout (see commit message / the
    // discussion in src/a5/docs/hardware.md "CANN AICPU thread dispatch").
    std::vector<int32_t> ordered = buckets[chosen];
    std::sort(ordered.begin(), ordered.end(), [&](int32_t a, int32_t b) {
        const auto &ca = user_cpus[a];
        const auto &cb = user_cpus[b];
        if (ca.phy_cpu_id != cb.phy_cpu_id) return ca.phy_cpu_id < cb.phy_cpu_id;
        if (ca.hyperthread_id != cb.hyperthread_id) return ca.hyperthread_id < cb.hyperthread_id;
        return ca.cpu_id < cb.cpu_id;
    });

    out_sched_indices.assign(ordered.begin(), ordered.begin() + n_sched);
    return true;
}

// Step 1.3 — spread across dies. We just take the first n_sched cpus from
// `user_cpus` sorted by cpu_id; this guarantees a deterministic result.
void fit_spread(const std::vector<AicpuLogicalCpu> &user_cpus, int32_t n_sched, std::vector<int32_t> &out_indices) {
    out_indices.clear();
    for (int32_t i = 0;
         i < static_cast<int32_t>(user_cpus.size()) && static_cast<int32_t>(out_indices.size()) < n_sched; ++i) {
        out_indices.push_back(i);
    }
}

// Pick the lowest cpu_id from `user_cpus` that satisfies `pred` AND is
// not already in `used`. Returns -1 if no candidate.
template <class Pred>
int32_t
pick_lowest_for_orch(const std::vector<AicpuLogicalCpu> &user_cpus, const std::vector<int32_t> &used, Pred pred) {
    int32_t best = -1;
    for (int32_t i = 0; i < static_cast<int32_t>(user_cpus.size()); ++i) {
        if (std::find(used.begin(), used.end(), i) != used.end()) continue;
        if (!pred(user_cpus[i])) continue;
        if (best < 0 || user_cpus[i].cpu_id < user_cpus[best].cpu_id) best = i;
    }
    return best;
}

}  // namespace

bool compute_allowed_cpus(
    const std::vector<AicpuLogicalCpu> &user_cpus, int32_t n_sched, int32_t n_orch,
    std::vector<int32_t> &out_allowed_cpus
) {
    out_allowed_cpus.clear();
    if (n_sched < 0 || n_orch < 0) return false;
    if (static_cast<int32_t>(user_cpus.size()) < n_sched + n_orch) return false;

    // Step 1 — place sched.
    std::vector<int32_t> sched_indices;
    auto by_cluster = [](const AicpuLogicalCpu &c) {
        return c.cluster_id;
    };
    auto by_die = [](const AicpuLogicalCpu &c) {
        return c.die_id;
    };
    if (n_sched > 0) {
        if (!try_fit_in_one(user_cpus, n_sched, by_cluster, sched_indices)) {
            if (!try_fit_in_one(user_cpus, n_sched, by_die, sched_indices)) {
                fit_spread(user_cpus, n_sched, sched_indices);
            }
        }
        if (static_cast<int32_t>(sched_indices.size()) != n_sched) return false;
    }

    // Step 2 — place orch with priority same-cluster > same-die > spread.
    // We currently expect n_orch == 1 but keep the loop general.
    std::vector<int32_t> orch_indices;
    for (int32_t k = 0; k < n_orch; ++k) {
        std::vector<int32_t> taken = sched_indices;
        taken.insert(taken.end(), orch_indices.begin(), orch_indices.end());

        // Determine the cluster_ids / die_ids occupied by sched so far.
        auto sched_has_cluster = [&](int32_t cid) {
            for (int32_t i : sched_indices)
                if (user_cpus[i].cluster_id == cid) return true;
            return false;
        };
        auto sched_has_die = [&](int32_t did) {
            for (int32_t i : sched_indices)
                if (user_cpus[i].die_id == did) return true;
            return false;
        };

        // 2.1 — same cluster as any sched.
        int32_t pick = pick_lowest_for_orch(user_cpus, taken, [&](const AicpuLogicalCpu &c) {
            return sched_has_cluster(c.cluster_id);
        });
        // 2.2 — fall back to same die.
        if (pick < 0) {
            pick = pick_lowest_for_orch(user_cpus, taken, [&](const AicpuLogicalCpu &c) {
                return sched_has_die(c.die_id);
            });
        }
        // 2.3 — spread (any free cpu).
        if (pick < 0) {
            pick = pick_lowest_for_orch(user_cpus, taken, [](const AicpuLogicalCpu &) {
                return true;
            });
        }
        if (pick < 0) return false;
        orch_indices.push_back(pick);
    }

    // Emit in the canonical [sched..., orch...] order.
    for (int32_t i : sched_indices)
        out_allowed_cpus.push_back(user_cpus[i].cpu_id);
    for (int32_t i : orch_indices)
        out_allowed_cpus.push_back(user_cpus[i].cpu_id);
    return true;
}

namespace {

auto topology_key(const AicpuLogicalCpu &cpu) {
    return std::make_tuple(cpu.die_id, cpu.cluster_id, cpu.phy_cpu_id, cpu.hyperthread_id, cpu.cpu_id);
}

bool validate_cpu_topology(const std::vector<AicpuLogicalCpu> &cpus) {
    std::set<int32_t> cpu_ids;
    std::set<std::pair<int32_t, int32_t>> physical_threads;
    for (const auto &cpu : cpus) {
        if (cpu.cpu_id < 0 || cpu.phy_cpu_id < 0 || cpu.cluster_id < 0 || cpu.die_id < 0) return false;
        if (cpu.hyperthread_id < 0 || cpu.hyperthread_id > 1) return false;
        if (!cpu_ids.insert(cpu.cpu_id).second) return false;
        if (!physical_threads.emplace(cpu.phy_cpu_id, cpu.hyperthread_id).second) return false;
    }
    return true;
}

bool validate_cpu_ids(const std::vector<AicpuLogicalCpu> &cpus) {
    std::set<int32_t> cpu_ids;
    for (const auto &cpu : cpus) {
        if (cpu.cpu_id < 0 || !cpu_ids.insert(cpu.cpu_id).second) return false;
    }
    return true;
}

size_t count_set_bits(uint64_t value) {
    size_t count = 0;
    while (value != 0) {
        value &= value - 1;
        ++count;
    }
    return count;
}

bool cpu_ids_are_covered_by_occupy(const std::vector<AicpuLogicalCpu> &cpus, uint64_t occupy) {
    return std::all_of(cpus.begin(), cpus.end(), [occupy](const AicpuLogicalCpu &cpu) {
        return cpu.cpu_id >= 0 && cpu.cpu_id < 64 && ((occupy >> cpu.cpu_id) & 1ULL) != 0;
    });
}

std::vector<int32_t> clusters_of(const std::vector<AicpuLogicalCpu> &cpus) {
    std::set<int32_t> clusters;
    for (const auto &cpu : cpus)
        clusters.insert(cpu.cluster_id);
    return {clusters.begin(), clusters.end()};
}

bool has_schedulable_smt_pair(const std::vector<AicpuLogicalCpu> &cpus) {
    std::set<std::pair<int32_t, int32_t>> threads;
    for (const auto &cpu : cpus)
        threads.emplace(cpu.phy_cpu_id, cpu.hyperthread_id);
    for (const auto &cpu : cpus) {
        if (threads.count({cpu.phy_cpu_id, 0}) != 0 && threads.count({cpu.phy_cpu_id, 1}) != 0) return true;
    }
    return false;
}

std::vector<AicpuLogicalCpu> sorted_cpus(const std::vector<AicpuLogicalCpu> &cpus) {
    std::vector<AicpuLogicalCpu> ordered = cpus;
    std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
        return topology_key(a) < topology_key(b);
    });
    return ordered;
}

int proximity_rank(const AicpuLogicalCpu &cpu, const AicpuLogicalCpu &orch) {
    if (cpu.cluster_id == orch.cluster_id) return 0;
    if (cpu.die_id == orch.die_id) return 1;
    return 2;
}

// Prefer the die with the most schedulable threads; tie → higher die_id.
int32_t preferred_die_id(const std::vector<AicpuLogicalCpu> &pool) {
    std::unordered_map<int32_t, int32_t> counts;
    for (const auto &cpu : pool)
        ++counts[cpu.die_id];
    int32_t best_die = -1;
    int32_t best_count = -1;
    for (const auto &entry : counts) {
        if (entry.second > best_count || (entry.second == best_count && entry.first > best_die)) {
            best_count = entry.second;
            best_die = entry.first;
        }
    }
    return best_die;
}

bool pick_orchestrator_primary(const std::vector<AicpuLogicalCpu> &pool, AicpuLogicalCpu &out_orch) {
    const int32_t die = preferred_die_id(pool);
    if (die < 0) return false;
    std::vector<AicpuLogicalCpu> primaries;
    for (const auto &cpu : pool) {
        if (cpu.die_id == die && cpu.hyperthread_id == 0) primaries.push_back(cpu);
    }
    if (primaries.empty()) return false;
    out_orch = sorted_cpus(primaries).back();
    return true;
}

}  // namespace

AicpuScenarioType classify_aicpu_scenario(
    uint32_t logical_cpu_count, const std::vector<AicpuLogicalCpu> &all_logical_cpus,
    const std::vector<AicpuLogicalCpu> &os_schedulable_cpus
) {
    // FG/PG1/PG2 come from cluster/die layout only. Scheduler SMT
    // availability is an orthogonal topology property, not a scenario.
    if (!validate_cpu_topology(all_logical_cpus) || !validate_cpu_topology(os_schedulable_cpus) ||
        logical_cpu_count != all_logical_cpus.size()) {
        return AicpuScenarioType::kUnknown;
    }
    for (const auto &candidate : os_schedulable_cpus) {
        auto it = std::find_if(all_logical_cpus.begin(), all_logical_cpus.end(), [&](const auto &cpu) {
            return cpu.cpu_id == candidate.cpu_id && cpu.phy_cpu_id == candidate.phy_cpu_id &&
                   cpu.hyperthread_id == candidate.hyperthread_id;
        });
        if (it == all_logical_cpus.end()) return AicpuScenarioType::kUnknown;
    }
    const auto clusters = clusters_of(all_logical_cpus);
    std::set<int32_t> dies;
    std::unordered_map<int32_t, int32_t> clusters_per_die;
    for (const auto &cpu : all_logical_cpus)
        dies.insert(cpu.die_id);
    for (int32_t cluster : clusters)
        ++clusters_per_die[cluster / 2];

    if (clusters.size() == 4 && dies == std::set<int32_t>({0, 1}) && clusters_per_die[0] == 2 &&
        clusters_per_die[1] == 2) {
        return AicpuScenarioType::kFg;
    }
    if (clusters.size() == 3 && dies == std::set<int32_t>({0, 1}) &&
        ((clusters_per_die[0] == 2 && clusters_per_die[1] == 1) ||
         (clusters_per_die[0] == 1 && clusters_per_die[1] == 2))) {
        return AicpuScenarioType::kPg1;
    }
    if (clusters.size() == 2 && dies == std::set<int32_t>({0, 1}) && clusters_per_die[0] == 1 &&
        clusters_per_die[1] == 1) {
        return AicpuScenarioType::kPg2;
    }
    return AicpuScenarioType::kUnknown;
}

bool compute_unknown_allowed_cpus(
    const AicpuTopology &topology, int32_t active_count, std::vector<int32_t> &out_allowed_cpus
) {
    out_allowed_cpus.clear();
    if (active_count < 2 || active_count > 5 || !validate_cpu_ids(topology.os_schedulable_cpus) ||
        topology.os_schedulable_cpus.size() < static_cast<size_t>(active_count)) {
        return false;
    }
    const bool has_topology = validate_cpu_topology(topology.os_schedulable_cpus);
    auto ordered = has_topology ? sorted_cpus(topology.os_schedulable_cpus) : topology.os_schedulable_cpus;
    if (!has_topology) {
        std::sort(ordered.begin(), ordered.end(), [](const auto &a, const auto &b) {
            return a.cpu_id < b.cpu_id;
        });
    }
    std::vector<int32_t> selected;
    selected.reserve(static_cast<size_t>(active_count));
    for (int32_t i = 0; i < active_count; ++i)
        selected.push_back(ordered[i].cpu_id);
    out_allowed_cpus = std::move(selected);
    return true;
}

bool compute_scenario_allowed_cpus(
    const AicpuTopology &topology, int32_t active_count, std::vector<int32_t> &out_allowed_cpus
) {
    out_allowed_cpus.clear();
    const auto &pool = topology.os_schedulable_cpus;
    if (active_count < 2 || active_count > 5 || !validate_cpu_topology(pool) ||
        pool.size() < static_cast<size_t>(active_count) || topology.scenario_type == AicpuScenarioType::kUnknown ||
        topology.scenario_type == AicpuScenarioType::kNotApplicable) {
        return false;
    }
    const size_t scheduler_count = static_cast<size_t>(active_count - 1);
    auto ordered = sorted_cpus(pool);
    std::vector<int32_t> selected;
    selected.reserve(static_cast<size_t>(active_count));

    if (topology.scenario_type == AicpuScenarioType::kPg2) {
        const AicpuLogicalCpu orch = ordered.back();
        std::vector<AicpuLogicalCpu> sched;
        for (const auto &cpu : ordered)
            if (cpu.cpu_id != orch.cpu_id) sched.push_back(cpu);
        std::sort(sched.begin(), sched.end(), [&](const auto &a, const auto &b) {
            return std::make_tuple(proximity_rank(a, orch), topology_key(a)) <
                   std::make_tuple(proximity_rank(b, orch), topology_key(b));
        });
        if (sched.size() < scheduler_count) return false;
        for (size_t i = 0; i < scheduler_count; ++i)
            selected.push_back(sched[i].cpu_id);
        selected.push_back(orch.cpu_id);
        out_allowed_cpus = std::move(selected);
        return true;
    }

    // FG / PG1: O on the die with the most schedulable threads (ht0 only).
    AicpuLogicalCpu orch{};
    if (!pick_orchestrator_primary(pool, orch)) return false;

    if (topology.scenario_type == AicpuScenarioType::kPg1) {
        // Minimise scheduler SMT sharing: place one primary per physical CPU
        // first, then consume siblings only when the requested scheduler count
        // cannot fit on distinct physical CPUs. Never share O's physical CPU.
        std::vector<AicpuLogicalCpu> sched;
        for (const auto &cpu : pool)
            if (cpu.phy_cpu_id != orch.phy_cpu_id) sched.push_back(cpu);
        std::sort(sched.begin(), sched.end(), [&](const auto &a, const auto &b) {
            return std::make_tuple(a.hyperthread_id, proximity_rank(a, orch), topology_key(a)) <
                   std::make_tuple(b.hyperthread_id, proximity_rank(b, orch), topology_key(b));
        });
        if (sched.size() < scheduler_count) return false;
        for (size_t i = 0; i < scheduler_count; ++i)
            selected.push_back(sched[i].cpu_id);
    } else {
        // FG: use N-1 dedicated physical-core primaries near O.
        std::vector<AicpuLogicalCpu> sched;
        for (const auto &cpu : pool) {
            if (cpu.hyperthread_id == 0 && cpu.phy_cpu_id != orch.phy_cpu_id) sched.push_back(cpu);
        }
        std::sort(sched.begin(), sched.end(), [&](const auto &a, const auto &b) {
            return std::make_tuple(proximity_rank(a, orch), topology_key(a)) <
                   std::make_tuple(proximity_rank(b, orch), topology_key(b));
        });
        if (sched.size() < scheduler_count) return false;
        for (size_t i = 0; i < scheduler_count; ++i)
            selected.push_back(sched[i].cpu_id);
    }
    selected.push_back(orch.cpu_id);
    out_allowed_cpus = std::move(selected);
    return true;
}

bool build_aicpu_launch_plan(
    const AicpuTopology &topology, int32_t requested_active_count, AicpuLaunchPlan &out_plan, std::string &out_error
) {
    out_plan = {};
    out_error.clear();
    out_plan.requested_active_count = requested_active_count;
    const bool automatic = requested_active_count == 0;
    if (!automatic && (requested_active_count < 2 || requested_active_count > PLATFORM_MAX_AICPU_THREADS)) {
        out_error = "requested active count must be in [2, " + std::to_string(PLATFORM_MAX_AICPU_THREADS) + "]";
        return false;
    }

    const size_t stable_reachable = topology.os_schedulable_cpus.size();
    out_plan.stable_reachable_count = static_cast<int32_t>(stable_reachable);
    if (stable_reachable < 2) {
        out_error = "fewer than two stable reachable AICPU CPUs";
        return false;
    }
    if (!topology.device_occupancy.occupy_valid || topology.device_occupancy.occupy == 0) {
        out_error = "device-side AICPU OCCUPY is unavailable";
        return false;
    }
    if (!cpu_ids_are_covered_by_occupy(topology.os_schedulable_cpus, topology.device_occupancy.occupy)) {
        out_error = "stable reachable AICPU CPUs are not a subset of device-side OCCUPY";
        return false;
    }
    const size_t launch_population = count_set_bits(topology.device_occupancy.occupy);
    if (launch_population > static_cast<size_t>(PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)) {
        out_error = "device-side AICPU OCCUPY population " + std::to_string(launch_population) +
                    " exceeds launch capacity " + std::to_string(PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH);
        return false;
    }

    const int32_t desired = automatic ? PLATFORM_DEFAULT_AICPU_THREAD_NUM : requested_active_count;
    const int32_t effective = automatic ? std::min<int32_t>(desired, out_plan.stable_reachable_count) : desired;
    if (!automatic && stable_reachable < static_cast<size_t>(effective)) {
        out_error = "stable reachable AICPU count " + std::to_string(stable_reachable) +
                    " cannot satisfy requested active count " + std::to_string(effective);
        return false;
    }

    out_plan.effective_active_count = effective;
    // The affinity gate only sees every user CPU when CANN launches one
    // representative for every bit in the authoritative device-side OCCUPY
    // mask. CPU_TOPO may be incomplete, so its intersection is used only for
    // active-role selection, never for physical launch coverage.
    out_plan.launch_count = static_cast<int32_t>(launch_population);
    out_plan.warn_stable_reachable_below_default =
        stable_reachable < static_cast<size_t>(PLATFORM_DEFAULT_AICPU_THREAD_NUM);
    out_plan.warn_cpu_topology_unavailable = topology.source != AicpuTopologySource::kDriver;

    bool selected = false;
    if (topology.generic_selection_only) {
        selected = compute_allowed_cpus(topology.os_schedulable_cpus, effective - 1, 1, out_plan.allowed_cpus);
    } else if (topology.scenario_type == AicpuScenarioType::kUnknown) {
        selected = compute_unknown_allowed_cpus(topology, effective, out_plan.allowed_cpus);
    } else {
        selected = compute_scenario_allowed_cpus(topology, effective, out_plan.allowed_cpus);
    }
    if (!selected) {
        out_error = "AICPU selection policy cannot satisfy effective active count " + std::to_string(effective);
        out_plan.allowed_cpus.clear();
        return false;
    }
    return true;
}

const char *aicpu_scenario_name(AicpuScenarioType scenario) {
    switch (scenario) {
    case AicpuScenarioType::kNotApplicable:
        return "NOT_APPLICABLE";
    case AicpuScenarioType::kFg:
        return "FG";
    case AicpuScenarioType::kPg1:
        return "PG1";
    case AicpuScenarioType::kPg2:
        return "PG2";
    case AicpuScenarioType::kUnknown:
        return "UNKNOWN";
    }
    return "UNKNOWN";
}

const char *aicpu_topology_source_name(AicpuTopologySource source) {
    switch (source) {
    case AicpuTopologySource::kDriver:
        return "driver";
    case AicpuTopologySource::kJsonFallback:
        return "json_fallback";
    case AicpuTopologySource::kOccupyFallback:
        return "occupy_fallback";
    }
    return "unknown";
}

namespace {

constexpr char kCpuTopoFallbackRelativePath[] = "src/a5/platform/onboard/host/aicpu_cpu_topo_fallback.json";
constexpr char kCpuTopoFallbackFileName[] = "aicpu_cpu_topo_fallback.json";

std::string json_escape(const std::string &value) {
    constexpr char kHex[] = "0123456789abcdef";
    std::string escaped;
    escaped.reserve(value.size());
    for (const unsigned char c : value) {
        switch (c) {
        case '"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20) {
                escaped += "\\u00";
                escaped += kHex[c >> 4];
                escaped += kHex[c & 0x0f];
            } else {
                escaped += static_cast<char>(c);
            }
            break;
        }
    }
    return escaped;
}

const char *aicpu_selection_policy_name(AicpuSelectionPolicy policy) {
    switch (policy) {
    case AicpuSelectionPolicy::kScenario:
        return "scenario";
    case AicpuSelectionPolicy::kGeneric:
        return "generic";
    case AicpuSelectionPolicy::kSequentialFallback:
        return "sequential_fallback";
    }
    return "unknown";
}

void append_json_int_array(std::ostringstream &out, const std::vector<int32_t> &values) {
    for (size_t i = 0; i < values.size(); ++i) {
        if (i != 0) out << ", ";
        out << values[i];
    }
}

void skip_json_ws(const char *&p) {
    while (*p != '\0' && std::isspace(static_cast<unsigned char>(*p)))
        ++p;
}

bool parse_json_uint(const char *&p, unsigned int &out) {
    skip_json_ws(p);
    if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
    unsigned long v = 0;
    while (std::isdigit(static_cast<unsigned char>(*p))) {
        v = v * 10UL + static_cast<unsigned long>(*p - '0');
        if (v > 0xffffffffUL) return false;
        ++p;
    }
    out = static_cast<unsigned int>(v);
    return true;
}

bool parse_json_string(const char *&p, std::string &out) {
    skip_json_ws(p);
    if (*p != '"') return false;
    ++p;
    out.clear();
    while (*p != '\0' && *p != '"') {
        if (*p == '\\') {
            ++p;
            if (*p == '\0') return false;
        }
        out.push_back(*p++);
    }
    if (*p != '"') return false;
    ++p;
    return true;
}

bool find_json_object_end(const char *start, const char *&end) {
    if (start == nullptr || *start != '{') return false;
    const char *cursor = start;
    int depth = 0;
    do {
        if (*cursor == '\0') return false;
        if (*cursor == '"') {
            std::string unused;
            if (!parse_json_string(cursor, unused)) return false;
            continue;
        }
        if (*cursor == '{') {
            ++depth;
        } else if (*cursor == '}') {
            --depth;
        }
        ++cursor;
    } while (depth > 0);
    end = cursor;
    return depth == 0;
}

// Find `"key"` then `:`, leave `p` on the value.
bool find_json_key(const char *&p, const char *key) {
    const std::string needle = std::string("\"") + key + "\"";
    const char *found = std::strstr(p, needle.c_str());
    if (found == nullptr) return false;
    p = found + needle.size();
    skip_json_ws(p);
    if (*p != ':') return false;
    ++p;
    skip_json_ws(p);
    return true;
}

// Skip one JSON value (number / string / true / false / null). Nested
// objects/arrays are not needed for cpu entries.
bool skip_json_value(const char *&p) {
    skip_json_ws(p);
    if (*p == '"') {
        std::string unused;
        return parse_json_string(p, unused);
    }
    if (std::strncmp(p, "true", 4) == 0) {
        p += 4;
        return true;
    }
    if (std::strncmp(p, "false", 5) == 0) {
        p += 5;
        return true;
    }
    if (std::strncmp(p, "null", 4) == 0) {
        p += 4;
        return true;
    }
    if (*p == '-' || std::isdigit(static_cast<unsigned char>(*p))) {
        if (*p == '-') ++p;
        if (!std::isdigit(static_cast<unsigned char>(*p))) return false;
        while (std::isdigit(static_cast<unsigned char>(*p)))
            ++p;
        return true;
    }
    return false;
}

bool parse_one_cpu_object(const char *&p, DsmiSingleCpu &out) {
    skip_json_ws(p);
    if (*p != '{') return false;
    ++p;
    out = {};
    bool got_cpu = false, got_phy = false, got_ht = false;
    while (true) {
        skip_json_ws(p);
        if (*p == '}') {
            ++p;
            return got_cpu && got_phy && got_ht;
        }
        std::string key;
        if (!parse_json_string(p, key)) return false;
        skip_json_ws(p);
        if (*p != ':') return false;
        ++p;
        skip_json_ws(p);
        // Topology wire fields are uints; role/available and other annotations
        // are documentation-only and must be skippable without failing the load.
        if (key == "cpu_id" || key == "phy_cpu_id" || key == "hyperthread_id") {
            unsigned int val = 0;
            if (!parse_json_uint(p, val) || val > 255U) return false;
            if (key == "cpu_id") {
                out.cpu_id = static_cast<uint8_t>(val);
                got_cpu = true;
            } else if (key == "phy_cpu_id") {
                out.phy_cpu_id = static_cast<uint8_t>(val);
                got_phy = true;
            } else {
                out.hyperthread_id = static_cast<uint8_t>(val);
                got_ht = true;
            }
        } else if (!skip_json_value(p)) {
            return false;
        }
        skip_json_ws(p);
        if (*p == ',') {
            ++p;
            continue;
        }
        if (*p == '}') {
            ++p;
            return got_cpu && got_phy && got_ht;
        }
        return false;
    }
}

const char *host_arch_name() {
#if defined(__x86_64__)
    return "x86_64";
#elif defined(__aarch64__)
    return "aarch64";
#else
    return "unsupported";
#endif
}

bool find_json_key_before(const char *start, const char *end, const char *key, const char *&value) {
    value = start;
    if (!find_json_key(value, key) || value >= end) return false;
    return true;
}

bool parse_cpu_topo_json_for_soc(
    const char *text, const char *soc_name, uint64_t occupy, DsmiCpuTopo &out, bool &out_generic_selection_only
) {
    if (text == nullptr || soc_name == nullptr || soc_name[0] == '\0') return false;
    out_generic_selection_only = false;
    const char *p = text;
    if (!find_json_key(p, "socs")) return false;
    if (*p != '{') return false;
    ++p;

    // Walk soc entries until we find soc_name.
    while (true) {
        skip_json_ws(p);
        if (*p == '}') return false;
        std::string key;
        if (!parse_json_string(p, key)) return false;
        skip_json_ws(p);
        if (*p != ':') return false;
        ++p;
        skip_json_ws(p);
        if (*p != '{') return false;

        if (key != soc_name) {
            const char *obj_end = nullptr;
            if (!find_json_object_end(p, obj_end)) return false;
            p = obj_end;
            skip_json_ws(p);
            if (*p == ',') {
                ++p;
                continue;
            }
            if (*p == '}') return false;
            return false;
        }

        const char *obj_end = nullptr;
        if (!find_json_object_end(p, obj_end)) return false;

        const char *value = nullptr;
        if (find_json_key_before(p, obj_end, "host_arch", value)) {
            std::string required_arch;
            if (!parse_json_string(value, required_arch) || required_arch != host_arch_name()) return false;
        }
        if (find_json_key_before(p, obj_end, "occupy_mask", value)) {
            unsigned int required_occupy = 0;
            if (!parse_json_uint(value, required_occupy) || occupy != required_occupy) return false;
        }
        if (find_json_key_before(p, obj_end, "selection_policy", value)) {
            std::string selection_policy;
            if (!parse_json_string(value, selection_policy) || selection_policy != "generic") return false;
            out_generic_selection_only = true;
        }

        // Parse matching soc object: look for "cpus" array inside.
        const char *obj = p;
        if (!find_json_key_before(obj, obj_end, "cpus", obj)) return false;
        if (*obj != '[') return false;
        ++obj;
        std::memset(&out, 0, sizeof(out));
        uint32_t n = 0;
        while (true) {
            skip_json_ws(obj);
            if (*obj == ']') {
                ++obj;
                out.total_nums = n;
                return n > 0 && n <= kCpuTopoMaxLogical;
            }
            if (n >= kCpuTopoMaxLogical) return false;
            if (!parse_one_cpu_object(obj, out.cpus[n])) return false;
            ++n;
            skip_json_ws(obj);
            if (*obj == ',') {
                ++obj;
                continue;
            }
            if (*obj == ']') {
                ++obj;
                out.total_nums = n;
                return n > 0;
            }
            return false;
        }
    }
}

bool read_cpu_topo_json_text(std::string &out_text) {
    Dl_info info{};
    if (dladdr(reinterpret_cast<const void *>(&read_cpu_topo_json_text), &info) == 0 || info.dli_fname == nullptr) {
        return false;
    }

    std::filesystem::path root = std::filesystem::absolute(info.dli_fname).parent_path();
    {
        std::ifstream input(root / kCpuTopoFallbackFileName);
        if (input) {
            out_text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
            return input.good() || input.eof();
        }
    }
    while (!root.empty()) {
        const std::filesystem::path path = root / kCpuTopoFallbackRelativePath;
        std::ifstream input(path);
        if (input) {
            out_text.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
            return input.good() || input.eof();
        }
        const std::filesystem::path parent = root.parent_path();
        if (parent == root) break;
        root = parent;
    }
    return false;
}

bool fill_dsmi_topo_from_json(
    const char *soc_name, uint64_t occupy, DsmiCpuTopo &out, bool &out_generic_selection_only
) {
    std::string text;
    if (!read_cpu_topo_json_text(text)) return false;
    return parse_cpu_topo_json_for_soc(text.c_str(), soc_name, occupy, out, out_generic_selection_only);
}

void dsmi_topo_to_logical(const DsmiCpuTopo &topo, std::vector<AicpuLogicalCpu> &out) {
    out.clear();
    out.reserve(topo.total_nums);
    for (uint32_t i = 0; i < topo.total_nums; ++i) {
        const DsmiSingleCpu &cpu = topo.cpus[i];
        AicpuLogicalCpu entry{};
        entry.cpu_id = static_cast<int32_t>(cpu.cpu_id);
        entry.phy_cpu_id = static_cast<int32_t>(cpu.phy_cpu_id);
        entry.hyperthread_id = static_cast<int32_t>(cpu.hyperthread_id);
        entry.cluster_id = entry.phy_cpu_id / 2;
        entry.die_id = entry.phy_cpu_id / 4;
        out.push_back(entry);
    }
}

}  // namespace

bool load_cpu_topo_from_json(
    const char *soc_name, uint64_t occupy, std::vector<AicpuLogicalCpu> &out_all_cpus, bool *out_generic_selection_only
) {
    out_all_cpus.clear();
    DsmiCpuTopo topo{};
    bool generic_selection_only = false;
    if (!fill_dsmi_topo_from_json(soc_name, occupy, topo, generic_selection_only)) return false;
    if (out_generic_selection_only != nullptr) *out_generic_selection_only = generic_selection_only;
    for (uint32_t i = 0; i < topo.total_nums; ++i) {
        if (topo.cpus[i].cpu_id >= kCpuTopoMaxLogical) {
            out_all_cpus.clear();
            return false;
        }
    }
    dsmi_topo_to_logical(topo, out_all_cpus);
    std::sort(out_all_cpus.begin(), out_all_cpus.end(), [](const AicpuLogicalCpu &a, const AicpuLogicalCpu &b) {
        return a.cpu_id < b.cpu_id;
    });
    return !out_all_cpus.empty();
}

std::string format_aicpu_topology_json(
    const AicpuTopology &topology, AicpuSelectionPolicy policy, const AicpuLaunchPlan &launch_plan
) {
    std::ostringstream out;
    out << "{\n"
        << "  \"architecture\": \"a5\",\n"
        << "  \"soc_name\": \"" << json_escape(topology.soc_name) << "\",\n"
        << "  \"topology_source\": \"" << aicpu_topology_source_name(topology.source) << "\",\n"
        << "  \"scenario_type\": \"" << aicpu_scenario_name(topology.scenario_type) << "\",\n"
        << "  \"selection_policy\": \"" << aicpu_selection_policy_name(policy) << "\",\n"
        << "  \"scheduler_smt_enabled\": " << (topology.scheduler_smt_enabled ? "true" : "false") << ",\n"
        << "  \"logical_cpu_count\": " << topology.logical_cpu_count << ",\n"
        << "  \"surviving_clusters\": [";
    append_json_int_array(out, topology.surviving_cluster_ids);
    out << "],\n"
        << "  \"device_masks\": {\n"
        << "    \"occupy\": {\"valid\": " << (topology.device_occupancy.occupy_valid ? "true" : "false")
        << ", \"value\": \"0x" << std::hex << topology.device_occupancy.occupy << "\"},\n"
        << "    \"pf_occupy\": {\"valid\": " << (topology.device_occupancy.pf_occupy_valid ? "true" : "false")
        << ", \"value\": \"0x" << topology.device_occupancy.pf_occupy << "\"},\n"
        << "    \"os_sched\": {\"valid\": " << (topology.device_occupancy.os_sched_valid ? "true" : "false")
        << ", \"value\": \"0x" << topology.device_occupancy.os_sched << "\"}\n"
        << std::dec << "  },\n"
        << "  \"os_schedulable_cpus\": [";
    for (size_t i = 0; i < topology.os_schedulable_cpus.size(); ++i) {
        const auto &cpu = topology.os_schedulable_cpus[i];
        if (i != 0) out << ',';
        out << "\n    {\"cpu_id\": " << cpu.cpu_id << ", \"phy_cpu_id\": " << cpu.phy_cpu_id
            << ", \"hyperthread_id\": " << cpu.hyperthread_id << ", \"cluster_id\": " << cpu.cluster_id
            << ", \"die_id\": " << cpu.die_id << '}';
    }
    if (!topology.os_schedulable_cpus.empty()) out << '\n';
    out << "  ],\n"
        << "  \"launch_plan\": {\n"
        << "    \"requested_active_count\": " << launch_plan.requested_active_count << ",\n"
        << "    \"effective_active_count\": " << launch_plan.effective_active_count << ",\n"
        << "    \"stable_reachable_count\": " << launch_plan.stable_reachable_count << ",\n"
        << "    \"launch_count\": " << launch_plan.launch_count << ",\n"
        << "    \"allowed_cpus\": [";
    append_json_int_array(out, launch_plan.allowed_cpus);
    out << "]\n  }\n}\n";
    return out.str();
}

}  // namespace pto::a5
