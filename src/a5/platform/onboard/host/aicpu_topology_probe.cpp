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
#include <cstring>
#include <mutex>
#include <unordered_map>

#include "common/unified_log.h"

namespace pto::a5 {

namespace {

// HAL constants — replicated locally so we don't need to pull
// driver/ascend_hal_base.h into the public header.  Values match
// `tools/cann-examples/query/query.cpp`'s usage.
constexpr int32_t kModuleAicpu = 1;
constexpr int32_t kModuleSystem = 0;
constexpr int32_t kInfoOccupy = 8;
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

// dlsym helpers — keep error reporting at WARN, callers fall back.
using HalGetDeviceInfoFn = int (*)(uint64_t deviceId, int32_t moduleType, int32_t infoType, int64_t *value);
using HalGetDeviceInfoByBuffFn =
    int (*)(uint64_t deviceId, int32_t moduleType, int32_t infoType, void *buf, int32_t *size);
using DsmiGetDeviceInfoFn =
    int (*)(uint32_t device_id, unsigned int main_cmd, unsigned int sub_cmd, void *buf, unsigned int *size);

HalGetDeviceInfoFn load_hal_get_device_info() {
    auto fn = reinterpret_cast<HalGetDeviceInfoFn>(dlsym(nullptr, "halGetDeviceInfo"));
    if (fn == nullptr) {
        LOG_WARN("aicpu_topology_probe: halGetDeviceInfo not found via dlsym");
    }
    return fn;
}

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

bool query_occupy(uint32_t device_id, uint64_t &out_mask) {
    auto fn = load_hal_get_device_info();
    if (fn == nullptr) return false;
    int64_t v = 0;
    int rc = fn(static_cast<uint64_t>(device_id), kModuleAicpu, kInfoOccupy, &v);
    if (rc != 0) {
        LOG_WARN("aicpu_topology_probe: halGetDeviceInfo(AICPU,OCCUPY) rc=%d", rc);
        return false;
    }
    out_mask = static_cast<uint64_t>(v);
    return true;
}

bool query_cpu_topo(uint32_t device_id, DsmiCpuTopo &out) {
    std::memset(&out, 0, sizeof(out));
    if (auto fn = load_hal_get_device_info_by_buff(); fn != nullptr) {
        int32_t sz = static_cast<int32_t>(sizeof(out));
        int rc = fn(static_cast<uint64_t>(device_id), kModuleSystem, kInfoCpuTopo, &out, &sz);
        if (rc == 0 && out.total_nums > 0 && out.total_nums <= kCpuTopoMaxLogical) return true;
        LOG_WARN("aicpu_topology_probe: halGetDeviceInfoByBuff(CPU_TOPO) rc=%d total=%u", rc, out.total_nums);
    }
    if (auto fn = load_dsmi_get_device_info(); fn != nullptr) {
        unsigned int sz = static_cast<unsigned int>(sizeof(out));
        int rc = fn(device_id, kDsmiSocInfoMainCmd, kDsmiSocInfoSubCmdCpuTopo, &out, &sz);
        if (rc == 0 && out.total_nums > 0 && out.total_nums <= kCpuTopoMaxLogical) return true;
        LOG_WARN("aicpu_topology_probe: dsmi_get_device_info(CPU_TOPO) rc=%d total=%u", rc, out.total_nums);
    }
    return false;
}

}  // namespace

namespace {

// AICPU topology is a per-device hardware fact — OCCUPY bitmap and DSMI
// CPU_TOPO don't change across runs of the same device, so probe once
// per device_id and cache. Subsequent launches reuse the cached result;
// only `compute_allowed_cpus` re-runs (it's pure logic, microseconds).
std::mutex s_topo_cache_mu;
std::unordered_map<uint32_t, std::vector<AicpuLogicalCpu>> s_topo_cache;

bool probe_aicpu_topology_uncached(uint32_t device_id, std::vector<AicpuLogicalCpu> &out_user_cpus) {
    out_user_cpus.clear();

    uint64_t occupy = 0;
    if (!query_occupy(device_id, occupy)) return false;

    DsmiCpuTopo topo{};
    if (!query_cpu_topo(device_id, topo)) return false;

    for (uint32_t i = 0; i < topo.total_nums; ++i) {
        const DsmiSingleCpu &c = topo.cpus[i];
        // Skip any cpu_id not in the device-side OCCUPY pool. Guard the
        // shift against cpu_id >= 64 (UB in C++) — no a5 SKU is expected
        // to expose more than 64 logical AICPU cpus, but a driver bug or
        // future SKU change shouldn't trip undefined behavior here.
        if (c.cpu_id >= 64 || ((occupy >> c.cpu_id) & 1ULL) == 0) continue;
        AicpuLogicalCpu e{};
        e.cpu_id = static_cast<int32_t>(c.cpu_id);
        e.phy_cpu_id = static_cast<int32_t>(c.phy_cpu_id);
        e.hyperthread_id = static_cast<int32_t>(c.hyperthread_id);
        // a5 cluster mapping: 2 phy/cluster, 2 cluster/die.
        e.cluster_id = e.phy_cpu_id / 2;
        e.die_id = e.phy_cpu_id / 4;
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

    // Cache miss — probe outside the lock so concurrent probes on different
    // devices don't serialize on the driver calls.
    std::vector<AicpuLogicalCpu> probed;
    bool ok = probe_aicpu_topology_uncached(device_id, probed);
    if (!ok) {
        // Don't cache failures — caller (or next launch) can retry.
        out_user_cpus.clear();
        return false;
    }

    LOG_INFO_V0("AICPU topology probed for device %u: %zu user-schedulable cpu_ids (cached)", device_id, probed.size());

    {
        std::lock_guard<std::mutex> lk(s_topo_cache_mu);
        // Last-writer-wins if another thread raced us. The probe is
        // deterministic for a given device so any result is equivalent.
        s_topo_cache[device_id] = probed;
    }
    out_user_cpus = std::move(probed);
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

}  // namespace pto::a5
