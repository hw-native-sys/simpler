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

// query — host-side CANN device-info CLI.
//
// Each subcommand wraps a small cluster of CANN ACL APIs; treat this file
// as a runnable reference for "how do I ask the driver for X?" from the
// host side. The tool is not shipped in the wheel; it lives under tools/
// for developer use.
//
// Usage:
//   query              full overview (toolkit version + every device)
//   query devices      device count and IDs
//   query device <id>  full per-device dump:
//                        - identification (SoC name, family, detected arch,
//                          phy_chip_id, smp_id, mainboard_id, virtual flag)
//                        - core counts (AIC / AIV / AICPU) and compute geometry
//                          (cube freq + MMAD MNK, AIV SIMD width)
//                        - memory hierarchy (L2 per cluster, L1 / L0A/B/C
//                          per AIC, UB per AIV, HBM per die)
//                      Each line has a short '# comment' explaining what it
//                      means. Buffer sizes come from CANN's platform_config
//                      ini because the matching ACL attrs return 0 on a3.
//   query mem <id>     HBM free / total / used
//   query version      CANN toolkit version (from compiler/version.info)

#include <acl/acl.h>
#include <driver/ascend_hal_base.h>
#include <driver/dsmi_common_interface.h>
#include <runtime/rt.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace {

constexpr double kGiB = 1024.0 * 1024.0 * 1024.0;

// Single point of CANN init / cleanup. aclInit is process-wide; we use
// nullptr (no JSON config) since these queries don't need profiling or
// dump features.
struct AclScope {
    AclScope() {
        aclError rc = aclInit(nullptr);
        if (rc != ACL_SUCCESS) {
            fprintf(stderr, "aclInit failed: %d\n", static_cast<int>(rc));
            std::exit(1);
        }
    }
    ~AclScope() { aclFinalize(); }
};

#define CHECK(call)                                                           \
    do {                                                                      \
        aclError _rc = (call);                                                \
        if (_rc != ACL_SUCCESS) {                                             \
            fprintf(stderr, "%s failed: %d\n", #call, static_cast<int>(_rc)); \
            return 1;                                                         \
        }                                                                     \
    } while (0)

int cmd_devices() {
    uint32_t count = 0;
    CHECK(aclrtGetDeviceCount(&count));
    printf("device_count: %u\n", count);
    for (uint32_t i = 0; i < count; ++i) {
        printf("  device %u\n", i);
    }
    return 0;
}

// CANN toolkit version lives in $ASCEND_HOME_PATH/compiler/version.info as
// a `Version=X.Y.Z` line. We deliberately do NOT call aclrtGetVersion —
// that returns the ACL runtime library version (e.g. 1.17.0), not the
// toolkit version users actually care about.
int cmd_version() {
    const char *home = std::getenv("ASCEND_HOME_PATH");
    if (!home) {
        fprintf(stderr, "ASCEND_HOME_PATH not set\n");
        return 1;
    }
    std::string path = std::string(home) + "/compiler/version.info";
    FILE *fp = std::fopen(path.c_str(), "r");
    if (!fp) {
        fprintf(stderr, "cannot open %s\n", path.c_str());
        return 1;
    }
    char line[256];
    int rc = 1;
    while (std::fgets(line, sizeof(line), fp)) {
        if (std::strncmp(line, "Version=", 8) == 0) {
            size_t n = std::strlen(line);
            while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
                line[--n] = '\0';
            }
            printf("cann_toolkit: %s  (%s)\n", line + 8, home);
            rc = 0;
            break;
        }
    }
    std::fclose(fp);
    if (rc != 0) {
        fprintf(stderr, "Version= line not found in %s\n", path.c_str());
    }
    return rc;
}

int cmd_mem(int device_id) {
    CHECK(aclrtSetDevice(device_id));
    // Device is now bound to this thread — every exit path below must
    // aclrtResetDevice before returning, including the error path.
    size_t free_bytes = 0, total_bytes = 0;
    aclError rc = aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes);
    if (rc != ACL_SUCCESS) {
        fprintf(stderr, "aclrtGetMemInfo failed: %d\n", static_cast<int>(rc));
        aclrtResetDevice(device_id);
        return 1;
    }
    size_t used_bytes = (total_bytes > free_bytes) ? (total_bytes - free_bytes) : 0;
    double used_pct = (total_bytes > 0) ? (100.0 * used_bytes / total_bytes) : 0.0;
    printf(
        "device %d HBM: free=%.2f GiB / total=%.2f GiB  (used=%.2f GiB, %.1f%%)\n", device_id, free_bytes / kGiB,
        total_bytes / kGiB, used_bytes / kGiB, used_pct
    );
    aclrtResetDevice(device_id);
    return 0;
}

// Read a single `key=value` field from the SoC's CANN platform_config ini
// at $ASCEND_HOME_PATH/{aarch64,x86_64}-linux/data/platform_config/<SoC>.ini.
// Returns the raw value string (no trailing newline), or "" if not found.
// The aarch64 subdir is tried first on aarch64 hosts per project codestyle.
//
// This is also the reliable source for buffer sizes (UB / L1 / L0A/B/C) —
// the matching ACL device-attribute query (e.g. ACL_DEV_ATTR_UBUF_PER_VECTOR_CORE)
// returns 0 on CANN 9.0 / a3, so the ini is the authoritative reading.
std::string read_cann_ini_field(const std::string &soc_name, const std::string &key) {
    const char *home = std::getenv("ASCEND_HOME_PATH");
    if (!home) return "";
#if defined(__aarch64__)
    static const char *kSubdirs[] = {"aarch64-linux", "x86_64-linux"};
#elif defined(__x86_64__)
    static const char *kSubdirs[] = {"x86_64-linux", "aarch64-linux"};
#else
    static const char *kSubdirs[] = {"aarch64-linux", "x86_64-linux"};
#endif
    std::string prefix = key + "=";
    for (const char *sub : kSubdirs) {
        std::string path = std::string(home) + "/" + sub + "/data/platform_config/" + soc_name + ".ini";
        FILE *fp = std::fopen(path.c_str(), "r");
        if (!fp) continue;
        char line[256];
        while (std::fgets(line, sizeof(line), fp)) {
            if (std::strncmp(line, prefix.c_str(), prefix.size()) == 0) {
                size_t n = std::strlen(line);
                while (n > 0 && (line[n - 1] == '\n' || line[n - 1] == '\r')) {
                    line[--n] = '\0';
                }
                std::fclose(fp);
                return std::string(line + prefix.size());
            }
        }
        std::fclose(fp);
    }
    return "";
}

std::string read_short_soc_version(const std::string &soc_name) {
    return read_cann_ini_field(soc_name, "Short_SoC_version");
}

// Authoritative mapping. The canonical reference is the per-SoC ini
// listing under CANN's platform_config/ directory — every SoC in a given
// family shares a Short_SoC_version. See
// docs/hardware/chip-architecture.md for the full table.
const char *arch_from_short_soc(const std::string &short_soc) {
    if (short_soc == "Ascend910B") return "a2";
    if (short_soc == "Ascend910_93") return "a3";
    if (short_soc == "Ascend950") return "a5";
    return nullptr;
}

// ---- HAL / DSMI driver-level queries (compile-time linked) ----------------
//
// CANN's public ACL exposes a stable, intentionally-abstracted view. HAL and
// DSMI sit one layer below and expose a richer/rawer view — e.g. AICPU
// physical bitmap, AICore die count, env type, hw version code, host CPU
// hyperthread topology. The driver libraries (libascend_hal.so,
// libdrvdsmi_host.so) are linked at build time; the binary won't load if
// either is missing on the host. The Ascend driver package is required.
//
// HAL field selection follows the calling-side annotations in
// $ASCEND_HOME_PATH/include/driver/ascend_hal_base.h:
//   * Queries tagged "used in device" (PF_CORE_NUM, PF_OCCUPY, OS_SCHED)
//     are device-only — they always return NOT_SUPPORT from host code and are
//     therefore intentionally NOT queried below.
//   * AICORE+DIE_NUM is meaningful on a5 (device = 2 dies) but returns
//     FAILED on a3 (device = 1 die); we attempt it and skip on failure.
//   * CPU_TOPO is undocumented in public CANN; HAL path fails on a3 and a5
//     in our tests, DSMI path works on a5 and fails on a3. We try both and
//     print whichever succeeds.

// CPU_TOPO struct + constants are not in CANN's public headers.
constexpr int kMaxCpuTopoNum = 64;
#ifndef INFO_TYPE_CPU_TOPO
constexpr int INFO_TYPE_CPU_TOPO = 59;
#endif
#ifndef DSMI_SOC_INFO_SUB_CMD_CPU_TOPO
constexpr unsigned int DSMI_SOC_INFO_SUB_CMD_CPU_TOPO = 2;
#endif

struct single_cpu_topology_info {
    unsigned long long cpu_mask;
    unsigned char cpu_id;
    unsigned char is_share;
    unsigned char phy_cpu_id;
    unsigned char hyperthread_id;
};

struct cpu_topology_info {
    unsigned int total_nums;
    single_cpu_topology_info single_cpu_topo[kMaxCpuTopoNum];
};

void print_cpu_topo(const cpu_topology_info &topo) {
    printf(
        "  cpu_topo (logical): %-16u  # AICPU logical CPUs visible to driver (physical + hyperthread)\n",
        topo.total_nums
    );
    unsigned n = topo.total_nums < kMaxCpuTopoNum ? topo.total_nums : kMaxCpuTopoNum;
    for (unsigned i = 0; i < n; ++i) {
        const auto &c = topo.single_cpu_topo[i];
        printf(
            "    cpu_id=%u phy_cpu_id=%u hyperthread_id=%u is_share=%u cpu_mask=0x%llx\n",
            static_cast<unsigned>(c.cpu_id), static_cast<unsigned>(c.phy_cpu_id),
            static_cast<unsigned>(c.hyperthread_id), static_cast<unsigned>(c.is_share), c.cpu_mask
        );
    }
}

void print_hal_extras(uint32_t devu) {
    printf("\n");
    char buf[64];

    // Runtime-API view: rtGetAiCpuCount is the legacy way to ask "how many
    // AICPU can my program schedule"; same value as
    // aclrtGetDeviceInfo(ACL_DEV_ATTR_AICPU_CORE_NUM) on every generation we
    // have data for, but kept here as the literal launcher.cpp counterpart.
    uint32_t rt_aicpu = 0;
    if (rtGetAiCpuCount(&rt_aicpu) == RT_ERROR_NONE) {
        printf("  rt_aicpu_count:     %-16u  # rtGetAiCpuCount (runtime API; matches ACL aicpu_cores)\n", rt_aicpu);
    }

    int64_t val = 0;
    if (halGetDeviceInfo(devu, MODULE_TYPE_SYSTEM, INFO_TYPE_ENV, &val) == 0) {
        const char *name = (val == 0) ? "mini" : (val == 1) ? "EP" : (val == 2) ? "RC" : (val == 3) ? "cloud" : "?";
        std::snprintf(buf, sizeof(buf), "%lld (%s)", static_cast<long long>(val), name);
        printf("  env:                %-16s  # 0=mini 1=EP 2=RC 3=cloud\n", buf);
    }
    if (halGetDeviceInfo(devu, MODULE_TYPE_SYSTEM, INFO_TYPE_VERSION, &val) == 0) {
        std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<long long>(val));
        printf("  hw_version:         %-16s  # hardware revision code\n", buf);
    }
    if (halGetDeviceInfo(devu, MODULE_TYPE_SYSTEM, INFO_TYPE_CORE_NUM, &val) == 0) {
        printf("  ts_num:             %-16lld  # Task Scheduler count\n", static_cast<long long>(val));
    }
    if (halGetDeviceInfo(devu, MODULE_TYPE_SYSTEM, INFO_TYPE_PHY_DIE_ID, &val) == 0) {
        printf(
            "  phy_die_id:         %-16lld  # die identifier (HAL view; differs from ACL phy_chip_id)\n",
            static_cast<long long>(val)
        );
    }
    if (halGetDeviceInfo(devu, MODULE_TYPE_AICPU, INFO_TYPE_OCCUPY, &val) == 0) {
        std::snprintf(buf, sizeof(buf), "0x%llx", static_cast<long long>(val));
        printf("  aicpu_occupy:       %-16s  # bitmap of AICPU cores in HAL view (cf. runtime-visible 6)\n", buf);
    }
    if (halGetDeviceInfo(devu, MODULE_TYPE_AICPU, INFO_TYPE_IN_USED, &val) == 0) {
        printf("  aicpu_in_used:      %-16lld  # AICPU cores currently in use\n", static_cast<long long>(val));
    }
    if (halGetDeviceInfo(devu, MODULE_TYPE_AICORE, INFO_TYPE_CORE_NUM, &val) == 0) {
        printf(
            "  aicore_hal:         %-16lld  # HAL's AICore count (may differ from ACL — see hardware.md)\n",
            static_cast<long long>(val)
        );
    }
    if (halGetDeviceInfo(devu, MODULE_TYPE_VECTOR_CORE, INFO_TYPE_CORE_NUM, &val) == 0) {
        printf("  aiv_hal:            %-16lld  # HAL's AIV count\n", static_cast<long long>(val));
    }
    if (halGetDeviceInfo(devu, MODULE_TYPE_AICORE, INFO_TYPE_DIE_NUM, &val) == 0) {
        printf(
            "  aicore_dies:        %-16lld  # dies per device (a5: 2; a3: query returns FAILED, line omitted)\n",
            static_cast<long long>(val)
        );
    }

    // CPU topology — try HAL first, fall back to DSMI.
    cpu_topology_info topo = {};
    bool got_topo = false;
    {
        int32_t sz = sizeof(topo);
        if (halGetDeviceInfoByBuff(devu, MODULE_TYPE_SYSTEM, INFO_TYPE_CPU_TOPO, &topo, &sz) == 0) got_topo = true;
    }
    if (!got_topo) {
        unsigned int sz = sizeof(topo);
        if (dsmi_get_device_info(devu, DSMI_MAIN_CMD_SOC_INFO, DSMI_SOC_INFO_SUB_CMD_CPU_TOPO, &topo, &sz) == 0) {
            got_topo = true;
        }
    }
    if (got_topo) print_cpu_topo(topo);
}

// Query AIC / AIV core counts. CANN exposes these as per-stream resource
// limits — we open a throwaway stream just to read them. On older CANN
// (pre-8.0) the resource enum may be missing; we report "(unavailable)"
// in that case so the rest of the device summary still prints.
void print_core_limits(aclrtStream stream) {
    uint32_t cube = 0;
    if (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_CUBE_CORE, &cube) == ACL_SUCCESS) {
        printf("  cube_cores (AIC):   %-16u  # runtime-visible (HAL physical may be higher; see hardware.md)\n", cube);
    } else {
        printf("  cube_cores (AIC):   (unavailable)\n");
    }
    uint32_t vector = 0;
    if (aclrtGetStreamResLimit(stream, ACL_RT_DEV_RES_VECTOR_CORE, &vector) == ACL_SUCCESS) {
        printf("  vector_cores (AIV): %-16u  # runtime-visible, 2 AIV per cluster\n", vector);
    } else {
        printf("  vector_cores (AIV): (unavailable)\n");
    }
}

int cmd_device(int device_id) {
    CHECK(aclrtSetDevice(device_id));

    uint32_t devu = static_cast<uint32_t>(device_id);
    const char *soc = aclrtGetSocName();

    printf("device %d:\n", device_id);

    // ---- Chip identification ----
    printf("  soc_name:           %-16s  # CANN's full SoC version string\n", soc ? soc : "(unknown)");
    if (soc) {
        std::string short_soc = read_short_soc_version(soc);
        if (!short_soc.empty()) {
            const char *arch = arch_from_short_soc(short_soc);
            printf("  short_soc_version:  %-16s  # CANN's chip-family identifier\n", short_soc.c_str());
            printf("  detected_arch:      %-16s  # mapped to this repo's arch name\n", arch ? arch : "(unknown)");
        } else {
            printf("  short_soc_version:  (no platform_config ini found)\n");
        }
    }
    int64_t phy_chip = 0;
    if (aclrtGetDeviceInfo(devu, ACL_DEV_ATTR_PHY_CHIP_ID, &phy_chip) == ACL_SUCCESS) {
        printf(
            "  phy_chip_id:        %-16lld  # which physical chip this device is on (a3: devices 2k/2k+1 share it)\n",
            static_cast<long long>(phy_chip)
        );
    }
    int64_t smp = 0;
    if (aclrtGetDeviceInfo(devu, ACL_DEV_ATTR_SMP_ID, &smp) == ACL_SUCCESS) {
        printf(
            "  smp_id:             %-16lld  # AICPU-OS-image group (devices with same smp_id share one OS image)\n",
            static_cast<long long>(smp)
        );
    }
    int64_t mainboard = 0;
    if (aclrtGetDeviceInfo(devu, ACL_DEV_ATTR_MAINBOARD_ID, &mainboard) == ACL_SUCCESS) {
        printf("  mainboard_id:       %-16lld  # server mainboard slot\n", static_cast<long long>(mainboard));
    }
    int64_t is_virtual = 0;
    if (aclrtGetDeviceInfo(devu, ACL_DEV_ATTR_IS_VIRTUAL, &is_virtual) == ACL_SUCCESS) {
        printf("  is_virtual:         %-16s  # compute-power-splitting mode\n", is_virtual ? "yes" : "no");
    }

    aclrtRunMode mode = ACL_HOST;
    if (aclrtGetRunMode(&mode) == ACL_SUCCESS) {
        printf(
            "  run_mode:           %-16s  # this process runs on host or device side\n",
            mode == ACL_HOST ? "host" : "device"
        );
    }

    // ---- Cores ----
    printf("\n");
    aclrtStream stream = nullptr;
    if (aclrtCreateStream(&stream) == ACL_SUCCESS) {
        print_core_limits(stream);
        aclrtDestroyStream(stream);
    }
    int64_t aicpu_cores = 0;
    if (aclrtGetDeviceInfo(devu, ACL_DEV_ATTR_AICPU_CORE_NUM, &aicpu_cores) == ACL_SUCCESS) {
        printf(
            "  aicpu_cores:        %-16lld  # runtime-visible AICPU (HAL OCCUPY mask reveals more; see hardware.md)\n",
            static_cast<long long>(aicpu_cores)
        );
    }

    // ---- Compute (ini-derived) ----
    if (soc) {
        char buf[64];
        std::string cube_freq = read_cann_ini_field(soc, "cube_freq");
        if (!cube_freq.empty()) {
            std::snprintf(buf, sizeof(buf), "%s MHz", cube_freq.c_str());
            printf("  cube_freq:          %-16s  # AIC compute clock\n", buf);
        }
        std::string m = read_cann_ini_field(soc, "cube_m_size");
        std::string n = read_cann_ini_field(soc, "cube_n_size");
        std::string k = read_cann_ini_field(soc, "cube_k_size");
        if (!m.empty() && !n.empty() && !k.empty()) {
            std::snprintf(buf, sizeof(buf), "%sx%sx%s", m.c_str(), n.c_str(), k.c_str());
            printf("  cube_block_mnk:     %-16s  # MMAD per-instruction M×N×K\n", buf);
        }
        std::string vec = read_cann_ini_field(soc, "vec_calc_size");
        if (!vec.empty()) {
            std::snprintf(buf, sizeof(buf), "%s lanes", vec.c_str());
            printf("  vec_calc_size:      %-16s  # AIV SIMD width\n", buf);
        }
    }

    // ---- Per-cluster memory ----
    printf("\n");
    char buf[64];
    int64_t l2 = 0;
    if (aclrtGetDeviceInfo(devu, ACL_DEV_ATTR_L2_CACHE_SIZE, &l2) == ACL_SUCCESS) {
        std::snprintf(buf, sizeof(buf), "%.0f MiB", l2 / (1024.0 * 1024.0));
        printf("  l2_cache:           %-16s  # per AICore cluster, shared by AIC + AIVs\n", buf);
    }

    // ---- Per-unit scratchpads (CANN ini; ACL UB attr returns 0) ----
    if (soc) {
        struct {
            const char *key;
            const char *label;
            const char *note;
        } kFields[] = {
            {"l1_size", "l1_per_aic:        ", "per-AIC scratchpad"},
            {"l0_a_size", "l0a_per_aic:       ", "MMAD matrix-A operand"},
            {"l0_b_size", "l0b_per_aic:       ", "MMAD matrix-B operand"},
            {"l0_c_size", "l0c_per_aic:       ", "MMAD matrix-C accumulator"},
            {"ub_size", "ubuf_per_aiv:      ", "per-AIV vector working set"},
        };
        for (const auto &f : kFields) {
            std::string v = read_cann_ini_field(soc, f.key);
            if (!v.empty()) {
                long long bytes = std::strtoll(v.c_str(), nullptr, 10);
                std::snprintf(buf, sizeof(buf), "%lld KiB", bytes / 1024);
                printf("  %s %-16s  # %s\n", f.label, buf, f.note);
            }
        }
    }

    // ---- Global memory ----
    printf("\n");
    size_t free_bytes = 0, total_bytes = 0;
    if (aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes) == ACL_SUCCESS) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB", total_bytes / kGiB);
        printf("  hbm_total:          %-16s  # per-die HBM (slightly < raw 64 GiB due to driver reserve)\n", buf);
        std::snprintf(buf, sizeof(buf), "%.2f GiB", free_bytes / kGiB);
        printf("  hbm_free:           %-16s  # available right now (sampled)\n", buf);
    }

    // ---- HAL extras (driver-level diagnostic view) ----
    print_hal_extras(devu);

    aclrtResetDevice(device_id);
    return 0;
}

int cmd_all() {
    if (cmd_version() != 0) return 1;
    printf("\n");

    uint32_t count = 0;
    CHECK(aclrtGetDeviceCount(&count));
    printf("device_count: %u\n\n", count);

    for (uint32_t i = 0; i < count; ++i) {
        cmd_device(static_cast<int>(i));
        printf("\n");
    }
    return 0;
}

void usage(const char *prog) {
    fprintf(
        stderr,
        "usage:\n"
        "  %s              full overview (version + every device)\n"
        "  %s devices      device count and IDs\n"
        "  %s device <id>  SoC name, AIC/AIV core counts, HBM total\n"
        "  %s mem <id>     HBM free / total / used\n"
        "  %s version      CANN runtime version\n",
        prog, prog, prog, prog, prog
    );
}

}  // namespace

int main(int argc, char **argv) {
    // `version` is a file-read against ASCEND_HOME_PATH — no driver, no
    // ACL init, so it runs even on a machine without an attached device.
    if (argc == 2 && std::string(argv[1]) == "version") {
        return cmd_version();
    }

    AclScope acl;

    if (argc == 1) {
        return cmd_all();
    }

    const std::string cmd = argv[1];
    if (cmd == "devices") return cmd_devices();
    if (cmd == "device" && argc == 3) return cmd_device(std::atoi(argv[2]));
    if (cmd == "mem" && argc == 3) return cmd_mem(std::atoi(argv[2]));

    usage(argv[0]);
    return 1;
}
