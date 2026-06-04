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
//
// aicpu_thread_spread — host launcher that asks CANN to start N AICPU
// threads and reads back what cpu_id each thread landed on.
//
// Usage: aicpu_thread_spread <device_id> <launch_count>
//
// Pipeline is the same as aicpu-device-query (dispatcher bootstrap + inner
// SO via rtsBinaryLoadFromFile + rtsLaunchCpuKernel), only:
//   * launch_count threads instead of 1 (passed as aicpu_num to
//     rtsLaunchCpuKernel)
//   * inner SO is aicpu_thread_spread.so; entry point simpler_aicpu_spread
//   * GM output is SpreadOutput { claim_counter; SpreadRecord[] }

#include <acl/acl.h>
#include <runtime/rt.h>
#include <runtime/runtime/rts/rts_kernel.h>

#include <algorithm>
#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

// ELF Build-ID fingerprinting — must match the dispatcher's hash so the
// derived preinstall basename agrees. Identical to query_device_hal.cpp
// (kept inline to keep this tool standalone).
namespace fp {

bool ElfBuildId(const char *data, size_t len, uint64_t *out) {
    if (len < 64) return false;
    if (std::memcmp(
            data,
            "\x7f"
            "ELF",
            4
        ) != 0)
        return false;
    if (data[4] != 2) return false;
    uint64_t e_shoff;
    uint16_t e_shentsize, e_shnum, e_shstrndx;
    std::memcpy(&e_shoff, data + 40, 8);
    std::memcpy(&e_shentsize, data + 58, 2);
    std::memcpy(&e_shnum, data + 60, 2);
    std::memcpy(&e_shstrndx, data + 62, 2);
    if (e_shentsize != 64 || e_shoff + (uint64_t)e_shentsize * e_shnum > len) return false;
    const char *strtab = nullptr;
    {
        const char *sh = data + e_shoff + (uint64_t)e_shentsize * e_shstrndx;
        uint64_t off;
        uint64_t sz;
        std::memcpy(&off, sh + 24, 8);
        std::memcpy(&sz, sh + 32, 8);
        if (off + sz > len) return false;
        strtab = data + off;
    }
    for (uint16_t i = 0; i < e_shnum; ++i) {
        const char *sh = data + e_shoff + (uint64_t)e_shentsize * i;
        uint32_t sh_name;
        uint32_t sh_type;
        uint64_t sh_off;
        uint64_t sh_size;
        std::memcpy(&sh_name, sh + 0, 4);
        std::memcpy(&sh_type, sh + 4, 4);
        std::memcpy(&sh_off, sh + 24, 8);
        std::memcpy(&sh_size, sh + 32, 8);
        if (sh_type != 7) continue;
        const char *nm = strtab + sh_name;
        if (std::strcmp(nm, ".note.gnu.build-id") != 0) continue;
        if (sh_size < 16 || sh_off + sh_size > len) return false;
        const char *p = data + sh_off;
        uint32_t namesz, descsz, type;
        std::memcpy(&namesz, p + 0, 4);
        std::memcpy(&descsz, p + 4, 4);
        std::memcpy(&type, p + 8, 4);
        if (type != 3 || descsz < 8) return false;
        size_t name_aligned = (namesz + 3u) & ~3u;
        const char *desc = p + 12 + name_aligned;
        std::memcpy(out, desc, 8);
        return true;
    }
    return false;
}

uint64_t Fnv1a(const char *data, size_t len) {
    uint64_t h = 0xcbf29ce484222325ULL;
    for (size_t i = 0; i < len; ++i) {
        h ^= (unsigned char)data[i];
        h *= 0x100000001b3ULL;
    }
    return h;
}

uint64_t Compute(const char *data, size_t len) {
    uint64_t fp;
    if (ElfBuildId(data, len, &fp)) return fp;
    return Fnv1a(data, len);
}

}  // namespace fp

namespace {

constexpr size_t kDeviceArgsBytes = 160;
constexpr uint32_t kMaxSlots = 64;  // upper bound on launch_count we accept

#define ACL_CHECK(call, msg)                                                    \
    do {                                                                        \
        aclError _rc = (call);                                                  \
        if (_rc != ACL_SUCCESS) {                                               \
            std::fprintf(stderr, "%s failed: %d (%s)\n", #call, (int)_rc, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

#define RT_CHECK(call, msg)                                                     \
    do {                                                                        \
        rtError_t _rc = (call);                                                 \
        if (_rc != RT_ERROR_NONE) {                                             \
            std::fprintf(stderr, "%s failed: %d (%s)\n", #call, (int)_rc, msg); \
            return 1;                                                           \
        }                                                                       \
    } while (0)

bool ReadFile(const std::string &path, std::vector<char> *out) {
    std::ifstream f(path, std::ios::binary);
    if (!f.is_open()) {
        std::fprintf(stderr, "open %s failed: %s\n", path.c_str(), std::strerror(errno));
        return false;
    }
    f.seekg(0, std::ios::end);
    out->resize((size_t)f.tellg());
    f.seekg(0);
    f.read(out->data(), out->size());
    return f.good() || f.eof();
}

struct DevBuf {
    void *ptr{nullptr};
    aclError Alloc(size_t n) {
        aclError rc = aclrtMalloc(&ptr, n, ACL_MEM_MALLOC_HUGE_FIRST);
        if (rc != ACL_SUCCESS) ptr = nullptr;
        return rc;
    }
    ~DevBuf() {
        if (ptr) aclrtFree(ptr);
    }
};

#pragma pack(push, 4)
struct SpreadRecord {
    int32_t thread_idx;
    int32_t cpu_id;
};
struct SpreadOutput {
    volatile uint32_t claim_counter;
    uint32_t _pad;
    SpreadRecord records[kMaxSlots];
};
#pragma pack(pop)
static_assert(sizeof(SpreadRecord) == 8, "SpreadRecord size drift");

// DeviceArgs layout — same offsets as aicpu-device-query so the dispatcher
// bootstrap path is unchanged. We just repurpose the post-bootstrap slots:
//   bootstrap:
//     96  aicpu_so_bin  (dispatcher bytes)
//     104 aicpu_so_len
//     112 device_id
//     120 inner_so_bin
//     128 inner_so_len
//   spread (overwrites slots 96, 104 after bootstrap):
//     96  output_addr  (SpreadOutput*)
//     104 max_slots
constexpr size_t kBootstrapDispBin = 96;
constexpr size_t kBootstrapDispLen = 104;
constexpr size_t kBootstrapDevId = 112;
constexpr size_t kBootstrapInnerBin = 120;
constexpr size_t kBootstrapInnerLen = 128;
constexpr size_t kSpreadOutputAddr = 96;
constexpr size_t kSpreadMaxSlots = 104;

void WriteU64(char *buf, size_t off, uint64_t v) { std::memcpy(buf + off, &v, sizeof(v)); }

std::string MakePreinstallPath(uint64_t fp, int device_id) {
    char buf[256];
    std::snprintf(
        buf, sizeof(buf), "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_%016lx_%d.so", fp, device_id
    );
    return buf;
}

std::string MakeJsonDescriptor(uint64_t fp, const std::string &so_basename) {
    char init_op[128], spread_op[128];
    std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
    std::snprintf(spread_op, sizeof(spread_op), "simpler_aicpu_spread_%016lx", fp);
    auto entry = [&](const char *op, const char *fn) {
        std::string s = "  \"";
        s += op;
        s += "\": {\n    \"opInfo\": {\n";
        s += "      \"functionName\": \"";
        s += fn;
        s += "\",\n      \"kernelSo\": \"";
        s += so_basename;
        s += "\",\n      \"opKernelLib\": \"AICPUKernel\",\n";
        s += "      \"computeCost\": \"100\",\n";
        s += "      \"engine\": \"DNN_VM_AICPU\",\n";
        s += "      \"flagAsync\": \"False\",\n";
        s += "      \"flagPartial\": \"False\",\n";
        s += "      \"userDefined\": \"False\"\n";
        s += "    }\n  }";
        return s;
    };
    std::string s = "{\n";
    s += entry(init_op, "simpler_aicpu_init");
    s += ",\n";
    s += entry(spread_op, "simpler_aicpu_spread");
    s += "\n}\n";
    return s;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <device_id> <launch_count>\n", argv[0]);
        return 1;
    }
    int device_id = std::atoi(argv[1]);
    int launch_count = std::atoi(argv[2]);
    if (launch_count < 1 || launch_count > (int)kMaxSlots) {
        std::fprintf(stderr, "launch_count %d out of range [1, %u]\n", launch_count, kMaxSlots);
        return 1;
    }

    const char *dispatcher_path_env = std::getenv("SIMPLER_DISPATCHER_SO");
    if (dispatcher_path_env == nullptr) {
        std::fprintf(stderr, "SIMPLER_DISPATCHER_SO env var required\n");
        return 1;
    }
    std::string dispatcher_path = dispatcher_path_env;
    const char *inner_path_env = std::getenv("SIMPLER_AICPU_SPREAD_SO");
    std::string inner_path = inner_path_env ?
                                 inner_path_env :
                                 "tools/cann-examples/aicpu-thread-spread/device/build/libaicpu_thread_spread.so";

    std::vector<char> dispatcher_bytes;
    std::vector<char> inner_bytes;
    if (!ReadFile(dispatcher_path, &dispatcher_bytes)) return 1;
    if (!ReadFile(inner_path, &inner_bytes)) return 1;

    ACL_CHECK(aclInit(nullptr), "aclInit");
    ACL_CHECK(aclrtSetDevice(device_id), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    DevBuf dev_dispatcher, dev_inner, dev_args, dev_out;
    ACL_CHECK(dev_dispatcher.Alloc(dispatcher_bytes.size()), "dispatcher GM");
    ACL_CHECK(dev_inner.Alloc(inner_bytes.size()), "inner GM");
    ACL_CHECK(dev_args.Alloc(kDeviceArgsBytes), "DeviceArgs GM");
    ACL_CHECK(dev_out.Alloc(sizeof(SpreadOutput)), "SpreadOutput GM");

    ACL_CHECK(
        aclrtMemcpy(
            dev_dispatcher.ptr, dispatcher_bytes.size(), dispatcher_bytes.data(), dispatcher_bytes.size(),
            ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "H2D dispatcher"
    );
    ACL_CHECK(
        aclrtMemcpy(
            dev_inner.ptr, inner_bytes.size(), inner_bytes.data(), inner_bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "H2D inner"
    );
    char hostargs[kDeviceArgsBytes] = {};
    WriteU64(hostargs, kBootstrapDispBin, reinterpret_cast<uint64_t>(dev_dispatcher.ptr));
    WriteU64(hostargs, kBootstrapDispLen, dispatcher_bytes.size());
    WriteU64(hostargs, kBootstrapDevId, (uint64_t)device_id);
    WriteU64(hostargs, kBootstrapInnerBin, reinterpret_cast<uint64_t>(dev_inner.ptr));
    WriteU64(hostargs, kBootstrapInnerLen, inner_bytes.size());
    ACL_CHECK(
        aclrtMemcpy(dev_args.ptr, kDeviceArgsBytes, hostargs, kDeviceArgsBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "H2D DeviceArgs(bootstrap)"
    );

    // Bootstrap dispatcher (writes our inner SO to /usr/lib64/aicpu_kernels/...).
    {
        struct Args {
            struct {
                uint64_t unused[5] = {0};
                uint64_t device_args_ptr = 0;
                uint64_t pad[20] = {0};
            } k_args;
            char kernel_name[32];
            char so_name[32];
            char op_name[32];
        } args = {};
        args.k_args.device_args_ptr = reinterpret_cast<uint64_t>(dev_args.ptr);
        std::strncpy(args.kernel_name, "DynTileFwkKernelServerInit", sizeof(args.kernel_name) - 1);
        std::strncpy(args.so_name, "libaicpu_extend_kernels.so", sizeof(args.so_name) - 1);
        args.op_name[0] = '\0';

        rtAicpuArgsEx_t rt_args = {};
        rt_args.args = &args;
        rt_args.argsSize = sizeof(args);
        rt_args.kernelNameAddrOffset = offsetof(Args, kernel_name);
        rt_args.soNameAddrOffset = offsetof(Args, so_name);

        RT_CHECK(
            rtAicpuKernelLaunchExWithArgs(
                rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, &rt_args, nullptr, stream, 0
            ),
            "rtAicpuKernelLaunchExWithArgs(bootstrap)"
        );
    }
    ACL_CHECK(aclrtSynchronizeStream(stream), "sync after bootstrap");

    // Fingerprint + JSON descriptor.
    uint64_t fp = fp::Compute(inner_bytes.data(), inner_bytes.size());
    std::string so_basename = "simpler_inner_" + [&] {
        char b[24];
        std::snprintf(b, sizeof(b), "%016lx_%d.so", fp, device_id);
        return std::string(b);
    }();
    std::printf("[bootstrap] inner SO at %s (fp=%016lx)\n", MakePreinstallPath(fp, device_id).c_str(), fp);

    char json_path_buf[128];
    std::snprintf(json_path_buf, sizeof(json_path_buf), "/tmp/simpler_inner_%016lx_%d.json", fp, getpid());
    std::string json_path = json_path_buf;
    {
        std::string json = MakeJsonDescriptor(fp, so_basename);
        std::ofstream f(json_path);
        if (!f.is_open()) {
            std::fprintf(stderr, "open %s failed\n", json_path.c_str());
            return 1;
        }
        f << json;
    }

    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option;
    load_config.numOpt = 1;
    void *binary_handle = nullptr;
    RT_CHECK(rtsBinaryLoadFromFile(json_path.c_str(), &load_config, &binary_handle), "rtsBinaryLoadFromFile");
    std::remove(json_path.c_str());

    rtFuncHandle init_handle = nullptr, spread_handle = nullptr;
    {
        char init_op[128], spread_op[128];
        std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
        std::snprintf(spread_op, sizeof(spread_op), "simpler_aicpu_spread_%016lx", fp);
        RT_CHECK(rtsFuncGetByName(binary_handle, init_op, &init_handle), "rtsFuncGetByName init");
        RT_CHECK(rtsFuncGetByName(binary_handle, spread_op, &spread_handle), "rtsFuncGetByName spread");
    }

    // Zero-init the output (claim_counter = 0) and point DeviceArgs at it.
    SpreadOutput host_out_init = {};
    ACL_CHECK(
        aclrtMemcpy(dev_out.ptr, sizeof(SpreadOutput), &host_out_init, sizeof(SpreadOutput), ACL_MEMCPY_HOST_TO_DEVICE),
        "H2D zero SpreadOutput"
    );

    std::memset(hostargs, 0, sizeof(hostargs));
    WriteU64(hostargs, kSpreadOutputAddr, reinterpret_cast<uint64_t>(dev_out.ptr));
    WriteU64(hostargs, kSpreadMaxSlots, (uint64_t)kMaxSlots);
    ACL_CHECK(
        aclrtMemcpy(dev_args.ptr, kDeviceArgsBytes, hostargs, kDeviceArgsBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "H2D DeviceArgs(spread)"
    );

    // Per-launch dance: launch single-threaded init (resets static counter
    // in the AICPU process), then launch N spread threads.
    struct LaunchArgs {
        uint64_t _pad[5] = {0};
        uint64_t device_args_ptr = 0;
        uint64_t reserved[20] = {0};
    } la = {};
    la.device_args_ptr = reinterpret_cast<uint64_t>(dev_args.ptr);

    rtCpuKernelArgs_t cpu_args = {};
    cpu_args.baseArgs.args = &la;
    cpu_args.baseArgs.argsSize = sizeof(la);
    rtLaunchKernelAttr_t attr = {};
    rtKernelLaunchCfg_t cfg = {&attr, 0};

    RT_CHECK(rtsLaunchCpuKernel(init_handle, 1u, stream, &cfg, &cpu_args), "rtsLaunchCpuKernel init");
    ACL_CHECK(aclrtSynchronizeStream(stream), "sync after init");

    // Launch N AICPU threads — aicpu_num = launch_count is the whole point of
    // this tool. CANN distributes the threads across cpu_ids; we want to see
    // which cpu_ids it picks.
    RT_CHECK(
        rtsLaunchCpuKernel(spread_handle, (uint32_t)launch_count, stream, &cfg, &cpu_args), "rtsLaunchCpuKernel spread"
    );
    ACL_CHECK(aclrtSynchronizeStream(stream), "sync after spread");

    SpreadOutput host_out{};
    ACL_CHECK(
        aclrtMemcpy(&host_out, sizeof(SpreadOutput), dev_out.ptr, sizeof(SpreadOutput), ACL_MEMCPY_DEVICE_TO_HOST),
        "D2H SpreadOutput"
    );

    std::printf(
        "\n=== device=%d  launch_count=%d  threads_reported=%u ===\n", device_id, launch_count, host_out.claim_counter
    );
    // claim_counter on the device is updated via racing non-atomic stores
    // (last-writer-wins) and can under-report, so use launch_count to
    // bound the scan — slots [0, launch_count) are deterministically
    // filled by the device-side fetch_add. claim_counter is kept above
    // as a diagnostic hint only.
    uint32_t reported =
        static_cast<uint32_t>(launch_count) < kMaxSlots ? static_cast<uint32_t>(launch_count) : kMaxSlots;
    // Sort by claim order for readability (which is also fetch-add order on the
    // device side, i.e. arrival order at the gate).
    std::vector<SpreadRecord> rs(reported);
    for (uint32_t i = 0; i < reported; ++i)
        rs[i] = host_out.records[i];
    std::printf("  By claim order (thread_idx -> cpu_id):\n");
    for (uint32_t i = 0; i < reported; ++i) {
        std::printf("    idx=%-2d  cpu_id=%d\n", rs[i].thread_idx, rs[i].cpu_id);
    }
    // Histogram of unique cpu_ids hit.
    std::vector<int> cpus;
    for (uint32_t i = 0; i < reported; ++i)
        cpus.push_back(rs[i].cpu_id);
    std::sort(cpus.begin(), cpus.end());
    std::printf("  cpu_id histogram (sorted, may repeat if CANN over-subscribed):\n    ");
    for (size_t i = 0; i < cpus.size(); ++i)
        std::printf("%s%d", (i ? " " : ""), cpus[i]);
    std::printf("\n");

    ACL_CHECK(aclrtDestroyStream(stream), "destroy stream");
    ACL_CHECK(aclrtResetDevice(device_id), "reset device");
    aclFinalize();
    return 0;
}
