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
// launch.cpp — host driver for the AICPU-only MMIO probes.
//
// Same Path-A dispatcher bootstrap as tools/cann-examples/aicpu-kernel-launch
// (dlopen libaicpu_extend_kernels.so, write our inner SO under
// /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/, fingerprint + register
// via rtsBinaryLoadFromFile, rtsLaunchCpuKernel). The only extra step is
// halMemCtl(ADDR_MAP_TYPE_REG_AIC_CTRL) so the AICPU SO knows where the
// register window starts.
//
// Output: pretty-prints MmioProbeResult into a table that reproduces the
// canonical Phase 4 + Phase 12 measurements from
// docs/hardware/mmio-performance.md.

#include <acl/acl.h>
#include <runtime/rt.h>
#include <runtime/runtime/rts/rts_kernel.h>
#include <driver/ascend_hal.h>
#include <driver/ascend_hal_define.h>

#include <cerrno>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <unistd.h>
#include <vector>

#include "../shared/probes_types.h"

// ELF Build-ID fingerprint — same code as
// aicpu-kernel-launch/host/launch_hello.cpp. Falls back to FNV-1a if the SO
// has no .note.gnu.build-id section.
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
    if (e_shentsize != 64) return false;
    if (e_shoff > len) return false;
    if ((uint64_t)e_shentsize * e_shnum > len - e_shoff) return false;
    if (e_shstrndx >= e_shnum) return false;
    const char *strtab = nullptr;
    uint64_t strtab_sz = 0;
    {
        const char *sh = data + e_shoff + (uint64_t)e_shentsize * e_shstrndx;
        uint64_t off;
        std::memcpy(&off, sh + 24, 8);
        std::memcpy(&strtab_sz, sh + 32, 8);
        if (off > len || strtab_sz > len - off) return false;
        strtab = data + off;
    }
    for (uint16_t i = 0; i < e_shnum; ++i) {
        const char *sh = data + e_shoff + (uint64_t)e_shentsize * i;
        uint32_t sh_name, sh_type;
        uint64_t sh_off, sh_size;
        std::memcpy(&sh_name, sh + 0, 4);
        std::memcpy(&sh_type, sh + 4, 4);
        std::memcpy(&sh_off, sh + 24, 8);
        std::memcpy(&sh_size, sh + 32, 8);
        if (sh_type != 7) continue;
        if (sh_name >= strtab_sz) continue;
        size_t max_name_len = strtab_sz - sh_name;
        size_t name_len = 0;
        while (name_len < max_name_len && strtab[sh_name + name_len] != '\0')
            ++name_len;
        if (name_len == max_name_len) continue;
        if (std::strcmp(strtab + sh_name, ".note.gnu.build-id") != 0) continue;
        if (sh_size < 16) return false;
        if (sh_off > len || sh_size > len - sh_off) return false;
        const char *p = data + sh_off;
        uint32_t namesz, descsz, type;
        std::memcpy(&namesz, p + 0, 4);
        std::memcpy(&descsz, p + 4, 4);
        std::memcpy(&type, p + 8, 4);
        if (type != 3 || descsz < 8) return false;
        if (namesz > sh_size) return false;
        size_t name_aligned = (namesz + 3u) & ~3u;
        if (name_aligned > sh_size - 12) return false;
        if (sh_size - 12 - name_aligned < 8) return false;
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
    uint64_t v;
    if (ElfBuildId(data, len, &v)) return v;
    return Fnv1a(data, len);
}
}  // namespace fp

namespace {

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
    std::streamoff size = f.tellg();
    if (size < 0) return false;
    out->resize(static_cast<size_t>(size));
    f.seekg(0);
    f.read(out->data(), static_cast<std::streamsize>(out->size()));
    if (f.gcount() != static_cast<std::streamsize>(out->size())) return false;
    return true;
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

struct AclScope {
    bool ok{false};
    explicit AclScope(const char *config = nullptr) { ok = aclInit(config) == ACL_SUCCESS; }
    ~AclScope() {
        if (ok) aclFinalize();
    }
};

struct DeviceContext {
    int device_id{-1};
    aclrtStream stream{nullptr};
    explicit DeviceContext(int dev) {
        if (aclrtSetDevice(dev) != ACL_SUCCESS) return;
        device_id = dev;
        if (aclrtCreateStream(&stream) != ACL_SUCCESS) stream = nullptr;
    }
    ~DeviceContext() {
        if (stream) aclrtDestroyStream(stream);
        if (device_id >= 0) aclrtResetDevice(device_id);
    }
    bool Valid() const { return device_id >= 0 && stream != nullptr; }
};

// halMemCtl(ADDR_MAP_TYPE_REG_AIC_CTRL) — same call as
// src/{arch}/platform/onboard/host/host_regs.cpp. Returns 0 on failure.
uint64_t QueryAicCtrlBase(int device_id) {
    auto fn = (int (*)(int, void *, size_t, void *, size_t *))dlsym(nullptr, "halMemCtl");
    if (fn == nullptr) {
        std::fprintf(stderr, "halMemCtl not found in symbol table — link libascend_hal\n");
        return 0;
    }
    AddrMapInPara in{};
    in.devid = static_cast<unsigned int>(device_id);
    in.addr_type = ADDR_MAP_TYPE_REG_AIC_CTRL;
    AddrMapOutPara out{};
    int rc = fn(0, &in, sizeof(in), &out, nullptr);
    if (rc != 0) {
        std::fprintf(stderr, "halMemCtl rc=%d\n", rc);
        return 0;
    }
    return out.ptr;
}

constexpr size_t kDeviceArgsBytes = 160;
constexpr size_t kBootstrapDispBin = 96;
constexpr size_t kBootstrapDispLen = 104;
constexpr size_t kBootstrapDevId = 112;
constexpr size_t kBootstrapInnerBin = 120;
constexpr size_t kBootstrapInnerLen = 128;

void WriteU64(char *buf, size_t off, uint64_t v) { std::memcpy(buf + off, &v, sizeof(v)); }

std::string MakeJsonDescriptor(uint64_t fp, const std::string &so_basename) {
    char init_op[128], run_op[128];
    std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
    std::snprintf(run_op, sizeof(run_op), "simpler_aicpu_run_%016lx", fp);
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
    s += entry(run_op, "simpler_aicpu_run");
    s += "\n}\n";
    return s;
}

void PrintResultTable(const MmioProbeResult &r) {
    auto tn = [](uint64_t t) -> uint64_t {
        return t * 20;
    };  // 50 MHz → 20 ns / tick
    std::printf("\n=== aicpu-mmio-probes result ===\n");
    std::printf("  probe_rc           = %d\n", r.probe_rc);
    std::printf("  magic              = 0x%08x  %s\n", r.magic, r.magic == kMmioProbeResultMagic ? "OK" : "BAD");
    std::printf("  AICPU pid          = %lu\n", (unsigned long)r.observed_pid);

    std::printf("\n  --- Phase 4: STR DMB ---\n");
    if (r.str_burst_n > 0) {
        uint64_t per_x100 = (r.str_burst_total_ticks * 100) / r.str_burst_n;
        std::printf(
            "    burst N=%lu  total=%lu ticks (~%lu ns)  per=%lu.%02lu ticks (~%lu ns/STR)\n",
            (unsigned long)r.str_burst_n, (unsigned long)r.str_burst_total_ticks, tn(r.str_burst_total_ticks),
            (unsigned long)(per_x100 / 100), (unsigned long)(per_x100 % 100), tn(per_x100) / 100
        );
    }
    std::printf(
        "    STR+LDR round trip = %lu ticks (~%lu ns)\n", (unsigned long)r.str_lat_round_trip, tn(r.str_lat_round_trip)
    );

    std::printf("\n  --- Phase 12: LDR COND serialization ---\n");
    if (r.ldr_n > 0) {
        uint64_t a_per = (r.ldr_a_total_ticks * 100) / r.ldr_n;
        std::printf(
            "    A: 1 thread, same core    per=%lu.%02lu ticks (~%lu ns/LDR)\n", (unsigned long)(a_per / 100),
            (unsigned long)(a_per % 100), tn(a_per) / 100
        );
        if (r.ldr_b_total_ticks > 0) {
            uint64_t b_per = (r.ldr_b_total_ticks * 100) / r.ldr_n;
            std::printf(
                "    B: 1 thread, rotate %u cores  per=%lu.%02lu ticks (~%lu ns/LDR)\n", r.probed_cores,
                (unsigned long)(b_per / 100), (unsigned long)(b_per % 100), tn(b_per) / 100
            );
        }
        for (uint32_t M = 1; M <= kProbeMaxConcurrentReaders; M++) {
            // Detect "M ran" by checking thread 0 is non-zero.
            if (r.ldr_c_thread_ticks[M - 1][0] == 0) continue;
            std::printf("    C: M=%u threads (each own core)\n", M);
            for (uint32_t t = 0; t < M; t++) {
                uint64_t total = r.ldr_c_thread_ticks[M - 1][t];
                uint64_t per = (total * 100) / r.ldr_n;
                std::printf(
                    "         thread=%u total=%lu ticks (~%lu ns)  per=%lu.%02lu ticks (~%lu ns/LDR)\n", t,
                    (unsigned long)total, tn(total), (unsigned long)(per / 100), (unsigned long)(per % 100),
                    tn(per) / 100
                );
            }
        }
    }
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <device_id> [n_aic_cores (default 3)]\n", argv[0]);
        std::fprintf(stderr, "env:\n");
        std::fprintf(stderr, "  SIMPLER_DISPATCHER_SO  -- path to libsimpler_aicpu_dispatcher.so\n");
        std::fprintf(stderr, "  AICPU_MMIO_PROBES_SO   -- path to libaicpu_mmio_probes.so\n");
        return 1;
    }
    int device_id = std::atoi(argv[1]);
    uint32_t n_aic_cores = (argc >= 3) ? static_cast<uint32_t>(std::atoi(argv[2])) : 3u;

    const char *dispatcher_env = std::getenv("SIMPLER_DISPATCHER_SO");
    if (dispatcher_env == nullptr) {
        std::fprintf(stderr, "SIMPLER_DISPATCHER_SO required\n");
        return 1;
    }
    const char *inner_env = std::getenv("AICPU_MMIO_PROBES_SO");
    std::string inner_path = inner_env ? inner_env : "../device/build/libaicpu_mmio_probes.so";

    std::vector<char> dispatcher_bytes, inner_bytes;
    if (!ReadFile(dispatcher_env, &dispatcher_bytes) || dispatcher_bytes.empty()) return 1;
    if (!ReadFile(inner_path, &inner_bytes) || inner_bytes.empty()) return 1;

    AclScope acl;
    if (!acl.ok) {
        std::fprintf(stderr, "aclInit failed\n");
        return 1;
    }
    DeviceContext dev(device_id);
    if (!dev.Valid()) {
        std::fprintf(stderr, "DeviceContext invalid for device %d\n", device_id);
        return 1;
    }
    aclrtStream stream = dev.stream;

    uint64_t aic_ctrl_base = QueryAicCtrlBase(device_id);
    if (aic_ctrl_base == 0) {
        std::fprintf(stderr, "halMemCtl(AIC_CTRL) returned 0\n");
        return 1;
    }
    std::printf("[probes] aic_ctrl_base = 0x%016lx, n_aic_cores = %u\n", aic_ctrl_base, n_aic_cores);

    DevBuf dev_dispatcher, dev_inner, dev_args, dev_result;
    ACL_CHECK(dev_dispatcher.Alloc(dispatcher_bytes.size()), "dispatcher GM");
    ACL_CHECK(dev_inner.Alloc(inner_bytes.size()), "inner GM");
    ACL_CHECK(dev_args.Alloc(kDeviceArgsBytes), "DeviceArgs GM");
    ACL_CHECK(dev_result.Alloc(sizeof(MmioProbeResult)), "Result GM");
    ACL_CHECK(aclrtMemset(dev_result.ptr, sizeof(MmioProbeResult), 0, sizeof(MmioProbeResult)), "memset result");

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

    // ---- Bootstrap dispatcher ----
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

    uint64_t fp = fp::Compute(inner_bytes.data(), inner_bytes.size());
    char base[64];
    std::snprintf(base, sizeof(base), "simpler_inner_%016lx_%d.so", fp, device_id);
    std::string so_basename = base;
    std::printf(
        "[bootstrap] inner landed at /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/%s (fp=%016lx)\n", base, fp
    );

    struct TempFile {
        std::string path;
        ~TempFile() {
            if (!path.empty()) std::remove(path.c_str());
        }
    } json_tmp;
    {
        char tmpl[] = "/tmp/aicpu_mmio_probes_XXXXXX.json";
        int fd = mkstemps(tmpl, 5);
        if (fd < 0) {
            std::fprintf(stderr, "mkstemps failed: %s\n", std::strerror(errno));
            return 1;
        }
        json_tmp.path = tmpl;
        std::string json = MakeJsonDescriptor(fp, so_basename);
        ssize_t written = write(fd, json.data(), json.size());
        close(fd);
        if (written != static_cast<ssize_t>(json.size())) return 1;
    }

    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option;
    load_config.numOpt = 1;
    void *binary_handle = nullptr;
    RT_CHECK(rtsBinaryLoadFromFile(json_tmp.path.c_str(), &load_config, &binary_handle), "rtsBinaryLoadFromFile");

    rtFuncHandle init_handle = nullptr, run_handle = nullptr;
    {
        char init_op[128], run_op[128];
        std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
        std::snprintf(run_op, sizeof(run_op), "simpler_aicpu_run_%016lx", fp);
        RT_CHECK(rtsFuncGetByName(binary_handle, init_op, &init_handle), "rtsFuncGetByName init");
        RT_CHECK(rtsFuncGetByName(binary_handle, run_op, &run_handle), "rtsFuncGetByName run");
    }
    (void)init_handle;

    // ---- Rewrite DeviceArgs for run() ----
    MmioProbeDeviceArgs run_args = {};
    run_args.result_addr = reinterpret_cast<uint64_t>(dev_result.ptr);
    run_args.input_token = ((uint64_t)getpid() << 32) ^ 0xC0DEDEC0AFAF1234ULL;
    run_args.aic_ctrl_reg_base = aic_ctrl_base;
    run_args.n_aic_cores_available = n_aic_cores;
    ACL_CHECK(
        aclrtMemcpy(dev_args.ptr, sizeof(run_args), &run_args, sizeof(run_args), ACL_MEMCPY_HOST_TO_DEVICE),
        "H2D DeviceArgs(run)"
    );

    // ---- Launch the probes ----
    {
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
        RT_CHECK(rtsLaunchCpuKernel(run_handle, 1u, stream, &cfg, &cpu_args), "rtsLaunchCpuKernel run");
    }
    ACL_CHECK(aclrtSynchronizeStream(stream), "sync after run");

    MmioProbeResult result = {};
    ACL_CHECK(
        aclrtMemcpy(&result, sizeof(result), dev_result.ptr, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST), "D2H result"
    );
    PrintResultTable(result);

    bool ok = (result.magic == kMmioProbeResultMagic) && (result.probe_rc == 0);
    return ok ? 0 : 2;
}
