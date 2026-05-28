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
// query_device_hal — host-side launcher for the device-side HAL query SO.
//
// Pipeline:
//   1. read dispatcher SO bytes + inner SO bytes from disk
//   2. aclInit + aclrtSetDevice
//   3. allocate GM for:
//        - dispatcher SO bytes
//        - inner SO bytes
//        - DeviceArgs struct (160 B)
//        - QueryRequest[]  input
//        - QueryResult[]   output
//   4. memcpy bytes H2D
//   5. rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC, AST_DYN_AICPU, ...) —
//      libaicpu_extend_kernels dlopens dispatcher, runs DynInit which writes
//      inner SO bytes to /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<dev>.so
//   6. aclrtSynchronizeStream
//   7. generate JSON descriptor pointing at preinstall path with two ops
//      (simpler_aicpu_init + simpler_aicpu_query), fingerprint-suffixed opTypes
//   8. rtsBinaryLoadFromFile(json_path, cpuKernelMode=0)
//   9. rtsFuncGetByName("simpler_aicpu_query_<fp>") -> query func handle
//  10. populate DeviceArgs with q_input_addr/count/output_addr (and reset header)
//  11. rtsLaunchCpuKernel(func, stream, kernelArgs, blockDim=1)
//  12. aclrtSynchronizeStream
//  13. D2H copy QueryResult[]
//  14. pretty-print
//
// Reuses the production dispatcher SO (built at build/lib/a2a3/dispatcher/);
// inner SO is our own libaicpu_query.so built next door.

#include <acl/acl.h>
#include <runtime/rt.h>
#include <runtime/runtime/rts/rts_kernel.h>

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
// derived preinstall basename agrees. We inline a minimal Build-ID reader
// here rather than reach into src/common/utils/elf_build_id.h to keep the
// example tool standalone.
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
    // 64-bit ELF header layout (ELFCLASS64 = 2 at offset 4).
    if (data[4] != 2) return false;
    uint64_t e_shoff;
    uint16_t e_shentsize, e_shnum, e_shstrndx;
    std::memcpy(&e_shoff, data + 40, 8);
    std::memcpy(&e_shentsize, data + 58, 2);
    std::memcpy(&e_shnum, data + 60, 2);
    std::memcpy(&e_shstrndx, data + 62, 2);
    if (e_shentsize != 64 || e_shoff + (uint64_t)e_shentsize * e_shnum > len) return false;
    // strtab section header.
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
    // Iterate sections, find one named ".note.gnu.build-id" with sh_type 7 (NOTE).
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
        // Parse note header.
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
struct QueryRequest {
    int32_t module_type;
    int32_t info_type;
};
struct QueryResult {
    int32_t rc;
    int32_t _pad;
    int64_t value;
};
#pragma pack(pop)

const char *kModuleName(int32_t m) {
    switch (m) {
    case 0:
        return "SYSTEM";
    case 1:
        return "AICPU";
    case 2:
        return "CCPU";
    case 3:
        return "DCPU";
    case 4:
        return "AICORE";
    case 5:
        return "TSCPU";
    case 7:
        return "VECTOR_CORE";
    default:
        return "?";
    }
}
const char *kInfoName(int32_t i) {
    switch (i) {
    case 1:
        return "CORE_NUM";
    case 5:
        return "OS_SCHED";
    case 6:
        return "IN_USED";
    case 8:
        return "OCCUPY";
    case 20:
        return "PF_CORE_NUM";
    case 21:
        return "PF_OCCUPY";
    case 41:
        return "DIE_NUM";
    default:
        // Most ENV/VERSION etc constants have specific numbers; just print id.
        static thread_local char buf[16];
        std::snprintf(buf, sizeof(buf), "INFO_%d", i);
        return buf;
    }
}

// Dispatcher's DeviceArgs layout. We populate the bootstrap fields
// (offsets 96 / 104 / 112 / 120 / 128) before bootstrapping and overwrite
// query fields (96 / 104 / 112) afterwards to repoint the same DeviceArgs
// buffer at the request/output. (The dispatcher only reads them during Init;
// the inner SO sees the rewrite at Launch time.)
//   bootstrap:
//     96  aicpu_so_bin (dispatcher bytes)
//     104 aicpu_so_len
//     112 device_id
//     120 inner_so_bin
//     128 inner_so_len
//   query:
//     96  q_input_addr  (overwrites aicpu_so_bin slot, but dispatcher is done)
//     104 q_input_count
//     112 q_output_addr (overwrites device_id slot, but dispatcher is done)
//
// We use ONE DeviceArgs buffer for both phases to keep the example small.
constexpr size_t kBootstrapDispBin = 96;
constexpr size_t kBootstrapDispLen = 104;
constexpr size_t kBootstrapDevId = 112;
constexpr size_t kBootstrapInnerBin = 120;
constexpr size_t kBootstrapInnerLen = 128;
constexpr size_t kQInputAddr = 96;
constexpr size_t kQInputCount = 104;
constexpr size_t kQOutputAddr = 112;

void WriteU64(char *buf, size_t off, uint64_t v) { std::memcpy(buf + off, &v, sizeof(v)); }

std::string MakePreinstallPath(uint64_t fp, int device_id) {
    char buf[256];
    std::snprintf(
        buf, sizeof(buf), "/usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_%016lx_%d.so", fp, device_id
    );
    return buf;
}

std::string MakeJsonDescriptor(uint64_t fp, const std::string &so_basename) {
    char init_op[128], query_op[128];
    std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
    std::snprintf(query_op, sizeof(query_op), "simpler_aicpu_query_%016lx", fp);
    // Match the schema CANN's rtsBinaryLoadFromFile expects (top-level
    // object keyed by opType, each opType maps to an opInfo block).
    // Defaults from AicpuOpConfig in src/common/host/load_aicpu_op.h.
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
    s += entry(query_op, "simpler_aicpu_query");
    s += "\n}\n";
    return s;
}

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <device_id>\n", argv[0]);
        return 1;
    }
    int device_id = std::atoi(argv[1]);

    const char *dispatcher_path_env = std::getenv("SIMPLER_DISPATCHER_SO");
    std::string dispatcher_path = dispatcher_path_env ?
                                      dispatcher_path_env :
                                      "/data/wcwxyai/workspace/simpler/.claude/worktrees/parallel-petting-newt/build/"
                                      "lib/a2a3/dispatcher/libsimpler_aicpu_dispatcher.so";
    const char *inner_path_env = std::getenv("SIMPLER_AICPU_QUERY_SO");
    std::string inner_path =
        inner_path_env ? inner_path_env : "tools/cann-examples/aicpu-device-query/device/build/libaicpu_query.so";

    std::vector<char> dispatcher_bytes;
    std::vector<char> inner_bytes;
    if (!ReadFile(dispatcher_path, &dispatcher_bytes)) return 1;
    if (!ReadFile(inner_path, &inner_bytes)) return 1;

    // Built-in initial query set — exactly what the kernel.cpp probe ran, so
    // we can sanity-check against the on-record results.
    std::vector<QueryRequest> requests = {
        {1 /* AICPU */, 5 /* OS_SCHED */},     {1 /* AICPU */, 8 /* OCCUPY */},     {1 /* AICPU */, 1 /* CORE_NUM */},
        {1 /* AICPU */, 20 /* PF_CORE_NUM */}, {1 /* AICPU */, 21 /* PF_OCCUPY */}, {2 /* CCPU */, 8 /* OCCUPY */},
        {2 /* CCPU */, 1 /* CORE_NUM */},      {3 /* DCPU */, 8 /* OCCUPY */},      {3 /* DCPU */, 1 /* CORE_NUM */},
        {5 /* TSCPU */, 8 /* OCCUPY */},       {5 /* TSCPU */, 1 /* CORE_NUM */},
    };
    std::vector<QueryResult> results(requests.size());

    ACL_CHECK(aclInit(nullptr), "aclInit");
    ACL_CHECK(aclrtSetDevice(device_id), "aclrtSetDevice");
    aclrtStream stream = nullptr;
    ACL_CHECK(aclrtCreateStream(&stream), "aclrtCreateStream");

    // ---- Allocate GM ----
    DevBuf dev_dispatcher, dev_inner, dev_args, dev_qin, dev_qout;
    ACL_CHECK(dev_dispatcher.Alloc(dispatcher_bytes.size()), "dispatcher GM");
    ACL_CHECK(dev_inner.Alloc(inner_bytes.size()), "inner GM");
    ACL_CHECK(dev_args.Alloc(kDeviceArgsBytes), "DeviceArgs GM");
    ACL_CHECK(dev_qin.Alloc(requests.size() * sizeof(QueryRequest)), "qin GM");
    ACL_CHECK(dev_qout.Alloc(results.size() * sizeof(QueryResult)), "qout GM");

    // ---- H2D dispatcher + inner SO + DeviceArgs (bootstrap fields) ----
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

    // ---- Bootstrap via libaicpu_extend_kernels ----
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

    // ---- Fingerprint inner SO + write JSON descriptor ----
    uint64_t fp = fp::Compute(inner_bytes.data(), inner_bytes.size());
    std::string so_basename = "simpler_inner_" + [&] {
        char b[24];
        std::snprintf(b, sizeof(b), "%016lx_%d.so", fp, device_id);
        return std::string(b);
    }();
    std::string preinstall_path = MakePreinstallPath(fp, device_id);
    std::printf("[bootstrap] inner SO at %s (fp=%016lx)\n", preinstall_path.c_str(), fp);

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

    rtFuncHandle init_handle = nullptr, query_handle = nullptr;
    {
        char init_op[128], query_op[128];
        std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
        std::snprintf(query_op, sizeof(query_op), "simpler_aicpu_query_%016lx", fp);
        RT_CHECK(rtsFuncGetByName(binary_handle, init_op, &init_handle), "rtsFuncGetByName init");
        RT_CHECK(rtsFuncGetByName(binary_handle, query_op, &query_handle), "rtsFuncGetByName query");
    }

    // ---- Rewrite DeviceArgs to point at the query I/O buffers ----
    ACL_CHECK(
        aclrtMemcpy(
            dev_qin.ptr, requests.size() * sizeof(QueryRequest), requests.data(),
            requests.size() * sizeof(QueryRequest), ACL_MEMCPY_HOST_TO_DEVICE
        ),
        "H2D QueryRequest"
    );
    std::memset(hostargs, 0, sizeof(hostargs));
    WriteU64(hostargs, kQInputAddr, reinterpret_cast<uint64_t>(dev_qin.ptr));
    WriteU64(hostargs, kQInputCount, (uint64_t)requests.size());
    WriteU64(hostargs, kQOutputAddr, reinterpret_cast<uint64_t>(dev_qout.ptr));
    ACL_CHECK(
        aclrtMemcpy(dev_args.ptr, kDeviceArgsBytes, hostargs, kDeviceArgsBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "H2D DeviceArgs(query)"
    );

    // ---- Launch simpler_aicpu_query ----
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
        RT_CHECK(rtsLaunchCpuKernel(query_handle, 1u, stream, &cfg, &cpu_args), "rtsLaunchCpuKernel query");
    }
    ACL_CHECK(aclrtSynchronizeStream(stream), "sync after query");

    // ---- D2H read results ----
    ACL_CHECK(
        aclrtMemcpy(
            results.data(), results.size() * sizeof(QueryResult), dev_qout.ptr, results.size() * sizeof(QueryResult),
            ACL_MEMCPY_DEVICE_TO_HOST
        ),
        "D2H QueryResult"
    );

    // ---- Pretty-print ----
    std::printf("\n=== device=%d  device-side HAL view (via dispatcher + inner SO) ===\n", device_id);
    for (size_t i = 0; i < requests.size(); ++i) {
        std::printf(
            "  %-12s + %-14s  rc=%d  val=0x%lx (%ld)\n", kModuleName(requests[i].module_type),
            kInfoName(requests[i].info_type), results[i].rc, (long)results[i].value, (long)results[i].value
        );
    }

    ACL_CHECK(aclrtDestroyStream(stream), "destroy stream");
    ACL_CHECK(aclrtResetDevice(device_id), "reset device");
    aclFinalize();
    return 0;
}
