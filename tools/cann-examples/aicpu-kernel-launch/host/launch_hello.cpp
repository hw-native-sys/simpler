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
// launch_hello — the minimum host-side example that walks the AICPU kernel
// launch pipeline end-to-end. Same pipeline as the production runtime; no
// business logic.
//
// Pipeline (Mode A zero-deploy — no sudo, no pre-deployment):
//   1. Read dispatcher SO bytes (from a normal `pip install .` runtime build)
//      and our inner SO bytes (hello_aicpu.so built next door) from disk.
//   2. aclInit + aclrtSetDevice + aclrtCreateStream.
//   3. Allocate GM for: dispatcher bytes, inner bytes, DeviceArgs (160 B),
//      HelloResult buffer (32 B).
//   4. H2D copy bytes + DeviceArgs(bootstrap fields).
//   5. Bootstrap via rtAicpuKernelLaunchExWithArgs(KERNEL_TYPE_AICPU_KFC,
//      libaicpu_extend_kernels.so) — this dlopens the dispatcher inside the
//      AICPU OS process, which writes our inner SO bytes to
//      /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/simpler_inner_<fp>_<dev>.so.
//   6. Fingerprint inner SO + write a JSON descriptor pointing at the
//      preinstall basename, register via rtsBinaryLoadFromFile.
//   7. rtsFuncGetByName for init + run handles.
//   8. Rewrite DeviceArgs so result_addr + input_token point at our buffer.
//   9. rtsLaunchCpuKernel(run_handle) + aclrtSynchronizeStream.
//  10. D2H read HelloResult, verify magic + echoed token + HAL output.
//
// Issue #822 / PR #537 context: this is the Path A flow
// (rtAicpuKernelLaunchExWithArgs + rtsBinaryLoadFromFile / rtsLaunchCpuKernel
// with the dispatcher acting as the SO uploader). Path B
// (kernelType=AICPU_CUSTOM, userDefined=True, runs in cust subprocess) lifts
// the SaveSoFile latch but hits the cust-subprocess L1 cache coherency bug
// described in #822 — see that issue + tools/cann-examples/aicpu-kernel-launch/
// README.md for why this example sticks to Path A.

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

// ELF Build-ID reader, inlined to keep this example standalone. Same logic
// as src/common/utils/elf_build_id.h — must agree with the dispatcher so
// the derived preinstall basename matches.
namespace fp {

// Robust against truncated / malformed inputs. Every offset/length is
// checked against `len` using subtract-from-len arithmetic (never
// `a + b > len`, which can overflow), and every index is checked against
// the corresponding count before use. Required because this example is
// documented as a copy-paste template — the production runtime's ELF
// reader (src/common/utils/elf_build_id.h) does the same checks.
bool ElfBuildId(const char *data, size_t len, uint64_t *out) {
    if (len < 64) return false;
    if (std::memcmp(
            data,
            "\x7f"
            "ELF",
            4
        ) != 0)
        return false;
    if (data[4] != 2) return false;  // ELFCLASS64
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
        uint32_t sh_name;
        uint32_t sh_type;
        uint64_t sh_off;
        uint64_t sh_size;
        std::memcpy(&sh_name, sh + 0, 4);
        std::memcpy(&sh_type, sh + 4, 4);
        std::memcpy(&sh_off, sh + 24, 8);
        std::memcpy(&sh_size, sh + 32, 8);
        if (sh_type != 7) continue;  // SHT_NOTE
        if (sh_name >= strtab_sz) continue;
        // Name must be NUL-terminated within strtab — otherwise strcmp can
        // walk off the end. Hand-rolled bounded scan instead of POSIX
        // strnlen, which isn't in std:: and needs _GNU_SOURCE on some libc.
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

constexpr size_t kDeviceArgsBytes = 160;
constexpr size_t kHelloResultBytes = 32;
constexpr uint64_t kExpectedMagic = 0xDEADBEEFC0FFEE01ULL;

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
    if (size < 0) {
        std::fprintf(stderr, "tellg failed on %s\n", path.c_str());
        return false;
    }
    out->resize(static_cast<size_t>(size));
    f.seekg(0);
    f.read(out->data(), static_cast<std::streamsize>(out->size()));
    // gcount() is the byte count of the most recent unformatted input — if
    // it's short, the file was truncated under us between tellg and read.
    if (f.gcount() != static_cast<std::streamsize>(out->size())) {
        std::fprintf(stderr, "short read on %s (%lld/%zu bytes)\n", path.c_str(), (long long)f.gcount(), out->size());
        return false;
    }
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

// RAII teardown wrappers. They exist so the cleanup happens on every
// exit path (each ACL_CHECK/RT_CHECK is an early return) and in the
// correct LIFO order:
//   DevBufs (aclrtFree)   -- innermost scope, destruct first
//   DeviceContext         -- aclrtDestroyStream + aclrtResetDevice
//   AclScope              -- aclFinalize, last
// Without this, the original implementation called aclrtFree from the
// DevBuf destructors AFTER aclFinalize() returned — undefined behavior
// in the ACL contract.
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

#pragma pack(push, 4)
struct HelloResult {
    uint64_t magic;
    uint64_t echoed_token;
    int32_t hal_rc;
    int32_t _pad;
    int64_t hal_value;
};
#pragma pack(pop)
static_assert(sizeof(HelloResult) == kHelloResultBytes, "HelloResult size drift");

// DeviceArgs offsets. The first 96 B are dispatcher-owned during bootstrap
// (so_bin / so_len / device_id / inner_so_bin / inner_so_len at 96/104/112/
// 120/128). After bootstrap completes the dispatcher is done with the
// struct, so we reuse the buffer by rewriting offsets 96 (result_addr) and
// 104 (input_token) before the run() launch.
constexpr size_t kBootstrapDispBin = 96;
constexpr size_t kBootstrapDispLen = 104;
constexpr size_t kBootstrapDevId = 112;
constexpr size_t kBootstrapInnerBin = 120;
constexpr size_t kBootstrapInnerLen = 128;
constexpr size_t kRunResultAddr = 96;
constexpr size_t kRunInputToken = 104;

void WriteU64(char *buf, size_t off, uint64_t v) { std::memcpy(buf + off, &v, sizeof(v)); }

std::string MakeJsonDescriptor(uint64_t fp, const std::string &so_basename) {
    char init_op[128], run_op[128];
    std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
    std::snprintf(run_op, sizeof(run_op), "simpler_aicpu_run_%016lx", fp);
    // CANN's rtsBinaryLoadFromFile expects a top-level JSON object keyed by
    // opType. Each opType maps to an opInfo block. Defaults match
    // AicpuOpConfig in src/common/host/load_aicpu_op.h — kernelType is
    // implicit (KERNEL_TYPE_AICPU_KFC) since userDefined is False.
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

}  // namespace

int main(int argc, char **argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <device_id>\n", argv[0]);
        std::fprintf(stderr, "env:   SIMPLER_DISPATCHER_SO  path to libsimpler_aicpu_dispatcher.so\n");
        std::fprintf(stderr, "       SIMPLER_HELLO_AICPU_SO path to hello_aicpu.so\n");
        return 1;
    }
    int device_id = std::atoi(argv[1]);

    const char *dispatcher_env = std::getenv("SIMPLER_DISPATCHER_SO");
    if (dispatcher_env == nullptr) {
        std::fprintf(stderr, "SIMPLER_DISPATCHER_SO must point at the dispatcher SO\n");
        std::fprintf(
            stderr, "  (built by `pip install .`, at build/lib/<arch>/dispatcher/libsimpler_aicpu_dispatcher.so)\n"
        );
        return 1;
    }
    const char *inner_env = std::getenv("SIMPLER_HELLO_AICPU_SO");
    std::string inner_path = inner_env ? inner_env : "../device/build/libhello_aicpu.so";

    std::vector<char> dispatcher_bytes, inner_bytes;
    if (!ReadFile(dispatcher_env, &dispatcher_bytes)) return 1;
    if (dispatcher_bytes.empty()) {
        std::fprintf(stderr, "dispatcher SO is empty: %s\n", dispatcher_env);
        return 1;
    }
    if (!ReadFile(inner_path, &inner_bytes)) return 1;
    if (inner_bytes.empty()) {
        std::fprintf(stderr, "inner SO is empty: %s\n", inner_path.c_str());
        return 1;
    }

    // Pick a token whose value is hard to confuse with stale GM (pid alone
    // would be too small; XOR in a constant + a few address bits).
    uint64_t input_token = ((uint64_t)getpid() << 32) ^ 0xA5A5C0DEF00DBEEFULL;

    AclScope acl;
    if (!acl.ok) {
        std::fprintf(stderr, "aclInit failed\n");
        return 1;
    }
    DeviceContext dev(device_id);
    if (!dev.Valid()) {
        std::fprintf(stderr, "aclrtSetDevice / aclrtCreateStream failed for device %d\n", device_id);
        return 1;
    }
    aclrtStream stream = dev.stream;

    // ---- Allocate GM ----
    // DevBufs are declared after AclScope + DeviceContext so LIFO
    // destruction frees GM (aclrtFree) BEFORE aclrtDestroyStream /
    // aclrtResetDevice (in ~DeviceContext) and aclFinalize (in ~AclScope).
    DevBuf dev_dispatcher, dev_inner, dev_args, dev_result;
    ACL_CHECK(dev_dispatcher.Alloc(dispatcher_bytes.size()), "dispatcher GM");
    ACL_CHECK(dev_inner.Alloc(inner_bytes.size()), "inner GM");
    ACL_CHECK(dev_args.Alloc(kDeviceArgsBytes), "DeviceArgs GM");
    ACL_CHECK(dev_result.Alloc(kHelloResultBytes), "HelloResult GM");
    // Zero the result buffer up front so a "kernel never wrote" failure is
    // distinguishable from a "kernel wrote zeros" failure on D2H readback.
    ACL_CHECK(aclrtMemset(dev_result.ptr, kHelloResultBytes, 0, kHelloResultBytes), "memset result");

    // ---- H2D dispatcher bytes + inner SO bytes ----
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

    // ---- DeviceArgs (bootstrap layout) ----
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
    // CANN's libaicpu_extend_kernels.so ships in every install; it dlopens
    // arbitrary AICPU SOs handed to it via DeviceArgs. The dispatcher SO
    // (built by the normal runtime pipeline at build/lib/.../dispatcher/)
    // is what does the inner-SO upload to the preinstall directory.
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

    // ---- Fingerprint inner SO + register via JSON descriptor ----
    uint64_t fp = fp::Compute(inner_bytes.data(), inner_bytes.size());
    char base[64];
    std::snprintf(base, sizeof(base), "simpler_inner_%016lx_%d.so", fp, device_id);
    std::string so_basename = base;
    std::printf("[bootstrap] inner SO landed at /usr/lib64/aicpu_kernels/0/aicpu_kernels_device/%s\n", base);
    std::printf("[bootstrap] fp=%016lx (token=%016lx)\n", fp, input_token);

    // Use mkstemp instead of a predictable /tmp path: dev box is shared,
    // a predictable name is symlink/overwrite-prone. RAII cleanup ensures
    // the descriptor is removed on every exit path (including RT_CHECK
    // early-returns below).
    struct TempFile {
        std::string path;
        ~TempFile() {
            if (!path.empty()) std::remove(path.c_str());
        }
    } json_tmp;
    {
        char tmpl[] = "/tmp/simpler_inner_XXXXXX";
        int fd = mkstemp(tmpl);
        if (fd < 0) {
            std::fprintf(stderr, "mkstemp failed: %s\n", std::strerror(errno));
            return 1;
        }
        json_tmp.path = tmpl;
        std::string json = MakeJsonDescriptor(fp, so_basename);
        ssize_t written = write(fd, json.data(), json.size());
        close(fd);
        if (written != static_cast<ssize_t>(json.size())) {
            std::fprintf(stderr, "write to %s failed (%zd/%zu bytes)\n", json_tmp.path.c_str(), written, json.size());
            return 1;
        }
    }

    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;
    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option;
    load_config.numOpt = 1;
    void *binary_handle = nullptr;
    RT_CHECK(rtsBinaryLoadFromFile(json_tmp.path.c_str(), &load_config, &binary_handle), "rtsBinaryLoadFromFile");
    // json_tmp destructor will remove the file when main returns; no
    // explicit std::remove call here.

    rtFuncHandle init_handle = nullptr, run_handle = nullptr;
    {
        char init_op[128], run_op[128];
        std::snprintf(init_op, sizeof(init_op), "simpler_aicpu_init_%016lx", fp);
        std::snprintf(run_op, sizeof(run_op), "simpler_aicpu_run_%016lx", fp);
        RT_CHECK(rtsFuncGetByName(binary_handle, init_op, &init_handle), "rtsFuncGetByName init");
        RT_CHECK(rtsFuncGetByName(binary_handle, run_op, &run_handle), "rtsFuncGetByName run");
    }
    (void)init_handle;  // init is a no-op; we still resolve it because the
                        // dispatcher requires the symbol to exist.

    // ---- Rewrite DeviceArgs for the run() phase ----
    std::memset(hostargs, 0, sizeof(hostargs));
    WriteU64(hostargs, kRunResultAddr, reinterpret_cast<uint64_t>(dev_result.ptr));
    WriteU64(hostargs, kRunInputToken, input_token);
    ACL_CHECK(
        aclrtMemcpy(dev_args.ptr, kDeviceArgsBytes, hostargs, kDeviceArgsBytes, ACL_MEMCPY_HOST_TO_DEVICE),
        "H2D DeviceArgs(run)"
    );

    // ---- rtsLaunchCpuKernel(run_handle) ----
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

    // ---- D2H + verify ----
    HelloResult got = {};
    ACL_CHECK(
        aclrtMemcpy(&got, sizeof(got), dev_result.ptr, sizeof(got), ACL_MEMCPY_DEVICE_TO_HOST), "D2H HelloResult"
    );

    bool ok = true;
    std::printf("\n=== device=%d  hello_aicpu HelloResult ===\n", device_id);
    std::printf("  magic         = 0x%016lx  %s\n", got.magic, got.magic == kExpectedMagic ? "OK" : "BAD");
    if (got.magic != kExpectedMagic) ok = false;
    std::printf(
        "  echoed_token  = 0x%016lx  %s (expected 0x%016lx)\n", got.echoed_token,
        got.echoed_token == input_token ? "OK" : "BAD", input_token
    );
    if (got.echoed_token != input_token) ok = false;
    std::printf(
        "  hal AICPU+CORE_NUM rc=%d val=%ld  (host-side rtGetAiCpuCount reports 6 on a3/a5)\n", got.hal_rc,
        (long)got.hal_value
    );

    // Stream destroy + device reset + aclFinalize are handled by
    // ~DeviceContext + ~AclScope on return. DevBufs destruct first (LIFO)
    // so GM is freed before the stream/device tear down.
    return ok ? 0 : 2;
}
