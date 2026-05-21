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
/**
 * AICPU Operation Loader Implementation
 */

#include "load_aicpu_op.h"

#include <unistd.h>

#include <cerrno>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <unordered_set>
#include <vector>

#include "acl/acl.h"
#include "common/unified_log.h"
#include "runtime/rt.h"

namespace host {

namespace {

std::string MakeInnerSoBasename(uint64_t fp) {
    char buf[64];
    snprintf(buf, sizeof(buf), "simpler_inner_%016lx.so", fp);
    return buf;
}

// Per-runtime unique opType — different LoadAicpuOp instances in the same
// process may register the same plain symbol names (simpler_aicpu_init / _exec);
// suffixing with the runtime SO fingerprint keeps CANN's global op registry
// from collapsing distinct registrations.
std::string MakeUniqueOpType(const char *base, uint64_t fp) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s_%016lx", base, fp);
    return buf;
}

uint64_t FingerprintBytes(const void *data, size_t len) {
    constexpr uint64_t kFnvOffset = 0xcbf29ce484222325ULL;
    constexpr uint64_t kFnvPrime = 0x100000001b3ULL;
    uint64_t h = kFnvOffset;
    size_t n = len < 64 ? len : 64;
    auto *p = reinterpret_cast<const unsigned char *>(data);
    for (size_t i = 0; i < n; ++i) {
        h ^= p[i];
        h *= kFnvPrime;
    }
    return h ^ static_cast<uint64_t>(len);
}

bool ReadFileBytes(const std::string &path, std::vector<char> &out) {
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in.is_open()) {
        LOG_ERROR("ReadFileBytes: cannot open %s: %s", path.c_str(), strerror(errno));
        return false;
    }
    std::streamsize len = in.tellg();
    in.seekg(0);
    out.resize(static_cast<size_t>(len));
    if (!in.read(out.data(), len)) {
        LOG_ERROR("ReadFileBytes: read failed for %s", path.c_str());
        return false;
    }
    return true;
}

struct DeviceBuf {
    void *ptr = nullptr;
    ~DeviceBuf() {
        if (ptr != nullptr) (void)aclrtFree(ptr);
    }
    aclError alloc(size_t bytes) { return aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST); }
};

// Process-level cache of inner-SO fingerprints we've already bootstrapped.
// Multiple DeviceRunner instances in the same process share one entry per
// runtime here; same-content uploads short-circuit. No mutex — host-side
// LoadAicpuOp construction is always serialized by the caller (Python GIL or
// sequential per-ChipWorker init), so concurrent insert never happens.
std::unordered_set<uint64_t> &BootstrappedFps() {
    static std::unordered_set<uint64_t> kSet;
    return kSet;
}

}  // namespace

int LoadAicpuOp::BootstrapDispatcher(
    const std::string &dispatcher_so_path, const void *inner_so_data, size_t inner_so_len, rtStream_t stream
) {
    if (inner_so_data == nullptr || inner_so_len == 0) {
        LOG_ERROR("BootstrapDispatcher: empty inner SO bytes");
        return -1;
    }
    inner_fp_ = FingerprintBytes(inner_so_data, inner_so_len);
    inner_so_basename_ = MakeInnerSoBasename(inner_fp_);

    if (BootstrappedFps().count(inner_fp_) > 0) {
        LOG_INFO_V2("BootstrapDispatcher: inner SO fp=%016lx already bootstrapped, skipping", inner_fp_);
        return 0;
    }

    std::vector<char> dispatcher_bytes;
    if (!ReadFileBytes(dispatcher_so_path, dispatcher_bytes)) return -1;
    size_t dispatcher_len = dispatcher_bytes.size();
    const char *inner_bytes = reinterpret_cast<const char *>(inner_so_data);
    size_t inner_len = inner_so_len;

    DeviceBuf dev_dispatcher;
    DeviceBuf dev_inner;
    aclError rc = dev_dispatcher.alloc(dispatcher_len);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMalloc(dispatcher) failed: %d", rc);
        return rc;
    }
    rc = aclrtMemcpy(
        dev_dispatcher.ptr, dispatcher_len, dispatcher_bytes.data(), dispatcher_len, ACL_MEMCPY_HOST_TO_DEVICE
    );
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMemcpy(dispatcher) failed: %d", rc);
        return rc;
    }
    rc = dev_inner.alloc(inner_len);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMalloc(inner) failed: %d", rc);
        return rc;
    }
    rc = aclrtMemcpy(dev_inner.ptr, inner_len, inner_bytes, inner_len, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMemcpy(inner) failed: %d", rc);
        return rc;
    }

    constexpr size_t kDeviceArgsBytes = 160;
    char host_dev_args[kDeviceArgsBytes] = {};
    auto write_qword = [&](size_t offset, uint64_t value) {
        std::memcpy(host_dev_args + offset, &value, sizeof(value));
    };
    write_qword(96, reinterpret_cast<uint64_t>(dev_dispatcher.ptr));
    write_qword(104, static_cast<uint64_t>(dispatcher_len));
    write_qword(112, 0);
    write_qword(120, reinterpret_cast<uint64_t>(dev_inner.ptr));
    write_qword(128, static_cast<uint64_t>(inner_len));

    DeviceBuf dev_args;
    rc = dev_args.alloc(kDeviceArgsBytes);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMalloc(device_args) failed: %d", rc);
        return rc;
    }
    rc = aclrtMemcpy(dev_args.ptr, kDeviceArgsBytes, host_dev_args, kDeviceArgsBytes, ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMemcpy(device_args) failed: %d", rc);
        return rc;
    }

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

    rtError_t rrc = rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, &rt_args, nullptr, stream, 0
    );
    if (rrc != RT_ERROR_NONE) {
        LOG_ERROR("BootstrapDispatcher: rtAicpuKernelLaunchExWithArgs failed: %d", rrc);
        return rrc;
    }
    rc = aclrtSynchronizeStream(stream);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtSynchronizeStream failed: %d", rc);
        return rc;
    }
    LOG_INFO_V0(
        "BootstrapDispatcher: bundled dispatcher (%zu B) + inner SO (%zu B) uploaded; inner SO at %s", dispatcher_len,
        inner_len, inner_so_basename_.c_str()
    );
    BootstrappedFps().insert(inner_fp_);
    return 0;
}

void LoadAicpuOp::Finalize() {
    if (binary_handle_ != nullptr) {
        rtError_t rc = rtsBinaryUnload(binary_handle_);
        if (rc != RT_ERROR_NONE) {
            LOG_WARN("rtsBinaryUnload failed: %d", rc);
        }
        binary_handle_ = nullptr;
    }
    func_handles_.clear();
    inner_fp_ = 0;
    inner_so_basename_.clear();
    if (!json_file_path_.empty()) {
        std::remove(json_file_path_.c_str());
        json_file_path_.clear();
    }
}

LoadAicpuOp::~LoadAicpuOp() { Finalize(); }

bool LoadAicpuOp::GenerateAicpuOpJson(const std::string &json_path, const std::string &kernel_so) {
    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
        LOG_ERROR("Failed to open JSON file for writing: %s", json_path.c_str());
        return false;
    }
    auto make_cfg = [&](const char *symbol_name) {
        AicpuOpConfig c;
        c.opType = MakeUniqueOpType(symbol_name, inner_fp_);
        c.functionName = symbol_name;
        c.kernelSo = kernel_so;
        c.opKernelLib = "AICPUKernel";
        c.userDefined = "False";
        return c;
    };
    std::vector<AicpuOpConfig> op_configs = {
        make_cfg(KernelNames::InitName),
        make_cfg(KernelNames::RunName),
    };
    json_file << "{\n";
    for (size_t i = 0; i < op_configs.size(); ++i) {
        const auto &c = op_configs[i];
        json_file << "  \"" << c.opType << "\": {\n";
        json_file << "    \"opInfo\": {\n";
        json_file << "      \"functionName\": \"" << c.functionName << "\",\n";
        json_file << "      \"kernelSo\": \"" << c.kernelSo << "\",\n";
        json_file << "      \"opKernelLib\": \"" << c.opKernelLib << "\",\n";
        json_file << "      \"computeCost\": \"" << c.computeCost << "\",\n";
        json_file << "      \"engine\": \"" << c.engine << "\",\n";
        json_file << "      \"flagAsync\": \"" << c.flagAsync << "\",\n";
        json_file << "      \"flagPartial\": \"" << c.flagPartial << "\",\n";
        json_file << "      \"userDefined\": \"" << c.userDefined << "\"\n";
        json_file << "    }\n";
        json_file << "  }" << (i < op_configs.size() - 1 ? "," : "") << "\n";
    }
    json_file << "}\n";
    return true;
}

int LoadAicpuOp::Init() {
    if (inner_fp_ == 0) {
        LOG_ERROR("LoadAicpuOp::Init: BootstrapDispatcher must be called first");
        return -1;
    }

    // Per-process JSON path. /tmp is always writable.
    char json_name_buf[128];
    snprintf(
        json_name_buf, sizeof(json_name_buf), "/tmp/simpler_inner_%016lx_%d.json", inner_fp_, static_cast<int>(getpid())
    );
    json_file_path_ = json_name_buf;

    if (!GenerateAicpuOpJson(json_file_path_, inner_so_basename_)) {
        json_file_path_.clear();
        return -1;
    }

    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;

    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option;
    load_config.numOpt = 1;

    LOG_INFO_V2("LoadAicpuOp::Init: JSON=%s inner_basename=%s", json_file_path_.c_str(), inner_so_basename_.c_str());

    rtError_t rc = rtsBinaryLoadFromFile(json_file_path_.c_str(), &load_config, &binary_handle_);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtsBinaryLoadFromFile failed for %s: %d", json_file_path_.c_str(), rc);
        std::remove(json_file_path_.c_str());
        json_file_path_.clear();
        return rc;
    }
    LOG_INFO_V2("LoadAicpuOp: Loaded inner SO via JSON, handle=%p", binary_handle_);

    const char *symbol_names[] = {KernelNames::InitName, KernelNames::RunName};
    for (const char *name : symbol_names) {
        std::string lookup_name = MakeUniqueOpType(name, inner_fp_);
        rtFuncHandle func_handle = nullptr;
        rc = rtsFuncGetByName(binary_handle_, lookup_name.c_str(), &func_handle);
        if (rc != RT_ERROR_NONE) {
            LOG_ERROR("rtsFuncGetByName failed for %s: %d", lookup_name.c_str(), rc);
            return rc;
        }
        func_handles_[name] = func_handle;
        LOG_INFO_V2("LoadAicpuOp: resolved handle for %s (opType=%s): %p", name, lookup_name.c_str(), func_handle);
    }
    return 0;
}

int LoadAicpuOp::AicpuKernelLaunch(rtFuncHandle func_handle, rtStream_t stream, KernelArgs *k_args, int aicpu_num) {
    rtCpuKernelArgs_t cpu_args = {};
    cpu_args.baseArgs.args = k_args;
    cpu_args.baseArgs.argsSize = sizeof(KernelArgs);

    rtKernelLaunchCfg_t kernelLaunchCfg = {nullptr, 0U};
    auto launchKernelAttr = std::make_unique<rtLaunchKernelAttr_t>();
    kernelLaunchCfg.attrs = launchKernelAttr.get();

    rtError_t rc =
        rtsLaunchCpuKernel(func_handle, static_cast<uint32_t>(aicpu_num), stream, &kernelLaunchCfg, &cpu_args);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtsLaunchCpuKernel failed: %d", rc);
        return rc;
    }
    return 0;
}

int LoadAicpuOp::LaunchBuiltInOp(rtStream_t stream, KernelArgs *k_args, int aicpu_num, const std::string &func_name) {
    auto it = func_handles_.find(func_name);
    if (it == func_handles_.end()) {
        LOG_ERROR("Function not found: %s", func_name.c_str());
        return -1;
    }
    return AicpuKernelLaunch(it->second, stream, k_args, aicpu_num);
}

}  // namespace host
