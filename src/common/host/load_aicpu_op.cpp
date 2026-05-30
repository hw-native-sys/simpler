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

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <mutex>
#include <set>
#include <utility>
#include <vector>

#include "acl/acl.h"
#include "common/unified_log.h"
#include "runtime/rt.h"
#include "utils/elf_build_id.h"

namespace host {

namespace {

std::string MakeInnerSoBasename(uint64_t fp, int device_id) {
    char buf[64];
    snprintf(buf, sizeof(buf), "simpler_inner_%016lx_%d.so", fp, device_id);
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

// ELF Build-ID-derived 64-bit fingerprint. Dispatcher SO uses the same
// helper on the device side, so both sides agree on the preinstall
// basename without any other channel of communication. See
// src/common/utils/elf_build_id.h for the fallback behavior when the SO
// was linked without a build-id note.
uint64_t FingerprintBytes(const void *data, size_t len) { return simpler::common::utils::elf_build_id_64(data, len); }

struct DeviceBuf {
    void *ptr = nullptr;
    ~DeviceBuf() {
        if (ptr != nullptr) (void)aclrtFree(ptr);
    }
    aclError alloc(size_t bytes) { return aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST); }
};

// Process-level cache of (inner-SO fingerprint, device_id) pairs we've already
// bootstrapped. Keyed by both because the preinstall file is now per-device
// (simpler_inner_<fp>_<device_id>.so) — a single process driving multiple
// devices must bootstrap each device's file even when the fp matches.
// Multiple DeviceRunner instances in the same process share this; same-content
// same-device uploads short-circuit. Guarded by a mutex so that callers
// releasing the Python GIL (e.g. nanobind methods marked
// `nb::call_guard<nb::gil_scoped_release>`) cannot race on the set's
// internals. The lock is uncontended on the steady-state path and only
// touched at DeviceRunner init time, so the overhead is negligible
// compared to keeping the invariant alive in a comment.
std::set<std::pair<uint64_t, int>> &BootstrappedFps() {
    static std::set<std::pair<uint64_t, int>> kSet;
    return kSet;
}
std::mutex &BootstrappedFpsMutex() {
    static std::mutex m;
    return m;
}

}  // namespace

int LoadAicpuOp::BootstrapDispatcher(
    const void *dispatcher_so_data, size_t dispatcher_so_len, const void *inner_so_data, size_t inner_so_len,
    rtStream_t stream, int device_id
) {
    if (dispatcher_so_data == nullptr || dispatcher_so_len == 0) {
        LOG_ERROR("BootstrapDispatcher: empty dispatcher SO bytes");
        return -1;
    }
    if (inner_so_data == nullptr || inner_so_len == 0) {
        LOG_ERROR("BootstrapDispatcher: empty inner SO bytes");
        return -1;
    }
    device_id_ = device_id;
    inner_fp_ = FingerprintBytes(inner_so_data, inner_so_len);
    // Per-device basename: the paired dies of one a2a3 chip (e.g. devices 8/9 =
    // chipId N die0/die1) share the preinstall filesystem. A content-only name
    // would have both dies write/rename/execute one shared
    // simpler_inner_<fp>.so; concurrent bootstrap there corrupts the mmap'd
    // image and faults simpler_aicpu_exec (507018 aicpu exception → chip fault
    // → 507899 cascade). Suffixing device_id isolates each die's file.
    inner_so_basename_ = MakeInnerSoBasename(inner_fp_, device_id_);

    {
        std::lock_guard<std::mutex> lk(BootstrappedFpsMutex());
        if (BootstrappedFps().count({inner_fp_, device_id_}) > 0) {
            LOG_INFO_V2(
                "BootstrapDispatcher: inner SO fp=%016lx dev=%d already bootstrapped, skipping", inner_fp_, device_id_
            );
            return 0;
        }
    }
    // Note: we deliberately drop the lock for the heavy bootstrap work and
    // re-take it for the post-insert below. Two threads racing on the same
    // fingerprint will each perform a bootstrap, which is harmless: CANN's
    // libaicpu_extend_kernels has a one-shot `firstCreatSo_` latch, and the
    // atomic tmp+rename in WriteBytes is idempotent across same-content
    // racers. Holding the lock across the upload would serialize all
    // multi-runtime ChipWorker init in the process — a real regression.

    size_t dispatcher_len = dispatcher_so_len;
    const char *inner_bytes = reinterpret_cast<const char *>(inner_so_data);
    size_t inner_len = inner_so_len;

    DeviceBuf dev_dispatcher;
    DeviceBuf dev_inner;
    aclError rc = dev_dispatcher.alloc(dispatcher_len);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMalloc(dispatcher) failed: %d", rc);
        return rc;
    }
    rc = aclrtMemcpy(dev_dispatcher.ptr, dispatcher_len, dispatcher_so_data, dispatcher_len, ACL_MEMCPY_HOST_TO_DEVICE);
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
    write_qword(112, static_cast<uint64_t>(device_id_));
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
    {
        std::lock_guard<std::mutex> lk(BootstrappedFpsMutex());
        BootstrappedFps().insert({inner_fp_, device_id_});
    }
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
    // Inputs are a closed set: opType / functionName are KernelNames::*
    // constants suffixed with a hex fingerprint, kernelSo is also hex-only,
    // and the remaining fields are hard-coded literals. No characters that
    // require JSON escaping can appear, so manual string concatenation is
    // safe. If you add a field whose value can be user-derived (paths,
    // user-supplied identifiers, etc.), switch to a real JSON serializer
    // before letting it through.
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

    // RAII cleanups: any non-zero return path below unwinds via these guards.
    // .release() flips them off once the corresponding state becomes part of
    // the LoadAicpuOp's steady-state ownership.
    struct JsonGuard {
        std::string &path;
        bool active = true;
        ~JsonGuard() {
            if (active && !path.empty()) {
                std::remove(path.c_str());
                path.clear();
            }
        }
        void release() { active = false; }
    } json_guard{json_file_path_};

    struct BinaryGuard {
        void *&handle;
        bool active = true;
        ~BinaryGuard() {
            if (active && handle != nullptr) {
                (void)rtsBinaryUnload(handle);
                handle = nullptr;
            }
        }
        void release() { active = false; }
    } binary_guard{binary_handle_};

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
        // binary_handle_ stays null; json_guard removes the JSON file.
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
            // binary_guard unloads the partially-registered binary, json_guard
            // removes the JSON file. Symmetric with the rtsBinaryLoadFromFile
            // failure branch above.
            return rc;
        }
        func_handles_[name] = func_handle;
        LOG_INFO_V2("LoadAicpuOp: resolved handle for %s (opType=%s): %p", name, lookup_name.c_str(), func_handle);
    }

    binary_guard.release();
    json_guard.release();
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
