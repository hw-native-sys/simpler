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
#include "aicpu_loader/finalize_args.h"

#include <unistd.h>

#include <cstddef>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <memory>
#include <vector>

#include "acl/acl.h"
#include "aicpu_loader/bootstrap_protocol.h"
#include "common/unified_log.h"
#include "runtime/rt.h"
#include "utils/elf_build_id.h"
#include "runtime_c_api.h"

namespace host {

namespace {

std::string MakeInnerSoBasename(uint64_t fp, int device_id) {
    char buf[64];
    snprintf(buf, sizeof(buf), "simpler_inner_%016lx_%d.so", fp, device_id);
    return buf;
}

// Per-runtime unique opType — different LoadAicpuOp instances in the same
// process may register the same plain symbol name (simpler_aicpu_exec);
// suffixing with the runtime SO fingerprint keeps CANN's global op registry
// from collapsing distinct registrations.
std::string MakeUniqueOpType(const char *base, uint64_t fp, int device_id) {
    char buf[128];
    snprintf(buf, sizeof(buf), "%s_%016lx_%d", base, fp, device_id);
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

}  // namespace

int LoadAicpuOp::BootstrapDispatcher(
    const void *dispatcher_so_data, size_t dispatcher_so_len, const void *inner_so_data, size_t inner_so_len,
    rtStream_t stream, int device_id
) {
    if (has_live_resources()) {
        LOG_ERROR("BootstrapDispatcher: previous resources must be finalized before another bootstrap");
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    bootstrap_complete_ = false;
    if (dispatcher_so_data == nullptr || dispatcher_so_len == 0) {
        LOG_ERROR("BootstrapDispatcher: empty dispatcher SO bytes");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (inner_so_data == nullptr || inner_so_len == 0) {
        LOG_ERROR("BootstrapDispatcher: empty inner SO bytes");
        return PTO_RUNTIME_ERR_INTERNAL;
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

    simpler_dispatcher::BootstrapResult result;
    DeviceBuf dev_result;
    rc = dev_result.alloc(sizeof(result));
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMalloc(result) failed: %d", rc);
        return rc;
    }
    rc = aclrtMemcpy(dev_result.ptr, sizeof(result), &result, sizeof(result), ACL_MEMCPY_HOST_TO_DEVICE);
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMemcpy(result initialization) failed: %d", rc);
        return rc;
    }

    simpler_dispatcher::BootstrapDeviceArgs host_dev_args;
    host_dev_args.aicpu_so_bin = reinterpret_cast<uint64_t>(dev_dispatcher.ptr);
    host_dev_args.aicpu_so_len = dispatcher_len;
    host_dev_args.device_id = static_cast<uint64_t>(device_id_);
    host_dev_args.inner_so_bin = reinterpret_cast<uint64_t>(dev_inner.ptr);
    host_dev_args.inner_so_len = inner_len;
    host_dev_args.result_addr = reinterpret_cast<uint64_t>(dev_result.ptr);

    DeviceBuf dev_args;
    rc = dev_args.alloc(sizeof(host_dev_args));
    if (rc != ACL_SUCCESS) {
        LOG_ERROR("BootstrapDispatcher: aclrtMalloc(device_args) failed: %d", rc);
        return rc;
    }
    rc = aclrtMemcpy(
        dev_args.ptr, sizeof(host_dev_args), &host_dev_args, sizeof(host_dev_args), ACL_MEMCPY_HOST_TO_DEVICE
    );
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

    // The task may still read all four buffers after an ambiguous enqueue or
    // wait/copy error. Transfer ownership before submitting, and retain until
    // stream completion and a valid result acknowledge this same request.
    bootstrap_buffers_ = {dev_dispatcher.ptr, dev_inner.ptr, dev_args.ptr, dev_result.ptr};
    bootstrap_inner_len_ = inner_len;
    bootstrap_acknowledged_ = bootstrap_upload_ok_ = false;
    dev_dispatcher.ptr = dev_inner.ptr = dev_args.ptr = dev_result.ptr = nullptr;
    bootstrap_stream_ = stream;
    const rtError_t rrc = rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", 1, &rt_args, nullptr, stream, 0
    );
    if (rrc != RT_ERROR_NONE) {
        LOG_ERROR("BootstrapDispatcher: launch failed: %d; retaining buffers for explicit close", rrc);
        return rrc;
    }
    rc = FinishBootstrap(true);
    if (rc != 0) return rc;
    LOG_INFO(
        "BootstrapDispatcher: bundled dispatcher (%zu B) + inner SO (%zu B) uploaded; inner SO at %s", dispatcher_len,
        inner_len, inner_so_basename_.c_str()
    );
    return 0;
}

int LoadAicpuOp::FinishBootstrap(bool require_success) {
    if (bootstrap_buffers_.empty()) return 0;
    if (!bootstrap_acknowledged_) {
        const int wait_rc = aclrtSynchronizeStream(bootstrap_stream_);
        if (wait_rc != 0) return wait_rc;
        simpler_dispatcher::BootstrapResult result;
        const int copy_rc =
            aclrtMemcpy(&result, sizeof(result), bootstrap_buffers_.back(), sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
        if (copy_rc != 0) {
            LOG_ERROR("BootstrapDispatcher: reading upload result failed: %d", copy_rc);
            return copy_rc;
        }
        using simpler_dispatcher::BootstrapStage;
        using simpler_dispatcher::BootstrapStatus;
        const bool complete = result.stage == BootstrapStage::Complete && result.written_bytes == bootstrap_inner_len_;
        const bool cache_hit = result.stage == BootstrapStage::CacheHit &&
                               (result.written_bytes == 0 || result.written_bytes == bootstrap_inner_len_);
        const bool success = result.status == BootstrapStatus::Success && (complete || cache_hit) &&
                             result.error_number == 0 && result.context_error == 0;
        const bool failure =
            result.status == BootstrapStatus::Failed && result.stage >= BootstrapStage::Context &&
            (result.stage <= BootstrapStage::Rename || result.stage == BootstrapStage::Verify ||
             result.stage == BootstrapStage::Link || result.stage == BootstrapStage::Unlink) &&
            (result.error_number > 0 || (result.stage == BootstrapStage::Context && result.context_error != 0));
        if (result.magic != simpler_dispatcher::kBootstrapResultMagic ||
            result.version != simpler_dispatcher::kBootstrapResultVersion || result.device_id != device_id_ ||
            result.requested_bytes != bootstrap_inner_len_ || result.written_bytes > bootstrap_inner_len_ ||
            std::memchr(result.path, '\0', sizeof(result.path)) == nullptr || (!success && !failure)) {
            LOG_ERROR("BootstrapDispatcher: missing or invalid upload result (device=%d)", device_id_);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (failure) {
            LOG_ERROR(
                "BootstrapDispatcher: SO cache failed: stage=%s errno=%d (%s) device=%d path=%s "
                "requested=%llu written=%llu context_rc=%d",
                simpler_dispatcher::bootstrap_stage_name(result.stage), result.error_number,
                result.error_number != 0 ? std::strerror(result.error_number) : "none", result.device_id, result.path,
                static_cast<unsigned long long>(result.requested_bytes),
                static_cast<unsigned long long>(result.written_bytes), result.context_error
            );
            if (result.capacity_valid != 0) {
                LOG_ERROR(
                    "BootstrapDispatcher: device filesystem block_bytes=%llu free_blocks=%llu "
                    "available_blocks=%llu free_inodes=%llu",
                    static_cast<unsigned long long>(result.block_bytes),
                    static_cast<unsigned long long>(result.free_blocks),
                    static_cast<unsigned long long>(result.available_blocks),
                    static_cast<unsigned long long>(result.free_inodes)
                );
            } else if (result.capacity_errno != 0) {
                LOG_ERROR("BootstrapDispatcher: device filesystem query failed: errno=%d", result.capacity_errno);
            }
        }
        bootstrap_acknowledged_ = true;
        bootstrap_upload_ok_ = success;
    }
    int free_error = 0;
    for (void *&buffer : bootstrap_buffers_) {
        if (buffer == nullptr) continue;
        const int rc = aclrtFree(buffer);
        if (rc == 0) buffer = nullptr;
        else if (free_error == 0) free_error = rc;
    }
    if (free_error != 0) return free_error;
    bootstrap_buffers_.clear();
    bootstrap_stream_ = nullptr;
    bootstrap_complete_ = bootstrap_upload_ok_;
    return require_success && !bootstrap_upload_ok_ ? PTO_RUNTIME_ERR_INTERNAL : 0;
}

int LoadAicpuOp::FinalizeCallables(rtStream_t control_stream) {
    if (!finalize_acknowledged_) {
        if (control_stream == nullptr) return PTO_RUNTIME_ERR_INVALID_STATE;
        if (finalize_result_ == nullptr) {
            const int rc = aclrtMalloc(
                &finalize_result_, sizeof(AicpuFinalizeResult) + alignof(AicpuFinalizeResult) - 1,
                ACL_MEM_MALLOC_HUGE_FIRST
            );
            if (rc != 0) return rc;
        }
        // Keep the complete publication cache line inside our allocation,
        // without relying on a provider-specific malloc alignment guarantee.
        constexpr uintptr_t alignment = alignof(AicpuFinalizeResult);
        auto *result_address = reinterpret_cast<AicpuFinalizeResult *>(
            (reinterpret_cast<uintptr_t>(finalize_result_) + alignment - 1) & ~(alignment - 1)
        );
        if (!finalize_submitted_) {
            const AicpuFinalizeResult pending{PTO_RUNTIME_ERR_INVALID_STATE, 0};
            int rc = aclrtMemcpy(result_address, sizeof(pending), &pending, sizeof(pending), ACL_MEMCPY_HOST_TO_DEVICE);
            if (rc != 0) return rc;
            if (finalize_stream_ == nullptr) finalize_stream_ = control_stream;
            AicpuFinalizeArgs args{reinterpret_cast<uint64_t>(result_address)};
            finalize_submitted_ = true;
            rc = LaunchBuiltInOp(finalize_stream_, &args, sizeof(args), 1, KernelNames::FinalizeName);
            if (rc != 0) return rc;
        }
        // A failed wait/copy leaves the request and binary named. A retry waits
        // for this same request; it must not enqueue cleanup twice ambiguously.
        int rc = aclrtSynchronizeStream(finalize_stream_);
        if (rc != 0) return rc;
        AicpuFinalizeResult result{PTO_RUNTIME_ERR_INVALID_STATE, 0};
        rc = aclrtMemcpy(&result, sizeof(result), result_address, sizeof(result), ACL_MEMCPY_DEVICE_TO_HOST);
        if (rc != 0) return rc;
        // Successful stream completion alone cannot acknowledge an enqueue
        // that returned an error. An untouched result retains the request;
        // only the device's completed record permits a fresh cleanup attempt.
        if (result.completed != 1) {
            LOG_ERROR("AICPU cold cleanup has no device acknowledgement; retaining request, registration and stream");
            return PTO_RUNTIME_ERR_INVALID_STATE;
        }
        finalize_submitted_ = false;
        if (result.status != 0) return result.status;
        finalize_acknowledged_ = true;
    }
    if (finalize_result_ != nullptr) {
        const int rc = aclrtFree(finalize_result_);
        if (rc != 0) return rc;
        finalize_result_ = nullptr;
        finalize_stream_ = nullptr;
    }
    return 0;
}

int LoadAicpuOp::Finalize(rtStream_t control_stream) {
    const int bootstrap_rc = FinishBootstrap();
    if (bootstrap_rc != 0) return bootstrap_rc;
    if (binary_handle_ != nullptr) {
        // Init rollback may retain a registration before all entry handles
        // resolve. It has never accepted a callable and owes only RTS unload.
        if (initialized_) {
            const int rc = FinalizeCallables(control_stream);
            if (rc != 0) return rc;
        }
        rtError_t rc = rtsBinaryUnload(binary_handle_);
        if (rc != RT_ERROR_NONE) {
            // The binary is still loaded on the device, and this handle is the
            // only thing that names it. Retaining the whole loaded state keeps
            // a retry a plain re-invocation rather than a half-cleared object.
            LOG_ERROR("rtsBinaryUnload failed: %d — loader retains its handle for a close retry", rc);
            return rc;
        }
        binary_handle_ = nullptr;
    }
    finalize_acknowledged_ = false;
    initialized_ = false;
    bootstrap_complete_ = false;
    func_handles_.clear();
    inner_fp_ = 0;
    inner_so_basename_.clear();
    if (!json_file_path_.empty()) {
        std::remove(json_file_path_.c_str());
        json_file_path_.clear();
    }
    return 0;
}

void LoadAicpuOp::ForgetWithoutUnload() {
    bootstrap_buffers_.clear();
    bootstrap_stream_ = nullptr;
    bootstrap_complete_ = false;
    initialized_ = false;
    finalize_result_ = nullptr;
    finalize_stream_ = nullptr;
    finalize_submitted_ = false;
    finalize_acknowledged_ = false;
    binary_handle_ = nullptr;
    func_handles_.clear();
    inner_fp_ = 0;
    inner_so_basename_.clear();
    if (!json_file_path_.empty()) {
        std::remove(json_file_path_.c_str());
        json_file_path_.clear();
    }
}

// Destruction cannot return an unload error to the caller. Explicit runner
// finalization reports that error; this fallback discards the return value.
LoadAicpuOp::~LoadAicpuOp() { (void)Finalize(); }

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
    auto make_cfg = [&](const std::string &symbol_name) {
        AicpuOpConfig c;
        c.opType = MakeUniqueOpType(symbol_name.c_str(), inner_fp_, device_id_);
        c.functionName = symbol_name;
        c.kernelSo = kernel_so;
        c.opKernelLib = "AICPUKernel";
        c.userDefined = "False";
        return c;
    };
    std::vector<AicpuOpConfig> op_configs;
    op_configs.reserve(kernel_symbols_.size());
    for (const std::string &sym : kernel_symbols_) {
        op_configs.push_back(make_cfg(sym));
    }
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

int LoadAicpuOp::Init(const std::vector<std::string> &extra_symbols) {
    if (inner_fp_ == 0 || !bootstrap_complete_) {
        LOG_ERROR("LoadAicpuOp::Init: BootstrapDispatcher must be called first");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (binary_handle_ != nullptr) {
        // A retained handle means a previous unload failed and the binary is
        // still loaded. Loading a second one would overwrite the only record of
        // it; a successful Finalize is the only way past this.
        LOG_ERROR("LoadAicpuOp::Init: a binary is still loaded; Finalize must retire it first");
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }

    // Base entries are exported by every runtime; the runtime reports any extra
    // entries it additionally exports (TMARB: register_callable; hbg: none), so
    // this loader carries no runtime-specific symbol knowledge.
    kernel_symbols_ = {KernelNames::RunName, KernelNames::InitName, KernelNames::FinalizeName};
    kernel_symbols_.insert(kernel_symbols_.end(), extra_symbols.begin(), extra_symbols.end());

    // Per-process JSON path. /tmp is always writable.
    char json_name_buf[128];
    snprintf(
        json_name_buf, sizeof(json_name_buf), "/tmp/simpler_inner_%016lx_%d_%d.json", inner_fp_, device_id_,
        static_cast<int>(getpid())
    );
    json_file_path_ = json_name_buf;

    if (!GenerateAicpuOpJson(json_file_path_, inner_so_basename_)) {
        json_file_path_.clear();
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // RAII cleanups: any non-zero return path below unwinds via these guards.
    // .release() flips them off once the corresponding state becomes part of
    // the LoadAicpuOp's steady-state ownership.
    struct JsonGuard {
        std::string &path;
        void *&binary;
        bool active = true;
        ~JsonGuard() {
            if (active && !path.empty() && binary == nullptr) {
                std::remove(path.c_str());
                path.clear();
            }
        }
        void release() { active = false; }
    } json_guard{json_file_path_, binary_handle_};

    struct BinaryGuard {
        void *&handle;
        bool active = true;
        ~BinaryGuard() {
            // A rollback whose unload fails keeps the handle: the binary is
            // still loaded, and `Finalize()` is the only thing that can retry
            // it. Clearing here would strand it for the process's life.
            if (active && handle != nullptr) {
                rtError_t rc = rtsBinaryUnload(handle);
                if (rc != RT_ERROR_NONE) {
                    LOG_ERROR("Init rollback: rtsBinaryUnload failed: %d — retaining handle for a close retry", rc);
                    return;
                }
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

    LOG_INFO("LoadAicpuOp::Init: JSON=%s inner_basename=%s", json_file_path_.c_str(), inner_so_basename_.c_str());

    rtError_t rc = rtsBinaryLoadFromFile(json_file_path_.c_str(), &load_config, &binary_handle_);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtsBinaryLoadFromFile failed for %s: %d", json_file_path_.c_str(), rc);
        // binary_handle_ stays null; json_guard removes the JSON file.
        return rc;
    }
    LOG_INFO("LoadAicpuOp: Loaded inner SO via JSON, handle=%p", binary_handle_);

    std::unordered_map<std::string, rtFuncHandle> resolved;

    // Resolve every registered symbol. The set is exactly what this runtime
    // declares it exports (base + runtime-reported extras), so each one must
    // resolve — a miss is a real build/registration error, not an optional gap.
    for (const std::string &name : kernel_symbols_) {
        std::string lookup_name = MakeUniqueOpType(name.c_str(), inner_fp_, device_id_);
        rtFuncHandle func_handle = nullptr;
        rc = rtsFuncGetByName(binary_handle_, lookup_name.c_str(), &func_handle);
        if (rc != RT_ERROR_NONE) {
            LOG_ERROR("rtsFuncGetByName failed for %s: %d", lookup_name.c_str(), rc);
            // binary_guard unloads the partially-registered binary, json_guard
            // removes the JSON file. Symmetric with the rtsBinaryLoadFromFile
            // failure branch above.
            return rc;
        }
        resolved[name] = func_handle;
        LOG_INFO("LoadAicpuOp: resolved handle for %s (opType=%s): %p", name.c_str(), lookup_name.c_str(), func_handle);
    }

    func_handles_ = std::move(resolved);
    initialized_ = true;
    binary_guard.release();
    json_guard.release();
    return 0;
}

int LoadAicpuOp::AicpuKernelLaunch(
    rtFuncHandle func_handle, rtStream_t stream, void *args, size_t args_size, int aicpu_num
) {
    if (args == nullptr || args_size == 0) {
        LOG_ERROR("AicpuKernelLaunch: invalid arguments (args is null or args_size is 0)");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    rtCpuKernelArgs_t cpu_args = {};
    cpu_args.baseArgs.args = args;
    cpu_args.baseArgs.argsSize = args_size;

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

int LoadAicpuOp::LaunchBuiltInOp(
    rtStream_t stream, void *args, size_t args_size, int aicpu_num, const std::string &func_name
) {
    auto it = func_handles_.find(func_name);
    if (it == func_handles_.end()) {
        LOG_ERROR("Function not found: %s", func_name.c_str());
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    return AicpuKernelLaunch(it->second, stream, args, args_size, aicpu_num);
}

}  // namespace host
