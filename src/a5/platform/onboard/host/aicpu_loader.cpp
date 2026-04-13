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
 * AICPU Loader Implementation
 */

#include "aicpu_loader.h"

#include <cstring>
#include <iostream>
#include <unordered_map>

#include "common/unified_log.h"
#include "common/kernel_args.h"

#ifdef BUILD_WITH_NEW_CANN
// New CANN RTS header for rtsLaunchCpuKernel interface (CANN 7.0+)
#include "runtime/runtime/rts/rts_kernel.h"
#include "runtime/runtime/kernel.h"
#endif

int AicpuLoader::init_with_binary(const std::vector<uint8_t>& aicpu_so_binary, const std::vector<std::string>& kernel_names) {
#ifdef BUILD_WITH_NEW_CANN
    // New interface: Load binary from memory and resolve function handles
    LOG_INFO("AicpuLoader: Using new rtsBinaryLoadFromData + rtsLaunchCpuKernel interface");

    if (aicpu_so_binary.empty()) {
        LOG_ERROR("AicpuLoader: AICPU binary is empty");
        return -1;
    }

    // 1. Load binary from memory: rtsBinaryLoadFromData
    // Try with CPU kernel mode configuration
    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 1;  // Load CPU so & json

    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option;
    load_config.numOpt = 1;

    rtError_t rc = rtsBinaryLoadFromData(
        aicpu_so_binary.data(),
        aicpu_so_binary.size(),
        &load_config,
        &binary_handle_
    );
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtsBinaryLoadFromData failed: %d (binary size=%zu)", rc, aicpu_so_binary.size());
        return rc;
    }
    LOG_INFO("AicpuLoader: Loaded binary from memory, handle=%p, size=%zu", binary_handle_, aicpu_so_binary.size());

    // Map kernel names to backend versions (actual symbol names in the .so)
    std::unordered_map<std::string, std::string> name_mapping = {
        {"DynTileFwkKernelServerInit", "DynTileFwkBackendKernelServerInit"},
        {"DynTileFwkKernelServer", "DynTileFwkBackendKernelServer"}
    };

    // 2. Resolve function handles: rtsFuncGetByName
    for (const auto& name : kernel_names) {
        // Map to the actual symbol name
        std::string actual_name = name;
        auto it = name_mapping.find(name);
        if (it != name_mapping.end()) {
            actual_name = it->second;
        }

        rtFuncHandle func_handle = nullptr;
        rc = rtsFuncGetByName(binary_handle_, actual_name.c_str(), &func_handle);
        if (rc != RT_ERROR_NONE) {
            LOG_ERROR("rtsFuncGetByName failed for %s (mapped from %s): %d", actual_name.c_str(), name.c_str(), rc);
            return rc;
        }
        func_handles_[name] = func_handle;  // Store with original name for lookup
        LOG_INFO("AicpuLoader: Resolved function handle for %s -> %s: %p", name.c_str(), actual_name.c_str(), func_handle);
    }

    return 0;

#else
    // Legacy interface: No pre-loading needed
    (void)aicpu_so_binary;
    (void)kernel_names;
    LOG_INFO("AicpuLoader: Using legacy rtAicpuKernelLaunchExWithArgs interface");
    return 0;
#endif
}

int AicpuLoader::init(const std::string& so_path, const std::vector<std::string>& kernel_names) {
#ifdef BUILD_WITH_NEW_CANN
    // New interface: Store kernel names for later resolution
    // Binary will be loaded via init_with_binary()
    LOG_INFO("AicpuLoader: Using new rtsLaunchCpuKernel interface (binary not loaded yet)");
    (void)so_path;
    (void)kernel_names;
    return 0;  // Binary will be loaded separately via init_with_binary()
#else
    // Legacy interface: No pre-loading needed
    (void)so_path;
    (void)kernel_names;
    LOG_INFO("AicpuLoader: Using legacy rtAicpuKernelLaunchExWithArgs interface");
    return 0;
#endif
}

int AicpuLoader::launch(rtStream_t stream, KernelArgs* k_args, const char* kernel_name, int aicpu_num) {
#ifdef BUILD_WITH_NEW_CANN
    // New interface: rtsLaunchCpuKernel
    auto it = func_handles_.find(kernel_name);
    if (it == func_handles_.end()) {
        LOG_ERROR("Kernel not found: %s", kernel_name);
        return -1;
    }

    rtFuncHandle func_handle = it->second;

    // Prepare args for new interface
    struct Args {
        KernelArgs k_args;
        char kernel_name[64];
        char so_name[64];
    } args;

    args.k_args = *k_args;
    std::strncpy(args.kernel_name, kernel_name, sizeof(args.kernel_name) - 1);
    args.kernel_name[sizeof(args.kernel_name) - 1] = '\0';
    std::strncpy(args.so_name, "libaicpu_extend_kernels.so", sizeof(args.so_name) - 1);
    args.so_name[sizeof(args.so_name) - 1] = '\0';

    rtCpuKernelArgs_t cpu_args = {};
    cpu_args.baseArgs.args = &args;
    cpu_args.baseArgs.argsSize = sizeof(args);
    cpu_args.baseArgs.kernelNameAddrOffset = offsetof(struct Args, kernel_name);
    cpu_args.baseArgs.soNameAddrOffset = offsetof(struct Args, so_name);
    cpu_args.baseArgs.hostInputInfoPtr = nullptr;
    cpu_args.baseArgs.kernelOffsetInfoPtr = nullptr;
    cpu_args.baseArgs.hostInputInfoNum = 0;
    cpu_args.baseArgs.kernelOffsetInfoNum = 0;
    cpu_args.baseArgs.isNoNeedH2DCopy = 0;
    cpu_args.baseArgs.timeout = 0;
    cpu_args.cpuParamHeadOffset = 0;

    // Launch: rtsLaunchCpuKernel
    rtError_t rc = rtsLaunchCpuKernel(func_handle, static_cast<uint32_t>(aicpu_num), stream, nullptr, &cpu_args);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtsLaunchCpuKernel failed for %s: %d", kernel_name, rc);
        return rc;
    }

    return 0;

#else
    // Legacy interface: rtAicpuKernelLaunchExWithArgs
    struct Args {
        KernelArgs k_args;
        char kernel_name[32];
        const char so_name[32] = {"libaicpu_extend_kernels.so"};
        const char op_name[32] = {""};
    } args;

    args.k_args = *k_args;
    std::strncpy(args.kernel_name, kernel_name, sizeof(args.kernel_name) - 1);
    args.kernel_name[sizeof(args.kernel_name) - 1] = '\0';

    rtAicpuArgsEx_t rt_args;
    std::memset(&rt_args, 0, sizeof(rt_args));
    rt_args.args = &args;
    rt_args.argsSize = sizeof(args);
    rt_args.kernelNameAddrOffset = offsetof(struct Args, kernel_name);
    rt_args.soNameAddrOffset = offsetof(struct Args, so_name);

    return rtAicpuKernelLaunchExWithArgs(
        rtKernelType_t::KERNEL_TYPE_AICPU_KFC, "AST_DYN_AICPU", aicpu_num, &rt_args, nullptr, stream, 0
    );
#endif
}

void AicpuLoader::finalize() {
#ifdef BUILD_WITH_NEW_CANN
    // New interface: Unload binary and clear handles
    if (binary_handle_ != nullptr) {
        rtError_t rc = rtsBinaryUnload(binary_handle_);
        if (rc != RT_ERROR_NONE) {
            LOG_WARN("rtsBinaryUnload failed: %d", rc);
        }
        binary_handle_ = nullptr;
    }
    func_handles_.clear();
    LOG_INFO("AicpuLoader: Finalized new interface");
#else
    // Legacy interface: No-op
    (void)this;  // Suppress unused warning
#endif
}
