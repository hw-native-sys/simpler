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
#include <fstream>
#include <unordered_map>
#include <vector>
#include <unistd.h>

#include "common/unified_log.h"
#include "common/kernel_args.h"

#ifdef BUILD_WITH_NEW_CANN
// New CANN RTS header for rtsLaunchCpuKernel interface (CANN 7.0+)
#include "runtime/runtime/rts/rts_kernel.h"
#include "runtime/runtime/kernel.h"

// Forward declarations for JSON structures
struct AicpuOpConfig {
    std::string functionName;
    std::string kernelSo;
    std::string opKernelLib;
    std::string computeCost = "100";
    std::string engine = "DNN_VM_AICPU";
    std::string flagAsync = "False";
    std::string flagPartial = "False";
    std::string userDefined = "False";
    std::string opType;
};

// Generate AICPU op info JSON file
static bool GenerateAicpuOpJson(const std::string &json_path, const std::vector<AicpuOpConfig> &op_configs) {
    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
        LOG_ERROR("Failed to open JSON file for writing: %s", json_path.c_str());
        return false;
    }

    json_file << "{\n";
    for (size_t i = 0; i < op_configs.size(); ++i) {
        const auto &config = op_configs[i];
        json_file << "  \"" << config.opType << "\": {\n";
        json_file << "    \"opInfo\": {\n";
        json_file << "      \"functionName\": \"" << config.functionName << "\",\n";
        json_file << "      \"kernelSo\": \"" << config.kernelSo << "\",\n";
        json_file << "      \"opKernelLib\": \"" << config.opKernelLib << "\",\n";
        json_file << "      \"computeCost\": \"" << config.computeCost << "\",\n";
        json_file << "      \"engine\": \"" << config.engine << "\",\n";
        json_file << "      \"flagAsync\": \"" << config.flagAsync << "\",\n";
        json_file << "      \"flagPartial\": \"" << config.flagPartial << "\",\n";
        json_file << "      \"userDefined\": \"" << config.userDefined << "\"\n";
        json_file << "    }\n";
        json_file << "  }" << (i < op_configs.size() - 1 ? "," : "") << "\n";
    }
    json_file << "}\n";
    json_file.close();

    LOG_INFO("Generated AICPU op info JSON: %s", json_path.c_str());
    return true;
}

#endif

int AicpuLoader::init_with_binary(
    const std::vector<uint8_t> &aicpu_binary, const std::vector<std::string> &kernel_names
) {
#ifdef BUILD_WITH_NEW_CANN
    // New interface: Load binary using JSON descriptor (pypto approach)
    LOG_INFO("AicpuLoader: Using new rtsBinaryLoadFromFile + rtsLaunchCpuKernel interface");
    LOG_INFO("AicpuLoader: Binary size=%zu bytes", aicpu_binary.size());

    // Step 1: Generate op info JSON at runtime (using only filename, not full path)
    const char *tmp_dir = std::getenv("TMPDIR") ? std::getenv("TMPDIR") : "/tmp";
    std::string json_path_template = std::string(tmp_dir) + "/simpler_aicpu_op_info_XXXXXX.json";
    std::vector<char> json_path_buffer(json_path_template.begin(), json_path_template.end());
    json_path_buffer.push_back('\0');

    int json_fd = mkstemps(json_path_buffer.data(), 5);
    if (json_fd == -1) {
        LOG_ERROR("Failed to create temporary JSON file");
        return -1;
    }
    close(json_fd);
    json_file_path_ = json_path_buffer.data();

    // Map opType (used for rtsFuncGetByName) to functionName (actual symbol in .so)
    std::unordered_map<std::string, std::string> name_mapping = {
        {"DynTileFwkKernelServerInit", "DynTileFwkBackendKernelServerInit"},
        {"DynTileFwkKernelServer", "DynTileFwkBackendKernelServer"}
    };

    // Create op configs for JSON generation
    // kernelSo uses only filename - runtime will find it via library search path
    std::vector<AicpuOpConfig> op_configs;
    for (const auto &name : kernel_names) {
        AicpuOpConfig config;
        config.opType = name;
        config.functionName = name_mapping[name];
        config.kernelSo = "libaicpu_kernel.so";  // Filename only, runtime searches library path
        config.opKernelLib = "KFCKernel";
        op_configs.push_back(config);
    }

    // Generate JSON file
    if (!GenerateAicpuOpJson(json_file_path_, op_configs)) {
        return -1;
    }

    // Step 2: Load binary handle from JSON: rtsBinaryLoadFromFile
    // cpuKernelMode=0: JSON only mode, runtime finds .so via library search path
    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 0;

    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option;
    load_config.numOpt = 1;

    rtError_t rc = rtsBinaryLoadFromFile(json_file_path_.c_str(), &load_config, &binary_handle_);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtsBinaryLoadFromFile failed for %s: %d", json_file_path_.c_str(), rc);
        return rc;
    }
    LOG_INFO("AicpuLoader: Loaded binary from JSON, handle=%p", binary_handle_);

    // Step 3: Resolve function handles: rtsFuncGetByName
    for (const auto &name : kernel_names) {
        rtFuncHandle func_handle = nullptr;
        rc = rtsFuncGetByName(binary_handle_, name.c_str(), &func_handle);
        if (rc != RT_ERROR_NONE) {
            LOG_ERROR("rtsFuncGetByName failed for %s: %d", name.c_str(), rc);
            return rc;
        }
        func_handles_[name] = func_handle;
        LOG_INFO("AicpuLoader: Resolved function handle for %s: %p", name.c_str(), func_handle);
    }

    return 0;

#else
    // Legacy interface: No pre-loading needed
    (void)so_path;
    (void)kernel_names;
    LOG_INFO("AicpuLoader: Using legacy rtAicpuKernelLaunchExWithArgs interface");
    return 0;
#endif
}

int AicpuLoader::init(const std::string &so_path, const std::vector<std::string> &kernel_names) {
#ifdef BUILD_WITH_NEW_CANN
    // New interface: Use init_with_binary() instead
    // This init() is kept for backward compatibility but does nothing
    (void)so_path;
    (void)kernel_names;
    LOG_INFO("AicpuLoader: Use init_with_binary() for new interface");
    return 0;
#else
    // Legacy interface: No pre-loading needed
    (void)so_path;
    (void)kernel_names;
    LOG_INFO("AicpuLoader: Using legacy rtAicpuKernelLaunchExWithArgs interface");
    return 0;
#endif
}

int AicpuLoader::launch(rtStream_t stream, KernelArgs *k_args, const char *kernel_name, int aicpu_num) {
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

    // Delete temporary JSON file if it was created
    if (!json_file_path_.empty()) {
        std::remove(json_file_path_.c_str());
        LOG_INFO("AicpuLoader: Deleted temporary JSON file: %s", json_file_path_.c_str());
        json_file_path_.clear();
    }

    LOG_INFO("AicpuLoader: Finalized new interface");
#else
    // Legacy interface: No-op
    (void)this;  // Suppress unused warning
#endif
}
