/*
 * Copyright (c) PyPTO Contributors.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OR ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 * -----------------------------------------------------------------------------------------------------------
 */
/**
 * AICPU Operation Loader Implementation
 */

#include "load_aicpu_op.h"

#include <cerrno>
#include <cstring>
#include <fstream>
#include <memory>

#include "common/unified_log.h"

#ifdef BUILD_WITH_NEW_CANN

namespace host {

LoadAicpuOp::~LoadAicpuOp()
{
#ifdef BUILD_WITH_NEW_CANN
    if (binary_handle_ != nullptr) {
        rtError_t rc = rtsBinaryUnload(binary_handle_);
        if (rc != RT_ERROR_NONE) {
            LOG_WARN("rtsBinaryUnload failed: %d", rc);
        }
        binary_handle_ = nullptr;
    }
    func_handles_.clear();

    if (!json_file_path_.empty()) {
        std::remove(json_file_path_.c_str());
        LOG_INFO("LoadAicpuOp: Deleted temporary JSON file: %s", json_file_path_.c_str());
        json_file_path_.clear();
    }
#endif
}

bool LoadAicpuOp::GenerateAicpuOpJson(const std::string& json_path, const std::string& kernel_so)
{
    std::ofstream json_file(json_path);
    if (!json_file.is_open()) {
        LOG_ERROR("Failed to open JSON file for writing: %s", json_path.c_str());
        return false;
    }

    AicpuOpConfig init_config;
    init_config.opType = KernelNames::InitName;
    init_config.functionName = "DynTileFwkKernelServerInit";
    init_config.kernelSo = kernel_so;
    init_config.opKernelLib = "KFCKernel";

    AicpuOpConfig run_config;
    run_config.opType = KernelNames::RunName;
    run_config.functionName = "DynTileFwkKernelServer";
    run_config.kernelSo = kernel_so;
    run_config.opKernelLib = "KFCKernel";

    AicpuOpConfig null_config;
    null_config.opType = KernelNames::NullName;
    null_config.functionName = "DynTileFwkKernelServerNull";
    null_config.kernelSo = kernel_so;
    null_config.opKernelLib = "AICPUKernel";

    std::vector<AicpuOpConfig> op_configs = {init_config, run_config, null_config};

    json_file << "{\n";
    for (size_t i = 0; i < op_configs.size(); ++i) {
        const auto& config = op_configs[i];
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

int LoadAicpuOp::Init(const std::string& dispatcher_so_path)
{
    // Generate JSON in the same directory as the SO, with the same basename
    // e.g. /path/libretr_kernels.so -> /path/libretr_kernels.json
    // cpuKernelMode=1 derives the SO path by replacing .json with .so
    std::string so_dir;
    size_t last_slash = dispatcher_so_path.rfind('/');
    if (last_slash != std::string::npos) {
        so_dir = dispatcher_so_path.substr(0, last_slash + 1);
    }

    std::string so_basename = dispatcher_so_path;
    if (last_slash != std::string::npos) {
        so_basename = dispatcher_so_path.substr(last_slash + 1);
    }
    // Replace .so suffix with .json
    std::string json_name = so_basename;
    size_t so_ext = json_name.rfind(".so");
    if (so_ext != std::string::npos) {
        json_name = json_name.substr(0, so_ext) + ".json";
    }

    json_file_path_ = so_dir + json_name;

    // kernelSo uses relative filename (scheduler resolves via ASCEND_AICPU_PATH)
    if (!GenerateAicpuOpJson(json_file_path_, so_basename)) {
        json_file_path_.clear();
        return -1;
    }

    // Load via rtsBinaryLoadFromFile with cpuKernelMode=1
    rtLoadBinaryOption_t option = {};
    option.optionId = RT_LOAD_BINARY_OPT_CPU_KERNEL_MODE;
    option.value.cpuKernelMode = 1;

    rtLoadBinaryConfig_t load_config = {};
    load_config.options = &option;
    load_config.numOpt = 1;

    LOG_INFO("LoadAicpuOp: JSON path: %s", json_file_path_.c_str());
    LOG_INFO("LoadAicpuOp: SO path: %s", dispatcher_so_path.c_str());

    rtError_t rc = rtsBinaryLoadFromFile(json_file_path_.c_str(), &load_config, &binary_handle_);
    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtsBinaryLoadFromFile failed for %s: %d", json_file_path_.c_str(), rc);
        std::remove(json_file_path_.c_str());
        json_file_path_.clear();
        return rc;
    }
    LOG_INFO("LoadAicpuOp: Loaded dispatcher SO, handle=%p", binary_handle_);

    // Step 4: Resolve function handles for all three kernels
    const char* kernel_names[] = {KernelNames::NullName, KernelNames::InitName, KernelNames::RunName};
    for (const char* name : kernel_names) {
        rtFuncHandle func_handle = nullptr;
        rc = rtsFuncGetByName(binary_handle_, name, &func_handle);
        if (rc != RT_ERROR_NONE) {
            LOG_ERROR("rtsFuncGetByName failed for %s: %d", name, rc);
            return rc;
        }
        func_handles_[name] = func_handle;
        LOG_INFO("LoadAicpuOp: Resolved function handle for %s: %p", name, func_handle);
    }

    return 0;
}

int LoadAicpuOp::AicpuKernelLaunch(
    rtFuncHandle func_handle, rtStream_t stream, KernelArgs* k_args, int aicpu_num, const std::string& kernel_name
) {
    (void)kernel_name;

    LOG_INFO("LoadAicpuOp::AicpuKernelLaunch: func_handle=%p, aicpu_num=%d", func_handle, aicpu_num);

    rtCpuKernelArgs_t cpu_args = {};
    cpu_args.baseArgs.args = k_args;
    cpu_args.baseArgs.argsSize = sizeof(KernelArgs);

    rtKernelLaunchCfg_t kernelLaunchCfg = {nullptr, 0U};
    auto launchKernelAttr = std::make_unique<rtLaunchKernelAttr_t>();
    kernelLaunchCfg.attrs = launchKernelAttr.get();

    LOG_INFO("LoadAicpuOp::AicpuKernelLaunch: calling rtsLaunchCpuKernel...");
    rtError_t rc = rtsLaunchCpuKernel(func_handle, static_cast<uint32_t>(aicpu_num), stream, &kernelLaunchCfg, &cpu_args);
    LOG_INFO("LoadAicpuOp::AicpuKernelLaunch: rtsLaunchCpuKernel returned %d", rc);

    if (rc != RT_ERROR_NONE) {
        LOG_ERROR("rtsLaunchCpuKernel failed: %d", rc);
        return rc;
    }

    return 0;
}

int LoadAicpuOp::LaunchBuiltInOp(
    rtStream_t stream, KernelArgs* k_args, int aicpu_num, const std::string& func_name, const std::string& kernel_name
) {
    LOG_INFO("LoadAicpuOp::LaunchBuiltInOp: func_name=%s, kernel_name=%s, aicpu_num=%d", func_name.c_str(), kernel_name.c_str(), aicpu_num);

    auto it = func_handles_.find(func_name);
    if (it == func_handles_.end()) {
        LOG_ERROR("Function not found: %s", func_name.c_str());
        return -1;
    }

    rtFuncHandle func_handle = it->second;
    LOG_INFO("LoadAicpuOp::LaunchBuiltInOp: calling AicpuKernelLaunch with func_handle=%p", func_handle);

    int rc = AicpuKernelLaunch(func_handle, stream, k_args, aicpu_num, kernel_name);
    LOG_INFO("LoadAicpuOp::LaunchBuiltInOp: AicpuKernelLaunch returned %d", rc);

    return rc;
}

}  // namespace host

#endif  // BUILD_WITH_NEW_CANN
