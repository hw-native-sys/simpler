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
 * @file load_aicpu_op.h
 * @brief Host-side AICPU operation loader using new CANN 7.0+ rtsLaunchCpuKernel interface
 *
 * This class provides the host-side wrapper for loading and launching AICPU kernels
 * through the two-layer dispatcher architecture. It generates JSON descriptors,
 * loads the dispatcher SO via rtsBinaryLoadFromFile, and launches kernels via
 * rtsLaunchCpuKernel.
 *
 * Architecture:
 * - Dispatcher SO (libretr_kernels.so) - Fixed outer layer (named to match CANN whitelist)
 * - Runtime SO (replaceable) - Different for each runtime (tensormap, ringbuffer, etc.)
 *
 * Three-phase launch pattern:
 * 1. Null phase (DynTileFwkKernelServerNull) - Pass inner SO binary to dispatcher
 * 2. Init phase (DynTileFwkKernelServerInit) - Initialize inner SO
 * 3. Run phase (DynTileFwkKernelServer) - Execute actual kernel
 *
 * IMPORTANT: In cpuKernelMode=1, Null phase is SKIPPED - scheduler handles SO loading
 */

#ifndef COMMON_HOST_LOAD_AICPU_OP_H_
#define COMMON_HOST_LOAD_AICPU_OP_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include "common/kernel_args.h"

#ifdef BUILD_WITH_NEW_CANN
#include "runtime/runtime/rts/rts_kernel.h"
#endif

namespace host {

/**
 * @brief AICPU operation configuration for JSON descriptor generation
 */
struct AicpuOpConfig {
    std::string functionName;  // Actual symbol name in SO (e.g., DynTileFwkBackendKernelServerInit)
    std::string kernelSo;      // SO filename (e.g., libaicpu_dispatcher.so)
    std::string opKernelLib;   // Kernel library type (KFCKernel or AICPUKernel)
    std::string computeCost = "100";
    std::string engine = "DNN_VM_AICPU";
    std::string flagAsync = "False";
    std::string flagPartial = "False";
    std::string userDefined = "False";
    std::string opType;        // External kernel name for rtsFuncGetByName lookup
};

/**
 * @brief Host-side AICPU operation loader
 *
 * Manages the lifecycle of loading and launching AICPU kernels through the
 * two-layer dispatcher architecture using CANN 7.0+ rtsLaunchCpuKernel interface.
 *
 * Reference: /data/fangjingzhi/pypto/framework/src/machine/runtime/load_aicpu_op.{h,cpp}
 */
class LoadAicpuOp {
public:
    LoadAicpuOp() = default;
    ~LoadAicpuOp();

    // Delete copy and move to ensure singleton behavior
    LoadAicpuOp(const LoadAicpuOp&) = delete;
    LoadAicpuOp& operator=(const LoadAicpuOp&) = delete;
    LoadAicpuOp(LoadAicpuOp&&) = delete;
    LoadAicpuOp& operator=(LoadAicpuOp&&) = delete;

    /**
     * @brief Initialize the loader by loading dispatcher SO
     *
     * Passes the dispatcher SO path directly to rtsBinaryLoadFromFile
     * and resolves function handles via rtsFuncGetByName.
     *
     * @param dispatcher_so_path Absolute path to libaicpu_dispatcher.so
     * @return 0 on success, error code on failure
     */
    int Init(const std::string& dispatcher_so_path);

    /**
     * @brief Launch a built-in dispatcher kernel
     *
     * Launches one of the three dispatcher kernels (Null/Init/Run) via
     * rtsLaunchCpuKernel.
     *
     * @param stream RTS stream for kernel launch
     * @param k_args Kernel arguments to pass to the AICPU kernel
     * @param aicpu_num Number of AICPU cores to use
     * @param func_name Kernel function name for rtsFuncGetByName lookup (PyptoNull/PyptoInit/PyptoRun)
     * @param kernel_name Actual symbol name in the SO (DynTileFwkKernelServerNull/Init/Server)
     * @return 0 on success, error code on failure
     */
    int LaunchBuiltInOp(
        rtStream_t stream, KernelArgs* k_args, int aicpu_num, const std::string& func_name, const std::string& kernel_name
    );

private:
#ifdef BUILD_WITH_NEW_CANN
    void* binary_handle_ = nullptr;                              // Handle from rtsBinaryLoadFromFile
    std::unordered_map<std::string, rtFuncHandle> func_handles_;  // Function handles from rtsFuncGetByName
    std::string json_file_path_;                                   // Path to generated JSON file (same dir/basename as SO)

    /**
     * @brief Generate JSON descriptor for dispatcher SO
     *
     * @param json_path Path where JSON file will be created
     * @param kernel_so Absolute path to the dispatcher SO (placed in kernelSo JSON field)
     * @return true on success, false on failure
     */
    bool GenerateAicpuOpJson(const std::string& json_path, const std::string& kernel_so);

    /**
     * @brief Launch AICPU kernel using rtsLaunchCpuKernel
     *
     * @param func_handle Function handle from rtsFuncGetByName
     * @param stream RTS stream
     * @param k_args Kernel arguments
     * @param aicpu_num Number of AICPU cores
     * @param kernel_name Kernel name to embed in args struct
     * @return 0 on success, error code on failure
     */
    int AicpuKernelLaunch(
        rtFuncHandle func_handle, rtStream_t stream, KernelArgs* k_args, int aicpu_num, const std::string& kernel_name
    );
#else
    // Dummy members for legacy build
    void* binary_handle_ = nullptr;
#endif
};

// Kernel name constants
namespace KernelNames {
    constexpr const char* NullName = "DynTileFwkKernelServerNull";     // Null phase
    constexpr const char* InitName = "DynTileFwkKernelServerInit";     // Init phase
    constexpr const char* RunName = "DynTileFwkKernelServer";          // Run phase
}

// Dispatcher SO name (use "retr_kernels" to match CANN whitelist)
namespace SoNames {
    constexpr const char* DispatcherSo = "libretr_kernels.so";
}

}  // namespace host

#endif  // COMMON_HOST_LOAD_AICPU_OP_H_
