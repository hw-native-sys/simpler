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
 * AICPU Loader Abstraction
 *
 * This file provides an abstraction layer for AICPU kernel launching that supports
 * both the legacy rtAicpuKernelLaunchExWithArgs API and the new rtsLaunchCpuKernel
 * interface available in newer CANN versions.
 *
 * The interface used is controlled by the BUILD_WITH_NEW_CANN compile flag:
 * - When undefined or OFF: Uses legacy rtAicpuKernelLaunchExWithArgs
 * - When ON: Uses new rtsLaunchCpuKernel / rtsBinaryLoadFromFile / rtsFuncGetByName
 */

#ifndef A2A3_PLATFORM_ONBOARD_HOST_AICPU_LOADER_H_
#define A2A3_PLATFORM_ONBOARD_HOST_AICPU_LOADER_H_

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <runtime/rt.h>

// Forward declarations
struct KernelArgs;

/**
 * @brief AICPU kernel loader abstraction
 *
 * Supports both legacy and new CANN AICPU launch interfaces through conditional compilation.
 */
class AicpuLoader {
public:
    AicpuLoader() = default;
    ~AicpuLoader() = default;

    /**
     * @brief Initialize the AICPU loader with binary data
     *
     * For the new interface (BUILD_WITH_NEW_CANN=ON), this generates a JSON descriptor
     * and loads the binary using rtsBinaryLoadFromFile. The .so file is referenced by
     * filename only (libaicpu_kernel.so) and must be findable via library search path.
     * For the legacy interface (BUILD_WITH_NEW_CANN=OFF), this is a no-op.
     *
     * @param aicpu_binary Binary data of the AICPU shared library (not used, kept for API compatibility)
     * @param kernel_names List of kernel function names to resolve
     * @return 0 on success, error code on failure
     */
    int init_with_binary(const std::vector<uint8_t>& aicpu_binary, const std::vector<std::string>& kernel_names);

    /**
     * @brief Initialize the AICPU loader (legacy interface compatibility)
     *
     * For the new interface (BUILD_WITH_NEW_CANN=ON), this stores kernel names for later use.
     * For the legacy interface (BUILD_WITH_NEW_CANN=OFF), this is a no-op.
     *
     * @param so_path Path to the AICPU shared library (not used in new interface)
     * @param kernel_names List of kernel function names to resolve
     * @return 0 on success, error code on failure
     */
    int init(const std::string& so_path, const std::vector<std::string>& kernel_names);

    /**
     * @brief Launch an AICPU kernel
     *
     * Unified interface that delegates to either legacy or new implementation.
     *
     * @param stream CUDA-style stream for execution
     * @param k_args Kernel arguments
     * @param kernel_name Name of the kernel to launch
     * @param aicpu_num Number of AICPU instances to launch
     * @return 0 on success, error code on failure
     */
    int launch(rtStream_t stream, KernelArgs* k_args, const char* kernel_name, int aicpu_num);

    /**
     * @brief Cleanup resources
     *
     * For the new interface, this unloads the binary and clears handles.
     * For the legacy interface, this is a no-op.
     */
    void finalize();

    // Disable copy and move
    AicpuLoader(const AicpuLoader&) = delete;
    AicpuLoader& operator=(const AicpuLoader&) = delete;
    AicpuLoader(AicpuLoader&&) = delete;
    AicpuLoader& operator=(AicpuLoader&&) = delete;

private:
#ifdef BUILD_WITH_NEW_CANN
    // New interface members
    void* binary_handle_ = nullptr;  // Binary handle from rtsBinaryLoadFromFile
    std::unordered_map<std::string, void*> func_handles_;  // Function handles (kernel_name -> func_handle)
    std::string json_file_path_;  // Path to temporary JSON descriptor file
#else
    // Legacy interface - no state needed
#endif
};

#endif  // A2A3_PLATFORM_ONBOARD_HOST_AICPU_LOADER_H_
