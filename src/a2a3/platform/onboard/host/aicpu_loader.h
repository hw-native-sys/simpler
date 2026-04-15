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
 * AICPU Loader Abstraction (Legacy Interface)
 *
 * Provides AICPU kernel launching via the legacy rtAicpuKernelLaunchExWithArgs API.
 * Used when BUILD_WITH_NEW_CANN is OFF. When BUILD_WITH_NEW_CANN is ON,
 * device_runner uses LoadAicpuOp (src/common/host/load_aicpu_op.h) instead.
 */

#ifndef A2A3_PLATFORM_ONBOARD_HOST_AICPU_LOADER_H_
#define A2A3_PLATFORM_ONBOARD_HOST_AICPU_LOADER_H_

#include <cstdint>
#include <string>
#include <vector>

#include <runtime/rt.h>

// Forward declarations
struct KernelArgs;

/**
 * @brief AICPU kernel loader (legacy interface)
 *
 * Launches AICPU kernels via the legacy rtAicpuKernelLaunchExWithArgs API.
 * Used as the fallback when BUILD_WITH_NEW_CANN is OFF.
 */
class AicpuLoader {
public:
    AicpuLoader() = default;
    ~AicpuLoader() = default;

    /**
     * @brief Initialize the AICPU loader with binary data (no-op for legacy interface)
     */
    int init_with_binary(const std::vector<uint8_t> &aicpu_binary, const std::vector<std::string> &kernel_names);

    /**
     * @brief Initialize the AICPU loader (no-op for legacy interface)
     */
    int init(const std::string &so_path, const std::vector<std::string> &kernel_names);

    /**
     * @brief Launch an AICPU kernel via legacy rtAicpuKernelLaunchExWithArgs
     */
    int launch(rtStream_t stream, KernelArgs *k_args, const char *kernel_name, int aicpu_num);

    /**
     * @brief Cleanup resources (no-op for legacy interface)
     */
    void finalize();

    // Disable copy and move
    AicpuLoader(const AicpuLoader &) = delete;
    AicpuLoader &operator=(const AicpuLoader &) = delete;
    AicpuLoader(AicpuLoader &&) = delete;
    AicpuLoader &operator=(AicpuLoader &&) = delete;
};

#endif  // A2A3_PLATFORM_ONBOARD_HOST_AICPU_LOADER_H_
