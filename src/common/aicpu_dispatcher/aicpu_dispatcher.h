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
 * AICPU Dispatcher SO - Two-layer architecture for runtime-specific AICPU kernels
 *
 * This dispatcher SO provides a two-layer architecture where:
 * - Outer layer (this SO) is fixed and handles dynamic SO loading
 * - Inner layer (runtime-specific SO) can be different for each runtime
 *
 * Architecture:
 * 1. DynTileFwkKernelServerNull - Load phase: receives inner SO binary, saves to AICPU filesystem
 * 2. DynTileFwkKernelServerInit - Init phase: delegates to inner SO's initialization
 * 3. DynTileFwkKernelServer - Run phase: delegates to inner SO's execution
 *
 * This allows different runtimes (tensormap, ringbuffer, etc.) to load their own
 * AICPU kernel implementations at runtime without recompiling the dispatcher.
 */

#ifndef COMMON_AICPU_DISPATCHER_AICPU_DISPATCHER_H_
#define COMMON_AICPU_DISPATCHER_AICPU_DISPATCHER_H_

#include <cstdint>
#include <fstream>
#include <dlfcn.h>
#include <string>
#include <unordered_map>
#include <mutex>

#include "common/unified_log.h"

// Function pointer type for AICPU kernel functions
using AicpuKernelFunc = int (*)(void*);

namespace aicpu_dispatcher {

// Function key constants for inner SO function lookup
constexpr uint64_t dyInitFuncKey = 2;
constexpr uint64_t dyExecFuncKey = 3;

// Kernel name constants (actual symbol names in inner SO)
constexpr char const* DY_TILE_FWK_BACKEND_KERNEL_SERVER_INIT = "DynTileFwkBackendKernelServerInit";
constexpr char const* DY_TILE_FWK_BACKEND_KERNEL_SERVER = "DynTileFwkBackendKernelServer";

/**
 * @brief Backend server handle manager for two-layer SO architecture
 *
 * Manages the lifecycle of the inner SO:
 * - Saves inner SO binary to /tmp/aicpu_kernels/
 * - Loads functions from inner SO using dlopen/dlsym
 * - Executes inner SO functions with provided arguments
 *
 * Data flow:
 * - Host passes inner SO binary via DeviceArgs (aicpu_so_bin, aicpu_so_len)
 * - Dispatcher's Null function receives KernelArgs->device_args pointer
 * - Binary is saved to filesystem and inner SO is loaded via dlopen
 */
class BackendServerHandleManager {
public:
    BackendServerHandleManager() = default;
    ~BackendServerHandleManager();

    /**
     * @brief Save inner SO binary to AICPU filesystem
     *
     * @param data Pointer to inner SO binary data
     * @param len Length of the binary data
     * @param deviceId Device ID for SO naming
     * @return true on success, false on failure
     */
    bool SaveSoFile(char* data, const uint64_t& len, uint8_t deviceId = 0);

    /**
     * @brief Load function symbols from inner SO
     *
     * Loads the init and run functions from the saved inner SO using dlopen/dlsym.
     */
    void SetTileFwkKernelMap();

    /**
     * @brief Execute a function from the inner SO
     *
     * @param args Arguments to pass to the function
     * @param funcKey Function key (2=init, 3=run)
     * @return Return value from the function, or error code
     */
    int ExecuteFunc(void* args, const uint64_t funcKey);

private:
    /**
     * @brief Load a specific function from the inner SO
     *
     * @param kernelName Name of the function to load (symbol name in inner SO)
     */
    void LoadTileFwkKernelFunc(const std::string& kernelName);

    /**
     * @brief Get a loaded function by its key
     *
     * @param funcKey Function key (2=init, 3=run)
     * @return Function pointer, or nullptr if not found
     */
    AicpuKernelFunc GetTileFwkKernelFunc(const uint64_t funcKey);

    std::unordered_map<uint64_t, AicpuKernelFunc> kernelKey2FuncHandle_;
    std::mutex funcLock_;
    void* soHandle_ = nullptr;
    bool firstLoadSo_ = false;
    std::string innerSoName_;
};

}  // namespace aicpu_dispatcher

// C-style exported functions (AICPU entry points)
extern "C" {
    __attribute__((visibility("default"))) uint32_t DynTileFwkKernelServerNull(void* args);
    __attribute__((visibility("default"))) uint32_t DynTileFwkKernelServerInit(void* args);
    __attribute__((visibility("default"))) uint32_t DynTileFwkKernelServer(void* args);
}

#endif  // COMMON_AICPU_DISPATCHER_AICPU_DISPATCHER_H_
