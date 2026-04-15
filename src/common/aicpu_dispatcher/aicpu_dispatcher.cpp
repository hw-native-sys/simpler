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
 * AICPU Dispatcher SO Implementation
 */

#include "aicpu_dispatcher.h"

#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

// Weak symbol fallback implementations for unified_log_* functions.
// When dispatcher SO is loaded independently by the AICPU scheduler daemon
// (via dlopen), these weak symbols provide a minimal stderr-based logger.
// When linked into host_runtime.so, the strong symbols from unified_log_host.cpp
// take precedence.
extern "C" {

__attribute__((weak)) void unified_log_error(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ERROR] [%s] ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

__attribute__((weak)) void unified_log_warn(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[WARN] [%s] ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

__attribute__((weak)) void unified_log_info(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[INFO] [%s] ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

__attribute__((weak)) void unified_log_debug(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[DEBUG] [%s] ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

__attribute__((weak)) void unified_log_always(const char* func, const char* fmt, ...) {
    va_list args;
    va_start(args, fmt);
    fprintf(stderr, "[ALWAYS] [%s] ", func);
    vfprintf(stderr, fmt, args);
    fprintf(stderr, "\n");
    va_end(args);
}

}  // extern "C"

// Forward declarations for simpler's KernelArgs and DeviceArgs structures.
// These MUST match the layouts defined in platform-specific kernel_args.h:
//   src/a2a3/platform/include/common/kernel_args.h
//   src/a5/platform/include/common/kernel_args.h
//
// Both platforms share the same layout for fields accessed here (device_args,
// runtime_args). a2a3 has an additional ffts_base_addr field at the end which
// this code does not access. The static_assert below ensures this struct is
// at least as large as the minimum platform layout.
struct KernelArgs {
    uint64_t unused[5] = {0};
    void* device_args{nullptr};  // Pointer to DeviceArgs in device memory
    void* runtime_args{nullptr};
    uint64_t regs{0};
};

// DeviceArgs structure as passed from DeviceRunner.
// Must match the layout in platform-specific DeviceArgs (host/device_runner.h).
struct DeviceArgs {
    uint64_t unused[12] = {0};
    uint64_t aicpu_so_bin{0};
    uint64_t aicpu_so_len{0};
};

static_assert(sizeof(KernelArgs) >= 64, "KernelArgs layout mismatch with platform kernel_args.h");
static_assert(sizeof(DeviceArgs) >= 112, "DeviceArgs layout mismatch with platform DeviceArgs");
static_assert(offsetof(KernelArgs, device_args) == 40, "KernelArgs::device_args offset mismatch");
static_assert(offsetof(KernelArgs, runtime_args) == 48, "KernelArgs::runtime_args offset mismatch");
static_assert(offsetof(DeviceArgs, aicpu_so_bin) == 96, "DeviceArgs::aicpu_so_bin offset mismatch");
static_assert(offsetof(DeviceArgs, aicpu_so_len) == 104, "DeviceArgs::aicpu_so_len offset mismatch");

namespace aicpu_dispatcher {

BackendServerHandleManager::~BackendServerHandleManager()
{
    if (soHandle_ != nullptr) {
        LOG_INFO("Closing inner SO handle: %s", innerSoName_.c_str());
        dlclose(soHandle_);
        soHandle_ = nullptr;
    }
}

bool BackendServerHandleManager::SaveSoFile(char* data, const uint64_t& len, uint8_t deviceId)
{
    std::lock_guard<std::mutex> lock(funcLock_);

    if (len < 1) {
        LOG_WARN("AICPU SO len is %lu, skipping save", len);
        return true;  // Don't fail for empty SO
    }

    // Generate inner SO file path based on device ID
    // Use /tmp/aicpu_kernels/ for better portability (no root requirement)
    const std::string dir_path = "/tmp/aicpu_kernels";
    innerSoName_ = dir_path + "/libaicpu_dispatcher_runtime_" + std::to_string(deviceId) + ".so";

    // Create directory if it doesn't exist
    struct stat st;
    if (stat(dir_path.c_str(), &st) != 0) {
        // Directory doesn't exist, create it
        if (mkdir(dir_path.c_str(), 0755) != 0) {
            LOG_ERROR("Failed to create directory %s: %s", dir_path.c_str(), strerror(errno));
            return false;
        }
        LOG_INFO("Created directory: %s", dir_path.c_str());
    }

    LOG_INFO("Saving inner AICPU SO to device %u: %s (size=%lu bytes)", deviceId, innerSoName_.c_str(), len);

    std::ofstream file(innerSoName_, std::ios::out | std::ios::binary);
    if (!file.is_open()) {
        LOG_ERROR("Failed to create inner SO file: %s", innerSoName_.c_str());
        return false;
    }

    // Write binary to file
    file.write(data, len);

    if (!file.good()) {
        LOG_ERROR("Failed to write inner SO file: %s", innerSoName_.c_str());
        file.close();
        return false;
    }
    file.close();

    LOG_INFO("Successfully saved inner AICPU SO for device %u: %s", deviceId, innerSoName_.c_str());
    return true;
}

void BackendServerHandleManager::SetTileFwkKernelMap()
{
    std::lock_guard<std::mutex> lock(funcLock_);

    if (firstLoadSo_) {
        return;  // Already loaded
    }

    // Load init function from inner SO
    (void)LoadTileFwkKernelFunc(DY_TILE_FWK_BACKEND_KERNEL_SERVER_INIT);
    // Load run function from inner SO
    (void)LoadTileFwkKernelFunc(DY_TILE_FWK_BACKEND_KERNEL_SERVER);

    firstLoadSo_ = true;
}

int BackendServerHandleManager::ExecuteFunc(void* args, const uint64_t funcKey)
{
    auto func = GetTileFwkKernelFunc(funcKey);
    if (func == nullptr) {
        LOG_ERROR("Function key %lu not found in inner SO %s", funcKey, innerSoName_.c_str());
        return -1;
    }

    return func(args);
}

void BackendServerHandleManager::LoadTileFwkKernelFunc(const std::string& kernelName)
{
    if (soHandle_ == nullptr) {
        soHandle_ = dlopen(innerSoName_.c_str(), RTLD_LAZY | RTLD_DEEPBIND);
        if (soHandle_ == nullptr) {
            char* error = dlerror();
            LOG_ERROR("Failed to dlopen inner SO %s: %s", innerSoName_.c_str(), error ? error : "unknown error");
            return;
        }
        LOG_INFO("Successfully dlopened inner SO: %s", innerSoName_.c_str());
    }

    // Map kernel name to function key
    uint64_t funcKey = 0;
    if (kernelName == DY_TILE_FWK_BACKEND_KERNEL_SERVER_INIT) {
        funcKey = dyInitFuncKey;
    } else if (kernelName == DY_TILE_FWK_BACKEND_KERNEL_SERVER) {
        funcKey = dyExecFuncKey;
    } else {
        LOG_ERROR("Unknown kernel name: %s", kernelName.c_str());
        return;
    }

    LOG_DEBUG("Loading function: name=%s, funcKey=%lu", kernelName.c_str(), funcKey);

    // Skip if function is already loaded
    auto iter = kernelKey2FuncHandle_.find(funcKey);
    if (iter != kernelKey2FuncHandle_.end()) {
        LOG_DEBUG("Function already loaded: %s (funcKey=%lu)", kernelName.c_str(), funcKey);
        return;
    }

    // Load the function
    AicpuKernelFunc funcEntry = reinterpret_cast<AicpuKernelFunc>(
        dlsym(soHandle_, kernelName.c_str())
    );
    if (funcEntry == nullptr) {
        char* error = dlerror();
        LOG_ERROR("Failed to dlsym %s from %s: %s",
                 kernelName.c_str(), innerSoName_.c_str(), error ? error : "unknown error");
        (void)dlclose(soHandle_);
        soHandle_ = nullptr;
        return;
    }
    LOG_INFO("Successfully loaded function: %s from %s", kernelName.c_str(), innerSoName_.c_str());
    kernelKey2FuncHandle_[funcKey] = funcEntry;
}

AicpuKernelFunc BackendServerHandleManager::GetTileFwkKernelFunc(const uint64_t funcKey)
{
    auto iter = kernelKey2FuncHandle_.find(funcKey);
    if (iter != kernelKey2FuncHandle_.end()) {
        return iter->second;
    }
    LOG_ERROR("Function key %lu not found", funcKey);
    return nullptr;
}

}  // namespace aicpu_dispatcher

namespace {

// Global instance of the handle manager
aicpu_dispatcher::BackendServerHandleManager g_handleManager;

}  // namespace

// C-style exported functions (AICPU entry points)
extern "C" {

__attribute__((visibility("default"))) uint32_t DynTileFwkKernelServerNull(void* args)
{
    if (args == nullptr) {
        LOG_ERROR("Dispatcher Load: args is null");
        return 1;
    }

    auto* kargs = reinterpret_cast<struct KernelArgs*>(args);
    auto* devArgs = reinterpret_cast<struct DeviceArgs*>(kargs->device_args);
    if (devArgs == nullptr) {
        LOG_ERROR("Dispatcher Load: DeviceArgs is null");
        return 1;
    }

    auto* data = reinterpret_cast<char*>(devArgs->aicpu_so_bin);
    if (devArgs->aicpu_so_len == 0) {
        LOG_WARN("Dispatcher Load: inner SO binary is empty, skipping load");
        return 0;
    }

    if (!g_handleManager.SaveSoFile(data, devArgs->aicpu_so_len)) {
        LOG_ERROR("Dispatcher Load: failed to save inner SO");
        return 1;
    }
    g_handleManager.SetTileFwkKernelMap();
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynTileFwkKernelServerInit(void* args)
{
    auto ret = g_handleManager.ExecuteFunc(args, aicpu_dispatcher::dyInitFuncKey);
    if (ret != 0) {
        LOG_ERROR("Dispatcher Init: inner SO init failed with code %d", ret);
        return 1;
    }
    return 0;
}

__attribute__((visibility("default"))) uint32_t DynTileFwkKernelServer(void* args)
{
    auto ret = g_handleManager.ExecuteFunc(args, aicpu_dispatcher::dyExecFuncKey);
    if (ret != 0) {
        LOG_ERROR("Dispatcher Run: inner SO run failed with code %d", ret);
        return 1;
    }
    return 0;
}

}  // extern "C"
