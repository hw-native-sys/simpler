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

#include "chip_worker.h"

#include <dlfcn.h>

#include <stdexcept>
#include <string>

namespace {

template <typename T>
T load_symbol(void* handle, const char* name) {
    dlerror();  // clear any existing error
    void* sym = dlsym(handle, name);
    const char* err = dlerror();
    if (err) {
        std::string msg = "dlsym failed for '";
        msg += name;
        msg += "': ";
        msg += err;
        throw std::runtime_error(msg);
    }
    return reinterpret_cast<T>(sym);
}

}  // namespace

ChipWorker::~ChipWorker() { reset(); }

void ChipWorker::init(int device_id,
    const std::string& host_lib_path,
    const uint8_t* aicpu_binary,
    size_t aicpu_size,
    const uint8_t* aicore_binary,
    size_t aicore_size) {
    if (initialized_) {
        throw std::runtime_error("ChipWorker already initialized; call reset() first");
    }

    // Load the host runtime shared library
    void* handle = dlopen(host_lib_path.c_str(), RTLD_NOW | RTLD_GLOBAL);
    if (!handle) {
        std::string err = "dlopen failed: ";
        const char* msg = dlerror();
        err += msg ? msg : "unknown error";
        throw std::runtime_error(err);
    }

    try {
        // Resolve C API symbols
        set_device_fn_ = load_symbol<SetDeviceFn>(handle, "set_device");
        get_runtime_size_fn_ = load_symbol<GetRuntimeSizeFn>(handle, "get_runtime_size");
        init_runtime_fn_ = load_symbol<InitRuntimeFn>(handle, "init_runtime");
        launch_runtime_fn_ = load_symbol<LaunchRuntimeFn>(handle, "launch_runtime");
        finalize_runtime_fn_ = load_symbol<FinalizeRuntimeFn>(handle, "finalize_runtime");
        enable_profiling_fn_ = load_symbol<EnableProfilingFn>(handle, "enable_runtime_profiling");

        // Set device
        int rc = set_device_fn_(device_id);
        if (rc != 0) {
            throw std::runtime_error("set_device failed with code " + std::to_string(rc));
        }
    } catch (...) {
        dlclose(handle);
        throw;
    }

    lib_handle_ = handle;
    device_id_ = device_id;

    // Cache platform binaries
    aicpu_binary_.assign(aicpu_binary, aicpu_binary + aicpu_size);
    aicore_binary_.assign(aicore_binary, aicore_binary + aicore_size);

    // Pre-allocate runtime buffer
    runtime_buf_.resize(get_runtime_size_fn_());

    initialized_ = true;
}

void ChipWorker::reset() {
    if (lib_handle_) {
        dlclose(lib_handle_);
    }
    lib_handle_ = nullptr;
    set_device_fn_ = nullptr;
    get_runtime_size_fn_ = nullptr;
    init_runtime_fn_ = nullptr;
    launch_runtime_fn_ = nullptr;
    finalize_runtime_fn_ = nullptr;
    enable_profiling_fn_ = nullptr;
    runtime_buf_.clear();
    aicpu_binary_.clear();
    aicore_binary_.clear();
    device_id_ = -1;
    initialized_ = false;
}

void ChipWorker::run(const void* callable, const void* args, const CallConfig& config) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }

    void* rt = runtime_buf_.data();

    // 1. Placement new + build graph
    int rc = init_runtime_fn_(rt, callable, args);
    if (rc != 0) {
        throw std::runtime_error("init_runtime failed with code " + std::to_string(rc));
    }

    // 2. Enable profiling AFTER init (placement new would overwrite the flag)
    if (config.enable_profiling) {
        rc = enable_profiling_fn_(rt, 1);
        if (rc != 0) {
            throw std::runtime_error("enable_runtime_profiling failed with code " + std::to_string(rc));
        }
    }

    // 3. Launch
    rc = launch_runtime_fn_(rt,
        config.aicpu_thread_num,
        config.block_dim,
        device_id_,
        aicpu_binary_.data(),
        aicpu_binary_.size(),
        aicore_binary_.data(),
        aicore_binary_.size(),
        config.orch_thread_num);
    if (rc != 0) {
        throw std::runtime_error("launch_runtime failed with code " + std::to_string(rc));
    }

    // 4. Finalize — copy results back, cleanup
    rc = finalize_runtime_fn_(rt);
    if (rc != 0) {
        throw std::runtime_error("finalize_runtime failed with code " + std::to_string(rc));
    }
}
