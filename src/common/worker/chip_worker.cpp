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

#include <fstream>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace {

template <typename T>
T load_symbol(void *handle, const char *name) {
    dlerror();  // clear any existing error
    void *sym = dlsym(handle, name);
    const char *err = dlerror();
    if (err) {
        std::string msg = "dlsym failed for '";
        msg += name;
        msg += "': ";
        msg += err;
        throw std::runtime_error(msg);
    }
    return reinterpret_cast<T>(sym);
}

template <typename T>
T try_load_symbol(void *handle, const char *name) {
    dlerror();  // clear any existing error
    void *sym = dlsym(handle, name);
    (void)dlerror();
    return reinterpret_cast<T>(sym);
}

// Process-wide singleton: libcpu_sim_context.so is loaded once with
// RTLD_GLOBAL so that host_runtime.so can resolve sim_context_set_* and
// pto_sim_get_* symbols at runtime.  Never dlclosed.
std::once_flag g_sim_context_once;
void *g_sim_context_handle = nullptr;

void ensure_sim_context_loaded(const std::string &path) {
    std::call_once(g_sim_context_once, [&]() {
        dlerror();
        g_sim_context_handle = dlopen(path.c_str(), RTLD_NOW | RTLD_GLOBAL);
        if (!g_sim_context_handle) {
            std::string err = "dlopen sim_context failed: ";
            const char *msg = dlerror();
            err += msg ? msg : "unknown error";
            throw std::runtime_error(err);
        }
    });
}

std::vector<uint8_t> read_binary_file(const std::string &path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) {
        throw std::runtime_error("Failed to open binary file: " + path);
    }
    auto size = f.tellg();
    if (size < 0) {
        throw std::runtime_error("Failed to determine size of binary file: " + path);
    }
    std::vector<uint8_t> buf(static_cast<size_t>(size));
    f.seekg(0);
    if (size > 0 && !f.read(reinterpret_cast<char *>(buf.data()), size)) {
        throw std::runtime_error("Failed to read binary file: " + path);
    }
    return buf;
}

}  // namespace

ChipWorker::~ChipWorker() { finalize(); }

void ChipWorker::init(
    const std::string &host_lib_path, const std::string &aicpu_path, const std::string &aicore_path,
    const std::string &sim_context_lib_path
) {
    if (finalized_) {
        throw std::runtime_error("ChipWorker already finalized; cannot reinitialize");
    }
    if (initialized_) {
        throw std::runtime_error("ChipWorker already initialized; runtime cannot be changed");
    }

    // Load the sim context SO with RTLD_GLOBAL (once per process) so that
    // PTO ISA TPUSH/TPOP can resolve pto_sim_get_subblock_id and
    // pto_sim_get_pipe_shared_state via dlsym(RTLD_DEFAULT).
    if (!sim_context_lib_path.empty()) {
        ensure_sim_context_loaded(sim_context_lib_path);
    }

    // Host runtime SO is loaded with RTLD_LOCAL so that different runtimes'
    // identically-named symbols (init_runtime_impl, run_runtime, etc.) do
    // not collide when switching runtimes within the same process.
    // Cross-runtime isolation relies on -fno-gnu-unique (#453) allowing
    // dlclose to actually unload the previous runtime's SO before loading
    // the next one.
    dlerror();
    void *handle = dlopen(host_lib_path.c_str(), RTLD_NOW | RTLD_LOCAL);
    if (!handle) {
        std::string err = "dlopen failed: ";
        const char *msg = dlerror();
        err += msg ? msg : "unknown error";
        throw std::runtime_error(err);
    }

    try {
        create_device_context_fn_ = load_symbol<CreateDeviceContextFn>(handle, "create_device_context");
        destroy_device_context_fn_ = load_symbol<DestroyDeviceContextFn>(handle, "destroy_device_context");
        set_device_fn_ = load_symbol<SetDeviceFn>(handle, "set_device");
        get_runtime_size_fn_ = load_symbol<GetRuntimeSizeFn>(handle, "get_runtime_size");
        run_runtime_fn_ = load_symbol<RunRuntimeFn>(handle, "run_runtime");
        finalize_device_fn_ = load_symbol<FinalizeDeviceFn>(handle, "finalize_device");
        host_malloc_fn_ = try_load_symbol<HostMallocFn>(handle, "host_malloc");
        host_free_fn_ = try_load_symbol<HostFreeFn>(handle, "host_free");
        host_register_mapped_fn_ = try_load_symbol<HostRegisterMappedFn>(handle, "host_register_mapped");
        host_unregister_mapped_fn_ = try_load_symbol<HostUnregisterMappedFn>(handle, "host_unregister_mapped");
        malloc_host_device_share_mem_fn_ =
            try_load_symbol<MallocHostDeviceShareMemFn>(handle, "mallocHostDeviceShareMem");
        free_host_device_share_mem_fn_ = try_load_symbol<FreeHostDeviceShareMemFn>(handle, "freeHostDeviceShareMem");
    } catch (...) {
        dlclose(handle);
        throw;
    }

    lib_handle_ = handle;

    device_ctx_ = create_device_context_fn_();
    if (device_ctx_ == nullptr) {
        dlclose(handle);
        lib_handle_ = nullptr;
        throw std::runtime_error("create_device_context returned null");
    }

    // Read platform binaries from files
    aicpu_binary_ = read_binary_file(aicpu_path);
    aicore_binary_ = read_binary_file(aicore_path);

    runtime_buf_.resize(get_runtime_size_fn_());

    initialized_ = true;
}

void ChipWorker::set_device(int device_id) {
    if (!initialized_) {
        throw std::runtime_error("ChipWorker not initialized; call init() first");
    }
    if (device_set_) {
        throw std::runtime_error("Device already set; call reset_device() before switching devices");
    }

    int rc = set_device_fn_(device_ctx_, device_id);
    if (rc != 0) {
        throw std::runtime_error("set_device failed with code " + std::to_string(rc));
    }
    device_id_ = device_id;
    device_set_ = true;
}

void ChipWorker::reset_device() {
    if (device_set_ && finalize_device_fn_) {
        finalize_device_fn_(device_ctx_);
    }
    device_id_ = -1;
    device_set_ = false;
}

uint64_t ChipWorker::host_malloc(size_t size) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }
    if (host_malloc_fn_ == nullptr) {
        throw std::runtime_error("host_malloc symbol is not available in the bound runtime");
    }
    void *ptr = host_malloc_fn_(device_ctx_, size);
    if (ptr == nullptr) {
        throw std::runtime_error("host_malloc returned null");
    }
    return reinterpret_cast<uint64_t>(ptr);
}

void ChipWorker::host_free(uint64_t host_ptr) {
    if (host_ptr == 0) {
        return;
    }
    if (host_free_fn_ == nullptr) {
        throw std::runtime_error("host_free symbol is not available in the bound runtime");
    }
    host_free_fn_(device_ctx_, reinterpret_cast<void *>(host_ptr));
}

uint64_t ChipWorker::host_register_mapped(uint64_t host_ptr, size_t size, int device_id) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }
    if (host_register_mapped_fn_ == nullptr) {
        throw std::runtime_error("host_register_mapped symbol is not available in the bound runtime");
    }
    int effective_device_id = (device_id >= 0) ? device_id : device_id_;
    void *dev_ptr = nullptr;
    int rc = host_register_mapped_fn_(device_ctx_, reinterpret_cast<void *>(host_ptr), size, effective_device_id, &dev_ptr);
    if (rc != 0 || dev_ptr == nullptr) {
        throw std::runtime_error("host_register_mapped failed with code " + std::to_string(rc));
    }
    return reinterpret_cast<uint64_t>(dev_ptr);
}

void ChipWorker::host_unregister_mapped(uint64_t host_ptr, int device_id) {
    if (host_ptr == 0) {
        return;
    }
    if (host_unregister_mapped_fn_ == nullptr) {
        throw std::runtime_error("host_unregister_mapped symbol is not available in the bound runtime");
    }
    int effective_device_id = (device_id >= 0) ? device_id : device_id_;
    int rc = host_unregister_mapped_fn_(device_ctx_, reinterpret_cast<void *>(host_ptr), effective_device_id);
    if (rc != 0) {
        throw std::runtime_error("host_unregister_mapped failed with code " + std::to_string(rc));
    }
}

void ChipWorker::mallocHostDeviceShareMem(uint64_t size, uint64_t *host_ptr, uint64_t *dev_ptr, int device_id) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }
    if (host_ptr == nullptr || dev_ptr == nullptr) {
        throw std::runtime_error("mallocHostDeviceShareMem requires non-null output pointers");
    }
    if (malloc_host_device_share_mem_fn_ == nullptr) {
        throw std::runtime_error("mallocHostDeviceShareMem symbol is not available in the bound runtime");
    }

    *host_ptr = 0;
    *dev_ptr = 0;
    int effective_device_id = (device_id >= 0) ? device_id : device_id_;
    void *host_ptr_raw = nullptr;
    void *dev_ptr_raw = nullptr;
    int rc = malloc_host_device_share_mem_fn_(static_cast<uint32_t>(effective_device_id), size, &host_ptr_raw, &dev_ptr_raw);
    if (rc != 0 || host_ptr_raw == nullptr || dev_ptr_raw == nullptr) {
        throw std::runtime_error("mallocHostDeviceShareMem failed with code " + std::to_string(rc));
    }

    *host_ptr = reinterpret_cast<uint64_t>(host_ptr_raw);
    *dev_ptr = reinterpret_cast<uint64_t>(dev_ptr_raw);
}

void ChipWorker::freeHostDeviceShareMem(uint64_t host_ptr, int device_id) {
    if (host_ptr == 0) {
        return;
    }
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }
    if (free_host_device_share_mem_fn_ == nullptr) {
        throw std::runtime_error("freeHostDeviceShareMem symbol is not available in the bound runtime");
    }

    int effective_device_id = (device_id >= 0) ? device_id : device_id_;
    int rc = free_host_device_share_mem_fn_(static_cast<uint32_t>(effective_device_id), reinterpret_cast<void *>(host_ptr));
    if (rc != 0) {
        throw std::runtime_error("freeHostDeviceShareMem failed with code " + std::to_string(rc));
    }
}

void ChipWorker::finalize() {
    reset_device();
    if (device_ctx_ != nullptr && destroy_device_context_fn_ != nullptr) {
        destroy_device_context_fn_(device_ctx_);
        device_ctx_ = nullptr;
    }
    if (lib_handle_) {
        dlclose(lib_handle_);
    }
    lib_handle_ = nullptr;
    create_device_context_fn_ = nullptr;
    destroy_device_context_fn_ = nullptr;
    set_device_fn_ = nullptr;
    get_runtime_size_fn_ = nullptr;
    run_runtime_fn_ = nullptr;
    finalize_device_fn_ = nullptr;
    host_malloc_fn_ = nullptr;
    host_free_fn_ = nullptr;
    host_register_mapped_fn_ = nullptr;
    host_unregister_mapped_fn_ = nullptr;
    malloc_host_device_share_mem_fn_ = nullptr;
    free_host_device_share_mem_fn_ = nullptr;
    runtime_buf_.clear();
    aicpu_binary_.clear();
    aicore_binary_.clear();
    initialized_ = false;
    finalized_ = true;
}

void ChipWorker::run(uint64_t callable, TaskArgsView args, const ChipCallConfig &config) {
    // L2 ABI edge: assemble the fixed-size ChipStorageTaskArgs POD from the
    // view and hand it to the runtime. This conversion used to happen at
    // submit time (stored on the slot); it now runs lazily in the worker so
    // the slot can carry a single TaskArgs irrespective of the destination.
    ChipStorageTaskArgs chip_storage = view_to_chip_storage(args);
    run(reinterpret_cast<const void *>(callable), &chip_storage, config);
}

void ChipWorker::run(const void *callable, const void *args, const ChipCallConfig &config) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }

    void *rt = runtime_buf_.data();

    int rc = run_runtime_fn_(
        device_ctx_, rt, callable, args, config.block_dim, config.aicpu_thread_num, device_id_, aicpu_binary_.data(),
        aicpu_binary_.size(), aicore_binary_.data(), aicore_binary_.size(), config.enable_profiling ? 1 : 0,
        config.enable_dump_tensor ? 1 : 0
    );
    if (rc != 0) {
        throw std::runtime_error("run_runtime failed with code " + std::to_string(rc));
    }
}
