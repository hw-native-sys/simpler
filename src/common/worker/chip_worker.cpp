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
T load_optional_symbol(void *handle, const char *name) {
    dlerror();
    void *sym = dlsym(handle, name);
    const char *err = dlerror();
    if (err) return nullptr;
    return reinterpret_cast<T>(sym);
}

// Process-wide singleton: libcpu_sim_context.so is loaded once with
// RTLD_GLOBAL so that PTO ISA kernel SOs can find pto_cpu_sim_* symbols
// via dlsym(RTLD_DEFAULT, ...).  Never dlclosed.
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
    // PTO ISA TPUSH/TPOP can resolve pto_cpu_sim_* via dlsym(RTLD_DEFAULT).
    if (!sim_context_lib_path.empty()) {
        ensure_sim_context_loaded(sim_context_lib_path);
    }

    // Host runtime SO is loaded with RTLD_LOCAL so that different runtimes'
    // identically-named symbols (init_runtime_impl, run_runtime, etc.) do
    // not collide when switching runtimes within the same process.
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
        device_malloc_ctx_fn_ = load_symbol<DeviceMallocCtxFn>(handle, "device_malloc_ctx");
        device_free_ctx_fn_ = load_symbol<DeviceFreeCtxFn>(handle, "device_free_ctx");
        copy_to_device_ctx_fn_ = load_symbol<CopyToDeviceCtxFn>(handle, "copy_to_device_ctx");
        copy_from_device_ctx_fn_ = load_symbol<CopyFromDeviceCtxFn>(handle, "copy_from_device_ctx");
        get_runtime_size_fn_ = load_symbol<GetRuntimeSizeFn>(handle, "get_runtime_size");
        run_runtime_fn_ = load_symbol<RunRuntimeFn>(handle, "run_runtime");
        finalize_device_fn_ = load_symbol<FinalizeDeviceFn>(handle, "finalize_device");
        comm_init_fn_ = load_optional_symbol<CommInitFn>(handle, "comm_init");
        comm_alloc_windows_fn_ = load_optional_symbol<CommAllocWindowsFn>(handle, "comm_alloc_windows");
        comm_get_local_window_base_fn_ = load_optional_symbol<CommGetLocalWindowBaseFn>(handle, "comm_get_local_window_base");
        comm_get_window_size_fn_ = load_optional_symbol<CommGetWindowSizeFn>(handle, "comm_get_window_size");
        comm_barrier_fn_ = load_optional_symbol<CommBarrierFn>(handle, "comm_barrier");
        comm_destroy_fn_ = load_optional_symbol<CommDestroyFn>(handle, "comm_destroy");
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
    device_malloc_ctx_fn_ = nullptr;
    device_free_ctx_fn_ = nullptr;
    copy_to_device_ctx_fn_ = nullptr;
    copy_from_device_ctx_fn_ = nullptr;
    get_runtime_size_fn_ = nullptr;
    run_runtime_fn_ = nullptr;
    finalize_device_fn_ = nullptr;
    comm_init_fn_ = nullptr;
    comm_alloc_windows_fn_ = nullptr;
    comm_get_local_window_base_fn_ = nullptr;
    comm_get_window_size_fn_ = nullptr;
    comm_barrier_fn_ = nullptr;
    comm_destroy_fn_ = nullptr;
    runtime_buf_.clear();
    aicpu_binary_.clear();
    aicore_binary_.clear();
    initialized_ = false;
    finalized_ = true;
}

void ChipWorker::run(const WorkerPayload &payload) {
    CallConfig config;
    config.block_dim = payload.block_dim;
    config.aicpu_thread_num = payload.aicpu_thread_num;
    config.enable_profiling = payload.enable_profiling;
    run(payload.callable, payload.args, config);
}

void ChipWorker::run(const void *callable, const void *args, const CallConfig &config) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }

    void *rt = runtime_buf_.data();

    int rc = run_runtime_fn_(
        device_ctx_, rt, callable, args, config.block_dim, config.aicpu_thread_num, device_id_, aicpu_binary_.data(),
        aicpu_binary_.size(), aicore_binary_.data(), aicore_binary_.size(), config.enable_profiling ? 1 : 0
    );
    if (rc != 0) {
        throw std::runtime_error("run_runtime failed with code " + std::to_string(rc));
    }
}

uint64_t ChipWorker::device_malloc(size_t size) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }
    void *ptr = device_malloc_ctx_fn_(device_ctx_, size);
    if (ptr == nullptr) {
        throw std::runtime_error("device_malloc_ctx failed");
    }
    return reinterpret_cast<uint64_t>(ptr);
}

void ChipWorker::device_free(uint64_t dev_ptr) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }
    device_free_ctx_fn_(device_ctx_, reinterpret_cast<void *>(dev_ptr));
}

void ChipWorker::copy_to_device(uint64_t dev_ptr, uint64_t host_ptr, size_t size) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }
    int rc = copy_to_device_ctx_fn_(device_ctx_, reinterpret_cast<void *>(dev_ptr), reinterpret_cast<const void *>(host_ptr), size);
    if (rc != 0) {
        throw std::runtime_error("copy_to_device_ctx failed with code " + std::to_string(rc));
    }
}

void ChipWorker::copy_from_device(uint64_t host_ptr, uint64_t dev_ptr, size_t size) {
    if (!device_set_) {
        throw std::runtime_error("ChipWorker device not set; call set_device() first");
    }
    int rc = copy_from_device_ctx_fn_(device_ctx_, reinterpret_cast<void *>(host_ptr), reinterpret_cast<const void *>(dev_ptr), size);
    if (rc != 0) {
        throw std::runtime_error("copy_from_device_ctx failed with code " + std::to_string(rc));
    }
}

uint64_t ChipWorker::comm_init(int rank, int nranks, int device_id, const std::string &rootinfo_path) {
    if (comm_init_fn_ == nullptr) {
        throw std::runtime_error("comm_init is not available in this runtime");
    }
    void *handle = comm_init_fn_(rank, nranks, device_id, rootinfo_path.c_str());
    if (handle == nullptr) {
        throw std::runtime_error("comm_init failed");
    }
    return reinterpret_cast<uint64_t>(handle);
}

uint64_t ChipWorker::comm_alloc_windows(uint64_t comm_handle, size_t win_size) {
    if (comm_alloc_windows_fn_ == nullptr) {
        throw std::runtime_error("comm_alloc_windows is not available in this runtime");
    }
    uint64_t device_ctx = 0;
    int rc = comm_alloc_windows_fn_(reinterpret_cast<void *>(comm_handle), win_size, &device_ctx);
    if (rc != 0) {
        throw std::runtime_error("comm_alloc_windows failed with code " + std::to_string(rc));
    }
    return device_ctx;
}

uint64_t ChipWorker::comm_get_local_window_base(uint64_t comm_handle) {
    if (comm_get_local_window_base_fn_ == nullptr) {
        throw std::runtime_error("comm_get_local_window_base is not available in this runtime");
    }
    uint64_t base = 0;
    int rc = comm_get_local_window_base_fn_(reinterpret_cast<void *>(comm_handle), &base);
    if (rc != 0) {
        throw std::runtime_error("comm_get_local_window_base failed with code " + std::to_string(rc));
    }
    return base;
}

size_t ChipWorker::comm_get_window_size(uint64_t comm_handle) {
    if (comm_get_window_size_fn_ == nullptr) {
        throw std::runtime_error("comm_get_window_size is not available in this runtime");
    }
    size_t win_size = 0;
    int rc = comm_get_window_size_fn_(reinterpret_cast<void *>(comm_handle), &win_size);
    if (rc != 0) {
        throw std::runtime_error("comm_get_window_size failed with code " + std::to_string(rc));
    }
    return win_size;
}

void ChipWorker::comm_barrier(uint64_t comm_handle) {
    if (comm_barrier_fn_ == nullptr) {
        throw std::runtime_error("comm_barrier is not available in this runtime");
    }
    int rc = comm_barrier_fn_(reinterpret_cast<void *>(comm_handle));
    if (rc != 0) {
        throw std::runtime_error("comm_barrier failed with code " + std::to_string(rc));
    }
}

void ChipWorker::comm_destroy(uint64_t comm_handle) {
    if (comm_destroy_fn_ == nullptr) {
        throw std::runtime_error("comm_destroy is not available in this runtime");
    }
    int rc = comm_destroy_fn_(reinterpret_cast<void *>(comm_handle));
    if (rc != 0) {
        throw std::runtime_error("comm_destroy failed with code " + std::to_string(rc));
    }
}
