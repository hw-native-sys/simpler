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

#ifndef SRC_COMMON_WORKER_CHIP_WORKER_H_
#define SRC_COMMON_WORKER_CHIP_WORKER_H_

#include <cstdint>
#include <string>
#include <vector>

#include "dist_types.h"

struct ChipCallConfig {
    int block_dim = 24;
    int aicpu_thread_num = 3;
    bool enable_profiling = false;
};

class ChipWorker : public IWorker {
public:
    ChipWorker() = default;
    ~ChipWorker();

    ChipWorker(const ChipWorker &) = delete;
    ChipWorker &operator=(const ChipWorker &) = delete;

    /// Bind the runtime library and cache platform binaries.
    /// Can only be called once per lifetime — the runtime cannot be changed.
    void init(
        const std::string &host_lib_path, const std::string &aicpu_path, const std::string &aicore_path,
        const std::string &sim_context_lib_path = ""
    );

    /// Set the target NPU device. Requires init() first.
    /// Can be called after reset_device() to switch to a different device.
    void set_device(int device_id);

    /// Release device resources only. The runtime binding remains intact.
    /// After this, set_device() can be called again with a new device ID.
    void reset_device();

    /// Allocate pinned host memory through the bound host runtime.
    uint64_t host_malloc(size_t size);

    /// Free pinned host memory allocated by host_malloc().
    void host_free(uint64_t host_ptr);

    /// Register pinned host memory and return the mapped device-visible address.
    uint64_t host_register_mapped(uint64_t host_ptr, size_t size, int device_id = -1);

    /// Unregister a mapped host buffer.
    void host_unregister_mapped(uint64_t host_ptr, int device_id = -1);

    /// Tear down everything: device resources and runtime library.
    /// Terminal — the object cannot be reused after this.
    void finalize();

    // IWorker: extract callable/args/config from payload and execute synchronously.
    void run(const WorkerPayload &payload) override;

    // Direct invocation (used by Python wrapper and internal tests).
    void run(const void *callable, const void *args, const ChipCallConfig &config);

    int device_id() const { return device_id_; }
    bool initialized() const { return initialized_; }
    bool device_set() const { return device_set_; }

private:
    using CreateDeviceContextFn = void *(*)();
    using DestroyDeviceContextFn = void (*)(void *);
    using SetDeviceFn = int (*)(void *, int);
    using GetRuntimeSizeFn = size_t (*)();
    using RunRuntimeFn = int (*)(
        void *, void *, const void *, const void *, int, int, int, const uint8_t *, size_t, const uint8_t *, size_t, int
    );
    using FinalizeDeviceFn = int (*)(void *);
    using HostMallocFn = void *(*)(void *, size_t);
    using HostFreeFn = void (*)(void *, void *);
    using HostRegisterMappedFn = int (*)(void *, void *, size_t, int, void **);
    using HostUnregisterMappedFn = int (*)(void *, void *, int);

    void *lib_handle_ = nullptr;
    CreateDeviceContextFn create_device_context_fn_ = nullptr;
    DestroyDeviceContextFn destroy_device_context_fn_ = nullptr;
    SetDeviceFn set_device_fn_ = nullptr;
    GetRuntimeSizeFn get_runtime_size_fn_ = nullptr;
    RunRuntimeFn run_runtime_fn_ = nullptr;
    FinalizeDeviceFn finalize_device_fn_ = nullptr;
    HostMallocFn host_malloc_fn_ = nullptr;
    HostFreeFn host_free_fn_ = nullptr;
    HostRegisterMappedFn host_register_mapped_fn_ = nullptr;
    HostUnregisterMappedFn host_unregister_mapped_fn_ = nullptr;
    void *device_ctx_ = nullptr;

    std::vector<uint8_t> runtime_buf_;
    std::vector<uint8_t> aicpu_binary_;
    std::vector<uint8_t> aicore_binary_;
    int device_id_ = -1;
    bool initialized_ = false;
    bool device_set_ = false;
    bool finalized_ = false;
};

#endif  // SRC_COMMON_WORKER_CHIP_WORKER_H_
