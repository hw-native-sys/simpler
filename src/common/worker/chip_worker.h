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

struct CallConfig {
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

    /// Tear down everything: device resources and runtime library.
    /// Terminal — the object cannot be reused after this.
    void finalize();

    // IWorker: extract callable/args/config from payload and execute synchronously.
    void run(const WorkerPayload &payload) override;

    // Direct invocation (used by Python wrapper and internal tests).
    void run(const void *callable, const void *args, const CallConfig &config);

    uint64_t device_malloc(size_t size);
    void device_free(uint64_t dev_ptr);
    void copy_to_device(uint64_t dev_ptr, uint64_t host_ptr, size_t size);
    void copy_from_device(uint64_t host_ptr, uint64_t dev_ptr, size_t size);

    uint64_t comm_init(int rank, int nranks, int device_id, const std::string &rootinfo_path);
    uint64_t comm_alloc_windows(uint64_t comm_handle, size_t win_size);
    uint64_t comm_get_local_window_base(uint64_t comm_handle);
    size_t comm_get_window_size(uint64_t comm_handle);
    void comm_barrier(uint64_t comm_handle);
    void comm_destroy(uint64_t comm_handle);

    int device_id() const { return device_id_; }
    bool initialized() const { return initialized_; }
    bool device_set() const { return device_set_; }

private:
    using CreateDeviceContextFn = void *(*)();
    using DestroyDeviceContextFn = void (*)(void *);
    using SetDeviceFn = int (*)(void *, int);
    using DeviceMallocCtxFn = void *(*)(void *, size_t);
    using DeviceFreeCtxFn = void (*)(void *, void *);
    using CopyToDeviceCtxFn = int (*)(void *, void *, const void *, size_t);
    using CopyFromDeviceCtxFn = int (*)(void *, void *, const void *, size_t);
    using GetRuntimeSizeFn = size_t (*)();
    using RunRuntimeFn = int (*)(
        void *, void *, const void *, const void *, int, int, int, const uint8_t *, size_t, const uint8_t *, size_t, int
    );
    using FinalizeDeviceFn = int (*)(void *);
    using CommInitFn = void *(*)(int, int, int, const char *);
    using CommAllocWindowsFn = int (*)(void *, size_t, uint64_t *);
    using CommGetLocalWindowBaseFn = int (*)(void *, uint64_t *);
    using CommGetWindowSizeFn = int (*)(void *, size_t *);
    using CommBarrierFn = int (*)(void *);
    using CommDestroyFn = int (*)(void *);

    void *lib_handle_ = nullptr;
    CreateDeviceContextFn create_device_context_fn_ = nullptr;
    DestroyDeviceContextFn destroy_device_context_fn_ = nullptr;
    SetDeviceFn set_device_fn_ = nullptr;
    DeviceMallocCtxFn device_malloc_ctx_fn_ = nullptr;
    DeviceFreeCtxFn device_free_ctx_fn_ = nullptr;
    CopyToDeviceCtxFn copy_to_device_ctx_fn_ = nullptr;
    CopyFromDeviceCtxFn copy_from_device_ctx_fn_ = nullptr;
    GetRuntimeSizeFn get_runtime_size_fn_ = nullptr;
    RunRuntimeFn run_runtime_fn_ = nullptr;
    FinalizeDeviceFn finalize_device_fn_ = nullptr;
    CommInitFn comm_init_fn_ = nullptr;
    CommAllocWindowsFn comm_alloc_windows_fn_ = nullptr;
    CommGetLocalWindowBaseFn comm_get_local_window_base_fn_ = nullptr;
    CommGetWindowSizeFn comm_get_window_size_fn_ = nullptr;
    CommBarrierFn comm_barrier_fn_ = nullptr;
    CommDestroyFn comm_destroy_fn_ = nullptr;
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
