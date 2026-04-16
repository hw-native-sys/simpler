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

#include "../task_interface/chip_call_config.h"
#include "../task_interface/task_args.h"
#include "dist_types.h"

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

    /// Allocate host memory and register it as a device-visible mapped buffer.
    void mallocHostDeviceShareMem(uint64_t size, uint64_t *host_ptr, uint64_t *dev_ptr, int device_id = -1);

    /// Unregister and free a mapped host buffer.
    void freeHostDeviceShareMem(uint64_t host_ptr, int device_id = -1);

    /// Tear down everything: device resources and runtime library.
    /// Terminal — the object cannot be reused after this.
    void finalize();

    // IWorker: build a ChipStorageTaskArgs POD from `args` and execute the
    // runtime synchronously. `callable` is a ChipCallable buffer pointer
    // cast to uint64.
    void run(uint64_t callable, TaskArgsView args, const ChipCallConfig &config) override;

    // Direct invocation (used by Python wrapper and internal tests) — bypasses
    // the TaskArgsView path and takes a ready-made ChipStorageTaskArgs POD.
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
        void *, void *, const void *, const void *, int, int, int, const uint8_t *, size_t, const uint8_t *, size_t,
        int, int
    );
    using FinalizeDeviceFn = int (*)(void *);
    using MallocHostDeviceShareMemFn = int (*)(uint32_t, uint64_t, void **, void **);
    using FreeHostDeviceShareMemFn = int (*)(uint32_t, void *);

    void *lib_handle_ = nullptr;
    CreateDeviceContextFn create_device_context_fn_ = nullptr;
    DestroyDeviceContextFn destroy_device_context_fn_ = nullptr;
    SetDeviceFn set_device_fn_ = nullptr;
    GetRuntimeSizeFn get_runtime_size_fn_ = nullptr;
    RunRuntimeFn run_runtime_fn_ = nullptr;
    FinalizeDeviceFn finalize_device_fn_ = nullptr;
    MallocHostDeviceShareMemFn malloc_host_device_share_mem_fn_ = nullptr;
    FreeHostDeviceShareMemFn free_host_device_share_mem_fn_ = nullptr;
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
