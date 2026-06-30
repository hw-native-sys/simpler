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
 * Onboard host common helpers — `KernelArgsHelper` implementation.
 *
 * Linked into both a2a3 and a5 `libhost_runtime.so`. The arch-specific
 * `KernelArgs` layout is brought in via `common/kernel_args.h` on the
 * include path (each arch CMake adds the right one).
 */

#include "device_runner_helpers.h"

#include <runtime/rt.h>

#include "common/unified_log.h"

int KernelArgsHelper::init_runtime_args(const Runtime &host_runtime, MemoryAllocator &allocator) {
    allocator_ = &allocator;

    // Only the device-read prefix of Runtime crosses to the device: trb copies
    // its `dev` descriptor (offset 0), hbg copies the whole object. Both start
    // at &host_runtime; runtime_device_copy_size() picks the right length per
    // runtime variant so this shared path stays runtime-agnostic.
    const uint64_t runtime_size = runtime_device_copy_size(host_runtime);
    if (args.runtime_args == nullptr) {
        void *runtime_dev = allocator_->alloc(runtime_size);
        if (runtime_dev == nullptr) {
            LOG_ERROR("Alloc for runtime_args failed");
            return -1;
        }
        args.runtime_args = reinterpret_cast<Runtime *>(runtime_dev);
    }
    int rc = rtMemcpy(args.runtime_args, runtime_size, &host_runtime, runtime_size, RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy for runtime failed: %d", rc);
        allocator_->free(args.runtime_args);
        args.runtime_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::finalize_runtime_args() {
    if (args.runtime_args != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(args.runtime_args);
        args.runtime_args = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::init_device_kernel_args(MemoryAllocator &allocator) {
    allocator_ = &allocator;
    if (device_k_args_ == nullptr) {
        void *dev_ptr = allocator_->alloc(sizeof(KernelArgs));
        if (dev_ptr == nullptr) {
            LOG_ERROR("Alloc for device KernelArgs failed");
            return -1;
        }
        device_k_args_ = reinterpret_cast<KernelArgs *>(dev_ptr);
    }
    int rc = rtMemcpy(device_k_args_, sizeof(KernelArgs), &args, sizeof(KernelArgs), RT_MEMCPY_HOST_TO_DEVICE);
    if (rc != 0) {
        LOG_ERROR("rtMemcpy for KernelArgs failed: %d", rc);
        allocator_->free(device_k_args_);
        device_k_args_ = nullptr;
        return rc;
    }
    return 0;
}

int KernelArgsHelper::finalize_device_kernel_args() {
    if (device_k_args_ != nullptr && allocator_ != nullptr) {
        int rc = allocator_->free(device_k_args_);
        device_k_args_ = nullptr;
        return rc;
    }
    return 0;
}
