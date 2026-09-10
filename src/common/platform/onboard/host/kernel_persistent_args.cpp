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
 * PersistentKernelArgs implementation.
 *
 * Linked into both a2a3 and a5 `libhost_runtime.so`, once per runtime variant:
 * the `KernelArgs` layout and the `Runtime` device-image length both come from
 * the include path the arch/runtime CMake sets up.
 */

#include "kernel_persistent_args.h"

#include "common/unified_log.h"
#include "runtime_c_api.h"

int PersistentKernelArgs::prepare_once(const Runtime &host_runtime, const PersistentArgsOps &ops, uint64_t device_id) {
    if (prepared_) return 0;
    if (!ops.valid()) {
        LOG_ERROR("PersistentKernelArgs::prepare_once: incomplete operation table");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    ops_ = ops;

    // Only the device-read prefix of Runtime crosses to the device: trb copies
    // its `dev` descriptor (offset 0), hbg copies the whole object.
    const size_t runtime_bytes = runtime_device_copy_size(host_runtime);
    void *runtime_dev = ops_.alloc(ops_.context, runtime_bytes);
    if (runtime_dev == nullptr) {
        LOG_ERROR("PersistentKernelArgs::prepare_once: alloc for runtime_args failed");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    int rc = ops_.copy_h2d(ops_.context, runtime_dev, runtime_bytes, &host_runtime, runtime_bytes);
    if (rc != 0) {
        LOG_ERROR("PersistentKernelArgs::prepare_once: copy of runtime_args failed: %d", rc);
        (void)ops_.free_(ops_.context, runtime_dev);
        return rc;
    }
    args_.runtime_args = reinterpret_cast<Runtime *>(runtime_dev);

    rc = ops_.fill_arch_fields(ops_.context, &args_, device_id);
    if (rc != 0) {
        LOG_ERROR("PersistentKernelArgs::prepare_once: arch field init failed: %d", rc);
        (void)ops_.free_(ops_.context, runtime_dev);
        args_ = KernelArgs{};
        return rc;
    }

    // Taken last: this is a whole-struct copy of `args_`, so every field the
    // device reads — including the two blocks above — must already be set.
    void *device_args = ops_.alloc(ops_.context, sizeof(KernelArgs));
    if (device_args == nullptr) {
        LOG_ERROR("PersistentKernelArgs::prepare_once: alloc for device KernelArgs failed");
        rc = PTO_RUNTIME_ERR_INTERNAL;
    } else {
        rc = ops_.copy_h2d(ops_.context, device_args, sizeof(KernelArgs), &args_, sizeof(KernelArgs));
        if (rc != 0) {
            LOG_ERROR("PersistentKernelArgs::prepare_once: copy of device KernelArgs failed: %d", rc);
            (void)ops_.free_(ops_.context, device_args);
        }
    }
    if (rc != 0) {
        if (args_.regs != 0) (void)ops_.free_(ops_.context, reinterpret_cast<void *>(args_.regs));
        (void)ops_.free_(ops_.context, runtime_dev);
        args_ = KernelArgs{};
        return rc;
    }

    device_k_args_ = reinterpret_cast<KernelArgs *>(device_args);
    prepared_ = true;
    return 0;
}

int PersistentKernelArgs::release_block(void *block, int &first_error) {
    const int rc = ops_.free_(ops_.context, block);
    if (rc != 0 && first_error == 0) first_error = rc;
    return rc;
}

int PersistentKernelArgs::finalize_once() {
    if (ops_.free_ == nullptr) {
        device_k_args_ = nullptr;
        args_ = KernelArgs{};
        prepared_ = false;
        return 0;
    }

    int first_error = 0;
    if (device_k_args_ != nullptr && release_block(device_k_args_, first_error) == 0) {
        device_k_args_ = nullptr;
    }
    if (args_.regs != 0 && release_block(reinterpret_cast<void *>(args_.regs), first_error) == 0) {
        args_.regs = 0;
    }
    if (args_.runtime_args != nullptr && release_block(args_.runtime_args, first_error) == 0) {
        args_.runtime_args = nullptr;
    }
    if (first_error != 0) return first_error;

    args_ = KernelArgs{};
    prepared_ = false;
    return 0;
}

void PersistentKernelArgs::abandon() {
    args_ = KernelArgs{};
    device_k_args_ = nullptr;
    ops_ = {};
    prepared_ = false;
}
