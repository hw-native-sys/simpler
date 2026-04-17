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
 * PTO Runtime C API - Implementation (Simulation)
 *
 * Platform-specific implementation of the public C API declared in
 * src/common/worker/pto_runtime_c_api.h.  Uses thread-based simulation.
 */

#include "pto_runtime_c_api.h"

#include "callable.h"
#include "task_args.h"

#include <new>
#include <pthread.h>
#include <vector>

#include "common/unified_log.h"
#include "cpu_sim_context.h"
#include "device_runner.h"
#include "runtime.h"

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in runtime_maker.cpp)
 * =========================================================================== */
int init_runtime_impl(Runtime *runtime, const ChipCallable *callable, const ChipStorageTaskArgs *orch_args);
int validate_runtime_impl(Runtime *runtime);

/* ===========================================================================
 * Per-thread DeviceRunner binding (set by run_runtime, read by HostApi wrappers)
 * =========================================================================== */

static pthread_key_t g_runner_key;
static pthread_once_t g_runner_key_once = PTHREAD_ONCE_INIT;
static void create_runner_key() { pthread_key_create(&g_runner_key, nullptr); }

static DeviceRunner *current_runner() { return static_cast<DeviceRunner *>(pthread_getspecific(g_runner_key)); }

/* ===========================================================================
 * Internal device-memory functions (used via Runtime.host_api, NOT dlsym'd)
 * =========================================================================== */

static void *device_malloc(size_t size) {
    try {
        return current_runner()->allocate_tensor(size);
    } catch (...) {
        return NULL;
    }
}

static void device_free(void *dev_ptr) {
    if (dev_ptr == NULL) return;
    try {
        current_runner()->free_tensor(dev_ptr);
    } catch (...) {}
}

static int copy_to_device(void *dev_ptr, const void *host_ptr, size_t size) {
    if (dev_ptr == NULL || host_ptr == NULL) return -1;
    try {
        return current_runner()->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

static int copy_from_device(void *host_ptr, const void *dev_ptr, size_t size) {
    if (host_ptr == NULL || dev_ptr == NULL) return -1;
    try {
        return current_runner()->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

static uint64_t upload_kernel_binary_wrapper(int func_id, const uint8_t *bin_data, size_t bin_size) {
    try {
        return current_runner()->upload_kernel_binary(func_id, bin_data, bin_size);
    } catch (...) {
        return 0;
    }
}

static void remove_kernel_binary_wrapper(int func_id) {
    try {
        current_runner()->remove_kernel_binary(func_id);
    } catch (...) {}
}

/* ===========================================================================
 * Public C API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

DeviceContextHandle create_device_context(void) {
    try {
        return static_cast<DeviceContextHandle>(new DeviceRunner());
    } catch (...) {
        return NULL;
    }
}

void destroy_device_context(DeviceContextHandle ctx) { delete static_cast<DeviceRunner *>(ctx); }

size_t get_runtime_size(void) { return sizeof(Runtime); }

int set_device(DeviceContextHandle ctx, int device_id) {
    (void)ctx;
    pto_cpu_sim_bind_device(device_id);
    pto_cpu_sim_acquire_device(device_id);
    return 0;
}

int mallocHostDeviceShareMem(uint32_t deviceId, uint64_t size, void **hostPtr, void **devPtr) {
    (void)deviceId;
    (void)size;
    if (hostPtr != nullptr) {
        *hostPtr = nullptr;
    }
    if (devPtr != nullptr) {
        *devPtr = nullptr;
    }
    LOG_ERROR("mallocHostDeviceShareMem is not supported on a5sim");
    return -1;
}

int freeHostDeviceShareMem(uint32_t deviceId, void *hostPtr) {
    (void)deviceId;
    (void)hostPtr;
    LOG_ERROR("freeHostDeviceShareMem is not supported on a5sim");
    return -1;
}

int run_runtime(
    DeviceContextHandle ctx, RuntimeHandle runtime, const void *callable, const void *args, int block_dim,
    int aicpu_thread_num, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size, const uint8_t *aicore_binary,
    size_t aicore_size, int enable_profiling, int enable_dump_tensor
) {
    if (ctx == NULL || runtime == NULL) return -1;

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);

    try {
        // Phase 1: placement new + build graph
        Runtime *r = new (runtime) Runtime();
        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;
        r->host_api.upload_kernel_binary = upload_kernel_binary_wrapper;
        r->host_api.remove_kernel_binary = remove_kernel_binary_wrapper;

        int rc = init_runtime_impl(
            r, reinterpret_cast<const ChipCallable *>(callable), reinterpret_cast<const ChipStorageTaskArgs *>(args)
        );
        if (rc != 0) {
            r->set_pto2_gm_sm_ptr(nullptr);
            validate_runtime_impl(r);
            r->~Runtime();
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        // Phase 2: profiling
        if (enable_profiling) {
            r->enable_profiling = true;
        }

        // Phase 3: launch
        std::vector<uint8_t> aicpu_vec;
        std::vector<uint8_t> aicore_vec;
        if (aicpu_binary != NULL && aicpu_size > 0) {
            aicpu_vec.assign(aicpu_binary, aicpu_binary + aicpu_size);
        }
        if (aicore_binary != NULL && aicore_size > 0) {
            aicore_vec.assign(aicore_binary, aicore_binary + aicore_size);
        }
        rc = runner->run(*r, block_dim, device_id, aicpu_vec, aicore_vec, aicpu_thread_num, enable_dump_tensor != 0);
        if (rc != 0) {
            validate_runtime_impl(r);
            r->~Runtime();
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        // Phase 4: finalize (copy results back)
        rc = validate_runtime_impl(r);
        r->~Runtime();
        pthread_setspecific(g_runner_key, nullptr);
        return rc;
    } catch (...) {
        pthread_setspecific(g_runner_key, nullptr);
        return -1;
    }
}

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        int rc = static_cast<DeviceRunner *>(ctx)->finalize();
        int dev = pto_cpu_sim_get_bound_device();
        if (dev >= 0) {
            pto_cpu_sim_release_device(dev);
        }
        return rc;
    } catch (...) {
        return -1;
    }
}

/* ===========================================================================
 * Internal helpers called from runtime_maker.cpp via Runtime.host_api
 * =========================================================================== */

void record_tensor_pair(RuntimeHandle runtime, void *host_ptr, void *dev_ptr, size_t size) {
    if (runtime == NULL) return;
    Runtime *r = static_cast<Runtime *>(runtime);
    r->record_tensor_pair(host_ptr, dev_ptr, size);
}

}  // extern "C"
