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
 * PTO Runtime C API - Implementation (On-board Hardware)
 *
 * Platform-specific implementation of the public C API declared in
 * src/common/worker/pto_runtime_c_api.h.  Uses real Ascend device execution.
 */

#include "pto_runtime_c_api.h"

#include "callable.h"
#include "task_args.h"

#include <pthread.h>
#include <vector>

#include "common/unified_log.h"
#include "device_runner.h"
#include "host/raii_scope_guard.h"
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
    (void)device_id;
    return 0;
}

int ensure_acl_ready_ctx(DeviceContextHandle ctx, int device_id) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->ensure_acl_ready(device_id);
    } catch (...) {
        return -1;
    }
}

/*
 * Stream creation/destruction exposed so the ChipWorker Python wrapper can
 * drive comm_init end-to-end without leaking aclrtStream lifetime (or ACL
 * libs) into Python.  Both entries go through the DeviceRunner so the ACL
 * ready-flag and device bookkeeping stay consistent with the normal run path.
 */
void *create_comm_stream_ctx(DeviceContextHandle ctx) {
    if (ctx == NULL) return NULL;
    try {
        return static_cast<DeviceRunner *>(ctx)->create_comm_stream();
    } catch (...) {
        return NULL;
    }
}

int destroy_comm_stream_ctx(DeviceContextHandle ctx, void *stream) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->destroy_comm_stream(stream);
    } catch (...) {
        return -1;
    }
}

void *device_malloc_ctx(DeviceContextHandle ctx, size_t size) {
    if (ctx == NULL) return NULL;
    try {
        return static_cast<DeviceRunner *>(ctx)->allocate_tensor(size);
    } catch (...) {
        return NULL;
    }
}

void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr) {
    if (ctx == NULL || dev_ptr == NULL) return;
    try {
        static_cast<DeviceRunner *>(ctx)->free_tensor(dev_ptr);
    } catch (...) {}
}

int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size) {
    if (ctx == NULL || dev_ptr == NULL || host_ptr == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    if (ctx == NULL || host_ptr == NULL || dev_ptr == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

int run_runtime(
    DeviceContextHandle ctx, RuntimeHandle runtime, const void *callable, const void *args, int block_dim,
    int aicpu_thread_num, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size, const uint8_t *aicore_binary,
    size_t aicore_size, int enable_l2_swimlane, int enable_dump_tensor, int enable_pmu, const char *output_prefix
) {
    if (ctx == NULL || runtime == NULL) return -1;
    if (aicpu_binary == NULL || aicpu_size == 0 || aicore_binary == NULL || aicore_size == 0) return -1;

    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    try {
        int rc = runner->prepare_run_context(device_id);
        if (rc != 0) return rc;
        auto run_context_guard = RAIIScopeGuard([runner]() {
            runner->release_run_context();
        });

        Runtime *r = new (runtime) Runtime();
        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;
        r->host_api.upload_kernel_binary = upload_kernel_binary_wrapper;
        r->host_api.remove_kernel_binary = remove_kernel_binary_wrapper;

        LOG_DEBUG("About to call init_runtime_impl, r=%p", (void *)r);
        rc = init_runtime_impl(
            r, reinterpret_cast<const ChipCallable *>(callable), reinterpret_cast<const ChipStorageTaskArgs *>(args)
        );
        LOG_DEBUG("init_runtime_impl returned: %d", rc);
        if (rc != 0) {
            r->set_gm_sm_ptr(nullptr);
            validate_runtime_impl(r);
            r->~Runtime();
            return rc;
        }

        runner->set_l2_swimlane_enabled(enable_l2_swimlane != 0);
        runner->set_dump_tensor_enabled(enable_dump_tensor != 0);
        runner->set_pmu_enabled(enable_pmu);
        runner->set_output_prefix(output_prefix);

        std::vector<uint8_t> aicpu_vec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicore_vec(aicore_binary, aicore_binary + aicore_size);
        rc = runner->run(*r, block_dim, device_id, aicpu_vec, aicore_vec, aicpu_thread_num);
        if (rc != 0) {
            validate_runtime_impl(r);
            r->~Runtime();
            return rc;
        }

        rc = validate_runtime_impl(r);
        r->~Runtime();
        return rc;
    } catch (...) {
        return -1;
    }
}

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->finalize();
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
