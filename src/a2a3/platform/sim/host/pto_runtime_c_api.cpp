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
#include "prepare_callable_common.h"
#include "task_args.h"

#include <new>
#include <pthread.h>

#include <utility>
#include <vector>

#include "common/unified_log.h"
#include "host_log.h"
#include "cpu_sim_context.h"
#include "device_runner.h"
#include "runtime.h"

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in runtime_maker.cpp)
 * =========================================================================== */
int prepare_callable_impl(
    const ChipCallable *callable, uint64_t (*upload_fn)(const void *), PreparedCallableArtifacts *out
);
int bind_prepared_to_runtime_impl(
    Runtime *runtime, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr, const ArgDirection *signature,
    int sig_count
);
int validate_runtime_impl(Runtime *runtime);

/* ===========================================================================
 * Per-thread DeviceRunner binding (set by prepare_callable / run_prepared, read by HostApi wrappers)
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

static uint64_t upload_chip_callable_buffer_wrapper(const void *callable) {
    try {
        return current_runner()->upload_chip_callable_buffer(static_cast<const ChipCallable *>(callable));
    } catch (...) {
        return 0;
    }
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
 * ACL lifecycle stubs.  Sim has no ACL / aclrtStream concept, so these
 * no-op to satisfy the uniform host_runtime.so ABI (ChipWorker dlsym's the
 * full extension surface unconditionally).  The paired comm_init / barrier /
 * destroy entry points already live in comm_sim.cpp.
 * =========================================================================== */

int ensure_acl_ready_ctx(DeviceContextHandle ctx, int device_id) {
    (void)ctx;
    (void)device_id;
    return 0;
}

void *create_comm_stream_ctx(DeviceContextHandle ctx) {
    (void)ctx;
    return NULL;
}

int destroy_comm_stream_ctx(DeviceContextHandle ctx, void *stream) {
    (void)ctx;
    (void)stream;
    return 0;
}

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size
) {
    if (ctx == NULL) return -1;

    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);
    int rc;
    try {
        rc = runner->attach_current_thread(device_id);
    } catch (...) {
        return -1;
    }
    if (rc != 0) return rc;

    try {
        std::vector<uint8_t> aicpu_vec;
        std::vector<uint8_t> aicore_vec;
        if (aicpu_binary != NULL && aicpu_size > 0) {
            aicpu_vec.assign(aicpu_binary, aicpu_binary + aicpu_size);
        }
        if (aicore_binary != NULL && aicore_size > 0) {
            aicore_vec.assign(aicore_binary, aicore_binary + aicore_size);
        }
        runner->set_executors(std::move(aicpu_vec), std::move(aicore_vec));
    } catch (...) {
        return -1;
    }
    // No CANN dlog on sim. HostLogger is owned by libsimpler_log.so.
    return 0;
}

/* ===========================================================================
 * Per-callable_id preparation
 * =========================================================================== */

int prepare_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
    if (ctx == NULL || callable == NULL) return -1;
    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);

    try {
        PreparedCallableArtifacts artifacts;
        int rc = prepare_callable_impl(
            reinterpret_cast<const ChipCallable *>(callable), upload_chip_callable_buffer_wrapper, &artifacts
        );
        if (rc != 0) {
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        // Re-pack ChildKernelAddr -> std::pair to match the existing
        // register_prepared_callable* signature.
        std::vector<std::pair<int, uint64_t>> kernel_addrs;
        kernel_addrs.reserve(artifacts.kernel_addrs.size());
        for (const ChildKernelAddr &c : artifacts.kernel_addrs) {
            kernel_addrs.emplace_back(c.func_id, c.device_addr);
        }

        if (artifacts.host_dlopen_handle != nullptr) {
            rc = runner->register_prepared_callable_host_orch(
                callable_id, artifacts.host_dlopen_handle, artifacts.host_orch_func_ptr, std::move(kernel_addrs),
                std::move(artifacts.signature)
            );
        } else {
            rc = runner->register_prepared_callable(
                callable_id, artifacts.orch_so_data, artifacts.orch_so_size, artifacts.func_name.c_str(),
                artifacts.config_name.c_str(), std::move(kernel_addrs), std::move(artifacts.signature)
            );
        }
        pthread_setspecific(g_runner_key, nullptr);
        return rc;
    } catch (...) {
        pthread_setspecific(g_runner_key, nullptr);
        return -1;
    }
}

int run_prepared(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, int block_dim,
    int aicpu_thread_num, int enable_l2_swimlane, int enable_dump_tensor, int enable_pmu, int enable_dep_gen,
    const char *output_prefix
) {
    if (ctx == NULL || runtime == NULL) return -1;
    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);

    if (!runner->has_prepared_callable(callable_id)) {
        LOG_ERROR("run_prepared: callable_id=%d not prepared", callable_id);
        return -1;
    }

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);

    try {
        Runtime *r = new (runtime) Runtime();
        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;
        r->host_api.upload_chip_callable_buffer = upload_chip_callable_buffer_wrapper;

        auto bind_result = runner->bind_prepared_callable_to_runtime(*r, callable_id);
        int rc = bind_result.rc;
        if (rc != 0) {
            r->~Runtime();
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        rc = bind_prepared_to_runtime_impl(
            r, reinterpret_cast<const ChipStorageTaskArgs *>(args), bind_result.host_orch_func_ptr,
            bind_result.signature, bind_result.sig_count
        );
        if (rc != 0) {
            r->set_gm_sm_ptr(nullptr);
            validate_runtime_impl(r);
            r->~Runtime();
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        runner->set_l2_swimlane_enabled(enable_l2_swimlane != 0);
        runner->set_dump_tensor_enabled(enable_dump_tensor != 0);
        runner->set_pmu_enabled(enable_pmu);
        runner->set_dep_gen_enabled(enable_dep_gen != 0);
        runner->set_output_prefix(output_prefix);

        rc = runner->run(*r, block_dim, aicpu_thread_num);
        if (rc != 0) {
            validate_runtime_impl(r);
            r->~Runtime();
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        rc = validate_runtime_impl(r);
        r->~Runtime();
        pthread_setspecific(g_runner_key, nullptr);
        return rc;
    } catch (...) {
        pthread_setspecific(g_runner_key, nullptr);
        return -1;
    }
}

int unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->unregister_prepared_callable(callable_id);
    } catch (...) {
        return -1;
    }
}

size_t get_host_dlopen_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunner *>(ctx)->host_dlopen_count();
    } catch (...) {
        return 0;
    }
}

size_t get_aicpu_dlopen_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunner *>(ctx)->aicpu_dlopen_count();
    } catch (...) {
        return 0;
    }
}

}  // extern "C"
