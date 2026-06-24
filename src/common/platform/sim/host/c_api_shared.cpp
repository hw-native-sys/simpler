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
 * Shared sim c_api glue — TSD binding, static wrappers, and the bulk of the
 * public C ABI surface, all written against SimDeviceRunnerBase * so the same
 * source file is linked into both arches' libhost_runtime.so (sim variant).
 *
 * Per-arch pto_runtime_c_api.cpp keeps only `create_device_context` (the one
 * line that requires the concrete DeviceRunner type) plus the acl/comm
 * placeholders (sim has no ACL; comm_init/barrier/destroy come from
 * src/common/platform_comm/comm_sim.cpp).
 *
 * Mirrors the onboard pattern from PR #928.
 */

#include "pto_runtime_c_api.h"

#include "callable.h"
#include "device_runner_base.h"
#include "prepare_callable_common.h"
#include "task_args.h"

#include <dlfcn.h>
#include <pthread.h>

#include <chrono>
#include <new>
#include <utility>
#include <vector>

#include "common/unified_log.h"
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in runtime_maker.cpp)
 * =========================================================================== */
int prepare_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out);
int bind_callable_to_runtime_impl(
    Runtime *runtime, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr, const ArgDirection *signature,
    int sig_count, uint64_t ring_task_window, uint64_t ring_heap, uint64_t ring_dep_pool,
    const uint64_t *ring_task_windows, const uint64_t *ring_heaps, const uint64_t *ring_dep_pools
);
int validate_runtime_impl(Runtime *runtime);

/* ===========================================================================
 * Per-thread DeviceRunner binding
 * =========================================================================== */

static pthread_key_t g_runner_key;
static pthread_once_t g_runner_key_once = PTHREAD_ONCE_INIT;
static void create_runner_key() { pthread_key_create(&g_runner_key, nullptr); }

static SimDeviceRunnerBase *current_runner() {
    return static_cast<SimDeviceRunnerBase *>(pthread_getspecific(g_runner_key));
}

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

static int device_memset(void *dev_ptr, int value, size_t size) {
    if (dev_ptr == NULL) return -1;
    try {
        return current_runner()->device_memset(dev_ptr, value, size);
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

static int setup_static_arena_wrapper(size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size) {
    try {
        return current_runner()->setup_static_arena(gm_heap_size, gm_sm_size, runtime_arena_size);
    } catch (...) {
        return -1;
    }
}

static void *acquire_pooled_gm_heap_wrapper() {
    try {
        return current_runner()->acquire_pooled_gm_heap();
    } catch (...) {
        return nullptr;
    }
}

static void *acquire_pooled_gm_sm_wrapper() {
    try {
        return current_runner()->acquire_pooled_gm_sm();
    } catch (...) {
        return nullptr;
    }
}

static void *acquire_pooled_runtime_arena_wrapper() {
    try {
        return current_runner()->acquire_pooled_runtime_arena();
    } catch (...) {
        return nullptr;
    }
}

/* ===========================================================================
 * Public C API (resolved by ChipWorker via dlsym)
 * =========================================================================== */

void destroy_device_context(DeviceContextHandle ctx) { delete static_cast<SimDeviceRunnerBase *>(ctx); }

size_t get_runtime_size(void) { return sizeof(Runtime); }

void *device_malloc_ctx(DeviceContextHandle ctx, size_t size) {
    if (ctx == NULL) return NULL;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->allocate_tensor(size);
    } catch (...) {
        return NULL;
    }
}

void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr) {
    if (ctx == NULL || dev_ptr == NULL) return;
    try {
        static_cast<SimDeviceRunnerBase *>(ctx)->free_tensor(dev_ptr);
    } catch (...) {}
}

int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size) {
    if (ctx == NULL || dev_ptr == NULL || host_ptr == NULL) return -1;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    if (ctx == NULL || host_ptr == NULL || dev_ptr == NULL) return -1;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        int rc = static_cast<SimDeviceRunnerBase *>(ctx)->finalize();
        int dev = pto_cpu_sim_get_bound_device();
        if (dev >= 0) {
            pto_cpu_sim_release_device(dev);
        }
        return rc;
    } catch (...) {
        return -1;
    }
}

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size
) {
    // Sim has no AICPU dispatcher (the simulator runs AICPU in-process). Accept
    // the parameters for ABI parity with the onboard implementation and ignore
    // them — callers that pass dispatcher bytes get the same shape as onboard,
    // and the dispatcher / preinstall load path on sim isn't taken anyway.
    (void)dispatcher_binary;
    (void)dispatcher_size;

    if (ctx == NULL) return -1;

    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
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
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);

    try {
        CallableArtifacts artifacts;
        int rc = prepare_callable_impl(
            reinterpret_cast<const ChipCallable *>(callable), upload_chip_callable_buffer_wrapper, &artifacts
        );
        if (rc != 0) {
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }
        auto host_dlopen_guard = RAIIScopeGuard([&artifacts]() {
            if (artifacts.host_dlopen_handle != nullptr) {
                dlclose(artifacts.host_dlopen_handle);
            }
        });

        std::vector<std::pair<int, uint64_t>> kernel_addrs;
        kernel_addrs.reserve(artifacts.kernel_addrs.size());
        for (const ChildKernelAddr &c : artifacts.kernel_addrs) {
            kernel_addrs.emplace_back(c.func_id, c.device_addr);
        }

        if (artifacts.host_dlopen_handle != nullptr) {
            rc = runner->register_callable_host_orch(
                callable_id, artifacts.host_dlopen_handle, artifacts.host_orch_func_ptr, std::move(kernel_addrs),
                std::move(artifacts.signature)
            );
            if (rc == 0) {
                host_dlopen_guard.dismiss();
            }
        } else {
            rc = runner->register_callable(
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
    int enable_scope_stats, uint64_t ring_task_window, uint64_t ring_heap, uint64_t ring_dep_pool,
    const uint64_t *ring_task_windows, const uint64_t *ring_heaps, const uint64_t *ring_dep_pools,
    const char *output_prefix, PtoRunTiming *out_timing
) {
    if (out_timing != NULL) {
        out_timing->host_wall_ns = 0;
        out_timing->device_wall_ns = 0;
    }
    if (ctx == NULL || runtime == NULL) return -1;
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);

    if (!runner->has_callable(callable_id)) {
        LOG_ERROR("run_prepared: callable_id=%d not prepared", callable_id);
        return -1;
    }

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);

    const auto host_t0 = std::chrono::steady_clock::now();

    try {
        Runtime *r = new (runtime) Runtime();
        // RAII the placement-new'd Runtime so its dtor fires on every exit
        // (normal returns, the rc-check early-returns below, AND the catch(...)
        // path). Mirrors the onboard c_api_shared fix from PR #928.
        auto runtime_guard = RAIIScopeGuard([r]() {
            r->~Runtime();
        });

        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;
        r->host_api.device_memset = device_memset;
        r->host_api.setup_static_arena = setup_static_arena_wrapper;
        r->host_api.acquire_pooled_gm_heap = acquire_pooled_gm_heap_wrapper;
        r->host_api.acquire_pooled_gm_sm = acquire_pooled_gm_sm_wrapper;
        r->host_api.acquire_pooled_runtime_arena = acquire_pooled_runtime_arena_wrapper;
        r->host_api.upload_chip_callable_buffer = upload_chip_callable_buffer_wrapper;

        auto bind_result = runner->bind_callable_to_runtime(*r, callable_id);
        int rc = bind_result.rc;
        if (rc != 0) {
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        rc = bind_callable_to_runtime_impl(
            r, reinterpret_cast<const ChipStorageTaskArgs *>(args), bind_result.host_orch_func_ptr,
            bind_result.signature, bind_result.sig_count, ring_task_window, ring_heap, ring_dep_pool, ring_task_windows,
            ring_heaps, ring_dep_pools
        );
        if (rc != 0) {
            r->set_gm_sm_ptr(nullptr);
            validate_runtime_impl(r);
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        runner->set_l2_swimlane_enabled(enable_l2_swimlane);
        runner->set_dump_tensor_enabled(enable_dump_tensor);
        runner->set_pmu_enabled(enable_pmu);
        runner->set_dep_gen_enabled(enable_dep_gen != 0);
        runner->set_scope_stats_enabled(enable_scope_stats != 0);
        runner->set_output_prefix(output_prefix);

        rc = runner->run(*r, block_dim, aicpu_thread_num);
        if (rc != 0) {
            validate_runtime_impl(r);
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        rc = validate_runtime_impl(r);
        pthread_setspecific(g_runner_key, nullptr);
        if (out_timing != NULL) {
            const auto host_t1 = std::chrono::steady_clock::now();
            out_timing->host_wall_ns =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(host_t1 - host_t0).count());
            out_timing->device_wall_ns = runner->last_device_wall_ns();
        }
        return rc;
    } catch (...) {
        pthread_setspecific(g_runner_key, nullptr);
        return -1;
    }
}

int unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->unregister_callable(callable_id);
    } catch (...) {
        return -1;
    }
}

size_t get_host_dlopen_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->host_dlopen_count();
    } catch (...) {
        return 0;
    }
}

size_t get_aicpu_dlopen_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->aicpu_dlopen_count();
    } catch (...) {
        return 0;
    }
}

}  // extern "C"
