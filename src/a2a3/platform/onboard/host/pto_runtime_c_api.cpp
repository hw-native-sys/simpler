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
#include "prepare_callable_common.h"
#include "task_args.h"

#include <pthread.h>

#include <chrono>
#include <cstdlib>
#include <utility>
#include <vector>

#include "common/unified_log.h"
#include "device_runner.h"
#include "host_log.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"

// Forward-declared (rather than #include "dlog_pub.h") so this TU does not
// require CANN's toolchain include path on the host build. Resolved at link
// time against libunified_dlog.so / libascendalog.so.
extern "C" int dlog_setlevel(int moduleId, int level, int enableEvent);

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

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunner *>(ctx)->finalize();
    } catch (...) {
        return -1;
    }
}

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size
) {
    if (ctx == NULL) return -1;

    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);

    // CANN dlog must be levelled BEFORE the device context is opened
    // (rtSetDevice inside attach_current_thread): CANN snapshots the
    // device-side log session's level at context-open time, so a later
    // dlog_setlevel is a no-op for the device side. HostLogger is already
    // seeded here by libsimpler_log.so's simpler_log_init() (runs earlier in
    // ChipWorker::init). Skipped when ASCEND_GLOBAL_LOG_LEVEL is externally
    // configured — CANN keeps that.
    if (std::getenv("ASCEND_GLOBAL_LOG_LEVEL") == NULL) {
        dlog_setlevel(-1, HostLogger::get_instance().level(), /*enableEvent*/ 0);
    }

    int rc;
    try {
        rc = runner->attach_current_thread(device_id);
    } catch (...) {
        return -1;
    }
    if (rc != 0) return rc;

    // Transfer ownership of the executor binaries to the runner. Subsequent
    // prepare_callable / run_prepared invocations reuse them — no per-run
    // binary push across the C ABI.
    try {
        std::vector<uint8_t> aicpu_vec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicore_vec(aicore_binary, aicore_binary + aicore_size);
        runner->set_executors(std::move(aicpu_vec), std::move(aicore_vec));
    } catch (...) {
        return -1;
    }
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
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    try {
        int rc = runner->prepare_run_context(runner->device_id());
        if (rc != 0) return rc;
        auto run_context_guard = RAIIScopeGuard([runner]() {
            runner->release_run_context();
        });

        PreparedCallableArtifacts artifacts;
        rc = prepare_callable_impl(
            reinterpret_cast<const ChipCallable *>(callable), upload_chip_callable_buffer_wrapper, &artifacts
        );
        if (rc != 0) {
            return rc;
        }

        // Re-pack ChildKernelAddr -> std::pair to match the existing
        // register_prepared_callable* signature. The named struct only crosses
        // the runtime-maker / device-runner interface; PreparedCallableState
        // stores the historical pair shape.
        std::vector<std::pair<int, uint64_t>> kernel_addrs;
        kernel_addrs.reserve(artifacts.kernel_addrs.size());
        for (const ChildKernelAddr &c : artifacts.kernel_addrs) {
            kernel_addrs.emplace_back(c.func_id, c.device_addr);
        }

        // hbg's prepare_callable_impl populates host_dlopen_handle; trb's
        // leaves it null and fills orch_so_data + func_name/config_name.
        if (artifacts.host_dlopen_handle != nullptr) {
            return runner->register_prepared_callable_host_orch(
                callable_id, artifacts.host_dlopen_handle, artifacts.host_orch_func_ptr, std::move(kernel_addrs),
                std::move(artifacts.signature)
            );
        }
        return runner->register_prepared_callable(
            callable_id, artifacts.orch_so_data, artifacts.orch_so_size, artifacts.func_name.c_str(),
            artifacts.config_name.c_str(), std::move(kernel_addrs), std::move(artifacts.signature)
        );
    } catch (...) {
        return -1;
    }
}

int run_prepared(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, int block_dim,
    int aicpu_thread_num, int enable_l2_swimlane, int enable_dump_tensor, int enable_pmu, int enable_dep_gen,
    const char *output_prefix, PtoRunTiming *out_timing
) {
    if (out_timing != NULL) {
        out_timing->host_wall_ns = 0;
        out_timing->device_wall_ns = 0;
    }
    if (ctx == NULL || runtime == NULL) return -1;
    DeviceRunner *runner = static_cast<DeviceRunner *>(ctx);

    if (!runner->has_prepared_callable(callable_id)) {
        LOG_ERROR("run_prepared: callable_id=%d not prepared", callable_id);
        return -1;
    }

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    const auto host_t0 = std::chrono::steady_clock::now();

    try {
        int rc = runner->prepare_run_context(runner->device_id());
        if (rc != 0) return rc;
        auto run_context_guard = RAIIScopeGuard([runner]() {
            runner->release_run_context();
        });

        Runtime *r = new (runtime) Runtime();
        r->host_api.device_malloc = device_malloc;
        r->host_api.device_free = device_free;
        r->host_api.copy_to_device = copy_to_device;
        r->host_api.copy_from_device = copy_from_device;
        r->host_api.upload_chip_callable_buffer = upload_chip_callable_buffer_wrapper;

        // Restore kernel addrs + orch symbol names + active_callable_id; the
        // returned host_orch_func_ptr is non-null only on the hbg path and is
        // handed straight into bind_prepared_to_runtime_impl below. signature
        // is the cached ChipCallable signature_[]; it's plumbed end-to-end for
        // per-tensor direction decisions in runtime_maker but is currently
        // unconsumed on both runtimes — see bind_prepared_to_runtime_impl.
        auto bind_result = runner->bind_prepared_callable_to_runtime(*r, callable_id);
        if (bind_result.rc != 0) {
            r->~Runtime();
            return bind_result.rc;
        }

        // Per-run binding (tensor args, GM heap, SM alloc)
        rc = bind_prepared_to_runtime_impl(
            r, reinterpret_cast<const ChipStorageTaskArgs *>(args), bind_result.host_orch_func_ptr,
            bind_result.signature, bind_result.sig_count
        );
        if (rc != 0) {
            r->set_gm_sm_ptr(nullptr);
            validate_runtime_impl(r);
            r->~Runtime();
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
            return rc;
        }

        rc = validate_runtime_impl(r);
        r->~Runtime();
        if (out_timing != NULL) {
            const auto host_t1 = std::chrono::steady_clock::now();
            out_timing->host_wall_ns =
                static_cast<uint64_t>(std::chrono::duration_cast<std::chrono::nanoseconds>(host_t1 - host_t0).count());
            out_timing->device_wall_ns = runner->last_device_wall_ns();
        }
        return rc;
    } catch (...) {
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

size_t get_aicpu_dlopen_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunner *>(ctx)->aicpu_dlopen_count();
    } catch (...) {
        return 0;
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

}  // extern "C"
