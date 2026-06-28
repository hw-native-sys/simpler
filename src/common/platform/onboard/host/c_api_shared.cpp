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
 * Shared `pto_runtime_c_api` glue — the byte-identical part of every arch's
 * onboard `pto_runtime_c_api.cpp`. Linked into each arch's
 * `libhost_runtime.so` directly (not as a separate library) so all C ABI
 * symbols are exported from each `.so` for ChipWorker's `dlsym`.
 *
 * Works through `DeviceRunnerBase *` and dispatches arch-specific
 * behavior (`run`, `finalize`, `set_dep_gen_enabled`) through the
 * virtuals declared on `DeviceRunnerBase`. The `create_device_context`
 * factory stays per-arch since it must know the concrete `DeviceRunner`
 * subclass to `new`. The HCCL / comm entrypoints
 * (`ensure_acl_ready_ctx`, `create_comm_stream_ctx`,
 * `destroy_comm_stream_ctx`, `comm_*`) also stay per-arch — a2a3 has
 * real implementations, a5 has stubs.
 */

#include "callable.h"
#include "device_runner_base.h"
#include "prepare_callable_common.h"
#include "pto_runtime_c_api.h"
#include "task_args.h"

#include <dlfcn.h>
#include <pthread.h>

#include <chrono>
#include <cstdlib>
#include <utility>
#include <vector>

#include "common/strace.h"
#include "common/unified_log.h"
#include "host_log.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"

// Forward-declared (rather than `#include "dlog_pub.h"`) so this TU does not
// require CANN's toolchain include path on the host build. Resolved at link
// time against `libunified_dlog.so` / `libascendalog.so`.
extern "C" int dlog_setlevel(int moduleId, int level, int enableEvent);

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in each runtime's runtime_maker.cpp)
 * =========================================================================== */
int prepare_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out);
int bind_callable_to_runtime_impl(
    Runtime *runtime, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr, const ArgDirection *signature,
    int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
);
int validate_runtime_impl(Runtime *runtime);

/* ===========================================================================
 * Per-thread DeviceRunnerBase binding (set by prepare_callable / run_prepared)
 * =========================================================================== */

static pthread_key_t g_runner_key;
static pthread_once_t g_runner_key_once = PTHREAD_ONCE_INIT;
static void create_runner_key() { pthread_key_create(&g_runner_key, nullptr); }

static DeviceRunnerBase *current_runner() { return static_cast<DeviceRunnerBase *>(pthread_getspecific(g_runner_key)); }

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
 *
 * `create_device_context` stays per-arch (must know the concrete
 * `DeviceRunner` subclass to `new`); everything else routes through
 * `DeviceRunnerBase *`.
 * =========================================================================== */

void destroy_device_context(DeviceContextHandle ctx) { delete static_cast<DeviceRunnerBase *>(ctx); }

size_t get_runtime_size(void) { return sizeof(Runtime); }

void *device_malloc_ctx(DeviceContextHandle ctx, size_t size) {
    if (ctx == NULL) return NULL;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->allocate_tensor(size);
    } catch (...) {
        return NULL;
    }
}

void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr) {
    if (ctx == NULL || dev_ptr == NULL) return;
    try {
        static_cast<DeviceRunnerBase *>(ctx)->free_tensor(dev_ptr);
    } catch (...) {}
}

int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size) {
    if (ctx == NULL || dev_ptr == NULL || host_ptr == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    if (ctx == NULL || host_ptr == NULL || dev_ptr == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
        int rc = runner->l3_l2_orch_comm_shutdown();
        int finalize_rc = runner->finalize();
        if (rc == 0) {
            rc = finalize_rc;
        }
        return rc;
    } catch (...) {
        return -1;
    }
}

int l3_l2_orch_comm_init_ctx(DeviceContextHandle ctx, void *control_block, size_t control_block_size) {
    if (ctx == NULL || control_block == NULL) return -1;
    try {
        DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
        if (!runner->l3_l2_orch_comm_supported()) {
            return PTO_RUNTIME_ERR_UNSUPPORTED;
        }
        return runner->l3_l2_orch_comm_init(control_block, control_block_size);
    } catch (...) {
        return -1;
    }
}

int l3_l2_orch_comm_shutdown_ctx(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->l3_l2_orch_comm_shutdown();
    } catch (...) {
        return -1;
    }
}

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size
) {
    if (ctx == NULL) return -1;

    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);

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
        // Dispatcher SO bytes are passed alongside the executors. Onboard
        // requires a non-empty buffer: BootstrapDispatcher reads from it to
        // upload the dispatcher + inner SO bundle through
        // libaicpu_extend_kernels. If the caller drives _ChipWorker.init
        // directly without a dispatcher path, this stays empty and the
        // ensure_device_initialized call below fails fast with a clear message.
        if (dispatcher_binary != NULL && dispatcher_size > 0) {
            std::vector<uint8_t> dispatcher_vec(dispatcher_binary, dispatcher_binary + dispatcher_size);
            runner->set_dispatcher_binary(std::move(dispatcher_vec));
        }
    } catch (...) {
        return -1;
    }

    // Eagerly run the one-shot device setup: create persistent AICPU/AICore
    // streams, upload the dispatcher + inner SO bundle, and resolve the per-
    // symbol rtFuncHandle for per-task launch — so the first prepare_callable
    // / run_prepared does not pay any of these costs. Streams live until
    // finalize_device; the cached rtFuncHandle on LoadAicpuOp and the
    // preinstall file both live until ~DeviceRunner.
    try {
        rc = runner->ensure_device_initialized();
    } catch (...) {
        return -1;
    }
    if (rc != 0) return rc;
    return 0;
}

/* ===========================================================================
 * Per-callable_id preparation
 * =========================================================================== */

int prepare_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
    if (ctx == NULL || callable == NULL) return -1;
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    try {
        int rc = runner->attach_current_thread(runner->device_id());
        if (rc != 0) return rc;

        CallableArtifacts artifacts;
        rc = prepare_callable_impl(
            reinterpret_cast<const ChipCallable *>(callable), upload_chip_callable_buffer_wrapper, &artifacts
        );
        if (rc != 0) {
            return rc;
        }
        auto host_dlopen_guard = RAIIScopeGuard([&artifacts]() {
            if (artifacts.host_dlopen_handle != nullptr) {
                dlclose(artifacts.host_dlopen_handle);
            }
        });

        // Re-pack ChildKernelAddr -> std::pair to match the existing
        // register_callable* signature. The named struct only crosses
        // the runtime-maker / device-runner interface; CallableState
        // stores the historical pair shape.
        std::vector<std::pair<int, uint64_t>> kernel_addrs;
        kernel_addrs.reserve(artifacts.kernel_addrs.size());
        for (const ChildKernelAddr &c : artifacts.kernel_addrs) {
            kernel_addrs.emplace_back(c.func_id, c.device_addr);
        }

        // hbg's prepare_callable_impl populates host_dlopen_handle; trb's
        // leaves it null and fills orch_so_data + func_name/config_name.
        bool needs_aicpu_prewarm = false;
        if (artifacts.host_dlopen_handle != nullptr) {
            rc = runner->register_callable_host_orch(
                callable_id, artifacts.host_dlopen_handle, artifacts.host_orch_func_ptr, std::move(kernel_addrs),
                std::move(artifacts.signature)
            );
            if (rc != 0) return rc;
            host_dlopen_guard.dismiss();
        } else {
            rc = runner->register_callable(
                callable_id, artifacts.orch_so_data, artifacts.orch_so_size, artifacts.func_name.c_str(),
                artifacts.config_name.c_str(), std::move(kernel_addrs), std::move(artifacts.signature)
            );
            if (rc != 0) return rc;
            needs_aicpu_prewarm = true;
        }
        if (needs_aicpu_prewarm) {
            rc = runner->prewarm_callable(callable_id);
            if (rc != 0) {
                runner->unregister_callable(callable_id);
                return rc;
            }
        }
        return 0;
    } catch (...) {
        return -1;
    }
}

// Emit device-domain trace markers for the AICPU phases. RunWall (the whole
// on-NPU wall, i.e. the former RunTiming.device_wall) is emitted at depth 2
// under runner_run; its preamble/so_load/graph_build/post_orch subdivisions are
// emitted at depth 3 beneath it. Phases never stamped (0 ns) are skipped.
// STRACE_DEV_SPAN_AT self-compiles to nothing when profiling is off, so no extra
// gate is needed here.
static void emit_device_phase_markers(DeviceRunnerBase *runner) {
    const uint64_t run_wall_ns = runner->last_device_phase_ns(AicpuPhase::RunWall);
    if (run_wall_ns != 0) {
        STRACE_DEV_SPAN_AT("run_prepared.runner_run.device_wall", 0, static_cast<long long>(run_wall_ns), 2);
    }
    struct PhaseName {
        AicpuPhase phase;
        const char *name;
    };
    static const PhaseName kPhases[] = {
        {AicpuPhase::Preamble, "run_prepared.runner_run.device_wall.preamble"},
        {AicpuPhase::SoLoad, "run_prepared.runner_run.device_wall.so_load"},
        {AicpuPhase::GraphBuild, "run_prepared.runner_run.device_wall.graph_build"},
        {AicpuPhase::PostOrch, "run_prepared.runner_run.device_wall.post_orch"},
        {AicpuPhase::OrchWindow, "run_prepared.runner_run.device_wall.orch"},
        {AicpuPhase::SchedWindow, "run_prepared.runner_run.device_wall.sched"},
    };
    for (const auto &p : kPhases) {
        const uint64_t ns = runner->last_device_phase_ns(p.phase);
        if (ns != 0) {
            STRACE_DEV_SPAN_AT(
                p.name, static_cast<long long>(runner->last_device_phase_start_ns(p.phase)), static_cast<long long>(ns),
                3
            );
        }
    }
}

int run_prepared(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, int block_dim,
    int aicpu_thread_num, int enable_l2_swimlane, int enable_dump_tensor, int enable_pmu, int enable_dep_gen,
    int enable_scope_stats, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool,
    const char *output_prefix
) {
    if (ctx == NULL || runtime == NULL) return -1;
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);

    if (!runner->has_callable(callable_id)) {
        LOG_ERROR("run_prepared: callable_id=%d not prepared", callable_id);
        return -1;
    }

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    STRACE_NEW_INV();
    STRACE_SET_HID(runner->callable_hash(callable_id));
    STRACE("run_prepared");

    try {
        int rc = runner->attach_current_thread(runner->device_id());
        if (rc != 0) return rc;

        Runtime *r = new (runtime) Runtime();
        // RAII the placement-new'd Runtime so its dtor fires on every exit
        // (normal returns, the rc-check early-returns below, AND the catch(...)
        // path). The prior manual `r->~Runtime()` on each return leaked the
        // Runtime on any exception thrown inside the try block.
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

        {
            STRACE("run_prepared.bind");
            // Restore kernel addrs + orch symbol names + active_callable_id; the
            // returned host_orch_func_ptr is non-null only on the hbg path and is
            // handed straight into bind_callable_to_runtime_impl below. signature
            // is the cached ChipCallable signature_[]; it's plumbed end-to-end for
            // per-tensor direction decisions in runtime_maker. trb consumes it to
            // skip the D2H copy-back of read-only INPUT tensors; hbg still
            // ignores it — see bind_callable_to_runtime_impl.
            auto bind_result = runner->bind_callable_to_runtime(*r, callable_id);
            if (bind_result.rc != 0) {
                return bind_result.rc;
            }

            // Per-run binding (tensor args, GM heap, SM alloc)
            rc = bind_callable_to_runtime_impl(
                r, reinterpret_cast<const ChipStorageTaskArgs *>(args), bind_result.host_orch_func_ptr,
                bind_result.signature, bind_result.sig_count, ring_task_window, ring_heap, ring_dep_pool
            );
        }
        if (rc != 0) {
            r->set_gm_sm_ptr(nullptr);
            validate_runtime_impl(r);
            return rc;
        }

        runner->set_l2_swimlane_enabled(enable_l2_swimlane);
        runner->set_dump_tensor_enabled(enable_dump_tensor);
        runner->set_pmu_enabled(enable_pmu);
        // Virtual: a2a3 and a5 wire through to their enable_dep_gen_; an arch
        // without dep_gen falls through to the base no-op.
        runner->set_dep_gen_enabled(enable_dep_gen != 0);
        runner->set_scope_stats_enabled(enable_scope_stats != 0);
        runner->set_output_prefix(output_prefix);

        {
            STRACE("run_prepared.runner_run");
            rc = runner->run(*r, block_dim, aicpu_thread_num);
        }
        if (rc != 0) {
            validate_runtime_impl(r);
            return rc;
        }
        if (r->register_new_callable_id()) {
            rc = runner->commit_aicpu_callable_load(r->get_active_callable_id());
            if (rc != 0) {
                validate_runtime_impl(r);
                return rc;
            }
        }

        {
            STRACE("run_prepared.validate");
            rc = validate_runtime_impl(r);
        }
        // Device-domain phase markers: the AICPU subdivision of the on-NPU wall
        // (device_wall + preamble/so_load/graph_build/post_orch/orch/sched).
        // host_wall is the run_prepared STRACE span; both flow via the log, not a
        // return value.
        emit_device_phase_markers(runner);
        return rc;
    } catch (...) {
        return -1;
    }
}

int unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->unregister_callable(callable_id);
    } catch (...) {
        return -1;
    }
}

size_t get_aicpu_dlopen_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->aicpu_dlopen_count();
    } catch (...) {
        return 0;
    }
}

size_t get_host_dlopen_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->host_dlopen_count();
    } catch (...) {
        return 0;
    }
}

}  // extern "C"
