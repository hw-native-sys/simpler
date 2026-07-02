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
#include "call_config.h"
#include "device_runner_base.h"
#include "prepare_callable_common.h"
#include "task_args.h"

#include <dlfcn.h>
#include <pthread.h>

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#include "common/device_phase.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in runtime_maker.cpp)
 * =========================================================================== */
int register_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out);
int validate_runtime_impl(Runtime *runtime, const HostApi *api);

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
 * Internal device-memory functions (wired into a HostApi and passed to the
 * runtime impls, NOT dlsym'd)
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

static void get_retained_temp_buffer(void **addr, size_t *size) {
    try {
        current_runner()->get_retained_temp_buffer(addr, size);
    } catch (...) {
        if (addr != nullptr) *addr = nullptr;
        if (size != nullptr) *size = 0;
    }
}

static void set_retained_temp_buffer(void *addr, size_t size) {
    try {
        current_runner()->set_retained_temp_buffer(addr, size);
    } catch (...) {}
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

static bool lookup_prebuilt_runtime_arena_cache_wrapper(
    uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base, void **sm_base,
    void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
) {
    try {
        return current_runner()->lookup_prebuilt_runtime_arena_cache(
            hash, key_data, key_size, gm_heap_base, sm_base, runtime_arena_base, runtime_off, image_data, image_size
        );
    } catch (...) {
        return false;
    }
}

static void mark_prebuilt_runtime_arena_cached_wrapper(
    uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base, void *sm_base, void *runtime_arena_base,
    size_t runtime_off, const void *image_data, size_t image_size
) {
    try {
        current_runner()->mark_prebuilt_runtime_arena_cached(
            hash, key_data, key_size, gm_heap_base, sm_base, runtime_arena_base, runtime_off, image_data, image_size
        );
    } catch (...) {}
}

// The HostApi is a set of context-free function pointers: each wrapper above
// recovers its runner from the thread-local current_runner(), so a single
// filled table is valid for every runner and every run. Build it once at load
// time rather than reassembling the pointer table on each simpler_run. Passed by
// address into bind_callable_to_runtime_impl / validate_runtime_impl.
static const HostApi g_host_api = {
    .device_malloc = device_malloc,
    .device_free = device_free,
    .copy_to_device = copy_to_device,
    .copy_from_device = copy_from_device,
    .device_memset = device_memset,
    .get_retained_temp_buffer = get_retained_temp_buffer,
    .set_retained_temp_buffer = set_retained_temp_buffer,
    .setup_static_arena = setup_static_arena_wrapper,
    .acquire_pooled_gm_heap = acquire_pooled_gm_heap_wrapper,
    .acquire_pooled_gm_sm = acquire_pooled_gm_sm_wrapper,
    .acquire_pooled_runtime_arena = acquire_pooled_runtime_arena_wrapper,
    .lookup_prebuilt_runtime_arena_cache = lookup_prebuilt_runtime_arena_cache_wrapper,
    .mark_prebuilt_runtime_arena_cached = mark_prebuilt_runtime_arena_cached_wrapper,
    .upload_chip_callable_buffer = upload_chip_callable_buffer_wrapper,
};

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
        SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
        int rc = runner->l3_l2_orch_comm_shutdown();
        int finalize_rc = runner->finalize();
        if (rc == 0) {
            rc = finalize_rc;
        }
        int dev = pto_cpu_sim_get_bound_device();
        if (dev >= 0) {
            pto_cpu_sim_release_device(dev);
        }
        return rc;
    } catch (...) {
        return -1;
    }
}

int l3_l2_orch_comm_init_ctx(DeviceContextHandle ctx, void *control_block, size_t control_block_size) {
    if (ctx == NULL || control_block == NULL) return -1;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->l3_l2_orch_comm_init(control_block, control_block_size);
    } catch (...) {
        return -1;
    }
}

int l3_l2_orch_comm_shutdown_ctx(DeviceContextHandle ctx) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->l3_l2_orch_comm_shutdown();
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

int simpler_register_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
    if (ctx == NULL || callable == NULL) return -1;
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);

    try {
        CallableArtifacts artifacts;
        int rc = register_callable_impl(
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

        bool needs_aicpu_register = false;
        if (artifacts.host_dlopen_handle != nullptr) {
            rc = runner->record_host_orch_callable(
                callable_id, artifacts.host_dlopen_handle, artifacts.host_orch_func_ptr, std::move(kernel_addrs),
                std::move(artifacts.signature)
            );
            if (rc == 0) {
                host_dlopen_guard.dismiss();
            }
        } else {
            rc = runner->record_device_orch_callable(
                callable_id, artifacts.orch_so_data, artifacts.orch_so_size, artifacts.func_name.c_str(),
                artifacts.config_name.c_str(), std::move(kernel_addrs), std::move(artifacts.signature)
            );
            if (rc == 0) {
                needs_aicpu_register = true;
            }
        }
        if (rc == 0 && needs_aicpu_register) {
            rc = runner->launch_device_register(callable_id);
            if (rc != 0) {
                runner->unregister_callable(callable_id);
            }
        }
        pthread_setspecific(g_runner_key, nullptr);
        return rc;
    } catch (...) {
        pthread_setspecific(g_runner_key, nullptr);
        return -1;
    }
}

// Runtime gate for device-domain phase emission. SIMPLER_DEVICE_PROFILING=0
// suppresses the device (clk=dev) markers so a deployment can profile host and
// device independently; any other value (or unset) keeps them on. Host-side
// [STRACE] spans are unaffected — they ride SIMPLER_PROFILING + the log level.
// Read once and cached (process-lifetime config knob).
static bool device_profiling_enabled() {
    static const bool enabled = [] {
        const char *v = std::getenv("SIMPLER_DEVICE_PROFILING");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

// Emit device-domain phase markers (RunWall + its 4 AICPU subdivisions),
// mirroring the onboard c_api. Phases never stamped (0 ns) are skipped.
// STRACE_DEV_SPAN_AT self-compiles to nothing when profiling is off.
static void emit_device_phase_markers(SimDeviceRunnerBase *runner) {
    if (!device_profiling_enabled()) return;
    const uint64_t run_wall_ns = runner->last_device_phase_ns(AicpuPhase::RunWall);
    if (run_wall_ns != 0) {
        STRACE_DEV_SPAN_AT("simpler_run.runner_run.device_wall", 0, static_cast<long long>(run_wall_ns), 2);
    }
    struct PhaseName {
        AicpuPhase phase;
        const char *name;
    };
    static const PhaseName kPhases[] = {
        {AicpuPhase::Preamble, "simpler_run.runner_run.device_wall.preamble"},
        {AicpuPhase::SoLoad, "simpler_run.runner_run.device_wall.so_load"},
        {AicpuPhase::GraphBuild, "simpler_run.runner_run.device_wall.graph_build"},
        {AicpuPhase::ConfigValidate, "simpler_run.runner_run.device_wall.config_validate"},
        {AicpuPhase::ArenaWire, "simpler_run.runner_run.device_wall.arena_wire"},
        {AicpuPhase::SmReset, "simpler_run.runner_run.device_wall.sm_reset"},
        {AicpuPhase::PostOrch, "simpler_run.runner_run.device_wall.post_orch"},
        {AicpuPhase::OrchWindow, "simpler_run.runner_run.device_wall.orch"},
        {AicpuPhase::SchedWindow, "simpler_run.runner_run.device_wall.sched"},
    };
    // RunWall is emitted above as device_wall; every other phase is in the table.
    static_assert(
        sizeof(kPhases) / sizeof(kPhases[0]) == NUM_AICPU_PHASES - 1,
        "kPhases[] must list every AicpuPhase except RunWall — add the new phase here"
    );
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

int simpler_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config
) {
    if (ctx == NULL || runtime == NULL || config == NULL) return -1;
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);

    if (!runner->has_callable(callable_id)) {
        LOG_ERROR("simpler_run: callable_id=%d not registered", callable_id);
        return -1;
    }

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);

    STRACE_NEW_INV();
    STRACE_SET_HID(static_cast<uint64_t>(callable_id));
    STRACE("simpler_run");

    try {
        Runtime *r = new (runtime) Runtime();
        // RAII the placement-new'd Runtime so its dtor fires on every exit
        // (normal returns, the rc-check early-returns below, AND the catch(...)
        // path). Mirrors the onboard c_api_shared fix from PR #928.
        auto runtime_guard = RAIIScopeGuard([r]() {
            r->~Runtime();
        });

        // Platform device-memory hooks. host_api is a platform capability, not
        // runtime state — the shared g_host_api table (built once at load time)
        // is passed explicitly into the runtime impls rather than stored on
        // `Runtime` or reassembled per run.
        int rc;
        {
            STRACE("simpler_run.bind");
            // One-step bind: replay CallableState + run the per-run binding. The
            // host_orch_func_ptr + signature stay inside the runner.
            rc = runner->bind_callable_to_runtime(
                *r, callable_id, &g_host_api, args, config->runtime_env.ring_task_window, config->runtime_env.ring_heap,
                config->runtime_env.ring_dep_pool
            );
        }
        if (rc != 0) {
            r->set_gm_sm_ptr(nullptr);
            validate_runtime_impl(r, &g_host_api);
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        {
            STRACE("simpler_run.runner_run");
            // run() latches the diagnostic enables from config via
            // apply_call_config() and consumes block_dim / aicpu_thread_num.
            rc = runner->run(*r, *config);
        }
        if (rc != 0) {
            validate_runtime_impl(r, &g_host_api);
            pthread_setspecific(g_runner_key, nullptr);
            return rc;
        }

        {
            STRACE("simpler_run.validate");
            rc = validate_runtime_impl(r, &g_host_api);
        }
        pthread_setspecific(g_runner_key, nullptr);
        emit_device_phase_markers(runner);
        return rc;
    } catch (...) {
        pthread_setspecific(g_runner_key, nullptr);
        return -1;
    }
}

int simpler_unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
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
