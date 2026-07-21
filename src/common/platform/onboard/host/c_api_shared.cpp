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
#include "call_config.h"
#include "device_runner_base.h"
#include "prepare_callable_common.h"
#include "pto_runtime_c_api.h"
#include "task_args.h"

#include <dlfcn.h>
#include <pthread.h>

#include <cstdlib>
#include <cstring>
#include <utility>
#include <vector>

#include "common/strace.h"
#include "common/unified_log.h"
#include "host_log.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"
#include "platform_comm/comm.h"

// Forward-declared (rather than `#include "dlog_pub.h"`) so this TU does not
// require CANN's toolchain include path on the host build. Resolved at link
// time against `libunified_dlog.so` / `libascendalog.so`.
extern "C" int dlog_setlevel(int moduleId, int level, int enableEvent);

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in each runtime's runtime_maker.cpp)
 * =========================================================================== */
int register_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out);
int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc);
int release_async_host_graph_pipeline(Runtime *runtime);
__attribute__((weak)) int release_async_host_graph_prepare(Runtime * /*runtime*/) { return 0; }

/* ===========================================================================
 * Per-thread DeviceRunnerBase binding (set by simpler_register_callable / simpler_run)
 * =========================================================================== */

static pthread_key_t g_runner_key;
static pthread_once_t g_runner_key_once = PTHREAD_ONCE_INIT;
static void create_runner_key() { pthread_key_create(&g_runner_key, nullptr); }

static DeviceRunnerBase *current_runner() { return static_cast<DeviceRunnerBase *>(pthread_getspecific(g_runner_key)); }

static void *capture_thread_context() { return current_runner(); }

static int bind_thread_context(void *context) {
    auto *runner = static_cast<DeviceRunnerBase *>(context);
    if (runner == nullptr) return -1;
    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, runner);
    int rc = runner->attach_current_thread(runner->device_id());
    if (rc != 0) pthread_setspecific(g_runner_key, nullptr);
    return rc;
}

static void unbind_thread_context() { pthread_setspecific(g_runner_key, nullptr); }

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

static int store_u64_release_to_device(void *dev_ptr, uint64_t value) {
    if (dev_ptr == nullptr) return -1;
    try {
        return current_runner()->copy_to_device(dev_ptr, &value, sizeof(value));
    } catch (...) {
        return -1;
    }
}

static int load_u64_acquire_from_device(uint64_t *value, const void *dev_ptr) {
    if (value == nullptr || dev_ptr == nullptr) return -1;
    try {
        return current_runner()->copy_from_device(value, dev_ptr, sizeof(*value));
    } catch (...) {
        return -1;
    }
}

static void *register_device_memory_to_host(void *dev_ptr, size_t bytes) {
    try {
        return current_runner()->register_device_memory_to_host(dev_ptr, bytes);
    } catch (...) {
        return nullptr;
    }
}

static void unregister_device_memory_from_host(void *dev_ptr) {
    try {
        current_runner()->unregister_device_memory_from_host(dev_ptr);
    } catch (...) {}
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

// Weak no-op default lives in device_runner_base.cpp; tensormap_and_ringbuffer
// links a strong override that builds + caches the prebuilt runtime-arena.
// simpler_init calls it directly for the fork-constant ring sizing.
extern "C" int prewarm_config_impl(
    const HostApi *api, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
);

// The HostApi is a set of context-free function pointers: each wrapper above
// recovers its runner from the thread-local current_runner(), so a single
// filled table is valid for every runner and every run. Build it once at load
// time rather than reassembling the pointer table on each simpler_run. Passed by
// address into bind_callable_to_runtime_impl / validate_runtime_impl.
static const HostApi g_host_api = {
    .capture_thread_context = capture_thread_context,
    .bind_thread_context = bind_thread_context,
    .unbind_thread_context = unbind_thread_context,
    .device_malloc = device_malloc,
    .device_free = device_free,
    .copy_to_device = copy_to_device,
    .copy_from_device = copy_from_device,
    .store_u64_release_to_device = store_u64_release_to_device,
    .load_u64_acquire_from_device = load_u64_acquire_from_device,
    .register_device_memory_to_host = register_device_memory_to_host,
    .unregister_device_memory_from_host = unregister_device_memory_from_host,
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
        return runner->finalize();
    } catch (...) {
        return -1;
    }
}

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *prewarm_config
) {
    if (ctx == NULL) return -1;

    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    // HostApi callbacks, including the prewarm path below, recover their
    // DeviceRunner from this thread-local binding.
    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

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
    // simpler_register_callable / simpler_run invocations reuse them — no per-run
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
    // symbol rtFuncHandle for per-task launch — so the first simpler_register_callable
    // / simpler_run does not pay any of these costs. Streams live until
    // finalize_device; the cached rtFuncHandle on LoadAicpuOp and the
    // preinstall file both live until ~DeviceRunner.
    try {
        rc = runner->ensure_device_initialized();
    } catch (...) {
        return -1;
    }
    if (rc != 0) return rc;

    // Prebuilt runtime-arena prewarm: the device is up, so build + cache the
    // arena for the fork-constant ring sizing now. trb provides a strong
    // prewarm_config_impl; other runtimes link the weak no-op. Only the ring
    // sizing is read.
    if (prewarm_config != NULL) {
        try {
            rc = prewarm_config_impl(
                &g_host_api, prewarm_config->runtime_env.ring_task_window, prewarm_config->runtime_env.ring_heap,
                prewarm_config->runtime_env.ring_dep_pool
            );
        } catch (...) {
            return -1;
        }
        if (rc != 0) return rc;
    }
    return 0;
}

/* ===========================================================================
 * Per-callable_id preparation
 * =========================================================================== */

int simpler_register_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
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
        auto chip_buffer_guard = RAIIScopeGuard([runner, &artifacts]() {
            if (artifacts.chip_buffer_hash != 0) {
                runner->release_chip_callable_buffer(artifacts.chip_buffer_hash);
            }
        });
        rc = register_callable_impl(
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
        // record_device_orch_callable* signature. The named struct only crosses
        // the runtime-maker / device-runner interface; CallableState
        // stores the historical pair shape.
        std::vector<std::pair<int, uint64_t>> kernel_addrs;
        kernel_addrs.reserve(artifacts.kernel_addrs.size());
        for (const ChildKernelAddr &c : artifacts.kernel_addrs) {
            kernel_addrs.emplace_back(c.func_id, c.device_addr);
        }

        // hbg's register_callable_impl populates host_dlopen_handle; trb's
        // leaves it null and fills orch_so_data + func_name/config_name.
        bool needs_aicpu_register = false;
        if (artifacts.host_dlopen_handle != nullptr) {
            rc = runner->record_host_orch_callable(
                callable_id, artifacts.chip_buffer_hash, artifacts.host_dlopen_handle, artifacts.host_orch_func_ptr,
                std::move(kernel_addrs), std::move(artifacts.signature)
            );
            if (rc != 0) return rc;
            host_dlopen_guard.dismiss();
            chip_buffer_guard.dismiss();
        } else {
            rc = runner->record_device_orch_callable(
                callable_id, artifacts.chip_buffer_hash, artifacts.chip_buffer_dev, artifacts.orch_so_data,
                artifacts.orch_so_size, artifacts.func_name.c_str(), artifacts.config_name.c_str(),
                std::move(kernel_addrs), std::move(artifacts.signature)
            );
            if (rc != 0) return rc;
            chip_buffer_guard.dismiss();
            needs_aicpu_register = true;
        }
        if (needs_aicpu_register) {
            rc = runner->launch_device_register(callable_id);
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

// Runtime gate for device-domain phase emission. SIMPLER_DEVICE_STRACE_ENABLE=0
// suppresses the device (clk=dev) markers so a deployment can profile host and
// device independently; any other value (or unset) keeps them on. Host-side
// [STRACE] spans are unaffected — they ride SIMPLER_HOST_STRACE + the log level.
// Read once and cached: getenv is not thread-safe against setenv, and the value
// is a process-lifetime config knob.
static bool device_profiling_enabled() {
    static const bool enabled = [] {
        const char *v = std::getenv("SIMPLER_DEVICE_STRACE_ENABLE");
        return v == nullptr || std::strcmp(v, "0") != 0;
    }();
    return enabled;
}

// Emit device-domain trace markers for the AICPU phases. RunWall (the whole
// on-NPU wall, i.e. the former RunTiming.device_wall) is emitted at depth 2
// under runner_run; its preamble/so_load/graph_build/post_orch subdivisions are
// emitted at depth 3 beneath it. Phases never stamped (0 ns) are skipped.
// STRACE_DEV_SPAN_AT self-compiles to nothing when profiling is off, so no extra
// gate is needed here.
static void emit_device_phase_markers(DeviceRunnerBase *runner) {
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

    // Selective task-timing slots: one span per complete slot, start = dispatch
    // and duration = finish - dispatch, both on the phase timeline so cross-slot
    // intervals (e.g. finish(slot_1) - dispatch(slot_0)) stay recoverable.
    // Untagged / incomplete slots read back 0/0 and are skipped.
    static const char *const kTaskSlotNames[NUM_TASK_TIMING_SLOTS] = {
        "simpler_run.runner_run.device_wall.task_slot_0",  "simpler_run.runner_run.device_wall.task_slot_1",
        "simpler_run.runner_run.device_wall.task_slot_2",  "simpler_run.runner_run.device_wall.task_slot_3",
        "simpler_run.runner_run.device_wall.task_slot_4",  "simpler_run.runner_run.device_wall.task_slot_5",
        "simpler_run.runner_run.device_wall.task_slot_6",  "simpler_run.runner_run.device_wall.task_slot_7",
        "simpler_run.runner_run.device_wall.task_slot_8",  "simpler_run.runner_run.device_wall.task_slot_9",
        "simpler_run.runner_run.device_wall.task_slot_10", "simpler_run.runner_run.device_wall.task_slot_11",
        "simpler_run.runner_run.device_wall.task_slot_12", "simpler_run.runner_run.device_wall.task_slot_13",
        "simpler_run.runner_run.device_wall.task_slot_14", "simpler_run.runner_run.device_wall.task_slot_15",
    };
    for (int s = 0; s < NUM_TASK_TIMING_SLOTS; ++s) {
        const uint64_t dispatch_ns = runner->last_task_slot_dispatch_ns(s);
        const uint64_t finish_ns = runner->last_task_slot_finish_ns(s);
        if (finish_ns > dispatch_ns) {
            STRACE_DEV_SPAN_AT(
                kTaskSlotNames[s], static_cast<long long>(dispatch_ns), static_cast<long long>(finish_ns - dispatch_ns),
                3
            );
        }
    }
}

int simpler_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config
) {
    if (ctx == NULL || runtime == NULL || config == NULL) return -1;
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);

    if (!runner->has_callable(callable_id)) {
        LOG_ERROR("simpler_run: callable_id=%d not registered", callable_id);
        return -1;
    }
    if (!runner->can_accept_run()) {
        LOG_ERROR(
            "simpler_run: runner is unusable after a prior device failure; refusing callable_id=%d before resource "
            "provisioning",
            callable_id
        );
        return -1;
    }

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    STRACE_NEW_INV();
    STRACE_SET_HID(runner->callable_hash(callable_id));
    STRACE("simpler_run");

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
        // Platform device-memory hooks. host_api is a platform capability, not
        // runtime state — the shared g_host_api table (built once at load time)
        // is passed explicitly into the runtime impls rather than stored on
        // `Runtime` or reassembled per run.
        {
            STRACE("simpler_run.bind");
            // One-step bind: restore kernel addrs + active_callable_id and run
            // the per-run binding (tensor args, GM heap, SM alloc). The
            // CallableState-derived host_orch_func_ptr + signature stay inside
            // the runner — no longer returned across this boundary.
            rc = runner->bind_callable_to_runtime(
                *r, callable_id, &g_host_api, args, config->runtime_env.ring_task_window, config->runtime_env.ring_heap,
                config->runtime_env.ring_dep_pool
            );
        }
        if (rc != 0) {
            r->set_gm_sm_ptr(nullptr);
            int validation_rc = validate_runtime_impl(r, &g_host_api, rc);
            return validation_rc != 0 ? validation_rc : rc;
        }

        {
            STRACE("simpler_run.runner_run");
            // run() latches the diagnostic enables from config via
            // apply_call_config() and consumes block_dim / aicpu_thread_num.
            rc = runner->run(*r, *config);
        }
        if (rc != 0) {
            int validation_rc = validate_runtime_impl(r, &g_host_api, rc);
            return validation_rc != 0 ? validation_rc : rc;
        }

        {
            STRACE("simpler_run.validate");
            rc = validate_runtime_impl(r, &g_host_api, 0);
        }
        // Device-domain phase markers: the AICPU subdivision of the on-NPU wall
        // (device_wall + preamble/so_load/graph_build/post_orch/orch/sched).
        // host_wall is the simpler_run STRACE span; both flow via the log, not a
        // return value.
        emit_device_phase_markers(runner);
        return rc;
    } catch (...) {
        return -1;
    }
}

int select_arena_bank_ctx(DeviceContextHandle ctx, unsigned arena_bank) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->select_arena_bank(arena_bank);
    } catch (...) {
        return -1;
    }
}

int simpler_prepare_request(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config,
    unsigned arena_bank
) {
    if (ctx == NULL || runtime == NULL || args == NULL || config == NULL) return -1;
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    if (!runner->has_callable(callable_id) || runner->select_arena_bank(arena_bank) != 0) return -1;
    auto arena_guard = RAIIScopeGuard([runner]() {
        runner->select_arena_bank(0);
    });

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    STRACE_NEW_INV();
    STRACE_SET_HID(runner->callable_hash(callable_id));
    STRACE("simpler_prepare_request");
    Runtime *r = nullptr;
    try {
        int rc = runner->attach_current_thread(runner->device_id());
        if (rc != 0) return rc;
        r = new (runtime) Runtime();
        {
            STRACE("simpler_prepare_request.bind");
            rc = runner->bind_callable_to_runtime(
                *r, callable_id, &g_host_api, args, config->runtime_env.ring_task_window, config->runtime_env.ring_heap,
                config->runtime_env.ring_dep_pool
            );
        }
        if (rc == 0) {
            rc = release_async_host_graph_prepare(r);
        }
        if (rc != 0) {
            r->set_gm_sm_ptr(nullptr);
            validate_runtime_impl(r, &g_host_api);
            r->~Runtime();
        }
        return rc;
    } catch (...) {
        if (r != nullptr) r->~Runtime();
        return -1;
    }
}

int simpler_execute_prepared(
    DeviceContextHandle ctx, RuntimeHandle runtime, const CallConfig *config, unsigned arena_bank
) {
    if (ctx == NULL || runtime == NULL || config == NULL) return -1;
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    if (runner->select_arena_bank(arena_bank) != 0) return -1;
    auto arena_guard = RAIIScopeGuard([runner]() {
        runner->select_arena_bank(0);
    });

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    STRACE_NEW_INV();
    STRACE("simpler_execute_prepared");
    Runtime *r = static_cast<Runtime *>(runtime);
    auto runtime_guard = RAIIScopeGuard([r]() {
        r->~Runtime();
    });
    try {
        int rc = runner->attach_current_thread(runner->device_id());
        if (rc == 0) {
            STRACE("simpler_execute_prepared.runner_run");
            rc = runner->run(*r, *config);
        }
        int validate_rc;
        {
            STRACE("simpler_execute_prepared.validate");
            validate_rc = validate_runtime_impl(r, &g_host_api);
        }
        emit_device_phase_markers(runner);
        return rc != 0 ? rc : validate_rc;
    } catch (...) {
        return -1;
    }
}

int simpler_unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
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

int simpler_provision_dma_workspace(DeviceContextHandle ctx, uint32_t required_mask) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->provision_dma_workspace(required_mask);
    } catch (...) {
        return -1;
    }
}

}  // extern "C"
