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
#include "native_run_state.h"

#include <dlfcn.h>
#include <pthread.h>

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>
#include <thread>
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

using OnboardNativeRunState = NativeRunState<DeviceRunnerBase>;
// Phase entry points validate raw caller storage before beginning object
// lifetime, so the on-storage magic must remain the leading bytes.
static_assert(__builtin_offsetof(OnboardNativeRunState, magic) == 0, "native-run magic must lead runtime storage");

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in each runtime's runtime_maker.cpp)
 * =========================================================================== */
int register_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out);
int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc);
__attribute__((weak)) int concurrent_native_prepare_supported_impl(void) { return 0; }
__attribute__((weak)) int prepared_run_config_compatible_impl(
    const HostApi * /*api*/, const uint64_t * /*ring_task_window*/, const uint64_t * /*ring_heap*/,
    const uint64_t * /*ring_dep_pool*/
) {
    return 1;
}

/* ===========================================================================
 * Per-thread DeviceRunnerBase binding (set by simpler_register_callable / simpler_run)
 * =========================================================================== */

static pthread_key_t g_runner_key;
static pthread_once_t g_runner_key_once = PTHREAD_ONCE_INIT;
static void create_runner_key() { pthread_key_create(&g_runner_key, nullptr); }

static DeviceRunnerBase *current_runner() { return static_cast<DeviceRunnerBase *>(pthread_getspecific(g_runner_key)); }

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
    .device_malloc = device_malloc,
    .device_free = device_free,
    .copy_to_device = copy_to_device,
    .copy_from_device = copy_from_device,
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

void destroy_device_context(DeviceContextHandle ctx) {
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    if (runner != nullptr && runner->native_runs_outstanding()) {
        LOG_ERROR("destroy_device_context: refusing to destroy a context with an unfinalized native run");
        return;
    }
    delete runner;
}

size_t get_runtime_size(void) { return sizeof(OnboardNativeRunState); }

size_t get_runtime_alignment(void) { return alignof(OnboardNativeRunState); }

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
        if (runner->native_runs_outstanding()) {
            LOG_ERROR("finalize_device: native run must be finalized first");
            return -1;
        }
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
    (void)runner->select_pipeline_slot(0);
    (void)runner->select_arena_bank(0);
    (void)runner->set_native_run_identity(0, 0, 0, 0);

    // CANN dlog must be levelled BEFORE the device context is opened
    // (rtSetDevice inside attach_current_thread): CANN snapshots the
    // device-side log session's level at context-open time, so a later
    // dlog_setlevel is a no-op for the device side. HostLogger is already
    // seeded here by libsimpler_log.so's simpler_log_init() (runs earlier in
    // ChipWorker::init). Skipped when ASCEND_GLOBAL_LOG_LEVEL is externally
    // configured — CANN keeps that.
    HostLogger::get_instance().configure_cann_log_level(dlog_setlevel);

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
    if (runner->native_runs_outstanding()) {
        LOG_ERROR("simpler_register_callable: native run must be finalized before mutating the callable registry");
        return -1;
    }

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
                callable_id, artifacts.chip_buffer_hash, artifacts.aicore_image_hash, artifacts.host_dlopen_handle,
                artifacts.host_orch_func_ptr, std::move(kernel_addrs), std::move(artifacts.signature)
            );
            if (rc != 0) return rc;
            host_dlopen_guard.dismiss();
            chip_buffer_guard.dismiss();
        } else {
            rc = runner->record_device_orch_callable(
                callable_id, artifacts.chip_buffer_hash, artifacts.aicore_image_hash, artifacts.chip_buffer_dev,
                artifacts.orch_so_data, artifacts.orch_so_size, artifacts.func_name.c_str(),
                artifacts.config_name.c_str(), std::move(kernel_addrs), std::move(artifacts.signature)
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

static OnboardNativeRunState *native_run_state(DeviceContextHandle ctx, RuntimeHandle runtime, const char *operation) {
    if (ctx == nullptr || runtime == nullptr) return nullptr;
    uint64_t magic = 0;
    std::memcpy(&magic, runtime, sizeof(magic));
    if (magic != OnboardNativeRunState::kMagic) {
        LOG_ERROR("%s: runtime does not contain a prepared native run", operation);
        return nullptr;
    }
    auto *state = static_cast<OnboardNativeRunState *>(runtime);
    if (state->runner != static_cast<DeviceRunnerBase *>(ctx)) {
        LOG_ERROR("%s: prepared run belongs to a different device context", operation);
        return nullptr;
    }
    return state;
}

static void
emit_native_run_host_wall(unsigned trace_inv, uint64_t trace_hid, long long trace_start_ns, const char *trace_attrs) {
    const long long end_ns = STRACE_NOW_NS();
    STRACE_CONTEXT(trace_inv, trace_hid, 0);
    STRACE_HOST_SPAN_AT_A("simpler_run", trace_start_ns, end_ns - trace_start_ns, 0, trace_attrs);
}

int supports_concurrent_native_prepare_ctx(DeviceContextHandle ctx) {
    return ctx != nullptr && concurrent_native_prepare_supported_impl() != 0 ? 1 : 0;
}

static int cleanup_failed_prepare(OnboardNativeRunState *state, int execution_rc, bool clear_gm_sm) {
    const unsigned trace_inv = state->trace_inv;
    const uint64_t trace_hid = state->trace_hid;
    const long long trace_start_ns = state->trace_start_ns;
    char trace_attrs[sizeof(state->trace_attrs)];
    std::memcpy(trace_attrs, state->trace_attrs, sizeof(trace_attrs));
    if (clear_gm_sm) state->runtime.set_gm_sm_ptr(nullptr);
    int validation_rc = -1;
    try {
        validation_rc = validate_runtime_impl(&state->runtime, &g_host_api, execution_rc);
    } catch (...) {
        validation_rc = -1;
    }
    int resources_rc = 0;
    if (state->runner_resources_owned) {
        try {
            resources_rc = state->runner->abandon_native_run_resources(state->pipeline_slot);
        } catch (...) {
            resources_rc = -1;
        }
        state->runner_resources_owned = false;
    }
    if (state->runner_claimed) {
        state->runner->release_native_run(state);
        state->runner_claimed = false;
    }
    if (state->runner_reserved) {
        state->runner->release_native_run_reservation(state);
        state->runner_reserved = false;
    }
    destroy_native_run_state(state);
    emit_native_run_host_wall(trace_inv, trace_hid, trace_start_ns, trace_attrs);
    if (validation_rc != 0) return validation_rc;
    if (resources_rc != 0) return resources_rc;
    return execution_rc;
}

int simpler_prepare_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config
) {
    if (ctx == nullptr || runtime == nullptr || config == nullptr) return -1;
    if (reinterpret_cast<uintptr_t>(runtime) % alignof(OnboardNativeRunState) != 0) {
        LOG_ERROR("simpler_prepare_run: runtime storage does not satisfy get_runtime_alignment()");
        return -1;
    }
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    if (!runner->has_callable(callable_id)) {
        LOG_ERROR("simpler_prepare_run: callable_id=%d not registered", callable_id);
        return -1;
    }
    if (!runner->can_accept_run()) {
        LOG_ERROR("simpler_prepare_run: runner is unusable after a prior device failure");
        return -1;
    }
    uint64_t magic = 0;
    std::memcpy(&magic, runtime, sizeof(magic));
    if (magic == OnboardNativeRunState::kMagic) {
        LOG_ERROR("simpler_prepare_run: runtime already contains a prepared run; finalize it before reuse");
        return -1;
    }
    if (magic != 0) {
        LOG_ERROR("simpler_prepare_run: runtime storage was not zero-initialized before its first use");
        return -1;
    }

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

    OnboardNativeRunState *state = nullptr;
    const uint64_t trace_hid = runner->callable_hash(callable_id);
    const unsigned trace_inv = STRACE_ALLOC_INV();
    const long long trace_start_ns = STRACE_NOW_NS();
    try {
        state = new (runtime) OnboardNativeRunState(runner, *config, trace_hid);
        const DeviceRunnerBase::NativeRunThreadSelection selection = runner->capture_native_run_thread_selection();
        (void)runner->set_native_run_identity(0, 0, 0, 0);
        state->run_id = selection.run_id;
        state->generation = selection.generation;
        state->dispatch_id = selection.dispatch_id;
        state->run_epoch = selection.run_epoch;
        state->pipeline_slot = selection.pipeline_slot;
        state->arena_bank = selection.arena_bank;
        std::snprintf(
            state->trace_attrs, sizeof(state->trace_attrs),
            "run_id=%llu slot=%u generation=%llu dispatch_id=%llu run_epoch=%llu",
            static_cast<unsigned long long>(state->run_id), state->pipeline_slot,
            static_cast<unsigned long long>(state->generation), static_cast<unsigned long long>(state->dispatch_id),
            static_cast<unsigned long long>(state->run_epoch)
        );
        const bool allow_prepared_successor =
            concurrent_native_prepare_supported_impl() != 0 && !config->diagnostics_any();
        if (!runner->try_reserve_native_run(state, state->pipeline_slot, state->arena_bank, allow_prepared_successor)) {
            LOG_ERROR("simpler_prepare_run: native-run admission is occupied (%s)", state->trace_attrs);
            destroy_native_run_state(state);
            return -1;
        }
        state->runner_reserved = true;
        const bool overlaps_active_run = allow_prepared_successor && runner->native_run_active();
        state->trace_inv = trace_inv;
        state->trace_start_ns = trace_start_ns;
        STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);

        int rc = runner->attach_current_thread(runner->device_id());
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);

        if (overlaps_active_run) {
            int compatibility_rc = 0;
            {
                STRACE("simpler_run.bind.compatibility");
                compatibility_rc = prepared_run_config_compatible_impl(
                    &g_host_api, config->runtime_env.ring_task_window, config->runtime_env.ring_heap,
                    config->runtime_env.ring_dep_pool
                );
            }
            if (compatibility_rc <= 0) {
                if (compatibility_rc == 0) {
                    LOG_INFO("successor RuntimeEnv requires depth-one native preparation");
                    compatibility_rc = PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
                }
                return cleanup_failed_prepare(state, compatibility_rc, true);
            }
        }

        state->runner_resources_owned = true;
        rc = runner->provision_native_run_resources(state->pipeline_slot);
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);

        rc = runner->prepare_launch_shape(state->runtime, state->config);
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);

        // Diagnostic binding reads runner-global collector configuration. It
        // is depth-one, while concurrent successor preparation must leave the
        // active run's configuration untouched until launch.
        if (!overlaps_active_run) runner->apply_call_config(state->config);

        {
            STRACE("simpler_run.bind");
            rc = runner->bind_callable_to_runtime(
                state->runtime, callable_id, &g_host_api, args, state->config.runtime_env.ring_task_window,
                state->config.runtime_env.ring_heap, state->config.runtime_env.ring_dep_pool
            );
        }
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);
        state->host_thread_state = runner->take_native_run_thread_state();
        return 0;
    } catch (...) {
        if (state != nullptr) return cleanup_failed_prepare(state, -1, true);
        return -1;
    }
}

int simpler_launch_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunState *state = native_run_state(ctx, runtime, "simpler_launch_run");
    if (state == nullptr || state->phase.load(std::memory_order_acquire) != NativeRunPhase::Prepared) return -1;
    if (!state->runner->can_accept_run() || !state->runner_reserved) return -1;
    if (!state->runner->try_acquire_native_run(state, &state->launch_signal)) {
        LOG_ERROR("simpler_launch_run: execution claim is occupied (%s)", state->trace_attrs);
        return -1;
    }
    state->runner_claimed = true;
    // The active predecessor may poison the device after this successor was
    // prepared but before the execution claim becomes available.
    if (!state->runner->can_accept_run()) {
        state->runner->release_native_run(state);
        state->runner_claimed = false;
        return -1;
    }

    // Phase entry points temporarily install the run's resource selection.
    // Preserve the caller's selection so interleaving finalize(A) and
    // launch(B) on one progress thread cannot leave that thread bound to A.
    const DeviceRunnerBase::NativeRunThreadSelection caller_selection =
        state->runner->capture_native_run_thread_selection();
    auto selection_guard = RAIIScopeGuard([runner = state->runner, caller_selection]() {
        runner->restore_native_run_thread_selection(caller_selection);
    });
    if (state->runner->select_pipeline_slot(state->pipeline_slot) != 0 ||
        state->runner->select_arena_bank(state->arena_bank) != 0) {
        state->runner->release_native_run(state);
        state->runner_claimed = false;
        return -1;
    }

    state->phase.store(NativeRunPhase::Launching, std::memory_order_release);

    try {
        const DeviceRunnerBase::NativeRunThreadSelection execution_selection =
            state->runner->capture_native_run_thread_selection();
        // The runner owns one persistent blocking execution thread. The
        // endpoint remains the sole FIFO/progress owner and polls this run's
        // completion state without creating a host thread for every launch.
        const bool submitted = state->runner->submit_native_execution([state, ctx, execution_selection]() {
            pthread_once(&g_runner_key_once, create_runner_key);
            pthread_setspecific(g_runner_key, ctx);
            state->runner->restore_native_run_thread_selection(execution_selection);
            STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);
            int rc = -1;
            bool entered_run = false;
            try {
                int attach_rc = state->runner->attach_current_thread(state->runner->device_id());
                if (attach_rc == 0) {
                    state->adopt_host_thread_state();
                    state->runner->activate_launch_shape(state->runtime);
                    {
                        STRACE("simpler_run.runner_run");
                        entered_run = true;
                        rc = state->runner->run(state->runtime, state->config);
                    }
                } else {
                    rc = attach_rc;
                }
            } catch (...) {
                rc = -1;
            }
            if (entered_run) {
                // run() owns stream retirement on every exit once entered.
                state->runner_resources_owned = false;
            } else if (state->runner_resources_owned) {
                int resources_rc = -1;
                try {
                    resources_rc = state->runner->abandon_native_run_resources(state->pipeline_slot);
                } catch (...) {}
                state->runner_resources_owned = false;
                if (rc == 0) rc = resources_rc;
            }
            pthread_setspecific(g_runner_key, nullptr);
            // Completion permits finalize to destroy `state`. Keep the launch
            // fallback notification before that publication so the persistent
            // executor never dereferences run-owned storage after done=true.
            state->launch_signal.notify();
            state->publish_execution_complete(rc);
        });
        if (!submitted) throw std::runtime_error("native execution thread is occupied");
    } catch (...) {
        state->runner->release_native_run(state);
        state->runner_claimed = false;
        state->phase.store(NativeRunPhase::Prepared, std::memory_order_release);
        return -1;
    }

    state->launch_signal.wait();
    if (state->execution_done.load(std::memory_order_acquire)) {
        state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
        return state->execution_rc.load(std::memory_order_relaxed);
    }
    state->phase.store(NativeRunPhase::Running, std::memory_order_release);
    return 0;
}

int simpler_poll_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunState *state = native_run_state(ctx, runtime, "simpler_poll_run");
    if (state == nullptr) return SIMPLER_NATIVE_RUN_POLL_ERROR;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    if (phase == NativeRunPhase::Prepared) return SIMPLER_NATIVE_RUN_POLL_ERROR;
    if (phase == NativeRunPhase::Complete || state->execution_done.load(std::memory_order_acquire)) {
        state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
        return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    }
    return SIMPLER_NATIVE_RUN_POLL_NOT_READY;
}

int simpler_wait_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunState *state = native_run_state(ctx, runtime, "simpler_wait_run");
    if (state == nullptr) return -1;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    if (phase == NativeRunPhase::Prepared || phase == NativeRunPhase::Launching) return -1;
    state->wait_for_execution_complete();
    state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
    return state->execution_rc.load(std::memory_order_relaxed);
}

int simpler_finalize_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunState *state = native_run_state(ctx, runtime, "simpler_finalize_run");
    if (state == nullptr) return -1;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    if (phase == NativeRunPhase::Launching) return -1;
    const unsigned trace_inv = state->trace_inv;
    const uint64_t trace_hid = state->trace_hid;
    const long long trace_start_ns = state->trace_start_ns;
    char trace_attrs[sizeof(state->trace_attrs)];
    std::memcpy(trace_attrs, state->trace_attrs, sizeof(trace_attrs));

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });
    STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);

    // Finalization can target A after the same caller has already prepared B.
    // Resource selection is phase-local; restore B's caller selection on every
    // return path, including validation and resource-retirement failures.
    const DeviceRunnerBase::NativeRunThreadSelection caller_selection =
        state->runner->capture_native_run_thread_selection();
    auto selection_guard = RAIIScopeGuard([runner = state->runner, caller_selection]() {
        runner->restore_native_run_thread_selection(caller_selection);
    });
    if (state->runner->select_pipeline_slot(state->pipeline_slot) != 0 ||
        state->runner->select_arena_bank(state->arena_bank) != 0) {
        return -1;
    }

    int execution_rc = -1;
    const bool launched = phase != NativeRunPhase::Prepared;
    if (launched) {
        state->wait_for_execution_complete();
        execution_rc = state->execution_rc.load(std::memory_order_relaxed);
    }

    int validation_rc = -1;
    try {
        if (!launched) state->runtime.set_gm_sm_ptr(nullptr);
        int attach_rc = state->runner->attach_current_thread(state->runner->device_id());
        if (attach_rc == 0) {
            {
                STRACE("simpler_run.validate");
                validation_rc = validate_runtime_impl(&state->runtime, &g_host_api, launched ? execution_rc : -1);
            }
            if (launched && execution_rc == 0) emit_device_phase_markers(state->runner);
        } else {
            validation_rc = attach_rc;
        }
    } catch (...) {
        validation_rc = -1;
    }

    int resources_rc = 0;
    if (!launched && state->runner_resources_owned) {
        try {
            resources_rc = state->runner->abandon_native_run_resources(state->pipeline_slot);
        } catch (...) {
            resources_rc = -1;
        }
        state->runner_resources_owned = false;
    }

    if (state->runner_claimed) {
        state->runner->release_native_run(state);
        state->runner_claimed = false;
    }
    if (state->runner_reserved) {
        state->runner->release_native_run_reservation(state);
        state->runner_reserved = false;
    }
    destroy_native_run_state(state);
    emit_native_run_host_wall(trace_inv, trace_hid, trace_start_ns, trace_attrs);
    if (validation_rc != 0) return validation_rc;
    if (resources_rc != 0) return resources_rc;
    return launched ? execution_rc : 0;
}

int simpler_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config
) {
    int rc = simpler_prepare_run(ctx, runtime, callable_id, args, config);
    if (rc != 0) return rc;
    rc = simpler_launch_run(ctx, runtime);
    if (rc == 0) rc = simpler_wait_run(ctx, runtime);
    int finalize_rc = simpler_finalize_run(ctx, runtime);
    return finalize_rc != 0 ? finalize_rc : rc;
}

int set_task_accepted_state_ctx(DeviceContextHandle ctx, volatile int32_t *state, int32_t accepted_value) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->set_task_accepted_state(state, accepted_value);
    } catch (...) {
        return -1;
    }
}

int select_pipeline_slot_ctx(DeviceContextHandle ctx, uint32_t slot_id) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->select_pipeline_slot(slot_id);
    } catch (...) {
        return -1;
    }
}

int select_arena_bank_ctx(DeviceContextHandle ctx, uint32_t bank_id) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->select_arena_bank(bank_id);
    } catch (...) {
        return -1;
    }
}

int set_native_run_identity_ctx(
    DeviceContextHandle ctx, uint64_t run_id, uint64_t generation, uint64_t dispatch_id, uint64_t run_epoch
) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->set_native_run_identity(
            run_id, generation, dispatch_id, run_epoch
        );
    } catch (...) {
        return -1;
    }
}

uint64_t get_arena_bank_gm_heap_base_ctx(DeviceContextHandle ctx, uint32_t bank_id) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->arena_bank_gm_heap_base(bank_id);
    } catch (...) {
        return 0;
    }
}

uint64_t get_retained_temp_addr_ctx(DeviceContextHandle ctx, uint32_t slot_id) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->retained_temp_addr(slot_id);
    } catch (...) {
        return 0;
    }
}

int simpler_unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
    if (ctx == NULL) return -1;
    try {
        DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
        if (runner->native_runs_outstanding()) {
            LOG_ERROR(
                "simpler_unregister_callable: native run must be finalized before mutating the callable registry"
            );
            return -1;
        }
        return runner->unregister_callable(callable_id);
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

size_t get_run_stream_set_create_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->run_stream_set_create_count();
    } catch (...) {
        return 0;
    }
}

size_t get_native_execution_thread_create_count(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->native_execution_thread_create_count();
    } catch (...) {
        return 0;
    }
}

size_t committed_device_memory_ctx(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->committed_device_memory();
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
