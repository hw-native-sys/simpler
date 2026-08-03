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
#include "native_run_state.h"

#include <dlfcn.h>
#include <pthread.h>

#include <cstdlib>
#include <cstring>
#include <new>
#include <thread>
#include <utility>
#include <vector>

#include "common/device_phase.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "cpu_sim_context.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"

using SimNativeRunState = NativeRunState<SimDeviceRunnerBase>;
// Phase entry points validate raw caller storage before beginning object
// lifetime, so the on-storage magic must remain the leading bytes.
static_assert(__builtin_offsetof(SimNativeRunState, magic) == 0, "native-run magic must lead runtime storage");

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in runtime_maker.cpp)
 * =========================================================================== */
int register_callable_impl(const ChipCallable *callable, uint64_t (*upload_fn)(const void *), CallableArtifacts *out);
int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc);

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

// The HostApi is a set of context-free function pointers: each wrapper above
// recovers its runner from the thread-local current_runner(), so a single
// filled table is valid for every runner and every run. Build it once at load
// time rather than reassembling the pointer table on each simpler_run. Passed by
// address into bind_callable_to_runtime_impl / validate_runtime_impl.

// Weak no-op default lives in device_runner_base.cpp; tensormap_and_ringbuffer
// links a strong override that builds + caches the prebuilt runtime-arena.
// simpler_init calls it directly for the fork-constant ring sizing.
extern "C" int prewarm_config_impl(
    const HostApi *api, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
);

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
 * =========================================================================== */

void destroy_device_context(DeviceContextHandle ctx) {
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
    if (runner != nullptr && runner->native_run_active()) {
        LOG_ERROR("destroy_device_context: refusing to destroy a context with an unfinalized native run");
        return;
    }
    delete runner;
}

size_t get_runtime_size(void) { return sizeof(SimNativeRunState); }

size_t get_runtime_alignment(void) { return alignof(SimNativeRunState); }

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
        if (runner->native_run_active()) {
            LOG_ERROR("finalize_device: native run must be finalized first");
            return -1;
        }
        int rc = runner->finalize();
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
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *prewarm_config
) {
    // Sim has no AICPU dispatcher (the simulator runs AICPU in-process). Accept
    // the parameters for ABI parity with the onboard implementation and ignore
    // them — callers that pass dispatcher bytes get the same shape as onboard,
    // and the dispatcher / preinstall load path on sim isn't taken anyway.
    (void)dispatcher_binary;
    (void)dispatcher_size;

    if (ctx == NULL) return -1;

    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
    // HostApi callbacks, including the prewarm path below, recover their
    // DeviceRunner from this thread-local binding.
    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });

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

    // Prebuilt runtime-arena prewarm for the fork-constant ring sizing, now that
    // the runner is attached. trb links a strong prewarm_config_impl; other
    // runtimes link the weak no-op. Only the ring sizing is read.
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
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
    if (runner->native_run_active()) {
        LOG_ERROR("simpler_register_callable: native run must be finalized before mutating the callable registry");
        return -1;
    }

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);

    try {
        CallableArtifacts artifacts;
        auto chip_buffer_guard = RAIIScopeGuard([runner, &artifacts]() {
            if (artifacts.chip_buffer_hash != 0) {
                runner->release_chip_callable_buffer(artifacts.chip_buffer_hash);
            }
        });
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
                callable_id, artifacts.chip_buffer_hash, artifacts.host_dlopen_handle, artifacts.host_orch_func_ptr,
                std::move(kernel_addrs), std::move(artifacts.signature)
            );
            if (rc == 0) {
                host_dlopen_guard.dismiss();
                chip_buffer_guard.dismiss();
            }
        } else {
            rc = runner->record_device_orch_callable(
                callable_id, artifacts.chip_buffer_hash, artifacts.chip_buffer_dev, artifacts.orch_so_data,
                artifacts.orch_so_size, artifacts.func_name.c_str(), artifacts.config_name.c_str(),
                std::move(kernel_addrs), std::move(artifacts.signature)
            );
            if (rc == 0) {
                chip_buffer_guard.dismiss();
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

// Runtime gate for device-domain phase emission. SIMPLER_DEVICE_STRACE_ENABLE=0
// suppresses the device (clk=dev) markers so a deployment can profile host and
// device independently; any other value (or unset) keeps them on. Host-side
// [STRACE] spans are unaffected — they ride SIMPLER_HOST_STRACE + the log level.
// Read once and cached (process-lifetime config knob).
static bool device_profiling_enabled() {
    static const bool enabled = [] {
        const char *v = std::getenv("SIMPLER_DEVICE_STRACE_ENABLE");
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

static SimNativeRunState *native_run_state(DeviceContextHandle ctx, RuntimeHandle runtime, const char *operation) {
    if (ctx == nullptr || runtime == nullptr) return nullptr;
    uint64_t magic = 0;
    std::memcpy(&magic, runtime, sizeof(magic));
    if (magic != SimNativeRunState::kMagic) {
        LOG_ERROR("%s: runtime does not contain a prepared native run", operation);
        return nullptr;
    }
    auto *state = static_cast<SimNativeRunState *>(runtime);
    if (state->runner != static_cast<SimDeviceRunnerBase *>(ctx)) {
        LOG_ERROR("%s: prepared run belongs to a different device context", operation);
        return nullptr;
    }
    return state;
}

static void emit_native_run_host_wall(unsigned trace_inv, uint64_t trace_hid, long long trace_start_ns) {
    const long long end_ns = STRACE_NOW_NS();
    STRACE_CONTEXT(trace_inv, trace_hid, 0);
    STRACE_HOST_SPAN_AT("simpler_run", trace_start_ns, end_ns - trace_start_ns, 0);
}

static int cleanup_failed_prepare(SimNativeRunState *state, int execution_rc, bool clear_gm_sm) {
    const unsigned trace_inv = state->trace_inv;
    const uint64_t trace_hid = state->trace_hid;
    const long long trace_start_ns = state->trace_start_ns;
    if (clear_gm_sm) state->runtime.set_gm_sm_ptr(nullptr);
    int validation_rc = -1;
    try {
        validation_rc = validate_runtime_impl(&state->runtime, &g_host_api, execution_rc);
    } catch (...) {
        validation_rc = -1;
    }
    if (state->runner_claimed) {
        state->runner->release_native_run(state);
        state->runner_claimed = false;
    }
    destroy_native_run_state(state);
    emit_native_run_host_wall(trace_inv, trace_hid, trace_start_ns);
    return validation_rc != 0 ? validation_rc : execution_rc;
}

int simpler_prepare_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config
) {
    if (ctx == nullptr || runtime == nullptr || config == nullptr) return -1;
    if (reinterpret_cast<uintptr_t>(runtime) % alignof(SimNativeRunState) != 0) {
        LOG_ERROR("simpler_prepare_run: runtime storage does not satisfy get_runtime_alignment()");
        return -1;
    }
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
    if (!runner->has_callable(callable_id)) {
        LOG_ERROR("simpler_prepare_run: callable_id=%d not registered", callable_id);
        return -1;
    }
    uint64_t magic = 0;
    std::memcpy(&magic, runtime, sizeof(magic));
    if (magic == SimNativeRunState::kMagic) {
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

    SimNativeRunState *state = nullptr;
    const uint64_t trace_hid = static_cast<uint64_t>(callable_id);
    const unsigned trace_inv = STRACE_ALLOC_INV();
    const long long trace_start_ns = STRACE_NOW_NS();
    try {
        state = new (runtime) SimNativeRunState(runner, *config, trace_hid);
        if (!runner->try_acquire_native_run(state, &state->launch_signal)) {
            LOG_ERROR("simpler_prepare_run: another native run is active on this device context");
            destroy_native_run_state(state);
            return -1;
        }
        state->runner_claimed = true;
        state->trace_inv = trace_inv;
        state->trace_start_ns = trace_start_ns;
        STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);

        int rc = runner->attach_current_thread(runner->device_id());
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);

        rc = runner->prepare_launch_shape(state->runtime, state->config);
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);

        runner->apply_call_config(state->config);

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
    SimNativeRunState *state = native_run_state(ctx, runtime, "simpler_launch_run");
    if (state == nullptr || state->phase.load(std::memory_order_acquire) != NativeRunPhase::Prepared) return -1;
    if (!state->runner_claimed || !state->runner->native_run_owned_by(state)) return -1;

    state->phase.store(NativeRunPhase::Launching, std::memory_order_release);

    try {
        // The compatibility backend uses one blocking executor per run. The
        // prepare-through-finalize runner claim limits it to one per context.
        state->executor = state->runner->create_thread([state, ctx]() {
            pthread_once(&g_runner_key_once, create_runner_key);
            pthread_setspecific(g_runner_key, ctx);
            STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);
            int rc = -1;
            try {
                int attach_rc = state->runner->attach_current_thread(state->runner->device_id());
                if (attach_rc == 0) {
                    state->adopt_host_thread_state();
                    {
                        STRACE("simpler_run.runner_run");
                        rc = state->runner->run(state->runtime, state->config);
                    }
                } else {
                    rc = attach_rc;
                }
            } catch (...) {
                rc = -1;
            }
            pthread_setspecific(g_runner_key, nullptr);
            state->execution_rc.store(rc, std::memory_order_relaxed);
            state->execution_done.store(true, std::memory_order_release);
            state->launch_signal.notify();
        });
    } catch (...) {
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
    SimNativeRunState *state = native_run_state(ctx, runtime, "simpler_poll_run");
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
    SimNativeRunState *state = native_run_state(ctx, runtime, "simpler_wait_run");
    if (state == nullptr) return -1;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    if (phase == NativeRunPhase::Prepared || phase == NativeRunPhase::Launching) return -1;
    if (state->executor.joinable()) state->executor.join();
    state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
    return state->execution_rc.load(std::memory_order_relaxed);
}

int simpler_finalize_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    SimNativeRunState *state = native_run_state(ctx, runtime, "simpler_finalize_run");
    if (state == nullptr) return -1;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    if (phase == NativeRunPhase::Launching) return -1;
    const unsigned trace_inv = state->trace_inv;
    const uint64_t trace_hid = state->trace_hid;
    const long long trace_start_ns = state->trace_start_ns;

    pthread_once(&g_runner_key_once, create_runner_key);
    pthread_setspecific(g_runner_key, ctx);
    auto tsd_guard = RAIIScopeGuard([]() {
        pthread_setspecific(g_runner_key, nullptr);
    });
    STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);

    int execution_rc = -1;
    const bool launched = phase != NativeRunPhase::Prepared;
    if (launched) {
        if (state->executor.joinable()) state->executor.join();
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

    if (state->runner_claimed) {
        state->runner->release_native_run(state);
        state->runner_claimed = false;
    }
    destroy_native_run_state(state);
    emit_native_run_host_wall(trace_inv, trace_hid, trace_start_ns);
    if (validation_rc != 0) return validation_rc;
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

int supports_concurrent_native_prepare_ctx(DeviceContextHandle) { return 0; }

int set_task_accepted_state_ctx(DeviceContextHandle ctx, volatile int32_t *state, int32_t accepted_value) {
    if (ctx == NULL) return -1;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->set_task_accepted_state(state, accepted_value);
    } catch (...) {
        return -1;
    }
}

/**
 * Simulation keeps no per-thread run selection, so the identity is carried only
 * by the onboard runner's trace attributes and is discarded here. Accepting it
 * keeps the pipeline symbol set uniform across every host runtime.
 */
int set_native_run_identity_ctx(DeviceContextHandle ctx, uint64_t, uint64_t, uint64_t, uint64_t) {
    return ctx == NULL ? -1 : 0;
}

int select_pipeline_slot_ctx(DeviceContextHandle ctx, uint32_t slot_id) {
    if (ctx == NULL) return -1;
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
    if (runner->native_run_active()) return -1;
    return runner->select_pipeline_slot(slot_id);
}

int select_arena_bank_ctx(DeviceContextHandle ctx, uint32_t bank_id) {
    if (ctx == NULL) return -1;
    SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
    if (runner->native_run_active()) return -1;
    return runner->select_arena_bank(bank_id);
}

uint64_t get_arena_bank_gm_heap_base_ctx(DeviceContextHandle ctx, uint32_t bank_id) {
    if (ctx == NULL) return 0;
    return static_cast<SimDeviceRunnerBase *>(ctx)->arena_bank_gm_heap_base(bank_id);
}

uint64_t get_retained_temp_addr_ctx(DeviceContextHandle ctx, uint32_t slot_id) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->retained_temp_addr(slot_id);
    } catch (...) {
        return 0;
    }
}

int simpler_unregister_callable(DeviceContextHandle ctx, int32_t callable_id) {
    if (ctx == NULL) return -1;
    try {
        SimDeviceRunnerBase *runner = static_cast<SimDeviceRunnerBase *>(ctx);
        if (runner->native_run_active()) {
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

size_t get_run_stream_set_create_count(DeviceContextHandle ctx) {
    // Simulation has no ACL streams, so it owns no run stream sets.
    (void)ctx;
    return 0;
}

size_t get_native_execution_thread_create_count(DeviceContextHandle ctx) {
    // Simulation retains its compatibility execution path.
    (void)ctx;
    return 0;
}

size_t committed_device_memory_ctx(DeviceContextHandle ctx) {
    if (ctx == NULL) return 0;
    try {
        return static_cast<SimDeviceRunnerBase *>(ctx)->committed_device_memory();
    } catch (...) {
        return 0;
    }
}

int simpler_provision_dma_workspace(DeviceContextHandle ctx, uint32_t required_mask) {
    // Simulation provides no async-DMA workspaces; a non-empty request fails
    // fast so an SDMA-enabled Worker cannot come up on sim.
    (void)ctx;
    return required_mask == 0 ? 0 : -1;
}

}  // extern "C"
