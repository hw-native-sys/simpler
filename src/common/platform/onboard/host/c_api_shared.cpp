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
#include "native_run_context.h"

#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>
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

using OnboardNativeRunContext = NativeRunContext<DeviceRunnerBase>;
// Phase entry points validate raw caller storage before beginning object
// lifetime, so the on-storage magic must remain the leading bytes.
static_assert(__builtin_offsetof(OnboardNativeRunContext, magic) == 0, "native-run magic must lead runtime storage");

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in each runtime's runtime_maker.cpp)
 * =========================================================================== */
int register_callable_impl(const ChipCallable *callable, const HostApi *api, CallableArtifacts *out);
int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc);
__attribute__((weak)) int concurrent_native_prepare_supported_impl(void) { return 0; }
__attribute__((weak)) int prepared_run_config_compatible_impl(
    const HostApi * /*api*/, const uint64_t * /*ring_task_window*/, const uint64_t * /*ring_heap*/,
    const uint64_t * /*ring_dep_pool*/
) {
    return 1;
}

/* ===========================================================================
 * Context-bound HostApi functions passed to runtime implementations.
 * =========================================================================== */

static void *device_malloc(void *runner_ctx, size_t size) {
    if (runner_ctx == nullptr) return nullptr;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->allocate_tensor(size);
    } catch (...) {
        return nullptr;
    }
}

static void device_free(void *runner_ctx, void *dev_ptr) {
    if (runner_ctx == nullptr || dev_ptr == nullptr) return;
    try {
        static_cast<DeviceRunnerBase *>(runner_ctx)->free_tensor(dev_ptr);
    } catch (...) {}
}

static int copy_to_device(void *runner_ctx, void *dev_ptr, const void *host_ptr, size_t size) {
    if (runner_ctx == nullptr || dev_ptr == nullptr || host_ptr == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return -1;
    }
}

static int copy_from_device(void *runner_ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    if (runner_ctx == nullptr || host_ptr == nullptr || dev_ptr == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return -1;
    }
}

static void *register_device_memory_to_host(void *runner_ctx, void *dev_ptr, size_t bytes) {
    if (runner_ctx == nullptr) return nullptr;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->register_device_memory_to_host(dev_ptr, bytes);
    } catch (...) {
        return nullptr;
    }
}

static void unregister_device_memory_from_host(void *runner_ctx, void *dev_ptr) {
    if (runner_ctx == nullptr) return;
    try {
        static_cast<DeviceRunnerBase *>(runner_ctx)->unregister_device_memory_from_host(dev_ptr);
    } catch (...) {}
}

static int device_memset(void *runner_ctx, void *dev_ptr, int value, size_t size) {
    if (runner_ctx == nullptr || dev_ptr == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->device_memset(dev_ptr, value, size);
    } catch (...) {
        return -1;
    }
}

static void get_retained_temp_buffer(void *runner_ctx, uint32_t pipeline_slot, void **addr, size_t *size) {
    if (runner_ctx == nullptr) {
        if (addr != nullptr) *addr = nullptr;
        if (size != nullptr) *size = 0;
        return;
    }
    try {
        static_cast<DeviceRunnerBase *>(runner_ctx)->get_retained_temp_buffer(pipeline_slot, addr, size);
    } catch (...) {
        if (addr != nullptr) *addr = nullptr;
        if (size != nullptr) *size = 0;
    }
}

static void set_retained_temp_buffer(void *runner_ctx, uint32_t pipeline_slot, void *addr, size_t size) {
    if (runner_ctx == nullptr) return;
    try {
        static_cast<DeviceRunnerBase *>(runner_ctx)->set_retained_temp_buffer(pipeline_slot, addr, size);
    } catch (...) {}
}

static bool
retained_temp_metadata_matches(void *runner_ctx, uint32_t pipeline_slot, const void *key_data, size_t key_size) {
    if (runner_ctx == nullptr) return false;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->retained_temp_metadata_matches(pipeline_slot, key_data, key_size);
    } catch (...) {
        return false;
    }
}

static void
set_retained_temp_metadata(void *runner_ctx, uint32_t pipeline_slot, const void *key_data, size_t key_size) {
    if (runner_ctx == nullptr) return;
    try {
        static_cast<DeviceRunnerBase *>(runner_ctx)->set_retained_temp_metadata(pipeline_slot, key_data, key_size);
    } catch (...) {}
}

static void *acquire_graph_execution_buffer(
    void *runner_ctx, uint32_t pipeline_slot, uint64_t graph_key, uint32_t occurrence, size_t bytes, size_t alignment
) {
    if (runner_ctx == nullptr) return nullptr;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->acquire_graph_execution_buffer(pipeline_slot, graph_key, occurrence, bytes, alignment);
    } catch (...) {
        return nullptr;
    }
}

static uint64_t upload_chip_callable_buffer_wrapper(void *runner_ctx, const void *callable) {
    if (runner_ctx == nullptr) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->upload_chip_callable_buffer(static_cast<const ChipCallable *>(callable));
    } catch (...) {
        return 0;
    }
}

static int setup_static_arena_wrapper(
    void *runner_ctx, uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size
) {
    if (runner_ctx == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->setup_static_arena(arena_bank, gm_heap_size, gm_sm_size, runtime_arena_size);
    } catch (...) {
        return -1;
    }
}

static void *acquire_pooled_gm_heap_wrapper(void *runner_ctx, uint32_t arena_bank) {
    if (runner_ctx == nullptr) return nullptr;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->acquire_pooled_gm_heap(arena_bank);
    } catch (...) {
        return nullptr;
    }
}

static void *acquire_pooled_gm_sm_wrapper(void *runner_ctx, uint32_t arena_bank) {
    if (runner_ctx == nullptr) return nullptr;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->acquire_pooled_gm_sm(arena_bank);
    } catch (...) {
        return nullptr;
    }
}

static void *acquire_pooled_runtime_arena_wrapper(void *runner_ctx, uint32_t arena_bank) {
    if (runner_ctx == nullptr) return nullptr;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->acquire_pooled_runtime_arena(arena_bank);
    } catch (...) {
        return nullptr;
    }
}

static bool lookup_prebuilt_runtime_arena_cache_wrapper(
    void *runner_ctx, uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void **gm_heap_base,
    void **sm_base, void **runtime_arena_base, size_t *runtime_off, const void **image_data, size_t *image_size
) {
    if (runner_ctx == nullptr) return false;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->lookup_prebuilt_runtime_arena_cache(
                arena_bank, hash, key_data, key_size, gm_heap_base, sm_base, runtime_arena_base, runtime_off,
                image_data, image_size
            );
    } catch (...) {
        return false;
    }
}

static void mark_prebuilt_runtime_arena_cached_wrapper(
    void *runner_ctx, uint32_t arena_bank, uint64_t hash, const void *key_data, size_t key_size, void *gm_heap_base,
    void *sm_base, void *runtime_arena_base, size_t runtime_off, const void *image_data, size_t image_size
) {
    if (runner_ctx == nullptr) return;
    try {
        static_cast<DeviceRunnerBase *>(runner_ctx)
            ->mark_prebuilt_runtime_arena_cached(
                arena_bank, hash, key_data, key_size, gm_heap_base, sm_base, runtime_arena_base, runtime_off,
                image_data, image_size
            );
    } catch (...) {}
}

// Weak no-op default lives in device_runner_base.cpp; tensormap_and_ringbuffer
// links a strong override that builds + caches the prebuilt runtime-arena.
// simpler_init calls it directly for the fork-constant ring sizing.
extern "C" int prewarm_config_impl(
    const HostApi *api, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
);

// One immutable function table is shared by all runners. Each HostApi value
// binds it to a specific runner and immutable per-run slot/bank selection.
static const HostApiOps g_host_api_ops = {
    .device_malloc = device_malloc,
    .device_free = device_free,
    .copy_to_device = copy_to_device,
    .copy_from_device = copy_from_device,
    .register_device_memory_to_host = register_device_memory_to_host,
    .unregister_device_memory_from_host = unregister_device_memory_from_host,
    .device_memset = device_memset,
    .get_retained_temp_buffer = get_retained_temp_buffer,
    .set_retained_temp_buffer = set_retained_temp_buffer,
    .retained_temp_metadata_matches = retained_temp_metadata_matches,
    .set_retained_temp_metadata = set_retained_temp_metadata,
    .acquire_graph_execution_buffer = acquire_graph_execution_buffer,
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

size_t get_runtime_size(void) { return sizeof(OnboardNativeRunContext); }

size_t get_runtime_alignment(void) { return alignof(OnboardNativeRunContext); }

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
            const HostApi prewarm_api(runner, 0, 0, &g_host_api_ops);
            rc = prewarm_config_impl(
                &prewarm_api, prewarm_config->runtime_env.ring_task_window, prewarm_config->runtime_env.ring_heap,
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

    try {
        int rc = runner->attach_current_thread(runner->device_id());
        if (rc != 0) return rc;

        CallableArtifacts artifacts;
        auto chip_buffer_guard = RAIIScopeGuard([runner, &artifacts]() {
            if (artifacts.chip_buffer_hash != 0) {
                runner->release_chip_callable_buffer(artifacts.chip_buffer_hash);
            }
        });
        const HostApi host_api(runner, 0, 0, &g_host_api_ops);
        rc = register_callable_impl(reinterpret_cast<const ChipCallable *>(callable), &host_api, &artifacts);
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

// Emit device-domain trace markers for the AICPU phases. RunWall (the whole
// on-NPU wall, i.e. the former RunTiming.device_wall) is emitted at depth 2
// under runner_run; its preamble/so_load/graph_build/post_orch subdivisions are
// emitted at depth 3 beneath it. Phases never stamped (0 ns) are skipped.
// Capture and emission share one gate, so a gated-off run performs no transfers
// for markers that cannot reach the log.
static void emit_device_phase_markers(DeviceRunnerBase *runner) {
    if (!device_phase_capture_enabled()) return;
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

static OnboardNativeRunContext *
native_run_context(DeviceContextHandle ctx, RuntimeHandle runtime, const char *operation) {
    if (ctx == nullptr || runtime == nullptr) return nullptr;
    uint64_t magic = 0;
    std::memcpy(&magic, runtime, sizeof(magic));
    if (magic != OnboardNativeRunContext::kMagic) {
        LOG_ERROR("%s: runtime does not contain a prepared native run", operation);
        return nullptr;
    }
    auto *state = static_cast<OnboardNativeRunContext *>(runtime);
    if (state->runner != static_cast<DeviceRunnerBase *>(ctx)) {
        LOG_ERROR("%s: prepared run belongs to a different device context", operation);
        return nullptr;
    }
    return state;
}

static void
emit_native_run_host_wall(uint64_t trace_inv, uint64_t trace_hid, long long trace_start_ns, const char *trace_attrs) {
    const long long end_ns = STRACE_NOW_NS();
    STRACE_CONTEXT(trace_inv, trace_hid, 0);
    STRACE_HOST_SPAN_AT_A("simpler_run", trace_start_ns, end_ns - trace_start_ns, 0, trace_attrs);
}

static void emit_native_run_runner_wall(OnboardNativeRunContext *state) {
    if (state->runner_trace_start_ns == 0) return;
    const long long end_ns = STRACE_NOW_NS();
    STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);
    STRACE_HOST_SPAN_AT(
        "simpler_run.runner_run", state->runner_trace_start_ns, end_ns - state->runner_trace_start_ns, 1
    );
    state->runner_trace_start_ns = 0;
}

int supports_concurrent_native_prepare_ctx(DeviceContextHandle ctx) {
    return ctx != nullptr && concurrent_native_prepare_supported_impl() != 0 ? 1 : 0;
}

static int cleanup_failed_prepare(OnboardNativeRunContext *state, int execution_rc, bool clear_gm_sm) {
    const uint64_t trace_inv = state->trace_inv;
    const uint64_t trace_hid = state->trace_hid;
    const long long trace_start_ns = state->trace_start_ns;
    char trace_attrs[sizeof(state->trace_attrs)];
    std::memcpy(trace_attrs, state->trace_attrs, sizeof(trace_attrs));
    if (clear_gm_sm) state->runtime.set_gm_sm_ptr(nullptr);
    int validation_rc = -1;
    try {
        validation_rc = validate_runtime_impl(&state->runtime, &state->host_api, execution_rc);
    } catch (...) {
        validation_rc = -1;
    }
    int resources_rc = 0;
    if (state->prepared_execution != nullptr) {
        try {
            state->runner->abandon_prepared_execution(*state->prepared_execution);
        } catch (...) {
            resources_rc = -1;
        }
    }
    if (state->runner_resources_owned) {
        try {
            int abandon_rc = state->runner->abandon_native_run_resources(state->descriptor.pipeline_slot);
            if (resources_rc == 0) resources_rc = abandon_rc;
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
    destroy_native_run_context(state);
    emit_native_run_host_wall(trace_inv, trace_hid, trace_start_ns, trace_attrs);
    if (validation_rc != 0) return validation_rc;
    if (resources_rc != 0) return resources_rc;
    return execution_rc;
}

int simpler_prepare_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor
) {
    if (ctx == nullptr || runtime == nullptr || config == nullptr || descriptor == nullptr) return -1;
    if (descriptor->pipeline_slot >= PTO_PIPELINE_MAX_DEPTH || descriptor->arena_bank >= PTO_PIPELINE_MAX_DEPTH) {
        LOG_ERROR(
            "simpler_prepare_run: descriptor selects slot=%u bank=%u outside [0, %u)", descriptor->pipeline_slot,
            descriptor->arena_bank, PTO_PIPELINE_MAX_DEPTH
        );
        return -1;
    }
    if (reinterpret_cast<uintptr_t>(runtime) % alignof(OnboardNativeRunContext) != 0) {
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
    if (magic == OnboardNativeRunContext::kMagic) {
        LOG_ERROR("simpler_prepare_run: runtime already contains a prepared run; finalize it before reuse");
        return -1;
    }
    if (magic != 0) {
        LOG_ERROR("simpler_prepare_run: runtime storage was not zero-initialized before its first use");
        return -1;
    }

    OnboardNativeRunContext *state = nullptr;
    const uint64_t trace_hid = runner->callable_hash(callable_id);
    const uint64_t trace_inv = STRACE_ALLOC_INV();
    const long long trace_start_ns = STRACE_NOW_NS();
    try {
        state = new (runtime) OnboardNativeRunContext(runner, *config, trace_hid, *descriptor, &g_host_api_ops);
        std::snprintf(
            state->trace_attrs, sizeof(state->trace_attrs),
            "run_id=%llu slot=%u generation=%llu dispatch_id=%llu run_epoch=%llu",
            static_cast<unsigned long long>(state->descriptor.run_id), state->descriptor.pipeline_slot,
            static_cast<unsigned long long>(state->descriptor.generation),
            static_cast<unsigned long long>(state->descriptor.dispatch_id),
            static_cast<unsigned long long>(state->descriptor.run_epoch)
        );
        const bool allow_prepared_successor =
            concurrent_native_prepare_supported_impl() != 0 && !config->diagnostics_any();
        if (!runner->try_reserve_native_run(
                state, state->descriptor.pipeline_slot, state->descriptor.arena_bank, allow_prepared_successor
            )) {
            LOG_ERROR("simpler_prepare_run: native-run admission is occupied (%s)", state->trace_attrs);
            destroy_native_run_context(state);
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
                    &state->host_api, config->runtime_env.ring_task_window, config->runtime_env.ring_heap,
                    config->runtime_env.ring_dep_pool
                );
            }
            if (compatibility_rc <= 0) {
                // A miss is normal — the successor keeps its lease and prepares
                // after the predecessor's fence — so it is reported at INFO and
                // kept distinct from a probe that failed to answer at all.
                // Without this the two are indistinguishable from outside: a
                // pipeline that silently never overlaps looks the same whether
                // the runtime_env disagrees or the feature is broken.
                if (compatibility_rc == 0) {
                    LOG_INFO(
                        "simpler_prepare_run: shared-arena layout differs from the active run; preparing at depth one "
                        "after its fence (%s)",
                        state->trace_attrs
                    );
                    compatibility_rc = PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE;
                } else {
                    LOG_ERROR(
                        "simpler_prepare_run: prepared-run compatibility probe failed: %d (%s)", compatibility_rc,
                        state->trace_attrs
                    );
                }
                return cleanup_failed_prepare(state, compatibility_rc, true);
            }
        }

        state->runner_resources_owned = true;
        rc = runner->provision_native_run_resources(state->descriptor.pipeline_slot);
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);

        rc = runner->prepare_launch_shape(state->runtime, state->config);
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);

        // Diagnostic binding reads runner-global collector configuration. It
        // is depth-one, while concurrent HBG preparation must leave the active
        // run's configuration untouched until launch.
        if (!overlaps_active_run) runner->apply_call_config(state->config);

        {
            STRACE("simpler_run.bind");
            rc = runner->bind_callable_to_runtime(
                state->runtime, callable_id, &state->host_api, args, state->config.runtime_env.ring_task_window,
                state->config.runtime_env.ring_heap, state->config.runtime_env.ring_dep_pool
            );
        }
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);
        rc = runner->prepare_execution(
            state->runtime, state->config, state->descriptor.pipeline_slot, state->identity(),
            &state->prepared_execution
        );
        if (rc != 0) return cleanup_failed_prepare(state, rc, true);
        state->runner_resources_owned = false;
        return 0;
    } catch (...) {
        if (state != nullptr) return cleanup_failed_prepare(state, -1, true);
        return -1;
    }
}

int simpler_launch_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunContext *state = native_run_context(ctx, runtime, "simpler_launch_run");
    if (state == nullptr || state->phase.load(std::memory_order_acquire) != NativeRunPhase::Prepared) return -1;
    // TEMPORARY (host_build_graph dsv4 bring-up): stop after prepare so the host
    // side — orchestration, graph construction, image relocation and H2D — can be
    // measured while the device execution of that graph still stalls. Sitting in
    // launch (not simpler_run) covers the split prepare/launch/wait entry points
    // the chip subprocess uses, which never call simpler_run. Outputs are never
    // produced, so any run under this variable is a timing harness, not a test.
    // Delete this together with the variable once the stall is diagnosed.
    if (std::getenv("SIMPLER_SKIP_DEVICE_RUN") != nullptr) {
        state->completion_rc = 0;
        state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
        return 0;
    }
    if (!state->runner->can_accept_run() || !state->runner_reserved) return -1;
    if (state->prepared_execution == nullptr ||
        !state->runner->try_acquire_native_run(state, state->identity(), &state->launch_permit)) {
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

    state->runner_trace_start_ns = STRACE_NOW_NS();
    int rc = -1;
    try {
        rc = state->runner->attach_current_thread(state->runner->device_id());
        if (rc == 0) {
            DeviceRunnerBase::LaunchOutcome launch =
                state->runner->launch_execution(std::move(state->prepared_execution), std::move(state->launch_permit));
            rc = launch.rc;
            state->prepared_execution = std::move(launch.prepared);
            state->active_execution = std::move(launch.active);
            if (launch.progress == LaunchProgress::Complete && !state->publish_acceptance(launch.receipt)) {
                LOG_ERROR("simpler_launch_run: launch receipt identity mismatch (%s)", state->trace_attrs);
                rc = -1;
            }
        }
    } catch (...) {
        rc = -1;
    }
    if (rc != 0) {
        state->completion_rc = rc;
        if (state->active_execution != nullptr) {
            state->phase.store(NativeRunPhase::Running, std::memory_order_release);
        } else {
            state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
            emit_native_run_runner_wall(state);
        }
        return rc;
    }
    state->completion_rc = 0;
    state->phase.store(NativeRunPhase::Running, std::memory_order_release);
    return 0;
}

int simpler_poll_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunContext *state = native_run_context(ctx, runtime, "simpler_poll_run");
    if (state == nullptr) return SIMPLER_NATIVE_RUN_POLL_ERROR;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    if (phase == NativeRunPhase::Prepared) return SIMPLER_NATIVE_RUN_POLL_ERROR;
    if (phase == NativeRunPhase::Complete) return SIMPLER_NATIVE_RUN_POLL_COMPLETE;
    int attach_rc = state->runner->attach_current_thread(state->runner->device_id());
    if (attach_rc != 0) return SIMPLER_NATIVE_RUN_POLL_ERROR;
    if (state->active_execution == nullptr) return SIMPLER_NATIVE_RUN_POLL_ERROR;
    return state->runner->poll_execution(*state->active_execution);
}

int simpler_wait_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunContext *state = native_run_context(ctx, runtime, "simpler_wait_run");
    if (state == nullptr) return -1;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    if (phase == NativeRunPhase::Prepared) return -1;
    if (phase == NativeRunPhase::Complete) return state->completion_rc;
    // drain_execution() synchronizes and destroys streams, reads device memory
    // and frees device allocations, all of which need this thread's CANN
    // device context. rtSetDevice is idempotent on an already-attached thread.
    int drain_rc = -1;
    try {
        drain_rc = state->runner->attach_current_thread(state->runner->device_id());
        if (drain_rc != 0) {
            LOG_ERROR("simpler_wait_run: attach_current_thread failed: %d (%s)", drain_rc, state->trace_attrs);
        } else {
            drain_rc = -1;
            if (state->active_execution != nullptr) {
                drain_rc = state->runner->drain_execution(*state->active_execution);
            }
        }
    } catch (...) {
        drain_rc = -1;
        LOG_ERROR("simpler_wait_run: drain threw (%s)", state->trace_attrs);
    }
    if (state->completion_rc == 0) state->completion_rc = drain_rc;
    state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
    emit_native_run_runner_wall(state);
    return state->completion_rc;
}

int simpler_finalize_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunContext *state = native_run_context(ctx, runtime, "simpler_finalize_run");
    if (state == nullptr) return -1;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    const uint64_t trace_inv = state->trace_inv;
    const uint64_t trace_hid = state->trace_hid;
    const long long trace_start_ns = state->trace_start_ns;
    char trace_attrs[sizeof(state->trace_attrs)];
    std::memcpy(trace_attrs, state->trace_attrs, sizeof(trace_attrs));

    STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);

    int execution_rc = state->completion_rc;
    // The launch transaction hands back an ActiveExecution only once it has
    // reached the device (LaunchProgress::Partial or Complete); a NotStarted
    // launch returns its PreparedExecution instead and leaves this null. So
    // `launched` means "this run owns device work" — it is what separates a run
    // that must be drained, whose rc is the run's result, and whose runtime
    // holds a live GM/SM pointer, from one that never touched a stream.
    const bool launched = state->active_execution != nullptr;
    // Both drain_execution() and validate_runtime_impl() touch the device, so
    // the attach covers each of them. rtSetDevice is idempotent on an
    // already-attached thread.
    int attach_rc = -1;
    try {
        attach_rc = state->runner->attach_current_thread(state->runner->device_id());
    } catch (...) {
        attach_rc = -1;
    }
    if (attach_rc != 0) {
        LOG_ERROR("simpler_finalize_run: attach_current_thread failed: %d (%s)", attach_rc, state->trace_attrs);
    }
    if (phase == NativeRunPhase::Running && launched) {
        int drain_rc = attach_rc;
        if (attach_rc == 0) {
            drain_rc = -1;
            try {
                drain_rc = state->runner->drain_execution(*state->active_execution);
            } catch (...) {
                LOG_ERROR("simpler_finalize_run: drain_execution threw (%s)", state->trace_attrs);
            }
        }
        if (execution_rc == 0) execution_rc = drain_rc;
        state->completion_rc = execution_rc;
        state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
    }
    emit_native_run_runner_wall(state);

    int validation_rc = -1;
    try {
        if (!launched) state->runtime.set_gm_sm_ptr(nullptr);
        if (attach_rc == 0) {
            {
                STRACE("simpler_run.validate");
                validation_rc = validate_runtime_impl(&state->runtime, &state->host_api, launched ? execution_rc : -1);
            }
            if (launched && execution_rc == 0) emit_device_phase_markers(state->runner);
        } else {
            validation_rc = attach_rc;
        }
    } catch (...) {
        validation_rc = -1;
    }

    int resources_rc = 0;
    if (state->prepared_execution != nullptr) {
        try {
            state->runner->abandon_prepared_execution(*state->prepared_execution);
        } catch (...) {
            resources_rc = -1;
        }
    }
    if (state->runner_resources_owned) {
        try {
            int abandon_rc = state->runner->abandon_native_run_resources(state->descriptor.pipeline_slot);
            if (resources_rc == 0) resources_rc = abandon_rc;
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
    destroy_native_run_context(state);
    emit_native_run_host_wall(trace_inv, trace_hid, trace_start_ns, trace_attrs);
    if (validation_rc != 0) return validation_rc;
    if (resources_rc != 0) return resources_rc;
    return launched ? execution_rc : 0;
}

int simpler_run(
    DeviceContextHandle ctx, RuntimeHandle runtime, int32_t callable_id, const void *args, const CallConfig *config,
    const NativeRunDescriptor *descriptor
) {
    int rc = simpler_prepare_run(ctx, runtime, callable_id, args, config, descriptor);
    if (rc != 0) return rc;
    rc = simpler_launch_run(ctx, runtime);
    if (rc == 0) rc = simpler_wait_run(ctx, runtime);
    int finalize_rc = simpler_finalize_run(ctx, runtime);
    return finalize_rc != 0 ? finalize_rc : rc;
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
