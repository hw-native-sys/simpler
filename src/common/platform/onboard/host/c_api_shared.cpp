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
 * Shared `runtime_c_api` glue — the byte-identical part of every arch's
 * onboard `runtime_c_api.cpp`. Linked into each arch's
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
#include "callable_protocol.h"
#include "call_config.h"
#include "device_runner_base.h"
#include "host/context_lifecycle.h"
#include "host/dep_gen_collector.h"  // make_deps_json_path
#include "host/kernel_entry_validation.h"
#include "host/kernel_pipeline_contract.h"
#include "worker/pipeline_contract.h"
#include "prepare_callable_common.h"
#include "run_retention_probe.h"
#include "runtime_c_api.h"
#include "task_args_wire.h"
#include "native_run_context.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#include "common/host_span.h"
#include "common/platform_config.h"
#include "common/host_span_scope.h"
#include "common/strace.h"
#include "common/unified_log.h"
#include "host/acl_error_log.h"
#include "host_log.h"
#include "host/host_clock_alignment_log.h"
#include "host/raii_scope_guard.h"
#include "runtime.h"
#include "platform_comm/comm.h"

// Forward-declared (rather than `#include "dlog_pub.h"`) so this TU does not
// require CANN's toolchain include path on the host build. Resolved at link
// time against `libunified_dlog.so` / `libascendalog.so`.
extern "C" int dlog_setlevel(int moduleId, int level, int enableEvent);

// Forward-declared for the same reason: the host-orchestrated graph capture lives
// in the host_build_graph runtime .so, and its header pulls in that runtime's own
// types. Each platform .so carries weak `false` / `-1` fallbacks for the runtimes
// that capture on the device instead — see each arch's device_runner.cpp.
extern "C" bool dep_gen_host_graph_active();
extern "C" int dep_gen_host_graph_emit(const char *deps_json_path);

using OnboardNativeRunContext = NativeRunContext<DeviceRunnerBase>;
// Phase entry points validate raw caller storage before beginning object
// lifetime, so the on-storage magic must remain the leading bytes.
static_assert(__builtin_offsetof(OnboardNativeRunContext, magic) == 0, "native-run magic must lead runtime storage");

/**
 * Write a host-orchestrated run's dependency graph, at the point its capture
 * window closes.
 *
 * The graph is complete when bind returns — host_build_graph runs its
 * orchestrator there — and it lives in state private to the thread that ran it.
 * Writing it here keeps the write on that thread and ahead of any later capture,
 * which is what the alternative (writing at drain) cannot promise: a drain may
 * land on another thread, and a successor's bind resets the capture state.
 *
 * The destination comes from this run's own config rather than the runner's,
 * which a concurrent prepare deliberately leaves untouched.
 *
 * A no-op for runtimes that capture on the device: their `dep_gen_host_graph_active`
 * is the weak `false`, and their graph is emitted from the collector at drain.
 *
 * With cross-run retention on, the graph is handed to the runner's exporter
 * here instead of being serialized here: the hand-off still happens on this
 * thread, which is the whole reason this call site exists, but the file is
 * written off the submit path. The synchronous branch below is unchanged.
 */
static void emit_host_dep_gen_graph(
    DeviceRunnerBase *runner, const CallConfig &config, uint64_t run_epoch, const char *trace_attrs
) {
    if (config.enable_dep_gen == 0 || !dep_gen_host_graph_active()) return;
    if (runner->host_graph_retains_runs()) {
        // The graph moves out of this thread's capture into storage the
        // exporter owns, and the file is published off this path. A failure is
        // the exporter's sticky error, which `flush_diagnostics` and `close`
        // report; this run's result is unchanged either way.
        if (!runner->seal_host_dep_gen_graph(DfxRunConfig::from(config), run_epoch)) {
            LOG_ERROR("dep_gen host graph was not handed to the background writer (%s)", trace_attrs);
        }
        return;
    }
    const std::string deps_path = make_deps_json_path(config.output_prefix);
    const int emit_rc = dep_gen_host_graph_emit(deps_path.c_str());
    if (emit_rc != 0) {
        LOG_ERROR("dep_gen host graph emit failed (%d) — deps.json not produced (%s)", emit_rc, trace_attrs);
    }
}

extern "C" {

/* ===========================================================================
 * Runtime Implementation Functions (defined in each runtime's runtime_maker.cpp)
 * =========================================================================== */
int register_callable_impl(const ChipCallable *callable, const HostApi *api, CallableArtifacts *out);
/**
 * Perform the device write a run's bind prepared.
 *
 * The bind computes its device execution image into staging that outlives it and
 * records where the bytes go; this ships them. Two steps rather than one, so the
 * write can be ordered against something — or captured and replayed — without
 * the host graph building that produced the bytes having to run again. Consumes
 * the record: publishing twice, or publishing a bind that recorded nothing, is an
 * error, because a run whose image never reached the device must not launch. A
 * runtime whose bind writes its own image where it builds it implements this as a
 * no-op.
 */
int publish_run_image_impl(Runtime *runtime, const HostApi *api);
/**
 * One run's input staging: copy each input-bearing binding's current host bytes
 * into the device buffer the bind gave it.
 *
 * Separate from the bind because the bind settles which device buffer each
 * caller tensor uses, not what is in it. This adapter calls it from
 * `simpler_prepare_run`, after the bind. A runtime whose host orchestrator reads
 * the inputs while it builds the graph stages them inside its own bind and
 * implements this as a no-op.
 */
int copy_in_run_inputs_impl(const Runtime *runtime, const HostApi *api);
/**
 * One run's result inspection: read the device-side runtime status when it
 * failed on the device, then copy every written tensor back to the caller's
 * buffers.
 *
 * Releases nothing, so that reading a run's results and retiring the device
 * memory behind them are separately orderable — which is what a partially
 * submitted run needs, where the results are readable but the bindings must be
 * retained until quiescence is proven. `launched` distinguishes a run that
 * reached a stream, and whose device-side status is therefore readable, from one
 * that never did; it is an argument rather than an inference from the image so
 * that a failing prepare does not have to mutate what it was about to publish.
 */
int copy_back_run_outputs_impl(const Runtime *runtime, const HostApi *api, int execution_rc, int launched);
/**
 * End the tensor bindings the bind recorded, releasing each the way its
 * provenance requires.
 */
int release_run_bindings_impl(Runtime *runtime, const HostApi *api);
__attribute__((weak)) int concurrent_native_prepare_supported_impl(void) { return 0; }
/**
 * Whether this runtime publishes a device-teardown report.
 *
 * The onboard platform runner is shared by every runtime built against it, so
 * the record it keeps is not by itself a statement about coverage. This is the
 * per-runtime gate that decides whether the record is a supported public
 * answer; a runtime that does not override it publishes nothing, which the
 * reader sees as "no observation" rather than as a teardown that did nothing.
 */
__attribute__((weak)) int teardown_report_supported_impl(void) { return 0; }
/**
 * Whether this runtime may have one run's native submission ordered behind
 * another's, so that a second run reaches the device while the first is still
 * executing.
 *
 * Per-runtime for the same reason the teardown report is: the platform runner is
 * shared, and what decides whether two launched runs are safe is the runtime's
 * own per-run state. A runtime that does not override this keeps the serial
 * path, where one run reaches the device at a time.
 */
__attribute__((weak)) int joined_native_launch_supported_impl(void) { return 0; }
/**
 * Whether a successor may be prepared into a shared arena bank beside the run
 * executing out of it.
 *
 * Asked only where two live runs select the same bank, so the question is
 * always "may this bind rewrite regions another run is still reading". Only the
 * runtime can answer it, from its own per-run state — and a runtime that has
 * not answered has not established anything, so the default is **no**. The
 * caller reads that as `PREPARED_INCOMPATIBLE` and prepares the successor after
 * its predecessor's fence, which is the behaviour that route had before a
 * second run could be prepared at all.
 *
 * A runtime whose arena regions are per-run never reaches this: its successor
 * gets its own bank from the slot lease and shares nothing to rule on.
 */
__attribute__((weak)) int prepared_run_config_compatible_impl(
    const HostApi * /*api*/, const uint64_t * /*ring_task_window*/, const uint64_t * /*ring_heap*/,
    const uint64_t * /*ring_dep_pool*/
) {
    return 0;
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
    if (runner_ctx == nullptr || dev_ptr == nullptr || host_ptr == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

static int copy_from_device(void *runner_ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    if (runner_ctx == nullptr || host_ptr == nullptr || dev_ptr == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
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

static void *acquire_child_memory_host_view(void *runner_ctx, void *dev_ptr, size_t bytes) {
    if (runner_ctx == nullptr) return nullptr;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->acquire_child_memory_host_view(dev_ptr, bytes);
    } catch (...) {
        return nullptr;
    }
}

static int device_memset(void *runner_ctx, void *dev_ptr, int value, size_t size) {
    if (runner_ctx == nullptr || dev_ptr == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->device_memset(dev_ptr, value, size);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
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

static int
acquire_retained_temp(void *runner_ctx, uint32_t pipeline_slot, size_t bytes, void **addr_out, size_t *size_out) {
    if (runner_ctx == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->acquire_retained_temp(pipeline_slot, bytes, addr_out, size_out);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

static int acquire_scheduler_state_storage(
    void *runner_ctx, uint32_t pipeline_slot, size_t bytes, size_t alignment, void **device_out, void **host_out
) {
    if (runner_ctx == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->acquire_scheduler_state_storage(pipeline_slot, bytes, alignment, device_out, host_out);
    } catch (...) {
        return -1;
    }
}

static int
acquire_graph_staging(void *runner_ctx, uint32_t pipeline_slot, size_t bytes, size_t alignment, void **staging_out) {
    if (runner_ctx == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->acquire_graph_staging(pipeline_slot, bytes, alignment, staging_out);
    } catch (...) {
        return -1;
    }
}

static int publish_graph_section(
    void *runner_ctx, uint32_t pipeline_slot, const void *bytes, size_t length, uint32_t *source_out, uint64_t *base_out
) {
    if (runner_ctx == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->publish_graph_section(pipeline_slot, bytes, length, source_out, base_out);
    } catch (...) {
        return -1;
    }
}

static void get_graph_definition_staging(void *runner_ctx, uint32_t pipeline_slot, void **addr, size_t *size) {
    if (addr != nullptr) *addr = nullptr;
    if (size != nullptr) *size = 0;
    if (runner_ctx == nullptr) return;
    try {
        static_cast<DeviceRunnerBase *>(runner_ctx)->get_graph_definition_staging(pipeline_slot, addr, size);
    } catch (...) {}
}

static int
acquire_sm_mirror(void *runner_ctx, uint32_t pipeline_slot, size_t bytes, size_t alignment, void **addr_out) {
    if (addr_out != nullptr) *addr_out = nullptr;
    if (runner_ctx == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->acquire_sm_mirror(pipeline_slot, bytes, alignment, addr_out);
    } catch (...) {
        return -1;
    }
}

static int
acquire_run_image_staging(void *runner_ctx, uint32_t pipeline_slot, size_t bytes, size_t alignment, void **addr_out) {
    if (addr_out != nullptr) *addr_out = nullptr;
    if (runner_ctx == nullptr) return -1;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->acquire_run_image_staging(pipeline_slot, bytes, alignment, addr_out);
    } catch (...) {
        return -1;
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

static uint32_t get_chip_swimlane_level(void *runner_ctx) {
    if (runner_ctx == nullptr) return 0;
    return static_cast<DeviceRunnerBase *>(runner_ctx)->chip_swimlane_level();
}

static bool publish_chip_swimlane_extension(
    void *runner_ctx, ChipSwimlaneExtensionSection section, const char *json_value, size_t json_size
) {
    return runner_ctx != nullptr &&
           static_cast<DeviceRunnerBase *>(runner_ctx)->publish_chip_swimlane_extension(section, json_value, json_size);
}

static void *host_phase_pool_arm(void *runner_ctx, uint32_t pipeline_slot, int producer_wants_records) {
    if (runner_ctx == nullptr) return nullptr;
    return static_cast<DeviceRunnerBase *>(runner_ctx)->host_phase_pool_arm(pipeline_slot, producer_wants_records != 0);
}

static void
host_phase_pool_finish(void *runner_ctx, uint32_t pipeline_slot, uint64_t submitted_tasks, uint64_t invocation_id) {
    if (runner_ctx == nullptr) return;
    static_cast<DeviceRunnerBase *>(runner_ctx)->host_phase_pool_finish(pipeline_slot, submitted_tasks, invocation_id);
}

static int setup_static_arena_wrapper(
    void *runner_ctx, uint32_t arena_bank, size_t gm_heap_size, size_t gm_sm_size, size_t runtime_arena_size
) {
    if (runner_ctx == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)
            ->setup_static_arena(arena_bank, gm_heap_size, gm_sm_size, runtime_arena_size);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
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

static const void *get_run_result(void *runner_ctx, uint32_t pipeline_slot, uint64_t run_epoch, size_t *bytes_out) {
    if (bytes_out != nullptr) *bytes_out = 0;
    if (runner_ctx == nullptr) return nullptr;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->device_run_result(pipeline_slot, run_epoch, bytes_out);
    } catch (...) {
        if (bytes_out != nullptr) *bytes_out = 0;
        return nullptr;
    }
}

// Weak no-op default lives in device_runner_base.cpp; tensormap_and_ringbuffer
// links a strong override that builds + caches the prebuilt runtime-arena.
// simpler_init calls it directly for the fork-constant ring sizing.
extern "C" int prewarm_config_impl(
    const HostApi *api, const uint64_t *ring_task_window, const uint64_t *ring_heap, const uint64_t *ring_dep_pool
);

// One immutable function table is shared by all runners. Each HostApi value
// binds it to a specific runner and immutable per-run slot/bank selection.
static int
declare_caller_device_writes(void *runner_ctx, uint64_t run_id, const CallerBufferSpan *spans, uint32_t count) {
    if (runner_ctx == nullptr || (count != 0 && spans == nullptr)) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    try {
        std::vector<CallerDeviceBuffers::Span> written(count);
        for (uint32_t i = 0; i < count; ++i) {
            written[i] = CallerDeviceBuffers::Span{spans[i].addr, spans[i].bytes};
        }
        std::size_t unresolved = 0;
        if (!static_cast<DeviceRunnerBase *>(runner_ctx)
                 ->declare_caller_buffer_writes(run_id, written.data(), written.size(), &unresolved)) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (unresolved != 0) {
            // Recorded, but not for every span the run named: an address this context cannot
            // resolve to a caller mint is an unknown owner, not an absent producer. The fence is
            // exactly as wide as what was recorded, so those bytes read as having no declared
            // producer — which is why this is said out loud rather than skipped silently.
            LOG_WARN(
                "declare_caller_device_writes: %zu of %u produced span(s) name no caller allocation of this "
                "device context; their bytes carry no readable-yet fence for a concurrently preparing run",
                unresolved, count
            );
        }
        return 0;
    } catch (...) {
        // Reported, never swallowed: a statement that was not recorded reads to every other run as
        // a buffer with no producer, so proceeding would serve bytes this run has not written.
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

static int caller_device_span_written_by_other_run(void *runner_ctx, uint64_t run_id, uint64_t addr, uint64_t bytes) {
    if (runner_ctx == nullptr) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(runner_ctx)->caller_buffer_written_by_other_run(run_id, addr, bytes) ?
                   1 :
                   0;
    } catch (...) {
        // Unreadable is the safe answer to a question that could not be answered: refusing one
        // access beats serving bytes a producer may not have written.
        return 1;
    }
}

static const HostApiOps g_host_api_ops = {
    .device_malloc = device_malloc,
    .device_free = device_free,
    .copy_to_device = copy_to_device,
    .copy_from_device = copy_from_device,
    .register_device_memory_to_host = register_device_memory_to_host,
    .unregister_device_memory_from_host = unregister_device_memory_from_host,
    .acquire_child_memory_host_view = acquire_child_memory_host_view,
    .device_memset = device_memset,
    .get_retained_temp_buffer = get_retained_temp_buffer,
    .set_retained_temp_buffer = set_retained_temp_buffer,
    .acquire_retained_temp = acquire_retained_temp,
    .acquire_graph_staging = acquire_graph_staging,
    .publish_graph_section = publish_graph_section,
    .get_graph_definition_staging = get_graph_definition_staging,
    .acquire_scheduler_state_storage = acquire_scheduler_state_storage,
    .acquire_sm_mirror = acquire_sm_mirror,
    .acquire_run_image_staging = acquire_run_image_staging,
    .setup_static_arena = setup_static_arena_wrapper,
    .acquire_pooled_gm_heap = acquire_pooled_gm_heap_wrapper,
    .acquire_pooled_gm_sm = acquire_pooled_gm_sm_wrapper,
    .acquire_pooled_runtime_arena = acquire_pooled_runtime_arena_wrapper,
    .lookup_prebuilt_runtime_arena_cache = lookup_prebuilt_runtime_arena_cache_wrapper,
    .mark_prebuilt_runtime_arena_cached = mark_prebuilt_runtime_arena_cached_wrapper,
    .upload_chip_callable_buffer = upload_chip_callable_buffer_wrapper,
    .get_chip_swimlane_level = get_chip_swimlane_level,
    .host_phase_pool_arm = host_phase_pool_arm,
    .host_phase_pool_finish = host_phase_pool_finish,
    .publish_chip_swimlane_extension = publish_chip_swimlane_extension,
    .get_run_result = get_run_result,
    .declare_caller_device_writes = declare_caller_device_writes,
    .caller_device_span_written_by_other_run = caller_device_span_written_by_other_run,
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
    // An unclosed kernel context still owns stream, event and argument handles
    // a captured ACLGraph may reference. Destroying it would free them under
    // the graph, so the context is deliberately leaked instead: the caller
    // closes it explicitly, or the process ends. The condition covers every
    // owner — a close whose stream and event destruction succeeded while an
    // argument release failed has not closed the context.
    if (runner != nullptr && runner->kernel_resources_live()) {
        LOG_ERROR("destroy_device_context: refusing to destroy an unclosed kernel context; leaving it alive");
        return;
    }
    delete runner;
}

size_t get_runtime_size(void) { return sizeof(OnboardNativeRunContext); }

size_t get_runtime_alignment(void) { return alignof(OnboardNativeRunContext); }

void *device_malloc_ctx(DeviceContextHandle ctx, size_t size) {
    if (ctx == NULL) return NULL;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->allocate_caller_buffer(size);
    } catch (...) {
        return NULL;
    }
}

void device_free_ctx(DeviceContextHandle ctx, void *dev_ptr) {
    if (ctx == NULL || dev_ptr == NULL) return;
    try {
        // The same guarded release the recording mint pairs with, so this legacy entry cannot
        // release an allocation a run may still reach, nor leave the table holding an address
        // whose pages are gone. It returns void, so a refusal can only be logged — which is
        // still the right outcome: not freeing is recoverable, freeing under a live borrow is
        // not. `device_free_caller_buffer_ctx` is the entry that reports it.
        (void)static_cast<DeviceRunnerBase *>(ctx)->free_caller_buffer(dev_ptr);
    } catch (...) {}
}

int device_free_caller_buffer_ctx(DeviceContextHandle ctx, void *dev_ptr) {
    if (ctx == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    if (dev_ptr == NULL) return 0;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->free_caller_buffer(dev_ptr);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int device_borrow_caller_buffers_ctx(
    DeviceContextHandle ctx, const CallerBufferSpan *spans, uint32_t count, uint64_t borrow_id
) {
    if (ctx == NULL || borrow_id == 0) return PTO_RUNTIME_ERR_INTERNAL;
    if (count != 0 && spans == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        std::vector<CallerDeviceBuffers::Span> resolved(count);
        for (uint32_t i = 0; i < count; ++i) {
            resolved[i] = CallerDeviceBuffers::Span{spans[i].addr, spans[i].bytes};
        }
        const bool held =
            static_cast<DeviceRunnerBase *>(ctx)->borrow_caller_buffers(borrow_id, resolved.data(), resolved.size());
        return held ? 0 : PTO_RUNTIME_ERR_INVALID_STATE;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

void device_release_caller_buffers_ctx(DeviceContextHandle ctx, uint64_t borrow_id, int keep) {
    if (ctx == NULL || borrow_id == 0) return;
    try {
        static_cast<DeviceRunnerBase *>(ctx)->release_caller_buffers(borrow_id, keep != 0);
    } catch (...) {}
}

int copy_to_device_ctx(DeviceContextHandle ctx, void *dev_ptr, const void *host_ptr, size_t size) {
    if (ctx == NULL || dev_ptr == NULL || host_ptr == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->copy_to_device(dev_ptr, host_ptr, size);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int copy_from_device_ctx(DeviceContextHandle ctx, void *host_ptr, const void *dev_ptr, size_t size) {
    if (ctx == NULL || host_ptr == NULL || dev_ptr == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->copy_from_device(host_ptr, dev_ptr, size);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

namespace {

/** This runner's teardown-proof storage, for the shared lifecycle operations. */
TeardownProofSlot proof_slot(DeviceRunnerBase *runner) {
    TeardownProofSlot slot;
    slot.ctx = runner;
    slot.get = [](void *ctx) {
        return static_cast<DeviceRunnerBase *>(ctx)->teardown_proof();
    };
    slot.set = [](void *ctx, TeardownProof proof) {
        static_cast<DeviceRunnerBase *>(ctx)->set_teardown_proof(proof);
    };
    return slot;
}

}  // namespace

int finalize_device(DeviceContextHandle ctx) {
    if (ctx == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    try {
        const int rc = runner->validate_current_context();
        if (rc != 0) return rc;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    ContextTeardownSteps steps;
    steps.ctx = runner;
    steps.runs_outstanding = [](void *c) {
        const bool outstanding = static_cast<DeviceRunnerBase *>(c)->native_runs_outstanding();
        if (outstanding) LOG_ERROR("finalize_device: native run must be finalized first");
        return outstanding;
    };
    // Publish whatever is still retained before any collector storage is
    // released. A no-op on a collector that retains no run, which is the
    // default.
    steps.publish_retained = [](void *c) {
        static_cast<DeviceRunnerBase *>(c)->finish_retained_runs();
    };
    steps.cleanup = [](void *c) {
        return static_cast<DeviceRunnerBase *>(c)->finalize();
    };
    return run_context_teardown(steps, proof_slot(runner));
}

int simpler_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *prewarm_config, int enable_sdma, const void *sdma_warmup_binary, uint64_t sdma_warmup_size
) {
    if (ctx == NULL) return PTO_RUNTIME_ERR_INTERNAL;

    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);

    // Latching the identity is the first thing this entry does, so a context
    // that already belongs to kernel mode is refused before any process- or
    // runner-state mutation below. Latching PROGRAM is idempotent, which is
    // what lets an init -> finalize -> init sequence on the same device run
    // again.
    const int latch_rc = runner->execution_mode_latch().latch(SIMPLER_MODE_PROGRAM);
    if (latch_rc != 0) {
        LOG_ERROR("simpler_init: refused — this context already belongs to kernel mode");
        return latch_rc;
    }

    // Mode isolation first, then this handle's own lifecycle. The refusal
    // below touches no resource and leaves the context exactly as it was, so a
    // caller that goes on to close it properly keeps whatever proof that close
    // earns. Latching PROGRAM is idempotent and therefore says nothing about
    // whether this context is already live, which is what the check decides
    // from instead.
    const TeardownProofSlot slot = proof_slot(runner);
    const ContextInitAdmission admission =
        admit_context_init(slot, runner->native_runs_outstanding(), runner->device_id() >= 0);
    if (admission != ContextInitAdmission::Admitted) {
        LOG_ERROR(
            "simpler_init: refused — this context must be finalized cleanly before it is initialized again "
            "(reason %u, teardown proof %u, device_id %d)",
            static_cast<unsigned>(admission), static_cast<unsigned>(runner->teardown_proof()), runner->device_id()
        );
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }

    // Admitted, so the proof is already spent: every step from here on is a
    // post-consume one, and each exit records that this context is no longer
    // proved.
    //
    // The install is what puts every route that reaches this entry — a
    // Worker's, and a direct C caller's — under the same manager, and it runs
    // before the eager prewarm below, which is the first thing that takes
    // device memory. Keeping it here rather than at the staging call is what
    // keeps a kernel context from ever being managed.
    ContextInstallSteps install_steps;
    install_steps.ctx = runner;
    install_steps.install_workspace = [](void *c) {
        auto *self = static_cast<DeviceRunnerBase *>(c);
        const int workspace_rc = self->install_staged_workspace();
        if (workspace_rc != 0) {
            LOG_ERROR("simpler_init: workspace ownership management could not be installed: %d", workspace_rc);
        }
        return workspace_rc;
    };
    install_steps.clear_staging = [](void *c) {
        static_cast<DeviceRunnerBase *>(c)->clear_staged_workspace();
    };
    // CANN dlog must be levelled BEFORE the device context is opened
    // (rtSetDevice inside attach_current_thread): CANN snapshots the
    // device-side log session's level at context-open time, so a later
    // dlog_setlevel is a no-op for the device side. HostLogger is already bound
    // to the process-owned state by ChipWorker before this call. Skipped when
    // ASCEND_GLOBAL_LOG_LEVEL is externally configured — CANN keeps that.
    install_steps.configure_logging = [](void *) {
        HostLogger::get_instance().configure_cann_log_level(dlog_setlevel);
    };
    const int install_rc = run_context_install(install_steps, slot);
    if (install_rc != 0) return install_rc;

    // Every exit below is past the consume above, so each one records that
    // this context's state is no longer proved. A caller must close it before
    // it may initialize again.
    int rc;
    try {
        rc = runner->attach_current_thread(device_id);
    } catch (...) {
        runner->mark_teardown_unresolved();
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (rc != 0) {
        runner->mark_teardown_unresolved();
        return rc;
    }

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
        // Recorded before the bring-up below, which provisions the workspace and
        // publishes its addresses in the same one-shot simpler_aicpu_init launch.
        const uint8_t *warmup_bytes = static_cast<const uint8_t *>(sdma_warmup_binary);
        std::vector<uint8_t> warmup_vec;
        if (warmup_bytes != NULL && sdma_warmup_size > 0) {
            warmup_vec.assign(warmup_bytes, warmup_bytes + sdma_warmup_size);
        }
        runner->set_dma_workspace_request(enable_sdma != 0, std::move(warmup_vec));
    } catch (...) {
        runner->mark_teardown_unresolved();
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Eagerly run the one-shot device setup: create persistent AICPU/AICore
    // streams, upload the dispatcher + inner SO bundle, resolve the per-symbol
    // rtFuncHandle for per-task launch, and provision + publish + warm the
    // async-DMA workspaces — so the first simpler_register_callable / simpler_run
    // does not pay any of these costs. Streams live until finalize_device; the
    // cached rtFuncHandle on LoadAicpuOp and the preinstall file both live until
    // ~DeviceRunner.
    try {
        rc = runner->ensure_device_initialized();
    } catch (...) {
        runner->mark_teardown_unresolved();
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (rc != 0) {
        runner->mark_teardown_unresolved();
        return rc;
    }

    // Prebuilt runtime-arena prewarm: the device is up, so build + cache the
    // arena for the fork-constant ring sizing now. trb provides a strong
    // prewarm_config_impl; other runtimes link the weak no-op. Only the ring
    // sizing is read.
    if (prewarm_config != NULL) {
        try {
            const HostApi prewarm_api(runner, 0, 0, 0, &g_host_api_ops);
            rc = prewarm_config_impl(
                &prewarm_api, prewarm_config->runtime_env.ring_task_window, prewarm_config->runtime_env.ring_heap,
                prewarm_config->runtime_env.ring_dep_pool
            );
        } catch (...) {
            runner->mark_teardown_unresolved();
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (rc != 0) {
            runner->mark_teardown_unresolved();
            return rc;
        }
    }
    return 0;
}

/* ===========================================================================
 * Per-callable_id preparation
 * =========================================================================== */

/**
 * Upload and record one callable on `runner`, leaving the AICPU-side
 * registration launch to the caller: program mode brings the device up lazily
 * and launches on the runner's own AICPU stream, kernel mode is already up and
 * launches on the kernel context's. `needs_aicpu_register` reports whether
 * that launch is owed — hbg resolves its orchestration host-side and owes
 * none.
 */
static int record_callable_on_runner(
    DeviceRunnerBase *runner, int32_t callable_id, const void *callable, bool *needs_aicpu_register
) {
    *needs_aicpu_register = false;
    CallableArtifacts artifacts;
    auto chip_buffer_guard = RAIIScopeGuard([runner, &artifacts]() {
        if (artifacts.chip_buffer_hash != 0) {
            runner->release_chip_callable_buffer(artifacts.chip_buffer_hash);
        }
    });
    const HostApi host_api(runner, 0, 0, 0, &g_host_api_ops);
    int rc = register_callable_impl(reinterpret_cast<const ChipCallable *>(callable), &host_api, &artifacts);
    if (rc != 0) return rc;

    auto host_dlopen_guard = RAIIScopeGuard([&artifacts]() {
        if (artifacts.host_dlopen_handle != nullptr) {
            dlclose(artifacts.host_dlopen_handle);
        }
    });

    // hbg's register_callable_impl populates host_dlopen_handle; trb's leaves
    // it null and fills orch_so_data + func_name/config_name.
    if (artifacts.host_dlopen_handle != nullptr) {
        rc = runner->record_host_orch_callable(
            callable_id, artifacts.chip_buffer_hash, artifacts.aicore_image_hash, artifacts.host_dlopen_handle,
            artifacts.host_orch_func_ptr, std::move(artifacts.signature)
        );
        if (rc != 0) return rc;
        host_dlopen_guard.dismiss();
        chip_buffer_guard.dismiss();
        return 0;
    }

    rc = runner->record_device_orch_callable(
        callable_id, artifacts.chip_buffer_hash, artifacts.aicore_image_hash, artifacts.chip_buffer_dev,
        artifacts.orch_so_data, artifacts.orch_so_size, artifacts.func_name.c_str(), artifacts.config_name.c_str(),
        std::move(artifacts.signature)
    );
    if (rc != 0) return rc;
    chip_buffer_guard.dismiss();
    *needs_aicpu_register = true;
    return 0;
}

int simpler_register_callable(DeviceContextHandle ctx, int32_t callable_id, const void *callable) {
    if (ctx == NULL || callable == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    if (runner->native_runs_outstanding()) {
        LOG_ERROR("simpler_register_callable: native run must be finalized before mutating the callable registry");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    try {
        int rc = runner->attach_current_thread(runner->device_id());
        if (rc != 0) return rc;

        bool needs_aicpu_register = false;
        rc = record_callable_on_runner(runner, callable_id, callable, &needs_aicpu_register);
        if (rc != 0) return rc;
        if (needs_aicpu_register) {
            rc = runner->launch_device_register(callable_id);
            if (rc != 0) {
                runner->unregister_callable(callable_id);
                return rc;
            }
        }
        return 0;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

// Emit device-domain trace markers for the AICPU phases. RunWall (the whole
// on-NPU wall, i.e. the former RunTiming.device_wall) is emitted at depth 2
// under runner_run; its preamble/so_load/graph_build/post_orch subdivisions are
// emitted at depth 3 beneath it. Phases never stamped (0 ns) are skipped.
// Capture and emission share one gate, so a gated-off run performs no transfers
// for markers that cannot reach the log.
static void emit_device_phase_markers(DeviceRunnerBase *runner, uint32_t pipeline_slot) {
    if (!device_phase_capture_enabled()) return;
    // One read of this run's own record: every field below belongs to the run
    // that owned `pipeline_slot`, not to whatever the runner is doing now.
    const DeviceRunnerBase::DeviceRunTiming &timing = runner->device_run_timing(pipeline_slot);
    const uint64_t run_wall_ns = timing.phase_ns[static_cast<int>(AicpuPhase::RunWall)];
    if (run_wall_ns != 0) {
        // `ts` stays 0: it is this run's device-clock origin, and the sub-phases
        // below are positioned against it, so containment would invert if this
        // bracket moved to an absolute instant. The raw RunWall bounds ride
        // alongside as attributes instead, because only an absolute tick is
        // comparable between two runs — the interval between run N's device end
        // and run N+1's device start is
        // `dev_start_cycle(N+1) - dev_end_cycle(N)`, differenced as integers and
        // converted with `dev_cnt_hz` afterwards.
        char dev_attrs[160];
        const int written = std::snprintf(
            dev_attrs, sizeof(dev_attrs), "clk=dev dev_id=%d dev_start_cycle=%llu dev_end_cycle=%llu dev_cnt_hz=%llu",
            timing.device_id, static_cast<unsigned long long>(timing.run_wall_start_cycles),
            static_cast<unsigned long long>(timing.run_wall_end_cycles),
            static_cast<unsigned long long>(timing.sys_cnt_hz)
        );
        if (written > 0 && static_cast<size_t>(written) < sizeof(dev_attrs)) {
            STRACE_DEV_SPAN_AT_A(
                "chip.run.runner_run.device_wall", 0, static_cast<long long>(run_wall_ns), 2, dev_attrs
            );
        } else {
            STRACE_DEV_SPAN_AT("chip.run.runner_run.device_wall", 0, static_cast<long long>(run_wall_ns), 2);
        }
    }
    struct PhaseName {
        AicpuPhase phase;
        const char *name;
    };
    static const PhaseName kPhases[] = {
        {AicpuPhase::Preamble, "chip.run.runner_run.device_wall.preamble"},
        {AicpuPhase::SoLoad, "chip.run.runner_run.device_wall.so_load"},
        {AicpuPhase::GraphBuild, "chip.run.runner_run.device_wall.graph_build"},
        {AicpuPhase::ConfigValidate, "chip.run.runner_run.device_wall.config_validate"},
        {AicpuPhase::ArenaWire, "chip.run.runner_run.device_wall.arena_wire"},
        {AicpuPhase::SmReset, "chip.run.runner_run.device_wall.sm_reset"},
        {AicpuPhase::PostOrch, "chip.run.runner_run.device_wall.post_orch"},
        {AicpuPhase::OrchWindow, "chip.run.runner_run.device_wall.orch"},
        {AicpuPhase::SchedWindow, "chip.run.runner_run.device_wall.sched"},
    };
    // RunWall is emitted above as device_wall; every other phase is in the table.
    static_assert(
        sizeof(kPhases) / sizeof(kPhases[0]) == NUM_AICPU_PHASES - 1,
        "kPhases[] must list every AicpuPhase except RunWall — add the new phase here"
    );
    for (const auto &p : kPhases) {
        const uint64_t ns = timing.phase_ns[static_cast<int>(p.phase)];
        if (ns != 0) {
            STRACE_DEV_SPAN_AT(
                p.name, static_cast<long long>(timing.phase_start_ns[static_cast<int>(p.phase)]),
                static_cast<long long>(ns), 3
            );
        }
    }

    // Selective task-timing slots: one span per complete slot, start = dispatch
    // and duration = finish - dispatch, both on the phase timeline so cross-slot
    // intervals (e.g. finish(slot_1) - dispatch(slot_0)) stay recoverable.
    // Untagged / incomplete slots read back 0/0 and are skipped.
    static const char *const kTaskSlotNames[NUM_TASK_TIMING_SLOTS] = {
        "chip.run.runner_run.device_wall.task_slot_0",  "chip.run.runner_run.device_wall.task_slot_1",
        "chip.run.runner_run.device_wall.task_slot_2",  "chip.run.runner_run.device_wall.task_slot_3",
        "chip.run.runner_run.device_wall.task_slot_4",  "chip.run.runner_run.device_wall.task_slot_5",
        "chip.run.runner_run.device_wall.task_slot_6",  "chip.run.runner_run.device_wall.task_slot_7",
        "chip.run.runner_run.device_wall.task_slot_8",  "chip.run.runner_run.device_wall.task_slot_9",
        "chip.run.runner_run.device_wall.task_slot_10", "chip.run.runner_run.device_wall.task_slot_11",
        "chip.run.runner_run.device_wall.task_slot_12", "chip.run.runner_run.device_wall.task_slot_13",
        "chip.run.runner_run.device_wall.task_slot_14", "chip.run.runner_run.device_wall.task_slot_15",
    };
    for (int s = 0; s < NUM_TASK_TIMING_SLOTS; ++s) {
        const uint64_t dispatch_ns = timing.task_slot_dispatch_ns[s];
        const uint64_t finish_ns = timing.task_slot_finish_ns[s];
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

/**
 * Publish one run's passive device-boundary times onto the host trace.
 *
 * A sibling of the device-wall span, on its own attribute budget and under the same non-diagnostic
 * capture gate, because it answers the question that wall cannot: the wall brackets
 * `aicpu_execute`, which returns at the AICore handshake, while `WholeOperatorEnd` is taken behind
 * the wait on this run's own AICore boundary and so is an instant at which its AICore kernel had
 * returned.
 *
 * Both values are `aclrtEventGetTimestamp` readings — "syscnt when event recorded", the device's
 * own counter — so a successor's `aic_start` is comparable with its predecessor's `wo_end` for
 * exactly the reason two runs' device walls are comparable: one free-running chip counter, never
 * reset per run. `ts_hz` is carried so a reader converts with the platform's own normalization
 * rather than assuming one, and so that a reader can see the quantum it is comparing at.
 *
 * A position that is unavailable is emitted as `*_rc=<why>` with no time. Nothing is emitted as a
 * time it does not have, so a reader cannot take an absence or an error for an ordering — and the
 * line is emitted for every launched run, including one at depth one whose whole-operator marker
 * was never recorded, so an absence is visible rather than silent.
 */
static void emit_device_boundary_marks(OnboardNativeRunContext *state) {
    if (!device_phase_capture_enabled()) return;
    const RunBoundaryMarks::Mark start =
        state->runner->run_boundary_mark(RunBoundaryMarks::Position::AicoreStart, state->identity());
    const RunBoundaryMarks::Mark end =
        state->runner->run_boundary_mark(RunBoundaryMarks::Position::WholeOperatorEnd, state->identity());
    char attrs[SIMPLER_HOST_SPAN_ATTRIBUTES_CAPACITY];
    (void)std::snprintf(
        attrs, sizeof(attrs), "clk=dev dev_id=%d aic_start=%llu aic_rc=%d wo_end=%llu wo_rc=%d ts_hz=%llu",
        state->runner->device_id(), static_cast<unsigned long long>(start.available ? start.timestamp : 0), start.rc,
        static_cast<unsigned long long>(end.available ? end.timestamp : 0), end.rc,
        static_cast<unsigned long long>(PLATFORM_ACL_EVENT_TIMESTAMP_FREQ_HZ)
    );
    STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);
    STRACE_HOST_SPAN_AT_A("chip.run.runner_run.device_boundaries", STRACE_NOW_NS(), 0, 2, attrs);
}

static void
emit_native_run_host_wall(uint64_t trace_inv, uint64_t trace_hid, long long trace_start_ns, const char *trace_attrs) {
    const long long end_ns = STRACE_NOW_NS();
    STRACE_CONTEXT(trace_inv, trace_hid, 0);
    STRACE_HOST_SPAN_AT_A("chip.run", trace_start_ns, end_ns - trace_start_ns, 0, trace_attrs);
}

static void emit_native_run_runner_wall(OnboardNativeRunContext *state) {
    if (state->runner_trace_start_ns == 0) return;
    const long long end_ns = STRACE_NOW_NS();
    STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);
    STRACE_HOST_SPAN_AT("chip.run.runner_run", state->runner_trace_start_ns, end_ns - state->runner_trace_start_ns, 1);
    state->runner_trace_start_ns = 0;
}

int supports_concurrent_native_prepare_ctx(DeviceContextHandle ctx) {
    return ctx != nullptr && concurrent_native_prepare_supported_impl() != 0 ? 1 : 0;
}

/**
 * Assemble what both of this run's evidence channels observed, without asking
 * the device anything.
 *
 * The boundary half is the observation the run's own drain or poll retained,
 * not a fresh poll: by the time this runs, that drain's cleanup has retired the
 * fence's arming, so a poll would answer `Error` about the fence rather than
 * about the run. The record half is the cached copy `read_device_run_result`
 * already took, with its read status, so a failed copy stays distinguishable
 * from a read never attempted. Neither half asks the device anything.
 */
static RunOutcomeEvidence collect_run_evidence(const OnboardNativeRunContext *state) {
    RunOutcomeEvidence evidence;
    evidence.boundaries = state->runner->observed_run_boundaries(state->identity());
    evidence.record_read =
        state->runner->device_run_result_read_status(state->descriptor.pipeline_slot, state->descriptor.run_epoch);
    evidence.terminal =
        state->runner->device_run_terminal(state->descriptor.pipeline_slot, state->descriptor.run_epoch);
    return evidence;
}

/**
 * Compare what the shared decision rule makes of this run against the
 * `execution_rc` the drain returned.
 *
 * The rule is `decide_run_execution`, the same one the fenced drain uses to
 * decide a normal success. The two agree by construction on that branch; what
 * this still covers is every shape the drain answered from the stream
 * synchronize or from a boundary failure instead. It reports and never
 * overrides: `execution_rc` is unchanged by this function and by everything it
 * calls. A run the rule cannot decide is not a disagreement — the producers
 * that publish nothing on a path are exactly what the audit is for, and the
 * reason names which path it was.
 */
static void report_terminal_disagreement(const OnboardNativeRunContext *state, int execution_rc) {
    const RunExecutionOutcome outcome = decide_run_execution(collect_run_evidence(state));
    switch (outcome.state) {
    case RunExecutionState::Succeeded:
        if (execution_rc != 0) {
            LOG_ERROR(
                "run terminal disagreement: the record decides success, execution reported %d (%s)", execution_rc,
                state->trace_attrs
            );
        }
        break;
    case RunExecutionState::Failed:
        if (execution_rc == 0) {
            LOG_ERROR(
                "run terminal disagreement: the record decides failure code %d (source %u), execution reported "
                "success (%s)",
                outcome.code, static_cast<unsigned>(outcome.source), state->trace_attrs
            );
        }
        break;
    case RunExecutionState::Pending:
    case RunExecutionState::Undecided:
        LOG_INFO(
            "run outcome %s: %s; execution reported %d (%s)", run_execution_state_name(outcome.state),
            outcome.reason != nullptr ? outcome.reason : "no reason given", execution_rc, state->trace_attrs
        );
        break;
    }
}

/**
 * Report one workspace fact for this run, at the boundary that produced it.
 *
 * A fact is never derived from a phase word or from the code a caller will
 * receive: this is called where the fact became true, and only there.
 */
static void note_workspace_fact(OnboardNativeRunContext *state, WorkspaceManager::RunFact fact) {
    if (state == nullptr || state->runner == nullptr) return;
    state->runner->note_workspace_run_fact(state->descriptor.pipeline_slot, state->descriptor.run_epoch, fact);
}

/**
 * Whether the loaded runtime declares its pooled arena regions per run.
 *
 * The declaration is what makes a per-slot bank meaningful: a `HOST_PER_RUN`
 * arena at a depth above one has one copy per run and every slot selects its
 * own, while a shared one has a single copy every slot selects. So this is the
 * statement a selected bank must agree with, read from the contract the loaded
 * runtime publishes rather than from any copy of it.
 */
static const PipelineResource *declared_arena_resource(const PipelineContract *contract) {
    if (contract == nullptr) return nullptr;
    const PipelineResource *arena = find_pipeline_resource(*contract, PTO_PIPELINE_GM_HEAP);
    if (arena == nullptr) arena = find_pipeline_resource(*contract, PTO_PIPELINE_RUNTIME_IMAGE);
    return arena;
}

static bool declares_per_run_arena() {
    const PipelineContract *contract = get_pipeline_contract();
    const PipelineResource *arena = declared_arena_resource(contract);
    if (arena == nullptr) return false;
    return pipeline_resource_copy_count(*contract, *arena) > 1;
}

/**
 * The arena bank this run's pipeline slot selects, derived from the contract the
 * loaded runtime publishes.
 *
 * The bank decides which copy of the pooled regions a bind writes and a run
 * executes out of, and those regions belong to this runtime — so the selection
 * is resolved here, from the declaration that governs them, rather than taken
 * from the descriptor a caller filled in. The two agree in every configuration
 * the contract describes: a per-run arena gives each slot its own bank whatever
 * depth a worker was granted, and a shared one gives every slot bank 0. They
 * can only disagree where the value did not survive the caller's own selection
 * or its transfer across this boundary, and a run prepared into the wrong bank
 * rewrites regions another run is executing from — which produces a wrong
 * result rather than a failure, so it must not be reachable through a
 * descriptor field.
 */
static uint32_t resolved_arena_bank_for_slot(uint32_t pipeline_slot) {
    const PipelineContract *contract = get_pipeline_contract();
    const PipelineResource *arena = declared_arena_resource(contract);
    if (arena == nullptr) return 0;
    return pipeline_resource_slot(*contract, *arena, PipelineSlotLease{pipeline_slot, 0, 0});
}

/**
 * What storage each live run's arena bank resolves to, at the one gate where a
 * successor prepares beside an executing predecessor.
 *
 * Two runs reach each other's pooled regions only if the bank their slot lease
 * selects resolves to the same storage, so this records what that selection
 * actually produced: the depth and arena classes the loaded runtime *declares*,
 * each live run's slot and the bank *selected* for it, and the committed heap
 * and runtime-image bases those banks name. Declared and selected are separate
 * fields because the selection reads the contract a worker latched, which is
 * what makes a differently-loaded contract distinguishable from a bank that was
 * recorded wrongly. A base reads 0 while its bank is uncommitted, which is not
 * an address — two zeros are two absences and establish no alias.
 *
 * The reservation table and the claim order are copied under the one mutex that
 * guards them and formatted afterwards; the arena lookups take no such lock and
 * allocate nothing. Nothing here changes what is admitted.
 */
static void emit_overlap_storage_marks(OnboardNativeRunContext *state, bool shares_arena_bank) {
    if (!simpler::host_trace::enabled() || state == nullptr || state->runner == nullptr) return;
    DeviceRunnerBase *runner = state->runner;
    DeviceRunnerBase::NativeRunReservationRow rows[PTO_PIPELINE_MAX_DEPTH];
    const size_t live = runner->snapshot_native_run_reservations(rows, PTO_PIPELINE_MAX_DEPTH);

    const PipelineContract *contract = get_pipeline_contract();
    const PipelineResource *heap =
        contract == nullptr ? nullptr : find_pipeline_resource(*contract, PTO_PIPELINE_GM_HEAP);
    const PipelineResource *image =
        contract == nullptr ? nullptr : find_pipeline_resource(*contract, PTO_PIPELINE_RUNTIME_IMAGE);

    char attrs[512];
    int at = std::snprintf(
        attrs, sizeof(attrs), "shares=%d decl_depth=%u heap_class=%d image_class=%d live=%zu",
        shares_arena_bank ? 1 : 0, contract == nullptr ? 0u : contract->pipeline_depth,
        heap == nullptr ? -1 : static_cast<int>(heap->resource_class),
        image == nullptr ? -1 : static_cast<int>(image->resource_class), live
    );
    for (size_t i = 0; i < live && at > 0 && static_cast<size_t>(at) < sizeof(attrs); ++i) {
        const void *heap_base = runner->acquire_pooled_gm_heap(rows[i].arena_bank);
        const void *image_base = runner->acquire_pooled_runtime_arena(rows[i].arena_bank);
        const int written = std::snprintf(
            attrs + at, sizeof(attrs) - static_cast<size_t>(at),
            " r%zu=%s/slot%u/bank%u/epoch%llu/heap0x%llx/img0x%llx", i,
            rows[i].owner == static_cast<const void *>(state) ? "self" : "peer", rows[i].pipeline_slot,
            rows[i].arena_bank, static_cast<unsigned long long>(rows[i].run_epoch),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(heap_base)),
            static_cast<unsigned long long>(reinterpret_cast<uintptr_t>(image_base))
        );
        if (written <= 0) break;
        at += written;
    }
    STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);
    STRACE_HOST_SPAN_AT_A("chip.run.bind.overlap", STRACE_NOW_NS(), 0, 1, attrs);
}

static int cleanup_failed_prepare(OnboardNativeRunContext *state, int execution_rc) {
    const uint64_t trace_inv = state->trace_inv;
    const uint64_t trace_hid = state->trace_hid;
    const long long trace_start_ns = state->trace_start_ns;
    char trace_attrs[sizeof(state->trace_attrs)];
    std::memcpy(trace_attrs, state->trace_attrs, sizeof(trace_attrs));
    // A prepare that failed produced no device work, so there is no status to
    // read and nothing written to copy back. Whatever bindings its bind got as
    // far as recording are this attempt's, and end with it.
    int validation_rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        validation_rc = release_run_bindings_impl(&state->runtime, &state->host_api);
    } catch (...) {
        validation_rc = PTO_RUNTIME_ERR_INTERNAL;
    }
    // A prepare that failed reached no launch transaction, so it owns no device
    // work. Its host borrowers are only proven gone when the release above both
    // returned and succeeded — a throw or a non-zero code leaves them unproven,
    // which is what keeps this run's blocks out of reuse.
    note_workspace_fact(state, WorkspaceManager::RunFact::NoDeviceSubmission);
    note_workspace_fact(state, WorkspaceManager::RunFact::CopybackReturned);
    if (validation_rc == 0) note_workspace_fact(state, WorkspaceManager::RunFact::BindingsReleased);
    int resources_rc = 0;
    if (state->prepared_execution != nullptr) {
        try {
            state->runner->abandon_prepared_execution(*state->prepared_execution);
        } catch (...) {
            resources_rc = PTO_RUNTIME_ERR_INTERNAL;
        }
    }
    if (state->runner_resources_owned) {
        try {
            int abandon_rc = state->runner->abandon_native_run_resources(state->descriptor.pipeline_slot);
            if (resources_rc == 0) resources_rc = abandon_rc;
        } catch (...) {
            resources_rc = PTO_RUNTIME_ERR_INTERNAL;
        }
        state->runner_resources_owned = false;
    }
    // Before the slot becomes reusable, for the same reason as in
    // simpler_finalize_run: a successor claiming this slot records against it,
    // and this run's terminal fact must land while the slot is still its own.
    // This path frees nothing — whatever it leaves unreferenced is released at
    // the next boundary that can prove the calling thread is attached, which
    // this one cannot (it is reached from the attach failure itself).
    note_workspace_fact(state, WorkspaceManager::RunFact::ContextDestroyed);
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
    if (ctx == nullptr || runtime == nullptr || config == nullptr || descriptor == nullptr)
        return PTO_RUNTIME_ERR_INTERNAL;
    if (descriptor->pipeline_slot >= PTO_PIPELINE_MAX_DEPTH || descriptor->arena_bank >= PTO_PIPELINE_MAX_DEPTH) {
        LOG_ERROR(
            "simpler_prepare_run: descriptor selects slot=%u bank=%u outside [0, %u)", descriptor->pipeline_slot,
            descriptor->arena_bank, PTO_PIPELINE_MAX_DEPTH
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (reinterpret_cast<uintptr_t>(runtime) % alignof(OnboardNativeRunContext) != 0) {
        LOG_ERROR("simpler_prepare_run: runtime storage does not satisfy get_runtime_alignment()");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    const int context_rc = runner->validate_current_context();
    if (context_rc != 0) return context_rc;
    if (runner->execution_mode_latch().is_kernel()) {
        if (!runner->kernel_execution_state().accepts_dispatch()) return PTO_RUNTIME_ERR_INVALID_STATE;
        const int args_rc = validate_kernel_run_args(args);
        if (args_rc != 0) {
            LOG_ERROR("simpler_prepare_run: kernel arguments require HOST/NONE or DEVICE/NONE");
            return args_rc;
        }
    }

    if (!runner->has_callable(callable_id)) {
        LOG_ERROR("simpler_prepare_run: callable_id=%d not registered", callable_id);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (!runner->accepts_new_run()) {
        LOG_ERROR("simpler_prepare_run: runner is unusable after a prior device failure");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    uint64_t magic = 0;
    std::memcpy(&magic, runtime, sizeof(magic));
    if (magic == OnboardNativeRunContext::kMagic) {
        LOG_ERROR("simpler_prepare_run: runtime already contains a prepared run; finalize it before reuse");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (magic != 0) {
        LOG_ERROR("simpler_prepare_run: runtime storage was not zero-initialized before its first use");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    OnboardNativeRunContext *state = nullptr;
    const uint64_t trace_hid = runner->callable_hash(callable_id);
    const uint64_t trace_inv = STRACE_ALLOC_INV();
    const long long trace_start_ns = STRACE_NOW_NS();
    try {
        // The bank is resolved from this runtime's own declaration before the
        // context is built, so every consumer of it — the reservation, the
        // HostApi the bind receives, `setup_static_arena` — reads one value with
        // one provenance. A descriptor that disagreed is reported and not used.
        NativeRunDescriptor resolved = *descriptor;
        resolved.arena_bank = resolved_arena_bank_for_slot(resolved.pipeline_slot);
        if (resolved.arena_bank != descriptor->arena_bank) {
            LOG_ERROR(
                "simpler_prepare_run: slot %u was given arena bank %u, but this runtime's declaration selects bank "
                "%u for it; using the declared selection",
                resolved.pipeline_slot, descriptor->arena_bank, resolved.arena_bank
            );
        }
        state = new (runtime) OnboardNativeRunContext(runner, *config, trace_hid, resolved, &g_host_api_ops);
        std::snprintf(
            state->trace_attrs, sizeof(state->trace_attrs),
            "run_id=%llu dispatch_id=%llu slot_id=%u generation=%llu run_epoch=%llu",
            static_cast<unsigned long long>(state->descriptor.run_id),
            static_cast<unsigned long long>(state->descriptor.dispatch_id), state->descriptor.pipeline_slot,
            static_cast<unsigned long long>(state->descriptor.generation),
            static_cast<unsigned long long>(state->descriptor.run_epoch)
        );
        const bool allow_prepared_successor = concurrent_native_prepare_supported_impl() != 0;
        if (!runner->try_reserve_native_run(
                state, state->descriptor.pipeline_slot, state->descriptor.arena_bank, allow_prepared_successor
            )) {
            LOG_ERROR("simpler_prepare_run: native-run admission is occupied (%s)", state->trace_attrs);
            destroy_native_run_context(state);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        state->runner_reserved = true;
        // Names the run every workspace request inside this prepare belongs to:
        // the arena commits below and the retained-temp grow record their
        // references against this identity.
        DeviceRunnerBase::WorkspacePlanScope workspace_plan(
            state->descriptor.pipeline_slot, state->descriptor.run_epoch
        );
        const bool overlaps_active_run = allow_prepared_successor && runner->native_run_active();
        state->trace_inv = trace_inv;
        state->trace_start_ns = trace_start_ns;
        if (config->enable_chip_swimlane >= 3) state->clock_log_offset = host_clock_alignment_log_offset();
        STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);

        int rc = runner->enter_run_thread();
        if (rc != 0) return cleanup_failed_prepare(state, rc);
        // This thread is now proven attached, which is what a device release
        // needs. Obsolete generations left by earlier runs — including a
        // staging a failed prepare abandoned — are released here rather than
        // at the boundary that produced them, because that boundary could not
        // prove attachment.
        //
        // A failure here is this call's only error, and it is reported rather
        // than dropped: a free or an unmap that did not happen is a device
        // fact, and a prepare that returned success would leave the caller
        // with no indication of it at all unless a later growth happened to
        // refuse. The cleanup path releases whatever this prepare had taken.
        const int reclaim_rc = runner->reclaim_workspace_obsolete();
        if (reclaim_rc != 0) return cleanup_failed_prepare(state, reclaim_rc);

        if (overlaps_active_run) {
            // The probe exists to protect a *shared* arena bank, so require it
            // only when this run actually shares one. A runtime whose arena
            // resources are per-run gets its own bank from the slot lease, and
            // `setup_static_arena` only ever touches the bank it is handed — so
            // its successor cannot grow or release the predecessor's regions and
            // has nothing for a probe to rule on. A runtime that shares a bank
            // and never implemented the probe has established nothing, and the
            // default answer says so.
            const bool shares_arena_bank =
                runner->arena_bank_shared_with_other_run(state, state->descriptor.arena_bank);
            emit_overlap_storage_marks(state, shares_arena_bank);
            // A run whose arena is declared per-run must not be holding a bank
            // another live run holds. The declaration and the selection are made
            // by different sides — the runtime declares the resource class, the
            // caller derives the bank from the slot lease — so the two can
            // disagree, and this is the only place that can see both. A
            // disagreement is not a layout a successor may prepare into: its
            // bind rewrites the regions the executing run reads, which produces
            // a wrong result rather than a failure. Refused as incompatible, so
            // the successor prepares after its predecessor's fence.
            if (shares_arena_bank && declares_per_run_arena()) {
                LOG_ERROR(
                    "simpler_prepare_run: slot %u selected arena bank %u, which a live run already holds, while this "
                    "runtime declares its arena per run; preparing at depth one after that run's fence (%s)",
                    state->descriptor.pipeline_slot, state->descriptor.arena_bank, state->trace_attrs
                );
                return cleanup_failed_prepare(state, PTO_RUNTIME_ERR_PREPARED_INCOMPATIBLE);
            }
            int compatibility_rc = shares_arena_bank ? 0 : 1;
            if (shares_arena_bank) {
                STRACE("chip.run.bind.compatibility");
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
                return cleanup_failed_prepare(state, compatibility_rc);
            }
        }

        state->runner_resources_owned = true;
        rc = runner->provision_native_run_resources(state->descriptor.pipeline_slot);
        if (rc != 0) return cleanup_failed_prepare(state, rc);

        rc = runner->prepare_launch_shape(state->runtime, state->config);
        if (rc != 0) return cleanup_failed_prepare(state, rc);

        // Latches what a device-context query answers from. Skipped for a
        // successor prepared against an active predecessor, whose configuration
        // is the one that query must keep reporting until it retires.
        if (!overlaps_active_run) runner->apply_call_config(state->config);

        // Unconditional, and from this run's own config: a host-orchestrating
        // runtime holds the captured graph in thread-local state between
        // orchestration and emit, so the arming has to happen on this thread
        // ahead of its bind whether or not it overlaps a predecessor. It writes
        // nothing the two runs share.
        runner->arm_host_dep_gen_capture(config->enable_dep_gen != 0);
        // Same reason, different state: a host-orchestrating bind records phase
        // events that belong to the run doing the binding rather than to
        // whichever run last held the claim.
        runner->begin_host_phase_run(state->descriptor.pipeline_slot, DfxRunConfig::from(*config));

        {
            STRACE("chip.run.bind");
            rc = runner->bind_callable_to_runtime(
                state->runtime, callable_id, &state->host_api, args, state->config.runtime_env.ring_task_window,
                state->config.runtime_env.ring_heap, state->config.runtime_env.ring_dep_pool
            );
        }
        if (rc != 0) return cleanup_failed_prepare(state, rc);
        // The bind prepared this run's device image into staging the slot owns;
        // this publishes it, before anything else touches the device arena. Two
        // calls rather than one because the write is orderable on its own: the
        // bytes are ready when the bind returns, and shipping them is this
        // caller's decision.
        {
            STRACE("chip.run.publish_image");
            rc = publish_run_image_impl(&state->runtime, &state->host_api);
        }
        if (rc != 0) {
            LOG_ERROR("simpler_prepare_run: publishing this run's image failed: %d (%s)", rc, state->trace_attrs);
            return cleanup_failed_prepare(state, rc);
        }
        emit_host_dep_gen_graph(runner, state->config, state->descriptor.run_epoch, state->trace_attrs);
        // This run's own input bytes, into the buffers its bind just named.
        {
            STRACE("chip.run.stage_inputs");
            rc = copy_in_run_inputs_impl(&state->runtime, &state->host_api);
        }
        if (rc != 0) {
            LOG_ERROR("simpler_prepare_run: staging this run's inputs failed: %d (%s)", rc, state->trace_attrs);
            return cleanup_failed_prepare(state, rc);
        }
        {
            STRACE("chip.run.prepare_execution");
            rc = runner->prepare_execution(
                state->runtime, state->config, state->descriptor.pipeline_slot, state->identity(),
                &state->prepared_execution
            );
        }
        if (rc != 0) return cleanup_failed_prepare(state, rc);
        // A launch-time property of the run, carried from the descriptor the
        // caller filled: whether it builds the boundary that makes it joinable.
        // Only honoured where the runtime supports being joined at all, so no
        // run constructs an edge the rest of this platform would never use.
        state->prepared_execution->joinable_boundary =
            state->descriptor.joinable_boundary != 0 && joined_native_launch_supported_impl() != 0;
        state->runner_resources_owned = false;
        return 0;
    } catch (...) {
        if (state != nullptr) return cleanup_failed_prepare(state, PTO_RUNTIME_ERR_INTERNAL);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

static int launch_prepared_run(OnboardNativeRunContext *state, const NativeRunJoin *join);

int simpler_launch_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunContext *state = native_run_context(ctx, runtime, "simpler_launch_run");
    if (state == nullptr || state->phase.load(std::memory_order_acquire) != NativeRunPhase::Prepared)
        return PTO_RUNTIME_ERR_INTERNAL;
    // TEMPORARY (host_build_graph dsv4 bring-up): stop after prepare so the host
    // side — orchestration, graph construction, image relocation and H2D — can be
    // measured while the device execution of that graph still stalls. Sitting in
    // launch (not simpler_run) covers the split prepare/launch/wait entry points
    // the chip subprocess uses, which never call simpler_run. Outputs are never
    // produced, so any run under this variable is a timing harness, not a test.
    // Delete this together with the variable once the stall is diagnosed.
    if (std::getenv("SIMPLER_SKIP_DEVICE_RUN") != nullptr) {
        // The host phase records describe the bind path this variable exists to
        // measure, so they are written here as well as in the device-run
        // teardown. Skipping the device must not skip the artifact.
        state->runner->write_host_phase_records_artifact(state->config.output_prefix, state->descriptor.pipeline_slot);
        state->completion_rc = 0;
        state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
        return 0;
    }
    return launch_prepared_run(state, nullptr);
}

int supports_joined_native_launch_ctx(DeviceContextHandle ctx) {
    if (ctx == nullptr || joined_native_launch_supported_impl() == 0) return 0;
    return static_cast<DeviceRunnerBase *>(ctx)->ready_to_join_launch() ? 1 : 0;
}

int simpler_launch_run_joined(DeviceContextHandle ctx, RuntimeHandle runtime, RuntimeHandle predecessor) {
    OnboardNativeRunContext *state = native_run_context(ctx, runtime, "simpler_launch_run_joined");
    if (state == nullptr || state->phase.load(std::memory_order_acquire) != NativeRunPhase::Prepared)
        return PTO_RUNTIME_ERR_INTERNAL;
    OnboardNativeRunContext *ahead = native_run_context(ctx, predecessor, "simpler_launch_run_joined");
    if (ahead == nullptr || ahead == state) return PTO_RUNTIME_ERR_INTERNAL;

    // Everything below this point is a refusal the caller is meant to absorb by
    // launching ordinarily instead, so it reports UNSUPPORTED and mutates
    // nothing: the run is still Prepared and still launchable once it reaches
    // the front. Only a malformed request is an error.
    if (joined_native_launch_supported_impl() == 0) return PTO_RUNTIME_ERR_UNSUPPORTED;
    if (!state->runner->ready_to_join_launch()) {
        LOG_INFO(
            "simpler_launch_run_joined: the runner cannot order a run behind another right now (%s)", state->trace_attrs
        );
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    if (ahead->phase.load(std::memory_order_acquire) != NativeRunPhase::Running || !ahead->runner_claimed) {
        LOG_INFO(
            "simpler_launch_run_joined: the named predecessor is not executing, so there is nothing to order behind "
            "(%s)",
            ahead->trace_attrs
        );
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    const NativeRunJoin join{ahead, ahead->identity()};
    if (!state->runner->has_whole_operator_boundary(join.predecessor_identity)) {
        LOG_INFO(
            "simpler_launch_run_joined: the named predecessor published no whole-operator boundary (%s)",
            ahead->trace_attrs
        );
        return PTO_RUNTIME_ERR_UNSUPPORTED;
    }
    // Deliberately *not* gated on whether that boundary has already fired. A
    // fired boundary is a satisfied ordering dependency, not a reason to refuse:
    // the queued wait is simply consumed at once, and the successor is still
    // correctly ordered after the predecessor's whole operator. Refusing there
    // would add a restriction the ordering model does not need — and it would
    // not establish anything either, because the boundary can fire in the
    // window between such a query and the submission completing.
    return launch_prepared_run(state, &join);
}

/**
 * Publish one joined launch's ordering observation onto the host trace.
 *
 * A point-in-time host span, on the mechanism the run's other markers already
 * use, so the record needs no API, no mailbox field and no protocol of its own
 * — and it is emitted at the default-visible timing tier, unlike a LOG_INFO
 * that the default threshold would suppress. It rides the run's own invocation
 * identity because the enclosing STRACE_CONTEXT is still bound here.
 *
 * Emitted whatever the answer. A reader that only ever saw the favourable case
 * could not tell silence from a negative result, and `observed` is separate
 * from `unfired` so a query that could not answer counts for neither side.
 */
static void emit_joined_launch_span(const DeviceRunnerBase::JoinedLaunchRecord &record) {
    char attrs[SIMPLER_HOST_SPAN_ATTRIBUTES_CAPACITY];
    (void)std::snprintf(
        attrs, sizeof(attrs),
        "s_epoch=%llu s_slot=%u s_disp=%llu p_epoch=%llu p_slot=%u p_disp=%llu observed=%d unfired=%d rc=%d",
        static_cast<unsigned long long>(record.successor.run_epoch), record.successor.pipeline_slot,
        static_cast<unsigned long long>(record.successor.dispatch_id),
        static_cast<unsigned long long>(record.predecessor.run_epoch), record.predecessor.pipeline_slot,
        static_cast<unsigned long long>(record.predecessor.dispatch_id), record.observed ? 1 : 0,
        record.predecessor_unfired ? 1 : 0, record.query_rc
    );
    STRACE_HOST_SPAN_AT_A("chip.run.joined_launch", STRACE_NOW_NS(), 0, 1, attrs);
}

/**
 * Take the execution claim and cross the device launch boundary.
 *
 * `join` is null for an ordinary launch, and then the claim is exclusive. A
 * non-null join is what the claim admits a second holder against, and what the
 * platform reads to queue the ordering edge; it is attached to the prepared run
 * at launch rather than at prepare because a prepared successor whose
 * predecessor retires first launches ordinarily.
 */
static int launch_prepared_run(OnboardNativeRunContext *state, const NativeRunJoin *join) {
    // The program launch transaction does not join the caller's stream.
    if (state->runner->execution_mode_latch().is_kernel()) return PTO_RUNTIME_ERR_INVALID_STATE;

    if (!state->runner->accepts_new_run() || !state->runner_reserved) return PTO_RUNTIME_ERR_INTERNAL;
    if (state->prepared_execution == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    state->prepared_execution->join = join != nullptr ? *join : NativeRunJoin{};
    if (!state->runner->try_acquire_native_run(state, state->identity(), &state->launch_permit, join)) {
        LOG_ERROR("launch_prepared_run: execution claim is occupied (%s)", state->trace_attrs);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    state->runner_claimed = true;
    // The active predecessor may poison the device after this successor was
    // prepared but before the execution claim becomes available, and the fault
    // channel may have matched a notice to this device in the same window.
    if (!state->runner->accepts_new_run()) {
        state->runner->release_native_run(state);
        state->runner_claimed = false;
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // launch_execution emits point-in-time markers from the DeviceRunner
    // without carrying trace identity through that interface. Restore the
    // prepared invocation on this API thread for the duration of the launch.
    STRACE_CONTEXT(state->trace_inv, state->trace_hid, 1);
    state->runner_trace_start_ns = STRACE_NOW_NS();
    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = state->runner->attach_current_thread(state->runner->device_id());
        if (rc == 0) {
            DeviceRunnerBase::LaunchOutcome launch =
                state->runner->launch_execution(std::move(state->prepared_execution), std::move(state->launch_permit));
            rc = launch.rc;
            // Where the fact becomes true, and from the transaction's own
            // report of how far it got rather than from a pointer that a later
            // unwind could clear: any progress past NotStarted means this run
            // owns device work. Recorded before the receipt check below, so a
            // failure there cannot lose it.
            if (launch.progress != LaunchProgress::NotStarted) {
                note_workspace_fact(state, WorkspaceManager::RunFact::Launched);
            }
            state->prepared_execution = std::move(launch.prepared);
            state->active_execution = std::move(launch.active);
            if (launch.progress == LaunchProgress::Complete && !state->publish_acceptance(launch.receipt)) {
                LOG_ERROR("launch_prepared_run: launch receipt identity mismatch (%s)", state->trace_attrs);
                rc = PTO_RUNTIME_ERR_INTERNAL;
            }
            // Only once this submission has succeeded, and only for a joined
            // one. Read here rather than before the launch because the question
            // is whether the *completed* enqueue preceded the predecessor's
            // completion; a read taken earlier leaves the window between it and
            // the submission returning, which is exactly the window in doubt.
            if (rc == 0 && join != nullptr && launch.progress == LaunchProgress::Complete) {
                emit_joined_launch_span(state->runner->note_joined_launch(*join, state->identity()));
            }
        }
    } catch (...) {
        rc = PTO_RUNTIME_ERR_INTERNAL;
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
    if (state == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    if (phase == NativeRunPhase::Prepared) return PTO_RUNTIME_ERR_INTERNAL;
    if (phase == NativeRunPhase::Complete) return state->completion_rc;
    // drain_execution() synchronizes and destroys streams, reads device memory
    // and frees device allocations, all of which need this thread's CANN
    // device context. rtSetDevice is idempotent on an already-attached thread.
    DrainOutcome drain{};
    // Published before the call, so a drain that lost its attach or threw is
    // never mistaken for one that never ran.
    note_workspace_fact(state, WorkspaceManager::RunFact::DrainAttempted);
    try {
        const int attach_rc = state->runner->attach_current_thread(state->runner->device_id());
        if (attach_rc != 0) {
            drain.device_rc = attach_rc;
            LOG_ERROR("simpler_wait_run: attach_current_thread failed: %d (%s)", attach_rc, state->trace_attrs);
        } else if (state->active_execution != nullptr) {
            drain = state->runner->drain_execution(*state->active_execution);
        }
    } catch (...) {
        drain = DrainOutcome{};
        LOG_ERROR("simpler_wait_run: drain threw (%s)", state->trace_attrs);
    }
    if (state->completion_rc == 0) state->completion_rc = drain.combined();
    // The device half alone proves this run's device work finished. A
    // diagnostics failure still fails the run above, but it is not evidence
    // about the device, and recording it as one would leave every workspace
    // block this run referenced permanently quarantined at context teardown.
    // The phase below is set on every path and proves nothing.
    if (drain.device_rc == 0) note_workspace_fact(state, WorkspaceManager::RunFact::DrainProvedComplete);
    state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
    emit_native_run_runner_wall(state);
    return state->completion_rc;
}

/**
 * #2267's late-read retention fixture. See run_retention_probe.h for what it
 * replaces and why production cannot produce the state it measures.
 *
 * This entry only resolves and validates the two runs; the sequence itself
 * lives beside the peer. On return the predecessor is still Running and the
 * successor is Complete, so the caller finalizes each exactly as it would after
 * an ordinary wait — the predecessor's drain, its DFX teardown and its
 * copy-back are all the production path, reached from the phase it is left in.
 */
int simpler_probe_run_retention(
    DeviceContextHandle ctx, RuntimeHandle runtime, RuntimeHandle runtime_successor,
    const RunRetentionProbeConfig *config, RunRetentionProbeReport *report
) {
    if (config == nullptr || report == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    OnboardNativeRunContext *state = native_run_context(ctx, runtime, "simpler_probe_run_retention");
    if (state == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (state->phase.load(std::memory_order_acquire) != NativeRunPhase::Running || state->active_execution == nullptr) {
        LOG_ERROR("simpler_probe_run_retention: the predecessor must be launched and still own device work");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    OnboardNativeRunContext *successor = nullptr;
    if (config->launch_successor != 0) {
        successor = native_run_context(ctx, runtime_successor, "simpler_probe_run_retention");
        if (successor == nullptr || successor == state) {
            LOG_ERROR("simpler_probe_run_retention: the successor must be a second prepared run");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        if (successor->phase.load(std::memory_order_acquire) != NativeRunPhase::Prepared ||
            successor->prepared_execution == nullptr) {
            LOG_ERROR("simpler_probe_run_retention: the successor must be prepared and not launched");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    }

    int rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        rc = state->runner->attach_current_thread(state->runner->device_id());
    } catch (...) {
        rc = PTO_RUNTIME_ERR_INTERNAL;
    }
    if (rc != 0) {
        LOG_ERROR("simpler_probe_run_retention: attach_current_thread failed: %d", rc);
        return rc;
    }

    std::unique_ptr<DeviceRunnerBase::ActiveExecution> active_successor;
    // The fixture performs the successor's drain itself, so the facts that
    // drain establishes are its to report. Starts as "no device evidence",
    // which is what a successor it never launched leaves it at.
    DrainOutcome successor_drain{};
    std::unique_ptr<DeviceRunnerBase::PreparedExecution> no_successor;
    // Passed as an lvalue so a successor the fixture never consumes — a refused
    // arm, or a launch that did not reach the device — comes back still owned by
    // its own context, which is what will release it.
    std::unique_ptr<DeviceRunnerBase::PreparedExecution> &successor_prepared =
        successor != nullptr ? successor->prepared_execution : no_successor;
    try {
        rc = run_retention_probe(
            *state->runner, *state->active_execution, successor_prepared, *config, report, &active_successor,
            &successor_drain
        );
    } catch (...) {
        LOG_ERROR("simpler_probe_run_retention: the sequence threw");
        rc = PTO_RUNTIME_ERR_INTERNAL;
    }

    if (successor != nullptr) {
        successor->active_execution = std::move(active_successor);
        if (successor->active_execution != nullptr) {
            // The fixture launched it, so the successor owns device work from
            // here. Reported for the successor's own identity, not this run's.
            note_workspace_fact(successor, WorkspaceManager::RunFact::Launched);
            // And the fixture drained it, which is the only place that drain is
            // performed: `simpler_finalize_run` skips its own drain for a run
            // already Complete, so a fact this call does not report is a fact
            // that run never gets. Without it the successor cannot retire, and
            // `ContextDestroyed` quarantines every block it referenced —
            // leaving the next run on its slot unable to register as a consumer
            // of a region the arena still publishes.
            note_workspace_fact(successor, WorkspaceManager::RunFact::DrainAttempted);
            // The device half alone, exactly as the ordinary wait decides it: a
            // diagnostics ownership failure still fails the run, and is not
            // evidence that the device finished.
            if (successor_drain.device_rc == 0) {
                note_workspace_fact(successor, WorkspaceManager::RunFact::DrainProvedComplete);
            }
            // The fixture drained it, so it reaches finalize in the same phase an
            // ordinary wait would leave it in. Set outright rather than only over
            // a zero: a context starts at -1 so a run that never completed cannot
            // read as success, and the fixture performed this run's launch and
            // drain, so it is what decides.
            successor->completion_rc = report->successor_drain_rc;
            successor->phase.store(NativeRunPhase::Complete, std::memory_order_release);
            emit_native_run_runner_wall(successor);
        }
        // Otherwise it never reached the device and is still Prepared, which is
        // the state finalize already knows how to abort.
    }
    return rc;
}

int simpler_finalize_run(DeviceContextHandle ctx, RuntimeHandle runtime) {
    OnboardNativeRunContext *state = native_run_context(ctx, runtime, "simpler_finalize_run");
    if (state == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    // A refused environment leaves the run owned for a retry on its context.
    const int context_rc = state->runner->validate_current_context();
    if (context_rc != 0) return context_rc;
    NativeRunPhase phase = state->phase.load(std::memory_order_acquire);
    const uint64_t trace_inv = state->trace_inv;
    const uint64_t trace_hid = state->trace_hid;
    const long long trace_start_ns = state->trace_start_ns;
    const uint64_t clock_log_offset = state->clock_log_offset;
    const std::string output_prefix = state->config.output_prefix;
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
    // An ownership fact, not a code. A live ActiveExecution proves this run
    // reached the device, so reporting a launch from it is sound; the converse
    // is not, which is why the launch transaction reports its own progress
    // where it happens. The ledger keeps that earlier report, so the branch
    // below can only ever add "never submitted" to a run no launch claimed.
    note_workspace_fact(
        state, launched ? WorkspaceManager::RunFact::Launched : WorkspaceManager::RunFact::NoDeviceSubmission
    );
    // Drain and copy-back need the correct current device. Program owns that
    // binding; a borrowed context only verifies the caller's binding.
    int attach_rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        attach_rc = state->runner->enter_run_thread();
    } catch (...) {
        attach_rc = PTO_RUNTIME_ERR_INTERNAL;
    }
    if (attach_rc != 0) {
        LOG_ERROR("simpler_finalize_run: attach_current_thread failed: %d (%s)", attach_rc, state->trace_attrs);
    }
    if (phase == NativeRunPhase::Running && launched) {
        DrainOutcome drain{};
        drain.device_rc = attach_rc;
        note_workspace_fact(state, WorkspaceManager::RunFact::DrainAttempted);
        if (attach_rc == 0) {
            drain.device_rc = PTO_RUNTIME_ERR_INTERNAL;
            try {
                drain = state->runner->drain_execution(*state->active_execution);
            } catch (...) {
                drain = DrainOutcome{};
                LOG_ERROR("simpler_finalize_run: drain_execution threw (%s)", state->trace_attrs);
            }
        }
        if (execution_rc == 0) execution_rc = drain.combined();
        // The device half alone; see `simpler_wait_run` for why a diagnostics
        // failure must not be read as an unproved device completion.
        if (drain.device_rc == 0) note_workspace_fact(state, WorkspaceManager::RunFact::DrainProvedComplete);
        state->completion_rc = execution_rc;
        state->phase.store(NativeRunPhase::Complete, std::memory_order_release);
    }
    emit_native_run_runner_wall(state);

    int validation_rc = PTO_RUNTIME_ERR_INTERNAL;
    try {
        if (attach_rc == 0) {
            // Immediately before the consumer, and after whichever drain
            // completed the run — `simpler_wait_run` may have done it, leaving
            // nothing for the catch-up drain above. Read on every launched run,
            // not only the failing ones: the device publishes a terminal record
            // for a success too. One read per run, and a fenced drain has
            // already taken it to decide the run, so this call reuses those
            // bytes; it still owns the read for a run whose drain never
            // reached that point.
            if (launched) {
                (void)state->runner->read_device_run_result(
                    state->descriptor.pipeline_slot, state->descriptor.run_epoch
                );
                report_terminal_disagreement(state, execution_rc);
                // A separate axis from this run's outcome: a notification names a
                // device and a stream, carries no run identity, and can arrive
                // late — 16 s is the longest lag measured, not a bound the SDK
                // promises — so it can name a fault from an earlier run than this
                // one, and cannot say whether any run was impaired. `execution_rc`
                // is settled above and is not read or written here; a matched
                // notice refuses *future* admission and nothing else.
                const uint64_t own_stream_faults = state->runner->consume_device_fault_notices();
                if (own_stream_faults != 0) {
                    LOG_WARN(
                        "device fault channel matched %llu notice(s) to this runner's run streams while finalizing "
                        "%s; this run's own outcome is unchanged and the device refuses further admission",
                        static_cast<unsigned long long>(own_stream_faults), state->trace_attrs
                    );
                }
            }
            {
                STRACE("chip.run.validate");
                validation_rc = copy_back_run_outputs_impl(
                    &state->runtime, &state->host_api, launched ? execution_rc : PTO_RUNTIME_ERR_INTERNAL,
                    launched ? 1 : 0
                );
                // This run is the only user of its bindings, so they end here,
                // after its outputs have come back through them.
                // Returning at all is what ends the copy-back reader, whether
                // it read this run's outputs or skipped them for a failed run;
                // the bindings are only proven released when their own code is
                // zero, because a failed release may leave a borrower behind.
                note_workspace_fact(state, WorkspaceManager::RunFact::CopybackReturned);
                const int release_rc = release_run_bindings_impl(&state->runtime, &state->host_api);
                if (release_rc == 0) note_workspace_fact(state, WorkspaceManager::RunFact::BindingsReleased);
                if (validation_rc == 0) validation_rc = release_rc;
            }
            if (launched && execution_rc == 0) {
                emit_device_phase_markers(state->runner, state->descriptor.pipeline_slot);
                // Placed here and only here: this run's completion is established, which is what
                // each retrieval's own host-blocking completion step is called after.
                emit_device_boundary_marks(state);
            }
        } else {
            validation_rc = attach_rc;
        }
    } catch (...) {
        validation_rc = PTO_RUNTIME_ERR_INTERNAL;
    }

    // Unconditional: this run is over, so its slot's device-timing storage is
    // reusable whether or not the result was emitted above. A run that never
    // launched, failed, lost its attach, or threw has no result to read — but it
    // still owns the slot, so skipping this on those paths would leave the slot
    // armed forever and cost every later run on it its capture.
    state->runner->release_device_run_timing(state->descriptor.pipeline_slot);

    int resources_rc = 0;
    if (state->prepared_execution != nullptr) {
        try {
            state->runner->abandon_prepared_execution(*state->prepared_execution);
        } catch (...) {
            resources_rc = PTO_RUNTIME_ERR_INTERNAL;
        }
    }
    if (state->runner_resources_owned) {
        try {
            int abandon_rc = state->runner->abandon_native_run_resources(state->descriptor.pipeline_slot);
            if (resources_rc == 0) resources_rc = abandon_rc;
        } catch (...) {
            resources_rc = PTO_RUNTIME_ERR_INTERNAL;
        }
        state->runner_resources_owned = false;
    }

    const bool export_clock_log = launched && execution_rc == 0 && validation_rc == 0 &&
                                  state->runner->host_clock_alignment_log_required(state->descriptor.pipeline_slot);
    // Opportunistic, and only where attachment is proven: `attach_rc` is this
    // call's own result, and every device step above is guarded by it. This
    // keeps a steady-state workload from carrying an obsolete generation until
    // its next prepare, without ever guessing that the thread is attached.
    //
    // Recorded rather than returned here: this run's lease, slot and context
    // still have to be released below, and an error from a *previous* run's
    // block must not displace this run's own execution or validation failure.
    // It is folded into the return precedence last, so it is reported exactly
    // when nothing that outranks it went wrong.
    int reclaim_rc = 0;
    if (attach_rc == 0) {
        reclaim_rc = state->runner->reclaim_workspace_obsolete();
    }

    // Before the slot becomes reusable, not after: releasing the claim is "the
    // point a successor's launch becomes admissible", and a successor that
    // claims this slot records its own facts against it. This run's terminal
    // fact has to land while the slot is still this run's, or it would arrive
    // against a record that already belongs to somebody else.
    //
    // Nothing between here and the destruction below reports a further fact,
    // so "no further fact can arrive" is still true when it is recorded.
    note_workspace_fact(state, WorkspaceManager::RunFact::ContextDestroyed);
    if (state->runner_claimed) {
        // The point a successor's launch becomes admissible. Ordering a
        // successor's device work against this boundary is what separates a
        // pipelined launch from a reordered one, and no other span marks it.
        STRACE("chip.run.claim_release");
        state->runner->release_native_run(state);
        state->runner_claimed = false;
    }
    if (state->runner_reserved) {
        state->runner->release_native_run_reservation(state);
        state->runner_reserved = false;
    }
    destroy_native_run_context(state);
    emit_native_run_host_wall(trace_inv, trace_hid, trace_start_ns, trace_attrs);
    if (export_clock_log && !export_host_clock_alignment_log(output_prefix, trace_inv, clock_log_offset)) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (validation_rc != 0) return validation_rc;
    if (resources_rc != 0) return resources_rc;
    if (launched && execution_rc != 0) return execution_rc;
    // Last: a standalone workspace-reclamation failure is still a failure, and
    // with nothing above it to report this is where the caller learns of it.
    if (reclaim_rc != 0) return reclaim_rc;
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
    if (ctx == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
        if (runner->native_runs_outstanding()) {
            LOG_ERROR(
                "simpler_unregister_callable: native run must be finalized before mutating the callable registry"
            );
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        return runner->unregister_callable(callable_id);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
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

int get_teardown_report(DeviceContextHandle ctx, void *out, size_t out_bytes) {
    if (ctx == NULL || out == nullptr || out_bytes != sizeof(SimplerTeardownReport)) {
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    }
    // The runner is shared across the runtimes built on this platform, so the
    // gate is the runtime's own capability rather than whether a record
    // happens to exist.
    if (teardown_report_supported_impl() == 0) return PTO_RUNTIME_ERR_UNSUPPORTED;
    try {
        // A runner outside the recording scope answers false, which is the
        // difference between "this backend observed nothing" and "a teardown
        // that did nothing".
        return static_cast<DeviceRunnerBase *>(ctx)->copy_teardown_report(static_cast<SimplerTeardownReport *>(out)) ?
                   0 :
                   PTO_RUNTIME_ERR_UNSUPPORTED;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
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

int simpler_set_workspace_budget_ctx(DeviceContextHandle ctx, uint64_t limit_bytes) {
    if (ctx == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    // Unchanged for every existing caller: a positive limit is still a
    // complete request, asking for management *and* that limit. What changed
    // is when it takes effect — it is recorded here and installed by
    // simpler_init once the program-mode latch is taken, so the ownership
    // ledger exists before the prewarm allocates rather than after.
    //
    // Zero is still refused. A caller that wants management without a limit
    // uses simpler_enable_workspace_management_ctx instead, so the two
    // requests stay distinguishable and no sentinel value is overloaded.
    if (limit_bytes == 0) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->stage_workspace_management(limit_bytes);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int simpler_enable_workspace_management_ctx(DeviceContextHandle ctx) {
    if (ctx == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        return static_cast<DeviceRunnerBase *>(ctx)->stage_workspace_management(0);
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int simpler_get_workspace_report_ctx(DeviceContextHandle ctx, SimplerWorkspaceReport *out, size_t out_bytes) {
    if (ctx == NULL || out == NULL) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    // The caller's own sizeof: a module built against a shorter record is
    // refused rather than written past.
    if (out_bytes < sizeof(SimplerWorkspaceReport)) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    try {
        if (!static_cast<DeviceRunnerBase *>(ctx)->workspace_report(out)) return PTO_RUNTIME_ERR_UNSUPPORTED;
        return 0;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int simpler_set_retain_runs_ctx(DeviceContextHandle ctx, int32_t enabled) {
    if (ctx == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        static_cast<DeviceRunnerBase *>(ctx)->set_retain_runs(enabled != 0);
        return 0;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int simpler_set_dfx_session_ctx(DeviceContextHandle ctx, int32_t enabled) {
    return simpler_set_retain_runs_ctx(ctx, enabled);
}

int simpler_flush_diagnostics_ctx(DeviceContextHandle ctx, int32_t timeout_ms, char *error, size_t error_capacity) {
    if (ctx == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    try {
        std::string reason;
        const int budget = timeout_ms > 0 ? timeout_ms : 30000;
        const int rc = static_cast<DeviceRunnerBase *>(ctx)->flush_diagnostics(budget, &reason);
        if (rc != 0 && error != NULL && error_capacity > 0) {
            std::snprintf(error, error_capacity, "%s", reason.c_str());
        }
        return rc;
    } catch (const std::exception &e) {
        if (error != NULL && error_capacity > 0) std::snprintf(error, error_capacity, "%s", e.what());
        return PTO_RUNTIME_ERR_INTERNAL;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int device_memory_info_ctx(DeviceContextHandle ctx, DeviceMemoryInfo *info) {
    if (ctx == NULL || info == NULL) return PTO_RUNTIME_ERR_INTERNAL;
    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    try {
        int rc = runner->attach_current_thread(runner->device_id());
        if (rc != 0) return rc;

        size_t free_bytes = 0;
        size_t total_bytes = 0;
        aclError acl_rc = aclrtGetMemInfo(ACL_HBM_MEM, &free_bytes, &total_bytes);
        if (acl_rc != ACL_SUCCESS) {
            LOG_ERROR("aclrtGetMemInfo(ACL_HBM_MEM) failed: %d", static_cast<int>(acl_rc));
            ACL_LOG_ERROR_DETAIL(acl_rc);
            return static_cast<int>(acl_rc);
        }
        info->free_bytes = static_cast<uint64_t>(free_bytes);
        info->total_bytes = static_cast<uint64_t>(total_bytes);
        return 0;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

/* ===========================================================================
 * Kernel-mode lifecycle
 *
 * Init and prepare create context-owned resources on a borrowed device.
 * Launch remains a rejecting stub, so supported() reports 0. Structural
 * argument validation is shared with the simulated components through
 * kernel_entry_validation.h.
 * =========================================================================== */

int simpler_kernel_mode_supported(DeviceContextHandle) { return 0; }

int simpler_kernel_mode_init(
    DeviceContextHandle ctx, int device_id, const uint8_t *aicpu_binary, size_t aicpu_size,
    const uint8_t *aicore_binary, size_t aicore_size, const uint8_t *dispatcher_binary, size_t dispatcher_size,
    const CallConfig *config, uint64_t context_generation
) {
    int rc = validate_kernel_init_args(
        ctx, device_id, aicpu_binary, aicpu_size, aicore_binary, aicore_size, dispatcher_binary, dispatcher_size,
        config, context_generation
    );
    if (rc != 0) return rc;

    // Validate the runtime's resource topology before the irreversible latch.
    // TMR resolves scratch sizes from config; HBG's per-run graph sizes are
    // determined later by host orchestration, on the common bind path.
    try {
        PipelineContract contract{};
        const int contract_rc = build_kernel_pipeline_contract_impl(config, &contract);
        if (contract_rc == PTO_RUNTIME_ERR_UNSUPPORTED) {
            LOG_ERROR("simpler_kernel_mode_init: kernel mode is not supported by this host runtime");
            return contract_rc;
        }
        if (contract_rc != 0) return contract_rc;
        if (!is_valid_pipeline_contract(&contract, SIMPLER_MODE_KERNEL) || !has_serviceable_arena_topology(contract) ||
            !has_serviceable_stream_topology(contract)) {
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    // A same-mode latch is idempotent, but initialization is not. Reject
    // reuse before replacing executors. The adopted device identity also
    // catches init failures whose resource rollback
    // returned the kernel execution state to New.
    if (runner->device_id() >= 0 || runner->kernel_execution_state().phase() != KernelContextPhase::New) {
        LOG_ERROR("simpler_kernel_mode_init: this context has already been initialized or requires close");
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    rc = runner->execution_mode_latch().latch(SIMPLER_MODE_KERNEL);
    if (rc != 0) {
        LOG_ERROR("simpler_kernel_mode_init: incompatible execution-mode latch");
        return rc;
    }

    // The caller's CANN log level is the caller's: unlike simpler_init, this
    // path opens no device context of its own, so it has nothing to level and
    // no standing to change a process-wide setting it does not own.
    try {
        std::vector<uint8_t> aicpu_vec(aicpu_binary, aicpu_binary + aicpu_size);
        std::vector<uint8_t> aicore_vec(aicore_binary, aicore_binary + aicore_size);
        runner->set_executors(std::move(aicpu_vec), std::move(aicore_vec));
        if (dispatcher_binary != NULL && dispatcher_size > 0) {
            std::vector<uint8_t> dispatcher_vec(dispatcher_binary, dispatcher_binary + dispatcher_size);
            runner->set_dispatcher_binary(std::move(dispatcher_vec));
        }
        rc = runner->init_kernel_context(device_id);
    } catch (...) {
        rc = PTO_RUNTIME_ERR_INTERNAL;
    }
    if (rc != 0) {
        runner->kernel_execution_state().poison(rc);
        return rc;
    }
    return 0;
}

int simpler_kernel_mode_prepare_callable(
    DeviceContextHandle ctx, int32_t callable_id, const void *callable, size_t callable_size
) {
    int rc = validate_kernel_prepare_callable_args(ctx, callable_id, callable, callable_size);
    if (rc != 0) return rc;
    rc = validate_kernel_callable_image(callable, callable_size);
    if (rc != 0) return rc;

    DeviceRunnerBase *runner = static_cast<DeviceRunnerBase *>(ctx);
    if (!runner->execution_mode_latch().is_kernel()) {
        LOG_ERROR("simpler_kernel_mode_prepare_callable: no live kernel context on this device context");
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    if (!runner->kernel_execution_state().accepts_dispatch()) {
        LOG_ERROR("simpler_kernel_mode_prepare_callable: the kernel context no longer accepts preparation");
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }

    try {
        // Registration uploads on the caller's device without rebinding it.
        rc = runner->enter_run_thread();
        if (rc != 0) return rc;

        // prepare_kernel_callable's AICPU registration self-skips a callable
        // whose orchestration was resolved host-side, so the flag it reports
        // adds nothing here.
        bool needs_aicpu_register = false;
        rc = record_callable_on_runner(runner, callable_id, callable, &needs_aicpu_register);
        if (rc != 0) return rc;

        rc = runner->prepare_kernel_callable(callable_id);
        if (rc != 0) {
            // Releasing the registration frees the device buffer holding the
            // orchestration SO. That is only safe when preparation failed
            // before anything on the device could read it: a registration whose
            // wait failed or timed out leaves an AICPU task whose completion is
            // not established, and poisons the context to say so. Keep the
            // callable owned in that case — it is released at close, once the
            // caller has established quiescence.
            if (runner->kernel_execution_state().accepts_dispatch()) {
                runner->unregister_callable(callable_id);
            } else {
                LOG_ERROR(
                    "simpler_kernel_mode_prepare_callable: retaining callable_id=%d — the context was poisoned while "
                    "device work on its image may still be outstanding",
                    callable_id
                );
            }
            return rc;
        }
        return 0;
    } catch (...) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
}

int simpler_kernel_mode_launch(DeviceContextHandle ctx, int32_t callable_id, const void *args, void *caller_stream) {
    const int rc = validate_kernel_launch_args(ctx, callable_id, args, caller_stream);
    if (rc != 0) return rc;
    LOG_ERROR("simpler_kernel_mode_launch: no live kernel context on this device context");
    return PTO_RUNTIME_ERR_INVALID_STATE;
}

}  // extern "C"
