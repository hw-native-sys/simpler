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
 * host_build_graph runtime maker (host orchestration).
 *
 * The HOST runs the orchestrator to completion, populates shared memory + the
 * prebuilt arena, and H2Ds the image; the device boots scheduler-only.
 *
 * register_callable_impl:
 *   - dlopens the orchestration SO on the host and resolves its entry points,
 *     which bind then invokes per run
 *
 * bind_callable_to_runtime_impl:
 *   - Gives host-memory tensor arguments slices of the pipeline slot's retained
 *     temporary buffer, copies every readable input H2D, and records one lease
 *     each with its transfer directions. The copy is here rather than in
 *     copy_in_run_inputs_impl because the orchestration entry below reads these
 *     tensors while it builds the graph
 *   - Runs the resolved orchestration entry to build the graph
 *   - Sets up runtime state for host orchestration
 *
 * copy_back_run_outputs_impl / release_run_bindings_impl:
 *   - Copies OUTPUT/INOUT tensors back from device to host (read-only inputs
 *     are skipped)
 *   - Releases the run's leases. The slices are no-ops: the retained buffer
 *     outlives the run and is freed once at Worker finalization.
 */

#include <dlfcn.h>
#include <stddef.h>
#include <stdint.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <atomic>
#include <cerrno>
#include <sys/resource.h>

#include <cinttypes>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cctype>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "assert_compat.h"
#include "common/device_run_result.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/host/dep_gen_host_graph.h"
#include "host_build_graph/host/bind_phase.h"
#include "host_build_graph/graph_execution.h"
#include "host_build_graph/host/host_tensor_access.h"
#include "host_build_graph/host/retained_prepare_plan.h"
#include "host_build_graph/host/graph_host_state.h"
#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/host/host_phase_trace.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/host/orch_ops.h"
#include "host_build_graph/host/ready_queue_sizing.h"
#include "host_build_graph/host/graph_recorder_pool.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/types.h"
#include "host_build_graph/runtime.h"
#include "../../../../common/runtime_status/error_log.h"
#include "../../../../common/task_interface/call_config.h"
#include "../../../../common/worker/runtime_c_api.h"
#include "callable.h"
#include "common/host_log_binding.h"
#include "common/host_phase_kind.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host_log.h"
#include "host/raii_scope_guard.h"
#include "host/kernel_pipeline_contract.h"
#include "utils/device_arena.h"
#include "utils/retained_temp_bump.h"
#include "utils/temp_buffer_plan.h"
#include "utils/program_tensor_args.h"
#include "utils/tensor_lease_release.h"
#include "prepare_callable_common.h"

// This file returns both kinds of negative status — a latched device code
// negated, and PTO_RUNTIME_ERR_* for a host-side failure — so a caller can
// attribute one to a mechanism only while the two bands stay disjoint. The
// second conjunct is the structural half and holds for any latched code; the
// first is a spot check on the highest one this runtime defines, so a new
// four-digit latched code needs the ceiling raised here as well.
static_assert(
    SIMPLER_ERROR_READY_QUEUE_OVERFLOW <= PTO_RUNTIME_LATCHED_CODE_MAX &&
        PTO_RUNTIME_ERR_BASE < -PTO_RUNTIME_LATCHED_CODE_MAX,
    "host-side C API codes must stay below the negation of every latched device code"
);

extern "C" const PipelineContract *get_pipeline_contract(void) {
    // Host orchestration materializes this run's own graph into the image it
    // uploads, so every device-resident region carries per-run content.
    //
    // PTO_PIPELINE_GM_SM is absent because hbg has no separate shared-memory
    // region: the image is the tail of the runtime-image region, so it shares that
    // region's classification. The arena-topology check skips an absent kind.
    // `pipeline_depth` is the maximum this runtime supports, not a configured
    // count: every device-resident region it names carries per-run content and
    // is selected by the caller's lease, so as many runs can hold one as a
    // context cares to grant. What a context actually grants is the smaller of
    // this and the request, and every per-run resource is built to that grant —
    // so declaring no limit here commits nothing.
    static const PipelineContract contract = {
        PTO_PIPELINE_CONTRACT_ABI_VERSION,
        4,
        UINT32_MAX,
        {
            {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
        },
    };
    return &contract;
}

extern "C" int build_kernel_pipeline_contract_impl(const CallConfig *config, PipelineContract *out) {
    if (out == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (config == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    // The graph and its heap are sized by each bind's host orchestration.
    // Init declares their bank topology, without claiming a fixed capacity.
    *out = *get_pipeline_contract();
    return 0;
}

extern "C" int concurrent_native_prepare_supported_impl(void) {
    // HBG can materialize a complete graph into the lease-selected unpublished
    // arena bank.
    return 1;
}

extern "C" int teardown_report_supported_impl(void) {
    // The onboard platform runner records a teardown for whichever runtime is
    // built on it; this runtime is the one whose reset paths the record's
    // stage and rc fields describe, so it is the one that publishes them.
    return 1;
}

extern "C" int joined_native_launch_supported_impl(void) {
    // Every per-run device region this runtime names is selected by the run's
    // own lease — the graph heap and the runtime image are both HOST_PER_RUN
    // above — so a second launched run reads and writes nothing the first is
    // still executing against. What orders the two on the device is a queued
    // event wait, not this answer.
    return 1;
}

extern "C" int joined_diagnostic_launch_supported_impl(void) {
    // The swimlane collector can hold two runs open on this runtime: buffers
    // carry the run that produced them, each run's loss is charged to its own
    // bucket, and the per-queue cut proves one run's hand-offs are all drained
    // without waiting for the queue to be empty. The producer also clears its
    // own record counters at init, so no host write lands on a predecessor
    // that is still recording.
    //
    // Declared separately from the ordering capability above because the two
    // ask different things: that one is about which device regions a run owns,
    // this one about whether a collector can hold a second run open. Making
    // either imply the other would tie ordinary non-diagnostic early enqueue
    // to swimlane bucket capacity.
    return 1;
}

extern "C" int args_dump_capture_supported_impl(void) {
    // An AICPU scheduler thread hands each task to its core and reaps its
    // completion on this runtime — there is no on-core scheduling route here —
    // so `dump_args_for_task` runs at both of a task's own boundaries
    // (scheduler_dispatch.cpp, scheduler_completion.cpp, and the cold path's
    // inline dispatch).
    return 1;
}

// RuntimeEnv (call_config.h) is the cross-runtime ABI for per-ring config and
// carries RUNTIME_ENV_RING_COUNT slots, shared with tensormap_and_ringbuffer.
// host_build_graph keeps one task table and reads slot 0, so it only needs the ABI
// to carry at least one.
static_assert(RUNTIME_ENV_RING_COUNT >= 1, "RuntimeEnv must carry the slot host_build_graph reads");

// The ring sizes were once settable process-wide through PTO2_RING_TASK_WINDOW /
// PTO2_RING_HEAP. They are per task now, through CallConfig.runtime_env, and
// nothing reads those names. Exporting one is therefore a silent
// misconfiguration -- the run takes the compile-time default and the requested
// sizing appears nowhere -- so it is reported once per bind.
static void warn_on_retired_ring_env() {
    static constexpr const char *kRetired[] = {"PTO2_RING_TASK_WINDOW", "PTO2_RING_HEAP"};
    for (const char *name : kRetired) {
        if (std::getenv(name) != nullptr) {
            LOG_WARN("%s is no longer read; size the rings per task via CallConfig.runtime_env", name);
        }
    }
}

// A RuntimeEnv knob array points into the #pragma pack(1) wire struct
// (call_config.h), so its uint64_t entries are only byte-aligned — runtime_env
// sits at offset 28 in CallConfig (after 7 int32_t), i.e. 4-byte but not 8-byte
// aligned. Reading them as `base[idx]` is an unaligned 8-byte load: UB, and fatal
// under UBSan (-fsanitize=alignment). Copy the bytes out instead. A null base
// means "no per-task overrides" -> 0 (unset).
static uint64_t read_ring_override(const uint64_t *base, int idx) {
    if (base == nullptr) {
        return 0;
    }
    uint64_t value;
    std::memcpy(&value, base + idx, sizeof(value));
    return value;
}

// ring_task_window points at the first slot of a per-ring array in the RuntimeEnv
// wire struct (0 = unset); hbg keeps one task table and reads slot 0. A per-task
// entry wins over the compile-time default, and there is nothing between them.
//
// The heap takes no configuration: its device region is committed after
// orchestration, sized to what the graph turned out to need, so there is nothing
// to resolve up front. RuntimeEnv::ring_heap stays the reclaiming runtime's knob;
// this function does not resolve it, and the caller reads it only to warn that it
// reaches nothing here.
static bool resolve_graph_task_capacity(const uint64_t *ring_task_window, uint64_t *task_capacity) {
    *task_capacity = CHIP_DEFAULT_GRAPH_TASKS;

    warn_on_retired_ring_env();

    const uint64_t override_value = read_ring_override(ring_task_window, 0);
    if (override_value != 0) {
        *task_capacity = override_value;
    }

    // Any positive count is usable: a task id indexes its slot directly, so no slot
    // lookup masks with this value. The power-of-two, >= 4 requirement belongs to
    // tensormap_and_ringbuffer, which does mask there, and is enforced in that
    // runtime's own resolve; neither the RuntimeEnv setter nor Worker.run constrains
    // the value, so this bound is the only one a ring_task_window passes through.
    //
    // The upper bound is the one place a task id IS masked: a sub-task's id carries
    // its modular task's local id in a fixed-width parent field, and that mint masks
    // rather than fails, so a count past the field would truncate a parent silently.
    // The shared-memory limit below is the tighter of the two in practice -- a slot
    // costs kilobytes, so INT32_MAX bytes runs out first -- which makes this a guard
    // that should never be the one to fire rather than a cap anyone meets.
    if (*task_capacity < 1 || *task_capacity > static_cast<uint64_t>(TaskId::GLOBAL_TASK_MAX_NUM)) {
        LOG_ERROR(
            "ring_task_window=%" PRIu64 " must be in [1, %d]: a modular task's local id has to fit the parent field "
            "of the sub-task ids it mints",
            *task_capacity, TaskId::GLOBAL_TASK_MAX_NUM
        );
        return false;
    }
    // A slot state reaches its payload and descriptor through a 32-bit
    // self-relative delta, so every pair of addresses in the shared-memory
    // image must be within INT32_MAX of each other.
    const uint64_t sm_bytes = sm_layout::segment_offsets(*task_capacity).end;
    if (sm_bytes > static_cast<uint64_t>(INT32_MAX)) {
        LOG_ERROR(
            "ring_task_window=%" PRIu64 " needs a %" PRIu64 "-byte shared memory image, past the %d-byte limit "
            "a slot state's self-relative payload/descriptor delta can span",
            *task_capacity, sm_bytes, INT32_MAX
        );
        return false;
    }

    return true;
}

static int32_t read_runtime_status(const Runtime *runtime, const HostApi *api, SharedMemoryHeader *host_header) {
    if (runtime == nullptr || api == nullptr || host_header == nullptr) {
        return 0;
    }

    void *device_sm = runtime->get_gm_sm_ptr();
    if (device_sm == nullptr) {
        return 0;
    }

    int hdr_rc = api->copy_from_device(host_header, device_sm, sizeof(SharedMemoryHeader));
    if (hdr_rc != 0) {
        LOG_WARN("Failed to copy the shared-memory header from device");
        return 0;
    }

    int32_t sched_error_code = host_header->sched_error_code.load(std::memory_order_relaxed);
    return runtime_status_from_error_code(sched_error_code);
}

static void release_run_tensor_leases(Runtime *runtime, const HostApi *api) {
    const TensorLeaseReleaseCounts counts = release_tensor_leases(runtime->tensor_leases(), api);
    LOG_DEBUG(
        "Released tensor leases: freed=%d buffer_noop=%d external_noop=%d", counts.freed, counts.buffer_noop,
        counts.external_noop
    );
}

namespace {

// host_build_graph is host-orchestration-first: the HOST dlopens the
// orchestration .so and runs it to completion. Every cross-task reference the
// shared memory and arena carry is an offset or an index from its own block, so
// the image the device schedules is the bytes the host wrote — there is nothing
// to fix up after the H2D copy.

bool write_all_bytes(int fd, const uint8_t *data, size_t size) {
    size_t total = 0;
    while (total < size) {
        ssize_t w = write(fd, data + total, size - total);
        if (w <= 0) {
            return false;
        }
        total += static_cast<size_t>(w);
    }
    return true;
}

// Materialize the orchestration .so bytes to a temp file so it can be dlopen'd
// on the host (dlopen needs a real path + the exec bit).
bool create_orch_so_tempfile(const uint8_t *data, size_t size, std::string *out_path) {
    char tmpl[] = "/tmp/orch_so_XXXXXX";
    int fd = mkstemp(tmpl);
    if (fd < 0) {
        return false;
    }
    if (fchmod(fd, 0755) != 0) {
        close(fd);
        unlink(tmpl);
        return false;
    }
    bool ok = write_all_bytes(fd, data, size);
    if (close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        unlink(tmpl);
        return false;
    }
    *out_path = tmpl;
    return true;
}

;

;

;

using HostPreparePlan = RetainedHostPreparePlan<RuntimeArenaLayout>;

int32_t commit_host_plan(Runtime *runtime, const HostApi *api, HostPreparePlan &plan) {
    const auto &layout = plan.layout;
    auto &host_arena = plan.host_arena;
    void *host_sm = plan.host_sm;
    const uint64_t task_capacity = plan.task_capacity;
    const uint64_t heap_bytes = plan.heap_bytes;
    const uint64_t image_bytes = plan.image_bytes;
    const int32_t total_tasks = plan.total_tasks;
    const auto &bind_usage = plan.bind_usage;
    const uint64_t device_arena_bytes = layout.off_copied_end + image_bytes;
    const BindPhaseMark static_arena_phase = bind_phase_begin();
    runtime->dev.sm_image_bytes = image_bytes;
    if (const int rc = api->setup_static_arena(heap_bytes, /*gm_sm_size=*/0, device_arena_bytes); rc != 0) {
        return rc;
    }
    {
        char attrs[kBindAttrsCapacity];
        snprintf(attrs, sizeof(attrs), "heap=%" PRIu64 " arena=%" PRIu64, heap_bytes, device_arena_bytes);
        record_bind_phase(HostPhaseKind::BindStaticArena, static_arena_phase, attrs);
    }

    const BindPhaseMark sm_phase = bind_phase_begin();
    void *device_arena = api->acquire_pooled_runtime_arena();
    if (device_arena == nullptr) {
        LOG_ERROR("%s", "host-orch: failed to acquire the pooled runtime arena");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    char *arena_dev = static_cast<char *>(device_arena);
    void *device_sm = arena_dev + layout.off_copied_end;
    runtime->set_gm_sm_ptr(device_sm);
    {
        char attrs[kBindAttrsCapacity];
        snprintf(attrs, sizeof(attrs), "bytes=%" PRIu64, image_bytes);
        record_bind_phase(HostPhaseKind::BindSharedMem, sm_phase, attrs, image_bytes);
    }

    const BindPhaseMark heap_phase = bind_phase_begin();
    void *gm_heap = api->acquire_pooled_gm_heap();
    record_bind_phase(HostPhaseKind::BindGmHeap, heap_phase);
    if (gm_heap == nullptr) {
        LOG_ERROR("host-orch: failed to acquire the pooled GM heap");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // The virtual heap window sits above every device address, which is what keeps
    // a real address from being mistaken for one the rebase has to move. Device
    // memory is the only source of addresses this pass cannot inspect one by one,
    // so its base is checked here.
    always_assert(
        reinterpret_cast<uint64_t>(gm_heap) < HEAP_VIRTUAL_BASE && "device memory reaches into the virtual heap window"
    );
    // The alignment bind_graph_definitions checked on the virtual base — a Graph
    // task's runtime storage must land on alignof(ChipTaskStorage) — carries to
    // the real base only while the two are congruent: both are aligned to
    // kDefaultBaseAlign, and that covers the storage's own requirement.
    static_assert(
        HEAP_VIRTUAL_BASE % DeviceArena::kDefaultBaseAlign == 0,
        "the virtual heap base must share the committed region's alignment"
    );
    static_assert(
        alignof(ChipTaskStorage) <= DeviceArena::kDefaultBaseAlign,
        "a sub-task's storage alignment must be covered by the heap region's base alignment"
    );
    always_assert(reinterpret_cast<uint64_t>(gm_heap) % DeviceArena::kDefaultBaseAlign == 0);
    const sm_layout::HeapRebase heap_rebase{reinterpret_cast<uint64_t>(gm_heap), heap_bytes};

    // One host source for one copy: the copied zone and shared-memory image at
    // exactly the offsets they occupy on the device.
    // Over-allocated and rounded up because every segment offset is
    // CHIP_ALIGN_SIZE-aligned and ChipTaskSlotState is alignas(64), which a byte
    // vector's data() is not.
    const uint64_t copied_bytes = layout.off_copied_end - layout.off_copied_begin;
    const uint64_t upload_bytes = copied_bytes + image_bytes;
    // Assembled in staging the runner retains rather than in a buffer that dies
    // with this frame: publication is a separate step, so the bytes have to still
    // be here when it reads them. Over-allocated by the alignment the layout
    // needs, and handed over uninitialized: what this bind writes are the copied
    // zone and the live segments `compact_live_image` emits, so the alignment gaps
    // between segments carry whatever the block last held. No device code reads a
    // gap — every reader indexes a segment through the layout — which is what makes
    // clearing the whole block per bind unnecessary rather than merely expensive.
    void *staging_addr = nullptr;
    if (api->acquire_run_image_staging(static_cast<size_t>(upload_bytes), CHIP_ALIGN_SIZE, &staging_addr) != 0 ||
        staging_addr == nullptr) {
        LOG_ERROR("host-orch: staging for a %" PRIu64 "-byte runtime image is unavailable", upload_bytes);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    char *upload_base = static_cast<char *>(staging_addr);

    // The copied zone carries no host address by construction: the orchestrator and
    // the ops table live on the host-side OrchContext, which never enters this image.
    std::memcpy(upload_base, static_cast<const char *>(host_arena.base()) + layout.off_copied_begin, copied_bytes);
    const uint64_t compacted = sm_layout::compact_live_image(
        static_cast<const char *>(host_sm), task_capacity, bind_usage, heap_rebase, upload_base + copied_bytes
    );
    always_assert(compacted == image_bytes);

    // Prepared, not published: `publish_run_image_impl` performs this write, and
    // records the BindArenaH2d segment that measures it. The pool counters travel
    // with the record because only this bind knows them; the rest of that
    // segment's attributes the publication rebuilds from `dev` and the length.
    runtime->set_pending_publication(
        arena_dev + layout.off_copied_begin, upload_base, upload_bytes, bind_usage.fanin_elems, bind_usage.tensor_elems,
        bind_usage.scalar_elems
    );
    return total_tasks;
}

}  // namespace

namespace {

// Re-enter only the resource commit. The callback and HOST copies have already
// completed, and their outputs stay owned by the slot while it waits.
int resume_host_plan(Runtime *runtime, const HostApi *api, const std::shared_ptr<HostPreparePlan> &plan) {
    if (plan->committed) return PTO_RUNTIME_ERR_INTERNAL;
    api->set_prepare_plan(plan);
    plan->resume(*runtime);
    host_phase_trace_note_submitted(static_cast<uint64_t>(plan->total_tasks));
    if (plan->staging_bytes != 0) {
        void *staging = nullptr;
        size_t capacity = 0;
        api->get_retained_temp_buffer(&staging, &capacity);
        // Device addresses are baked into the callback's graph. Its stamped
        // slot excludes trim, so this is a current-epoch reference, never a grow.
        if (staging != plan->staging_backing || capacity < plan->staging_bytes) return PTO_RUNTIME_ERR_INTERNAL;
        const int rc = api->acquire_retained_temp(plan->staging_bytes, &staging, &capacity);
        if (rc != 0) {
            if (rc == SIMPLER_RUNTIME_ERR_RESOURCE_WAIT) {
                if (api->get_prepare_plan() != plan) return SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE;
                plan->park(*runtime);
            }
            return rc;
        }
        if (staging != plan->staging_backing || capacity < plan->staging_bytes) return PTO_RUNTIME_ERR_INTERNAL;
    }
    const int32_t result = commit_host_plan(runtime, api, *plan);
    if (result == SIMPLER_RUNTIME_ERR_RESOURCE_WAIT) {
        // A backend without retained-plan support must fail rather than invoke
        // an already completed callback again on the caller's retry.
        if (api->get_prepare_plan() != plan) return SIMPLER_RUNTIME_ERR_RESOURCE_UNAVAILABLE;
        plan->park(*runtime);
        return result;
    }
    if (result < 0) return result;
    runtime->dev.host_total_tasks = result;
    void *device_arena = api->acquire_pooled_runtime_arena();
    if (device_arena == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    runtime->set_prebuilt_arena(device_arena, plan->layout.off_runtime);
    plan->committed = true;
    if (api->get_prepare_plan() != plan) {
        dep_gen_host_graph_restore_for_export(&plan->diagnostic_graph, plan->diagnostic_outcome);
    }
    return 0;
}

}  // namespace

/**
 * Stage the per-callable resources (kernel binaries + orchestration SO) into
 * CallableArtifacts for subsequent per-run binding. Nothing here depends on
 * per-run argument values, so registration runs once per callable_id.
 *
 * @param callable  ChipCallable carrying the orch SO + child kernel binaries
 * @param api       Context-bound platform operations used during registration
 * @param out       Callable-owned artifacts retained across runs
 * @return 0 on success, -1 on failure
 */
extern "C" int register_callable_impl(const ChipCallable *callable, const HostApi *api, CallableArtifacts *out) {
    if (callable == nullptr) {
        LOG_ERROR("Callable pointer is null");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (api == nullptr || out == nullptr) {
        LOG_ERROR("HostApi or out is null");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    *out = CallableArtifacts{};
    out->signature.assign(callable->signature_, callable->signature_ + callable->sig_count());

    LOG_INFO("Registering %d kernel(s) in register_callable_impl", callable->child_count());
    if (upload_and_collect_child_addrs(
            callable, api, &out->kernel_addrs, &out->chip_buffer_dev, &out->chip_buffer_hash, &out->aicore_image_hash
        ) != 0) {
        LOG_ERROR("Failed to upload ChipCallable buffer");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    for (const ChildKernelAddr &c : out->kernel_addrs) {
        if (c.func_id < 0 || c.func_id >= CHIP_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d is out of range [0, %d)", c.func_id, CHIP_MAX_FUNC_ID);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
    }

    const uint8_t *orch_so_binary = static_cast<const uint8_t *>(callable->binary_data());
    size_t orch_so_size = callable->binary_size();

    if (orch_so_binary == nullptr || orch_so_size == 0) {
        LOG_ERROR("Orchestration SO binary is required for host orchestration");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    out->orch_so_data = orch_so_binary;
    out->orch_so_size = orch_so_size;
    out->func_name = callable->func_name();
    out->config_name = callable->config_name();

    // host_build_graph host-orch: dlopen the orchestration .so ON THE HOST and
    // resolve its entry symbol now. The handle is held across the prepared
    // callable's lifetime (closed by DeviceRunner::unregister_callable via
    // host_dlopen_handle); bind_callable_to_runtime_impl invokes the resolved
    // entry per run. This is what makes the host-side dlopen observable
    // (host_dlopen_count) while the AICPU never dlopens the orch .so.
    {
        const char *orch_func_name = callable->func_name();
        if (orch_func_name == nullptr || orch_func_name[0] == '\0') {
            LOG_ERROR("host-orch: orchestration function name is empty");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        std::string so_path;
        if (!create_orch_so_tempfile(orch_so_binary, orch_so_size, &so_path)) {
            LOG_ERROR("host-orch: failed to materialize orchestration .so");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        void *handle = dlopen(so_path.c_str(), RTLD_NOW | RTLD_LOCAL);
        if (handle == nullptr) {
            LOG_ERROR("host-orch: dlopen failed: %s", dlerror());
            unlink(so_path.c_str());
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        // Unlinked as soon as it is mapped: the handle keeps the image alive regardless of
        // path, and every failure return below then leaves nothing behind in /tmp.
        unlink(so_path.c_str());
        const char *bind_log_error = nullptr;
        if (simpler::log::bind_loaded_host_log_state(handle, HostLogger::get_instance().state(), &bind_log_error) !=
            0) {
            LOG_ERROR(
                "host-orch: failed to bind host-log state: %s",
                bind_log_error != nullptr ? bind_log_error : "unknown error"
            );
            dlclose(handle);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        void *entry = dlsym(handle, orch_func_name);
        if (entry == nullptr) {
            LOG_ERROR("host-orch: dlsym('%s') failed: %s", orch_func_name, dlerror());
            dlclose(handle);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        // The orch .so has its own framework_bind_orch_context and the context
        // pointer it sets (orchestration/common.cpp is compiled into it); resolve it
        // now so the per-run bind can set it before the .so's inline rt_submit_* run.
        void *bind_sym = dlsym(handle, "framework_bind_orch_context");
        if (bind_sym == nullptr) {
            LOG_ERROR("host-orch: orch .so does not export framework_bind_orch_context: %s", dlerror());
            dlclose(handle);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        // The recorder pool is the runtime's, one per process, so this is a plain call
        // rather than a symbol in the .so just loaded: it parks the prewarmed workers and
        // stands each one's recording storage up, and a later registration finds the pool
        // already warm. Failing here fails registration, where the caller can act on it,
        // instead of abandoning a recording mid-bind.
        if (!graph_recorder_prewarm()) {
            LOG_ERROR("host-orch: could not park the graph recorder pool");
            dlclose(handle);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        auto *eps = new HostOrchEntryPoints{};
        eps->entry = reinterpret_cast<OrchestrationEntryFunc>(entry);
        eps->bind = reinterpret_cast<OrchestrationBindFunc>(bind_sym);
        // Optional, and looked up without failing the registration: a .so built before this
        // entry point existed holds nothing per bind. dlerror() is drained here so the next
        // caller does not read this miss as its own failure.
        eps->finish = reinterpret_cast<OrchestrationFinishFunc>(dlsym(handle, "framework_finish_orch"));
        (void)dlerror();
        out->host_dlopen_handle = handle;
        out->host_orch_func_ptr = eps;
        LOG_INFO("host-orch: loaded orchestration entry '%s' on host", orch_func_name);
    }
    LOG_INFO("Orchestration SO: %zu bytes uploaded", orch_so_size);
    return 0;
}

/**
 * This runtime resolves a kernel entry from the CoreCallable object the
 * dispatch path is handed, so its registration blocks carry the object table
 * alone and no resolved-entry view is built for them.
 */
extern "C" bool runtime_uses_callable_entry_table_impl() { return false; }

/**
 * Per-run binding: build device-side argument storage (tensor copy-out, GM
 * heap, shared memory) and publish it to the runtime. Assumes the
 * callable-side state (kernel binaries, orch SO bytes, func/config names)
 * is already populated by register_callable_impl.
 *
 * Splitting this from register_callable_impl matches the per-callable_id
 * design: register/simpler_run invokes this every call, while the prep
 * half runs only once per callable_id.
 *
 * @param runtime    Pointer to the per-run Runtime
 * @param api        Context-bound platform operations for this run
 * @param orch_args  Separated tensor/scalar arguments for this run
 * @return 0 on success, -1 on failure
 */
extern "C" int bind_callable_to_runtime_impl(
    Runtime *runtime, const HostApi *api, const ChipStorageTaskArgs *orch_args, void *host_orch_func_ptr,
    const ArgDirection *signature, int sig_count, const uint64_t *ring_task_window, const uint64_t *ring_heap,
    [[maybe_unused]] const uint64_t *ring_dep_pool  // polling has no dep_pool; kept for ABI stability
) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (orch_args == nullptr) {
        LOG_ERROR("orch_args pointer is null");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const int transfer_status = validate_program_tensor_transfers(orch_args, true, signature, sig_count);
    if (transfer_status != 0) return transfer_status;
    // host_build_graph host-orch: register_callable_impl resolved the
    // orchestration entry on the host and passed it here as host_orch_func_ptr;
    // it is run below (after the arena is built) against a host SM mirror.
    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();
    LOG_INFO("RT2 bind: %d tensors + %d scalars, host orchestration mode", tensor_count, scalar_count);

    if (auto saved = api->get_prepare_plan()) {
        const auto plan = std::static_pointer_cast<HostPreparePlan>(saved);
        if (plan->committed) return PTO_RUNTIME_ERR_INTERNAL;
        host_phase_trace_begin(api);
        const int rc = resume_host_plan(runtime, api, plan);
        if (rc != 0) host_phase_trace_end(api);
        return rc;
    }

    // Rebinding an unpublished run must not replace its source or device owners.
    if (runtime->pending_publication().bytes != 0 || !runtime->pending_publication().prerequisites.empty()) {
        LOG_ERROR("bind_callable_to_runtime_impl: unpublished metadata still belongs to this run");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    host_phase_trace_begin(api);
    bool prepared = false;
    auto host_phase_guard = RAIIScopeGuard([runtime, api, &prepared]() {
        if (!prepared) {
            runtime->clear_pending_publication();
            host_phase_trace_end(api);
        }
    });

    uint64_t task_capacity = 0;
    if (!resolve_graph_task_capacity(ring_task_window, &task_capacity)) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // The heap takes no configuration, so a set ring_heap reaches nothing in this
    // runtime — most often it is a config written for the reclaiming one, whose knob
    // it still is.
    if (read_ring_override(ring_heap, 0) != 0) {
        LOG_WARN(
            "%s", "host_build_graph ignores ring_heap: its graph heap is committed after orchestration at the "
                  "size the graph turned out to need"
        );
    }

    // Build device args: copy from input, replace host tensor pointers with device pointers
    ChipStorageTaskArgs device_args;

    // Explicit HOST/NONE views are validated before any tensor copy and remain
    // bound until all host recording threads have finished.
    HostTensorAccessor tensor_access;
    for (int i = 0; i < tensor_count; ++i) {
        const ChipTensor &t = orch_args->tensor(i);
        if (t.address_space == AddressSpace::HOST && t.transfer == TensorTransfer::NONE &&
            !tensor_access.add(t, signature[i])) {
            LOG_ERROR("host-orch: invalid or overlapping HOST argument %d", i);
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        }
    }

    // Caller device writes remain recorded for the lane's dependency and
    // lifetime checks; they do not grant a host view of the allocation.
    std::vector<CallerBufferSpan> written_spans;

    // A lease recorded by an earlier bind names an offset this bind is about to
    // re-slice, so carrying one over would copy this run's bytes back to that
    // run's host pointer. Every exit path from here on leaves the ledger owned
    // by the caller's Runtime, and this is where it starts empty.
    runtime->tensor_leases().clear();

    // The retained temporary buffer is always used on the hbg path — it is an
    // internal allocation optimization, not user-facing config. The buffer
    // lives on the runner across runs, one per pipeline slot; here we grow it
    // to this run's packed size and bump-slice from it. A run holds its
    // pipeline slot from bind through validate and a concurrent reservation is
    // admitted only on a distinct slot, so no other run can re-slice this
    // buffer while these slices are live.
    //
    // Inside the args span: growing the buffer is the only device allocation
    // this phase can make, so a steady-state workload reporting no allocation
    // cost here is a measurement rather than a definition.
    const BindPhaseMark args_phase = bind_phase_begin();
    RetainedTempBump bump;
    const size_t required_temp_bytes = packed_temp_bytes(orch_args);
    if (const int rc = bump.begin_status(api, required_temp_bytes); rc != 0) {
        return rc;
    }
    uint64_t h2d_bytes = 0;
    int h2d_tensors = 0;
    for (int i = 0; i < tensor_count; i++) {
        ChipTensor t = orch_args->tensor(i);

        // Caller tensors are the one class of address orchestration sees that this
        // pass did not mint, so this is where they are checked against the virtual
        // heap window they must stay below — an address inside it would be rebased
        // as if it were a graph-heap allocation.
        if (t.is_device_memory()) {
            always_assert(t.buffer.addr < HEAP_VIRTUAL_BASE && "caller tensor reaches into the virtual heap window");
            LOG_DEBUG("  ChipTensor %d: child memory, pass-through (0x%" PRIx64 ")", i, t.buffer.addr);
            // The signature supplies the caller-write set. An uncovered device
            // argument cannot be classified as read-only or writable.
            if (signature == nullptr || i >= sig_count) {
                LOG_ERROR(
                    "host-orch: device tensor %d has no entry in this callable's %d-entry signature. A device "
                    "argument is passed through by address, so its direction is the only thing its handling "
                    "depends on -- give the callable a signature that covers every tensor it is called with",
                    i, sig_count
                );
                return runtime_status_from_error_code(SIMPLER_ERROR_INVALID_ARGS);
            }
            const bool writes = signature[i] == ArgDirection::OUT || signature[i] == ArgDirection::INOUT;
            if (writes) written_spans.push_back(CallerBufferSpan{t.buffer.addr, t.buffer.size});
            device_args.add_tensor(t);
            continue;
        }

        if (t.transfer == TensorTransfer::NONE) {
            device_args.add_tensor(t);
            continue;
        }

        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
        size_t size = static_cast<size_t>(t.nbytes());
        // An empty tensor addresses nothing, so it takes no slice and carries a
        // null address rather than one aliasing the next tensor's.
        if (size == 0) {
            t.address_space = AddressSpace::DEVICE;
            t.transfer = TensorTransfer::NONE;
            t.buffer.addr = 0;
            device_args.add_tensor(t);
            continue;
        }

        void *dev_ptr = bump.acquire(size);
        if (dev_ptr == nullptr) {
            LOG_ERROR(
                "Retained temp buffer slice miss for tensor %d: bytes=%zu offset=%zu capacity=%zu", i, size,
                bump.next_offset(), bump.capacity()
            );
            return PTO_RUNTIME_ERR_INTERNAL;
        }

        // Pure write-only OUTPUT buffers are never read by the kernel and hold
        // no meaningful host content, so they need no copy-in — the
        // kernel defines what it writes and any unwritten bytes are undefined.
        // IN / INOUT (read-before-write) are copied in H2D. The copy happens here
        // rather than in copy_in_run_inputs_impl because the orchestrator below
        // runs on the host and reads these tensors while it builds the graph;
        // that function documents what makes the earlier copy sound.
        bool is_pure_output = (signature != nullptr && i < sig_count && signature[i] == ArgDirection::OUT);
        bool needs_copy_in = !is_pure_output;
        if (needs_copy_in) {
            int rc = api->copy_to_device(dev_ptr, host_ptr, size);
            if (rc != 0) {
                LOG_ERROR("Failed to copy tensor %d in to the device", i);
                return PTO_RUNTIME_ERR_INTERNAL;
            }
            h2d_bytes += static_cast<uint64_t>(size);
            ++h2d_tensors;
        }
        // Read-only INPUT tensors are never written by the kernel, so there is
        // no point copying them back D2H at the end. Index the signature
        // by the orch tensor index `i` (device-space tensors are skipped above
        // but do not consume a separate signature slot — scalars follow the
        // tensor entries). Anything not provably IN keeps the safe default of
        // copying back.
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        runtime->tensor_leases().push_back(
            {host_ptr, dev_ptr, size, needs_copy_in, needs_copy_back, TensorReleaseKind::BufferNoop}
        );
        LOG_DEBUG("  ChipTensor %d: %zu bytes at %p", i, size, dev_ptr);

        t.address_space = AddressSpace::DEVICE;
        t.transfer = TensorTransfer::NONE;
        t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
        always_assert(t.buffer.addr < HEAP_VIRTUAL_BASE && "an argument slice reaches into the virtual heap window");
        device_args.add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        device_args.add_scalar(orch_args->scalar(i));
    }
    {
        char attrs[kBindAttrsCapacity];
        snprintf(attrs, sizeof(attrs), "ntensor=%d h2d=%d bytes=%" PRIu64, tensor_count, h2d_tensors, h2d_bytes);
        record_bind_phase(HostPhaseKind::BindArgs, args_phase, attrs);
    }

    // Lay out the per-Worker static device arena. The GM heap and the prebuilt
    // runtime arena are two independent pooled device allocations, and neither is
    // committed here: the arena's size is known only once orchestration has
    // submitted its tasks, and the heap's only once orchestration has allocated
    // its intermediate buffers. Both are committed by the single
    // setup_static_arena in run_host_orchestration. Owned by DeviceRunner across
    // runs — do NOT record in tensor_leases(); the free is deferred to
    // DeviceRunner::finalize(). The runtime-arena size is determined by replaying
    // the reserve sequence on a host-side arena.
    uint64_t sm_size = SharedMemoryHandle::calculate_size(task_capacity);

    const BindPhaseMark arena_build_phase = bind_phase_begin();
    auto plan = std::make_shared<HostPreparePlan>();
    plan->staging_bytes = required_temp_bytes == 0 ? 0 : required_temp_bytes + RetainedTempBump::kAlignment - 1;
    size_t staging_capacity = 0;
    api->get_retained_temp_buffer(&plan->staging_backing, &staging_capacity);
    DeviceArena &host_arena = plan->host_arena;
    RuntimeArenaLayout &layout = plan->layout;
    layout = runtime_reserve_layout(host_arena, task_capacity);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    {
        char attrs[kBindAttrsCapacity];
        snprintf(attrs, sizeof(attrs), "bytes=%" PRIu64, static_cast<uint64_t>(layout.arena_size));
        record_bind_phase(HostPhaseKind::BindArenaBuild, arena_build_phase, attrs);
    }

    // Set up orchestration state (consumed by the host orchestrator below)
    runtime->set_orch_args(device_args);

    // -------------------------------------------------------------------------
    // Build the prebuilt runtime-arena image on host.
    //
    // We pre-compute every byte the AICPU's runtime arena would otherwise have
    // to write at boot: layout offsets, sub-structure init data, and pointers
    // back to the SM / GM heap. Then we rtMemcpy the image into the pooled
    // runtime-arena region that DeviceRunner keeps alive across runs. AICPU
    // boot becomes attach + wire (cheap pointer fixup) + sm_handle->init (SM
    // reset) + a handful of device-only field fixups.
    // -------------------------------------------------------------------------
    const BindPhaseMark runtime_init_phase = bind_phase_begin();
    // No SM base: the scheduler and sm_handle are device-written now, so nothing
    // here stores one, and the region is not even committed yet.
    RuntimeContext *rt =
        runtime_init_data_from_layout(host_arena, layout, MODE_EXECUTE, /*sm_dev_base=*/nullptr, sm_size);
    if (rt == nullptr) {
        LOG_ERROR("runtime_init_data_from_layout failed");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    runtime_wire_arena_pointers(host_arena, layout, rt);
    // Stash the layout inside the RuntimeContext image so the AICPU can recover every
    // arena-internal offset after the copy. It is written before orchestration
    // because orchestration assembles the image this header is part of, and the
    // publication uploads what it assembled. The runtime arena's device base does
    // NOT travel — it is on the host Runtime (set_prebuilt_arena below), since the
    // AICPU needs that pointer before it can dereference the image.
    rt->prebuilt_layout = layout;
    record_bind_phase(HostPhaseKind::BindRuntimeInit, runtime_init_phase);

    // host_build_graph host-orch: run the orchestrator on the host now, against
    // a host SM mirror, and ship the populated SM to the device. The arena
    // (copied to the device below) carries the scheduler state; the orchestrator
    // itself stays on the host, and the device boots scheduler-only.
    // register_callable_impl guarantees host_orch_func_ptr is non-null on success
    // (it fails the whole prepare otherwise), so this is an assertion-style
    // guard, not a fallback path.
    if (host_orch_func_ptr == nullptr) {
        LOG_ERROR("host-orch: orchestration entry points were not resolved");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // Before orchestration, which is the window in which a concurrently preparing run's host
    // graph build can ask whether these bytes have a producer. Declared even when the list is
    // empty, so a re-prepared run's previous declaration is replaced rather than left standing.
    // A statement that could not be recorded fails the bind here: it would otherwise read to
    // every other run as a buffer nobody produces, which is the one answer that lets a build
    // consume bytes this run has not written yet.
    if (!api->declare_caller_device_writes(written_spans.data(), static_cast<uint32_t>(written_spans.size()))) {
        LOG_ERROR(
            "%s", "host-orch: could not declare the caller device buffers this run produces, so another run's graph "
                  "build could read them as finished; refusing the prepare instead"
        );
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    {
        ChipTaskArgs orch_l2;
        orch_l2.create_from_entry_storage(runtime->get_orch_args());
        int32_t total_tasks = hbg::run_host_orchestration(
            runtime, api, tensor_access, rt, host_arena, layout, sm_size, task_capacity, host_orch_func_ptr, orch_l2,
            *plan
        );
        const BindPhaseMark view_close_phase = bind_phase_begin();
        tensor_access.close();
        record_bind_phase(HostPhaseKind::BindHostViewClose, view_close_phase, "");
        if (total_tasks < 0) {
            LOG_ERROR("host-orch: orchestration run failed");
            return total_tasks;
        }
        runtime->dev.host_total_tasks = total_tasks;
        LOG_INFO("host-orch: submitted %d tasks on host", total_tasks);
    }

    plan->diagnostic_outcome = dep_gen_host_graph_take(&plan->diagnostic_graph);
    plan->park(*runtime);
    const int commit_rc = resume_host_plan(runtime, api, plan);
    if (commit_rc != 0) return commit_rc;

    LOG_INFO("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

    prepared = true;
    return 0;
}

/**
 * Consume this run's metadata in dependency order, synchronously.
 * A failed copy stops publication and consumes the remaining sources. Device
 * destinations remain owned until the failed-prepare cleanup releases them.
 */
extern "C" int publish_run_image_impl(Runtime *runtime, const HostApi *api) {
    if (runtime == nullptr || api == nullptr) {
        LOG_ERROR("publish_run_image_impl: null runtime or HostApi");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    auto host_phase_guard = RAIIScopeGuard([api]() {
        host_phase_trace_end(api);
    });
    auto publication = runtime->take_pending_publication();
    if (publication.bytes == 0 || publication.device_target == nullptr || publication.source == nullptr) {
        LOG_ERROR("publish_run_image_impl: no prepared runtime image to publish");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // The slot owns snapshots before this local record is destroyed. Onboard launch
    // publishes them on its AICPU stream before the descriptor and entry fork;
    // the large execution image stays on the prepare path to overlap execution.
    std::vector<HostMetadataRegion> metadata;
    metadata.reserve(publication.prerequisites.size() + (api->capture_recording() ? 1 : 0));
    for (const auto &region : publication.prerequisites)
        metadata.push_back({region.device_target, region.source, region.bytes});
    if (api->capture_recording())
        metadata.push_back({publication.device_target, publication.source, publication.bytes});
    const int metadata_rc = api->stage_launch_metadata(metadata.data(), metadata.size());
    if (metadata_rc != 0) {
        LOG_ERROR("host-orch: launch metadata staging failed: %d", metadata_rc);
        return metadata_rc;
    }
    if (api->capture_recording()) return 0;
    const BindPhaseMark h2d_phase = bind_phase_begin();
    if (api->copy_to_device(publication.device_target, publication.source, publication.bytes) != 0) {
        LOG_ERROR("host-orch: H2D of the runtime image failed");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    {
        // The image size and the task count are on the descriptor this bind
        // filled, and the copied zone is what the length has left over, so only
        // the pool counters had to travel.
        const uint64_t image_bytes = runtime->dev.sm_image_bytes;
        const uint64_t copied_bytes = publication.bytes - image_bytes;
        // The widest attribute string a segment formats: eight uint64 fields plus
        // their labels. With the counters ahead of it in the recorded string, this
        // is the tail a truncation eats first.
        char attrs[kBindAttrsCapacity];
        snprintf(
            attrs, sizeof(attrs),
            "nt=%" PRIu64 " bytes=%" PRIu64 " copied=%" PRIu64 " sm=%" PRIu64 " args=%" PRIu64 "/%" PRIu64 "/%" PRIu64,
            static_cast<uint64_t>(runtime->dev.host_total_tasks), publication.bytes, copied_bytes, image_bytes,
            publication.fanin_elems, publication.tensor_elems, publication.scalar_elems
        );
        record_bind_phase(HostPhaseKind::BindArenaH2d, h2d_phase, attrs, publication.bytes);
    }
    if (auto saved = api->get_prepare_plan()) {
        // Keep diagnostics owned by the plan until publication succeeds. An
        // unpublished abort or failed copy must not strand its graph in TLS.
        const auto plan = std::static_pointer_cast<HostPreparePlan>(saved);
        dep_gen_host_graph_restore_for_export(&plan->diagnostic_graph, plan->diagnostic_outcome);
    }
    return 0;
}

/**
 * Stage one run's inputs. A no-op for this runtime.
 *
 * host_build_graph runs its orchestrator on the host during bind, and that
 * orchestrator reads the input tensors it was given while it builds the graph,
 * so the bytes have to be in place before orchestration rather than after it —
 * `bind_callable_to_runtime_impl` copies them there. What makes that sound is
 * that a bind serves exactly one run: the graph this bind materializes carries
 * the values it read, so the bytes and the graph belong to the same run.
 */
extern "C" int copy_in_run_inputs_impl(const Runtime * /*runtime*/, const HostApi * /*api*/) { return 0; }

/**
 * Release the tensor bindings the bind recorded.
 *
 * Its own entry rather than the tail of the copy-back, so that reading a run's
 * results and retiring the device memory behind them are separately orderable.
 */
extern "C" int release_run_bindings_impl(Runtime *runtime, const HostApi *api) {
    if (runtime == nullptr || api == nullptr) {
        LOG_ERROR("release_run_bindings_impl: null runtime or HostApi");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    // Close only this run's trace; another run may already own the recorder.
    host_phase_trace_end(api);
    runtime->clear_pending_publication();
    release_run_tensor_leases(runtime, api);
    // The dispatch table is owned by bind_callable_to_runtime, which clears it
    // before replaying the active callable's addresses. The chip-callable device
    // buffer behind those addresses is pool-managed by DeviceRunner (keyed by
    // content hash) and bulk-freed in DeviceRunner::finalize(), so re-running the
    // same callable repeatedly does not re-upload.
    return 0;
}

/**
 * Inspect one run's results.
 *
 * This function:
 * 1. Reads the device-side runtime status when the run failed on the device
 * 2. Copies written tensors from device back to host
 *
 * It releases nothing; `release_run_bindings_impl` ends the bindings it reads.
 *
 * @param runtime       Pointer to Runtime
 * @param execution_rc  Device-runner drain status after successful enqueue,
 *                      or enqueue status on failure
 * @param launched      Nonzero when this run reached a stream, and its
 *                      device-side status is therefore readable
 * @return 0 on success, -1 on failure
 */
extern "C" int copy_back_run_outputs_impl(const Runtime *runtime, const HostApi *api, int execution_rc, int launched) {
    if (runtime == nullptr) {
        LOG_ERROR("Runtime pointer is null");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    if (api == nullptr) {
        LOG_ERROR("HostApi pointer is null");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    int rc = 0;

    LOG_INFO("=== Copying Results Back to Host ===");

    // Copy all recorded tensors from device back to host
    const TensorLease *tensor_leases = runtime->tensor_leases().data();
    int tensor_lease_count = static_cast<int>(runtime->tensor_leases().size());

    LOG_INFO("ChipTensor leases to process: %d", tensor_lease_count);

    bool skip_tensor_copy_back = execution_rc != 0;
    int32_t runtime_status = 0;
    SharedMemoryHeader host_header;
    memset(&host_header, 0, sizeof(host_header));

    // The shared-memory status is device state, readable only for a run that
    // reached a stream.
    if (execution_rc != 0 && launched != 0) {
        runtime_status = read_runtime_status(runtime, api, &host_header);
    }
    if (runtime_status != 0) {
        int32_t sched_error_code = host_header.sched_error_code.load(std::memory_order_relaxed);
        LOG_RUNTIME_FAILURE(SIMPLER_ERROR_NONE, sched_error_code, runtime_status);
    }

    if (skip_tensor_copy_back) {
        LOG_WARN("Skipping tensor copy-back because execution failed");
    } else {
        for (int i = 0; i < tensor_lease_count; i++) {
            const TensorLease &lease = tensor_leases[i];

            // Skip if device pointer is null
            if (lease.dev_ptr == nullptr) {
                LOG_WARN("ChipTensor %d has null device pointer, skipping", i);
                continue;
            }

            // If host pointer is null, this is a device-only allocation (no copy-back)
            if (lease.host_ptr == nullptr) {
                LOG_DEBUG("ChipTensor %d: device-only allocation (no copy-back)", i);
                continue;
            }

            // Read-only INPUT tensors were uploaded H2D but the kernel never
            // wrote them — copying them back (potentially ~GB) is pure waste.
            // They are still released through release_kind below.
            if (!lease.needs_copy_back) {
                LOG_DEBUG("ChipTensor %d: read-only input, skipping copy-back", i);
                continue;
            }

            int copy_rc = api->copy_from_device(lease.host_ptr, lease.dev_ptr, lease.size);
            if (copy_rc != 0) {
                LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
                rc = copy_rc;
            } else {
                LOG_DEBUG("ChipTensor %d: %zu bytes copied to host", i, lease.size);
            }
        }
    }

    LOG_INFO("=== Result Copy-Back Complete ===");

    if (rc == 0 && runtime_status != 0) {
        rc = runtime_status;
    }

    return rc;
}

extern "C" size_t native_capture_result_bytes_impl() { return sizeof(DeviceRunResultRegion); }

extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count) {
    if (count != nullptr) *count = 0;
    return nullptr;
}
