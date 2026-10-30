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

#include "host_build_graph/host_graph_build.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "assert_compat.h"
#include "common/host_api.h"
#include "common/platform_config.h"
#include "host/raii_scope_guard.h"
#include "host_build_graph/host/bind_phase.h"
#include "host_build_graph/host/dep_gen_host_graph.h"
#include "host_build_graph/host/graph_host_state.h"
#include "host_build_graph/host/graph_recorder_pool.h"
#include "host_build_graph/host/host_phase_trace.h"
#include "host_build_graph/host/host_tensor_access.h"
#include "host_build_graph/host/orch_ops.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/host/ready_queue_sizing.h"
#include "host_build_graph/host/retained_prepare_plan.h"
#include "host_build_graph/runtime.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/types.h"
#include "runtime_status/error_log.h"
#include "utils/device_arena.h"
#include "worker/runtime_c_api.h"

namespace {

// host_build_graph host-orch: the orchestrator builds the task graph in a host
// shared-memory mirror, and every cross-task reference it stores is an offset or
// an index from its own block. The bytes the host wrote are therefore the bytes
// the device schedules, with no on-device or pre-copy pointer fixup.

// What the Definition pass copied to the device: the distinct objects, the bytes
// of the one block holding them (inter-object alignment padding included), and how
// many of them the recorders could not build in the block, so that this pass had to
// copy them in. The object count is smaller than the run's Graph task count, which exceeds it
// by the replay factor — one Definition serves every task with its key.
struct DefinitionUploads {
    size_t count;
    uint64_t bytes;
    size_t spilled;
};

// Prepare the run's Definition objects and bind every outer Graph task to the one
// with its key. The recorders built most or all of them in place in the block's
// host staging, each as [GraphDefinitionHeader][Definition image] at the offset it
// claimed, so this pass writes the headers, copies in whatever did not fit, and
// records the used prefix for synchronous publication. Device initial classify
// replaces each task's graph_context with an execution constructed in its own heap.
int32_t bind_graph_definitions(
    Runtime *runtime, const HostApi *api, const hbg::GraphTaskDefinitionBindings &definition_bindings,
    DefinitionUploads *uploads
) {
    *uploads = DefinitionUploads{};
    const size_t count = definition_bindings.bindings.size();
    const auto align_up = [](size_t value) {
        return (value + GRAPH_DEFINITION_OBJECT_ALIGN - 1) & ~(GRAPH_DEFINITION_OBJECT_ALIGN - 1);
    };
    const auto &packed = definition_bindings.objects;
    const size_t block_bytes = definition_bindings.bytes;
    uploads->spilled = definition_bindings.spilled;

    // Named before anything is packed, so a run with no Graph task leaves the
    // device expecting nothing rather than inheriting a predecessor's section.
    runtime->publish_graph_section(GraphSectionSource::None, 0, 0);

    std::byte *staging = nullptr;
    if (block_bytes != 0) {
        const BindPhaseMark pack_phase = bind_phase_begin();
        void *staging_addr = nullptr;
        // Growing the staging preserves what the recorders wrote into it, and the
        // offsets above name positions rather than addresses, so a block that moves
        // here costs nothing. Nothing is recording by now, which is what makes the
        // move safe at all.
        if (api->acquire_graph_staging(block_bytes, GRAPH_DEFINITION_OBJECT_ALIGN, &staging_addr) != 0) {
            LOG_ERROR(
                "host-orch: failed to retain %zu bytes for %zu Graph Definition object(s)", block_bytes, packed.size()
            );
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        staging = static_cast<std::byte *>(staging_addr);
        for (const auto &[key, object] : packed) {
            std::byte *base = staging + object.object_offset;
            std::byte *image = base + sizeof(GraphDefinitionHeader);
            if (object.copy != nullptr) std::memcpy(image, object.copy, object.image_bytes);
            // Built value-initialized and copied over the whole header, so every byte
            // of the object's framing — padding included — is defined by this write
            // rather than by what the retained staging held before it.
            const auto *definition = reinterpret_cast<const GraphDefinition *>(image);
            GraphDefinitionHeader framing{};
            framing.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
            framing.graph_key = definition->graph_key;
            framing.definition_bytes = definition->total_bytes;
            std::memcpy(base, &framing, sizeof(framing));
            const size_t object_bytes = sizeof(GraphDefinitionHeader) + object.image_bytes;
            const size_t padded = align_up(object_bytes);
            std::memset(base + object_bytes, 0, padded - object_bytes);
        }
        uploads->count = packed.size();
        uploads->bytes = block_bytes;
        // Into the run's AICPU launch package, which RTS copies with the launch:
        // no device block is allocated, uploaded or freed for these bytes, and a
        // length the package cannot represent is refused here, before any
        // submission.
        uint32_t section_source = static_cast<uint32_t>(GraphSectionSource::None);
        uint64_t section_base = 0;
        if (api->publish_graph_section(staging, block_bytes, &section_source, &section_base) != 0) {
            LOG_ERROR("host-orch: failed to carry %zu Definition byte(s) to the device", block_bytes);
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        // Which carrier delivered it is the platform's answer, not this bind's:
        // onboard it is the launch package RTS copies, in simulation a run-owned
        // host snapshot.
        runtime->publish_graph_section(
            static_cast<GraphSectionSource>(section_source), section_base, static_cast<uint32_t>(block_bytes)
        );
        char attrs[kBindAttrsCapacity];
        snprintf(
            attrs, sizeof(attrs), "defs=%zu bytes=%zu submissions=%zu spilled=%zu", uploads->count, block_bytes, count,
            uploads->spilled
        );
        // Packing, not an upload: the phase measures assembling the section and
        // taking it into the launch package, and the transfer it used to name is
        // now part of the launch RTS makes.
        record_bind_phase(HostPhaseKind::BindGraphPack, pack_phase, attrs, block_bytes);
    }

    for (const auto &[slot, offset] : definition_bindings.bindings) {
        slot->to_descriptor().graph_definition_offset = offset;
        slot->graph_context = nullptr;
    }
    return 0;
}

struct GraphHostStateBinding {
    explicit GraphHostStateBinding(OrchestratorState &orchestrator, GraphHostState *state) :
        orchestrator(orchestrator) {
        orchestrator.graph_host_state = state;
    }
    ~GraphHostStateBinding() { orchestrator.graph_host_state = nullptr; }

    OrchestratorState &orchestrator;
};

}  // namespace

int32_t hbg::build_graph(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, RuntimeContext *rt, uint64_t sm_size,
    uint64_t task_capacity, void *host_orch_func_ptr, const ChipTaskArgs &orch_l2, GraphBuild &build
) {
    if (build.build_complete || build.graph_state) return PTO_RUNTIME_ERR_INVALID_STATE;
    if (runtime == nullptr || api == nullptr || rt == nullptr || host_orch_func_ptr == nullptr || task_capacity == 0 ||
        task_capacity > static_cast<uint64_t>(TaskId::GLOBAL_TASK_MAX_NUM) ||
        sm_size != SharedMemoryHandle::calculate_size(task_capacity))
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    // The dep_gen graph belongs to the orchestration that is about to run.
    dep_gen_host_graph_begin_capture();

    // Init-on-write: descriptors, payloads, slot_states and task_states are
    // each written per task at submit and read only for [0, total_tasks). Zero
    // only the fixed-size header here; the per-slot segments are initialized in
    // orch::prepare_task and shipped bounded to total_tasks below.
    const sm_layout::SegmentOffsets sm_segs = sm_layout::segment_offsets(task_capacity);
    // The mirror belongs to the runner, one per pipeline slot, and lives past the
    // bind that writes it: at the configured task capacity it is tens of MB, far
    // above the block size glibc recycles, so a per-bind buffer costs an mmap and
    // an munmap per bind. Nothing carries over inside it — the header is cleared
    // here, and the prefix of each segment that ships is one this bind wrote. The
    // layout needs CHIP_ALIGN_SIZE: every segment offset is a multiple of it and
    // ChipTaskSlotState is alignas(64).
    void *host_sm = nullptr;
    if (api->acquire_sm_mirror(static_cast<size_t>(sm_size), CHIP_ALIGN_SIZE, &host_sm) != 0 || host_sm == nullptr) {
        LOG_ERROR("host-orch: host SM mirror of %" PRIu64 " bytes unavailable", sm_size);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    std::memset(host_sm, 0, sm_segs.storage);

    // Re-point the orchestrator half at the host SM (scheduler keeps device SM).
    // Storage is retained per slot; the orchestrator and its context belong to this bind.
    auto scratch = api->get_orchestration_scratch();
    if (!scratch) {
        try {
            scratch = std::make_shared<OrchestrationScratch>();
        } catch (const std::bad_alloc &) {
            LOG_ERROR("host-orch: failed to allocate retained orchestration storage");
            return PTO_RUNTIME_ERR_INTERNAL;
        }
        api->set_orchestration_scratch(scratch);
    }
    auto &storage = *static_cast<OrchestrationScratch *>(scratch.get());
    OrchestratorState orchestrator{};
    storage.swap_storage(orchestrator);
    OrchContext orch_ctx{};
    orch_ctx.orchestrator = &orchestrator;
    orch_ctx.rt = rt;
    RAIIScopeGuard retained_storage_return([&storage, &orchestrator]() {
        storage.swap_storage(orchestrator);
    });
    // The graph heap is allocated out of the HEAP_VIRTUAL_BASE window: its device
    // region is committed below, once this pass has revealed how many bytes it
    // actually needs, and compact_live_image moves every address the orchestrator
    // wrote onto the real base before the image travels.
    if (!orchestrator.init(host_sm, reinterpret_cast<void *>(HEAP_VIRTUAL_BASE), MAX_HEAP_CAPACITY, task_capacity)) {
        LOG_ERROR("host-orch: orchestrator init against host SM failed");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Initialize the host SM header (ring flow control) so submit_task can run.
    SharedMemoryHandle &host_sm_handle = build.host_sm_handle;
    if (!host_sm_handle.init(host_sm, sm_size, task_capacity)) {
        LOG_ERROR("host-orch: host SM init failed");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // The recorders build their Definition objects straight into the retained
    // staging block, so it is claimed before orchestration starts and at whatever
    // capacity the previous bind left behind — the run's real total is not known
    // until every recording has ended. What does not fit is built in its own
    // buffer and copied by the upload, which then grows the block, so the arena
    // reaches a run's high-water mark within one bind of needing it.
    GraphDefinitionArena definition_arena{};
    definition_arena.object_prefix_bytes = sizeof(GraphDefinitionHeader);
    definition_arena.object_align = GRAPH_DEFINITION_OBJECT_ALIGN;
    {
        void *staging = nullptr;
        size_t staging_bytes = 0;
        api->get_graph_definition_staging(&staging, &staging_bytes);
        if (staging != nullptr && reinterpret_cast<uintptr_t>(staging) % definition_arena.object_align == 0) {
            definition_arena.base = static_cast<std::byte *>(staging);
            definition_arena.capacity = staging_bytes;
        }
    }

    build.graph_state = make_graph_host_state(definition_arena);
    GraphHostStatePtr &graph_state = build.graph_state;
    if (!graph_state) {
        LOG_ERROR("host-orch: failed to allocate Graph host state");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    GraphHostStateBinding graph_binding(orchestrator, graph_state.get());

    // Install the ops table (host s_runtime_ops) and latch this run's cluster
    // counts. worker_count is published by DeviceRunner::prepare_launch_shape
    // before this bind, so the host orchestrator sees the same geometry the
    // AICPU re-derives from the handshake at boot.
    const int32_t block_dim = runtime->get_worker_count() / PLATFORM_CORES_PER_BLOCKDIM;
    if (block_dim < 1) {
        LOG_ERROR("host-orch: worker_count %d yields no clusters", runtime->get_worker_count());
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    orch_bind_ops(&orch_ctx);
    orchestrator.total_cluster_count = block_dim * PLATFORM_AIC_CORES_PER_BLOCKDIM;
    orchestrator.total_aiv_count = block_dim * PLATFORM_AIV_CORES_PER_BLOCKDIM;
    rt->mode = MODE_EXECUTE;
    // Host get/set operates only on explicit HOST/NONE arguments. Uploaded
    // arguments and runtime outputs are DEVICE tensors and cannot be read here.
    const auto *entry_points = reinterpret_cast<const HostOrchEntryPoints *>(host_orch_func_ptr);
    if (entry_points->bind == nullptr) {
        LOG_ERROR("host-orch: orch .so framework_bind_orch_context was not resolved");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    orch_ctx.tensor_access = &tensor_access;
    // Binds the orchestration .so's own framework_current_orch_context, which its
    // inline rt_submit_* read. The host library links a same-named copy from
    // orchestration/common.cpp, but nothing outside the .so includes
    // orchestration_api.h, so nothing reads that one — rt_scope_* and
    // rt_orchestration_done take the context as an argument.
    entry_points->bind(&orch_ctx);

    // What the .so holds for this bind is released when this function leaves, by whichever
    // path: the state it covers -- the Graph Config registry today -- is paired with
    // graph_state owned by the build's exclusive workspace lease. The
    // bound context is one of them: orch_ctx is a local of this frame, so the .so's
    // global may not outlive it. Unbinding after finish() keeps the context reachable
    // for whatever the .so still does there.
    RAIIScopeGuard orch_finish([entry_points]() {
        if (entry_points->finish != nullptr) entry_points->finish();
        entry_points->bind(nullptr);
    });

    const BindPhaseMark orch_phase = bind_phase_begin();
    {
        // Recorder jobs borrow this build's state and execute code from its SO.
        // Drain before commit can retire their entries, and before unwinding can
        // release graph_state, orchestrator, or the tensor-access views.
        RAIIScopeGuard recorder_completion([&orch_ctx]() {
            orch_ctx.ops->graph_record_wait(&orch_ctx);
        });
        rt_scope_begin(&orch_ctx);
        entry_points->entry(orch_l2);
        rt_scope_end(&orch_ctx);
    }
    rt_orchestration_done(&orch_ctx);
#if SIMPLER_ORCH_PROFILING
    // Per-sub-step cumulatives across this bind's submits. The accumulators only
    // exist in a SIMPLER_ORCH_PROFILING build (build_runtimes.py --profiling-orch 1),
    // and reading them also resets them, so this is the bind's own total. Emitted
    // as spans rather than LOG_INFO because INFO is suppressed at the default log
    // level. Like the phase spans these are summed cost shares, not intervals.
    {
        const OrchProfilingData prof = orchestrator_get_profiling();
        const std::pair<const char *, uint64_t> steps[] = {
            {"alloc", prof.alloc_ns},   {"args", prof.args_ns},   {"lookup", prof.lookup_ns},
            {"insert", prof.insert_ns}, {"fanin", prof.fanin_ns},
        };
        for (const auto &step : steps) {
            if (step.second == 0) continue;
            LOG_TIMING("host-orch step=%s ns=%" PRIu64 " submits=%" PRId64, step.first, step.second, prof.submit_count);
        }
    }
#endif

    // A latched fatal means the graph in the mirror is not the graph the orchestration
    // described — a heap or tensormap exhaustion drops tasks, a fanin overflow drops
    // edges. Uploading it would launch the device on an incomplete graph and surface
    // the cause as whatever the device notices second, usually a scheduler timeout.
    if (orchestrator.is_fatal()) {
        // The latched code is the diagnosis, so it is what the caller sees — through the
        // same mapping the run path uses, since a caller cannot tell which of the two
        // noticed.
        const int32_t orch_error = orchestrator.fatal_code.load(std::memory_order_acquire);
        const int32_t status = runtime_status_from_error_code(orch_error);
        LOG_RUNTIME_FAILURE(orch_error, SIMPLER_ERROR_NONE, status);
        LOG_ERROR(
            "host-orch: refusing to upload an incomplete graph after %" PRIu64 " heap bytes",
            orchestrator.task_allocator.heap_used_bytes()
        );
        return status;
    }

    const int32_t total_tasks = orchestrator.task_allocator.active_count();
    {
        char attrs[kBindAttrsCapacity];
        snprintf(
            attrs, sizeof(attrs), "tasks=%" PRId32 " heap_used=%" PRIu64 " sm_mirror=%" PRIu64, total_tasks,
            orchestrator.task_allocator.heap_used_bytes(), sm_size
        );
        record_bind_phase(HostPhaseKind::BindHostOrch, orch_phase, attrs);
    }
    // After the span closes: the reduction walks a few hundred records and emits
    // five markers, which must not be charged to the bind it measures.

    // total_tasks sizes the bounded per-segment H2D copies below; a value outside
    // [0, task_capacity] would make those copies read/write out of bounds.
    if (total_tasks < 0 || static_cast<uint64_t>(total_tasks) > task_capacity) {
        LOG_ERROR("host-orch: total_tasks %d out of range [0, %" PRIu64 "]", total_tasks, task_capacity);
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Submission already classified each slot while its fields were hot.
    ReadyQueuePopulations ready_queue_populations = orchestrator.ready_queue_populations;

    if (!hbg::resolve_graph_task_definitions(*graph_state, ready_queue_populations, build.definitions)) {
        LOG_ERROR("host-orch: invalid or oversized Graph Definition plan");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    ready_queue_populations = build.definitions.populations;

    ReadyQueueCapacities ready_queue_capacities{};
    const int32_t ready_queue_status = derive_ready_queue_capacities(ready_queue_populations, &ready_queue_capacities);
    if (ready_queue_status != 0) {
        LOG_ERROR(
            "host-orch: ready queue reachable population exceeds %" PRIu64 " (ready=%" PRIu64 "/%" PRIu64 "/%" PRIu64
            ", sync=%" PRIu64 "/%" PRIu64 "/%" PRIu64 ", dummy=%" PRIu64 ", graph_prepare=%" PRIu64 ")",
            READY_QUEUE_CAPACITY_LIMIT, ready_queue_populations.ready[0], ready_queue_populations.ready[1],
            ready_queue_populations.ready[2], ready_queue_populations.ready_sync[0],
            ready_queue_populations.ready_sync[1], ready_queue_populations.ready_sync[2], ready_queue_populations.dummy,
            ready_queue_populations.graph_prepare
        );
        LOG_RUNTIME_FAILURE(SIMPLER_ERROR_NONE, SIMPLER_ERROR_READY_QUEUE_OVERFLOW, ready_queue_status);
        return ready_queue_status;
    }
    rt->prebuilt_layout.sched.capacities = ready_queue_capacities;
    host_phase_trace_note_submitted(static_cast<uint64_t>(total_tasks));

    // The count travels inside the header the restack copies wholesale, which is
    // what lets the device bound its slot walks without a second
    // carrier. Written after the range check above, so the value the device reads
    // is one the segments are actually pitched to.
    reinterpret_cast<SharedMemoryHeader *>(host_sm)->tasks.total_tasks = total_tasks;

    // The device reads no task slot past total_tasks, so only that prefix of each
    // segment has to travel. In the mirror the orchestrator wrote, the four
    // prefixes are a whole task_capacity apart, which would make the upload four
    // copies of a few hundred kilobytes each — and at these sizes a copy_to_device
    // is priced by the call, not by the bytes.
    //
    // So the prefixes are restacked into an image pitched to total_tasks, where
    // they are contiguous, and that image goes up as one copy. The device attaches
    // with the same pitch, which is sound because a task id is its own slot index:
    // every id is below total_tasks and indexes the image directly.
    const uint64_t nt = static_cast<uint64_t>(total_tasks);
    // What this bind actually put in the pools. The orchestrator's cursors are the
    // exact populated extent of each one — no scan of the mirror is needed, and the
    // image ships that much rather than the worst case the mirror is dimensioned for.
    const OrchestratorState &orch_state = orchestrator;
    const sm_layout::BindUsage bind_usage{
        nt,
        static_cast<uint64_t>(orch_state.fanin_pool_cursor),
        static_cast<uint64_t>(orch_state.tensor_pool_cursor),
        static_cast<uint64_t>(orch_state.scalar_pool_cursor),
    };
    const uint64_t image_bytes = sm_layout::segment_offsets(sm_layout::image_extents(bind_usage)).end;

    const uint64_t used_heap = orchestrator.task_allocator.heap_used_bytes();
    if (used_heap > UINT64_MAX - (DeviceArena::kDefaultBaseAlign - 1))
        return runtime_status_from_error_code(SIMPLER_ERROR_SUBMIT_CAPACITY_EXCEEDED);
    build.sm_mirror = host_sm;
    build.task_capacity = task_capacity;
    build.total_tasks = total_tasks;
    build.ready_queue_populations = ready_queue_populations;
    build.ready_queue_capacities = ready_queue_capacities;
    build.bind_usage = bind_usage;
    build.image_bytes = image_bytes;
    build.heap_bytes = CHIP_ALIGN_UP(std::max<uint64_t>(used_heap, CHIP_ALIGN_SIZE), DeviceArena::kDefaultBaseAlign);
    build.build_complete = true;
    return total_tasks;
}

int32_t hbg::run_host_orchestration(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, RuntimeContext *rt,
    DeviceArena &host_arena, const RuntimeArenaLayout &layout, uint64_t sm_size, uint64_t task_capacity,
    void *host_orch_func_ptr, const ChipTaskArgs &orch_l2, RetainedHostPreparePlan<RuntimeArenaLayout> &plan
) {
    hbg::GraphBuild build;
    const int32_t build_status =
        hbg::build_graph(runtime, api, tensor_access, rt, sm_size, task_capacity, host_orch_func_ptr, orch_l2, build);
    if (build_status < 0) return build_status;
    const int32_t total_tasks = build.total_tasks;
    void *host_sm = build.sm_mirror;
    auto &host_sm_handle = build.host_sm_handle;
    const auto &bind_usage = build.bind_usage;
    const uint64_t image_bytes = build.image_bytes;
    runtime->dev.sm_image_bytes = image_bytes;
    DefinitionUploads definition_uploads{};
    const int32_t definition_status = bind_graph_definitions(runtime, api, build.definitions, &definition_uploads);
    if (definition_status != 0) return definition_status;

    // Only now are both sizes known, so this is where the two device regions are
    // committed: the arena up to its shared-memory tail, and the graph heap to the
    // bytes orchestration actually handed out. setup_static_arena commits per
    // region and short-circuits a request an existing one already covers, so a
    // repeated workload pays for neither twice and the heap is grow-only across a
    // Worker's binds.
    // The compact shared-memory image is the only per-run tail in the device
    // arena. GraphExecution is initialized later in each outer Graph heap.
    // A graph whose every output is caller-owned allocates nothing, but its tasks
    // still carry the window's base as their zero-length packed buffer, so the
    // region has to exist for that address to be rebasable. Asking for 0 would
    // release it and leave acquire_pooled_gm_heap with nothing to return.
    //
    // Rounded up to the region's base alignment: the committed span then ends on the
    // same boundary it starts on, so an access at the tail of the last packed buffer
    // stays inside the region even when its width exceeds the bytes that buffer
    // asked for.
    const uint64_t heap_bytes =
        CHIP_ALIGN_UP(std::max<uint64_t>(build.heap_bytes, CHIP_ALIGN_SIZE), DeviceArena::kDefaultBaseAlign);
    plan.host_sm = host_sm;
    plan.sm_size = sm_size;
    plan.task_capacity = task_capacity;
    plan.heap_bytes = heap_bytes;
    plan.image_bytes = image_bytes;
    plan.total_tasks = total_tasks;
    plan.bind_usage = bind_usage;
    return total_tasks;
}
