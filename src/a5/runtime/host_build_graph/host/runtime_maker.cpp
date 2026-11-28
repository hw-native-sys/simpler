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
 * compact task image, and stages it for device publication.
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

#include "host/scheduler_notification_address.h"
#include "host/scheduler_notifications.h"

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
#include <filesystem>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>
#include <unordered_map>
#include <vector>

#include "assert_compat.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/host/dep_gen_host_graph.h"
#include "image_pack.h"
#include "graph_validation.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_args_dump.h"
#include "scheduler/scheduler_types.h"
#include "host_build_graph/host/host_tensor_access.h"
#include "host_build_graph/host/retained_prepare_plan.h"
#include "host_build_graph/host/graph_host_state.h"
#include "host_build_graph/host/host_phase_trace.h"
#include "host_build_graph/host/orchestrator.h"
#include "host_build_graph/host/orch_ops.h"
#include "host_build_graph/host/graph_recorder_pool.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/types.h"
#include "host_build_graph/runtime.h"
#include "state.h"
#include "../../../../common/runtime_status/error_log.h"
#include "../../../../common/task_interface/call_config.h"
#include "../../../../common/worker/runtime_c_api.h"
#include "callable.h"
#include "common/chip_swimlane_profiling.h"
#include "common/host_log_binding.h"
#include "common/host_phase_kind.h"
#include "common/platform_config.h"
#include "common/unified_log.h"
#include "host_log.h"
#include "host/platform_compile_info.h"
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
/**
 * Run-resource sets a kernel-mode context of this runtime declares.
 *
 * A separate number from the program declaration below: `simpler_kernel_mode_init`
 * validates this contract and discards it, so the two are independent
 * declarations rather than one capability read twice.
 */
static constexpr uint32_t kKernelPipelineDepth = 2;

extern "C" const PipelineContract *get_pipeline_contract(void) {
    // Each launch supplies its initial image. Ordered executions share the
    // device heap and the writable portion of the runtime arena.
    // pipeline_depth bounds supported resource sets without preallocating them.
    static const PipelineContract contract = {
        PTO_PIPELINE_CONTRACT_ABI_VERSION,
        4,
        UINT32_MAX,
        {
            {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_POOLED_SCRATCH, 0},
            {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_POOLED_SCRATCH, 0},
            {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
        },
    };
    return &contract;
}

extern "C" int build_kernel_pipeline_contract_impl(const CallConfig *config, PipelineContract *out) {
    if (out == nullptr) return PTO_RUNTIME_ERR_INTERNAL;
    if (config == nullptr) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    // Retained preparations keep the shared heap allocation alive. Kernel's
    // resource-set capacity remains independent of the program path's depth.
    *out = *get_pipeline_contract();
    out->pipeline_depth = kKernelPipelineDepth;
    return 0;
}

extern "C" int concurrent_native_prepare_supported_impl(void) {
    // Each prepare assembles its own host image.
    return 1;
}

extern "C" int
prepared_run_config_compatible_impl(const HostApi *api, const uint64_t *, const uint64_t *, const uint64_t *) {
    // Bind writes host staging. The device writes shared scratch only after
    // launch ordering takes effect, and growth retains the previous backing.
    return api != nullptr && api->launch_owns_execution_image();
}

extern "C" int set_run_ccu_workspace_impl(Runtime *runtime, uint64_t address) {
    runtime->dev.ccu_workspace_addr = address;
    return 0;
}

extern "C" int joined_native_launch_supported_impl(void) {
    // A queued device event wait orders use of the shared heap and scheduler scratch.
    return 1;
}

extern "C" int joined_diagnostic_launch_supported_impl(void) {
    // Every diagnostic producer this runtime admits can hold two runs open.
    // Argument dumping is admitted only on the AICore-scheduler route.
    // Every producer stamps the run epoch its AICPU op
    // was launched with onto the buffer it publishes, and settles that run's
    // own accounting into the terminal bank its pipeline slot reserves, so the
    // host tells the two apart by identity rather than by arrival order.
    //
    // AicoreLifecycle initializes and settles each run's diagnostics across
    // its init and shutdown partitions.
    return 1;
}

extern "C" int args_dump_capture_supported_impl(void) { return 1; }

// RuntimeEnv (call_config.h) is the cross-runtime ABI for per-ring config and
// carries RUNTIME_ENV_RING_COUNT slots, shared with tensormap_and_ringbuffer.
// host_build_graph keeps one task table and reads slot 0, so it only needs the ABI
// to carry at least one.
static_assert(RUNTIME_ENV_RING_COUNT >= 1, "RuntimeEnv must carry the slot host_build_graph reads");

// Host monotonic clock, shared with the record pool so spans and records can be
// read against each other.
static int64_t bind_now_ns() { return static_cast<int64_t>(host_phase_now_ns()); }

// Close one segment of the bind path, recording it and keeping its attributes for
// the line the breakdown prints at the end of the bind.
//
// The breakdown is LOG_TIMING lines rather than `[STRACE]` markers on purpose:
// the marker grammar is the platform's public per-run-stage contract (see
// runtime_c_api.h and docs/dfx/host-trace.md) whose consumers key off a fixed
// stage set, while everything below is host_build_graph's internal breakdown of
// one stage. LOG_TIMING sits at the default log threshold, so these are visible
// without a flag and at any --rounds.
// Minor faults the process has taken. First touch of a freshly mapped region traps
// once per page, and the bind maps its shared-memory mirror and arenas per call, so
// a phase's fault count is what separates work from page-table cost — a count, so it
// does not move with how loaded the box is.
//
// CPU time answers what a count cannot: whether a phase's wall time was spent running
// or waiting. It comes from per-thread CPU clocks and never from rusage —
// `ru_utime`/`ru_stime` are accounted per scheduler tick, 10 ms at CLK_TCK=100, so on
// a phase of a millisecond they quantise to either zero or a whole tick and no split
// survives. A per-thread clock reads the scheduler's running total in nanoseconds.
//
// Two CPU figures, because a phase runs on more than one thread. `cpu_ns` is the bind
// thread's own, so a phase's `dur - cpu_ns` is the time that thread spent off CPU;
// `recorder_cpu_ns` is every recording worker's summed, so its ratio to `dur` is how
// many threads' worth of work ran alongside. Only the first can be subtracted from the
// wall. `thread_minflt` is the same split applied to the fault count: it says how much
// of `minflt` the bind thread took rather than the recorders.
struct BindKernelCounters {
    uint64_t minflt;
    uint64_t thread_minflt;
    uint64_t nivcsw;  // involuntary: the scheduler took the CPU away
    uint64_t nvcsw;   // voluntary: the thread blocked
    uint64_t cpu_ns;
    uint64_t recorder_cpu_ns;
};

static uint64_t thread_cpu_ns() {
    timespec ts{};
    if (clock_gettime(CLOCK_THREAD_CPUTIME_ID, &ts) != 0) return 0;
    return static_cast<uint64_t>(ts.tv_sec) * 1000000000ULL + static_cast<uint64_t>(ts.tv_nsec);
}

static BindKernelCounters bind_kernel_counters() {
    BindKernelCounters counters{};
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        counters.minflt = static_cast<uint64_t>(usage.ru_minflt);
        counters.nivcsw = static_cast<uint64_t>(usage.ru_nivcsw);
        counters.nvcsw = static_cast<uint64_t>(usage.ru_nvcsw);
    }
#if defined(__linux__)
    // RUSAGE_THREAD is a Linux extension. Darwin has no per-thread rusage, so
    // thread_minflt stays zero there and the process total in minflt is the whole of
    // what is available.
    rusage thread_usage{};
    if (getrusage(RUSAGE_THREAD, &thread_usage) == 0) {
        counters.thread_minflt = static_cast<uint64_t>(thread_usage.ru_minflt);
    }
#endif
    counters.cpu_ns = thread_cpu_ns();
    counters.recorder_cpu_ns = graph_recorder_pool().worker_cpu_ns();
    return counters;
}

// One segment's two origins: the instant its duration measures from and the counter
// mark its faults measure from. Both live in the caller's frame, so a phase that
// opens while another is still open cannot disturb it — the two would share a single
// mark if one were held here, and the outer phase would then report counts measured
// from the inner phase's start while its duration still spanned its own.
//
// A phase's own counts are the delta since its own start, so they cover the same
// span its duration does. The marks do not partition the bind: the stretches
// between one phase's close and the next one's open belong to neither, in counts
// exactly as in time. The process-wide counters are process-wide on purpose, so a
// phase that runs while the Graph recorders are working is charged their faults too
// — it is the bind's total page-table cost that is being attributed, not one
// thread's.
struct BindPhaseMark {
    int64_t start_ns{0};
    BindKernelCounters counters{};
};

// Open one segment. Every phase start comes from here, so a phase's two spans cannot
// drift apart — a fault count taken over a wider span than the clock describes work
// the phase does not contain.
static BindPhaseMark bind_phase_begin() {
    BindPhaseMark mark{};
    if (host_phase_breakdown_enabled()) {
        mark.counters = bind_kernel_counters();
    }
    mark.start_ns = bind_now_ns();
    return mark;
}

static void
record_bind_phase(HostPhaseKind kind, const BindPhaseMark &mark, const char *attrs = "", uint64_t payload = 0) {
    const int64_t start_ns = mark.start_ns;
    if (!host_phase_breakdown_enabled()) {
        host_phase_record_bind(static_cast<uint32_t>(kind), static_cast<uint64_t>(start_ns), attrs, payload);
        return;
    }
    const BindKernelCounters now = bind_kernel_counters();
    // Closes both spans at the same instant. The counters have to be read before the
    // attribute string can be formatted, and the record path takes four early returns
    // before it would reach a clock of its own, so leaving the end timestamp to it
    // would let the duration cover work the counts do not. Both ends of the phase now
    // read counters and then the clock, one call apart.
    const int64_t end_ns = bind_now_ns();
    auto since = [](uint64_t current, uint64_t mark) {
        return current >= mark ? current - mark : 0;
    };
    // Counters first, caller attributes after. The buffer is the span attribute
    // field's own width, so an overlong segment loses its tail: a caller's
    // quantity is recoverable from the artifact's `detail` or the run's own
    // sizing, while the counters are this record's only copy and phase_time_split
    // has nothing to fall back on.
    char with_counters[kBindAttrsCapacity];
    const int formatted = snprintf(
        with_counters, sizeof(with_counters),
        "minflt=%" PRIu64 " tminflt=%" PRIu64 " nivcsw=%" PRIu64 " nvcsw=%" PRIu64 " cpu_ns=%" PRIu64
        " rec_cpu_ns=%" PRIu64 "%s%s",
        since(now.minflt, mark.counters.minflt), since(now.thread_minflt, mark.counters.thread_minflt),
        since(now.nivcsw, mark.counters.nivcsw), since(now.nvcsw, mark.counters.nvcsw),
        since(now.cpu_ns, mark.counters.cpu_ns), since(now.recorder_cpu_ns, mark.counters.recorder_cpu_ns),
        *attrs == '\0' ? "" : " ", attrs
    );
    // Mark the cut here, because nothing downstream can: this buffer is exactly
    // as wide as the span's attribute field, so the value the logger receives
    // already fits and its own `~` marker never fires. An unmarked truncation
    // reads as a complete attribute list that is one field short.
    if (formatted >= static_cast<int>(sizeof(with_counters))) {
        with_counters[sizeof(with_counters) - 2] = '~';
    }
    host_phase_record_bind(
        static_cast<uint32_t>(kind), static_cast<uint64_t>(start_ns), with_counters, payload,
        static_cast<uint64_t>(end_ns)
    );
}

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

    // Any positive count is usable: a task ref indexes its slot directly, so no slot
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

static void release_run_tensor_leases(Runtime *runtime, const HostApi *api) {
    const TensorLeaseReleaseCounts counts = release_tensor_leases(runtime->tensor_leases(), api);
    LOG_DEBUG(
        "Released tensor leases: freed=%d buffer_noop=%d external_noop=%d", counts.freed, counts.buffer_noop,
        counts.external_noop
    );
}

namespace {

extern "C" bool publish_runtime_chip_swimlane_extensions(Runtime *runtime) noexcept {
    try {
        return runtime == nullptr || simpler::percounter::publish_calendar_profiling(runtime);
    } catch (...) {
        return false;
    }
}

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
    std::string tmpl = (std::filesystem::temp_directory_path() / "orch_so_XXXXXX").string();
    int fd = mkstemp(tmpl.data());
    if (fd < 0) {
        return false;
    }
    if (fchmod(fd, 0755) != 0) {
        close(fd);
        unlink(tmpl.c_str());
        return false;
    }
    bool ok = write_all_bytes(fd, data, size);
    if (close(fd) != 0) {
        ok = false;
    }
    if (!ok) {
        unlink(tmpl.c_str());
        return false;
    }
    *out_path = tmpl;
    return true;
}

// The orchestration .so exports these (submit_task form).
typedef void (*OrchestrationEntryFunc)(const ChipTaskArgs &);
typedef void (*OrchestrationBindFunc)(OrchContext *);
typedef void (*OrchestrationFinishFunc)();

// Resolved orchestration .so entry points. register_callable_impl allocates one
// of these (the entry, plus the .so's own framework_bind_orch_context, which sets
// the .so-private context pointer its inline rt_submit_* read) and stores its
// pointer in CallableArtifacts::host_orch_func_ptr. Owned for the callable's
// lifetime alongside host_dlopen_handle.
struct HostOrchEntryPoints {
    OrchestrationEntryFunc entry{nullptr};
    OrchestrationBindFunc bind{nullptr};
    // Releases the per-bind registry before its orchestration context is unbound.
    OrchestrationFinishFunc finish{nullptr};
};

// host_build_graph host-orch: the orchestrator builds the task graph in a host
// shared-memory mirror, and every cross-task reference it stores is an offset or
// an index from its own block. The bytes the host wrote are therefore the bytes
// the device schedules, with no on-device or pre-copy pointer fixup.

// Prepare the run's Definition objects and bind every outer Graph task to the one
// with its key. The recorders built most or all of them in place in the block's
// host staging, each as [GraphImageHeader][Definition image] at the offset it
// claimed, so this pass writes the headers, copies in whatever did not fit, and
// records the used prefix for synchronous publication.
using GraphCalendarHints = std::map<int32_t, simpler::hbg::GraphCalendarHint>;

bool bind_graph_definitions(
    Runtime *runtime, const HostApi *api, GraphHostState &graph_state, int32_t total_tasks,
    GraphCalendarHints &calendar_hints
) {
    std::vector<bool> bound_invocations;
    if (graph_host_upload_count(graph_state) != 0) bound_invocations.resize(static_cast<size_t>(total_tasks), false);
    const size_t count = graph_host_upload_count(graph_state);
    std::vector<GraphHostImage> definitions = graph_host_definitions(graph_state);
    const auto align_up = [](size_t value) {
        return (value + GRAPH_DEFINITION_OBJECT_ALIGN - 1) & ~(GRAPH_DEFINITION_OBJECT_ALIGN - 1);
    };
    struct PackedDefinition {
        size_t object_offset;   // of the object's header, from the block base
        size_t image_bytes;     // the Definition image alone
        const std::byte *copy;  // the image to copy in, or nullptr when built in place
        uint64_t execution_bytes{0};
    };
    std::unordered_map<uint64_t, PackedDefinition> packed;
    // Objects the recorders built already occupy the arena's used prefix at the
    // offsets they claimed, so the block starts out that long and the rest are
    // appended past them.
    size_t block_bytes = graph_host_arena_used(graph_state);
    size_t spilled = 0;
    for (const GraphHostImage &entry : definitions) {
        if (entry.bytes < sizeof(Graph)) continue;
        if (entry.spill == nullptr) {
            packed.emplace(entry.graph_key, PackedDefinition{entry.object_offset, entry.bytes, nullptr});
            continue;
        }
        const size_t object_offset = block_bytes;
        block_bytes += align_up(sizeof(GraphImageHeader) + entry.bytes);
        packed.emplace(entry.graph_key, PackedDefinition{object_offset, entry.bytes, entry.spill});
        ++spilled;
    }

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
            return false;
        }
        staging = static_cast<std::byte *>(staging_addr);
        for (const auto &[key, object] : packed) {
            std::byte *base = staging + object.object_offset;
            std::byte *image = base + sizeof(GraphImageHeader);
            if (object.copy != nullptr) std::memcpy(image, object.copy, object.image_bytes);
            // Built value-initialized and copied over the whole header, so every byte
            // of the object's framing — padding included — is defined by this write
            // rather than by what the retained staging held before it.
            const auto *definition = reinterpret_cast<const Graph *>(image);
            GraphImageHeader framing{};
            framing.magic = GRAPH_DEFINITION_OBJECT_MAGIC;
            framing.graph_key = definition->graph_key;
            framing.definition_bytes = definition->total_bytes;
            std::memcpy(base, &framing, sizeof(framing));
            const size_t object_bytes = sizeof(GraphImageHeader) + object.image_bytes;
            const size_t padded = align_up(object_bytes);
            std::memset(base + object_bytes, 0, padded - object_bytes);
        }
        char attrs[kBindAttrsCapacity];
        snprintf(
            attrs, sizeof(attrs), "defs=%zu bytes=%zu submissions=%zu spilled=%zu", packed.size(), block_bytes, count,
            spilled
        );
        // The host section is the original Definition source for protocol
        // validation; commit copies native plans into scheduler GM.
        record_bind_phase(HostPhaseKind::BindGraphPack, pack_phase, attrs, block_bytes);
    }

    for (auto &[key, object] : packed) {
        const auto *image = staging + object.object_offset + sizeof(GraphImageHeader);
        std::string error;
        if (!validate_graph_image(image, object.image_bytes, key, &error)) {
            LOG_ERROR("host-orch: invalid Graph Definition: %s", error.c_str());
            return false;
        }
        const auto &definition = *reinterpret_cast<const Graph *>(image);
        if (!task_image::graph_storage_bytes(definition, &object.execution_bytes)) return false;
        for (int32_t task = 0; task < definition.task_count; ++task) {
            SubTaskDefinition child{};
            std::memcpy(
                &child, image + definition.off_sub_tasks + static_cast<size_t>(task) * sizeof(child), sizeof(child)
            );
            for (uint8_t lane = 0; lane < SUBTASK_SLOT_COUNT; ++lane) {
                if (child.active_mask.has(static_cast<SubtaskSlot>(lane)) &&
                    runtime->get_function_bin_addr(child.kernel_id[lane]) == 0) {
                    LOG_ERROR("A5 HBG: Graph child has an unregistered callable");
                    return false;
                }
            }
        }
    }

    for (size_t index = 0; index < count; ++index) {
        std::optional<GraphHostUpload> upload = graph_host_upload(graph_state, index);
        if (!upload.has_value() || upload->outer_slot == nullptr || upload->outer_slot->task_kind != TaskKind::GRAPH) {
            LOG_ERROR("host-orch: invalid pending Graph task");
            return false;
        }
        auto object_it = packed.find(upload->graph_key);
        if (object_it == packed.end() || staging == nullptr) {
            LOG_ERROR("host-orch: Graph task has no matching prepared Definition object");
            return false;
        }
        // Validate the prepared bytes that publication will copy to the device.
        const auto *definition =
            reinterpret_cast<const Graph *>(staging + object_it->second.object_offset + sizeof(GraphImageHeader));
        if (definition->total_bytes != object_it->second.image_bytes) {
            LOG_ERROR("host-orch: Graph task has no matching prepared Definition object");
            return false;
        }
        const uint64_t storage_bytes = object_it->second.execution_bytes;
        if (definition->task_count <= 0 || definition->task_count > SUB_TASK_MAX_NUM ||
            definition->graph_key != upload->graph_key ||
            upload->outer_slot->to_payload().tensor_count != definition->boundary_tensor_count ||
            upload->outer_slot->to_payload().scalar_count != definition->boundary_scalar_count) {
            LOG_ERROR("host-orch: invalid Graph Definition for task");
            return false;
        }
        const uintptr_t outer_base =
            reinterpret_cast<uintptr_t>(upload->outer_slot->to_descriptor().packed_buffer_base);
        const uintptr_t outer_end = reinterpret_cast<uintptr_t>(upload->outer_slot->to_descriptor().packed_buffer_end);
        if (outer_end < outer_base || definition->required_heap > UINTPTR_MAX - outer_base ||
            storage_bytes > outer_end - outer_base ||
            definition->required_heap > outer_end - outer_base - storage_bytes) {
            LOG_ERROR("host-orch: Graph runtime storage does not fit its outer task heap");
            return false;
        }
        const uintptr_t storage_addr = outer_base + definition->required_heap;
        if (storage_addr % 64 != 0) {
            LOG_ERROR("host-orch: Graph runtime storage address is misaligned");
            return false;
        }
        // Protocol source capture names the original Definition by its section
        // offset. The device uses the native node map and instance bindings.
        upload->outer_slot->to_descriptor().graph_definition_offset =
            static_cast<uint32_t>(object_it->second.object_offset + sizeof(GraphImageHeader));
        upload->outer_slot->graph_context = nullptr;
        const TaskId shell_id = upload->outer_slot->to_descriptor().task_id;
        if (!shell_id.is_global() || shell_id.local_id() < 0 || shell_id.local_id() >= total_tasks ||
            bound_invocations[static_cast<size_t>(shell_id.local_id())]) {
            LOG_ERROR("host-orch: invalid or duplicate Graph shell identity");
            return false;
        }
        bound_invocations[static_cast<size_t>(shell_id.local_id())] = true;
        if (upload->calendar_hint) calendar_hints.emplace(shell_id.local_id(), *upload->calendar_hint);
    }
    return true;
}

struct GraphHostStateBinding {
    explicit GraphHostStateBinding(OrchestratorState &orchestrator, GraphHostState *state) :
        orchestrator(orchestrator) {
        orchestrator.graph_host_state = state;
    }
    ~GraphHostStateBinding() { orchestrator.graph_host_state = nullptr; }

    OrchestratorState &orchestrator;
};

// The two bootstrap words move together: a released run must not name resident
// storage, and a selected run requires a context base. Both words are published
// through publish_scheduler_bootstrap.
//
// Prepared resources retain the device target. Releasing the binding drops
// the profiling lookup and the descriptor's references to that target.
void release_scheduler_state(Runtime *runtime) {
    if (runtime == nullptr) return;
    simpler::percounter::release_calendar_profiling(runtime);
    runtime->publish_scheduler_bootstrap(0, 0);
    for (int32_t i = 0; i < runtime->get_worker_count(); ++i) {
        runtime->get_workers()[i].aicpu_ready = 0;
        runtime->get_workers()[i].task = 0;
    }
}

bool create_scheduler_state(
    Runtime *runtime, const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    uint64_t task_window_size, const GraphCalendarHints &calendar_hints,
    const sm_layout::SegmentOffsets &device_segments, void *uploaded_sm
) {
    release_scheduler_state(runtime);
    if (total_tasks < 0 || task_window_size == 0 || static_cast<uint64_t>(total_tasks) > task_window_size) {
        LOG_ERROR(
            "A5 HBG AICore scheduler: invalid graph size tasks=%d window=%" PRIu64, total_tasks, task_window_size
        );
        return false;
    }
    const int32_t worker_count = runtime->get_worker_count();
    if (worker_count <= 0 || worker_count > static_cast<int32_t>(SCHEDULER_WORKER_CAPACITY) ||
        worker_count % PLATFORM_CORES_PER_BLOCKDIM != 0) {
        LOG_ERROR("A5 HBG AICore scheduler: invalid worker count %d for complete clusters", worker_count);
        return false;
    }
    std::vector<int64_t> inline_completed_task_refs;
    std::vector<SchedulerTaskMetadata> task_metadata(static_cast<size_t>(total_tasks));
    uint64_t classified = 0;
    // The progress byte is PENDING or COMPLETED throughout this walk, never
    // PUBLISHED: that value is stored only by the device dispatch path, which
    // has not run yet, and orch::prepare_task resets every claimed slot this
    // bind. So `is_completed` and its negation partition the tasks here, and a
    // future host-side publication would silently change what these tests mean.
    for (int64_t task_ref = 0; task_ref < total_tasks; ++task_ref) {
        ChipTaskSlotState &slot = host_sm_handle.header->tasks.get_slot_state_by_task_id(task_ref);
        if (slot.task_kind == TaskKind::GRAPH) {
            ++classified;
            continue;
        }
        const auto &packed = task_image::task_at(uploaded_sm, device_segments, task_ref);
        const auto &payload = packed.payload;
        bool inline_dispatch_task = false;
        if (packed.task.task_id != TaskId::make_global(task_ref) || payload.fanin_count < 0 ||
            payload.fanin_count > CHIP_MAX_FANIN || (payload.fanin_count > 0 && payload.fanin_data() == nullptr)) {
            LOG_ERROR("A5 HBG calendar: invalid task or fanin count at ref=%" PRId64, task_ref);
            return false;
        }
        const auto *fanins = payload.fanin_data();
        for (int32_t i = 0; i < payload.fanin_count; ++i) {
            if (fanins[i] < 0 || fanins[i] >= task_ref || std::find(fanins, fanins + i, fanins[i]) != fanins + i) {
                LOG_ERROR("A5 HBG calendar: invalid fanin at task ref=%" PRId64, task_ref);
                return false;
            }
        }
        LaneMask active_mask = slot.active_mask;
        if (slot.active_mask.is_empty()) {
            const bool control_shape = slot.logical_block_num == 1 && slot.total_required_subtasks == 0 &&
                                       !slot.task_attrs.requires_sync_start() && !slot.task_attrs.has_predicate();
            const bool completed = host_sm_handle.header->tasks.is_completed(task_ref);
            if (control_shape && completed && slot.task_attrs.allow_early_resolve()) {
                inline_completed_task_refs.push_back(task_ref);
                continue;
            }
            if (!control_shape || completed) {
                LOG_ERROR("A5 HBG calendar: invalid no-kernel task ref=%" PRId64, task_ref);
                return false;
            }
            inline_dispatch_task = true;
            active_mask = LaneMask::aiv0();
        }
        const uint32_t active_subtasks = static_cast<uint32_t>(active_mask.count());
        if (slot.logical_block_num <= 0) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: task ref=%" PRId64 " has invalid block_num=%d", task_ref,
                slot.logical_block_num
            );
            return false;
        }
        const uint32_t logical_block_num = static_cast<uint32_t>(slot.logical_block_num);
        uint32_t expected_subtasks = 0;
        if (!inline_dispatch_task &&
            !scheduler_subtask_count_checked(logical_block_num, active_subtasks, &expected_subtasks)) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: task ref=%" PRId64 " block/subtask product exceeds the int16 task contract",
                task_ref
            );
            return false;
        }
        if ((!inline_dispatch_task && (!slot.active_mask.is_valid() ||
                                       static_cast<uint32_t>(slot.total_required_subtasks) != expected_subtasks)) ||
            host_sm_handle.header->tasks.is_completed(task_ref) ||
            (slot.task_attrs.has_predicate() && (active_subtasks != 1 || logical_block_num != 1))) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: task ref=%" PRId64
                " has inconsistent shape/state or uses predicate/early-resolve",
                task_ref
            );
            return false;
        }
        for (uint32_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            const bool enabled = slot.active_mask.has(static_cast<SubtaskSlot>(subtask_slot));
            const int32_t kernel_id = packed.task.kernel_id[subtask_slot];
            if ((!enabled && kernel_id != INVALID_KERNEL_ID) ||
                (enabled &&
                 (kernel_id < 0 || kernel_id >= CHIP_MAX_FUNC_ID || runtime->get_function_bin_addr(kernel_id) == 0))) {
                LOG_ERROR(
                    "A5 HBG AICore scheduler: task ref=%" PRId64 " kernel id %d has no registered callable", task_ref,
                    kernel_id
                );
                return false;
            }
        }
        const auto &predicate = payload.predicate;
        if (!scheduler_dispatch_predicate_metadata_valid(
                predicate.addr, predicate.elem_size, static_cast<uint8_t>(predicate.op)
            )) {
            LOG_ERROR("A5 HBG AICore scheduler: task ref=%" PRId64 " has malformed predicate metadata", task_ref);
            return false;
        }
        if (slot.task_attrs.requires_sync_start() && logical_block_num == 1) {
            LOG_ERROR("A5 HBG AICore scheduler: sync-start task ref=%" PRId64 " must have block_num > 1", task_ref);
            return false;
        }
        SchedulerTaskMetadata &metadata = task_metadata[static_cast<size_t>(task_ref)];
        for (uint32_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            metadata.kernel_ids[subtask_slot] = static_cast<uint16_t>(packed.task.kernel_id[subtask_slot]);
        }
        metadata.active_mask = active_mask;
        metadata.inline_task = inline_dispatch_task;
        metadata.logical_block_num = static_cast<uint16_t>(logical_block_num);
        ++classified;
    }

    if (classified + inline_completed_task_refs.size() != static_cast<uint64_t>(total_tasks)) return false;
    return simpler::percounter::publish_calendar_state(
        runtime, api, host_sm_handle, total_tasks, task_metadata, inline_completed_task_refs, calendar_hints,
        device_segments, uploaded_sm
    );
}

struct HostPreparePlan : RetainedHostPrepareState {
    GraphCalendarHints calendar_hints;
};

int32_t commit_host_plan(Runtime *runtime, const HostApi *api, HostPreparePlan &plan) {
    void *host_sm = plan.host_sm;
    const uint64_t task_capacity = plan.task_capacity;
    const uint64_t heap_bytes = plan.heap_bytes;
    const uint64_t image_bytes = plan.image_bytes;
    const int32_t total_tasks = plan.total_tasks;
    const auto &bind_usage = plan.bind_usage;
    const uint64_t upload_bytes = image_bytes;
    const sm_layout::SegmentOffsets device_segments =
        task_image::layout(static_cast<const char *>(host_sm), task_capacity, bind_usage);
    const BindPhaseMark static_arena_phase = bind_phase_begin();
    runtime->dev.sm_image_bytes = image_bytes;
    SharedMemoryHandle host_sm_handle;
    // The populated host mirror retains its reservation pitch across commit retries.
    if (!host_sm_handle.attach_populated(host_sm, plan.sm_size, task_capacity, task_capacity, plan.sm_size))
        return PTO_RUNTIME_ERR_INTERNAL;
    if (const int rc = api->setup_static_arena(heap_bytes, /*gm_sm_size=*/0, upload_bytes); rc != 0) {
        return rc;
    }
    {
        char attrs[kBindAttrsCapacity];
        snprintf(attrs, sizeof(attrs), "heap=%" PRIu64 " arena=%" PRIu64, heap_bytes, upload_bytes);
        record_bind_phase(HostPhaseKind::BindStaticArena, static_arena_phase, attrs);
    }

    const BindPhaseMark sm_phase = bind_phase_begin();
    void *device_arena = api->acquire_pooled_runtime_arena();
    if (device_arena == nullptr) {
        LOG_ERROR("%s", "host-orch: failed to acquire the pooled runtime arena");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    void *device_sm = device_arena;
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
    // task's runtime argument storage must be cache-line aligned — carries to
    // the real base only while the two are congruent: both are aligned to
    // kDefaultBaseAlign, and that covers the storage's own requirement.
    static_assert(
        HEAP_VIRTUAL_BASE % DeviceArena::kDefaultBaseAlign == 0,
        "the virtual heap base must share the committed region's alignment"
    );
    static_assert(
        CHIP_ALIGN_SIZE <= DeviceArena::kDefaultBaseAlign,
        "a sub-task's storage alignment must be covered by the heap region's base alignment"
    );
    always_assert(reinterpret_cast<uint64_t>(gm_heap) % DeviceArena::kDefaultBaseAlign == 0);
    const sm_layout::HeapRebase heap_rebase{reinterpret_cast<uint64_t>(gm_heap), heap_bytes};

    // Task records and the read-only graph share one delivered image. Mutable
    // scheduler storage is allocated and initialized separately.
    constexpr size_t image_alignment = std::max<uint64_t>(CHIP_ALIGN_SIZE, SCHEDULER_STATE_ALIGNMENT);
    static_assert(DeviceArena::kDefaultBaseAlign % image_alignment == 0);
    void *staging_addr = nullptr;
    if (api->acquire_run_image_staging(static_cast<size_t>(upload_bytes), image_alignment, &staging_addr) != 0 ||
        staging_addr == nullptr) {
        LOG_ERROR("host-orch: staging for a %" PRIu64 "-byte runtime image is unavailable", upload_bytes);
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    char *upload_base = static_cast<char *>(staging_addr);

    const uint64_t compacted = task_image::pack(
        static_cast<const char *>(host_sm), task_capacity, bind_usage, heap_rebase, upload_base,
        reinterpret_cast<uint64_t>(device_sm)
    );
    always_assert(compacted == image_bytes);
    if (!create_scheduler_state(
            runtime, api, host_sm_handle, total_tasks, task_capacity, plan.calendar_hints, device_segments, upload_base
        ))
        return PTO_RUNTIME_ERR_INTERNAL;

    // Prepared, not published: `publish_run_image_impl` stages this write for
    // the launch-owned upload and records its host staging cost. The pool counters travel
    // with the record because only this bind knows them; the rest of that
    // segment's attributes the publication rebuilds from `dev` and the length.
    runtime->set_pending_publication(
        device_sm, upload_base, upload_bytes, bind_usage.fanin_elems, bind_usage.tensor_elems, bind_usage.scalar_elems
    );
    return total_tasks;
}

int32_t run_host_orchestration(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, uint64_t sm_size, uint64_t task_capacity,
    void *host_orch_func_ptr, const ChipTaskArgs &orch_l2, HostPreparePlan &plan
) {
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
    orch_ctx.retain_graph_config_id = true;
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
    SharedMemoryHandle host_sm_handle;
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
    GraphImageArena definition_arena{};
    definition_arena.object_prefix_bytes = sizeof(GraphImageHeader);
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

    GraphHostStatePtr graph_state = make_graph_host_state(definition_arena);
    if (graph_state) graph_state->execution_storage_bytes = task_image::graph_storage_bytes;
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
    // Host get/set operates only on explicit HOST/NONE arguments. Uploaded
    // arguments and runtime outputs are DEVICE tensors and cannot be read here.
    const auto *entry_points = reinterpret_cast<const HostOrchEntryPoints *>(host_orch_func_ptr);
    if (entry_points->bind == nullptr || entry_points->finish == nullptr) {
        LOG_ERROR("host-orch: orch .so bind/finish callbacks were not resolved");
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
    // graph_state below, which is a local of this function and dies the same way. The
    // bound context is one of them: orch_ctx is a local of this frame, so the .so's
    // global may not outlive it. Unbinding after finish() keeps the context reachable
    // for whatever the .so still does there.
    RAIIScopeGuard orch_finish([entry_points]() {
        entry_points->finish();
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

    if (!bind_graph_definitions(runtime, api, *graph_state, total_tasks, plan.calendar_hints)) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }

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
    // with the same pitch, which is sound because a task ref is its own slot index:
    // every ref is below total_tasks and indexes the image directly.
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
    const uint64_t image_bytes = task_image::layout(static_cast<const char *>(host_sm), task_capacity, bind_usage).end;
    runtime->dev.sm_image_bytes = image_bytes;

    // Orchestration determines the required span. Bind checks it against the
    // configured fixed capacity before committing any device heap or image.
    // The runtime arena contains only the compact shared-memory image.
    // Each outer Graph heap includes the selected runtime's invocation backing.
    // A graph whose every output is caller-owned allocates nothing, but its tasks
    // still carry the window's base as their zero-length packed buffer, so the
    // region has to exist for that address to be rebasable. Asking for 0 would
    // release it and leave acquire_pooled_gm_heap with nothing to return.
    //
    // Rounded up to the region's base alignment: the committed span then ends on the
    // same boundary it starts on, so an access at the tail of the last packed buffer
    // stays inside the region even when its width exceeds the bytes that buffer
    // asked for.
    const uint64_t heap_bytes = CHIP_ALIGN_UP(
        std::max<uint64_t>(orchestrator.task_allocator.heap_used_bytes(), CHIP_ALIGN_SIZE),
        DeviceArena::kDefaultBaseAlign
    );
    plan.host_sm = host_sm;
    plan.sm_size = sm_size;
    plan.task_capacity = task_capacity;
    plan.heap_bytes = heap_bytes;
    plan.image_bytes = image_bytes;
    plan.total_tasks = total_tasks;
    plan.bind_usage = bind_usage;
    return total_tasks;
}

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
        void *finish_sym = dlsym(handle, "framework_finish_orch");
        if (bind_sym == nullptr || finish_sym == nullptr) {
            LOG_ERROR("host-orch: orch .so must export framework_bind_orch_context and framework_finish_orch");
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
        eps->finish = reinterpret_cast<OrchestrationFinishFunc>(finish_sym);
        out->host_dlopen_handle = handle;
        out->host_orch_func_ptr = eps;
        LOG_INFO("host-orch: loaded orchestration entry '%s' on host", orch_func_name);
    }
    LOG_INFO("Orchestration SO: %zu bytes uploaded", orch_so_size);
    return 0;
}

/**
 * The A5 host_build_graph AICore scheduler dispatches from resolved kernel-entry
 * addresses it reads out of the registration-owned entry table, so registration
 * builds that view alongside the object one.
 */
extern "C" bool runtime_uses_callable_entry_table_impl() { return true; }

/**
 * Per-run binding: build device-side argument storage (tensor copy-out, GM
 * heap, shared memory) and publish it to the runtime. Assumes the
 * callable-side state (kernel binaries, orch SO bytes, func/config names)
 * is already populated by register_callable_impl.
 *
 * Splitting this from register_callable_impl matches the per-callable_id
 * design: register/simpler_run_v2 invokes this every call, while the prep
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
            runtime->clear_prepared_publication();
            host_phase_trace_end(api);
        }
    });

    uint64_t task_capacity = 0;
    if (!resolve_graph_task_capacity(ring_task_window, &task_capacity)) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    const uint64_t heap_override = read_ring_override(ring_heap, 0);
    const uint64_t heap_capacity = heap_override == 0 ? 256ULL * 1024 * 1024 : heap_override;
    if (heap_capacity < 1024 || heap_capacity > MAX_HEAP_CAPACITY) {
        LOG_ERROR("HBG heap capacity must be in [1024, %" PRIu64 "] bytes", uint64_t{MAX_HEAP_CAPACITY});
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
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

    // A lease recorded by an earlier bind names an offset this bind is about to
    // re-slice, so carrying one over would copy this run's bytes back to that
    // run's host pointer. Every exit path from here on leaves the ledger owned
    // by the caller's Runtime, and this is where it starts empty.
    runtime->tensor_leases().clear();

    // No scheduler is selected until this bind selects one. Without this, a bind
    // that fails before it reaches a selection would leave the previous run's
    // mode and its base — naming an allocation this bind may already have freed —
    // in the descriptor the next launch uploads. The handshake words the platform
    // zeroes in prepare_launch_shape give the same guarantee for their half.
    runtime->publish_scheduler_bootstrap(0, 0);

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

    const uint64_t sm_size = SharedMemoryHandle::calculate_size(task_capacity);
    auto plan = std::make_shared<HostPreparePlan>();
    plan->staging_bytes = required_temp_bytes == 0 ? 0 : required_temp_bytes + RetainedTempBump::kAlignment - 1;
    size_t staging_capacity = 0;
    api->get_retained_temp_buffer(&plan->staging_backing, &staging_capacity);
    runtime->set_orch_args(device_args);

    // Record host tasks into the SM mirror before building their device calendar.
    // register_callable_impl guarantees host_orch_func_ptr is non-null on success
    // (it fails the whole prepare otherwise), so this is an assertion-style
    // guard, not a fallback path.
    if (host_orch_func_ptr == nullptr) {
        LOG_ERROR("host-orch: orchestration entry points were not resolved");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    {
        ChipTaskArgs orch_l2;
        orch_l2.create_from_entry_storage(runtime->get_orch_args());
        int32_t total_tasks = run_host_orchestration(
            runtime, api, tensor_access, sm_size, task_capacity, host_orch_func_ptr, orch_l2, *plan
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

    if (plan->heap_bytes > heap_capacity) {
        LOG_ERROR(
            "HBG heap capacity exceeded: required=%" PRIu64 " configured=%" PRIu64 " bytes", plan->heap_bytes,
            heap_capacity
        );
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    }
    plan->heap_bytes = heap_capacity;

    plan->diagnostic_outcome = dep_gen_host_graph_take(&plan->diagnostic_graph);
    plan->park(*runtime);
    const int commit_rc = resume_host_plan(runtime, api, plan);
    if (commit_rc != 0) return commit_rc;

    LOG_INFO("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

    prepared = true;
    return 0;
}

/**
 * Stage this run's metadata in dependency order for the existing launch owner.
 * A failed stage consumes the pending sources. Device destinations remain
 * owned until the failed-prepare cleanup releases them.
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
    // The platform snapshots the sources before this record dies. Onboard
    // launches carry the image in the RTS packet and queue prerequisites
    // before AICPU entry. Synchronous backends copy all regions.
    std::vector<HostMetadataRegion> metadata;
    metadata.reserve(publication.prerequisites.size() + 1);
    for (const auto &region : publication.prerequisites)
        metadata.push_back({region.device_target, region.source, region.bytes});
    metadata.push_back(
        {publication.device_target, publication.source, publication.bytes, true, publication.mutable_prefix_bytes,
         publication.image_references.data(), publication.image_references.size()}
    );
    const BindPhaseMark staging_phase = bind_phase_begin();
    const int metadata_rc =
        api->stage_metadata_with_references(metadata.data(), metadata.size(), publication.image_references);
    if (metadata_rc != 0) {
        LOG_ERROR("host-orch: launch metadata staging failed: %d", metadata_rc);
        return metadata_rc;
    }
    {
        const uint64_t image_bytes = runtime->dev.sm_image_bytes;
        // Keep the workspace address visible even when the attribute buffer fills.
        char attrs[kBindAttrsCapacity];
        snprintf(
            attrs, sizeof(attrs),
            "dest=%p nt=%" PRIu64 " bytes=%" PRIu64 " copied=0 sm=%" PRIu64 " args=%" PRIu64 "/%" PRIu64 "/%" PRIu64,
            publication.device_target, static_cast<uint64_t>(runtime->dev.host_total_tasks), publication.bytes,
            image_bytes, publication.fanin_elems, publication.tensor_elems, publication.scalar_elems
        );
        record_bind_phase(HostPhaseKind::BindImageStage, staging_phase, attrs, publication.bytes);
    }
    if (auto saved = api->get_prepare_plan()) {
        // Keep diagnostics owned by the plan until publication succeeds. An
        // unpublished abort or failed stage must not strand its graph in TLS.
        const auto plan = std::static_pointer_cast<HostPreparePlan>(saved);
        dep_gen_host_graph_restore_for_export(&plan->diagnostic_graph, plan->diagnostic_outcome);
    }
    runtime->retain_prepared_publication(std::move(publication));
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
 * Release the tensor bindings the bind recorded, and end this run's claim on
 * the scheduler state it published.
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
    runtime->clear_prepared_publication();
    release_run_tensor_leases(runtime, api);
    release_scheduler_state(runtime);
    // The dispatch table is owned by bind_callable_to_runtime, which clears it
    // before replaying the active callable's addresses. The chip-callable device
    // buffer behind those addresses is pool-managed by DeviceRunner (keyed by
    // content hash) and bulk-freed in DeviceRunner::finalize(), so re-running the
    // same callable repeatedly does not re-upload.
    return 0;
}

extern "C" bool runtime_has_completion_feedback_impl(const Runtime *runtime) {
    const auto publication = runtime->prepared_publication();
    return publication && static_cast<bool>(publication->on_execution_complete);
}

extern "C" void runtime_collect_completion_feedback_impl(const Runtime *runtime, const HostApi *api) {
    const auto publication = runtime->prepared_publication();
    if (publication && publication->on_execution_complete) publication->on_execution_complete(*api);
}

/**
 * Inspect one run's results.
 *
 * This function:
 * 1. Reports the observed run terminal status when device execution failed
 * 2. Copies written tensors from device back to host
 *
 * It releases nothing; `release_run_bindings_impl` ends the bindings it reads.
 *
 * @param runtime       Pointer to Runtime
 * @param execution_rc  Device-runner drain status after successful enqueue,
 *                      or enqueue status on failure
 * @param launched      Nonzero when this run reached a stream and may have an
 *                      observed terminal status
 * @return 0 on success, -1 on failure
 */
extern "C" int copy_back_run_outputs_impl(const Runtime *runtime, const HostApi *api, int execution_rc, int launched) {
    if (runtime == nullptr && execution_rc == 0) {
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
    const TensorLease *tensor_leases = runtime != nullptr ? runtime->tensor_leases().data() : nullptr;
    int tensor_lease_count = runtime != nullptr ? static_cast<int>(runtime->tensor_leases().size()) : 0;

    LOG_INFO("ChipTensor leases to process: %d", tensor_lease_count);

    bool skip_tensor_copy_back = execution_rc != 0;
    // The shared header may belong to a successor; status belongs to the
    // launch-owned terminal record observed for this run.
    const int32_t runtime_status = execution_rc != 0 && launched != 0 ? api->observed_run_status().value_or(0) : 0;
    if (runtime_status != 0) {
        const char *name = runtime_status < 0 && runtime_status >= -PTO_RUNTIME_LATCHED_CODE_MAX ?
                               error_name(-runtime_status) :
                               "unknown";
        LOG_ERROR("Observed runtime status: %d (shared scheduler header not read) error=%s", runtime_status, name);
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

    return rc;
}

// host_build_graph resolves orchestration on the host, but the A5 platform
// exports the topology query entry used before the first normal launch.
extern "C" const char *const *runtime_extra_aicpu_symbols(size_t *count) {
    static const char *const kExtra[] = {"simpler_aicpu_query_topology"};
    if (count != nullptr) {
        *count = sizeof(kExtra) / sizeof(kExtra[0]);
    }
    return kExtra;
}
