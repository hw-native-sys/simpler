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
 * Runtime Builder - rt2 Implementation (host_build_graph: Host Orchestration)
 *
 * Provides init_runtime_impl and validate_runtime_impl functions for rt2 runtime.
 * The HOST runs the orchestrator to completion, populates shared memory + the
 * prebuilt arena, and H2Ds the image; the device boots scheduler-only.
 *
 * init_runtime_impl:
 *   - Converts host tensor pointers to device pointers (all inputs copied H2D;
 *     only OUTPUT/INOUT tensors are copied back D2H)
 *   - dlopens the orchestration SO on the host and runs it to build the graph
 *   - Sets up runtime state for host orchestration
 *
 * validate_runtime_impl:
 *   - Copies OUTPUT/INOUT tensors back from device to host (read-only inputs
 *     are skipped)
 *   - Frees device memory
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
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <unordered_map>
#include <vector>

#include "assert_compat.h"
#include "host_build_graph/runtime_status.h"
#include "host_build_graph/dep_gen_host_graph.h"
#include "host_build_graph/graph_execution.h"
#include "scheduler/scheduler_graph.h"
#include "scheduler/scheduler_types.h"
#include "host_build_graph/host_tensor_access.h"
#include "host_build_graph/graph_host_state.h"
#include "host_build_graph/host_graph_build.h"
#include "host_build_graph/graph_definition_pack.h"
#include "host_build_graph/host_phase_trace.h"
#include "host_build_graph/orchestrator.h"
#include "host_build_graph/ready_queue_sizing.h"
#include "host_build_graph/graph_recorder_pool.h"
#include "host_build_graph/runtime_core.h"
#include "host_build_graph/shared_memory.h"
#include "host_build_graph/types.h"
#include "host_build_graph/runtime.h"
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
static_assert(
    SCHEDULER_PROFILING_TASK_TIMING_LEVEL == static_cast<uint64_t>(ChipSwimlaneLevel::TASK_TIMING) &&
        SCHEDULER_PROFILING_SCHEDULE_TIMING_LEVEL == static_cast<uint64_t>(ChipSwimlaneLevel::SCHEDULE_TIMING) &&
        SCHEDULER_PROFILING_SCHED_PHASES_LEVEL == static_cast<uint64_t>(ChipSwimlaneLevel::SCHED_PHASES),
    "AICore Scheduler profiling levels must match the chip-swimlane contract"
);

extern "C" int build_kernel_pipeline_contract_impl(const CallConfig *, PipelineContract *) {
    return PTO_RUNTIME_ERR_UNSUPPORTED;
}

extern "C" const PipelineContract *get_pipeline_contract(void) {
    // Host orchestration materializes this run's own graph into the image it
    // uploads, so every device-resident region carries per-run content.
    //
    // PTO_PIPELINE_GM_SM is absent because hbg has no separate shared-memory
    // region: the image is the tail of the runtime-image region, so it shares that
    // region's classification. The arena-topology check skips an absent kind.
    static const PipelineContract contract = {
        PTO_PIPELINE_CONTRACT_ABI_VERSION,
        4,
        2,
        {
            {PTO_PIPELINE_GM_HEAP, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_RUNTIME_IMAGE, PTO_PIPELINE_HOST_PER_RUN, 0},
            {PTO_PIPELINE_AICPU_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
            {PTO_PIPELINE_AICORE_STREAM, PTO_PIPELINE_EXEC_HANDLE, 0},
        },
    };
    return &contract;
}

extern "C" int concurrent_native_prepare_supported_impl(void) {
    // HBG can materialize a complete graph into the lease-selected unpublished
    // arena bank. The common C API keeps collector-bearing configurations on
    // the sequential path until their state is per-epoch.
    return 1;
}

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

    // Any positive count is usable: a task id indexes its slot directly, so
    // nothing masks with this value. The power-of-two, >= 4 requirement belongs to
    // tensormap_and_ringbuffer, which does mask, and is enforced in that runtime's
    // own resolve; neither the RuntimeEnv setter nor Worker.run constrains the
    // value, so this bound is the only one a ring_task_window passes through.
    if (*task_capacity < 1 || *task_capacity > static_cast<uint64_t>(INT32_MAX)) {
        LOG_ERROR("ring_task_window=%" PRIu64 " must be in [1, INT32_MAX]", *task_capacity);
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

static int32_t read_runtime_status(Runtime *runtime, const HostApi *api, SharedMemoryHeader *host_header) {
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

namespace {

struct SchedulerStateOwner {
    void *allocation;
    void *state_base;
    uint64_t allocation_size;
    AicoreSchedulerLayout layout;
    const HostApi *api;
};

std::mutex scheduler_state_owners_mutex;
std::unordered_map<Runtime *, SchedulerStateOwner> scheduler_state_owners;

struct SchedulerJsonRecord {
    uint64_t start_cycles;
    uint64_t end_cycles;
    uint64_t loop_iter;
    const char *kind;
    uint64_t tasks_processed;
    uint64_t task_id;
    bool has_task;
};

const char *scheduler_core_type_name(int32_t core_type) {
    return core_type == static_cast<int32_t>(CoreType::AIC) ? "aic" : "aiv";
}

void append_scheduler_record(
    std::vector<SchedulerJsonRecord> *records, uint64_t start_cycles, uint64_t end_cycles, uint64_t loop_iter,
    const char *kind, uint64_t tasks_processed, uint64_t task_id = 0, bool has_task = true
) {
    if (records == nullptr || start_cycles == 0 || end_cycles < start_cycles) return;
    records->push_back({start_cycles, end_cycles, loop_iter, kind, tasks_processed, task_id, has_task});
}

bool publish_aicore_scheduler_profiling(Runtime *runtime, const HostApi *api) {
    const uint32_t level = api->chip_swimlane_level();
    if (level == 0) return true;

    SchedulerStateOwner owner{};
    {
        std::scoped_lock lock(scheduler_state_owners_mutex);
        auto it = scheduler_state_owners.find(runtime);
        if (it == scheduler_state_owners.end()) return true;
        owner = it->second;
    }

    std::vector<uint8_t> storage(static_cast<size_t>(owner.layout.total_size + SCHEDULER_STATE_ALIGNMENT - 1));
    const uintptr_t aligned_address = (reinterpret_cast<uintptr_t>(storage.data()) + SCHEDULER_STATE_ALIGNMENT - 1) &
                                      ~(static_cast<uintptr_t>(SCHEDULER_STATE_ALIGNMENT) - 1);
    void *host_base = reinterpret_cast<void *>(aligned_address);
    if (api->copy_from_device(host_base, owner.state_base, static_cast<size_t>(owner.layout.total_size)) != 0) {
        LOG_WARN("A5 HBG: failed to copy AICore Scheduler profiling state");
        return false;
    }

    const auto *contexts = scheduler_state_at<SchedulerWorkerContext>(host_base, owner.layout.worker_contexts_offset);
    const auto *traces = scheduler_state_at<SchedulerTaskTrace>(host_base, owner.layout.trace_cells_offset);
    const auto *controls = scheduler_state_at<SchedulerTaskControl>(host_base, owner.layout.task_controls_offset);

    std::ostringstream tasks_json;
    tasks_json << "[";
    bool first_task = true;
    for (uint64_t task_id = 0; task_id < owner.layout.task_count; ++task_id) {
        const SchedulerTaskTrace &trace = traces[task_id];
        if (trace.valid == 0 || trace.kernel_start_cycles == 0 || trace.kernel_end_cycles < trace.kernel_start_cycles ||
            trace.worker_id >= SCHEDULER_WORKER_CAPACITY)
            continue;
        const uint64_t receive_to_start =
            trace.ready_observe_cycles != 0 && trace.kernel_start_cycles >= trace.ready_observe_cycles ?
                trace.kernel_start_cycles - trace.ready_observe_cycles :
                0;
        if (!first_task) tasks_json << ",";
        tasks_json << "\n    [" << trace.worker_id << ", " << task_id << ", " << task_id << ", "
                   << trace.kernel_start_cycles << ", " << trace.kernel_end_cycles << ", " << receive_to_start << "]";
        first_task = false;
    }
    if (!first_task) tasks_json << "\n  ";
    tasks_json << "]";
    const std::string task_payload = tasks_json.str();
    if (!api->publish_chip_swimlane_extension(
            ChipSwimlaneExtensionSection::AicoreTasks, task_payload.c_str(), task_payload.size()
        )) {
        LOG_WARN("A5 HBG: failed to publish AICore task records");
        return false;
    }

    if (level >= static_cast<uint32_t>(ChipSwimlaneLevel::SCHEDULE_TIMING)) {
        std::ostringstream scheduler_tasks_json;
        scheduler_tasks_json << "{\n    \"schema_version\": 1,\n    \"producer\": \"aicore\",\n    \"records\": [";
        bool first_scheduler_task = true;
        for (uint64_t task_id = 0; task_id < owner.layout.task_count; ++task_id) {
            const SchedulerTaskTrace &trace = traces[task_id];
            if (trace.valid == 0 || trace.kernel_start_cycles == 0 ||
                trace.kernel_end_cycles < trace.kernel_start_cycles || trace.worker_id >= SCHEDULER_WORKER_CAPACITY)
                continue;
            if (trace.dispatch_end_cycles == 0 || trace.complete_start_cycles < trace.kernel_end_cycles) {
                LOG_WARN("A5 HBG: incomplete Scheduler task timing for task id=%" PRIu64, task_id);
                return false;
            }
            if (!first_scheduler_task) scheduler_tasks_json << ",";
            scheduler_tasks_json << "\n      [" << trace.worker_id << ", " << task_id << ", "
                                 << trace.dispatch_end_cycles << ", " << trace.complete_start_cycles << "]";
            first_scheduler_task = false;
        }
        if (!first_scheduler_task) scheduler_tasks_json << "\n    ";
        scheduler_tasks_json << "]\n  }";
        const std::string scheduler_tasks_payload = scheduler_tasks_json.str();
        if (!api->publish_chip_swimlane_extension(
                ChipSwimlaneExtensionSection::SchedulerTasks, scheduler_tasks_payload.c_str(),
                scheduler_tasks_payload.size()
            )) {
            LOG_WARN("A5 HBG: failed to publish AICore Scheduler task timing");
            return false;
        }

        const auto *lifecycle =
            scheduler_state_at<AicpuThreadLifecycleTrace>(host_base, owner.layout.aicpu_lifecycle_traces_offset);
        std::ostringstream lifecycle_json;
        lifecycle_json << "[";
        bool first = true;
        for (uint64_t thread = 0; thread < PLATFORM_MAX_AICPU_THREADS; ++thread) {
            const AicpuThreadLifecycleTrace &trace = lifecycle[thread];
            if (trace.handshake_start_cycles == 0) continue;
            if (!first) lifecycle_json << ",";
            lifecycle_json << "\n    {\"aicpu_thread_id\": " << trace.aicpu_thread_id
                           << ", \"handshake_start_cycles\": " << trace.handshake_start_cycles
                           << ", \"handshake_complete_cycles\": " << trace.handshake_complete_cycles
                           << ", \"config_start_cycles\": " << trace.config_start_cycles
                           << ", \"topology_complete_cycles\": " << trace.topology_complete_cycles
                           << ", \"context_publish_start_cycles\": " << trace.context_publish_start_cycles
                           << ", \"context_publish_complete_cycles\": " << trace.context_publish_complete_cycles
                           << ", \"bootstrap_wait_start_cycles\": " << trace.bootstrap_wait_start_cycles
                           << ", \"bootstrap_complete_cycles\": " << trace.bootstrap_complete_cycles
                           << ", \"register_release_start_cycles\": " << trace.register_release_start_cycles
                           << ", \"register_release_end_cycles\": " << trace.register_release_end_cycles
                           << ", \"exit_signal_start_cycles\": " << trace.exit_signal_start_cycles
                           << ", \"exit_signal_end_cycles\": " << trace.exit_signal_end_cycles
                           << ", \"exit_wait_start_cycles\": " << trace.exit_wait_start_cycles
                           << ", \"exit_wait_end_cycles\": " << trace.exit_wait_end_cycles << "}";
            first = false;
        }
        if (!first) lifecycle_json << "\n  ";
        lifecycle_json << "]";
        const std::string lifecycle_payload = lifecycle_json.str();
        if (!api->publish_chip_swimlane_extension(
                ChipSwimlaneExtensionSection::AicpuLifecycleRecords, lifecycle_payload.c_str(), lifecycle_payload.size()
            )) {
            LOG_WARN("A5 HBG: failed to publish AICPU lifecycle records");
            return false;
        }
    }

    if (level < static_cast<uint32_t>(ChipSwimlaneLevel::SCHED_PHASES)) return true;

    std::vector<std::vector<SchedulerJsonRecord>> records(SCHEDULER_WORKER_CAPACITY);
    for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        const SchedulerWorkerContext &context = contexts[worker];
        if (context.is_scheduler == 0 || context.worker_index >= SCHEDULER_WORKER_CAPACITY) continue;
        append_scheduler_record(
            &records[context.worker_index], context.bootstrap_start_cycles, context.target_bootstrap_end_cycles, 0,
            "bootstrap", context.bootstrap_task_count, 0, false
        );
    }
    for (uint64_t task_id = 0; task_id < owner.layout.task_count; ++task_id) {
        const SchedulerTaskTrace &trace = traces[task_id];
        if (trace.state_probe_scheduler_worker_id < SCHEDULER_WORKER_CAPACITY) {
            append_scheduler_record(
                &records[trace.state_probe_scheduler_worker_id], trace.state_probe_start_cycles,
                trace.state_probe_end_cycles, trace.dispatch_loop_iter, "state_probe", 1, task_id
            );
        }
        const auto ready_source = static_cast<SchedulerReadySource>(trace.ready_source);
        const auto publication_mode = static_cast<SchedulerPublicationMode>(trace.publication_mode);
        const bool refill = publication_mode == SchedulerPublicationMode::REFILL;
        if (trace.dispatch_scheduler_worker_id < SCHEDULER_WORKER_CAPACITY && !refill) {
            append_scheduler_record(
                &records[trace.dispatch_scheduler_worker_id], trace.dispatch_start_cycles, trace.dispatch_end_cycles,
                trace.dispatch_loop_iter, ready_source == SchedulerReadySource::STOLEN ? "worksteal" : "dispatch", 1,
                task_id
            );
        }
        if (trace.complete_scheduler_worker_id < SCHEDULER_WORKER_CAPACITY) {
            append_scheduler_record(
                &records[trace.complete_scheduler_worker_id], trace.complete_start_cycles, trace.complete_end_cycles,
                trace.complete_loop_iter, "complete", 1, task_id
            );
        }
        if (trace.refill_scheduler_worker_id < SCHEDULER_WORKER_CAPACITY) {
            append_scheduler_record(
                &records[trace.refill_scheduler_worker_id], trace.refill_start_cycles, trace.refill_end_cycles,
                trace.refill_loop_iter, "refill", 1, trace.refill_task_id
            );
        }
        const SchedulerTaskControl &control = controls[task_id];
        if (control.scheduler_worker_id < SCHEDULER_WORKER_CAPACITY) {
            append_scheduler_record(
                &records[control.scheduler_worker_id], control.completion_resolve_start_cycles,
                control.completion_resolve_end_cycles, control.completion_resolve_loop_iter, "resolve", 1, task_id
            );
        }
    }

    std::vector<uint32_t> dropped_by_worker(SCHEDULER_WORKER_CAPACITY);
    if (owner.layout.activity_buffers_offset != 0) {
        const auto *activity =
            scheduler_state_at<SchedulerActivityBuffer>(host_base, owner.layout.activity_buffers_offset);
        for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
            const SchedulerWorkerContext &context = contexts[worker];
            if (context.is_scheduler == 0) continue;
            if (context.scheduler_index >= SCHEDULER_CLUSTER_CAPACITY) {
                LOG_WARN(
                    "A5 HBG: invalid Scheduler activity index for worker=%" PRIu64 ": %" PRIu64, worker,
                    context.scheduler_index
                );
                return false;
            }
            const SchedulerActivityBuffer &buffer = activity[context.scheduler_index];
            const uint32_t committed = buffer.committed;
            if (!scheduler_activity_record_count_valid(committed)) {
                LOG_WARN("A5 HBG: invalid Scheduler activity count for worker=%" PRIu64 ": %u", worker, committed);
                return false;
            }
            for (uint32_t index = 0; index < committed; ++index) {
                const SchedulerIdleRecord &record = buffer.records[index];
                append_scheduler_record(
                    &records[worker], record.start_time, record.end_time, record.loop_iter, "idle", 0, 0, false
                );
            }
            dropped_by_worker[worker] = buffer.dropped;
        }
    }

    std::ostringstream scheduler_json;
    scheduler_json << "{\n    \"schema_version\": 1,\n    \"streams\": [";
    bool first_stream = true;
    for (uint64_t worker = 0; worker < SCHEDULER_WORKER_CAPACITY; ++worker) {
        const SchedulerWorkerContext &context = contexts[worker];
        const uint32_t dropped = dropped_by_worker[worker];
        if (context.is_scheduler == 0 || (records[worker].empty() && dropped == 0)) continue;
        std::sort(records[worker].begin(), records[worker].end(), [](const auto &lhs, const auto &rhs) {
            if (lhs.start_cycles != rhs.start_cycles) return lhs.start_cycles < rhs.start_cycles;
            if (lhs.end_cycles != rhs.end_cycles) return lhs.end_cycles < rhs.end_cycles;
            return std::strcmp(lhs.kind, rhs.kind) < 0;
        });
        if (!first_stream) scheduler_json << ",";
        scheduler_json << "\n      {\"platform\": \"a5\", \"runtime\": \"host_build_graph\", "
                          "\"producer\": \"aicore\", \"scheduler_id\": "
                       << context.scheduler_index << ", \"worker_id\": " << worker << ", \"core_type\": \""
                       << scheduler_core_type_name(context.core_type)
                       << "\", \"physical_core_id\": " << context.physical_core_id
                       << ", \"capture\": {\"committed\": " << records[worker].size() << ", \"dropped\": " << dropped
                       << ", \"truncated\": " << (dropped == 0 ? "false" : "true") << "}, \"records\": [";
        for (size_t index = 0; index < records[worker].size(); ++index) {
            const SchedulerJsonRecord &record = records[worker][index];
            if (index != 0) scheduler_json << ",";
            scheduler_json << "\n        {\"start_cycles\": " << record.start_cycles
                           << ", \"end_cycles\": " << record.end_cycles << ", \"loop_iter\": " << record.loop_iter
                           << ", \"kind\": \"" << record.kind << "\", \"tasks_processed\": " << record.tasks_processed
                           << ", \"task_id\": ";
            if (record.has_task) scheduler_json << record.task_id;
            else scheduler_json << "null";
            scheduler_json << "}";
        }
        if (!records[worker].empty()) scheduler_json << "\n      ";
        scheduler_json << "], \"metrics\": []}";
        first_stream = false;
    }
    if (!first_stream) scheduler_json << "\n    ";
    scheduler_json << "]\n  }";
    const std::string scheduler_payload = scheduler_json.str();
    if (!api->publish_chip_swimlane_extension(
            ChipSwimlaneExtensionSection::SchedulerRecords, scheduler_payload.c_str(), scheduler_payload.size()
        )) {
        LOG_WARN("A5 HBG: failed to publish AICore Scheduler records");
        return false;
    }
    return true;
}

extern "C" bool publish_runtime_chip_swimlane_extensions(Runtime *runtime) noexcept {
    try {
        if (runtime == nullptr) return true;
        const HostApi *api = nullptr;
        {
            std::scoped_lock lock(scheduler_state_owners_mutex);
            auto it = scheduler_state_owners.find(runtime);
            if (it == scheduler_state_owners.end()) return true;
            api = it->second.api;
        }
        return api != nullptr && publish_aicore_scheduler_profiling(runtime, api);
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

using OrchestrationEntryFunc = void (*)(const ChipTaskArgs &);
using OrchestrationBindFunc = void (*)(RuntimeContext *);
using simpler::hbg::HostOrchEntryPoints;

// host_build_graph host-orch: the orchestrator builds the task graph in a host
// shared-memory mirror, and every cross-task reference it stores is an offset or
// an index from its own block. The bytes the host wrote are therefore the bytes
// the device schedules, with no on-device or pre-copy pointer fixup.

struct GraphHostStateBinding {
    explicit GraphHostStateBinding(OrchestratorState &orchestrator, GraphHostState *state) :
        orchestrator(orchestrator) {
        orchestrator.graph_host_state = state;
    }
    ~GraphHostStateBinding() { orchestrator.graph_host_state = nullptr; }

    OrchestratorState &orchestrator;
};

void release_scheduler_state(Runtime *runtime, const HostApi *api) {
    if (runtime == nullptr || api == nullptr) return;
    SchedulerStateOwner owner{};
    {
        std::scoped_lock lock(scheduler_state_owners_mutex);
        auto it = scheduler_state_owners.find(runtime);
        if (it != scheduler_state_owners.end()) {
            owner = it->second;
            scheduler_state_owners.erase(it);
        }
    }
    if (owner.allocation != nullptr) api->device_free(owner.allocation);
    for (int32_t i = 0; i < runtime->get_worker_count(); ++i) {
        runtime->workers[i].aicpu_ready = 0;
        runtime->workers[i].task = 0;
    }
}

void select_legacy_scheduler(Runtime *runtime, uint32_t mode) {
    always_assert(
        mode == SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH || mode == SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE
    );
    for (int32_t i = 0; i < runtime->get_worker_count(); ++i) {
        runtime->workers[i].aicpu_ready = mode;
        runtime->workers[i].task = 0;
    }
}

bool create_scheduler_state(
    Runtime *runtime, const HostApi *api, SharedMemoryHandle &host_sm_handle, int32_t total_tasks,
    uint64_t task_window_size, const sm_layout::SegmentOffsets &device_segments
) {
    release_scheduler_state(runtime, api);
    if (total_tasks < 0 || task_window_size == 0 || static_cast<uint64_t>(total_tasks) > task_window_size) {
        LOG_ERROR(
            "A5 HBG AICore scheduler: invalid graph size tasks=%d window=%" PRIu64, total_tasks, task_window_size
        );
        return false;
    }
    SchedulerGraphView host_graph{
        reinterpret_cast<uint64_t>(host_sm_handle.header->tasks.task_storage),
        0,
        static_cast<uint64_t>(total_tasks),
        task_window_size - 1,
    };
    for (int64_t task_id = 0; task_id < total_tasks; ++task_id) {
        ChipTaskSlotState &slot = host_sm_handle.header->tasks.get_slot_state_by_task_id(task_id);
        if (slot.task_kind == TaskKind::GRAPH) {
            select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_GRAPH);
            LOG_INFO("A5 HBG: retaining AICPU scheduling for a graph-execution run");
            return true;
        }
    }
    std::vector<int64_t> inline_completed_task_ids;
    std::vector<SchedulerTaskMetadata> task_metadata(static_cast<size_t>(total_tasks));
    uint64_t aic_task_count = 0;
    uint64_t aiv_task_count = 0;
    uint64_t executable_task_count = 0;
    uint64_t executable_subtask_count = 0;
    uint64_t gang_task_count = 0;
    uint64_t aic_worker_demand = 0;
    uint64_t aiv_worker_demand = 0;
    int64_t legacy_shape_task_id = -1;
    // The progress byte is PENDING or COMPLETED throughout this walk, never
    // PUBLISHED: that value is stored only by the device dispatch path, which
    // has not run yet, and orch::prepare_task resets every claimed slot this
    // bind. So `is_completed` and its negation partition the tasks here, and a
    // future host-side publication would silently change what these tests mean.
    for (int64_t task_id = 0; task_id < total_tasks; ++task_id) {
        ChipTaskSlotState &slot = host_sm_handle.header->tasks.get_slot_state_by_task_id(task_id);
        SchedulerTaskShape shape{};
        SchedulerGraphResult status = scheduler_classify_task_shape(host_graph, task_id, &shape);
        bool inline_dispatch_task = false;
        if (status != SchedulerGraphResult::OK) {
            bool inline_completed_task =
                status == SchedulerGraphResult::UNSUPPORTED_SHAPE && slot.active_mask.raw() == 0 &&
                slot.logical_block_num == 1 && slot.total_required_subtasks == 0 &&
                host_sm_handle.header->tasks.is_completed(task_id) && slot.task_attrs.allow_early_resolve() &&
                !slot.task_attrs.requires_sync_start() && !slot.task_attrs.has_predicate();
            if (inline_completed_task) {
                inline_completed_task_ids.push_back(task_id);
                continue;
            }
            inline_dispatch_task = status == SchedulerGraphResult::UNSUPPORTED_SHAPE && slot.active_mask.raw() == 0 &&
                                   slot.logical_block_num == 1 && slot.total_required_subtasks == 0 &&
                                   !host_sm_handle.header->tasks.is_completed(task_id) &&
                                   !slot.task_attrs.requires_sync_start() && !slot.task_attrs.has_predicate();
            if (!inline_dispatch_task) {
                LOG_ERROR(
                    "A5 HBG AICore scheduler: invalid v0 task id=%" PRId64 " status=%" PRIu64, task_id,
                    static_cast<uint64_t>(status)
                );
                return false;
            }
            shape.task_id = task_id;
            shape.active_mask = SUBTASK_MASK_AIV0;
        }
        // Ordinary tasks retain the existing ChipTaskSlotState scheduling
        // semantics. Only the legacy inline-dispatch fallback synthesizes its
        // AIV0 execution shape because it has no active subtask in the slot.
        const uint8_t classified_active_mask = shape.active_mask;
        const ActiveMask metadata_active_mask =
            inline_dispatch_task ? ActiveMask(classified_active_mask) : slot.active_mask;
        const uint8_t active_mask = metadata_active_mask.core_mask();
        const uint32_t active_subtasks = static_cast<uint32_t>(__builtin_popcount(active_mask));
        if (slot.logical_block_num <= 0) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: task id=%" PRId64 " has invalid block_num=%d", task_id, slot.logical_block_num
            );
            return false;
        }
        const uint32_t logical_block_num = static_cast<uint32_t>(slot.logical_block_num);
        if (active_subtasks == 0 || logical_block_num > UINT16_MAX / active_subtasks) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: task id=%" PRId64 " block/subtask product exceeds scheduler state capacity",
                task_id
            );
            return false;
        }
        const uint32_t expected_subtasks = logical_block_num * active_subtasks;
        if ((!inline_dispatch_task && (slot.active_mask.raw() != classified_active_mask ||
                                       static_cast<uint32_t>(slot.total_required_subtasks) != expected_subtasks)) ||
            expected_subtasks > UINT16_MAX || host_sm_handle.header->tasks.is_completed(task_id) ||
            (slot.task_attrs.has_predicate() && (active_subtasks != 1 || logical_block_num != 1))) {
            LOG_ERROR(
                "A5 HBG AICore scheduler: task id=%" PRId64
                " has inconsistent shape/state or uses predicate/early-resolve",
                task_id
            );
            return false;
        }
        for (uint32_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            if ((active_mask & (1U << subtask_slot)) == 0 || inline_dispatch_task) continue;
            const int32_t kernel_id = shape.kernel_ids[subtask_slot];
            if (kernel_id < 0 || kernel_id >= RUNTIME_MAX_FUNC_ID || runtime->get_function_bin_addr(kernel_id) == 0) {
                LOG_ERROR(
                    "A5 HBG AICore scheduler: task id=%" PRId64 " kernel id %d has no registered callable", task_id,
                    kernel_id
                );
                return false;
            }
        }
        __gm__ uint8_t *payload = scheduler_graph_payload(host_graph, task_id);
        const auto *predicate =
            reinterpret_cast<const SchedulerDispatchPredicate *>(payload + SCHEDULER_GRAPH_PREDICATE_OFFSET);
        if (!scheduler_dispatch_predicate_metadata_valid(predicate->addr, predicate->elem_size, predicate->op)) {
            LOG_ERROR("A5 HBG AICore scheduler: task id=%" PRId64 " has malformed predicate metadata", task_id);
            return false;
        }
        int32_t fanin_count = *reinterpret_cast<__gm__ int32_t *>(payload + SCHEDULER_GRAPH_FANIN_COUNT_OFFSET);
        if (fanin_count < 0) {
            LOG_ERROR("A5 HBG AICore scheduler: task id=%" PRId64 " has invalid fanin count %d", task_id, fanin_count);
            return false;
        }
        if (slot.task_attrs.requires_sync_start() && logical_block_num == 1) {
            LOG_ERROR("A5 HBG AICore scheduler: sync-start task id=%" PRId64 " must have block_num > 1", task_id);
            return false;
        }
        if (!scheduler_resident_v0_task_shape_supported(
                active_subtasks, logical_block_num, slot.task_attrs.requires_sync_start()
            )) {
            if (legacy_shape_task_id < 0) legacy_shape_task_id = task_id;
        }
        SchedulerTaskMetadata &metadata = task_metadata[static_cast<size_t>(task_id)];
        for (uint32_t subtask_slot = 0; subtask_slot < 3; ++subtask_slot) {
            metadata.kernel_ids[subtask_slot] = (active_mask & (1U << subtask_slot)) != 0 ?
                                                    static_cast<uint16_t>(shape.kernel_ids[subtask_slot]) :
                                                    UINT16_MAX;
        }
        metadata.active_mask = active_mask;
        metadata.flags = scheduler_task_metadata_flags_from_submit_state(
            metadata_active_mask, slot.task_attrs, slot.logical_block_num, fanin_count != 0, inline_dispatch_task
        );
        metadata.logical_block_num = static_cast<uint16_t>(logical_block_num);
        metadata.total_required_subtasks = static_cast<uint16_t>(expected_subtasks);
        metadata.timing_slot = slot.task_attrs.timing_slot();
        if ((active_mask & 1U) != 0) {
            ++aic_task_count;
            aic_worker_demand = std::max<uint64_t>(aic_worker_demand, logical_block_num);
        }
        const uint32_t active_aiv_subtasks = active_subtasks - ((active_mask & 1U) != 0 ? 1U : 0U);
        if (active_aiv_subtasks != 0) {
            ++aiv_task_count;
            aiv_worker_demand = std::max<uint64_t>(aiv_worker_demand, logical_block_num * active_aiv_subtasks);
        }
        if (scheduler_task_is_gang(metadata.flags)) ++gang_task_count;
        executable_subtask_count += expected_subtasks;
        ++executable_task_count;
    }

    if (executable_task_count + inline_completed_task_ids.size() != static_cast<uint64_t>(total_tasks)) {
        LOG_ERROR("A5 HBG AICore scheduler: task metadata does not cover the graph");
        return false;
    }
    if (legacy_shape_task_id >= 0) {
        select_legacy_scheduler(runtime, SCHEDULER_RUNTIME_MODE_LEGACY_UNSUPPORTED_SHAPE);
        LOG_INFO(
            "A5 HBG: retaining AICPU scheduling for task id=%" PRId64 " with MIX, SPMD, or sync-start shape",
            legacy_shape_task_id
        );
        return true;
    }

    AicoreSchedulerLayout layout{};
    if (!scheduler_plan_layout(
            static_cast<uint64_t>(total_tasks), aic_task_count, aiv_task_count, &layout,
            api->chip_swimlane_level() >= static_cast<uint32_t>(ChipSwimlaneLevel::SCHED_PHASES)
        ) ||
        layout.total_size > std::numeric_limits<uint64_t>::max() - (SCHEDULER_STATE_ALIGNMENT - 1)) {
        LOG_ERROR("A5 HBG AICore scheduler: scheduler state layout overflow");
        return false;
    }
    layout.executable_task_count = executable_task_count;
    layout.executable_subtask_count = executable_subtask_count;
    layout.gang_task_count = gang_task_count;
    layout.aic_worker_demand = aic_worker_demand;
    layout.aiv_worker_demand = aiv_worker_demand;

    const uint64_t allocation_size = layout.total_size + SCHEDULER_STATE_ALIGNMENT - 1;
    void *allocation = api->device_malloc(static_cast<size_t>(allocation_size));
    if (allocation == nullptr) {
        LOG_ERROR("A5 HBG AICore scheduler: failed to allocate %" PRIu64 " scheduler state bytes", allocation_size);
        return false;
    }
    const uintptr_t aligned_address = (reinterpret_cast<uintptr_t>(allocation) + SCHEDULER_STATE_ALIGNMENT - 1) &
                                      ~(static_cast<uintptr_t>(SCHEDULER_STATE_ALIGNMENT) - 1);

    std::vector<uint8_t> storage(static_cast<size_t>(allocation_size));
    const uintptr_t host_aligned_address =
        (reinterpret_cast<uintptr_t>(storage.data()) + SCHEDULER_STATE_ALIGNMENT - 1) &
        ~(static_cast<uintptr_t>(SCHEDULER_STATE_ALIGNMENT) - 1);
    void *host_base = reinterpret_cast<void *>(host_aligned_address);
    if (!scheduler_init_data_from_layout(host_base, layout)) {
        api->device_free(allocation);
        LOG_ERROR("A5 HBG AICore scheduler: failed to initialize scheduler state");
        return false;
    }

    auto *task_controls = scheduler_state_at<SchedulerTaskControl>(host_base, layout.task_controls_offset);
    for (int64_t task_id : inline_completed_task_ids) {
        task_controls[task_id].state = static_cast<int64_t>(SchedulerTaskState::DONE);
        task_controls[task_id].wake_list_head = SCHEDULER_WAKE_LIST_CLOSED;
    }
    static_assert(
        SCHEDULER_CALLABLE_CAPACITY == RUNTIME_MAX_FUNC_ID,
        "scheduler state callable table must cover the runtime table"
    );
    auto *callable_addresses = scheduler_state_at<uint64_t>(host_base, layout.callable_addresses_offset);
    const bool cpu_sim = std::strcmp(get_platform(), "a5sim") == 0;
    for (uint32_t func_id = 0; func_id < SCHEDULER_CALLABLE_CAPACITY; ++func_id) {
        const uint64_t callable_address = runtime->get_function_bin_addr(static_cast<int32_t>(func_id));
        callable_addresses[func_id] =
            callable_address == 0 ?
                0 :
                (cpu_sim ? reinterpret_cast<const CoreCallable *>(callable_address)->resolved_addr() :
                           callable_address + CoreCallable::binary_data_offset());
    }
    auto *metadata = scheduler_state_at<SchedulerTaskMetadata>(host_base, layout.task_metadata_offset);
    std::copy(task_metadata.begin(), task_metadata.end(), metadata);

    const uint64_t device_sm_address = reinterpret_cast<uint64_t>(runtime->get_gm_sm_ptr());
    auto *run_control = scheduler_state_at<SchedulerRunControl>(host_base, layout.run_control_offset);
    run_control->expected_task_count = static_cast<uint64_t>(total_tasks);
    run_control->inline_completed_count = inline_completed_task_ids.size();
    run_control->aic_task_count = aic_task_count;
    run_control->aiv_task_count = aiv_task_count;
    run_control->aic_worker_demand = aic_worker_demand;
    run_control->aiv_worker_demand = aiv_worker_demand;
    run_control->chip_swimlane_level = api->chip_swimlane_level();
    run_control->dispatch_payloads_offset = layout.dispatch_payloads_offset;
    run_control->task_metadata_offset = layout.task_metadata_offset;
    run_control->ready_inboxes_offset = layout.ready_inboxes_offset;
    run_control->ready_directory_offset = layout.ready_directory_offset;
    run_control->gang_coordinator_offset = layout.gang_coordinator_offset;
    run_control->gang_cohorts_offset = layout.gang_cohorts_offset;
    run_control->scheduler_timeout_cycles = SCHEDULER_TIMEOUT_CYCLES;
    run_control->error_task_id = UINT64_MAX;
    run_control->error_core_id = UINT64_MAX;
    run_control->error_core_type = UINT64_MAX;
    auto *gang_coordinator = scheduler_state_at<SchedulerGangCoordinator>(host_base, layout.gang_coordinator_offset);
    gang_coordinator->gang_task_count = gang_task_count;

    auto *contexts = scheduler_state_at<SchedulerWorkerContext>(host_base, layout.worker_contexts_offset);
    int32_t aic_rank = 0;
    int32_t aiv_rank = 0;
    for (int32_t i = 0; i < runtime->get_worker_count(); ++i) {
        SchedulerWorkerContext &context = contexts[i];
        context.core_type = static_cast<int32_t>(runtime->workers[i].core_type);
        context.physical_core_id = -1;
        context.type_rank = context.core_type == static_cast<int32_t>(CoreType::AIC) ? aic_rank++ : aiv_rank++;
        context.active = 0;
        context.run_control_offset = layout.run_control_offset;
        context.task_controls_offset = layout.task_controls_offset;
        context.completion_inboxes_offset = layout.completion_inboxes_offset;
        context.task_metadata_offset = layout.task_metadata_offset;
        context.aicpu_lifecycle_traces_offset = layout.aicpu_lifecycle_traces_offset;
        context.ready_inboxes_offset = layout.ready_inboxes_offset;
        context.ready_owner_states_offset = layout.ready_owner_states_offset;
        context.ready_directory_offset = layout.ready_directory_offset;
        context.worker_contexts_offset = layout.worker_contexts_offset;
        context.dispatch_slots_offset = layout.dispatch_slots_offset;
        context.callable_addresses_offset = layout.callable_addresses_offset;
        context.runtime_worker_count = static_cast<uint64_t>(runtime->get_worker_count());
        context.bootstrap_done = 0;
        context.gang_coordinator_offset = layout.gang_coordinator_offset;
        context.gang_cohorts_offset = layout.gang_cohorts_offset;
        context.gang_participants_offset = layout.gang_participants_offset;
        context.gang_commands_offset = layout.gang_commands_offset;
        context.graph_storage_address = device_sm_address + device_segments.storage;
        context.graph_reserved_address = 0;
        context.scheduler_state_base_address = aligned_address;
        context.dispatch_payload_offset = layout.dispatch_payloads_offset + static_cast<uint64_t>(i) *
                                                                                SCHEDULER_PENDING_SLOT_COUNT *
                                                                                sizeof(DispatchPayload);
        context.trace_cells_offset = layout.trace_cells_offset;
        context.task_window_last_index = task_window_size - 1;
        context.graph_task_count = static_cast<uint64_t>(total_tasks);
        context.worker_index = static_cast<uint64_t>(i);
    }

    if (api->copy_to_device(
            reinterpret_cast<void *>(aligned_address), host_base, static_cast<size_t>(layout.total_size)
        ) != 0) {
        api->device_free(allocation);
        LOG_ERROR("A5 HBG AICore scheduler: failed to publish scheduler state");
        return false;
    }
    for (int32_t i = 0; i < runtime->get_worker_count(); ++i) {
        runtime->workers[i].aicpu_ready = SCHEDULER_RUNTIME_MODE_RESIDENT_PENDING;
        runtime->workers[i].task =
            aligned_address + layout.worker_contexts_offset + static_cast<uint64_t>(i) * sizeof(SchedulerWorkerContext);
    }
    {
        std::scoped_lock lock(scheduler_state_owners_mutex);
        scheduler_state_owners.emplace(
            runtime,
            SchedulerStateOwner{allocation, reinterpret_cast<void *>(aligned_address), allocation_size, layout, api}
        );
    }
    LOG_INFO("A5 HBG: selected AICore Scheduler for %d tasks", total_tasks);
    return true;
}

}  // namespace

int32_t simpler::hbg::build_graph(
    Runtime *runtime, HostTensorAccessor &tensor_access, LeasedWorkspace workspace,
    const HostOrchEntryPoints &entry_points, const ChipTaskArgs &args, GraphBuild &build
) {
    build.build_complete = false;
    build.definitions.reset();
    build.workspace = {};
    build.total_tasks = 0;
    build.ready_queue_populations = {};
    build.ready_queue_capacities = {};
    build.bind_usage = {};
    build.image_bytes = 0;
    build.heap_bytes = 0;
    build.definition_bytes = 0;
    RuntimeContext *rt = workspace.runtime_context;
    void *host_sm = workspace.sm_mirror;
    const uint64_t sm_size = workspace.sm_bytes;
    const uint64_t task_capacity = workspace.task_capacity;
    const auto &definition_arena = workspace.definitions;
    if (!runtime || !rt || !host_sm || !entry_points.entry || !entry_points.bind || task_capacity == 0 ||
        task_capacity > INT32_MAX || sm_size < SharedMemoryHandle::calculate_size(task_capacity) ||
        reinterpret_cast<uintptr_t>(host_sm) % CHIP_ALIGN_SIZE != 0)
        return PTO_RUNTIME_ERR_INTERNAL;
    // The dep_gen graph belongs to the orchestration that is about to run.
    dep_gen_host_graph_begin_capture();

    // Init-on-write: descriptors, payloads, slot_states and task_states are
    // each written per task at submit and read only for [0, total_tasks). Zero
    // only the fixed-size header here; the per-slot segments are initialized in
    // orch::prepare_task and shipped bounded to total_tasks below.
    const sm_layout::SegmentOffsets sm_segs = sm_layout::segment_offsets(task_capacity);
    std::memset(host_sm, 0, sm_segs.storage);

    // Re-point the orchestrator half at the host SM (scheduler keeps device SM).
    // The result owns this state; the runtime binding is active only during build.
    OrchestratorState &orchestrator = build.orchestrator;
    rt->orchestrator = &orchestrator;
    RAIIScopeGuard orchestrator_binding([rt]() {
        rt->orchestrator = nullptr;
    });
    // The graph heap is allocated out of the HEAP_VIRTUAL_BASE window: its device
    // region is committed by upload_for_program_mode at the measured size. compact_live_image
    // moves every address the orchestrator wrote onto the real base before upload.
    if (!orchestrator.init(
            host_sm, reinterpret_cast<void *>(HEAP_VIRTUAL_BASE), HEAP_VIRTUAL_CAPACITY, task_capacity
        )) {
        LOG_ERROR("host-orch: orchestrator init against host SM failed");
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    // Initialize the host SM header (ring flow control) so submit_task can run.
    SharedMemoryHandle &host_sm_handle = build.host_sm_handle;
    if (!host_sm_handle.init(host_sm, sm_size, task_capacity)) {
        LOG_ERROR("host-orch: host SM init failed");
        return PTO_RUNTIME_ERR_INTERNAL;
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
    runtime_bind_ops(rt);
    orchestrator.total_cluster_count = block_dim * PLATFORM_AIC_CORES_PER_BLOCKDIM;
    orchestrator.total_aiv_count = block_dim * PLATFORM_AIV_CORES_PER_BLOCKDIM;
    rt->mode = MODE_EXECUTE;
    // get_tensor_data/set_tensor_data resolve buffer.addr through the host
    // views registered at staging time (host_build_graph/host_tensor_access.h),
    // so the host orchestrator can read control tensors (e.g. paged_attention's
    // context_lens/block_table) whether or not the platform maps device memory
    // into the host address space.

    rt->active_callable_hash = reinterpret_cast<uint64_t>(entry_points.entry);
    rt->tensor_access = &tensor_access;
    // Binds the orchestration .so's own framework_current_runtime, which its
    // inline rt_submit_* read. The host library links a same-named copy from
    // orchestration/common.cpp, but nothing outside the .so includes
    // orchestration_api.h, so nothing reads that one — rt_scope_* and
    // rt_orchestration_done take the runtime as an argument.
    entry_points.bind(rt);

    const BindPhaseMark orch_phase = bind_phase_begin();
    rt_scope_begin(rt);
    entry_points.entry(args);
    rt_scope_end(rt);
    rt_orchestration_done(rt);
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

    ReadyQueuePopulations ready_queue_populations{};
    SharedMemoryTaskHeader &tasks = host_sm_handle.header->tasks;
    for (int32_t task_id = 0; task_id < total_tasks; ++task_id) {
        const ChipTaskSlotState &slot = tasks.get_slot_state_by_task_id(task_id);
        ready_queue_populations.add_task(slot.active_mask, slot.task_attrs, slot.task_kind);
    }

    auto definitions = std::make_unique<GraphDefinitionPlan>();
    if (!pack_graph_definitions(*graph_state, definition_arena, *definitions, ready_queue_populations))
        return PTO_RUNTIME_ERR_INTERNAL;
    ReadyQueueCapacities ready_queue_capacities{};
    const int32_t ready_queue_status = derive_ready_queue_capacities(ready_queue_populations, &ready_queue_capacities);
    if (ready_queue_status != 0) {
        LOG_ERROR(
            "host-orch: ready queue reachable population exceeds %" PRIu64 " (ready=%" PRIu64 "/%" PRIu64 "/%" PRIu64
            ", sync=%" PRIu64 "/%" PRIu64 "/%" PRIu64 ", dummy=%" PRIu64 ", graph=%" PRIu64 "/%" PRIu64 ")",
            READY_QUEUE_CAPACITY_LIMIT, ready_queue_populations.ready[0], ready_queue_populations.ready[1],
            ready_queue_populations.ready[2], ready_queue_populations.ready_sync[0],
            ready_queue_populations.ready_sync[1], ready_queue_populations.ready_sync[2], ready_queue_populations.dummy,
            ready_queue_populations.graph_ready, ready_queue_populations.graph_prepare
        );
        LOG_RUNTIME_FAILURE(SIMPLER_ERROR_NONE, SIMPLER_ERROR_READY_QUEUE_OVERFLOW, ready_queue_status);
        return ready_queue_status;
    }
    build.ready_queue_capacities = ready_queue_capacities;
    build.definition_bytes = definitions->bytes;
    build.definitions = std::move(definitions);
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
    const uint64_t heap_bytes = CHIP_ALIGN_UP(
        std::max<uint64_t>(orchestrator.task_allocator.heap_used_bytes(), CHIP_ALIGN_SIZE),
        DeviceArena::kDefaultBaseAlign
    );
    build.workspace = workspace;
    build.total_tasks = total_tasks;
    build.ready_queue_populations = ready_queue_populations;
    build.bind_usage = bind_usage;
    build.image_bytes = image_bytes;
    build.heap_bytes = heap_bytes;
    host_phase_trace_note_submitted(static_cast<uint64_t>(total_tasks));
    build.build_complete = true;
    return total_tasks;
}

int32_t simpler::hbg::get_graph_resource_requirements(
    const GraphBuild &build, const RuntimeArenaLayout &layout, GraphResourceRequirements &requirements
) {
    if (!build.build_complete || !build.graph_state || !build.definitions || build.total_tasks < 0) {
        return PTO_RUNTIME_ERR_INVALID_STATE;
    }
    if (layout.task_capacity != build.workspace.task_capacity ||
        static_cast<uint64_t>(build.total_tasks) > layout.task_capacity ||
        layout.off_copied_begin > layout.off_copied_end || layout.off_copied_end != layout.arena_size ||
        build.heap_bytes == 0 || build.image_bytes == 0 || build.image_bytes > UINT64_MAX - layout.off_copied_end) {
        return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    }

    GraphResourceRequirements next{};
    next.layout = {
        HBG_RUNTIME_LAYOUT_ABI_VERSION, RuntimeArchitecture::A5, layout.task_capacity, layout.arena_size,
        layout.off_copied_begin,        layout.off_copied_end,
    };
    next.gm_heap_bytes = build.heap_bytes;
    next.runtime_image_bytes = layout.off_copied_end + build.image_bytes;
    next.graph_definition_bytes = build.definition_bytes;

    // A5 may replace the AICPU scheduler with per-graph AICore state. Graph
    // execution itself remains on the legacy path, so only a build with no
    // Definition payload needs this extra region. Size an upper bound from H1's
    // task count without walking task records again.
    if (build.definition_bytes == 0) {
        static_assert(SCHEDULER_STATE_ALIGNMENT <= DeviceArena::kDefaultBaseAlign);
        AicoreSchedulerLayout scheduler_layout{};
        const uint64_t tasks = static_cast<uint64_t>(build.total_tasks);
        if (!scheduler_plan_layout(tasks, tasks, tasks, &scheduler_layout) ||
            scheduler_layout.total_size > UINT64_MAX - (SCHEDULER_STATE_ALIGNMENT - 1)) {
            return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
        }
        next.scheduler_state_bytes = scheduler_layout.total_size + SCHEDULER_STATE_ALIGNMENT - 1;
    }

    uint64_t ignored = 0;
    if (!next.logical_device_bytes(ignored)) return PTO_RUNTIME_ERR_INVALID_ARGUMENT;
    requirements = next;
    return 0;
}

int32_t simpler::hbg::upload_for_program_mode(
    Runtime *runtime, const HostApi *api, RuntimeContext *rt, DeviceArena &host_arena, const RuntimeArenaLayout &layout,
    GraphBuild &build
) {
    if (!build.build_complete || rt != build.workspace.runtime_context) return PTO_RUNTIME_ERR_INTERNAL;
    GraphHostStatePtr &graph_state = build.graph_state;
    void *host_sm = build.workspace.sm_mirror;
    const uint64_t task_capacity = build.workspace.task_capacity;
    const int32_t total_tasks = build.total_tasks;
    const ReadyQueueCapacities &ready_queue_capacities = build.ready_queue_capacities;
    // Upload each distinct Definition as its own retained device object and bind
    // every outer Graph task to it. Per-invocation data already lives in that
    // task's payload regions and is copied with the shared-memory image below.
    const BindPhaseMark graph_phase = bind_phase_begin();
    const auto &definitions = *build.definitions;
    if (!upload_graph_definitions(api, build)) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    {
        // `bytes` is what this segment copied: the Definition objects, which are all
        // it copies. `defs` and `submissions` differ by the replay count — one
        // Definition serves every Graph task with its key. `spilled` is how many
        // objects the recorders could not build in the block, and so is 0 for a bind
        // the retained staging was big enough for. It is deliberately not spelled
        // `copied=`, which on arena_h2d means a zone rather than a count.
        char attrs[kBindAttrsCapacity];
        snprintf(
            attrs, sizeof(attrs), "defs=%zu bytes=%zu submissions=%zu spilled=%zu", definitions.objects.size(),
            definitions.bytes, graph_host_upload_count(*graph_state), definitions.spilled
        );
        record_bind_phase(HostPhaseKind::BindGraphUpload, graph_phase, attrs, definitions.bytes);
    }

    rt->prebuilt_layout.sched.capacities = ready_queue_capacities;

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
    const sm_layout::BindUsage &bind_usage = build.bind_usage;
    const uint64_t image_bytes = build.image_bytes;
    runtime->sm_image_bytes = image_bytes;

    // The build measured both sizes. Upload commits the arena up to its
    // shared-memory tail and the graph heap to the
    // bytes orchestration actually handed out. setup_static_arena commits per
    // region and short-circuits a request an existing one already covers, so a
    // repeated workload pays for neither twice and the heap is grow-only across a
    // Worker's binds.
    const BindPhaseMark static_arena_phase = bind_phase_begin();
    // The compact shared-memory image is the only per-run tail in the device
    // arena. GraphExecution is initialized later in each outer Graph heap.
    const uint64_t device_arena_bytes = layout.off_copied_end + image_bytes;
    // A graph whose every output is caller-owned allocates nothing, but its tasks
    // still carry the window's base as their zero-length packed buffer, so the
    // region has to exist for that address to be rebasable. Asking for 0 would
    // release it and leave acquire_pooled_gm_heap with nothing to return.
    //
    // Rounded up to the region's base alignment: the committed span then ends on the
    // same boundary it starts on, so an access at the tail of the last packed buffer
    // stays inside the region even when its width exceeds the bytes that buffer
    // asked for.
    const uint64_t heap_bytes = build.heap_bytes;
    if (api->setup_static_arena(heap_bytes, /*gm_sm_size=*/0, device_arena_bytes) != 0) {
        LOG_ERROR(
            "host-orch: failed to commit %" PRIu64 " bytes of graph heap + %" PRIu64 " bytes of device runtime arena",
            heap_bytes, device_arena_bytes
        );
        return PTO_RUNTIME_ERR_INTERNAL;
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
    // The alignment pack_graph_definitions checked on the virtual base — a Graph
    // task's runtime storage must land on alignof(ChipTaskStorage) — carries to
    // the real base only while the two are congruent: both are aligned to
    // kDefaultBaseAlign, and that covers the storage's own requirement.
    static_assert(
        HEAP_VIRTUAL_BASE % DeviceArena::kDefaultBaseAlign == 0,
        "the virtual heap base must share the committed region's alignment"
    );
    static_assert(
        alignof(ChipTaskStorage) <= DeviceArena::kDefaultBaseAlign,
        "an in-graph task's storage alignment must be covered by the heap region's base alignment"
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
    std::vector<std::byte> storage(upload_bytes + CHIP_ALIGN_SIZE, std::byte{0});
    char *upload_base = reinterpret_cast<char *>(
        (reinterpret_cast<uintptr_t>(storage.data()) + CHIP_ALIGN_SIZE - 1) &
        ~static_cast<uintptr_t>(CHIP_ALIGN_SIZE - 1)
    );

    // The build guard clears orchestrator; the copied image also excludes Host ops.
    rt->ops = nullptr;
    std::memcpy(upload_base, static_cast<const char *>(host_arena.base()) + layout.off_copied_begin, copied_bytes);
    const uint64_t compacted = sm_layout::compact_live_image(
        static_cast<const char *>(host_sm), task_capacity, bind_usage, heap_rebase, upload_base + copied_bytes
    );
    always_assert(compacted == image_bytes);

    const sm_layout::SegmentOffsets device_segments = sm_layout::segment_offsets(sm_layout::image_extents(bind_usage));
    if (!create_scheduler_state(runtime, api, build.host_sm_handle, total_tasks, task_capacity, device_segments)) {
        return PTO_RUNTIME_ERR_INTERNAL;
    }

    const BindPhaseMark h2d_phase = bind_phase_begin();
    if (api->copy_to_device(arena_dev + layout.off_copied_begin, upload_base, upload_bytes) != 0) {
        LOG_ERROR("host-orch: H2D of the runtime image failed");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    {
        // The widest attribute string a segment formats: eight uint64 fields plus
        // their labels. With the counters ahead of it in the recorded string, this
        // is the tail a truncation eats first.
        char attrs[kBindAttrsCapacity];
        snprintf(
            attrs, sizeof(attrs),
            "nt=%" PRIu64 " bytes=%" PRIu64 " copied=%" PRIu64 " sm=%" PRIu64 " args=%" PRIu64 "/%" PRIu64 "/%" PRIu64,
            nt, upload_bytes, copied_bytes, image_bytes, bind_usage.fanin_elems, bind_usage.tensor_elems,
            bind_usage.scalar_elems
        );
        record_bind_phase(HostPhaseKind::BindArenaH2d, h2d_phase, attrs, upload_bytes);
    }
    return total_tasks;
}

namespace {

int32_t run_host_orchestration(
    Runtime *runtime, const HostApi *api, HostTensorAccessor &tensor_access, RuntimeContext *rt,
    DeviceArena &host_arena, const RuntimeArenaLayout &layout, uint64_t sm_size, uint64_t task_capacity,
    void *host_orch_func_ptr, const ChipTaskArgs &orch_l2
) {
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

    simpler::hbg::GraphBuild build;
    const auto &entry_points = *static_cast<const HostOrchEntryPoints *>(host_orch_func_ptr);
    const int32_t status = simpler::hbg::build_graph(
        runtime, tensor_access, {rt, host_sm, sm_size, task_capacity, definition_arena}, entry_points, orch_l2, build
    );
    if (status < 0) return status;
    return simpler::hbg::upload_for_program_mode(runtime, api, rt, host_arena, layout, build);
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
        if (c.func_id < 0 || c.func_id >= RUNTIME_MAX_FUNC_ID) {
            LOG_ERROR("func_id=%d is out of range [0, %d)", c.func_id, RUNTIME_MAX_FUNC_ID);
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
        // The orch .so has its own framework_bind_runtime / g_current_runtime
        // (orchestration/common.cpp is compiled into it); resolve it now so the
        // per-run bind can set it before the .so's inline rt_submit_* run.
        void *bind_sym = dlsym(handle, "framework_bind_runtime");
        if (bind_sym == nullptr) {
            LOG_ERROR("host-orch: orch .so does not export framework_bind_runtime: %s", dlerror());
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
        out->host_dlopen_handle = handle;
        out->host_orch_func_ptr = eps;
        LOG_INFO("host-orch: loaded orchestration entry '%s' on host", orch_func_name);
    }
    LOG_INFO("Orchestration SO: %zu bytes staged", orch_so_size);
    return 0;
}

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
    // host_build_graph host-orch: register_callable_impl resolved the
    // orchestration entry on the host and passed it here as host_orch_func_ptr;
    // it is run below (after the arena is built) against a host SM mirror.
    int tensor_count = orch_args->tensor_count();
    int scalar_count = orch_args->scalar_count();
    LOG_INFO("RT2 bind: %d tensors + %d scalars, host orchestration mode", tensor_count, scalar_count);

    // Arm before the first segment below: the record pool has to exist for
    // `args`, which runs well before the device collector is provisioned. The
    // guard ends the bind on every exit, not just the successful one — a bind
    // that fails part-way is exactly when its breakdown is worth having, and an
    // unfinished bind publishes nothing.
    host_phase_trace_begin(api);
    auto host_phase_guard = RAIIScopeGuard([]() {
        host_phase_trace_end();
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

    // This run's host-view window. The accessor owns every mapping it
    // registers and releases them on every exit path, so no host view outlives
    // the point at which a task could make it stale.
    HostTensorAccessor tensor_access(api);

    const BindPhaseMark args_phase = bind_phase_begin();
    uint64_t staged_bytes = 0;
    int staged_tensors = 0;
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

        void *host_ptr = reinterpret_cast<void *>(static_cast<uintptr_t>(t.buffer.addr));
        size_t size = static_cast<size_t>(t.nbytes());

        void *dev_ptr = api->device_malloc(size);
        if (dev_ptr == nullptr) {
            LOG_ERROR("Failed to allocate device memory for tensor %d", i);
            return PTO_RUNTIME_ERR_INTERNAL;
        }

        // Pure write-only OUTPUT buffers are never read by the kernel and hold
        // no meaningful host content, so they need no device staging — the
        // kernel defines what it writes and any unwritten bytes are undefined.
        // IN / INOUT (read-before-write) are staged H2D.
        bool is_pure_output = (signature != nullptr && i < sig_count && signature[i] == ArgDirection::OUT);
        if (!is_pure_output) {
            int rc = api->copy_to_device(dev_ptr, host_ptr, size);
            if (rc != 0) {
                LOG_ERROR("Failed to stage tensor %d to device", i);
                api->device_free(dev_ptr);
                return PTO_RUNTIME_ERR_INTERNAL;
            }
            staged_bytes += static_cast<uint64_t>(size);
            ++staged_tensors;
        }
        // Read-only INPUT tensors are never written by the kernel, so there is
        // no point copying them back D2H at the end. Index the signature
        // by the orch tensor index `i` (device-space tensors are skipped above
        // but do not consume a separate signature slot — scalars follow the
        // tensor entries). Anything not provably IN keeps the safe default of
        // copying back.
        bool needs_copy_back = !(signature != nullptr && i < sig_count && signature[i] == ArgDirection::IN);
        runtime->tensor_pairs_.push_back({host_ptr, dev_ptr, size, needs_copy_back});
        LOG_DEBUG("  ChipTensor %d: %zu bytes at %p", i, size, dev_ptr);

        // host_build_graph runs the orchestrator on the host, which may read
        // staged control tensors (e.g. paged_attention's context_lens and
        // block_table) via get_tensor_data to shape the graph. A pure output
        // has no valid readable bytes before execution, and a5 cannot map it;
        // exposing its caller buffer would therefore make reads unsafe. Leave
        // it unregistered so both get_tensor_data and set_tensor_data fail
        // closed during orchestration.
        if (!is_pure_output && !tensor_access.add(reinterpret_cast<uint64_t>(dev_ptr), size, host_ptr)) {
            LOG_ERROR("host-orch: no host view for tensor %d (dev_ptr %p, %zu bytes)", i, dev_ptr, size);
            return PTO_RUNTIME_ERR_INTERNAL;
        }

        t.buffer.addr = reinterpret_cast<uint64_t>(dev_ptr);
        always_assert(t.buffer.addr < HEAP_VIRTUAL_BASE && "device_malloc reaches into the virtual heap window");
        device_args.add_tensor(t);
    }
    for (int i = 0; i < scalar_count; i++) {
        device_args.add_scalar(orch_args->scalar(i));
    }
    {
        char attrs[kBindAttrsCapacity];
        snprintf(
            attrs, sizeof(attrs), "ntensor=%d staged=%d bytes=%" PRIu64, tensor_count, staged_tensors, staged_bytes
        );
        record_bind_phase(HostPhaseKind::BindArgs, args_phase, attrs);
    }

    // Lay out the per-Worker static device arena. The GM heap and the prebuilt
    // runtime arena are two independent pooled device allocations, and neither is
    // committed here: the arena's size is known only once orchestration has
    // submitted its tasks, and the heap's only once orchestration has allocated
    // its intermediate buffers. Both are committed by the single
    // setup_static_arena in run_host_orchestration. Owned by DeviceRunner across
    // runs — do NOT record in tensor_pairs_; the free is deferred to
    // DeviceRunner::finalize(). The runtime-arena size is determined by replaying
    // the reserve sequence on a host-side arena.
    uint64_t sm_size = SharedMemoryHandle::calculate_size(task_capacity);

    const BindPhaseMark arena_build_phase = bind_phase_begin();
    DeviceArena host_arena;
    RuntimeArenaLayout layout = runtime_reserve_layout(host_arena, task_capacity);
    if (host_arena.commit(DeviceArena::kDefaultBaseAlign) == nullptr) {
        LOG_ERROR("Failed to commit host arena for prebuilt runtime image");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    {
        char attrs[kBindAttrsCapacity];
        snprintf(attrs, sizeof(attrs), "bytes=%" PRIu64, static_cast<uint64_t>(layout.arena_size));
        record_bind_phase(HostPhaseKind::BindArenaBuild, arena_build_phase, attrs);
    }

    // The shared memory is placed at the end of orchestration, so until then this
    // bind has none. Clearing the pointer keeps a failure before that point from
    // leaving the previous bind's address for the error-code read to follow.
    runtime->set_gm_sm_ptr(nullptr);

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
    // because orchestration is what performs that copy, and the runtime header is
    // part of what travels. The runtime arena's device base does NOT travel — it is
    // on the host Runtime (set_prebuilt_arena below), since the AICPU needs that
    // pointer before it can dereference the image.
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
    {
        ChipTaskArgs orch_l2;
        orch_l2.create_from_entry_storage(runtime->get_orch_args());
        int32_t total_tasks = run_host_orchestration(
            runtime, api, tensor_access, rt, host_arena, layout, sm_size, task_capacity, host_orch_func_ptr, orch_l2
        );
        // The orchestrator is the only host-view reader; from here the device
        // owns these buffers, so drop the window on both exits.
        const size_t view_count = tensor_access.mapping_count();
        const uint64_t view_bytes = tensor_access.mapped_bytes();
        const BindPhaseMark view_close_phase = bind_phase_begin();
        tensor_access.close();
        {
            char attrs[kBindAttrsCapacity];
            snprintf(attrs, sizeof(attrs), "count=%zu bytes=%" PRIu64, view_count, view_bytes);
            record_bind_phase(HostPhaseKind::BindHostViewClose, view_close_phase, attrs);
        }
        if (total_tasks < 0) {
            LOG_ERROR("host-orch: orchestration run failed");
            return total_tasks;
        }
        runtime->host_total_tasks = total_tasks;
        LOG_INFO("host-orch: submitted %d tasks on host", total_tasks);
    }

    // Orchestration is what committed the device region, sized to cover its
    // shared-memory tail, so this is the first point at which its base exists.
    void *runtime_arena_dev = api->acquire_pooled_runtime_arena();
    if (runtime_arena_dev == nullptr) {
        LOG_ERROR("%s", "Failed to acquire the pooled runtime arena after orchestration");
        return PTO_RUNTIME_ERR_INTERNAL;
    }
    runtime->set_prebuilt_arena(runtime_arena_dev, layout.off_runtime);

    LOG_INFO("Device orchestration ready: %d tensors + %d scalars", tensor_count, scalar_count);

    return 0;
}

/**
 * Validate runtime results and cleanup.
 *
 * This function:
 * 1. Copies recorded tensors from device back to host
 * 2. Frees device memory for recorded tensors
 * 3. Clears tensor pair state
 *
 * @param runtime       Pointer to Runtime
 * @param execution_rc  Device-runner drain status after successful enqueue,
 *                      or enqueue status on failure
 * @return 0 on success, -1 on failure
 */
extern "C" int validate_runtime_impl(Runtime *runtime, const HostApi *api, int execution_rc) {
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
    TensorPair *tensor_pairs = runtime->tensor_pairs_.data();
    int tensor_pair_count = static_cast<int>(runtime->tensor_pairs_.size());

    LOG_INFO("ChipTensor pairs to process: %d", tensor_pair_count);

    bool skip_tensor_copy_back = execution_rc != 0;
    int32_t runtime_status = 0;
    SharedMemoryHeader host_header;
    memset(&host_header, 0, sizeof(host_header));

    if (execution_rc != 0) {
        runtime_status = read_runtime_status(runtime, api, &host_header);
    }
    if (runtime_status != 0) {
        int32_t sched_error_code = host_header.sched_error_code.load(std::memory_order_relaxed);
        LOG_RUNTIME_FAILURE(SIMPLER_ERROR_NONE, sched_error_code, runtime_status);
    }

    if (skip_tensor_copy_back) {
        LOG_WARN("Skipping tensor copy-back because execution failed");
    } else {
        for (int i = 0; i < tensor_pair_count; i++) {
            const TensorPair &pair = tensor_pairs[i];

            // Skip if device pointer is null
            if (pair.dev_ptr == nullptr) {
                LOG_WARN("ChipTensor %d has null device pointer, skipping", i);
                continue;
            }

            // If host pointer is null, this is a device-only allocation (no copy-back)
            if (pair.host_ptr == nullptr) {
                LOG_DEBUG("ChipTensor %d: device-only allocation (no copy-back)", i);
                continue;
            }

            // Read-only INPUT tensors were uploaded H2D but the kernel never
            // wrote them — copying them back (potentially ~GB) is pure waste.
            // They are still device_free'd in the cleanup loop below.
            if (!pair.needs_copy_back) {
                LOG_DEBUG("ChipTensor %d: read-only input, skipping copy-back", i);
                continue;
            }

            int copy_rc = api->copy_from_device(pair.host_ptr, pair.dev_ptr, pair.size);
            if (copy_rc != 0) {
                LOG_ERROR("Failed to copy tensor %d from device: %d", i, copy_rc);
                rc = copy_rc;
            } else {
                LOG_DEBUG("ChipTensor %d: %zu bytes copied to host", i, pair.size);
            }
        }
    }

    // Cleanup device tensors
    LOG_INFO("=== Cleaning Up ===");
    for (int i = 0; i < tensor_pair_count; i++) {
        if (tensor_pairs[i].dev_ptr != nullptr) {
            api->device_free(tensor_pairs[i].dev_ptr);
        }
    }
    LOG_INFO("Freed %d device allocations", tensor_pair_count);
    release_scheduler_state(runtime, api);

    // The dispatch table is owned by bind_callable_to_runtime, which clears it
    // before replaying the active callable's addresses. The chip-callable device
    // buffer behind those addresses is pool-managed by DeviceRunner (keyed by
    // content hash) and bulk-freed in DeviceRunner::finalize(), so re-running the
    // same callable repeatedly does not re-upload.

    // Clear tensor pairs
    runtime->tensor_pairs_.clear();

    LOG_INFO("=== Finalize Complete ===");

    if (rc == 0 && runtime_status != 0) {
        rc = runtime_status;
    }

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
