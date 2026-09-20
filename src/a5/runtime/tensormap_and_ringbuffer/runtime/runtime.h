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
 * Runtime Class - Device Execution and Handshake Control
 *
 * This class manages device-side execution through AICPU-AICore handshake
 * protocol. Task graph construction is handled by RuntimeContext; this class
 * only handles:
 * - Handshake buffers for AICPU-AICore communication
 * - Execution parameters (block_dim, aicpu_thread_num)
 * - simpler::tmr::Tensor pair management for host-device memory tracking
 * - Device orchestration state (gm_sm_ptr_, orch_args_)
 * - Function address mapping (func_id_to_addr_)
 *
 * Task dispatch uses a per-core DispatchPayload written by the scheduler.
 * At dispatch time, build_payload() copies tensor pointers and scalars from
 * the task payload into the per-core args[], populates SPMD context, then
 * signals AICore via DATA_MAIN_BASE.
 */

#ifndef SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
#define SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_

#include <stddef.h>  // for offsetof
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include <type_traits>
#include <vector>

#include "common/core_type.h"
#include "common/host_api.h"
#include "common/platform_config.h"
#include "aicpu/platform_aicpu_affinity.h"  // MAX_GATE_THREADS (aicpu_allowed_cpus bound)
#include "dispatch_payload.h"
#include "task_args.h"
#include "tensormap_and_ringbuffer/entry_args.h"  // EntryArgsStorage
#include "utils/tensor_lease.h"

// =============================================================================
// Configuration Macros
// =============================================================================

#define RUNTIME_MAX_WORKER PLATFORM_MAX_CORES  // 36 AIC + 72 AIV cores
#define RUNTIME_MAX_FUNC_ID 1024
#define RUNTIME_MAX_ORCH_SYMBOL_NAME 64

// =============================================================================
// Data Structures
// =============================================================================

/**
 * Handshake Structure - Shared between Host, AICPU, and AICore
 *
 * This structure facilitates communication and synchronization between
 * AICPU and AICore during task execution.
 *
 * Protocol State Machine:
 * 1. AICore publishes physical_core_id, core_type, and aicore_done on launch
 * 2. AICPU publishes the task pointer and opens the register window with DATA_MAIN_BASE=IDLE
 * 3. AICore observes window-open, reports initial idle state, and reads the task pointer
 * 4. Task Dispatch: AICPU writes DATA_MAIN_BASE after updating the per-core payload
 * 5. Task Execution: AICore reads the cached DispatchPayload and executes
 * 6. Task Completion: AICore writes FIN to COND; AICPU observes completion
 * 7. Shutdown: AICPU writes the exit signal to DATA_MAIN_BASE; AICore exits
 *
 * Each AICore instance has its own handshake buffer to enable concurrent
 * task execution across multiple cores.
 */

/**
 * Handshake buffer for AICPU-AICore communication
 *
 * Each AICore has its own handshake buffer for synchronization with AICPU.
 * The structure is cache-line aligned (64 bytes) to prevent false sharing
 * between cores and optimize cache coherency operations.
 *
 * Profiling state lives outside this struct: enablement bits and per-core
 * ring/reg addresses travel through `KernelArgs::enable_profiling_flag` +
 * `KernelArgs::aicore_* per-core address arrays`, which the AICore kernel entry
 * forwards into platform-owned per-core slots
 * (`aicore/aicore_profiling_state.h`). Adding a profiling sub-feature does
 * not require touching this struct anymore.
 *
 * Field Access Patterns:
 * - aicpu_ready: Reserved legacy field; the current handshake does not use it
 * - aicore_done: Written by AICore, read by AICPU (final report; physical_core_id
 *   and core_type are published alongside it in the same write)
 * - task: Written by AICPU before window-open, read by AICore after window-open
 * - core_type: Written by AICore (with aicore_done), read by AICPU
 * - physical_core_id: Written by AICore (with aicore_done), read by AICPU
 */
struct Handshake {
    volatile uint32_t aicpu_ready;  // Legacy layout field; unused by the current handshake
    volatile uint32_t aicore_done;  // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;         // DispatchPayload* published before register window-open
    volatile CoreType core_type;    // Core type: CoreType::AIC or CoreType::AIV (reported by AICore with aicore_done)
    volatile uint32_t physical_core_id;  // Physical core ID (reported by AICore with aicore_done)
    volatile uint64_t report_epoch;      // Commit marker for a native program run's report; 0 = unstamped
} __attribute__((aligned(64)));

// One whole cache line per worker, which is what lets the AICore publish its
// report with a single write-back. The payload offsets are the device-side wire
// contract: AICore writes them and the AICPU sweeps read them back, so a field
// that moved would mis-decode silently. `report_epoch` occupies padding the
// struct already had.
static_assert(sizeof(Handshake) == 64);
static_assert(std::is_standard_layout_v<Handshake> && std::is_trivially_copyable_v<Handshake>);
static_assert(offsetof(Handshake, aicpu_ready) == 0);
static_assert(offsetof(Handshake, aicore_done) == 4);
static_assert(offsetof(Handshake, task) == 8);
static_assert(offsetof(Handshake, core_type) == 16);
static_assert(offsetof(Handshake, physical_core_id) == 20);
static_assert(offsetof(Handshake, report_epoch) == 24);

/**
 * Whether `handshake` carries a report this run may act on.
 *
 * `expected_epoch` is the run's own identity, which the host supplies in
 * `KernelArgs::run_result_epoch` and the AICPU reads back through
 * `get_platform_run_result_epoch()`. A native program run passes a non-zero
 * value and is answered only by a report stamped with exactly that number, so a
 * marker left by an earlier run is rejected rather than mistaken for this one's.
 *
 * A kernel/persistent launch passes 0 and keeps the original predicate: its
 * producer writes no stamp, and its per-run reset is what makes `aicore_done`
 * meaningful. Accepting the epoch here does not order the payload reads that
 * follow — the caller's existing `rmb()` does that.
 */
inline bool aicore_report_accepted(const volatile Handshake *handshake, uint64_t expected_epoch) {
    if (handshake->aicore_done == 0) return false;
    return expected_epoch == 0 || handshake->report_epoch == expected_epoch;
}

// =============================================================================
// Device launch descriptor
// =============================================================================

/**
 * DeviceRuntimeLaunchDesc - the device-copied half of Runtime.
 *
 * This is the ONLY part of Runtime that crosses the host->device boundary: the
 * host fills it, `device_runner_helpers.cpp` rtMemcpy's a prefix from offset 0
 * of the Runtime image, and the AICPU/AICore read these fields back. It is the
 * first member of Runtime (offsetof == 0), so the narrowed copy needs no offset
 * arithmetic.
 *
 * Three lengths, in order. `runtime_device_copy_size` is what a steady-state run
 * re-publishes and stops before `workers`; `runtime_device_initialized_prefix_size`
 * adds `workers`, and is what the first publication onto an allocation sends so
 * the handshake region starts defined. This runtime has no gate tail, so that
 * second length equals `runtime_device_extent_size`.
 *
 * Adding a field here grows the device image; adding a field to Runtime's
 * host-only tail does not. Keep it standard-layout (static_assert below) so the
 * rtMemcpy is well-defined. alignas(64) keeps sizeof a multiple of the cache
 * line so the device-copied image starts and ends on cache-line boundaries and
 * never shares a line with Runtime's host-only tail.
 */
struct alignas(64) DeviceRuntimeLaunchDesc {
    int worker_count;  // Number of active workers

    // Execution parameters for AICPU scheduling.
    //
    // aicpu_thread_num is the *total* AICPU thread count launched on this run
    // (= orch + schedulers). AicpuExecutor splits this into one orchestrator
    // thread (highest idx, runs aicpu_orchestration_entry) and the remaining
    // aicpu_thread_num-1 scheduler threads that dispatch tasks to AICore.
    int aicpu_thread_num;

    // Filter-style affinity gate input (a5 onboard). Host fills before
    // launch from device-side OCCUPY + DSMI CPU_TOPO via
    // pto::a5::compute_allowed_cpus. The on-device gate keeps threads whose
    // sched_getcpu() lands on one of these cpu_ids; exec_idx = position in
    // this array drives sched/orch role assignment. Indices 0..count-2 are
    // scheduler slots, index count-1 is the orchestrator slot. Sized to
    // MAX_GATE_THREADS (the shared gate/ABI bound, ≥ any launch count) for
    // headroom — current policy is 4 sched + 1 orch = 5 active.
    int32_t aicpu_allowed_cpus[MAX_GATE_THREADS];
    int32_t aicpu_allowed_cpu_count;
    // Actual AICPU thread launch count for this run. Host sets from
    // popcount(OCCUPY) via the topology probe. See the matching field in
    // src/common/host_build_graph/runtime.h for rationale.
    int32_t aicpu_launch_count;

    // kernel binary resolution: kernel_id -> GM function_bin_addr mapping
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    // Serial orchestrator -> scheduler start control.
    // When true, scheduler threads wait until orchestration has fully built the
    // task graph before entering resolve_and_dispatch().
    // Controlled via SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE environment variable.
    bool serial_orch_sched;

    void *gm_sm_ptr_;                                   // GM pointer to shared memory (device)
    simpler::tmr::EntryArgsStorage orch_args_storage_;  // Entry args, adopted on the host

    // Prebuilt-arena fast path (trb only). Set by the host before rtMemcpy'ing
    // Runtime to device; AICPU reads them in the boot path to skip
    // runtime_create_from_sm and reuse the pooled, prebuilt arena buffer
    // (already populated by runtime_init_data_from_layout + wire on host).
    void *prebuilt_arena_base_;
    size_t prebuilt_runtime_offset_;

    // Per-callable_id dispatch. AICPU dispatches via
    // `orch_so_table_[active_callable_id_]`.
    int32_t active_callable_id_;

    // Handshake buffers for AICPU-AICore communication, one 64-byte line per
    // worker.
    //
    // Last, and outside the per-run uploaded prefix: every field the device
    // reads here is written on the device — the AICore publishes its report and
    // the AICPU the task pointer it answers with — and no host value is
    // consumed. A steady-state run re-uploads none of it; the first publication
    // onto a given allocation carries it once, which is what gives a fresh
    // block a defined starting value. This runtime has no gate tail, so the
    // initialized prefix ends with this array, at the end of the descriptor.
    Handshake workers[RUNTIME_MAX_WORKER];
};

// =============================================================================
// Runtime Class
// =============================================================================

/**
 * Runtime class for device execution and handshake control
 *
 * This class manages AICPU-AICore communication through handshake buffers.
 * Task graph construction is handled by RuntimeContext; this class only handles
 * execution control and device orchestration state.
 *
 * Layout: the device-read fields live in the first member `dev`
 * (DeviceRuntimeLaunchDesc); everything below it is host-only and is never
 * uploaded. The host/device boundary is therefore the `dev.` prefix, not a
 * fragile field-ordering convention.
 */
class Runtime {
public:
    // The device-copied half. MUST stay the first member: device_runner_helpers
    // copies sizeof(DeviceRuntimeLaunchDesc) bytes from offset 0.
    DeviceRuntimeLaunchDesc dev;

    /**
     * Constructor - zero-initialize all arrays
     */
    Runtime();

    // =========================================================================
    // Accessors for the device-copied fields in `dev`
    //
    // These exist with identical signatures on the host_build_graph Runtime so
    // the shared platform layer (device_runner*.cpp, kernel.cpp) can compile
    // against either variant. trb-only code reads `runtime->dev.X` directly.
    // =========================================================================

    int get_worker_count() const { return dev.worker_count; }
    void set_worker_count(int n) { dev.worker_count = n; }
    int get_aicpu_thread_num() const { return dev.aicpu_thread_num; }
    void set_aicpu_thread_num(int n) { dev.aicpu_thread_num = n; }
    Handshake *get_workers() { return dev.workers; }
    const Handshake *get_workers() const { return dev.workers; }
    int32_t get_aicpu_allowed_cpu_count() const { return dev.aicpu_allowed_cpu_count; }
    void set_aicpu_allowed_cpu_count(int32_t n) { dev.aicpu_allowed_cpu_count = n; }
    int32_t get_aicpu_launch_count() const { return dev.aicpu_launch_count; }
    void set_aicpu_launch_count(int32_t n) { dev.aicpu_launch_count = n; }
    int32_t *get_aicpu_allowed_cpus() { return dev.aicpu_allowed_cpus; }
    size_t aicpu_allowed_cpus_capacity() const {
        return sizeof(dev.aicpu_allowed_cpus) / sizeof(dev.aicpu_allowed_cpus[0]);
    }

    // =========================================================================
    // Performance Profiling
    // =========================================================================

    // =========================================================================
    // Device orchestration (for AICPU thread 3)
    // =========================================================================

    void *get_gm_sm_ptr() const;
    const simpler::tmr::EntryArgsStorage &get_orch_args() const;
    void set_gm_sm_ptr(void *p);
    void set_orch_args(const ChipStorageTaskArgs &args);

    // Prebuilt-arena fast path (trb only). Set by host's
    // bind_callable_to_runtime_impl; consumed by AICPU at boot to attach a
    // DeviceArena to `prebuilt_arena_base_` and pick up the RuntimeContext at
    // `prebuilt_arena_base_ + prebuilt_runtime_offset_`. Both stay zero on
    // first construction (Runtime() ctor zeros them) so a non-prebuilt boot
    // path can still detect "no prebuilt image set" via nullptr.
    void set_prebuilt_arena(void *arena_base, size_t runtime_off);
    void *get_prebuilt_arena_base() const;
    size_t get_prebuilt_runtime_offset() const;

    // Per-callable_id dispatch. callable_id must be in
    // [0, MAX_REGISTERED_CALLABLE_IDS); the AICPU dispatches the orch SO via
    // orch_so_table_[callable_id]. The SO itself is delivered to the AICPU at
    // register time (RegisterCallableArgs), not through Runtime.
    void set_active_callable_id(int32_t callable_id);
    int32_t get_active_callable_id() const;

    uint64_t get_function_bin_addr(int func_id) const;
    /**
     * Replay a previously-uploaded kernel address onto a fresh Runtime.
     * Used by DeviceRunner::bind_callable_to_runtime to rebind prepared
     * kernel binaries onto the runtime before each run.
     */
    void replay_function_bin_addr(int func_id, uint64_t addr);

    /**
     * Drop every func_id -> CoreCallable address mapping.
     *
     * Each mapping points into one callable's retained ChipCallable buffer,
     * which unregistering that callable frees. `bind_callable_to_runtime` calls
     * this before replaying the active callable's addresses so no entry outlives
     * the buffer it points into: the scheduler dereferences these addresses and
     * the AICore calls what it finds there.
     */
    void clear_function_bin_addrs();

    // =========================================================================
    // Host-only state (not copied to device)
    // =========================================================================

    // Host-side tensor ledger for the run's H2D and D2H transfers. Populated by
    // runtime_maker.cpp from orch_args at bind time, iterated by
    // copy_in_run_inputs_impl and copy_back_run_outputs_impl, and released by
    // release_run_bindings_impl. Host-only (after `dev`): never uploaded.
    std::vector<TensorLease> tensor_leases_;

    // The launch shape's AIC/AIV rule, one entry per active worker. Host state
    // with host readers only: the DFX swimlane collector and the sim's per-core
    // thread start need the rule before any core has reported, and
    // `dev.workers[i].core_type` cannot serve them — it is outside the per-run
    // uploaded prefix and the device overwrites it with each core's own report.
    std::vector<CoreType> core_type_rule_;

    /**
     * Record this launch shape's AIC/AIV rule: the first `aic_count` of
     * `worker_count` workers are AIC and the rest AIV.
     *
     * Set wherever `worker_count` is, and read by every host consumer of the
     * rule. Callers must not read `dev.workers[i].core_type` for it — that word
     * belongs to the core's own report.
     */
    void set_core_type_rule(int worker_count, int aic_count) {
        core_type_rule_.assign(static_cast<size_t>(worker_count < 0 ? 0 : worker_count), CoreType::AIV);
        for (size_t i = 0; i < core_type_rule_.size() && i < static_cast<size_t>(aic_count < 0 ? 0 : aic_count); ++i) {
            core_type_rule_[i] = CoreType::AIC;
        }
    }

    /** This launch shape's rule for worker `i`; AIV for an index it never covered. */
    CoreType core_type_rule(int i) const {
        if (i < 0 || static_cast<size_t>(i) >= core_type_rule_.size()) return CoreType::AIV;
        return core_type_rule_[static_cast<size_t>(i)];
    }

    size_t core_type_rule_count() const { return core_type_rule_.size(); }
};

// `dev` must be the first member so the narrowed H2D copy starts at offset 0.
// Runtime is not standard-layout (std::vector member + mixed access), so guard
// the offsetof against -Winvalid-offsetof; the offset itself is well-defined
// for a first member.
#if defined(__GNUC__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Winvalid-offsetof"
#endif
static_assert(offsetof(Runtime, dev) == 0, "DeviceRuntimeLaunchDesc must be the first member of Runtime");
#if defined(__GNUC__)
#pragma GCC diagnostic pop
#endif
static_assert(
    std::is_standard_layout_v<DeviceRuntimeLaunchDesc>,
    "DeviceRuntimeLaunchDesc must be standard-layout: it is rtMemcpy'd to device"
);
static_assert(
    std::is_trivially_copyable_v<DeviceRuntimeLaunchDesc>,
    "DeviceRuntimeLaunchDesc must be trivially copyable: it is rtMemcpy'd to device"
);
static_assert(
    sizeof(DeviceRuntimeLaunchDesc) % 64 == 0,
    "DeviceRuntimeLaunchDesc size must be a multiple of 64 so the device-copied image "
    "stays cache-line aligned"
);
static_assert(
    offsetof(DeviceRuntimeLaunchDesc, workers) % 64 == 0,
    "workers must start on a cache line: each Handshake is one line the AICore writes back whole"
);
static_assert(
    offsetof(DeviceRuntimeLaunchDesc, workers) + sizeof(DeviceRuntimeLaunchDesc::workers) ==
        sizeof(DeviceRuntimeLaunchDesc),
    "workers must end the descriptor on this runtime: it has no gate tail, so the initialized prefix is "
    "the whole extent and a field appended behind it would never be published"
);

// Bytes a steady-state run uploads: the descriptor before the handshake region.
// Defined per-runtime so the shared device_runner_helpers.cpp /
// kernel_persistent_args.cpp paths stay runtime-agnostic.
size_t runtime_device_copy_size(const Runtime &rt);

// Bytes the first publication onto a device allocation uploads: through the end
// of the handshake region. A5 trb has no post-close gate array, so this equals
// the device extent below.
size_t runtime_device_initialized_prefix_size(const Runtime &rt);

// Bytes of device memory a Runtime image occupies, and the size every allocation
// backing a device `Runtime` must use. Never smaller than
// `runtime_device_initialized_prefix_size`; equal to it on this runtime.
size_t runtime_device_extent_size(const Runtime &rt);

#endif  // SRC_A5_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
