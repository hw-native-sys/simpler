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
 * - The reference to the active callable's registration-owned function tables
 *
 * Task dispatch uses a per-core DispatchPayload written by the scheduler.
 * At dispatch time, build_payload() copies tensor pointers and scalars from
 * the task payload into the per-core args[], populates SPMD context, then
 * signals AICore via DATA_MAIN_BASE.
 */

#pragma once

#include <stddef.h>  // for offsetof
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include <type_traits>
#include <vector>

#include "common/core_type.h"
#include "common/host_api.h"
#include "common/chip_swimlane_profiling.h"
#include "common/platform_config.h"
#include "aicpu/platform_aicpu_affinity.h"  // MAX_GATE_THREADS (aicpu_allowed_cpus bound)
#include "dispatch_payload.h"
#include "task_args.h"
#include "aicore_teardown.h"
#include "common/launch_entry_args.h"             // EntryArgsSource, LaunchEntryArgsPlan
#include "tensormap_and_ringbuffer/entry_args.h"  // EntryArgsStorage
#include "utils/tensor_lease.h"

// =============================================================================
// Configuration Macros
// =============================================================================

#define RUNTIME_MAX_WORKER PLATFORM_MAX_CORES  // 24 AIC + 48 AIV cores
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
 * Field Access Patterns:
 * - aicpu_ready: Reserved legacy field; the current handshake does not use it
 * - aicore_done: Written by AICore, read by AICPU (final report; physical_core_id
 *   and core_type are published alongside it in the same write)
 * - task: Written by AICPU before window-open, read by AICore after window-open
 * - core_type: Written by AICore (with aicore_done), read by AICPU (CoreType::AIC or CoreType::AIV)
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

// The AICore owns this line's writeback: it flushes the whole line with
// dcci(..., CACHELINE_OUT) on its report and again on exit. A word the AICPU
// must publish independently cannot live here — a stale line writeback would
// overwrite it. The post-close return gates live in
// DeviceRuntimeLaunchDesc::teardown_gates, one isolated line each.
static_assert(sizeof(Handshake) == 64);
static_assert(std::is_standard_layout_v<Handshake> && std::is_trivially_copyable_v<Handshake>);
// The payload offsets are the device-side wire contract: AICore writes them and
// the AICPU sweeps read them back, so a field that moved would mis-decode
// silently. `report_epoch` occupies padding the struct already had.
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
 * This is the ONLY part of Runtime that reaches device memory: the host fills it,
 * `device_runner_helpers.cpp` allocates `sizeof(DeviceRuntimeLaunchDesc)` bytes
 * for it and rtMemcpy's a prefix from offset 0 of the Runtime image, and the
 * AICPU/AICore read these fields back. It is the first member of Runtime
 * (offsetof == 0), so the narrowed copy needs no offset arithmetic.
 *
 * Three lengths, in order. `runtime_device_copy_size` is what a steady-state run
 * re-publishes: it stops inside `orch_args_storage_`, after the tensor slots that
 * run filled, and so is the only one of the three that varies per run rather than
 * per runtime — at capacity it reaches exactly `workers`.
 * `runtime_device_initialized_prefix_size` is the full range through `workers`,
 * stopping before `teardown_gates`, and is what the first publication onto an
 * allocation sends so the handshake region and every args slot start defined;
 * `runtime_device_extent_size` is the whole allocation. Both remain per-runtime
 * constants. The gate tail is in no copy: its host storage is never initialized.
 *
 * Adding a field here grows the device image; adding a field to Runtime's
 * host-only tail does not. Keep it standard-layout (static_assert below) so the
 * rtMemcpy is well-defined. alignas(64) makes sizeof a multiple of the cache
 * line so the per-run cache_invalidate_range(runtime, sizeof(dev)) never rounds
 * into a neighbouring line.
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

    // Filter-style affinity gate input (a2a3 onboard). Host fills these
    // before launch from AICPU OCCUPY, and the device gate keeps threads whose
    // sched_getcpu() lands on one of the cpu_ids. The array position is the
    // deterministic exec_idx used by AicpuExecutor for sched/orch role
    // assignment; the highest active index is the orchestrator slot.
    int32_t aicpu_allowed_cpus[MAX_GATE_THREADS];
    int32_t aicpu_allowed_cpu_count;
    int32_t aicpu_launch_count;

    // Reference to the active callable's registration-owned func_id ->
    // CoreCallable object address table: the device address of entry 0 and the
    // number of entries the table holds. The table itself lives in that
    // callable's retained registration block and is written by the single
    // registration copy that also delivers the code those addresses name, so a
    // run publishes the reference and never the contents. Both are zero until a
    // bind names a callable; a func_id at or past the length is unmapped.
    uint64_t callable_table_addr_;
    uint32_t callable_table_len_;

    // Serial orchestrator -> scheduler start control.
    // When true, scheduler threads wait until orchestration has fully built the
    // task graph before entering resolve_and_dispatch().
    // Controlled via SIMPLER_TMR_SERIAL_ORCH_SCHED_ENABLE environment variable.
    bool serial_orch_sched;

    void *gm_sm_ptr_;  // GM pointer to shared memory (device)

    // Prebuilt-arena fast path (trb only). Set by the host before rtMemcpy'ing
    // Runtime to device; AICPU reads them in the boot path to skip
    // runtime_create_from_sm and reuse the pooled, prebuilt arena buffer
    // (already populated by runtime_init_data_from_layout + wire on host).
    void *prebuilt_arena_base_;
    size_t prebuilt_runtime_offset_;

    // Per-callable_id dispatch. AICPU dispatches via
    // `orch_so_table_[active_callable_id_]`.
    int32_t active_callable_id_;

    // How this run's entry arguments reached the device, and how many of each
    // the sender put there. Ahead of the storage they describe, so the shortest
    // publication still carries them: on the launch route that publication ends
    // at `orch_args_storage_` and the values arrive as launch arguments
    // instead.
    //
    // Duplicated in the launch header, where the AICPU compares the two. That
    // comparison finds a run whose two sides disagree; it says nothing about
    // whether values carrying the same counts are this run's.
    uint32_t entry_tensor_count_;
    uint32_t entry_scalar_count_;
    uint32_t entry_args_source_;  // EntryArgsSource

    // Entry args, adopted on the host. Last of the uploaded fields, because it
    // is the only one whose useful length is a property of the run: a
    // steady-state publication stops after `tensor_count_` of its
    // CHIP_MAX_TENSOR_ARGS slots (`runtime_device_copy_size`), and every field
    // above it is therefore inside every publication regardless of that count.
    // Slots past the count keep whatever an earlier run of this allocation
    // wrote; the consumers are count-bounded and must stay so.
    simpler::tmr::EntryArgsStorage orch_args_storage_;

    // Handshake buffers for AICPU-AICore communication, one 64-byte line per
    // worker.
    //
    // Outside the per-run uploaded prefix: every field the device reads here is
    // written on the device — the AICore publishes its report and the AICPU the
    // task pointer it answers with — and no host value is consumed. A
    // steady-state run re-uploads none of it; the first publication onto a given
    // allocation carries it once, which is what gives a fresh block a defined
    // starting value.
    Handshake workers[RUNTIME_MAX_WORKER];

    // Post-close return gates, one isolated cache line per worker. The AICPU
    // stores here only after that worker's register window is closed; the
    // AICore bypass-loads its own entry and returns once it reads RELEASE.
    // Separate from workers[] because the AICore flushes its whole Handshake
    // line, which would overwrite a gate sharing it.
    //
    // Last, and outside the uploaded prefix: the AICPU zeroes every active entry
    // in `pre_handshake_init` and executes `wmb()` before it publishes
    // `hs_setup_done_`, and no register window opens before that publication. So
    // the meaningful initial value is produced on the device ahead of every read
    // of it, and no host-supplied gate value is consumed. The allocation still
    // covers this array.
    AicoreTeardownControl teardown_gates[RUNTIME_MAX_WORKER];
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
    AicoreTeardownControl *get_teardown_gates() { return dev.teardown_gates; }
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

    // Entry-argument route, as the run that published this descriptor named it.
    // The host writes these three words into the publication itself rather than
    // into this object, because which route a run takes is decided after the
    // values were captured — so on the host these read what `Runtime()` set,
    // and on the device they read what that run published.
    EntryArgsSource get_entry_args_source() const;
    uint32_t get_entry_tensor_count() const;
    uint32_t get_entry_scalar_count() const;

    /**
     * Adopt entry values from raw launch-argument bytes.
     *
     * `payload` is the launch package's entry region, read as bytes: its base
     * alignment belongs to whoever allocated the launch arguments, so nothing
     * types it until it has landed in the 64-byte-aligned storage above.
     * Rejects counts outside capacity or disagreeing with the descriptor's own,
     * and writes nothing when it rejects.
     */
    bool adopt_entry_args_from_launch(const void *payload, uint32_t tensor_count, uint32_t scalar_count);

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

    /**
     * The active callable's CoreCallable object address for `func_id`, read
     * through the host view of the registration-owned table.
     *
     * Returns 0 for a func_id at or past the table's length, and for every
     * func_id while no callable is bound. Reads host memory: the device address
     * the descriptor carries is never dereferenced here.
     */
    uint64_t get_function_bin_addr(int func_id) const;

    /**
     * Bind this run to one registration-owned function-table pair.
     *
     * `host_view` addresses `len` object-address entries the registration owns
     * for as long as the callable stays registered; `object_table_addr` is that
     * same table's device address, and `entry_table_addr` names its
     * resolved-entry sibling — zero on a runtime whose device scheduler reads
     * no such view.
     */
    void
    set_callable_tables(const uint64_t *host_view, uint64_t object_table_addr, uint64_t entry_table_addr, uint32_t len);

    /**
     * Drop the function-table reference.
     *
     * A bind installs the reference, and a bind that fails afterwards must not
     * leave the previous callable's standing: unregistering that callable frees
     * the block these addresses point into, the scheduler dereferences them,
     * and the AICore calls what it finds there.
     */
    void clear_callable_tables();

    /** Device address of the bound resolved-entry table, 0 when none is bound. */
    uint64_t callable_entry_table_addr() const;

    /** Entries in the bound function tables, 0 when none is bound. */
    uint32_t callable_table_len() const;

    // =========================================================================
    // Host-only state (not copied to device)
    // =========================================================================

    // Host-side tensor ledger for the run's H2D and D2H transfers. Populated by
    // runtime_maker.cpp from orch_args at bind time, iterated by
    // copy_in_run_inputs_impl and copy_back_run_outputs_impl, and released by
    // release_run_bindings_impl. Host-only (after `dev`): never uploaded.
    std::vector<TensorLease> tensor_leases_;

    // Host view of the active callable's registration-owned object-address
    // table, borrowed for as long as that callable stays registered, plus the
    // device address of its resolved-entry sibling. Host-only because the first
    // is a host pointer, and because the host lookup must not reach for the
    // device address `dev` carries.
    const uint64_t *callable_table_host_{nullptr};
    uint64_t callable_entry_table_addr_{0};

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
    "DeviceRuntimeLaunchDesc size must be a multiple of 64 so cache_invalidate_range(sizeof(dev)) "
    "stays cache-line aligned"
);
static_assert(
    offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_) + sizeof(simpler::tmr::EntryArgsStorage) ==
        offsetof(DeviceRuntimeLaunchDesc, workers),
    "orch_args_storage_ must be the last uploaded member: runtime_device_copy_size() stops inside it, so a "
    "field placed between it and workers would fall outside every steady-state publication"
);
static_assert(
    offsetof(DeviceRuntimeLaunchDesc, orch_args_storage_) % 64 == 0,
    "orch_args_storage_ must start on a cache line: the steady-state length is its offset plus a whole "
    "number of 64-byte-aligned Tensor slots, and both halves have to stay line-aligned"
);
static_assert(
    offsetof(DeviceRuntimeLaunchDesc, workers) % 64 == 0,
    "workers must start on a cache line: each Handshake is one line the AICore writes back whole"
);
static_assert(
    offsetof(DeviceRuntimeLaunchDesc, workers) + sizeof(DeviceRuntimeLaunchDesc::workers) ==
        offsetof(DeviceRuntimeLaunchDesc, teardown_gates),
    "workers must be the last host-initialized member: the initialized prefix ends where it ends, and a "
    "field placed between it and the gates would be uploaded once instead of every run"
);
static_assert(
    offsetof(DeviceRuntimeLaunchDesc, teardown_gates) % 64 == 0,
    "teardown_gates must start on a cache line: the AICore flushes a whole Handshake line and a gate "
    "sharing one would be overwritten"
);
static_assert(
    offsetof(DeviceRuntimeLaunchDesc, teardown_gates) + sizeof(DeviceRuntimeLaunchDesc::teardown_gates) ==
        sizeof(DeviceRuntimeLaunchDesc),
    "teardown_gates must end the descriptor: a field appended behind it would sit outside the uploaded "
    "prefix and never receive its host value"
);

// Bytes a steady-state run uploads: the descriptor before the handshake region.
// Defined per-runtime so the shared device_runner_helpers.cpp /
// kernel_persistent_args.cpp paths stay runtime-agnostic.
size_t runtime_device_copy_size(const Runtime &rt);

// Bytes the first publication onto a device allocation uploads: through the end
// of the handshake region, stopping before the host-uninitialized gate tail.
size_t runtime_device_initialized_prefix_size(const Runtime &rt);

// Bytes of device memory a Runtime image occupies. Never smaller than
// `runtime_device_initialized_prefix_size`, and the size every allocation
// backing a device `Runtime` must use: the device addresses gates inside the
// tail this exceeds the published prefixes by.
size_t runtime_device_extent_size(const Runtime &rt);

// This run's entry-argument routing facts, captured from `rt` while it is still
// this run's. Defined per-runtime, and `supported` is false on a runtime with no
// launch route, which is what keeps the shared host launch path runtime-agnostic.
LaunchEntryArgsPlan runtime_launch_entry_args_plan(const Runtime &rt);
/**
 * Whether a launch header may be adopted, decided before any payload address is
 * formed from it.
 *
 * Every rejection is its own verdict so a caller can say which check failed,
 * and the order is the contract: an undefined source is refused before it is
 * compared, the comparison before the offset, the offset before the counts, and
 * the counts — both against capacity and against the descriptor's own — before
 * anything derives an address. `Descriptor` means this run's values are already
 * in place and there is nothing to adopt.
 */
enum class LaunchEntryArgsVerdict : int32_t {
    Descriptor,
    Adopt,
    UndefinedSource,
    SourceMismatch,
    UnexpectedOffset,
    CountsPastCapacity,
    CountsMismatch,
};

LaunchEntryArgsVerdict classify_launch_entry_args(
    const Runtime &rt, uint32_t source, uint32_t offset, uint32_t tensor_count, uint32_t scalar_count
);
