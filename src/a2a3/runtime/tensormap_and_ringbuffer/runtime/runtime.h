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
 * protocol. Task graph construction is handled by PTO2Runtime; this class
 * only handles:
 * - Handshake buffers for AICPU-AICore communication
 * - Execution parameters (block_dim, aicpu_thread_num)
 * - Tensor pair management for host-device memory tracking
 * - Device orchestration state (gm_sm_ptr_, orch_args_)
 * - Function address mapping (func_id_to_addr_)
 *
 * Task dispatch uses a per-core PTO2DispatchPayload written by the scheduler.
 * At dispatch time, build_payload() copies tensor pointers and scalars from
 * the task payload into the per-core args[], populates SPMD context, then
 * signals AICore via DATA_MAIN_BASE.
 */

#ifndef SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
#define SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_

#include <stddef.h>  // for offsetof
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>   // for fprintf, printf
#include <string.h>  // for memset

#include <type_traits>
#include <vector>

#include "common/core_type.h"
#include "common/host_api.h"
#include "common/l2_swimlane_profiling.h"
#include "common/platform_config.h"
#include "aicpu/platform_aicpu_affinity.h"  // MAX_GATE_THREADS (aicpu_allowed_cpus bound)
#include "pto2_dispatch_payload.h"
#include "task_args.h"

// =============================================================================
// Configuration Macros
// =============================================================================

#define RUNTIME_MAX_ARGS 128
#define RUNTIME_MAX_WORKER 72  // 24 AIC + 48 AIV cores
#define RUNTIME_MAX_FUNC_ID 1024
#define RUNTIME_MAX_ORCH_SYMBOL_NAME 64

// Default ready queue shards: one shard per worker thread (total minus orchestrator)
constexpr int RUNTIME_DEFAULT_READY_QUEUE_SHARDS = PLATFORM_MAX_AICPU_THREADS - 1;

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
 * 1. Initialization: AICPU sets aicpu_ready=1
 * 2. Acknowledgment: AICore sets aicore_done=core_id+1
 * 3. Task Dispatch: AICPU writes DATA_MAIN_BASE after updating the per-core payload
 * 4. Task Execution: AICore reads the cached PTO2DispatchPayload and executes
 * 5. Task Completion: AICore writes FIN to COND; AICPU observes completion
 * 6. Shutdown: AICPU sets control=1, AICore exits
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
 * - aicpu_ready: Written by AICPU, read by AICore
 * - aicore_done: Written by AICore, read by AICPU (final report; physical_core_id
 *   and core_type are published alongside it in the same write)
 * - task: Written by AICPU, read by AICore (0 = not ready, non-zero = PTO2DispatchPayload*)
 * - core_type: Written by AICore (with aicore_done), read by AICPU (CoreType::AIC or CoreType::AIV)
 * - physical_core_id: Written by AICore (with aicore_done), read by AICPU
 */
struct Handshake {
    volatile uint32_t aicpu_ready;  // AICPU ready signal: 0=not ready, 1=ready
    volatile uint32_t aicore_done;  // AICore ready signal: 0=not ready, core_id+1=ready
    volatile uint64_t task;         // Init: PTO2DispatchPayload* (set before aicpu_ready); runtime: unused
    volatile CoreType core_type;    // Core type: CoreType::AIC or CoreType::AIV (reported by AICore with aicore_done)
    volatile uint32_t physical_core_id;  // Physical core ID (reported by AICore with aicore_done)
} __attribute__((aligned(64)));

/**
 * Tensor pair for tracking host-device memory mappings.
 * Used for copy-back during finalize.
 */
struct TensorPair {
    void *host_ptr;
    void *dev_ptr;
    size_t size;
    // false for read-only INPUT tensors: they are never written by the kernel,
    // so the end-of-run D2H copy-back is skipped. OUTPUT/INOUT/unknown
    // keep the safe default of copying back.
    bool needs_copy_back = true;
};

/**
 * Task structure - Compatibility stub for platform layer
 *
 * RT2 uses PTO2DispatchPayload instead of Task for task dispatch.
 * This stub exists only for API compatibility with device_runner.cpp.
 * Since get_task_count() returns 0, this struct is never actually used.
 */
struct Task {
    int func_id;
    uint64_t function_bin_addr;
};

// =============================================================================
// Device launch descriptor
// =============================================================================

/**
 * DeviceRuntimeLaunchDesc - the device-copied half of Runtime.
 *
 * This is the ONLY part of Runtime that crosses the host->device boundary: the
 * host fills it, `device_runner_helpers.cpp` rtMemcpy's exactly
 * `sizeof(DeviceRuntimeLaunchDesc)` bytes from offset 0 of the Runtime image,
 * and the AICPU/AICore read these fields back. It is the first member of
 * Runtime (offsetof == 0), so the narrowed copy needs no offset arithmetic.
 *
 * Adding a field here grows the device image; adding a field to Runtime's
 * host-only tail does not. Keep it standard-layout (static_assert below) so the
 * rtMemcpy is well-defined. alignas(64) makes sizeof a multiple of the cache
 * line so the per-run cache_invalidate_range(runtime, sizeof(dev)) never rounds
 * into a neighbouring line (the leading Handshake is already 64-aligned, but the
 * explicit alignas keeps the property if fields are ever reordered).
 */
struct alignas(64) DeviceRuntimeLaunchDesc {
    // Handshake buffers for AICPU-AICore communication
    Handshake workers[RUNTIME_MAX_WORKER];  // Worker (AICore) handshake buffers
    int worker_count;                       // Number of active workers

    // Execution parameters for AICPU scheduling.
    //
    // aicpu_thread_num is the *total* AICPU thread count launched on this run
    // (= orch + schedulers). AicpuExecutor splits this into one orchestrator
    // thread (highest idx, runs aicpu_orchestration_entry) and the remaining
    // aicpu_thread_num-1 scheduler threads that dispatch tasks to AICore.
    int aicpu_thread_num;
    int ready_queue_shards;  // Number of ready queue shards (1..MAX_AICPU_THREADS, default MAX-1)

    // Filter-style affinity gate input (a2a3 onboard). Host fills these
    // before launch from AICPU OCCUPY, and the device gate keeps threads whose
    // sched_getcpu() lands on one of the cpu_ids. The array position is the
    // deterministic exec_idx used by AicpuExecutor for sched/orch role
    // assignment; the highest active index is the orchestrator slot.
    int32_t aicpu_allowed_cpus[MAX_GATE_THREADS];
    int32_t aicpu_allowed_cpu_count;
    int32_t aicpu_launch_count;

    // PTO2 integration: kernel_id -> GM function_bin_addr mapping
    uint64_t func_id_to_addr_[RUNTIME_MAX_FUNC_ID];

    // Serial orchestrator -> scheduler start control.
    // When true, scheduler threads wait until orchestration has fully built the
    // task graph before entering resolve_and_dispatch().
    // Controlled via PTO2_SERIAL_ORCH_SCHED environment variable.
    bool serial_orch_sched;

    void *gm_sm_ptr_;                        // GM pointer to PTO2 shared memory (device)
    ChipStorageTaskArgs orch_args_storage_;  // Copy of args for device

    // Prebuilt-arena fast path (trb only). Set by the host before rtMemcpy'ing
    // Runtime to device; AICPU reads them in the boot path to skip
    // runtime_create_from_sm and reuse the pooled, prebuilt arena buffer
    // (already populated by runtime_init_data_from_layout + wire on host).
    void *prebuilt_arena_base_;
    size_t prebuilt_runtime_offset_;

    // Per-callable_id dispatch. AICPU dispatches via
    // `orch_so_table_[active_callable_id_]`.
    int32_t active_callable_id_;
};

// =============================================================================
// Runtime Class
// =============================================================================

/**
 * Runtime class for device execution and handshake control
 *
 * This class manages AICPU-AICore communication through handshake buffers.
 * Task graph construction is handled by PTO2Runtime; this class only handles
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
    const ChipStorageTaskArgs &get_orch_args() const;
    void set_gm_sm_ptr(void *p);
    void set_orch_args(const ChipStorageTaskArgs &args);

    // Prebuilt-arena fast path (trb only). Set by host's
    // bind_callable_to_runtime_impl; consumed by AICPU at boot to attach a
    // DeviceArena to `prebuilt_arena_base_` and pick up the PTO2Runtime at
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

    // =========================================================================
    // Deprecated API (for platform compatibility, always returns 0/nullptr)
    // Task graph is now managed by PTO2Runtime, not Runtime
    // =========================================================================

    /** @deprecated Task count is now in PTO2 shared memory */
    int get_task_count() const { return 0; }

    /** @deprecated RT2 uses PTO2DispatchPayload, not Task. Always returns nullptr. */
    Task *get_task(int) { return nullptr; }

    // =========================================================================
    // Host-only state (not copied to device)
    // =========================================================================

    // Host-side tensor ledger for D2H copy-back at finalize. Populated by
    // runtime_maker.cpp from orch_args at bind time, then iterated in
    // validate_runtime_impl. Host-only (after `dev`): never uploaded.
    std::vector<TensorPair> tensor_pairs_;
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

// Number of bytes of the Runtime image that must be copied to the device.
// trb returns sizeof(DeviceRuntimeLaunchDesc) (only `dev` is device-read);
// host_build_graph returns sizeof(Runtime) (its device image is the whole
// object). Defined per-runtime so the shared device_runner_helpers.cpp copy
// path stays runtime-agnostic.
size_t runtime_device_copy_size(const Runtime &rt);

#endif  // SRC_A2A3_RUNTIME_TENSORMAP_AND_RINGBUFFER_RUNTIME_RUNTIME_H_
