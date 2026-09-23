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
 * @file kernel_args.h
 * @brief KernelArgs payload - Shared between Host, AICPU, and AICore
 *
 * This structure is the Simpler runtime payload read by AICPU and AICore
 * kernels. It contains pointers to device memory for runtime data, profiling
 * buffers, and platform state.
 *
 * Platform Support:
 * - a5: Real hardware with CANN runtime compatibility
 * - a5sim: Host-based simulation using standard memory
 *
 * Memory Layout (a5):
 * This platform struct is the front-less per-task runtime payload, passed
 * directly to the onboard AICPU launch (rtsLaunchCpuKernel) and copied to
 * device memory for AICore — no CANN launch front is needed on this path.
 * The bootstrap dispatcher has its own private KernelArgs/DeviceArgs ABI in
 * src/common/aicpu_loader/device/aicpu_dispatcher.cpp.
 *
 * Memory Layout (a5sim):
 * For simulation, the layout is maintained for API compatibility, though
 * we use host memory instead of device memory.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "common/dma_workspace.h"

// Forward declarations
class Runtime;

// Symbol-name capacity for the device orchestration entry/config functions.
// Must match RUNTIME_MAX_ORCH_SYMBOL_NAME in the runtime's runtime.h; a
// static_assert in the TMARB AICPU executor (where both headers are visible)
// enforces the equality.
#define INIT_ARGS_MAX_ORCH_SYMBOL_NAME 64

#ifdef __cplusplus
extern "C" {
#endif

// Define __may_used_by_aicore__ qualifier for platform compatibility
#if defined(__DAV_VEC__) || defined(__DAV_CUBE__)
#define __may_used_by_aicore__ __gm__
#else
#define __may_used_by_aicore__
#endif

/**
 * Kernel arguments payload
 *
 * This structure is the payload passed to AICPU kernels by the host and copied
 * to device memory for AICore kernels.
 *
 * Field Access Patterns:
 * - runtime_args: Written by host, read by AICPU (task runtime, includes
 *   handshake buffers)
 * - dep_gen_data_base: Written by host platform, read by AICPU platform layer;
 *   zero when dep_gen capture is unused
 *
 * Consumer paths:
 *       - AICPU: receives this KernelArgs directly via rtsLaunchCpuKernel
 *       - AICore: receives `AicoreLaunchArgs` via KERNEL_ENTRY, a projection of
 *         this struct carrying the subset its entry reads. It never reads this
 *         struct itself.
 */
struct KernelArgs {
    // Offset-locked front: the front-less launch protocol and the device
    // entries require runtime_args @ 0 and regs @ 8 (see static_asserts below).
    __may_used_by_aicore__ Runtime *runtime_args{nullptr};  // Task runtime in device memory
    uint64_t regs{0};                                       // Per-core register base address array (platform-specific)
    // Remaining 64-bit fields grouped before the 32-bit tail so the struct needs
    // no interior alignment padding. Order among these is free (device reads by
    // field name, not offset); only runtime_args/regs are offset-locked.
    uint64_t dump_data_base{0};  // Dump shared memory base address; use explicit flags to detect enablement
    // chip swimlane shared memory base address; use explicit flags to detect enablement
    uint64_t chip_swimlane_data_base{0};
    // This run's chip-swimlane terminal-snapshot bank, resolved by the host from
    // the run's pipeline slot. 0 whenever the host resolved no bank, including
    // every run with swimlane off; the device treats 0 as "publish no snapshot"
    // and never derives an address of its own.
    uint64_t chip_swimlane_run_terminal_bank{0};
    uint64_t pmu_data_base{0};      // PMU buffer base address (device memory); 0 = PMU disabled
    uint64_t dep_gen_data_base{0};  // dep_gen shared memory base address; use explicit flags to detect enablement
    // Profiling per-core address arrays (moved out of Handshake). Each *_addrs
    // field is a device pointer to uint64_t[num_aicore]. AICore KERNEL_ENTRY
    // indexes by block_idx and forwards into per-core platform state.
    // ChipSwimlaneActiveHead* per core (rotation channel); 0 when chip swimlane is off
    uint64_t chip_swimlane_aicore_rotation_table{0};
    uint64_t aicore_pmu_ring_addrs{0};  // PmuAicoreRing* per core; 0 when PMU is off
    uint64_t scope_stats_data_base{0};  // ScopeStatsBuffer device pointer; 0 when scope_stats is off.
                                        // a5 has no halHostRegister — host keeps a separate shadow and
                                        // refreshes it via rtMemcpy DEVICE_TO_HOST at dump time.
    // Device pointer to an 8-byte buffer that the platform AICPU entry writes
    // the run-wall (ns) into. Allocated once at simpler_init, kept resident.
    // See the a2a3 kernel_args.h for the full design rationale (CANN's
    // AICPU args copy makes inline fields write-only).
    uint64_t device_wall_data_base{0};

    // Device pointer to this run's result region (DeviceRunResultRegion), and
    // the run epoch its device side publishes into that region. Same reason the
    // wall base travels here rather than inline: AICPU gets KernelArgs as a
    // CANN-private copy, so an inline field would be write-only from AICPU.
    //
    // Unlike the wall buffer this is NOT gated on diagnostics — a run's error
    // result has to survive whether or not timing capture is on. The host does
    // not clear the region per run; the epoch is what makes a previous run's
    // payload recognisable as stale, so arming costs no H2D.
    // Both zero when no region was allocated.
    uint64_t run_result_data_base{0};
    uint64_t run_result_epoch{0};
    // AICPU die of each scheduler slot, packed per common/scheduler_die_partition.h.
    // 0 means "no slot known", which keeps the device on round-robin cluster
    // ownership. a5-only: the host_build_graph runtime ignores it.
    uint64_t sched_thread_die_bits{0};
    // 32-bit tail (two adjacent uint32_t — no interior padding).
    uint32_t enable_profiling_flag{0};  // Profiling umbrella bitmask; dump_args|chip_swimlane|pmu|dep_gen|scope_stats
    // Opaque always-false guard read by the AICore SIMT meta anchor (AIV
    // KERNEL_ENTRY). The host never sets it non-zero; its only purpose is to be
    // a runtime-valued condition the compiler cannot constant-fold, so the
    // never-executed SIMT launch in simt_anchor.h survives DCE and bisheng
    // still classifies the entry as SIMT. Keep it last (trailing field).
    uint32_t force_simt_anchor{0};
};

static_assert(offsetof(KernelArgs, runtime_args) == 0, "KernelArgs::runtime_args offset drift");
static_assert(offsetof(KernelArgs, regs) == 8, "KernelArgs::regs offset drift");

// The swimlane bases are not offset-locked by any device contract — AICPU reads
// them by field name from a CANN-private copy of the whole struct. These pin the
// measured a5 layout so that appending, reordering, or widening a field is a
// build failure rather than a silently different launch payload: `sizeof` is
// what `launch_aicpu_payload` hands to `rtsLaunchCpuKernel` as `argsSize`, and
// what `PersistentKernelArgs::prepare_once` allocates and copies H2D. The values
// differ from a2a3's: this struct has no `ffts_base_addr` and carries two
// trailing uint32_t rather than one.
static_assert(offsetof(KernelArgs, chip_swimlane_data_base) == 24, "KernelArgs::chip_swimlane_data_base offset drift");
static_assert(
    offsetof(KernelArgs, chip_swimlane_run_terminal_bank) == 32,
    "KernelArgs::chip_swimlane_run_terminal_bank offset drift"
);
static_assert(sizeof(KernelArgs) == 120, "KernelArgs launch-payload size drift");
static_assert(alignof(KernelArgs) == 8, "KernelArgs launch-payload alignment drift");
// No conditional members: the struct body carries no preprocessor branch, so
// these values are the same in every translation unit that sees this header.
static_assert(__is_trivially_copyable(KernelArgs), "KernelArgs must be memcpy-able to the device");
static_assert(__is_standard_layout(KernelArgs), "KernelArgs must be standard-layout");

/**
 * AicoreLaunchArgs - the AICore entry's launch argument block.
 *
 * Mirrors `KERNEL_ENTRY(aicore_kernel)`'s parameter list field for field: ccec
 * demotes a struct parameter to a hidden pointer, so the entry takes a flat
 * scalar list and this is the host-side image of it. Changing either without
 * the other silently mis-decodes the block.
 *
 * Every address an AICore entry needs is here, so the entry publishes its
 * per-core state from these values alone and reads no GM to do it. The driver
 * copies the block during the launch call, so the host builds it after
 * collector arming and these are this run's final values. `force_simt_anchor`
 * stays a launch argument so its value is opaque to the optimizer where the
 * never-taken SIMT branch is emitted.
 */
struct AicoreLaunchArgs {
    uint64_t runtime_args;
    uint32_t enable_profiling_flag;
    uint32_t force_simt_anchor;
    uint64_t chip_swimlane_aicore_rotation_table;
    uint64_t aicore_pmu_ring_addrs;
    // `KernelArgs::regs`, which the entry indexes by physical core id to resolve
    // this core's PMU MMIO base. The AICPU reads the same table for its own
    // dispatch windows.
    uint64_t pmu_reg_addrs;
    // This run's report identity. Non-zero on a native program launch, and the
    // AICore commits it last in its handshake report so the AICPU accepts only
    // this run's report. Zero on a kernel/persistent launch, which keeps the
    // `aicore_done != 0` predicate.
    uint64_t report_epoch;
};

static_assert(sizeof(AicoreLaunchArgs) == 48, "AicoreLaunchArgs size drift");

/**
 * Fill the `AicoreLaunchArgs` fields every architecture shares, from the
 * `KernelArgs` the host already built for the AICPU.
 *
 * `report_epoch` is the projection that gives both processors the same run
 * identity: the AICPU reads `KernelArgs::run_result_epoch` directly, the AICore
 * has no `KernelArgs`, so the identical value travels here. A native program
 * launch supplies a non-zero epoch; a kernel/persistent launch leaves it 0,
 * which selects the protocol that predates the stamp.
 */
inline void fill_shared_launch_args(AicoreLaunchArgs &args, const KernelArgs &k_args) {
    args.runtime_args = reinterpret_cast<uint64_t>(k_args.runtime_args);
    args.enable_profiling_flag = k_args.enable_profiling_flag;
    args.report_epoch = k_args.run_result_epoch;
}

/**
 * Fill the fields of `AicoreLaunchArgs` that only this architecture has, so the
 * shared launch path can build the block without knowing which arch it is on.
 */
inline void fill_arch_launch_args(AicoreLaunchArgs &args, const KernelArgs &k_args) {
    args.force_simt_anchor = k_args.force_simt_anchor;
    args.chip_swimlane_aicore_rotation_table = k_args.chip_swimlane_aicore_rotation_table;
    args.aicore_pmu_ring_addrs = k_args.aicore_pmu_ring_addrs;
    args.pmu_reg_addrs = k_args.regs;
}

/**
 * InitArgs - per-device runtime configuration
 *
 * Uploaded at worker init via `simpler_aicpu_init`, before any
 * register_callable/exec launch. Republished when first-use provisioning adds
 * an async-DMA workspace. The values do not ride on per-run KernelArgs; the
 * resident AICPU SO keeps the latest configuration across task launches.
 *
 * `regs` is intentionally NOT here. The host owns the per-core register table
 * per device context, but on a5 the table is also read by the AICore
 * KERNEL_ENTRY off the per-run device KernelArgs copy, so it stays in
 * KernelArgs.
 */
struct InitArgs {
    uint32_t device_id{0};            // ACL device ordinal -> set_orch_device_id
    uint32_t log_level{25};           // Threshold: DEBUG=10, INFO=20, TIMING=25, WARN=30, ERROR=40, NUL=60
    int32_t scheduler_timeout_ms{0};  // AICPU no-progress watchdog (ms); 0 -> compile default
    // Per-engine async-DMA workspace dev addrs -> set_dma_workspace_addr(kind, .);
    // indexed by DmaWorkspaceKind; 0 = that engine unavailable.
    uint64_t dma_workspace_addr[DMA_WORKSPACE_KIND_COUNT]{};
};

struct AicpuTopologyQueryResult {
    int32_t occupy_rc{-1};
    int32_t pf_occupy_rc{-1};
    int32_t os_sched_rc{-1};
    int32_t reserved{0};
    uint64_t occupy{0};
    uint64_t pf_occupy{0};
    uint64_t os_sched{0};
};

struct AicpuTopologyQueryArgs {
    uint64_t result_addr{0};
};

static_assert(
    std::is_trivially_copyable_v<AicpuTopologyQueryResult> && std::is_standard_layout_v<AicpuTopologyQueryResult>,
    "AicpuTopologyQueryResult must remain a memcpy-safe wire type"
);
static_assert(sizeof(AicpuTopologyQueryResult) == 40, "AicpuTopologyQueryResult ABI size drift");
static_assert(offsetof(AicpuTopologyQueryResult, occupy) == 16, "AicpuTopologyQueryResult::occupy offset drift");
static_assert(offsetof(AicpuTopologyQueryResult, os_sched) == 32, "AicpuTopologyQueryResult::os_sched offset drift");
static_assert(
    std::is_trivially_copyable_v<AicpuTopologyQueryArgs> && std::is_standard_layout_v<AicpuTopologyQueryArgs>,
    "AicpuTopologyQueryArgs must remain a memcpy-safe wire type"
);
static_assert(sizeof(AicpuTopologyQueryArgs) == 8, "AicpuTopologyQueryArgs ABI size drift");

/**
 * RegisterCallableArgs - device orchestration SO registration payload
 *
 * Uploaded by the host register_callable path via `simpler_aicpu_register_callable`.
 * Carries only the orchestration-SO descriptor the AICPU executor needs to
 * (re)dlopen a callable's device-orch SO — extracted from Runtime so the
 * register path no longer H2D's a full Runtime. On hbg this is all-zero
 * (host-side orchestration; no device dlopen) and the entry is a no-op.
 */
struct RegisterCallableArgs {
    int32_t active_callable_id{-1};                                  // orch_so_table_ slot
    uint64_t dev_orch_so_addr{0};                                    // device address of the orch SO image
    uint64_t dev_orch_so_size{0};                                    // orch SO image size in bytes
    char device_orch_func_name[INIT_ARGS_MAX_ORCH_SYMBOL_NAME]{};    // entry symbol
    char device_orch_config_name[INIT_ARGS_MAX_ORCH_SYMBOL_NAME]{};  // config symbol
};

#ifdef __cplusplus
}
#endif
