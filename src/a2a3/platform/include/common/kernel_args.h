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
 * - a2a3: Real hardware with CANN runtime compatibility
 * - a2a3sim: Host-based simulation using standard memory
 *
 * Memory Layout (a2a3):
 * This platform struct is the front-less per-task runtime payload, passed
 * directly to the onboard AICPU launch (rtsLaunchCpuKernel) and copied to
 * device memory for AICore — no CANN launch front is needed on this path.
 * The bootstrap dispatcher has its own private KernelArgs/DeviceArgs ABI in
 * src/common/aicpu_loader/device/aicpu_dispatcher.cpp.
 *
 * Memory Layout (a2a3sim):
 * For simulation, the layout is maintained for API compatibility, though
 * we use host memory instead of device memory.
 */

#ifndef PLATFORM_COMMON_KERNEL_ARGS_H_
#define PLATFORM_COMMON_KERNEL_ARGS_H_

#include <cstddef>
#include <cstdint>

#include "common/dma_workspace.h"
#include "common/launch_entry_args.h"

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
 * - dump_data_base: Written by host, read by AICPU platform layer; zero when
 *   args dump is unused
 * - pmu_data_base: Written by host platform, read by AICPU platform layer;
 *   zero when PMU is unused
 * - dep_gen_data_base: Written by host platform, read by AICPU platform layer;
 *   zero when dep_gen capture is unused
 *
 * enable_profiling_flag bit definitions (umbrella bitmask — "profiling" is
 * the umbrella, each bit is a parallel diagnostics sub-feature):
 * - bit0: args dump enabled
 * - bit1: chip swimlane enabled
 * - bit2: PMU enabled
 * - bit3: dep_gen capture enabled
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
    // Remaining 64-bit fields. Grouped before the 32-bit tail so the struct
    // needs no interior alignment padding — every uint64_t lands on its natural
    // 8-byte boundary and the lone trailing uint32_t carries only harmless tail
    // padding. Order among these is free (device reads by field name, not
    // offset); only runtime_args/regs are offset-locked.
    uint64_t ffts_base_addr{0};  // FFTS base address for AICore
    uint64_t dump_data_base{0};  // Dump shared memory base address; use explicit flags to detect enablement
    // chip swimlane shared memory base address; use explicit flags to detect enablement
    uint64_t chip_swimlane_data_base{0};
    // This run's chip-swimlane terminal-snapshot bank, resolved by the host from
    // the run's pipeline slot. 0 whenever the host resolved no bank, including
    // every run with swimlane off; the device treats 0 as "publish no snapshot"
    // and never derives an address of its own.
    uint64_t chip_swimlane_run_terminal_bank{0};
    uint64_t pmu_data_base{0};  // PMU shared memory base address; use explicit flags to detect enablement
    // Per-core PMU MMIO register base address array. 0 on sim, and 0 when this
    // run leaves PMU off; the table itself is device-constant and outlives any
    // single run.
    uint64_t pmu_reg_addrs{0};
    uint64_t dep_gen_data_base{0};      // dep_gen shared memory base address; use explicit flags to detect enablement
    uint64_t scope_stats_data_base{0};  // ScopeStatsBuffer shared memory base; 0 when scope_stats is off.
                                        // Allocated by host's ScopeStatsCollector, read+written by AICPU's
                                        // scope_stats_collector via set_platform_scope_stats_base.
    // Device ptr to a uint64_t[num_aicore] table holding each core's
    // ChipSwimlaneActiveHead address (rotation channel). AICore kernel entry indexes
    // by block_idx and forwards into platform set/get state. 0 when chip swimlane is off.
    uint64_t chip_swimlane_aicore_rotation_table{0};
    // Device pointer to the run-wall buffer the platform AICPU entry writes.
    // Allocated once and kept resident, reset each run. Onboard AICPU receives
    // KernelArgs as a CANN-private copy (see launch_aicpu_payload), so an
    // inline field would be write-only from AICPU;
    // the dedicated host-allocated buffer's address travels via this field.
    // Onboard layout: one { start_cycle, end_cycle } pair per launched AICPU
    // thread (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH pairs, raw sys-counter
    // cycles). Each surviving thread writes its own slot (plain stores, no
    // atomics); the host reduces max(end) - min(start) -> ns on readback (see
    // ensure_device_wall_buffer / read_device_wall_ns). Sim keeps the simpler
    // single-uint64 wall_ns write-through (sim AICPU and host share memory).
    // Zero when the buffer was not allocated.
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
    // 32-bit tail.
    uint32_t enable_profiling_flag{0};  // Profiling umbrella bitmask; dump_args|chip_swimlane|pmu|dep_gen|scope_stats

    // How this run's entry arguments reached the AICPU, restated from the
    // descriptor so the device can reject a pair that disagrees.
    // `entry_args_offset` is where the entry region starts inside this launch
    // package, `LAUNCH_ENVELOPE_HEADER_BYTES` when the launch route carries the
    // values and 0 when it does not. The region itself is raw bytes appended
    // after this header — tensor descriptors then scalars — and is not part of
    // this struct: `argsSize` is what tells RTS how far past it to copy.
    uint32_t entry_args_offset{0};
    uint32_t entry_tensor_count{0};
    uint32_t entry_scalar_count{0};
    uint32_t entry_args_source{static_cast<uint32_t>(EntryArgsSource::Descriptor)};
};

static_assert(offsetof(KernelArgs, runtime_args) == 0, "KernelArgs::runtime_args offset drift");
static_assert(offsetof(KernelArgs, regs) == 8, "KernelArgs::regs offset drift");

// The swimlane bases are not offset-locked by any device contract — AICPU reads
// them by field name from a CANN-private copy of the whole struct. These pin the
// measured a2a3 layout so that appending, reordering, or widening a field is a
// build failure rather than a silently different launch payload: `sizeof` is
// what `launch_aicpu_payload` hands to `rtsLaunchCpuKernel` as `argsSize`, and
// what `PersistentKernelArgs::prepare_once` allocates and copies H2D.
static_assert(offsetof(KernelArgs, chip_swimlane_data_base) == 32, "KernelArgs::chip_swimlane_data_base offset drift");
static_assert(
    offsetof(KernelArgs, chip_swimlane_run_terminal_bank) == 40,
    "KernelArgs::chip_swimlane_run_terminal_bank offset drift"
);
static_assert(sizeof(KernelArgs) == 136, "KernelArgs launch-payload size drift");
static_assert(alignof(KernelArgs) == 8, "KernelArgs launch-payload alignment drift");
// No conditional members: the struct body carries no preprocessor branch, so
// these values are the same in every translation unit that sees this header.
static_assert(__is_trivially_copyable(KernelArgs), "KernelArgs must be memcpy-able to the device");
static_assert(__is_standard_layout(KernelArgs), "KernelArgs must be standard-layout");
// The launch package puts the entry region at a fixed offset on every arch, so
// this header has to fit inside it.
static_assert(
    sizeof(KernelArgs) <= LAUNCH_ENVELOPE_HEADER_BYTES,
    "KernelArgs must fit in the launch package header the entry region starts after"
);

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
 * collector arming and these are this run's final values.
 */
struct AicoreLaunchArgs {
    uint64_t runtime_args;
    uint32_t enable_profiling_flag;
    uint64_t ffts_base_addr;
    uint64_t chip_swimlane_aicore_rotation_table;
    // This run's report identity. Non-zero on a native program launch, and the
    // AICore commits it last in its handshake report so the AICPU accepts only
    // this run's report. Zero on a kernel/persistent launch, which keeps the
    // `aicore_done != 0` predicate.
    uint64_t report_epoch;
};

static_assert(sizeof(AicoreLaunchArgs) == 40, "AicoreLaunchArgs size drift");

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
    args.ffts_base_addr = k_args.ffts_base_addr;
    args.chip_swimlane_aicore_rotation_table = k_args.chip_swimlane_aicore_rotation_table;
}

/**
 * InitArgs - per-device runtime configuration
 *
 * Uploaded at worker init via `simpler_aicpu_init`, before any
 * register_callable/exec launch. Republished when first-use provisioning adds
 * an async-DMA workspace. The values do not ride on per-run KernelArgs; the
 * resident AICPU SO keeps the latest configuration across task launches.
 *
 * `regs` / `pmu_reg_addrs` are intentionally NOT here. The host owns both
 * tables per device context, but their addresses still reach the device on the
 * per-run KernelArgs copy the AICPU exec entry latches, so moving them would
 * change that publication protocol rather than just host-side ownership.
 */
struct InitArgs {
    uint32_t device_id{0};            // ACL device ordinal -> set_orch_device_id
    uint32_t log_level{25};           // Threshold: DEBUG=10, INFO=20, TIMING=25, WARN=30, ERROR=40, NUL=60
    int32_t scheduler_timeout_ms{0};  // AICPU no-progress watchdog (ms); 0 -> compile default
    // Per-engine async-DMA workspace dev addrs -> set_dma_workspace_addr(kind, .);
    // indexed by DmaWorkspaceKind; 0 = that engine unavailable.
    uint64_t dma_workspace_addr[DMA_WORKSPACE_KIND_COUNT]{};
    // Distance from a GM address to its nocache alias, as the driver reports it
    // for this device -> set_dev_l2_cache_offset(.). The device maps each page twice,
    // once cached and once not; `addr + offset` selects the uncached mapping, so
    // a load through it does not allocate in L2. 0 means the device exposes no
    // such alias, and `addr + 0` leaves the load ordinary and cached.
    uint64_t l2_cache_offset{0};
};

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

#endif  // PLATFORM_COMMON_KERNEL_ARGS_H_
