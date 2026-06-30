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

#ifndef PLATFORM_COMMON_KERNEL_ARGS_H_
#define PLATFORM_COMMON_KERNEL_ARGS_H_

#include <cstddef>
#include <cstdint>

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
 *       - AICore: receives device KernelArgs* via KERNEL_ENTRY
 */
struct KernelArgs {
    __may_used_by_aicore__ Runtime *runtime_args{nullptr};  // Task runtime in device memory
    uint64_t regs{0};                                       // Per-core register base address array (platform-specific)
    uint64_t dump_data_base{0};  // Dump shared memory base address; use explicit flags to detect enablement
    // L2 swimlane shared memory base address; use explicit flags to detect enablement
    uint64_t l2_swimlane_data_base{0};
    uint64_t pmu_data_base{0};      // PMU buffer base address (device memory); 0 = PMU disabled
    uint64_t dep_gen_data_base{0};  // dep_gen shared memory base address; use explicit flags to detect enablement
    // Profiling per-core address arrays (moved out of Handshake). Each *_addrs
    // field is a device pointer to uint64_t[num_aicore]. AICore KERNEL_ENTRY
    // indexes by block_idx and forwards into per-core platform state.
    // L2SwimlaneActiveHead* per core (rotation channel); 0 when L2 swimlane is off
    uint64_t l2_swimlane_aicore_rotation_table{0};
    uint64_t aicore_pmu_ring_addrs{0};  // PmuAicoreRing* per core; 0 when PMU is off
    uint64_t scope_stats_data_base{0};  // ScopeStatsBuffer device pointer; 0 when scope_stats is off.
                                        // a5 has no halHostRegister — host keeps a separate shadow and
                                        // refreshes it via rtMemcpy DEVICE_TO_HOST at dump time.
    uint32_t enable_profiling_flag{0};  // Profiling umbrella bitmask; dump_tensor|l2_swimlane|pmu|dep_gen|scope_stats
    uint32_t _pad{0};                   // Alignment padding

    // Device pointer to an 8-byte buffer that the platform AICPU entry writes
    // the run-wall (ns) into. Allocated once at simpler_init, kept resident.
    // See the a2a3 kernel_args.h for the full design rationale (CANN's
    // AICPU args copy makes inline fields write-only).
    uint64_t device_wall_data_base{0};
    // Opaque always-false guard read by the AICore SIMT meta anchor (AIV
    // KERNEL_ENTRY). The host never sets it non-zero; its only purpose is to be
    // a runtime-valued condition the compiler cannot constant-fold, so the
    // never-executed SIMT launch in simt_anchor.h survives DCE and bisheng
    // still classifies the entry as SIMT. Keep it last (trailing field).
    uint32_t force_simt_anchor{0};
};

static_assert(offsetof(KernelArgs, runtime_args) == 0, "KernelArgs::runtime_args offset drift");
static_assert(offsetof(KernelArgs, regs) == 8, "KernelArgs::regs offset drift");

/**
 * InitArgs - per-device one-shot invariants
 *
 * Uploaded once at worker init via the `simpler_aicpu_init` entry, before any
 * register_callable/exec launch. Carries the values fixed for the lifetime of
 * the device context (orch device id, log config) so they no longer ride on
 * the per-run KernelArgs: latched once into the resident AICPU SO globals and
 * surviving every subsequent per-task launch.
 *
 * `regs` is intentionally NOT here — on a5 the per-core register table is also
 * read by the AICore KERNEL_ENTRY off the per-run device KernelArgs copy, so it
 * stays in KernelArgs.
 */
struct InitArgs {
    uint32_t device_id{0};   // ACL device ordinal -> set_orch_device_id
    uint32_t log_level{1};   // Severity floor: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NUL
    uint32_t log_info_v{5};  // INFO verbosity threshold (0..9); default V5
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
