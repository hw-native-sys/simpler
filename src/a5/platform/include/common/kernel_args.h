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
 * @brief KernelArgs Structure - Shared between Host, AICPU, and AICore
 *
 * This structure is used to pass arguments to both AICPU and AICore kernels.
 * It contains pointers to device memory for arguments and runtime data.
 *
 * Platform Support:
 * - a5: Real hardware with CANN runtime compatibility
 * - a5sim: Host-based simulation using standard memory
 *
 * Memory Layout (a5):
 * This structure's layout is hardcoded in libaicpu_extend_kernels.so, which
 * expects specific offsets for deviceArgs fields. The unused[5] array provides
 * the required offset alignment for compatibility with the CANN runtime.
 *
 * Memory Layout (a5sim):
 * For simulation, the layout is maintained for API compatibility, though
 * we use host memory instead of device memory.
 */

#ifndef PLATFORM_COMMON_KERNEL_ARGS_H_
#define PLATFORM_COMMON_KERNEL_ARGS_H_

#include <cstdint>

// Forward declarations
class DeviceArgs;
class Runtime;

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
 * Kernel arguments structure
 *
 * This structure is passed to AICPU kernels by the host.
 *
 * Field Access Patterns:
 * - unused[5]: Padding for alignment with CANN runtime expectations
 * - device_args: Written by host, read by AICPU (contains aicpu_so_bin/aicpu_so_len)
 * - runtime_args: Written by host, read by AICPU (task runtime, includes
 *   handshake buffers)
 * - dep_gen_data_base: Written by host platform, read by AICPU platform layer;
 *   zero when dep_gen capture is unused
 *
 * Note: AICore kernels receive Runtime* directly, not KernelArgs
 *       - AICPU: accesses runtime_args->workers directly
 *       - AICore: receives Runtime* pointer with workers at offset 0
 */
struct KernelArgs {
    uint64_t unused[5] = {0};                               // Alignment padding (required by CANN runtime offset)
    DeviceArgs *device_args{nullptr};                       // Device arguments (AICPU reads, contains SO info)
    __may_used_by_aicore__ Runtime *runtime_args{nullptr};  // Task runtime in device memory
    uint64_t regs{0};                                       // Per-core register base address array (platform-specific)
    uint64_t dump_data_base{0};  // Dump shared memory base address; use explicit flags to detect enablement
    uint64_t l2_swimlane_data_base{
        0
    };  // L2 swimlane shared memory base address; use explicit flags to detect enablement
    uint64_t pmu_data_base{0};      // PMU buffer base address (device memory); 0 = PMU disabled
    uint64_t dep_gen_data_base{0};  // dep_gen shared memory base address; use explicit flags to detect enablement
    // Profiling per-core address arrays (moved out of Handshake). Each *_addrs
    // field is a device pointer to uint64_t[num_aicore]. AICore KERNEL_ENTRY
    // indexes by block_idx and forwards into per-core platform state.
    uint64_t aicore_l2_swimlane_ring_addrs{0};  // L2SwimlaneAicoreRing* per core; 0 when L2 swimlane is off
    uint64_t aicore_pmu_ring_addrs{0};          // PmuAicoreRing* per core; 0 when PMU is off
    uint64_t scope_stats_data_base{0};          // ScopeStatsBuffer device pointer; 0 when scope_stats is off.
                                                // a5 has no halHostRegister — host keeps a separate shadow and
                                                // refreshes it via rtMemcpy DEVICE_TO_HOST at dump time.
    uint32_t log_level{1};                      // Severity floor: 0=DEBUG, 1=INFO, 2=WARN, 3=ERROR, 4=NUL
    uint32_t log_info_v{5};                     // INFO verbosity threshold (0..9); default V5
    uint32_t enable_profiling_flag{0};  // Profiling umbrella bitmask; dump_tensor|l2_swimlane|pmu|dep_gen|scope_stats
    uint32_t _pad{0};                   // Alignment padding

    // Device pointer to an 8-byte buffer that the platform AICPU entry writes
    // the run-wall (ns) into. Allocated once at simpler_init, kept resident.
    // See the a2a3 kernel_args.h for the full design rationale (CANN's
    // AICPU args copy makes inline fields write-only).
    uint64_t device_wall_data_base{0};
    // ACL device ordinal. Pushed to the AICPU so the executor can suffix the
    // staged orchestration SO name (libdevice_orch_<pid>_<cid>_<device_id>.so),
    // mirroring the per-device simpler_inner preinstall fix. Trailing field —
    // keeps the CANN-fixed front offsets and AICore-read fields in place.
    uint32_t device_id{0};
};

#ifdef __cplusplus
}
#endif

#endif  // PLATFORM_COMMON_KERNEL_ARGS_H_
