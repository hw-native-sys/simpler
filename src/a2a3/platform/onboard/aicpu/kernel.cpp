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
#include <cstdio>

#include "common/unified_log.h"
#include "common/kernel_args.h"
#include "common/platform_config.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "runtime.h"

// Run-wall capture: g_device_start_cycle is set once in
// simpler_aicpu_init (single-threaded launch); each thread
// of the multi-threaded simpler_aicpu_exec writes the converted
// (end - start) into KernelArgs.device_wall_ns on exit. Plain stores —
// last-writer-wins is fine for wall measurement (concurrent exiting threads'
// `my_end` values differ by µs, the final overwrite is within benchmark
// noise of the true max-end - min-start).
static uint64_t g_device_start_cycle = 0;

// Forward declaration of aicpu_execute (implemented in aicpu_executor.cpp)
extern "C" int aicpu_execute(Runtime *arg);

/**
 * AICPU kernel initialization entry point.
 *
 * Called once per run by the main aicpu_scheduler. Host registers this SO
 * via `rtsBinaryLoadFromFile` (JSON load, cpuKernelMode=0) and resolves
 * this symbol via `rtsFuncGetByName`; the per-task launch goes through
 * `rtsLaunchCpuKernel` on the cached `rtFuncHandle`. The bootstrap
 * dispatcher only writes this SO to the preinstall path — it does not
 * dlsym these symbols itself.
 *
 * @param arg Pointer to KernelArgs structure
 * @return 0 on success, -1 on error
 */
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *arg) {
    init_log_switch();
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }
    auto k_args = (KernelArgs *)arg;
    set_log_level(static_cast<int>(k_args->log_level));
    set_log_info_v(static_cast<int>(k_args->log_info_v));

    // Init is launched single-threaded (block_dim=1), so the race-free spot
    // to capture run start and reset the wall accumulator. Subsequent
    // simpler_aicpu_exec threads stamp end on their way out, via
    // the device-resident 8-byte buffer addressed by device_wall_data_base.
    g_device_start_cycle = get_sys_cnt_aicpu();
    if (k_args->device_wall_data_base != 0) {
        *reinterpret_cast<uint64_t *>(k_args->device_wall_data_base) = 0;
    }

    LOG_INFO_V0("%s", "Runtime Executor Init: Initializing AICPU kernel");
    return 0;
}

/**
 * AICPU kernel main execution entry point.
 *
 * Called per-thread by the main aicpu_scheduler via the cached
 * `rtFuncHandle` resolved during host-side init (see
 * `simpler_aicpu_init` docstring for the load path).
 *
 * @param arg Pointer to KernelArgs structure containing runtime_args
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *arg) {
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    // Extract Runtime from KernelArgs
    auto k_args = (KernelArgs *)arg;
    Runtime *runtime = k_args->runtime_args;

    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid runtime_args: null pointer");
        return -1;
    }

    // Push host-published log config into device globals.
    set_log_level(static_cast<int>(k_args->log_level));
    set_log_info_v(static_cast<int>(k_args->log_info_v));

    // Store platform regs before calling aicpu_execute
    // Dump enable is an execution control flag propagated via handshake.
    // The dump base address is only the backing storage location.
    set_platform_regs(k_args->regs);
    // Device ordinal for per-device orchestration-SO naming in the executor.
    set_orch_device_id(static_cast<int>(k_args->device_id));
    set_platform_dump_base(k_args->dump_data_base);
    set_dump_tensor_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR));
    set_platform_l2_perf_base(k_args->l2_perf_data_base);
    set_platform_aicore_rotation_table(k_args->aicore_ring_addr);
    set_l2_swimlane_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE));
    set_platform_pmu_base(k_args->pmu_data_base);
    set_platform_pmu_reg_addrs(k_args->pmu_reg_addrs);
    set_pmu_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_PMU));
    set_platform_dep_gen_base(k_args->dep_gen_data_base);
    set_dep_gen_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_DEP_GEN));
    set_scope_stats_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_SCOPE_STATS));
    set_platform_scope_stats_base(k_args->scope_stats_data_base);

    // Affinity gate: drop excess threads before entering runtime
    if (!platform_aicpu_affinity_gate(runtime->aicpu_thread_num, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)) {
        LOG_INFO_V0("Thread dropped by cluster affinity");
        return 0;
    }

    LOG_INFO_V0("%s", "simpler_aicpu_exec: Calling aicpu_execute with Runtime");
    int rc = aicpu_execute(runtime);
    if (rc != 0) {
        LOG_ERROR("simpler_aicpu_exec: aicpu_execute failed with rc=%d", rc);
        return rc;
    }
    LOG_INFO_V0("%s", "simpler_aicpu_exec: aicpu_execute completed successfully");

    // Stamp end into the device_wall buffer (addressed via
    // device_wall_data_base). Last-writer-wins across threads — wall
    // measurements are µs-scale tolerant, see g_device_start_cycle comment.
    uint64_t my_end = get_sys_cnt_aicpu();
    if (k_args->device_wall_data_base != 0 && my_end > g_device_start_cycle) {
        *reinterpret_cast<uint64_t *>(k_args->device_wall_data_base) =
            static_cast<uint64_t>(cycles_to_us(my_end - g_device_start_cycle) * 1000.0);
    }

    return rc;
}
