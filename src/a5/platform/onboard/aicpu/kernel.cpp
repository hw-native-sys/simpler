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
#include "aicpu/device_log.h"
#include "aicpu/device_time.h"
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "runtime.h"

// Run-wall capture: g_device_start_cycle is set once in
// simpler_aicpu_init (single-threaded launch); each thread
// of the multi-threaded simpler_aicpu_exec writes the converted
// (end - start) into KernelArgs.device_wall_ns on exit. Plain stores —
// last-writer-wins is fine for wall measurement.
static uint64_t g_device_start_cycle = 0;

// Forward declaration of aicpu_execute (implemented in aicpu_executor.cpp)
extern "C" int aicpu_execute(Runtime *arg);

/**
 * AICPU kernel initialization entry point.
 *
 * Called once per run by the main aicpu_scheduler. Host registers this SO
 * via `rtsBinaryLoadFromFile` (Mode B JSON load, cpuKernelMode=0) and
 * resolves this symbol via `rtsFuncGetByName`; the per-task launch goes
 * through `rtsLaunchCpuKernel` on the cached `rtFuncHandle`. The bootstrap
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

    // Init is launched single-threaded (block_dim=1) — race-free spot to
    // capture run start and reset the device_wall buffer.
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
 * `rtFuncHandle` resolved during host-side Mode B init (see
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

    // Store platform regs before calling aicpu_execute. Profiling enable
    // bits live on KernelArgs::enable_profiling_flag now (no longer
    // mirrored into Handshake), so decode the umbrella bitmask once and
    // hand it to the existing platform-state setters.
    set_platform_regs(k_args->regs);
    set_platform_dump_base(k_args->dump_data_base);
    set_dump_tensor_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR));
    set_platform_l2_perf_base(k_args->l2_perf_data_base);
    set_l2_swimlane_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE));
    set_platform_pmu_base(k_args->pmu_data_base);
    set_pmu_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_PMU));

    // Affinity gate: drop excess threads before entering runtime
    if (!platform_aicpu_affinity_gate(runtime->sche_cpu_num, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)) {
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

    // Stamp end into the device_wall buffer. Last-writer-wins across threads.
    uint64_t my_end = get_sys_cnt_aicpu();
    if (k_args->device_wall_data_base != 0 && my_end > g_device_start_cycle) {
        *reinterpret_cast<uint64_t *>(k_args->device_wall_data_base) =
            static_cast<uint64_t>(cycles_to_us(my_end - g_device_start_cycle) * 1000.0);
    }

    return rc;
}
