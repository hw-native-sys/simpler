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
#include "aicpu/l2_perf_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "runtime.h"

// Forward declaration of aicpu_execute (implemented in aicpu_executor.cpp)
extern "C" int aicpu_execute(Runtime *arg);

extern "C" __attribute__((visibility("default"))) int StaticTileFwkBackendKernelServer(void *arg) {
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    return 0;
}

/**
 * AICPU kernel initialization entry point
 *
 * This function is called once during kernel initialization by the CANN
 * runtime. It initializes logging and validates kernel arguments.
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to KernelArgs structure
 * @return 0 on success, -1 on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServerInit(void *arg) {
    init_log_switch();
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }
    auto k_args = (KernelArgs *)arg;
    set_log_level(static_cast<int>(k_args->log_level));
    set_log_info_v(static_cast<int>(k_args->log_info_v));

    LOG_INFO_V0("%s", "Runtime Executor Init: Initializing AICPU kernel");
    return 0;
}

/**
 * AICPU kernel main execution entry point
 *
 * This is the main entry point for the AICPU runtime executor kernel.
 * It extracts the Runtime from KernelArgs and delegates to AicpuExecute.
 *
 * Note: Function name is hardcoded in libaicpu_extend_kernels.so
 *
 * @param arg Pointer to KernelArgs structure containing runtime_args
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int DynTileFwkBackendKernelServer(void *arg) {
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
    set_platform_dump_base(k_args->dump_data_base);
    set_dump_tensor_enabled(GET_PROFILING_FLAG(runtime->workers[0].enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR));
    set_platform_l2_perf_base(k_args->l2_perf_data_base);
    set_l2_swimlane_enabled(GET_PROFILING_FLAG(runtime->workers[0].enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE));
    set_platform_pmu_base(k_args->pmu_data_base);
    set_pmu_enabled(GET_PROFILING_FLAG(runtime->workers[0].enable_profiling_flag, PROFILING_FLAG_PMU));

    // Affinity gate: drop excess threads before entering runtime
    if (!platform_aicpu_affinity_gate(runtime->sche_cpu_num, PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH)) {
        LOG_INFO_V0("Thread dropped by cluster affinity");
        return 0;
    }

    LOG_INFO_V0("%s", "DynTileFwkBackendKernelServer: Calling aicpu_execute with Runtime");
    int rc = aicpu_execute(runtime);
    if (rc != 0) {
        LOG_ERROR("DynTileFwkBackendKernelServer: aicpu_execute failed with rc=%d", rc);
        return rc;
    }
    LOG_INFO_V0("%s", "DynTileFwkBackendKernelServer: aicpu_execute completed successfully");

    return rc;
}
