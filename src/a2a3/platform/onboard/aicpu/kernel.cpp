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
#include "aicpu/l2_swimlane_collector_aicpu.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "runtime.h"

// Run-wall capture: the host allocates a device buffer addressed by
// KernelArgs.device_wall_data_base holding one { start_cycle, end_cycle } pair
// per launched AICPU thread (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH pairs,
// raw sys-counter cycles), and resets it to { UINT64_MAX, 0 } before each run.
// Every surviving simpler_aicpu_exec thread writes its own start/end into its
// own slot (indexed by platform_aicpu_affinity_thread_idx()) — plain stores,
// no cross-thread atomics. The host reads the whole array and reduces once:
// wall = max(end) - min(start). No single-threaded pre-pass is needed to
// seed the start.

// Forward declaration of aicpu_execute (implemented in aicpu_executor.cpp)
extern "C" int aicpu_execute(Runtime *arg);
extern "C" int aicpu_prewarm_callable(Runtime *arg);

/**
 * AICPU kernel main execution entry point.
 *
 * Called per-thread by the main aicpu_scheduler. Host registers this SO via
 * `rtsBinaryLoadFromFile` (JSON load, cpuKernelMode=0) and resolves this
 * symbol via `rtsFuncGetByName`; each per-task launch goes through
 * `rtsLaunchCpuKernel` on the cached `rtFuncHandle`. The bootstrap dispatcher
 * only writes this SO to the preinstall path — it does not dlsym this symbol
 * itself.
 *
 * @param arg Pointer to KernelArgs structure containing runtime_args
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *arg) {
    // Snapshot CANN log severity. Idempotent across the concurrent exec
    // threads — same snapshot value.
    init_log_switch();
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
    set_dump_args_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_DUMP_TENSOR));
    set_platform_l2_swimlane_base(k_args->l2_swimlane_data_base);
    set_platform_l2_swimlane_aicore_rotation_table(k_args->l2_swimlane_aicore_rotation_table);
    set_l2_swimlane_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_L2_SWIMLANE));
    set_platform_pmu_base(k_args->pmu_data_base);
    set_platform_pmu_reg_addrs(k_args->pmu_reg_addrs);
    set_pmu_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_PMU));
    set_platform_dep_gen_base(k_args->dep_gen_data_base);
    set_dep_gen_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_DEP_GEN));
    set_scope_stats_enabled(GET_PROFILING_FLAG(k_args->enable_profiling_flag, PROFILING_FLAG_SCOPE_STATS));
    set_platform_scope_stats_base(k_args->scope_stats_data_base);

    // Filter-style affinity gate. Host computed ALLOWED_CPUS from AICPU
    // OCCUPY and wrote it into Runtime; the device side only matches
    // sched_getcpu() against that table and exposes the table position as
    // exec_idx.
    if (runtime->aicpu_allowed_cpu_count <= 0 || runtime->aicpu_launch_count <= 0) {
        LOG_ERROR(
            "AICPU affinity inputs missing: allowed_cpu_count=%d launch_count=%d (host probe must run before exec)",
            runtime->aicpu_allowed_cpu_count, runtime->aicpu_launch_count
        );
        return -1;
    }
    if (!platform_aicpu_affinity_gate_filter(
            runtime->aicpu_allowed_cpus, runtime->aicpu_allowed_cpu_count, runtime->aicpu_launch_count
        )) {
        LOG_INFO_V0("Thread dropped by filter affinity gate");
        return 0;
    }

    // Run-wall: record this thread's start into its own slot (plain store,
    // no cross-thread contention). Slot = affinity-gate exec index.
    const int32_t wall_slot = platform_aicpu_affinity_thread_idx();
    uint64_t *const wall = reinterpret_cast<uint64_t *>(k_args->device_wall_data_base);
    const bool wall_ok = wall != nullptr && wall_slot >= 0 && wall_slot < PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH;
    if (wall_ok) {
        wall[wall_slot * 2] = get_sys_cnt_aicpu();
    }

    LOG_INFO_V0("%s", "simpler_aicpu_exec: Calling aicpu_execute with Runtime");
    int rc = aicpu_execute(runtime);
    if (rc != 0) {
        LOG_ERROR("simpler_aicpu_exec: aicpu_execute failed with rc=%d", rc);
        return rc;
    }
    LOG_INFO_V0("%s", "simpler_aicpu_exec: aicpu_execute completed successfully");

    // Run-wall: record this thread's end into its own slot (plain store).
    // Host reduces max(end) - min(start) → ns (see wall-capture note above).
    if (wall_ok) {
        wall[wall_slot * 2 + 1] = get_sys_cnt_aicpu();
    }

    return rc;
}

extern "C" __attribute__((visibility("default"))) int simpler_aicpu_prewarm_callable(void *arg) {
    init_log_switch();
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid prewarm kernel arguments: null pointer");
        return -1;
    }

    auto k_args = (KernelArgs *)arg;
    Runtime *runtime = k_args->runtime_args;
    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid prewarm runtime_args: null pointer");
        return -1;
    }

    set_log_level(static_cast<int>(k_args->log_level));
    set_log_info_v(static_cast<int>(k_args->log_info_v));
    set_orch_device_id(static_cast<int>(k_args->device_id));

    LOG_INFO_V0("%s", "simpler_aicpu_prewarm_callable: prewarming callable");
    int rc = aicpu_prewarm_callable(runtime);
    if (rc != 0) {
        LOG_ERROR("simpler_aicpu_prewarm_callable: prewarm failed with rc=%d", rc);
        return rc;
    }
    LOG_INFO_V0("%s", "simpler_aicpu_prewarm_callable: prewarm completed");
    return 0;
}
