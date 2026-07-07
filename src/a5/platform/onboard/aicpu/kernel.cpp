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
#include "aicpu/aicpu_device_config.h"
#include "aicpu/dep_gen_collector_aicpu.h"
#include "aicpu/device_log.h"
#include "aicpu/device_phase_aicpu.h"
#include "aicpu/device_time.h"
#include "aicpu/l2_swimlane_collector_aicpu.h"
#include "aicpu/platform_regs.h"
#include "aicpu/platform_aicpu_affinity.h"
#include "aicpu/pmu_collector_aicpu.h"
#include "aicpu/scope_stats_collector_aicpu.h"
#include "aicpu/tensor_dump_aicpu.h"
#include "runtime.h"

// Run-wall capture: the host allocates a device buffer addressed by
// KernelArgs.device_wall_data_base holding one { start_cycle, end_cycle } pair
// per launched AICPU thread (PLATFORM_MAX_AICPU_THREADS_JUST_FOR_LAUNCH pairs,
// raw sys-counter cycles), and resets it to { UINT64_MAX, 0 } before each run.
// Every surviving simpler_aicpu_exec thread writes its own start/end into its
// own slot (indexed by platform_aicpu_affinity_thread_idx()) — plain stores,
// no cross-thread atomics. The host reads the array and reduces once:
// wall = max(end) - min(start). No single-threaded pre-pass is needed to
// seed the start.

// Forward declaration of aicpu_execute (implemented in aicpu_executor.cpp).
// simpler_aicpu_register_callable is NOT declared/forwarded here: it is
// exported directly by the TMARB runtime (host_build_graph does not export it).
extern "C" int aicpu_execute(Runtime *arg);

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
 * @param arg Pointer to the front-less KernelArgs payload (runtime_args @ 0)
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_exec(void *arg) {
    // Log severity was snapshot once by simpler_aicpu_init at worker init; the
    // resident SO keeps it across launches, so exec does not re-snapshot.
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid kernel arguments: null pointer");
        return -1;
    }

    KernelArgs *k_args = reinterpret_cast<KernelArgs *>(arg);
    Runtime *runtime = k_args->runtime_args;

    if (runtime == nullptr) {
        LOG_ERROR("%s", "Invalid runtime_args: null pointer");
        return -1;
    }

    // Per-device invariants (log config, orch device id) were latched once by
    // simpler_aicpu_init at worker init. Profiling enable bits live on
    // KernelArgs::enable_profiling_flag now (no longer mirrored into
    // Handshake), so decode the umbrella bitmask once and hand it to the
    // existing platform-state setters.
    set_platform_regs(k_args->regs);
    set_platform_dump_base(k_args->dump_data_base);
    set_dump_args_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_DUMP_TENSOR));
    set_platform_l2_swimlane_base(k_args->l2_swimlane_data_base);
    set_platform_l2_swimlane_aicore_rotation_table(k_args->l2_swimlane_aicore_rotation_table);
    set_l2_swimlane_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_L2_SWIMLANE));
    set_platform_pmu_base(k_args->pmu_data_base);
    set_pmu_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_PMU));
    set_platform_dep_gen_base(k_args->dep_gen_data_base);
    set_dep_gen_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_DEP_GEN));
    set_scope_stats_enabled(SIMPLER_GET_DFX_FLAG(k_args->enable_profiling_flag, SIMPLER_DFX_FLAG_SCOPE_STATS));
    set_platform_scope_stats_base(k_args->scope_stats_data_base);

    // Filter-style affinity gate (a5). Host probed the topology, computed
    // ALLOWED_CPUS, and wrote it into runtime->aicpu_allowed_cpus[]. The
    // gate barriers exactly runtime->aicpu_launch_count threads (= the
    // count CANN was told to launch, which equals popcount(OCCUPY) — see
    // src/a5/platform/onboard/host/device_runner.cpp), keeps those whose
    // sched_getcpu() ∈ allowed_cpus, and exposes the deterministic
    // exec_idx via platform_aicpu_affinity_thread_idx() — the executor
    // reads it to assign sched/orch role.
    //
    // If the host probe didn't populate the gate inputs (allowed_count or
    // launch_count is 0) the gate would unconditionally drop every thread
    // and we'd silently return success without ever calling aicpu_execute.
    // That's a host-side bug; fail loud so it surfaces instead of
    // producing nothing at runtime.
    if (runtime->get_aicpu_allowed_cpu_count() <= 0 || runtime->get_aicpu_launch_count() <= 0) {
        LOG_ERROR(
            "AICPU affinity inputs missing: allowed_cpu_count=%d launch_count=%d (host probe must run before exec)",
            runtime->get_aicpu_allowed_cpu_count(), runtime->get_aicpu_launch_count()
        );
        return -1;
    }
    if (!platform_aicpu_affinity_gate_filter(
            runtime->get_aicpu_allowed_cpus(), runtime->get_aicpu_allowed_cpu_count(), runtime->get_aicpu_launch_count()
        )) {
        LOG_INFO_V0("Thread dropped by filter affinity gate");
        return 0;
    }

    // Publish the phase-buffer base so the finer preamble/so_load/graph_build/
    // post_orch + orch/sched phases stamped inside aicpu_execute / the scheduler
    // resolve their per-thread slot via platform_aicpu_affinity_thread_idx()
    // (no C++ thread_local — see docs/dynamic-linking.md). Idempotent across the
    // concurrent exec threads (same base). Run-wall is stamped here.
    set_platform_phase_base(k_args->device_wall_data_base);
    AicpuPhaseScope run_wall(AicpuPhase::RunWall);

    LOG_INFO_V0("%s", "simpler_aicpu_exec: Calling aicpu_execute with Runtime");
    int rc = aicpu_execute(runtime);
    if (rc != 0) {
        LOG_ERROR("simpler_aicpu_exec: aicpu_execute failed with rc=%d", rc);
        return rc;
    }
    LOG_INFO_V0("%s", "simpler_aicpu_exec: aicpu_execute completed successfully");

    // Run-wall end is stamped by run_wall's destructor (covers the early return
    // above too); host reduces max(end) - min(start) → ns.
    return rc;
}

/**
 * AICPU per-device init entry point.
 *
 * Launched once at worker init (before any register_callable / exec), this
 * latches the per-device invariants — log config and orchestration device id —
 * into the resident AICPU SO globals. Because the inner SO stays dlopen'd in
 * the AICPU OS process across launches, these globals survive every subsequent
 * per-task launch, so exec / register_callable no longer re-push them.
 *
 * @param arg Pointer to an InitArgs payload
 * @return 0 on success, non-zero on error
 */
extern "C" __attribute__((visibility("default"))) int simpler_aicpu_init(void *arg) {
    init_log_switch();
    if (arg == nullptr) {
        LOG_ERROR("%s", "Invalid init kernel arguments: null pointer");
        return -1;
    }

    InitArgs *init_args = reinterpret_cast<InitArgs *>(arg);
    set_log_level(static_cast<int>(init_args->log_level));
    set_log_info_v(static_cast<int>(init_args->log_info_v));
    set_orch_device_id(static_cast<int>(init_args->device_id));
    set_scheduler_timeout_ms(static_cast<int>(init_args->scheduler_timeout_ms));

    LOG_INFO_V0("%s", "simpler_aicpu_init: per-device invariants latched");
    return 0;
}
