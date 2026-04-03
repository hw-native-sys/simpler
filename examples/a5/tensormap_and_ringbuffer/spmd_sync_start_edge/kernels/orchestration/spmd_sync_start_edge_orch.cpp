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
 * SPMD Sync-Start Boundary Orchestration
 *
 * Tests edge-case block_num values relative to per-thread cluster capacity
 * (8 clusters per sched thread, 24 total clusters).
 *
 * Tasks:
 *   T0: block_num=1,  sync_start=true   (degenerate: always fast path)
 *   T1: block_num=8,  sync_start=true   (exactly one thread's capacity)
 *   T2: block_num=9,  sync_start=true   (one over: must enter drain)
 *   T3: block_num=23, sync_start=true   (max valid: total_clusters - 1)
 *   T4: block_num=1,  sync_start=false  (baseline)
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_SPMD_MIX_AIC 0
#define FUNC_SPMD_MIX_AIV0 1
#define FUNC_SPMD_MIX_AIV1 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;  // NOLINT(readability/casting)
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static void submit_mix(Tensor &out, int16_t block_num, int64_t base_cl, bool sync_start) {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_SPMD_MIX_AIC;
    mk.aiv0_kernel_id = FUNC_SPMD_MIX_AIV0;
    mk.aiv1_kernel_id = FUNC_SPMD_MIX_AIV1;

    Arg args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_core_num(block_num);
    args.launch_spec.set_require_sync_start(sync_start);
    pto2_rt_submit_task(mk, args);
}

__attribute__((visibility("default"))) void
aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args, int orch_thread_num, int orch_thread_index) {
    (void)orch_thread_num;  // NOLINT(readability/casting)
    if (orch_thread_index != 0) return;

    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    // T0: block_num=1, sync_start=true (degenerate: always fast path, 3 CL)
    submit_mix(ext_output, 1, 0, true);
    // T1: block_num=8, sync_start=true (exactly one thread's cluster capacity, 24 CL)
    submit_mix(ext_output, 8, 3, true);
    // T2: block_num=9, sync_start=true (one over single thread -> must drain, 27 CL)
    submit_mix(ext_output, 9, 27, true);
    // T3: block_num=23, sync_start=true (max valid = total_clusters - 1, 69 CL)
    submit_mix(ext_output, 23, 54, true);
    // T4: block_num=1, sync_start=false (baseline, 3 CL)
    submit_mix(ext_output, 1, 123, false);

    LOG_ALWAYS("[spmd_sync_start_edge] Submitted 5 tasks: block_num=1,8,9,23 (sync) + 1 (baseline)");
}

}  // extern "C"
