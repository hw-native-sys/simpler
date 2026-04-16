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
 * SPMD Sync-Start Orchestration
 *
 * Submits MIX tasks with require_sync_start=true to verify that the scheduler
 * atomically launches all blocks before any can run.
 *
 * Tasks:
 *   T0: block_num=2,  require_sync_start=true   (basic sync launch)
 *   T1: block_num=8,  require_sync_start=true   (larger batch)
 *   T2: block_num=2,  require_sync_start=false  (normal, as baseline)
 *   T3: block_num=12, require_sync_start=true   (cross-thread batch)
 *
 * Each block writes float(block_idx) to its allocated cache-line slot,
 * identical to spmd_multiblock_mix so the same kernel binaries can be reused.
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

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    // T0: 2 blocks, sync_start=true  (6 CL)
    submit_mix(ext_output, 2, 0, true);
    // T1: 8 blocks, sync_start=true  (24 CL)
    submit_mix(ext_output, 8, 6, true);
    // T2: 2 blocks, sync_start=false (6 CL, baseline)
    submit_mix(ext_output, 2, 30, false);
    // T3: 12 blocks, sync_start=true (36 CL)
    submit_mix(ext_output, 12, 36, true);

    LOG_ALWAYS("[spmd_sync_start] Submitted 4 tasks (3 sync_start + 1 baseline)");
}

}  // extern "C"
