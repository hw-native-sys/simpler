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
 * SPMD Multi-Block MIX Orchestration
 *
 * Submits three MIX tasks (AIC + AIV0 + AIV1) with increasing block_num:
 *   T0: block_num=2   — basic multi-block MIX
 *   T1: block_num=8   — saturates one sched thread (8 clusters)
 *   T2: block_num=12  — forces cross-thread re-push via ready_queue
 *
 * Each task writes to a disjoint region of the output tensor using the
 * base_cl scalar to offset the block writes.
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

static void
submit_spmd_mix(int32_t aic_id, int32_t aiv0_id, int32_t aiv1_id, Tensor &out, int16_t block_num, int64_t base_cl) {
    MixedKernels mk;
    mk.aic_kernel_id = aic_id;
    mk.aiv0_kernel_id = aiv0_id;
    mk.aiv1_kernel_id = aiv1_id;

    Arg args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_core_num(block_num);
    pto2_rt_submit_task(mk, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    // T0: 2 blocks (6 CL) — basic multi-block MIX
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, 2, 0);

    // T1: 8 blocks (24 CL) — saturate one sched thread's clusters
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, 8, 6);

    // T2: 12 blocks (36 CL) — cross-thread dispatch via ready_queue re-push
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, 12, 30);

    // T3: 24 blocks (72 CL) — occupy all clusters across all 3 sched threads
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, 24, 66);

    // T4: 48 blocks (144 CL) — two full rounds of all clusters
    submit_spmd_mix(FUNC_SPMD_MIX_AIC, FUNC_SPMD_MIX_AIV0, FUNC_SPMD_MIX_AIV1, ext_output, 48, 138);

    LOG_ALWAYS("[spmd_multiblock_mix] Submitted 5 MIX tasks: block_num=2,8,12,24,48");
}

}  // extern "C"
