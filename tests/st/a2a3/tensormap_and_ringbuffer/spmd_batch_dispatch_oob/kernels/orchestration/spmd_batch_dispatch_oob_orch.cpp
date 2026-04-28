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
 * Regression test for batch dispatch OOB (issue #565).
 *
 * Submits two MIX tasks with block_num=48 back-to-back so they are both
 * in the ready queue when the scheduler runs pop_ready_tasks_batch.
 *
 * Args layout: [output]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"

#define FUNC_AIC 0
#define FUNC_AIV0 1
#define FUNC_AIV1 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig
aicpu_orchestration_config(const ChipStorageTaskArgs &orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 1,
    };
}

static void submit_spmd_mix(Tensor &out, int16_t block_num, int64_t base_cl) {
    MixedKernels mk;
    mk.aic_kernel_id = FUNC_AIC;
    mk.aiv0_kernel_id = FUNC_AIV0;
    mk.aiv1_kernel_id = FUNC_AIV1;

    Arg args;
    args.add_inout(out);
    args.add_scalar(base_cl);
    args.launch_spec.set_block_num(block_num);
    rt_submit_task(mk, args);
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(const ChipStorageTaskArgs &orch_args) {
    Tensor ext_output = from_tensor_arg(orch_args.tensor(0));

    // Two back-to-back tasks with block_num=48 (2x cluster count).
    // Both land in the ready queue simultaneously, triggering got=2 in
    // pop_ready_tasks_batch — the scenario that causes OOB without the fix.
    submit_spmd_mix(ext_output, 48, 0);
    submit_spmd_mix(ext_output, 48, 144);

    LOG_ALWAYS("[spmd_batch_dispatch_oob] Submitted 2 MIX tasks: block_num=48,48");
}

}  // extern "C"
