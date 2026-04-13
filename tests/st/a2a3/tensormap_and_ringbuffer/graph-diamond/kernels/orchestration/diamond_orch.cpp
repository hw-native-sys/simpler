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
 * Diamond (Fork-Join) Orchestration (tensormap_and_ringbuffer Runtime)
 *
 * Builds a diamond DAG: A -> {B_0, B_1, ..., B_{W-1}} -> D
 *
 *   seed -> [Source A] -> a_out -> [Branch B_0] -> b_out_0 -.
 *                               -> [Branch B_1] -> b_out_1 -+-> [Merge D] -> result
 *                               -> ...                       |
 *                               -> [Branch B_{W-1}] -> b_out_{W-1} -'
 *
 * Source A is always AIV. Merge D is always AIV (NOOP kernel).
 * Branch tasks vary by mode:
 *   mode=0: All AIV branches
 *   mode=1: All AIC branches
 *   mode=2: Alternating AIC/AIV branches (even=AIC, odd=AIV)
 *
 * Tests: fan-out + fan-in combined, the most common real DAG pattern.
 * With mixed modes, also tests cross-core-type dependency coordination.
 *
 * Arg layout: [seed, result, width, mode]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_INC_AIC 0
#define FUNC_INC_AIV 1
#define FUNC_NOOP_AIV 2

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 4,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index) {
    if (orch_thread_index != 0) {
        return;
    }

    Tensor seed = from_tensor_arg(orch_args.tensor(0));
    Tensor result = from_tensor_arg(orch_args.tensor(1));
    int width = static_cast<int>(orch_args.scalar(0));
    int mode = static_cast<int>(orch_args.scalar(1));

    LOG_ALWAYS("[diamond] width=%d, mode=%d (0=AIV, 1=AIC, 2=mixed)", width, mode);

    uint32_t scalar_shape[1] = {1};
    TensorCreateInfo ci(scalar_shape, 1, DataType::FLOAT32);

    // Source A (always AIV): seed -> a_out
    Arg src_params;
    src_params.add_input(seed);
    src_params.add_output(ci);
    TaskOutputTensors source_outs = pto2_rt_submit_aiv_task(FUNC_INC_AIV, src_params);

    // Build merge args incrementally
    Arg merge_params;
    merge_params.add_inout(result);

    // Branch tasks: each reads source output, produces branch output
    for (int i = 0; i < width; i++) {
        Arg bp;
        bp.add_input(source_outs.get_ref(0));
        bp.add_output(ci);

        TaskOutputTensors branch_outs;
        if (mode == 0) {
            // All AIV
            branch_outs = pto2_rt_submit_aiv_task(FUNC_INC_AIV, bp);
        } else if (mode == 1) {
            // All AIC
            branch_outs = pto2_rt_submit_aic_task(FUNC_INC_AIC, bp);
        } else {
            // Mixed: even=AIC, odd=AIV
            if (i % 2 == 0) {
                branch_outs = pto2_rt_submit_aic_task(FUNC_INC_AIC, bp);
            } else {
                branch_outs = pto2_rt_submit_aiv_task(FUNC_INC_AIV, bp);
            }
        }

        merge_params.add_input(branch_outs.get_ref(0));
    }

    // Merge D (always AIV): waits for all branches, increments result
    pto2_rt_submit_aiv_task(FUNC_NOOP_AIV, merge_params);

    LOG_ALWAYS("[diamond] Submitted 1 source + %d branches + 1 merge task", width);
}

}  // extern "C"
