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
 * Chain Orchestration (tensormap_and_ringbuffer Runtime)
 *
 * Builds a linear dependency chain of N tasks:
 *   seed -> [Task_0] -> intermediate_0 -> [Task_1] -> ... -> [Task_{N-1}] -> result
 *
 * Each task reads its input and writes output = input + 1.0.
 * After N tasks, result = N.0 (starting from seed = 0.0).
 *
 * Tasks 0..N-2 produce runtime-allocated intermediate tensors (OUTPUT).
 * Task N-1 writes to the external result tensor (INOUT).
 * This tests the runtime's INPUT->OUTPUT dependency resolution across a chain.
 *
 * Arg layout: [seed, result, chain_len]
 */

#include <stddef.h>
#include <stdint.h>

#include "pto_orchestration_api.h"  // NOLINT(build/include_subdir)

#define FUNC_INC 0

extern "C" {

__attribute__((visibility("default"))) PTO2OrchestrationConfig aicpu_orchestration_config(
    const ChipStorageTaskArgs& orch_args) {
    (void)orch_args;
    return PTO2OrchestrationConfig{
        .expected_arg_count = 3,
    };
}

__attribute__((visibility("default"))) void aicpu_orchestration_entry(
    const ChipStorageTaskArgs& orch_args, int orch_thread_num, int orch_thread_index) {
    if (orch_thread_index != 0) {
        return;
    }

    Tensor seed = from_tensor_arg(orch_args.tensor(0));
    Tensor result = from_tensor_arg(orch_args.tensor(1));
    int chain_len = static_cast<int>(orch_args.scalar(0));

    LOG_ALWAYS("[chain_N] chain_len=%d", chain_len);

    uint32_t scalar_shape[1] = {1};
    TensorCreateInfo ci(scalar_shape, 1, DataType::FLOAT32);

    if (chain_len == 1) {
        // Single task: seed -> result
        Arg params;
        params.add_input(seed);
        params.add_inout(result);
        pto2_rt_submit_aiv_task(FUNC_INC, params);
    } else {
        // First task: seed -> intermediate_0
        Arg first_params;
        first_params.add_input(seed);
        first_params.add_output(ci);
        TaskOutputTensors prev = pto2_rt_submit_aiv_task(FUNC_INC, first_params);

        // Middle tasks: intermediate_{i-1} -> intermediate_i
        for (int i = 1; i < chain_len - 1; i++) {
            Arg params;
            params.add_input(prev.get_ref(0));
            params.add_output(ci);
            prev = pto2_rt_submit_aiv_task(FUNC_INC, params);
        }

        // Last task: intermediate_{N-2} -> result
        Arg last_params;
        last_params.add_input(prev.get_ref(0));
        last_params.add_inout(result);
        pto2_rt_submit_aiv_task(FUNC_INC, last_params);
    }

    LOG_ALWAYS("[chain_N] Submitted %d chained tasks", chain_len);
}

}  // extern "C"
